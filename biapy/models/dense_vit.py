"""
Plain Vision Transformer (ViT) encoder shared by BiaPy's dense-prediction transformer architectures.

It is the base of ``DPT`` (``biapy.models.dpt``) and ``ViTReadout`` (``biapy.models.vit_readout``), which only
differ in how they turn the ViT tokens back into a full-resolution prediction. Compared with the ViT used in UNETR
it adds:

- Overlapping tokens: the patch embedding can move with a stride smaller than its kernel (padding the image so
  the token grid is ``image_size / stride``). That is how Cellpose's ``CPDINO`` model doubles the token grid of a
  pretrained 16x16 ViT without touching its patch embedding weights.
- Stochastic depth (``drop_path_rate``), increasing linearly along the blocks, as Cellpose-SAM's layer dropout.
- A final layer normalization, applied to every token map returned (as DINOv2's ``get_intermediate_layers``).
- Optional gradient checkpointing, to fine-tune large ViTs with many tokens.

The attribute names (``patch_embed.proj``, ``cls_token``, ``pos_embed``, ``blocks``, ``norm``) follow timm/DINOv2,
so pretrained encoders (e.g. Cell-DINO through ``biapy.models.celldino_vit.load_celldino_pretrained_encoder``) load
directly into them.
"""

from functools import partial
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from timm.models.vision_transformer import Block
from timm.layers import trunc_normal_

from biapy.models.blocks import prepare_activation_layers
from biapy.models.celldino_vit import CELLDINO_VIT_PARAMS

# Predefined ViT backbones. "custom" uses the values passed to the model; the rest are fully defined by the preset.
# Optional "init_values" enables LayerScale and "norm_eps" sets the epsilon of the LayerNorms (1e-6 otherwise).
DENSE_VIT_MODELS = {
    "vit_base_patch16": dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0),
    "vit_large_patch16": dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4.0),
    "vit_huge_patch14": dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4.0),
    "celldino_vit": dict(
        patch_size=CELLDINO_VIT_PARAMS["patch_size"],
        embed_dim=CELLDINO_VIT_PARAMS["embed_dim"],
        depth=CELLDINO_VIT_PARAMS["depth"],
        num_heads=CELLDINO_VIT_PARAMS["num_heads"],
        mlp_ratio=CELLDINO_VIT_PARAMS["mlp_ratio"],
        init_values=CELLDINO_VIT_PARAMS["init_values"],
        norm_eps=CELLDINO_VIT_PARAMS["norm_eps"],
    ),
}


class OverlapPatchEmbed(nn.Module):
    """
    2D patch embedding whose stride can be smaller than its kernel (overlapping tokens).

    The image is padded by ``(patch_size - stride) / 2`` on each side, so the token grid is ``img_size / stride``.
    With ``stride == patch_size`` it is the standard, non-overlapping, ViT patch embedding.
    """

    def __init__(self, img_size: int, patch_size: int, stride: int, in_chans: int, embed_dim: int):
        super().__init__()
        if stride < 1 or stride > patch_size or (patch_size - stride) % 2 != 0:
            raise ValueError(
                f"The token stride ({stride}) must be between 1 and the token size ({patch_size}), and their "
                "difference must be even so the image can be padded symmetrically"
            )
        if img_size % stride != 0:
            raise ValueError(f"The input size ({img_size}) must be divisible by the token stride ({stride})")
        self.patch_size = patch_size
        self.stride = stride
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=(patch_size - stride) // 2
        )
        self.grid_size = img_size // stride
        self.num_patches = self.grid_size**2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed ``(B, C, H, W)`` images into ``(B, H/stride * W/stride, embed_dim)`` tokens."""
        return self.proj(x).flatten(2).transpose(1, 2)


class DenseViTBase(nn.Module):
    """
    Base class of the dense-prediction ViT architectures: builds the ViT encoder and the output heads.

    Subclasses build their decoder, returning ``num_features`` channels at the input resolution, and call
    ``self._build_heads(num_features)`` and ``self._format_output(features)``.
    """

    def _build_encoder(
        self,
        input_shape: Tuple[int, ...],
        patch_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        vit_model: str,
        token_stride: int,
        drop_path_rate: float,
        grad_checkpointing: bool,
    ):
        """
        Build the ViT encoder.

        Parameters
        ----------
        input_shape : Tuple[int, ...]
            Input shape, ``(y, x, channels)``. Only 2D square inputs are supported.

        patch_size, embed_dim, depth, num_heads, mlp_ratio : int, int, int, int, float
            ViT hyperparameters. Ignored unless ``vit_model`` is "custom".

        vit_model : str
            "custom" or one of ``DENSE_VIT_MODELS``.

        token_stride : int
            Stride of the patch embedding. ``-1`` (or the token size) gives non-overlapping tokens.

        drop_path_rate : float
            Stochastic depth rate of the last block; it increases linearly from 0 in the first one.

        grad_checkpointing : bool
            Whether to recompute the transformer blocks in the backward pass instead of storing their activations.
        """
        if len(input_shape) != 3:
            raise ValueError(f"{type(self).__name__} only supports 2D data, i.e. 'input_shape' as (y, x, channels)")
        if input_shape[0] != input_shape[1]:
            raise ValueError(f"{type(self).__name__} needs square inputs. Provided: {input_shape}")

        init_values = None
        norm_eps = 1e-6
        if vit_model != "custom":
            if vit_model not in DENSE_VIT_MODELS:
                raise ValueError(
                    f"'vit_model' needs to be 'custom' or one of {list(DENSE_VIT_MODELS)}. Provided: '{vit_model}'"
                )
            if vit_model == "celldino_vit" and input_shape[-1] != CELLDINO_VIT_PARAMS["in_chans"]:
                raise ValueError(
                    f"'celldino_vit' needs {CELLDINO_VIT_PARAMS['in_chans']} input channel, the input has "
                    f"{input_shape[-1]}."
                )
            vit_params = DENSE_VIT_MODELS[vit_model]
            patch_size = vit_params["patch_size"]
            embed_dim = vit_params["embed_dim"]
            depth = vit_params["depth"]
            num_heads = vit_params["num_heads"]
            mlp_ratio = vit_params["mlp_ratio"]
            init_values = vit_params.get("init_values")
            norm_eps = vit_params.get("norm_eps", norm_eps)
            print(f"Building {type(self).__name__}'s ViT backbone as '{vit_model}': {vit_params}")
        token_stride = patch_size if token_stride <= 0 else token_stride

        self.input_shape = input_shape
        self.embed_dim = embed_dim
        self.depth = depth
        self.token_stride = token_stride
        self.grad_checkpointing = grad_checkpointing
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)

        self.patch_embed = OverlapPatchEmbed(
            img_size=input_shape[0],
            patch_size=patch_size,
            stride=token_stride,
            in_chans=input_shape[-1],
            embed_dim=embed_dim,
        )
        self.grid_size = self.patch_embed.grid_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        drop_path = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    init_values=init_values,
                    drop_path=drop_path[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        print(
            f"{type(self).__name__}'s ViT: {depth} blocks, {patch_size}x{patch_size} tokens with stride "
            f"{token_stride} ({self.grid_size}x{self.grid_size} token grid), stochastic depth up to {drop_path_rate}"
        )

    def _init_encoder_weights(self):
        """Initialize the encoder as timm's ViT does (pretrained weights, if any, are loaded afterwards)."""
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)

        def _init(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        self.blocks.apply(_init)
        self.norm.apply(_init)

    def forward_tokens(self, x: torch.Tensor, layers: List[int]) -> List[torch.Tensor]:
        """
        Run the ViT encoder and return the normalized tokens of the selected blocks.

        Parameters
        ----------
        x : torch.Tensor
            Input images, ``(B, C, H, W)``.

        layers : List[int]
            Blocks (1-indexed) whose output is returned.

        Returns
        -------
        List[torch.Tensor]
            One ``(B, 1 + N, embed_dim)`` tensor per selected block, class token first, after the final norm.
        """
        B = x.shape[0]
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1) + self.pos_embed
        outs = []
        for i, blk in enumerate(self.blocks):
            if self.grad_checkpointing and self.training and torch.is_grad_enabled():
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
            if i + 1 in layers:
                outs.append(self.norm(x))
        return outs

    def tokens_to_map(self, tokens: torch.Tensor) -> torch.Tensor:
        """Reshape ``(B, N, D)`` patch tokens (no class token) into a ``(B, D, h, w)`` feature map."""
        B, N, D = tokens.shape
        return tokens.transpose(1, 2).reshape(B, D, self.grid_size, self.grid_size)

    def _build_heads(
        self,
        in_channels: int,
        output_channels: List[int],
        output_channel_info: List[str],
        explicit_activations: bool,
        head_activations: List[str],
        return_one_tensor: bool,
    ):
        """Build one 1x1 convolution per output head (as in UNETR)."""
        if len(output_channels) == 0:
            raise ValueError("'output_channels' needs to has at least one value")
        print("Selected output channels:")
        for i, info in enumerate(output_channel_info):
            print(f"  - {i} channel for {info} output")
        self.output_channels = output_channels
        self.output_channel_info = output_channel_info
        self.return_class = "class" in output_channel_info
        self.explicit_activations = explicit_activations
        self.return_one_tensor = return_one_tensor
        if self.explicit_activations:
            assert len(head_activations) == sum(output_channels), (
                "If 'explicit_activations' is True, 'head_activations' needs to have the same number of values as "
                "'output_channels'"
            )
            self.head_activations, self.class_head_activations = prepare_activation_layers(
                head_activations, output_channel_info, output_channels
            )
            if self.return_class and self.class_head_activations is None:
                raise ValueError("If 'return_class' is True, 'head_activations' must be provided.")
        self.heads = nn.Sequential()
        for out_ch in output_channels:
            self.heads.append(nn.Conv2d(in_channels, out_ch, kernel_size=1, padding="same"))

    def _format_output(self, feats: torch.Tensor) -> Dict | torch.Tensor:
        """Apply the output heads to the decoder features and pack the outputs as the rest of BiaPy's models."""
        class_outs, outs = [], []
        for i, head in enumerate(self.heads):
            if "class" not in self.output_channel_info[i]:
                outs.append(head(feats))
            else:
                class_outs.append(head(feats))
        outs = torch.cat(outs, dim=1)

        if self.explicit_activations:
            if len(self.head_activations) == 1:
                outs = self.head_activations[0](outs)
            else:
                for i, act in enumerate(self.head_activations):
                    outs[:, i : i + 1] = act(outs[:, i : i + 1])
            if self.return_class and self.class_head_activations is not None:
                for i, act in enumerate(self.class_head_activations):
                    class_outs[i] = act(class_outs[i])

        if not self.return_class:
            return outs
        if self.return_one_tensor:
            return torch.cat((outs, torch.argmax(torch.cat(class_outs, dim=1), dim=1).unsqueeze(1)), dim=1)
        return {"pred": outs, "class": torch.cat(class_outs, dim=1)}
