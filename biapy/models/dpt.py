"""
DPT (Dense Prediction Transformer) for BiaPy.

Reference: `Vision Transformers for Dense Prediction <https://arxiv.org/abs/2103.13413>`_ (Ranftl, Bochkovskiy and
Koltun, ICCV 2021), and the implementation in https://github.com/isl-org/DPT. It is also the decoder DINOv2 uses for
its dense (depth) evaluation.

Unlike UNETR, which builds a separate stack of transposed convolutions per skip connection, DPT:

1. **Reads** four ViT layers (evenly spaced along the encoder by default), folding the class token into the patch
   tokens ("ignore", "add" or "project").
2. **Reassembles** each of them into an image-like feature map and resamples it to a different scale, forming a
   pyramid at 4x, 2x, 1x and 0.5x the token-grid resolution (1/4, 1/8, 1/16 and 1/32 of the input for 16x16 tokens).
3. **Fuses** the pyramid from coarse to fine with residual convolutional units (RefineNet-style fusion blocks),
   upsampling by two at each step.
4. A small **head** brings the fused features to the input resolution.
"""

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from biapy.models.blocks import get_activation, get_norm_2d
from biapy.models.dense_vit import DenseViTBase


def _activation(name: str) -> nn.Module:
    """BiaPy activation, never in-place (the inputs are reused by the residual connections)."""
    act = get_activation(name)
    if hasattr(act, "inplace"):
        act.inplace = False
    return act


class ResidualConvUnit(nn.Module):
    """DPT's residual convolutional unit: ``x + conv(act(conv(act(x))))``."""

    def __init__(self, features: int, normalization: str, activation: str):
        super().__init__()
        bias = normalization == "none"
        self.act1 = _activation(activation)
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, padding=1, bias=bias)
        self.norm1 = get_norm_2d(normalization, features)
        self.act2 = _activation(activation)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1, bias=bias)
        self.norm2 = get_norm_2d(normalization, features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.norm1(self.conv1(self.act1(x)))
        out = self.norm2(self.conv2(self.act2(out)))
        return x + out


class FeatureFusionBlock(nn.Module):
    """
    DPT's fusion block: adds the (refined) skip features to the incoming ones, refines the sum and upsamples it.
    """

    def __init__(self, features: int, normalization: str, activation: str):
        super().__init__()
        self.res_conv_unit1 = ResidualConvUnit(features, normalization, activation)
        self.res_conv_unit2 = ResidualConvUnit(features, normalization, activation)
        self.out_conv = nn.Conv2d(features, features, kernel_size=1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None, size) -> torch.Tensor:
        if skip is not None:
            x = x + self.res_conv_unit1(skip)
        x = self.res_conv_unit2(x)
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=True)
        return self.out_conv(x)


class DPT(DenseViTBase):
    """DPT: ViT encoder + reassemble/fusion convolutional decoder (Ranftl et al., 2021)."""

    def __init__(
        self,
        input_shape,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        vit_model: str = "custom",
        token_stride: int = -1,
        drop_path_rate: float = 0.0,
        grad_checkpointing: bool = False,
        layers: List[int] = [],
        features: int = 256,
        reassemble_channels: List[int] = [],
        readout: str = "project",
        normalization: str = "none",
        decoder_activation: str = "relu",
        output_channels: List[int] = [1],
        output_channel_info: List[str] = ["F"],
        explicit_activations: bool = False,
        head_activations: List[str] = ["ce_sigmoid"],
        return_one_tensor: bool = False,
    ):
        """
        Initialize the DPT model.

        Parameters
        ----------
        input_shape : Tuple[int, ...]
            Input shape, ``(y, x, channels)``. 2D square inputs only.

        patch_size, embed_dim, depth, num_heads, mlp_ratio : int, int, int, int, float, optional
            ViT hyperparameters. Ignored unless ``vit_model`` is "custom".

        vit_model : str, optional
            "custom" or one of ``biapy.models.dense_vit.DENSE_VIT_MODELS`` (e.g. "celldino_vit").

        token_stride : int, optional
            Stride of the patch embedding. ``-1`` uses the token size (non-overlapping tokens, as in DPT).

        drop_path_rate : float, optional
            Stochastic depth of the last block, increasing linearly from 0.

        grad_checkpointing : bool, optional
            Recompute the transformer blocks in the backward pass to save memory.

        layers : List[int], optional
            The four ViT blocks (1-indexed, increasing) the decoder reads, from the finest to the coarsest level of
            the pyramid. Empty spaces them evenly, e.g. ``[6, 12, 18, 24]`` for a 24-block ViT (as DPT-Large).

        features : int, optional
            Channels of the fusion stage (256 in DPT).

        reassemble_channels : List[int], optional
            Channels of the four reassembled maps. Empty uses DPT's: ``[256, 512, 1024, 1024]`` for ViTs of 1024
            or more dimensions, ``[D/8, D/4, D/2, D]`` otherwise (``[96, 192, 384, 768]`` for ViT-B).

        readout : str, optional
            How the class token is folded into the patch tokens: "ignore", "add" or "project" (DPT's default,
            concatenating it to each token followed by a linear layer and a GELU).

        normalization : str, optional
            Normalization of the residual convolutional units (``'none'`` in DPT, ``'bn'``, ``'sync_bn'``, ``'in'``
            or ``'gn'``).

        decoder_activation : str, optional
            Activation of the decoder (ReLU in DPT).

        output_channels, output_channel_info, explicit_activations, head_activations, return_one_tensor : optional
            Output heads configuration, as in the rest of BiaPy's models.
        """
        super().__init__()
        self._build_encoder(
            input_shape,
            patch_size,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            vit_model,
            token_stride,
            drop_path_rate,
            grad_checkpointing,
        )
        D = self.embed_dim
        if len(layers) == 0:
            layers = [self.depth * (i + 1) // 4 for i in range(4)]
        if len(layers) != 4 or any(b >= a for a, b in zip(layers[1:], layers[:-1])) or not 1 <= layers[0] <= layers[-1] <= self.depth:
            raise ValueError(
                f"DPT needs 4 increasing ViT blocks in [1, {self.depth}] to read from. Provided: {layers}"
            )
        if len(reassemble_channels) == 0:
            reassemble_channels = [256, 512, 1024, 1024] if D >= 1024 else [D // 8, D // 4, D // 2, D]
        if len(reassemble_channels) != 4:
            raise ValueError(f"'reassemble_channels' needs 4 values. Provided: {reassemble_channels}")
        if readout not in ["ignore", "add", "project"]:
            raise ValueError(f"'readout' must be 'ignore', 'add' or 'project'. Provided: '{readout}'")
        self.layers = list(layers)
        self.readout = readout
        print(f"DPT reading the ViT blocks {self.layers} (readout '{readout}'), reassembled into {reassemble_channels} channels")

        # 1. Readout: fold the class token into the patch tokens
        if readout == "project":
            self.readout_projects = nn.ModuleList([nn.Sequential(nn.Linear(2 * D, D), nn.GELU()) for _ in range(4)])

        # 2. Reassemble: tokens -> maps at 4x, 2x, 1x and 0.5x the token grid resolution
        resample = [
            nn.ConvTranspose2d(reassemble_channels[0], reassemble_channels[0], kernel_size=4, stride=4),
            nn.ConvTranspose2d(reassemble_channels[1], reassemble_channels[1], kernel_size=2, stride=2),
            nn.Identity(),
            nn.Conv2d(reassemble_channels[3], reassemble_channels[3], kernel_size=3, stride=2, padding=1),
        ]
        self.reassemble = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(D, c, kernel_size=1), r) for c, r in zip(reassemble_channels, resample)]
        )
        # Project every level of the pyramid to the fusion channels
        self.layer_rn = nn.ModuleList(
            [nn.Conv2d(c, features, kernel_size=3, padding=1, bias=False) for c in reassemble_channels]
        )

        # 3. Fusion, from the coarsest level to the finest one
        self.fusion = nn.ModuleList(
            [FeatureFusionBlock(features, normalization, decoder_activation) for _ in range(4)]
        )

        # 4. Head (as DPT's depth head): to the input resolution
        self.head_conv1 = nn.Conv2d(features, features // 2, kernel_size=3, padding=1)
        self.head_conv2 = nn.Conv2d(features // 2, 32, kernel_size=3, padding=1)
        self.head_act = _activation(decoder_activation)

        self._build_heads(
            32,
            output_channels,
            output_channel_info,
            explicit_activations,
            head_activations,
            return_one_tensor,
        )
        self._init_encoder_weights()

    def _fold_readout(self, tokens: torch.Tensor, i: int) -> torch.Tensor:
        """Fold the class token of ``(B, 1 + N, D)`` tokens into the ``N`` patch tokens."""
        cls, patches = tokens[:, :1], tokens[:, 1:]
        if self.readout == "add":
            return patches + cls
        if self.readout == "project":
            return self.readout_projects[i](torch.cat([patches, cls.expand_as(patches)], dim=-1))
        return patches

    def forward(self, input: torch.Tensor) -> Dict | torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        input : torch.Tensor
            Input images, ``(B, C, H, W)``.

        Returns
        -------
        Dict or torch.Tensor
            Model output, as in the rest of BiaPy's models.
        """
        tokens = self.forward_tokens(input, self.layers)
        pyramid = []
        for i, t in enumerate(tokens):
            x = self.reassemble[i](self.tokens_to_map(self._fold_readout(t, i)))
            pyramid.append(self.layer_rn[i](x))

        # Coarse to fine: each fusion block upsamples to the size of the next (finer) level
        x = self.fusion[3](pyramid[3], None, pyramid[2].shape[-2:])
        x = self.fusion[2](x, pyramid[2], pyramid[1].shape[-2:])
        x = self.fusion[1](x, pyramid[1], pyramid[0].shape[-2:])
        x = self.fusion[0](x, pyramid[0], tuple(2 * s for s in pyramid[0].shape[-2:]))

        x = self.head_conv1(x)
        x = F.interpolate(x, size=input.shape[-2:], mode="bilinear", align_corners=True)
        x = self.head_act(self.head_conv2(x))
        return self._format_output(x)
