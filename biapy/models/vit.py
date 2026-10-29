"""
This module implements a Vision Transformer (ViT) model, extending the `timm` library's `VisionTransformer` to support custom functionalities, particularly for different input dimensionalities (2D and 3D) and global pooling options.

The Vision Transformer processes images by dividing them into fixed-size patches,
linearly embedding each patch, and then processing the resulting sequence of
embeddings with a standard Transformer encoder. This module is often used as
a backbone for various computer vision tasks, including classification and
self-supervised learning (e.g., Masked Autoencoders).

Classes:

- ``VisionTransformer``: An extended ViT model with support for 2D/3D inputs, global pooling and the predefined
  backbones of ``biapy.models.vit_backbones.common.VIT_BACKBONES`` ('MODEL.VIT_MODEL').

References:

- timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
- DeiT: https://github.com/facebookresearch/deit
- Masked Autoencoders Are Scalable Vision Learners: https://arxiv.org/abs/2111.06377
"""

from functools import partial
from typing import List

import torch
import torch.nn as nn
import timm.models.vision_transformer

from biapy.models.blocks import prepare_activation_layers
from biapy.models.vit_backbones.common import VIT_BACKBONES, PatchEmbed, build_vit_blocks, check_vit_backbone
from biapy.models.vit_backbones.dinov3 import DINOV3_VIT_PARAMS


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """
    Vision Transformer (ViT) model with extensions for 2D/3D input and global pooling.

    This class inherits from `timm.models.vision_transformer.VisionTransformer`
    and customizes it by replacing the default `patch_embed` with a custom
    implementation (`biapy.models.vit_backbones.common.PatchEmbed`) that supports 2D and 3D inputs.
    It also adds an option for global average pooling of patch tokens for classification.

    Reference: `Masked Autoencoders Are Scalable Vision Learners <https://arxiv.org/abs/2111.06377>`_.

    Parameters
    ----------
    ndim : int, optional
        Number of input dimensions (2 for 2D images, 3 for 3D images). Defaults to 2.

    global_pool : bool, optional
        If True, applies global average pooling to the patch tokens (excluding the
        class token) before the final classification head. If False, uses the
        class token's output for classification (standard ViT behavior). Defaults to False.

    **kwargs
        Arbitrary keyword arguments passed to the base `timm.models.vision_transformer.VisionTransformer`
        constructor, such as `img_size`, `patch_size`, `in_chans`, `embed_dim`, `depth`,
        `num_heads`, `mlp_ratio`, `qkv_bias`, `norm_layer`, etc.
    """

    def __init__(
        self, 
        ndim: int = 2, 
        global_pool: bool = False, 
        head_activations: List[str] = ["ce_sigmoid"],         
        output_channel_info: List[str] = ["F"],
        explicit_activations: bool = False,
        vit_model: str = "custom",
        **kwargs
    ):
        """
        Initialize the VisionTransformer model.

        Calls the base `timm.models.vision_transformer.VisionTransformer` constructor,
        then customizes the `patch_embed` layer and handles the global pooling
        configuration, potentially removing the original normalization layer if
        global pooling is enabled.

        Parameters
        ----------
        ndim : int, optional
            Number of input dimensions (2 for 2D images, 3 for 3D images). Defaults to 2.
        global_pool : bool, optional
            If True, enables global average pooling of patch tokens. Defaults to False.
        explicit_activations : bool, optional
            If True, enables explicit activation functions. Defaults to False.
        vit_model : str, optional
            "custom" (built from ``kwargs``) or one of ``VIT_BACKBONES``
            (``biapy.models.vit_backbones.common``), which overrides the geometry in ``kwargs``.
        **kwargs
            Keyword arguments to pass to the parent `timm.models.vision_transformer.VisionTransformer`
            constructor. These typically include `img_size`, `patch_size`, `in_chans`,
            `embed_dim`, `depth`, `num_heads`, `mlp_ratio`, `qkv_bias`, `norm_layer`.
        """
        blocks_kind = "timm"
        if vit_model != "custom":
            check_vit_backbone(vit_model, "vit", ndim, kwargs.get("in_chans", 3))
            preset = VIT_BACKBONES[vit_model]
            blocks_kind = preset["blocks"]
            for k in ["patch_size", "embed_dim", "depth", "num_heads", "mlp_ratio", "qkv_bias", "norm_layer",
                      "init_values", "pre_norm"]:
                kwargs.pop(k, None)
            kwargs.update(
                patch_size=preset["patch_size"],
                embed_dim=preset["embed_dim"],
                # SAM 3/DINOv3 blocks are replaced below: build a single placeholder
                depth=1 if blocks_kind in ["sam3", "dinov3"] else preset["depth"],
                num_heads=preset["num_heads"],
                mlp_ratio=preset["mlp_ratio"],
                qkv_bias=True,
            )
            if "norm_eps" in preset:
                kwargs["norm_layer"] = partial(nn.LayerNorm, eps=preset["norm_eps"])
            if blocks_kind == "timm" and "init_values" in preset:
                kwargs["init_values"] = preset["init_values"]
            if blocks_kind == "sam3":
                kwargs["pre_norm"] = True  # SAM 3's 'ln_pre'
            if blocks_kind == "dinov3":
                # RoPE instead of a position embedding, plus register tokens
                kwargs["reg_tokens"] = DINOV3_VIT_PARAMS["num_register_tokens"]
                kwargs["pos_embed"] = "none"
            print(f"Building the ViT backbone as '{vit_model}': {preset}")

        super(VisionTransformer, self).__init__(**kwargs)
        self.ndim = ndim
        self.global_pool = global_pool
        self.explicit_activations = explicit_activations
        if self.explicit_activations:
            self.class_head_activations, _ = prepare_activation_layers(head_activations, output_channel_info, [self.num_classes])
        if self.global_pool:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
            embed_dim = kwargs["embed_dim"]
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

        # Replace with our PatchEmbed implementation and re-define all dependant variables
        self.patch_embed = PatchEmbed(
            img_size=kwargs["img_size"],
            patch_size=kwargs["patch_size"],
            in_chans=kwargs["in_chans"],
            ndim=self.ndim,
            embed_dim=kwargs["embed_dim"],
            bias=True,
        )
        num_patches = self.patch_embed.num_patches
        # Models built without position embedding (e.g. DINOv3's, which uses RoPE within its blocks) keep it that way
        if self.pos_embed is not None:
            embed_len = num_patches if self.no_embed_class else num_patches + self.num_prefix_tokens
            self.pos_embed = nn.Parameter(torch.randn(1, embed_len, kwargs["embed_dim"]) * 0.02)

        if blocks_kind in ["sam3", "dinov3"]:
            if self.reg_token is not None:
                nn.init.normal_(self.reg_token, std=0.02)
            grid_size = self.patch_embed.grid_size
            self.blocks = build_vit_blocks(
                blocks_kind,
                preset["embed_dim"],
                preset["depth"],
                preset["num_heads"],
                preset["mlp_ratio"],
                (grid_size, grid_size),
                self.num_prefix_tokens,
            )

    def forward_features(self, x):
        """
        Perform the forward pass through the Vision Transformer's encoder.

        This method processes the input image, converts it into patch embeddings,
        adds positional embeddings and the class token, and then passes the
        sequence through the transformer blocks. Finally, it applies either
        global pooling or extracts the class token's output based on `self.global_pool`.

        Parameters
        ----------
        x : torch.Tensor
            The input image tensor.
            Expected shape for 2D: `(batch_size, channels, height, width)`.
            Expected shape for 3D: `(batch_size, channels, depth, height, width)`.

        Returns
        -------
        torch.Tensor
            The output feature representation from the ViT encoder.
            - If `global_pool` is True: `(batch_size, embed_dim)` (pooled patch tokens).
            - If `global_pool` is False: `(batch_size, embed_dim)` (class token output).
        """
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        # Register tokens (e.g. DINOv3's) go between the class token and the patch tokens
        if self.reg_token is not None:
            x = torch.cat((x[:, :1], self.reg_token.expand(B, -1, -1), x[:, 1:]), dim=1)
        x = self.pos_drop(x)
        # Normalization applied before the blocks. It is an identity unless the model was created
        # with 'pre_norm', which is the case of the SAM 3 backbone (its 'ln_pre' layer).
        x = self.norm_pre(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, self.num_prefix_tokens :, :].mean(dim=1)  # global pool without cls (and register) tokens
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def forward(self, x):
        outs = super(VisionTransformer, self).forward(x)
        # Apply activations to the output heads if explicit_activations is True
        if self.explicit_activations:
            outs = self.class_head_activations[0](outs)
        
        return outs
