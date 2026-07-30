"""
This module implements a Vision Transformer (ViT) model, extending the `timm` library's `VisionTransformer` to support custom functionalities, particularly for different input dimensionalities (2D and 3D) and global pooling options.

The Vision Transformer processes images by dividing them into fixed-size patches,
linearly embedding each patch, and then processing the resulting sequence of
embeddings with a standard Transformer encoder. This module is often used as
a backbone for various computer vision tasks, including classification and
self-supervised learning (e.g., Masked Autoencoders).

Classes:

- ``VisionTransformer``: An extended ViT model with support for 2D/3D inputs and global pooling.

Functions:

- ``vit_base_patch16``: Factory function for a base-sized ViT model with 16x16 patches.
- ``vit_large_patch16``: Factory function for a large-sized ViT model with 16x16 patches.
- ``vit_huge_patch14``: Factory function for a huge-sized ViT model with 14x14 patches.
- ``sam3_vit``: Factory function for a ViT with SAM 3's image encoder as backbone.

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
from biapy.models.tr_layers import PatchEmbed
from biapy.models.sam3_vit import SAM3_VIT_PARAMS, build_sam3_blocks


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """
    Vision Transformer (ViT) model with extensions for 2D/3D input and global pooling.

    This class inherits from `timm.models.vision_transformer.VisionTransformer`
    and customizes it by replacing the default `patch_embed` with a custom
    implementation (`biapy.models.tr_layers.PatchEmbed`) that supports 2D and 3D inputs.
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
        **kwargs
            Keyword arguments to pass to the parent `timm.models.vision_transformer.VisionTransformer`
            constructor. These typically include `img_size`, `patch_size`, `in_chans`,
            `embed_dim`, `depth`, `num_heads`, `mlp_ratio`, `qkv_bias`, `norm_layer`.
        """
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
        embed_len = num_patches if self.no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, kwargs["embed_dim"]) * 0.02)

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
        x = x + self.pos_embed
        x = self.pos_drop(x)
        # Normalization applied before the blocks. It is an identity unless the model was created
        # with 'pre_norm', which is the case of the SAM 3 backbone (its 'ln_pre' layer).
        x = self.norm_pre(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
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

def vit_base_patch16(**kwargs):
    """
    Create a Vision Transformer (ViT) model with a Base-sized encoder and 16x16 patches.

    This function serves as a convenient constructor for a specific ViT configuration,
    often used as a standard baseline.

    Parameters
    ----------
    **kwargs
        Arbitrary keyword arguments to be passed to the `VisionTransformer`
        constructor. This allows overriding default parameters like `img_size`,
        `in_chans`, `ndim`, `global_pool`, etc.

    Returns
    -------
    model : VisionTransformer
        An initialized Base-sized ViT model.
    """
    # The patch size is defined by the model itself, so the one provided (e.g. 'MODEL.VIT_TOKEN_SIZE')
    # is discarded. Without this, passing it would raise a "multiple values for 'patch_size'" error.
    kwargs.pop("patch_size", None)
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs,
    )
    return model


def vit_large_patch16(**kwargs):
    """
    Create a Vision Transformer (ViT) model with a Large-sized encoder and 16x16 patches.

    This function provides a constructor for a larger ViT configuration,
    suitable for tasks requiring more capacity.

    Parameters
    ----------
    **kwargs
        Arbitrary keyword arguments to be passed to the `VisionTransformer`
        constructor. This allows overriding default parameters like `img_size`,
        `in_chans`, `ndim`, `global_pool`, etc.

    Returns
    -------
    model : VisionTransformer
        An initialized Large-sized ViT model.
    """
    # The patch size is defined by the model itself, so the one provided (e.g. 'MODEL.VIT_TOKEN_SIZE')
    # is discarded. Without this, passing it would raise a "multiple values for 'patch_size'" error.
    kwargs.pop("patch_size", None)
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs,
    )
    return model


def vit_huge_patch14(**kwargs):
    """
    Create a Vision Transformer (ViT) model with a Huge-sized encoder and 14x14 patches.

    This function provides a constructor for the largest ViT configuration,
    designed for tasks demanding maximum model capacity.

    Parameters
    ----------
    **kwargs
        Arbitrary keyword arguments to be passed to the `VisionTransformer`
        constructor. This allows overriding default parameters like `img_size`,
        `in_chans`, `ndim`, `global_pool`, etc.

    Returns
    -------
    model : VisionTransformer
        An initialized Huge-sized ViT model.
    """
    # The patch size is defined by the model itself, so the one provided (e.g. 'MODEL.VIT_TOKEN_SIZE')
    # is discarded. Without this, passing it would raise a "multiple values for 'patch_size'" error.
    kwargs.pop("patch_size", None)
    model = VisionTransformer(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs,
    )
    return model


def sam3_vit(**kwargs):
    """
    Create a Vision Transformer (ViT) with the image encoder of SAM 3.

    The encoder is built exactly as SAM 3's one (1024 embedding dimensions, 32 blocks, 16 heads,
    a 4.625 MLP ratio, 2D rotary position embeddings and window attention in all the blocks but
    the global ones), so its pretrained weights can be loaded into it afterwards with
    `biapy.models.sam3_vit.load_sam3_pretrained_encoder`. Every value defining the architecture is
    the one of SAM 3, its ``14x14`` token size included, so the size of the input images needs to
    be a multiple of it.

    Unlike SAM 3's trunk, the model keeps a class token, as it is what BiaPy's ViT uses to
    classify. It is randomly initialized, as there is none in SAM 3's checkpoint, and it is not
    rotated within the attention, as it has no position in the token grid.

    Parameters
    ----------
    **kwargs
        Arbitrary keyword arguments to be passed to the `VisionTransformer` constructor, such as
        `img_size`, `in_chans` or `num_classes`. Every ViT parameter defining the architecture of
        SAM 3's encoder, the token size included, is ignored, as it is set by the encoder itself.

    Returns
    -------
    model : VisionTransformer
        An initialized ViT with SAM 3's image encoder as backbone.
    """
    params = SAM3_VIT_PARAMS
    if kwargs.get("ndim", 2) != 2:
        raise ValueError(
            "SAM 3's image encoder ('sam3_vit') can only be used with 2D data, as its pretrained weights "
            "are 2D (its patch embedding projects 3-channel 2D images). Set 'MODEL.VIT_MODEL' to another "
            "value to work with 3D data."
        )
    # Everything defining the architecture is set by SAM 3's encoder, so the values provided
    # (e.g. 'MODEL.VIT_TOKEN_SIZE') are discarded
    for k in ["patch_size", "embed_dim", "depth", "num_heads", "mlp_ratio", "qkv_bias", "norm_layer", "pre_norm"]:
        kwargs.pop(k, None)

    model = VisionTransformer(
        patch_size=params["patch_size"],
        embed_dim=params["embed_dim"],
        # The blocks are replaced below by SAM 3's ones, so only one is created here to avoid
        # allocating (and immediately discarding) the parameters of the 32 blocks of the encoder
        depth=1,
        num_heads=params["num_heads"],
        mlp_ratio=params["mlp_ratio"],
        qkv_bias=params["qkv_bias"],
        norm_layer=partial(nn.LayerNorm, eps=params["norm_eps"]),
        pre_norm=True,  # SAM 3's 'ln_pre'
        **kwargs,
    )
    grid_size = model.patch_embed.grid_size
    model.blocks = build_sam3_blocks((grid_size, grid_size), num_prefix_tokens=model.num_prefix_tokens)
    print(
        f"SAM 3 image encoder built with {params['depth']} blocks over a {grid_size}x{grid_size} token grid "
        f"({params['patch_size']}x{params['patch_size']} tokens)"
    )
    return model
