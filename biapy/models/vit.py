"""
This module implements a Vision Transformer (ViT) model, extending the `timm` library's `VisionTransformer` to support custom functionalities, particularly for different input dimensionalities (2D and 3D) and global pooling options.

The Vision Transformer processes images by dividing them into fixed-size patches,
linearly embedding each patch, and then processing the resulting sequence of
embeddings with a standard Transformer encoder. This module is often used as
a backbone for various computer vision tasks, including classification and
self-supervised learning (e.g., Masked Autoencoders).

Classes:
--------
- VisionTransformer: An extended ViT model with support for 2D/3D inputs and global pooling.

Functions:
----------
- vit_base_patch16: Factory function for a base-sized ViT model with 16x16 patches.
- vit_large_patch16: Factory function for a large-sized ViT model with 16x16 patches.
- vit_huge_patch14: Factory function for a huge-sized ViT model with 14x14 patches.

References:
-----------
- timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
- DeiT: https://github.com/facebookresearch/deit
- Masked Autoencoders Are Scalable Vision Learners: https://arxiv.org/abs/2111.06377
"""

from functools import partial

import torch
import torch.nn as nn
import timm.models.vision_transformer

from biapy.models.tr_layers import PatchEmbed


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

    def __init__(self, ndim=2, global_pool=False, **kwargs):
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
        **kwargs
            Keyword arguments to pass to the parent `timm.models.vision_transformer.VisionTransformer`
            constructor. These typically include `img_size`, `patch_size`, `in_chans`,
            `embed_dim`, `depth`, `num_heads`, `mlp_ratio`, `qkv_bias`, `norm_layer`.
        """
        super(VisionTransformer, self).__init__(**kwargs)
        self.ndim = ndim
        self.global_pool = global_pool

        if self.global_pool:
            norm_layer = kwargs["norm_layer"]
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

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


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
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
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
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
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
    model = VisionTransformer(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model
