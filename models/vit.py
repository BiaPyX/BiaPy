### Adapted from https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L380

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import timm.models.vision_transformer
from typing import Union, Tuple

from models.tr_layers import PatchEmbed

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """
    Mask autoenconder (MAE) with VisionTransformer (ViT) backbone. 
    
    Reference: `Masked Autoencoders Are Scalable Vision Learners <https://arxiv.org/abs/2111.06377>`_.
    
    Parameters
    ----------
    ndim : int, optional
        Number of input dimensions.

    global_pool : bool, optional
        Whether to use global pooling or not. 

    Returns
    -------
    model : Torch model
        ViT model.
    """
    def __init__(self, ndim=2, global_pool = False, **kwargs):
        super(VisionTransformer, self).__init__( **kwargs)
        self.ndim = ndim
        self.global_pool = global_pool

        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm
        
        # Replace with our PatchEmbed implementation and re-define all dependant variables 
        self.patch_embed = PatchEmbed(
            img_size=kwargs['img_size'],
            patch_size=kwargs['patch_size'],
            in_chans=kwargs['in_chans'],
            ndim=self.ndim,
            embed_dim=kwargs['embed_dim'],
            bias=True,  
        )
        num_patches = self.patch_embed.num_patches
        embed_len = num_patches if self.no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, kwargs['embed_dim']) * .02)

    def forward_features(self, x):
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
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
