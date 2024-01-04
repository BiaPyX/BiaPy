import torch
import numpy as np
import torch.nn as nn
from timm.layers.format import Format
from typing import Callable, List, Optional, Tuple

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    output_fmt: Format

    def __init__(
            self,
            img_size: Optional[int] = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            ndim: int = 2, 
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            output_fmt: Optional[str] = None,
            bias: bool = True,
            strict_img_size: bool = True,
    ):
        super().__init__()
        self.ndim = ndim
        self.patch_size = (patch_size,)*self.ndim
        if img_size is not None:
            self.img_size = (img_size,)*self.ndim
            self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
            self.num_patches = self.grid_size[0]**self.ndim
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = Format(output_fmt)
        else:
            # flatten spatial dim and transpose to channels last, kept for bwd compat
            self.flatten = flatten
            self.output_fmt = Format.NCHW
        self.strict_img_size = strict_img_size

        if self.ndim == 2:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        else:
            self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        
    def forward(self, x):
        if self.ndim == 2:
            B, C, H, W = x.shape
        else:
            B, C, Z, H, W = x.shape
        if self.img_size is not None:
            if self.strict_img_size:
                if self.ndim == 2:
                    assert H == self.img_size[0], f"Input height ({H}) doesn't match model ({self.img_size[0]})."
                    assert W == self.img_size[1], f"Input width ({W}) doesn't match model ({self.img_size[1]})."
                else:
                    assert Z == self.img_size[0], f"Input depth ({Z}) doesn't match model ({self.img_size[0]})."
                    assert H == self.img_size[1], f"Input height ({H}) doesn't match model ({self.img_size[1]})."
                    assert W == self.img_size[2], f"Input width ({W}) doesn't match model ({self.img_size[2]})."
            else:
                if self.ndim == 2:
                    assert Z % self.patch_size[0] == 0, f"Input height ({Z}) should be divisible by patch size ({self.patch_size[0]})."
                    assert H % self.patch_size[1] == 0, f"Input height ({H}) should be divisible by patch size ({self.patch_size[1]})."
                    assert W % self.patch_size[2] == 0, f"Input width ({W}) should be divisible by patch size ({self.patch_size[2]})."
                else:
                    assert H % self.patch_size[0] == 0, f"Input height ({H}) should be divisible by patch size ({self.patch_size[0]})."
                    assert W % self.patch_size[1] == 0, f"Input width ({W}) should be divisible by patch size ({self.patch_size[1]})."

        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        elif self.output_fmt != Format.NCHW:
            x = nchw_to(x, self.output_fmt)
        x = self.norm(x)
        return x