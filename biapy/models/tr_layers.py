import torch.nn as nn
from typing import Callable, Optional


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        ndim: int = 2,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten: bool = True,
        bias: bool = True,
        strict_img_size: bool = True,
    ):
        super().__init__()
        self.ndim = ndim
        self.patch_size = patch_size
        self.img_size = img_size
        self.grid_size = self.img_size // self.patch_size
        self.num_patches = self.grid_size**self.ndim
        self.flatten = flatten
        self.strict_img_size = strict_img_size

        if self.ndim == 2:
            self.proj = nn.Conv2d(
                in_chans,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
                bias=bias,
            )
        else:
            self.proj = nn.Conv3d(
                in_chans,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
                bias=bias,
            )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        if self.ndim == 2:
            B, C, H, W = x.shape
            Z = -1
        else:
            B, C, Z, H, W = x.shape
        if self.strict_img_size:
            assert (
                H == self.img_size
            ), f"Input height ({H}) doesn't match model ({self.img_size})."
            assert (
                W == self.img_size
            ), f"Input width ({W}) doesn't match model ({self.img_size})."

        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        x = self.norm(x)
        return x
