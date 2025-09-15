"""
This module provides the `PatchEmbed` class, a fundamental component used in Vision Transformers (ViT) to convert raw image data into sequences of flattened patches (tokens) suitable for transformer processing.

The `PatchEmbed` class handles the projection of image pixels into a higher-dimensional
embedding space and the subsequent flattening and optional normalization of these
patches. It supports both 2D and 3D image inputs.

Classes:

- ``PatchEmbed``: Transforms an input image into a sequence of embedded patches.

"""
import torch.nn as nn
from typing import Callable, Optional


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding module.

    This module converts an input image into a sequence of non-overlapping
    patches and then projects these patches into a higher-dimensional embedding
    space. Optionally, it applies normalization to the embeddings. It is a
    core component for Vision Transformers (ViT).
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
        """
        Initialize the PatchEmbed module.

        Sets up the convolutional layer for patch projection and an optional
        normalization layer. It calculates the number of patches and grid size
        based on the input `img_size` and `patch_size`.

        Parameters
        ----------
        img_size : int, optional
            The spatial size (height and width for 2D, or depth, height, and width for 3D,
            assuming square/cubic dimensions) of the input image. Defaults to 224.
        patch_size : int, optional
            The size of the square/cubic patch (token) that the image is divided into.
            Defaults to 16.
        in_chans : int, optional
            The number of input image channels (e.g., 3 for RGB, 1 for grayscale).
            Defaults to 3.
        ndim : int, optional
            The number of spatial dimensions of the input data (2 for 2D, 3 for 3D).
            Defaults to 2.
        embed_dim : int, optional
            The dimensionality of the output embedding for each patch. Defaults to 768.
        norm_layer : Optional[Callable], optional
            A normalization layer constructor (e.g., `nn.LayerNorm`). If provided,
            normalization is applied after patch projection. If `None`, no normalization.
            Defaults to None.
        flatten : bool, optional
            If True, the output feature maps from the convolutional projection are
            flattened into a sequence of tokens (`NLC` format). If False, the
            output retains its spatial dimensions (`NCHW` or `NCDHW`). Defaults to True.
        bias : bool, optional
            If True, adds a learnable bias to the convolutional layer. Defaults to True.
        strict_img_size : bool, optional
            If True, asserts that the input image's height and width (and depth for 3D)
            exactly match `img_size` during the forward pass. Defaults to True.
        """
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
        """
        Perform the forward pass of the PatchEmbed module.

        Projects the input image into patches, optionally flattens them into
        a sequence, and applies normalization. It also includes an assertion
        for strict image size matching if `strict_img_size` is True.

        Parameters
        ----------
        x : torch.Tensor
            The input image tensor.

            - For 2D: `(batch_size, channels, height, width)`
            - For 3D: `(batch_size, channels, depth, height, width)`

        Returns
        -------
        torch.Tensor
            The embedded patches.
            
            - If `flatten` is True: `(batch_size, num_patches, embed_dim)`
            - If `flatten` is False: `(batch_size, embed_dim, grid_size_D, grid_size_H, grid_size_W)`
              (spatial dimensions will be `img_size / patch_size`)

        Raises
        ------
        AssertionError
            If `strict_img_size` is True and the input image dimensions do not
            match the `img_size` specified during initialization.
        """
        if self.ndim == 2:
            B, C, H, W = x.shape
            Z = -1
        else:
            B, C, Z, H, W = x.shape
        if self.strict_img_size:
            assert H == self.img_size, f"Input height ({H}) doesn't match model ({self.img_size})."
            assert W == self.img_size, f"Input width ({W}) doesn't match model ({self.img_size})."

        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        x = self.norm(x)
        return x
