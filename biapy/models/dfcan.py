"""
This module implements the Deep Fourier Channel Attention Network (DFCAN) for super-resolution, along with its core building blocks.

The DFCAN model leverages Fourier domain processing and channel attention
mechanisms to enhance image details, particularly in microscopy images.

Key components include:

- `fftshift2d` and `fftshift3d`: Functions for shifting the zero-frequency
  component in 2D and 3D Fourier transforms.
- `RCAB` (Residual Channel Attention Block): A fundamental block incorporating
  Fourier-based channel attention.
- `ResGroup`: A sequential group of RCABs with a residual connection.
- `DFCAN`: The main super-resolution model integrating these components.

The implementation is adapted from the original DFCAN-pytorch repository.
"""
# Adapted from https://github.com/L0-zhang/DFCAN-pytorch

import torch
import torch.nn as nn
import torch.fft


def fftshift2d(img, size_psc=128):
    """
    Shifts the zero-frequency component of a 2D Fourier transform to the center of the spectrum.

    This function rearranges the quadrants of a 2D tensor (image) after a Fourier transform
    so that the zero-frequency component is at the center. This is a common operation
    in Fourier optics and signal processing.

    Parameters
    ----------
    img : torch.Tensor
        The input 2D tensor (image) in frequency domain, typically after `torch.fft.fftn`.
        Expected shape: (batch_size, channels, height, width).
    size_psc : int, optional
        Placeholder parameter, not directly used in the current implementation but
        might indicate expected patch size. Defaults to 128.

    Returns
    -------
    torch.Tensor
        The frequency-shifted 2D tensor with the same shape as the input.
    """
    bs, ch, h, w = img.shape
    fs11 = img[:, :, h // 2 :, w // 2 :]
    fs12 = img[:, :, h // 2 :, : w // 2]
    fs21 = img[:, :, : h // 2, w // 2 :]
    fs22 = img[:, :, : h // 2, : w // 2]
    output = torch.cat([torch.cat([fs11, fs21], dim=2), torch.cat([fs12, fs22], dim=2)], dim=3)
    return output


def fftshift3d(img, size_psc=128):
    """
    Shifts the zero-frequency component of a 3D Fourier transform to the center of the spectrum.

    This function rearranges the octants of a 3D tensor (volume) after a Fourier transform
    so that the zero-frequency component is at the center.

    Parameters
    ----------
    img : torch.Tensor
        The input 3D tensor (volume) in frequency domain, typically after `torch.fft.fftn`.
        Expected shape: (batch_size, channels, depth, height, width).
    size_psc : int, optional
        Placeholder parameter, not directly used in the current implementation but
        might indicate expected patch size. Defaults to 128.

    Returns
    -------
    torch.Tensor
        The frequency-shifted 3D tensor with the same shape as the input.
    """
    bs, ch, z, h, w = img.shape
    fs111 = img[:, :, z // 2 :, h // 2 :, w // 2 :]
    fs121 = img[:, :, z // 2 :, h // 2 :, : w // 2]
    fs211 = img[:, :, z // 2 :, : h // 2, w // 2 :]
    fs221 = img[:, :, z // 2 :, : h // 2, : w // 2]
    fs112 = img[:, :, : z // 2, h // 2 :, w // 2 :]
    fs122 = img[:, :, : z // 2, h // 2 :, : w // 2]
    fs212 = img[:, :, : z // 2, : h // 2, w // 2 :]
    fs222 = img[:, :, : z // 2, : h // 2, : w // 2]
    output1 = torch.cat([torch.cat([fs111, fs211], dim=3), torch.cat([fs121, fs221], dim=3)], dim=4)
    output2 = torch.cat([torch.cat([fs112, fs212], dim=3), torch.cat([fs122, fs222], dim=3)], dim=4)
    return torch.cat([output1, output2], dim=2)


class RCAB(nn.Module):
    """
    Residual Channel Attention Block (RCAB).

    This block enhances features by incorporating a Fourier-based channel attention
    mechanism within a residual structure. It operates by transforming features
    into the frequency domain, applying an attention mechanism, and then combining
    the result with the original features.
    """

    def __init__(self, size_psc=128, conv=nn.Conv2d):  # size_psc：crop_size input_shape：depth
        """
        Initialize the Residual Channel Attention Block.

        Sets up convolutional layers, activation functions (GELU, ReLU, Sigmoid),
        an adaptive pooling layer, and selects the appropriate FFT shift function
        based on the convolutional layer type (2D or 3D).

        Parameters
        ----------
        size_psc : int, optional
            The size of the patch (e.g., crop size) that the network is designed to process.
            Defaults to 128.
        conv : Type[nn.Conv2d | nn.Conv3d], optional
            The convolutional layer type to use within the block (e.g., `nn.Conv2d` or `nn.Conv3d`).
            Defaults to `nn.Conv2d`.
        """
        super().__init__()
        self.size_psc = size_psc
        self.conv_gelu1 = nn.Sequential(conv(64, 64, kernel_size=3, stride=1, padding="same"), nn.GELU())
        self.conv_gelu2 = nn.Sequential(conv(64, 64, kernel_size=3, stride=1, padding="same"), nn.GELU())
        self.conv_relu1 = nn.Sequential(conv(64, 64, kernel_size=3, stride=1, padding="same"), nn.ReLU())
        self.avg_pool = nn.AdaptiveAvgPool2d(1) if conv == nn.Conv2d else nn.AdaptiveAvgPool3d(1)
        self.conv_relu2 = nn.Sequential(conv(64, 4, kernel_size=1, stride=1, padding=0), nn.ReLU())
        self.conv_sigmoid = nn.Sequential(conv(4, 64, kernel_size=1, stride=1, padding=0), nn.Sigmoid())
        self.fftshiffunc = fftshift2d if conv == nn.Conv2d else fftshift3d

    def forward(self, x, gamma=0.8):
        """
        Perform the forward pass of the Residual Channel Attention Block.

        Processes the input `x` through initial convolutions, then transforms it
        to the frequency domain using FFT, applies a power law (gamma correction)
        and frequency shifting. An attention map is generated from this frequency
        domain representation and then used to scale the features. A residual
        connection adds the processed features back to the original input.

        Parameters
        ----------
        x : torch.Tensor
            The input feature tensor. Expected shape for 2D: (B, 64, H, W);
            for 3D: (B, 64, D, H, W).
        gamma : float, optional
            The power law exponent applied to the absolute values of the
            Fourier transformed features. Defaults to 0.8.

        Returns
        -------
        torch.Tensor
            The output feature tensor after applying the RCAB. Same shape as input `x`.
        """
        x0 = x.to(torch.float32)
        x = self.conv_gelu1(x)
        x = self.conv_gelu2(x)
        x1 = x.to(torch.float32)
        x = torch.fft.fftn(x.to(torch.float32), dim=(2, 3))
        x = torch.pow(torch.abs(x) + 1e-8, gamma)  # abs
        x = self.fftshiffunc(x, self.size_psc)
        x = self.conv_relu1(x)
        x = self.avg_pool(x)
        x = self.conv_relu2(x)
        x = self.conv_sigmoid(x)
        x = x1 * x
        x = x0 + x
        return x


class ResGroup(nn.Module):
    """
    A group of Residual Channel Attention Blocks (RCABs).

    This module stacks multiple RCABs and incorporates a residual connection
    around the entire group, allowing for deeper networks while maintaining
    gradient flow.
    """

    def __init__(self, n_RCAB=4, size_psc=128, conv=nn.Conv2d):
        """
        Initialize a Residual Group.

        Creates a sequential block of `n_RCAB` RCAB instances, each configured
        with the given `size_psc` and convolutional layer type.

        Parameters
        ----------
        n_RCAB : int, optional
            The number of `RCAB` blocks to include in this group. Defaults to 4.
        size_psc : int, optional
            The patch size parameter passed to each `RCAB`. Defaults to 128.
        conv : Type[nn.Conv2d | nn.Conv3d], optional
            The convolutional layer type passed to each `RCAB`. Defaults to `nn.Conv2d`.
        """
        super().__init__()
        RCABs = []
        for _ in range(n_RCAB):
            RCABs.append(RCAB(size_psc, conv=conv))
        self.RCABs = nn.Sequential(*RCABs)

    def forward(self, x):
        """
        Perform the forward pass of the Residual Group.

        Passes the input `x` through the sequence of `RCAB` blocks and then
        adds the original input `x` to the result, forming a residual connection.

        Parameters
        ----------
        x : torch.Tensor
            The input feature tensor.

        Returns
        -------
        torch.Tensor
            The output feature tensor after processing through the residual group.
            Same shape as input `x`.
        """
        x0 = x
        x = self.RCABs(x)
        x = x0 + x
        return x


class DFCAN(nn.Module):
    """
    Fourier Channel Attention Network (DFCAN) for Super-Resolution.

    DFCAN is a deep learning architecture designed for single image super-resolution,
    leveraging Fourier domain processing and channel attention mechanisms to
    enhance image details. It is composed of an initial feature extraction
    block, multiple Residual Groups (each containing Residual Channel Attention Blocks),
    a sub-pixel convolution for upsampling, and a final convolutional layer.

    References
    ----------
    - `Evaluation and development of deep neural networks for image super-resolution in optical
      microscopy <https://www.nature.com/articles/s41592-020-01048-5>`_.
    """

    def __init__(self, ndim, input_shape, scale=2, n_ResGroup=4, n_RCAB=4):
        """
        Initialize the DFCAN model for super-resolution.

        Sets up the network architecture including the initial convolution block,
        multiple Residual Groups (composed of RCABs), an upsampling layer
        (PixelShuffle), and the final output convolution.

        Parameters
        ----------
        ndim : int
            Number of spatial dimensions of the input data (2 for 2D, 3 for 3D).
        input_shape : Tuple[int, ...]
            Shape of the input tensor, typically (depth, height, width, channels) for 3D
            or (height, width, channels) for 2D, with channels as the last element.
            `input_shape[-1]` refers to the number of input channels.
            `input_shape[0]` (if 3D) or `input_shape[0]` (if 2D) refers to a spatial dimension
            used for `size_psc`.
        scale : int or tuple, optional
            The upsampling factor. If a single integer, it's used for both dimensions.
            If a tuple, it's specific for each dimension. Defaults to 2.
        n_ResGroup : int, optional
            The number of `ResGroup` blocks to use in the network. Defaults to 4.
        n_RCAB : int, optional
            The number of `RCAB` blocks within each `ResGroup`. Defaults to 4.
        """
        super().__init__()
        if type(scale) is tuple:
            scale = scale[0]
        self.ndim = ndim
        size_psc = input_shape[0]
        conv = nn.Conv3d if self.ndim == 3 else nn.Conv2d

        self.input = nn.Sequential(
            conv(input_shape[-1], 64, kernel_size=3, stride=1, padding="same"),
            nn.GELU(),
        )
        ResGroups = []
        for _ in range(n_ResGroup):
            ResGroups.append(ResGroup(n_RCAB=n_RCAB, size_psc=size_psc, conv=conv))
        self.RGs = nn.Sequential(*ResGroups)
        self.conv_gelu = nn.Sequential(
            conv(64, 64 * (scale**2), kernel_size=3, stride=1, padding="same"),
            nn.GELU(),
        )
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.conv_sigmoid = nn.Sequential(
            conv(64, input_shape[-1], kernel_size=3, stride=1, padding="same"),
            nn.Sigmoid(),
        )

    def forward(self, x) -> torch.Tensor:
        """
        Perform the forward pass of the DFCAN model.

        The input `x` first undergoes initial feature extraction. It then passes
        through a series of Residual Groups (RGs) for hierarchical feature refinement.
        After the RGs, a convolution prepares the features for upsampling via PixelShuffle.
        Finally, another convolution produces the super-resolved output, often
        with a sigmoid activation for intensity normalization.

        Parameters
        ----------
        x : torch.Tensor
            The input low-resolution image tensor.
            Expected shape for 2D: (batch_size, channels, height, width).
            Expected shape for 3D: (batch_size, channels, depth, height, width).

        Returns
        -------
        torch.Tensor
            The super-resolved output image tensor.
            The spatial dimensions will be scaled by the `scale` factor, and
            the number of channels will match `input_shape[-1]`.
        """
        x = self.input(x)
        x = self.RGs(x)
        x = self.conv_gelu(x)
        x = self.pixel_shuffle(x)  # upsampling
        x = self.conv_sigmoid(x)
        return x
