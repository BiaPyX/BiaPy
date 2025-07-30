"""
This module implements the Enhanced Deep Residual Networks for Single Image Super-Resolution (EDSR) model.

It includes the main EDSR model architecture and its essential building blocks:

- `SR_convblock`: A residual convolutional block used within the EDSR network.
- `SR_upsampling`: An upsampling block employing sub-pixel convolution (PixelShuffle)
  for efficient image scaling.

The code is adapted from the Keras EDSR example, providing a PyTorch implementation
for both 2D and 3D image super-resolution tasks.
"""
import torch
import torch.nn as nn


class EDSR(nn.Module):
    """
    Enhanced Deep Residual Networks for Single Image Super-Resolution (EDSR) model.

    Reference: `Enhanced Deep Residual Networks for Single Image Super-Resolution <https://arxiv.org/abs/1707.02921>`_.

    Code adapted from https://keras.io/examples/vision/edsr
    """

    def __init__(
        self,
        ndim=2,
        num_filters=64,
        num_of_residual_blocks=16,
        upsampling_factor=2,
        num_channels=3,
    ):
        """
        Initialize the EDSR model.

        Sets up the network's components, including the entry-point convolutional layer,
        a stack of residual blocks, a convolutional layer for the global skip connection,
        and the final upsampling and output layers.

        Parameters
        ----------
        ndim : int, optional
            Number of spatial dimensions of the input data (2 for 2D, 3 for 3D).
            Defaults to 2.
        num_filters : int, optional
            The number of filters (channels) used throughout the main body of the network
            within residual blocks. Defaults to 64.
        num_of_residual_blocks : int, optional
            The total number of `SR_convblock` residual blocks to stack. Defaults to 16.
        upsampling_factor : int or tuple, optional
            The overall upsampling factor for the image. If a tuple, only the first element
            is used as `PixelShuffle` expects a single integer factor. Defaults to 2.
        num_channels : int, optional
            The number of input and output channels of the image (e.g., 3 for RGB, 1 for grayscale).
            Defaults to 3.
        """
        super(EDSR, self).__init__()
        if type(upsampling_factor) is tuple:
            upsampling_factor = upsampling_factor[0]
        self.ndim = ndim

        if self.ndim == 3:
            conv = nn.Conv3d
        else:
            conv = nn.Conv2d

        self.first_conv_of_block = conv(num_channels, num_filters, kernel_size=3, padding="same")

        self.resblock = nn.Sequential()
        # 16 residual blocks
        for i in range(num_of_residual_blocks):
            self.resblock.append(SR_convblock(conv, num_filters))

        self.last_conv_of_block = conv(num_filters, num_filters, kernel_size=3, padding="same")
        self.last_block = nn.Sequential(
            SR_upsampling(conv, num_filters, upsampling_factor),
            conv(num_filters, num_channels, kernel_size=3, padding="same"),
        )

    def forward(self, x):
        """
        Perform the forward pass of the EDSR model.

        The input `x` first goes through an initial convolution. The output of this
        convolution is then passed through a series of residual blocks. A global
        residual connection adds the initial convolutional output (after another
        convolution) to the output of the residual blocks. Finally, the combined
        features are upsampled and mapped to the desired number of output channels.

        Parameters
        ----------
        x : torch.Tensor
            The input low-resolution image tensor.
            Expected shape for 2D: `(batch_size, num_channels, height, width)`.
            Expected shape for 3D: `(batch_size, num_channels, depth, height, width)`.

        Returns
        -------
        torch.Tensor
            The super-resolved output image tensor.
            The spatial dimensions are upscaled by `upsampling_factor`, and the
            number of channels matches `num_channels`.
        """
        # Initial feature extraction
        out = x_new = self.first_conv_of_block(x)

        # Pass through residual blocks
        out = self.resblock(out)

        # Apply global skip connection
        x_new = self.last_conv_of_block(x_new)
        out = out + x_new

        # Final upsampling and output
        out = self.last_block(out)
        return out


class SR_convblock(nn.Module):
    """
    A basic residual convolutional block for Super-Resolution networks.

    This block consists of two convolutional layers with identity skip connection,
    designed to learn residual features effectively. It avoids batch normalization
    as per the original EDSR architecture.
    """

    def __init__(self, conv, num_filters):
        """
        Initialize the Super-Resolution convolutional block.

        Sets up two convolutional layers, both maintaining the number of filters
        and using a 3x3 (or 3x3x3 for 3D) kernel with 'same' padding.

        Parameters
        ----------
        conv : Type[nn.Conv2d | nn.Conv3d]
            The convolutional layer type to use (e.g., `nn.Conv2d` or `nn.Conv3d`).
        num_filters : int
            The number of input and output channels for both convolutional layers within the block.
        """
        super(SR_convblock, self).__init__()
        self.conv1 = conv(num_filters, num_filters, kernel_size=3, padding="same")
        self.conv2 = conv(num_filters, num_filters, kernel_size=3, padding="same")

    def forward(self, x):
        """
        Perform the forward pass of the Super-Resolution convolutional block.

        The input `x` passes sequentially through two convolutional layers.
        The output of the second convolution is then added back to the original
        input `x` via an identity skip connection.

        Parameters
        ----------
        x : torch.Tensor
            The input feature tensor. Expected shape: `(batch_size, num_filters, ...spatial_dims)`.

        Returns
        -------
        torch.Tensor
            The output feature tensor after processing through the residual block.
            Same shape as input `x`.
        """
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + x
        return out


class SR_upsampling(nn.Module):
    """
    Super-resolution upsampling block using PixelShuffle.

    This block handles the upscaling operation using sub-pixel convolutions,
    which is an efficient way to increase spatial resolution while reducing
    checkerboard artifacts often seen with transposed convolutions. It can
    perform 2x or 4x upsampling.
    """

    def __init__(self, conv, num_filters, factor=2):
        """
        Initialize the Super-resolution upsampling block.

        Sets up convolutional layers that prepare feature maps for the `PixelShuffle`
        operation. It dynamically adjusts the number of intermediate convolutions
        based on whether the total upscaling `factor` is 2 or 4.

        Parameters
        ----------
        conv : Type[nn.Conv2d | nn.Conv3d]
            The convolutional layer type to use.
        num_filters : int
            The number of input channels to the upsampling block.
        factor : int, optional
            The total upscaling factor for the block (e.g., 2 or 4).
            If 4, it performs two successive 2x upsampling steps. Defaults to 2.
        """
        super(SR_upsampling, self).__init__()
        self.f = 2 if factor == 4 else factor
        self.conv1 = conv(num_filters, num_filters * (self.f**2), kernel_size=3, padding="same")
        self.conv2 = None
        if factor == 4:
            self.conv2 = conv(num_filters, num_filters * (self.f**2), kernel_size=3, padding="same")

    def forward(self, x) -> torch.Tensor:
        """
        Perform the forward pass of the Super-resolution upsampling block.

        The input `x` first undergoes a convolution to prepare for the first
        `PixelShuffle` operation. If the upscaling `factor` is 4, an additional
        convolution and `PixelShuffle` are applied.

        Parameters
        ----------
        x : torch.Tensor
            The input feature tensor before upsampling.
            Expected shape: `(batch_size, num_filters, ...spatial_dims)`.

        Returns
        -------
        torch.Tensor
            The upsampled output tensor. Its spatial dimensions will be scaled by
            the specified `factor`.
        """
        out = self.conv1(x)
        out = torch.nn.functional.pixel_shuffle(out, upscale_factor=self.f)
        if self.conv2:
            out = self.conv2(out)
            out = torch.nn.functional.pixel_shuffle(out, upscale_factor=self.f)
        return out
