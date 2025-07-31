"""
This module implements the Wide Activation for Efficient and Accurate Image Super-Resolution (WDSR) model.

WDSR is a convolutional neural network designed for single image super-resolution.
It introduces "wide activation" (using a larger number of feature maps in intermediate
layers of residual blocks) and employs weight normalization to stabilize training
and improve performance. The model consists of an initial convolutional layer,
a series of residual blocks, a final reconstruction layer, and a skip connection
with a PixelShuffle layer for upsampling.

Classes:
--------
- wdsr: The main WDSR model.
- Block: The residual block used within the WDSR architecture.

Reference:
----------
`Wide Activation for Efficient and Accurate Image Super-Resolution <https://arxiv.org/abs/1808.08718>`_.

Adapted from:
-------------
https://github.com/yjn870/WDSR-pytorch/tree/master
"""
import math
import torch
import torch.nn as nn
import torch.nn.init as init


class wdsr(nn.Module):
    """
    WDSR (Wide Activation for Efficient and Accurate Image Super-Resolution) model.

    This model is designed for single image super-resolution, aiming to reconstruct
    a high-resolution image from a low-resolution input. It utilizes residual blocks
    with wide activations and weight normalization for improved efficiency and accuracy.

    Reference: `Wide Activation for Efficient and Accurate Image Super-Resolution <https://arxiv.org/abs/1808.08718>`_.

    Adapted from `here <https://github.com/yjn870/WDSR-pytorch/tree/master>`_.
    """

    def __init__(
        self,
        scale,
        num_filters=32,
        num_res_blocks=16,
        res_block_expansion=6,
        num_channels=1,
    ):
        """
        Initialize the WDSR model.

        Sets up the main body of the network, including the initial convolution,
        a sequence of residual blocks (`Block`s), and the final convolutional layer.
        It also defines a skip connection path and a PixelShuffle layer for upsampling.
        Weight normalization is applied to convolutional layers.

        Parameters
        ----------
        scale : int | Tuple[int, ...]
            The super-resolution upscaling factor. If a tuple is provided (e.g., for
            multi-dimensional scaling), only the first element is used as the
            upscaling factor for `nn.PixelShuffle`.
        num_filters : int, optional
            The number of feature maps in the main body of the network. Defaults to 32.
        num_res_blocks : int, optional
            The number of residual blocks to stack in the network's body. Defaults to 16.
        res_block_expansion : int, optional
            The expansion factor for the intermediate channels within each residual block.
            This defines the "wide activation". Defaults to 6.
        num_channels : int, optional
            The number of input and output image channels (e.g., 1 for grayscale, 3 for RGB).
            Defaults to 1.
        """
        super(wdsr, self).__init__()
        # Extract the single integer scale factor from the input
        if type(scale) is tuple:
            scale = scale[0]
        
        kernel_size = 3 # Kernel size for main body convolutions
        skip_kernel_size = 5 # Kernel size for the skip connection convolution
        weight_norm = torch.nn.utils.weight_norm # Alias for weight normalization utility
        num_outputs = scale * scale * num_channels # Output channels needed for PixelShuffle

        # Main body of the WDSR model
        body = []
        # Initial convolutional layer with weight normalization
        conv = weight_norm(nn.Conv2d(num_channels, num_filters, kernel_size, padding=kernel_size // 2))
        init.ones_(conv.weight_g)
        init.zeros_(conv.bias)
        body.append(conv)

        # Add the specified number of residual blocks
        for _ in range(num_res_blocks):
            body.append(
                Block(
                    num_filters,
                    kernel_size,
                    res_block_expansion,
                    weight_norm=weight_norm,
                    res_scale=1 / math.sqrt(num_res_blocks),
                )
            )

        # Final convolutional layer to adjust the number of output channels
        conv = weight_norm(nn.Conv2d(num_filters, num_outputs, kernel_size, padding=kernel_size // 2))
        init.ones_(conv.weight_g)
        init.zeros_(conv.bias)
        body.append(conv)
        self.body = nn.Sequential(*body)

        # Skip connection (potentially with a convolution)
        skip = []
        # If the number of input channels does not match the number of outputs,
        # add a convolutional layer to adjust the channels
        if num_channels != num_outputs:
            conv = weight_norm(
                nn.Conv2d(
                    num_channels,
                    num_outputs,
                    skip_kernel_size,
                    padding=skip_kernel_size // 2,
                )
            )
            init.ones_(conv.weight_g)
            init.zeros_(conv.bias)
            skip.append(conv)
        self.skip = nn.Sequential(*skip) # Wrap skip layers in a Sequential module

        # PixelShuffle layer for upsampling
        shuf = []
        if scale > 1:
            shuf.append(nn.PixelShuffle(scale))
        self.shuf = nn.Sequential(*shuf)

    def forward(self, x) -> torch.Tensor:
        """
        Perform the forward pass of the WDSR model.

        The input `x` first passes through the main body of the network.
        A skip connection (potentially with a convolution) is added to the
        output of the main body. Finally, the combined features are passed
        through the PixelShuffle layer for super-resolution.

        Parameters
        ----------
        x : torch.Tensor
            The input low-resolution image tensor.
            Expected shape: `(batch_size, num_channels, H_lr, W_lr)`.

        Returns
        -------
        torch.Tensor
            The super-resolved high-resolution image tensor.
            Expected shape: `(batch_size, num_channels, H_hr, W_hr)`,
            where `H_hr = H_lr * scale` and `W_hr = W_lr * scale`.
        """
        x = self.body(x) + self.skip(x)
        x = self.shuf(x)
        return x


class Block(nn.Module):
    """
    Residual block used in the WDSR model.

    This block implements the "wide activation" concept by expanding the number
    of channels in its intermediate convolutional layer. It includes a residual
    connection and applies weight normalization.
    """

    def __init__(
        self,
        num_residual_units,
        kernel_size,
        width_multiplier=1,
        weight_norm=torch.nn.utils.weight_norm,
        res_scale=1,
    ):
        """
        Initialize a residual block for WDSR.

        Sets up two convolutional layers with an intermediate expansion of channels
        (controlled by `width_multiplier`). ReLU activation is applied after the
        first convolution. Weight normalization is used, and the output of the
        second convolution is scaled by `res_scale` before being added to the input.

        Parameters
        ----------
        num_residual_units : int
            The number of input and output channels for the residual block.
        kernel_size : int
            The kernel size for the convolutional layers within the block.
        width_multiplier : int, optional
            The factor by which the number of channels is multiplied in the
            intermediate convolutional layer. This creates the "wide activation".
            Defaults to 1.
        weight_norm : Callable, optional
            A callable for applying weight normalization (e.g., `torch.nn.utils.weight_norm`).
            Defaults to `torch.nn.utils.weight_norm`.
        res_scale : float, optional
            A scaling factor applied to the output of the convolutional path before
            adding it to the residual connection. This helps stabilize training
            in very deep networks. Defaults to 1.
        """
        super(Block, self).__init__()
        body = []
        conv = weight_norm(
            nn.Conv2d(
                num_residual_units,
                int(num_residual_units * width_multiplier),
                kernel_size,
                padding=kernel_size // 2,
            )
        )
        init.constant_(conv.weight_g, 2.0)
        init.zeros_(conv.bias)
        body.append(conv)
        body.append(nn.ReLU(True))
        conv = weight_norm(
            nn.Conv2d(
                int(num_residual_units * width_multiplier),
                num_residual_units,
                kernel_size,
                padding=kernel_size // 2,
            )
        )
        init.constant_(conv.weight_g, res_scale)
        init.zeros_(conv.bias)
        body.append(conv)

        self.body = nn.Sequential(*body)

    def forward(self, x):
        """
        Perform the forward pass of the residual block.

        The input `x` is processed through the convolutional path. The output
        of this path is then added to the original input `x` via a residual connection.

        Parameters
        ----------
        x : torch.Tensor
            The input feature tensor to the residual block.

        Returns
        -------
        torch.Tensor
            The output feature tensor after applying the residual block. Same shape as input `x`.
        """
        x = self.body(x) + x
        return x
