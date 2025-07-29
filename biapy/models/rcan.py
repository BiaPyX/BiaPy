"""
This module implements the Deep Residual Channel Attention Networks (RCAN) model, a prominent architecture for image super-resolution.

RCAN leverages very deep residual networks combined with channel attention
mechanisms to achieve high-quality image reconstruction. The model is built
upon several key components:

Classes:
--------
- ChannelAttention: Implements a channel attention mechanism that recalibrates
  channel-wise feature responses by modeling interdependencies between channels.
- RCAB (Residual Channel Attention Block): A fundamental building block that
  combines residual learning with the ChannelAttention mechanism.
- RG (Residual Group): A collection of multiple RCABs, followed by a
  convolutional layer, with a global residual connection.
- rcan: The main RCAN model, integrating the initial feature extraction,
  multiple Residual Groups, and an optional upscaling module for super-resolution.

The implementation supports both 2D and 3D image inputs and is adapted from
the official RCAN-pytorch repository.

Reference:
----------
`Image Super-Resolution Using Very Deep Residual Channel Attention Networks
<https://openaccess.thecvf.com/content_ECCV_2018/html/Yulun_Zhang_Image_Super-Resolution_Using_ECCV_2018_paper.html>`_.

Adapted from:
-------------
`https://github.com/yjn870/RCAN-pytorch`_.
"""
import torch
from torch import nn

class ChannelAttention(nn.Module):
    """
    Implements a Channel Attention mechanism.

    This module recalibrates channel-wise feature responses by adaptively
    learning the interdependencies between channels. It uses global average
    pooling to compute channel-wise statistics, followed by a small MLP
    (two 1x1 convolutions with SiLU and Sigmoid activations) to predict
    channel-wise scaling factors.
    """

    def __init__(self, num_features, reduction, ndim=2):
        """
        Initialize the ChannelAttention module.

        Sets up the adaptive pooling layer and the sequential convolutional
        layers that form the core of the channel attention mechanism.
        The choice between 2D and 3D layers depends on `ndim`.

        Parameters
        ----------
        num_features : int
            The number of input and output channels for the attention module.
        reduction : int
            The reduction ratio for the intermediate channel dimension in the
            MLP-like structure. A higher reduction leads to a smaller model
            but might reduce expressive power.
        ndim : int, optional
            The number of spatial dimensions of the input data (2 for 2D, 3 for 3D).
            Defaults to 2.
        """
        super(ChannelAttention, self).__init__()
        if ndim == 2:
            conv = nn.Conv2d
            avg_pool = nn.AdaptiveAvgPool2d
        else:
            conv = nn.Conv3d
            avg_pool = nn.AdaptiveAvgPool3d
        self.module = nn.Sequential(
            avg_pool(1),
            conv(num_features, num_features // reduction, kernel_size=1),
            nn.SiLU(inplace=True),
            conv(num_features // reduction, num_features, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Perform the forward pass of the ChannelAttention module.

        Computes channel attention weights from the input `x` and then
        multiplies `x` element-wise by these weights, effectively
        recalibrating the features.

        Parameters
        ----------
        x : torch.Tensor
            The input feature tensor.
            Expected shape for 2D: `(batch_size, num_features, H, W)`.
            Expected shape for 3D: `(batch_size, num_features, D, H, W)`.

        Returns
        -------
        torch.Tensor
            The feature tensor after applying channel attention. Same shape as input `x`.
        """
        return x * self.module(x)


class RCAB(nn.Module):
    """
    Residual Channel Attention Block (RCAB).

    This block is a fundamental building unit of RCAN. It combines a residual
    connection with two convolutional layers and a ChannelAttention module
    to enhance feature learning and improve reconstruction quality.
    """

    def __init__(self, num_features, reduction, ndim=2):
        """
        Initialize the Residual Channel Attention Block.

        Sets up two convolutional layers and integrates a `ChannelAttention`
        module within a residual learning framework. The choice of 2D or 3D
        convolution depends on `ndim`.

        Parameters
        ----------
        num_features : int
            The number of input and output channels for the convolutional layers
            and the `ChannelAttention` module within the block.
        reduction : int
            The reduction ratio passed to the `ChannelAttention` module.
        ndim : int, optional
            The number of spatial dimensions of the input data (2 for 2D, 3 for 3D).
            Defaults to 2.
        """
        super(RCAB, self).__init__()
        if ndim == 2:
            conv = nn.Conv2d
        else:
            conv = nn.Conv3d
            
        self.module = nn.Sequential(
            conv(num_features, num_features, kernel_size=3, padding="same"),
            nn.SiLU(inplace=True),
            conv(num_features, num_features, kernel_size=3, padding="same"),
            ChannelAttention(num_features, reduction, ndim=ndim),
        )

    def forward(self, x):
        """
        Perform the forward pass of the Residual Channel Attention Block.

        The input `x` is processed through a sequence of convolutions and
        channel attention. The output of this sequence is then added back
        to the original input `x` via a residual connection.

        Parameters
        ----------
        x : torch.Tensor
            The input feature tensor to the RCAB.

        Returns
        -------
        torch.Tensor
            The output feature tensor after applying the RCAB. Same shape as input `x`.
        """
        return x + self.module(x)


class RG(nn.Module):
    """
    Residual Group (RG).

    A Residual Group consists of multiple Residual Channel Attention Blocks (RCABs)
    followed by a convolutional layer. It incorporates a global residual connection
    around the entire group, allowing for the construction of very deep networks.
    """

    def __init__(self, num_features, num_rcab, reduction, ndim=2):
        """
        Initialize a Residual Group.

        Constructs a sequence of `num_rcab` RCABs and appends a final
        convolutional layer. The entire sequence is wrapped in `nn.Sequential`.

        Parameters
        ----------
        num_features : int
            The number of features (channels) processed throughout the group.
        num_rcab : int
            The number of `RCAB` blocks to include in this Residual Group.
        reduction : int
            The reduction ratio passed to each `RCAB`'s `ChannelAttention` module.
        ndim : int, optional
            The number of spatial dimensions (2 for 2D, 3 for 3D). Defaults to 2.
        """
        super(RG, self).__init__()
        if ndim == 2:
            conv = nn.Conv2d
        else:
            conv = nn.Conv3d
        self.module = [RCAB(num_features, reduction, ndim=ndim) for _ in range(num_rcab)]
        self.module.append(conv(num_features, num_features, kernel_size=3, padding="same"))
        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        """
        Perform the forward pass of the Residual Group.

        The input `x` is processed through the sequence of RCABs and the final
        convolutional layer. The output of this sequence is then added back
        to the original input `x` via a global residual connection.

        Parameters
        ----------
        x : torch.Tensor
            The input feature tensor to the Residual Group.

        Returns
        -------
        torch.Tensor
            The output feature tensor after processing through the Residual Group.
            Same shape as input `x`.
        """
        return x + self.module(x)


class rcan(nn.Module):
    """
    Deep Residual Channel Attention Networks (RCAN) model.

    RCAN is a very deep residual network designed for image super-resolution.
    It utilizes Residual Groups (RG) composed of Residual Channel Attention Blocks (RCABs)
    to learn hierarchical features and enhance reconstruction quality by focusing
    on informative channels. The model includes an initial feature extraction layer,
    multiple RGs, a global skip connection, and an optional upscaling layer.

    Reference: `Image Super-Resolution Using Very Deep Residual Channel Attention Networks
    <https://openaccess.thecvf.com/content_ECCV_2018/html/Yulun_Zhang_Image_Super-Resolution_Using_ECCV_2018_paper.html>`_.

    Adapted from `here <https://github.com/yjn870/RCAN-pytorch>`_.
    """

    def __init__(
        self,
        ndim,
        num_channels=3,
        filters=64,
        scale=2,
        num_rg=10,
        num_rcab=20,
        reduction=16,
        upscaling_layer=True,
    ):
        """
        Initialize the RCAN model.

        Sets up the initial shallow feature extraction layer, a sequence of
        Residual Groups (RGs), a global convolutional layer, an optional
        upscaling module (using PixelShuffle), and the final reconstruction layer.
        The choice between 2D and 3D layers depends on `ndim`.

        Parameters
        ----------
        ndim : int
            The number of spatial dimensions of the input data (2 for 2D, 3 for 3D).
        num_channels : int, optional
            The number of input and output image channels (e.g., 3 for RGB). Defaults to 3.
        filters : int, optional
            The number of feature maps (channels) used throughout the main body of
            the network (e.g., within RGs and RCABs). Defaults to 64.
        scale : int | Tuple[int, ...], optional
            The super-resolution upscaling factor. If a tuple, only the first element
            is used as `PixelShuffle` expects a single integer factor. Defaults to 2.
        num_rg : int, optional
            The number of Residual Groups (RGs) to stack in the network. Defaults to 10.
        num_rcab : int, optional
            The number of RCABs within each Residual Group. Defaults to 20.
        reduction : int, optional
            The reduction ratio for the `ChannelAttention` module within each RCAB. Defaults to 16.
        upscaling_layer : bool, optional
            If True, an upscaling layer (using PixelShuffle) is included before the
            final convolutional layer to perform super-resolution. If False, the
            model outputs at the same resolution as the input features. Defaults to True.
        """
        super(rcan, self).__init__()
        if type(scale) is tuple:
            scale = scale[0]
        self.ndim = ndim
        self.upscaling_layer = upscaling_layer
        if ndim == 2:
            conv = nn.Conv2d
        else:
            conv = nn.Conv3d
        
        # Shallow Feature Extraction (SF)
        self.sf = conv(num_channels, filters, kernel_size=3, padding="same")

        # Residual Groups (RGs)
        self.rgs = nn.Sequential(*[RG(filters, num_rcab, reduction, ndim=ndim) for _ in range(num_rg)])

        # Global Skip Connection Convolution
        self.conv1 = conv(filters, filters, kernel_size=3, padding="same")

        # Optional Upscaling Layer
        if upscaling_layer:
            self.upscale = nn.Sequential(
                conv(filters, filters * (scale**2), kernel_size=3, padding="same"),
                nn.PixelShuffle(scale),
            )
        
        # Final Reconstruction Layer
        self.conv2 = conv(filters, num_channels, kernel_size=3, padding="same")

    def forward(self, x) -> torch.Tensor:
        """
        Perform the forward pass of the RCAN model.

        The input `x` first undergoes shallow feature extraction. Then, it passes
        through a sequence of Residual Groups (RGs). A global residual connection
        adds the output of the RGs to the initial features. Optionally, an
        upscaling layer is applied, followed by a final convolutional layer to
        produce the super-resolved output.

        Parameters
        ----------
        x : torch.Tensor
            The input image tensor.
            Expected shape for 2D: `(batch_size, num_channels, H, W)`.
            Expected shape for 3D: `(batch_size, num_channels, D, H, W)`.

        Returns
        -------
        torch.Tensor
            The super-resolved output image tensor.
            If `upscaling_layer` is True, its spatial dimensions will be scaled
            by the `scale` factor. The number of channels will match `num_channels`.
        """
        x = self.sf(x)
        residual = x
        x = self.rgs(x)
        x = self.conv1(x)
        x += residual
        if self.upscaling_layer:
            x = self.upscale(x)
        x = self.conv2(x)
        return x
