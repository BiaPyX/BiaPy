"""
This module contains a collection of fundamental building blocks for convolutional neural networks, primarily designed for biomedical image segmentation architectures like U-Nets and their variants.

It provides modular components for various operations, including:

- **Convolutional Layers:** Basic `ConvBlock` and `DoubleConvBlock` for standard feature extraction.
- **Attention Mechanisms:** `AttentionBlock` to integrate attention gating into skip connections.
- **Squeeze-and-Excitation Networks:** `SqExBlock` for channel-wise feature recalibration.
- **ConvNeXt Blocks:** Both `ConvNeXtBlock_V1` and `ConvNeXtBlock_V2` for modern, efficient
  feature processing with depthwise convolutions, layer normalization, and residual connections.
- **Global Response Normalization:** `GRN` layer used within ConvNeXt V2 for enhanced feature discrimination.
- **Upsampling Blocks:** `UpBlock`, `UpConvNeXtBlock_V1`, and `UpConvNeXtBlock_V2` for
  decoder paths, handling upsampling, skip connection concatenation, and feature refinement.

The blocks are designed to be flexible, supporting both 2D and 3D operations,
various normalization types, activations, and configurable parameters like kernel sizes and dropout.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.stochastic_depth import StochasticDepth
from torchvision.ops.misc import Permute
from typing import Dict, Optional, List, Tuple, Any, Type


class ConvBlock(nn.Module):
    """
    Implements a standard Convolutional Block.

    This block consists of a convolutional layer followed by optional
    normalization, activation, dropout, and a Squeeze-and-Excitation (SE) block.
    It serves as a versatile building component in various convolutional
    neural network architectures.
    """

    def __init__(
        self,
        conv,
        in_size,
        out_size,
        k_size,
        padding: int | str = "same",
        stride=1,
        bias=True,
        act=None,
        norm="none",
        dropout=0,
        se_block=False,
    ):
        """
        Initialize the Convolutional Block.

        Sets up the core convolutional layer along with configurable
        normalization, activation, dropout, and an optional Squeeze-and-Excitation block.

        Parameters
        ----------
        conv : Type[nn.Conv2d | nn.Conv3d]
            The convolutional layer type to use (e.g., `nn.Conv2d` for 2D, `nn.Conv3d` for 3D).
        in_size : int
            Number of input feature channels.
        out_size : int
            Number of output feature channels.
        k_size : int or tuple
            Kernel size for the convolutional layer.
        padding : int or str, optional
            Padding type for the convolutional layer. Can be an integer or "same".
            If "same", padding is calculated to maintain output spatial dimensions.
            Defaults to "same".
        stride : int or tuple, optional
            Stride for the convolutional layer. Defaults to 1.
        bias : bool, optional
            Whether to include a bias term in the convolutional layer. Defaults to `True`.
        act : Optional[str], optional
            Activation layer to use after normalization. E.g., "relu", "gelu".
            If `None`, no activation is applied. Defaults to `None`.
        norm : str, optional
            Normalization layer type to use after convolution.
            Options include `'bn'` (BatchNorm), `'sync_bn'` (SyncBatchNorm),
            `'in'` (InstanceNorm), `'gn'` (GroupNorm), or `'none'` (no normalization).
            Defaults to "none".
        dropout : float, optional
            Dropout probability to apply after activation (if any). If 0, no dropout.
            Defaults to 0.
        se_block : bool, optional
            Whether to add a Squeeze-and-Excitation (`SqExBlock`) after all other
            operations in the block. Defaults to `False`.
        """
        super(ConvBlock, self).__init__()
        block = []

        block.append(conv(in_size, out_size, kernel_size=k_size, padding=padding, stride=stride, bias=bias))
        if norm != "none":
            if conv == nn.Conv2d:
                block.append(get_norm_2d(norm, out_size))
            else:
                block.append(get_norm_3d(norm, out_size))
        if act:
            block.append(get_activation(act))
        if dropout > 0:
            block.append(nn.Dropout(dropout))
        if se_block:
            block.append(SqExBlock(out_size, ndim=2 if conv == nn.Conv2d else 3))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        """
        Perform the forward pass of the Convolutional Block.

        Processes the input tensor sequentially through the defined layers:
        convolution, optional normalization, optional activation, optional dropout,
        and optional Squeeze-and-Excitation.

        Parameters
        ----------
        x : torch.Tensor
            The input feature tensor.
            Expected shape for 2D: (batch_size, in_size, height, width).
            Expected shape for 3D: (batch_size, in_size, depth, height, width).

        Returns
        -------
        torch.Tensor
            The output tensor after passing through the block.
            Its shape will be (batch_size, out_size, H', W') or (batch_size, out_size, D', H', W'),
            where H', W' (and D') depend on `padding` and `stride`.
        """
        out = self.block(x)
        return out


class DoubleConvBlock(nn.Module):
    """
    Implements a Double Convolutional Block.

    This block consists of two sequential `ConvBlock` layers. It is a common
    building component in many convolutional neural network architectures,
    especially in U-Net-like models, to extract features.
    """

    def __init__(
        self,
        conv,
        in_size,
        out_size,
        k_size,
        act=None,
        norm="none",
        dropout=0,
        se_block=False,
    ):
        """
        Initialize the Double Convolutional Block.

        Sets up two `ConvBlock` layers sequentially. The first `ConvBlock`
        transforms the input from `in_size` channels to `out_size` channels,
        and the second `ConvBlock` maintains `out_size` channels.

        Parameters
        ----------
        conv : Type[nn.Conv2d | nn.Conv3d]
            The convolutional layer type to use within each `ConvBlock`.
        in_size : int
            Number of input feature channels to the first `ConvBlock`.
        out_size : int
            Number of output feature channels for the entire `DoubleConvBlock`.
            Both internal `ConvBlock`s will output this number of channels.
        k_size : int or tuple
            Kernel size for the convolutional layers within each `ConvBlock`.
        act : Optional[str], optional
            Activation layer to use within each `ConvBlock`. Defaults to `None`.
        norm : str, optional
            Normalization layer type to use within each `ConvBlock`.
            Options include `'bn'`, `'sync_bn'`, `'in'`, `'gn'`, or `'none'`.
            Defaults to "none".
        dropout : float, optional
            Dropout value to be fixed within each `ConvBlock`. Defaults to 0.
        se_block : bool, optional
            Whether to add a Squeeze-and-Excitation (SE) block at the end of
            each `ConvBlock`. Defaults to `False`.
        """
        super(DoubleConvBlock, self).__init__()
        block = []
        block.append(
            ConvBlock(
                conv=conv,
                in_size=in_size,
                out_size=out_size,
                k_size=k_size,
                act=act,
                norm=norm,
                dropout=dropout,
                se_block=se_block,
            )
        )
        block.append(
            ConvBlock(
                conv=conv,
                in_size=out_size,
                out_size=out_size,
                k_size=k_size,
                act=act,
                norm=norm,
                dropout=dropout,
                se_block=se_block,
            )
        )
        self.block = nn.Sequential(*block)

    def forward(self, x):
        """
        Perform the forward pass of the Double Convolutional Block.

        Processes the input tensor sequentially through the two `ConvBlock` layers.

        Parameters
        ----------
        x : torch.Tensor
            The input feature tensor.
            Expected shape for 2D: (batch_size, in_size, height, width).
            Expected shape for 3D: (batch_size, in_size, depth, height, width).

        Returns
        -------
        torch.Tensor
            The output tensor after passing through both `ConvBlock` layers.
            Its shape will be (batch_size, out_size, H', W') or (batch_size, out_size, D', H', W'),
            where H', W' (and D') match the input spatial dimensions if padding is 'same'.
        """
        out = self.block(x)
        return out


class ConvNeXtBlock_V1(nn.Module):
    """
    Implements a single ConvNeXt V1 block.

    This block is a fundamental building component of ConvNeXt V1 networks,
    featuring a depthwise convolution, a LayerNorm-Linear-GELU-Linear path,
    layer scaling, and a stochastic depth residual connection.
    """

    def __init__(self, ndim, conv, dim, layer_scale=1e-6, stochastic_depth_prob=0.0, layer_norm=None, k_size=7):
        """
        Initialize the ConvNeXt V1 block.

        Sets up the depthwise convolution, permutation-aware Layer Normalization,
        an MLP with GELU activation, optional learnable layer scaling, and a
        stochastic depth regularizer for the residual connection.

        Parameters
        ----------
        ndim : int
            Number of dimensions of the input data (2 for 2D, 3 for 3D).
        conv : Type[nn.Conv2d | nn.Conv3d]
            The convolutional layer type to use for the depthwise convolution.
        dim : int
            Number of input and output channels for the block.
        layer_scale : float, optional
            Initial value for the learnable layer scale parameter. If > 0,
            a `nn.Parameter` is created for scaling. Defaults to 1e-6.
        stochastic_depth_prob : float, optional
            The probability of dropping the residual branch during training.
            Defaults to 0.0 (no dropout).
        layer_norm : Optional[Type[nn.LayerNorm]], optional
            The Layer Normalization layer type to use. If `None`, `nn.LayerNorm` is used.
            Defaults to `None`.
        k_size : int or tuple, optional
            Height, width, and depth (for 3D) of the depthwise convolution window.
            Defaults to 7.
        """
        super().__init__()

        if layer_norm is None:
            layer_norm = nn.LayerNorm

        if ndim == 3:
            pre_ln_permutation = Permute([0, 2, 3, 4, 1])
            post_ln_permutation = Permute([0, 4, 1, 2, 3])
            layer_scale_dim = (dim, 1, 1, 1)
            pad = (0, 3, 3) if k_size[0] == 1 else (3, 3, 3)
        elif ndim == 2:
            pre_ln_permutation = Permute([0, 2, 3, 1])
            post_ln_permutation = Permute([0, 3, 1, 2])
            layer_scale_dim = (dim, 1, 1)
            pad = (3, 3)

        self.block = nn.Sequential(
            conv(dim, dim, kernel_size=k_size, padding=pad, groups=dim, bias=True),  # depthwise conv
            pre_ln_permutation,
            layer_norm(dim, eps=1e-6),
            nn.Linear(
                in_features=dim, out_features=4 * dim, bias=True
            ),  # pointwise/1x1 convs, implemented with linear layers
            nn.GELU(),
            nn.Linear(in_features=4 * dim, out_features=dim, bias=True),
            post_ln_permutation,
        )
        self.layer_scale = (
            nn.Parameter(torch.ones(layer_scale_dim) * layer_scale, requires_grad=True) if layer_scale > 0 else None
        )
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, x):
        """
        Perform the forward pass of the ConvNeXt V1 block.

        Processes the input through a depthwise convolution, layer normalization,
        and an MLP with GELU. The output of this path is optionally scaled
        by a learnable layer scale parameter and then added to the original
        input via a residual connection, applying stochastic depth.

        Parameters
        ----------
        x : torch.Tensor
            The input feature tensor.
            Expected shape for 2D: (batch_size, dim, height, width).
            Expected shape for 3D: (batch_size, dim, depth, height, width).

        Returns
        -------
        torch.Tensor
            The output tensor of the ConvNeXt V1 block, with the same shape as the input.
        """
        result = self.block(x)
        if self.layer_scale is not None:
            result = self.layer_scale * result
        result = x + self.stochastic_depth(result)
        return result


class GRN(nn.Module):
    """
    Implement the Global Response Normalization (GRN) layer.

    This layer enhances feature discrimination by normalizing features
    based on global responses across channels, as introduced in ConvNeXt V2.
    It includes learnable parameters for scaling and shifting.
    """

    def __init__(self, dim):
        """
        Initialize the Global Response Normalization (GRN) layer.

        Sets up learnable scaling (`gamma`) and biasing (`beta`) parameters
        for the normalization process, initialized to zeros.

        Parameters
        ----------
        dim : int
            The number of input feature channels/dimensions.
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        """
        Perform the forward pass of the GRN layer.

        Calculates the L2 norm (`Gx`) across spatial dimensions for each channel.
        Then, it normalizes `Gx` by its mean across channels (`Nx`).
        Finally, it applies the learnable `gamma` and `beta` parameters to `x * Nx`
        and adds the original input `x` as a residual connection.

        Parameters
        ----------
        x : torch.Tensor
            The input feature tensor. Expected to be in a channel-last format
            for proper normalization (e.g., [B, D, H, W, C] for 3D, or [B, H, W, C] for 2D)
            when `gamma` and `beta` are applied. Assuming the input `x` is permuted
            to (B, spatial_dims..., C) before this layer.

        Returns
        -------
        torch.Tensor
            The normalized feature tensor, with the same shape as the input `x`.
        """
        # Gx: L2 norm over spatial dimensions, keepdim=True for broadcasting
        # Assuming x is (B, D, H, W, C) or (B, H, W, C) due to pre/post LN permutation
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        # Nx: Normalize Gx by its mean across channels
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)

        # Apply gamma, beta, and add residual
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtBlock_V2(nn.Module):
    """
    Implements a single ConvNeXt V2 block.

    This block is a fundamental building component of ConvNeXt V2 networks,
    featuring a depthwise convolution, a Permute-LayerNorm-Linear-GELU-GRN-Linear-Permute
    path, and a stochastic depth residual connection.
    """

    def __init__(self, ndim, conv, dim, stochastic_depth_prob=0.0, layer_norm=None, k_size=7):
        """
        Initialize the ConvNeXt V2 block.

        Sets up the depthwise convolution, a permutation-aware Layer Normalization,
        an MLP with GELU and Global Response Normalization (GRN), and a stochastic
        depth regularizer for the residual connection.

        Parameters
        ----------
        ndim : int
            Number of dimensions of the input data (2 for 2D, 3 for 3D).
        conv : Type[nn.Conv2d | nn.Conv3d]
            The convolutional layer type to use for the depthwise convolution.
        dim : int
            Number of input and output channels for the block.
        stochastic_depth_prob : float, optional
            The probability of dropping the residual branch during training.
            Defaults to 0.0 (no dropout).
        layer_norm : Optional[Type[nn.LayerNorm]], optional
            The Layer Normalization layer type to use. If `None`, `nn.LayerNorm` is used.
            Defaults to `None`.
        k_size : int or tuple, optional
            Height, width, and depth (for 3D) of the depthwise convolution window.
            Defaults to 7.
        """
        super().__init__()

        if layer_norm is None:
            layer_norm = nn.LayerNorm

        if ndim == 3:
            pre_ln_permutation = Permute([0, 2, 3, 4, 1])
            post_ln_permutation = Permute([0, 4, 1, 2, 3])
            pad = (0, 3, 3) if k_size[0] == 1 else (3, 3, 3)
        elif ndim == 2:
            pre_ln_permutation = Permute([0, 2, 3, 1])
            post_ln_permutation = Permute([0, 3, 1, 2])
            pad = (3, 3)

        self.block = nn.Sequential(
            conv(dim, dim, kernel_size=k_size, padding=pad, groups=dim, bias=True),  # depthwise conv
            pre_ln_permutation,
            layer_norm(dim, eps=1e-6),
            nn.Linear(in_features=dim, out_features=4 * dim, bias=True),
            nn.GELU(),
            GRN(4 * dim),
            nn.Linear(in_features=4 * dim, out_features=dim, bias=True),
            post_ln_permutation,
        )
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, x):
        """
        Perform the forward pass of the ConvNeXt V2 block.

        Processes the input through a depthwise convolution, layer normalization,
        and an MLP with GELU and GRN. The output of this path is then added
        to the original input via a residual connection, optionally applying
        stochastic depth.

        Parameters
        ----------
        x : torch.Tensor
            The input feature tensor.
            Expected shape for 2D: (batch_size, dim, height, width).
            Expected shape for 3D: (batch_size, dim, depth, height, width).

        Returns
        -------
        torch.Tensor
            The output tensor of the ConvNeXt V2 block, with the same shape as the input.
        """
        result = self.block(x)
        result = x + self.stochastic_depth(result)
        return result


class UpBlock(nn.Module):
    """
    Implements a standard Upsampling block, commonly used in the decoder path of U-Net-like architectures.

    This block performs an upsampling operation, concatenates the upsampled features
    with a skip connection (bridge) from the encoder, and then processes the combined
    features through a `DoubleConvBlock`. It supports different upsampling modes
    and optional attention gating.
    """

    def __init__(
        self,
        ndim,
        convtranspose,
        in_size,
        out_size,
        z_down,
        up_mode,
        conv,
        k_size,
        act=None,
        norm="none",
        dropout=0,
        attention_gate=False,
        se_block=False,
    ):
        """
        Initialize the Upsampling block.

        Sets up the upsampling layer (either transpose convolution or `nn.Upsample`
        followed by convolution), an optional attention gate, and a `DoubleConvBlock`
        for feature refinement after concatenation.

        Parameters
        ----------
        ndim : int
            Number of dimensions of the input data (2 for 2D, 3 for 3D).
        convtranspose : Type[nn.ConvTranspose2d | nn.ConvTranspose3d]
            The transpose convolutional layer type to use if `up_mode` is 'convtranspose'.
        in_size : int
            Number of input channels from the previous decoder stage (input to upsampling).
        out_size : int
            Number of output channels for this upsampling block after concatenation and processing.
        z_down : int
            Downsampling factor applied in the z-dimension for 3D data during upsampling.
            Only relevant if `ndim` is 3.
        up_mode : str
            The upsampling mode to use.
            - 'convtranspose': Uses a transpose convolution (`convtranspose`) for upsampling.
            - 'upsampling': Uses `nn.Upsample` (bilinear for 2D, trilinear for 3D) followed
                            by a 1x1 convolution to adjust channels.
        conv : Type[nn.Conv2d | nn.Conv3d]
            The convolutional layer type to use within internal blocks (e.g., `DoubleConvBlock`).
        k_size : int or tuple
            Kernel size for the convolutional layers within the `DoubleConvBlock`.
        act : Optional[str], optional
            Activation layer to use within the `DoubleConvBlock`. Defaults to `None`.
        norm : str, optional
            Normalization layer type to use within the upsampling path and `DoubleConvBlock`.
            Options include `'bn'`, `'sync_bn'`, `'in'`, `'gn'`, or `'none'`.
            Defaults to "none".
        dropout : float, optional
            Dropout value to be fixed within the `DoubleConvBlock`. Defaults to 0.
        attention_gate : bool, optional
            Whether to use an attention gate (`AttentionBlock`) to gate the skip connection
            before concatenation. Defaults to `False`.
        se_block : bool, optional
            Whether to add a Squeeze-and-Excitation (SE) block within the `DoubleConvBlock`.
            Defaults to `False`.
        """
        super(UpBlock, self).__init__()
        self.ndim = ndim
        block = []
        mpool = (z_down, 2, 2) if ndim == 3 else (2, 2)
        if up_mode == "convtranspose":
            block.append(convtranspose(in_size, out_size, kernel_size=mpool, stride=mpool))
        elif up_mode == "upsampling":
            block.append(nn.Upsample(mode="bilinear" if ndim == 2 else "trilinear", scale_factor=mpool))
            block.append(conv(in_size, out_size, kernel_size=1))
        if norm != "none":
            if conv == nn.Conv2d:
                block.append(get_norm_2d(norm, out_size))
            else:
                block.append(get_norm_3d(norm, out_size))
        if act is not None:
            block.append(get_activation(act))
        self.up = nn.Sequential(*block)

        if attention_gate:
            self.attention_gate = AttentionBlock(conv=conv, in_size=out_size, out_size=out_size // 2, norm=norm)
        else:
            self.attention_gate = None
        self.conv_block = DoubleConvBlock(
            conv=conv,
            in_size=out_size * 2,
            out_size=out_size,
            k_size=k_size,
            act=act,
            norm=norm,
            dropout=dropout,
            se_block=se_block,
        )

    def forward(self, x, bridge):
        """
        Perform the forward pass of the Upsampling block.

        First, it upsamples the input tensor `x`. If an attention gate is enabled,
        it uses the upsampled `x` and the `bridge` tensor to compute attention,
        then concatenates the upsampled `x` with the (potentially attended) `bridge`.
        Finally, the concatenated tensor is processed by a `DoubleConvBlock`.

        Parameters
        ----------
        x : torch.Tensor
            The input feature tensor from the previous decoder stage (lower resolution).
            Expected shape: (batch_size, in_size, D, H, W) or (batch_size, in_size, H, W).
        bridge : torch.Tensor
            The skip connection tensor from the corresponding encoder stage (higher resolution).
            Expected shape: (batch_size, out_size, D', H', W') or (batch_size, out_size, H', W'),
            where D', H', W' match the spatial dimensions after upsampling `x`.

        Returns
        -------
        torch.Tensor
            The output tensor of the upsampling block. Its shape will be
            (batch_size, out_size, D', H', W') or (batch_size, out_size, H', W'),
            matching the upsampled spatial dimensions and `out_size` channels.
        """
        up = self.up(x)
        if self.attention_gate is not None:
            attn = self.attention_gate(up, bridge)
            out = torch.cat([up, attn], 1)
        else:
            out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out


class UpConvNeXtBlock_V1(nn.Module):
    """
    Implements an Upsampling block using ConvNeXt V1 components.

    This block is designed for the upsampling path of U-Net-like architectures,
    combining upsampling with concatenation of skip connections, optional
    attention gating, and a sequence of ConvNeXt V1 blocks for feature refinement.
    """

    def __init__(
        self,
        ndim,
        convtranspose,
        in_size,
        out_size,
        z_down,
        up_mode,
        conv,
        attention_gate=False,
        se_block=False,
        cn_layers=1,
        sd_probs=[0.0],
        layer_scale=1e-6,
        layer_norm=None,
        k_size=7,
    ):
        """
        Initialize an Upsampling block using ConvNeXt V1 components.

        This block is designed for the upsampling path of U-Net-like architectures,
        combining upsampling with concatenation of skip connections, optional
        attention gating, and a sequence of ConvNeXt V1 blocks for feature refinement.

        Parameters
        ----------
        ndim : int
            Number of dimensions of the input data (2 for 2D, 3 for 3D).
        convtranspose : Type[nn.ConvTranspose2d | nn.ConvTranspose3d]
            The transpose convolutional layer type to use. Only used if ``up_mode`` is ``'convtranspose'``.
        in_size : int
            Number of input channels from the previous decoder stage (input to upsampling).
        out_size : int
            Number of output channels for this upsampling block after concatenation and processing.
        z_down : int, optional
            Downsampling factor applied in the z-dimension for 3D data during upsampling.
            Only relevant if `ndim` is 3. Defaults to 2.
        up_mode : str
            The upsampling mode to use.
            - 'convtranspose': Uses a transpose convolution (`convtranspose`) for upsampling.
            - 'upsampling': Uses `nn.Upsample` (bilinear for 2D, trilinear for 3D) followed
                            by a 1x1 convolution to adjust channels.
        conv : Type[nn.Conv2d | nn.Conv3d]
            The convolutional layer type to use within internal blocks (e.g., `ConvBlock`).
        attention_gate : bool, optional
            Whether to use an attention gate (`AttentionBlock`) to gate the skip connection
            before concatenation. Defaults to `False`.
        se_block : bool, optional
            Whether to add a Squeeze-and-Excitation (SE) block within the initial
            `ConvBlock` used after concatenation. Defaults to `False`.
        cn_layers : int, optional
            Number of `ConvNeXtBlock_V1` layers to stack after the initial concatenation and convolution.
            Defaults to 1.
        sd_probs : list of float, optional
            List of stochastic depth probabilities for each `ConvNeXtBlock_V1` layer.
            The length of this list should match `cn_layers`. Defaults to `[0.0]`.
        layer_scale : float, optional
            Layer scale parameter used in `ConvNeXtBlock_V1`. Defaults to 1e-6.
        layer_norm : Optional[nn.LayerNorm], optional
            The Layer Normalization layer type to use. If `None`, `nn.LayerNorm` is used.
            This normalization is applied before upsampling. Defaults to `None`.
        k_size : int or tuple, optional
            Height, width, and depth (for 3D) of the depthwise convolution window
            within the `ConvNeXtBlock_V1` layers. Defaults to 7.
        """
        super(UpConvNeXtBlock_V1, self).__init__()
        self.ndim = ndim
        block = []
        mpool = (z_down, 2, 2) if ndim == 3 else (2, 2)

        if ndim == 3:
            pre_ln_permutation = Permute([0, 2, 3, 4, 1])
            post_ln_permutation = Permute([0, 4, 1, 2, 3])
        else:
            pre_ln_permutation = Permute([0, 2, 3, 1])
            post_ln_permutation = Permute([0, 3, 1, 2])

        if layer_norm is not None:
            block.append(nn.Sequential(pre_ln_permutation, layer_norm(in_size), post_ln_permutation))
        else:
            layer_norm = nn.LayerNorm
            block.append(nn.Sequential(pre_ln_permutation, layer_norm(in_size), post_ln_permutation))

        # Upsampling
        if up_mode == "convtranspose":
            block.append(convtranspose(in_size, out_size, kernel_size=mpool, stride=mpool))
        elif up_mode == "upsampling":
            block.append(nn.Upsample(mode="bilinear" if ndim == 2 else "trilinear", scale_factor=mpool))
            block.append(conv(in_size, out_size, kernel_size=1))

        self.up = nn.Sequential(*block)

        # Define attention gate
        if attention_gate:
            self.attention_gate = AttentionBlock(conv=conv, in_size=out_size, out_size=out_size // 2)
        else:
            self.attention_gate = None

        # Convolution block to change dimensions of concatenated tensor
        self.conv_block = ConvBlock(conv, in_size=out_size * 2, out_size=out_size, k_size=1, se_block=se_block)

        # ConvNeXtBlock
        stage = nn.ModuleList()
        for i in reversed(range(cn_layers)):
            stage.append(
                ConvNeXtBlock_V1(ndim, conv, out_size, layer_scale, sd_probs[i], layer_norm=layer_norm, k_size=k_size)
            )
        self.cn_block = nn.Sequential(*stage)

    def forward(self, x, bridge):
        """
        Perform the forward pass of the UpConvNeXtBlock_V1.

        First, it upsamples the input tensor `x`. If an attention gate is enabled,
        it uses the upsampled `x` and the `bridge` tensor to compute attention,
        then concatenates the upsampled `x` with the (potentially attended) `bridge`.
        Finally, the concatenated tensor is processed by an initial convolutional
        block and then refined through a sequence of ConvNeXt V1 blocks.

        Parameters
        ----------
        x : torch.Tensor
            The input feature tensor from the previous decoder stage (lower resolution).
            Expected shape: (batch_size, in_size, D, H, W) or (batch_size, in_size, H, W).
        bridge : torch.Tensor
            The skip connection tensor from the corresponding encoder stage (higher resolution).
            Expected shape: (batch_size, out_size, D', H', W') or (batch_size, out_size, H', W'),
            where D', H', W' match the spatial dimensions after upsampling `x`.

        Returns
        -------
        torch.Tensor
            The output tensor of the upsampling block. Its shape will be
            (batch_size, out_size, D', H', W') or (batch_size, out_size, H', W'),
            matching the upsampled spatial dimensions and `out_size` channels.
        """
        up = self.up(x)
        if self.attention_gate is not None:
            attn = self.attention_gate(up, bridge)
            out = torch.cat([up, attn], 1)
        else:
            out = torch.cat([up, bridge], 1)

        out = self.conv_block(out)
        out = self.cn_block(out)
        return out


class UpConvNeXtBlock_V2(nn.Module):
    """
    Implements an Upsampling block using ConvNeXt V2 components.

    This block is designed for the upsampling path of U-Net-like architectures,
    combining upsampling with concatenation of skip connections, optional
    attention gating, and a sequence of ConvNeXt V2 blocks for feature refinement.
    """

    def __init__(
        self,
        ndim,
        convtranspose,
        in_size,
        out_size,
        z_down,
        up_mode,
        conv,
        attention_gate=False,
        se_block=False,
        cn_layers=1,
        sd_probs=[0.0],
        layer_norm=None,
        k_size=7,
    ):
        """
        Initialize an Upsampling block using ConvNeXt V2 components.

        This block is designed for the upsampling path of U-Net-like architectures,
        combining upsampling with concatenation of skip connections, optional
        attention gating, and a sequence of ConvNeXt V2 blocks for feature refinement.

        Parameters
        ----------
        ndim : int
            Number of dimensions of the input data (2 for 2D, 3 for 3D).
        convtranspose : Type[nn.ConvTranspose2d | nn.ConvTranspose3d]
            The transpose convolutional layer type to use if `up_mode` is 'convtranspose'.
        in_size : int
            Number of input channels from the previous decoder stage (input to upsampling).
        out_size : int
            Number of output channels for this upsampling block after concatenation and processing.
        z_down : int, optional
            Downsampling factor applied in the z-dimension for 3D data during upsampling.
            Only relevant if `ndim` is 3. Defaults to 2.
        up_mode : str
            The upsampling mode to use.
            - 'convtranspose': Uses a transpose convolution (`convtranspose`) for upsampling.
            - 'upsampling': Uses `nn.Upsample` (bilinear for 2D, trilinear for 3D) followed
                            by a 1x1 convolution to adjust channels.
        conv : Type[nn.Conv2d | nn.Conv3d]
            The convolutional layer type to use within internal blocks (e.g., `ConvBlock`).
        attention_gate : bool, optional
            Whether to use an attention gate (`AttentionBlock`) to gate the skip connection
            before concatenation. Defaults to `False`.
        se_block : bool, optional
            Whether to add a Squeeze-and-Excitation (SE) block within the initial
            `ConvBlock` used after concatenation. Defaults to `False`.
        cn_layers : int, optional
            Number of `ConvNeXtBlock_V2` layers to stack after the initial concatenation and convolution.
            Defaults to 1.
        sd_probs : list of float, optional
            List of stochastic depth probabilities for each `ConvNeXtBlock_V2` layer.
            The length of this list should match `cn_layers`. Defaults to `[0.0]`.
        layer_norm : Optional[nn.LayerNorm], optional
            The Layer Normalization layer type to use. If `None`, `nn.LayerNorm` is used.
            This normalization is applied before upsampling. Defaults to `None`.
        k_size : int or tuple, optional
            Height, width, and depth (for 3D) of the depthwise convolution window
            within the `ConvNeXtBlock_V2` layers. Defaults to 7.
        """
        super(UpConvNeXtBlock_V2, self).__init__()
        self.ndim = ndim
        block = []
        mpool = (z_down, 2, 2) if ndim == 3 else (2, 2)

        if ndim == 3:
            pre_ln_permutation = Permute([0, 2, 3, 4, 1])
            post_ln_permutation = Permute([0, 4, 1, 2, 3])
        else:
            pre_ln_permutation = Permute([0, 2, 3, 1])
            post_ln_permutation = Permute([0, 3, 1, 2])

        if layer_norm is not None:
            block.append(nn.Sequential(pre_ln_permutation, layer_norm(in_size), post_ln_permutation))
        else:
            layer_norm = nn.LayerNorm
            block.append(nn.Sequential(pre_ln_permutation, layer_norm(in_size), post_ln_permutation))

        # Upsampling
        if up_mode == "convtranspose":
            block.append(convtranspose(in_size, out_size, kernel_size=mpool, stride=mpool))
        elif up_mode == "upsampling":
            block.append(nn.Upsample(mode="bilinear" if ndim == 2 else "trilinear", scale_factor=mpool))
            block.append(conv(in_size, out_size, kernel_size=1))

        self.up = nn.Sequential(*block)

        # Define attention gate
        if attention_gate:
            self.attention_gate = AttentionBlock(conv=conv, in_size=out_size, out_size=out_size // 2)
        else:
            self.attention_gate = None

        # Convolution block to change dimensions of concatenated tensor
        self.conv_block = ConvBlock(conv, in_size=out_size * 2, out_size=out_size, k_size=1, se_block=se_block)

        # ConvNeXtBlock
        stage = nn.ModuleList()
        for i in reversed(range(cn_layers)):
            stage.append(ConvNeXtBlock_V2(ndim, conv, out_size, sd_probs[i], layer_norm=layer_norm, k_size=k_size))
        self.cn_block = nn.Sequential(*stage)

    def forward(self, x, bridge):
        """
        Perform the forward pass of the UpConvNeXtBlock_V2.

        First, it upsamples the input tensor `x`. If an attention gate is enabled,
        it uses the upsampled `x` and the `bridge` tensor to compute attention,
        then concatenates the upsampled `x` with the (potentially attended) `bridge`.
        Finally, the concatenated tensor is processed by an initial convolutional
        block and then refined through a sequence of ConvNeXt V2 blocks.

        Parameters
        ----------
        x : torch.Tensor
            The input feature tensor from the previous decoder stage (lower resolution).
            Expected shape: (batch_size, in_size, D, H, W) or (batch_size, in_size, H, W).
        bridge : torch.Tensor
            The skip connection tensor from the corresponding encoder stage (higher resolution).
            Expected shape: (batch_size, out_size, D', H', W') or (batch_size, out_size, H', W'),
            where D', H', W' match the spatial dimensions after upsampling `x`.

        Returns
        -------
        torch.Tensor
            The output tensor of the upsampling block. Its shape will be
            (batch_size, out_size, D', H', W') or (batch_size, out_size, H', W'),
            matching the upsampled spatial dimensions and `out_size` channels.
        """
        up = self.up(x)
        if self.attention_gate is not None:
            attn = self.attention_gate(up, bridge)
            out = torch.cat([up, attn], 1)
        else:
            out = torch.cat([up, bridge], 1)

        out = self.conv_block(out)
        out = self.cn_block(out)
        return out


class AttentionBlock(nn.Module):
    """
    Implements an Attention Block, as proposed in Attention U-Net.

    This block refines skip connections in U-Net-like architectures by generating
    attention coefficients. It learns to focus on salient features from the
    skip pathway (`x`) guided by the features from the coarser, upsampled pathway (`g`).

    Reference: `Attention U-Net: Learning Where to Look for the Pancreas <https://arxiv.org/abs/1804.03999>`_.
    """

    def __init__(self, conv, in_size, out_size, norm="none"):
        """
        Initialize the Attention Block with convolutional layers for gating and input signals.

        Sets up three distinct convolutional pathways: `w_g` for the gating signal,
        `w_x` for the skip connection input, and `psi` for generating the attention map.
        Each pathway includes optional normalization.

        Parameters
        ----------
        conv : Type[nn.Conv2d | nn.Conv3d]
            The convolutional layer type to use (e.g., `nn.Conv2d` for 2D, `nn.Conv3d` for 3D).
        in_size : int
            Number of input feature channels for both the gating signal (`g`) and
            the skip connection input (`x`).
        out_size : int
            Number of output channels for the intermediate convolutional layers
            (`w_g` and `w_x` outputs). The `psi` layer reduces this to 1 channel.
        norm : str, optional
            Normalization layer type to use within the convolutional sub-blocks.
            Options include `'bn'` (BatchNorm), `'sync_bn'` (SyncBatchNorm),
            `'in'` (InstanceNorm), `'gn'` (GroupNorm), or `'none'` (no normalization).
            Defaults to "none".
        """
        super(AttentionBlock, self).__init__()
        w_g = []
        w_g.append(conv(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=True))
        if norm != "none":
            if conv == nn.Conv2d:
                w_g.append(get_norm_2d(norm, out_size))
            else:
                w_g.append(get_norm_3d(norm, out_size))
        self.w_g = nn.Sequential(*w_g)

        w_x = []
        w_x.append(conv(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=True))
        if norm != "none":
            if conv == nn.Conv2d:
                w_g.append(get_norm_2d(norm, out_size))
            else:
                w_g.append(get_norm_3d(norm, out_size))
        self.w_x = nn.Sequential(*w_x)

        psi = []
        psi.append(conv(out_size, 1, kernel_size=1, stride=1, padding=0, bias=True))
        if norm != "none":
            if conv == nn.Conv2d:
                psi.append(get_norm_2d(norm, 1))
            else:
                psi.append(get_norm_3d(norm, 1))
        psi.append(nn.Sigmoid())
        self.psi = nn.Sequential(*psi)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        Perform the forward pass of the Attention Block.

        Processes the gating signal `g` and the skip connection `x` independently
        through convolutional layers. Their outputs are summed, passed through
        a ReLU activation, and then a 1x1 convolution with Sigmoid activation
        generates the attention coefficients (`psi`). Finally, these attention
        coefficients are multiplied element-wise with the skip connection `x`
        to produce the attention-gated output.

        Parameters
        ----------
        g : torch.Tensor
            The gating signal tensor from the coarser (upsampled) pathway.
            Expected shape: (batch_size, in_size, D, H, W) or (batch_size, in_size, H, W).
        x : torch.Tensor
            The skip connection tensor from the corresponding encoder pathway.
            Expected shape: (batch_size, in_size, D, H, W) or (batch_size, in_size, H, W).
            Spatial dimensions must match those of `g` after any necessary upsampling of `g`.

        Returns
        -------
        torch.Tensor
            The attention-gated feature tensor, with the same shape as `x`.
        """
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return psi * x


class SqExBlock(nn.Module):
    """
    Implements the Squeeze-and-Excitation (SE) block, a computational unit that adaptively recalibrates channel-wise feature responses.

    This block enhances the representational power of a network by explicitly
    modeling interdependencies between channels, allowing the network to
    perform feature recalibration.

    Reference: `Squeeze and Excitation Networks <https://arxiv.org/abs/1709.01507>`_.
    Credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4
    """

    def __init__(self, c, r=16, ndim=2):
        """
        Initialize the Squeeze-and-Excitation block.

        Sets up the squeeze operation (global average pooling) and the excitation
        operation (two fully connected layers with ReLU and Sigmoid activations).

        Parameters
        ----------
        c : int
            Number of input channels to the block.
        r : int, optional
            Reduction ratio for the number of channels in the excitation branch.
            The hidden dimension will be `c // r`. Defaults to 16.
        ndim : int, optional
            Number of dimensions of the input data.
            Use 2 for 2D data (e.g., images) which implies `nn.AdaptiveAvgPool2d`.
            Use 3 for 3D data (e.g., volumetric scans) which implies `nn.AdaptiveAvgPool3d`.
            Defaults to 2.
        """
        super().__init__()
        self.ndim = ndim
        self.squeeze = nn.AdaptiveAvgPool2d(1) if ndim == 2 else nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Perform the forward pass of the Squeeze-and-Excitation block.

        Applies a global average pooling (squeeze) to the input to aggregate
        spatial information into a channel descriptor. This descriptor is then
        passed through a fully connected excitation network to predict
        channel-wise attention weights. Finally, these weights are applied
        to the input feature map by channel-wise multiplication.

        Parameters
        ----------
        x : torch.Tensor
            The input feature tensor.
            Expected shape for 2D: (batch_size, channels, height, width).
            Expected shape for 3D: (batch_size, channels, depth, height, width).

        Returns
        -------
        torch.Tensor
            The recalibrated feature tensor, with the same shape as the input `x`.
        """
        bs = x.shape[0]
        c = x.shape[1]
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y)
        if self.ndim == 2:
            y = y.view(bs, c, 1, 1)
        else:
            y = y.view(bs, c, 1, 1, 1)
        return x * y.expand_as(x)


class ResConvBlock(nn.Module):
    """
    Implements a Residual Convolutional Block.

    This block is a core component often used in U-Net-like architectures to build
    encoder and decoder paths. It consists of a sequence of convolutional layers
    with a skip connection, allowing for better gradient flow and feature reuse.
    It supports optional pre-activation, Squeeze-and-Excitation blocks, and
    an initial extra convolutional layer.
    """

    def __init__(
        self,
        conv,
        in_size,
        out_size,
        k_size,
        act=None,
        norm="none",
        dropout=0,
        skip_k_size=1,
        skip_norm="none",
        first_block=False,
        se_block=False,
        extra_conv=False,
    ):
        """
        Initialize a Residual Convolutional Block.

        This block is a core component often used in U-Net-like architectures to build
        encoder and decoder paths. It consists of a sequence of convolutional layers
        with a skip connection, allowing for better gradient flow and feature reuse.
        It supports optional pre-activation, Squeeze-and-Excitation blocks, and
        an initial extra convolutional layer.

        Parameters
        ----------
        conv : Type[nn.Conv2d | nn.Conv3d]
            The convolutional layer type to use (e.g., `nn.Conv2d` for 2D, `nn.Conv3d` for 3D).
        in_size : int
            Number of input feature channels to the block.
        out_size : int
            Number of output feature channels for the convolutional layers within the block.
        k_size : int or tuple
            Kernel size for the main convolutional layers within the block.
        act : Optional[str], optional
            Activation layer to use after the first main convolution.
            If `None`, no activation is applied. Defaults to `None`.
        norm : str, optional
            Normalization layer type to use within the `ConvBlock` components.
            Options include `'bn'` (BatchNorm), `'sync_bn'` (SyncBatchNorm),
            `'in'` (InstanceNorm), `'gn'` (GroupNorm), or `'none'` (no normalization).
            Defaults to "none".
        dropout : float, optional
            Dropout value to be fixed within the `ConvBlock` components.
            If 0, no dropout is applied. Defaults to 0.
        skip_k_size : int, optional
            Kernel size for the convolution in the skip connection path.
            Used to adjust channel dimensions if `in_size` and `out_size` differ
            or to ensure correct output shape. Defaults to 1.
        skip_norm : str, optional
            Normalization layer type to use in the skip connection path.
            Options are `'bn'`, `'sync_bn'`, `'in'`, `'gn'`, or `'none'`.
            Defaults to "none".
        first_block : bool, optional
            If `True`, indicates that this is the first residual block in a sequence,
            which affects the application of Full Pre-Activation layers (normalization
            and activation are not applied before the first convolution in this case).
            Defaults to `False`.
            Reference: `Identity Mappings in Deep Residual Networks <https://arxiv.org/pdf/1603.05027.pdf>`_.
        se_block : bool, optional
            Whether to add a Squeeze-and-Excitation (SE) block at the end of the full
            residual block. Defaults to `False`.
        extra_conv : bool, optional
            If `True`, adds an additional convolutional layer with pre-activation
            before the main residual path, as described in Kisuk et al, 2017.
            Reference: `https://arxiv.org/pdf/1706.00120`. Defaults to `False`.
        """
        super(ResConvBlock, self).__init__()
        block = []
        pre_conv = []
        if not first_block:
            if not extra_conv:
                if norm != "none":
                    if conv == nn.Conv2d:
                        block.append(get_norm_2d(norm, in_size))
                    else:
                        block.append(get_norm_3d(norm, in_size))
                if act is not None:
                    block.append(get_activation(act))
            else:
                if norm != "none":
                    if conv == nn.Conv2d:
                        pre_conv.append(get_norm_2d(norm, in_size))
                    else:
                        pre_conv.append(get_norm_3d(norm, in_size))
                if act is not None:
                    pre_conv.append(get_activation(act))
        if extra_conv:
            pre_conv.append(
                ConvBlock(
                    conv=conv,
                    in_size=in_size,
                    out_size=out_size,
                    k_size=k_size,
                    act=act,
                    norm=norm,
                    dropout=dropout,
                )
            )
            in_size = out_size
            self.pre_conv = nn.Sequential(*pre_conv)
        else:
            self.pre_conv = None

        block.append(
            ConvBlock(
                conv=conv,
                in_size=in_size,
                out_size=out_size,
                k_size=k_size,
                act=act,
                norm=norm,
                dropout=dropout,
            )
        )
        block.append(ConvBlock(conv=conv, in_size=out_size, out_size=out_size, k_size=k_size))

        self.block = nn.Sequential(*block)

        if not extra_conv:
            block = []
            block.append(conv(in_size, out_size, kernel_size=skip_k_size, padding="same"))
            if skip_norm != "none":
                if conv == nn.Conv2d:
                    block.append(get_norm_2d(skip_norm, out_size))
                else:
                    block.append(get_norm_3d(skip_norm, out_size))
            self.shortcut = nn.Sequential(*block)
        else:
            self.shortcut = nn.Identity()

        if se_block:
            # add the Squeeze-and-Excitation block at the end of the full block (as in PyTC)
            # (https://github.com/zudi-lin/pytorch_connectomics/blob/master/connectomics/model/block/residual.py#L147-L155)
            self.se_block = SqExBlock(out_size, ndim=2 if conv == nn.Conv2d else 3)
        else:
            self.se_block = nn.Identity()

    def forward(self, x):
        """
        Perform the forward pass through the Residual Convolutional Block.

        Processes the input tensor through an optional pre-convolutional layer,
        then through the main convolutional blocks, and finally adds a skip
        connection. An optional Squeeze-and-Excitation block is applied at the end.

        Parameters
        ----------
        x : torch.Tensor
            The input feature tensor. Its shape should be (batch_size, in_size, D, H, W)
            or (batch_size, in_size, H, W).

        Returns
        -------
        torch.Tensor
            The output tensor after processing through the residual block.
            Its shape will be (batch_size, out_size, D', H', W') or
            (batch_size, out_size, H', W'), where D', H', W' match the input
            spatial dimensions if `padding="same"` is used.
        """
        if self.pre_conv is not None:
            x = self.pre_conv(x)
        out = self.block(x) + self.shortcut(x)
        return out


class ResUpBlock(nn.Module):
    """
    Implements a Residual Upsampling block, typically used in the decoder path of U-Net-like architectures.

    This block performs an upsampling operation on the input feature map, concatenates it
    with a corresponding skip connection (bridge) from the encoder path, and then
    processes the combined features through a `ResConvBlock`. It supports different
    upsampling modes and integrates residual connections for improved feature propagation.

    Parameters
    ----------
    ndim : int
        Number of dimensions of the input data (2 for 2D, 3 for 3D).
    convtranspose : Type[nn.ConvTranspose2d | nn.ConvTranspose3d]
        The transpose convolutional layer type to use if `up_mode` is 'convtranspose'.
    in_size : int
        Number of input channels to the upsampling operation (from the previous decoder stage).
    out_size : int
        Number of output channels for the final `ResConvBlock` in this upsampling stage.
    in_size_bridge : int
        Number of channels of the skip connection (bridge) tensor from the encoder path.
    z_down : int, optional
        Downsampling factor applied in the z-dimension for 3D data during upsampling.
        Only relevant if `ndim` is 3. Defaults to 2.
    up_mode : str
        The upsampling mode to use.
        - 'convtranspose': Uses a transpose convolution (`convtranspose`) for upsampling.
        - 'upsampling': Uses `nn.Upsample` (bilinear for 2D, trilinear for 3D) followed
                        by a 1x1 convolution.
    conv : Type[nn.Conv2d | nn.Conv3d]
        The convolutional layer type to use within the internal `ResConvBlock`.
    k_size : int or tuple
        Kernel size for the convolutional layers within the `ResConvBlock`.
    act : str, optional
        Activation function to use within the `ResConvBlock`. Defaults to `None`.
    norm : str, optional
        Normalization layer type to use within the `ResConvBlock`.
        Options include `'bn'`, `'sync_bn'`, `'in'`, `'gn'`, or `'none'`.
        Defaults to "none".
    skip_k_size : int, optional
        Kernel size for the skip connection convolution within the `ResConvBlock`.
        Used in ResUNet++. Defaults to 1.
    skip_norm : str, optional
        Normalization layer type for the skip connection within the `ResConvBlock`.
        Defaults to "none".
    dropout : float, optional
        Dropout value to be fixed within the `ResConvBlock`. Defaults to 0.
    se_block : bool, optional
        Whether to add Squeeze-and-Excitation blocks within the `ResConvBlock`.
        Defaults to `False`.
    extra_conv : bool, optional
        Whether to add an extra convolutional layer before the residual block
        within the `ResConvBlock` (as in Kisuk et al, 2017). Defaults to `False`.
    """
    
    def __init__(
        self,
        ndim,
        convtranspose,
        in_size,
        out_size,
        in_size_bridge,
        z_down,
        up_mode,
        conv,
        k_size,
        act=None,
        norm="none",
        skip_k_size=1,
        skip_norm="none",
        dropout=0,
        se_block=False,
        extra_conv=False,
    ):
        """
        Initialize a Residual Upsampling block, typically used in the decoder path of U-Net-like architectures.

        This block performs an upsampling operation on the input feature map, concatenates it
        with a corresponding skip connection (bridge) from the encoder path, and then
        processes the combined features through a `ResConvBlock`. It supports different
        upsampling modes and integrates residual connections for improved feature propagation.

        Parameters
        ----------
        ndim : int
            Number of dimensions of the input data (2 for 2D, 3 for 3D).
        convtranspose : Type[nn.ConvTranspose2d | nn.ConvTranspose3d]
            The transpose convolutional layer type to use if `up_mode` is 'convtranspose'.
        in_size : int
            Number of input channels to the upsampling operation (from the previous decoder stage).
        out_size : int
            Number of output channels for the final `ResConvBlock` in this upsampling stage.
        in_size_bridge : int
            Number of channels of the skip connection (bridge) tensor from the encoder path.
        z_down : int, optional
            Downsampling factor applied in the z-dimension for 3D data during upsampling.
            Only relevant if `ndim` is 3. Defaults to 2.
        up_mode : str
            The upsampling mode to use.
            - 'convtranspose': Uses a transpose convolution (`convtranspose`) for upsampling.
            - 'upsampling': Uses `nn.Upsample` (bilinear for 2D, trilinear for 3D) followed
                            by a 1x1 convolution.
        conv : Type[nn.Conv2d | nn.Conv3d]
            The convolutional layer type to use within the internal `ResConvBlock`.
        k_size : int or tuple
            Kernel size for the convolutional layers within the `ResConvBlock`.
        act : str, optional
            Activation function to use within the `ResConvBlock`. Defaults to `None`.
        norm : str, optional
            Normalization layer type to use within the `ResConvBlock`.
            Options include `'bn'`, `'sync_bn'`, `'in'`, `'gn'`, or `'none'`.
            Defaults to "none".
        skip_k_size : int, optional
            Kernel size for the skip connection convolution within the `ResConvBlock`.
            Used in ResUNet++. Defaults to 1.
        skip_norm : str, optional
            Normalization layer type for the skip connection within the `ResConvBlock`.
            Defaults to "none".
        dropout : float, optional
            Dropout value to be fixed within the `ResConvBlock`. Defaults to 0.
        se_block : bool, optional
            Whether to add Squeeze-and-Excitation blocks within the `ResConvBlock`.
            Defaults to `False`.
        extra_conv : bool, optional
            Whether to add an extra convolutional layer before the residual block
            within the `ResConvBlock` (as in Kisuk et al, 2017). Defaults to `False`.
        """
        super(ResUpBlock, self).__init__()
        self.ndim = ndim
        mpool = (z_down, 2, 2) if ndim == 3 else (2, 2)
        if up_mode == "convtranspose":
            self.up = convtranspose(in_size, in_size, kernel_size=mpool, stride=mpool)
        elif up_mode == "upsampling":
            self.up = nn.Upsample(mode="bilinear" if ndim == 2 else "trilinear", scale_factor=mpool)

        self.conv_block = ResConvBlock(
            conv=conv,
            in_size=in_size + in_size_bridge,
            out_size=out_size,
            k_size=k_size,
            act=act,
            norm=norm,
            dropout=dropout,
            skip_k_size=skip_k_size,
            skip_norm=skip_norm,
            se_block=se_block,
            extra_conv=extra_conv,
        )

    def forward(self, x, bridge):
        """
        Perform the forward pass of the Residual Upsampling block.

        First, it upsamples the input tensor `x`. Then, it concatenates the upsampled
        tensor with the `bridge` tensor (skip connection) along the channel dimension.
        Finally, the combined tensor is passed through a `ResConvBlock`.

        Parameters
        ----------
        x : torch.Tensor
            The input feature tensor from the previous decoder stage.
            Expected shape: (batch_size, in_size, D, H, W) or (batch_size, in_size, H, W).
        bridge : torch.Tensor
            The skip connection tensor from the corresponding encoder stage.
            Expected shape: (batch_size, in_size_bridge, D', H', W'), where D', H', W'
            match the spatial dimensions after upsampling `x`.

        Returns
        -------
        torch.Tensor
            The output tensor of the upsampling block. Its shape will be
            (batch_size, out_size, D', H', W') or (batch_size, out_size, H', W'),
            where D', H', W' are the upsampled spatial dimensions.
        """
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out


class HRBasicBlock(nn.Module):
    """
    Implements a Basic block for High-Resolution Networks (HRNet).

    This block serves as a fundamental building block in HRNet architectures,
    designed to maintain high-resolution feature representations throughout the network.
    It consists of two convolutional layers with a residual connection.

    Reference:
        `High-Resolution Representations for Labeling Pixels and Regions <https://arxiv.org/abs/1904.04514>`_.

    Parameters
    ----------
    conv : Type[nn.Conv2d | nn.Conv3d]
        The convolutional layer type to use (e.g., `nn.Conv2d` for 2D, `nn.Conv3d` for 3D).
    in_size : int
        Number of input feature channels to the block.
    out_size : int
        Number of output feature channels for the convolutional layers within the block.
        The final output channels of the block will also be `out_size` (since `expansion` is 1).
    stride : int, optional
        Stride for the first convolutional layer (`conv1_block`). Defaults to 1.
    act : Optional[nn.Module], optional
        Activation layer to apply after the first convolution (`conv1_block`).
        If `None`, no activation is applied. Defaults to `None`.
    norm : str, optional
        Normalization layer type to use within the `ConvBlock` components.
        Options include `'bn'` (BatchNorm), `'sync_bn'` (SyncBatchNorm),
        `'in'` (InstanceNorm), `'gn'` (GroupNorm), or `'none'` (no normalization).
        Defaults to "none".
    dropout : int, optional
        Dropout rate to apply within the `ConvBlock` components.
        If 0, no dropout is applied. Defaults to 0.
    downsample : Optional[nn.Module], optional
        An optional downsampling layer to apply to the residual connection if
        the input `in_size` and `out_size` do not match, or if `stride > 1`.
        Defaults to `None`.
    """

    expansion = 1

    def __init__(
        self,
        conv: Type[nn.Conv2d | nn.Conv3d],
        in_size: int,
        out_size: int,
        stride: int = 1,
        act: Optional[nn.Module] = None,
        norm: str = "none",
        dropout: int = 0,
        downsample: Optional[nn.Module] = None,
    ):
        """
        Initialize the HRBasicBlock.

        Configures two convolutional layers with optional normalization, activation,
        and dropout, along with a residual connection.

        Parameters
        ----------
        conv : Type[nn.Conv2d | nn.Conv3d]
            The convolutional layer type to use.
        in_size : int
            Number of input feature channels.
        out_size : int
            Number of output feature channels.
        stride : int, optional
            Stride for the first convolution. Defaults to 1.
        act : Optional[nn.Module], optional
            Activation layer for the first convolution. Defaults to `None`.
        norm : str, optional
            Normalization layer type. Defaults to "none".
        dropout : int, optional
            Dropout value. Defaults to 0.
        downsample : Optional[nn.Module], optional
            Downsample layer for the residual connection. Defaults to `None`.
        """
        super(HRBasicBlock, self).__init__()
        self.conv1_block = ConvBlock(
            conv=conv,
            in_size=in_size,
            out_size=out_size,
            k_size=3,
            padding=1,
            stride=stride,
            act=act,
            norm=norm,
            dropout=dropout,
            bias=False,
        )

        self.conv2_block = ConvBlock(
            conv=conv,
            in_size=out_size,
            out_size=out_size,
            k_size=3,
            padding=1,
            stride=1,
            act=None,
            norm=norm,
            dropout=dropout,
            bias=False,
        )

        self.relu_in = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        Perform the forward pass through the HRBasicBlock.

        Processes the input through two convolutional layers and adds it to a
        residual connection. An optional downsampling layer is applied to the
        residual if necessary.

        Parameters
        ----------
        x : torch.Tensor
            The input feature tensor. Its shape should be (batch_size, in_size, D, H, W)
            or (batch_size, in_size, H, W).

        Returns
        -------
        torch.Tensor
            The output tensor after processing through the basic block and
            applying the residual connection. Its shape will be
            (batch_size, out_size, D', H', W') or (batch_size, out_size, H', W'),
            where D', H', W' depend on the stride.
        """
        residual = x

        out = self.conv1_block(x)
        out = self.conv2_block(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_in(out)

        return out


class HRBottleneck(nn.Module):
    """
    Implements the Bottleneck block for High-Resolution Networks (HRNet).

    This block is a building component of HRNet architectures, designed to
    efficiently process features by reducing and then expanding the channel
    dimensions, while incorporating a residual connection. It maintains a
    high-resolution representation throughout the network.

    Reference:
        `High-Resolution Representations for Labeling Pixels and Regions <https://arxiv.org/abs/1904.04514>`_.

    Parameters
    ----------
    conv : Type[nn.Conv2d | nn.Conv3d]
        Convolutional layer to use in the residual block.

    in_size : int
        Input feature maps of the convolutional layers.

    out_size : int
        Output feature maps of the convolutional layers.

    stride : int, optional
        Stride of the convolutional layers. Default is 1.

    act : Optional[nn.Module], optional
        Activation layer to use. Default is None, which means no activation layer is applied.

    norm : str, optional
        Normalization layer (one of ``'bn'``, ``'sync_bn'`, ``'in'``, ``'gn'`` or ``'none'``). Default
        is "none".

    dropout : int, optional
        Dropout value to be fixed. Default is 0, which means no dropout is applied.

    downsample : Optional[nn.Module], optional
        Downsample layer to apply if the input and output sizes do not match. Default is None.
    """

    expansion = 4

    def __init__(
        self,
        conv: Type[nn.Conv2d | nn.Conv3d],
        in_size: int,
        out_size: int,
        stride: int = 1,
        act: Optional[nn.Module] = None,
        norm: str = "none",
        dropout: int = 0,
        downsample: Optional[nn.Module] = None,
    ):
        """
        Initialize the HRBottleneck block.

        Configures three convolutional layers (1x1, 3x3, 1x1) with optional
        normalization, activation, and dropout, along with a residual connection.

        Parameters
        ----------
        conv : Type[nn.Conv2d | nn.Conv3d]
            The convolutional layer type to use.
        in_size : int
            Number of input feature channels.
        out_size : int
            Number of output feature channels for the internal convolutions.
        stride : int, optional
            Stride for the 3x3 convolution. Defaults to 1.
        act : Optional[nn.Module], optional
            Activation layer for the final convolution. Defaults to `None`.
        norm : str, optional
            Normalization layer type. Defaults to "none".
        dropout : int, optional
            Dropout value. Defaults to 0.
        downsample : Optional[nn.Module], optional
            Downsample layer for the residual connection. Defaults to `None`.
        """
        super(HRBottleneck, self).__init__()
        self.conv1_block = ConvBlock(
            conv=conv,
            in_size=in_size,
            out_size=out_size,
            k_size=1,
            padding=0,
            stride=1,
            act=None,
            norm=norm,
            dropout=dropout,
            bias=False,
        )

        self.conv2_block = ConvBlock(
            conv=conv,
            in_size=out_size,
            out_size=out_size,
            k_size=3,
            padding=1,
            stride=stride,
            act=None,
            norm=norm,
            dropout=dropout,
            bias=False,
        )

        self.conv3_block = ConvBlock(
            conv=conv,
            in_size=out_size,
            out_size=out_size * 4,
            k_size=1,
            padding=0,
            stride=1,
            act=act,
            norm=norm,
            dropout=dropout,
            bias=False,
        )

        self.relu_in = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        Perform the forward pass through the HRBottleneck block.

        Processes the input through a sequence of three convolutional layers
        and adds it to a residual connection. An optional downsampling layer
        is applied to the residual if necessary.

        Parameters
        ----------
        x : torch.Tensor
            The input feature tensor. Its shape should be (batch_size, in_size, D, H, W)
            or (batch_size, in_size, H, W).

        Returns
        -------
        torch.Tensor
            The output tensor after processing through the bottleneck block and
            applying the residual connection. Its shape will be
            (batch_size, out_size * expansion, D', H', W') or
            (batch_size, out_size * expansion, H', W'), where D', H', W'
            depend on the stride.
        """
        residual = x

        out = self.conv1_block(x)
        out = self.conv2_block(out)
        out = self.conv3_block(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_in(out)

        return out


def get_activation(activation: str = "relu") -> nn.Module:
    """
    Get the specified activation layer.

    Parameters
    ----------
    activation : str, optional
        One of ``'relu'``, ``'tanh'``, ``'leaky_relu'``, ``'elu'``, ``'gelu'``,
        ``'silu'``, ``'sigmoid'``, ``'softmax'``,``'swish'``, 'efficient_swish'``,
        ``'linear'`` and ``'none'``.
    """
    assert activation in [
        "relu",
        "tanh",
        "leaky_relu",
        "elu",
        "gelu",
        "silu",
        "sigmoid",
        "softmax",
        "linear",
        "none",
    ], "Get unknown activation key {}".format(activation)
    activation_dict = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "leaky_relu": nn.LeakyReLU(negative_slope=0.2),
        "elu": nn.ELU(alpha=1.0),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
        "sigmoid": nn.Sigmoid(),
        "softmax": nn.Softmax(dim=1),
        "linear": nn.Identity(),
        "none": nn.Identity(),
    }
    return activation_dict[activation]


def get_norm_3d(norm: str, out_channels: int, bn_momentum: float = 0.1) -> nn.Module:
    """
    Get the specified normalization layer for a 3D model.

    Code adapted from Pytorch for Connectomics:
        `<https://github.com/zudi-lin/pytorch_connectomics/blob/6fbd5457463ae178ecd93b2946212871e9c617ee/connectomics/model/utils/misc.py#L330-L408>`_.

    Args:
        norm (str): one of ``'bn'``, ``'sync_bn'`` ``'in'``, ``'gn'`` or ``'none'``.
        out_channels (int): channel number.
        bn_momentum (float): the momentum of normalization layers.
    Returns:
        nn.Module: the normalization layer
    """
    assert norm in [
        "bn",
        "sync_bn",
        "gn",
        "in",
        "none",
    ], "Get unknown normalization layer key {}".format(norm)
    if norm == "gn":
        assert out_channels % 8 == 0, "GN requires channels to separable into 8 groups"
    selected_norm = {
        "bn": nn.BatchNorm3d,
        "sync_bn": nn.SyncBatchNorm,
        "in": nn.InstanceNorm3d,
        "gn": lambda channels: nn.GroupNorm(8, channels),
        "none": nn.Identity,
    }[norm]
    if norm in ["bn", "sync_bn", "in"]:
        return selected_norm(out_channels, momentum=bn_momentum)
    else:
        return selected_norm(out_channels)


def get_norm_2d(norm: str, out_channels: int, bn_momentum: float = 0.1) -> nn.Module:
    """
    Get the specified normalization layer for a 2D model.

    Code adapted from Pytorch for Connectomics:
        `<https://github.com/zudi-lin/pytorch_connectomics/blob/6fbd5457463ae178ecd93b2946212871e9c617ee/connectomics/model/utils/misc.py#L330-L408>`_.

    Args:
        norm (str): one of ``'bn'``, ``'sync_bn'`` ``'in'``, ``'gn'`` or ``'none'``.
        out_channels (int): channel number.
        bn_momentum (float): the momentum of normalization layers.
    Returns:
        nn.Module: the normalization layer
    """
    assert norm in [
        "bn",
        "sync_bn",
        "gn",
        "in",
        "none",
    ], "Get unknown normalization layer key {}".format(norm)
    selected_norm = {
        "bn": nn.BatchNorm2d,
        "sync_bn": nn.SyncBatchNorm,
        "in": nn.InstanceNorm2d,
        "gn": lambda channels: nn.GroupNorm(16, channels),
        "none": nn.Identity,
    }[norm]
    if norm in ["bn", "sync_bn", "in"]:
        return selected_norm(out_channels, momentum=bn_momentum)
    else:
        return selected_norm(out_channels)


class ResUNetPlusPlus_AttentionBlock(nn.Module):
    """
    Implements an attention block as used in the ResUNet++ architecture.

    This block is designed to refine skip connections in a U-Net-like architecture
    by selectively emphasizing relevant features. It combines information from
    both the encoder (downsampling path) and decoder (upsampling path) to
    generate an attention map, which is then applied to the decoder's input.

    Reference:
        Adapted from `here <https://github.com/rishikksh20/ResUnet/blob/master/core/modules.py>`_.

    Parameters
    ----------
    conv : Type[nn.Conv2d | nn.Conv3d]
        The convolutional layer type to use (e.g., `nn.Conv2d` for 2D, `nn.Conv3d` for 3D).
    maxpool : Type[nn.MaxPool2d | nn.MaxPool3d]
        The max-pooling layer type to use.
    input_encoder : int
        Number of input channels from the encoder path (larger feature map).
    input_decoder : int
        Number of input channels from the decoder path (upsampled feature map, skip connection).
    output_dim : int
        The desired number of output channels for the internal convolutional layers
        within the attention block.
    z_down : int, optional
        Downsampling factor for the z-dimension (depth) in 3D max-pooling.
        Only relevant if `conv` is `nn.Conv3d`. Defaults to 2.
    norm : str, optional
        Normalization layer type to use within the convolutional sub-blocks.
        Options include `'bn'` (BatchNorm), `'sync_bn'` (SyncBatchNorm),
        `'in'` (InstanceNorm), `'gn'` (GroupNorm), or `'none'` (no normalization).
        Defaults to "none".
    """

    def __init__(
        self,
        conv,
        maxpool,
        input_encoder,
        input_decoder,
        output_dim,
        z_down=2,
        norm="none",
    ):
        """
        Initialize the ResUNetPlusPlus_AttentionBlock.

        Sets up convolutional paths for processing encoder and decoder inputs,
        followed by a combined attention convolution.

        Parameters
        ----------
        conv : Type[nn.Conv2d | nn.Conv3d]
            The convolutional layer type to use.
        maxpool : Type[nn.MaxPool2d | nn.MaxPool3d]
            The max-pooling layer type to use.
        input_encoder : int
            Number of input channels from the encoder path.
        input_decoder : int
            Number of input channels from the decoder path (skip connection).
        output_dim : int
            The desired number of channels for the intermediate feature maps.
        z_down : int, optional
            Downsampling factor for the z-dimension in 3D max-pooling. Defaults to 2.
        norm : str, optional
            Normalization layer type to use within the convolutional blocks. Defaults to "none".
        """
        super(ResUNetPlusPlus_AttentionBlock, self).__init__()

        block = []
        if norm != "none":
            if conv == nn.Conv2d:
                block.append(get_norm_2d(norm, input_encoder))
            else:
                block.append(get_norm_3d(norm, input_encoder))
        block += [
            nn.ReLU(),
            conv(input_encoder, output_dim, 3, padding=1),
            maxpool((2, 2)) if conv == nn.Conv2d else maxpool((z_down, 2, 2)),
        ]
        self.conv_encoder = nn.Sequential(*block)

        block = []
        if norm != "none":
            if conv == nn.Conv2d:
                block.append(get_norm_2d(norm, input_decoder))
            else:
                block.append(get_norm_3d(norm, input_decoder))
        block += [nn.ReLU(), conv(input_decoder, output_dim, 3, padding=1)]
        self.conv_decoder = nn.Sequential(*block)

        block = []
        if norm != "none":
            if conv == nn.Conv2d:
                block.append(get_norm_2d(norm, output_dim))
            else:
                block.append(get_norm_3d(norm, output_dim))
        block += [nn.ReLU(), conv(output_dim, 1, 1)]
        self.conv_attn = nn.Sequential(*block)

    def forward(self, x1, x2):
        """
        Perform the forward pass of the attention block.

        It processes inputs from the encoder (`x1`) and decoder (`x2`), sums them,
        applies an attention convolution, and then multiplies the resulting attention
        map with the decoder input (`x2`).

        Parameters
        ----------
        x1 : torch.Tensor
            The input tensor from the encoder path (downsampled feature map).
            Expected shape: (batch_size, input_encoder, D, H, W) or (batch_size, input_encoder, H, W).
        x2 : torch.Tensor
            The input tensor from the decoder path (upsampled feature map, skip connection).
            Expected shape: (batch_size, input_decoder, D, H, W) or (batch_size, input_decoder, H, W).

        Returns
        -------
        torch.Tensor
            The attended output tensor, with the same shape as `x2`.
        """
        out = self.conv_encoder(x1) + self.conv_decoder(x2)
        out = self.conv_attn(out)
        return out * x2


class ASPP(nn.Module):
    """
    Implements the Atrous Spatial Pyramid Pooling (ASPP) module.

    ASPP captures multi-scale contextual information by employing parallel atrous
    convolutions with different dilation rates. This allows the model to
    effectively enlarge the receptive field and capture context at various scales.

    Reference:
        Adapted from `here <https://github.com/rishikksh20/ResUnet/blob/master/core/modules.py>`_.

    Parameters
    ----------
    conv : Type[nn.Conv2d | nn.Conv3d]
        The convolutional layer type to use (e.g., `nn.Conv2d` for 2D, `nn.Conv3d` for 3D).
    in_dims : int
        Number of input channels.
    out_dims : int
        Number of output channels for each atrous convolution block and the final output.
    norm : str, optional
        Normalization layer type to use within each atrous convolution block.
        Options include `'bn'` (BatchNorm), `'sync_bn'` (SyncBatchNorm),
        `'in'` (InstanceNorm), `'gn'` (GroupNorm), or `'none'` (no normalization).
        Defaults to "none".
    rate : list of int, optional
        A list of integers specifying the dilation rates for the parallel atrous
        convolutions. Defaults to `[6, 12, 18]`.
    """

    def __init__(self, conv, in_dims, out_dims, norm="none", rate=[6, 12, 18]):
        """
        Initialize the Atrous Spatial Pyramid Pooling (ASPP) module.

        Sets up parallel atrous convolutional blocks with different dilation rates
        and a final 1x1 convolution for combining their outputs. Each atrous block
        includes a convolution, ReLU activation, and an optional normalization layer.

        Parameters
        ----------
        conv : Type[nn.Conv2d | nn.Conv3d]
            The convolutional layer type to use (e.g., `nn.Conv2d` for 2D, `nn.Conv3d` for 3D).
        in_dims : int
            Number of input channels to the ASPP module.
        out_dims : int
            Number of output channels for each individual atrous convolution block,
            and also the final output channels of the ASPP module.
        norm : str, optional
            Normalization layer type to use after the convolution and before ReLU
            in each atrous block. Options include `'bn'`, `'sync_bn'`, `'in'`,
            `'gn'`, or `'none'`. Defaults to "none".
        rate : list of int, optional
            A list of dilation rates to be used for the parallel atrous convolutions.
            The length of this list determines the number of parallel atrous blocks.
            Defaults to `[6, 12, 18]`.
        """
        super(ASPP, self).__init__()

        block = [
            conv(in_dims, out_dims, 3, stride=1, padding=rate[0], dilation=rate[0]),
            nn.ReLU(inplace=True),
        ]
        if norm != "none":
            if conv == nn.Conv2d:
                block.append(get_norm_2d(norm, out_dims))
            else:
                block.append(get_norm_3d(norm, out_dims))
        self.aspp_block1 = nn.Sequential(*block)
        block = [
            conv(in_dims, out_dims, 3, stride=1, padding=rate[1], dilation=rate[1]),
            nn.ReLU(inplace=True),
        ]
        if norm != "none":
            if conv == nn.Conv2d:
                block.append(get_norm_2d(norm, out_dims))
            else:
                block.append(get_norm_3d(norm, out_dims))
        self.aspp_block2 = nn.Sequential(*block)
        block = [
            conv(in_dims, out_dims, 3, stride=1, padding=rate[2], dilation=rate[2]),
            nn.ReLU(inplace=True),
        ]
        if norm != "none":
            if conv == nn.Conv2d:
                block.append(get_norm_2d(norm, out_dims))
            else:
                block.append(get_norm_3d(norm, out_dims))
        self.aspp_block3 = nn.Sequential(*block)

        self.output = conv(len(rate) * out_dims, out_dims, 1)

    def forward(self, x):
        """
        Perform the forward pass of the ASPP module.

        The input tensor `x` is processed by multiple parallel atrous convolutions
        (each with a different dilation rate), and their outputs are concatenated
        along the channel dimension. A final 1x1 convolution is then applied to
        reduce the channel count to `out_dims`.

        Parameters
        ----------
        x : torch.Tensor
            The input feature tensor. Its shape should be (batch_size, in_dims, D, H, W)
            for 3D data or (batch_size, in_dims, H, W) for 2D data.

        Returns
        -------
        torch.Tensor
            The output tensor after ASPP processing. Its shape will be
            (batch_size, out_dims, D, H, W) or (batch_size, out_dims, H, W),
            depending on the input dimensionality.
        """
        x1 = self.aspp_block1(x)
        x2 = self.aspp_block2(x)
        x3 = self.aspp_block3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        return self.output(out)


class ProjectionHead(nn.Module):
    """
    Implements a projection head for self-supervised learning, designed to project input features into a lower-dimensional space and normalize the output.

    This module can configure its projection layer to be either a simple linear
    layer or a convolutional MLP (Multi-Layer Perceptron) structure, and supports
    different normalization types.

    Parameters
    ----------
    ndim : int
        Number of dimensions of the input data. Supports 2 (for 2D data) or 3 (for 3D data).
    in_channels : int
        Number of input feature channels.
    proj_dim : int, optional
        The desired dimension of the projected output features. Defaults to 256.
    proj : str, optional
        Specifies the type of projection layer to use.
        - 'linear': Uses a single 1x1 convolutional layer (equivalent to a linear projection).
        - 'convmlp': Employs a convolutional MLP structure, consisting of a 1x1 convolution,
                     batch normalization, ReLU activation, and another 1x1 convolution.
        Defaults to 'convmlp'.
    bn_type : str, optional
        Defines the type of batch normalization to apply within the 'convmlp' projection.
        - 'sync_bn': Synchronized Batch Normalization.
        - 'none': No batch normalization is applied.
        Defaults to 'sync_bn'.
    """

    def __init__(self, ndim, in_channels, proj_dim=256, proj="convmlp", bn_type="sync_bn"):
        """
        Initialize the ProjectionHead module with specified dimensions, input channels, projection type, and normalization settings.

        The appropriate convolutional and normalization functions (2D or 3D) are selected
        based on the `ndim` parameter.

        Parameters
        ----------
        ndim : int
            Number of dimensions of the input data (2 for 2D, 3 for 3D).
        in_channels : int
            Number of input channels for the projection head.
        proj_dim : int, optional
            Dimension of the projected output. Defaults to 256.
        proj : str, optional
            Type of projection to use. Options are 'linear' or 'convmlp'.
            'linear' uses a simple 1x1 convolution. 'convmlp' uses a sequence of
            convolution, batch normalization, ReLU, and another convolution.
            Defaults to 'convmlp'.
        bn_type : str, optional
            Type of batch normalization to use if `proj` is 'convmlp'.
            Options are 'sync_bn' or 'none'. Defaults to 'sync_bn'.
        """
        super(ProjectionHead, self).__init__()
        self.ndim = ndim
        if self.ndim == 3:
            self.conv_call = nn.Conv3d
            self.norm_func = get_norm_3d
        else:
            self.conv_call = nn.Conv2d
            self.norm_func = get_norm_2d

        if proj == "linear":
            self.proj = self.conv_call(in_channels, proj_dim, kernel_size=1)
        elif proj == "convmlp":
            self.proj = nn.Sequential(
                self.conv_call(in_channels, in_channels, kernel_size=1),
                self.norm_func(bn_type, in_channels),
                nn.ReLU(inplace=True),
                self.conv_call(in_channels, proj_dim, kernel_size=1),
            )

    def forward(self, x):
        """
        Perform the forward pass through the projection head.

        The input tensor `x` is first passed through the configured projection layer,
        and then the output is L2-normalized along the channel dimension.

        Parameters
        ----------
        x : torch.Tensor
            The input feature tensor. Its shape should be (batch_size, in_channels, D, H, W)
            for 3D data or (batch_size, in_channels, H, W) for 2D data.

        Returns
        -------
        torch.Tensor
            The L2-normalized projected output tensor. Its shape will be
            (batch_size, proj_dim, D, H, W) or (batch_size, proj_dim, H, W),
            depending on `ndim`.
        """
        return F.normalize(self.proj(x), p=2, dim=1)
