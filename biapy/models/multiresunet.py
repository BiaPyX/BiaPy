"""
This module implements the MultiResUNet architecture, a U-Net variant designed for multimodal biomedical image segmentation.

MultiResUNet enhances the standard U-Net by incorporating "MultiRes Blocks"
in the encoder and decoder paths, and "ResPaths" for skip connections.
These components aim to improve feature representation and information flow
across different scales.

Key components implemented in this file include:

Classes:

- ``Conv_batchnorm``: A basic convolutional block with optional batch normalization and activation.
- ``Multiresblock``: The MultiRes Block, a core component that processes features
  through parallel convolutional paths of different kernel sizes (3x3, 5x5, 7x7)
  and fuses them.
- ``Respath``: The ResPath module, which acts as an enhanced skip connection,
  applying residual convolutional blocks to features before they are concatenated
  in the decoder.
- ``MultiResUnet``: The main MultiResUNet model, combining the encoder, decoder,
  and skip connections using the MultiRes Blocks and ResPaths.

The implementation supports both 2D and 3D inputs, various normalization types,
and optional multi-head outputs, including a contrastive learning projection.

Reference:
`MultiResUNet : Rethinking the U-Net Architecture for Multimodal Biomedical Image
Segmentation <https://arxiv.org/abs/1902.04049>`_

Code Adapted From:
https://github.com/nibtehaz/MultiResUNet
"""
import torch
import torch.nn as nn
from typing import Dict, List

from biapy.models.heads import ProjectionHead
from biapy.models.blocks import prepare_activation_layers


class Conv_batchnorm(torch.nn.Module):
    """
    A basic convolutional block with optional batch normalization and activation.

    This module combines a convolutional layer, a batch normalization layer,
    and an activation function (ReLU by default), providing a standard building
    block for neural networks.

    Parameters
    ----------
    conv : Torch conv layer
        Convolutional layer type to use (e.g., `nn.Conv2d`, `nn.Conv3d`).

    batchnorm : Torch batch normalization layer
        Batch normalization layer type to use (e.g., `nn.BatchNorm2d`, `nn.BatchNorm3d`).

    num_in_filters : int
        Number of input channels for the convolutional layer.

    num_out_filters : int
        Number of output channels for the convolutional layer.

    kernel_size : Tuple of ints
        Size of the convolving kernel (e.g., 3 for 3x3, (3,3,3) for 3x3x3).

    stride : Tuple of ints, optional
        Stride of the convolution. Defaults to 1.

    activation : str, optional
        Activation function to apply after convolution and batch normalization.
        Currently supports "relu" or "None" (for no activation). Defaults to "relu".
    """

    def __init__(
        self,
        conv,
        batchnorm,
        num_in_filters,
        num_out_filters,
        kernel_size,
        stride=1,
        activation="relu",
    ):
        """
        Initialize the Conv_batchnorm block.

        Sets up the convolutional layer, batch normalization layer, and
        stores the chosen activation function type.

        Parameters
        ----------
        conv : Type[nn.Conv2d | nn.Conv3d]
            The convolutional layer type to use (e.g., `nn.Conv2d` for 2D, `nn.Conv3d` for 3D).
        batchnorm : Type[nn.BatchNorm2d | nn.BatchNorm3d]
            The batch normalization layer type to use (e.g., `nn.BatchNorm2d`, `nn.BatchNorm3d`).
        num_in_filters : int
            The number of input channels for the convolutional layer.
        num_out_filters : int
            The number of output channels for the convolutional layer.
        kernel_size : int | Tuple[int, ...]
            The size of the convolving kernel. Can be a single integer for square/cubic kernels
            or a tuple for specific dimensions.
        stride : int | Tuple[int, ...], optional
            The stride of the convolution. Can be a single integer or a tuple. Defaults to 1.
        activation : str, optional
            The name of the activation function to apply after convolution and batch normalization.
            Currently supports "relu" or "None" (for no activation). Defaults to "relu".
        """
        super().__init__()
        
        self.activation = activation
        self.conv1 = conv(
            in_channels=num_in_filters,
            out_channels=num_out_filters,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
        )
        self.batchnorm = batchnorm(num_out_filters)

    def forward(self, x):
        """
        Perform the forward pass of the convolutional block.

        Applies convolution, batch normalization, and then the specified
        activation function to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to the block.

        Returns
        -------
        torch.Tensor
            The output tensor after convolution, batch normalization, and activation.
        """
        x = self.conv1(x)
        x = self.batchnorm(x)

        if self.activation == "relu":
            return torch.nn.functional.relu(x)
        else:
            return x


class Multiresblock(torch.nn.Module):
    """
    MultiRes Block as described in the MultiResUNet paper.

    This block enhances feature extraction by processing input through parallel
    convolutional paths with different effective receptive fields (3x3, 5x5, 7x7)
    and then concatenating their outputs. It also includes a shortcut connection
    and batch normalization.

    Parameters
    ----------
    conv : Torch conv layer
        Convolutional layer type to use (e.g., `nn.Conv2d`, `nn.Conv3d`).

    batchnorm : Torch batch normalization layer
        Batch normalization layer type to use (e.g., `nn.BatchNorm2d`, `nn.BatchNorm3d`).

    num_in_channels : int
        Number of input channels coming into the MultiRes Block.

    num_filters : int
        Base number of filters for calculating the output filter counts for
        the internal convolutional paths.

    alpha : float, optional
        Alpha hyperparameter (default: 1.67). Used to scale the total number
        of filters, influencing the capacity of the block.
    """

    def __init__(self, conv, batchnorm, num_in_channels, num_filters, alpha=1.67):
        """
        Initialize the MultiRes Block.

        Calculates the number of filters for each parallel convolutional path
        (3x3, 5x5, 7x7) based on `num_filters` and `alpha`. It then sets up
        these parallel paths using `Conv_batchnorm` blocks, along with a
        shortcut connection and final batch normalization layers.

        Parameters
        ----------
        conv : Type[nn.Conv2d | nn.Conv3d]
            The convolutional layer type to use.
        batchnorm : Type[nn.BatchNorm2d | nn.BatchNorm3d]
            The batch normalization layer type to use.
        num_in_channels : int
            The number of input channels for the MultiRes Block.
        num_filters : int
            The base number of filters used to determine the output channel counts
            for the internal convolutional paths.
        alpha : float, optional
            The scaling factor for the total number of filters (`W`). Defaults to 1.67.
        """
        super().__init__()
        self.alpha = alpha
        self.W = num_filters * alpha

        filt_cnt_3x3 = int(self.W * 0.167)
        filt_cnt_5x5 = int(self.W * 0.333)
        filt_cnt_7x7 = int(self.W * 0.5)
        num_out_filters = filt_cnt_3x3 + filt_cnt_5x5 + filt_cnt_7x7

        self.shortcut = Conv_batchnorm(
            conv,
            batchnorm,
            num_in_channels,
            num_out_filters,
            kernel_size=1,
            activation="None",
        )

        self.conv_3x3 = Conv_batchnorm(
            conv,
            batchnorm,
            num_in_channels,
            filt_cnt_3x3,
            kernel_size=3,
            activation="relu",
        )

        self.conv_5x5 = Conv_batchnorm(
            conv,
            batchnorm,
            filt_cnt_3x3,
            filt_cnt_5x5,
            kernel_size=3,
            activation="relu",
        )

        self.conv_7x7 = Conv_batchnorm(
            conv,
            batchnorm,
            filt_cnt_5x5,
            filt_cnt_7x7,
            kernel_size=3,
            activation="relu",
        )

        self.batch_norm1 = batchnorm(num_out_filters)
        self.batch_norm2 = batchnorm(num_out_filters)

    def forward(self, x):
        """
        Perform the forward pass of the MultiRes Block.

        The input `x` first goes through a shortcut connection. Simultaneously,
        it passes through three sequential convolutional paths (3x3, 5x5, 7x7).
        The outputs of these paths are concatenated, batch normalized, and then
        added to the shortcut output. A final batch normalization and ReLU
        activation are applied.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to the MultiRes Block.

        Returns
        -------
        torch.Tensor
            The output tensor of the MultiRes Block.
        """
        shrtct = self.shortcut(x)

        a = self.conv_3x3(x)
        b = self.conv_5x5(a)
        c = self.conv_7x7(b)

        x = torch.cat([a, b, c], dim=1)
        x = self.batch_norm1(x)

        x = x + shrtct
        x = self.batch_norm2(x)
        x = torch.nn.functional.relu(x)

        return x


class Respath(torch.nn.Module):
    """
    ResPath module for MultiResUNet.

    ResPath acts as an enhanced skip connection by applying a series of
    residual convolutional blocks to the features before they are
    concatenated into the decoder path. This helps to reduce the semantic
    gap between encoder and decoder features.

    Parameters
    ----------
    conv : Torch conv layer
        Convolutional layer type to use (e.g., `nn.Conv2d`, `nn.Conv3d`).

    batchnorm : Torch batch normalization layer
        Batch normalization layer type to use (e.g., `nn.BatchNorm2d`, `nn.BatchNorm3d`).

    num_in_filters : int
        Number of input channels coming into the ResPath.

    num_out_filters : int
        Number of output channels for each convolutional block within the ResPath.

    respath_length : int
        The number of residual convolutional blocks to stack in the ResPath.
    """

    def __init__(self, conv, batchnorm, num_in_filters, num_out_filters, respath_length):
        """
        Initialize the ResPath module.

        Sets up a sequence of `respath_length` residual convolutional blocks.
        Each block consists of a convolutional layer and a shortcut connection,
        followed by batch normalization and ReLU. The input and output filter
        counts are managed across these blocks.

        Parameters
        ----------
        conv : Type[nn.Conv2d | nn.Conv3d]
            The convolutional layer type to use.
        batchnorm : Type[nn.BatchNorm2d | nn.BatchNorm3d]
            The batch normalization layer type to use.
        num_in_filters : int
            The number of input channels for the first block in the ResPath.
        num_out_filters : int
            The number of output channels for each convolutional block within the ResPath.
        respath_length : int
            The number of residual convolutional blocks to stack in this ResPath.
        """
        super().__init__()

        self.respath_length = respath_length
        self.shortcuts = torch.nn.ModuleList([])
        self.convs = torch.nn.ModuleList([])
        self.bns = torch.nn.ModuleList([])

        for i in range(self.respath_length):
            if i == 0:
                self.shortcuts.append(
                    Conv_batchnorm(
                        conv,
                        batchnorm,
                        num_in_filters,
                        num_out_filters,
                        kernel_size=1,
                        activation="None",
                    )
                )
                self.convs.append(
                    Conv_batchnorm(
                        conv,
                        batchnorm,
                        num_in_filters,
                        num_out_filters,
                        kernel_size=3,
                        activation="relu",
                    )
                )
            else:
                self.shortcuts.append(
                    Conv_batchnorm(
                        conv,
                        batchnorm,
                        num_out_filters,
                        num_out_filters,
                        kernel_size=1,
                        activation="None",
                    )
                )
                self.convs.append(
                    Conv_batchnorm(
                        conv,
                        batchnorm,
                        num_out_filters,
                        num_out_filters,
                        kernel_size=3,
                        activation="relu",
                    )
                )

            self.bns.append(batchnorm(num_out_filters))

    def forward(self, x):
        """
        Perform the forward pass of the ResPath.

        The input `x` passes through a sequence of `respath_length` residual
        convolutional blocks. Each block involves a convolutional path and
        a shortcut connection, followed by batch normalization and ReLU activation.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to the ResPath.

        Returns
        -------
        torch.Tensor
            The output tensor after processing through all residual blocks in the ResPath.
        """
        for short, conv, bn in zip(self.shortcuts, self.convs, self.bns):

            shortcut = short(x)

            x = conv(x)
            x = bn(x)
            x = torch.nn.functional.relu(x)

            x = x + shortcut
            x = bn(x)
            x = torch.nn.functional.relu(x)

        return x


class MultiResUnet(torch.nn.Module):
    def __init__(
        self,
        ndim,
        input_channels,
        alpha=1.67,
        z_down=[2, 2, 2, 2],
        output_channels=[1],
        upsampling_factor=(),
        upsampling_position="pre",
        contrast: bool = False,
        contrast_proj_dim: int = 256,
        explicit_activations: bool = False,
        activations: List[List[str]] = [],
    ):
        """
        Create 2D/3D MultiResUNet model.

        Reference: `MultiResUNet : Rethinking the U-Net Architecture for Multimodal Biomedical Image
        Segmentation <https://arxiv.org/abs/1902.04049>`_.

        Parameters
        ----------
        ndim : int
            Number of dimensions of the input data.

        input_channels: int
            Number of channels in image.

        alpha: float, optional
            Alpha hyperparameter (default: 1.67)

        z_down : List of ints, optional
            Downsampling used in z dimension. Set it to ``1`` if the dataset is not isotropic.

        output_channels : list of int, optional
            Output channels of the network. It must be a list of lenght ``1`` or ``2``. When two
            numbers are provided two task to be done is expected (multi-head). Possible scenarios are:
            
                * instances + classification on instance segmentation
                * points + classification in detection.

        upsampling_factor : tuple of ints, optional
            Factor of upsampling for super resolution workflow for each dimension.

        upsampling_position : str, optional
            Whether the upsampling is going to be made previously (``pre`` option) to the model
            or after the model (``post`` option).
        
        contrast : bool, optional
            Whether to add contrastive learning head to the model. Default is ``False``.

        contrast_proj_dim : int, optional
            Dimension of the projection head for contrastive learning. Default is ``256``.

        explicit_activations : bool, optional
            If True, uses explicit activation functions in the last layers. 
        
        activations : List[List[str]], optional
            Activation functions to apply to the outputs if `explicit_activations` is True.

        Raises
        ------
        ValueError
            If 'output_channels' is empty or has more than two values.
        """
        super().__init__()
        self.ndim = ndim
        self.alpha = alpha
        self.output_channels = output_channels
        self.multihead = len(output_channels) == 2
        self.contrast = contrast
        self.explicit_activations = explicit_activations
        if self.explicit_activations:
            self.out_activations, self.class_activation = prepare_activation_layers(activations)
        if len(output_channels) == 0:
            raise ValueError("'output_channels' needs to has at least one value")
        if len(output_channels) != 1 and len(output_channels) != 2:
            raise ValueError(f"'output_channels' must be a list of one or two values at max, not {output_channels}")

        if self.ndim == 3:
            conv = nn.Conv3d
            convtranspose = nn.ConvTranspose3d
            batchnorm_layer = nn.BatchNorm3d
            pooling = nn.MaxPool3d
            dropout = nn.Dropout3d
        else:
            conv = nn.Conv2d
            convtranspose = nn.ConvTranspose2d
            batchnorm_layer = nn.BatchNorm2d
            pooling = nn.MaxPool2d
            dropout = nn.Dropout2d

        # Super-resolution
        self.pre_upsampling = None
        if len(upsampling_factor) > 1 and upsampling_position == "pre":
            self.pre_upsampling = convtranspose(
                input_channels,
                input_channels,
                kernel_size=upsampling_factor,
                stride=upsampling_factor,
            )

        # Encoder Path
        self.multiresblock1 = Multiresblock(conv, batchnorm_layer, input_channels, 32)
        self.in_filters1 = int(32 * self.alpha * 0.167) + int(32 * self.alpha * 0.333) + int(32 * self.alpha * 0.5)
        mpool = (z_down[0], 2, 2) if self.ndim == 3 else (2, 2)
        self.pool1 = pooling(mpool)
        self.respath1 = Respath(conv, batchnorm_layer, self.in_filters1, 32, respath_length=4)

        self.multiresblock2 = Multiresblock(conv, batchnorm_layer, self.in_filters1, 32 * 2)
        self.in_filters2 = (
            int(32 * 2 * self.alpha * 0.167) + int(32 * 2 * self.alpha * 0.333) + int(32 * 2 * self.alpha * 0.5)
        )
        mpool = (z_down[1], 2, 2) if self.ndim == 3 else (2, 2)
        self.pool2 = pooling(mpool)
        self.respath2 = Respath(conv, batchnorm_layer, self.in_filters2, 32 * 2, respath_length=3)

        self.multiresblock3 = Multiresblock(conv, batchnorm_layer, self.in_filters2, 32 * 4)
        self.in_filters3 = (
            int(32 * 4 * self.alpha * 0.167) + int(32 * 4 * self.alpha * 0.333) + int(32 * 4 * self.alpha * 0.5)
        )
        mpool = (z_down[2], 2, 2) if self.ndim == 3 else (2, 2)
        self.pool3 = pooling(mpool)
        self.respath3 = Respath(conv, batchnorm_layer, self.in_filters3, 32 * 4, respath_length=2)

        self.multiresblock4 = Multiresblock(conv, batchnorm_layer, self.in_filters3, 32 * 8)
        self.in_filters4 = (
            int(32 * 8 * self.alpha * 0.167) + int(32 * 8 * self.alpha * 0.333) + int(32 * 8 * self.alpha * 0.5)
        )
        mpool = (z_down[3], 2, 2) if self.ndim == 3 else (2, 2)
        self.pool4 = pooling(mpool)
        self.respath4 = Respath(conv, batchnorm_layer, self.in_filters4, 32 * 8, respath_length=1)

        self.multiresblock5 = Multiresblock(conv, batchnorm_layer, self.in_filters4, 32 * 16)
        self.in_filters5 = (
            int(32 * 16 * self.alpha * 0.167) + int(32 * 16 * self.alpha * 0.333) + int(32 * 16 * self.alpha * 0.5)
        )

        # Decoder path
        mpool = (z_down[3], 2, 2) if self.ndim == 3 else (2, 2)
        self.upsample6 = convtranspose(self.in_filters5, 32 * 8, kernel_size=mpool, stride=mpool)
        self.concat_filters1 = 32 * 8 * 2
        self.multiresblock6 = Multiresblock(conv, batchnorm_layer, self.concat_filters1, 32 * 8)
        self.in_filters6 = (
            int(32 * 8 * self.alpha * 0.167) + int(32 * 8 * self.alpha * 0.333) + int(32 * 8 * self.alpha * 0.5)
        )

        mpool = (z_down[2], 2, 2) if self.ndim == 3 else (2, 2)
        self.upsample7 = convtranspose(self.in_filters6, 32 * 4, kernel_size=mpool, stride=mpool)
        self.concat_filters2 = 32 * 4 * 2
        self.multiresblock7 = Multiresblock(conv, batchnorm_layer, self.concat_filters2, 32 * 4)
        self.in_filters7 = (
            int(32 * 4 * self.alpha * 0.167) + int(32 * 4 * self.alpha * 0.333) + int(32 * 4 * self.alpha * 0.5)
        )

        mpool = (z_down[1], 2, 2) if self.ndim == 3 else (2, 2)
        self.upsample8 = convtranspose(self.in_filters7, 32 * 2, kernel_size=mpool, stride=mpool)
        self.concat_filters3 = 32 * 2 * 2
        self.multiresblock8 = Multiresblock(conv, batchnorm_layer, self.concat_filters3, 32 * 2)
        self.in_filters8 = (
            int(32 * 2 * self.alpha * 0.167) + int(32 * 2 * self.alpha * 0.333) + int(32 * 2 * self.alpha * 0.5)
        )

        mpool = (z_down[0], 2, 2) if self.ndim == 3 else (2, 2)
        self.upsample9 = convtranspose(self.in_filters8, 32, kernel_size=mpool, stride=mpool)
        self.concat_filters4 = 32 * 2
        self.multiresblock9 = Multiresblock(conv, batchnorm_layer, self.concat_filters4, 32)
        self.in_filters9 = int(32 * self.alpha * 0.167) + int(32 * self.alpha * 0.333) + int(32 * self.alpha * 0.5)

        # Super-resolution
        self.post_upsampling = None
        if len(upsampling_factor) > 1 and upsampling_position == "post":
            self.post_upsampling = convtranspose(
                self.in_filters9,
                self.in_filters9,
                kernel_size=upsampling_factor,
                stride=upsampling_factor,
            )

        if self.contrast:
            # extra added layers
            self.last_block = nn.Sequential(
                conv(self.in_filters9, self.in_filters9, kernel_size=3, stride=1, padding=1),
                batchnorm_layer,
                dropout(0.10),
                conv(self.in_filters9, output_channels[0], kernel_size=1, stride=1, padding=0, bias=False),
            )

            self.proj_head = ProjectionHead(ndim=self.ndim, in_channels=self.in_filters9, proj_dim=contrast_proj_dim)
        else:
            self.last_block = Conv_batchnorm(
                conv,
                batchnorm_layer,
                self.in_filters9,
                output_channels[0],
                kernel_size=1,
                activation="None",
            )

        # Multi-head:
        #   Instance segmentation: instances + classification
        #   Detection: points + classification
        self.last_class_head = None
        if self.multihead:
            self.last_class_head = conv(self.in_filters9, output_channels[1], kernel_size=1, padding="same")

    def forward(self, x: torch.Tensor) -> Dict | torch.Tensor:
        """
        Perform the forward pass of the ResPath.

        The input `x` passes through a sequence of `respath_length` residual
        convolutional blocks. Each block involves a convolutional path and
        a shortcut connection, followed by batch normalization and ReLU activation.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to the ResPath.

        Returns
        -------
        torch.Tensor
            The output tensor after processing through all residual blocks in the ResPath.
        """
        # Super-resolution
        if self.pre_upsampling:
            x = self.pre_upsampling(x)

        x_multires1 = self.multiresblock1(x)
        x_pool1 = self.pool1(x_multires1)
        x_multires1 = self.respath1(x_multires1)

        x_multires2 = self.multiresblock2(x_pool1)
        x_pool2 = self.pool2(x_multires2)
        x_multires2 = self.respath2(x_multires2)

        x_multires3 = self.multiresblock3(x_pool2)
        x_pool3 = self.pool3(x_multires3)
        x_multires3 = self.respath3(x_multires3)

        x_multires4 = self.multiresblock4(x_pool3)
        x_pool4 = self.pool4(x_multires4)
        x_multires4 = self.respath4(x_multires4)

        x_multires5 = self.multiresblock5(x_pool4)

        up6 = torch.cat([self.upsample6(x_multires5), x_multires4], dim=1)
        x_multires6 = self.multiresblock6(up6)

        up7 = torch.cat([self.upsample7(x_multires6), x_multires3], dim=1)
        x_multires7 = self.multiresblock7(up7)

        up8 = torch.cat([self.upsample8(x_multires7), x_multires2], dim=1)
        x_multires8 = self.multiresblock8(up8)

        up9 = torch.cat([self.upsample9(x_multires8), x_multires1], dim=1)
        x_multires9 = self.multiresblock9(up9)

        feats = x_multires9
        # Super-resolution
        if self.post_upsampling:
            feats = self.post_upsampling(feats)

        # Regular output
        out = self.last_block(feats)

        if self.explicit_activations:
            # If there is only one activation, apply it to the whole tensor
            if len(self.out_activations) == 1:
                out = self.out_activations[0](out)
            else:
                for i, act in enumerate(self.out_activations):
                    out[:, i:i+1] = act(out[:, i:i+1])

        out_dict = {
            "pred": out,
        }

        # Contrastive learning head
        if self.contrast:
            out_dict["embed"] = self.proj_head(feats)

        # Multi-head output
        #   Instance segmentation: instances + classification
        #   Detection: points + classification
        if self.multihead and self.last_class_head:
            class_head_out = self.last_class_head(feats)
            if self.explicit_activations:
                for i, act in enumerate(self.class_activation):
                    class_head_out[:, i:i+1] = act(class_head_out[:, i:i+1])
            out_dict["class"] = class_head_out

        if len(out_dict.keys()) == 1:
            return out_dict["pred"]
        else:
            return out_dict
    