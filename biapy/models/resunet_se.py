"""
This module implements the Residual U-Net with Squeeze-and-Excitation (SE) blocks, referred to as ResUNet_SE.

This architecture is a variant of the U-Net, enhanced with residual connections and channel-wise attention mechanisms to improve feature learning and representation for various image analysis tasks.

The ResUNet_SE is designed for both 2D and 3D image processing, supporting
different normalization types, kernel sizes, and flexible upsampling strategies.
It can be configured for multi-task learning with multiple output heads,
including an optional contrastive learning projection.

Key components and functionalities include:

Classes:

- ``ResUNet_SE``: The main Residual U-Net model with Squeeze-and-Excitation blocks.

This module relies on building blocks defined in `biapy.models.blocks`, such as
`ResConvBlock`, `ResUpBlock`, `ConvBlock`, `ProjectionHead`, and normalization helpers.

Reference:
`Squeeze and Excitation Networks <https://arxiv.org/abs/1709.01507>`_.

Image representation:

.. image:: ../../img/models/unet.png
    :width: 100%
    :align: center

Image created with `PlotNeuralNet <https://github.com/HarisIqbal88/PlotNeuralNet>`_.
"""
import torch
import torch.nn as nn
from typing import Dict 

from biapy.models.blocks import (
    ResConvBlock,
    ResUpBlock,
    ConvBlock,
    get_norm_2d, 
    get_norm_3d
)
from biapy.models.heads import ProjectionHead


class ResUNet_SE(nn.Module):
    """
    Create Residual 2D/3D U-Net with Squeeze-and-Excitation (SE) blocks.

    This model extends the classic U-Net architecture by incorporating
    residual connections within its convolutional blocks and Squeeze-and-Excitation
    modules for channel-wise attention. This design aims to improve feature
    learning and propagation, leading to better performance in dense prediction
    tasks like image segmentation.

    Reference: `Squeeze and Excitation Networks <https://arxiv.org/abs/1709.01507>`_.

    Parameters
    ----------
    image_shape : Tuple[int, ...]
        Dimensions of the input image. E.g., `(y, x, channels)` for 2D or
        `(z, y, x, channels)` for 3D. The last element `image_shape[-1]`
        should be the number of input channels.

    activation : str, optional
        Activation layer to use throughout the network (e.g., "ELU", "ReLU").
        Defaults to "ELU".

    feature_maps : List[int], optional
        A list specifying the number of feature maps (channels) at each level
        of the U-Net. The length of this list defines the depth of the U-Net.
        Defaults to `[32, 64, 128, 256]`.

    drop_values : List[float], optional
        A list of dropout probabilities to apply at each level of the U-Net.
        Its length should match the number of levels (i.e., `len(feature_maps)`).
        Defaults to `[0.1, 0.1, 0.1, 0.1]`.

    normalization : str, optional
        Type of normalization layer to use throughout the network. Options include
        `'bn'` (Batch Normalization), `'sync_bn'` (Synchronized Batch Normalization for multi-GPU),
        `'in'` (Instance Normalization), `'gn'` (Group Normalization), or `'none'`.
        Defaults to "none".

    k_size : int, optional
        Kernel size for most convolutional layers in the network. Defaults to 3.

    upsample_layer : str, optional
        Type of layer to use for upsampling in the decoder path.
        Two options: "convtranspose" (using `nn.ConvTranspose2d`/`3d`) or
        "upsampling" (using `nn.Upsample` followed by convolution).
        Defaults to "convtranspose".

    z_down : List[int], optional
        For 3D data, a list of downsampling factors for the z-dimension at each
        pooling stage in the encoder. Set elements to `1` if the dataset is not
        isotropic and z-downsampling is not desired at that stage.
        Its length should match the number of pooling stages (`len(feature_maps) - 1`).
        Defaults to `[2, 2, 2, 2]`.

    output_channels : List[int], optional
        Specifies the number of output channels for the final prediction head(s).
        Must be a list of length 1 for a single output task (e.g., semantic segmentation)
        or length 2 for multi-head tasks (e.g., instances + classification in instance segmentation,
        or points + classification in detection). Defaults to `[1]`.

    upsampling_factor : Tuple[int, ...], optional
        Factor of upsampling for super-resolution workflows. If provided,
        it dictates the kernel and stride for an initial or final transposed
        convolution. Defaults to an empty tuple `()`, meaning no super-resolution.

    upsampling_position : str, optional
        Determines where super-resolution upsampling is applied:

        - ``"pre"``: Upsampling is performed *before* the main U-Net model.
        - ``"post"``: Upsampling is performed *after* the main U-Net model.
        
        Defaults to "pre".

    isotropy : bool or List[bool], optional
        Controls whether to use 3D or 2D convolutions at each U-Net level when
        the input is 3D.

        - If `True` (bool), all levels use 3D convolutions.
        - If `False` (bool), all levels use 2D convolutions (1xKxK kernels for 3D input).
        - If `List[bool]`, specifies for each level (encoder/decoder pair) whether
          to use 3D (True) or 2D (False) kernels. Its length should match `len(feature_maps)`.
        
        Defaults to False.

    larger_io : bool, optional
        If True, uses extra and larger kernels (k_size+2) in the input and
        output layers for potentially better initial/final feature extraction.
        Defaults to True.

    extra_conv : bool, optional
        If True, adds an additional convolutional layer before the residual
        blocks in the encoder and decoder, as seen in some U-Net variants
        (e.g., Kisuk et al., 2017). Defaults to True.

    contrast : bool, optional
        Whether to add a contrastive learning projection head to the model.
        If True, an additional output `embed` will be available in the forward pass.
        Defaults to `False`.

    contrast_proj_dim : int, optional
        Dimension of the projection head for contrastive learning, if `contrast` is True.
        Defaults to `256`.

    Returns
    -------
    model : nn.Module
        The constructed ResUNet_SE model.

    Calling this function with its default parameters returns the following network:

    .. image:: ../../img/models/unet.png
        :width: 100%
        :align: center

    Image created with `PlotNeuralNet <https://github.com/HarisIqbal88/PlotNeuralNet>`_.
    """

    def __init__(
        self,
        image_shape=(256, 256, 1),
        activation="ELU",
        feature_maps=[32, 64, 128, 256],
        drop_values=[0.1, 0.1, 0.1, 0.1],
        normalization="none",
        k_size=3,
        upsample_layer="convtranspose",
        z_down=[2, 2, 2, 2],
        output_channels=[1],
        upsampling_factor=(),
        upsampling_position="pre",
        isotropy=False,
        larger_io=True,
        extra_conv=True,
        contrast: bool = False,
        contrast_proj_dim: int = 256,
    ):
        """
        Initialize the ResUNet_SE model.

        Sets up the encoder (downsampling path), decoder (upsampling path),
        bottleneck, and optional super-resolution and multi-head output layers.
        It dynamically selects 2D or 3D convolutional, pooling, and normalization
        layers based on `ndim` and `isotropy` settings.

        Parameters
        ----------
        image_shape : Tuple[int, ...], optional
            Input image dimensions. Defaults to (256, 256, 1).
        activation : str, optional
            Activation function name. Defaults to "ELU".
        feature_maps : List[int], optional
            Number of feature maps at each U-Net level. Defaults to `[32, 64, 128, 256]`.
        drop_values : List[float], optional
            Dropout probabilities for each level. Defaults to `[0.1, 0.1, 0.1, 0.1]`.
        normalization : str, optional
            Normalization layer type. Defaults to "none".
        k_size : int, optional
            Kernel size for convolutions. Defaults to 3.
        upsample_layer : str, optional
            Upsampling method ("convtranspose" or "upsampling"). Defaults to "convtranspose".
        z_down : List[int], optional
            Z-dimension downsampling factors for 3D data. Defaults to `[2, 2, 2, 2]`.
        output_channels : List[int], optional
            Number of channels for the output head(s). Can be length 1 or 2.
            Defaults to `[1]`.
        upsampling_factor : Tuple[int, ...], optional
            Factor for super-resolution upsampling. Defaults to `()`.
        upsampling_position : str, optional
            Position of super-resolution upsampling ("pre" or "post"). Defaults to "pre".
        isotropy : bool | List[bool], optional
            Controls 3D vs 2D convolutions for 3D input. Defaults to False.
        larger_io : bool, optional
            Whether to use larger kernels in input/output layers. Defaults to True.
        extra_conv : bool, optional
            Whether to add an extra conv layer before residual blocks. Defaults to True.
        contrast : bool, optional
            Whether to add a contrastive learning projection head. Defaults to `False`.
        contrast_proj_dim : int, optional
            Dimension of the contrastive projection head. Defaults to `256`.

        Raises
        ------
        ValueError
            If 'output_channels' is empty or has more than two values.
        """
        super(ResUNet_SE, self).__init__()

        if len(output_channels) == 0:
            raise ValueError("'output_channels' needs to has at least one value")
        if len(output_channels) != 1 and len(output_channels) != 2:
            raise ValueError(f"'output_channels' must be a list of one or two values at max, not {output_channels}")

        self.depth = len(feature_maps) - 1
        self.ndim = 3 if len(image_shape) == 4 else 2
        self.z_down = z_down
        self.output_channels = output_channels
        self.multihead = len(output_channels) == 2
        self.contrast = contrast
        if type(isotropy) == bool:
            isotropy = isotropy * len(feature_maps)
        if self.ndim == 3:
            conv = nn.Conv3d
            convtranspose = nn.ConvTranspose3d
            pooling = nn.MaxPool3d
            norm_func = get_norm_3d
            dropout = nn.Dropout3d
        else:
            conv = nn.Conv2d
            convtranspose = nn.ConvTranspose2d
            pooling = nn.MaxPool2d
            norm_func = get_norm_2d
            dropout = nn.Dropout2d

        # Super-resolution
        self.pre_upsampling = None
        if len(upsampling_factor) > 1 and upsampling_position == "pre":
            self.pre_upsampling = convtranspose(
                image_shape[-1],
                image_shape[-1],
                kernel_size=upsampling_factor,
                stride=upsampling_factor,
            )

        # ENCODER
        self.down_path = nn.ModuleList()
        self.mpooling_layers = nn.ModuleList()
        in_channels = image_shape[-1]

        # extra (larger) input layer
        if larger_io:
            kernel_size = (k_size + 2, k_size + 2) if self.ndim == 2 else (k_size + 2, k_size + 2, k_size + 2)
            if isotropy[0] is False and self.ndim == 3:
                kernel_size = (1, k_size + 2, k_size + 2)
            self.conv_in = ConvBlock(
                conv=conv,
                in_size=in_channels,
                out_size=feature_maps[0],
                k_size=kernel_size,
                act=activation,
                norm=normalization,
            )
            in_channels = feature_maps[0]
        else:
            self.conv_in = None

        for i in range(self.depth):
            kernel_size = (k_size, k_size) if self.ndim == 2 else (k_size, k_size, k_size)
            if isotropy[i] is False and self.ndim == 3:
                kernel_size = (1, k_size, k_size)
            self.down_path.append(
                ResConvBlock(
                    conv,
                    in_channels,
                    feature_maps[i],
                    kernel_size,
                    activation,
                    norm=normalization,
                    dropout=drop_values[i],
                    se_block=True,
                    first_block=True if i == 0 else False,
                    extra_conv=extra_conv,
                )
            )
            mpool = (z_down[i], 2, 2) if self.ndim == 3 else (2, 2)
            self.mpooling_layers.append(pooling(mpool))
            in_channels = feature_maps[i]

        kernel_size = (k_size, k_size) if self.ndim == 2 else (k_size, k_size, k_size)
        if isotropy[-1] is False and self.ndim == 3:
            kernel_size = (1, k_size, k_size)
        self.bottleneck = ResConvBlock(
            conv,
            in_channels,
            feature_maps[-1],
            kernel_size,
            activation,
            norm=normalization,
            dropout=drop_values[-1],
            se_block=True,
            extra_conv=extra_conv,
        )

        # DECODER
        self.up_path = nn.ModuleList()
        in_channels = feature_maps[-1]
        for i in range(self.depth - 1, -1, -1):
            kernel_size = (k_size, k_size) if self.ndim == 2 else (k_size, k_size, k_size)
            if isotropy[i] is False and self.ndim == 3:
                kernel_size = (1, k_size, k_size)
            self.up_path.append(
                ResUpBlock(
                    ndim=self.ndim,
                    convtranspose=convtranspose,
                    in_size=in_channels,
                    out_size=feature_maps[i],
                    in_size_bridge=feature_maps[i],
                    z_down=z_down[i],
                    up_mode=upsample_layer,
                    conv=conv,
                    k_size=kernel_size,
                    act=activation,
                    norm=normalization,
                    dropout=drop_values[i],
                    se_block=True,
                    extra_conv=extra_conv,
                )
            )
            in_channels = feature_maps[i]

        # extra (larger) output layer
        if larger_io:
            kernel_size = (k_size + 2, k_size + 2) if self.ndim == 2 else (k_size + 2, k_size + 2, k_size + 2)
            if isotropy[0] is False and self.ndim == 3:
                kernel_size = (1, k_size + 2, k_size + 2)
            self.conv_out = ConvBlock(
                conv=conv,
                in_size=feature_maps[0],
                out_size=feature_maps[0],
                k_size=kernel_size,
                act=activation,
                norm=normalization,
            )
        else:
            self.conv_out = None

        # Super-resolution
        self.post_upsampling = None
        if len(upsampling_factor) > 1 and upsampling_position == "post":
            self.post_upsampling = convtranspose(
                feature_maps[0],
                feature_maps[0],
                kernel_size=upsampling_factor,
                stride=upsampling_factor,
            )

        if self.contrast:
            # extra added layers
            self.last_block = nn.Sequential(
                conv(feature_maps[0], feature_maps[0], kernel_size=3, stride=1, padding=1),
                norm_func(normalization, feature_maps[0]),
                dropout(0.10),
                conv(feature_maps[0], output_channels[0], kernel_size=1, stride=1, padding=0, bias=False),
            )

            self.proj_head = ProjectionHead(ndim=self.ndim, in_channels=feature_maps[0], proj_dim=contrast_proj_dim)
        else:
            self.last_block = conv(feature_maps[0], output_channels[0], kernel_size=1, padding="same")

        # Multi-head:
        #   Instance segmentation: instances + classification
        #   Detection: points + classification
        self.last_class_head = None
        if self.multihead:
            self.last_class_head = conv(feature_maps[0], output_channels[1], kernel_size=1, padding="same")

        self.apply(self._init_weights)

    def forward(self, x) -> Dict | torch.Tensor:
        """
        Perform the forward pass of the ResUNet_SE model.

        The input `x` first undergoes optional pre-upsampling for super-resolution
        and an optional larger input convolution. It then passes through the
        encoder path (downsampling blocks with pooling), followed by a bottleneck.
        The decoder path upsamples features, concatenates them with corresponding
        skip connections from the encoder, and processes them through upsampling blocks.
        Finally, optional post-upsampling and the final prediction head(s) are applied.

        Parameters
        ----------
        x : torch.Tensor
            The input image tensor.
            Expected shape for 2D: `(batch_size, input_channels, height, width)`.
            Expected shape for 3D: `(batch_size, input_channels, depth, height, width)`.

        Returns
        -------
        Dict | torch.Tensor
            If `contrast` is True or `multihead` is True, returns a dictionary
            containing different output tensors (e.g., "pred", "embed", "class").
            Otherwise, returns only the primary prediction tensor.
            The primary prediction tensor ("pred") will have `output_channels[0]`
            channels and spatial dimensions corresponding to the original input
            (or upsampled input if `upsampling_position` is "pre").
        """
        # Super-resolution
        if self.pre_upsampling:
            x = self.pre_upsampling(x)

        # extra large-kernel input layer
        if self.conv_in:
            x = self.conv_in(x)

        # Down
        blocks = []
        for i, layers in enumerate(zip(self.down_path, self.mpooling_layers)):
            down, pool = layers
            x = down(x)
            if i != len(self.down_path):
                blocks.append(x)
                x = pool(x)

        x = self.bottleneck(x)

        # Up
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        # extra large-kernel output layer
        if self.conv_out:
            x = self.conv_out(x)

        feats = x
        # Super-resolution
        if self.post_upsampling:
            feats = self.post_upsampling(feats)

        # Regular output
        out = self.last_block(feats)
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
            out_dict["class"] = self.last_class_head(feats)

        if len(out_dict.keys()) == 1:
            return out_dict["pred"]
        else:
            return out_dict

    def _init_weights(self, m):
        """
        Initialize the weights of convolutional, linear, and LayerNorm layers.

        Applies Xavier uniform initialization to convolutional and linear layer weights
        (with bias set to 0 if present). For LayerNorm, weights are set to 1.0 and
        biases to 0.

        Parameters
        ----------
        m : nn.Module
            The module whose weights are to be initialized.
        """
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
