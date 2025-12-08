"""Residual U-Net model definition for 2D/3D biomedical image taks.

This module provides the ResUNet class, which implements a configurable Residual U-Net architecture
with support for 2D and 3D data, super-resolution, multi-head outputs, and optional
contrastive learning head.

Reference:
    Road Extraction by Deep Residual U-Net
    https://ieeexplore.ieee.org/abstract/document/8309343
"""
import torch
import torch.nn as nn
from typing import Dict 

from biapy.models.blocks import ResConvBlock, ResUpBlock, ConvBlock, get_norm_2d, get_norm_3d
from biapy.models.heads import ProjectionHead

class ResUNet(nn.Module):
    """Configurable 2D/3D Residual U-Net model for image segmentation and super-resolution.

    Supports multi-head outputs and optional contrastive learning head.
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
        contrast: bool = False,
        contrast_proj_dim: int = 256,
    ):
        """
        Create 2D/3D Residual U-Net.

        Reference: `Road Extraction by Deep Residual U-Net <https://ieeexplore.ieee.org/abstract/document/8309343>`_.

        Parameters
        ----------
        image_shape : 3D/4D tuple
            Dimensions of the input image. E.g. ``(y, x, channels)`` or ``(z, y, x, channels)``.

        activation : str, optional
            Activation layer.

        feature_maps : array of ints, optional
            Feature maps to use on each level.

        drop_values : float, optional
            Dropout value to be fixed.

        normalization : str, optional
            Normalization layer (one of ``'bn'``, ``'sync_bn'`` ``'in'``, ``'gn'`` or ``'none'``).

        k_size : int, optional
            Kernel size.

        upsample_layer : str, optional
            Type of layer to use to make upsampling. Two options: "convtranspose" or "upsampling".

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

        isotropy : bool or list of bool, optional
            Whether to use 3d or 2d convolutions at each U-Net level even if input is 3d.

        larger_io : bool, optional
            Whether to use extra and larger kernels in the input and output layers.

        contrast : bool, optional
            Whether to add contrastive learning head to the model. Default is ``False``.

        contrast_proj_dim : int, optional
            Dimension of the projection head for contrastive learning. Default is ``256``.

        Returns
        -------
        model : Torch model
            Residual U-Net model.


        Calling this function with its default parameters returns the following network:

        .. image:: ../../img/models/unet.png
            :width: 100%
            :align: center

        Image created with `PlotNeuralNet <https://github.com/HarisIqbal88/PlotNeuralNet>`_.
        """
        super(ResUNet, self).__init__()

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
            dropout = nn.Dropout3d

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
                    conv=conv,
                    in_size=in_channels,
                    out_size=feature_maps[i],
                    k_size=kernel_size,
                    act=activation,
                    norm=normalization,
                    dropout=drop_values[i],
                    first_block=True if i == 0 else False,
                )
            )
            mpool = (z_down[i], 2, 2) if self.ndim == 3 else (2, 2)
            self.mpooling_layers.append(pooling(mpool))
            in_channels = feature_maps[i]

        kernel_size = (k_size, k_size) if self.ndim == 2 else (k_size, k_size, k_size)
        if isotropy[-1] is False and self.ndim == 3:
            kernel_size = (1, k_size, k_size)
        self.bottleneck = ResConvBlock(
            conv=conv,
            in_size=in_channels,
            out_size=feature_maps[-1],
            k_size=kernel_size,
            act=activation,
            norm=normalization,
            dropout=drop_values[-1],
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
        """Forward pass of the Residual U-Net model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, channels, ...).

        Returns
        -------
        Dict or torch.Tensor
            Model output. Returns a dictionary if multi-head or contrastive outputs are enabled,
            otherwise returns the main prediction tensor.
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
