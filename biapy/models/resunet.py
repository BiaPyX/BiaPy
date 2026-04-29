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
from typing import Dict, List 

from biapy.models.blocks import (
    ResConvBlock, 
    ResUpBlock, 
    ConvBlock, 
    get_norm_2d, 
    get_norm_3d, 
    prepare_activation_layers, 
    init_weights
)
from biapy.models.heads import ProjectionHead

class ResUNet(nn.Module):
    """
    Configurable 2D/3D Residual U-Net model for image segmentation and super-resolution.

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
        yx_down=[2, 2, 2, 2],
        z_down=[2, 2, 2, 2],
        output_channels=[1],
        separated_decoders=False,
        output_channel_info=["F"],
        explicit_activations: bool = False,
        head_activations: List[str] = ["ce_sigmoid"],
        upsampling_factor=(),
        upsampling_position="pre",
        isotropy=False,
        larger_io=True,
        contrast: bool = False,
        contrast_proj_dim: int = 256,
        return_one_tensor: bool = False,
    ):
        """
        Create 2D/3D Residual U-Net.

        Reference: `Road Extraction by Deep Residual U-Net <https://ieeexplore.ieee.org/abstract/document/8309343>`_.

        Parameters
        ----------
        image_shape : 3D/4D tuple
            Dimensions of the input image. E.g. ``(y, x, channels)`` or ``(z, y, x, channels)``.

        activation : str, optional
            Activation layer to be used throughout the model.

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

        yx_down : List of ints, optional
            Downsampling used in y and x dimensions. Set it to ``1`` if the dataset is not isotropic.

        output_channels : list of int, optional
            Output channels of the network. If one value is provided, the model will have a single output head. 
            If two values are provided, the model will have two output heads (e.g. for multi-task learning with 
            instance segmentation and classification).

        separated_decoders : bool, optional
            Whether to use separated decoders for each output head.

        output_channel_info : list of str, optional
            Information about the type of output channels. Possible values are:
            - "X": where X is a letter, e.g. "F" for foreground, "D" for distance, "R" for rays, "C" for cpntours, etc.
            - "class": classification (e.g. for multi-task learning)

        explicit_activations : bool, optional
            If True, uses explicit activation functions in the last layers.
        
        head_activations : List[str], optional
            Activation functions to apply to each output head if `explicit_activations` is True.

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

        return_one_tensor : bool, optional
            Whether to return a single tensor with all outputs concatenated (if False, returns a dictionary
            with separate entries). Default is ``False``.

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
        if contrast and len(output_channels) > 2:
            raise ValueError("If 'contrast' is True, 'output_channels' can only have two values at max: one for the main output and one for the class.")
        print("Selected output channels:")        
        for i, info in enumerate(output_channel_info):
            print(f"  - {i} channel for {info} output")

        self.depth = len(feature_maps) - 1
        self.ndim = 3 if len(image_shape) == 4 else 2
        self.z_down = z_down
        self.yx_down = yx_down
        self.output_channels = output_channels
        self.output_channel_info = output_channel_info
        self.return_class = True if "class" in output_channel_info else False
        self.contrast = contrast
        self.explicit_activations = explicit_activations
        self.return_one_tensor = return_one_tensor
        if self.explicit_activations:
            assert len(head_activations) == sum(output_channels), "If 'explicit_activations' is True, 'head_activations' needs to "
            "have the same number of values as 'output_channels'"
            self.head_activations, self.class_head_activations = prepare_activation_layers(head_activations, output_channel_info, output_channels)
            if self.return_class and self.class_head_activations is None:
                raise ValueError("If 'return_class' is True, 'head_activations' must be provided.")
        if type(isotropy) == bool:
            isotropy = [isotropy] * len(feature_maps)
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
        if len(upsampling_factor) > 0 and upsampling_position == "pre":
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
            if not isotropy[0] and self.ndim == 3:
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
            if not isotropy[i] and self.ndim == 3:
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
            mpool = (z_down[i], yx_down[i], yx_down[i]) if self.ndim == 3 else (yx_down[i], yx_down[i])
            self.mpooling_layers.append(pooling(mpool))
            in_channels = feature_maps[i]

        kernel_size = (k_size, k_size) if self.ndim == 2 else (k_size, k_size, k_size)
        if not isotropy[-1] and self.ndim == 3:
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
        self.num_decoders = 1 if not separated_decoders else len(output_channels)
        self.up_paths = nn.ModuleList([nn.ModuleList() for _ in range(self.num_decoders)])
        for j in range(self.num_decoders):
            in_channels = feature_maps[-1]
            for i in range(self.depth - 1, -1, -1):
                kernel_size = (k_size, k_size) if self.ndim == 2 else (k_size, k_size, k_size)
                if not isotropy[i] and self.ndim == 3:
                    kernel_size = (1, k_size, k_size)
                self.up_paths[j].append(
                    ResUpBlock(
                        ndim=self.ndim,
                        convtranspose=convtranspose,
                        in_size=in_channels, 
                        out_size=feature_maps[i],
                        in_size_bridge=feature_maps[i],
                        z_down=z_down[i],
                        yx_down=yx_down[i],
                        up_mode=upsample_layer,
                        conv=conv,
                        k_size=kernel_size,
                        act=activation,
                        norm=normalization,
                        dropout=drop_values[i],
                    ) # type: ignore
                )
                in_channels = feature_maps[i]

        # extra (larger) output layer
        if larger_io:
            kernel_size = (k_size + 2, k_size + 2) if self.ndim == 2 else (k_size + 2, k_size + 2, k_size + 2)
            if not isotropy[0] and self.ndim == 3:
                kernel_size = (1, k_size + 2, k_size + 2)
            self.conv_out = nn.ModuleList([
                ConvBlock(
                    conv=conv,
                    in_size=feature_maps[0],
                    out_size=feature_maps[0],
                    k_size=kernel_size,
                    act=activation,
                    norm=normalization,
                ) for _ in range(self.num_decoders)
            ])
        else:
            self.conv_out = None

        # Super-resolution
        self.post_upsampling = None
        if len(upsampling_factor) > 0 and upsampling_position == "post":
            self.post_upsampling = convtranspose(
                feature_maps[0],
                feature_maps[0],
                kernel_size=upsampling_factor,
                stride=upsampling_factor,
            )

        if self.contrast:
            # extra added layers
            self.heads = nn.Sequential(
                conv(feature_maps[0], feature_maps[0], kernel_size=3, stride=1, padding=1),
                norm_func(normalization, feature_maps[0]),
                dropout(0.10),
                conv(feature_maps[0], output_channels[0], kernel_size=1, stride=1, padding=0, bias=False),
            )

            self.proj_head = ProjectionHead(ndim=self.ndim, in_channels=feature_maps[0], proj_dim=contrast_proj_dim)
        else:
            self.heads = nn.Sequential()
            for i, out_ch in enumerate(output_channels):
                self.heads.append(conv(feature_maps[0], out_ch, kernel_size=1, padding="same"))

        init_weights(self)

    def forward(self, x) -> Dict | torch.Tensor:
        """
        Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width) for 2D or (batch_size, channels, depth, height, width) for 3D.

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

        # Encoder
        blocks = []
        for i, layers in enumerate(zip(self.down_path, self.mpooling_layers)):
            down, pool = layers
            x = down(x)
            blocks.append(x)
            x = pool(x)

        x_bot = self.bottleneck(x)

        # Decoder
        feats = []
        for j in range(self.num_decoders):
            x = x_bot
            for i, up in enumerate(self.up_paths[j]):
                x = up(x, blocks[-i - 1])
            feats.append(x)

        # extra large-kernel output layer
        if self.conv_out:
            for j in range(self.num_decoders):
                feats[j] = self.conv_out[j](feats[j])

        # Super-resolution
        if self.post_upsampling:
            feats[0] = self.post_upsampling(feats[0])

        out_dict = {}

        # Pass the features through the output heads
        class_outs, outs = [], []
        for i, head in enumerate(self.heads):
            feat = feats[i] if self.num_decoders > 1 else feats[0]
            if "class" in self.output_channel_info[i]:
                class_outs.append(head(feat))
            else:
                outs.append(head(feat))

        outs = torch.cat(outs, dim=1)

        # Apply activations to the output heads if explicit_activations is True
        if self.explicit_activations:
            # If there is only one activation, apply it to the whole tensor
            if len(self.head_activations) == 1:
                outs = self.head_activations[0](outs)
            else:
                for i, act in enumerate(self.head_activations):
                    outs[:, i:i+1] = act(outs[:, i:i+1])

            if self.return_class and self.class_head_activations is not None:
                for i, act in enumerate(self.class_head_activations):
                    class_outs[i] = act(class_outs[i])

        out_dict = {
            "pred": outs,
        }
        if self.return_class:
            out_dict["class"] = torch.cat(class_outs, dim=1)

        # Contrastive learning head
        if self.contrast:
            out_dict["embed"] = self.proj_head(feats[0])

        if len(out_dict.keys()) == 1:
            return out_dict["pred"]
        else:
            if self.return_one_tensor:
                if "class" in out_dict:
                    return torch.cat((out_dict["pred"], torch.argmax(out_dict["class"], dim=1).unsqueeze(1)), dim=1)
                else:
                    return out_dict["pred"]
            return out_dict

