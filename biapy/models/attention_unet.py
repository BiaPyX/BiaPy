"""
Attention U-Net Implementation.

This module implements a 2D/3D U-Net architecture with attention gates for improved 
feature learning in medical image segmentation and other computer vision tasks.

The implementation is based on the paper:
"Attention U-Net: Learning Where to Look for the Pancreas" 
https://arxiv.org/abs/1804.03999

The attention mechanism helps the model focus on relevant spatial regions while 
suppressing irrelevant background activations, leading to improved segmentation 
performance especially in medical imaging applications.

Classes:
    Attention_U_Net: Main U-Net architecture with attention gates
"""
import torch
import torch.nn as nn

from biapy.models.blocks import DoubleConvBlock, UpBlock, get_norm_2d, get_norm_3d, prepare_activation_layers, init_weights
from typing import Dict, List
from biapy.models.heads import ProjectionHead

class Attention_U_Net(nn.Module):
    """
    Configurable 2D/3D Attention U-Net model for image segmentation and super-resolution.

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
        output_channel_info=["F"],
        explicit_activations: bool = False,
        head_activations: List[str] = ["ce_sigmoid"],
        upsampling_factor=(),
        upsampling_position="pre",
        contrast: bool = False,
        contrast_proj_dim: int = 256,
    ):
        """
        Create 2D/3D U-Net with Attention blocks.

        Reference: `Attention U-Net: Learning Where to Look for the Pancreas <https://arxiv.org/abs/1804.03999>`_.

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

        output_channels : list of int, optional
            Output channels of the network. If one value is provided, the model will have a single output head. 
            If two values are provided, the model will have two output heads (e.g. for multi-task learning with 
            instance segmentation and classification).

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

        contrast : bool, optional
            Whether to add contrastive learning head to the model. Default is ``False``.

        contrast_proj_dim : int, optional
            Dimension of the projection head for contrastive learning. Default is ``256``.

        Returns
        -------
        model : Torch model
            Attention U-Net model.


        Calling this function with its default parameters returns the following network:

        .. image:: ../../img/models/unet.png
            :width: 100%
            :align: center

        Image created with `PlotNeuralNet <https://github.com/HarisIqbal88/PlotNeuralNet>`_.

        That networks incorporates in skip connecions Attention Gates (AG), which can be seen as follows:

        .. image:: ../../img/models/attention_gate.png
            :width: 100%
            :align: center

        Image extracted from `Attention U-Net: Learning Where to Look for the Pancreas <https://arxiv.org/abs/1804.03999>`_.
        """
        super(Attention_U_Net, self).__init__()

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
        self.output_channels = output_channels
        self.output_channel_info = output_channel_info
        self.return_class = True if "class" in output_channel_info else False
        self.contrast = contrast
        self.explicit_activations = explicit_activations
        if self.explicit_activations:
            assert len(head_activations) == len(output_channels), "If 'explicit_activations' is True, 'head_activations' needs to "
            "have the same number of values as 'output_channels'"
            self.head_activations, self.class_head_activations = prepare_activation_layers(head_activations, output_channel_info)
            if self.return_class and self.class_head_activations is None:
                raise ValueError("If 'return_class' is True, 'head_activations' must be provided.")
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
        for i in range(self.depth):
            self.down_path.append(
                DoubleConvBlock(
                    conv,
                    in_channels,
                    feature_maps[i],
                    k_size,
                    activation,
                    normalization,
                    drop_values[i],
                )
            )
            mpool = (self.z_down[i], 2, 2) if self.ndim == 3 else (2, 2)
            self.mpooling_layers.append(pooling(mpool))
            in_channels = feature_maps[i]

        self.bottleneck = DoubleConvBlock(
            conv,
            in_channels,
            feature_maps[-1],
            k_size,
            activation,
            normalization,
            drop_values[-1],
        )

        # DECODER
        self.up_path = nn.ModuleList()
        in_channels = feature_maps[-1]
        for i in range(self.depth - 1, -1, -1):
            self.up_path.append(
                UpBlock(
                    self.ndim,
                    convtranspose,
                    in_channels,
                    feature_maps[i],
                    z_down[i],
                    upsample_layer,
                    conv,
                    k_size,
                    activation,
                    normalization,
                    drop_values[i],
                    attention_gate=True,
                )
            )
            in_channels = feature_maps[i]

        # Super-resolution
        self.post_upsampling = None
        if len(upsampling_factor) > 1 and upsampling_position == "post":
            self.post_upsampling = convtranspose(
                feature_maps[0],
                feature_maps[0],
                kernel_size=upsampling_factor,
                stride=upsampling_factor,
            )

        # To store which head corresponds to which output channel in the multi-head scenario
        self.out_head_map = []

        if self.contrast:
            # extra added layers
            self.heads = nn.Sequential(
                conv(feature_maps[0], feature_maps[0], kernel_size=3, stride=1, padding=1),
                norm_func(normalization, feature_maps[0]),
                dropout(0.10),
                conv(feature_maps[0], output_channels[0], kernel_size=1, stride=1, padding=0, bias=False),
            )

            self.proj_head = ProjectionHead(ndim=self.ndim, in_channels=feature_maps[0], proj_dim=contrast_proj_dim)
            self.out_head_map += [0] * output_channels[0]
        else:
            self.heads = nn.Sequential()
            for i, out_ch in enumerate(output_channels):
                self.heads.append(conv(feature_maps[0], out_ch, kernel_size=1, padding="same"))
                self.out_head_map += [i] * out_ch

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

        # Encoder
        blocks = []
        for i, layers in enumerate(zip(self.down_path, self.mpooling_layers)):
            down, pool = layers
            x = down(x)
            blocks.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        # Decoder
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        feats = x
        # Super-resolution
        if self.post_upsampling:
            feats = self.post_upsampling(feats)

        out_dict = {}

        # Pass the features through the output heads
        class_outs, outs = [], []
        for i, head_id in enumerate(self.out_head_map):
            if "class" not in self.output_channel_info[i]:
                outs.append(self.heads[head_id](feats))
            else:
                class_outs.append(self.heads[head_id](feats))  
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
            out_dict["embed"] = self.proj_head(feats)

        if len(out_dict.keys()) == 1:
            return out_dict["pred"]
        else:
            return out_dict