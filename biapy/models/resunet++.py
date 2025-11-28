"""
ResUNet++ model definition for 2D and 3D biomedical image segmentation.

This module implements the ResUNet++ architecture, a deep learning model tailored for
semantic segmentation tasks in biomedical imaging. It extends the traditional U-Net
architecture with residual connections, squeeze-and-excitation (SE) blocks, attention
mechanisms, and atrous spatial pyramid pooling (ASPP), offering enhanced feature 
representation and robustness across 2D and 3D image data.

The implementation is flexible to support tasks like:

- Semantic segmentation
- Instance segmentation (with multi-head output)
- Point detection
- Super-resolution
- Contrastive learning

Reference:
    ResUNet++: An Advanced Architecture for Medical Image Segmentation
    https://arxiv.org/pdf/1911.07067.pdf
"""
import torch
import torch.nn as nn
from typing import Dict

from biapy.models.blocks import (
    ResConvBlock,
    ResUpBlock,
    SqExBlock,
    ResUNetPlusPlus_AttentionBlock,
    get_norm_2d, 
    get_norm_3d
)
from biapy.models.heads import ASPP, ProjectionHead

class ResUNetPlusPlus(nn.Module):
    """
    Implementation of the ResUNet++ architecture for 2D and 3D image segmentation.

    This model integrates residual blocks, SE blocks, attention mechanisms, and ASPP modules
    into a U-Net-like encoder-decoder architecture, enhancing performance on complex biomedical images.

    Parameters
    ----------
    image_shape : tuple
        Input image shape. For 2D: (Y, X, C), for 3D: (Z, Y, X, C).

    activation : str, optional
        Activation function to use (e.g., "ReLU", "ELU").

    feature_maps : list of int
        Number of feature maps at each encoder level.

    drop_values : list of float
        Dropout values at each level.

    normalization : str
        Normalization layer to apply ("bn", "sync_bn", "in", "gn", or "none").

    k_size : int
        Kernel size for convolutions.

    upsample_layer : str
        Upsampling layer type: "convtranspose" or "upsampling".

    z_down : list of int
        Downsampling factor along the Z-axis for each encoder level. Set to 1 for 2D data.

    output_channels : list of int
        Number of output channels. If length 2, multi-task outputs (e.g., segmentation + classification).

    upsampling_factor : tuple of int, optional
        Upsampling scale factor for super-resolution workflows.

    upsampling_position : str
        Position of upsampling: "pre" (before model) or "post" (after model).

    contrast : bool
        Whether to add a contrastive learning head.

    contrast_proj_dim : int
        Dimensionality of the projection head for contrastive learning.

    Attributes
    ----------
    encoder : nn.ModuleList
        Encoder layers with residual connections and SE blocks.

    decoder : nn.ModuleList
        Decoder layers with residual upsampling.

    aspp : ASPP
        Atrous Spatial Pyramid Pooling module used at bottleneck.

    final_conv : nn.Module
        Final convolution layers for segmentation and optional classification output.

    contrast_head : nn.Module, optional
        Optional projection head for contrastive learning.

    Example
    -------
    >>> model = ResUNetPlusPlus(image_shape=(128, 128, 1))
    >>> output = model(torch.rand(1, 1, 128, 128))
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
        contrast: bool = False,
        contrast_proj_dim: int = 256,
    ):
        """
        Create 2D/3D ResUNet++.

        Reference: `ResUNet++: An Advanced Architecture for Medical Image Segmentation <https://arxiv.org/pdf/1911.07067.pdf>`_.

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

        contrast : bool, optional
            Whether to add contrastive learning head to the model. Default is ``False``.

        contrast_proj_dim : int, optional
            Dimension of the projection head for contrastive learning. Default is ``256``.

        Returns
        -------
        model : Torch model
            ResUNet++ model.


        Calling this function with its default parameters returns the following network:

        .. image:: ../../img/models/unet.png
            :width: 100%
            :align: center

        Image created with `PlotNeuralNet <https://github.com/HarisIqbal88/PlotNeuralNet>`_.
        """
        super(ResUNetPlusPlus, self).__init__()

        if len(output_channels) == 0:
            raise ValueError("'output_channels' needs to has at least one value")
        if len(output_channels) != 1 and len(output_channels) != 2:
            raise ValueError(f"'output_channels' must be a list of one or two values at max, not {output_channels}")

        self.depth = len(feature_maps) - 2
        self.ndim = 3 if len(image_shape) == 4 else 2
        self.z_down = z_down
        self.output_channels = output_channels
        self.multihead = len(output_channels) == 2
        self.contrast = contrast
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
        self.sqex_blocks = nn.ModuleList()
        self.down_path.append(
            ResConvBlock(
                conv=conv,
                in_size=image_shape[-1],
                out_size=feature_maps[0],
                k_size=k_size,
                act=activation,
                norm=normalization,
                dropout=drop_values[0],
                skip_k_size=k_size,
                skip_norm=normalization,
                first_block=True,
            )
        )
        self.sqex_blocks.append(SqExBlock(feature_maps[0], ndim=self.ndim))
        mpool = (z_down[0], 2, 2) if self.ndim == 3 else (2, 2)
        self.mpooling_layers.append(pooling(mpool))
        in_channels = feature_maps[0]
        for i in range(self.depth):
            self.down_path.append(
                ResConvBlock(
                    conv=conv,
                    in_size=in_channels,
                    out_size=feature_maps[i + 1],
                    k_size=k_size,
                    act=activation,
                    norm=normalization,
                    dropout=drop_values[i],
                    skip_k_size=k_size,
                    skip_norm=normalization,
                    first_block=False,
                )
            )
            mpool = (z_down[i + 1], 2, 2) if self.ndim == 3 else (2, 2)
            self.mpooling_layers.append(pooling(mpool))
            in_channels = feature_maps[i + 1]
            if i != self.depth - 1:
                self.sqex_blocks.append(SqExBlock(in_channels, ndim=self.ndim))
        self.sqex_blocks.append(
            None
        )  # So it can be used zip() with the length of self.down_path and self.mpooling_layers
        self.aspp_bridge = ASPP(
            conv=conv,
            in_dims=in_channels,
            out_dims=feature_maps[-1],
            norm=normalization,
        )

        # DECODER
        self.up_path = nn.ModuleList()
        self.attentions = nn.ModuleList()
        for i in range(self.depth - 1, -1, -1):
            self.attentions.append(
                ResUNetPlusPlus_AttentionBlock(
                    conv=conv,
                    maxpool=pooling,
                    input_encoder=feature_maps[i],
                    input_decoder=feature_maps[i + 2],
                    output_dim=feature_maps[i + 2],
                    norm=normalization,
                    z_down=z_down[i + 1],
                )
            )
            self.up_path.append(
                ResUpBlock(
                    ndim=self.ndim,
                    convtranspose=convtranspose,
                    in_size=feature_maps[i + 2],
                    out_size=feature_maps[i + 1],
                    in_size_bridge=feature_maps[i],
                    z_down=z_down[i + 1],
                    up_mode=upsample_layer,
                    conv=conv,
                    k_size=k_size,
                    act=activation,
                    norm=normalization,
                    dropout=drop_values[i + 2],
                    skip_k_size=k_size,
                    skip_norm=normalization,
                )
            )
        self.aspp_out = ASPP(
            conv=conv,
            in_dims=feature_maps[1],
            out_dims=feature_maps[0],
            norm=normalization,
        )

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
        Forward pass of the ResUNet++ model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W) or (B, C, D, H, W) for 2D/3D inputs.

        Returns
        -------
        torch.Tensor or list of torch.Tensor
            Model output(s). If multiple output channels are configured, returns a list.
        """
        # Super-resolution
        if self.pre_upsampling:
            x = self.pre_upsampling(x)

        # Down
        blocks = []
        for i, layers in enumerate(zip(self.down_path, self.sqex_blocks, self.mpooling_layers)):
            down, sqex, pool = layers
            x = down(x)
            if i < len(self.down_path) - 1:  # Avoid last block
                x = sqex(x)
            if i != len(self.down_path):
                if i != 0:  # First level is not downsampled
                    x = pool(x)
                blocks.append(x)

        x = self.aspp_bridge(x)

        # Up
        for i, layers in enumerate(zip(self.attentions, self.up_path)):
            att, up = layers
            x = att(blocks[-i - 2], x)
            x = up(x, blocks[-i - 2])

        x = self.aspp_out(x)

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
