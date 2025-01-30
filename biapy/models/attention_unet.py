import torch
import torch.nn as nn
from typing import List

from biapy.models.blocks import DoubleConvBlock, UpBlock


class Attention_U_Net(nn.Module):
    """
    Create 2D/3D U-Net with Attention blocks.

    Reference: `Attention U-Net: Learning Where to Look for the Pancreas <https://arxiv.org/abs/1804.03999>`_.

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
    ):
        super(Attention_U_Net, self).__init__()
        
        if len(output_channels) == 0:
            raise ValueError("'output_channels' needs to has at least one value")
        if len(output_channels) != 1 and len(output_channels) != 2:
            raise ValueError(f"'output_channels' must be a list of one or two values at max, not {output_channels}")
        
        self.depth = len(feature_maps) - 1
        self.ndim = 3 if len(image_shape) == 4 else 2
        self.z_down = z_down
        self.output_channels = output_channels
        self.multihead = len(output_channels) == 2
        if self.ndim == 3:
            conv = nn.Conv3d
            convtranspose = nn.ConvTranspose3d
            pooling = nn.MaxPool3d
        else:
            conv = nn.Conv2d
            convtranspose = nn.ConvTranspose2d
            pooling = nn.MaxPool2d

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

        self.last_block = conv(feature_maps[0], output_channels[0], kernel_size=1, padding="same")
        # Multi-head: 
        #   Instance segmentation: instances + classification
        #   Detection: points + classification 
        self.last_class_head = None
        if self.multihead == 2:
            self.last_class_head = conv(feature_maps[0], output_channels[1], kernel_size=1, padding="same")

        self.apply(self._init_weights)

    def forward(self, x) -> torch.Tensor | List[torch.Tensor]:
        # Super-resolution
        if self.pre_upsampling is not None:
            x = self.pre_upsampling(x)

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

        # Super-resolution
        if self.post_upsampling is not None:
            x = self.post_upsampling(x)

        class_head_out = torch.empty(())
        if self.multihead and self.last_class_head is not None:
            class_head_out = self.last_class_head(x)

        x = self.last_block(x)

        if self.multihead:
            return [x, class_head_out]
        else:
            return x

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
