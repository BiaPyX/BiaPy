import torch
import torch.nn as nn
from typing import List

from biapy.models.blocks import UpConvNeXtBlock_V1, ConvNeXtBlock_V1
from torchvision.ops.misc import Permute


class U_NeXt_V1(nn.Module):
    """
    Create 2D/3D U-NeXt.

    Reference: `U-Net: Convolutional Networks for Biomedical Image Segmentation <https://arxiv.org/abs/1505.04597>`_,
    `A ConvNet for the 2020s <https://arxiv.org/abs/2201.03545>`_.

    Parameters
    ----------
    image_shape : 3D/4D tuple
        Dimensions of the input image. E.g. ``(y, x, channels)`` or ``(z, y, x, channels)``.

    activation : str, optional
        Activation layer.

    feature_maps : array of ints, optional
        Feature maps to use on each level.

    upsample_layer : str, optional
        Type of layer to use to make upsampling. Two options: "convtranspose" or "upsampling".

    z_down : List of ints, optional
        Downsampling used in z dimension. Set it to ``1`` if the dataset is not isotropic.

    n_classes: int, optional
        Number of classes.

    output_channels : str, optional
        Channels to operate with. Possible values: ``BC``, ``BCD``, ``BP``, ``BCDv2``,
        ``BDv2``, ``Dv2`` and ``BCM``.

    upsampling_factor : tuple of ints, optional
        Factor of upsampling for super resolution workflow for each dimension.

    upsampling_position : str, optional
        Whether the upsampling is going to be made previously (``pre`` option) to the model
        or after the model (``post`` option).

    stochastic_depth_prob: float, optional
        Maximum stochastic depth probability. This probability will progressively increase with each
        layer, reaching its maximum value at the bottleneck layer.

    layer_scale: float, optional
        Layer Scale parameter.

    cn_layers:
        Number of times each ConvNext block is repeated in each level. This array should be the same length
        as the 'feature_maps' attribute.


    Returns
    -------
    model : Torch model
        U-NeXt model.


    Calling this function with its default parameters returns the following network:

    .. image:: ../../img/models/unext.png
        :width: 100%
        :align: center


    """

    def __init__(
        self,
        image_shape=(256, 256, 1),
        feature_maps=[32, 64, 128, 256],
        upsample_layer="convtranspose",
        z_down=[2, 2, 2, 2],
        n_classes=1,
        output_channels="BC",
        upsampling_factor=(),
        upsampling_position="pre",
        stochastic_depth_prob=0.1,
        layer_scale=1e-6,
        cn_layers=[2, 2, 2, 2],
    ):
        super(U_NeXt_V1, self).__init__()
        self.depth = len(feature_maps) - 1
        self.ndim = 3 if len(image_shape) == 4 else 2
        self.z_down = z_down
        self.n_classes = 1 if n_classes <= 2 else n_classes
        self.multiclass = True if n_classes > 2 and output_channels is not None else False
        layer_norm = nn.LayerNorm

        if self.ndim == 3:
            conv = nn.Conv3d
            convtranspose = nn.ConvTranspose3d
            pre_ln_permutation = Permute([0, 2, 3, 4, 1])
            post_ln_permutation = Permute([0, 4, 1, 2, 3])
        else:
            conv = nn.Conv2d
            convtranspose = nn.ConvTranspose2d
            pre_ln_permutation = Permute([0, 2, 3, 1])
            post_ln_permutation = Permute([0, 3, 1, 2])

        # Super-resolution
        self.pre_upsampling = None
        if len(upsampling_factor) > 0 and upsampling_position == "pre":
            self.pre_upsampling = convtranspose(
                image_shape[-1],
                image_shape[-1],
                kernel_size=upsampling_factor,
                stride=upsampling_factor,
            )

        self.down_path = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        in_channels = image_shape[-1]

        # STEM
        mpool = (z_down[0], 2, 2) if self.ndim == 3 else (2, 2)
        self.down_path.append(
            nn.Sequential(
                conv(in_channels, feature_maps[0], kernel_size=mpool, stride=mpool),
                pre_ln_permutation,
                layer_norm(feature_maps[0]),
                post_ln_permutation,
            )
        )

        # Encoder
        stage_block_id = 0
        total_stage_blocks = sum(cn_layers)
        sd_probs = []
        for i in range(self.depth):
            stage = nn.ModuleList()
            sd_probs_stage = []

            # ConvNeXtBlocks
            for _ in range(cn_layers[i]):
                sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                stage.append(
                    ConvNeXtBlock_V1(
                        self.ndim,
                        conv,
                        feature_maps[i],
                        layer_scale,
                        sd_prob,
                        layer_norm,
                    )
                )
                stage_block_id += 1
                sd_probs_stage.append(sd_prob)
            self.down_path.append(nn.Sequential(*stage))
            sd_probs.append(sd_probs_stage)

            # Downsampling
            mpool = (z_down[i], 2, 2) if self.ndim == 3 else (2, 2)
            self.downsample_layers.append(
                nn.Sequential(
                    pre_ln_permutation,
                    layer_norm(feature_maps[i]),
                    post_ln_permutation,
                    conv(
                        feature_maps[i],
                        feature_maps[i + 1],
                        kernel_size=mpool,
                        stride=mpool,
                    ),
                )
            )

        # BOTTLENECK
        stage = nn.ModuleList()
        for _ in range(cn_layers[-1]):
            sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
            stage.append(ConvNeXtBlock_V1(self.ndim, conv, feature_maps[-1], layer_scale, sd_prob, layer_norm))
            stage_block_id += 1
        self.bottleneck = nn.Sequential(*stage)

        # DECODER
        self.up_path = nn.ModuleList()
        in_channels = feature_maps[-1]
        upsample_layer = "upsampling"
        for i in range(self.depth - 1, -1, -1):
            self.up_path.append(
                UpConvNeXtBlock_V1(
                    self.ndim,
                    convtranspose,
                    in_channels,
                    feature_maps[i],
                    z_down[i],
                    upsample_layer,
                    conv,
                    attention_gate=False,
                    cn_layers=cn_layers[i],
                    sd_probs=sd_probs[i],
                    layer_scale=layer_scale,
                    layer_norm=layer_norm,
                )
            )
            in_channels = feature_maps[i]

        # Inverted Stem
        mpool = (z_down[0], 2, 2) if self.ndim == 3 else (2, 2)
        self.up_path.append(
            nn.Sequential(
                convtranspose(feature_maps[0], feature_maps[0], kernel_size=mpool, stride=mpool),
                pre_ln_permutation,
                layer_norm(feature_maps[0]),
                post_ln_permutation,
            )
        )

        # Super-resolution
        self.post_upsampling = None
        if len(upsampling_factor) > 0 and upsampling_position == "post":
            self.post_upsampling = convtranspose(
                feature_maps[0],
                feature_maps[0],
                kernel_size=upsampling_factor,
                stride=upsampling_factor,
            )

        # Instance segmentation
        if output_channels is not None:
            if output_channels == "Dv2":
                self.last_block = conv(feature_maps[0], 1, kernel_size=1, padding="same")
            elif output_channels in ["BC", "BP"]:
                self.last_block = conv(feature_maps[0], 2, kernel_size=1, padding="same")
            elif output_channels in ["BDv2", "BD"]:
                self.last_block = conv(feature_maps[0], 2, kernel_size=1, padding="same")
            elif output_channels in ["BCM", "BCD", "BCDv2"]:
                self.last_block = conv(feature_maps[0], 3, kernel_size=1, padding="same")
        # Other
        else:
            self.last_block = conv(feature_maps[0], self.n_classes, kernel_size=1, padding="same")

        # Multi-head: instances + classification
        self.last_class_head = None
        if self.multiclass:
            self.last_class_head = conv(feature_maps[0], self.n_classes, kernel_size=1, padding="same")

        self.apply(self._init_weights)

    def forward(self, x) -> torch.Tensor | List[torch.Tensor]:
        # Super-resolution
        if self.pre_upsampling is not None:
            x = self.pre_upsampling(x)

        # Down
        blocks = []
        x = self.down_path[0](x)  # (stem)
        for i, layers in enumerate(zip(self.down_path[1:], self.downsample_layers)):
            down, pool = layers
            x = down(x)
            if i != len(self.down_path):
                blocks.append(x)
                x = pool(x)

        x = self.bottleneck(x)

        # Up
        for i, up in enumerate(self.up_path[:-1]):
            x = up(x, blocks[-i - 1])

        x = self.up_path[-1](x)

        if self.post_upsampling is not None:
            x = self.post_upsampling(x)

        class_head_out = torch.empty(())
        if self.multiclass and self.last_class_head is not None:
            class_head_out = self.last_class_head(x)

        x = self.last_block(x)

        if self.multiclass:
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
