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

    stochastic_depth_prob: float, optional
        Maximum stochastic depth probability. This probability will progressively increase with each
        layer, reaching its maximum value at the bottleneck layer.

    layer_scale: float, optional
        Layer Scale parameter.

    cn_layers:
        Number of times each ConvNext block is repeated in each level. This array should be the same length
        as the 'feature_maps' attribute.

    isotropy : bool or list of bool, optional
        Whether to use 3d or 2d depthwise convolutions at each U-NeXt level even if input is 3d.

    stem_k_size : int, optional
        Size of the stem kernel (default: 2).

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
        output_channels=[1],
        upsampling_factor=(),
        upsampling_position="pre",
        stochastic_depth_prob=0.1,
        layer_scale=1e-6,
        cn_layers=[2, 2, 2, 2],
        isotropy=True,
        stem_k_size=2,
    ):
        super(U_NeXt_V1, self).__init__()

        if len(output_channels) == 0:
            raise ValueError("'output_channels' needs to has at least one value")
        if len(output_channels) != 1 and len(output_channels) != 2:
            raise ValueError(f"'output_channels' must be a list of one or two values at max, not {output_channels}")

        self.depth = len(feature_maps) - 1
        self.ndim = 3 if len(image_shape) == 4 else 2
        self.z_down = z_down
        self.output_channels = output_channels
        self.multihead = len(output_channels) == 2
        layer_norm = nn.LayerNorm

        # convert isotropy to list if it is a single bool
        if type(isotropy) == bool:
            isotropy = isotropy * len(feature_maps)

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
        z_factor = int(min(z_down[0] / stem_k_size, 1))
        mpool = (stem_k_size * z_factor, stem_k_size, stem_k_size) if self.ndim == 3 else (stem_k_size, stem_k_size)
        self.down_path.append(
            nn.Sequential(
                conv(in_channels, feature_maps[0], kernel_size=mpool, stride=mpool),
                pre_ln_permutation,
                layer_norm(feature_maps[0]),
                post_ln_permutation,
            )
        )

        # depthwise kernel size for ConvNeXt block
        kernel_size = (7, 7) if self.ndim == 2 else (7, 7, 7)

        # Encoder
        stage_block_id = 0
        total_stage_blocks = sum(cn_layers)
        sd_probs = []
        for i in range(self.depth):
            stage = nn.ModuleList()
            sd_probs_stage = []

            # adjust depthwise kernel size if needed
            if isotropy[i] is False and self.ndim == 3:
                kernel_size = (1, 7, 7)

            # ConvNeXtBlocks
            for _ in range(cn_layers[i]):
                sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                stage.append(
                    ConvNeXtBlock_V1(
                        self.ndim, conv, feature_maps[i], layer_scale, sd_prob, layer_norm, k_size=kernel_size
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
        if isotropy[-1] is False and self.ndim == 3:
            kernel_size = (1, 7, 7)
        for _ in range(cn_layers[-1]):
            sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
            stage.append(
                ConvNeXtBlock_V1(
                    self.ndim, conv, feature_maps[-1], layer_scale, sd_prob, layer_norm, k_size=kernel_size
                )
            )
            stage_block_id += 1
        self.bottleneck = nn.Sequential(*stage)

        # DECODER
        self.up_path = nn.ModuleList()
        in_channels = feature_maps[-1]

        for i in range(self.depth - 1, -1, -1):
            if isotropy[i] is False and self.ndim == 3:
                kernel_size = (1, 7, 7)
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
                    k_size=kernel_size,
                )
            )
            in_channels = feature_maps[i]

        # Inverted Stem
        mpool = (stem_k_size * z_factor, stem_k_size, stem_k_size) if self.ndim == 3 else (stem_k_size, stem_k_size)
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

        self.last_block = conv(feature_maps[0], output_channels[0], kernel_size=1, padding="same")
        # Multi-head:
        #   Instance segmentation: instances + classification
        #   Detection: points + classification
        self.last_class_head = None
        if self.multihead:
            self.last_class_head = conv(feature_maps[0], output_channels[1], kernel_size=1, padding="same")

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
