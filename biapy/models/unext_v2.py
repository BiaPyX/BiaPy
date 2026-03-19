"""
This module implements the U-NeXt (Version 2) architecture, a U-Net based model that incorporates the latest advancements from ConvNeXt V2 blocks.

It aims to combine the strong hierarchical feature learning of U-Nets with the improved
design principles of ConvNeXt V2, which are co-designed and scaled with Masked
Autoencoders for enhanced performance.

U-NeXt_V2 is designed for both 2D and 3D image segmentation tasks. It features
a ConvNeXt V2-style encoder and decoder, with specialized blocks for downsampling,
upsampling, and the bottleneck. It supports various configurations, including
optional super-resolution, multi-head outputs, and stochastic depth for regularization.

Classes:

- ``U_NeXt_V2``: The main U-NeXt model (Version 2).

This module relies on building blocks defined in `biapy.models.blocks`, such as
`UpConvNeXtBlock_V2`, `ConvNeXtBlock_V2`, and `ProjectionHead`.

References:

- `U-Net: Convolutional Networks for Biomedical Image Segmentation <https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28>`_
- `ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders <https://openaccess.thecvf.com/content/CVPR2023/html/Woo_ConvNeXt_V2_Co-Designing_and_Scaling_ConvNets_With_Masked_Autoencoders_CVPR_2023_paper.html>`_.

Image representation:

.. image:: ../../img/models/unext.png
    :width: 100%
    :align: center

"""

import torch
import torch.nn as nn
from typing import Dict, List

from biapy.models.blocks import UpConvNeXtBlock_V2, ConvNeXtBlock_V2, prepare_activation_layers, init_weights
from torchvision.ops.misc import Permute
from biapy.models.heads import ProjectionHead


class U_NeXt_V2(nn.Module):
    """
    Create 2D/3D U-NeXt V2 (U-Net based model with ConvNeXt V2 blocks).

    U-NeXt V2 combines the classic U-Net architecture with modern ConvNeXt V2 blocks,
    leveraging the co-design and scaling principles from Masked Autoencoders. This
    model aims to achieve high performance in biomedical image segmentation by
    integrating strong hierarchical feature learning with efficient and robust
    convolutional designs.

    Reference: `U-Net: Convolutional Networks for Biomedical Image Segmentation <https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28>`_,
    `ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders <https://openaccess.thecvf.com/content/CVPR2023/html/Woo_ConvNeXt_V2_Co-Designing_and_Scaling_ConvNets_With_Masked_Autoencoders_CVPR_2023_paper.html>`_.
    """

    def __init__(
        self,
        image_shape=(256, 256, 1),
        feature_maps=[32, 64, 128, 256],
        upsample_layer="convtranspose",
        z_down=[2, 2, 2, 2],
        yx_down=[2, 2, 2, 2],
        output_channels=[1],
        separated_decoders=False,
        output_channel_info=["F"],
        explicit_activations: bool = False,
        head_activations: List[str] = ["ce_sigmoid"],
        upsampling_factor=(),
        upsampling_position="pre",
        stochastic_depth_prob=0.1,
        cn_layers=[2, 2, 2, 2],
        isotropy=True,
        stem_k_size=2,
        contrast: bool = False,
        contrast_proj_dim: int = 256,
    ):
        """
        Initialize the U-NeXt_V2 model.

        Sets up the ConvNeXt V2-style encoder (downsampling path), decoder (upsampling path), stem, bottleneck, and optional super-resolution and multi-head output layers.
        It dynamically selects 2D or 3D convolutional and normalization layers based on `ndim` and `isotropy` settings. Stochastic depth probabilities are
        progressively increased across layers.

        Parameters
        ----------
        image_shape : Tuple[int, ...]
            Dimensions of the input image. E.g., `(y, x, channels)` for 2D or
            `(z, y, x, channels)` for 3D. The last element `image_shape[-1]`
            should be the number of input channels.

        activation : str, optional
            Activation layer to be used throughout the model. (Note: ConvNeXt V2 blocks typically use GELU, this parameter
            might be less relevant for internal block activations but could apply to
            other parts if customized).

        feature_maps : List[int], optional
            A list specifying the number of feature maps (channels) at each level
            of the U-NeXt. The length of this list defines the depth of the network.
            Defaults to `[32, 64, 128, 256]`.

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

        yx_down : List[int], optional
            A list of downsampling factors for the y and x dimensions at each pooling
            stage in the encoder. Its length should match the number of pooling stages (`len(feature_maps) - 1`).
            Defaults to `[2, 2, 2, 2]`.

        output_channels : List[int], optional
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

        upsampling_factor : Tuple[int, ...], optional
            Factor of upsampling for super-resolution workflows. If provided,
            it dictates the kernel and stride for an initial or final transposed
            convolution. Defaults to an empty tuple `()`, meaning no super-resolution.

        upsampling_position : str, optional
            Determines where super-resolution upsampling is applied:
            - ``"pre"``: Upsampling is performed *before* the main U-NeXt model.
            - ``"post"``: Upsampling is performed *after* the main U-NeXt model.
            Defaults to "pre".

        stochastic_depth_prob : float, optional
            Maximum stochastic depth probability. This probability will progressively
            increase with each layer, reaching its maximum value at the bottleneck layer.
            Defaults to 0.1.

        cn_layers : List[int]
            Number of ConvNeXt V2 blocks repeated in each level (stage) of the encoder
            and bottleneck. This list should have the same length as `feature_maps`.
            Defaults to `[2, 2, 2, 2]`.

        isotropy : bool or List[bool], optional
            Controls whether to use 3D or 2D depthwise convolutions at each U-NeXt
            level when the input is 3D.
            - If `True` (bool), all levels use 3D depthwise convolutions.
            - If `False` (bool), all levels use 2D depthwise convolutions (1xKxK kernels for 3D input).
            - If `List[bool]`, specifies for each level whether to use 3D (True) or 2D (False) kernels.
            Defaults to True.

        stem_k_size : int, optional
            Size of the kernel for the initial stem layer's pooling/convolution. Defaults to 2.

        contrast : bool, optional
            Whether to add a contrastive learning projection head to the model.
            If True, an additional output `embed` will be available in the forward pass.
            Defaults to `False`.

        contrast_proj_dim : int, optional
            Dimension of the projection head for contrastive learning, if `contrast` is True.
            Defaults to `256`.

        """
        super(U_NeXt_V2, self).__init__()

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
        layer_norm = nn.LayerNorm
        self.contrast = contrast
        self.explicit_activations = explicit_activations
        if self.explicit_activations:
            assert len(head_activations) == sum(output_channels), "If 'explicit_activations' is True, 'head_activations' needs to "
            "have the same number of values as 'output_channels'"
            self.head_activations, self.class_head_activations = prepare_activation_layers(head_activations, output_channel_info)
            if self.return_class and self.class_head_activations is None:
                raise ValueError("If 'return_class' is True, 'head_activations' must be provided.")

        # convert isotropy to list if it is a single bool
        if type(isotropy) == bool:
            isotropy = [isotropy] * len(feature_maps)

        if self.ndim == 3:
            conv = nn.Conv3d
            convtranspose = nn.ConvTranspose3d
            pre_ln_permutation = Permute([0, 2, 3, 4, 1])
            post_ln_permutation = Permute([0, 4, 1, 2, 3])
            dropout = nn.Dropout3d
        else:
            conv = nn.Conv2d
            convtranspose = nn.ConvTranspose2d
            pre_ln_permutation = Permute([0, 2, 3, 1])
            post_ln_permutation = Permute([0, 3, 1, 2])
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

        self.down_path = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        in_channels = image_shape[-1]

        # STEM
        z_factor = int(max(z_down[0] / stem_k_size, 1))
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
            if not isotropy[i] and self.ndim == 3:
                kernel_size = (1, 7, 7)

            # ConvNeXtBlocks
            for _ in range(cn_layers[i]):
                sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                stage.append(
                    ConvNeXtBlock_V2(self.ndim, conv, feature_maps[i], sd_prob, layer_norm, k_size=kernel_size)
                )
                stage_block_id += 1
                sd_probs_stage.append(sd_prob)
            self.down_path.append(nn.Sequential(*stage))
            sd_probs.append(sd_probs_stage)

            # Downsampling
            mpool = (z_down[i], yx_down[i], yx_down[i]) if self.ndim == 3 else (yx_down[i], yx_down[i])
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
        if not isotropy[-1] and self.ndim == 3:
            kernel_size = (1, 7, 7)
        for _ in range(cn_layers[-1]):
            sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
            stage.append(ConvNeXtBlock_V2(self.ndim, conv, feature_maps[-1], sd_prob, layer_norm, k_size=kernel_size))
            stage_block_id += 1
        self.bottleneck = nn.Sequential(*stage)

        # DECODER
        self.num_decoders = 1 if not separated_decoders else len(output_channels)
        self.up_paths = nn.ModuleList([nn.ModuleList() for _ in range(self.num_decoders)])
        for j in range(self.num_decoders):
            in_channels = feature_maps[-1]
            for i in range(self.depth - 1, -1, -1):
                if not isotropy[i] and self.ndim == 3:
                    kernel_size = (1, 7, 7)
                self.up_paths[j].append(
                    UpConvNeXtBlock_V2(
                        ndim=self.ndim,
                        convtranspose=convtranspose,
                        in_size=in_channels,
                        out_size=feature_maps[i],
                        z_down=z_down[i],
                        yx_down=yx_down[i],
                        up_mode=upsample_layer,
                        conv=conv,
                        attention_gate=False,
                        cn_layers=cn_layers[i],
                        sd_probs=sd_probs[i],
                        layer_norm=layer_norm,
                        k_size=kernel_size,
                    ) # type: ignore
                )
                in_channels = feature_maps[i]

            # Inverted Stem
            mpool = (stem_k_size * z_factor, stem_k_size, stem_k_size) if self.ndim == 3 else (stem_k_size, stem_k_size)
            self.up_paths[j].append(
                nn.Sequential(
                    convtranspose(feature_maps[0], feature_maps[0], kernel_size=mpool, stride=mpool),
                    pre_ln_permutation,
                    layer_norm(feature_maps[0]),
                    post_ln_permutation,
                ) # type: ignore
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

        if self.contrast:
            # extra added layers
            self.heads = nn.Sequential(
                conv(feature_maps[0], feature_maps[0], kernel_size=3, stride=1, padding=1),
                layer_norm(feature_maps[0]),
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

        # Encoder
        blocks = []
        x = self.down_path[0](x)  # (stem)
        for i, layers in enumerate(zip(self.down_path[1:], self.downsample_layers)):
            down, pool = layers
            x = down(x)
            blocks.append(x)
            x = pool(x)

        x_bot = self.bottleneck(x)

        # Decoder
        feats = []
        for j in range(self.num_decoders):
            x = x_bot
            for i, up in enumerate(self.up_paths[j][:-1]):
                x = up(x, blocks[-i - 1])
            x = self.up_paths[j][-1](x)
            feats.append(x)

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
            return out_dict
