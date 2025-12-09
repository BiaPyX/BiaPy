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

from biapy.models.blocks import UpConvNeXtBlock_V2, ConvNeXtBlock_V2, prepare_activation_layers
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

    Parameters
    ----------
    image_shape : Tuple[int, ...]
        Dimensions of the input image. E.g., `(y, x, channels)` for 2D or
        `(z, y, x, channels)` for 3D. The last element `image_shape[-1]`
        should be the number of input channels.

    activation : str, optional
        Activation layer. (Note: ConvNeXt V2 blocks typically use GELU, this parameter
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

    Returns
    -------
    model : nn.Module
        The constructed U-NeXt V2 model.
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
        cn_layers=[2, 2, 2, 2],
        isotropy=True,
        stem_k_size=2,
        contrast: bool = False,
        contrast_proj_dim: int = 256,
        explicit_activations: bool = False,
        activations: list = None,
    ):
        """
        Initialize the U-NeXt_V2 model.

        Sets up the ConvNeXt V2-style encoder (downsampling path), decoder (upsampling path), stem, bottleneck, and optional super-resolution and multi-head output layers.
        It dynamically selects 2D or 3D convolutional and normalization layers based on `ndim` and `isotropy` settings. Stochastic depth probabilities are
        progressively increased across layers.

        Parameters
        ----------
        image_shape : Tuple[int, ...], optional
            Input image dimensions. Defaults to (256, 256, 1).
        feature_maps : List[int], optional
            Number of feature maps at each U-NeXt level. Defaults to `[32, 64, 128, 256]`.
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
        stochastic_depth_prob : float, optional
            Maximum stochastic depth probability. Defaults to 0.1.
        cn_layers : List[int], optional
            Number of ConvNeXt V2 blocks per level. Defaults to `[2, 2, 2, 2]`.
        isotropy : bool | List[bool], optional
            Controls 3D vs 2D depthwise convolutions for 3D input. Defaults to True.
        stem_k_size : int, optional
            Kernel size for the stem layer. Defaults to 2.
        contrast : bool, optional
            Whether to add a contrastive learning projection head. Defaults to `False`.
        contrast_proj_dim : int, optional
            Dimension of the contrastive projection head. Defaults to `256`.
        explicit_activations : bool, optional
            If True, uses explicit activation functions in the last layers.
        activations : List[List[str]], optional
            Activation functions to apply to the outputs if `explicit_activations` is True.
        Raises
        ------
        ValueError
            If 'output_channels' is empty or has more than two values.
        """
        super(U_NeXt_V2, self).__init__()

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
        self.contrast = contrast
        self.explicit_activations = explicit_activations
        if self.explicit_activations:
            self.out_activations, self.class_activation = prepare_activation_layers(activations)

        # convert isotropy to list if it is a single bool
        if type(isotropy) == bool:
            isotropy = isotropy * len(feature_maps)

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
            if isotropy[i] is False and self.ndim == 3:
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
            stage.append(ConvNeXtBlock_V2(self.ndim, conv, feature_maps[-1], sd_prob, layer_norm, k_size=kernel_size))
            stage_block_id += 1
        self.bottleneck = nn.Sequential(*stage)

        # DECODER
        self.up_path = nn.ModuleList()
        in_channels = feature_maps[-1]

        for i in range(self.depth - 1, -1, -1):
            if isotropy[i] is False and self.ndim == 3:
                kernel_size = (1, 7, 7)
            self.up_path.append(
                UpConvNeXtBlock_V2(
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

        if self.contrast:
            # extra added layers
            self.last_block = nn.Sequential(
                conv(feature_maps[0], feature_maps[0], kernel_size=3, stride=1, padding=1),
                layer_norm(feature_maps[0]),
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
        Perform the forward pass of the U-NeXt_V2 model.

        The input `x` first undergoes optional pre-upsampling for super-resolution.
        It then passes through the ConvNeXt V2-style encoder path (stem, ConvNeXt V2 blocks,
        and downsampling layers), followed by a bottleneck. The decoder path upsamples
        features, concatenates them with corresponding skip connections from the encoder,
        and processes them through `UpConvNeXtBlock_V2` modules. Finally, optional
        post-upsampling and the final prediction head(s) are applied.

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

        feats = x
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

    def _init_weights(self, m):
        """
        Initialize the weights of convolutional, linear, and LayerNorm layers.

        Applies Xavier uniform initialization to convolutional and linear layer weights
        (with bias set to 0 if present). For LayerNorm, weights are set to 1.0 and
        biases to 0. This method is typically called using `model.apply()`.

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
