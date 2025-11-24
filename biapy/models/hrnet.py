"""
This file implements the High-Resolution Net (HRNet) model and its core building blocks,
designed for dense prediction tasks in 2D and 3D imaging.

The HRNet architecture maintains high-resolution representations throughout the
network by connecting high-to-low resolution convolution streams in parallel
and facilitating repeated information exchange across these streams.

Key components:

- ``HighResolutionNet``: The main HRNet model.
- ``HighResolutionModule``: Core HRNet building block that manages multi-resolution fusion.
- ``HRBasicBlock``: Basic residual block for HRNet.
- ``HRBottleneck``: Bottleneck residual block for HRNet.

Reference:  
`Deep High-Resolution Representation Learning for Visual Recognition <https://ieeexplore.ieee.org/abstract/document/9052469>`_

Code adapted from:  
`Exploring Cross-Image Pixel Contrast for Semantic Segmentation <https://github.com/tfzhou/ContrastiveSeg/tree/main>`_
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Type

from biapy.models.blocks import HRBasicBlock, HRBottleneck, ConvBlock, get_norm_3d, get_norm_2d, ProjectionHead

class HighResolutionNet(nn.Module):
    """
    Implements a 2D/3D High-Resolution Net (HRNet) model.

    HRNet is a convolutional neural network architecture designed to maintain
    high-resolution representations throughout the network. It achieves this
    by employing parallel high-to-low resolution convolution streams and
    repeatedly exchanging information across these streams. This design
    is particularly effective for dense prediction tasks like semantic
    segmentation, instance segmentation, and object detection, where
    preserving spatial detail is crucial.

    Reference: `Deep High-Resolution Representation Learning for Visual Recognition <https://ieeexplore.ieee.org/abstract/document/9052469>`_.

    Code adapted from: `Exploring Cross-Image Pixel Contrast for Semantic Segmentation <https://github.com/tfzhou/ContrastiveSeg/tree/main>`_.

    Parameters
    ----------
    cfg : Dict
        HRNet configuration dictionary. Expected keys define the network structure:
        * ``NUM_MODULES`` (int): Number of modules within each stage.
        * ``NUM_BRANCHES`` (int): Number of parallel branches (resolution streams) in a stage.
        * ``NUM_BLOCKS`` (List[int]): List specifying the number of blocks per branch.
        * ``NUM_CHANNELS`` (List[int]): List specifying the number of channels for each branch.
        * ``BLOCK`` (str): Type of building block, e.g., 'BASIC' for `HRBasicBlock` or 'BOTTLENECK' for `HRBottleneck`.
        * ``Z_DOWN`` (bool): For 3D HRNet, whether to downsample the z-axis (True) or keep its original resolution (False).

    image_shape : Tuple[int, ...]
        Dimensions of the input image. E.g., `(y, x, channels)` for 2D or `(z, y, x, channels)` for 3D.
        The last element `image_shape[-1]` should be the number of input channels.

    normalization : str, optional
        Type of normalization layer to use throughout the network. Options include
        `'bn'` (Batch Normalization), `'sync_bn'` (Synchronized Batch Normalization for multi-GPU),
        `'in'` (Instance Normalization), `'gn'` (Group Normalization), or `'none'`.
        Defaults to "none".

    output_channels : List[int], optional
        Specifies the number of output channels for the final prediction head(s).
        Must be a list of length 1 for a single output task (e.g., semantic segmentation)
        or length 2 for multi-head tasks (e.g., instances + classification in instance segmentation,
        or points + classification in detection). Defaults to `[1]`.

    contrast : bool, optional
        If True, an additional projection head (`ProjectionHead`) is created to generate
        an embedding suitable for contrastive learning. Defaults to False.

    contrast_proj_dim : int, optional
        The output dimension of the projection embedding when `contrast` is True. Defaults to 256.

    Returns
    -------
    model : nn.Module
        The constructed HRNet model.
    """

    def __init__(
        self,
        cfg: Dict,
        image_shape: Tuple[int, ...] = (256, 256, 1),
        normalization: str = "none",
        output_channels: List[int] = [1],
        contrast: bool = False,
        contrast_proj_dim: int = 256,
    ):
        """
        Initialize the HighResolutionNet model.

        Configures the HRNet architecture based on the provided parameters,
        setting up convolutional layers, normalization types, and the
        multi-resolution stages. It also prepares for optional contrastive
        learning and multi-head outputs.

        Parameters
        ----------
        cfg : Dict
            HRNet configuration dictionary, detailing module, branch, block, and channel counts,
            block type, and z-axis downsampling for 3D.
        image_shape : Tuple[int, ...], optional
            Input image dimensions, used to determine 2D/3D mode and input channels.
            Defaults to (256, 256, 1).
        normalization : str, optional
            Type of normalization layer (e.g., 'bn', 'sync_bn', 'none'). Defaults to "none".
        output_channels : List[int], optional
            Number of channels for the output head(s). Supports one or two outputs.
            Defaults to [1].
        contrast : bool, optional
            If True, enables a projection head for contrastive learning. Defaults to False.
        contrast_proj_dim : int, optional
            Output dimension for the contrastive projection head. Defaults to 256.

        Raises
        ------
        ValueError
            If 'output_channels' is empty or has more than two values.
        """
        if len(output_channels) == 0:
            raise ValueError("'output_channels' needs to has at least one value")
        if len(output_channels) != 1 and len(output_channels) != 2:
            raise ValueError(f"'output_channels' must be a list of one or two values at max, not {output_channels}")

        self.blocks_dict = {"BASIC": HRBasicBlock, "BOTTLENECK": HRBottleneck}
        self.output_channels = output_channels
        self.multihead = len(output_channels) == 2
        self.in_size = 64
        self.ndim = 3 if len(image_shape) == 4 else 2
        self.contrast = contrast
        self.z_down = cfg["Z_DOWN"]

        if self.ndim == 3:
            self.conv_call = nn.Conv3d
            self.norm_func = get_norm_3d
            self.dropout = nn.Dropout3d
            mpool = (2, 2, 2) if self.z_down else (1, 2, 2)
        else:
            self.conv_call = nn.Conv2d
            self.norm_func = get_norm_2d
            self.dropout = nn.Dropout2d
            mpool = (2, 2)

        in_channels = image_shape[-1]

        super(HighResolutionNet, self).__init__()
        self.conv1_block = ConvBlock(
            conv=self.conv_call,
            in_size=in_channels,
            out_size=64,
            k_size=3,
            padding=1,
            stride=mpool,
            act="none",
            norm=normalization,
            bias=False,
        )
        self.conv2_block = ConvBlock(
            conv=self.conv_call,
            in_size=64,
            out_size=64,
            k_size=3,
            padding=1,
            stride=mpool,
            act="relu",
            norm=normalization,
            bias=False,
        )
        self.layer1 = self._make_layer(HRBottleneck, 64, 64, 4, norm=normalization)

        self.stage2_cfg = cfg["STAGE2"]
        num_channels = self.stage2_cfg["NUM_CHANNELS"]
        block = self.blocks_dict[self.stage2_cfg["BLOCK"]]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]

        self.transition1 = self._make_transition_layer([256], num_channels, norm=normalization, mpool=mpool)

        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels, norm=normalization, mpool=mpool
        )

        self.stage3_cfg = cfg["STAGE3"]
        num_channels = self.stage3_cfg["NUM_CHANNELS"]
        block = self.blocks_dict[self.stage3_cfg["BLOCK"]]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels, norm=normalization, mpool=mpool
        )
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels, norm=normalization, mpool=mpool
        )

        self.stage4_cfg = cfg["STAGE4"]
        num_channels = self.stage4_cfg["NUM_CHANNELS"]
        block = self.blocks_dict[self.stage4_cfg["BLOCK"]]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels, norm=normalization, mpool=mpool
        )

        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True, norm=normalization, mpool=mpool
        )

        in_channels = sum(self.stage4_cfg["NUM_CHANNELS"])
        if self.contrast:
            # extra added layers    
            self.last_block = nn.Sequential(
                self.conv_call(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                self.norm_func(normalization, in_channels),
                self.dropout(0.10),
                self.conv_call(in_channels, self.output_channels[0], kernel_size=1, stride=1, padding=0, bias=False),
            )

            self.proj_head = ProjectionHead(ndim=self.ndim, in_channels=in_channels, proj_dim=contrast_proj_dim)
        else:
            self.last_block = self.conv_call(in_channels, self.output_channels[0], kernel_size=1, padding="same")

        # Multi-head:
        #   Instance segmentation: instances + classification
        #   Detection: points + classification
        self.last_class_head = None
        if self.multihead:
            self.last_class_head = self.conv_call(in_channels, self.output_channels[1], kernel_size=1, padding="same")

    def _make_transition_layer(
        self,
        num_channels_pre_layer: List[int],
        num_channels_cur_layer: List[int],
        norm: str,
        mpool: Tuple[int, ...] = (2, 2),
    ):
        """
        Create transition layers between stages of the HRNet.

        These layers handle the transition of feature maps between stages that might
        have different numbers of branches or different channel configurations.
        They include convolutional blocks to adjust channels and spatial dimensions.

        Parameters
        ----------
        num_channels_pre_layer : List[int]
            Number of channels in the previous layer.

        num_channels_cur_layer : List[int]
            Number of channels in the current layer.

        norm : str
            Normalization layer to use (one of 'bn', 'sync_bn', 'in', 'gn', or 'none').

        mpool : Tuple[int, ...], optional
            Downsampling factor for the pooling operation. Used to downsample the features. Default is (2, 2).

        Returns
        -------
        transition_layers : nn.ModuleList
            List of transition layers between the previous and current layers.
        """
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        ConvBlock(
                            conv=self.conv_call,
                            in_size=num_channels_pre_layer[i],
                            out_size=num_channels_cur_layer[i],
                            k_size=3,
                            padding=1,
                            stride=1,
                            act="relu",
                            norm=norm,
                            bias=False,
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(
                        ConvBlock(
                            conv=self.conv_call,
                            in_size=inchannels,
                            out_size=outchannels,
                            k_size=3,
                            padding=1,
                            stride=mpool,
                            act="relu",
                            norm=norm,
                            bias=False,
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(
        self,
        block: Type[HRBasicBlock | HRBottleneck],
        in_size: int,
        out_size: int,
        blocks: int,
        stride: int = 1,
        norm: str = "none",
    ):
        """
        Construct a sequential layer consisting of multiple HRNet building blocks.

        This method generates a sequence of `HRBasicBlock` or `HRBottleneck` instances,
        optionally including a downsampling projection for the first block if input/output
        dimensions or strides differ.

        Parameters
        ----------
        block : Type[HRBasicBlock | HRBottleneck]
            Type of block to use in the layer (either HRBasicBlock or HRBottleneck).

        in_size : int
            Number of input channels for the first block in the layer.

        out_size : int
            Number of output channels for the blocks in the layer.

        blocks : int
            Number of blocks to create in the layer.

        stride : int, optional
            Stride of the first convolutional layer in the layer. Default is 1.

        norm : str, optional
            Normalization layer to use (one of 'bn', 'sync_bn', 'in',
            'gn', or 'none'). Default is 'none'.    ç

        Returns
        -------
        layer : nn.Sequential
            Sequential container with the blocks of the layer.
        """
        downsample = None
        if stride != 1 or in_size != out_size * block.expansion:
            downsample = nn.Sequential(
                self.conv_call(in_size, out_size * block.expansion, kernel_size=1, stride=stride, bias=False),
                self.norm_func(norm, out_size * block.expansion),
            )

        layers = []
        layers.append(block(self.conv_call, in_size, out_size, stride, downsample=downsample, norm=norm))

        in_size = out_size * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.conv_call, in_size, out_size, norm=norm))

        return nn.Sequential(*layers)

    def _make_stage(
        self,
        layer_config,
        num_inchannels,
        multi_scale_output=True,
        norm="none",
        mpool: Tuple[int, ...] = (2, 2),
    ):
        """
        Construct a full stage of the HRNet, consisting of multiple HighResolutionModule instances.

        Each stage of HRNet typically involves multiple parallel branches at different
        resolutions, with information exchange between them. This method creates the modules
        that manage these branches and their interactions.

        Parameters
        ----------
        layer_config : Dict
            Configuration dictionary for the stage. Expected keys are:
                * ``NUM_MODULES``, int: number of modules to create
                * ``NUM_BRANCHES``, int: number of branches in the stage
                * ``NUM_BLOCKS``, List[int]: Number of blocks per branch
                * ``NUM_CHANNELS``, List[int]: Number of channels per branch
                * ``BLOCK``, str: block type. Options: ['BASIC', "BOTTLENECK"]

        num_inchannels : List[int]
            Number of input channels for each branch in the stage.

        multi_scale_output : bool, optional
            Whether to output features at multiple scales or not. Default is True.

        norm : str, optional
            Normalization layer to use (one of 'bn', 'sync_bn', 'in',
            'gn', or 'none'). Default is 'none'.

        mpool : Tuple[int, ...], optional
            Downsampling factor for the pooling operation. Used to downsample the features. Default is (2, 2).

        Returns
        -------
        modules : nn.Sequential
            Sequential container with the modules of the stage.

        num_inchannels : List[int]
            Number of input channels for the next stage.
        """
        num_modules = layer_config["NUM_MODULES"]
        num_branches = layer_config["NUM_BRANCHES"]
        num_blocks = layer_config["NUM_BLOCKS"]
        num_channels = layer_config["NUM_CHANNELS"]
        block = self.blocks_dict[layer_config["BLOCK"]]

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    ndim=self.ndim,
                    num_branches=num_branches,
                    blocks=block,
                    num_blocks=num_blocks,
                    num_inchannels=num_inchannels,
                    num_channels=num_channels,
                    multi_scale_output=reset_multi_scale_output,
                    norm=norm,
                    mpool=mpool,
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x) -> Dict | torch.Tensor:
        """
        Perform the forward pass of the HighResolutionNet.

        The input `x` first goes through initial convolutional blocks. Then, it
        propagates through a series of HRNet stages, where feature maps are
        processed in parallel across multiple resolutions and information is
        exchanged. Finally, features from all resolutions are fused, and passed
        through a final prediction head. Optionally, a contrastive learning
        projection head and/or a multi-head classification output can be included.

        Parameters
        ----------
        x : torch.Tensor
            The input image tensor.
            Expected shape for 2D: `(batch_size, channels, height, width)`.
            Expected shape for 3D: `(batch_size, channels, depth, height, width)`.

        Returns
        -------
        Dict or torch.Tensor
            If `contrast` is True or `multihead` is True, returns a dictionary
            containing output tensors (e.g., 'pred', 'embed', 'class').
            Otherwise, returns a single prediction tensor.
        """
        x = self.conv1_block(x)
        x = self.conv2_block(x)

        x = self.layer1(x)
        x_list = []
        for i in range(self.stage2_cfg["NUM_BRANCHES"]):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg["NUM_BRANCHES"]):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        if os.environ.get("drop_stage4"):
            return y_list

        x_list = []
        for i in range(self.stage4_cfg["NUM_BRANCHES"]):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        feat1 = y_list[0]

        if feat1.ndim == 4:
            _, _, h, w = y_list[0].size()
            feat2 = F.interpolate(y_list[1], size=(h, w), mode="bilinear", align_corners=True)
            feat3 = F.interpolate(y_list[2], size=(h, w), mode="bilinear", align_corners=True)
            feat4 = F.interpolate(y_list[3], size=(h, w), mode="bilinear", align_corners=True)
        else:
            _, _, d, h, w = y_list[0].size()
            feat2 = F.interpolate(y_list[1], size=(d, h, w), mode="trilinear", align_corners=True)
            feat3 = F.interpolate(y_list[2], size=(d, h, w), mode="trilinear", align_corners=True)
            feat4 = F.interpolate(y_list[3], size=(d, h, w), mode="trilinear", align_corners=True)

        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        out = self.last_block(feats)
        out_dict = {
            "pred": out,
        }
        if self.contrast:
            emb = self.proj_head(feats)
            out_dict["embed"] = emb

        if self.multihead and self.last_class_head:
            class_head_out = self.last_class_head(feats)
            out_dict["class"] = class_head_out

        if len(out_dict.keys()) == 1:
            return out_dict["pred"]
        else:
            return out_dict


class HighResolutionModule(nn.Module):
    """
    Implements the High Resolution Module for HRNet.

    This module is a core building block of the HRNet architecture. It consists
    of multiple parallel convolutional branches at different resolutions, along
    with fusion layers that enable rich information exchange across these
    branches. This design helps maintain high-resolution representations
    throughout the network.

    Pararameters
    ------------
    ndim : int
        Number of dimensions of the input data (2 for 2D, 3 for 3D).

    num_branches : int
        Number of branches in the module.

    blocks : Type[HRBasicBlock | HRBottleneck]
        Type of block to use in the module (either HRBasicBlock or HRBottleneck).

    num_blocks : List[int]
        Number of blocks in each branch.

    num_inchannels : List[int]
        Number of input channels for each branch.

    num_channels : List[int]
        Number of output channels for each branch.

    multi_scale_output : bool
        Whether to output features at multiple scales or not.

    norm : str
        Normalization layer to use (one of 'bn', 'sync_bn', 'in', 'gn', or 'none').

    mpool : Tuple[int, ...]
        Downsampling factor for the pooling operation. Used to downsample the features.
    """

    def __init__(
        self,
        ndim: int,
        num_branches: int,
        blocks: Type[HRBasicBlock | HRBottleneck],
        num_blocks: List[int],
        num_inchannels: List[int],
        num_channels: List[int],
        multi_scale_output: bool = True,
        norm: str = "none",
        mpool: Tuple[int, ...] = (2, 2),
    ):
        """
        Initialize a High Resolution Module.

        Sets up the parallel branches with their respective convolutional blocks
        and constructs the information fusion layers that allow features to be
        exchanged across different resolutions. It also determines the
        appropriate convolutional and normalization layers based on the
        dimensionality (`ndim`).

        Parameters
        ----------
        ndim : int
            Number of spatial dimensions of the input (2 for 2D, 3 for 3D).
        num_branches : int
            The number of parallel resolution branches within this module.
        blocks : Type[HRBasicBlock | HRBottleneck]
            The class of residual block to be used within the branches.
        num_blocks : List[int]
            A list where each element specifies the number of `blocks` for the
            corresponding branch. Its length must match `num_branches`.
        num_inchannels : List[int]
            A list where each element specifies the input channel count for the
            corresponding branch. Its length must match `num_branches`.
        num_channels : List[int]
            A list where each element specifies the output channel count for the
            corresponding branch after processing through its blocks. Its length
            must match `num_branches`.
        multi_scale_output : bool, optional
            If True, the module's forward pass will output features at all scales
            by fusing and returning all branch outputs. If False, only a single
            (typically high-resolution) output might be expected, depending on
            subsequent processing. Defaults to True.
        norm : str, optional
            The type of normalization layer to apply within the module's blocks
            and fusion layers (e.g., 'bn', 'sync_bn', 'in', 'gn', 'none').
            Defaults to "none".
        mpool : Tuple[int, ...], optional
            The downsampling factor for max-pooling operations, primarily used
            in the fusion layers when features from higher resolution branches
            are downsampled to match lower resolution ones. Defaults to (2, 2).

        Raises
        ------
        ValueError
            If the lengths of `num_blocks`, `num_inchannels`, or `num_channels`
            do not match `num_branches`.
        """
        super(HighResolutionModule, self).__init__()
        self.ndim = ndim
        if self.ndim == 3:
            self.conv_call = nn.Conv3d
            self.norm_func = get_norm_3d
        else:
            self.conv_call = nn.Conv2d
            self.norm_func = get_norm_2d
        self._check_branches(num_branches, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels, norm=norm)
        self.fuse_layers = self._make_fuse_layers(norm=norm, mpool=mpool)
        self.relu = nn.ReLU(inplace=False)

    def _check_branches(
        self, num_branches: int, num_blocks: List[int], num_inchannels: List[int], num_channels: List[int]
    ):
        """
        Check if the number of branches, blocks, input channels and output channels are consistent.

        Parameters
        ----------
        num_branches : int
            Number of branches in the module.

        num_blocks : List[int]
            Number of blocks in each branch.

        num_inchannels : List[int]
            Number of input channels for each branch.

        num_channels : List[int]
            Number of output channels for each branch.
        """
        if num_branches != len(num_blocks):
            error_msg = "NUM_BRANCHES({}) <> NUM_BLOCKS({})".format(num_branches, len(num_blocks))
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = "NUM_BRANCHES({}) <> NUM_CHANNELS({})".format(num_branches, len(num_channels))
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = "NUM_BRANCHES({}) <> NUM_INCHANNELS({})".format(num_branches, len(num_inchannels))
            raise ValueError(error_msg)

    def _make_one_branch(
        self,
        branch_index: int,
        block: Type[HRBasicBlock | HRBottleneck],
        num_blocks: List[int],
        num_channels: List[int],
        stride: int = 1,
        norm: str = "none",
    ):
        """
        Create one branch of the High Resolution Module.

        Parameters
        ----------
        branch_index : int
            Index of the branch to create.

        block : Type[HRBasicBlock | HRBottleneck]
            Type of block to use in the branch (either HRBasicBlock or HRBottleneck).

        num_blocks : List[int]
            Number of blocks in the branch.

        num_channels : List[int]
            Number of output channels for the branch.

        stride : int, optional
            Stride of the first convolutional layer in the branch. Default is 1.

        norm : str, optional
            Normalization layer to use (one of 'bn', 'sync_bn', 'in', 'gn', or 'none'). Default is 'none'.
        """
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                self.conv_call(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                self.norm_func(norm, num_channels[branch_index] * block.expansion),
            )

        layers = []
        layers.append(
            block(
                conv=self.conv_call,
                in_size=self.num_inchannels[branch_index],
                out_size=num_channels[branch_index] * block.expansion,
                stride=stride,
                norm=norm,
                downsample=downsample,
            )
        )
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    conv=self.conv_call,
                    in_size=self.num_inchannels[branch_index],
                    out_size=num_channels[branch_index],
                    norm=norm,
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(
        self,
        num_branches: int,
        block: Type[HRBasicBlock | HRBottleneck],
        num_blocks: List[int],
        num_channels: List[int],
        norm: str,
    ):
        """
        Create branches for the High Resolution Module.

        Parameters
        ----------
        num_branches : int
            Number of branches to create.

        block : Type[HRBasicBlock | HRBottleneck]
            Type of block to use in the branches (either HRBasicBlock or HRBottleneck).

        num_blocks : List[int]
            Number of blocks in each branch.

        num_channels : List[int]
            Number of output channels for each branch.

        norm : str
            Normalization layer to use (one of 'bn', 'sync_bn', 'in', 'gn', or 'none').

        Returns
        -------
        branches : nn.ModuleList
            List of branches created for the High Resolution Module.
        """
        branches = []

        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels, norm=norm))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self, norm: str, mpool: Tuple[int, ...] = (2, 2)):
        """
        Construct the fusion layers for exchanging information between branches.

        These layers enable the High Resolution Module to repeatedly fuse outputs
        from parallel branches at different resolutions. They consist of
        convolutional operations and optional upsampling/downsampling to align
        feature map dimensions for element-wise summation.

        Parameters
        ----------
        norm : str
            Normalization layer type to use within the fusion layers.
        mpool : Tuple[int, ...], optional
            Downsampling factor for pooling operations when features are downsampled
            from a higher resolution branch to a lower resolution one during fusion.
            Defaults to (2, 2).

        Returns
        -------
        fuse_layers : nn.ModuleList or None
            A list of lists of `nn.Module` (or `None` for identity connections)
            representing the fusion operations. Returns `None` if `num_branches` is 1.
        """
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        ConvBlock(
                            conv=self.conv_call,
                            in_size=num_inchannels[j],
                            out_size=num_inchannels[i],
                            k_size=1,
                            padding=0,
                            stride=1,
                            norm=norm,
                            bias=False,
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                ConvBlock(
                                    conv=self.conv_call,
                                    in_size=num_inchannels[j],
                                    out_size=num_outchannels_conv3x3,
                                    k_size=3,
                                    padding=1,
                                    stride=mpool,
                                    norm=norm,
                                    bias=False,
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                ConvBlock(
                                    conv=self.conv_call,
                                    in_size=num_inchannels[j],
                                    out_size=num_outchannels_conv3x3,
                                    k_size=3,
                                    padding=1,
                                    stride=mpool,
                                    act="relu",
                                    norm=norm,
                                    bias=False,
                                ),
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        """
        Retrieve the current number of input channels for each branch.

        This method provides access to the dynamically updated `num_inchannels`
        list, which reflects the channel counts of features after they have
        passed through the respective blocks within this module. This is useful
        for configuring subsequent stages or modules.

        Returns
        -------
        List[int]
            A list where each element represents the number of channels for
            the corresponding branch's output.
        """
        return self.num_inchannels

    def forward(self, x):
        """
        Perform the forward pass of the High Resolution Module.

        The input is a list of tensors, where each tensor corresponds to a
        feature map from a parallel resolution branch. Each feature map
        first passes through its respective branch's convolutional blocks.
        Then, the outputs from all branches are fused by upsampling or
        downsampling as necessary, followed by element-wise summation to
        create new feature maps at each target resolution.

        Parameters
        ----------
        x : List[torch.Tensor]
            A list of input feature tensors, where each tensor corresponds to
            a different resolution branch. The order typically goes from highest
            to lowest resolution.

        Returns
        -------
        List[torch.Tensor]
            A list of output feature tensors, representing the fused and
            processed features at potentially multiple scales. If
            `multi_scale_output` is True, the list will contain features
            for all output resolutions; otherwise, it might contain only
            the highest resolution output.
        """
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    if x[i].ndim == 4:
                        y = y + F.interpolate(
                            self.fuse_layers[i][j](x[j]),
                            size=[height_output, width_output],
                            mode="bilinear",
                            align_corners=True,
                        )
                    else:
                        depth_output = x[i].shape[-3]
                        y = y + F.interpolate(
                            self.fuse_layers[i][j](x[j]),
                            size=[depth_output, height_output, width_output],
                            mode="trilinear",
                            align_corners=True,
                        )
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse

class HighResolutionNext(nn.Module):
    """
    Implements the High Resolution Next model for HRNet.

    Parameters
    ----------
    cfg : Dict
        HRNet configuration. Expected keys are:
            * ``STAGE1``, Dict: configuration for stage 1
            * ``STAGE2``, Dict: configuration for stage 2
            * ``STAGE3``, Dict: configuration for stage 3
            * ``STAGE4``, Dict: configuration for stage 4

    norm : str, optional
        Normalization layer to use (one of 'bn', 'sync_bn', 'in', 'gn', or 'none'). Default is 'none'.

    ndim : int, optional
        Number of dimensions of the input data (2 for 2D, 3 for 3D). Default is 2.
    
    z_down : bool, optional
        Whether to downsample the z-axis or not. If ``False`` it will not downsample the z-axis. Default is ``False``.
    """

    def __init__(self, cfg, norm="none", ndim=2, z_down=False):
        """
        Initialize the High Resolution Next (HRNet) model.

        Constructs the entire HRNet architecture, including the stem, and
        sequential stages (Stage 1 to Stage 4), each potentially composed of
        multiple High Resolution Modules and transition layers. The model
        dynamically adapts its convolutional and normalization layers based
        on the input data's dimensionality (`ndim`).

        Parameters
        ----------
        cfg : Dict
            A dictionary containing the full configuration for the HRNet model.
            It must specify the architecture details for each stage (STAGE1,
            STAGE2, STAGE3, STAGE4), including the number of modules, branches,
            blocks, and channels, as well as the block type to use.
        norm : str, optional
            The type of normalization layer to use throughout the network.
            Options include 'bn' (BatchNorm), 'sync_bn' (SyncBatchNorm),
            'in' (InstanceNorm), 'gn' (GroupNorm), or 'none'. Defaults to "none".
        ndim : int, optional
            The number of spatial dimensions of the input data. Use 2 for 2D
            data (e.g., images) and 3 for 3D data (e.g., volumetric scans).
            Defaults to 2.
        z_down : bool, optional
            Applicable only when `ndim` is 3. If True, the z-axis (depth) will
            also be downsampled during pooling operations in the stem and
            transition layers. If False, downsampling will only occur in the
            x and y dimensions, preserving the z-resolution. Defaults to False.
        """
        super(HighResolutionNext, self).__init__()
        self.ndim = ndim
        self.z_down = z_down
        if self.ndim == 3:
            self.conv_call = nn.Conv3d
            self.norm_func = get_norm_3d
            self.dropout = nn.Dropout3d
            mpool = (2, 2, 2) if self.z_down else (1, 2, 2)
        else:
            self.conv_call = nn.Conv2d
            self.norm_func = get_norm_2d
            self.dropout = nn.Dropout2d
            mpool = (2, 2)

        # stem net
        self.conv1 = self.conv_call(3, 64, kernel_size=3, stride=mpool, padding=1, bias=False)
        self.bn1 = self.norm_func(norm, 64)
        self.relu = nn.ReLU()

        self.stage1_cfg = cfg["STAGE1"]
        num_channels = self.stage1_cfg["NUM_CHANNELS"]
        block = self.blocks_dict[self.stage1_cfg["BLOCK"]]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition0 = self._make_transition_layer([64], num_channels, norm=norm, mpool=mpool)
        self.stage1, pre_stage_channels = self._make_stage(self.stage1_cfg, num_channels, norm=norm, mpool=mpool)

        self.stage2_cfg = cfg["STAGE2"]
        num_channels = self.stage2_cfg["NUM_CHANNELS"]
        block = self.blocks_dict[self.stage2_cfg["BLOCK"]]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(pre_stage_channels, num_channels, norm=norm, mpool=mpool)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels, norm=norm, mpool=mpool)

        self.stage3_cfg = cfg["STAGE3"]
        num_channels = self.stage3_cfg["NUM_CHANNELS"]
        block = self.blocks_dict[self.stage3_cfg["BLOCK"]]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels, norm=norm, mpool=mpool)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels, norm=norm, mpool=mpool)

        self.stage4_cfg = cfg["STAGE4"]
        num_channels = self.stage4_cfg["NUM_CHANNELS"]
        block = self.blocks_dict[self.stage4_cfg["BLOCK"]]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels, norm=norm, mpool=mpool)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True, norm=norm, mpool=mpool
        )

    def _make_transition_layer(
        self,
        num_channels_pre_layer: List[int],
        num_channels_cur_layer: List[int],
        norm: str,
        mpool: Tuple[int, ...] = (2, 2),
    ):
        """
        Create transition layers between stages of the HRNet.

        Parameters
        ----------
        num_channels_pre_layer : List[int]
            Number of channels in the previous layer.

        num_channels_cur_layer : List[int]
            Number of channels in the current layer.

        norm : str
            Normalization layer to use (one of 'bn', 'sync_bn', 'in', 'gn', or 'none').

        mpool : Tuple[int, ...], optional
            Downsampling factor for the pooling operation. Used to downsample the features. Default is (2, 2).

        Returns
        -------
        transition_layers : nn.ModuleList
            List of transition layers between the previous and current layers.

        """
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            self.conv_call(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False),
                            self.norm_func(norm, num_channels_cur_layer[i]),
                            nn.ReLU(),
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            self.conv_call(inchannels, outchannels, 3, mpool, 1, bias=False),
                            self.norm_func(norm, outchannels),
                            nn.ReLU(),
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_stage(
        self,
        layer_config: Dict,
        num_inchannels: List[int],
        multi_scale_output: bool = True,
        norm: str = "none",
        mpool: Tuple[int, ...] = (2, 2),
    ):
        """
        Create a stage of the HRNet.

        Parameters
        ----------
        layer_config : Dict
            Configuration dictionary for the stage. Expected keys are:
                * ``NUM_MODULES``, int: number of modules to create
                * ``NUM_BRANCHES``, int: number of branches in the stage
                * ``NUM_BLOCKS``, List[int]: Number of blocks per branch
                * ``NUM_CHANNELS``, List[int]: Number of channels per branch
                * ``BLOCK``, str: block type. Options: ['BASIC', "BOTTLENECK"]

        num_inchannels : List[int]
            Number of input channels for each branch in the stage.

        multi_scale_output : bool, optional
            Whether to output features at multiple scales or not. Default is True.

        norm : str, optional
            Normalization layer to use (one of 'bn', 'sync_bn', 'in', 'gn', or 'none'). Default is 'none'.

        mpool : Tuple[int, ...], optional
            Downsampling factor for the pooling operation. Used to downsample the features. Default is (2, 2).

        Returns
        -------
        modules : nn.Sequential
            Sequential container with the modules of the stage.

        num_inchannels : List[int]
            Number of input channels for the next stage.
        """
        num_modules = layer_config["NUM_MODULES"]
        num_branches = layer_config["NUM_BRANCHES"]
        num_blocks = layer_config["NUM_BLOCKS"]
        num_channels = layer_config["NUM_CHANNELS"]
        self.blocks_dict = {"BASIC": HRBasicBlock, "BOTTLENECK": HRBottleneck}
        block = self.blocks_dict[layer_config["BLOCK"]]

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    self.ndim,
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    reset_multi_scale_output,
                    norm,
                    mpool,
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        """
        Perform the forward pass through the High Resolution Next (HRNet) model.

        The input tensor first passes through the stem network (initial
        convolution, batch normalization, and ReLU activation). Then, it
        progresses through multiple stages (Stage 1 to Stage 4). Each stage
        begins with a transition layer that prepares the feature maps for
        the multi-resolution High Resolution Modules within that stage.
        Features are maintained and exchanged across different resolutions
        throughout these stages, ultimately returning a list of multi-scale
        feature maps from the final stage.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to the network, typically an image or volumetric data.
            Expected shape for 2D: (Batch, Channels, Height, Width)
            Expected shape for 3D: (Batch, Channels, Depth, Height, Width)

        Returns
        -------
        List[torch.Tensor]
            A list of output feature tensors, where each tensor corresponds to
            a different resolution branch from the final stage (Stage 4).
            The list is ordered from highest to lowest resolution.
        """
        # Stem network
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Stage 1
        x_list = []
        for i in range(self.stage1_cfg["NUM_BRANCHES"]):
            # Apply transition0 layer if it exists for the current branch, otherwise use x directly
            if self.transition0[i] is not None:
                x_list.append(self.transition0[i](x))
            else:
                x_list.append(x)
        # Pass the list of features through Stage 1 modules
        y_list = self.stage1(x_list)

        # Stage 2
        x_list = []
        for i in range(self.stage2_cfg["NUM_BRANCHES"]):
            if self.transition1[i] is not None:
                # For the first branch (highest resolution), transition from y_list[0]
                # For subsequent branches, transition from the last (lowest resolution) feature map of previous stage
                if i == 0:
                    x_list.append(self.transition1[i](y_list[0]))
                else:
                    x_list.append(self.transition1[i](y_list[-1]))
            else:
                # If no transition layer, use the corresponding branch's output from the previous stage
                x_list.append(y_list[i])
        # Pass the list of features through Stage 2 modules
        y_list = self.stage2(x_list)

        # Stage 3
        x_list = []
        for i in range(self.stage3_cfg["NUM_BRANCHES"]):
            if self.transition2[i] is not None:
                # Transition always from the last (lowest resolution) feature map of the previous stage
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                # If no transition layer, use the corresponding branch's output from the previous stage
                x_list.append(y_list[i])
        # Pass the list of features through Stage 3 modules
        y_list = self.stage3(x_list)

        # Stage 4 (final stage)
        x_list = []
        for i in range(self.stage4_cfg["NUM_BRANCHES"]):
            if self.transition3[i] is not None:
                # Transition always from the last (lowest resolution) feature map of the previous stage
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                # If no transition layer, use the corresponding branch's output from the previous stage
                x_list.append(y_list[i])
        # Pass the list of features through Stage 4 modules
        x = self.stage4(x_list)
        return x
