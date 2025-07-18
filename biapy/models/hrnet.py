import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Type

from biapy.models.blocks import HRBasicBlock, HRBottleneck, ConvBlock, get_norm_3d, get_norm_2d, ProjectionHead


blocks_dict = {"BASIC": HRBasicBlock, "BOTTLENECK": HRBottleneck}


class HighResolutionNet(nn.Module):
    """
    Create 2D/3D HRNet.

    Reference: `Deep High-Resolution Representation Learning for Visual Recognition <https://ieeexplore.ieee.org/abstract/document/9052469>`_.

    Code adapted from: `Exploring Cross-Image Pixel Contrast for Semantic Segmentation <https://github.com/tfzhou/ContrastiveSeg/tree/main>`_.

    Parameters
    ----------
    cfg : Dict
        HRNet configuration. Exoected keys are:
            * ``NUM_MODULES``, int: number of modules to create
            * ``NUM_BRANCHES``, int: number of modules to create
            * ``NUM_BLOCKS``, List[int]: Number of blocks per branch
            * ``NUM_CHANNELS``, List[int]: Number of channels per branch
            * ``BLOCK``, str: block type. Options: ['BASIC', "BOTTLENECK"]
            * ``Z_DOWN``, bool: whether to downsample the z-axis or not. If ``False`` it will not downsample the z-axis.

    image_shape : 3D/4D tuple
        Dimensions of the input image. E.g. ``(y, x, channels)`` or ``(z, y, x, channels)``.

    normalization : str, optional
        Normalization layer (one of ``'bn'``, ``'sync_bn'`` ``'in'``, ``'gn'`` or ``'none'``).

    z_down : bool, optional 
        Whether to downsample the z-axis or not. If ``False`` it will not downsample the z-axis. Default is ``False``.
        
    output_channels : list of int, optional
        Output channels of the network. It must be a list of lenght ``1`` or ``2``. When two
        numbers are provided two task to be done is expected (multi-head). Possible scenarios are:
            * instances + classification on instance segmentation
            * points + classification in detection.

    contrast: bool
        Whether to create a projection embedding for contrastive learning.

    contrast_proj_dim : int
        Dimensions of the projection embedding.
    Returns
    -------
    model : Torch model
        HRNet model.
    """

    def __init__(
        self,
        cfg: Dict,
        image_shape: Tuple[int, ...] = (256, 256, 1),
        normalization: str = "none",
        z_down: bool = False,
        output_channels: List[int] = [1],
        contrast: bool = False,
        contrast_proj_dim: int = 256,
    ):
        if len(output_channels) == 0:
            raise ValueError("'output_channels' needs to has at least one value")
        if len(output_channels) != 1 and len(output_channels) != 2:
            raise ValueError(f"'output_channels' must be a list of one or two values at max, not {output_channels}")

        self.output_channels = output_channels
        self.multihead = len(output_channels) == 2
        self.in_size = 64
        self.ndim = 3 if len(image_shape) == 4 else 2
        self.contrast = contrast
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
        block = blocks_dict[self.stage2_cfg["BLOCK"]]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]

        self.transition1 = self._make_transition_layer([256], num_channels, norm=normalization, mpool=mpool)

        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels, norm=normalization, mpool=mpool
        )

        self.stage3_cfg = cfg["STAGE3"]
        num_channels = self.stage3_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage3_cfg["BLOCK"]]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels, norm=normalization, mpool=mpool
        )
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels, norm=normalization, mpool=mpool
        )

        self.stage4_cfg = cfg["STAGE4"]
        num_channels = self.stage4_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage4_cfg["BLOCK"]]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels, norm=normalization, mpool=mpool
        )

        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True, norm=normalization, mpool=mpool
        )

        in_channels = 720  # 48 + 96 + 192 + 384
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
        Make a layer of blocks for the HRNet.

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
        block = blocks_dict[layer_config["BLOCK"]]

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

    def forward(self, x):
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

        return out_dict


class HighResolutionModule(nn.Module):
    """
    High Resolution Module for HRNet.

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
        return self.num_inchannels

    def forward(self, x):
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
    High Resolution Next model for HRNet.

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
        block = blocks_dict[self.stage1_cfg["BLOCK"]]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition0 = self._make_transition_layer([64], num_channels, norm=norm, mpool=mpool)
        self.stage1, pre_stage_channels = self._make_stage(self.stage1_cfg, num_channels, norm=norm, mpool=mpool)

        self.stage2_cfg = cfg["STAGE2"]
        num_channels = self.stage2_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage2_cfg["BLOCK"]]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(pre_stage_channels, num_channels, norm=norm, mpool=mpool)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels, norm=norm, mpool=mpool)

        self.stage3_cfg = cfg["STAGE3"]
        num_channels = self.stage3_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage3_cfg["BLOCK"]]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels, norm=norm, mpool=mpool)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels, norm=norm, mpool=mpool)

        self.stage4_cfg = cfg["STAGE4"]
        num_channels = self.stage4_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage4_cfg["BLOCK"]]
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
        block = blocks_dict[layer_config["BLOCK"]]

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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x_list = []
        for i in range(self.stage1_cfg["NUM_BRANCHES"]):
            if self.transition0[i] is not None:
                x_list.append(self.transition0[i](x))
            else:
                x_list.append(x)
        y_list = self.stage1(x_list)

        x_list = []
        for i in range(self.stage2_cfg["NUM_BRANCHES"]):
            if self.transition1[i] is not None:
                if i == 0:
                    x_list.append(self.transition1[i](y_list[0]))
                else:
                    x_list.append(self.transition1[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg["NUM_BRANCHES"]):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg["NUM_BRANCHES"]):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)
        return x
