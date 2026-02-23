"""
Standalone STU-Net (STUNet) implementation extracted from OrgMIM (https://github.com/yanchaoz/OrgMIM).

This file is self-contained (PyTorch only) and provides:
- BasicResBlock
- Upsample_Layer_nearest
- STUNet (3D segmentation network)
- Convenience constructors: STUNet_small / STUNet_base / STUNet_large
- Optional pretrained-encoder loading helpers (compatible with OrgMIM checkpoints)

Origin:
- Adapted from OrgMIM: https://github.com/yanchaoz/OrgMIM

Notes:
- This is a 3D network (Conv3d). If you need 2D, the same pattern can be ported.
- By default, forward() returns a list with a single element: [full_res_logits].
  If deep supervision is enabled (deep_supervision=True),
  it returns a tuple like nnU-Net: (full_res, upscaled_aux1, upscaled_aux2, ...)
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from biapy.models.blocks import init_weights, prepare_activation_layers

class BasicResBlock(nn.Module):
    """
    Residual block used by OrgMIM's STUNet.

    conv3d -> instancenorm -> leakyrelu -> conv3d -> instancenorm -> residual add -> leakyrelu
    Optionally uses a 1x1 conv on the skip path to match channels/stride.
    """
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: Union[int, Sequence[int]] = 3,
        padding: Union[int, Sequence[int]] = 1,
        stride: Union[int, Sequence[int]] = 1,
        use_1x1conv: bool = False,
    ):
        super().__init__()
        self.conv1 = nn.Conv3d(
            input_channels, output_channels,
            kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.norm1 = nn.InstanceNorm3d(output_channels, affine=True)
        self.act1 = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv3d(output_channels, output_channels, kernel_size=kernel_size, padding=padding)
        self.norm2 = nn.InstanceNorm3d(output_channels, affine=True)
        self.act2 = nn.LeakyReLU(inplace=True)

        self.conv3 = nn.Conv3d(input_channels, output_channels, kernel_size=1, stride=stride) if use_1x1conv else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.act1(self.norm1(y))
        y = self.norm2(self.conv2(y))
        if self.conv3 is not None:
            x = self.conv3(x)
        y = y + x
        return self.act2(y)


class Upsample_Layer_nearest(nn.Module):
    """
    Nearest-neighbor upsampling followed by 1x1x1 conv to set channel count.
    """
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        pool_op_kernel_size: Sequence[int],
        mode: str = "nearest",
    ):
        super().__init__()
        self.conv = nn.Conv3d(input_channels, output_channels, kernel_size=1)
        self.pool_op_kernel_size = tuple(int(i) for i in pool_op_kernel_size)
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.interpolate(x, scale_factor=self.pool_op_kernel_size, mode=self.mode)
        x = self.conv(x)
        return x


def _elementwise_prod(kernels: Sequence[Sequence[int]]) -> List[int]:
    """Element-wise product over a list of kernel-size vectors."""
    if len(kernels) == 0:
        return []
    out = [1] * len(kernels[0])
    for k in kernels:
        for i, v in enumerate(k):
            out[i] *= int(v)
    return out

class STUNet(nn.Module):
    """
    OrgMIM STUNet segmentation model (3D). 
    Reference: https://github.com/yanchaoz/OrgMIM 

    Parameters
    ----------
    image_shape : Tuple[int, ...]
        Shape of the input image (including channels as last dimension).

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

    depth : Sequence[int]
        Number of residual blocks per stage.

    dims : Sequence[int]
        Number of feature channels per stage.

    pool_op_kernel_sizes : Optional[Sequence[Sequence[int]]]
        Pooling kernel sizes per stage.

    conv_kernel_sizes : Optional[Sequence[Sequence[int]]]
        Convolution kernel sizes per stage.
    
    explicit_activations : bool, optional
        If True, uses explicit activation functions in the last layers.

    head_activations : List[List[str]], optional
        Activation functions to apply to the outputs if `explicit_activations` is True.

    deep_supervision : bool
        Whether to enable deep supervision (multiple outputs).

    Returns
    -------
    STUNet
        STUNet model instance.
    """
    def __init__(
        self,
        image_shape: Tuple[int, ...] = (256, 256, 1),
        output_channels: List[int] = [1],
        output_channel_info=["F"],
        explicit_activations: bool = False,
        head_activations: List[str] = ["ce_sigmoid"],
        depth: Sequence[int] = (1, 1, 1, 1, 1, 1),
        dims: Sequence[int] = (32, 64, 128, 256, 512, 512),
        pool_op_kernel_sizes: Optional[Sequence[Sequence[int]]] = None,
        conv_kernel_sizes: Optional[Sequence[Sequence[int]]] = None,
        *,
        deep_supervision: bool = True,
    ):
        super().__init__()

        if len(output_channels) == 0:
            raise ValueError("'output_channels' needs to has at least one value")
        print("Selected output channels:")        
        for i, info in enumerate(output_channel_info):
            print(f"  - {i} channel for {info} output")
        self.output_channels = output_channels
        self.output_channel_info = output_channel_info
        self.explicit_activations = explicit_activations
        self.return_class = True if "class" in output_channel_info else False
        self.image_shape = image_shape
        self.ndim = 3 if len(image_shape) == 4 else 2
        self.input_channels = int(image_shape[-1])

        if self.explicit_activations:
            assert len(head_activations) == len(output_channels), "If 'explicit_activations' is True, 'head_activations' needs to "
            "have the same number of values as 'output_channels'"
            self.head_activations, self.class_head_activations = prepare_activation_layers(head_activations, output_channel_info)
            if self.return_class and self.class_head_activations is None:
                raise ValueError("If 'return_class' is True, 'head_activations' must be provided.")

        if self.ndim == 3:
            self.conv_op = nn.Conv3d
            # self.norm_func = get_norm_3d
            # self.dropout = nn.Dropout3d
            # mpool = (2, 2, 2) if self.z_down else (1, 2, 2)
        else:
            self.conv_op = nn.Conv2d
            # self.norm_func = get_norm_2d
            # self.dropout = nn.Dropout2d
            # mpool = (2, 2)

        self.final_nonlin = lambda x: x  # logits
        self._deep_supervision = bool(deep_supervision)
        self.upscale_logits = False

        if conv_kernel_sizes is None or pool_op_kernel_sizes is None:
            raise ValueError("pool_op_kernel_sizes and conv_kernel_sizes must be provided (see STUNet_* helpers).")

        self.input_shape_must_be_divisible_by = _elementwise_prod(pool_op_kernel_sizes)

        self.pool_op_kernel_sizes = [list(map(int, k)) for k in pool_op_kernel_sizes]
        self.conv_kernel_sizes = [list(map(int, k)) for k in conv_kernel_sizes]

        self.conv_pad_sizes: List[List[int]] = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([i // 2 for i in krnl])

        num_pool = len(self.pool_op_kernel_sizes)
        if num_pool != len(dims) - 1:
            raise ValueError(f"Expected len(pool_op_kernel_sizes)=len(dims)-1, got {num_pool} vs {len(dims)-1}")

        dims = list(map(int, dims))
        depth = list(map(int, depth))

        # -----------------------------------------
        # Encoder (downsampling via strided conv in first block of each stage)
        # -----------------------------------------
        self.conv_blocks_context = nn.ModuleList()

        stage0 = nn.Sequential(
            BasicResBlock(self.input_channels, dims[0], self.conv_kernel_sizes[0], self.conv_pad_sizes[0], use_1x1conv=True),
            *[
                BasicResBlock(dims[0], dims[0], self.conv_kernel_sizes[0], self.conv_pad_sizes[0])
                for _ in range(depth[0] - 1)
            ],
        )
        self.conv_blocks_context.append(stage0)

        for d in range(1, num_pool + 1):
            stage = nn.Sequential(
                BasicResBlock(
                    dims[d - 1],
                    dims[d],
                    self.conv_kernel_sizes[d],
                    self.conv_pad_sizes[d],
                    stride=self.pool_op_kernel_sizes[d - 1],
                    use_1x1conv=True,
                ),
                *[
                    BasicResBlock(dims[d], dims[d], self.conv_kernel_sizes[d], self.conv_pad_sizes[d])
                    for _ in range(depth[d] - 1)
                ],
            )
            self.conv_blocks_context.append(stage)

        # -----------------------------------------
        # Upsampling layers
        # -----------------------------------------
        self.upsample_layers = nn.ModuleList()
        for u in range(num_pool):
            upsample_layer = Upsample_Layer_nearest(dims[-1 - u], dims[-2 - u], self.pool_op_kernel_sizes[-1 - u])
            self.upsample_layers.append(upsample_layer)

        # -----------------------------------------
        # Decoder
        # -----------------------------------------
        self.conv_blocks_localization = nn.ModuleList()
        for u in range(num_pool):
            stage = nn.Sequential(
                BasicResBlock(
                    dims[-2 - u] * 2,
                    dims[-2 - u],
                    self.conv_kernel_sizes[-2 - u],
                    self.conv_pad_sizes[-2 - u],
                    use_1x1conv=True,
                ),
                *[
                    BasicResBlock(
                        dims[-2 - u],
                        dims[-2 - u],
                        self.conv_kernel_sizes[-2 - u],
                        self.conv_pad_sizes[-2 - u],
                    )
                    for _ in range(depth[-2 - u] - 1)
                ],
            )
            self.conv_blocks_localization.append(stage)

        # -----------------------------------------
        # Outputs (one per decoder stage, nnU-Net style)
        # -----------------------------------------
        self.seg_outputs = nn.ModuleList()
        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs.append(nn.Conv3d(dims[-2 - ds], output_channels[0], kernel_size=1, padding="same"))

        # Deep supervision upscalers (OrgMIM uses identity lambdas)
        self.upscale_logits_ops = nn.ModuleList([nn.Identity() for _ in range(num_pool - 1)])

        # To store which head corresponds to which output channel in the multi-head scenario
        self.out_head_map = []
        self.heads = nn.Sequential()
        for i, out_ch in enumerate(output_channels):
            self.heads.append(nn.Conv3d(output_channels[0], out_ch, kernel_size=1, padding="same"))
            self.out_head_map += [i] * out_ch

        init_weights(self)

    def forward(self, x: torch.Tensor):
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
        skips: List[torch.Tensor] = []
        seg_outputs: List[torch.Tensor] = []

        # encoder (collect skips except bottleneck)
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)

        # bottleneck
        x = self.conv_blocks_context[-1](x)

        # decoder
        for u in range(len(self.conv_blocks_localization)):
            x = self.upsample_layers[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        # Regular output
        # if self._deep_supervision:
        #     feats = [seg_outputs[-1]] + [
        #             op(aux) for op, aux in zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])
        #         ]
        # else:
        # For now we deactivate deep supervision 
        feats = seg_outputs[-1]

        out_dict = {}

        # Pass the features through the output heads
        class_outs, outs = [], []
        for i, head_id in enumerate(self.out_head_map):
            if "class" not in self.output_channel_info[i]:
                outs.append(self.heads[head_id](feats))
            else:
                class_outs.append(self.heads[head_id](feats))  
        outs = torch.cat(outs, dim=1)

        # Apply head_activations to the output heads if explicit_activations is True
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

        if len(out_dict.keys()) == 1:
            return out_dict["pred"]
        else:
            return out_dict


# --------------------------------------------------------------------------------------
# Convenience presets (same as OrgMIM)
# --------------------------------------------------------------------------------------

def _common_kernels():
    conv_kernel_sizes = [[3, 3, 3]] * 6
    pool_op_kernel_sizes = [
        [2, 2, 2],
        [2, 2, 2],
        [2, 2, 2],
        [2, 2, 2],
        [1, 1, 1],
    ]
    return conv_kernel_sizes, pool_op_kernel_sizes


def STUNet_base(image_shape: Tuple[int, ...] = (256, 256, 1), output_channels: List[int] = [1], output_channel_info: List[str] = ["F"], 
                deep_supervision: bool = True, explicit_activations: bool = False, head_activations: List[str] = []) -> STUNet:
    conv_kernel_sizes, pool_op_kernel_sizes = _common_kernels()
    return STUNet(
        image_shape=image_shape,
        output_channels=output_channels,
        output_channel_info=output_channel_info,
        head_activations=head_activations,
        explicit_activations=explicit_activations,
        depth=[1] * 6,
        dims=[32, 64, 128, 256, 512, 512],
        pool_op_kernel_sizes=pool_op_kernel_sizes,
        conv_kernel_sizes=conv_kernel_sizes,
        deep_supervision=deep_supervision,
    )

def STUNet_small(image_shape: Tuple[int, ...] = (256, 256, 1), output_channels: List[int] = [1], output_channel_info: List[str] = ["F"], 
                 deep_supervision: bool = True, explicit_activations: bool = False, head_activations: List[str] = []) -> STUNet:
    conv_kernel_sizes, pool_op_kernel_sizes = _common_kernels()
    return STUNet(
        image_shape=image_shape,
        output_channels=output_channels,
        output_channel_info=output_channel_info,
        head_activations=head_activations,
        explicit_activations=explicit_activations,
        depth=[1] * 6,
        dims=[16, 32, 64, 128, 256, 256],
        pool_op_kernel_sizes=pool_op_kernel_sizes,
        conv_kernel_sizes=conv_kernel_sizes,
        deep_supervision=deep_supervision,  
    )


def STUNet_large(image_shape: Tuple[int, ...] = (256, 256, 1), output_channels: List[int] = [1], output_channel_info: List[str] = ["F"], 
                 deep_supervision: bool = True, explicit_activations: bool = False, head_activations: List[str] = []) -> STUNet:
    conv_kernel_sizes, pool_op_kernel_sizes = _common_kernels()
    return STUNet(
        image_shape=image_shape,
        output_channels=output_channels,
        output_channel_info=output_channel_info,
        head_activations=head_activations,
        explicit_activations=explicit_activations,
        depth=[2] * 6,
        dims=[64, 128, 256, 512, 1024, 1024],
        pool_op_kernel_sizes=pool_op_kernel_sizes,
        conv_kernel_sizes=conv_kernel_sizes,
        deep_supervision=deep_supervision,
    )


# --------------------------------------------------------------------------------------
# Optional: OrgMIM-compatible pretrained encoder loading
# --------------------------------------------------------------------------------------

PRETRAINED_STUNET: Dict[str, Dict[str, str]] = {
    "orgmim_cnn_base": {
        "url": "https://huggingface.co/yanchaoz/OrgMIM-models/resolve/main/orgmim_spark_b_learner.ckpt",
    },
    "orgmim_cnn_small": {
        "url": "https://huggingface.co/yanchaoz/OrgMIM-models/resolve/main/orgmim_spark_s_learner.ckpt",
    },
    "orgmim_cnn_large": {
        "url": "https://huggingface.co/yanchaoz/OrgMIM-models/resolve/main/orgmim_spark_l_learner.ckpt",
    },
}


def download_pretrained_ckpt(url: str, map_location: str = "cpu") -> Dict[str, Any]:
    """
    Download checkpoint automatically (cached by torch).

    Returns the loaded checkpoint dict (typically contains 'model_weights').
    """
    return torch.hub.load_state_dict_from_url(url, map_location=map_location, check_hash=False)


def load_stunet_pretrained_encoder_from_ckpt(model: STUNet, checkpoint: Dict[str, Any]) -> None:
    """
    Load OrgMIM pretrained encoder weights into a segmentation STUNet.

    OrgMIM checkpoints store weights under checkpoint['model_weights'] and encoder keys include 'encoder'.
    """
    pretrained_dict = checkpoint["model_weights"]
    new_dict = OrderedDict()
    for k, v in pretrained_dict.items():
        if "encoder" in k:
            # remove 'sp_cnn.' prefix when present
            new_k = k.split("sp_cnn.")[-1]
            new_dict[new_k] = v
    model.load_state_dict(new_dict, strict=False)
    print("[STUNet] Pretrained encoder loaded")


def build_stunet(
    variant: str,
    image_shape: Tuple[int, ...] = (256, 256, 1),
    output_channels: List[int] = [1],
    output_channel_info=["F"],
    explicit_activations: bool = False,
    head_activations: List[str] = ["ce_sigmoid"],
    deep_supervision: bool = True,
    pretrained: Union[bool, str] = False,
    map_location: str = "cpu",
) -> STUNet:
    """
    Build a STUNet model (small, base, large) with optional pretrained encoder loading.

    Parameters
    ----------
    variant : str
        One of 'small', 'base', 'large'.
    image_shape : Tuple[int, ...]
        Shape of the input image (including channels as last dimension).
    output_channels : List[int]
        Number of output channels (one value for single-head, two for multi-head).  
    output_channel_info : list of str
        Information about the type of output channels. Possible values are:
        - "X": where X is a letter, e.g. "F" for foreground, "D" for distance, "R" for rays, "C" for cpntours, etc.
        - "class": classification (e.g. for multi-task learning)
    explicit_activations : bool
        Whether to apply explicit head_activations to outputs.    
    head_activations : List[List[str]]
        Activation functions for outputs.
    deep_supervision : bool
        Whether to enable deep supervision (multiple outputs).
    pretrained : Union[bool, str]
        If True, load default pretrained weights for the variant.
        If str, it can be a key in PRETRAINED_STUNET or a URL.
    map_location : str
        Device to map the loaded checkpoint.
    """
    v = variant.lower()
    if v == "small":
        model = STUNet_small(
            image_shape=image_shape, output_channels=output_channels, output_channel_info=output_channel_info, 
            deep_supervision=deep_supervision, explicit_activations=explicit_activations, head_activations=head_activations,
        )
        default_key = "orgmim_cnn_small"
    elif v == "base":
        model = STUNet_base(
            image_shape=image_shape, output_channels=output_channels, output_channel_info=output_channel_info, 
            deep_supervision=deep_supervision, explicit_activations=explicit_activations, head_activations=head_activations,
        )
        default_key = "orgmim_cnn_base"
    elif v == "large":
        model = STUNet_large(
            image_shape=image_shape, output_channels=output_channels, output_channel_info=output_channel_info, 
            deep_supervision=deep_supervision, explicit_activations=explicit_activations, head_activations=head_activations,
        )
        default_key = "orgmim_cnn_large"
    else:
        raise ValueError("variant must be one of: small | base | large")

    if pretrained:
        if isinstance(pretrained, str):
            if pretrained in PRETRAINED_STUNET:
                url = PRETRAINED_STUNET[pretrained]["url"]
            elif pretrained.startswith("http://") or pretrained.startswith("https://"):
                url = pretrained
            else:
                raise ValueError(f"Unknown pretrained spec: {pretrained}")
        else:
            url = PRETRAINED_STUNET.get(default_key, PRETRAINED_STUNET["orgmim_cnn_base"])["url"]

        ckpt = download_pretrained_ckpt(url, map_location=map_location)
        load_stunet_pretrained_encoder_from_ckpt(model, ckpt)

    return model
