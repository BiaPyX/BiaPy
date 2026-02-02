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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from biapy.models.blocks import prepare_activation_layers


# --------------------------------------------------------------------------------------
# Initialization (replacement for nnU-Net's InitWeights_He)
# --------------------------------------------------------------------------------------

class InitWeights_He:
    """Kaiming/He initialization compatible with nnU-Net's default settings."""
    def __init__(self, neg_slope: float = 1e-2):
        self.neg_slope = float(neg_slope)

    def __call__(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.InstanceNorm3d, nn.BatchNorm3d)):
            if hasattr(module, "weight") and module.weight is not None:
                nn.init.ones_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)


# --------------------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------------------

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

    output_channels : List[int]
        Number of output channels (one value for single-head, two for multi-head).  

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

    activations : List[List[str]], optional
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
        depth: Sequence[int] = (1, 1, 1, 1, 1, 1),
        dims: Sequence[int] = (32, 64, 128, 256, 512, 512),
        pool_op_kernel_sizes: Optional[Sequence[Sequence[int]]] = None,
        conv_kernel_sizes: Optional[Sequence[Sequence[int]]] = None,
        explicit_activations: bool = False,
        activations: List[List[str]] = [],
        *,
        deep_supervision: bool = True,
    ):
        super().__init__()

        if len(output_channels) == 0:
            raise ValueError("'output_channels' needs to has at least one value")
        if len(output_channels) != 1 and len(output_channels) != 2:
            raise ValueError(f"'output_channels' must be a list of one or two values at max, not {output_channels}")
        self.output_channels = output_channels
        self.multihead = len(output_channels) == 2
        self.image_shape = image_shape
        self.ndim = 3 if len(image_shape) == 4 else 2
        self.input_channels = int(image_shape[-1])
        self.weightInitializer = InitWeights_He(1e-2)

        self.explicit_activations = explicit_activations
        if self.explicit_activations:
            self.out_activations, self.class_activation = prepare_activation_layers(activations)

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

        # Multi-head:
        #   Instance segmentation: instances + classification
        #   Detection: points + classification
        self.last_class_head = None
        if self.multihead:
            self.last_class_head = self.conv_op(self.input_channels, output_channels[1], kernel_size=1, padding="same")

        self.apply(self.weightInitializer)

    def forward(self, x: torch.Tensor):
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
        if self._deep_supervision:
            out = [seg_outputs[-1]] + [
                    op(aux) for op, aux in zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])
                ]
        else:
            out = [seg_outputs[-1]]

        for j in range(len(out)):
            if self.explicit_activations:
                # If there is only one activation, apply it to the whole tensor
                if len(self.out_activations) == 1:
                    out[j] = self.out_activations[0](out[j])
                else:
                    for i, act in enumerate(self.out_activations):
                        out[j][:, i:i+1] = act(out[j][:, i:i+1])

        out_dict = {
            "pred": out,
        }

        # Multi-head output
        #   Instance segmentation: instances + classification
        #   Detection: points + classification
        if self.multihead and self.last_class_head:
            class_head_out = self.last_class_head(out[0])
            if self.explicit_activations:
                for i, act in enumerate(self.class_activation):
                    class_head_out[:, i:i+1] = act(class_head_out[:, i:i+1])
            out_dict["class"] = class_head_out

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


def STUNet_base(image_shape: Tuple[int, ...] = (256, 256, 1), output_channels: List[int] = [1], *, deep_supervision: bool = True,
                explicit_activations: bool = False, activations: List[List[str]] = []) -> STUNet:
    conv_kernel_sizes, pool_op_kernel_sizes = _common_kernels()
    return STUNet(
        image_shape=image_shape,
        output_channels=output_channels,
        depth=[1] * 6,
        dims=[32, 64, 128, 256, 512, 512],
        pool_op_kernel_sizes=pool_op_kernel_sizes,
        conv_kernel_sizes=conv_kernel_sizes,
        deep_supervision=deep_supervision,
        explicit_activations=explicit_activations,
        activations=activations,
    )

def STUNet_small(image_shape: Tuple[int, ...] = (256, 256, 1), output_channels: List[int] = [1], *, deep_supervision: bool = True,
                explicit_activations: bool = False, activations: List[List[str]] = []) -> STUNet:
    conv_kernel_sizes, pool_op_kernel_sizes = _common_kernels()
    return STUNet(
        image_shape=image_shape,
        output_channels=output_channels,
        depth=[1] * 6,
        dims=[16, 32, 64, 128, 256, 256],
        pool_op_kernel_sizes=pool_op_kernel_sizes,
        conv_kernel_sizes=conv_kernel_sizes,
        deep_supervision=deep_supervision,
        explicit_activations=explicit_activations,
        activations=activations,
    )


def STUNet_large(image_shape: Tuple[int, ...] = (256, 256, 1), output_channels: List[int] = [1], *, deep_supervision: bool = True,
                explicit_activations: bool = False, activations: List[List[str]] = []) -> STUNet:
    conv_kernel_sizes, pool_op_kernel_sizes = _common_kernels()
    return STUNet(
        image_shape=image_shape,
        output_channels=output_channels,
        depth=[2] * 6,
        dims=[64, 128, 256, 512, 1024, 1024],
        pool_op_kernel_sizes=pool_op_kernel_sizes,
        conv_kernel_sizes=conv_kernel_sizes,
        deep_supervision=deep_supervision,
        explicit_activations=explicit_activations,
        activations=activations,
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
    deep_supervision: bool = True,
    explicit_activations: bool = False,
    activations: List[List[str]] = [],
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
    deep_supervision : bool
        Whether to enable deep supervision (multiple outputs).
    explicit_activations : bool
        Whether to apply explicit activations to outputs.
    activations : List[List[str]]
        Activation functions for outputs.
    pretrained : Union[bool, str]
        If True, load default pretrained weights for the variant.
        If str, it can be a key in PRETRAINED_STUNET or a URL.
    map_location : str
        Device to map the loaded checkpoint.
    """
    v = variant.lower()
    if v == "small":
        model = STUNet_small(
            image_shape=image_shape, output_channels=output_channels, deep_supervision=deep_supervision,
            explicit_activations=explicit_activations, activations=activations,
        )
        default_key = "orgmim_cnn_small"
    elif v == "base":
        model = STUNet_base(
            image_shape=image_shape, output_channels=output_channels, deep_supervision=deep_supervision,
            explicit_activations=explicit_activations, activations=activations,
        )
        default_key = "orgmim_cnn_base"
    elif v == "large":
        model = STUNet_large(
            image_shape=image_shape, output_channels=output_channels, deep_supervision=deep_supervision,
            explicit_activations=explicit_activations, activations=activations,
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
