"""Patch embedding, predefined ViT backbones and their blocks, checks and pretrained weight loading."""

from typing import Dict, List, Optional, Sequence, Tuple

import torch.nn as nn
from timm.models.vision_transformer import Block

from biapy.models.vit_backbones.cellpose_sam import CELLPOSE_SAM_VIT_PARAMS, build_cellpose_sam_blocks
from biapy.models.vit_backbones.celldino import CELLDINO_VIT_PARAMS
from biapy.models.vit_backbones.dinov3 import DINOV3_VIT_PARAMS, build_dinov3_blocks
from biapy.models.vit_backbones.sam3 import SAM3_VIT_PARAMS, build_sam3_blocks


class PatchEmbed(nn.Module):
    """2D/3D patch embedding. A ``stride`` smaller than ``patch_size`` gives overlapping tokens (the image is padded so
    the token grid is ``img_size / stride``)."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        ndim: int = 2,
        embed_dim: int = 768,
        stride: Optional[int] = None,
        bias: bool = True,
        flatten: bool = True,
        strict_img_size: bool = True,
    ):
        """
        Parameters
        ----------
        img_size, patch_size, in_chans, ndim, embed_dim : int
            Input size (square/cubic), token size, input channels, spatial dimensions (2/3) and token channels.

        stride : int, optional
            Projection stride. ``None`` = ``patch_size`` (non-overlapping).

        bias : bool, optional
            Projection bias.

        flatten : bool, optional
            Return ``(B, N, embed_dim)`` tokens instead of a map.

        strict_img_size : bool, optional
            Check the y/x input size.
        """
        super().__init__()
        stride = patch_size if stride is None or stride <= 0 else stride
        if stride > patch_size or (patch_size - stride) % 2 != 0:
            raise ValueError(
                f"The token stride ({stride}) must be between 1 and the token size ({patch_size}), and their "
                "difference must be even so the image can be padded symmetrically"
            )
        if img_size % stride != 0:
            raise ValueError(f"The input size ({img_size}) must be divisible by the token stride ({stride})")
        self.ndim = ndim
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride
        self.grid_size = img_size // stride
        self.num_patches = self.grid_size**ndim
        self.flatten = flatten
        self.strict_img_size = strict_img_size
        conv = nn.Conv2d if ndim == 2 else nn.Conv3d
        self.proj = conv(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=(patch_size - stride) // 2, bias=bias
        )

    def forward(self, x):
        """Embed ``(B, C, (Z), Y, X)`` images into ``(B, N, embed_dim)`` tokens (or a token map)."""
        if self.strict_img_size:
            H, W = x.shape[-2:]
            assert H == self.img_size, f"Input height ({H}) doesn't match model ({self.img_size})."
            assert W == self.img_size, f"Input width ({W}) doesn't match model ({self.img_size})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NC(Z)YX -> NLC
        return x


# Predefined backbones ('MODEL.VIT_MODEL'). "blocks": "timm" (plain ViT), "sam3", "dinov3" or "cellpose_sam".
# "init_values" enables LayerScale; "norm_eps" overrides the architecture's LayerNorm epsilon. "in_chans": fixed
# input channels (None = any). "pretrained": supports 'MODEL.VIT_PRETRAINED_WEIGHTS'.
VIT_BACKBONES: Dict[str, Dict] = {
    "vit_base_patch16": dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, blocks="timm", ndims=(2, 3),
        in_chans=None, archs=("vit", "unetr", "dpt", "vit_readout"), pretrained=False,
    ),
    "vit_large_patch16": dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4.0, blocks="timm", ndims=(2, 3),
        in_chans=None, archs=("vit", "unetr", "dpt", "vit_readout"), pretrained=False,
    ),
    "vit_huge_patch14": dict(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4.0, blocks="timm", ndims=(2, 3),
        in_chans=None, archs=("vit", "dpt", "vit_readout"), pretrained=False,
    ),
    "sam3_vit": dict(
        patch_size=SAM3_VIT_PARAMS["patch_size"],
        embed_dim=SAM3_VIT_PARAMS["embed_dim"],
        depth=SAM3_VIT_PARAMS["depth"],
        num_heads=SAM3_VIT_PARAMS["num_heads"],
        mlp_ratio=SAM3_VIT_PARAMS["mlp_ratio"],
        norm_eps=SAM3_VIT_PARAMS["norm_eps"],
        blocks="sam3", ndims=(2,), in_chans=None, archs=("vit", "unetr"), pretrained=True,
    ),
    "celldino_vit": dict(
        patch_size=CELLDINO_VIT_PARAMS["patch_size"],
        embed_dim=CELLDINO_VIT_PARAMS["embed_dim"],
        depth=CELLDINO_VIT_PARAMS["depth"],
        num_heads=CELLDINO_VIT_PARAMS["num_heads"],
        mlp_ratio=CELLDINO_VIT_PARAMS["mlp_ratio"],
        init_values=CELLDINO_VIT_PARAMS["init_values"],
        norm_eps=CELLDINO_VIT_PARAMS["norm_eps"],
        blocks="timm", ndims=(2,), in_chans=CELLDINO_VIT_PARAMS["in_chans"],
        archs=("vit", "unetr", "dpt", "vit_readout"), pretrained=True,
    ),
    "dinov3_vit": dict(
        patch_size=DINOV3_VIT_PARAMS["patch_size"],
        embed_dim=DINOV3_VIT_PARAMS["embed_dim"],
        depth=DINOV3_VIT_PARAMS["depth"],
        num_heads=DINOV3_VIT_PARAMS["num_heads"],
        mlp_ratio=DINOV3_VIT_PARAMS["mlp_ratio"],
        init_values=DINOV3_VIT_PARAMS["init_values"],
        norm_eps=DINOV3_VIT_PARAMS["norm_eps"],
        blocks="dinov3", ndims=(2,), in_chans=None, archs=("vit", "unetr", "dpt", "vit_readout"), pretrained=True,
    ),
    "cellpose_sam_vit": dict(
        patch_size=CELLPOSE_SAM_VIT_PARAMS["patch_size"],
        embed_dim=CELLPOSE_SAM_VIT_PARAMS["embed_dim"],
        depth=CELLPOSE_SAM_VIT_PARAMS["depth"],
        num_heads=CELLPOSE_SAM_VIT_PARAMS["num_heads"],
        mlp_ratio=CELLPOSE_SAM_VIT_PARAMS["mlp_ratio"],
        norm_eps=CELLPOSE_SAM_VIT_PARAMS["norm_eps"],
        blocks="cellpose_sam", ndims=(2,), in_chans=None, archs=("vit_readout",), pretrained=True,
    ),
}

# Token size under UNETR, whose decoder needs a power of two
UNETR_PATCH_SIZE = {"sam3_vit": 16}


def check_vit_backbone(vit_model: str, arch: str, ndim: int, in_chans: int):
    """Raise if a predefined backbone can not be used with ``arch`` and ``ndim``-D data with ``in_chans`` channels."""
    if vit_model == "custom":
        return
    if vit_model not in VIT_BACKBONES:
        raise ValueError(f"'vit_model' needs to be 'custom' or one of {list(VIT_BACKBONES)}. Provided: '{vit_model}'")
    preset = VIT_BACKBONES[vit_model]
    if arch not in preset["archs"]:
        raise ValueError(f"'{vit_model}' can only be used with the {list(preset['archs'])} architectures, not '{arch}'")
    if ndim not in preset["ndims"]:
        raise ValueError(f"'{vit_model}' can only be used with {' or '.join(f'{d}D' for d in preset['ndims'])} data")
    if preset["in_chans"] is not None and in_chans != preset["in_chans"]:
        raise ValueError(f"'{vit_model}' needs {preset['in_chans']} input channel(s), but the data has {in_chans}")


def build_vit_blocks(
    blocks: str,
    embed_dim: int,
    depth: int,
    num_heads: int,
    mlp_ratio: float,
    grid_size: Tuple[int, int],
    num_prefix_tokens: int,
    norm_layer=nn.LayerNorm,
    init_values: Optional[float] = None,
    drop_path: Optional[Sequence[float]] = None,
) -> nn.ModuleList:
    """
    Transformer blocks of a backbone.

    Parameters
    ----------
    blocks : str
        "timm" (plain ViT, built from the rest of arguments), "sam3", "dinov3" or "cellpose_sam" (fixed geometry).

    embed_dim, depth, num_heads, mlp_ratio, norm_layer, init_values :
        Plain ViT block settings (``init_values`` = LayerScale, ``None`` disables it).

    grid_size : Tuple[int, int]
        Token grid (y, x), for rotary position embeddings.

    num_prefix_tokens : int
        Class/register tokens before the grid.

    drop_path : Sequence[float], optional
        Stochastic depth rate per block.

    Returns
    -------
    blocks : nn.ModuleList
        The blocks.
    """
    if blocks == "sam3":
        return build_sam3_blocks(grid_size, num_prefix_tokens=num_prefix_tokens)
    if blocks == "dinov3":
        return build_dinov3_blocks(grid_size, num_prefix_tokens, drop_path)
    if blocks == "cellpose_sam":
        return build_cellpose_sam_blocks(drop_path)
    drop_path = [0.0] * depth if drop_path is None else list(drop_path)
    return nn.ModuleList(
        [
            Block(
                embed_dim,
                num_heads,
                mlp_ratio,
                qkv_bias=True,
                init_values=init_values,
                drop_path=drop_path[i],
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ]
    )


def load_pretrained_vit_backbone(model: nn.Module, vit_model: str, weights: str) -> Dict[str, int]:
    """Load the released weights of ``vit_model`` ('MODEL.VIT_PRETRAINED_WEIGHTS' = ``weights``) into ``model``."""
    if vit_model == "sam3_vit":
        from biapy.models.vit_backbones.sam3 import load_sam3_pretrained_encoder

        return load_sam3_pretrained_encoder(model, weights=weights)
    if vit_model == "dinov3_vit":
        from biapy.models.vit_backbones.dinov3 import load_dinov3_pretrained_encoder

        return load_dinov3_pretrained_encoder(model, weights=weights)
    if vit_model == "celldino_vit":
        from biapy.models.vit_backbones.celldino import load_celldino_pretrained_encoder

        return load_celldino_pretrained_encoder(model, weights=weights)
    if vit_model == "cellpose_sam_vit":
        from biapy.models.vit_backbones.cellpose_sam import load_cellpose_sam_pretrained

        return load_cellpose_sam_pretrained(model, weights=weights)
    raise ValueError(f"'{vit_model}' has no pretrained weights")


def vit_backbones_with_weights() -> List[str]:
    """Backbones with pretrained weights."""
    return [name for name, preset in VIT_BACKBONES.items() if preset["pretrained"]]
