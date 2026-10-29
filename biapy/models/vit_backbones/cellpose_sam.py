"""
Cellpose-SAM's encoder (``CPSAM`` in https://github.com/MouseLand/cellpose), without the cellpose/segment_anything
packages: SAM's ViT-L image encoder with 8x8 tokens, global attention with decomposed relative position embeddings
in every block, a 32x32 absolute position embedding (256x256 tiles), no class token and SAM's neck. With
``vit_readout`` (3 readout features, no refinement) it is the whole Cellpose-SAM network. Parameter names follow
SAM's, so the released weights (``cpsam``, ``cpsam_v2``) map one to one.

References: https://www.biorxiv.org/content/10.1101/2025.04.28.651001 and https://arxiv.org/abs/2304.02643 (SAM's
encoder code, Apache 2.0).
"""

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath

from biapy.models.vit_backbones.weights import fetch_vit_weights, load_matching_tensors, read_checkpoint_tensors

# Fixed by the released checkpoints
CELLPOSE_SAM_VIT_PARAMS = {
    "patch_size": 8,
    "embed_dim": 1024,
    "depth": 24,
    "num_heads": 16,
    "mlp_ratio": 4.0,
    "qkv_bias": True,
    "norm_eps": 1e-6,
    "in_chans": 3,
    "out_chans": 256,
    # Relative position table sizes: SAM's global blocks span its 64x64 grid, the windowed ones 14 tokens (all are
    # global in Cellpose-SAM, the tables are interpolated)
    "global_attn_indexes": (5, 11, 17, 23),
    "sam_grid_size": 64,
    "sam_window_size": 14,
    # Position embedding grid (256x256 tiles)
    "grid_size": 32,
}

# Released checkpoints
CELLPOSE_SAM_WEIGHTS_URL = "https://huggingface.co/mouseland/cellpose-sam/resolve/main/"
CELLPOSE_SAM_WEIGHTS = ["cpsam", "cpsam_v2"]

# BiaPy channel -> Cellpose-SAM readout channel (y flow, x flow, cell probability logit)
CELLPOSE_SAM_OUTPUTS = {"Gv": 0, "Gh": 1, "F": 2}


class LayerNorm2d(nn.Module):
    """LayerNorm over the channels of a ``(B, C, H, W)`` tensor."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


def _get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """Relative position embeddings ``(q_size, k_size, head_dim)``, interpolating the table if needed (as SAM)."""
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos_resized[relative_coords.long()]


class SAMAttention(nn.Module):
    """Multi-head self-attention with SAM's decomposed relative position embeddings (from MViTv2)."""

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True, rel_pos_size: int = 64):
        """``rel_pos_size``: grid size the relative position tables are created for (``2 * size - 1`` entries)."""
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.rel_pos_h = nn.Parameter(torch.zeros(2 * rel_pos_size - 1, head_dim))
        self.rel_pos_w = nn.Parameter(torch.zeros(2 * rel_pos_size - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # (3, B * num_heads, H * W, head_dim)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        Rh = _get_rel_pos(H, H, self.rel_pos_h)
        Rw = _get_rel_pos(W, W, self.rel_pos_w)
        r_q = q.reshape(B * self.num_heads, H, W, -1)
        rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
        rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)
        attn = (attn.view(-1, H, W, H, W) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]).view(-1, H * W, H * W)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        return self.proj(x)


class SAMMLP(nn.Module):
    """SAM's MLP (``lin1`` - GELU - ``lin2``)."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.lin1 = nn.Linear(dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class SAMBlock(nn.Module):
    """SAM block with global attention on a ``(B, H, W, C)`` token grid."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        rel_pos_size: int = 64,
        drop_path: float = 0.0,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn = SAMAttention(dim, num_heads, qkv_bias=qkv_bias, rel_pos_size=rel_pos_size)
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        self.mlp = SAMMLP(dim, int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def build_cellpose_sam_blocks(drop_path: Optional[Sequence[float]] = None) -> nn.ModuleList:
    """The 24 blocks of the encoder, with optional per-block stochastic depth."""
    params = CELLPOSE_SAM_VIT_PARAMS
    drop_path = [0.0] * params["depth"] if drop_path is None else list(drop_path)
    return nn.ModuleList(
        [
            SAMBlock(
                dim=params["embed_dim"],
                num_heads=params["num_heads"],
                mlp_ratio=params["mlp_ratio"],
                qkv_bias=params["qkv_bias"],
                rel_pos_size=(
                    params["sam_grid_size"] if i in params["global_attn_indexes"] else params["sam_window_size"]
                ),
                drop_path=drop_path[i],
                norm_eps=params["norm_eps"],
            )
            for i in range(params["depth"])
        ]
    )


def build_cellpose_sam_neck() -> nn.Sequential:
    """SAM's neck: ``embed_dim`` tokens to ``out_chans`` channels."""
    params = CELLPOSE_SAM_VIT_PARAMS
    embed_dim, out_chans = params["embed_dim"], params["out_chans"]
    return nn.Sequential(
        nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),
        LayerNorm2d(out_chans),
        nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
        LayerNorm2d(out_chans),
    )


def _cellpose_sam_readout_targets(output_channel_info: List[str]) -> Optional[List[List[int]]]:
    """Per head, the Cellpose-SAM readout channel of each output channel, or ``None`` if they are not exactly F/Gv/Gh."""
    targets = [[CELLPOSE_SAM_OUTPUTS.get(ch, -1) for ch in info.split("+")] for info in output_channel_info]
    flat = sorted(t for head in targets for t in head)
    return targets if flat == sorted(CELLPOSE_SAM_OUTPUTS.values()) else None


def load_cellpose_sam_pretrained(
    model: nn.Module,
    weights: str,
    verbose: bool = True,
) -> Dict[str, int]:
    """
    Load Cellpose-SAM's weights into a ``vit_readout`` model built with ``vit_model='cellpose_sam_vit'``.

    The encoder is always loaded: with fewer than 3 input channels the first patch-embedding kernels are kept (as
    Cellpose-SAM does), and the position embedding is interpolated for grids other than 32x32. The readout is loaded
    too when the model can hold it (3 readout features, no refinement, heads predicting F/Gv/Gh), making it the whole
    Cellpose-SAM network.

    Parameters
    ----------
    model : nn.Module
        Target model.

    weights : str
        "cpsam", "cpsam_v2", a URL or a local checkpoint.

    verbose : bool, optional
        Whether to print a report.

    Returns
    -------
    report : Dict[str, int]
        Number of tensors ``loaded`` and ``missing``.
    """
    if verbose:
        print(f"Loading Cellpose-SAM's pretrained weights from '{weights}' ...")
    path = fetch_vit_weights(
        weights,
        named_urls={name: CELLPOSE_SAM_WEIGHTS_URL + name for name in CELLPOSE_SAM_WEIGHTS},
        cache_subdir="cellpose_sam",
    )
    checkpoint = read_checkpoint_tensors(path, wrapper_keys=[])
    checkpoint = {(k[len("module.") :] if k.startswith("module.") else k): v for k, v in checkpoint.items()}
    if "W2" not in checkpoint or "encoder.neck.0.weight" not in checkpoint or "encoder.cls_token" in checkpoint:
        raise RuntimeError(
            f"'{weights}' is not a Cellpose-SAM checkpoint. Use {CELLPOSE_SAM_WEIGHTS} or one fine-tuned from them."
        )

    new_state_dict = {}
    for k, v in checkpoint.items():
        if k.startswith("encoder."):
            new_state_dict[k[len("encoder.") :]] = v

    in_chans = model.patch_embed.proj.weight.shape[1]  # type: ignore
    w = new_state_dict["patch_embed.proj.weight"]
    if in_chans > w.shape[1]:
        raise ValueError(f"Cellpose-SAM's weights support up to {w.shape[1]} input channels, the model has {in_chans}")
    if in_chans < w.shape[1]:
        new_state_dict["patch_embed.proj.weight"] = w[:, :in_chans]
        if verbose:
            print(f"    - patch embedding reduced from {w.shape[1]} to {in_chans} input channel(s) (first kernels kept)")

    grid = model.patch_embed.grid_size  # type: ignore
    pos_embed = new_state_dict["pos_embed"]
    if pos_embed.shape[1] != grid:
        new_state_dict["pos_embed"] = F.interpolate(
            pos_embed.permute(0, 3, 1, 2), size=(grid, grid), mode="bicubic", align_corners=False
        ).permute(0, 2, 3, 1)
        if verbose:
            print(f"    - position embedding interpolated from {pos_embed.shape[1]}x{pos_embed.shape[1]} to {grid}x{grid}")

    # Readout: 1x1 conv to 3 * stride^2 channels + pixel shuffle, as 'vit_readout'
    targets = _cellpose_sam_readout_targets(list(getattr(model, "output_channel_info", [])))
    readout_ok = (
        getattr(model, "readout_features", None) == len(CELLPOSE_SAM_OUTPUTS)
        and getattr(model, "refine_layers", None) == 0
        and targets is not None
        and model.readout.weight.shape == checkpoint["out.weight"].shape[:2]  # type: ignore
    )
    if readout_ok:
        new_state_dict["readout.weight"] = checkpoint["out.weight"][:, :, 0, 0]
        new_state_dict["readout.bias"] = checkpoint["out.bias"]
        # Heads just select Cellpose-SAM's outputs
        for i, (head, head_targets) in enumerate(zip(model.heads, targets)):  # type: ignore
            weight = torch.zeros_like(head.weight)
            for j, t in enumerate(head_targets):
                weight[j, t] = 1.0
            new_state_dict[f"heads.{i}.weight"] = weight
            new_state_dict[f"heads.{i}.bias"] = torch.zeros_like(head.bias)
    elif verbose:
        print(
            "    - only the encoder is loaded (the readout needs 'MODEL.VIT_READOUT.FEATURES' = 3, 'REFINE_LAYERS' = 0 "
            "and DATA_CHANNELS 'F', 'Gv', 'Gh')"
        )

    return load_matching_tensors(
        model,
        new_state_dict,
        ("patch_embed.", "pos_embed", "blocks.", "neck.") + (("readout.", "heads.") if readout_ok else ()),
        "Cellpose-SAM's " + ("encoder and readout" if readout_ok else "encoder"),
        verbose=verbose,
    )
