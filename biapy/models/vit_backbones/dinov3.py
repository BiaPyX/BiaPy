"""
DINOv3 ViT-L/16 image encoder for BiaPy.

This module reproduces DINOv3's ViT-L/16 (distilled from its 7B model on LVD-1689M) so its pretrained weights can be
used as the backbone of BiaPy's ``vit``, ``unetr``, ``dpt`` and ``vit_readout`` architectures. It is not a plain ViT:

- There is no learned position embedding. The position of each token is encoded with 2D axial rotary embeddings
  (RoPE) applied to the queries and keys inside every block, with the token coordinates normalized to [-1, 1] so
  they do not depend on the size of the token grid.
- Four register tokens are placed between the class token and the patch tokens. They take no part in the token
  grid (they are not rotated) and are dropped from the features returned by the encoder.
- LayerScale in every block, LayerNorms with 1e-5 epsilon and no bias on the keys (the key part of the ``qkv`` bias
  is masked out; unlike in a plain ViT it would not cancel out in the softmax, as RoPE rotates it per position).

The attribute names (``patch_embed.proj``, ``cls_token``, ``reg_token``, ``blocks.*.{norm1,attn.qkv,attn.proj,ls1,
norm2,mlp.fc1,mlp.fc2,ls2}``, ``norm``) follow timm, so the released weights map one to one.

Classes:

- ``DINOv3Attention``: multi-head attention with DINOv3's 2D RoPE.
- ``DINOv3Block``: transformer block with RoPE attention and LayerScale.

Functions:

- ``dinov3_rope_sincos``: rotary sin/cos of a token grid.
- ``build_dinov3_blocks``: create the stack of blocks of the encoder.
- ``load_dinov3_pretrained_encoder``: fetch the released weights and load them into a BiaPy model.

Reference: `DINOv3 <https://arxiv.org/abs/2508.10104>`_, https://github.com/facebookresearch/dinov3.
"""

import re
import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, LayerScale, Mlp

from biapy.models.vit_backbones.weights import (
    adapt_rgb_patch_embed,
    fetch_vit_weights,
    load_matching_tensors,
    max_block_index,
    read_checkpoint_tensors,
    strip_tensor_prefix,
)

# Geometry of DINOv3's ViT-L/16 ('dinov3_vitl16'). These values are not configurable: they are the ones of the
# released checkpoint, and any deviation would make its weights unusable.
DINOV3_VIT_PARAMS = {
    "patch_size": 16,
    "embed_dim": 1024,
    "depth": 24,
    "num_heads": 16,
    "mlp_ratio": 4.0,
    "qkv_bias": True,
    "norm_eps": 1e-5,
    "in_chans": 3,
    "init_values": 1e-5,
    "num_register_tokens": 4,
    "rope_base": 100.0,
}


def dinov3_rope_sincos(
    head_dim: int,
    grid_h: int,
    grid_w: int,
    base: float = 100.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute DINOv3's 2D axial rotary embedding (``normalize_coords="separate"``) of a token grid.

    The centre of each token is normalized to [-1, 1] along each axis, so the result does not depend on the
    resolution of the grid. A quarter of the channels of each head encode the y coordinate and another quarter the
    x one, duplicated to cover the two halves rotated by ``dinov3_apply_rope``.

    Parameters
    ----------
    head_dim : int
        Number of channels of each attention head. Must be a multiple of 4.

    grid_h : int
        Number of tokens along the y axis.

    grid_w : int
        Number of tokens along the x axis.

    base : float, optional
        Base period of the rotary embedding. Defaults to ``100.0``.

    Returns
    -------
    sin, cos : Tuple[torch.Tensor, torch.Tensor]
        Tensors of shape ``(grid_h * grid_w, head_dim)``.
    """
    if head_dim % 4 != 0:
        raise ValueError(f"'head_dim' needs to be a multiple of 4 to build 2D RoPE. Provided: {head_dim}")
    periods = base ** (2 * torch.arange(head_dim // 4, dtype=torch.float32) / (head_dim // 2))
    coords_h = torch.arange(0.5, grid_h, dtype=torch.float32) / grid_h
    coords_w = torch.arange(0.5, grid_w, dtype=torch.float32) / grid_w
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1).flatten(0, 1)
    coords = 2.0 * coords - 1.0
    angles = 2 * math.pi * coords[:, :, None] / periods[None, None, :]  # (HW, 2, head_dim // 4)
    angles = angles.flatten(1, 2).tile(2)  # (HW, head_dim)
    return torch.sin(angles), torch.cos(angles)


def dinov3_apply_rope(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    """Rotate ``(..., N, head_dim)`` queries or keys, pairing channel ``i`` with ``i + head_dim / 2``."""
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat([-x2, x1], dim=-1) * sin


class DINOv3Attention(nn.Module):
    """
    Multi-head attention with 2D RoPE and no key bias, as in DINOv3.

    Same ``qkv``/``proj`` layout as timm's attention so the released weights map one to one. The tokens before the
    grid (class and register tokens) are not rotated, as they have no position.
    """

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True):
        """
        Initialize the attention layer.

        Parameters
        ----------
        dim : int
            Number of channels of the tokens.

        num_heads : int
            Number of attention heads. ``dim`` must be divisible by it.

        qkv_bias : bool, optional
            Whether to add a learnable bias to the queries and values (never to the keys). Defaults to ``True``.
        """
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"'dim' ({dim}) needs to be divisible by 'num_heads' ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        if qkv_bias:
            bias_mask = torch.ones(dim * 3)
            bias_mask[dim : 2 * dim] = 0
            self.register_buffer("qkv_bias_mask", bias_mask, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        num_prefix_tokens: int = 0,
    ) -> torch.Tensor:
        """
        Perform the forward pass of the attention layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tokens of shape ``(batch_size, num_prefix_tokens + grid_h * grid_w, dim)``.

        rope : Tuple[torch.Tensor, torch.Tensor], optional
            ``(sin, cos)`` of the token grid, as returned by ``dinov3_rope_sincos``.

        num_prefix_tokens : int, optional
            Number of tokens at the beginning of the sequence that are not part of the grid.

        Returns
        -------
        torch.Tensor
            Output tokens, with the same shape as the input ones.
        """
        B, N, C = x.shape
        bias = self.qkv.bias * self.qkv_bias_mask.to(self.qkv.bias.dtype) if self.qkv.bias is not None else None
        qkv = F.linear(x, self.qkv.weight, bias)
        q, k, v = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        if rope is not None:
            # As DINOv3, rotate in the precision of the rotary embedding (fp32) and cast back
            sin, cos = rope
            p = num_prefix_tokens
            q = torch.cat([q[:, :, :p], dinov3_apply_rope(q[:, :, p:].to(sin.dtype), sin, cos).to(q.dtype)], dim=2)
            k = torch.cat([k[:, :, :p], dinov3_apply_rope(k[:, :, p:].to(sin.dtype), sin, cos).to(k.dtype)], dim=2)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class DINOv3Block(nn.Module):
    """
    Transformer block of DINOv3's encoder.

    Same structure and naming as timm's block (``norm1`` - attention - ``ls1``, ``norm2`` - MLP - ``ls2``, both with
    residual connections), with DINOv3's RoPE attention. The rotary embedding of the token grid is computed once and
    kept as a (non-persistent) buffer.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        grid_size: Tuple[int, int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        init_values: Optional[float] = 1e-5,
        drop_path: float = 0.0,
        num_prefix_tokens: int = 0,
        rope_base: float = 100.0,
        norm_eps: float = 1e-5,
    ):
        """
        Initialize the block.

        Parameters
        ----------
        dim : int
            Number of channels of the tokens.

        num_heads : int
            Number of attention heads.

        grid_size : Tuple[int, int]
            Number of tokens of the input along the y and x axes.

        mlp_ratio : float, optional
            Ratio to multiply ``dim`` to obtain the hidden size of the MLP.

        qkv_bias : bool, optional
            Whether to add a learnable bias to the queries and values.

        init_values : float, optional
            Initial value of the LayerScale. ``None`` disables it.

        drop_path : float, optional
            Stochastic depth rate.

        num_prefix_tokens : int, optional
            Number of tokens at the beginning of the sequence that are not part of the grid (class and registers).

        rope_base : float, optional
            Base period of the rotary embedding.

        norm_eps : float, optional
            Epsilon of the layer normalizations.
        """
        super().__init__()
        self.num_prefix_tokens = num_prefix_tokens
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn = DINOv3Attention(dim, num_heads, qkv_bias=qkv_bias)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.GELU)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        sin, cos = dinov3_rope_sincos(dim // num_heads, grid_size[0], grid_size[1], base=rope_base)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.register_buffer("rope_cos", cos, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block to ``(batch_size, num_prefix_tokens + grid_h * grid_w, dim)`` tokens."""
        rope = (self.rope_sin, self.rope_cos)
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), rope, self.num_prefix_tokens)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


def build_dinov3_blocks(
    grid_size: Tuple[int, int],
    num_prefix_tokens: int,
    drop_path: Optional[Sequence[float]] = None,
) -> nn.ModuleList:
    """
    Create the stack of transformer blocks of DINOv3's ViT-L/16.

    Parameters
    ----------
    grid_size : Tuple[int, int]
        Number of tokens of the input along the y and x axes.

    num_prefix_tokens : int
        Number of tokens at the beginning of the sequence that are not part of the grid (class and registers).

    drop_path : Sequence[float], optional
        Stochastic depth rate of each block. No stochastic depth by default.

    Returns
    -------
    blocks : nn.ModuleList
        The ``24`` blocks of the encoder.
    """
    params = DINOV3_VIT_PARAMS
    drop_path = [0.0] * params["depth"] if drop_path is None else list(drop_path)
    return nn.ModuleList(
        [
            DINOv3Block(
                dim=params["embed_dim"],
                num_heads=params["num_heads"],
                grid_size=grid_size,
                mlp_ratio=params["mlp_ratio"],
                qkv_bias=params["qkv_bias"],
                init_values=params["init_values"],
                drop_path=drop_path[i],
                num_prefix_tokens=num_prefix_tokens,
                rope_base=params["rope_base"],
                norm_eps=params["norm_eps"],
            )
            for i in range(params["depth"])
        ]
    )


def _dinov3_gated_message(weights: str, error: Exception) -> str:
    """Build the message shown when the DINOv3 weights can not be downloaded."""
    return (
        f"Could not get DINOv3's pretrained weights from '{weights}'.\n"
        "DINOv3 is a gated model, so its weights can only be downloaded after accepting its license:\n"
        "  - Hugging Face: open https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m , log in and request "
        "access. Once granted, authenticate this machine ('hf auth login' or 'export HF_TOKEN=hf_xxx').\n"
        "  - Meta: request access at https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/ and set "
        "'MODEL.VIT_PRETRAINED_WEIGHTS' to the e-mailed URL of 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth' or to "
        "a local copy of it.\n"
        "Alternatively, leave 'MODEL.VIT_PRETRAINED_WEIGHTS' empty ('') to train the model from scratch.\n"
        f"Error reported: {type(error).__name__}: {error}"
    )


def _dinov3_read_encoder(path: str) -> Dict[str, torch.Tensor]:
    """Encoder tensors of an original (``.pth``) or Hugging Face (``.safetensors``) DINOv3 checkpoint, in the original naming."""
    checkpoint = read_checkpoint_tensors(path, wrapper_keys=["teacher", "model", "state_dict"])
    if any("embeddings.patch_embeddings" in k for k in checkpoint):
        return _dinov3_from_hf(checkpoint)
    return strip_tensor_prefix(checkpoint, "storage_tokens", "DINOv3's ViT encoder (original naming)")


def _dinov3_from_hf(checkpoint: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert a Hugging Face ``DINOv3ViTModel`` state dict to the original DINOv3 naming."""
    simple = [
        (r"(?:^|\.)embeddings\.cls_token$", "cls_token"),
        (r"(?:^|\.)embeddings\.register_tokens$", "storage_tokens"),
        (r"(?:^|\.)embeddings\.patch_embeddings\.(weight|bias)$", r"patch_embed.proj.\1"),
        (r"(?:^|\.)layer\.(\d+)\.attention\.o_proj\.(weight|bias)$", r"blocks.\1.attn.proj.\2"),
        (r"(?:^|\.)layer\.(\d+)\.layer_scale(\d)\.lambda1$", r"blocks.\1.ls\2.gamma"),
        (r"(?:^|\.)layer\.(\d+)\.mlp\.up_proj\.(weight|bias)$", r"blocks.\1.mlp.fc1.\2"),
        (r"(?:^|\.)layer\.(\d+)\.mlp\.down_proj\.(weight|bias)$", r"blocks.\1.mlp.fc2.\2"),
        (r"(?:^|\.)layer\.(\d+)\.norm(\d)\.(weight|bias)$", r"blocks.\1.norm\2.\3"),
        (r"^(?:.*\.)?norm\.(weight|bias)$", r"norm.\1"),
    ]
    encoder, qkv = {}, {}
    for k, v in checkpoint.items():
        m = re.search(r"(?:^|\.)layer\.(\d+)\.attention\.(q|k|v)_proj\.(weight|bias)$", k)
        if m:
            qkv.setdefault((int(m.group(1)), m.group(3)), {})[m.group(2)] = v
            continue
        for pattern, repl in simple:
            m = re.search(pattern, k)
            if m:
                encoder[m.expand(repl)] = v
                break

    for (i, kind), parts in qkv.items():
        if "q" not in parts or "v" not in parts:
            continue
        # The keys have no bias in DINOv3 (it is masked out), so it is filled with zeros when missing
        k_part = parts.get("k", torch.zeros_like(parts["q"]) if kind == "bias" else None)
        if k_part is None:
            continue
        encoder[f"blocks.{i}.attn.qkv.{kind}"] = torch.cat([parts["q"], k_part, parts["v"]], dim=0)
    return encoder


def load_dinov3_pretrained_encoder(
    model: nn.Module,
    weights: str,
    verbose: bool = True,
) -> Dict[str, int]:
    """
    Load DINOv3's pretrained ViT-L/16 encoder into a BiaPy model.

    The tensors are mapped into the ``patch_embed``, ``cls_token``, ``reg_token``, ``blocks`` and, if the model has
    it, ``norm`` of the given model, adapting the patch embedding to the number of input channels. There is no
    position embedding to adapt: the rotary embedding of each block is computed for the token grid actually used.

    Parameters
    ----------
    model : nn.Module
        Model to load the weights into, built with ``build_dinov3_blocks`` and a ``reg_token`` parameter.

    weights : str
        Local path to a checkpoint (original ``.pth`` or Hugging Face ``.safetensors``), URL to download it from or
        identifier of a Hugging Face Hub repository, e.g. ``'facebook/dinov3-vitl16-pretrain-lvd1689m'``.

    verbose : bool, optional
        Whether to print a report of what was loaded.

    Returns
    -------
    report : Dict[str, int]
        Number of tensors ``loaded`` into the model and number of them left ``missing``.
    """
    if verbose:
        print(f"Loading DINOv3's pretrained ViT encoder from '{weights}' ...")

    in_chans = model.patch_embed.proj.weight.shape[1]  # type: ignore
    encoder = _dinov3_read_encoder(
        fetch_vit_weights(weights, hf_filenames=["model.safetensors"], access_message=_dinov3_gated_message)
    )

    params = DINOV3_VIT_PARAMS
    ckpt_depth = max_block_index(encoder)
    ckpt_embed_dim = encoder["cls_token"].shape[-1] if "cls_token" in encoder else -1
    ckpt_mlp = encoder["blocks.0.mlp.fc1.weight"].shape[0] if "blocks.0.mlp.fc1.weight" in encoder else -1
    if (
        ckpt_depth != params["depth"]
        or ckpt_embed_dim != params["embed_dim"]
        or ckpt_mlp != int(params["embed_dim"] * params["mlp_ratio"])
    ):
        raise RuntimeError(
            f"The encoder stored in '{weights}' does not match 'dinov3_vit' (DINOv3's ViT-L/16): it has {ckpt_depth} "
            f"blocks of {ckpt_embed_dim} dimensions and an MLP of {ckpt_mlp}, while it should have {params['depth']} "
            f"blocks of {params['embed_dim']} dimensions and an MLP of {int(params['embed_dim'] * params['mlp_ratio'])}. "
            "Use the 'dinov3_vitl16' weights (e.g. 'facebook/dinov3-vitl16-pretrain-lvd1689m')."
        )

    new_state_dict = {}
    if "patch_embed.proj.weight" in encoder:
        new_state_dict["patch_embed.proj.weight"] = adapt_rgb_patch_embed(
            encoder["patch_embed.proj.weight"], in_chans, "DINOv3", verbose=verbose
        )
    for src, dst in [("patch_embed.proj.bias", "patch_embed.proj.bias"), ("cls_token", "cls_token"), ("storage_tokens", "reg_token")]:
        if src in encoder:
            new_state_dict[dst] = encoder[src]
    # UNETR (and ViT with global pooling) have no final 'norm', so it is only copied if present
    if hasattr(model, "norm") and not isinstance(model.norm, nn.Identity):
        for suffix in ["weight", "bias"]:
            if f"norm.{suffix}" in encoder:
                new_state_dict[f"norm.{suffix}"] = encoder[f"norm.{suffix}"]
    for k, v in encoder.items():
        if k.startswith("blocks.") and not k.endswith("bias_mask"):
            if k.endswith("attn.qkv.bias"):
                # The key part of the bias is masked out in DINOv3; zero it so it stays inert
                v = v.clone()
                v[v.shape[0] // 3 : 2 * v.shape[0] // 3] = 0
            new_state_dict[k] = v

    return load_matching_tensors(
        model,
        new_state_dict,
        ("patch_embed.", "blocks.", "cls_token", "reg_token", "norm."),
        "DINOv3's encoder",
        verbose=verbose,
    )
