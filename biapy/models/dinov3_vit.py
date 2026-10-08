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

import os
import re
import math
import warnings
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, LayerScale, Mlp

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


def _dinov3_weights_path(weights: str) -> str:
    """
    Get a local path to the DINOv3 weights.

    Parameters
    ----------
    weights : str
        Local path to a checkpoint, URL to download it from (e.g. the one e-mailed by Meta) or identifier of a
        Hugging Face Hub repository, e.g. ``'facebook/dinov3-vitl16-pretrain-lvd1689m'``.

    Returns
    -------
    path : str
        Local path to the downloaded (or provided) checkpoint file.
    """
    if os.path.isfile(weights):
        return weights

    if weights.startswith(("http://", "https://")):
        from urllib.parse import urlparse

        dst = os.path.join(torch.hub.get_dir(), "checkpoints", os.path.basename(urlparse(weights).path))
        if not os.path.isfile(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            print(f"    - downloading '{weights}' to '{dst}'")
            torch.hub.download_url_to_file(weights, dst)
        return dst

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError, LocalTokenNotFoundError, RepositoryNotFoundError

    try:
        return hf_hub_download(weights, "model.safetensors")
    except (GatedRepoError, LocalTokenNotFoundError, RepositoryNotFoundError) as e:
        raise RuntimeError(_dinov3_gated_message(weights, e)) from e
    except Exception as e:
        if "401" in str(e) or "403" in str(e) or "gated" in str(e).lower() or "authenticat" in str(e).lower():
            raise RuntimeError(_dinov3_gated_message(weights, e)) from e
        raise


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
    """
    Read DINOv3's encoder tensors from a checkpoint, with the original (Meta) naming.

    Both the original checkpoints (``dinov3_vitl16_pretrain_lvd1689m-*.pth``, also within a training checkpoint) and
    the Hugging Face ``transformers`` ones (``model.safetensors``) are supported. The latter are converted back to the
    original naming, merging their separate ``q_proj``, ``k_proj`` and ``v_proj`` into ``qkv``.

    Parameters
    ----------
    path : str
        Local path to the checkpoint, either a ``.safetensors`` or a PyTorch file.

    Returns
    -------
    encoder : Dict[str, torch.Tensor]
        Tensors of the encoder, e.g. ``'blocks.0.attn.qkv.weight'`` or ``'storage_tokens'``.
    """
    if path.endswith(".safetensors"):
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as f:
            checkpoint = {k: f.get_tensor(k) for k in f.keys()}
    else:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        for key in ["teacher", "model", "state_dict"]:
            if isinstance(checkpoint, dict) and key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
        checkpoint = {k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)}

    if any("embeddings.patch_embeddings" in k for k in checkpoint):
        return _dinov3_from_hf(checkpoint)

    reference = "storage_tokens"
    prefix = next((k[: -len(reference)] for k in checkpoint if k.endswith(reference)), None)
    if prefix is None:
        raise RuntimeError(
            f"Could not find DINOv3's ViT encoder in the provided weights: no tensor ending in '{reference}' (original "
            "naming) or containing 'embeddings.patch_embeddings' (Hugging Face naming) was found. Some keys found: "
            f"{sorted(checkpoint)[:5]}"
        )
    return {k[len(prefix) :]: v for k, v in checkpoint.items() if k.startswith(prefix)}


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


def _dinov3_adapt_patch_embed(weight: torch.Tensor, in_chans: int, verbose: bool = True) -> torch.Tensor:
    """
    Adapt DINOv3's RGB patch embedding to the number of input channels of the model.

    Parameters
    ----------
    weight : torch.Tensor
        Pretrained patch embedding of shape ``(embed_dim, 3, 16, 16)``.

    in_chans : int
        Number of channels of the images the model is going to be trained with. It can only be ``1`` or ``3``.

    verbose : bool, optional
        Whether to print what is being adapted.

    Returns
    -------
    weight : torch.Tensor
        Patch embedding of shape ``(embed_dim, in_chans, 16, 16)``.
    """
    if in_chans == 1:
        # Adding up the three kernels reproduces exactly the response the pretrained model would give to a
        # grayscale image replicated into its three channels
        weight = weight.sum(dim=1, keepdim=True)
        if verbose:
            print(
                "    - patch embedding adapted from 3 (RGB) to 1 channel by adding up its three kernels, which is "
                "equivalent to replicating the grayscale image into the three input channels"
            )
    elif in_chans != 3:
        raise ValueError(
            f"DINOv3's pretrained weights can only be loaded with 1 or 3 input channels, but the images have "
            f"{in_chans}. DINOv3 was trained on RGB images, and BiaPy can only adapt its patch embedding "
            "automatically when the input is grayscale (1 channel), by adding up its three kernels. Keep the channel "
            "of interest (1 channel), combine them into an RGB image (3 channels), or set "
            "'MODEL.VIT_PRETRAINED_WEIGHTS' to '' to train from scratch."
        )
    return weight


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
    encoder = _dinov3_read_encoder(_dinov3_weights_path(weights))

    params = DINOV3_VIT_PARAMS
    ckpt_depth = 1 + max((int(k.split(".")[1]) for k in encoder if k.startswith("blocks.")), default=-1)
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
        new_state_dict["patch_embed.proj.weight"] = _dinov3_adapt_patch_embed(
            encoder["patch_embed.proj.weight"], in_chans, verbose=verbose
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

    model_state = model.state_dict()
    to_load, skipped = {}, []
    for k, v in new_state_dict.items():
        if k in model_state and model_state[k].shape == v.shape:
            to_load[k] = v
        else:
            skipped.append(k)

    tracked_prefixes = ("patch_embed.", "blocks.", "cls_token", "reg_token", "norm.")
    missing = [k for k in model_state if k.startswith(tracked_prefixes) and k not in to_load]

    model.load_state_dict(to_load, strict=False)

    if verbose:
        print(f"    - {len(to_load)} tensors of DINOv3's encoder loaded into the model")
        if missing:
            print(f"    - {len(missing)} encoder tensors were NOT found in the checkpoint: {missing[:6]}")
    if skipped:
        warnings.warn(
            f"{len(skipped)} tensors of DINOv3's checkpoint could not be loaded, as they do not exist in the model or "
            f"their shape differs: {skipped[:6]}"
        )

    return {"loaded": len(to_load), "missing": len(missing)}
