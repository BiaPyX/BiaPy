"""
SAM 3 (Segment Anything Model 3) ViT image encoder for BiaPy.

This module reproduces the image encoder ("trunk") of SAM 3 so its pretrained weights can be
used as the backbone of BiaPy's ``vit`` and ``unetr`` architectures. It is not a plain ViT: the
trunk applies 2D axial rotary position embeddings (RoPE) inside the attention of every block and
restricts the attention to windows of tokens in all but a few blocks, which are global. Building
the blocks in the very same way is what makes the pretrained weights behave as they were trained.

The geometry of the encoder is fixed by the released checkpoint (see ``SAM3_VIT_PARAMS``):

- 32 transformer blocks with 1024 embedding dimensions and 16 heads
- MLP hidden size of 4736 (i.e. a 4.625 ratio) with GELU activation
- ``14x14`` patches over ``1008x1008`` inputs, pretrained at ``336x336`` (a ``24x24`` token grid)
- Window attention over ``24x24`` tokens, except in blocks 7, 15, 23 and 31, which are global
- A layer normalization applied to the tokens before the blocks (``ln_pre``)

Classes:

- ``SAM3Attention``: multi-head attention with 2D axial RoPE.
- ``SAM3Block``: transformer block with RoPE attention, optionally windowed.

Functions:

- ``sam3_axial_freqs_cis``: 2D axial rotary frequencies of a token grid.
- ``sam3_apply_rope``: apply those frequencies to the queries and keys.
- ``build_sam3_blocks``: create the stack of blocks of the encoder.
- ``load_sam3_pretrained_encoder``: fetch the released weights and load them into a BiaPy model.

Reference: `SAM 3: Segment Anything with Concepts <https://huggingface.co/facebook/sam3>`_.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import Mlp

from biapy.models.vit_backbones.weights import (
    adapt_rgb_patch_embed,
    fetch_vit_weights,
    interpolate_grid_pos_embed,
    load_matching_tensors,
    max_block_index,
    read_checkpoint_tensors,
    strip_tensor_prefix,
)

# Geometry of SAM 3's image encoder. These values are not configurable: they are the ones of the
# released checkpoint, and any deviation would make its weights unusable.
SAM3_VIT_PARAMS = {
    "patch_size": 14,
    "embed_dim": 1024,
    "depth": 32,
    "num_heads": 16,
    # 4736 (the checkpoint's MLP hidden size) / 1024 embedding dimensions
    "mlp_ratio": 4.625,
    "qkv_bias": True,
    "norm_eps": 1e-6,
    "in_chans": 3,
    # Attention is computed over windows of 24x24 tokens, but for the blocks below, which are global.
    # Both values are taken from the checkpoint, where the rotary frequencies of blocks 7, 15, 23 and
    # 31 cover the whole 72x72 token grid of a 1008x1008 input while the rest cover just 24x24 tokens.
    "window_size": 24,
    "global_attn_indexes": [7, 15, 23, 31],
    "rope_theta": 10000.0,
    # Token grid the position embedding was pretrained with (336x336 inputs and 14x14 patches)
    "pretrain_grid_size": 24,
}


def sam3_axial_freqs_cis(
    head_dim: int,
    grid_h: int,
    grid_w: int,
    theta: float = 10000.0,
) -> torch.Tensor:
    """
    Compute the 2D axial rotary frequencies (RoPE) used within SAM 3's attention.

    Half of the frequencies encode the position of the token along the x axis and the other half
    along the y axis, so each token of the grid gets a different rotation of the queries and keys.
    The result matches exactly the ``freqs_cis`` buffers stored in the released checkpoint.

    Parameters
    ----------
    head_dim : int
        Number of channels of each attention head. Must be a multiple of 4.

    grid_h : int
        Number of tokens along the y axis.

    grid_w : int
        Number of tokens along the x axis.

    theta : float, optional
        Base period of the rotary embedding. Defaults to ``10000.0``.

    Returns
    -------
    freqs_cis : torch.Tensor
        Complex tensor of shape ``(grid_h * grid_w, head_dim // 2)`` with the rotation to apply
        to each token.
    """
    if head_dim % 4 != 0:
        raise ValueError(f"'head_dim' needs to be a multiple of 4 to build 2D RoPE. Provided: {head_dim}")

    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 4)[: head_dim // 4].float() / head_dim))
    t = torch.arange(grid_h * grid_w, dtype=torch.float32)
    t_x = (t % grid_w).float()
    t_y = torch.div(t, grid_w, rounding_mode="floor").float()
    freqs_x = torch.outer(t_x, freqs)
    freqs_y = torch.outer(t_y, freqs)
    return torch.cat(
        [
            torch.polar(torch.ones_like(freqs_x), freqs_x),
            torch.polar(torch.ones_like(freqs_y), freqs_y),
        ],
        dim=-1,
    )


def sam3_apply_rope(q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rotate the queries and keys with the given 2D axial rotary frequencies.

    Consecutive pairs of channels are interpreted as a complex number and multiplied by the
    corresponding frequency, which is how the position ends up encoded in the attention scores.

    Parameters
    ----------
    q : torch.Tensor
        Queries of shape ``(batch_size, num_heads, num_tokens, head_dim)``.

    k : torch.Tensor
        Keys, with the same shape as ``q``.

    freqs_cis : torch.Tensor
        Complex tensor of shape ``(num_tokens, head_dim // 2)`` as returned by
        `sam3_axial_freqs_cis`.

    Returns
    -------
    q : torch.Tensor
        Rotated queries, with the same shape and dtype as the input ones.

    k : torch.Tensor
        Rotated keys, with the same shape and dtype as the input ones.
    """
    q_ = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_ = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
    freqs = freqs_cis.reshape(1, 1, *freqs_cis.shape)
    q_out = torch.view_as_real(q_ * freqs).flatten(3)
    k_out = torch.view_as_real(k_ * freqs).flatten(3)
    return q_out.type_as(q), k_out.type_as(k)


class SAM3Attention(nn.Module):
    """
    Multi-head attention with 2D axial rotary position embeddings, as used by SAM 3's ViT.

    It is a standard attention layer, with the same ``qkv``/``proj`` layout as `timm`'s one so the
    released weights map one to one, except that the queries and keys are rotated with the rotary
    frequencies of the token grid before computing the attention scores.
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
            Whether to add a learnable bias to the query, key and value projection. Defaults to ``True``.
        """
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"'dim' ({dim}) needs to be divisible by 'num_heads' ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, freqs_cis: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Perform the forward pass of the attention layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tokens of shape ``(batch_size, num_tokens, dim)``.

        freqs_cis : torch.Tensor, optional
            Rotary frequencies to apply, of shape ``(num_tokens, head_dim // 2)``. If ``None`` no
            rotation is made, i.e. the layer behaves as a plain attention layer.

        Returns
        -------
        torch.Tensor
            Output tokens, with the same shape as the input ones.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if freqs_cis is not None:
            q, k = sam3_apply_rope(q, k, freqs_cis)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class SAM3Block(nn.Module):
    """
    Transformer block of SAM 3's image encoder.

    Same structure as `timm`'s block (``norm1`` - attention - ``norm2`` - MLP, both with residual
    connections), so the released weights can be loaded directly into it, but with 2D rotary
    position embeddings inside the attention and, optionally, with the attention restricted to
    windows of tokens.

    When the token grid is not larger than the window there is nothing to partition, so the block
    attends to all the tokens at once. That is the usual situation with BiaPy's patch sizes and it
    keeps every block within the token grid it was trained with.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        grid_size: Tuple[int, int],
        mlp_ratio: float = 4.625,
        qkv_bias: bool = True,
        window_size: int = 0,
        num_prefix_tokens: int = 0,
        rope_theta: float = 10000.0,
        norm_eps: float = 1e-6,
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
            Ratio to multiply ``dim`` to obtain the hidden size of the MLP. Defaults to ``4.625``,
            the one of SAM 3.

        qkv_bias : bool, optional
            Whether to add a learnable bias to the query, key and value projection. Defaults to ``True``.

        window_size : int, optional
            Size (in tokens) of the square windows the attention is restricted to. Set it to ``0``
            to attend to the whole token grid, i.e. to make the block a global one. Defaults to ``0``.

        num_prefix_tokens : int, optional
            Number of tokens at the beginning of the sequence that are not part of the token grid
            (e.g. a class token). They are not rotated, as they have no position in the grid.
            Defaults to ``0``.

        rope_theta : float, optional
            Base period of the rotary embedding. Defaults to ``10000.0``.

        norm_eps : float, optional
            Epsilon of the layer normalizations. Defaults to ``1e-6``.
        """
        super().__init__()
        self.grid_size = tuple(grid_size)
        self.num_prefix_tokens = num_prefix_tokens
        # There is only something to partition when the token grid does not fit within a window
        self.window_size = (
            window_size if window_size > 0 and max(self.grid_size) > window_size else 0
        )

        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn = SAM3Attention(dim, num_heads, qkv_bias=qkv_bias)
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.GELU)

        # Rotary frequencies of the token grid the attention sees: a window when the block is
        # windowed and the whole grid otherwise. Prefix tokens get an identity rotation.
        attn_grid = (self.window_size, self.window_size) if self.window_size > 0 else self.grid_size
        freqs_cis = sam3_axial_freqs_cis(dim // num_heads, attn_grid[0], attn_grid[1], theta=rope_theta)
        if num_prefix_tokens > 0:
            freqs_cis = torch.cat([torch.ones(num_prefix_tokens, freqs_cis.shape[1], dtype=freqs_cis.dtype), freqs_cis])
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def _windowed_attn(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the attention over windows of the token grid.

        The grid is padded to a multiple of the window size, split into windows and, if there are
        prefix tokens, they are attached to every window so they can still gather information from
        all of them (their output is then averaged back).

        Parameters
        ----------
        x : torch.Tensor
            Normalized tokens of shape ``(batch_size, num_prefix_tokens + grid_h * grid_w, dim)``.

        Returns
        -------
        torch.Tensor
            Output tokens, with the same shape as the input ones.
        """
        B, _, C = x.shape
        p, ws = self.num_prefix_tokens, self.window_size
        gh, gw = self.grid_size

        prefix, patches = x[:, :p], x[:, p:]
        patches = patches.reshape(B, gh, gw, C)

        # Pad the grid so it can be split into an exact number of windows
        pad_h, pad_w = (ws - gh % ws) % ws, (ws - gw % ws) % ws
        if pad_h or pad_w:
            patches = F.pad(patches, (0, 0, 0, pad_w, 0, pad_h))
        ph, pw = gh + pad_h, gw + pad_w

        # (B, ph, pw, C) -> (B * num_windows, ws * ws, C)
        windows = patches.reshape(B, ph // ws, ws, pw // ws, ws, C)
        windows = windows.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, C)
        num_windows = windows.shape[0] // B

        if p > 0:
            windows = torch.cat([prefix.repeat_interleave(num_windows, dim=0), windows], dim=1)

        windows = self.attn(windows, self.freqs_cis)

        if p > 0:
            prefix_out = windows[:, :p].reshape(B, num_windows, p, C).mean(dim=1)
            windows = windows[:, p:]
        patches = windows.reshape(B, ph // ws, pw // ws, ws, ws, C)
        patches = patches.permute(0, 1, 3, 2, 4, 5).reshape(B, ph, pw, C)
        if pad_h or pad_w:
            patches = patches[:, :gh, :gw]
        patches = patches.reshape(B, gh * gw, C)

        return torch.cat([prefix_out, patches], dim=1) if p > 0 else patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the block.

        Parameters
        ----------
        x : torch.Tensor
            Input tokens of shape ``(batch_size, num_prefix_tokens + grid_h * grid_w, dim)``.

        Returns
        -------
        torch.Tensor
            Output tokens, with the same shape as the input ones.
        """
        y = self.norm1(x)
        y = self._windowed_attn(y) if self.window_size > 0 else self.attn(y, self.freqs_cis)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


def build_sam3_blocks(grid_size: Tuple[int, int], num_prefix_tokens: int = 0) -> nn.ModuleList:
    """
    Create the stack of transformer blocks of SAM 3's image encoder.

    Parameters
    ----------
    grid_size : Tuple[int, int]
        Number of tokens of the input along the y and x axes.

    num_prefix_tokens : int, optional
        Number of tokens at the beginning of the sequence that are not part of the token grid
        (e.g. a class token). Defaults to ``0``.

    Returns
    -------
    blocks : nn.ModuleList
        The ``32`` blocks of the encoder, where those in ``SAM3_VIT_PARAMS['global_attn_indexes']``
        attend to the whole token grid and the rest to windows of tokens.
    """
    params = SAM3_VIT_PARAMS
    return nn.ModuleList(
        [
            SAM3Block(
                dim=params["embed_dim"],
                num_heads=params["num_heads"],
                grid_size=grid_size,
                mlp_ratio=params["mlp_ratio"],
                qkv_bias=params["qkv_bias"],
                window_size=0 if i in params["global_attn_indexes"] else params["window_size"],
                num_prefix_tokens=num_prefix_tokens,
                rope_theta=params["rope_theta"],
                norm_eps=params["norm_eps"],
            )
            for i in range(params["depth"])
        ]
    )


def _sam3_gated_message(weights: str, error: Exception) -> str:
    """
    Build the message shown when the SAM 3 weights can not be downloaded.

    Parameters
    ----------
    weights : str
        Identifier of the Hugging Face Hub repository that could not be accessed.

    error : Exception
        Error raised while trying to download the weights.

    Returns
    -------
    message : str
        Message explaining how to get access to the weights.
    """
    return (
        f"Could not download SAM 3's pretrained weights from the Hugging Face repository '{weights}'.\n"
        "SAM 3 is a gated model, so downloading it requires accepting its license and being logged in:\n"
        f"  1) Open https://huggingface.co/{weights} , log in and accept the conditions to access it.\n"
        "  2) Authenticate this machine, either logging in from the terminal:\n"
        "         hf auth login          (huggingface-cli login in older versions of huggingface_hub)\n"
        "     or exporting an access token created in https://huggingface.co/settings/tokens :\n"
        "         export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx\n"
        "Then launch BiaPy again. Alternatively, you can set 'MODEL.VIT_PRETRAINED_WEIGHTS' to the path "
        "of a local file with the weights, or leave it empty ('') to train the model from scratch.\n"
        f"Error reported by huggingface_hub: {type(error).__name__}: {error}"
    )


def load_sam3_pretrained_encoder(
    model: nn.Module,
    weights: str,
    verbose: bool = True,
) -> Dict[str, int]:
    """
    Load SAM 3's image encoder into ``model`` (``patch_embed``, ``pos_embed``, ``ln_pre``/``norm_pre``, ``blocks``),
    adapting the patch embedding to the model's channels and token size and interpolating the position embedding.

    Parameters
    ----------
    model : nn.Module
        Target model.

    weights : str
        Hugging Face repository (e.g. ``'facebook/sam3'``) or local checkpoint.

    verbose : bool, optional
        Whether to print a report.

    Returns
    -------
    report : Dict[str, int]
        Number of tensors ``loaded`` and ``missing``.
    """
    if verbose:
        print(f"Loading SAM 3's pretrained image encoder from '{weights}' ...")

    patch_size = model.patch_embed.patch_size  # type: ignore
    grid = model.patch_embed.grid_size  # type: ignore
    grid_size = (grid, grid) if isinstance(grid, int) else tuple(grid)
    in_chans = model.patch_embed.proj.weight.shape[1]  # type: ignore
    num_prefix_tokens = int(model.pos_embed.shape[1] - grid_size[0] * grid_size[1])  # type: ignore

    # The original checkpoint (its naming is the one reproduced here) is preferred
    path = fetch_vit_weights(
        weights,
        hf_filenames=["sam3.pt", "sam3.safetensors", "model.safetensors"],
        access_message=_sam3_gated_message,
    )
    trunk = strip_tensor_prefix(
        read_checkpoint_tensors(path, wrapper_keys=["model", "state_dict", "model_weights"]),
        "blocks.0.attn.qkv.weight",
        "SAM 3's image encoder",
    )

    # A different encoder would load partially and silently
    ckpt_depth = max_block_index(trunk)
    ckpt_embed_dim = trunk["patch_embed.proj.weight"].shape[0] if "patch_embed.proj.weight" in trunk else -1
    if ckpt_depth != SAM3_VIT_PARAMS["depth"] or ckpt_embed_dim != SAM3_VIT_PARAMS["embed_dim"]:
        raise RuntimeError(
            f"The image encoder stored in '{weights}' does not match the one BiaPy builds for 'sam3_vit': it has "
            f"{ckpt_depth} blocks of {ckpt_embed_dim} dimensions, while SAM 3's has {SAM3_VIT_PARAMS['depth']} "
            f"blocks of {SAM3_VIT_PARAMS['embed_dim']}. These weights seem to come from a different model or from "
            "a version of SAM whose encoder changed, so they can not be loaded into this backbone."
        )

    new_state_dict = {}
    # SAM 3's projection has no bias: zero it
    if "patch_embed.proj.weight" in trunk:
        new_state_dict["patch_embed.proj.weight"] = adapt_rgb_patch_embed(
            trunk["patch_embed.proj.weight"], in_chans, "SAM 3", patch_size=patch_size, verbose=verbose
        )
        if getattr(model.patch_embed.proj, "bias", None) is not None:
            new_state_dict["patch_embed.proj.bias"] = torch.zeros_like(model.patch_embed.proj.bias)
    if "pos_embed" in trunk:
        new_state_dict["pos_embed"] = interpolate_grid_pos_embed(
            trunk["pos_embed"], grid_size, num_prefix_tokens, verbose=verbose
        )
    ln_pre_name = "ln_pre" if hasattr(model, "ln_pre") else "norm_pre"
    for suffix in ["weight", "bias"]:
        if f"ln_pre.{suffix}" in trunk and hasattr(model, ln_pre_name):
            new_state_dict[f"{ln_pre_name}.{suffix}"] = trunk[f"ln_pre.{suffix}"]
    # Same naming as SAM3Block. The rotary frequencies are recomputed per block for the grid used
    for k, v in trunk.items():
        if k.startswith("blocks.") and not k.endswith("freqs_cis"):
            new_state_dict[k] = v

    return load_matching_tensors(
        model,
        new_state_dict,
        ("patch_embed.", "blocks.", "pos_embed", ln_pre_name),
        "SAM 3's encoder",
        ignore_suffixes=("freqs_cis",),
        verbose=verbose,
    )
