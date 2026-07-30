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

import os
import math
import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import Mlp

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


def _sam3_weights_path(weights: str) -> str:
    """
    Get a local path to the SAM 3 weights, downloading them from the Hugging Face Hub if needed.

    Parameters
    ----------
    weights : str
        Local path to a checkpoint or identifier of a Hugging Face Hub repository, e.g.
        ``'facebook/sam3'``.

    Returns
    -------
    path : str
        Local path to the downloaded (or provided) checkpoint file.
    """
    if os.path.isfile(weights):
        return weights

    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import GatedRepoError, LocalTokenNotFoundError, RepositoryNotFoundError

    # The original checkpoint is preferred, as the naming of its layers is the one this module
    # reproduces. The rest are only tried in case the repository does not contain it.
    preferred_files = ["sam3.pt", "sam3.safetensors", "model.safetensors"]
    try:
        repo_files = HfApi().list_repo_files(weights)
        filename = next((f for f in preferred_files if f in repo_files), None)
        if filename is None:
            raise FileNotFoundError(
                f"Could not find any of the expected weight files {preferred_files} in the Hugging Face "
                f"repository '{weights}'. Files available: {sorted(repo_files)[:20]}"
            )
        return hf_hub_download(weights, filename)
    except (GatedRepoError, LocalTokenNotFoundError) as e:
        raise RuntimeError(_sam3_gated_message(weights, e)) from e
    except RepositoryNotFoundError as e:
        # A private/gated repository is reported as "not found" when the request is not authenticated
        raise RuntimeError(_sam3_gated_message(weights, e)) from e
    except Exception as e:
        if "401" in str(e) or "403" in str(e) or "gated" in str(e).lower() or "authenticat" in str(e).lower():
            raise RuntimeError(_sam3_gated_message(weights, e)) from e
        raise


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


def _sam3_read_trunk(path: str) -> Dict[str, torch.Tensor]:
    """
    Read the tensors of SAM 3's image encoder from a checkpoint file.

    Only the tensors of the encoder are read, so the rest of the model (which is several times
    larger) is never loaded into memory.

    Parameters
    ----------
    path : str
        Local path to the checkpoint, either a ``.safetensors`` or a PyTorch file.

    Returns
    -------
    trunk : Dict[str, torch.Tensor]
        Tensors of the image encoder, with its prefix stripped from the keys, e.g.
        ``'blocks.0.attn.qkv.weight'``.
    """
    if path.endswith(".safetensors"):
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            prefix = _sam3_trunk_prefix(keys)
            return {k[len(prefix) :]: f.get_tensor(k) for k in keys if k.startswith(prefix)}

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    for key in ["model", "state_dict", "model_weights"]:
        if isinstance(checkpoint, dict) and key in checkpoint and isinstance(checkpoint[key], dict):
            checkpoint = checkpoint[key]
            break
    checkpoint = {k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)}
    prefix = _sam3_trunk_prefix(list(checkpoint))
    return {k[len(prefix) :]: v for k, v in checkpoint.items() if k.startswith(prefix)}


def _sam3_trunk_prefix(keys: List[str]) -> str:
    """
    Find the prefix under which the image encoder is stored within a SAM 3 checkpoint.

    Parameters
    ----------
    keys : List[str]
        Names of the tensors stored in the checkpoint.

    Returns
    -------
    prefix : str
        Prefix of the image encoder, e.g. ``'detector.backbone.vision_backbone.trunk.'``.
    """
    reference = "blocks.0.attn.qkv.weight"
    for k in keys:
        if k.endswith(reference):
            return k[: -len(reference)]
    raise RuntimeError(
        "Could not find SAM 3's image encoder within the provided weights: no tensor ending in "
        f"'{reference}' was found. This module expects the layer naming of the original SAM 3 "
        "checkpoint (e.g. 'detector.backbone.vision_backbone.trunk.blocks.0.attn.qkv.weight'), so "
        "the file may correspond to a different model or to a converted version of it. Some of the "
        f"keys found are: {sorted(keys)[:5]}"
    )


def _sam3_adapt_patch_embed(
    weight: torch.Tensor,
    in_chans: int,
    patch_size: int,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Adapt the patch embedding of SAM 3 to the number of channels and patch size used by the model.

    SAM 3 was trained on RGB images with ``14x14`` patches, so the projection needs to be adapted
    when the input has one channel (e.g. grayscale microscopy images) or when a different patch
    size is used.

    Parameters
    ----------
    weight : torch.Tensor
        Pretrained patch embedding of shape ``(embed_dim, 3, 14, 14)``.

    in_chans : int
        Number of channels of the images the model is going to be trained with. It can only be
        ``1`` or ``3``.

    patch_size : int
        Patch (token) size of the model.

    verbose : bool, optional
        Whether to print what is being adapted. Defaults to ``True``.

    Returns
    -------
    weight : torch.Tensor
        Patch embedding of shape ``(embed_dim, in_chans, patch_size, patch_size)``.
    """
    if in_chans == 1:
        # Adding up the three kernels reproduces exactly the response the pretrained model would
        # give to a grayscale image replicated into its three channels
        weight = weight.sum(dim=1, keepdim=True)
        if verbose:
            print(
                "    - patch embedding adapted from 3 (RGB) to 1 channel by adding up its three kernels, "
                "which is equivalent to replicating the grayscale image into the three input channels"
            )
    elif in_chans != 3:
        raise ValueError(
            f"SAM 3's pretrained weights can only be loaded with 1 or 3 input channels, but the images "
            f"have {in_chans}. SAM 3 was trained on RGB images, and BiaPy can only adapt its patch "
            "embedding automatically when the input is grayscale (1 channel), by adding up its three "
            f"kernels. With {in_chans} channels there is no meaningful way of doing it, so the data needs "
            "to be converted beforehand: keep the channel of interest (1 channel), combine them into an "
            "RGB image (3 channels), or set 'MODEL.VIT_PRETRAINED_WEIGHTS' to '' to train from scratch "
            f"with the {in_chans} channels."
        )

    pretrained_size = weight.shape[-1]
    if pretrained_size != patch_size:
        # Resize the kernel to the patch size used by the model, keeping the magnitude of its
        # response by compensating the change in the number of elements added up by the convolution
        weight = F.interpolate(
            weight.float(), size=(patch_size, patch_size), mode="bicubic", align_corners=False
        ) * (pretrained_size / patch_size) ** 2
        if verbose:
            print(
                f"    - patch embedding resized from {pretrained_size}x{pretrained_size} to "
                f"{patch_size}x{patch_size} to match the token size of the model"
            )
    return weight


def _sam3_adapt_pos_embed(
    pos_embed: torch.Tensor,
    grid_size: Tuple[int, int],
    num_prefix_tokens: int,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Interpolate SAM 3's position embedding to the token grid of the model.

    Parameters
    ----------
    pos_embed : torch.Tensor
        Pretrained position embedding of shape ``(1, 1 + 24 * 24, embed_dim)``, where the first
        entry corresponds to a class token.

    grid_size : Tuple[int, int]
        Number of tokens of the model along the y and x axes.

    num_prefix_tokens : int
        Number of tokens at the beginning of the model's sequence that are not part of the token
        grid (e.g. a class token).

    verbose : bool, optional
        Whether to print what is being adapted. Defaults to ``True``.

    Returns
    -------
    pos_embed : torch.Tensor
        Position embedding of shape ``(1, num_prefix_tokens + grid_h * grid_w, embed_dim)``.
    """
    embed_dim = pos_embed.shape[-1]
    prefix, grid = pos_embed[:, :1], pos_embed[:, 1:]
    src = int(math.sqrt(grid.shape[1]))
    if src * src != grid.shape[1]:
        raise ValueError(f"Unexpected position embedding with {grid.shape[1]} grid entries, a square number was expected")

    if (src, src) != tuple(grid_size):
        grid = grid.reshape(1, src, src, embed_dim).permute(0, 3, 1, 2)
        grid = F.interpolate(grid.float(), size=tuple(grid_size), mode="bicubic", align_corners=False)
        grid = grid.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], embed_dim)
        if verbose:
            print(
                f"    - position embedding interpolated from a {src}x{src} to a "
                f"{grid_size[0]}x{grid_size[1]} token grid"
            )

    return torch.cat([prefix.repeat(1, num_prefix_tokens, 1), grid], dim=1) if num_prefix_tokens > 0 else grid


def load_sam3_pretrained_encoder(
    model: nn.Module,
    weights: str,
    verbose: bool = True,
) -> Dict[str, int]:
    """
    Load SAM 3's pretrained image encoder into a BiaPy model.

    The weights are downloaded from the Hugging Face Hub (and cached, so it only happens once) and
    mapped into the ``patch_embed``, ``pos_embed``, ``ln_pre`` and ``blocks`` of the given model,
    adapting the patch embedding to the number of input channels and to the patch size used, and
    interpolating the position embedding to the model's token grid. All those values are taken from
    the model itself, so the weights always end up adapted to the model they are loaded into.

    Parameters
    ----------
    model : nn.Module
        Model to load the weights into. It must expose SAM 3's encoder as ``patch_embed``,
        ``pos_embed``, ``blocks`` and, optionally, ``ln_pre``/``norm_pre``.

    weights : str
        Identifier of a Hugging Face Hub repository, e.g. ``'facebook/sam3'`` or
        ``'facebook/sam3.1'``, or path to a local checkpoint file.

    verbose : bool, optional
        Whether to print a report of what was loaded. Defaults to ``True``.

    Returns
    -------
    report : Dict[str, int]
        Number of tensors ``loaded`` into the model and number of them left ``missing``.
    """
    if verbose:
        print(f"Loading SAM 3's pretrained image encoder from '{weights}' ...")

    # Everything needed to adapt the weights is taken from the model, so they can not disagree
    patch_size = model.patch_embed.patch_size  # type: ignore
    grid = model.patch_embed.grid_size  # type: ignore
    grid_size = (grid, grid) if isinstance(grid, int) else tuple(grid)
    in_chans = model.patch_embed.proj.weight.shape[1]  # type: ignore
    num_prefix_tokens = int(model.pos_embed.shape[1] - grid_size[0] * grid_size[1])  # type: ignore

    trunk = _sam3_read_trunk(_sam3_weights_path(weights))

    # Make sure the encoder stored is the one this module reproduces. SAM 3 and SAM 3.1 share the very
    # same image encoder, but a future version of it may not, and loading it partially would silently
    # give a model with randomly initialized blocks.
    ckpt_depth = 1 + max((int(k.split(".")[1]) for k in trunk if k.startswith("blocks.")), default=-1)
    ckpt_embed_dim = trunk["patch_embed.proj.weight"].shape[0] if "patch_embed.proj.weight" in trunk else -1
    if ckpt_depth != SAM3_VIT_PARAMS["depth"] or ckpt_embed_dim != SAM3_VIT_PARAMS["embed_dim"]:
        raise RuntimeError(
            f"The image encoder stored in '{weights}' does not match the one BiaPy builds for 'sam3_vit': it has "
            f"{ckpt_depth} blocks of {ckpt_embed_dim} dimensions, while SAM 3's has {SAM3_VIT_PARAMS['depth']} "
            f"blocks of {SAM3_VIT_PARAMS['embed_dim']}. These weights seem to come from a different model or from "
            "a version of SAM whose encoder changed, so they can not be loaded into this backbone."
        )

    new_state_dict = {}
    # Patch embedding. SAM 3 has no bias in its projection, so it is zeroed to keep it that way.
    if "patch_embed.proj.weight" in trunk:
        new_state_dict["patch_embed.proj.weight"] = _sam3_adapt_patch_embed(
            trunk["patch_embed.proj.weight"], in_chans, patch_size, verbose=verbose
        )
        if getattr(model.patch_embed.proj, "bias", None) is not None:
            new_state_dict["patch_embed.proj.bias"] = torch.zeros_like(model.patch_embed.proj.bias)

    # Position embedding
    if "pos_embed" in trunk:
        new_state_dict["pos_embed"] = _sam3_adapt_pos_embed(
            trunk["pos_embed"], grid_size, num_prefix_tokens, verbose=verbose
        )

    # Layer normalization applied before the blocks
    ln_pre_name = "ln_pre" if hasattr(model, "ln_pre") else "norm_pre"
    for suffix in ["weight", "bias"]:
        if f"ln_pre.{suffix}" in trunk and hasattr(model, ln_pre_name):
            new_state_dict[f"{ln_pre_name}.{suffix}"] = trunk[f"ln_pre.{suffix}"]

    # Transformer blocks. Their naming is the same in the checkpoint and in `SAM3Block`, so they
    # are copied as they are. The rotary frequencies are not: they are computed by each block for
    # the token grid actually used, which the ones stored in the checkpoint do not match.
    for k, v in trunk.items():
        if k.startswith("blocks.") and not k.endswith("freqs_cis"):
            new_state_dict[k] = v

    model_state = model.state_dict()
    to_load, skipped = {}, []
    for k, v in new_state_dict.items():
        if k in model_state and model_state[k].shape == v.shape:
            to_load[k] = v
        else:
            skipped.append(k)

    missing = [k for k in model_state if k.startswith(("patch_embed.", "blocks.", "pos_embed", ln_pre_name)) and k not in to_load]
    # The rotary frequencies are buffers computed by each block, never loaded
    missing = [k for k in missing if not k.endswith("freqs_cis")]

    model.load_state_dict(to_load, strict=False)

    if verbose:
        print(f"    - {len(to_load)} tensors of SAM 3's encoder loaded into the model")
        if missing:
            print(f"    - {len(missing)} encoder tensors were NOT found in the checkpoint: {missing[:6]}")
    if skipped:
        warnings.warn(
            f"{len(skipped)} tensors of SAM 3's checkpoint could not be loaded, as they do not exist in the "
            f"model or their shape differs: {skipped[:6]}"
        )

    return {"loaded": len(to_load), "missing": len(missing)}
