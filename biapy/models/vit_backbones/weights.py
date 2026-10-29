"""Helpers shared by the pretrained ViT backbone loaders (``biapy.models.vit_backbones``)."""

import os
import warnings
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def fetch_vit_weights(
    weights: str,
    hf_filenames: Sequence[str] = (),
    named_urls: Optional[Dict[str, str]] = None,
    cache_subdir: str = "",
    access_message: Optional[Callable[[str, Exception], str]] = None,
) -> str:
    """
    Local path to a checkpoint, downloading it if needed.

    Parameters
    ----------
    weights : str
        Local path, name in ``named_urls``, URL, or Hugging Face repository (if ``hf_filenames`` is given).

    hf_filenames : Sequence[str], optional
        Files to look for in the Hugging Face repository, by preference.

    named_urls : Dict[str, str], optional
        URLs of released checkpoints by name.

    cache_subdir : str, optional
        Subfolder of PyTorch's hub checkpoint folder for URL downloads.

    access_message : Callable[[str, Exception], str], optional
        Builds the error message for gated repositories.

    Returns
    -------
    path : str
        Local path to the checkpoint.
    """
    if os.path.isfile(weights):
        return weights
    if named_urls and weights in named_urls:
        weights = named_urls[weights]

    if weights.startswith(("http://", "https://")):
        from urllib.parse import urlparse

        dst = os.path.join(torch.hub.get_dir(), "checkpoints", cache_subdir, os.path.basename(urlparse(weights).path))
        if not os.path.isfile(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            print(f"    - downloading '{weights}' to '{dst}'")
            torch.hub.download_url_to_file(weights, dst)
        return dst

    if not hf_filenames:
        raise ValueError(
            f"Pretrained weights '{weights}' not found: it is neither a local file nor a URL"
            + (f" nor one of {sorted(named_urls)}" if named_urls else "")
        )

    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import GatedRepoError, LocalTokenNotFoundError, RepositoryNotFoundError

    try:
        repo_files = HfApi().list_repo_files(weights)
        filename = next((f for f in hf_filenames if f in repo_files), None)
        if filename is None:
            raise FileNotFoundError(
                f"Could not find any of the expected weight files {list(hf_filenames)} in the Hugging Face "
                f"repository '{weights}'. Files available: {sorted(repo_files)[:20]}"
            )
        return hf_hub_download(weights, filename)
    except (GatedRepoError, LocalTokenNotFoundError, RepositoryNotFoundError) as e:
        # Gated repositories are reported as "not found" without authentication
        if access_message is None:
            raise
        raise RuntimeError(access_message(weights, e)) from e
    except Exception as e:
        gated = "401" in str(e) or "403" in str(e) or "gated" in str(e).lower() or "authenticat" in str(e).lower()
        if gated and access_message is not None:
            raise RuntimeError(access_message(weights, e)) from e
        raise


def read_checkpoint_tensors(
    path: str,
    wrapper_keys: Sequence[str] = ("model", "state_dict"),
    keep: Optional[Callable[[str], bool]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Tensors of a ``.pth``/``.safetensors`` checkpoint.

    Parameters
    ----------
    path : str
        Checkpoint path.

    wrapper_keys : Sequence[str], optional
        Keys a training checkpoint may nest the weights under (first found is used).

    keep : Callable[[str], bool], optional
        Name filter.

    Returns
    -------
    tensors : Dict[str, torch.Tensor]
        Tensors by name.
    """
    if path.endswith(".safetensors"):
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as f:
            return {k: f.get_tensor(k) for k in f.keys() if keep is None or keep(k)}

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    for key in wrapper_keys:
        if isinstance(checkpoint, dict) and key in checkpoint and isinstance(checkpoint[key], dict):
            checkpoint = checkpoint[key]
            break
    return {k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor) and (keep is None or keep(k))}


def strip_tensor_prefix(tensors: Dict[str, torch.Tensor], reference: str, what: str) -> Dict[str, torch.Tensor]:
    """
    Keep the tensors under the prefix of the one ending in ``reference``, with the prefix removed.

    Parameters
    ----------
    tensors : Dict[str, torch.Tensor]
        Checkpoint tensors.

    reference : str
        Name ending of an encoder tensor, e.g. ``'blocks.0.attn.qkv.weight'``.

    what : str
        Encoder name, for the error message.

    Returns
    -------
    encoder : Dict[str, torch.Tensor]
        Encoder tensors.
    """
    prefix = next((k[: -len(reference)] for k in tensors if k.endswith(reference)), None)
    if prefix is None:
        raise RuntimeError(
            f"Could not find {what} in the weights: no tensor ending in '{reference}'. Some keys: {sorted(tensors)[:5]}"
        )
    return {k[len(prefix) :]: v for k, v in tensors.items() if k.startswith(prefix)}


def adapt_rgb_patch_embed(
    weight: torch.Tensor,
    in_chans: int,
    what: str,
    patch_size: Optional[int] = None,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Adapt an RGB patch embedding to 1 or 3 input channels and, optionally, another token size.

    Parameters
    ----------
    weight : torch.Tensor
        ``(embed_dim, 3, k, k)`` patch embedding.

    in_chans : int
        1 or 3.

    what : str
        Encoder name, for the messages.

    patch_size : int, optional
        Token size to resize the kernel to.

    verbose : bool, optional
        Whether to print what is adapted.

    Returns
    -------
    weight : torch.Tensor
        ``(embed_dim, in_chans, patch_size, patch_size)`` patch embedding.
    """
    if in_chans == 1:
        # Summing the kernels = feeding the grayscale image replicated into 3 channels
        weight = weight.sum(dim=1, keepdim=True)
        if verbose:
            print("    - patch embedding adapted from 3 to 1 channel (kernels summed)")
    elif in_chans != 3:
        raise ValueError(
            f"{what}'s pretrained weights need 1 or 3 input channels, got {in_chans}. Convert the data or set "
            "'MODEL.VIT_PRETRAINED_WEIGHTS' to '' to train from scratch."
        )
    pretrained_size = weight.shape[-1]
    if patch_size is not None and pretrained_size != patch_size:
        # Rescaled to keep the response magnitude
        weight = F.interpolate(weight.float(), size=(patch_size, patch_size), mode="bicubic", align_corners=False) * (
            pretrained_size / patch_size
        ) ** 2
        if verbose:
            print(f"    - patch embedding resized from {pretrained_size}x{pretrained_size} to {patch_size}x{patch_size}")
    return weight


def interpolate_grid_pos_embed(
    pos_embed: torch.Tensor,
    grid_size: Tuple[int, int],
    num_prefix_tokens: int,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Interpolate a ``(1, 1 + h * w, D)`` position embedding (class token first) to another token grid.

    Parameters
    ----------
    pos_embed : torch.Tensor
        Pretrained position embedding.

    grid_size : Tuple[int, int]
        Target token grid (y, x).

    num_prefix_tokens : int
        Prefix tokens of the model (all get the class token entry).

    verbose : bool, optional
        Whether to print what is adapted.

    Returns
    -------
    pos_embed : torch.Tensor
        ``(1, num_prefix_tokens + grid_h * grid_w, D)`` position embedding.
    """
    embed_dim = pos_embed.shape[-1]
    prefix, grid = pos_embed[:, :1], pos_embed[:, 1:]
    src = int(round(grid.shape[1] ** 0.5))
    if src * src != grid.shape[1]:
        raise ValueError(f"Unexpected position embedding with {grid.shape[1]} grid entries, a square number was expected")
    if (src, src) != tuple(grid_size):
        grid = grid.reshape(1, src, src, embed_dim).permute(0, 3, 1, 2)
        grid = F.interpolate(grid.float(), size=tuple(grid_size), mode="bicubic", align_corners=False)
        grid = grid.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], embed_dim)
        if verbose:
            print(f"    - position embedding interpolated from {src}x{src} to {grid_size[0]}x{grid_size[1]}")
    return torch.cat([prefix.repeat(1, num_prefix_tokens, 1), grid], dim=1) if num_prefix_tokens > 0 else grid


def load_matching_tensors(
    model: nn.Module,
    new_state_dict: Dict[str, torch.Tensor],
    tracked_prefixes: Tuple[str, ...],
    what: str,
    ignore_suffixes: Tuple[str, ...] = (),
    verbose: bool = True,
) -> Dict[str, int]:
    """
    Load the tensors that match the model's names and shapes, reporting the rest.

    Parameters
    ----------
    model : nn.Module
        Target model.

    new_state_dict : Dict[str, torch.Tensor]
        Tensors with the model's naming.

    tracked_prefixes : Tuple[str, ...]
        Model tensors expected to be loaded (reported as missing otherwise).

    what : str
        What is loaded, for the messages.

    ignore_suffixes : Tuple[str, ...], optional
        Model tensors never loaded (e.g. computed buffers).

    verbose : bool, optional
        Whether to print a report.

    Returns
    -------
    report : Dict[str, int]
        Number of tensors ``loaded`` and ``missing``.
    """
    model_state = model.state_dict()
    to_load: Dict[str, torch.Tensor] = {}
    skipped: List[str] = []
    for k, v in new_state_dict.items():
        if k in model_state and model_state[k].shape == v.shape:
            to_load[k] = v
        else:
            skipped.append(k)
    missing = [
        k for k in model_state if k.startswith(tracked_prefixes) and k not in to_load and not k.endswith(ignore_suffixes)
    ]
    model.load_state_dict(to_load, strict=False)
    if verbose:
        print(f"    - {len(to_load)} tensors of {what} loaded into the model")
        if missing:
            print(f"    - {len(missing)} tensors were NOT found in the checkpoint: {missing[:6]}")
    if skipped:
        warnings.warn(
            f"{len(skipped)} tensors of {what} not loaded (not in the model or other shape): {skipped[:6]}"
        )
    return {"loaded": len(to_load), "missing": len(missing)}


def max_block_index(names: Iterable[str]) -> int:
    """Number of transformer blocks in a checkpoint (``'blocks.<i>.*'`` names)."""
    return 1 + max((int(k.split(".")[1]) for k in names if k.startswith("blocks.")), default=-1)
