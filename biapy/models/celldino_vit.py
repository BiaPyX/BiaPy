"""Cell-DINO (channel-adaptive DINOv2 ViT-L/16) image encoder for BiaPy."""

import os
import re
import warnings
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

CELLDINO_VIT_PARAMS = {
    "patch_size": 16,
    "embed_dim": 1024,
    "depth": 24,
    "num_heads": 16,
    "mlp_ratio": 4.0,
    "qkv_bias": True,
    "norm_eps": 1e-6,
    "in_chans": 1,
    "init_values": 1.0,
    "pretrain_grid_size": 14,
}


def _celldino_weights_path(weights: str) -> str:
    """Local-file-only lookup (Cell-DINO weights are not on the HF Hub)."""
    if os.path.isfile(weights):
        return weights
    raise RuntimeError(_celldino_gated_message(weights))


def _celldino_gated_message(weights: str) -> str:
    return (
        f"Could not find a Cell-DINO checkpoint at '{weights}'.\n"
        "Cell-DINO weights are not downloaded automatically (not hosted on the HF Hub):\n"
        "  1) Request access at https://ai.meta.com/resources/models-and-libraries/cell-dino-downloads/\n"
        "  2) Download the 'channel_adaptive_dino_vitl16' checkpoint from the e-mailed URL.\n"
        "  3) Set 'MODEL.VIT_PRETRAINED_WEIGHTS' to that local file path."
    )


def _celldino_read_encoder(path: str) -> Dict[str, torch.Tensor]:
    """Read the ViT encoder tensors from a checkpoint, flattening block_chunks nesting."""
    if path.endswith(".safetensors"):
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as f:
            checkpoint = {k: f.get_tensor(k) for k in f.keys()}
    else:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        for key in ["teacher", "student", "model", "state_dict"]:
            if isinstance(checkpoint, dict) and key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
        checkpoint = {k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)}

    prefix = _celldino_encoder_prefix(list(checkpoint))
    encoder = {}
    for k, v in checkpoint.items():
        if not k.startswith(prefix):
            continue
        k = k[len(prefix) :]
        # 'blocks.<chunk>.<index>.<rest>' -> 'blocks.<index>.<rest>' (index is already global)
        m = re.match(r"^blocks\.\d+\.(\d+)\.(.*)$", k)
        if m:
            k = f"blocks.{m.group(1)}.{m.group(2)}"
        encoder[k] = v
    return encoder


def _celldino_encoder_prefix(keys: List[str]) -> str:
    reference = "cls_token"
    for k in keys:
        if k.endswith(reference):
            return k[: -len(reference)]
    raise RuntimeError(
        f"Could not find Cell-DINO's ViT encoder in the provided weights: no tensor ending in "
        f"'{reference}' was found. Some keys found: {sorted(keys)[:5]}"
    )


def _celldino_adapt_pos_embed(
    pos_embed: torch.Tensor,
    grid_size: Tuple[int, int],
    num_prefix_tokens: int,
    verbose: bool = True,
) -> torch.Tensor:
    """Bicubic-interpolate the pretrained position embedding to the model's token grid."""
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


def load_celldino_pretrained_encoder(
    model: torch.nn.Module,
    weights: str,
    verbose: bool = True,
) -> Dict[str, int]:
    """Load Cell-DINO's pretrained ViT encoder into a BiaPy model (`vit` or `unetr` backbone)."""
    if verbose:
        print(f"Loading Cell-DINO's pretrained ViT encoder from '{weights}' ...")

    in_chans = model.patch_embed.proj.weight.shape[1]  # type: ignore
    if in_chans != CELLDINO_VIT_PARAMS["in_chans"]:
        raise ValueError(
            f"Cell-DINO's pretrained weights need {CELLDINO_VIT_PARAMS['in_chans']} input channel, "
            f"model was built with {in_chans}."
        )
    grid = model.patch_embed.grid_size  # type: ignore
    grid_size = (grid, grid) if isinstance(grid, int) else tuple(grid)
    num_prefix_tokens = int(model.pos_embed.shape[1] - grid_size[0] * grid_size[1])  # type: ignore

    encoder = _celldino_read_encoder(_celldino_weights_path(weights))

    ckpt_depth = 1 + max((int(k.split(".")[1]) for k in encoder if k.startswith("blocks.")), default=-1)
    ckpt_embed_dim = encoder["cls_token"].shape[-1] if "cls_token" in encoder else -1
    if ckpt_depth != CELLDINO_VIT_PARAMS["depth"] or ckpt_embed_dim != CELLDINO_VIT_PARAMS["embed_dim"]:
        raise RuntimeError(
            f"Encoder in '{weights}' does not match 'celldino_vit': {ckpt_depth} blocks of "
            f"{ckpt_embed_dim} dims found, expected {CELLDINO_VIT_PARAMS['depth']} of "
            f"{CELLDINO_VIT_PARAMS['embed_dim']}."
        )

    # UNETR has no final 'norm' (only consumes intermediate block outputs), so only copy it if present
    tensor_names = ["patch_embed.proj.weight", "patch_embed.proj.bias", "cls_token"]
    if hasattr(model, "norm") and not isinstance(model.norm, torch.nn.Identity):
        tensor_names += ["norm.weight", "norm.bias"]
    new_state_dict = {}
    for name in tensor_names:
        if name in encoder:
            new_state_dict[name] = encoder[name]

    if "pos_embed" in encoder:
        new_state_dict["pos_embed"] = _celldino_adapt_pos_embed(
            encoder["pos_embed"], grid_size, num_prefix_tokens, verbose=verbose
        )

    for k, v in encoder.items():
        if k.startswith("blocks."):
            new_state_dict[k] = v

    model_state = model.state_dict()
    to_load, skipped = {}, []
    for k, v in new_state_dict.items():
        if k in model_state and model_state[k].shape == v.shape:
            to_load[k] = v
        else:
            skipped.append(k)

    tracked_prefixes = ("patch_embed.", "blocks.", "pos_embed", "cls_token", "norm.")
    missing = [k for k in model_state if k.startswith(tracked_prefixes) and k not in to_load]

    model.load_state_dict(to_load, strict=False)

    if verbose:
        print(f"    - {len(to_load)} tensors loaded")
        if missing:
            print(f"    - {len(missing)} encoder tensors NOT found in the checkpoint: {missing[:6]}")
    if skipped:
        warnings.warn(f"{len(skipped)} checkpoint tensors could not be loaded (missing or shape mismatch): {skipped[:6]}")

    return {"loaded": len(to_load), "missing": len(missing)}
