"""
Cell-DINO (https://doi.org/10.1371/journal.pcbi.1013828): a plain ViT-L/16 with LayerScale on 1-channel images (timm
blocks). Geometry and weight loader; the weights must be downloaded manually (not on the Hugging Face Hub).
"""

import os
import re
from typing import Dict

import torch

from biapy.models.vit_backbones.weights import (
    interpolate_grid_pos_embed,
    load_matching_tensors,
    max_block_index,
    read_checkpoint_tensors,
    strip_tensor_prefix,
)

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


def _celldino_read_encoder(weights: str) -> Dict[str, torch.Tensor]:
    """Encoder tensors of a local Cell-DINO checkpoint, with its ``block_chunks`` nesting flattened."""
    if not os.path.isfile(weights):
        raise RuntimeError(
            f"Could not find a Cell-DINO checkpoint at '{weights}'.\n"
            "Cell-DINO weights are not downloaded automatically (not hosted on the HF Hub):\n"
            "  1) Request access at https://ai.meta.com/resources/models-and-libraries/cell-dino-downloads/\n"
            "  2) Download the 'channel_adaptive_dino_vitl16' checkpoint from the e-mailed URL.\n"
            "  3) Set 'MODEL.VIT_PRETRAINED_WEIGHTS' to that local file path."
        )
    checkpoint = read_checkpoint_tensors(weights, wrapper_keys=["teacher", "student", "model", "state_dict"])
    encoder = {}
    for k, v in strip_tensor_prefix(checkpoint, "cls_token", "Cell-DINO's ViT encoder").items():
        # 'blocks.<chunk>.<i>.*' -> 'blocks.<i>.*'
        m = re.match(r"^blocks\.\d+\.(\d+)\.(.*)$", k)
        if m:
            k = f"blocks.{m.group(1)}.{m.group(2)}"
        encoder[k] = v
    return encoder


def load_celldino_pretrained_encoder(
    model: torch.nn.Module,
    weights: str,
    verbose: bool = True,
) -> Dict[str, int]:
    """
    Load Cell-DINO's encoder (patch embedding, class token, position embedding interpolated to the model's grid,
    blocks and, if present, final norm) from the local ``weights`` checkpoint into ``model``.
    """
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

    encoder = _celldino_read_encoder(weights)
    ckpt_depth = max_block_index(encoder)
    ckpt_embed_dim = encoder["cls_token"].shape[-1] if "cls_token" in encoder else -1
    if ckpt_depth != CELLDINO_VIT_PARAMS["depth"] or ckpt_embed_dim != CELLDINO_VIT_PARAMS["embed_dim"]:
        raise RuntimeError(
            f"Encoder in '{weights}' does not match 'celldino_vit': {ckpt_depth} blocks of "
            f"{ckpt_embed_dim} dims found, expected {CELLDINO_VIT_PARAMS['depth']} of "
            f"{CELLDINO_VIT_PARAMS['embed_dim']}."
        )

    # UNETR has no final norm
    tensor_names = ["patch_embed.proj.weight", "patch_embed.proj.bias", "cls_token"]
    if hasattr(model, "norm") and not isinstance(model.norm, torch.nn.Identity):
        tensor_names += ["norm.weight", "norm.bias"]
    new_state_dict = {name: encoder[name] for name in tensor_names if name in encoder}
    if "pos_embed" in encoder:
        new_state_dict["pos_embed"] = interpolate_grid_pos_embed(
            encoder["pos_embed"], grid_size, num_prefix_tokens, verbose=verbose
        )
    for k, v in encoder.items():
        if k.startswith("blocks."):
            new_state_dict[k] = v

    return load_matching_tensors(
        model,
        new_state_dict,
        ("patch_embed.", "blocks.", "pos_embed", "cls_token", "norm."),
        "Cell-DINO's encoder",
        verbose=verbose,
    )
