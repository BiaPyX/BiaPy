#!/usr/bin/env python3
"""
Compute class weights for CrossEntropyLoss from a folder of segmentation masks.

Usage:
  python compute_ce_weights.py /path/to/masks \
      --num-classes 6 \
      --ignore-index 255 \
      --strategy inverse \
      --normalize mean \
      --save weights.json

Strategies:
  - inverse  : total_pixels / class_pixels (classic inverse frequency)
  - median   : median_frequency_balancing = median(freqs) / freq[c]
  - effective: (1 - beta) / (1 - beta**class_pixels)  [Cui et al. 2019], with --beta

Normalization:
  - none     : raw weights
  - mean     : scale so mean(weight)=1 (nice default so loss scale stays similar)
  - sum1     : scale so sum(weight)=C
"""

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
import numpy as np
from biapy.data.data_manipulation import read_img_as_ndarray
from tqdm import tqdm

def discover_files(folder: Path, exts=(".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")) -> List[Path]:
    files = []
    for ext in exts:
        files.extend(folder.rglob(f"*{ext}"))
    return sorted(files)

def safe_bincount(arr: np.ndarray, minlength: Optional[int] = None) -> np.ndarray:
    arr_flat = arr.reshape(-1)
    # guard against negative labels (e.g., ignore index) in bincount
    arr_flat = arr_flat[arr_flat >= 0]
    if minlength is None:
        return np.bincount(arr_flat)
    return np.bincount(arr_flat, minlength=minlength)

def accumulate_counts(
    files: Iterable[Path],
    num_classes: Optional[int],
    ignore_index: Optional[int],
) -> Tuple[np.ndarray, int]:
    counts = None
    total_pixels = 0
    for p in tqdm(files):
        mask = read_img_as_ndarray(str(p.resolve()), is_3d=False).squeeze()
        mask[...,1] *= 2
        mask = np.max(mask, axis=-1)
        if mask.ndim != 2:
            raise ValueError(f"Mask {p} is not 2D; got shape={mask.shape}")
        mask = mask.astype(np.int64, copy=False)

        if ignore_index is not None:
            mask = np.where(mask == ignore_index, -1, mask)

        # infer classes if not provided
        if num_classes is None:
            max_id = mask.max() if mask.size else -1
            # ignore_index may be larger than real classes; we excluded it above
            inferred = int(max(0, max_id)) + 1
            if counts is None:
                counts = np.zeros(inferred, dtype=np.int64)
            elif inferred > len(counts):
                counts = np.pad(counts, (0, inferred - len(counts)), constant_values=0)

            bc = safe_bincount(mask, minlength=len(counts))
        else:
            if counts is None:
                counts = np.zeros(num_classes, dtype=np.int64)
            bc = safe_bincount(mask, minlength=num_classes)

        counts[: len(bc)] += bc
        total_pixels += (mask >= 0).sum()

    if counts is None:
        counts = np.zeros(0, dtype=np.int64)
    return counts, total_pixels
def compute_weights(
    counts: np.ndarray,
    strategy: str = "inverse",
    beta: float = 0.9999,
    eps: float = 1e-12,
    k_enet: float = 1.02,
    alpha_power: float = 0.5,
) -> np.ndarray:
    C = len(counts)
    if C == 0:
        return np.array([], dtype=np.float64)

    total = counts.sum()
    freqs = np.maximum(counts.astype(np.float64) / (total + eps), eps)

    if strategy == "inverse":
        w = 1.0 / freqs
    elif strategy == "median":
        med = float(np.median(freqs))
        w = med / freqs
    elif strategy == "effective":
        # Cui et al. 2019 – if using pixel counts, set beta extremely close to 1.
        safe_counts = np.maximum(counts.astype(np.float64), eps)
        w = (1.0 - beta) / (1.0 - np.power(beta, safe_counts))
    elif strategy == "enet":
        # ENet weighting: 1 / log(k + p_c). k ~ 1.02 works well.
        w = 1.0 / np.log(k_enet + freqs)
    elif strategy == "power":
        # Smoothed inverse frequency relative to mean frequency.
        w = (freqs / freqs.mean()) ** (-alpha_power)
    else:
        raise ValueError(f"Unknown strategy '{strategy}'")

    return w

def normalize_weights(w: np.ndarray, mode: str) -> np.ndarray:
    if w.size == 0:
        return w
    if mode == "none":
        return w
    elif mode == "mean":
        m = w.mean()
        return w / (m if m != 0 else 1.0)
    elif mode == "sum1":
        C = w.size
        s = w.sum()
        return w * (C / s) if s != 0 else w
    else:
        raise ValueError(f"Unknown normalization '{mode}'")

def blend_to_one(w: np.ndarray, lam: float) -> np.ndarray:
    # lam=0 -> all ones; lam=1 -> original weights
    return (1.0 - lam) + lam * w

def clip_weights(w: np.ndarray, wmin: float, wmax: float) -> np.ndarray:
    return np.clip(w, wmin, wmax)

def main():
    parser = argparse.ArgumentParser(description="Compute class weights for CrossEntropyLoss from mask folder.")
    parser.add_argument("folder", type=str, help="Path to folder containing mask images")
    parser.add_argument("--num-classes", type=int, default=None, help="Number of classes (if omitted, infer from data)")
    parser.add_argument("--ignore-index", type=int, default=None, help="Ignore index used in masks (e.g., 255)")
    parser.add_argument("--strategy", type=str, default="enet",
                    choices=["inverse", "median", "effective", "enet", "power"])
    parser.add_argument("--beta", type=float, default=0.9999)
    parser.add_argument("--k-enet", type=float, default=1.02, help="k for ENet: 1/log(k+freq)")
    parser.add_argument("--alpha-power", type=float, default=0.5, help="alpha for power-law")
    parser.add_argument("--mix-to-one", type=float, default=1.0,
                        help="lambda for blending weights toward 1.0; 1=no blend, 0=all ones")
    parser.add_argument("--clip-min", type=float, default=None)
    parser.add_argument("--clip-max", type=float, default=None)
    parser.add_argument("--normalize", type=str, default="mean",
                        choices=["none", "mean", "sum1"], help="Normalization of weights")
    parser.add_argument("--save", type=str, default=None, help="Optional path to save weights JSON")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    files = discover_files(folder)
    if not files:
        raise FileNotFoundError(f"No mask files found under {folder}")

    counts, total = accumulate_counts(files, args.num_classes, args.ignore_index)

    weights = compute_weights(counts, strategy=args.strategy, beta=args.beta,
                          k_enet=args.k_enet, alpha_power=args.alpha_power)
    weights = normalize_weights(weights, mode=args.normalize)

    # optional blend/clip to keep them near 1
    weights = blend_to_one(weights, lam=args.mix_to_one)
    if args.clip_min is not None and args.clip_max is not None:
        weights = clip_weights(weights, args.clip_min, args.clip_max)

    # Print a concise summary
    print("\n=== Pixel Counts per Class ===")
    for i, c in enumerate(counts):
        print(f"class {i}: {c}")
    print(f"total labeled pixels: {total}")

    print("\n=== Weights (as Python list) ===")
    py_list = [float(f"{x:.8f}") for x in weights.tolist()]
    print(py_list)

    print("\n=== PyTorch snippet ===")
    print("import torch")
    print(f"weights = torch.tensor({py_list}, dtype=torch.float32)")
    print("criterion = torch.nn.CrossEntropyLoss(weight=weights)")

    if args.save:
        out = {
            "counts": counts.tolist(),
            "total_pixels": int(total),
            "strategy": args.strategy,
            "normalize": args.normalize,
            "beta": args.beta,
            "weights": py_list,
        }
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved to: {args.save}")

if __name__ == "__main__":
    main()
