"""
Visualize the membrane-repair corruption augmentors (biapy/data/generators/membrane_augmentors.py)
on a single input membrane image, so their effect/parameters can be sanity-checked before wiring
them into a training run.

For each augmentor, ``--n-examples`` independent draws are applied (forcing ``prob=1.0`` so the
effect is always visible) and saved next to the original as individual TIFFs, plus one combined
overview PNG grid (rows = augmentor, columns = original + each draw) for a quick look.

Example
-------
python test_membrane_augmentors.py --membrane-image /path/to/membrane.tif \
    --mito-image /path/to/mito.tif --out-dir /path/to/out
"""
import argparse
import os
import random
import sys

import numpy as np
import matplotlib.pyplot as plt

_DEFAULT_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize the membrane-repair corruption augmentors on a single input image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--code-dir", default=_DEFAULT_CODE_DIR, help="Path to the BiaPy repo root.")
    parser.add_argument(
        "--membrane-image", required=True,
        help="Path to the membrane image. 2D (y, x) or 3D (z, y, x) single-channel.",
    )
    parser.add_argument(
        "--mito-image", default=None,
        help="Optional path to the mito image (same shape as --membrane-image). "
        "Omit to skip mito_border_erasure_aug.",
    )
    parser.add_argument(
        "--z-slice", type=int, default=None,
        help="Which z-slice to test if the input is 3D. Defaults to the middle slice.",
    )
    parser.add_argument("--out-dir", required=True, help="Directory to save the per-draw TIFFs and overview PNG.")
    parser.add_argument("--n-examples", type=int, default=4, help="Number of independent draws per augmentor.")
    parser.add_argument("--seed", type=int, default=42)

    # Augmentor parameters (mirrors PROBLEM.IMAGE_TO_IMAGE.MEMBRANE_REPAIR defaults, see biapy/config/config.py).
    parser.add_argument("--gap-length-range", type=float, nargs=2, default=(0.1, 0.5), metavar=("MIN", "MAX"))
    parser.add_argument("--gap-thickness-range", type=float, nargs=2, default=(2, 8), metavar=("MIN", "MAX"))
    parser.add_argument("--gap-n-lines", type=int, nargs=2, default=(1, 3), metavar=("MIN", "MAX"))

    parser.add_argument("--bridge-length-range", type=float, nargs=2, default=(3, 15), metavar=("MIN", "MAX"))
    parser.add_argument("--bridge-line-width", type=int, default=1)

    parser.add_argument("--island-size-range", type=float, nargs=2, default=(2, 6), metavar=("MIN", "MAX"))

    parser.add_argument("--mito-length-range", type=float, nargs=2, default=(5, 20), metavar=("MIN", "MAX"))

    parser.add_argument("--skeleton-radius-range", type=int, nargs=2, default=(1, 2), metavar=("MIN", "MAX"))
    parser.add_argument("--skeleton-n-spurs", type=int, nargs=2, default=(0, 2), metavar=("MIN", "MAX"))
    parser.add_argument("--skeleton-spur-length-range", type=int, nargs=2, default=(2, 4), metavar=("MIN", "MAX"))

    return parser.parse_args()


def load_2d_channel(imread, path: str, z: "int | None") -> np.ndarray:
    """Load ``path`` and return a single ``(h, w)`` float32 slice normalized to ``[0, 1]``."""
    img, _ = imread(path)
    img = np.squeeze(img)
    if img.ndim == 3:
        z = img.shape[0] // 2 if z is None else z
        img = img[z]
    elif img.ndim != 2:
        raise ValueError(f"Expected a 2D or 3D single-channel image, got shape {img.shape} for {path}")
    img = img.astype(np.float32)
    if img.max() > 1:
        img = img / img.max()
    return img


def main() -> None:
    args = parse_args()

    sys.path.insert(0, args.code_dir)
    from biapy.data.data_manipulation import imread, save_tif
    from biapy.data.generators.membrane_augmentors import (
        mito_border_erasure,
        skeleton_perturbation,
        slice_dropout,
        spurious_bridge,
        spurious_island,
        synthetic_gap,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)
    random.seed(args.seed)

    membrane = load_2d_channel(imread, args.membrane_image, args.z_slice)
    h, w = membrane.shape
    mito = load_2d_channel(imread, args.mito_image, args.z_slice) if args.mito_image else None

    n_channels = 2 if mito is not None else 1
    base = np.zeros((h, w, n_channels), dtype=np.float32)
    base[..., 0] = membrane
    if mito is not None:
        base[..., 1] = mito
    membrane_idx, mito_idx = 0, (1 if mito is not None else None)

    def run_gap(img):
        return synthetic_gap(
            img, membrane_idx, ndim=2, prob=1.0,
            length_range=args.gap_length_range, thickness_range=args.gap_thickness_range,
            n_lines=args.gap_n_lines,
        )

    def run_bridge(img):
        return spurious_bridge(
            img, membrane_idx, ndim=2, prob=1.0,
            length_range=args.bridge_length_range, line_width=args.bridge_line_width,
        )

    def run_island(img):
        return spurious_island(img, membrane_idx, ndim=2, prob=1.0, size_range=args.island_size_range)

    def run_mito_border_erasure(img):
        return mito_border_erasure(
            img, membrane_idx, mito_idx, ndim=2, prob=1.0, length_range=args.mito_length_range,
        )

    def run_skeleton_perturb(img):
        return skeleton_perturbation(
            img, membrane_idx, ndim=2, prob=1.0,
            radius_range=args.skeleton_radius_range, n_spurs=args.skeleton_n_spurs,
            spur_length_range=args.skeleton_spur_length_range,
        )

    def run_slice_dropout(img):
        return slice_dropout(img, (membrane_idx,), prob=1.0, ndim=2)

    augmentors = {
        "gap_aug": run_gap,
        "bridge_aug": run_bridge,
        "island_aug": run_island,
        "skeleton_perturb_aug": run_skeleton_perturb,
        "slice_dropout_aug": run_slice_dropout,
    }
    if mito is not None:
        augmentors["mito_border_erasure_aug"] = run_mito_border_erasure

    save_tif(np.expand_dims(base[..., 0:1], 0), args.out_dir, ["original.tif"], verbose=False)

    results = {}
    for name, fn in augmentors.items():
        draws = []
        for i in range(args.n_examples):
            out = fn(base)
            draws.append(out[..., membrane_idx])
            save_tif(np.expand_dims(out[..., 0:1], 0), args.out_dir, [f"{name}_{i}.tif"], verbose=False)
        results[name] = draws

    n_rows = len(augmentors)
    n_cols = args.n_examples + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.2 * n_rows), squeeze=False)
    for row, (name, draws) in enumerate(results.items()):
        axes[row][0].imshow(membrane, cmap="gray", vmin=0, vmax=1)
        axes[row][0].set_ylabel(name, fontsize=9)
        axes[row][0].set_xticks([])
        axes[row][0].set_yticks([])
        if row == 0:
            axes[row][0].set_title("original")
        for col, draw in enumerate(draws, start=1):
            axes[row][col].imshow(draw, cmap="gray", vmin=0, vmax=1)
            axes[row][col].set_xticks([])
            axes[row][col].set_yticks([])
            if row == 0:
                axes[row][col].set_title(f"draw {col}")
    fig.tight_layout()
    overview_path = os.path.join(args.out_dir, "overview.png")
    fig.savefig(overview_path, dpi=150)
    print(f"Saved per-augmentor TIFFs and overview grid to {args.out_dir} ({overview_path})")


if __name__ == "__main__":
    main()
