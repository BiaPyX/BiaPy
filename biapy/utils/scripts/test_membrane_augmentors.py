"""
Visualize the membrane-repair corruption augmentors (biapy/data/generators/membrane_augmentors.py)
on a single input membrane image, so their effect/parameters can be sanity-checked before wiring
them into a training run.

For each augmentor, ``--n-examples`` independent draws are applied (forcing ``prob=1.0`` so the
effect is always visible) and saved next to the original as individual TIFFs, plus one combined
overview SVG grid (rows = augmentor, columns = original, draw 1, draw 1 diff, draw 2, draw 2
diff, ...) for a quick look. In each diff panel only the altered pixels are shown: pixels added by
the augmentor in red, pixels it removed in blue.

``artifact_aug`` draws alternate band/blobs so both are always shown. Pass ``--raw-image`` to also
see its effect on a second "raw" channel (extra ``artifact_aug (raw)`` row).

Example
-------
python test_membrane_augmentors.py --membrane-image /path/to/membrane.tif \
    --raw-image /path/to/raw.tif --out-dir /path/to/out
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
        "--raw-image", default=None,
        help="Optional path to a second 'raw' channel (same shape as --membrane-image), to also "
        "show artifact_aug's effect on it (as an extra 'artifact_aug (raw)' row).",
    )
    parser.add_argument(
        "--z-slice", type=int, default=None,
        help="Which z-slice to test if the input is 3D. Defaults to the middle slice.",
    )
    parser.add_argument("--out-dir", required=True, help="Directory to save the per-draw TIFFs and overview SVG.")
    parser.add_argument("--n-examples", type=int, default=2, help="Number of independent draws per augmentor.")
    parser.add_argument("--seed", type=int, default=42)

    # Augmentor parameters (mirrors PROBLEM.IMAGE_TO_IMAGE.MEMBRANE_REPAIR defaults, see biapy/config/config.py).
    parser.add_argument("--gap-length-range", type=float, nargs=2, default=(0.3, 1.0), metavar=("MIN", "MAX"))
    parser.add_argument("--gap-thickness-range", type=float, nargs=2, default=(4, 9), metavar=("MIN", "MAX"))
    parser.add_argument("--gap-n-lines", type=int, nargs=2, default=(1, 3), metavar=("MIN", "MAX"))

    parser.add_argument("--bridge-length-range", type=float, nargs=2, default=(0.3, 1.0), metavar=("MIN", "MAX"))
    parser.add_argument("--bridge-thickness-range", type=float, nargs=2, default=(4, 9), metavar=("MIN", "MAX"))
    parser.add_argument("--bridge-n-lines", type=int, nargs=2, default=(1, 3), metavar=("MIN", "MAX"))

    # No --artifact-band-prob: draws alternate band/blobs directly so both are always shown.
    parser.add_argument(
        "--artifact-band-thickness-range", type=float, nargs=2, default=(50, 70), metavar=("MIN", "MAX")
    )
    parser.add_argument(
        "--artifact-blob-size-range", type=float, nargs=2, default=(0.1, 0.3), metavar=("MIN", "MAX"),
        help="Blob radius as a fraction of min(height, width).",
    )
    parser.add_argument("--artifact-blob-n-range", type=int, nargs=2, default=(1, 3), metavar=("MIN", "MAX"))

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


def diff_rgb(orig: np.ndarray, draw: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    ``(h, w, 3)`` RGB image showing only the pixels ``draw`` altered relative to ``orig``
    (both thresholded to binary): added pixels in red, removed pixels in blue, everything else
    black.
    """
    orig_b = orig > threshold
    draw_b = draw > threshold
    rgb = np.zeros((*orig.shape, 3), dtype=np.float32)
    rgb[draw_b & ~orig_b] = (1.0, 0.0, 0.0)
    rgb[orig_b & ~draw_b] = (0.0, 0.0, 1.0)
    return rgb


def main() -> None:
    args = parse_args()

    sys.path.insert(0, args.code_dir)
    from biapy.data.data_manipulation import imread, save_tif
    from biapy.data.generators.membrane_augmentors import (
        artifact_corruption,
        skeleton_perturbation,
        slice_dropout,
        spurious_bridge,
        synthetic_gap,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)
    random.seed(args.seed)

    membrane = load_2d_channel(imread, args.membrane_image, args.z_slice)
    raw = load_2d_channel(imread, args.raw_image, args.z_slice) if args.raw_image else None
    membrane_idx, raw_idx = 0, (1 if raw is not None else None)
    base = np.stack([membrane, raw], axis=-1) if raw is not None else membrane[..., None]

    def run_gap(img):
        return synthetic_gap(
            img, membrane_idx, ndim=2, prob=1.0,
            length_range=args.gap_length_range, thickness_range=args.gap_thickness_range,
            n_lines=args.gap_n_lines,
        )

    def run_bridge(img):
        return spurious_bridge(
            img, membrane_idx, ndim=2, prob=1.0,
            length_range=args.bridge_length_range, thickness_range=args.bridge_thickness_range,
            n_lines=args.bridge_n_lines,
        )

    def run_artifact(img, band_prob):
        return artifact_corruption(
            img, membrane_idx, ndim=2, prob=1.0,
            band_prob=band_prob,
            band_thickness_range=args.artifact_band_thickness_range,
            blob_size_range=args.artifact_blob_size_range, blob_n_range=args.artifact_blob_n_range,
        )

    def run_skeleton_perturb(img):
        return skeleton_perturbation(
            img, membrane_idx, ndim=2, prob=1.0,
            radius_range=args.skeleton_radius_range, n_spurs=args.skeleton_n_spurs,
            spur_length_range=args.skeleton_spur_length_range,
        )

    def run_slice_dropout(img):
        return slice_dropout(img, (membrane_idx,), prob=1.0, ndim=2)

    save_tif(np.expand_dims(base, 0), args.out_dir, ["original.tif"], verbose=False)

    # name -> (reference channel to diff/display against, list of per-draw channel slices)
    results = {}
    for name, fn in [("gap_aug", run_gap), ("bridge_aug", run_bridge)]:
        draws = []
        for i in range(args.n_examples):
            out = fn(base)
            draws.append(out[..., membrane_idx])
            save_tif(np.expand_dims(out, 0), args.out_dir, [f"{name}_{i}.tif"], verbose=False)
        results[name] = (membrane, draws)

    # Alternate band/blobs across draws so both artifact types are always depicted.
    artifact_draws, artifact_raw_draws = [], []
    for i in range(args.n_examples):
        out = run_artifact(base, band_prob=1.0 if i % 2 == 0 else 0.0)
        artifact_draws.append(out[..., membrane_idx])
        if raw is not None:
            artifact_raw_draws.append(out[..., raw_idx])
        save_tif(np.expand_dims(out, 0), args.out_dir, [f"artifact_aug_{i}.tif"], verbose=False)
    results["artifact_aug"] = (membrane, artifact_draws)
    if raw is not None:
        results["artifact_aug (raw)"] = (raw, artifact_raw_draws)

    for name, fn in [("skeleton_perturb_aug", run_skeleton_perturb), ("slice_dropout_aug", run_slice_dropout)]:
        draws = []
        for i in range(args.n_examples):
            out = fn(base)
            draws.append(out[..., membrane_idx])
            save_tif(np.expand_dims(out, 0), args.out_dir, [f"{name}_{i}.tif"], verbose=False)
        results[name] = (membrane, draws)

    n_rows = len(results)
    n_cols = 1 + 2 * args.n_examples
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.2 * n_rows), squeeze=False)
    for row, (name, (ref, draws)) in enumerate(results.items()):
        axes[row][0].imshow(ref, cmap="gray", vmin=0, vmax=1)
        axes[row][0].set_ylabel(name, fontsize=9)
        axes[row][0].set_xticks([])
        axes[row][0].set_yticks([])
        if row == 0:
            axes[row][0].set_title("original")
        for i, draw in enumerate(draws):
            draw_col, diff_col = 1 + 2 * i, 2 + 2 * i
            axes[row][draw_col].imshow(draw, cmap="gray", vmin=0, vmax=1)
            axes[row][draw_col].set_xticks([])
            axes[row][draw_col].set_yticks([])
            axes[row][diff_col].imshow(diff_rgb(ref, draw))
            axes[row][diff_col].set_xticks([])
            axes[row][diff_col].set_yticks([])
            if row == 0:
                axes[row][draw_col].set_title(f"draw {i + 1}")
                axes[row][diff_col].set_title(f"draw {i + 1} diff")
    fig.tight_layout()
    overview_path = os.path.join(args.out_dir, "overview.svg")
    fig.savefig(overview_path, bbox_inches="tight")
    print(f"Saved per-augmentor TIFFs and overview grid to {args.out_dir} ({overview_path})")


if __name__ == "__main__":
    main()
