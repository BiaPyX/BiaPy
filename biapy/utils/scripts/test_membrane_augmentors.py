"""
Visualize the membrane-repair corruption augmentors (biapy/data/generators/membrane_augmentors.py)
on a single input membrane image, so their effect/parameters can be sanity-checked before wiring
them into a training run.

For each augmentor, ``--n-examples`` independent draws are applied (forcing ``prob=1.0`` so the
effect is always visible) and saved next to the original as individual TIFFs, plus one combined
overview grid (SVG and PNG; rows = augmentor, columns = original, draw 1, draw 1 diff, draw 2,
draw 2 diff, ...) for a quick look. In each diff panel only the altered pixels are shown: pixels
added by the augmentor in red, pixels it removed in blue.

``artifact_aug`` draws alternate band/blobs so both are always shown. Pass ``--raw-image`` to also
see its effect on a second "raw" channel (extra ``artifact_aug (raw)`` row).

Pass ``--labels-image`` (instance labels) and ``--raw-image`` to also show ``internal_membrane_aug`` (fake mito
outlines tracing ridge rings) and ``ridge_membrane_aug``, both built from the ridge response of the raw image. Both are 3D: they run on the whole (cropped) volume. Each draw shows the
z-slice with the most added membrane, and its diff is drawn over the raw image when given. An extra
``continuity.png`` shows ``--continuity-slices`` consecutive slices around it, to check how the fake membranes
persist along z, and ``internal_membrane_steps.png`` shows, for ``--steps-slices`` slices of the first
``internal_membrane_aug`` draw, the raw image, its ridge response, the thresholded ridge (with the detected objects)
and the painted fake. Their draws are saved as ImageJ TIFFs (Z, C, Y, X) with channels membrane, raw, added membrane.

Membrane and raw can come from the same multi-channel file with ``--membrane-channel``/``--raw-channel``.

Example
-------
python test_membrane_augmentors.py --membrane-image /path/to/membrane.tif \
    --raw-image /path/to/raw.tif --out-dir /path/to/out

With a 2-channel (membrane, raw) volume, its instance labels and a crop:
python test_membrane_augmentors.py --membrane-image mem_raw.tif --membrane-channel 0 --ignore-value 255 \
    --raw-image mem_raw.tif --raw-channel 1 --labels-image instances.tif \
    --crop 300 364 400 656 400 656 --out-dir /path/to/out
"""
import argparse
import os
import random
import sys

import numpy as np
import matplotlib.pyplot as plt
import tifffile
from scipy.ndimage import binary_erosion, find_objects as ndi_find_objects, label as ndi_label

_DEFAULT_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize the membrane-repair corruption augmentors on a single input image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--code-dir", default=_DEFAULT_CODE_DIR, help="Path to the BiaPy repo root.")
    parser.add_argument(
        "--membrane-image", required=True,
        help="Path to the membrane image. 2D (y, x) or 3D (z, y, x), or multi-channel with --membrane-channel.",
    )
    parser.add_argument(
        "--membrane-channel", type=int, default=None,
        help="Channel of --membrane-image to use if it is multi-channel ((c, y, x) or (z, c, y, x)).",
    )
    parser.add_argument(
        "--ignore-value", type=float, default=None,
        help="Ignore label in the membrane (e.g. 255): set to 0, as the training generator does.",
    )
    parser.add_argument(
        "--raw-image", default=None,
        help="Optional path to a second 'raw' channel (same shape as --membrane-image), to also "
        "show artifact_aug's effect on it (as an extra 'artifact_aug (raw)' row). Needed for ridge_membrane_aug.",
    )
    parser.add_argument("--raw-channel", type=int, default=None, help="Channel of --raw-image if multi-channel.")
    parser.add_argument(
        "--labels-image", default=None,
        help="Optional instance labels (same shape as the membrane) to show internal/ridge_membrane_aug.",
    )
    parser.add_argument(
        "--crop", type=int, nargs=6, default=None, metavar=("Z0", "Z1", "Y0", "Y1", "X0", "X1"),
        help="Crop 3D inputs to [Z0:Z1, Y0:Y1, X0:X1] before anything else.",
    )
    parser.add_argument(
        "--z-slice", type=int, default=None,
        help="Which z-slice (within the crop) to test the 2D augmentors on if the input is 3D. Defaults to the "
        "middle slice.",
    )
    parser.add_argument("--out-dir", required=True, help="Directory to save the per-draw TIFFs and overview.")
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

    parser.add_argument("--internal-n-range", type=int, nargs=2, default=(20, 40), metavar=("MIN", "MAX"))
    parser.add_argument("--internal-thresholds", type=float, nargs="+", default=(0.5, 0.55, 0.6, 0.65, 0.7))
    parser.add_argument("--internal-gap", type=int, default=10)
    parser.add_argument("--internal-radius-range", type=float, nargs=2, default=(4, 50), metavar=("MIN", "MAX"))
    parser.add_argument("--internal-darker", type=float, default=0.05)
    parser.add_argument("--internal-z-range", type=int, nargs=2, default=(3, 8), metavar=("MIN", "MAX"))
    parser.add_argument("--internal-thickness-range", type=int, nargs=2, default=(2, 4), metavar=("MIN", "MAX"))
    parser.add_argument("--internal-keep-range", type=float, nargs=2, default=(0.5, 1.0), metavar=("MIN", "MAX"))

    parser.add_argument("--ridge-n-range", type=int, nargs=2, default=(6, 12), metavar=("MIN", "MAX"))
    parser.add_argument("--ridge-box-z-range", type=int, nargs=2, default=(8, 24), metavar=("MIN", "MAX"))
    parser.add_argument("--ridge-box-xy-range", type=int, nargs=2, default=(48, 160), metavar=("MIN", "MAX"))
    parser.add_argument("--ridge-threshold-range", type=float, nargs=2, default=(0.45, 0.6), metavar=("MIN", "MAX"))
    parser.add_argument("--ridge-z-range", type=int, nargs=2, default=(3, 8), metavar=("MIN", "MAX"))
    parser.add_argument("--ridge-thickness-range", type=int, nargs=2, default=(1, 3), metavar=("MIN", "MAX"))
    parser.add_argument("--ridge-z-sigma", type=float, default=1.0)
    parser.add_argument("--ridge-min-area", type=float, default=20)

    parser.add_argument("--dust-n-regions", type=int, nargs=2, default=(2, 6), metavar=("MIN", "MAX"))
    parser.add_argument("--dust-radius-range", type=float, nargs=2, default=(15, 50), metavar=("MIN", "MAX"))
    parser.add_argument("--dust-density-range", type=float, nargs=2, default=(0.3, 0.8), metavar=("MIN", "MAX"))
    parser.add_argument("--dust-decay-range", type=float, nargs=2, default=(2, 8), metavar=("MIN", "MAX"))
    parser.add_argument("--dust-cloud-sigma-range", type=float, nargs=2, default=(1.5, 3.0), metavar=("MIN", "MAX"))
    parser.add_argument("--dust-speck-sigma-range", type=float, nargs=2, default=(0.7, 1.2), metavar=("MIN", "MAX"))
    parser.add_argument("--dust-speck-density", type=float, default=0.3)
    parser.add_argument("--dust-ring-density", type=float, default=0.3)

    parser.add_argument(
        "--continuity-slices", type=int, default=7,
        help="Consecutive z-slices shown per draw of internal/ridge_membrane_aug in continuity.png.",
    )
    parser.add_argument(
        "--steps-slices", type=int, default=4,
        help="Z-slices of the first internal_membrane_aug draw shown in internal_membrane_steps.png.",
    )
    parser.add_argument(
        "--steps-z", type=int, nargs="+", default=None,
        help="Show these z-slices (within the crop) in internal_membrane_steps.png instead of the most painted ones.",
    )

    return parser.parse_args()


def load_volume(imread, path: str, channel: "int | None", crop: "list | None") -> np.ndarray:
    """Load ``path`` as a single-channel ``(h, w)`` or ``(z, h, w)`` array, optionally selecting a channel and cropping."""
    img, _ = imread(path)
    img = np.squeeze(img)
    if channel is not None:
        img = img[channel] if img.ndim == 3 else img[:, channel]
    if img.ndim not in (2, 3):
        raise ValueError(f"Expected a 2D or 3D single-channel image, got shape {img.shape} for {path}")
    if crop is not None and img.ndim == 3:
        z0, z1, y0, y1, x0, x1 = crop
        img = img[z0:z1, y0:y1, x0:x1]
    return img


def to_unit(img: np.ndarray, ignore_value: "float | None" = None) -> np.ndarray:
    """Float32 copy normalized to ``[0, 1]``, with ``ignore_value`` pixels set to 0 first."""
    img = img.astype(np.float32)
    if ignore_value is not None:
        img[img == ignore_value] = 0
    if img.max() > 1:
        img = img / img.max()
    return img


def diff_rgb(orig: np.ndarray, draw: np.ndarray, threshold: float = 0.5, background: "np.ndarray | None" = None) -> np.ndarray:
    """
    ``(h, w, 3)`` RGB image showing only the pixels ``draw`` altered relative to ``orig``
    (both thresholded to binary): added pixels in red, removed pixels in blue, everything else
    black (or ``background`` in gray, if given).
    """
    orig_b = orig > threshold
    draw_b = draw > threshold
    if background is None:
        rgb = np.zeros((*orig.shape, 3), dtype=np.float32)
    else:
        rgb = np.repeat(background[..., None].astype(np.float32), 3, axis=-1)
    rgb[draw_b & ~orig_b] = (1.0, 0.0, 0.0)
    rgb[orig_b & ~draw_b] = (0.0, 0.0, 1.0)
    return rgb


def main() -> None:
    args = parse_args()

    sys.path.insert(0, args.code_dir)
    from biapy.data.data_manipulation import imread, save_tif
    from biapy.data.generators.membrane_augmentors import (
        artifact_corruption,
        _ridge_enclosed_objects,
        internal_membranes,
        membrane_dust,
        ridge_membrane_response,
        ridge_membranes,
        skeleton_perturbation,
        slice_dropout,
        spurious_bridge,
        synthetic_gap,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)
    random.seed(args.seed)

    mem_vol = to_unit(load_volume(imread, args.membrane_image, args.membrane_channel, args.crop), args.ignore_value)
    raw_vol = to_unit(load_volume(imread, args.raw_image, args.raw_channel, args.crop)) if args.raw_image else None
    labels_vol = load_volume(imread, args.labels_image, None, args.crop) if args.labels_image else None
    is_3d = mem_vol.ndim == 3
    z = (mem_vol.shape[0] // 2 if args.z_slice is None else args.z_slice) if is_3d else None

    membrane = mem_vol[z] if is_3d else mem_vol
    raw = None if raw_vol is None else (raw_vol[z] if is_3d else raw_vol)
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

    def run_dust(img):
        return membrane_dust(
            img, membrane_idx, ndim=2, prob=1.0,
            n_regions=args.dust_n_regions, radius_range=args.dust_radius_range,
            density_range=args.dust_density_range, decay_range=args.dust_decay_range,
            cloud_sigma_range=args.dust_cloud_sigma_range, speck_sigma_range=args.dust_speck_sigma_range,
            speck_density=args.dust_speck_density, ring_density=args.dust_ring_density,
        )

    def run_slice_dropout(img):
        return slice_dropout(img, (membrane_idx,), prob=1.0, ndim=2)

    def run_internal(img):
        return internal_membranes(
            img, membrane_idx, raw_idx, labels_vol, ndim=3 if is_3d else 2, prob=1.0,
            n_range=args.internal_n_range, thresholds=tuple(args.internal_thresholds), gap=args.internal_gap,
            radius_range=args.internal_radius_range, darker=args.internal_darker, z_range=args.internal_z_range,
            thickness_range=args.internal_thickness_range, keep_range=args.internal_keep_range,
        )

    def run_ridge(img):
        return ridge_membranes(
            img, membrane_idx, raw_idx, labels_vol, ndim=3 if is_3d else 2, prob=1.0,
            n_range=args.ridge_n_range, box_z_range=args.ridge_box_z_range, box_xy_range=args.ridge_box_xy_range,
            threshold_range=args.ridge_threshold_range, z_range=args.ridge_z_range,
            thickness_range=args.ridge_thickness_range, z_sigma=args.ridge_z_sigma,
            min_area=args.ridge_min_area,
        )

    save_tif(np.expand_dims(base, 0), args.out_dir, ["original.tif"], verbose=False)

    # name -> (reference image for the first column, list of (draw, diff reference, diff background, title))
    results = {}

    def plain_draws(fn, name, ref=membrane, channel=membrane_idx, outs=None):
        draws = []
        for i in range(args.n_examples):
            out = fn(base) if outs is None else outs[i]
            draws.append((out[..., channel], ref, None, f"draw {i + 1}"))
            if outs is None:
                save_tif(np.expand_dims(out, 0), args.out_dir, [f"{name}_{i}.tif"], verbose=False)
        results[name] = (ref, draws)

    for name, fn in [("gap_aug", run_gap), ("bridge_aug", run_bridge)]:
        plain_draws(fn, name)

    # Alternate band/blobs across draws so both artifact types are always depicted.
    artifact_outs = []
    for i in range(args.n_examples):
        out = run_artifact(base, band_prob=1.0 if i % 2 == 0 else 0.0)
        artifact_outs.append(out)
        save_tif(np.expand_dims(out, 0), args.out_dir, [f"artifact_aug_{i}.tif"], verbose=False)
    plain_draws(None, "artifact_aug", outs=artifact_outs)
    if raw is not None:
        plain_draws(None, "artifact_aug (raw)", ref=raw, channel=raw_idx, outs=artifact_outs)

    for name, fn in [
        ("skeleton_perturb_aug", run_skeleton_perturb),
        ("slice_dropout_aug", run_slice_dropout),
        ("membrane_dust_aug", run_dust),
    ]:
        plain_draws(fn, name)

    # name -> list of (added mask, shown z-slice) per draw, for continuity.png
    continuity = {}
    if labels_vol is not None:
        base_vol = np.stack([mem_vol, raw_vol], axis=-1) if raw_vol is not None else mem_vol[..., None]
        # Both are anchored on the ridge response of the raw image
        augs = [("internal_membrane_aug", run_internal), ("ridge_membrane_aug", run_ridge)] if raw_vol is not None else []
        for name, fn in augs:
            draws, continuity[name] = [], []
            for i in range(args.n_examples):
                out = fn(base_vol)
                added = (out[..., membrane_idx] > 0.5) & (base_vol[..., membrane_idx] <= 0.5)
                zi = int(np.argmax(added.sum(axis=(1, 2)))) if is_3d else None
                sl = (lambda a: a[zi]) if is_3d else (lambda a: a)
                background = None if raw_vol is None else sl(raw_vol)
                title = f"draw {i + 1}" + (f" (z={zi})" if is_3d else "")
                draws.append((sl(out[..., membrane_idx]), sl(mem_vol), background, title))
                continuity[name].append((added, zi))
                vol = np.concatenate([out, added[..., None].astype(out.dtype)], axis=-1)
                if is_3d:
                    vol = np.moveaxis(vol, -1, 1)  # (z, c, y, x)
                    tifffile.imwrite(
                        os.path.join(args.out_dir, f"{name}_{i}.tif"),
                        (vol * 255).astype(np.uint8), imagej=True, metadata={"axes": "ZCYX"},
                    )
                else:
                    save_tif(np.expand_dims(vol, 0), args.out_dir, [f"{name}_{i}.tif"], verbose=False)
                if is_3d:
                    comps, _ = ndi_label(added, structure=np.ones((3, 3, 3)))
                    zext = sorted(sl[0].stop - sl[0].start for sl in ndi_find_objects(comps))
                    print(f"{name} draw {i + 1}: {int(added.sum())} membrane px added "
                          f"({added.sum() / max(1, (labels_vol > 0).sum()):.2%} of neuron voxels), {len(zext)} 3D pieces "
                          f"spanning {zext} slices, shown z-slice {zi}")
            results[name] = (draws[0][1], draws)

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
        for i, (draw, diff_ref, background, title) in enumerate(draws):
            draw_col, diff_col = 1 + 2 * i, 2 + 2 * i
            axes[row][draw_col].imshow(draw, cmap="gray", vmin=0, vmax=1)
            axes[row][draw_col].set_xticks([])
            axes[row][draw_col].set_yticks([])
            axes[row][diff_col].imshow(diff_rgb(diff_ref, draw, background=background))
            axes[row][diff_col].set_xticks([])
            axes[row][diff_col].set_yticks([])
            if row == 0 or title != f"draw {i + 1}":
                axes[row][draw_col].set_title(title, fontsize=8)
                axes[row][diff_col].set_title(f"{title.split(' (')[0]} diff", fontsize=8)
    fig.tight_layout()
    overview_path = os.path.join(args.out_dir, "overview.svg")
    fig.savefig(overview_path, bbox_inches="tight")
    fig.savefig(os.path.join(args.out_dir, "overview.png"), bbox_inches="tight", dpi=150)

    if continuity and is_3d:
        half = args.continuity_slices // 2
        rows = [(name, i, added, zi) for name, items in continuity.items() for i, (added, zi) in enumerate(items)]
        fig, axes = plt.subplots(
            len(rows), args.continuity_slices, figsize=(2.2 * args.continuity_slices, 2.2 * len(rows)), squeeze=False
        )
        for row, (name, i, added, zi) in enumerate(rows):
            z0 = int(np.clip(zi - half, 0, max(0, mem_vol.shape[0] - args.continuity_slices)))
            for col in range(args.continuity_slices):
                ax = axes[row][col]
                ax.set_xticks([])
                ax.set_yticks([])
                z = z0 + col
                if z >= mem_vol.shape[0]:
                    ax.axis("off")
                    continue
                background = raw_vol[z] if raw_vol is not None else mem_vol[z]
                rgb = np.repeat(background[..., None].astype(np.float32), 3, axis=-1)
                rgb[added[z]] = (1.0, 0.0, 0.0)
                ax.imshow(rgb)
                ax.set_title(f"z={z}", fontsize=8)
                if col == 0:
                    ax.set_ylabel(f"{name}\ndraw {i + 1}", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "continuity.png"), bbox_inches="tight", dpi=150)

    if continuity.get("internal_membrane_aug") and is_3d:
        added = continuity["internal_membrane_aug"][0][0]
        thr = min(args.internal_thresholds)
        response = ridge_membrane_response(raw_vol)
        objects = _ridge_enclosed_objects(
            response, raw_vol, labels_vol, tuple(args.internal_thresholds), args.internal_gap,
            tuple(args.internal_radius_range), args.internal_darker,
        )
        counts = added.sum(axis=(1, 2))
        if args.steps_z:
            zs = list(args.steps_z)
        else:
            zs = sorted(z for z in np.argsort(counts)[-args.steps_slices:] if counts[z] > 0)
        titles = ["raw", "ridge response", f"ridge > {thr} (cyan: detected objects)", "raw + painted fake (red)"]
        fig, axes = plt.subplots(len(zs), 4, figsize=(4 * 4, 4 * len(zs)), squeeze=False)
        for row, z in enumerate(zs):
            gray = lambda a: np.repeat(a[..., None].astype(np.float32), 3, axis=-1)
            hot = gray(response[z] > thr)
            hot[objects[z] & ~binary_erosion(objects[z])] = (0.0, 1.0, 1.0)
            painted = gray(raw_vol[z])
            painted[added[z]] = (1.0, 0.0, 0.0)
            for col, img in enumerate([gray(raw_vol[z]), gray(response[z]), hot, painted]):
                axes[row][col].imshow(img)
                axes[row][col].set_xticks([])
                axes[row][col].set_yticks([])
                if row == 0:
                    axes[row][col].set_title(titles[col], fontsize=10)
            axes[row][0].set_ylabel(f"z={z}", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "internal_membrane_steps.png"), bbox_inches="tight", dpi=120)
    print(f"Saved per-augmentor TIFFs and overview grid to {args.out_dir} ({overview_path})")


if __name__ == "__main__":
    main()
