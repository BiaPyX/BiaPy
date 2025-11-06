#!/usr/bin/env python3
"""
Convert foreground probabilities into instance labels using
peak_local_max + marker-controlled watershed.

Usage:
    python probs_to_instances.py /path/to/input_probs /path/to/output_labels
"""

from pathlib import Path
import argparse
import numpy as np

from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.filters import threshold_otsu
from tqdm import tqdm

from biapy.data.data_manipulation import read_img_as_ndarray, save_tif



# --- Tunables (kept in-code per your request to only have two CLI args) ---
MIN_DISTANCE = 15  # minimum separation (in pixels) between peaks
PEAK_THRESH_ABS = 0.5  # set to a float (e.g., 0.2) to require peaks >= that prob; None disables
USE_OTSU_FOR_MASK = True  # True: mask = probs > otsu; False: mask = probs > 0
USE_EDT = True  # If True, use distance transform for watershed; else use inverted probs
# -------------------------------------------------------------------------


def probs_to_instances(probs: np.ndarray) -> np.ndarray:
    """
    Given a 2D foreground probability map (values in [0, 1]),
    return a 2D integer-labeled instance map.

    Steps:
      1) Create a foreground mask (Otsu by default).
      2) Detect local maxima on the probability surface.
      3) Use those maxima as markers for watershed on the inverted probs.
    """
    if probs.ndim != 2:
        raise ValueError(f"Expected a 2D array; got shape {probs.shape}")

    # Ensure float (no scaling; assume caller provided probabilities already)
    p = probs.astype(np.float32, copy=False)

    # 1) Foreground mask
    if USE_OTSU_FOR_MASK:
        try:
            t = threshold_otsu(p)
            mask = p > t
        except Exception:
            # Fallback if Otsu is ill-defined (e.g., nearly constant image)
            mask = p > 0
    else:
        mask = p > 0

    if not np.any(mask):
        return np.zeros_like(p, dtype=np.uint16)

    if USE_EDT:
        from scipy.ndimage import distance_transform_edt

        # Use distance transform as the topography for watershed
        p = distance_transform_edt(mask).astype(np.float32, copy=False)

    # 2) Local maxima (on probabilities directly)
    #    Build labeled markers array with unique ids per-peak.
    coordinates = peak_local_max(
        p,
        min_distance=MIN_DISTANCE,
        threshold_abs=PEAK_THRESH_ABS if PEAK_THRESH_ABS is not None else None,
        labels=mask,  # restrict peaks to foreground
        exclude_border=False,
    )

    if coordinates.size == 0:
        # No peaks -> nothing to grow; return empty labels
        return np.zeros_like(p, dtype=np.uint16)

    markers = np.zeros_like(p, dtype=np.uint16)
    # Assign a unique marker id to each peak coordinate
    for i, (r, c) in enumerate(coordinates, start=1):
        markers[r, c] = i

    # 3) Marker-controlled watershed on the inverted probability map
    #    Higher probability should be "lower" in the topography -> negate.
    labels = watershed(-p, markers=markers, mask=mask)

    return labels.astype(np.uint16, copy=False)


def main():
    parser = argparse.ArgumentParser(
        description="Convert foreground probability maps into instance labels using peak_local_max + watershed."
    )
    parser.add_argument("input_dir", type=Path, help="Folder with foreground probability images")
    parser.add_argument("output_dir", type=Path, help="Folder to write instance label images")

    args = parser.parse_args()
    in_dir: Path = args.input_dir
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Process every file in the input directory (non-recursive).
    # Your read_image() decides what extensions are supported.
    files = sorted([p for p in in_dir.iterdir() if p.is_file()])

    if not files:
        print(f"[INFO] No files found in {in_dir}")
        return

    for fp in tqdm(files):
        probs = read_img_as_ndarray(str(fp)).squeeze()

        try:
            labels = probs_to_instances(probs)
        except Exception as e:
            print(f"[WARN] Skipping '{fp.name}': processing error ({e})")
            continue

        save_tif(np.expand_dims(np.expand_dims(labels, axis=-1),0), out_dir, [fp.stem + "_labels.tif" ], verbose=False)

    print("[DONE]")


if __name__ == "__main__":
    main()
