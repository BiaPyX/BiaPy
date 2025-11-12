#!/usr/bin/env python3
import argparse
import importlib
import os
from pathlib import Path
from typing import Callable, Iterable, List, Tuple
from tqdm import tqdm

import numpy as np
import pandas as pd

from biapy.data.data_manipulation import read_img_as_ndarray

def find_label_dirs(root: Path) -> Iterable[Path]:
    """Yield all directories named exactly 'label' under root."""
    for dirpath, dirnames, filenames in os.walk(root):
        # iterate a copy to avoid modifying traversal
        for d in list(dirnames):
            if d == "label":
                yield Path(dirpath) / d


def iter_label_images(label_dir: Path,
                      extensions: Tuple[str, ...]) -> Iterable[Path]:
    """Yield image paths inside a label directory filtered by extension."""
    for name in sorted(os.listdir(label_dir)):
        p = label_dir / name
        if p.is_file() and p.suffix.lower() in extensions:
            yield p


def instances_from_label(label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (instance_ids, sizes) excluding background 0.
    Uses np.unique for safety with sparse/non-contiguous IDs.
    """
    # squeeze singleton dims (e.g., HxWx1 or 1xHxW from some readers)
    lbl = np.asarray(label)
    lbl = np.squeeze(lbl)

    # ensure integer-ish dtype
    if not np.issubdtype(lbl.dtype, np.integer):
        # if it's float but integer-valued, this will be safe; otherwise it's a data issue
        lbl = lbl.astype(np.int64)

    ids, counts = np.unique(lbl, return_counts=True)
    # drop background id 0 (assumed)
    keep = ids != 0
    return ids[keep], counts[keep]


def main():
    parser = argparse.ArgumentParser(
        description="Collect instance sizes from label images across datasets."
    )
    parser.add_argument("--input-dir", required=True, type=Path,
                        help="Root directory containing datasets (e.g., /datasets)")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Directory where the CSV will be saved")
    parser.add_argument("--extensions", type=str, default=".tif,.tiff,.png",
                        help="Comma-separated list of label image extensions to include "
                             "(default: .tif,.tiff,.png)")
    parser.add_argument("--csv-name", type=str, default="instance_sizes.csv",
                        help="CSV filename to write (default: instance_sizes.csv)")
    args = parser.parse_args()

    root: Path = args.input_dir
    outdir: Path = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / args.csv_name
    extensions = tuple(e.strip().lower() for e in args.extensions.split(",") if e.strip())

    rows: List[dict] = []

    # Walk and process
    for label_dir in tqdm(find_label_dirs(root)):
        print("Processing label dir:", label_dir)
        for img_path in tqdm(iter_label_images(label_dir, extensions)):
            is_3d = any([True for x in ["cartocell", "snemi3d", "mitoem"] if x in str(label_dir).lower()])
            label_arr = read_img_as_ndarray(str(img_path), is_3d=is_3d).squeeze()
            inst_ids, sizes = instances_from_label(label_arr)
            # append rows
            for iid, sz in zip(inst_ids, sizes):
                rows.append({
                    "image_path": str(img_path),
                    "instance_id": int(iid),
                    "size": int(sz),
                })

    # Build and save CSV
    if rows:
        df = pd.DataFrame(rows, columns=["image_path", "instance_id", "size"])
    else:
        # still write an empty CSV with headers for pipeline stability
        df = pd.DataFrame(columns=["image_path", "instance_id", "size"])

    df.to_csv(csv_path, index=False)
    print(f"✅ Wrote {len(df)} rows to {csv_path}")

    ######################

    # Read the CSV
    df = pd.read_csv(csv_path)

    # Extract dataset and split from the path
    df['dataset'] = df['image_path'].str.extract(r'instance_seg_paper/([^/]+)/')[0]
    df['split'] = df['image_path'].str.extract(r'instance_seg_paper/[^/]+/([^/]+)/')[0]

    # Filter only train split
    df_train = df[df['split'].str.contains('train', case=False, na=False)]

    # Compute per-dataset percentile thresholds
    filtered_rows = []
    for dataset, subset in df_train.groupby('dataset'):
        low = subset['size'].quantile(0.005)   # 0.5 percentile
        high = subset['size'].quantile(0.998)  # 99.8 percentile
        filtered = subset[(subset['size'] >= low) & (subset['size'] <= high)]
        filtered_rows.append(filtered)

    # Concatenate filtered subsets
    df_filtered = pd.concat(filtered_rows, ignore_index=True)

    # Compute min and max per dataset after filtering
    stats = (
        df_filtered.groupby('dataset')['size']
        .agg(['min', 'max'])
        .reset_index()
        .sort_values('dataset')
    )
    print(stats)
    # TRAIN:
    # dataset   min      max
    # 0  CartoCell   849     4508 -> +20% = 5400
    # 1   LIVECell    21     4295 -> +20% = 5200
    # 2     Lizard     3      435 -> +20% = 550
    # 3     MitoEM  1638   551926 -> not filter max
    # 4   Omnipose    27     6118 -> +20% = 7400
    # 5    SNEMI3D    24  2657691 -> not filter max

    # TEST:
    #      dataset  min      max
    # 0  CartoCell  841     5157
    # 1   LIVECell   11     5207
    # 2     Lizard   3      442
    # 3     MitoEM  220   408336
    # 4   Omnipose   32     8837
    # 5    SNEMI3D   27  1461877

if __name__ == "__main__":
    main()
