"""
Compute MAE/MSE/SSIM/PSNR/PCC for LightMyCells predictions vs. GT. GT is laid out as BiaPy's
MULTIPLE_RAW_ONE_TARGET_LOADER test sets: one GT tif per Study folder, e.g.
<gt_dir>/<Study_id>_<Organelle>/<Study_id>_<Organelle>.ome.tiff

pred_dir can be flat (BiaPy's own per_image output, one file per raw z-slice named like the raw
input; Study/GT pairing recovered via --organelle) or nested (one Study_id subfolder per
prediction, same layout as gt_dir).

All metrics are computed after percentile-clipping (2/99.8) each prediction and GT and rescaling
them to [0,1] (data_range=1.0), so predictions saved in any intensity scale are comparable. The
summary reports the mean per image (z-slice), per sample (image_id) and per study.

Example:
    python calculate_lightmycells_metrics.py \
        --pred_dir .../lightmycellsv2_bestfocus_nucleus_1/per_image \
        --gt_dir /data5/dfranco/datasets/LightMyCells2/nucleus/val/label \
        --organelle Nucleus \
        --output_csv .../test_results_metrics_recomputed.csv
"""
import argparse
import os
import re

import cv2
import numpy as np
import torch
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, PearsonCorrCoef
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
from tqdm import tqdm

from biapy.data.data_manipulation import read_img_as_ndarray
from biapy.data.norm import norm_range01, percentile_clip

parser = argparse.ArgumentParser(
    description="Calculate I2I metrics (MAE/MSE/SSIM/PSNR/PCC) for LightMyCells-style predictions",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("-pred_dir", "--pred_dir", required=True, help="Dir of predicted tifs (flat or Study_id subfolders)")
parser.add_argument("-gt_dir", "--gt_dir", required=True, help="Dir of Study_id subfolders holding one GT tif each")
parser.add_argument(
    "-organelle",
    "--organelle",
    default=None,
    help="Organelle name (e.g. Nucleus/Actin/Mitochondria/Tubulin), required only for flat pred_dir layouts",
)
parser.add_argument("-output_csv", "--output_csv", required=True, help="Where to write the per-image metrics CSV")
args = vars(parser.parse_args())

# Modality may contain hyphens, e.g. "DIC-PC"
STUDY_PREFIX_RE = re.compile(r"^(Study_\d+_[^_]+_image_\d+)_")


def find_gt_file(gt_study_dir: str) -> str:
    files = [f for f in os.listdir(gt_study_dir) if f.endswith((".tif", ".tiff"))]
    assert len(files) == 1, f"Expected exactly one GT file in {gt_study_dir}, found {files}"
    return os.path.join(gt_study_dir, files[0])


def to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)


def norm_scale_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    pred_c, _, _ = percentile_clip(pred.astype(np.float32), per_lower_bound=2.0, per_upper_bound=99.8)
    pred_n, _, _ = norm_range01(pred_c, div_using_max_and_scale=True, max_val_to_div=None, min_val_to_div=None)
    gt_c, _, _ = percentile_clip(gt.astype(np.float32), per_lower_bound=2.0, per_upper_bound=99.8)
    gt_n, _, _ = norm_range01(gt_c, div_using_max_and_scale=True, max_val_to_div=None, min_val_to_div=None)

    pred_t, gt_t = to_tensor(np.asarray(pred_n)), to_tensor(np.asarray(gt_n))
    mae = MeanAbsoluteError()(pred_t, gt_t).item()
    mse = MeanSquaredError()(pred_t, gt_t).item()
    ssim = structural_similarity_index_measure(pred_t, gt_t, data_range=1.0).item()
    psnr = peak_signal_noise_ratio(pred_t, gt_t, data_range=1.0).item()
    pcc_val = PearsonCorrCoef()(pred_t.flatten(), gt_t.flatten()).item()
    return {"mae": mae, "mse": mse, "ssim": ssim, "psnr": psnr, "pcc": pcc_val}


def collect_pairs(pred_dir: str, gt_dir: str, organelle: str | None):
    """Yields (pred_path, gt_path, pred_filename) triples for every prediction found."""
    entries = sorted(os.listdir(pred_dir))
    study_subdirs = [d for d in entries if os.path.isdir(os.path.join(pred_dir, d))]

    if study_subdirs:
        for study in study_subdirs:
            pred_study_dir = os.path.join(pred_dir, study)
            gt_study_dir = os.path.join(gt_dir, study)
            if not os.path.isdir(gt_study_dir):
                print(f"WARNING: no GT dir for {study}, skipping")
                continue
            gt_path = find_gt_file(gt_study_dir)
            for pred_fname in sorted(f for f in os.listdir(pred_study_dir) if f.endswith((".tif", ".tiff"))):
                yield os.path.join(pred_study_dir, pred_fname), gt_path, pred_fname
    else:
        assert organelle is not None, "--organelle is required when --pred_dir holds a flat list of files"
        # Case-insensitive lookup so e.g. "nucleus" matches "<Study>_Nucleus" GT folders
        gt_dirs_lower = {d.lower(): d for d in os.listdir(gt_dir) if os.path.isdir(os.path.join(gt_dir, d))}
        for pred_fname in entries:
            if not pred_fname.endswith((".tif", ".tiff")):
                continue
            m = STUDY_PREFIX_RE.match(pred_fname)
            assert m is not None, f"Could not parse a Study_id prefix out of '{pred_fname}'"
            gt_study_name = gt_dirs_lower.get(f"{m.group(1)}_{organelle}".lower())
            if gt_study_name is None:
                print(f"WARNING: no GT dir {m.group(1)}_{organelle} in {gt_dir} for prediction {pred_fname}, skipping")
                continue
            gt_study_dir = os.path.join(gt_dir, gt_study_name)
            gt_path = find_gt_file(gt_study_dir)
            yield os.path.join(pred_dir, pred_fname), gt_path, pred_fname


def main():
    pairs = list(collect_pairs(args["pred_dir"], args["gt_dir"], args["organelle"]))
    print(f"Found {len(pairs)} predictions under {args['pred_dir']}")

    gt_cache: dict[str, np.ndarray] = {}
    rows = []
    for pred_path, gt_path, pred_fname in tqdm(pairs, desc="images"):
        if gt_path not in gt_cache:
            gt_cache[gt_path] = np.squeeze(read_img_as_ndarray(gt_path, is_3d=False))
        gt = gt_cache[gt_path]

        pred = np.squeeze(read_img_as_ndarray(pred_path, is_3d=False))
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

        sample = STUDY_PREFIX_RE.match(pred_fname)
        sample = sample.group(1) if sample is not None else os.path.basename(os.path.dirname(pred_path))
        row = {"file": pred_fname, "sample": sample, "study": "_".join(sample.split("_")[:2])}
        row.update(norm_scale_metrics(pred, gt))
        rows.append(row)

    metrics = ["mae", "mse", "ssim", "psnr", "pcc"]
    cols = ["file", "sample", "study"] + metrics
    os.makedirs(os.path.dirname(args["output_csv"]) or ".", exist_ok=True)
    with open(args["output_csv"], "w") as f:
        f.write(",".join(cols) + "\n")
        for row in rows:
            f.write(",".join(str(row[c]) for c in cols) + "\n")

    print(f"\nWrote {len(rows)} rows to {args['output_csv']}\n")
    print("#############\n#  RESULTS  #\n#############")
    # Image means are dominated by studies with many samples/z-slices, so also average per sample and per study
    for level in ["file", "sample", "study"]:
        groups: dict[str, list[dict]] = {}
        for row in rows:
            groups.setdefault(row[level], []).append(row)
        print(f"Mean per {level} ({len(groups)} groups):")
        for m in metrics:
            print(f"  {m}: {np.mean([np.mean([r[m] for r in g]) for g in groups.values()])}")


if __name__ == "__main__":
    main()
