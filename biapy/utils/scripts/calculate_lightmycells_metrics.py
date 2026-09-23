"""
Compute MAE/MSE/SSIM/PSNR/PCC for LightMyCells predictions vs. GT. GT is laid out as BiaPy's
MULTIPLE_RAW_ONE_TARGET_LOADER test sets: one GT tif per Study folder, e.g.
<gt_dir>/<Study_id>_<Organelle>/<Study_id>_<Organelle>.ome.tiff

pred_dir can be flat (BiaPy's own per_image output, one file per raw z-slice named like the raw
input; Study/GT pairing recovered via --organelle) or nested (one Study_id subfolder per
prediction, same layout as gt_dir).

Reports two scales: plain mae/mse/ssim/psnr/pcc (matches test_results_metrics.csv columns, with
PSNR/SSIM given a fixed data_range from the GT dtype instead of an inferred one - see the
image_to_image.py fix), and *_norm (each image percentile-clipped 2/99.8 and rescaled to [0,1],
data_range=1.0).

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

STUDY_PREFIX_RE = re.compile(r"^(Study_\d+_[A-Za-z0-9]+_image_\d+)_")


def find_gt_file(gt_study_dir: str) -> str:
    files = [f for f in os.listdir(gt_study_dir) if f.endswith((".tif", ".tiff"))]
    assert len(files) == 1, f"Expected exactly one GT file in {gt_study_dir}, found {files}"
    return os.path.join(gt_study_dir, files[0])


def to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)


def fixed_range_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    data_range = float(np.iinfo(gt.dtype).max) if np.issubdtype(gt.dtype, np.integer) else None
    pred_t, gt_t = to_tensor(pred), to_tensor(gt)
    mae = MeanAbsoluteError()(pred_t, gt_t).item()
    mse = MeanSquaredError()(pred_t, gt_t).item()
    ssim = structural_similarity_index_measure(pred_t, gt_t, data_range=data_range).item()
    psnr = peak_signal_noise_ratio(pred_t, gt_t, data_range=data_range).item()
    pcc_val = PearsonCorrCoef()(pred_t.flatten(), gt_t.flatten()).item()
    return {"mae": mae, "mse": mse, "ssim": ssim, "psnr": psnr, "pcc": pcc_val}


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
    return {"mae_norm": mae, "mse_norm": mse, "ssim_norm": ssim, "psnr_norm": psnr}


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
        for pred_fname in entries:
            if not pred_fname.endswith((".tif", ".tiff")):
                continue
            m = STUDY_PREFIX_RE.match(pred_fname)
            assert m is not None, f"Could not parse a Study_id prefix out of '{pred_fname}'"
            gt_study_dir = os.path.join(gt_dir, f"{m.group(1)}_{organelle}")
            if not os.path.isdir(gt_study_dir):
                print(f"WARNING: no GT dir {gt_study_dir} for prediction {pred_fname}, skipping")
                continue
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

        row = {"file": pred_fname}
        row.update(fixed_range_metrics(pred, gt))
        row.update(norm_scale_metrics(pred, gt))
        rows.append(row)

    cols = ["file", "mae", "mse", "ssim", "psnr", "pcc", "mae_norm", "mse_norm", "ssim_norm", "psnr_norm"]
    os.makedirs(os.path.dirname(args["output_csv"]) or ".", exist_ok=True)
    with open(args["output_csv"], "w") as f:
        f.write(",".join(cols) + "\n")
        for row in rows:
            f.write(",".join(str(row[c]) for c in cols) + "\n")

    print(f"\nWrote {len(rows)} rows to {args['output_csv']}\n")
    print("#############\n#  RESULTS  #\n#############")
    for m in ["mae", "mse", "ssim", "psnr", "pcc", "mae_norm", "mse_norm", "ssim_norm", "psnr_norm"]:
        vals = [row[m] for row in rows]
        print(f"Mean {m}: {np.mean(vals)}")


if __name__ == "__main__":
    main()
