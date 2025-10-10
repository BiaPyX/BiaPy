#!/usr/bin/env python3
"""
PR Curve for Instance Segmentation (IoU-based Hungarian matching) — SVG output
MEMORY-EFFICIENT: no (P,G,H,W) tensors; intersections computed via label contingency.

Usage:
    python pr_curve_instance_seg.py /path/to/gt /path/to/preds_root \
        --iou-thresh 0.5 --thresholds 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
        --exts .png .tif .tiff .npy --out pr_curve.svg
"""

import argparse
import os
import sys
from glob import glob
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
import plotly.graph_objects as go
from biapy.data.data_manipulation import read_img_as_ndarray

# ========= YOU WILL IMPLEMENT THIS =========
def load_instance_map(path: str) -> np.ndarray:
    """
    TODO: Implement this to return a 2D integer label map (H, W):
        - 0 = background
        - 1..N = instance IDs (each ID is one instance)
    """
    
    return read_img_as_ndarray(path).squeeze()


# ============ Utilities ============

def parse_args():
    p = argparse.ArgumentParser(description="Compute PR curve over thresholds for instance segmentation.")
    p.add_argument("gt_dir", type=str, help="Path to ground-truth instances directory.")
    p.add_argument("preds_root", type=str, help="Path to predictions root containing *_<thr> subdirs.")
    p.add_argument("--iou-thresh", type=float, default=0.5, help="IoU threshold for TP (default: 0.5).")
    p.add_argument("--thresholds", type=float, nargs="+",
                   default=[round(x * 0.1, 1) for x in range(0, 10)],
                   help="Thresholds to evaluate; looks for subdirs ending with _<thr> (default: 0.0..0.9).")
    p.add_argument("--exts", type=str, nargs="+", default=[".png", ".tif", ".tiff", ".npy"],
                   help="File extensions to consider (default: .png .tif .tiff .npy).")
    p.add_argument("--out", type=str, default="pr_curve.svg", help="Output SVG file for the PR curve.")
    return p.parse_args()


def discover_threshold_dirs(preds_root: str, thresholds: List[float]) -> Dict[float, str]:
    found = {}
    subdirs = [d for d in glob(os.path.join(preds_root, "*")) if os.path.isdir(d)]
    endings = {os.path.basename(d): d for d in subdirs}
    for t in thresholds:
        suffix = f"_{t:.1f}"
        matches = [d for name, d in endings.items() if name.endswith(suffix)]
        if len(matches) == 1:
            found[t] = matches[0]
        elif len(matches) > 1:
            exact = os.path.join(preds_root, suffix)
            found[t] = exact if exact in matches else sorted(matches, key=len)[0]
    return found


def list_gt_files(gt_dir: str, exts: List[str]) -> List[str]:
    files = []
    for ext in exts:
        files.extend(glob(os.path.join(gt_dir, f"*{ext}")))
    return sorted(files)


def relabel_consecutive(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map arbitrary instance IDs to 1..K consecutively (0 stays 0).
    Returns:
        labels_re (H,W) int
        id_map: array of original ids at index new_id (len K+1, id_map[0]==0)
    """
    assert labels.ndim == 2
    # unique positive ids
    ids = np.unique(labels)
    ids = ids[ids > 0]
    if ids.size == 0:
        return labels.astype(np.int32, copy=False), np.array([0], dtype=ids.dtype)
    new_ids = np.arange(1, ids.size + 1, dtype=np.int32)
    # build LUT
    max_id = int(ids.max())
    lut = np.zeros(max_id + 1, dtype=np.int32)
    lut[ids] = new_ids
    labels_re = labels.copy()
    pos = labels_re > 0
    labels_re[pos] = lut[labels_re[pos]]
    id_map = np.zeros(new_ids.size + 1, dtype=ids.dtype)
    id_map[1:] = ids
    return labels_re, id_map


def iou_from_labelmaps(pred: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """
    Memory-efficient IoU table between instances in pred and gt label maps.
    - Remaps labels to consecutive ids.
    - Computes intersections via bincount on paired labels.
    Returns:
        iou (P x G) dense float32 matrix,
        P (#pred instances), G (#gt instances)
    """
    if pred.shape != gt.shape:
        # Return empty table; caller will treat as zero predictions
        return np.zeros((0, 0), dtype=np.float32), 0, 0

    # Relabel to 1..P and 1..G
    pred_re, _ = relabel_consecutive(pred)
    gt_re, _ = relabel_consecutive(gt)

    P = int(pred_re.max())
    G = int(gt_re.max())
    if P == 0 or G == 0:
        return np.zeros((P, G), dtype=np.float32), P, G

    # Areas (pixel counts) per instance (size P+1 / G+1 with 0 ignored)
    area_pred = np.bincount(pred_re.ravel(), minlength=P + 1).astype(np.int64)
    area_gt = np.bincount(gt_re.ravel(), minlength=G + 1).astype(np.int64)

    # Build contingency of (pred_id, gt_id) > 0 pairs using a single bincount
    # Encode pair index = pred_id * (G+1) + gt_id
    pair_index = pred_re.astype(np.int64) * (G + 1) + gt_re.astype(np.int64)
    # We only care where both are > 0
    valid = (pred_re > 0) & (gt_re > 0)
    pair_counts = np.bincount(pair_index[valid], minlength=(P + 1) * (G + 1))

    # Indices with non-zero intersections (skip rows/cols where id==0)
    nz = pair_counts.nonzero()[0]
    if nz.size == 0:
        return np.zeros((P, G), dtype=np.float32), P, G

    # Convert back to (p, g)
    g_ids = nz % (G + 1)
    p_ids = nz // (G + 1)

    # Filter out any where p==0 or g==0 (background)
    mask_fg = (p_ids > 0) & (g_ids > 0)
    p_ids = p_ids[mask_fg]
    g_ids = g_ids[mask_fg]
    inter_vals = pair_counts[nz][mask_fg].astype(np.float64)

    # Prepare IoU matrix (dense, but small: P x G)
    iou = np.zeros((P, G), dtype=np.float32)

    # Compute unions and IoUs only for intersecting pairs
    unions = (area_pred[p_ids] + area_gt[g_ids] - inter_vals).astype(np.float64)
    iou_vals = (inter_vals / np.maximum(unions, 1.0)).astype(np.float32)

    # Fill
    iou[p_ids - 1, g_ids - 1] = iou_vals  # shift to 0-based indices

    return iou, P, G


def match_and_count_from_labelmaps(pred_map: np.ndarray, gt_map: np.ndarray, iou_thresh: float) -> Tuple[int, int, int]:
    """
    Compute TP/FP/FN by:
      - building IoU table via contingency (no 4D arrays),
      - Hungarian matching under IoU >= threshold.
    """
    iou, P, G = iou_from_labelmaps(pred_map, gt_map)
    if P == 0 and G == 0:
        return 0, 0, 0
    if P == 0:
        return 0, 0, G
    if G == 0:
        return 0, P, 0

    # Build cost: disallow below-threshold pairs
    large_cost = 1e6
    cost = 1.0 - iou  # (P, G)
    # For pairs with IoU == 0, cost==1; we must still block those < thresh
    cost = cost.copy()
    cost[iou < iou_thresh] = large_cost

    # Hungarian (works with rectangular)
#!/usr/bin/env python3
"""
PR Curve for Instance Segmentation (IoU-based Hungarian matching) — SVG output
MEMORY-EFFICIENT: no (P,G,H,W) tensors; intersections computed via label contingency.

Usage:
    python pr_curve_instance_seg.py /path/to/gt /path/to/preds_root \
        --iou-thresh 0.5 --thresholds 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
        --exts .png .tif .tiff .npy --out pr_curve.svg
"""

import argparse
import os
import sys
from glob import glob
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
import plotly.graph_objects as go


# ========= YOU WILL IMPLEMENT THIS =========
def load_instance_map(path: str) -> np.ndarray:
    """
    TODO: Implement this to return a 2D integer label map (H, W):
        - 0 = background
        - 1..N = instance IDs (each ID is one instance)
    """
    return read_img_as_ndarray(path).squeeze()


# ============ Utilities ============

def parse_args():
    p = argparse.ArgumentParser(description="Compute PR curve over thresholds for instance segmentation.")
    p.add_argument("gt_dir", type=str, help="Path to ground-truth instances directory.")
    p.add_argument("preds_root", type=str, help="Path to predictions root containing *_<thr> subdirs.")
    p.add_argument("--iou-thresh", type=float, default=0.5, help="IoU threshold for TP (default: 0.5).")
    p.add_argument("--thresholds", type=float, nargs="+",
                   default=[round(x * 0.1, 1) for x in range(0, 10)],
                   help="Thresholds to evaluate; looks for subdirs ending with _<thr> (default: 0.0..0.9).")
    p.add_argument("--exts", type=str, nargs="+", default=[".png", ".tif", ".tiff", ".npy"],
                   help="File extensions to consider (default: .png .tif .tiff .npy).")
    p.add_argument("--out", type=str, default="pr_curve.svg", help="Output SVG file for the PR curve.")
    return p.parse_args()


def discover_threshold_dirs(preds_root: str, thresholds: List[float]) -> Dict[float, str]:
    found = {}
    subdirs = [d for d in glob(os.path.join(preds_root, "*")) if os.path.isdir(d)]
    endings = {os.path.basename(d): d for d in subdirs}
    for t in thresholds:
        suffix = f"_{t:.1f}"
        matches = [d for name, d in endings.items() if name.endswith(suffix)]
        if len(matches) == 1:
            found[t] = matches[0]
        elif len(matches) > 1:
            exact = os.path.join(preds_root, suffix)
            found[t] = exact if exact in matches else sorted(matches, key=len)[0]
    return found


def list_gt_files(gt_dir: str, exts: List[str]) -> List[str]:
    files = []
    for ext in exts:
        files.extend(glob(os.path.join(gt_dir, f"*{ext}")))
    return sorted(files)


def relabel_consecutive(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map arbitrary instance IDs to 1..K consecutively (0 stays 0).
    Returns:
        labels_re (H,W) int
        id_map: array of original ids at index new_id (len K+1, id_map[0]==0)
    """
    assert labels.ndim == 2
    # unique positive ids
    ids = np.unique(labels)
    ids = ids[ids > 0]
    if ids.size == 0:
        return labels.astype(np.int32, copy=False), np.array([0], dtype=ids.dtype)
    new_ids = np.arange(1, ids.size + 1, dtype=np.int32)
    # build LUT
    max_id = int(ids.max())
    lut = np.zeros(max_id + 1, dtype=np.int32)
    lut[ids] = new_ids
    labels_re = labels.copy()
    pos = labels_re > 0
    labels_re[pos] = lut[labels_re[pos]]
    id_map = np.zeros(new_ids.size + 1, dtype=ids.dtype)
    id_map[1:] = ids
    return labels_re, id_map


def iou_from_labelmaps(pred: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """
    Memory-efficient IoU table between instances in pred and gt label maps.
    - Remaps labels to consecutive ids.
    - Computes intersections via bincount on paired labels.
    Returns:
        iou (P x G) dense float32 matrix,
        P (#pred instances), G (#gt instances)
    """
    if pred.shape != gt.shape:
        # Return empty table; caller will treat as zero predictions
        return np.zeros((0, 0), dtype=np.float32), 0, 0

    # Relabel to 1..P and 1..G
    pred_re, _ = relabel_consecutive(pred)
    gt_re, _ = relabel_consecutive(gt)

    P = int(pred_re.max())
    G = int(gt_re.max())
    if P == 0 or G == 0:
        return np.zeros((P, G), dtype=np.float32), P, G

    # Areas (pixel counts) per instance (size P+1 / G+1 with 0 ignored)
    area_pred = np.bincount(pred_re.ravel(), minlength=P + 1).astype(np.int64)
    area_gt = np.bincount(gt_re.ravel(), minlength=G + 1).astype(np.int64)

    # Build contingency of (pred_id, gt_id) > 0 pairs using a single bincount
    # Encode pair index = pred_id * (G+1) + gt_id
    pair_index = pred_re.astype(np.int64) * (G + 1) + gt_re.astype(np.int64)
    # We only care where both are > 0
    valid = (pred_re > 0) & (gt_re > 0)
    pair_counts = np.bincount(pair_index[valid], minlength=(P + 1) * (G + 1))

    # Indices with non-zero intersections (skip rows/cols where id==0)
    nz = pair_counts.nonzero()[0]
    if nz.size == 0:
        return np.zeros((P, G), dtype=np.float32), P, G

    # Convert back to (p, g)
    g_ids = nz % (G + 1)
    p_ids = nz // (G + 1)

    # Filter out any where p==0 or g==0 (background)
    mask_fg = (p_ids > 0) & (g_ids > 0)
    p_ids = p_ids[mask_fg]
    g_ids = g_ids[mask_fg]
    inter_vals = pair_counts[nz][mask_fg].astype(np.float64)

    # Prepare IoU matrix (dense, but small: P x G)
    iou = np.zeros((P, G), dtype=np.float32)

    # Compute unions and IoUs only for intersecting pairs
    unions = (area_pred[p_ids] + area_gt[g_ids] - inter_vals).astype(np.float64)
    iou_vals = (inter_vals / np.maximum(unions, 1.0)).astype(np.float32)

    # Fill
    iou[p_ids - 1, g_ids - 1] = iou_vals  # shift to 0-based indices

    return iou, P, G


def match_and_count_from_labelmaps(pred_map: np.ndarray, gt_map: np.ndarray, iou_thresh: float) -> Tuple[int, int, int]:
    """
    Compute TP/FP/FN by:
      - building IoU table via contingency (no 4D arrays),
      - Hungarian matching under IoU >= threshold.
    """
    iou, P, G = iou_from_labelmaps(pred_map, gt_map)
    if P == 0 and G == 0:
        return 0, 0, 0
    if P == 0:
        return 0, 0, G
    if G == 0:
        return 0, P, 0

    # Build cost: disallow below-threshold pairs
    large_cost = 1e6
    cost = 1.0 - iou  # (P, G)
    # For pairs with IoU == 0, cost==1; we must still block those < thresh
    cost = cost.copy()
    cost[iou < iou_thresh] = large_cost

    # Hungarian (works with rectangular)
    row_ind, col_ind = linear_sum_assignment(cost)

    # Count TPs where matched and IoU >= thresh
    tps = int(np.sum(iou[row_ind, col_ind] >= iou_thresh))
    fps = P - tps
    fns = G - tps
    return tps, fps, fns


def evaluate_threshold_dir(gt_dir: str, pred_dir: str, exts: List[str], iou_thresh: float) -> Tuple[int, int, int, int]:
    gt_files = list_gt_files(gt_dir, exts)
    stem_to_gt = {os.path.splitext(os.path.basename(p))[0]: p for p in gt_files}
    tp_tot = fp_tot = fn_tot = 0
    evaluated = 0

    for stem, gt_path in stem_to_gt.items():
        # Find prediction file with same stem in pred_dir
        pred_path = None
        for ext in exts:
            cand = os.path.join(pred_dir, f"{stem}{ext}")
            if os.path.exists(cand):
                pred_path = cand
                break

        try:
            gt_map = load_instance_map(gt_path)
            if pred_path is None:
                # No preds: all GT instances are FN
                _, _, G = iou_from_labelmaps(np.zeros_like(gt_map), gt_map)
                tp, fp, fn = 0, 0, G
            else:
                pred_map = load_instance_map(pred_path)
                if gt_map.shape != pred_map.shape:
                    # Conservative: treat as zero preds
                    _, _, G = iou_from_labelmaps(np.zeros_like(gt_map), gt_map)
                    tp, fp, fn = 0, 0, G
                else:
                    tp, fp, fn = match_and_count_from_labelmaps(pred_map, gt_map, iou_thresh)

            tp_tot += tp; fp_tot += fp; fn_tot += fn
            evaluated += 1

        except Exception as e:
            print(f"[WARN] Skipping '{stem}': {e}", file=sys.stderr)
            continue

    return tp_tot, fp_tot, fn_tot, evaluated


def precision_recall(tp: int, fp: int, fn: int) -> Tuple[float, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return prec, rec


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    order = np.argsort(recall)
    r = recall[order]; p = precision[order]
    return float(np.trapz(p, r))


def main():
    args = parse_args()
    thr_dirs = discover_threshold_dirs(args.preds_root, args.thresholds)
    if not thr_dirs:
        print("No threshold subdirectories found. Expect names ending with '_<thr>' like '_0.3'.", file=sys.stderr)
        sys.exit(1)

    results = []  # (thr, TP, FP, FN, prec, rec, n_images)
    for thr in sorted(thr_dirs.keys()):
        pred_dir = thr_dirs[thr]
        tp, fp, fn, n_imgs = evaluate_threshold_dir(args.gt_dir, pred_dir, args.exts, args.iou_thresh)
        prec, rec = precision_recall(tp, fp, fn)
        results.append((thr, tp, fp, fn, prec, rec, n_imgs))
        print(f"[{thr:.1f}] imgs={n_imgs:4d}  TP={tp:6d}  FP={fp:6d}  FN={fn:6d}  Precision={prec:.4f}  Recall={rec:.4f}")

    if not results:
        print("No results computed.", file=sys.stderr)
        sys.exit(1)

        # --- Prepare PR points (keep threshold order unless you prefer sorting) ---
    thresholds = [r[0] for r in results]
    precisions = [r[4] for r in results]
    recalls    = [r[5] for r in results]

    # Optional: assert shapes + basic sanity
    assert len(thresholds) == len(precisions) == len(recalls), "length mismatch building PR arrays"

    # Compute AP using recall-sorted copies (does not affect what we plot)
    order = np.argsort(recalls)
    ap = float(np.trapezoid(np.array(precisions)[order], np.array(recalls)[order]))

    # --- (Recommended) Write the exact points we plot, for sanity checks ---
    # Comment out if you don't want the CSV
    try:
        import csv
        with open("pr_points.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["threshold", "recall(x)", "precision(y)"])
            for t, r, p in zip(thresholds, recalls, precisions):
                w.writerow([f"{t:.3f}", f"{r:.6f}", f"{p:.6f}"])
        print("Wrote PR points to pr_points.csv")
    except Exception as e:
        print(f"[WARN] Could not write pr_points.csv: {e}", file=sys.stderr)

    # --- Plotly PR curve (x=recall, y=precision) ---
    import plotly.graph_objects as go
    fig = go.Figure()

    # Points/line
    fig.add_trace(go.Scatter(
        x=recalls,                 # X = recall
        y=precisions,              # Y = precision
        mode="lines+markers",
        name="PR curve",
        hovertemplate=(
            "Threshold=%{customdata:.1f}<br>"
            "Recall (x)=%{x:.4f}<br>"
            "Precision (y)=%{y:.4f}<extra></extra>"
        ),
        customdata=np.array(thresholds, dtype=float),  # show threshold in hover
    ))

    # Threshold text labels at each point
    fig.add_trace(go.Scatter(
        x=recalls,
        y=precisions,
        mode="text",
        text=[f"{t:.1f}" for t in thresholds],
        textposition="top center",
        name="threshold labels",
        hoverinfo="skip",
        showlegend=False,
    ))

    fig.update_layout(
        title=f"Precision–Recall Curve (IoU ≥ {args.iou_thresh:.2f}) — AP≈{ap:.4f}",
        xaxis_title="Recall (x)",
        yaxis_title="Precision (y)",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        width=900,
        height=600,
        template="plotly_white",
    )

    # Save SVG only (requires kaleido)
    fig.write_image(args.out, format="svg")
    print(f"\nSaved PR curve SVG to: {args.out}")

    # Console table
    print("\nThreshold  Images   TP      FP      FN      Precision   Recall")
    print("---------  ------  ------  ------  ------   ---------   -------")
    for thr, tp, fp, fn, prec, rec, n_imgs in results:
        print(f"{thr:>8.1f}  {n_imgs:>6d}  {tp:>6d}  {fp:>6d}  {fn:>6d}   {prec:>9.4f}   {rec:>7.4f}")


if __name__ == "__main__":
    main()
