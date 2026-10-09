"""
Strategy for the detection workflow, from the fingerprints of the images and of the CSV files with the points.

* Distances from the spacing of the points: BiaPy's training targets are disks around each point, and
  detections closer than half the distance between the closest true neighbours are taken as duplicates.
  With ``min_sep``, the 5th percentile of the distance from each point to its nearest one:
    - points dilated with radius min_sep / 4 (clipped to [1, 5]), so the disks of neighbours never touch;
    - duplicate removal radius, minimum distance between peaks and matching tolerance: min_sep / 2.
  Distances are in XY pixels, with Z scaled by the voxel spacing (set as DATA.TEST.RESOLUTION).
  This part is also applied without training (e.g. testing a pretrained model) if the test points were
  analyzed.
* Training: as for semantic segmentation (nnU-Net-like U-Net, patch and batch size, foreground
  oversampling, no elastic deformations). BiaPy always uses cross-entropy for detection.
"""
import math

import numpy as np

from . import common

NEIGHBOUR_PERCENTILE = 5
MAX_POINT_DILATION = 5


def point_distances(points_fp, spacing):
    """
    Distance from each point to its nearest one, in XY pixels (Z scaled by the anisotropy of ``spacing``),
    or None if there are not enough points.
    """
    v = np.array(points_fp.get("nn_vectors") or [], dtype=float)
    if len(v) == 0:
        return None
    rel = np.array(spacing, dtype=float) / float(spacing[-1])
    return np.sqrt(((v * rel[: v.shape[1]]) ** 2).sum(axis=1))


def detection_distances(points_fp, spacing):
    """Point dilation, removal radius, peak min. distance and tolerance (see the module docstring), or None."""
    d = point_distances(points_fp, spacing)
    if d is None:
        return None
    min_sep = float(np.percentile(d, NEIGHBOUR_PERCENTILE))
    radius = max(1, int(min_sep / 2))
    dilation = int(np.clip(math.floor(min_sep / 4), 1, MAX_POINT_DILATION))
    if len(spacing) == 3:
        ratio = spacing[0] / spacing[-1]
        dilation = [max(1, int(round(dilation / ratio))), dilation, dilation]
    else:
        dilation = [dilation, dilation]
    return {
        "min_sep": min_sep,
        "median_nn": float(np.median(d)),
        "dilation": dilation,
        "radius": radius,
    }


def _relative_spacing(spacing):
    return tuple(round(float(s) / float(spacing[-1]), 3) for s in spacing)


def plan_detection(cfg, sample_info, vram_bytes, vram_source):
    """Applies the strategy to ``cfg``. Returns a report as plan_semantic_seg()."""
    ndim = 3 if cfg["PROBLEM"]["NDIM"] == "3D" else 2
    lines = []
    train_reason = common.not_plannable_reason(cfg)
    raw_fp = common.fingerprint(sample_info, "DATA.TRAIN.PATH", "images") or common.fingerprint(
        sample_info, "DATA.TEST.PATH", "images"
    )
    points_fp = common.fingerprint(sample_info, "DATA.TRAIN.GT_PATH", "detection_points") or common.fingerprint(
        sample_info, "DATA.TEST.GT_PATH", "detection_points"
    )
    if raw_fp is None or points_fp is None:
        return {"applied": False, "reason": train_reason or "the data was not analyzed"}

    spacing, spacing_source = common.choose_spacing(raw_fp, None, ndim)
    common.describe_data(raw_fp, spacing, spacing_source, lines)
    ppi = points_fp["points_per_image"]
    lines.append("Points: {} (median {:.0f} per image)".format(ppi["total"], ppi["median"]))
    dist = detection_distances(points_fp, spacing)
    if dist is not None:
        lines.append(
            "Distance to the nearest point: median {:.1f} px, 5th percentile {:.1f} px".format(
                dist["median_nn"], dist["min_sep"]
            )
        )
        cfg["PROBLEM"].setdefault("DETECTION", {})["CENTRAL_POINT_DILATION"] = dist["dilation"]
        r = dist["dilation"]
        lines.append("Training targets: disks of radius {} px around each point".format(
            r[-1] if len(set(r)) == 1 else common.fmt(r) + " (z, y, x)"))
        if cfg["TEST"].get("ENABLE"):
            pp = cfg["TEST"].setdefault("POST_PROCESSING", {})
            pp["REMOVE_CLOSE_POINTS"] = True
            pp["REMOVE_CLOSE_POINTS_RADIUS"] = dist["radius"]
            cfg["TEST"]["DET_PEAK_LOCAL_MAX_MIN_DISTANCE"] = dist["radius"]
            cfg["TEST"]["DET_TOLERANCE"] = dist["radius"]
            cfg["DATA"].setdefault("TEST", {})["RESOLUTION"] = str(_relative_spacing(spacing))
            lines.append(
                "Detections closer than {} px are merged; matching tolerance {} px".format(dist["radius"], dist["radius"])
            )

    if train_reason:
        return {"applied": dist is not None, "reason": train_reason, "lines": lines}

    n_classes = int(cfg["DATA"].get("N_CLASSES", 2))
    out_channels = 1 + (n_classes if n_classes > 2 else 0)
    plan = common.apply_unet_plan(cfg, raw_fp, spacing, out_channels, vram_bytes, vram_source, lines)
    if plan is None:
        return {"applied": dist is not None, "reason": "the images are too small to plan a U-Net", "lines": lines}
    lines.append("Loss: cross-entropy (BiaPy's for detection); no elastic deformations")

    # Foreground: the dilated points
    if dist is not None:
        disk = float(np.prod([2 * r + 1 for r in (dist["dilation"] if ndim == 3 else dist["dilation"][-2:])]))
        fg = min(1.0, ppi["median"] * disk / float(np.prod(raw_fp["median_shape"])))
        common.foreground_oversampling(cfg, raw_fp, fg, plan["patch_size"], lines)
    return {"applied": True, "lines": lines, "plan": plan, "spacing": spacing, "spacing_source": spacing_source}
