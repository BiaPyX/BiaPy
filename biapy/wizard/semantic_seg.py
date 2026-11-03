"""
Strategy for the semantic segmentation workflow, following nnU-Net: U-Net topology, patch and batch size from the
voxel spacing, median image shape and GPU memory (:mod:`biapy.wizard.unet`), Dice + cross-entropy, foreground
oversampling when the foreground is scarce, the classes found in the masks, nnU-Net's amount of training
(common.training_budget) and no elastic deformations. The rest are BiaPy's defaults.
"""
from . import common


def plan_semantic_seg(cfg, sample_info, vram_bytes, vram_source, training_length="full"):
    """
    Applies the strategy to ``cfg`` (the dictionary that becomes the YAML).

    Returns
    -------
    report : dict
        ``applied`` (bool), ``reason`` (when not applied), ``lines`` (human readable summary) and ``plan``.
    """
    reason = common.not_plannable_reason(cfg)
    if reason:
        return {"applied": False, "reason": reason}
    raw_fp = common.fingerprint(sample_info, "DATA.TRAIN.PATH", "images")
    mask_fp = common.fingerprint(sample_info, "DATA.TRAIN.GT_PATH", "semantic_masks")
    if raw_fp is None or mask_fp is None:
        return {"applied": False, "reason": "the training data was not analyzed"}

    ndim = 3 if cfg["PROBLEM"]["NDIM"] == "3D" else 2
    lines = []
    spacing, spacing_source = common.choose_spacing(raw_fp, mask_fp.get("objects"), ndim)
    common.describe_data(raw_fp, spacing, spacing_source, lines)

    # Classes: from all the labels found when checking the masks (biapy.wizard.data_check.semantic_classes)
    n_classes = int(cfg["DATA"].get("N_CLASSES", 2))
    labels = mask_fp["labels"]
    if n_classes > 2 and len(labels) < n_classes:
        missing = sorted(set(range(n_classes)) - set(int(v) for v in labels))
        lines.append("WARNING: labels {} do not appear in the analyzed masks".format(missing))
    fractions = ", ".join("{}: {:.1%}".format(k, v) for k, v in mask_fp["class_voxel_fraction"].items())
    lines.append("Classes: {} (fraction of the voxels per label: {})".format(n_classes, fractions))

    out_channels = n_classes if n_classes > 2 else 1
    plan = common.apply_unet_plan(cfg, raw_fp, spacing, out_channels, vram_bytes, vram_source, lines)
    if plan is None:
        return {"applied": False, "reason": "the images are too small to plan a U-Net", "lines": lines}
    cfg.setdefault("LOSS", {})
    cfg["LOSS"]["TYPE"] = ["DICE", "CE"]
    cfg["LOSS"]["WEIGHTS"] = [1.0, 1.0]
    lines.append("Loss: Dice + cross-entropy; no elastic deformations")

    common.warn_objects_larger_than_patch(mask_fp.get("objects"), plan["patch_size"], lines)
    common.foreground_oversampling(cfg, raw_fp, mask_fp["foreground_fraction"]["median"], plan["patch_size"], lines)
    common.training_budget(cfg, raw_fp, plan["patch_size"], lines, training_length)
    return {
        "applied": True, "training_planned": True, "lines": lines, "plan": plan, "spacing": spacing,
        "spacing_source": spacing_source,
    }
