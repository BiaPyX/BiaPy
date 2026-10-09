"""
Strategy for the semantic segmentation workflow, from the fingerprints of the training images and masks.

What is set, following nnU-Net:

* U-Net topology, patch size and batch size from the voxel spacing, the median image shape and the GPU
  memory (see unet.py).
* Loss: Dice + cross-entropy.
* Foreground oversampling: when the foreground is scarce, patches are sampled so that at least a third of
  them are centred on foreground (nnU-Net forces a third of the patches of each batch to contain it).
* The number of classes, from all the labels found in the masks.
* No elastic deformations (nnU-Net v2 does not use them either). Besides, BiaPy 3.7.1's (cv2.remap) can
  deadlock the data loader workers with patches of 512x512 or larger.

What is kept from the BiaPy defaults: normalization (z-score per image after percentile clipping, which is
what nnU-Net does for non-CT images), optimizer, learning rate schedule, epochs and the other augmentations
(BiaPy's 3D rotations are in-plane and its zoom leaves Z alone, as nnU-Net's for anisotropic data).
"""
from . import common


def plan_semantic_seg(cfg, sample_info, vram_bytes, vram_source):
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
    return {"applied": True, "lines": lines, "plan": plan, "spacing": spacing, "spacing_source": spacing_source}
