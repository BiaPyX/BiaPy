"""
Parts shared by the workflow strategies: when a strategy can be applied, the voxel spacing to plan with,
the U-Net plan (unet.py) written into the configuration and nnU-Net's foreground oversampling.
"""
import numpy as np

from . import unet

# nnU-Net forces 1/3 of the patches to contain foreground
FOREGROUND_OVERSAMPLING = 0.33
# Random patches make BiaPy split the validation set by images, so they need a few of them
MIN_IMAGES_RANDOM_PATCHES = 10
# Spacing estimated from the shape of the objects: minimum number of objects and anisotropy to use it
MIN_OBJECTS_FOR_ANISOTROPY = 20
MIN_ESTIMATED_ANISOTROPY = 2.0  # below 2 nnU-Net would downsample all axes together anyway
MAX_ESTIMATED_ANISOTROPY = 10.0


def fingerprint(sample_info, key, kind):
    """Fingerprint of the folder checked for ``key`` (e.g. "DATA.TRAIN.PATH") if it is of that kind."""
    fp = (sample_info.get(key) or {}).get("fingerprint")
    return fp if fp and fp.get("kind") == kind and "error" not in fp else None


def not_plannable_reason(cfg):
    """Why the model can't be planned (no training, pretrained model), or None."""
    if not cfg["TRAIN"].get("ENABLE"):
        return "no training"
    if cfg["MODEL"].get("SOURCE", "biapy") != "biapy" or cfg["MODEL"].get("LOAD_CHECKPOINT"):
        return "the model is pretrained, its architecture can't be changed"
    return None


def fmt(v):
    return "x".join(str(int(round(x))) if float(x).is_integer() else "{:.3g}".format(x) for x in v)


def estimate_anisotropy_from_objects(objects):
    """
    Z/XY spacing ratio of a 3D dataset without calibration, assuming that its objects are, on average, as
    large along Z as in XY (in physical units). ``objects`` as in the masks fingerprints ({label: {"count",
    "extent_median", ...}}); the label with most objects is used. None if not reliable.
    """
    if not objects:
        return None
    obj = max(objects.values(), key=lambda o: o["count"])
    if obj["count"] < MIN_OBJECTS_FOR_ANISOTROPY or len(obj["extent_median"]) != 3:
        return None
    z, y, x = obj["extent_median"]
    ratio = ((y + x) / 2) / max(z, 1.0)
    if ratio < MIN_ESTIMATED_ANISOTROPY:
        return None
    return float(min(round(ratio, 1), MAX_ESTIMATED_ANISOTROPY))


def choose_spacing(raw_fp, objects, ndim):
    """Spacing used for planning and where it comes from."""
    if raw_fp.get("spacing") and raw_fp["n_calibrated"] > 0:
        sp = raw_fp["spacing"]
        source = "image metadata ({} of {} images calibrated)".format(raw_fp["n_calibrated"], raw_fp["n_files"])
        return [float(v) for v in sp], source
    if ndim == 3:
        ratio = estimate_anisotropy_from_objects(objects)
        if ratio is not None:
            return [ratio, 1.0, 1.0], "estimated from the shape of the objects in the masks (no calibration found)"
    return [1.0] * ndim, "assumed isotropic (no calibration found)"


def describe_data(raw_fp, spacing, spacing_source, lines):
    unit = " µm" if raw_fp.get("spacing") else ""
    lines.append("Voxel spacing: {}{} ({})".format(fmt(spacing), unit, spacing_source))
    if not raw_fp.get("spacing_consistent", True):
        lines.append("WARNING: the images do not have the same voxel spacing; the median one was used")
    lines.append(
        "Images: {} with {} channel(s), median shape {}".format(
            raw_fp["n_files"], raw_fp["channels"], fmt(raw_fp["median_shape"])
        )
    )


def train_fraction(cfg):
    return 1 - float(cfg["DATA"].get("VAL", {}).get("SPLIT_TRAIN", 0.1))


def apply_unet_plan(cfg, raw_fp, spacing, out_channels, vram_bytes, vram_source, lines):
    """
    Plans the U-Net (unet.py) and writes it into ``cfg``: model, patch size, test padding, batch size and no
    elastic deformations (nnU-Net v2 does not use them either; besides, BiaPy 3.7.1's can deadlock the data
    loader workers with large patches). Returns the plan, or None if the images are too small for a U-Net.
    """
    ndim = len(spacing)
    plan = unet.plan_unet(
        spacing, raw_fp["median_shape"], raw_fp["channels"], out_channels,
        raw_fp["total_voxels"] * train_fraction(cfg), vram_bytes,
    )
    if len(plan["pools"]) < 2:
        return None
    patch = plan["patch_size"]
    cfg["MODEL"].pop("Z_DOWN", None)
    cfg["MODEL"].update(unet.plan_to_biapy(plan))
    cfg["DATA"]["PATCH_SIZE"] = str(tuple(patch) + (raw_fp["channels"],))
    if cfg["TEST"].get("ENABLE"):
        cfg["DATA"].setdefault("TEST", {})["PADDING"] = str(tuple(x // 6 for x in patch))
    cfg["TRAIN"]["BATCH_SIZE"] = plan["batch_size"]
    cfg.setdefault("AUGMENTOR", {})["ELASTIC"] = False

    lines.append(
        "U-Net: {} levels, feature maps {}, downsampling per level {}".format(
            len(plan["feature_maps"]), plan["feature_maps"], " ".join(fmt(p) for p in plan["pools"])
        )
    )
    if ndim == 3 and not all(k[0] == 3 for k in plan["conv_kernels"]):
        lines.append(
            "Anisotropic data: 1x3x3 kernels in the first {} level(s)".format(
                sum(1 for k in plan["conv_kernels"] if k[0] != 3)
            )
        )
    lines.append(
        "Patch {}, batch size {} (estimated {} GB of {} GB usable; GPU memory: {})".format(
            fmt(patch), plan["batch_size"], plan["vram_estimate_gb"], plan["vram_budget_gb"], vram_source
        )
    )
    return plan


def warn_objects_larger_than_patch(objects, patch, lines, what="objects of class {}"):
    for lab, obj in (objects or {}).items():
        if any(e > p for e, p in zip(obj["extent_p90"], patch)):
            lines.append(
                "WARNING: {} are often larger than the patch (90th percentile size {})".format(
                    what.format(lab), fmt(obj["extent_p90"])
                )
            )


def foreground_oversampling(cfg, raw_fp, fg_fraction, patch, lines):
    """
    nnU-Net's foreground oversampling with BiaPy's probability map: when the foreground is scarce, patches
    are taken at random so that at least a third of them are centred on foreground.
    """
    if fg_fraction >= FOREGROUND_OVERSAMPLING:
        return
    n_train = int(raw_fp["n_files"] * train_fraction(cfg))
    if n_train < MIN_IMAGES_RANDOM_PATCHES:
        lines.append(
            "Foreground is scarce ({:.1%} of the voxels in the median image) but there are too few images to "
            "sample patches at random (at least {} are needed for the validation split); the images are tiled "
            "instead".format(fg_fraction, MIN_IMAGES_RANDOM_PATCHES)
        )
        return
    w_fg = round(FOREGROUND_OVERSAMPLING + (1 - FOREGROUND_OVERSAMPLING) * fg_fraction, 2)
    tiles = float(np.prod([np.ceil(s / p) for s, p in zip(raw_fp["median_shape"], patch)]))
    dtrain = cfg["DATA"].setdefault("TRAIN", {})
    dtrain["EXTRACT_RANDOM_PATCH"] = True
    dtrain["PROBABILITY_MAP"] = True
    dtrain["W_FOREGROUND"] = w_fg
    dtrain["W_BACKGROUND"] = round(1 - w_fg, 2)
    # As many patches per epoch as tiles would have been extracted from the images
    dtrain["REPLICATE"] = max(1, int(round(tiles)))
    lines.append(
        "Foreground is scarce ({:.1%} of the voxels in the median image): random patches, {:.0%} of them "
        "centred on foreground".format(fg_fraction, w_fg)
    )
