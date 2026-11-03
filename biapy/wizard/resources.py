"""
RAM (training and test), GPU memory (training) and disk a wizard configuration needs, estimated for the front-ends to
tell the user. Measured constants:

* Training RAM, the largest of: computing the normalization of each image (1.3 GiB + image + 16 bytes per voxel, 8
  more per extra channel), the class locations of foreground oversampling (1.3 GiB + mask + 5 bytes per voxel, 2 with
  instance labels) and the training loop (2.5 GiB + workers * (image + mask)).
* Disk: checkpoint, 12 bytes per parameter (weights and AdamW moments); raw predictions, 4 bytes per voxel and output
  channel (2 with TEST.REDUCE_MEMORY); final segmentations, about 1 byte per voxel; class locations, about 75 KB per
  training image.

The test RAM is estimated in :mod:`biapy.wizard.inference`.
"""
import os

import numpy as np

from . import common

GIB = 1024**3
# The three phases of training described above: fixed bytes and bytes per voxel of the largest training image
TRAIN_PREP_FIXED_BYTES = 1.3 * GIB
TRAIN_PREP_BYTES_PER_VOXEL = 16  # one channel; each more adds 8
TRAIN_LOCATIONS_BYTES_PER_VOXEL = {"INSTANCE_SEG": 2, "other": 5}
TRAIN_LOOP_FIXED_BYTES = 2.5 * GIB
CHECKPOINT_BYTES_PER_PARAMETER = 12
PREDICTION_BYTES_PER_VOXEL = {"float32": 4, "float16": 2}
SEGMENTATION_BYTES_PER_VOXEL = 1
CLASS_LOCATIONS_BYTES_PER_IMAGE = 75e3


def _gb(b):
    return "{:.1f} GB".format(b / 1e9) if b >= 0.05e9 else "{:.0f} MB".format(b / 1e6)


def _itemsize(fp, default=1):
    return max([np.dtype(d).itemsize for d in (fp or {}).get("dtype") or []] or [default])


def _patch(cfg):
    from .inference import _tuple

    ndim = 3 if cfg["PROBLEM"]["NDIM"] == "3D" else 2
    patch = _tuple(cfg["DATA"].get("PATCH_SIZE"), ndim + 1, default=None)
    return patch if patch and all(p is not None and p > 0 for p in patch) else None


def estimate_resources(cfg, sample_info, report, test_estimate=None, ram_bytes=None, vram_bytes=None):
    """
    RAM, GPU memory and disk space the job of ``cfg`` (the Wizard's configuration, already planned) will need.

    Parameters
    ----------
    cfg : dict
        The configuration (as written to the YAML).
    sample_info : dict
        The data checks of the Wizard (their fingerprints).
    report : dict
        The plan (:func:`biapy.wizard.planning.plan_config`), for the U-Net it planned.
    test_estimate : dict, optional
        How the test images are predicted and the RAM it takes (:func:`biapy.wizard.inference.plan_inference`).
    ram_bytes, vram_bytes : float, optional
        RAM of this computer and GPU memory planned for.

    Returns
    -------
    resources : dict
        ``ram_train``, ``ram_test``, ``ram_total``, ``gpu_train``, ``gpu_total`` (bytes, or None if not estimated),
        ``test_mode`` (how the test images are predicted: "whole", "reduce" or "chunks"),
        ``disk`` (list of ``{"what", "bytes"}``), ``disk_total`` and ``lines``, a human readable summary.
    """
    out = {
        "ram_total": ram_bytes, "ram_train": None, "ram_test": None, "test_mode": None, "gpu_train": None,
        "gpu_total": vram_bytes,
    }
    disk = []
    workflow = cfg["PROBLEM"]["TYPE"]
    train = bool(cfg.get("TRAIN", {}).get("ENABLE"))
    test = bool(cfg.get("TEST", {}).get("ENABLE"))
    plan = report.get("plan")
    patch = _patch(cfg)
    out_ch = common.output_channels(cfg)

    if train:
        raw = common.fingerprint(sample_info, "DATA.TRAIN.PATH", "images")
        if raw:
            voxels = float(np.prod(raw["max_shape"]))
            channels = raw.get("channels") or 1
            image = voxels * channels * _itemsize(raw)
            # Instance labels are usually 16 or 32 bits; classes and points, 8
            mask = voxels * (2 if workflow == "INSTANCE_SEG" else 1) if sample_info.get("DATA.TRAIN.GT_PATH") else 0
            workers = int(cfg.get("SYSTEM", {}).get("NUM_WORKERS", 3))
            workers = workers if workers >= 0 else 3
            phases = [
                TRAIN_PREP_FIXED_BYTES + image + voxels * (TRAIN_PREP_BYTES_PER_VOXEL + 8 * (channels - 1)),
                TRAIN_LOOP_FIXED_BYTES + workers * (image + mask),
            ]
            if cfg["DATA"].get("TRAIN", {}).get("PATCH_SAMPLING") == "foreground_oversampling":
                per_voxel = TRAIN_LOCATIONS_BYTES_PER_VOXEL.get(workflow, TRAIN_LOCATIONS_BYTES_PER_VOXEL["other"])
                phases.append(TRAIN_PREP_FIXED_BYTES + mask + voxels * per_voxel)
                disk.append(("training patch locations", raw["n_files"] * CLASS_LOCATIONS_BYTES_PER_IMAGE))
            out["ram_train"] = max(phases)
        if plan:
            out["gpu_train"] = plan["vram_estimate_gb"] * GIB
            from . import unet

            in_ch = (raw.get("channels") if raw else None) or 1
            params = unet.count_parameters(plan, in_ch, out_ch, plan["feature_maps"])
            disk.append(("model checkpoint", params * CHECKPOINT_BYTES_PER_PARAMETER))
        elif cfg["MODEL"].get("LOAD_CHECKPOINT") and os.path.isfile(str(cfg.get("PATHS", {}).get("CHECKPOINT_FILE", ""))):
            disk.append(("model checkpoint", os.path.getsize(cfg["PATHS"]["CHECKPOINT_FILE"])))
        freq = int(cfg["TRAIN"].get("SAVE_TRAIN_PREDS_FREQ", -1))
        if freq > 0 and patch:
            saves = int(np.ceil(int(cfg["TRAIN"].get("EPOCHS", 1)) / freq))
            samples = int(cfg["TRAIN"].get("SAVE_TRAIN_PREDS_NUM_SAMPLES", 1))
            disk.append(("training predictions", saves * samples * float(np.prod(patch[:-1])) * out_ch * 4))

    if test:
        fp = common.fingerprint(sample_info, "DATA.TEST.PATH", "images")
        if fp:
            voxels = float(fp["total_voxels"])
            if workflow == "SUPER_RESOLUTION":
                factor = cfg["PROBLEM"].get("SUPER_RESOLUTION", {}).get("UPSCALING")
                voxels *= float(np.prod(factor)) if factor else 1
            if test_estimate:
                out["ram_test"] = test_estimate["bytes"]
                out["test_mode"] = test_estimate["mode"]
            mode = (test_estimate or {}).get("mode", "whole")
            if workflow != "CLASSIFICATION":
                if mode != "chunks":
                    dtype = "float16" if cfg["TEST"].get("REDUCE_MEMORY") else "float32"
                    disk.append(("test predictions", voxels * out_ch * PREDICTION_BYTES_PER_VOXEL[dtype]))
                if workflow in ("SEMANTIC_SEG", "INSTANCE_SEG", "DETECTION"):
                    tif = mode == "chunks" and cfg["TEST"].get("BY_CHUNKS", {}).get("SAVE_OUT_TIF")
                    disk.append(("test segmentations", voxels * SEGMENTATION_BYTES_PER_VOXEL * (2 if tif else 1)))

    out["disk"] = [{"what": w, "bytes": float(b)} for w, b in disk]
    out["disk_total"] = float(sum(b for _, b in disk))
    out["lines"] = _lines(out, train, test)
    return out


def _lines(r, train, test):
    lines = []
    ram = [
        "about {} to train".format(_gb(r["ram_train"])) if r["ram_train"] else None,
        "about {} to test ({})".format(
            _gb(r["ram_test"]),
            {"whole": "images predicted whole", "reduce": "with TEST.REDUCE_MEMORY", "chunks": "by chunks"}[r["test_mode"]],
        )
        if r["ram_test"]
        else None,
    ]
    ram = [x for x in ram if x]
    if ram:
        lines.append(
            "RAM: {}{}".format(" and ".join(ram), " (this computer has {})".format(_gb(r["ram_total"])) if r["ram_total"] else "")
        )
    if r["gpu_train"]:
        lines.append(
            "GPU memory: about {} to train{}".format(
                _gb(r["gpu_train"]), " (of {})".format(_gb(r["gpu_total"])) if r["gpu_total"] else ""
            )
        )
    if r["disk"]:
        lines.append("Disk: about {} in the job's folder".format(_gb(r["disk_total"])))
    return lines
