"""
Configuration adapted to the data, in the spirit of nnU-Net (https://github.com/MIC-DKFZ/nnUNet): with the
fingerprints of the data checked in the wizard (:mod:`biapy.wizard.fingerprint`), the workflow's strategy sets
the model, patch size, batch size, loss and sampling (:mod:`biapy.wizard.unet` and one module per workflow).

The wizard does not ask for the size of the objects to choose the patch size: workflows without a strategy yet,
or when it can't be applied (e.g. a pretrained model), get the patch size of the checkpoint to load or a fixed
one.
"""
import ast
import importlib
import os

# Strategy of each workflow: "module:function" called as function(cfg, sample_info, vram_bytes, vram_source)
PLANNERS = {
    "SEMANTIC_SEG": "semantic_seg:plan_semantic_seg",
    "INSTANCE_SEG": "instance_seg:plan_instance_seg",
    "DETECTION": "detection:plan_detection",
}
# Patch size of the workflows without a strategy (or when it can't be applied) and no checkpoint to take it
# from. Denoising and super-resolution keep the ones BiaPy-GUI sets for them (see set_default_config()).
FIXED_PATCH_SIZE = {"2D": (256, 256), "3D": (20, 256, 256)}

# GPU memory assumed when no NVIDIA GPU is found (CPU or Apple silicon), so the plan stays small
DEFAULT_VRAM_MB = 4096


def query_gpus():
    """NVIDIA GPUs from nvidia-smi, which numbers them the same way BiaPy's ``--gpu`` does (no PyTorch needed)."""
    import shutil
    import subprocess

    gpus = []
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.run(
                [smi, "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=20,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return gpus
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3 and parts[0].isdigit():
                gpus.append({"index": int(parts[0]), "name": parts[1], "memory_mb": int(float(parts[2]))})
    return gpus


def vram_for_planning(gpus):
    """GPU memory to plan for (bytes) and a description. ``gpus`` as returned by :func:`query_gpus`."""
    if gpus:
        gpu = max(gpus, key=lambda g: g["memory_mb"])
        return gpu["memory_mb"] * 1024**2, "{} MB of GPU {} ({})".format(gpu["memory_mb"], gpu["index"], gpu["name"])
    return DEFAULT_VRAM_MB * 1024**2, "no NVIDIA GPU found, {} MB assumed".format(DEFAULT_VRAM_MB)


def plan_config(cfg, sample_info, gpus):
    """
    Adapts ``cfg`` (the configuration dictionary built from the Wizard answers) to the data.

    Returns
    -------
    report : dict
        ``applied``, ``reason`` (when not applied), ``lines`` (human readable summary of the plan) and
        ``patch_size_source`` when the patch size was not planned.
    """
    workflow = cfg.get("PROBLEM", {}).get("TYPE")
    if workflow in PLANNERS:
        module, function = PLANNERS[workflow].split(":")
        planner = getattr(importlib.import_module("biapy.wizard." + module), function)
        vram, source = vram_for_planning(gpus)
        report = planner(cfg, sample_info, vram, source)
    else:
        report = {"applied": False, "reason": "no strategy for {} yet".format(workflow)}
    if cfg["MODEL"].get("LOAD_CHECKPOINT"):
        apply_checkpoint_settings(cfg, report)
    if patch_size_unset(cfg):
        set_unplanned_patch_size(cfg, report)
    return report


def patch_size_unset(cfg):
    patch = cfg.get("DATA", {}).get("PATCH_SIZE")
    if isinstance(patch, str):
        patch = ast.literal_eval(patch)
    return not patch or patch[0] == -1


def checkpoint_cfg(path):
    """Configuration a BiaPy checkpoint (.pth) was trained with (in the current BiaPy's format), or None."""
    if not path or not str(path).endswith(".pth") or not os.path.isfile(path):
        return None
    from biapy.engine.check_configuration import convert_old_model_cfg_to_current_version
    from biapy.utils.misc import load_checkpoint_file

    checkpoint = load_checkpoint_file(path)
    cfg = checkpoint.get("cfg") if isinstance(checkpoint, dict) else None
    return convert_old_model_cfg_to_current_version(dict(cfg)) if cfg else None


def apply_checkpoint_settings(cfg, report):
    """
    Settings that must match the checkpoint to load (BiaPy refuses different ones): the patch size, if not
    planned, and the instance segmentation channels (the representation is not asked with a checkpoint).
    """
    lines = report.setdefault("lines", [])
    try:
        ck = checkpoint_cfg(cfg.get("PATHS", {}).get("CHECKPOINT_FILE"))
    except Exception as e:
        lines.append("Could not read the configuration of the checkpoint: {}".format(e))
        return
    if not ck:
        return
    patch = (ck.get("DATA") or {}).get("PATCH_SIZE")
    ndim = 3 if cfg["PROBLEM"]["NDIM"] == "3D" else 2
    if patch_size_unset(cfg) and patch and len(patch) == ndim + 1:
        current = cfg["DATA"].get("PATCH_SIZE")
        current = ast.literal_eval(current) if isinstance(current, str) else current
        channels = current[-1] if current else patch[-1]
        cfg["DATA"]["PATCH_SIZE"] = str(tuple(int(v) for v in patch[:-1]) + (channels,))
        if cfg.get("TEST", {}).get("ENABLE"):
            cfg["DATA"].setdefault("TEST", {})["PADDING"] = str(tuple(int(x) // 6 for x in patch[:-1]))
        report["patch_size_source"] = "the checkpoint"
        lines.append("Patch size {} from the checkpoint".format("x".join(str(int(v)) for v in patch[:-1])))
    inst = ((ck.get("PROBLEM") or {}).get("INSTANCE_SEG") or {})
    if cfg["PROBLEM"].get("TYPE") == "INSTANCE_SEG" and inst.get("DATA_CHANNELS"):
        dst = cfg["PROBLEM"].setdefault("INSTANCE_SEG", {})
        dst["DATA_CHANNELS"] = list(inst["DATA_CHANNELS"])
        if inst.get("DATA_CHANNELS_EXTRA_OPTS"):
            dst["DATA_CHANNELS_EXTRA_OPTS"] = inst["DATA_CHANNELS_EXTRA_OPTS"]
        dst.pop("DATA_MW_TH_TYPE", None)
        lines.append("Instance channels {} from the checkpoint".format(", ".join(dst["DATA_CHANNELS"])))


def set_unplanned_patch_size(cfg, report):
    """Patch size when it was not planned nor taken from a checkpoint: FIXED_PATCH_SIZE."""
    ndim = cfg["PROBLEM"]["NDIM"]
    current = cfg["DATA"].get("PATCH_SIZE")
    current = ast.literal_eval(current) if isinstance(current, str) else current
    channels = current[-1] if current else 1
    patch = FIXED_PATCH_SIZE[ndim]
    cfg["DATA"]["PATCH_SIZE"] = str(tuple(patch) + (channels,))
    if cfg.get("TEST", {}).get("ENABLE"):
        cfg["DATA"].setdefault("TEST", {})["PADDING"] = str(tuple(x // 6 for x in patch))
    report["patch_size_source"] = "fixed default"
    report.setdefault("lines", []).append("Patch size {} (fixed default)".format("x".join(str(v) for v in patch)))
    if "reason" in report:
        report["reason"] += "; patch size {} (fixed default)".format("x".join(str(v) for v in patch))
