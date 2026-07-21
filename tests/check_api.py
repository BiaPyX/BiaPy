"""
Consistency check for BiaPy's Python API (``biapy.build_config`` / ``BiaPy`` / ``predict``).

Steps:
  1) Train + test an instance segmentation model (``C`` channel only, ``unet``, 10 epochs) on
     the Stardist_v2_2D data, fully through the API.
  2) Reusing those weights, build a fresh ``BiaPy`` per workflow, load the model and run
     ``predict`` over an in-memory image (the output is meaningless for non-instance
     workflows, that is fine). For instance/semantic segmentation the prediction is compared
     with the GT; for the rest metrics are skipped.
  3) Assert that a predict-only workflow (built with ``save_files=False``) writes NOTHING to
     disk: the output folder is never even created, neither at construction nor after
     ``predict``.

Run: ``python tests/check_api.py --out_folder <dir> --gpu 0``
"""
import os
import sys
import glob
import argparse
from zipfile import ZipFile

import numpy as np
import gdown
from skimage.io import imread

from biapy import BiaPy, build_config

STARDIST_DRIVE_URL = "https://drive.google.com/uc?id=1b7_WDDGEEaEoIpO_1EefVr0w0VQaetmg"
PATCH = (256, 256, 1)

parser = argparse.ArgumentParser(description="Check BiaPy Python API consistency")
parser.add_argument("--out_folder", required=True, help="Working/output folder")
parser.add_argument("--gpu", default="", help="GPU id (e.g. '0') or '' for CPU")
args = parser.parse_args()

inst_dir = os.path.join(args.out_folder, "instance_seg")
data_dir = os.path.join(inst_dir, "Stardist_v2_2D", "data")
train_x = os.path.join(data_dir, "train", "raw")
train_y = os.path.join(data_dir, "train", "label")
test_x = os.path.join(data_dir, "test", "raw")
test_y = os.path.join(data_dir, "test", "label")
result_dir = os.path.join(args.out_folder, "api_output")


def list_files(d):
    """Set of all file paths under ``d`` (empty if it does not exist)."""
    if not os.path.isdir(d):
        return set()
    return {os.path.join(dp, f) for dp, _, fs in os.walk(d) for f in fs}


def assert_no_output(tag, jobdir, when):
    """Assert that ``jobdir`` was not created and holds no files. Returns True when clean."""
    exists = os.path.isdir(jobdir)
    files = list_files(jobdir)
    if exists or files:
        print(f"[{tag}] FAIL ({when}): expected nothing on disk, but dir_exists={exists}, "
              f"files={sorted(files)[:5]}")
        return False
    print(f"[{tag}] OK ({when}): no output folder / files created")
    return True


def first_image(folder):
    """Read the first image found in ``folder``."""
    files = sorted(glob.glob(os.path.join(folder, "*")))
    if not files:
        raise FileNotFoundError(f"No files in {folder}")
    return imread(files[0])


# ---------------------------------------------------------------------------
# 0) Download the Stardist_v2_2D data if not present
# ---------------------------------------------------------------------------
if not os.path.isdir(train_x):
    os.makedirs(inst_dir, exist_ok=True)
    zip_path = os.path.join(inst_dir, "Stardist_v2_2D.zip")
    if not os.path.exists(zip_path):
        gdown.download(STARDIST_DRIVE_URL, zip_path, quiet=False)
    with ZipFile(zip_path) as z:
        z.extractall(os.path.join(inst_dir, "Stardist_v2_2D"))

# ---------------------------------------------------------------------------
# 1) Train + test instance seg (C channel, unet, 10 epochs) through the API
# ---------------------------------------------------------------------------
print("\n=== [1] API train + test: INSTANCE_SEG (C channel, unet, 10 epochs) ===")
train_name = "api_instance_train"
train_cfg = build_config(
    workflow="INSTANCE_SEG",
    dims="2D",
    phase="both",
    patch_size=PATCH,
    model={"architecture": "unet"},
    train_data={"path": train_x, "gt_path": train_y, "in_memory": True},
    val_data={"split_train": 0.1},
    test_data={"path": test_x, "gt_path": test_y, "in_memory": False, "load_gt": True},
    extra_config={
        "PROBLEM": {"INSTANCE_SEG": {"DATA_CHANNELS": "C"}},
        "TRAIN": {"EPOCHS": 10, "PATIENCE": -1},
    },
)
biapy = BiaPy(train_cfg, result_dir=result_dir, name=train_name, gpu=args.gpu)
biapy.run_job()

checkpoint = os.path.join(result_dir, train_name, "checkpoints", f"{train_name}_1-checkpoint-best.pth")
assert os.path.exists(checkpoint), f"Checkpoint was not created: {checkpoint}"
print(f"Checkpoint created: {checkpoint}")

# A test image + its GT to predict over / compare against
image = first_image(test_x)
gt = first_image(test_y)

# ---------------------------------------------------------------------------
# 2) Reuse the weights: predict per workflow + assert no files written
# ---------------------------------------------------------------------------
# All these workflows build a 1-channel 2D unet, so the trained weights load; only the
# weights are loaded (ITEMS_TO_LOAD_FROM_CHECKPOINT=['weights']) and unmatched layers are
# skipped, so the same model can be reused across workflows.
COMPARE_GT = {"INSTANCE_SEG", "SEMANTIC_SEG"}
WORKFLOWS = ["INSTANCE_SEG", "SEMANTIC_SEG", "DENOISING", "SUPER_RESOLUTION", "IMAGE_TO_IMAGE"]

overall_ok = True
for wf in WORKFLOWS:
    print(f"\n=== [2] predict via API: {wf} ===")
    name = f"api_predict_{wf.lower()}"
    jobdir = os.path.join(result_dir, name)

    extra = {
        "PATHS": {"CHECKPOINT_FILE": checkpoint},
        "MODEL": {"ITEMS_TO_LOAD_FROM_CHECKPOINT": ["weights"]},
    }
    if wf == "INSTANCE_SEG":
        extra["PROBLEM"] = {"INSTANCE_SEG": {"DATA_CHANNELS": "C"}}
    elif wf == "SEMANTIC_SEG":
        extra["DATA"] = {"N_CLASSES": 2}
    elif wf == "SUPER_RESOLUTION":
        extra["PROBLEM"] = {"SUPER_RESOLUTION": {"UPSCALING": (1, 1)}}  # restoration -> keep size
        extra["DATA"] = {"NORMALIZATION": {"TYPE": "div"}}  # SR requires div/scale_range norm

    hard = wf in COMPARE_GT
    try:
        # save_files=False (and no result_dir): an ephemeral predict-only workflow that must
        # not touch the disk at all.
        test_cfg = build_config(
            workflow=wf,
            dims="2D",
            phase="test",
            patch_size=PATCH,
            model={"architecture": "unet", "load_checkpoint": True, "skip_unmatched_layers": True},
            test_data={"path": test_x},
            extra_config=extra,
        )
        b = BiaPy(test_cfg, name=name, gpu=args.gpu, save_files=False)

        # Nothing must be written: not at construction, and not by predict().
        clean = assert_no_output(wf, jobdir, "after construction")
        pred = b.predict(image, gt=gt) if hard else b.predict(image)
        clean = assert_no_output(wf, jobdir, "after predict") and clean
        if not clean:
            overall_ok = False

        if pred is None:
            msg = f"[{wf}] predict returned None (no prediction captured)"
            if hard:
                print(f"FAIL: {msg}")
                overall_ok = False
            else:
                print(f"WARN: {msg}")
            continue
        print(f"[{wf}] prediction shape: {np.asarray(pred).shape}")

        # GT comparison only for instance/semantic segmentation
        if wf == "SEMANTIC_SEG":
            prob = np.asarray(pred).squeeze()
            pbin = (prob > 0.5).astype(np.uint8)
            gtb = (np.asarray(gt).squeeze() > 0).astype(np.uint8)
            inter = np.logical_and(pbin, gtb).sum()
            union = np.logical_or(pbin, gtb).sum()
            iou = float(inter) / float(union) if union else 0.0
            print(f"[SEMANTIC_SEG] foreground IoU vs GT: {iou:.3f}")
        elif wf == "INSTANCE_SEG":
            n_pred = int(len(np.unique(pred)) - 1)  # minus background
            n_gt = int(len(np.unique(gt)) - 1)
            print(f"[INSTANCE_SEG] instances predicted={n_pred} vs GT={n_gt}")
            if n_pred == 0:
                print("FAIL: INSTANCE_SEG predicted 0 instances")
                overall_ok = False

    except Exception as e:  # noqa: BLE001
        msg = f"[{wf}] raised: {type(e).__name__}: {e}"
        if hard:
            print(f"FAIL: {msg}")
            overall_ok = False
        else:
            print(f"WARN (non-critical workflow): {msg}")

# ---------------------------------------------------------------------------
# 3) Load a whole workflow from the trained .pth checkpoint
# ---------------------------------------------------------------------------
# The BiaPy .pth embeds the configuration, so the workflow (INSTANCE_SEG here) is inferred
# and rebuilt automatically, both when passing the checkpoint straight to the constructor
# (third instantiation option) and via BiaPy.load_workflow_from_model. A .safetensors has no
# config inside, so it must be rejected with a message pointing to build_config().
print("\n=== [3] load the workflow from the trained .pth ===")
for entry, label in (
    (lambda **kw: BiaPy(checkpoint, **kw), "BiaPy(checkpoint)"),
    (lambda **kw: BiaPy.load_workflow_from_model(checkpoint, **kw), "load_workflow_from_model"),
):
    name = "api_load_" + label.split("(")[0].replace(".", "_")
    jobdir = os.path.join(result_dir, name)
    try:
        # save_files=False (no result_dir): loading a model and predicting must write nothing.
        b = entry(save_files=False, name=name, gpu=args.gpu)
        print(f"[{label}] inferred workflow={b.cfg.PROBLEM.TYPE} dims={b.cfg.PROBLEM.NDIM}")
        if b.cfg.PROBLEM.TYPE != "INSTANCE_SEG":
            print(f"FAIL: expected INSTANCE_SEG inferred from the .pth, got {b.cfg.PROBLEM.TYPE}")
            overall_ok = False

        clean = assert_no_output(label, jobdir, "after construction")
        pred = b.predict(image)
        clean = assert_no_output(label, jobdir, "after predict") and clean
        if not clean:
            overall_ok = False

        if pred is None:
            print(f"FAIL: [{label}] predict returned None")
            overall_ok = False
        else:
            n_pred = int(len(np.unique(pred)) - 1)
            n_gt = int(len(np.unique(gt)) - 1)
            print(f"[{label}] instances predicted={n_pred} vs GT={n_gt}")
            if n_pred == 0:
                print(f"FAIL: [{label}] predicted 0 instances")
                overall_ok = False
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: [{label}] raised: {type(e).__name__}: {e}")
        overall_ok = False

# A .safetensors checkpoint must be rejected (no embedded config)
try:
    BiaPy.load_workflow_from_model("dummy_model.safetensors")
    print("FAIL: '.safetensors' should have raised")
    overall_ok = False
except ValueError as e:
    if "build_config" in str(e):
        print("[load_workflow_from_model] OK: .safetensors correctly rejected with a helpful message")
    else:
        print(f"FAIL: unexpected .safetensors error message: {e}")
        overall_ok = False

print("\n==================================")
print("API check: " + ("PASSED" if overall_ok else "FAILED"))
print("==================================")
sys.exit(0 if overall_ok else 1)
