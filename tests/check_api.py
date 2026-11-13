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
  4) Load the workflow from the trained ``.pth`` with a ``result_dir`` and run ``test()``:
     logs and results must go to the new job folder, never to the training job's paths
     embedded in the checkpoint.
  5) ``load_workflow_from_model`` rejects a ``.safetensors`` with a helpful message.
  6) Load the workflow from a BMZ nickname and run ``test()`` with GT: the model's channels must
     be in the config from the start (it used to fail the constructor's config check).

The output is organized as in ``run_checks.py``: each check prints a header, the path of the log file that
receives everything BiaPy prints while it runs, and a final ``[OK]``/``[ERROR]`` line.

Run: ``python tests/check_api.py --out_folder <dir> --gpu 0``
"""
import os
import sys
import glob
import argparse
import builtins
import traceback
from contextlib import contextmanager
from zipfile import ZipFile

import numpy as np
import gdown
from skimage.io import imread

from biapy import BiaPy, build_config

# BiaPy replaces builtins.print (adding a timestamp) once a workflow is built: keep the original one so the
# lines of this script look like those of run_checks.py
_print = builtins.print

STARDIST_DRIVE_URL = "https://drive.google.com/uc?id=1b7_WDDGEEaEoIpO_1EefVr0w0VQaetmg"
PATCH = (256, 256, 1)

parser = argparse.ArgumentParser(description="Check BiaPy Python API consistency")
parser.add_argument("--out_folder", required=True, help="Working/output folder")
parser.add_argument("--gpu", default="",
                    help="GPU id (e.g. '0'), '' for CPU or 'auto' for a GPU free right now (skipping the check if none is)")
args = parser.parse_args()


def report(*values):
    """Print to the console (not to the log of the running check)."""
    _print(*values, flush=True)


if args.gpu == "auto":
    # The machine is shared with other jobs: take a GPU nobody is using instead of crashing them (or this check)
    from gpu_utils import free_gpus, gpu_report

    gpus_free = free_gpus()
    if len(gpus_free) == 0:
        report(f"[SKIPPED] No free GPU at the moment, so the API check was not run:\n{gpu_report()}")
        sys.exit(1)
    args.gpu = gpus_free[0]
    report(f"Free GPUs: {gpus_free} ; using {args.gpu}")

inst_dir = os.path.join(args.out_folder, "instance_seg")
data_dir = os.path.join(inst_dir, "Stardist_v2_2D", "data")
train_x = os.path.join(data_dir, "train", "raw")
train_y = os.path.join(data_dir, "train", "label")
test_x = os.path.join(data_dir, "test", "raw")
test_y = os.path.join(data_dir, "test", "label")
result_dir = os.path.join(args.out_folder, "api_output")
# Outside result_dir: some checks assert that nothing is written in their job folder
log_dir = os.path.join(args.out_folder, "api_logs")
os.makedirs(log_dir, exist_ok=True)


def list_files(d):
    """Set of all file paths under ``d`` (empty if it does not exist)."""
    if not os.path.isdir(d):
        return set()
    return {os.path.join(dp, f) for dp, _, fs in os.walk(d) for f in fs}


def check_no_output(checks, jobdir, when):
    """Add to ``checks`` whether ``jobdir`` was not created and holds no files."""
    exists = os.path.isdir(jobdir)
    files = list_files(jobdir)
    checks.append((not exists and not files,
                   f"expected nothing on disk {when}, but dir_exists={exists}, files={sorted(files)[:5]}"))


def first_image(folder):
    """Read the first image found in ``folder``."""
    files = sorted(glob.glob(os.path.join(folder, "*")))
    if not files:
        raise FileNotFoundError(f"No files in {folder}")
    return imread(files[0])


@contextmanager
def logged_to(log_file):
    """
    Send everything written to stdout/stderr to ``log_file`` instead of the console. It is done at file
    descriptor level so that the output of C code and child processes (e.g. data loader workers) goes there too.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    with open(log_file, "w") as f:
        os.dup2(f.fileno(), 1)
        os.dup2(f.fileno(), 2)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])


test_results = []


def run_check(key, description, func, critical=True):
    """
    Run one check as ``run_checks.py`` runs a test. ``func(checks)`` appends ``(passed, message)`` tuples
    to ``checks``. If ``critical`` is False, its failures are reported but do not fail the API check.
    Returns whether it passed.
    """
    report("\n==============================================")
    report(f"Running {key}: {description}")
    report("==============================================")
    log_file = os.path.join(log_dir, key + ".log")
    report(f"Log: {log_file}")
    report("Running job . . .")
    checks = []
    try:
        with logged_to(log_file):
            try:
                func(checks)
            except Exception:
                traceback.print_exc()
                raise
    except Exception as e:  # noqa: BLE001
        report(f"An error occurred during {key} execution.")
        report(f"{type(e).__name__}: {e}")
        checks.append((False, "raised an exception"))

    for passed, message in checks:
        if not passed:
            report(f"Internal check not passed: {message}")
    n_passed = sum(passed for passed, _ in checks)
    if n_passed == len(checks):
        report(f"** {key} job: [OK] ({len(checks)} internal checks passed)")
    else:
        report(f"** {key} job: [ERROR] ({n_passed} of {len(checks)} internal checks passed)")
        if not critical:
            report("(non-critical workflow: not counted as a failure)")
    passed = n_passed == len(checks) or not critical
    test_results.append(passed)
    return passed


def finish():
    """Print the final summary as run_checks.py does and exit."""
    report("\nFinish tests!!")
    report(f"Test passed: ({sum(test_results)}/{len(test_results)})")
    sys.exit(0 if all(test_results) else 1)


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
train_name = "api_instance_train"
checkpoint = os.path.join(result_dir, train_name, "checkpoints", f"{train_name}_1-checkpoint-best.pth")


def train_and_test(checks):
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
    checks.append((os.path.exists(checkpoint), f"checkpoint was not created: {checkpoint}"))


if not run_check("API1", "train + test: INSTANCE_SEG (C channel, unet, 10 epochs)", train_and_test):
    report("The rest of the checks reuse the model trained in API1, so they were not run")
    finish()

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


def predict_with_workflow(wf):
    def check(checks):
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
        check_no_output(checks, jobdir, "after construction")
        pred = b.predict(image, gt=gt) if wf in COMPARE_GT else b.predict(image)
        check_no_output(checks, jobdir, "after predict")

        checks.append((pred is not None, "predict returned None (no prediction captured)"))
        if pred is None:
            return
        print(f"[{wf}] prediction shape: {np.asarray(pred).shape}")
        # The same instance predicts again (e.g. the Fiji plugin keeps the model loaded between images)
        again = b.predict(image)
        checks.append((again is not None and np.asarray(again).shape == np.asarray(pred).shape,
                       "a second predict() on the same instance did not give the same shape"))

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
            checks.append((n_pred > 0, "INSTANCE_SEG predicted 0 instances"))

    return check


for i, wf in enumerate(WORKFLOWS):
    run_check(f"API2.{i + 1}", f"predict via API: {wf}", predict_with_workflow(wf), critical=wf in COMPARE_GT)

# ---------------------------------------------------------------------------
# 3) Load a whole workflow from the trained .pth checkpoint
# ---------------------------------------------------------------------------
# The BiaPy .pth embeds the configuration, so the workflow (INSTANCE_SEG here) is inferred
# and rebuilt automatically, both when passing the checkpoint straight to the constructor
# (third instantiation option) and via BiaPy.load_workflow_from_model. A .safetensors has no
# config inside, so it must be rejected with a message pointing to build_config().


def load_from_checkpoint(entry, label):
    def check(checks):
        name = "api_load_" + label.split("(")[0].replace(".", "_")
        jobdir = os.path.join(result_dir, name)
        # save_files=False (no result_dir): loading a model and predicting must write nothing.
        b = entry(save_files=False, name=name, gpu=args.gpu)
        print(f"[{label}] inferred workflow={b.cfg.PROBLEM.TYPE} dims={b.cfg.PROBLEM.NDIM}")
        checks.append((b.cfg.PROBLEM.TYPE == "INSTANCE_SEG",
                       f"expected INSTANCE_SEG inferred from the .pth, got {b.cfg.PROBLEM.TYPE}"))

        check_no_output(checks, jobdir, "after construction")
        pred = b.predict(image)
        check_no_output(checks, jobdir, "after predict")

        checks.append((pred is not None, "predict returned None"))
        if pred is not None:
            n_pred = int(len(np.unique(pred)) - 1)
            n_gt = int(len(np.unique(gt)) - 1)
            print(f"[{label}] instances predicted={n_pred} vs GT={n_gt}")
            checks.append((n_pred > 0, "predicted 0 instances"))

    return check


for i, (entry, label) in enumerate((
    (lambda **kw: BiaPy(checkpoint, **kw), "BiaPy(checkpoint)"),
    (lambda **kw: BiaPy.load_workflow_from_model(checkpoint, **kw), "load_workflow_from_model"),
)):
    run_check(f"API3.{i + 1}", f"load the workflow from the trained .pth: {label}", load_from_checkpoint(entry, label))

# ---------------------------------------------------------------------------
# 4) Load with a result_dir + test(), and reject .safetensors
# ---------------------------------------------------------------------------
# With a result_dir, the loaded workflow must write only inside the new job folder: the paths
# embedded in the checkpoint (logs, results...) belong to the training job, which may not exist
# here (e.g. a model trained on Colab embeds '/content/...').


def load_with_result_dir(checks):
    name = "api_load_with_result_dir"
    train_jobdir = os.path.join(result_dir, train_name)
    train_files_before = list_files(train_jobdir)
    b = BiaPy.load_workflow_from_model(checkpoint, result_dir=result_dir, name=name, gpu=args.gpu)
    expected_log_dir = os.path.join(result_dir, name, "logs")
    checks.append((os.path.abspath(b.cfg.LOG.LOG_DIR) == os.path.abspath(expected_log_dir),
                   f"LOG.LOG_DIR is '{b.cfg.LOG.LOG_DIR}', expected '{expected_log_dir}'"))
    b.update_config({"TEST.ENABLE": True, "DATA.TEST.PATH": test_x})
    b.test()
    checks.append((len(list_files(os.path.join(result_dir, name, "results"))) > 0, "test() wrote no results"))
    new_in_train_job = list_files(train_jobdir) - train_files_before
    checks.append((not new_in_train_job, f"wrote into the training job folder: {sorted(new_in_train_job)[:5]}"))


run_check("API4", "load the workflow from the trained .pth with a result_dir + test()", load_with_result_dir)


def reject_safetensors(checks):
    # A .safetensors checkpoint must be rejected (no embedded config)
    try:
        BiaPy.load_workflow_from_model("dummy_model.safetensors")
        checks.append((False, "'.safetensors' should have raised"))
    except ValueError as e:
        checks.append(("build_config" in str(e), f"unexpected .safetensors error message: {e}"))


run_check("API5", "load_workflow_from_model rejects a .safetensors with a helpful message", reject_safetensors)

# ---------------------------------------------------------------------------
# 6) Load a workflow from a BMZ nickname + test() with GT
# ---------------------------------------------------------------------------
# The RDF-derived options (e.g. the instance channels) must be in the config from the start: the
# constructor checks it before the workflow merges the BMZ options, and with the schema default
# channels ('B'+'C') that check failed ("SEED_CHANNELS must not be empty").
BMZ_INSTANCE_MODEL = "resourceful-otter"  # 2D F+C instance segmentation


def load_from_bmz(checks):
    name = "api_load_bmz"
    b = BiaPy.load_workflow_from_model(BMZ_INSTANCE_MODEL, result_dir=result_dir, name=name, gpu=args.gpu)
    checks.append((b.cfg.PROBLEM.TYPE == "INSTANCE_SEG", f"expected INSTANCE_SEG, got {b.cfg.PROBLEM.TYPE}"))
    channels = list(b.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS)
    checks.append((channels[:2] == ["F", "C"], f"expected the model's F+C channels, got {channels}"))
    b.update_config({"TEST.ENABLE": True, "DATA.TEST.PATH": test_x, "DATA.TEST.LOAD_GT": True, "DATA.TEST.GT_PATH": test_y})
    b.test()
    checks.append((len(list_files(os.path.join(result_dir, name, "results"))) > 0, "test() wrote no results"))


run_check("API6", f"load the workflow from a BMZ nickname ({BMZ_INSTANCE_MODEL}) + test() with GT", load_from_bmz)

finish()
