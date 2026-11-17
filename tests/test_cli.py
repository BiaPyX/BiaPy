"""
Tests of the subcommands of the ``biapy`` command (``validate-config``, ``check-data``, ``list-workflows``,
``inspect-image``, ``get-config-template``, ``create-config``, ``search-bmz-models``, ``inspect-results`` and ``plan``):
their ``--json`` output (``schema_version``, ``ok``, ``errors``, ``warnings`` and ``data``), their text output and their
exit codes.

How to run
----------
No GPU, no internet and no real data are needed: the data is small and synthetic, written to a temporary folder.
The subcommands run in this process (``biapy.main`` with ``sys.argv``), so BiaPy is loaded once for all the tests::

    pytest tests/test_cli.py -q
"""
import json
import sys

import numpy as np
import pytest
import tifffile

from biapy import main
from biapy._biapy import VALID_WORKFLOWS, validate_config

REPORT_KEYS = ["schema_version", "ok", "errors", "warnings", "data"]


###########
# Helpers #
###########
def _run(monkeypatch, capsys, *args):
    """Runs ``biapy <args>`` in this process and returns its exit code, standard output and standard error."""
    monkeypatch.setattr(sys, "argv", ["biapy", *args])
    with pytest.raises(SystemExit) as e:
        main()
    out = capsys.readouterr()
    return e.value.code, out.out, out.err


def _run_json(monkeypatch, capsys, *args):
    """Runs ``biapy <args> --json``, checks that its output has the common keys and returns its exit code and report."""
    code, out, _ = _run(monkeypatch, capsys, *args, "--json")
    report = json.loads(out)  # stdout holds only the JSON
    assert list(report) == REPORT_KEYS
    assert report["ok"] == (code == 0)
    assert report["ok"] or report["errors"]  # a failure always says why
    return code, report


def _semantic_2d(tmp_path, n=6, missing_masks=0):
    """``n`` grayscale images and their masks with labels 0, 1 and 2, minus the last ``missing_masks`` masks."""
    rng = np.random.default_rng(0)
    x, y = tmp_path / "x", tmp_path / "y"
    x.mkdir()
    y.mkdir()
    for i in range(n):
        tifffile.imwrite(str(x / f"{i}.tif"), rng.integers(0, 255, (128, 128), dtype=np.uint8))
        if i < n - missing_masks:
            mask = np.zeros((128, 128), np.uint8)
            mask[20:50, 20:50] = 1
            mask[70:110, 60:100] = 2
            tifffile.imwrite(str(y / f"{i}.tif"), mask)
    return str(x), str(y)


#########################
# Common to all of them #
#########################
@pytest.mark.parametrize(
    "args",
    [
        ["validate-config"],
        ["check-data"],
        ["inspect-image"],
        ["get-config-template"],
        ["create-config"],
        ["inspect-results"],
        ["plan"],
    ],
)
def test_wrong_arguments(monkeypatch, capsys, args):
    """Missing required arguments: argparse's exit code 2."""
    code, _, _ = _run(monkeypatch, capsys, *args)
    assert code == 2


def test_text_errors_go_to_stderr(monkeypatch, capsys, tmp_path):
    code, out, err = _run(monkeypatch, capsys, "inspect-image", str(tmp_path / "missing.tif"))
    assert code == 1
    assert out == ""
    assert "ERROR:" in err


##################
# list-workflows #
##################
def test_list_workflows(monkeypatch, capsys):
    code, report = _run_json(monkeypatch, capsys, "list-workflows")
    assert code == 0
    assert report["data"]["workflows"] == VALID_WORKFLOWS

    code, out, _ = _run(monkeypatch, capsys, "list-workflows")
    assert code == 0
    assert out.split() == VALID_WORKFLOWS


#################
# inspect-image #
#################
@pytest.mark.parametrize(
    "shape,metadata,suggested,axes_known",
    [
        ((40, 50), None, "2D", True),  # YX
        ((6, 40, 50), {"axes": "ZYX"}, "3D", True),  # ImageJ stack with its axes
        ((40, 50, 3), None, "2D", True),  # RGB (YXS)
        ((6, 40, 50), "none", None, False),  # 3D stack saved without axes (QYX/IYX): both readings are possible
    ],
)
def test_inspect_image(monkeypatch, capsys, tmp_path, shape, metadata, suggested, axes_known):
    path = str(tmp_path / "image.tif")
    if metadata == "none":
        tifffile.imwrite(path, np.zeros(shape, np.uint8), metadata=None)
    elif metadata:
        tifffile.imwrite(path, np.zeros(shape, np.uint8), imagej=True, metadata=metadata)
    else:
        tifffile.imwrite(path, np.zeros(shape, np.uint8))

    code, report = _run_json(monkeypatch, capsys, "inspect-image", path)
    assert code == 0
    data = report["data"]
    assert data["suggested_ndim"] == suggested
    assert data["axes_known"] == axes_known
    assert tuple(data["stored"]["shape"]) == shape


def test_inspect_missing_image(monkeypatch, capsys, tmp_path):
    code, report = _run_json(monkeypatch, capsys, "inspect-image", str(tmp_path / "missing.tif"))
    assert code == 1
    assert set(report["data"]) == {"stored", "2D", "3D", "axes_known", "suggested_ndim"}  # same keys when it fails


#######################
# get-config-template #
#######################
@pytest.mark.parametrize("ndim", ["2D", "3D"])
@pytest.mark.parametrize("workflow", VALID_WORKFLOWS)
def test_get_config_template(monkeypatch, capsys, workflow, ndim):
    code, report = _run_json(monkeypatch, capsys, "get-config-template", "--workflow", workflow, "--ndim", ndim)
    assert code == 0
    assert report["data"]["config"]["PROBLEM"]["TYPE"] == workflow
    assert report["data"]["config"]["PROBLEM"]["NDIM"] == ndim


def test_get_config_template_store(monkeypatch, capsys, tmp_path):
    path = tmp_path / "config.yaml"
    code, report = _run_json(
        monkeypatch, capsys, "get-config-template", "--workflow", "SEMANTIC_SEG", "--ndim", "2D", "--store", str(path)
    )
    assert code == 0
    assert report["data"]["stored_in"] == str(path)
    assert "SEMANTIC_SEG" in path.read_text()


def test_get_config_template_wrong_workflow(monkeypatch, capsys):
    code, report = _run_json(monkeypatch, capsys, "get-config-template", "--workflow", "FOO", "--ndim", "2D")
    assert code == 1
    assert report["data"]["config"] is None


###################
# validate-config #
###################
def test_validate_config(monkeypatch, capsys, tmp_path):
    path = str(tmp_path / "config.yaml")
    _run(monkeypatch, capsys, "get-config-template", "--workflow", "SEMANTIC_SEG", "--ndim", "2D", "--store", path)

    code, report = _run_json(monkeypatch, capsys, "validate-config", path, "--no-data-check")
    assert code == 0
    assert report["data"] == {"workflow": "SEMANTIC_SEG", "ndim": "2D"}

    # The template's data paths (/path/to/data) don't exist
    code, report = _run_json(monkeypatch, capsys, "validate-config", path)
    assert code == 1


def test_validate_missing_config(monkeypatch, capsys, tmp_path):
    code, report = _run_json(monkeypatch, capsys, "validate-config", str(tmp_path / "missing.yaml"))
    assert code == 1


##############
# check-data #
##############
def test_check_data(monkeypatch, capsys, tmp_path):
    x, y = _semantic_2d(tmp_path)
    code, report = _run_json(
        monkeypatch, capsys, "check-data", "--workflow", "SEMANTIC_SEG", "--ndim", "2D", "--raw", x, "--gt", y
    )
    assert code == 0
    folders = report["data"]["folders"]
    assert folders["DATA.TRAIN.PATH"]["n_files"] == 6
    assert folders["DATA.TRAIN.PATH"]["channels"] == 1
    assert folders["DATA.TRAIN.GT_PATH"]["classes"] == 3


def test_check_data_missing_mask(monkeypatch, capsys, tmp_path):
    x, y = _semantic_2d(tmp_path, missing_masks=1)
    code, report = _run_json(
        monkeypatch, capsys, "check-data", "--workflow", "SEMANTIC_SEG", "--ndim", "2D", "--raw", x, "--gt", y
    )
    assert code == 1


def test_check_data_config_and_folders_together(monkeypatch, capsys, tmp_path):
    """A configuration or the folders of a workflow, not both."""
    code, _, _ = _run(monkeypatch, capsys, "check-data", str(tmp_path / "c.yaml"), "--workflow", "SEMANTIC_SEG")
    assert code == 2


#################
# create-config #
#################
def test_create_config(monkeypatch, capsys, tmp_path):
    x, y = _semantic_2d(tmp_path)
    path = tmp_path / "config.yaml"
    code, report = _run_json(
        monkeypatch, capsys, "create-config", "--workflow", "SEMANTIC_SEG", "--ndim", "2D",
        "--train-raw", x, "--train-gt", y, "--quick", "--store", str(path),
    )
    assert code == 0
    data = report["data"]
    assert data["config"]["PROBLEM"]["TYPE"] == "SEMANTIC_SEG"
    assert data["config"]["DATA"]["N_CLASSES"] == 3
    assert data["summary"]  # what the wizard decided
    # The configuration written is valid for BiaPy, with the data paths that exist
    cfg = validate_config(str(path), check_data_paths=True)
    assert cfg.PROBLEM.TYPE == "SEMANTIC_SEG"


def test_create_config_data_error(monkeypatch, capsys, tmp_path):
    x, y = _semantic_2d(tmp_path, missing_masks=1)
    code, report = _run_json(
        monkeypatch, capsys, "create-config", "--workflow", "SEMANTIC_SEG", "--ndim", "2D",
        "--train-raw", x, "--train-gt", y,
    )
    assert code == 1
    assert report["data"]["config"] is None  # no configuration for data that can't be used


#####################
# search-bmz-models #
#####################
def test_search_bmz_models_wrong_ndim(monkeypatch, capsys):
    """Checked before searching, so no internet is needed (a real search is not tested: it needs internet)."""
    code, report = _run_json(monkeypatch, capsys, "search-bmz-models", "--ndim", "4D")
    assert code == 1
    assert report["data"]["models"] == []


###################
# inspect-results #
###################
def _job(tmp_path, run_id=1, date="2026_01_01_10_00_00", finished=True, train=True, test=True, first_epoch=0):
    """
    A job's folder as BiaPy leaves it: one run of ``my_job`` with its log, configuration and outputs. It trains 3 of
    its 5 epochs, the best one is the second, and it stops by early stopping, unless it resumes the training from
    ``first_epoch``.
    """
    job, name = tmp_path / "my_job", f"my_job_{run_id}"
    logs, results = job / "logs", job / "results" / name
    for folder in (logs, job / "config_files", job / "checkpoints", results / "per_image"):
        folder.mkdir(parents=True, exist_ok=True)
    log = "[10:00:00.000000] ~~~ Epoch 1/5 ~~~\n"
    if train and not first_epoch:
        log += "[10:00:01.000000] Early stopping\n"
    if test:
        log += "[10:00:02.000000] #  RESULTS  #\n[10:00:02.000000] Test Foreground IoU (merge patches): 0.6\n"
    if finished:
        log += f"[10:00:02.000000] FINISHED JOB {name} !!\n"
    (logs / f"{name}_log_{run_id}_{date}.txt").write_text(log)
    config = "PROBLEM:\n  TYPE: SEMANTIC_SEG\n  NDIM: 2D\nTRAIN:\n  EPOCHS: 5\n"
    (job / "config_files" / f"my_config_{run_id}_{date}.yaml").write_text(config)
    if train:
        losses = [0.6, 0.4, 0.5]
        epochs = [{"train_loss": 0.5, "test_loss": loss, "epoch": first_epoch + e} for e, loss in enumerate(losses)]
        (logs / f"{name}_log_{date}_training_log.json").write_text("\n".join(json.dumps(e) for e in epochs) + "\n")
        (job / "checkpoints" / f"{name}-checkpoint-best.pth").write_bytes(b"")
    if test:
        (results / "test_results_metrics.csv").write_text("file,iou\na.tif,0.5\nb.tif,0.7\n")
        for f in ("a.tif", "b.tif"):
            (results / "per_image" / f).write_bytes(b"")
    return str(job)


def test_inspect_results(monkeypatch, capsys, tmp_path):
    job = _job(tmp_path)
    code, report = _run_json(monkeypatch, capsys, "inspect-results", job)
    assert code == 0
    data = report["data"]
    assert data["job"] == "my_job_1"
    assert data["finished"]
    assert data["config"]["workflow"] == "SEMANTIC_SEG"
    assert data["training"]["epochs"] == 3
    assert data["training"]["planned_epochs"] == 5
    assert data["training"]["stopped_early"]
    assert data["training"]["last"]["epoch"] == 2  # counted from 0, as in BiaPy's training log
    assert data["training"]["best"]["epoch"] == 1  # the lowest validation loss
    assert data["test"]["metrics"] == {"Foreground IoU (merge patches)": 0.6}
    assert data["test"]["metrics_file"].endswith("test_results_metrics.csv")
    assert data["checkpoints"]["best"].endswith("my_job_1-checkpoint-best.pth")
    assert data["predictions"]["per_image"]["n_files"] == 2

    code, out, _ = _run(monkeypatch, capsys, "inspect-results", job)
    assert code == 0
    assert "best:      epoch 2:" in out  # counted from 1, as BiaPy shows them
    assert "Foreground IoU (merge patches)=0.6" in out


def test_inspect_results_resumed(monkeypatch, capsys, tmp_path):
    """Training resumed from the checkpoint of epoch 2: its log has epochs 2-4, and it didn't stop early."""
    code, report = _run_json(monkeypatch, capsys, "inspect-results", _job(tmp_path, first_epoch=2))
    training = report["data"]["training"]
    assert training["epochs"] == 5
    assert training["stopped_early"] is False


def test_inspect_results_test_metrics(monkeypatch, capsys, tmp_path):
    """The test metrics BiaPy printed after "RESULTS" in the last execution, named as it prints them."""
    job = tmp_path / "my_job"
    _job(tmp_path)
    (job / "logs" / "my_job_1_log_1_2026_01_01_10_00_00.txt").write_text(
        "[10:00:00.000000] Test IoU (merge patches): 0.1\n"  # before RESULTS: not a result
        "[10:00:01.000000] #  RESULTS  #\n"
        "[10:00:01.000000] Train loss: 0.2\n"
        "[10:00:01.000000] Test Foreground IoU (per patch): 0.5\n"
        "[10:00:01.000000] Test Foreground IoU (merge patches): 0.6\n"
        "[10:00:01.000000] Test Foreground IoU (merge patches - post-processing): 0.65\n"
        "[10:00:01.000000] Test accuracy: 0.9\n"  # classification: without how it was computed
        "[10:00:01.000000] Test Confusion matrix: [[1 0]\n"  # not a number
        "[10:00:01.000000] Test Dice (merge patches): nan\n"
        "[10:00:02.000000] FINISHED JOB my_job_1 !!\n"
    )
    code, out, _ = _run(monkeypatch, capsys, "inspect-results", str(job), "--json")
    assert "NaN" not in out
    assert json.loads(out)["data"]["test"]["metrics"] == {
        "Foreground IoU (per patch)": 0.5,
        "Foreground IoU (merge patches)": 0.6,
        "Foreground IoU (merge patches - post-processing)": 0.65,
        "accuracy": 0.9,
        "Dice (merge patches)": None,
    }


def test_inspect_results_classification_predictions(monkeypatch, capsys, tmp_path):
    """Classification writes its predictions to predictions.csv, not to a folder."""
    job = tmp_path / "my_job"
    _job(tmp_path, train=False, test=False)
    (job / "results" / "my_job_1" / "predictions.csv").write_text("filename,class\na.tif,0\nb.tif,1\nc.tif,1\n")
    code, report = _run_json(monkeypatch, capsys, "inspect-results", str(job))
    assert report["data"]["predictions"]["predictions.csv"]["n_files"] == 3


def test_inspect_results_nan(monkeypatch, capsys, tmp_path):
    """A training that diverged (NaN loss): valid JSON, without NaN."""
    job = tmp_path / "my_job"
    _job(tmp_path)
    nan = float("nan")
    epochs = [{"train_loss": 0.5, "test_loss": 0.4, "epoch": 0}, {"train_loss": nan, "test_loss": nan, "epoch": 1}]
    (job / "logs" / "my_job_1_log_2026_01_01_10_00_00_training_log.json").write_text(
        "\n".join(json.dumps(e) for e in epochs) + "\n"  # json.dumps writes NaN, as BiaPy does
    )

    code, out, _ = _run(monkeypatch, capsys, "inspect-results", str(job), "--json")
    assert code == 0
    assert "NaN" not in out
    data = json.loads(out)["data"]
    assert data["training"]["last"]["test_loss"] is None
    assert data["training"]["best"]["epoch"] == 0  # BiaPy never saves the best checkpoint with a NaN loss


def test_inspect_results_not_finished(monkeypatch, capsys, tmp_path):
    """A run that stopped (or is still running), before testing: described, but not finished."""
    code, report = _run_json(monkeypatch, capsys, "inspect-results", _job(tmp_path, finished=False, test=False))
    assert code == 0
    assert not report["data"]["finished"]
    assert report["data"]["training"]["stopped_early"] is None  # it may still be training
    assert report["data"]["test"] is None


def test_inspect_results_runs(monkeypatch, capsys, tmp_path):
    """By default, the run done last; another one with --run-id."""
    _job(tmp_path, run_id=1, date="2026_01_01_10_00_00")
    job = _job(tmp_path, run_id=2, date="2026_01_02_10_00_00", train=False)
    code, report = _run_json(monkeypatch, capsys, "inspect-results", job)
    assert report["data"]["job"] == "my_job_2"
    assert report["data"]["training"] is None

    code, report = _run_json(monkeypatch, capsys, "inspect-results", job, "--run-id", "1")
    assert report["data"]["job"] == "my_job_1"
    assert report["data"]["training"]["epochs"] == 3

    code, report = _run_json(monkeypatch, capsys, "inspect-results", job, "--run-id", "3")
    assert code == 1


def test_inspect_results_no_job(monkeypatch, capsys, tmp_path):
    code, report = _run_json(monkeypatch, capsys, "inspect-results", str(tmp_path / "missing"))
    assert code == 1
    code, report = _run_json(monkeypatch, capsys, "inspect-results", str(tmp_path))  # a folder, but not a job's
    assert code == 1


########
# plan #
########
def _plan_config(monkeypatch, capsys, tmp_path, missing_masks=0):
    """The configuration create-config makes for ``_semantic_2d`` data, with a test set."""
    x, y = _semantic_2d(tmp_path, missing_masks=missing_masks)
    path = str(tmp_path / "config.yaml")
    _run(
        monkeypatch, capsys, "create-config", "--workflow", "SEMANTIC_SEG", "--ndim", "2D", "--train-raw", x,
        "--train-gt", y, "--test-raw", x, "--test-gt", y, "--quick", "--store", path,
    )
    return path


def test_plan(monkeypatch, capsys, tmp_path):
    config = _plan_config(monkeypatch, capsys, tmp_path)
    result_dir = tmp_path / "output"
    code, report = _run_json(monkeypatch, capsys, "plan", config, "--result_dir", str(result_dir), "--name", "my_job")
    assert code == 0
    data = report["data"]
    assert data["workflow"] == "SEMANTIC_SEG"
    assert data["phases"] == {"train": True, "test": True}
    assert data["job"]["job"] == "my_job_1"
    assert not data["job"]["exists"]
    assert data["data"]["DATA.TRAIN.PATH"]["n_files"] == 6
    assert data["data"]["DATA.TRAIN.PATH"]["max_shape"] == [128, 128]
    assert data["data"]["DATA.TRAIN.GT_PATH"]["classes"] == 3
    assert data["model"]["source"] == "biapy"
    assert data["training"]["epochs"] > 0
    assert data["hardware"]["ram_bytes"] > 0
    assert data["estimates"]["ram_train"] > 0
    assert not result_dir.exists()  # nothing written

    code, out, _ = _run(monkeypatch, capsys, "plan", config, "--result_dir", str(result_dir), "--name", "my_job")
    assert code == 0
    assert "Job:         my_job_1 (SEMANTIC_SEG 2D): train and test" in out


def test_plan_without_result_dir(monkeypatch, capsys, tmp_path):
    """Only the configuration: everything but the folders the job would write."""
    config = _plan_config(monkeypatch, capsys, tmp_path)
    code, report = _run_json(monkeypatch, capsys, "plan", config)
    assert code == 0
    assert report["data"]["job"] is None
    assert report["data"]["data"]["DATA.TRAIN.PATH"]["n_files"] == 6


def test_plan_existing_run(monkeypatch, capsys, tmp_path):
    """The run already exists: it can be run, but it would be written over."""
    config = _plan_config(monkeypatch, capsys, tmp_path)
    (tmp_path / "output" / "my_job" / "results" / "my_job_1").mkdir(parents=True)
    code, report = _run_json(
        monkeypatch, capsys, "plan", config, "--result_dir", str(tmp_path / "output"), "--name", "my_job"
    )
    assert code == 0
    assert report["data"]["job"]["exists"]
    assert any("would write over" in w for w in report["warnings"])


def test_plan_data_error(monkeypatch, capsys, tmp_path):
    """The data can't be used: the plan says why, and the job can't be run."""
    config = _plan_config(monkeypatch, capsys, tmp_path)
    (tmp_path / "y" / "5.tif").unlink()  # a mask missing
    code, report = _run_json(monkeypatch, capsys, "plan", config, "--result_dir", str(tmp_path / "output"))
    assert code == 1
    assert report["data"]["workflow"] == "SEMANTIC_SEG"  # the rest of the plan is still given


def test_plan_missing_config(monkeypatch, capsys, tmp_path):
    code, report = _run_json(
        monkeypatch, capsys, "plan", str(tmp_path / "missing.yaml"), "--result_dir", str(tmp_path / "output")
    )
    assert code == 1
    assert report["data"]["job"] is None
