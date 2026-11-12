"""
Tests of the subcommands of the ``biapy`` command (``validate-config``, ``check-data``, ``list-workflows``,
``inspect-image``, ``get-config-template``, ``create-config`` and ``search-bmz-models``): their ``--json`` output
(``schema_version``, ``ok``, ``errors``, ``warnings`` and ``data``), their text output and their exit codes.

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
