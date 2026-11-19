"""
BiaPy: Accessible Deep Learning on Bioimages.

BiaPy is an open-source library and application that streamlines the use of
common deep-learning workflows for a large variety of bioimage analysis tasks,
including 2D and 3D semantic segmentation, instance segmentation, object detection,
image denoising, single image super-resolution, self-supervised learning, and
image classification.

This `__init__.py` file serves as the entry point for the BiaPy package,
exposing the `BiaPy` class and defining the `main` function for command-line
execution.

Version: 3.7.3
"""

__version__ = "3.7.3"

import argparse
import contextlib
import json
import os
import sys

import yaml

# The engine (BiaPy, build_config, validate_config) is imported on first use: importing it loads PyTorch and every workflow,
# which takes several seconds, and lightweight parts such as biapy.wizard don't need it.
_LAZY = {"BiaPy": "._biapy", "build_config": "._biapy", "validate_config": "._biapy", "check_data": "._biapy", "check_data_for_workflow": "._biapy", "list_workflows": "._biapy", "inspect_image": "._biapy", "get_config_template": "._biapy", "create_config": "._biapy", "search_bmz_models": "._biapy", "inspect_results": "._biapy", "plan_job": "._biapy", "run_job": "._biapy", "job_status": "._biapy", "stop_job": "._biapy"}
__all__ = ["BiaPy", "build_config", "validate_config", "check_data", "check_data_for_workflow", "list_workflows", "inspect_image", "get_config_template", "create_config", "search_bmz_models", "inspect_results", "plan_job", "run_job", "job_status", "stop_job", "main", "__version__"]


def __getattr__(name):
    if name in _LAZY:
        import importlib

        value = getattr(importlib.import_module(_LAZY[name], __name__), name)
        globals()[name] = value  # later accesses don't go through __getattr__
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_LAZY))


def _command_parser(command, description):
    """Parser of a subcommand (``biapy <command>``), with the ``--json`` option all of them have."""
    parser = argparse.ArgumentParser(prog=f"biapy {command}", description=description)
    parser.add_argument("--json", action="store_true", help="Print the result as JSON, for agents and scripts")
    return parser


# Version of the JSON the subcommands print with --json: increase it when a field is removed, renamed or changes meaning
JSON_SCHEMA_VERSION = "1.0"


def _report(ok=True, errors=None, warnings=None, **data):
    """
    Result of a subcommand: ``schema_version``, ``ok``, ``errors`` and ``warnings``, and the subcommand's own data in
    ``data``.
    """
    return {
        "schema_version": JSON_SCHEMA_VERSION,
        "ok": ok,
        "errors": list(errors or []),
        "warnings": list(warnings or []),
        "data": data,
    }


def _from_report(report):
    """
    The report of a Python function (``ok``, ``errors``, ``warnings`` and its data, all at the same level, as
    :func:`biapy.check_data` returns) as the result of a subcommand.
    """
    data = {k: v for k, v in report.items() if k not in ("ok", "errors", "warnings")}
    return _report(report["ok"], report["errors"], report["warnings"], **data)


def _finish(report, as_json, print_text):
    """
    Print the result of a subcommand and exit.

    With ``as_json``, the whole ``report`` is printed as JSON. Otherwise its warnings and errors go to stderr and, if
    it went well, ``print_text(report["data"])`` prints the rest for people. The exit code is 0 if ``report["ok"]``,
    1 if not.
    """
    if as_json:
        # The wizard imports numpy: imported here so that 'import biapy' stays light
        from biapy.wizard.config_builder import to_jsonable

        print(json.dumps(to_jsonable(report), indent=2))
    else:
        for warning in report["warnings"]:
            print(f"WARNING: {warning}", file=sys.stderr)
        for error in report["errors"]:
            print(f"ERROR: {error}", file=sys.stderr)
        if report["ok"]:
            print_text(report["data"])
    sys.exit(0 if report["ok"] else 1)


def _validate_config_cli(argv):
    """
    Validate a configuration file from the command line (see :func:`biapy.validate_config`).

    Nothing is run: no data is loaded, no model is built and nothing is written to disk.

    Usage::

        biapy validate-config <path_to_config_file> [--no-data-check] [--json]

    ``--no-data-check`` skips the check that the data paths exist, so a configuration can be validated before the
    data is available.

    Parameters
    ----------
    argv : list of str
        Command-line arguments after ``validate-config``.

    Exit codes
    ----------
    0
        The configuration is valid.
    1
        The configuration is not valid (the error names the wrong key).
    2
        Wrong command-line arguments (raised by ``argparse``).
    """
    parser = _command_parser("validate-config", "Check a BiaPy YAML configuration without running anything.")
    parser.add_argument("config", help="Path to the YAML configuration file")
    parser.add_argument("--no-data-check", action="store_true", help="Skip the check that the data paths exist")
    args = parser.parse_args(argv)

    from ._biapy import validate_config

    try:
        # BiaPy prints notes while checking: keep stdout for the result only
        with contextlib.redirect_stdout(sys.stderr):
            cfg = validate_config(args.config, check_data_paths=not args.no_data_check)
        report = _report(workflow=cfg.PROBLEM.TYPE, ndim=cfg.PROBLEM.NDIM)
    except (ValueError, AssertionError, FileNotFoundError, KeyError) as e:
        report = _report(ok=False, errors=[str(e)], workflow=None, ndim=None)

    def print_text(r):
        print(f"OK: {args.config} is a valid {r['workflow']} {r['ndim']} configuration")

    _finish(report, args.json, print_text)


def _check_data_cli(argv):
    """
    Check from the command line that some data can be used by BiaPy, without running anything.

    Usage::

        biapy check-data <path_to_config_file> [--percentage 10] [--json]
        biapy check-data --workflow SEMANTIC_SEG --ndim 2D --raw <folder> [--gt <folder>] [--json]

    With a configuration, the data folders it points to are checked and compared with it
    (see :func:`biapy.check_data`). Without one, the folders given are checked for the workflow, as the
    wizard does (see :func:`biapy.check_data_for_workflow`).

    Parameters
    ----------
    argv : list of str
        Command-line arguments after ``check-data``.

    Exit codes
    ----------
    0
        The data can be used.
    1
        The data or the configuration is not valid.
    2
        Wrong command-line arguments (raised by ``argparse``).
    """
    parser = _command_parser(
        "check-data", "Check that some data can be used by BiaPy: the data of a configuration, or folders for a workflow."
    )
    parser.add_argument("config", nargs="?", help="Path to the YAML configuration file")
    parser.add_argument("--workflow", help="Without a configuration: PROBLEM.TYPE, e.g. SEMANTIC_SEG")
    parser.add_argument("--ndim", help="Without a configuration: 2D or 3D")
    parser.add_argument("--raw", help="Without a configuration: folder with the raw images")
    parser.add_argument("--gt", default="", help="Without a configuration: folder with the targets")
    parser.add_argument(
        "--percentage", type=float, default=100, help="Percentage of each Zarr/H5 image to check (default: 100)"
    )
    args = parser.parse_args(argv)

    by_workflow = args.workflow or args.ndim or args.raw or args.gt
    if args.config and by_workflow:
        parser.error("give either a configuration or --workflow/--ndim/--raw, not both")
    if not args.config and not (args.workflow and args.ndim and args.raw):
        parser.error("give a configuration, or --workflow, --ndim and --raw")

    from ._biapy import check_data, check_data_for_workflow

    try:
        # The checks print their progress: keep stdout for the result only
        with contextlib.redirect_stdout(sys.stderr):
            if args.config:
                report = check_data(args.config, percentage=args.percentage)
            else:
                report = check_data_for_workflow(args.workflow, args.ndim, args.raw, args.gt, args.percentage)
        report = _from_report(report)
    except (ValueError, AssertionError, FileNotFoundError, KeyError) as e:
        report = _report(ok=False, errors=[str(e)], folders={}, upscaling=None)

    def print_text(r):
        for key, found in r["folders"].items():
            print(f"OK: {key}: {found['n_files']} files, channels={found['channels']}, classes={found['classes']}")
        if r["upscaling"]:
            print(f"OK: upscaling {tuple(r['upscaling'])}")

    _finish(report, args.json, print_text)


def _list_workflows_cli(argv):
    """
    List from the command line the workflows available in BiaPy (see :func:`biapy.list_workflows`).

    Usage::

        biapy list-workflows [--json]

    Parameters
    ----------
    argv : list of str
        Command-line arguments after ``list-workflows``.
    """
    parser = _command_parser("list-workflows", "List the workflows available in BiaPy (the values of PROBLEM.TYPE).")
    args = parser.parse_args(argv)

    from ._biapy import list_workflows

    report = _report(workflows=list_workflows())
    _finish(report, args.json, lambda r: print("\n".join(r["workflows"])))


def _inspect_image_cli(argv):
    """
    Describe from the command line an image as BiaPy reads it (see :func:`biapy.inspect_image`).

    Usage::

        biapy inspect-image <path_to_image> [--json]

    Parameters
    ----------
    argv : list of str
        Command-line arguments after ``inspect-image``.

    Exit codes
    ----------
    0
        The image was described.
    1
        The image does not exist or can't be read.
    2
        Wrong command-line arguments (raised by ``argparse``).
    """
    parser = _command_parser("inspect-image", "Describe an image as BiaPy reads it, to know whether it is 2D or 3D.")
    parser.add_argument("path", help="Path to the image")
    args = parser.parse_args(argv)

    from ._biapy import inspect_image

    try:
        report = _report(**inspect_image(args.path))
    except Exception as e:
        empty = {"stored": None, "2D": None, "3D": None, "axes_known": None, "suggested_ndim": None}
        report = _report(ok=False, errors=[str(e)], **empty)

    def print_text(r):
        s = r["stored"]
        print(f"Stored:    shape {s['shape']}, axes {s['axes'] or 'unknown'}, {s['dtype']}")
        for ndim in ("2D", "3D"):
            reading = r[ndim]
            if "error" in reading:
                print(f"As {ndim}:     not possible ({reading['error']})")
                continue
            shape = reading["shape"]
            # BiaPy's order is (y, x, c) in 2D and (z, y, x, c) in 3D
            meaning = f"{shape[-1]} channel" + ("s" if shape[-1] != 1 else "")
            if ndim == "3D":
                meaning = f"{shape[0]} slices, " + meaning
            print(f"As {ndim}:     {str(shape):<20}  -> {meaning}")
        ndim = r["suggested_ndim"]
        if ndim:
            if r["axes_known"]:
                why = "the file has a Z axis" if ndim == "3D" else "the file has no Z axis"
            else:
                why = "it can only be read this way"
            print(f"Suggested: PROBLEM.NDIM: {ndim} ({why})")
        else:
            print("Suggested: unknown (the file doesn't store its axes): choose PROBLEM.NDIM from the readings above")

    _finish(report, args.json, print_text)


def _get_config_template_cli(argv):
    """
    Print or save from the command line the basic configuration template of a workflow (see
    :func:`biapy.get_config_template`).

    Usage::

        biapy get-config-template --workflow SEMANTIC_SEG --ndim 2D [--store <yaml>] [--json]

    Parameters
    ----------
    argv : list of str
        Command-line arguments after ``get-config-template``.

    Exit codes
    ----------
    0
        The template was printed or saved.
    1
        Wrong workflow or dimensions.
    2
        Wrong command-line arguments (raised by ``argparse``).
    """
    parser = _command_parser("get-config-template", "Print or save the basic configuration template of a workflow.")
    parser.add_argument("--workflow", required=True, help="PROBLEM.TYPE, e.g. SEMANTIC_SEG")
    parser.add_argument("--ndim", required=True, help="2D or 3D")
    parser.add_argument("--store", help="YAML file to write the template to")
    args = parser.parse_args(argv)

    from ._biapy import get_config_template

    try:
        report = _report(config=get_config_template(args.workflow, args.ndim, store_yaml=args.store), stored_in=args.store)
    except ValueError as e:
        report = _report(ok=False, errors=[str(e)], config=None, stored_in=None)

    def print_text(r):
        if r["stored_in"]:
            print(f"OK: template written to {r['stored_in']}")
        else:
            print(yaml.safe_dump(r["config"], default_flow_style=False, sort_keys=False), end="")

    _finish(report, args.json, print_text)


def _create_config_cli(argv):
    """
    Create from the command line a configuration adapted to the data, as the wizard does (see
    :func:`biapy.create_config`).

    Usage::

        biapy create-config --workflow SEMANTIC_SEG --ndim 2D --train-raw <folder> [--train-gt <folder>]
                            [--test-raw <folder>] [--test-gt <folder>] [--quick] [--store <yaml>] [--json]

    Without ``--store`` nor ``--json``, the configuration is printed as YAML.

    Parameters
    ----------
    argv : list of str
        Command-line arguments after ``create-config``.

    Exit codes
    ----------
    0
        The configuration was created.
    1
        The data can't be used.
    2
        Wrong command-line arguments (raised by ``argparse``).
    """
    parser = _command_parser(
        "create-config", "Create a configuration adapted to the data, as the wizard does, to train a model from scratch."
    )
    parser.add_argument("--workflow", required=True, help="PROBLEM.TYPE, e.g. SEMANTIC_SEG")
    parser.add_argument("--ndim", required=True, help="2D or 3D")
    parser.add_argument("--train-raw", required=True, help="Folder with the training raw images")
    parser.add_argument("--train-gt", default="", help="Folder with the training targets")
    parser.add_argument("--test-raw", default="", help="Folder with the test raw images")
    parser.add_argument("--test-gt", default="", help="Folder with the test targets")
    parser.add_argument("--quick", action="store_true", help="A quick training, to check that everything works")
    parser.add_argument("--store", help="YAML file to write the configuration to")
    args = parser.parse_args(argv)

    from ._biapy import create_config

    try:
        # The checks print their progress: keep stdout for the result only
        with contextlib.redirect_stdout(sys.stderr):
            report = create_config(
                args.workflow, args.ndim, args.train_raw, args.train_gt, args.test_raw, args.test_gt,
                training_length="quick" if args.quick else "full", store_yaml=args.store,
            )
        report = _from_report(report)
    except ValueError as e:
        report = _report(ok=False, errors=[str(e)], folders={}, config=None, summary=[])

    def print_text(r):
        for line in r["summary"]:
            print(f"# {line}")
        if args.store:
            print(f"OK: configuration written to {args.store}")
        else:
            # The wizard imports numpy: imported here so that 'import biapy' stays light
            from biapy.wizard.config_builder import tuples_to_lists

            print(yaml.safe_dump(tuples_to_lists(r["config"]), default_flow_style=False), end="")

    _finish(report, args.json, print_text)


def _search_bmz_models_cli(argv):
    """
    Search from the command line the BioImage Model Zoo for pretrained models BiaPy can use (see
    :func:`biapy.search_bmz_models`).

    Usage::

        biapy search-bmz-models [--workflow INSTANCE_SEG] [--ndim 2D] [--nchannels 1] [--nclasses 2]
                                [--query nuclei] [--json]

    Parameters
    ----------
    argv : list of str
        Command-line arguments after ``search-bmz-models``.

    Exit codes
    ----------
    0
        The search was done (even if no model was found).
    1
        The search failed (wrong workflow/ndim, no internet...).
    2
        Wrong command-line arguments (raised by ``argparse``).
    """
    parser = _command_parser(
        "search-bmz-models", "Search the BioImage Model Zoo for pretrained models BiaPy can use (it needs internet)."
    )
    parser.add_argument("--workflow", default="all", help="PROBLEM.TYPE, e.g. INSTANCE_SEG (default: all)")
    parser.add_argument("--ndim", default="all", help="2D or 3D (default: all)")
    parser.add_argument("--nchannels", default="all", help="Channels of the images, e.g. 1 (default: all)")
    parser.add_argument("--nclasses", default="all", help="Classes of the data (default: all)")
    parser.add_argument("--query", help="Words to find in the model's name, description or tags, e.g. 'nuclei'")
    args = parser.parse_args(argv)

    from ._biapy import search_bmz_models

    print("Searching the BioImage Model Zoo (it may take a minute)...", file=sys.stderr)
    try:
        report = _report(models=search_bmz_models(args.workflow, args.ndim, args.nclasses, args.nchannels, args.query))
    except Exception as e:
        report = _report(ok=False, errors=[str(e)], models=[])

    def print_text(r):
        if not r["models"]:
            print("No compatible models found")
        for m in r["models"]:
            channels = m["opts"].get("DATA.PATCH_SIZE", ("?",))[-1]
            print(f"{m['id']:24s} {m['workflow_type']} {m['ndim']}, {channels} channel(s)   {m['name'][:50]}   {m['url']}")

    _finish(report, args.json, print_text)


def _inspect_results_cli(argv):
    """
    Describe from the command line what a BiaPy job left in its folder (see :func:`biapy.inspect_results`).

    Usage::

        biapy inspect-results <job_folder> [--run-id 1] [--json]

    Parameters
    ----------
    argv : list of str
        Command-line arguments after ``inspect-results``.

    Exit codes
    ----------
    0
        The job was described (even if it didn't finish: see ``finished``).
    1
        The folder doesn't exist or holds no run.
    2
        Wrong command-line arguments (raised by ``argparse``).
    """
    parser = _command_parser(
        "inspect-results", "Describe what a BiaPy job left in its folder: training, test metrics, checkpoints, outputs."
    )
    parser.add_argument("job_dir", help="Folder of the job: <result_dir>/<name>")
    parser.add_argument("--run-id", type=int, help="Run to describe (default: the one run last)")
    args = parser.parse_args(argv)

    from ._biapy import inspect_results

    try:
        # The analysis of the predictions prints its progress: keep stdout for the result only
        with contextlib.redirect_stdout(sys.stderr):
            result = inspect_results(args.job_dir, args.run_id)
        report = _report(warnings=result.pop("warnings"), **result)
    except FileNotFoundError as e:
        keys = [
            "job", "run_id", "finished", "log", "config", "training", "test", "checkpoints", "predictions",
            "statistics", "charts",
        ]
        report = _report(ok=False, errors=[str(e)], **dict.fromkeys(keys))

    def numbers(values):
        """The numbers in ``values`` (a dict), as 'name=value, ...'."""
        return ", ".join(f"{k}={v:.4g}" for k, v in values.items() if k != "epoch" and isinstance(v, (int, float)))

    def print_text(r):
        config = r["config"] or {}
        print(f"Job:         {r['job']} ({config.get('workflow', '?')} {config.get('ndim', '?')})")
        finished = "yes" if r["finished"] else f"no (it stopped, or it is still running), see {r['log']}"
        print(f"Finished:    {finished}")
        training = r["training"]
        if training:
            stopped = " (stopped early)" if training["stopped_early"] else ""
            print(f"Training:    {training['epochs']} of {training['planned_epochs']} epochs{stopped}")
            for which in ("last", "best"):
                if training[which]:
                    # BiaPy shows the epochs counting from 1 ("Epoch 76/100"), its training log from 0
                    print(f"  {which + ':':<10} epoch {training[which]['epoch'] + 1}: {numbers(training[which])}")
        if r["test"]:
            metrics = r["test"]["metrics"]
            # The matching metrics of instance segmentation, e.g. "f1 (IoU 0.5, merge patches)", by threshold
            matching = {k: v for k, v in metrics.items() if "(IoU " in k}
            others = {k: v for k, v in metrics.items() if k not in matching}
            if others:
                print(f"Test:        {numbers(others)}")
            # The range of each metric over the test images (which image has each value is in the JSON)
            per_image = {k: v for k, v in (r["test"]["per_image"] or {}).items() if " TH " not in k}
            if per_image:
                ranges = ", ".join(f"{k} {v['min']:.4g} to {v['max']:.4g}" for k, v in per_image.items())
                print(f"By image:    {ranges}")
            for group in dict.fromkeys(k[k.index("(IoU "):] for k in matching):
                names = ("precision", "recall", "f1", "panoptic_quality")
                values = {k: matching.get(f"{k} {group}") for k in names}
                print(f"Matching:    {group[1:-1]}: {numbers({k: v for k, v in values.items() if v is not None})}")
        checkpoints = r["checkpoints"]
        if checkpoints["all"]:
            print(f"Checkpoints: {checkpoints['best'] or checkpoints['all'][-1]} ({len(checkpoints['all'])} in total)")
        for name, folder in r["predictions"].items():
            files = "images" if name == "predictions.csv" else "files"  # one row by image
            print(f"Output:      {name}: {folder['n_files']} {files} in {folder['path']}")
        s = r["statistics"]
        if s:
            if "points_per_image" in s:  # detection
                p = s["points_per_image"]
                text = f"{p['total']} points, by image median {p['median']:g} (max {p['max']})"
            elif "true_classes" in s:  # classification
                true = s["true_classes"] or {}
                text = "predicted as " + ", ".join(
                    f"class {c}: {s['classes'].get(c, 0)}" + (f" (true: {true.get(c, 0)})" if true else "")
                    for c in sorted(set(s["classes"]) | set(true), key=lambda c: (len(c), c))
                )
            elif "intensity" in s:  # predicted images, by channel
                channels = [c for c in s["intensity"] if c["mean"] is not None]
                text = "intensity mean " + ", ".join(
                    f"{c['mean']:.4g} (min {c['min']:.4g}, max {c['max']:.4g})" for c in channels
                ) + (" by channel" if len(channels) > 1 else "")
            else:  # semantic and instance segmentation
                f = s["foreground_fraction"]
                text = f"foreground {f['mean']:.3g} of the image (min {f['min']:.3g}, max {f['max']:.3g})"
                if "n_instances" in s:
                    i = s["instances_per_image"]
                    text = f"{s['n_instances']} objects, by image median {i['median']:g} (max {i['max']}), " + text
            shape = f", shape {'x'.join(str(n) for n in s['median_shape'])}" if s.get("median_shape") else ""
            print(f"Statistics:  {text}{shape}")

    _finish(report, args.json, print_text)


def _plan_cli(argv):
    """
    Describe from the command line what running a configuration would do, without running anything (see
    :func:`biapy.plan_job`).

    Usage::

        biapy plan <path_to_config_file> [--result_dir <folder> [--name my_job] [--run_id 1]] [--percentage 10] [--json]

    ``--result_dir``, ``--name`` and ``--run_id`` are the ones the job would be run with (``biapy --config ...``): with
    them, the plan also says where the job would write and whether that run already exists.

    Parameters
    ----------
    argv : list of str
        Command-line arguments after ``plan``.

    Exit codes
    ----------
    0
        The job can be run (its configuration is valid and its data can be used).
    1
        The configuration is not valid or its data can't be used.
    2
        Wrong command-line arguments (raised by ``argparse``).
    """
    parser = _command_parser(
        "plan", "Describe what running a configuration would do (data, model, outputs, resources), without running it."
    )
    parser.add_argument("config", help="Path to the YAML configuration file")
    parser.add_argument(
        "--result_dir", help="Folder where the job would write, to check it (whether the run already exists)"
    )
    parser.add_argument("--name", default="unknown_job", help="Name of the job (default: unknown_job)")
    parser.add_argument("--run_id", type=int, default=1, help="Run of the job (default: 1)")
    parser.add_argument(
        "--percentage", type=float, default=100, help="Percentage of each Zarr/H5 image to check (default: 100)"
    )
    args = parser.parse_args(argv)

    from ._biapy import plan_job

    try:
        # The checks print their progress: keep stdout for the result only
        with contextlib.redirect_stdout(sys.stderr):
            report = plan_job(args.config, args.result_dir, args.name, args.run_id, args.percentage)
        report = _from_report(report)
    except (ValueError, AssertionError, FileNotFoundError, KeyError) as e:
        keys = ["workflow", "ndim", "phases", "job", "data", "upscaling", "model", "training", "hardware", "estimates"]
        report = _report(ok=False, errors=[str(e)], **dict.fromkeys(keys))

    def gb(value):
        return f"{value / 1024**3:.1f} GB" if value is not None else "not estimated"

    def listed(value):
        """A value that may be a list (e.g. TRAIN.OPTIMIZER, one per optimizer), without brackets."""
        return ", ".join(str(v) for v in value) if isinstance(value, (list, tuple)) else str(value)

    def print_text(r):
        phases = " and ".join(p for p in ("train", "test") if r["phases"][p]) or "nothing"
        job = r["job"]
        print(f"Job:         {job['job'] + ' ' if job else ''}({r['workflow']} {r['ndim']}): {phases}")
        if job:
            exists = " (the run exists: it would be written over)" if job["exists"] else ""
            print(f"Writes to:   {job['job_dir']}{exists}")
        else:
            print("Writes to:   not checked (give --result_dir and --name to check it)")
        for key, found in r["data"].items():
            shape = f", {found['max_shape']}" if found["max_shape"] else ""
            classes = f", {found['classes']} classes" if found["classes"] else ""
            channels = f", {found['channels']} channel(s)" if found["channels"] else ""
            print(f"Data:        {key}: {found['n_files']} files{channels}{classes}{shape}")
        model = r["model"]
        checkpoint = f", loading {model['checkpoint']}" if model["checkpoint"] else ""
        print(f"Model:       {model['architecture']} ({model['source']}{checkpoint})")
        t = r["training"]
        if t:
            print(f"Training:    {t['epochs']} epochs, batch size {t['batch_size']}, {listed(t['optimizer'])} "
                  f"(lr {listed(t['lr'])}), patch size {t['patch_size']}")
        # The same GPUs together, e.g. "7 x NVIDIA L4 (23034 MB)"
        gpus = [f"{g['name']} ({g['memory_mb']} MB)" for g in r["hardware"]["gpus"]]
        gpus = ", ".join(f"{gpus.count(g)} x {g}" for g in dict.fromkeys(gpus)) or "no NVIDIA GPU"
        print(f"Hardware:    {gpus}; RAM {gb(r['hardware']['ram_bytes'])}")
        e = r["estimates"] or {}
        print(f"Estimates:   RAM {gb(e.get('ram_train'))} to train, {gb(e.get('ram_test'))} to test; "
              f"disk {gb(e.get('disk_total'))}; GPU memory not estimated")

    _finish(report, args.json, print_text)


def _run_cli(argv):
    """
    Run a job in the background from the command line, after checking it as ``biapy plan`` does (see
    :func:`biapy.run_job`).

    Usage::

        biapy run <path_to_config_file> --result_dir <folder> [--name my_job] [--run_id 1] [--gpu 0] [--overwrite]
            [--json]

    Parameters
    ----------
    argv : list of str
        Command-line arguments after ``run``.

    Exit codes
    ----------
    0
        The job was started.
    1
        It was not started: the configuration is not valid, its data can't be used, or the run is running or exists.
    2
        Wrong command-line arguments (raised by ``argparse``).
    """
    parser = _command_parser(
        "run",
        "Run a job in the background and return at once, after checking it as 'biapy plan' does. Follow it with "
        "'biapy job-status' and stop it with 'biapy stop-job'.",
    )
    parser.add_argument("config", help="Path to the YAML configuration file")
    parser.add_argument("--result_dir", required=True, help="Folder where the job writes")
    parser.add_argument("--name", default="unknown_job", help="Name of the job (default: unknown_job)")
    parser.add_argument("--run_id", type=int, default=1, help="Run of the job (default: 1)")
    parser.add_argument("--gpu", help="GPU(s) to use, e.g. 0, or 0,1 to use two (default: the CPU)")
    parser.add_argument("--overwrite", action="store_true", help="Run it even if the run exists, writing over it")
    args = parser.parse_args(argv)

    from ._biapy import run_job

    try:
        # The checks print their progress: keep stdout for the result only
        with contextlib.redirect_stdout(sys.stderr):
            report = run_job(args.config, args.result_dir, args.name, args.run_id, args.gpu, args.overwrite)
        report = _from_report(report)
    except (ValueError, AssertionError, FileNotFoundError, KeyError) as e:
        keys = ["job_dir", "job", "run_id", "pid", "command", "output"]
        report = _report(ok=False, errors=[str(e)], **dict.fromkeys(keys))

    def print_text(r):
        print(f"Started:     {r['job']} (process {r['pid']})")
        print(f"Output:      {r['output']}")
        print(f"Follow it:   biapy job-status {r['job_dir']} --run-id {r['run_id']}")
        print(f"Stop it:     biapy stop-job {r['job_dir']} --run-id {r['run_id']}")

    _finish(report, args.json, print_text)


def _job_status_cli(argv):
    """
    Say from the command line whether a job is running, where it is and, if it failed, why (see
    :func:`biapy.job_status`).

    Usage::

        biapy job-status <job_folder> [--run-id 1] [--json]

    Parameters
    ----------
    argv : list of str
        Command-line arguments after ``job-status``.

    Exit codes
    ----------
    0
        The state was given (even if the job failed: see ``state``).
    1
        The folder doesn't exist or holds no run.
    2
        Wrong command-line arguments (raised by ``argparse``).
    """
    parser = _command_parser("job-status", "Say whether a job is running, where it is and, if it failed, why.")
    parser.add_argument("job_dir", help="Folder of the job: <result_dir>/<name>")
    parser.add_argument("--run-id", type=int, help="Run to describe (default: the one started last)")
    args = parser.parse_args(argv)

    from ._biapy import job_status

    try:
        report = _report(**job_status(args.job_dir, args.run_id))
    except FileNotFoundError as e:
        keys = ["job", "run_id", "state", "pid", "progress", "resources", "error", "output"]
        report = _report(ok=False, errors=[str(e)], **dict.fromkeys(keys))

    def print_text(r):
        running = f" (process {r['pid']})" if r["pid"] else ""
        print(f"Job:         {r['job']}: {r['state']}{running}")
        p = r["progress"]
        if p["epoch"]:
            times = f" ({p['elapsed']} of an estimated {p['estimated_total']})" if p["elapsed"] else ""
            print(f"Training:    epoch {p['epoch']} of {p['epochs']}{times}")
        if p["last"]:
            values = ", ".join(f"{k}={v:.4g}" for k, v in p["last"].items() if k != "epoch" and isinstance(v, float))
            print(f"Last epoch:  {values}")
        if p["test_images"] is not None:
            print(f"Test:        {p['test_images']} images processed")
        res = r["resources"]
        if res:
            gpu = f", GPU memory {res['gpu_memory_mb']} MB" if res["gpu_memory_mb"] else ""
            print(f"Resources:   CPU {res['cpu_percent']}%, RAM {res['ram_bytes'] / 1024**3:.1f} GB{gpu}")
        if r["error"]:
            print("Error:\n" + r["error"])
        print(f"Output:      {r['output']}")

    _finish(report, args.json, print_text)


def _stop_job_cli(argv):
    """
    Stop from the command line a job started with ``biapy run`` (see :func:`biapy.stop_job`).

    Usage::

        biapy stop-job <job_folder> [--run-id 1] [--timeout 30] [--json]

    Parameters
    ----------
    argv : list of str
        Command-line arguments after ``stop-job``.

    Exit codes
    ----------
    0
        The job was stopped, or it was not running.
    1
        No job started with ``biapy run`` was found.
    2
        Wrong command-line arguments (raised by ``argparse``).
    """
    parser = _command_parser("stop-job", "Stop a job started with 'biapy run'.")
    parser.add_argument("job_dir", help="Folder of the job: <result_dir>/<name>")
    parser.add_argument("--run-id", type=int, help="Run to stop (default: the one started last)")
    parser.add_argument(
        "--timeout", type=float, default=30, help="Seconds to wait for it to end before killing it (default: 30)"
    )
    args = parser.parse_args(argv)

    from ._biapy import stop_job

    try:
        result = stop_job(args.job_dir, args.run_id, args.timeout)
        warnings = [] if result["was_running"] else [f"The run {result['job']} was not running"]
        report = _report(warnings=warnings, **result)
    except FileNotFoundError as e:
        report = _report(ok=False, errors=[str(e)], **dict.fromkeys(["job", "run_id", "was_running", "killed"]))

    def print_text(r):
        if r["was_running"]:
            print(f"Stopped:     {r['job']}" + (" (killed: it didn't end in time)" if r["killed"] else ""))

    _finish(report, args.json, print_text)


def main():
    r"""
    Run BiaPy.

    Main entry point for the BiaPy command-line interface.

    This function parses command-line arguments, initializes the `BiaPy`
    application with the provided configuration, and executes the specified job.

    Command-line usage examples:

    **Normal execution (single GPU/CPU):**

    .. code-block:: bash

        python -u main.py \
            --config $input_job_cfg_file \
            --result_dir $result_dir \
            --name $job_name \
            --run_id $job_counter \
            --gpu 0

    **Distributed execution (e.g., with 2 GPUs):**

    .. code-block:: bash

        python -u -m torch.distributed.run \
            --nproc_per_node=2 \
            main.py \
            --config $input_job_cfg_file \
            --result_dir $result_dir \
            --name $job_name \
            --run_id $job_counter \
            --gpu 0,1

    Arguments parsed:

    - ``--config`` (required): Path to the YAML configuration file for the job.
    - ``--result_dir``: Path to the directory where the job's output results will be stored.
      Defaults to the user's home directory.
    - ``--name``: Name of the job. Defaults to "unknown_job".
    - ``--run_id``: Run number for the same job, useful for multiple runs with the same name.
      Defaults to 1.
    - ``--gpu``: GPU device ID(s) (e.g., "0" for single GPU, "0,1" for multiple)
      or "mps" for Apple Silicon.
    - ``-v``, ``--version``: Displays the BiaPy version.

    Distributed training specific arguments:

    - ``--world_size``: Total number of distributed processes. Defaults to 1.
    - ``--local_rank``: Rank of the current node/process within the distributed setup.
      Automatically set by ``torch.distributed.launch``. Defaults to -1.
    - ``--dist_on_itp``: Flag to enable distributed training on Interactive Training Platform (ITP).
      (Internal use case).
    - ``--dist_url``: URL used to set up distributed training (e.g., "env://").
      Defaults to "env://".
    - ``--dist_backend``: Backend to use for distributed communication.
      Choices: "nccl" (default, for GPUs), "gloo" (for CPUs).

    Upon execution, it initializes a `BiaPy` instance with the parsed arguments
    and calls its `run_job` method to start the workflow. The program exits
    with status 0 upon successful completion.
    """
    # Subcommands have their own options and don't run a job; any other call runs a job
    subcommands = {"validate-config": _validate_config_cli, "check-data": _check_data_cli, "list-workflows": _list_workflows_cli, "inspect-image": _inspect_image_cli, "get-config-template": _get_config_template_cli, "create-config": _create_config_cli, "search-bmz-models": _search_bmz_models_cli, "inspect-results": _inspect_results_cli, "plan": _plan_cli, "run": _run_cli, "job-status": _job_status_cli, "stop-job": _stop_job_cli}
    if len(sys.argv) > 1 and sys.argv[1] in subcommands:
        subcommands[sys.argv[1]](sys.argv[2:])

    ##########################
    #   ARGS COMPROBATION    #
    ##########################

    # Normal exec:
    # python -u main.py \
    #     --config $input_job_cfg_file \
    #     --result_dir $result_dir \
    #     --name $job_name \
    #     --run_id $job_counter \
    #     --gpu 0
    # Distributed:
    # python -u -m torch.distributed.run \
    #     --nproc_per_node=2 \
    #     main.py \
    #     --config $input_job_cfg_file \
    #     --result_dir $result_dir \
    #     --name $job_name \
    #     --run_id $job_counter \
    #     --gpu 0,1

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Other commands (they don't run any job; see 'biapy <command> --help'):\n"
        "  validate-config       check a configuration\n"
        "  check-data            check the data of a configuration, or folders for a workflow\n"
        "  list-workflows        list the workflows\n"
        "  inspect-image         describe an image: 2D or 3D, channels\n"
        "  get-config-template   print or save the basic configuration template of a workflow\n"
        "  create-config         create a configuration adapted to the data, as the wizard does\n"
        "  search-bmz-models     search the BioImage Model Zoo for pretrained models\n"
        "  inspect-results       describe what a job left in its folder: training, test metrics, outputs\n"
        "  plan                  describe what running a configuration would do, without running it\n"
        "  run                   run a job in the background, after checking it as 'plan' does\n"
        "  job-status            say whether a job is running, where it is and, if it failed, why\n"
        "  stop-job              stop a job started with 'run'\n",
    )

    parser.add_argument("--config", required=True, help="Path to the configuration file")
    parser.add_argument(
        "--result_dir",
        required=True,
        help="Path to where the resulting output of the job will be stored",
    )
    parser.add_argument("--name", help="Job name", default="unknown_job")
    parser.add_argument("--run_id", help="Run number of the same job", type=int, default=1)
    parser.add_argument(
        "--gpu",
        help="GPU number according to 'nvidia-smi' command / MPS device (Apple Silicon)",
        type=str,
    )
    parser.add_argument("-v", "--version", action="version", version="BiaPy version " + str(__version__))

    # Distributed training parameters
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument(
        "--local_rank",
        default=-1,
        type=int,
        help="Node rank for distributed training. Necessary for using the torch.distributed.launch utility.",
    )
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")
    parser.add_argument(
        "--dist_backend",
        type=str,
        default="nccl",
        choices=["nccl", "gloo"],
        help="Backend to use in distributed mode",
    )
    args = parser.parse_args()

    from ._biapy import BiaPy

    _biapy = BiaPy(**vars(args), verbose=True)
    _biapy.run_job()
    sys.exit(0)