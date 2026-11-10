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
import json
import os
import sys

# The engine (BiaPy, build_config, validate_config) is imported on first use: importing it loads PyTorch and every workflow,
# which takes several seconds, and lightweight parts such as biapy.wizard don't need it.
_LAZY = {"BiaPy": "._biapy", "build_config": "._biapy", "validate_config": "._biapy", "check_data": "._biapy", "check_data_for_workflow": "._biapy", "list_workflows": "._biapy", "inspect_image": "._biapy", "get_config_template": "._biapy", "create_config": "._biapy", "search_bmz_models": "._biapy"}
__all__ = ["BiaPy", "build_config", "validate_config", "check_data", "check_data_for_workflow", "list_workflows", "inspect_image", "get_config_template", "create_config", "search_bmz_models", "main", "__version__"]


def __getattr__(name):
    if name in _LAZY:
        import importlib

        value = getattr(importlib.import_module(_LAZY[name], __name__), name)
        globals()[name] = value  # later accesses don't go through __getattr__
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_LAZY))


def _validate_config_cli(argv):
    """
    Validate a configuration file from the command line.

    This function is intended to be used as a command-line utility to validate
    a BiaPy configuration file without running anything: no data is loaded, no
    model is built and nothing is written to disk. It prints ``OK`` if the
    configuration is valid, or the first error found (naming the wrong key) otherwise.

    Usage::

        biapy validate-config <path_to_config_file>
        biapy validate-config <path_to_config_file> --no-data-check

    ``--no-data-check`` skips the check that the data paths exist, so a configuration
    can be validated before the data is available.

    Parameters
    ----------
    argv : list of str
        Command-line arguments after ``validate-config``.

    Exit codes
    ----------
    0
        The configuration is valid.
    1
        The configuration is not valid (the error is printed to stderr).
    2
        Wrong command-line arguments (raised by ``argparse``).
    """
    from ._biapy import validate_config
    parser = argparse.ArgumentParser(
        prog="biapy validate-config",
        description="Check a BiaPy YAML configuration without running anything.",
    )
    parser.add_argument("config", help="Path to the YAML configuration file")
    parser.add_argument(
        "--no-data-check",
        action="store_true",
        help="Skip the check that the data paths exist",
    )
    args = parser.parse_args(argv)

    try:
        cfg = validate_config(args.config, check_data_paths=not args.no_data_check)
    except (ValueError, AssertionError, FileNotFoundError, KeyError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"OK: {args.config} is a valid {cfg.PROBLEM.TYPE} {cfg.PROBLEM.NDIM} configuration")
    sys.exit(0)


def _check_data_cli(argv):
    """
    Check from the command line that some data can be used by BiaPy, without running anything.

    Usage::

        biapy check-data <path_to_config_file> [--percentage 10] [--json]
        biapy check-data --workflow SEMANTIC_SEG --ndim 2D --raw <folder> [--gt <folder>] [--json]

    With a configuration, the data folders it points to are checked and compared with it
    (see :func:`biapy.check_data`). Without one, the folders given are checked for the workflow, as the
    wizard does (see :func:`biapy.check_data_for_workflow`). ``--json`` prints the full report as JSON on
    stdout, for agents and scripts.

    Parameters
    ----------
    argv : list of str
        Command-line arguments after ``check-data``.

    Exit codes
    ----------
    0
        The data can be used.
    1
        The data or the configuration is not valid (the errors are printed to stderr, or in the JSON).
    2
        Wrong command-line arguments (raised by ``argparse``).
    """
    import contextlib

    parser = argparse.ArgumentParser(
        prog="biapy check-data",
        description="Check that some data can be used by BiaPy: the data of a configuration, or folders for a workflow.",
    )
    parser.add_argument("config", nargs="?", help="Path to the YAML configuration file")
    parser.add_argument("--workflow", help="Without a configuration: PROBLEM.TYPE, e.g. SEMANTIC_SEG")
    parser.add_argument("--ndim", help="Without a configuration: 2D or 3D")
    parser.add_argument("--raw", help="Without a configuration: folder with the raw images")
    parser.add_argument("--gt", default="", help="Without a configuration: folder with the targets")
    parser.add_argument(
        "--percentage", type=float, default=100, help="Percentage of each Zarr/H5 image to check (default: 100)"
    )
    parser.add_argument("--json", action="store_true", help="Print the full report as JSON")
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
    except (ValueError, AssertionError, FileNotFoundError, KeyError) as e:
        report = {"ok": False, "errors": [str(e)], "warnings": [], "folders": {}, "upscaling": None}

    if args.json:
        from biapy.wizard.config_builder import to_jsonable

        print(json.dumps(to_jsonable(report), indent=2))
    else:
        for warn in report["warnings"]:
            print(f"WARNING: {warn}", file=sys.stderr)
        for err in report["errors"]:
            print(f"ERROR: {err}", file=sys.stderr)
        if report["ok"]:
            for key, found in report["folders"].items():
                print(f"OK: {key}: {found['n_files']} files, channels={found['channels']}, classes={found['classes']}")
            if report["upscaling"]:
                print(f"OK: upscaling {tuple(report['upscaling'])}")
    sys.exit(0 if report["ok"] else 1)


def _list_workflows_cli(argv):
    """
    List from the command line the workflows available in BiaPy.

    Usage::

        biapy list-workflows
        biapy list-workflows --json

    Parameters
    ----------
    argv : list of str
        Command-line arguments after ``list-workflows``.
    """
    from ._biapy import list_workflows

    parser = argparse.ArgumentParser(
        prog="biapy list-workflows",
        description="List the workflows available in BiaPy (the values of PROBLEM.TYPE).",
    )
    parser.add_argument("--json", action="store_true", help="Print the list as JSON")
    args = parser.parse_args(argv)

    workflows = list_workflows()
    print(json.dumps(workflows) if args.json else "\n".join(workflows))
    sys.exit(0)


def _inspect_image_cli(argv):
    """
    Describe from the command line an image as BiaPy reads it (see :func:`biapy.inspect_image`).

    Usage::

        biapy inspect-image <path_to_image>
        biapy inspect-image <path_to_image> --json

    Parameters
    ----------
    argv : list of str
        Command-line arguments after ``inspect-image``.
    """
    parser = argparse.ArgumentParser(
        prog="biapy inspect-image",
        description="Describe an image as BiaPy reads it, to know whether it is 2D or 3D.",
    )
    parser.add_argument("path", help="Path to the image")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON")
    args = parser.parse_args(argv)

    from ._biapy import inspect_image

    try:
        report = inspect_image(args.path)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        s = report["stored"]
        print(f"Stored:    shape {s['shape']}, axes {s['axes'] or 'unknown'}, {s['dtype']}")
        for ndim in ("2D", "3D"):
            r = report[ndim]
            if "error" in r:
                print(f"As {ndim}:     not possible ({r['error']})")
                continue
            shape = r["shape"]
            # BiaPy's order is (y, x, c) in 2D and (z, y, x, c) in 3D
            meaning = f"{shape[-1]} channel" + ("s" if shape[-1] != 1 else "")
            if ndim == "3D":
                meaning = f"{shape[0]} slices, " + meaning
            print(f"As {ndim}:     {str(shape):<20}  -> {meaning}")
        ndim = report["suggested_ndim"]
        if ndim:
            if report["axes_known"]:
                why = "the file has a Z axis" if ndim == "3D" else "the file has no Z axis"
            else:
                why = "it can only be read this way"
            print(f"Suggested: PROBLEM.NDIM: {ndim} ({why})")
        else:
            print("Suggested: unknown (the file doesn't store its axes): choose PROBLEM.NDIM from the readings above")
    sys.exit(0)


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
    """
    parser = argparse.ArgumentParser(
        prog="biapy get-config-template",
        description="Print or save the basic configuration template of a workflow.",
    )
    parser.add_argument("--workflow", required=True, help="PROBLEM.TYPE, e.g. SEMANTIC_SEG")
    parser.add_argument("--ndim", required=True, help="2D or 3D")
    parser.add_argument("--store", help="YAML file to write the template to")
    parser.add_argument("--json", action="store_true", help="Print the template as JSON")
    args = parser.parse_args(argv)

    from ._biapy import get_config_template

    try:
        config = get_config_template(args.workflow, args.ndim, store_yaml=args.store)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(config, indent=2))
    elif args.store:
        print(f"OK: template written to {args.store}")
    else:
        import yaml

        print(yaml.safe_dump(config, default_flow_style=False, sort_keys=False), end="")
    sys.exit(0)


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
        The data can't be used (the errors are printed to stderr, or in the JSON).
    2
        Wrong command-line arguments (raised by ``argparse``).
    """
    import contextlib

    parser = argparse.ArgumentParser(
        prog="biapy create-config",
        description="Create a configuration adapted to the data, as the wizard does, to train a model from scratch.",
    )
    parser.add_argument("--workflow", required=True, help="PROBLEM.TYPE, e.g. SEMANTIC_SEG")
    parser.add_argument("--ndim", required=True, help="2D or 3D")
    parser.add_argument("--train-raw", required=True, help="Folder with the training raw images")
    parser.add_argument("--train-gt", default="", help="Folder with the training targets")
    parser.add_argument("--test-raw", default="", help="Folder with the test raw images")
    parser.add_argument("--test-gt", default="", help="Folder with the test targets")
    parser.add_argument("--quick", action="store_true", help="A quick training, to check that everything works")
    parser.add_argument("--store", help="YAML file to write the configuration to")
    parser.add_argument("--json", action="store_true", help="Print the full report as JSON")
    args = parser.parse_args(argv)

    from ._biapy import create_config
    from biapy.wizard.config_builder import to_jsonable, tuples_to_lists

    try:
        # The checks print their progress: keep stdout for the result only
        with contextlib.redirect_stdout(sys.stderr):
            report = create_config(
                args.workflow, args.ndim, args.train_raw, args.train_gt, args.test_raw, args.test_gt,
                training_length="quick" if args.quick else "full", store_yaml=args.store,
            )
    except ValueError as e:
        report = {"ok": False, "errors": [str(e)], "warnings": [], "folders": {}, "config": None, "summary": []}

    if args.json:
        print(json.dumps(to_jsonable(report), indent=2))
    else:
        for warn in report["warnings"]:
            print(f"WARNING: {warn}", file=sys.stderr)
        for err in report["errors"]:
            print(f"ERROR: {err}", file=sys.stderr)
        if report["ok"]:
            for line in report["summary"]:
                print(f"# {line}")
            if args.store:
                print(f"OK: configuration written to {args.store}")
            else:
                import yaml

                print(yaml.safe_dump(tuples_to_lists(report["config"]), default_flow_style=False), end="")
    sys.exit(0 if report["ok"] else 1)


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
    parser = argparse.ArgumentParser(
        prog="biapy search-bmz-models",
        description="Search the BioImage Model Zoo for pretrained models BiaPy can use (it needs internet).",
    )
    parser.add_argument("--workflow", default="all", help="PROBLEM.TYPE, e.g. INSTANCE_SEG (default: all)")
    parser.add_argument("--ndim", default="all", help="2D or 3D (default: all)")
    parser.add_argument("--nchannels", default="all", help="Channels of the images, e.g. 1 (default: all)")
    parser.add_argument("--nclasses", default="all", help="Classes of the data (default: all)")
    parser.add_argument("--query", help="Words to find in the model's name, description or tags, e.g. 'nuclei'")
    parser.add_argument("--json", action="store_true", help="Print the models as JSON")
    args = parser.parse_args(argv)

    from ._biapy import search_bmz_models
    from biapy.wizard.config_builder import to_jsonable

    print("Searching the BioImage Model Zoo (it may take a minute)...", file=sys.stderr)
    try:
        models = search_bmz_models(args.workflow, args.ndim, args.nclasses, args.nchannels, args.query)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(to_jsonable(models), indent=2))
    elif not models:
        print("No compatible models found")
    else:
        for m in models:
            channels = m["opts"].get("DATA.PATCH_SIZE", ("?",))[-1]
            print(f"{m['id']:24s} {m['workflow_type']} {m['ndim']}, {channels} channel(s)   {m['name'][:50]}   {m['url']}")
    sys.exit(0)


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
    subcommands = {"validate-config": _validate_config_cli, "check-data": _check_data_cli, "list-workflows": _list_workflows_cli, "inspect-image": _inspect_image_cli, "get-config-template": _get_config_template_cli, "create-config": _create_config_cli, "search-bmz-models": _search_bmz_models_cli}
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
        "  search-bmz-models     search the BioImage Model Zoo for pretrained models\n",
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