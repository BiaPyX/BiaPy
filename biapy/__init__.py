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
import os
import sys

# The engine (BiaPy, build_config, validate_config) is imported on first use: importing it loads PyTorch and every workflow,
# which takes several seconds, and lightweight parts such as biapy.wizard don't need it.
_LAZY = {"BiaPy": "._biapy", "build_config": "._biapy", "validate_config": "._biapy", "check_data": "._biapy", "check_data_for_workflow": "._biapy", "list_workflows": "._biapy"}
__all__ = ["BiaPy", "build_config", "validate_config", "check_data", "check_data_for_workflow", "list_workflows", "main", "__version__"]


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
    import json

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
    import json

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
    # Subcommands only check or list things; any other call runs a job
    subcommands = {"validate-config": _validate_config_cli, "check-data": _check_data_cli, "list-workflows": _list_workflows_cli}
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
        epilog="To check without running anything: biapy validate-config <config> [--no-data-check] | "
        "biapy check-data <config> | biapy check-data --workflow <TYPE> --ndim <2D|3D> --raw <folder> [--gt <folder>]. "
        "To list the workflows: biapy list-workflows",
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