#!/usr/bin/env python3
"""
Run the sanity-check experiments of the "Representation matters" paper.

Each sanity check is described by a ``sanity_<name>.yaml`` file placed next to this
script. Those YAML files are prepared for a *train + test* run, so when only the test
phase is requested this script writes a modified copy of the configuration with:

    * ``TRAIN.ENABLE: False``
    * ``MODEL.LOAD_CHECKPOINT: True``
    * ``PATHS.CHECKPOINT_FILE`` -> checkpoint matched for this experiment

The original YAML files are never modified: every run gets its own configuration copy
inside ``<output_dir>/config_files``.

The script must be executed with the BiaPy environment active, e.g.::

    conda activate BiaPy_env

    # Only inference, reusing already trained weights
    python run_sanity_experiments.py --experiment cellpose \
        --output_dir /path/to/output --checkpoint_dir /path/to/checkpoints

    # Train from scratch and then test, for every representation
    python run_sanity_experiments.py --experiment all --mode train+test \
        --output_dir /path/to/output

No path is hardcoded: the BiaPy entry point is resolved relative to this file (it lives
inside the BiaPy repository) and the interpreter used to launch it is the one running
this script, unless ``--biapy_main``/``--python`` are given.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

# Prefix shared by every sanity-check configuration file, job name and checkpoint
CONFIG_PREFIX = "sanity_"
CONFIG_SUFFIX = ".yaml"
CHECKPOINT_EXTENSIONS = (".pth", ".safetensors")
# Run number used for every job, so the checkpoints are always named 'sanity_<name>_1-...'
RUN_ID = 1

SCRIPT_DIR = Path(__file__).resolve().parent
# templates/instance_segmentation/Representation_matters_paper/sanity_experiments -> repo root
DEFAULT_REPO_ROOT = SCRIPT_DIR.parents[3]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Experiment discovery
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def discover_experiments(config_dir):
    """
    Collect every sanity-check configuration available in ``config_dir``.

    Parameters
    ----------
    config_dir : Path
        Directory holding the ``sanity_<name>.yaml`` files.

    Returns
    -------
    dict of str, Path
        Experiment name (lowercase, e.g. ``cellpose``) -> configuration file path.
    """
    experiments = {}
    for cfg_file in sorted(config_dir.glob(f"{CONFIG_PREFIX}*{CONFIG_SUFFIX}")):
        name = cfg_file.stem[len(CONFIG_PREFIX):].lower()
        experiments[name] = cfg_file
    return experiments


def normalize_experiment_names(requested, experiments):
    """
    Translate the user request into the list of experiments to run.

    Accepts ``all``, the plain name (``cellpose``) or the full config stem
    (``sanity_cellpose``), in any capitalization.
    """
    available = ", ".join(sorted(experiments))
    selected = []
    for raw in requested:
        name = raw.strip().lower()
        if name == "all":
            return sorted(experiments)
        if name.startswith(CONFIG_PREFIX):
            name = name[len(CONFIG_PREFIX):]
        if name.endswith(CONFIG_SUFFIX):
            name = name[: -len(CONFIG_SUFFIX)]
        if name not in experiments:
            raise SystemExit(
                f"[ERROR] Unknown sanity check '{raw}'. Available options: {available}, all"
            )
        if name not in selected:
            selected.append(name)
    return selected


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Checkpoint handling
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def find_checkpoint(checkpoint_dir, job_name):
    """
    Find the checkpoint of a given experiment inside ``checkpoint_dir``.

    Every checkpoint is expected to contain the job name (e.g. ``sanity_cellpose``) in its
    filename, which is how BiaPy names them: ``<job_name>_1-checkpoint-best.pth``.

    Parameters
    ----------
    checkpoint_dir : Path
        Directory where all the checkpoints are placed.
    job_name : str
        Name of the job, e.g. ``sanity_cellpose``.

    Returns
    -------
    Path
        Checkpoint file to load.
    """
    if not checkpoint_dir.is_dir():
        raise SystemExit(f"[ERROR] Checkpoint directory not found: {checkpoint_dir}")

    candidates = [
        f
        for f in sorted(checkpoint_dir.rglob("*"))
        if f.is_file() and f.suffix in CHECKPOINT_EXTENSIONS and job_name in f.name
    ]
    if not candidates:
        found = [f.name for f in sorted(checkpoint_dir.rglob("*")) if f.suffix in CHECKPOINT_EXTENSIONS]
        raise SystemExit(
            f"[ERROR] No checkpoint containing '{job_name}' found in {checkpoint_dir}.\n"
            f"        Checkpoints available there: {found if found else 'none'}\n"
            f"        Expected something like '{job_name}_{RUN_ID}-checkpoint-best.pth'."
        )

    # Prefer the best-on-validation checkpoint, then any other BiaPy checkpoint
    def rank(path):
        if "-checkpoint-best" in path.name:
            return 0
        if "-checkpoint-" in path.name:
            return 1
        return 2

    best_rank = min(rank(c) for c in candidates)
    candidates = [c for c in candidates if rank(c) == best_rank]

    if len(candidates) > 1:
        # Several equally valid options: try to break the tie with the run number used
        by_run_id = [c for c in candidates if f"{job_name}_{RUN_ID}-" in c.name]
        if len(by_run_id) == 1:
            return by_run_id[0]
        raise SystemExit(
            f"[ERROR] Several checkpoints match '{job_name}' in {checkpoint_dir}:\n"
            + "\n".join(f"          - {c}" for c in candidates)
            + "\n        Keep only one of them in the checkpoint directory."
        )

    return candidates[0]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Configuration file preparation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def set_config_value(config, dotted_key, value):
    """Set ``dotted_key`` (e.g. ``MODEL.LOAD_CHECKPOINT``) within a nested dict."""
    keys = dotted_key.split(".")
    node = config
    for key in keys[:-1]:
        if not isinstance(node.get(key), dict):
            node[key] = {}
        node = node[key]
    node[keys[-1]] = value


def build_config(src_config, out_config, mode, checkpoint_file, data_dir):
    """
    Write the configuration file that will be handed to BiaPy.

    Parameters
    ----------
    src_config : Path
        Original ``sanity_<name>.yaml`` file (left untouched).
    out_config : Path
        Where the resulting configuration is written.
    mode : str
        Either ``test`` or ``train+test``.
    checkpoint_file : Path or None
        Checkpoint to load. Only used in ``test`` mode.
    data_dir : Path or None
        Root of the dataset. If given, the train/test paths of the YAML are replaced
        assuming the ``<data_dir>/{train,test}/{images,masks}`` layout of DSB2018.

    Returns
    -------
    dict
        The configuration finally written, useful to report the changes applied.
    """
    with open(src_config, "r") as f:
        config = yaml.safe_load(f)

    set_config_value(config, "TEST.ENABLE", True)

    if mode == "test":
        set_config_value(config, "TRAIN.ENABLE", False)
        set_config_value(config, "MODEL.LOAD_CHECKPOINT", True)
        # Set explicitly so the checkpoint is picked no matter how it is named
        set_config_value(config, "PATHS.CHECKPOINT_FILE", str(checkpoint_file))
    else:
        set_config_value(config, "TRAIN.ENABLE", True)
        set_config_value(config, "MODEL.LOAD_CHECKPOINT", False)

    if data_dir is not None:
        set_config_value(config, "DATA.TRAIN.PATH", str(data_dir / "train" / "images"))
        set_config_value(config, "DATA.TRAIN.GT_PATH", str(data_dir / "train" / "masks"))
        set_config_value(config, "DATA.TEST.PATH", str(data_dir / "test" / "images"))
        set_config_value(config, "DATA.TEST.GT_PATH", str(data_dir / "test" / "masks"))

    out_config.parent.mkdir(parents=True, exist_ok=True)
    with open(out_config, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)

    return config


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Job launching
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def run_job(command, log_file):
    """
    Run ``command``, mirroring its output both to the console and to ``log_file``.

    Returns
    -------
    int
        Return code of the process.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
        return process.wait()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the sanity-check experiments of the 'Representation matters' paper with BiaPy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Test only (default), loading the trained weights\n"
            "  python run_sanity_experiments.py -e cellpose -o ./sanity_out -c ./checkpoints\n\n"
            "  # Every sanity check, training from scratch and then testing\n"
            "  python run_sanity_experiments.py -e all -m train+test -o ./sanity_out\n"
        ),
    )
    parser.add_argument(
        "-e",
        "--experiment",
        nargs="+",
        required=True,
        metavar="NAME",
        help="Sanity check(s) to run, e.g. 'cellpose' (case insensitive). Use 'all' to run every one of them.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        type=Path,
        help="Directory where all the generated files (results, configs and logs) will be stored.",
    )
    parser.add_argument(
        "-m",
        "--mode",
        default="test",
        choices=["test", "train+test", "train_test"],
        help="Whether to only test the models or to train them and then test. Default: test.",
    )
    parser.add_argument(
        "-c",
        "--checkpoint_dir",
        type=Path,
        default=None,
        help=(
            "Directory holding the checkpoints of every experiment, searched recursively. Their filenames "
            "must contain the experiment's job name (e.g. 'sanity_cellpose'). Required in 'test' mode, "
            "ignored in 'train+test' mode."
        ),
    )
    parser.add_argument(
        "--config_dir",
        type=Path,
        default=SCRIPT_DIR,
        help="Directory containing the 'sanity_<name>.yaml' files. Default: the directory of this script.",
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        type=Path,
        default=None,
        help=(
            "Root of the dataset, expected to follow the '<data_dir>/{train,test}/{images,masks}' layout. "
            "If not provided, the paths already set in the YAML files are used."
        ),
    )
    parser.add_argument(
        "--gpu",
        default="0",
        help="GPU(s) to use, following 'nvidia-smi' numbering. Use '' to run on CPU. Default: 0.",
    )
    parser.add_argument(
        "--biapy_main",
        type=Path,
        default=DEFAULT_REPO_ROOT / "main.py",
        help="Path to BiaPy's 'main.py'. Default: the one of the repository this script belongs to.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used to launch BiaPy. Default: the one running this script.",
    )
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Stop as soon as one experiment fails instead of continuing with the remaining ones.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only prepare the configuration files and print the commands, without running BiaPy.",
    )

    args = parser.parse_args()
    args.mode = "train+test" if args.mode == "train_test" else args.mode
    return args


def main():
    args = parse_args()

    config_dir = args.config_dir.resolve()
    if not config_dir.is_dir():
        raise SystemExit(f"[ERROR] Configuration directory not found: {config_dir}")

    experiments = discover_experiments(config_dir)
    if not experiments:
        raise SystemExit(
            f"[ERROR] No '{CONFIG_PREFIX}*{CONFIG_SUFFIX}' configuration file found in {config_dir}"
        )
    to_run = normalize_experiment_names(args.experiment, experiments)

    if args.mode == "test" and args.checkpoint_dir is None:
        raise SystemExit(
            "[ERROR] In 'test' mode the trained weights must be provided. Set --checkpoint_dir with the "
            "directory holding the checkpoints, or use --mode train+test to train the models first."
        )
    if not args.biapy_main.is_file():
        raise SystemExit(
            f"[ERROR] BiaPy's main.py not found at {args.biapy_main}. Provide it with --biapy_main."
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.checkpoint_dir.resolve() if args.checkpoint_dir is not None else None
    data_dir = args.data_dir.resolve() if args.data_dir is not None else None

    print("#" * 80)
    print(f"# Sanity checks to run : {', '.join(to_run)}")
    print(f"# Mode                 : {args.mode}")
    print(f"# Output directory     : {output_dir}")
    print(f"# Checkpoint directory : {checkpoint_dir if args.mode == 'test' else 'not used (training from scratch)'}")
    print(f"# Dataset directory    : {data_dir if data_dir else 'the one set in each YAML file'}")
    print(f"# Python               : {args.python}")
    print(f"# BiaPy entry point    : {args.biapy_main}")
    print("#" * 80)

    results = {}
    for experiment in to_run:
        job_name = f"{CONFIG_PREFIX}{experiment}"
        src_config = experiments[experiment]
        out_config = output_dir / "config_files" / f"{job_name}_{args.mode.replace('+', '_')}.yaml"
        log_file = output_dir / "logs" / f"{job_name}_{args.mode.replace('+', '_')}.log"

        checkpoint_file = None
        if args.mode == "test":
            assert checkpoint_dir is not None
            checkpoint_file = find_checkpoint(checkpoint_dir, job_name)

        build_config(src_config, out_config, args.mode, checkpoint_file, data_dir)

        command = [
            args.python,
            "-u",
            str(args.biapy_main),
            "--config",
            str(out_config),
            "--result_dir",
            str(output_dir),
            "--name",
            job_name,
            "--run_id",
            str(RUN_ID),
            "--gpu",
            args.gpu,
        ]

        print("\n" + "=" * 80)
        print(f"[{experiment.upper()}] {'Training + testing' if args.mode == 'train+test' else 'Testing'}")
        print(f"  Source config : {src_config}")
        print(f"  Run config    : {out_config}")
        if checkpoint_file is not None:
            print(f"  Checkpoint    : {checkpoint_file}")
        print(f"  Log file      : {log_file}")
        print(f"  Command       : {' '.join(command)}")
        print("=" * 80, flush=True)

        if args.dry_run:
            results[experiment] = "skipped (dry run)"
            continue

        start = time.time()
        return_code = run_job(command, log_file)
        elapsed = time.time() - start

        if return_code == 0:
            results[experiment] = f"OK ({elapsed / 60:.1f} min)"
        else:
            results[experiment] = f"FAILED (return code {return_code}, see {log_file})"
            if args.stop_on_error:
                print(f"\n[ERROR] '{experiment}' failed and --stop_on_error was set. Aborting.")
                break

    print("\n" + "#" * 80)
    print("# Summary")
    print("#" * 80)
    for experiment in to_run:
        print(f"  {experiment:<12} : {results.get(experiment, 'not run')}")
    print(f"\nAll the generated files are in: {output_dir}")

    failed = [e for e, r in results.items() if r.startswith("FAILED")]
    sys.exit(1 if failed or len(results) != len(to_run) else 0)


if __name__ == "__main__":
    main()
