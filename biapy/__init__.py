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

Version: 3.6.3
"""

__version__ = "3.6.3"

import argparse
import os
import sys
from ._biapy import BiaPy


def main():
    r"""
    Run BiaPy.

    Main entry point for the BiaPy command-line interface.

    This function parses command-line arguments, initializes the `BiaPy`
    application with the provided configuration, and executes the specified job.

    Command-line usage examples:

    **Normal execution (single GPU/CPU):**
    ```bash
    python -u main.py \\
        --config $input_job_cfg_file \\
        --result_dir $result_dir \\
        --name $job_name \\
        --run_id $job_counter \\
        --gpu 0
    ```

    **Distributed execution (e.g., with 2 GPUs):**
    ```bash
    python -u -m torch.distributed.run \\
        --nproc_per_node=2 \\
        main.py \\
        --config $input_job_cfg_file \\
        --result_dir $result_dir \\
        --name $job_name \\
        --run_id $job_counter \\
        --gpu 0,1
    ```

    Arguments parsed:
    - `--config` (required): Path to the YAML configuration file for the job.
    - `--result_dir`: Path to the directory where the job's output results will be stored.
                      Defaults to the user's home directory.
    - `--name`: Name of the job. Defaults to "unknown_job".
    - `--run_id`: Run number for the same job, useful for multiple runs with the same name.
                  Defaults to 1.
    - `--gpu`: GPU device ID(s) (e.g., "0" for single GPU, "0,1" for multiple)
               or "mps" for Apple Silicon.
    - `-v`, `--version`: Displays the BiaPy version.

    Distributed training specific arguments:
    - `--world_size`: Total number of distributed processes. Defaults to 1.
    - `--local_rank`: Rank of the current node/process within the distributed setup.
                      Automatically set by `torch.distributed.launch`. Defaults to -1.
    - `--dist_on_itp`: Flag to enable distributed training on Interactive Training Platform (ITP).
                       (Internal use case).
    - `--dist_url`: URL used to set up distributed training (e.g., "env://").
                    Defaults to "env://".
    - `--dist_backend`: Backend to use for distributed communication.
                        Choices: "nccl" (default, for GPUs), "gloo" (for CPUs).

    Upon execution, it initializes a `BiaPy` instance with the parsed arguments
    and calls its `run_job` method to start the workflow. The program exits
    with status 0 upon successful completion.
    """
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the configuration file")
    parser.add_argument(
        "--result_dir",
        help="Path to where the resulting output of the job will be stored",
        default=os.getenv("HOME"),
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

    _biapy = BiaPy(**vars(args))
    _biapy.run_job()
    sys.exit(0)
