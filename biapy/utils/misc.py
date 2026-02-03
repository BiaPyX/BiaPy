"""
This module provides a collection of utility functions and classes primarily designed to support distributed training, logging, and model management within a PyTorch deep learning workflow.

It includes functionalities for:
- Initializing and managing distributed training environments (DDP).
- Controlling print statements for master processes in distributed setups.
- Setting random seeds for reproducibility.
- Gradient norm calculation.
- Saving and loading model checkpoints.
- Converting data formats between PyTorch tensors and NumPy arrays.
- Logging metrics to TensorBoard.
- Tracking and smoothing metric values during training.
- Iterating with progress logging.
- Updating nested dictionaries.
- Cleaning directory walks by excluding specific files/directories.

The module aims to streamline common deep learning operations, especially
in distributed and large-scale training scenarios.
"""
import os
import re
import sys
import builtins
import time
import glob
import datetime
import numpy as np
from collections import defaultdict, deque
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from pathlib import Path
from yacs.config import CfgNode as CN
from functools import partial
import collections.abc
import gc
from typing import (
    Optional,
    Tuple,
    List,
    Iterator,
)
from numpy.typing import NDArray
import multiprocessing

from torch import inf
from datetime import timedelta

original_print = builtins.print


def setup_for_distributed(is_master):
    """
    Disable printing for non-master processes in a distributed training setup.

    This function replaces the built-in `print` function with a custom one
    that only prints output if the current process is the master process (rank 0),
    or if `force=True` is passed to the print call. This prevents cluttered
    output when running on multiple GPUs/nodes.

    Parameters
    ----------
    is_master : bool
        True if the current process is the master process (rank 0), False otherwise.
    """
    builtin_print = original_print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print("[{}] ".format(now), end="")  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    """
    Check if PyTorch distributed backend is available and initialized.

    Returns
    -------
    bool
        True if distributed training is available and initialized, False otherwise.
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """
    Return the total number of participating processes in the distributed group.

    Returns 0 if distributed mode is not initialized.

    Returns
    -------
    int
        The world size.
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """
    Return the rank of the current process in the distributed group.

    Returns 0 if distributed mode is not initialized. The master process typically has rank 0.

    Returns
    -------
    int
        The rank of the current process.
    """
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    """
    Check if the current process is the main (master) process (rank 0).

    Returns
    -------
    bool
        True if the current process is the main process, False otherwise.
    """
    return get_rank() == 0


def init_devices(args, cfg):
    """
    Initialize the PyTorch distributed environment and sets up the device for the current process.

    This function handles different distributed setup scenarios (e.g., ITP, environment variables, SLURM).
    It sets the appropriate GPU device, initializes the process group, and configures
    the custom print function for distributed logging.

    Parameters
    ----------
    args : Any
        An object containing command-line arguments or configuration,
        expected to have attributes like `dist_on_itp`, `gpu`, `dist_backend`, `dist_url`.
    cfg : YACS CN object
        The configuration object, used to determine the default device if CUDA is not available.

    Returns
    -------
    torch.device
        The PyTorch device assigned to the current process.

    Raises
    ------
    AssertionError
        If distributed training is attempted without GPUs when environment variables are set.
    """
    if args.dist_on_itp:
        args.rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        args.world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        args.gpu = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        args.dist_url = "tcp://%s:%s" % (
            os.environ["MASTER_ADDR"],
            os.environ["MASTER_PORT"],
        )
        os.environ["LOCAL_RANK"] = str(args.gpu)
        os.environ["RANK"] = str(args.rank)
        os.environ["WORLD_SIZE"] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        assert torch.cuda.is_available(), "Distributed training without GPUs is not supported!"

        env_dict = {
            key: os.environ[key]
            for key in (
                "MASTER_ADDR",
                "MASTER_PORT",
                "RANK",
                "LOCAL_RANK",
                "WORLD_SIZE",
            )
        }
        print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif (
        "SLURM_PROCID" in os.environ
        and args.gpu is not None
        and len(np.unique(np.array(args.gpu.strip().split(",")))) > 1
    ):
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        if torch.cuda.is_available() and args.gpu is not None:
            device = torch.device("cuda")
        else:
            device = torch.device(cfg.SYSTEM.DEVICE)
        return device

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    print(
        "| distributed init (rank {}): {}, gpu {}".format(args.rank, args.dist_url, args.gpu),
        flush=True,
    )
    if cfg.TEST.BY_CHUNKS.ENABLE and cfg.TEST.BY_CHUNKS.WORKFLOW_PROCESS.ENABLE:
        os.environ["NCCL_BLOCKING_WAIT"] = "0"  # not to enforce timeout in nccl backend
        timeout_ms = 36000000
    else:
        timeout_ms = 1800000

    if not is_dist_avail_and_initialized():
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
            timeout=timedelta(seconds=timeout_ms),
        )
    else:
        # If it was initialized means that something may have been running in the past so here
        # we are trying to clean all the cache as much as possible
        torch.cuda.empty_cache()
        gc.collect()

    dist.barrier()
    setup_for_distributed(args.rank == 0)
    if args.rank == 0:
        device = torch.device("cuda" if torch.cuda.is_available() else cfg.SYSTEM.DEVICE)
    else:
        device = torch.device(f"cuda:{args.rank}" if torch.cuda.is_available() else cfg.SYSTEM.DEVICE)
    return device


def set_seed(seed=42):
    """
    Set the random seed for reproducibility across multiple Python modules and PyTorch.

    The seed is adjusted by the distributed rank to ensure different random
    states for each process in a distributed setup, which can be beneficial
    for certain operations (e.g., data loading).

    Parameters
    ----------
    seed : int, optional
        The base seed value. Defaults to 42.
    """
    seed = seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    """
    Compute the total norm of gradients for a collection of parameters.

    This function is typically used for gradient clipping.

    Parameters
    ----------
    parameters : Iterable[torch.Tensor] or torch.Tensor
        An iterable of model parameters or a single parameter tensor.
    norm_type : float, optional
        The type of the norm (e.g., 2.0 for L2 norm, `inf` for max norm).
        Defaults to 2.0.

    Returns
    -------
    torch.Tensor
        The total norm of the gradients. Returns a tensor with value 0.0 if no
        parameters have gradients.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
            norm_type,
        )
    return total_norm


def save_model(cfg, biapy_version, jobname, epoch, model_without_ddp, optimizer, model_build_kwargs=None, extension="pth"):
    """
    Save the model checkpoint to the specified path.

    This function saves the model's state dictionary, optimizer state, current epoch,
    configuration, and BiaPy version. It ensures that saving is performed only by
    the main process in a distributed setup.

    Parameters
    ----------
    cfg : YACS CN object
        The configuration object.
    biapy_version : str
        The current version of BiaPy.
    jobname : str
        The name of the current job/experiment.
    epoch : int
        The current epoch number.
    model_without_ddp : nn.Module
        The model instance, typically the unwrapped model if using DistributedDataParallel.
    optimizer : torch.optim.Optimizer
        The optimizer's state.
    model_build_kwargs : Optional[Dict], optional
        Keyword arguments used to build the model, useful for re-instantiating
        the model from the checkpoint. Defaults to None.
    extension : str, optional
        The file extension for the checkpoint file. Options are 'pth' (native PyTorch format)
        or 'safetensors' (https://github.com/huggingface/safetensors). Defaults to "pth".

    Returns
    -------
    Path
        The path to the saved checkpoint file.
    """
    output_dir = Path(cfg.PATHS.CHECKPOINT)
    checkpoint_paths = [output_dir / "{}-checkpoint-{}.{}".format(jobname, str(epoch), extension)]

    for checkpoint_path in checkpoint_paths:
        to_save = {
            "model_build_kwargs": model_build_kwargs,
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "cfg": cfg,
            "biapy_version": biapy_version,
        }

        save_on_master(to_save, checkpoint_path)
    if len(checkpoint_paths) > 0:
        return checkpoint_paths[0]


def save_on_master(model_dict, checkpoint_path):
    """
    Save a PyTorch object only if the current process is the main (master) process.

    This is a wrapper around `torch.save` to ensure that checkpoints are not
    redundantly saved by all processes in a distributed training setup.

    Parameters
    ----------
    *args : Any
        Positional arguments to pass to `torch.save`.
    **kwargs : Any
        Keyword arguments to pass to `torch.save`.
    """
    if is_main_process():
        if str(checkpoint_path).endswith(".pth"):
            torch.save(model_dict, checkpoint_path)
        elif str(checkpoint_path).endswith(".safetensors"):
            from safetensors.torch import save_file
            save_file(model_dict["model"], checkpoint_path)
        else:
            raise ValueError("Unsupported checkpoint extension: {}".format(checkpoint_path))

def get_checkpoint_path(cfg, jobname):
    """
    Determine the path to the checkpoint file to load.

    It selects the checkpoint based on `cfg.PATHS.CHECKPOINT_FILE`,
    `cfg.MODEL.LOAD_CHECKPOINT_EPOCH` ("last_on_train" or "best_on_val"),
    and the `jobname`.

    Parameters
    ----------
    cfg : YACS CN object
        The configuration object. Key parameters:

        - `cfg.PATHS.CHECKPOINT`: Base directory for checkpoints.
        - `cfg.PATHS.CHECKPOINT_FILE`: Explicit path to a checkpoint file (if set).
        - `cfg.MODEL.LOAD_CHECKPOINT_EPOCH`: Strategy for selecting checkpoint
          ("last_on_train" or "best_on_val").
          
    jobname : str
        The name of the current job/experiment.

    Returns
    -------
    str
        The absolute path to the checkpoint file without the extension (without the .pth or .safetensors).

    Raises
    ------
    NotImplementedError
        If `cfg.MODEL.LOAD_CHECKPOINT_EPOCH` is an unrecognized value.
    """
    checkpoint_dir = Path(cfg.PATHS.CHECKPOINT)

    # Select the checkpoint source file
    if cfg.PATHS.CHECKPOINT_FILE != "":
        resume = cfg.PATHS.CHECKPOINT_FILE
        resume, _ = os.path.splitext(resume)
    else:
        if cfg.MODEL.LOAD_CHECKPOINT_EPOCH == "last_on_train":
            all_checkpoints = glob.glob(os.path.join(checkpoint_dir, "{}-checkpoint-*".format(jobname)))
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split("-")[-1].split(".")[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                resume = os.path.join(checkpoint_dir, "{}-checkpoint-{}".format(jobname, latest_ckpt))
        elif cfg.MODEL.LOAD_CHECKPOINT_EPOCH == "best_on_val":
            resume = os.path.join(checkpoint_dir, "{}-checkpoint-best".format(jobname))
        else:
            raise NotImplementedError
    return resume

def load_model_checkpoint(cfg, jobname, model_without_ddp, device, optimizer=None, just_extract_checkpoint_info=False, skip_unmatched_layers=False):
    """
    Load a model checkpoint from disk.

    This function handles loading the model's state dictionary, optimizer state,
    and epoch number from a checkpoint file. It can also be configured to
    only extract configuration information or to skip layers with mismatched shapes.

    Parameters
    ----------
    cfg : YACS CN object
        The configuration object. Key parameters:
        - `cfg.PATHS.CHECKPOINT_FILE`: Explicit path to checkpoint.
        - `cfg.MODEL.LOAD_CHECKPOINT_EPOCH`: Strategy for checkpoint selection.
        - `cfg.MODEL.LOAD_CHECKPOINT_ONLY_WEIGHTS`: If True, only model weights are loaded.
    jobname : str
        The name of the current job/experiment.
    model_without_ddp : nn.Module
        The model instance (unwrapped if DDP is used) to load weights into.
    device : torch.device
        The device to map the loaded checkpoint to.
    optimizer : Optional[torch.optim.Optimizer], optional
        The optimizer instance to load state into. If None, optimizer state is not loaded.
        Defaults to None.
    just_extract_checkpoint_info : bool, optional
        If True, only the configuration (`cfg`) and BiaPy version from the checkpoint
        are returned, without loading model or optimizer states. Defaults to False.
    skip_unmatched_layers : bool, optional
        If True, layers in the checkpoint that have different shapes than the
        current model's layers will be skipped during loading. Defaults to False.

    Returns
    -------
    Tuple[int | CN | None, str | None]
        If `just_extract_checkpoint_info` is True: returns `(checkpoint_cfg, biapy_version)`.
        Otherwise: returns `(start_epoch, resume_path)`.
        `checkpoint_cfg` and `biapy_version` can be `None` if not found in the checkpoint.

    Raises
    ------
    FileNotFoundError
        If the specified checkpoint file does not exist.
    """
    start_epoch = 0

    resume = get_checkpoint_path(cfg, jobname)

    # Take the first existing file with supported extension
    for ext in ['.pth', '.safetensors']:
        if os.path.exists(resume + ext):
            resume += ext
            break

    if not os.path.exists(resume):
        raise FileNotFoundError(f"Checkpoint file {resume} not found (considering .pth and .safetensors extensions)")
    else:
        if just_extract_checkpoint_info:
            print("Extracting model from checkpoint file {}".format(resume))
        else:
            print("Loading checkpoint from file {}".format(resume))

    # Load checkpoint file
    torch.serialization.add_safe_globals([CN])
    torch.serialization.add_safe_globals([set])
    torch.serialization.add_safe_globals([partial])
    torch.serialization.add_safe_globals([torch.nn.modules.normalization.LayerNorm])
    if resume.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(resume, map_location=device, check_hash=True)
    elif resume.endswith(".safetensors"):
        from safetensors.torch import load_file
        checkpoint = {
            "model": load_file(resume, device="cpu")
        }
    else: # ends with .pth
        checkpoint = torch.load(resume, map_location=device, weights_only=True)

    if just_extract_checkpoint_info:
        if "cfg" not in checkpoint and not resume.endswith(".safetensors"):
            print(
                "Checkpoint seems to not be from BiaPy (v3.5.1 or later) as model building args couldn't be extracted. Thus, "
                "the model will be built based on the current configuration"
            )
        return (
            checkpoint["cfg"] if "cfg" in checkpoint else None,
            checkpoint["biapy_version"] if "biapy_version" in checkpoint else None,
        )

    if 'model' in checkpoint:
        checkpoint_state_dict = checkpoint['model']
    elif 'model_state_dict' in checkpoint:
        checkpoint_state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        # Common convention in PyTorch Lightning
        checkpoint_state_dict = checkpoint['state_dict']
    else:
        checkpoint_state_dict = checkpoint

    if not skip_unmatched_layers:
        model_without_ddp.load_state_dict(checkpoint_state_dict, strict=False)
    else:
        # Filter out layers with mismatched shapes
        filtered_state_dict = {}
        model_state_dict = model_without_ddp.state_dict()
        for k, v in checkpoint_state_dict.items():
            if k in model_state_dict:
                if v.shape == model_state_dict[k].shape:
                    filtered_state_dict[k] = v
                else:
                    print(f"Skipping layer '{k}' due to shape mismatch: checkpoint {v.shape} vs model {model_state_dict[k].shape}")
            else:
                print(f"Skipping unexpected layer '{k}' not found in model.")

        # Load only matching parameters
        model_without_ddp.load_state_dict(filtered_state_dict, strict=False)

    print("Model weights loaded!")

    if cfg.MODEL.LOAD_CHECKPOINT_ONLY_WEIGHTS:
        return start_epoch, resume

    # Load also opt, epoch and scaler info
    if "optimizer" in checkpoint and optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"], strict=False)
        print("Optimizer info loaded!")

    if "epoch" in checkpoint:
        start_epoch = checkpoint["epoch"]
        if isinstance(start_epoch, str):
            start_epoch = 0
        print("Epoch loaded!")

    return start_epoch, resume


def all_reduce_mean(x):
    """
    Perform an all-reduce operation on a scalar or single-element tensor, then computes the mean across all processes in a distributed group.

    If not in a distributed environment, returns the input value directly.

    Parameters
    ----------
    x : float or torch.Tensor
        The scalar value or single-element tensor to be reduced.

    Returns
    -------
    float
        The mean of `x` across all processes.
    """
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x


def to_pytorch_format(x: torch.Tensor | NDArray, axes_order: Tuple, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    """
    Convert a NumPy array or PyTorch tensor to PyTorch tensor format with a specified axis order and moves it to the target device.

    Parameters
    ----------
    x : torch.Tensor or numpy.ndarray
        The input data.
    axes_order : Tuple[int, ...]
        A tuple specifying the desired permutation of axes.
        For example, `(0, 3, 1, 2)` for `(N, H, W, C)` to `(N, C, H, W)`.
    device : torch.device
        The target PyTorch device (e.g., "cuda", "cpu").
    dtype : torch.dtype, optional
        The desired data type for the output tensor. Defaults to `torch.float32`.

    Returns
    -------
    torch.Tensor
        The converted PyTorch tensor.
    """
    if isinstance(x, torch.Tensor) and torch.is_tensor(x):
        return x.to(dtype).permute(axes_order).to(device, non_blocking=True)
    else:
        return torch.from_numpy(x).to(dtype).permute(axes_order).to(device, non_blocking=True)


def to_numpy_format(x, axes_order_back):
    """
    Convert a PyTorch tensor back to a NumPy array with a specified axis order.

    Parameters
    ----------
    x : torch.Tensor
        The input PyTorch tensor.
    axes_order_back : Tuple[int, ...]
        A tuple specifying the desired permutation of axes to revert to
        the original NumPy-like order.

    Returns
    -------
    numpy.ndarray
        The converted NumPy array.
    """
    return x.permute(axes_order_back).cpu().numpy()


def time_text(t):
    """
    Format a time duration (in seconds) into a human-readable string.

    Formats as 'Xh', 'Xm', or 'Xs' depending on the duration.

    Parameters
    ----------
    t : float
        Time duration in seconds.

    Returns
    -------
    str
        Formatted time string.
    """
    if t >= 3600:
        return "{:.1f}h".format(t / 3600)
    elif t >= 60:
        return "{:.1f}m".format(t / 60)
    else:
        return "{:.1f}s".format(t)


class TensorboardLogger(object):
    """A simple wrapper for `tensorboardX.SummaryWriter` to log scalar metrics."""

    def __init__(self, log_dir):
        """
        Initialize the TensorboardLogger.

        Parameters
        ----------
        log_dir : str
            The directory where TensorBoard log files will be saved.
        """
        self.writer = SummaryWriter(logdir=log_dir)
        self.step = 0

    def set_step(self, step=None):
        """
        Set the current global step for logging.

        If `step` is None, increments the internal step counter.

        Parameters
        ----------
        step : Optional[int], optional
            The specific step number to set. If None, increments the current step.
            Defaults to None.
        """
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, head="scalar", step=None, **kwargs):
        """
        Log scalar values to TensorBoard.

        Parameters
        ----------
        head : str, optional
            The main category for the scalar (e.g., "train_loss", "val_metrics").
            Defaults to "scalar".
        step : Optional[int], optional
            The specific global step to log this update at. If None, uses the
            internal `self.step`. Defaults to None.
        **kwargs : float | int | torch.Tensor
            Keyword arguments where keys are metric names (e.g., "loss", "accuracy")
            and values are the corresponding scalar values (can be PyTorch tensors
            or Python floats/ints).
        """
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.writer.add_scalar(head + "/" + k, v, self.step if step is None else step)

    def flush(self):
        """Ensure all pending events have been written to disk."""
        self.writer.flush()


class SmoothedValue(object):
    """Track a series of values and provides access to smoothed values (median, average) over a sliding window or the global series average."""

    def __init__(self, window_size=20, fmt=None):
        """
        Initialize the SmoothedValue tracker.

        Parameters
        ----------
        window_size : int, optional
            The size of the sliding window for calculating median and average.
            Defaults to 20.
        fmt : Optional[str], optional
            A format string for displaying the value. Placeholders include
            `{median}`, `{avg}`, `{global_avg}`, `{max}`, `{value}`.
            Defaults to "{median:.4f} ({global_avg:.4f})".
        """
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt
        self.eps = sys.float_info.epsilon

    def update(self, value, n=1):
        """
        Update the tracker with a new value.

        Parameters
        ----------
        value : float
            The new value to add.
        n : int, optional
            The number of samples represented by this value (e.g., batch size).
            Defaults to 1.
        """
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Synchronize the `count` and `total` attributes across all processes in a distributed environment using `dist.all_reduce`.

        Warning: This method does *not* synchronize the `deque` (sliding window).
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        """Return the median of the values in the current sliding window."""
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        """Return the average of the values in the current sliding window."""
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self) -> float:
        """Return the global average of all values recorded since initialization."""
        return self.total / (self.count + self.eps)

    @property
    def max(self):
        """Return the maximum value in the current sliding window."""
        return max(self.deque)

    @property
    def value(self):
        """Return the most recently updated value."""
        return self.deque[-1]

    def __str__(self):
        """Return a formatted string representation of the smoothed value."""
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):
    """Aggregate and logs various metrics using `SmoothedValue` objects."""

    def __init__(self, delimiter="\t", verbose=False):
        r"""
        Initialize the MetricLogger.

        Parameters
        ----------
        delimiter : str, optional
            The string used to separate metrics when printing. Defaults to \t".
        verbose : bool, optional
            If True, additional information (e.g., max GPU memory) is printed.
            Defaults to False.
        """
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.verbose = verbose

    def update(self, **kwargs):
        """
        Update the values of tracked metrics.

        Parameters
        ----------
        **kwargs : float | int | torch.Tensor
            Keyword arguments where keys are metric names and values are their
            current scalar values.
        """
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        """Allow direct access to `SmoothedValue` objects via attribute lookup."""
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))

    def __str__(self):
        """Return a string representation of all tracked metrics."""
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        """Synchronize all tracked `SmoothedValue` meters across distributed processes."""
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        """
        Add a custom `SmoothedValue` meter to the logger.

        Parameters
        ----------
        name : str
            The name of the meter.
        meter : SmoothedValue
            The `SmoothedValue` instance to add.
        """
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        """
        Log progress for an iterable, printing metrics at a specified frequency.

        Parameters
        ----------
        iterable : Iterable[Any]
            The iterable (e.g., DataLoader) to iterate over.
        print_freq : int
            The frequency (in iterations) at which to print log messages.
        header : Optional[str], optional
            An optional header string to prepend to log messages. Defaults to None.

        Yields
        ------
        Any
            Items from the input `iterable`.
        """
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_msg = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "iter-time: {time}",
        ]
        if torch.cuda.is_available() and self.verbose:
            log_msg.append("max mem: {memory:.0f}MB")
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available() and self.verbose:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("{} Total time: {} ({:.4f} s / it)".format(header, total_time_str, total_time / len(iterable)))


def update_dict_with_existing_keys(d, u, not_recognized_keys=[], not_recognized_key_vals=[]):
    """
    Recursively update a dictionary `d` with values from dictionary `u`, only for keys that already exist in `d`.

    This function is useful for updating configuration dictionaries while
    ensuring that no new keys are introduced from the update dictionary.
    It also tracks keys from `u` that were not found in `d`.

    Parameters
    ----------
    d : Dict
        The dictionary to be updated (destination).
    u : Dict
        The dictionary containing update values (source).
    not_recognized_keys : Optional[List], optional
        A list to append keys from `u` that were not found in `d`.
        If None, a new list is created. Defaults to None.
    not_recognized_key_vals : Optional[List], optional
        A list to append values corresponding to `not_recognized_keys`.
        If None, a new list is created. Defaults to None.

    Returns
    -------
    Tuple[Dict, List, List]
        - `d`: The updated dictionary.
        - `not_recognized_keys`: List of keys from `u` not found in `d`.
        - `not_recognized_key_vals`: List of values from `u` corresponding to `not_recognized_keys`.
    """
    for k, v in u.items():
        if k in d:
            if isinstance(v, collections.abc.Mapping):
                d[k], _, _ = update_dict_with_existing_keys(
                    d.get(k, {}), v, not_recognized_keys, not_recognized_key_vals
                )
            else:
                if k in d:
                    d[k] = v
                else:
                    not_recognized_keys.append(k)
                    not_recognized_key_vals.append(v)
        else:
            not_recognized_keys.append(k)
            not_recognized_key_vals.append(v)

    return d, not_recognized_keys, not_recognized_key_vals

def os_walk_clean(
    path: str,
    exclude_files: Tuple = ("Thumbs.db", "desktop.ini", ".DS_Store"),
    exclude_dirs: Tuple = (".git", "__pycache__")
) -> Iterator[Tuple[str, List[str], List[str]]]:
    """
    Clean os.walk + robust natural sorting (numeric-aware).
    
    Parameters
    ----------
    path : str
        The root directory to walk.
    exclude_files : tuple, optional
        Filenames to exclude from the results. Defaults to common system files.
    exclude_dirs : tuple, optional
        Directory names to exclude from the results. Defaults to common system directories.
    Yields
    ------
    Iterator[Tuple[str, List[str], List[str]]]
        Yields tuples of (root, dirs, files) with excluded items removed and
        directories/files sorted in natural order.
    """

    def natural_key(s):
        # Split filename into chunks of digits and non-digits,
        # keeping all chunks as strings but zero-pad digits for proper order.
        parts = re.findall(r'\d+|\D+', s)
        # Pad numeric chunks so '2' < '10' < '100'
        return [p.zfill(10) if p.isdigit() else p.lower() for p in parts]

    for root, dirs, files in os.walk(path):
        dirs[:]  = [d for d in dirs  if d not in exclude_dirs and not d.startswith('.')]
        files    = [f for f in files if f not in exclude_files and not f.startswith('.')]

        # Safe natural sort
        dirs.sort(key=natural_key)
        files.sort(key=natural_key)
        yield root, dirs, files

def resolve_cpu_budget(user_num_cpus: int) -> int:
    """Total CPU cores budget for the entire job."""
    if user_num_cpus == -1:
        # If you use CPU affinity / SLURM cpuset, you may want to respect that instead of cpu_count()
        return multiprocessing.cpu_count()
    return max(1, int(user_num_cpus))

def compute_threads_and_workers(
    user_num_cpus: int,
    world_size: int,
    training_samples: Optional[int] = None,
    max_workers_cap: int = 8
) -> Tuple[int, int, int, int]:
    """
    Compute CPU budget, CPU per rank, main threads, and DataLoader workers per rank.

    Parameters
    ----------
    user_num_cpus : int
        User-specified number of CPUs (-1 to use all available).

    world_size : int
        Number of distributed ranks/processes.

    training_samples : int, optional
        Number of training samples (to limit workers for small datasets).

    max_workers_cap : int, optional
        Maximum cap on DataLoader workers per rank. Defaults to 8.

    Returns
    ------- 
    Tuple[int, int, int, int]
        - `cpu_budget`: Total CPU cores budget for the job.
        - `cpu_per_rank`: CPU cores allocated per rank.
        - `main_threads`: Number of main threads for training process.
        - `num_workers`: Number of DataLoader workers per rank.
    """
    cpu_budget = resolve_cpu_budget(user_num_cpus)
    world_size = max(1, int(world_size))

    cpu_per_rank = max(1, cpu_budget // world_size)

    # Conservative: keep training-process threads modest so DataLoader can breathe
    main_threads = min(4, cpu_per_rank)

    # Leave 1 core for OS/overhead
    workers_per_rank_budget = max(0, cpu_per_rank - main_threads - 1)

    num_workers = min(workers_per_rank_budget, max_workers_cap)

    # Also don't spawn more workers than you have samples (helps tiny datasets)
    if training_samples is not None:
        num_workers = min(num_workers, max(0, int(training_samples)))

    return cpu_budget, cpu_per_rank, main_threads, num_workers