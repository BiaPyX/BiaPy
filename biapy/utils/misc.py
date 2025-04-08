import os
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
    Tuple,
    Literal,
    Dict,
)
from numpy.typing import NDArray

# from torch._six import inf
from torch import inf
from datetime import timedelta

original_print = builtins.print


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
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
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def init_devices(args, cfg):
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
    Sets the seed on multiple python modules to obtain results as reproducible as possible.

    Parameters
    ----------
    seed : int, optional
        Seed value.
    """
    seed = seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
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


def save_model(cfg, biapy_version, jobname, epoch, model_without_ddp, optimizer, model_build_kwargs=None):
    output_dir = Path(cfg.PATHS.CHECKPOINT)
    checkpoint_paths = [output_dir / "{}-checkpoint-{}.pth".format(jobname, str(epoch))]

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


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def get_checkpoint_path(cfg, jobname):
    checkpoint_dir = Path(cfg.PATHS.CHECKPOINT)

    # Select the checkpoint source file
    if cfg.PATHS.CHECKPOINT_FILE != "":
        resume = cfg.PATHS.CHECKPOINT_FILE
    else:
        if cfg.MODEL.LOAD_CHECKPOINT_EPOCH == "last_on_train":
            all_checkpoints = glob.glob(os.path.join(checkpoint_dir, "{}-checkpoint-*.pth".format(jobname)))
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split("-")[-1].split(".")[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                resume = os.path.join(checkpoint_dir, "{}-checkpoint-{}.pth".format(jobname, latest_ckpt))
        elif cfg.MODEL.LOAD_CHECKPOINT_EPOCH == "best_on_val":
            resume = os.path.join(checkpoint_dir, "{}-checkpoint-best.pth".format(jobname))
        else:
            raise NotImplementedError

    return resume


def load_model_checkpoint(cfg, jobname, model_without_ddp, device, optimizer=None, just_extract_checkpoint_info=False):
    start_epoch = 0

    resume = get_checkpoint_path(cfg, jobname)

    if not os.path.exists(resume):
        raise FileNotFoundError(f"Checkpoint file {resume} not found")
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
    else:
        checkpoint = torch.load(resume, map_location=device, weights_only=True)

    if just_extract_checkpoint_info:
        if "cfg" not in checkpoint:
            print(
                "Checkpoint seems to not be from BiaPy (v3.5.1 or later) as model building args couldn't be extracted. Thus, "
                "the model will be built based on the current configuration"
            )
        return (
            checkpoint["cfg"] if "cfg" in checkpoint else None,
            checkpoint["biapy_version"] if "biapy_version" in checkpoint else None,
        )

    model_without_ddp.load_state_dict(checkpoint["model"], strict=False)
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
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x


def to_pytorch_format(x: torch.Tensor | NDArray, axes_order: Tuple, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor) and torch.is_tensor(x):
        return x.to(dtype).permute(axes_order).to(device, non_blocking=True)
    else:
        return torch.from_numpy(x).to(dtype).permute(axes_order).to(device, non_blocking=True)


def to_numpy_format(x, axes_order_back):
    return x.permute(axes_order_back).cpu().numpy()


def time_text(t):
    if t >= 3600:
        return "{:.1f}h".format(t / 3600)
    elif t >= 60:
        return "{:.1f}m".format(t / 60)
    else:
        return "{:.1f}s".format(t)


class TensorboardLogger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(logdir=log_dir)
        self.step = 0

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, head="scalar", step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.writer.add_scalar(head + "/" + k, v, self.step if step is None else step)

    def flush(self):
        self.writer.flush()


class SmoothedValue(object):
    """
    Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt
        self.eps = sys.float_info.epsilon

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
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
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self) -> float:
        return self.total / (self.count + self.eps)

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):
    def __init__(self, delimiter="\t", verbose=False):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.verbose = verbose

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
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


def convert_old_model_cfg_to_current_version(keys_to_convert, values_to_check, biapy_old_version=None):
    new_cfg_list = []
    if biapy_old_version is None:
        print("There is no BiaPy version information in the checkpoint")
    else:
        print(f"Checkpoint in version: {biapy_old_version}")

    for k, v in zip(keys_to_convert, values_to_check):
        print(f"Trying to convert {k} key from old checkpoint")

        # BiaPy version less than 3.5.5
        if biapy_old_version is None:
            if k == "BATCH_NORMALIZATION" and v == True:
                new_cfg_list += ["MODEL.NORMALIZATION", "bn"]
            if k == "SOURCE_MODEL_DOI" and v != "":
                new_cfg_list += ["MODEL.BMZ.SOURCE_MODEL_ID", v]

    if len(new_cfg_list) > 0:
        print(f"Configuration to be translated: {new_cfg_list}")

    return new_cfg_list
