import torch
import math
import sys
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from typing import Callable, Optional
from torch.utils.data import DataLoader
from yacs.config import CfgNode as CN

from biapy.utils.misc import MetricLogger, SmoothedValue, TensorboardLogger, all_reduce_mean
from biapy.engine import Scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from biapy.engine.schedulers.warmup_cosine_decay import WarmUpCosineDecayScheduler


def train_one_epoch(
    cfg: CN,
    model: nn.Module | nn.parallel.DistributedDataParallel,
    model_call_func: Callable,
    loss_function: Callable,
    activations: Callable,
    metric_function: Callable,
    prepare_targets: Callable,
    data_loader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    epoch: int,
    log_writer: Optional[TensorboardLogger] = None,
    lr_scheduler: Optional[Scheduler] = None,
    verbose: bool = False,
):

    model.train(True)

    # Ensure correct order of each epoch info by adding loss first
    metric_logger = MetricLogger(delimiter="  ", verbose=verbose)
    metric_logger.add_meter("loss", SmoothedValue())

    header = "Epoch: [{}]".format(epoch + 1)
    print_freq = 10

    optimizer.zero_grad()

    for step, (batch, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # Apply warmup cosine decay scheduler if selected
        # (notice we use a per iteration (instead of per epoch) lr scheduler)
        if (
            epoch % cfg.TRAIN.ACCUM_ITER == 0
            and cfg.TRAIN.LR_SCHEDULER.NAME == "warmupcosine"
            and lr_scheduler
            and isinstance(lr_scheduler, WarmUpCosineDecayScheduler)
        ):
            lr_scheduler.adjust_learning_rate(optimizer, step / len(data_loader) + epoch)

        # Gather inputs
        targets = prepare_targets(targets, batch)

        if batch.shape[1:-1] != cfg.DATA.PATCH_SIZE[:-1]:
            raise ValueError(
                "Trying to input data with different shape than 'DATA.PATCH_SIZE'. Check your configuration."
                f" Input: {batch.shape[1:-1]} vs PATCH_SIZE: {cfg.DATA.PATCH_SIZE[:-1]}"
            )

        # Pass the images through the model
        # TODO: control autocast and mixed precision
        outputs = activations(model_call_func(batch, is_train=True), training=True)
        loss = loss_function(outputs, targets)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # Calculate the metrics
        metric_function(outputs, targets, metric_logger=metric_logger)

        # Forward pass scaling the loss
        loss /= cfg.TRAIN.ACCUM_ITER
        if (step + 1) % cfg.TRAIN.ACCUM_ITER == 0:
            loss.backward()
            optimizer.step()  # update weight
            optimizer.zero_grad()
            if lr_scheduler and isinstance(lr_scheduler, OneCycleLR) and cfg.TRAIN.LR_SCHEDULER.NAME == "onecycle":
                lr_scheduler.step()

        if device.type != "cpu":
            getattr(torch, device.type).synchronize()

        # Update loss in loggers
        metric_logger.update(loss=loss_value)
        loss_value_reduce = all_reduce_mean(loss_value)
        if log_writer:
            log_writer.update(loss=loss_value_reduce, head="loss")

        # Update lr in loggers
        max_lr = 0.0
        for group in optimizer.param_groups:
            max_lr = max(max_lr, group["lr"])
        if step == 0:
            metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.update(lr=max_lr)
        if log_writer:
            log_writer.update(lr=max_lr, head="opt")

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("[Train] averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    cfg: CN,
    model: nn.Module | nn.parallel.DistributedDataParallel,
    model_call_func: Callable,
    loss_function: Callable,
    activations: Callable,
    metric_function: Callable,
    prepare_targets: Callable,
    epoch: int,
    data_loader: DataLoader,
    lr_scheduler: Optional[Scheduler] = None,
):

    # Ensure correct order of each epoch info by adding loss first
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("loss", SmoothedValue())
    header = "Epoch: [{}]".format(epoch + 1)

    # Switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        # Gather inputs
        images = batch[0]
        targets = batch[1]
        targets = prepare_targets(targets, images)

        # Pass the images through the model
        outputs = activations(model_call_func(images, is_train=True), training=True)
        loss = loss_function(outputs, targets)

        # Calculate the metrics
        metric_function(outputs, targets, metric_logger=metric_logger)

        metric_logger.update(loss=loss.item())

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print("[Val] averaged stats:", metric_logger)

    # Apply reduceonplateau scheduler if the global validation has been reduced
    if (
        lr_scheduler
        and isinstance(lr_scheduler, ReduceLROnPlateau)
        and cfg.TRAIN.LR_SCHEDULER.NAME == "reduceonplateau"
    ):
        lr_scheduler.step(metric_logger.meters["loss"].global_avg, epoch=epoch)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
