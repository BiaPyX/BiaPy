"""
Training and evaluation engine for BiaPy.

This module provides functions to train and evaluate deep learning models for
one epoch, handling distributed training, logging, learning rate scheduling,
and memory bank operations for contrastive/self-supervised learning.
"""
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
from biapy.models.memory_bank import MemoryBank


def train_one_epoch(
    cfg: CN,
    model: nn.Module | nn.parallel.DistributedDataParallel,
    model_call_func: Callable,
    loss_function: Callable,
    metric_function: Callable,
    prepare_targets: Callable,
    data_loader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    epoch: int,
    log_writer: Optional[TensorboardLogger] = None,
    lr_scheduler: Optional[Scheduler] = None,
    verbose: bool = False,
    memory_bank: Optional[MemoryBank] = None,
    total_iters: int=0,
    contrast_warmup_iters: int=0,
):
    """
    Train the model for one epoch.

    Handles forward and backward passes, loss computation, metric logging,
    optimizer steps, learning rate scheduling, and optional memory bank updates.

    Parameters
    ----------
    cfg : CN
        BiaPy configuration node.
    model : nn.Module or nn.parallel.DistributedDataParallel
        Model to train.
    model_call_func : Callable
        Function to call the model (handles multi-heads, etc.).
    loss_function : Callable
        Loss function.
    metric_function : Callable
        Metric computation function.
    prepare_targets : Callable
        Function to prepare targets for loss/metrics.
    data_loader : DataLoader
        Training data loader.
    optimizer : Optimizer
        Optimizer for model parameters.
    device : torch.device
        Device to use.
    epoch : int
        Current epoch number.
    log_writer : TensorboardLogger, optional
        Logger for TensorBoard.
    lr_scheduler : Scheduler, optional
        Learning rate scheduler.
    verbose : bool, optional
        Verbosity flag.
    memory_bank : MemoryBank, optional
        Memory bank for contrastive/self-supervised learning.
    total_iters : int, optional
        Total iterations completed (for contrastive warmup).
    contrast_warmup_iters : int, optional
        Number of warmup iterations for contrastive learning.

    Returns
    -------
    dict
        Dictionary of averaged metrics for the epoch.
    int
        Number of steps (batches) processed.
    """
    # Switch to training mode
    model.train(True)

    # Ensure correct order of each epoch info by adding loss first
    metric_logger = MetricLogger(delimiter="  ", verbose=verbose)
    metric_logger.add_meter("loss", SmoothedValue())

    # Set up the header for logging
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
        outputs = model_call_func(batch, is_train=True)

        # Loss function call
        if memory_bank is not None:
            if total_iters + step >= contrast_warmup_iters:
                with_embed = True
            else:
                with_embed = False

            outputs = {
                "pred": outputs["pred"],
                "embed": outputs["embed"],
                'key': outputs["embed"].detach(),
                'pixel_queue': memory_bank.pixel_queue,
                'segment_queue': memory_bank.segment_queue,
            }

            loss = loss_function(outputs, targets, with_embed=with_embed)

            memory_bank.dequeue_and_enqueue(
                outputs['key'], targets.detach(),
            )
        else:
            loss = loss_function(outputs, targets)

        # Separate metric if precalculated inside the loss (e.g. Embedding loss)
        precalculated_metric, precalculated_metric_name = None, None
        if isinstance(loss, tuple):
            precalculated_metric = loss[1]
            precalculated_metric_name = loss[2]
            loss = loss[0]

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # Calculate the metrics
        if precalculated_metric is None:
            metric_function(outputs, targets, metric_logger=metric_logger)
        else:
            metric_logger.meters[precalculated_metric_name].update(precalculated_metric)

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
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, step


@torch.no_grad()
def evaluate(
    cfg: CN,
    model: nn.Module | nn.parallel.DistributedDataParallel,
    model_call_func: Callable,
    loss_function: Callable,
    metric_function: Callable,
    prepare_targets: Callable,
    epoch: int,
    data_loader: DataLoader,
    lr_scheduler: Optional[Scheduler] = None,
    memory_bank: Optional[MemoryBank] = None,
):
    """
    Evaluate the model on the validation set.

    Runs the model in evaluation mode, computes loss and metrics, and updates
    learning rate scheduler if needed.

    Parameters
    ----------
    cfg : CN
        BiaPy configuration node.
    model : nn.Module or nn.parallel.DistributedDataParallel
        Model to evaluate.
    model_call_func : Callable
        Function to call the model.
    loss_function : Callable
        Loss function.
    metric_function : Callable
        Metric computation function.
    prepare_targets : Callable
        Function to prepare targets for loss/metrics.
    epoch : int
        Current epoch number.
    data_loader : DataLoader
        Validation data loader.
    lr_scheduler : Scheduler, optional
        Learning rate scheduler.
    memory_bank : MemoryBank, optional
        Memory bank for contrastive/self-supervised learning.

    Returns
    -------
    dict
        Dictionary of averaged metrics for the validation set.
    """
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
        outputs = model_call_func(images, is_train=True)
        
        # Loss function call
        if memory_bank is not None:
            with_embed = False

            outputs = {
                "pred": outputs["pred"],
                "embed": outputs["embed"],
                'key': outputs["pred"].detach(),
                'pixel_queue': memory_bank.pixel_queue,
                'segment_queue': memory_bank.segment_queue,
            }

            loss = loss_function(outputs, targets, with_embed=with_embed)
        else:
            loss = loss_function(outputs, targets)

        # Separate metric if precalculated inside the loss (e.g. Embedding loss)
        precalculated_metric, precalculated_metric_name = None, None
        if isinstance(loss, tuple):
            precalculated_metric = loss[1]
            precalculated_metric_name = loss[2]
            loss = loss[0]
        
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # Calculate the metrics
        if precalculated_metric is not None:
            metric_logger.meters[precalculated_metric_name].update(precalculated_metric)
        else:
            metric_function(outputs, targets, metric_logger=metric_logger)

        # Update loss in loggers
        metric_logger.update(loss=loss)

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
