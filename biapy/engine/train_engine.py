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
    model_d: Optional[nn.Module | nn.parallel.DistributedDataParallel] = None,
    optimizer_d: Optional[Optimizer] = None,
    lr_scheduler_d: Optional[Scheduler] = None,
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
    is_gan = model_d is not None and optimizer_d is not None

    # Switch to training mode
    model.train(True)
    if is_gan:
        model_d.train(True)

    # Ensure correct order of each epoch info by adding loss first
    metric_logger = MetricLogger(delimiter="  ", verbose=verbose)
    if is_gan:
        metric_logger.add_meter("loss_g", SmoothedValue())
        metric_logger.add_meter("loss_d", SmoothedValue())
    else:
        metric_logger.add_meter("loss", SmoothedValue())

    # Set up the header for logging
    header = "Epoch: [{}]".format(epoch + 1)
    print_freq = 10

    optimizer.zero_grad()
    if is_gan:
        optimizer_d.zero_grad()

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
            if is_gan and lr_scheduler_d and isinstance(lr_scheduler_d, WarmUpCosineDecayScheduler):
                lr_scheduler_d.adjust_learning_rate(optimizer_d, step / len(data_loader) + epoch)


        # Gather inputs
        targets = prepare_targets(targets, batch)

        if batch.shape[1:-1] != cfg.DATA.PATCH_SIZE[:-1]:
            raise ValueError(
                "Trying to input data with different shape than 'DATA.PATCH_SIZE'. Check your configuration."
                f" Input: {batch.shape[1:-1]} vs PATCH_SIZE: {cfg.DATA.PATCH_SIZE[:-1]}"
            )

        if is_gan:
            assert model_d is not None and optimizer_d is not None

            if (
                torch.isnan(batch).any()
                or torch.isinf(batch).any()
                or torch.isnan(targets).any()
                or torch.isinf(targets).any()
            ):
                print("Warning: NaN or Inf detected in input. Skipping batch.")
                continue

            # Phase 1: discriminator update
            optimizer_d.zero_grad()
            fake_img = model_call_func(batch, is_train=True)
            if isinstance(fake_img, dict):
                fake_img = fake_img["pred"]
            fake_img = torch.clamp(fake_img, 0, 1)

            d_real = model_d(targets)
            d_fake = model_d(fake_img.detach())
            loss_d = loss_function.forward_discriminator(d_real, d_fake)

            if torch.isnan(loss_d) or torch.isinf(loss_d):
                print("Warning: NaN or Inf detected in discriminator loss. Skipping batch.")
                continue

            loss_d.backward()
            optimizer_d.step()

            if lr_scheduler_d and isinstance(lr_scheduler_d, OneCycleLR) and cfg.TRAIN.LR_SCHEDULER.NAME == "onecycle":
                lr_scheduler_d.step()

            # Phase 2: generator update
            optimizer.zero_grad()
            outputs = model_call_func(batch, is_train=True)
            if isinstance(outputs, dict):
                outputs = outputs["pred"]
            outputs = torch.clamp(outputs, 0, 1)

            d_fake_for_g = model_d(outputs)
            loss = loss_function.forward_generator(outputs, targets, d_fake_for_g)

            if torch.isnan(loss) or torch.isinf(loss):
                print("Warning: NaN or Inf detected in generator loss. Skipping batch.")
                continue

            loss.backward()
            optimizer.step()

            if lr_scheduler and isinstance(lr_scheduler, OneCycleLR) and cfg.TRAIN.LR_SCHEDULER.NAME == "onecycle":
                lr_scheduler.step()

            metric_function(outputs, targets, metric_logger=metric_logger)

            loss_g_value = loss.item()
            loss_d_value = loss_d.item()
            metric_logger.update(loss_g=loss_g_value, loss_d=loss_d_value)

            if log_writer:
                log_writer.update(loss_g=all_reduce_mean(loss_g_value), head="loss")
                log_writer.update(loss_d=all_reduce_mean(loss_d_value), head="loss")

            max_lr_g = 0.0
            max_lr_d = 0.0
            for group in optimizer.param_groups:
                max_lr_g = max(max_lr_g, group["lr"])
            for group in optimizer_d.param_groups:
                max_lr_d = max(max_lr_d, group["lr"])

            if step == 0:
                metric_logger.add_meter("lr_g", SmoothedValue(window_size=1, fmt="{value:.6f}"))
                metric_logger.add_meter("lr_d", SmoothedValue(window_size=1, fmt="{value:.6f}"))

            metric_logger.update(lr_g=max_lr_g, lr_d=max_lr_d)
            if log_writer:
                log_writer.update(lr_g=max_lr_g, head="opt")
                log_writer.update(lr_d=max_lr_d, head="opt")
            continue

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

    if is_gan and cfg.TRAIN.LR_SCHEDULER.NAME not in ["reduceonplateau", "onecycle", "warmupcosine"]:
        if lr_scheduler:
            lr_scheduler.step()
        if lr_scheduler_d:
            lr_scheduler_d.step()

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
    model_d: Optional[nn.Module | nn.parallel.DistributedDataParallel] = None,
    lr_scheduler_d: Optional[Scheduler] = None,
    device: Optional[torch.device] = None,
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
    is_gan = model_d is not None

    # Ensure correct order of each epoch info by adding loss first
    metric_logger = MetricLogger(delimiter="  ")
    if is_gan:
        metric_logger.add_meter("loss_g", SmoothedValue())
        metric_logger.add_meter("loss_d", SmoothedValue())
    else:
        metric_logger.add_meter("loss", SmoothedValue())
    header = "Epoch: [{}]".format(epoch + 1)

    # Switch to evaluation mode
    model.eval()
    if is_gan:
        model_d.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        # Gather inputs
        images = batch[0]
        targets = batch[1]
        targets = prepare_targets(targets, images)

        if is_gan:
            assert model_d is not None
            outputs = model_call_func(images, is_train=False)
            if isinstance(outputs, dict):
                outputs = outputs["pred"]
            outputs = torch.clamp(outputs, 0, 1)

            d_fake_val = model_d(outputs)
            d_real_val = model_d(targets)
            loss = loss_function.forward_generator(outputs, targets, d_fake_val)
            loss_d = loss_function.forward_discriminator(d_real_val, d_fake_val)

            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print(f"Validation loss is {loss_value}, skipping batch.")
                continue

            metric_function(outputs, targets, metric_logger=metric_logger)
            metric_logger.update(loss_g=loss_value, loss_d=loss_d.item())
            continue

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
    if cfg.TRAIN.LR_SCHEDULER.NAME == "reduceonplateau":
        if is_gan:
            if lr_scheduler and isinstance(lr_scheduler, ReduceLROnPlateau):
                lr_scheduler.step(metric_logger.meters["loss_g"].global_avg, epoch=epoch)
            if lr_scheduler_d and isinstance(lr_scheduler_d, ReduceLROnPlateau):
                lr_scheduler_d.step(metric_logger.meters["loss_d"].global_avg, epoch=epoch)
        elif lr_scheduler and isinstance(lr_scheduler, ReduceLROnPlateau):
            lr_scheduler.step(metric_logger.meters["loss"].global_avg, epoch=epoch)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
