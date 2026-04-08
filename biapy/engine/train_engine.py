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
    optimizer: list[Optimizer],
    device: torch.device,
    epoch: int,
    log_writer: Optional[TensorboardLogger] = None,
    lr_scheduler: Optional[list[Optional[Scheduler]]] = None,
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
    optimizer : List[Optimizer]
        Optimizer for model parameters.
    device : torch.device
        Device to use.
    epoch : int
        Current epoch number.
    log_writer : TensorboardLogger, optional
        Logger for TensorBoard.
    lr_scheduler : List[Scheduler], optional
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
    has_discriminator = hasattr(model, "discriminator") and model.discriminator is not None
    lr_scheduler = [None] * len(optimizer) if lr_scheduler is None else lr_scheduler

    # Ensure correct order of each epoch info by adding loss first
    metric_logger = MetricLogger(delimiter="  ", verbose=verbose)
    for i in range(len(optimizer)):
        loss_name = "loss" if i == 0 else f"loss_{i}"
        metric_logger.add_meter(loss_name, SmoothedValue())

    # Set up the header for logging
    header = "Epoch: [{}]".format(epoch + 1)
    print_freq = 10

    for opt in optimizer:
        opt.zero_grad()

    for step, (batch, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # Apply warmup cosine decay scheduler if selected
        # (notice we use a per iteration (instead of per epoch) lr scheduler)
        if (
            epoch % cfg.TRAIN.ACCUM_ITER == 0
            and cfg.TRAIN.LR_SCHEDULER.NAME == "warmupcosine"
        ):
            for sched, opt in zip(lr_scheduler, optimizer):
                if sched and isinstance(sched, WarmUpCosineDecayScheduler):
                    sched.adjust_learning_rate(opt, step / len(data_loader) + epoch)

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
        losses = []
        if has_discriminator and len(optimizer) > 1:
            fake_img = outputs["pred"] if isinstance(outputs, dict) else outputs
            fake_img = torch.clamp(fake_img, 0, 1)

            d_fake_for_g = model.discriminator(fake_img)
            loss_g = loss_function.forward_generator(fake_img, targets, d_fake_for_g)
            losses.append(loss_g)

            d_real = model.discriminator(targets)
            d_fake = model.discriminator(fake_img.detach())
            loss_d = loss_function.forward_discriminator(d_real, d_fake)
            losses.append(loss_d)
            
        elif memory_bank is not None:
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
            losses.append(loss)
        else:
            loss = loss_function(outputs, targets)
            losses.append(loss)

        # Separate metric if precalculated inside the loss (e.g. Embedding loss)
        precalculated_metric, precalculated_metric_name = None, None
        if isinstance(losses[0], tuple):
            precalculated_metric = losses[0][1]
            precalculated_metric_name = losses[0][2]
            losses[0] = losses[0][0]

        for l_val in losses:
            loss_value = l_val.item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

        # Calculate the metrics
        if precalculated_metric is None:
            metric_function(outputs, targets, metric_logger=metric_logger)
        else:
            metric_logger.meters[precalculated_metric_name].update(precalculated_metric)

        # Forward pass scaling the loss
        if (step + 1) % cfg.TRAIN.ACCUM_ITER == 0:
            for i, (opt, loss_tensor) in enumerate(zip(optimizer, losses)):
                loss_tensor = loss_tensor / cfg.TRAIN.ACCUM_ITER

                loss_tensor.backward()
                opt.step()  # update weight
                opt.zero_grad()
                
                if lr_scheduler[i] and isinstance(lr_scheduler[i], OneCycleLR) and cfg.TRAIN.LR_SCHEDULER.NAME == "onecycle":
                    lr_scheduler[i].step()

        if device.type != "cpu":
            getattr(torch, device.type).synchronize()

        # Update loss in loggers
        for i, loss_tensor in enumerate(losses):
            loss_name = "loss" if i == 0 else f"loss_{i}"
            val = loss_tensor.item() * cfg.TRAIN.ACCUM_ITER
            metric_logger.update(**{loss_name: val})
            loss_value_reduce = all_reduce_mean(val)
            if log_writer:
                log_writer.update(head="loss", **{loss_name: loss_value_reduce})

        # Update lr in loggers
        for i, opt in enumerate(optimizer):
            lr_name = "lr" if i == 0 else f"lr_{i}"
            max_lr = 0.0
            for group in opt.param_groups:
                max_lr = max(max_lr, group["lr"])
            if step == 0:
                metric_logger.add_meter(lr_name, SmoothedValue(window_size=1, fmt="{value:.6f}"))
            metric_logger.update(**{lr_name: max_lr})
            if log_writer:
                log_writer.update(head="opt", **{lr_name: max_lr})

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
    lr_scheduler: Optional[list[Optional[Scheduler]]] = None,
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
    has_discriminator = hasattr(model, "discriminator") and model.discriminator is not None
    # Ensure correct order of each epoch info by adding loss first
    metric_logger = MetricLogger(delimiter="  ")
    num_losses = 2 if has_discriminator and len(lr_scheduler) > 1 else 1
    for i in range(num_losses):
        loss_name = "loss" if i == 0 else f"loss_{i}"
        metric_logger.add_meter(loss_name, SmoothedValue())
    header = "Epoch: [{}]".format(epoch + 1)

    # Switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        # Gather inputs
        images = batch[0]
        targets = batch[1]
        targets = prepare_targets(targets, images)

        # Pass the images through the model
        outputs = model_call_func(images, is_train=True) # Im not Undertanding why is this True? 
        
        # Loss function call
        losses = []
        if has_discriminator and len(lr_scheduler) > 1:
            fake_img = outputs["pred"] if isinstance(outputs, dict) else outputs
            fake_img = torch.clamp(fake_img, 0, 1)

            d_fake_for_g = model.discriminator(fake_img)
            loss_g = loss_function.forward_generator(fake_img, targets, d_fake_for_g)
            losses.append(loss_g)

            d_real = model.discriminator(targets)
            d_fake = model.discriminator(fake_img.detach())
            loss_d = loss_function.forward_discriminator(d_real, d_fake)
            losses.append(loss_d)
            
        elif memory_bank is not None:
            outputs = {
                "pred": outputs["pred"],
                "embed": outputs["embed"],
                'key': outputs["pred"].detach(),
                'pixel_queue': memory_bank.pixel_queue,
                'segment_queue': memory_bank.segment_queue,
            }
            loss = loss_function(outputs, targets, with_embed=with_embed)
            losses.append(loss)
        else:
            loss = loss_function(outputs, targets)
            losses.append(loss)

        # Separate metric if precalculated inside the loss (e.g. Embedding loss)
        precalculated_metric, precalculated_metric_name = None, None
        if isinstance(losses[0], tuple):
            precalculated_metric = losses[0][1]
            precalculated_metric_name = losses[0][2]
            losses[0] = losses[0][0]

        for l_val in losses:
            loss_value = l_val.item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

        # Calculate the metrics
        if precalculated_metric is not None:
            metric_logger.meters[precalculated_metric_name].update(precalculated_metric)
        else:
            metric_function(outputs, targets, metric_logger=metric_logger)

        # Update loss in loggers
        for i, loss_tensor in enumerate(losses):
            loss_name = "loss" if i == 0 else f"loss_{i}"
            metric_logger.update(**{loss_name: loss_tensor.item()})

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print("[Val] averaged stats:", metric_logger)

    # Apply reduceonplateau scheduler if the global validation has been reduced
    if cfg.TRAIN.LR_SCHEDULER.NAME == "reduceonplateau":
        for i, sched in enumerate(lr_scheduler):
            if sched and isinstance(sched, ReduceLROnPlateau):
                loss_name = "loss" if i == 0 else f"loss_{i}"
                sched.step(metric_logger.meters[loss_name].global_avg, epoch=epoch)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
