# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import torch
import math
import sys
import numpy as np
from typing import Iterable
from timm.utils import accuracy

from utils.misc import MetricLogger, SmoothedValue, all_reduce_mean, to_pytorch_format

def train_one_epoch(cfg, model, loss_function, activations, metric_function, prepare_targets, data_loader, optimizer, 
    device, loss_scaler, epoch, log_writer=None, lr_scheduler=None, start_steps=0, axis_order=(0,3,1,2)):

    model.train(True)
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch+1)
    print_freq = 10

    optimizer.zero_grad()
                        
    for step, (batch, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        it = start_steps + step  # global training iteration

        # Gather inputs
        batch = to_pytorch_format(batch, axis_order, device)
        targets = prepare_targets(targets, batch)

        # Pass the images through the model
        # TODO: control autocast and mixed precision
        with torch.cuda.amp.autocast(enabled=False):
            outputs = activations(model(batch))
            loss = loss_function(outputs, targets)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # Calculate the metrics
        metric_function(outputs, targets, device, metric_logger)

        # Forward pass scaling the loss
        loss /= cfg.TRAIN.ACCUM_ITER
        if (step + 1) % cfg.TRAIN.ACCUM_ITER == 0:
            loss.backward()
            optimizer.step() #update weight        
            optimizer.zero_grad()
            if lr_scheduler is not None and cfg.TRAIN.LR_SCHEDULER.NAME == 'onecycle':
                lr_scheduler.step() 

        torch.cuda.synchronize()

        # Update metrics 
        metric_logger.update(loss=loss_value)
        max_lr = 0.
        for group in optimizer.param_groups:
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)

        # Update logging file writer
        loss_value_reduce = all_reduce_mean(loss_value)
        if log_writer is not None:
            log_writer.update(loss=loss_value_reduce, head="loss")
            if lr_scheduler is not None:
                log_writer.update(lr=max_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("[Train] averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(cfg, model, loss_function, activations, metric_function, prepare_targets, device, epoch, 
    data_loader, lr_scheduler, axis_order=(0,3,1,2)):
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch+1)

    # Switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        # Gather inputs
        images = batch[0]
        targets = batch[1]
        images = to_pytorch_format(images, axis_order, device)
        targets = prepare_targets(targets, images)

        # Pass the images through the model
        # TODO: control autocast and mixed precision
        with torch.cuda.amp.autocast(enabled=False):  
            outputs = activations(model(images))
            loss = loss_function(outputs, targets)
        
        # Calculate the metrics
        metric_function(outputs, targets, device, metric_logger)
    
        metric_logger.update(loss=loss.item())

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print("[Val] averaged stats:", metric_logger)

    # Apply reduceonplateau scheduler if the global validation has been reduced
    if lr_scheduler is not None and cfg.TRAIN.LR_SCHEDULER.NAME == 'reduceonplateau':
        lr_scheduler.step(metric_logger.meters['loss'].global_avg)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
