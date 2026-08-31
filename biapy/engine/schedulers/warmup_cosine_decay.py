"""
Learning rate scheduler with warmup and cosine decay for BiaPy.

This module provides the WarmUpCosineDecayScheduler class, which implements a learning
rate schedule with a linear warmup phase followed by cosine decay, as commonly used
in modern deep learning training pipelines.
"""
### Adapted from https://github.com/facebookresearch/mae/blob/main/util/lr_sched.py
from torch.optim.optimizer import Optimizer
import math


class WarmUpCosineDecayScheduler:
    """
    Learning rate scheduler with linear warmup, an optional flat hold, and cosine decay.

    Three phases, in order: 1) linear ramp from 0 to ``lr`` over ``warmup_epochs`` epochs
    (skipped if ``warmup_epochs == 0``); 2) held flat at ``lr`` until ``decay_start_epoch``
    (skipped if ``decay_start_epoch == warmup_epochs``); 3) half-cycle cosine decay down to
    ``min_lr`` over the remaining epochs. ``TRAIN.LR_SCHEDULER.WARMUP_COSINE_DECAY_EPOCHS``
    drives phase 1 and ``TRAIN.LR_SCHEDULER.COSINE_DECAY_FRACTION`` drives when phase 3
    starts ("delayed cosine decay") -- both are optional and combinable.
    """

    def __init__(
        self,
        lr: float,
        min_lr: float,
        warmup_epochs: int,
        decay_start_epoch: int,
        epochs: int,
    ):
        """
        Initialize the WarmUpCosineDecayScheduler.

        Parameters
        ----------
        lr : float
            Initial (maximum) learning rate.
        min_lr : float
            Minimum learning rate after decay.
        warmup_epochs : int
            Number of epochs spent linearly ramping up from 0 to ``lr``. 0 disables the ramp.
        decay_start_epoch : int
            Epoch at which cosine decay starts. Must be >= ``warmup_epochs``; the epochs in
            between are held flat at ``lr``. Equal to ``warmup_epochs`` disables the hold.
        epochs : int
            Total number of training epochs.
        """
        self.lr = lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.decay_start_epoch = decay_start_epoch
        self.epochs = epochs

    def adjust_learning_rate(
        self,
        optimizer: Optimizer,
        epoch: float | int
    ) -> float:
        """
        Ramp up, hold, then decay the learning rate with half-cycle cosine.

        Parameters
        ----------
        optimizer : Optimizer
            PyTorch optimizer whose learning rate will be adjusted.
        epoch : float or int
            Current epoch (can be fractional for finer granularity).

        Returns
        -------
        lr : float
            The adjusted learning rate.
        """
        if epoch < self.warmup_epochs:
            lr = self.lr * epoch / self.warmup_epochs
        elif epoch < self.decay_start_epoch:
            lr = self.lr
        else:
            lr = self.min_lr + (self.lr - self.min_lr) * 0.5 * (
                1.0 + math.cos(math.pi * (epoch - self.decay_start_epoch) / (self.epochs - self.decay_start_epoch))
            )
        for param_group in optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr
        return lr
