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
    Learning rate scheduler with linear warmup and cosine decay.

    This scheduler increases the learning rate linearly for a specified number of warmup epochs,
    then decays it following a half-cycle cosine schedule down to a minimum learning rate.
    """

    def __init__(
        self, 
        lr: float, 
        min_lr: float, 
        warmup_epochs: int, 
        epochs: int
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
            Number of epochs for linear warmup.
        epochs : int
            Total number of training epochs.
        """
        self.lr = lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs

    def adjust_learning_rate(
        self, 
        optimizer: Optimizer, 
        epoch: float | int
    ) -> float:
        """
        Decay the learning rate with half-cycle cosine after warmup.

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
        else:
            lr = self.min_lr + (self.lr - self.min_lr) * 0.5 * (
                1.0 + math.cos(math.pi * (epoch - self.warmup_epochs) / (self.epochs - self.warmup_epochs))
            )
        for param_group in optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr
        return lr
