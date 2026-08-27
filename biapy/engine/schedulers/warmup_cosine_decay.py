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
    Learning rate scheduler with a pre-decay phase and cosine decay.

    This scheduler keeps the learning rate on a fixed pattern for a number of epochs,
    then decays it following a half-cycle cosine schedule down to a minimum learning rate.
    The pre-decay phase is either a linear warmup from 0 (``hold_lr=False``, driven by
    ``TRAIN.LR_SCHEDULER.WARMUP_COSINE_DECAY_EPOCHS``) or a constant plateau at ``lr``
    (``hold_lr=True``, "delayed cosine decay", driven by ``TRAIN.LR_SCHEDULER.COSINE_DECAY_FRACTION``).
    Both are selected via ``TRAIN.LR_SCHEDULER.NAME == "warmupcosine"``.
    """

    def __init__(
        self,
        lr: float,
        min_lr: float,
        warmup_epochs: int,
        epochs: int,
        hold_lr: bool = False,
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
            Number of epochs before cosine decay starts (linear warmup, or constant hold
            when ``hold_lr=True``).
        epochs : int
            Total number of training epochs.
        hold_lr : bool, optional
            If True, keep the learning rate constant at ``lr`` before decay starts instead
            of linearly ramping it up from 0 ("delayed cosine decay").
        """
        self.lr = lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.hold_lr = hold_lr

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
            lr = self.lr if self.hold_lr else self.lr * epoch / self.warmup_epochs
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
