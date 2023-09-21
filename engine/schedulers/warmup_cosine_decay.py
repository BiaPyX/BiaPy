### Adapted from https://github.com/facebookresearch/mae/blob/main/util/lr_sched.py

import math

class WarmUpCosineDecayScheduler:
    def __init__(self, lr, min_lr, warmup_epochs, epochs):
        self.lr = lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs

    def adjust_learning_rate(self, optimizer, epoch):
        """Decay the learning rate with half-cycle cosine after warmup"""
        if epoch < self.warmup_epochs:
            lr = self.lr * epoch / self.warmup_epochs 
        else:
            lr = self.min_lr + (self.lr - self.min_lr) * 0.5 * \
                (1. + math.cos(math.pi * (epoch - self.warmup_epochs) / (self.epochs - self.warmup_epochs)))
        for param_group in optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr
        return lr