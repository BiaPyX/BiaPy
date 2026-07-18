### Adapted from cellpose.train (https://github.com/MouseLand/cellpose/blob/main/cellpose/train.py)
from torch.optim.optimizer import Optimizer
import numpy as np


class WarmUpReduceOnPlateauScheduler:

    def __init__(
        self,
        lr: float,
        epochs: int,
    ):
        self.lr = lr
        self.epochs = epochs
        self.LR = self._build_schedule(lr, epochs)

    @staticmethod
    def _build_schedule(
        learning_rate: float,
        n_epochs: int,
    ) -> np.ndarray:
        LR = np.linspace(0, learning_rate, 10)
        LR = np.append(LR, learning_rate * np.ones(max(0, n_epochs - 10)))
        if n_epochs > 300:
            LR = LR[:-100]
            for _ in range(10):
                LR = np.append(LR, LR[-1] / 2 * np.ones(10))
        elif n_epochs > 100:
            LR = LR[:-50]
            for _ in range(10):
                LR = np.append(LR, LR[-1] / 2 * np.ones(5))
        return LR

    def adjust_learning_rate(
        self,
        optimizer: Optimizer,
        epoch: float | int,
    ) -> float:
        idx = min(int(epoch), len(self.LR) - 1)
        lr = float(self.LR[idx])
        for param_group in optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr
        return lr
