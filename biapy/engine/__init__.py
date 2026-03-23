"""
BiaPy engine package.

This package contains core workflow classes, training and evaluation engines,
metrics, and learning rate schedulers for deep learning pipelines in BiaPy.
"""
import timm.optim
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from torch import nn
from yacs.config import CfgNode as CN
from typing import Tuple, Union

from biapy.engine.schedulers.warmup_cosine_decay import WarmUpCosineDecayScheduler
from biapy.utils.callbacks import EarlyStopping

Scheduler = Union[ReduceLROnPlateau, WarmUpCosineDecayScheduler, OneCycleLR]


def prepare_optimizer(
    cfg: CN,
    model_without_ddp: nn.Module | nn.parallel.DistributedDataParallel,
    steps_per_epoch: int,
    is_gan: bool = False,
) -> Tuple[Optimizer, Scheduler | None]:
    """
    Create and configure the optimizer and learning rate scheduler for the given model.

    This function selects and initializes the optimizer (e.g., Adam, AdamW) and, if specified,
    the learning rate scheduler (ReduceLROnPlateau, WarmUpCosineDecayScheduler, or OneCycleLR)
    based on the configuration.

    Parameters
    ----------
    cfg : YACS CN object
        Configuration object with optimizer and scheduler settings.
    model_without_ddp : nn.Module or nn.parallel.DistributedDataParallel or dict
        The model to optimize.
    steps_per_epoch : int
        Number of steps (batches) per training epoch.
    is_gan : bool, optional
        Whether to create optimizer/scheduler pairs for GAN generator and discriminator.

    Returns
    -------
    optimizer : Optimizer or dict
        Configured optimizer for the model or dict with generator/discriminator optimizers in GAN mode.
    lr_scheduler : Scheduler or None or dict
        Configured scheduler for the model or dict with generator/discriminator schedulers in GAN mode.
    """
    def _make_scheduler(optimizer: Optimizer, lr_value: float) -> Scheduler | None:
        lr_scheduler = None
        if cfg.TRAIN.LR_SCHEDULER.NAME != "":
            if cfg.TRAIN.LR_SCHEDULER.NAME == "reduceonplateau":
                lr_scheduler = ReduceLROnPlateau(
                    optimizer,
                    patience=cfg.TRAIN.LR_SCHEDULER.REDUCEONPLATEAU_PATIENCE,
                    factor=cfg.TRAIN.LR_SCHEDULER.REDUCEONPLATEAU_FACTOR,
                    min_lr=cfg.TRAIN.LR_SCHEDULER.MIN_LR,
                )
            elif cfg.TRAIN.LR_SCHEDULER.NAME == "warmupcosine":
                lr_scheduler = WarmUpCosineDecayScheduler(
                    lr=lr_value,
                    min_lr=cfg.TRAIN.LR_SCHEDULER.MIN_LR,
                    warmup_epochs=cfg.TRAIN.LR_SCHEDULER.WARMUP_COSINE_DECAY_EPOCHS,
                    epochs=cfg.TRAIN.EPOCHS,
                )
            elif cfg.TRAIN.LR_SCHEDULER.NAME == "onecycle":
                lr_scheduler = OneCycleLR(
                    optimizer,
                    lr_value,
                    epochs=cfg.TRAIN.EPOCHS,
                    steps_per_epoch=steps_per_epoch,
                )
        return lr_scheduler

    def _make_optimizer(model: nn.Module | nn.parallel.DistributedDataParallel, train_cfg: dict):
        lr_value = train_cfg["lr"]
        opt_name = train_cfg["optimizer"]
        betas = train_cfg["betas"]
        w_decay = train_cfg["weight_decay"]

        lr = lr_value if cfg.TRAIN.LR_SCHEDULER.NAME != "warmupcosine" else cfg.TRAIN.LR_SCHEDULER.MIN_LR
        opt_args = {}
        if opt_name in ["ADAM", "ADAMW"]:
            opt_args["betas"] = betas

        optimizer = timm.optim.create_optimizer_v2(
            model,
            opt=opt_name,
            lr=lr,
            weight_decay=w_decay,
            **opt_args,
        )
        print(optimizer)
        lr_scheduler = _make_scheduler(optimizer, lr_value)
        return optimizer, lr_scheduler

    g_train_cfg = {
        "lr": cfg.TRAIN.LR,
        "optimizer": cfg.TRAIN.OPTIMIZER,
        "betas": cfg.TRAIN.OPT_BETAS,
        "weight_decay": cfg.TRAIN.W_DECAY,
    }

    if not is_gan:
        return _make_optimizer(model_without_ddp, g_train_cfg)

    d_train_cfg = {
        "lr": cfg.TRAIN.LR_D,
        "optimizer": cfg.TRAIN.OPTIMIZER_D,
        "betas": cfg.TRAIN.OPT_BETAS_D,
        "weight_decay": cfg.TRAIN.W_DECAY,
    }

    optimizer_g, scheduler_g = _make_optimizer(model_without_ddp["generator"], g_train_cfg)
    optimizer_d, scheduler_d = _make_optimizer(model_without_ddp["discriminator"], d_train_cfg)

    return {"generator": optimizer_g, "discriminator": optimizer_d}, {"generator": scheduler_g, "discriminator": scheduler_d,}


def build_callbacks(cfg: CN) -> EarlyStopping | None:
    """
    Create training and validation generators.

    Parameters
    ----------
    cfg : YACS CN object
        Configuration.

    Returns
    -------
    callbacks : List of callbacks
        All callbacks to be applied to a model.
    """
    # Stop early and restore the best model weights when finished the training
    earlystopper = None
    if cfg.TRAIN.PATIENCE != -1:
        earlystopper = EarlyStopping(patience=cfg.TRAIN.PATIENCE)

    # if cfg.TRAIN.PROFILER:
    #     tb_callback = tf.keras.callbacks.TensorBoard(log_dir=cfg.PATHS.PROFILER, profile_batch=cfg.TRAIN.PROFILER_BATCH_RANGE)

    return earlystopper
