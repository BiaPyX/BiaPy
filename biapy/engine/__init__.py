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
) -> Tuple[list[Optimizer], list[Scheduler | None]]:
    """
    Create and configure the optimizer and learning rate scheduler for the given model.

    This function selects and initializes the optimizer (e.g., Adam, AdamW) and, if specified,
    the learning rate scheduler (ReduceLROnPlateau, WarmUpCosineDecayScheduler, or OneCycleLR)
    based on the configuration.

    Parameters
    ----------
    cfg : YACS CN object
        Configuration object with optimizer and scheduler settings.
    model_without_ddp : nn.Module or nn.parallel.DistributedDataParallel
        The model to optimize.
    steps_per_epoch : int
        Number of steps (batches) per training epoch.

    Returns
    -------
    optimizer : List[Optimizer]
        Configured optimizers for the models.
    lr_scheduler : Scheduler or None
        Configured learning rate schedulers, or None if not specified.
    """

    optimizers = []
    lr_schedulers = []
    
    if hasattr(model_without_ddp, 'discriminator') and model_without_ddp.discriminator is not None:
        param_groups = [
            [p for n, p in model_without_ddp.named_parameters() if not n.startswith("discriminator.")], # Generator
            model_without_ddp.discriminator.parameters()                                                # Discriminator
        ]
    else:
        param_groups = [model_without_ddp.parameters()]

    ## Not quite sure if this is the best place to do this
    if len(cfg.TRAIN.OPTIMIZER) != len(param_groups):
        raise ValueError(
            f"Configuration mismatch: You requested {len(cfg.TRAIN.OPTIMIZER)} optimizers, "
            f"but the model has {len(param_groups)} parameter group(s). "
            f"Check your TRAIN.OPTIMIZER list in the config."
        )

    for i in range(len(cfg.TRAIN.OPTIMIZER)):
        lr = cfg.TRAIN.LR if cfg.TRAIN.LR_SCHEDULER.NAME != "warmupcosine" else cfg.TRAIN.LR_SCHEDULER.MIN_LR
        opt_args = {}
        if cfg.TRAIN.OPTIMIZER[i] in ["ADAM", "ADAMW"]:
            opt_args["betas"] = cfg.TRAIN.OPT_BETAS[i] if i < len(cfg.TRAIN.OPT_BETAS) else cfg.TRAIN.OPT_BETAS[0]
        optimizer = timm.optim.create_optimizer_v2(
            param_groups[i],
            opt=cfg.TRAIN.OPTIMIZER[i],
            lr=lr,
            weight_decay=cfg.TRAIN.W_DECAY,
            **opt_args,
        )
        print(optimizer)
        optimizers.append(optimizer)

        # Learning rate schedulers
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
                    lr=cfg.TRAIN.LR[i],
                    min_lr=cfg.TRAIN.LR_SCHEDULER.MIN_LR,
                    warmup_epochs=cfg.TRAIN.LR_SCHEDULER.WARMUP_COSINE_DECAY_EPOCHS,
                    epochs=cfg.TRAIN.EPOCHS,
                )
            elif cfg.TRAIN.LR_SCHEDULER.NAME == "onecycle":
                lr_scheduler = OneCycleLR(
                    optimizer,
                    cfg.TRAIN.LR[i],
                    epochs=cfg.TRAIN.EPOCHS,
                    steps_per_epoch=steps_per_epoch,
                )
        
        lr_schedulers.append(lr_scheduler)

    return optimizers, lr_schedulers


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
