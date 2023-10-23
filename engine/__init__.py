import os
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
import timm
import timm.optim.optim_factory as optim_factory

from engine.schedulers.warmup_cosine_decay import WarmUpCosineDecayScheduler
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from utils.callbacks import EarlyStopping

def prepare_optimizer(cfg, model_without_ddp, steps_per_epoch):
    """Select the optimizer, loss and metrics for the given model.

       Parameters
       ----------
       cfg : YACS CN object
           Configuration.
    """
    lr = cfg.TRAIN.LR if cfg.TRAIN.LR_SCHEDULER.NAME != "warmupcosine" else cfg.TRAIN.LR_SCHEDULER.MIN_LR
    optimizer = optim_factory.create_optimizer_v2(model_without_ddp, opt=cfg.TRAIN.OPTIMIZER, lr=lr, weight_decay=cfg.TRAIN.W_DECAY)
    print(optimizer)

    # Learning rate schedulers
    lr_scheduler = None
    if cfg.TRAIN.LR_SCHEDULER.NAME != '':
        if cfg.TRAIN.LR_SCHEDULER.NAME == 'reduceonplateau':
            lr_scheduler = ReduceLROnPlateau(optimizer, patience=cfg.TRAIN.LR_SCHEDULER.REDUCEONPLATEAU_PATIENCE,
                factor=cfg.TRAIN.LR_SCHEDULER.REDUCEONPLATEAU_FACTOR, min_lr=cfg.TRAIN.LR_SCHEDULER.MIN_LR,)
        elif cfg.TRAIN.LR_SCHEDULER.NAME == 'warmupcosine':
            lr_scheduler = WarmUpCosineDecayScheduler(lr=cfg.TRAIN.LR, min_lr=cfg.TRAIN.LR_SCHEDULER.MIN_LR,
                warmup_epochs=cfg.TRAIN.LR_SCHEDULER.WARMUP_COSINE_DECAY_EPOCHS, epochs=cfg.TRAIN.EPOCHS)
        elif cfg.TRAIN.LR_SCHEDULER.NAME == 'onecycle':
            lr_scheduler = OneCycleLR(optimizer, cfg.TRAIN.LR, epochs=cfg.TRAIN.EPOCHS,
                steps_per_epoch=steps_per_epoch)

    loss_scaler = NativeScaler()

    return optimizer, lr_scheduler, loss_scaler

def build_callbacks(cfg):
    """Create training and validation generators.

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
