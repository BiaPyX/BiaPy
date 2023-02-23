import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from utils.callbacks import ModelCheckpoint, TimeHistory
from engine.metrics import (jaccard_index, jaccard_index_softmax, IoU_instances, instance_segmentation_loss, weighted_bce_dice_loss,
                            masked_bce_loss, masked_jaccard_index, PSNR, n2v_loss_mse, MAE_instances)
from engine.schedulers.one_cycle import OneCycleScheduler
from engine.schedulers.cosine_decay import WarmUpCosineDecayScheduler


def prepare_optimizer(cfg, model):
    """Select the optimizer, loss and metrics for the given model.

       Parameters
       ----------
       cfg : YACS CN object
           Configuration.

       model : Keras model
           Model to be compiled with the selected options.
    """

    # Select the optimizer
    if cfg.TRAIN.OPTIMIZER == "SGD":
        opt = tf.keras.optimizers.SGD(learning_rate=cfg.TRAIN.LR, momentum=0.99, nesterov=False)
    elif cfg.TRAIN.OPTIMIZER == "ADAM":
        opt = tf.keras.optimizers.Adam(learning_rate=cfg.TRAIN.LR, beta_1=0.9, beta_2=0.999, amsgrad=False)

    # Compile the model
    metric_name = []
    if cfg.PROBLEM.TYPE == "CLASSIFICATION":
        metric_name.append("accuracy")
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=[metric_name])
    elif cfg.PROBLEM.TYPE in ["SEMANTIC_SEG", 'DETECTION']:
        if cfg.LOSS.TYPE == "CE": 
            if cfg.MODEL.N_CLASSES == 1 or cfg.MODEL.N_CLASSES == 2: # Binary case
                fname = jaccard_index
                loss_name = 'binary_crossentropy'
                metric_name.append("jaccard_index")
            else: # Multiclass
                # Use softmax jaccard if it is not going to be done in the last layer of the model
                if cfg.MODEL.LAST_ACTIVATION != 'softmax':
                    fname = jaccard_index_softmax  
                    loss_name = 'categorical_crossentropy'
                    metric_name.append("jaccard_index_softmax")
                else:
                    fname = jaccard_index
                    metric_name.append("jaccard_index")
                    loss_name = 'sparse_categorical_crossentropy'
            model.compile(optimizer=opt, loss=loss_name, metrics=[fname]) 
        elif cfg.LOSS.TYPE == "MASKED_BCE":
            metric_name.append("masked_jaccard_index")
            model.compile(optimizer=opt, loss=masked_bce_loss, metrics=[masked_jaccard_index])
        elif cfg.LOSS.TYPE == "W_CE_DICE":
            model.compile(optimizer=opt, loss=weighted_bce_dice_loss(w_dice=0.66, w_bce=0.33), metrics=[jaccard_index])
            metric_name.append("jaccard_index")
    elif cfg.PROBLEM.TYPE == "INSTANCE_SEG":
        metrics = []
        if cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS in ["BC", "BCM", "BP"]:
            metrics.append(IoU_instances(first_not_binary_channel=len(cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS)))
            metric_name.append("jaccard_index_instances")
        if cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS in ["BCD", 'BCDv2']:
            metrics.append(IoU_instances(first_not_binary_channel=2))
            metrics.append(MAE_instances(distance_channel=-1))
            metric_name.append("jaccard_index_instances")
            metric_name.append("mae_distance_channel")
        if cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS == "BDv2":
            metrics.append(IoU_instances(first_not_binary_channel=1))
            metrics.append(MAE_instances(distance_channel=-1))
            metric_name.append("jaccard_index_instances")
            metric_name.append("mae_distance_channel")
        if cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS == "Dv2":
            metrics.append(MAE_instances(distance_channel=-1))
            metric_name.append("mae_distance_channel")
        model.compile(optimizer=opt, loss=instance_segmentation_loss(cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNEL_WEIGHTS, cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS),
                      metrics=metrics)   
    elif cfg.PROBLEM.TYPE in ["SUPER_RESOLUTION", "SELF_SUPERVISED"]:
        print("Overriding 'LOSS.TYPE' to set it to MAE")
        model.compile(optimizer=opt, loss="mae", metrics=[PSNR])
        metric_name.append("PSNR")
    elif cfg.PROBLEM.TYPE == "DENOISING":
        print("Overriding 'LOSS.TYPE' to set it to N2V loss (masked MSE)")
        model.compile(optimizer=opt, loss=n2v_loss_mse(), metrics=[n2v_loss_mse()])
        metric_name.append("n2v_mse")
    return metric_name

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

    callbacks = []

    # To measure the time
    time_callback = TimeHistory()
    callbacks.append(time_callback)

    # Stop early and restore the best model weights when finished the training
    earlystopper = EarlyStopping(monitor=cfg.TRAIN.EARLYSTOPPING_MONITOR, patience=cfg.TRAIN.PATIENCE, verbose=1,
                                 restore_best_weights=True)
    callbacks.append(earlystopper)

    # Save the best model into a h5 file in case one need again the weights learned
    os.makedirs(cfg.PATHS.CHECKPOINT, exist_ok=True)
    checkpointer = ModelCheckpoint(cfg.PATHS.CHECKPOINT_FILE, monitor=cfg.TRAIN.CHECKPOINT_MONITOR, verbose=1,
                                   save_best_only=True)
    callbacks.append(checkpointer)

    # Learning rate schedulers
    if cfg.TRAIN.LR_SCHEDULER.NAME != '':
        if cfg.TRAIN.LR_SCHEDULER.NAME == 'reduceonplateau':
            lr_schedule = ReduceLROnPlateau(monitor=cfg.TRAIN.EARLYSTOPPING_MONITOR, factor=cfg.TRAIN.LR_SCHEDULER.REDUCEONPLATEAU_FACTOR, 
                patience=cfg.TRAIN.LR_SCHEDULER.REDUCEONPLATEAU_PATIENCE, min_lr=cfg.TRAIN.LR_SCHEDULER.MIN_LR, verbose=1)
        elif cfg.TRAIN.LR_SCHEDULER.NAME == 'warmupcosine':
            lr_schedule = WarmUpCosineDecayScheduler(cfg.TRAIN.LR, warmup_learning_rate=cfg.TRAIN.LR_SCHEDULER.WARMUP_COSINE_DECAY_LR, 
                warmup_epochs=cfg.TRAIN.LR_SCHEDULER.WARMUP_COSINE_DECAY_EPOCHS, hold_base_rate_steps=cfg.TRAIN.LR_SCHEDULER.WARMUP_COSINE_DECAY_HOLD_EPOCHS,
                min_lr=cfg.TRAIN.LR_SCHEDULER.MIN_LR, save_freq=cfg.TRAIN.LR_SCHEDULER.SAVE_FREQ, save_dir=cfg.PATHS.CHARTS, verbose=1)
        elif cfg.TRAIN.LR_SCHEDULER.NAME == 'onecycle':
            lr_schedule = OneCycleScheduler(cfg.TRAIN.LR, cfg.TRAIN.LR_SCHEDULER.ONE_CYCLE_STEP, save_freq=cfg.TRAIN.LR_SCHEDULER.SAVE_FREQ,
                save_dir=cfg.PATHS.CHARTS)

        callbacks.append(lr_schedule)
    return callbacks
