import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from utils.callbacks import ModelCheckpoint, TimeHistory
from engine.metrics import (jaccard_index, jaccard_index_softmax, jaccard_index_instances, 
                            instance_segmentation_loss, weighted_bce_dice_loss)


def prepare_optimizer(cfg, model):
    """Select the optimizer, loss and metrics for the given model.
                                                                                                                        
       Parameters                                                                                                       
       ----------                                                                                                       
       cfg : YACS CN object                                                                               
           Configuration.                                                                                         
                                                                                                                        
       model : Keras model
           Model to be compiled with the selected options. 
    """      

    assert cfg.TRAIN.OPTIMIZER in ['SGD', 'ADAM']
    assert cfg.LOSS.TYPE in ['CE', 'W_CE_DICE']

    # Select the optimizer
    if cfg.TRAIN.OPTIMIZER == "SGD":
        opt = tf.keras.optimizers.SGD(lr=cfg.TRAIN.LR, momentum=0.99, decay=0.0, nesterov=False)
    elif cfg.TRAIN.OPTIMIZER == "ADAM":
        opt = tf.keras.optimizers.Adam(lr=cfg.TRAIN.LR, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    # Compile the model
    if cfg.LOSS.TYPE == "CE" and cfg.PROBLEM.TYPE == "SEMANTIC_SEG":
        if cfg.MODEL.N_CLASSES > 1:
            model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=[jaccard_index_softmax])
        else:
            model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[jaccard_index])
    elif cfg.LOSS.TYPE == "CE" and cfg.PROBLEM.TYPE == "INSTANCE_SEG": 
        if cfg.MODEL.N_CLASSES > 1: 
            raise ValueError("Not implemented pipeline option: N_CLASSES > 1 and INSTANCE_SEG")
        else:
            if cfg.DATA.CHANNELS == "B" or cfg.DATA.CHANNELS == "BC":
                model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[jaccard_index])
            else:
                model.compile(optimizer=opt, loss=instance_segmentation_loss(channel_weights, output_channels),
                              metrics=[jaccard_index_instances])
    elif cfg.LOSS.TYPE == "W_CE_DICE" and cfg.PROBLEM.TYPE == "SEMANTIC_SEG":
        model.compile(optimizer=opt, loss=weighted_bce_dice_loss(w_dice=0.66, w_bce=0.33), metrics=[jaccard_index])
    elif cfg.LOSS.TYPE == "W_CE_DICE" and cfg.PROBLEM.TYPE == "INSTANCE_SEG":
        raise ValueError("Not implemented pipeline option: LOSS.TYPE == W_CE_DICE and INSTANCE_SEG")
                                                                                                                        

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
    earlystopper = EarlyStopping(patience=cfg.TRAIN.PATIENCE, verbose=1, restore_best_weights=True)
    callbacks.append(earlystopper)

    # Save the best model into a h5 file in case one need again the weights learned
    os.makedirs(cfg.PATHS.CHECKPOINT, exist_ok=True)
    checkpointer = ModelCheckpoint(cfg.PATHS.CHECKPOINT_FILE, verbose=1, save_best_only=True)
    callbacks.append(checkpointer)

    return callbacks
