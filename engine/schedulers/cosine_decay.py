""" Code adapted from https://scorrea92.medium.com/cosine-learning-rate-decay-e8b50aa455b """

import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, LearningRateScheduler


class WarmUpCosineDecayScheduler(keras.callbacks.Callback):
    """Cosine decay with warmup learning rate scheduler. """
    def __init__(self, learning_rate_base, global_step_init=0, warmup_learning_rate=0.0, warmup_epochs=0,
        hold_base_rate_steps=0, min_lr=None, stop_epoch=None, save_dir=None, verbose=0):
        """
        Constructor for cosine decay with warmup learning rate scheduler. It consist in 2 phases: 1) a warm up phase 
        which consists of increasing the learning rate from ``warmup_learning_rate`` to ``learning_rate_base`` value 
        by a factor during a certain number of epochs defined by ``hold_base_rate_steps`` ; 2) after this will began  
        the decay of the learning rate value using the cosine function. Find a detailed explanation in: 
        https://scorrea92.medium.com/cosine-learning-rate-decay-e8b50aa455b

        Parameters
        ----------
        learning_rate_base : float
            Base learning rate.

        global_step_init : int, optional
            Initial global step, e.g. from previous checkpoint.

        warmup_learning_rate : float, optional
            Initial learning rate to start the warm up from.

        warmup_epochs : int, optional
            Epochs to do the warming up. 

        hold_base_rate_steps : int, optional 
            Number of steps to hold base learning rate before decaying.

        min_lr : float, optional
            Lower bound on the learning rate.
        
        stop_epoch : int, optional
            Epoch to stop the decay.

        save_dir : str, optional
            Path to the directory to save the plots of the scheduler learning rate.

        verbose : int, optional 
            ``0``: quiet, ``1``: update messages. 
        """

        super(WarmUpCosineDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.last_lr = learning_rate_base
        self.lr_changed = False
        self.phase = "Warm-up"
        self.global_step = global_step_init
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_epochs = warmup_epochs
        self.hold_base_rate_steps = hold_base_rate_steps
        self.learning_rates = []
        self.save_dir = save_dir
        self.verbose = verbose
        self.stop_epoch = stop_epoch
        self.min_lr = min_lr        
        self.epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        if self.verbose > 0 and self.lr_changed:
            print("\n{} phase: setting learning rate to {}".format(self.phase, self.last_lr))

    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)
        self.lr_changed = False if self.last_lr == lr else True
        self.last_lr = lr           

    def on_batch_begin(self, batch, logs=None):
        total_steps = int(self.params['epochs'] * self.params['steps'])
        warmup_steps = int(self.warmup_epochs * self.params['steps'])
        hold_steps = int(self.hold_base_rate_steps * self.params['steps'])
        
        if self.global_step < warmup_steps:
            self.phase = "Warm-up"
        elif self.global_step > warmup_steps + hold_steps:
            self.phase = "Cosine decay"

        lr = self.cosine_decay_with_warmup(global_step=self.global_step, learning_rate_base=self.learning_rate_base,
            total_steps=total_steps, warmup_learning_rate=self.warmup_learning_rate, warmup_steps=warmup_steps,
            hold_base_rate_steps=hold_steps)

        if self.stop_epoch is not None and self.stop_epoch > 0 and self.epoch >= self.stop_epoch:
            if self.min_lr is not None:
                K.set_value(self.model.optimizer.lr, self.min_lr)
            else:
                self.min_lr = lr
                K.set_value(self.model.optimizer.lr, self.min_lr)
        else:
            K.set_value(self.model.optimizer.lr, lr)

    def on_train_end(self, logs=None):

        self.plot()

    def cosine_decay_with_warmup(self, global_step, learning_rate_base, total_steps, warmup_learning_rate=0.0,
        warmup_steps=0, hold_base_rate_steps=0):
        """
        Cosine decay schedule with warm up period. Cosine annealing learning rate as described in
        `SGDR: Stochastic Gradient Descent with Warm Restarts <https://arxiv.org/abs/1608.03983>`_.
        In this schedule, the learning rate grows linearly from ``warmup_learning_rate`` to ``learning_rate_base`` 
        for ``warmup_steps``, then transitions to a cosine decay schedule.

        Parameters
        ----------
        global_step : int, optional
            Global step.

        learning_rate_base : float, optional
            Base learning rate.

        total_steps : int, optional
            Total number of training steps.

        warmup_learning_rate : float, optional
            Initial learning rate for warm up.

        warmup_steps : int, optional
            Number of warmup steps.
            
        hold_base_rate_steps : int, optional
            Number of steps to hold base learning rate before decaying.

        Returns
        -------
        lr : float
            A float representing learning rate.
        """
        if total_steps < warmup_steps:
            raise ValueError('total_steps must be larger or equal to warmup_steps.')
        learning_rate = 0.5 * learning_rate_base * (
            1 + np.cos(np.pi * (global_step - warmup_steps - hold_base_rate_steps) /
                float(total_steps - warmup_steps - hold_base_rate_steps)))               
            
        if hold_base_rate_steps > 0:
            learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                     learning_rate, learning_rate_base)
        if warmup_steps > 0:
            slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
            warmup_rate = slope * global_step + warmup_learning_rate
            learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                     learning_rate)
        return np.where(global_step > total_steps, 0.0, learning_rate)
    
    def plot(self):
        plt.plot(self.learning_rates)
        plt.title('Learning Rate')
        plt.ylabel('Value')
        plt.xlabel('Epoch')
        plt.savefig(os.path.join(self.save_dir, 'warmup_cosine_decay_schel.png'))
        plt.clf()