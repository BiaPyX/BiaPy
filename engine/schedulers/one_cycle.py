"""
Super-convergence with one-cycle policy (as in the fastai2 library).
Original code by Andrich van Wyk: https://www.avanwyk.com/tensorflow-2-super-convergence-with-the-1cycle-policy/

Example of application (simply add it as a callback when calling model.fit(...)):

    epochs = 3
    lr = 5e-3
    steps = np.ceil(len(x_train) / batch_size) * epochs
    lr_schedule = OneCycleScheduler(lr, steps)

    model = build_model()
    optimizer = tf.keras.optimizers.Adam( learning_rate=lr )
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_ds,epochs=epochs, callbacks=[lr_schedule])

"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import logging
import os

logging.getLogger('tensorflow').setLevel(logging.ERROR)

from tensorflow.keras.callbacks import Callback

class CosineAnnealer:
    
    def __init__(self, start, end, steps):
        self.start = start
        self.end = end
        self.steps = steps
        self.n = 0
        
    def step(self):
        self.n += 1
        cos = np.cos(np.pi * (self.n / self.steps)) + 1
        return self.end + (self.start - self.end) / 2. * cos


class OneCycleScheduler(Callback):
    """ 
    `Callback` that schedules the learning rate on a 1cycle policy as per `Leslie Smith's paper <https://arxiv.org/pdf/1803.09820.pdf>`_.
    If the model supports a momentum (or beta_1) parameter, it will also be adapted by the schedule. The implementation adopts additional improvements
    as per `the fastai2 library <https://docs.fast.ai/callback.schedule.html#learner.fit_one_cycle>`_, where only two phases are used and the adaptation is done
    using cosine annealing. In phase 1 the LR increases from ``lr_max / div_factor`` to ``lr_max`` and momentum decreases from ``mom_max`` to
    ``mom_min``. In the second phase the LR decreases from ``lr_max`` to ``lr_max / div_final`` and momemtum from ``mom_max`` to
    ``mom_min``. By default the phases are not of equal length, with the phase 1 percentage controlled by the parameter ``phase_1_pct``.
    """

    def __init__(self, lr_max, steps, mom_min=0.9, mom_max=0.99, phase_1_pct=0.25, div_factor=25., div_final=1e5, save_dir=None):
        super(OneCycleScheduler, self).__init__()
        lr_min = lr_max / div_factor
        final_lr = lr_max / div_final
        phase_1_steps = steps * phase_1_pct
        phase_2_steps = steps - phase_1_steps
        
        self.save_dir = save_dir
        self.phase_1_steps = phase_1_steps
        self.phase_2_steps = phase_2_steps
        self.phase = 0
        self.step = 0
        
        self.phases = [[CosineAnnealer(lr_min, lr_max, phase_1_steps), CosineAnnealer(mom_max, mom_min, phase_1_steps)], 
                 [CosineAnnealer(lr_max, final_lr, phase_2_steps), CosineAnnealer(mom_min, mom_max, phase_2_steps)]]
        
        self.lrs = []
        self.moms = []

    def on_train_begin(self, logs=None):
        self.phase = 0
        self.step = 0

        self.set_lr(self.lr_schedule().start)
        self.set_momentum(self.mom_schedule().start)
        
    def on_train_batch_begin(self, batch, logs=None):
        self.lrs.append(self.get_lr())
        self.moms.append(self.get_momentum())

    def on_train_batch_end(self, batch, logs=None):
        self.step += 1
        if self.step >= self.phase_1_steps:
            self.phase = 1
            
        self.set_lr(self.lr_schedule().step())
        self.set_momentum(self.mom_schedule().step())

    def on_train_end(self, epoch, logs=None):
        self.plot()

    def get_lr(self):
        try:
            return tf.keras.backend.get_value(self.model.optimizer.lr)
        except AttributeError:
            return None
        
    def get_momentum(self):
        try:
            if hasattr( self.model.optimizer, 'momentum' ):
                return tf.keras.backend.get_value(self.model.optimizer.momentum)
            else:
                return tf.keras.backend.get_value(self.model.optimizer.beta_1)
        except AttributeError:
            return None
        
    def set_lr(self, lr):
        try:
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        except AttributeError:
            pass # ignore
        
    def set_momentum(self, mom):
        try:
            if hasattr( self.model.optimizer, 'momentum' ):
                if not tf.is_tensor( self.model.optimizer.momentum ):
                    self.model.optimizer.momentum = mom
                else:
                    tf.keras.backend.set_value(self.model.optimizer.momentum, mom)
            else:
                if not tf.is_tensor( self.model.optimizer.beta_1 ):
                    self.model.optimizer.beta_1 = mom
                else:
                    tf.keras.backend.set_value(self.model.optimizer.beta_1, mom)
        except AttributeError as e:
            pass # ignore

    def lr_schedule(self):
        return self.phases[self.phase][0]
    
    def mom_schedule(self):
        return self.phases[self.phase][1]
    
    def plot(self):
        ax = plt.subplot(1, 2, 1)
        ax.plot(self.lrs)
        ax.set_title('Learning Rate - One cycle')
        ax = plt.subplot(1, 2, 2)
        ax.plot(self.moms)
        ax.set_title('Momentum - One cycle')
        plt.savefig(os.path.join(self.save_dir, 'one_cycle_schel.png'))
        plt.clf()
