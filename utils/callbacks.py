"Code copied from `Tensorflow/keras/callbacks.py <https://github.com/tensorflow/tensorflow/blob/b36436b087bd8e8701ef51718179037cccdfc26e/tensorflow/python/keras/callbacks.py#L1057>`_ just inserting a few lines on the prints to avoid this `error <https://github.com/tensorflow/tensorflow/issues/35100>`_."

import os
import time
import warnings
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.io import imsave, imread

import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.
       Copied from: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """
    def __init__(self, patience=7, delta=0, trace_func=print):
        """
        Parameters
        ----------
        patience : int, optional
            How long to wait after last time validation loss improved.

        delta : float, optional
            Minimum change in the monitored quantity to qualify as an improvement.

        trace_func : function, optional
            Trace print function.
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func

    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0

# class MAE_callback():
#     """ Save MAE process in selected epochs. """
#     def __init__(self, test_gen, epoch_interval=None, out_dir="MAE_checks"):
#         self.epoch_interval = epoch_interval
#         self.test_gen = test_gen
#         self.out_dir = out_dir

#     def on_epoch_end(self, epoch, logs=None):
#         if self.epoch_interval and epoch != 0 and epoch % self.epoch_interval == 0:
#             test_images = next(iter(self.test_gen))

#             idx = np.random.choice(test_images.shape[0])
#             cmap = 'gray' if test_images[idx].shape[-1] == 1 else ''
#             test_images_patches = self.model.patchify(test_images)[idx]

#             latent, mask, ids_restore, unmasked_ids, masked_ids = self.model.encoder(test_images, training=False)
#             reconstructed = self.model.decoder(latent, training=False, ids_restore=ids_restore)[idx] 
#             masked_img = self.model.generate_masked_image(test_images_patches, unmasked_ids[idx])
            
#             fig = plt.figure(figsize=(30, 5), constrained_layout=True)
#             grid = gridspec.GridSpec(1, 5, figure=fig)

#             ax1 = fig.add_subplot(grid[0])
#             ax1.imshow(test_images[idx], cmap=cmap)
#             ax1.set_title(f"Original: {epoch:03d}")

#             rows, cols = self.model.encoder.patch_embed.grid_size
#             patch_size = self.model.encoder.patch_embed.patch_size[0]
#             gs00 = gridspec.GridSpecFromSubplotSpec(rows, cols, subplot_spec=grid[1])
#             for i in range(rows):
#                 for j in range(cols):
#                     ax = fig.add_subplot(gs00[i, j], xticklabels=[],yticklabels=[])
#                     ax.imshow(tf.reshape(test_images_patches[(i*cols)+j],(patch_size, patch_size)), cmap=cmap)
#                     ax.set_xticks([])
#                     ax.set_yticks([])

#             ax3 = fig.add_subplot(grid[2])
#             ax3.imshow(self.model.unpatchify(test_images_patches), cmap=cmap)
#             ax3.set_title(f"Unpatchify: {epoch:03d}")

#             ax4 = fig.add_subplot(grid[3])
#             ax4.imshow(self.model.unpatchify(masked_img), cmap=cmap)
#             ax4.set_title(f"Masked: {epoch:03d}")

#             ax5 = fig.add_subplot(grid[4])
#             ax5.imshow(self.model.unpatchify(reconstructed), cmap=cmap)
#             ax5.set_title(f"Reconstructed: {epoch:03d}")
            
#             os.makedirs(self.out_dir, exist_ok=True)
#             f = os.path.join(self.out_dir, "img_{}.png".format(epoch))
#             plt.savefig(f)
#             plt.close()
#             print("Saving MAE callback image . . .")
           
