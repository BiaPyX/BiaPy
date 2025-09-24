"""
This module provides an implementation of an Early Stopping mechanism, a common regularization technique used in machine learning to prevent overfitting during model training.

The `EarlyStopping` class monitors a validation metric (typically validation loss)
and stops the training process if the metric does not improve for a specified
number of epochs (patience).

Classes:

- EarlyStopping: Implements the early stopping logic.

This implementation is adapted from a widely used PyTorch early stopping script.
"""
import numpy as np
from typing import (
    Callable,
)
from biapy.utils.misc import is_main_process

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.

    This class monitors a specified metric (e.g., validation loss) during training.
    If the metric does not show improvement (beyond a `delta` threshold) for a
    number of epochs defined by `patience`, the `early_stop` flag is set to True,
    signaling that training should be halted.

    Copied from: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """

    def __init__(
        self, 
        patience: int=7, 
        delta: float=0, 
        trace_func: Callable=print
    ):
        """
        Initialize the EarlyStopping instance.

        Sets up the parameters for monitoring validation loss and determining
        when to stop training.

        Parameters
        ----------
        patience : int, optional
            How many epochs to wait for improvement in validation loss after the
            last observed improvement. If no improvement is seen within this many
            epochs, training will be stopped. Defaults to 7.
        delta : float, optional
            Minimum change in the monitored quantity (validation loss) to qualify
            as an improvement. Any change smaller than `delta` is considered no
            improvement. Defaults to 0.
        trace_func : Callable, optional
            A function used for printing messages (e.g., `print` or a custom logger).
            This function will be called to log the early stopping counter.
            Defaults to `print`.
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.trace_func = trace_func

    def __call__(self, val_loss: float):
        """
        Update the internal state of the early stopping mechanism based on the current validation loss.

        This method should be called after each epoch's validation phase. It
        compares the current `val_loss` with the best score observed so far.
        If no significant improvement is made, the internal counter is incremented.
        If the counter exceeds `patience`, the `early_stop` flag is set to True.

        Parameters
        ----------
        val_loss : float
            The current validation loss for the epoch.
        """
        score = -val_loss # We want to maximize score, so minimize -val_loss

        if self.best_score is None:
            # First epoch, initialize best_score
            self.best_score = score
        elif score < self.best_score + self.delta:
            # No significant improvement
            self.counter += 1
            if is_main_process():
                self.trace_func(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Improvement detected, reset counter and update best score
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
