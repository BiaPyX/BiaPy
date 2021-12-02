import sys
import random
import os
import numpy as np
import tensorflow as tf
from skimage.io import imread

from utils.util import img_to_onehot_encoding


class simple_data_generator(tf.keras.utils.Sequence):
    """Image data generator without data augmentation. Used only for test data.

       Parameters
       ----------
       X : Numpy 5D/4D array, optional
           Data. E.g. ``(num_of_images, x, y, z, channels)`` or ``(num_of_images, x, y, channels)``.

       d_path : Str, optional
           Path to load the data from.

       provide_Y: bool, optional
           Wheter to return ground truth, using ``Y`` or loading from ``dm_path``.

       Y : Numpy 5D/4D array, optional
           Data mask. E.g. ``(num_of_images, x, y, z, channels)`` or ``(num_of_images, x, y, channels)``.

       dm_path : Str, optional
           Path to load themask  data mask from.

       dims: str, optional
           Dimension of the data. Possible options: ``2D`` or ``3D``.

       batch_size : int, optional
           Size of the batches.

       seed : int, optional
           Seed for random functions.

       shuffle_each_epoch : bool, optional
           To shuffle data after each epoch.
    """

    def __init__(self, X=None, d_path=None, provide_Y=False, Y=None, dm_path=None, dims='2D', batch_size=1, seed=42,
                 shuffle_each_epoch=False):

        if X is None and d_path is None:
            raise ValueError("One between 'X' or 'd_path' must be provided")
        if provide_Y:
            if Y is None and dm_path is None:
                raise ValueError("One between 'Y' or 'dm_path' must be provided")
        assert dims in ['2D', '3D']

        self.X = X
        self.Y = Y
        self.d_path = d_path
        self.dm_path = dm_path
        self.provide_Y = provide_Y
        self.data_path = sorted(next(os.walk(d_path))[2]) if X is None else None
        if provide_Y:
            self.data_mask_path = sorted(next(os.walk(dm_path))[2]) if Y is None else None
        self.shuffle_each_epoch = shuffle_each_epoch
        self.seed = seed
        self.batch_size = batch_size
        self.total_batches_seen = 0
        self.data_3d = True if dims == '3D' else False
        # Check if a division is required
        if X is None:
            self.len = len(self.data_path)
            if self.data_path[0].endswith('.npy'):
                img = np.load(os.path.join(d_path, self.data_path[0]))
            else:
                img = imread(os.path.join(d_path, self.data_path[0]))
        else:
            self.len = len(X)
            img = X[0]
        self.div_X_on_load = True if np.max(img) > 100 else False
        self.o_indexes = np.arange(self.len)
        if provide_Y:
            if Y is None:
                if self.data_mask_path[0].endswith('.npy'):
                    mask = np.load(os.path.join(dm_path, self.data_mask_path[0]))
                else:
                    mask = imread(os.path.join(dm_path, self.data_mask_path[0]))
            else:
                mask = Y[0]
            self.div_Y_on_load = True if np.max(mask) > 100 else False
        self.on_epoch_end()

    def __len__(self):
        """Defines the length of the generator"""
        return int(np.ceil(self.len/self.batch_size))


    def __getitem__(self, index):
        """Generation of one batch of data.

           Parameters
           ----------
           index : int
               Batch index counter.

           Returns
           -------
           batch_x : List of 5D/4D Numpy array
               Corresponding X elements of the batch.

           batch_y : List of 5D/4D Numpy array
               Corresponding Y elements of the batch.
        """

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_x = []
        if self.provide_Y: batch_y = []

        for i, j in zip(range(len(indexes)), indexes):
            if self.X is None:
                if self.data_path[0].endswith('.npy'):
                    img = np.load(os.path.join(self.d_path, self.data_path[j]))
                else:
                    img = imread(os.path.join(self.d_path, self.data_path[j]))
                img = np.squeeze(img)
            else:
                img = self.X[j]
            if self.provide_Y:
                if self.Y is None:
                    if self.data_mask_path[0].endswith('.npy'):
                        mask = np.load(os.path.join(self.dm_path, self.data_mask_path[j]))
                    else:
                        mask = imread(os.path.join(self.dm_path, self.data_mask_path[j]))
                    mask = np.squeeze(mask)
                else:
                    mask = self.Y[j]

            if self.data_3d:
                if img.ndim == 3: img = np.expand_dims(img, -1)
                img = img.transpose((1,2,0,3))
                img = np.expand_dims(img, 0)
                if self.provide_Y:
                    if mask.ndim == 3: mask = np.expand_dims(mask, -1)
                    mask = mask.transpose((1,2,0,3))
                    mask = np.expand_dims(mask, 0)
            else:
                if img.ndim == 2:
                    img = np.expand_dims(img, -1)
                else:
                    if img.shape[0] <= 3: img = img.transpose((1,2,0))
                img = np.expand_dims(img, 0)
                if self.provide_Y:
                    if mask.ndim == 2:
                        mask = np.expand_dims(mask, -1)
                    else:
                        if mask.shape[0] <= 3: mask = mask.transpose((1,2,0))
                    mask = np.expand_dims(mask, 0)
            batch_x.append(img)
            if self.provide_Y: batch_y.append(mask)

        batch_x = np.concatenate(batch_x)
        if self.provide_Y: batch_y = np.concatenate(batch_y)

        # Divide the values
        if self.div_X_on_load: batch_x = batch_x/255
        if self.provide_Y:
            if self.div_Y_on_load: batch_y = batch_y/255

        self.total_batches_seen += 1

        if self.provide_Y:
            return batch_x, batch_y
        else:
            return batch_x


    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.indexes = self.o_indexes
        if self.shuffle_each_epoch:
            random.Random(self.seed + self.total_batches_seen).shuffle(self.indexes)

