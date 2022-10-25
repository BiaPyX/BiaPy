import random
import os
import numpy as np
import tensorflow as tf
from skimage.io import imread
from PIL import Image
from PIL.TiffTags import TAGS
from utils.util import normalize, norm_range01


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

       instance_problem : bool, optional
           To not divide the labels if being in an instance segmenation problem.
           
       norm_custom_mean : float, optional
           Mean of the data used to normalize.

       norm_custom_std : float, optional
           Std of the data used to normalize.
    """

    def __init__(self, X=None, d_path=None, provide_Y=False, Y=None, dm_path=None, dims='2D', batch_size=1, seed=42,
                 shuffle_each_epoch=False, instance_problem=False, norm_custom_mean=None, 
                 norm_custom_std=None):

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
        if X is None:
            self.len = len(self.data_path)
        else:
            self.len = len(X)
        self.o_indexes = np.arange(self.len)
        self.ax = None
        self.ay = None

        # Check if a division is required
        self.X_norm = {}
        self.X_norm['type'] = 'div'
        if provide_Y:
            self.Y_norm = {}
            self.Y_norm['type'] = 'div'
        img, mask, xnorm = self.load_sample(0)

        if norm_custom_mean is not None and norm_custom_std is not None:
            self.X_norm['type'] = 'custom'
            self.X_norm['mean'] = norm_custom_mean
            self.X_norm['std'] = norm_custom_std
        else:
            self.X_norm.update(xnorm)

        if mask is not None:
            self.Y_norm = {}
            self.Y_norm['type'] = 'div'
            if (np.max(mask) > 30 and not instance_problem):
                self.Y_norm['div'] = 1   

        self.on_epoch_end()


    def load_sample(self, idx):
        """Load one data sample given its corresponding index."""
        mask = None
        # Choose the data source
        if self.X is None:
            if self.d_path[0].endswith('.npy'):
                img = np.load(os.path.join(self.d_path, self.data_path[idx]))
                if self.provide_Y:
                    mask = np.load(os.path.join(self.dm_path, self.data_mask_path[idx]))
            else:
                img = imread(os.path.join(self.d_path, self.data_path[idx]))

                if img.ndim == 4 and self.data_path[idx].endswith('.tif'):
                    # Obtain axis position once
                    if self.ax is None:
                        img_aux = Image.open(os.path.join(self.d_path, self.data_path[idx]))
                        meta_dict = {TAGS[key] : img_aux.tag[key] for key in img_aux.tag_v2}
                        axis = meta_dict['ImageDescription'][0].split('\n')[-2].split('=')[-1]
                        self.ax = {}
                        for k, c in enumerate(axis):
                            self.ax[c] = k
                        del img_aux
                    if 'Z' in self.ax:
                        img = img.transpose((self.ax['Z'],self.ax['Y'],self.ax['X'],self.ax['C']))

                if self.provide_Y:
                    mask = imread(os.path.join(self.dm_path, self.data_mask_path[idx]))
                    if mask.ndim == 4 and self.data_mask_path[idx].endswith('.tif'):
                        # Obtain axis position once
                        if self.ay is None:
                            img_aux = Image.open(os.path.join(self.dm_path, self.data_mask_path[idx]))
                            meta_dict = {TAGS[key] : img_aux.tag[key] for key in img_aux.tag_v2}
                            axis = meta_dict['ImageDescription'][0].split('\n')[-2].split('=')[-1]
                            self.ay = {}
                            for k, c in enumerate(axis):
                                self.ay[c] = k
                            del img_aux
                        if 'Z' in self.ay:
                            mask = mask.transpose((self.ay['Z'],self.ay['Y'],self.ay['X'],self.ay['C']))

            img = np.squeeze(img)
            if self.provide_Y:
                mask = np.squeeze(mask) 
        else:
            img = self.X[idx]
            img = np.squeeze(img)

            if self.provide_Y:
                mask = self.Y[idx]
                mask = np.squeeze(mask)

        # Correct dimensions 
        if self.data_3d:
            if img.ndim == 3: img = np.expand_dims(img, -1)
        else:
            if img.ndim == 2: img = np.expand_dims(img, -1) 
        if self.provide_Y:
            if self.data_3d:
                if mask.ndim == 3: mask = np.expand_dims(mask, -1)
            else:
                if mask.ndim == 2: mask = np.expand_dims(mask, -1)

        # Normalization
        xnorm = None
        if self.X_norm['type'] == 'div':
            img, xnorm = norm_range01(img)
        elif self.X_norm['type'] == 'custom':
            img = normalize(img, self.X_norm['mean'], self.X_norm['std'])
        if self.provide_Y:
            if 'div' in self.Y_norm:
                mask = mask/255
            
        img = np.expand_dims(img, 0).astype(np.float32)
        if self.provide_Y:
            mask = np.expand_dims(mask, 0).astype(np.uint8)
        return img, mask, xnorm


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
        normx, normy = [], []

        for i, j in zip(range(len(indexes)), indexes):
            img, mask, norm = self.load_sample(j)
            
            if self.provide_Y: 
                batch_y.append(mask)
                normy.append(self.Y_norm)

            batch_x.append(img)    
            if norm is not None:
                self.X_norm.update(norm)
            normx.append(self.X_norm)
            
        batch_x = np.concatenate(batch_x)
        if self.provide_Y: batch_y = np.concatenate(batch_y)

        self.total_batches_seen += 1
        
        if self.provide_Y:
            return batch_x, normx, batch_y, normy
        else:
            return batch_x, normx

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.indexes = self.o_indexes
        if self.shuffle_each_epoch:
            random.Random(self.seed + self.total_batches_seen).shuffle(self.indexes)

