import random
import os
import numpy as np
import tensorflow as tf
from skimage.io import imread
from PIL import Image
from PIL.TiffTags import TAGS

from data.pre_processing import normalize, norm_range01


class test_pair_data_generator(tf.keras.utils.Sequence):
    """
    Image data generator without data augmentation. Used only for test data.

    Parameters
    ----------
    X : Numpy 5D/4D array, optional
        Data. E.g. ``(num_of_images, z, y, x, channels)`` for ``3D`` or ``(num_of_images, y, x, channels)`` for ``2D``.

    d_path : Str, optional
        Path to load the data from.

    provide_Y: bool, optional
        Whether to return ground truth, using ``Y`` or loading from ``dm_path``.

    Y : Numpy 5D/4D array, optional
        Data mask. E.g. ``(num_of_images, z, y, x, channels)`` for ``3D`` or ``(num_of_images, y, x, channels)`` for ``2D``.

    dm_path : Str, optional
        Path to load the mask data from.

    dims: str, optional
        Dimension of the data. Possible options: ``2D`` or ``3D``.

    seed : int, optional
        Seed for random functions.

    instance_problem : bool, optional
        To not divide the labels if being in an instance segmenation problem.
        
    norm_custom_mean : float, optional
        Mean of the data used to normalize.

    norm_custom_std : float, optional
        Std of the data used to normalize.

    sample_ids :  List of ints, optional
        When cross validation is used specific training samples are passed to the generator.
    
    """
    def __init__(self, ndim, X=None, d_path=None, provide_Y=False, Y=None, dm_path=None, seed=42,
                 instance_problem=False, normalizeY='as_mask', norm_custom_mean=None, 
                 norm_custom_std=None, sample_ids=None):

        if X is None and d_path is None:
            raise ValueError("One between 'X' or 'd_path' must be provided")
        if provide_Y:
            if Y is None and dm_path is None:
                raise ValueError("One between 'Y' or 'dm_path' must be provided")
        assert normalizeY in ['as_mask', 'as_image', 'none']
        
        self.X = X
        self.Y = Y
        self.d_path = d_path
        self.dm_path = dm_path
        self.provide_Y = provide_Y
        self.data_path = sorted(next(os.walk(d_path))[2]) if X is None else None
        if sample_ids is not None and self.data_path is not None:
            self.data_path = [x for i, x in enumerate(self.data_path) if i in sample_ids]
        if provide_Y:
            self.data_mask_path = sorted(next(os.walk(dm_path))[2]) if Y is None else None
            if sample_ids is not None and self.data_mask_path is not None:
                self.data_mask_path = [x for i, x in enumerate(self.data_mask_path) if i in sample_ids]
        self.seed = seed
        self.ndim = ndim
        if X is None:
            self.len = len(self.data_path)
        else:
            self.len = len(X)
        self.o_indexes = np.arange(self.len)
        self.normalizeY = normalizeY
        
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
            if normalizeY == 'as_mask':
                self.Y_norm['type'] = 'div'
                if (np.max(mask) > 30 and not instance_problem):
                    self.Y_norm['div'] = 1   
            elif normalizeY == 'as_image':
                self.Y_norm.update(self.X_norm)

    def load_sample(self, idx):
        """Load one data sample given its corresponding index."""
        mask = None
        # Choose the data source
        if self.X is None:
            if self.data_path[idx].endswith('.npy'):
                img = np.load(os.path.join(self.d_path, self.data_path[idx]))
                if self.provide_Y:
                    mask = np.load(os.path.join(self.dm_path, self.data_mask_path[idx]))
            else:
                img = imread(os.path.join(self.d_path, self.data_path[idx]))
                img = np.squeeze(img)
                if self.provide_Y:
                    mask = imread(os.path.join(self.dm_path, self.data_mask_path[idx]))
                    mask = np.squeeze(mask)  
        else:
            img = self.X[idx]
            img = np.squeeze(img)

            if self.provide_Y:
                mask = self.Y[idx]
                mask = np.squeeze(mask)

        # Correct dimensions 
        if self.ndim == 3:
            if img.ndim == 3: 
                img = np.expand_dims(img, -1)
            else:
                min_val = min(img.shape)
                channel_pos = img.shape.index(min_val)
                if channel_pos != 3 and img.shape[channel_pos] <= 4:
                    new_pos = [x for x in range(4) if x != channel_pos]+[channel_pos,]
                    img = img.transpose(new_pos)
        else:
            if img.ndim == 2: 
                img = np.expand_dims(img, -1) 
            else:
                if img.shape[0] <= 3: img = img.transpose((1,2,0))
        if self.provide_Y:
            if self.ndim == 3:
                if mask.ndim == 3: 
                    mask = np.expand_dims(mask, -1)
                else:
                    min_val = min(mask.shape)
                    channel_pos = mask.shape.index(min_val)
                    if channel_pos != 3 and mask.shape[channel_pos] <= 4:
                        new_pos = [x for x in range(4) if x != channel_pos]+[channel_pos,]
                        mask = mask.transpose(new_pos)
            else:
                if mask.ndim == 2: 
                    mask = np.expand_dims(mask, -1)
                else:
                    if mask.shape[0] <= 3: mask = mask.transpose((1,2,0))

        # Normalization
        xnorm = None
        if self.X_norm['type'] == 'div':
            img, xnorm = norm_range01(img)
        elif self.X_norm['type'] == 'custom':
            img = normalize(img, self.X_norm['mean'], self.X_norm['std'])
        if self.provide_Y:
            if self.normalizeY == 'as_mask':
                if 'div' in self.Y_norm:
                    mask = mask/255
            elif self.normalizeY == 'as_image':
                if self.X_norm['type'] == 'div':
                    mask, xnorm = norm_range01(mask)
                elif self.X_norm['type'] == 'custom':
                    mask = normalize(mask, self.X_norm['mean'], self.X_norm['std'])
           
        img = np.expand_dims(img, 0).astype(np.float32)
        if self.provide_Y:
            mask = np.expand_dims(mask, 0)
            if self.normalizeY == 'as_mask':
                mask = mask.astype(np.uint8)
        return img, mask, xnorm


    def __len__(self):
        """Defines the length of the generator"""
        return self.len


    def __getitem__(self, index):
        """Generation of one pair of data.

           Parameters
           ----------
           index : int
               Sample index counter.

           Returns
           -------
           img : 3D/4D Numpy array
               X element, for instance, an image. E.g. ``(z, y, x, channels)`` if ``2D`` or 
               ``(y, x, channels)`` if ``3D``. 
               
           mask : 3D/4D Numpy array
               Y element, for instance, a mask. E.g. ``(z, y, x, channels)`` if ``2D`` or 
               ``(y, x, channels)`` if ``3D``.
        """
        img, mask, norm = self.load_sample(index)
        
        if norm is not None:
            self.X_norm.update(norm)
                    
        if self.provide_Y:
            return img, self.X_norm, mask, self.Y_norm
        else:
            return img, self.X_norm


