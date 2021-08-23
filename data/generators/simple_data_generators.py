import sys
import random
import os
import numpy as np
import tensorflow as tf
from skimage.io import imread

from utils.util import img_to_onehot_encoding


class simple_3D_data_generator(tf.keras.utils.Sequence):
    """3D ImageDataGenerator.

       Parameters
       ----------
       d_path : Str
           Path to load the data from.

       seed : int, optional
           Seed for random functions.
            
       batch_size : int, optional
           Size of the batches.
    
       shuffle_each_epoch : bool, optional
           To shuffle data after each epoch.
    """

    def __init__(self, X=None, d_path=None, batch_size=1, seed=42, shuffle_each_epoch=False):
    
        if X is None and d_path is None:
            raise ValueError("One between 'X' or 'd_path' must be provided")

        self.X = X
        self.d_path = d_path
        print("d_paht: {}".format(d_path))
        self.data_path = sorted(next(os.walk(d_path))[2]) if X is None else None
        self.shuffle_each_epoch = shuffle_each_epoch 
        self.seed = seed
        self.batch_size = batch_size
        self.len = len(self.data_path)
        self.o_indexes = np.arange(self.len)
        self.total_batches_seen = 0

        # Check if a division is required 
        img = imread(os.path.join(d_path, self.data_path[0])) if X is None else X[0]
        self.div_on_load = True if np.max(img) > 100 else False
        if img.ndim == 3: img = np.expand_dims(img, -1)
        img = img.transpose((1,2,0,3))
        self.shape = img.shape

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
           batch : 5D Numpy array
               Corresponding X elements of the batch. E.g. ``(batch_size_value, x, y, z, channels)``.
        """

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch = np.zeros((len(indexes), *self.shape), dtype=np.uint8)
                   
        for i, j in zip(range(len(indexes)), indexes):
            img = imread(os.path.join(self.d_path, self.data_path[j])) if self.X is None else self.X[j]
            if img.ndim == 3: img = np.expand_dims(img, -1)
            img = img.transpose((1,2,0,3))
            batch[i] = img

        # Divide the values 
        if self.div_on_load: batch = batch/255

        self.total_batches_seen += 1
                                                                                                
        return batch


    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.indexes = self.o_indexes
        if self.shuffle_each_epoch:
            random.Random(self.seed + self.total_batches_seen).shuffle(self.indexes)

