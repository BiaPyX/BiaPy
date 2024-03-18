import random
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from skimage.io import imread
from PIL import Image
from PIL.TiffTags import TAGS

from biapy.data.pre_processing import normalize, norm_range01
from biapy.data.generators.augmentors import center_crop_single, resize_img


class test_single_data_generator(Dataset):
    """
    Image data generator without data augmentation. Used only for test data.

    Parameters
    ----------
    ndim : int
        Dimensions of the data (``2`` for 2D and ``3`` for 3D).

    ptype : str
        Problem type. Options ['ssl','classification'].

    X : Numpy 5D/4D array, optional
        Data. E.g. ``(num_of_images, z, y, x, channels)``  for ``3D`` or ``(num_of_images, y, x, channels)`` for ``2D``.

    d_path : Str, optional
        Path to load the data from.

    test_by_chunks : bool, optional
        Not used in this generator yet but added for compatibility. 

    provide_Y: bool, optional
        Whether the ground truth has been provided or not.

    Y : Numpy 2D array, optional
        Image classes. E.g. ``(num_of_images, class)``.

    dm_path : Str, optional
        Not used here.

    dims: str, optional
        Dimension of the data. Possible options: ``2D`` or ``3D``.

    seed : int, optional
        Seed for random functions.

    instance_problem : bool, optional
        Not used here.

    norm_type : str, optional
        Type of normalization to be made. Options available: ``div`` or ``custom``.

    not_normalize : bool, optional
        Whether to normalize the data or not. Useful in BMZ model as the normalization is made during the inference. 
        
    norm_custom_mean : float, optional
        Mean of the data used to normalize.

    norm_custom_std : float, optional
        Std of the data used to normalize.

    norm_custom_mode :  str, optional
        Whether to apply the normalization by sample or with all dataset statistics. Options: ``'image'`` or ``'dataset'``.
        
    reduce_mem : bool, optional
        To reduce the dtype from float32 to float16. 

    crop_center : bool, optional
        Whether to extract a

    sample_ids :  List of ints, optional
        When cross validation is used specific training samples are passed to the generator. 
        Not used in this generator. 

    convert_to_rgb : bool, optional
        In case RGB images are expected, e.g. if ``crop_shape`` channel is 3, those images that are grayscale are 
        converted into RGB.
    """
    def __init__(self, ndim, ptype, X=None, d_path=None, test_by_chunks=False, provide_Y=False, Y=None, 
        dm_path=None, seed=42, instance_problem=False, norm_type='div', not_normalize=False, norm_custom_mean=None, 
        norm_custom_std=None, norm_custom_mode=None, crop_center=False, reduce_mem=False, resize_shape=None, 
        sample_ids=None, convert_to_rgb=False):
        if X is None and d_path is None:
            raise ValueError("One between 'X' or 'd_path' must be provided")
        if crop_center and resize_shape is None:
            raise ValueError("'resize_shape' need to be provided if 'crop_center' is enabled")
        assert ptype in ['ssl', 'classification']

        self.ptype = ptype
        self.X = X
        self.Y = Y
        self.d_path = d_path
        self.provide_Y = provide_Y
        self.convert_to_rgb = convert_to_rgb
        
        if not reduce_mem:
            self.dtype = np.float32  
            self.dtype_str = "float32"
        else:
            self.dtype = np.float16
            self.dtype_str = "float16"
        self.crop_center = crop_center
        self.resize_shape = resize_shape

        if self.ptype == "classification":
            self.class_names = sorted(next(os.walk(d_path))[1])
            self.class_numbers = {}
            for i, c_name in enumerate(self.class_names):
                self.class_numbers[c_name] = i
            self.classes = {}
            if self.X is None:
                self.data_path = []
                print("Collecting data ids . . .")
                for folder in self.class_names:
                    print("Analizing folder {}".format(os.path.join(d_path,folder)))
                    ids = sorted(next(os.walk(os.path.join(d_path,folder)))[2])
                    print("Found {} samples".format(len(ids)))
                    for i in range(len(ids)):
                        self.classes[ids[i]] = folder
                        self.data_path.append(ids[i])
                self.len = len(self.data_path)
                if self.len == 0:
                    raise ValueError("No image found in {}".format(d_path))
            else:
                self.len = len(X)
        else:
            self.data_path = sorted(next(os.walk(d_path))[2])
            self.len = len(self.data_path)
        self.seed = seed
        self.ndim = ndim
        self.o_indexes = np.arange(self.len)
        
        self.not_normalize = not_normalize
        # Check if a division is required
        self.X_norm = {}
        self.X_norm['type'] = 'none'
        if not self.not_normalize:
            self.X_norm['type'] = 'div'
            img, _, xnorm = self.load_sample(0)

            if norm_type == 'custom':
                if norm_custom_mean is not None and norm_custom_std is not None:
                    self.X_norm['mean'] = norm_custom_mean
                    self.X_norm['std'] = norm_custom_std
                    del img
                self.X_norm['type'] = 'custom'
                self.X_norm['mode'] = norm_custom_mode
                self.X_norm['orig_dtype'] = img.dtype
            if xnorm is not None:
                self.X_norm.update(xnorm)

    def load_sample(self, idx):
        """Load one data sample given its corresponding index."""
        img_class = None

        # Choose the data source
        if self.X is not None:
            img = self.X[idx]
            img = np.squeeze(img)

            if self.provide_Y:
                img_class = self.Y[idx] if self.ptype == "classification" else 0
        else:
            sample_id = self.data_path[idx]
            if self.ptype == "classification":
                sample_class_dir = self.classes[sample_id]
                f = os.path.join(self.d_path, sample_class_dir, sample_id)
                img_class = self.class_numbers[sample_class_dir]
            else:
                f = os.path.join(self.d_path, sample_id)
                img_class = 0 
            img = np.load(f) if sample_id.endswith('.npy') else imread(f)
            img = np.squeeze(img)

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

        # Normalization
        xnorm = None
        if not self.not_normalize:
            if self.X_norm['type'] == 'div':
                img, xnorm = norm_range01(img, dtype=self.dtype)
            elif self.X_norm['type'] == 'custom':
                if self.X_norm['mode'] == "image":
                    img = normalize(img, img.mean(), img.std(), out_type=self.dtype_str)
                else:
                    img = normalize(img, self.X_norm['mean'], self.X_norm['std'], out_type=self.dtype_str)
            
        img = np.expand_dims(img, 0).astype(self.dtype)

        if self.convert_to_rgb and img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)

        return img, img_class, xnorm


    def __len__(self):
        """Defines the length of the generator"""
        return self.len


    def __getitem__(self, index):
        """Generation of one sample of data.

           Parameters
           ----------
           index : int
               Sample index counter.

           Returns
           -------
           img : 3D Numpy array
               X element, for instance, an image. E.g. ``(y, x, channels)``.

           img_class : ints
               Y element, for instance, a class number.
        """

        img, img_class, norm = self.load_sample(index)
        
        if self.crop_center and img.shape[:-1] != self.resize_shape[:-1]:
            img = center_crop_single(img[0], self.resize_shape)
            img = resize_img(img, self.resize_shape[:-1])
            img = np.expand_dims(img,0)

        if norm is not None:
            self.X_norm.update(norm)           

        if self.ptype == "classification":
            if self.provide_Y:
                return img, self.X_norm, img_class, None
            else:
                return img, self.X_norm
        else: # SSL - MAE
            return img, self.X_norm
            
    def get_data_normalization(self):
        return self.X_norm