import random
import torch
from torch.utils.data import Dataset
import os
import h5py
import numpy as np
from skimage.io import imread
from PIL import Image
from PIL.TiffTags import TAGS

from biapy.data.pre_processing import normalize, norm_range01, percentile_clip


class test_pair_data_generator(Dataset):
    """
    Image data generator without data augmentation. Used only for test data.

    Parameters
    ----------
    X : Numpy 5D/4D array, optional
        Data. E.g. ``(num_of_images, z, y, x, channels)`` for ``3D`` or ``(num_of_images, y, x, channels)`` for ``2D``.

    d_path : Str, optional
        Path to load the data from.

    test_by_chunks : bool, optional
        Tell the generator that the data is going to be read by chunks and by H5/Zarr files. 

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

    norm_dict : str, optional
        Normalization instructions. 

    reduce_mem : bool, optional
        To reduce the dtype from float32 to float16. 

    sample_ids :  List of ints, optional
        When cross validation is used specific training samples are passed to the generator.
    
    convert_to_rgb : bool, optional
        In case RGB images are expected, e.g. if ``crop_shape`` channel is 3, those images that are grayscale are 
        converted into RGB.
    
    multiple_raw_images : bool, optional
        Whether to consider more than one raw images or not. In this case, a folder per each sample is expected. Visit
        `LightMyCells challenge approach <https://biapy.readthedocs.io/en/latest/tutorials/image-to-image/lightmycells.html>`_ 
        for a real use case.  
    """
    def __init__(self, ndim, X=None, d_path=None, test_by_chunks=False, provide_Y=False, Y=None, dm_path=None, seed=42,
                 instance_problem=False, norm_dict=None, reduce_mem=False, sample_ids=None, convert_to_rgb=False,
                 multiple_raw_images=False):

        if X is None and d_path is None:
            raise ValueError("One between 'X' or 'd_path' must be provided")
        if provide_Y:
            if Y is None and dm_path is None:
                raise ValueError("One between 'Y' or 'dm_path' must be provided")
        assert norm_dict['mask_norm'] in ['as_mask', 'as_image', 'none']
        assert norm_dict != None, "Normalization instructions must be provided with 'norm_dict'"

        self.X = X
        self.Y = Y
        self.d_path = d_path
        self.dm_path = dm_path
        self.test_by_chunks = test_by_chunks
        self.provide_Y = provide_Y
        self.convert_to_rgb = convert_to_rgb
        self.norm_dict = norm_dict
        self.multiple_raw_images = multiple_raw_images

        if not reduce_mem:
            self.dtype = np.float32  
            self.dtype_str = "float32"
        else:
            self.dtype = np.float16
            self.dtype_str = "float16"

        self.all_files_in_same_folder = not self.multiple_raw_images
        if self.multiple_raw_images:
            self.data_paths = sorted(next(os.walk(d_path))[1])
            if len(self.data_paths) == 0:
                self.all_files_in_same_folder = True
                print("Seems that even 'PROBLEM.IMAGE_TO_IMAGE.MULTIPLE_RAW_ONE_TARGET_LOADER' was selected the test "
                    "data is not organized in a folder each sample, so BiaPy is trying to load the data as all files"
                    "are within the same directory.")

        if self.all_files_in_same_folder:
            self.data_path = sorted(next(os.walk(d_path))[2]) if X is None else None
            if self.data_path is not None and len(self.data_path) == 0:
                self.data_path = sorted(next(os.walk(d_path))[1])
            if sample_ids is not None and self.data_path is not None:
                self.data_path = [x for i, x in enumerate(self.data_path) if i in sample_ids]
            if provide_Y:
                self.data_mask_path = sorted(next(os.walk(dm_path))[2]) if Y is None else None
                if self.data_mask_path is not None and len(self.data_mask_path) == 0:
                    self.data_mask_path = sorted(next(os.walk(dm_path))[1])
                if sample_ids is not None and self.data_mask_path is not None:
                    self.data_mask_path = [x for i, x in enumerate(self.data_mask_path) if i in sample_ids]
                    
                if self.data_path is not None and self.data_mask_path is not None:
                    if len(self.data_path) != len(self.data_mask_path):
                        raise ValueError("Different number of raw and ground truth items ({} vs {}). "
                            "Please check the data!".format(len(self.data_path), len(self.data_mask_path)))
        self.seed = seed
        self.ndim = ndim
        if X is None:
            if not self.all_files_in_same_folder:
                self.data = {}
                self.data_paths = sorted(next(os.walk(d_path))[1])
                if self.provide_Y:
                    self.data_mask_path = sorted(next(os.walk(dm_path))[1])
                    if len(self.data_paths) != len(self.data_mask_path):
                        raise ValueError("Different number of raw and ground truth items ({} vs {}). "
                            "Please check the data!".format(len(self.data_path), len(self.data_mask_path)))
                c = 0
                for i in range(len(self.data_paths)):
                    if self.provide_Y:
                        gt_image_path = next(os.walk(os.path.join(dm_path,self.data_mask_path[i])))[2][0]
                    associated_raw_image_dir = os.path.join(d_path, self.data_paths[i]) 
                    
                    samples = sorted(next(os.walk(associated_raw_image_dir))[2])
                    for j in range(len(samples)):
                        self.data[f"sample_{c}"] = {}
                        self.data[f"sample_{c}"]["raw"] = os.path.join(associated_raw_image_dir,samples[j])
                        if self.provide_Y:
                            self.data[f"sample_{c}"]["gt"] = os.path.join(dm_path,self.data_mask_path[i],gt_image_path)
                        c += 1

                self.sample_list = list(self.data.keys())
                self.len = len(self.sample_list)
                if self.len == 0:
                    raise ValueError("No image found in {}".format(d_path))
            else:
                self.len = len(self.data_path)
                if len(self.data_path) == 0:
                    if test_by_chunks:
                        print("No image found in {} folder. Assumming that files are zarr directories.")
                        self.data_path = sorted(next(os.walk(d_path))[1])
                        if provide_Y:
                            self.data_mask_path = sorted(next(os.walk(dm_path))[1])
                        if len(self.data_path) == 0:
                            raise ValueError("No zarr files found in {}".format(d_path))
                    else:
                        raise ValueError("No test image found in {}".format(d_path))
        else:
            self.len = len(X)
        
        # Check if a division is required
        if provide_Y:
            self.Y_norm = {}
            self.Y_norm['type'] = 'div'
        img, mask, xnorm, _, _ = self.load_sample(0)

        if norm_dict['enable']:
            self.norm_dict['orig_dtype'] = img.dtype if isinstance(img, np.ndarray) else "Zarr"

        if xnorm:
            self.norm_dict.update(xnorm)

        if mask is not None and not test_by_chunks:
            self.Y_norm = {}
            if norm_dict['mask_norm'] == 'as_mask':
                self.Y_norm['type'] = 'div'
                if (np.max(mask) > 30 and not instance_problem):
                    self.Y_norm['div'] = 1   
            elif norm_dict['mask_norm'] == 'as_image':
                self.Y_norm.update(self.norm_dict)

    def norm_X(self, img):
        """
        X data normalization.

        Parameters
        ----------
        img : 3D/4D Numpy array
            X element, for instance, an image. E.g. ``(y, x, channels)`` in ``2D`` and ``(z, y, x, channels)`` in ``3D``.

        Returns
        -------
        img : 3D/4D Numpy array
            X element normalized. E.g. ``(y, x, channels)`` in ``2D`` and ``(z, y, x, channels)`` in ``3D``.
        
        xnorm : dict, optional
            Normalization info. 
        """
        xnorm = None
        # Percentile clipping
        if 'lower_bound' in self.norm_dict:
            if self.norm_dict['application_mode'] == "image":
                img, _, _ = percentile_clip(img, lower=self.norm_dict['lower_bound'],                                     
                    upper=self.norm_dict['upper_bound'])
            elif self.norm_dict['application_mode'] == "dataset" and not self.norm_dict['clipped']:
                img, _, _ = percentile_clip(img, lwr_perc_val=self.norm_dict['dataset_X_lower_value'],                                     
                    uppr_perc_val=self.norm_dict['dataset_X_upper_value'])
        if self.norm_dict['type'] == 'div':
            img, xnorm = norm_range01(img, dtype=self.dtype)
        elif self.norm_dict['type'] == 'scale_range':
            img, xnorm = norm_range01(img, dtype=self.dtype, div_using_max_and_scale=True)
        elif self.norm_dict['type'] == 'custom':
            if self.norm_dict['application_mode'] == "image":
                xnorm = {}
                xnorm['mean'] = img.mean()
                xnorm['std'] = img.std()
                img = normalize(img, img.mean(), img.std(), out_type=self.dtype_str)
            else:
                img = normalize(img, self.norm_dict['mean'], self.norm_dict['std'], out_type=self.dtype_str)
        return img, xnorm

    def norm_Y(self, mask):   
        """
        Y data normalization.

        Parameters
        ----------
        mask : 3D/4D Numpy array
            Y element, for instance, an image's mask. E.g. ``(y, x, channels)`` in ``2D`` and ``(z, y, x, channels)`` in 
            ``3D``.

        Returns
        -------
        mask : 3D/4D Numpy array
            Y element normalized. E.g. ``(y, x, channels)`` in ``2D`` and ``(z, y, x, channels)`` in ``3D``.
        
        ynorm : dict, optional
            Normalization info.
        """
        ynorm = None
        if self.norm_dict['mask_norm'] == 'as_mask':
            if 'div' in self.Y_norm:
                mask = mask/255
        elif self.norm_dict['mask_norm'] == 'as_image':
            # Percentile clipping
            if 'lower_bound' in self.norm_dict:
                if self.norm_dict['application_mode'] == "image":
                    mask, _, _ = percentile_clip(mask, lower=self.norm_dict['lower_bound'],                                     
                        upper=self.norm_dict['upper_bound'])
                elif self.norm_dict['application_mode'] == "dataset" and not self.norm_dict['clipped']:
                    mask, _, _ = percentile_clip(mask, lower=self.norm_dict['dataset_X_lower_value'],                                     
                        upper=self.norm_dict['dataset_X_upper_value'])
                        
            if self.norm_dict['type'] == 'div':
                mask, ynorm = norm_range01(mask, dtype=self.dtype)
            elif self.norm_dict['type'] == 'scale_range':
                mask, xnorm = norm_range01(mask, dtype=self.dtype, div_using_max_and_scale=True)
            elif self.norm_dict['type'] == 'custom':
                if self.norm_dict['application_mode'] == "image":
                    ynorm = {}
                    ynorm['mean'] = mask.mean()
                    ynorm['std'] = mask.std()
                    mask = normalize(mask, mask.mean(), mask.std(), out_type=self.dtype_str)
                else:
                    mask = normalize(mask, self.norm_dict['mean'], self.norm_dict['std'], out_type=self.dtype_str)
        return mask, ynorm

    def load_sample(self, idx):
        """Load one data sample given its corresponding index."""
        mask, ynorm, filename = None, None, None
        # Choose the data source
        if self.X is None:
            if not self.all_files_in_same_folder:
                k = self.sample_list[idx]
                img = imread(self.data[k]["raw"])
                img = np.squeeze(img)
                filename = self.data[k]["raw"]
                if self.provide_Y:
                    mask = imread(self.data[k]["gt"])
                    mask = np.squeeze(mask)   
            else:
                filename = os.path.join(self.d_path, self.data_path[idx])
                if self.data_path[idx].endswith('.npy'):
                    img = np.load(os.path.join(self.d_path, self.data_path[idx]))
                    if self.provide_Y:
                        mask = np.load(os.path.join(self.dm_path, self.data_mask_path[idx]))
                elif self.data_path[idx].endswith('.hdf5') or self.data_path[idx].endswith('.h5'):
                    if not self.test_by_chunks:
                        img = h5py.File(os.path.join(self.d_path, self.data_path[idx]),'r')
                        img = img[list(img)[0]]
                        if self.provide_Y:
                            mask = h5py.File(os.path.join(self.dm_path, self.data_mask_path[idx]),'r')
                            mask = mask[list(mask)[0]]
                    else:
                        img = os.path.join(self.d_path, self.data_path[idx])
                        if self.provide_Y:
                            mask = os.path.join(self.dm_path, self.data_mask_path[idx])
                elif self.data_path[idx].endswith('.zarr'):
                    if self.test_by_chunks:
                        img = os.path.join(self.d_path, self.data_path[idx])
                        if self.provide_Y:
                            mask = os.path.join(self.dm_path, self.data_mask_path[idx])
                    else:
                        raise ValueError("If you are using Zarr images please set 'TEST.BY_CHUNKS.ENABLE' and configure "
                            "its options.")
                else:
                    img = imread(os.path.join(self.d_path, self.data_path[idx]))
                    img = np.squeeze(img)
                    if self.provide_Y:
                        mask = imread(os.path.join(self.dm_path, self.data_mask_path[idx]))
                        mask = np.squeeze(mask)  
        else:
            filename = idx
            img = self.X[idx]
            img = np.squeeze(img)

            if self.provide_Y:
                mask = self.Y[idx]
                mask = np.squeeze(mask)

        # Correct dimensions 
        if not self.test_by_chunks:
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

        xnorm = self.norm_dict 
        if not self.test_by_chunks:
            # Normalization
            if self.norm_dict['enable']:
                img, xnorm = self.norm_X(img)
            if self.provide_Y and self.norm_dict['enable']:
                mask, ynorm = self.norm_Y(mask)

            img = np.expand_dims(img, 0).astype(self.dtype)
            if self.provide_Y:
                mask = np.expand_dims(mask, 0)
                if self.norm_dict['mask_norm'] == 'as_mask':
                    mask = mask.astype(np.uint8)

        if self.convert_to_rgb and img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
            
        return img, mask, xnorm, ynorm, filename


    def __len__(self):
        """Defines the length of the generator"""
        return self.len


    def __getitem__(self, index):
        """
        Generation of one pair of data.

        Parameters
        ----------
        index : int
            Sample index counter.

        Returns
        -------
        dict : dict
            Dictionary containing:

            img : 4D/5D Numpy array
                X element, for instance, an image. E.g. ``(1, z, y, x, channels)`` if ``2D`` or 
                ``(1, y, x, channels)`` if ``3D``. 
                
            X_norm : dict
                X element normalization steps.

            file : str or int
                Processed image file path or integer position in loaded data.
            
            mask : 4D/5D Numpy array, optional
                Y element, for instance, a mask. E.g. ``(1, z, y, x, channels)`` if ``2D`` or 
                    ``(1, y, x, channels)`` if ``3D``.
            
            Y_norm : dict, optional
                Y element normalization steps.
        """
        img, mask, xnorm, ynorm, filename = self.load_sample(index)
        
        if xnorm is not None:
            self.norm_dict.update(xnorm)
        if ynorm is not None:
            self.Y_norm.update(ynorm)

        if self.provide_Y:
            return {"X": img, "X_norm": self.norm_dict, "Y": mask, "Y_norm": self.Y_norm, "file": filename}
        else:
            return {"X": img, "X_norm": self.norm_dict, "file": filename}

    def get_data_normalization(self):
        return self.norm_dict
