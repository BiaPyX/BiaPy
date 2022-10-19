from abc import ABCMeta, abstractmethod
import tensorflow as tf
import numpy as np
import random
import os
import sys
import imgaug as ia
from tqdm import tqdm
from imgaug import augmenters as iaa
from skimage.io import imread
from imgaug.augmentables.heatmaps import HeatmapsOnImage

from data.data_2D_manipulation import random_crop
from data.data_3D_manipulation import random_3D_crop
from engine.denoising import (rand_float_coords2D, rand_float_coords3D, get_stratified_coords2D, get_stratified_coords3D, get_value_manipulation,                         
                              apply_structN2Vmask, apply_structN2Vmask3D)                    
from utils.util import img_to_onehot_encoding, normalize
from data.generators.augmentors import (cutout, cutblur, cutmix, cutnoise, misalignment, brightness_em, contrast_em,
                                        brightness, contrast, missing_parts, shuffle_channels, grayscale, GridMask)

class BaseDataGenerator(tf.keras.utils.Sequence, metaclass=ABCMeta):
    """Custom 2D BaseDataGenerator based on `imgaug <https://github.com/aleju/imgaug-doc>`_
       and our own `augmentors.py <https://github.com/danifranco/BiaPy/blob/master/generators/augmentors.py>`_
       transformations. 

       Based on `microDL <https://github.com/czbiohub/microDL>`_ and
       `Shervine's blog <https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly>`_.

       Parameters
       ----------
       X : 4D Numpy array
           Data. E.g. ``(num_of_images, y, x, channels)``.

       Y : 4D Numpy array
           Mask data. E.g. ``(num_of_images, y, x, 1)``.

       batch_size : int, optional
           Size of the batches.

       seed : int, optional
           Seed for random functions.

       shuffle_each_epoch : bool, optional
           To decide if the indexes will be shuffled after every epoch.

       in_memory : bool, optional
           If ``True`` data used will be ``X`` and ``Y``. If ``False`` it will be loaded directly from disk using
           ``data_paths``.

       data_paths : List of str, optional
          If ``in_memory`` is ``True`` this list should contain the paths to load data and masks. ``data_paths[0]``
          should be data path and ``data_paths[1]`` masks path.

       da : bool, optional
           To activate the data augmentation.

       da_prob : float, optional
               Probability of doing each transformation.

       rotation90 : bool, optional
           To make square (90, 180,270) degree rotations.

       rand_rot : bool, optional
           To make random degree range rotations.

       rnd_rot_range : tuple of float, optional
           Range of random rotations. E. g. ``(-180, 180)``.

       shear : bool, optional
           To make shear transformations.

       shear_range : tuple of int, optional
           Degree range to make shear. E. g. ``(-20, 20)``.

       zoom : bool, optional
           To make zoom on images.

       zoom_range : tuple of floats, optional
           Zoom range to apply. E. g. ``(0.8, 1.2)``.

       shift : float, optional
           To make shifts.

       shift_range : tuple of float, optional
           Range to make a shift. E. g. ``(0.1, 0.2)``.

       affine_mode: str, optional
           Method to use when filling in newly created pixels. Same meaning as in `skimage` (and `numpy.pad()`).
           E.g. ``constant``, ``reflect`` etc.

       vflip : bool, optional
           To activate vertical flips.

       hflip : bool, optional
           To activate horizontal flips.

       elastic : bool, optional
           To make elastic deformations.

       e_alpha : tuple of ints, optional
            Strength of the distortion field. E. g. ``(240, 250)``.

       e_sigma : int, optional
           Standard deviation of the gaussian kernel used to smooth the distortion fields.

       e_mode : str, optional
           Parameter that defines the handling of newly created pixels with the elastic transformation.

       g_blur : bool, optional
           To insert gaussian blur on the images.

       g_sigma : tuple of floats, optional
           Standard deviation of the gaussian kernel. E. g. ``(1.0, 2.0)``.

       median_blur : bool, optional
           To blur an image by computing median values over neighbourhoods.

       mb_kernel : tuple of ints, optional
           Median blur kernel size. E. g. ``(3, 7)``.

       motion_blur : bool, optional
           Blur images in a way that fakes camera or object movements.

       motb_k_range : int, optional
           Kernel size to use in motion blur.

       gamma_contrast : bool, optional
           To insert gamma constrast changes on images.

       gc_gamma : tuple of floats, optional
           Exponent for the contrast adjustment. Higher values darken the image. E. g. ``(1.25, 1.75)``.

       brightness : bool, optional
           To aply brightness to the images as `PyTorch Connectomics
           <https://github.com/zudi-lin/pytorch_connectomics/blob/master/connectomics/data/augmentation/grayscale.py>`_.

       brightness_factor : tuple of 2 floats, optional
           Strength of the brightness range, with valid values being ``0 <= brightness_factor <= 1``. E.g. ``(0.1, 0.3)``.

       brightness_mode : str, optional
           Apply same brightness change to the whole image or diffent to slice by slice.

       contrast : boolen, optional
           To apply contrast changes to the images as `PyTorch Connectomics
           <https://github.com/zudi-lin/pytorch_connectomics/blob/master/connectomics/data/augmentation/grayscale.py>`_.

       contrast_factor : tuple of 2 floats, optional
           Strength of the contrast change range, with valid values being ``0 <= contrast_factor <= 1``.
           E.g. ``(0.1, 0.3)``.

       contrast_mode : str, optional
           Apply same contrast change to the whole image or diffent to slice by slice.

       brightness_em : bool, optional
           To aply brightness to the images as `PyTorch Connectomics
           <https://github.com/zudi-lin/pytorch_connectomics/blob/master/connectomics/data/augmentation/grayscale.py>`_.

       brightness_em_factor : tuple of 2 floats, optional
           Strength of the brightness range, with valid values being ``0 <= brightness_em_factor <= 1``. E.g. ``(0.1, 0.3)``.

       brightness_em_mode : str, optional
           Apply same brightness change to the whole image or diffent to slice by slice.

       contrast_em : boolen, optional
           To apply contrast changes to the images as `PyTorch Connectomics
           <https://github.com/zudi-lin/pytorch_connectomics/blob/master/connectomics/data/augmentation/grayscale.py>`_.

       contrast_em_factor : tuple of 2 floats, optional
           Strength of the contrast change range, with valid values being ``0 <= contrast_em_factor <= 1``.
           E.g. ``(0.1, 0.3)``.

       contrast_em_mode : str, optional
           Apply same contrast change to the whole image or diffent to slice by slice.

       dropout : bool, optional
           To set a certain fraction of pixels in images to zero.

       drop_range : tuple of floats, optional
           Range to take a probability ``p`` to drop pixels. E.g. ``(0, 0.2)`` will take a ``p`` folowing ``0<=p<=0.2``
           and then drop ``p`` percent of all pixels in the image (i.e. convert them to black pixels).

       cutout : bool, optional
           To fill one or more rectangular areas in an image using a fill mode.

       cout_nb_iterations : tuple of ints, optional
           Range of number of areas to fill the image with. E. g. ``(1, 3)``.

       cout_size : tuple of floats, optional
           Range to select the size of the areas in % of the corresponding image size. Values between ``0`` and ``1``.
           E. g. ``(0.2, 0.4)``.

       cout_cval : int, optional
           Value to fill the area of cutout with.

       cout_apply_to_mask : boolen, optional
           Whether to apply cutout to the mask.

       cutblur : boolean, optional
           Blur a rectangular area of the image by downsampling and upsampling it again.

       cblur_size : tuple of floats, optional
           Range to select the size of the area to apply cutblur on. E. g. ``(0.2, 0.4)``.

       cblur_inside : boolean, optional
           If ``True`` only the region inside will be modified (cut LR into HR image). If ``False`` the ``50%`` of the
           times the region inside will be modified (cut LR into HR image) and the other ``50%`` the inverse will be
           done (cut HR into LR image). See Figure 1 of the official `paper <https://arxiv.org/pdf/2004.00448.pdf>`_.

       cutmix : boolean, optional
           Combine two images pasting a region of one image to another.

       cmix_size : tuple of floats, optional
           Range to select the size of the area to paste one image into another. E. g. ``(0.2, 0.4)``.

       cnoise : boolean, optional
           Randomly add noise to a cuboid region in the image.

       cnoise_scale : tuple of floats, optional
           Range to choose a value that will represent the % of the maximum value of the image that will be used as the
           std of the Gaussian Noise distribution. E.g. ``(0.1, 0.2)``.

       cnoise_nb_iterations : tuple of ints, optional
           Number of areas with noise to create. E.g. ``(1, 3)``.

       cnoise_size : tuple of floats, optional
           Range to choose the size of the areas to transform. E.g. ``(0.2, 0.4)``.

       misalignment : boolean, optional
           To add miss-aligment augmentation.

       ms_displacement : int, optional
           Maximum pixel displacement in `xy`-plane for misalignment.

       ms_rotate_ratio : float, optional
           Ratio of rotation-based mis-alignment

       missing_parts : boolean, optional
           Augment the image by creating a black line in a random position.

       missp_iterations : tuple of 2 ints, optional
           Iterations to dilate the missing line with. E.g. ``(30, 40)``.

       grayscale : bool, optional
           Whether to augment images converting partially in grayscale.

       gridmask : bool, optional
           Whether to apply gridmask to the image. See the official `paper <https://arxiv.org/abs/2001.04086v1>`_ for
           more information about it and its parameters.

       grid_ratio : float, optional
           Determines the keep ratio of an input image (``r`` in the original paper).

       grid_d_range : tuple of floats, optional
           Range to choose a ``d`` value. It represents the % of the image size. E.g. ``(0.4,1)``.

       grid_rotate : float, optional
           Rotation of the mask in GridMask. Needs to be between ``[0,1]`` where 1 is 360 degrees.

       grid_invert : bool, optional
           Whether to invert the mask of GridMask.

       channel_shuffle : bool, optional
           Whether to shuflle the channels of the images.

       random_crops_in_DA : bool, optional
           Decide to make random crops in DA (before transformations).

       shape : 3D int tuple, optional
           Shape of the desired images when using 'random_crops_in_DA'.

       resolution : 2D tuple of floats, optional
           Resolution of the given data ``(y,x)``. E.g. ``(8,8)``.

       prob_map : 4D Numpy array or str, optional
           If it is an array, it should represent the probability map used to make random crops when
           ``random_crops_in_DA`` is set. If str given should be the path to read these maps from.

       val : bool, optional
           Advise the generator that the images will be to validate the model to not make random crops (as the val.
           data must be the same on each epoch). Valid when ``random_crops_in_DA`` is set.

       n_classes : int, optional
           Number of classes. If ``> 1`` one-hot encoding will be done on the ground truth.

       out_number : int, optional
           Number of output returned by the network. Used to produce same number of ground truth data on each batch.

       extra_data_factor : int, optional
           Factor to multiply the batches yielded in a epoch. It acts as if ``X`` and ``Y``` where concatenated
           ``extra_data_factor`` times.

       n2v : bool, optional
           Whether to create `Noise2Void <https://openaccess.thecvf.com/content_CVPR_2019/papers/Krull_Noise2Void_-_Learning_Denoising_From_Single_Noisy_Images_CVPR_2019_paper.pdf>`_
           mask. Used in DENOISING problem type. 
        
       n2v_perc_pix : float, optional
           Input image pixels to be manipulated. 

       n2v_manipulator : str, optional
           How to manipulate the input pixels. Most pixel manipulators will compute the replacement value based on a neighborhood.
           Possible options: `normal_withoutCP`: samples the neighborhood according to a normal gaussian distribution, but without 
           the center pixel; `normal_additive`: adds a random number to the original pixel value. The random number is sampled from 
           a gaussian distribution with zero-mean and sigma = `n2v_neighborhood_radius` ; `normal_fitted`: uses a random value from 
           a gaussian normal distribution with mean equal to the mean of the neighborhood and standard deviation equal to the 
           standard deviation of the neighborhood ; `identity`: performs no pixel manipulation.

       n2v_neighborhood_radius : int, optional
           Neighborhood size to use when manipulating the values. 

       n2v_structMask : Array of ints, optional
           Masking kernel for StructN2V to hide pixels adjacent to main blind spot. Value 1 = 'hidden', Value 0 = 'non hidden'. 
           Nested lists equivalent to ndarray. Must have odd length in each dimension (center pixel is blind spot). ``None`` 
           implies normal N2V masking.

       max_samples_to_analyse : int, optinal
           Samples to be analysed when the data is not in memory and mean/std need to be calculated. 
       
       norm_custom_mean : float, optional
           Mean of the data used to normalize.

       norm_custom_std : float, optional
           Std of the data used to normalize.
       
       instance_problem : bool, optional
           To not divide the labels if being in an instance segmenation problem.
    """

    def __init__(self, ndim, X, Y, batch_size=32, seed=0, shuffle_each_epoch=False, in_memory=True, data_paths=None, da=True,
                 da_prob=0.5, rotation90=False, rand_rot=False, rnd_rot_range=(-180,180), shear=False,
                 shear_range=(-20,20), zoom=False, zoom_range=(0.8,1.2), shift=False, shift_range=(0.1,0.2),
                 affine_mode='constant', vflip=False, hflip=False, elastic=False, e_alpha=(240,250), e_sigma=25,
                 e_mode='constant', g_blur=False, g_sigma=(1.0,2.0), median_blur=False, mb_kernel=(3,7), motion_blur=False,
                 motb_k_range=(3,8), gamma_contrast=False, gc_gamma=(1.25,1.75), brightness=False, brightness_factor=(1,3),
                 brightness_mode='2D', contrast=False, contrast_factor=(1,3), contrast_mode='2D', brightness_em=False,
                 brightness_em_factor=(1,3), brightness_em_mode='2D', contrast_em=False, contrast_em_factor=(1,3),
                 contrast_em_mode='2D', dropout=False, drop_range=(0, 0.2), cutout=False, cout_nb_iterations=(1,3),
                 cout_size=(0.2,0.4), cout_cval=0, cout_apply_to_mask=False, cutblur=False, cblur_size=(0.1,0.5),
                 cblur_down_range=(2,8), cblur_inside=True, cutmix=False, cmix_size=(0.2,0.4), cutnoise=False,
                 cnoise_scale=(0.1,0.2), cnoise_nb_iterations=(1,3), cnoise_size=(0.2,0.4), misalignment=False,
                 ms_displacement=16, ms_rotate_ratio=0.0, missing_parts=False, missp_iterations=(30, 40),
                 grayscale=False, channel_shuffle=False, gridmask=False, grid_ratio=0.6, grid_d_range=(0.4,1),
                 grid_rotate=1, grid_invert=False, random_crops_in_DA=False, shape=(256,256,1), resolution=(-1,),
                 prob_map=None, val=False, n_classes=1, out_number=1, extra_data_factor=1, n2v=False, 
                 n2v_perc_pix=0.198, n2v_manipulator='uniform_withCP', n2v_neighborhood_radius=5, 
                 n2v_structMask=np.array([[0,1,1,1,1,1,1,1,1,1,0]]), max_samples_to_analyse=10, 
                 norm_custom_mean=None, norm_custom_std=None, instance_problem=False):

        self.ndim = ndim
        self.z_size = -1 
        
        if in_memory:
            if X.ndim != (self.ndim+2) or Y.ndim != (self.ndim+2):
                raise ValueError("X and Y must be a {}D Numpy array".format((self.ndim+1)))
            if X.shape[:(self.ndim+1)] != Y.shape[:(self.ndim+1)]:
                raise ValueError("The shape of X and Y must be the same. {} != {}".format(X.shape[:(self.ndim+1)], Y.shape[:(self.ndim+1)]))

        if in_memory and (X is None or Y is None):
            raise ValueError("'X' and 'Y' need to be provided together with 'in_memory'")

        if not in_memory and len(data_paths) != 2:
            raise ValueError("'data_paths' must contain the following paths: 1) data path ; 2) data masks path")

        if len(resolution) == 1 and resolution[0] == -1:
            resolution = (1,1) if self.ndim == 2 else (1,1,1)
        
        if len(resolution) != self.ndim:
            raise ValueError("The resolution needs to be a tuple with {} values".format(self.ndim))

        if random_crops_in_DA:
            if shape is None:
                raise ValueError("'shape' must be provided when 'random_crops_in_DA is enabled")
            if in_memory:
                if ndim == 3:
                    if shape[0] > X.shape[1] or shape[1] > X.shape[2] or shape[2] > X.shape[3]:
                        raise ValueError("Given 'shape' is bigger than the data provided")
                else:
                    if shape[0] > X.shape[1] or shape[1] > X.shape[2]:
                        raise ValueError("Given 'shape' is bigger than the data provided")
                    if shape[0] != shape[1]:
                        raise ValueError("When 'random_crops_in_DA' is selected the shape given must be square, e.g. (256, 256, 1)")

        if not in_memory and not random_crops_in_DA:
            print("WARNING: you are going to load samples from disk (as 'in_memory=False') and "
                  "'random_crops_in_DA=False' so all samples are expected to have the same shape. If it is not "
                  "the case set batch_size to 1 or the generator will throw an error")

        if rotation90 and rand_rot:
            print("Warning: you selected double rotation type. Maybe you should set only 'rand_rot'?")

        self.batch_size = batch_size
        self.in_memory = in_memory
        if not in_memory:
            # Save paths where the data is stored
            self.paths = data_paths
            self.data_paths = sorted(next(os.walk(data_paths[0]))[2])
            self.data_mask_path = sorted(next(os.walk(data_paths[1]))[2])
            self.len = len(self.data_paths)
            
            # Check if a division is required
            self.X_norm = {}
            self.X_norm['type'] = 'div'
            img, _ = self.load_sample(0)
            self.X_channels = img.shape[-1]
            if np.max(img) > 100:
                self.X_norm['div'] = 1  

            self.shape = shape if random_crops_in_DA else img.shape

            # X data analysis
            if norm_custom_mean is not None and norm_custom_std is not None:
                nsamples = min(max_samples_to_analyse, len(self.data_paths))
                sam = []
                for i in range(np.random.choice(len(self.data_paths), nsamples, replace=False)):
                    img, _ = self.load_sample(i)
                    sam.append(img)
                sam = np.array(sam)
                self.X_norm['type'] = 'custom'
                self.X_norm['mean'] = np.mean(sam)
                self.X_norm['std'] = np.std(sam)
                del sam
            del img

            # Y data analysis
            self.first_no_bin_channel = -1
            self.div_Y_on_load_bin_channels = False
            self.div_Y_on_load_no_bin_channels = False
            found = False
            # Loop over a few masks to ensure foreground class is present to decide normalization
            for i in range(min(10,len(self.data_mask_path))):
                _, mask = self.load_sample(i)
                
                # Store wheter all channels of the gt are binary or not (i.e. distance transform channel)
                if not found and (img.dtype is np.dtype(np.float32) or img.dtype is np.dtype(np.float64)) and instance_problem:
                    for j in range(img.shape[-1]):
                        if len(np.unique(img[...,j])) > 2:
                            self.first_no_bin_channel = j
                            found = True
                            break

                # If found high values divide masks
                if self.first_no_bin_channel != -1:
                    if self.first_no_bin_channel != 0:
                        if np.max(img[...,:self.first_no_bin_channel]) > 100: self.div_Y_on_load_bin_channels = True
                        if np.max(img[...,self.first_no_bin_channel:]) > 100: self.div_Y_on_load_no_bin_channels = True
                    else:
                        if np.max(img) > 100: self.div_Y_on_load_bin_channels = True
                        if np.max(img) > 100: self.div_Y_on_load_no_bin_channels = True 
                else:
                    if np.max(img) > 100: self.div_Y_on_load_bin_channels = True 

            self.Y_channels = mask.shape[-1]
            self.Y_dtype = mask.dtype
            del mask
        else:
            self.X = X
            self.Y = Y
            self.Y_channels = Y.shape[-1]
            self.Y_dtype = Y.dtype
            self.X_channels = X.shape[-1]
            self.len = len(self.X)
            self.shape = shape if random_crops_in_DA else X.shape[1:]

            # X data analysis
            self.X_norm = {}
            if norm_custom_mean is not None and norm_custom_std is not None:
                self.X_norm['type'] = 'custom'
                self.X_norm['mean'] = np.mean(self.X)
                self.X_norm['std'] = np.std(self.X)
            else:
                self.X_norm['type'] = 'div'
                if np.max(X) > 100:
                    self.X_norm['div'] = 1

            # Y data analysis
            self.first_no_bin_channel = -1
            self.div_Y_on_load_bin_channels = False
            self.div_Y_on_load_no_bin_channels = False
            if (self.Y.dtype is np.dtype(np.float32) or self.Y.dtype is np.dtype(np.float64)) and instance_problem:
                for i in range(self.Y.shape[-1]):
                    if len(np.unique(self.Y[...,i])) > 2:
                        self.first_no_bin_channel = i
                        break
            if self.first_no_bin_channel != -1:
                if self.first_no_bin_channel != 0:
                    self.div_Y_on_load_bin_channels = True if np.max(self.Y[...,:self.first_no_bin_channel]) > 100 else False
                    self.div_Y_on_load_no_bin_channels = True if np.max(self.Y[...,self.first_no_bin_channel:]) > 100 else False
                else:
                    self.div_Y_on_load_bin_channels = False
                    self.div_Y_on_load_no_bin_channels = True if np.max(self.Y) > 100 else False
            else:
                self.div_Y_on_load_bin_channels = True if np.max(self.Y) > 100 else False
        
        if self.ndim == 2:
            resolution = tuple(resolution[i] for i in [1, 0]) # y, x -> x, y
            self.res_relation = (1.0,resolution[0]/resolution[1])
        else:
            resolution = tuple(resolution[i] for i in [2, 1, 0]) # z, y, x -> x, y, z
            self.res_relation = (1.0,resolution[0]/resolution[1],resolution[0]/resolution[2])
        self.resolution = resolution
        self.o_indexes = np.arange(self.len)
        self.shuffle = shuffle_each_epoch
        self.n_classes = n_classes
        self.out_number = out_number
        self.da = da
        self.da_prob = da_prob
        self.random_crops_in_DA = random_crops_in_DA
        self.cutout = cutout
        self.cout_nb_iterations = cout_nb_iterations
        self.cout_size = cout_size
        self.cout_cval = cout_cval
        self.cout_apply_to_mask = cout_apply_to_mask
        self.cutblur = cutblur
        self.cblur_size = cblur_size
        self.cblur_down_range = cblur_down_range
        self.cblur_inside = cblur_inside
        self.cutmix = cutmix
        self.cmix_size = cmix_size
        self.cutnoise = cutnoise
        self.cnoise_scale = cnoise_scale
        self.cnoise_nb_iterations = cnoise_nb_iterations
        self.cnoise_size = cnoise_size
        self.misalignment = misalignment
        self.ms_displacement = ms_displacement
        self.ms_rotate_ratio = ms_rotate_ratio
        self.brightness = brightness
        self.contrast = contrast
        self.brightness_em = brightness_em
        self.contrast_em = contrast_em
        self.missing_parts = missing_parts
        self.missp_iterations = missp_iterations
        self.grayscale = grayscale
        self.gridmask = gridmask
        self.grid_ratio = grid_ratio
        self.grid_d_range = grid_d_range
        self.grid_rotate = grid_rotate
        self.grid_invert = grid_invert
        self.grid_d_size = (self.shape[0]*grid_d_range[0], self.shape[1]*grid_d_range[1])
        self.channel_shuffle = channel_shuffle

        # Denoising options
        self.n2v = n2v
        if self.n2v:
            self.box_size = np.round(np.sqrt(100/n2v_perc_pix)).astype(np.int)
            self.get_stratified_coords = get_stratified_coords2D if self.ndim == 2 else get_stratified_coords3D
            self.rand_float = rand_float_coords2D(self.box_size) if self.ndim == 2 else rand_float_coords3D(self.box_size) 
            self.value_manipulation = get_value_manipulation(n2v_manipulator, n2v_neighborhood_radius)
            self.n2v_structMask = n2v_structMask 
            self.apply_structN2Vmask_func = apply_structN2Vmask if self.ndim == 2 else apply_structN2Vmask3D
            self.Y_shape = self.shape[:self.ndim] + (self.Y_channels*2,)
        else:
            self.Y_shape = self.shape[:self.ndim]+(self.Y_channels,)

        self.prob_map = None
        if random_crops_in_DA and prob_map is not None:
            if isinstance(prob_map, str):
                f = sorted(next(os.walk(prob_map))[2])
                self.prob_map = []
                for i in range(len(f)):
                    self.prob_map.append(os.path.join(prob_map, f[i]))
            else:
                self.prob_map = prob_map

        self.val = val
        if extra_data_factor > 1:
            self.extra_data_factor = extra_data_factor
            self.o_indexes = np.concatenate([self.o_indexes]*extra_data_factor)
        else:
            self.extra_data_factor = 1
        self.total_batches_seen = 0

        self.da_options = []
        self.trans_made = ''
        if rotation90:
            self.da_options.append(iaa.Sometimes(da_prob, iaa.Rot90((1, 3))))
            self.trans_made += '_rot[90,180,270]'
        if rand_rot:
            self.da_options.append(iaa.Sometimes(da_prob, iaa.Affine(rotate=rnd_rot_range, mode=affine_mode)))
            self.trans_made += '_rrot'+str(rnd_rot_range)
        if shear:
            self.da_options.append(iaa.Sometimes(da_prob, iaa.Affine(rotate=shear_range, mode=affine_mode)))
            self.trans_made += '_shear'+str(shear_range)
        if zoom:
            self.da_options.append(iaa.Sometimes(da_prob, iaa.Affine(scale={"x": zoom_range, "y": zoom_range}, mode=affine_mode)))
            self.trans_made += '_zoom'+str(zoom_range)
        if shift:
            self.da_options.append(iaa.Sometimes(da_prob, iaa.Affine(translate_percent=shift_range, mode=affine_mode)))
            self.trans_made += '_shift'+str(shift_range)
        if vflip:
            self.da_options.append(iaa.Flipud(da_prob))
            self.trans_made += '_vflip'
        if hflip:
            self.da_options.append(iaa.Fliplr(da_prob))
            self.trans_made += '_hflip'
        if elastic:
            self.da_options.append(iaa.Sometimes(da_prob,iaa.ElasticTransformation(alpha=e_alpha, sigma=e_sigma, mode=e_mode)))
            self.trans_made += '_elastic'+str(e_alpha)+'+'+str(e_sigma)+'+'+str(e_mode)
        if g_blur:
            self.da_options.append(iaa.Sometimes(da_prob,iaa.GaussianBlur(g_sigma)))
            self.trans_made += '_gblur'+str(g_sigma)
        if median_blur:
            self.da_options.append(iaa.Sometimes(da_prob,iaa.MedianBlur(k=mb_kernel)))
            self.trans_made += '_mblur'+str(mb_kernel)
        if motion_blur:
            self.da_options.append(iaa.Sometimes(da_prob,iaa.MotionBlur(k=motb_k_range)))
            self.trans_made += '_motb'+str(motb_k_range)
        if gamma_contrast:
            self.da_options.append(iaa.Sometimes(da_prob,iaa.GammaContrast(gc_gamma)))
            self.trans_made += '_gcontrast'+str(gc_gamma)
        if brightness:
            self.brightness_factor = brightness_factor
            self.brightness_mode = brightness_mode # Not used
            self.trans_made += '_brightness'+str(brightness_factor)
        if contrast:
            self.contrast_factor = contrast_factor
            self.contrast_mode = contrast_mode # Not used
            self.trans_made += '_contrast'+str(contrast_factor)
        if brightness_em:
            self.brightness_em_factor = brightness_em_factor
            self.brightness_em_mode = brightness_em_mode # Not used
            self.trans_made += '_brightness_em'+str(brightness_em_factor)
        if contrast_em:
            self.contrast_em_factor = contrast_em_factor
            self.contrast_em_mode = contrast_em_mode # Not used
            self.trans_made += '_contrast_em'+str(contrast_em_factor)
        if dropout:
            self.da_options.append(iaa.Sometimes(da_prob, iaa.Dropout(p=drop_range)))
            self.trans_made += '_drop'+str(drop_range)

        if grayscale: self.trans_made += '_gray'
        if gridmask: self.trans_made += '_gridmask'+str(self.grid_ratio)+'+'+str(self.grid_d_range)+'+'+str(self.grid_rotate)+'+'+str(self.grid_invert)
        if channel_shuffle: self.trans_made += '_chshuffle'
        if cutout: self.trans_made += '_cout'+str(cout_nb_iterations)+'+'+str(cout_size)+'+'+str(cout_cval)+'+'+str(cout_apply_to_mask)
        if cutblur: self.trans_made += '_cblur'+str(cblur_size)+'+'+str(cblur_down_range)+'+'+str(cblur_inside)
        if cutmix: self.trans_made += '_cmix'+str(cmix_size)
        if cutnoise: self.trans_made += '_cnoi'+str(cnoise_scale)+'+'+str(cnoise_nb_iterations)+'+'+str(cnoise_size)
        if misalignment: self.trans_made += '_msalg'+str(ms_displacement)+'+'+str(ms_rotate_ratio)
        if missing_parts: self.trans_made += '_missp'+'+'+str(missp_iterations)

        self.trans_made = self.trans_made.replace(" ", "")
        self.seq = iaa.Sequential(self.da_options)
        self.seed = seed
        ia.seed(seed)
        
        self.random_crop_func = random_3D_crop if self.ndim == 2 else random_crop

        self.on_epoch_end()

    @abstractmethod
    def get_transformed_samples(self, num_examples, save_to_dir=False, out_dir='aug', 
        train=True, random_images=True, draw_grid=True):
        NotImplementedError

    @abstractmethod
    def save_aug_samples(self, img, mask, orig_images, i, pos, out_dir, draw_grid, point_dict):
        NotImplementedError

    @abstractmethod
    def ensure_shape(self, img, mask):
        NotImplementedError

    @abstractmethod
    def apply_imgaug(self, image, mask, heat):
        NotImplementedError

    def __len__(self):
        """Defines the number of batches per epoch."""
        return int(np.ceil((self.len*self.extra_data_factor)/self.batch_size))

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        ia.seed(self.seed + self.total_batches_seen)
        self.indexes = self.o_indexes
        if self.shuffle:
            random.Random(self.seed + self.total_batches_seen).shuffle(self.indexes)

    def load_sample(self, idx):
        """Load one data sample given its corresponding index."""
        # Choose the data source
        if self.in_memory:
            img = self.X[idx]
            mask = self.Y[idx]
        else:
            if self.data_paths[idx].endswith('.npy'):
                img = np.load(os.path.join(self.paths[0], self.data_paths[idx]))
                mask = np.load(os.path.join(self.paths[1], self.data_mask_path[idx]))
            else:
                img = imread(os.path.join(self.paths[0], self.data_paths[idx]))
                mask = imread(os.path.join(self.paths[1], self.data_mask_path[idx]))

        img, mask = self.ensure_shape(img, mask)

        # X normalization
        if self.X_norm['type'] == 'div':
            if 'div' in self.X_norm:
                img = img/255
        elif self.X_norm['type'] == 'custom':
            img = normalize(img, self.X_norm['mean'], self.X_norm['std'])

        # Y normalization    
        if self.first_no_bin_channel != -1:
            if self.div_Y_on_load_bin_channels:
                mask[...,:self.first_no_bin_channel] = mask[...,:self.first_no_bin_channel]/255
            if self.div_Y_on_load_no_bin_channels:
                if self.first_no_bin_channel != 0:
                    mask[...,self.first_no_bin_channel:] = mask[...,self.first_no_bin_channel:]/255
                else:
                    mask = mask/255
        else:
            if self.div_Y_on_load_bin_channels: mask = mask/255

        return img, mask

    def __getitem__(self, index):
        """Generation of one batch of data.

           Parameters
           ----------
           index : int
               Batch index counter.

           Returns
           -------
           batch_x : 4D/5D Numpy array
               Corresponding X elements of the batch. E.g. ``(batch_size_value, z, y, x, channels)`` if 2D 
               or ``(batch_size_value, y, x, channels)`` if 3D. 
               
           batch_y : 4D/5D Numpy array
               Corresponding Y elements of the batch. E.g. ``(batch_size_value, z, y, x, channels)`` if 2D 
               or ``(batch_size_value, y, x, channels)`` if 3D.
        """
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_x = np.zeros((len(indexes), *self.shape), dtype=np.float32)
        batch_y = np.zeros((len(indexes), *self.Y_shape), dtype=self.Y_dtype)

        for i, j in zip(range(len(indexes)), indexes):
            img, mask =  self.load_sample(j)

            # Apply random crops if it is selected
            if self.random_crops_in_DA:
                # Capture probability map
                if self.prob_map is not None:
                    if isinstance(self.prob_map, list):
                        img_prob = np.load(self.prob_map[j])
                    else:
                        img_prob = self.prob_map[j]
                else:
                    img_prob = None

                batch_x[i], batch_y[i] = self.random_crop_func(img, mask, self.shape[:self.ndim], self.val, vol_prob=img_prob)
            else:
                batch_x[i], batch_y[i] = img, mask

            # Apply transformations
            if self.da:
                e_img, e_mask = None, None
                if self.cutmix:
                    extra_img = np.random.randint(0, self.len-1) if self.len > 2 else 0
                    e_img, e_mask =  self.load_sample(extra_img)

                batch_x[i], batch_y[i] = self.apply_transform(batch_x[i], batch_y[i], e_im=e_img, e_mask=e_mask)

        # Prepare mask when denoising with Noise2Void
        if self.n2v:
            batch_y = batch_y.astype(np.float32)
            self.prepare_n2v(batch_x, batch_y)

        # One-hot enconde
        if self.n_classes > 1 and (self.n_classes != self.Y_channels):
            batch_y_ = np.zeros((len(indexes), ) + self.shape[:self.ndim] + (self.n_classes,))
            for k in range(len(indexes)):
                batch_y_[i] = np.asarray(img_to_onehot_encoding(batch_y[k][0]))

            batch_y = batch_y_

        self.total_batches_seen += 1
        if self.out_number == 1:
            return batch_x, batch_y
        else:
            return ([batch_x], [batch_y]*self.out_number)
 

    def apply_transform(self, image, mask, e_im=None, e_mask=None):
        """Transform the input image and its mask at the same time with one of the selected choices based on a
           probability.

           Parameters
           ----------
           image : 3D/4D Numpy array
               Image to transform. E.g. ``(y, x, channels)`` in 2D or ``(z, y, x, channels)`` in 3D.

           mask : 3D/4D Numpy array
               Mask to transform. E.g. ``(y, x, channels)`` in 2D or ``(z, y, x, channels)`` in 3D.

           e_img : 3D/4D Numpy array
               Extra image to help transforming ``image``. E.g. ``(y, x, channels)`` in 2D or 
               ``(z, y, x, channels)`` in 3D.

           e_mask : 3D/4D Numpy array
               Extra mask to help transforming ``mask``. E.g. ``(y, x, channels)`` in 2D or 
               ``(z, y, x, channels)`` in 3D.

           Returns
           -------
           image : 3D/4D Numpy array
               Transformed image. E.g. ``(y, x, channels)`` in 2D or ``(z, y, x, channels)`` in 3D.

           mask : 3D/4D Numpy array
               Transformed image mask. E.g. ``(y, x, channels)`` in 2D or ``(z, y, x, channels)`` in 3D.
        """
        # Split heatmaps from masks
        if self.first_no_bin_channel != -1:
            if self.first_no_bin_channel != 0:
                heat = mask[...,self.first_no_bin_channel:]
                mask = mask[...,:self.first_no_bin_channel]
            else:
                heat = mask
                mask = np.zeros(mask.shape) # Fake mask
            o_heat_shape = heat.shape
            o_mask_shape = mask.shape
            heat = heat.reshape(heat.shape[:(self.ndim-1)]+(heat.shape[2]*heat.shape[3],))
            heat = HeatmapsOnImage(heat, shape=heat.shape, min_value=0.0, max_value=np.max(heat)+sys.float_info.epsilon)
        else:
            heat = None

        # Save shape
        o_img_shape = image.shape
        o_mask_shape = mask.shape

        # Convert to grayscale
        if self.grayscale and random.uniform(0, 1) < self.da_prob:
            image = grayscale(image)

        # Apply channel shuffle
        if self.channel_shuffle and random.uniform(0, 1) < self.da_prob:
            image = shuffle_channels(image)

        # Reshape 3D volumes to 2D image type with multiple channels to pass through imgaug lib
        if self.ndim == 3:
            image = image.reshape(image.shape[:2]+(image.shape[2]*image.shape[3],))
            mask = mask.reshape(mask.shape[:2]+(mask.shape[2]*mask.shape[3],))
            if e_im is not None: e_im = e_im.reshape(e_im.shape[:2]+(e_im.shape[2]*e_im.shape[3],))
            if e_mask is not None: e_mask = e_mask.reshape(e_mask.shape[:2]+(e_mask.shape[2]*e_mask.shape[3],))
            #if e_heat is not None: e_heat = e_heat.reshape(e_heat.shape[:2]+(e_heat.shape[2]*e_heat.shape[3],))

        # Apply cutout
        if self.cutout and random.uniform(0, 1) < self.da_prob:
            image, mask = cutout(image, mask, self.X_channels, self.z_size, self.cout_nb_iterations, self.cout_size,
                                 self.cout_cval, self.res_relation, self.cout_apply_to_mask)

        # Apply cblur
        if self.cutblur and random.uniform(0, 1) < self.da_prob:
            image = cutblur(image, self.cblur_size, self.cblur_down_range, self.cblur_inside)

        # Apply cutmix
        if self.cutmix and random.uniform(0, 1) < self.da_prob:
            image, mask = cutmix(image, e_im, mask, e_mask, self.cmix_size)

        # Apply cutnoise
        if self.cutnoise and random.uniform(0, 1) < self.da_prob:
            image = cutnoise(image, self.cnoise_scale, self.cnoise_nb_iterations, self.cnoise_size)

        # Apply misalignment
        if self.misalignment and random.uniform(0, 1) < self.da_prob:
            rel = str(o_img_shape[-1])+"_"+str(o_mask_shape[-1])
            image, mask = misalignment(image, mask, self.ms_displacement, self.ms_rotate_ratio, c_relation=rel)

        # Apply brightness
        if self.brightness and random.uniform(0, 1) < self.da_prob:
            image = brightness(image, brightness_factor=self.brightness_factor, mode=self.brightness_mode)

        # Apply contrast
        if self.contrast and random.uniform(0, 1) < self.da_prob:
            image = contrast(image, contrast_factor=self.contrast_factor, mode=self.contrast_mode)

        # Apply brightness (EM)
        if self.brightness_em and random.uniform(0, 1) < self.da_prob:
            image = brightness_em(image, brightness_factor=self.brightness_em_factor, mode=self.brightness_em_mode)

        # Apply contrast (EM)
        if self.contrast_em and random.uniform(0, 1) < self.da_prob:
            image = contrast_em(image, contrast_factor=self.contrast_em_factor, mode=self.contrast_em_mode)

        # Apply missing parts
        if self.missing_parts and random.uniform(0, 1) < self.da_prob:
            image = missing_parts(image, self.missp_iterations)

        # Apply GridMask
        if self.gridmask and random.uniform(0, 1) < self.da_prob:
            image = GridMask(image, self.X_channels, self.z_size, self.grid_ratio, self.grid_d_size, self.grid_rotate,
                             self.grid_invert)

        # Apply transformations to the volume and its mask
        image, mask, heat_out = self.apply_imgaug(image, mask, heat)

        # Recover the original shape
        image = image.reshape(o_img_shape)
        mask = mask.reshape(o_mask_shape)

        # Merge heatmaps and masks again
        if self.first_no_bin_channel != -1:
            heat = heat_out.get_arr()
            heat = heat.reshape(o_heat_shape)
            if self.first_no_bin_channel != 0:
                mask = np.concatenate((mask,heat),axis=-1)
            else:
                mask = heat

        return image, mask

    def get_transformed_samples(self, num_examples, random_images=True, save_to_dir=True, out_dir='aug', train=False,
                                draw_grid=True):
        """Apply selected transformations to a defined number of images from the dataset.

           Parameters
           ----------
           num_examples : int
               Number of examples to generate.

           random_images : bool, optional
               Randomly select images from the dataset. If ``False`` the examples will be generated from the start of
               the dataset.

           save_to_dir : bool, optional
               Save the images generated. The purpose of this variable is to check the images generated by data
               augmentation.

           out_dir : str, optional
               Name of the folder where the examples will be stored.

           train : bool, optional
               To avoid drawing a grid on the generated images. This should be set when the samples will be used for
               training.
           draw_grid : bool, optional
               Draw a grid in the generated samples. Useful to see some types of deformations.

           Returns
           -------
           sample_x : List of 3D/4D Numpy array
               Transformed images. E.g. list of ``(y, x, channels)`` if 2D or 
               ``(z, y, x, channels)`` of 3D.

           sample_y : List of 3D/4D Numpy array
               Transformed image mask. E.g. list of ``(y, x, channels)`` if 2D or 
               ``(z, y, x, channels)`` of 3D.

           Examples
           --------
           ::

               # EXAMPLE 1
               # Generate 10 samples following with the example 1 of the class definition
               X_train = np.ones((1776, 256, 256, 1))
               Y_train = np.ones((1776, 256, 256, 1))

               data_gen_args = dict(X=X_train, Y=Y_train, batch_size=6, shape=(256, 256, 1), shuffle_each_epoch=True,
                                    rotation_range=True, vflip=True, hflip=True)

               train_generator = BaseDataGenerator(**data_gen_args)

               train_generator.get_transformed_samples(10, save_to_dir=True, train=False, out_dir='da_dir')

               # EXAMPLE 2
               # If random crop in DA-time is choosen, as the example 2 of the class definition, the call should be the
               # same but two more images will be stored: img and mask representing the random crop extracted. There a
               # red point is painted representing the pixel choosen to be the center of the random crop and a blue
               # square which delimits crop boundaries

               prob_map = calculate_2D_volume_prob_map(Y_train, 0.94, 0.06, save_file='prob_map.npy')

               data_gen_args = dict(X=X_train, Y=Y_train, batch_size=6, shape=(256, 256, 1), shuffle_each_epoch=True,
                                    rotation_range=True, vflip=True, hflip=True, random_crops_in_DA=True, prob_map=True,
                                    prob_map=prob_map)
               train_generator = BaseDataGenerator(**data_gen_args)

               train_generator.get_transformed_samples(10, save_to_dir=True, train=False, out_dir='da_dir')


           Example 2 will store two additional images as the following:

           +-------------------------------------------+-------------------------------------------+
           | .. figure:: ../../img/rd_crop_2d.png      | .. figure:: ../../img/rd_crop_mask_2d.png |
           |   :width: 80%                             |   :width: 80%                             |
           |   :align: center                          |   :align: center                          |
           |                                           |                                           |
           |   Original crop                           |   Original crop mask                      |
           +-------------------------------------------+-------------------------------------------+

           Together with these images another pair of images will be stored: the crop made and a transformed version of
           it, which is really the generator output.

           For instance, setting ``elastic=True`` the above extracted crop should be transformed as follows:

           +-------------------------------------------------+-------------------------------------------------+
           | .. figure:: ../../img/original_crop_2d.png      | .. figure:: ../../img/original_crop_mask_2d.png |
           |   :width: 80%                                   |   :width: 80%                                   |
           |   :align: center                                |   :align: center                                |
           |                                                 |                                                 |
           |   Original crop                                 |   Original crop mask                            |
           +-------------------------------------------------+-------------------------------------------------+
           | .. figure:: ../../img/elastic_crop_2d.png       | .. figure:: ../../img/elastic_crop_mask_2d.png  |
           |   :width: 80%                                   |   :width: 80%                                   |
           |   :align: center                                |   :align: center                                |
           |                                                 |                                                 |
           |   Elastic transformation applied                |   Elastic transformation applied                |
           +-------------------------------------------------+-------------------------------------------------+

           The grid is only painted if ``train=False`` which should be used just to display transformations made.
           Selecting random rotations between 0 and 180 degrees should generate the following:

           +--------------------------------------------------------+--------------------------------------------------------+
           | .. figure:: ../../img/original_rd_rot_crop_2d.png      | .. figure:: ../../img/original_rd_rot_crop_mask_2d.png |
           |   :width: 80%                                          |   :width: 80%                                          |
           |   :align: center                                       |   :align: center                                       |
           |                                                        |                                                        |
           |   Original crop                                        |   Original crop mask                                   |
           +--------------------------------------------------------+--------------------------------------------------------+
           | .. figure:: ../../img/rd_rot_crop_2d.png               | .. figure:: ../../img/rd_rot_crop_mask_2d.png          |
           |   :width: 80%                                          |   :width: 80%                                          |
           |   :align: center                                       |   :align: center                                       |
           |                                                        |                                                        |
           |   Random rotation [0, 180] applied                     |   Random rotation [0, 180] applied                     |
           +--------------------------------------------------------+--------------------------------------------------------+
        """
        if random_images == False and num_examples > self.len:
            num_examples = self.len
            print("WARNING: More samples requested than the ones available. 'num_examples' fixed to {}".format(num_examples))

        sample_x = []
        sample_y = []

        point_dict = None
        orig_images = None
        # Generate the examples
        print("0) Creating samples of data augmentation . . .")
        for i in tqdm(range(num_examples)):
            if random_images:
                pos = random.randint(0,self.len-1) if self.len > 2 else 0
            else:
                pos = i

            img, mask = self.load_sample(pos)

            # Apply random crops if it is selected
            if self.random_crops_in_DA:
                # Capture probability map
                if self.prob_map is not None:
                    if isinstance(self.prob_map, list):
                        img_prob = np.load(self.prob_map[pos])
                    else:
                        img_prob = self.prob_map[pos]
                else:
                    img_prob = None

                if self.ndim == 2:
                    img, mask, oy, ox,\
                    s_y, s_x = random_crop(img, mask, self.shape[:2], self.val, img_prob=img_prob, draw_prob_map_points=True)
                else:
                    img, mask, oz, oy, ox,\
                    s_z, s_y, s_x = random_3D_crop(img, mask, self.shape[:3], self.val, vol_prob=img_prob,
                                                   draw_prob_map_points=True)
                if save_to_dir:
                    point_dict = {}
                    point_dict['oy'], point_dict['ox'], point_dict['s_y'], point_dict['s_x'] = oy, ox, s_y, s_x
                if self.ndim == 3:
                    point_dict['oz'], point_dict['s_z'] = oz, s_z

                sample_x.append(img)
                sample_y.append(mask)
            else:
                sample_x.append(img)
                sample_y.append(mask)

            if save_to_dir:
                orig_images = {}
                orig_images['o_x'] = np.copy(img)
                orig_images['o_y'] = np.copy(mask)
                orig_images['o_x2'] = np.copy(img)
                orig_images['o_y2'] = np.copy(mask)

            # Apply transformations
            if self.da:
                if not train and draw_grid:
                    self.draw_grid(sample_x[i])
                    self.draw_grid(sample_y[i])

                e_img, e_mask = None, None
                if self.cutmix:
                    extra_img = np.random.randint(0, self.len-1) if self.len > 2 else 0
                    e_img, e_mask = self.load_sample(extra_img)
                
                sample_x[i], sample_y[i] = self.apply_transform(
                    sample_x[i], sample_y[i], e_im=e_img, e_mask=e_mask)

            if self.n2v:
                mask = np.repeat(sample_y[i], self.Y_channels*2, axis=-1).astype(np.float32)
                self.prepare_n2v(np.expand_dims(img,0), np.expand_dims(mask,0))
                sample_y[i] = mask

            if save_to_dir:
                self.save_aug_samples(sample_x[i], sample_y[i], orig_images, i, pos, out_dir, draw_grid, point_dict)


    def draw_grid(self, im, grid_width=50):
        """Draw grid of the specified size on an image.

           Parameters
           ----------
           im : 3D Numpy array
               Image to be modified. E. g. ``(y, x, channels)``

           grid_width : int, optional
               Grid's width.
        """
        v = 1 if int(np.max(im)) == 0 else int(np.max(im))

        if self.ndim == 2:
            for i in range(0, im.shape[0], grid_width):
                im[i] = v
            for j in range(0, im.shape[1], grid_width):
                im[:, j] = v
        else:
            for k in range(0, im.shape[0]):
                for i in range(0, im.shape[2], grid_width):
                    if im.shape[-1] == 1:
                        im[k,:,i] = v
                    else:
                        im[k,:,i] = [v]*im.shape[-1]
                for j in range(0, im.shape[1], grid_width):
                    if im.shape[-1] == 1:
                        im[k,j] = v
                    else:
                        im[k,j] = [v]*im.shape[-1]
    
    def prepare_n2v(self, batch_x, batch_y):
        for c in range(self.Y_channels):
            for j in range(batch_x.shape[0]):       
                coords = self.get_stratified_coords(self.rand_float, box_size=self.box_size, shape=self.shape)                             
                indexing = (j,) + coords + (c,)
                indexing_mask = (j,) + coords + (c + self.Y_channels, )
                y_val = batch_x[indexing]
                x_val = self.value_manipulation(batch_x[j, ..., c], coords, self.ndim, self.n2v_structMask)
                
                batch_y[indexing] = y_val
                batch_y[indexing_mask] = 1
                batch_x[indexing] = x_val

                if self.n2v_structMask is not None:
                    self.apply_structN2Vmask_func(batch_x[j, ..., c], coords, self.n2v_structMask)
                   
    