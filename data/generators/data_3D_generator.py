import sys
import numpy as np
import tensorflow as tf
import random
import os
import imageio
import imgaug as ia
from tqdm import tqdm
from skimage.io import imread
from scipy.ndimage import rotate
from scipy.ndimage.interpolation import shift
from PIL import Image
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from imgaug.augmentables.heatmaps import HeatmapsOnImage
from imgaug import parameters as iap
from skimage.io import imsave
from utils.util import img_to_onehot_encoding
from data.generators.augmentors import (cutout, cutblur, cutmix, cutnoise, misalignment, brightness_em, contrast_em,
                                        brightness, contrast, missing_parts, shuffle_channels, grayscale, GridMask)
from data.data_3D_manipulation import random_3D_crop


class VoxelDataGenerator(tf.keras.utils.Sequence):
    """Custom 3D ImageDataGenerator based on `imgaug <https://github.com/aleju/imgaug-doc>`_ and our own
       `augmentors.py <https://github.com/danifranco/EM_Image_Segmentation/blob/master/generators/augmentors.py>`_
       transformations.

       Based on `microDL <https://github.com/czbiohub/microDL>`_ and
       `Shervine's blog <https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly>`_.

       Parameters
       ----------
       X : Numpy 5D array
           Data. E.g. ``(num_of_images, x, y, z, channels)``.

       Y : Numpy 5D array
           Mask data. E.g. ``(num_of_images, x, y, z, channels)``.

       in_memory : bool, optional
           If ``True`` data used will be ``X`` and ``Y``. If ``False`` it will be loaded directly from disk using
           ``data_paths``.

       data_paths : List of str, optional
           If ``in_memory`` is ``True`` this list should contain the paths to load data and masks. ``data_paths[0]``
           should be data path and ``data_paths[1]`` masks path.

       random_crops_in_DA : bool, optional
           To extract random subvolumes from the given data. If not, the data must be 5D and is assumed that the
           subvolumes are prepared.

       shape : 4D tuple of ints, optional
           Shape of the subvolume to be extracted randomly from the data. E. g. ``(x, y, z, channels)``.

       resolution : 3D tuple of floats, optional
           Resolution of the given data ``(x,y,z)``. E.g. ``(8,8,30)``.

       prob_map : 5D Numpy array or str, optional
           If it is an array, it should represent the probability map used to make random crops when
           ``random_crops_in_DA`` is set. If ``str`` is given it should be the path to read these maps from.

       seed : int, optional
           Seed for random functions.

       shuffle_each_epoch : bool, optional
           To shuffle data after each epoch.

       batch_size : int, optional
           Size of the batches.

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

       zflip : bool, optional
           To activate flips in z dimension.

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
           Range to select the size of the areas in % of the corresponding
           image size. Values between ``0`` and ``1``. E. g. ``(0.2, 0.4)``.

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

       channel_shuffle : bool, optional
           Whether to shuflle the channels of the images.

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

       n_classes : int, optional
           Number of classes. If ``> 1`` one-hot encoding will be done on the ground truth.

       out_number : int, optional
           Number of output returned by the network. Used to produce same number of ground truth data on each batch.

       val : bool, optional
           Advice the generator that the volumes will be used to validate the model to not make random crops (as the
           validation data must be the same on each epoch). Valid when ``random_crops_in_DA``
           is set.

       extra_data_factor : int, optional
           Factor to multiply the batches yielded in a epoch. It acts as if ``X`` and ``Y``` where concatenated
           ``extra_data_factor`` times.
    """

    def __init__(self, X, Y, in_memory=True, data_paths=None, random_crops_in_DA=False, shape=None, resolution=(1,1,1),
                 prob_map=None, seed=0, shuffle_each_epoch=False, batch_size=32, da=True, da_prob=0.5, rotation90=False,
                 rand_rot=False, rnd_rot_range=(-180,180), shear=False, shear_range=(-20,20), zoom=False,
                 zoom_range=(0.8,1.2), shift=False, shift_range=(0.1,0.2), affine_mode='constant', vflip=False,
                 hflip=False, zflip=False, elastic=False, e_alpha=(240,250), e_sigma=25, e_mode='constant', g_blur=False,
                 g_sigma=(1.0,2.0), median_blur=False, mb_kernel=(3,7), motion_blur=False, motb_k_range=(3,8),
                 gamma_contrast=False, gc_gamma=(1.25,1.75), brightness=False, brightness_factor=(1,3),
                 brightness_mode="3D", contrast=False, contrast_factor=(1,3), contrast_mode="3D", brightness_em=False,
                 brightness_em_factor=(1,3), brightness_em_mode="3D", contrast_em=False, contrast_em_factor=(1,3),
                 contrast_em_mode="3D", dropout=False, drop_range=(0,0.2), cutout=False, cout_nb_iterations=(1,3),
                 cout_size=(0.2,0.4), cout_cval=0, cout_apply_to_mask=False, cutblur=False, cblur_size=(0.2,0.4),
                 cblur_down_range=(2,8), cblur_inside=True, cutmix=False, cmix_size=(0.2,0.4), cutnoise=False,
                 cnoise_scale=(0.1,0.2), cnoise_nb_iterations=(1,3), cnoise_size=(0.2,0.4), misalignment=False,
                 ms_displacement=16, ms_rotate_ratio=0.0, missing_parts=False, missp_iterations=(30, 40), grayscale=False,
                 channel_shuffle=False, gridmask=False, grid_ratio=0.6, grid_d_range=(0.4,1), grid_rotate=1,
                 grid_invert=False, n_classes=1, out_number=1, val=False, extra_data_factor=1):

        if in_memory:
            if X.ndim != 5 or Y.ndim != 5:
                raise ValueError("X and Y must be a 5D Numpy array")
            if X.shape[:4] != Y.shape[:4]:
                raise ValueError("The shape of X and Y must be the same. {} != {}".format(X.shape[:4], Y.shape[:4]))

        if in_memory and (X is None or Y is None):
            raise ValueError("'X' and 'Y' need to be provided together with 'in_memory'")

        if not in_memory and len(data_paths) != 2:
            raise ValueError("'data_paths' must contain the following paths: 1) data path ; 2) data masks path")

        if len(resolution) == 1 and resolution[0] == -1:
            resolution = (1,1,1)

        if len(resolution) != 3:
            raise ValueError("The resolution needs to be a tuple with 3 values")

        if random_crops_in_DA:
            if shape is None:
                raise ValueError("'shape' must be provided when 'random_crops_in_DA is enabled")
            if in_memory:
                if shape[0] > X.shape[1] or shape[1] > X.shape[2] or shape[2] > X.shape[3]:
                    raise ValueError("Given 'shape' is bigger than the data provided")

        if not in_memory and not random_crops_in_DA:
            print("WARNING: you are going to load samples from disk (as 'in_memory=False') and "
                  "'random_crops_in_DA=False' so all samples are expected to have the same shape. If it is not the "
                  "case set batch_size to 1 or the generator will throw an error")

        if rotation90 and rand_rot:
            print("Warning: you selected double rotation type. Maybe you should set only 'rand_rot'?")

        if not in_memory:
            # Save paths where the data is stored
            self.paths = data_paths
            self.data_paths = sorted(next(os.walk(data_paths[0]))[2])
            self.data_mask_path = sorted(next(os.walk(data_paths[1]))[2])
            self.len = len(self.data_paths)

            # Check if a division is required
            if self.data_paths[0].endswith('.npy'):
                img = np.load(os.path.join(data_paths[0], self.data_paths[0]))
            else:
                img = imread(os.path.join(data_paths[0], self.data_paths[0]))
            if img.ndim == 3: img = np.expand_dims(img, -1)
            img = img.transpose((1,2,0,3))
            self.X_channels = img.shape[-1]
            self.z_size = img.shape[-2]
            # Ensure uint8
            if img.dtype == np.uint16:
                img = normalize(img, 0, 65535)
            self.div_X_on_load = True if np.max(img) > 10 else False
            self.shape = shape if random_crops_in_DA else img.shape
            # Loop over a few masks to ensure foreground class is present
            self.first_no_bin_channel = -1
            self.div_Y_no_bin_channels_max = 0
            self.div_Y_no_bin_channels_min = 9999999
            print("Calculating generator values . . .")
            for i in range(min(10,len(self.data_mask_path))):
                if self.data_mask_path[i].endswith('.npy'):
                    img = np.load(os.path.join(self.paths[1], self.data_mask_path[i]))
                else:
                    img = imread(os.path.join(data_paths[1], self.data_mask_path[i]))
                if img.ndim == 3: img = np.expand_dims(img, -1)
                # Store wheter all channels of the gt are binary or not
                # (e.g. distance transform channel)
                if img.dtype is np.dtype(np.float32) or \
                   img.dtype is np.dtype(np.float64):
                    for j in range(img.shape[-1]):
                        if len(np.unique(img[...,j])) > 2:
                            self.first_no_bin_channel = j
                            break
                if self.first_no_bin_channel != -1:
                    if self.first_no_bin_channel != 0:
                        self.div_Y_on_load_bin_channels = True if np.max(img[...,:self.first_no_bin_channel]) > 10 else False
                        max_no_bin_channel = np.max(img[...,self.first_no_bin_channel:])
                        self.div_Y_on_load_no_bin_channels = True if max_no_bin_channel > 200 else False
                        if max_no_bin_channel > self.div_Y_no_bin_channels_max:
                            self.div_Y_no_bin_channels_max = max_no_bin_channel
                        min_no_bin_channel = np.min(img[...,self.first_no_bin_channel:])
                        if min_no_bin_channel < self.div_Y_no_bin_channels_min:
                            self.div_Y_no_bin_channels_min = min_no_bin_channel
                    else:
                        self.div_Y_on_load_bin_channels = True if np.max(img) > 10 else False
                        max_no_bin_channel = np.max(img)
                        self.div_Y_on_load_no_bin_channels = True if max_no_bin_channel > 200 else False
                        if max_no_bin_channel > self.div_Y_no_bin_channels_max:
                            self.div_Y_no_bin_channels_max = max_no_bin_channel
                        min_no_bin_channel = np.min(img)
                        if min_no_bin_channel < self.div_Y_no_bin_channels_min:
                            self.div_Y_no_bin_channels_min = min_no_bin_channel
                else:
                    self.div_Y_on_load_bin_channels = True if np.max(img) > 10 else False

                self.channels = img.shape[-1]
                self.Y_dtype = img.dtype
                del img
        else:
            self.X = X.astype(np.uint8)
            self.Y = Y
            self.Y_dtype = Y.dtype
            # Store wheter all channels of the gt are binary or not (e.g. distance transform channel)
            self.first_no_bin_channel = -1
            self.div_Y_no_bin_channels_max = 0
            self.div_Y_no_bin_channels_min = 9999999
            if Y.dtype is np.dtype(np.float32) or Y.dtype is np.dtype(np.float64):
                for i in range(Y.shape[-1]):
                    if len(np.unique(Y[...,i])) > 2:
                        self.first_no_bin_channel = i
                        break
            self.div_X_on_load = True if np.max(X) > 10 else False
            if self.first_no_bin_channel != -1:
                if self.first_no_bin_channel != 0:
                    self.div_Y_on_load_bin_channels = True if np.max(Y[...,:self.first_no_bin_channel]) > 10 else False
                    max_no_bin_channel = np.max(Y[...,self.first_no_bin_channel:])
                    self.div_Y_on_load_no_bin_channels = True if max_no_bin_channel > 200 else False
                    self.div_Y_no_bin_channels_max = max_no_bin_channel
                    self.div_Y_no_bin_channels_min = np.min(Y[...,self.first_no_bin_channel:])
                else:
                    self.div_Y_on_load_bin_channels = False
                    max_no_bin_channel = np.max(Y)
                    self.div_Y_on_load_no_bin_channels = True if max_no_bin_channel > 200 else False
                    self.div_Y_no_bin_channels_max = max_no_bin_channel
                    self.div_Y_no_bin_channels_min = np.min(Y)
            else:
                self.div_Y_on_load_bin_channels = True if np.max(Y) > 10 else False

            self.channels = Y.shape[-1]
            self.X_channels = X.shape[-1]
            self.z_size = X.shape[-2]
            self.len = len(self.X)
            self.shape = shape if random_crops_in_DA else X.shape[1:]

        self.prob_map = None
        if random_crops_in_DA and prob_map is not None:
            if isinstance(prob_map, str):
                f = sorted(next(os.walk(prob_map))[2])
                self.prob_map = []
                for i in range(len(f)):
                    self.prob_map.append(os.path.join(prob_map, f[i]))
            else:
                self.prob_map = prob_map

        self.resolution = resolution
        self.res_relation = (1.0,resolution[0]/resolution[1],resolution[0]/resolution[2])
        self.n_classes = n_classes
        self.out_number = out_number
        self.random_crops_in_DA = random_crops_in_DA
        self.in_memory = in_memory
        self.seed = seed
        self.shuffle_each_epoch = shuffle_each_epoch
        self.da = da
        self.da_prob = da_prob
        self.vflip = vflip
        self.hflip = hflip
        self.zflip = zflip
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
        self.missing_parts = missing_parts
        self.missp_iterations = missp_iterations
        self.grayscale = grayscale
        self.channel_shuffle = channel_shuffle
        self.gridmask = gridmask
        self.grid_ratio = grid_ratio
        self.grid_d_range = grid_d_range
        self.grid_rotate = grid_rotate
        self.grid_invert = grid_invert
        self.grid_d_size = (self.shape[0]*grid_d_range[0], self.shape[1]*grid_d_range[1],\
                            self.shape[2]*grid_d_range[0], self.shape[2]*grid_d_range[1])
        self.val = val
        self.batch_size = batch_size
        self.o_indexes = np.arange(self.len)
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
            self.da_options.append(iaa.Flipud(0.5))
            self.trans_made += '_vflip'
        if hflip:
            self.da_options.append(iaa.Fliplr(0.5))
            self.trans_made += '_hflip'
        if zflip: self.trans_made += '_zflip'
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
        self.brightness = brightness
        if brightness:
            self.brightness_factor = brightness_factor
            self.brightness_mode = brightness_mode
            self.trans_made += '_brightness'+str(brightness_factor)
        self.contrast = contrast
        if contrast:
            self.contrast_factor = contrast_factor
            self.contrast_mode = contrast_mode
            self.trans_made += '_contrast'+str(contrast_factor)
        self.brightness_em = brightness_em
        if brightness_em:
            self.brightness_em_factor = brightness_em_factor
            self.brightness_em_mode = brightness_em_mode
            self.trans_made += '_brightness_em'+str(brightness_em_factor)
        self.contrast_em = contrast_em
        if contrast_em:
            self.contrast_em_factor = contrast_em_factor
            self.contrast_em_mode = contrast_em_mode
            self.trans_made += '_contrast_em'+str(contrast_em_factor)
        if dropout:
            self.da_options.append(iaa.Sometimes(da_prob, iaa.Dropout(p=drop_range)))
            self.trans_made += '_drop'+str(drop_range)
        if grayscale: self.trans_made += '_gray'
        if channel_shuffle: self.trans_made += '_chshuffle'
        if gridmask: self.trans_made += '_gridmask'+str(self.grid_ratio)+str(self.grid_d_range)+str(self.grid_rotate)+str(self.grid_invert)
        if cutout: self.trans_made += '_cout'+str(cout_nb_iterations)+'+'+str(cout_size)+'+'+str(cout_cval)+'+'+str(cout_apply_to_mask)
        if cutblur: self.trans_made += '_cblur'+str(cblur_size)+'+'+str(cblur_down_range)+'+'+str(cblur_inside)
        if cutmix: self.trans_made += '_cmix'+str(cmix_size)
        if cutnoise: self.trans_made += '_cnoi'+str(cnoise_scale)+'+'+str(cnoise_nb_iterations)+'+'+str(cnoise_size)
        if misalignment: self.trans_made += '_msalg'+str(ms_displacement)+'+'+str(ms_rotate_ratio)
        if missing_parts: self.trans_made += '_missp'+'+'+str(missp_iterations)

        self.trans_made = self.trans_made.replace(" ", "")
        if not self.da: self.trans_made = ''
        self.seq = iaa.Sequential(self.da_options)
        ia.seed(seed)
        self.on_epoch_end()


    def __len__(self):
        """Defines the length of the generator"""
        return int(np.ceil(self.len*self.extra_data_factor/self.batch_size))


    def __draw_grid(self, im, grid_width=50):
        """Draw grid of the specified size on an image.

           Parameters
           ----------
           im : 4D Numpy array
               Image to be modified. E. g. ``(x, y, z, channels)``

           grid_width : int, optional
               Grid's width.
        """
        v = 1 if int(np.max(im)) == 0 else int(np.max(im))

        for k in range(0, im.shape[2]):
            for i in range(0, im.shape[0], grid_width):
                if im.shape[-1] == 1:
                    im[i,:,k] = v
                else:
                    im[i,:,k] = [v]*im.shape[-1]
            for j in range(0, im.shape[1], grid_width):
                if im.shape[-1] == 1:
                    im[:,j,k] = v
                else:
                    im[:,j,k] = [v]*im.shape[-1]


    def __getitem__(self, index):
        """Generation of one batch of data.

           Parameters
           ----------
           index : int
               Batch index counter.

           Returns
           -------
           batch_x : 5D Numpy array
               Corresponding X elements of the batch. E.g. ``(batch_size_value, x, y, z, channels)``.

           batch_y : 5D Numpy array
               Corresponding Y elements of the batch. E.g. ``(batch_size_value, x, y, z, channels)``.
        """

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_x = np.zeros((len(indexes), *self.shape), dtype=np.uint8)
        batch_y = np.zeros((len(indexes), *self.shape[:3])+(self.channels,), dtype=self.Y_dtype)

        for i, j in zip(range(len(indexes)), indexes):

            # Choose the data source
            if self.in_memory:
                img = self.X[j]
                mask = self.Y[j]
            else:
                if self.data_paths[j].endswith('.npy'):
                    img = np.load(os.path.join(self.paths[0], self.data_paths[j]))
                    mask = np.load(os.path.join(self.paths[1], self.data_mask_path[j]))
                else:
                    img = imread(os.path.join(self.paths[0], self.data_paths[j]))
                    mask = imread(os.path.join(self.paths[1], self.data_mask_path[j]))
                if img.ndim == 3: img = np.expand_dims(img, -1)
                if mask.ndim == 3: mask = np.expand_dims(mask, -1)
                img = img.transpose((1,2,0,3))
                mask = mask.transpose((1,2,0,3))
                # Ensure uint8
                if img.dtype == np.uint16:
                    img = normalize(img, 0, 65535)

            # Apply ramdom crops if it is selected
            if self.random_crops_in_DA:
                # Capture probability map
                if self.prob_map is not None:
                    if isinstance(self.prob_map, list):
                        img_prob = np.load(self.prob_map[j])
                    else:
                        img_prob = self.prob_map[j]
                else:
                    img_prob = None

                batch_x[i], batch_y[i] = random_3D_crop(img, mask, self.shape[:3], self.val, vol_prob=img_prob)
            else:
                batch_x[i], batch_y[i] = img, mask

            # Apply transformations
            if self.da:
                extra_img = np.random.randint(0, self.len-1) if self.len > 2 else 0
                if self.in_memory:
                    e_img = self.X[extra_img]
                    e_mask = self.Y[extra_img]
                else:
                    if self.data_paths[extra_img].endswith('.npy'):
                        e_img = np.load(os.path.join(self.paths[0], self.data_paths[extra_img]))
                        e_mask = np.load(os.path.join(self.paths[1], self.data_mask_path[extra_img]))
                    else:
                        e_img = imread(os.path.join(self.paths[0], self.data_paths[extra_img]))
                        e_mask = imread(os.path.join(self.paths[1], self.data_mask_path[extra_img]))
                    if e_img.ndim == 3: e_img = np.expand_dims(e_img, -1)
                    if e_mask.ndim == 3: e_mask = np.expand_dims(e_mask, -1)
                    e_img = e_img.transpose((1,2,0,3))
                    e_mask = e_mask.transpose((1,2,0,3))
                    # Ensure uint8
                    if e_img.dtype == np.uint16:
                        e_img = normalize(e_img, 0, 65535)

                batch_x[i], batch_y[i] = self.apply_transform(batch_x[i], batch_y[i], e_im=e_img, e_mask=e_mask)

        # Divide the values
        if self.div_X_on_load: batch_x = batch_x/255
        if self.first_no_bin_channel != -1:
            if self.div_Y_on_load_bin_channels:
                batch_y[...,:self.first_no_bin_channel] = batch_y[...,:self.first_no_bin_channel]/255
            if self.div_Y_on_load_no_bin_channels:
                if self.first_no_bin_channel != 0:
                    batch_y[...,self.first_no_bin_channel:] = batch_y[...,self.first_no_bin_channel:]/255
                else:
                    batch_y = batch_y/255
                    #(batch_y[...,self.first_no_bin_channel:]-self.div_Y_no_bin_channels_min)/(self.div_Y_no_bin_channels_max-self.div_Y_no_bin_channels_min)
        else:
            if self.div_Y_on_load_bin_channels: batch_y = batch_y/255

        if self.n_classes > 1 and (self.n_classes != self.channels):
            batch_y_ = np.zeros((len(indexes), ) + self.shape[:3] + (self.n_classes,))
            for k in range(len(indexes)):
                batch_y_[i] = np.asarray(img_to_onehot_encoding(batch_y[k][0]))

            batch_y = batch_y_

        self.total_batches_seen += 1

        if self.out_number == 1:
            return batch_x, batch_y
        else:
            return ([batch_x], [batch_y]*self.out_number)


    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        ia.seed(self.seed + self.total_batches_seen)
        self.indexes = self.o_indexes
        if self.shuffle_each_epoch:
            random.Random(self.seed + self.total_batches_seen).shuffle(self.indexes)


    def apply_transform(self, image, mask, e_im=None, e_mask=None):
        """Transform the input image and its mask at the same time with one of the selected choices based on a
           probability.

           Parameters
           ----------
           image : 4D Numpy array
               Image to transform. E.g. ``(x, y, z, channels)``.

           mask : 4D Numpy array
               Mask to transform. E.g. ``(x, y, z, channels)``.

           e_img : 4D Numpy array
               Extra image to help transforming ``image``. E.g. ``(x, y, z, channels)``.

           e_mask : 4D Numpy array
               Extra mask to help transforming ``mask``. E.g. ``(x, y, z, channels)``.

           Returns
           -------
           image : 4D Numpy array
               Transformed image. E.g. ``(x, y, z, channels)``.

           mask : 4D Numpy array
               Transformed image mask. E.g. ``(x, y, z, channels)``.
        """

        # Apply flips in z as imgaug can not do it
        if self.zflip and random.uniform(0, 1) < self.da_prob:
            l_image = []
            l_mask = []
            for i in range(image.shape[-1]):
                l_image.append(np.expand_dims(np.flip(image[...,i], 2), -1))
            for i in range(mask.shape[-1]):
                l_mask.append(np.expand_dims(np.flip(mask[...,i], 2), -1))
            image = np.concatenate(l_image, axis=-1)
            mask = np.concatenate(l_mask, axis=-1)

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
            heat = heat.reshape(heat.shape[:2]+(heat.shape[2]*heat.shape[3],))
            heat = HeatmapsOnImage(heat, shape=heat.shape, min_value=0.0, max_value=np.max(heat)+sys.float_info.epsilon)
        else:
            heat = None

        # Change dtype to supported one by imgaug
        image = image.astype(np.uint8)
        mask = mask.astype(np.uint8)

        # Save shapes
        o_img_shape = image.shape
        o_mask_shape = mask.shape

        # Convert to grayscale
        if self.grayscale and random.uniform(0, 1) < self.da_prob:
            image = grayscale(image)

        # Apply channel shuffle
        if self.channel_shuffle and random.uniform(0, 1) < self.da_prob:
            image = shuffle_channels(image)

        # Reshape 3D volumes to 2D image type with multiple channels to pass
        # through imgaug lib
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
        segmap = SegmentationMapsOnImage(mask, shape=mask.shape)
        image, vol_mask, heat_out = self.seq(image=image, segmentation_maps=segmap, heatmaps=heat)
        mask = vol_mask.get_arr()

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


    def get_transformed_samples(self, num_examples, random_images=True, save_to_dir=True, out_dir='aug_3d', train=False,
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
           sample_x : List of 4D Numpy array
               Transformed images. E.g. list of ``(x, y, z, channels)``.

           sample_y : List of 4D Numpy array
               Transformed image mask. E.g. list of ``(x, y, z, channels)``.
        """

        if random_images == False and num_examples > self.len:
            num_examples = self.len
            print("WARNING: More samples requested than the ones available. 'num_examples' fixed to {}".format(num_examples))

        sample_x = []
        sample_y = []

        # Generate the examples
        print("0) Creating samples of data augmentation . . .")
        for i in tqdm(range(num_examples)):
            if random_images:
                pos = random.randint(0,self.len-1) if self.len > 2 else 0
            else:
                pos = i

            # Choose the data source
            if self.in_memory:
                img = np.copy(self.X[pos])
                mask = np.copy(self.Y[pos])
            else:
                if self.data_paths[pos].endswith('.npy'):
                    img = np.load(os.path.join(self.paths[0], self.data_paths[pos]))
                    mask = np.load(os.path.join(self.paths[1], self.data_mask_path[pos]))
                else:
                    img = imread(os.path.join(self.paths[0], self.data_paths[pos]))
                    mask = imread(os.path.join(self.paths[1], self.data_mask_path[pos]))
                if img.ndim == 3: img = np.expand_dims(img, -1)
                if mask.ndim == 3: mask = np.expand_dims(mask, -1)
                img = img.transpose((1,2,0,3))
                mask = mask.transpose((1,2,0,3))
                # Ensure uint8
                if img.dtype == np.uint16:
                    img = normalize(img, 0, 65535)


            # Apply ramdom crops if it is selected
            if self.random_crops_in_DA:
                # Capture probability map
                if self.prob_map is not None:
                    if isinstance(self.prob_map, list):
                        img_prob = np.load(self.prob_map[pos])
                    else:
                        img_prob = self.prob_map[pos]
                else:
                    img_prob = None
                vol, vol_mask, ox, oy, oz,\
                s_x, s_y, s_z = random_3D_crop(img, mask, self.shape[:3], self.val, vol_prob=img_prob,
                                               draw_prob_map_points=True)
                sample_x.append(vol)
                sample_y.append(vol_mask)
                del vol, vol_mask
            else:
                sample_x.append(img)
                sample_y.append(mask)

            if save_to_dir:
                o_x = np.copy(img)
                o_y = np.copy(mask)
                o_x2 = np.copy(img)
                o_y2 = np.copy(mask)

            # Apply transformations
            if self.da:
                if not train and draw_grid:
                    self.__draw_grid(sample_x[i])
                    self.__draw_grid(sample_y[i])

                extra_img = np.random.randint(0, self.len-1) if self.len > 2 else 0
                if self.in_memory:
                    e_img = self.X[extra_img]
                    e_mask = self.Y[extra_img]
                else:
                    if self.data_paths[extra_img].endswith('.npy'):
                        e_img = np.load(os.path.join(self.paths[0], self.data_paths[extra_img]))
                        e_mask = np.load(os.path.join(self.paths[1], self.data_mask_path[extra_img]))
                    else:
                        e_img = imread(os.path.join(self.paths[0], self.data_paths[extra_img]))
                        e_mask = imread(os.path.join(self.paths[1], self.data_mask_path[extra_img]))
                    if e_img.ndim == 3: e_img = np.expand_dims(e_img, -1)
                    if e_mask.ndim == 3: e_mask = np.expand_dims(e_mask, -1)
                    e_img = e_img.transpose((1,2,0,3))
                    e_mask = e_mask.transpose((1,2,0,3))
                    # Ensure uint8
                    if e_img.dtype == np.uint16:
                        e_img = normalize(e_img, 0, 65535)

                sample_x[i], sample_y[i] = self.apply_transform(
                    sample_x[i], sample_y[i], e_im=e_img, e_mask=e_mask)

            # Save transformed 3D volumes
            if save_to_dir:
                os.makedirs(out_dir, exist_ok=True)
                # Original image/mask
                f = os.path.join(out_dir, "orig_x_"+str(pos)+"_"+str(i)+"_"+self.trans_made+'.tiff')
                if draw_grid: self.__draw_grid(o_x)
                aux = np.expand_dims((np.transpose(o_x, (2,3,0,1))).astype(np.float32), -1)
                imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False)
                f = os.path.join(out_dir, "orig_y_"+str(pos)+"_"+str(i)+"_"+self.trans_made+'.tiff')
                if draw_grid: self.__draw_grid(o_y)
                aux = np.expand_dims((np.transpose(o_y, (2,3,0,1))).astype(np.float32), -1)
                imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False)
                # Transformed
                f = os.path.join(out_dir, "x_aug_"+str(pos)+"_"+str(i)+"_"+self.trans_made+'.tiff')
                aux = np.expand_dims((np.transpose(sample_x[i], (2,3,0,1))).astype(np.float32), -1)
                imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False)
                # Mask
                f = os.path.join(out_dir, "y_aug_"+str(pos)+"_"+str(i)+"_"+self.trans_made+'.tiff')
                aux = np.expand_dims((np.transpose(sample_y[i], (2,3,0,1))).astype(np.float32), -1)
                imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False)

                # Save the original images with a red point and a blue square that represents the point selected with
                # the probability map and the random volume extracted from the original data
                if self.random_crops_in_DA and self.prob_map is not None and i == 0:
                    os.makedirs(out_dir, exist_ok=True)

                    print("The selected point of the random crop was [{},{},{}]".format(ox,oy,oz))

                    aux = (o_x2).astype(np.uint8)
                    if aux.shape[-1] == 1: aux = np.repeat(aux, 3, axis=3)
                    auxm = (o_y2).astype(np.uint8)
                    if auxm.shape[-1] == 1: auxm = np.repeat(auxm, 3, axis=3)

                    print(aux.shape)
                    for s in range(aux.shape[2]):
                        if s >= s_z and s < s_z+self.shape[2]:
                            im = Image.fromarray(aux[:,:,s,0])
                            im = im.convert('RGB')
                            px = im.load()
                            m = Image.fromarray(auxm[:,:,s,0])
                            m = m.convert('RGB')
                            py = m.load()
                            # Paint a blue square that represents the crop made
                            for col in range(s_x, s_x+self.shape[0]):
                                px[s_y, col] = (0, 0, 255)
                                px[s_y+self.shape[0]-1, col] = (0, 0, 255)
                                py[s_y, col] = (0, 0, 255)
                                py[s_y+self.shape[0]-1, col] = (0, 0, 255)
                            for row in range(s_y, s_y+self.shape[1]):
                                px[row, s_x] = (0, 0, 255)
                                px[row, s_x+self.shape[1]-1] = (0, 0, 255)
                                py[row, s_x] = (0, 0, 255)
                                py[row, s_x+self.shape[1]-1] = (0, 0, 255)

                            # Paint the selected point in red
                            if s == oz:
                                p_size=6
                                for row in range(oy-p_size,oy+p_size):
                                    for col in range(ox-p_size,ox+p_size):
                                        if col >= 0 and col < img.shape[0] and row >= 0 and row < img.shape[1]:
                                            px[row, col] = (255, 0, 0)
                                            py[row, col] = (255, 0, 0)

                            aux[:,:,s,:] = im
                            auxm[:,:,s,:] = m

                    aux = np.expand_dims((np.transpose(aux, (2,3,0,1))).astype(np.float32), -1)
                    f = os.path.join(out_dir, "extract_example_"+str(pos)+"_"+"_mark_x_"+self.trans_made+'.tiff')
                    imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False)
                    auxm = np.expand_dims((np.transpose(auxm, (2,3,0,1))).astype(np.float32), -1)
                    f = os.path.join(out_dir, "extract_example_"+str(pos)+"_"+"_mark_y_"+self.trans_made+'.tiff')
                    imsave(f, auxm, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False)
                    del o_x2, o_y2

        return sample_x, sample_y

