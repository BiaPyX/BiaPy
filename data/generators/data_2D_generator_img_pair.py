import tensorflow as tf
import numpy as np
import random
import os
from tqdm import tqdm
import imgaug as ia
from skimage.io import imsave, imread
from imgaug import augmenters as iaa

from utils.util import normalize
from data.data_2D_manipulation import random_crop
from data.generators.augmentors import (cutout, cutblur, cutmix, cutnoise, misalignment, brightness, contrast,
                                        brightness_em, contrast_em, missing_parts, grayscale, shuffle_channels, GridMask)


class PairImageDataGenerator(tf.keras.utils.Sequence):
    """Custom 2D ImageDataGenerator based on `imgaug <https://github.com/aleju/imgaug-doc>`_
       and our own `augmentors.py <https://github.com/danifranco/BiaPy/blob/master/generators/augmentors.py>`_
       transformations.

       Based on `microDL <https://github.com/czbiohub/microDL>`_ and
       `Shervine's blog <https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly>`_.

       Parameters
       ----------
       X : 4D Numpy array
           Data. E.g. ``(num_of_images, x, y, channels)``.

       Y : 4D Numpy array
           Mask data. E.g. ``(num_of_images, x, y, 1)``.

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
          If ``in_memory`` is ``True`` this list should contain the paths to load 1st image and 2nd image. 
          ``data_paths[0]`` should be 1st image path and ``data_paths[1]`` 2st image path.

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

       random_crop_scale : int, optional
           Scale factor the second image when ``random_crops_in_DA`` is ``True``.

       shape : 3D int tuple, optional
           Shape of the desired images when using 'random_crops_in_DA'.

       resolution : 2D tuple of floats, optional
           Resolution of the given data ``(x,y)``. E.g. ``(8,8)``.

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


       Examples
       --------
       ::

           # EXAMPLE 1
           # Define train and val generators to make random rotations between 0 and
           # 180 degrees. Notice that DA is disabled in val data

           X_train = np.ones((1776, 256, 256, 1))
           Y_train = np.ones((1776, 256, 256, 1))
           X_val = np.ones((204, 256, 256, 1))
           Y_val = np.ones((204, 256, 256, 1))

           data_gen_args = dict(X=X_train, Y=Y_train, batch_size=6, shuffle_each_epoch=True, rand_rot=True, vflip=True,
                                hflip=True)
           data_gen_val_args = dict(X=X_val, Y=Y_val, batch_size=6, shuffle_each_epoch=True, da=False, val=True)
           train_generator = ImageDataGenerator(**data_gen_args)
           val_generator = ImageDataGenerator(**data_gen_val_args)


           # EXAMPLE 2
           # Generate random crops on DA-time. To allow that notice that the
           # data in this case is bigger in width and height than example 1, to
           # allow a (256, 256) random crop extraction

           X_train = np.zeros((148, 768, 1024, 1))
           Y_train = np.zeros((148, 768, 1024, 1))
           X_val = np.zeros((17, 768, 1024, 1))
           Y_val = np.zeros((17, 768, 1024, 1))

           # Create a prbobability map for each image. Here we define foreground
           # probability much higher than the background simulating a class inbalance
           # With this, the probability of take the center pixel of the random crop
           # that corresponds to the foreground class will be so high
           prob_map = calculate_2D_volume_prob_map(
                Y_train, 0.94, 0.06, save_file='prob_map.npy')

           data_gen_args = dict(X=X_train, Y=Y_train, batch_size=6, shuffle_each_epoch=True, rand_rot=True, vflip=True,
                                hflip=True, random_crops_in_DA=True, shape=(256, 256, 1), prob_map=prob_map)

           data_gen_val_args = dict(X=X_val, Y=Y_val, batch_size=6, random_crops_in_DA=True, shape=(256, 256, 1),
                                    shuffle_each_epoch=True, da=False, val=True)
           train_generator = ImageDataGenerator(**data_gen_args)
           val_generator = ImageDataGenerator(**data_gen_val_args)
    """

    def __init__(self, X, Y, batch_size=32, seed=0, shuffle_each_epoch=False, in_memory=True, data_paths=None, da=True,
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
                 grid_rotate=1, grid_invert=False, random_crops_in_DA=False, random_crop_scale=1, shape=(256,256,1), 
                 resolution=(1,1), prob_map=None, val=False, n_classes=1, out_number=1, extra_data_factor=1):

        if in_memory:
            if type(X) != list:
                if X.ndim != 4 or Y.ndim != 4:
                    raise ValueError("X and Y must be a 4D Numpy array")
                if X.shape[:3] != Y.shape[:3]:
                    raise ValueError("The shape of X and Y must be the same. {} != {}".format(X.shape[:3], Y.shape[:3]))
            else:
                if X[0].ndim != 4 or Y[0].ndim != 4:
                    raise ValueError("X[0] and Y[0] must be a 4D Numpy array")
        if in_memory and (X is None or Y is None):
            raise ValueError("'X' and 'Y' need to be provided together with 'in_memory'")

        if not in_memory and len(data_paths) != 2:
            raise ValueError("'data_paths' must contain the following paths: 1) data path ; 2) data masks path")

        if random_crops_in_DA and (shape[0] != shape[1]):
            raise ValueError("When 'random_crops_in_DA' is selected the shape given must be square, e.g. (256, 256, 1)")

        if not in_memory and not random_crops_in_DA:
            print("WARNING: you are going to load samples from disk (as 'in_memory=False') and "
                  "'random_crops_in_DA=False' so all samples are expected to have the same shape. If it is not "
                  "the case set batch_size to 1 or the generator will throw an error")

        if rotation90 and rand_rot:
            print("Warning: you selected double rotation type. Maybe you should set only 'rand_rot'?")

        if len(resolution) == 1 and resolution[0] == -1:
            resolution = (1,1)

        if len(resolution) != 2:
            raise ValueError("The resolution needs to be a tuple with 2 values")

        self.batch_size = batch_size
        self.in_memory = in_memory
        if not in_memory:
            # Save paths where the data is stored
            self.paths = data_paths
            self.data_imgA_paths = sorted(next(os.walk(data_paths[0]))[2])
            self.data_imgB_paths = sorted(next(os.walk(data_paths[1]))[2])
            self.len = len(self.data_imgA_paths)

            # Check if a division is required
            if self.data_imgA_paths[0].endswith('.npy'):
                img = np.load(os.path.join(data_paths[0], self.data_imgA_paths[0]))
            else:
                img = imread(os.path.join(data_paths[0], self.data_imgA_paths[0]))
            if img.ndim == 2:
                img = np.expand_dims(img, -1)
            else:
                if img.shape[0] == 1 or img.shape[0] == 3: img = img.transpose((1,2,0))
            # Ensure uint8
            if img.dtype == np.uint16:
                if np.max(img) > 255:
                    img = normalize(img, 0, 65535)
                else:
                    img = img.astype(np.uint8)

            self.shape_imgA = shape if random_crops_in_DA else img.shape
            # Loop over a few images to ensure foreground class is present
            for i in range(min(10,len(self.data_imgB_paths))):
                if self.data_imgB_paths[i].endswith('.npy'):
                    img = np.load(os.path.join(data_paths[1], self.data_imgB_paths[i]))
                else:
                    img = imread(os.path.join(data_paths[1], self.data_imgB_paths[i]))
            if img.ndim == 2:
                img = np.expand_dims(img, -1)
            else:
                img = img.transpose((1,2,0))
            self.channels = img.shape[-1]
            self.shape_imgB = (shape[0]*random_crop_scale, shape[1]*random_crop_scale, shape[2]) if random_crops_in_DA else img.shape
            del img
        else:
            if type(X) != list:
                self.X = X.astype(np.uint8) 
                self.Y = Y.astype(np.uint8)
                self.channels = Y.shape[-1] 
            else:
                self.X = X 
                self.Y = Y
                self.channels = Y[0].shape[-1]
            self.len = len(self.X)
            if random_crops_in_DA:
                self.shape_imgA = shape  
                self.shape_imgB = (shape[0]*random_crop_scale, shape[1]*random_crop_scale, shape[2]) 
            else:
                if type(X) != list:
                    self.shape_imgA = X.shape[1:]
                    self.shape_imgB = Y.shape[1:]  
                else:
                    self.shape_imgA = X[0].shape[1:]
                    self.shape_imgB = Y[0].shape[1:]
                
        self.resolution = resolution
        self.res_relation = (1.0,resolution[0]/resolution[1])
        self.o_indexes = np.arange(self.len)
        self.shuffle = shuffle_each_epoch
        self.n_classes = n_classes
        self.out_number = out_number
        self.da = da
        self.da_prob = da_prob
        self.random_crops_in_DA = random_crops_in_DA
        self.random_crop_scale = random_crop_scale
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
        self.grid_d_size = (self.shape_imgA[0]*grid_d_range[0], self.shape_imgA[1]*grid_d_range[1])
        self.channel_shuffle = channel_shuffle

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
        if gridmask: self.trans_made += '_gridmask'+str(self.grid_ratio)+str(self.grid_d_range)+str(self.grid_rotate)+str(self.grid_invert)
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
        self.on_epoch_end()


    def __len__(self):
        """Defines the number of batches per epoch."""
        return int(np.ceil((self.len*self.extra_data_factor)/self.batch_size))


    def __getitem__(self, index):
        """Generation of one batch data.

           Parameters
           ----------
           index : int
               Batch index counter.

           Returns
           -------
           batch_x : 4D Numpy array
               Corresponding X elements of the batch. E.g. ``(batch_size, x, y, channels)``.

           batch_y : 4D Numpy array
               Corresponding Y elements of the batch. E.g. ``(batch_size, x, y, channels)``.
        """

        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        batch_x = np.zeros((len(indexes), *self.shape_imgA), dtype=np.uint8)
        batch_y = np.zeros((len(indexes), *self.shape_imgB), dtype=np.uint8)

        for i, j in zip(range(len(indexes)), indexes):

            # Choose the data source
            if self.in_memory:
                imgA = self.X[j]
                imgB = self.Y[j]
                if imgA.ndim == 4: 
                    imgA = imgA[0]
                if imgB.ndim == 4: 
                    imgB = imgB[0]
            else:
                if self.data_imgA_paths[j].endswith('.npy'):
                    imgA = np.load(os.path.join(self.paths[0], self.data_imgA_paths[j]))
                    imgB = np.load(os.path.join(self.paths[1], self.data_imgB_paths[j]))
                else:
                    imgA = imread(os.path.join(self.paths[0], self.data_imgA_paths[j]))
                    imgB = imread(os.path.join(self.paths[1], self.data_imgB_paths[j]))
                if imgA.ndim == 2:
                    imgA = np.expand_dims(imgA, -1)
                elif imgA.ndim == 4: 
                    imgA = imgA[0]
                elif imgA.ndim == 3:
                    if imgA.shape[0] == 1 or imgA.shape[0] == 3: imgA = imgA.transpose((1,2,0))
                if imgB.ndim == 2:
                    imgB = np.expand_dims(imgB, -1)
                elif imgB.ndim == 4: 
                    imgB = imgB[0]
                elif imgB.ndim == 3:
                    if imgB.shape[0] == 1 or imgB.shape[0] == 3: imgB = imgB.transpose((1,2,0))

                # Ensure uint8
                if imgA.dtype == np.uint16:
                    if np.max(imgA) > 255:
                        imgA = normalize(imgA, 0, 65535)
                    else:
                        imgA = imgA.astype(np.uint8)

                if imgB.dtype == np.uint16:
                    if np.max(imgB) > 255:
                        imgB = normalize(imgB, 0, 65535)
                    else:
                        imgB = imgB.astype(np.uint8)

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

                batch_x[i], batch_y[i] = random_crop(imgA, imgB, self.shape_imgB[:2], self.val, img_prob=img_prob, 
                    scale=self.random_crop_scale)
            else:
                batch_x[i], batch_y[i] = imgA, imgB

            # Apply transformations
            if self.da:
                extra_img = np.random.randint(0, self.len-1) if self.len > 2 else 0
                if self.in_memory:
                    e_imgA = self.X[extra_img]
                    e_imgB = self.Y[extra_img]
                    if e_imgA.ndim == 4: 
                        e_imgA = e_imgA[0]
                    if e_imgB.ndim == 4: 
                        e_imgB = e_imgB[0]
                else:
                    if self.data_imgA_paths[extra_img].endswith('.npy'):
                        e_imgA = np.load(os.path.join(self.paths[0], self.data_imgA_paths[extra_img]))
                        e_imgB = np.load(os.path.join(self.paths[1], self.data_imgB_paths[extra_img]))
                    else:
                        e_imgA = imread(os.path.join(self.paths[0], self.data_imgA_paths[extra_img]))
                        e_imgB = imread(os.path.join(self.paths[1], self.data_imgB_paths[extra_img]))
                    if e_imgA.ndim == 2:
                        e_imgA = np.expand_dims(e_imgA, -1)
                    else:
                        if e_imgA.shape[0] == 1 or e_imgA.shape[0] == 3: e_imgA = e_imgA.transpose((1,2,0))

                    # Ensure uint8
                    if e_imgA.dtype == np.uint16:
                        if np.max(e_imgA) > 255:
                            e_imgA = normalize(e_imgA, 0, 65535)
                        else:
                            e_imgA = e_imgA.astype(np.uint8)

                    if e_imgB.ndim == 2:
                        e_imgB = np.expand_dims(e_imgB, -1)
                    else:
                        if e_imgB.shape[0] == 1 or e_imgB.shape[0] == 3: e_imgB = e_imgB.transpose((1,2,0))

                batch_x[i], batch_y[i] = self.apply_transform(batch_x[i], batch_y[i], e_imgA=e_imgA, e_imgB=e_imgB)

        # Divide the values
        #if self.div_X_on_load: batch_x = batch_x/255
        #if self.div_Y_on_load: batch_y = batch_y/255

        self.total_batches_seen += 1

        if self.out_number == 1:
            return batch_x, batch_y
        else:
            return ([batch_x], [batch_y]*self.out_number)


    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        ia.seed(self.seed + self.total_batches_seen)
        self.indexes = self.o_indexes
        if self.shuffle:
            random.Random(self.seed + self.total_batches_seen).shuffle(self.indexes)


    def apply_transform(self, imgA, imgB, e_imgA=None, e_imgB=None):
        """Transform the input imgA and its imgB at the same time

           Parameters
           ----------
           imgA : 3D Numpy array
               Image to transform. E.g. ``(x, y, channels)``.

           imgB : 3D Numpy array
               Mask to transform. E.g. ``(x, y, channels)``.

           e_imgA : D Numpy array
               Extra image to help transforming ``imgA``. E.g. ``(x, y, channels)``.

           e_imgB : 4D Numpy array
               Extra imgB to help transforming ``imgB``. E.g. ``(x, y, channels)``.

           Returns
           -------
           trans_image : 3D Numpy array
               Transformed image. E.g. ``(x, y, channels)``.

           trans_imgB : 3D Numpy array
               Transformed image imgB. E.g. ``(x, y, channels)``.
        """

        # Apply cutout
        if self.cutout and random.uniform(0, 1) < self.da_prob:
            imgA, imgB = cutout(imgA, imgB, self.channels, -1, self.cout_nb_iterations, self.cout_size, self.cout_cval,
                                 self.res_relation, self.cout_apply_to_mask)

        # Apply cblur
        if self.cutblur and random.uniform(0, 1) < self.da_prob:
            imgA = cutblur(imgA, self.cblur_size, self.cblur_down_range, self.cblur_inside)

        # Apply cutmix
        if self.cutmix and random.uniform(0, 1) < self.da_prob:
            imgA, imgB = cutmix(imgA, e_imgA, imgB, e_imgB, self.cmix_size)

        # Apply cutnoise
        if self.cutnoise and random.uniform(0, 1) < self.da_prob:
            imgA = cutnoise(imgA, self.cnoise_scale, self.cnoise_nb_iterations, self.cnoise_size)

        # Apply misalignment
        if self.misalignment and random.uniform(0, 1) < self.da_prob:
            imgA, imgB = misalignment(imgA, imgB, self.ms_displacement, self.ms_rotate_ratio)

        # Apply brightness
        if self.brightness and random.uniform(0, 1) < self.da_prob:
            imgA = brightness(imgA, brightness_factor=self.brightness_factor)

        # Apply contrast
        if self.contrast and random.uniform(0, 1) < self.da_prob:
            imgA = contrast(imgA, contrast_factor=self.contrast_factor)

        # Apply brightness (EM)
        if self.brightness_em and random.uniform(0, 1) < self.da_prob:
            imgA = brightness_em(imgA, brightness_em_factor=self.brightness_em_factor)

        # Apply contrast (EM)
        if self.contrast_em and random.uniform(0, 1) < self.da_prob:
            imgA = contrast_em(imgA, contrast_em_factor=self.contrast_em_factor)

        # Apply missing parts
        if self.missing_parts and random.uniform(0, 1) < self.da_prob:
            imgA = missing_parts(imgA, self.missp_iterations)

        # Convert to grayscale
        if self.grayscale and random.uniform(0, 1) < self.da_prob:
            imgA = grayscale(imgA)

        # Apply channel shuffle
        if self.channel_shuffle and random.uniform(0, 1) < self.da_prob:
            imgA = shuffle_channels(imgA)

        # Apply GridMask
        if self.gridmask and random.uniform(0, 1) < self.da_prob:
            imgA = GridMask(imgA, self.channels, -1, self.grid_ratio, self.grid_d_size, self.grid_rotate,
                             self.grid_invert)

        # Apply transformations to both images
        augseq_det = self.seq.to_deterministic()
        imgA = augseq_det.augment_image(imgA)
        imgB = augseq_det.augment_image(imgB)
        return imgA, imgB


    def __draw_grid(self, im, grid_width=50):
        """Draw grid of the specified size on an image.

           Parameters
           ----------
           im : 3D Numpy array
               Image to be modified. E. g. ``(x, y, channels)``

           grid_width : int, optional
               Grid's width.
        """
        v = 1 if int(np.max(im)) == 0 else int(np.max(im))

        for i in range(0, im.shape[0], grid_width):
            im[i] = v
        for j in range(0, im.shape[1], grid_width):
            im[:, j] = v


    def get_transformed_samples(self, num_examples, save_to_dir=False, out_dir='aug', save_prefix=None, train=True,
                                random_images=True, draw_grid=True):
        """Apply selected transformations to a defined number of images from the dataset.

           Parameters
           ----------
           num_examples : int
               Number of examples to generate.

           save_to_dir : bool, optional
               Save the images generated. The purpose of this variable is to check the images generated by data
               augmentation.

           out_dir : str, optional
               Name of the folder where the examples will be stored. If any provided the examples will be generated
               under a folder ``aug``.

           save_prefix : str, optional
               Prefix to add to the generated examples' name.

           train : bool, optional
               To avoid drawing a grid on the generated images. This should be set when the samples will be used for
               training.

           random_images : bool, optional
               Randomly select images from the dataset. If ``False`` the examples will be generated from the start of
               the dataset.

           draw_grid : bool, optional
               Draw a grid in the generated samples. Useful to see some types of deformations.

           Returns
           -------
           batch_x : 4D Numpy array
               Batch of imageA. E.g. ``(num_examples, x, y, channels)``.

           batch_y : 4D Numpy array
               Batch of imageB. E.g. ``(num_examples, x, y, channels)``.
        """

        if random_images == False and num_examples > self.len:
            num_examples = self.len
            print("WARNING: More samples requested than the ones available. 'num_examples' fixed to {}".format(num_examples))

        batch_x = np.zeros((num_examples, *self.shape_imgA), dtype=np.uint8)
        batch_y = np.zeros((num_examples, *self.shape_imgB), dtype=np.uint8)

        if save_to_dir:
            p = '_' if save_prefix is None else str(save_prefix)
            os.makedirs(out_dir, exist_ok=True)

        # Generate the examples
        print("0) Creating the examples of data augmentation . . .")
        for i in tqdm(range(num_examples)):
            if random_images:
                pos = np.random.randint(0, self.len-1) if self.len > 2 else 0
            else:
                pos = i

            # Take the data samples
            if self.in_memory:
                imgA = self.X[pos]
                imgB = self.Y[pos]
                if imgA.ndim == 4: 
                    imgA = imgA[0]
                if imgB.ndim == 4: 
                    imgB = imgB[0]
            else:
                if self.data_imgA_paths[pos].endswith('.npy'):
                    imgA = np.load(os.path.join(self.paths[0], self.data_imgA_paths[pos]))
                    imgB = np.load(os.path.join(self.paths[1], self.data_imgB_paths[pos]))
                else:
                    imgA = imread(os.path.join(self.paths[0], self.data_imgA_paths[pos]))
                    imgB = imread(os.path.join(self.paths[1], self.data_imgB_paths[pos]))
                if imgA.ndim == 2:
                    imgA = np.expand_dims(imgA, -1)
                elif imgA.ndim == 4: 
                    imgA = imgA[0]
                elif imgA.ndim == 3: 
                    if imgA.shape[0] == 1 or imgA.shape[0] == 3: imgA = imgA.transpose((1,2,0))
                # Ensure uint8
                if imgA.dtype == np.uint16:
                    if np.max(imgA) > 255:
                        imgA = normalize(imgA, 0, 65535)
                    else:
                        imgA = imgA.astype(np.uint8)

                if imgB.ndim == 2:
                    imgB = np.expand_dims(imgB, -1)
                elif imgB.ndim == 4: 
                    imgB = imgB[0]
                elif imgB.ndim == 3:
                    if imgB.shape[0] == 1 or imgB.shape[0] == 3: imgB = imgB.transpose((1,2,0))
                # Ensure uint8
                if imgB.dtype == np.uint16:
                    if np.max(imgB) > 255:
                        imgB = normalize(imgB, 0, 65535)
                    else:
                        imgB = imgB.astype(np.uint8)
                
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

                batch_x[i], batch_y[i] = random_crop(imgA, imgB, self.shape_imgB[:2], self.val, 
                    img_prob=img_prob, scale=self.random_crop_scale)
            else:
                batch_x[i], batch_y[i] = imgA, imgB

            if save_to_dir:
                o_x = np.copy(batch_x[i])
                o_y = np.copy(batch_y[i])

            # Apply transformations
            if self.da:
                if not train and draw_grid:
                    self.__draw_grid(batch_x[i])
                    self.__draw_grid(batch_y[i])

                extra_img = np.random.randint(0, self.len-1) if self.len > 2 else 0
                if self.in_memory:
                    e_imgA = self.X[extra_img]
                    e_imgB = self.Y[extra_img]
                    if e_imgA.ndim == 4: 
                        e_imgA = e_imgA[0]
                    if e_imgB.ndim == 4: 
                        e_imgB = e_imgB[0]
                else:
                    if self.data_imgA_paths[extra_img].endswith('.npy'):
                        imgA = np.load(os.path.join(self.paths[0], self.data_imgA_paths[extra_img]))
                        imgB = np.load(os.path.join(self.paths[1], self.data_imgB_paths[extra_img]))
                    else:
                        e_imgA = imread(os.path.join(self.paths[0], self.data_imgA_paths[extra_img]))
                        e_imgB = imread(os.path.join(self.paths[1], self.data_imgB_paths[extra_img]))
                    if e_imgA.ndim == 2:
                        e_imgA = np.expand_dims(e_imgA, -1)
                    else:
                        if e_imgA.shape[0] == 1 or e_imgA.shape[0] == 3: e_imgA = e_imgA.transpose((1,2,0))

                    # Ensure uint8
                    if e_imgA.dtype == np.uint16:
                        if np.max(e_imgA) > 255:
                            e_imgA = normalize(e_imgA, 0, 65535)
                        else:
                            e_imgA = e_imgA.astype(np.uint8)

                    if e_imgB.ndim == 2:
                        e_imgB = np.expand_dims(e_imgB, -1)
                    else:
                        if e_imgB.shape[0] == 1 or e_imgB.shape[0] == 3: e_imgB = e_imgB.transpose((1,2,0))

                batch_x[i], batch_y[i] = self.apply_transform(
                    batch_x[i], batch_y[i], e_imgA=e_imgA, e_imgB=e_imgB)

            if save_to_dir:
                # Save original images
                if draw_grid:
                    self.__draw_grid(o_x)
                    self.__draw_grid(o_y)

                f = os.path.join(out_dir,str(i)+"_"+str(pos)+'_orig_x'+self.trans_made+".tif")
                aux = np.expand_dims(np.expand_dims(o_x.transpose((2,0,1)), -1), 0).astype(np.float32)
                imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False)
                f = os.path.join(out_dir,str(i)+"_"+str(pos)+'_orig_y'+self.trans_made+".tif")
                aux = np.expand_dims(np.expand_dims(o_y.transpose((2,0,1)), -1), 0).astype(np.float32)
                imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False)

                # Save transformed images
                f = os.path.join(out_dir,str(i)+"_"+str(pos)+p+'x'+self.trans_made+".tif")
                aux = np.expand_dims(np.expand_dims(batch_x[i].transpose((2,0,1)), -1), 0).astype(np.float32)
                imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False)
                f = os.path.join(out_dir,str(i)+"_"+str(pos)+p+'y'+self.trans_made+".tif")
                aux = np.expand_dims(np.expand_dims(batch_y[i].transpose((2,0,1)), -1), 0).astype(np.float32)
                imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False)

        print("### END TR-SAMPLES ###")
        return batch_x, batch_y

