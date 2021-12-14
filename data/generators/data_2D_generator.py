import tensorflow as tf
import numpy as np
import random
import os
from tqdm import tqdm
from PIL import Image
from PIL import ImageEnhance
import imgaug as ia
from skimage.io import imsave, imread
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

from utils.util import img_to_onehot_encoding
from data.data_2D_manipulation import random_crop
from data.generators.augmentors import cutout, cutblur, cutmix, cutnoise, misalignment, brightness, contrast, missing_parts


class ImageDataGenerator(tf.keras.utils.Sequence):
    """Custom 2D ImageDataGenerator based on `imgaug <https://github.com/aleju/imgaug-doc>`_
       and our own `augmentors.py <https://github.com/danifranco/EM_Image_Segmentation/blob/master/generators/augmentors.py>`_
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
           To aply brightness to the images.

       brightness_factor : tuple of 2 floats, optional
           Strength of the brightness range, with valid values being ``0 <= brightness_factor <= 1``. E.g. ``(0.1, 0.3)``.

       contrast : boolen, optional
           To apply contrast changes to the images.

       contrast_factor : tuple of 2 floats, optional
           Strength of the contrast change range, with valid values being ``0 <= contrast_factor <= 1``. E.g. ``(0.1, 0.3)``.

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
           Wheter to apply cutout to the mask.

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
           Scale of the random noise. E.g. ``(0.1, 0.2)``.

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

       random_crops_in_DA : bool, optional
           Decide to make random crops in DA (before transformations).

       shape : 3D int tuple, optional
           Shape of the desired images when using 'random_crops_in_DA'.

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
                 shear_range=(-20,20), zoom=False, zoom_range=(0.8,1.2), shift=False, shift_range=(0.1,0.2), vflip=False,
                 hflip=False, elastic=False, e_alpha=(240,250), e_sigma=25, e_mode='constant', g_blur=False,
                 g_sigma=(1.0,2.0), median_blur=False, mb_kernel=(3,7), motion_blur=False, motb_k_range=(3,8),
                 gamma_contrast=False, gc_gamma=(1.25,1.75), brightness=False, brightness_factor=(1,3), contrast=False,
                 contrast_factor=(1,3), dropout=False, drop_range=(0, 0.2), cutout=False, cout_nb_iterations=(1,3),
                 cout_size=(0.2,0.4), cout_cval=0, cout_apply_to_mask=False, cutblur=False, cblur_size=(0.1,0.5),
                 cblur_down_range=(2,8), cblur_inside=True, cutmix=False, cmix_size=(0.2,0.4), cutnoise=False,
                 cnoise_scale=(0.1,0.2), cnoise_nb_iterations=(1,3), cnoise_size=(0.2,0.4), misalignment=False,
                 ms_displacement=16, ms_rotate_ratio=0.0, missing_parts=False, missp_iterations=(30, 40),
                 random_crops_in_DA=False, shape=(256,256,1), prob_map=None, val=False, n_classes=1, out_number=1,
                 extra_data_factor=1):

        if in_memory:
            if X.ndim != 4 or Y.ndim != 4:
                raise ValueError("X and Y must be a 4D Numpy array")
            if X.shape[:3] != Y.shape[:3]:
                raise ValueError("The shape of X and Y must be the same. {} != {}".format(X.shape[:3], Y.shape[:3]))

        if in_memory and (X is None or Y is None):
            raise ValueError("'X' and 'Y' need to be provided together with 'in_memory'")

        if not in_memory and len(data_paths) != 2:
            raise ValueError("'data_paths' must contain the following paths: 1) data path ; 2) data masks path")

        if random_crops_in_DA and (shape[0] != shape[1]):
            raise ValuError("When 'random_crops_in_DA' is selected the shape given must be square, e.g. (256, 256, 1)")

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
            if self.data_paths[0].endswith('.npy'):
                img = np.load(os.path.join(data_paths[0], self.data_paths[0]))
            else:
                img = imread(os.path.join(data_paths[0], self.data_paths[0]))
            if img.ndim == 2:
                img = np.expand_dims(img, -1)
            else:
                if img.shape[0] == 1 or img.shape[0] == 3: img = img.transpose((1,2,0))
            self.div_X_on_load = True if np.max(img) > 100 else False
            self.shape = shape if random_crops_in_DA else img.shape
            # Loop over a few masks to ensure foreground class is present
            self.div_Y_on_load = False
            for i in range(min(10,len(self.data_mask_path))):
                if self.data_mask_path[i].endswith('.npy'):
                    img = np.load(os.path.join(data_paths[1], self.data_mask_path[i]))
                else:
                    img = imread(os.path.join(data_paths[1], self.data_mask_path[i]))
                if np.max(img) > 100: self.div_Y_on_load = True
            if img.ndim == 2:
                img = np.expand_dims(img, -1)
            else:
                img = img.transpose((1,2,0))
            self.channels = img.shape[-1]
            del img
        else:
            self.X = X.astype(np.uint8)
            self.Y = Y.astype(np.uint8)
            self.div_X_on_load = True if np.max(X) > 100 else False
            self.div_Y_on_load = True if np.max(Y) > 100 else False
            self.channels = Y.shape[-1]
            self.len = len(self.X)
            self.shape = shape if random_crops_in_DA else X.shape[1:]

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
        self.missing_parts = missing_parts
        self.missp_iterations = missp_iterations

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
            self.da_options.append(iaa.Sometimes(da_prob, iaa.Affine(rotate=rnd_rot_range)))
            self.trans_made += '_rrot'+str(rnd_rot_range)
        if shear:
            self.da_options.append(iaa.Sometimes(da_prob, iaa.Affine(rotate=shear_range)))
            self.trans_made += '_shear'+str(shear_range)
        if zoom:
            self.da_options.append(iaa.Sometimes(da_prob, iaa.Affine(scale={"x": zoom_range, "y": zoom_range})))
            self.trans_made += '_zoom'+str(zoom_range)
        if shift:
            self.da_options.append(iaa.Sometimes(da_prob, iaa.Affine(translate_percent=shift_range)))
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
            self.trans_made += '_brightness'+str(brightness_factor)
        if contrast:
            self.contrast_factor = contrast_factor
            self.trans_made += '_contrast'+str(contrast_factor)
        if dropout:
            self.da_options.append(iaa.Sometimes(da_prob, iaa.Dropout(p=drop_range)))
            self.trans_made += '_drop'+str(drop_range)
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

        batch_x = np.zeros((len(indexes), *self.shape), dtype=np.uint8)
        batch_y = np.zeros((len(indexes), *self.shape[:2])+(self.channels,), dtype=np.uint8)

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
                if img.ndim == 2:
                    img = np.expand_dims(img, -1)
                else:
                    if img.shape[0] == 1 or img.shape[0] == 3: img = img.transpose((1,2,0))
                if mask.ndim == 2:
                    mask = np.expand_dims(mask, -1)
                else:
                    if mask.shape[0] == 1 or mask.shape[0] == 3: mask = mask.transpose((1,2,0))

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

                batch_x[i], batch_y[i] = random_crop(img, mask, self.shape[:2], self.val, img_prob=img_prob)
            else:
                batch_x[i], batch_y[i] = img, mask

            # Apply transformations
            if self.da:
                extra_img = np.random.randint(0, self.len-1)
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
                    if e_img.ndim == 2:
                        e_img = np.expand_dims(e_img, -1)
                    else:
                        if e_img.shape[0] == 1 or e_img.shape[0] == 3: e_img = e_img.transpose((1,2,0))
                    if e_mask.ndim == 2:
                        e_mask = np.expand_dims(e_mask, -1)
                    else:
                        if e_mask.shape[0] == 1 or e_mask.shape[0] == 3: e_mask = e_mask.transpose((1,2,0))

                batch_x[i], batch_y[i] = self.apply_transform(batch_x[i], batch_y[i], e_im=e_img, e_mask=e_mask)

        # Divide the values
        if self.div_X_on_load: batch_x = batch_x/255
        if self.div_Y_on_load: batch_y = batch_y/255

        # One-hot enconde
        if self.n_classes > 1 and (self.n_classes != self.channels):
            batch_y_ = np.zeros((len(indexes),) + self.shape[:2] + (self.n_classes,), dtype=np.uint8)
            for k in range(len(indexes)):
                batch_y_[k] = np.asarray(img_to_onehot_encoding(batch_y[k], self.n_classes))
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
        if self.shuffle:
            random.Random(self.seed + self.total_batches_seen).shuffle(self.indexes)


    def apply_transform(self, image, mask, e_im=None, e_mask=None):
        """Transform the input image and its mask at the same time with one of the selected choices based on a
           probability.

           Parameters
           ----------
           image : 3D Numpy array
               Image to transform. E.g. ``(x, y, channels)``.

           mask : 3D Numpy array
               Mask to transform. E.g. ``(x, y, channels)``.

           e_img : D Numpy array
               Extra image to help transforming ``image``. E.g. ``(x, y, channels)``.

           e_mask : 4D Numpy array
               Extra mask to help transforming ``mask``. E.g. ``(x, y, channels)``.

           Returns
           -------
           trans_image : 3D Numpy array
               Transformed image. E.g. ``(x, y, channels)``.

           trans_mask : 3D Numpy array
               Transformed image mask. E.g. ``(x, y, channels)``.
        """

        # Apply cutout
        if self.cutout and random.uniform(0, 1) < self.da_prob:
            image, mask = cutout(image, mask, self.cout_nb_iterations, self.cout_size, self.cout_cval,
                                 self.cout_apply_to_mask)

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
            image, mask = misalignment(image, mask, self.ms_displacement, self.ms_rotate_ratio)

        # Apply brightness
        if self.brightness and random.uniform(0, 1) < self.da_prob:
            image = brightness(image, brightness_factor=self.brightness_factor)

        # Apply contrast
        if self.contrast and random.uniform(0, 1) < self.da_prob:
            image = contrast(image, contrast_factor=self.contrast_factor)

        # Apply missing parts
        if self.missing_parts and random.uniform(0, 1) < self.da_prob:
            image = missing_parts(image, self.missp_iterations)

        # Apply transformations to the volume and its mask
        segmap = SegmentationMapsOnImage(mask, shape=mask.shape)
        image, vol_mask = self.seq(image=image, segmentation_maps=segmap)
        mask = vol_mask.get_arr()

        return image, mask


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
                                random_images=True):
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

           Returns
           -------
           batch_x : 4D Numpy array
               Batch of data. E.g. ``(num_examples, x, y, channels)``.

           batch_y : 4D Numpy array
               Batch of data mask. E.g. ``(num_examples, x, y, channels)``.


           Examples
           --------
           ::

               # EXAMPLE 1
               # Generate 10 samples following with the example 1 of the class definition
               X_train = np.ones((1776, 256, 256, 1))
               Y_train = np.ones((1776, 256, 256, 1))

               data_gen_args = dict(X=X_train, Y=Y_train, batch_size=6, shape=(256, 256, 1), shuffle_each_epoch=True,
                                    rotation_range=True, vflip=True, hflip=True)

               train_generator = ImageDataGenerator(**data_gen_args)

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
               train_generator = ImageDataGenerator(**data_gen_args)

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

        batch_x = np.zeros((num_examples, *self.shape), dtype=np.uint8)
        batch_y = np.zeros((num_examples, *self.shape[:2])+(self.channels,), dtype=np.uint8)

        if save_to_dir:
            p = '_' if save_prefix is None else str(save_prefix)
            os.makedirs(out_dir, exist_ok=True)

        # Generate the examples
        print("0) Creating the examples of data augmentation . . .")
        for i in tqdm(range(num_examples)):
            pos = np.random.randint(0, self.len-1) if random_images else i

            # Take the data samples
            if self.in_memory:
                img = self.X[pos]
                mask = self.Y[pos]
            else:
                if self.data_paths[pos].endswith('.npy'):
                    img = np.load(os.path.join(self.paths[0], self.data_paths[pos]))
                    mask = np.load(os.path.join(self.paths[1], self.data_mask_path[pos]))
                else:
                    img = imread(os.path.join(self.paths[0], self.data_paths[pos]))
                    mask = imread(os.path.join(self.paths[1], self.data_mask_path[pos]))
                if img.ndim == 2:
                    img = np.expand_dims(img, -1)
                else:
                    if img.shape[0] == 1 or img.shape[0] == 3: img = img.transpose((1,2,0))
                if mask.ndim == 2:
                    mask = np.expand_dims(mask, -1)
                else:
                    if mask.shape[0] == 1 or mask.shape[0] == 3: mask = mask.transpose((1,2,0))

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

                batch_x[i], batch_y[i], ox, oy,\
                s_x, s_y = random_crop(img, mask, self.shape[:2], self.val, img_prob=img_prob, draw_prob_map_points=True)
            else:
                batch_x[i], batch_y[i] = img, mask

            if save_to_dir:
                o_x = np.copy(batch_x[i])
                o_y = np.copy(batch_y[i])

            # Apply transformations
            if self.da:
                if not train:
                    self.__draw_grid(batch_x[i])
                    self.__draw_grid(batch_y[i])

                extra_img = np.random.randint(0, self.len-1)
                if self.in_memory:
                    e_img = self.X[extra_img]
                    e_mask = self.Y[extra_img]
                else:
                    if self.data_paths[extra_img].endswith('.npy'):
                        img = np.load(os.path.join(self.paths[0], self.data_paths[extra_img]))
                        mask = np.load(os.path.join(self.paths[1], self.data_mask_path[extra_img]))
                    else:
                        e_img = imread(os.path.join(self.paths[0], self.data_paths[extra_img]))
                        e_mask = imread(os.path.join(self.paths[1], self.data_mask_path[extra_img]))
                    if e_img.ndim == 2:
                        e_img = np.expand_dims(e_img, -1)
                    else:
                        if e_img.shape[0] == 1 or e_img.shape[0] == 3: e_img = e_img.transpose((1,2,0))
                    if e_mask.ndim == 2:
                        e_mask = np.expand_dims(e_mask, -1)
                    else:
                        if e_mask.shape[0] == 1 or e_mask.shape[0] == 3: e_mask = e_mask.transpose((1,2,0))

                batch_x[i], batch_y[i] = self.apply_transform(
                    batch_x[i], batch_y[i], e_im=e_img, e_mask=e_mask)

            if save_to_dir:
                # Save original images
                self.__draw_grid(o_x)
                self.__draw_grid(o_y)

                f = os.path.join(out_dir,str(pos)+'_orig_x'+self.trans_made+".tif")
                aux = np.expand_dims(np.expand_dims(o_x.transpose((2,0,1)), -1), 0).astype(np.float32)
                imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False)
                f = os.path.join(out_dir,str(pos)+'_orig_y'+self.trans_made+".tif")
                aux = np.expand_dims(np.expand_dims(o_y.transpose((2,0,1)), -1), 0).astype(np.float32)
                imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False)

                # Save transformed images
                f = os.path.join(out_dir, str(pos)+p+'x'+self.trans_made+".tif")
                aux = np.expand_dims(np.expand_dims(batch_x[i].transpose((2,0,1)), -1), 0).astype(np.float32)
                imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False)
                f = os.path.join(out_dir, str(pos)+p+'y'+self.trans_made+".tif")
                aux = np.expand_dims(np.expand_dims(batch_y[i].transpose((2,0,1)), -1), 0).astype(np.float32)
                imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False)

                # Save the original images with a point that represents the selected coordinates to be the center of
                # the crop
                if self.random_crops_in_DA and self.prob_map is not None:
                    if self.in_memory:
                        img = self.X[pos]
                    else:
                        if self.data_paths[pos].endswith('.npy'):
                            img = np.load(os.path.join(self.paths[0], self.data_paths[pos]))
                        else:
                            img = imread(os.path.join(self.paths[0], self.data_paths[pos]))
                        if img.ndim == 2:
                            img = np.expand_dims(img, -1)
                        else:
                            img = img.transpose((1,2,0))
                        if np.max(img) < 100: img = img

                    if self.shape[-1] == 1:
                        im = Image.fromarray(np.repeat(img, 3, axis=2), 'RGB')
                    else:
                        im = Image.fromarray(img, 'RGB')
                    px = im.load()

                    # Paint the selected point in red
                    p_size=6
                    for col in range(oy-p_size, oy+p_size):
                        for row in range(ox-p_size, ox+p_size):
                            if col >= 0 and col < img.shape[0] and \
                               row >= 0 and row < img.shape[1]:
                               px[row, col] = (255, 0, 0)

                    # Paint a blue square that represents the crop made
                    for row in range(s_x, s_x+self.shape[0]):
                        px[row, s_y] = (0, 0, 255)
                        px[row, s_y+self.shape[0]-1] = (0, 0, 255)
                    for col in range(s_y, s_y+self.shape[0]):
                        px[s_x, col] = (0, 0, 255)
                        px[s_x+self.shape[0]-1, col] = (0, 0, 255)

                    im.save(os.path.join(out_dir, str(pos)+p+'mark_x'+self.trans_made+'.png'))

                    if self.in_memory:
                        mask = self.Y[pos]
                    else:
                        if self.data_paths[pos].endswith('.npy'):
                            mask = np.load(os.path.join(self.paths[1], self.data_mask_path[pos]))
                        else:
                            mask = imread(os.path.join(self.paths[1], self.data_mask_path[pos]))
                        if mask.ndim == 2:
                            mask = np.expand_dims(mask, -1)
                        else:
                            mask = mask.transpose((1,2,0))
                        if np.max(mask) < 100: mask = mask
                    if mask.shape[-1] == 1:
                        m = Image.fromarray(np.repeat(mask, 3, axis=2), 'RGB')
                    else:
                        m = Image.fromarray(mask, 'RGB')
                    px = m.load()

                    # Paint the selected point in red
                    for col in range(oy-p_size, oy+p_size):
                        for row in range(ox-p_size, ox+p_size):
                            if col >= 0 and col < mask.shape[0] and \
                               row >= 0 and row < mask.shape[1]:
                               px[row, col] = (255, 0, 0)

                    # Paint a blue square that represents the crop made
                    for row in range(s_x, s_x+self.shape[0]):
                        px[row, s_y] = (0, 0, 255)
                        px[row, s_y+self.shape[0]-1] = (0, 0, 255)
                    for col in range(s_y, s_y+self.shape[0]):
                        px[s_x, col] = (0, 0, 255)
                        px[s_x+self.shape[0]-1, col] = (0, 0, 255)

                    m.save(os.path.join(out_dir, str(pos)+p+'mark_y'+self.trans_made+'.png'))

        print("### END TR-SAMPLES ###")
        return batch_x, batch_y

