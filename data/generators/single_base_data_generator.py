from abc import ABCMeta, abstractmethod
import tensorflow as tf
import numpy as np
import random
import os
import cv2
from tqdm import tqdm
import imgaug as ia
from skimage.io import imsave, imread
from imgaug import augmenters as iaa

from data.pre_processing import normalize, norm_range01
from data.generators.augmentors import random_crop_single, random_3D_crop_single, resize_img

class SingleBaseDataGenerator(tf.keras.utils.Sequence, metaclass=ABCMeta):
    """Custom BaseDataGenerator based on `imgaug <https://github.com/aleju/imgaug-doc>`_
       and our own `augmentors.py <https://github.com/danifranco/BiaPy/blob/master/generators/augmentors.py>`_
       transformations.

       Based on `microDL <https://github.com/czbiohub/microDL>`_ and
       `Shervine's blog <https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly>`_.

       Parameters
       ----------
       X : 4D/5D Numpy array
           Data. E.g. ``(num_of_images, y, x, channels)`` for ``2D`` or ``(num_of_images, z, y, x, channels)`` for ``3D``.

       Y : 2D Numpy array
           Image class. ``(num_of_images, class)``.

       data_path : List of str, optional
          If ``in_memory`` is ``True`` this should contain the path to load images.

       n_classes : int
           Number of classes to predict.

       seed : int, optional
           Seed for random functions.

       in_memory : bool, optional
           If ``True`` data used will be ``X`` and ``Y``. If ``False`` it will be loaded directly from disk using
           ``data_path``.

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

       dropout : bool, optional
           To set a certain fraction of pixels in images to zero.

       drop_range : tuple of floats, optional
           Range to take a probability ``p`` to drop pixels. E.g. ``(0, 0.2)`` will take a ``p`` folowing ``0<=p<=0.2``
           and then drop ``p`` percent of all pixels in the image (i.e. convert them to black pixels).

       val : bool, optional
           Advise the generator that the images will be to validate the model to not make random crops (as the val.
           data must be the same on each epoch). Valid when ``random_crops_in_DA`` is set.

       resize_shape : tuple of ints, optional
           If defined the input samples will be scaled into that shape.
    
       norm_custom_mean : float, optional
           Mean of the data used to normalize.

       norm_custom_std : float, optional
           Std of the data used to normalize.
    """

    def __init__(self, ndim, X, Y, data_path, n_classes, seed=0, in_memory=False, da=True, da_prob=0.5, rotation90=False, 
                 rand_rot=False, rnd_rot_range=(-180,180), shear=False, shear_range=(-20,20), zoom=False, zoom_range=(0.8,1.2), 
                 shift=False, shift_range=(0.1,0.2), affine_mode='constant', vflip=False, hflip=False, elastic=False, e_alpha=(240,250), 
                 e_sigma=25, e_mode='constant', g_blur=False, g_sigma=(1.0,2.0), median_blur=False, mb_kernel=(3,7), 
                 motion_blur=False, motb_k_range=(3,8), gamma_contrast=False, gc_gamma=(1.25,1.75), dropout=False, 
                 drop_range=(0, 0.2), val=False, resize_shape=None, norm_custom_mean=None, norm_custom_std=None):

        
        self.ndim = ndim
        self.in_memory = in_memory
        self.z_size = -1 

        if in_memory:
            if X.ndim != (self.ndim+2):
                raise ValueError("X must be a {}D Numpy array".format((self.ndim+1)))

        if in_memory and (X is None or Y is None):
            raise ValueError("'X' and 'Y' need to be provided together with 'in_memory'")

        # Save paths where the data is stored
        if not in_memory:
            self.data_path = data_path
            self.class_names = sorted(next(os.walk(data_path))[1])
            self.class_numbers = {}
            for i, c_name in enumerate(self.class_names):
                self.class_numbers[c_name] = i
            self.classes = {}
            self.all_samples = []
            print("Collecting data ids . . .")
            for folder in self.class_names:
                print("Analizing folder {}".format(os.path.join(data_path,folder)))
                ids = sorted(next(os.walk(os.path.join(data_path,folder)))[2])
                print("Found {} samples".format(len(ids)))
                for i in range(len(ids)):
                    self.classes[ids[i]] = folder
                    self.all_samples.append(ids[i])
            temp = random.shuffle(list(zip(self.all_samples, self.classes)) )
            self.all_samples, self.classes = zip(*temp)
            self.all_samples, self.classes = list(self.all_samples), list(self.classes)

            present_classes = np.unique(np.array(self.classes))
            if len(present_classes) != n_classes:
                raise ValueError("MODEL.N_CLASSES is {} but {} classes found: {}"
                    .format(n_classes, len(present_classes),present_classes))

            self.length = len(self.all_samples)

            # X data analysis
            self.X_norm = {}
            if norm_custom_mean is not None and norm_custom_std is not None:
                sam = []
                for i in range(len(self.data_path)):
                    img, _ = self.load_sample(i)
                    sam.append(img)
                    if resize_shape[-1] != img.shape[-1]:
                        raise ValueError("Channel of the patch size given {} does not correspond with the loaded image {}. "
                                         "Please, check the channels of the images!".format(resize_shape[-1], img.shape[-1]))
                sam = np.array(sam)
                self.X_norm['type'] = 'custom'
                self.X_norm['mean'] = np.mean(sam)
                self.X_norm['std'] = np.std(sam)
                self.X_norm['orig_dtype'] = sam.dtype
                del sam
            else:                
                self.X_norm['type'] = 'div'
                img, _ = self.load_sample(0)
                img, nsteps = norm_range01(img)
                self.X_norm.update(nsteps)
                if resize_shape[-1] != img.shape[-1]:
                    raise ValueError("Channel of the patch size given {} does not correspond with the loaded image {}. "
                                    "Please, check the channels of the images!".format(resize_shape[-1], img.shape[-1]))
        else:
            self.X = X
            self.Y = Y

            present_classes = np.unique(np.array(self.Y))
            if len(present_classes) != n_classes:
                raise ValueError("MODEL.N_CLASSES is {} but {} classes found: {}"
                    .format(n_classes, len(present_classes), present_classes))

            self.length = len(self.X)
    
            # X data analysis and normalization
            self.X_norm = {}
            if norm_custom_mean is not None and norm_custom_std is not None:
                self.X_norm['type'] = 'custom'
                self.X_norm['mean'] = np.mean(self.X)
                self.X_norm['std'] = np.std(self.X)
                self.X_norm['orig_dtype'] = self.X.dtype

                self.X = normalize(self.X, self.X_norm['mean'], self.X_norm['std'])
            else:
                self.X_norm['type'] = 'div'
                if type(X) != list:
                    self.X, normx = norm_range01(self.X)
                else:
                    self.X[0], normx = norm_range01(self.X[0])
                    for i in range(1,len(self.X)):
                        self.X[i], _ = norm_range01(self.X[i])
                self.X_norm.update(normx)
            
            t = "Training" if not val else "Validation"
            if type(X) != list:
                print("{} data normalization - min: {} , max: {} , mean: {}"
                    .format(t,np.min(self.X), np.max(self.X), np.mean(self.X)))
            else:
                print("{} data[0] normalization - min: {} , max: {} , mean: {}"
                    .format(t,np.min(self.X[0]), np.max(self.X[0]), np.mean(self.X[0])))

            img, _ = self.load_sample(0)

        self.shape = resize_shape if resize_shape is not None else img.shape

        self.o_indexes = np.arange(self.length)
        self.n_classes = n_classes
        self.da = da
        self.da_prob = da_prob
        self.val = val

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
        if dropout:
            self.da_options.append(iaa.Sometimes(da_prob, iaa.Dropout(p=drop_range)))
            self.trans_made += '_drop'+str(drop_range)

        self.trans_made = self.trans_made.replace(" ", "")
        self.seq = iaa.Sequential(self.da_options)
        self.seed = seed
        ia.seed(seed)
        self.indexes = self.o_indexes.copy()
        self.len = self.__len__() 
        self.random_crop_func = random_3D_crop_single if self.ndim == 3 else random_crop_single

    @abstractmethod
    def get_transformed_samples(self, num_examples, save_to_dir=False, out_dir='aug', 
        train=True, random_images=True, draw_grid=True):
        NotImplementedError

    @abstractmethod
    def save_aug_samples(self, img, orig_image, i, pos, out_dir, draw_grid):
        NotImplementedError

    @abstractmethod
    def ensure_shape(self, img, mask):
        NotImplementedError

    def __len__(self):
        """Defines the number of samples per epoch."""
        return self.length

    def load_sample(self, idx):
        """Load one data sample given its corresponding index."""
        # Choose the data source
        if self.in_memory:
            img = self.X[idx]
            img = np.squeeze(img)
            img_class = self.Y[idx]
        else:
            sample_id = self.all_samples[idx]
            sample_class_dir = self.classes[sample_id]
            if sample_id.endswith('.npy'):
                img = np.load(os.path.join(self.data_path, sample_class_dir, sample_id))
            else:
                img = imread(os.path.join(self.data_path, sample_class_dir, sample_id))
            img = np.squeeze(img)

            # X normalization
            if self.X_norm:
                if self.X_norm['type'] == 'div':
                    img, _ = norm_range01(img)
                elif self.X_norm['type'] == 'custom':
                    img = normalize(img, self.X_norm['mean'], self.X_norm['std'])

            img_class = self.class_numbers[sample_class_dir]
        img = self.ensure_shape(img)

        return img, img_class
        
    def getitem(self, index):
        return self.__getitem__(index)

    def __getitem__(self, index):
        """Generation of one sample data.

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
        img, img_class =  self.load_sample(index)

        if img.shape[:-1] != self.shape[:-1]:
            img = self.random_crop_func(img, self.shape[:-1], self.val)
            img = resize_img(img, self.shape[:-1])

        # Apply transformations
        if self.da:
            img = self.apply_transform(img)

        return img, img_class

    def apply_transform(self, image):
        """Transform the input image with one of the selected choices based on a probability.

           Parameters
           ----------
           image : 3D Numpy array
               Image to transform. E.g. ``(y, x, channels)``.

           Returns
           -------
           trans_image : 3D Numpy array
               Transformed image. E.g. ``(y, x, channels)``.
        """
        # Save shape
        o_img_shape = image.shape

        # Reshape 3D volumes to 2D image type with multiple channels to pass through imgaug lib
        if self.ndim == 3:
            image = image.reshape(image.shape[:2]+(image.shape[2]*image.shape[3],))

        # Apply transformations to the image
        image = self.seq(image=image)

        # Recover the original shape
        image = image.reshape(o_img_shape)

        return image

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

        for i in range(0, im.shape[0], grid_width):
            im[i] = v
        for j in range(0, im.shape[1], grid_width):
            im[:, j] = v

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
           sample_x : 4D Numpy array
               Batch of data. E.g. ``(num_examples, y, x, channels)``.
        """

        if random_images == False and num_examples > self.length:
            num_examples = self.length
            print("WARNING: More samples requested than the ones available. 'num_examples' fixed to {}".format(num_examples))

        sample_x = []

        # Generate the examples
        print("0) Creating the examples of data augmentation . . .")
        for i in tqdm(range(num_examples)):
            if random_images:
                pos = random.randint(0,self.length-1) if self.length > 2 else 0
            else:
                pos = i

            img, img_class = self.load_sample(pos)

            if save_to_dir:
                orig_images = {}
                orig_images['o_x'] = np.copy(img)

            # Apply transformations
            if self.da:
                if not train and draw_grid:
                    self.draw_grid(img)

                img = self.apply_transform(img)

            sample_x.append(img)

            if save_to_dir:
                self.save_aug_samples(sample_x[i], orig_images, i, pos, out_dir, draw_grid)

        print("### END TR-SAMPLES ###")
        return sample_x

