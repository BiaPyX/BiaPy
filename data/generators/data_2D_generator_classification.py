import tensorflow as tf
import numpy as np
import random
import os
import cv2
from tqdm import tqdm
from PIL import Image
from PIL import ImageEnhance
import imgaug as ia
from skimage.io import imsave, imread
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

from utils.util import img_to_onehot_encoding
from data.data_2D_manipulation import random_crop_classification


class ClassImageDataGenerator(tf.keras.utils.Sequence):
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

       data_path : List of str, optional
          If ``in_memory`` is ``True`` this should contain the path to load images.

       n_classes : int
           Number of classes to predict.

       batch_size : int, optional
           Size of the batches.

       seed : int, optional
           Seed for random functions.

       shuffle_each_epoch : bool, optional
           To decide if the indexes will be shuffled after every epoch.

       in_memory : bool, optional
           If ``True`` data used will be ``X`` and ``Y``. If ``False`` it will be loaded directly from disk using
           ``data_paths``.

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
    """

    def __init__(self, X, Y, data_path, n_classes, batch_size=32, seed=0, shuffle_each_epoch=False, in_memory=False,
                 da=True, da_prob=0.5, rotation90=False, rand_rot=False, rnd_rot_range=(-180,180), shear=False,
                 shear_range=(-20,20), zoom=False, zoom_range=(0.8,1.2), shift=False, shift_range=(0.1,0.2), vflip=False,
                 hflip=False, elastic=False, e_alpha=(240,250), e_sigma=25, e_mode='constant', g_blur=False,
                 g_sigma=(1.0,2.0), median_blur=False, mb_kernel=(3,7), motion_blur=False, motb_k_range=(3,8),
                 gamma_contrast=False, gc_gamma=(1.25,1.75), dropout=False, drop_range=(0, 0.2), val=False,
                 resize_shape=None):

        self.batch_size = batch_size
        self.in_memory = in_memory

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
            random.shuffle(self.all_samples)
            self.len = len(self.all_samples)
        else:
            self.X = X
            self.Y = Y
            self.len = self.X.shape[0]

        # Check if a division is required
        if not in_memory:
            img = imread(os.path.join(data_path, self.classes[self.all_samples[0]], self.all_samples[0]))
            if img.ndim == 2:
                img = np.expand_dims(img, -1)
            else:
                if img.shape[0] <= 3: img = img.transpose((1,2,0))

            # Ensure uint8
            if img.dtype == np.uint16:
                if np.max(img) > 255:
                    img = normalize(img, 0, 65535)
                else:
                    img = img.astype(np.uint8)
        else:
            img = self.X[0]
        self.div_X_on_load = True if np.max(img) > 100 else False
        self.shape = resize_shape if resize_shape is not None else img.shape

        self.o_indexes = np.arange(self.len)
        self.shuffle = shuffle_each_epoch
        self.n_classes = n_classes
        self.da = da
        self.da_prob = da_prob

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
        if dropout:
            self.da_options.append(iaa.Sometimes(da_prob, iaa.Dropout(p=drop_range)))
            self.trans_made += '_drop'+str(drop_range)

        self.trans_made = self.trans_made.replace(" ", "")
        self.seq = iaa.Sequential(self.da_options)
        self.seed = seed
        ia.seed(seed)
        self.on_epoch_end()


    def __len__(self):
        """Defines the number of batches per epoch."""
        return int(np.ceil(self.len/self.batch_size))


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

           batch_y : List of ints
               Corresponding classes of X. E.g. ``(batch_size)``.
        """

        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        batch_x = np.zeros((len(indexes), *self.shape), dtype=np.uint8)
        batch_y = np.zeros(len(indexes), dtype=np.uint8)

        for i, j in zip(range(len(indexes)), indexes):

            # Choose the data source
            if self.in_memory:
                img = self.X[j]
                batch_y[i] = self.Y[j]
            else:
                sample_id = self.all_samples[j]
                sample_class_dir = self.classes[sample_id]
                img = imread(os.path.join(self.data_path, sample_class_dir, sample_id))
                if img.ndim == 2:
                    img = np.expand_dims(img, -1)
                else:
                    if img.shape[0] <= 3: img = img.transpose((1,2,0))

                # Ensure uint8
                if img.dtype == np.uint16:
                    if np.max(img) > 255:
                        img = normalize(img, 0, 65535)
                    else:
                        img = img.astype(np.uint8)

                batch_y[i] = self.class_numbers[sample_class_dir]

            if img.shape[:-1] != self.shape[:-1]:
                img = self.resize_img(img, self.shape)

            batch_x[i] = img

            # Apply transformations
            if self.da:
                extra_img = np.random.randint(0, self.len-1)
                if self.in_memory:
                    e_img = self.X[extra_img]
                else:
                    sample_id = self.all_samples[extra_img]
                    sample_class_dir = self.classes[sample_id]
                    e_img = imread(os.path.join(self.data_path, sample_class_dir, sample_id))
                    if e_img.ndim == 2:
                        e_img = np.expand_dims(e_img, -1)
                    else:
                        if e_img.shape[0] == 1 or e_img.shape[0] == 3: e_img = e_img.transpose((1,2,0))

                    # Ensure uint8
                    if e_img.dtype == np.uint16:
                        if np.max(e_img) > 255:
                            e_img = normalize(e_img, 0, 65535)
                        else:
                            e_img = e_img.astype(np.uint8)

                batch_x[i] = self.apply_transform(batch_x[i], e_im=e_img)

        # Divide the values
        if self.div_X_on_load: batch_x = batch_x/255

        self.total_batches_seen += 1
        batch_y = tf.keras.utils.to_categorical(batch_y, self.n_classes)

        return batch_x, batch_y

    # For EfficientNet
    def resize_img(self, img, shape):
        return cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        ia.seed(self.seed + self.total_batches_seen)
        self.indexes = self.o_indexes
        if self.shuffle:
            random.Random(self.seed + self.total_batches_seen).shuffle(self.indexes)


    def apply_transform(self, image, e_im=None):
        """Transform the input image with one of the selected choices based on a probability.

           Parameters
           ----------
           image : 3D Numpy array
               Image to transform. E.g. ``(x, y, channels)``.

           e_img : D Numpy array
               Extra image to help transforming ``image``. E.g. ``(x, y, channels)``.

           Returns
           -------
           trans_image : 3D Numpy array
               Transformed image. E.g. ``(x, y, channels)``.
        """

        # Apply transformations to the image
        return self.seq(image=image)


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
               Corresponding classes of X. E.g. ``(batch_size)``.
        """

        if random_images == False and num_examples > self.len:
            num_examples = self.len
            print("WARNING: More samples requested than the ones available. 'num_examples' fixed to {}".format(num_examples))

        batch_x = np.zeros((num_examples, *self.shape), dtype=np.uint8)
        batch_y = np.zeros(num_examples, dtype=np.uint8)

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
            else:
                sample_id = self.all_samples[pos]
                sample_class_dir = self.classes[sample_id]
                img = imread(os.path.join(self.data_path, sample_class_dir, sample_id))
                batch_y[i] = self.class_numbers[sample_class_dir]
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

            batch_x[i] = img

            if save_to_dir:
                o_x = np.copy(batch_x[i])

            # Apply transformations
            if self.da:
                if not train:
                    self.__draw_grid(batch_x[i])

                extra_img = np.random.randint(0, self.len-1)
                if self.in_memory:
                    e_img = self.X[extra_img]
                else:
                    sample_id = self.all_samples[extra_img]
                    sample_class_dir = self.classes[sample_id]
                    e_img = imread(os.path.join(self.data_path, sample_class_dir, sample_id))
                    if e_img.ndim == 2:
                        e_img = np.expand_dims(e_img, -1)
                    else:
                        if e_img.shape[0] == 1 or e_img.shape[0] == 3: e_img = e_img.transpose((1,2,0))
                    # Ensure uint8
                    if e_img.dtype == np.uint16:
                        if np.max(e_img) > 255:
                            e_img = normalize(e_img, 0, 65535)
                        else:
                            e_img = e_img.astype(np.uint8)

                batch_x[i] = self.apply_transform(batch_x[i], e_im=e_img)

            if save_to_dir:
                # Save original images
                self.__draw_grid(o_x)

                f = os.path.join(out_dir,str(pos)+'_orig_x'+self.trans_made+".tif")
                aux = np.expand_dims(np.expand_dims(o_x.transpose((2,0,1)), -1), 0).astype(np.float32)
                imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False)

                # Save transformed images
                f = os.path.join(out_dir, str(pos)+p+'x'+self.trans_made+".tif")
                aux = np.expand_dims(np.expand_dims(batch_x[i].transpose((2,0,1)), -1), 0).astype(np.float32)
                imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False)

        print("### END TR-SAMPLES ###")
        return batch_x, batch_y

