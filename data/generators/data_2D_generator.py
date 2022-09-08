import numpy as np
import random
import os
from tqdm import tqdm
from PIL import Image
from skimage.io import imsave, imread
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

from data.generators.base_data_generator import BaseDataGenerator
from utils.util import img_to_onehot_encoding, ensure_2D_dims_and_datatype, normalize, denormalize
from data.data_2D_manipulation import random_crop
from data.generators.augmentors import (cutout, cutblur, cutmix, cutnoise, misalignment, brightness, contrast,
                                        brightness_em, contrast_em, missing_parts, grayscale, shuffle_channels, GridMask)
from engine.denoising import apply_structN2Vmask                     

class ImageDataGenerator(BaseDataGenerator):
    """Custom 2D ImageDataGenerator based on `imgaug <https://github.com/aleju/imgaug-doc>`_
       and our own `augmentors.py <https://github.com/danifranco/BiaPy/blob/master/generators/augmentors.py>`_
       transformations. This generator will yield an image and its corresponding mask.

       Based on `microDL <https://github.com/czbiohub/microDL>`_ and
       `Shervine's blog <https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly>`_.
    """
    def __init__(self, **kwars):
        super().__init__(**kwars)
        
    def __getitem__(self, index):
        """Generation of one batch data.

           Parameters
           ----------
           index : int
               Batch index counter.

           Returns
           -------
           batch_x : 4D Numpy array
               Corresponding X elements of the batch. E.g. ``(batch_size, y, x, channels)``.

           batch_y : 4D Numpy array
               Corresponding Y elements of the batch. E.g. ``(batch_size, y, x, channels)``.
        """

        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        batch_x = np.zeros((len(indexes), *self.shape), dtype=np.float32)
        batch_y = np.zeros((len(indexes), *self.Y_shape), dtype=np.uint8)

        for i, j in zip(range(len(indexes)), indexes):

            img, mask = self.load_sample(j)
 
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

                batch_x[i], batch_y[i] = random_crop(img, mask, self.shape[:2], self.val, img_prob=img_prob)
            else:
                batch_x[i], batch_y[i] = img, mask

            # Apply transformations
            if self.da:
                e_img, e_mask = None, None
                if self.cutmix:
                    extra_img = np.random.randint(0, self.len-1) if self.len > 2 else 0
                    e_img, e_mask = self.load_sample(extra_img)

                batch_x[i], batch_y[i] = self.apply_transform(batch_x[i], batch_y[i], e_im=e_img, e_mask=e_mask)

        if self.n2v:
            batch_y = batch_y.astype(np.float32)
            self.prepare_n2v(batch_x, batch_y)

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

    def load_sample(self, idx):
        """Load one data sample given its corresponding index."""

        # Choose the data source
        if self.in_memory:
            img = self.X[idx]
            mask = self.Y[idx]

            if self.X_norm['type'] == 'div':
                if 'div' in self.X_norm:
                    img = img/255
                if 'div' in self.Y_norm:
                    mask = mask/255
            elif self.X_norm['type'] == 'custom':
                img = normalize(img, self.X_norm['mean'], self.X_norm['std'])
        else:
            if self.data_paths[idx].endswith('.npy'):
                img = np.load(os.path.join(self.paths[0], self.data_paths[idx]))
                mask = np.load(os.path.join(self.paths[1], self.data_mask_path[idx]))
            else:
                img = imread(os.path.join(self.paths[0], self.data_paths[idx]))
                mask = imread(os.path.join(self.paths[1], self.data_mask_path[idx]))

            img = ensure_2D_dims_and_datatype(img, self.X_norm)
            mask = ensure_2D_dims_and_datatype(mask, self.Y_norm, is_mask=True)
        return img, mask

    def apply_transform(self, image, mask, e_im=None, e_mask=None):
        """Transform the input image and its mask at the same time with one of the selected choices based on a
           probability.

           Parameters
           ----------
           image : 3D Numpy array
               Image to transform. E.g. ``(y, x, channels)``.

           mask : 3D Numpy array
               Mask to transform. E.g. ``(y, x, channels)``.

           e_img : D Numpy array
               Extra image to help transforming ``image``. E.g. ``(y, x, channels)``.

           e_mask : 4D Numpy array
               Extra mask to help transforming ``mask``. E.g. ``(y, x, channels)``.

           Returns
           -------
           trans_image : 3D Numpy array
               Transformed image. E.g. ``(y, x, channels)``.

           trans_mask : 3D Numpy array
               Transformed image mask. E.g. ``(y, x, channels)``.
        """
        # Change dtype to supported one by imgaug
        mask = mask.astype(np.uint8)

        # Apply cutout
        if self.cutout and random.uniform(0, 1) < self.da_prob:
            image, mask = cutout(image, mask, self.X_channels, -1, self.cout_nb_iterations, self.cout_size, self.cout_cval,
                                 self.res_relation, self.cout_apply_to_mask)

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

        # Apply brightness (EM)
        if self.brightness_em and random.uniform(0, 1) < self.da_prob:
            image = brightness_em(image, brightness_em_factor=self.brightness_em_factor)

        # Apply contrast (EM)
        if self.contrast_em and random.uniform(0, 1) < self.da_prob:
            image = contrast_em(image, contrast_em_factor=self.contrast_em_factor)

        # Apply missing parts
        if self.missing_parts and random.uniform(0, 1) < self.da_prob:
            image = missing_parts(image, self.missp_iterations)

        # Convert to grayscale
        if self.grayscale and random.uniform(0, 1) < self.da_prob:
            image = grayscale(image)

        # Apply channel shuffle
        if self.channel_shuffle and random.uniform(0, 1) < self.da_prob:
            image = shuffle_channels(image)

        # Apply GridMask
        if self.gridmask and random.uniform(0, 1) < self.da_prob:
            image = GridMask(image, self.X_channels, -1, self.grid_ratio, self.grid_d_size, self.grid_rotate,
                             self.grid_invert)

        # Apply transformations to the volume and its mask
        segmap = SegmentationMapsOnImage(mask, shape=mask.shape)
        image, vol_mask = self.seq(image=image, segmentation_maps=segmap)
        mask = vol_mask.get_arr()

        return image, mask


    def get_transformed_samples(self, num_examples, save_to_dir=False, out_dir='aug', train=True,
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
               Batch of data. E.g. ``(num_examples, y, x, channels)``.

           batch_y : 4D Numpy array
               Batch of data mask. E.g. ``(num_examples, y, x, channels)``.


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
        batch_x, batch_y = [], []
        if random_images == False and num_examples > self.len:
            num_examples = self.len
            print("WARNING: More samples requested than the ones available. 'num_examples' fixed to {}".format(num_examples))

        if save_to_dir:
            os.makedirs(out_dir, exist_ok=True)

        # Generate the examples
        print("0) Creating the examples of data augmentation . . .")
        for i in tqdm(range(num_examples)):
            if random_images:
                pos = np.random.randint(0, self.len-1) if self.len > 2 else 0
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

                img, mask, oy, ox,\
                s_y, s_x = random_crop(img, mask, self.shape[:2], self.val, img_prob=img_prob, draw_prob_map_points=True)

            if save_to_dir:
                o_x = np.copy(img)
                o_y = np.copy(mask)

            # Apply transformations
            if self.da:
                if not train and draw_grid:
                    self.draw_grid(img)
                    self.draw_grid(mask)

                e_img, e_mask = None, None
                if self.cutmix:
                    extra_img = np.random.randint(0, self.len-1) if self.len > 2 else 0
                    e_img, e_mask = self.load_sample(extra_img)

                img, mask = self.apply_transform(img, mask, e_im=e_img, e_mask=e_mask)

            if self.n2v:
                mask = np.repeat(mask, self.channels*2, axis=-1).astype(np.float32)
                self.prepare_n2v(np.expand_dims(img,0), np.expand_dims(mask,0))
            
            batch_x.append(img)
            batch_y.append(mask)

            if save_to_dir:
                # Save original images
                if draw_grid:
                    self.draw_grid(o_x)
                    self.draw_grid(o_y)

                f = os.path.join(out_dir,str(i)+"_"+str(pos)+'_orig_x'+self.trans_made+".tif")
                aux = np.expand_dims(np.expand_dims(o_x.transpose((2,0,1)), -1), 0).astype(np.float32)
                imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False)
                f = os.path.join(out_dir,str(i)+"_"+str(pos)+'_orig_y'+self.trans_made+".tif")
                aux = np.expand_dims(np.expand_dims(o_y.transpose((2,0,1)), -1), 0).astype(np.float32)
                imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False)

                # Save transformed images
                f = os.path.join(out_dir,str(i)+"_"+str(pos)+'_x'+self.trans_made+".tif")
                aux = np.expand_dims(np.expand_dims(img.transpose((2,0,1)), -1), 0).astype(np.float32)
                imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False)
                f = os.path.join(out_dir,str(i)+"_"+str(pos)+'_y'+self.trans_made+".tif")
                aux = np.expand_dims(np.expand_dims(mask.transpose((2,0,1)), -1), 0).astype(np.float32)
                imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False)

                # Save the original images with a point that represents the selected coordinates to be the center of
                # the crop
                if self.random_crops_in_DA and self.prob_map is not None:
                    img, mask = self.load_sample(pos)

                    # Undo normalization
                    if self.X_norm['type'] == 'div':
                        if 'div' in self.X_norm:
                            img = img*255
                        if 'div' in self.Y_norm:
                            mask = mask*255
                    elif self.X_norm['type'] == 'custom':
                        img = denormalize(img, self.X_norm['mean'], self.X_norm['std'])
                    
                    img, mask = img.astype(np.uint8), mask.astype(np.uint8)

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

                    im.save(os.path.join(out_dir, str(i)+"_"+str(pos)+'_mark_x'+self.trans_made+'.png'))

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

                    m.save(os.path.join(out_dir, str(i)+"_"+str(pos)+'_mark_y'+self.trans_made+'.png'))
            
        print("### END TR-SAMPLES ###")
        return batch_x, batch_y

    def prepare_n2v(self, batch_x, batch_y):
        for c in range(self.channels):
            for j in range(batch_x.shape[0]):       
                coords = self.get_stratified_coords(self.rand_float, box_size=self.box_size, shape=self.shape)                             
                indexing = (j,) + coords + (c,)
                indexing_mask = (j,) + coords + (c + self.channels, )
                y_val = batch_x[indexing]
                x_val = self.value_manipulation(batch_x[j, ..., c], coords,
                    2, self.structN2Vmask)
                
                batch_y[indexing] = y_val
                batch_y[indexing_mask] = 1
                batch_x[indexing] = x_val

                if self.structN2Vmask is not None:
                    apply_structN2Vmask(batch_x[j, ..., c], coords, self.structN2Vmask)