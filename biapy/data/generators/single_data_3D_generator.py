import numpy as np
import random
import os
import cv2
from tqdm import tqdm
import imgaug as ia
from skimage.io import imread
from imgaug import augmenters as iaa

from biapy.utils.util import save_tif
from biapy.data.generators.single_base_data_generator import SingleBaseDataGenerator
from biapy.data.pre_processing import denormalize

class Single3DImageDataGenerator(SingleBaseDataGenerator):
    """Custom 3D data generator based on `imgaug <https://github.com/aleju/imgaug-doc>`_ and our own
       `augmentors.py <https://github.com/BiaPyX/BiaPy/blob/master/biapy/data/generators/augmentors.py>`_
       transformations. This generator will yield an image and its corresponding mask.

       Based on `microDL <https://github.com/czbiohub/microDL>`_ and
       `Shervine's blog <https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly>`_.

       Parameters
       ----------
       zflip : bool, optional
           To activate flips in z dimension.
    """
    def __init__(self, zflip=False, **kwars):
        super().__init__(**kwars)
        #self.z_size = self.X.shape[1]
        self.zflip = zflip

    def ensure_shape(self, img):
        # Shape adjustment
        if img.ndim == 3:
            img = np.expand_dims(img, -1)
        else:
            min_val = min(img.shape)
            channel_pos = img.shape.index(min_val)
            if channel_pos != 3 and img.shape[channel_pos] <= 4:
                new_pos = [x for x in range(4) if x != channel_pos]+[channel_pos,]
                img = img.transpose(new_pos)
        
        if self.convert_to_rgb and self.shape[-1] == 3 and img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
            
        return img

    def apply_transform(self, image):
        # Transpose them so we can merge the z and c channels easily.
        # z, y, x, c --> x, y, z, c
        image = image.transpose((2,1,0,3))

        # Apply flips in z as imgaug can not do it
        if self.zflip and random.uniform(0, 1) < self.da_prob:
            l_image = []
            for i in range(image.shape[-1]):
                l_image.append(np.expand_dims(np.flip(image[...,i], 2), -1))
            image = np.concatenate(l_image, axis=-1)

        image = super().apply_transform(image)

        # x, y, z, c --> z, y, x, c
        return image.transpose((2,1,0,3))

    def save_aug_samples(self, img, orig_images, i, pos, out_dir, draw_grid):
        # Undo X normalization
        if self.X_norm['type'] == 'div' and 'div' in self.X_norm:
            orig_images['o_x'] = orig_images['o_x']*255
            img = img*255
        elif self.X_norm['type'] == 'custom':
            img = denormalize(img, self.X_norm['mean'], self.X_norm['std'])
            orig_images['o_x'] = denormalize(orig_images['o_x'], self.X_norm['mean'], self.X_norm['std'])

        # Original image
        if draw_grid: self.draw_grid(orig_images['o_x'])
        aux = np.expand_dims(orig_images['o_x'], 0).astype(np.float32)
        save_tif(aux, out_dir, [str(i)+"_orig_x_"+str(pos)+"_"+self.trans_made+'.tif'], verbose=False)
        # Transformed
        aux = np.expand_dims(img, 0).astype(np.float32)
        save_tif(aux, out_dir, [str(i)+"_x_aug_"+str(pos)+"_"+self.trans_made+'.tif'], verbose=False)
