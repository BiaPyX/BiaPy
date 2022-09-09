import numpy as np
import random
import os
from PIL import Image
from skimage.io import imsave

from data.generators.base_data_generator import BaseDataGenerator
from utils.util import denormalize

class VoxelDataGenerator(BaseDataGenerator):
    """Custom 3D data generator based on `imgaug <https://github.com/aleju/imgaug-doc>`_ and our own
       `augmentors.py <https://github.com/danifranco/BiaPy/blob/master/generators/augmentors.py>`_
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

        self.ax_x = None
        if not self.in_memory:
            # Load one image to check axis position
            if self.data_paths[0].endswith('.tif'):
                from PIL import Image
                from PIL.TiffTags import TAGS
                img_aux = Image.open(os.path.join(self.data_paths[0], self.data_paths[0]))
                meta_dict = {TAGS[key] : img_aux.tag[key] for key in img_aux.tag_v2}
                axis = meta_dict['ImageDescription'][0].split('\n')[-2].split('=')[-1]
                self.ax_x = {}
                for k, c in enumerate(axis):
                    self.ax_x[c] = k  
                if 'Z' in self.ax_x: 
                    img_aux = img_aux.transpose((self.ax_x['Z'],self.ax_x['Y'],self.ax_x['X'],self.ax_x['C']))
                self.X_channels = img_aux.shape[-1]
                del img_aux

        self.z_size = self.shape[0] if self.random_crops_in_DA else self.X.shape[1]
        self.zflip = zflip

    def ensure_shape(self, img, mask):
        # Shape adjustment
        if img.ndim == 3: 
            img = np.expand_dims(img, -1)
        elif img.ndim == 4 and self.ax_x is not None:
            if 'Z' in self.ax_x: 
                img = img.transpose((self.ax_x['Z'],self.ax_x['Y'],self.ax_x['X'],self.ax_x['C']))
        if mask.ndim == 3: 
            mask = np.expand_dims(mask, -1)
        return img, mask

    def apply_transform(self, image, mask, e_im=None, e_mask=None):
        # Transpose them so we can merge the z and c channels easily. 
        # z, y, x, c --> x, y, z, c
        image = image.transpose((2,1,0,3))
        mask = mask.transpose((2,1,0,3))

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

        image, mask = super().apply_transform(image, mask, e_im, e_mask)

        # x, y, z, c --> z, y, x, c
        return image.transpose((2,1,0,3)), mask.transpose((2,1,0,3))

    def save_aug_samples(self, img, mask, orig_images, i, pos, out_dir, draw_grid, point_dict):
        # Undo X normalization 
        if self.X_norm['type'] == 'div' and 'div' in self.X_norm: 
            orig_images['o_x'] = orig_images['o_x']*255
            orig_images['o_x2'] = orig_images['o_x2']*255
            img = img*255
        elif self.X_norm['type'] == 'custom':
            img = denormalize(img, self.X_norm['mean'], self.X_norm['std'])
            orig_images['o_x'] = denormalize(orig_images['o_x'], self.X_norm['mean'], self.X_norm['std'])
            orig_images['o_x2'] = denormalize(orig_images['o_x2'], self.X_norm['mean'], self.X_norm['std'])

        # Undo Y normalization 
        if self.first_no_bin_channel != -1:
            if self.div_Y_on_load_bin_channels:
                orig_images['o_y'][...,:self.first_no_bin_channel] = orig_images['o_y'][...,:self.first_no_bin_channel]*255
                orig_images['o_y2'][...,:self.first_no_bin_channel] = orig_images['o_y2'][...,:self.first_no_bin_channel]*255
                mask[...,:self.first_no_bin_channel] = mask[...,:self.first_no_bin_channel]*255
            if self.div_Y_on_load_no_bin_channels:
                if self.first_no_bin_channel != 0:
                    orig_images['o_y'][...,self.first_no_bin_channel:] = orig_images['o_y'][...,self.first_no_bin_channel:]*255
                    orig_images['o_y2'][...,self.first_no_bin_channel:] = orig_images['o_y2'][...,self.first_no_bin_channel:]*255
                    mask[...,self.first_no_bin_channel:] = mask[...,self.first_no_bin_channel:]*255
                else:
                    orig_images['o_y'] = orig_images['o_y']*255
                    orig_images['o_y2'] = orig_images['o_y2']*255
                    mask = mask*255
        else:
            if self.div_Y_on_load_bin_channels: 
                orig_images['o_y'] = orig_images['o_y']*255
                orig_images['o_y2'] = orig_images['o_y2']*255
                mask = mask*255

        os.makedirs(out_dir, exist_ok=True)
        # Original image/mask
        f = os.path.join(out_dir, str(i)+"_orig_x_"+str(pos)+"_"+self.trans_made+'.tiff')
        if draw_grid: self.draw_grid(orig_images['o_x'])
        aux = np.expand_dims((np.transpose(orig_images['o_x'], (0,3,1,2))).astype(np.float32), -1)
        imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False)
        f = os.path.join(out_dir, str(i)+"_orig_y_"+str(pos)+"_"+self.trans_made+'.tiff')
        if draw_grid: self.draw_grid(orig_images['o_y'])
        aux = np.expand_dims((np.transpose(orig_images['o_y'], (0,3,1,2))).astype(np.float32), -1)
        imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False)
        # Transformed
        f = os.path.join(out_dir, str(i)+"_x_aug_"+str(pos)+"_"+self.trans_made+'.tiff')
        aux = np.expand_dims((np.transpose(img, (0,3,1,2))).astype(np.float32), -1)
        imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False)
        # Mask
        f = os.path.join(out_dir, str(i)+"_y_aug_"+str(pos)+"_"+self.trans_made+'.tiff')
        aux = np.expand_dims((np.transpose(mask, (0,3,1,2))).astype(np.float32), -1)
        imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False)

        # Save the original images with a red point and a blue square that represents the point selected with
        # the probability map and the random volume extracted from the original data
        if self.random_crops_in_DA and self.prob_map is not None and i == 0:
            os.makedirs(out_dir, exist_ok=True)

            print("The selected point of the random crop was [{},{},{}]".format(point_dict['oz'],point_dict['oy'],point_dict['ox']))

            aux = (orig_images['o_x2']).astype(np.uint8)
            if aux.shape[-1] == 1: aux = np.repeat(aux, 3, axis=3)
            auxm = (orig_images['o_y2']).astype(np.uint8)
            if auxm.shape[-1] == 1: auxm = np.repeat(auxm, 3, axis=3)

            for s in range(aux.shape[0]):
                if s >= point_dict['s_z'] and s < point_dict['s_z']+self.shape[0]:
                    im = Image.fromarray(aux[s,...,0])
                    im = im.convert('RGB')
                    px = im.load()
                    m = Image.fromarray(auxm[s,...,0])
                    m = m.convert('RGB')
                    py = m.load()

                    # Paint a blue square that represents the crop made. 
                    # Here the axis are x, y and not y, x (numpy)
                    for row in range(point_dict['s_x'], point_dict['s_x']+self.shape[2]):
                        px[row, point_dict['s_y']] = (0, 0, 255)
                        px[row, point_dict['s_y']+self.shape[1]-1] = (0, 0, 255)
                        py[row, point_dict['s_y']] = (0, 0, 255)
                        py[row, point_dict['s_y']+self.shape[1]-1] = (0, 0, 255)
                    for col in range(point_dict['s_y'], point_dict['s_y']+self.shape[1]):
                        px[point_dict['s_x'], col] = (0, 0, 255)
                        px[point_dict['s_x']+self.shape[2]-1, col] = (0, 0, 255)
                        py[point_dict['s_x'], col] = (0, 0, 255)
                        py[point_dict['s_x']+self.shape[2]-1, col] = (0, 0, 255)

                    # Paint the selected point in red
                    if s == point_dict['oz']:
                        p_size=6
                        for row in range(point_dict['ox']-p_size,point_dict['ox']+p_size):
                            for col in range(point_dict['oy']-p_size,point_dict['oy']+p_size):
                                if col >= 0 and col < img.shape[1] and row >= 0 and row < img.shape[2]:
                                    px[row, col] = (255, 0, 0)
                                    py[row, col] = (255, 0, 0)

                    aux[s] = im
                    auxm[s] = m

            aux = np.expand_dims((np.transpose(aux, (0,3,1,2))).astype(np.float32), -1)
            f = os.path.join(out_dir, "extract_example_"+str(pos)+"_mark_x_"+self.trans_made+'.tiff')
            imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False)
            auxm = np.expand_dims((np.transpose(auxm, (0,3,1,2))).astype(np.float32), -1)
            f = os.path.join(out_dir, "extract_example_"+str(pos)+"_mark_y_"+self.trans_made+'.tiff')
            imsave(f, auxm, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False)


