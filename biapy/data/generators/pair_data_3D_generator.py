import numpy as np
import random
import os
from PIL import Image
from skimage.io import imread

from biapy.utils.util import save_tif
from biapy.data.generators.pair_base_data_generator import PairBaseDataGenerator

class Pair3DImageDataGenerator(PairBaseDataGenerator):
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
        self.z_size = self.shape[0] if self.random_crops_in_DA else self.X.shape[1]
        self.zflip = zflip
        self.grid_d_size = (self.shape[1]*self.grid_d_range[0], self.shape[2]*self.grid_d_range[1],\
                            self.shape[0]*self.grid_d_range[0], self.shape[0]*self.grid_d_range[1])

    def ensure_shape(self, img, mask):
        # Shape adjustment
        if img.ndim == 3: 
            img = np.expand_dims(img, -1)
        else:                    
            min_val = min(img.shape)
            channel_pos = img.shape.index(min_val)
            if channel_pos != 3 and img.shape[channel_pos] <= 4:
                new_pos = [x for x in range(4) if x != channel_pos]+[channel_pos,]
                img = img.transpose(new_pos)
        if self.Y_provided:
            if mask.ndim == 3: 
                mask = np.expand_dims(mask, -1)
            else:
                min_val = min(mask.shape)
                channel_pos = mask.shape.index(min_val)
                if channel_pos != 3 and mask.shape[channel_pos] <= 4:
                    new_pos = [x for x in range(4) if x != channel_pos]+[channel_pos,]
                    mask = mask.transpose(new_pos)
            
        # Super-resolution check. if random_crops_in_DA is activated the images have not been cropped yet,
        # so this check can not be done and it will be done in the random crop
        if not self.random_crops_in_DA and self.Y_provided and self.random_crop_scale != 1:
            s = [img.shape[1]*self.random_crop_scale, img.shape[2]*self.random_crop_scale]
            if all(x!=y for x,y in zip(s,mask.shape[1:-1])):
                raise ValueError("Images loaded need to be LR and its HR version. LR shape:"
                    " {} vs HR shape {} is not x{} larger (z axis not taken into account)"
                    .format(img.shape[:-1], mask.shape[:-1], self.random_crop_scale))

        if self.Y_provided:
            return img, mask
        else:
            return img

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

    def save_aug_samples(self, img, mask, orig_images, i, pos, out_dir, point_dict):
        aux = np.expand_dims(orig_images['o_x'],0).astype(np.float32)
        save_tif(aux, out_dir, [str(i)+"_orig_x_"+str(pos)+"_"+self.trans_made+'.tif'], verbose=False)

        aux = np.expand_dims(orig_images['o_y'],0).astype(np.float32)
        save_tif(aux, out_dir, [str(i)+"_orig_y_"+str(pos)+"_"+self.trans_made+'.tif'], verbose=False)

        # Save transformed images/masks
        aux = np.expand_dims(img,0).astype(np.float32)
        save_tif(aux, out_dir, [str(i)+"_x_aug_"+str(pos)+"_"+self.trans_made+'.tif'], verbose=False)
        aux = np.expand_dims(mask,0).astype(np.float32)
        save_tif(aux, out_dir, [str(i)+"_y_aug_"+str(pos)+"_"+self.trans_made+'.tif'], verbose=False)

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

            aux = np.expand_dims(aux,0).astype(np.float32)
            save_tif(aux, out_dir, ["extract_example_"+str(pos)+"_mark_x_"+self.trans_made+'.tif'], verbose=False)

            auxm = np.expand_dims(auxm,0).astype(np.float32)
            save_tif(auxm, out_dir, ["extract_example_"+str(pos)+"_mark_y_"+self.trans_made+'.tif'], verbose=False)


