import numpy as np
import os
from PIL import Image
from skimage.io import imsave

from data.generators.pair_base_data_generator import PairBaseDataGenerator
from data.pre_processing import denormalize

class Pair2DImageDataGenerator(PairBaseDataGenerator):
    """Custom 2D data generator based on `imgaug <https://github.com/aleju/imgaug-doc>`_
       and our own `augmentors.py <https://github.com/danifranco/BiaPy/blob/master/generators/augmentors.py>`_
       transformations. This generator will yield an image and its corresponding mask.

       Based on `microDL <https://github.com/czbiohub/microDL>`_ and
       `Shervine's blog <https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly>`_.
    """
    def __init__(self, **kwars):
        super().__init__(**kwars)

    def ensure_shape(self, img, mask):
        # Shape adjustment
        if img.ndim == 2:
            img = np.expand_dims(img, -1)
        else:
            if img.shape[0] <= 3: img = img.transpose((1,2,0))     
        if mask.ndim == 2: 
            mask = np.expand_dims(mask, -1)
        else:
            if mask.shape[0] <= 3: mask = mask.transpose((1,2,0))

        # Super-resolution check
        s = [img.shape[0]*self.random_crop_scale, img.shape[1]*self.random_crop_scale]
        if all(x!=y for x,y in zip(s,mask.shape[:-1])):
            raise ValueError("Images loaded need to be LR and its HR version. LR shape:"
                " {} vs HR shape {} is not x{} larger".format(img.shape[:-1], mask.shape[:-1], self.random_crop_scale))

        return img, mask

    def save_aug_samples(self, img, mask, orig_images, i, pos, out_dir, draw_grid, point_dict):
        if draw_grid:
            self.draw_grid(orig_images['o_x'])
            self.draw_grid(orig_images['o_y'])
            
        os.makedirs(out_dir, exist_ok=True)
        f = os.path.join(out_dir,str(i)+"_"+str(pos)+'_orig_x'+self.trans_made+".tif")
        aux = np.expand_dims(np.expand_dims(orig_images['o_x'].transpose((2,0,1)), -1), 0).astype(np.float32)
        imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False, compression=('zlib', 1))
        f = os.path.join(out_dir,str(i)+"_"+str(pos)+'_orig_y'+self.trans_made+".tif")
        aux = np.expand_dims(np.expand_dims(orig_images['o_y'].transpose((2,0,1)), -1), 0).astype(np.float32)
        imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False, compression=('zlib', 1))

        # Save transformed images
        f = os.path.join(out_dir,str(i)+"_"+str(pos)+'_x'+self.trans_made+".tif")
        aux = np.expand_dims(np.expand_dims(img.transpose((2,0,1)), -1), 0).astype(np.float32)
        imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False, compression=('zlib', 1))
        f = os.path.join(out_dir,str(i)+"_"+str(pos)+'_y'+self.trans_made+".tif")
        aux = np.expand_dims(np.expand_dims(mask.transpose((2,0,1)), -1), 0).astype(np.float32)
        imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False, compression=('zlib', 1))

        # Save the original images with a point that represents the selected coordinates to be the center of
        # the crop
        if self.random_crops_in_DA and self.prob_map is not None:
            img, mask = self.load_sample(pos)

            # Undo X normalization 
            if self.X_norm['type'] == 'div':
                if 'div' in self.X_norm:
                    img = img*255
            elif self.X_norm['type'] == 'custom':
                img = denormalize(img, self.X_norm['mean'], self.X_norm['std'])

            # Undo Y normalization
            if self.normalizeY == 'as_mask':
                if self.div_Y_on_load_bin_channels: 
                    mask = mask*255
            elif self.normalizeY == 'as_image':
                if self.X_norm['type'] == 'div':
                    if 'div' in self.X_norm:
                        mask = mask*255
                elif self.X_norm['type'] == 'custom':
                    mask = denormalize(mask, self.X_norm['mean'], self.X_norm['std'])
                    
            img, mask = img.astype(np.uint8), mask.astype(np.uint8)

            if self.shape[-1] == 1:
                im = Image.fromarray(np.repeat(img, 3, axis=2), 'RGB')
            else:
                im = Image.fromarray(img, 'RGB')
            px = im.load()

            # Paint the selected point in red
            p_size=6
            for col in range(point_dict['oy']-p_size, point_dict['oy']+p_size):
                for row in range(point_dict['ox']-p_size, point_dict['ox']+p_size):
                    if col >= 0 and col < img.shape[0] and \
                        row >= 0 and row < img.shape[1]:
                        px[row, col] = (255, 0, 0)

            # Paint a blue square that represents the crop made
            for row in range(point_dict['s_x'], point_dict['s_x']+self.shape[0]):
                px[row, point_dict['s_y']] = (0, 0, 255)
                px[row, point_dict['s_y']+self.shape[0]-1] = (0, 0, 255)
            for col in range(point_dict['s_y'], point_dict['s_y']+self.shape[0]):
                px[point_dict['s_x'], col] = (0, 0, 255)
                px[point_dict['s_x']+self.shape[0]-1, col] = (0, 0, 255)

            im.save(os.path.join(out_dir, str(i)+"_"+str(pos)+'_mark_x'+self.trans_made+'.png'))

            if mask.shape[-1] == 1:
                m = Image.fromarray(np.repeat(mask, 3, axis=2), 'RGB')
            else:
                m = Image.fromarray(mask, 'RGB')
            px = m.load()

            # Paint the selected point in red
            for col in range(point_dict['oy']-p_size, point_dict['oy']+p_size):
                for row in range(point_dict['ox']-p_size, point_dict['ox']+p_size):
                    if col >= 0 and col < mask.shape[0] and \
                        row >= 0 and row < mask.shape[1]:
                        px[row, col] = (255, 0, 0)

            # Paint a blue square that represents the crop made
            for row in range(point_dict['s_x'], point_dict['s_x']+self.shape[0]):
                px[row, point_dict['s_y']] = (0, 0, 255)
                px[row, point_dict['s_y']+self.shape[0]-1] = (0, 0, 255)
            for col in range(point_dict['s_y'], point_dict['s_y']+self.shape[0]):
                px[point_dict['s_x'], col] = (0, 0, 255)
                px[point_dict['s_x']+self.shape[0]-1, col] = (0, 0, 255)

            m.save(os.path.join(out_dir, str(i)+"_"+str(pos)+'_mark_y'+self.trans_made+'.png'))
