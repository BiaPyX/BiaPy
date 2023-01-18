import numpy as np
import os
from skimage.io import imsave

from data.generators.single_base_data_generator import SingleBaseDataGenerator

class Single2DImageDataGenerator(SingleBaseDataGenerator):
    """Custom 2D data generator based on `imgaug <https://github.com/aleju/imgaug-doc>`_
       and our own `augmentors.py <https://github.com/danifranco/BiaPy/blob/master/generators/augmentors.py>`_
       transformations. This generator will yield an image and its corresponding class.

       Based on `microDL <https://github.com/czbiohub/microDL>`_ and
       `Shervine's blog <https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly>`_.
    """
    def __init__(self, **kwars):
        super().__init__(**kwars)

    def ensure_shape(self, img):
        # Shape adjustment
        if img.ndim == 2:
            img = np.expand_dims(img, -1)
        else:
            if img.shape[0] <= 3: img = img.transpose((1,2,0))   
        return img 

    def save_aug_samples(self, img, orig_image, i, pos, out_dir, draw_grid):
        if draw_grid:
            self.draw_grid(orig_image['o_x'])
            
        os.makedirs(out_dir, exist_ok=True)
        f = os.path.join(out_dir,str(i)+"_"+str(pos)+'_orig_x'+self.trans_made+".tif")
        aux = np.expand_dims(np.expand_dims(orig_image['o_x'].transpose((2,0,1)), -1), 0).astype(np.float32)
        imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False, compression=('zlib', 1))

        # Save transformed images
        f = os.path.join(out_dir,str(i)+"_"+str(pos)+'_x'+self.trans_made+".tif")
        aux = np.expand_dims(np.expand_dims(img.transpose((2,0,1)), -1), 0).astype(np.float32)
        imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False, compression=('zlib', 1))
