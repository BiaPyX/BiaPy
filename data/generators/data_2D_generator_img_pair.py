import numpy as np
import os
from skimage.io import imread, imsave

from data.pre_processing import normalize, norm_range01
from data.generators.base_data_generator import BaseDataGenerator

class PairImageDataGenerator(BaseDataGenerator):
    """Custom 2D data generator based on `imgaug <https://github.com/aleju/imgaug-doc>`_
       and our own `augmentors.py <https://github.com/danifranco/BiaPy/blob/master/generators/augmentors.py>`_
       transformations. This generator will yield two images.

       Based on `microDL <https://github.com/czbiohub/microDL>`_ and
       `Shervine's blog <https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly>`_.

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
           train_generator = PairImageDataGenerator(**data_gen_args)
           val_generator = PairImageDataGenerator(**data_gen_val_args)


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
           train_generator = PairImageDataGenerator(**data_gen_args)
           val_generator = PairImageDataGenerator(**data_gen_val_args)
    """
    def __init__(self, random_crop_scale=1, **kwars):
        super().__init__(**kwars)
        self.Y_shape = (self.shape[0]*random_crop_scale, self.shape[1]*random_crop_scale, self.shape[2]) 

        # Normalize Y as in base data generator is was consider a mask and the normalization is different
        if self.in_memory:
            if self.X_norm['type'] == 'div':
                self.Y, _ = norm_range01(self.Y)
            elif self.X_norm['type'] == 'custom':
                self.Y = normalize(self.Y, self.X_norm['mean'], self.X_norm['std'])
                
    def load_sample(self, idx):
        """Load one data sample given its corresponding index."""

        # Choose the data source
        if self.in_memory:
            imgA = self.X[idx]
            imgB = self.Y[idx]
            if imgA.ndim == 4: 
                imgA = imgA[0]
            if imgB.ndim == 4: 
                imgB = imgB[0]
        else:
            if self.data_paths[idx].endswith('.npy'):
                imgA = np.load(os.path.join(self.paths[0], self.data_paths[idx]))
                imgB = np.load(os.path.join(self.paths[1], self.data_mask_path[idx]))
            else:
                imgA = imread(os.path.join(self.paths[0], self.data_paths[idx]))
                imgB = imread(os.path.join(self.paths[1], self.data_mask_path[idx]))
            imgA = np.squeeze(imgA)
            imgB = np.squeeze(imgB)

            # Normalization
            if self.X_norm:
                if self.X_norm['type'] == 'div':
                    imgA, _ = norm_range01(imgA)
                    imgB, _ = norm_range01(imgB)
                elif self.X_norm['type'] == 'custom':
                    imgA = normalize(imgA, self.X_norm['mean'], self.X_norm['std'])
                    imgB = normalize(imgB, self.X_norm['mean'], self.X_norm['std'])

        imgA, imgB = self.ensure_shape(imgA, imgB)

        return imgA, imgB

    def ensure_shape(self, imgA, imgB):
        # Shape adjustment
        if imgA.ndim == 2:
            imgA = np.expand_dims(imgA, -1)
        else:
            if imgA.shape[0] <= 3: imgA = imgA.transpose((1,2,0))
            
        if imgB.ndim == 2:
            imgB = np.expand_dims(imgB, -1)
        else:
            if imgB.shape[0] <= 3: imgB = imgB.transpose((1,2,0))

        return imgA, imgB

    def apply_imgaug(self, imgA, imgB, heat):
        # Apply transformations to both images
        augseq_det = self.seq.to_deterministic()
        imgA = augseq_det.augment_image(imgA)
        imgB = augseq_det.augment_image(imgB)
        return imgA, imgB, None

    def save_aug_samples(self, img, mask, orig_images, i, pos, out_dir, draw_grid, point_dict):
        if draw_grid:
            self.draw_grid(orig_images['o_x'])
            self.draw_grid(orig_images['o_y'])
            
        os.makedirs(out_dir, exist_ok=True)
        f = os.path.join(out_dir,str(i)+"_"+str(pos)+'_orig_x'+self.trans_made+".tif")
        aux = np.expand_dims(np.expand_dims(orig_images['o_x'].transpose((2,0,1)), -1), 0).astype(np.float32)
        imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False)
        f = os.path.join(out_dir,str(i)+"_"+str(pos)+'_orig_y'+self.trans_made+".tif")
        aux = np.expand_dims(np.expand_dims(orig_images['o_y'].transpose((2,0,1)), -1), 0).astype(np.float32)
        imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False)

        # Save transformed images
        f = os.path.join(out_dir,str(i)+"_"+str(pos)+'_x'+self.trans_made+".tif")
        aux = np.expand_dims(np.expand_dims(img.transpose((2,0,1)), -1), 0).astype(np.float32)
        imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False)
        f = os.path.join(out_dir,str(i)+"_"+str(pos)+'_y'+self.trans_made+".tif")
        aux = np.expand_dims(np.expand_dims(mask.transpose((2,0,1)), -1), 0).astype(np.float32)
        imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False)