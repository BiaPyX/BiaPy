import numpy as np
import tensorflow as tf
import random
import math
import os
from tqdm import tqdm
from skimage.io import imread
from util import array_to_img, img_to_array, save_img
from scipy.ndimage import rotate
from scipy.ndimage.interpolation import shift
from data_manipulation import img_to_onehot_encoding
from PIL import Image
import imageio
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import imgaug as ia
from imgaug import parameters as iap


class VoxelDataGenerator(tf.keras.utils.Sequence):
    """Custom ImageDataGenerator for 3D images.
    """

    def __init__(self, X, Y, random_subvolumes_in_DA=False, subvol_shape=None,
                 seed=42, shuffle_each_epoch=False, batch_size=32, da=True, 
                 hist_eq=False, flip=False, rotation=False, elastic=False,
                 g_blur=False, gamma_contrast=False, softmax_out=False, val=False, 
                 prob_map=None):
        """ImageDataGenerator constructor. Based on transformations from 
           https://github.com/aleju/imgaug.
                                                                                
       Args:                                                                    
            X (Numpy 5D array): data. E.g. (image_number, z, x, y, channels).

            Y (Numpy 5D array): mask data. E.g. (image_number, z, x, y, channels).

            random_subvolumes_in_DA (bool, optional): flag to extract random 
            subvolumes from the given data. If not, the data must be 5D and is 
            assumed that the subvolumes are prepared. 
    
            subvol_shape (4D tuple of ints, optional): shape of the subvolume to
            be extracted randomly from the data. E. g. (z, x, y, channels).
            
            seed (int, optional): seed for random functions.
                
            shuffle_each_epoch (bool, optional): flag to shuffle data after each 
            epoch.

            batch_size (int, optional): size of the batches.
            
            da (bool, optional): flag to activate the data augmentation.
            
            hist_eq (bool, optional): flag to make histogram equalization on 
            images.

            flip (bool, optional): flag to activate flips.
        
            rotation (bool, optional): flag to make 90º, 180º or 270º rotations.

            elastic (bool, optional): flag to make elastic deformations.
            
            g_blur (bool, optional): flag to insert gaussian blur on the images.

            gamma_contrast (bool, optional): flag to insert gamma constrast 
            changes on images. 

            softmax_out (bool, optional): flag to advice that the output of the
            network has in the last layer a softmax activation or one channel
            per class. If so one-hot encoded will be done on the ground truth.

            val (bool, optional): advice the generator that the volumes will be
            used to validate the model to not make random crops (as the val. 
            data must be the same on each epoch). Valid when 
            random_subvolumes_in_DA is set.

            prob_map (5D Numpy array, optional): probability map used to make
            random crops when random_subvolumes_in_DA is set.
        """

        if X.shape != Y.shape:
            raise ValueError("The shape of X and Y must be the same")
        if X.ndim != 5 or Y.ndim != 5:
            raise ValueError("X and Y must be a 5D Numpy array")
        if random_subvolumes_in_DA:
            if subvol_shape is None:
                raise ValueError("'subvol_shape' must be provided when "
                                 "'random_subvolumes_in_DA is enabled")         
            if subvol_shape[0] > X.shape[1] or subvol_shape[1] > X.shape[2] or \
               subvol_shape[2] > X.shape[3]:
                raise ValueError("Given 'subvol_shape' is bigger than the data "
                                 "provided")
        self.divide = True if np.max(X) > 1 else False
        self.X = (X).astype(np.uint8)
        self.Y = (Y).astype(np.uint8)
        self.softmax_out = softmax_out
        self.random_subvolumes_in_DA = random_subvolumes_in_DA
        self.seed = seed
        self.shuffle_each_epoch = shuffle_each_epoch
        self.da = da
        self.val = val
        self.batch_size = batch_size
        self.total_batches_seen = 0
        self.prob_map = prob_map
        if random_subvolumes_in_DA:
            self.shape = subvol_shape
        else:
            self.shape = X.shape[1:]

        self.da_options = []
        self.da_forced = []
        self.da_names = []

        # da_options will be used to create the DA pipeline and da_forced to 
        # visualize the transformations choosen by the generator when function
        # get_transformed_samples is invoked
        if hist_eq:
            self.da_options.append(iaa.HistogramEqualization())

            self.da_forced.append(iaa.HistogramEqualization())
            self.da_names.append("hist_eq")

        if flip:
            self.da_options.append(iaa.OneOf([iaa.Fliplr(0.5),iaa.Flipud(0.5)]))

            self.da_names.append("hflips")
            self.da_forced.append(iaa.Fliplr(1))
            self.da_names.append("vflips")
            self.da_forced.append(iaa.Flipud(1))
        
        if rotation:
            self.da_options.append(iaa.Rot90(0,3))

            self.da_names.append("rot90")
            self.da_forced.append(iaa.Rot90(1))
            self.da_names.append("rot180")
            self.da_forced.append(iaa.Rot90(2))
            self.da_names.append("rot270")
            self.da_forced.append(iaa.Rot90(3))

        if elastic:
            self.da_options.append(iaa.Sometimes(0.5,iaa.ElasticTransformation(alpha=(0, 5.0), sigma=0.25)))

            self.da_names.append("elastic")
            self.da_forced.append(iaa.ElasticTransformation(alpha=(0, 5.0), sigma=4))

        if g_blur:
            self.da_options.append(iaa.Sometimes(0.5,iaa.GaussianBlur(sigma=(0.0, 2.0))))
        
            self.da_names.append("g_blur")
            self.da_forced.append(iaa.GaussianBlur(sigma=(0.0, 2.0)))
            
        if gamma_contrast:
            self.da_options.append(iaa.Sometimes(0.5,iaa.GammaContrast((0.5, 2.0))))

            self.da_names.append("gamma_contrast")
            self.da_forced.append(iaa.GammaContrast((0.5, 2.0)))

        self.seq = iaa.Sequential(self.da_options, random_order=True)

        self.on_epoch_end()

    def __len__(self):
        """Defines the number of batches per epoch."""
    
        return int(np.ceil(self.X.shape[0]/self.batch_size))

    def __getitem__(self, index):
        """Generation of one batch of data. 
           Args:
                index (int): batch index counter.
            
           Returns:
                batch_x (Numpy array): corresponding X elements of the batch.
                E.g. (batch_size_value, x, y, z, channels).

                batch_y (Numpy array): corresponding Y elements of the batch.
                E.g. (batch_size_value, x, y, z, channels).
        """

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_x = np.zeros((len(indexes), ) + self.shape, dtype=np.uint8)
        batch_y = np.zeros((len(indexes), ) + self.shape, dtype=np.uint8)

        for i, j in zip(range(len(indexes)), indexes):
            if self.random_subvolumes_in_DA:
                batch_x[i], batch_y[i] = random_3D_crop(
                    self.X[j], self.Y[j], self.shape, self.val, 
                    vol_prob=(self.prob_map[j] if self.prob_map is not None else None))
            else:
                batch_x[i] = np.copy(self.X[j])
                batch_y[i] = np.copy(self.Y[j])

            if self.da:
                segmap = SegmentationMapsOnImage(batch_y[i,...,0], shape=batch_x[i,...,0].shape)
                vol, vol_mask = self.seq(image=batch_x[i,...,0], segmentation_maps=segmap)
                batch_x[i] = np.expand_dims(vol, axis=-1)
                batch_y[i] = np.expand_dims(vol_mask.get_arr(), axis=-1)

        if self.divide:
            batch_x = batch_x/255
            batch_y = batch_y/255

        if self.softmax_out:
            batch_y_ = np.zeros((len(indexes), ) + self.shape[:3] + (2,))
            for i in range(len(indexes)):
                batch_y_[i] = np.asarray(img_to_onehot_encoding(batch_y[i]))

            batch_y = batch_y_

        self.total_batches_seen += 1
        return batch_x, batch_y    

    def on_epoch_end(self):
        """Updates indexes after each epoch."""

        self.indexes = np.arange(self.X.shape[0])
        if self.shuffle_each_epoch:
            random.Random(self.seed + self.total_batches_seen).shuffle(self.indexes)


    def get_transformed_samples(self, num_examples, random_images=True, 
                                save_to_dir=True, out_dir='aug_3d'):
        """Apply selected transformations to a defined number of images from
           the dataset. 
            
           Args:
                num_examples (int): number of examples to generate.
            
                random_images (bool, optional): randomly select images from the
                dataset. If False the examples will be generated from the start
                of the dataset. 

                save_to_dir (bool, optional): save the images generated. The 
                purpose of this variable is to check the images generated by 
                data augmentation.

                out_dir (str, optional): name of the folder where the
                examples will be stored. 
        """    

        sample_x = np.zeros((num_examples, ) + self.shape, dtype=np.uint8)
        sample_y = np.zeros((num_examples, ) + self.shape, dtype=np.uint8)

        # Generate the examples 
        print("0) Creating samples of data augmentation . . .")
        for i in tqdm(range(num_examples)):
            if random_images:
                pos = random.randint(0,self.X.shape[0]-1) 
            else:
                pos = i

            if self.random_subvolumes_in_DA:
                vol, vol_mask, oz, ox, oy,\
                s_z, s_x, s_y= random_3D_crop(
                    self.X[pos], self.Y[pos], self.shape, self.val,
                    draw_prob_map_points=True,
                    vol_prob=(self.prob_map[pos] if self.prob_map is not None else None))
            else:
                vol = np.copy(self.X[pos])
                vol_mask = np.copy(self.Y[pos])

            if not self.da:
                sample_x[i] = vol
                sample_y[i] = vol_mask
            else:
                segmap = SegmentationMapsOnImage(
                    vol_mask[...,0], shape=vol[...,0].shape)
                vol, vol_mask = self.seq(image=vol[...,0], segmentation_maps=segmap) 
                sample_x[i] = np.expand_dims(vol, axis=-1)
                sample_y[i] = np.expand_dims(vol_mask.get_arr(), axis=-1)

            # Save transformed 3D volumes 
            if save_to_dir:
                os.makedirs(out_dir, exist_ok=True)
                d = len(str(self.X.shape[3]))
                for x in range(len(self.da_names)):
                    transform = iaa.Sequential(self.da_forced[x])
                    img, smap = transform(image=self.X[pos,...,0], segmentation_maps=segmap)
                    for z in range(self.X.shape[3]):
                        cells = []
                        cells.append(self.X[pos,:,:,z])
                        cells.append(self.Y[pos,:,:,z])
                        cells.append(np.expand_dims(img[:,:,z], axis=-1))
                        cells.append(np.expand_dims((smap.get_arr()[...,z]).astype(np.uint8), axis=-1))
                        grid_image = ia.draw_grid(cells, cols=4)
                        name = "t_"+str(pos).zfill(d)+"_"+str(z).zfill(d)+"_"+self.da_names[x]+".png"
                        imageio.imwrite(os.path.join(out_dir,name), grid_image)

                # Save the original images with a red point and a blue square 
                # that represents the point selected with the probability map 
                # and the random volume extracted from the original data
                if self.random_subvolumes_in_DA and self.prob_map is not None:
                    rc_out_dir = os.path.join(out_dir, 'rd_crop' + str(pos))
                    os.makedirs(rc_out_dir, exist_ok=True)

                    print("The selected point on the random crop was [{},{},{}]"
                          .format(oz,ox,oy))

                    for i in range(self.X[pos].shape[0]):
                        im = Image.fromarray((self.X[pos,i,...,0]*255).astype(np.uint8)) 
                        im = im.convert('RGB')                                                  
                        px = im.load()                                                          
                        mask = Image.fromarray((self.Y[pos,i,...,0]*255).astype(np.uint8))
                        mask = mask.convert('RGB')
                        py = mask.load()
                       
                        if i == oz:
                            # Paint the selected point in red
                            p_size=6
                            for row in range(oy-p_size,oy+p_size):
                                for col in range(ox-p_size,ox+p_size): 
                                    if col >= 0 and col < self.X[pos].shape[1] and \
                                       row >= 0 and row < self.X[pos].shape[2]:
                                       px[row, col] = (255, 0, 0) 
                                       py[row, col] = (255, 0, 0) 
                   
                        if i >= s_z and i < s_z+self.shape[0]: 
                            # Paint a blue square that represents the crop made 
                            for col in range(s_x, s_x+self.shape[1]):
                                px[s_y, col] = (0, 0, 255)
                                px[s_y+self.shape[1]-1, col] = (0, 0, 255)
                                py[s_y, col] = (0, 0, 255)
                                py[s_y+self.shape[1]-1, col] = (0, 0, 255)
                            for row in range(s_y, s_y+self.shape[2]):                    
                                px[row, s_x] = (0, 0, 255)
                                px[row, s_x+self.shape[2]-1] = (0, 0, 255)
                                py[row, s_x] = (0, 0, 255)
                                py[row, s_x+self.shape[2]-1] = (0, 0, 255)
                         
                        im.save(os.path.join(
                                    rc_out_dir, 'rc_x_' + str(i) + '.png'))
                        mask.save(os.path.join(
                                      rc_out_dir, 'rc_y_' + str(i) + '.png'))          
        return sample_x, sample_y


def random_3D_crop(vol, vol_mask, random_crop_size, val=False, vol_prob=None, 
                   weights_on_data=False, weight_map=None,
                   draw_prob_map_points=False):
    """Random 3D crop """

    rows, cols, deep = vol.shape[0], vol.shape[1], vol.shape[2]
    dx, dy, dz, c = random_crop_size
    if val:
        x = 0
        y = 0
        z = 0
        ox = 0
        oy = 0
        oz = 0
    else:
        if vol_prob is not None:
            prob = vol_prob.ravel() 
            
            # Generate the random coordinates based on the distribution
            choices = np.prod(vol_prob.shape)
            index = np.random.choice(choices, size=1, p=prob)
            coordinates = np.unravel_index(index, shape=vol_prob.shape)
            z = int(coordinates[0])
            x = int(coordinates[1])
            y = int(coordinates[2])
            oz = int(coordinates[0])
            ox = int(coordinates[1])
            oy = int(coordinates[2])
            
            # Adjust the coordinates to be the origin of the crop and control to
            # not be out of the volume
            if x < int(random_crop_size[0]/2):
                x = 0
            elif x > vol.shape[0] - int(random_crop_size[0]/2):
                x = vol.shape[0] - random_crop_size[0]
            else: 
                x -= int(random_crop_size[0]/2)
            
            if y < int(random_crop_size[1]/2):
                y = 0
            elif y > vol.shape[1] - int(random_crop_size[1]/2):
                y = vol.shape[1] - random_crop_size[1]
            else:
                y -= int(random_crop_size[1]/2)

            if z < int(random_crop_size[2]/2):
                z = 0
            elif z > vol.shape[2] - int(random_crop_size[2]/2):
                z = vol.shape[2] - random_crop_size[2]
            else:
                z -= int(random_crop_size[2]/2)
        else:
            ox = 0
            oy = 0
            oz = 0
            x = np.random.randint(0, rows - dx + 1)                                
            y = np.random.randint(0, cols - dy + 1)
            z = np.random.randint(0, deep - dz + 1)

    if draw_prob_map_points:
        return vol[x:(x+dx), y:(y+dy), z:(z+dz), :], \
               vol_mask[x:(x+dx), y:(y+dy), z:(z+dz), :], ox, oy, oz, x, y, z
    else:
        if weights_on_data:
            return vol[x:(x+dx), y:(y+dy), z:(z+dz), :], \
                   vol_mask[x:(x+dx), y:(y+dy), z:(z+dz), :],\
                   weight_map[x:(x+dx), y:(y+dy), z:(z+dz), :]         
        else:
            return vol[x:(x+dx), y:(y+dy), z:(z+dz), :], \
                   vol_mask[x:(x+dx), y:(y+dy), z:(z+dz), :]
