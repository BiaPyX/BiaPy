import numpy as np
import tensorflow as tf
import random
import os
from tqdm import tqdm
from skimage.io import imread
from util import img_to_onehot_encoding 
from scipy.ndimage import rotate
from scipy.ndimage.interpolation import shift
from PIL import Image
import imageio
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import imgaug as ia
from imgaug import parameters as iap
from skimage.io import imsave                                                   


class VoxelDataGenerator(tf.keras.utils.Sequence):
    """Custom ImageDataGenerator for 3D images.
    """

    def __init__(self, X, Y, random_subvolumes_in_DA=False, subvol_shape=None,
                 seed=42, shuffle_each_epoch=False, batch_size=32, da=True, 
                 da_prob=0.5, rotation90=False, rand_rot=False,
                 rnd_rot_range=(-180,180), shear=False, shear_range=(-20,20),
                 zoom=False, zoom_range=(0.8,1.2), shift=False,
                 shift_range=(0.1,0.2), flip=False, elastic=False, 
                 e_alpha=(240,250), e_sigma=25, e_mode='constant', g_blur=False,
                 g_sigma=(1.0,2.0), median_blur=False, mb_kernel=(3,7),
                 gamma_contrast=False, gc_gamma=(1.25,1.75), cutout=False,  
                 cout_nb_iterations=(1,3), cout_size=0.2,
                 cout_fill_mode='constant', dropout=False, drop_range=(0, 0.2),
                 n_classes=1, out_number=1, val=False, prob_map=None,
                 extra_data_factor=1):
        """ImageDataGenerator constructor. Based on transformations from 
           `imgaug <https://github.com/aleju/imgaug>`_ library. Here a brief
           description of each transformation parameter is made. Find a complete
           explanation of the library `documentation <https://imgaug.readthedocs.io/en/latest/index.html>`_. 
           
                                                                                
           Parameters
           ----------
           X : Numpy 5D array
               Data. E.g. ``(num_of_images, x, y, z, channels)``.

           Y : Numpy 5D array
               Mask data. E.g. ``(num_of_images, x, y, z, channels)``.

           random_subvolumes_in_DA : bool, optional
               To extract random subvolumes from the given data. If not, the 
               data must be 5D and is assumed that the subvolumes are prepared. 
    
           subvol_shape : 4D tuple of ints, optional
               Shape of the subvolume to be extracted randomly from the data. 
               E. g. ``(x, y, z, channels)``.
            
           seed : int, optional
               Seed for random functions.
                
           shuffle_each_epoch : bool, optional
               To shuffle data after each epoch.

           batch_size : int, optional
               Size of the batches.
            
           da : bool, optional
               To activate the data augmentation.
            
           da_prob : float, optional
               Probability of doing each transformation.
            
           rotation90 : bool, optional                                          
               To make square (90, 180,270) degree rotations.
        
           rand_rot : bool, optional                                            
               To make random degree range rotations.                  
           
           rand_rot_range : tuple of float, optional
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

           flip : bool, optional
               To activate flips (both horizontal and vertical).
        
           elastic : bool, optional
               To make elastic deformations.

           e_alpha : tuple of ints, optional
                Strength of the distortion field. E. g. ``(240, 250)``.
               
           e_sigma : int, optional
               Standard deviation of the gaussian kernel used to smooth the 
               distortion fields. 

           e_mode : str, optional
               Parameter that defines the handling of newly created pixels with 
               the elastic transformation. 
            
           g_blur : bool, optional
               To insert gaussian blur on the images.
        
           g_sigma : tuple of floats, optional
               Standard deviation of the gaussian kernel. E. g. ``(1.0, 2.0)``.

           median_blur : bool, optional                                      
               To blur an image by computing median values over neighbourhoods.
                                                                                
           mb_kernel : tuple of ints, optional                                  
               Median blur kernel size. E. g. ``(3, 7)``.                                   
           
           gamma_contrast : bool, optional
               To insert gamma constrast changes on images. 

           gc_gamma : tuple of floats, optional                                  
               Exponent for the contrast adjustment. Higher values darken the 
               image. E. g. ``(1.25, 1.75)``. 

           cutout : bool, optional                                      
               To fill one or more rectangular areas in an image using a fill 
               mode.

           cout_nb_iterations : tuple of ints, optional
               Range of number of areas to fill the image with. E. g. ``(1, 3)``. 

           cout_size : float, optional                         
               Size of the areas in % of the corresponding image size. Value 
               between ``0`` and ``1``.

           cout_fill_mode : str, optional                                      
               Parameter that defines the handling of newly created pixels with
               cutout.

           dropout : bool, optional
               To set a certain fraction of pixels in images to zero.
           
           drop_range : tuple of floats, optional
               Range to take a probability ``p`` to drop pixels. E.g. ``(0, 0.2)``
               will take a ``p`` folowing ``0<=p<=0.2`` and then drop ``p`` 
               percent of all pixels in the image (i.e. convert them to black 
               pixels).

           n_classes : int, optional
               Number of classes. If ``> 1`` one-hot encoding will be done on 
               the ground truth.

           out_number : int, optional                                               
               Number of output returned by the network. Used to produce same 
               number of ground truth data on each batch. 

           val : bool, optional
               Advice the generator that the volumes will be used to validate
               the model to not make random crops (as the validation data must
               be the same on each epoch). Valid when ``random_subvolumes_in_DA`` 
               is set.

           prob_map : 5D Numpy array, optional
               Probability map used to make random crops when
               ``random_subvolumes_in_DA`` is set.
            
           extra_data_factor : int, optional
               Factor to multiply the batches yielded in a epoch. It acts as if
               ``X`` and ``Y``` where concatenated ``extra_data_factor`` times.
        """

        if X.ndim != 5 or Y.ndim != 5:
            raise ValueError("X and Y must be a 5D Numpy array")
        if X.shape[:4] != Y.shape[:4]:                                          
            raise ValueError("The shape of X and Y must be the same. {} != {}"
                             .format(X.shape[:4], Y.shape[:4]))
        if random_subvolumes_in_DA:
            if subvol_shape is None:
                raise ValueError("'subvol_shape' must be provided when "
                                 "'random_subvolumes_in_DA is enabled")         
            if subvol_shape[0] > X.shape[1] or subvol_shape[1] > X.shape[2] or \
               subvol_shape[2] > X.shape[3]:
                raise ValueError("Given 'subvol_shape' is bigger than the data "
                                 "provided")

        if rotation90 and rand_rot:
            print("Warning: you selected double rotation type. Maybe you should"
                  " set only 'rand_rot'?")

        self.X = (X/255).astype(np.float32) if np.max(X) > 100 else X.astype(np.float32)
        self.X_c = self.X.shape[-1]
        self.X_z = self.X.shape[-2]
        self.Y = (Y/255).astype(np.uint8) if np.max(Y) > 100 else Y.astype(np.uint8)
        self.Y_c = self.Y.shape[-1]                                             
        self.Y_z = self.Y.shape[-2]
        self.n_classes = n_classes
        self.out_number = out_number
        self.channels = Y.shape[-1] 
        self.random_subvolumes_in_DA = random_subvolumes_in_DA
        self.seed = seed
        self.shuffle_each_epoch = shuffle_each_epoch
        self.da = da
        self.da_prob = da_prob
        self.flip = flip
        self.val = val
        self.batch_size = batch_size
        self.o_indexes = np.arange(len(self.X))
        if extra_data_factor > 1:
            self.extra_data_factor = extra_data_factor
            self.o_indexes = np.concatenate([self.o_indexes]*extra_data_factor)
        else:
            self.extra_data_factor = 1
        self.prob_map = prob_map
        if random_subvolumes_in_DA:
            self.shape = subvol_shape
        else:
            self.shape = X.shape[1:]
        self.total_batches_seen = 0

        self.da_options = []
        self.trans_made = ''
        if rotation90:
            self.da_options.append(iaa.Sometimes(da_prob, iaa.Rot90((1, 3))))
            self.trans_made += '_rot[90,180,270]'
        if rand_rot:
            self.da_options.append(iaa.Sometimes(da_prob, iaa.Affine(rotate=rand_rot_range)))
            self.trans_made += '_rrot'+str(rand_rot_range)
        if shear:
            self.da_options.append(iaa.Sometimes(da_prob, iaa.Affine(rotate=shear_range)))                                
            self.trans_made += '_shear'+str(shear_range)
        if zoom:
            self.da_options.append(iaa.Sometimes(da_prob, iaa.Affine(scale={"x": zoom_range, "y": zoom_range})))
            self.trans_made += '_zoom'+str(zoom_range)
        if shift:                                                               
            self.da_options.append(iaa.Sometimes(da_prob, iaa.Affine(translate_percent=shift_range)))
            self.trans_made += '_shift'+str(shift_range)                        
        if flip:
            self.da_options.append(iaa.Flipud(0.5))
            self.da_options.append(iaa.Fliplr(0.5))                                  
            self.trans_made += '_flip'
        if elastic:
            self.da_options.append(iaa.Sometimes(da_prob,iaa.ElasticTransformation(alpha=e_alpha, sigma=e_sigma, mode=e_mode)))
            self.trans_made += '_elastic'+str(e_alpha)+'+'+str(e_sigma)+'+'+str(e_mode)
        if g_blur:
            self.da_options.append(iaa.Sometimes(da_prob,iaa.GaussianBlur(g_sigma)))
            self.trans_made += '_gblur'+str(g_sigma)
        if gamma_contrast:
            self.da_options.append(iaa.Sometimes(da_prob,iaa.GammaContrast(gc_gamma)))
            self.trans_made += '_gcontrast'+str(gc_gamma)
        if median_blur:
            self.da_options.append(iaa.Sometimes(da_prob,iaa.MedianBlur(k=mb_kernel)))
            self.trans_made += '_mblur'+str(mb_kernel)
        if cutout:
            self.da_options.append(iaa.Sometimes(da_prob, iaa.Cutout(nb_iterations=cout_nb_iterations, size=cout_size, fill_mode=cout_fill_mode, squared=False)))       
            self.trans_made += '_cout'+str(cout_nb_iterations)+'+'+str(cout_size)+'+'+str(cout_fill_mode)
        if dropout: 
            self.da_options.append(iaa.Sometimes(da_prob, iaa.Dropout(p=drop_range)))
            self.trans_made += '_drop'+str(drop_range)

        self.trans_made = self.trans_made.replace(" ", "")
        self.seq = iaa.Sequential(self.da_options)
        ia.seed(seed)
        self.on_epoch_end()

    def __len__(self):
        """Defines the length of the generator"""
        return int(np.ceil(self.X.shape[0]*self.extra_data_factor/self.batch_size))

    def __draw_grid(self, im, grid_width=50, v=1):
        """Draw grid of the specified size on an image. 
           
           Parameters
           ----------                                                                
           im : 4D Numpy array
               Image to be modified. E. g. ``(x, y, z, channels)``
                
           grid_width : int, optional
               Grid's width. 

           v : int, optional
               Value to create the grid with.
        """

        for k in range(0, im.shape[2]):
            for i in range(0, im.shape[0], grid_width):
                if im.shape[-1] == 1:
                    im[i,:,k] = v
                else:
                    im[i,:,k] = [v]*im.shape[-1]
            for j in range(0, im.shape[1], grid_width):
                if im.shape[-1] == 1:
                    im[:,j,k] = v
                else:
                    im[:,j,k] = [v]*im.shape[-1]

    def __getitem__(self, index):
        """Generation of one batch of data. 
           
           Parameters
           ----------
           index : int
               Batch index counter.
            
           Returns
           -------  
           batch_x : 5D Numpy array
               Corresponding X elements of the batch.
               E.g. ``(batch_size_value, x, y, z, channels)``.

           batch_y : 5D Numpy array
               Corresponding Y elements of the batch.
               E.g. ``(batch_size_value, x, y, z, channels)``.
        """

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_x = np.zeros((len(indexes), ) + self.shape)
        batch_y = np.zeros((len(indexes), ) + self.shape[:3]+(self.channels,), 
                           dtype=np.uint8)

        for i, j in zip(range(len(indexes)), indexes):
            if self.random_subvolumes_in_DA:
                batch_x[i], batch_y[i] = random_3D_crop(
                    self.X[0], self.Y[0], self.shape, self.val, 
                    vol_prob=(self.prob_map[0] if self.prob_map is not None else None))
            else:
                batch_x[i] = np.copy(self.X[j])
                batch_y[i] = np.copy(self.Y[j])

            if self.da:
                batch_x[i], batch_y[i] = self.apply_transform(batch_x[i], batch_y[i])

        if self.n_classes > 1 and (self.n_classes != self.channels):
            batch_y_ = np.zeros((len(indexes), ) + self.shape[:3] + (self.n_classes,))
            for i in range(len(indexes)):
                batch_y_[i] = np.asarray(img_to_onehot_encoding(batch_y[i]))

            batch_y = batch_y_

        self.total_batches_seen += 1
                                                                                        
        if self.out_number == 1:                                                
            return batch_x, batch_y                                             
        else:                                                                   
            return ([batch_x], [batch_y]*self.out_number)

    def on_epoch_end(self):
        """Updates indexes after each epoch."""

        ia.seed(self.seed + self.total_batches_seen)
        self.indexes = self.o_indexes
        if self.shuffle_each_epoch:
            random.Random(self.seed + self.total_batches_seen).shuffle(self.indexes)

    def apply_transform(self, image, mask, grid=False):
        """Transform the input image and its mask at the same time with one of
           the selected choices based on a probability.
    
           Parameters
           ----------
           image : 4D Numpy array
               Image to transform. E.g. ``(x, y, z, channels)``.

           mask : 4D Numpy array
               Mask to transform. E.g. ``(x, y, z, channels)``.
    
           Returns
           -------
           trans_image : 4D Numpy array
               Transformed image. E.g. ``(x, y, z, channels)``.

           trans_mask : 4D Numpy array
               Transformed image mask. E.g. ``(x, y, z, channels)``.
        """

        # Apply flips in z as imgaug can not do it 
        prob = random.uniform(0, 1)
        if self.flip and prob < self.da_prob:
            l_image = []
            l_mask = []
            for i in range(image.shape[-1]):                                
                l_image.append(np.expand_dims(np.flip(image[...,i], 2), -1))
            for i in range(mask.shape[-1]):
                l_mask.append(np.expand_dims(np.flip(mask[...,i], 2), -1))
            image = np.concatenate(l_image, axis=-1)
            mask = np.concatenate(l_mask, axis=-1)

        # Reshape 3D volumes to 2D image type with multiple channels to pass 
        # through imgaug lib
        o_img_shape = image.shape 
        o_mask_shape = mask.shape 
        image = image.reshape(image.shape[:2]+(self.X_z*self.X_c, ))
        mask = mask.reshape(mask.shape[:2]+(self.Y_z*self.Y_c, ))
      
        # Apply transformations to the volume and its mask
        segmap = SegmentationMapsOnImage(mask, shape=mask.shape)            
        image, vol_mask = self.seq(image=image, segmentation_maps=segmap)   
        mask = vol_mask.get_arr()
        
        # Recover the original shape 
        image = image.reshape(o_img_shape)
        mask = mask.reshape(o_mask_shape)

        return image, mask

    def get_transformed_samples(self, num_examples, random_images=True, 
                                save_to_dir=True, out_dir='aug_3d', train=False):
        """Apply selected transformations to a defined number of images from
           the dataset. 
            
           Parameters
           ----------
           num_examples : int
               Number of examples to generate.
            
           random_images : bool, optional
               Randomly select images from the dataset. If False the examples
               will be generated from the start of the dataset. 

           save_to_dir : bool, optional
               Save the images generated. The purpose of this variable is to
               check the images generated by data augmentation.

           out_dir : str, optional
               Name of the folder where the examples will be stored. 

           train : bool, optional
               To avoid drawing a grid on the generated images. This should be
               set when the samples will be used for training.
        """    

        if random_images == False and num_examples > self.X.shape[0]:   
            num_examples = self.X.shape[0]
            print("WARNING: More samples requested than the ones available. "
                  "'num_examples' fixed to {}".format(num_examples))
            
        sample_x = np.zeros((num_examples, ) + self.shape, dtype=np.float32)
        sample_y = np.zeros((num_examples, ) + self.shape[:3]+(self.channels,),
                            dtype=np.uint8)

        # Generate the examples 
        print("0) Creating samples of data augmentation . . .")
        for i in tqdm(range(num_examples)):
            ia.seed(i)
            if random_images or self.random_subvolumes_in_DA:
                pos = random.randint(0,self.X.shape[0]-1) 
            else:
                pos = i

            if self.random_subvolumes_in_DA:
                vol, vol_mask, ox, oy, oz,\
                s_x, s_y, s_z = random_3D_crop(
                    self.X[pos], self.Y[pos], self.shape, self.val,
                    draw_prob_map_points=True,
                    vol_prob=(self.prob_map[pos] if self.prob_map is not None else None))
            else:
                vol = np.copy(self.X[pos])
                vol_mask = np.copy(self.Y[pos])

            if not self.da:
                sample_x[i] = vol
                sample_y[i] = vol_mask
                self.trans_made = ''
            else:
                if not train:
                    self.__draw_grid(vol)
                    self.__draw_grid(vol_mask)

                sample_x[i], sample_y[i] = self.apply_transform(vol, vol_mask)

            # Save transformed 3D volumes 
            if save_to_dir:
                os.makedirs(out_dir, exist_ok=True)
                # Original image/mask
                f = os.path.join(out_dir, "orig_x_"+str(pos)+self.trans_made+'.tiff')
                aux = self.X[pos].copy()
                self.__draw_grid(aux)
                aux = np.expand_dims((np.transpose(aux, (2,0,1,3))*255).astype(np.uint8), 1)
                imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'})
                f = os.path.join(out_dir, "orig_y_"+str(pos)+self.trans_made+'.tiff')
                aux = self.Y[pos].copy()
                self.__draw_grid(aux)
                aux = np.expand_dims((np.transpose(aux, (2,0,1,3))*255).astype(np.uint8), 1)
                imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'})
                # Transformed
                f = os.path.join(out_dir, "x_aug_"+str(pos)+self.trans_made+'.tiff')
                aux = np.expand_dims((np.transpose(sample_x[i], (2,0,1,3))*255).astype(np.uint8), 1)
                imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'})
                # Mask
                f = os.path.join(out_dir, "y_aug_"+str(pos)+self.trans_made+'.tiff')
                aux = np.expand_dims((np.transpose(sample_y[i], (2,0,1,3))*255).astype(np.uint8), 1)
                imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'})

                # Save the original images with a red point and a blue square 
                # that represents the point selected with the probability map 
                # and the random volume extracted from the original data
                if self.random_subvolumes_in_DA and self.prob_map is not None and i == 0:
                    rc_out_dir = os.path.join(out_dir, 'rd_crop' + str(pos))
                    os.makedirs(rc_out_dir, exist_ok=True)

                    print("The selected point on the random crop was [{},{},{}]"
                          .format(ox,oy,oz))

                    d = len(str(self.X[pos].shape[2]))
                    for i in range(self.X[pos].shape[2]):
                        im = Image.fromarray((self.X[pos,:,:,i,0]).astype(np.uint8)) 
                        im = im.convert('RGB')                                                  
                        px = im.load()                                                          
                        mask = Image.fromarray((self.Y[pos,:,:,i,0]).astype(np.uint8))
                        mask = mask.convert('RGB')
                        py = mask.load()
                       
                        if i == oz:
                            # Paint the selected point in red
                            p_size=6
                            for row in range(oy-p_size,oy+p_size):
                                for col in range(ox-p_size,ox+p_size): 
                                    if col >= 0 and col < self.X[pos].shape[0] and \
                                       row >= 0 and row < self.X[pos].shape[1]:
                                       px[row, col] = (255, 0, 0) 
                                       py[row, col] = (255, 0, 0) 
                   
                        if i >= s_z and i < s_z+self.shape[2]: 
                            # Paint a blue square that represents the crop made 
                            for col in range(s_x, s_x+self.shape[0]):
                                px[s_y, col] = (0, 0, 255)
                                px[s_y+self.shape[0]-1, col] = (0, 0, 255)
                                py[s_y, col] = (0, 0, 255)
                                py[s_y+self.shape[0]-1, col] = (0, 0, 255)
                            for row in range(s_y, s_y+self.shape[1]):                    
                                px[row, s_x] = (0, 0, 255)
                                px[row, s_x+self.shape[1]-1] = (0, 0, 255)
                                py[row, s_x] = (0, 0, 255)
                                py[row, s_x+self.shape[1]-1] = (0, 0, 255)
                         
                        im.save(os.path.join(
                                    rc_out_dir,'rc_x_'+str(i).zfill(d)+'.png'))
                        mask.save(os.path.join(
                                      rc_out_dir,'rc_y_'+str(i).zfill(d)+'.png'))          
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
