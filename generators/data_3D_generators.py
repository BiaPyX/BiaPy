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
                 shift_range=0, flip=False, rotation=False, elastic=False,
                 g_blur=False, gamma_contrast=False, n_classes=1, out_number=1,
                 val=False, prob_map=None, extra_data_factor=1):
        """ImageDataGenerator constructor. Based on transformations from 
           https://github.com/aleju/imgaug.
                                                                                
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
            
           shift_range : float, optional
               Range to make a shift. It must be a number between ``0`` and ``1``. 

           flip : bool, optional
               To activate flips.
        
           rotation : bool, optional
               To make ``[-180, 180]`` degree range rotations.

           elastic : bool, optional
               To make elastic deformations.
            
           g_blur : bool, optional
               To insert gaussian blur on the images.

           gamma_contrast : bool, optional
               To insert gamma constrast changes on images. 

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

        self.X = (X/255).astype(np.float32) if np.max(X) > 250 else X.astype(np.float32)
        self.Y = (Y/255).astype(np.uint8) if np.max(Y) > 250 else Y.astype(np.uint8)
        self.rgb = True if self.X.shape[-1] != 1 else False
        self.n_classes = n_classes
        self.out_number = out_number
        self.channels = Y.shape[-1] 
        self.random_subvolumes_in_DA = random_subvolumes_in_DA
        self.seed = seed
        self.shuffle_each_epoch = shuffle_each_epoch
        self.da = da
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
        self.flip = flip
        self.rotation = rotation
        self.shift_range = shift_range 
        self.total_batches_seen = 0
        self.imgaug = False

        self.da_options = []
        self.trans_made = ''
        if elastic:
            self.da_options.append(iaa.Sometimes(0.5,iaa.ElasticTransformation(alpha=(240, 250), sigma=25, mode="reflect")))
            self.trans_made += '_elastic'
            self.imgaug = True
        if g_blur:
            self.da_options.append(iaa.Sometimes(0.5,iaa.GaussianBlur(sigma=(1.0, 2.0))))
            self.trans_made += '_gblur'
            self.imgaug = True
        if gamma_contrast:
            self.da_options.append(iaa.Sometimes(0.5,iaa.GammaContrast((1.25, 1.75))))
            self.trans_made += '_gcontrast'
            self.imgaug = True
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
                batch_x[i], batch_y[i], _ = self.apply_transform(batch_x[i], batch_y[i])

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
        trans_made = ''

        # IMG
        if not self.rgb:
            image = image[...,0]
        else:
            l_image = []
            for i in range(image.shape[-1]):
                l_image.append(image[...,i])
        # MASK
        m_channel = mask.shape[-1]
        if m_channel == 1:
            mask = mask[...,0]
        else:
            l_mask = []
            for i in range(mask.shape[-1]):
                l_mask.append(mask[...,i])
        
        # [0-0.25): x axis flip
        # [0.25-0.5): y axis flip
        # [0.5-0.75): z axis flip
        # [0.75-1]: nothing
        #
        # x axis flip
        prob = random.uniform(0, 1)
        if self.flip and prob < 0.25:   
            # IMG
            if not self.rgb:
                image = np.flip(image, 0)
            else:
                for i in range(image.shape[-1]):
                    l_image[i] = np.flip(l_image[i], 0)
            # MASK
            if m_channel == 1:
                mask = np.flip(mask, 0)
            else:
                for i in range(mask.shape[-1]):
                    l_mask[i] = np.flip(l_mask[i], 0)
            trans_made = '_xf'
        # y axis flip
        elif self.flip and 0.25 <= prob < 0.5:
            # IMG
            if not self.rgb:                                                        
                image = np.flip(image, 1)
            else:
                for i in range(image.shape[-1]):                                
                    l_image[i] = np.flip(l_image[i], 1)                         
            # MASK
            if m_channel == 1:
                mask = np.flip(mask, 1)
            else:
                for i in range(mask.shape[-1]):
                    l_mask[i] = np.flip(l_mask[i], 1)
            trans_made = '_yf'
        # z axis flip
        elif self.flip and 0.5 <= prob < 0.75:
            # IMG
            if not self.rgb:                                                        
                image = np.flip(image, 2)       
            else:
                for i in range(image.shape[-1]):                                
                    l_image[i] = np.flip(l_image[i], 2)                         
            # MASK
            if m_channel == 1:
                mask = np.flip(mask, 2)
            else:
                for i in range(mask.shape[-1]):
                    l_mask[i] = np.flip(l_mask[i], 2)
            trans_made = '_zf'
       
        # [0-0.25): 90º rotation
        # [0.25-0.5): 180º rotation
        # [0.5-0.75): 270º rotation
        # [0.75-1]: nothing
        # 90º rotation on x axis
        prob = random.uniform(0, 1)
        if self.rotation and prob < 0.25:   
            # IMG
            if not self.rgb:
                image = rotate(image, axes=(0, 1), angle=90, mode='reflect',    
                               reshape=False)
            else:
                for i in range(image.shape[-1]):                                 
                    l_image[i] = rotate(l_image[i], axes=(0, 1), angle=90, mode='reflect',
                                        reshape=False)
            # MASK
            if m_channel == 1:
                mask = rotate(mask, axes=(0, 1), angle=90, mode='reflect',
                          reshape=False)
            else:
                for i in range(mask.shape[-1]):
                    l_mask[i] = rotate(l_mask[i], axes=(0, 1), angle=90, mode='reflect',
                          reshape=False)
            trans_made += '_yr90'
        # 180º rotation on x axis
        elif self.rotation and 0.25 <= prob < 0.5:
            # IMG
            if not self.rgb:
                image = rotate(image, axes=(0, 1), angle=180, mode='reflect',
                               reshape=False)
            else:
                for i in range(image.shape[-1]):                                
                    l_image[i] = rotate(l_image[i], axes=(0, 1), angle=180, mode='reflect',
                                        reshape=False)                          
            # MASK
            if m_channel == 1:
                mask = rotate(mask, axes=(0, 1), angle=180, mode='reflect',
                          reshape=False)
            else:
                for i in range(mask.shape[-1]):
                    l_mask[i] = rotate(l_mask[i], axes=(0, 1), angle=180, mode='reflect',
                          reshape=False)
            trans_made += '_yr180'
        # 270º rotation on x axis
        elif self.rotation and 0.5 <= prob < 0.75:
            # IMG
            if not self.rgb:
                image = rotate(image, axes=(0, 1), angle=270, mode='reflect',
                               reshape=False)
            else:
                for i in range(image.shape[-1]):                                
                    l_image[i] = rotate(l_image[i], axes=(0, 1), angle=270, mode='reflect',
                                        reshape=False)                          
            # MASK
            if m_channel == 1:
                mask = rotate(mask, axes=(0, 1), angle=270, mode='reflect',
                          reshape=False)
            else:
                for i in range(mask.shape[-1]):
                    l_mask[i] = rotate(l_mask[i], axes=(0, 1), angle=270, mode='reflect',
                          reshape=False)
            trans_made += '_yr270'

        # [0-0.25): x axis shift 
        # [0.25-0.5): y axis shift
        # [0.5-0.75): z axis shift 
        # [0.75-1]: nothing
        #
        # x axis shift 
        prob = random.uniform(0, 1)
        if self.shift_range != 0 and prob < 0.25:
            s = [0] * image.ndim
            s[0] = int(self.shift_range * image.shape[0])
            trans_made += '_xs' 
        # y axis shift 
        elif self.shift_range != 0 and 0.25 <= prob < 0.5:                   
            s = [0] * image.ndim                                          
            s[1] = int(self.shift_range * image.shape[1])          
            trans_made += '_ys'
        # z axis shift
        elif self.shift_range != 0 and 0.5 <= prob < 0.75:                   
            s = [0] * image.ndim                                          
            s[2] = int(self.shift_range * image.shape[2])          
            trans_made += '_zs'

        if self.shift_range != 0 and prob < 0.75:
            # IMG                                                               
            if not self.rgb:                                                    
                shift(image, shift=s, mode='reflect')                           
            else:                                                               
                for i in range(image.shape[-1]):                                
                    shift(l_image[i], shift=s, mode='reflect')                  
            # MASK                                                              
            if m_channel == 1:                                             
                shift(mask, shift=s, mode='reflect')                            
            else:                                                               
                for i in range(mask.shape[-1]):                                 
                    shift(l_mask[i], shift=s, mode='reflect')

        if self.imgaug:
            self.seq.deterministic = True
            if m_channel == 1 and not self.rgb:
                segmap = SegmentationMapsOnImage(mask, shape=mask.shape)
                image, vol_mask = self.seq(image=image, segmentation_maps=segmap)
                mask = vol_mask.get_arr()
            elif mask.shape[-1] != 1 and not self.rgb:
                for i in range(mask.shape[-1]):                             
                    self.seq.to_deterministic()
                    segmap = SegmentationMapsOnImage(l_mask[i], shape=l_mask[i].shape)
                    if i == 0:                                              
                        image, vol_mask = self.seq(image=image, segmentation_maps=segmap)
                    else:                                                   
                        _, vol_mask = self.seq(image=image, segmentation_maps=segmap)
                    l_mask[i] = vol_mask.get_arr()
            elif m_channel == 1 and self.rgb:
                mask = mask.astype(np.uint8)
                segmap = SegmentationMapsOnImage(mask, shape=mask.shape)
                for i in range(image.shape[-1]):                                 
                    if i == 0:                                                  
                        l_image[i], vol_mask = self.seq(image=l_image[i], segmentation_maps=segmap)
                    else:                                                       
                        l_image[i] = self.seq(image=l_image[i])
                mask = vol_mask.get_arr()
            else:
                for i in range(image.shape[-1]):                             
                    if i < len(l_mask):
                        segmap = SegmentationMapsOnImage(l_mask[i], shape=l_mask[i].shape)

                    l_image[i], vol_mask = self.seq(image=l_image[i], segmentation_maps=segmap)

                    if i < len(l_mask):
                        l_mask[i] = vol_mask.get_arr()
                
                # Apply transformations to the rest of the masks (if any)
                if len(l_mask) > len(l_image):
                    offset = len(l_mask) - len(l_image)
                    for i in range(offset):
                        self.seq.to_deterministic()
                        j = len(l_image) + i
                        segmap = SegmentationMapsOnImage(l_mask[j], shape=l_mask[j].shape)
                        _, vol_mask = self.seq(image=l_image[-1], segmentation_maps=segmap)
                    l_mask[j] = vol_mask.get_arr() 

            self.seq.deterministic = False
            trans_made += self.trans_made
        
        if trans_made == '':
            trans_made = '_none'

        # IMG
        if not self.rgb:
            image = np.expand_dims(image, axis=-1)
        else:
            for i in range(image.shape[-1]):                                     
                    l_image[i] = np.expand_dims(l_image[i], -1)
            image = np.concatenate(l_image, axis=-1)

        # MASK
        if m_channel == 1:
            mask = np.expand_dims(mask, axis=-1)
        else:
            for i in range(mask.shape[-1]):                                 
                    l_mask[i] = np.expand_dims(l_mask[i], -1)                   
            mask = np.concatenate(l_mask, axis=-1)

        return image, mask, trans_made

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
                t_str = ''
            else:
                if not train:
                    self.__draw_grid(vol)
                    self.__draw_grid(vol_mask)

                sample_x[i], sample_y[i], t_str = \
                    self.apply_transform(vol, vol_mask)

            # Save transformed 3D volumes 
            if save_to_dir:
                os.makedirs(out_dir, exist_ok=True)
                # Original image/mask
                f = os.path.join(out_dir, "orig_x_"+str(pos)+t_str+'.tiff')
                aux = self.X[pos].copy()
                self.__draw_grid(aux)
                aux = np.expand_dims((np.transpose(aux, (2,0,1,3))*255).astype(np.uint8), 1)
                imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'})
                f = os.path.join(out_dir, "orig_y_"+str(pos)+t_str+'.tiff')
                aux = self.Y[pos].copy()
                self.__draw_grid(aux)
                aux = np.expand_dims((np.transpose(aux, (2,0,1,3))*255).astype(np.uint8), 1)
                imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'})
                # Transformed
                f = os.path.join(out_dir, "x_aug_"+str(pos)+t_str+'.tiff')
                aux = np.expand_dims((np.transpose(sample_x[i], (2,0,1,3))*255).astype(np.uint8), 1)
                imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'})
                # Mask
                f = os.path.join(out_dir, "y_aug_"+str(pos)+t_str+'.tiff')
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
