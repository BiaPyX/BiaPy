import tensorflow as tf
import numpy as np
import random
import os
from tqdm import tqdm
from PIL import Image
from PIL import ImageEnhance
from data_2D_manipulation import random_crop
from util import img_to_onehot_encoding
import imgaug as ia                                                             
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


class ImageDataGenerator(tf.keras.utils.Sequence):
    """Custom ImageDataGenerator based on `imgaug <https://github.com/aleju/imgaug-doc>`_
       transformations. 

       Based on https://github.com/czbiohub/microDL and 
       https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

       Parameters
       ----------
       X : 4D Numpy array
           Data. E.g. ``(num_of_images, x, y, channels)``.

       Y : 4D Numpy array
           Mask data. E.g. ``(num_of_images, x, y, 1)``.

       batch_size : int, optional
           Size of the batches.

       shape : 3D int tuple, optional
           Shape of the desired images.

       shuffle : bool, optional
           To decide if the indexes will be shuffled after every epoch.

       da : bool, optional
           To activate the data augmentation.

       hist_eq : bool, optional
           To make histogram equalization on images.

       rotation90 : bool, optional
           To make rotations of ``90º``, ``180º`` or ``270º``.

       rotation_range : float, optional
           Range of rotation degrees.

       vflips : bool, optional
           To make vertical flips.

       hflips : bool, optional
           To make horizontal flips.

       elastic : bool, optional
           To make elastic deformations.

       g_blur : bool, optional
           To insert gaussian blur on the images.

       median_blur : bool, optional
           To insert median blur.

       gamma_contrast : bool, optional
           To insert gamma constrast changes on images.

       random_crops_in_DA : bool, optional
           Decide to make random crops in DA (before transformations).

       prob_map : bool, optional
           Take the crop center based on a given probability ditribution.

       train_prob : 4D Numpy array, optional
           Probabilities of each pixels to use with ``prob_map=True``.
           E.g. ``(num_of_images, x, y, channels)``.

       val : bool, optional
           Advise the generator that the images will be to validate the
           model to not make random crops (as the val. data must be the same
           on each epoch). Valid when ``random_crops_in_DA`` is set.

       n_classes : int, optional
           Number of classes. If ``> 1`` one-hot encoding will be done on
           the ground truth.

       extra_data_factor : int, optional
           Factor to multiply the batches yielded in a epoch. It acts as if
           ``X`` and ``Y``` where concatenated ``extra_data_factor`` times.


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

           data_gen_args = dict(
               X=X_train, Y=Y_train, batch_size=6, shape=(256, 256, 1), shuffle=True,
               rotation_range=True, vflip=True, hflip=True)

           data_gen_val_args = dict(
               X=X_val, Y=Y_val, batch_size=6, shape=(256, 256, 1), shuffle=True,
               da=False, val=True)

           train_generator = ImageDataGenerator(**data_gen_args)                       
           val_generator = ImageDataGenerator(**data_gen_val_args)  


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
           train_prob = calculate_2D_volume_prob_map(
                Y_train, 0.94, 0.06, save_file=''prob_map.npy')
          
           data_gen_args = dict(
               X=X_train, Y=Y_train, batch_size=6, shape=(256, 256, 1), shuffle=True,
               rotation_range=True, vflip=True, hflip=True, random_crops_in_DA=True,
               prob_map=True, train_prob=train_prob)

           data_gen_val_args = dict(
               X=X_val, Y=Y_val, batch_size=6, shape=(256, 256, 1), shuffle=True,
               da=False, val=True)

           train_generator = ImageDataGenerator(**data_gen_args)
           val_generator = ImageDataGenerator(**data_gen_val_args)
    """

    def __init__(self, X, Y, batch_size=32, shape=(256,256,1), shuffle=False,
                 da=True, hist_eq=False, rotation90=False, rotation_range=0.0,
                 vflip=False, hflip=False, elastic=False, g_blur=False,
                 median_blur=False, gamma_contrast=False,
                 random_crops_in_DA=False, prob_map=False, train_prob=None, 
                 val=False, n_classes=1, extra_data_factor=1):

        if rotation_range != 0 and rotation90:
            raise ValueError("'rotation_range' and 'rotation90' can not be set "
                             "together")
        if median_blur and g_blur:
            raise ValuError("'median_blur' and 'g_blur' can not be set together")
        if random_crops_in_DA and (shape[0] != shape[1]):
            raise ValuError("When 'random_crops_in_DA' is selected the shape "
                            "given must be square, e.g. (256, 256, 1)")
            
        self.shape = shape
        self.batch_size = batch_size
        self.divideX = True if np.max(X) > 1 else False
        self.divideY = True if np.max(Y) > 1 else False
        self.X = np.asarray(X, dtype=np.uint8)
        self.Y = np.asarray(Y, dtype=np.uint8)
        self.shuffle = shuffle
        self.n_classes = n_classes
        self.da = da
        self.random_crops_in_DA = random_crops_in_DA
        self.train_prob = train_prob
        self.val = val
        self.o_indexes = np.arange(len(self.X))
        if extra_data_factor > 1:
            self.extra_data_factor = extra_data_factor
            self.o_indexes = np.concatenate([self.o_indexes]*extra_data_factor)
        else:
            self.extra_data_factor = 1
        self.on_epoch_end()
        
        da_options = []
        self.t_made = ''
        if hist_eq:
            da_options.append(iaa.HistogramEqualization())
            self.t_made += '_heq'
        if rotation90:
            da_options.append(iaa.Rot90((0, 3)))
            self.t_made += '_rot[0,90,180,270]'
        if rotation_range != 0:
            t = iaa.Affine(rotate=(-rotation_range,rotation_range), mode='reflect')
            # Force the reflect mode on segmentation maps
            t._mode_segmentation_maps = "reflect"
            da_options.append(t)
            self.t_made += '_rot_range[-'+str(rotation_range)+','+str(rotation_range)+']'
        if vflip:
            da_options.append(iaa.Flipud(0.5))
            self.t_made += '_vf'
        if hflip:
            da_options.append(iaa.Fliplr(0.5))                                  
            self.t_made += '_hf'
        if elastic:
            da_options.append(iaa.Sometimes(0.5,iaa.ElasticTransformation(alpha=(240, 250), sigma=25, mode="reflect")))
            self.t_made += '_elastic' 
        if g_blur:
            da_options.append(iaa.Sometimes(0.5,iaa.GaussianBlur(sigma=(1.0, 2.0))))
            self.t_made += '_gblur'
        if median_blur:
            da_options.append(iaa.Sometimes(0.5,iaa.MedianBlur(k=(3,7))))
            self.t_made += '_mblur'
        if gamma_contrast:
            da_options.append(iaa.Sometimes(0.5,iaa.GammaContrast((1.25, 1.75))))
            self.t_made += '_gcontrast'
        self.seq = iaa.Sequential(da_options)
        self.t_made = '_none' if self.t_made == '' else self.t_made


    def __len__(self):
        """Defines the number of batches per epoch."""
    
        return int(np.ceil(self.X.shape[0]*self.extra_data_factor/self.batch_size))


    def __getitem__(self, index):
        """Generation of one batch data. 

           Parameters
           ----------
           index : int
               Batch index counter.
            
           Returns
           -------
           batch_x : 4D Numpy array
               Corresponding X elements of the batch. 
               E.g. ``(batch_size, x, y, channels)``.

           batch_y : 4D Numpy array
               Corresponding Y elements of the batch.
               E.g. ``(batch_size, x, y, channels)``.
        """

        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        batch_x = np.zeros((len(indexes), *self.shape), dtype=np.uint8)
        batch_y = np.zeros((len(indexes), *self.shape[:2]+(1,)), dtype=np.uint8)

        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        for i, j in zip(range(len(indexes)), indexes):
            if self.random_crops_in_DA:
                batch_x[i], batch_y[i] = random_crop(
                    self.X[j], self.Y[j], self.shape[:2],
                    self.val, img_prob=(self.train_prob[j] if self.train_prob is not None else None))
            else:
                batch_x[i], batch_y[i] = self.X[j], self.Y[j]
            
            if self.da: 
                segmap = SegmentationMapsOnImage(
                    batch_y[i], shape=batch_y[i].shape)
                t_img, t_mask = self.seq(
                    image=batch_x[i], segmentation_maps=segmap)
                t_mask = t_mask.get_arr()                                       
                batch_x[i] = t_img
                batch_y[i] = t_mask
                
        # Need to divide before transformations as some imgaug library functions
        # need uint8 datatype
        if self.divideX:
            batch_x = batch_x.astype('float32')
            batch_x *= 1./255
        #if self.divideY:
        #    batch_y *= 1./255
        batch_y = batch_y.astype('float32')

        if self.n_classes > 1:
            batch_y_ = np.zeros((len(indexes),) + self.shape[:2] + (self.n_classes,))
            for i in range(len(indexes)):
                batch_y_[i] = np.asarray(img_to_onehot_encoding(
                                             batch_y[i], self.n_classes))

            batch_y = batch_y_

        return batch_x, batch_y


    def on_epoch_end(self):
        """Updates indexes after each epoch."""

        self.indexes = self.o_indexes
        if self.shuffle:
            np.random.shuffle(self.indexes)


    def __draw_grid(self, im, grid_width=50, v=1):
        """Draw grid of the specified size on an image. 
           
           Parameters
           ----------                                                                
           im : 3D Numpy array
               Image to be modified. E. g. ``(x, y, channels)``
                
           grid_width : int, optional
               Grid's width. 

           v : int, optional
               Value to create the grid with.
        """

        for i in range(0, im.shape[0], grid_width):
            im[i] = v
        for j in range(0, im.shape[1], grid_width):
            im[:, j] = v


    def get_transformed_samples(self, num_examples, save_to_dir=False, 
                                out_dir='aug', save_prefix=None, train=True, 
                                random_images=True, force_full_images=False):
        """Apply selected transformations to a defined number of images from
           the dataset. 
            
           Parameters
           ----------
           num_examples : int
               Number of examples to generate.

           save_to_dir : bool, optional
               Save the images generated. The purpose of this variable is to
               check the images generated by data augmentation.

           out_dir : str, optional
               Name of the folder where the examples will be stored. If any
               provided the examples will be generated under a folder ``aug``.

           save_prefix : str, optional
               Prefix to add to the generated examples' name. 

           train : bool, optional
               To avoid drawing a grid on the generated images. This should be
               set when the samples will be used for training.

           random_images : bool, optional
               Randomly select images from the dataset. If ``False`` the examples
               will be generated from the start of the dataset. 

           force_full_images : bool, optional
               Force the usage of the entire images. Useful to generate extra
               images and override ``random_crops_in_DA`` functionality.


           Returns
           -------
           batch_x : 4D Numpy array
               Batch of data. E.g. ``(num_examples, x, y, channels)``.

           batch_y : 4D Numpy array
               Batch of data mask. E.g. ``(num_examples, x, y, channels)``.

          
           Examples
           --------
           ::

               # EXAMPLE 1
               # Generate 10 samples following with the example 1 of the class definition
               X_train = np.ones((1776, 256, 256, 1))                               
               Y_train = np.ones((1776, 256, 256, 1))                               
                                                                                
               data_gen_args = dict(                                                
                   X=X_train, Y=Y_train, batch_size=6, shape=(256, 256, 1),
                   shuffle=True, rotation_range=True, vflip=True, hflip=True)                     
                                                                                
               train_generator = ImageDataGenerator(**data_gen_args)                       

               train_generator.get_transformed_samples(                                
                   10, save_to_dir=True, train=False, out_dir='da_dir') 

               # EXAMPLE 2
               # If random crop in DA-time is choosen, as the example 2 of the class definition, 
               # the call should be the same but two more images will be stored: img and mask
               # representing the random crop extracted. There a red point is painted representing 
               # the pixel choosen to be the center of the random crop and a blue square which
               # delimits crop boundaries

               train_prob = calculate_2D_volume_prob_map(                           
                   Y_train, 0.94, 0.06, save_file=''prob_map.npy')                 
                                                                                
               data_gen_args = dict(                                                
                   X=X_train, Y=Y_train, batch_size=6, shape=(256, 256, 1), shuffle=True,
                   rotation_range=True, vflip=True, hflip=True, random_crops_in_DA=True,
                   prob_map=True, train_prob=train_prob)                            
               train_generator = ImageDataGenerator(**data_gen_args)
            
                train_generator.get_transformed_samples(                                
                   10, save_to_dir=True, train=False, out_dir='da_dir')
            

           Example 2 will store two additional images as the following:

           +--------------------------------------+-------------------------------------------+
           | .. figure:: img/rd_crop_2d.png       | .. figure:: img/rd_crop_mask_2d.png       |
           |   :width: 80%                        |   :width: 70%                             |
           |   :align: center                     |   :align: center                          |
           |                                      |                                           |
           |   Original crop                      |   Original crop mask                      |
           +--------------------------------------+-------------------------------------------+

           Together with these images another pair of images will be stored: the crop made and a 
           transformed version of it, which is really the generator output. 
    
           For instance, setting ``elastic=True`` the above extracted crop should be transformed as follows:
        
           +--------------------------------------+-------------------------------------------+
           | .. figure:: img/original_crop_2d.png | .. figure:: img/original_crop_mask_2d.png |
           |   :width: 80%                        |   :width: 70%                             |
           |   :align: center                     |   :align: center                          |
           |                                      |                                           |
           |   Original crop                      |   Original crop mask                      |
           +--------------------------------------+-------------------------------------------+
           | .. figure:: img/elastic_crop_2d.png  | .. figure:: img/elastic_crop_mask_2d.png  |
           |   :width: 80%                        |   :width: 70%                             |
           |   :align: center                     |   :align: center                          |
           |                                      |                                           |
           |   Elastic transformation of the crop |   Elastic transformation of them crop mask|
           +--------------------------------------+-------------------------------------------+

           The grid is only painted if ``train=False`` which should be used just to display transformations made.
           Selecting random rotations between 0 and 180 degrees should generate the following:
            
           +---------------------------------------------+--------------------------------------------------+
           | .. figure:: img/original_rd_rot_crop_2d.png | .. figure:: img/original_rd_rot_crop_mask_2d.png |
           |   :width: 80%                               |   :width: 70%                                    |
           |   :align: center                            |   :align: center                                 |
           |                                             |                                                  |
           |   Original crop                             |   Original crop mask                             |
           +---------------------------------------------+--------------------------------------------------+
           | .. figure:: img/rd_rot_crop_2d.png          | .. figure:: img/rd_rot_crop_mask_2d.png          |
           |   :width: 80%                               |   :width: 70%                                    |
           |   :align: center                            |   :align: center                                 |
           |                                             |                                                  |
           |   Random rotation [0, 180] of the crop      |   Random rotation [0, 180] of the crop mask      |
           +---------------------------------------------+--------------------------------------------------+
        """

        print("### TR-SAMPLES ###")

        if self.random_crops_in_DA and not force_full_images:
            batch_x = np.zeros((num_examples,) + self.shape, dtype=np.uint8)
            batch_y = np.zeros((num_examples,) + self.shape[:2]+(1,), dtype=np.uint8)
        else:
            batch_x = np.zeros((num_examples,) + self.X.shape[1:], dtype=np.uint8)
            batch_y = np.zeros((num_examples,) + self.Y.shape[1:3]+(1,), dtype=np.uint8)

        if save_to_dir:
            p = '_' if save_prefix is None else str(save_prefix)
            os.makedirs(out_dir, exist_ok=True)
   
        grid = False if train else True
                 
        # Generate the examples 
        print("0) Creating the examples of data augmentation . . .")
        for i in tqdm(range(num_examples)):
            if random_images:
                pos = random.randint(1,self.X.shape[0]-1) 
            else:
                pos = i

            # Apply crops if selected
            if self.random_crops_in_DA and not force_full_images:
                batch_x[i], batch_y[i], ox, oy,\
                s_x, s_y = random_crop(self.X[pos], self.Y[pos], self.shape[:2], 
                    self.val, draw_prob_map_points=True,
                    img_prob=(self.train_prob[pos] if self.train_prob is not None else None))
            else:
                batch_x[i] = self.X[pos]
                batch_y[i] = self.Y[pos]

            if not train:
                self.__draw_grid(batch_x[i])
                self.__draw_grid(batch_y[i], v=255)

            if save_to_dir:
                if self.X.shape[-1] > 1:
                    o_x = np.copy(batch_x[i])                                 
                else:
                    o_x = np.copy(batch_x[i,...,0])
                o_y = np.copy(batch_y[i,...,0])

            # Apply transformations
            if self.da:                                                         
                segmap = SegmentationMapsOnImage(                               
                    batch_y[i], shape=batch_y[i].shape)                         
                t_img, t_mask = self.seq(                                       
                    image=batch_x[i], segmentation_maps=segmap)                 
                t_mask = t_mask.get_arr()                                       
                batch_x[i] = t_img                                              
                batch_y[i] = t_mask

            if save_to_dir:
                # Save original images
                self.__draw_grid(o_x)                                           
                self.__draw_grid(o_y, v=255)  
                if self.X.shape[-1] > 1:
                    im = Image.fromarray(o_x, 'RGB')
                else:
                    im = Image.fromarray(o_x)
                    im = im.convert('L')                                            
                im.save(os.path.join(out_dir,str(pos)+'_orig_x'+self.t_made+".png"))
                mask = Image.fromarray(o_y)
                mask = mask.convert('L')                                        
                mask.save(os.path.join(out_dir,str(pos)+'_orig_y'+self.t_made+".png"))

                # Save transformed images
                if self.X.shape[-1] > 1:
                    im = Image.fromarray(batch_x[i], 'RGB')
                else:
                    im = Image.fromarray(batch_x[i,:,:,0])
                    im = im.convert('L')
                im.save(os.path.join(out_dir, str(pos)+p+'x'+self.t_made+".png"))
                mask = Image.fromarray(batch_y[i,:,:,0])
                mask = mask.convert('L')
                mask.save(os.path.join(out_dir, str(pos)+p+'y'+self.t_made+".png"))

                if self.n_classes > 1:
                    h_maks = np.zeros(self.shape[:2] + (self.n_classes,))
                    h_maks = np.asarray(img_to_onehot_encoding(
                                          batch_y[i], self.n_classes))
                    print("h_maks: {}".format(h_maks.shape))
                    for i in range(self.n_classes):
                        a = Image.fromarray(h_maks[...,i]*255)
                        a= a.convert('L')
                        a.save(os.path.join(out_dir, str(pos)+"h_mask_"+str(i)+".png"))

                # Save the original images with a point that represents the 
                # selected coordinates to be the center of the crop
                if self.random_crops_in_DA and self.train_prob is not None\
                   and not force_full_images:
                    if self.X.shape[-1] > 1:
                        im = Image.fromarray(self.X[pos], 'RGB') 
                    else:
                        im = Image.fromarray(self.X[pos,:,:,0]) 
                    im = im.convert('RGB')                                                  
                    px = im.load()                                                          
                        
                    # Paint the selected point in red
                    p_size=6
                    for col in range(oy-p_size, oy+p_size):
                        for row in range(ox-p_size, ox+p_size): 
                            if col >= 0 and col < self.X.shape[1] and \
                               row >= 0 and row < self.X.shape[2]:
                               px[row, col] = (255, 0, 0) 
                    
                    # Paint a blue square that represents the crop made 
                    for row in range(s_x, s_x+self.shape[0]):
                        px[row, s_y] = (0, 0, 255)
                        px[row, s_y+self.shape[0]-1] = (0, 0, 255)
                    for col in range(s_y, s_y+self.shape[0]):                    
                        px[s_x, col] = (0, 0, 255)
                        px[s_x+self.shape[0]-1, col] = (0, 0, 255)

                    im.save(os.path.join(out_dir, str(pos)+p+'mark_x'+self.t_made+'.png'))
                   
                    mask = Image.fromarray(self.Y[pos,:,:,0]) 
                    mask = mask.convert('RGB')                                      
                    px = mask.load()                                              
                        
                    # Paint the selected point in red
                    for col in range(oy-p_size, oy+p_size):                       
                        for row in range(ox-p_size, ox+p_size):                   
                            if col >= 0 and col < self.Y.shape[1] and \
                               row >= 0 and row < self.Y.shape[2]:                
                               px[row, col] = (255, 0, 0)

                    # Paint a blue square that represents the crop made
                    for row in range(s_x, s_x+self.shape[0]):                
                        px[row, s_y] = (0, 0, 255)                          
                        px[row, s_y+self.shape[0]-1] = (0, 0, 255)       
                    for col in range(s_y, s_y+self.shape[0]):                
                        px[s_x, col] = (0, 0, 255)                          
                        px[s_x+self.shape[0]-1, col] = (0, 0, 255)

                    mask.save(os.path.join(out_dir, str(pos)+p+'mark_y'+self.t_made+'.png'))          
                
        print("### END TR-SAMPLES ###")
        return batch_x, batch_y

