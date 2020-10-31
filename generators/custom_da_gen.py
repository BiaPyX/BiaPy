import tensorflow as tf
import numpy as np
import random
import os
from tqdm import tqdm
from PIL import Image
from PIL import ImageEnhance
from data_manipulation import img_to_onehot_encoding, random_crop
import imgaug as ia                                                             
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


class ImageDataGenerator(tf.keras.utils.Sequence):
    """Custom ImageDataGenerator.

       Based on:
           https://github.com/czbiohub/microDL 
           https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    """

    def __init__(self, X, Y, batch_size=32, shape=(256,256,1), shuffle=False,
                 da=True, hist_eq=False, rotation90=False, rotation_range=0.0,
                 vflip=False, hflip=False, elastic=False, g_blur=False,
                 median_blur=False, gamma_contrast=False,
                 random_crops_in_DA=False, prob_map=False, train_prob=None, 
                 val=False, n_classes=1, extra_data_factor=1):

        """ImageDataGenerator constructor.
                                                                                
       Args:                                                                    
            X (4D Numpy array): data. E.g. (num_of_images, x, y, channels).

            Y (4D Numpy array): mask data. E.g. (num_of_images, x, y, channels).

            batch_size (int, optional): size of the batches.

            shape (3D int tuple, optional): shape of the desired images.

            shuffle (bool, optional): to decide if the indexes will be shuffled
            after every epoch. 

            da (bool, optional): to activate the data augmentation. 

            hist_eq (bool, optional): to make histogram equalization on images.
            
            rotation90 (bool, optional): to make rotations of 90º, 180º or 270º.

            rotation_range (float, optional): range of rotation degrees.

            vflips (bool, optional): to make vertical flips. 

            hflips (bool, optional): to make horizontal flips.
            
            elastic (bool, optional): to make elastic deformations.

            g_blur (bool, optional): to insert gaussian blur on the images.

            median_blur (bool, optional): to insert median blur.

            gamma_contrast (bool, optional): to insert gamma constrast 
            changes on images. 

            random_crops_in_DA (bool, optional): decide to make random crops in 
            DA (before transformations).

            prob_map (bool, optional): take the crop center based on a given    
            probability ditribution.

            train_prob (numpy array, optional): probabilities of each pixels to
            use with prob_map actived. 

            val (bool, optional): advice the generator that the images will be
            to validate the model to not make random crops (as the val. data must
            be the same on each epoch). Valid when random_crops_in_DA is set.

            n_classes (int, optional): number of classes. If ``> 1`` one-hot
            encoding will be done on the ground truth.

            extra_data_factor (int, optional): factor to multiply the batches 
            yielded in a epoch. It acts as if ``X`` and ``Y``` where concatenated
            ``extra_data_factor`` times.
        """

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
        self.divide = True if np.max(X) > 1 else False
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
           Args:
                index (int): batch index counter.
            
           Returns:
               batch_x (4D Numpy array): corresponding X elements of the batch.
               E.g. (batch_size, x, y, channels).

               batch_y (4D Numpy array): corresponding Y elements of the batch.
               E.g. (batch_size, x, y, channels).
        """

        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        batch_x = np.zeros((len(indexes), *self.shape), dtype=np.uint8)
        batch_y = np.zeros((len(indexes), *self.shape), dtype=np.uint8)

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
        if self.divide:
            batch_x = batch_x.astype('float32')
            batch_y = batch_y.astype('float32')
            batch_x *= 1./255
            batch_y *= 1./255

        if self.n_classes > 1:
            batch_y_ = np.zeros((len(indexes),) + self.shape[:2] + (self.n_classes,))
            for i in range(len(indexes)):
                batch_y_[i] = np.asarray(img_to_onehot_encoding(batch_y[i]))

            batch_y = batch_y_

        return batch_x, batch_y


    def on_epoch_end(self):
        """Updates indexes after each epoch."""

        self.indexes = self.o_indexes
        if self.shuffle:
            np.random.shuffle(self.indexes)


    def __draw_grid(self, im, grid_width=50, v=1):
        """Draw grid of the specified size on an image. 
           
           Args:                                                                
                im (2D Numpy array): image to be modified. E. g. (x, y)
                
                grid_width (int, optional): grid's width. 

                v (int, optional): value to create the grid with.
        """

        for i in range(0, im.shape[0], grid_width):
            im[i, :] = v
        for j in range(0, im.shape[1], grid_width):
            im[:, j] = v


    def get_transformed_samples(self, num_examples, save_to_dir=False, 
                                out_dir='aug', save_prefix=None, train=True, 
                                original_elastic=True, random_images=True, 
                                force_full_images=False):
        """Apply selected transformations to a defined number of images from
           the dataset. 
            
           Args:
                num_examples (int): number of examples to generate.

                save_to_dir (bool, optional): save the images generated. The 
                purpose of this variable is to check the images generated by 
                data augmentation.

                out_dir (str, optional): name of the folder where the
                examples will be stored. If any provided the examples will be
                generated under a folder 'aug'.

                save_prefix (str, optional): prefix to add to the generated 
                examples' name. 

                train (bool, optional): to avoid drawing a grid on the 
                generated images. This should be set when the samples will be
                used for training.

                original_elastic (bool, optional): to save also the original
                images when an elastic transformation is performed.

                random_images (bool, optional): randomly select images from the
                dataset. If False the examples will be generated from the start
                of the dataset. 

                force_full_images (bool, optional): force the usage of the entire
                images. Useful to generate extra images and overide
                'self.random_crops_in_DA' functionality.


            Returns:
                batch_x (4D Numpy array): batch of data.
                E.g. (num_examples, x, y, channels).

                batch_y (4D Numpy array): batch of data mask.
                E.g. (num_examples, x, y, channels).
        """

        print("### TR-SAMPLES ###")

        if self.random_crops_in_DA and not force_full_images:
            batch_x = np.zeros((num_examples,) + self.shape, dtype=np.uint8)
            batch_y = np.zeros((num_examples,) + self.shape, dtype=np.uint8)
        else:
            batch_x = np.zeros((num_examples,) + self.X.shape[1:], dtype=np.uint8)
            batch_y = np.zeros((num_examples,) + self.Y.shape[1:], dtype=np.uint8)

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
                self.__draw_grid(batch_x[i,...,0])
                self.__draw_grid(batch_y[i,...,0], v=255)

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
                im = Image.fromarray(self.X[pos,:,:,0])                          
                im = im.convert('L')                                            
                im.save(os.path.join(out_dir,str(pos)+'_orig_x'+self.t_made+".png"))
                mask = Image.fromarray(self.Y[pos,:,:,0])                        
                mask = mask.convert('L')                                        
                mask.save(os.path.join(out_dir,str(pos)+'_orig_y'+self.t_made+".png"))

                # Save transformed images
                im = Image.fromarray(batch_x[i,:,:,0])
                im = im.convert('L')
                im.save(os.path.join(out_dir, str(pos)+p+'x'+self.t_made+".png"))
                mask = Image.fromarray(batch_y[i,:,:,0])
                mask = mask.convert('L')
                mask.save(os.path.join(out_dir, str(pos)+p+'y'+self.t_made+".png"))

                # Save the original images with a point that represents the 
                # selected coordinates to be the center of the crop
                if self.random_crops_in_DA and self.train_prob is not None\
                   and not force_full_images:
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

