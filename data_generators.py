import tensorflow as tf
import numpy as np
import random
import os
import cv2
import sys
import math
from tqdm import tqdm
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy import ndimage
from PIL import Image
from PIL import ImageEnhance
from texttable import Texttable
from tensorflow.keras.preprocessing.image import ImageDataGenerator as kerasDA
from util import array_to_img, img_to_array, do_save_wm, make_weight_map
from data_manipulation  import img_to_onehot_encoding


class ImageDataGenerator(tf.keras.utils.Sequence):
    """Custom ImageDataGenerator.

       Based on:
           https://github.com/czbiohub/microDL 
           https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    """

    def __init__(self, X, Y, batch_size=32, dim=(256,256), n_channels=1, 
                 shuffle=False, da=True, e_prob=0.0, elastic=False, vflip=False,
                 hflip=False, rotation90=False, rotation_range=0.0, 
                 brightness_range=None, median_filter_size=[0, 0], 
                 random_crops_in_DA=False, crop_length=0, prob_map=False, 
                 train_prob=None, val=False, softmax_out=False):
        """ImageDataGenerator constructor.
                                                                                
       Args:                                                                    
            X (4D Numpy array): data. E.g. (image_number, x, y, channels).

            Y (4D Numpy array): mask data. E.g. (image_number, x, y, channels).

            batch_size (int, optional): size of the batches.

            dim (2D int tuple, optional): dimension of the desired images. As no 
            effect if random_crops_in_DA is active, as the dimension will be 
            selected by that variable instead.

            n_channels (int, optional): number of channels of the input images.

            shuffle (bool, optional): to decide if the indexes will be shuffled
            after every epoch. 

            da (bool, optional): to activate the data augmentation. 

            e_prob (float, optional): probability of making elastic
            transformations. 

            elastic (bool, optional): to make elastic transformations.

            vflip (bool, optional): if true vertical flip are made.

            hflip (bool, optional): if true horizontal flips are made.

            rotation90 (bool, optional): to make rotations of 90º, 180º or 270º.

            rotation_range (float, optional): range of rotation degrees.

            brightness_range (2D float tuple, optional): Range for picking a
            brightness shift value from.

            median_filter_size (int, optional): size of the median filter. If 0 
            no median filter will be applied. 

            random_crops_in_DA (bool, optional): decide to make random crops in 
            DA (before transformations).

            crop_length (int, optional): length of the random crop after DA.
    
            prob_map (bool, optional): take the crop center based on a given    
            probability ditribution.

            train_prob (numpy array, optional): probabilities of each pixels to
            use with prob_map actived. 

            val (bool, optional): advice the generator that the images will be
            to validate the model to not make random crops (as the val. data must
            be the same on each epoch). Valid when random_crops_in_DA is set.
        """

        if rotation_range != 0 and rotation90 == True:
            raise ValueError("'rotation_range' and 'rotation90' can not be set "
                             "together")

        self.dim = dim
        self.batch_size = batch_size
        self.X = X/255 if np.max(X) > 1 else X
        self.Y = Y/255 if np.max(Y) > 1 else Y
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.da = da
        self.e_prob = e_prob
        self.elastic = elastic
        self.vflip = vflip
        self.hflip = hflip
        self.rotation90 = rotation90
        self.rotation_range = rotation_range
        if brightness_range is None:
            self.brightness_range = [1.0, 1.0]
        else:
            self.brightness_range = brightness_range
        self.median_filter_size = median_filter_size
        self.random_crops_in_DA = random_crops_in_DA
        self.crop_length = crop_length
        self.train_prob = train_prob
        self.val = val
        self.softmax_out = softmax_out 
        self.on_epoch_end()
        
        if self.X.shape[1] == self.X.shape[2] or self.random_crops_in_DA == True:
            self.squared = True
        else:
            self.squared = False
            if rotation90 == True:
                print("[AUG] Images not square, only 180 rotations will be done.")

        # Create a list which will hold a counter of the number of times a 
        # transformation is performed. 
        self.t_counter = [0 ,0 ,0 ,0 ,0 ,0] 

        if self.random_crops_in_DA == True:
            self.dim = (self.crop_length, self.crop_length)

    def __len__(self):
        """Defines the number of batches per epoch."""
    
        return int(np.ceil(self.X.shape[0]/self.batch_size))

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

        batch_x = np.empty((len(indexes), *self.dim, self.n_channels))
        batch_y = np.empty((len(indexes), *self.dim, self.n_channels))

        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        for i, j in zip(range(len(indexes)), indexes):
            if self.random_crops_in_DA == True:
                batch_x[i], batch_y[i] = random_crop(
                    self.X[j], self.Y[j], (self.crop_length, self.crop_length),
                    self.val, img_prob=(self.train_prob[j] if self.train_prob is not None else None))
            else:
                batch_x[i], batch_y[i] = self.X[j], self.Y[j]
            
            if self.da == True: 
                batch_x[i], batch_y[i], _ = self.apply_transform(
                    batch_x[i], batch_y[i])
                
        if self.softmax_out == True:
            batch_y_ = np.zeros((len(indexes), ) + self.Y.shape[1:3] + (2,))
            for i in range(len(indexes)):
                batch_y_[i] = np.asarray(img_to_onehot_encoding(batch_y[i]))

            batch_y = batch_y_

        return batch_x, batch_y

    def print_da_stats(self):
        """Print the counter of the transformations made in a table."""

        t = Texttable()
        t.add_rows([['Elastic', 'V. flip', 'H. flip', '90º rot.', '180º rot.',
                     '270º rot.'], [self.t_counter[0], self.t_counter[1],
                     self.t_counter[2], self.t_counter[3], self.t_counter[4], 
                     self.t_counter[5]] ])
        print(t.draw())

    def on_epoch_end(self):
        """Updates indexes after each epoch."""

        self.indexes = np.arange(len(self.X))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __draw_grid(self, im, grid_width=50):
        """Draw grid of the specified size on an image. 
           
           Args:                                                                
               im (2D Numpy array): image to be modified. E. g. (x, y)

               grid_width (int, optional): grid's width. 
        """

        v = 1

        for i in range(0, im.shape[1], grid_width):
            im[:, i] = v
        for j in range(0, im.shape[0], grid_width):
            im[j, :] = v

    def apply_transform(self, image, mask, grid=False):
        """Transform the input image and its mask at the same time with one of
           the selected choices based on a probability. 
                
           Args:
                image (2D Numpy array): image to be transformed. E. g. (x, y)
        
                mask (2D Numpy array): image's mask. E. g. (x, y)

                grid (bool, optional): Draws a grid in to the elastic 
                transfomations to visualize it clearly. Do not set this option 
                to train the network!
            
           Returns: 
                trans_image (2D Numpy array): transformed image. E. g. (x, y)

                trans_mask (2D Numpy array): transformed image mask. E. g. (x, y)

                transform_string (str): string formed by applied transformations.
        """

        trans_image = np.copy(image)
        trans_mask = np.copy(mask)
        transform_string = '' 
        transformed = False

        # Elastic transformation
        prob = random.uniform(0, 1)
        if self.elastic == True and prob < self.e_prob:

            if grid == True:
                self.__draw_grid(trans_image)
                self.__draw_grid(trans_mask)

            im_concat = np.concatenate((trans_image, trans_mask), axis=2)            

            im_concat_r = elastic_transform(
                im_concat, im_concat.shape[1]*2, im_concat.shape[1]*0.08,
                im_concat.shape[1]*0.08)

            trans_image = np.expand_dims(im_concat_r[...,0], axis=-1)
            trans_mask = np.expand_dims(im_concat_r[...,1], axis=-1)
            transform_string = '_e'
            transformed = True
            self.t_counter[0] += 1
     
 
        # [0-0.25): vertical flip
        # [0.25-0.5): horizontal flip
        # [0.5-0.75): vertical + horizontal flip
        # [0.75-1]: nothing
        #
        # Vertical flip
        prob = random.uniform(0, 1)
        if self.vflip == True and 0 <= prob < 0.25:
            trans_image = np.flip(trans_image, 0)
            trans_mask = np.flip(trans_mask, 0)
            transform_string = transform_string + '_vf'
            transformed = True 
            self.t_counter[1] += 1
        # Horizontal flip
        elif self.hflip == True and 0.25 <= prob < 0.5:
            trans_image = np.flip(trans_image, 1)
            trans_mask = np.flip(trans_mask, 1)
            transform_string = transform_string + '_hf'
            transformed = True
            self.t_counter[2] += 1 
        # Vertical and horizontal flip
        elif self.hflip == True and 0.5 <= prob < 0.75:
            trans_image = np.flip(trans_image, 0)                               
            trans_mask = np.flip(trans_mask, 0)
            trans_image = np.flip(trans_image, 1)                               
            trans_mask = np.flip(trans_mask, 1)
            transform_string = transform_string + '_hfvf'
            transformed = True
            self.t_counter[1] += 1
            self.t_counter[2] += 1
            
        # Free rotation from -range to range (in degrees)
        if self.rotation_range != 0:
            theta = np.random.uniform(-self.rotation_range, self.rotation_range)
            trans_image = ndimage.rotate(trans_image, theta, reshape=False, 
                                         mode='reflect', order=1)
            trans_mask = ndimage.rotate(trans_mask, theta, reshape=False, 
                                        mode='reflect', order=0)
            transform_string = transform_string + '_rRange' + str(int(theta))
            transformed = True

        # Rotation with multiples of 90 degrees
        # [0-0.25): 90º rotation
        # [0.25-0.5): 180º rotation
        # [0.5-0.75): 270º rotation
        # [0.75-1]: nothing
        # Note: if the images are not squared only 180 rotations will be done.
        prob = random.uniform(0, 1)
        if self.squared == True:
            # 90 degree rotation
            if self.rotation90 == True and 0 <= prob < 0.25:
                trans_image = np.rot90(trans_image)
                trans_mask = np.rot90(trans_mask)
                transform_string = transform_string + '_r90'
                transformed = True 
                self.t_counter[3] += 1
            # 180 degree rotation
            elif self.rotation90 == True and 0.25 <= prob < 0.5:
                trans_image = np.rot90(trans_image, 2)
                trans_mask = np.rot90(trans_mask, 2)
                transform_string = transform_string + '_r180'
                transformed = True 
                self.t_counter[4] += 1
            # 270 degree rotation
            elif self.rotation90 == True and 0.5 <= prob < 0.75:
                trans_image = np.rot90(trans_image, 3)
                trans_mask = np.rot90(trans_mask, 3)
                transform_string = transform_string + '_r270'
                transformed = True 
                self.t_counter[5] += 1
        else:
            if self.rotation90 == True and 0 <= prob < 0.5:
                trans_image = np.rot90(trans_image, 2)                          
                trans_mask = np.rot90(trans_mask, 2)                            
                transform_string = transform_string + '_r180'                   
                transformed = True                                              
                self.t_counter[4] += 1

        # Brightness
        if self.brightness_range != [1.0, 1.0]:
            brightness = np.random.uniform(self.brightness_range[0],
                                           self.brightness_range[1])
            trans_image = array_to_img(trans_image)
            trans_image = imgenhancer_Brightness = ImageEnhance.Brightness(trans_image)
            trans_image = imgenhancer_Brightness.enhance(brightness)
            trans_image = img_to_array(trans_image)
            transform_string = transform_string + '_b' + str(round(brightness, 2))
            transformed = True
            
        # Median filter
        if self.median_filter_size != [0, 0]:
            mf_size = np.random.randint(self.median_filter_size[0], 
                                        self.median_filter_size[1])
            if mf_size % 2 == 0:
                mf_size += 1
            trans_image = cv2.medianBlur(trans_image.astype('int16'), mf_size)
            trans_image = np.expand_dims(trans_image, axis=-1).astype('float32') 
            transform_string = transform_string + '_mf' + str(mf_size)
            transformed = True

        if transformed == False:
            transform_string = '_none'         

        return trans_image, trans_mask, transform_string


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

                train (bool, optional): flag to avoid drawing a grid on the 
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

        if self.random_crops_in_DA == True and force_full_images == False:
            batch_x = np.zeros((num_examples, self.crop_length, self.crop_length,
                                self.X.shape[3]), dtype=np.float32)                                  
            batch_y = np.zeros((num_examples, self.crop_length, self.crop_length,
                                self.Y.shape[3]), dtype=np.float32)
        else:
            batch_x = np.zeros((num_examples,) + self.X.shape[1:], dtype=np.float32)
            batch_y = np.zeros((num_examples,) + self.Y.shape[1:], dtype=np.float32)

        if save_to_dir == True:
            prefix = ""
            if save_prefix is not None:
                prefix = str(save_prefix)
    
            os.makedirs(out_dir, exist_ok=True)
   
        grid = False if train == True else True
                 
        # Generate the examples 
        print("0) Creating the examples of data augmentation . . .")
        for i in tqdm(range(num_examples)):
            if random_images == True:
                pos = random.randint(1,self.X.shape[0]-1) 
            else:
                pos = i

            # Apply crops if selected
            if self.random_crops_in_DA == True and force_full_images == False:
                batch_x[i], batch_y[i], ox, oy,\
                s_x, s_y = random_crop(
                    self.X[pos], self.Y[pos], 
                    (self.crop_length, self.crop_length), self.val, 
                    draw_prob_map_points=True,
                    img_prob=(self.train_prob[pos] if self.train_prob is not None else None))
            else:
                batch_x[i] = self.X[pos]
                batch_y[i] = self.Y[pos]

            batch_x[i], batch_y[i], t_str = self.apply_transform(
                batch_x[i], batch_y[i], grid=grid)

            # Save transformed images
            if save_to_dir == True:    
                im = Image.fromarray(batch_x[i,:,:,0]*255)
                im = im.convert('L')
                im.save(os.path.join(
                            out_dir, prefix + 'x_' + str(pos) + t_str + ".png"))
                mask = Image.fromarray(batch_y[i,:,:,0]*255)
                mask = mask.convert('L')
                mask.save(os.path.join(
                              out_dir, prefix + 'y_' + str(pos) + t_str + ".png"))

                # Save the original images with a point that represents the 
                # selected coordinates to be the center of the crop
                if self.random_crops_in_DA == True and self.train_prob is not None\
                   and force_full_images == False:
                    im = Image.fromarray(self.X[pos,:,:,0]*255) 
                    im = im.convert('RGB')                                                  
                    px = im.load()                                                          
                        
                    # Paint the selected point in red
                    p_size=6
                    for col in range(oy-p_size,oy+p_size):
                        for row in range(ox-p_size,ox+p_size): 
                            if col >= 0 and col < self.X.shape[1] and \
                               row >= 0 and row < self.X.shape[2]:
                               px[row, col] = (255, 0, 0) 
                    
                    # Paint a blue square that represents the crop made 
                    for row in range(s_x, s_x+self.crop_length):
                        px[row, s_y] = (0, 0, 255)
                        px[row, s_y+self.crop_length-1] = (0, 0, 255)
                    for col in range(s_y, s_y+self.crop_length):                    
                        px[s_x, col] = (0, 0, 255)
                        px[s_x+self.crop_length-1, col] = (0, 0, 255)

                    im.save(os.path.join(
                                out_dir, prefix + 'mark_x_' + str(pos) + t_str 
                                + '.png'))
                   
                    mask = Image.fromarray(self.Y[pos,:,:,0]*255) 
                    mask = mask.convert('RGB')                                      
                    px = mask.load()                                              
                        
                    # Paint the selected point in red
                    for col in range(oy-p_size,oy+p_size):                       
                        for row in range(ox-p_size,ox+p_size):                   
                            if col >= 0 and col < self.Y.shape[1] and \
                               row >= 0 and row < self.Y.shape[2]:                
                               px[row, col] = (255, 0, 0)

                    # Paint a blue square that represents the crop made
                    for row in range(s_x, s_x+self.crop_length):                
                        px[row, s_y] = (0, 0, 255)                          
                        px[row, s_y+self.crop_length-1] = (0, 0, 255)       
                    for col in range(s_y, s_y+self.crop_length):                
                        px[s_x, col] = (0, 0, 255)                          
                        px[s_x+self.crop_length-1, col] = (0, 0, 255)

                    mask.save(os.path.join(
                                  out_dir, prefix + 'mark_y_' + str(pos) 
                                  + t_str + '.png'))          
                
                # Save also the original images if an elastic transformation 
                # was made
                if original_elastic == True and '_e' in t_str: 
                    im = Image.fromarray(self.X[i,:,:,0]*255)
                    im = im.convert('L')
                    im.save(os.path.join(
                                out_dir, prefix + 'x_' + str(pos) + t_str 
                                + '_original.png'))
    
                    mask = Image.fromarray(self.Y[i,:,:,0]*255)
                    mask = mask.convert('L')
                    mask.save(os.path.join(
                                  out_dir, prefix + 'y_' + str(pos) + t_str 
                                  + '_original.png'))

        print("### END TR-SAMPLES ###")
        return batch_x, batch_y


def keras_da_generator(X_train=None, Y_train=None, X_val=None, Y_val=None, 
                       ld_img_from_disk=False, data_paths=None, target_size=None, 
                       c_target_size=None, batch_size_value=1, val=True, 
                       save_examples=True, out_dir='aug', 
                       hflip=False, vflip=False, seedValue=42, rotation_range=180, 
                       fill_mode='reflect', preproc_function=False, 
                       featurewise_center=False, brightness_range=None, 
                       channel_shift_range=0.0, shuffle_train=True,
                       shuffle_val=False, featurewise_std_normalization=False, 
                       zoom=False, w_shift_r=0.0, h_shift_r=0.0, shear_range=0,
                       random_crops_in_DA=False, crop_length=0, 
                       weights_on_data=False, weights_path=None):             
                                                                                
    """Makes data augmentation of the given input data.                         
                                                                                
       Args:                                                                    
            X_train (4D Numpy array, optional): train data. If this arg is provided 
            data_paths arg value will be avoided.
            E.g. (image_number, x, y, channels).

            Y_train (4D Numpy array, optional): train mask data.                             
            E.g. (image_number, x, y, channels).

            X_val (4D Numpy array, optional): validation data.
            E.g. (image_number, x, y, channels).

            Y_val (4D Numpy array, optional): validation mask data.
            E.g. (image_number, x, y, channels).

            ld_img_from_disk (bool, optional): flag to make the generator with
            images loaded form disk instead of use X_train, Y_train, X_val 
            and Y_val.

            data_paths (list of str, optional): list of paths where the data is 
            stored. Use this instead of X_train and Y_train args to do not 
            charge the data in memory and make a generator over the paths 
            instead. The path order is this: 1) train images path 2) train masks
            path 3) validation images path 4) validation masks path 5) test 
            images path 6) test masks path 7) complete images path (this last 
            useful to make the smoothing post processing, as it requires the 
            reconstructed data) 8) complete mask path. To provide the validation 
            data val must be set to True. If no validation data provided the 
            order of the paths is the same avoiding validation ones.

            target_size (2D int tuple, optional): size where the images will be 
            resized if data_paths is defined. 

            c_target_size (2D int tuple, optional): size where complete images 
            will be resized if data_paths is defined. 

            batch_size_value (int, optional): batch size.

            val (bool, optional): If True validation data generator will be 
            returned.

            save_examples (bool, optional): if true 5 examples of DA are stored.

            out_dir (string, optional): save directory suffix.                  

            hflip (bool, optional): if true horizontal flips are made.          

            vflip (bool, optional): if true vertical flip are made.             

            seedValue (int, optional): seed value.                              

            rotation_range (int, optional): range of rotation degrees.

            fill_mode (str, optional): ImageDataGenerator of Keras fill mode    
            values.

            preproc_function (bool, optional): if true preprocess function to   
            make random 180 degrees rotations are performed.                    

            featurewise_center (bool, optional): set input mean to 0 over the   
            dataset, feature-wise.

            brightness_range (2D float tuple, optional): range for picking a 
            brightness shift value from.

            channel_shift_range (float, optional): range for random channel 
            shifts.

            shuffle_train (bool, optional): randomize the training data on each 
            epoch.

            shuffle_val(bool, optional): randomize the validation data on each 
            epoch.

            featurewise_std_normalization (bool, optional): divide inputs by std 
            of the dataset, feature-wise.                                       

            zoom (bool, optional): make random zoom in the images.              

            w_shift_r (float, optional): width shift range.

            h_shift_r (float, optional): height shift range.

            shear_range (float, optional): range to apply shearing 
            transformations. 

            random_crops_in_DA (bool, optional): decide to make random crops 
            in DA (after transformations).                                           

            crop_length (int, optional): length of the random crop before DA. 
            weights_on_data (bool, optional): flag to make a weights generator 
            for the weighted loss. 

       Returns:                                                                 
            train_generator (Iterator): train data iterator.                                                           

            val_generator (Iterator, optional): validation data iterator.

            X_test_aug (Iterator, optional): test data iterator.

            Y_test_aug (Iterator, optional): test masks iterator.

            W_test_aug (Iterator, optional): test weight maps iterator.

            X_complete_aug (Iterator, optional): original data iterator useful 
            to make the smoothing post processing, as it requires the 
            reconstructed data.
            
            Y_complete_aug (Iterator, optional): original mask iterator.

            W_complete_aug (Iterator, optional): original weight maps iterator.

            n_train_samples (int, optional): number of training samples.  

            n_val_samples (int, optional): number of validation samples.

            n_test_samples (int, optional): number of test samples.
    """                                                                         

    if X_train is None and ld_img_from_disk == False:
        raise ValueError("One between X_train or ld_img_from_disk must be selected")

    if ld_img_from_disk == True and (target_size is None or c_target_size is None):
        raise ValueError("target_size and c_target_size must be specified when "
                         "ld_img_from_disk is selected")

    if ld_img_from_disk == True and len(data_paths) != 8: 
        raise ValueError(
            "data_paths must contain the following paths: 1) train path ; 2) "
            "train masks path ; 3) validation path ; 4) validation masks path ; "
            "5) test path ; 6) test masks path ; 7) complete images path 8) "
            "complete image mask path")

    if weights_on_data == True and weights_path is None:
       raise ValueError(
           "'weights_path' must be provided when weights is selected")

    zoom_val = 0.25 if zoom == True else 0                                      
                                                                                
    data_gen_args1 = dict(
        horizontal_flip=hflip, vertical_flip=vflip, fill_mode=fill_mode, 
        rotation_range=rotation_range, featurewise_center=featurewise_center,            
        featurewise_std_normalization=featurewise_std_normalization, 
        zoom_range=zoom_val, width_shift_range=w_shift_r,
        height_shift_range=h_shift_r, shear_range=shear_range,
        channel_shift_range=channel_shift_range,
        brightness_range=brightness_range, rescale=1./255)
    data_gen_args2 = dict(
        horizontal_flip=hflip, vertical_flip=vflip, fill_mode=fill_mode, 
        rotation_range=rotation_range, zoom_range=zoom_val, 
        width_shift_range=w_shift_r, height_shift_range=h_shift_r, 
        shear_range=shear_range, rescale=1./255)                              

    # Obtaining the path where the data is stored                                                                                 
    if ld_img_from_disk == True:
        train_path = data_paths[0]
        train_mask_path = data_paths[1]
        val_path = data_paths[2]
        val_mask_path = data_paths[3]
        test_path = data_paths[4]
        test_mask_path = data_paths[5]
        complete_path = data_paths[6]
        complete_mask_path = data_paths[7]
                            
    # Generators
    X_datagen_train = kerasDA(**data_gen_args1)
    Y_datagen_train = kerasDA(**data_gen_args2)                                 
    X_datagen_test = kerasDA(rescale=1./255)
    Y_datagen_test = kerasDA(rescale=1./255)                                 
    if ld_img_from_disk == True:
        complete_datagen = kerasDA(rescale=1./255)
        complete_mask_datagen = kerasDA(rescale=1./255)                                 
    if val == True:
        X_datagen_val = kerasDA(rescale=1./255)
        Y_datagen_val = kerasDA(rescale=1./255)                                                   
    if weights_on_data == True:
        w_datagen = kerasDA(**data_gen_args2)

    # Save a few examples 
    if save_examples == True:
        print("Saving some samples of the train generator . . .")        
        os.makedirs(out_dir, exist_ok=True)

        if ld_img_from_disk == False:
            i = 0
            for batch in X_datagen_train.flow(
                X_train, save_to_dir=out_dir, batch_size=batch_size_value,
                shuffle=True, seed=seedValue, save_prefix='x', save_format='png'):
                i = i + 1
                if i > 2:
                    break
            i = 0
            for batch in Y_datagen_train.flow(
                Y_train, save_to_dir=out_dir, batch_size=batch_size_value,
                shuffle=True, seed=seedValue, save_prefix='y', save_format='png'):
                i = i + 1
                if i > 2:
                    break
        else:
            i = 0
            for batch in X_datagen_train.flow_from_directory(
                train_path, save_to_dir=out_dir, target_size=target_size,
                batch_size=batch_size_value, shuffle=True, seed=seedValue,
                save_prefix='x', save_format='png'):
                i = i + 1
                if i > 2:
                    break
            i = 0
            for batch in Y_datagen_train.flow_from_directory(
                train_mask_path, save_to_dir=out_dir, target_size=target_size,
                batch_size=batch_size_value, shuffle=True, seed=seedValue,
                save_prefix='y', save_format='png'):
                i = i + 1
                if i > 2:
                    break
  
    # Create the generators with the provided data
    if ld_img_from_disk == False:
        X_train_aug = X_datagen_train.flow(X_train, batch_size=batch_size_value,       
                                           shuffle=shuffle_train, seed=seedValue)
        Y_train_aug = Y_datagen_train.flow(Y_train, batch_size=batch_size_value,       
                                           shuffle=shuffle_train, seed=seedValue)

    # Create the generator loading images directly from disk
    else:
        print("Train data loaded from directory: {}".format(train_path))
        X_train_aug = X_datagen_train.flow_from_directory(
            train_path, target_size=target_size, class_mode=None, 
            color_mode="grayscale", batch_size=batch_size_value,
            shuffle=shuffle_train, seed=seedValue)
        Y_train_aug = Y_datagen_train.flow_from_directory(
            train_mask_path, target_size=target_size, class_mode=None,
            color_mode="grayscale", batch_size=batch_size_value,
            shuffle=shuffle_train, seed=seedValue)
        n_train_samples = X_train_aug.n 
        
        print("Test data loaded from directory: {}".format(test_path))
        X_test_aug = X_datagen_test.flow_from_directory(
            test_path, target_size=target_size, class_mode=None, 
            color_mode="grayscale", batch_size=batch_size_value, shuffle=False, 
            seed=seedValue)
        Y_test_aug = Y_datagen_test.flow_from_directory(
            test_mask_path, target_size=target_size, class_mode=None,
            color_mode="grayscale", batch_size=batch_size_value, shuffle=False, 
            seed=seedValue)

        n_test_samples = X_test_aug.n

        print("Complete data loaded from directory: {}".format(complete_path))
        X_complete_aug = complete_datagen.flow_from_directory(
            complete_path, target_size=c_target_size, class_mode=None,
            color_mode="grayscale", batch_size=batch_size_value, shuffle=False)
        Y_complete_aug = complete_datagen.flow_from_directory(
            complete_mask_path, target_size=c_target_size, class_mode=None,
            color_mode="grayscale", batch_size=batch_size_value, shuffle=False)

    # Create the validation generator
    if ld_img_from_disk == False:
        if val == True:
            X_val_aug = X_datagen_val.flow(X_val, batch_size=batch_size_value,
                                           shuffle=shuffle_val, seed=seedValue)
            Y_val_aug = Y_datagen_val.flow(Y_val, batch_size=batch_size_value,
                                           shuffle=shuffle_val, seed=seedValue)
    else:
        print("Validation data loaded from directory: {}".format(val_path))
        X_val_aug = X_datagen_val.flow_from_directory(
            val_path, target_size=target_size, batch_size=batch_size_value,
            class_mode=None, color_mode="grayscale", shuffle=shuffle_val, 
            seed=seedValue)
        Y_val_aug = Y_datagen_val.flow_from_directory(
            val_mask_path, target_size=target_size, batch_size=batch_size_value,
            class_mode=None, color_mode="grayscale", shuffle=shuffle_val, 
            seed=seedValue)
        n_val_samples = X_val_aug.n

    # Create the weight map generator
    if weights_on_data == True:
        train_w_path = os.path.join(weights_path, 'train')
        val_w_path = os.path.join(weights_path, 'val')
        test_w_path = os.path.join(weights_path, 'test')
        if ld_img_from_disk == True:
            complete_w_path = os.path.join(weights_path, 'complete')            
        
        # Create generator from disk
        if ld_img_from_disk == True:
   
            # Create train maks generator without augmentation
            print("Create train mask generator in case we need it to create the"
                  " map weigths" )
            Y_train_no_aug = kerasDA().flow_from_directory(
                train_mask_path, target_size=target_size, class_mode=None,
                color_mode="grayscale", batch_size=batch_size_value, 
                shuffle=False)

            prepare_weight_maps(
                train_w_path, val_w_path, test_w_path, c_w_path=complete_w_path, 
                ld_img_from_disk=True, Y_train_aug=Y_train_no_aug, 
                Y_val_aug=Y_val_aug, Y_test_aug=Y_test_aug, 
                Y_cmp_aug=Y_complete_aug, batch_size_value=batch_size_value)

        # Create generator from data
        else:
            prepare_weight_maps(
                train_w_path, val_w_path, test_w_path, Y_train=Y_train, 
                Y_val=Y_val, Y_test=Y_test, batch_size_value=batch_size_value)

        # Retrieve weight-maps
        t_filelist = sorted(next(os.walk(train_w_path))[2])
        v_filelist = sorted(next(os.walk(val_w_path))[2])
        te_filelist = sorted(next(os.walk(test_w_path))[2])
        if ld_img_from_disk == True:
            c_filelist = sorted(next(os.walk(complete_w_path))[2])
        
        # Loads all weight-map images in a list
        t_weights = [np.load(os.path.join(train_w_path, fname)) for fname in t_filelist]
        t_weights = np.array(t_weights, dtype=np.float32)
        t_weights = t_weights.reshape((len(t_weights),target_size[0],
                                       target_size[1],1))
        v_weights = [np.load(os.path.join(val_w_path, fname)) for fname in v_filelist]
        v_weights = np.array(v_weights, dtype=np.float32)
        v_weights = v_weights.reshape((len(v_weights),target_size[0],
                                       target_size[1],1))
        te_weights = [np.load(os.path.join(test_w_path, fname)) for fname in te_filelist]
        te_weights = np.array(te_weights, dtype=np.float32)                       
        te_weights = te_weights.reshape((len(te_weights),target_size[0],           
                                       target_size[1],1))
        if ld_img_from_disk == True:
            c_weights = [np.load(os.path.join(complete_w_path, fname)) for fname in c_filelist]
            c_weights = np.array(c_weights, dtype=np.float32)                     
            c_weights = c_weights.reshape((len(c_weights),c_target_size[0],        
                                          c_target_size[1],1))

        # Create the weight generator 
        W_train_aug = w_datagen.flow(t_weights, batch_size=batch_size_value,
                                     shuffle=shuffle_train, seed=seedValue)
        W_val_aug = w_datagen.flow(v_weights, batch_size=batch_size_value,
                                   shuffle=shuffle_val, seed=seedValue)
        W_test_aug = w_datagen.flow(te_weights, batch_size=batch_size_value,
                                    shuffle=False)
        if ld_img_from_disk == True:
            W_cmp_aug = w_datagen.flow(c_weights, batch_size=batch_size_value,    
                                       shuffle=False)
    else:
        W_train_aug = W_val_aug = W_test_aug = None

        if ld_img_from_disk == True:
            W_cmp_aug = None
        

    # Combine generators into one which yields image, masks and weights (if used)
    train_generator = combine_generators(X_train_aug, Y_train_aug, W_train_aug)                 
    if val == True:
        val_generator = combine_generators(X_val_aug, Y_val_aug, W_val_aug)
    
    # Make random crops over the generators                                                               
    if random_crops_in_DA == True:                                                
        train_generator = crop_generator(train_generator, crop_length, 
                                         weights_on_data=weights_on_data)
        if val == True:
            val_generator = crop_generator(val_generator, crop_length, val=True,
                                           weights_on_data=weights_on_data)
 
    if ld_img_from_disk == True:
        return train_generator, val_generator, X_test_aug, Y_test_aug, \
               W_test_aug, X_complete_aug, Y_complete_aug, W_cmp_aug, \
               n_train_samples, n_val_samples, n_test_samples
    else:
        if val == True:
            return train_generator, val_generator
        else:
            return train_generator


def keras_gen_samples(num_samples, X_data=None, Y_data=None, 
                      ld_img_from_disk=False, data_paths=None, target_size=None, 
                      batch_size_value=1, shuffle_data=True, hflip=True, 
                      vflip=True, seedValue=42, rotation_range=180, 
                      fill_mode='reflect', preproc_function=None,
                      featurewise_center=False, brightness_range=None,
                      channel_shift_range=0.0,
                      featurewise_std_normalization=False,
                      zoom=False, w_shift_r=0.0, h_shift_r=0.0, shear_range=0):

    """Generates samples based on keras da generator.
    
       Args:
            num_samples (int): number of sampels to create.

            X_data (4D Numpy array, optional): data used to generate samples.
            E.g. (image_number, x, y, channels).

            Y_data (4D Numpy array, optional): mask used to generate samples. 
            E.g. (image_number, x, y, channels).

            ld_img_from_disk (bool, optional): flag to advise the function to 
            load images from disk instead of use X_data or Y_data.

            data_paths (list of str): path were the data and mask are stored to 
            generate the new samples.

            target_size (2D int tuple, optional): shape of the images to load 
            from disk.

            batch_size_value (int, optional): batch size value.

            shuffle_data (bool, optional): shuffle the data while generating new
            samples. 

            hflip (bool, optional): if true horizontal flips are made.

            vflip (bool, optional): if true vertical flip are made.

            seedValue (int, optional): seed value.

            rotation_range (int, optional): range of rotation degrees.

            fill_mode (str, optional): ImageDataGenerator of Keras fill mode
            values.

            preproc_function (bool, optional): if true preprocess function to
            make random 180 degrees rotations are performed.

            featurewise_center (bool, optional): set input mean to 0 over the
            dataset, feature-wise.

            brightness_range (2D float tuple, optional): range for picking a 
            brightness shift value from.

            channel_shift_range (float, optional): range for random channel
            shifts.

            featurewise_std_normalization (bool, optional): divide inputs by std
            of the dataset, feature-wise.

            zoom (bool, optional): make random zoom in the images.

            w_shift_r (float, optional): width shift range.

            h_shift_r (float, optional): height shift range.

            shear_range (float, optional): range to apply shearing
            transformations.
        
       Return:
            x_samples (4D Numpy array): data new samples.
            E.g. (num_samples, x, y, channels).

            y_samples (4D numpy array): mask new samples.
            E.g. (num_samples, x, y, channels).
    """

    if num_samples == 0:
        return None

    if X_data is None and ld_img_from_disk == False:
        raise ValueError("One between X_data or ld_img_from_disk must be selected")

    if ld_img_from_disk == True and target_size is None:
        raise ValueError("target_size must be specified when ld_img_from_disk "
                         "is selected")

    if ld_img_from_disk == True and len(data_paths) != 2:
        raise ValueError("data_paths must contain the following paths: 1) data "
                         "path ; 2) data masks path")

    zoom_val = 0.25 if zoom == True else 0

    data_gen_args1 = dict(
        horizontal_flip=hflip, vertical_flip=vflip, fill_mode=fill_mode, 
        rotation_range=rotation_range, preprocessing_function=preproc_function,
        featurewise_center=featurewise_center, 
        featurewise_std_normalization=featurewise_std_normalization,
        zoom_range=zoom_val, width_shift_range=w_shift_r, 
        height_shift_range=h_shift_r, shear_range=shear_range, 
        channel_shift_range=channel_shift_range, 
        brightness_range=brightness_range, rescale=1./255)
    data_gen_args2 = dict(
        horizontal_flip=hflip, vertical_flip=vflip, fill_mode=fill_mode, 
        rotation_range=rotation_range, preprocessing_function=preproc_function,
        zoom_range=zoom_val, width_shift_range=w_shift_r, 
        height_shift_range=h_shift_r, shear_range=shear_range, rescale=1./255)

    X_datagen = kerasDA(**data_gen_args1)
    Y_datagen = kerasDA(**data_gen_args2)

    # Use X_data and Y_data to generate the samples 
    if ld_img_from_disk == False:
        x_samples = np.zeros((num_samples,) + X_data.shape[1:], dtype=np.float32)
        y_samples = np.zeros((num_samples,) + Y_data.shape[1:], dtype=np.float32)

        n_batches = int(num_samples / batch_size_value) \
                    + (num_samples % batch_size_value > 0)

        print("Generating new data samples . . .")
        i = 0                                                                   
        c = 0                                                                   
        for batch in X_datagen.flow(X_data, batch_size=batch_size_value,        
                                    shuffle=shuffle_data, seed=seedValue):      
            for j in range(0, batch.shape[0]):                                  
                x_samples[c] = batch[j]                                         
                c += 1                                                          
            i = i + 1                                                           
            if i >= n_batches:                                                  
                break                                                           
        i = 0                                                                   
        c = 0                                                                   
        for batch in Y_datagen.flow(Y_data, batch_size=batch_size_value,        
                                    shuffle=shuffle_data, seed=seedValue):      
            for j in range(0, batch.shape[0]):                                  
                y_samples[c] = batch[j]                                         
                c += 1                                                          
            i = i + 1                                                           
            if i >= n_batches:                                                  
                break

    # Use data_paths to load images from disk to make more samples
    else:
        x_samples = np.zeros((num_samples, X_data.shape[1], X_data.shape[2],
                            X_data.shape[3]), dtype=np.float32)
        y_samples = np.zeros((num_samples, Y_data.shape[1], Y_data.shape[2],
                            Y_data.shape[3]), dtype=np.float32)

        n_batches = int(num_samples / batch_size_value) \
                    + (num_samples % batch_size_value > 0)

        print("Generating new data samples . . .")
        i = 0
        c = 0
        for batch in X_datagen.flow_from_directory(
            data_paths[0], target_size=target_size, batch_size=batch_size_value,    
            shuffle=shuffle_data, seed=seedValue):
            for j in range(0, batch_size_value):
                x_samples[c] = batch[j]
                c += 1
            i = i + 1
            if i >= n_batches:
                break
        i = 0
        c = 0
        for batch in Y_datagen.flow_from_directory(
            data_paths[1], target_size=target_size, batch_size=batch_size_value,
            shuffle=shuffle_data, seed=seedValue):
            for j in range(0, batch_size_value):
                y_samples[c] = batch[j]
                c += 1
            i = i + 1
            if i >= n_batches:
                break

    return x_samples, y_samples


def prepare_weight_maps(train_w_path, val_w_path, test_w_path, c_w_path=None,
                        Y_train=None, Y_val=None, Y_test=None, 
                        ld_img_from_disk=False, Y_train_aug=None, Y_val_aug=None, 
                        Y_test_aug=None, Y_cmp_aug=None, batch_size_value=1):
    """Prepare weight maps saving them into a given path. If the paths are created
       it suposses that weight maps are already generated. 

       Args:
            train_w_path (str): path where train images should be stored.

            val_w_path (str): path where validation images should be stored.

            test_w_path (str): path where test images should be stored.

            c_w_path (str): path where complete images should be stored. 
            Ignored if 'ld_img_from_disk' is False.

            Y_train (4D Numpy array, optional): train mask data used to generate 
            weight maps. E.g. (image_number, x, y, channels).

            Y_val (Numpy array, optional): validation mask data used to generate
            weight maps. E.g. (image_number, x, y, channels).

            Y_test (Numpy array, optional): test mask data used to generate
            weight maps. E.g. (image_number, x, y, channels).

            ld_img_from_disk (bool, optional): flag to advise the function to
            load images from disk instead of use Y_train_data or Y_val.

            Y_train_aug (Keras ImageDataGenerator, optional): train mask 
            generator used to produce weight maps.

            Y_val_aug (Keras ImageDataGenerator, optional): validation mask
            generator used to produce weight maps.

            Y_test_aug (Keras ImageDataGenerator, optional): test mask generator 
            used to produce weight maps.

            Y_cmp_aug (Keras ImageDataGenerator, optional): complete image mask 
            generator used to produce weight maps.

            batch_size_value (int, optional): batch size value. 
    """

    if Y_train is None and ld_img_from_disk == False:
        raise ValueError("Y_train or ld_img_from_disk must be selected.")
    
    if ld_img_from_disk == True and (Y_train_aug is None or Y_val_aug is None\
        or Y_test_aug is None):
        raise ValueError("When ld_img_from_disk is selected Y_train_aug, "
                         "Y_val_aug and Y_test_aug must be provided")
    if c_w_path is not None and Y_cmp_aug is None:
        raise ValueError("'Y_cmp_aug' must be provided when c_w_path is provided")

    if ld_img_from_disk == False:
        if not os.path.exists(train_w_path):
            print("Constructing train weight maps with Y_train . . .")
            os.makedirs(train_w_path)
            do_save_wm(Y_train, train_w_path)

        if not os.path.exists(val_w_path):
            print("Constructing validation weight maps with Y_val . . .")
            os.makedirs(val_w_path)
            do_save_wm(Y_val, val_w_path)

        if not os.path.exists(test_w_path):
            print("Constructing test weight maps with Y_test . . .")
            os.makedirs(test_w_path)
            do_save_wm(Y_test, test_w_path)
    else:
        if not os.path.exists(train_w_path):
            print("Constructing train weight maps from disk . . .")
            os.makedirs(train_w_path)

            iterations = math.ceil(Y_train_aug.n/batch_size_value)
            
            # Count number of digits in n. This is important for the number
            # of leading zeros in the name of the maps
            d = len(str(Y_train_aug.n))

            cont = 0
            for i in tqdm(range(iterations)):
                batch = next(Y_train_aug)

                for j in range(0, batch.shape[0]):
                    if cont >= Y_train_aug.n:
                       break

                    img_map = make_weight_map(batch[j].copy())

                    # Resize correctly the maps so that it can be used in the model
                    rows, cols = img_map.shape
                    img_map = img_map.reshape((rows, cols, 1))

                    # Saving files as .npy files
                    np.save(os.path.join(
                                train_w_path, "w_" + str(cont).zfill(d)), img_map)

                    cont += 1

            Y_train_aug.reset()
        else:
            print("Train weight maps are already prepared!")
            
        if not os.path.exists(val_w_path):
            print("Constructing validation weight maps from disk . . .")
            os.makedirs(val_w_path)

            iterations = math.ceil(Y_val_aug.n/batch_size_value)

            # Count number of digits in n. This is important for the number
            # of leading zeros in the name of the maps
            d = len(str(Y_val_aug.n))

            cont = 0
            for i in tqdm(range(iterations)):
                batch = next(Y_val_aug)

                for j in range(0, batch.shape[0]):
                    if cont >= Y_val_aug.n:
                        break

                    img_map = make_weight_map(batch[j].copy())

                    # Resize correctly the maps so that it can be used in the model
                    rows, cols = img_map.shape
                    img_map = img_map.reshape((rows, cols, 1))

                    # Saving files as .npy files
                    np.save(os.path.join(
                                val_w_path, "w_" + str(cont).zfill(d)), img_map)

                    cont += 1

            Y_val_aug.reset()
        else:                                                                   
            print("Validation weight maps are already prepared!")

        if not os.path.exists(test_w_path):                                      
            print("Constructing test weight maps from disk . . .")        
            os.makedirs(test_w_path)                                             
                                                                                
            iterations = math.ceil(Y_test_aug.n/batch_size_value)                
                                                                                
            # Count number of digits in n. This is important for the number     
            # of leading zeros in the name of the maps                          
            d = len(str(Y_test_aug.n))                                           
                                                                                
            cont = 0                                                            
            for i in tqdm(range(iterations)):                                 
                batch = next(Y_test_aug)                                         
                for j in range(0, batch.shape[0]):                              
                    if cont >= Y_test_aug.n:                                     
                        break                                                   
                                                                                
                    img_map = make_weight_map(batch[j].copy())                  
                                                                                
                    # Resize correctly the maps so that it can be used in the model
                    rows, cols = img_map.shape                                  
                    img_map = img_map.reshape((rows, cols, 1))                  
                                                                                
                    # Saving files as .npy files                                
                    np.save(os.path.join(
                                test_w_path, "w_" + str(cont).zfill(d)), img_map)                     
                                                                                
                    cont += 1                                                   
                                                                                
            Y_test_aug.reset()
        else:                                                                   
            print("Test weight maps are already prepared!")

        if not os.path.exists(c_w_path):                                     
            print("Constructing complete image weight maps from disk . . .")              
            os.makedirs(c_w_path)                                            
                                                                                
            iterations = math.ceil(Y_cmp_aug.n/batch_size_value)               
                                                                                
            # Count number of digits in n. This is important for the number     
            # of leading zeros in the name of the maps                          
            d = len(str(Y_cmp_aug.n))                                          
                                                                                
            cont = 0                                                            
            for i in tqdm(range(iterations)):                                 
                batch = next(Y_cmp_aug)                                        
                for j in range(0, batch.shape[0]):                              
                    if cont >= Y_cmp_aug.n:                                    
                        break                                                   
                    
                    print("Making the map")                                                            
                    img_map = make_weight_map(batch[j].copy())                  
                    print("Map created")
                                                                                
                    # Resize correctly the maps so that it can be used in the model
                    rows, cols = img_map.shape                                  
                    img_map = img_map.reshape((rows, cols, 1))                  
                                                                                
                    # Saving files as .npy files                                
                    np.save(os.path.join(
                                c_w_path, "w_" + str(cont).zfill(d)), img_map)
                                                                                
                    cont += 1                                                   
                                                                                
            Y_cmp_aug.reset()
        else:                                                                   
            print("Complete image weight maps are already prepared!")

    print("Weight maps are prepared!")
    

def combine_generators(X_aug, Y_aug, W_aug=None):
    """Combine generators into one.
    
       Args:
            X_aug (Keras ImageDataGenerator): image generator.

            Y_aug (Keras ImageDataGenerator): mask generator.

            W_aug (Keras ImageDataGenerator, optional): weight map generator.
    
       Return:
            The combination of given generators.
    """

    if W_aug is None:
        generator = zip(X_aug, Y_aug)

        for (img, label) in generator:
            yield (img, label)
    else:
        generator = zip(X_aug, Y_aug, W_aug)
        for (img, label, weight) in generator:
            yield ([img, weight], label) 
        
        
def crop_generator(batches, crop_length, val=False, prob_map=False, 
                   weights_on_data=False):
    """Take as input a Keras ImageGen (Iterator) and generate random
       crops from the image batches generated by the original iterator.
        
       Based on:                                                                
           https://jkjung-avt.github.io/keras-image-cropping/  
    """

    while True:
        batch_x, batch_y = next(batches)
        if weights_on_data == True:
            x, w = batch_x 
            y = batch_y
            batch_crops_w = np.zeros((x.shape[0], crop_length, crop_length, 1))
        else:
            x = batch_x
            y = batch_y

        batch_crops_x = np.zeros((x.shape[0], crop_length, crop_length, 1))
        batch_crops_y = np.zeros((x.shape[0], crop_length, crop_length, 1))

        for i in range(x.shape[0]):
            if weights_on_data == True:
                batch_crops_x[i],\
                batch_crops_y[i],\
                batch_crops_w[i] = random_crop(
                    x[i], y[i], (crop_length, crop_length), val=val, 
                    weights_on_data=True, weight_map=w[i])

                yield ([batch_crops_x, batch_crops_w], batch_crops_y)
            
            else:
                batch_crops_x[i],\
                batch_crops_y[i] = random_crop(
                    batch_x[i], batch_y[i], (crop_length, crop_length), val=val, 
                    weights_on_data=weights_on_data) 

                yield (batch_crops_x, batch_crops_y)
        

def random_crop(image, mask, random_crop_size, val=False, 
                draw_prob_map_points=False, img_prob=None, weights_on_data=False,
                weight_map=None):
    """Random crop.

       Based on:                                                                
           https://jkjung-avt.github.io/keras-image-cropping/
    """

    if weights_on_data == True and weight_map is None:
        raise ValueError("When 'weights_on_data' is selected weight_map must be "
                         " provided")

    if weights_on_data == True:
        img, we = image
    else:
        img = image
   
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    if val == True:
        x = 0
        y = 0
        ox = 0
        oy = 0
    else:
        if img_prob is not None:
            prob = img_prob.ravel() 
            
            # Generate the random coordinates based on the distribution
            choices = np.prod(img_prob.shape)
            index = np.random.choice(choices, size=1, p=prob)
            coordinates = np.unravel_index(index, dims=img_prob.shape)
            x = int(coordinates[1][0])
            y = int(coordinates[0][0])
            ox = int(coordinates[1][0]) 
            oy = int(coordinates[0][0]) 
            
            # Adjust the coordinates to be the origin of the crop and control to
            # not be out of the image
            if y < int(random_crop_size[0]/2):
                y = 0
            elif y > img.shape[0] - int(random_crop_size[0]/2):
                y = img.shape[0] - random_crop_size[0]
            else:
                y -= int(random_crop_size[0]/2)
           
            if x < int(random_crop_size[1]/2):
                x = 0
            elif x > img.shape[1] - int(random_crop_size[1]/2):
                x = img.shape[1] - random_crop_size[1]
            else: 
                x -= int(random_crop_size[1]/2)
        else:
            x = np.random.randint(0, width - dx + 1)                                
            y = np.random.randint(0, height - dy + 1)

    if draw_prob_map_points == True:
        return img[y:(y+dy), x:(x+dx), :], mask[y:(y+dy), x:(x+dx), :], ox, oy, x, y
    else:
        if weights_on_data == True:
            return img[y:(y+dy), x:(x+dx), :], mask[y:(y+dy), x:(x+dx), :],\
                   weight_map[y:(y+dy), x:(x+dx), :]         
        else:
            return img[y:(y+dy), x:(x+dx), :], mask[y:(y+dy), x:(x+dx), :]


def calculate_z_filtering(data, mf_size=5):
    """Applies a median filtering in the z dimension of the provided data.

       Args:
            data (4D Numpy array): data to apply the filter to.
            E.g. (image_number, x, y, channels).

            mf_size (int, optional): size of the median filter. Must be an odd 
            number.

       Returns:
            out_data (4D Numpy array): data resulting from the application of   
            the median filter. E.g. (image_number, x, y, channels).
    """
   
    out_data = np.copy(data) 

    # Must be odd
    if mf_size % 2 == 0:
       mf_size += 1

    for i in range(0, data.shape[2]):
        sl = data[:, :, i, 0]     
        sl = cv2.medianBlur(sl, mf_size)
        out_data[:, :, i] = np.expand_dims(sl, axis=-1)
        
    return out_data


def elastic_transform(image, alpha, sigma, alpha_affine, seed=None):
    """Elastic deformation of images as described in [Simard2003]_ (with i
       modifications).
       [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
       Based on:
           https://gist.github.com/erniejunior/601cdf56d2b424757de5
       Code obtained from:
           https://www.kaggle.com/bguberfain/elastic-transform-for-data-augmentation
    """

    if seed is None:
        random_state = np.random.RandomState(None)
    else:
        random_state = np.random.RandomState(seed)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]
                       + square_size, center_square[1]-square_size],
                      center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine,
                                       size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1],
                           borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]),
                          np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), \
                         np.reshape(z, (-1, 1))
    map_ = map_coordinates(image, indices, order=1, mode='reflect')
    map_ = map_.reshape(shape)
    return map_


def fixed_dregee(image):
    """Rotate given image with a fixed degree

       Args:
            image (2D Numpy array): image to be rotated. E. g. (x, y).

       Returns:
            out_image (2D Numpy array): image rotated. E. g. (x, y).
    """

    img = np.array(image)

    # get image height, width
    (h, w) = img.shape[:2]
    # calculate the center of the image
    center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, 180, 1.0)
    out_image = cv2.warpAffine(img, M, (w, h))
    out_image = np.expand_dims(out_image, axis=-1)
    return out_image
