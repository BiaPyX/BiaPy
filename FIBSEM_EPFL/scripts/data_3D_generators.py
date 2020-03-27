import numpy as np
import keras
import random
import math
import os
from tqdm import tqdm
from skimage.io import imread
from util import array_to_img, img_to_array, save_img
from scipy.ndimage import rotate
from scipy.ndimage.interpolation import shift
from data_manipulation import img_to_onehot_encoding


class VoxelDataGeneratorFromDisk(keras.utils.Sequence):
    """Custom ImageDataGenerator for 3D images loaded from the disk.
    """

    def __init__(self, data_dir, mask_dir=None, dim=(256, 256, 128, 1), 
                 seed=42, batch_size=32, da=True, num_data_per_epoch=0, 
                 o_dim_vol=(750, 4096, 4096)):
        """ImageDataGenerator constructor.
                                                                                
       Args:                                                                    
            data_dir (str): path to the data directory. It should contain one 
            subdirectory per class as for flow_from_directory function of Keras.
            Only one class is supported. 

            mask_dir (str, optional): path to the mask directory. It should 
            contain one subdirectory per class as for flow_from_directory 
            function of Keras.  Only one class is supported.

            dim (tuple, optional): dimension of the desired images. 
            E. g. (x, y, z, channels).

            seed (int, optional): seed for random functions.

            batch_size (int, optional): size of the batches.

            da (bool, optional): to activate the data augmentation.

            num_data_per_epoch (int, optional): amount of data to process per 
            epoch. If the dataset has more data the rest is discarded. 

            o_dim_vol (tuple, optional): original dataset shape.
            E. g. (number_of_images, x, y)
        """

        classes = []
        for subdir in sorted(os.listdir(data_dir)):
            if os.path.isdir(os.path.join(data_dir, subdir)):
                classes.append(subdir)
        if mask_dir is not None:
            for subdir in sorted(os.listdir(mask_dir)):
                if os.path.isdir(os.path.join(mask_dir, subdir)):
                    classes.append(subdir)
        if len(classes) > 2:
            raise ValueError("More than one class detected on the provided " 
                             "data/mask directories. For this data " 
                             "augmentation only 1 is permitted.")
        if len(dim) != 4:
            raise ValueError("Dimension of the image stack must be 4, e.g. "
                             "(x, y, z, channels).")

        self.data_dir = os.path.join(data_dir, classes[0])
        if mask_dir is not None:
            self.mask_dir = os.path.join(mask_dir, classes[1])
        else:
            self.mask_dir = None

        if num_data_per_epoch != 0:
            if o_dim_vol[1]%dim[0] != 0 or o_dim_vol[2]%dim[1] != 0:
                raise ValueError("The original volume's shape %s must be a "
                                 "multiple of the requested dimension %s." 
                                 % (o_dim_vol, dim))

            print("The number of data per epoch to operate with must be near {}"
                  "in a {} volume".format(num_data_per_epoch, o_dim_vol))

            stacks_in_z = int(np.floor(o_dim_vol[0]/dim[2]))
            num_volumes = round(num_data_per_epoch/(stacks_in_z*dim[0]*dim[1]*dim[2]))

            print("As the stack shape is {}, the number of stacks that covers"
                  "the maximun depth of the original volumen is {}"\
                  .format(dim, stacks_in_z))
            print("The number of full depth stacks to process per epoch is {}"\
                  .format(num_volumes))
        
            c_per_row = o_dim_vol[1]/dim[0]
            c_per_col = o_dim_vol[2]/dim[1]
            c_total = c_per_row*c_per_col

            vol_start = 0
   
            # Initialize data indexes  
            self.o_data_indexes = []
            vol_data_lists = []
            for i in range(num_volumes):
                vol_data_lists.append([])
                
            for i, files in enumerate(sorted(next(os.walk(self.data_dir))[2])):
                v_num = int(i % c_total)
                
                if v_num >= vol_start and v_num < vol_start+num_volumes:             
                    vol_data_lists[v_num].append(files)
           
            for i in range(num_volumes):
                self.o_data_indexes.extend(vol_data_lists[i])
                
            # Initialize mask indexes
            if mask_dir is not None:
                self.o_mask_indexes = []
                vol_mask_lists = []
                for i in range(num_volumes):
                    vol_mask_lists.append([])

                for i, files in enumerate(sorted(next(os.walk(self.mask_dir))[2])):
                    v_num = int(i % c_total)
    
                    if v_num >= vol_start and v_num < vol_start+num_volumes:
                        vol_mask_lists[v_num].append(files)
    
                for i in range(num_volumes):
                    self.o_mask_indexes.extend(vol_mask_lists[i])
                
            else:
                self.o_mask_indexes = None

        else:
            self.o_data_indexes = sorted(next(os.walk(self.data_dir))[2])
            if mask_dir is not None:
                self.o_mask_indexes = sorted(next(os.walk(self.mask_dir))[2])
            else:
                self.o_mask_indexes = None

        self.n = len(self.o_data_indexes)
        self.seed = seed
        self.da = da
        self.dim = dim
        self.batch_size = batch_size
        self.total_batches_seen = 0
        self.num_images_to_form_stack = batch_size*dim[2]
  
        print("Detected {} data samples in {}".format(self.n, self.data_dir)) 
        if mask_dir is not None:
            print("Detected {} mask samples in {}".format(self.n, self.mask_dir)) 

        self.on_epoch_end()

    def __len__(self):
        """Defines the number of batches per epoch."""
    
        return int(np.floor(self.n/self.batch_size))

    def __getitem__(self, index):
        """Generation of one batch of data. 
           Args:
                index (int): batch index counter.
            
           Returns:
                batch_x (5D Numpy array): corresponding X elements of the batch.
                E. g. (batch_size, x, y, z, channels). 

                batch_y (5D Numpy array, optional): corresponding Y elements of the 
                batch. E. g. (batch_size, x, y, z, channels).
        """

        batch_x = np.zeros((self.batch_size,) + self.dim)
        d_indexes = self.data_indexes[index*self.num_images_to_form_stack:(index+1)*self.num_images_to_form_stack]
        img_stack = np.zeros((self.dim))

        if self.mask_dir is not None:
            batch_y = np.zeros((self.batch_size,) + self.dim)
            m_indexes = self.mask_indexes[index*self.num_images_to_form_stack:(index+1)*self.num_images_to_form_stack]
            mask_stack = np.zeros((self.dim))

        j = 0
        cont = 0
        for i, (d_ind, m_ind) in enumerate(zip(d_indexes, m_indexes)):
            img = imread(os.path.join(self.data_dir, d_ind))

            # Convert image into grayscale
            if len(img.shape) >= 3:
                img = img[:, :, 0]
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=-1)

            img_stack[:,:,j,:] = img

            if self.mask_dir is not None:
                mask = imread(os.path.join(self.mask_dir, m_ind))
                # Convert image into grayscale
                if len(mask.shape) >= 3:
                    mask = mask[:, :, 0]
                if len(mask.shape) == 2:
                    mask = np.expand_dims(mask, axis=-1)
                mask_stack[:,:,j,:] = mask

            j += 1
            
            # Add the 3D stack completed to the batch
            if j % self.dim[2] == 0:
                j = 0

                if self.da == True:
                # Make data augmentation
                    batch_x[cont] = img_stack
                    if self.mask_dir is not None:
                        batch_y[cont] = mask_stack
                else:
                    batch_x[cont] = img_stack
                    if self.mask_dir is not None:
                        batch_y[cont] = mask_stack

                cont += 1

        self.total_batches_seen += 1
    
        if self.mask_dir is not None:
            return batch_x, batch_y    
        else: 
            return batch_x

    def on_epoch_end(self):
        """Updates indexes after each epoch."""

        self.data_indexes = self.o_data_indexes.copy()
        if self.mask_dir is not None:
            self.mask_indexes = self.o_mask_indexes.copy()


class VoxelDataGenerator(keras.utils.Sequence):
    """Custom ImageDataGenerator for 3D images.
    """

    def __init__(self, X, Y, random_subvolumes_in_DA=True, seed=42, 
                 shuffle_each_epoch=False, batch_size=32, da=True, 
                 rotation_range=0, square_rotations=True, flip=True, 
                 shift_range=0, softmax_out=False):
        """ImageDataGenerator constructor.
                                                                                
       Args:                                                                    
            X (Numpy 4D array): data. E.g. (image_number, x, y, channels).

            Y (Numpy 4D array): mask data.  E.g. (image_number, x, y, channels).

            random_subvolumes_in_DA (bool, optional): flag to extract random 
            subvolumes from the given data. If not, the data must be 5D and is 
            assumed that the subvolumes are prepared. 
            
            seed (int, optional): seed for random functions.
                
            shuffle_each_epoch (bool, optional): flag to shuffle data after each 
            epoch.

            batch_size (int, optional): size of the batches.
            
            da (bool, optional): flag to activate the data augmentation.
            
            rotation_range (int, optional): degrees of rotation from 0. It must 
            be les equal 180. 
                
            square_rotations (bool, optional): flag to make square rotations of
            90º, -90º and 180º instead of using 'rotation_range'.
            
            flip (bool, optional): flag to activate flips.
        
            shift_range (float, optional): range to make a shift. It must be a 
            number between 0 and 1. 
        """

        if X.shape != Y.shape:
            raise ValueError("The shape of X and Y must be the same")
        if (X.ndim != 4 and X.ndim != 5) or (Y.ndim != 4 and Y.ndim != 5):
            raise ValueError("X and Y must be a 4D/5D Numpy array")
        if X.ndim == 5 and random_subvolumes_in_DA == True:
            raise ValueError("Data must be 4D to generate random subvolumes")
        if rotation_range > 180:
            raise ValueError("'rotation_range' must be a number between 0 and 180")
        if shift_range < 0 or shift_range > 1:
            raise ValueError("'shift_range' must be a float between 0 and 1")
        if square_rotations == True and rotation_range != 0:
            raise ValueError("'square_rotations' or 'rotation_range' can not be "
                             " selected at the same time")

        self.X = X/255 if np.max(X) > 1 else X
        self.Y = Y/255 if np.max(Y) > 1 else Y
        self.softmax_out = softmax_out
        self.random_subvolumes_in_DA = random_subvolumes_in_DA
        self.seed = seed
        self.shuffle_each_epoch = shuffle_each_epoch
        self.da = da
        self.batch_size = batch_size
        self.rotation_range = rotation_range
        self.square_rotations = square_rotations
        self.flip = flip
        self.shift_range = shift_range 
        self.total_batches_seen = 0
  
        self.on_epoch_end()

    def __len__(self):
        """Defines the number of batches per epoch."""
    
        return int(np.floor(self.X.shape[0]/self.batch_size))

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

        batch_x = np.zeros((self.batch_size, ) +  self.X.shape[1:])
        batch_y = np.zeros((self.batch_size, ) +  self.Y.shape[1:])
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        for i, j in zip(range(self.batch_size), indexes):
            if self.random_subvolumes_in_DA == True:
                # Random crop here
                print("Random crop here")
            else:
                im = np.copy(self.X[j])
                mask = np.copy(self.Y[j])

            if self.da == False:
                batch_x[i] = im
                batch_y[i] = mask
            else:
                batch_x[i], batch_y[i] = self.apply_transform(im, mask)

        if self.softmax_out == True:
            batch_y_ = np.zeros((self.batch_size, ) + self.Y.shape[1:4] + (2,))
            for i in range(self.batch_size):
                batch_y_[i] = np.asarray(img_to_onehot_encoding(batch_y[i]))

            batch_y = batch_y_

        self.total_batches_seen += 1
        return batch_x, batch_y    

    def on_epoch_end(self):
        """Updates indexes after each epoch."""

        self.indexes = np.arange(len(self.X))
        if self.shuffle_each_epoch == True:
            random.Random(self.seed + self.total_batches_seen).shuffle(self.indexes)

    def apply_transform(self, image, mask, grid=False):
        """Transform the input image and its mask at the same time with one of
           the selected choices based on a probability.
    
           Args:
                image (4D Numpy array): image to transform.
                E.g. (x, y, z, channels).

                mask (4D Numpy array): mask to transform.
                E.g. (x, y, z, channels).
    
           Returns:
                trans_image (4D Numpy array): transformed image.
                E.g. (x, y, z, channels).

                trans_mask (4D Numpy array): transformed image mask.
                E.g. (x, y, z, channels).
        """
            
        trans_image = np.copy(image)
        trans_mask = np.copy(mask)

        # [0-0.25): y axis flip
        # [0.25-0.5): z axis flip
        # [0.5-0.75): y and z axis flip
        # [0.75-1]: nothing
        #
        # y axis flip
        prob = random.uniform(0, 1)
        if self.flip == True and prob < 0.25:
            trans_image = np.flip(trans_image, 0)
            trans_mask = np.flip(trans_mask, 0)
        # z axis flip
        elif self.flip == True and 0.25 <= prob < 0.5:
            trans_image = np.flip(trans_image, 1)
            trans_mask = np.flip(trans_mask, 1)
        # y and z axis flip
        elif self.flip == True and 0.5 <= prob < 0.75:
            trans_image = np.flip(trans_image, 0)                               
            trans_mask = np.flip(trans_mask, 0)
            trans_image = np.flip(trans_image, 1)                               
            trans_mask = np.flip(trans_mask, 1)
       
        if self.square_rotations == False:
            # [0-0.25): x axis rotation
            # [0.25-0.5): y axis rotation
            # [0.5-0.75): z axis rotation
            # [0.75-1]: nothing
            prob = random.uniform(0, 1) 
            theta = np.random.uniform(-self.rotation_range, self.rotation_range)
            # x axis rotation
            if self.rotation_range != 0 and prob < 0.25:
                rotate(trans_image, axes=(2, 3), angle=theta, mode='reflect', 
                       reshape=False) 
                rotate(trans_mask, axes=(2, 3), angle=theta, mode='reflect', 
                       reshape=False)
            # y axis rotation
            elif self.rotation_range != 0 and 0.25 <= prob < 0.5:
                rotate(trans_image, axes=(3, 1), angle=theta, mode='reflect', 
                       reshape=False)
                rotate(trans_mask, axes=(3, 1), angle=theta, mode='reflect', 
                       reshape=False)
            # z axis rotation
            elif self.rotation_range != 0 and 0.5 <= prob < 0.75:
                rotate(trans_image, axes=(1, 2), angle=theta, mode='reflect', 
                       reshape=False)
                rotate(trans_mask, axes=(1, 2), angle=theta, mode='reflect', 
                       reshape=False)
        else:
            # [0-0.25): 90º rotation
            # [0.25-0.5): -90º rotation
            # [0.5-0.75): 180º rotation
            # [0.75-1]: nothing
            prob = random.uniform(0, 1)
            theta = np.random.uniform(-self.rotation_range, self.rotation_range)
            # 90º rotation on y axis
            if prob < 0.25:
                rotate(trans_image, axes=(3, 1), angle=90, mode='reflect',
                       reshape=False)
                rotate(trans_mask, axes=(3, 1), angle=90, mode='reflect',
                       reshape=False)
            # -90º rotation on y axis
            elif 0.25 <= prob < 0.5:
                rotate(trans_image, axes=(3, 1), angle=-90, mode='reflect',
                       reshape=False)
                rotate(trans_mask, axes=(3, 1), angle=-90, mode='reflect',
                       reshape=False)
            # 180º rotation on y axis
            elif 0.5 <= prob < 0.75:
                rotate(trans_image, axes=(3, 1), angle=180, mode='reflect',
                       reshape=False)
                rotate(trans_mask, axes=(3, 1), angle=180, mode='reflect',
                       reshape=False)

        # [0-0.25): x axis shift 
        # [0.25-0.5): y axis shift
        # [0.5-0.75): z axis shift 
        # [0.75-1]: nothing
        #
        # x axis shift 
        if self.shift_range != 0 and prob < 0.25:
            s = [0] * trans_image.ndim
            s[0] = math.floor(self.shift_range * trans_image.shape[0])
            shift(trans_image, shift=s, mode='reflect')
            shift(trans_mask, shift=s, mode='reflect')
        # y axis shift 
        elif self.shift_range != 0 and 0.25 <= prob < 0.5:                   
            s = [1] * trans_image.ndim                                          
            s[1] = math.floor(self.shift_range * trans_image.shape[1])          
            shift(trans_image, shift=s, mode='reflect')
            shift(trans_mask, shift=s, mode='reflect')
        # z axis shift
        elif self.shift_range != 0 and 0.5 <= prob < 0.75:                   
            s = [2] * trans_image.ndim                                          
            s[2] = math.floor(self.shift_range * trans_image.shape[2])          
            shift(trans_image, shift=s, mode='reflect')
            shift(trans_mask, shift=s, mode='reflect')

        return trans_image, trans_mask


    def get_transformed_samples(num_examples, random_images=True, 
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

        sample_x = np.zeros((num_examples, ) +  self.X.shape[1:])
        sample_y = np.zeros((num_examples, ) +  self.Y.shape[1:])

        # Generate the examples 
        print("0) Creating samples of data augmentation . . .")
        for i in tqdm(range(0,num_examples)):
            if random_images == True:
                pos = random.randint(1,self.X.shape[0]-1) 
            else:
                pos = cont 

            if self.da == False:
                sample_x[i] = self.X[pos]
                sample_y[i] = self.Y[pos]
            else:
                sample_x[i], sample_y[i] = self.apply_transform(self.X[pos], 
                                                                self.Y[pos])

        # Save transformed 3d volumes 
        if save_to_dir == True:
            save_img(X=sample_x, data_dir=out_dir, Y=sample_y, 
                     mask_dir=out_dir, prefix="aug_3d_smp")

        return sample_x, sample_y
