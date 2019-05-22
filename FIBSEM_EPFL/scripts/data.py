import numpy as np
import random
import pandas as pd
import os
import cv2
import keras
from tqdm import tqdm
from skimage.io import imread, imshow
from sklearn.model_selection import train_test_split
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from PIL import Image


def load_data(train_path, train_mask_path, test_path, test_mask_path, 
              image_shape, create_val=True, val_split=0.1, seedValue=42):                                            
    """ Load train, validation and test data from the given paths.       
                                                                        
       Args:                                                            
            train_path (str): path to the training data.                
            train_mask_path (str): path to the training data masks.     
            test_path (str): path to the test data.                     
            test_mask_path (str): path to the test data masks.          
            image_shape (array of 3 int): dimensions of the images.     
            create_val (bool, optional): if true validation data is created.                                                    
            val_split (float, optional): % of the train data used as    
            validation (value between o and 1).                         
            seedValue (int, optional): seed value.                      
                                                                        
       Returns:                                                         
            In case create_val == True:                                 
                X_train (numpy array): train images.                    
                Y_train (numpy array): train images' mask.              
                X_val (numpy array): validation images.                 
                Y_val (numpy array): validation images' mask.           
                X_test (numpy array): test images.                      
                Y_test (numpy array): test images' mask.                
            If not:                                                     
                X_train (numpy array): train images.                    
                Y_train (numpy array): train images' mask.              
                X_test (numpy array): test images.                      
                Y_test (numpy array): test images' mask.                
    """                                                                 
                                                                        
    train_ids = sorted(next(os.walk(train_path))[2])                    
    train_mask_ids = sorted(next(os.walk(train_mask_path))[2])          
                                                                        
    test_ids = sorted(next(os.walk(test_path))[2])                      
    test_mask_ids = sorted(next(os.walk(test_mask_path))[2])            
                                                                        
    # Get and resize train images and masks                             
    X_train = np.zeros((len(train_ids), image_shape[0], image_shape[1], 
                        image_shape[2]), dtype=np.uint8)                
    Y_train = np.zeros((len(train_mask_ids), image_shape[0], image_shape[1],
                        image_shape[2]), dtype=np.uint8) 
                                                                        
    print('Loading train images . . .')                                 
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):     
        img = imread(os.path.join(train_path, id_))                     
        img = np.expand_dims(img, axis=-1)                              
        X_train[n] = img                                                
                                                                        
    print('Loading train masks . . .')                                  
    for n, id_ in tqdm(enumerate(train_mask_ids), total=len(train_mask_ids)):                      
        mask = imread(os.path.join(train_mask_path, id_))               
        mask = np.expand_dims(mask, axis=-1)                            
        Y_train[n] = mask                                               
                                                                        
    Y_train = Y_train/255                                               
                                                                        
    # Get and resize test images and masks                              
    X_test = np.zeros((len(test_ids), image_shape[0], image_shape[1],   
                       image_shape[2]), dtype=np.uint8)                 
    Y_test = np.zeros((len(test_mask_ids), image_shape[0], image_shape[1],
                       image_shape[2]), dtype=np.uint8) 
                                                                        
    print('Loading test images . . .')                                  
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):       
        img = imread(os.path.join(test_path, id_))                      
        img = np.expand_dims(img, axis=-1)                              
        X_test[n] = img                                                 
                                                                        
    print('Loading test masks . . .')                                   
    for n, id_ in tqdm(enumerate(test_mask_ids), total=len(test_mask_ids)):                       
        mask = imread(os.path.join(test_mask_path, id_))                
        mask = np.expand_dims(mask, axis=-1)                            
        Y_test[n] = mask                                                
                                                                        
    Y_test = Y_test/255                                                 
                                                                        
    if (create_val == True):                                            
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train,
                                                          train_size=1-val_split,
                                                          test_size=val_split,
                                                          random_state=seedValue)      
 
        return X_train, Y_train, X_val, Y_val, X_test, Y_test           
    else:                                                               
        return X_train, Y_train, X_test, Y_test                         
   

def crop_data(data, data_mask, width, height):                          
    """ Crop data into smaller pieces                                    
                                                                        
       Args:                                                            
            data (4D numpy array): data to crop.                        
            data_mask (4D numpy array): data masks to crop.             
            width (str): output image width.                            
            height (str): output image height.                          
                                                                        
       Returns:                                                         
            cropped_data (4D numpy array): cropped data images.         
            cropped_data_mask (4D numpy array): cropped data masks.     
    """                                                                 
                                                                        
    print("Cropping [" + str(data.shape[1]) + ', ' + str(data.shape[2]) 
          + "] images into [" + str(width) + ', ' + str(height) + "] . . .")                                                  
                                                                        
    # Calculate the number of images to be generated                    
    h_num = int(data.shape[1] / width) + (data.shape[1] % width > 0)    
    v_num = int(data.shape[2] / height) + (data.shape[2] % height > 0)  
    total_cropped = data.shape[0]*h_num*v_num                           
                                                                        
    # Crop data                                                         
    cropped_data = np.zeros((total_cropped, width, height, data.shape[3]),
                            dtype=np.uint8)            
    cont=0                                                              
    for img_num in range(0, data.shape[0]):                             
        for i in range(0, h_num):                                       
            for j in range(0, v_num):                                   
                cropped_data[cont]= data[img_num, (i*width):((i+1)*height),      
                                         (j*width):((j+1)*height)]      
                cont=cont+1                                             
                                                                        
    # Crop mask data                                                    
    cropped_data_mask = np.zeros((total_cropped, width, height, data.shape[3]),
                                 dtype=np.uint8)       
    cont=0                                                              
    for img_num in range(0, data.shape[0]):                             
        for i in range(0, h_num):                                       
            for j in range(0, v_num):                                   
                cropped_data_mask[cont]= data_mask[img_num,             
                                                  (i*width):((i+1)*height),
                                                  (j*width):((j+1)*height)]
                cont=cont+1                                             
                                                                        
    return cropped_data, cropped_data_mask                              

                          
# Function to distort image
def elastic_transform(image, alpha, sigma, alpha_affine, seed=None):
    """ Elastic deformation of images as described in [Simard2003]_ (with i
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


class ImageDataGenerator(keras.utils.Sequence):
    """ Custom ImageDataGenerator.

        Based on:
            https://github.com/czbiohub/microDL 
            https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    """

    def __init__(self, X, Y, batch_size=32, dim=(256,256), n_channels=1, 
                 shuffle=False, transform_prob=0.9, elastic_transform=False,
                 vflip=False, hflip=False, rotation=False):
        """ ImageDataGenerator constructor.
                                                                                
       Args:                                                                    
            X (numpy array): train data.                                  
            Y (numpy array): train mask data.                             
            batch_size (int, optional): size of the batches.
            dim (tuple, optional): dimension of the desired images.
            n_channels (int, optional): number of channels of the input images.
            shuffle (bool, optional): to decide if the indexes will be shuffled
            after every epoch. 
            transform_prob (float, optional): value between 0 and 1 to determine 
            the probability of doing transformations to the original images. 
            elastic_transform (bool, optional): to make elastic transformations.
            vflip (bool, optional): if true vertical flip are made.
            hflip (bool, optional): if true horizontal flips are made.          
            rotation (bool, optional): to make rotations of 90º, 180º or 270º.
        """

        self.dim = dim
        self.batch_size = batch_size
        self.Y = Y
        self.X = X
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.transform_prob = transform_prob
        self.elastic_transform = elastic_transform
        self.vflip = vflip
        self.hflip = hflip
        self.rotation = rotation
        self.on_epoch_end()
    
        self.choices = []                                                       
        if self.elastic_transform == True:                                           
            self.choices.append(0)                                              
        if self.vflip == True:                                                       
            self.choices.append(1)                                              
        if self.hflip == True:                                                       
            self.choices.append(2)                                              
        if self.rotation == True:                                                    
            self.choices.append(3)                                              
            self.choices.append(4)                                              
            self.choices.append(5)

    def __len__(self):
        """ Defines the number of batches per epoch. """
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        """ Generation of one batch data. 

            Arg:
                index (int): batch index counter.
            
            Return:
                batch_x (numpy array): corresponding X elements of the batch.
                batch_y (numpy array): corresponding Y elements of the batch.
        """

        batch_x = np.empty((self.batch_size, *self.dim, self.n_channels))
        batch_y = np.empty((self.batch_size, *self.dim, self.n_channels))

        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        for i, j in zip( range(0,self.batch_size), indexes ):
            if len(self.choices) == 0: 
                 batch_x[i], batch_y[i] = self.X[j], self.Y[j]
            else:
                batch_x[i], batch_y[i], _ = self.apply_transform(self.X[j],
                                                              self.Y[j])
 
        return batch_x, batch_y

    def on_epoch_end(self):
        """ Updates indexes after each epoch. """
        self.indexes = np.arange(len(self.X))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __draw_grid(self, im, grid_width=50, m=False):
        """ Draw grid of the specified size on a image. 
           
            Arg:                                                                
                im (2D numpy array): image to be modified.
                grid_width (int, optional): grid's width. 
                m (bool, optional): advice the method to change the grid value
                if the input image is a mask.
        """

        if m == True:
            v = 1
        else:
            v = 255

        for i in range(0, im.shape[1], grid_width):
            im[:, i] = v
        for j in range(0, im.shape[0], grid_width):
            im[j, :] = v

    def apply_transform(self, image, mask, flow=False):
        """ Transform the input image and its mask at the same time with one of
            the selected choices based on a probability. 
                
            Args:
                image (2D numpy array): image to be transformed.
                mask (2D numpy array): image's mask.
                flow (bool, optional): forces the transform independetly of the
                previously selected probability. Also draws a grid in to the 
                elastic transfomations to visualize it clearly. Do not set this 
                option to train the network!
        """

        prob = random.uniform(0, 1)
        transform_string = 'none'

        if prob < self.transform_prob or flow == True:
            # Select one of the transformations randomly
            transform_id = np.random.choice(self.choices, 1)

            # Elastic transformation
            if transform_id == 0:
                
                if flow == True:
                    self.__draw_grid(image)
                    self.__draw_grid(mask, m=True)

                im_concat = np.concatenate((image, mask), axis=2)            

                im_concat_r = elastic_transform(im_concat, im_concat.shape[1]*2,
                                                im_concat.shape[1]*0.08,
                                                im_concat.shape[1]*0.08)

                trans_image = np.expand_dims(im_concat_r[...,0], axis=-1)
                trans_mask = np.expand_dims(im_concat_r[...,1], axis=-1)
                transform_string = 'elastic'
            # Vertical flip
            elif transform_id == 1:
                trans_image = np.flip(image, 0)
                trans_mask = np.flip(mask, 0)
                transform_string = 'vflip'
            # Horizontal flip
            elif transform_id == 2:
                trans_image = np.flip(image, 1)
                trans_mask = np.flip(mask, 1)
                transform_string = 'hflip'
            # 90 degree rotation
            elif transform_id == 3:
                trans_image = np.rot90(image)
                trans_mask = np.rot90(mask)
                transform_string = 'r90'
            # 180 degree rotation
            elif transform_id == 4:
                trans_image = np.rot90(image, 2)
                trans_mask = np.rot90(mask, 2)
                transform_string = 'r180'
            # 270 degree rotation
            elif transform_id == 5:
                trans_image = np.rot90(image, 3)
                trans_mask = np.rot90(mask, 3)
                transform_string = 'r270'
            else:
                msg = str(transform_id) + " not in allowed aug_idx: 0-5"
                raise ValueError(msg)
            return trans_image, trans_mask, transform_string
        else:
            return image, mask, transform_string


    def flow_on_examples(self, num_examples, job_id="none_job_id", 
                         save_to_dir='aug', save_prefix=None,
                         original_elastic=True, random_images=True):
        """ Apply selected transformations to a defined number of images from
            the dataset. The purpose of this method is to check the images 
            generated by data augmentation. 
            
            Args:
                num_examples (int): number of examples to generate.
                job_id (str, optional): job identifier. If any provided the
                examples will be generated under a folder called 'aug/none_job_id'.
                save_to_dir (str, optional): name of the folder where the 
                examples will be stored. If any provided the examples will be 
                generated under a folder called 'aug/none_job_id'.
                save_prefix (str, optional): prefix to add to the generated 
                examples' name. 
                original_elastic (bool, optional): to save also the original
                images when an elastic transformation is performed.
                random_images (bool, optional): randomly select images from the
                dataset. If False the examples will be generated from the start
                of the dataset. 
        """

        prefix = ""
        if save_prefix != None: 
            prefix = str(save_prefix)

        save_dir = os.path.join(save_to_dir, job_id) 
        if not os.path.exists(save_dir):                              
            os.makedirs(save_dir)

        # Generate the examples 
        for i in range(0,num_examples):
            if random_images == True:
                pos = random.randint(1,self.X.shape[0]) 
            else:
                pos = cont 

            im = self.X[pos]
            mask = self.Y[pos]

            out_im, out_mask, t_st = self.apply_transform(im, mask, flow=True)

            out_im = Image.fromarray(out_im[:,:,0])                           
            out_im = out_im.convert('L')                                                    
            out_im.save(os.path.join(save_dir, prefix + 'x_' + str(pos) + '_' 
                                     + t_st + ".png"))          
                 
            out_mask = Image.fromarray(out_mask[:,:,0]*255)                           
            out_mask = out_mask.convert('L')                                                    
            out_mask.save(os.path.join(save_dir, prefix + 'y_' + str(pos) + '_' 
                                       + t_st + ".png"))          
                 
            if original_elastic == True and t_st == 'elastic': 
                im = Image.fromarray(im[:,:,0])
                im = im.convert('L')
                im.save(os.path.join(save_dir, prefix + 'x_' + str(pos) + '_'
                                     + t_st + '_original.png'))

                mask = Image.fromarray(mask[:,:,0]*255)
                mask = mask.convert('L')
                mask.save(os.path.join(save_dir, prefix + 'y_' + str(pos) + '_'
                                       + t_st + '_original.png'))

