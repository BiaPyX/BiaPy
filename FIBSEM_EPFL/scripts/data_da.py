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


def load_data(train_path, train_mask_path, test_path, test_mask_path, 
              image_shape, create_val=True, val_split=0.1, seedValue=42):                                            
    """Load train, validation and test data from the given paths.       
                                                                        
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
    """Crop data into smaller pieces                                    
                                                                        
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
    """ Custom ImageDataGenerator based on 

        Based on:
            https://github.com/czbiohub/microDL 
            https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    """

    def __init__(self, X, Y, batch_size=32, dim=(256,256), n_channels=1, 
                 shuffle=False, seedValue=42, transform_prob=0.9, 
                 elastic_transform=False, vflip=False, hflip=False,
                 rotation=False):
        """ImageDataGenerator constructor.
                                                                                
       Args:                                                                    
            X (numpy array): train data.                                  
            Y (numpy array): train mask data.                             
            batch_size (int, optional): size of the batches.
            dim (tuple, optional): dimension of the desired images.
            n_channels (int, optional): number of channels of the input images.
            shuffle (bool, optional): to decide if the indexes will be shuffled
            after every epoch. 
            seedValue (int, optional): seed value.
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
        self.seed = seedValue
        self.transform_prob = transform_prob
        self.elastic_transform = elastic_transform
        self.vflip = vflip
        self.hflip = hflip
        self.rotation = rotation
        self.on_epoch_end()
    
         # Construct a list with the selected choices
        self.choices = []
        if elastic_transform == True:
            self.choices.append(0)
        if vflip == True:
            self.choices.append(1)
        if hflip == True:
            self.choices.append(2)
        if rotation == True:
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
                batch_x[i], batch_y[i] = self.apply_transform(self.X[j],
                                                              self.Y[j])
 
        return batch_x, batch_y

    def on_epoch_end(self):
        """ Updates indexes after each epoch. """
        self.indexes = np.arange(len(self.X))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def apply_transform(self, image, mask):
        prob = random.uniform(0, 1)
        if prob < self.transform_prob:
            # Select one of the transformations randomly
            transform_id = np.random.choice(self.choices, 1)

            # Elastic transformation
            if transform_id == 0:
                im_concat = np.concatenate((image, mask), axis=2)            
                im_concat_r = elastic_transform(im_concat, im_concat.shape[1]*2,
                                                im_concat.shape[1]*0.08,
                                                im_concat.shape[1]*0.08,
                                                seed=self.seed)

                trans_image = np.expand_dims(im_concat_r[...,0], axis=-1)
                trans_mask = np.expand_dims(im_concat_r[...,1], axis=-1)
            # Vertical flip
            elif transform_id == 1:
                trans_image = np.flip(image, 0)
                trans_mask = np.flip(mask, 0)
            # Horizontal flip
            elif transform_id == 2:
                trans_image = np.flip(image, 1)
                trans_mask = np.flip(mask, 1)
            # 90 degree rotation
            elif transform_id == 3:
                trans_image = np.rot90(image)
                trans_mask = np.rot90(mask)
            # 180 degree rotation
            elif transform_id == 4:
                trans_image = np.rot90(image, 2)
                trans_mask = np.rot90(mask, 2)
            # 270 degree rotation
            elif transform_id == 5:
                trans_image = np.rot90(image, 3)
                trans_mask = np.rot90(mask, 3)
            else:
                msg = str(transform_id) + " not in allowed aug_idx: 0-5"
                raise ValueError(msg)
            return trans_image, trans_mask
        else:
            return image, mask

