import numpy as np
import random
import os
import cv2
import keras
import sys
import math
from tqdm import tqdm
from skimage.io import imread
from sklearn.model_selection import train_test_split
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy import ndimage
from PIL import Image
from texttable import Texttable
from keras.preprocessing.image import ImageDataGenerator as kerasDA
from util import Print

def load_data(train_path, train_mask_path, test_path, test_mask_path, 
              image_train_shape, image_test_shape, create_val=True, 
              val_split=0.1, shuffle_val=True, seedValue=42, numOutputChannels=1,
              extra_train_data=0):
    """Load train, validation and test data from the given paths. If the images 
       to be loaded are smaller than the given dimension it will be sticked in 
       the (0, 0).
                                                                        
       Args:                                                            
            train_path (str): path to the training data.                
            train_mask_path (str): path to the training data masks.     
            test_path (str): path to the test data.                     
            test_mask_path (str): path to the test data masks.          
            image_train_shape (array of 3 int): dimensions of the images.     
            image_test_shape (array of 3 int): dimensions of the images.     
            create_val (bool, optional): if true validation data is created.                                                    
            val_split (float, optional): % of the train data used as    
            validation (value between 0 and 1).
            seedValue (int, optional): seed value.
            shuffle_val (bool, optional): take random training examples to      
            create validation data.
            numOutputChannels (int, optional): number of output channels.
            extra_train_data (int, optional): number of training extra data to 
            be created. Only used to increase the data arrays' size. 
                                                                        
       Returns:                                                         
                X_train (numpy array): train images.                    
                Y_train (numpy array): train images' mask.              
                X_val (numpy array, optional): validation images 
                (create_val==True).
                Y_val (numpy array, optional): validation images' mask 
                (create_val==True).
                X_test (numpy array): test images.                      
                Y_test (numpy array): test images' mask.                
    """      
    
    Print("Loading images . . .")
                                                                        
    train_ids = sorted(next(os.walk(train_path))[2])                    
    train_mask_ids = sorted(next(os.walk(train_mask_path))[2])          
                                                                        
    test_ids = sorted(next(os.walk(test_path))[2])                      
    test_mask_ids = sorted(next(os.walk(test_mask_path))[2])            
                                                                        
    # Get and resize train images and masks                             
    X_train = np.zeros((len(train_ids)+extra_train_data, image_train_shape[1], 
                        image_train_shape[0], image_train_shape[2]),
                        dtype=np.int16)                
    Y_train = np.zeros((len(train_mask_ids)+extra_train_data, image_train_shape[1], 
                        image_train_shape[0], numOutputChannels ),
                        dtype=np.int16) 
                                                                        
    Print("[LOAD] Loading train images . . .") 
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):     
        img = imread(os.path.join(train_path, id_))                     
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        X_train[n,:,:,:] = img

    Print('[LOAD] Loading train masks . . .')
    for n, id_ in tqdm(enumerate(train_mask_ids), total=len(train_mask_ids)):                      
        mask = imread(os.path.join(train_mask_path, id_))               
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=-1)
        Y_train[n,:,:,:] = mask
                                                                        
    Y_train = Y_train/255                                               
    
    # Get and resize test images and masks                              
    X_test = np.zeros((len(test_ids)+extra_train_data, image_test_shape[1], image_test_shape[0],   
                       image_test_shape[2]), dtype=np.int16)                 
    Y_test = np.zeros((len(test_mask_ids)+extra_train_data, image_test_shape[1], 
                       image_test_shape[0], numOutputChannels ), dtype=np.int16)
                                                                        
    Print("[LOAD] Loading test images . . .")
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):       
        img = imread(os.path.join(test_path, id_))                      
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        X_test[n,:,:,:] = img

    Print("[LOAD] Loading test masks . . .")
    for n, id_ in tqdm(enumerate(test_mask_ids), total=len(test_mask_ids)):                       
        mask = imread(os.path.join(test_mask_path, id_))                
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=-1)
        Y_test[n,:,:,:] = mask
                                                                        
    Y_test = Y_test/255                                                 
    
    norm_value = np.mean(X_train)
 
    if create_val == True:                                            
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train,
                                                          test_size=val_split,
                                                          shuffle=shuffle_val,
                                                          random_state=seedValue)      

        Print("[LOAD] Loaded train data shape is: " + str(X_train.shape))
        Print("[LOAD] Loaded test data shape is: " + str(X_test.shape))
        Print("[LOAD] Loaded validation data shape is: " + str(X_val.shape))

        return X_train, Y_train, X_val, Y_val, X_test, Y_test, norm_value
    else:                                                               
        Print("[LOAD] Loaded train data shape is: " + str(X_train.shape))
        Print("[LOAD] Loaded test data shape is: " + str(X_test.shape))

        return X_train, Y_train, X_test, Y_test, norm_value                         


def __foreground_percentage(mask, class_tag=1):
    """ Percentage of pixels that corresponds to the class in the given image.
        
        Args: 
            mask (numpy 2D array): image mask to analize.
            class_tag (int, optional): class to find in the image.
        Return:
            float: percentage of pixels that corresponds to the class. Value
            between 0 and 1.
    """

    c = 0
    for i in range(0, mask.shape[0]):
        for j in range(0, mask.shape[1]):     
            if mask[i][j] == class_tag:
                c = c + 1

    return (c*100)/(mask.shape[0]*mask.shape[1])


def crop_data(data, data_mask, width, height, force_shape=[0, 0], discard=False,
              d_percentage=0):                          
    """Crop data into smaller pieces.
                                                                        
       Args:                                                            
            data (4D numpy array): data to crop.                        
            data_mask (4D numpy array): data masks to crop.             
            width (str): output image width.
            height (str): output image height.
            force_shape (int tuple, optional): force horizontal and vertical 
            crops to the given numbers.
            d_percentage (int, optional): number between 0 and 100. The images 
            that have less foreground pixels than the given number will be 
            discarded.
                                                                        
       Returns:                                                         
            cropped_data (4D numpy array): cropped data images.         
            cropped_data_mask (4D numpy array): cropped data masks.     
            force_shape (int tuple): number of horizontal and vertical crops 
            made. Useful for future crop calls. 
    """                                                                 
                                                                        
    Print("Cropping [" + str(data.shape[1]) + ', ' + str(data.shape[2]) 
          + "] images into [" + str(width) + ', ' + str(height) + "] . . .")
    
    # Calculate the number of images to be generated                    
    if force_shape == [0, 0]:
        h_num = int(data.shape[1] / width) + (data.shape[1] % width > 0)
        v_num = int(data.shape[2] / height) + (data.shape[2] % height > 0)
        force_shape = [h_num, v_num]
    else:
        h_num = force_shape[0]
        v_num = force_shape[1]
        Print("[CROP] Force crops to [" + str(h_num) + ", " + str(v_num) + "]")

    total_cropped = data.shape[0]*h_num*v_num    

    # Resize data to adjust to a value divisible by height x width
    r_data = np.zeros((data.shape[0], h_num*height, v_num*width, data.shape[3]),      
                      dtype=np.int16)    
    r_data[:data.shape[0],:data.shape[1],:data.shape[2],:data.shape[3]] = data
    r_data_mask = np.zeros((data_mask.shape[0], h_num*height, v_num*width,
                            data_mask.shape[3]), dtype=np.int16)
    r_data_mask[:data_mask.shape[0],:data_mask.shape[1],
                :data_mask.shape[2],:data_mask.shape[3]] = data_mask
    if data.shape != r_data.shape:
        Print("[CROP] Resized data from " + str(data.shape) + " to " 
              + str(r_data.shape) + " to be divisible by the shape provided")

    discarded = 0                                                                    
    cont = 0
    selected_images  = []

    # Discard images from the data set
    if discard == True:
        Print("[CROP] Selecting images to discard . . .")
        for img_num in tqdm(range(0, r_data.shape[0])):                             
            for i in range(0, h_num):                                       
                for j in range(0, v_num):
                    p = __foreground_percentage(r_data_mask[img_num,
                                                            (i*width):((i+1)*height),
                                                            (j*width):((j+1)*height)])
                    if p > d_percentage: 
                        selected_images.append(cont)
                    else:
                        discarded = discarded + 1

                    cont = cont + 1

    # Crop data                                                         
    cropped_data = np.zeros(((total_cropped-discarded), height, width,     
                             r_data.shape[3]), dtype=np.int16)
    cropped_data_mask = np.zeros(((total_cropped-discarded), height, width, 
                                  r_data_mask.shape[3]), dtype=np.int16)
    
    cont = 0                                                              
    l_i = 0
    Print("[CROP] Cropping images . . .")
    for img_num in tqdm(range(0, r_data.shape[0])): 
        for i in range(0, h_num):                                       
            for j in range(0, v_num):                     
                if discard == True and len(selected_images) != 0:
                    if selected_images[l_i] == cont \
                       or l_i == len(selected_images) - 1:

                        cropped_data[l_i]= r_data[img_num, 
                                                  (i*width):((i+1)*height), 
                                                  (j*width):((j+1)*height),:]

                        cropped_data_mask[l_i]= r_data_mask[img_num,                 
                                                            (i*width):((i+1)*height),
                                                            (j*width):((j+1)*height),:]

                        if l_i != len(selected_images) - 1:
                            l_i = l_i + 1
                else: 
              
                    cropped_data[cont]= r_data[img_num, (i*width):((i+1)*height),      
                                               (j*width):((j+1)*height),:]
                                                                        
                    cropped_data_mask[cont]= r_data_mask[img_num,             
                                                         (i*width):((i+1)*height),
                                                         (j*width):((j+1)*height),:]
                cont = cont + 1                                             
                                                                        
    if discard == True:
        Print("[CROP] " + str(discarded) + " images discarded. New shape after " 
              + "cropping and discarding is " + str(cropped_data.shape))
    else:
        Print("[CROP] New data shape is: " + str(cropped_data.shape))
    Print("[CROP] New mask data shape is: " + str(cropped_data_mask.shape))

    return cropped_data, cropped_data_mask, force_shape


def crop_data_with_overlap(data, data_mask, window_size, subdivision):
    """Crop data into smaller pieces with the minimun overlap.

       Args:
            data (4D numpy array): data to crop.
            data_mask (4D numpy array): data mask to crop.
            window_size (int): crop size .
            subdivision (int): number of crops to create.

       Returns:
            cropped_data (4D numpy array): cropped image data.
            cropped_data_mask (4D numpy array): cropped image data masks.
    """

    Print("[OV-CROP] Cropping [" + str(data.shape[1]) + ', ' + str(data.shape[2])
          + "] images into [" + str(window_size) + ', ' + str(window_size)
          + "] with overlapping. . .")

    assert (subdivision % 2 == 0 or subdivision == 1), "Error: " \
            + "subdivision must be 1 or an even number" 
    assert window_size <= data.shape[1], "Error: window_size " \
            + str(window_size) + " greater than data width " \
            + str(data.shape[1])
    assert window_size <= data.shape[2], "Error: window_size " \
            + str(window_size) + " greater than data height " \
            + str(data.shape[2])

    # Crop data
    total_cropped = data.shape[0]*subdivision
    cropped_data = np.zeros((total_cropped, window_size, window_size,
                             data.shape[3]), dtype=np.int16)
    cropped_data_mask = np.zeros((total_cropped, window_size, window_size,
                             data.shape[3]), dtype=np.int16)

    # Find the mininum overlap configuration with the number of crops to create
    min_d = sys.maxsize
    rows = 1
    columns = 1
    for i in range(1, int(subdivision/2)+1, 1):
        if subdivision % i == 0 and abs((subdivision/i) - i) < min_d:
            min_d = abs((subdivision/i) - i)
            rows = i
            columns = int(subdivision/i)
        
    Print("[OV-CROP] The minimum overlap has been found with rows=" + str(rows) \
          + " and columns=" + str(columns))

    # Calculate the amount of overlap, the division remainder to obtain an 
    # offset to adjust the last crop and the step size. All of this values per
    # x/y or column/row axis
    if rows != 1:
        y_ov = int(abs(data.shape[1] - window_size*rows)/(rows-1))
        r_y = abs(data.shape[1] - window_size*rows) % (rows-1) 
        step_y = window_size - y_ov
    else:
        y_ov = 0
        r_y = 0
        step_y = data.shape[1]

    if columns != 1:
        x_ov = int(abs(data.shape[2] - window_size*columns)/(columns-1))
        r_x = abs(data.shape[2] - window_size*columns) % (columns-1) 
        step_x = window_size - x_ov
    else:
        x_ov = 0
        r_x = 0
        step_x = data.shape[2]

    # Create the crops
    cont = 0
    for k, img_num in tqdm(enumerate(range(0, data.shape[0]))):
        for i in range(0, data.shape[1]-y_ov, step_y):
            for j in range(0, data.shape[2]-x_ov, step_x):
                d_y = 0 if (i+window_size) < data.shape[1] else r_y
                d_x = 0 if (j+window_size) < data.shape[2] else r_x

                cropped_data[cont] = data[k, i-d_y:i+window_size, j-d_x:j+window_size, :]
                cropped_data_mask[cont] = data_mask[k, i-d_y:i+window_size, j-d_x:j+window_size, :]
                cont = cont + 1

    Print("[OV-CROP] New data shape is: " + str(cropped_data.shape))

    return cropped_data, cropped_data_mask


def merge_data_with_overlap(data, original_shape, window_size, subdivision, 
                            out_dir, ov_map=True, ov_data_img=0):
    """Merge data with an amount of overlap. Used to undo the crop made by the 
       function crop_data_with_overlap.

       Args:
            data (4D numpy array): data to merge.
            original_shape (tuple): original dimensions to reconstruct. 
            window_size (int): crop size.
            subdivision (int): number of crops to merge.
            out_dir (string): directory where the images will be save.
            ov_map (bool, optional): whether to create overlap map.
            ov_data_img (int, optional): number of the image on the data to 
            create the overlappng map.

       Returns:
            merged_data (4D numpy array): merged image data.
    """

    Print("[OV-MERGE] Merging [" + str(data.shape[1]) + ', ' + str(data.shape[2])
          + "] images into [" + str(original_shape[1]) + ", " 
          + str(original_shape[0]) + "] with overlapping . . .")

    # Merged data
    total_images = int(data.shape[0]/subdivision)
    merged_data = np.zeros((total_images, original_shape[1], original_shape[0],
                             data.shape[3]), dtype=np.int16)

    # Matrices to store the amount of overlap. The first is used to store the
    # number of crops to merge for each pixel. The second matrix is used to 
    # paint the overlapping map
    overlap_matrix = np.zeros((original_shape[1], original_shape[0],
                             data.shape[3]), dtype=np.int16)
    if ov_map == True:
        ov_map_matrix = np.zeros((original_shape[1], original_shape[0],
                                   data.shape[3]), dtype=np.int16)

    # Find the mininum overlap configuration with the number of crops to create
    min_d = sys.maxsize
    rows = 1
    columns = 1
    for i in range(1, int(subdivision/2)+1, 1):
        if subdivision % i == 0 and abs((subdivision/i) - i) < min_d:
            min_d = abs((subdivision/i) - i)
            rows = i
            columns = int(subdivision/i)

    Print("[OV-MERGE] The minimum overlap has been found with [" + str(rows) \
          + ", " + str(columns) + "]")

    # Calculate the amount of overlap, the division remainder to obtain an
    # offset to adjust the last crop and the step size. All of this values per
    # x/y or column/row axis
    if rows != 1:
        y_ov = int(abs(original_shape[1] - window_size*rows)/(rows-1))
        r_y = abs(original_shape[1] - window_size*rows) % (rows-1)
        step_y = window_size - y_ov
    else:
        y_ov = 0
        r_y = 0
        step_y = original_shape[1]

    if columns != 1:
        x_ov = int(abs(original_shape[0] - window_size*columns)/(columns-1))
        r_x = abs(original_shape[0] - window_size*columns) % (columns-1)
        step_x = window_size - x_ov
    else:
        x_ov = 0
        r_x = 0
        step_x = original_shape[0]

    # Calculate the overlapping matrix
    for i in range(0, original_shape[1]-y_ov, step_y):
        for j in range(0, original_shape[0]-x_ov, step_x):
            d_y = 0 if (i+window_size) < original_shape[1] else r_y
            d_x = 0 if (j+window_size) < original_shape[0] else r_x

            overlap_matrix[i-d_y:i+window_size, j-d_x:j+window_size, :] += 1
            if ov_map == True:
                ov_map_matrix[i-d_y:i+window_size, j-d_x:j+window_size, :] += 1

    # Mark the border of each crop in the map
    if ov_map == True:
        for i in range(0, original_shape[1]-y_ov, step_y):
            for j in range(0, original_shape[0]-x_ov, step_x):
                d_y = 0 if (i+window_size) < original_shape[1] else r_y
                d_x = 0 if (j+window_size) < original_shape[0] else r_x
                
                # Paint the grid
                ov_map_matrix[i-d_y:(i+window_size-1), j-d_x] = -4 
                ov_map_matrix[i-d_y:(i+window_size-1), (j+window_size-1-d_x)] = -4 
                ov_map_matrix[i-d_y, j-d_x:(j+window_size-1)] = -4 
                ov_map_matrix[(i+window_size-1-d_y), j-d_x:(j+window_size-1)] = -4 
  
    # Merge the overlapping crops
    cont = 0
    for k, img_num in tqdm(enumerate(range(0, total_images))):
        for i in range(0, original_shape[1]-y_ov, step_y):
            for j in range(0, original_shape[0]-x_ov, step_x):
                d_y = 0 if (i+window_size) < original_shape[1] else r_y
                d_x = 0 if (j+window_size) < original_shape[0] else r_x
                merged_data[k, i-d_y:i+window_size, j-d_x:j+window_size, :] += data[cont]
                cont += 1
           
        merged_data[k] = np.true_divide(merged_data[k], overlap_matrix)

    # Save a copy of the merged data with the overlapped regions colored as: 
    # green when 2 crops overlap, yellow when (2 < x < 8) and red when more than 
    # 7 overlaps are merged 
    if ov_map == True:
        ov_map_matrix[ np.where(ov_map_matrix >= 8) ] = -1
        ov_map_matrix[ np.where(ov_map_matrix >= 3) ] = -2
        ov_map_matrix[ np.where(ov_map_matrix >= 2) ] = -3

        im = Image.fromarray(merged_data[ov_data_img,:,:,0]*255)
        im = im.convert('RGB')
        px = im.load()
        width, height = im.size
        for im_i in range(width): 
            for im_j in range(height):
                # White borders
                if ov_map_matrix[im_j, im_i] == -4: 
                    # White
                    px[im_i, im_j] = (255, 255, 255)

                # 2 overlaps
                elif ov_map_matrix[im_j, im_i] == -3: 
                    if merged_data[ov_data_img, im_j, im_i, 0] == 1:
                        # White + green
                        px[im_i, im_j] = (73, 100, 73)
                    else:
                        # Black + green
                        px[im_i, im_j] = (0, 74, 0)

                # 2 < x < 8 overlaps
                elif ov_map_matrix[im_j, im_i] == -2:
                    if merged_data[ov_data_img, im_j, im_i, 0] == 1:
                        # White + yellow
                        px[im_i, im_j] = (100, 100, 73)
                    else:
                        # Black + yellow
                        px[im_i, im_j] = (74, 74, 0)

                # 8 >= overlaps
                elif ov_map_matrix[im_j, im_i] == -1:
                    if merged_data[ov_data_img, im_j, im_i, 0] == 1:
                        # White + red
                        px[im_i, im_j] = (100, 73, 73)
                    else:
                        # Black + red
                        px[im_i, im_j] = (74, 0, 0)

        im.save(os.path.join(out_dir,"merged_ov_map.png"))
  
    Print("[MERGE-OV-CROP] New data shape is: " + str(merged_data.shape))

    return merged_data


def mix_data(data, num, out_shape=[1, 1], grid=True):
    """Combine images from input data into a bigger one given shape. It is the 
       opposite function of crop_data().

       Args:                                                                    
            data (4D numpy array): data to crop.                                
            num (int, optional): number of examples to convert.
            out_shape (int tuple, optional): number of horizontal and vertical
            images to combine in a single one.
            grid (bool, optional): make the grid in the output image.
                                                                                
       Returns:                                                                 
            mixed_data (4D numpy array): mixed data images.                 
            mixed_data_mask (4D numpy array): mixed data masks.
    """

    if grid == True:
        if np.max(data) > 1:
            v = 255
        else:
            v = 1

    width = data.shape[1]
    height = data.shape[2] 

    # Mix data
    mixed_data = np.zeros((num, out_shape[1]*width, out_shape[0]*height, data.shape[3]),
                          dtype=np.int16)
    cont = 0
    for img_num in tqdm(range(0, num)):
        for i in range(0, out_shape[1]):
            for j in range(0, out_shape[0]):
                
                if cont == data.shape[0]:
                    return mixed_data

                mixed_data[img_num, (i*width):((i+1)*height), 
                           (j*width):((j+1)*height)] = data[cont]
                
                if grid == True:
                    mixed_data[img_num,(i*width):((i+1)*height)-1,
                              (j*width)] = v
                    mixed_data[img_num,(i*width):((i+1)*height)-1,
                              ((j+1)*width)-1] = v
                    mixed_data[img_num,(i*height),
                              (j*width):((j+1)*height)-1] = v
                    mixed_data[img_num,((i+1)*height)-1,
                              (j*width):((j+1)*height)-1] = v
                cont = cont + 1

    return mixed_data


def check_crops(data, out_dim, num_examples=2, include_crops=True,
                out_dir="check_crops", job_id="none_job_id", suffix="_none_", 
                grid=True):
    """Check cropped images by the function crop_data(). 
        
       Args:
            data (4D numpy array): data to crop.
            out_dim (int 2D tuple): width and height of the image to be 
            constructed.
            num_examples (int, optional): number of examples to create.
            include_crops (bool, optional): to save cropped images or only the 
            image to contruct.  
            out_dir (string, optional): directory where the images will be save.
            job_id (str, optional): job identifier. If any provided the
            examples will be generated under a folder 'out_dir/none_job_id'.
            suffix (string, optional): suffix to add in image names. 
            grid (bool, optional): make the grid in the output image.
    """
   
    # First checks
    if out_dim[0] < data.shape[1] or out_dim[1] < data.shape[2]:
        Print("[C_CROP] Aborting: out_dim must be equal or greater than" 
              + "data.shape")
        return
    out_dir = os.path.join(out_dir, job_id)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # For mask data
    if np.max(data) > 1:
        v = 1
    else:
        v = 255
   
    # Calculate horizontal and vertical image number for the data
    h_num = int(out_dim[0] / data.shape[1]) + (out_dim[0] % data.shape[1] > 0)
    v_num = int(out_dim[1] / data.shape[2]) + (out_dim[1] % data.shape[2] > 0)
    total = h_num*v_num

    if total*num_examples > data.shape[0]:
        num_examples = math.ceil(data.shape[0]/total)
        total = num_examples
        Print("[CHECK_CROP] Requested num_examples too high for data. Set " 
              + "automatically to " + str(num_examples))
    else:
        total = total*num_examples

    if include_crops == True:
        Print("[CHECK_CROP] Saving cropped data images . . .")
        for i in tqdm(range(0, total)):
            # grayscale images
            if data.shape[3] == 1:
                im = Image.fromarray(data[i,:,:,0]*v)
                im = im.convert('L')
            # RGB images
            else:
                aux = np.asarray( data[i,:,:,:]*v, dtype="uint8" )
                im = Image.fromarray( aux, 'RGB' )

            im.save(os.path.join(out_dir,"c_" + suffix + str(i) + ".png"))

    Print("[CHECK_CROP] Obtaining " + str(num_examples) + " images of ["
          + str(data.shape[1]*h_num) + "," + str(data.shape[2]*v_num) + "] from ["
          + str(data.shape[1]) + "," + str(data.shape[2]) + "]")
    m_data = mix_data(data, num_examples, out_shape=[h_num, v_num], grid=grid) 
    
    Print("[CHECK_CROP] Saving data mixed images . . .")
    for i in tqdm(range(0, num_examples)):
        im = Image.fromarray(m_data[i,:,:,0]*v)
        im = im.convert('L')
        im.save(os.path.join(out_dir,"f" + suffix + str(i) + ".png"))


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


class ImageDataGenerator(keras.utils.Sequence):
    """Custom ImageDataGenerator.
       Based on:
           https://github.com/czbiohub/microDL 
           https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    """

    def __init__(self, X, Y, batch_size=32, dim=(256,256), n_channels=1, 
                 shuffle=False, da=True, e_prob=0.0, elastic=False, vflip=False,
                 hflip=False, rotation90=False, rotation_range=0.0, 
                 crops_before_DA=False, crop_length=0, val=False):
        """ImageDataGenerator constructor.
                                                                                
       Args:                                                                    
            X (numpy array): data.                                  
            Y (numpy array): mask data.                             
            batch_size (int, optional): size of the batches.
            dim (tuple, optional): dimension of the desired images. As no effect 
            if crops_before_DA is active, as the dimension will be selected by 
            that variable instead.
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
            crop_after_DA (bool, optional): decide to make random crops after
            apply DA transformations.
            crop_length (int, optional): length of the random crop after DA.
            val (bool, optional): advice the generator that the images will be
            to validate the model to not make random crops (as the val. data must
            be the same on each epoch).
        """

        self.dim = dim
        self.batch_size = batch_size
        self.Y = Y
        self.X = X
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.da = da
        self.e_prob = e_prob
        self.elastic = elastic
        self.vflip = vflip
        self.hflip = hflip
        self.rotation90 = rotation90
        self.rotation_range = rotation_range
        self.crops_before_DA = crops_before_DA
        self.crop_length = crop_length
        self.val = val
        self.on_epoch_end()
        
        if self.X.shape[1] == self.X.shape[2] or self.crops_before_DA == True:
            self.squared = True
        else:
            self.squared = False
            if rotation90 == True:
                Print("[AUG] Images not square, only 180 rotations will be done.")

        # Create a list which will hold a counter of the number of times a 
        # transformation is performed. 
        self.t_counter = [0 ,0 ,0 ,0 ,0 ,0] 

        if self.crops_before_DA == True:
            self.dim = (self.crop_length, self.crop_length)

    def __len__(self):
        """Defines the number of batches per epoch."""
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        """Generation of one batch data. 
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

        for i, j in zip(range(0,self.batch_size), indexes):
            if self.da == False: 
                if self.crops_before_DA == True:
                    batch_x[i], batch_y[i] = random_crop(self.X[j], self.Y[j], 
                                                         (self.crop_length, self.crop_length),
                                                         self.val)
                else:
                    batch_x[i], batch_y[i] = self.X[j], self.Y[j]
            else:
                if self.crops_before_DA == True:
                    batch_x[i], batch_y[i] = random_crop(self.X[j], self.Y[j],
                                                         (self.crop_length, self.crop_length), 
                                                         self.val) 
                    batch_x[i], batch_y[i], _ = self.apply_transform(batch_x[i],
                                                                     batch_y[i])
                else:
                    batch_x[i], batch_y[i], _ = self.apply_transform(self.X[j],
                                                                     self.Y[j])
                
 
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

    def __draw_grid(self, im, grid_width=50, m=False):
        """Draw grid of the specified size on an image. 
           
           Args:                                                                
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
        """Transform the input image and its mask at the same time with one of
           the selected choices based on a probability. 
                
           Args:
               image (2D numpy array): image to be transformed.
               mask (2D numpy array): image's mask.
               flow (bool, optional): forces the transform independetly of the
               previously selected probability. Also draws a grid in to the 
               elastic transfomations to visualize it clearly. Do not set this 
               option to train the network!
        """
        trans_image = image
        trans_mask = mask
        transform_string = '' 
        transformed = False

        # Elastic transformation
        prob = random.uniform(0, 1)
        if (self.elastic == True or flow == True) and prob < self.e_prob:

            if flow == True:
                self.__draw_grid(trans_image)
                self.__draw_grid(trans_mask, m=True)

            im_concat = np.concatenate((trans_image, trans_mask), axis=2)            

            im_concat_r = elastic_transform(im_concat, im_concat.shape[1]*2,
                                            im_concat.shape[1]*0.08,
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
        if (self.vflip == True or flow == True) and 0 <= prob < 0.25:
            trans_image = np.flip(trans_image, 0)
            trans_mask = np.flip(trans_mask, 0)
            transform_string = transform_string + '_vf'
            transformed = True 
            self.t_counter[1] += 1
        # Horizontal flip
        elif (self.hflip == True or flow == True) and 0.25 <= prob < 0.5:
            trans_image = np.flip(trans_image, 1)
            trans_mask = np.flip(trans_mask, 1)
            transform_string = transform_string + '_hf'
            transformed = True
            self.t_counter[2] += 1 
        # Vertical and horizontal flip
        elif (self.hflip == True or flow == True) and 0.5 <= prob < 0.75:
            trans_image = np.flip(trans_image, 0)                               
            trans_mask = np.flip(trans_mask, 0)
            trans_image = np.flip(trans_image, 1)                               
            trans_mask = np.flip(trans_mask, 1)
            transform_string = transform_string + '_hfvf'
            transformed = True
            self.t_counter[1] += 1
            self.t_counter[2] += 1
            
        # Free rotation from -range to range (in degrees)
        if (self.rotation_range != 0):
            theta = np.random.uniform(-self.rotation_range, self.rotation_range)
            trans_image = ndimage.rotate(trans_image, theta, reshape=False, 
                                         mode='reflect', order=1)
            trans_mask = ndimage.rotate(trans_mask, theta, reshape=False, 
                                        mode='reflect', order=0)
            transform_string = transform_string + '_rRange'
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
            if (self.rotation90 == True or flow == True) and 0 <= prob < 0.25:
                trans_image = np.rot90(trans_image)
                trans_mask = np.rot90(trans_mask)
                transform_string = transform_string + '_r90'
                transformed = True 
                self.t_counter[3] += 1
            # 180 degree rotation
            elif (self.rotation90 == True or flow == True) and 0.25 <= prob < 0.5:
                trans_image = np.rot90(trans_image, 2)
                trans_mask = np.rot90(trans_mask, 2)
                transform_string = transform_string + '_r180'
                transformed = True 
                self.t_counter[4] += 1
            # 270 degree rotation
            elif (self.rotation90 == True or flow == True) and 0.5 <= prob < 0.75:
                trans_image = np.rot90(trans_image, 3)
                trans_mask = np.rot90(trans_mask, 3)
                transform_string = transform_string + '_r270'
                transformed = True 
                self.t_counter[5] += 1
        else:
            if (self.rotation90 == True or flow == True) and 0 <= prob < 0.5:
                trans_image = np.rot90(trans_image, 2)                          
                trans_mask = np.rot90(trans_mask, 2)                            
                transform_string = transform_string + '_r180'                   
                transformed = True                                              
                self.t_counter[4] += 1

        if transformed == False:
            transform_string = '_none'         

        return trans_image, trans_mask, transform_string


    def flow_on_examples(self, num_examples, job_id="none_job_id", out_dir='aug',
                         save_prefix=None, original_elastic=True,
                         random_images=True):
        """Apply selected transformations to a defined number of images from
           the dataset. The purpose of this method is to check the images 
           generated by data augmentation. 
            
           Args:
               num_examples (int): number of examples to generate.
               job_id (str, optional): job identifier. If any provided the
               examples will be generated under a folder 'aug/none_job_id'.
               out_dir (str, optional): name of the folder where the 
               examples will be stored. If any provided the examples will be 
               generated under a folder 'aug/none_job_id'.
               save_prefix (str, optional): prefix to add to the generated 
               examples' name. 
               original_elastic (bool, optional): to save also the original
               images when an elastic transformation is performed.
               random_images (bool, optional): randomly select images from the
               dataset. If False the examples will be generated from the start
               of the dataset. 
        """
        Print("[FLOW] Creating the examples of data augmentation . . .")

        prefix = ""
        if save_prefix is not None:
            prefix = str(save_prefix)

        out_dir = os.path.join(out_dir, job_id) 
        if not os.path.exists(out_dir):                              
            os.makedirs(out_dir)

        # Generate the examples 
        for i in tqdm(range(0,num_examples)):
            if random_images == True:
                pos = random.randint(1,self.X.shape[0]-1) 
            else:
                pos = cont 

            im = self.X[pos]
            mask = self.Y[pos]

            out_im, out_mask, t_str = self.apply_transform(im, mask, flow=True)

            out_im = Image.fromarray(out_im[:,:,0])                           
            out_im = out_im.convert('L')                                                    
            out_im.save(os.path.join(out_dir, prefix + 'x_' + str(pos) + t_str 
                                     + ".png"))          
                 
            out_mask = Image.fromarray(out_mask[:,:,0]*255)                           
            out_mask = out_mask.convert('L')                                                    
            out_mask.save(os.path.join(out_dir, prefix + 'y_' + str(pos) + t_str
                                       + ".png"))          
                
            # Save also the original images if an elastic transformation was made
            if original_elastic == True and '_e' in t_str: 
                im = Image.fromarray(im[:,:,0])
                im = im.convert('L')
                im.save(os.path.join(out_dir, prefix + 'x_' + str(pos) + t_str 
                                     + '_original.png'))

                mask = Image.fromarray(mask[:,:,0]*255)
                mask = mask.convert('L')
                mask.save(os.path.join(out_dir, prefix + 'y_' + str(pos) + t_str
                                       + '_original.png'))


def fixed_dregee(image):
    """Rotate given image with a fixed degree

       Args:
            image (img): image to be rotated.

       Returns:
            out_image (numpy array): image rotated.
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


def keras_da_generator(X_train, Y_train, X_val, Y_val, batch_size_value,        
                       save_examples=True, job_id="none_job_id", out_dir='aug',
                       hflip=True, vflip=True, seedValue=42, rotation_range=180,
                       fill_mode='reflect', preproc_function=False, 
                       featurewise_center=False, brightness_range=None,
                       channel_shift_range=0.0, shuffle=True,
                       featurewise_std_normalization=False, zoom=False,         
                       w_shift_r=0.0, h_shift_r=0.0, shear_range=0,
                       crops_before_DA=False, crop_length=0):             
                                                                                
    """Makes data augmentation of the given input data.                         
                                                                                
       Args:                                                                    
            X_train_path (numpy array): train data.                                  
            Y_train_path (numpy array): train mask data.                             
            X_val_path (numpy array): validation data.                               
            Y_val_path (numpy array): validation mask data.                          
            batch_size_value (int): batch size.                                 
            save_examples (bool, optional): if true 5 examples of DA are stored.
            job_id (str, optional): job identifier. If any provided the         
            examples will be generated under a folder 'aug/none_job_id'.        
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
            brightness_range (tuple or list of two floats, optional): range for picking a brightness shift value from.
            channel_shift_range (float, optional): range for random channel shifts.
            shuffle (bool, optional): randomize the training data.
            featurewise_std_normalization (bool, optional): divide inputs by std 
            of the dataset, feature-wise.                                       
            zoom (bool, optional): make random zoom in the images.              
            w_shift_r (float, optional): width shift range.
            h_shift_r (float, optional): height shift range.
            shear_range (float, optional): range to apply shearing 
            transformations. 
            crops_before_DA (bool, optional): decide to make random crops before
            apply DA transformations.                                           
            crop_length (int, optional): length of the random crop before DA. 
                                                                                
       Returns:                                                                 
            train_generator (Keras iterable of flow_from_directory): train data 
            iterator.                                                           
            val_generator (Keras iterable of flow_from_directory): validation   
            data iterator.                                                      
    """                                                                         
                                                                                
    zoom_val = 0.25 if zoom == True else 0                                      
                                                                                
    if preproc_function == True:                                                
        data_gen_args1 = dict(horizontal_flip=hflip, vertical_flip=vflip,       
                              fill_mode=fill_mode,                              
                              preprocessing_function=fixed_dregee,              
                              featurewise_center=featurewise_center,            
                              featurewise_std_normalization=featurewise_std_normalization,
                              zoom_range=zoom_val, width_shift_range=w_shift_r,
                              height_shift_range=h_shift_r, 
                              shear_range=shear_range)                              
        data_gen_args2 = dict(horizontal_flip=hflip, vertical_flip=vflip,       
                              fill_mode=fill_mode,                              
                              preprocessing_function=fixed_dregee,              
                              zoom_range=zoom_val, width_shift_range=w_shift_r,
                              height_shift_range=h_shift_r, 
                              shear_range=shear_range)                              
    else:                                                                       
        data_gen_args1 = dict(horizontal_flip=hflip, vertical_flip=vflip,       
                              fill_mode=fill_mode, rotation_range=rotation_range,          
                              featurewise_center=featurewise_center,            
                              featurewise_std_normalization=featurewise_std_normalization,
                              zoom_range=zoom_val, width_shift_range=w_shift_r,
                              height_shift_range=h_shift_r, 
                              shear_range=shear_range,
                              channel_shift_range=channel_shift_range,
                              brightness_range=brightness_range )
        data_gen_args2 = dict(horizontal_flip=hflip, vertical_flip=vflip,       
                              fill_mode=fill_mode, rotation_range=rotation_range,          
                              zoom_range=zoom_val, width_shift_range=w_shift_r,
                              height_shift_range=h_shift_r, 
                              shear_range=shear_range)                              
                                                                                
                                                                                
    # Train data, provide the same seed and keyword arguments to the fit and    
    # flow methods                                                              
    X_datagen_train = kerasDA(**data_gen_args1)                                 
    Y_datagen_train = kerasDA(**data_gen_args2)                                 
                                                                                
    # Validation data, no data augmentation, but we create a generator anyway   
    X_datagen_val = kerasDA()                                                   
    Y_datagen_val = kerasDA()                                                   
    #X_datagen_val.fit(X_val, augment=False, seed=seedValue)
    #Y_datagen_val.fit(Y_val, augment=False, seed=seedValue)

    # Check a few of generated images
    if save_examples == True:
        
        out_dir = os.path.join(out_dir, job_id)
        if not os.path.exists(out_dir):          
            os.makedirs(out_dir)
     
        i = 0
        for batch in X_datagen_train.flow(X_train, 
                                          save_to_dir=out_dir,
                                          batch_size=batch_size_value,
                                          shuffle=False, seed=seedValue,
                                          save_prefix='x', save_format='jpeg'):
            i = i + 1
            if i > 2:
                break
        i = 0
        for batch in Y_datagen_train.flow(Y_train, 
                                          save_to_dir=out_dir,
                                          batch_size=batch_size_value,
                                          shuffle=False, seed=seedValue,
                                          save_prefix='y', save_format='jpeg'):
            i = i + 1
            if i > 2:
                break

    X_train_augmented = X_datagen_train.flow(X_train,                           
                                             batch_size=batch_size_value,       
                                             shuffle=shuffle, seed=seedValue)
    Y_train_augmented = Y_datagen_train.flow(Y_train,                           
                                             batch_size=batch_size_value,       
                                             shuffle=shuffle, seed=seedValue)
    X_val_flow = X_datagen_val.flow(X_val, batch_size=batch_size_value,         
                                    shuffle=False, seed=seedValue)              
    Y_val_flow = Y_datagen_val.flow(Y_val, batch_size=batch_size_value,         
                                    shuffle=False, seed=seedValue)              
             
    # Combine generators into one which yields image and masks                  
    train_generator = zip(X_train_augmented, Y_train_augmented)                 
    val_generator = zip(X_val_flow, Y_val_flow)
                                                                   
    if crops_before_DA == True:                                                
        train_generator = crop_generator(train_generator, crop_length)
        val_generator = crop_generator(val_generator, crop_length, val=True)

    return train_generator, val_generator


def crop_generator(batches, crop_length, val=False):
    """Take as input a Keras ImageGen (Iterator) and generate random
       crops from the image batches generated by the original iterator.
        
       Based on:                                                                
           https://jkjung-avt.github.io/keras-image-cropping/  
    """

    while True:
        batch_x, batch_y = next(batches)
        batch_crops_x = np.zeros((batch_x.shape[0], crop_length, crop_length, 1))
        batch_crops_y = np.zeros((batch_y.shape[0], crop_length, crop_length, 1))
        for i in range(batch_x.shape[0]):
            batch_crops_x[i], batch_crops_y[i] = random_crop(batch_x[i], batch_y[i], (crop_length, crop_length), val)
        yield (batch_crops_x, batch_crops_y)


def random_crop(img, mask, random_crop_size, val=False):
    """Random crop.
       Based on:                                                                
           https://jkjung-avt.github.io/keras-image-cropping/
    """

    assert img.shape[2] == 1
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    if val == True:
        x = 0
        y = 0
    else:
        x = np.random.randint(0, width - dx + 1)                                
        y = np.random.randint(0, height - dy + 1)
    return img[y:(y+dy), x:(x+dx), :], mask[y:(y+dy), x:(x+dx), :]


