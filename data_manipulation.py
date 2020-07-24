import numpy as np
import os
import sys
import math
import random
from tqdm import tqdm
from skimage.io import imread
from sklearn.model_selection import train_test_split
from PIL import Image
from util import foreground_percentage


def load_and_prepare_2D_data(train_path, train_mask_path, test_path, test_mask_path, 
              image_train_shape, image_test_shape, create_val=True, 
              val_split=0.1, shuffle_val=True, seedValue=42, e_d_data=[], 
              e_d_mask=[], e_d_data_dim=[], e_d_dis=[], num_crops_per_dataset=0, 
              make_crops=True, crop_shape=None, check_crop=True, 
              check_crop_path="check_crop", d_percentage=0):         

    """Load train, validation and test images from the given paths to create 2D
       data. If the images to be loaded are smaller than the given dimension it 
       will be sticked in the (0, 0).
                                                                        
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

            e_d_data (list of str, optional): list of paths where the extra data
            of other datasets are stored.

            e_d_mask (list of str, optional): list of paths where the extra data
            mask of other datasets are stored.

            e_d_data_dim (list of 3D int tuple, optional): list of shapes of the 
            extra datasets provided. 

            e_d_dis (list of float, optional): discard percentages of the extra
            datasets provided. Values between 0 and 1.

            num_crops_per_dataset (int, optional): number of crops per extra
            dataset to take into account. Useful to ensure that all the datasets
            have the same weight during network trainning. 

            make_crops (bool, optional): flag to make crops on data.

            crop_shape (3D int tuple, optional): shape of the crops.

            check_crop (bool, optional): to save the crops made to ensure they
            are generating as one wish.

            check_crop_path (str, optional): path to save the crop samples.

            d_percentage (int, optional): number between 0 and 100. The images
            that have less foreground pixels than the given number will be
            discarded.

       Returns:                                                         
            X_train (4D Numpy array): train images. 
            E.g. (image_number, x, y, channels).

            Y_train (4D Numpy array): train images' mask.              
            E.g. (image_number, x, y, channels).

            X_val (4D Numpy array, optional): validation images (create_val==True).
            E.g. (image_number, x, y, channels).

            Y_val (4D Numpy array, optional): validation images' mask 
            (create_val==True). E.g. (image_number, x, y, channels).

            X_test (4D Numpy array): test images. 
            E.g. (image_number, x, y, channels).

            Y_test (4D Numpy array): test images' mask.                
            E.g. (image_number, x, y, channels).
        
            orig_test_shape (tuple of ints): test data original shape.

            norm_value (int): normalization value calculated.

            crop_made (bool): True if crops have been made.
    """      
   
    print("### LOAD ###")
                                                                        
    tr_shape = (image_train_shape[1], image_train_shape[0], image_train_shape[2])
    print("0) Loading train images . . .")
    X_train = load_data_from_dir(train_path, tr_shape)
    print("1) Loading train masks . . .")
    Y_train = load_data_from_dir(train_mask_path, tr_shape)

    if num_crops_per_dataset != 0:
        X_train = X_train[:num_crops_per_dataset]
        Y_train = Y_train[:num_crops_per_dataset]

    te_shape = (image_test_shape[1], image_test_shape[0], image_test_shape[2])
    print("2) Loading test images . . .")
    X_test = load_data_from_dir(test_path, te_shape)
    print("3) Loading test masks . . .")
    Y_test = load_data_from_dir(test_mask_path, te_shape)

    orig_test_shape = tuple(Y_test.shape[i] for i in [0, 2, 1, 3])

    # Create validation data splitting the train
    if create_val:
        X_train, X_val, \
        Y_train, Y_val = train_test_split(
            X_train, Y_train, test_size=val_split, shuffle=shuffle_val,
            random_state=seedValue)

    # Crop the data
    if make_crops:
        print("4) Crop data activated . . .")
        print("4.1) Cropping train data . . .")
        X_train, Y_train, _ = crop_data(
            X_train, crop_shape, data_mask=Y_train, d_percentage=d_percentage)   

        print("4.2) Cropping test data . . .")
        X_test, Y_test, _ = crop_data(X_test, crop_shape, data_mask=Y_test)
        
        if create_val:
            print("4.3) Cropping validation data . . .")
            X_val, Y_val, _ = crop_data(
                X_val, crop_shape, data_mask=Y_val, d_percentage=d_percentage)

        if check_crop:
            print("4.4) Checking the crops . . .")
            check_crops(X_train, (image_test_shape[0], image_test_shape[1]),
                        num_examples=3, out_dir=check_crop_path, suffix="_x_", 
                        grid=True)
            check_crops(Y_train, (image_test_shape[0], image_test_shape[1]),
                        num_examples=3, out_dir=check_crop_path, suffix="_y_", 
                        grid=True)
        
        crop_made = True
    else:
        crop_made = False

    # Load the extra datasets
    if e_d_data:
        print("5) Loading extra datasets . . .")
        for i in range(len(e_d_data)):
            print("5.{}) extra dataset in {} . . .".formated(i, e_d_data[i])) 
            train_ids = sorted(next(os.walk(e_d_data[i]))[2])
            train_mask_ids = sorted(next(os.walk(e_d_mask[i]))[2])

            d_dim = e_d_data_dim[i]
            e_X_train = np.zeros((len(train_ids), d_dim[1], d_dim[0], d_dim[2]),
                                 dtype=np.float32)
            e_Y_train = np.zeros((len(train_mask_ids), d_dim[1], d_dim[0], 
                                 d_dim[2]), dtype=np.float32)

            print("5.{}) Loading data of the extra dataset . . .".format(i))
            for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
                im = imread(os.path.join(e_d_data[i], id_))
                if len(im.shape) == 2:
                    im = np.expand_dims(im, axis=-1)
                e_X_train[n] = im

            print("5.{}) Loading masks of the extra dataset . . .".format(i))
            for n, id_ in tqdm(enumerate(train_mask_ids), total=len(train_mask_ids)):
                mask = imread(os.path.join(e_d_mask[i], id_))
                if len(mask.shape) == 2:
                    mask = np.expand_dims(mask, axis=-1)
                e_Y_train[n] = mask

            if make_crops == False:
                if d_dim[1] != image_test_shape[1] and \
                   d_dim[0] != image_test_shape[0]:
                    raise ValueError(
                        "extra dataset shape {} is not equal the original "
                        "dataset shape ({}, {})".format(d_dim, \
                        image_test_shape[1], image_test_shape[0]))
            else:
                print("5.{}) Cropping the extra dataset . . .".format(i))
                e_X_train, e_Y_train, _ = crop_data(
                    e_X_train, crop_shape, data_mask=e_Y_train, 
                    d_percentage=e_d_dis[i])
                if num_crops_per_dataset != 0:
                    e_X_train = e_X_train[:num_crops_per_dataset]
                    e_Y_train = e_Y_train[:num_crops_per_dataset]

                if check_crop:
                    print("5.{}) Checking the crops of the extra dataset . . ."\
                          .format(i))
                    check_crops(e_X_train, (d_dim[0], d_dim[1]), num_examples=3, 
                                out_dir=check_crop_path, 
                                suffix="_e" + str(i) + "x_", grid=True)
                    check_crops(e_Y_train, (d_dim[0], d_dim[1]), num_examples=3,
                                out_dir=check_crop_path,
                                suffix="_e" + str(i) + "y_", grid=True)

            # Concatenate datasets
            X_train = np.vstack((X_train, e_X_train))
            Y_train = np.vstack((Y_train, e_Y_train))

    if create_val:
        print("*** Loaded train data shape is: {}".format(X_train.shape))
        print("*** Loaded validation data shape is: {}".format(X_val.shape))
        print("*** Loaded test data shape is: {}".format(X_test.shape))
        print("### END LOAD ###")

        # Calculate normalization value
        norm_value = np.mean(X_train)

        return X_train, Y_train, X_val, Y_val, X_test, Y_test, orig_test_shape, \
               norm_value, crop_made
    else:
        print("*** Loaded train data shape is: {}".format(X_train.shape))
        print("*** Loaded test data shape is: {}".format(X_test.shape))
        print("### END LOAD ###")

        # Calculate normalization value
        norm_value = np.mean(X_train)

        return X_train, Y_train, X_test, Y_test, orig_test_shape, norm_value, \
               crop_made


def load_and_prepare_3D_data(train_path, train_mask_path, test_path, 
                             test_mask_path, image_train_shape, image_test_shape, 
                             create_val=True, shuffle_val=True, val_split=0.1, 
                             seedValue=42, train_subvol_shape=None, 
                             test_subvol_shape=None, 
                             random_subvolumes_in_DA=False, ov_test=0.5):         

    """Load train, validation and test images from the given paths to create 2D
       data. If the images to be loaded are smaller than the given dimension it
       will be sticked in the (0, 0).
                                                                        
       Args:                                                            
            train_path (str): path to the training data.                

            train_mask_path (str): path to the training data masks.     

            test_path (str): path to the test data.                     

            test_mask_path (str): path to the test data masks.          

            image_train_shape (array of 3 int): dimensions of the images.     

            image_test_shape (array of 3 int): dimensions of the images.     

            create_val (bool, optional): if true validation data is created.                                                    
    
            shuffle_val (bool, optional): take random training examples to      
            create validation data.

            val_split (float, optional): % of the train data used as    
            validation (value between 0 and 1).

            seedValue (int, optional): seed value.

            train_subvol_shape (Tuple, optional): shape of the train subvolumes 
            to create. 

            test_subvol_shape (Tuple, optional): shape of the test subvolumes 
            to create. 

            random_subvolumes_in_DA (bool, optional): flag to advice the method 
            that not preparation of the data must be done, as random subvolumes
            will be created on DA, and the whole volume will be used for that.

       Returns:                                                         
            X_train (4D Numpy array): train images. 
            E.g. (image_number, x, y, channels).

            Y_train (4D Numpy array): train images' mask.              
            E.g. (image_number, x, y, channels).

            X_val (4D Numpy array, optional): validation images (create_val==True).
            E.g. (image_number, x, y, channels).

            Y_val (4D Numpy array, optional): validation images' mask 
            (create_val==True). E.g. (image_number, x, y, channels).

            X_test (4D Numpy array): test images. 
            E.g. (image_number, x, y, channels).

            Y_test (4D Numpy array): test images' mask.                
            E.g. (image_number, x, y, channels).
        
            orig_test_shape (tuple of ints): test data original shape.

            norm_value (int): normalization value calculated.
    """      
   
    print("### LOAD ###")
                                                                        
    tr_shape = (image_train_shape[1], image_train_shape[0], image_train_shape[2])
    print("0) Loading train images . . .")
    X_train = load_data_from_dir(train_path, tr_shape)
    print("1) Loading train masks . . .")
    Y_train = load_data_from_dir(train_mask_path, tr_shape)

    te_shape = (image_test_shape[1], image_test_shape[0], image_test_shape[2])
    print("2) Loading test images . . .")
    X_test = load_data_from_dir(test_path, te_shape)
    print("3) Loading test masks . . .")
    Y_test = load_data_from_dir(test_mask_path, te_shape)

    orig_test_shape = tuple(Y_test.shape[i] for i in [0, 2, 1, 3])

    # Force 3D data preparation if create_val is selected 
    mirrored_subvols = 0
    if random_subvolumes_in_DA == False:
        print("Preparing train data subvolumes . . .")
        X_train, Y_train, mirrored_subvols = prepare_3D_volume_data(
            X_train, Y_train, train_subvol_shape)
    print("Preparing test data subvolumes . . .")
    X_test, Y_test, _ = prepare_3D_volume_data(
        X_test, Y_test, test_subvol_shape, overlap=True, ov=ov_test)

    # Create validation data splitting the train
    if create_val:
        # Calculate the percentage of images to extract the validation data from 
        # avoiding those subvolumes that have been created using the rest of the
        # images and mirroring in the prepare_3D_volume_data function
        if mirrored_subvols != 0:
            val_split = (val_split*mirrored_subvols)/X_train.shape[0]
            X_train2, X_val, \
            Y_train2, Y_val = train_test_split(
                X_train[:mirrored_subvols], Y_train[:mirrored_subvols],
                test_size=val_split, shuffle=shuffle_val, random_state=seedValue)
            X_train = np.concatenate((X_train2, X_train[mirrored_subvols:]), axis=0)
            Y_train = np.concatenate((Y_train2, Y_train[mirrored_subvols:]), axis=0)
        else:       
            X_train, X_val, \
            Y_train, Y_val = train_test_split(
                X_train, Y_train, test_size=val_split, shuffle=shuffle_val, 
                random_state=seedValue)

    if create_val:
        print("*** Loaded train data shape is: {}".format(X_train.shape))
        print("*** Loaded validation data shape is: {}".format(X_val.shape))
        print("*** Loaded test data shape is: {}".format(X_test.shape))
        print("### END LOAD ###")

        # Calculate normalization value
        norm_value = np.mean(X_train)

        return X_train, Y_train, X_val, Y_val, X_test, Y_test, orig_test_shape, \
               norm_value
    else:                                                               
        print("*** Loaded train data shape is: {}".format(X_train.shape))
        print("*** Loaded test data shape is: {}".format(X_test.shape))
        print("### END LOAD ###")

        # Calculate normalization value
        norm_value = np.mean(X_train)

        return X_train, Y_train, X_test, Y_test, orig_test_shape, norm_value


def load_data_from_dir(data_dir, shape):
    """Load data from a directory.
        
       Args:
            data_dir (str): path to read the data from.

            shape (3D int tuple): shape of the data.

       Return:
            data (4D Numpy array): data loaded. 
            E.g. (image_number, x, y, channels).
    """
    print("Loading data from {}".format(data_dir))
    ids = sorted(next(os.walk(data_dir))[2])
    data = np.zeros(shape[:2] + (len(ids), shape[2]), dtype=np.float32)

    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        img = imread(os.path.join(data_dir, id_))

        # Convert the image into grayscale
        if len(img.shape) >= 3:
            img = img[:, :, 0]
            img = np.expand_dims(img, axis=-1)

        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)

        data[...,n,:] = img

    return data


def crop_data(data, crop_shape, data_mask=None, force_shape=(0, 0), 
              d_percentage=0):                          
    """Crop data into smaller pieces.
                                                                        
       Args:                                                            
            data (4D Numpy array): data to crop. 
            E.g. (image_number, x, y, channels).

            crop_shape (3D int tuple): output image shape.
            E.g. (image_number, x, y, channels).

            data_mask (4D Numpy array, optional): data masks to crop.
            E.g. (image_number, x, y, channels).

            force_shape (2D int tuple, optional): force horizontal and vertical 
            crops to the given numbers.

            d_percentage (int, optional): number between 0 and 100. The images 
            that have less foreground pixels than the given number will be 
            discarded. Only available if data_mask is provided.
                                                                        
       Returns:                                                         
            cropped_data (4D Numpy array): cropped data images.         

            cropped_data_mask (4D Numpy array): cropped data masks.     

            force_shape (2D int tuple): number of horizontal and vertical crops 
            made. Useful for future crop calls. 
    """                                                                 

    print("### CROP ###")                                                                    
    print("Cropping [{},{}] images into {} . . .".format(data.shape[1], \
          data.shape[2], crop_shape)) 
  
    # Calculate the number of images to be generated                    
    if force_shape == (0, 0):
        h_num = int(data.shape[1] / crop_shape[0]) + (data.shape[1] % crop_shape[0] > 0)
        v_num = int(data.shape[2] / crop_shape[1]) + (data.shape[2] % crop_shape[1] > 0)
        force_shape = (h_num, v_num)
    else:
        h_num = force_shape[0]
        v_num = force_shape[1]
        print("Force crops to [{}, {}]".format(h_num, v_num))

    total_cropped = data.shape[0]*h_num*v_num    

    # Resize data to adjust to a value divisible by height and width
    r_data = np.zeros((data.shape[0], h_num*crop_shape[1], v_num*crop_shape[0], 
                       data.shape[3]), dtype=np.float32)    
    r_data[:data.shape[0],:data.shape[1],:data.shape[2],:data.shape[3]] = data
    if data_mask is not None:
        r_data_mask = np.zeros((data_mask.shape[0], h_num*crop_shape[1], 
                                v_num*crop_shape[0], data_mask.shape[3]), 
                               dtype=np.float32)
        r_data_mask[:data_mask.shape[0],:data_mask.shape[1],
                    :data_mask.shape[2],:data_mask.shape[3]] = data_mask
    if data.shape != r_data.shape:
        print("Resized data from {} to {} to be divisible by the shape provided"\
              .format(data.shape, r_data.shape))

    discarded = 0                                                                    
    cont = 0
    selected_images  = []

    # Discard images from the data set
    if d_percentage > 0 and data_mask is not None:
        print("0) Selecting images to discard . . .")
        for img_num in tqdm(range(0, r_data.shape[0])):                             
            for i in range(0, h_num):                                       
                for j in range(0, v_num):
                    p = foreground_percentage(r_data_mask[
                        img_num, (i*crop_shape[0]):((i+1)*crop_shape[1]),
                        (j*crop_shape[0]):((j+1)*crop_shape[1])], 255)
                    if p > d_percentage: 
                        selected_images.append(cont)
                    else:
                        discarded = discarded + 1

                    cont = cont + 1

    # Crop data                                                         
    cropped_data = np.zeros(((total_cropped-discarded), crop_shape[1], 
                              crop_shape[0], r_data.shape[3]), dtype=np.float32)
    if data_mask is not None:
        cropped_data_mask = np.zeros(((total_cropped-discarded), crop_shape[1], 
                                       crop_shape[0], r_data_mask.shape[3]), 
                                     dtype=np.float32)
    
    cont = 0                                                              
    l_i = 0
    print("1) Cropping images . . .")
    for img_num in tqdm(range(0, r_data.shape[0])): 
        for i in range(0, h_num):                                       
            for j in range(0, v_num):                     
                if d_percentage > 0 and data_mask is not None \
                   and len(selected_images) != 0:
                    if selected_images[l_i] == cont \
                       or l_i == len(selected_images) - 1:

                        cropped_data[l_i] = r_data[
                            img_num, (i*crop_shape[0]):((i+1)*crop_shape[1]), 
                            (j*crop_shape[0]):((j+1)*crop_shape[1]),:]

                        cropped_data_mask[l_i] = r_data_mask[
                            img_num, (i*crop_shape[0]):((i+1)*crop_shape[1]),
                            (j*crop_shape[0]):((j+1)*crop_shape[1]),:]

                        if l_i != len(selected_images) - 1:
                            l_i = l_i + 1
                else: 
              
                    cropped_data[cont] = r_data[
                        img_num, (i*crop_shape[0]):((i+1)*crop_shape[1]),      
                        (j*crop_shape[0]):((j+1)*crop_shape[1]),:]
                                                                        
                    if data_mask is not None:
                        cropped_data_mask[cont] = r_data_mask[
                            img_num, (i*crop_shape[0]):((i+1)*crop_shape[1]),
                            (j*crop_shape[0]):((j+1)*crop_shape[1]),:]
                cont = cont + 1                                             
                                                                        
    if d_percentage > 0 and data_mask is not None:
        print("**** {} images discarded. New shape after cropping and discarding "
              "is {}".format(discarded, cropped_data.shape)) 
        print("### END CROP ###")
    else:
        print("**** New data shape is: {}".format(cropped_data.shape))
        print("### END CROP ###")

    if data_mask is not None:
        return cropped_data, cropped_data_mask, force_shape
    else:
        return cropped_data, force_shape


def crop_data_with_overlap(data, data_mask, window_size, subdivision):
    """Crop data into smaller pieces with the minimun overlap.

       Args:
            data (4D Numpy array): data to crop.
            E.g. (image_number, x, y, channels).

            data_mask (4D Numpy array): data mask to crop.
            E.g. (image_number, x, y, channels).

            window_size (int): crop size.

            subdivision (int): number of crops to create.

       Returns:
            cropped_data (4D Numpy array): cropped image data.
            E.g. (image_number, x, y, channels).

            cropped_data_mask (4D Numpy array): cropped image data masks.
            E.g. (image_number, x, y, channels).
    """

    print("### OV-CROP ###")
    print("Cropping {} images into ({}, {}) with overlapping. . ."\
          .format(data.shape[1:], window_size, window_size))

    if subdivision != 1 and subdivision % 2 != 0:
        raise ValueError("'subdivision' must be 1 or an even number")
    if window_size > data.shape[1]:
        raise ValueError("'window_size' greater than {}".format(data.shape[1]))
    if window_size > data.shape[2]:
        raise ValueError("'window_size' greater than {}".format(data.shape[2]))

    # Crop data
    total_cropped = data.shape[0]*subdivision
    cropped_data = np.zeros((total_cropped, window_size, window_size,
                             data.shape[3]), dtype=np.float32)
    cropped_data_mask = np.zeros((total_cropped, window_size, window_size,
                                 data_mask.shape[3]), dtype=np.float32)

    # Find the mininum overlap configuration with the number of crops to create
    min_d = sys.maxsize
    rows = 1
    columns = 1
    for i in range(1, int(subdivision/2)+1, 1):
        if subdivision % i == 0 and abs((subdivision/i) - i) < min_d:
            min_d = abs((subdivision/i) - i)
            rows = i
            columns = int(subdivision/i)
        
    print("The minimum overlap has been found with rows={} and columns={}"\
          .format(rows, columns))

    if subdivision != 1:
        if window_size*rows < data.shape[1]:
            raise ValueError(
                "Total height of all the crops per row must be greater or equal "
                "{} and it is only {}".format(data.shape[1], window_size*rows))
        if window_size*columns < data.shape[2]:
            raise ValueError(
                "Total width of all the crops per row must be greater or equal "
                "{} and it is only {}".format(data.shape[2], window_size*columns))

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
    print("0) Cropping data with the minimun overlap . . .")
    for k, img_num in tqdm(enumerate(range(0, data.shape[0]))):
        for i in range(0, data.shape[1]-y_ov, step_y):
            for j in range(0, data.shape[2]-x_ov, step_x):
                d_y = 0 if (i+window_size) < data.shape[1] else r_y
                d_x = 0 if (j+window_size) < data.shape[2] else r_x

                cropped_data[cont] = data[
                    k, i-d_y:i+window_size, j-d_x:j+window_size, :]
                cropped_data_mask[cont] = data_mask[
                    k, i-d_y:i+window_size, j-d_x:j+window_size, :]
                cont = cont + 1

    print("**** New data shape is: {}".format(cropped_data.shape))
    print("### END OV-CROP ###")

    return cropped_data, cropped_data_mask


def crop_3D_data_with_overlap(data, vol_shape, data_mask=None, overlap_z=0.5):
    """Crop 3D data into smaller volumes with the minimun overlap in z.
       Reverse function of merge_3D_data_with_overlap().

       Args:
            data (4D Numpy array): data to crop.
            E.g. (image_number, x, y, channels).

            vol_shape (4D int tuple): shape of the desired volumes to be created.
        
            data_mask (4D Numpy array, optional): data mask to crop.
            E.g. (image_number, x, y, channels).

            overlap_z (float, optional): amount of overlap on z dimension. The 
            value must be on range [0, 1), that is, 0% or 99% of overlap. 

       Returns:
            cropped_data (5D Numpy array): cropped image data.
            E.g. (vol_number, z, x, y, channels).

            cropped_data_mask (5D Numpy array, optional): cropped image data 
            masks. E.g. (vol_number, z, x, y, channels).
    """

    print("### 3D-OV-CROP ###")
    print("Cropping {} images into {} with overlapping. . ."
          .format(data.shape, vol_shape))

    if overlap_z >= 1 or overlap_z < 0:
        raise ValueError("'overlap_z' must be a float on range [0, 1)")
    if len(vol_shape) != 4:
        raise ValueError("'vol_shape' must be 4D int tuple")
    if vol_shape[0] > data.shape[0]:
        raise ValueError("'vol_shape[0]' {} greater than {}"
                         .format(vol_shape[0], data.shape[0]))
    if vol_shape[1] > data.shape[1]:
        raise ValueError("'vol_shape[1]' {} greater than {}"
                         .format(vol_shape[1], data.shape[1]))
    if vol_shape[2] > data.shape[2]:
        raise ValueError("'vol_shape[2]' {} greater than {}"
                         .format(vol_shape[2], data.shape[2]))
   
    # Minimun overlap
    if overlap_z == 0: 
        vols_per_z = math.ceil(data.shape[2]/vol_shape[2])
        excess_z = (vols_per_z*vol_shape[2])-data.shape[2]
        step_z = vol_shape[2]-int(excess_z/(vols_per_z-1))
        last_z = excess_z%(vols_per_z-1) 
        r_div = vol_shape[2]-last_z
    else: 
        overlap_z = 1-overlap_z
        vols_per_z = math.ceil(data.shape[2]/(vol_shape[2]*overlap_z))  
        step_z = int(vol_shape[2]*overlap_z)
        r_div = int(vol_shape[2]-(data.shape[2]%(vol_shape[2]*overlap_z)))

    vols_per_x = math.ceil(data.shape[0]/vol_shape[0])
    vols_per_y = math.ceil(data.shape[1]/vol_shape[1])
    ov_x = (vol_shape[0]*vols_per_x)-data.shape[0]
    ov_y = (vol_shape[1]*vols_per_y)-data.shape[1]
   
    print("{},{},{} patches per x,y,z axis"
          .format(vols_per_x, vols_per_y, vols_per_z))
    
    if r_div != 0:
        print("WARNING: The last volumes in z will be filled with the last "
              "images in mirror mode to complete the volume with the shape "
              "provided, so be careful and resize the data before run the "
              "metrics on it!")

    total_vol = vols_per_x*vols_per_y*vols_per_z
    cropped_data = np.zeros((total_vol,) + vol_shape)
    if data_mask is not None:
        cropped_data_mask = np.zeros((total_vol,) + vol_shape)

    c = 0
    f_z = 0
    for z in range(vols_per_z):
        for x in range(vols_per_x):
            for y in range(vols_per_y):
                ov_x_ = ov_x if x != 0 else 0
                ov_y_ = ov_y if y != 0 else 0
               
                if (z*step_z)+vol_shape[2] < data.shape[2]:
                    cropped_data[c] = \
                        data[(x*vol_shape[0])-ov_x_:((x+1)*vol_shape[0])-ov_x_,
                             (y*vol_shape[1])-ov_y_:((y+1)*vol_shape[1])-ov_y_, 
                             z*step_z:(z*step_z)+vol_shape[2]]
                    if data_mask is not None:
                        cropped_data_mask[c] = \
                            data_mask[(x*vol_shape[0])-ov_x_:((x+1)*vol_shape[0])-ov_x_,
                                      (y*vol_shape[1])-ov_y_:((y+1)*vol_shape[1])-ov_y_,
                                      z*step_z:(z*step_z)+vol_shape[2]]
    
                # Fill the final volumes in z with the rests of the data
                else:
                    cropped_data[c,:,:,0:r_div-(f_z*step_z)] = \
                        data[(x*vol_shape[0])-ov_x_:((x+1)*vol_shape[0])-ov_x_,
                             (y*vol_shape[1])-ov_y_:((y+1)*vol_shape[1])-ov_y_,
                             z*step_z:(z*step_z)+vol_shape[2]]
                    if data_mask is not None:
                        cropped_data_mask[c,:,:,0:r_div-(f_z*step_z)] = \
                            data_mask[(x*vol_shape[0])-ov_x_:((x+1)*vol_shape[0])-ov_x_,
                                      (y*vol_shape[1])-ov_y_:((y+1)*vol_shape[1])-ov_y_,
                                      z*step_z:(z*step_z)+vol_shape[2]]

                    for s in range(vol_shape[2]-(r_div-(f_z*step_z))):
                        cropped_data[c,:,:,(r_div-(f_z*step_z))+s:vol_shape[2],0]  = \
                            data[(x*vol_shape[0])-ov_x_:((x+1)*vol_shape[0])-ov_x_,
                                 (y*vol_shape[1])-ov_y_:((y+1)*vol_shape[1])-ov_y_,
                                 data.shape[2]-1-s]
                    
                        if data_mask is not None:
                            cropped_data_mask[c,:,:,(r_div-(f_z*step_z))+s:vol_shape[2],0] = \
                                data_mask[(x*vol_shape[0])-ov_x_:((x+1)*vol_shape[0])-ov_x_,
                                          (y*vol_shape[1])-ov_y_:((y+1)*vol_shape[1])-ov_y_,
                                          data.shape[2]-1-s]
                c += 1

        # To adjust the final volumes in z
        if (z*step_z)+vol_shape[2] > data.shape[2]:
            f_z += 1

    print("**** New data shape is: {}".format(cropped_data.shape))
    print("### END OV-CROP ###")

    if data_mask is not None:
        return cropped_data, cropped_data_mask
    else:
        return cropped_data


def merge_data_with_overlap(data, original_shape, window_size, subdivision, 
                            out_dir=None, ov_map=True, ov_data_img=0):
    """Merge data with an amount of overlap. Used to undo the crop made by the 
       function crop_data_with_overlap.

       Args:
            data (4D Numpy array): data to merge.
            E.g. (image_number, x, y, channels).

            original_shape (4D int tuple): original dimensions to reconstruct. 

            window_size (int): crop size.

            subdivision (int): number of crops to merge.

            out_dir (str, optional): directory where the images will be save.

            ov_map (bool, optional): whether to create overlap map.

            ov_data_img (int, optional): number of the image on the data to 
            create the overlappng map.

       Returns:
            merged_data (4D Numpy array): merged image data.
            E.g. (image_number, x, y, channels).
    """

    print("### MERGE-OV-CROP ###")
    print("Merging {} images into ({},{}) with overlapping . . ."
          .format(data.shape[1:], original_shape[2], original_shape[1]))

    # Merged data
    total_images = int(data.shape[0]/subdivision)
    merged_data = np.zeros((total_images, original_shape[2], original_shape[1],
                             data.shape[3]), dtype=np.float32)

    # Matrices to store the amount of overlap. The first is used to store the
    # number of crops to merge for each pixel. The second matrix is used to 
    # paint the overlapping map
    overlap_matrix = np.zeros((original_shape[2], original_shape[1],
                             data.shape[3]), dtype=np.float32)
    if ov_map:
        ov_map_matrix = np.zeros((original_shape[2], original_shape[1],
                                   data.shape[3]), dtype=np.float32)

    # Find the mininum overlap configuration with the number of crops to create
    min_d = sys.maxsize
    rows = 1
    columns = 1
    for i in range(1, int(subdivision/2)+1, 1):
        if subdivision % i == 0 and abs((subdivision/i) - i) < min_d:
            min_d = abs((subdivision/i) - i)
            rows = i
            columns = int(subdivision/i)

    print("The minimum overlap has been found with ({}, {})"
          .format(rows, columns))

    # Calculate the amount of overlap, the division remainder to obtain an
    # offset to adjust the last crop and the step size. All of this values per
    # x/y or column/row axis
    if rows != 1:
        y_ov = int(abs(original_shape[2] - window_size*rows)/(rows-1))
        r_y = abs(original_shape[2] - window_size*rows) % (rows-1)
        step_y = window_size - y_ov
    else:
        y_ov = 0
        r_y = 0
        step_y = original_shape[2]

    if columns != 1:
        x_ov = int(abs(original_shape[1] - window_size*columns)/(columns-1))
        r_x = abs(original_shape[1] - window_size*columns) % (columns-1)
        step_x = window_size - x_ov
    else:
        x_ov = 0
        r_x = 0
        step_x = original_shape[1]

    # Calculate the overlapping matrix
    for i in range(0, original_shape[2]-y_ov, step_y):
        for j in range(0, original_shape[1]-x_ov, step_x):
            d_y = 0 if (i+window_size) < original_shape[2] else r_y
            d_x = 0 if (j+window_size) < original_shape[1] else r_x

            overlap_matrix[i-d_y:i+window_size, j-d_x:j+window_size, :] += 1
            if ov_map:
                ov_map_matrix[i-d_y:i+window_size, j-d_x:j+window_size, :] += 1

    # Mark the border of each crop in the map
    if ov_map:
        for i in range(0, original_shape[2]-y_ov, step_y):
            for j in range(0, original_shape[1]-x_ov, step_x):
                d_y = 0 if (i+window_size) < original_shape[2] else r_y
                d_x = 0 if (j+window_size) < original_shape[1] else r_x
                
                # Paint the grid
                ov_map_matrix[i-d_y:(i+window_size-1), j-d_x] = -4 
                ov_map_matrix[i-d_y:(i+window_size-1), (j+window_size-1-d_x)] = -4 
                ov_map_matrix[i-d_y, j-d_x:(j+window_size-1)] = -4 
                ov_map_matrix[(i+window_size-1-d_y), j-d_x:(j+window_size-1)] = -4 
  
    # Merge the overlapping crops
    cont = 0
    print("0) Merging the overlapping crops . . .")
    for k, img_num in tqdm(enumerate(range(0, total_images))):
        for i in range(0, original_shape[2]-y_ov, step_y):
            for j in range(0, original_shape[1]-x_ov, step_x):
                d_y = 0 if (i+window_size) < original_shape[2] else r_y
                d_x = 0 if (j+window_size) < original_shape[1] else r_x
                merged_data[k, i-d_y:i+window_size, j-d_x:j+window_size, :] += data[cont]
                cont += 1
           
        merged_data[k] = np.true_divide(merged_data[k], overlap_matrix)

    # Save a copy of the merged data with the overlapped regions colored as: 
    # green when 2 crops overlap, yellow when (2 < x < 8) and red when more than 
    # 7 overlaps are merged 
    if out_dir is not None and ov_map:
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
  
    print("**** New data shape is: {}".format(merged_data.shape))
    print("### END MERGE-OV-CROP ###")

    return merged_data


def merge_data_without_overlap(data, num, out_shape=(1, 1), grid=True):
    """Combine images from input data into a bigger one given shape. It is the 
       opposite function of crop_data().

       Args:                                                                    
            data (4D Numpy array): data to crop.                                
            E.g. (image_number, x, y, channels).

            num (int, optional): number of examples to convert.

            out_shape (2D int tuple, optional): number of horizontal and 
            vertical images to combine in a single one.

            grid (bool, optional): make the grid in the output image.

       Returns:                                                                 
            mixed_data (4D Numpy array): mixed data images.                 
            E.g. (image_number, x, y, channels).

            mixed_data_mask (4D Numpy array): mixed data masks.
            E.g. (image_number, x, y, channels).
    """

    print("### MERGE-CROP ###")

    width = data.shape[1]
    height = data.shape[2] 

    mixed_data = np.zeros((num, out_shape[1]*width, out_shape[0]*height, 
                           data.shape[3]), dtype=np.float32)
    cont = 0
    print("0) Merging crops . . .")
    for img_num in tqdm(range(0, num)):
        for i in range(0, out_shape[1]):
            for j in range(0, out_shape[0]):
                
                if cont == data.shape[0]:
                    return mixed_data

                mixed_data[img_num, (i*width):((i+1)*height), 
                           (j*width):((j+1)*height)] = data[cont]
                
                if grid:
                    mixed_data[
                        img_num,(i*width):((i+1)*height)-1, (j*width)] = 255
                    mixed_data[
                        img_num,(i*width):((i+1)*height)-1, ((j+1)*width)-1] = 255
                    mixed_data[
                        img_num,(i*height), (j*width):((j+1)*height)-1] = 255
                    mixed_data[
                        img_num,((i+1)*height)-1, (j*width):((j+1)*height)-1] = 255
                cont = cont + 1

    print("**** New data shape is: {}".format(mixed_data.shape))
    print("### END MERGE-CROP ###")
    return mixed_data


def merge_3D_data_with_overlap(data, o_vol_shape, data_mask=None,
                               overlap_z=0.5):
    """Merge 3D smaller volumes in a 3D full volume with a defined overlap.
       Reverse operation of crop_3D_data_with_overlap().

       Args:
            data (5D Numpy array): data to crop.
            E.g. (volume_number, z, x, y, channels).

            orig_vol_shape (4D int tuple): shape of the desired volumes to be 
            created.

            data_mask (4D Numpy array, optional): data mask to crop.
            E.g. (volume_number, z, x, y, channels).

            overlap_z (float, optional): amount of overlap on z dimension. The
            value must be on range [0, 1), that is, 0% or 99% of overlap.

       Returns:
            cropped_data (5D Numpy array): cropped image data.
            E.g. (vol_number, z, x, y, channels).

            cropped_data_mask (5D Numpy array, optional): cropped image data 
            masks. E.g. (image_number, z, x, y, channels).
    """ 
 
    print("### MERGE-3D-OV-CROP ###")

    orig_vol_shape = tuple(o_vol_shape[i] for i in [0, 2, 1, 3]) 

    merged_data = np.zeros((orig_vol_shape))
    if data_mask is not None:
        merged_data_mask = np.zeros((orig_vol_shape))
    ov_map_counter = np.zeros((orig_vol_shape))

    h_num = data.shape[1]
    v_num = data.shape[2]
    d_num = data.shape[3]

    # Minimun overlap
    if overlap_z == 0:
        vols_per_z = math.ceil(orig_vol_shape[2]/d_num)
        excess_z = (vols_per_z*d_num)-orig_vol_shape[2]
        step_z = d_num-int(excess_z/(vols_per_z-1))
        last_z = excess_z%(vols_per_z-1) 
        r_div = d_num-last_z 
    else:
        overlap_z = 1-overlap_z
        vols_per_z = math.ceil(orig_vol_shape[2]/(d_num*overlap_z))
        step_z = int(d_num*overlap_z)
        r_div = int(d_num-(orig_vol_shape[2]%(d_num*overlap_z)))

    vols_per_x = math.ceil(orig_vol_shape[0]/h_num)
    vols_per_y = math.ceil(orig_vol_shape[1]/v_num) 
    
    if r_div != 0:
        print("WARNING: Is assumed that the last {} slices in z have been filled"
              " with the last image to complete the volume, so they will be "
              "discarded".format(r_div))

    # Calculating overlap
    ov_x = (h_num*vols_per_x)-orig_vol_shape[0]
    ov_y = (v_num*vols_per_y)-orig_vol_shape[1]

    c = 0
    f_z = 0
    for z in range(vols_per_z):
        for x in range(vols_per_x):
            for y in range(vols_per_y):     
                ov_x_ = ov_x if x != 0 else 0
                ov_y_ = ov_y if y != 0 else 0
                if (z*step_z)+d_num < orig_vol_shape[2]:
                    merged_data[(x*h_num)-ov_x_:((x+1)*h_num)-ov_x_, 
                                (y*v_num)-ov_y_:((y+1)*v_num)-ov_y_,
                                z*step_z:(z*step_z)+d_num] += data[c]
   
                    if data_mask is not None: 
                        merged_data_mask[(x*h_num)-ov_x_:((x+1)*h_num)-ov_x_,
                                         (y*v_num)-ov_y_:((y+1)*v_num)-ov_y_,
                                         z*step_z:(z*step_z)+d_num] += \
                                            data_mask[c]

                    ov_map_counter[(x*h_num)-ov_x_:((x+1)*h_num)-ov_x_,
                                   (y*v_num)-ov_y_:((y+1)*v_num)-ov_y_,
                                   z*step_z:(z*step_z)+d_num] += 1
                else:
                    merged_data[(x*h_num)-ov_x_:((x+1)*h_num)-ov_x_,
                                (y*v_num)-ov_y_:((y+1)*v_num)-ov_y_,
                                z*step_z:orig_vol_shape[2]] +=  \
                        data[c,:,:,0:r_div-(f_z*step_z)] 
                    if data_mask is not None:
                        merged_data_mask[(x*h_num)-ov_x_:((x+1)*h_num)-ov_x_,
                                         (y*v_num)-ov_y_:((y+1)*v_num)-ov_y_,
                                         z*step_z:orig_vol_shape[2]] +=  \
                            data_mask[c,:,:,0:r_div-(f_z*step_z)]

                    ov_map_counter[(x*h_num)-ov_x_:((x+1)*h_num)-ov_x_,
                                   (y*v_num)-ov_y_:((y+1)*v_num)-ov_y_,
                                   z*step_z:orig_vol_shape[2]] += 1
                c += 1
                    
        # To adjust the final volumes in z
        if (z*step_z)+d_num > orig_vol_shape[2]:
            f_z += 1

    print("**** New data shape is: {}".format(merged_data.shape))
    print("### END MERGE-3D-OV-CROP ###")        

    merged_data = np.true_divide(merged_data, ov_map_counter)
    if data_mask is not None: 
        merged_data_mask = np.true_divide(merged_data_mask, ov_map_counter)
        return merged_data, merged_data_mask
    else:
        return merged_data
        


def check_crops(data, out_dim, num_examples=2, include_crops=True,
                out_dir="check_crops", suffix="_none_", grid=True):
    """Check cropped images by the function crop_data(). 
        
       Args:
            data (4D Numpy array): data to crop.
            E.g. (image_number, x, y, channels).
    
            out_dim (int 2D tuple): width and height of the image to be 
            constructed.

            num_examples (int, optional): number of examples to create.

            include_crops (bool, optional): to save cropped images or only the 
            image to contruct.  

            out_dir (string, optional): directory where the images will be save.

            suffix (string, optional): suffix to add in image names. 

            grid (bool, optional): make the grid in the output image.
    """
  
    print("### CHECK-CROPS ###")

    if out_dim[0] < data.shape[1] or out_dim[1] < data.shape[2]:
        raise ValueError("'out_dim' must be equal or greater than 'data.shape'")

    os.makedirs(out_dir, exist_ok=True)

    # Calculate horizontal and vertical image number for the data
    h_num = int(out_dim[0] / data.shape[1]) + (out_dim[0] % data.shape[1] > 0)
    v_num = int(out_dim[1] / data.shape[2]) + (out_dim[1] % data.shape[2] > 0)
    total = h_num*v_num

    if total*num_examples > data.shape[0]:
        num_examples = math.ceil(data.shape[0]/total)
        total = num_examples
        print("Requested num_examples too high for data. Set automatically to {}"\
              .format(num_examples))
    else:
        total = total*num_examples

    if include_crops:
        print("0) Saving cropped data images . . .")
        for i in tqdm(range(0, total)):
            # grayscale images
            if data.shape[3] == 1:
                im = Image.fromarray(data[i,:,:,0])
                im = im.convert('L')
            # RGB images
            else:
                aux = np.asarray( data[i,:,:,:], dtype="uint8" )
                im = Image.fromarray( aux, 'RGB' )

            im.save(os.path.join(out_dir,"c_" + suffix + str(i) + ".png"))

    print("0) Reconstructing {} images of ({}, {}) from {}".format(num_examples,\
          data.shape[1]*h_num, data.shape[2]*v_num, data.shape[1:]))
    m_data = merge_data_without_overlap(
        data, num_examples, out_shape=(h_num, v_num), grid=grid)
    print("1) Saving data mixed images . . .")
    for i in tqdm(range(0, num_examples)):
        im = Image.fromarray(m_data[i,:,:,0])
        im = im.convert('L')
        im.save(os.path.join(out_dir,"f" + suffix + str(i) + ".png"))

    print("### END CHECK-CROP ###")


def check_binary_masks(path):
    """Check wheter the data masks is binary checking the a few random images of
       the given path. If the function gives no error one should assume that the
       masks are correct.

       Args:
            path (str): path to the data mask.
    """
    print("Checking wheter the images in {} are binary . . .".format(path))

    ids = sorted(next(os.walk(path))[2])

    # Check only 4 random images or less if there are not as many
    num_sample = [4, len(ids)]
    numbers = random.sample(range(0, len(ids)), min(num_sample))
    for i in numbers:
        img = imread(os.path.join(path, ids[i]))
        values, _ = np.unique(img, return_counts=True)
        if len(values) > 2 :
            raise ValueError(
                "Error: given masks are not binary. Please correct the images "
                "before training. (image: {})\nValues: {}"\
                .format(os.path.join(path, ids[i]), values))


def prepare_3D_volume_data(X, Y, shape=(256, 256, 82), overlap=False,
                           use_rest=True, ov=0.5):                          
    """Prepare given data into 3D subvolumes to train a 3D network.

       Args:
            X (Numpy 4D array): data. E.g. (x, y, image_number, channels).      
                                                                                
            Y (Numpy 4D array): mask data.  E.g. (x, y, image_number, channels).
                                                                                
            shape (3D int tuple, optional): dimension of the desired images.           

            overlap (bool, optional): flag to create subvolumes with overlap
            forcing to use all the data. If False as much as possible subvolumes 
            of the desired shape will be created with the available data, and 
            the rest of images will be dropped out (depens on 'use_rest' value).
            Activate it when dealing with test data, as all the images must be 
            present.
        
            use_rest (bool, optional): flag to try to use the rest of the images
            to create more training data using mirroring. 
            
            ov (float, optional): amount of overlap on z dimension. The value 
            must be on range [0, 1), that is, 0% or 99% of overlap.

       Returns:
            X_prep (Numpy 5D array): X data separated in different subvolumes 
            with the desired shape. E.g. (subvolume_number, ) + shape
            
            Y_prep (Numpy 5D array): Y data separated in different subvolumes     
            with the desired shape. E.g. (subvolume_number, ) + shape
            
            mirrored_subvols (int): number of subvolumes created with the rest
            of the images. A valid value is returned when 'use_rest' is True.
    """
                                                                            
    if X.shape != Y.shape:                                                      
        raise ValueError("The shape of X and Y must be the same")               
    if X.ndim != 4 or Y.ndim != 4:                                              
        raise ValueError("X or Y must be a 4D Numpy array")                     
    print("shape: {}".format(shape))
    if len(shape) != 4:
        raise ValueError("'shape' must be 4D")
    
    # Calculate the rest                                                        
    rest = X.shape[2] % shape[2]                                                

    mirrored_subvols = 0
    if overlap:
        print("Overlap cropping selected")

        X, Y = crop_3D_data_with_overlap(X, shape, data_mask=Y, overlap_z=ov)
        return X, Y, mirrored_subvols

    if overlap and use_rest:
        print("'use_rest' is involved in 'overlap': ignoring . . .'")
    if rest != 0 and use_rest == False:                                                               
        print("As the number of images required to form a stack 3D is not "
              "multiple of images provided and no overlap+use_rest is selected,"
              " the last {} image(s) will be unused".format(rest))
                                                                                
    # Calculate crops are per axis                                      
    h_num = int(X.shape[0]/shape[0])
    v_num = int(X.shape[1]/shape[1])
    crops_per_image = h_num*v_num                                               
                                                                                
    num_sub_volum = int(math.floor(X.shape[2]/shape[2])*crops_per_image)             
                                                                                
    X_prep = np.zeros((num_sub_volum, ) + shape)
    Y_prep = np.zeros((num_sub_volum, ) + shape)
                                                                                
    # Reshape the data to generate desired 3D subvolumes                        
    print("Generating 3D subvolumes . . .")                                     
    print("Converting {} data to {}".format(X.shape, X_prep.shape))             
    print("Filling [0:{}] subvolumes with [0:{}] images . . ."                  
          .format(crops_per_image-1, shape[2]-1))                               
    subvolume = 0                                                               
    vol_slice = 0                                                               
    total_subvol_filled = 0                                                     
    for k in range(X.shape[2]-rest):                                                 
        for i in range(h_num):                                                  
            for j in range(v_num):                                              
                im = X[(i*shape[0]):((i+1)*shape[1]),(j*shape[0]):((j+1)*shape[1]),k]
                mask = Y[(i*shape[0]):((i+1)*shape[1]),(j*shape[0]):((j+1)*shape[1]),k]
                  
                X_prep[subvolume, ..., vol_slice, :] = im                               
                Y_prep[subvolume, ..., vol_slice, :] = mask                             
                                                                                
                subvolume += 1                                                  
                if subvolume == (total_subvol_filled+crops_per_image):
                    subvolume = total_subvol_filled
                    vol_slice += 1                                              
                                                                                
                    # Reached this point we will have filled part of            
                    # the subvolumes                                            
                    if vol_slice == shape[2] and (k+1) != (X.shape[2]-rest):                                   
                        total_subvol_filled += crops_per_image                  
                        subvolume = total_subvol_filled
                        vol_slice = 0                                           
                                                                                
                        print("Filling [{}:{}] subvolumes with [{}:{}] "        
                              "images . . .".format(total_subvol_filled,        
                              total_subvol_filled+crops_per_image-1, k+1,       
                              k+shape[2]-1))
    print("Data prepared shape: {}".format(X_prep.shape))

    if rest != 0 and use_rest:
        if rest*2 < shape[2]:
            print("As mirroring will be used at least the rest of the images need"
                  " to be half of the size in z provided. Rest={} ; z-images={}"
                  .format(rest, shape[2]))
        else:
            mirrored_subvols = crops_per_image
            print("Last {} images will be used to create {} more subvolumes "
                  "doing mirroring".format(rest, mirrored_subvols))

            X_prep2 = np.zeros((crops_per_image, ) + shape)
            Y_prep2 = np.zeros((crops_per_image, ) + shape)
            
            img_to_mirror = shape[2]-rest
            subvolume = 0
            vol_slice = 0
            s = 0
            for k in range(X.shape[2]-rest, X.shape[2]+img_to_mirror):
                if k >= X.shape[2]:
                    s += 1
                for i in range(h_num):
                    for j in range(v_num):
                        im = X[k-s, (i*shape[0]):((i+1)*shape[1]),(j*shape[0]):((j+1)*shape[1])]
                        mask = Y[k-s, (i*shape[0]):((i+1)*shape[1]),(j*shape[0]):((j+1)*shape[1])]

                        X_prep2[subvolume, ..., vol_slice] = im
                        Y_prep2[subvolume, ..., vol_slice] = mask
                        
                        subvolume += 1
                        if subvolume == crops_per_image:
                            subvolume = 0
                            vol_slice += 1

            X_prep = np.concatenate((X_prep, X_prep2), axis=0)
            Y_prep = np.concatenate((Y_prep, Y_prep2), axis=0)
            print("Rest used. New data prepared shape is now: {}"
                  .format(X_prep.shape))

    return X_prep, Y_prep, mirrored_subvols


def img_to_onehot_encoding(img, num_classes=2):
    """Converts image given into one-hot encode format.
       Args:
            img (Numpy 4D array): data. E.g. (z, x, y, channels).
            
            num_classes (int, optional): number of classes to distinguish.
       
       Return: 
            one_hot_labels (Numpy 4D array): data one-hot encoded. 
            E.g. (z, x, y, num_classes)

    """

    if img.ndim == 4:
        shape = img.shape[:3]+(num_classes,)
    else:
        shape = img.shape[:2]+(num_classes,)

    encoded_image = np.zeros(shape, dtype=np.int8)

    for i in range(num_classes):
        if img.ndim == 4:
            encoded_image[:,:,:,i] = np.all(img.reshape((-1,1)) == i, axis=1).reshape(shape[:3])
        else:
            encoded_image[:,:,i] = np.all(img.reshape((-1,1)) == i, axis=1).reshape(shape[:2])

    return encoded_image
