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
       data. 

       Parameters
       ----------                                                            
       train_path : str
           Path to the training data.                

       train_mask_path : str
           Path to the training data masks.     

       test_path : str
           Path to the test data.                     

       test_mask_path : str
           Path to the test data masks.          

       image_train_shape : 3D tuple
           Dimensions of the images to load. E.g. ``(x, y, channels)``.

       image_test_shape : 3D tuple
           Dimensions of the images to load. E.g. ``(x, y, channels)``.

       create_val : bool, optional
           If true validation data is created (from the train data).                                                    

       val_split : float, optional
            % of the train data used as validation (value between ``0`` and ``1``).

       seedValue : int, optional
            Seed value.

       shuffle_val : bool, optional
            Take random training examples to create validation data.

       e_d_data : list of str, optional
           List of paths where the extra data of other datasets are stored. If
           ``make_crops`` is not enabled, these extra datasets must have the 
           same image shape as the main dataset since they are going to be 
           stacked in a unique array.

       e_d_mask : list of str, optional
           List of paths where the extra data mask of other datasets are stored. 
           Same constraints as ``e_d_data``.  

       e_d_data_dim : list of 3D int tuple, optional
           List of shapes of the extra datasets provided. Same constraints as 
           ``e_d_data``.

       e_d_dis : list of float, optional
           Discard percentages of the extra datasets provided. Values between 
           ``0`` and ``1``. Same constraints as ``e_d_data``.

       num_crops_per_dataset : int, optional
           Number of crops per extra dataset to take into account. Useful to 
           ensure that all the datasets have the same weight during network 
           trainning. 

       make_crops : bool, optional
           To make crops on data.

       crop_shape : 3D int tuple, optional
           Shape of the crops. E.g. ``(x, y, channels)``.

       check_crop : bool, optional
           To save the crops made to ensure they are generating as one wish.

       check_crop_path : str, optional
           Path to save the crop samples.

       d_percentage : int, optional
           Number between 0 and 100. The images having less percentage of 
           foreground pixels than the given percentage are discarded.

       Returns
       -------                                                         
       X_train : 4D Numpy array
           Train images. E.g. ``(num_of_images, y, x, channels)``.
        
       Y_train : 4D Numpy array
           Train images' mask. E.g. ``(num_of_images, y, x, channels)``.

       X_val : 4D Numpy array, optional
           Validation images (``create_val==True``). E.g. ``(num_of_images, 
           y, x, channels)``.

       Y_val : 4D Numpy array, optional
           Validation images' mask (``create_val==True``). E.g. 
           ``(num_of_images, y, x, channels)``.

       X_test : 4D Numpy array
           Test images. E.g. ``(num_of_images, y, x, channels)``.

       Y_test : 4D Numpy array
           Test images' mask. E.g. ``(num_of_images, y, x, channels)``.

       orig_test_shape : 4D int tuple
           Test data original shape. E.g. ``(num_of_images, x, y, channels)``

       norm_value : int
           mean of the train data in case we need to make a normalization.

       crop_made : bool
           True if crops have been made.

       Examples
       --------
       ::
        
           # EXAMPLE 1
           # Case where we need to load the data (creating a validation split)
           train_path = "data/train/x"
           train_mask_path = "data/train/y"
           test_path = "data/test/y"
           test_mask_path = "data/test/y"

           # Original image shape is (1024, 768, 165), so each image shape should be this:
           img_train_shape = (1024, 768, 1)
           img_test_shape = (1024, 768, 1)

           X_train, Y_train, X_val,
           Y_val, X_test, Y_test,
           orig_test_shape, norm_value, crops_made = load_and_prepare_2D_data(
               train_path, train_mask_path, test_path, test_mask_path, img_train_shape,
               img_test_shape, val_split=0.1, shuffle_val=True, make_crops=False)
               

           # The function will print the shapes of the generated arrays. In this example:
           #     *** Loaded train data shape is: (148, 768, 1024, 1)
           #     *** Loaded validation data shape is: (17, 768, 1024, 1)
           #     *** Loaded test data shape is: (165, 768, 1024, 1)
           #
           # Notice height and width swap because of Numpy ndarray terminology 
            

           # EXAMPLE 2 
           # Same as the first example but creating patches of (256x256)
           X_train, Y_train, X_val,
           Y_val, X_test, Y_test,
           orig_test_shape, norm_value, crops_made = load_and_prepare_2D_data(
               train_path, train_mask_path, test_path, test_mask_path, img_train_shape, 
               img_test_shape, val_split=0.1, shuffle_val=True, make_crops=True,
               crop_shape=(256, 256, 1), check_crop=True, check_crop_path="check_folder")

           # The function will print the shapes of the generated arrays. In this example:
           #    *** Loaded train data shape is: (1776, 256, 256, 1)
           #    *** Loaded validation data shape is: (204, 256, 256, 1)
           #    *** Loaded test data shape is: (1980, 256, 256, 1)


           # EXAMPLE 3
           # Same as the first example but definig extra datasets to be loaded and stacked together 
           # with the main dataset. Extra variables to be defined: 
           extra_datasets_data_list.append('/data2/train/x')
           extra_datasets_mask_list.append('/data2/train/y')
           extra_datasets_data_dim_list.append((877, 967, 1))
           extra_datasets_discard.append(0) # To not discard any image 

           X_train, Y_train, X_val,                                             
           Y_val, X_test, Y_test,                                               
           orig_test_shape, norm_value, crops_made = load_and_prepare_2D_data(  
               train_path, train_mask_path, test_path, test_mask_path, img_train_shape, 
               mg_test_shape, val_split=0.1, shuffle_val=True, make_crops=True,
               crop_shape=(256, 256, 1), check_crop=True, check_crop_path="check_folder"
               e_d_data=extra_datasets_data_list, e_d_mask=extra_datasets_mask_list, 
               e_d_data_dim=extra_datasets_data_dim_list, e_d_dis=extra_datasets_discard)
                
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
            print("5.{}) extra dataset in {} . . .".format(i, e_d_data[i])) 
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
                             test_subvol_shape, train_subvol_shape, 
                             create_val=True, shuffle_val=True, val_split=0.1, 
                             seedValue=42, random_subvolumes_in_DA=False, 
                             overlap_train=False, use_rest_train=True, ov=(0,0,0)):         
    """Load train, validation and test images from the given paths to create a 
       3D data representation. All the test data will be used to create a 3D
       volume of ``test_subvol_shape`` shape (taking into account ``ov``).

       Parameters
       ----------                                                            
       train_path : str
           Path to the training data.                

       train_mask_path : str
           Path to the training data masks.     

       test_path : str                                                          
           Path to the test data.                                               
                                                                                
       test_mask_path : str                                                     
           Path to the test data masks.                                         
                                                                                
       image_train_shape : 3D tuple
           Dimensions of the images to load. E.g. ``(x, y, channels)``.
                                                                                
       image_test_shape : 3D tuple
           Dimensions of the images to load. E.g. ``(x, y, channels)``.

       train_subvol_shape : 4D tuple
            Shape of the train subvolumes to create. E.g. ``(x, y, z, channels)``.

       test_subvol_shape : 4D tuple
            Shape of the test subvolumes to create. E.g. ``(x, y, z, channels)``.

       create_val : bool, optional                                              
           If true validation data is created (from the train data).                                                    

       shuffle_val : bool, optional                                             
            Take random training examples to create validation data.
                                                                                
       val_split : float, optional                                              
            % of the train data used as validation (value between ``0`` and ``1``).
                                                                                
       seedValue : int, optional                                                
            Seed value.                                                     

       random_subvolumes_in_DA : bool, optional
           To advice the method that not preparation of the data must be done, 
           as random subvolumes will be created on DA, and the whole volume will 
           be used for that.

       overlap_train : bool, optional
           To make training subvolumes as overlapping patches using ``ov``.

       use_rest_train : bool, optional
           Wheter to use the train data remainder when the subvolume shape 
           selected has no exact division with it. More info at :func:`~crop_data`
           function.

       ov : Tuple of 3 floats, optional                                         
           Amount of minimum overlap on x, y and z dimensions. The values must 
           be on range ``[0, 1)``, that is, ``0%`` or ``99%`` of overlap.      
           E. g. ``(x, y, z)``.   

       Returns
       -------                                                         
       X_train : 5D Numpy array                                                 
           Train images. E.g. ``(num_of_images, y, x, z, channels)``.               
                                                                                
       Y_train : 5D Numpy array                                                 
           Train images' mask. E.g. ``(num_of_images, y, x, z, channels)``.         
                                                                                
       X_val : 5D Numpy array, optional                                         
           Validation images (``create_val==True``). E.g. ``(num_of_images,      
           y, x, z, channels)``.                                                   
                                                                                
       Y_val : 5D Numpy array, optional                                         
           Validation images' mask (``create_val==True``). E.g. ``(num_of_images,
           y, x, z, channels)``.                                  
                                                                                
       X_test : 5D Numpy array                                                  
           Test images. E.g. ``(num_of_images, y, x, z, channels)``.                

       Y_test : 5D Numpy array      
           Test images' mask. E.g. ``(num_of_images, y, x, z, channels)``.  

       orig_test_shape : 4D int tuple
           Test data original shape. E.g. ``(num_of_images, x, y, channels)``.

       norm_value : int
           Normalization value calculated.

       Examples
       --------
       ::

           # EXAMPLE 1
           # Case where we need to load the data and creating a validation split
           train_path = "data/train/x"
           train_mask_path = "data/train/y"
           test_path = "data/test/y"
           test_mask_path = "data/test/y"

           # Original image shape is (1024, 768, 165), so each image shape should be this:
           img_train_shape = (1024, 768, 1)
           img_test_shape = (1024, 768, 1)

           # 3D subvolume shape needed
           train_3d_shape = (80, 80, 80, 1)
           test_3d_shape = (80, 80, 80, 1)

           X_train, Y_train, X_val,
           Y_val, X_test, Y_test,
           orig_test_shape, norm_value = load_and_prepare_3D_data(
               train_path, train_mask_path, test_path, test_mask_path, img_train_shape,
               img_test_shape, val_split=0.1, create_val=True, shuffle_val=True,
               ov_=(0,0,0), train_subvol_shape=train_3d_shape,
               test_subvol_shape=test_3d_shape)

           # The function will print the shapes of the generated arrays. In this example:
           #     *** Loaded train data shape is: (194, 80, 80, 80, 1)
           #     *** Loaded validation data shape is: (22, 80, 80, 80, 1)
           #     *** Loaded test data shape is: (390, 80, 80, 80, 1)
           #
           # For the test data, 390 subvolumes have been created. As you may noticed, 
           # a minimum overlap is set (ov=(0,0,0)), leading to more subvolume
           # creation compared to train+val.

           # EXAMPLE 2                                                          
           # As the example 1 but defining a minimum overlap of 50% in both train
           # and test data. Notice how the number of subvolumes has been increased 
           #                                                                    
           X_train, Y_train, X_val,                                             
           Y_val, X_test, Y_test,                                               
           orig_test_shape, norm_value = load_and_prepare_3D_data(              
               train_path, train_mask_path, test_path, test_mask_path, img_train_shape,
               img_test_shape, val_split=0.1, create_val=True, shuffle_val=True,
               overlap_train=True, ov=(0.5,0.5,0.5), train_subvol_shape=train_3d_shape,              
               test_subvol_shape=test_3d_shape)                                 
                                                                                
           # The function will print the shapes of the generated arrays. In this example:
           #     *** Loaded train data shape is: (1710, 80, 80, 80, 1)
           #     *** Loaded validation data shape is: (190, 80, 80, 80, 1)       
           #     *** Loaded test data shape is: (1900, 80, 80, 80, 1)            
           #

           # EXAMPLE 3
           # As the example 1 but when random_subvolumes_in_DA is True and no validation data
           # needs to be created. The test data should be of the same shape as the example 1.
           # However, the returned train data will be the same but adding an extra dimension 
           # on the first axis. This is made to consider the entire data as a unique subvolume.
           # More information about this in 3D generator. 
           # 
           X_train, Y_train, X_val,
           Y_val, X_test, Y_test,
           orig_test_shape, norm_value = load_and_prepare_3D_data(
               train_path, train_mask_path, test_path, test_mask_path, img_train_shape,
               img_test_shape, create_val=False, random_subvolumes_in_DA=True, ov=(0,0,0),
               train_subvol_shape=train_3d_shape, test_subvol_shape=test_3d_shape)

           # The function will print the shapes of the generated arrays. In this example:
           #     *** Loaded train data shape is: (1, 768, 1024, 165, 1)
           #     *** Loaded test data shape is: (390, 80, 80, 80, 1)
           # Notice height and width swap because of Numpy ndarray terminology
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

    orig_test_shape = tuple(Y_test.shape[i] for i in [0, 1, 2, 3])

    if not random_subvolumes_in_DA:
        print("Cropping train data subvolumes . . .")
        if overlap_train:
            X_test, Y_test = crop_3D_data_with_overlap(
                X_test, test_subvol_shape, data_mask=Y_test, overlap=ov) 
        else:
            X_train, Y_train = crop_3D_data(
                X_train, train_subvol_shape, use_rest=use_rest_train,
                data_mask=Y_train)

    print("Cropping test data subvolumes . . .")
    X_test, Y_test = crop_3D_data_with_overlap(
        X_test, test_subvol_shape, data_mask=Y_test, overlap=ov)

    # Create validation data splitting the train
    if create_val:
        X_train, X_val, \
        Y_train, Y_val = train_test_split(
            X_train, Y_train, test_size=val_split, shuffle=shuffle_val, 
            random_state=seedValue)

    # Convert the original volumes as they were a unique subvolume
    if random_subvolumes_in_DA:                                                 
        X_train = np.expand_dims(np.transpose(X_train, (1,2,0,3)), axis=0)      
        Y_train = np.expand_dims(np.transpose(Y_train, (1,2,0,3)), axis=0)    
        if create_val:
            X_val = np.expand_dims(np.transpose(X_val, (1,2,0,3)), axis=0)
            Y_val = np.expand_dims(np.transpose(Y_val, (1,2,0,3)), axis=0)

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

       Parameters
       ----------
       data_dir : str 
           Path to read the data from.
   
       shape : 3D int tuple
           Shape of the data. E.g. ``(x, y, channels)``.
       
       Returns
       -------        
       data : 4D Numpy array
           Data loaded. E.g. ``(num_of_images, y, x, channels)``.

       Examples
       --------
       ::

           # EXAMPLE 1
           # Case where we need to load 165 images of shape (1024, 768)
           data_path = "data/train/x"
           data_shape = (1024, 768, 1)
           
           load_data_from_dir(data_path, data_shape) 

           # The function will print the shape of the created array. In this example:
           #     *** Loaded data shape is (165, 768, 1024, 1)
           # Notice height and width swap because of Numpy ndarray terminology
    """

    print("Loading data from {}".format(data_dir))
    ids = sorted(next(os.walk(data_dir))[2])
    data = np.zeros((len(ids), ) + shape, dtype=np.float32)

    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        img = imread(os.path.join(data_dir, id_))

        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)

        data[n] = img

    print("*** Loaded data shape is {}".format(data.shape))
    return data


def crop_data(data, crop_shape, data_mask=None, force_shape=(0, 0), 
              d_percentage=0):                          
    """Crop data into smaller pieces of ``crop_shape``. If there is no exact 
       division between the data shape and ``crop_shape`` in a specific dimension
       zeros will be added.
        
       The opposite function is :func:`~merge_data_without_overlap`.

       Parameters
       ----------                                                            
       data : 4D Numpy array
           Data to crop. E.g. ``(num_of_images, x, y, channels)``.

       crop_shape : 3D int tuple
           Output image shape. E.g. ``(x, y, channels)``.

       data_mask : 4D Numpy array, optional
           Data masks to crop. E.g. ``(num_of_images, x, y, channels)``.

       force_shape : 2D int tuple, optional
           Force number of horizontal and vertical crops to the given numbers. 
           E. g. ``(4, 5)`` should create ``20`` crops: ``4`` rows and ``5`` crop 
           per each row.

       d_percentage : int, optional
           Number between ``0`` and ``100``. The images that have less foreground 
           pixels than the given number will be discarded. Only available if 
           ``data_mask`` is provided.
                                                                        
       Returns
       -------                                                         
       cropped_data : 4D Numpy array. 
           Cropped data images. E.g. ``(num_of_images, x, y, channels)``.

       cropped_data_mask : 4D Numpy array
           Cropped data masks. E.g. ``(num_of_images, x, y, channels)``.

       force_shape : 2D int tuple
           Number of horizontal and vertical crops made. Useful for future      
           crop/merge calls. 
        
       Examples
       --------
       ::

           # EXAMPLE 1
           # Divide in (256, 256, 1) crops a given data 
           X_train = np.ones((165, 768, 1024, 1)) 
           Y_train = np.ones((165, 768, 1024, 1)) 
           
           X_train, Y_train, _ = crop_data(X_train, (256, 256, 1), data_mask=Y_train)
        
           # The function will print the shape of the created array. In this example:
           #     **** New data shape is: (1980, 256, 256, 1)

           # EXAMPLE 2
           # Divide in (256, 256, 1) crops a given data that has no exact division
           # with that shape
           X_train = np.ones((165, 700, 900))
           Y_train = np.ones((165, 700, 900))

           X_train, Y_train, _ = crop_data(X_train, (256, 256, 1), data_mask=Y_train)

           # The function will print the shape of the created array. In this example:
           #     **** New data shape is: (1980, 256, 256, 1)
           # Here some of the crops, concretelly the last in height and width, will
           # have a black part (which are zeros)
    
       See :func:`~check_crops` function for a visual example.
    """                                                                 

    print("### CROP ###")                                                                    
    print("Cropping [{},{}] images into {} . . .".format(data.shape[1], \
          data.shape[2], crop_shape)) 
  
    # Calculate the number of images to be generated                    
    if force_shape == (0, 0):
        h_num = math.ceil(data.shape[1] / crop_shape[0])
        v_num = math.ceil(data.shape[2] / crop_shape[1])
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


def crop_data_with_overlap(data, data_mask, window_size, n_crops):
    """Crop data into small square pieces with overlap. The difference with
       :func:`~crop_data` is that this function allows you to create patches with 
       overlap. 

       As the inference should be make with the same crop shape as the one used 
       to train the network, sometimes that shape has no exact division with the 
       test data shape. On that case, instead of use :func:`~crop_data` that 
       would add zeros in the last crop of each row and column, you can use this
       function and create the crops using only image data information. You can 
       calculate the number of crops you want, and the function will calculate 
       the minimum overlap along x and y axis that satisfies that number. 

       The opposite function is :func:`~merge_data_with_overlap`.

       Parameters
       ----------
       data : 4D Numpy array
           Data to crop. E.g. ``(num_of_images, x, y, channels)``.

       data_mask : 4D Numpy array
           Data mask to crop. E.g. ``(num_of_images, x, y, channels)``.

       window_size : int
           Crop size.

       n_crops : int
           Number of crops to create. Must an even number.

       Returns
       -------
       cropped_data : 4D Numpy array
           Cropped image data. E.g. ``(num_of_images, x, y, channels)``.

       cropped_data_mask : 4D Numpy array)
           Cropped image data masks. E.g. ``(num_of_images, x, y, channels)``.

       Examples
       --------
       ::

           # EXAMPLE 1
           # Divide in 16 crops of (256, 256) a given data
           X_train = np.ones((165, 768, 1024, 1))
           Y_train = np.ones((165, 768, 1024, 1))

           X_test, Y_test = crop_data_with_overlap(X_test, Y_test, 256, 16)

           # The function will print the shape of the created array. In this example:
           #     **** New data shape is: (2640, 256, 256, 1)
           # Notice that, as we need 16 crops and only with 12 can be cover all 
           # the image, some of the crops have being overlapped, leading to more
           # crops than using 'crop_data' function (where 1980 are created)
    """

    print("### OV-CROP ###")
    print("Cropping {} images into ({}, {}) with overlapping. . ."\
          .format(data.shape[1:], window_size, window_size))

    if n_crops % 2 != 0:
        raise ValueError("'n_crops' must be an even number")
    if window_size > data.shape[1]:
        raise ValueError("'window_size' greater than {}".format(data.shape[1]))
    if window_size > data.shape[2]:
        raise ValueError("'window_size' greater than {}".format(data.shape[2]))

    # Crop data
    total_cropped = data.shape[0]*n_crops
    cropped_data = np.zeros((total_cropped, window_size, window_size,
                             data.shape[3]), dtype=np.float32)
    cropped_data_mask = np.zeros((total_cropped, window_size, window_size,
                                 data_mask.shape[3]), dtype=np.float32)

    # Find the mininum overlap configuration with the number of crops to create
    min_d = sys.maxsize
    rows = 1
    columns = 1
    for i in range(1, int(n_crops/2)+1, 1):
        if n_crops % i == 0 and abs((n_crops/i) - i) < min_d:
            min_d = abs((n_crops/i) - i)
            rows = i
            columns = int(n_crops/i)
        
    print("The minimum overlap has been found with rows={} and columns={}"\
          .format(rows, columns))

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
    print("0) Cropping data with the minimum overlap . . .")
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


def crop_3D_data_with_overlap(data, vol_shape, data_mask=None, overlap=(0,0,0)):
    """Crop 3D data into smaller volumes with the minimum overlap in z.

       The opposite function is :func:`~merge_3D_data_with_overlap`.

       Parameters
       ----------
       data : 4D Numpy array
           Data to crop. E.g. ``(num_of_images, x, y, channels)``.

       vol_shape : 4D int tuple
           Shape of the volumes to created. E.g. ``(x, y, z, channels)``.
        
       data_mask : 4D Numpy array, optional
            Data mask to crop. E.g. ``(num_of_images, x, y, channels)``.
    
       overlap : Tuple of 3 floats, optional
            Amount of minimum overlap on x, y and z dimensions. The values must
            be on range ``[0, 1)``, that is, ``0%`` or ``99%`` of overlap. 
            E. g. ``(x, y, z)``.

       Returns
       -------
       cropped_data : 5D Numpy array
           Cropped image data. E.g. ``(vol_number, x, y, z, channels)``.

       cropped_data_mask : 5D Numpy array, optional
           Cropped image data masks. E.g. ``(vol_number, x, y, z, channels)``.
        
       Examples
       --------
       ::

           # EXAMPLE 1   
           # Following the example introduced in load_and_prepare_3D_data function, the 
           # cropping of a volume with shape (165, 1024, 765) should be done by the 
           # following call: 

           X_train = np.ones((165, 768, 1024, 1))
           Y_train = np.ones((165, 768, 1024, 1))

           X_train, Y_train = crop_3D_data_with_overlap(
                X_train, (80, 80, 80, 1), data_mask=Y_train, overlap=(0.5,0.5,0.5))

           # The function will print the shape of the generated arrays. In this example:
           #     **** New data shape is: (1900, 80, 80, 80, 1)

       A visual explanation of the process:                                     
                                                                                
       .. image:: img/crop_3D_ov.png                                               
           :width: 80%                                                          
           :align: center
    
       Note: this image do not respect the proportions.

       ::  

           # EXAMPLE 2 
           # Same data crop but without overlap
          
           X_train, Y_train = crop_3D_data_with_overlap(                        
                X_train, (80, 80, 80, 1), data_mask=Y_train, overlap=(0,0,0))
           
           # The function will print the shape of the generated arrays. In this example:  
           #     **** New data shape is: (390, 80, 80, 80, 1)
           #
           # Notice how differs the amount of subvolumes created compared to the 
           # first example
    """

    print("### 3D-OV-CROP ###")
    print("Cropping {} images into {} with overlapping . . ."
          .format(data.shape, vol_shape))
    print("Minimum overlap selected: {}".format(overlap))

    vol_shape = tuple(vol_shape[i] for i in [2, 0, 1, 3])

    if (overlap[0] >= 1 or overlap[0] < 0)\
       and (overlap[1] >= 1 or overlap[1] < 0)\
       and (overlap[2] >= 1 or overlap[2] < 0): 
        raise ValueError("'overlap' values must be floats between range [0, 1)")
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
  
    # Calculate overlapping variables
    overlap_x = 1 if overlap[0] == 0 else 1-overlap[0]                       
    overlap_y = 1 if overlap[1] == 0 else 1-overlap[1]                       
    overlap_z = 1 if overlap[2] == 0 else 1-overlap[2]

    # X
    vols_per_x = math.ceil(data.shape[1]/(vol_shape[1]*overlap_x))
    excess_x = int((vols_per_x*vol_shape[1]*overlap_x)-data.shape[1])
    step_x = int(vol_shape[1]*overlap_x)-int(excess_x/(vols_per_x-1))
    last_x = excess_x%(vols_per_x-1)

    # Y
    vols_per_y = math.ceil(data.shape[2]/(vol_shape[2]*overlap_y))
    excess_y = int((vols_per_y*vol_shape[2]*overlap_y)-data.shape[2])           
    step_y = int(vol_shape[2]*overlap_y)-int(excess_y/(vols_per_y-1))      
    last_y = excess_y%(vols_per_y-1)

    # Z
    vols_per_z = math.ceil(data.shape[0]/(vol_shape[0]*overlap_z))              
    excess_z = int((vols_per_z*vol_shape[0]*overlap_z)-data.shape[0])           
    step_z = int(vol_shape[0]*overlap_z)-int(excess_z/(vols_per_z-1))           
    last_z = excess_z%(vols_per_z-1)

    vols_per_x += 1 if overlap_x == 1 else 0
    vols_per_y += 1 if overlap_y == 1 else 0
    vols_per_z += 1 if overlap_z == 1 else 0

    last_x += int(excess_x/(vols_per_x-1)) if overlap_x != 1 else 0
    last_y += int(excess_y/(vols_per_y-1)) if overlap_y != 1 else 0
    last_z += int(excess_z/(vols_per_z-1)) if overlap_z != 1 else 0

    # Real overlap calculation for printing 
    real_ov_x = (vol_shape[0]-step_x)/vol_shape[0]
    real_ov_y = (vol_shape[1]-step_y)/vol_shape[1]
    real_ov_z = (vol_shape[2]-step_z)/vol_shape[2]
    print("Real overlapping: {}".format((real_ov_x,real_ov_y,real_ov_z)))

    print("{},{},{} patches per x,y,z axis"
          .format((vols_per_x-1),(vols_per_y-1),(vols_per_z-1)))

    total_vol = (vols_per_z-1)*(vols_per_x-1)*(vols_per_y-1)
    cropped_data = np.zeros((total_vol,) + vol_shape)
    if data_mask is not None:
        cropped_data_mask = np.zeros((total_vol,) + vol_shape)

    c = 0
    for z in range(vols_per_z-1):
        for x in range(vols_per_x-1):
            for y in range(vols_per_y-1):
                d_x = 0 if (x*step_x+vol_shape[1]) < data.shape[1] else last_x
                d_y = 0 if (y*step_y+vol_shape[2]) < data.shape[2] else last_y
                d_z = 0 if (z*step_z+vol_shape[0]) < data.shape[0] else last_z

                cropped_data[c] = \
                    data[z*step_z-d_z:(z*step_z)+vol_shape[0]-d_z, x*step_x-d_x:x*step_x+vol_shape[1]-d_x, y*step_y-d_y:y*step_y+vol_shape[2]-d_y]
                if data_mask is not None:
                    cropped_data_mask[c] = \
                        data_mask[z*step_z-d_z:(z*step_z)+vol_shape[0]-d_z, x*step_x-d_x:x*step_x+vol_shape[1]-d_x, y*step_y-d_y:y*step_y+vol_shape[2]-d_y]
                c += 1

    cropped_data = cropped_data.transpose(0,2,3,1,4)

    print("**** New data shape is: {}".format(cropped_data.shape))
    print("### END 3D-OV-CROP ###")

    if data_mask is not None:
        cropped_data_mask = cropped_data_mask.transpose(0,2,3,1,4)
        return cropped_data, cropped_data_mask
    else:
        return cropped_data


def merge_data_with_overlap(data, original_shape, window_size, n_crops, 
                            out_dir=None, ov_map=True, ov_data_img=0):
    """Merge data with an amount of overlap.
    
       The opposite function is :func:`~crop_data_with_overlap`.

       Parameters
       ----------
       data : 4D Numpy array
           Data to merge. E.g. ``(num_of_images, x, y, channels)``.

       original_shape : 4D int tuple
           Shape of the original data. E.g. ``(num_of_images, x, y, channels)``

       window_size : int    
           Crop size.

       n_crops : int
           Number of crops to merge.

       out_dir : str, optional
           If provided an image that represents the overlap made will be saved. 
           The image will be colored as follows: green region when ``==2`` crops 
           overlap, yellow when ``2 < x < 8`` and red when ``=<8`` or more 
           overlaps are merged.

       ov_map : bool, optional
           Whether to create overlap map.

       ov_data_img : int, optional
           Number of the image on the data to create the overlap map.

       Returns
       -------
       merged_data : 4D Numpy array
           Merged image data. E.g. ``(num_of_images, x, y, channels)``.

       Examples
       --------
       ::

           # EXAMPLE 1
           # First divide the data in crops of (512, 512) and merge after that
           X_train = np.ones((165, 768, 1024, 1))
           Y_train = np.ones((165, 768, 1024, 1))

           X_test, Y_test = crop_data_with_overlap(X_test, Y_test, 512, 4)

           X_test = merge_data_with_overlap(X_test, (165, 768, 1024, 1), 512, 4,
                                            out_dir='out_dir')

       As example of different overlap maps are presented below. Example 1 above
       should store an image similar to the one on the top-right, as n_crops=4.

       +-------------------------------------+-------------------------------------------+
       | .. figure:: img/FIBSEM_test_0.png   | .. figure:: img/merged_ov_map_4.png       |
       |   :width: 80%                       |   :width: 70%                             |
       |   :align: center                    |   :align: center                          |
       |                                     |                                           |
       |   Original image                    |   Overlap map when n_crops=4              |
       +-------------------------------------+-------------------------------------------+
       | .. figure:: img/merged_ov_map_8.png | .. figure:: img/merged_ov_map_12.png      |
       |   :width: 80%                       |   :width: 70%                             |
       |   :align: center                    |   :align: center                          |
       |                                     |                                           |
       |   Overlap map when n_crops=8        |   Overlap map when n_crops=12             |
       +-------------------------------------+-------------------------------------------+
    """

    print("### MERGE-OV-CROP ###")
    print("Merging {} images into ({},{}) with overlapping . . ."
          .format(data.shape[1:], original_shape[2], original_shape[1]))

    # Merged data
    total_images = int(data.shape[0]/n_crops)
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
    for i in range(1, int(n_crops/2)+1, 1):
        if n_crops % i == 0 and abs((n_crops/i) - i) < min_d:
            min_d = abs((n_crops/i) - i)
            rows = i
            columns = int(n_crops/i)

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
        os.makedirs(out_dir, exist_ok=True)

        ov_map_matrix[ np.where(ov_map_matrix >= 8) ] = -1
        ov_map_matrix[ np.where(ov_map_matrix >= 3) ] = -2
        ov_map_matrix[ np.where(ov_map_matrix >= 2) ] = -3

        im = Image.fromarray(merged_data[ov_data_img,:,:,0])
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
    """Combine images from input data into a bigger one given shape. 

       The opposite function of :func:`~crop_data`.

       Parameters
       ----------                                                                    
       data : 4D Numpy array
           Data to crop. E.g. ``(num_of_images, x, y, channels)``.

       num : int, optional
           Number of examples to convert.

       out_shape : 2D int tuple, optional
           Number of horizontal and vertical images to combine in a single one.

       grid : bool, optional
           Make the grid in the output image.

       Returns
       -------                                                                 
       mixed_data : 4D Numpy array
           Mixed data images. E.g. ``(num_of_images, x, y, channels)``.

       mixed_data_mask : 4D Numpy array
           Mixed data masks. E.g. ``(num_of_images, x, y, channels)``.

       Examples
       --------
       ::

           # EXAMPLE 1
           # As the first example introduced in crop_data function, the merge after 
           # the crop should be done as follows:
           X_train = np.ones((165, 768, 1024, 1)) 
           Y_train = np.ones((165, 768, 1024, 1)) 
           
           X_train, Y_train, f_shape = crop_data(X_train, (256, 256, 1), data_mask=Y_train)

           X_train, Y_train = merge_data_without_overlap(
               X_train, (256, 256, 1), f_shape, data_mask=Y_train)
        
           # The function will print the shape of the created array. In this example:
           #     **** New data shape is: (1980, 256, 256, 1)

           # f_shape could be calculate as a division between the original data 
           # and the crop shapes. For instance:
           h_num = math.ceil(768/256)                   
           v_num = math.ceil(1024/265)
           f_shape = (h_num, v_num) # (3, 4)
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


def merge_3D_data_with_overlap(data, orig_vol_shape, data_mask=None,
                               overlap=(0,0,0)):
    """Merge 3D subvolumes in a 3D volume with a defined overlap.

       The opposite function is :func:`~crop_3D_data_with_overlap`.

       Parameters
       ----------
       data : 5D Numpy array
           Data to crop. E.g. ``(volume_number, x, y, z, channels)``.

       orig_vol_shape : 4D int tuple
           Shape of the volumes to created.

       data_mask : 4D Numpy array, optional
           Data mask to crop. E.g. ``(volume_number, x, y, z, channels)``.

       overlap : Tuple of 3 floats, optional                                    
            Amount of minimum overlap on x, y and z dimensions. Should be the 
            same as used in :func:`~crop_3D_data_with_overlap`. The values must 
            be on range ``[0, 1)``, that is, ``0%`` or ``99%`` of overlap.      
            E. g. ``(x, y, z)``. 

       Returns
       -------
       merged_data : 4D Numpy array
           Cropped image data. E.g. ``(num_of_images, x, y, channels)``.

       merged_data_mask : 5D Numpy array, optional
           Cropped image data masks. E.g. ``(num_of_images, x, y, channels)``.

       Examples                                                                 
       --------                                                                 
       ::                                                                       
                                                                                
           # EXAMPLE 1                                                          
           # Following the example introduced in crop_3D_data_with_overlap function, the 
           # merge after the cropping should be done as follows:
                                                                                
           X_train = np.ones((165, 768, 1024, 1))                               
           Y_train = np.ones((165, 768, 1024, 1))                               
                                                                                
           X_train, Y_train = crop_3D_data_with_overlap(                        
                X_train, (80, 80, 80, 1), data_mask=Y_train, overlap=(0.5,0.5,0.5))

           X_train, Y_train = merge_3D_data_with_overlap(
                X_train, (80, 80, 80, 1), data_mask=Y_train, overlap=(0.5,0.5,0.5))
                                                                                
           # The function will print the shape of the generated arrays. In this example:
           #     **** New data shape is: (165, 768, 1024, 1)
                                                                                
           # EXAMPLE 2                                                          
           # In the same way, if no overlap in cropping was selected, the merge call
           # should be as follows:
                                                                                
           X_train, Y_train = merge_3D_data_with_overlap(
                X_train, (80, 80, 80, 1), data_mask=Y_train, overlap=(0,0,0))   
                                                                                
           # The function will print the shape of the generated arrays. In this example:  
           #     **** New data shape is: (165, 768, 1024, 1)
    """ 
 
    print("### MERGE-3D-OV-CROP ###")
    if (overlap[0] >= 1 or overlap[0] < 0)\
       and (overlap[1] >= 1 or overlap[1] < 0)\
       and (overlap[2] >= 1 or overlap[2] < 0):                                 
        raise ValueError("'overlap' values must be floats between range [0, 1)")

    merged_data = np.zeros((orig_vol_shape))
    if data_mask is not None:
        merged_data_mask = np.zeros((orig_vol_shape))
    ov_map_counter = np.zeros((orig_vol_shape))

    # Calculate overlapping variables                                           
    overlap_x = 1 if overlap[0] == 0 else 1-overlap[0]                          
    overlap_y = 1 if overlap[1] == 0 else 1-overlap[1]                          
    overlap_z = 1 if overlap[2] == 0 else 1-overlap[2]  

    vols_per_x = math.ceil(orig_vol_shape[1]/(data.shape[1]*overlap_x))
    excess_x = int((vols_per_x*data.shape[1]*overlap_x)-orig_vol_shape[1])
    step_x = int(data.shape[1]*overlap_x)-int(excess_x/(vols_per_x-1))
    last_x = excess_x%(vols_per_x-1)

    vols_per_y = math.ceil(orig_vol_shape[2]/(data.shape[2]*overlap_y))
    excess_y = int((vols_per_y*data.shape[2]*overlap_y)-orig_vol_shape[2])
    step_y = int(data.shape[2]*overlap_y)-int(excess_y/(vols_per_y-1))
    last_y = excess_y%(vols_per_y-1)

    vols_per_z = math.ceil(orig_vol_shape[0]/(data.shape[3]*overlap_z))
    excess_z = int((vols_per_z*data.shape[3]*overlap_z)-orig_vol_shape[0])
    step_z = int(data.shape[3]*overlap_z)-int(excess_z/(vols_per_z-1))
    last_z = excess_z%(vols_per_z-1)

    vols_per_x += 1 if overlap_x == 1 else 0
    vols_per_y += 1 if overlap_y == 1 else 0
    vols_per_z += 1 if overlap_z == 1 else 0
                                                                                
    last_x += int(excess_x/(vols_per_x-1)) if overlap_x != 1 else 0
    last_y += int(excess_y/(vols_per_y-1)) if overlap_y != 1 else 0
    last_z += int(excess_z/(vols_per_z-1)) if overlap_z != 1 else 0

    # Real overlap calculation for printing                                     
    real_ov_x = (data.shape[1]-step_x)/data.shape[1]                              
    real_ov_y = (data.shape[2]-step_y)/data.shape[2]                              
    real_ov_z = (data.shape[3]-step_z)/data.shape[3]                              
    print("Real overlapping: {}".format((real_ov_x,real_ov_y,real_ov_z))) 

    c = 0
    for z in range(vols_per_z-1):
        for x in range(vols_per_x-1):
            for y in range(vols_per_y-1):     
                d_x = 0 if (x*step_x+data.shape[1]) < orig_vol_shape[1] else last_x
                d_y = 0 if (y*step_y+data.shape[2]) < orig_vol_shape[2] else last_y
                d_z = 0 if (z*step_z+data.shape[3]) < orig_vol_shape[0] else last_z

                merged_data[z*step_z-d_z:(z*step_z)+data.shape[3]-d_z, 
                            x*step_x-d_x:x*step_x+data.shape[1]-d_x, 
                            y*step_y-d_y:y*step_y+data.shape[2]-d_y] += data[c].transpose(2,0,1,3)
   
                if data_mask is not None: 
                    merged_data_mask[z*step_z-d_z:(z*step_z)+data.shape[3]-d_z,
                            x*step_x-d_x:x*step_x+data.shape[1]-d_x,
                            y*step_y-d_y:y*step_y+data.shape[2]-d_y] += \
                                          data_mask[c].transpose(2,0,1,3)

                ov_map_counter[z*step_z-d_z:(z*step_z)+data.shape[3]-d_z,
                            x*step_x-d_x:x*step_x+data.shape[1]-d_x,
                            y*step_y-d_y:y*step_y+data.shape[2]-d_y] += 1
                c += 1
                    
    merged_data = np.true_divide(merged_data, ov_map_counter)

    print("**** New data shape is: {}".format(merged_data.shape))
    print("### END MERGE-3D-OV-CROP ###")        

    if data_mask is not None: 
        merged_data_mask = np.true_divide(merged_data_mask, ov_map_counter)
        return merged_data, merged_data_mask
    else:
        return merged_data


def check_crops(data, out_dim, num_examples=2, include_crops=True,
                out_dir="check_crops", suffix="_none_", grid=True):
    """Check cropped images by the function :func:`~crop_data`.
        
       Parameters
       ----------
       data : 4D Numpy array
           Data to crop. E.g. ``(num_of_images, x, y, channels)``.
    
       out_dim : int 2D tuple
           Width and height of the image to be constructed. E. g. ``(y, x)``.

       num_examples : int, optional
           Number of examples to create.

       include_crops : bool, optional
           To save cropped images or only the image to contruct.  

       out_dir : str, optional
           Directory where the images will be save.

       suffix : str, optional
           Suffix to add in image names. 

       grid : bool, optional
           Make the grid in the output image.

       Examples
       --------
       ::

           # EXAMPLE 1
           # Check crops made in the first example of 'crop_data' function
           X_train = np.ones((165, 768, 1024))
           Y_train = np.ones((165, 768, 1024))

           X_train, Y_train, _ = crop_data(
               X_train, (256, 256, 1), data_mask=Y_train)

           # Notice that the original shape in 'out_dim' is provided, that is, (1024, 768)
           # instead of (768, 1024)
           check_crops(X_train, (1024, 768), num_examples=1, out_dir='out', grid=True)

           # 3 images samples will be stored in 'out' directory 

       The example above will store 12 individual crops (4x3, height x width), and 
       two images of the original shape: data image and its mask. For instance:

       +-----------------------------------------------+----------------------------------------------+
       | .. figure:: img/check_crop_data.png           | .. figure:: img/check_crop_mask.png          |
       |   :width: 80%                                 |   :width: 80%                                |
       |   :align: center                              |   :align: center                             |
       |                                               |                                              |
       | Original image (the grid should be each crop) | Original mask (the grid should be each crop) |
       +-----------------------------------------------+----------------------------------------------+
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

       Parameters
       ----------
       path : str
           Path to the data mask.
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


def crop_3D_data(data, vol_shape, data_mask=None, use_rest=False):
    """Crop 3D data into smaller volumes without overlap. If there is no exact
       division between the data shape and ``vol_shape`` in a specific dimension
       it will be discarded or zeros will be added if ``use_rest`` is True.

       Parameters
       ----------
       data : Numpy 4D array
           Data. E.g. ``(num_of_images, x, y, channels)``.      
                                                                           
       vol_shape : 4D int tuple
           Shape of the volumes to created. E.g. ``(x, y, z, channels)``.

       data_mask : Numpy 4D array, optional
           Mask data. E.g. ``(num_of_images, x, y, channels)``.
        
       use_rest : bool, optional
           Controls how the rest data will be processed. When there is no exact
           division between the data shape and ``vol_shape`` in a specific
           dimension, the remainder data is not enough to create another 
           subvolume. If True, that data will be used completing the rest of the 
           subvolume with zeros. If False, that remainder will be dropped (notice 
           that this option will make the data impossible to reconstruct 100% 
           later on).

       Returns
       -------
       cropped_data : Numpy 5D array
           data data separated in different subvolumes with the provided shape. E.g.
           ``(subvolume_number, ) + shape``.
            
       cropped_data_mask : Numpy 5D array
           data_mask data separated in different subvolumes with the provided shape. E.g. 
           ``(subvolume_number, ) + shape``.
    """
        
    print("### 3D-CROP ###")
    print("Cropping {} images into {} . . ."
          .format(data.shape, vol_shape))
                                                                    
    if len(vol_shape) != 4:
        raise ValueError("'vol_shape' must be 4D int tuple")
    if data.ndim != 4:
        raise ValueError("data must be a 4D Numpy array")
    if data_mask is not None:
        if data_mask.ndim != 4:                                              
            raise ValueError("data_mask must be a 4D Numpy array") 
    if vol_shape[0] > data.shape[0]:
        raise ValueError("'vol_shape[0]' {} greater than {}"
                         .format(vol_shape[0], data.shape[0]))
    if vol_shape[1] > data.shape[1]:
        raise ValueError("'vol_shape[1]' {} greater than {}"
                         .format(vol_shape[1], data.shape[1]))
    if vol_shape[2] > data.shape[2]:
        raise ValueError("'vol_shape[2]' {} greater than {}"
                         .format(vol_shape[2], data.shape[2]))

    # Calculate crops per axis                                      
    vols_per_x = math.ceil(data.shape[1]/vol_shape[0])
    vols_per_y = math.ceil(data.shape[2]/vol_shape[1])
    vols_per_z = math.ceil(data.shape[0]/vol_shape[2])
    
    _d_x = 1 if data.shape[1]%vol_shape[0] and not use_rest else 0
    _d_y = 1 if data.shape[2]%vol_shape[1] and not use_rest else 0
    _d_z = 1 if data.shape[0]%vol_shape[2] and not use_rest else 0
    
    num_sub_volum = (vols_per_x-_d_x)*(vols_per_y-_d_y)*(vols_per_z-_d_z)

    # Calculate the excess
    last_x = vols_per_x*vol_shape[0]-data.shape[1]
    last_y = vols_per_y*vol_shape[1]-data.shape[2]
    last_z = vols_per_z*vol_shape[2]-data.shape[0]
    print("Zeros added per dimension: ({},{},{})"
          .format(last_x,last_y,last_z))
                                                                                
    cropped_data = np.zeros((num_sub_volum, ) + vol_shape)
    if data_mask is not None:
        cropped_data_mask = np.zeros((num_sub_volum, ) + vol_shape)
                                                                                
    print("{},{},{} patches per x,y,z axis"
          .format((vols_per_x),(vols_per_y),(vols_per_z)))

    # Reshape the data to generate needed 3D subvolumes                        
    c = 0
    for z in range(vols_per_z):
        for x in range(vols_per_x):                                                  
            for y in range(vols_per_y):
                d_x = 0 if (((x+1)*vol_shape[0])) < data.shape[1] else last_x
                d_y = 0 if (((y+1)*vol_shape[1])) < data.shape[2] else last_y
                d_z = 0 if (((z+1)*vol_shape[2])) < data.shape[0] else last_z

                if d_x != 0 and not use_rest: break
                if d_y != 0 and not use_rest: break
                if d_z != 0 and not use_rest: break

                cropped_data[c,0:vol_shape[2]-d_z,
                               0:vol_shape[0]-d_x,
                               0:vol_shape[1]-d_y] = \
                    data[(z*vol_shape[2]):((z+1)*vol_shape[2])-d_z,
                         (x*vol_shape[0]):((x+1)*vol_shape[0])-d_x,
                         (y*vol_shape[1]):((y+1)*vol_shape[1])-d_y]
                if data_mask is not None:
                    cropped_data_mask[c,0:vol_shape[2]-d_z,                              
                                        0:vol_shape[0]-d_x,                              
                                        0:vol_shape[1]-d_y] = \
                        data_mask[(z*vol_shape[2]):((z+1)*vol_shape[2])-d_z,                 
                                  (x*vol_shape[0]):((x+1)*vol_shape[0])-d_x,                 
                                  (y*vol_shape[1]):((y+1)*vol_shape[1])-d_y]
                c += 1                                                                

    cropped_data = cropped_data.transpose(0,2,3,1,4)
    print("**** New data shape is: {}".format(cropped_data.shape))
    print("### END 3D-CROP ###")

    if data_mask is not None:
        cropped_data_mask = cropped_data_mask.transpose(0,2,3,1,4)
        return cropped_data, cropped_data_mask
    else:
        return cropped_data


def img_to_onehot_encoding(img, num_classes=2):
    """Converts image given into one-hot encode format.

       Parameters
       ----------
       img : Numpy 4D array
           Image. E.g. ``(x, y, z, channels)``.
            
       num_classes : int, optional
           Number of classes to distinguish.
       
       Returns
       ------- 
       one_hot_labels : Numpy 4D array
           Data one-hot encoded. E.g. ``(x, y, z, num_classes)``.

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


def random_crop(image, mask, random_crop_size, val=False,                       
                draw_prob_map_points=False, img_prob=None, weight_map=None):                                               
    """Random crop.                                                             
        
       Parameters
       ----------
       image : Numpy 3D array                                                     
           Image. E.g. ``(x, y, channels)``.

       mask : Numpy 3D array                                                     
           Image mask. E.g. ``(x, y, channels)``.                                    
                
       random_crop_size : 2 int tuple
           Size of the crop. E.g. ``(height, width)``.
    
       val : bool, optional
           If the image provided is going to be used in the validation data. 
           This forces to crop from the origin, e. g. ``(0, 0)`` point.
    
       draw_prob_map_points : bool
           To return the pixel choosen to be the center of the crop. 

       img_prob : Numpy 3D array, optional
           Probability of each pixel to be choosen as the center of the crop. 
           E. .g. ``(x, y, channels)``.

       weight_map : bool, optional
           Weight map of the given image. E.g. ``(x, y, channels)``.

       Returns
       -------
       img : 2D Numpy array
           Crop of the given image. E. g. ``(height, width)``.

       weight_map : 2D Numpy array, optional
           Crop of the given image's weigth map. E. g. ``(height, width)``.
       
       ox : int, optional
           X coordinate in the complete image of the choosed central pixel to 
           make the crop.
       oy : int, optional
           Y coordinate in the complete image of the choosed central pixel to    
           make the crop.
       x : int, optional
           X coordinate in the complete image where the crop starts. 
       y : int, optional
           Y coordinate in the complete image where the crop starts.
    """                                                                         
                                                                                
    if weight_map is not None:
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
            ox = 0                                                              
            oy = 0                                                              
            x = np.random.randint(0, width - dx + 1)                            
            y = np.random.randint(0, height - dy + 1)                           
                                                                                
    if draw_prob_map_points == True:                                            
        return img[y:(y+dy), x:(x+dx), :], mask[y:(y+dy), x:(x+dx), :], ox, oy, x, y
    else:                                                                       
        if weight_map is not None:
            return img[y:(y+dy), x:(x+dx), :], mask[y:(y+dy), x:(x+dx), :],\
                   weight_map[y:(y+dy), x:(x+dx), :]                            
        else:                                                                   
            return img[y:(y+dy), x:(x+dx), :], mask[y:(y+dy), x:(x+dx), :]      
                                                                                

