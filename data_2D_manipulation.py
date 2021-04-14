import numpy as np
import os
import sys
import math
import random
from tqdm import tqdm
from skimage.io import imread
from sklearn.model_selection import train_test_split
from PIL import Image
from util import load_data_from_dir, foreground_percentage


def load_and_prepare_2D_data(train_path, train_mask_path, test_path, test_mask_path, 
    image_train_shape, image_test_shape, create_val=True, val_split=0.1,
    shuffle_val=True, seedValue=42, e_d_data=[], e_d_mask=[], e_d_data_dim=[],
    num_crops_per_dataset=0, make_crops=True, crop_shape=None, ov=(0, 0), 
    overlap_train=False, check_crop=True, check_crop_path="check_crop"):
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

       num_crops_per_dataset : int, optional
           Number of crops per extra dataset to take into account. Useful to 
           ensure that all the datasets have the same weight during network 
           trainning. 

       make_crops : bool, optional
           To make crops on data.

       crop_shape : 3D int tuple, optional
           Shape of the crops. E.g. ``(x, y, channels)``.

       ov : 2 floats tuple, optional                                         
           Amount of minimum overlap on x and y dimensions. The values must be on
           range ``[0, 1)``, that is, ``0%`` or ``99%`` of overlap. E. g. ``(x, y)``.   

       overlap_train : bool, optional
           If ``True`` ``ov`` will be used to crop training data. ``False`` to 
           force minimum overap instead: ``ov=(0,0)``. 

       check_crop : bool, optional
           To save the crops made to ensure they are generating as one wish.

       check_crop_path : str, optional
           Path to save the crop samples.

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

           X_train, Y_train, X_val,                                             
           Y_val, X_test, Y_test,                                               
           orig_test_shape, norm_value, crops_made = load_and_prepare_2D_data(  
               train_path, train_mask_path, test_path, test_mask_path, img_train_shape, 
               mg_test_shape, val_split=0.1, shuffle_val=True, make_crops=True,
               crop_shape=(256, 256, 1), check_crop=True, check_crop_path="check_folder"
               e_d_data=extra_datasets_data_list, e_d_mask=extra_datasets_mask_list, 
               e_d_data_dim=extra_datasets_data_dim_list)
                
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

    # Create validation data splitting the train
    if create_val:
        X_train, X_val, \
        Y_train, Y_val = train_test_split(
            X_train, Y_train, test_size=val_split, shuffle=shuffle_val,
            random_state=seedValue)

    # Save original data shape 
    orig_train_shape = Y_train.shape
    orig_test_shape = Y_test.shape

    # Crop the data
    if make_crops:
        print("4) Crop data activated . . .")
        print("4.1) Cropping train data . . .")
        t_ov = ov if overlap_train else (0,0)
        X_train, Y_train = crop_data_with_overlap(
            X_train, crop_shape, data_mask=Y_train, overlap=t_ov)

        print("4.2) Cropping test data . . .")
        X_test, Y_test = crop_data_with_overlap(
            X_test, crop_shape, data_mask=Y_test, overlap=ov)
        
        if create_val:
            print("4.3) Cropping validation data . . .")
            X_val, Y_val = crop_data_with_overlap(
                X_val, crop_shape, data_mask=Y_val, overlap=(0,0))

        if check_crop:
            print("4.4) Checking the crops . . .")
            check_crops(X_train, orig_train_shape, t_ov, num_examples=3,
                        out_dir=check_crop_path, prefix="X_train_")

            check_crops(Y_train, orig_train_shape, t_ov, num_examples=3,
                        out_dir=check_crop_path, prefix="Y_train_")
        
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
                    e_X_train, crop_shape, data_mask=e_Y_train)
                    
                if num_crops_per_dataset != 0:
                    e_X_train = e_X_train[:num_crops_per_dataset]
                    e_Y_train = e_Y_train[:num_crops_per_dataset]

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


def crop_data_with_overlap(data, crop_shape, data_mask=None, overlap=(0,0),
                           padding=(0,0), verbose=True):
    """Crop data into small square pieces with overlap. The difference with
       :func:`~crop_data` is that this function allows you to create patches with 
       overlap. 

       The opposite function is :func:`~merge_data_with_overlap`.

       Parameters
       ----------
       data : 4D Numpy array
           Data to crop. E.g. ``(num_of_images, x, y, channels)``.
        
       crop_shape : 3 int tuple
           Shape of the crops to create. E.g. ``(x, y, channels)``.
        
       data_mask : 4D Numpy array, optional
           Data mask to crop. E.g. ``(num_of_images, x, y, channels)``.
    
       overlap : Tuple of 2 floats, optional
           Amount of minimum overlap on x and y dimensions. The values must be on
           range ``[0, 1)``, that is, ``0%`` or ``99%`` of overlap. E. g. ``(x, y)``.
        
       padding : tuple of ints, optional                                       
           Size of padding to be added on each axis ``(x, y)``. E.g. ``(24, 24)``.
             
       verbose : bool, optional
            To print information about the crop to be made.
            
       Returns
       -------
       cropped_data : 4D Numpy array
           Cropped image data. E.g. ``(num_of_images, x, y, channels)``.

       cropped_data_mask : 4D Numpy array, optional
           Cropped image data masks. E.g. ``(num_of_images, x, y, channels)``.

       Examples
       --------
       ::

           # EXAMPLE 1
           # Divide in crops of (256, 256) a given data with the minimum overlap 
           X_train = np.ones((165, 768, 1024, 1))
           Y_train = np.ones((165, 768, 1024, 1))

           X_train, Y_train = crop_data_with_overlap(X_train, (256, 256, 1), Y_train, (0, 0))

           # Notice that as the shape of the data has exact division with the wnanted 
           # crops shape so no overlap will be made. The function will print the following 
           # information:
           #     Minimum overlap selected: (0, 0)
           #     Real overlapping (%): (0.0, 0.0)
           #     Real overlapping (pixels): (0.0, 0.0)
           #     (3, 4) patches per (x,y) axis 
           #     **** New data shape is: (1980, 256, 256, 1)


           # EXAMPLE 2
           # Same as example 1 but with 25% of overlap between crops
           X_train, Y_train = crop_data_with_overlap(X_train, (256, 256, 1), Y_train, (0.25, 0.25))

           # The function will print the following information:
           #     Minimum overlap selected: (0.25, 0.25)
           #     Real overlapping (%): (0.33203125, 0.3984375)
           #     Real overlapping (pixels): (85.0, 102.0)
           #     (4, 6) patches per (x,y) axis
           #     **** New data shape is: (3960, 256, 256, 1)


           # EXAMPLE 3
           # Same as example 1 but with 50% of overlap between crops
           X_train, Y_train = crop_data_with_overlap(X_train, (256, 256, 1), Y_train, (0.5, 0.5))

           # The function will print the shape of the created array. In this example:
           #     Minimum overlap selected: (0.5, 0.5)
           #     Real overlapping (%): (0.59765625, 0.5703125)
           #     Real overlapping (pixels): (153.0, 146.0)
           #     (6, 8) patches per (x,y) axis
           #     **** New data shape is: (7920, 256, 256, 1)

        
           # EXAMPLE 4
           # Same as example 2 but with 50% of overlap only in x axis
           X_train, Y_train = crop_data_with_overlap(X_train, (256, 256, 1), Y_train, (0.5, 0))

           # The function will print the shape of the created array. In this example:
           #     Minimum overlap selected: (0.5, 0)
           #     Real overlapping (%): (0.59765625, 0.0)
           #     Real overlapping (pixels): (153.0, 0.0)
           #     (6, 4) patches per (x,y) axis
           #     **** New data shape is: (3960, 256, 256, 1)
    """

    if verbose:
        print("### OV-CROP ###")
        print("Cropping {} images into {} with overlapping. . ."\
              .format(data.shape, crop_shape))
        print("Minimum overlap selected: {}".format(overlap))
    
    if (overlap[0] >= 1 or overlap[0] < 0) and (overlap[1] >= 1 or overlap[1] < 0):
        raise ValueError("'overlap' values must be floats between range [0, 1)")

    padded_data = np.zeros((data.shape[0],
                            data.shape[1]+2*padding[0],
                            data.shape[2]+2*padding[1], 
                            data.shape[3]))
    padded_data[:, padding[0]:padding[0]+data.shape[1],
                padding[1]:padding[1]+data.shape[2], :] = data                                                

    if data_mask is not None:    
        padded_data_mask = np.zeros((data_mask.shape[0],
                                     data_mask.shape[1]+2*padding[0],
                                     data_mask.shape[2]+2*padding[1],
                                     data_mask.shape[3]))
        padded_data_mask[:, padding[0]:padding[0]+data_mask.shape[1],
                         padding[1]:padding[1]+data_mask.shape[2], :] = data_mask                                                

    padded_crop_shape = crop_shape
    crop_shape = (crop_shape[0]-2*padding[0], crop_shape[1]-2*padding[1], crop_shape[2])
        
    # Calculate overlapping variables
    overlap_x = 1 if overlap[0] == 0 else 1-overlap[0]                       
    overlap_y = 1 if overlap[1] == 0 else 1-overlap[1]                       

    # X
    crops_per_x = math.ceil(data.shape[1]/(crop_shape[0]*overlap_x))
    excess_x = int((crops_per_x*crop_shape[0])-((crops_per_x-1)*overlap[0]*crop_shape[0]))-data.shape[1]
    ex = 0 if crops_per_x == 1 else int(excess_x/(crops_per_x-1))
    step_x = int(crop_shape[0]*overlap_x)-ex
    last_x = 0 if crops_per_x == 1 else excess_x%(crops_per_x-1)

    # Y
    crops_per_y = math.ceil(data.shape[2]/(crop_shape[1]*overlap_y))
    excess_y = int((crops_per_y*crop_shape[1])-((crops_per_y-1)*overlap[1]*crop_shape[1]))-data.shape[2]
    ex = 0 if crops_per_y == 1 else int(excess_y/(crops_per_y-1))
    step_y = int(crop_shape[1]*overlap_y)-ex
    last_y = 0 if crops_per_y == 1 else excess_y%(crops_per_y-1)

    # Real overlap calculation for printing 
    real_ov_x = (crop_shape[0]-step_x)/crop_shape[0]
    real_ov_y = (crop_shape[1]-step_y)/crop_shape[1]
    if verbose:
        print("Real overlapping (%): {}".format((real_ov_x,real_ov_y)))
        print("Real overlapping (pixels): {}".format((crop_shape[0]*real_ov_x,
              crop_shape[1]*real_ov_y)))
        print("{} patches per (x,y) axis".format((crops_per_x,crops_per_y)))

    total_vol = data.shape[0]*(crops_per_x)*(crops_per_y)
    cropped_data = np.zeros((total_vol,) + padded_crop_shape, dtype=data.dtype)
    if data_mask is not None:
        cropped_data_mask = np.zeros((total_vol,)+padded_crop_shape[:2]+(data_mask.shape[-1],), dtype=data_mask.dtype)

    c = 0
    for z in range(data.shape[0]):
        for x in range(crops_per_x):
            for y in range(crops_per_y):
                d_x = 0 if (x*step_x+crop_shape[0]) < data.shape[1] else last_x
                d_y = 0 if (y*step_y+crop_shape[1]) < data.shape[2] else last_y

                cropped_data[c] = \
                    padded_data[z,
                                x*step_x-d_x:x*step_x+crop_shape[0]-d_x+2*padding[0],
                                y*step_y-d_y:y*step_y+crop_shape[1]-d_y+2*padding[1]]

                if data_mask is not None:
                    cropped_data_mask[c] = \
                        padded_data_mask[z,
                                         x*step_x-d_x:x*step_x+crop_shape[0]-d_x+2*padding[0],   
                                         y*step_y-d_y:y*step_y+crop_shape[1]-d_y+2*padding[1]]
                c += 1

    if verbose:
        print("**** New data shape is: {}".format(cropped_data.shape))
        print("### END OV-CROP ###")

    if data_mask is not None:
        return cropped_data, cropped_data_mask
    else:
        return cropped_data


def merge_data_with_overlap(data, original_shape, data_mask=None, overlap=(0,0),
                            padding=(0,0), verbose=True, out_dir=None, prefix=""):
    """Merge data with an amount of overlap.
    
       The opposite function is :func:`~crop_data_with_overlap`.

       Parameters
       ----------
       data : 4D Numpy array
           Data to merge. E.g. ``(num_of_images, x, y, channels)``.

       original_shape : 4D int tuple
           Shape of the original data. E.g. ``(num_of_images, x, y, channels)``

       data_mask : 4D Numpy array, optional
           Data mask to merge. E.g. ``(num_of_images, x, y, channels)``.

       overlap : Tuple of 2 floats, optional                                    
           Amount of minimum overlap on x, y and z dimensions. Should be the same
           as used in :func:`~crop_data_with_overlap`. The values must be on range
           ``[0, 1)``, that is, ``0%`` or ``99%`` of overlap. E. g. ``(x, y)``. 
                                                                                       
       padding : tuple of ints, optional                                        
           Size of padding to be added on each axis ``(x, y)``. E.g. ``(24, 24)``.
                                                                                
       verbose : bool, optional                                                 
            To print information about the crop to be made.                     
           
       out_dir : str, optional
           If provided an image that represents the overlap made will be saved. 
           The image will be colored as follows: green region when ``==2`` crops 
           overlap, yellow when ``2 < x < 6`` and red when ``=<6`` or more crops
           are merged.

       prefix : str, optional
           Prefix to save overlap map with. 

       Returns
       -------
       merged_data : 4D Numpy array
           Merged image data. E.g. ``(num_of_images, x, y, channels)``.
        
       merged_data_mask : 4D Numpy array, optional
           Merged image data mask. E.g. ``(num_of_images, x, y, channels)``.

       Examples
       --------
       ::

           # EXAMPLE 1
           # Merge the data of example 1 of 'crop_data_with_overlap' function

           # 1) CROP
           X_train = np.ones((165, 768, 1024, 1))
           Y_train = np.ones((165, 768, 1024, 1))
           X_train, Y_train = crop_data_with_overlap(X_train, (256, 256, 1), Y_train, (0, 0))

           # 2) MERGE
           X_train, Y_train = merge_data_with_overlap(
               X_train, (165, 768, 1024, 1), Y_train, (0, 0), out_dir='out_dir')

           # The function will print the following information:
           #     Minimum overlap selected: (0, 0)
           #     Real overlapping (%): (0.0, 0.0)
           #     Real overlapping (pixels): (0.0, 0.0)
           #     (3, 4) patches per (x,y) axis
           #     **** New data shape is: (165, 768, 1024, 1)


           # EXAMPLE 2
           # Merge the data of example 2 of 'crop_data_with_overlap' function
           X_train, Y_train = merge_data_with_overlap(
               X_train, (165, 768, 1024, 1), Y_train, (0.25, 0.25), out_dir='out_dir')

           # The function will print the following information:
           #     Minimum overlap selected: (0.25, 0.25)
           #     Real overlapping (%): (0.33203125, 0.3984375)
           #     Real overlapping (pixels): (85.0, 102.0)
           #     (3, 5) patches per (x,y) axis
           #     **** New data shape is: (165, 768, 1024, 1)


           # EXAMPLE 3
           # Merge the data of example 3 of 'crop_data_with_overlap' function
           X_train, Y_train = merge_data_with_overlap(
               X_train, (165, 768, 1024, 1), Y_train, (0.5, 0.5), out_dir='out_dir')

           # The function will print the shape of the created array. In this example:
           #     Minimum overlap selected: (0.5, 0.5)
           #     Real overlapping (%): (0.59765625, 0.5703125)
           #     Real overlapping (pixels): (153.0, 146.0)
           #     (6, 8) patches per (x,y) axis
           #     **** New data shape is: (165, 768, 1024, 1)
          
            
           # EXAMPLE 4
           # Merge the data of example 1 of 'crop_data_with_overlap' function
           X_train, Y_train = merge_data_with_overlap(
               X_train, (165, 768, 1024, 1), Y_train, (0.5, 0), out_dir='out_dir')

           # The function will print the shape of the created array. In this example:
           #     Minimum overlap selected: (0.5, 0)
           #     Real overlapping (%): (0.59765625, 0.0)
           #     Real overlapping (pixels): (153.0, 0.0)
           #     (6, 4) patches per (x,y) axis
           #     **** New data shape is: (165, 768, 1024, 1)


       As example of different overlap maps are presented below. 

       +---------------------------------------+-------------------------------------------+
       | .. figure:: img/merged_ov_map_0.png   | .. figure:: img/merged_ov_map_0.25.png    |
       |   :width: 80%                         |   :width: 70%                             |
       |   :align: center                      |   :align: center                          |
       |                                       |                                           |
       |   Example 1 overlapping map           |   Example 2 overlapping map               |
       +---------------------------------------+-------------------------------------------+
       | .. figure:: img/merged_ov_map_0.5.png | .. figure:: img/merged_ov_map_0.5inx.png  |
       |   :width: 80%                         |   :width: 70%                             |
       |   :align: center                      |   :align: center                          |
       |                                       |                                           |
       |   Example 3 overlapping map           |   Example 4 overlapping map               |
       +---------------------------------------+-------------------------------------------+
    """

    if verbose:
        print("### MERGE-OV-CROP ###")
        print("Merging {} images into {} with overlapping . . ."
              .format(data.shape, original_shape))
        print("Minimum overlap selected: {}".format(overlap))
    
    if (overlap[0] >= 1 or overlap[0] < 0) and (overlap[1] >= 1 or overlap[1] < 0):
        raise ValueError("'overlap' values must be floats between range [0, 1)")

    # Remove the padding
    data = data[:, padding[0]:data.shape[1]-padding[0],
                padding[1]:data.shape[2]-padding[1], :]

    merged_data = np.zeros((original_shape), dtype=data.dtype)
    if data_mask is not None:
        merged_data_mask = np.zeros((original_shape), dtype=data_mask.dtype)
        data_mask = data_mask[:, padding[0]:data_mask.shape[1]-padding[0],
                              padding[1]:data_mask.shape[2]-padding[1], :]

    ov_map_counter = np.zeros(original_shape, dtype=np.int32)
    if out_dir is not None:
        crop_grid = np.zeros(original_shape[1:], dtype=np.int32)

    # Calculate overlapping variables                                           
    overlap_x = 1 if overlap[0] == 0 else 1-overlap[0]                          
    overlap_y = 1 if overlap[1] == 0 else 1-overlap[1]                          

    # X
    crops_per_x = math.ceil(original_shape[1]/(data.shape[1]*overlap_x))
    excess_x = int((crops_per_x*data.shape[1])-((crops_per_x-1)*overlap[0]*data.shape[1]))-original_shape[1]
    ex = 0 if crops_per_x == 1 else int(excess_x/(crops_per_x-1))
    step_x = int(data.shape[1]*overlap_x)-ex
    last_x = 0 if crops_per_x == 1 else excess_x%(crops_per_x-1)
    
    # Y
    crops_per_y = math.ceil(original_shape[2]/(data.shape[2]*overlap_y))
    excess_y = int((crops_per_y*data.shape[2])-((crops_per_y-1)*overlap[1]*data.shape[2]))-original_shape[2]
    ex = 0 if crops_per_y == 1 else int(excess_y/(crops_per_y-1))
    step_y = int(data.shape[2]*overlap_y)-ex
    last_y = 0 if crops_per_y == 1 else excess_y%(crops_per_y-1)

    # Real overlap calculation for printing                                     
    real_ov_x = (data.shape[1]-step_x)/data.shape[1]                              
    real_ov_y = (data.shape[2]-step_y)/data.shape[2]                              
    if verbose:
        print("Real overlapping (%): {}".format((real_ov_x,real_ov_y))) 
        print("Real overlapping (pixels): {}".format((data.shape[1]*real_ov_x,
              data.shape[2]*real_ov_y)))
        print("{} patches per (x,y) axis".format((crops_per_x,crops_per_y)))

    c = 0
    for z in range(original_shape[0]):
        for x in range(crops_per_x):
            for y in range(crops_per_y):     
                d_x = 0 if (x*step_x+data.shape[1]) < original_shape[1] else last_x
                d_y = 0 if (y*step_y+data.shape[2]) < original_shape[2] else last_y

                merged_data[z,
                    x*step_x-d_x:x*step_x+data.shape[1]-d_x, 
                    y*step_y-d_y:y*step_y+data.shape[2]-d_y] += data[c]
   
                if data_mask is not None: 
                    merged_data_mask[z,
                        x*step_x-d_x:x*step_x+data.shape[1]-d_x,
                        y*step_y-d_y:y*step_y+data.shape[2]-d_y] += data_mask[c]

                ov_map_counter[z,
                    x*step_x-d_x:x*step_x+data.shape[1]-d_x,
                    y*step_y-d_y:y*step_y+data.shape[2]-d_y] += 1
                
                if z == 0 and out_dir is not None:
                    crop_grid[x*step_x-d_x,
                              y*step_y-d_y:y*step_y+data.shape[2]-d_y] = 1
                    crop_grid[x*step_x+data.shape[1]-d_x-1,
                              y*step_y-d_y:y*step_y+data.shape[2]-d_y] = 1
                    crop_grid[x*step_x-d_x:x*step_x+data.shape[1]-d_x,
                              y*step_y-d_y] = 1
                    crop_grid[x*step_x-d_x:x*step_x+data.shape[1]-d_x,
                              y*step_y+data.shape[2]-d_y-1] = 1

                c += 1
                    
    merged_data = np.true_divide(merged_data, ov_map_counter)
    if data_mask is not None:
        merged_data_mask = np.true_divide(merged_data_mask, ov_map_counter)

    # Save a copy of the merged data with the overlapped regions colored as: 
    # green when 2 crops overlap, yellow when (2 < x < 6) and red when more than 
    # 6 overlaps are merged 
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

        ov_map = ov_map_counter[0]
        ov_map = ov_map.astype('int32') 

        ov_map[np.where(ov_map_counter[0] >= 2)] = -3
        ov_map[np.where(ov_map_counter[0] >= 3)] = -2
        ov_map[np.where(ov_map_counter[0] >= 6)] = -1
        ov_map[np.where(crop_grid == 1)] = -4

        # Paint overlap regions
        im = Image.fromarray(merged_data[0,...,0])
        im = im.convert('RGBA')
        px = im.load()
        width, height = im.size
        for im_i in range(width): 
            for im_j in range(height):
                # White borders
                if ov_map[im_j, im_i] == -4: 
                    px[im_i, im_j] = (255, 255, 255, 255)
                # Overlap zone
                elif ov_map[im_j, im_i] == -3: 
                    px[im_i, im_j] = tuple(map(sum, zip((0, 74, 0, 125), px[im_i, im_j])))
                # 2 < x < 6 overlaps
                elif ov_map[im_j, im_i] == -2:
                    px[im_i, im_j] = tuple(map(sum, zip((74, 74, 0, 125), px[im_i, im_j])))
                # 6 >= overlaps
                elif ov_map[im_j, im_i] == -1:
                    px[im_i, im_j] = tuple(map(sum, zip((74, 0, 0, 125), px[im_i, im_j])))

        im.save(os.path.join(out_dir, prefix + "merged_ov_map.png"))
  
    if verbose:
        print("**** New data shape is: {}".format(merged_data.shape))
        print("### END MERGE-OV-CROP ###")

    if data_mask is not None: 
        return merged_data, merged_data_mask
    else:
        return merged_data


def check_crops(data, original_shape, ov, num_examples=1, include_crops=True,
                out_dir="check_crops", prefix=""):
    """Check cropped images by the function :func:`~crop_data` and 
       :func:`~crop_data_with_overlap`. 
        
       Parameters
       ----------
       data : 4D Numpy array
           Data to crop. E.g. ``(num_of_images, x, y, channels)``.

       original_shape : Tuple of 4 ints
           Shape of the original data. E.g. ``(num_of_images, x, y, channels)``.

       ov : Tuple of 2 floats, optional                                    
           Amount of minimum overlap on x and y dimensions. The values must be on
           range ``[0, 1)``, that is, ``0%`` or ``99%`` of overlap. E. g. ``(x, y)``. 

       num_examples : int, optional
           Number of examples to create.

       include_crops : bool, optional
           To save cropped images or only the image to contruct.  

       out_dir : str, optional
           Directory where the images will be save.

       prefix : str, optional
           Prefix to save overlap map with.

       Examples
       --------
       ::

           # EXAMPLE 1
           # Check crops made in the first example of 'crop_data' function
           original_shape = (165, 768, 1024)
           X_train = np.ones(original_shape)
           Y_train = np.ones(original_shape)

           X_train, Y_train = crop_data_with_overlap(X_train, (256, 256, 1), Y_train, (0, 0))

           check_crops(X_train, original_shape, num_examples=1, out_dir='out')

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

    os.makedirs(out_dir, exist_ok=True)

    # Calculate horizontal and vertical image number for the data
    h_num = math.ceil(original_shape[1] / data.shape[1])
    v_num = math.ceil(original_shape[2] / data.shape[2])
    total = h_num*v_num*num_examples

    if total > data.shape[0]:
        num_examples = math.ceil(data.shape[0]/(h_num*v_num))
        total = num_examples
        print("Requested num_examples too high for data. Set automatically to {}"\
              .format(num_examples))

    if include_crops:
        print("0) Saving cropped data images . . .")
        for i in tqdm(range(total)):
            # Grayscale images
            if data.shape[3] == 1:
                im = Image.fromarray(data[i,...,0])
                im = im.convert('L')
            # RGB images
            else:
                aux = np.asarray(data[i], dtype="uint8")
                im = Image.fromarray(aux, 'RGB')

            im.save(os.path.join(out_dir, prefix + "c_" + str(i) + ".png"))

    merge_data_with_overlap(data, original_shape, overlap=ov, out_dir=out_dir,
                            prefix=prefix)

    print("### END CHECK-CROP ###")


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
                                                                                

def crop_data(data, crop_shape, data_mask=None, force_shape=(0, 0), 
              d_percentage=0):                          
    """Crop data into smaller pieces of ``crop_shape``. If there is no exact 
       division between the data shape and ``crop_shape`` in a specific dimension
       zeros will be added. 
        
       DEFERRED: use :func:`~crop_data_with_overlap` instead.

       The opposite function is :func:`~merge_data`.

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


def merge_data(data, num, out_shape=(1, 1), grid=False):
    """Combine images from input data into a bigger one given shape. 
       The opposite function of :func:`~crop_data`.
        
       DEFERRED: use :func:`~merge_data_with_overlap` instead.

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
           X_train, Y_train = merge_data(
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


