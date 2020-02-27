import numpy as np
import os
import sys
import math
import random
from tqdm import tqdm
from skimage.io import imread
from sklearn.model_selection import train_test_split
from PIL import Image
from util import Print 

def load_data(train_path, train_mask_path, test_path, test_mask_path, 
              image_train_shape, image_test_shape, create_val=True, 
              val_split=0.1, shuffle_val=True, seedValue=42, 
              job_id="none_job_id", e_d_data=[], e_d_mask=[], e_d_data_dim=[], 
              e_d_dis=[], num_crops_per_dataset=0, make_crops=True, 
              crop_shape=None, check_crop=True, d_percentage=0, 
              prepare_subvolumes=False, subvol_shape=None, tab=""):         

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
            job_id (str, optional): job identifier. If any provided the examples
            of the check_crop function will be generated under a folder 
            'check_crops/none_job_id'.
            e_d_data (list of str, optional): list of paths where the extra data
            of other datasets are stored.
            e_d_mask (list of str, optional): list of paths where the extra data
            mask of other datasets are stored.
            e_d_data_dim (list of int tuple, optional): list of shapes of the 
            extra datasets provided. 
            e_d_dis (list of float, optional): discard percentages of the extra
            datasets provided. Values between 0 and 1.
            num_crops_per_dataset (int, optional): number of crops per extra
            dataset to take into account. Useful to ensure that all the datasets
            have the same weight during network trainning. 
            make_crops (bool, optional): flag to make crops on data.
            crop_shape (tuple of int, optional): shape of the crops.
            check_crop (bool, optional): to save the crops made to ensure they
            are generating as one wish.
            d_percentage (int, optional): number between 0 and 100. The images
            that have less foreground pixels than the given number will be
            discarded.
            prepare_subvolumes (bool, optional): flag to prepare 3D subvolumes 
            (use this option only to train a 3D network). 
            subvol_shape (Tuple, optional): 
            tab (str, optional): tabulation mark to add at the begining of the 
            prints.
                                                                        
       Returns:                                                         
            X_train (Numpy array): train images.                    
            Y_train (Numpy array): train images' mask.              
            X_val (Numpy array, optional): validation images (create_val==True).
            Y_val (Numpy array, optional): validation images' mask 
            (create_val==True).
            X_test (Numpy array): test images.                      
            Y_test (Numpy array): test images' mask.                
            norm_value (int): normalization value calculated.
            crop_made (bool): True if crops have been made.
    """      
   
    if make_crops == True and prepare_subvolumes == True:
        raise ValueError("'make_crops' and 'prepare_subvolumes' both enabled are"
                         " incompatible")
    if prepare_subvolumes == True:
        if e_d_data:
            raise ValueError("No extra datasets can be used when "
                             "'prepare_subvolumes' is enabled")
        if subvol_shape is None:
            raise ValueError("'subvol_shape' must be provided if "
                             "'prepare_subvolumes' is enabled")
        if shuffle_val == True:
            raise ValueError("'shuffle_val' can not be enabled when "
                             "'prepare_subvolumes' is also enabled")

    Print(tab + "### LOAD ###")
                                                                        
    train_ids = sorted(next(os.walk(train_path))[2])                    
    train_mask_ids = sorted(next(os.walk(train_mask_path))[2])          
    
    test_ids = sorted(next(os.walk(test_path))[2])                      
    test_mask_ids = sorted(next(os.walk(test_mask_path))[2])            
                                                                        
    # Get and resize train images and masks                             
    X_train = np.zeros((len(train_ids), image_train_shape[1], 
                        image_train_shape[0], image_train_shape[2]),
                        dtype=np.float32)                
    Y_train = np.zeros((len(train_mask_ids), image_train_shape[1], 
                        image_train_shape[0], image_train_shape[2]),
                        dtype=np.float32) 
                                                                        
    Print(tab + "0) Loading train images . . .") 
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids), desc=tab):     
        img = imread(os.path.join(train_path, id_))                     
        # Convert the image into grayscale
        if len(img.shape) >= 3:
            img = img[:, :, 0]
            img = np.expand_dims(img, axis=-1)

        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        X_train[n] = img

    Print(tab + "1) Loading train masks . . .")
    for n, id_ in tqdm(enumerate(train_mask_ids), total=len(train_mask_ids), desc=tab):                      
        mask = imread(os.path.join(train_mask_path, id_))               
        # Convert the image into grayscale
        if len(mask.shape) >= 3:
            mask = mask[:, :, 0]
            mask = np.expand_dims(mask, axis=-1)

        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=-1)
        Y_train[n] = mask
                                                                        
    if num_crops_per_dataset != 0:
        X_train = X_train[:num_crops_per_dataset]
        Y_train = Y_train[:num_crops_per_dataset]

    # Get and resize test images and masks                              
    X_test = np.zeros((len(test_ids), image_test_shape[1], image_test_shape[0],
                      image_test_shape[2]), dtype=np.float32)                 
    Y_test = np.zeros((len(test_mask_ids), image_test_shape[1], 
                       image_test_shape[0], image_test_shape[2]), dtype=np.float32)
                                                                        
    Print(tab + "2) Loading test images . . .")
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids), desc=tab):       
        img = imread(os.path.join(test_path, id_))                      
        # Convert the image into grayscale
        if len(img.shape) >= 3:
            img = img[:, :, 0]
            img = np.expand_dims(img, axis=-1)

        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        X_test[n] = img

    Print(tab + "3) Loading test masks . . .")
    for n, id_ in tqdm(enumerate(test_mask_ids), total=len(test_mask_ids), desc=tab):                       
        mask = imread(os.path.join(test_mask_path, id_))                
        # Convert the image into grayscale
        if len(mask.shape) >= 3:
            mask = mask[:, :, 0]
            mask = np.expand_dims(mask, axis=-1)

        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=-1)
        Y_test[n] = mask

    Y_test = Y_test/255 

    # Used for 3D networks. This must be done before create the validation split
    # as the amount of images that will be in validation will not be enough to 
    # create 3D data subvolumes
    if prepare_subvolumes == True:                                              
        X_train, Y_train = prepare_subvolume_data(X_train, Y_train, subvol_shape)
        X_test, Y_test = prepare_subvolume_data(X_test, Y_test, subvol_shape)

    # Create validation data splitting the train
    if create_val == True:
        X_train, X_val, \
        Y_train, Y_val = train_test_split(X_train, Y_train,
                                          test_size=val_split,
                                          shuffle=shuffle_val,
                                          random_state=seedValue)
    # Crop the data
    if make_crops == True:
        Print(tab + "4) Crop data activated . . .")
        Print(tab + "    4.1) Cropping train data . . .")
        X_train, Y_train, _ = crop_data(X_train, crop_shape, data_mask=Y_train, 
                                        d_percentage=d_percentage, 
                                        tab=tab + "    ")    

        Print(tab + "    4.2) Cropping test data . . .")
        X_test, Y_test, _ = crop_data(X_test, crop_shape, data_mask=Y_test,
                                      tab=tab + "    ")
        
        if create_val == True:
            Print(tab + "    4.3) Cropping validation data . . .")
            X_val, Y_val, _ = crop_data(X_val, crop_shape, data_mask=Y_val,
                                        d_percentage=d_percentage,
                                        tab=tab + "    ")

        if check_crop == True:
            Print(tab + "    4.4) Checking the crops . . .")
            check_crops(X_train, [image_test_shape[0], image_test_shape[1]],
                        num_examples=3, out_dir="check_crops", job_id=job_id, 
                        suffix="_x_", grid=True, tab=tab + "    ")
            check_crops(Y_train, [image_test_shape[0], image_test_shape[1]],
                        num_examples=3, out_dir="check_crops", job_id=job_id, 
                        suffix="_y_", grid=True, tab=tab + "    ")
        
        image_test_shape[1] = crop_shape[1]
        image_test_shape[0] = crop_shape[0]
        crop_made = True
    else:
        crop_made = False

    # Load the extra datasets
    if e_d_data:
        Print(tab + "5) Loading extra datasets . . .")
        for i in range(len(e_d_data)):
            Print(tab + "    5." + str(i) + ") extra dataset in " 
                  + str(e_d_data[i]) + " . . .")
            train_ids = sorted(next(os.walk(e_d_data[i]))[2])
            train_mask_ids = sorted(next(os.walk(e_d_mask[i]))[2])

            d_dim = e_d_data_dim[i]
            e_X_train = np.zeros((len(train_ids), d_dim[1], d_dim[0], d_dim[2]),
                                 dtype=np.float32)
            e_Y_train = np.zeros((len(train_mask_ids), d_dim[1], d_dim[0], 
                                 d_dim[2]), dtype=np.float32)

            Print(tab + "    5." + str(i) + ") Loading data of the extra "
                  + "dataset . . .")
            for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids), desc=tab):
                im = imread(os.path.join(e_d_data[i], id_))
                if len(im.shape) == 2:
                    im = np.expand_dims(im, axis=-1)
                e_X_train[n] = im

            Print(tab + "    5." + str(i) + ") Loading masks of the extra "
                  + "dataset . . .")
            for n, id_ in tqdm(enumerate(train_mask_ids), total=len(train_mask_ids), desc=tab):
                mask = imread(os.path.join(e_d_mask[i], id_))
                if len(mask.shape) == 2:
                    mask = np.expand_dims(mask, axis=-1)
                e_Y_train[n] = mask

            if make_crops == False:
                assert d_dim[1] == image_test_shape[1] and \
                       d_dim[0] == image_test_shape[0], "Error: "\
                       + "extra dataset shape (" + str(d_dim) + ") is "\
                       + "not equal the original dataset shape ("\
                       + str(image_test_shape[1]) + ", "\
                       + str(image_test_shape[0]) + ")"        
            else:
                Print(tab + "    5." + str(i) + ") Cropping the extra dataset"
                      + " . . .")
                e_X_train, e_Y_train, _ = crop_data(e_X_train, crop_shape,
                                                    data_mask=e_Y_train, 
                                                    d_percentage=e_d_dis[i],
                                                    tab=tab + "    ")
                if num_crops_per_dataset != 0:
                    e_X_train = e_X_train[:num_crops_per_dataset]
                    e_Y_train = e_Y_train[:num_crops_per_dataset]

                if check_crop == True:
                    Print(tab + "    5." + str(i) + ") Checking the crops of the"
                          + " extra dataset . . .")
                    check_crops(e_X_train, [d_dim[0], d_dim[1]], num_examples=3, 
                                out_dir="check_crops", job_id=job_id, 
                                suffix="_e" + str(i) + "x_", grid=True, 
                                tab=tab + "    ")
                    check_crops(e_Y_train, [d_dim[0], d_dim[1]], num_examples=3,
                                out_dir="check_crops", job_id=job_id, 
                                suffix="_e" + str(i) + "y_", grid=True,
                                tab=tab + "    ")

            # Concatenate datasets
            X_train = np.vstack((X_train, e_X_train))
            Y_train = np.vstack((Y_train, e_Y_train))

    if create_val == True:                                            
        Print(tab + "*** Loaded train data shape is: " + str(X_train.shape))
        Print(tab + "*** Loaded validation data shape is: " + str(X_val.shape))
        Print(tab + "*** Loaded test data shape is: " + str(X_test.shape))
        Print(tab + "### END LOAD ###")

        # Calculate normalization value
        norm_value = np.mean(X_train)

        return X_train, Y_train, X_val, Y_val, X_test, Y_test, norm_value,\
               crop_made
    else:                                                               
        Print(tab + "*** Loaded train data shape is: " + str(X_train.shape))
        Print(tab + "*** Loaded test data shape is: " + str(X_test.shape))
        Print(tab + "### END LOAD ###")

        # Calculate normalization value
        norm_value = np.mean(X_train)

        return X_train, Y_train, X_test, Y_test, norm_value, crop_made                         


def __foreground_percentage(mask, class_tag=255):
    """ Percentage of pixels that corresponds to the class in the given image.
        
        Args: 
             mask (2D Numpy array): image mask to analize.
             class_tag (int, optional): class to find in the image.

        Returns:
             float: percentage of pixels that corresponds to the class. Value
             between 0 and 1.
    """

    c = 0
    for i in range(0, mask.shape[0]):
        for j in range(0, mask.shape[1]):     
            if mask[i][j] == class_tag:
                c = c + 1

    return (c*100)/(mask.shape[0]*mask.shape[1])


def crop_data(data, crop_shape, data_mask=None, force_shape=[0, 0], 
              d_percentage=0, tab=""):                          
    """Crop data into smaller pieces.
                                                                        
       Args:                                                            
            data (4D Numpy array): data to crop.                        
            crop_shape (str tuple): output image shape.
            data_mask (4D Numpy array, optional): data masks to crop.
            force_shape (int tuple, optional): force horizontal and vertical 
            crops to the given numbers.
            d_percentage (int, optional): number between 0 and 100. The images 
            that have less foreground pixels than the given number will be 
            discarded. Only available if data_mask is provided.
            tab (str, optional): tabulation mark to add at the begining of the
            prints.
                                                                        
       Returns:                                                         
            cropped_data (4D Numpy array): cropped data images.         
            cropped_data_mask (4D Numpy array): cropped data masks.     
            force_shape (int tuple): number of horizontal and vertical crops 
            made. Useful for future crop calls. 
    """                                                                 

    Print(tab + "### CROP ###")                                                                    
    Print(tab + "Cropping [" + str(data.shape[1]) + ', ' + str(data.shape[2]) 
          + "] images into " + str(crop_shape) + " . . .")
  
    # Calculate the number of images to be generated                    
    if force_shape == [0, 0]:
        h_num = int(data.shape[1] / crop_shape[0]) + (data.shape[1] % crop_shape[0] > 0)
        v_num = int(data.shape[2] / crop_shape[1]) + (data.shape[2] % crop_shape[1] > 0)
        force_shape = [h_num, v_num]
    else:
        h_num = force_shape[0]
        v_num = force_shape[1]
        Print(tab + "Force crops to [" + str(h_num) + ", " + str(v_num) + "]")

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
        Print(tab + "Resized data from " + str(data.shape) + " to " 
              + str(r_data.shape) + " to be divisible by the shape provided")

    discarded = 0                                                                    
    cont = 0
    selected_images  = []

    # Discard images from the data set
    if d_percentage > 0 and data_mask is not None:
        Print(tab + "0) Selecting images to discard . . .")
        for img_num in tqdm(range(0, r_data.shape[0]), desc=tab):                             
            for i in range(0, h_num):                                       
                for j in range(0, v_num):
                    p = __foreground_percentage(r_data_mask[img_num,
                                                            (i*crop_shape[0]):((i+1)*crop_shape[1]),
                                                            (j*crop_shape[0]):((j+1)*crop_shape[1])])
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
    Print(tab + "1) Cropping images . . .")
    for img_num in tqdm(range(0, r_data.shape[0]), desc=tab): 
        for i in range(0, h_num):                                       
            for j in range(0, v_num):                     
                if d_percentage > 0 and data_mask is not None \
                   and len(selected_images) != 0:
                    if selected_images[l_i] == cont \
                       or l_i == len(selected_images) - 1:

                        cropped_data[l_i] = r_data[img_num, (i*crop_shape[0]):((i+1)*crop_shape[1]), 
                                                   (j*crop_shape[0]):((j+1)*crop_shape[1]),:]

                        cropped_data_mask[l_i] = r_data_mask[img_num, (i*crop_shape[0]):((i+1)*crop_shape[1]),
                                                             (j*crop_shape[0]):((j+1)*crop_shape[1]),:]

                        if l_i != len(selected_images) - 1:
                            l_i = l_i + 1
                else: 
              
                    cropped_data[cont] = r_data[img_num, (i*crop_shape[0]):((i+1)*crop_shape[1]),      
                                                (j*crop_shape[0]):((j+1)*crop_shape[1]),:]
                                                                        
                    if data_mask is not None:
                        cropped_data_mask[cont] = r_data_mask[img_num, (i*crop_shape[0]):((i+1)*crop_shape[1]),
                                                              (j*crop_shape[0]):((j+1)*crop_shape[1]),:]
                cont = cont + 1                                             
                                                                        
    if d_percentage > 0 and data_mask is not None:
        Print(tab + "**** " + str(discarded) + " images discarded. New shape" 
              + " after cropping and discarding is " + str(cropped_data.shape))
        Print(tab + "### END CROP ###")
    else:
        Print(tab + "**** New data shape is: " + str(cropped_data.shape))
        Print(tab + "### END CROP ###")

    if data_mask is not None:
        return cropped_data, cropped_data_mask, force_shape
    else:
        return cropped_data, force_shape


def crop_data_with_overlap(data, data_mask, window_size, subdivision, tab=""):
    """Crop data into smaller pieces with the minimun overlap.

       Args:
            data (4D Numpy array): data to crop.
            data_mask (4D Numpy array): data mask to crop.
            window_size (int): crop size .
            subdivision (int): number of crops to create.
            tab (str, optional): tabulation mark to add at the begining of the
            prints.

       Returns:
            cropped_data (4D Numpy array): cropped image data.
            cropped_data_mask (4D Numpy array): cropped image data masks.
    """

    Print(tab + "### OV-CROP ###")
    Print(tab + "Cropping [" + str(data.shape[1]) + ', ' + str(data.shape[2])
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
                             data.shape[3]), dtype=np.float32)
    cropped_data_mask = np.zeros((total_cropped, window_size, window_size,
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
        
    Print(tab + "The minimum overlap has been found with rows=" + str(rows)  + 
          " and columns=" + str(columns))

    if subdivision != 1:
        assert window_size*rows >= data.shape[1], "Error: total width of all the"\
                + " crops per row must be greater or equal " + str(data.shape[1])\
                + " and it is only " + str(window_size*rows)
        assert window_size*columns >= data.shape[2], "Error: total width of all"\
                " the crops per column must be greater or equal " \
                + str(data.shape[2]) + " and it is only " \
                + str(window_size*columns)

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
    Print(tab + "0) Cropping data with the minimun overlap . . .")
    for k, img_num in tqdm(enumerate(range(0, data.shape[0])), desc=tab):
        for i in range(0, data.shape[1]-y_ov, step_y):
            for j in range(0, data.shape[2]-x_ov, step_x):
                d_y = 0 if (i+window_size) < data.shape[1] else r_y
                d_x = 0 if (j+window_size) < data.shape[2] else r_x

                cropped_data[cont] = data[k, i-d_y:i+window_size, j-d_x:j+window_size, :]
                cropped_data_mask[cont] = data_mask[k, i-d_y:i+window_size, j-d_x:j+window_size, :]
                cont = cont + 1

    Print(tab + "**** New data shape is: " + str(cropped_data.shape))
    Print(tab + "### END OV-CROP ###")

    return cropped_data, cropped_data_mask


def merge_data_with_overlap(data, original_shape, window_size, subdivision, 
                            out_dir, ov_map=True, ov_data_img=0, tab=""):
    """Merge data with an amount of overlap. Used to undo the crop made by the 
       function crop_data_with_overlap.

       Args:
            data (4D Numpy array): data to merge.
            original_shape (tuple): original dimensions to reconstruct. 
            window_size (int): crop size.
            subdivision (int): number of crops to merge.
            out_dir (string): directory where the images will be save.
            ov_map (bool, optional): whether to create overlap map.
            ov_data_img (int, optional): number of the image on the data to 
            create the overlappng map.
            tab (str, optional): tabulation mark to add at the begining of the
            prints.

       Returns:
            merged_data (4D Numpy array): merged image data.
    """

    Print(tab + "### MERGE-OV-CROP ###")
    Print(tab + "Merging [" + str(data.shape[1]) + ', ' + str(data.shape[2]) + 
          "] images into [" + str(original_shape[1]) + ", " 
          + str(original_shape[0]) + "] with overlapping . . .")

    # Merged data
    total_images = int(data.shape[0]/subdivision)
    merged_data = np.zeros((total_images, original_shape[1], original_shape[0],
                             data.shape[3]), dtype=np.float32)

    # Matrices to store the amount of overlap. The first is used to store the
    # number of crops to merge for each pixel. The second matrix is used to 
    # paint the overlapping map
    overlap_matrix = np.zeros((original_shape[1], original_shape[0],
                             data.shape[3]), dtype=np.float32)
    if ov_map == True:
        ov_map_matrix = np.zeros((original_shape[1], original_shape[0],
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

    Print(tab + "The minimum overlap has been found with [" + str(rows) + ", " 
          + str(columns) + "]")

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
    Print(tab + "0) Merging the overlapping crops . . .")
    for k, img_num in tqdm(enumerate(range(0, total_images)), desc=tab):
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
  
    Print(tab + "**** New data shape is: " + str(merged_data.shape))
    Print(tab + "### END MERGE-OV-CROP ###")

    return merged_data


def merge_data_without_overlap(data, num, out_shape=[1, 1], grid=True, tab=""):
    """Combine images from input data into a bigger one given shape. It is the 
       opposite function of crop_data().

       Args:                                                                    
            data (4D Numpy array): data to crop.                                
            num (int, optional): number of examples to convert.
            out_shape (int tuple, optional): number of horizontal and vertical
            images to combine in a single one.
            grid (bool, optional): make the grid in the output image.
            tab (str, optional): tabulation mark to add at the begining of the
            prints.
                                                                                
       Returns:                                                                 
            mixed_data (4D Numpy array): mixed data images.                 
            mixed_data_mask (4D Numpy array): mixed data masks.
    """

    Print(tab + "### MERGE-CROP ###")

    # To difference between data and masks
    if grid == True:
        if np.max(data) > 1:
            v = 255
        else:
            v = 1

    width = data.shape[1]
    height = data.shape[2] 

    mixed_data = np.zeros((num, out_shape[1]*width, out_shape[0]*height, 
                           data.shape[3]), dtype=np.float32)
    cont = 0
    Print(tab + "0) Merging crops . . .")
    for img_num in tqdm(range(0, num), desc=tab):
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

    Print(tab + "### END MERGE-CROP ###")
    return mixed_data


def check_crops(data, out_dim, num_examples=2, include_crops=True,
                out_dir="check_crops", job_id="none_job_id", suffix="_none_", 
                grid=True, tab=""):
    """Check cropped images by the function crop_data(). 
        
       Args:
            data (4D Numpy array): data to crop.
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
            tab (str, optional): tabulation mark to add at the begining of the
            prints.
    """
  
    Print(tab + "### CHECK-CROPS ###")
 
    assert out_dim[0] >= data.shape[1] or out_dim[1] >= data.shape[2], "Error: "\
           + " out_dim must be equal or greater than data.shape"

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
        Print(tab + "Requested num_examples too high for data. Set automatically"
              + " to " + str(num_examples))
    else:
        total = total*num_examples

    if include_crops == True:
        Print(tab + "0) Saving cropped data images . . .")
        for i in tqdm(range(0, total), desc=tab):
            # grayscale images
            if data.shape[3] == 1:
                im = Image.fromarray(data[i,:,:,0]*v)
                im = im.convert('L')
            # RGB images
            else:
                aux = np.asarray( data[i,:,:,:]*v, dtype="uint8" )
                im = Image.fromarray( aux, 'RGB' )

            im.save(os.path.join(out_dir,"c_" + suffix + str(i) + ".png"))

    Print(tab + "0) Reconstructing " + str(num_examples) + " images of ["
          + str(data.shape[1]*h_num) + "," + str(data.shape[2]*v_num) + "] from "
          + "[" + str(data.shape[1]) + "," + str(data.shape[2]) + "] crops")
    m_data = merge_data_without_overlap(data, num_examples, 
                                        out_shape=[h_num, v_num], grid=grid,
                                        tab=tab + "    ") 
    Print(tab + "1) Saving data mixed images . . .")
    for i in tqdm(range(0, num_examples), desc=tab):
        im = Image.fromarray(m_data[i,:,:,0]*v)
        im = im.convert('L')
        im.save(os.path.join(out_dir,"f" + suffix + str(i) + ".png"))

    Print(tab + "### END CHECK-CROP ###")


def check_binary_masks(path):
    """Check wheter the data masks is binary checking the a few random images of
       the given path. If the function gives no error one should assume that the
       masks are correct.

       Args:
            path (str): path to the data mask.
    """
    Print("Checking wheter the images in " + path + " are binary . . .")

    ids = sorted(next(os.walk(path))[2])

    # Check only 4 random images or less if there are not as many
    num_sample = [4, len(ids)]
    numbers = random.sample(range(0, len(ids)), min(num_sample))
    for i in numbers:
        img = imread(os.path.join(path, ids[i]))
        values, _ = np.unique(img, return_counts=True)
        if len(values) > 2 :
            raise ValueError("Error: given masks are not binary. Please correct "
                             "the images before training. (image: {})\n"
                             "Values: {}".format(os.path.join(path, ids[i]), values))


def prepare_subvolume_data(X, Y, shape=(82, 256, 256, 1)):                          
    """Prepare given data into 3D subvolumes to train a 3D network.

       Args:
            X (Numpy 4D array): data. E.g. (image_number, x, y, channels).      
                                                                                
            Y (Numpy 4D array): mask data.  E.g. (image_number, x, y, channels).
                                                                                
            shape (tuple, optional): dimension of the desired images.           
    
       Returns:
            X_prep (Numpy 5D array): X data separated in different subvolumes 
            with the desired shape. E.g. (subvolume_number, ) + shape
            
            Y_prep (Numpy 5D array): Y data separated in different subvolumes     
            with the desired shape. E.g. (subvolume_number, ) + shape
    """
                                                                            
    if X.shape != Y.shape:                                                      
        raise ValueError("The shape of X and Y must be the same")               
    if X.ndim != 4 or Y.ndim != 4:                                              
        raise ValueError("X or Y must be a 4D Numpy array")                     
    if len(shape) != 4:
        raise ValueError("'shape' must be 4D")
    if X.shape[1] % shape[1] != 0 or X.shape[2] % shape[2] != 0:                
        raise ValueError("Shape must be divisible by the shape of X" )          
                                                                                
    # Calculate the rest                                                        
    rest = X.shape[0] % shape[0]                                                
    if rest != 0:                                                               
        print(("As the number of images required to form a stack 3D is "        
               "not multiple of images provided, {} last image(s) will "        
               "be unused").format(rest))                                       
                                                                                
    # Calculate of many crops are per axis                                      
    h_num = int(X.shape[1]/shape[1])
    v_num = int(X.shape[2]/shape[2])
    crops_per_image = h_num*v_num                                               
                                                                                
    num_sub_volum = int(math.floor(X.shape[0]/shape[0])*crops_per_image)             
    print("{} subvolumes of {} will be created".format(num_sub_volum,           
          shape[-3:]))                                                          
                                                                                
    X_prep = np.zeros((num_sub_volum, ) + shape)                                
    Y_prep = np.zeros((num_sub_volum, ) + shape)                                
                                                                                
    # Reshape the data to generate desired 3D subvolumes                        
    print("Generating 3D subvolumes . . .")                                     
    print("Converting {} data to {}".format(X.shape, X_prep.shape))             
    print("Filling [0:{}] subvolumes with [0:{}] images . . ."                  
          .format(crops_per_image-1, shape[0]-1))                               
    subvolume = 0                                                               
    vol_slice = 0                                                               
    total_subvol_filled = 0                                                     
    for k in range(X.shape[0]-rest):                                                 
        for i in range(h_num):                                                  
            for j in range(v_num):                                              
                im = X[k, (i*shape[1]):((i+1)*shape[2]),(j*shape[1]):((j+1)*shape[2])]
                mask = Y[k, (i*shape[1]):((i+1)*shape[2]),(j*shape[1]):((j+1)*shape[2])]
                  
                X_prep[subvolume, vol_slice] = im                               
                Y_prep[subvolume, vol_slice] = mask                             
                                                                                
                subvolume += 1                                                  
                if subvolume == (total_subvol_filled+crops_per_image):
                    subvolume = total_subvol_filled
                    vol_slice += 1                                              
                                                                                
                    # Reached this point we will have filled part of            
                    # the subvolumes                                            
                    if vol_slice == shape[0] and (k+1) != (X.shape[0]-rest):                                   
                        total_subvol_filled += crops_per_image                  
                        subvolume = total_subvol_filled
                        vol_slice = 0                                           
                                                                                
                        print("Filling [{}:{}] subvolumes with [{}:{}] "        
                              "images . . .".format(total_subvol_filled,        
                              total_subvol_filled+crops_per_image-1, k+1,       
                              k+shape[0]-1))

    return X_prep, Y_prep
