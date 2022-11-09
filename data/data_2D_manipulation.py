from re import X
import numpy as np
import os
import math
from tqdm import tqdm
from skimage.io import imread
from sklearn.model_selection import train_test_split, StratifiedKFold
from PIL import Image
from utils.util import load_data_from_dir
from data.pre_processing import normalize
from skimage.io import imsave

def load_and_prepare_2D_train_data(train_path, train_mask_path, val_split=0.1, seed=0, shuffle_val=True, e_d_data=[],
    e_d_mask=[], e_d_data_dim=[], num_crops_per_dataset=0, random_crops_in_DA=False, crop_shape=None, ov=(0,0),
    padding=(0,0), reflect_to_complete_shape=False):
    """Load train and validation images from the given paths to create 2D data.

       Parameters
       ----------
       train_path : str
           Path to the training data.

       train_mask_path : str
           Path to the training data masks.

       val_split : float, optional
            % of the train data used as validation (value between ``0`` and ``1``).

       seed : int, optional
            Seed value.

       shuffle_val : bool, optional
            Take random training examples to create validation data.

       e_d_data : list of str, optional
           List of paths where the extra data of other datasets are stored. If ``make_crops`` is not enabled, these
           extra datasets must have the same image shape as the main dataset since they are going to be stacked in a
           unique array.

       e_d_mask : list of str, optional
           List of paths where the extra data mask of other datasets are stored. Same constraints as ``e_d_data``.

       e_d_data_dim : list of 3D int tuple, optional
           List of shapes of the extra datasets provided. Same constraints as ``e_d_data``.

       num_crops_per_dataset : int, optional
           Number of crops per extra dataset to take into account. Useful to ensure that all the datasets have the same
           weight during network trainning.

       random_crops_in_DA : bool, optional
           To advice the method that not preparation of the data must be done, as random subvolumes will be created on
           DA, and the whole volume will be used for that.

       crop_shape : 3D int tuple, optional
           Shape of the crops. E.g. ``(x, y, channels)``.

       ov : 2 floats tuple, optional
           Amount of minimum overlap on x and y dimensions. The values must be on range ``[0, 1)``, that is, ``0%`` or
           ``99%`` of overlap. E.g. ``(x, y)``.

       padding : tuple of ints, optional
           Size of padding to be added on each axis ``(x, y)``. E.g. ``(24, 24)``

       reflect_to_complete_shape : bool, optional
           Wheter to increase the shape of the dimension that have less size than selected patch size padding it with
           'reflect'.
           
       Returns
       -------
       X_train : 4D Numpy array
           Train images. E.g. ``(num_of_images, y, x, channels)``.

       Y_train : 4D Numpy array
           Train images' mask. E.g. ``(num_of_images, y, x, channels)``.

       X_val : 4D Numpy array, optional
           Validation images (``val_split > 0``). E.g. ``(num_of_images, y, x, channels)``.

       Y_val : 4D Numpy array, optional
           Validation images' mask (``val_split > 0``). E.g. ``(num_of_images, y, x, channels)``.

       filenames : List of str
           Loaded train filenames.

       Examples
       --------
       ::

           # EXAMPLE 1
           # Case where we need to load the data (creating a validation split)
           train_path = "data/train/x"
           train_mask_path = "data/train/y"

           # Original image shape is (1024, 768, 165), so each image shape should be this:
           img_train_shape = (1024, 768, 1)

           X_train, Y_train, X_val,
           Y_val, crops_made = load_and_prepare_2D_data(train_path, train_mask_path, img_train_shape, val_split=0.1,
               shuffle_val=True, make_crops=False)


           # The function will print the shapes of the generated arrays. In this example:
           #     *** Loaded train data shape is: (148, 768, 1024, 1)
           #     *** Loaded validation data shape is: (17, 768, 1024, 1)
           #
           # Notice height and width swap because of Numpy ndarray terminology


           # EXAMPLE 2
           # Same as the first example but creating patches of (256x256)
           X_train, Y_train, X_val,
           Y_val, crops_made = load_and_prepare_2D_data(train_path, train_mask_path, img_train_shape, val_split=0.1,
               shuffle_val=True, make_crops=True, crop_shape=(256, 256, 1))

           # The function will print the shapes of the generated arrays. In this example:
           #    *** Loaded train data shape is: (1776, 256, 256, 1)
           #    *** Loaded validation data shape is: (204, 256, 256, 1)


           # EXAMPLE 3
           # Same as the first example but defining extra datasets to be loaded and stacked together
           # with the main dataset. Extra variables to be defined:
           extra_datasets_data_list.append('/data2/train/x')
           extra_datasets_mask_list.append('/data2/train/y')
           extra_datasets_data_dim_list.append((877, 967, 1))

           X_train, Y_train, X_val,
           Y_val, crops_made = load_and_prepare_2D_data(train_path, train_mask_path, img_train_shape, val_split=0.1,
               shuffle_val=True, make_crops=True, crop_shape=(256, 256, 1), e_d_data=extra_datasets_data_list, 
               e_d_mask=extra_datasets_mask_list, e_d_data_dim=extra_datasets_data_dim_list)
    """

    print("### LOAD ###")

    # Disable crops when random_crops_in_DA is selected
    crop = False if random_crops_in_DA else True

    # Check validation
    create_val = True if val_split > 0 else False

    print("0) Loading train images . . .")
    X_train, orig_train_shape, _, t_filenames = load_data_from_dir(train_path, crop=crop, crop_shape=crop_shape, overlap=ov,
                                                         padding=padding, return_filenames=True,
                                                         reflect_to_complete_shape=reflect_to_complete_shape)
    if train_mask_path is not None:                                            
        print("1) Loading train masks . . .")
        Y_train, _, _, _ = load_data_from_dir(train_mask_path, crop=crop, crop_shape=crop_shape, overlap=ov,
                                                        padding=padding, return_filenames=True,
                                                        reflect_to_complete_shape=reflect_to_complete_shape)
    else:
        Y_train = np.zeros(X_train.shape, dtype=np.uint8) # Fake mask val

    if num_crops_per_dataset != 0:
        X_train = X_train[:num_crops_per_dataset]
        Y_train = Y_train[:num_crops_per_dataset]

    # Create validation data splitting the train
    if create_val:
        X_train, X_val, Y_train, Y_val = train_test_split(
            X_train, Y_train, test_size=val_split, shuffle=shuffle_val, random_state=seed)

    # Load the extra datasets
    if e_d_data:
        print("Loading extra datasets . . .")
        for i in range(len(e_d_data)):
            print("{} extra dataset in {} . . .".format(i, e_d_data[i]))
            train_ids = sorted(next(os.walk(e_d_data[i]))[2])
            train_mask_ids = sorted(next(os.walk(e_d_mask[i]))[2])

            d_dim = e_d_data_dim[i]
            e_X_train = np.zeros((len(train_ids), d_dim[1], d_dim[0], d_dim[2]), dtype=np.float32)
            e_Y_train = np.zeros((len(train_mask_ids), d_dim[1], d_dim[0], d_dim[2]), dtype=np.float32)

            print("{} Loading data of the extra dataset . . .".format(i))
            for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
                im = imread(os.path.join(e_d_data[i], id_))
                if len(im.shape) == 2:
                    im = np.expand_dims(im, axis=-1)
                # Ensure uint8
                if im.dtype == np.uint16:
                    if np.max(im) > 255:
                        im = normalize(im, 0, 65535)
                    else:
                        im = im.astype(np.uint8)
                e_X_train[n] = im

            print("{} Loading masks of the extra dataset . . .".format(i))
            for n, id_ in tqdm(enumerate(train_mask_ids), total=len(train_mask_ids)):
                mask = imread(os.path.join(e_d_mask[i], id_))
                if len(mask.shape) == 2:
                    mask = np.expand_dims(mask, axis=-1)
                e_Y_train[n] = mask

            print("{} Cropping the extra dataset . . .".format(i))
            if crop_shape != e_X_train.shape[1:]:
                e_X_train, e_Y_train = crop_data_with_overlap(e_X_train, crop_shape, data_mask=e_Y_train,
                                                              overlap=ov, padding=padding, verbose=False)

            if num_crops_per_dataset != 0:
                e_X_train = e_X_train[:num_crops_per_dataset]
                e_Y_train = e_Y_train[:num_crops_per_dataset]

            # Concatenate datasets
            X_train = np.vstack((X_train, e_X_train))
            Y_train = np.vstack((Y_train, e_Y_train))

    s = X_train.shape if not random_crops_in_DA else X_train[0].shape
    sm = Y_train.shape if not random_crops_in_DA else Y_train[0].shape
    if create_val:
        sv = X_val.shape if not random_crops_in_DA else X_val[0].shape
        svm = Y_val.shape if not random_crops_in_DA else Y_val[0].shape
        print("*** Loaded train data shape is: {}".format(s))
        print("*** Loaded train mask shape is: {}".format(sm))
        print("*** Loaded validation data shape is: {}".format(sv))
        print("*** Loaded validation mask shape is: {}".format(svm))
        print("### END LOAD ###")

        return X_train, Y_train, X_val, Y_val, t_filenames
    else:
        print("*** Loaded train data shape is: {}".format(s))
        print("### END LOAD ###")

        return X_train, Y_train, t_filenames


def crop_data_with_overlap(data, crop_shape, data_mask=None, overlap=(0,0), padding=(0,0), verbose=True):
    """Crop data into small square pieces with overlap. The difference with :func:`~crop_data` is that this function
       allows you to create patches with overlap.

       The opposite function is :func:`~merge_data_with_overlap`.

       Parameters
       ----------
       data : 4D Numpy array
           Data to crop. E.g. ``(num_of_images, y, x, channels)``.

       crop_shape : 3 int tuple
           Shape of the crops to create. E.g. ``(x, y, channels)``.

       data_mask : 4D Numpy array, optional
           Data mask to crop. E.g. ``(num_of_images, y, x, channels)``.

       overlap : Tuple of 2 floats, optional
           Amount of minimum overlap on x and y dimensions. The values must be on range ``[0, 1)``, that is, ``0%`` or
           ``99%`` of overlap. E. g. ``(x, y)``.

       padding : tuple of ints, optional
           Size of padding to be added on each axis ``(x, y)``. E.g. ``(24, 24)``.

       verbose : bool, optional
            To print information about the crop to be made.

       Returns
       -------
       cropped_data : 4D Numpy array
           Cropped image data. E.g. ``(num_of_images, y, x, channels)``.

       cropped_data_mask : 4D Numpy array, optional
           Cropped image data masks. E.g. ``(num_of_images, y, x, channels)``.

       Examples
       --------
       ::

           # EXAMPLE 1
           # Divide in crops of (256, 256) a given data with the minimum overlap
           X_train = np.ones((165, 768, 1024, 1))
           Y_train = np.ones((165, 768, 1024, 1))

           X_train, Y_train = crop_data_with_overlap(X_train, (256, 256, 1), Y_train, (0, 0))

           # Notice that as the shape of the data has exact division with the wnanted crops shape so no overlap will be
           # made. The function will print the following information:
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

    if data_mask is not None:
        if data.shape[:-1] != data_mask.shape[:-1]:
            raise ValueError("data and data_mask shapes mismatch: {} vs {}".format(data.shape[:-1], data_mask.shape[:-1]))

    for i,p in enumerate(padding):
        if p >= crop_shape[i]//2:
            raise ValueError("'Padding' can not be greater than the half of 'crop_shape'. Max value for this {} input shape is {}"
                              .format(data.shape, [(crop_shape[0]//2)-1,(crop_shape[1]//2)-1]))

    if verbose:
        print("### OV-CROP ###")
        print("Cropping {} images into {} with overlapping. . ."\
              .format(data.shape, crop_shape))
        print("Minimum overlap selected: {}".format(overlap))
        print("Padding: {}".format(padding))

    if (overlap[0] >= 1 or overlap[0] < 0) and (overlap[1] >= 1 or overlap[1] < 0):
        raise ValueError("'overlap' values must be floats between range [0, 1)")

    padded_data = np.pad(data,((0,0),(padding[1],padding[1]),(padding[0],padding[0]),(0,0)), 'reflect')
    if data_mask is not None:
        padded_data_mask = np.pad(data_mask,((0,0),(padding[1],padding[1]),(padding[0],padding[0]),(0,0)), 'reflect')

    crop_shape = tuple(crop_shape[i] for i in [1, 0, 2])
    padding = tuple(padding[i] for i in [1, 0])

    # Calculate overlapping variables
    overlap_x = 1 if overlap[0] == 0 else 1-overlap[0]
    overlap_y = 1 if overlap[1] == 0 else 1-overlap[1]

    # Y
    step_y = int((crop_shape[0]-padding[0]*2)*overlap_y)
    crops_per_y = math.ceil(data.shape[1]/step_y)
    last_y = 0 if crops_per_y == 1 else (((crops_per_y-1)*step_y)+crop_shape[0])-padded_data.shape[1]

    # X
    step_x = int((crop_shape[1]-padding[1]*2)*overlap_x)
    crops_per_x = math.ceil(data.shape[2]/step_x)
    last_x = 0 if crops_per_x == 1 else (((crops_per_x-1)*step_x)+crop_shape[1])-padded_data.shape[2]

    # Real overlap calculation for printing
    real_ov_y = ((crop_shape[0]-padding[0]*2)-step_y)/(crop_shape[0]-padding[0]*2)
    real_ov_x = ((crop_shape[1]-padding[1]*2)-step_x)/(crop_shape[1]-padding[1]*2)

    if verbose:
        print("Real overlapping (%): {}".format(real_ov_x,real_ov_y))
        print("Real overlapping (pixels): {}".format((crop_shape[1]-padding[1]*2)*real_ov_x,
            (crop_shape[0]-padding[0]*2)*real_ov_y))
        print("{} patches per (x,y) axis".format(crops_per_x,crops_per_y))

    total_vol = data.shape[0]*(crops_per_x)*(crops_per_y)
    cropped_data = np.zeros((total_vol,) + crop_shape, dtype=data.dtype)
    if data_mask is not None:
        cropped_data_mask = np.zeros((total_vol,)+crop_shape[:2]+(data_mask.shape[-1],), dtype=data_mask.dtype)

    c = 0
    for z in range(data.shape[0]):
        for y in range(crops_per_y):
            for x in range(crops_per_x):
                d_y = 0 if (y*step_y+crop_shape[1]) < padded_data.shape[1] else last_y
                d_x = 0 if (x*step_x+crop_shape[0]) < padded_data.shape[2] else last_x

                cropped_data[c] = \
                    padded_data[z,
                                y*step_y-d_y:y*step_y+crop_shape[1]-d_y,
                                x*step_x-d_x:x*step_x+crop_shape[0]-d_x]

                if data_mask is not None:
                    cropped_data_mask[c] = \
                        padded_data_mask[z,
                                         y*step_y-d_y:y*step_y+crop_shape[1]-d_y,
                                         x*step_x-d_x:x*step_x+crop_shape[0]-d_x]
                c += 1

    if verbose:
        print("**** New data shape is: {}".format(cropped_data.shape))
        print("### END OV-CROP ###")

    if data_mask is not None:
        return cropped_data, cropped_data_mask
    else:
        return cropped_data


def merge_data_with_overlap(data, original_shape, data_mask=None, overlap=(0,0), padding=(0,0), verbose=True,
    out_dir=None, prefix=""):
    """Merge data with an amount of overlap.

       The opposite function is :func:`~crop_data_with_overlap`.

       Parameters
       ----------
       data : 4D Numpy array
           Data to merge. E.g. ``(num_of_images, y, x, channels)``.

       original_shape : 4D int tuple
           Shape of the original data. E.g. ``(num_of_images, y, x, channels)``

       data_mask : 4D Numpy array, optional
           Data mask to merge. E.g. ``(num_of_images, y, x, channels)``.

       overlap : Tuple of 2 floats, optional
           Amount of minimum overlap on x and y dimensions. Should be the same as used in
           :func:`~crop_data_with_overlap`. The values must be on range ``[0, 1)``, that is, ``0%`` or ``99%`` of
           overlap. E. g. ``(y, x)``.

       padding : tuple of ints, optional
           Size of padding to be added on each axis ``(y, x)``. E.g. ``(24, 24)``.

       verbose : bool, optional
            To print information about the crop to be made.

       out_dir : str, optional
           If provided an image that represents the overlap made will be saved. The image will be colored as follows:
           green region when ``==2`` crops overlap, yellow when ``2 < x < 6`` and red when ``=<6`` or more crops are
           merged.

       prefix : str, optional
           Prefix to save overlap map with.

       Returns
       -------
       merged_data : 4D Numpy array
           Merged image data. E.g. ``(num_of_images, y, x, channels)``.

       merged_data_mask : 4D Numpy array, optional
           Merged image data mask. E.g. ``(num_of_images, y, x, channels)``.

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

       +--------------------------------------------+--------------------------------------------+
       | .. figure:: ../img/merged_ov_map_0.png     | .. figure:: ../img/merged_ov_map_0.25.png  |
       |   :width: 80%                              |   :width: 80%                              |
       |   :align: center                           |   :align: center                           |
       |                                            |                                            |
       |   Example 1 overlapping map                |   Example 2 overlapping map                |
       +--------------------------------------------+--------------------------------------------+
       | .. figure:: ../img/merged_ov_map_0.5.png   | .. figure:: ../img/merged_ov_map_0.5inx.png|
       |   :width: 80%                              |   :width: 80%                              |
       |   :align: center                           |   :align: center                           |
       |                                            |                                            |
       |   Example 3 overlapping map                |   Example 4 overlapping map                |
       +--------------------------------------------+--------------------------------------------+
    """

    if data_mask is not None:
        if data.shape[:-1] != data_mask.shape[:-1]:
            raise ValueError("data and data_mask shapes mismatch: {} vs {}".format(data.shape[:-1], data_mask.shape[:-1]))

    for i,p in enumerate(padding):
        if p >= data.shape[i+1]//2:
            raise ValueError("'Padding' can not be greater than the half of 'data' shape. Max value for this {} input shape is {}"
                                .format(data.shape, [(data.shape[1]//2)-1,(data.shape[2]//2)-1]))

    if verbose:
        print("### MERGE-OV-CROP ###")
        print("Merging {} images into {} with overlapping . . .".format(data.shape, original_shape))
        print("Minimum overlap selected: {}".format(overlap))
        print("Padding: {}".format(padding))

    if (overlap[0] >= 1 or overlap[0] < 0) and (overlap[1] >= 1 or overlap[1] < 0):
        raise ValueError("'overlap' values must be floats between range [0, 1)")

    padding = tuple(padding[i] for i in [1, 0])

    # Remove the padding
    pad_input_shape = data.shape
    data = data[:, padding[0]:data.shape[1]-padding[0], padding[1]:data.shape[2]-padding[1]]

    merged_data = np.zeros((original_shape), dtype=np.float32)
    if data_mask is not None:
        merged_data_mask = np.zeros((original_shape), dtype=np.float32)
        data_mask = data_mask[:, padding[0]:data_mask.shape[1]-padding[0], padding[1]:data_mask.shape[2]-padding[1]]

    ov_map_counter = np.zeros(original_shape, dtype=np.int32)
    if out_dir is not None:
        crop_grid = np.zeros(original_shape[1:], dtype=np.int32)

    # Calculate overlapping variables
    overlap_x = 1 if overlap[0] == 0 else 1-overlap[0]
    overlap_y = 1 if overlap[1] == 0 else 1-overlap[1]

    padded_data_shape = [original_shape[1]+2*padding[0], original_shape[2]+2*padding[1]]

    # Y
    step_y = int((pad_input_shape[1]-padding[0]*2)*overlap_y)
    crops_per_y = math.ceil(original_shape[1]/step_y)
    last_y = 0 if crops_per_y == 1 else (((crops_per_y-1)*step_y)+pad_input_shape[1])-padded_data_shape[0]

    # X
    step_x = int((pad_input_shape[2]-padding[1]*2)*overlap_x)
    crops_per_x = math.ceil(original_shape[2]/step_x)
    last_x = 0 if crops_per_x == 1 else (((crops_per_x-1)*step_x)+pad_input_shape[2])-padded_data_shape[1]

    # Real overlap calculation for printing
    real_ov_y = ((pad_input_shape[1]-padding[0]*2)-step_y)/(pad_input_shape[1]-padding[0]*2)
    real_ov_x = ((pad_input_shape[2]-padding[1]*2)-step_x)/(pad_input_shape[2]-padding[1]*2)
    if verbose:
        print("Real overlapping (%): {}".format((real_ov_x,real_ov_y)))
        print("Real overlapping (pixels): {}".format(((pad_input_shape[2]-padding[1]*2)*real_ov_x,
            (pad_input_shape[1]-padding[0]*2)*real_ov_y)))
        print("{} patches per (x,y) axis".format((crops_per_x,crops_per_y)))

    c = 0
    for z in range(original_shape[0]):
        for y in range(crops_per_y):
            for x in range(crops_per_x):
                d_y = 0 if (y*step_y+data.shape[1]) < original_shape[1] else last_y
                d_x = 0 if (x*step_x+data.shape[2]) < original_shape[2] else last_x

                merged_data[z,y*step_y-d_y:y*step_y+data.shape[1]-d_y, x*step_x-d_x:x*step_x+data.shape[2]-d_x] += data[c]

                if data_mask is not None:
                    merged_data_mask[z, y*step_y-d_y:y*step_y+data.shape[1]-d_y, x*step_x-d_x:x*step_x+data.shape[2]-d_x] += data_mask[c]

                ov_map_counter[z, y*step_y-d_y:y*step_y+data.shape[1]-d_y, x*step_x-d_x:x*step_x+data.shape[2]-d_x] += 1

                if z == 0 and out_dir is not None:
                    crop_grid[y*step_y-d_y:y*step_y+data.shape[1]-d_y, x*step_x-d_x] = 1
                    crop_grid[y*step_y-d_y:y*step_y+data.shape[1]-d_y, x*step_x+data.shape[2]-d_x-1] = 1
                    crop_grid[y*step_y-d_y, x*step_x-d_x:x*step_x+data.shape[2]-d_x] = 1
                    crop_grid[y*step_y+data.shape[1]-d_y-1, x*step_x-d_x:x*step_x+data.shape[2]-d_x] = 1

                c += 1

    merged_data = np.true_divide(merged_data, ov_map_counter).astype(data.dtype)
    if data_mask is not None:
        merged_data_mask = np.true_divide(merged_data_mask, ov_map_counter).astype(data_mask.dtype)

    # Save a copy of the merged data with the overlapped regions colored as: green when 2 crops overlap, yellow when
    # (2 < x < 6) and red when more than 6 overlaps are merged
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
                if ov_map[im_j, im_i, 0] == -4:
                    px[im_i, im_j] = (255, 255, 255, 255)
                # Overlap zone
                elif ov_map[im_j, im_i, 0] == -3:
                    px[im_i, im_j] = tuple(map(sum, zip((0, 74, 0, 125), px[im_i, im_j])))
                # 2 < x < 6 overlaps
                elif ov_map[im_j, im_i, 0] == -2:
                    px[im_i, im_j] = tuple(map(sum, zip((74, 74, 0, 125), px[im_i, im_j])))
                # 6 >= overlaps
                elif ov_map[im_j, im_i, 0] == -1:
                    px[im_i, im_j] = tuple(map(sum, zip((74, 0, 0, 125), px[im_i, im_j])))

        im.save(os.path.join(out_dir, prefix + "merged_ov_map.png"))

    if verbose:
        print("**** New data shape is: {}".format(merged_data.shape))
        print("### END MERGE-OV-CROP ###")

    if data_mask is not None:
        return merged_data, merged_data_mask
    else:
        return merged_data


def random_crop(image, mask, random_crop_size, val=False, draw_prob_map_points=False, img_prob=None, weight_map=None,
                scale=1):
    """Random crop.

       Parameters
       ----------
       image : Numpy 3D array
           Image. E.g. ``(y, x, channels)``.

       mask : Numpy 3D array
           Image mask. E.g. ``(y, x, channels)``.

       random_crop_size : 2 int tuple
           Size of the crop. E.g. ``(height, width)``.

       val : bool, optional
           If the image provided is going to be used in the validation data. This forces to crop from the origin,
           e. g. ``(0, 0)`` point.

       draw_prob_map_points : bool, optional
           To return the pixel chosen to be the center of the crop.

       img_prob : Numpy 3D array, optional
           Probability of each pixel to be chosen as the center of the crop. E. .g. ``(y, x, channels)``.

       weight_map : bool, optional
           Weight map of the given image. E.g. ``(y, x, channels)``.

       scale : int, optional
           Scale factor the second image given.

       Returns
       -------
       img : 2D Numpy array
           Crop of the given image. E.g. ``(y, x)``.

       weight_map : 2D Numpy array, optional
           Crop of the given image's weigth map. E.g. ``(y, x)``.

       ox : int, optional
           X coordinate in the complete image of the chose central pixel to make the crop.

       oy : int, optional
           Y coordinate in the complete image of the chose central pixel to make the crop.

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
    dy, dx = random_crop_size[0]//scale, random_crop_size[1]//scale
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
        return img[y:(y+dy), x:(x+dx)], mask[y*scale:(y+dy)*scale, x*scale:(x+dx)*scale], oy, ox, y, x
    else:
        if weight_map is not None:
            return img[y:(y+dy), x:(x+dx)], mask[y*scale:(y+dy)*scale, x*scale:(x+dx)*scale], weight_map[y:(y+dy), x:(x+dx)]
        else:
            return img[y:(y+dy), x:(x+dx)], mask[y*scale:(y+dy)*scale, x*scale:(x+dx)*scale]


def random_crop_classification(image, random_crop_size, val=False, draw_prob_map_points=False, weight_map=None):
    """Random crop.

       Parameters
       ----------
       image : Numpy 3D array
           Image. E.g. ``(x, y, channels)``.

       random_crop_size : 2 int tuple
           Size of the crop. E.g. ``(height, width)``.

       val : bool, optional
           If the image provided is going to be used in the validation data. This forces to crop from the origin,
           e. g. ``(0, 0)`` point.

       draw_prob_map_points : bool, optional
           To return the pixel chosen to be the center of the crop.

       weight_map : bool, optional
           Weight map of the given image. E.g. ``(x, y, channels)``.

       Returns
       -------
       img : 2D Numpy array
           Crop of the given image. E.g. ``(x, y)``.

       weight_map : 2D Numpy array, optional
           Crop of the given image's weigth map. E.g. ``(x, y)``.

       ox : int, optional
           X coordinate in the complete image of the chose central pixel to make the crop.

       oy : int, optional
           Y coordinate in the complete image of the chose central pixel to make the crop.

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
        ox = 0
        oy = 0
        x = np.random.randint(0, width - dx + 1)
        y = np.random.randint(0, height - dy + 1)

    if draw_prob_map_points == True:
        return img[y:(y+dy), x:(x+dx)], ox, oy, x, y
    else:
        if weight_map is not None:
            return img[y:(y+dy), x:(x+dx)], weight_map[y:(y+dy), x:(x+dx)]
        else:
            return img[y:(y+dy), x:(x+dx)]


def load_data_classification(cfg, test=False):
    """Load data to train classification methods.

       Parameters
       ----------
       test : bool, optional
           To load test data isntead of train/validation.

       Returns
       -------
       X_data : 4D Numpy array
           Train/test images. E.g. ``(num_of_images, y, x, channels)``.

       Y_data : 1D Numpy array
           Train/test images' classes. E.g. ``(num_of_images)``.

       ids : List of str
           Filenames loaded.

       X_val : 4D Numpy array, optional
           Validation images. E.g. ``(num_of_images, y, x, channels)``.

       Y_val : 1D Numpy array, optional
           Validation images' classes. E.g. ``(num_of_images)``.
    """

    print("### LOAD ###")
    if not test:
        path = cfg.DATA.TRAIN.PATH
    else:
        path = cfg.DATA.TEST.PATH

    if not test:
        if not cfg.DATA.VAL.CROSS_VAL:
            X_data_npy_file = os.path.join(path, '../prepared_npy', 'X_data.npy')
            Y_data_npy_file = os.path.join(path, '../prepared_npy', 'Y_data.npy')
            X_val_npy_file = os.path.join(path, '../prepared_npy', 'X_val.npy')
            Y_val_npy_file = os.path.join(path, '../prepared_npy', 'Y_val.npy')
        else:
            f_info = str(cfg.DATA.VAL.CROSS_VAL_FOLD)+'of'+str(cfg.DATA.VAL.CROSS_VAL_NFOLD)
            X_data_npy_file = os.path.join(path, '../prepared_npy', 'X_data'+f_info+'.npy')
            Y_data_npy_file = os.path.join(path, '../prepared_npy', 'Y_data'+f_info+'.npy')
            X_val_npy_file = os.path.join(path, '../prepared_npy', 'X_val'+f_info+'.npy')
            Y_val_npy_file = os.path.join(path, '../prepared_npy', 'Y_val'+f_info+'.npy')
    else:
        if not cfg.DATA.TEST.USE_VAL_AS_TEST:
            X_data_npy_file = os.path.join(path, '../prepared_npy', 'X_data.npy')
            Y_data_npy_file = os.path.join(path, '../prepared_npy', 'Y_data.npy')
        else:
            f_info = str(cfg.DATA.VAL.CROSS_VAL_FOLD)+'of'+str(cfg.DATA.VAL.CROSS_VAL_NFOLD)
            X_data_npy_file = os.path.join(path, '../prepared_npy', 'X_val'+f_info+'.npy')
            Y_data_npy_file = os.path.join(path, '../prepared_npy', 'Y_val'+f_info+'.npy')

    if not os.path.exists(X_data_npy_file):
        print("Seems to be the first run as no data is prepared. Creating .npy files: {}".format(X_data_npy_file))
        class_names = sorted(next(os.walk(path))[1])
        X_data, Y_data = [], []
        for c_num, folder in enumerate(class_names):
            print("Analizing folder {}".format(os.path.join(path,folder)))
            ids = sorted(next(os.walk(os.path.join(path,folder)))[2])
            print("Found {} samples".format(len(ids)))
            class_X_data, class_Y_data = [], []
            for i in tqdm(range(len(ids)), leave=False):
                img = imread(os.path.join(path, folder, ids[i]))
                if img.ndim == 2:
                    img = np.expand_dims(img, -1)
                else:
                    if img.shape[0] <= 3: img = img.transpose((1,2,0))
                img = np.expand_dims(img, 0).astype(np.uint8)

                # Ensure uint8
                if img.dtype == np.uint16:
                    if np.max(img) > 255:
                        img = normalize(img, 0, 65535)
                    else:
                        img = img.astype(np.uint8)

                class_X_data.append(img)
                class_Y_data.append(np.expand_dims(np.array(c_num),0).astype(np.uint8))

            class_X_data = np.concatenate(class_X_data, 0)
            class_Y_data = np.concatenate(class_Y_data, 0)
            X_data.append(class_X_data)
            Y_data.append(class_Y_data)

        # Fuse all data
        X_data = np.concatenate(X_data, 0)
        Y_data = np.concatenate(Y_data, 0)
        Y_data = np.squeeze(Y_data)

        os.makedirs(os.path.join(path, '../prepared_npy'), exist_ok=True)
        if not test:
            if cfg.DATA.VAL.CROSS_VAL:
                skf = StratifiedKFold(n_splits=cfg.DATA.VAL.CROSS_VAL_NFOLD, shuffle=cfg.DATA.VAL.RANDOM,
                    random_state=cfg.SYSTEM.SEED)
                f_num = 1
                for train_index, test_index in skf.split(X_data, Y_data):
                    if cfg.DATA.VAL.CROSS_VAL_FOLD == f_num:
                        X_data, X_val = X_data[train_index], X_data[test_index]
                        Y_data, Y_val = Y_data[train_index], Y_data[test_index]
                        break
                    f_num+= 1
            else:
                X_data, X_val, Y_data, Y_val = train_test_split(X_data, Y_data, test_size=cfg.DATA.VAL.SPLIT_TRAIN,
                    shuffle=cfg.DATA.VAL.RANDOM, random_state=cfg.SYSTEM.SEED)
            np.save(X_val_npy_file, X_val)
            np.save(Y_val_npy_file, Y_val)
        np.save(X_data_npy_file, X_data)
        np.save(Y_data_npy_file, Y_data)
    else:
        X_data = np.load(X_data_npy_file)
        Y_data = np.load(Y_data_npy_file)
        if not test:
            X_val = np.load(X_val_npy_file)
            Y_val = np.load(Y_val_npy_file)

    if not test:
        print("*** Loaded train data shape is: {}".format(X_data.shape))
        print("*** Loaded validation data shape is: {}".format(X_val.shape))
        print("### END LOAD ###")
        return X_data, Y_data, X_val, Y_val
    else:
        print("*** Loaded test data shape is: {}".format(X_data.shape))
        return X_data, Y_data, ids

