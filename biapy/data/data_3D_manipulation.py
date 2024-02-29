import math
import os
import h5py
import numpy as np
from skimage.io import imread
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold

from biapy.utils.util import load_3d_images_from_dir, order_dimensions, read_chunked_data

def load_and_prepare_3D_data(train_path, train_mask_path, cross_val=False, cross_val_nsplits=5, cross_val_fold=1, 
    val_split=0.1, seed=0, shuffle_val=True, crop_shape=(80, 80, 80, 1), y_upscaling=(1,1,1), random_crops_in_DA=False, 
    ov=(0,0,0), padding=(0,0,0), minimum_foreground_perc=-1, reflect_to_complete_shape=False, convert_to_rgb=False,
    preprocess_cfg=None, is_y_mask=False, preprocess_f=None):
    """
    Load train and validation images from the given paths to create 3D data.

    Parameters
    ----------
    train_path : str
        Path to the training data.

    train_mask_path : str
        Path to the training data masks.

    cross_val : bool, optional
        Whether to use cross validation or not. 

    cross_val_nsplits : int, optional
        Number of folds for the cross validation. 
    
    cross_val_fold : int, optional
        Number of the fold to be used as validation. 

    val_split : float, optional
        ``%`` of the train data used as validation (value between ``0`` and ``1``).

    seed : int, optional
        Seed value.

    shuffle_val : bool, optional
        Take random training examples to create validation data.

    crop_shape : 4D tuple
        Shape of the train subvolumes to create. E.g. ``(z, y, x, channels)``.

    y_upscaling : Tuple of 3 ints, optional
        Upscaling to be done when loading Y data. Use for super-resolution workflow.

    random_crops_in_DA : bool, optional
        To advice the method that not preparation of the data must be done, as random subvolumes will be created on
        DA, and the whole volume will be used for that.

    ov : Tuple of 3 floats, optional
        Amount of minimum overlap on x, y and z dimensions. The values must be on range ``[0, 1)``, that is, ``0%``
        or ``99%`` of overlap. E. g. ``(z, y, x)``.

    padding : Tuple of ints, optional
        Size of padding to be added on each axis ``(z, y, x)``. E.g. ``(24, 24, 24)``.

    minimum_foreground_perc : float, optional
        Minimum percetnage of foreground that a sample need to have no not be discarded. 

    reflect_to_complete_shape : bool, optional
        Wheter to increase the shape of the dimension that have less size than selected patch size padding it with
        'reflect'.

    self_supervised_args : dict, optional
        Arguments to create ground truth data for self-supervised workflow. 

    convert_to_rgb : bool, optional
        In case RGB images are expected, e.g. if ``crop_shape`` channel is 3, those images that are grayscale are 
        converted into RGB.

    preprocess_cfg : dict, optional
        Configuration parameters for preprocessing, is necessary in case you want to apply any preprocessing.

    is_y_mask : bool, optional
        Whether the data are masks. It is used to control the preprocessing of the data.
    
    preprocess_f : function, optional
        The preprocessing function, is necessary in case you want to apply any preprocessing.
        
    Returns
    -------
    X_train : 5D Numpy array
        Train images. E.g. ``(num_of_images, z, y, x, channels)``.

    Y_train : 5D Numpy array
        Train images' mask. E.g. ``(num_of_images, z, y, x, channels)``.

    X_val : 5D Numpy array, optional
        Validation images (``val_split > 0``). E.g. ``(num_of_images, z, y, x, channels)``.

    Y_val : 5D Numpy array, optional
        Validation images' mask (``val_split > 0``). E.g. ``(num_of_images, z, y, x, channels)``.

    filenames : List of str
        Loaded train filenames.

    Examples
    --------
    ::

        # EXAMPLE 1
        # Case where we need to load the data and creating a validation split
        train_path = "data/train/x"
        train_mask_path = "data/train/y"

        # Train data is (15, 91, 1024, 1024) where (number_of_images, z, y, x), so each image shape should be this:
        img_train_shape = (91, 1024, 1024, 1)
        # 3D subvolume shape needed
        train_3d_shape = (40, 256, 256, 1)

        X_train, Y_train, X_val,
        Y_val, filenames = load_and_prepare_3D_data_v2(train_path, train_mask_path, train_3d_shape,
                                                        val_split=0.1, shuffle_val=True, ov=(0,0,0))

        # The function will print the shapes of the generated arrays. In this example:
        #     *** Loaded train data shape is: (315, 40, 256, 256, 1)
        #     *** Loaded train mask shape is: (315, 40, 256, 256, 1)
        #     *** Loaded validation data shape is: (35, 40, 256, 256, 1)
        #     *** Loaded validation mask shape is: (35, 40, 256, 256, 1)
        #
    """

    print("### LOAD ###")

    # Disable crops when random_crops_in_DA is selected
    if random_crops_in_DA:
        crop = False  
    else:
        if cross_val:
            crop = False
            # Delay the crop to be made after cross validation
            delay_crop = True  
        else:
            crop = True
            delay_crop = False  

    # Check validation
    if val_split > 0 or cross_val:
        create_val = True  
    else:
        create_val = False

    print("0) Loading train images . . .")
    X_train, _, _, t_filenames = load_3d_images_from_dir(train_path, crop=crop, crop_shape=crop_shape,
        overlap=ov, padding=padding, return_filenames=True, reflect_to_complete_shape=reflect_to_complete_shape,
        convert_to_rgb=convert_to_rgb, preprocess_cfg=preprocess_cfg, is_mask=False, preprocess_f=preprocess_f)

    if train_mask_path is not None:
        print("1) Loading train GT . . .")
        scrop = (crop_shape[0]*y_upscaling[0], crop_shape[1]*y_upscaling[1], crop_shape[2]*y_upscaling[2], crop_shape[3])
        Y_train, _, _ = load_3d_images_from_dir(train_mask_path, crop=crop, crop_shape=scrop, overlap=ov,
            padding=padding, reflect_to_complete_shape=reflect_to_complete_shape, check_channel=False, check_drange=False,
            preprocess_cfg=preprocess_cfg, is_mask=is_y_mask, preprocess_f=preprocess_f)
    else:
        Y_train = None

    if isinstance(X_train, list):
        raise NotImplementedError("If you arrived here means that your images are not all of the same shape, and you "
                                  "select DATA.EXTRACT_RANDOM_PATCH = True, so no crops are made to ensure all images "
                                  "have the same shape. Please, crop them into your DATA.PATCH_SIZE and run again (you "
                                  "can use one of the script from here to crop: https://github.com/BiaPyX/BiaPy/tree/master/biapy/utils/scripts)")

    # Discard images that do not surpass the foreground percentage threshold imposed 
    if minimum_foreground_perc != -1 and Y_train is not None:
        print("Data that do not have {}% of foreground is discarded".format(minimum_foreground_perc))

        X_train_keep = []
        Y_train_keep = []
        are_lists = True if type(Y_train) is list else False

        samples_discarded = 0
        for i in tqdm(range(len(Y_train)), leave=False):
            labels, npixels = np.unique((Y_train[i]>0).astype(np.uint8), return_counts=True)
 
            discard = False
            if len(labels) == 1:
                discard = True
            else:
                total_pixels = 1
                for val in list(Y_train[i].shape):
                    total_pixels *= val
                    
                if (sum(npixels[1:]/total_pixels)) < minimum_foreground_perc:
                    discard = True

            if discard:
                samples_discarded += 1
            else:
                if are_lists:
                    X_train_keep.append(X_train[i])
                    Y_train_keep.append(Y_train[i])
                else:
                    X_train_keep.append(np.expand_dims(X_train[i],0))
                    Y_train_keep.append(np.expand_dims(Y_train[i],0))
        del X_train, Y_train
        
        if len(X_train_keep) == 0:
            raise ValueError("'TRAIN.MINIMUM_FOREGROUND_PER' value is too high, leading to the discarding of all training samples. Please, "
                "reduce its value.")

        if not are_lists:
            X_train_keep = np.concatenate(X_train_keep)
            Y_train_keep = np.concatenate(Y_train_keep)
        
        # Rename 
        X_train, Y_train = X_train_keep, Y_train_keep 
        del X_train_keep, Y_train_keep 

        print("{} samples discarded!".format(samples_discarded)) 
        if type(Y_train) is not list:      
            print("*** Remaining data shape is {}".format(X_train.shape))
            if X_train.shape[0] <= 1 and create_val: 
                raise ValueError("0 or 1 sample left to train, which is insufficent. "
                "Please, decrease the percentage to be more permissive")
        else:
            print("*** Remaining data shape is {}".format((len(X_train),)+X_train[0].shape[1:]))
            if len(X_train) <= 1 and create_val:
                raise ValueError("0 or 1 sample left to train, which is insufficent. "
                "Please, decrease the percentage to be more permissive")

    if Y_train is not None and len(X_train) != len(Y_train):
        raise ValueError("Different number of raw and ground truth items ({} vs {}). "
            "Please check the data!".format(len(X_train), len(Y_train)))
            
    # Create validation data splitting the train
    if create_val:
        print("Creating validation data")
        Y_val = None
        if not cross_val:
            if Y_train is not None:
                X_train, X_val, Y_train, Y_val = train_test_split(
                    X_train, Y_train, test_size=val_split, shuffle=shuffle_val, random_state=seed)
            else:
                X_train, X_val = train_test_split(
                    X_train, test_size=val_split, shuffle=shuffle_val, random_state=seed)
        else:
            skf = StratifiedKFold(n_splits=cross_val_nsplits, shuffle=shuffle_val,
                random_state=seed)
            fold = 1
            train_index, test_index = None, None

            y_len = len(Y_train) if Y_train is not None else len(X_train)
            for t_index, te_index in skf.split(np.zeros(len(X_train)), np.zeros(y_len)):
                if cross_val_fold == fold:
                    X_train, X_val = X_train[t_index], X_train[te_index]
                    if Y_train is not None:
                        Y_train, Y_val = Y_train[t_index], Y_train[te_index]
                    train_index, test_index = t_index.copy(), te_index.copy()
                    break
                fold+= 1

            if len(test_index) > 5:
                print("Fold number {}. Printing the first 5 ids: {}".format(fold, test_index[:5]))
            else:
                print("Fold number {}. Indexes used in cross validation: {}".format(fold, test_index))

            # Then crop after cross validation
            if delay_crop:
                # X_train
                data = []
                for img_num in range(len(X_train)):
                    if X_train[img_num].shape != crop_shape[:3]+(X_train[img_num].shape[-1],):
                        img = X_train[img_num]
                        img = crop_3D_data_with_overlap(X_train[img_num][0] if isinstance(X_train, list) else X_train[img_num], 
                            crop_shape[:3]+(X_train[img_num].shape[-1],), overlap=ov, padding=padding, verbose=False)
                    data.append(img)
                X_train = np.concatenate(data)
                del data

                # Y_train
                if Y_train is not None:
                    data_mask = []
                    scrop = (crop_shape[0], crop_shape[1]*y_upscaling[0], crop_shape[2]*y_upscaling[1], crop_shape[3]*y_upscaling[2])
                    for img_num in range(len(Y_train)):
                        if Y_train[img_num].shape != scrop[:3]+(Y_train[img_num].shape[-1],):
                            img = Y_train[img_num]
                            img = crop_3D_data_with_overlap(Y_train[img_num][0] if isinstance(Y_train, list) else Y_train[img_num],
                                scrop[:3]+(Y_train[img_num].shape[-1],), overlap=ov, padding=padding, verbose=False)
                        data_mask.append(img)
                    Y_train = np.concatenate(data_mask)
                    del data_mask
                    
                # X_val
                data = []
                for img_num in range(len(X_val)):
                    if X_val[img_num].shape != crop_shape[:3]+(X_val[img_num].shape[-1],):
                        img = X_val[img_num]
                        img = crop_3D_data_with_overlap(X_val[img_num][0] if isinstance(X_val, list) else X_val[img_num], 
                            crop_shape[:3]+(X_val[img_num].shape[-1],), overlap=ov, padding=padding, verbose=False)
                    data.append(img)
                X_val = np.concatenate(data)
                del data

                # Y_val
                if Y_val is not None:
                    data_mask = []
                    scrop = (crop_shape[0], crop_shape[1]*y_upscaling[0], crop_shape[2]*y_upscaling[1], crop_shape[3]*y_upscaling[2])
                    for img_num in range(len(Y_val)):
                        if Y_val[img_num].shape != scrop[:3]+(Y_val[img_num].shape[-1],):
                            img = Y_val[img_num]
                            img = crop_3D_data_with_overlap(Y_val[img_num][0] if isinstance(Y_val, list) else Y_val[img_num],
                                scrop[:3]+(Y_val[img_num].shape[-1],), overlap=ov, padding=padding, verbose=False)
                        data_mask.append(img)
                    Y_val = np.concatenate(data_mask)
                    del data_mask

    # Convert the original volumes as they were a unique subvolume
    if random_crops_in_DA and X_train.ndim == 4:
        X_train = np.expand_dims(X_train, axis=0)
        if Y_train is not None:
            Y_train = np.expand_dims(Y_train, axis=0)
        if create_val:
            X_val = np.expand_dims(X_val, axis=0)
            if Y_val is not None:
                Y_val = np.expand_dims(Y_val, axis=0)

    if create_val:
        print("*** Loaded train data shape is: {}".format(X_train.shape))
        if Y_train is not None:
            print("*** Loaded train GT shape is: {}".format(Y_train.shape))
        print("*** Loaded validation data shape is: {}".format(X_val.shape))
        if Y_val is not None:
            print("*** Loaded validation GT shape is: {}".format(Y_val.shape))
        if not cross_val:
            return X_train, Y_train, X_val, Y_val, t_filenames
        else:
            return X_train, Y_train, X_val, Y_val, t_filenames, test_index
    else:
        print("*** Loaded train data shape is: {}".format(X_train.shape))
        if Y_train is not None:
            print("*** Loaded train GT shape is: {}".format(Y_train.shape))
        return X_train, Y_train, t_filenames


def load_and_prepare_3D_efficient_format_data(train_path, train_mask_path, input_img_axes, input_mask_axes=None, cross_val=False, 
    cross_val_nsplits=5, cross_val_fold=1, val_split=0.1, seed=0, shuffle_val=True, crop_shape=(80, 80, 80, 1), y_upscaling=(1,1,1), 
    ov=(0,0,0), padding=(0,0,0), minimum_foreground_perc=-1):
    """
    Load train and validation images from the given paths to create 3D data.

    Parameters
    ----------
    train_path : str
        Path to the training data.

    train_mask_path : str
        Path to the training data masks.

    input_img_axes : str
        Order of axes of the data in ``train_path``. One between ['TZCYX', 'TZYXC', 'ZCYX', 'ZYXC'].

    input_mask_axes : str, optional
        Order of axes of the data in ``train_mask_path``. One between ['TZCYX', 'TZYXC', 'ZCYX', 'ZYXC'].

    cross_val : bool, optional
        Whether to use cross validation or not. 

    cross_val_nsplits : int, optional
        Number of folds for the cross validation. 
    
    cross_val_fold : int, optional
        Number of the fold to be used as validation. 

    val_split : float, optional
        ``%`` of the train data used as validation (value between ``0`` and ``1``).

    seed : int, optional
        Seed value.

    shuffle_val : bool, optional
        Take random training examples to create validation data.

    crop_shape : 4D tuple
        Shape of the train subvolumes to create. E.g. ``(z, y, x, channels)``.

    y_upscaling : Tuple of 3 ints, optional
        Upscaling to be done when loading Y data. Use for super-resolution workflow.

    ov : Tuple of 3 floats, optional
        Amount of minimum overlap on x, y and z dimensions. The values must be on range ``[0, 1)``, that is, ``0%``
        or ``99%`` of overlap. E. g. ``(z, y, x)``.

    padding : Tuple of ints, optional
        Size of padding to be added on each axis ``(z, y, x)``. E.g. ``(24, 24, 24)``.

    minimum_foreground_perc : float, optional
        Minimum percetnage of foreground that a sample need to have no not be discarded. 

    Returns
    -------
    X_train : 5D Numpy array
        Train images. E.g. ``(num_of_images, z, y, x, channels)``.

    Y_train : 5D Numpy array
        Train images' mask. E.g. ``(num_of_images, z, y, x, channels)``.

    X_val : 5D Numpy array, optional
        Validation images (``val_split > 0``). E.g. ``(num_of_images, z, y, x, channels)``.

    Y_val : 5D Numpy array, optional
        Validation images' mask (``val_split > 0``). E.g. ``(num_of_images, z, y, x, channels)``.
    """

    print("### LOAD DATA INFO ###")

    # Check validation
    if val_split > 0 or cross_val:
        create_val = True  
    else:
        create_val = False

    print("0) Loading train image information . . .")
    X_train, X_train_total_patches = load_3D_efficient_files(train_path, input_img_axes, crop_shape, ov, padding)

    if train_mask_path is not None:
        if input_mask_axes is None:
            raise ValueError("input_mask_axes need to be provided")

        print("0) Loading train GT information . . .")
        scrop = (crop_shape[0]*y_upscaling[0], crop_shape[1]*y_upscaling[1], crop_shape[2]*y_upscaling[2], crop_shape[3])
        Y_train, Y_train_total_patches = load_3D_efficient_files(train_mask_path, input_mask_axes, scrop, ov, padding, 
            check_channel=False)

        for i in range(len(Y_train_total_patches)):
            if Y_train_total_patches[i] != X_train_total_patches[i]:
                raise ValueError(f"Seems that the image {X_train[i]['filepath']} and its mask pair {Y_train[i]['filepath']} have "
                    f"different data, as they led to different total amount of patches ({Y_train_total_patches[i]} vs {X_train_total_patches[i]})")
                    
    # Discard images that do not surpass the foreground percentage threshold imposed 
    if minimum_foreground_perc != -1 and Y_train is not None:
        print("Data that do not have {}% of foreground is discarded".format(minimum_foreground_perc))

        X_train_remove = []
        samples_discarded = 0
        last_data_file = {}

        for i in tqdm(range(len(Y_train)), leave=False):
            data_info = Y_train[i]

            if 'filepath' not in last_data_file or last_data_file['filepath'] != data_info['filepath']:
                if 'filepath' in last_data_file and isinstance(file, h5py.File):
                    file.close()
                file, data = read_chunked_data(data_info['filepath'])
                last_data_file = data_info.copy()

            # Prepare slices to extract the patch
            slices = []
            for j in range(len(data_info['patch_coords'])):
                if isinstance(data_info['patch_coords'][j], int):
                    # +1 to prevent 0 length axes that can not be removed with np.squeeze later
                    slices.append(slice(data_info['patch_coords'][j]+1)) 
                else:
                    slices.append(slice(data_info['patch_coords'][j][0],data_info['patch_coords'][j][1]))

            img = np.array(data[tuple(slices)])
            labels, npixels = np.unique((img>0).astype(np.uint8), return_counts=True)

            discard = False
            if len(labels) == 1:
                discard = True
            else:
                total_pixels = 1
                for val in list(img.shape):
                    total_pixels *= val

                if (sum(npixels[1:]/total_pixels)) < minimum_foreground_perc:
                    discard = True

            if discard:
                samples_discarded += 1
                X_train_remove.append(i)
        
        if len(Y_train)-len(X_train_remove) <= 1:
            raise ValueError("'TRAIN.MINIMUM_FOREGROUND_PER' value is too high, leading to the discarding of all training samples. Please, "
                "reduce its value.")

        # Remove samples 
        for i in range(len(X_train_remove)):
            del X_train[i], Y_train[i]

        # Rearrange ids 
        X_train = {c:v[1] for c,v in enumerate(X_train.items())}
        Y_train = {c:v[1] for c,v in enumerate(Y_train.items())}

        print("{} samples discarded!".format(samples_discarded)) 
        print("*** Remaining data samples: {}".format(len(X_train)))   
   
    if Y_train is not None and len(X_train) != len(Y_train):
        raise ValueError("Different number of raw and ground truth items ({} vs {}). "
            "Please check the data!".format(len(X_train), len(Y_train)))
    
    # Create validation data splitting the train
    if create_val:
        print("Creating validation data")
        Y_val = None
        if not cross_val:
            if Y_train is not None:
                X_train, X_val, Y_train, Y_val = train_test_split(
                    X_train, Y_train, test_size=val_split, shuffle=shuffle_val, random_state=seed)
            else:
                X_train, X_val = train_test_split(
                    X_train, test_size=val_split, shuffle=shuffle_val, random_state=seed)
        else:
            skf = StratifiedKFold(n_splits=cross_val_nsplits, shuffle=shuffle_val,
                random_state=seed)
            fold = 1
            train_index, test_index = None, None

            y_len = len(Y_train) if Y_train is not None else len(X_train)
            for t_index, te_index in skf.split(np.zeros(len(X_train)), np.zeros(y_len)):
                if cross_val_fold == fold:
                    X_train, X_val = X_train[t_index], X_train[te_index]
                    if Y_train is not None:
                        Y_train, Y_val = Y_train[t_index], Y_train[te_index]
                    train_index, test_index = t_index.copy(), te_index.copy()
                    break
                fold+= 1

            if len(test_index) > 5:
                print("Fold number {}. Printing the first 5 ids: {}".format(fold, test_index[:5]))
            else:
                print("Fold number {}. Indexes used in cross validation: {}".format(fold, test_index))

    patch_coords = order_dimensions(X_train[0]['patch_coords'], input_order=input_img_axes, output_order="ZYX", default_value=0)

    shape = (
        len(X_train), 
        patch_coords[0][1]-patch_coords[0][0],
        patch_coords[1][1]-patch_coords[1][0],
        patch_coords[2][1]-patch_coords[2][0],
    )
    if Y_train is not None:
        patch_coords = order_dimensions(Y_train[0]['patch_coords'], input_order=input_mask_axes, output_order="ZYX", default_value=0)
        yshape = (
            len(Y_train), 
            patch_coords[0][1]-patch_coords[0][0],
            patch_coords[1][1]-patch_coords[1][0],
            patch_coords[2][1]-patch_coords[2][0],
        )
    if create_val:
        print("*** Loaded train data shape is: {}".format(shape))
        if Y_train is not None:
            print("*** Loaded train GT shape is: {}".format(yshape))
        patch_coords = order_dimensions(X_val[0]['patch_coords'], input_order=input_img_axes, output_order="ZYX", default_value=0)
        shape = (
            len(X_val), 
            patch_coords[0][1]-patch_coords[0][0],
            patch_coords[1][1]-patch_coords[1][0],
            patch_coords[2][1]-patch_coords[2][0],
        )
        print("*** Loaded validation data shape is: {}".format(shape))
        if Y_val is not None:
            patch_coords = order_dimensions(Y_val[0]['patch_coords'], input_order=input_mask_axes, output_order="ZYX", default_value=0)
            shape = (
                len(Y_val), 
                patch_coords[0][1]-patch_coords[0][0],
                patch_coords[1][1]-patch_coords[1][0],
                patch_coords[2][1]-patch_coords[2][0],
            )
            print("*** Loaded validation GT shape is: {}".format(shape))
        if not cross_val:
            return X_train, Y_train, X_val, Y_val
        else:
            return X_train, Y_train, X_val, Y_val, test_index
    else:
        print("*** Loaded train data shape is: {}".format(shape))
        if Y_train is not None:
            print("*** Loaded train GT shape is: {}".format(yshape))
        return X_train, Y_train

def load_3D_efficient_files(data_path, input_axes, crop_shape, overlap, padding, check_channel=True):
    """
    Load information of all patches that can be extracted from all the Zarr/H5 samples in ``data_path``.

    Parameters
    ----------
    data_path : str
        Path to the training data.

    input_axes : str
        Order of axes of the data in ``data_path``. One between ['TZCYX', 'TZYXC', 'ZCYX', 'ZYXC'].

    crop_shape : 4D tuple
        Shape of the train subvolumes to create. E.g. ``(z, y, x, channels)``.

    overlap : Tuple of 3 floats, optional
        Amount of minimum overlap on x, y and z dimensions. The values must be on range ``[0, 1)``, that is, ``0%``
        or ``99%`` of overlap. E. g. ``(z, y, x)``.

    padding : Tuple of ints, optional
        Size of padding to be added on each axis ``(z, y, x)``. E.g. ``(24, 24, 24)``.

    check_channel : bool, optional
        Whether to check if the crop_shape channel matches with the loaded images' one. 
        
    Returns
    -------
    data_info : dict
        All patch info that can be extracted from all the Zarr/H5 samples in ``data_path``.

    data_info_total_patches : List of ints
        Amount of patches extracted from each sample in ``data_path``.
    """
    data_info = {}
    data_total_patches = []
    c = 0
    for i, filename in enumerate(data_path):
        file, data = read_chunked_data(filename)

        # Get the total patches so we can use tqdm so the user can see the time
        obj = extract_3D_patch_with_overlap_yield(data, crop_shape, input_axes, overlap=overlap, padding=padding, 
            total_ranks=1, rank=0, return_only_stats=True, verbose=True)
        __unnamed_iterator = iter(obj)
        while True:
            try:
                obj = next(__unnamed_iterator)
            except StopIteration:  # StopIteration caught here without inspecting it
                break 
        del __unnamed_iterator                          
        total_patches, z_vol_info, list_of_vols_in_z = obj

        for obj in tqdm(extract_3D_patch_with_overlap_yield(data, crop_shape, input_axes, overlap=overlap, 
            padding=padding, total_ranks=1, rank=0, verbose=False), total=total_patches):
            
            img, patch_coords, _, _, _ = obj
            
            data_info[c] = {}
            data_info[c]['filepath'] = filename
            data_info[c]['patch_coords'] = order_dimensions(patch_coords, input_order="ZYX", output_order=input_axes, default_value=0)

            c += 1 

            if check_channel and crop_shape[-1] != img.shape[-1]:
                raise ValueError("Channel of the patch size given {} does not correspond with the loaded image {}. "
                    "Please, check the channels of the images!".format(crop_shape[-1], img.shape[-1]))

        if isinstance(file, h5py.File):
            file.close()
            
        data_total_patches.append(total_patches)
    
    return data_info, data_total_patches

def crop_3D_data_with_overlap(data, vol_shape, data_mask=None, overlap=(0,0,0), padding=(0,0,0), verbose=True,
    median_padding=False):
    """Crop 3D data into smaller volumes with a defined overlap. The opposite function is :func:`~merge_3D_data_with_overlap`.

       Parameters
       ----------
       data : 4D Numpy array
           Data to crop. E.g. ``(z, y, x, channels)``.

       vol_shape : 4D int tuple
           Shape of the volumes to create. E.g. ``(z, y, x, channels)``.

       data_mask : 4D Numpy array, optional
            Data mask to crop. E.g. ``(z, y, x, channels)``.

       overlap : Tuple of 3 floats, optional
            Amount of minimum overlap on x, y and z dimensions. The values must be on range ``[0, 1)``, that is, ``0%``
            or ``99%`` of overlap. E.g. ``(z, y, x)``.

       padding : tuple of ints, optional
           Size of padding to be added on each axis ``(z, y, x)``. E.g. ``(24, 24, 24)``.

       verbose : bool, optional
            To print information about the crop to be made.

       median_padding : bool, optional
           If ``True`` the padding value is the median value. If ``False``, the added values are zeroes.

       Returns
       -------
       cropped_data : 5D Numpy array
           Cropped image data. E.g. ``(vol_number, z, y, x, channels)``.

       cropped_data_mask : 5D Numpy array, optional
           Cropped image data masks. E.g. ``(vol_number, z, y, x, channels)``.

       Examples
       --------
       ::

           # EXAMPLE 1
           # Following the example introduced in load_and_prepare_3D_data function, the cropping of a volume with shape
           # (165, 1024, 765) should be done by the following call:
           X_train = np.ones((165, 768, 1024, 1))
           Y_train = np.ones((165, 768, 1024, 1))
           X_train, Y_train = crop_3D_data_with_overlap(X_train, (80, 80, 80, 1), data_mask=Y_train,
                                                        overlap=(0.5,0.5,0.5))
           # The function will print the shape of the generated arrays. In this example:
           #     **** New data shape is: (2600, 80, 80, 80, 1)

       A visual explanation of the process:

       .. image:: ../../img/crop_3D_ov.png
           :width: 80%
           :align: center

       Note: this image do not respect the proportions.
       ::

           # EXAMPLE 2
           # Same data crop but without overlap

           X_train, Y_train = crop_3D_data_with_overlap(X_train, (80, 80, 80, 1), data_mask=Y_train, overlap=(0,0,0))

           # The function will print the shape of the generated arrays. In this example:
           #     **** New data shape is: (390, 80, 80, 80, 1)
           #
           # Notice how differs the amount of subvolumes created compared to the first example

           #EXAMPLE 2
           #In the same way, if the addition of (64,64,64) padding is required, the call should be done as shown:
           X_train, Y_train = crop_3D_data_with_overlap(
                X_train, (80, 80, 80, 1), data_mask=Y_train, overlap=(0.5,0.5,0.5), padding=(64,64,64))
    """

    if verbose:
        print("### 3D-OV-CROP ###")
        print("Cropping {} images into {} with overlapping . . .".format(data.shape, vol_shape))
        print("Minimum overlap selected: {}".format(overlap))
        print("Padding: {}".format(padding))

    if data.ndim != 4:
        raise ValueError("data expected to be 4 dimensional, given {}".format(data.shape))
    if data_mask is not None:
        if data_mask.ndim != 4:
            raise ValueError("data_mask expected to be 4 dimensional, given {}".format(data_mask.shape))
        if data.shape[:-1] != data_mask.shape[:-1]:
            raise ValueError("data and data_mask shapes mismatch: {} vs {}".format(data.shape[:-1], data_mask.shape[:-1]))
    if len(vol_shape) != 4:
        raise ValueError("vol_shape expected to be of length 4, given {}".format(vol_shape))
    if vol_shape[0] > data.shape[0]:
        raise ValueError("'vol_shape[0]' {} greater than {} (you can reduce 'DATA.PATCH_SIZE' or use 'DATA.REFLECT_TO_COMPLETE_SHAPE')"
            .format(vol_shape[0], data.shape[0]))
    if vol_shape[1] > data.shape[1]:
        raise ValueError("'vol_shape[1]' {} greater than {} (you can reduce 'DATA.PATCH_SIZE' or use 'DATA.REFLECT_TO_COMPLETE_SHAPE')"
            .format(vol_shape[1], data.shape[1]))
    if vol_shape[2] > data.shape[2]:
        raise ValueError("'vol_shape[2]' {} greater than {} (you can reduce 'DATA.PATCH_SIZE' or use 'DATA.REFLECT_TO_COMPLETE_SHAPE')"
            .format(vol_shape[2], data.shape[2]))
    if (overlap[0] >= 1 or overlap[0] < 0) or (overlap[1] >= 1 or overlap[1] < 0) or (overlap[2] >= 1 or overlap[2] < 0):
        raise ValueError("'overlap' values must be floats between range [0, 1)")
    for i,p in enumerate(padding):
        if p >= vol_shape[i]//2:
            raise ValueError("'Padding' can not be greater than the half of 'vol_shape'. Max value for this {} input shape is {}"
                             .format(data.shape, [(vol_shape[0]//2)-1,(vol_shape[1]//2)-1,(vol_shape[2]//2)-1]))

    padded_data = np.pad(data,((padding[0],padding[0]),(padding[1],padding[1]),(padding[2],padding[2]),(0,0)), 'reflect')
    if data_mask is not None:
        padded_data_mask = np.pad(data_mask,((padding[0],padding[0]),(padding[1],padding[1]),(padding[2],padding[2]),(0,0)), 'reflect')
    if median_padding:
    	padded_data[0:padding[0], :, :, :] = np.median(data[0, :, :, :])
    	padded_data[padding[0]+data.shape[0]:2*padding[0]+data.shape[0], :, :, :] = np.median(data[-1, :, :, :])
    	padded_data[:, 0:padding[1], :, :] = np.median(data[:, 0, :, :])
    	padded_data[:, padding[1]+data.shape[1]:2*padding[1]+data.shape[0], :, :] = np.median(data[:, -1, :, :])
    	padded_data[:, :, 0:padding[2], :] = np.median(data[:, :, 0, :])
    	padded_data[ :, :, padding[2]+data.shape[2]:2*padding[2]+data.shape[2], :] = np.median(data[:, :, -1, :])
    padded_vol_shape = vol_shape

    # Calculate overlapping variables
    overlap_z = 1 if overlap[0] == 0 else 1-overlap[0]
    overlap_y = 1 if overlap[1] == 0 else 1-overlap[1]
    overlap_x = 1 if overlap[2] == 0 else 1-overlap[2]

    # Z
    step_z = int((vol_shape[0]-padding[0]*2)*overlap_z)
    vols_per_z = math.ceil(data.shape[0]/step_z)
    last_z = 0 if vols_per_z == 1 else (((vols_per_z-1)*step_z)+vol_shape[0])-padded_data.shape[0]
    ovz_per_block = last_z//(vols_per_z-1) if vols_per_z > 1 else 0
    step_z -= ovz_per_block
    last_z -= ovz_per_block*(vols_per_z-1)

    # Y
    step_y = int((vol_shape[1]-padding[1]*2)*overlap_y)
    vols_per_y = math.ceil(data.shape[1]/step_y)
    last_y = 0 if vols_per_y == 1 else (((vols_per_y-1)*step_y)+vol_shape[1])-padded_data.shape[1]
    ovy_per_block = last_y//(vols_per_y-1) if vols_per_y > 1 else 0
    step_y -= ovy_per_block
    last_y -= ovy_per_block*(vols_per_y-1)

    # X
    step_x = int((vol_shape[2]-padding[2]*2)*overlap_x)
    vols_per_x = math.ceil(data.shape[2]/step_x)
    last_x = 0 if vols_per_x == 1 else (((vols_per_x-1)*step_x)+vol_shape[2])-padded_data.shape[2]
    ovx_per_block = last_x//(vols_per_x-1) if vols_per_x > 1 else 0
    step_x -= ovx_per_block
    last_x -= ovx_per_block*(vols_per_x-1)

    # Real overlap calculation for printing
    real_ov_z = ovz_per_block/(vol_shape[0]-padding[0]*2)
    real_ov_y = ovy_per_block/(vol_shape[1]-padding[1]*2)
    real_ov_x = ovx_per_block/(vol_shape[2]-padding[2]*2)
    if verbose:
        print("Real overlapping (%): {}".format((real_ov_z,real_ov_y,real_ov_x)))
        print("Real overlapping (pixels): {}".format(((vol_shape[0]-padding[0]*2)*real_ov_z,
              (vol_shape[1]-padding[1]*2)*real_ov_y,(vol_shape[2]-padding[2]*2)*real_ov_x)))
        print("{} patches per (z,y,x) axis".format((vols_per_z,vols_per_x,vols_per_y)))
    
    total_vol = vols_per_z*vols_per_y*vols_per_x
    cropped_data = np.zeros((total_vol,) + padded_vol_shape, dtype=data.dtype)
    if data_mask is not None:
        cropped_data_mask = np.zeros((total_vol,) + padded_vol_shape[:3]+(data_mask.shape[-1],), dtype=data_mask.dtype)

    c = 0
    for z in range(vols_per_z):
        for y in range(vols_per_y):
            for x in range(vols_per_x):
                d_z = 0 if (z*step_z+vol_shape[0]) < padded_data.shape[0] else last_z
                d_y = 0 if (y*step_y+vol_shape[1]) < padded_data.shape[1] else last_y
                d_x = 0 if (x*step_x+vol_shape[2]) < padded_data.shape[2] else last_x

                cropped_data[c] = padded_data[z*step_z-d_z:z*step_z+vol_shape[0]-d_z,
                                              y*step_y-d_y:y*step_y+vol_shape[1]-d_y,
                                              x*step_x-d_x:x*step_x+vol_shape[2]-d_x]
                if data_mask is not None:
                    cropped_data_mask[c] = padded_data_mask[z*step_z-d_z:(z*step_z)+vol_shape[0]-d_z,
                                                            y*step_y-d_y:y*step_y+vol_shape[1]-d_y,
                                                            x*step_x-d_x:x*step_x+vol_shape[2]-d_x]
                c += 1

    if verbose:
        print("**** New data shape is: {}".format(cropped_data.shape))
        print("### END 3D-OV-CROP ###")

    if data_mask is not None:
        return cropped_data, cropped_data_mask
    else:
        return cropped_data


def merge_3D_data_with_overlap(data, orig_vol_shape, data_mask=None, overlap=(0,0,0), padding=(0,0,0), verbose=True):
    """Merge 3D subvolumes in a 3D volume with a defined overlap.

       The opposite function is :func:`~crop_3D_data_with_overlap`.

       Parameters
       ----------
       data : 5D Numpy array
           Data to crop. E.g. ``(volume_number, z, y, x, channels)``.

       orig_vol_shape : 4D int tuple
           Shape of the volumes to create.

       data_mask : 4D Numpy array, optional
           Data mask to crop. E.g. ``(volume_number, z, y, x, channels)``.

       overlap : Tuple of 3 floats, optional
            Amount of minimum overlap on x, y and z dimensions. Should be the same as used in
            :func:`~crop_3D_data_with_overlap`. The values must be on range ``[0, 1)``, that is, ``0%`` or ``99%`` of
            overlap. E.g. ``(z, y, x)``.

       padding : tuple of ints, optional
           Size of padding to be added on each axis ``(z, y, x)``. E.g. ``(24, 24, 24)``.

       verbose : bool, optional
            To print information about the crop to be made.

       Returns
       -------
       merged_data : 4D Numpy array
           Cropped image data. E.g. ``(z, y, x, channels)``.

       merged_data_mask : 5D Numpy array, optional
           Cropped image data masks. E.g. ``(z, y, x, channels)``.

       Examples
       --------
       ::

           # EXAMPLE 1
           # Following the example introduced in crop_3D_data_with_overlap function, the merge after the cropping
           # should be done as follows:

           X_train = np.ones((165, 768, 1024, 1))
           Y_train = np.ones((165, 768, 1024, 1))

           X_train, Y_train = crop_3D_data_with_overlap(X_train, (80, 80, 80, 1), data_mask=Y_train, overlap=(0.5,0.5,0.5))
           X_train, Y_train = merge_3D_data_with_overlap(X_train, (165, 768, 1024, 1), data_mask=Y_train, overlap=(0.5,0.5,0.5))

           # The function will print the shape of the generated arrays. In this example:
           #     **** New data shape is: (165, 768, 1024, 1)

           # EXAMPLE 2
           # In the same way, if no overlap in cropping was selected, the merge call
           # should be as follows:

           X_train, Y_train = merge_3D_data_with_overlap(X_train, (165, 768, 1024, 1), data_mask=Y_train, overlap=(0,0,0))

           # The function will print the shape of the generated arrays. In this example:
           #     **** New data shape is: (165, 768, 1024, 1)

           # EXAMPLE 3
           # On the contrary, if no overlap in cropping was selected but a padding of shape
           # (64,64,64) is needed, the merge call should be as follows:

           X_train, Y_train = merge_3D_data_with_overlap(X_train, (165, 768, 1024, 1), data_mask=Y_train, overlap=(0,0,0),
               padding=(64,64,64))

           # The function will print the shape of the generated arrays. In this example:
           #     **** New data shape is: (165, 768, 1024, 1)
    """
    if data_mask is not None:
        if data.shape[:-1] != data_mask.shape[:-1]:
            raise ValueError("data and data_mask shapes mismatch: {} vs {}".format(data.shape[:-1], data_mask.shape[:-1]))

    if (overlap[0] >= 1 or overlap[0] < 0) or (overlap[1] >= 1 or overlap[1] < 0) or (overlap[2] >= 1 or overlap[2] < 0):
        raise ValueError("'overlap' values must be floats between range [0, 1)")

    if verbose:
        print("### MERGE-3D-OV-CROP ###")
        print("Merging {} images into {} with overlapping . . .".format(data.shape, orig_vol_shape))
        print("Minimum overlap selected: {}".format(overlap))
        print("Padding: {}".format(padding))

    # Remove the padding
    pad_input_shape = data.shape
    data = data[:, padding[0]:data.shape[1]-padding[0],
                padding[1]:data.shape[2]-padding[1],
                padding[2]:data.shape[3]-padding[2], :]

    merged_data = np.zeros((orig_vol_shape), dtype=np.float32)
    if data_mask is not None:
        data_mask = data_mask[:, padding[0]:data_mask.shape[1]-padding[0],
                              padding[1]:data_mask.shape[2]-padding[1],
                              padding[2]:data_mask.shape[3]-padding[2], :]
        merged_data_mask = np.zeros(orig_vol_shape[:3]+(data_mask.shape[-1],), dtype=np.float32)
    ov_map_counter = np.zeros((orig_vol_shape[:-1]+(1,)), dtype=np.uint16)

    # Calculate overlapping variables
    overlap_z = 1 if overlap[0] == 0 else 1-overlap[0]
    overlap_y = 1 if overlap[1] == 0 else 1-overlap[1]
    overlap_x = 1 if overlap[2] == 0 else 1-overlap[2]
    
    padded_vol_shape = [orig_vol_shape[0]+2*padding[0], orig_vol_shape[1]+2*padding[1], orig_vol_shape[2]+2*padding[2]]

    # Z
    step_z = int((pad_input_shape[1]-padding[0]*2)*overlap_z)
    vols_per_z = math.ceil(orig_vol_shape[0]/step_z)
    last_z = 0 if vols_per_z == 1 else (((vols_per_z-1)*step_z)+pad_input_shape[1])-padded_vol_shape[0]
    ovz_per_block = last_z//(vols_per_z-1) if vols_per_z > 1 else 0
    step_z -= ovz_per_block
    last_z -= ovz_per_block*(vols_per_z-1)

    # Y
    step_y = int((pad_input_shape[2]-padding[1]*2)*overlap_y)
    vols_per_y = math.ceil(orig_vol_shape[1]/step_y)
    last_y = 0 if vols_per_y == 1 else (((vols_per_y-1)*step_y)+pad_input_shape[2])-padded_vol_shape[1]
    ovy_per_block = last_y//(vols_per_y-1) if vols_per_y > 1 else 0
    step_y -= ovy_per_block
    last_y -= ovy_per_block*(vols_per_y-1)

    # X
    step_x = int((pad_input_shape[3]-padding[2]*2)*overlap_x)
    vols_per_x = math.ceil(orig_vol_shape[2]/step_x)
    last_x = 0 if vols_per_x == 1 else (((vols_per_x-1)*step_x)+pad_input_shape[3])-padded_vol_shape[2]
    ovx_per_block = last_x//(vols_per_x-1) if vols_per_x > 1 else 0
    step_x -= ovx_per_block
    last_x -= ovx_per_block*(vols_per_x-1)

    # Real overlap calculation for printing
    real_ov_z = ovz_per_block/(pad_input_shape[1]-padding[0]*2) 
    real_ov_y = ovy_per_block/(pad_input_shape[2]-padding[1]*2) 
    real_ov_x = ovx_per_block/(pad_input_shape[3]-padding[2]*2) 

    if verbose:
        print("Real overlapping (%): {}".format((real_ov_z,real_ov_y,real_ov_x)))
        print("Real overlapping (pixels): {}".format(((pad_input_shape[1]-padding[0]*2)*real_ov_z,
            (pad_input_shape[2]-padding[1]*2)*real_ov_y,(pad_input_shape[3]-padding[2]*2)*real_ov_x)))
        print("{} patches per (z,y,x) axis".format((vols_per_z,vols_per_x,vols_per_y)))

    c = 0
    for z in range(vols_per_z):
        for y in range(vols_per_y):
            for x in range(vols_per_x):
                d_z = 0 if (z*step_z+data.shape[1]) < orig_vol_shape[0] else last_z
                d_y = 0 if (y*step_y+data.shape[2]) < orig_vol_shape[1] else last_y
                d_x = 0 if (x*step_x+data.shape[3]) < orig_vol_shape[2] else last_x

                merged_data[z*step_z-d_z:(z*step_z)+data.shape[1]-d_z,
                            y*step_y-d_y:y*step_y+data.shape[2]-d_y,
                            x*step_x-d_x:x*step_x+data.shape[3]-d_x] += data[c]

                if data_mask is not None:
                    merged_data_mask[z*step_z-d_z:(z*step_z)+data.shape[1]-d_z,
                                     y*step_y-d_y:y*step_y+data.shape[2]-d_y,
                                     x*step_x-d_x:x*step_x+data.shape[3]-d_x] += data_mask[c]

                ov_map_counter[z*step_z-d_z:(z*step_z)+data.shape[1]-d_z,
                               y*step_y-d_y:y*step_y+data.shape[2]-d_y,
                               x*step_x-d_x:x*step_x+data.shape[3]-d_x] += 1
                c += 1

    merged_data = np.true_divide(merged_data, ov_map_counter).astype(data.dtype)

    if verbose:
        print("**** New data shape is: {}".format(merged_data.shape))
        print("### END MERGE-3D-OV-CROP ###")

    if data_mask is not None:
        merged_data_mask = np.true_divide(merged_data_mask, ov_map_counter).astype(data_mask.dtype)
        return merged_data, merged_data_mask
    else:
        return merged_data

def extract_3D_patch_with_overlap_yield(data, vol_shape, axis_order, overlap=(0,0,0), padding=(0,0,0), total_ranks=1, 
    rank=0, return_only_stats=False, verbose=False):
    """
    Extract 3D patches into smaller patches with a defined overlap. Is supports multi-GPU inference
    by setting ``total_ranks`` and ``rank`` variables. Each GPU will process a evenly number of 
    volumes in ``Z`` axis. If the number of volumes in ``Z`` to be yielded are not divisible by the 
    number of GPUs the first GPUs will process one more volume. 

    Parameters
    ----------
    data : H5 dataset
        Data to extract patches from. E.g. ``(z, y, x, channels)``.

    vol_shape : 4D int tuple
        Shape of the patches to create. E.g. ``(z, y, x, channels)``.

    axis_order : str
        Order of axes of ``data``. One between ['TZCYX', 'TZYXC', 'ZCYX', 'ZYXC'].

    overlap : Tuple of 3 floats, optional
        Amount of minimum overlap on x, y and z dimensions. Should be the same as used in
        :func:`~crop_3D_data_with_overlap`. The values must be on range ``[0, 1)``, that is, ``0%`` or ``99%`` of
        overlap. E.g. ``(z, y, x)``.
        
    padding : tuple of ints, optional
        Size of padding to be added on each axis ``(z, y, x)``. E.g. ``(24, 24, 24)``.

    total_ranks : int, optional
        Total number of GPUs.

    rank : int, optional
        Rank of the current GPU. 

    return_only_stats : bool, optional
        To just return the crop statistics without yielding any patch. Useful to precalculate how many patches
        are going to be created before doing it. 

    verbose : bool, optional
        To print useful information for debugging. 

    Yields
    ------
    img : 4D Numpy array
        Extracted patch from ``data``. E.g. ``(z, y, x, channels)``.

    real_patch_in_data : Tuple of tuples of ints
        Coordinates of patch of each axis. Needed to reconstruct the entire image. 
        E.g. ``((0, 20), (0, 8), (16, 24))`` means that the yielded patch should be
        inserted in possition [0:20,0:8,16:24]. This calculate the padding made, so
        only a portion of the real ``vol_shape`` is used. 

    total_vol : int
        Total number of crops to extract. 

    z_vol_info : dict, optional
        Information of how the volumes in ``Z`` are inserted into the original data size. 
        E.g. ``{0: [0, 20], 1: [20, 40], 2: [40, 60], 3: [60, 80], 4: [80, 100]}`` means that 
        the first volume will be place in ``[0:20]`` position, the second will be placed in 
        ``[20:40]`` and so on. 

    list_of_vols_in_z : list of list of int, optional
        Volumes in ``Z`` axis that each GPU will process. E.g. ``[[0, 1, 2], [3, 4]]`` means that
        the first GPU will process volumes ``0``, ``1`` and ``2`` (``3`` in total) whereas the second 
        GPU will process volumes ``3`` and ``4``. 
    """
    if verbose and rank == 0:
        print("### 3D-OV-CROP ###")
        print("Cropping {} images into {} with overlapping (axis order: {}). . .".format(data.shape, vol_shape, axis_order))
        print("Minimum overlap selected: {}".format(overlap))
        print("Padding: {}".format(padding))

    if len(vol_shape) != 4:
        raise ValueError("vol_shape expected to be of length 4, given {}".format(vol_shape))

    t_dim, z_dim, c_dim, y_dim, x_dim = order_dimensions(data.shape, axis_order)

    if vol_shape[0] > z_dim:
        raise ValueError("'vol_shape[0]' {} greater than {} (you can reduce 'DATA.PATCH_SIZE')"
            .format(vol_shape[0], z_dim))
    if vol_shape[1] > y_dim:
        raise ValueError("'vol_shape[1]' {} greater than {} (you can reduce 'DATA.PATCH_SIZE')"
            .format(vol_shape[1], y_dim))
    if vol_shape[2] > x_dim:
        raise ValueError("'vol_shape[2]' {} greater than {} (you can reduce 'DATA.PATCH_SIZE')"
            .format(vol_shape[2], x_dim))
    if (overlap[0] >= 1 or overlap[0] < 0) or (overlap[1] >= 1 or overlap[1] < 0) or (overlap[2] >= 1 or overlap[2] < 0):
        raise ValueError("'overlap' values must be floats between range [0, 1)")
    for i,p in enumerate(padding):
        if p >= vol_shape[i]//2:
            raise ValueError("'Padding' can not be greater than the half of 'vol_shape'. Max value for this {} input shape is {}"
                             .format(data_shape, [(vol_shape[0]//2)-1,(vol_shape[1]//2)-1,(vol_shape[2]//2)-1]))

    padded_data_shape = [z_dim+padding[0]*2,y_dim+padding[1]*2,x_dim+padding[2]*2,c_dim]
    padded_vol_shape = vol_shape

    # Calculate overlapping variables
    overlap_z = 1 if overlap[0] == 0 else 1-overlap[0]
    overlap_y = 1 if overlap[1] == 0 else 1-overlap[1]
    overlap_x = 1 if overlap[2] == 0 else 1-overlap[2]

    # Z
    step_z = int((vol_shape[0]-padding[0]*2)*overlap_z)
    vols_per_z = math.ceil(z_dim/step_z)
    last_z = 0 if vols_per_z == 1 else (((vols_per_z-1)*step_z)+vol_shape[0])-padded_data_shape[0]
    ovz_per_block = last_z//(vols_per_z-1) if vols_per_z > 1 else 0
    step_z -= ovz_per_block
    last_z -= ovz_per_block*(vols_per_z-1)

    # Y
    step_y = int((vol_shape[1]-padding[1]*2)*overlap_y)
    vols_per_y = math.ceil(y_dim/step_y)
    last_y = 0 if vols_per_y == 1 else (((vols_per_y-1)*step_y)+vol_shape[1])-padded_data_shape[1]
    ovy_per_block = last_y//(vols_per_y-1) if vols_per_y > 1 else 0
    step_y -= ovy_per_block
    last_y -= ovy_per_block*(vols_per_y-1)

    # X
    step_x = int((vol_shape[2]-padding[2]*2)*overlap_x)
    vols_per_x = math.ceil(x_dim/step_x)
    last_x = 0 if vols_per_x == 1 else (((vols_per_x-1)*step_x)+vol_shape[2])-padded_data_shape[2]
    ovx_per_block = last_x//(vols_per_x-1) if vols_per_x > 1 else 0
    step_x -= ovx_per_block
    last_x -= ovx_per_block*(vols_per_x-1)

    # Real overlap calculation for printing
    real_ov_z = ovz_per_block/(vol_shape[0]-padding[0]*2)
    real_ov_y = ovy_per_block/(vol_shape[1]-padding[1]*2)
    real_ov_x = ovx_per_block/(vol_shape[2]-padding[2]*2)
    if verbose and rank == 0:
        print("Real overlapping (%): {}".format((real_ov_z,real_ov_y,real_ov_x)))
        print("Real overlapping (pixels): {}".format(((vol_shape[0]-padding[0]*2)*real_ov_z,
              (vol_shape[1]-padding[1]*2)*real_ov_y,(vol_shape[2]-padding[2]*2)*real_ov_x)))
        print("{} patches per (z,y,x) axis".format((vols_per_z,vols_per_x,vols_per_y)))
    
    vols_in_z = vols_per_z//total_ranks
    vols_per_z_per_rank = vols_in_z
    if vols_per_z%total_ranks > rank: 
        vols_per_z_per_rank += 1
    total_vol = vols_per_z_per_rank*vols_per_y*vols_per_x

    c = 0
    list_of_vols_in_z = []
    z_vol_info = {}
    for i in range(total_ranks):
        vols = (vols_per_z//total_ranks) + 1 if vols_per_z%total_ranks > i else vols_in_z
        for j in range(vols):
            z = c+j
            real_start_z = z*step_z
            real_finish_z = min(real_start_z+step_z+ovz_per_block, z_dim)
            z_vol_info[z] = [real_start_z, real_finish_z]
        list_of_vols_in_z.append(list(range(c,c+vols)))
        c += vols
    if verbose and rank == 0:
        print(f"List of volume IDs to be processed by each GPU: {list_of_vols_in_z}")
        print(f"Positions of each volume in Z axis: {z_vol_info}")
        print("Rank {}: Total number of patches: {} - {} patches per (z,y,x) axis (per GPU)"
            .format(rank, total_vol, (vols_per_z_per_rank,vols_per_x,vols_per_y)))

    if return_only_stats:
        yield total_vol, z_vol_info, list_of_vols_in_z
        return

    for _z in range(vols_per_z_per_rank):
        z = list_of_vols_in_z[rank][0]+_z
        for y in range(vols_per_y):
            for x in range(vols_per_x):
                d_z = 0 if (z*step_z+vol_shape[0]) < padded_data_shape[0] else last_z
                d_y = 0 if (y*step_y+vol_shape[1]) < padded_data_shape[1] else last_y
                d_x = 0 if (x*step_x+vol_shape[2]) < padded_data_shape[2] else last_x

                start_z = max(0, z*step_z-d_z-padding[0])
                finish_z = min(z*step_z+vol_shape[0]-d_z-padding[0], z_dim)
                start_y = max(0, y*step_y-d_y-padding[1])
                finish_y = min(y*step_y+vol_shape[1]-d_y-padding[1], y_dim)
                start_x = max(0, x*step_x-d_x-padding[2])
                finish_x = min(x*step_x+vol_shape[2]-d_x-padding[2], x_dim)

                slices = [
                    slice(start_z, finish_z),
                    slice(start_y, finish_y),
                    slice(start_x, finish_x),
                    slice(None), # Channel
                ]

                data_ordered_slices = order_dimensions(
                    slices,
                    input_order="ZYXC",
                    output_order=axis_order,
                    default_value=0)

                img = data[tuple(data_ordered_slices)]

                # The image should have the channel dimension at the end
                current_order = np.array(range(len(img.shape)))
                transpose_order = order_dimensions(
                    current_order, #
                    input_order="ZYXC",
                    output_order=axis_order,
                    default_value=np.nan)

                # determine the transpose order
                transpose_order = [x for x in transpose_order if not np.isnan(x)]
                transpose_order = np.argsort(transpose_order)
                transpose_order = current_order[transpose_order]

                img = np.transpose(img, transpose_order)

                pad_z_left = padding[0]-z*step_z-d_z if start_z <= 0 else 0
                pad_z_right = (start_z+vol_shape[0])-z_dim if start_z+vol_shape[0] > z_dim else 0
                pad_y_left = padding[1]-y*step_y-d_y if start_y <= 0 else 0
                pad_y_right = (start_y+vol_shape[1])-y_dim if start_y+vol_shape[1] > y_dim else 0
                pad_x_left = padding[2]-x*step_x-d_x if start_x <= 0 else 0
                pad_x_right = (start_x+vol_shape[2])-x_dim if start_x+vol_shape[2] > x_dim else 0

                if img.ndim == 3:
                    img = np.pad(img,((pad_z_left,pad_z_right),(pad_y_left,pad_y_right),(pad_x_left,pad_x_right)), 'reflect')
                    img = np.expand_dims(img, -1)
                else:
                    img = np.pad(img,((pad_z_left,pad_z_right),(pad_y_left,pad_y_right),(pad_x_left,pad_x_right),(0,0)), 'reflect')

                assert img.shape == vol_shape, "Something went wrong during the patch extraction!"
                
                real_patch_in_data = [
                    [z*step_z-d_z,(z*step_z)+vol_shape[0]-d_z-(padding[0]*2)],
                    [y*step_y-d_y,(y*step_y)+vol_shape[1]-d_y-(padding[1]*2)],
                    [x*step_x-d_x,(x*step_x)+vol_shape[2]-d_x-(padding[2]*2)]
                ]

                if rank == 0:
                    yield img, real_patch_in_data, total_vol, z_vol_info, list_of_vols_in_z
                else:
                    yield img, real_patch_in_data, total_vol


def load_3d_data_classification(data_dir, patch_shape, convert_to_rgb=False, expected_classes=None, cross_val=False, cross_val_nsplits=5, 
    cross_val_fold=1, val_split=0.1, seed=0, shuffle_val=True):
    """
    Load 3D data to train classification methods.

    Parameters
    ----------
    data_dir : str
        Path to the training data.

    patch_shape: Tuple of ints
        Shape of the patch. E.g. ``(z, y, x, channels)``.

    convert_to_rgb : bool, optional
        In case RGB images are expected, e.g. if ``crop_shape`` channel is 3, those images that are grayscale are 
        converted into RGB.

    expected_classes : int, optional
        Expected number of classes to be loaded. 

    cross_val : bool, optional
        Whether to use cross validation or not. 

    cross_val_nsplits : int, optional
        Number of folds for the cross validation. 
    
    cross_val_fold : int, optional
        Number of the fold to be used as validation. 

    val_split : float, optional
        % of the train data used as validation (value between ``0`` and ``1``).

    seed : int, optional
        Seed value.

    shuffle_val : bool, optional
        Take random training examples to create validation data.

    Returns
    -------
    X_data : 5D Numpy array
        Train/test images. E.g. ``(num_of_images, z, y, x, channels)``.

    Y_data : 1D Numpy array
        Train/test images' classes. E.g. ``(num_of_images)``.

    X_val : 4D Numpy array, optional
        Validation images. E.g. ``(num_of_images, z, y, x, channels)``.

    Y_val : 1D Numpy array, optional
        Validation images' classes. E.g. ``(num_of_images)``.
    
    all_ids : List of str
        Loaded data filenames.

    val_index : List of ints
        Indexes of the samples beloging to the validation.
    """

    print("### LOAD ###")

    # Check validation
    if val_split > 0 or cross_val:
        create_val = True  
    else:
        create_val = False

    all_ids = []
    class_names = sorted(next(os.walk(data_dir))[1])
    if len(class_names) < 1:
        raise ValueError("There is no folder/class in {}".format(data_dir))
    if expected_classes is not None:
        if expected_classes != len(class_names): 
            raise ValueError("Found number of classes ({}) and 'MODEL.N_CLASSES' ({}) must match"
                .format(len(class_names), expected_classes))
        else:
            print("Found {} classes".format(len(class_names)))

    X_data, Y_data = [], []
    for c_num, folder in enumerate(class_names):
        f = os.path.join(data_dir, folder)
        print("Analizing folder {}".format(f))
        ids = sorted(next(os.walk(f))[2])
        if len(ids) == 0:
            raise ValueError("There are no images in class {}".format(f))
        else:
            print("Found {} samples".format(len(ids)))

        # Loading images 
        images, _, _, image_ids = load_3d_images_from_dir(f, return_filenames=True, crop_shape=patch_shape, 
            convert_to_rgb=convert_to_rgb)

        X_data.append(images)
        Y_data.append((c_num,)*len(ids))
        all_ids += image_ids

    # Fuse all data
    X_data = np.concatenate(X_data, 0)
    Y_data = np.concatenate(Y_data, 0)
    Y_data = np.squeeze(Y_data)

    # Create validation data splitting the train
    if create_val:
        print("Creating validation data")
        if not cross_val:
            X_data, X_val, Y_data, Y_val = train_test_split(
                X_data, Y_data, test_size=val_split, shuffle=shuffle_val, random_state=seed)
        else:
            skf = StratifiedKFold(n_splits=cross_val_nsplits, shuffle=shuffle_val,
                random_state=seed)
            fold = 1
            train_index, test_index = None, None

            for t_index, te_index in skf.split(X_data, Y_data):
                if cross_val_fold == fold:
                    X_data, X_val = X_data[t_index], X_data[te_index]
                    Y_data, Y_val = Y_data[t_index], Y_data[te_index]
                    train_index, test_index = t_index.copy(), te_index.copy()
                    break
                fold+= 1
            if len(test_index) > 5:
                print("Fold number {}. Printing the first 5 ids: {}".format(fold, test_index[:5]))
            else:
                print("Fold number {}. Indexes used in cross validation: {}".format(fold, test_index))

    if create_val:
        print("*** Loaded train data shape is: {}".format(X_data.shape))
        print("*** Loaded validation data shape is: {}".format(X_val.shape))
        if not cross_val:
            return X_data, Y_data, X_val, Y_val, all_ids
        else:
            return X_data, Y_data, X_val, Y_val, all_ids, test_index
    else:
        print("*** Loaded train data shape is: {}".format(X_data.shape))
        return X_data, Y_data, all_ids
