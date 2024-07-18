import numpy as np
import os
import math
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold
from PIL import Image

from biapy.utils.util import load_data_from_dir
from biapy.utils.misc import is_main_process


def load_and_prepare_2D_train_data(
    train_path,
    train_mask_path,
    cross_val=False,
    cross_val_nsplits=5,
    cross_val_fold=1,
    val_split=0.1,
    seed=0,
    shuffle_val=True,
    num_crops_per_dataset=0,
    random_crops_in_DA=False,
    crop_shape=None,
    y_upscaling=(1, 1),
    ov=(0, 0),
    padding=(0, 0),
    minimum_foreground_perc=-1,
    reflect_to_complete_shape=False,
    convert_to_rgb=False,
    preprocess_cfg=None,
    is_y_mask=False,
    preprocess_f=None,
):
    """
    Load train and validation images from the given paths to create 2D data.

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
        % of the train data used as validation (value between ``0`` and ``1``).

    seed : int, optional
        Seed value.

    shuffle_val : bool, optional
        Take random training examples to create validation data.

    num_crops_per_dataset : int, optional
        Number of crops per extra dataset to take into account. Useful to ensure that all the datasets have the same
        weight during network trainning.

    random_crops_in_DA : bool, optional
        To advice the method that not preparation of the data must be done, as random subvolumes will be created on
        DA, and the whole volume will be used for that.

    crop_shape : 3D int tuple, optional
        Shape of the crops. E.g. ``(y, x, channels)``.

    y_upscaling : 2 int tuple, optional
        Upscaling to be done when loading Y data. User for super-resolution workflow.

    ov : 2 floats tuple, optional
        Amount of minimum overlap on x and y dimensions. The values must be on range ``[0, 1)``, that is, ``0%`` or
        ``99%`` of overlap. E.g. ``(y, x)``.

    padding : tuple of ints, optional
        Size of padding to be added on each axis ``(y, x)``. E.g. ``(24, 24)``

    minimum_foreground_perc : float, optional
        Minimum percetnage of foreground that a sample need to have no not be discarded.

    reflect_to_complete_shape : bool, optional
        Wheter to increase the shape of the dimension that have less size than selected patch size padding it with
        'reflect'.

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

    test_index : List of ints
        Indexes of the samples beloging to the validation.

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

    """

    print("### LOAD ###")

    # Disable crops when random_crops_in_DA is selected
    delay_crop = False
    if random_crops_in_DA:
        crop = False
    else:
        if cross_val:
            crop = False
            # Delay the crop to be made after cross validation
            delay_crop = True
        else:
            crop = True

    # Check validation
    if val_split > 0 or cross_val:
        create_val = True
    else:
        create_val = False

    print("0) Loading train images . . .")
    X_train, orig_train_shape, _, t_filenames = load_data_from_dir(
        train_path,
        crop=crop,
        crop_shape=crop_shape,
        overlap=ov,
        padding=padding,
        return_filenames=True,
        reflect_to_complete_shape=reflect_to_complete_shape,
        convert_to_rgb=convert_to_rgb,
        preprocess_cfg=preprocess_cfg,
        is_mask=False,
        preprocess_f=preprocess_f,
    )
    if train_mask_path is not None:
        print("1) Loading train GT . . .")
        scrop = (
            crop_shape[0] * y_upscaling[0],
            crop_shape[1] * y_upscaling[1],
            crop_shape[2],
        )
        Y_train, _, _, _ = load_data_from_dir(
            train_mask_path,
            crop=crop,
            crop_shape=scrop,
            overlap=ov,
            padding=padding,
            return_filenames=True,
            check_channel=False,
            check_drange=False,
            reflect_to_complete_shape=reflect_to_complete_shape,
            preprocess_cfg=preprocess_cfg,
            is_mask=is_y_mask,
            preprocess_f=preprocess_f,
        )

        # Check that the shape of all images match
        if isinstance(Y_train, list):
            for i in range(len(Y_train)):
                xshape = X_train[i].shape
                yshape = Y_train[i].shape
                real_x_shape = (
                    xshape[0] * y_upscaling[0],
                    xshape[1] * y_upscaling[1],
                    xshape[2],
                )
                real_y_shape = (
                    yshape[0] * y_upscaling[0],
                    yshape[1] * y_upscaling[1],
                    yshape[2],
                )
                print(real_x_shape, real_y_shape)
                if real_x_shape != real_y_shape:
                    raise ValueError(
                        f"There is a mismatch between input image and its corresponding ground truth ({real_x_shape} vs "
                        f"{real_y_shape}). Please check the images. Specifically, the sample that doesn't match is the number {i}"
                        f" (file: {t_filenames[i]})"
                    )
    else:
        Y_train = None

    # Discard images that do not surpass the foreground percentage threshold imposed
    if minimum_foreground_perc != -1 and Y_train is not None:
        print("Data that do not have {}% of foreground is discarded".format(minimum_foreground_perc))

        X_train_keep = []
        Y_train_keep = []
        are_lists = True if type(Y_train) is list else False

        samples_discarded = 0
        for i in tqdm(range(len(Y_train)), leave=False, disable=not is_main_process()):
            labels, npixels = np.unique((Y_train[i] > 0).astype(np.uint8), return_counts=True)

            total_pixels = 1
            for val in list(Y_train[i].shape):
                total_pixels *= val

            discard = False
            if len(labels) == 1:
                discard = True
            else:
                if (sum(npixels[1:] / total_pixels)) < minimum_foreground_perc:
                    discard = True

            if discard:
                samples_discarded += 1
            else:
                if are_lists:
                    X_train_keep.append(X_train[i])
                    Y_train_keep.append(Y_train[i])
                else:
                    X_train_keep.append(np.expand_dims(X_train[i], 0))
                    Y_train_keep.append(np.expand_dims(Y_train[i], 0))
        del X_train, Y_train

        if len(X_train_keep) == 0:
            raise ValueError(
                "'TRAIN.MINIMUM_FOREGROUND_PER' value is too high, leading to the discarding of all training samples. Please, "
                "reduce its value."
            )

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
                raise ValueError(
                    "0 or 1 sample left to train, which is insufficent. "
                    "Please, decrease the percentage to be more permissive"
                )
        else:
            print("*** Remaining data shape is {}".format((len(X_train),) + X_train[0].shape[1:]))
            if len(X_train) <= 1 and create_val:
                raise ValueError(
                    "0 or 1 sample left to train, which is insufficent. "
                    "Please, decrease the percentage to be more permissive"
                )

    if num_crops_per_dataset != 0:
        X_train = X_train[:num_crops_per_dataset]
        if Y_train is not None:
            Y_train = Y_train[:num_crops_per_dataset]

    if Y_train is not None and len(X_train) != len(Y_train):
        raise ValueError(
            "Different number of raw and ground truth items ({} vs {}). "
            "Please check the data!".format(len(X_train), len(Y_train))
        )

    # Create validation data splitting the train
    if create_val:
        print("Creating validation data")
        Y_val = None
        if not cross_val:
            if Y_train is not None:
                X_train, X_val, Y_train, Y_val = train_test_split(
                    X_train,
                    Y_train,
                    test_size=val_split,
                    shuffle=shuffle_val,
                    random_state=seed,
                )
            else:
                X_train, X_val = train_test_split(X_train, test_size=val_split, shuffle=shuffle_val, random_state=seed)
        else:
            skf = StratifiedKFold(n_splits=cross_val_nsplits, shuffle=shuffle_val, random_state=seed)
            fold = 1
            train_index, test_index = None, None

            y_len = len(Y_train) if Y_train is not None else len(X_train)
            for t_index, te_index in skf.split(np.zeros(len(X_train)), np.zeros(y_len)):
                if cross_val_fold == fold:
                    if not isinstance(X_train, list):
                        X_train, X_val = X_train[t_index], X_train[te_index]
                    else:
                        X_val = []
                        for val_idx in te_index:
                            X_val.append(X_train[val_idx])
                        for val_idx in te_index:
                            del X_val[val_idx]
                    if Y_train is not None:
                        if not isinstance(Y_train, list):
                            Y_train, Y_val = Y_train[t_index], Y_train[te_index]
                        else:
                            Y_val = []
                            for val_idx in te_index:
                                Y_val.append(Y_val[val_idx])
                            for val_idx in te_index:
                                del Y_val[val_idx]
                    train_index, test_index = t_index.copy(), te_index.copy()
                    break
                fold += 1

            if len(test_index) > 5:
                print("Fold number {}. Printing the first 5 ids: {}".format(fold, test_index[:5]))
            else:
                print("Fold number {}. Indexes used in cross validation: {}".format(fold, test_index))

            # Then crop after cross validation
            if delay_crop:
                # X_train
                data = []
                for img_num in range(len(X_train)):
                    if X_train[img_num].shape != crop_shape[:2] + (X_train[img_num].shape[-1],):
                        img = crop_data_with_overlap(
                            (X_train[img_num][0] if isinstance(X_train, list) else np.expand_dims(X_train[img_num], 0)),
                            crop_shape[:2] + (X_train[img_num].shape[-1],),
                            overlap=ov,
                            padding=padding,
                            verbose=False,
                        )
                    data.append(img)
                X_train = np.concatenate(data)
                del data

                # Y_train
                if Y_train is not None:
                    data_mask = []
                    scrop = (
                        crop_shape[0] * y_upscaling[0],
                        crop_shape[1] * y_upscaling[1],
                        crop_shape[2],
                    )
                    for img_num in range(len(Y_train)):
                        if Y_train[img_num].shape != scrop[:2] + (Y_train[img_num].shape[-1],):
                            img = crop_data_with_overlap(
                                (
                                    Y_train[img_num][0]
                                    if isinstance(Y_train, list)
                                    else np.expand_dims(Y_train[img_num], 0)
                                ),
                                scrop[:2] + (Y_train[img_num].shape[-1],),
                                overlap=ov,
                                padding=padding,
                                verbose=False,
                            )
                        data_mask.append(img)
                    Y_train = np.concatenate(data_mask)
                    del data_mask

                # X_val
                data = []
                for img_num in range(len(X_val)):
                    if X_val[img_num].shape != crop_shape[:2] + (X_val[img_num].shape[-1],):
                        img = crop_data_with_overlap(
                            (X_val[img_num][0] if isinstance(X_val, list) else np.expand_dims(X_val[img_num], 0)),
                            crop_shape[:2] + (X_val[img_num].shape[-1],),
                            overlap=ov,
                            padding=padding,
                            verbose=False,
                        )
                    data.append(img)
                X_val = np.concatenate(data)
                del data

                # Y_val
                if Y_val is not None:
                    data_mask = []
                    scrop = (
                        crop_shape[0] * y_upscaling[0],
                        crop_shape[1] * y_upscaling[1],
                        crop_shape[2],
                    )
                    for img_num in range(len(Y_val)):
                        if Y_val[img_num].shape != scrop[:2] + (Y_val[img_num].shape[-1],):
                            img = crop_data_with_overlap(
                                (Y_val[img_num][0] if isinstance(Y_val, list) else np.expand_dims(Y_val[img_num], 0)),
                                scrop[:2] + (Y_val[img_num].shape[-1],),
                                overlap=ov,
                                padding=padding,
                                verbose=False,
                            )
                        data_mask.append(img)
                    Y_val = np.concatenate(data_mask)
                    del data_mask

    # Check that the shape of all images match
    if Y_train is not None:
        if not isinstance(X_train, list):
            if Y_train.shape[0] != X_train.shape[0]:
                raise ValueError(
                    f"Seems that input images do not correspond to their ground truth in shape ({X_train.shape[0]} samples vs "
                    f"{Y_train.shape[0]} samples). Please check the images. If you are in super-resolution workflow maybe you did not "
                    "configured properly 'PROBLEM.SUPER_RESOLUTION.UPSCALING' variable"
                )
        else:
            if Y_train[0].shape[0] != X_train[0].shape[0]:
                raise ValueError(
                    f"Seems that input images do not correspond to their ground truth in shape ({X_train[0].shape[0]} samples vs "
                    f"{Y_train[0].shape[0]} samples). Please check the images. If you are in super-resolution workflow maybe you did not "
                    "configured properly 'PROBLEM.SUPER_RESOLUTION.UPSCALING' variable"
                )

    s = X_train.shape if not isinstance(X_train, list) else (len(X_train),) + X_train[0].shape[1:]
    if Y_train is not None:
        sm = Y_train.shape if not isinstance(Y_train, list) else (len(Y_train),) + Y_train[0].shape[1:]
    if create_val:
        sv = X_val.shape if not isinstance(X_val, list) else (len(X_val),) + X_val[0].shape[1:]
        if Y_val is not None:
            svm = Y_val.shape if not isinstance(Y_val, list) else (len(Y_val),) + Y_val[0].shape[1:]
        if not isinstance(X_train, list):
            print("Not all samples seem to have the same shape. Number of samples: {}".format(len(X_train)))
        print("*** Loaded train data shape is: {}".format(s))
        if Y_train is not None:
            print("*** Loaded train GT shape is: {}".format(sm))
        print("*** Loaded validation data shape is: {}".format(sv))
        if Y_val is not None:
            print("*** Loaded validation GT shape is: {}".format(svm))
        print("### END LOAD ###")

        if not cross_val:
            return X_train, Y_train, X_val, Y_val, t_filenames
        else:
            return X_train, Y_train, X_val, Y_val, t_filenames, test_index
    else:
        print("*** Loaded train data shape is: {}".format(s))
        print("### END LOAD ###")

        return X_train, Y_train, t_filenames


def crop_data_with_overlap(data, crop_shape, data_mask=None, overlap=(0, 0), padding=(0, 0), verbose=True):
    """
    Crop data into small square pieces with overlap. The difference with :func:`~crop_data` is that this function
    allows you to create patches with overlap.

    The opposite function is :func:`~merge_data_with_overlap`.

    Parameters
    ----------
    data : 4D Numpy array
        Data to crop. E.g. ``(num_of_images, y, x, channels)``.

    crop_shape : 3 int tuple
        Shape of the crops to create. E.g. ``(y, x, channels)``.

    data_mask : 4D Numpy array, optional
        Data mask to crop. E.g. ``(num_of_images, y, x, channels)``.

    overlap : Tuple of 2 floats, optional
        Amount of minimum overlap on x and y dimensions. The values must be on range ``[0, 1)``, that is, ``0%`` or
        ``99%`` of overlap. E. g. ``(y, x)``.

    padding : tuple of ints, optional
        Size of padding to be added on each axis ``(y, x)``. E.g. ``(24, 24)``.

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

    if data.ndim != 4:
        raise ValueError("data expected to be 4 dimensional, given {}".format(data.shape))
    if data_mask is not None:
        if data.ndim != 4:
            raise ValueError("data mask expected to be 4 dimensional, given {}".format(data_mask.shape))
        if data.shape[:-1] != data_mask.shape[:-1]:
            raise ValueError(
                "data and data_mask shapes mismatch: {} vs {}".format(data.shape[:-1], data_mask.shape[:-1])
            )

    for i, p in enumerate(padding):
        if p >= crop_shape[i] // 2:
            raise ValueError(
                "'Padding' can not be greater than the half of 'crop_shape'. Max value for this {} input shape is {}".format(
                    data.shape, [(crop_shape[0] // 2) - 1, (crop_shape[1] // 2) - 1]
                )
            )
    if len(crop_shape) != 3:
        raise ValueError("crop_shape expected to be of length 3, given {}".format(crop_shape))
    if crop_shape[0] > data.shape[1]:
        raise ValueError(
            "'crop_shape[0]' {} greater than {} (you can reduce 'DATA.PATCH_SIZE' or use 'DATA.REFLECT_TO_COMPLETE_SHAPE')".format(
                crop_shape[0], data.shape[1]
            )
        )
    if crop_shape[1] > data.shape[2]:
        raise ValueError(
            "'crop_shape[1]' {} greater than {} (you can reduce 'DATA.PATCH_SIZE' or use 'DATA.REFLECT_TO_COMPLETE_SHAPE')".format(
                crop_shape[1], data.shape[2]
            )
        )
    if (overlap[0] >= 1 or overlap[0] < 0) or (overlap[1] >= 1 or overlap[1] < 0):
        raise ValueError("'overlap' values must be floats between range [0, 1)")

    if verbose:
        print("### OV-CROP ###")
        print("Cropping {} images into {} with overlapping. . .".format(data.shape, crop_shape))
        print("Minimum overlap selected: {}".format(overlap))
        print("Padding: {}".format(padding))

    if (overlap[0] >= 1 or overlap[0] < 0) and (overlap[1] >= 1 or overlap[1] < 0):
        raise ValueError("'overlap' values must be floats between range [0, 1)")

    padded_data = np.pad(
        data,
        ((0, 0), (padding[1], padding[1]), (padding[0], padding[0]), (0, 0)),
        "reflect",
    )
    if data_mask is not None:
        padded_data_mask = np.pad(
            data_mask,
            ((0, 0), (padding[1], padding[1]), (padding[0], padding[0]), (0, 0)),
            "reflect",
        )

    # Calculate overlapping variables
    overlap_x = 1 if overlap[0] == 0 else 1 - overlap[0]
    overlap_y = 1 if overlap[1] == 0 else 1 - overlap[1]

    # Y
    step_y = int((crop_shape[0] - padding[0] * 2) * overlap_y)
    crops_per_y = math.ceil(data.shape[1] / step_y)
    last_y = 0 if crops_per_y == 1 else (((crops_per_y - 1) * step_y) + crop_shape[0]) - padded_data.shape[1]
    ovy_per_block = last_y // (crops_per_y - 1) if crops_per_y > 1 else 0
    step_y -= ovy_per_block
    last_y -= ovy_per_block * (crops_per_y - 1)

    # X
    step_x = int((crop_shape[1] - padding[1] * 2) * overlap_x)
    crops_per_x = math.ceil(data.shape[2] / step_x)
    last_x = 0 if crops_per_x == 1 else (((crops_per_x - 1) * step_x) + crop_shape[1]) - padded_data.shape[2]
    ovx_per_block = last_x // (crops_per_x - 1) if crops_per_x > 1 else 0
    step_x -= ovx_per_block
    last_x -= ovx_per_block * (crops_per_x - 1)

    # Real overlap calculation for printing
    real_ov_y = ovy_per_block / (crop_shape[0] - padding[0] * 2)
    real_ov_x = ovx_per_block / (crop_shape[1] - padding[1] * 2)

    if verbose:
        print("Real overlapping (%): {}".format(real_ov_x, real_ov_y))
        print(
            "Real overlapping (pixels): {}".format(
                (crop_shape[1] - padding[1] * 2) * real_ov_x,
                (crop_shape[0] - padding[0] * 2) * real_ov_y,
            )
        )
        print("{} patches per (x,y) axis".format(crops_per_x, crops_per_y))

    total_vol = data.shape[0] * (crops_per_x) * (crops_per_y)
    cropped_data = np.zeros((total_vol,) + crop_shape, dtype=data.dtype)
    if data_mask is not None:
        cropped_data_mask = np.zeros(
            (total_vol,) + crop_shape[:2] + (data_mask.shape[-1],),
            dtype=data_mask.dtype,
        )

    c = 0
    for z in range(data.shape[0]):
        for y in range(crops_per_y):
            for x in range(crops_per_x):
                d_y = 0 if (y * step_y + crop_shape[0]) < padded_data.shape[1] else last_y
                d_x = 0 if (x * step_x + crop_shape[1]) < padded_data.shape[2] else last_x

                cropped_data[c] = padded_data[
                    z,
                    y * step_y - d_y : y * step_y + crop_shape[0] - d_y,
                    x * step_x - d_x : x * step_x + crop_shape[1] - d_x,
                ]

                if data_mask is not None:
                    cropped_data_mask[c] = padded_data_mask[
                        z,
                        y * step_y - d_y : y * step_y + crop_shape[0] - d_y,
                        x * step_x - d_x : x * step_x + crop_shape[1] - d_x,
                    ]
                c += 1

    if verbose:
        print("**** New data shape is: {}".format(cropped_data.shape))
        print("### END OV-CROP ###")

    if data_mask is not None:
        return cropped_data, cropped_data_mask
    else:
        return cropped_data


def merge_data_with_overlap(
    data,
    original_shape,
    data_mask=None,
    overlap=(0, 0),
    padding=(0, 0),
    verbose=True,
    out_dir=None,
    prefix="",
):
    """
    Merge data with an amount of overlap.

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

    +-----------------------------------------------+-----------------------------------------------+
    | .. figure:: ../../img/merged_ov_map_0.png     | .. figure:: ../../img/merged_ov_map_0.25.png  |
    |   :width: 80%                                 |   :width: 80%                                 |
    |   :align: center                              |   :align: center                              |
    |                                               |                                               |
    |   Example 1 overlapping map                   |   Example 2 overlapping map                   |
    +-----------------------------------------------+-----------------------------------------------+
    | .. figure:: ../../img/merged_ov_map_0.5.png   | .. figure:: ../../img/merged_ov_map_0.5inx.png|
    |   :width: 80%                                 |   :width: 80%                                 |
    |   :align: center                              |   :align: center                              |
    |                                               |                                               |
    |   Example 3 overlapping map                   |   Example 4 overlapping map                   |
    +-----------------------------------------------+-----------------------------------------------+
    """

    if data_mask is not None:
        if data.shape[:-1] != data_mask.shape[:-1]:
            raise ValueError(
                "data and data_mask shapes mismatch: {} vs {}".format(data.shape[:-1], data_mask.shape[:-1])
            )

    for i, p in enumerate(padding):
        if p >= data.shape[i + 1] // 2:
            raise ValueError(
                "'Padding' can not be greater than the half of 'data' shape. Max value for this {} input shape is {}".format(
                    data.shape, [(data.shape[1] // 2) - 1, (data.shape[2] // 2) - 1]
                )
            )

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
    data = data[
        :,
        padding[0] : data.shape[1] - padding[0],
        padding[1] : data.shape[2] - padding[1],
    ]

    merged_data = np.zeros((original_shape), dtype=np.float32)
    if data_mask is not None:
        merged_data_mask = np.zeros((original_shape[:-1] + (data_mask.shape[-1],)), dtype=np.float32)
        data_mask = data_mask[
            :,
            padding[0] : data_mask.shape[1] - padding[0],
            padding[1] : data_mask.shape[2] - padding[1],
        ]

    ov_map_counter = np.zeros(original_shape[:-1] + (1,), dtype=np.int32)
    if out_dir is not None:
        crop_grid = np.zeros(original_shape[1:], dtype=np.int32)

    # Calculate overlapping variables
    overlap_x = 1 if overlap[0] == 0 else 1 - overlap[0]
    overlap_y = 1 if overlap[1] == 0 else 1 - overlap[1]

    padded_data_shape = [
        original_shape[1] + 2 * padding[0],
        original_shape[2] + 2 * padding[1],
    ]

    # Y
    step_y = int((pad_input_shape[1] - padding[0] * 2) * overlap_y)
    crops_per_y = math.ceil(original_shape[1] / step_y)
    last_y = 0 if crops_per_y == 1 else (((crops_per_y - 1) * step_y) + pad_input_shape[1]) - padded_data_shape[0]
    ovy_per_block = last_y // (crops_per_y - 1) if crops_per_y > 1 else 0
    step_y -= ovy_per_block
    last_y -= ovy_per_block * (crops_per_y - 1)

    # X
    step_x = int((pad_input_shape[2] - padding[1] * 2) * overlap_x)
    crops_per_x = math.ceil(original_shape[2] / step_x)
    last_x = 0 if crops_per_x == 1 else (((crops_per_x - 1) * step_x) + pad_input_shape[2]) - padded_data_shape[1]
    ovx_per_block = last_x // (crops_per_x - 1) if crops_per_x > 1 else 0
    step_x -= ovx_per_block
    last_x -= ovx_per_block * (crops_per_x - 1)

    # Real overlap calculation for printing
    real_ov_y = ovy_per_block / (pad_input_shape[1] - padding[0] * 2)
    real_ov_x = ovx_per_block / (pad_input_shape[2] - padding[1] * 2)
    if verbose:
        print("Real overlapping (%): {}".format((real_ov_x, real_ov_y)))
        print(
            "Real overlapping (pixels): {}".format(
                (
                    (pad_input_shape[2] - padding[1] * 2) * real_ov_x,
                    (pad_input_shape[1] - padding[0] * 2) * real_ov_y,
                )
            )
        )
        print("{} patches per (x,y) axis".format((crops_per_x, crops_per_y)))

    c = 0
    for z in range(original_shape[0]):
        for y in range(crops_per_y):
            for x in range(crops_per_x):
                d_y = 0 if (y * step_y + data.shape[1]) < original_shape[1] else last_y
                d_x = 0 if (x * step_x + data.shape[2]) < original_shape[2] else last_x

                merged_data[
                    z,
                    y * step_y - d_y : y * step_y + data.shape[1] - d_y,
                    x * step_x - d_x : x * step_x + data.shape[2] - d_x,
                ] += data[c]

                if data_mask is not None:
                    merged_data_mask[
                        z,
                        y * step_y - d_y : y * step_y + data.shape[1] - d_y,
                        x * step_x - d_x : x * step_x + data.shape[2] - d_x,
                    ] += data_mask[c]

                ov_map_counter[
                    z,
                    y * step_y - d_y : y * step_y + data.shape[1] - d_y,
                    x * step_x - d_x : x * step_x + data.shape[2] - d_x,
                ] += 1

                if z == 0 and out_dir is not None:
                    crop_grid[
                        y * step_y - d_y : y * step_y + data.shape[1] - d_y,
                        x * step_x - d_x,
                    ] = 1
                    crop_grid[
                        y * step_y - d_y : y * step_y + data.shape[1] - d_y,
                        x * step_x + data.shape[2] - d_x - 1,
                    ] = 1
                    crop_grid[
                        y * step_y - d_y,
                        x * step_x - d_x : x * step_x + data.shape[2] - d_x,
                    ] = 1
                    crop_grid[
                        y * step_y + data.shape[1] - d_y - 1,
                        x * step_x - d_x : x * step_x + data.shape[2] - d_x,
                    ] = 1

                c += 1

    merged_data = np.true_divide(merged_data, ov_map_counter).astype(data.dtype)
    if data_mask is not None:
        merged_data_mask = np.true_divide(merged_data_mask, ov_map_counter).astype(data_mask.dtype)

    # Save a copy of the merged data with the overlapped regions colored as: green when 2 crops overlap, yellow when
    # (2 < x < 6) and red when more than 6 overlaps are merged
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

        ov_map = ov_map_counter[0]
        ov_map = ov_map.astype("int32")

        ov_map[np.where(ov_map_counter[0] >= 2)] = -3
        ov_map[np.where(ov_map_counter[0] >= 3)] = -2
        ov_map[np.where(ov_map_counter[0] >= 6)] = -1
        ov_map[np.where(crop_grid == 1)] = -4

        # Paint overlap regions
        im = Image.fromarray(merged_data[0, ..., 0])
        im = im.convert("RGBA")
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


def load_data_classification(
    data_dir,
    patch_shape,
    convert_to_rgb=True,
    expected_classes=None,
    cross_val=False,
    cross_val_nsplits=5,
    cross_val_fold=1,
    val_split=0.1,
    seed=0,
    shuffle_val=True,
    preprocess_cfg=None,
    preprocess_f=None,
):
    """
    Load data to train classification methods.

    Parameters
    ----------
    data_dir : str
        Path to the training data.

    patch_shape: Tuple of ints
        Shape of the patch. E.g. ``(y, x, channels)``.

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

    preprocess_cfg : dict, optional
        Configuration parameters for preprocessing, is necessary in case you want to apply any preprocessing.

    preprocess_f : function, optional
        The preprocessing function, is necessary in case you want to apply any preprocessing.

    Returns
    -------
    X_data : 4D Numpy array
        Train images. E.g. ``(num_of_images, y, x, channels)``.

    Y_data : 4D Numpy array
        Train images' mask. E.g. ``(num_of_images, y, x, channels)``.

    X_val : 4D Numpy array, optional
        Validation images (``val_split > 0``). E.g. ``(num_of_images, y, x, channels)``.

    Y_val : 4D Numpy array, optional
        Validation images' mask (``val_split > 0``). E.g. ``(num_of_images, y, x, channels)``.

    all_ids : List of str
        Loaded data filenames.

    test_index : List of ints
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
            raise ValueError(
                "Found number of classes ({}) and 'MODEL.N_CLASSES' ({}) must match".format(
                    len(class_names), expected_classes
                )
            )
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
        images, _, _, image_ids = load_data_from_dir(
            f,
            return_filenames=True,
            crop_shape=patch_shape,
            convert_to_rgb=convert_to_rgb,
            preprocess_cfg=preprocess_cfg,
            is_mask=False,
            preprocess_f=preprocess_f,
        )

        X_data.append(images)
        Y_data.append((c_num,) * len(ids))
        all_ids += image_ids

    # Fuse all data
    try:
        X_data = np.concatenate(X_data, 0)
    except:
        raise ValueError(
            "Seems that there is a problem merging the image into just one array. Are you sure that all the images"
            " have the same shape? If not, do not try to load it into memory (DATA.*.IN_MEMORY variables)"
        )
    Y_data = np.concatenate(Y_data, 0)
    Y_data = np.squeeze(Y_data)

    # Create validation data splitting the train
    if create_val:
        print("Creating validation data")
        if not cross_val:
            if len(X_data) == 1:
                raise ValueError(
                    "Validation data can not be extracted from training data as it only has one sample. Please check the data."
                )
            X_data, X_val, Y_data, Y_val = train_test_split(
                X_data,
                Y_data,
                test_size=val_split,
                shuffle=shuffle_val,
                random_state=seed,
                stratify=Y_data,
            )
        else:
            if len(X_data) < cross_val_nsplits:
                raise ValueError(
                    f"Validation data can not be extracted from training data as the number of splits ({cross_val_nsplits}) "
                    f"is greater than the number of samples {len(X_data)}. Please check the data."
                )
            skf = StratifiedKFold(n_splits=cross_val_nsplits, shuffle=shuffle_val, random_state=seed)
            fold = 1
            train_index, test_index = None, None

            for t_index, te_index in skf.split(X_data, Y_data):
                if cross_val_fold == fold:
                    if not isinstance(X_data, list):
                        X_data, X_val = X_data[t_index], X_data[te_index]
                        Y_data, Y_val = Y_data[t_index], Y_data[te_index]
                    else:
                        X_val = []
                        Y_val = []
                        for val_idx in te_index:
                            X_val.append(X_data[val_idx])
                            Y_val.append(Y_data[val_idx])
                        for val_idx in te_index:
                            del X_val[val_idx]
                            del Y_val[val_idx]
                    train_index, test_index = t_index.copy(), te_index.copy()
                    break
                fold += 1
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
