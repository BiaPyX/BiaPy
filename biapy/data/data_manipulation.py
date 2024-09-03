import os
import h5py
import numpy as np
from tqdm import tqdm
from skimage.io import imread
from sklearn.model_selection import train_test_split, StratifiedKFold

from biapy.utils.util import read_chunked_data, read_chunked_nested_data
from biapy.utils.misc import is_main_process
from biapy.data.data_2D_manipulation import crop_data_with_overlap, ensure_2d_shape
from biapy.data.data_3D_manipulation import (
    crop_3D_data_with_overlap,
    extract_3D_patch_with_overlap_yield,
    order_dimensions,
    ensure_3d_shape,
)


def load_and_prepare_train_data(
    train_path,
    train_mask_path,
    train_in_memory,
    train_ov,
    train_padding,
    val_path,
    val_mask_path,
    val_in_memory,
    val_ov,
    val_padding,
    cross_val=False,
    cross_val_nsplits=5,
    cross_val_fold=1,
    val_split=0.1,
    seed=0,
    shuffle_val=True,
    train_preprocess_f=None,
    train_preprocess_cfg=None,
    train_filter_conds=[],
    train_filter_vals=None,
    train_filter_signs=None,
    val_preprocess_f=None,
    val_preprocess_cfg=None,
    val_filter_conds=[],
    val_filter_vals=None,
    val_filter_signs=None,
    filter_by_entire_image=True,
    random_crops_in_DA=False,
    crop_shape=None,
    y_upscaling=(1, 1),
    reflect_to_complete_shape=False,
    convert_to_rgb=False,
    is_y_mask=False,
    is_3d=False,
    train_zarr_data_information=None,
    val_zarr_data_information=None,
    multiple_raw_images=False,
):
    """
    Load training and validation data.

    Parameters
    ----------
    train_path : str
        Path to the training data.

    train_mask_path : str
        Path to the training data masks.

    train_in_memory : str
        Whether the training data must be loaded in memory or not.

    train_ov : 2D/3D float tuple, optional
        Amount of minimum overlap on x and y dimensions for train data. The values must be on range ``[0, 1)``,
        that is, ``0%`` or ``99%`` of overlap. Shape is ``(y, x)`` for 2D or ``(z, y, x)`` for 3D.

    train_padding : 2D/3D int tuple, optional
        Size of padding to be added on each axis to the train data. Shape is ``(y, x)`` for 2D or ``(z, y, x)`` for 3D.

    val_path : str
        Path to the validation data.

    val_mask_path : str
        Path to the validation data masks.

    val_in_memory : str
        Whether the validation data must be loaded in memory or not.

    val_ov : 2D/3D float tuple, optional
        Amount of minimum overlap on x and y dimensions for val data. The values must be on range ``[0, 1)``,
        that is, ``0%`` or ``99%`` of overlap. Shape is ``(y, x)`` for 2D or ``(z, y, x)`` for 3D.

    val_padding : 2D/3D int tuple, optional
        Size of padding to be added on each axis to the val data. Shape is ``(y, x)`` for 2D or ``(z, y, x)`` for 3D.

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

    train_preprocess_f : function, optional
        The train preprocessing function, is necessary in case you want to apply any preprocessing.

    train_preprocess_cfg : dict, optional
        Configuration parameters for train preprocessing, is necessary in case you want to apply any preprocessing.

    train_filter_conds : list of lists of str
        Filter conditions to be applied to the train data. The three variables, ``filter_conds``, ``filter_vals`` and ``filter_vals``
        will compose a list of conditions to remove the samples from the list. They are list of list of conditions. For instance, the
        conditions can be like this: ``[['A'], ['B','C']]``. Then, if the sample satisfies the first list of conditions, only 'A'
        in this first case (from ['A'] list), or satisfy 'B' and 'C' (from ['B','C'] list) it will be removed. In each sublist all the
        conditions must be satisfied. Available properties are: [``'foreground'``, ``'mean'``, ``'min'``, ``'max'``].
        Each property descrition:
          * ``'foreground'`` is defined as the mask foreground percentage.
          * ``'mean'`` is defined as the mean value.
          * ``'min'`` is defined as the min value.
          * ``'max'`` is defined as the max value.

    train_filter_vals : list of int/float
        Represent the values of the properties listed in ``train_filter_conds`` that the images need to satisfy to not be dropped.

    train_filter_signs : list of list of str
        Signs to do the comparison for train data filtering. Options: [``'gt'``, ``'ge'``, ``'lt'``, ``'le'``] that corresponds to
        "greather than", e.g. ">", "greather equal", e.g. ">=", "less than", e.g. "<", and "less equal" e.g. "<=" comparisons.

    val_preprocess_f : function, optional
        The validation preprocessing function, is necessary in case you want to apply any preprocessing.

    val_preprocess_cfg : dict, optional
        Configuration parameters for validation preprocessing, is necessary in case you want to apply any preprocessing.

    val_filter_conds : list of lists of str
        Filter conditions to be applied to the validation data. The three variables, ``filter_conds``, ``filter_vals`` and ``filter_vals``
        will compose a list of conditions to remove the images from the list. They are list of list of conditions. For instance, the
        conditions can be like this: ``[['A'], ['B','C']]``. Then, if the sample satisfies the first list of conditions, only 'A'
        in this first case (from ['A'] list), or satisfy 'B' and 'C' (from ['B','C'] list) it will be removed. In each sublist all
        the conditions must be satisfied. Available properties are: [``'foreground'``, ``'mean'``, ``'min'``, ``'max'``].
        Each property descrition:
          * ``'foreground'`` is defined as the mask foreground percentage.
          * ``'mean'`` is defined as the mean value.
          * ``'min'`` is defined as the min value.
          * ``'max'`` is defined as the max value.

    val_filter_vals : list of int/float
        Represent the values of the properties listed in ``val_filter_conds`` that the images need to satisfy to not be dropped.

    val_filter_signs : list of list of str
        Signs to do the comparison for validation data filtering. Options: [``'gt'``, ``'ge'``, ``'lt'``, ``'le'``] that corresponds to
        "greather than", e.g. ">", "greather equal", e.g. ">=", "less than", e.g. "<", and "less equal" e.g. "<=" comparisons.

    filter_by_entire_image : bool, optional
        If filtering is done this will decide how the filtering will be done:
            * ``True``: apply filter image by image. 
            * ``False``: apply filtering sample by sample. Each sample represents a patch within an image.

    random_crops_in_DA : bool, optional
        To advice the method that not preparation of the data must be done, as random subvolumes will be created on
        DA, and the whole volume will be used for that.

    crop_shape : 3D/4D int tuple, optional
        Shape of the crops. E.g. ``(y, x, channels)`` for 2D and ``(z, y, x, channels)`` for 3D.

    y_upscaling : 2D/3D int tuple, optional
        Upscaling to be done when loading Y data. User for super-resolution workflow.

    reflect_to_complete_shape : bool, optional
        Wheter to increase the shape of the dimension that have less size than selected patch size padding it with
        'reflect'.

    convert_to_rgb : bool, optional
        In case RGB images are expected, e.g. if ``crop_shape`` channel is 3, those images that are grayscale are
        converted into RGB.

    is_y_mask : bool, optional
        Whether the data are masks. It is used to control the preprocessing of the data.

    is_3d : bool, optional
        Whether if the expected images to read are 3D or not.

    train_zarr_data_information : dict, optional
        Additional information when using Zarr/H5 files for training. The following keys are expected:
            * ``"raw_path"``: path where the raw images reside within the zarr (used when ``multiple_data_within_zarr`` is ``True``).
            * ``"gt_path"``: path where the mask images reside within the zarr (used when ``multiple_data_within_zarr`` is ``True``).
            * ``"use_gt_path"``: whether the GT that should be used or not.
            * ``"multiple_data_within_zarr"``: Whether if your input Zarr contains the raw images and labels together or not.
            * ``"input_img_axes"``: order of the axes of the images.
            * ``"input_mask_axes"``: order of the axes of the masks.

    val_zarr_data_information : dict, optional
        Additional information when using Zarr/H5 files for validation. Same keys as ``train_zarr_data_information``
        are expected.

    multiple_raw_images : bool, optional
        When a folder of folders for each image is expected. In each of those subfolder different versions of the same image 
        are placed. Visit the following tutorial for a real use case and a more detailed description:
        `Light My Cells <https://biapy.readthedocs.io/en/latest/tutorials/image-to-image/lightmycells.html>`_.
        This is used when ``PROBLEM.IMAGE_TO_IMAGE.MULTIPLE_RAW_ONE_TARGET_LOADER`` is selected. 

    Returns
    -------
    X_train : list of dict
        Loaded train X data. Each item in the list represents a sample of the dataset. Each sample is represented as follows:
            * ``"filename"``: name of the image to extract the data sample from.
            * ``"dir"``: directory where the image resides.
            * ``"coords"``: dictionary with the coordinates to extract the sample from the image. If ``None`` it implies that a random 
              patch needs to be extracted. Following keys are avaialable:
                * ``"z_start"``: starting point of the patch in Z axis.
                * ``"z_end"``: end point of the patch in Z axis.
                * ``"y_start"``: starting point of the patch in Y axis.
                * ``"y_end"``: end point of the patch in Y axis.
                * ``"x_start"``: starting point of the patch in X axis.
                * ``"x_end"``: end point of the patch in X axis.
            * ``"original_data_shape"``: shape of the image where the samples is extracted (useful for reconstructing it later),
            * ``"shape"``: shape of the sample.
            * ``"img"`` (optional): image sample itself. It is a ndarrray of  ``(y, x, channels)`` in ``2D`` and
              ``(z, y, x, channels)`` in ``3D``. Provided when ``train_in_memory`` is ``True``.
            * ``"gt_associated_id"`` (optional): position of associated ground truth of the sample within its list. Present if 
              ``multiple_raw_images`` is ``True``.
            * ``"parallel_data"``(optional): to ``True`` is the sample is a Zarr/H5 file. Not present otherwise.
            * ``"input_axes"`` (optional): order of the axes in Zarr. Not present in non-Zarr/H5 files.
            * ``"path_in_zarr"``(optional): path where the data resides within the Zarr. Provided when ``multiple_data_within_zarr`` was 
              set in ``train_zarr_data_information``.  

    Y_train : list of dict
        Loaded train Y data. Each item in the list represents a sample of the dataset. Each sample is represented as follows:
            * ``"filename"``: name of the image to extract the data sample from.
            * ``"dir"``: directory where the image resides.
            * ``"coords"``: dictionary with the coordinates to extract the sample from the image. If ``None`` it implies that a random 
              patch needs to be extracted. Following keys are avaialable:
                * ``"z_start"``: starting point of the patch in Z axis.
                * ``"z_end"``: end point of the patch in Z axis.
                * ``"y_start"``: starting point of the patch in Y axis.
                * ``"y_end"``: end point of the patch in Y axis.
                * ``"x_start"``: starting point of the patch in X axis.
                * ``"x_end"``: end point of the patch in X axis.
            * ``"original_data_shape"``: shape of the image where the samples is extracted (useful for reconstructing it later),
            * ``"shape"``: shape of the sample.
            * ``"img"`` (optional): image sample itself. It is a ndarrray of  ``(y, x, channels)`` in ``2D`` and
              ``(z, y, x, channels)`` in ``3D``. Provided when ``train_in_memory`` is ``True``.
            * ``"gt_associated_id"`` (optional): position of associated ground truth of the sample within its list. Present if 
              ``multiple_raw_images`` is ``True``.
            * ``"parallel_data"``(optional): to ``True`` is the sample is a Zarr/H5 file. Not present otherwise.
            * ``"input_axes"`` (optional): order of the axes in Zarr. Not present in non-Zarr/H5 files.
            * ``"path_in_zarr"``(optional): path where the data resides within the Zarr. Provided when ``multiple_data_within_zarr`` was 
              set in ``train_zarr_data_information``.  

    X_val : list of dict
        Loaded validation X data. Each item in the list represents a sample of the dataset. Each sample is represented as follows:
            * ``"filename"``: name of the image to extract the data sample from.
            * ``"dir"``: directory where the image resides.
            * ``"coords"``: dictionary with the coordinates to extract the sample from the image. If ``None`` it implies that a random 
              patch needs to be extracted. Following keys are avaialable:
                * ``"z_start"``: starting point of the patch in Z axis.
                * ``"z_end"``: end point of the patch in Z axis.
                * ``"y_start"``: starting point of the patch in Y axis.
                * ``"y_end"``: end point of the patch in Y axis.
                * ``"x_start"``: starting point of the patch in X axis.
                * ``"x_end"``: end point of the patch in X axis.
            * ``"original_data_shape"``: shape of the image where the samples is extracted (useful for reconstructing it later),
            * ``"shape"``: shape of the sample.
            * ``"img"`` (optional): image sample itself. It is a ndarrray of  ``(y, x, channels)`` in ``2D`` and
              ``(z, y, x, channels)`` in ``3D``. Provided when ``val_in_memory`` is ``True``.
            * ``"gt_associated_id"`` (optional): position of associated ground truth of the sample within its list. Present if 
              ``multiple_raw_images`` is ``True``.
            * ``"parallel_data"``(optional): to ``True`` is the sample is a Zarr/H5 file. Not present otherwise.
            * ``"input_axes"`` (optional): order of the axes in Zarr. Not present in non-Zarr/H5 files.
            * ``"path_in_zarr"``(optional): path where the data resides within the Zarr. Provided when ``multiple_data_within_zarr`` was 
              set in ``val_zarr_data_information``.  

    Y_val : list of dict
        Loaded validation Y data. Each item in the list represents a sample of the dataset. Each sample is represented as follows:
            * ``"filename"``: name of the image to extract the data sample from.
            * ``"dir"``: directory where the image resides.
            * ``"coords"``: dictionary with the coordinates to extract the sample from the image. If ``None`` it implies that a random 
              patch needs to be extracted. Following keys are avaialable:
                * ``"z_start"``: starting point of the patch in Z axis.
                * ``"z_end"``: end point of the patch in Z axis.
                * ``"y_start"``: starting point of the patch in Y axis.
                * ``"y_end"``: end point of the patch in Y axis.
                * ``"x_start"``: starting point of the patch in X axis.
                * ``"x_end"``: end point of the patch in X axis.
            * ``"original_data_shape"``: shape of the image where the samples is extracted (useful for reconstructing it later),
            * ``"shape"``: shape of the sample.
            * ``"img"`` (optional): image sample itself. It is a ndarrray of  ``(y, x, channels)`` in ``2D`` and
              ``(z, y, x, channels)`` in ``3D``. Provided when ``val_in_memory`` is ``True``.
            * ``"gt_associated_id"`` (optional): position of associated ground truth of the sample within its list. Present if 
              ``multiple_raw_images`` is ``True``.
            * ``"parallel_data"``(optional): to ``True`` is the sample is a Zarr/H5 file. Not present otherwise.
            * ``"input_axes"`` (optional): order of the axes in Zarr. Not present in non-Zarr/H5 files.
            * ``"path_in_zarr"``(optional): path where the data resides within the Zarr. Provided when ``multiple_data_within_zarr`` was 
              set in ``val_zarr_data_information``.  
    """
    train_shape_will_change = False
    if train_preprocess_f is not None:
        if train_preprocess_cfg is None:
            raise ValueError("'train_preprocess_cfg' needs to be provided with 'train_preprocess_f'")
        if train_preprocess_cfg.RESIZE.ENABLE:
            train_shape_will_change = True
    val_shape_will_change = False
    if val_preprocess_f is not None:
        if val_preprocess_cfg is None:
            raise ValueError("'val_preprocess_cfg' needs to be provided with 'val_preprocess_f'")
        if val_preprocess_cfg.RESIZE.ENABLE:
            val_shape_will_change = True

    print("### LOAD ###")
    # Disable crops when random_crops_in_DA is selected
    crop = False if random_crops_in_DA else True

    # Check validation
    if val_split > 0 or cross_val:
        create_val_from_train = True
    else:
        create_val_from_train = False

    X_train, Y_train, X_val, Y_val = None, None, None, None

    # Create X_train and Y_train
    train_using_zarr = False
    if not multiple_raw_images:
        ids = sorted(next(os.walk(train_path))[2])
        fids = sorted(next(os.walk(train_path))[1])
        if len(ids) == 0:
            if len(fids) == 0:  # Trying Zarr
                raise ValueError("No images found in dir {}".format(train_path))

            # Working with Zarr
            if not is_3d:
                raise ValueError("Zarr image handle is only available for 3D problems")
            train_using_zarr = True

            X_train = samples_from_zarr(
                list_of_data=fids,
                data_path=train_path,
                zarr_data_info=train_zarr_data_information,
                crop_shape=crop_shape,
                ov=train_ov,
                padding=train_padding,
                is_mask=False,
                is_3d=is_3d,
            )
        else:
            X_train = samples_from_image_list(
                list_of_data=ids,
                data_path=train_path,
                crop=crop,
                crop_shape=crop_shape,
                ov=train_ov,
                padding=train_padding,
                is_mask=False,
                is_3d=is_3d,
                reflect_to_complete_shape=reflect_to_complete_shape,
                convert_to_rgb=convert_to_rgb,
                preprocess_f=train_preprocess_f if train_shape_will_change else None,
                preprocess_cfg=train_preprocess_cfg if train_shape_will_change else None,
            )

        # Extract a list of all training gt images
        if train_mask_path is not None:
            ids = sorted(next(os.walk(train_mask_path))[2])
            fids = sorted(next(os.walk(train_mask_path))[1])
            if len(ids) == 0:
                if len(fids) == 0:  # Trying Zarr
                    raise ValueError("No images found in dir {}".format(train_mask_path))
                Y_train = samples_from_zarr(
                    list_of_data=fids,
                    data_path=train_mask_path,
                    zarr_data_info=train_zarr_data_information,
                    crop_shape=crop_shape,
                    ov=train_ov,
                    padding=train_padding,
                    is_mask=True,
                    is_3d=is_3d,
                )
            else:

                # Calculate shape with upsampling
                if is_3d:
                    real_shape = (
                        crop_shape[0] * y_upscaling[0],
                        crop_shape[1] * y_upscaling[1],
                        crop_shape[2] * y_upscaling[2],
                        crop_shape[3],
                    )
                else:
                    real_shape = (
                        crop_shape[0] * y_upscaling[0],
                        crop_shape[1] * y_upscaling[1],
                        crop_shape[2],
                    )
                Y_train = samples_from_image_list(
                    list_of_data=ids,
                    data_path=train_mask_path,
                    crop=crop,
                    crop_shape=real_shape,
                    ov=train_ov,
                    padding=train_padding,
                    is_mask=True,
                    is_3d=is_3d,
                    reflect_to_complete_shape=reflect_to_complete_shape,
                    convert_to_rgb=convert_to_rgb,
                    preprocess_f=train_preprocess_f if train_shape_will_change else None,
                    preprocess_cfg=train_preprocess_cfg if train_shape_will_change else None,
                )
    else:
        if train_mask_path is None:
            raise ValueError("Implementation error. Contact BiaPy team")

        X_train, Y_train = samples_from_image_list_multiple_raw_one_gt(
            data_path=train_path,
            gt_path=train_mask_path,
            crop_shape=crop_shape,
            ov=train_ov,
            padding=train_padding,
            crop=crop,
            is_3d=is_3d,
            reflect_to_complete_shape=reflect_to_complete_shape,
            convert_to_rgb=convert_to_rgb,
            preprocess_f=train_preprocess_f if train_shape_will_change else None,
            preprocess_cfg=train_preprocess_cfg if train_shape_will_change else None,
        )

    # Check that the shape of all images match
    if train_mask_path is not None:
        gt_id = 0
        for i in range(len(X_train)):
            xshape = X_train[i]["shape"]
            if "gt_associated_id" in X_train[i]:
                yshape = Y_train[X_train[i]["gt_associated_id"]]["shape"]
            else:
                yshape = Y_train[i]["shape"]

            if is_3d:
                upsampled_x_shape = (
                    xshape[0] * y_upscaling[0],
                    xshape[1] * y_upscaling[1],
                    xshape[2] * y_upscaling[2],
                )
            else:
                upsampled_x_shape = (
                    xshape[0] * y_upscaling[0],
                    xshape[1] * y_upscaling[1],
                )
            if upsampled_x_shape != yshape[: len(upsampled_x_shape)]:
                filepath = os.path.join(X_train[i]["dir"], X_train[i]["filename"])
                raise ValueError(
                    f"There is a mismatch between input image and its corresponding ground truth ({upsampled_x_shape} vs "
                    f"{yshape}). Please check the images. Specifically, the sample that doesn't match is within "
                    f"the file: {filepath})"
                )

    if len(train_filter_conds) > 0:
        obj = filter_samples_by_properties(
            X_train,
            is_3d,
            train_filter_conds,
            train_filter_vals,
            train_filter_signs,
            y_filenames=Y_train,
            filter_by_entire_image=filter_by_entire_image,
            zarr_data_information=train_zarr_data_information if train_using_zarr else None,
        )
        if train_mask_path is not None:
            X_train, Y_train = obj
        else:
            X_train = obj
        del obj

    val_using_zarr = False
    if create_val_from_train:
        # Create IDs based on images or samples, depending if we are working with Zarr images or not. This is required to
        # create the validation data
        x_train_files = list(set([os.path.join(x["dir"], x["filename"]) for x in X_train]))
        if len(x_train_files) == 1:
            print("As only one sample was found BiaPy will assume that it is big enough to hold multiple training samples "
                  "so the validation will be created extracting samples from it too.")
        if train_using_zarr or len(x_train_files) == 1:
            x_train_ids = np.array(range(0, len(X_train)))
            if train_mask_path is not None:
                y_train_ids = np.array(range(0, len(Y_train)))
                if not multiple_raw_images and len(x_train_ids) != len(y_train_ids):
                    raise ValueError(
                        f"Raw image number ({len(x_train_ids)}) and ground truth file mismatch ({len(y_train_ids)}). Please check the data!"
                    )
        else:
            x_train_files.sort()
            x_train_ids = np.array(range(0, len(x_train_files)))
            if train_mask_path is not None:
                y_train_files = list(set([os.path.join(x["dir"], x["filename"]) for x in Y_train]))
                y_train_files.sort()
                y_train_ids = np.array(range(0, len(y_train_files)))
                if not multiple_raw_images and len(x_train_ids) != len(y_train_ids):
                    raise ValueError(
                        f"Raw image number ({len(x_train_ids)}) and ground truth file mismatch ({len(y_train_ids)}). Please check the data!"
                    )
                
        val_path = train_path
        val_mask_path = train_mask_path
        val_zarr_data_information = train_zarr_data_information
        val_using_zarr = train_using_zarr
        if not cross_val:
            if train_mask_path is not None:
                x_train_ids, x_val_ids, y_train_ids, y_val_ids = train_test_split(
                    x_train_ids,
                    y_train_ids,
                    test_size=val_split,
                    shuffle=shuffle_val,
                    random_state=seed,
                )
            else:
                x_train_ids, x_val_ids = train_test_split(
                    x_train_ids, test_size=val_split, shuffle=shuffle_val, random_state=seed
                )
        else:
            skf = StratifiedKFold(n_splits=cross_val_nsplits, shuffle=shuffle_val, random_state=seed)
            fold = 1

            y_len = len(y_train_ids) if train_mask_path is not None else len(x_train_ids)
            for t_index, te_index in skf.split(np.zeros(len(x_train_ids)), np.zeros(y_len)):
                if cross_val_fold == fold:
                    x_train_ids, x_val_ids = x_train_ids[t_index], x_train_ids[te_index]
                    if train_mask_path is not None:
                        y_train_ids, y_val_ids = y_train_ids[t_index], y_train_ids[te_index]

                    train_index, test_index = t_index.copy(), te_index.copy()
                    break
                fold += 1

            if len(test_index) > 5:
                print("Fold number {}. Printing the first 5 ids: {}".format(fold, test_index[:5]))
            else:
                print("Fold number {}. Indexes used in cross validation: {}".format(fold, test_index))
            x_val_ids = test_index.copy()

        # It's important to sort them in order to speed up load_images_to_sample_list() process
        x_val_ids.sort()
        x_train_ids.sort()
        # Separate validation from train. Using Zarr the method is simpler as only each sample is processed. In other cases 
        # the validation will be composed by samples that are extracted from the images selected to be part of the validation.
        if train_using_zarr or len(x_train_files) == 1:
            # Create validation data from train.
            X_val = [X_train[i] for i in x_val_ids]
            if Y_train is not None:
                Y_val = [Y_train[i] for i in x_val_ids]

            # Remove val samples from train. 
            X_train = [X_train[i] for i in x_train_ids]
            if Y_train is not None:
                Y_train = [Y_train[i] for i in x_train_ids]
        else:
            # Create validation data from train.
            x_val_files = [x_train_files[i] for i in x_val_ids]
            X_val = [x for x in X_train if os.path.join(x["dir"], x["filename"]) in x_val_files]
            if train_mask_path is not None:
                y_val_files = [y_train_files[i] for i in y_val_ids]
                Y_val = [x for x in Y_train if os.path.join(x["dir"], x["filename"]) in y_val_files]
            # Remove val samples from train. 
            x_train_files = [x_train_files[i] for i in x_train_ids]
            if train_mask_path is not None:
                y_train_files = [y_train_files[i] for i in x_train_ids]
            X_train = [x for x in X_train if os.path.join(x["dir"], x["filename"]) in x_train_files]
            if train_mask_path is not None:
                Y_train = [x for x in Y_train if os.path.join(x["dir"], x["filename"]) in y_train_files]
    else:
        if not multiple_raw_images:
            # Extract a list of all validation images
            val_ids = sorted(next(os.walk(val_path))[2])
            val_fids = sorted(next(os.walk(val_path))[1])
            if len(val_ids) == 0:
                if len(val_fids) == 0:  # Trying Zarr
                    raise ValueError("No images found in dir {}".format(val_path))

                # Working with Zarr
                if not is_3d:
                    raise ValueError("Zarr image handle is only available for 3D problems")
                val_using_zarr = True

                X_val = samples_from_zarr(
                    list_of_data=val_fids,
                    data_path=val_path,
                    zarr_data_info=val_zarr_data_information,
                    crop_shape=crop_shape,
                    ov=val_ov,
                    padding=val_padding,
                    is_mask=False,
                    is_3d=is_3d,
                )
            else:
                X_val = samples_from_image_list(
                    list_of_data=val_ids,
                    data_path=val_path,
                    crop=crop,
                    crop_shape=crop_shape,
                    ov=val_ov,
                    padding=val_padding,
                    is_mask=False,
                    is_3d=is_3d,
                    reflect_to_complete_shape=reflect_to_complete_shape,
                    convert_to_rgb=convert_to_rgb,
                    preprocess_f=val_preprocess_f if val_shape_will_change else None,
                    preprocess_cfg=val_preprocess_cfg if val_shape_will_change else None,
                )

            # Extract a list of all validation gt images
            if val_mask_path is not None:
                val_ids = sorted(next(os.walk(val_mask_path))[2])
                val_fids = sorted(next(os.walk(val_mask_path))[1])
                if len(val_ids) == 0:
                    if len(val_fids) == 0:  # Trying Zarr
                        raise ValueError("No images found in dir {}".format(val_mask_path))

                    # Working with Zarr
                    if not is_3d:
                        raise ValueError("Zarr image handle is only available for 3D problems")

                    Y_val = samples_from_zarr(
                        list_of_data=val_fids,
                        data_path=val_mask_path,
                        zarr_data_info=val_zarr_data_information,
                        crop_shape=crop_shape,
                        ov=val_ov,
                        padding=val_padding,
                        is_mask=True,
                        is_3d=is_3d,
                    )
                else:
                    # Calculate shape with upsampling
                    if is_3d:
                        real_shape = (
                            crop_shape[0] * y_upscaling[0],
                            crop_shape[1] * y_upscaling[1],
                            crop_shape[2] * y_upscaling[2],
                            crop_shape[3],
                        )
                    else:
                        real_shape = (
                            crop_shape[0] * y_upscaling[0],
                            crop_shape[1] * y_upscaling[1],
                            crop_shape[2],
                        )
                    Y_val = samples_from_image_list(
                        list_of_data=val_ids,
                        data_path=val_mask_path,
                        crop=crop,
                        crop_shape=real_shape,
                        ov=val_ov,
                        padding=val_padding,
                        is_mask=True,
                        is_3d=is_3d,
                        reflect_to_complete_shape=reflect_to_complete_shape,
                        convert_to_rgb=convert_to_rgb,
                        preprocess_f=val_preprocess_f if val_shape_will_change else None,
                        preprocess_cfg=val_preprocess_cfg if val_shape_will_change else None,
                    )
        else:
            if val_mask_path is None:
                raise ValueError("Implementation error. Contact BiaPy team")

            X_val, Y_val = samples_from_image_list_multiple_raw_one_gt(
                data_path=val_path,
                gt_path=val_mask_path,
                crop_shape=crop_shape,
                ov=val_ov,
                padding=val_padding,
                crop=crop,
                is_3d=is_3d,
                reflect_to_complete_shape=reflect_to_complete_shape,
                convert_to_rgb=convert_to_rgb,
                preprocess_f=val_preprocess_f if val_shape_will_change else None,
                preprocess_cfg=val_preprocess_cfg if val_shape_will_change else None,
            )

        # Check that the shape of all images match
        if val_mask_path is not None:
            gt_id = 0
            for i in range(len(X_val)):
                xshape = X_val[i]["shape"]
                if "gt_associated_id" in X_val[i]:
                    yshape = Y_val[X_val[i]["gt_associated_id"]]["shape"]
                else:
                    yshape = Y_val[i]["shape"]

                if is_3d:
                    upsampled_x_shape = (
                        xshape[0] * y_upscaling[0],
                        xshape[1] * y_upscaling[1],
                        xshape[2] * y_upscaling[2],
                    )
                else:
                    upsampled_x_shape = (
                        xshape[0] * y_upscaling[0],
                        xshape[1] * y_upscaling[1],
                    )
                if upsampled_x_shape != yshape[: len(upsampled_x_shape)]:
                    filepath = os.path.join(X_val[i]["dir"], X_val[i]["filename"])
                    raise ValueError(
                        f"There is a mismatch between input image and its corresponding ground truth ({upsampled_x_shape} vs "
                        f"{yshape}). Please check the images. Specifically, the sample that doesn't match is within "
                        f"the file {filepath})"
                    )

        if len(val_filter_conds) > 0:
            obj = filter_samples_by_properties(
                X_val,
                is_3d,
                val_filter_conds,
                val_filter_vals,
                val_filter_signs,
                y_filenames=Y_val,
                filter_by_entire_image=filter_by_entire_image,
                zarr_data_information=val_zarr_data_information if val_using_zarr else None,
            )
            if val_mask_path is not None:
                X_val, Y_val = obj
            else:
                X_val = obj
            del obj

        x_val_ids = np.array(range(0, len(X_val)))
        if val_mask_path is not None:
            y_val_ids = np.array(range(0, len(Y_val)))
            if not multiple_raw_images and len(x_val_ids) != len(y_val_ids):
                raise ValueError(
                    f"Raw image number ({len(x_val_ids)}) and ground truth file mismatch ({len(y_val_ids)}). Please check the data!"
                )
        
    if train_in_memory:
        print("* Loading train images . . .")
        load_images_to_sample_list(
            list_of_images=X_train,
            crop_shape=crop_shape,
            reflect_to_complete_shape=reflect_to_complete_shape,
            convert_to_rgb=convert_to_rgb,
            preprocess_cfg=train_preprocess_cfg,
            preprocess_f=train_preprocess_f,
            is_3d=is_3d,
            zarr_data_information=train_zarr_data_information if train_using_zarr else None,
        )
        if train_mask_path is not None:
            print("* Loading train GT . . .")
            if is_3d:
                real_shape = (
                    crop_shape[0] * y_upscaling[0],
                    crop_shape[1] * y_upscaling[1],
                    crop_shape[2] * y_upscaling[2],
                    crop_shape[3],
                )
            else:
                real_shape = (
                    crop_shape[0] * y_upscaling[0],
                    crop_shape[1] * y_upscaling[1],
                    crop_shape[2],
                )
            load_images_to_sample_list(
                list_of_images=Y_train,
                crop_shape=real_shape,
                reflect_to_complete_shape=reflect_to_complete_shape,
                convert_to_rgb=convert_to_rgb,
                preprocess_cfg=train_preprocess_cfg,
                is_mask=is_y_mask,
                preprocess_f=train_preprocess_f,
                is_3d=is_3d,
                zarr_data_information=train_zarr_data_information if train_using_zarr else None,
            )

    if val_in_memory:
        print("* Loading validation images . . .")
        load_images_to_sample_list(
            list_of_images=X_val,
            crop_shape=crop_shape,
            reflect_to_complete_shape=reflect_to_complete_shape,
            convert_to_rgb=convert_to_rgb,
            preprocess_cfg=val_preprocess_cfg,
            preprocess_f=val_preprocess_f,
            is_3d=is_3d,
            zarr_data_information=val_zarr_data_information if val_using_zarr else None,
        )

        if val_mask_path is not None:
            print("* Loading validation GT . . .")
            if is_3d:
                real_shape = (
                    crop_shape[0] * y_upscaling[0],
                    crop_shape[1] * y_upscaling[1],
                    crop_shape[2] * y_upscaling[2],
                    crop_shape[3],
                )
            else:
                real_shape = (
                    crop_shape[0] * y_upscaling[0],
                    crop_shape[1] * y_upscaling[1],
                    crop_shape[2],
                )
            load_images_to_sample_list(
                list_of_images=Y_val,
                crop_shape=real_shape,
                reflect_to_complete_shape=reflect_to_complete_shape,
                convert_to_rgb=convert_to_rgb,
                preprocess_cfg=val_preprocess_cfg,
                is_mask=is_y_mask,
                preprocess_f=val_preprocess_f,
                is_3d=is_3d,
                zarr_data_information=val_zarr_data_information if val_using_zarr else None,
            )

    print("### LOAD RESULTS ###")
    if X_train[0]["coords"] == None:
        print(
            "The samples have not been cropped so they may have different shapes. Because of that only first sample's shape will be printed!"
        )
    X_data_shape = (len(X_train),) + X_train[0]["shape"]
    print("*** Loaded train data shape is: {}".format(X_data_shape))
    if Y_train is not None:
        Y_data_shape = (len(Y_train),) + Y_train[0]["shape"]
        print("*** Loaded train GT shape is: {}".format(Y_data_shape))
    else:
        Y_train = X_train.copy()

    X_data_shape = (len(X_val),) + X_val[0]["shape"]
    print("*** Loaded validation data shape is: {}".format(X_data_shape))
    if Y_val is not None:
        Y_data_shape = (len(Y_val),) + Y_val[0]["shape"]
        print("*** Loaded validation GT shape is: {}".format(Y_data_shape))
    else:
        Y_val = X_val.copy()
    print("### END LOAD ###")

    return X_train, Y_train, X_val, Y_val


def load_and_prepare_test_data(
    test_path,
    test_mask_path,
    multiple_raw_images=False,
):
    """
    Load test data.

    Parameters
    ----------
    test_path : str
        Path to the test data.

    test_mask_path : str
        Path to the test data masks.

    multiple_raw_images : bool, optional
        When a folder of folders for each image is expected. In each of those subfolder different versions of the same image 
        are placed. Visit the following tutorial for a real use case and a more detailed description:
        `Light My Cells <https://biapy.readthedocs.io/en/latest/tutorials/image-to-image/lightmycells.html>`_.
        This is used when ``PROBLEM.IMAGE_TO_IMAGE.MULTIPLE_RAW_ONE_TARGET_LOADER`` is selected. 

    Returns
    -------
    X_train : list of dict
        Loaded train X data. Each item in the list represents a sample of the dataset. Each sample is represented as follows:
            * ``"filename"``: name of the image to extract the data sample from.
            * ``"dir"``: directory where the image resides.

    Y_train : list of dict, optional
        Loaded train Y data. Each item in the list represents a sample of the dataset. Each sample is represented as follows:
            * ``"train_path"``: name of the image to extract the data sample from.
            * ``"dir"``: directory where the image resides.

    test_filenames : list of str
        List of test filenames.
    """

    print("### LOAD ###")

    X_test, Y_test = [], None

    # Just read the images from test folder 
    if not os.path.exists(test_path):
        raise ValueError(f"{test_path} doesn't exist")
    
    ids = sorted(next(os.walk(test_path))[2])
    if not multiple_raw_images or len(ids) > 0:
        fids = sorted(next(os.walk(test_path))[1])
        if len(ids) == 0:
            if len(fids) == 0:  # Trying Zarr
                raise ValueError("No images found in dir {}".format(test_path))
            test_filenames = fids
        else:
            test_filenames = ids

        for i in range(len(test_filenames)):
            sample_dict = {
                "filename": test_filenames[i],
                "dir": test_path,
            }
            X_test.append(sample_dict)

        # Extract a list of all training gt images
        if test_mask_path is not None:
            Y_test = []
            
            if not os.path.exists(test_mask_path):
                raise ValueError(f"{test_mask_path} doesn't exist")
            
            ids = sorted(next(os.walk(test_mask_path))[2])
            fids = sorted(next(os.walk(test_mask_path))[1])
            if len(ids) == 0:
                if len(fids) == 0:  # Trying Zarr
                    raise ValueError("No images found in dir {}".format(test_mask_path))
                selected_ids = fids
            else:
                selected_ids = ids

            for i in range(len(selected_ids)):
                sample_dict = {
                    "filename": selected_ids[i],
                    "dir": test_mask_path,
                }
                Y_test.append(sample_dict)
    else:
        test_filenames = sorted(next(os.walk(test_path))[1])
        if len(test_filenames) == 0:
            raise ValueError("No folders found in dir {}".format(test_path))
        for folder in test_filenames:
            sample_path = os.path.join(test_path, folder)
            ids = sorted(next(os.walk(sample_path))[2])
            if len(ids) == 0:
                raise ValueError("No images found in dir {}".format(sample_path))
            for i in range(len(ids)):
                sample_dict = {
                    "filename": ids[i],
                    "dir": sample_path,
                }
                X_test.append(sample_dict)

        # Extract a list of all training gt images
        if test_mask_path is not None:
            Y_test = []
            fids = sorted(next(os.walk(test_mask_path))[1])
            if len(fids) == 0:
                raise ValueError("No folders found in dir {}".format(test_mask_path))
            for folder in fids:
                sample_path = os.path.join(test_mask_path, folder)
                ids = sorted(next(os.walk(sample_path))[2])
                if len(ids) == 0:
                    raise ValueError("No images found in dir {}".format(sample_path))
                for i in range(len(ids)):
                    sample_dict = {
                        "filename": ids[i],
                        "dir": sample_path,
                    }
                    Y_test.append(sample_dict)

    return X_test, Y_test, test_filenames


def load_and_prepare_cls_test_data(
    test_path,
    use_val_as_test,
    expected_classes,
    crop_shape,
    is_3d=True,
    reflect_to_complete_shape=True,
    convert_to_rgb=False,
    use_val_as_test_info=None,
):
    """
    Load test data.

    Parameters
    ----------
    train_path : str
        Path to the training data.

    use_val_as_test : bool
        Whether to use validation data as test.

    expected_classes : int, optional
        Expected number of classes to be loaded.

    crop_shape : 3D/4D int tuple, optional
        Shape of the crops. E.g. ``(y, x, channels)`` for 2D and ``(z, y, x, channels)`` for 3D.

    is_3d: bool, optional
        Whether the data to load is expected to be 3D or not.

    reflect_to_complete_shape : bool, optional
        Wheter to increase the shape of the dimension that have less size than selected patch size padding it with
        'reflect'.

    convert_to_rgb : bool, optional
        In case RGB images are expected, e.g. if ``crop_shape`` channel is 3, those images that are grayscale are
        converted into RGB.

    use_val_as_test_info : dict, optional
        Additional information to create the test set based on the validation. Used when ``use_val_as_test`` is ``True``.
        The expected keys of the dictionary are as follows:
            * ``"cross_val_samples_ids"``: ids of the validation samples (out of the cross validation).
            * ``"train_path"``: training path, as the data must be extracted from there.
            * ``"selected_fold``": fold selected in cross validation.
            * ``"n_splits"``: folds to create in cross validation.
            * ``"shuffle"``: whether to shuffle the data or not.
            * ``"seed"``: mathematical seed.

    Returns
    -------
    X_test : list of dict
        Loaded test data. Each item in the list represents a sample of the dataset. Each sample is represented as follows:
            * ``"filename"``: name of the image to extract the data sample from.
            * ``"dir"``: directory where the image resides.
            * ``"shape"``: shape of the sample.
            * ``"class_name"``: name of the class.
            * ``"class"``: integer that represents the class (``-1`` if no ground truth provided).

    test_filenames : list of str
        List of test filenames.
    """

    print("### LOAD ###")

    X_test = []

    if not use_val_as_test:
        path_to_process = test_path
    else:
        assert use_val_as_test_info is not None, "use_val_as_test_info can not be None when use_val_as_test is True"
        path_to_process = use_val_as_test_info["train_path"]

    X_test = samples_from_class_list(
        data_path=path_to_process,
        expected_classes=expected_classes,
        crop_shape=crop_shape,
        is_3d=is_3d,
        reflect_to_complete_shape=reflect_to_complete_shape,
        convert_to_rgb=convert_to_rgb,
    )
    test_filenames = [os.path.join(x["dir"], x["filename"]) for x in X_test]

    if use_val_as_test:
        # The test is the validation, and as it is only available when validation is obtained from train and when
        # cross validation is enabled, the test set files reside in the train folder
        if use_val_as_test_info["cross_val_samples_ids"] is None:
            x_test_ids = np.array(range(0, len(X_test)))
            # Split the test as it was the validation when train is not enabled
            skf = StratifiedKFold(
                n_splits=use_val_as_test_info["n_splits"],
                shuffle=use_val_as_test_info["shuffle"],
                random_state=use_val_as_test_info["seed"],
            )
            fold = 1
            A = B = np.zeros(len(x_test_ids))
            for _, te_index in skf.split(A, B):
                if use_val_as_test_info["selected_fold"] == fold:
                    use_val_as_test_info["cross_val_samples_ids"] = te_index.copy()
                    break
                fold += 1
            if len(use_val_as_test_info["cross_val_samples_ids"]) > 5:
                print(
                    "Fold number {} used for test data. Printing the first 5 ids: {}".format(
                        fold, use_val_as_test_info["cross_val_samples_ids"][:5]
                    )
                )
            else:
                print(
                    "Fold number {}. Indexes used in cross validation: {}".format(
                        fold, use_val_as_test_info["cross_val_samples_ids"]
                    )
                )

        if use_val_as_test_info["cross_val_samples_ids"] is not None:
            use_val_as_test_info["cross_val_samples_ids"].sort()
            X_test = [X_test[i] for i in use_val_as_test_info["cross_val_samples_ids"]]
            test_filenames = [test_filenames[i] for i in use_val_as_test_info["cross_val_samples_ids"]]

    return X_test, test_filenames


def load_data_from_dir(data_path, is_3d=False, **kwargs):
    """
    Create dataset samples from the given list.

    Parameters
    ----------
    data_path : str
        Path to read the images from.

    is_3d : bool, optional
        Whether if the expected images to read are 3D or not.
    """
    using_zarr = False
    # Create sample list
    if "multiple_raw_images" in kwargs and kwargs["multiple_raw_images"]:
        data_samples = samples_from_image_list_multiple_raw_one_gt(data_path=data_path, **kwargs)
    else:
        if not os.path.exists(data_path):
            raise ValueError(f"{data_path} folder does not exist")
        ids = sorted(next(os.walk(data_path))[2])
        fids = sorted(next(os.walk(data_path))[1])
        if len(ids) == 0:
            if len(fids) == 0:  # Trying Zarr
                raise ValueError("No images found in dir {}".format(data_path))
            else:
                using_zarr = True
                # Working with Zarr
                if not is_3d:
                    raise ValueError("Zarr image handle is only available for 3D problems")
                list_of_images = fids
        else:
            list_of_images = ids

        if list_of_images[0].endswith(".zarr"):
            fname = samples_from_zarr
        else:
            fname = samples_from_image_list

        data_samples = fname(list_of_data=list_of_images, **kwargs)

    print(f"Loading images from {data_path}")
    load_images_to_sample_list(
        list_of_images=data_samples,
        crop_shape=kwargs["crop_shape"] if "crop_shape" in kwargs else None,
        reflect_to_complete_shape=(
            kwargs["reflect_to_complete_shape"] if "reflect_to_complete_shape" in kwargs else False
        ),
        convert_to_rgb=kwargs["convert_to_rgb"] if "convert_to_rgb" in kwargs else False,
        preprocess_cfg=kwargs["preprocess_cfg"] if "preprocess_cfg" in kwargs else None,
        preprocess_f=kwargs["preprocess_f"] if "preprocess_f" in kwargs else None,
        is_3d=is_3d,
        zarr_data_information=(
            kwargs["zarr_data_information"] if using_zarr and "zarr_data_information" in kwargs else None
        ),
    )

    return data_samples


def load_cls_data_from_dir(
    data_path,
    expected_classes,
    crop_shape,
    is_3d=True,
    reflect_to_complete_shape=True,
    convert_to_rgb=False,
    preprocess_f=None,
    preprocess_cfg=None,
):
    """
    Create dataset samples from the given list following a classification workflow directory tree.

    Parameters
    ----------
    data_path : str
        Path to read the images from.

    expected_classes : int, optional
        Expected number of classes to be loaded.

    crop_shape : 3D/4D int tuple, optional
        Shape of the crops. E.g. ``(y, x, channels)`` for 2D and ``(z, y, x, channels)`` for 3D.

    is_3d : bool, optional
        Whether if the expected images to read are 3D or not.

    reflect_to_complete_shape : bool, optional
        Wheter to increase the shape of the dimension that have less size than selected patch size padding it with
        'reflect'.

    convert_to_rgb : bool, optional
        In case RGB images are expected, e.g. if ``crop_shape`` channel is 3, those images that are grayscale are
        converted into RGB.

    preprocess_f : function, optional
        The preprocessing function, is necessary in case you want to apply any preprocessing.

    preprocess_cfg : dict, optional
        Configuration parameters for preprocessing, is necessary in case you want to apply any preprocessing.

    Returns
    -------
    data_samples : list of dicts
        Samples generated out of ``data_path``. Each item in the list represents a sample of the dataset containing:
            * ``"filename"``: name of the image to extract the data sample from.
            * ``"dir"``: directory where the image resides.
            * ``"shape"``: shape of the sample.
            * ``"class_name"``: name of the class.
            * ``"class"``: integer that represents the class (``-1`` if no ground truth provided).
    """
    data_samples = samples_from_class_list(
        data_path=data_path,
        expected_classes=expected_classes,
        crop_shape=crop_shape,
        is_3d=is_3d,
        reflect_to_complete_shape=reflect_to_complete_shape,
        convert_to_rgb=convert_to_rgb,
    )

    print(f"Loading images from {data_path}")
    load_images_to_sample_list(
        list_of_images=data_samples,
        crop_shape=crop_shape,
        reflect_to_complete_shape=reflect_to_complete_shape,
        convert_to_rgb=convert_to_rgb,
        preprocess_cfg=preprocess_cfg,
        preprocess_f=preprocess_f,
        is_3d=is_3d,
    )

    return data_samples


def load_and_prepare_train_data_cls(
    train_path,
    train_in_memory,
    val_path,
    val_in_memory,
    expected_classes,
    cross_val=False,
    cross_val_nsplits=5,
    cross_val_fold=1,
    val_split=0.1,
    seed=0,
    shuffle_val=True,
    train_preprocess_f=None,
    train_preprocess_cfg=None,
    train_filter_conds=[],
    train_filter_vals=None,
    train_filter_signs=None,
    val_preprocess_f=None,
    val_preprocess_cfg=None,
    val_filter_conds=[],
    val_filter_vals=None,
    val_filter_signs=None,
    crop_shape=None,
    reflect_to_complete_shape=False,
    convert_to_rgb=False,
    is_3d=False,
):
    """
    Load data to train classification methods.

    Parameters
    ----------
    train_path : str
        Path to the training data.

    train_in_memory : str
        Whether the train data must be loaded in memory or not.

    val_path : str
        Path to the validation data.

    val_in_memory : str
        Whether the validation data must be loaded in memory or not.

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

    train_preprocess_f : function, optional
        The train preprocessing function, is necessary in case you want to apply any preprocessing.

    train_preprocess_cfg : dict, optional
        Configuration parameters for train preprocessing, is necessary in case you want to apply any preprocessing.

    train_filter_conds : list of lists of str
        Filter conditions to be applied to the train data. The three variables, ``filter_conds``, ``filter_vals`` and ``filter_vals``
        will compose a list of conditions to remove the samples from the list. They are list of list of conditions. For instance, the
        conditions can be like this: ``[['A'], ['B','C']]``. Then, if the sample satisfies the first list of conditions, only 'A'
        in this first case (from ['A'] list), or satisfy 'B' and 'C' (from ['B','C'] list) it will be removed. In each sublist all the
        conditions must be satisfied. Available properties are: [``'foreground'``, ``'mean'``, ``'min'``, ``'max'``].
        Each property descrition:
          * ``'foreground'`` is defined as the mask foreground percentage.
          * ``'mean'`` is defined as the mean value.
          * ``'min'`` is defined as the min value.
          * ``'max'`` is defined as the max value.

    train_filter_vals : list of int/float
        Represent the values of the properties listed in ``train_filter_conds`` that the images need to satisfy to not be dropped.

    train_filter_signs : list of list of str
        Signs to do the comparison for train data filtering. Options: [``'gt'``, ``'ge'``, ``'lt'``, ``'le'``] that corresponds to
        "greather than", e.g. ">", "greather equal", e.g. ">=", "less than", e.g. "<", and "less equal" e.g. "<=" comparisons.

    val_preprocess_f : function, optional
        The validation preprocessing function, is necessary in case you want to apply any preprocessing.

    val_preprocess_cfg : dict, optional
        Configuration parameters for validation preprocessing, is necessary in case you want to apply any preprocessing.

    val_filter_conds : list of lists of str
        Filter conditions to be applied to the validation data. The three variables, ``filter_conds``, ``filter_vals`` and ``filter_vals``
        will compose a list of conditions to remove the images from the list. They are list of list of conditions. For instance, the
        conditions can be like this: ``[['A'], ['B','C']]``. Then, if the sample satisfies the first list of conditions, only 'A'
        in this first case (from ['A'] list), or satisfy 'B' and 'C' (from ['B','C'] list) it will be removed. In each sublist all
        the conditions must be satisfied. Available properties are: [``'foreground'``, ``'mean'``, ``'min'``, ``'max'``].
        Each property descrition:
          * ``'foreground'`` is defined as the mask foreground percentage.
          * ``'mean'`` is defined as the mean value.
          * ``'min'`` is defined as the min value.
          * ``'max'`` is defined as the max value.

    val_filter_vals : list of int/float
        Represent the values of the properties listed in ``val_filter_conds`` that the images need to satisfy to not be dropped.

    val_filter_signs : list of list of str
        Signs to do the comparison for validation data filtering. Options: [``'gt'``, ``'ge'``, ``'lt'``, ``'le'``] that corresponds to
        "greather than", e.g. ">", "greather equal", e.g. ">=", "less than", e.g. "<", and "less equal" e.g. "<=" comparisons.

    crop_shape : 3D/4D int tuple, optional
        Shape of the crops. E.g. ``(y, x, channels)`` for 2D and ``(z, y, x, channels)`` for 3D.

    reflect_to_complete_shape : bool, optional
        Wheter to increase the shape of the dimension that have less size than selected patch size padding it with
        'reflect'.

    convert_to_rgb : bool, optional
        In case RGB images are expected, e.g. if ``crop_shape`` channel is 3, those images that are grayscale are
        converted into RGB.

    is_3d : bool, optional
        Whether if the expected images to read are 3D or not.

    Returns
    -------
    X_train : list of dict
        Loaded train data. Each item in the list represents a sample of the dataset. Each sample is represented as follows:
            * ``"filename"``: name of the image to extract the data sample from.
            * ``"dir"``: directory where the image resides.
            * ``"shape"``: shape of the sample.
            * ``"class_name"``: name of the class.
            * ``"class"``: integer that represents the class (``-1`` if no ground truth provided).
            * ``"img"`` (optional): image sample itself. It is a ndarrray of  ``(y, x, channels)`` in ``2D`` and
              ``(z, y, x, channels)`` in ``3D``. Provided when ``val_in_memory`` is ``True``.

    X_val : list of dict
        Loaded validation data. Each item in the list represents a sample of the dataset. Each sample is represented as follows:
            * ``"filename"``: name of the image to extract the data sample from.
            * ``"dir"``: directory where the image resides.
            * ``"shape"``: shape of the sample.
            * ``"class_name"``: name of the class.
            * ``"class"``: integer that represents the class (``-1`` if no ground truth provided).
            * ``"img"`` (optional): image sample itself. It is a ndarrray of  ``(y, x, channels)`` in ``2D`` and
              ``(z, y, x, channels)`` in ``3D``. Provided when ``val_in_memory`` is ``True``.

    x_val_ids : list of int
        Indexes of the samples beloging to the validation. Used in cross-validation.
    """
    print("### LOAD ###")
    # Check validation
    if val_split > 0 or cross_val:
        create_val_from_train = True
    else:
        create_val_from_train = False

    X_train, X_val = None, None

    X_train = samples_from_class_list(
        data_path=train_path,
        expected_classes=expected_classes,
        crop_shape=crop_shape,
        is_3d=is_3d,
        reflect_to_complete_shape=reflect_to_complete_shape,
        convert_to_rgb=convert_to_rgb,
    )

    if len(train_filter_conds) > 0:
        X_train = filter_samples_by_properties(
            X_train,
            is_3d,
            train_filter_conds,
            train_filter_vals,
            train_filter_signs,
        )

    x_train_ids = np.array(range(0, len(X_train)))
    y_train_ids = np.array([x["class"] for x in X_train])
    if create_val_from_train:
        val_path = train_path
        if not cross_val:
            x_train_ids, x_val_ids = train_test_split(
                x_train_ids, test_size=val_split, shuffle=shuffle_val, random_state=seed
            )
        else:
            skf = StratifiedKFold(n_splits=cross_val_nsplits, shuffle=shuffle_val, random_state=seed)
            fold = 1

            for t_index, te_index in skf.split(x_train_ids, y_train_ids):
                if cross_val_fold == fold:
                    x_train_ids, x_val_ids = x_train_ids[t_index], x_train_ids[te_index]
                    train_index, test_index = t_index.copy(), te_index.copy()
                    break
                fold += 1

            if len(test_index) > 5:
                print("Fold number {}. Printing the first 5 ids: {}".format(fold, test_index[:5]))
            else:
                print("Fold number {}. Indexes used in cross validation: {}".format(fold, test_index))
            x_val_ids = test_index.copy()

        # Create validation data from train. It's important to sort them in order to speed up load_images_to_sample_list() process
        x_val_ids.sort()
        X_val = [X_train[i] for i in x_val_ids]

        # Remove val samples from train. It's important to sort them in order to speed up load_images_to_sample_list() process
        x_train_ids.sort()
        X_train = [X_train[i] for i in x_train_ids]
    else:
        X_val = samples_from_class_list(
            data_path=val_path,
            expected_classes=expected_classes,
            crop_shape=crop_shape,
            is_3d=is_3d,
            reflect_to_complete_shape=reflect_to_complete_shape,
            convert_to_rgb=convert_to_rgb,
        )

        if len(val_filter_conds) > 0:
            X_val = filter_samples_by_properties(
                X_val,
                is_3d,
                val_filter_conds,
                val_filter_vals,
                val_filter_signs,
            )

        x_val_ids = np.array(range(0, len(X_val)))

    if train_in_memory:
        print("* Loading train images . . .")
        load_images_to_sample_list(
            list_of_images=X_train,
            crop_shape=crop_shape,
            reflect_to_complete_shape=reflect_to_complete_shape,
            convert_to_rgb=convert_to_rgb,
            preprocess_cfg=train_preprocess_cfg,
            preprocess_f=train_preprocess_f,
            is_3d=is_3d,
        )

    if val_in_memory:
        print("* Loading validation images . . .")
        load_images_to_sample_list(
            list_of_images=X_val,
            crop_shape=crop_shape,
            reflect_to_complete_shape=reflect_to_complete_shape,
            convert_to_rgb=convert_to_rgb,
            preprocess_cfg=val_preprocess_cfg,
            preprocess_f=val_preprocess_f,
            is_3d=is_3d,
        )

    print("### LOAD RESULTS ###")
    X_data_shape = (len(X_train),) + X_train[0]["shape"]
    print("*** Loaded train data shape is: {}".format(X_data_shape))
    X_data_shape = (len(X_val),) + X_val[0]["shape"]
    print("*** Loaded validation data shape is: {}".format(X_data_shape))
    print("### END LOAD ###")

    return X_train, X_val, x_val_ids


def samples_from_image_list(
    list_of_data,
    data_path,
    crop_shape,
    ov,
    padding,
    crop=True,
    is_mask=False,
    is_3d=True,
    reflect_to_complete_shape=True,
    convert_to_rgb=False,
    preprocess_f=None,
    preprocess_cfg=None,
):
    """
    Create dataset samples from the given list. This function does not load the data.

    Parameters
    ----------
    list_of_data : list of str
        Filenames of the images to read.

    data_path : str
        Directory of the images to read.

    crop_shape : 3D/4D int tuple
        Shape of the crops. E.g. ``(y, x, channels)`` for 2D and ``(z, y, x, channels)`` for 3D.

    ov : 2D/3D float tuple
        Amount of minimum overlap on x and y dimensions. The values must be on range ``[0, 1)``,
        that is, ``0%`` or ``99%`` of overlap. Shape is ``(y, x)`` for 2D or ``(z, y, x)`` for 3D.

    padding : 2D/3D int tuple
        Size of padding to be added on each axis. Shape is ``(y, x)`` for 2D or ``(z, y, x)`` for 3D.

    crop : bool, optional
        Whether if the data needs to be cropped or not.

    is_mask : bool, optional
        Whether the data are masks. It is used to control the preprocessing of the data.

    is_3d: bool, optional
        Whether the data to load is expected to be 3D or not.

    reflect_to_complete_shape : bool, optional
        Wheter to increase the shape of the dimension that have less size than selected patch size padding it with
        'reflect'.

    convert_to_rgb : bool, optional
        In case RGB images are expected, e.g. if ``crop_shape`` channel is 3, those images that are grayscale are
        converted into RGB.

    preprocess_f : function, optional
        The preprocessing function, is necessary in case you want to apply any preprocessing.

    preprocess_cfg : dict, optional
        Configuration parameters for preprocessing, is necessary in case you want to apply any preprocessing.

    Returns
    -------
    sample_list : list of dicts
        Samples generated out of ``list_of_data``. Each item in the list represents a sample of the dataset containing:
            * ``"filename"``: name of the image to extract the data sample from
            * ``"dir"``: directory where the image resides
            * ``"coords"``: dictionary with the coordinates to extract the sample from the image. If ``None`` it implies that a random 
              patch needs to be extracted. Following keys are avaialable:
                * ``"z_start"``: starting point of the patch in Z axis.
                * ``"z_end"``: end point of the patch in Z axis.
                * ``"y_start"``: starting point of the patch in Y axis.
                * ``"y_end"``: end point of the patch in Y axis.
                * ``"x_start"``: starting point of the patch in X axis.
                * ``"x_end"``: end point of the patch in X axis.
            * ``"original_data_shape"``: shape of the image where the samples is extracted (useful for reconstructing it later)
            * ``"shape"``: shape of the sample.
    """
    if preprocess_f is not None and preprocess_cfg is None:
        raise ValueError("'preprocess_cfg' needs to be provided with 'preprocess_f'")

    crop_funct = crop_3D_data_with_overlap if is_3d else crop_data_with_overlap
    sample_list = []
    channel_expected = -1
    data_range_expected = -1

    for i in range(len(list_of_data)):
        # Read image
        img_path = os.path.join(data_path, list_of_data[i])
        img, _ = load_img_data(img_path, is_3d=is_3d)

        # Apply preprocessing
        if preprocess_f is not None:
            if is_mask:
                img = preprocess_f(preprocess_cfg, y_data=[img], is_2d=not is_3d, is_y_mask=is_mask)[0]
            else:
                img = preprocess_f(preprocess_cfg, x_data=[img], is_2d=not is_3d)[0]
        if reflect_to_complete_shape:
            img = pad_and_reflect(img, crop_shape, verbose=False)

        if crop_shape[-1] == 3 and convert_to_rgb and img.shape[-1] != 3:
            img = np.repeat(img, 3, axis=-1)

        # Channel check within dataset images
        if channel_expected == -1:
            channel_expected = img.shape[-1]
        if img.shape[-1] != channel_expected:
            raise ValueError(
                f"All images need to have the same number of channels and represent same information to "
                "ensure the deep learning model can be trained correctly. However, the current image (with "
                f"{channel_expected} channels) appears to have a different number of channels than the first image"
                f"(with {img.shape[-1]} channels) in the folder. Current image: {img_path}"
            )

        # Channel check compared with crop_shape
        if not is_mask:
            if crop_shape[-1] != img.shape[-1]:
                raise ValueError(
                    "Channel of the patch size given {} does not correspond with the loaded image {}. "
                    "Please, check the channels of the images!".format(crop_shape[-1], img.shape[-1])
                )

        # Data range check
        if not is_mask:
            if data_range_expected == -1:
                data_range_expected = data_range(img)
            drange = data_range(img)
            if data_range_expected != drange:
                raise ValueError(
                    f"All images must be within the same data range. However, the current image (with a "
                    f"range of {drange}) appears to be in a different data range than the first image (with a range "
                    f"of {data_range_expected}) in the folder. Current image: {img_path}"
                )

        original_data_shape = img.shape
        crop_coords = None

        if crop and img.shape != crop_shape[:-1] + (img.shape[-1],):
            crop_coords = crop_funct(
                np.expand_dims(img, axis=0) if not is_3d else img,
                crop_shape[:-1] + (img.shape[-1],),
                overlap=ov,
                padding=padding,
                verbose=False,
                load_data=False,
            )
            img_shape = crop_shape[:-1] + (img.shape[-1],)
            tot_samples_to_insert = len(crop_coords)
        else:
            img_shape = img.shape
            tot_samples_to_insert = 1

        for j in range(tot_samples_to_insert):
            sample_dict = {
                "filename": list_of_data[i],
                "dir": data_path,
                "coords": crop_coords[j] if crop_coords is not None else None,
                "original_data_shape": original_data_shape,
                "shape": img_shape,
            }
            sample_list.append(sample_dict)

    return sample_list


def samples_from_zarr(list_of_data, data_path, zarr_data_info, crop_shape, ov, padding, is_mask=False, is_3d=True):
    """
    Create dataset samples from the given list. This function does not load the data.

    Parameters
    ----------
    list_of_data : list of str
        Filenames of the images to read.

    data_path : str
        Directory of the images to read.

    zarr_data_info : dict, optional
        Additional information when using Zarr/H5 files for training. The following keys are expected:
            * ``"raw_path"``: path where the raw images reside within the zarr (used when ``multiple_data_within_zarr`` is ``True``).
            * ``"gt_path"``: path where the mask images reside within the zarr (used when ``multiple_data_within_zarr`` is ``True``).
            * ``"multiple_data_within_zarr"``: Whether if your input Zarr contains the raw images and labels together or not.
            * ``"input_img_axes"``: order of the axes of the images.
            * ``"input_mask_axes"``: order of the axes of the masks.

    crop_shape : 3D/4D int tuple, optional
        Shape of the crops. E.g. ``(y, x, channels)`` for 2D and ``(z, y, x, channels)`` for 3D.

    ov : 2D/3D float tuple, optional
        Amount of minimum overlap on x and y dimensions. The values must be on range ``[0, 1)``,
        that is, ``0%`` or ``99%`` of overlap. Shape is ``(y, x)`` for 2D or ``(z, y, x)`` for 3D.

    padding : 2D/3D int tuple, optional
        Size of padding to be added on each axis. Shape is ``(y, x)`` for 2D or ``(z, y, x)`` for 3D.

    is_mask : bool, optional
        Whether the data are masks. It is used to control the preprocessing of the data.

    is_3d: bool, optional
        Whether the data to load is expected to be 3D or not.

    Returns
    -------
    sample_list : list of dicts
        Samples generated out of ``list_of_data``. Each item in the list represents a sample of the dataset containing:
            * ``"filename"``: name of the image to extract the data sample from.
            * ``"dir"``: directory where the image resides.
            * ``"coords"``: dictionary with the coordinates to extract the sample from the image. If ``None`` it implies that a random 
              patch needs to be extracted. Following keys are avaialable:
                * ``"z_start"``: starting point of the patch in Z axis.
                * ``"z_end"``: end point of the patch in Z axis.
                * ``"y_start"``: starting point of the patch in Y axis.
                * ``"y_end"``: end point of the patch in Y axis.
                * ``"x_start"``: starting point of the patch in X axis.
                * ``"x_end"``: end point of the patch in X axis.
            * ``"original_data_shape"``: shape of the image where the samples is extracted (useful for reconstructing it later).
            * ``"shape"``: shape of the sample.
            * ``"parallel_data"``: to ``True`` always as the sample is a Zarr/H5 file.
            * ``"input_axes"``: order of the axes in Zarr.
            * ``"path_in_zarr"``(optional): path where the data resides within the Zarr. Present if ``multiple_data_within_zarr`` was 
              provided in ``zarr_data_info``.  
    """
    # Extract a list of all training samples within the Zarr
    sample_list = []
    channel_expected = -1
    for i in range(len(list_of_data)):
        sample_path = os.path.join(data_path, list_of_data[i])
        data_within_zarr_path = None
        if zarr_data_info["multiple_data_within_zarr"]:
            if not is_mask:
                data_within_zarr_path = zarr_data_info["raw_path"] 
            else:
                data_within_zarr_path = zarr_data_info["gt_path"] if zarr_data_info["use_gt_path"] else None
        
        data, file = load_img_data(sample_path, is_3d=is_3d, data_within_zarr_path=data_within_zarr_path)
        key_to_check = "input_img_axes" if not is_mask else "input_mask_axes"
        if "C" in zarr_data_info[key_to_check]:
            pos = zarr_data_info[key_to_check].index("C")
            channel = data.shape[pos] if pos < len(data.shape) else 1
        else:
            channel = 1

        # Channel check within dataset images
        if channel_expected == -1:
            channel_expected = channel
        if channel != channel_expected:
            raise ValueError(
                f"All images need to have the same number of channels and represent same information to "
                "ensure the deep learning model can be trained correctly. However, the current image (with "
                f"{channel_expected} channels) appears to have a different number of channels than the first image"
                f"(with {channel} channels) in the folder. Current image: {sample_path}"
            )

        # Channel check compared with crop_shape
        if crop_shape is not None and not is_mask:
            if crop_shape[-1] != channel:
                raise ValueError(
                    "Channel of the patch size given {} does not correspond with the loaded image {}. "
                    "Please, check the channels of the images!".format(crop_shape[-1], channel)
                )

        # Get the total patches so we can use tqdm so the user can see the time
        obj = extract_3D_patch_with_overlap_yield(
            data,
            crop_shape[:-1] + (channel,),
            zarr_data_info["input_img_axes"] if not is_mask else zarr_data_info["input_mask_axes"],
            overlap=ov,
            padding=padding,
            total_ranks=1,
            rank=0,
            return_only_stats=True,
            load_data=False,
            verbose=False,
        )
        __unnamed_iterator = iter(obj)
        while True:
            try:
                obj = next(__unnamed_iterator)
            except StopIteration:  # StopIteration caught here without inspecting it
                break
        del __unnamed_iterator
        total_patches, _, _ = obj

        for_img_cond = extract_3D_patch_with_overlap_yield(
            data,
            crop_shape[:-1] + (channel,),
            zarr_data_info["input_img_axes"] if not is_mask else zarr_data_info["input_mask_axes"],
            overlap=ov,
            padding=padding,
            total_ranks=1,
            load_data=False,
            rank=0,
            verbose=False,
        )
        for obj in tqdm(for_img_cond, total=total_patches, disable=not is_main_process()):
            coords, _, _, _ = obj

            # Crete crop_shape from coords as the sample is not loaded to speed up the process
            if is_3d:
                crop_shape = (
                    coords["z_end"] - coords["z_start"],
                    coords["y_end"] - coords["y_start"],
                    coords["x_end"] - coords["x_start"],
                    channel,
                )
            else:
                crop_shape = (
                    coords["y_end"] - coords["y_start"],
                    coords["x_end"] - coords["x_start"],
                    channel,
                )

            sample_dict = {
                "filename": list_of_data[i],
                "dir": data_path,
                "coords": coords,
                "original_data_shape": data.shape,
                "shape": crop_shape,
                "parallel_data": True,
                "input_axes": zarr_data_info["input_img_axes"] if not is_mask else zarr_data_info["input_mask_axes"],
            }
            if data_within_zarr_path is not None:
                sample_dict["path_in_zarr"] = data_within_zarr_path

            sample_list.append(sample_dict)

        if isinstance(file, h5py.File):
            file.close()

    return sample_list


def samples_from_image_list_multiple_raw_one_gt(
    data_path,
    gt_path,
    crop_shape,
    ov,
    padding,
    crop=True,
    is_3d=True,
    reflect_to_complete_shape=True,
    convert_to_rgb=False,
    preprocess_f=None,
    preprocess_cfg=None,
):
    """
    Create dataset samples from the given lists. This function does not load the data.

    Parameters
    ----------
    data_path : str
        Directory of the images to read.

    gt_path : str
        Directory to read ground truth images from.

    crop_shape : 3D/4D int tuple, optional
        Shape of the crops. E.g. ``(y, x, channels)`` for 2D and ``(z, y, x, channels)`` for 3D.

    ov : 2D/3D float tuple, optional
        Amount of minimum overlap on x and y dimensions. The values must be on range ``[0, 1)``,
        that is, ``0%`` or ``99%`` of overlap. Shape is ``(y, x)`` for 2D or ``(z, y, x)`` for 3D.

    padding : 2D/3D int tuple, optional
        Size of padding to be added on each axis. Shape is ``(y, x)`` for 2D or ``(z, y, x)`` for 3D.

    crop : bool, optional
        Whether if the data needs to be cropped or not.

    is_3d: bool, optional
        Whether the data to load is expected to be 3D or not.

    reflect_to_complete_shape : bool, optional
        Wheter to increase the shape of the dimension that have less size than selected patch size padding it with
        'reflect'.

    convert_to_rgb : bool, optional
        In case RGB images are expected, e.g. if ``crop_shape`` channel is 3, those images that are grayscale are
        converted into RGB.

    preprocess_f : function, optional
        The preprocessing function, is necessary in case you want to apply any preprocessing.

    preprocess_cfg : dict, optional
        Configuration parameters for preprocessing, is necessary in case you want to apply any preprocessing.

    Returns
    -------
    sample_list : list of dicts
        Samples generated out of ``data_path``. Each item in the list represents a sample of the dataset containing:
            * ``"filename"``: name of the image to extract the data sample from.
            * ``"dir"``: directory where the image resides.
            * ``"coords"``: dictionary with the coordinates to extract the sample from the image. If ``None`` it implies that a random 
              patch needs to be extracted. Following keys are avaialable:
                * ``"z_start"``: starting point of the patch in Z axis.
                * ``"z_end"``: end point of the patch in Z axis.
                * ``"y_start"``: starting point of the patch in Y axis.
                * ``"y_end"``: end point of the patch in Y axis.
                * ``"x_start"``: starting point of the patch in X axis.
                * ``"x_end"``: end point of the patch in X axis.
            * ``"original_data_shape"``: shape of the image where the samples is extracted (useful for reconstructing it later).
            * ``"shape"``: shape of the sample.
            * ``"gt_associated_id"``: position of associated ground truth of the sample within its list.

    gt_sample_list : list of dicts
        Samples generated out of ``gt_path``. Each item in the list represents a sample of the dataset containing:
            * ``"filename"``: name of the image to extract the data sample from.
            * ``"dir"``: directory where the image resides.
            * ``"coords"``: dictionary with the coordinates to extract the sample from the image. If ``None`` it implies that a random 
              patch needs to be extracted. Following keys are avaialable:
                * ``"z_start"``: starting point of the patch in Z axis.
                * ``"z_end"``: end point of the patch in Z axis.
                * ``"y_start"``: starting point of the patch in Y axis.
                * ``"y_end"``: end point of the patch in Y axis.
                * ``"x_start"``: starting point of the patch in X axis.
                * ``"x_end"``: end point of the patch in X axis.
            * ``"original_data_shape"``: shape of the image where the samples is extracted (useful for reconstructing it later).
            * ``"shape"``: shape of the sample.
            * ``"gt_associated_id"``: position of associated ground truth of the sample within its list.
    """
    if preprocess_f is not None and preprocess_cfg is None:
        raise ValueError("'preprocess_cfg' needs to be provided with 'preprocess_f'")

    crop_funct = crop_3D_data_with_overlap if is_3d else crop_data_with_overlap
    data_gt_path = sorted(next(os.walk(gt_path))[1])
    sample_list = []
    gt_sample_list = []
    filenames = []

    if len(data_gt_path) == 0:
        raise ValueError("No image folder found in dir {}".format(data_gt_path))

    gt_sample_channel_expected = -1
    gt_sample_data_range_expected = -1
    raw_sample_channel_expected = -1
    raw_sample_data_range_expected = -1
    cont = 0
    for id_ in tqdm(data_gt_path, total=len(data_gt_path), disable=not is_main_process()):
        # Read image
        gt_id = sorted(next(os.walk(os.path.join(gt_path, id_)))[2])[0]
        gt_sample_path = os.path.join(gt_path, id_, gt_id)
        filenames.append(gt_sample_path)
        gt_sample, _ = load_img_data(gt_sample_path, is_3d=is_3d)

        # Apply preprocessing
        if preprocess_f is not None:
            gt_sample = preprocess_f(preprocess_cfg, x_data=[gt_sample], is_2d=not is_3d)[0]

        if reflect_to_complete_shape:
            gt_sample = pad_and_reflect(gt_sample, crop_shape, verbose=False)

        if crop_shape[-1] == 3 and convert_to_rgb and gt_sample.shape[-1] != 3:
            gt_sample = np.repeat(gt_sample, 3, axis=-1)

        # Channel check within dataset images
        if gt_sample_channel_expected == -1:
            gt_sample_channel_expected = gt_sample.shape[-1]
        if gt_sample.shape[-1] != gt_sample_channel_expected:
            raise ValueError(
                f"All images need to have the same number of channels and represent same information to "
                "ensure the deep learning model can be trained correctly. However, the current image (with "
                f"{gt_sample_channel_expected} channels) appears to have a different number of channels than the first image"
                f"(with {gt_sample.shape[-1]} channels) in the folder. Current image: {gt_sample_path}"
            )

        # Channel check compared with crop_shape
        if crop_shape is not None:
            if crop_shape[-1] != gt_sample.shape[-1]:
                raise ValueError(
                    "Channel of the patch size given {} does not correspond with the loaded image {}. "
                    "Please, check the channels of the images!".format(crop_shape[-1], gt_sample.shape[-1])
                )

        # Data range check
        if gt_sample_data_range_expected == -1:
            gt_sample_data_range_expected = data_range(gt_sample)
        drange = data_range(gt_sample)
        if gt_sample_data_range_expected != drange:
            raise ValueError(
                f"All images must be within the same data range. However, the current image (with a "
                f"range of {drange}) appears to be in a different data range than the first image (with a range "
                f"of {gt_sample_data_range_expected}) in the folder. Current image: {gt_sample_path}"
            )

        # Extract all raw images for the current gt sample
        associated_raw_image_dir = os.path.join(data_path, id_)
        if not os.path.exists(associated_raw_image_dir):
            raise ValueError(f"Folder {associated_raw_image_dir} with multiple raw images not found.")
        raw_samples = sorted(next(os.walk(associated_raw_image_dir))[2])
        if len(raw_samples) == 0:
            raise ValueError("No image folder found in dir {}".format(raw_samples))

        original_data_shape = gt_sample.shape
        crop_coords = None

        if crop and gt_sample.shape != crop_shape[:-1] + (gt_sample.shape[-1],):
            crop_coords = crop_funct(
                np.expand_dims(gt_sample, axis=0) if not is_3d else gt_sample,
                crop_shape[:-1] + (gt_sample.shape[-1],),
                overlap=ov,
                padding=padding,
                verbose=False,
                load_data=False,
            )
            gt_shape = crop_shape[:-1] + (gt_sample.shape[-1],)
            gt_tot_samples_to_insert = len(crop_coords)
        else:
            gt_shape = gt_sample.shape
            gt_tot_samples_to_insert = 1

        for i in range(gt_tot_samples_to_insert):
            sample_dict = {
                "filename": gt_id,
                "dir": os.path.join(gt_path, id_),
                "coords": crop_coords[i] if crop_coords is not None else None,
                "original_data_shape": original_data_shape,
                "shape": gt_shape,
            }
            gt_sample_list.append(sample_dict)

        # For each gt samples there are multiple raw images
        for raw_sample_id in raw_samples:
            # Read image
            raw_sample_path = os.path.join(associated_raw_image_dir, raw_sample_id)
            raw_sample, _ = load_img_data(raw_sample_path, is_3d=is_3d)

            # Apply preprocessing
            if preprocess_f is not None:
                raw_sample = preprocess_f(preprocess_cfg, x_data=[raw_sample], is_2d=not is_3d)[0]

            if reflect_to_complete_shape:
                raw_sample = pad_and_reflect(raw_sample, crop_shape, verbose=False)

            if crop_shape[-1] == 3 and convert_to_rgb and raw_sample.shape[-1] != 3:
                raw_sample = np.repeat(raw_sample, 3, axis=-1)

            # Channel check within dataset images
            if raw_sample_channel_expected == -1:
                raw_sample_channel_expected = raw_sample.shape[-1]
            if raw_sample.shape[-1] != raw_sample_channel_expected:
                raise ValueError(
                    f"All images need to have the same number of channels and represent same information to "
                    "ensure the deep learning model can be trained correctly. However, the current image (with "
                    f"{raw_sample_channel_expected} channels) appears to have a different number of channels than the first image"
                    f"(with {raw_sample.shape[-1]} channels) in the folder. Current image: {raw_sample_path}"
                )

            # Channel check compared with crop_shape
            if crop_shape is not None:
                if crop_shape[-1] != raw_sample.shape[-1]:
                    raise ValueError(
                        "Channel of the patch size given {} does not correspond with the loaded image {}. "
                        "Please, check the channels of the images!".format(crop_shape[-1], raw_sample.shape[-1])
                    )

            # Data range check
            if raw_sample_data_range_expected == -1:
                raw_sample_data_range_expected = data_range(raw_sample)
            drange = data_range(raw_sample)
            if raw_sample_data_range_expected != drange:
                raise ValueError(
                    f"All images must be within the same data range. However, the current image (with a "
                    f"range of {drange}) appears to be in a different data range than the first image (with a range "
                    f"of {raw_sample_data_range_expected}) in the folder. Current image: {gt_sample_path}"
                )

            original_data_shape = raw_sample.shape
            crop_coords = None

            if crop and raw_sample.shape != crop_shape[:-1] + (raw_sample.shape[-1],):
                crop_coords = crop_funct(
                    np.expand_dims(raw_sample, axis=0) if not is_3d else raw_sample,
                    crop_shape[:-1] + (raw_sample.shape[-1],),
                    overlap=ov,
                    padding=padding,
                    verbose=False,
                    load_data=False,
                )
                raw_shape = crop_shape[:-1] + (raw_sample.shape[-1],)
                tot_samples_to_insert = len(crop_coords)
            else:
                raw_shape = raw_sample.shape
                tot_samples_to_insert = 1

            for i in range(tot_samples_to_insert):
                sample_dict = {
                    "filename": raw_sample_id,
                    "dir": associated_raw_image_dir,
                    "coords": crop_coords[i] if crop_coords is not None else None,
                    "original_data_shape": original_data_shape,
                    "shape": raw_shape,
                    "gt_associated_id": cont+i,  # this extra variable is added
                }
                sample_list.append(sample_dict)

        cont += gt_tot_samples_to_insert

    return sample_list, gt_sample_list


def samples_from_class_list(
    data_path, expected_classes, crop_shape, is_3d=True, reflect_to_complete_shape=True, convert_to_rgb=False
):
    """
    Create dataset samples from the given path taking into account that each subfolder represents a class.
    This function does not load the data.

    Parameters
    ----------
    data_path : str
        Directory of the images to read.

    expected_classes : int, optional
        Expected number of classes to be loaded. Set to -1 if you don't expect any.

    crop_shape : 3D/4D int tuple, optional
        Shape of the crops. E.g. ``(y, x, channels)`` for 2D and ``(z, y, x, channels)`` for 3D.

    is_3d: bool, optional
        Whether the data to load is expected to be 3D or not.

    reflect_to_complete_shape : bool, optional
        Wheter to increase the shape of the dimension that have less size than selected patch size padding it with
        'reflect'.

    convert_to_rgb : bool, optional
        In case RGB images are expected, e.g. if ``crop_shape`` channel is 3, those images that are grayscale are
        converted into RGB.

    Returns
    -------
    sample_list : list of dicts
        Samples generated out of ``data_path``. Each item in the list represents a sample of the dataset containing:
            * ``"filename"``: name of the image to extract the data sample from.
            * ``"dir"``: directory where the image resides.
            * ``"shape"``: shape of the sample.
            * ``"class_name"``: name of the class.
            * ``"class"``: integer that represents the class (``-1`` if no ground truth provided).
    """
    if expected_classes != -1:
        list_of_classes = sorted(next(os.walk(data_path))[1])
        if len(list_of_classes) < 1:
            raise ValueError("There is no folder/class in {}".format(data_path))

        if expected_classes is not None:
            if expected_classes != len(list_of_classes):
                raise ValueError(
                    "Found number of classes ({}) and 'MODEL.N_CLASSES' ({}) must match".format(
                        len(list_of_classes), expected_classes
                    )
                )
            else:
                print("Found {} classes".format(len(list_of_classes)))
        gt_loaded = True
    else:
        list_of_classes = [os.path.basename(data_path)]
        data_path = os.path.dirname(data_path)
        gt_loaded = False

    xsample_list = []
    for c_num, class_name in enumerate(list_of_classes):
        class_folder = os.path.join(data_path, class_name)

        ids = sorted(next(os.walk(class_folder))[2])
        if len(ids) == 0:
            raise ValueError("There are no images in class {}".format(class_folder))

        channel_expected = -1
        data_range_expected = -1
        for id_ in ids:
            # Read image
            img_path = os.path.join(class_folder, id_)
            img, _ = load_img_data(img_path, is_3d=is_3d)

            if reflect_to_complete_shape:
                img = pad_and_reflect(img, crop_shape, verbose=False)

            if crop_shape[-1] == 3 and convert_to_rgb and img.shape[-1] != 3:
                img = np.repeat(img, 3, axis=-1)

            # Channel check within dataset images
            if channel_expected == -1:
                channel_expected = img.shape[-1]
            if img.shape[-1] != channel_expected:
                raise ValueError(
                    f"All images need to have the same number of channels and represent same information to "
                    "ensure the deep learning model can be trained correctly. However, the current image (with "
                    f"{channel_expected} channels) appears to have a different number of channels than the first image"
                    f"(with {img.shape[-1]} channels) in the folder. Current image: {img_path}"
                )

            # Channel check compared with crop_shape
            if crop_shape is not None:
                if crop_shape[-1] != img.shape[-1]:
                    raise ValueError(
                        "Channel of the patch size given {} does not correspond with the loaded image {}. "
                        "Please, check the channels of the images!".format(crop_shape[-1], img.shape[-1])
                    )

            # Data range check
            if data_range_expected == -1:
                data_range_expected = data_range(img)
            drange = data_range(img)
            if data_range_expected != drange:
                raise ValueError(
                    f"All images must be within the same data range. However, the current image (with a "
                    f"range of {drange}) appears to be in a different data range than the first image (with a range "
                    f"of {data_range_expected}) in the folder. Current image: {img_path}"
                )

            sample_dict = {
                "filename": id_,
                "dir": class_folder,
                "shape": img.shape,
                "class_name": class_name,
                "class": c_num if gt_loaded else -1,
            }
            xsample_list.append(sample_dict)

    return xsample_list


def filter_samples_by_properties(
    x_filenames,
    is_3d,
    filter_conds,
    filter_vals,
    filter_signs,
    filter_by_entire_image=True, 
    y_filenames=None,
    zarr_data_information=None,
):
    """
    Filter samples from ``x_filenames`` using defined conditions. The filtering will be done using the images each sample is extracted 
    from. However, if ``zarr_data_info`` is provided the function will assume that Zarr/h5 files are provided, so the filtering will be 
    performed sample by sample.

    Parameters
    ----------
    x_filenames : list of dict
        X samples to process. Each sample is represented as follows:
            * ``"filename"``: name of the image to extract the data sample from.
            * ``"dir"``: directory where the image resides.
            * ``"coords"``: dictionary with the coordinates to extract the sample from the image. If ``None`` it implies that a random 
              patch needs to be extracted. Following keys are avaialable:
                * ``"z_start"``: starting point of the patch in Z axis.
                * ``"z_end"``: end point of the patch in Z axis.
                * ``"y_start"``: starting point of the patch in Y axis.
                * ``"y_end"``: end point of the patch in Y axis.
                * ``"x_start"``: starting point of the patch in X axis.
                * ``"x_end"``: end point of the patch in X axis.         
            * ``"original_data_shape"``: shape of the image where the samples is extracted (useful for reconstructing it later),
            * ``"shape"``: shape of the sample.
            * ``"img"`` (optional): image sample itself. It is a ndarrray of  ``(y, x, channels)`` in ``2D`` and
              ``(z, y, x, channels)`` in ``3D``. Provided when ``train_in_memory`` is ``True``.
            * ``"gt_associated_id"`` (optional): position of associated ground truth of the sample within its list. Present if 
              ``multiple_raw_images`` is ``True``.

    is_3d: bool, optional
        Whether the data to load is expected to be 3D or not.

    filter_conds : list of lists of str
        Filter conditions to be applied. The three variables, ``filter_conds``, ``filter_vals`` and ``filter_vals`` will compose a
        list of conditions to remove the images from the list. They are list of list of conditions. For instance, the conditions can
        be like this: ``[['A'], ['B','C']]``. Then, if the sample satisfies the first list of conditions, only 'A' in this first case
        (from ['A'] list), or satisfy 'B' and 'C' (from ['B','C'] list) it will be removed. In each sublist all the conditions must be
        satisfied. Available properties are: [``'foreground'``, ``'mean'``, ``'min'``, ``'max'``]. Each property descrition:
          * ``'foreground'`` is defined as the mask foreground percentage.
          * ``'mean'`` is defined as the mean value.
          * ``'min'`` is defined as the min value.
          * ``'max'`` is defined as the max value.

    filter_vals : list of int/float
        Represent the values of the properties listed in ``filter_conds`` that the images need to satisfy to not be dropped.

    filter_signs  :list of list of str
        Signs to do the comparison. Options: [``'gt'``, ``'ge'``, ``'lt'``, ``'le'``] that corresponds to "greather than", e.g. ">",
        "greather equal", e.g. ">=", "less than", e.g. "<", and "less equal" e.g. "<=" comparisons.

    filter_by_entire_image : bool, optional
        This decides how the filtering is done:
            * ``True``: apply filter image by image. 
            * ``False``: apply filtering sample by sample. Each sample represents a patch within an image.

    y_filenames : list of dict, optional
        Y samples to process.

    zarr_data_info : dict, optional
        Additional information when using Zarr/H5 files for training. The following keys are expected:
            * ``"raw_path"``: path where the raw images reside within the zarr (used when ``multiple_data_within_zarr`` is ``True``).
            * ``"gt_path"``: path where the mask images reside within the zarr (used when ``multiple_data_within_zarr`` is ``True``).
            * ``"multiple_data_within_zarr"``: Whether if your input Zarr contains the raw images and labels together or not.
            * ``"input_img_axes"``: order of the axes of the images.
            * ``"input_mask_axes"``: order of the axes of the masks.

    Returns
    -------
    new_x_filenames : list of dict
        ``x_filenames`` list filtered.

    new_y_filenames : list of dict, optional
        ``y_filenames`` list filtered.
    """
    # Filter samples by properties
    print("Applying filtering to data samples . . .")

    new_x_filenames = []
    if y_filenames is not None:
        new_y_filenames = []
    
    foreground_filter_requested = any([True for cond in filter_conds if "foreground" in cond])
    if foreground_filter_requested:
        if y_filenames is None:
            raise ValueError("'foreground' condition can not be used for filtering when 'y_filenames' was not provided")
        
    using_zarr = False
    if zarr_data_information is not None:
        using_zarr = True
        print("Assuming we are working with Zarr/H5 images so the filtering will be done patch by patch.")
        print(f"Number of samples before filtering: {len(x_filenames)}")
    else:
        images = list(set([os.path.join(x["dir"], x["filename"]) for x in x_filenames]))
        if foreground_filter_requested:
            masks = list(set([os.path.join(x["dir"], x["filename"]) for x in y_filenames])) 
        print(f"Number of samples before filtering: {len(images)}")

    if not using_zarr and filter_by_entire_image:
        not_discarded_images = []
        for n, image_path in tqdm(enumerate(images), disable=not is_main_process()):
            # Load X data
            img, _ = load_img_data(image_path, is_3d=is_3d)

            # Load Y data
            if foreground_filter_requested:
                mask_path = os.path.join(masks[n]["dir"], masks[n]["filename"])
                mask, _ = load_img_data(mask_path, is_3d=is_3d)
            else:
                mask = None

            satisfy_conds = sample_satisfy_conds(img, filter_conds, filter_vals, filter_signs, mask=mask)

            if not satisfy_conds:
                not_discarded_images.append(image_path)
            else:
                print(f"Discarding file {image_path}")
                
        # Keep only those samples from not discarded images 
        for n, sample in enumerate(x_filenames):
            if os.path.join(sample["dir"], sample["filename"]) in not_discarded_images:
                new_x_filenames.append(x_filenames[n])
                if y_filenames is not None:
                    new_y_filenames.append(y_filenames[n])

        number_of_samples = len(not_discarded_images)
    else:
        img_path, mask_path = "", ""
        file, mfile = None, None               
        for n, sample in tqdm(enumerate(x_filenames), disable=not is_main_process()):
            # Load X data
            filepath = os.path.join(sample["dir"], sample["filename"])
            if img_path != filepath:
                img_path = filepath
                if file is not None and isinstance(file, h5py.File):
                    file.close()
                data_within_zarr_path = zarr_data_information["raw_path"] if zarr_data_information["multiple_data_within_zarr"] else None
                xdata, file = load_img_data(img_path, is_3d=is_3d, data_within_zarr_path=data_within_zarr_path)

                # Load Y data
                if foreground_filter_requested:
                    filepath = os.path.join(y_filenames[n]["dir"], y_filenames[n]["filename"])
                    mask_path = filepath
                    if mfile is not None and isinstance(mfile, h5py.File):
                        mfile.close()
                    data_within_zarr_path = None
                    if zarr_data_information["multiple_data_within_zarr"]:
                        data_within_zarr_path = zarr_data_information["gt_path"] if zarr_data_information["use_gt_path"] else None
                    ydata, mfile = load_img_data(mask_path, is_3d=is_3d, data_within_zarr_path=data_within_zarr_path)
                else:
                    ydata, mfile = None, None

            # Capture patches within image/mask
            coords = sample["coords"]
            if foreground_filter_requested:
                mcoords = y_filenames[n]["coords"]

            # Prepare slices to extract the patch
            if is_3d:
                xslices = (
                    slice(None),
                    slice(coords["z_start"], coords["z_end"]),
                    slice(coords["y_start"], coords["y_end"]),
                    slice(coords["x_start"], coords["x_end"]),
                    slice(None),
                )
            else:
                xslices = (
                    slice(None),
                    slice(coords["y_start"], coords["y_end"]),
                    slice(coords["x_start"], coords["x_end"]),
                    slice(None),
                )

            xdata_ordered_slices = order_dimensions(
                xslices,
                input_order="TZYXC",
                output_order=zarr_data_information["input_img_axes"],
                default_value=0,
            )

            if is_3d:
                yslices = (
                    slice(None),
                    slice(mcoords["z_start"], mcoords["z_end"]),
                    slice(mcoords["y_start"], mcoords["y_end"]),
                    slice(mcoords["x_start"], mcoords["x_end"]),
                    slice(None),
                )
            else:
                yslices = (
                    slice(None),
                    slice(mcoords["y_start"], mcoords["y_end"]),
                    slice(mcoords["x_start"], mcoords["x_end"]),
                    slice(None),
                )
            ydata_ordered_slices = order_dimensions(
                yslices,
                input_order="TZYXC",
                output_order=zarr_data_information["input_mask_axes"],
                default_value=0,
            )

            img = xdata[xdata_ordered_slices]
            if foreground_filter_requested:
                mask = ydata[ydata_ordered_slices]

            satisfy_conds = sample_satisfy_conds(img, filter_conds, filter_vals, filter_signs, mask=mask)

            if not satisfy_conds:
                new_x_filenames.append(x_filenames[n])
                if y_filenames is not None:
                    new_y_filenames.append(y_filenames[n])

            number_of_samples = len(new_x_filenames)

    if number_of_samples == 0:
        raise ValueError(
            "Filters set with 'DATA.TRAIN.FILTER_SAMPLES.*' variables led to discard all training samples. Aborting!"
        )
    elif number_of_samples == 1:
        raise ValueError(
            "Filters set with 'DATA.TRAIN.FILTER_SAMPLES.*' variables led to discard all training samples but one. Aborting!"
        )

    print(f"Number of samples after filtering: {number_of_samples}")
    if y_filenames is not None:
        return new_x_filenames, new_y_filenames
    else:
        return new_x_filenames


def sample_satisfy_conds(img, filter_conds, filter_vals, filter_signs, mask=None):
    """
    Whether ``img`` satisfy at least one of the conditions composed by ``filter_conds``, ``filter_vals``, ``filter_sings``.

    Parameters
    ----------
    img : 4D/5D Numpy array
        Image to check if satisfy conditions. E.g. ``(z, y, x, num_classes)`` for 3D or ``(y, x, num_classes)`` for 2D.

    filter_conds : list of lists of str
        Filter conditions to be applied. The three variables, ``filter_conds``, ``filter_vals`` and ``filter_vals`` will compose a
        list of conditions to remove the images from the list. They are list of list of conditions. For instance, the conditions can
        be like this: ``[['A'], ['B','C']]``. Then, if the sample satisfies the first list of conditions, only 'A' in this first case
        (from ['A'] list), or satisfy 'B' and 'C' (from ['B','C'] list) it will be removed. In each sublist all the conditions must
        be satisfied. Available properties are: [``'foreground'``, ``'mean'``, ``'min'``, ``'max'``]. Each property descrition:
          * ``'foreground'`` is defined as the mask foreground percentage.
          * ``'mean'`` is defined as the mean value.
          * ``'min'`` is defined as the min value.
          * ``'max'`` is defined as the max value.

    filter_vals : list of int/float
        Represent the values of the properties listed in ``filter_conds`` that the images need to satisfy to not be dropped.

    filter_signs  :list of list of str
        Signs to do the comparison. Options: [``'gt'``, ``'ge'``, ``'lt'``, ``'le'``] that corresponds to "greather than", e.g. ">",
        "greather equal", e.g. ">=", "less than", e.g. "<", and "less equal" e.g. "<=" comparisons.

    mask : 4D/5D Numpy array
        Mask to check if satisfy "foreground" condition in ``filter_conds``. E.g. ``(z, y, x, num_classes)`` for 3D or
        ``(y, x, num_classes)`` for 2D.

    Returns
    -------
    satisfy_conds : bool
        Whether if the sample satisfy one of the conditions or not.
    """
    satisfy_conds = False
    # Check if the sample satisfies a condition
    for i, cond in enumerate(filter_conds):
        comps = []
        for j, c in enumerate(cond):
            if c == "foreground":
                labels, npixels = np.unique((mask > 0).astype(np.uint8), return_counts=True)

                total_pixels = 1
                for val in list(mask.shape):
                    total_pixels *= val

                if labels[0] == 0:
                    npixels = npixels[1:]
                value_to_compare = sum(npixels) / total_pixels
            else:
                if c == "min":
                    value_to_compare = img.min()
                elif c == "max":
                    value_to_compare = img.max()
                elif c == "mean":
                    value_to_compare = img.mean()

            # Check each list of conditions
            if filter_signs[i][j] == "gt":
                if value_to_compare > filter_vals[i][j]:
                    comps.append(True)
                else:
                    comps.append(False)
            elif filter_signs[i][j] == "ge":
                if value_to_compare >= filter_vals[i][j]:
                    comps.append(True)
                else:
                    comps.append(False)
            elif filter_signs[i][j] == "lt":
                if value_to_compare < filter_vals[i][j]:
                    comps.append(True)
                else:
                    comps.append(False)
            elif filter_signs[i][j] == "le":
                if value_to_compare <= filter_vals[i][j]:
                    comps.append(True)
                else:
                    comps.append(False)

        # Check if the conditions where satified
        if all(comps):
            satisfy_conds = True
            break

    return satisfy_conds


def load_images_to_sample_list(
    list_of_images,
    crop_shape=None,
    reflect_to_complete_shape=False,
    convert_to_rgb=False,
    preprocess_cfg=None,
    is_mask=False,
    preprocess_f=None,
    is_3d=False,
    zarr_data_information=None,
):
    """
    Load images into the sample list ``list_of_images``: creating ``"img"`` and modifiying ``"shape"`` keys. The process done faster
    if the samples extracted from the same image are in continuous positions within the list.

    Parameters
    ----------
    list_of_images : list of dict
        Loaded data. Each item in the list represents a sample of the dataset. Each sample is represented as follows:
            * ``"filename"``: name of the image to extract the data sample from.
            * ``"dir"``: directory where the image resides.
            * ``"coords"``: dictionary with the coordinates to extract the sample from the image. If ``None`` it implies that a random 
              patch needs to be extracted. Following keys are avaialable:
                * ``"z_start"``: starting point of the patch in Z axis.
                * ``"z_end"``: end point of the patch in Z axis.
                * ``"y_start"``: starting point of the patch in Y axis.
                * ``"y_end"``: end point of the patch in Y axis.
                * ``"x_start"``: starting point of the patch in X axis.
                * ``"x_end"``: end point of the patch in X axis.
            * ``"original_data_shape"``: shape of the image where the samples is extracted (useful for reconstructing it later),
            * ``"shape"``: shape of the sample.
            * ``"img"``: image sample itself. It is a ndarrray of  ``(y, x, channels)`` in ``2D`` and
              ``(z, y, x, channels)`` in ``3D``.

    crop_shape : 3D/4D int tuple, optional
        Shape of the expected crops. E.g. ``(y, x, channels)`` for 2D and ``(z, y, x, channels)`` for 3D.

    reflect_to_complete_shape : bool, optional
        Whether to increase the shape of the dimension that have less size than selected patch size padding it with
        'reflect'.

    convert_to_rgb : bool, optional
        In case RGB images are expected, e.g. if ``crop_shape`` channel is 3, those images that are grayscale are
        converted into RGB.

    preprocess_cfg : dict, optional
        Configuration parameters for preprocessing, is necessary in case you want to apply any preprocessing.

    is_mask : bool, optional
        Whether the data are masks. It is used to control the preprocessing of the data.

    preprocess_f : function, optional
        The preprocessing function, is necessary in case you want to apply any preprocessing.

    is_3d: bool, optional
        Whether the data to load is expected to be 3D or not.

    zarr_data_information : dict, optional
        Additional information of where to find the data within the Zarr files.
    """
    if preprocess_f is not None and preprocess_cfg == None:
        raise ValueError("The preprocessing configuration ('preprocess_cfg') is missing.")

    channel_expected = -1
    data_range_expected = -1
    img_path = ""
    file = None
    for sample in tqdm(list_of_images, total=len(list_of_images), disable=not is_main_process()):
        # Read image if it is different from the last sample's
        filepath = os.path.join(sample["dir"], sample["filename"])
        if img_path != filepath:
            img_path = filepath
            if file is not None and isinstance(file, h5py.File):
                file.close()
            data_within_zarr_path = None
            if zarr_data_information is not None and zarr_data_information["multiple_data_within_zarr"]:
                data_within_zarr_path = (
                    zarr_data_information["raw_path"] if not is_mask else zarr_data_information["gt_path"]
                )
            data, file = load_img_data(img_path, is_3d=is_3d, data_within_zarr_path=data_within_zarr_path)

            # Channel check within dataset images
            if channel_expected == -1:
                channel_expected = data.shape[-1]
            if data.shape[-1] != channel_expected:
                raise ValueError(
                    f"All images need to have the same number of channels and represent same information to "
                    "ensure the deep learning model can be trained correctly. However, the current image (with "
                    f"{channel_expected} channels) appears to have a different number of channels than the first image"
                    f"(with {data.shape[-1]} channels) in the folder. Current image: {img_path}"
                )

            # Channel check compared with crop_shape
            if crop_shape is not None and not is_mask:
                channel_to_compare = data.shape[-1] if not convert_to_rgb else 3
                if crop_shape[-1] != channel_to_compare:
                    if not convert_to_rgb:
                        raise ValueError(
                            "Channel of the patch size given {} does not correspond with the loaded image {}. "
                            "Please, check the channels of the images!".format(crop_shape[-1], channel_to_compare)
                        )
                    else:
                        raise ValueError(
                            "Channel of the patch size given {} does not correspond with the loaded image {} "
                            "(remember that 'DATA.FORCE_RGB' was selected). Please, check the channels of the "
                            "images!".format(crop_shape[-1], channel_to_compare)
                        )
            # Data range check
            if not is_mask and isinstance(data, np.ndarray):
                if data_range_expected == -1:
                    data_range_expected = data_range(data)
                drange = data_range(data)
                if data_range_expected != drange:
                    raise ValueError(
                        f"All images must be within the same data range. However, the current image (with a "
                        f"range of {drange}) appears to be in a different data range than the first image (with a range "
                        f"of {data_range_expected}) in the folder. Current image: {img_path}"
                    )

            # Apply preprocessing
            if preprocess_f is not None:
                if is_mask:
                    data = preprocess_f(preprocess_cfg, y_data=[data], is_2d=not is_3d, is_y_mask=is_mask)[0]
                else:
                    data = preprocess_f(preprocess_cfg, x_data=[data], is_2d=not is_3d)[0]

        # Prepare slices to extract the patch
        if "coords" in sample and sample["coords"] is not None:
            coords = sample["coords"]
            if is_3d:
                xslices = (
                    slice(None),
                    slice(coords["z_start"], coords["z_end"]),
                    slice(coords["y_start"], coords["y_end"]),
                    slice(coords["x_start"], coords["x_end"]),
                    slice(None),
                )
            else:
                xslices = (
                    slice(None),
                    slice(coords["y_start"], coords["y_end"]),
                    slice(coords["x_start"], coords["x_end"]),
                    slice(None),
                )

            if zarr_data_information is not None:
                key = "input_img_axes" if not is_mask else "input_mask_axes"
                data_ordered_slices = order_dimensions(
                    xslices,
                    input_order="TZYXC",
                    output_order=zarr_data_information[key],
                    default_value=0,
                )
            else:
                data_ordered_slices = xslices[1:]

            # Extract the patch within the image
            img = data[data_ordered_slices]

            if zarr_data_information is not None:
                img = ensure_3d_shape(img.squeeze(), path=filepath)
        else:
            img = data

        if reflect_to_complete_shape:
            img = pad_and_reflect(img, crop_shape, verbose=False)

        if crop_shape is not None and crop_shape[-1] == 3 and convert_to_rgb and not is_mask and img.shape[-1] != 3:
            img = np.repeat(img, 3, axis=-1)

        # Insert the image
        sample["img"] = img
        sample["shape"] = img.shape

    if "coords" in list_of_images[0] and list_of_images[0]["coords"] != None:
        data_shape = (len(list_of_images),) + list_of_images[0]["shape"]
        print("*** Loaded data shape is {}".format(data_shape))
    else:
        print("Not all samples seem to have the same shape. Number of samples: {}".format(len(list_of_images)))
        print("*** First sample shape is {}".format(list_of_images[0]["shape"]))


def pad_and_reflect(img, crop_shape, verbose=False):
    """
    Load data from a directory.

    Parameters
    ----------
    img : 3D/4D Numpy array
        Image to pad. E.g. ``(y, x, channels)`` or ``(z, y, x, channels)``.

    crop_shape : Tuple of 3/4 int, optional
        Shape of the subvolumes to create when cropping.  E.g. ``(y, x, channels)`` or ``(z, y, x, channels)``.

    verbose : bool, optional
        Whether to output information.

    Returns
    -------
    img : 3D/4D Numpy array
        Image padded. E.g. ``(y, x, channels)`` for 2D and ``(z, y, x, channels)`` for 3D.
    """
    if img.ndim == 4 and len(crop_shape) != 4:
        raise ValueError(f"'crop_shape' needs to have 4 values as the input array has 4 dims. Provided crop_shape: {crop_shape}")
    if img.ndim == 3 and len(crop_shape) != 3:
        raise ValueError(f"'crop_shape' needs to have 3 values as the input array has 3 dims. Provided crop_shape: {crop_shape}")

    if img.ndim == 4:
        if img.shape[0] < crop_shape[0]:
            diff = crop_shape[0] - img.shape[0]
            o_shape = img.shape
            img = np.pad(img, ((diff, 0), (0, 0), (0, 0), (0, 0)), "reflect")
            if verbose:
                print("Reflected from {} to {}".format(o_shape, img.shape))

        if img.shape[1] < crop_shape[1]:
            diff = crop_shape[1] - img.shape[1]
            o_shape = img.shape
            img = np.pad(img, ((0, 0), (diff, 0), (0, 0), (0, 0)), "reflect")
            if verbose:
                print("Reflected from {} to {}".format(o_shape, img.shape))

        if img.shape[2] < crop_shape[2]:
            diff = crop_shape[2] - img.shape[2]
            o_shape = img.shape
            img = np.pad(img, ((0, 0), (0, 0), (diff, 0), (0, 0)), "reflect")
            if verbose:
                print("Reflected from {} to {}".format(o_shape, img.shape))
    else:
        if img.shape[0] < crop_shape[0]:
            diff = crop_shape[0] - img.shape[0]
            o_shape = img.shape
            img = np.pad(img, ((diff, 0), (0, 0), (0, 0)), "reflect")
            if verbose:
                print("Reflected from {} to {}".format(o_shape, img.shape))

        if img.shape[1] < crop_shape[1]:
            diff = crop_shape[1] - img.shape[1]
            o_shape = img.shape
            img = np.pad(img, ((0, 0), (diff, 0), (0, 0)), "reflect")
            if verbose:
                print("Reflected from {} to {}".format(o_shape, img.shape))
    return img


def img_to_onehot_encoding(img, num_classes=2):
    """
    Converts image given into one-hot encode format.

    The opposite function is :func:`~onehot_encoding_to_img`.

    Parameters
    ----------
    img : Numpy 3D/4D array
        Image. E.g. ``(y, x, channels)`` or ``(z, y, x, channels)``.

    num_classes : int, optional
        Number of classes to distinguish.

    Returns
    -------
    one_hot_labels : Numpy 3D/4D array
        Data one-hot encoded. E.g. ``(y, x, num_classes)`` or ``(z, y, x, num_classes)``.
    """
    if img.ndim == 4:
        shape = img.shape[:3] + (num_classes,)
    else:
        shape = img.shape[:2] + (num_classes,)

    encoded_image = np.zeros(shape, dtype=np.int8)

    for i in range(num_classes):
        if img.ndim == 4:
            encoded_image[:, :, :, i] = np.all(img.reshape((-1, 1)) == i, axis=1).reshape(shape[:3])
        else:
            encoded_image[:, :, i] = np.all(img.reshape((-1, 1)) == i, axis=1).reshape(shape[:2])

    return encoded_image


def onehot_encoding_to_img(encoded_image):
    """
    Converts one-hot encode image into an image with jus tone channel and all the classes represented by an integer.

    The opposite function is :func:`~img_to_onehot_encoding`.

    Parameters
    ----------
    encoded_image : Numpy 3D/4D array
        Image. E.g. ``(y, x, channels)`` or ``(z, y, x, channels)``.

    Returns
    -------
    img : Numpy 3D/4D array
        Data one-hot encoded. E.g. ``(z, y, x, num_classes)``.
    """
    if encoded_image.ndim == 4:
        shape = encoded_image.shape[:3] + (1,)
    else:
        shape = encoded_image.shape[:2] + (1,)

    img = np.zeros(shape, dtype=np.int8)
    for i in range(img.shape[-1]):
        img[encoded_image[..., i] == 1] = i

    return img


def load_img_data(path, is_3d=False, data_within_zarr_path=None):
    """
    Load data from a given path.

    Parameters
    ----------
    path : str
        Path to the image to read.

    is_3d : bool, optional
        Whether if the expected image to read is 3D or not.

    data_within_zarr_path : str, optional
        Path to find the data within the Zarr file. E.g. 'volumes.labels.neuron_ids'.

    Returns
    -------
    data : Zarr, H5 or Numpy 3D/4D array
        Data read. E.g. ``(z, y, x, num_classes)`` for 3D or ``(y, x, num_classes)`` for 2D.

    file : str
        File of the data read. Useful to close it in case it is an H5 file.
    """
    if path.endswith(".zarr") or path.endswith(".hdf5") or path.endswith(".h5"):
        if data_within_zarr_path:
            file, data = read_chunked_nested_data(path, data_within_zarr_path)
        else:
            file, data = read_chunked_data(path)
    else:
        data = read_img_as_ndarray(path, is_3d=is_3d)
        file = path

    return data, file


def read_img_as_ndarray(path, is_3d=False):
    """
    Read an image from a given path.

    Parameters
    ----------
    path : str
        Path to the image to read.

    is_3d : bool, optional
        Whether if the expected image to read is 3D or not.

    Returns
    -------
    img : Numpy 3D/4D array
        Image read. E.g. ``(z, y, x, num_classes)`` for 3D or ``(y, x, num_classes)`` for 2D.
    """
    # Read image
    if path.endswith(".npy"):
        img = np.load(path)
    elif path.endswith(".hdf5") or path.endswith(".h5"):
        img = h5py.File(path, "r")
        img = np.array(img[list(img)[0]])
    elif path.endswith(".zarr"):
        _, img = read_chunked_data(path)
        img = np.array(img)
    else:
        img = imread(path)
    img = np.squeeze(img)

    if not is_3d:
        img = ensure_2d_shape(img, path)
    else:
        img = ensure_3d_shape(img, path)

    return img

def check_value(value, value_range=(0, 1)):
    """
    Checks if a value is within a range
    """
    if isinstance(value, list) or isinstance(value, tuple):
        for i in range(len(value)):
            if isinstance(value[i], np.ndarray):
                if value_range[0] <= np.min(value[i]) or np.max(value[i]) <= value_range[1]:
                    return False
            else:
                if not (value_range[0] <= value[i] <= value_range[1]):
                    return False
        return True
    else:
        if isinstance(value, np.ndarray):
            if value_range[0] <= np.min(value) and np.max(value) <= value_range[1]:
                return True
        else:
            if value_range[0] <= value <= value_range[1]:
                return True
        return False


def data_range(x):
    if not isinstance(x, np.ndarray):
        raise ValueError("Input array of type {} and not numpy array".format(type(x)))
    if check_value(x, (0, 1)):
        return "01 range"
    elif check_value(x, (0, 255)):
        return "uint8 range"
    elif check_value(x, (0, 65535)):
        return "uint16 range"
    else:
        return "none_range"
