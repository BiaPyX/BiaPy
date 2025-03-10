import os
import h5py
import torch
import tifffile
import imageio
import numpy as np
from typing import (
    List,
    Tuple,
    Dict,
    Optional,
    Callable,
    Any,
)
from numpy.typing import NDArray

from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold

from biapy.data.dataset import BiaPyDataset, DatasetFile, DataSample, PatchCoords
from biapy.data.norm import Normalization
from biapy.utils.misc import is_main_process
from biapy.data.data_2D_manipulation import crop_data_with_overlap, ensure_2d_shape
from biapy.data.data_3D_manipulation import (
    crop_3D_data_with_overlap,
    extract_3D_patch_with_overlap_yield,
    order_dimensions,
    ensure_3d_shape,
)


def load_and_prepare_train_data(
    train_path: str,
    train_mask_path: str,
    train_in_memory: str,
    train_ov: Tuple[float, ...],
    train_padding: Tuple[int, ...],
    val_path: str,
    val_mask_path: str,
    val_in_memory: bool,
    val_ov: Tuple[float, ...],
    val_padding: Tuple[int, ...],
    norm_module: Normalization,
    crop_shape: Tuple[int, ...],
    cross_val: bool = False,
    cross_val_nsplits: int = 5,
    cross_val_fold: int = 1,
    val_split: float = 0.1,
    seed: int = 0,
    shuffle_val: bool = True,
    train_preprocess_f: Optional[Callable] = None,
    train_preprocess_cfg: Optional[Dict] = None,
    train_filter_conds: List[List[str]] = [],
    train_filter_vals: List[List[float]] = [],
    train_filter_signs: List[List[str]] = [],
    val_preprocess_f: Optional[Callable] = None,
    val_preprocess_cfg: Optional[Dict] = None,
    val_filter_conds: List[List[str]] = [],
    val_filter_vals: List[List[float]] = [],
    val_filter_signs: List[List[str]] = [],
    filter_by_entire_image: bool = True,
    norm_before_filter: bool = False,
    random_crops_in_DA: bool = False,
    y_upscaling: Tuple[int, ...] = (1, 1),
    reflect_to_complete_shape: bool = False,
    convert_to_rgb: bool = False,
    is_y_mask: bool = False,
    is_3d: bool = False,
    train_zarr_data_information: Optional[Dict] = None,
    val_zarr_data_information: Optional[Dict] = None,
    multiple_raw_images: bool = False,
    save_filtered_images: bool = True,
    save_filtered_images_dir: Optional[str] = None,
    save_filtered_images_num: int = 3,
) -> Tuple[BiaPyDataset, BiaPyDataset, BiaPyDataset, BiaPyDataset]:
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

    norm_module : Normalization
        Information about the normalization.

    crop_shape : 3D/4D int tuple, optional
        Shape of the crops. E.g. ``(y, x, channels)`` for 2D and ``(z, y, x, channels)`` for 3D.

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
        * ``'diff'`` is defined as the difference between ground truth and raw images. Require ``y_dataset`` to be provided.
        * ``'diff_by_min_max_ratio'`` is defined as the difference between ground truth and raw images multiplied by the ratio
          between raw image max and min.
        * ``'target_mean'`` is defined as the mean intensity value of the raw image targets. Require ``y_dataset`` to be provided.
        * ``'target_min'`` is defined as the min intensity value of the raw image targets. Require ``y_dataset`` to be provided.
        * ``'target_max'`` is defined as the max intensity value of the raw image targets. Require ``y_dataset`` to be provided.
        * ``'diff_by_target_min_max_ratio'`` is defined as the difference between ground truth and raw images multiplied by the
          ratio between ground truth image max and min.

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
        * ``'diff'`` is defined as the difference between ground truth and raw images. Require ``y_dataset`` to be provided.
        * ``'diff_by_min_max_ratio'`` is defined as the difference between ground truth and raw images multiplied by the ratio
          between raw image max and min.
        * ``'target_mean'`` is defined as the mean intensity value of the raw image targets. Require ``y_dataset`` to be provided.
        * ``'target_min'`` is defined as the min intensity value of the raw image targets. Require ``y_dataset`` to be provided.
        * ``'target_max'`` is defined as the max intensity value of the raw image targets. Require ``y_dataset`` to be provided.
        * ``'diff_by_target_min_max_ratio'`` is defined as the difference between ground truth and raw images multiplied by the
          ratio between ground truth image max and min.

    val_filter_vals : list of int/float
        Represent the values of the properties listed in ``val_filter_conds`` that the images need to satisfy to not be dropped.

    val_filter_signs : list of list of str
        Signs to do the comparison for validation data filtering. Options: [``'gt'``, ``'ge'``, ``'lt'``, ``'le'``] that corresponds to
        "greather than", e.g. ">", "greather equal", e.g. ">=", "less than", e.g. "<", and "less equal" e.g. "<=" comparisons.

    filter_by_entire_image : bool, optional
        If filtering is done this will decide how the filtering will be done:
            * ``True``: apply filter image by image.
            * ``False``: apply filtering sample by sample. Each sample represents a patch within an image.

    norm_before_filter : bool, optional
        Whether to apply normalization before filtering. Be aware then that the values for filtering may change.

    random_crops_in_DA : bool, optional
        To advice the method that not preparation of the data must be done, as random subvolumes will be created on
        DA, and the whole volume will be used for that.

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
        * ``"raw_path"``, str: path where the raw images reside within the zarr (used when ``multiple_data_within_zarr`` is ``True``).
        * ``"gt_path"``, str: path where the mask images reside within the zarr (used when ``multiple_data_within_zarr`` is ``True``).
        * ``"use_gt_path"``, bool: whether the GT that should be used or not.
        * ``"multiple_data_within_zarr"``, bool: whether if your input Zarr contains the raw images and labels together or not.
        * ``"input_img_axes"``, tuple of int: order of the axes of the images.
        * ``"input_mask_axes"``, tuple of int: order of the axes of the masks.

    val_zarr_data_information : dict, optional
        Additional information when using Zarr/H5 files for validation. Same keys as ``train_zarr_data_information``
        are expected.

    multiple_raw_images : bool, optional
        When a folder of folders for each image is expected. In each of those subfolder different versions of the same image
        are placed. Visit the following tutorial for a real use case and a more detailed description:
        `Light My Cells <https://biapy.readthedocs.io/en/latest/tutorials/image-to-image/lightmycells.html>`_.
        This is used when ``PROBLEM.IMAGE_TO_IMAGE.MULTIPLE_RAW_ONE_TARGET_LOADER`` is selected.

    save_filtered_images : bool, optional
        Whether to save or not filtered images.

    save_filtered_images_dir : str, optional
        Directory to save filtered images.

    save_filtered_images_num : int, optional
        Number of filtered images to save. Only work when ``save_filtered_images`` is ``True``.

    Returns
    -------
    X_train : BiaPyDataset
        Loaded train X dataset.

    Y_train : BiaPyDataset
        Loaded train Y dataset.

    X_val : list of dict
        Loaded validation X dataset.

    Y_val : list of dict
        Loaded validation Y dataset.
    """
    train_shape_will_change = False
    if train_preprocess_f:
        if train_preprocess_cfg is None:
            raise ValueError("'train_preprocess_cfg' needs to be provided with 'train_preprocess_f'")
        if train_preprocess_cfg.RESIZE.ENABLE:
            train_shape_will_change = True
    val_shape_will_change = False
    if val_preprocess_f:
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

        print("Gathering raw images for training data . . .")
        if len(ids) == 0 or (len(ids) > 0 and any(ids[0].endswith(x) for x in [".h5", ".hdf5", ".hdf"])):  # Zarr
            if len(ids) == 0 and len(fids) == 0:  # Trying Zarr
                raise ValueError("No images found in dir {}".format(train_path))

            # Working with Zarr
            if not is_3d:
                raise ValueError("Zarr image handle is only available for 3D problems")
            train_using_zarr = True

            if norm_module.measure_by == "image":
                print(
                    "WARNING: normalization by image is not possible when using Zarr/H5 files. It will be done by patch instead."
                )

            assert train_zarr_data_information
            X_train = samples_from_zarr(
                list_of_data=fids if len(ids) == 0 else ids,
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
                norm_module=norm_module,
                is_mask=False,
                is_3d=is_3d,
                reflect_to_complete_shape=reflect_to_complete_shape,
                convert_to_rgb=convert_to_rgb,
                preprocess_f=train_preprocess_f if train_shape_will_change else None,
                preprocess_cfg=train_preprocess_cfg if train_shape_will_change else None,
            )

        # Extract a list of all training gt images
        if train_mask_path:
            print("Gathering labels for training data . . .")
            ids = sorted(next(os.walk(train_mask_path))[2])
            fids = sorted(next(os.walk(train_mask_path))[1])
            if len(ids) == 0 or (len(ids) > 0 and any(ids[0].endswith(x) for x in [".h5", ".hdf5", ".hdf"])):  # Zarr
                if len(ids) == 0 and len(fids) == 0:  # Trying Zarr
                    raise ValueError("No images found in dir {}".format(train_mask_path))
                assert train_zarr_data_information
                Y_train = samples_from_zarr(
                    list_of_data=fids if len(ids) == 0 else ids,
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
                    assert len(crop_shape) == 4
                    assert len(y_upscaling) == 3
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
                    norm_module=norm_module,
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

        print("Gathering raw and label images train information . . .")
        X_train, Y_train = samples_from_image_list_multiple_raw_one_gt(
            data_path=train_path,
            gt_path=train_mask_path,
            crop_shape=crop_shape,
            ov=train_ov,
            padding=train_padding,
            norm_module=norm_module,
            crop=crop,
            is_3d=is_3d,
            reflect_to_complete_shape=reflect_to_complete_shape,
            convert_to_rgb=convert_to_rgb,
            preprocess_f=train_preprocess_f if train_shape_will_change else None,
            preprocess_cfg=train_preprocess_cfg if train_shape_will_change else None,
        )

    # Check that the shape of all images match
    if train_mask_path and Y_train:
        print("Checking training raw and label images' shapes . . .")
        if not multiple_raw_images and len(X_train.sample_list) != len(Y_train.sample_list):
            mistmatch_message = shape_mismatch_message(X_train, Y_train)
            m = (
                "Mistmatch between number of raw samples ({}) and number of corresponding masks ({}). Please, check that the raw"
                "format and labels have same shape. {}".format(
                    len(X_train.sample_list), len(Y_train.sample_list), mistmatch_message
                )
            )
            raise ValueError(m)

        for i in range(len(X_train.sample_list)):
            xshape = X_train.sample_list[i].get_shape()
            gt_associated_id = X_train.sample_list[i].get_gt_associated_id()
            if gt_associated_id is not None:
                yshape = Y_train.sample_list[gt_associated_id].get_shape()
            else:
                yshape = Y_train.sample_list[i].get_shape()

            # The shape is not saved when 'DATA.EXTRACT_RANDOM_PATCH' is activated so set the crop_shape
            if not xshape:
                xshape = crop_shape[:-1]
            if not yshape:
                yshape = crop_shape[:-1]

            if is_3d:
                assert len(y_upscaling) == 3 and len(xshape) == 3
                upsampled_x_shape = (
                    xshape[0] * y_upscaling[0],
                    xshape[1] * y_upscaling[1],
                    xshape[2] * y_upscaling[2],
                )
            else:
                assert len(y_upscaling) == 2 and len(xshape) == 2
                upsampled_x_shape = (
                    xshape[0] * y_upscaling[0],
                    xshape[1] * y_upscaling[1],
                )
            if upsampled_x_shape != yshape[: len(upsampled_x_shape)]:
                filepath = X_train.dataset_info[X_train.sample_list[i].fid]
                raise ValueError(
                    f"There is a mismatch between input image and its corresponding ground truth ({upsampled_x_shape} vs "
                    f"{yshape}). Please check the images. Specifically, the sample that doesn't match is within "
                    f"the file: {filepath})"
                )

    if len(train_filter_conds) > 0:
        save_example_dir = None
        if save_filtered_images and save_filtered_images_dir:
            save_example_dir = os.path.join(save_filtered_images_dir, "train")
        filter_samples_by_properties(
            X_train,
            is_3d,
            train_filter_conds,
            train_filter_vals,
            train_filter_signs,
            y_dataset=Y_train,
            crop_shape=crop_shape,
            reflect_to_complete_shape=reflect_to_complete_shape,
            filter_by_entire_image=filter_by_entire_image if not random_crops_in_DA else True,
            norm_before_filter=norm_before_filter,
            norm_module=norm_module,
            zarr_data_information=train_zarr_data_information if train_using_zarr else None,
            save_filtered_images=save_filtered_images,
            save_filtered_images_dir=save_example_dir,
            save_filtered_images_num=save_filtered_images_num,
        )

    val_using_zarr = False
    if create_val_from_train:
        print("Creating validation data from train . . .")
        # Create IDs based on images or samples, depending if we are working with Zarr images or not. This is required to
        # create the validation data
        x_train_files = [x.path for x in X_train.dataset_info]
        if len(x_train_files) == 1:
            print(
                "As only one sample was found BiaPy will assume that it is big enough to hold multiple training samples "
                "so the validation will be created extracting samples from it too."
            )
        if train_using_zarr or len(x_train_files) == 1:
            x_train_ids = np.array(range(0, len(X_train.sample_list)))
            if train_mask_path and Y_train:
                y_train_ids = np.array(range(0, len(Y_train.sample_list)))
                if not multiple_raw_images and len(x_train_ids) != len(y_train_ids):
                    raise ValueError(
                        f"Raw image number ({len(x_train_ids)}) and ground truth file mismatch ({len(y_train_ids)}). Please check the data!"
                    )
            clean_by = "sample"
        else:
            x_train_files.sort()
            x_train_ids = np.array(range(0, len(x_train_files)))
            if train_mask_path and Y_train:
                y_train_ids = np.array(range(0, len(Y_train.dataset_info)))
                if not multiple_raw_images and len(x_train_ids) != len(y_train_ids):
                    raise ValueError(
                        f"Raw image number ({len(x_train_ids)}) and ground truth file mismatch ({len(y_train_ids)}). Please check the data!"
                    )
            clean_by = "image"

        val_path = train_path
        val_mask_path = train_mask_path
        val_zarr_data_information = train_zarr_data_information
        val_using_zarr = train_using_zarr
        if not cross_val:
            if train_mask_path:
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

            y_len = len(y_train_ids) if train_mask_path else len(x_train_ids)
            for t_index, te_index in skf.split(np.zeros(len(x_train_ids)), np.zeros(y_len)):
                if cross_val_fold == fold:
                    x_train_ids, x_val_ids = x_train_ids[t_index], x_train_ids[te_index]
                    if train_mask_path:
                        y_train_ids, y_val_ids = y_train_ids[t_index], y_train_ids[te_index]

                    train_index, test_index = t_index.copy(), te_index.copy()
                    break
                fold += 1

            if len(test_index) > 5:
                print("Fold number {}. Printing the first 5 ids: {}".format(fold, test_index[:5]))
            else:
                print("Fold number {}. Indexes used in cross validation: {}".format(fold, test_index))
            x_val_ids = test_index.copy()

        # It's important to sort them in order to speed up load_images_to_dataset() process
        x_val_ids.sort()
        x_train_ids.sort()

        # Create validation data from train.
        X_val = X_train.copy()
        X_val.clean_dataset(x_val_ids, clean_by=clean_by)
        if Y_train:
            Y_val = Y_train.copy()
            Y_val.clean_dataset(x_val_ids, clean_by=clean_by)

        # Remove val samples from train.
        X_train.clean_dataset(x_train_ids, clean_by=clean_by)
        if Y_train:
            Y_train.clean_dataset(x_train_ids, clean_by=clean_by)

        if clean_by == "sample":
            print(
                "Raw samples chosen for training (first 10 only): {}".format(str(x_train_ids[:10]).replace("]", "...]"))
            )
            print(
                "Raw samples chosen for validation (first 10 only): {}".format(
                    str(x_val_ids[:10]).replace("]", " ...]")
                )
            )
        else:
            print("Raw images chosen for training: {}".format([x.path for x in X_train.dataset_info]))
            print("Raw images chosen for validation: {}".format([x.path for x in X_val.dataset_info]))
    else:
        if not multiple_raw_images:
            print("Gathering raw images for validation data . . .")
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

                if norm_module.measure_by == "image":
                    print(
                        "WARNING: normalization by image is not possible when using Zarr/H5 files. It will be done by patch instead."
                    )

                assert val_zarr_data_information
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
                    norm_module=norm_module,
                    is_mask=False,
                    is_3d=is_3d,
                    reflect_to_complete_shape=reflect_to_complete_shape,
                    convert_to_rgb=convert_to_rgb,
                    preprocess_f=val_preprocess_f if val_shape_will_change else None,
                    preprocess_cfg=val_preprocess_cfg if val_shape_will_change else None,
                )

            # Extract a list of all validation gt images
            if val_mask_path:
                print("Gathering labels for validation data . . .")
                val_ids = sorted(next(os.walk(val_mask_path))[2])
                val_fids = sorted(next(os.walk(val_mask_path))[1])
                if len(val_ids) == 0:
                    if len(val_fids) == 0:  # Trying Zarr
                        raise ValueError("No images found in dir {}".format(val_mask_path))

                    # Working with Zarr
                    if not is_3d:
                        raise ValueError("Zarr image handle is only available for 3D problems")

                    assert val_zarr_data_information
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
                        assert len(y_upscaling) == 3 and len(crop_shape) == 4
                        real_shape = (
                            crop_shape[0] * y_upscaling[0],
                            crop_shape[1] * y_upscaling[1],
                            crop_shape[2] * y_upscaling[2],
                            crop_shape[3],
                        )
                    else:
                        assert len(crop_shape) == 3 and len(y_upscaling) == 2
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
                        norm_module=norm_module,
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

            print("Gathering raw and label images for validation data . . .")
            X_val, Y_val = samples_from_image_list_multiple_raw_one_gt(
                data_path=val_path,
                gt_path=val_mask_path,
                crop_shape=crop_shape,
                ov=val_ov,
                padding=val_padding,
                norm_module=norm_module,
                crop=crop,
                is_3d=is_3d,
                reflect_to_complete_shape=reflect_to_complete_shape,
                convert_to_rgb=convert_to_rgb,
                preprocess_f=val_preprocess_f if val_shape_will_change else None,
                preprocess_cfg=val_preprocess_cfg if val_shape_will_change else None,
            )

        # Check that the shape of all images match
        if val_mask_path and Y_val:
            print("Checking validation raw and label images' shapes . . .")
            for i in range(len(X_val.sample_list)):
                xshape = X_val.sample_list[i].get_shape()
                gt_associated_id = X_val.sample_list[i].get_gt_associated_id()
                if gt_associated_id is not None:
                    yshape = Y_val.sample_list[gt_associated_id].get_shape()
                else:
                    yshape = Y_val.sample_list[i].get_shape()

                # The shape is not saved when 'DATA.EXTRACT_RANDOM_PATCH' is activated so set the crop_shape
                if not xshape:
                    xshape = crop_shape[:-1]
                if not yshape:
                    yshape = crop_shape[:-1]

                if is_3d:
                    assert len(y_upscaling) == 3 and len(xshape) == 3
                    upsampled_x_shape = (
                        xshape[0] * y_upscaling[0],
                        xshape[1] * y_upscaling[1],
                        xshape[2] * y_upscaling[2],
                    )
                else:
                    assert len(y_upscaling) == 2 and len(xshape) == 2
                    upsampled_x_shape = (
                        xshape[0] * y_upscaling[0],
                        xshape[1] * y_upscaling[1],
                    )
                if upsampled_x_shape != yshape[: len(upsampled_x_shape)]:
                    filepath = X_val.dataset_info[X_val.sample_list[i].fid]
                    raise ValueError(
                        f"There is a mismatch between input image and its corresponding ground truth ({upsampled_x_shape} vs "
                        f"{yshape}). Please check the images. Specifically, the sample that doesn't match is within "
                        f"the file {filepath})"
                    )

        if len(val_filter_conds) > 0:
            save_example_dir = None
            if save_filtered_images and save_filtered_images_dir:
                save_example_dir = os.path.join(save_filtered_images_dir, "val")
            filter_samples_by_properties(
                X_val,
                is_3d,
                val_filter_conds,
                val_filter_vals,
                val_filter_signs,
                y_dataset=Y_val,
                crop_shape=crop_shape,
                reflect_to_complete_shape=reflect_to_complete_shape,
                filter_by_entire_image=filter_by_entire_image if not random_crops_in_DA else True,
                norm_before_filter=norm_before_filter,
                norm_module=norm_module,
                zarr_data_information=val_zarr_data_information if val_using_zarr else None,
                save_filtered_images=save_filtered_images,
                save_filtered_images_dir=save_example_dir,
                save_filtered_images_num=save_filtered_images_num,
            )

        x_val_ids = np.array(range(0, len(X_val.sample_list)))
        if val_mask_path and Y_val:
            y_val_ids = np.array(range(0, len(Y_val.sample_list)))
            if not multiple_raw_images and len(x_val_ids) != len(y_val_ids):
                raise ValueError(
                    f"Raw image number ({len(x_val_ids)}) and ground truth file mismatch ({len(y_val_ids)}). Please check the data!"
                )

    if train_in_memory:
        print("* Loading train images . . .")
        load_images_to_dataset(
            dataset=X_train,
            crop_shape=crop_shape,
            reflect_to_complete_shape=reflect_to_complete_shape,
            convert_to_rgb=convert_to_rgb,
            preprocess_cfg=train_preprocess_cfg,
            preprocess_f=train_preprocess_f,
            is_3d=is_3d,
            zarr_data_information=train_zarr_data_information if train_using_zarr else None,
        )
        if train_mask_path and Y_train:
            print("* Loading train GT . . .")
            if is_3d:
                assert len(y_upscaling) == 3 and len(crop_shape) == 4
                real_shape = (
                    crop_shape[0] * y_upscaling[0],
                    crop_shape[1] * y_upscaling[1],
                    crop_shape[2] * y_upscaling[2],
                    crop_shape[3],
                )
            else:
                assert len(y_upscaling) == 2 and len(crop_shape) == 3
                real_shape = (
                    crop_shape[0] * y_upscaling[0],
                    crop_shape[1] * y_upscaling[1],
                    crop_shape[2],
                )
            load_images_to_dataset(
                dataset=Y_train,
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
        load_images_to_dataset(
            dataset=X_val,
            crop_shape=crop_shape,
            reflect_to_complete_shape=reflect_to_complete_shape,
            convert_to_rgb=convert_to_rgb,
            preprocess_cfg=val_preprocess_cfg,
            preprocess_f=val_preprocess_f,
            is_3d=is_3d,
            zarr_data_information=val_zarr_data_information if val_using_zarr else None,
        )

        if val_mask_path and Y_val:
            print("* Loading validation GT . . .")
            if is_3d:
                assert len(y_upscaling) == 3 and len(crop_shape) == 4
                real_shape = (
                    crop_shape[0] * y_upscaling[0],
                    crop_shape[1] * y_upscaling[1],
                    crop_shape[2] * y_upscaling[2],
                    crop_shape[3],
                )
            else:
                assert len(y_upscaling) == 2 and len(crop_shape) == 3
                real_shape = (
                    crop_shape[0] * y_upscaling[0],
                    crop_shape[1] * y_upscaling[1],
                    crop_shape[2],
                )
            load_images_to_dataset(
                dataset=Y_val,
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
    if X_train.sample_list[0].coords == None:
        print(
            "The samples have not been cropped so they may have different shapes. Because of that only first sample's shape will be printed!"
        )
    sample_shape = X_train.sample_list[0].get_shape()
    if not sample_shape:
        sample_shape = crop_shape
    X_data_shape = (len(X_train.sample_list),) + sample_shape
    print("*** Loaded train data shape is: {}".format(X_data_shape))
    if Y_train:
        sample_shape = Y_train.sample_list[0].get_shape()
        if not sample_shape:
            sample_shape = crop_shape
        Y_data_shape = (len(Y_train.sample_list),) + sample_shape
        print("*** Loaded train GT shape is: {}".format(Y_data_shape))
    else:
        Y_train = X_train.copy()

    sample_shape = X_val.sample_list[0].get_shape()
    if not sample_shape:
        sample_shape = crop_shape
    X_data_shape = (len(X_val.sample_list),) + sample_shape
    print("*** Loaded validation data shape is: {}".format(X_data_shape))
    if Y_val:
        sample_shape = Y_val.sample_list[0].get_shape()
        if not sample_shape:
            sample_shape = crop_shape
        Y_data_shape = (len(Y_val.sample_list),) + sample_shape
        print("*** Loaded validation GT shape is: {}".format(Y_data_shape))
    else:
        Y_val = X_val.copy()
    print("### END LOAD ###")

    return X_train, Y_train, X_val, Y_val


def load_and_prepare_test_data(
    test_path: str,
    test_mask_path: Optional[str],
    multiple_raw_images: Optional[bool] = False,
    test_zarr_data_information: Optional[Dict] = None,
) -> Tuple[BiaPyDataset, Optional[BiaPyDataset], List]:
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

    test_zarr_data_information : dict, optional
        Additional information when using Zarr/H5 files for test. The following keys are expected:
            * ``"raw_path"``, str: path where the raw images reside within the zarr.
            * ``"gt_path"``, str: path where the mask images reside within the zarr.
            * ``"use_gt_path"``, str: whether the GT that should be used or not.

    Returns
    -------
    X_train : list of dict
        Loaded train X data. Each item in the list represents a sample of the dataset. Each sample is represented as follows:
            * ``"filename"``, str: name of the image to extract the data sample from.
            * ``"dir"``, str: directory where the image resides.

    Y_train : list of dict, optional
        Loaded train Y data. Each item in the list represents a sample of the dataset. Each sample is represented as follows:
            * ``"train_path"``, str: name of the image to extract the data sample from.
            * ``"dir"``, str: directory where the image resides.

    test_filenames : list of str
        List of test filenames.
    """

    print("### LOAD ###")

    sample_list = []
    dataset_info = []
    Y_test = None
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
            dataset_info.append(DatasetFile(path=os.path.join(test_path, test_filenames[i])))
            sample_data = DataSample(fid=i, coords=None)
            if test_zarr_data_information:
                sample_data.path_in_zarr = test_zarr_data_information["raw_path"]
            sample_list.append(sample_data)

        # Extract a list of all gt images
        if test_mask_path:
            y_dataset_info = []
            y_sample_list = []
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
                y_dataset_info.append(DatasetFile(path=os.path.join(test_mask_path, selected_ids[i])))
                sample_data = DataSample(fid=i, coords=None)
                if test_zarr_data_information:
                    if test_zarr_data_information["use_gt_path"]:
                        sample_data.path_in_zarr = test_zarr_data_information["gt_path"]
                    else:
                        sample_data.path_in_zarr = test_zarr_data_information["raw_path"]
                y_sample_list.append(sample_data)
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
                dataset_info.append(DatasetFile(path=os.path.join(sample_path, ids[i])))
                sample_list.append(DataSample(fid=i, coords=None))

        # Extract a list of all training gt images
        if test_mask_path:
            y_dataset_info = []
            y_sample_list = []
            fids = sorted(next(os.walk(test_mask_path))[1])
            if len(fids) == 0:
                raise ValueError("No folders found in dir {}".format(test_mask_path))
            for folder in fids:
                sample_path = os.path.join(test_mask_path, folder)
                ids = sorted(next(os.walk(sample_path))[2])
                if len(ids) == 0:
                    raise ValueError("No images found in dir {}".format(sample_path))
                for i in range(len(ids)):
                    y_dataset_info.append(DatasetFile(path=os.path.join(sample_path, ids[i])))
                    y_sample_list.append(DataSample(fid=i, coords=None))

    X_test = BiaPyDataset(dataset_info=dataset_info, sample_list=sample_list)
    if test_mask_path:
        Y_test = BiaPyDataset(dataset_info=y_dataset_info, sample_list=y_sample_list)
    return X_test, Y_test, test_filenames


def load_and_prepare_cls_test_data(
    test_path: str,
    use_val_as_test: bool,
    expected_classes: int,
    crop_shape: Tuple[int, ...],
    is_3d: bool = True,
    reflect_to_complete_shape: bool = True,
    convert_to_rgb: bool = False,
    use_val_as_test_info: Optional[Dict] = None,
):
    """
    Load test data.

    Parameters
    ----------
    train_path : str
        Path to the training data.

    use_val_as_test : bool
        Whether to use validation data as test.

    expected_classes : int
        Expected number of classes to be loaded.

    crop_shape : 3D/4D int tuple
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
            * ``"cross_val_samples_ids"``, list of int: ids of the validation samples (out of the cross validation).
            * ``"train_path"``, str: training path, as the data must be extracted from there.
            * ``"selected_fold``", int: fold selected in cross validation.
            * ``"n_splits"``, int: folds to create in cross validation.
            * ``"shuffle"``, bool: whether to shuffle the data or not.
            * ``"seed"``, int: mathematical seed.

    Returns
    -------
    X_test : list of dict
        Loaded test data. Each item in the list represents a sample of the dataset. Each sample is represented as follows:
            * ``"filename"``, str: name of the image to extract the data sample from.
            * ``"dir"``, str: directory where the image resides.
            * ``"class_name"``, str: name of the class.
            * ``"class"``, int: represents the class (``-1`` if no ground truth provided).

    test_filenames : list of str
        List of test filenames.
    """

    print("### LOAD ###")

    X_test = []

    if not use_val_as_test:
        path_to_process = test_path
    else:
        assert use_val_as_test_info, "'use_val_as_test_info' can not be None when 'use_val_as_test' is 'True'"
        path_to_process = use_val_as_test_info["train_path"]

    X_test = samples_from_class_list(
        data_path=path_to_process,
        expected_classes=expected_classes,
        crop_shape=crop_shape,
        is_3d=is_3d,
        reflect_to_complete_shape=reflect_to_complete_shape,
        convert_to_rgb=convert_to_rgb,
    )
    test_filenames = [X_test.dataset_info[x.fid] for x in X_test.sample_list]

    if use_val_as_test:
        # The test is the validation, and as it is only available when validation is obtained from train and when
        # cross validation is enabled, the test set files reside in the train folder
        assert use_val_as_test_info
        if use_val_as_test_info["cross_val_samples_ids"] is None:
            x_test_ids = np.array(range(0, len(X_test.sample_list)))
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
            X_test.clean_dataset(use_val_as_test_info["cross_val_samples_ids"])
            test_filenames = [test_filenames[i] for i in use_val_as_test_info["cross_val_samples_ids"]]

    return X_test, test_filenames


def load_data_from_dir(
    data_path: str, is_3d: bool = False, **kwargs
) -> BiaPyDataset | Tuple[BiaPyDataset, BiaPyDataset]:
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
    if isinstance(data_samples, tuple):
        x_samples, y_samples = data_samples
        load_images_to_dataset(
            dataset=x_samples,
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
        load_images_to_dataset(
            dataset=y_samples,
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
        return x_samples, y_samples
    else:
        load_images_to_dataset(
            dataset=data_samples,
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
    data_path: str,
    expected_classes: int,
    crop_shape: Optional[Tuple[int, ...]],
    is_3d: bool = True,
    reflect_to_complete_shape: bool = True,
    convert_to_rgb: bool = False,
    preprocess_f: Optional[Callable] = None,
    preprocess_cfg: Optional[Dict] = None,
) -> BiaPyDataset:
    """
    Create dataset samples from the given list following a classification workflow directory tree.

    Parameters
    ----------
    data_path : str
        Path to read the images from.

    expected_classes : int
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
    data_samples : BiaPyDataset
        Dataset created out of ``data_path``.
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
    load_images_to_dataset(
        dataset=data_samples,
        crop_shape=crop_shape,
        reflect_to_complete_shape=reflect_to_complete_shape,
        convert_to_rgb=convert_to_rgb,
        preprocess_cfg=preprocess_cfg,
        preprocess_f=preprocess_f,
        is_3d=is_3d,
    )

    return data_samples


def load_and_prepare_train_data_cls(
    train_path: str,
    train_in_memory: bool,
    val_path: str,
    val_in_memory: bool,
    expected_classes: int,
    crop_shape: Tuple[int, ...],
    cross_val: bool = False,
    cross_val_nsplits: int = 5,
    cross_val_fold: int = 1,
    val_split: float = 0.1,
    seed: int = 0,
    shuffle_val: bool = True,
    train_preprocess_f: Optional[Callable] = None,
    train_preprocess_cfg: Optional[Dict] = None,
    train_filter_conds: List[List[str]] = [],
    train_filter_vals: List[List[float | int]] = [],
    train_filter_signs: List[List[str]] = [],
    val_preprocess_f: Optional[Callable] = None,
    val_preprocess_cfg: Optional[Dict] = None,
    val_filter_conds: List[List[str]] = [],
    val_filter_vals: List[List[int | float]] = [],
    val_filter_signs: List[List[str]] = [],
    norm_before_filter: bool = False,
    norm_module: Optional[Normalization] = None,
    reflect_to_complete_shape: bool = False,
    convert_to_rgb: bool = False,
    is_3d: bool = False,
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

    expected_classes : int
        Expected number of classes to be loaded.

    crop_shape : 3D/4D int tuple
        Shape of the crops. E.g. ``(y, x, channels)`` for 2D and ``(z, y, x, channels)`` for 3D.

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
        * ``'diff'`` is defined as the difference between ground truth and raw images. Require ``y_dataset`` to be provided.
        * ``'diff_by_min_max_ratio'`` is defined as the difference between ground truth and raw images multiplied by the ratio
          between raw image max and min.
        * ``'target_mean'`` is defined as the mean intensity value of the raw image targets. Require ``y_dataset`` to be provided.
        * ``'target_min'`` is defined as the min intensity value of the raw image targets. Require ``y_dataset`` to be provided.
        * ``'target_max'`` is defined as the max intensity value of the raw image targets. Require ``y_dataset`` to be provided.
        * ``'diff_by_target_min_max_ratio'`` is defined as the difference between ground truth and raw images multiplied by the
          ratio between ground truth image max and min.

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
        * ``'diff'`` is defined as the difference between ground truth and raw images. Require ``y_dataset`` to be provided.
        * ``'diff_by_min_max_ratio'`` is defined as the difference between ground truth and raw images multiplied by the ratio
          between raw image max and min.
        * ``'target_mean'`` is defined as the mean intensity value of the raw image targets. Require ``y_dataset`` to be provided.
        * ``'target_min'`` is defined as the min intensity value of the raw image targets. Require ``y_dataset`` to be provided.
        * ``'target_max'`` is defined as the max intensity value of the raw image targets. Require ``y_dataset`` to be provided.
        * ``'diff_by_target_min_max_ratio'`` is defined as the difference between ground truth and raw images multiplied by the
          ratio between ground truth image max and min.

    val_filter_vals : list of int/float
        Represent the values of the properties listed in ``val_filter_conds`` that the images need to satisfy to not be dropped.

    val_filter_signs : list of list of str
        Signs to do the comparison for validation data filtering. Options: [``'gt'``, ``'ge'``, ``'lt'``, ``'le'``] that corresponds to
        "greather than", e.g. ">", "greather equal", e.g. ">=", "less than", e.g. "<", and "less equal" e.g. "<=" comparisons.

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
            * ``"filename"``, str: name of the image to extract the data sample from.
            * ``"dir"``, str: directory where the image resides.
            * ``"class_name"``, str: name of the class.
            * ``"class"``, int: represents the class (``-1`` if no ground truth provided).
            * ``"img"``, ndarray (optional): image sample itself. It is of ``(y, x, channels)`` in ``2D`` and
              ``(z, y, x, channels)`` in ``3D``. Provided when ``val_in_memory`` is ``True``.

    X_val : list of dict
        Loaded validation data. Each item in the list represents a sample of the dataset. Each sample is represented as follows:
            * ``"filename"``, str: name of the image to extract the data sample from.
            * ``"dir"``, str: directory where the image resides.
            * ``"class_name"``, str: name of the class.
            * ``"class"``, int: represents the class (``-1`` if no ground truth provided).
            * ``"img"``, ndarray (optional): image sample itself. It is of ``(y, x, channels)`` in ``2D`` and
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
        filter_samples_by_properties(
            X_train,
            is_3d,
            train_filter_conds,
            train_filter_vals,
            train_filter_signs,
            crop_shape=crop_shape,
            reflect_to_complete_shape=reflect_to_complete_shape,
            norm_before_filter=norm_before_filter,
            norm_module=norm_module,
        )

    x_train_ids = np.array(range(0, len(X_train.sample_list)))
    y_train_ids = np.array([x.class_num for x in X_train.dataset_info])
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

        # Create validation data from train. It's important to sort them in order to speed up load_images_to_dataset() process
        x_val_ids.sort()
        x_train_ids.sort()
        X_val = X_train.copy()
        X_train.clean_dataset(x_train_ids)
        X_val.clean_dataset(x_val_ids)
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
            filter_samples_by_properties(
                X_val,
                is_3d,
                val_filter_conds,
                val_filter_vals,
                val_filter_signs,
                crop_shape=crop_shape,
                reflect_to_complete_shape=reflect_to_complete_shape,
                norm_before_filter=norm_before_filter,
                norm_module=norm_module,
            )

        x_val_ids = np.array(range(0, len(X_val.sample_list)))

    if train_in_memory:
        print("* Loading train images . . .")
        load_images_to_dataset(
            dataset=X_train,
            crop_shape=crop_shape,
            reflect_to_complete_shape=reflect_to_complete_shape,
            convert_to_rgb=convert_to_rgb,
            preprocess_cfg=train_preprocess_cfg,
            preprocess_f=train_preprocess_f,
            is_3d=is_3d,
        )

    if val_in_memory:
        print("* Loading validation images . . .")
        load_images_to_dataset(
            dataset=X_val,
            crop_shape=crop_shape,
            reflect_to_complete_shape=reflect_to_complete_shape,
            convert_to_rgb=convert_to_rgb,
            preprocess_cfg=val_preprocess_cfg,
            preprocess_f=val_preprocess_f,
            is_3d=is_3d,
        )

    print("### LOAD RESULTS ###")
    X_data_shape = (len(X_train.sample_list),) + crop_shape
    print("*** Loaded train data shape is: {}".format(X_data_shape))
    X_data_shape = (len(X_val.sample_list),) + crop_shape
    print("*** Loaded validation data shape is: {}".format(X_data_shape))
    print("### END LOAD ###")

    return X_train, X_val, x_val_ids


def samples_from_image_list(
    list_of_data: List[str],
    data_path: str,
    crop_shape: Tuple[int, ...],
    ov: Tuple[float, ...],
    padding: Tuple[int, ...],
    norm_module: Normalization,
    crop: bool = True,
    is_mask: bool = False,
    is_3d: bool = True,
    reflect_to_complete_shape: bool = True,
    convert_to_rgb: bool = False,
    preprocess_f: Optional[Callable] = None,
    preprocess_cfg: Optional[Dict] = None,
) -> BiaPyDataset:
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

    norm_module : Normalization
        Information about the normalization.

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
    dataset : BiaPyDataset
        Dataset.
    """
    if preprocess_f and preprocess_cfg is None:
        raise ValueError("'preprocess_cfg' needs to be provided with 'preprocess_f'")

    crop_funct = crop_3D_data_with_overlap if is_3d else crop_data_with_overlap
    sample_list = []
    dataset_info = []
    channel_expected = -1
    data_range_expected = -1
    for i in range(len(list_of_data)):
        # Read image
        img_path = os.path.join(data_path, list_of_data[i])
        img, _ = load_img_data(img_path, is_3d=is_3d)

        # Apply preprocessing
        if preprocess_f:
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

        if crop and (
            img.shape <= crop_shape[:-1] + (img.shape[-1],) or img.shape >= crop_shape[:-1] + (img.shape[-1],)
        ):
            crop_coords = crop_funct(
                np.expand_dims(img, axis=0) if not is_3d else img,
                crop_shape[:-1] + (img.shape[-1],),
                overlap=ov,
                padding=padding,
                verbose=False,
                load_data=False,
            )
            tot_samples_to_insert = len(crop_coords)
        else:
            tot_samples_to_insert = 1

        dataset_file = DatasetFile(
            path=os.path.join(data_path, list_of_data[i]),
            shape=original_data_shape,
        )
        norm_module.set_stats_from_image(img)
        norm_module.set_DatasetFile_from_stats(dataset_file)
        dataset_info.append(dataset_file)
        for j in range(tot_samples_to_insert):
            data_sample = DataSample(
                fid=i,
                coords=crop_coords[j] if crop_coords else None,  # type: ignore
            )
            sample_list.append(data_sample)

    return BiaPyDataset(dataset_info=dataset_info, sample_list=sample_list)


def samples_from_zarr(
    list_of_data: List[str],
    data_path: str,
    zarr_data_info: Dict,
    crop_shape: Tuple[int, ...],
    ov: Tuple[float, ...],
    padding: Tuple[int, ...],
    is_mask: bool = False,
    is_3d: bool = True,
) -> BiaPyDataset:
    """
    Create dataset samples from the given list. This function does not load the data.

    Parameters
    ----------
    list_of_data : list of str
        Filenames of the images to read.

    data_path : str
        Directory of the images to read.

    zarr_data_info : dict
        Additional information when using Zarr/H5 files for training. The following keys are expected:
            * ``"raw_path"``: path where the raw images reside within the zarr (used when ``multiple_data_within_zarr`` is ``True``).
            * ``"gt_path"``: path where the mask images reside within the zarr (used when ``multiple_data_within_zarr`` is ``True``).
            * ``"multiple_data_within_zarr"``: Whether if your input Zarr contains the raw images and labels together or not.
            * ``"input_img_axes"``: order of the axes of the images.
            * ``"input_mask_axes"``: order of the axes of the masks.

    crop_shape : 3D/4D int tuple
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
    dataset : BiaPyDataset
        Dataset.
    """
    # Extract a list of all training samples within the Zarr
    sample_list = []
    dataset_info = []
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
        if not is_mask:
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
        total_patches, _, _ = obj  # type: ignore

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
        dataset_info.append(
            DatasetFile(
                path=os.path.join(data_path, list_of_data[i]),
                shape=data.shape,
                parallel_data=True,
                input_axes=zarr_data_info["input_img_axes"] if not is_mask else zarr_data_info["input_mask_axes"],
            )
        )
        for obj in tqdm(for_img_cond, total=total_patches, disable=not is_main_process()):  # type: ignore
            coords, _, _, _ = obj  # type: ignore

            # Create crop_shape from coords as the sample is not loaded to speed up the process
            if is_3d:
                crop_shape = (
                    coords.z_end - coords.z_start,
                    coords.y_end - coords.y_start,
                    coords.x_end - coords.x_start,
                    channel,
                )
            else:
                crop_shape = (
                    coords.y_end - coords.y_start,
                    coords.x_end - coords.x_start,
                    channel,
                )

            assert isinstance(coords, PatchCoords)
            sample_dict = DataSample(
                fid=i,
                coords=coords,
            )
            if data_within_zarr_path:
                sample_dict.path_in_zarr = data_within_zarr_path

            sample_list.append(sample_dict)

        if isinstance(file, h5py.File):
            file.close()

    return BiaPyDataset(dataset_info=dataset_info, sample_list=sample_list)


def samples_from_image_list_multiple_raw_one_gt(
    data_path: str,
    gt_path: str,
    crop_shape: Tuple[int, ...],
    ov: Tuple[float, ...],
    padding: Tuple[int, ...],
    norm_module: Normalization,
    crop: bool = True,
    is_3d: bool = True,
    reflect_to_complete_shape: bool = True,
    convert_to_rgb: bool = False,
    preprocess_f: Optional[Callable] = None,
    preprocess_cfg: Optional[Dict] = None,
) -> Tuple[BiaPyDataset, BiaPyDataset]:
    """
    Create dataset samples from the given lists. This function does not load the data.

    Parameters
    ----------
    data_path : str
        Directory of the images to read.

    gt_path : str
        Directory to read ground truth images from.

    crop_shape : 3D/4D int tuple
        Shape of the crops. E.g. ``(y, x, channels)`` for 2D and ``(z, y, x, channels)`` for 3D.

    ov : 2D/3D float tuple
        Amount of minimum overlap on x and y dimensions. The values must be on range ``[0, 1)``,
        that is, ``0%`` or ``99%`` of overlap. Shape is ``(y, x)`` for 2D or ``(z, y, x)`` for 3D.

    padding : 2D/3D int tuple
        Size of padding to be added on each axis. Shape is ``(y, x)`` for 2D or ``(z, y, x)`` for 3D.

    norm_module : dict
        Information about the normalization. Expected keys are:
            * ``"type"``, str: normalization type to use. Possible options:
                - ``"div"`` to divide values from ``0/255`` (or ``0/65535`` if ``uint16``) in ``[0,1]`` range.
                - ``"scale_range"`` same as ``"div"`` but scaling the range to ``[0-max]`` and then dividing by the maximum
                  value of the data and not by ``255`` or ``65535``.
                - ``"zero_mean_unit_variance"`` to substract the mean and divide by std. In this option ``mean`` and ``std``
                  can also be provided or they will be calculated from the input.
            * "measure_by", str: how to measure the values needed for normalization. Possible options:
                - ``"image"`` to calculate the values per image
                - ``"patch"`` to calculate the values per patch

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
    dataset : BiaPyDataset
        X dataset.

    gt_dataset : BiaPyDataset
        Y dataset.
    """
    if preprocess_f and preprocess_cfg is None:
        raise ValueError("'preprocess_cfg' needs to be provided with 'preprocess_f'")

    crop_funct = crop_3D_data_with_overlap if is_3d else crop_data_with_overlap
    data_gt_path = sorted(next(os.walk(gt_path))[1])
    sample_list = []
    dataset_info = []
    gt_sample_list = []
    gt_dataset_info = []
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
        if preprocess_f:
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
        if crop_shape:
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

        if crop and (
            gt_sample.shape <= crop_shape[:-1] + (gt_sample.shape[-1],)
            or gt_sample.shape >= crop_shape[:-1] + (gt_sample.shape[-1],)
        ):
            crop_coords = crop_funct(
                np.expand_dims(gt_sample, axis=0) if not is_3d else gt_sample,
                crop_shape[:-1] + (gt_sample.shape[-1],),
                overlap=ov,
                padding=padding,
                verbose=False,
                load_data=False,
            )
            gt_tot_samples_to_insert = len(crop_coords)
        else:
            gt_tot_samples_to_insert = 1

        data_file = DatasetFile(path=os.path.join(gt_path, id_, gt_id), shape=original_data_shape)
        norm_module.set_stats_from_image(gt_sample)
        norm_module.set_DatasetFile_from_stats(data_file)
        gt_dataset_info.append(data_file)
        for i in range(gt_tot_samples_to_insert):
            coords = None
            if crop_coords is not None:
                coords = crop_coords[i]
                assert isinstance(coords, PatchCoords)
            data_sample = DataSample(
                fid=len(gt_dataset_info) - 1,
                coords=coords,
            )
            gt_sample_list.append(data_sample)

        # For each gt samples there are multiple raw images
        for raw_sample_id in raw_samples:
            # Read image
            raw_sample_path = os.path.join(associated_raw_image_dir, raw_sample_id)
            raw_sample, _ = load_img_data(raw_sample_path, is_3d=is_3d)

            # Apply preprocessing
            if preprocess_f:
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
            if crop_shape:
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
                tot_samples_to_insert = len(crop_coords)
            else:
                tot_samples_to_insert = 1

            dataset_file = DatasetFile(
                path=os.path.join(associated_raw_image_dir, raw_sample_id),
                shape=original_data_shape,
            )
            norm_module.set_stats_from_image(raw_sample)
            norm_module.set_DatasetFile_from_stats(dataset_file)
            dataset_info.append(dataset_file)
            for i in range(tot_samples_to_insert):
                data_sample = DataSample(
                    fid=len(dataset_info) - 1,
                    coords=crop_coords[i] if crop_coords else None,  # type: ignore
                    gt_associated_id=cont + i,  # this extra variable is added
                )
                sample_list.append(data_sample)

        cont += gt_tot_samples_to_insert

    return (
        BiaPyDataset(dataset_info=dataset_info, sample_list=sample_list),
        BiaPyDataset(dataset_info=gt_dataset_info, sample_list=gt_sample_list),
    )


def samples_from_class_list(
    data_path: str,
    crop_shape: Optional[Tuple[int, ...]] = None,
    expected_classes: int = -1,
    is_3d: bool = True,
    reflect_to_complete_shape: bool = True,
    convert_to_rgb: bool = False,
) -> BiaPyDataset:
    """
    Create dataset samples from the given path taking into account that each subfolder represents a class.
    This function does not load the data.

    Parameters
    ----------
    data_path : str
        Directory of the images to read.

    crop_shape : 3D/4D int tuple, optional
        Shape of the crops. E.g. ``(y, x, channels)`` for 2D and ``(z, y, x, channels)`` for 3D.

    expected_classes : int, optional
        Expected number of classes to be loaded. Set to -1 if you don't expect any.

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
    sample_list : list of DataSample
        Samples generated out of ``data_path``.
    """
    if expected_classes != -1:
        list_of_classes = sorted(next(os.walk(data_path))[1])
        if len(list_of_classes) < 1:
            raise ValueError("There is no folder/class in {}".format(data_path))

        if expected_classes:
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
    xdataset_info = []
    data_file_count = 0
    for c_num, class_name in enumerate(list_of_classes):
        class_folder = os.path.join(data_path, class_name)

        ids = sorted(next(os.walk(class_folder))[2])
        if len(ids) == 0:
            raise ValueError("There are no images in class {}".format(class_folder))

        channel_expected = -1
        data_range_expected = -1
        for j, id_ in enumerate(ids):
            # Read image
            img_path = os.path.join(class_folder, id_)
            img, _ = load_img_data(img_path, is_3d=is_3d)

            if reflect_to_complete_shape and crop_shape:
                img = pad_and_reflect(img, crop_shape, verbose=False)

            if crop_shape and crop_shape[-1] == 3 and convert_to_rgb and img.shape[-1] != 3:
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
            if crop_shape and crop_shape[-1] != img.shape[-1]:
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

            xdataset_info.append(
                DatasetFile(
                    path=img_path,
                    shape=img.shape,
                    class_name=class_name,
                    class_num=c_num if gt_loaded else -1,
                )
            )
            sample_dict = DataSample(
                fid=data_file_count,
                coords=None,
            )

            xsample_list.append(sample_dict)
            data_file_count += 1

    return BiaPyDataset(dataset_info=xdataset_info, sample_list=xsample_list)


def filter_samples_by_properties(
    x_dataset: BiaPyDataset,
    is_3d: bool,
    filter_conds: List[List[str]],
    filter_vals: List[List[int | float]],
    filter_signs: List[List[str]],
    crop_shape: Tuple[int, ...],
    reflect_to_complete_shape: bool = False,
    filter_by_entire_image: bool = True,
    norm_before_filter: bool = False,
    norm_module: Optional[Normalization] = None,
    y_dataset: Optional[BiaPyDataset] = None,
    zarr_data_information: Optional[Dict] = None,
    save_filtered_images: bool = True,
    save_filtered_images_dir: Optional[str] = None,
    save_filtered_images_num: int = 3,
):
    """
    Filter samples from ``x_dataset`` using defined conditions. The filtering will be done using the images each sample is extracted
    from. However, if ``zarr_data_info`` is provided the function will assume that Zarr/h5 files are provided, so the filtering will be
    performed sample by sample.

    Parameters
    ----------
    x_dataset : BiaPyDataset
        X dataset to filter samples from.

    is_3d: bool, optional
        Whether the data to load is expected to be 3D or not.

    filter_conds : list of lists of str
        Filter conditions to be applied. The three variables, ``filter_conds``, ``filter_vals`` and ``filter_vals`` will compose a
        list of conditions to remove the images from the list. They are list of list of conditions. For instance, the conditions can
        be like this: ``[['A'], ['B','C']]``. Then, if the sample satisfies the first list of conditions, only 'A' in this first case
        (from ['A'] list), or satisfy 'B' and 'C' (from ['B','C'] list) it will be removed. In each sublist all the conditions must be
        satisfied. Available properties are: [``'foreground'``, ``'mean'``, ``'min'``, ``'max'``, ``diff``, ``target_mean``,
        ``target_min``, ``target_max``]. Each property descrition:
        * ``'foreground'`` is defined as the mask foreground percentage.
        * ``'mean'`` is defined as the mean value.
        * ``'min'`` is defined as the min value.
        * ``'max'`` is defined as the max value.
        * ``'diff'`` is defined as the difference between ground truth and raw images. Require ``y_dataset`` to be provided.
        * ``'diff_by_min_max_ratio'`` is defined as the difference between ground truth and raw images multiplied by the ratio
          between raw image max and min.
        * ``'target_mean'`` is defined as the mean intensity value of the raw image targets. Require ``y_dataset`` to be provided.
        * ``'target_min'`` is defined as the min intensity value of the raw image targets. Require ``y_dataset`` to be provided.
        * ``'target_max'`` is defined as the max intensity value of the raw image targets. Require ``y_dataset`` to be provided.
        * ``'diff_by_target_min_max_ratio'`` is defined as the difference between ground truth and raw images multiplied by the
          ratio between ground truth image max and min.

    filter_vals : list of int/float
        Represent the values of the properties listed in ``filter_conds`` that the images need to satisfy to not be dropped.

    filter_signs  :list of list of str
        Signs to do the comparison. Options: [``'gt'``, ``'ge'``, ``'lt'``, ``'le'``] that corresponds to "greather than", e.g. ">",
        "greather equal", e.g. ">=", "less than", e.g. "<", and "less equal" e.g. "<=" comparisons.

    crop_shape : 3D/4D int tuple
        Shape of the crops. E.g. ``(y, x, channels)`` for 2D and ``(z, y, x, channels)`` for 3D.

    reflect_to_complete_shape : bool, optional
        Wheter to increase the shape of the dimension that have less size than selected patch size padding it with
        'reflect'.

    filter_by_entire_image : bool, optional
        This decides how the filtering is done:
            * ``True``: apply filter image by image.
            * ``False``: apply filtering sample by sample. Each sample represents a patch within an image.

    y_dataset : BiaPyDataset, optional
        Y dataset to filter samples from.

    zarr_data_info : dict, optional
        Additional information when using Zarr/H5 files for training. The following keys are expected:
            * ``"raw_path"``: path where the raw images reside within the zarr (used when ``multiple_data_within_zarr`` is ``True``).
            * ``"gt_path"``: path where the mask images reside within the zarr (used when ``multiple_data_within_zarr`` is ``True``).
            * ``"multiple_data_within_zarr"``: Whether if your input Zarr contains the raw images and labels together or not.
            * ``"input_img_axes"``: order of the axes of the images.
            * ``"input_mask_axes"``: order of the axes of the masks.

    save_filtered_images : bool, optional
        Whether to save or not filtered images.

    save_filtered_images_dir : str, optional
        Directory to save filtered images.

    save_filtered_images_num : int, optional
        Number of filtered images to save. Only work when ``save_filtered_images`` is ``True``.

    Returns
    -------
    new_x_filenames : list of dict
        ``x_dataset`` list filtered.

    new_y_filenames : list of dict, optional
        ``y_dataset`` list filtered.
    """
    if norm_before_filter and norm_module is None:
        raise ValueError("'norm_module' can not be None when 'norm_before_filter' is active")

    if save_filtered_images:
        if not save_filtered_images_dir:
            raise ValueError("'save_filtered_images_dir' can not be None when 'save_filtered_images' is enabled")
        save_filtered_images_count = 0
        save_not_filtered_images_count = 0

    # Filter samples by properties
    print("Applying filtering to data samples . . .")

    use_Y_data = False
    for cond in filter_conds:
        if (
            "foreground" in cond
            or "diff" in cond
            or "diff_by_min_max_ratio" in cond
            or "diff_by_target_min_max_ratio" in cond
            or "target_mean" in cond
            or "target_min" in cond
            or "target_max" in cond
        ):
            use_Y_data = True

    if use_Y_data and y_dataset is None:
        raise ValueError("Check filtering conditions as some of them require 'y_dataset' that was not provided")

    using_zarr = False
    if zarr_data_information:
        using_zarr = True
        print("Assuming we are working with Zarr/H5 images so the filtering will be done patch by patch.")
        print(f"Number of samples before filtering: {len(x_dataset.sample_list)}")
    else:
        if filter_by_entire_image:
            images = [x.path for x in x_dataset.dataset_info]
            images.sort()
            if use_Y_data and y_dataset:
                masks = [x.path for x in y_dataset.dataset_info]
                masks.sort()
            print(f"Number of samples before filtering: {len(images)}")
        else:
            print(f"Number of samples before filtering: {len(x_dataset.sample_list)}")

    if not using_zarr and filter_by_entire_image:
        clean_by = "image"
        samples_to_maintain = []
        for n, image_path in tqdm(enumerate(images), total=len(images), disable=not is_main_process()):
            # Load X data
            img, _ = load_img_data(image_path, is_3d=is_3d)

            # Load Y data
            if use_Y_data:
                mask, _ = load_img_data(masks[n], is_3d=is_3d)
            else:
                mask = None

            if norm_before_filter:
                assert norm_module is not None
                img, _ = norm_module.apply_image_norm(img)
                if use_Y_data:
                    assert mask is not None
                    mask, _ = norm_module.apply_mask_norm(mask)
                    assert isinstance(mask, np.ndarray)
            assert isinstance(img, np.ndarray)
            satisfy_conds = sample_satisfy_conds(
                img,
                filter_conds,
                filter_vals,
                filter_signs,
                mask=mask,
                img_ratio=float(img.max())-float(img.min()),
                mask_ratio=(float(mask.max()) - float(mask.min())) if mask is not None else 0,
            )

            if not satisfy_conds:
                samples_to_maintain.append(n)
                if (
                    save_filtered_images
                    and save_filtered_images_dir
                    and save_not_filtered_images_count < save_filtered_images_num
                ):
                    save_tif(
                        np.expand_dims(img, 0),
                        os.path.join(save_filtered_images_dir, "not-filtered"),
                        [os.path.basename(image_path)],
                        verbose=False,
                    )
                    save_not_filtered_images_count += 1
            else:
                print(f"Discarding file {image_path}")
                if (
                    save_filtered_images
                    and save_filtered_images_dir
                    and save_filtered_images_count < save_filtered_images_num
                ):
                    save_tif(
                        np.expand_dims(img, 0),
                        os.path.join(save_filtered_images_dir, "filtered"),
                        [os.path.basename(image_path)],
                        verbose=False,
                    )
                    save_filtered_images_count += 1
    else:
        img_path, mask_path = "", ""
        clean_by = "sample"
        samples_to_maintain = []
        file, mfile, mask = None, None, None
        for n, sample in tqdm(
            enumerate(x_dataset.sample_list), total=len(x_dataset.sample_list), disable=not is_main_process()
        ):
            # Load X data
            filepath = x_dataset.dataset_info[sample.fid].path
            if img_path != filepath:
                old_img_path = img_path
                img_path = filepath
                if file and isinstance(file, h5py.File):
                    file.close()
                data_within_zarr_path = (
                    zarr_data_information["raw_path"]
                    if zarr_data_information and zarr_data_information["multiple_data_within_zarr"]
                    else None
                )
                xdata, file = load_img_data(img_path, is_3d=is_3d, data_within_zarr_path=data_within_zarr_path)

                if reflect_to_complete_shape and crop_shape:
                    xdata = pad_and_reflect(xdata, crop_shape, verbose=False)

                # Load Y data
                if use_Y_data:
                    assert y_dataset is not None
                    filepath = y_dataset.dataset_info[sample.fid].path
                    mask_path = filepath
                    if mfile and isinstance(mfile, h5py.File):
                        mfile.close()
                    data_within_zarr_path = None
                    if zarr_data_information and zarr_data_information["multiple_data_within_zarr"]:
                        data_within_zarr_path = (
                            zarr_data_information["gt_path"] if zarr_data_information["use_gt_path"] else None
                        )
                    ydata, mfile = load_img_data(mask_path, is_3d=is_3d, data_within_zarr_path=data_within_zarr_path)

                    if reflect_to_complete_shape and crop_shape:
                        ydata = pad_and_reflect(ydata, crop_shape, verbose=False)

                else:
                    ydata, mfile = None, None

                if norm_before_filter:
                    assert norm_module is not None
                    norm_module.set_stats_from_DatasetFile(x_dataset.dataset_info[sample.fid])
                    xdata, _ = norm_module.apply_image_norm(xdata)
                    if use_Y_data:
                        assert ydata is not None and y_dataset is not None
                        norm_module.set_stats_from_DatasetFile(y_dataset.dataset_info[sample.fid])
                        ydata, _ = norm_module.apply_mask_norm(ydata)

                if save_filtered_images and save_filtered_images_dir:
                    if "xdata_fil_example" in locals():
                        save_tif(
                            np.expand_dims(xdata_fil_example, 0),
                            save_filtered_images_dir,
                            [os.path.basename(old_img_path)],
                            verbose=True,
                        )
                        save_filtered_images_count += 1
                    if save_filtered_images_count == save_filtered_images_num:
                        del xdata_fil_example
                        save_filtered_images_count += 1
                    elif save_filtered_images_count < save_filtered_images_num:
                        xdata_fil_example = np.zeros(xdata.shape, dtype=xdata.dtype)  # type: ignore

            # Capture patches within image/mask
            coords = sample.coords
            if use_Y_data:
                assert y_dataset is not None
                mcoords = y_dataset.sample_list[n].coords

            # Prepare slices to extract the patch
            assert coords is not None
            if is_3d:
                xslices = (
                    slice(None),
                    slice(coords.z_start, coords.z_end),
                    slice(coords.y_start, coords.y_end),
                    slice(coords.x_start, coords.x_end),
                    slice(None),
                )
            else:
                xslices = (
                    slice(None),
                    slice(coords.y_start, coords.y_end),
                    slice(coords.x_start, coords.x_end),
                    slice(None),
                )
            if zarr_data_information:
                xdata_ordered_slices = order_dimensions(
                    xslices,
                    input_order="TZYXC",
                    output_order=zarr_data_information["input_img_axes"],
                    default_value=0,
                )
            else:
                xdata_ordered_slices = tuple([x for x in xslices if x != slice(None)])

            if use_Y_data:
                assert mcoords is not None
                if is_3d:
                    yslices = (
                        slice(None),
                        slice(mcoords.z_start, mcoords.z_end),
                        slice(mcoords.y_start, mcoords.y_end),
                        slice(mcoords.x_start, mcoords.x_end),
                        slice(None),
                    )
                else:
                    yslices = (
                        slice(None),
                        slice(mcoords.y_start, mcoords.y_end),
                        slice(mcoords.x_start, mcoords.x_end),
                        slice(None),
                    )
                if zarr_data_information:
                    ydata_ordered_slices = order_dimensions(
                        yslices,
                        input_order="TZYXC",
                        output_order=zarr_data_information["input_mask_axes"],
                        default_value=0,
                    )
                else:
                    ydata_ordered_slices = tuple([x for x in yslices if x != slice(None)])

            img = xdata[xdata_ordered_slices]  # type: ignore
            if use_Y_data:
                assert ydata is not None
                mask = ydata[ydata_ordered_slices]  # type: ignore
                assert isinstance(mask, np.ndarray)

            assert isinstance(img, np.ndarray)
            satisfy_conds = sample_satisfy_conds(
                img,
                filter_conds,
                filter_vals,
                filter_signs,
                mask=mask,
                img_ratio=(float(xdata.max())-float(xdata.min())),
                mask_ratio=(float(ydata.max())-float(ydata.min())) if ydata is not None else 0,
            )

            if not satisfy_conds:
                samples_to_maintain.append(n)
                if save_filtered_images and "xdata_fil_example" in locals():
                    xdata_fil_example[xdata_ordered_slices] = img

    if (
        save_filtered_images
        and save_filtered_images_dir
        and "xdata_fil_example" in locals()
        and save_filtered_images_count <= save_filtered_images_num
    ):
        save_tif(
            np.expand_dims(xdata_fil_example, 0),
            save_filtered_images_dir,
            [os.path.basename(img_path)],
            verbose=True,
        )
    del xdata_fil_example

    x_dataset.clean_dataset(samples_to_maintain, clean_by=clean_by)
    if y_dataset:
        y_dataset.clean_dataset(samples_to_maintain, clean_by=clean_by)
    number_of_samples = len(samples_to_maintain)

    if number_of_samples == 0:
        raise ValueError(
            "Filters set with 'DATA.TRAIN.FILTER_SAMPLES.*' variables led to discard all training samples. Aborting!"
        )
    elif number_of_samples == 1:
        raise ValueError(
            "Filters set with 'DATA.TRAIN.FILTER_SAMPLES.*' variables led to discard all training samples but one. Aborting!"
        )

    print(f"Number of samples after filtering: {number_of_samples}")


def sample_satisfy_conds(
    img: NDArray,
    filter_conds: List[List[str]],
    filter_vals: List[List[float | int]],
    filter_signs: List[List[str]],
    mask: Optional[NDArray] = None,
    img_ratio: float = 0,
    mask_ratio: float = 0,
) -> bool:
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
        * ``'mean'`` is defined as the mean value of the input.
        * ``'min'`` is defined as the min value of the input.
        * ``'max'`` is defined as the max value of the input.
        * ``'diff'`` is defined as the difference between ground truth and raw images. Require ``y_dataset`` to be provided.
        * ``'diff_by_min_max_ratio'`` is defined as the difference between ground truth and raw images multiplied by the ratio
          between raw image max and min.
        * ``'target_mean'`` is defined as the mean intensity value of the raw image targets. Require ``y_dataset`` to be provided.
        * ``'target_min'`` is defined as the min intensity value of the raw image targets. Require ``y_dataset`` to be provided.
        * ``'target_max'`` is defined as the max intensity value of the raw image targets. Require ``y_dataset`` to be provided.
        * ``'diff_by_target_min_max_ratio'`` is defined as the difference between ground truth and raw images multiplied by the
          ratio between ground truth image max and min.

    filter_vals : list of int/float
        Represent the values of the properties listed in ``filter_conds`` that the images need to satisfy to not be dropped.

    filter_signs : list of list of str
        Signs to do the comparison. Options: [``'gt'``, ``'ge'``, ``'lt'``, ``'le'``] that corresponds to "greather than", e.g. ">",
        "greather equal", e.g. ">=", "less than", e.g. "<", and "less equal" e.g. "<=" comparisons.

    mask : 4D/5D Numpy array, optional
        Mask to check if satisfy "foreground" condition in ``filter_conds``. E.g. ``(z, y, x, num_classes)`` for 3D or
        ``(y, x, num_classes)`` for 2D.

    img_ratio : float, optional
        Ratio of the input image. Expected to be ``(img.max - img.min)`` of the entire image. 

    mask_ratio : float, optional
        Minimum value of the entire image. Expected to be ``(mask.max - mask.min)`` of the entire image.

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
                assert mask is not None
                labels, npixels = np.unique((mask > 0).astype(np.uint8), return_counts=True)

                total_pixels = 1
                for val in list(mask.shape):
                    total_pixels *= val

                if labels[0] == 0:
                    npixels = npixels[1:]
                value_to_compare = sum(npixels) / total_pixels
            elif c == "diff":
                assert mask is not None
                value_to_compare = np.sum(abs(img - mask))
            elif c == "diff_by_min_max_ratio":
                assert mask is not None
                value_to_compare = np.sum(abs(img - mask)) * img_ratio
            elif c == "diff_by_target_min_max_ratio":
                assert mask is not None
                value_to_compare = np.sum(abs(img - mask)) * mask_ratio
            elif c == "min":
                value_to_compare = img.min()
            elif c == "max":
                value_to_compare = img.max()
            elif c == "mean":
                value_to_compare = img.mean()
            elif c == "target_min":
                assert mask is not None
                value_to_compare = mask.min()
            elif c == "target_max":
                assert mask is not None
                value_to_compare = mask.max()
            elif c == "target_mean":
                assert mask is not None
                value_to_compare = mask.mean()

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


def load_images_to_dataset(
    dataset: BiaPyDataset,
    crop_shape: Optional[Tuple[int, ...]],
    reflect_to_complete_shape: bool = False,
    convert_to_rgb: bool = False,
    is_mask: bool = False,
    is_3d: bool = False,
    preprocess_cfg: Optional[Dict] = None,
    preprocess_f: Optional[Callable] = None,
    zarr_data_information: Optional[Dict] = None,
):
    """
    Load images into the ``dataset``: creating ``"img"`` key. The process done faster
    if the samples extracted from the same image are in continuous positions within the list.

    Parameters
    ----------
    dataset : BiaPyDataset
        Loaded data.

    crop_shape : 3D/4D int tuple
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
    if preprocess_f and preprocess_cfg == None:
        raise ValueError("The preprocessing configuration ('preprocess_cfg') is missing.")

    channel_expected = -1
    data_range_expected = -1
    img_path = ""
    file = None
    for sample in tqdm(dataset.sample_list, total=len(dataset.sample_list), disable=not is_main_process()):
        # Read image if it is different from the last sample's
        filepath = dataset.dataset_info[sample.fid].path
        if img_path != filepath:
            img_path = filepath
            if file and isinstance(file, h5py.File):
                file.close()
            data_within_zarr_path = None
            if zarr_data_information and zarr_data_information["multiple_data_within_zarr"]:
                if not is_mask:
                    data_within_zarr_path = zarr_data_information["raw_path"]
                else:
                    data_within_zarr_path = (
                        zarr_data_information["gt_path"] if zarr_data_information["use_gt_path"] else None
                    )
            data, file = load_img_data(img_path, is_3d=is_3d, data_within_zarr_path=data_within_zarr_path)

            # Disable channel checking if it is not present. Can happen with a Zarr/H5 dataset, as in the load_img_data()
            # the axis were not checked to not have the data loaded in memory
            check_channel = True
            key = "input_img_axes" if not is_mask else "input_mask_axes"
            if zarr_data_information and "C" not in zarr_data_information[key]:
                check_channel = False

            # Channel check within dataset images
            if channel_expected == -1:
                channel_expected = data.shape[-1]
            if check_channel and data.shape[-1] != channel_expected:
                raise ValueError(
                    f"All images need to have the same number of channels and represent same information to "
                    "ensure the deep learning model can be trained correctly. However, the current image (with "
                    f"{channel_expected} channels) appears to have a different number of channels than the first image"
                    f"(with {data.shape[-1]} channels) in the folder. Current image: {img_path}"
                )

            # Channel check compared with crop_shape
            if check_channel and crop_shape and not is_mask:
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
            if preprocess_f:
                if is_mask:
                    data = preprocess_f(preprocess_cfg, y_data=[data], is_2d=not is_3d, is_y_mask=is_mask)[0]
                else:
                    data = preprocess_f(preprocess_cfg, x_data=[data], is_2d=not is_3d)[0]

        # Prepare slices to extract the patch
        if sample.coords and sample.coords:
            coords = sample.coords
            if is_3d:
                xslices = (
                    slice(None),
                    slice(coords.z_start, coords.z_end),
                    slice(coords.y_start, coords.y_end),
                    slice(coords.x_start, coords.x_end),
                    slice(None),
                )
            else:
                xslices = (
                    slice(None),
                    slice(coords.y_start, coords.y_end),
                    slice(coords.x_start, coords.x_end),
                    slice(None),
                )

            if zarr_data_information:
                data_ordered_slices = order_dimensions(
                    xslices,
                    input_order="TZYXC",
                    output_order=zarr_data_information[key],
                    default_value=0,
                )
            else:
                data_ordered_slices = xslices[1:]

            # Extract the patch within the image
            img = data[data_ordered_slices]  # type: ignore

            if zarr_data_information:
                img = ensure_3d_shape(img.squeeze(), path=filepath)
        else:
            img = data

        if crop_shape and reflect_to_complete_shape:
            img = pad_and_reflect(img, crop_shape, verbose=False)

        if crop_shape and crop_shape[-1] == 3 and convert_to_rgb and not is_mask and img.shape[-1] != 3:
            img = np.repeat(img, 3, axis=-1)

        # Insert the image
        sample.img = img

    sshape = dataset.sample_list[0].get_shape()
    if sshape:
        data_shape = (len(dataset.sample_list),) + sshape
        print("*** Loaded data shape is {}".format(data_shape))
    else:
        print(
            "Samples of shape {} will be randomly extracted. Number of samples: {}".format(
                crop_shape, len(dataset.sample_list)
            )
        )


def pad_and_reflect(img: NDArray, crop_shape: Tuple[int, ...], verbose: bool = False) -> NDArray:
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
        raise ValueError(
            f"'crop_shape' needs to have 4 values as the input array has 4 dims. Provided crop_shape: {crop_shape}"
        )
    if img.ndim == 3 and len(crop_shape) != 3:
        raise ValueError(
            f"'crop_shape' needs to have 3 values as the input array has 3 dims. Provided crop_shape: {crop_shape}"
        )

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


def img_to_onehot_encoding(img: NDArray, num_classes: int = 2) -> NDArray:
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


def onehot_encoding_to_img(encoded_image: NDArray) -> NDArray:
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


def load_img_data(
    path: str, is_3d: bool = False, data_within_zarr_path: Optional[str] = None
) -> Tuple[NDArray[Any], str]:
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
        Data read. E.g. ``(z, y, x, channels)`` for 3D or ``(y, x, channels)`` for 2D.

    file : str
        File of the data read. Useful to close it in case it is an H5 file.
    """
    if any(path.endswith(x) for x in [".zarr", ".h5", ".hdf5", ".hdf"]):
        from biapy.data.data_3D_manipulation import (
            read_chunked_data,
            read_chunked_nested_data,
        )

        if data_within_zarr_path:
            file, data = read_chunked_nested_data(path, data_within_zarr_path)
        else:
            file, data = read_chunked_data(path)
    else:
        data = read_img_as_ndarray(path, is_3d=is_3d)
        file = path

    return data, file  # type: ignore


def read_img_as_ndarray(path: str, is_3d: bool = False) -> NDArray:
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
        Image read. E.g. ``(z, y, x, channels)`` for 3D or ``(y, x, channels)`` for 2D.
    """
    # Read image
    if path.endswith(".npy"):
        img = np.load(path)
    elif path.endswith(".pt"):
        img = torch.load(path, weights_only=True, map_location="cpu").numpy()
    elif any(path.endswith(x) for x in [".h5", ".hdf5", ".hdf"]):
        img = h5py.File(path, "r")
        img = np.array(img[list(img)[0]])
    elif path.endswith(".zarr"):
        from biapy.data.data_3D_manipulation import read_chunked_data

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


def imread(path: str) -> NDArray:
    """
    Read an image from a given path. In the past from ``skimage.io import imread``
    was used but now it is deprecated.

    Parameters
    ----------
    path : str
        Path to the image to read.

    Returns
    -------
    img : Numpy array
        Image read.
    """
    if path.lower().endswith((".tiff", ".tif")):
        return tifffile.imread(path)
    else:
        return imageio.imread(path)


def imwrite(path: str, image: NDArray):
    """
    Writes ``data`` in the given ``path``. In the past from ``skimage.io import imsave``
    was used but now it is deprecated.

    Parameters
    ----------
    path : str
        Path to the image to read.

    image : Numpy array
        Image to store.
    """
    image = np.array(image)
    if path.lower().endswith((".tiff", ".tif")):
        assert image.ndim == 6, f"Image to write needs to have 6 dimensions (axes: TZCYXS). Image shape: {image.shape}"
        try:
            tifffile.imwrite(
                path,
                image,
                imagej=True,
                metadata={"axes": "TZCYXS"},
                compression="zlib",
                compressionargs={"level": 8},
            )
        except:
            tifffile.imwrite(path, image, imagej=True, metadata={"axes": "TZCYXS"})
    else:
        imageio.imwrite(path, image)


def check_value(
    value: int | float | Tuple[int | float] | List[int | float] | NDArray,
    value_range: Tuple[int | float, int | float] = (0, 1),
) -> bool:
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


def data_range(x: NDArray) -> str:
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


def check_masks(path: str, n_classes: int = 2, is_3d: bool = False):
    """
    Check whether the data masks have the correct labels inspection a few random images of the given path. If the
    function gives no error one should assume that the masks are correct.

    Parameters
    ----------
    path : str
        Path to the data mask.

    n_classes : int, optional
        Maximum classes that the masks must contain.

    is_3d : bool, optional
        Whether if the expected image to read is 3D or not.
    """
    print("Checking ground truth classes in {} . . .".format(path))

    ids = sorted(next(os.walk(path))[2])
    classes_found = []
    m = ""
    error = False
    for i in tqdm(range(len(ids))):
        if any(ids[i].endswith(x) for x in [".zarr", ".h5", ".hdf5", ".hdf"]):
            raise ValueError(
                "Mask checking with Zarr not implemented in BiaPy yet. Disable 'DATA.*.CHECK_DATA' variables to continue"
            )
        else:
            img = read_img_as_ndarray(os.path.join(path, ids[i]), is_3d=is_3d)
            values = np.unique(img)
            if len(values) > n_classes:
                print(
                    "Error: given mask ({}) has more classes than specified in 'MODEL.N_CLASSES'. "
                    "Values found: {}".format(os.path.join(path, ids[i]), values)
                )
                error = True

            classes_found += list(values)
            classes_found = list(set(classes_found))

    if len(classes_found) > n_classes:
        m += (
            "Number of classes found across images is greater than the value specified in 'MODEL.N_CLASSES'. "
            f"Classes found: {classes_found}\n"
        )
        error = True

    if error:
        m += (
            "'MODEL.N_CLASSES' variable value must be set taking into account the background class. E.g. if mask has [0,1,2] "
            "values 'MODEL.N_CLASSES' should be 3.\nCorrect the errors in the masks above to continue"
        )
        raise ValueError(m)


def shape_mismatch_message(X_data: BiaPyDataset, Y_data: BiaPyDataset) -> str:
    """
    Builds an error message with the shape mismatch between two provided data ``X_data`` and ``Y_data``.

    Parameters
    ----------
    X_data : BiaPyDataset
        X data.

    Y_data : BiaPyDataset
        Y data.

    Returns
    -------
    mistmatch_message : str
        Message containing which samples mismatch.
    """
    mistmatch_message = ""
    for xsample, ysample in zip(X_data.dataset_info, Y_data.dataset_info):
        if xsample.get_shape()[:-1] != ysample.get_shape()[:-1]:
            mistmatch_message += "\n"
            mistmatch_message += "Raw file: '{}'\n".format(xsample.path)
            mistmatch_message += "Corresponding label file: '{}'\n".format(ysample.path)
            mistmatch_message += "Raw shape: {}\n".format(xsample.get_shape())
            mistmatch_message += "Label shape: {}\n".format(ysample.get_shape())
            mistmatch_message += "--\n"

    if mistmatch_message != "":
        mistmatch_message = (
            f"Here is a list of the pair raw and label that does not match in shape:\n{mistmatch_message}"
        )

    return mistmatch_message


def save_tif(X: NDArray, data_dir: str, filenames: Optional[List[str]] = None, verbose: bool = True):
    """
    Save images in the given directory.

    Parameters
    ----------
    X : 4D/5D numpy array
        Data to save as images. The first dimension must be the number of images. E.g.
        ``(num_of_images, y, x, channels)`` or ``(num_of_images, z, y, x, channels)``.

    data_dir : str
        Path to store X images.

    filenames : List, optional
        Filenames that should be used when saving each image.

    verbose : bool, optional
         To print saving information.
    """

    if verbose:
        s = X.shape if not isinstance(X, list) else X[0].shape
        print("Saving {} data as .tif in folder: {}".format(s, data_dir))

    os.makedirs(data_dir, exist_ok=True)
    if filenames:
        if len(filenames) != len(X):
            raise ValueError(
                "Filenames array and length of X have different shapes: {} vs {}".format(len(filenames), len(X))
            )

    if not isinstance(X, list):
        _dtype = X.dtype if X.dtype in [np.uint8, np.uint16, np.float32] else np.float32
        ndims = X.ndim
    else:
        _dtype = X[0].dtype if X[0].dtype in [np.uint8, np.uint16, np.float32] else np.float32
        ndims = X[0].ndim

    d = len(str(len(X)))
    for i in tqdm(range(len(X)), leave=False, disable=not is_main_process()):
        if filenames is None:
            f = os.path.join(data_dir, str(i).zfill(d) + ".tif")
        else:
            f = os.path.join(data_dir, os.path.splitext(filenames[i])[0] + ".tif")
        if ndims == 4:
            if not isinstance(X, list):
                aux = np.expand_dims(np.expand_dims(X[i], 0).transpose((0, 3, 1, 2)), -1).astype(_dtype)
            else:
                aux = np.expand_dims(np.expand_dims(X[i][0], 0).transpose((0, 3, 1, 2)), -1).astype(_dtype)
        else:
            if not isinstance(X, list):
                aux = np.expand_dims(X[i].transpose((0, 3, 1, 2)), -1).astype(_dtype)
            else:
                aux = np.expand_dims(X[i][0].transpose((0, 3, 1, 2)), -1).astype(_dtype)
        imwrite(f, np.expand_dims(aux, 0))


def save_tif_pair_discard(
    X: NDArray,
    Y: NDArray,
    data_dir: str,
    suffix: str = "",
    filenames: Optional[List] = None,
    discard: bool = True,
    verbose: bool = True,
):
    """
    Save images in the given directory.

    Parameters
    ----------
    X : 4D/5D numpy array
        Data to save as images. The first dimension must be the number of images. E.g.
        ``(num_of_images, y, x, channels)`` or ``(num_of_images, z, y, x, channels)``.

    Y : 4D/5D numpy array
        Data mask to save. The first dimension must be the number of images. E.g.
        ``(num_of_images, y, x, channels)`` or ``(num_of_images, z, y, x, channels)``.

    data_dir : str
        Path to store X images.

    suffix : str, optional
        Suffix to apply on output directory.

    filenames : List, optional
        Filenames that should be used when saving each image.

    discard : bool, optional
        Whether to discard image/mask pairs if the mask has no label information.

    verbose : bool, optional
         To print saving information.
    """

    if verbose:
        s = X.shape if not isinstance(X, list) else X[0].shape
        print("Saving {} data as .tif in folder: {}".format(s, data_dir))

    os.makedirs(os.path.join(data_dir, "x" + suffix), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "y" + suffix), exist_ok=True)
    if filenames:
        if len(filenames) != len(X):
            raise ValueError(
                "Filenames array and length of X have different shapes: {} vs {}".format(len(filenames), len(X))
            )

    _dtype = X.dtype if X.dtype in [np.uint8, np.uint16, np.float32] else np.float32
    d = len(str(len(X)))
    for i in tqdm(range(X.shape[0]), leave=False, disable=not is_main_process()):
        if len(np.unique(Y[i])) >= 2 or not discard:
            if filenames is None:
                f1 = os.path.join(data_dir, "x" + suffix, str(i).zfill(d) + ".tif")
                f2 = os.path.join(data_dir, "y" + suffix, str(i).zfill(d) + ".tif")
            else:
                f1 = os.path.join(data_dir, "x" + suffix, os.path.splitext(filenames[i])[0] + ".tif")
                f2 = os.path.join(data_dir, "y" + suffix, os.path.splitext(filenames[i])[0] + ".tif")
            if X.ndim == 4:
                aux = np.expand_dims(np.expand_dims(X[i], 0).transpose((0, 3, 1, 2)), -1).astype(_dtype)
            else:
                aux = np.expand_dims(X[i].transpose((0, 3, 1, 2)), -1).astype(_dtype)
            imwrite(f1, np.expand_dims(aux, 0))
            if Y.ndim == 4:
                aux = np.expand_dims(np.expand_dims(Y[i], 0).transpose((0, 3, 1, 2)), -1).astype(_dtype)
            else:
                aux = np.expand_dims(Y[i].transpose((0, 3, 1, 2)), -1).astype(_dtype)
            imwrite(f2, np.expand_dims(aux, 0))


def save_npy_files(X: NDArray, data_dir: str, filenames: Optional[List[str]] = None, verbose: bool = True):
    """
    Save images in the given directory.

    Parameters
    ----------
    X : 4D/5D numpy array
        Data to save as images. The first dimension must be the number of images. E.g.
        ``(num_of_images, y, x, channels)`` or ``(num_of_images, z, y, x, channels)``.

    data_dir : str
        Path to store X images.

    filenames : List, optional
        Filenames that should be used when saving each image.

    verbose : bool, optional
         To print saving information.
    """

    if verbose:
        s = X.shape if not isinstance(X, list) else X[0].shape
        print("Saving {} data as .npy in folder: {}".format(s, data_dir))

    os.makedirs(data_dir, exist_ok=True)
    if filenames:
        if len(filenames) != len(X):
            raise ValueError(
                "Filenames array and length of X have different shapes: {} vs {}".format(len(filenames), len(X))
            )

    d = len(str(len(X)))
    for i in tqdm(range(len(X)), leave=False, disable=not is_main_process()):
        if filenames is None:
            f = os.path.join(data_dir, str(i).zfill(d) + ".npy")
        else:
            f = os.path.join(data_dir, os.path.splitext(filenames[i])[0] + ".npy")
        if isinstance(X, list):
            np.save(f, X[i][0])
        else:
            np.save(f, X[i])


def reduce_dtype(
    x: NDArray,
    x_min: float,
    x_max: float,
    out_min: float = 0,
    out_max: float = 1,
    out_type: str = "float32",
    eps: float = 1e-6,
) -> NDArray:
    """
    Reduce the data type of the given input to the selected range using the formula:
    ``results = ((x - x_min)/(x_max - x_min)) * (out_max - out_min)``

    Parameters
    ----------
    x : 3D/4D Numpy array
        Image to reduce it's data type. E.g. ``(y, x, channels)`` in ``2D`` and ``(z, y, x, channels)`` in ``3D``.

    x_min: float
        ``x_min`` in the formula above.

    x_max: float
        ``x_max`` in the formula above.

    out_min: float, optional
        ``out_min`` in the formula above.

    out_max: float, optional
        ``out_max`` in the formula above.

    out_type : str, optional
        Type of the output data.

    eps : float, optional
        Epsilon to use in order to avoid zero division.

    Returns
    -------
    x : 3D/4D Numpy array
        Data type reduced image. E.g. ``(y, x, channels)`` in ``2D`` and ``(z, y, x, channels)`` in ``3D``.
    """
    from biapy.data.norm import torch_numpy_dtype_dict

    if isinstance(x, np.ndarray):
        if not isinstance(x, np.floating):
            x = x.astype(np.float32)
        return ((np.array((x - x_min) / (x_max - x_min + eps)) * (out_max - out_min)) + out_min).astype(
            torch_numpy_dtype_dict[out_type][1]
        )
    else:  # Tensor considered
        if not torch.is_floating_point(x):
            x = x.to(torch.float32)
        return ((((x - x_min) / (x_max - x_min + eps)) * (out_max - out_min)) + out_min).to(
            torch_numpy_dtype_dict[out_type][0]
        )
