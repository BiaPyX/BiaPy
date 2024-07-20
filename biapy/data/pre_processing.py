import os
import torch
import scipy
import h5py
import zarr
import sys
import numpy as np
from tqdm import tqdm
import pandas as pd
from skimage.segmentation import clear_border, find_boundaries
from scipy.ndimage.morphology import binary_dilation as binary_dilation_scipy
from scipy.ndimage.measurements import center_of_mass
from skimage.morphology import disk, dilation, binary_dilation
from skimage.measure import label, regionprops
from skimage.transform import resize
from skimage.feature import canny
from skimage.exposure import equalize_adapthist
from skimage.color import rgb2gray
from skimage.filters import gaussian, median

from biapy.utils.util import (
    load_data_from_dir,
    load_3d_images_from_dir,
    save_tif,
    seg2aff_pni,
    seg_widen_border,
    write_chunked_data,
    read_chunked_data,
    order_dimensions,
    read_img,
)
from biapy.utils.misc import is_main_process
from biapy.data.data_3D_manipulation import (
    load_3D_efficient_files,
    load_img_part_from_efficient_file,
)


#########################
# INSTANCE SEGMENTATION #
#########################
def create_instance_channels(cfg, data_type="train"):
    """
    Create training and validation new data with appropiate channels based on ``PROBLEM.INSTANCE_SEG.DATA_CHANNELS``
    for instance segmentation.

    Parameters
    ----------
    cfg : YACS CN object
        Configuration.

    data_type: str, optional
        Wheter to create training or validation instance channels.

    Returns
    -------
    filenames: List of str
        Image paths.
    """

    assert data_type in ["train", "val"]
    tag = "TRAIN" if data_type == "train" else "VAL"

    # Checking if the user inputted Zarr/H5 files
    if getattr(cfg.DATA, tag).INPUT_ZARR_MULTIPLE_DATA:
        zarr_files = sorted(next(os.walk(getattr(cfg.DATA, tag).PATH))[1])
        h5_files = sorted(next(os.walk(getattr(cfg.DATA, tag).PATH))[2])
    else:
        zarr_files = sorted(next(os.walk(getattr(cfg.DATA, tag).GT_PATH))[1])
        h5_files = sorted(next(os.walk(getattr(cfg.DATA, tag).GT_PATH))[2])

    # Find patches info so we can iterate over them to create the instance mask
    working_with_zarr_h5_files = False
    if (
        cfg.PROBLEM.NDIM == "3D"
        and (len(zarr_files) > 0 and ".zarr" in zarr_files[0])
        or (len(h5_files) > 0 and ".h5" in h5_files[0])
    ):
        working_with_zarr_h5_files = True
        # Check if the raw images and labels are within the same file
        mult_dat = None
        data_path = getattr(cfg.DATA, tag).GT_PATH
        if getattr(cfg.DATA, tag).INPUT_ZARR_MULTIPLE_DATA:
            data_path = getattr(cfg.DATA, tag).PATH

        if len(zarr_files) > 0 and ".zarr" in zarr_files[0]:
            print("Working with Zarr files . . .")
            img_files = [os.path.join(data_path, x) for x in zarr_files]
        elif len(h5_files) > 0 and ".h5" in h5_files[0]:
            print("Working with H5 files . . .")
            img_files = [os.path.join(data_path, x) for x in h5_files]

        Y, Y_total_patches = load_3D_efficient_files(
            img_files,
            input_axes=getattr(cfg.DATA, tag).INPUT_IMG_AXES_ORDER,
            crop_shape=cfg.DATA.PATCH_SIZE,
            overlap=getattr(cfg.DATA, tag).OVERLAP,
            padding=getattr(cfg.DATA, tag).PADDING,
        )
    else:
        Y = sorted(next(os.walk(getattr(cfg.DATA, tag).GT_PATH))[2])
    del zarr_files, h5_files

    print("Creating Y_{} channels . . .".format(data_type))
    # Create the mask patch by patch (Zarr/H5)
    if working_with_zarr_h5_files and isinstance(Y, dict):
        savepath = (
            data_path + "_" + cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS + "_" + cfg.PROBLEM.INSTANCE_SEG.DATA_CONTOUR_MODE
        )
        if "D" in cfg.PROBLEM.INSTANCE_SEG.DATA_CONTOUR_MODE:
            dtype_str = "float32"
            raise ValueError("Currently distance creation using Zarr by chunks is not implemented.")
        else:
            dtype_str = "uint8"

        mask = None
        last_zarr_file = None
        for i in tqdm(range(len(Y.keys())), disable=not is_main_process()):
            # Extract the patch to process
            patch_coords = Y[i]["patch_coords"]
            img = load_img_part_from_efficient_file(
                Y[i]["filepath"],
                patch_coords,
                data_axis_order=getattr(cfg.DATA, tag).INPUT_IMG_AXES_ORDER,
                data_path=getattr(cfg.DATA, tag).INPUT_ZARR_MULTIPLE_DATA_GT_PATH,
            )
            if img.ndim == 3:
                img = np.expand_dims(img, -1)
            if img.ndim == 4:
                img = np.expand_dims(img, 0)

            # Create the instance mask
            if cfg.MODEL.N_CLASSES > 2:
                if img.shape[-1] != 2:
                    raise ValueError(
                        "In instance segmentation, when 'MODEL.N_CLASSES' are more than 2 labels need to have two channels, "
                        "e.g. (256,256,2), containing the instance segmentation map (first channel) and classification map (second channel)."
                    )
                else:
                    class_channel = np.expand_dims(img[..., 1].copy(), -1)
                    img = labels_into_channels(
                        img,
                        mode=cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS,
                        save_dir=getattr(cfg.PATHS, tag + "_INSTANCE_CHANNELS_CHECK"),
                        fb_mode=cfg.PROBLEM.INSTANCE_SEG.DATA_CONTOUR_MODE,
                    )
                    img = np.concatenate([img, class_channel], axis=-1)
            else:
                img = labels_into_channels(
                    img,
                    mode=cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS,
                    save_dir=getattr(cfg.PATHS, tag + "_INSTANCE_CHANNELS_CHECK"),
                    fb_mode=cfg.PROBLEM.INSTANCE_SEG.DATA_CONTOUR_MODE,
                )
            img = img[0]

            # Create the Zarr file where the mask will be placed
            if mask is None or os.path.basename(Y[i]["filepath"]) != last_zarr_file:
                last_zarr_file = os.path.basename(Y[i]["filepath"])
                imgfile, data = read_chunked_data(Y[i]["filepath"])
                fname = os.path.join(savepath, os.path.basename(Y[i]["filepath"]))
                fid_mask = zarr.open_group(fname, mode="w")

                # Determine data shape
                out_data_shape = np.array(data.shape)
                if "C" not in getattr(cfg.DATA, tag).INPUT_IMG_AXES_ORDER:
                    out_data_shape = tuple(out_data_shape) + (img.shape[-1],)
                    out_data_order = getattr(cfg.DATA, tag).INPUT_IMG_AXES_ORDER + "C"
                    channel_pos = -1
                else:
                    out_data_shape[getattr(cfg.DATA, tag).INPUT_IMG_AXES_ORDER.index("C")] = 1
                    out_data_shape = tuple(out_data_shape)
                    out_data_order = getattr(cfg.DATA, tag).INPUT_IMG_AXES_ORDER
                    channel_pos = getattr(cfg.DATA, tag).INPUT_IMG_AXES_ORDER.index("C")

                mask = fid_mask.create_dataset("data", shape=out_data_shape, dtype=dtype_str)
                del data, imgfile, fname

            # Adjust slices to calculate where to insert the predicted patch. This slice does not have into account the
            # channel so any of them can be inserted
            slices = (
                slice(patch_coords[0][0], patch_coords[0][1]),
                slice(patch_coords[1][0], patch_coords[1][1]),
                slice(patch_coords[2][0], patch_coords[2][1]),
                slice(0, out_data_shape[channel_pos]),
            )
            data_ordered_slices = tuple(
                order_dimensions(
                    slices,
                    input_order="ZYXC",
                    output_order=out_data_order,
                    default_value=0,
                )
            )

            # Adjust patch slice to transpose it before inserting intop the final data
            current_order = np.array(range(len(img.shape)))
            transpose_order = order_dimensions(
                current_order,
                input_order="ZYXC",
                output_order=out_data_order,
                default_value=np.nan,
            )
            transpose_order = [x for x in transpose_order if not np.isnan(x)]

            # Place the patch into the Zarr
            mask[data_ordered_slices] = img.transpose(transpose_order)
    else:
        for i in tqdm(range(len(Y)), disable=not is_main_process()):
            img = read_img(
                os.path.join(getattr(cfg.DATA, tag).GT_PATH, Y[i]),
                is_3d=not cfg.PROBLEM.NDIM == "2D",
            )
            if cfg.MODEL.N_CLASSES > 2:
                if img.shape[-1] != 2:
                    raise ValueError(
                        "In instance segmentation, when 'MODEL.N_CLASSES' are more than 2 labels need to have two channels, "
                        "e.g. (256,256,2), containing the instance segmentation map (first channel) and classification map "
                        "(second channel)."
                    )
                class_channel = np.expand_dims(img[..., 1].copy(), -1)

            img = labels_into_channels(
                np.expand_dims(img, 0),
                mode=cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS,
                save_dir=getattr(cfg.PATHS, tag + "_INSTANCE_CHANNELS_CHECK"),
                fb_mode=cfg.PROBLEM.INSTANCE_SEG.DATA_CONTOUR_MODE,
            )[0]

            if cfg.MODEL.N_CLASSES > 2:
                img = np.concatenate([img, class_channel], axis=-1)

            save_tif(
                np.expand_dims(img, 0),
                data_dir=getattr(cfg.DATA, tag).INSTANCE_CHANNELS_MASK_DIR,
                filenames=[Y[i]],
                verbose=False,
            )

        X = sorted(next(os.walk(getattr(cfg.DATA, tag).PATH))[2])
        print("Creating X_{} channels . . .".format(data_type))
        for i in tqdm(range(len(X)), disable=not is_main_process()):
            img = read_img(
                os.path.join(getattr(cfg.DATA, tag).PATH, X[i]),
                is_3d=not cfg.PROBLEM.NDIM == "2D",
            )
            save_tif(
                np.expand_dims(img, 0),
                data_dir=getattr(cfg.DATA, tag).INSTANCE_CHANNELS_DIR,
                filenames=[X[i]],
                verbose=False,
            )

            # Save just three images to check everything was generated correctly
            if i < 3:
                save_tif(
                    np.expand_dims(img, 0),
                    getattr(cfg.PATHS, tag + "_INSTANCE_CHANNELS_CHECK"),
                    filenames=["vol" + str(i) + ".tif"],
                    verbose=False,
                )


def create_test_instance_channels(cfg):
    """
    Create test new data with appropiate channels based on ``PROBLEM.INSTANCE_SEG.DATA_CHANNELS`` for instance segmentation.

    Parameters
    ----------
    cfg : YACS CN object
        Configuration.
    """
    if cfg.DATA.TEST.LOAD_GT:
        Y = sorted(next(os.walk(cfg.DATA.TEST.GT_PATH)[2]))
        print("Creating Y_test channels . . .")
        for i in tqdm(range(len(Y)), disable=not is_main_process()):
            img = read_img(
                os.path.join(cfg.DATA.TEST.GT_PATH, Y[i]),
                is_3d=not cfg.PROBLEM.NDIM == "2D",
            )
            img = labels_into_channels(
                np.expand_dims(img, 0),
                mode=cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS,
                save_dir=cfg.PATHS.TEST_INSTANCE_CHANNELS_CHECK,
                fb_mode=cfg.PROBLEM.INSTANCE_SEG.DATA_CONTOUR_MODE,
            )[0]
            save_tif(
                np.expand_dims(img, 0),
                data_dir=cfg.DATA.TEST.INSTANCE_CHANNELS_MASK_DIR,
                filenames=[Y[i]],
                verbose=False,
            )

    X = sorted(next(os.walk(cfg.DATA.TEST.PATH)[2]))
    print("Creating X_test channels . . .")
    for i in tqdm(range(len(X)), disable=not is_main_process()):
        img = read_img(os.path.join(cfg.DATA.TEST.PATH, X[i]), is_3d=not cfg.PROBLEM.NDIM == "2D")
        save_tif(
            np.expand_dims(img, 0),
            data_dir=cfg.DATA.TEST.INSTANCE_CHANNELS_DIR,
            filenames=[X[i]],
            verbose=False,
        )

        # Save just three images to check everything was generated correctly
        if i < 3:
            save_tif(
                np.expand_dims(img, 0),
                cfg.PATHS.TEST_INSTANCE_CHANNELS_CHECK,
                filenames=["vol" + str(i) + ".tif"],
                verbose=False,
            )


def labels_into_channels(data_mask, mode="BC", fb_mode="outer", save_dir=None):
    """
    Converts input semantic or instance segmentation data masks into different binary channels to train an instance segmentation
    problem.

    Parameters
    ----------
    data_mask : 4D/5D Numpy array
        Data mask to create the new array from. It is expected to have just one channel. E.g. ``(10, 200, 1000, 1000, 1)``

    mode : str, optional
        Operation mode. Possible values: ``C``, ``BC``, ``BCM``, ``BCD``, ``BD``, ``BCDv2``, ``Dv2``, ``BDv2`` and ``BP``.
         - 'B' stands for 'Binary segmentation', containing each instance region without the contour.
         - 'C' stands for 'Contour', containing each instance contour.
         - 'D' stands for 'Distance', each pixel containing the distance of it to the center of the object.
         - 'M' stands for 'Mask', contains the B and the C channels, i.e. the foreground mask.
           Is simply achieved by binarizing input instance masks.
         - 'Dv2' stands for 'Distance V2', which is an updated version of 'D' channel calculating background distance as well.
         - 'P' stands for 'Points' and contains the central points of an instance (as in Detection workflow)
         - 'A' stands for 'Affinities" and contains the affinity values for each dimension

    fb_mode : str, optional
       Mode of the find_boundaries function from ``scikit-image`` or "dense". More info in:
       `find_boundaries() <https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.find_boundaries>`_.
       Choose "dense" to label as contour every pixel that is not in ``B`` channel.

    save_dir : str, optional
        Path to store samples of the created array just to debug it is correct.

    Returns
    -------
    new_mask : 5D Numpy array
        5D array with 3 channels instead of one. E.g. ``(10, 200, 1000, 1000, 3)``
    """

    assert data_mask.ndim in [5, 4]
    assert mode in ["C", "BC", "BCM", "BCD", "BD", "BCDv2", "Dv2", "BDv2", "BP", "A"]

    d_shape = 4 if data_mask.ndim == 5 else 3
    if mode in ["BCDv2", "Dv2", "BDv2"]:
        c_number = 4
    elif mode in ["BCD", "BCM"]:
        c_number = 3
    elif mode in ["BC", "BP", "BD"]:
        c_number = 2
    elif mode in ["C"]:
        c_number = 1
    elif mode in ["A"]:
        # the number of affinity channels depends on the dimensions of the input image
        c_number = 3 if data_mask.ndim == 5 else 2

    if "D" in mode:
        dtype = np.float32
    else:
        dtype = np.uint8

    new_mask = np.zeros(data_mask.shape[:d_shape] + (c_number,), dtype=dtype)
    for img in tqdm(range(data_mask.shape[0]), disable=not is_main_process()):
        vol = data_mask[img, ..., 0].astype(np.int64)
        instances = np.unique(vol)
        instance_count = len(instances)

        # Background distance
        if "Dv2" in mode:
            # Background distance
            vol_b_dist = np.invert(vol > 0)
            vol_b_dist = scipy.ndimage.distance_transform_edt(vol_b_dist)
            vol_b_dist = np.max(vol_b_dist) - vol_b_dist
            new_mask[img, ..., 3] = vol_b_dist.copy()
        # Affinities
        if "A" in mode:
            ins_vol = np.copy(vol)
            if fb_mode == "dense":
                ins_vol = seg_widen_border(vol)
                # if ins_vol.ndim == 3:
                #    for i in range(ins_vol.shape[0]):
                #        ins_vol[i,...] = erosion(vol[i,...], disk(1))
                # else:
                #    ins_vol = erosion(vol, disk(1))
            affs = seg2aff_pni(ins_vol, dtype=dtype)
            affs = np.transpose(affs, (1, 2, 3, 0))
            if c_number == 2:
                affs = np.squeeze(affs, 0)
            new_mask[img] = affs
        # Semantic mask
        if "B" in mode and instance_count != 1:
            new_mask[img, ..., 0] = (vol > 0).copy().astype(np.uint8)

        # Central points
        if "P" in mode and instance_count != 1:
            coords = center_of_mass(vol > 0, vol, instances[1:])
            coords = np.round(coords).astype(int)
            for coord in coords:
                if data_mask.ndim == 5:
                    z, y, x = coord
                    new_mask[img, z, y, x, 1] = 1
                else:
                    y, x = coord
                    new_mask[img, y, x, 1] = 1

            if data_mask.ndim == 5:
                for i in range(new_mask.shape[1]):
                    new_mask[img, i, ..., 1] = dilation(new_mask[img, i, ..., 1], disk(3))
            else:
                new_mask[img, ..., 1] = dilation(new_mask[img, ..., 1], disk(3))

        # Contour
        if ("C" in mode or "Dv2" in mode) and instance_count != 1:
            c_channel = 0 if mode == "C" else 1
            f = "thick" if fb_mode == "dense" else fb_mode
            new_mask[img, ..., c_channel] = find_boundaries(vol, mode=f).astype(np.uint8)
            if fb_mode == "dense" and mode != "BCM":
                if new_mask[img, ..., c_channel].ndim == 2:
                    new_mask[img, ..., c_channel] = 1 - binary_dilation(new_mask[img, ..., c_channel], disk(1))
                else:
                    for j in range(new_mask[img, ..., c_channel].shape[0]):
                        new_mask[img, j, ..., c_channel] = 1 - binary_dilation(
                            new_mask[img, j, ..., c_channel], disk(1)
                        )
                new_mask[img, ..., c_channel] = 1 - ((vol > 0) * new_mask[img, ..., c_channel])
            if "B" in mode:
                # Remove contours from segmentation maps
                new_mask[img, ..., 0][np.where(new_mask[img, ..., 1] == 1)] = 0
            if mode == "BCM":
                new_mask[img, ..., 2] = (vol > 0).astype(np.uint8)

        if ("D" in mode or "Dv2" in mode) and instance_count != 1:
            # Foreground distance
            new_mask[img, ..., -1] = scipy.ndimage.distance_transform_edt(new_mask[img, ..., 0])
            props = regionprops(vol, new_mask[img, ..., -1])
            max_values = np.zeros(vol.shape)
            for i in range(len(props)):
                max_values = np.where(vol == props[i].label, props[i].intensity_max, max_values)
            new_mask[img, ..., -1] = max_values - new_mask[img, ..., -1]

    # Normalize and merge distance channels
    if "Dv2" in mode:
        # Normalize background
        b_min = np.min(new_mask[..., 3])
        b_max = np.max(new_mask[..., 3])
        new_mask[..., 3] = (new_mask[..., 3] - b_min) / (b_max - b_min)

        if instance_count != 1:
            # Normalize foreground
            f_min = np.min(new_mask[..., 2])
            f_max = np.max(new_mask[..., 2])
            new_mask[..., 2] = (new_mask[..., 2] - f_min) / (f_max - f_min)

            new_mask[..., 2] = new_mask[..., 3] - new_mask[..., 2]

            # The intersection of the channels is the contour channel, so set it to the maximum value 1
            new_mask[..., 2][new_mask[..., 1] > 0] = 1

        if mode == "BCDv2":
            new_mask = new_mask[..., :3]
        elif mode == "BDv2":
            new_mask = new_mask[..., [0, -1]]
        elif mode == "Dv2":
            new_mask = np.expand_dims(new_mask[..., 2], -1)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        suffix = []
        if mode == "Dv2":
            suffix.append("_distance.tif")
        elif mode != "C":
            suffix.append("_semantic.tif")
        if mode in ["C", "BC", "BCM", "BCD", "BCDv2"]:
            suffix.append("_contour.tif")
            if mode in ["BCD", "BCDv2"]:
                suffix.append("_distance.tif")
            elif mode == "BCM":
                suffix.append("_binary_mask.tif")
        elif mode == "BP":
            suffix.append("_points.tif")
        elif mode in ["BDv2", "BD"]:
            suffix.append("_distance.tif")
        elif mode == "A":
            suffix.append("_affinity.tif")

        for i in range(min(3, len(new_mask))):
            for j in range(len(suffix)):
                aux = new_mask[i, ..., j]
                aux = np.expand_dims(np.expand_dims(aux, -1), 0)
                save_tif(aux, save_dir, filenames=["vol" + str(i) + suffix[j]], verbose=False)
                save_tif(
                    np.expand_dims(data_mask[i], 0),
                    save_dir,
                    filenames=["vol" + str(i) + "_y.tif"],
                    verbose=False,
                )
    return new_mask


#############
# DETECTION #
#############
def create_detection_masks(cfg, data_type="train"):
    """
    Create detection masks based on CSV files.

    Parameters
    ----------
    cfg : YACS CN object
        Configuration.

        data_type: str, optional
                Wheter to create train, validation or test masks.
    """

    assert data_type in ["train", "val", "test"]

    if data_type == "train":
        tag = "TRAIN"
    elif data_type == "val":
        tag = "VAL"
    else:
        tag = "TEST"
    img_dir = getattr(cfg.DATA, tag).PATH
    label_dir = getattr(cfg.DATA, tag).GT_PATH
    out_dir = getattr(cfg.DATA, tag).DETECTION_MASK_DIR
    img_ids = sorted(next(os.walk(img_dir))[2])
    working_with_chunked_data = False
    if len(img_ids) == 0:
        img_ids = sorted(next(os.walk(img_dir))[1])
        working_with_chunked_data = True
    if len(img_ids) == 0:
        raise ValueError(f"No data found in folder {img_dir}")
    img_ext = "." + img_ids[0].split(".")[-1]
    if working_with_chunked_data and ".zarr" != img_ext:
        raise ValueError(f"No data found in folder {img_dir}")
    ids = sorted(next(os.walk(label_dir))[2])

    classes = cfg.MODEL.N_CLASSES if cfg.MODEL.N_CLASSES > 2 else 1
    if len(img_ids) != len(ids):
        raise ValueError(
            "Different number of CSV files and images found ({} vs {}). "
            "Please check that every image has one and only one CSV file".format(len(ids), len(img_ids))
        )
    if cfg.PROBLEM.NDIM == "2D":
        req_columns = ["axis-0", "axis-1"] if classes == 1 else ["axis-0", "axis-1", "class"]
    else:
        req_columns = ["axis-0", "axis-1", "axis-2"] if classes == 1 else ["axis-0", "axis-1", "axis-2", "class"]
    req_dim = len(req_columns)

    print("Creating {} detection masks . . .".format(data_type))
    for i in range(len(ids)):
        img_filename = os.path.splitext(ids[i])[0] + img_ext
        if not os.path.exists(os.path.join(out_dir, img_filename)) and not os.path.exists(
            os.path.join(out_dir, img_ids[i])
        ):
            print("Attempting to create mask from CSV file: {}".format(os.path.join(label_dir, ids[i])))
            if not os.path.exists(os.path.join(img_dir, img_filename)):
                print(
                    "WARNING: The image seems to have different name than its CSV file. Using the CSV file that's "
                    "in the same spot (within the CSV files list) where the image is in its own list of images. Check if it is correct!"
                )
                img_filename = img_ids[i]
            print("Its respective image seems to be: {}".format(os.path.join(img_dir, img_filename)))

            df = pd.read_csv(os.path.join(label_dir, ids[i]))
            df = df.dropna()
            if ".zarr" != img_ext:
                img = read_img(
                    os.path.join(img_dir, img_filename),
                    is_3d=not cfg.PROBLEM.NDIM == "2D",
                )
                shape = img.shape[:-1]
            else:
                img_zarr_file, img = read_chunked_data(os.path.join(img_dir, img_filename))
                shape = img.shape if img.ndim == 3 else img.shape[:-1]

                if isinstance(img_zarr_file, h5py.File):
                    img_zarr_file.close()
                del img_zarr_file

            del img

            # Discard first index column to not have error if it is not sorted
            p_number = df.iloc[:, 0].to_list()
            df = df.rename(columns=lambda x: x.strip())  # trim spaces in column names
            cols_not_in_file = [x for x in req_columns if x not in df.columns]
            if len(cols_not_in_file) > 0:
                if len(cols_not_in_file) == 1:
                    m = f"'{cols_not_in_file[0]}' column is not present in CSV file: {os.path.join(label_dir, ids[i])}"
                else:
                    m = f"{cols_not_in_file} columns are not present in CSV file: {os.path.join(label_dir, ids[i])}"
                raise ValueError(m)

            # Convert them to int in case they are floats
            df["axis-0"] = df["axis-0"].astype("int")
            df["axis-1"] = df["axis-1"].astype("int")
            if cfg.PROBLEM.NDIM == "3D":
                df["axis-2"] = df["axis-2"].astype("int")

            df = df.sort_values(by=["axis-0"])

            # Obtain the points
            z_axis_point = df["axis-0"]
            y_axis_point = df["axis-1"]
            if cfg.PROBLEM.NDIM == "3D":
                x_axis_point = df["axis-2"]

            # Class column present
            if "class" in req_columns:
                df["class"] = df["class"].astype("int")
                class_point = np.array(df["class"])

                uniq = np.sort(np.unique(class_point))
                if uniq[0] != 1:
                    raise ValueError("Class number must start with 1")
                if not all(uniq == np.array(range(1, classes + 1))):
                    raise ValueError("Classes must be consecutive, e.g [1,2,3,4..]. Given {}".format(uniq))
            else:
                if classes > 1:
                    raise ValueError("MODEL.N_CLASSES > 1 but no class specified in CSV file")
                class_point = [1] * len(z_axis_point)

            # Create masks
            print("Creating all points . . .")
            mask = np.zeros((shape + (classes,)), dtype=np.uint8)
            for j in tqdm(
                range(len(z_axis_point)),
                disable=not is_main_process(),
                total=len(z_axis_point),
                leave=False,
            ):
                a0_coord = z_axis_point[j]
                a1_coord = y_axis_point[j]
                if cfg.PROBLEM.NDIM == "3D":
                    a2_coord = x_axis_point[j]
                c_point = class_point[j] - 1

                if c_point + 1 > mask.shape[-1]:
                    raise ValueError(
                        "Class {} detected while MODEL.N_CLASSES was set to {}. Please check it!".format(
                            c_point + 1, classes
                        )
                    )

                # Paint the point
                if cfg.PROBLEM.NDIM == "3D":
                    if a0_coord < mask.shape[0] and a1_coord < mask.shape[1] and a2_coord < mask.shape[2]:
                        cpd = cfg.PROBLEM.DETECTION.CENTRAL_POINT_DILATION
                        if (
                            1
                            in mask[
                                max(0, a0_coord - 1) : min(mask.shape[0], a0_coord + 2),
                                max(0, a1_coord - 1 - cpd) : min(mask.shape[1], a1_coord + 2 + cpd),
                                max(0, a2_coord - 1 - cpd) : min(mask.shape[2], a2_coord + 2 + cpd),
                                c_point,
                            ]
                        ):
                            print(
                                "WARNING: possible duplicated point in (3,9,9) neighborhood: coords {} , class {} "
                                "(point number {} in CSV)".format((a0_coord, a1_coord, a2_coord), c_point, p_number[j])
                            )

                        mask[a0_coord, a1_coord, a2_coord, c_point] = 1
                        if a1_coord + 1 < mask.shape[1]:
                            mask[a0_coord, a1_coord + 1, a2_coord, c_point] = 1
                        if a1_coord - 1 > 0:
                            mask[a0_coord, a1_coord - 1, a2_coord, c_point] = 1
                        if a2_coord + 1 < mask.shape[2]:
                            mask[a0_coord, a1_coord, a2_coord + 1, c_point] = 1
                        if a2_coord - 1 > 0:
                            mask[a0_coord, a1_coord, a2_coord - 1, c_point] = 1
                        if cfg.PROBLEM.DETECTION.CENTRAL_POINT_DILATION == 0:
                            if a1_coord + 1 < mask.shape[1] and a2_coord + 1 < mask.shape[2]:
                                mask[a0_coord, a1_coord + 1, a2_coord + 1, c_point] = 1
                            if a1_coord - 1 > 0 and a2_coord - 1 > 0:
                                mask[a0_coord, a1_coord - 1, a2_coord - 1, c_point] = 1
                            if a1_coord - 1 > 0 and a2_coord + 1 < mask.shape[2]:
                                mask[a0_coord, a1_coord - 1, a2_coord + 1, c_point] = 1
                            if a1_coord + 1 < mask.shape[1] and a2_coord - 1 > 0:
                                mask[a0_coord, a1_coord + 1, a2_coord - 1, c_point] = 1
                    else:
                        print(
                            "WARNING: discarding point {} which seems to be out of shape: {}".format(
                                [a0_coord, a1_coord, a2_coord], shape
                            )
                        )
                else:
                    if a0_coord < mask.shape[0] and a1_coord < mask.shape[1]:
                        if (
                            1
                            in mask[
                                max(0, a0_coord - 4) : min(mask.shape[0], a0_coord + 5),
                                max(0, a1_coord - 4) : min(mask.shape[1], a1_coord + 5),
                                c_point,
                            ]
                        ):
                            print(
                                "WARNING: possible duplicated point in (9,9) neighborhood: coords {} , class {} "
                                "(point number {} in CSV)".format((a0_coord, a1_coord), c_point, p_number[j])
                            )

                        mask[a0_coord, a1_coord, c_point] = 1
                        if a1_coord + 1 < mask.shape[1]:
                            mask[a0_coord, a1_coord + 1, c_point] = 1
                        if a1_coord - 1 > 0:
                            mask[a0_coord, a1_coord - 1, c_point] = 1
                    else:
                        print(
                            "WARNING: discarding point {} which seems to be out of shape: {}".format(
                                [a0_coord, a1_coord], shape
                            )
                        )

            # Dilate the mask
            if cfg.PROBLEM.DETECTION.CENTRAL_POINT_DILATION > 0:
                print("Dilating all points . . .")
                if cfg.PROBLEM.NDIM == "2D":
                    mask = np.expand_dims(mask, 0)
                for k in tqdm(
                    range(mask.shape[0]),
                    total=len(mask),
                    leave=False,
                    disable=not is_main_process(),
                ):
                    for ch in range(mask.shape[-1]):
                        mask[k, ..., ch] = binary_dilation_scipy(
                            mask[k, ..., ch],
                            iterations=1,
                            structure=disk(cfg.PROBLEM.DETECTION.CENTRAL_POINT_DILATION),
                        )
                if cfg.PROBLEM.NDIM == "2D":
                    mask = mask[0]

            if cfg.PROBLEM.DETECTION.CHECK_POINTS_CREATED:
                print("Check points created to see if some of them are very close that create a large label")
                error_found = False
                for ch in tqdm(
                    range(mask.shape[-1]),
                    total=len(mask),
                    leave=False,
                    disable=not is_main_process(),
                ):
                    _, index, counts = np.unique(
                        label(clear_border(mask[..., ch])),
                        return_counts=True,
                        return_index=True,
                    )
                    # 0 is background so valid element is 1. We will compare that value with the rest
                    ref_value = counts[1]
                    for k in range(2, len(counts)):
                        if abs(ref_value - counts[k]) > 5:
                            point = np.unravel_index(index[k], mask[..., ch].shape)
                            print(
                                "WARNING: There is a point (coords {}) with size very different from "
                                "the rest. Maybe that cell has several labels: please check it! Normally all point "
                                "have {} pixels but this one has {}.".format(point, ref_value, counts[k])
                            )
                            error_found = True

                if error_found:
                    raise ValueError(
                        "Duplicate points have been found so please check them before continuing. "
                        "If you consider that the points are valid simply disable "
                        "'PROBLEM.DETECTION.CHECK_POINTS_CREATED' so this check is not done again!"
                    )
            if working_with_chunked_data:
                write_chunked_data(
                    np.expand_dims(mask, 0),
                    out_dir,
                    img_filename,
                    dtype_str="uint8",
                    verbose=True,
                )
            else:
                save_tif(np.expand_dims(mask, 0), out_dir, [img_filename])
        else:
            if os.path.exists(os.path.join(out_dir, img_filename)):
                print(
                    "Mask file {} found for CSV file: {}".format(
                        os.path.join(out_dir, img_filename),
                        os.path.join(label_dir, ids[i]),
                    )
                )
            else:
                print(
                    "Mask file {} found for CSV file: {}".format(
                        os.path.join(out_dir, img_ids[i]),
                        os.path.join(label_dir, ids[i]),
                    )
                )


#######
# SSL #
#######
def create_ssl_source_data_masks(cfg, data_type="train"):
    """
    Create SSL source data.

    Parameters
    ----------
    cfg : YACS CN object
        Configuration.

        data_type: str, optional
                Wheter to create train, validation or test source data.
    """

    assert data_type in ["train", "val", "test"]
    tag = data_type.upper()

    img_dir = getattr(cfg.DATA, tag).PATH
    out_dir = getattr(cfg.DATA, tag).SSL_SOURCE_DIR
    ids = sorted(next(os.walk(img_dir))[2])
    add_noise = True if cfg.PROBLEM.SELF_SUPERVISED.NOISE > 0 else False

    print("Creating {} SSL source. . .".format(data_type))
    for i in range(len(ids)):
        if not os.path.exists(os.path.join(out_dir, ids[i])):
            print("Crappifying file {} to create SSL source".format(os.path.join(img_dir, ids[i])))
            img = read_img(os.path.join(img_dir, ids[i]), is_3d=not cfg.PROBLEM.NDIM == "2D")
            img = crappify(
                img,
                resizing_factor=cfg.PROBLEM.SELF_SUPERVISED.RESIZING_FACTOR,
                add_noise=add_noise,
                noise_level=cfg.PROBLEM.SELF_SUPERVISED.NOISE,
            )

            save_tif(np.expand_dims(img, 0), out_dir, [ids[i]])
        else:
            print("Source file {} found".format(os.path.join(img_dir, ids[i])))


def crappify(input_img, resizing_factor, add_noise=True, noise_level=None, Down_up=True):
    """
    Crappifies input image by adding Gaussian noise and downsampling and upsampling it so the resolution
    gets worsen.

    input_img : 4D/5D Numpy array
        Data to be modified. E.g. ``(y, x, channels)`` if working with 2D images or
        ``(z, y, x, channels)`` if working with 3D.

    resizing_factor : floats
        Downsizing factor to reshape the image.

    add_noise : boolean, optional
        Indicating whether to add gaussian noise before applying the resizing.

    noise_level: float, optional
        Number between ``[0,1]`` indicating the std of the Gaussian noise N(0,std).

    Down_up : bool, optional
        Indicating whether to perform a final upsampling operation to obtain an image of the
        same size as the original but with the corresponding loss of quality of downsizing and
        upsizing.

    Returns
    -------
    img : 4D/5D Numpy array
        Train images. E.g. ``(y, x, channels)`` if working with 2D images or
        ``(z, y, x, channels)`` if working with 3D.
    """
    if input_img.ndim == 3:
        w, h, c = input_img.shape
        org_sz = (w, h)
    else:
        d, w, h, c = input_img.shape
        org_sz = (d, w, h)
        new_d = int(d / np.sqrt(resizing_factor))

    new_w = int(w / np.sqrt(resizing_factor))
    new_h = int(h / np.sqrt(resizing_factor))

    if input_img.ndim == 3:
        targ_sz = (new_w, new_h)
    else:
        targ_sz = (new_d, new_w, new_h)

    img = input_img.copy()
    if add_noise:
        img = add_gaussian_noise(img, noise_level)

    img = resize(
        img,
        targ_sz,
        order=1,
        mode="reflect",
        clip=True,
        preserve_range=True,
        anti_aliasing=False,
    )

    if Down_up:
        img = resize(
            img,
            org_sz,
            order=1,
            mode="reflect",
            clip=True,
            preserve_range=True,
            anti_aliasing=False,
        )

    return img.astype(input_img.dtype)


def add_gaussian_noise(image, percentage_of_noise):
    """
    Adds Gaussian noise to an input image.

    Parameters
    ----------
    image : 3D Numpy array
        Image to be added Gaussian Noise with 0 mean and a certain std. E.g. ``(y, x, channels)``.

    percentage_of_noise : float
        percentage of the maximum value of the image that will be used as the std of the Gaussian Noise
        distribution.

    Returns
    -------
    out : 3D Numpy array
        Transformed image. E.g. ``(y, x, channels)``.
    """
    max_value = np.max(image)
    noise_level = percentage_of_noise * max_value
    noise = np.random.normal(loc=0, scale=noise_level, size=image.shape)
    noisy_img = np.clip(image + noise, 0, max_value).astype(image.dtype)
    return noisy_img


################
# SEMANTIC SEG #
################
def calculate_2D_volume_prob_map(Y, Y_path=None, w_foreground=0.94, w_background=0.06, save_dir=None):
    """
    Calculate the probability map of the given 2D data.

    Parameters
    ----------
    Y : 4D Numpy array
        Data to calculate the probability map from. E. g. ``(num_of_images, y, x, channel)``

    Y_path : str, optional
        Path to load the data from in case ``Y=None``.

    w_foreground : float, optional
        Weight of the foreground. This value plus ``w_background`` must be equal ``1``.

    w_background : float, optional
        Weight of the background. This value plus ``w_foreground`` must be equal ``1``.

    save_dir : str, optional
        Path to the file where the probability map will be stored.

    Raises
    ------
    ValueError
        if ``Y`` does not have 4 dimensions.

    ValueError
        if ``w_foreground + w_background > 1``.

    Returns
    -------
    Array : Str or 4D Numpy array
        Path where the probability map/s is/are stored if ``Y_path`` was given and there are images of different
        shapes. Otherwise, an array that represents the probability map of ``Y`` or all loaded data files from
        ``Y_path`` will be returned.
    """

    if Y is not None:
        if Y.ndim != 4:
            raise ValueError("'Y' must be a 4D Numpy array")

    if Y is None and Y_path is None:
        raise ValueError("'Y' or 'Y_path' need to be provided")

    if Y is not None:
        prob_map = np.copy(Y).astype(np.float32)
        l = prob_map.shape[0]
        channels = prob_map.shape[-1]
        v = np.max(prob_map)
    else:
        prob_map, _, _ = load_data_from_dir(Y_path)
        l = len(prob_map)
        channels = prob_map[0].shape[-1]
        v = np.max(prob_map[0])

    if isinstance(prob_map, list):
        first_shape = prob_map[0][0].shape
    else:
        first_shape = prob_map[0].shape

    print("Connstructing the probability map . . .")
    maps = []
    diff_shape = False
    for i in tqdm(range(l), disable=not is_main_process()):
        if isinstance(prob_map, list):
            _map = prob_map[i][0].copy().astype(np.float32)
        else:
            _map = prob_map[i].copy().astype(np.float32)

        for k in range(channels):
            # Remove artifacts connected to image border
            _map[:, :, k] = clear_border(_map[:, :, k])

            foreground_pixels = (_map[:, :, k] == v).sum()
            background_pixels = (_map[:, :, k] == 0).sum()

            if foreground_pixels == 0:
                _map[:, :, k][np.where(_map[:, :, k] == v)] = 0
            else:
                _map[:, :, k][np.where(_map[:, :, k] == v)] = w_foreground / foreground_pixels
            if background_pixels == 0:
                _map[:, :, k][np.where(_map[:, :, k] == 0)] = 0
            else:
                _map[:, :, k][np.where(_map[:, :, k] == 0)] = w_background / background_pixels

            # Necessary to get all probs sum 1
            s = _map[:, :, k].sum()
            if s == 0:
                t = 1
                for x in _map[:, :, k].shape:
                    t *= x
                _map[:, :, k].fill(1 / t)
            else:
                _map[:, :, k] = _map[:, :, k] / _map[:, :, k].sum()

        if first_shape != _map.shape:
            diff_shape = True
        maps.append(_map)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        if not diff_shape:
            for i in range(len(maps)):
                maps[i] = np.expand_dims(maps[i], 0)
            maps = np.concatenate(maps)
            print("Saving the probability map in {}".format(save_dir))
            np.save(os.path.join(save_dir, "prob_map.npy"), maps)
            return maps
        else:
            print(
                "As the files loaded have different shapes, the probability map for each one will be stored"
                " separately in {}".format(save_dir)
            )
            d = len(str(l))
            for i in range(l):
                f = os.path.join(save_dir, "prob_map" + str(i).zfill(d) + ".npy")
                np.save(f, maps[i])
            return save_dir


def calculate_3D_volume_prob_map(Y, Y_path=None, w_foreground=0.94, w_background=0.06, save_dir=None):
    """
    Calculate the probability map of the given 3D data.

    Parameters
    ----------
    Y : 5D Numpy array
        Data to calculate the probability map from. E. g. ``(num_subvolumes, z, y, x, channel)``

    Y_path : str, optional
        Path to load the data from in case ``Y=None``.

    w_foreground : float, optional
        Weight of the foreground. This value plus ``w_background`` must be equal ``1``.

    w_background : float, optional
        Weight of the background. This value plus ``w_foreground`` must be equal ``1``.

    save_dir : str, optional
        Path to the directory where the probability map will be stored.

    Returns
    -------
    Array : Str or 5D Numpy array
        Path where the probability map/s is/are stored if ``Y_path`` was given and there are images of different
        shapes. Otherwise, an array that represents the probability map of ``Y`` or all loaded data files from
        ``Y_path`` will be returned.

    Raises
    ------
    ValueError
        if ``Y`` does not have 5 dimensions.
    ValueError
        if ``w_foreground + w_background > 1``.
    """

    if Y is not None:
        if Y.ndim != 5:
            raise ValueError("'Y' must be a 5D Numpy array")

    if Y is None and Y_path is None:
        raise ValueError("'Y' or 'Y_path' need to be provided")

    if Y is not None:
        prob_map = np.copy(Y).astype(np.float32)
        l = prob_map.shape[0]
        channels = prob_map.shape[-1]
        v = np.max(prob_map)
    else:
        prob_map, _, _ = load_3d_images_from_dir(Y_path)
        l = len(prob_map)
        channels = prob_map[0].shape[-1]
        v = np.max(prob_map[0])

    if isinstance(prob_map, list):
        first_shape = prob_map[0][0].shape
    else:
        first_shape = prob_map[0].shape

    print("Constructing the probability map . . .")
    maps = []
    diff_shape = False
    for i in range(l):
        if isinstance(prob_map, list):
            _map = prob_map[i][0].copy().astype(np.float64)
        else:
            _map = prob_map[i].copy().astype(np.float64)

        for k in range(channels):
            for j in range(_map.shape[0]):
                # Remove artifacts connected to image border
                _map[j, :, :, k] = clear_border(_map[j, :, :, k])
            foreground_pixels = (_map[:, :, :, k] == v).sum()
            background_pixels = (_map[:, :, :, k] == 0).sum()

            if foreground_pixels == 0:
                _map[:, :, :, k][np.where(_map[:, :, :, k] == v)] = 0
            else:
                _map[:, :, :, k][np.where(_map[:, :, :, k] == v)] = w_foreground / foreground_pixels
            if background_pixels == 0:
                _map[:, :, :, k][np.where(_map[:, :, :, k] == 0)] = 0
            else:
                _map[:, :, :, k][np.where(_map[:, :, :, k] == 0)] = w_background / background_pixels

            # Necessary to get all probs sum 1
            s = _map[:, :, :, k].sum()
            if s == 0:
                t = 1
                for x in _map[:, :, :, k].shape:
                    t *= x
                _map[:, :, :, k].fill(1 / t)
            else:
                _map[:, :, :, k] = _map[:, :, :, k] / _map[:, :, :, k].sum()

        if first_shape != _map.shape:
            diff_shape = True
        maps.append(_map)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        if not diff_shape:
            for i in range(len(maps)):
                maps[i] = np.expand_dims(maps[i], 0)
            maps = np.concatenate(maps)
            print("Saving the probability map in {}".format(save_dir))
            np.save(os.path.join(save_dir, "prob_map.npy"), maps)
            return maps
        else:
            print(
                "As the files loaded have different shapes, the probability map for each one will be stored "
                "separately in {}".format(save_dir)
            )
            d = len(str(l))
            for i in range(l):
                f = os.path.join(save_dir, "prob_map" + str(i).zfill(d) + ".npy")
                np.save(f, maps[i])
            return save_dir


###########
# GENERAL #
###########
def norm_range01(x, dtype=np.float32, div_using_max_and_scale=False, div_using_max_and_scale_per_channel=False):
    norm_steps = {}
    norm_steps["orig_dtype"] = x.dtype

    if div_using_max_and_scale:
        norm_steps["min_val_scale"] = x.min()
        norm_steps["max_val_scale"] = x.max()

    if x.dtype in [np.uint8, torch.uint8]:
        if div_using_max_and_scale_per_channel:
            if not div_using_max_and_scale:
                x = x / 255
            else:
                x = x.astype(dtype)
                for c in range(x.shape[-1]):
                    x[...,c] = (x[...,c] - x[...,c].min()) / (x[...,c].max() - x[...,c].min() + sys.float_info.epsilon)
        else:
            x = x / 255 if not div_using_max_and_scale else (x - x.min()) / (x.max() - x.min() + sys.float_info.epsilon)
        norm_steps["div"] = 1
    else:
        if (isinstance(x, np.ndarray) and np.max(x) > 255) or (torch.is_tensor(x) and torch.max(x) > 255):
            norm_steps["reduced_{}".format(x.dtype)] = 1
            if div_using_max_and_scale_per_channel:
                x = x.astype(dtype)
                for c in range(x.shape[-1]):
                    x[...,c] = reduce_dtype(
                        x[...,c],
                        0 if not div_using_max_and_scale else x[...,c].min(),
                        65535 if not div_using_max_and_scale else x[...,c].max(),
                        out_min=0,
                        out_max=1,
                        out_type=dtype,
                    )
            else:
                x = reduce_dtype(
                    x,
                    0 if not div_using_max_and_scale else x.min(),
                    65535 if not div_using_max_and_scale else x.max(),
                    out_min=0,
                    out_max=1,
                    out_type=dtype,
                )
        elif (isinstance(x, np.ndarray) and np.max(x) > 2) or (torch.is_tensor(x) and torch.max(x) > 2):
            if div_using_max_and_scale_per_channel:
                if not div_using_max_and_scale:
                    x = x / 255
                else:
                    x = x.astype(dtype)
                    for c in range(x.shape[-1]):
                        x[...,c] = (x[...,c] - x[...,c].min()) / (x[...,c].max() - x[...,c].min() + sys.float_info.epsilon)
            else:
                x = x / 255 if not div_using_max_and_scale else (x - x.min()) / (x.max() - x.min() + sys.float_info.epsilon)                
            norm_steps["div"] = 1

    if torch.is_tensor(x):
        x = x.to(dtype)
    else:
        x = x.astype(dtype)
    return x, norm_steps


def undo_norm_range01(x, xnorm, min_val_scale=None, max_val_scale=None):
    if min_val_scale is not None and max_val_scale is None:
        raise ValueError("max_val_scale can not be None when min_val_scale is provided")
    if max_val_scale is not None and min_val_scale is None:
        raise ValueError("min_val_scale can not be None when max_val_scale is provided")

    if "div" == xnorm["type"]:
        # Prevent values go outside expected range
        if isinstance(x, np.ndarray):
            x = np.clip(x, 0, 1)
        else:
            x = torch.clamp(x, 0, 1)
        if "div" in xnorm:
            x = (x * 255) if max_val_scale is None else (x * max_val_scale) + min_val_scale
            if isinstance(x, np.ndarray):
                x = x.astype(np.uint8)
            else:
                x = x.to(torch.uint8)
        else:
            reductions = [key for key, value in xnorm.items() if "reduced" in key.lower()]
            if len(reductions) > 0:
                reductions = reductions[0]
                reductions = reductions.replace("reduced_", "")
                x = (x * 65535) if max_val_scale is None else (x * max_val_scale) + min_val_scale
                if isinstance(x, np.ndarray):
                    x = x.astype(eval("np.{}".format(reductions)))
                else:
                    x = x.to(eval("torch.{}".format(reductions)))
    return x


def reduce_dtype(x, x_min, x_max, out_min=0, out_max=1, out_type=np.float32):
    if isinstance(x, np.ndarray):
        return ((np.array((x - x_min) / (x_max - x_min)) * (out_max - out_min)) + out_min).astype(out_type)
    else:  # Tensor considered
        return ((((x - x_min) / (x_max - x_min)) * (out_max - out_min)) + out_min).to(out_type)


def normalize(data, means, stds, out_type="float32"):
    numpy_torch_dtype_dict = {
        "bool": [torch.bool, bool],
        "uint8": [torch.uint8, np.uint8],
        "int8": [torch.int8, np.int8],
        "int16": [torch.int16, np.int16],
        "int32": [torch.int32, np.int32],
        "int64": [torch.int64, np.int64],
        "float16": [torch.float16, np.float16],
        "float32": [torch.float32, np.float32],
        "float64": [torch.float64, np.float64],
        "complex64": [torch.complex64, np.complex64],
        "complex128": [torch.complex128, np.complex128],
    }
    if torch.is_tensor(data):
        if stds == 0:
            return data.to(numpy_torch_dtype_dict[out_type][0])
        else:
            return ((data - means) / stds).to(numpy_torch_dtype_dict[out_type][0])
    else:
        if stds == 0:
            return data.astype(numpy_torch_dtype_dict[out_type][1])
        else:
            return ((data - means) / stds).astype(numpy_torch_dtype_dict[out_type][1])


def denormalize(data, means, stds, out_type="float32"):
    numpy_torch_dtype_dict = {
        "bool": [torch.bool, bool],
        "uint8": [torch.uint8, np.uint8],
        "int8": [torch.int8, np.int8],
        "int16": [torch.int16, np.int16],
        "int32": [torch.int32, np.int32],
        "int64": [torch.int64, np.int64],
        "float16": [torch.float16, np.float16],
        "float32": [torch.float32, np.float32],
        "float64": [torch.float64, np.float64],
        "complex64": [torch.complex64, np.complex64],
        "complex128": [torch.complex128, np.complex128],
    }
    if torch.is_tensor(data):
        return ((data * stds) + means).to(numpy_torch_dtype_dict[out_type][0])
    else:
        return ((data * stds) + means).astype(numpy_torch_dtype_dict[out_type][1])


def percentile_clip(x, lower=0.1, upper=99.9, lwr_perc_val=None, uppr_perc_val=None):
    x_lwr = float(np.percentile(x, lower)) if lwr_perc_val is None else lwr_perc_val
    x_upr = float(np.percentile(x, upper)) if uppr_perc_val is None else uppr_perc_val
    if "float" not in str(x.dtype):
        x_lwr = int(x_lwr)
        x_upr = int(x_upr)
    return np.clip(x, x_lwr, x_upr, out=x), x_lwr, x_upr


def resize_images(images, **kwards):
    """
    The function resizes all the images using the specified parameters or default values if not provided.

    Parameters
    ----------
    images: Numpy array or list of numpy arrays
        The `images` parameter is the list of all input images that you want to resize.

    output_shape: iterable
        Size of the generated output image. E.g. `(256,256)`

    (kwards): optional
        `skimage.transform.resize() <https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.resize>`__
        parameters are also allowed.

    Returns
    -------
    resized_images: Numpy array or list of numpy arrays
        The resized images. The returned data will use the same data type as the given `images`.

    """

    resized_images = [resize(img, **kwards).astype(img.dtype) for img in images]
    if isinstance(images, np.ndarray):
        resized_images = np.array(resized_images, dtype=images.dtype)
    return resized_images


def apply_gaussian_blur(images, **kwards):
    """
    The function applies a Gaussian blur to all images.

    Parameters
    ----------
    images: Numpy array or list of numpy arrays
        The input images on which the Gaussian blur will be applied.

    (kwards): optional
        `skimage.filters.gaussian() <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.gaussian>`__
        parameters are also allowed.

    Returns
    -------
    blurred_images: Numpy array or list of numpy arrays
        A Gaussian blurred images. The returned data will use the same data type as the given `images`.

    """

    def _process(image, **kwards):
        im = gaussian(image, **kwards)  # returns 0-1 range
        if np.issubdtype(image.dtype, np.integer):
            im = im * np.iinfo(image.dtype).max
        im = im.astype(image.dtype)
        return im

    blurred_images = [_process(img, **kwards) for img in images]
    if isinstance(images, np.ndarray):
        blurred_images = np.array(blurred_images, dtype=images.dtype)
    return blurred_images


def apply_median_blur(images, **kwards):
    """
    The function applies a median blur filter to all images.

    Parameters
    ----------
    image: Numpy array or list of numpy arrays
        The input image on which the median blur operation will be applied.

    (kwards): optional
        `skimage.filters.median() <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.median>`__
        parameters are also allowed.

    Returns
    -------
    blurred_images: Numpy array or list of numpy arrays
        The median-blurred images. The returned data will use the same data type as the given `images`.

    """

    blurred_images = [median(img, **kwards).astype(img.dtype) for img in images]
    if isinstance(images, np.ndarray):
        blurred_images = np.array(blurred_images, dtype=images.dtype)
    return blurred_images


def detect_edges(images, **kwards):
    """
    The function `detect_edges` takes the 2D images as input, converts it to grayscale if necessary, and
    applies the Canny edge detection algorithm to detect edges in the image.

    Parameters
    ----------
    images: Numpy array or list of numpy arrays
        The list of all input images on which the edge detection will be performed. It can be either a color image with
        shape (height, width, 3) or a grayscale image with shape (height, width, 1).

    (kwards): optional
        `skimage.feature.canny() <https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.canny>`__
        parameters are also allowed.

    Returns
    -------
    edges: Numpy array or list of numpy arrays
        The edges of the input images. The returned numpy arrays will be uint8, where background is black (0) and edges white (255).
        The returned data will use the same structure as the given `images` (list[numpy array] or numpy array).

    """

    def to_gray(image):
        c = image.shape[-1]
        if c == 3:
            image = rgb2gray(image)
        elif c == 1:
            image = image[..., 0]
        else:
            raise ValueError(
                f"Detect edges function does not allow given ammount of channels ({c} channels). "
                "Only accepts grayscale and RGB 2D images (1 or 3 channels)."
            )
        return image

    def set_uint8(image):
        im = image.astype(np.uint8)
        im = im[..., np.newaxis]  # add channel dim
        im = im * 255
        return im

    edges = [set_uint8(canny(to_gray(img), **kwards)) for img in images]
    if isinstance(images, np.ndarray):
        edges = np.array(edges, dtype=np.uint8)
    return edges


def _histogram_matching(source_imgs, target_imgs):
    """
    Given a set of target images, it will obtain their mean histogram
    and applies histogram matching to all images from sorce images.

    Parameters
    ----------
    source_imgs: Numpy array or list of numpy arrays
        The images of the source domain, to which the histogram matching is to be applied.

    target_imgs: Numpy array
        The target domain images, from which mean histogram will be obtained.

    Returns
    -------
    matched_images : list of numpy arrays
        A set of source images with target's histogram
    """

    hist_mean, _ = np.histogram(target_imgs.ravel(), bins=np.arange(np.iinfo(target_imgs.dtype).max + 1))
    hist_mean = hist_mean / target_imgs.shape[0]  # number of images

    # calculate normalized quantiles
    tmpl_size = np.sum(hist_mean)
    tmpl_quantiles = np.cumsum(hist_mean) / tmpl_size

    del target_imgs

    # based on scikit implementation.
    # source: https://github.com/scikit-image/scikit-image/blob/v0.18.0/skimage/exposure/histogram_matching.py#L22-L70
    def _match_cumulative_cdf(source, tmpl_quantiles):
        src_values, src_unique_indices, src_counts = np.unique(source.ravel(), return_inverse=True, return_counts=True)

        # calculate normalized quantiles
        src_size = source.size  # number of pixels
        src_quantiles = np.cumsum(src_counts) / src_size  # normalize
        interp_a_values = np.interp(src_quantiles, tmpl_quantiles, np.arange(len(tmpl_quantiles)))

        return interp_a_values[src_unique_indices].reshape(source.shape)

    # apply histogram matching
    results = [_match_cumulative_cdf(image, tmpl_quantiles).astype(image.dtype) for image in source_imgs]
    return results


def apply_histogram_matching(images, reference_path, is_2d):
    """
    The function returns the images with their histogram matched to the histogram of the reference images,
    loaded from the given reference_path.

    Parameters
    ----------
    images: Numpy array or list of numpy arrays
        The list of input images whose histogram needs to be matched to the reference histogram. It should be a
        numpy array representing the image.

    reference_path: str
        The reference_path is the directory path to the reference images. From reference images, we will extract
        the reference histogram with which we want to match the histogram of the images. It represents
        the desired distribution of pixel intensities in the output image.

    is_2d: bool, optional
        The value indicate if the data given in ``reference_path`` is 2D (``is_2d = True``) or 3D (``is_2d = False``).
        Defaults to True.

    Returns
    -------
    matched_images : Numpy array or list of numpy arrays
        The result of matching the histogram of the input images to the histogram of the reference image.
        The returned data will use the same data type as the given `images`.
    """
    f_name = load_data_from_dir if is_2d else load_3d_images_from_dir
    references, *_ = f_name(reference_path)

    matched_images = _histogram_matching(images, references)
    if isinstance(images, np.ndarray):
        matched_images = np.array(matched_images, dtype=images.dtype)
    return matched_images


def apply_clahe(images, **kwards):
    """
    The function applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to an image and
    returns the result.

    Parameters
    ----------
    images: Numpy array or list of numpy arrays
        The list of input images that you want to apply the CLAHE (Contrast Limited Adaptive Histogram Equalization)
        algorithm to.

    (kwards): optional
        `skimage.exposure.equalize_adapthist() <https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.equalize_adapthist>`__
        parameters are also allowed.

    Returns
    -------
    processed_images: Numpy array or list of numpy arrays
        The images after applying the Contrast Limited Adaptive Histogram Equalization (CLAHE) algorithm.
        The returned data will use the same data type as the given `images`.
    """

    def _process(image, **kwards):
        im = equalize_adapthist(image, **kwards)  # returns 0-1 range
        if np.issubdtype(image.dtype, np.integer):
            im = im * np.iinfo(image.dtype).max
        im = im.astype(image.dtype)
        return im

    processed_images = [_process(img, **kwards) for img in images]
    if isinstance(images, np.ndarray):
        processed_images = np.array(processed_images, dtype=images.dtype)
    return processed_images


def preprocess_data(cfg, x_data=[], y_data=[], is_2d=True, is_y_mask=False):
    """
    The function preprocesses data by applying various image processing techniques.

    Parameters
    ----------
    cfg: dict
        The `cfg` parameter is a configuration object that contains various settings for
        preprocessing the data. It is used to control the behavior of different preprocessing techniques
        such as image resizing, blurring, histogram matching, etc.

    x_data: 4D/5D numpy array or list of 3D/4D numpy arrays, optional
        The input data (images) to be preprocessed. The first dimension must be the number of images.
        E.g. ``(num_of_images, y, x, channels)`` or ``(num_of_images, z, y, x, channels)``.
        In case of using a list, the format of the images remains the same. Each item in the list
        corresponds to a different image.

    y_data: 4D/5D numpy array or list of 3D/4D numpy arrays, optional
        The target data that corresponds to the x_data. The first dimension must be the number of images.
        E.g. ``(num_of_images, y, x, channels)`` or ``(num_of_images, z, y, x, channels)``.
        In case of using a list, the format of the images remains the same. Each item in the list
        corresponds to a different image.

    is_2d: bool, optional
        A boolean flag indicating whether the reference data for histogram matching is 2D or not. Defaults to True.

    is_y_mask: bool, optional
        is_y_mask is a boolean parameter that indicates whether the y_data is a mask or not. If
        it is set to True, the resize operation for y_data will use the nearest neighbor interpolation
        method (order=0), otherwise it will use the interpolation method specified in the cfg.RESIZE.ORDER
        parameter. Defaults to False.

    Returns
    -------
    x_data: 4D/5D numpy array or list of 3D/4D numpy arrays, optional
        Preprocessed data. The same structure and dimensionality of the given data will be returned.

    y_data: 4D/5D numpy array or list of 3D/4D numpy arrays, optional
        Preprocessed data. The same structure and dimensionality of the given data will be returned.
    """

    if cfg.RESIZE.ENABLE:
        print("Preprocessing: applying resize . . .")
        if len(x_data) > 0:
            x_data = resize_images(
                x_data,
                output_shape=cfg.RESIZE.OUTPUT_SHAPE,
                order=cfg.RESIZE.ORDER,
                mode=cfg.RESIZE.MODE,
                cval=cfg.RESIZE.CVAL,
                clip=cfg.RESIZE.CLIP,
                preserve_range=cfg.RESIZE.PRESERVE_RANGE,
                anti_aliasing=cfg.RESIZE.ANTI_ALIASING,
            )
        if len(y_data) > 0:
            # if y is a mask, then use nearest
            y_order = 0 if is_y_mask else cfg.RESIZE.ORDER
            y_data = resize_images(
                y_data,
                output_shape=cfg.RESIZE.OUTPUT_SHAPE,
                order=y_order,
                mode=cfg.RESIZE.MODE,
                cval=cfg.RESIZE.CVAL,
                clip=cfg.RESIZE.CLIP,
                preserve_range=cfg.RESIZE.PRESERVE_RANGE,
                anti_aliasing=cfg.RESIZE.ANTI_ALIASING,
            )

    if len(x_data) > 0:
        if cfg.GAUSSIAN_BLUR.ENABLE:
            print("Preprocessing: applying gaussian blur . . .")
            x_data = apply_gaussian_blur(
                x_data,
                sigma=cfg.GAUSSIAN_BLUR.SIGMA,
                mode=cfg.GAUSSIAN_BLUR.MODE,
                channel_axis=cfg.GAUSSIAN_BLUR.CHANNEL_AXIS,
            )
        if cfg.MEDIAN_BLUR.ENABLE:
            print("Preprocessing: applying median blur . . .")
            x_data = apply_median_blur(
                x_data,
                footprint=cfg.MEDIAN_BLUR.FOOTPRINT,
            )
        if cfg.MATCH_HISTOGRAM.ENABLE:
            print("Preprocessing: applying histogram matching . . .")
            x_data = apply_histogram_matching(
                x_data,
                reference_path=cfg.MATCH_HISTOGRAM.REFERENCE_PATH,
                is_2d=is_2d,
            )
        if cfg.CLAHE.ENABLE:
            print("Preprocessing: applying CLAHE . . .")
            x_data = apply_clahe(
                x_data,
                kernel_size=cfg.CLAHE.KERNEL_SIZE,
                clip_limit=cfg.CLAHE.CLIP_LIMIT,
            )
        if cfg.CANNY.ENABLE:
            print("Preprocessing: applying Canny . . .")
            x_data = detect_edges(
                x_data,
                low_threshold=cfg.CANNY.LOW_THRESHOLD,
                high_threshold=cfg.CANNY.HIGH_THRESHOLD,
            )

    if len(x_data) > 0 and len(y_data) > 0:
        return x_data, y_data
    if len(y_data) > 0:
        return y_data
    else:
        return x_data
