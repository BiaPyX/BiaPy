import os
import torch
import scipy
import h5py
import zarr
import numpy as np
from tqdm import tqdm
import pandas as pd
from skimage.segmentation import clear_border, find_boundaries, watershed
from scipy.ndimage import distance_transform_edt, center_of_mass
from scipy.ndimage import binary_dilation as binary_dilation_scipy
from skimage.morphology import disk, dilation, binary_dilation
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import resize
from skimage.feature import canny
from skimage.exposure import equalize_adapthist
from skimage.color import rgb2gray
from skimage.filters import gaussian, median
from numpy.typing import NDArray
from typing import (
    List,
    Optional,
    Dict,
    Tuple
)

from biapy.data.dataset import BiaPyDataset
from biapy.utils.util import (
    seg2aff_pni,
    seg_widen_border,
)
from biapy.utils.misc import is_main_process
from biapy.data.data_3D_manipulation import (
    load_3D_efficient_files,
    load_img_part_from_efficient_file,
    order_dimensions,
    read_chunked_data,
    read_chunked_nested_data,
    write_chunked_data,
)
from biapy.data.data_manipulation import (
    read_img_as_ndarray,
    load_data_from_dir,
    save_tif,
)
from biapy.config.config import Config

#########################
# INSTANCE SEGMENTATION #
#########################
def create_instance_channels(
    cfg: Config, 
    data_type: str="train"
):
    """
    Create training and validation new data with appropiate channels based on ``PROBLEM.INSTANCE_SEG.DATA_CHANNELS``
    for instance segmentation.

    Parameters
    ----------
    cfg : YACS CN object
        Configuration.

    data_type: str, optional
        Wheter to create training or validation instance channels.
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
        or (len(h5_files) > 0 and any(h5_files[0].endswith(x) for x in [".h5", ".hdf5", ".hdf"]))
    ):
        working_with_zarr_h5_files = True
        # Check if the raw images and labels are within the same file
        data_path = getattr(cfg.DATA, tag).GT_PATH
        path_to_gt_data = None
        if getattr(cfg.DATA, tag).INPUT_ZARR_MULTIPLE_DATA:
            data_path = getattr(cfg.DATA, tag).PATH
            if cfg.PROBLEM.INSTANCE_SEG.TYPE == "synapses":
                path_to_gt_data = getattr(cfg.DATA, tag).INPUT_ZARR_MULTIPLE_DATA_RAW_PATH
            else:
                path_to_gt_data = getattr(cfg.DATA, tag).INPUT_ZARR_MULTIPLE_DATA_GT_PATH

        if len(zarr_files) > 0 and ".zarr" in zarr_files[0]:
            print("Working with Zarr files . . .")
            img_files = [os.path.join(data_path, x) for x in zarr_files]
        elif len(h5_files) > 0 and any(h5_files[0].endswith(x) for x in [".h5", ".hdf5", ".hdf"]):
            print("Working with H5 files . . .")
            img_files = [os.path.join(data_path, x) for x in h5_files]

        Y, Y_total_patches = load_3D_efficient_files(
            img_files,
            input_axes=getattr(cfg.DATA, tag).INPUT_IMG_AXES_ORDER,
            crop_shape=cfg.DATA.PATCH_SIZE,
            overlap=getattr(cfg.DATA, tag).OVERLAP,
            padding=getattr(cfg.DATA, tag).PADDING,
            data_within_zarr_path=path_to_gt_data,
        )
        zarr_data_information = {
            "axis_order": getattr(cfg.DATA, tag).INPUT_IMG_AXES_ORDER,
            "z_axe_pos": getattr(cfg.DATA, tag).INPUT_IMG_AXES_ORDER.index("Z"),
            "y_axe_pos": getattr(cfg.DATA, tag).INPUT_IMG_AXES_ORDER.index("Y"),
            "x_axe_pos": getattr(cfg.DATA, tag).INPUT_IMG_AXES_ORDER.index("X"),
            "id_path": cfg.DATA.TRAIN.INPUT_ZARR_MULTIPLE_DATA_ID_PATH,
            "partners_path": cfg.DATA.TRAIN.INPUT_ZARR_MULTIPLE_DATA_PARTNERS_PATH,
            "locations_path": cfg.DATA.TRAIN.INPUT_ZARR_MULTIPLE_DATA_LOCATIONS_PATH,
            "resolution_path": cfg.DATA.TRAIN.INPUT_ZARR_MULTIPLE_DATA_RESOLUTION_PATH,
        }

    else:
        if cfg.PROBLEM.INSTANCE_SEG.TYPE == "synapses":
            raise ValueError("Synapse detection is only available for 3D Zarr/H5 data")
        Y = sorted(next(os.walk(getattr(cfg.DATA, tag).GT_PATH))[2])
    del zarr_files, h5_files

    print("Creating Y_{} channels . . .".format(data_type))
    # Create the mask patch by patch (Zarr/H5)
    if working_with_zarr_h5_files and isinstance(Y, dict):
        savepath = str(data_path) + "_" + cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS
        if cfg.PROBLEM.INSTANCE_SEG.TYPE == "regular":
            if "C" in cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS:
                savepath += "_" + cfg.PROBLEM.INSTANCE_SEG.DATA_CONTOUR_MODE
        else:
            post_dil = "".join(str(cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.POSTSITE_DILATION)[1:-1].replace(",", "")).replace(
                " ", "_"
            )
            post_d_dil = "".join(
                str(cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.POSTSITE_DILATION_DISTANCE_CHANNELS)[1:-1].replace(",", "")
            ).replace(" ", "_")
            savepath += "_" + post_dil + "_" + post_d_dil

        if "D" in cfg.PROBLEM.INSTANCE_SEG.DATA_CONTOUR_MODE:
            dtype_str = "float32"
            raise ValueError("Currently distance creation using Zarr by chunks is not implemented.")
        else:
            dtype_str = "uint8"

        # For synapses the process of the channel creation is different: not patch by patch but paiting each post-synaptic
        # points for each pre-synaptic point
        if cfg.PROBLEM.INSTANCE_SEG.TYPE == "synapses" and len(Y) > 0:
            synapse_channel_creation(
                data_info=Y,
                zarr_data_information=zarr_data_information,
                savepath=savepath,
                mode=cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS,
                postsite_dilation=cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.POSTSITE_DILATION,
                postsite_distance_channel_dilation=cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.POSTSITE_DILATION_DISTANCE_CHANNELS,
                normalize_values=cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.NORMALIZE_DISTANCES,
            )
        else:  # regular instances, not synapses
            mask = None
            imgfile = None
            last_parallel_file = None
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
                if mask is None or os.path.basename(Y[i]["filepath"]) != last_parallel_file:
                    last_parallel_file = os.path.basename(Y[i]["filepath"])

                    # Close last open H5 file
                    if mask and isinstance(fid_mask, h5py.File):
                        fid_mask.close()

                    if path_to_gt_data:
                        imgfile, data = read_chunked_nested_data(Y[i]["filepath"], path_to_gt_data)
                    else:
                        imgfile, data = read_chunked_data(Y[i]["filepath"])
                    fname = os.path.join(
                        savepath, os.path.basename(Y[i]["filepath"])
                    )
                    os.makedirs(savepath, exist_ok=True)
                    if any(fname.endswith(x) for x in [".h5", ".hdf5", ".hdf"]):
                        fid_mask = h5py.File(fname, "w")
                    else:  # Zarr file
                        fid_mask = zarr.open_group(fname, mode="w")

                    # Determine data shape
                    out_data_shape = np.array(data.shape)
                    if "C" not in getattr(cfg.DATA, tag).INPUT_IMG_AXES_ORDER:
                        out_data_shape = tuple(out_data_shape) + (img.shape[-1],)
                        out_data_order = getattr(cfg.DATA, tag).INPUT_IMG_AXES_ORDER + "C"
                        channel_pos = -1
                    else:
                        out_data_shape[getattr(cfg.DATA, tag).INPUT_IMG_AXES_ORDER.index("C")] = img.shape[-1]
                        out_data_shape = tuple(out_data_shape)
                        out_data_order = getattr(cfg.DATA, tag).INPUT_IMG_AXES_ORDER
                        channel_pos = getattr(cfg.DATA, tag).INPUT_IMG_AXES_ORDER.index("C")

                    if any(fname.endswith(x) for x in [".h5", ".hdf5", ".hdf"]):
                        mask = fid_mask.create_dataset("data", shape=out_data_shape, dtype=dtype_str)
                        # mask = fid_mask.create_dataset("data", out_data_shape, compression="lzf", dtype=dtype_str)
                    else:
                        mask = fid_mask.create_dataset("data", shape=out_data_shape, dtype=dtype_str)

                    # Close H5 file read for the data shape
                    if isinstance(imgfile, h5py.File):
                        imgfile.close()

                    del data, fname

                slices = (
                    slice(patch_coords.z_start, patch_coords.z_end),
                    slice(patch_coords.y_start, patch_coords.y_end),
                    slice(patch_coords.x_start, patch_coords.x_end),
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
                transpose_order = [x for x in np.array(transpose_order) if not np.isnan(x)]

                # Place the patch into the Zarr
                mask[data_ordered_slices] = img.transpose(transpose_order)

            # Close last open H5 file
            if mask and isinstance(imgfile, h5py.File):
                imgfile.close()
    else:
        for i in tqdm(range(len(Y)), disable=not is_main_process()):
            img = read_img_as_ndarray(
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


def create_test_instance_channels(
    cfg: Config
):
    """
    Create test new data with appropiate channels based on ``PROBLEM.INSTANCE_SEG.DATA_CHANNELS`` for instance segmentation.

    Parameters
    ----------
    cfg : YACS CN object
        Configuration.
    """
    path = cfg.DATA.TEST.PATH if cfg.DATA.TEST.INPUT_ZARR_MULTIPLE_DATA else cfg.DATA.TEST.GT_PATH
    Y = sorted(next(os.walk(path))[2])
    print("Creating Y_test channels . . .")
    for i in tqdm(range(len(Y)), disable=not is_main_process()):
        img = read_img_as_ndarray(
            os.path.join(path, Y[i]),
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


def labels_into_channels(
    data_mask: NDArray, 
    mode: str="BC", 
    fb_mode: str="outer", 
    save_dir: Optional[str]=None
) -> NDArray:
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

    if save_dir:
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


def synapse_channel_creation(
    data_info: Dict,
    zarr_data_information: Dict,
    savepath: str,
    mode: str="BF",
    postsite_dilation: List[int]=[2, 4, 4],
    postsite_distance_channel_dilation: List[int]=[3, 10, 10],
    normalize_values: bool=False,
):
    """
    Creates different channels that represent a synapse segmentation problem to train an instance segmentation
    problem. This function is only prepared to read an H5/Zarr file that follows
    `CREMI data format <https://cremi.org/data/>`__.

    Parameters
    ----------
    data_info : dict
        All patches that can be extracted from all the Zarr/H5 samples in ``data_path``.
        Keys created are:
            * ``"filepath"``: path to the file where the patch was extracted.
            * ``"full_shape"``: shape of the data within the file where the patch was extracted.
            * ``"patch_coords"``: coordinates of the data that represents the patch.

    zarr_data_information : dict
        Information when using Zarr/H5 files. Assumes that the H5/Zarr files contain the information according
        `CREMI data format <https://cremi.org/data/>`__. The following keys are expected:
            * ``"z_axe_pos"``: position of z axis of the data within the file.
            * ``"y_axe_pos"``: position of y axis of the data within the file.
            * ``"x_axe_pos"``: position of x axis of the data within the file.
            * ``"id_path"``: path within the file where the ``ids`` are stored.
              Reference in CREMI: ``annotations/ids``
            * ``"partners_path"``: path within the file where ``partners`` is stored.
              Reference in CREMI: ``annotations/partners``
            * ``"locations_path"``: path within the file where ``locations`` is stored.
              Reference in CREMI: ``annotations/locations``
            * ``"resolution_path"``: path within the file where ``resolution`` is stored.
              Reference in CREMI: ``["volumes/raw"].attrs["offset"]``

    savepath : str
        Path to save the data created.

    mode : str, optional
        Operation mode. Possible values: ``BF``.
         - 'B' stands for 'Binary segmentation'
         - 'F' stands for 'Flows'

    postsite_dilation : tuple of ints, optional
        Dilation to be used in the postsynapse sites ('B' channel).

    postsite_distance_channel_dilation : tuple of ints, optional
        Dilation to be used in the postsynapse sites when creating the distance channels ('F' channels).

    normalize_values : bool, optional
        Whether to normalize distance values or not.

    Returns
    -------
    new_mask : 5D Numpy array
        5D array with 3 channels instead of one. E.g. ``(10, 200, 1000, 1000, 3)``

    patch_offset : list of list
        Pixels used on each axis to pad the patch in order to not cut some of the values in the edges.
    """
    # TODO: seguir con esta funcion
    # assert len(data_shape) in [5, 4]
    # d_shape = data_shape[1:-1]
    # dim = len(d_shape)
    assert mode in ["BF"]
    if mode == "BF":
        channels = 4

    dtype_str = "float32"
    unique_files = []
    unique_shapes = []
    for i in range(len(data_info)):
        if data_info[i]["filepath"] not in unique_files:
            unique_files.append(data_info[i]["filepath"])
            unique_shapes.append(data_info[i]["full_shape"])

    dilation_width = [
        max(postsite_dilation[0], postsite_distance_channel_dilation[0]) + 1,
        max(postsite_dilation[1], postsite_distance_channel_dilation[1]) + 1,
        max(postsite_dilation[2], postsite_distance_channel_dilation[2]) + 1,
    ]
    channels = 4
    ellipse_footprint_cpd = generate_ellipse_footprint(postsite_dilation)
    ellipse_footprint_cpd2 = generate_ellipse_footprint(postsite_distance_channel_dilation)

    print("Collecting all pre/post-synaptic points")
    for filename, data_shape in tqdm(zip(unique_files, unique_shapes), disable=not is_main_process()):
        # Take all the information within the dataset
        files = []
        file, ids = read_chunked_nested_data(filename, zarr_data_information["id_path"])
        ids = list(np.array(ids))
        files.append(file)
        # file, types = read_chunked_nested_data(filename, cfg.DATA.TRAIN.INPUT_ZARR_MULTIPLE_DATA_TYPES_PATH)
        # files.append(file)
        file, partners = read_chunked_nested_data(filename, zarr_data_information["partners_path"])
        partners = np.array(partners)
        files.append(file)
        file, locations = read_chunked_nested_data(filename, zarr_data_information["locations_path"])
        locations = np.array(locations)
        files.append(file)
        file, resolution = read_chunked_nested_data(filename, zarr_data_information["resolution_path"])
        files.append(file)
        try:
            resolution = resolution.attrs["resolution"]
        except:
            raise ValueError(
                "There is no 'resolution' attribute in '{}'. Add it like: data['{}'].attrs['resolution'] = (8,8,8)".format(
                    zarr_data_information["resolution_path"], zarr_data_information["resolution_path"]
                )
            )

        # Close files
        for f in files:
            if isinstance(f, h5py.File):
                f.close()
        del files

        # Link to each patch sample its corresponding pre/post synaptic sites
        pre_post_points = {}
        for i in tqdm(range(len(partners)), disable=not is_main_process()):
            pre_id, post_id = partners[i]
            pre_position = ids.index(pre_id)
            post_position = ids.index(post_id)
            pre_loc = locations[pre_position] // resolution
            post_loc = locations[post_position] // resolution

            insert_pre = False
            insert_post = False
            # Pre point in data shape
            if not (
                (pre_loc[0] < 0 or pre_loc[0] >= data_shape[0])
                or (pre_loc[1] < 0 or pre_loc[1] >= data_shape[1])
                or (pre_loc[2] < 0 or pre_loc[2] >= data_shape[2])
            ):
                insert_pre = True
            else:
                print(
                    "WARNING: discarding presynaptic point {} which seems to be out of shape: {}".format(
                        pre_loc, data_shape
                    )
                )
            # Post point in data shape
            if not (
                (post_loc[0] < 0 or post_loc[0] >= data_shape[0])
                or (post_loc[1] < 0 or post_loc[1] >= data_shape[1])
                or (post_loc[2] < 0 or post_loc[2] >= data_shape[2])
            ):
                insert_post = True
            else:
                print(
                    "WARNING: discarding postsynaptic point {} which seems to be out of shape: {}".format(
                        post_loc, data_shape
                    )
                )

            pre_key = str(pre_loc)
            if insert_pre and insert_post and pre_key not in pre_post_points:
                pre_post_points[pre_key] = []
            pre_post_points[pre_key].append(post_loc)

        if len(pre_post_points) > 0:
            # Create the Zarr file where the mask will be placed
            fname = os.path.join(savepath, os.path.basename(filename))
            os.makedirs(savepath, exist_ok=True)
            if any(fname.endswith(x) for x in [".h5", ".hdf5", ".hdf"]):
                fid_mask = h5py.File(fname, "w")
            else:  # Zarr file
                fid_mask = zarr.open_group(fname, mode="w")

            # Determine data shape
            out_data_shape = np.array(data_shape)
            if "C" not in zarr_data_information["axis_order"]:
                out_data_shape = tuple(out_data_shape) + (channels,)
                out_data_order = zarr_data_information["axis_order"] + "C"
                c_axe_pos = -1
            else:
                out_data_shape[zarr_data_information["axis_order"].index("C")] = channels
                out_data_shape = tuple(out_data_shape)
                out_data_order = zarr_data_information["axis_order"]
                c_axe_pos = zarr_data_information["axis_order"].index("C")

            if any(fname.endswith(x) for x in [".h5", ".hdf5", ".hdf"]):
                mask = fid_mask.create_dataset("data", shape=out_data_shape, dtype=dtype_str)
                # mask = fid_mask.create_dataset("data", out_data_shape, compression="lzf", dtype=dtype_str)
            else:
                mask = fid_mask.create_dataset("data", shape=out_data_shape, dtype=dtype_str)

            print("Paiting all postsynaptic sites")
            for pre_site, post_sites in tqdm(pre_post_points.items(), disable=not is_main_process()):
                pre_point_global = [int(float(x)) for x in " ".join(pre_site[1:-1].split()).split(" ")]

                # Take the patch to extract so to draw all the postsynaptic sites using the minimun patch size
                patch_coords = None
                for post_point in post_sites:
                    if patch_coords is None:
                        patch_coords = [
                            max(0, post_point[0] - dilation_width[0]),
                            min(out_data_shape[zarr_data_information["z_axe_pos"]], post_point[0] + dilation_width[0]),
                            max(0, post_point[1] - dilation_width[1]),
                            min(out_data_shape[zarr_data_information["y_axe_pos"]], post_point[1] + dilation_width[1]),
                            max(0, post_point[2] - dilation_width[2]),
                            min(out_data_shape[zarr_data_information["x_axe_pos"]], post_point[2] + dilation_width[2]),
                        ]
                    else:
                        patch_coords = [
                            min(max(0, post_point[0] - dilation_width[0]), patch_coords[0]),
                            max(
                                min(
                                    out_data_shape[zarr_data_information["z_axe_pos"]],
                                    post_point[0] + dilation_width[0],
                                ),
                                patch_coords[1],
                            ),
                            min(max(0, post_point[1] - dilation_width[1]), patch_coords[2]),
                            max(
                                min(
                                    out_data_shape[zarr_data_information["y_axe_pos"]],
                                    post_point[1] + dilation_width[1],
                                ),
                                patch_coords[3],
                            ),
                            min(max(0, post_point[2] - dilation_width[2]), patch_coords[4]),
                            max(
                                min(
                                    out_data_shape[zarr_data_information["x_axe_pos"]],
                                    post_point[2] + dilation_width[2],
                                ),
                                patch_coords[5],
                            ),
                        ]

                assert patch_coords
                patch_coords = [int(x) for x in patch_coords]
                patch_shape = (
                    patch_coords[1] - patch_coords[0],
                    patch_coords[3] - patch_coords[2],
                    patch_coords[5] - patch_coords[4],
                )
                seeds = np.zeros(patch_shape, dtype=np.uint64)
                mask_to_grow = np.zeros(patch_shape, dtype=np.uint8)

                # Paiting each post-synaptic site
                label_to_pre_site = {}
                label_count = 1
                for post_point_global in post_sites:
                    post_point = [
                        int(post_point_global[0] - patch_coords[0]),
                        int(post_point_global[1] - patch_coords[2]),
                        int(post_point_global[2] - patch_coords[4]),
                    ]
                    pre_point = [
                        int(pre_point_global[0] - patch_coords[0]),
                        int(pre_point_global[1] - patch_coords[2]),
                        int(pre_point_global[2] - patch_coords[4]),
                    ]

                    if (
                        post_point[0] < seeds.shape[0]
                        and post_point[1] < seeds.shape[1]
                        and post_point[2] < seeds.shape[2]
                    ):
                        seeds[
                            max(0, post_point[0] - dilation_width[0]) : min(
                                post_point[0] + dilation_width[0], seeds.shape[0]
                            ),
                            post_point[1],
                            post_point[2],
                        ] = label_count
                        label_to_pre_site[label_count] = list(pre_point)
                        label_count += 1

                        mask_to_grow[post_point[0], post_point[1], post_point[2]] = 1
                    else:
                        raise ValueError(
                            "Point {} seems to be out of shape: {}".format(
                                [post_point[0], post_point[1], post_point[2]], seeds.shape
                            )
                        )

                # First channel creation
                channel_0 = binary_dilation_scipy(
                    mask_to_grow,
                    iterations=1,
                    structure=ellipse_footprint_cpd,
                )
                mask_to_grow = binary_dilation_scipy(
                    mask_to_grow,
                    iterations=1,
                    structure=ellipse_footprint_cpd2,
                )
                for z in range(len(seeds)):
                    semantic = distance_transform_edt(mask_to_grow[z])
                    assert isinstance(semantic, np.ndarray)
                    seeds[z] = watershed(-semantic, seeds[z], mask=mask_to_grow[z])

                # Flow channel creation
                hv_map = create_flow_channels(
                    seeds,
                    ref_point="presynaptic",
                    label_to_pre_site=label_to_pre_site,
                    normalize_values=normalize_values,
                )

                hv_map = np.concatenate([np.expand_dims(channel_0, -1), hv_map], axis=-1)
                del channel_0

                slices = (
                    slice(patch_coords[0], patch_coords[1]),
                    slice(patch_coords[2], patch_coords[3]),
                    slice(patch_coords[4], patch_coords[5]),
                    slice(0, out_data_shape[c_axe_pos]),
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
                current_order = np.array(range(len(hv_map.shape)))
                transpose_order = order_dimensions(
                    current_order,
                    input_order="ZYXC",
                    output_order=out_data_order,
                    default_value=np.nan,
                )
                transpose_order = [x for x in np.array(transpose_order) if not np.isnan(x)]

                # Place the patch into the Zarr
                mask[data_ordered_slices] += hv_map.transpose(transpose_order) * (mask[data_ordered_slices] == 0)

            # Close file
            if isinstance(fid_mask, h5py.File):
                fid_mask.close()


def create_flow_channels(
    data: NDArray, 
    ref_point: str="center", 
    label_to_pre_site: Optional[Dict]=None, 
    normalize_values: bool=True
):
    """
    Obtain the horizontal and vertical distance maps for each instance. Depth distance is also calculated if
    the ``data`` provided is 3D.

    Parameters
    ----------
    data : 2D/3D Numpy array
        Instance mask to create the flow channels from. E.g. ``(500, 500)`` for 2D and ``(200, 1000, 1000)`` for 3D.

    ref_point : str, optional
        Reference point to be used to create the flow channels. Possible values: ``center``, ``presynaptic``.
         - 'center': point to the centroid.
         - 'presynaptic': point to the presynaptic site. To use this ``label_to_pre_site`` must be provided.

    label_to_pre_site : dict, optional
        Reference of the presynaptic site for each label within the provided volume (``data``).

    normalize_values : bool, optional
        Whether to normalize the values or not.

    Returns
    -------
    new_mask : 3D/4D Numpy array
        Flow channels. E.g. ``(500, 500, 2)`` for 2D and ``(200, 1000, 1000, 3)`` for 3D.
    """
    assert ref_point in ["center", "presynaptic"]
    if ref_point == "presynaptic" and label_to_pre_site is None:
        raise ValueError("'label_to_pre_site' must be provided when 'ref_point' is 'presynaptic'")

    orig_data = data.copy()  # instance ID map
    dim = data.ndim
    x_map = np.zeros(orig_data.shape, dtype=np.float32)
    y_map = np.zeros(orig_data.shape, dtype=np.float32)
    if dim == 3:
        z_map = np.zeros(orig_data.shape, dtype=np.float32)

    props = regionprops_table(orig_data, properties=("label", "bbox", "centroid"))
    for k, inst_id in tqdm(enumerate(props["label"]), total=len(props["label"]), leave=False):
        inst_map = np.array(orig_data == inst_id, np.uint8)
        if dim == 2:
            inst_box = [props["bbox-0"][k], props["bbox-2"][k], props["bbox-1"][k], props["bbox-3"][k]]
        else:
            inst_box = [
                props["bbox-0"][k],
                props["bbox-3"][k],
                props["bbox-1"][k],
                props["bbox-4"][k],
                props["bbox-2"][k],
                props["bbox-5"][k],
            ]

        # Extract the patch
        if dim == 2:
            inst_box[0] = max(0, inst_box[0] - 2)
            inst_box[2] = max(0, inst_box[2] - 2)
            inst_box[1] = min(inst_map.shape[0], inst_box[1] + 2)
            inst_box[3] = min(inst_map.shape[1], inst_box[3] + 2)
            inst_map = inst_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
        else:
            inst_box[0] = max(0, inst_box[0] - 2)
            inst_box[2] = max(0, inst_box[2] - 2)
            inst_box[4] = max(0, inst_box[4] - 2)
            inst_box[1] = min(inst_map.shape[0], inst_box[1] + 2)
            inst_box[3] = min(inst_map.shape[1], inst_box[3] + 2)
            inst_box[5] = min(inst_map.shape[2], inst_box[5] + 2)
            inst_map = inst_map[
                inst_box[0] : inst_box[1],
                inst_box[2] : inst_box[3],
                inst_box[4] : inst_box[5],
            ]

        if dim == 2 and (inst_map.shape[0] < 2 or inst_map.shape[1] < 2):
            continue
        elif dim == 3 and (inst_map.shape[0] < 2 or inst_map.shape[1] < 2 or inst_map.shape[2] < 2):
            continue

        # instance center of mass, rounded to nearest pixel
        if ref_point == "center":
            if dim == 2:
                inst_com = [
                    props["centroid-0"][k],
                    props["centroid-1"][k],
                ]
            else:
                inst_com = [
                    props["centroid-0"][k],
                    props["centroid-1"][k],
                    props["centroid-2"][k],
                ]
        else:  # presynaptic
            assert label_to_pre_site
            if inst_id not in label_to_pre_site:
                raise ValueError(f"Label {inst_id} not in 'label_to_pre_site'")
            inst_com = label_to_pre_site[inst_id]

        # Move reference point inside bbox
        inst_com[0] -= inst_box[0]
        inst_com[1] -= inst_box[2]
        if dim == 3:
            inst_com[2] -= inst_box[4]

        if any(np.isnan(inst_com)):
            continue

        if dim == 2:
            inst_com[0] = int(inst_com[0] + 0.5)
            inst_com[1] = int(inst_com[1] + 0.5)
            inst_y_range = np.arange(1, inst_map.shape[0] + 1)
            inst_x_range = np.arange(1, inst_map.shape[1] + 1)

            # shifting center of pixels grid to instance center of mass/presynaptic site
            inst_x_range -= inst_com[1]
            inst_y_range -= inst_com[0]

            inst_x, inst_y = np.meshgrid(inst_x_range, inst_y_range)
        else:
            inst_com[0] = int(inst_com[0] + 0.5)
            inst_com[1] = int(inst_com[1] + 0.5)
            inst_com[2] = int(inst_com[2] + 0.5)
            inst_z_range = np.arange(1, inst_map.shape[0] + 1)
            inst_y_range = np.arange(1, inst_map.shape[1] + 1)
            inst_x_range = np.arange(1, inst_map.shape[2] + 1)

            # shifting center of pixels grid to instance center of mass/presynaptic site
            inst_z_range -= inst_com[0]
            inst_y_range -= inst_com[1]
            inst_x_range -= inst_com[2]

            inst_z, inst_y, inst_x = np.meshgrid(inst_z_range, inst_y_range, inst_x_range, indexing="ij")

            # remove coord outside of instance (Z)
            inst_z[inst_map == 0] = 0
            inst_z = inst_z.astype("float32")

        # remove coord outside of instance (Y and X)
        inst_y[inst_map == 0] = 0
        inst_x[inst_map == 0] = 0
        inst_x = inst_x.astype("float32")
        inst_y = inst_y.astype("float32")

        if normalize_values:
            # normalize min into -1 scale
            if np.min(inst_y) < 0:
                inst_y[inst_y < 0] /= -np.amin(inst_y[inst_y < 0])
            if np.min(inst_x) < 0:
                inst_x[inst_x < 0] /= -np.amin(inst_x[inst_x < 0])
            if dim == 3:
                if np.min(inst_z) < 0:
                    inst_z[inst_z < 0] /= -np.amin(inst_z[inst_z < 0])
            # normalize max into +1 scale
            if np.max(inst_y) > 0:
                inst_y[inst_y > 0] /= np.amax(inst_y[inst_y > 0])
            if np.max(inst_x) > 0:
                inst_x[inst_x > 0] /= np.amax(inst_x[inst_x > 0])
            if dim == 3:
                if np.max(inst_z) > 0:
                    inst_z[inst_z > 0] /= np.amax(inst_z[inst_z > 0])

        if dim == 2:
            x_map_box = x_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
            x_map_box[inst_map > 0] = inst_x[inst_map > 0]

            y_map_box = y_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
            y_map_box[inst_map > 0] = inst_y[inst_map > 0]
        else:
            z_map_box = z_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3], inst_box[4] : inst_box[5]]
            z_map_box[inst_map > 0] = inst_z[inst_map > 0]

            y_map_box = y_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3], inst_box[4] : inst_box[5]]
            y_map_box[inst_map > 0] = inst_y[inst_map > 0]

            x_map_box = x_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3], inst_box[4] : inst_box[5]]
            x_map_box[inst_map > 0] = inst_x[inst_map > 0]

    if dim == 2:
        hv_map = np.dstack([y_map, x_map])
    else:
        hv_map = np.stack([z_map, y_map, x_map], axis=-1)

    return hv_map


#############
# DETECTION #
#############
def generate_ellipse_footprint(
    shape=[1, 1, 1],
) -> NDArray:
    """
    Generate footprint of an ellipse in a n-dimensional image.

    Parameters
    ----------
    shape : list, optional
        Shape of the hyperball with the given side lengths.

    Returns
    -------
    distances : np.ndarray
        Ellipse footprint.
    """
    center = (np.array(shape) / 2).astype(int)

    ranges = [
        np.arange(int(center[i] - shape[i]), int(center[i] + shape[i]) + 1) if shape[i] > 0 else [center[i]]
        for i in range(len(center))
    ]
    grids = np.meshgrid(*ranges, indexing="ij")

    # put all the dimensions at least to 1
    shape = [1 if i == 0 else i for i in shape]

    distances = np.array([((grids[d] - center[d]) ** 2) / shape[d] ** 2 for d in range(len(center))])
    distances = np.sum(distances, axis=0) <= 1

    return distances.astype(bool)


def create_detection_masks(
    cfg: Config, 
    data_type: str="train"
):
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

    channels = 2 if cfg.MODEL.N_CLASSES > 2 else 1
    dtype = np.uint8 if cfg.MODEL.N_CLASSES < 255 else np.uint16
    if len(img_ids) != len(ids):
        raise ValueError(
            "Different number of CSV files and images found ({} vs {}). "
            "Please check that every image has one and only one CSV file".format(len(ids), len(img_ids))
        )
    if cfg.PROBLEM.NDIM == "2D":
        req_columns = ["axis-0", "axis-1"] if channels == 1 else ["axis-0", "axis-1", "class"]
    else:
        req_columns = ["axis-0", "axis-1", "axis-2"] if channels == 1 else ["axis-0", "axis-1", "axis-2", "class"]

    cpd = cfg.PROBLEM.DETECTION.CENTRAL_POINT_DILATION
    ellipse_footprint = generate_ellipse_footprint(cpd)

    print("Creating {} detection masks . . .".format(data_type))
    for i in range(len(ids)):
        img_filename = os.path.splitext(ids[i])[0] + img_ext
        if not os.path.exists(os.path.join(out_dir, img_filename)) and not os.path.exists(
            os.path.join(out_dir, img_ids[i])
        ):
            file_path = os.path.join(label_dir, ids[i])
            print("Attempting to create mask from CSV file: {}".format(file_path))
            if not os.path.exists(os.path.join(img_dir, img_filename)):
                print(
                    "WARNING: The image seems to have different name than its CSV file. Using the CSV file that's "
                    "in the same spot (within the CSV files list) where the image is in its own list of images. Check if it is correct!"
                )
                img_filename = img_ids[i]
            print("Its respective image seems to be: {}".format(os.path.join(img_dir, img_filename)))

            df = pd.read_csv(file_path)
            df = df.dropna()
            if ".zarr" != img_ext:
                img = read_img_as_ndarray(
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
                    m = f"'{cols_not_in_file[0]}' column is not present in CSV file: {file_path}"
                else:
                    m = f"{cols_not_in_file} columns are not present in CSV file: {file_path}"
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
            else:
                if cfg.MODEL.N_CLASSES > 2:
                    raise ValueError("MODEL.N_CLASSES > 2 but no class specified in CSV file")
                class_point = [1] * len(z_axis_point)

            # Create masks
            print("Creating all points . . .")
            mask = np.zeros((shape + (channels,)), dtype=dtype)
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
                c_point = class_point[j]

                if c_point > cfg.MODEL.N_CLASSES:
                    raise ValueError(
                        "Class {} detected while 'MODEL.N_CLASSES' was set to {}. Please check it!".format(
                            c_point, cfg.MODEL.N_CLASSES
                        )
                    )

                # Paint the point
                if cfg.PROBLEM.NDIM == "3D":
                    if a0_coord < mask.shape[0] and a1_coord < mask.shape[1] and a2_coord < mask.shape[2]:
                        patch_coords = [
                            max(0, a0_coord - 1 - cpd[0]),
                            min(mask.shape[0], a0_coord + 1 + cpd[0]),
                            max(0, a1_coord - 1 - cpd[1]),
                            min(mask.shape[1], a1_coord + 1 + cpd[1]),
                            max(0, a2_coord - 1 - cpd[2]),
                            min(mask.shape[2], a2_coord + 1 + cpd[2]),
                        ]
                        patch_shape = (
                            patch_coords[1] - patch_coords[0],
                            patch_coords[3] - patch_coords[2],
                            patch_coords[5] - patch_coords[4],
                        )
                        if (
                            1
                            in mask[
                                patch_coords[0] : patch_coords[1],
                                patch_coords[2] : patch_coords[3],
                                patch_coords[4] : patch_coords[5],
                                0,
                            ]
                        ):
                            print(
                                "WARNING: possible duplicated point in (3,9,9) neighborhood: coords {} , class {} "
                                "(point number {} in CSV)".format((a0_coord, a1_coord, a2_coord), c_point, p_number[j])
                            )

                        # Move coordinates to local patch boundaries
                        a0_coord = a0_coord - patch_coords[0]
                        a1_coord = a1_coord - patch_coords[2]
                        a2_coord = a2_coord - patch_coords[4]

                        patch = np.zeros(patch_shape, dtype=dtype)
                        patch[a0_coord, a1_coord - 1 : a1_coord + 1, a2_coord - 1 : a2_coord + 1] = 1
                    else:
                        print(
                            "WARNING: discarding point {} which seems to be out of shape: {}".format(
                                [a0_coord, a1_coord, a2_coord], shape
                            )
                        )
                else:
                    if a0_coord < mask.shape[0] and a1_coord < mask.shape[1]:
                        patch_coords = [
                            max(0, a0_coord - 1 - cpd[0]),
                            min(mask.shape[0], a0_coord + 1 + cpd[0]),
                            max(0, a1_coord - 1 - cpd[1]),
                            min(mask.shape[1], a1_coord + 1 + cpd[1]),
                        ]
                        patch_shape = (
                            patch_coords[1] - patch_coords[0],
                            patch_coords[3] - patch_coords[2],
                        )

                        if (
                            1
                            in mask[
                                patch_coords[0] : patch_coords[1],
                                patch_coords[2] : patch_coords[3],
                                0,
                            ]
                        ):
                            print(
                                "WARNING: possible duplicated point in (9,9) neighborhood: coords {} , class {} "
                                "(point number {} in CSV)".format((a0_coord, a1_coord), c_point, p_number[j])
                            )

                        # Move coordinates to local patch boundaries
                        a0_coord = a0_coord - patch_coords[0]
                        a1_coord = a1_coord - patch_coords[2]

                        patch = np.zeros(patch_shape, dtype=dtype)
                        patch[a0_coord, a1_coord - 1 : a1_coord + 1] = 1
                    else:
                        print(
                            "WARNING: discarding point {} which seems to be out of shape: {}".format(
                                [a0_coord, a1_coord], shape
                            )
                        )
                patch = binary_dilation_scipy(patch, iterations=1, structure=ellipse_footprint)
                patch = np.expand_dims(patch, -1)
                if channels > 1:
                    patch = np.concatenate([patch, patch.copy() * c_point], axis=-1)

                # Insert the information without touching previous points
                if cfg.PROBLEM.NDIM == "3D":
                    mask[
                        patch_coords[0] : patch_coords[1],
                        patch_coords[2] : patch_coords[3],
                        patch_coords[4] : patch_coords[5],
                    ] = patch * (
                        mask[
                            patch_coords[0] : patch_coords[1],
                            patch_coords[2] : patch_coords[3],
                            patch_coords[4] : patch_coords[5],
                        ]
                        == 0
                    )
                else:
                    mask[
                        patch_coords[0] : patch_coords[1],
                        patch_coords[2] : patch_coords[3],
                    ] = patch * (
                        mask[
                            patch_coords[0] : patch_coords[1],
                            patch_coords[2] : patch_coords[3],
                        ]
                        == 0
                    )

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
                        label(clear_border(mask[..., ch])),# type: ignore
                        return_counts=True,
                        return_index=True,
                    ) # type: ignore
                    # 0 is background so valid element is 1. We will compare that value with the rest
                    if len(counts) > 1:
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
                        file_path,
                    )
                )
            else:
                print(
                    "Mask file {} found for CSV file: {}".format(
                        os.path.join(out_dir, img_ids[i]),
                        file_path,
                    )
                )


#######
# SSL #
#######
def create_ssl_source_data_masks(
    cfg: Config, 
    data_type: str ="train"
):
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
            img = read_img_as_ndarray(os.path.join(img_dir, ids[i]), is_3d=not cfg.PROBLEM.NDIM == "2D")
            img = crappify(
                img,
                resizing_factor=cfg.PROBLEM.SELF_SUPERVISED.RESIZING_FACTOR,
                add_noise=add_noise,
                noise_level=cfg.PROBLEM.SELF_SUPERVISED.NOISE,
            )

            save_tif(np.expand_dims(img, 0), out_dir, [ids[i]])
        else:
            print("Source file {} found".format(os.path.join(img_dir, ids[i])))


def crappify(
    input_img: NDArray, 
    resizing_factor: float, 
    add_noise: bool=True, 
    noise_level: Optional[float]=None, 
    Down_up: bool=True
):
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
        assert noise_level
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


def add_gaussian_noise(
    image: NDArray, 
    percentage_of_noise: float
) -> NDArray:
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
def calculate_volume_prob_map(
    Y: BiaPyDataset, 
    is_3d: bool=False, 
    w_foreground: float=0.94, 
    w_background: float=0.06, 
    save_dir=None
) -> List[NDArray] | NDArray:
    """
    Calculate the probability map of the given data.

    Parameters
    ----------
    Y : list of dict
        Data to calculate the probability map from. Each item in the list represents a sample of the dataset.
        Expected keys:
            * ``"filename"``: name of the image to extract the data sample from.
            * ``"dir"``: directory where the image resides.
            * ``"img"``: image sample itself. It is a ndarrray of  ``(y, x, channels)`` in ``2D`` and
              ``(z, y, x, channels)``in ``3D``. Provided if the user selected to load data into memory.
        If ``"img"`` is provided ``"filename"`` and ``"filename"`` are not necessary, and vice versa.

    w_foreground : float, optional
        Weight of the foreground. This value plus ``w_background`` must be equal ``1``.

    w_background : float, optional
        Weight of the background. This value plus ``w_foreground`` must be equal ``1``.

    save_dir : str, optional
        Path to the file where the probability map will be stored.

    Returns
    -------
    maps : NDArray or list of NDArray
        Probability map(s) of all samples in ``Y.sample_list``.
    """
    print("Constructing the probability map . . .")
    maps = []
    diff_shape = False
    first_shape = None
    Ylen = len(Y.sample_list)
    for i in tqdm(range(Ylen), disable=not is_main_process()):
        if Y.sample_list[i].img_is_loaded():
            _map = Y.sample_list[i].img.copy().astype(np.float32)
        else:
            path = Y.dataset_info[Y.sample_list[i].fid].path
            _map = read_img_as_ndarray(path, is_3d=is_3d).astype(np.float32)

        for k in range(_map.shape[-1]):
            if is_3d:
                for j in range(_map.shape[0]):
                    # Remove artifacts connected to image border
                    _map[j, ..., k] = clear_border(_map[j, ..., k])
            else:
                # Remove artifacts connected to image border
                _map[..., k] = clear_border(_map[..., k])

            foreground_pixels = (_map[..., k] > 0).sum()
            background_pixels = (_map[..., k] == 0).sum()

            if foreground_pixels == 0:
                _map[..., k][np.where(_map[..., k] > 0)] = 0
            else:
                _map[..., k][np.where(_map[..., k] > 0)] = w_foreground / foreground_pixels
            if background_pixels == 0:
                _map[..., k][np.where(_map[..., k] == 0)] = 0
            else:
                _map[..., k][np.where(_map[..., k] == 0)] = w_background / background_pixels

            # Necessary to get all probs sum 1
            s = _map[..., k].sum()
            if s == 0:
                t = 1
                for x in _map[..., k].shape:
                    t *= x
                _map[..., k].fill(1 / t)
            else:
                _map[..., k] = _map[..., k] / _map[..., k].sum()
        if first_shape is None:
            first_shape = _map.shape
        if first_shape != _map.shape:
            diff_shape = True
        maps.append(_map)

    if not diff_shape:
        for i in range(len(maps)):
            maps[i] = np.expand_dims(maps[i], 0)
        maps = np.concatenate(maps)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        if not diff_shape:
            print("Saving the probability map in {}".format(save_dir))
            np.save(os.path.join(save_dir, "prob_map.npy"), maps)
            return maps
        else:
            print(
                "As the files loaded have different shapes, the probability map for each one will be stored"
                " separately in {}".format(save_dir)
            )
            d = len(str(Ylen))
            for i in range(Ylen):
                f = os.path.join(save_dir, "prob_map" + str(i).zfill(d) + ".npy")
                np.save(f, maps[i])
    return maps


###########
# GENERAL #
###########

def resize_images(
    images: List[NDArray], 
    **kwards
) -> List[NDArray]:
    """
    The function resizes all the images using the specified parameters or default values if not provided.

    Parameters
    ----------
    images: list of Numpy arrays
        The `images` parameter is the list of all input images that you want to resize.

    output_shape: iterable
        Size of the generated output image. E.g. `(256,256)`

    (kwards): optional
        `skimage.transform.resize() <https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.resize>`__
        parameters are also allowed.

    Returns
    -------
    resized_images: list of Numpy arrays
        The resized images. The returned data will use the same data type as the given `images`.

    """
    resized_images = [resize(img, **kwards).astype(img.dtype) for img in images]
    return resized_images


def apply_gaussian_blur(
    images: List[NDArray], 
    **kwards
) -> List[NDArray]:
    """
    The function applies a Gaussian blur to all images.

    Parameters
    ----------
    images: list of Numpy arrays
        The input images on which the Gaussian blur will be applied.

    (kwards): optional
        `skimage.filters.gaussian() <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.gaussian>`__
        parameters are also allowed.

    Returns
    -------
    blurred_images: list of Numpy arrays
        A Gaussian blurred images. The returned data will use the same data type as the given `images`.

    """

    def _process(image, **kwards):
        im = gaussian(image, **kwards)  # returns 0-1 range
        if np.issubdtype(image.dtype, np.integer):
            im = im * np.iinfo(image.dtype).max
        im = im.astype(image.dtype)
        return im

    blurred_images = [_process(img, **kwards) for img in images]
    return blurred_images


def apply_median_blur(
    images: List[NDArray], 
    **kwards
) -> List[NDArray]:
    """
    The function applies a median blur filter to all images.

    Parameters
    ----------
    image: list of Numpy arrays
        The input image on which the median blur operation will be applied.

    (kwards): optional
        `skimage.filters.median() <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.median>`__
        parameters are also allowed.

    Returns
    -------
    blurred_images: list of Numpy arrays
        The median-blurred images. The returned data will use the same data type as the given `images`.

    """
    blurred_images = [median(img, **kwards).astype(img.dtype) for img in images]
    return blurred_images


def detect_edges(
    images: List[NDArray], 
    **kwards
) -> List[NDArray]:
    """
    The function `detect_edges` takes the 2D images as input, converts it to grayscale if necessary, and
    applies the Canny edge detection algorithm to detect edges in the image.

    Parameters
    ----------
    images: list of Numpy arrays
        The list of all input images on which the edge detection will be performed. It can be either a color image with
        shape (height, width, 3) or a grayscale image with shape (height, width, 1).

    (kwards): optional
        `skimage.feature.canny() <https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.canny>`__
        parameters are also allowed.

    Returns
    -------
    edges: list of Numpy arrays
        The edges of the input images. The returned Numpy arrays will be uint8, where background is black (0) and edges white (255).
        The returned data will use the same structure as the given `images` (list[Numpy array] or Numpy array).

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
    return edges


def _histogram_matching(
    source_imgs: List[NDArray], 
    target_imgs: List[NDArray]
) -> List[NDArray]:
    """
    Given a set of target images, it will obtain their mean histogram
    and applies histogram matching to all images from sorce images.

    Parameters
    ----------
    source_imgs: list of Numpy arrays
        The images of the source domain, to which the histogram matching is to be applied.

    target_imgs: list of Numpy array
        The target domain images, from which mean histogram will be obtained.

    Returns
    -------
    matched_images : list of Numpy arrays
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


def apply_histogram_matching(
    images: List[NDArray], 
    reference_path: str, 
    is_2d: bool
):
    """
    The function returns the images with their histogram matched to the histogram of the reference images,
    loaded from the given reference_path.

    Parameters
    ----------
    images: list of Numpy arrays
        The list of input images whose histogram needs to be matched to the reference histogram. It should be a
        Numpy array representing the image.

    reference_path: str
        The reference_path is the directory path to the reference images. From reference images, we will extract
        the reference histogram with which we want to match the histogram of the images. It represents
        the desired distribution of pixel intensities in the output image.

    is_2d: bool, optional
        The value indicate if the data given in ``reference_path`` is 2D (``is_2d = True``) or 3D (``is_2d = False``).
        Defaults to True.

    Returns
    -------
    matched_images : list of Numpy arrays
        The result of matching the histogram of the input images to the histogram of the reference image.
        The returned data will use the same data type as the given `images`.
    """
    references = load_data_from_dir(reference_path, is_3d=not is_2d)
    matched_images = _histogram_matching(images, [x.img for x in references.sample_list if x.img_is_loaded()])
    return matched_images


def apply_clahe(
    images: List[NDArray], 
    **kwards
) -> List[NDArray] :
    """
    The function applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to an image and
    returns the result.

    Parameters
    ----------
    images: list of Numpy arrays
        The list of input images that you want to apply the CLAHE (Contrast Limited Adaptive Histogram Equalization)
        algorithm to.

    (kwards): optional
        `skimage.exposure.equalize_adapthist() <https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.equalize_adapthist>`__
        parameters are also allowed.

    Returns
    -------
    processed_images: list of Numpy arrays
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
    return processed_images


def preprocess_data(
    cfg: Config, 
    x_data: List[NDArray]=[], 
    y_data: List[NDArray]=[], 
    is_2d: bool=True, 
    is_y_mask: bool=False
) -> List[NDArray] | Tuple[List[NDArray], List[NDArray]]:
    """
    The function preprocesses data by applying various image processing techniques.

    Parameters
    ----------
    cfg: dict
        The `cfg` parameter is a configuration object that contains various settings for
        preprocessing the data. It is used to control the behavior of different preprocessing techniques
        such as image resizing, blurring, histogram matching, etc.

    x_data: list of 3D/4D Numpy arrays, optional
        The input data (images) to be preprocessed. The first dimension must be the number of images.
        E.g. ``(num_of_images, y, x, channels)`` or ``(num_of_images, z, y, x, channels)``.
        In case of using a list, the format of the images remains the same. Each item in the list
        corresponds to a different image.

    y_data: list of 3D/4D Numpy arrays, optional
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
    x_data: list of 3D/4D Numpy arrays, optional
        Preprocessed data. The same structure and dimensionality of the given data will be returned.

    y_data: list of 3D/4D Numpy arrays, optional
        Preprocessed data. The same structure and dimensionality of the given data will be returned.
    """

    if len(y_data) > 0:
        if cfg.RESIZE.ENABLE:
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
        if cfg.RESIZE.ENABLE:
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
        if cfg.GAUSSIAN_BLUR.ENABLE:
            x_data = apply_gaussian_blur(
                x_data,
                sigma=cfg.GAUSSIAN_BLUR.SIGMA,
                mode=cfg.GAUSSIAN_BLUR.MODE,
                channel_axis=cfg.GAUSSIAN_BLUR.CHANNEL_AXIS,
            )
        if cfg.MEDIAN_BLUR.ENABLE:
            x_data = apply_median_blur(
                x_data,
                footprint=np.ones(cfg.MEDIAN_BLUR.KERNEL_SIZE, dtype=np.uint8).tolist(),
            )
        if cfg.MATCH_HISTOGRAM.ENABLE:
            x_data = apply_histogram_matching(
                x_data,
                reference_path=cfg.MATCH_HISTOGRAM.REFERENCE_PATH,
                is_2d=is_2d,
            )
        if cfg.CLAHE.ENABLE:
            x_data = apply_clahe(
                x_data,
                kernel_size=cfg.CLAHE.KERNEL_SIZE,
                clip_limit=cfg.CLAHE.CLIP_LIMIT,
            )
        if cfg.CANNY.ENABLE:
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
