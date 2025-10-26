"""
Pre-processing utilities for image and mask data in deep learning workflows.

This module provides pre-processing functions for instance segmentation, detection mask creation, self-supervised learning data generation, semantic segmentation probability maps, and general image processing operations such as resizing, blurring, edge detection, histogram matching, and CLAHE. It supports both 2D and 3D data formats and integrates with BiaPy configuration objects for flexible data pipelines.
"""
import os
import edt
import h5py
import zarr
import numpy as np
from tqdm import tqdm
import pandas as pd
from skimage.segmentation import clear_border, find_boundaries, watershed
from scipy.ndimage import generic_filter, generate_binary_structure, grey_closing, center_of_mass, map_coordinates, binary_dilation as binary_dilation_scipy
from skimage.morphology import disk, ball, binary_dilation, binary_erosion, skeletonize
from skimage.measure import label, regionprops_table
from skimage.transform import resize
from skimage.feature import canny
from skimage.exposure import equalize_adapthist
from skimage.color import rgb2gray
from skimage.filters import gaussian, median
from yacs.config import CfgNode as CN
from numpy.typing import NDArray
from typing import List, Optional, Dict, Tuple, Sequence

from biapy.data.dataset import BiaPyDataset
from biapy.utils.util import (
    seg2aff_pni,
    seg_widen_border,
)
from biapy.utils.misc import is_main_process, get_rank, get_world_size, os_walk_clean
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

#########################
# INSTANCE SEGMENTATION #
#########################
def create_instance_channels(cfg: CN, data_type: str = "train"):
    """
    Create training and validation new data with appropiate channels based on ``PROBLEM.INSTANCE_SEG.DATA_CHANNELS`` for instance segmentation.

    Parameters
    ----------
    cfg : YACS CN object
        Configuration.

    data_type: str, optional
        Wheter to create training or validation instance channels.
    """
    assert data_type in ["train", "val", "test"]
    if data_type == "train":
        tag = "TRAIN"
    elif data_type == "val":
        tag = "VAL"
    else:  # test
        tag = "TEST"

    # Checking if the user inputted Zarr/H5 files
    if getattr(cfg.DATA, tag).INPUT_ZARR_MULTIPLE_DATA:
        try:
            zarr_files = next(os_walk_clean(getattr(cfg.DATA, tag).PATH))[1]
        except StopIteration:
            raise ValueError("No Zarr/N5 files found in the input path: {}".format(getattr(cfg.DATA, tag).PATH))
        try:
            h5_files = next(os_walk_clean(getattr(cfg.DATA, tag).PATH))[2]
        except StopIteration:
            raise ValueError("No H5 files found in the input path: {}".format(getattr(cfg.DATA, tag).PATH))
    else:
        try:
            zarr_files = next(os_walk_clean(getattr(cfg.DATA, tag).GT_PATH))[1]
            h5_files = next(os_walk_clean(getattr(cfg.DATA, tag).GT_PATH))[2]
        except StopIteration:
            raise ValueError("No Zarr/N5 or H5 files found in the GT path: {}".format(getattr(cfg.DATA, tag).GT_PATH))

    # Find patches info so we can iterate over them to create the instance mask
    working_with_zarr_h5_files = False
    if (
        cfg.PROBLEM.NDIM == "3D"
        and (len(zarr_files) > 0 and any(True for x in [".zarr", ".n5"] if x in zarr_files[0]))
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

        if len(zarr_files) > 0 and any(True for x in [".zarr", ".n5"] if x in zarr_files[0]):
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
            "axes_order": getattr(cfg.DATA, tag).INPUT_IMG_AXES_ORDER,
            "z_axe_pos": getattr(cfg.DATA, tag).INPUT_IMG_AXES_ORDER.index("Z"),
            "y_axe_pos": getattr(cfg.DATA, tag).INPUT_IMG_AXES_ORDER.index("Y"),
            "x_axe_pos": getattr(cfg.DATA, tag).INPUT_IMG_AXES_ORDER.index("X"),
            "id_path": getattr(cfg.DATA, tag).INPUT_ZARR_MULTIPLE_DATA_ID_PATH,
            "partners_path": getattr(cfg.DATA, tag).INPUT_ZARR_MULTIPLE_DATA_PARTNERS_PATH,
            "locations_path": getattr(cfg.DATA, tag).INPUT_ZARR_MULTIPLE_DATA_LOCATIONS_PATH,
            "resolution_path": getattr(cfg.DATA, tag).INPUT_ZARR_MULTIPLE_DATA_RESOLUTION_PATH,
        }

    else:
        if cfg.PROBLEM.INSTANCE_SEG.TYPE == "synapses":
            raise ValueError("Synapse detection is only available for 3D Zarr/H5 data")
        Y = next(os_walk_clean(getattr(cfg.DATA, tag).GT_PATH))[2]
    del zarr_files, h5_files

    print("Creating Y_{} channels . . .".format(data_type))
    # Create the mask patch by patch (Zarr/H5)
    if working_with_zarr_h5_files and isinstance(Y, dict):
        if "D" in cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS:
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
                savepath=getattr(cfg.DATA, tag).INSTANCE_CHANNELS_MASK_DIR,
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
                    data_axes_order=getattr(cfg.DATA, tag).INPUT_IMG_AXES_ORDER,
                    data_path=getattr(cfg.DATA, tag).INPUT_ZARR_MULTIPLE_DATA_GT_PATH,
                )
                if img.ndim == 3:
                    img = np.expand_dims(img, -1)

                # Create the instance mask
                if cfg.DATA.N_CLASSES > 2:
                    if img.shape[-1] != 2:
                        raise ValueError(
                            "In instance segmentation, when 'DATA.N_CLASSES' are more than 2 labels need to have two channels, "
                            "e.g. (256,256,2), containing the instance segmentation map (first channel) and classification map (second channel)."
                        )
                    else:
                        class_channel = np.expand_dims(img[..., 1].copy(), -1)
                        img = labels_into_channels(
                            img,
                            mode=cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS,
                            channel_extra_opts=cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS_EXTRA_OPTS[0],
                            save_dir=getattr(cfg.PATHS, tag + "_INSTANCE_CHANNELS_CHECK"),
                        )
                        img = np.concatenate([
                            np.expand_dims(img,0), 
                            np.expand_dims(class_channel,0)
                        ], axis=-1)
                else:
                    img = labels_into_channels(
                        img,
                        mode=cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS,
                        channel_extra_opts=cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS_EXTRA_OPTS[0],
                        save_dir=getattr(cfg.PATHS, tag + "_INSTANCE_CHANNELS_CHECK"),
                    )

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
                    fname = os.path.join(getattr(cfg.DATA, tag).INSTANCE_CHANNELS_MASK_DIR, os.path.basename(Y[i]["filepath"]))
                    os.makedirs(getattr(cfg.DATA, tag).INSTANCE_CHANNELS_MASK_DIR, exist_ok=True)
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
        rank = get_rank()
        world_size = get_world_size()
        N = len(Y)
        it = range(rank, N, world_size)
        for i in tqdm(it, disable=not is_main_process()):
            img = read_img_as_ndarray(
                os.path.join(getattr(cfg.DATA, tag).GT_PATH, Y[i]),
                is_3d=not cfg.PROBLEM.NDIM == "2D",
            )
            if cfg.DATA.N_CLASSES > 2:
                if img.shape[-1] != 2:
                    raise ValueError(
                        "In instance segmentation, when 'DATA.N_CLASSES' are more than 2 labels need to have two channels, "
                        "e.g. (256,256,2), containing the instance segmentation map (first channel) and classification map "
                        "(second channel)."
                    )
                class_channel = np.expand_dims(img[..., 1].copy(), -1)

            img = labels_into_channels(
                img,
                mode=cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS,
                channel_extra_opts=cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS_EXTRA_OPTS[0],
                save_dir=getattr(cfg.PATHS, tag + "_INSTANCE_CHANNELS_CHECK"),
            )

            if cfg.DATA.N_CLASSES > 2:
                img = np.concatenate([img, class_channel], axis=-1)

            save_tif(
                np.expand_dims(img, 0),
                data_dir=getattr(cfg.DATA, tag).INSTANCE_CHANNELS_MASK_DIR,
                filenames=[Y[i]],
                verbose=False,
            )


def labels_into_channels(
    instance_labels: NDArray, 
    mode: List[str] = ["I", "C"], 
    channel_extra_opts: Dict = {},
    resolution: List[int|float] = [1,1,1], 
    save_dir: Optional[str] = None,
) -> NDArray:
    """
    Convert input semantic or instance segmentation data masks into different binary channels to train an instance segmentation problem.

    Parameters
    ----------
    instance_labels : 3D/4D Numpy array
        Instance labels to be used to extract the channels from. E.g. ``(200, 1000, 1000, 1)``

    mode : List, optional
        Operation mode. Possible values: ``C``, ``BC``, ``BCM``, ``BCD``, ``BD``, ``BCDv2``, ``Dv2``, ``BDv2`` and ``BP``.
         - 'B' stands for 'Binary segmentation', containing each instance region without the contour.
         - 'C' stands for 'Contour', containing each instance contour.
         - 'D' stands for 'Distance', each pixel containing the distance of it to the center of the object.
         - 'M' stands for 'Mask', contains the B and the C channels, i.e. the foreground mask.
           Is simply achieved by binarizing input instance masks.
         - 'Dv2' stands for 'Distance V2', which is an updated version of 'D' channel calculating background distance as well.
         - 'P' stands for 'Points' and contains the central points of an instance (as in Detection workflow)
         - 'A' stands for 'Affinities" and contains the affinity values for each dimension

    channel_extra_opts : dict, optional
        Additional options for each output channel (e.g., {"I": {"erosion": 1}}).

    resolution : Tuple of int/float
        Resolution of the data, in ``(z,y,x)`` to calibrate coordinates. E.g. ``[30,8,8]``.

    save_dir : str, optional
        Path to store samples of the created array just to debug it is correct.

    Returns
    -------
    new_mask : 3D/4D Numpy array
        Instance representations. The shape will be as the input ``instance_labels`` but with the amount of channels
        requested. E.g. ``(200, 1000, 1000, 3)``
    """
    assert len(resolution) == 3, "'resolution' must be a list of 3 int/float"
    assert instance_labels.ndim in [3, 4]
    c_number = 0
    for ch in mode:
        if ch == "R":
            nrays = channel_extra_opts["R"]["nrays"]
            c_number += nrays
        elif ch == "A":
            affs = (
                len(channel_extra_opts["A"]["z_affinities"])
                +len(channel_extra_opts["A"]["y_affinities"])
                +len(channel_extra_opts["A"]["x_affinities"])
            )
            c_number += affs
        elif ch in ["E_sigma", "E_seediness"]:
            continue  # not special channels, just extra targets for embeddings
        else:
            c_number += 1

    if any(x for x in ["Db", "Dc", "Dn", "D", "H", "V", "Z", "R", "We"] if x in mode):
        dtype = np.float32
    elif "E_offset" in mode:
        dtype = instance_labels.dtype
        # Ensure that no floating-point dtype is used for the embeddings. 
        # This allows the normalization module to correctly recognize them 
        # as integer or binary channels, ensuring that the subsequent 
        # data augmentation processes the samples as intended.
        if np.issubdtype(dtype, np.floating):
            if instance_labels.max() > 255:
                dtype = np.uint16
            else:
                dtype = np.uint8
    else:
        dtype = np.uint8
        
    new_mask = np.zeros(instance_labels.shape[:-1] + (c_number,), dtype=dtype)
    vol = instance_labels[..., 0]

    # Precompute regionprops only when needed
    needs_props = any(x in mode for x in ("H", "V", "Z"))
    if needs_props:
        # label, bbox, centroid (you didn't have intensity stats here)
        props_tbl = regionprops_table(vol, properties=("label", "bbox", "centroid"))
        # Convenience view as list of labels
        instances = [0] + list(props_tbl["label"])
        instance_count = len(instances)
    else:
        instances = list(np.unique(vol))
        instance_count = len(instances)
    instances = [inst for inst in instances if inst != 0] # remove background

    if instance_count <= 1:
        return new_mask  # only background
    
    fg_mask = (vol > 0).astype(np.uint8)
    bg_mask = (vol == 0).astype(np.uint8)

    # Precompute flow channels if any of H/V/Z is requested
    if any(ch in mode for ch in ("H", "V", "Z")):
        norm_flag = True
        for ch in ("H", "V", "Z"): 
            if ch in channel_extra_opts and "norm" in channel_extra_opts[ch]:
                norm_flag = bool(channel_extra_opts[ch]["norm"])
                break
        hv_channels = create_flow_channels(
            vol,
            ref_point="center",
            normalize_values=norm_flag,
            calc_props=props_tbl,
        )

    # ---------- Foreground (F) / Background (B) ----------
    for ch, mask_expr in (("F", fg_mask), ("B", bg_mask)):
        if ch in mode:
            # Check if erosion/dilation is requested as the process needs the original volume
            # to make it per-instance
            er_k = channel_extra_opts.get(ch, {}).get("erosion", 0)
            dil_k = channel_extra_opts.get(ch, {}).get("dilation", 0)            
            erode, dilate = False, False
            if (isinstance(er_k, int) and er_k > 0) or (isinstance(er_k, list) and any([x for x in er_k if x > 0])):
                erode = True
            if (isinstance(dil_k, int) and dil_k > 0) or (isinstance(dil_k, list) and any([x for x in dil_k if x > 0])):
                dilate = True
            if erode or dilate:
                mask = np.zeros_like(fg_mask, dtype=np.uint8)
                dil_k = [dil_k,]*mask.ndim if isinstance(dil_k, int) else dil_k
                dil_k = generate_ellipse_footprint(dil_k)
                er_k = [er_k,]*mask.ndim if isinstance(er_k, int) else er_k
                er_k = generate_ellipse_footprint(er_k)
                for lb in instances:
                    m = (vol == lb)
                    if not np.any(m):
                        continue
                    if dilate:
                        m = binary_dilation(m.astype(np.uint8), footprint=dil_k).astype(np.uint8)
                    if erode:
                        m = binary_erosion(m.astype(np.uint8), footprint=er_k).astype(np.uint8)
                    mask[m > 0] = lb
            else:
                mask = mask_expr.astype(np.uint8)

            new_mask[..., mode.index(ch)] = mask

    # ---------- P (central part) ----------
    if "P" in mode:
        p_opts = channel_extra_opts.get("P", {})
        p_type = p_opts.get("type", "centroid")
        p_dil  = p_opts.get("dilation", 1)
        p_ero  = p_opts.get("erosion", 1)

        p_out = np.zeros_like(fg_mask, dtype=np.uint8)
        if p_type == "skeleton":
            for lb in instances:
                m = (vol == lb)
                if not np.any(m):
                    continue
                sk = skeletonize(m.astype(np.uint8)).astype(np.uint8)
                p_out += sk
        else:
            com_list = center_of_mass(fg_mask, labels=vol, index=instances)
            # Mark each centroid (guard against rounding outside bounds)
            if p_out.ndim == 2:
                H, W = p_out.shape
                for cy, cx in com_list:
                    y = int(round(cy)); x = int(round(cx))
                    if 0 <= y < H and 0 <= x < W:
                        p_out[y, x] = 1
            elif p_out.ndim == 3:
                Z, Y, X = p_out.shape
                for cz, cy, cx in com_list:
                    z = int(round(cz)); y = int(round(cy)); x = int(round(cx))
                    if 0 <= z < Z and 0 <= y < Y and 0 <= x < X:
                        p_out[z, y, x] = 1
            else:
                raise ValueError(f"Unsupported ndim {p_out.ndim} for P[type='centroid']")

        # Optional dilation (in pixels / voxels)
        if (isinstance(p_dil, int) and p_dil > 0) or (isinstance(p_dil, list) and any([x for x in p_dil if x > 0])):
            p_dil = [p_dil,]*p_out.ndim if isinstance(p_dil, int) else p_dil
            p_out = binary_dilation(p_out, footprint=generate_ellipse_footprint(p_dil)).astype(np.uint8)
        # Optional erosion (in pixels / voxels)
        if (isinstance(p_ero, int) and p_ero > 0) or (isinstance(p_ero, list) and any([x for x in p_ero if x > 0])):
            p_ero = [p_ero,]*p_out.ndim if isinstance(p_ero, int) else p_ero
            p_out = binary_erosion(p_out, footprint=generate_ellipse_footprint(p_ero)).astype(np.uint8)

        # Write the channel
        new_mask[..., mode.index("P")] = p_out

    # ---------- C (contours) ----------
    if "C" in mode:
        c_mode = channel_extra_opts.get("C", {}).get("mode", "thick")
        if c_mode == "dense":
            # synthetic "dense" edges: dilate FG and XOR with FG to thicken borders on both sides
            fg = fg_mask
            if fg.ndim == 2:
                rim = binary_dilation(fg, disk(1)).astype(np.uint8) ^ fg
                new_mask[..., mode.index("C")] = rim
            else:
                out = np.zeros_like(fg)
                for j in range(fg.shape[0]):
                    out[j] = (binary_dilation(fg[j], disk(1)).astype(np.uint8) ^ fg[j])
                new_mask[..., mode.index("C")] = out
        else:
            # valid skimage modes: inner|outer|thick|subpixel
            new_mask[..., mode.index("C")] = find_boundaries(vol, mode=c_mode).astype(np.uint8)

    # ---------- Dc (distance to center/skeleton) ----------
    if "Dc" in mode:
        dc_type = channel_extra_opts.get("Dc", {}).get("type", "center")
        dc_channel = np.zeros_like(vol, dtype=np.float32)
        
        for lab in instances:  # skip background
            m = (vol == lab)
            if not np.any(m):
                continue

            # tight bbox to speed up ops
            idxs = np.where(m)
            if vol.ndim == 2:
                y0, y1 = idxs[0].min(), idxs[0].max() + 1
                x0, x1 = idxs[1].min(), idxs[1].max() + 1
                sub = m[y0:y1, x0:x1]

                if dc_type == "skeleton":
                    sk = skeletonize(sub.astype(np.uint8)).astype(bool)
                    dist_to_sk = edt.edt(~sk, anisotropy=resolution, parallel=-1)
                    dc_channel[y0:y1, x0:x1][sub] = dist_to_sk[sub]
                else:
                    # centroid in GLOBAL coords
                    ys, xs = np.where(sub)
                    cy = y0 + ys.mean()
                    cx = x0 + xs.mean()
                    # compute distances on bbox grid, then fill only inside the instance
                    yy, xx = np.ogrid[y0:y1, x0:x1]
                    dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)
                    dc_channel[y0:y1, x0:x1][sub] = dist[sub]

            else:  # vol.ndim == 3
                z0, z1 = idxs[0].min(), idxs[0].max() + 1
                y0, y1 = idxs[1].min(), idxs[1].max() + 1
                x0, x1 = idxs[2].min(), idxs[2].max() + 1
                sub = m[z0:z1, y0:y1, x0:x1]

                if dc_type == "skeleton":
                    sk = skeletonize(sub.astype(np.uint8)).astype(bool)
                    dist_to_sk = edt.edt(~sk, anisotropy=resolution, parallel=-1)
                    dc_channel[z0:z1, y0:y1, x0:x1][sub] = dist_to_sk[sub]
                else:
                    # centroid in GLOBAL coords
                    zs, ys, xs = np.where(sub)
                    cz = z0 + zs.mean()
                    cy = y0 + ys.mean()
                    cx = x0 + xs.mean()
                    zz, yy, xx = np.ogrid[z0:z1, y0:y1, x0:x1]
                    dist = np.sqrt((zz - cz)**2 + (yy - cy)**2 + (xx - cx)**2)
                    dc_channel[z0:z1, y0:y1, x0:x1][sub] = dist[sub]
        
        assert isinstance(dc_channel, np.ndarray), "Expected dc_channel to be a numpy array"
        # Normalization
        if channel_extra_opts.get("Dc", {}).get("norm", False):
            dc_channel = norm_channel(
                dc_channel, 
                vol, 
                instances,
            )
        new_mask[..., mode.index("Dc")] = dc_channel 

    # ---------- Db (distance to boundary) ----------
    if "Db" in mode:
        db_channel = edt.edt(vol, anisotropy=resolution, parallel=-1)
        assert isinstance(db_channel, np.ndarray), "Expected db to be a numpy array"

        # Normalization
        if channel_extra_opts.get("Db", {}).get("norm", False):
            db_channel = norm_channel(
                db_channel, 
                vol, 
                instances,
            )
        new_mask[..., mode.index("Db")] = db_channel

    # ---------- Dn (distance to neighbor) ----------
    if "Dn" in mode:
        dn_opts = channel_extra_opts.get("Dn", {})
        dn_norm = bool(dn_opts.get("norm", False))
        power = float(dn_opts.get("decline_power", 3.0)) 
        closing_size  = int(dn_opts.get("closing_size", 3))  # neighborhood size for grey closing

        dn_channel = np.zeros_like(vol, dtype=np.float32)
        
        # Mask to remember which cell pixels belong to cells that have at least one OTHER instance
        has_neighbor_px = np.zeros_like(vol, dtype=bool)

        for lab in instances: 
            cur = (vol == lab)
            if not np.any(cur):
                continue

            other_instances = fg_mask & (~cur)
            if not np.any(other_instances):
                # No other labeled object anywhere -> keep this cell at 0 (suppressed)
                continue

            has_neighbor_px[cur] = True

            # Per paper: (selected cell ∪ background) = 1; "other cells" = 0  (distance to other cells)
            fg = (bg_mask | cur)
            d = edt.edt(fg, anisotropy=resolution, parallel=-1).astype(np.float32)

            if dn_norm:
                # Paper path: cut->normalize [0,1]->invert (1 - ..)
                d_cell = d[cur].copy()
                m = d_cell.max()
                if m > 0:
                    d_cell /= m
                else:
                    d_cell.fill(0.0)
                dn_channel[cur] = 1.0 - d_cell
            else:
                # Store raw distances for now (we'll handle per-image norm or unnormalized inversion later)
                dn_channel[cur] = d[cur]

        invert_mask = has_neighbor_px  # operate only on cells that actually have neighbors

        # --- Unnormalized but still inverted ---
        if not dn_norm:
            if np.any(invert_mask):
                M = float(dn_channel[invert_mask].max())
                if M > 0:
                    dn_channel[invert_mask] = M - dn_channel[invert_mask]
            # background & isolated cells remain 0

        # Grayscale closing after merging & inversion
        if closing_size > 0:
            size = (closing_size,) * dn_channel.ndim
            dn_channel = grey_closing(dn_channel, size=size)

        if power is not None and power != 1.0:
            if dn_norm :
                # values in [0,1] already → direct power
                dn_channel = np.power(dn_channel, power, dtype=np.float32)
            else:
                # unnormalized: temporarily normalize on the meaningful support, power, then unnormalize
                if np.any(invert_mask):
                    M2 = float(dn_channel[invert_mask].max())
                    if M2 > 0:
                        tmp = dn_channel[invert_mask] / M2
                        tmp = np.power(tmp, power, dtype=np.float32)
                        dn_channel[invert_mask] = tmp * M2

        new_mask[..., mode.index("Dn")] = dn_channel.astype(np.float32)
        
    # ---------- D (signed distance, global) ----------
    if "D" in mode:
        alpha = channel_extra_opts.get("D", {}).get("alpha", 1.0)
        beta = channel_extra_opts.get("D", {}).get("beta", 1.0)

        # 1) Signed distance
        sdist = edt.edt(fg_mask, anisotropy=resolution, parallel=-1)/alpha - edt.edt(bg_mask, anisotropy=resolution, parallel=-1)/beta
        assert isinstance(sdist, np.ndarray), "Expected sdist to be a numpy array"

        # 2) Map GT to [-1, 1] with tanh (COSEM-style: "Whole-cell organelle segmentation in volume electron microscopy")
        tanh_on = channel_extra_opts.get("D", {}).get("norm", True)
        if tanh_on:
            sdist = np.tanh(sdist)
            
        new_mask[..., mode.index("D")] = sdist
        
    # ---------- H / V / Z (flow-like channels) ----------
    if "Z" in mode:
        new_mask[..., mode.index("Z")] = hv_channels[...,0]
    if "V" in mode:
        ch_pos = 0 if new_mask[..., mode.index("V")].ndim == 2 else 1
        new_mask[..., mode.index("V")] = hv_channels[...,ch_pos]
    if "H" in mode:
        ch_pos = 1 if new_mask[..., mode.index("H")].ndim == 2 else 2
        new_mask[..., mode.index("H")] = hv_channels[...,ch_pos]

    # ---------- T (touching area) ----------
    if "T" in mode:
        new_mask[..., mode.index("T")] = touching_mask_nd(
            vol,
            connectivity=new_mask[..., mode.index("T")].ndim
        )

    # ---------- A (affinities) ----------
    if "A" in mode:
        ins_vol = vol
        wb = int(channel_extra_opts["A"].get("widen_borders", 1))
        if wb:
            ins_vol = seg_widen_border(vol, tsz_h=wb)
            
        k = 0
        for zaff, yaff, xaff in zip(
            channel_extra_opts["A"].get("z_affinities", []),
            channel_extra_opts["A"].get("y_affinities", []),
            channel_extra_opts["A"].get("x_affinities", []),
        ):
            affs = seg2aff_pni(ins_vol, dz=zaff, dy=yaff, dx=xaff, dtype=dtype)  # shape: (n_affs, Z, Y, X)
            affs = np.transpose(affs, (1, 2, 3, 0))  # shape: (Z, Y, X, n_affs)
            new_mask[..., k*3:(k+1)*3] = affs
            k += 1

    # ---------- R (radial distances) ----------
    if "R" in mode:
        r_opts = channel_extra_opts.get("R", {})
        ndim = 2 if new_mask[..., mode.index("R")].ndim == 2 else 3
        nrays = int(r_opts.get("nrays", 32 if ndim == 2 else 96))

        rays = generate_rays(n_rays=nrays, ndim=ndim).astype(np.float32)
        spacing = None if new_mask[..., mode.index("R")].ndim == 2 else resolution

        new_mask[..., mode.index("R"):mode.index("R")+nrays] = radial_distances(vol, rays, spacing=spacing)

    # ---------- E (Embeddings) ----------
    # Here we only use E_offset as extra target for the embeddings branch
    if "E_offset" in mode: 
        new_mask[..., mode.index("E_offset")] = vol.copy()

    if "We" in mode:
        new_mask[..., mode.index("We")] =  unet_border_weight_map(vol, w0=10.0, sigma=5.0, resolution=resolution)

    # ---------- M (Legacy mask used in CartoCell) ----------
    if "M" in mode: 
        # Binary mask = F + C
        f_ch = new_mask[..., mode.index("F")]
        c_ch = new_mask[..., mode.index("C")]
        new_mask[..., mode.index("M")] = np.clip(f_ch + c_ch, 0, 1).astype(np.uint8)

    # Save examples of each channel
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for j, mod in enumerate(mode):
            if mod == "B":
                suffix = "_background.tif"
            elif mod == "F":
                suffix = "_foreground.tif"
            elif mod == "P":
                suffix = "_central_part.tif"
            elif mod == "C":
                suffix = "_contour.tif"
            elif mod == "H":
                suffix = "_horizontal_distance.tif"
            elif mod == "V":
                suffix = "_vertical_distance.tif"
            elif mod == "Z":
                suffix = "_z_distance.tif"
            elif mod == "Db":
                suffix = "_distance_to_border.tif"
            elif mod == "Dc":
                suffix = "_distance_to_center.tif"
            elif mod == "Dn":
                suffix = "_distance_to_neighbor.tif"
            elif mod == "D":
                suffix = "_distance.tif"
            elif mod == "R":
                suffix = "_radial_distances.tif"
            elif mod == "T":
                suffix = "_touching.tif"
            elif mod == "A":
                suffix = "_affinity.tif"
            elif mod == "E_offset":
                suffix = "_embedding_instances.tif"
            elif mod in ["E_sigma", "E_seediness"]:
                continue  
            elif mod == "We":
                suffix = "_border_weights.tif"
            elif mod == "M":
                suffix = "_CartoCell_M_channel.tif"
            else:
                raise ValueError("Unknown channel type: {}".format(mod))

            aux = new_mask[..., j]
            aux = np.expand_dims(np.expand_dims(aux, -1), 0)
            save_tif(aux, save_dir, filenames=["vol" + suffix[j]], verbose=False)
            save_tif(
                np.expand_dims(instance_labels, 0),
                save_dir,
                filenames=["vol" + "_y.tif"],
                verbose=False,
            )
    return new_mask


def norm_channel(channel: NDArray, vol: NDArray, instances: list[int]) -> NDArray:
    """
    Normalize a channel based on instance masks.

    Parameters
    ----------
    channel : NDArray
        The channel to normalize (e.g. db_channel).
    vol : NDArray
        Instance mask volume, same shape as channel.
    instances : list[int]
        List of instance IDs in `vol`. Background (0) will be ignored.

    Returns
    -------
    NDArray
        Normalized channel, same shape as input.
    """
    instances = [inst for inst in instances if inst != 0]  # drop background
    normed = np.zeros_like(channel, dtype=np.float32)

    for inst in instances:
        mask = (vol == inst)
        if not np.any(mask):
            continue

        values = channel[mask]
        mi, ma = values.min(), values.max()

        # Avoid division by zero
        if ma == mi:
            normed[mask] = 0
        else:
            normed[mask] = (values - mi) / (ma - mi)

    return normed

def unet_border_weight_map(
    instances: np.ndarray,
    w0: float = 10.0,
    sigma: float = 5.0,
    apply_only_background: bool = True,
    resolution: List[int|float] | None = None,
) -> np.ndarray:
    """
    U-Net border-aware weight map (Ronneberger et al. 2015) for 2D or 3D labels.

    Parameters
    ----------
    instances : np.ndarray, shape (H, W) or (D, H, W), dtype int
        0/`background` for background, 1..N (or any ints != background) are instance ids.
    
    w0 : float
        Border weight magnitude.
    
    sigma : float
        Spatial decay (in same units as resolution).

    apply_only_background : bool
        If True, apply the exponential term only on background (as in the paper).

    resolution : List[int|float] | None
        Voxel spacing along each axis (z,y,x) or (y,x). If None, isotropic spacing of 1 is assumed.

    Returns
    -------
    w : np.ndarray, same shape as `instances`, dtype float32
        Border weight map.
    """
    if instances.ndim not in (2, 3):
        raise ValueError(f"`instances` must be 2D or 3D, got shape {instances.shape}")

    inst = instances.astype(np.int32, copy=False)
    shp = inst.shape

    # collect unique instance ids excluding background
    ids = np.unique(inst)
    ids = ids[ids != 0]

    # Need at least two distinct instances for the (d1 + d2) term to be meaningful
    if ids.size < 2:
        return np.zeros(shp, dtype=np.float32)

    # Compute distance-to-each-instance via EDT on the complement of that instance
    # distances[k, ...] = distance to instance ids[k]
    distances = np.empty((ids.size, *shp), dtype=np.float32)
    for k, lab in enumerate(ids):
        # edt computes distance to zeros -> pass mask that's zero *inside* the object
        # equivalently: distance to the boundary of object `lab`
        distances[k] = edt.edt(inst != lab, anisotropy=resolution, parallel=-1)

    # nearest and second-nearest distances at each voxel/pixel
    d1 = distances.min(axis=0)
    d2 = np.partition(distances, 1, axis=0)[1]

    # Border emphasis term
    denom = 2.0 * (sigma ** 2)
    w_border = w0 * np.exp(-((d1 + d2) ** 2) / denom, dtype=np.float64)
    w_border = w_border.astype(np.float32, copy=False)

    if apply_only_background:
        w_border *= (inst == 0)

    return w_border


def touching_mask_nd(labels: NDArray, connectivity: int = 1) -> NDArray:
    """
    Create a binary mask of touching pixels/voxels for an N-D labeled instance mask.

    Parameters
    ----------
    labels : NDArray
        N-D array of instance labels (0 = background, 1..N = instances).

    connectivity : int, optional
        Neighborhood connectivity passed to `generate_binary_structure`.
        1 = 6-neigh for 3D / 4-neigh for 2D, 
        2 = 18-neigh for 3D / 8-neigh for 2D,
        3 = 26-neigh for 3D (if ndim==3).

    Returns
    -------
    touch : NDArray
        Binary mask with 1 where a voxel touches at least one *different* instance.
    """
    # Neighborhood footprint including the center
    footprint = generate_binary_structure(labels.ndim, connectivity)

    def is_touching(window):
        center = window[len(window)//2]
        if center == 0:  # background is never touching
            return 0
        # unique neighbor labels (including center); drop 0 and center label
        uniq = np.unique(window)
        return 1 if np.any((uniq != 0) & (uniq != center)) else 0

    touch = generic_filter(
        labels,
        is_touching,
        footprint=footprint,
        mode='constant',
        cval=0
    )
    return touch.astype(np.uint8)

def generate_rays(n_rays: int, ndim: int, jitter: bool=False, seed: int=0):
    """
    Unit directions in R^ndim.
    - 2D: uniform angles on circle -> (R,2) [dx,dy]
    - 3D: Fibonacci sphere -> (R,3) [dx,dy,dz]

    Parameters
    ----------
    n_rays : int
        Number of rays to generate.
    ndim : int
        Dimensionality (2 or 3).
    jitter : bool, optional
        Whether to add jitter to 3D rays (default: False).
    seed : int, optional
        Random seed for jitter (default: 0).    

    Returns
    -------
    rays : (n_rays, 2) or (n_rays, 3) Numpy array
        Unit vectors along which to compute distances.
    """
    if ndim == 2:
        a = np.linspace(0, 2*np.pi, n_rays, endpoint=False, dtype=np.float32)
        return np.stack([np.cos(a), np.sin(a)], axis=1).astype(np.float32)
    elif ndim == 3:
        rng = np.random.default_rng(seed) if jitter else None
        i = np.arange(n_rays, dtype=np.float32)
        phi = (1 + np.sqrt(5.0)) / 2.0
        z = 1 - 2*(i + 0.5) / n_rays
        r = np.sqrt(np.maximum(0.0, 1 - z*z))
        theta = 2*np.pi*i/phi
        if jitter:
            theta += rng.uniform(-np.pi/n_rays, np.pi/n_rays, size=n_rays)
            z += rng.uniform(-1/n_rays, 1/n_rays, size=n_rays)
            z = np.clip(z, -1.0, 1.0); r = np.sqrt(np.maximum(0.0, 1 - z*z))
        x = r * np.cos(theta); y = r * np.sin(theta)
        dirs = np.stack([x, y, z], axis=1).astype(np.float32)
        dirs /= (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
        return dirs
    else:
        raise ValueError("Only 2D and 3D are supported.")


def radial_distances(
    labels: NDArray,
    rays: NDArray,
    max_dist: Optional[float] = None,
    spacing: Optional[Sequence[float]] = None,
    max_iters: int = 50,
) -> NDArray:
    """
    Compute radial distances from each foreground pixel to the instance boundary along specified rays.

    Parameters
    ----------
    labels : NDArray
        2D or 3D array of instance labels (0 = background, 1..N = instances).
    rays : (n_rays, 2) or (n_rays, 3) Numpy array
        Unit vectors along which to compute distances.
    max_dist : float, optional
        Maximum distance to cap at. If None, no capping is done.
    spacing : sequence of float, optional
        Physical spacing of the data in each dimension. If None, assumes isotropic spacing of 1.0.
    max_iters : int
        Maximum number of steps to march along each ray.

    Returns
    -------
    D : NDArray
        Array of shape (H, W, n_rays) or (D, H, W, n_rays) with distances in physical units.
        Background pixels have distance 0 in all rays.
    """
    labels = np.asarray(labels)
    ndim = labels.ndim
    assert rays.ndim == 2 and rays.shape[1] == ndim
    spacing = np.ones(ndim, np.float32) if spacing is None else np.asarray(spacing, np.float32)
    shape = labels.shape
    n_rays = rays.shape[0]

    # normalize rays in index space (row/col[/z] units)
    rays_idx = rays.astype(np.float32)
    norms = np.linalg.norm(rays_idx, axis=1, keepdims=True) + 1e-12
    rays_idx /= norms

    # per-ray physical step length for one unit in index space
    ray_step_phys = np.linalg.norm(rays_idx * spacing, axis=1)  # shape (n_rays,)

    D = np.zeros(shape + (n_rays,), np.float32)

    fg = np.argwhere(labels > 0)
    H, W = shape[0], shape[1] if ndim == 2 else (shape[0], shape[1])  # for bounds
    for (i, j, *rest) in fg:
        inst_id = int(labels[i, j]) if ndim == 2 else int(labels[i, j, rest[0]])
        p0 = np.array([i, j] if ndim == 2 else [i, j, rest[0]], np.float32)  # pixel center reference

        for k in range(n_rays):
            u = rays_idx[k]  # unit in index space
            x = np.zeros(ndim, np.float32)  # accumulated offset in index space

            # march in unit steps like the ref (||u||=1)
            for _ in range(max_iters * (max(shape) + 2)):  # safe cap
                x += u
                p_samp = p0 + x
                # rounded sampling
                if ndim == 2:
                    ii = int(np.rint(p_samp[0])); jj = int(np.rint(p_samp[1]))
                    out = (ii < 0 or ii >= shape[0] or jj < 0 or jj >= shape[1])
                    changed = (not out) and (labels[ii, jj] != inst_id)
                else:
                    ii = int(np.rint(p_samp[0])); jj = int(np.rint(p_samp[1])); kk = int(np.rint(p_samp[2]))
                    out = (ii < 0 or ii >= shape[0] or jj < 0 or jj >= shape[1] or kk < 0 or kk >= shape[2])
                    changed = (not out) and (labels[ii, jj, kk] != inst_id)

                if out or changed:
                    max_comp = np.max(np.abs(u)) + 1e-12
                    t_corr = 1.0 - 0.5 / max_comp
                    x = x - t_corr * u   # pull back along dominant axis
                    # distance in pixels
                    dist_idx = float(np.linalg.norm(x))
                    # convert to physical units if requested
                    dist = dist_idx * float(ray_step_phys[k])
                    if max_dist is not None and dist > max_dist:
                        dist = max_dist
                    if ndim == 2:
                        D[i, j, k] = dist
                    else:
                        D[i, j, rest[0], k] = dist
                    break

    return D


def euler_integration(flow: NDArray, coords: NDArray, n_steps: int = 200, dt: float = 1.0, suppressed: bool = True):
    """
    Euler integration of flow field starting at coords.
    
    Parameters
    ----------
    flow : (2, H, W) or (3, D, H, W) Numpy array
        Flow field (y,x) or (z,y,x).
    coords : (N, 2) or (N, 3) Numpy array
        Starting coordinates (y,x) or (z,y,x) in index space.
    n_steps : int
        Number of integration steps.
    dt : float
        Integration step size.
    suppressed : bool
        Whether to use time-suppressed integration (dt/(t+1)) or not (constant dt). 

    Returns
    -------
    pos : (N, 2) or (N, 3) Numpy array
        Final positions after integration.
    """
    pos = coords.astype(float).copy()
    H, W = flow.shape[1:]

    for t in range(n_steps):
        # Interpolate flow at current positions
        fy = map_coordinates(flow[0], [pos[:,0], pos[:,1]], order=1, mode='nearest')
        fx = map_coordinates(flow[1], [pos[:,0], pos[:,1]], order=1, mode='nearest')
        step = np.stack([fy, fx], axis=1)

        # suppression factor
        factor = dt / (t+1) if suppressed else dt

        pos += factor * step

        # keep inside bounds
        pos[:,0] = np.clip(pos[:,0], 0, H-1)
        pos[:,1] = np.clip(pos[:,1], 0, W-1)

    return pos  # final positions for clustering


def synapse_channel_creation(
    data_info: Dict,
    zarr_data_information: Dict,
    savepath: str,
    mode: List[str] = ["F_pre", "F"],
    postsite_dilation: List[int] = [2, 4, 4],
    postsite_distance_channel_dilation: List[int] = [3, 10, 10],
    normalize_values: bool = False,
):
    """
    Create different channels that represent a synapse segmentation problem to train an instance segmentation problem.
    
    This function is only prepared to read an H5/Zarr file that follows `CREMI data format <https://cremi.org/data/>`__.

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

    mode : List, optional
        Operation mode. 

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
    channels = len(mode)
    F_pre_pos = mode.index("F_pre")
    F_post_pos = 1 if F_pre_pos == 0 else 1 

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
            if insert_pre and insert_post:
                if pre_key not in pre_post_points:
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
            if "C" not in zarr_data_information["axes_order"]:
                out_data_shape = tuple(out_data_shape) + (channels,)
                out_data_order = zarr_data_information["axes_order"] + "C"
                c_axe_pos = -1
            else:
                out_data_shape[zarr_data_information["axes_order"].index("C")] = channels
                out_data_shape = tuple(out_data_shape)
                out_data_order = zarr_data_information["axes_order"]
                c_axe_pos = zarr_data_information["axes_order"].index("C")

            if any(fname.endswith(x) for x in [".h5", ".hdf5", ".hdf"]):
                mask = fid_mask.create_dataset("data", shape=out_data_shape, dtype=dtype_str)
                # mask = fid_mask.create_dataset("data", out_data_shape, compression="lzf", dtype=dtype_str)
            else:
                mask = fid_mask.create_dataset("data", shape=out_data_shape, dtype=dtype_str)

            print("Paiting all postsynaptic sites")
            width_reference = dilation_width
            for pre_site, post_sites in tqdm(pre_post_points.items(), disable=not is_main_process()):
                pre_point_global = [int(float(x)) for x in " ".join(pre_site[1:-1].split()).split(" ")]

                # Do not draw those pre that do not have post associated as they may be errors
                if len(post_sites) == 0:
                    continue
                ref_point = post_sites[0]
                patch_coords = [
                    max(0, ref_point[0] - width_reference[0]),
                    min(out_data_shape[zarr_data_information["z_axe_pos"]], ref_point[0] + width_reference[0]),
                    max(0, ref_point[1] - width_reference[1]),
                    min(out_data_shape[zarr_data_information["y_axe_pos"]], ref_point[1] + width_reference[1]),
                    max(0, ref_point[2] - width_reference[2]),
                    min(out_data_shape[zarr_data_information["x_axe_pos"]], ref_point[2] + width_reference[2]),
                ]

                # Take into account the pre point too
                if set(mode) == {"F_pre", "F_post"}:
                    ref_point = pre_point_global
                    patch_coords = [
                        min(max(0, ref_point[0] - width_reference[0]), patch_coords[0]),
                        max(
                            min(
                                out_data_shape[zarr_data_information["z_axe_pos"]],
                                ref_point[0] + width_reference[0],
                            ),
                            patch_coords[1],
                        ),
                        min(max(0, ref_point[1] - width_reference[1]), patch_coords[2]),
                        max(
                            min(
                                out_data_shape[zarr_data_information["y_axe_pos"]],
                                ref_point[1] + width_reference[1],
                            ),
                            patch_coords[3],
                        ),
                        min(max(0, ref_point[2] - width_reference[2]), patch_coords[4]),
                        max(
                            min(
                                out_data_shape[zarr_data_information["x_axe_pos"]],
                                ref_point[2] + width_reference[2],
                            ),
                            patch_coords[5],
                        ),
                    ]

                # Take the patch to extract so to draw all the postsynaptic sites using the minimun patch size
                for post_point in post_sites:
                    ref_point = post_point
                    patch_coords = [
                        min(max(0, ref_point[0] - width_reference[0]), patch_coords[0]),
                        max(
                            min(
                                out_data_shape[zarr_data_information["z_axe_pos"]],
                                ref_point[0] + width_reference[0],
                            ),
                            patch_coords[1],
                        ),
                        min(max(0, ref_point[1] - width_reference[1]), patch_coords[2]),
                        max(
                            min(
                                out_data_shape[zarr_data_information["y_axe_pos"]],
                                ref_point[1] + width_reference[1],
                            ),
                            patch_coords[3],
                        ),
                        min(max(0, ref_point[2] - width_reference[2]), patch_coords[4]),
                        max(
                            min(
                                out_data_shape[zarr_data_information["x_axe_pos"]],
                                ref_point[2] + width_reference[2],
                            ),
                            patch_coords[5],
                        ),
                    ]

                patch_coords = [int(x) for x in patch_coords]
                patch_shape = (
                    patch_coords[1] - patch_coords[0],
                    patch_coords[3] - patch_coords[2],
                    patch_coords[5] - patch_coords[4],
                )

                # Prepare the slices to be used when inserting the data into the generated Zarr/h5 file
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

                pre_point = [
                    int(pre_point_global[0] - patch_coords[0]),
                    int(pre_point_global[1] - patch_coords[2]),
                    int(pre_point_global[2] - patch_coords[4]),
                ]

                # Paiting each post-synaptic site
                if set(mode) == {"F_pre", "F"}:
                    seeds = np.zeros(patch_shape, dtype=np.uint64)
                    mask_to_grow = np.zeros(patch_shape, dtype=np.uint8)
                    label_to_pre_site = {}
                    label_count = 1
                    for post_point_global in post_sites:
                        post_point = [
                            int(post_point_global[0] - patch_coords[0]),
                            int(post_point_global[1] - patch_coords[2]),
                            int(post_point_global[2] - patch_coords[4]),
                        ]

                        if (
                            post_point[0] < seeds.shape[0]
                            and post_point[1] < seeds.shape[1]
                            and post_point[2] < seeds.shape[2]
                        ):
                            seeds[
                                max(0, post_point[0] - width_reference[0]) : min(
                                    post_point[0] + width_reference[0], seeds.shape[0]
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
                        semantic = edt.edt(mask_to_grow[z], anisotropy=resolution, parallel=-1)
                        assert isinstance(semantic, np.ndarray)
                        seeds[z] = watershed(-semantic, seeds[z], mask=mask_to_grow[z])

                    # Flow channel creation
                    out_map = create_flow_channels(
                        seeds,
                        ref_point="presynaptic",
                        label_to_pre_site=label_to_pre_site,
                        normalize_values=normalize_values,
                    )

                    if F_pre_pos == 0:
                        out_map = np.concatenate([np.expand_dims(channel_0, -1), out_map], axis=-1)
                    else:
                        out_map = np.concatenate([out_map, np.expand_dims(channel_0, -1)], axis=-1)
                    del channel_0
                else:
                    out_map = np.zeros(patch_shape + (channels,), dtype=np.uint8)

                    # Pre point
                    out_map[
                        max(0, pre_point[0] - 1) : min(pre_point[0] + 1, out_map.shape[0]),
                        pre_point[1],
                        pre_point[2],
                        F_pre_pos,
                    ] = 1

                    # Paint the pre sites in channel 0 and post sites in channel 1
                    for post_point_global in post_sites:
                        post_point = [
                            int(post_point_global[0] - patch_coords[0]),
                            int(post_point_global[1] - patch_coords[2]),
                            int(post_point_global[2] - patch_coords[4]),
                        ]

                        if (
                            post_point[0] < out_map.shape[0]
                            and post_point[1] < out_map.shape[1]
                            and post_point[2] < out_map.shape[2]
                        ):
                            # Post point
                            out_map[
                                max(0, post_point[0] - 1) : min(post_point[0] + 1, out_map.shape[0]),
                                post_point[1],
                                post_point[2],
                                F_post_pos,
                            ] = 1
                        else:
                            raise ValueError(
                                "Point {} seems to be out of shape: {}".format(
                                    [post_point[0], post_point[1], post_point[2]], out_map.shape
                                )
                            )
                    for c in range(out_map.shape[-1]):
                        out_map[..., c] = binary_dilation_scipy(
                            out_map[..., c],
                            iterations=1,
                            structure=ellipse_footprint_cpd,
                        )

                # Adjust patch slice to transpose it before inserting intop the final data
                current_order = np.array(range(len(out_map.shape)))
                transpose_order = order_dimensions(
                    current_order,
                    input_order="ZYXC",
                    output_order=out_data_order,
                    default_value=np.nan,
                )
                transpose_order = [x for x in np.array(transpose_order) if not np.isnan(x)]

                # Place the patch into the Zarr
                mask[data_ordered_slices] += out_map.transpose(transpose_order) * (mask[data_ordered_slices] == 0)

            # Close file
            if isinstance(fid_mask, h5py.File):
                fid_mask.close()


def create_flow_channels(
    data: NDArray, 
    ref_point: str = "center", 
    label_to_pre_site: Optional[Dict] = None, 
    normalize_values: bool = True,
    calc_props: Optional[Dict] = None,
):
    """
    Obtain the horizontal and vertical distance maps for each instance.
    
    Depth distance is also calculated if the ``data`` provided is 3D.

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

    calc_props : dict, optional
        If region properties have already been calculated, they can be provided here to avoid recalculation.

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

    if calc_props is None:
        props = regionprops_table(orig_data, properties=("label", "bbox", "centroid"))
    else:
        props = calc_props

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
    distances : NDArray
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


def create_detection_masks(cfg: CN, data_type: str = "train"):
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
    img_ids = next(os_walk_clean(img_dir))[2]
    working_with_chunked_data = False
    if len(img_ids) == 0:
        img_ids = next(os_walk_clean(img_dir))[1]
        working_with_chunked_data = True
    if len(img_ids) == 0:
        raise ValueError(f"No data found in folder {img_dir}")
    img_ext = "." + img_ids[0].split(".")[-1]
    if working_with_chunked_data and img_ext not in [".n5", ".zarr"]:
        raise ValueError(f"No data found in folder {img_dir}")
    ids = next(os_walk_clean(label_dir))[2]

    channels = 2 if cfg.DATA.N_CLASSES > 2 else 1
    dtype = np.uint8 if cfg.DATA.N_CLASSES < 255 else np.uint16
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
            if img_ext not in [".zarr", ".n5"]:
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
                if cfg.DATA.N_CLASSES > 2:
                    raise ValueError("DATA.N_CLASSES > 2 but no class specified in CSV file")
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

                if c_point > cfg.DATA.N_CLASSES:
                    raise ValueError(
                        "Class {} detected while 'DATA.N_CLASSES' was set to {}. Please check it!".format(
                            c_point, cfg.DATA.N_CLASSES
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
                        label(clear_border(mask[..., ch])),  # type: ignore
                        return_counts=True,
                        return_index=True,
                    )  # type: ignore
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
                    crop_shape=cfg.DATA.PATCH_SIZE,
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
def create_ssl_source_data_masks(cfg: CN, data_type: str = "train"):
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
    ids = next(os_walk_clean(img_dir))[2]
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
    add_noise: bool = True,
    noise_level: Optional[float] = None,
    Down_up: bool = True,
):
    """
    Crappify input image by adding Gaussian noise and downsampling and upsampling it so the resolution gets worsen.

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


def add_gaussian_noise(image: NDArray, percentage_of_noise: float) -> NDArray:
    """
    Add Gaussian noise to an input image.

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
    Y: BiaPyDataset, is_3d: bool = False, w_foreground: float = 0.94, w_background: float = 0.06, save_dir=None
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


def resize_images(images: List[NDArray], **kwards) -> List[NDArray]:
    """
    Resize all the images using the specified parameters or default values if not provided.

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


def apply_gaussian_blur(images: List[NDArray], **kwards) -> List[NDArray]:
    """
    Apply a Gaussian blur to all images.

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


def apply_median_blur(images: List[NDArray], **kwards) -> List[NDArray]:
    """
    Apply a median blur filter to all images.

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


def detect_edges(images: List[NDArray], **kwards) -> List[NDArray]:
    """
    Detect edges in the given images using the Canny edge detection algorithm.
    
    The function `detect_edges` takes the 2D images as input, converts it to grayscale if necessary, and applies the Canny edge detection algorithm to detect edges in the image.

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


def _histogram_matching(source_imgs: List[NDArray], target_imgs: List[NDArray]) -> List[NDArray]:
    """
    Apply histogram matching to a set of source images based on the mean histogram of target images.
    
    Given a set of target images, it will obtain their mean histogram
    and applies histogram matching to all images from source images.

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
    # Concatenate all target images to compute the reference histogram
    target_concat = np.concatenate([img.ravel() for img in target_imgs])

    # Get the data type from the first image (assuming all have same dtype)
    dtype = target_imgs[0].dtype

    hist_mean, _ = np.histogram(target_concat, bins=np.arange(np.iinfo(dtype).max + 2))  # +2 because bins are edges

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


def apply_histogram_matching(images: List[NDArray], reference_path: str, is_2d: bool):
    """
    Apply histogram matching to a list of images based on the histogram of reference images.

    The function returns the images with their histogram matched to the histogram of the reference images, loaded from the given ``reference_path``.

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
    matched_images = _histogram_matching(images, references)
    return matched_images


def apply_clahe(images: List[NDArray], **kwards) -> List[NDArray]:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to a list of images.

    The function applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to an image and returns the result.

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
    cfg: CN, x_data: List[NDArray] = [], y_data: List[NDArray] = [], is_2d: bool = True, is_y_mask: bool = False
) -> List[NDArray] | Tuple[List[NDArray], List[NDArray]]:
    """
    Pre-process data by applying various image processing techniques.

    Parameters
    ----------
    cfg: dict
        The `cfg` parameter is a configuration object that contains various settings for preprocessing the data. 
        It is used to control the behavior of different preprocessing techniques such as image resizing, blurring, 
        histogram matching, etc.

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
