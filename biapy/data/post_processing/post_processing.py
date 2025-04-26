import os
import math
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import fill_voids
import edt
import zarr
from tqdm import tqdm
from scipy.signal import find_peaks
from scipy.spatial import cKDTree # type: ignore
from scipy.spatial.distance import cdist
from scipy.ndimage import rotate, grey_dilation, binary_erosion, binary_dilation, median_filter, center_of_mass
from scipy.signal import savgol_filter
from skimage import morphology
from skimage.morphology import disk, ball, remove_small_objects, dilation, erosion
from skimage.segmentation import watershed, relabel_sequential
from skimage.filters import rank, threshold_otsu
from skimage.measure import label, regionprops_table
from skimage.exposure import equalize_adapthist
from skimage.feature import peak_local_max, blob_log
from scipy.ndimage import binary_dilation as binary_dilation_scipy

import diplib as dip
from typing import (
    Tuple,
    Optional,
    Dict,
    List,
    Callable,
    Type
)
from numpy.typing import NDArray

from biapy.utils.misc import to_numpy_format, to_pytorch_format
from biapy.data.data_manipulation import read_img_as_ndarray, save_tif, imread, reduce_dtype
from biapy.data.pre_processing import generate_ellipse_footprint

def watershed_by_channels(
    data: NDArray,
    channels: str,
    ths: Dict={},
    remove_before: bool=False,
    thres_small_before: int=10,
    seed_morph_sequence: List[str]=[],
    seed_morph_radius: List[int]=[],
    erode_and_dilate_foreground: bool=False,
    fore_erosion_radius: int=5,
    fore_dilation_radius: int=5,
    rmv_close_points: bool=False,
    remove_close_points_radius: int=-1,
    resolution: List[float|int]=[1., 1., 1.],
    watershed_by_2d_slices: bool=False,
    save_dir: Optional[str]=None,
):
    """
    Convert binary foreground probability maps and instance contours to instance masks via watershed segmentation
    algorithm.

    Implementation based on `PyTorch Connectomics' process.py
    <https://github.com/zudi-lin/pytorch_connectomics/blob/master/connectomics/utils/process.py>`_.

    Parameters
    ----------
    data : 4D Numpy array
        Binary foreground labels and contours data to apply watershed into. E.g. ``(397, 1450, 2000, 2)``.

    channels : str
        Channel type used. Possible options: ``A``, ``C``, ``BC``, ``BCM``, ``BCD``, ``BCDv2``, ``Dv2`` and ``BDv2``.

    ths : float, optional
        Thresholds to be used on each channel. ``TH_BINARY_MASK`` used in the semantic mask to create watershed seeds;
        ``TH_CONTOUR`` used in the contours to create watershed seeds; ``TH_FOREGROUND`` used in the semantic mask to create the
        foreground mask; ``TH_POINTS`` used in the point mask to create watershed seeds; ``TH_DISTANCE`` used in the
        distances to create watershed seeds.

    remove_before : bool, optional
        To remove objects before watershed.

    thres_small_before : int, optional
        Theshold to remove small objects created by the watershed.

    seed_morph_sequence : List of str, optional
        List of strings to determine the morphological filters to apply to instance seeds. They will be done in that order.
        E.g. ``['dilate','erode']``.

    seed_morph_radius: List of ints, optional
        List of ints to determine the radius of the erosion or dilation for instance seeds.

    erode_and_dilate_foreground : bool, optional
        To erode and dilate the foreground mask before using marker controlled watershed. The idea is to
        remove the small holes that may be produced so the instances grow without them.

    fore_erosion_radius: int, optional
        Radius to erode the foreground mask.

    fore_dilation_radius: int, optional
        Radius to dilate the foreground mask.

    rmv_close_points : bool, optional
        To remove close points to each other. Used in 'BP' channel configuration.

    remove_close_points_radius : float, optional
        Radius from each point to decide what points to keep. Used in 'BP' channel configuration.
        E.g. ``10.0``.

    resolution : tuple of int/float
        Resolution of the data in ``(z,y,x)`` to calibrate coordinates. E.g. ``[30,8,8]``.

    watershed_by_2d_slices : bool, optional
        Whether to apply or not the watershed to create instances slice by slice in a 3D problem. This can solve instances invading
        others if the objects in Z axis overlap too much.

    save_dir :  str, optional
        Directory to save watershed output into.
    """

    assert channels in [
        "A",
        "C",
        "BC",
        "BCM",
        "BCD",
        "BCDv2",
        "Dv2",
        "BDv2",
        "BP",
        "BD",
        "BCP",
    ]
    assert len(resolution) == 3, "'resolution' must be a list of 3 int/float"

    def erode_seed_and_foreground():
        nonlocal seed_map
        nonlocal foreground
        assert isinstance(seed_map, np.ndarray) and isinstance(foreground, np.ndarray) 

        if len(seed_morph_sequence) != 0:
            print("Applying {} to seeds . . .".format(seed_morph_sequence))
        if erode_and_dilate_foreground:
            print("Foreground erosion . . .")

        if len(seed_morph_sequence) != 0:
            morph_funcs = []
            for operation in seed_morph_sequence:
                if operation == "dilate":
                    morph_funcs.append(binary_dilation)
                elif operation == "erode":
                    morph_funcs.append(binary_erosion)

        image3d = True if seed_map.ndim == 3 else False
        if not image3d:
            seed_map = np.expand_dims(seed_map, 0)
            foreground = np.expand_dims(foreground, 0)

        for i in tqdm(range(seed_map.shape[0])):
            if len(seed_morph_sequence) != 0:
                for k, morph_function in enumerate(morph_funcs):
                    seed_map[i] = morph_function(seed_map[i], disk(radius=seed_morph_radius[k]))

            if erode_and_dilate_foreground:
                foreground[i] = binary_dilation(foreground[i], disk(radius=fore_erosion_radius))
                foreground[i] = binary_erosion(foreground[i], disk(radius=fore_dilation_radius))

        if not image3d:
            seed_map = seed_map.squeeze()
            foreground = foreground.squeeze()

    if channels in ["BC", "BCM"]:
        if ths["TYPE"] == "auto":
            ths["TH_BINARY_MASK"] = threshold_otsu(data[..., 0])
            ths["TH_CONTOUR"] = threshold_otsu(data[..., 1])
            ths["TH_FOREGROUND"] = ths["TH_BINARY_MASK"] / 2
        seed_map = (data[..., 0] > ths["TH_BINARY_MASK"]) * (data[..., 1] < ths["TH_CONTOUR"])
        foreground = data[..., 0] > ths["TH_FOREGROUND"]

        if len(seed_morph_sequence) != 0 or erode_and_dilate_foreground:
            erode_seed_and_foreground()

        semantic = data[..., 0]
        # semantic = edt.edt(foreground*(1-seed_map), anisotropy=resolution[::-1], black_border=False, order='F')
        seed_map = label(seed_map, connectivity=1)
    elif channels in ["C"]:
        if ths["TYPE"] == "auto":
            ths["TH_BINARY_MASK"] = threshold_otsu(1 - data[..., 0])
            ths["TH_CONTOUR"] = threshold_otsu(data[..., 0])
            ths["TH_FOREGROUND"] = ths["TH_BINARY_MASK"] / 2
        seed_map = (1 - data[..., 0] > ths["TH_BINARY_MASK"]) * (data[..., 0] < ths["TH_CONTOUR"])
        foreground = 1 - data[..., 0] > ths["TH_FOREGROUND"]

        if len(seed_morph_sequence) != 0 or erode_and_dilate_foreground:
            erode_seed_and_foreground()

        # semantic = edt.edt(foreground, anisotropy=resolution[::-1], black_border=False, order='F')
        # use contour channel as input to watershed
        semantic = data[..., 0]
        seed_map = label(seed_map, connectivity=1)
    elif channels in ["A"]:
        # For now use the minimum values between all affinities (to enhance borders)
        foreground_probs = np.min(data, axis=-1)

        if ths["TYPE"] == "auto":
            ths["TH_BINARY_MASK"] = threshold_otsu(foreground_probs)
            ths["TH_CONTOUR"] = threshold_otsu(1 - foreground_probs)
            ths["TH_FOREGROUND"] = ths["TH_BINARY_MASK"] / 2
        seed_map = (foreground_probs > ths["TH_BINARY_MASK"]) * (1 - foreground_probs < ths["TH_CONTOUR"])
        foreground = foreground_probs > ths["TH_FOREGROUND"]

        if len(seed_morph_sequence) != 0 or erode_and_dilate_foreground:
            erode_seed_and_foreground()

        # use contour channel as input to watershed
        semantic = 1 - foreground_probs
        seed_map = label(seed_map, connectivity=1)
    elif channels in ["BP"]:
        if ths["TYPE"] == "auto":
            ths["TH_POINTS"] = threshold_otsu(data[..., 1])
            ths["TH_FOREGROUND"] = threshold_otsu(data[..., 0])

        seed_map = data[..., 1] > ths["TH_POINTS"]
        foreground = data[..., 0] > ths["TH_FOREGROUND"]

        print("Creating the central points . . .")
        seed_map = label(seed_map, connectivity=1)
        assert isinstance(seed_map, np.ndarray)
        instances = np.unique(seed_map)[1:]
        seed_coordinates = center_of_mass(seed_map, label(seed_map), instances)
        seed_coordinates = np.round(seed_coordinates).astype(int)

        if rmv_close_points:
            seed_coordinates = remove_close_points(
                seed_coordinates,
                remove_close_points_radius,
                resolution,
                ndim=seed_map.ndim,
            )

        seed_map = np.zeros(data.shape[:-1], dtype=np.uint8)
        for sd in tqdm(seed_coordinates, total=len(seed_coordinates)):
            z, y, x = sd
            seed_map[z, y, x] = 1

        semantic = -edt.edt(1 - seed_map, anisotropy=resolution[::-1], black_border=False, order="F")

        if len(seed_morph_sequence) != 0 or erode_and_dilate_foreground:
            erode_seed_and_foreground()

        seed_map = label(seed_map, connectivity=1)
    elif channels in ["BCP"]:
        if ths["TYPE"] == "auto":
            ths["TH_BINARY_MASK"] = threshold_otsu(data[..., 0])
            ths["TH_CONTOUR"] = threshold_otsu(data[..., 1])
            ths["TH_POINTS"] = threshold_otsu(data[..., 2])
            ths["TH_FOREGROUND"] = ths["TH_BINARY_MASK"] / 2

        seed_map = (data[..., 0] > ths["TH_BINARY_MASK"]) * (data[..., 1] < ths["TH_CONTOUR"]) * (data[..., 2] > ths["TH_POINTS"])
        foreground = data[..., 0] > ths["TH_FOREGROUND"]

        if len(seed_morph_sequence) != 0 or erode_and_dilate_foreground:
            erode_seed_and_foreground()

        semantic = data[..., 0]
        # semantic = edt.edt(foreground*(1-seed_map), anisotropy=resolution[::-1], black_border=False, order='F')
        seed_map = label(seed_map, connectivity=1)
    elif channels in ["BD"]:
        semantic = data[..., 0]
        if ths["TYPE"] == "auto":
            ths["TH_BINARY_MASK"] = threshold_otsu(data[..., 0])
            ths["TH_FOREGROUND"] = ths["TH_BINARY_MASK"] / 2
        seed_map = (data[..., 0] > ths["TH_BINARY_MASK"]) * (data[..., 1] < ths["TH_DISTANCE"])
        foreground = semantic > ths["TH_FOREGROUND"]
        seed_map = label(seed_map, connectivity=1)
    elif channels in ["BCD"]:
        semantic = data[..., 0]
        if ths["TYPE"] == "auto":
            ths["TH_BINARY_MASK"] = threshold_otsu(data[..., 0])
            ths["TH_CONTOUR"] = threshold_otsu(data[..., 1])
            ths["TH_FOREGROUND"] = ths["TH_BINARY_MASK"] / 2

        seed_map = (
            (data[..., 0] > ths["TH_BINARY_MASK"])
            * (data[..., 1] < ths["TH_CONTOUR"])
            * (data[..., 2] < ths["TH_DISTANCE"])
        )
        foreground = semantic > ths["TH_FOREGROUND"]

        if len(seed_morph_sequence) != 0 or erode_and_dilate_foreground:
            erode_seed_and_foreground()
        seed_map = label(seed_map, connectivity=1)
    else:  # 'BCDv2', 'Dv2', 'BDv2'
        semantic = data[..., -1]
        foreground = None
        if channels == "BCDv2":  # 'BCDv2'
            if ths["TYPE"] == "auto":
                ths["TH_BINARY_MASK"] = threshold_otsu(data[..., 0])
                ths["TH_CONTOUR"] = threshold_otsu(data[..., 1])
            seed_map = (
                (data[..., 0] > ths["TH_BINARY_MASK"])
                * (data[..., 1] < ths["TH_CONTOUR"])
                * (data[..., 1] < ths["TH_DISTANCE"])
            )
            background_seed = binary_dilation(
                ((data[..., 0] > ths["TH_BINARY_MASK"]) + (data[..., 1] > ths["TH_CONTOUR"])).astype(np.uint8),
                iterations=2,
            )
            assert isinstance(seed_map, np.ndarray)
            seed_map, num = label(seed_map, connectivity=1, return_num=True) # type: ignore

            # Create background seed and label correctly
            background_seed = 1 - background_seed
            background_seed[background_seed == 1] = num + 1
            seed_map = seed_map + background_seed
            del background_seed
        elif channels == "BDv2":  # 'BDv2'
            if ths["TYPE"] == "auto":
                ths["TH_BINARY_MASK"] = threshold_otsu(data[..., 0])
            seed_map = (data[..., 0] > ths["TH_BINARY_MASK"]) * (data[..., 1] < ths["TH_DISTANCE"])
            background_seed = binary_dilation((data[..., 1] < ths["TH_DISTANCE"]).astype(np.uint8), iterations=2)
            seed_map = label(seed_map, connectivity=1)
            background_seed = label(background_seed, connectivity=1)
            assert isinstance(background_seed, np.ndarray) and isinstance(seed_map, np.ndarray)

            props = regionprops_table(seed_map, properties=("area", "centroid"))
            for n in range(len(props["centroid-0"])):
                label_center = [
                    props["centroid-0"][n],
                    props["centroid-1"][n],
                    props["centroid-2"][n],
                ]
                instance_to_remove = background_seed[label_center]
                background_seed[background_seed == instance_to_remove] = 0
            seed_map = seed_map + background_seed
            del background_seed
            seed_map = label(seed_map, connectivity=1)  # re-label again
        elif channels == "Dv2":  # 'Dv2'
            seed_map = data[..., 0] < ths["TH_DISTANCE"]
            seed_map = label(seed_map, connectivity=1)

        if len(seed_morph_sequence) != 0:
            erode_seed_and_foreground()

    # Print the thresholds used in automatic case
    if ths["TYPE"] == "auto":
        print("Thresholds used: {}".format(ths))

    if remove_before:
        seed_map = remove_small_objects(seed_map, thres_small_before)
        seed_map, _, _ = relabel_sequential(seed_map)

    assert isinstance(seed_map, np.ndarray) and foreground is not None

    # Choose appropiate dtype
    max_value = np.max(seed_map)
    if max_value < 255:
        appropiate_dtype = np.uint8
    elif max_value < 65535:
        appropiate_dtype = np.uint16
    else:
        appropiate_dtype = np.uint32

    if watershed_by_2d_slices:
        print("Doing watershed by 2D slices")
        segm = np.zeros(seed_map.shape, dtype=appropiate_dtype)
        for z in tqdm(range(len(segm))):
            segm[z] = watershed(-semantic[z], seed_map[z], mask=foreground[z])
    else:
        segm = watershed(-semantic, seed_map, mask=foreground)
        segm = segm.astype(appropiate_dtype)

    if save_dir:
        save_tif(
            np.expand_dims(np.expand_dims(seed_map, -1), 0).astype(segm.dtype),
            save_dir,
            ["seed_map.tif"],
            verbose=False,
        )
        save_tif(
            np.expand_dims(np.expand_dims(semantic, -1), 0).astype(np.float32),
            save_dir,
            ["semantic.tif"],
            verbose=False,
        )
        if channels in ["A", "C", "BC", "BCM", "BCD", "BP"]:
            save_tif(
                np.expand_dims(np.expand_dims(foreground, -1), 0).astype(np.uint8),
                save_dir,
                ["foreground.tif"],
                verbose=False,
            )
    return segm

def create_synapses(data: NDArray,
    channels: str,
    point_creation_func: str = "peak_local_max",
    min_th_to_be_peak: float = 0.2,
    min_distance: int=1,
    min_sigma: int=5,
    max_sigma: int=10,
    num_sigma: int=2,
    exclude_border: bool = False,
) -> Tuple[NDArray, Dict]:
    """
    Convert binary foreground probability maps and instance contours to instance masks via watershed segmentation
    algorithm.

    Implementation based on `PyTorch Connectomics' process.py
    <https://github.com/zudi-lin/pytorch_connectomics/blob/master/connectomics/utils/process.py>`_.

    Parameters
    ----------
    data : 4D Numpy array
        Binary foreground labels and contours data to apply watershed into. E.g. ``(397, 1450, 2000, 2)``.

    channels : str
        Channel type used. Possible options: ``A``, ``C``, ``BC``, ``BCM``, ``BCD``, ``BCDv2``, ``Dv2`` and ``BDv2``.
    """
    assert channels in [
        "B",
        "BF",
    ]
    assert point_creation_func in ["peak_local_max", "blob_log"]

    d_result = {}
    ids, probs = [], []
    if channels == "BF":
        print("TODO")
        import pdb; pdb.set_trace()
    elif channels == "B":
        # Take the coords of the predicted points
        all_coords = []
        max_value = 0
        data = data.astype(np.float32)
        for c in range(data.shape[-1]):
            if point_creation_func == "peak_local_max":
                coords = peak_local_max(
                    data[..., c],
                    min_distance=min_distance,
                    threshold_abs=min_th_to_be_peak,
                    exclude_border=exclude_border,
                )
                coords = coords.astype(int)
            else:
                coords = blob_log(
                    data[..., c] * 255,
                    min_sigma=min_sigma,
                    max_sigma=max_sigma,
                    num_sigma=num_sigma,
                    threshold=min_th_to_be_peak,
                    exclude_border=exclude_border,
                )
                coords = coords[:, :3].astype(int)  # Remove sigma

            all_coords.append(coords)
            if max_value < len(coords):
                max_value = len(coords)
        
        # Choose appropiate dtype
        max_value = np.max(data)
        if max_value < 255:
            appropiate_dtype = np.uint8
        elif max_value < 65535:
            appropiate_dtype = np.uint16
        else:
            appropiate_dtype = np.uint32
        new_data = np.zeros(data.shape, dtype=appropiate_dtype)

        lbl_cont = 1
        ellipse_footprint_cpd = generate_ellipse_footprint([2,4,4])
        for c, coords in enumerate(all_coords):
            for coord in coords:
                z,y,x = coord
                new_data[int(z), int(y), int(x), c] = lbl_cont
                probs.append(float(data[z,y,x,c]))
                ids.append(lbl_cont)
                lbl_cont += 1

            # Dilate the labels
            new_data[...,c] = binary_dilation_scipy(
                new_data[...,c],
                iterations=1,
                structure=ellipse_footprint_cpd,
            )

        d_result["ids"] = ids
        d_result["tag"] = (["pre",]*len(all_coords[0])) + (["post",]*len(all_coords[1]))
        d_result["probabilities"] = probs
        d_result["points"] = np.concatenate(all_coords, axis=0)

        return new_data, d_result
    return np.zeros((10,10)), {"da":"da"} 

def apply_median_filtering(
    data: NDArray, 
    axes: str="xy", 
    mf_size: int=5
) -> NDArray:
    """
    Applies a median filtering to the specified axes of the provided data.

    Parameters
    ----------
    data : 4D/5D Numpy array
        Data to apply the filter to. E.g. ``(num_of_images, y, x, channels)`` for 2D and
        ``(num_of_images, z, y, x, channels)`` for 3D.

    axes : str, optional
        Median filters to apply. There are multiple options: ``'xy'`` or ``'yx'`` to apply the filter in ``x``
        and ``y`` axes together; ``'zy'`` or ``'yz'`` to apply the filter in ``y`` and ``z`` axes
        together; ``'zx'`` or ``'xz'``: to apply the filter in ``x`` and ``z`` axes together;
        ``'z'``: to apply the filter only in ``z`` axis.

    mf_size : int, optional
        Size of the median filter. If an odd number is not provided, ``1`` will be added to ensure it remains odd.

    Returns
    -------
    Array : 4D/5D Numpy array
        Filtered data. E.g. ``(num_of_images, y, x, channels)`` for 2D and
        ``(num_of_images, z, y, x, channels)`` for 3D.
    """
    assert axes in ["xy", "yx", "zy", "yz", "zx", "xz", "z"]

    is_3d = True if data.ndim == 5 else False

    # Must be odd
    if mf_size % 2 == 0:
        mf_size += 1

    for i in range(data.shape[0]):
        for c in range(data.shape[-1]):
            s = None
            if axes in ["xy", "yx"]:
                s = (1, mf_size, mf_size) if is_3d else (mf_size, mf_size)
            elif axes in ["zy", "yz"]:
                s = (mf_size, mf_size, 1)
            elif axes in ["zx", "xz"]:
                s = (mf_size, 1, mf_size)
            else:  # "z"
                s = (mf_size, 1, 1)
            data[i, ..., c] = median_filter(data[i, ..., c], size=s)
    return data


def ensemble8_2d_predictions(
    o_img: NDArray,
    pred_func: Callable,
    axes_order_back: Tuple[int,...],
    axes_order: Tuple[int,...],
    device: torch.device,
    batch_size_value: int=1,
    mode="mean",
) -> torch.Tensor:
    """
    Outputs the mean prediction of a given image generating its 8 possible rotations and flips.

    Parameters
    ----------
    o_img : 3D Numpy array
        Input image. E.g. ``(y, x, channels)``.

    pred_func : function
        Function to make predictions.

    axes_order_back : tuple
        Axis order to convert from tensor to numpy. E.g. ``(0,3,1,2)``.

    axes_order : tuple
        Axis order to convert from numpy to tensor. E.g. ``(0,3,1,2)``.
 
    device : Torch device
        Device used.
 
    batch_size_value : int, optional
        Batch size value.

    mode : str, optional
        Ensemble mode. Possible options: "mean", "min", "max".

    Returns
    -------
    out : 3D Numpy array
        Output image ensembled. E.g. ``(y, x, channels)``.

    Examples
    --------
    ::

        # EXAMPLE 1
        # Apply ensemble to each image of X_test
        X_test = np.ones((165, 768, 1024, 1))
        out_X_test = np.zeros(X_test.shape, dtype=(np.float32))

        for i in tqdm(range(X_test.shape[0])):
            pred_ensembled = ensemble8_2d_predictions(X_test[i],
                pred_func=(lambda img_batch_subdiv: model(img_batch_subdiv)), n_classes=n_classes)
            out_X_test[i] = pred_ensembled
    """
    assert mode in ["mean", "min", "max"], "Get unknown ensemble mode {}".format(mode)

    # Prepare all the image transformations per channel
    total_img = []
    for channel in range(o_img.shape[-1]):
        aug_img = []

        # Transformations per channel
        _img = np.expand_dims(o_img[..., channel], -1)

        # Convert into square image to make the rotations properly
        pad_to_square = _img.shape[0] - _img.shape[1]
        if pad_to_square < 0:
            img = np.pad(_img, [(abs(pad_to_square), 0), (0, 0), (0, 0)], "reflect")
        else:
            img = np.pad(_img, [(0, 0), (pad_to_square, 0), (0, 0)], "reflect")

        # Make 8 different combinations of the img
        aug_img.append(img)
        aug_img.append(np.rot90(img, axes=(0, 1), k=1))
        aug_img.append(np.rot90(img, axes=(0, 1), k=2))
        aug_img.append(np.rot90(img, axes=(0, 1), k=3))
        aug_img.append(img[:, ::-1])
        img_aux = img[:, ::-1]
        aug_img.append(np.rot90(img_aux, axes=(0, 1), k=1))
        aug_img.append(np.rot90(img_aux, axes=(0, 1), k=2))
        aug_img.append(np.rot90(img_aux, axes=(0, 1), k=3))
        aug_img = np.array(aug_img)

        total_img.append(aug_img)

    del aug_img, img_aux

    # Merge channels
    total_img = np.concatenate(total_img, -1)

    # Make the prediction
    _decoded_aug_img = []
    l = int(math.ceil(total_img.shape[0] / batch_size_value))
    for i in range(l):
        top = (i + 1) * batch_size_value if (i + 1) * batch_size_value < total_img.shape[0] else total_img.shape[0]
        r_aux = pred_func(total_img[i * batch_size_value : top])

        # Take just the first output of the network in case it returns more than one output
        if isinstance(r_aux, list):
            r_aux = to_numpy_format(r_aux[0], axes_order_back)
        else:
            r_aux = to_numpy_format(r_aux, axes_order_back)
        _decoded_aug_img.append(r_aux)
    _decoded_aug_img = np.concatenate(_decoded_aug_img)

    # Undo the combinations of the img
    arr = []
    for c in range(_decoded_aug_img.shape[-1]):
        # Remove the last channel to make the transformations correctly
        decoded_aug_img = _decoded_aug_img[..., c].astype(np.float32)

        # Undo the combinations of the image
        out_img = []
        out_img.append(decoded_aug_img[0])
        out_img.append(np.rot90(decoded_aug_img[1], axes=(0, 1), k=3))
        out_img.append(np.rot90(decoded_aug_img[2], axes=(0, 1), k=2))
        out_img.append(np.rot90(decoded_aug_img[3], axes=(0, 1), k=1))
        out_img.append(decoded_aug_img[4][:, ::-1])
        out_img.append(np.rot90(decoded_aug_img[5], axes=(0, 1), k=3)[:, ::-1])
        out_img.append(np.rot90(decoded_aug_img[6], axes=(0, 1), k=2)[:, ::-1])
        out_img.append(np.rot90(decoded_aug_img[7], axes=(0, 1), k=1)[:, ::-1])
        out_img = np.array(out_img)
        out_img = np.expand_dims(out_img, -1)
        arr.append(out_img)

    out_img = np.concatenate(arr, -1)
    del decoded_aug_img, _decoded_aug_img, arr

    if pad_to_square != 0:
        if pad_to_square < 0:
            out = np.zeros(
                (
                    out_img.shape[0],
                    img.shape[0] + pad_to_square,
                    img.shape[1],
                    out_img.shape[-1],
                )
            )
        else:
            out = np.zeros(
                (
                    out_img.shape[0],
                    img.shape[0],
                    img.shape[1] - pad_to_square,
                    out_img.shape[-1],
                )
            )
    else:
        out = np.zeros(out_img.shape)

    # Undo the padding
    for i in range(out_img.shape[0]):
        if pad_to_square < 0:
            out[i] = out_img[i, abs(pad_to_square) :, :]
        else:
            out[i] = out_img[i, :, abs(pad_to_square) :]

    funct = np.mean
    if mode == "min":
        funct = np.min
    elif mode == "max":
        funct = np.max
    out = np.expand_dims(funct(out, axis=0), 0)
    out = to_pytorch_format(out, axes_order, device)
    return out


def ensemble16_3d_predictions(
    vol: NDArray, 
    pred_func: Callable, 
    axes_order_back: Tuple[int,...], 
    axes_order: Tuple[int,...],
    device: torch.device,
    batch_size_value: int=1, 
    mode: str="mean"
) -> torch.Tensor:
    """
    Outputs the mean prediction of a given image generating its 16 possible rotations and flips.

    Parameters
    ----------
    o_img : 4D Numpy array
        Input image. E.g. ``(z, y, x, channels)``.

    pred_func : function
        Function to make predictions.

    axes_order_back : tuple
        Axis order to convert from tensor to numpy. E.g. ``(0,3,1,2,4)``.

    axes_order : tuple
        Axis order to convert from numpy to tensor. E.g. ``(0,3,1,2)``.
 
    device : Torch device
        Device used.

    batch_size_value : int, optional
        Batch size value.

    mode : str, optional
        Ensemble mode. Possible options: "mean", "min", "max".

    Returns
    -------
    out : 4D Numpy array
        Output image ensembled. E.g. ``(z, y, x, channels)``.

    Examples
    --------
    ::

        # EXAMPLE 1
        # Apply ensemble to each image of X_test
        X_test = np.ones((10, 165, 768, 1024, 1))
        out_X_test = np.zeros(X_test.shape, dtype=(np.float32))

        for i in tqdm(range(X_test.shape[0])):
            pred_ensembled = ensemble8_2d_predictions(X_test[i],
                pred_func=(lambda img_batch_subdiv: model(img_batch_subdiv)))
            out_X_test[i] = pred_ensembled
    """
    assert mode in ["mean", "min", "max"], "Get unknown ensemble mode {}".format(mode)

    total_vol = []
    for channel in range(vol.shape[-1]):

        aug_vols = []

        # Transformations per channel
        _vol = vol[..., channel]

        # Convert into square image to make the rotations properly
        pad_to_square = _vol.shape[2] - _vol.shape[1]
        if pad_to_square < 0:
            volume = np.pad(_vol, [(0, 0), (0, 0), (abs(pad_to_square), 0)], "reflect")
        else:
            volume = np.pad(_vol, [(0, 0), (pad_to_square, 0), (0, 0)], "reflect")

        # Make 16 different combinations of the volume
        aug_vols.append(volume)
        aug_vols.append(rotate(volume, mode="reflect", axes=(2, 1), angle=90, reshape=False))
        aug_vols.append(rotate(volume, mode="reflect", axes=(2, 1), angle=180, reshape=False))
        aug_vols.append(rotate(volume, mode="reflect", axes=(2, 1), angle=270, reshape=False))
        volume_aux = np.flip(volume, 0)
        aug_vols.append(volume_aux)
        aug_vols.append(rotate(volume_aux, mode="reflect", axes=(2, 1), angle=90, reshape=False))
        aug_vols.append(rotate(volume_aux, mode="reflect", axes=(2, 1), angle=180, reshape=False))
        aug_vols.append(rotate(volume_aux, mode="reflect", axes=(2, 1), angle=270, reshape=False))
        volume_aux = np.flip(volume, 1)
        aug_vols.append(volume_aux)
        aug_vols.append(rotate(volume_aux, mode="reflect", axes=(2, 1), angle=90, reshape=False))
        aug_vols.append(rotate(volume_aux, mode="reflect", axes=(2, 1), angle=180, reshape=False))
        aug_vols.append(rotate(volume_aux, mode="reflect", axes=(2, 1), angle=270, reshape=False))
        volume_aux = np.flip(volume, 2)
        aug_vols.append(volume_aux)
        aug_vols.append(rotate(volume_aux, mode="reflect", axes=(2, 1), angle=90, reshape=False))
        aug_vols.append(rotate(volume_aux, mode="reflect", axes=(2, 1), angle=180, reshape=False))
        aug_vols.append(rotate(volume_aux, mode="reflect", axes=(2, 1), angle=270, reshape=False))
        aug_vols = np.array(aug_vols)

        # Add the last channel again
        aug_vols = np.expand_dims(aug_vols, -1)
        total_vol.append(aug_vols)

    del aug_vols, volume_aux
    # Merge channels
    total_vol = np.concatenate(total_vol, -1)

    _decoded_aug_vols = []

    l = int(math.ceil(total_vol.shape[0] / batch_size_value))
    for i in range(l):
        top = (i + 1) * batch_size_value if (i + 1) * batch_size_value < total_vol.shape[0] else total_vol.shape[0]
        r_aux = pred_func(total_vol[i * batch_size_value : top])

        # Take just the first output of the network in case it returns more than one output
        if isinstance(r_aux, list):
            r_aux = to_numpy_format(r_aux[0], axes_order_back)
        else:
            r_aux = to_numpy_format(r_aux, axes_order_back)

        if r_aux.ndim == 4:
            r_aux = np.expand_dims(r_aux, 0)
        _decoded_aug_vols.append(r_aux)

    _decoded_aug_vols = np.concatenate(_decoded_aug_vols)
    volume = np.expand_dims(volume, -1)

    arr = []
    for c in range(_decoded_aug_vols.shape[-1]):
        # Remove the last channel to make the transformations correctly
        decoded_aug_vols = _decoded_aug_vols[..., c].astype(np.float32)
        # Undo the combinations of the volume
        out_vols = []
        out_vols.append(np.array(decoded_aug_vols[0]))
        out_vols.append(
            rotate(
                np.array(decoded_aug_vols[1]),
                mode="reflect",
                axes=(2, 1),
                angle=-90,
                reshape=False,
            )
        )
        out_vols.append(
            rotate(
                np.array(decoded_aug_vols[2]),
                mode="reflect",
                axes=(2, 1),
                angle=-180,
                reshape=False,
            )
        )
        out_vols.append(
            rotate(
                np.array(decoded_aug_vols[3]),
                mode="reflect",
                axes=(2, 1),
                angle=-270,
                reshape=False,
            )
        )
        out_vols.append(np.flip(np.array(decoded_aug_vols[4]), 0))
        out_vols.append(
            np.flip(
                rotate(
                    np.array(decoded_aug_vols[5]),
                    mode="reflect",
                    axes=(2, 1),
                    angle=-90,
                    reshape=False,
                ),
                0,
            )
        )
        out_vols.append(
            np.flip(
                rotate(
                    np.array(decoded_aug_vols[6]),
                    mode="reflect",
                    axes=(2, 1),
                    angle=-180,
                    reshape=False,
                ),
                0,
            )
        )
        out_vols.append(
            np.flip(
                rotate(
                    np.array(decoded_aug_vols[7]),
                    mode="reflect",
                    axes=(2, 1),
                    angle=-270,
                    reshape=False,
                ),
                0,
            )
        )
        out_vols.append(np.flip(np.array(decoded_aug_vols[8]), 1))
        out_vols.append(
            np.flip(
                rotate(
                    np.array(decoded_aug_vols[9]),
                    mode="reflect",
                    axes=(2, 1),
                    angle=-90,
                    reshape=False,
                ),
                1,
            )
        )
        out_vols.append(
            np.flip(
                rotate(
                    np.array(decoded_aug_vols[10]),
                    mode="reflect",
                    axes=(2, 1),
                    angle=-180,
                    reshape=False,
                ),
                1,
            )
        )
        out_vols.append(
            np.flip(
                rotate(
                    np.array(decoded_aug_vols[11]),
                    mode="reflect",
                    axes=(2, 1),
                    angle=-270,
                    reshape=False,
                ),
                1,
            )
        )
        out_vols.append(np.flip(np.array(decoded_aug_vols[12]), 2))
        out_vols.append(
            np.flip(
                rotate(
                    np.array(decoded_aug_vols[13]),
                    mode="reflect",
                    axes=(2, 1),
                    angle=-90,
                    reshape=False,
                ),
                2,
            )
        )
        out_vols.append(
            np.flip(
                rotate(
                    np.array(decoded_aug_vols[14]),
                    mode="reflect",
                    axes=(2, 1),
                    angle=-180,
                    reshape=False,
                ),
                2,
            )
        )
        out_vols.append(
            np.flip(
                rotate(
                    np.array(decoded_aug_vols[15]),
                    mode="reflect",
                    axes=(2, 1),
                    angle=-270,
                    reshape=False,
                ),
                2,
            )
        )

        out_vols = np.array(out_vols)
        out_vols = np.expand_dims(out_vols, -1)
        arr.append(out_vols)

    out_vols = np.concatenate(arr, -1)
    del decoded_aug_vols, _decoded_aug_vols, arr

    # Create the output data
    if pad_to_square != 0:
        if pad_to_square < 0:
            out = np.zeros(
                (
                    out_vols.shape[0],
                    volume.shape[0],
                    volume.shape[1],
                    volume.shape[2] + pad_to_square,
                    out_vols.shape[-1],
                )
            )
        else:
            out = np.zeros(
                (
                    out_vols.shape[0],
                    volume.shape[0],
                    volume.shape[1] - pad_to_square,
                    volume.shape[2],
                    out_vols.shape[-1],
                )
            )
    else:
        out = np.zeros(out_vols.shape)

    # Undo the padding
    for i in range(out_vols.shape[0]):
        if pad_to_square < 0:
            out[i] = out_vols[i, :, :, abs(pad_to_square) :, :]
        else:
            out[i] = out_vols[i, :, abs(pad_to_square) :, :, :]

    funct = np.mean
    if mode == "min":
        funct = np.min
    elif mode == "max":
        funct = np.max
    out = np.expand_dims(funct(out, axis=0), 0)
    out = to_pytorch_format(out, axes_order, device)
    return out


def create_th_plot(
    ths: List[float],
    y_list: list[int|float],
    chart_dir: str,
    th_name: str="TH_BINARY_MASK",
    per_sample: bool=True,
    ideal_value: Optional[int|float]=None,
):
    """Create plots for threshold value calculation.

    Parameters
    ----------
    ths : List of floats
        List of thresholds. It will be the ``x`` axis.

    y_list : List of ints/floats
        Values of ``y`` axis.

    chart_dir : str, optional
        Path where the charts are stored.

    th_name : str, optional
        Name of the threshold.

    per_sample : bool, optional
        Create the plot per list in ``y_list``.

    ideal_value : int/float, optional
        Value that should be the ideal optimum. It is going to be marked with a red line in the chart.
    """

    assert th_name in [
        "TH_BINARY_MASK",
        "TH_CONTOUR",
        "TH_FOREGROUND",
        "TH_DISTANCE",
        "TH_DIST_FOREGROUND",
    ]
    fig, ax = plt.subplots(figsize=(25, 10))
    ths = [str(i) for i in ths] # type: ignore
    num_points = len(ths)

    N = len(y_list)
    colors = list(range(0, N))
    c_labels = list("vol_" + str(i) for i in range(0, N))
    if per_sample:
        for i in range(N):
            l = "_nolegend_" if i > 30 else c_labels[i]
            ax.plot(ths, y_list[i], label=l, alpha=0.4)

        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height]) # type: ignore
        # Put a legend to the right of the current axis
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    else:
        y_list = np.array(y_list) # type: ignore
        y_mean = np.mean(y_list, axis=0)
        y_std = np.std(y_list, axis=0)
        ax.plot(ths, y_mean, label="sample (mean)")
        plt.fill_between(ths, y_mean - y_std, y_mean + y_std, alpha=0.25)
        if ideal_value:
            plt.axhline(ideal_value, color="r")
            trans = transforms.blended_transform_factory(ax.get_yticklabels()[0].get_transform(), ax.transData)
            ax.text(
                1.12,
                ideal_value,
                "Ideal (mean)",
                color="red",
                transform=trans,
                ha="right",
                va="center",
            )
        ax.legend(loc="center right")

    # Set labels of x axis
    plt.xticks(ths)
    a = np.arange(num_points)
    ax.xaxis.set_ticks(a)
    ax.xaxis.set_ticklabels(ths) # type: ignore

    plt.title("Threshold " + str(th_name))
    plt.xlabel("Threshold")
    if th_name == "TH_FOREGROUND" or th_name == "TH_DIST_FOREGROUND":
        plt.ylabel("IoU")
    else:
        plt.ylabel("Number of objects")
    p = "_per_validation_sample" if per_sample else ""
    plt.savefig(os.path.join(chart_dir, str(th_name) + p + ".svg"), format="svg", dpi=100)
    plt.show()


def voronoi_on_mask(
    data:NDArray, 
    mask: NDArray, 
    th: float=0, 
    verbose: bool=False
) -> NDArray:
    """
    Apply Voronoi to the voxels not labeled yet marked by the mask. It is done using distances from the un-labeled
    voxels to the cell perimeters.

    Parameters
    ----------
    data : 2D/3D Numpy array
        Data to apply Voronoi. ``(y, x)`` for 2D or ``(z, y, x)`` for 3D.
        E.g. ``(397, 1450, 2000)`` for 3D.

    mask : 3D/4D Numpy array
        Data mask to determine which points need to be proccessed. ``(z, y, x, channels)`` e.g.
        ``(397, 1450, 2000, 3)``.

    th : float, optional
        Threshold used to binarize the input. If th=0, otsu threshold is used.

    thres_small : int, optional
        Theshold to remove small objects created by the watershed.

    verbose : bool, optional
         To print saving information.

    Returns
    -------
    data : 4D Numpy array
        Image with Voronoi applied. ``(num_of_images, z, y, x)`` e.g. ``(1, 397, 1450, 2000)``
    """

    if data.ndim != 2 and data.ndim != 3:
        raise ValueError("Data must be 2/3 dimensional, provided {}".format(data.shape))
    if mask.ndim != 3 and mask.ndim != 4:
        raise ValueError("Data mask must be 3/4 dimensional, provided {}".format(mask.shape))
    if mask.shape[-1] < 2:
        raise ValueError("Mask needs to have two channels at least, received {}".format(mask.shape[-1]))

    if verbose:
        print("Applying Voronoi {}D . . .".format(data.ndim))

    image3d = False if data.ndim != 2 else True

    if image3d:
        data = np.expand_dims(data, 0)
        mask = np.expand_dims(mask, 0)

    # Extract mask from prediction
    if mask.shape[-1] == 3:
        mask = mask[..., 2]
    else:
        mask = mask[..., 0] + mask[..., 1]

    mask_shape = np.shape(mask)

    # Binarize
    if th == 0:
        thresh = threshold_otsu(mask)
    else:
        thresh = th
    binaryMask = mask > thresh

    # Close to fill holes
    closedBinaryMask = morphology.closing(binaryMask, morphology.ball(radius=5)).astype(np.uint8)

    voronoiCyst = data * closedBinaryMask
    binaryVoronoiCyst = (voronoiCyst > 0) * 1
    binaryVoronoiCyst = binaryVoronoiCyst.astype("uint8")

    # Cell Perimeter
    erodedVoronoiCyst = morphology.binary_erosion(binaryVoronoiCyst, morphology.ball(radius=2))
    cellPerimeter = binaryVoronoiCyst - erodedVoronoiCyst

    # Define ids to fill where there is mask but no labels
    idsToFill = np.argwhere((closedBinaryMask == 1) & (data == 0))
    labelPerId = np.zeros(np.size(idsToFill))

    idsPerim = np.argwhere(cellPerimeter == 1)
    labelsPerimIds = voronoiCyst[cellPerimeter == 1]

    # Generating voronoi
    for nId in tqdm(range(1, len(idsToFill))):
        distCoord = cdist([idsToFill[nId]], idsPerim)
        idSeedMin = np.argwhere(distCoord == np.min(distCoord))
        idSeedMin = idSeedMin[0][1]
        labelPerId[nId] = labelsPerimIds[idSeedMin]
        voronoiCyst[idsToFill[nId][0], idsToFill[nId][1], idsToFill[nId][2]] = labelsPerimIds[idSeedMin]

    if image3d:
        data = data[0]
        mask = mask[0]
        voronoiCyst = voronoiCyst[0]

    return voronoiCyst


def remove_close_points_by_mask(
    points: NDArray | List[List[int | float]], 
    radius: float, 
    raw_predictions: NDArray | Type[zarr.hierarchy.Group] | Type[zarr.core.Array], # type: ignore
    bin_th: float,
    resolution: List[int|float], 
    channel_to_look_into: int = 1,
    classes: Optional[List[int]]=None, 
    ndim: int=3, 
    return_drops: bool=False,
) -> List[List[int | float]] | Tuple[List[List[int | float]], List[int]] | Tuple[List[List[int | float]], List[int], List[bool]]:
    """
    Remove all points from ``point_list`` that are at a ``radius`` or less distance from each other but conditioned that the must 
    lay in the same mask label. For that last label creation the given ``raw_predictions`` is used, which is expected to be model's 
    raw prediction. It is binarized using ``bin_th`` threshold and then the labels are created using connected-components.

    Parameters
    ----------
    points : ndarray of floats
        List of 3D points. E.g. ``((0,0,0), (1,1,1)``.

    radius : float
        Radius from each point to decide what points to keep. E.g. ``10.0``.

    resolution : Tuple of int/float
        Resolution of the data, in ``(z,y,x)`` to calibrate coordinates. E.g. ``[30,8,8]``.

    ndim : int, optional
        Number of dimension of the data.

    return_drops : bool, optional
        Whether to return or not a list containing the positions of the points removed.

    search_in_mask : bool, optional
    
    Returns
    -------
    new_point_list : List of lists of floats
        New list of points after removing those at a distance of ``radius``
        or less from each other.
    """
    assert len(resolution) == 3, "'resolution' must be a list of 3 int/float"

    print("Removing close points . . .")
    print("Initial number of points: " + str(len(points)))

    point_list = points.copy()

    # Resolution adjust
    for i in range(len(point_list)):
        point_list[i][0] = point_list[i][0] * resolution[0]
        point_list[i][1] = point_list[i][1] * resolution[1]
        if ndim == 3:
            point_list[i][2] = point_list[i][2] * resolution[2]

    mynumbers = [tuple(point) for point in point_list]

    tree = cKDTree(mynumbers)  # build k-dimensional tree
    pairs = tree.query_pairs(radius)  # find all pairs closer than radius

    neighbors = {}  # create dictionary of neighbors
    for i, j in pairs:  # iterate over all pairs
        if i not in neighbors:
            neighbors[i] = [j]
        else:
            neighbors[i].append(j)
        if j not in neighbors:
            neighbors[j] = [i]
        else:
            neighbors[j].append(i)

    discard = []
    keep = {}
    for i in range(0, len(point_list)):
        keep[i] = True

    # Iterate over all the groups found
    for item in neighbors.items():
        group_of_points = item[1]
        first_point = points[item[0]]
        patch_coords = [
            slice(first_point[0] - 1, first_point[0] + 1),
            slice(first_point[1] - 1, first_point[1] + 1),
            slice(first_point[2] - 1, first_point[2] + 1),
        ]

        # Determine the patch to extract
        for i in range(len(group_of_points)):
            point = points[group_of_points[i]]
            patch_coords = [
                slice(
                    max(0, min(patch_coords[0].start - 1, point[0] - 1)),
                    min(max(patch_coords[0].stop + 1, point[0] + 1), raw_predictions.shape[0]),
                ),
                slice(
                    max(0, min(patch_coords[1].start - 1, point[1] - 1)),
                    min(max(patch_coords[1].stop + 1, point[1] + 1), raw_predictions.shape[1]),
                ),
                slice(
                    max(0, min(patch_coords[2].start - 1, point[2] - 1)),
                    min(max(patch_coords[2].stop + 1, point[2] + 1), raw_predictions.shape[2]),
                ),
            ]
        patch_coords += [slice(channel_to_look_into, channel_to_look_into + 1),]
        patch_coords = tuple(patch_coords)

        # Extract the patch, binarize and apply connected-components
        patch = np.array(raw_predictions[patch_coords] > bin_th, dtype=np.uint8).squeeze()
        patch = label(patch)

        # Groups points by labels
        labels_detected = {}
        for point_id in group_of_points:
            point_coord = points[point_id]
            coord_in_patch = point_coord-[patch_coords[0].start,patch_coords[1].start, patch_coords[2].start]
            label_of_point = patch[int(coord_in_patch[0]),int(coord_in_patch[1]),int(coord_in_patch[2])]
            if label_of_point not in labels_detected:
                labels_detected[label_of_point] = []
            labels_detected[label_of_point].append(point_id)
        
        # TODO: Create a new point coord based on the mean values of all the points 
        # Take only the first point in case more than one were grouped
        for item in labels_detected.items():
            for i in range(1,len(item[1])):
                if item[1][i] in keep:
                    del keep[item[1][i]]
                    discard.append(item[1][i])

    # points to keep
    keep = list(keep.keys())
    new_point_list = [list(points[i]) for i in keep]
    print("Final number of points: " + str(len(new_point_list)))

    if classes:
        new_class_list = [classes[i] for i in keep]
        if return_drops:
            return new_point_list, new_class_list, list(discard)
        else:
            return new_point_list, new_class_list
    else:
        if return_drops:
            return new_point_list, list(discard)
        else:
            return new_point_list

def remove_close_points(
    points: NDArray | List[List[int | float]], 
    radius: float, 
    resolution: List[int|float], 
    classes: Optional[List[int]]=None, 
    ndim: int=3, 
    return_drops: bool=False,
) -> List[List[int | float]] | Tuple[List[List[int | float]], List[int]] | Tuple[List[List[int | float]], List[int], List[bool]]:
    """
    Remove all points from ``point_list`` that are at a ``radius`` or less distance from each other.

    Parameters
    ----------
    points : ndarray of floats
        List of 3D points. E.g. ``((0,0,0), (1,1,1)``.

    radius : float
        Radius from each point to decide what points to keep. E.g. ``10.0``.

    resolution : Tuple of int/float
        Resolution of the data, in ``(z,y,x)`` to calibrate coordinates. E.g. ``[30,8,8]``.

    ndim : int, optional
        Number of dimension of the data.

    return_drops : bool, optional
        Whether to return or not a list containing the positions of the points removed.

    Returns
    -------
    new_point_list : List of lists of floats
        New list of points after removing those at a distance of ``radius``
        or less from each other.
    """
    assert len(resolution) == 3, "'resolution' must be a list of 3 int/float"

    print("Removing close points . . .")
    print("Initial number of points: " + str(len(points)))

    point_list = points.copy()

    # Resolution adjust
    for i in range(len(point_list)):
        point_list[i][0] = point_list[i][0] * resolution[0]
        point_list[i][1] = point_list[i][1] * resolution[1]
        if ndim == 3:
            point_list[i][2] = point_list[i][2] * resolution[2]

    mynumbers = [tuple(point) for point in point_list]

    if len(mynumbers) == 0:
        return []

    tree = cKDTree(mynumbers)  # build k-dimensional tree

    pairs = tree.query_pairs(radius)  # find all pairs closer than radius

    neighbors = {}  # create dictionary of neighbors

    for i, j in pairs:  # iterate over all pairs
        if i not in neighbors:
            neighbors[i] = {j}
        else:
            neighbors[i].add(j)
        if j not in neighbors:
            neighbors[j] = {i}
        else:
            neighbors[j].add(i)

    positions = [i for i in range(0, len(point_list))]

    keep = []
    discard = set()
    for node in positions:
        if node not in discard:  # if node already in discard set: skip
            keep.append(node)  # add node to keep list
            discard.update(neighbors.get(node, set()))  # add node's neighbors to discard set

    # points to keep
    new_point_list = [list(points[i]) for i in keep]
    print("Final number of points: " + str(len(new_point_list)))

    if classes:
        new_class_list = [classes[i] for i in keep]
        if return_drops:
            return new_point_list, new_class_list, list(discard)
        else:
            return new_point_list, new_class_list
    else:
        if return_drops:
            return new_point_list, list(discard)
        else:
            return new_point_list

def detection_watershed(
    seeds: NDArray,
    coords: List[List[int | float]],
    data_filename: str,
    first_dilation: List[float|int],
    ndim: int=3,
    donuts_classes: List[int]=[-1],
    donuts_patch: List[int]=[13, 120, 120],
    donuts_nucleus_diameter: int=30,
    save_dir: Optional[str] = None,
) -> NDArray:
    """
    Grow given detection seeds.

    Parameters
    ----------
    seeds : 3D/4D Numpy array
        Binary foreground labels and contours data to apply watershed into. E.g. ``(1450, 2000, 1)``
        for ``2D`` and ``(397, 1450, 2000, 1)`` for ``3D``.

    coords : List of list of 3 ints
        Coordinates of all detected points.

    data_filename : str
        Path to load the image paired with seeds.

    first_dilation : List of float
        Seed dilation before watershed.

    ndim : int, optional
        Number of dimensions. E.g. for ``2D`` set it to ``2`` and for ``3D`` to ``3``.

    donuts_classes : List of ints, optional
        Classes to check a donuts type cell. Set to ``-1`` to disable it.

    donuts_patch : List of ints, optional
        Patch to analize donuts cells. Give a shape that covers all sizes of this type of
        cells.

    donuts_nucleus_diameter : int, optional
        Aproximate nucleus diameter for donuts type cells.

    save_dir :  str, optional
        Directory to save watershed output into.

    Returns
    -------
    segm : 2D/3D Numpy array
        Image with Voronoi applied. E.g. ``(y, x)`` for ``2D`` and ``(z, y, x)`` for ``3D``.
    """
    print("Applying detection watershed . . .")

    # Read the test image
    img = read_img_as_ndarray(data_filename, is_3d=(ndim == 3)).squeeze()
    img = reduce_dtype(img, np.min(img), np.max(img), out_min=0, out_max=255, out_type="uint8")
    img = equalize_adapthist(img)

    # Dilate first the seeds if needed
    print("Dilating a bit the seeds . . .")
    seeds = seeds.squeeze()
    if all(x != 0 for x in first_dilation):
        seeds += (binary_dilation(seeds, structure=np.ones(first_dilation))).astype(np.uint8) # type: ignore

    # Background seed
    seeds = label(seeds) # type: ignore
    max_seed = np.max(seeds)
    if max_seed < 255:
        seeds = seeds.astype(np.uint8)
    else:
        seeds = seeds.astype(np.uint16)

    if ndim == 2:
        seeds[:4, :4] = max_seed + 1
        background_label = seeds[1, 1]
    else:
        seeds[0, :4, :4] = max_seed + 1
        background_label = seeds[0, 1, 1]

    # Try to dilate those instances that have 'donuts' like shape and that might have problems with the watershed
    if donuts_classes[0] != -1:
        for dclass in donuts_classes:
            nticks = [x // 8 for x in donuts_patch]
            nticks = [x + (1 - x % 2) for x in nticks]
            half_spatch = [x // 2 for x in donuts_patch]

            for i in tqdm(range(len(coords)), leave=False):
                c = coords[i]

                # Patch coordinates
                l = seeds[c[0], c[1], c[2]] # type: ignore
                if ndim == 2:
                    y1, y2 = max(c[0] - half_spatch[0], 0), min(c[0] + half_spatch[0], img.shape[0])
                    x1, x2 = max(c[1] - half_spatch[1], 0), min(c[1] + half_spatch[1], img.shape[1])
                    img_patch = img[y1:y2, x1:x2]
                    seed_patch = seeds[y1:y2, x1:x2]

                    # Extract horizontal and vertical line
                    line_y = img_patch[:, half_spatch[1]]
                    line_x = img_patch[half_spatch[0], :]
                else:
                    z1, z2 = max(c[0] - half_spatch[0], 0), min(c[0] + half_spatch[0], img.shape[0])
                    y1, y2 = max(c[1] - half_spatch[1], 0), min(c[1] + half_spatch[1], img.shape[1])
                    x1, x2 = max(c[2] - half_spatch[2], 0), min(c[2] + half_spatch[2], img.shape[2])
                    img_patch = img[z1:z2, y1:y2, x1:x2]
                    seed_patch = seeds[z1:z2, y1:y2, x1:x2]

                    # Extract horizontal and vertical line
                    line_y = img_patch[half_spatch[0], :, half_spatch[2]]
                    line_x = img_patch[half_spatch[0], half_spatch[1], :]

                fillable_patch = seed_patch.copy()
                seed_patch = (seed_patch == l) * l
                fillable_patch = fillable_patch == 0

                if save_dir:
                    aux = np.expand_dims(np.expand_dims((img_patch).astype(np.float32), -1), 0)
                    save_tif(aux, save_dir, ["{}_patch.tif".format(l)], verbose=False)

                    # Save the verticial and horizontal lines in the patch to debug
                    patch_y = np.zeros(img_patch.shape, dtype=np.float32)
                    if ndim == 2:
                        patch_y[:, half_spatch[1]] = img_patch[:, half_spatch[1]]
                    else:
                        patch_y[half_spatch[0], :, half_spatch[2]] = img_patch[half_spatch[0], :, half_spatch[2]]

                    aux = np.expand_dims(np.expand_dims((patch_y).astype(np.float32), -1), 0)
                    save_tif(aux, save_dir, ["{}_y_line.tif".format(l)], verbose=False)

                    patch_x = np.zeros(img_patch.shape, dtype=np.float32)
                    if ndim == 2:
                        patch_x[half_spatch[0], :] = img_patch[half_spatch[0], :]
                    else:
                        patch_x[half_spatch[0], half_spatch[1], :] = img_patch[half_spatch[0], half_spatch[1], :]
                    aux = np.expand_dims(np.expand_dims((patch_x).astype(np.float32), -1), 0)
                    save_tif(aux, save_dir, ["{}_x_line.tif".format(l)], verbose=False)
                    # Save vertical and horizontal line plots to debug
                    plt.title("Line graph")
                    plt.plot(list(range(len(line_y))), line_y, color="red")
                    plt.savefig(os.path.join(save_dir, "{}_line_y.png".format(l)))
                    plt.clf()
                    plt.title("Line graph")
                    plt.plot(list(range(len(line_x))), line_x, color="red")
                    plt.savefig(os.path.join(save_dir, "{}_line_x.png".format(l)))
                    plt.clf()

                # Smooth them to analize easily
                line_y = savgol_filter(line_y, nticks[1], 2)
                line_x = savgol_filter(line_x, nticks[2], 2)

                if save_dir:
                    # Save vertical and horizontal lines again but now filtered
                    plt.title("Line graph")
                    plt.plot(list(range(len(line_y))), line_y, color="red")
                    plt.savefig(os.path.join(save_dir, "{}_line_y_filtered.png".format(l)))
                    plt.clf()
                    plt.title("Line graph")
                    plt.plot(list(range(len(line_x))), line_x, color="red")
                    plt.savefig(os.path.join(save_dir, "{}_line_x_filtered.png".format(l)))
                    plt.clf()

                # Find maximums
                peak_y, _ = find_peaks(line_y)
                peak_x, _ = find_peaks(line_x)

                # Find minimums
                mins_y, _ = find_peaks(-line_y)
                mins_x, _ = find_peaks(-line_x)

                # Find the donuts shape cells
                # Vertical line
                mid = len(line_y) // 2
                mid_value = line_y[min(mins_y, key=lambda x: abs(x - mid))]
                found_left_peak, found_right_peak = False, False
                max_right, max_left = 0.0, 0.0
                max_right_pos, max_left_pos = -1, -1
                for peak_pos in peak_y:
                    if line_y[peak_pos] >= mid_value * 1.5:
                        # Left side
                        if peak_pos <= mid:
                            found_left_peak = True
                            if line_y[peak_pos] > max_left:
                                max_left = line_y[peak_pos]
                                max_left_pos = peak_pos
                        # Right side
                        else:
                            found_right_peak = True
                            if line_y[peak_pos] > max_right:
                                max_right = line_y[peak_pos]
                                max_right_pos = peak_pos
                ushape_in_liney = found_left_peak and found_right_peak
                y_diff_dilation = max_right_pos - max_left_pos
                if ushape_in_liney:
                    y_left_gradient = min(line_y[:max_left_pos]) < max_left * 0.7
                    y_right_gradient = min(line_y[max_right_pos:]) < max_right * 0.7

                # Horizontal line
                mid = len(line_x) // 2
                mid_value = line_x[min(mins_x, key=lambda x: abs(x - mid))]
                found_left_peak, found_right_peak = False, False
                max_right, max_left = 0.0, 0.0
                max_right_pos, max_left_pos = -1, -1
                for peak_pos in peak_x:
                    if line_x[peak_pos] >= mid_value * 1.5:
                        # Left side
                        if peak_pos <= mid:
                            found_left_peak = True
                            if line_x[peak_pos] > max_left:
                                max_left = line_x[peak_pos]
                                max_left_pos = peak_pos
                        # Right side
                        else:
                            found_right_peak = True
                            if line_x[peak_pos] > max_right:
                                max_right = line_x[peak_pos]
                                max_right_pos = peak_pos
                ushape_in_linex = found_left_peak and found_right_peak
                x_diff_dilation = max_right_pos - max_left_pos
                if ushape_in_linex:
                    x_left_gradient = min(line_x[:max_left_pos]) < max_left * 0.7
                    x_right_gradient = min(line_x[max_right_pos:]) < max_right * 0.7

                # Donuts shape cell found
                if ushape_in_liney and ushape_in_linex:
                    # Calculate the dilation to be made based on the nucleus size
                    if ndim == 2:
                        donuts_cell_dilation = [
                            y_diff_dilation - first_dilation[0],
                            x_diff_dilation - first_dilation[1],
                        ]
                        donuts_cell_dilation = [
                            donuts_cell_dilation[0] - int(donuts_cell_dilation[0] * 0.4),
                            donuts_cell_dilation[1] - int(donuts_cell_dilation[1] * 0.4),
                        ]
                    else:
                        donuts_cell_dilation = [
                            first_dilation[0],
                            y_diff_dilation - first_dilation[1],
                            x_diff_dilation - first_dilation[2],
                        ]
                        donuts_cell_dilation = [
                            donuts_cell_dilation[0],
                            donuts_cell_dilation[1] - int(donuts_cell_dilation[1] * 0.4),
                            donuts_cell_dilation[2] - int(donuts_cell_dilation[2] * 0.4),
                        ]

                    # If the center is not wide the cell is not very large
                    dilate = True
                    if x_diff_dilation + y_diff_dilation < donuts_nucleus_diameter * 2:
                        print("Instance {} has 'donuts' shape but it seems to be not very large!".format(l))
                    else:
                        print("Instance {} has 'donuts' shape!".format(l))
                        if not y_left_gradient:
                            print("    - Its vertical left part seems to have low gradient")
                            dilate = False
                        if not y_right_gradient:
                            print("    - Its vertical right part seems to have low gradient")
                            dilate = False
                        if not x_left_gradient:
                            print("    - Its horizontal left part seems to have low gradient")
                            dilate = False
                        if not x_right_gradient:
                            print("    - Its horizontal right part seems to have low gradient")
                            dilate = False
                    if dilate:
                        if all(x > 0 for x in donuts_cell_dilation):
                            seed_patch = grey_dilation(seed_patch, footprint=np.ones((donuts_cell_dilation))) # type: ignore
                            if ndim == 2:
                                seeds[y1:y2, x1:x2] += seed_patch * fillable_patch
                            else:
                                seeds[z1:z2, y1:y2, x1:x2] += seed_patch * fillable_patch
                    else:
                        print("    - Not dilating it!")
                else:
                    print("Instance {} checked".format(l))

    print("Calculating gradient . . .")
    start = time.time()
    if ndim == 2:
        gradient = rank.gradient(img, disk(3)).astype(np.uint8)
    else:
        gradient = rank.gradient(img, ball(3)).astype(np.uint8)
    end = time.time()
    grad_elapsed = end - start
    print("Gradient took {} seconds".format(int(grad_elapsed)))

    print("Run watershed . . .")
    segm = watershed(gradient, seeds)

    # Remove background label
    segm[segm == background_label] = 0

    # Dilate a bit the instances
    if ndim == 2:
        segm += dilation(segm, disk(5)) * (segm == 0)
        segm = erosion(segm, disk(3))
    else:
        for i in range(segm.shape[0]):
            dil_slice = segm[i] == 0
            dil_slice = dilation(segm[i], disk(5)) * dil_slice
            segm[i] += dil_slice
            segm[i] = erosion(segm[i], disk(2))

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        aux = np.expand_dims(np.expand_dims((img).astype(np.float32), -1), 0)
        save_tif(aux, save_dir, ["img.tif"], verbose=False)

        aux = np.expand_dims(np.expand_dims((gradient).astype(np.float32), -1), 0)
        save_tif(aux, save_dir, ["gradient.tif"], verbose=False)

        aux = np.expand_dims(np.expand_dims((seeds).astype(np.float32), -1), 0)
        save_tif(aux, save_dir, ["seed_map.tif"], verbose=False)

        aux = np.expand_dims(np.expand_dims((segm).astype(np.float32), -1), 0)
        save_tif(aux, save_dir, ["watershed.tif"], verbose=False)

    return segm


def measure_morphological_props_and_filter(
    img: NDArray,
    resolution: List[float|int],
    filter_instances: bool=False,
    properties: List[List[str]]=[[]],
    prop_values=[[]],
    comp_signs=[[]],
):
    """
    Measures the properties of input image's instances. It calculates each instance id, number of pixels, area/volume
    (2D/3D respec. and taking into account the ``resolution``), diameter, perimeter/surface_area (2D/3D respec.),
    circularity/sphericity (2D/3D respec.) and elongation properties. All instances that satisfy the conditions composed
    by ``properties``, ``prop_values`` and ``comp_signs`` variables will be removed from ``img``. Apart from returning
    all properties this function will return also a list identifying those instances that satisfy and not satify the
    conditions. Those removed will be marked as 'Removed' whereas the rest are 'Correct'. Some of the properties follow
    the formulas used in `MorphoLibJ library for Fiji <https://doi.org/10.1093/bioinformatics/btw413>`__.

    Parameters
    ----------
    img : 2D/3D Numpy array
        Image with instances. E.g. ``(1450, 2000)`` for 2D and ``(397, 1450, 2000)`` for 3D.

    resolution : tuple of int/float
        Resolution of the data.

    filter_instances : bool, optional
        Whether to do instance filtering or not.

    properties : List of lists of str, optional
        List of lists of properties to remove the instances. Options available: ``['circularity', 'npixels', 'area', 'diameter',
        'elongation', 'sphericity', 'perimeter']``. E.g. ``[['size'], ['circularity', 'npixels']]``.

    prop_values : List of lists of floats/ints, optional
        List of lists of values for each property. E.g. ``[[70], [0.7, 2000]]``.

    comp_signs : List of list of str, optional
        List of lists of signs to compose the conditions, together ``properties`` ``prop_values``, that the instances must
        satify to be removed from the input ``img``. E.g. ``[['le'], ['lt', 'ge']]``.

    Returns
    -------
    img : 2D/3D Numpy array
        Input image without the instances that do not satisfy the circularity constraint.
        Image with instances. E.g. ``(1450, 2000)`` for 2D and ``(397, 1450, 2000)`` for 3D.

    d_result : dict
        Results of the morphological measurements. All the information of the non-filtered
        instances (if declared to do so) are listed. It contains:

        labels : Array of ints
            Instance label list.

        centers : Array of ints
            Coordinates of the centers of each instance.

        npixels : Array of ints
            Number of pixels of each instance.

        areas : Array of ints
            Area/volume (2D/3D) of each instance based on the given ``resolution``.

        circularities : Array of ints
            Circularity/sphericity (2D/3D) of each instance. In 2D, ``circularity`` of an instance is defined
            as the ratio of area over the square of the perimeter, normalized such that the value
            for a disk equals one: ``(4 * PI * area) / (perimeter^2)``. While values of circularity
            range theoretically within the interval [0;1], the measurements errors of the perimeter
            may produce circularity values above 1 (`Lehmann et al. <https://doi.org/10.1093/bioinformatics/btw413>`__).
            In 3D, ``sphericity`` is is the ratio of the squared volume over the cube of the surface area,
            normalized such that the value for a ball equals one: ``(36 * PI)*((volume^2)/(perimeter^3))``.

        diameters : Array of ints
            Diameter of each instance obtained from the bounding box.

        elongations : Array of ints
            Elongation of each instance. It is the inverse of the circularity. The values of elongation range from
            ``1`` for round particles and increase for elongated particles. In 2D it is calculated as:
            ``(perimeter^2)/(4 * PI * area)``. In 3D: ``(sqrt(surface area^3))/ (6 * volume * sqrt(PI))`` where ``sqrt``
            is the square root. For the 3D `diplib library <https://diplib.org/diplib-docs/features.html#shape_features_P2A>`__
            is used (corresponds to 'P2A' metric in diplib).

        perimeter : Array of ints
            In 2D, approximates the contour as a line through the centers of border pixels using a 4-connectivity.
            In 3D, it is the surface area computed using
            `the diplib library <https://diplib.org/diplib-docs/features.html#shape_features_P2A>`__.

        comment : List of str
            List containing 'Correct' string when the instance surpass the circularity threshold and 'Removed'
            otherwise.

        conditions : List of str
            List of conditions that each instance has satisfy or not.
    """
    print("Checking the properties of instances . . .")
    
    assert len(resolution) == 3, "'resolution' must be a list of 3 int/float" 

    image3d = True if img.ndim == 3 else False
    correct_str = "Correct"
    unsure_str = "Removed"

    comment = []
    label_list = []
    label_list, npixels = np.unique(img, return_counts=True)

    # Delete background instance '0'
    if label_list[0] == 0:
        label_list = label_list[1:]
        npixels = npixels[1:]

    total_labels = len(label_list)
    comment = ["none" for i in range(total_labels)]

    # Measure array definitions
    areas = np.zeros(total_labels, dtype=np.uint32)
    diameters = np.zeros(total_labels, dtype=np.uint32)
    centers = np.zeros((total_labels, 3 if image3d else 2), dtype=np.uint16)
    circularities = np.zeros(total_labels, dtype=np.float32)
    perimeters = np.zeros(total_labels, dtype=np.uint32)
    elongations = np.zeros(total_labels, dtype=np.float32)

    # Area, diameter, center, circularity (if 2D), elongation (if 2D) and perimeter (if 2D) calculation over the whole image
    lprops = ["label", "bbox", "perimeter"] if not image3d else ["label", "bbox"]
    props = regionprops_table(img, properties=(lprops))
    for k, l in tqdm(enumerate(props["label"]), total=len(props["label"]), leave=False):
        label_index = np.where(label_list == l)[0]
        pixels = npixels[label_index]

        if image3d:
            vol = pixels * (resolution[0] * resolution[1] * resolution[2])
            diam = max(
                props["bbox-3"][k] - props["bbox-0"][k],
                props["bbox-4"][k] - props["bbox-1"][k],
                props["bbox-5"][k] - props["bbox-2"][k],
            )
            center = [
                props["bbox-0"][k] + ((props["bbox-3"][k] - props["bbox-0"][k]) // 2),
                props["bbox-1"][k] + ((props["bbox-4"][k] - props["bbox-1"][k]) // 2),
                props["bbox-2"][k] + ((props["bbox-5"][k] - props["bbox-2"][k]) // 2),
            ]
        else:
            vol = pixels * (resolution[0] * resolution[1])
            diam = max(
                props["bbox-2"][k] - props["bbox-0"][k],
                props["bbox-3"][k] - props["bbox-1"][k],
            )
            center = [
                props["bbox-0"][k] + ((props["bbox-2"][k] - props["bbox-0"][k]) // 2),
                props["bbox-1"][k] + ((props["bbox-3"][k] - props["bbox-1"][k]) // 2),
            ]
            perimeter = props["perimeter"][k]
            elongations[label_index] = (perimeter * perimeter) / (4 * math.pi * pixels) if pixels > 0 else 0
            circularity = (4 * math.pi * pixels) / (perimeter * perimeter) if perimeter > 0 else 0

            perimeters[label_index] = perimeter
            circularities[label_index] = circularity

        areas[label_index] = vol
        diameters[label_index] = diam
        centers[label_index] = center

    if total_labels > 0:
        img = dip.Image(img.astype(img.dtype.name)) # type: ignore

        features = ["SurfaceArea", "P2A"] if image3d else ["P2A"]
        measurement = dip.MeasurementTool.Measure(img, features=features) # type: ignore

        for lbl in measurement.Objects():
            label_index = np.where(label_list == lbl)[0]
            elongations[label_index] = measurement["P2A"][lbl]
            if image3d:
                perimeters[label_index] = measurement["SurfaceArea"][lbl]
                pixels = npixels[label_index]
                sphericity = (
                    (36 * math.pi * pixels**2) / (perimeters[label_index] ** 3) if perimeters[label_index] > 0 else 0
                )
                circularities[label_index] = sphericity

        # Convert diplib.PyDIP_bin.Image back into numpy array
        img = np.array(img)

    # Remove those instances that do not satisfy the properties
    conditions = []
    labels_removed = 0
    for i in tqdm(range(len(circularities)), leave=False):
        conditions.append([])
        if filter_instances:
            for k, list_of_conditions in enumerate(properties):
                # Check each list of conditions
                comps = []
                for j, prop in enumerate(list_of_conditions):
                    if prop in ["circularity", "sphericity"]:
                        value_to_compare = circularities[i]
                    elif prop == "npixels":
                        value_to_compare = npixels[i]
                    elif prop == "area":
                        value_to_compare = areas[i]
                    elif prop == "diameter":
                        value_to_compare = diameters[i]
                    elif prop == "perimeter":
                        value_to_compare = perimeters[i]
                    elif prop == "elongation":
                        value_to_compare = elongations[i]

                    if comp_signs[k][j] == "gt":
                        if value_to_compare > prop_values[k][j]:
                            comps.append(True)
                        else:
                            comps.append(False)
                    elif comp_signs[k][j] == "ge":
                        if value_to_compare >= prop_values[k][j]:
                            comps.append(True)
                        else:
                            comps.append(False)
                    elif comp_signs[k][j] == "lt":
                        if value_to_compare < prop_values[k][j]:
                            comps.append(True)
                        else:
                            comps.append(False)
                    elif comp_signs[k][j] == "le":
                        if value_to_compare <= prop_values[k][j]:
                            comps.append(True)
                        else:
                            comps.append(False)

                # Check if the conditions where satified
                if all(comps):
                    conditions[-1].append(True)
                else:
                    conditions[-1].append(False)

        # If satisfied all conditions remove the instance
        if any(conditions[-1]):
            comment[i] = unsure_str
            img[img == label_list[i]] = 0
            labels_removed += 1
        else:
            comment[i] = correct_str
    cir_name = "sphericities" if image3d else "circularities"
    d_result = {
        "labels": label_list,
        "centers": centers,
        "npixels": npixels,
        "areas": areas,
        cir_name: circularities,
        "diameters": diameters,
        "perimeters": perimeters,
        "elongations": elongations,
        "comment": comment,
        "conditions": conditions,
    }

    print(
        "Removed {} instances by properties ({}), {} instances left".format(
            labels_removed, properties, total_labels - labels_removed
        )
    )

    return img, d_result


def find_neighbors(
    img: NDArray, 
    label: int, 
    neighbors: int=1
):
    """
    Find neighbors of a label in a given image.

    Parameters
    ----------
    img : 2D/3D Numpy array
        Image with instances. E.g. ``(1450, 2000)`` for 2D and ``(397, 1450, 2000)`` for 3D.

    label : int
        Label to find the neighbors of.

    neighbors : int, optional
        Number of neighbors in each axis to explore.

    Returns
    -------
    neighbors  : list of ints
        Neighbors instance ids of the given label.
    """

    list_of_neighbors = []
    label_points = np.where((img == label) > 0)
    if img.ndim == 3:
        for p in range(len(label_points[0])):
            coord = [label_points[0][p], label_points[1][p], label_points[2][p]]
            for i in range(-neighbors, neighbors + 1):
                for j in range(-neighbors, neighbors + 1):
                    for k in range(-neighbors, neighbors + 1):
                        z = min(max(coord[0] + i, 0), img.shape[0] - 1)
                        y = min(max(coord[1] + j, 0), img.shape[1] - 1)
                        x = min(max(coord[2] + k, 0), img.shape[2] - 1)
                        if img[z, y, x] not in list_of_neighbors and img[z, y, x] != label and img[z, y, x] != 0:
                            list_of_neighbors.append(img[z, y, x])
    else:
        for p in range(len(label_points[0])):
            coord = [label_points[0][p], label_points[1][p]]
            for i in range(-neighbors, neighbors + 1):
                for j in range(-neighbors, neighbors + 1):
                    y = min(max(coord[0] + i, 0), img.shape[0] - 1)
                    x = min(max(coord[1] + j, 0), img.shape[1] - 1)
                    if img[y, x] not in list_of_neighbors and img[y, x] != label and img[y, x] != 0:
                        list_of_neighbors.append(img[y, x])
    return list_of_neighbors


def repare_large_blobs(
    img: NDArray, 
    size_th: int=10000
):
    """
    Try to repare large instances by merging neighbors ones with it and by removing possible central holes.

    Parameters
    ----------
    img : 2D/3D Numpy array
        Image with instances. E.g. ``(1450, 2000)`` for 2D and ``(397, 1450, 2000)`` for 3D.

    size_th : int, optional
        Size that the instances need to be larger than to be analised.

    Returns
    -------
    img : 2D/3D Numpy array
        Input image without the large instances repaired. E.g. ``(1450, 2000)`` for 2D and
        ``(397, 1450, 2000)`` for 3D.
    """
    print("Reparing large instances (more than {} pixels) . . .".format(size_th))
    image3d = True if img.ndim == 3 else False

    props = regionprops_table(img, properties=("label", "area", "bbox"))
    for k, l in tqdm(enumerate(props["label"]), total=len(props["label"]), leave=False):
        if props["area"][k] >= size_th:
            if image3d:
                sz, fz, sy, fy, sx, fx = (
                    props["bbox-0"][k],
                    props["bbox-3"][k],
                    props["bbox-1"][k],
                    props["bbox-4"][k],
                    props["bbox-2"][k],
                    props["bbox-5"][k],
                )
                patch = img[sz:fz, sy:fy, sx:fx].copy()
            else:
                sy, fy, sx, fx = (
                    props["bbox-0"][k],
                    props["bbox-2"][k],
                    props["bbox-1"][k],
                    props["bbox-3"][k],
                )
                patch = img[sy:fy, sx:fx].copy()

            inst_patches, inst_pixels = np.unique(patch, return_counts=True)
            if len(inst_patches) > 2:
                neighbors = find_neighbors(patch, l)

                # Merge neighbors with the big label
                for i in range(len(neighbors)):
                    ind = np.where(props["label"] == neighbors[i])[0]

                    # Only merge labels if the small neighbor instance is fully contained in the large one
                    contained_in_large_blob = True
                    if image3d:
                        neig_sz, neig_fz = props["bbox-0"][ind], props["bbox-3"][ind]
                        neig_sy, neig_fy = props["bbox-1"][ind], props["bbox-4"][ind]
                        neig_sx, neig_fx = props["bbox-2"][ind], props["bbox-5"][ind]

                        if neig_sz < sz or neig_fz > fz or neig_sy < sy or neig_fy > fy or neig_sx < sx or neig_fx > fx:
                            neigbor_ind_in_patch = list(inst_patches).index(neighbors[i])
                            pixels_in_patch = inst_pixels[neigbor_ind_in_patch]
                            # pixels outside the patch of that neighbor are greater than 30% means that probably it will
                            # represent another blob so do not merge
                            if (props["area"][ind][0] - pixels_in_patch) / props["area"][ind][0] > 0.30:
                                contained_in_large_blob = False
                    else:
                        neig_sy, neig_fy = props["bbox-0"][ind], props["bbox-2"][ind]
                        neig_sx, neig_fx = props["bbox-1"][ind], props["bbox-3"][ind]
                        if neig_sy < sy or neig_fy > fy or neig_sx < sx or neig_fx > fx:
                            contained_in_large_blob = False

                    if contained_in_large_blob:
                        img[img == neighbors[i]] = l

            # Fills holes
            if image3d:
                patch = img[sz:fz, sy:fy, sx:fx].copy()
            else:
                patch = img[sy:fy, sx:fx].copy()
            only_label_patch = patch.copy()
            only_label_patch[only_label_patch != l] = 0
            if image3d:
                for i in range(only_label_patch.shape[0]):
                    only_label_patch[i] = fill_voids.fill(only_label_patch[i]) * l
            else:
                only_label_patch = fill_voids.fill(only_label_patch) * l
            patch[patch == l] = 0
            patch += only_label_patch
            if image3d:
                img[sz:fz, sy:fy, sx:fx] = patch
            else:
                img[sy:fy, sx:fx] = patch
    return img


def apply_binary_mask(
    X: NDArray, 
    bin_mask_dir: str
):
    """
    Apply a binary mask to remove values outside it.

    Parameters
    ----------
    X : 3D/4D Numpy array
        Data to apply the mask. E.g. ``(y, x, channels)`` for 2D or ``(z, y, x, channels)`` for 3D.

    bin_mask_dir : str, optional
        Directory where the binary mask are located.

    Returns
    -------
    X : 3D/4D Numpy array
        Data with the mask applied. E.g. ``(y, x, channels)`` for 2D or ``(z, y, x, channels)`` for 3D.
    """

    if X.ndim != 4 and X.ndim != 3:
        raise ValueError("'X' needs to have 3 or 4 dimensions and not {}".format(X.ndim))

    print("Applying binary mask(s) from {}".format(bin_mask_dir))

    ids = sorted(next(os.walk(bin_mask_dir))[2])

    if len(ids) == 1:
        one_file = True
        print(
            "It is assumed that the mask found {} is valid for all 'X' data".format(os.path.join(bin_mask_dir, ids[0]))
        )
    else:
        one_file = False

    if one_file:
        mask = imread(os.path.join(bin_mask_dir, ids[0]))
        mask = np.squeeze(mask)

        if X.ndim != mask.ndim + 1 and X.ndim != mask.ndim + 2:
            raise ValueError(
                "Mask found has {} dims, shape: {}. Need to be of {} or {} dims instead".format(
                    mask.ndim, mask.shape, mask.ndim + 1, mask.ndim + 2
                )
            )

        if mask.ndim == X.ndim - 1:
            for c in range(X.shape[-1]):
                X[..., c] = X[..., c] * (mask > 0)
        else:  # mask.ndim == 2 and X.ndim == 4
            for k in range(X.shape[0]):
                for c in range(X.shape[-1]):
                    X[k, ..., c] = X[k, ..., c] * (mask > 0)
    else:
        for i in tqdm(range(len(ids))):
            mask = imread(os.path.join(bin_mask_dir, ids[i]))
            mask = np.squeeze(mask)

            if X.ndim != mask.ndim + 1 and X.ndim != mask.ndim + 2:
                raise ValueError(
                    "Mask found has {} dims, shape: {}. Need to be of {} or {} dims instead".format(
                        mask.ndim, mask.shape, mask.ndim + 1, mask.ndim + 2
                    )
                )

            if mask.ndim == X.ndim - 1:
                for c in range(X.shape[-1]):
                    X[..., c] = X[..., c] * (mask > 0)
            else:  # mask.ndim == 2 and X.ndim == 4
                for k in range(X.shape[0]):
                    for c in range(X.shape[-1]):
                        X[k, ..., c] = X[k, ..., c] * (mask > 0)
    return X
