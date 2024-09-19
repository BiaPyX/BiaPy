import math
import h5py
import numpy as np
from tqdm import tqdm

from biapy.utils.util import (
    read_chunked_data,
    read_chunked_nested_data,
)
from biapy.utils.misc import is_main_process


def load_3D_efficient_files(
    data_path,
    input_axes,
    crop_shape,
    overlap,
    padding,
    check_channel=True,
    data_within_zarr_path=None,
):
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

    data_within_zarr_path : str, optional
        Path to find the data within the Zarr file. E.g. 'volumes.labels.neuron_ids'.

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
    assert len(crop_shape) == 4, f"Provided crop_shape is not a 4D tuple: {crop_shape}"

    for i, filename in enumerate(data_path):
        print(f"Reading Zarr/H5 file: {filename}")
        if data_within_zarr_path:
            file, data = read_chunked_nested_data(filename, data_within_zarr_path)
        else:
            file, data = read_chunked_data(filename)

        # Modify crop_shape with the channel
        c_index = -1
        try:
            c_index = input_axes.index("C")
            crop_shape = crop_shape[:-1] + (data.shape[c_index],)
        except:
            pass

        # Get the total patches so we can use tqdm so the user can see the time
        obj = extract_3D_patch_with_overlap_yield(
            data,
            crop_shape,
            input_axes,
            overlap=overlap,
            padding=padding,
            total_ranks=1,
            rank=0,
            return_only_stats=True,
            verbose=True,
        )
        __unnamed_iterator = iter(obj)
        while True:
            try:
                obj = next(__unnamed_iterator)
            except StopIteration:  # StopIteration caught here without inspecting it
                break
        del __unnamed_iterator
        total_patches, z_vol_info, list_of_vols_in_z = obj

        for obj in tqdm(
            extract_3D_patch_with_overlap_yield(
                data,
                crop_shape,
                input_axes,
                overlap=overlap,
                padding=padding,
                total_ranks=1,
                rank=0,
                verbose=False,
            ),
            total=total_patches,
            disable=not is_main_process(),
        ):

            img, patch_coords, _, _, _ = obj

            data_info[c] = {}
            data_info[c]["filepath"] = filename
            data_info[c]["patch_coords"] = order_dimensions(
                patch_coords,
                input_order="ZYX",
                output_order=input_axes,
                default_value=img.shape[c_index],
            )

            c += 1

            if check_channel and crop_shape[-1] != img.shape[-1]:
                raise ValueError(
                    "Channel of the patch size given {} does not correspond with the loaded image {}. "
                    "Please, check the channels of the images!".format(crop_shape[-1], img.shape[-1])
                )

        if isinstance(file, h5py.File):
            file.close()

        data_total_patches.append(total_patches)

    return data_info, data_total_patches


def load_img_part_from_efficient_file(filepath, patch_coords, data_axis_order="ZYXC", data_path=None):
    """
    Loads from ``filepath`` the patch determined by ``patch_coords``.

    Parameters
    ----------
    filepath : str
        Path to the Zarr/H5 file to read the patch from.

    patch_coords : list of dict
        Coordinates of the crop where the following keys are expected:
            * ``"z_start"``: starting point of the patch in Z axis,
            * ``"z_end"``: end point of the patch in Z axis,
            * ``"y_start"``: starting point of the patch in Y axis,
            * ``"y_end"``: end point of the patch in Y axis,
            * ``"x_start"``: starting point of the patch in X axis,
            * ``"x_end"``: end point of the patch in X axis,

    data_axis_order : str
        Order of axes of ``data``. E.g. 'TZCYX', 'TZYXC', 'ZCYX', 'ZYXC'.

    data_path : str, optional
        Path to find the data within the Zarr file. E.g. 'volumes.labels.neuron_ids'.

    Returns
    -------
    img : Numpy array
        Extracted patch. E.g. ``(z, y, x, channels)``.
    """
    if data_path is not None:
        imgfile, img = read_chunked_nested_data(filepath, data_path)
    else:
        imgfile, img = read_chunked_data(filepath)

    img = extract_patch_from_efficient_file(img, patch_coords, data_axis_order=data_axis_order)

    if isinstance(imgfile, h5py.File):
        imgfile.close()

    return img


def extract_patch_from_efficient_file(data, patch_coords, data_axis_order="ZYXC"):
    """
    Loads from ``filepath`` the patch determined by ``patch_coords``.

    Parameters
    ----------
    data : Zarr/H5 data
        Data to extract the patch from.

    patch_coords : dict
        Coordinates of the crop where the following keys are expected:
            * ``"z_start"``: starting point of the patch in Z axis.
            * ``"z_end"``: end point of the patch in Z axis.
            * ``"y_start"``: starting point of the patch in Y axis.
            * ``"y_end"``: end point of the patch in Y axis.
            * ``"x_start"``: starting point of the patch in X axis.
            * ``"x_end"``: end point of the patch in X axis.

    data_axis_order : str
        Order of axes of ``data``. E.g. 'TZCYX', 'TZYXC', 'ZCYX', 'ZYXC'.

    Returns
    -------
    img : Numpy array
        Extracted patch. E.g. ``(z, y, x, channels)``.
    """
    patch_coords = np.array(
        [
            [patch_coords["z_start"], patch_coords["z_end"]],
            [patch_coords["y_start"], patch_coords["y_end"]],
            [patch_coords["x_start"], patch_coords["x_end"]],
        ]
    )
    # Prepare slices to extract the patch
    slices = []
    for j in range(len(patch_coords)):
        if isinstance(patch_coords[j], int):
            # +1 to prevent 0 length axes that can not be removed with np.squeeze later
            slices.append(slice(0, patch_coords[j] + 1))
        else:
            slices.append(slice(patch_coords[j][0], patch_coords[j][1]))
    slices.append(slice(None)) # Channel
    
    # Convert slices into Zarr axis position
    data_ordered_slices = order_dimensions(
        tuple(slices), input_order="ZYXC", output_order=data_axis_order, default_value=0
    )

    # Extract patch
    img = np.squeeze(np.array(data[data_ordered_slices]))

    img = ensure_3d_shape(img.squeeze())

    return img


def crop_3D_data_with_overlap(
    data,
    vol_shape,
    data_mask=None,
    overlap=(0, 0, 0),
    padding=(0, 0, 0),
    verbose=True,
    median_padding=False,
    load_data=True,
):
    """
    Crop 3D data into smaller volumes with a defined overlap. The opposite function is :func:`~merge_3D_data_with_overlap`.

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

    load_data : bool, optional
        Whether to create the patches or not. It saves memory in case you only need the coordiantes of the cropped patches.

    Returns
    -------
    cropped_data : 5D Numpy array, optional
        Cropped image data. E.g. ``(vol_number, z, y, x, channels)``. Returned if ``load_data`` is ``True``.

    cropped_data_mask : 5D Numpy array, optional
        Cropped image data masks. E.g. ``(vol_number, z, y, x, channels)``. Returned if ``load_data`` is ``True``
        and ``data_mask`` is provided.

    crop_coords : list of dict
        Coordinates of each crop where the following keys are available:
            * ``"z_start"``: starting point of the patch in Z axis.
            * ``"z_end"``: end point of the patch in Z axis.
            * ``"y_start"``: starting point of the patch in Y axis.
            * ``"y_end"``: end point of the patch in Y axis.
            * ``"x_start"``: starting point of the patch in X axis.
            * ``"x_end"``: end point of the patch in X axis.

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
            raise ValueError(
                "data and data_mask shapes mismatch: {} vs {}".format(data.shape[:-1], data_mask.shape[:-1])
            )
    if len(vol_shape) != 4:
        raise ValueError("vol_shape expected to be of length 4, given {}".format(vol_shape))
    if vol_shape[0] > data.shape[0]:
        raise ValueError(
            "'vol_shape[0]' {} greater than {} (you can reduce 'DATA.PATCH_SIZE' or use 'DATA.REFLECT_TO_COMPLETE_SHAPE')".format(
                vol_shape[0], data.shape[0]
            )
        )
    if vol_shape[1] > data.shape[1]:
        raise ValueError(
            "'vol_shape[1]' {} greater than {} (you can reduce 'DATA.PATCH_SIZE' or use 'DATA.REFLECT_TO_COMPLETE_SHAPE')".format(
                vol_shape[1], data.shape[1]
            )
        )
    if vol_shape[2] > data.shape[2]:
        raise ValueError(
            "'vol_shape[2]' {} greater than {} (you can reduce 'DATA.PATCH_SIZE' or use 'DATA.REFLECT_TO_COMPLETE_SHAPE')".format(
                vol_shape[2], data.shape[2]
            )
        )
    if (
        (overlap[0] >= 1 or overlap[0] < 0)
        or (overlap[1] >= 1 or overlap[1] < 0)
        or (overlap[2] >= 1 or overlap[2] < 0)
    ):
        raise ValueError("'overlap' values must be floats between range [0, 1)")
    for i, p in enumerate(padding):
        if p >= vol_shape[i] // 2:
            raise ValueError(
                "'Padding' can not be greater than the half of 'vol_shape'. Max value for this {} input shape is {}".format(
                    vol_shape,
                    [
                        (vol_shape[0] // 2) - 1,
                        (vol_shape[1] // 2) - 1,
                        (vol_shape[2] // 2) - 1,
                    ],
                )
            )

    padded_data = np.pad(
        data,
        (
            (padding[0], padding[0]),
            (padding[1], padding[1]),
            (padding[2], padding[2]),
            (0, 0),
        ),
        "reflect",
    )
    if data_mask is not None:
        padded_data_mask = np.pad(
            data_mask,
            (
                (padding[0], padding[0]),
                (padding[1], padding[1]),
                (padding[2], padding[2]),
                (0, 0),
            ),
            "reflect",
        )
    if median_padding:
        padded_data[0 : padding[0], :, :, :] = np.median(data[0, :, :, :])
        padded_data[padding[0] + data.shape[0] : 2 * padding[0] + data.shape[0], :, :, :] = np.median(data[-1, :, :, :])
        padded_data[:, 0 : padding[1], :, :] = np.median(data[:, 0, :, :])
        padded_data[:, padding[1] + data.shape[1] : 2 * padding[1] + data.shape[0], :, :] = np.median(data[:, -1, :, :])
        padded_data[:, :, 0 : padding[2], :] = np.median(data[:, :, 0, :])
        padded_data[:, :, padding[2] + data.shape[2] : 2 * padding[2] + data.shape[2], :] = np.median(data[:, :, -1, :])
    padded_vol_shape = vol_shape

    # Calculate overlapping variables
    overlap_z = 1 if overlap[0] == 0 else 1 - overlap[0]
    overlap_y = 1 if overlap[1] == 0 else 1 - overlap[1]
    overlap_x = 1 if overlap[2] == 0 else 1 - overlap[2]

    # Z
    step_z = int((vol_shape[0] - padding[0] * 2) * overlap_z)
    vols_per_z = math.ceil(data.shape[0] / step_z)
    last_z = 0 if vols_per_z == 1 else (((vols_per_z - 1) * step_z) + vol_shape[0]) - padded_data.shape[0]
    ovz_per_block = last_z // (vols_per_z - 1) if vols_per_z > 1 else 0
    step_z -= ovz_per_block
    last_z -= ovz_per_block * (vols_per_z - 1)

    # Y
    step_y = int((vol_shape[1] - padding[1] * 2) * overlap_y)
    vols_per_y = math.ceil(data.shape[1] / step_y)
    last_y = 0 if vols_per_y == 1 else (((vols_per_y - 1) * step_y) + vol_shape[1]) - padded_data.shape[1]
    ovy_per_block = last_y // (vols_per_y - 1) if vols_per_y > 1 else 0
    step_y -= ovy_per_block
    last_y -= ovy_per_block * (vols_per_y - 1)

    # X
    step_x = int((vol_shape[2] - padding[2] * 2) * overlap_x)
    vols_per_x = math.ceil(data.shape[2] / step_x)
    last_x = 0 if vols_per_x == 1 else (((vols_per_x - 1) * step_x) + vol_shape[2]) - padded_data.shape[2]
    ovx_per_block = last_x // (vols_per_x - 1) if vols_per_x > 1 else 0
    step_x -= ovx_per_block
    last_x -= ovx_per_block * (vols_per_x - 1)

    # Real overlap calculation for printing
    real_ov_z = ovz_per_block / (vol_shape[0] - padding[0] * 2)
    real_ov_y = ovy_per_block / (vol_shape[1] - padding[1] * 2)
    real_ov_x = ovx_per_block / (vol_shape[2] - padding[2] * 2)
    if verbose:
        print("Real overlapping (%): {}".format((real_ov_z, real_ov_y, real_ov_x)))
        print(
            "Real overlapping (pixels): {}".format(
                (
                    (vol_shape[0] - padding[0] * 2) * real_ov_z,
                    (vol_shape[1] - padding[1] * 2) * real_ov_y,
                    (vol_shape[2] - padding[2] * 2) * real_ov_x,
                )
            )
        )
        print("{} patches per (z,y,x) axis".format((vols_per_z, vols_per_x, vols_per_y)))

    total_vol = vols_per_z * vols_per_y * vols_per_x
    if load_data:
        cropped_data = np.zeros((total_vol,) + padded_vol_shape, dtype=data.dtype)
        if data_mask is not None:
            cropped_data_mask = np.zeros(
                (total_vol,) + padded_vol_shape[:3] + (data_mask.shape[-1],),
                dtype=data_mask.dtype,
            )

    c = 0
    crop_coords = []
    for z in range(vols_per_z):
        for y in range(vols_per_y):
            for x in range(vols_per_x):
                d_z = 0 if (z * step_z + vol_shape[0]) < padded_data.shape[0] else last_z
                d_y = 0 if (y * step_y + vol_shape[1]) < padded_data.shape[1] else last_y
                d_x = 0 if (x * step_x + vol_shape[2]) < padded_data.shape[2] else last_x

                if load_data:
                    cropped_data[c] = padded_data[
                        z * step_z - d_z : z * step_z + vol_shape[0] - d_z,
                        y * step_y - d_y : y * step_y + vol_shape[1] - d_y,
                        x * step_x - d_x : x * step_x + vol_shape[2] - d_x,
                    ]

                crop_coords.append(
                    {
                        "z_start": z * step_z - d_z,
                        "z_end": z * step_z + vol_shape[0] - d_z,
                        "y_start": y * step_y - d_y,
                        "y_end": y * step_y + vol_shape[1] - d_y,
                        "x_start": x * step_x - d_x,
                        "x_end": x * step_x + vol_shape[2] - d_x,
                    }
                )

                if load_data and data_mask is not None:
                    cropped_data_mask[c] = padded_data_mask[
                        z * step_z - d_z : (z * step_z) + vol_shape[0] - d_z,
                        y * step_y - d_y : y * step_y + vol_shape[1] - d_y,
                        x * step_x - d_x : x * step_x + vol_shape[2] - d_x,
                    ]
                c += 1

    if verbose:
        print("**** New data shape is: {}".format(cropped_data.shape))
        print("### END 3D-OV-CROP ###")

    if load_data:
        if data_mask is not None:
            return cropped_data, cropped_data_mask, crop_coords
        else:
            return cropped_data, crop_coords
    else:
        return crop_coords


def merge_3D_data_with_overlap(
    data,
    orig_vol_shape,
    data_mask=None,
    overlap=(0, 0, 0),
    padding=(0, 0, 0),
    verbose=True,
):
    """
    Merge 3D subvolumes in a 3D volume with a defined overlap.

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
            raise ValueError(
                "data and data_mask shapes mismatch: {} vs {}".format(data.shape[:-1], data_mask.shape[:-1])
            )

    if (
        (overlap[0] >= 1 or overlap[0] < 0)
        or (overlap[1] >= 1 or overlap[1] < 0)
        or (overlap[2] >= 1 or overlap[2] < 0)
    ):
        raise ValueError("'overlap' values must be floats between range [0, 1)")

    if verbose:
        print("### MERGE-3D-OV-CROP ###")
        print("Merging {} images into {} with overlapping . . .".format(data.shape, orig_vol_shape))
        print("Minimum overlap selected: {}".format(overlap))
        print("Padding: {}".format(padding))

    # Remove the padding
    pad_input_shape = data.shape
    data = data[
        :,
        padding[0] : data.shape[1] - padding[0],
        padding[1] : data.shape[2] - padding[1],
        padding[2] : data.shape[3] - padding[2],
        :,
    ]

    merged_data = np.zeros((orig_vol_shape), dtype=np.float32)
    if data_mask is not None:
        data_mask = data_mask[
            :,
            padding[0] : data_mask.shape[1] - padding[0],
            padding[1] : data_mask.shape[2] - padding[1],
            padding[2] : data_mask.shape[3] - padding[2],
            :,
        ]
        merged_data_mask = np.zeros(orig_vol_shape[:3] + (data_mask.shape[-1],), dtype=np.float32)
    ov_map_counter = np.zeros((orig_vol_shape[:-1] + (1,)), dtype=np.uint16)

    # Calculate overlapping variables
    overlap_z = 1 if overlap[0] == 0 else 1 - overlap[0]
    overlap_y = 1 if overlap[1] == 0 else 1 - overlap[1]
    overlap_x = 1 if overlap[2] == 0 else 1 - overlap[2]

    padded_vol_shape = [
        orig_vol_shape[0] + 2 * padding[0],
        orig_vol_shape[1] + 2 * padding[1],
        orig_vol_shape[2] + 2 * padding[2],
    ]

    # Z
    step_z = int((pad_input_shape[1] - padding[0] * 2) * overlap_z)
    vols_per_z = math.ceil(orig_vol_shape[0] / step_z)
    last_z = 0 if vols_per_z == 1 else (((vols_per_z - 1) * step_z) + pad_input_shape[1]) - padded_vol_shape[0]
    ovz_per_block = last_z // (vols_per_z - 1) if vols_per_z > 1 else 0
    step_z -= ovz_per_block
    last_z -= ovz_per_block * (vols_per_z - 1)

    # Y
    step_y = int((pad_input_shape[2] - padding[1] * 2) * overlap_y)
    vols_per_y = math.ceil(orig_vol_shape[1] / step_y)
    last_y = 0 if vols_per_y == 1 else (((vols_per_y - 1) * step_y) + pad_input_shape[2]) - padded_vol_shape[1]
    ovy_per_block = last_y // (vols_per_y - 1) if vols_per_y > 1 else 0
    step_y -= ovy_per_block
    last_y -= ovy_per_block * (vols_per_y - 1)

    # X
    step_x = int((pad_input_shape[3] - padding[2] * 2) * overlap_x)
    vols_per_x = math.ceil(orig_vol_shape[2] / step_x)
    last_x = 0 if vols_per_x == 1 else (((vols_per_x - 1) * step_x) + pad_input_shape[3]) - padded_vol_shape[2]
    ovx_per_block = last_x // (vols_per_x - 1) if vols_per_x > 1 else 0
    step_x -= ovx_per_block
    last_x -= ovx_per_block * (vols_per_x - 1)

    # Real overlap calculation for printing
    real_ov_z = ovz_per_block / (pad_input_shape[1] - padding[0] * 2)
    real_ov_y = ovy_per_block / (pad_input_shape[2] - padding[1] * 2)
    real_ov_x = ovx_per_block / (pad_input_shape[3] - padding[2] * 2)

    if verbose:
        print("Real overlapping (%): {}".format((real_ov_z, real_ov_y, real_ov_x)))
        print(
            "Real overlapping (pixels): {}".format(
                (
                    (pad_input_shape[1] - padding[0] * 2) * real_ov_z,
                    (pad_input_shape[2] - padding[1] * 2) * real_ov_y,
                    (pad_input_shape[3] - padding[2] * 2) * real_ov_x,
                )
            )
        )
        print("{} patches per (z,y,x) axis".format((vols_per_z, vols_per_x, vols_per_y)))

    c = 0
    for z in range(vols_per_z):
        for y in range(vols_per_y):
            for x in range(vols_per_x):
                d_z = 0 if (z * step_z + data.shape[1]) < orig_vol_shape[0] else last_z
                d_y = 0 if (y * step_y + data.shape[2]) < orig_vol_shape[1] else last_y
                d_x = 0 if (x * step_x + data.shape[3]) < orig_vol_shape[2] else last_x

                merged_data[
                    z * step_z - d_z : (z * step_z) + data.shape[1] - d_z,
                    y * step_y - d_y : y * step_y + data.shape[2] - d_y,
                    x * step_x - d_x : x * step_x + data.shape[3] - d_x,
                ] += data[c]

                if data_mask is not None:
                    merged_data_mask[
                        z * step_z - d_z : (z * step_z) + data.shape[1] - d_z,
                        y * step_y - d_y : y * step_y + data.shape[2] - d_y,
                        x * step_x - d_x : x * step_x + data.shape[3] - d_x,
                    ] += data_mask[c]

                ov_map_counter[
                    z * step_z - d_z : (z * step_z) + data.shape[1] - d_z,
                    y * step_y - d_y : y * step_y + data.shape[2] - d_y,
                    x * step_x - d_x : x * step_x + data.shape[3] - d_x,
                ] += 1
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


def extract_3D_patch_with_overlap_yield(
    data,
    vol_shape,
    axis_order,
    overlap=(0, 0, 0),
    padding=(0, 0, 0),
    total_ranks=1,
    rank=0,
    return_only_stats=False,
    load_data=True,
    verbose=False,
):
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

    load_data: bool, optional
        Whether to load data from file or not. Useful to speed up the process if only patch coords are needed.

    verbose : bool, optional
        To print useful information for debugging.

    Yields
    ------
    img : 4D Numpy array, optional
        Extracted patch from ``data``. E.g. ``(z, y, x, channels)``. Returned if ``load_data`` is ``True``.

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
        print(
            "Cropping {} images into {} with overlapping (axis order: {}). . .".format(
                data.shape, vol_shape, axis_order
            )
        )
        print("Minimum overlap selected: {}".format(overlap))
        print("Padding: {}".format(padding))

    if len(vol_shape) != 4:
        raise ValueError("vol_shape expected to be of length 4, given {}".format(vol_shape))

    t_dim, z_dim, c_dim, y_dim, x_dim = order_dimensions(data.shape, axis_order)

    if vol_shape[0] > z_dim:
        raise ValueError(
            "'vol_shape[0]' {} greater than {} (you can reduce 'DATA.PATCH_SIZE')".format(vol_shape[0], z_dim)
        )
    if vol_shape[1] > y_dim:
        raise ValueError(
            "'vol_shape[1]' {} greater than {} (you can reduce 'DATA.PATCH_SIZE')".format(vol_shape[1], y_dim)
        )
    if vol_shape[2] > x_dim:
        raise ValueError(
            "'vol_shape[2]' {} greater than {} (you can reduce 'DATA.PATCH_SIZE')".format(vol_shape[2], x_dim)
        )
    if (
        (overlap[0] >= 1 or overlap[0] < 0)
        or (overlap[1] >= 1 or overlap[1] < 0)
        or (overlap[2] >= 1 or overlap[2] < 0)
    ):
        raise ValueError("'overlap' values must be floats between range [0, 1)")
    for i, p in enumerate(padding):
        if p >= vol_shape[i] // 2:
            raise ValueError(
                "'Padding' can not be greater than the half of 'vol_shape'. Max value for this {} input shape is {}".format(
                    vol_shape,
                    [
                        (vol_shape[0] // 2) - 1,
                        (vol_shape[1] // 2) - 1,
                        (vol_shape[2] // 2) - 1,
                    ],
                )
            )

    padded_data_shape = [
        z_dim + padding[0] * 2,
        y_dim + padding[1] * 2,
        x_dim + padding[2] * 2,
        c_dim,
    ]
    padded_vol_shape = vol_shape

    # Calculate overlapping variables
    overlap_z = 1 if overlap[0] == 0 else 1 - overlap[0]
    overlap_y = 1 if overlap[1] == 0 else 1 - overlap[1]
    overlap_x = 1 if overlap[2] == 0 else 1 - overlap[2]

    # Z
    step_z = int((vol_shape[0] - padding[0] * 2) * overlap_z)
    vols_per_z = math.ceil(z_dim / step_z)
    last_z = 0 if vols_per_z == 1 else (((vols_per_z - 1) * step_z) + vol_shape[0]) - padded_data_shape[0]
    ovz_per_block = last_z // (vols_per_z - 1) if vols_per_z > 1 else 0
    step_z -= ovz_per_block
    last_z -= ovz_per_block * (vols_per_z - 1)

    # Y
    step_y = int((vol_shape[1] - padding[1] * 2) * overlap_y)
    vols_per_y = math.ceil(y_dim / step_y)
    last_y = 0 if vols_per_y == 1 else (((vols_per_y - 1) * step_y) + vol_shape[1]) - padded_data_shape[1]
    ovy_per_block = last_y // (vols_per_y - 1) if vols_per_y > 1 else 0
    step_y -= ovy_per_block
    last_y -= ovy_per_block * (vols_per_y - 1)

    # X
    step_x = int((vol_shape[2] - padding[2] * 2) * overlap_x)
    vols_per_x = math.ceil(x_dim / step_x)
    last_x = 0 if vols_per_x == 1 else (((vols_per_x - 1) * step_x) + vol_shape[2]) - padded_data_shape[2]
    ovx_per_block = last_x // (vols_per_x - 1) if vols_per_x > 1 else 0
    step_x -= ovx_per_block
    last_x -= ovx_per_block * (vols_per_x - 1)

    # Real overlap calculation for printing
    real_ov_z = ovz_per_block / (vol_shape[0] - padding[0] * 2)
    real_ov_y = ovy_per_block / (vol_shape[1] - padding[1] * 2)
    real_ov_x = ovx_per_block / (vol_shape[2] - padding[2] * 2)
    if verbose and rank == 0:
        print("Real overlapping (%): {}".format((real_ov_z, real_ov_y, real_ov_x)))
        print(
            "Real overlapping (pixels): {}".format(
                (
                    (vol_shape[0] - padding[0] * 2) * real_ov_z,
                    (vol_shape[1] - padding[1] * 2) * real_ov_y,
                    (vol_shape[2] - padding[2] * 2) * real_ov_x,
                )
            )
        )
        print("{} patches per (z,y,x) axis".format((vols_per_z, vols_per_x, vols_per_y)))

    vols_in_z = vols_per_z // total_ranks
    vols_per_z_per_rank = vols_in_z
    if vols_per_z % total_ranks > rank:
        vols_per_z_per_rank += 1
    total_vol = vols_per_z_per_rank * vols_per_y * vols_per_x

    c = 0
    list_of_vols_in_z = []
    z_vol_info = {}
    for i in range(total_ranks):
        vols = (vols_per_z // total_ranks) + 1 if vols_per_z % total_ranks > i else vols_in_z
        for j in range(vols):
            z = c + j
            real_start_z = z * step_z
            real_finish_z = min(real_start_z + step_z + ovz_per_block, z_dim)
            z_vol_info[z] = [real_start_z, real_finish_z]
        list_of_vols_in_z.append(list(range(c, c + vols)))
        c += vols
    if verbose and rank == 0:
        print(f"List of volume IDs to be processed by each GPU: {list_of_vols_in_z}")
        print(f"Positions of each volume in Z axis: {z_vol_info}")
        print(
            "Rank {}: Total number of patches: {} - {} patches per (z,y,x) axis (per GPU)".format(
                rank, total_vol, (vols_per_z_per_rank, vols_per_x, vols_per_y)
            )
        )

    if return_only_stats:
        yield total_vol, z_vol_info, list_of_vols_in_z
        return

    for _z in range(vols_per_z_per_rank):
        z = list_of_vols_in_z[rank][0] + _z
        for y in range(vols_per_y):
            for x in range(vols_per_x):
                d_z = 0 if (z * step_z + vol_shape[0]) < padded_data_shape[0] else last_z
                d_y = 0 if (y * step_y + vol_shape[1]) < padded_data_shape[1] else last_y
                d_x = 0 if (x * step_x + vol_shape[2]) < padded_data_shape[2] else last_x

                start_z = max(0, z * step_z - d_z - padding[0])
                finish_z = min(z * step_z + vol_shape[0] - d_z - padding[0], z_dim)
                start_y = max(0, y * step_y - d_y - padding[1])
                finish_y = min(y * step_y + vol_shape[1] - d_y - padding[1], y_dim)
                start_x = max(0, x * step_x - d_x - padding[2])
                finish_x = min(x * step_x + vol_shape[2] - d_x - padding[2], x_dim)

                slices = [
                    slice(start_z, finish_z),
                    slice(start_y, finish_y),
                    slice(start_x, finish_x),
                    slice(None),  # Channel
                ]

                data_ordered_slices = order_dimensions(
                    slices, input_order="ZYXC", output_order=axis_order, default_value=0
                )

                real_patch_in_data = {
                    "z_start": z * step_z - d_z,
                    "z_end": (z * step_z) + vol_shape[0] - d_z - (padding[0] * 2),
                    "y_start": y * step_y - d_y,
                    "y_end": (y * step_y) + vol_shape[1] - d_y - (padding[1] * 2),
                    "x_start": x * step_x - d_x,
                    "x_end": (x * step_x) + vol_shape[2] - d_x - (padding[2] * 2),
                }

                if load_data:
                    img = data[tuple(data_ordered_slices)]

                    # The image should have the channel dimension at the end
                    current_order = np.array(range(len(img.shape)))
                    transpose_order = order_dimensions(
                        current_order,  #
                        input_order="ZYXC",
                        output_order=axis_order,
                        default_value=np.nan,
                    )

                    # determine the transpose order
                    transpose_order = [x for x in transpose_order if not np.isnan(x)]
                    transpose_order = np.argsort(transpose_order)
                    transpose_order = current_order[transpose_order]

                    img = np.transpose(img, transpose_order)

                    pad_z_left = padding[0] - z * step_z - d_z if start_z <= 0 else 0
                    pad_z_right = (start_z + vol_shape[0]) - z_dim if start_z + vol_shape[0] > z_dim else 0
                    pad_y_left = padding[1] - y * step_y - d_y if start_y <= 0 else 0
                    pad_y_right = (start_y + vol_shape[1]) - y_dim if start_y + vol_shape[1] > y_dim else 0
                    pad_x_left = padding[2] - x * step_x - d_x if start_x <= 0 else 0
                    pad_x_right = (start_x + vol_shape[2]) - x_dim if start_x + vol_shape[2] > x_dim else 0

                    if img.ndim == 3:
                        img = np.pad(
                            img,
                            (
                                (pad_z_left, pad_z_right),
                                (pad_y_left, pad_y_right),
                                (pad_x_left, pad_x_right),
                            ),
                            "reflect",
                        )
                        img = np.expand_dims(img, -1)
                    else:
                        img = np.pad(
                            img,
                            (
                                (pad_z_left, pad_z_right),
                                (pad_y_left, pad_y_right),
                                (pad_x_left, pad_x_right),
                                (0, 0),
                            ),
                            "reflect",
                        )

                    assert (
                        img.shape[:-1] == vol_shape[:-1]
                    ), f"Image shape and expected shape differ: {img.shape} vs {vol_shape}"

                    if rank == 0:
                        yield img, real_patch_in_data, total_vol, z_vol_info, list_of_vols_in_z
                    else:
                        yield img, real_patch_in_data, total_vol
                else:
                    if rank == 0:
                        yield real_patch_in_data, total_vol, z_vol_info, list_of_vols_in_z
                    else:
                        yield real_patch_in_data, total_vol


def order_dimensions(data, input_order, output_order="TZCYX", default_value=1):
    """
    Reorder data from any input order to output order.

    Parameters
    ----------
    data : Numpy array like
        data to reorder. E.g. ``(z, y, x, channels)``.

    input_order : str
        Order of the input data. E.g. ``ZYXC``.

    output_order : str, optional
        Order of the output data. E.g. ``TZCYX``.

    default_value : Any, optional
        Default value to use when a dimension is not present in the input order.

    Returns
    -------
    shape : Tuple
        Reordered data. E.g. ``(t, z, channel, y, x)``.
    """

    if input_order == output_order:
        return data

    output_data = []

    for i in range(len(output_order)):
        if output_order[i] in input_order:
            output_data.append(data[input_order.index(output_order[i])])
        else:
            output_data.append(default_value)
    return tuple(output_data)


def ensure_3d_shape(img, path=None):
    """
    Read an image from a given path.

    Parameters
    ----------
    img : ndarray
        Image read.

    path : str
        Path of the image (just use to print possible errors).

    Returns
    -------
    img : Numpy 4D array
        Image read. E.g. ``(z, y, x, num_classes)``.
    """
    if img.ndim < 3:
        if path is not None:
            m = "Read image seems to be 2D: {}. Path: {}".format(img.shape, path)
        else:
            m = "Read image seems to be 2D: {}".format(img.shape)
        raise ValueError(m)

    if img.ndim == 3:
        img = np.expand_dims(img, -1)
    else:
        min_val = min(img.shape)
        channel_pos = img.shape.index(min_val)
        if channel_pos != 3 and img.shape[channel_pos] <= 4:
            new_pos = [x for x in range(4) if x != channel_pos] + [
                channel_pos,
            ]
            img = img.transpose(new_pos)
    return img
