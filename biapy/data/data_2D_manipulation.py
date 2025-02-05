import numpy as np
import os
import math
from PIL import Image


def crop_data_with_overlap(
    data, crop_shape, data_mask=None, overlap=(0, 0), padding=(0, 0), verbose=True, load_data=True
):
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

    load_data : bool, optional
        Whether to create the patches or not. It saves memory in case you only need the coordiantes of the cropped patches.

    Returns
    -------
    cropped_data : 4D Numpy array, optional
        Cropped image data. E.g. ``(num_of_images, y, x, channels)``. Returned if ``load_data`` is ``True``.

    cropped_data_mask : 4D Numpy array, optional
        Cropped image data masks. E.g. ``(num_of_images, y, x, channels)``. Returned if ``load_data`` is ``True``
        and ``data_mask`` is provided.

    crop_coords : list of dict
        Coordinates of each crop where the following keys are available:
            * ``"z"``: image used to extract the crop.
            * ``"y_start"``: starting point of the patch in Y axis.
            * ``"y_end"``: end point of the patch in Y axis.
            * ``"x_start"``: starting point of the patch in X axis.
            * ``"x_end"``: end point of the patch in X axis.

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
                    crop_shape, [(crop_shape[0] // 2) - 1, (crop_shape[1] // 2) - 1]
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
    if load_data:
        cropped_data = np.zeros((total_vol,) + crop_shape, dtype=data.dtype)
        if data_mask is not None:
            cropped_data_mask = np.zeros(
                (total_vol,) + crop_shape[:2] + (data_mask.shape[-1],),
                dtype=data_mask.dtype,
            )

    crop_coords = []
    c = 0
    for z in range(data.shape[0]):
        for y in range(crops_per_y):
            for x in range(crops_per_x):
                d_y = 0 if (y * step_y + crop_shape[0]) < padded_data.shape[1] else last_y
                d_x = 0 if (x * step_x + crop_shape[1]) < padded_data.shape[2] else last_x

                if load_data:
                    cropped_data[c] = padded_data[
                        z,
                        y * step_y - d_y : y * step_y + crop_shape[0] - d_y,
                        x * step_x - d_x : x * step_x + crop_shape[1] - d_x,
                    ]

                if load_data and data_mask is not None:
                    cropped_data_mask[c] = padded_data_mask[
                        z,
                        y * step_y - d_y : y * step_y + crop_shape[0] - d_y,
                        x * step_x - d_x : x * step_x + crop_shape[1] - d_x,
                    ]

                crop_coords.append(
                    {
                        "z": z,
                        "y_start": y * step_y - d_y,
                        "y_end": y * step_y + crop_shape[0] - d_y,
                        "x_start": x * step_x - d_x,
                        "x_end": x * step_x + crop_shape[1] - d_x,
                    }
                )

                c += 1

    if verbose:
        print("**** New data shape is: {}".format(cropped_data.shape))
        print("### END OV-CROP ###")

    if load_data:
        if data_mask is not None:
            return cropped_data, cropped_data_mask, crop_coords
        else:
            return cropped_data, crop_coords
    else:
        return crop_coords


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


def ensure_2d_shape(img, path=None):
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
    img : Numpy 3D array
        Image read. E.g. ``(y, x, channels)``.
    """
    if img.ndim > 3:
        if path is not None:
            m = "Read image seems to be 3D: {}. Path: {}".format(img.shape, path)
        else:
            m = "Read image seems to be 3D: {}".format(img.shape)
        raise ValueError(m)

    if img.ndim == 2:
        img = np.expand_dims(img, -1)
    else:
        # Ensure channel axis is always in the first position (assuming Z is already set)
        min_val = min(img.shape)
        channel_pos = img.shape.index(min_val)
        if channel_pos != 2:
            new_pos = [x for x in range(3) if x != channel_pos] + [
                channel_pos,
            ]
            img = img.transpose(new_pos)
    return img
