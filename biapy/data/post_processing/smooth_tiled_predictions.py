# MIT License
# Copyright (c) 2017 Vooban Inc.
# Coded by: Guillaume Chevalier
# Source to original code and license:
#     https://github.com/Vooban/Smoothly-Blend-Image-Patches
#     https://github.com/Vooban/Smoothly-Blend-Image-Patches/blob/master/LICENSE


"""
Smoothly blend tiled prediction patches for large images.

This module provides utilities to perform smooth predictions on large images by
dividing them into overlapping tiles, applying a prediction function to each tile,
and blending the results using spline-based windowing. This approach reduces edge
artifacts and improves prediction quality for models that operate on patches.
"""

import numpy as np
import scipy.signal
from tqdm import tqdm
import gc

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    PLOT_PROGRESS = True
    # See end of file for the rest of the __main__.
else:
    PLOT_PROGRESS = False


def _spline_window(window_size, power=2):
    """
    Generate a squared spline window function for smooth blending.

    Source: https://www.wolframalpha.com/input/?i=y%3Dx**2,+y%3D-(x-2)**2+%2B2,+y%3D(x-4)**2,+from+y+%3D+0+to+2

    Parameters
    ----------
    window_size : int
        Size of the window (patch).
    power : int, optional
        Power of the spline (default is 2).

    Returns
    -------
    wind : np.ndarray
        1D window array for blending.
    """
    intersection = int(window_size / 4)
    wind_outer = (abs(2 * (scipy.signal.triang(window_size))) ** power) / 2
    wind_outer[intersection:-intersection] = 0

    wind_inner = 1 - (abs(2 * (scipy.signal.triang(window_size) - 1)) ** power) / 2
    wind_inner[:intersection] = 0
    wind_inner[-intersection:] = 0

    wind = wind_inner + wind_outer
    wind = wind / np.average(wind)
    return wind


cached_2d_windows = dict()


def _window_2D(window_size, power=2):
    """
    Create a 2D window function for smooth blending of overlapping patches.

    Parameters
    ----------
    window_size : int
        Size of the window (patch).
    power : int, optional
        Power of the spline (default is 2).

    Returns
    -------
    wind : np.ndarray
        2D window array for blending.
    """
    # Memoization
    global cached_2d_windows
    key = "{}_{}".format(window_size, power)
    if key in cached_2d_windows:
        wind = cached_2d_windows[key]
    else:
        wind = _spline_window(window_size, power)
        wind = np.expand_dims(np.expand_dims(wind, -1), -1)
        wind = wind * wind.transpose(1, 0, 2)
        if PLOT_PROGRESS:
            # For demo purpose, let's look once at the window:
            plt.imshow(wind[:, :, 0], cmap="viridis")
            plt.title("2D Windowing Function for a Smooth Blending of " "Overlapping Patches")
            plt.show()
        cached_2d_windows[key] = wind
    return wind


def _pad_img(img, window_size, subdivisions):
    """
    Pad an image with reflected borders for tiled prediction.

    Parameters
    ----------
    img : np.ndarray
        Input image of shape (y, x, nb_channels).
    window_size : int
        Size of the window (patch).
    subdivisions : int
        Number of subdivisions (overlap factor).

    Returns
    -------
    ret : np.ndarray
        Padded image.
    """
    aug = int(round(window_size * (1 - 1.0 / subdivisions)))
    more_borders = ((aug, aug), (aug, aug), (0, 0))
    ret = np.pad(img, pad_width=more_borders, mode="reflect")
    # gc.collect()

    if PLOT_PROGRESS:
        # For demo purpose, let's look once at the window:
        plt.imshow(ret)
        plt.title(
            "Padded Image for Using Tiled Prediction Patches\n" "(notice the reflection effect on the padded borders)"
        )
        plt.show()
    return ret


def _unpad_img(padded_img, window_size, subdivisions):
    """
    Remove padding added by `_pad_img`.

    Parameters
    ----------
    padded_img : np.ndarray
        Image with padding.
    window_size : int
        Size of the window (patch).
    subdivisions : int
        Number of subdivisions (overlap factor).

    Returns
    -------
    ret : np.ndarray
        Unpadded image.
    """
    aug = int(round(window_size * (1 - 1.0 / subdivisions)))
    ret = padded_img[aug:-aug, aug:-aug, :]
    # gc.collect()
    return ret


def _rotate_mirror_do(im):
    """
    Generate all 8 dihedral (D4) rotations and mirrors of an image.

    Parameters
    ----------
    im : np.ndarray
        Input image of shape (y, x, nb_channels).

    Returns
    -------
    mirrs : list of np.ndarray
        List of 8 rotated and mirrored images.
    """
    mirrs = []
    mirrs.append(np.array(im))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=1))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=2))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=3))
    im = np.array(im)[:, ::-1]
    mirrs.append(np.array(im))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=1))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=2))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=3))
    return mirrs


def _rotate_mirror_undo(im_mirrs):
    """
    Merge a list of 8 rotated/mirrored images back to the original orientation.

    Parameters
    ----------
    im_mirrs : list of np.ndarray
        List of 8 rotated and mirrored images.

    Returns
    -------
    np.ndarray
        Averaged image in original orientation.
    """
    origs = []
    origs.append(np.array(im_mirrs[0]))
    origs.append(np.rot90(np.array(im_mirrs[1]), axes=(0, 1), k=3))
    origs.append(np.rot90(np.array(im_mirrs[2]), axes=(0, 1), k=2))
    origs.append(np.rot90(np.array(im_mirrs[3]), axes=(0, 1), k=1))
    origs.append(np.array(im_mirrs[4])[:, ::-1])
    origs.append(np.rot90(np.array(im_mirrs[5]), axes=(0, 1), k=3)[:, ::-1])
    origs.append(np.rot90(np.array(im_mirrs[6]), axes=(0, 1), k=2)[:, ::-1])
    origs.append(np.rot90(np.array(im_mirrs[7]), axes=(0, 1), k=1)[:, ::-1])
    return np.mean(origs, axis=0)


def _windowed_subdivs(padded_img, window_size, subdivisions, n_classes, pred_func):
    """
    Create tiled overlapping patches and apply prediction function.

    Parameters
    ----------
    padded_img : np.ndarray
        Padded input image.
    window_size : int
        Size of the window (patch).
    subdivisions : int
        Number of subdivisions (overlap factor).
    n_classes : int
        Number of output channels/classes.
    pred_func : callable
        Prediction function to apply to each patch.

    Returns
    -------
    subdivs : np.ndarray
        5D array of predicted patches of shape = (
            nb_patches_along_X,
            nb_patches_along_Y,
            patches_resolution_along_X,
            patches_resolution_along_Y,
            nb_output_channels
        )

    Note:
        patches_resolution_along_X == patches_resolution_along_Y == window_size
    """
    WINDOW_SPLINE_2D = _window_2D(window_size=window_size, power=2)

    step = int(window_size / subdivisions)
    padx_len = padded_img.shape[0]
    pady_len = padded_img.shape[1]
    subdivs = []

    for i in range(0, padx_len - window_size + 1, step):
        subdivs.append([])
        for j in range(0, pady_len - window_size + 1, step):
            patch = padded_img[i : i + window_size, j : j + window_size, :]
            subdivs[-1].append(patch)

    # Here, `gc.collect()` clears RAM between operations.
    # It should run faster if they are removed, if enough memory is available.
    gc.collect()
    subdivs = np.array(subdivs)
    gc.collect()
    a, b, c, d, e = subdivs.shape
    subdivs = subdivs.reshape(a * b, c, d, e)
    gc.collect()

    subdivs = pred_func(subdivs)
    if isinstance(subdivs, list):
        subdivs = subdivs[-1]
    gc.collect()
    subdivs = np.array([patch * WINDOW_SPLINE_2D for patch in subdivs])
    gc.collect()

    # Such 5D array:
    subdivs = subdivs.reshape(a, b, c, d, n_classes)
    gc.collect()

    return subdivs


def create_gen(subdivs, subdivs_m, subdivs_w):
    """
    Create a generator that yields ([image, weights], label) tuples from patch arrays.

    Parameters
    ----------
    subdivs : np.ndarray
        Array of image patches.
    subdivs_m : np.ndarray
        Array of label patches.
    subdivs_w : np.ndarray
        Array of weight patches.

    Yields
    ------
    tuple
        A tuple of ([image_patch, weight_patch], label_patch) for each patch.
    """
    gen = zip(subdivs, subdivs_m, subdivs_w)
    for img, label, weights in gen:
        yield ([img, weights], label)


def _recreate_from_subdivs(subdivs, window_size, subdivisions, padded_out_shape):
    """
    Merge tiled overlapping patches smoothly into a single image.

    Parameters
    ----------
    subdivs : np.ndarray
        5D array of predicted patches.
    window_size : int
        Size of the window (patch).
    subdivisions : int
        Number of subdivisions (overlap factor).
    padded_out_shape : tuple
        Shape of the output image (with padding).

    Returns
    -------
    y : np.ndarray
        Reconstructed image from patches.
    """
    step = int(window_size / subdivisions)
    padx_len = padded_out_shape[0]
    pady_len = padded_out_shape[1]

    y = np.zeros(padded_out_shape)

    a = 0
    for i in range(0, padx_len - window_size + 1, step):
        b = 0
        for j in range(0, pady_len - window_size + 1, step):
            windowed_patch = subdivs[a, b]
            y[i : i + window_size, j : j + window_size] = y[i : i + window_size, j : j + window_size] + windowed_patch
            b += 1
        a += 1
    return y / (subdivisions**2)


def predict_img_with_smooth_windowing(input_img, window_size, subdivisions, n_classes, pred_func):
    """
    Predict an image by applying a function to overlapping patches and blending smoothly.

    This function divides the input image into overlapping patches, applies the
    prediction function to each patch, and blends the results using a spline window.
    See 6th, 7th and 8th ideas here:
    http://blog.kaggle.com/2017/05/09/dstl-satellite-imagery-competition-3rd-place-winners-interview-vladimir-sergey/

    Parameters
    ----------
    input_img : np.ndarray
        Input image of shape (y, x, nb_channels).
    window_size : int
        Size of the window (patch).
    subdivisions : int
        Number of subdivisions (overlap factor).
    n_classes : int
        Number of output channels/classes.
    pred_func : callable
        Prediction function to apply to each patch.

    Returns
    -------
    prd : np.ndarray
        Output image with smoothly blended predictions.
    """
    # Pad the input image to allow for overlapping patches.
    pad = _pad_img(input_img, window_size, subdivisions)
    # Generate all 8 dihedral (D4) rotations and mirrors of the padded image.
    pads = _rotate_mirror_do(pad)

    # Note that the implementation could be more memory-efficient by merging
    # the behavior of `_windowed_subdivs` and `_recreate_from_subdivs` into
    # one loop doing in-place assignments to the new image matrix, rather than
    # using a temporary 5D array.

    # It would also be possible to allow different (and impure) window functions
    # that might not tile well. Adding their weighting to another matrix could
    # be done to later normalize the predictions correctly by dividing the whole
    # reconstructed thing by this matrix of weightings - to normalize things
    # back from an impure windowing function that would have badly weighted
    # windows.

    # For example, since the U-net of Kaggle's DSTL satellite imagery feature
    # prediction challenge's 3rd place winners use a different window size for
    # the input and output of the neural net's patches predictions, it would be
    # possible to fake a full-size window which would in fact just have a narrow
    # non-zero dommain. This may require to augment the `subdivisions` argument
    # to 4 rather than 2.

    res = []
    for pad in tqdm(pads):
        # For every rotation:
        sd = _windowed_subdivs(pad, window_size, subdivisions, n_classes, pred_func)
        one_padded_result = _recreate_from_subdivs(
            sd,
            window_size,
            subdivisions,
            padded_out_shape=list(pad.shape[:-1]) + [n_classes],
        )

        res.append(one_padded_result)

    # Merge after rotations:
    padded_results = _rotate_mirror_undo(res)

    prd = _unpad_img(padded_results, window_size, subdivisions)

    prd = prd[: input_img.shape[0], : input_img.shape[1], :]

    if PLOT_PROGRESS:
        plt.imshow(prd)
        plt.title("Smoothly Merged Patches that were Tiled Tighter")
        plt.show()
    return prd


def predict_img_with_overlap(input_img, window_size, subdivisions, n_classes, pred_func):
    """
    Predict an image by applying a function to overlapping patches (no rotation/mirroring).

    This function divides the input image into overlapping patches, applies the
    prediction function to each patch, and blends the results using a spline window.
    Unlike `predict_img_with_smooth_windowing`, it does not use dihedral augmentations.

    Parameters
    ----------
    input_img : np.ndarray
        Input image of shape (y, x, nb_channels).
    window_size : int
        Size of the window (patch).
    subdivisions : int
        Number of subdivisions (overlap factor).
    n_classes : int
        Number of output channels/classes.
    pred_func : callable
        Prediction function to apply to each patch.

    Returns
    -------
    prd : np.ndarray
        Output image with smoothly blended predictions.
    """
    pad = _pad_img(input_img, window_size, subdivisions)

    sd = _windowed_subdivs(pad, window_size, subdivisions, n_classes, pred_func)
    one_padded_result = _recreate_from_subdivs(
        sd,
        window_size,
        subdivisions,
        padded_out_shape=list(pad.shape[:-1]) + [n_classes],
    )

    prd = _unpad_img(one_padded_result, window_size, subdivisions)

    prd = prd[: input_img.shape[0], : input_img.shape[1], :]

    if PLOT_PROGRESS:
        plt.imshow(prd)
        plt.title("Smoothly Merged Patches that were Tiled Tighter")
        plt.show()
    return prd
