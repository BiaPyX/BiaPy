# MIT License
# Copyright (c) 2017 Vooban Inc.
# Coded by: Guillaume Chevalier
# Source to original code and license:
#     https://github.com/Vooban/Smoothly-Blend-Image-Patches
#     https://github.com/Vooban/Smoothly-Blend-Image-Patches/blob/master/LICENSE


"""Do smooth predictions on an image from tiled prediction patches."""


import numpy as np
import scipy.signal
from tqdm import tqdm
import gc
from tensorflow.keras.preprocessing.image import ImageDataGenerator as kerasDA
import math

from scipy.ndimage import rotate

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    PLOT_PROGRESS = True
    # See end of file for the rest of the __main__.
else:
    PLOT_PROGRESS = False


def _spline_window(window_size, power=2):
    """
    Squared spline (power=2) window function:
    https://www.wolframalpha.com/input/?i=y%3Dx**2,+y%3D-(x-2)**2+%2B2,+y%3D(x-4)**2,+from+y+%3D+0+to+2
    """
    intersection = int(window_size/4)
    wind_outer = (abs(2*(scipy.signal.triang(window_size))) ** power)/2
    wind_outer[intersection:-intersection] = 0

    wind_inner = 1 - (abs(2*(scipy.signal.triang(window_size) - 1)) ** power)/2
    wind_inner[:intersection] = 0
    wind_inner[-intersection:] = 0

    wind = wind_inner + wind_outer
    wind = wind / np.average(wind)
    return wind


cached_2d_windows = dict()
def _window_2D(window_size, power=2):
    """
    Make a 1D window function, then infer and return a 2D window function.
    Done with an augmentation, and self multiplication with its transpose.
    Could be generalized to more dimensions.
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
            plt.title("2D Windowing Function for a Smooth Blending of "
                      "Overlapping Patches")
            plt.show()
        cached_2d_windows[key] = wind
    return wind


def _pad_img(img, window_size, subdivisions):
    """
    Add borders to img for a "valid" border pattern according to "window_size" and
    "subdivisions".
    Image is an np array of shape (x, y, nb_channels).
    """
    aug = int(round(window_size * (1 - 1.0/subdivisions)))
    more_borders = ((aug, aug), (aug, aug), (0, 0))
    ret = np.pad(img, pad_width=more_borders, mode='reflect')
    # gc.collect()

    if PLOT_PROGRESS:
        # For demo purpose, let's look once at the window:
        plt.imshow(ret)
        plt.title("Padded Image for Using Tiled Prediction Patches\n"
                  "(notice the reflection effect on the padded borders)")
        plt.show()
    return ret


def _unpad_img(padded_img, window_size, subdivisions):
    """
    Undo what's done in the `_pad_img` function.
    Image is an np array of shape (x, y, nb_channels).
    """
    aug = int(round(window_size * (1 - 1.0/subdivisions)))
    ret = padded_img[
        aug:-aug,
        aug:-aug,
        :
    ]
    # gc.collect()
    return ret


def _rotate_mirror_do(im):
    """
    Duplicate an np array (image) of shape (x, y, nb_channels) 8 times, in order
    to have all the possible rotations and mirrors of that image that fits the
    possible 90 degrees rotations.

    It is the D_4 (D4) Dihedral group:
    https://en.wikipedia.org/wiki/Dihedral_group
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
    merges a list of 8 np arrays (images) of shape (x, y, nb_channels) generated
    from the `_rotate_mirror_do` function. Each images might have changed and
    merging them implies to rotated them back in order and average things out.

    It is the D_4 (D4) Dihedral group:
    https://en.wikipedia.org/wiki/Dihedral_group
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
    Create tiled overlapping patches.

    Returns:
        5D numpy array of shape = (
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

    step = int(window_size/subdivisions)
    padx_len = padded_img.shape[0]
    pady_len = padded_img.shape[1]
    subdivs = []

    for i in range(0, padx_len-window_size+1, step):
        subdivs.append([])
        for j in range(0, pady_len-window_size+1, step):
            patch = padded_img[i:i+window_size, j:j+window_size, :]
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
    gc.collect()
    subdivs = np.array([patch * WINDOW_SPLINE_2D for patch in subdivs])
    gc.collect()

    subdivs = subdivs[-1]
 
    # Such 5D array:
    subdivs = subdivs.reshape(a, b, c, d, n_classes)
    gc.collect()

    return subdivs

def _windowed_subdivs_weighted(padded_img, padded_mask, weight_map, batch_size_value, window_size, subdivisions, n_classes, pred_func):
    """
    Create tiled overlapping patches with weights.

    Returns:
        5D numpy array of shape = (
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

    step = int(window_size/subdivisions)
    padx_len = padded_img.shape[0]
    pady_len = padded_img.shape[1]
    subdivs = []
    subdivs_m = []
    subdivs_w = []

    for i in range(0, padx_len-window_size+1, step):
        subdivs.append([])
        subdivs_m.append([])
        subdivs_w.append([])
        for j in range(0, pady_len-window_size+1, step):
            patch = padded_img[i:i+window_size, j:j+window_size, :]
            subdivs[-1].append(patch)
            patch = padded_mask[i:i+window_size, j:j+window_size, :]
            subdivs_m[-1].append(patch)
            patch = weight_map[i:i+window_size, j:j+window_size, :]
            subdivs_w[-1].append(patch)
    
    # Here, `gc.collect()` clears RAM between operations.
    # It should run faster if they are removed, if enough memory is available.
    subdivs = np.array(subdivs)
    subdivs_m = np.array(subdivs_m)
    subdivs_w = np.array(subdivs_w)
    a, b, c, d, e = subdivs.shape
    subdivs = subdivs.reshape(a * b, c, d, e)
    subdivs_m = subdivs_m.reshape(a * b, c, d, e)
    subdivs_w = subdivs_w.reshape(a * b, c, d, e)
    r = np.zeros((a * b, c, d, e))

    # merge images and weights to predict
    X_datagen = kerasDA()
    Y_datagen = kerasDA()
    W_datagen = kerasDA()

    X_aug = X_datagen.flow(subdivs, batch_size=batch_size_value, shuffle=False)
    Y_aug = Y_datagen.flow(subdivs_m, batch_size=batch_size_value, shuffle=False)
    W_aug = W_datagen.flow(subdivs_w, batch_size=batch_size_value, shuffle=False)

    gen = create_gen(X_aug, Y_aug, W_aug)
    r = pred_func(gen, steps=math.ceil(X_aug.n/batch_size_value))
            
    subdivs = np.array([patch * WINDOW_SPLINE_2D for patch in r])

    # Such 5D array:
    subdivs = subdivs.reshape(a, b, c, d, n_classes)

    return subdivs

def create_gen(subdivs, subdivs_m, subdivs_w):
    gen = zip(subdivs, subdivs_m, subdivs_w)
    for (img, label, weights) in gen:
        yield ([img, weights], label)

def _recreate_from_subdivs(subdivs, window_size, subdivisions, padded_out_shape):
    """
    Merge tiled overlapping patches smoothly.
    """
    step = int(window_size/subdivisions)
    padx_len = padded_out_shape[0]
    pady_len = padded_out_shape[1]

    y = np.zeros(padded_out_shape)

    a = 0
    for i in range(0, padx_len-window_size+1, step):
        b = 0
        for j in range(0, pady_len-window_size+1, step):
            windowed_patch = subdivs[a, b]
            y[i:i+window_size, j:j+window_size] = y[i:i+window_size, j:j+window_size] + windowed_patch
            b += 1
        a += 1
    return y / (subdivisions ** 2)

def predict_img_with_smooth_windowing(input_img, window_size, subdivisions, n_classes, pred_func):
    """
    Apply the `pred_func` function to square patches of the image, and overlap
    the predictions to merge them smoothly.
    See 6th, 7th and 8th idea here:
    http://blog.kaggle.com/2017/05/09/dstl-satellite-imagery-competition-3rd-place-winners-interview-vladimir-sergey/
    """
    pad = _pad_img(input_img, window_size, subdivisions)
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
            sd, window_size, subdivisions,
            padded_out_shape=list(pad.shape[:-1])+[n_classes])

        res.append(one_padded_result)

    # Merge after rotations:
    padded_results = _rotate_mirror_undo(res)

    prd = _unpad_img(padded_results, window_size, subdivisions)

    prd = prd[:input_img.shape[0], :input_img.shape[1], :]

    if PLOT_PROGRESS:
        plt.imshow(prd)
        plt.title("Smoothly Merged Patches that were Tiled Tighter")
        plt.show()
    return prd

def predict_img_with_overlap(input_img, window_size, subdivisions, n_classes, pred_func):
    """Based on predict_img_with_smooth_windowing but works just with the 
       original image instead of creating 8 new ones.
    """
    pad = _pad_img(input_img, window_size, subdivisions)

    sd = _windowed_subdivs(pad, window_size, subdivisions, n_classes, pred_func)
    one_padded_result = _recreate_from_subdivs(sd, window_size, subdivisions,
                                               padded_out_shape=list(pad.shape[:-1])+[n_classes])

    prd = _unpad_img(one_padded_result, window_size, subdivisions)

    prd = prd[:input_img.shape[0], :input_img.shape[1], :]

    if PLOT_PROGRESS:
        plt.imshow(prd)
        plt.title("Smoothly Merged Patches that were Tiled Tighter")
        plt.show()
    return prd

def predict_img_with_overlap_weighted(input_img, input_mask, weight_map, batch_size_value, window_size, subdivisions, n_classes, pred_func):
    """Based on predict_img_with_smooth_windowing but works just with the
       original image (adding a weight map) instead of creating 8 new ones.
    """
    pad = _pad_img(input_img, window_size, subdivisions)
    pad_mask = _pad_img(input_mask, window_size, subdivisions)
    pad_w = _pad_img(weight_map, window_size, subdivisions)

    sd = _windowed_subdivs_weighted(pad, pad_mask, pad_w, batch_size_value, window_size, subdivisions, n_classes, pred_func)
    one_padded_result = _recreate_from_subdivs(sd, window_size, subdivisions,
                                               padded_out_shape=list(pad.shape[:-1])+[n_classes])

    prd = _unpad_img(one_padded_result, window_size, subdivisions)

    prd = prd[:input_img.shape[0], :input_img.shape[1], :]

    if PLOT_PROGRESS:
        plt.imshow(prd)
        plt.title("Smoothly Merged Patches that were Tiled Tighter")
        plt.show()
    return prd

def cheap_tiling_prediction(img, window_size, n_classes, pred_func):
    """
    Does predictions on an image without tiling.
    """
    original_shape = img.shape
    full_border = img.shape[0] + (window_size - (img.shape[0] % window_size))
    prd = np.zeros((full_border, full_border, n_classes))
    tmp = np.zeros((full_border, full_border, original_shape[-1]))
    tmp[:original_shape[0], :original_shape[1], :] = img
    img = tmp
    print(img.shape, tmp.shape, prd.shape)
    for i in tqdm(range(0, prd.shape[0], window_size)):
        for j in range(0, prd.shape[0], window_size):
            im = img[i:i+window_size, j:j+window_size]
            prd[i:i+window_size, j:j+window_size] = pred_func([im])
    prd = prd[:original_shape[0], :original_shape[1]]
    if PLOT_PROGRESS:
        plt.imshow(prd)
        plt.title("Cheaply Merged Patches")
        plt.show()
    return prd


def get_dummy_img(xy_size=128, nb_channels=3):
    """
    Create a random image with different luminosity in the corners.

    Returns an array of shape (xy_size, xy_size, nb_channels).
    """
    x = np.random.random((xy_size, xy_size, nb_channels))
    x = x + np.ones((xy_size, xy_size, 1))
    lin = np.expand_dims(
        np.expand_dims(
            np.linspace(0, 1, xy_size),
            nb_channels),
        nb_channels)
    x = x * lin
    x = x * lin.transpose(1, 0, 2)
    x = x + x[::-1, ::-1, :]
    x = x - np.min(x)
    x = x / np.max(x) / 2
    gc.collect()
    if PLOT_PROGRESS:
        plt.imshow(x)
        plt.title("Random image for a test")
        plt.show()
    return x


def round_predictions(prd, nb_channels_out, thresholds):
    """
    From a threshold list `thresholds` containing one threshold per output
    channel for comparison, the predictions are converted to a binary mask.
    """
    assert (nb_channels_out == len(thresholds))
    prd = np.array(prd)
    for i in range(nb_channels_out):
        # Per-pixel and per-channel comparison on a threshold to
        # binarize prediction masks:
        prd[:, :, i] = prd[:, :, i] > thresholds[i]
    return prd


def ensemble8_2d_predictions(o_img, pred_func, batch_size_value=1, 
                             n_classes=2):
                             
    """ Outputs the mean prediction of a given image generating its 8 possible 
        rotations/flips and blending them.
    """

    aug_img = []
        
    # Convert into square image to make the rotations properly
    pad_to_square = o_img.shape[0] - o_img.shape[1]
   
    if pad_to_square < 0:
        img = np.pad(o_img, [(abs(pad_to_square), 0), (0, 0), (0, 0)], 'reflect') 
    else:
        img = np.pad(o_img, [(0, 0), (pad_to_square, 0), (0, 0)], 'reflect')
    
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
    decoded_aug_img = np.zeros(aug_img.shape)
    
    for i in range(aug_img.shape[0]):
        if n_classes > 1:
            decoded_aug_img[i] = np.array(np.expand_dims(pred_func(np.expand_dims(aug_img[i], 0))[...,1], -1))[-1]
        else:
            decoded_aug_img[i] = np.array(pred_func(np.expand_dims(aug_img[i], 0)))[-1]

    # Undo the combinations of the img
    out_img = []
    out_img.append(decoded_aug_img[0])
    out_img.append(np.rot90(decoded_aug_img[1], axes=(0, 1), k=3))
    out_img.append(np.rot90(decoded_aug_img[2], axes=(0, 1), k=2))
    out_img.append(np.rot90(decoded_aug_img[3], axes=(0, 1), k=1))
    out_img.append(decoded_aug_img[4][:, ::-1])
    out_img.append(np.rot90(decoded_aug_img[5], axes=(0, 1), k=3)[:, ::-1])
    out_img.append(np.rot90(decoded_aug_img[6], axes=(0, 1), k=2)[:, ::-1])
    out_img.append(np.rot90(decoded_aug_img[7], axes=(0, 1), k=1)[:, ::-1])

    # Create the output data
    out_img = np.array(out_img) 
    if pad_to_square != 0:
        if pad_to_square < 0:
            out = np.zeros((out_img.shape[0], img.shape[0]+pad_to_square, 
                            img.shape[1], img.shape[2]))
        else:
            out = np.zeros((out_img.shape[0], img.shape[0], 
                            img.shape[1]-pad_to_square, img.shape[2]))
    else:
        out = np.zeros(out_img.shape)

    # Undo the padding
    for i in range(out_img.shape[0]):
        if pad_to_square < 0:
            out[i] = out_img[i,abs(pad_to_square):,:]
        else:
            out[i] = out_img[i,:,abs(pad_to_square):]

    return np.mean(out, axis=0)


def smooth_3d_predictions(vol, pred_func, batch_size_value=1, 
                          n_classes=2):
    """ Outputs the mean prediction of a given subvolume generating its 16 
        possible rotations/flips and blending them.
    """

    aug_vols = []
        
    # Convert into square image to make the rotations properly
    pad_to_square = vol.shape[0] - vol.shape[1]
   
    if pad_to_square < 0:
        volume = np.pad(vol, [(abs(pad_to_square),0), (0,0), (0,0), (0,0)], 'reflect') 
    else:
        volume = np.pad(vol, [(0,0), (pad_to_square,0), (0,0), (0,0)], 'reflect')
    
    # Make 16 different combinations of the volume 
    aug_vols.append(volume) 
    aug_vols.append(rotate(volume, mode='reflect', axes=(1, 2), angle=90, reshape=False))
    aug_vols.append(rotate(volume, mode='reflect', axes=(1, 2), angle=180, reshape=False))
    aug_vols.append(rotate(volume, mode='reflect', axes=(1, 2), angle=270, reshape=False))
    volume_aux = np.flip(volume, 0)
    aug_vols.append(volume_aux)
    aug_vols.append(rotate(volume_aux, mode='reflect', axes=(1, 2), angle=90, reshape=False))
    aug_vols.append(rotate(volume_aux, mode='reflect', axes=(1, 2), angle=180, reshape=False))
    aug_vols.append(rotate(volume_aux, mode='reflect', axes=(1, 2), angle=270, reshape=False))
    volume_aux = np.flip(volume, 1)
    aug_vols.append(volume_aux)
    aug_vols.append(rotate(volume_aux, mode='reflect', axes=(1, 2), angle=90, reshape=False))
    aug_vols.append(rotate(volume_aux, mode='reflect', axes=(1, 2), angle=180, reshape=False))
    aug_vols.append(rotate(volume_aux, mode='reflect', axes=(1, 2), angle=270, reshape=False))
    volume_aux = np.flip(volume, 2)
    aug_vols.append(volume_aux)
    aug_vols.append(rotate(volume_aux, mode='reflect', axes=(1, 2), angle=90, reshape=False))
    aug_vols.append(rotate(volume_aux, mode='reflect', axes=(1, 2), angle=180, reshape=False))
    aug_vols.append(rotate(volume_aux, mode='reflect', axes=(1, 2), angle=270, reshape=False))
    del volume_aux

    aug_vols = np.array(aug_vols)
    decoded_aug_vols = np.zeros(aug_vols.shape)

    for i in range(aug_vols.shape[0]):
        if n_classes > 1:
            decoded_aug_vols[i] = np.expand_dims(pred_func(np.expand_dims(aug_vols[i], 0))[...,1], -1)
        else:
            decoded_aug_vols[i] = pred_func(np.expand_dims(aug_vols[i], 0))

    # Undo the combinations of the volume
    out_vols = []
    out_vols.append(np.array(decoded_aug_vols[0]))
    out_vols.append(rotate(np.array(decoded_aug_vols[1]), mode='reflect', axes=(1, 2), angle=-90, reshape=False))
    out_vols.append(rotate(np.array(decoded_aug_vols[2]), mode='reflect', axes=(1, 2), angle=-180, reshape=False))
    out_vols.append(rotate(np.array(decoded_aug_vols[3]), mode='reflect', axes=(1, 2), angle=-270, reshape=False))
    out_vols.append(np.flip(np.array(decoded_aug_vols[4]), 0))
    out_vols.append(np.flip(rotate(np.array(decoded_aug_vols[5]), mode='reflect', axes=(1, 2), angle=-90, reshape=False), 0))
    out_vols.append(np.flip(rotate(np.array(decoded_aug_vols[6]), mode='reflect', axes=(1, 2), angle=-180, reshape=False), 0))
    out_vols.append(np.flip(rotate(np.array(decoded_aug_vols[7]), mode='reflect', axes=(1, 2), angle=-270, reshape=False), 0))
    out_vols.append(np.flip(np.array(decoded_aug_vols[8]), 1))
    out_vols.append(np.flip(rotate(np.array(decoded_aug_vols[9]), mode='reflect', axes=(1, 2), angle=-90, reshape=False), 1))
    out_vols.append(np.flip(rotate(np.array(decoded_aug_vols[10]), mode='reflect', axes=(1, 2), angle=-180, reshape=False), 1))
    out_vols.append(np.flip(rotate(np.array(decoded_aug_vols[11]), mode='reflect', axes=(1, 2), angle=-270, reshape=False), 1))
    out_vols.append(np.flip(np.array(decoded_aug_vols[12]), 2))
    out_vols.append(np.flip(rotate(np.array(decoded_aug_vols[13]), mode='reflect', axes=(1, 2), angle=-90, reshape=False), 2))
    out_vols.append(np.flip(rotate(np.array(decoded_aug_vols[14]), mode='reflect', axes=(1, 2), angle=-180, reshape=False), 2))
    out_vols.append(np.flip(rotate(np.array(decoded_aug_vols[15]), mode='reflect', axes=(1, 2), angle=-270, reshape=False), 2))
  
    # Create the output data
    out_vols = np.array(out_vols) 
    if pad_to_square != 0:
        if pad_to_square < 0:
            out = np.zeros((out_vols.shape[0], volume.shape[0]+pad_to_square, 
                            volume.shape[1], volume.shape[2], volume.shape[3]))
        else:
            out = np.zeros((out_vols.shape[0], volume.shape[0], 
                            volume.shape[1]-pad_to_square, volume.shape[2], 
                            volume.shape[3]))
    else:
        out = np.zeros(out_vols.shape)

    # Undo the padding
    for i in range(out_vols.shape[0]):
        if pad_to_square < 0:
            out[i] = out_vols[i,abs(pad_to_square):,:,:,:]
        else:
            out[i] = out_vols[i,:,abs(pad_to_square):,:,:]

    return np.mean(out, axis=0)
