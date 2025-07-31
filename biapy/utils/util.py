"""
This module provides a collection of utility functions for image processing,
data manipulation, and visualization, primarily geared towards bioimage analysis
workflows.

It includes functionalities for:
- Generating plots for training loss and metrics.
- Creating threshold-based metric plots.
- Generating weight maps for U-Net-like models to handle object boundaries.
- Organizing images into class-specific folders based on foreground percentage.
- Visualizing learned filters of convolutional layers.
- Ensuring image dimensions are divisible by a given factor for downsampling.
- Converting segmentation masks to affinity graphs (for 3D data).
- Validating and reshaping image volumes.
- Implementing `im2col` for patch extraction.
- Widening segmentation borders.
- Calculating SHA256 checksums for files.
"""
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import copy
from PIL import Image
from tqdm import tqdm
from skimage import measure
from hashlib import sha256
from numpy.typing import NDArray, DTypeLike

from biapy.engine.metrics import jaccard_index_numpy
from biapy.utils.misc import is_main_process


def create_plots(results, metrics, job_id, chartOutDir):
    """
    Create loss and main metric plots with the given results.

    This function visualizes the training and validation loss, as well as
    training and validation values for each given metric across epochs.
    Plots are saved as PNG images in the specified output directory.

    Parameters
    ----------
    results : Dict
        A dictionary containing training history. Expected keys are 'loss',
        'val_loss' (optional), and entries for each metric (e.g., 'jaccard_index')
        and its validation counterpart (e.g., 'val_jaccard_index').
    metrics : List[str]
        A list of metric names (e.g., ["jaccard_index", "f1_score"]) present in `results`.
    job_id : str
        A unique identifier for the job, used in plot titles and filenames.
    chartOutDir : str
        The directory where the generated chart images will be stored.

    Examples
    --------
    >>> # Assuming 'results' is a dictionary like:
    >>> # {'loss': [...], 'val_loss': [...], 'jaccard_index': [...], 'val_jaccard_index': [...]}
    >>> # create_plots(results, ['jaccard_index'], 'my_experiment', './charts/')

    +-----------------------------------------------+-----------------------------------------------+
    | .. figure:: ../../img/chart_loss.png          | .. figure:: ../../img/chart_jaccard_index.png |
    |   :width: 80%                                 |   :align: center                              |
    |                                               |                                               |
    |   Loss values on each epoch                   |   Jaccard index values on each epoch          |
    +-----------------------------------------------+-----------------------------------------------+
    """

    print("Creating training plots . . .")
    os.makedirs(chartOutDir, exist_ok=True)

    # For matplotlib errors in display
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    # Loss
    plt.plot(results["loss"])
    if "val_loss" in results:
        plt.plot(results["val_loss"])
    plt.title("Model JOBID=" + job_id + " loss")
    plt.ylabel("Value")
    plt.xlabel("Epoch")
    if "val_loss" in results:
        plt.legend(["Train loss", "Val. loss"], loc="upper left")
    else:
        plt.legend(["Train loss"], loc="upper left")
    plt.savefig(os.path.join(chartOutDir, job_id + "_loss.png"))
    plt.clf()

    # Metric
    for i in range(len(metrics)):
        plt.plot(results[metrics[i]])
        plt.plot(results["val_" + metrics[i]])
        plt.title("Model JOBID=" + job_id + " " + metrics[i])
        plt.ylabel("Value")
        plt.xlabel("Epoch")
        plt.legend([f"Train {metrics[i]}", f"Val. {metrics[i]}"], loc="upper left")
        plt.savefig(os.path.join(chartOutDir, job_id + "_" + metrics[i] + ".png"))
        plt.clf()


def threshold_plots(preds_test, Y_test, n_dig, job_id, job_file, char_dir, r_val=0.5):
    """
    Generate plots showing metric values (e.g., Jaccard index) across different binarization thresholds applied to predictions.

    The predictions are binarized using thresholds from 0.1 to 0.9 (inclusive, step 0.1).
    For each threshold, the Jaccard index is calculated against the ground truth.
    A plot is generated visualizing these metric values.

    Parameters
    ----------
    preds_test : NDArray
        Predictions made by the model, typically a 4D NumPy array
        of shape `(num_of_images, y, x, channels)` with float values.
    Y_test : NDArray
        Ground truth masks, typically a 4D NumPy array
        of shape `(num_of_images, y, x, channels)` with integer labels.
    n_dig : int
        The number of digits used for encoding temporal indices (e.g., `3`).
        This parameter seems to be a remnant from a previous use case (DET calculation binary)
        and might not be directly used in the current function's logic, but kept for compatibility.
    job_id : str
        Identifier for the job.
    job_file : str
        Combined identifier for the job and run number (e.g., "278_3"), used in filenames.
    char_dir : str
        Path to the directory where the generated charts will be stored.
    r_val : float, optional
        A specific threshold value (between 0.1 and 0.9) for which the Jaccard index
        will be returned. Defaults to 0.5.

    Returns
    -------
    float
        The Jaccard index value obtained when binarizing predictions with the `r_val` threshold.

    Examples
    --------
    >>> # Assuming preds_test and Y_test are loaded NumPy arrays
    >>> # t_jac_at_0_5 = threshold_plots(preds_test, Y_test, 3, 'my_job', 'my_job_run1', './threshold_charts/', r_val=0.5)

    Will generate one chart for the IoU. In the x axis represents the 9 different thresholds applied, that is:
    ``0.1, 0.2, 0.3, ..., 0.9``. The y axis is the value of the metric in each chart. For instance, the Jaccard/IoU
    chart will look like this:

    .. image:: ../../img/278_3_threshold_Jaccard.png
        :width: 60%
        :align: center

    In this example, the best value, ``0.868``, is obtained with a threshold of ``0.4``.
    """

    char_dir = os.path.join(char_dir, "t_" + job_file)

    t_jac = np.zeros(9)
    objects = []
    r_val_pos = 0

    for i, t in enumerate(np.arange(0.1, 1.0, 0.1)):

        if t == r_val:
            r_val_pos = i

        objects.append(str("%.2f" % float(t)))

        # Threshold images
        bin_preds_test = (preds_test > t).astype(np.uint8)

        print("Calculate metrics . . .")
        t_jac[i] = jaccard_index_numpy(Y_test, bin_preds_test)

        print("t_jac[{}]: {}".format(i, t_jac[i]))

    # For matplotlib errors in display
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    os.makedirs(char_dir, exist_ok=True)

    # Plot Jaccard values
    plt.clf()
    plt.plot(objects, t_jac)
    plt.title("Model JOBID=" + job_file + " Jaccard", y=1.08)
    plt.ylabel("Value")
    plt.xlabel("Threshold")
    for k, point in enumerate(zip(objects, t_jac)):
        plt.text(point[0], point[1], "%.3f" % float(t_jac[k]))
    plt.savefig(os.path.join(char_dir, job_file + "_threshold_Jaccard.png"))
    plt.clf()

    return t_jac[r_val_pos]


def make_weight_map(label, binary=True, w0=10, sigma=5):
    """
    Generate a weight map for semantic segmentation, particularly useful for separating tightly packed objects, following the methodology of the original U-Net paper.

    The weight map `W(x)` is a sum of two components:

    1. A class balancing map `W_c(x)`: assigns higher weight to foreground pixels.
    2. A distance-based map: `w0 * exp(-((d1 + d2)^2) / (2 * sigma^2))`. This component
       is high near boundaries between touching objects, where `d1` is the distance
       to the closest object and `d2` is the distance to the second closest object.

    Based on `unet/py_files/helpers.py <https://github.com/deepimagej/python4deepimagej/blob/499955a264e1b66c4ed2c014cb139289be0e98a4/unet/py_files/helpers.py>`_.

    Parameters
    ----------
    label : NDArray
       A 2D or 3D NumPy array representing a label image. If 3D, it's assumed
       to be `(y, x, channels)` and only the first channel is used.
       Objects are typically labeled with unique positive integers, background is 0.
    binary : bool, optional
       If True, the input `label` is treated as a binary mask (0 for background,
       >0 for foreground) and then distinct objects are extracted. If False,
       it's assumed `label` already contains distinct object IDs (or 0/1 for binary).
       Defaults to True.
    w0 : float, optional
       Weight factor controlling the importance of the distance-based component
       for separating tightly associated entities. Defaults to 10.
    sigma : int, optional
       Standard deviation of the Gaussian function used in the distance-based
       component. Controls the spread of the boundary weights. Defaults to 5.

    Returns
    -------
    NDArray
        A 2D NumPy array representing the generated weight map, with the same
        spatial dimensions as the input `label`.

    Examples
    --------
    >>> # Assuming 'label_image' is a 2D NumPy array with object labels
    >>> # weight_map = make_weight_map(label_image, binary=True, w0=10, sigma=5)

    Notice that weight has been defined where the objects are almost touching
    each other.

    .. image:: ../../img/weight_map.png
        :width: 650
        :align: center
        
    """
    # Initialization.
    lab = np.array(label)
    lab_multi = lab

    if len(lab.shape) == 3:
        lab = lab[:, :, 0]

    # Get shape of label.
    rows, cols = lab.shape

    if binary:
        # Converts the label into a binary image with background = 0
        # and cells = 1.
        lab[lab == 255] = 1

        # Builds w_c which is the class balancing map. In our case, we want
        # cells to have weight 2 as they are more important than background
        # which is assigned weight 1.
        w_c = np.array(lab, dtype=float)
        w_c[w_c == 1] = 1
        w_c[w_c == 0] = 0.5

        # Converts the labels to have one class per object (cell).
        lab_multi = measure.label(lab, connectivity=8, background=0)
        assert isinstance(lab_multi, np.ndarray)
        components = np.unique(lab_multi)
    else:
        # Converts the label into a binary image with background = 0.
        # and cells = 1.
        lab[lab > 0] = 1

        # Builds w_c which is the class balancing map. In our case, we want
        # cells to have weight 2 as they are more important than background
        # which is assigned weight 1.
        w_c = np.array(lab, dtype=float)
        w_c[w_c == 1] = 1
        w_c[w_c == 0] = 0.5
        components = np.unique(lab)

    n_comp = len(components) - 1

    maps = np.zeros((n_comp, rows, cols))

    map_weight = np.zeros((rows, cols))

    if n_comp >= 2:
        for i in range(n_comp):

            # Only keeps current object.
            tmp = lab_multi == components[i + 1]

            # Invert tmp so that it can have the correct distance.
            # transform
            tmp = ~tmp

            # For each pixel, computes the distance transform to
            # each object.
            maps[i][:][:] = scipy.ndimage.distance_transform_edt(tmp)

        maps = np.sort(maps, axis=0)

        # Get distance to the closest object (d1) and the distance to the second
        # object (d2).
        d1 = maps[0][:][:]
        d2 = maps[1][:][:]

        map_weight = w0 * np.exp(-((d1 + d2) ** 2) / (2 * (sigma**2))) * (lab == 0).astype(int)

    map_weight += w_c

    return map_weight


def do_save_wm(labels, path, binary=True, w0=10, sigma=5):
    """
    Generate weight maps for a batch of label images and save them as NumPy files.

    This function iterates through a 4D array of label images, applies the
    `make_weight_map` function to each, and saves the resulting weight maps
    into a specified directory structure.

    Based on `deepimagejunet/py_files/helpers.py <https://github.com/deepimagej/python4deepimagej/blob/499955a264e1b66c4ed2c014cb139289be0e98a4/unet/py_files/helpers.py>`_.

    Parameters
    ----------
    labels : NDArray
        A 4D NumPy array of label images, typically `(num_of_images, y, x, channels)`.
    path : str
        The base directory where the weight maps should be saved. A subdirectory
        named "weight" will be created within this path.
    binary : bool, optional
        Corresponds to whether or not the labels are binary, passed to `make_weight_map`.
        Defaults to True.
    w0 : float, optional
        Controls the importance of separating tightly associated entities, passed to `make_weight_map`.
        Defaults to 10.
    sigma : int, optional
        Represents the standard deviation of the Gaussian used for the weight map,
        passed to `make_weight_map`. Defaults to 5.
    """
    # Copy labels.
    labels_ = copy.deepcopy(labels)

    # Perform weight maps.
    for i in range(len(labels_)):
        labels_[i] = make_weight_map(labels[i].copy(), binary, w0, sigma)

    maps = np.array(labels_)

    n, rows, cols = maps.shape

    # Resize correctly the maps so that it can be used in the model.
    maps = maps.reshape((n, rows, cols, 1))

    # Count number of digits in n. This is important for the number
    # of leading zeros in the name of the maps.
    n_digits = len(str(n))

    # Save path with correct leading zeros.
    path_to_save = path + "weight/{b:0" + str(n_digits) + "d}.npy"

    # Saving files as .npy files.
    for i in range(len(labels_)):
        np.save(path_to_save.format(b=i), labels_[i])

    return None


def foreground_percentage(mask, class_tag):
    """
    Calculate the percentage of pixels in a given mask that correspond to a specific class.

    Parameters
    ----------
    mask : NDArray
        A 2D or 3D NumPy array representing an image mask. If 3D, it's assumed
        to be `(y, x, channels)` and only the first channel is used.
    class_tag : int
        The integer label of the class to count.

    Returns
    -------
    float
        The percentage of pixels labeled as `class_tag` in the mask,
        as a value between 0.0 and 1.0.
    """
    c = 0
    for i in range(0, mask.shape[0]):
        for j in range(0, mask.shape[1]):
            if mask[i, j, 0] == class_tag:
                c = c + 1

    return c / (mask.shape[0] * mask.shape[1])


def divide_images_on_classes(data, data_mask, out_dir, num_classes=2, th=0.8):
    """
    Organize images into class-specific folders based on the percentage of
    foreground pixels belonging to each class in their corresponding masks.

    For each class, a subdirectory is created. An image and its mask are
    saved into a class's folder if the percentage of pixels labeled as that
    class in the mask exceeds a given threshold.

    Parameters
    ----------
    data : NDArray
        A 4D NumPy array of input images, typically `(num_of_images, y, x, channels)`.
        Only the first channel `data[:,:,:,0]` is used for saving.
    data_mask : NDArray
        A 4D NumPy array of corresponding mask images, typically `(num_of_images, y, x, channels)`.
        Only the first channel `data_mask[:,:,:,0]` is used for analysis and saving.
    out_dir : str
        The base path where the class-specific folders ("x/classX" and "y/classX")
        will be created and images saved.
    num_classes : int, optional
        The total number of classes to consider (from 0 to `num_classes - 1`).
        Defaults to 2.
    th : float, optional
        The minimum percentage (between 0.0 and 1.0) of pixels that must be labeled
        as a specific class in a mask for its corresponding image and mask to be
        saved into that class's folder. Defaults to 0.8.
    """
    # Create the directories
    for i in range(num_classes):
        os.makedirs(os.path.join(out_dir, "x", "class" + str(i)), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "y", "class" + str(i)), exist_ok=True)

    print("Dividing provided data into {} classes . . .".format(num_classes))
    d = len(str(data.shape[0]))
    for i in tqdm(range(data.shape[0]), disable=not is_main_process()):
        # Assign the image to a class if it has, in percentage, more pixels of
        # that class than the given threshold
        for j in range(num_classes):
            t = foreground_percentage(data_mask[i], j)
            if t > th:
                im = Image.fromarray(data[i, :, :, 0])
                im = im.convert("L")
                im.save(
                    os.path.join(
                        os.path.join(out_dir, "x", "class" + str(j)),
                        "im_" + str(i).zfill(d) + ".png",
                    )
                )
                im = Image.fromarray(data_mask[i, :, :, 0] * 255)
                im = im.convert("L")
                im.save(
                    os.path.join(
                        os.path.join(out_dir, "y", "class" + str(j)),
                        "mask_" + str(i).zfill(d) + ".png",
                    )
                )


def save_filters_of_convlayer(model, out_dir, l_num=None, name=None, prefix="", img_per_row=8):
    """
    Create and save an image visualizing the filters learned by a specific convolutional layer within a Keras model.

    The layer can be identified by its numerical index (`l_num`) or its name (`name`).
    If both are provided, `name` takes precedence. The filters are normalized
    to 0-1 for visualization and arranged in a grid.

    Inspired by https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks

    Parameters
    ----------
    model : Any
        The Keras Model object containing the layers.
    out_dir : str
        The directory where the output image will be stored.
    l_num : Optional[int], optional
        The numerical index of the convolutional layer to extract filters from.
        Defaults to None.
    name : Optional[str], optional
        The name of the convolutional layer to extract filters from.
        Defaults to None.
    prefix : str, optional
        A string prefix to add to the output image filename. Defaults to "".
    img_per_row : int, optional
        The number of filters to display per row in the output image grid.
        Defaults to 8.

    Raises
    ------
    ValueError
        If neither `l_num` nor `name` is provided.

    Examples
    --------
    To save the filters learned by the layer called ``conv1`` one can call
    the function as follows ::

        save_filters_of_convlayer(model, char_dir, name="conv1", prefix="model")

    That will save in ``out_dir`` an image like this:

    .. image:: ../../img/save_filters.png
        :width: 60%
        :align: center
    """

    if l_num is None and name is None:
        raise ValueError("One between 'l_num' or 'name' must be provided")

    # For matplotlib errors in display
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    # Find layer number of the layer named by 'name' variable
    if name is not None:
        pos = 0
        for layer in model.layers:
            if name == layer.name:
                break
            pos += 1
        l_num = pos

    filters, biases = model.layers[l_num].get_weights()

    # normalize filter values to 0-1 so we can visualize them
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)

    rows = int(math.floor(filters.shape[3] / img_per_row))
    i = 0
    for r in range(rows):
        for c in range(img_per_row):
            ax = plt.subplot(rows, img_per_row, i + 1)
            ax.set_xticks([])
            ax.set_yticks([])
            f = filters[:, :, 0, i]
            plt.imshow(filters[:, :, 0, i], cmap="gray")

            i += 1

    prefix += "_" if prefix != "" else prefix
    plt.savefig(os.path.join(out_dir, prefix + "f_layer" + str(l_num) + ".png"))
    plt.clf()


def check_downsample_division(X, d_levels):
    """
    Ensure that the spatial dimensions of a 4D NumPy array `X` are divisible by `2` raised to the power of `d_levels`. Padding is applied if necessary.

    This is crucial for U-Net like architectures or other models that perform
    multiple levels of downsampling (e.g., pooling layers).

    Parameters
    ----------
    X : NDArray
        The input data, a 4D NumPy array with shape `(num_images, height, width, channels)`.
    d_levels : int
        The number of downsampling levels (e.g., if `d_levels=3`, dimensions
        must be divisible by `2^3 = 8`).

    Returns
    -------
    X_padded : NDArray
        The padded data, with spatial dimensions divisible by `2^d_levels`.
    original_shape : Tuple[int, ...]
        The original shape of the input `X`.
    """
    d_val = pow(2, d_levels)
    dy = math.ceil(X.shape[1] / d_val)
    dx = math.ceil(X.shape[2] / d_val)
    o_shape = X.shape
    if dy * d_val != X.shape[1] or dx * d_val != X.shape[2]:
        X = np.pad(
            X,
            (
                (0, 0),
                (0, (dy * d_val) - X.shape[1]),
                (0, (dx * d_val) - X.shape[2]),
                (0, 0),
            ),
        )
        print("Data has been padded to be downsampled {} times. Its shape now is: {}".format(d_levels, X.shape))
    return X, o_shape


def seg2aff_pni(img, dz=1, dy=1, dx=1, 
    dtype: DTypeLike = np.float32):
    """
    Transform a 3D segmentation mask into a 3D affinity graph (4D tensor).

    The affinity graph has 3 channels corresponding to affinities in the z, y, and x directions.
    An affinity value is 1 if two adjacent voxels (at specified distances `dz`, `dy`, `dx`)
    belong to the same segment (and are not background, i.e., label > 0), and 0 otherwise.

    Adapted from PyTorch for Connectomics:
    https://github.com/zudi-lin/pytorch_connectomics/commit/6fbd5457463ae178ecd93b2946212871e9c617ee

    Parameters
    ----------
    img : NDArray
        A 3D NumPy array representing an indexed image, where each index
        corresponds to a unique segment. Background is typically 0.
    dz : int, optional
        Distance in voxels in the z (depth) direction to calculate affinity from.
        Must be less than `img.shape[-3]`. Defaults to 1.
    dy : int, optional
        Distance in voxels in the y (height) direction to calculate affinity from.
        Must be less than `img.shape[-2]`. Defaults to 1.
    dx : int, optional
        Distance in voxels in the x (width) direction to calculate affinity from.
        Must be less than `img.shape[-1]`. Defaults to 1.
    dtype : DTypeLike, optional
        The desired data type for the output affinity map. Defaults to `np.float32`.

    Returns
    -------
    ret : NDArray
        A 4D NumPy array representing the 3D affinity graph, with shape
        `(3, D, H, W)` where the first dimension corresponds to z, y, x affinities.

    Raises
    ------
    AssertionError
        If `dz`, `dy`, or `dx` are zero or exceed the corresponding image dimension.
    """
    img = check_volume(img)
    ret = np.zeros((3,) + img.shape, dtype=dtype)

    # z-affinity.
    assert dz and abs(dz) < img.shape[-3]
    if dz > 0:
        ret[0, dz:, :, :] = (img[dz:, :, :] == img[:-dz, :, :]) & (img[dz:, :, :] > 0)
    else:
        dz = abs(dz)
        ret[0, :-dz, :, :] = (img[dz:, :, :] == img[:-dz, :, :]) & (img[dz:, :, :] > 0)

    # y-affinity.
    assert dy and abs(dy) < img.shape[-2]
    if dy > 0:
        ret[1, :, dy:, :] = (img[:, dy:, :] == img[:, :-dy, :]) & (img[:, dy:, :] > 0)
    else:
        dy = abs(dy)
        ret[1, :, :-dy, :] = (img[:, dy:, :] == img[:, :-dy, :]) & (img[:, dy:, :] > 0)

    # x-affinity.
    assert dx and abs(dx) < img.shape[-1]
    if dx > 0:
        ret[2, :, :, dx:] = (img[:, :, dx:] == img[:, :, :-dx]) & (img[:, :, dx:] > 0)
    else:
        dx = abs(dx)
        ret[2, :, :, :-dx] = (img[:, :, dx:] == img[:, :, :-dx]) & (img[:, :, dx:] > 0)

    return ret


def check_volume(data):
    """
    Ensure that the input data is a 3D NumPy array.

    If the input is 2D, it adds a new z-axis. If it's 4D with a batch size of 1,
    it reshapes it to 3D by removing the batch dimension. Raises an error for
    other dimensions.

    Original code: https://github.com/torms3/DataProvider/blob/master/python/utils.py#L11

    Parameters
    ----------
    data : NDArray
        The input data array. Can be 2D, 3D, or 4D (with batch size 1).

    Returns
    -------
    NDArray
        A 3D NumPy array.

    Raises
    ------
    RuntimeError
        If `data` is not a NumPy array or has an unsupported number of dimensions.
    AssertionError
        If `data` is 4D but its batch dimension is not 1.
    """
    assert isinstance(data, np.ndarray)

    if data.ndim == 2:
        data = data[np.newaxis, ...]
    elif data.ndim == 3:
        pass
    elif data.ndim == 4:
        assert data.shape[0] == 1
        data = np.reshape(data, data.shape[-3:])
    else:
        raise RuntimeError("data must be a numpy 3D array")

    assert data.ndim == 3
    return data


def im2col(A, BSZ, stepsize=1):
    """
    Implement the `im2col` (image to column) operation, which extracts sliding windows (patches) from an input 2D array and arranges them as columns in a new 2D array.

    This is a common operation in convolutional neural networks for efficient
    convolution implementation.

    Parameters
    ----------
    A : NDArray
        The input 2D NumPy array (image).
    BSZ : Tuple[int, int]
        A tuple `(patch_height, patch_width)` specifying the size of the sliding window.
    stepsize : int, optional
        The stride (step size) for sliding the window. Defaults to 1.

    Returns
    -------
    NDArray
        A 2D NumPy array where each row is a flattened patch from the input `A`.
    """
    # Parameters
    M, N = A.shape
    # Get Starting block indices
    start_idx = np.arange(0, M - BSZ[0] + 1, stepsize)[:, None] * N + np.arange(0, N - BSZ[1] + 1, stepsize)
    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(BSZ[0])[:, None] * N + np.arange(BSZ[1])
    # Get all actual indices & index into input array for final output
    return np.take(A, start_idx.ravel()[:, None] + offset_idx.ravel())


def seg_widen_border(seg, tsz_h=1):
    """
    Widen the border of segments in a label image by marking pixels as background if they are at the boundary between two different segments.

    This is based on Kisuk Lee's thesis (A.1.4): "we preprocessed the ground truth seg such that any voxel centered
    on a 3 x 3 x 1 window containing more than one positive segment ID (zero is reserved for background) is
    marked as background."

    Parameters
    ----------
    seg : NDArray
        The input label image (2D or 3D NumPy array). Background is 0, segments are positive integers.
    tsz_h : int, optional
        Half-size of the square/cube window used to check for multiple segment IDs.
        A `tsz_h=1` corresponds to a 3x3 (or 3x3x3 for 3D) window. Defaults to 1.

    Returns
    -------
    NDArray
        The label image with widened segment borders (boundary pixels set to 0).
    """
    tsz = 2 * tsz_h + 1
    sz = seg.shape
    if len(sz) == 3:
        for z in range(sz[0]):
            mm = seg[z].max()
            patch = im2col(np.pad(seg[z], ((tsz_h, tsz_h), (tsz_h, tsz_h)), "reflect"), [tsz, tsz])
            p0 = patch.max(axis=1)
            patch[patch == 0] = mm + 1
            p1 = patch.min(axis=1)
            seg[z] = seg[z] * ((p0 == p1).reshape(sz[1:]))
    else:
        mm = seg.max()
        patch = im2col(np.pad(seg, ((tsz_h, tsz_h), (tsz_h, tsz_h)), "reflect"), [tsz, tsz])
        p0 = patch.max(axis=1)
        patch[patch == 0] = mm + 1
        p1 = patch.min(axis=1)
        seg = seg * ((p0 == p1).reshape(sz))
    return seg


def create_file_sha256sum(
    filename: str
) -> str:
    """
    Calculate the SHA256 checksum of a given file.

    This function reads the file in chunks to efficiently compute the hash,
    even for large files, without loading the entire file into memory.

    Parameters
    ----------
    filename : str
        The path to the file for which to calculate the SHA256 sum.

    Returns
    -------
    str
        The hexadecimal SHA256 checksum of the file.
    """
    h = sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(filename, "rb", buffering=0) as f:
        while n := f.readinto(mv):
            h.update(mv[:n])
    return h.hexdigest()
