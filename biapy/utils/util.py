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

    Parameters
    ----------
    results : Keras History object
        Record of training loss values and metrics values at successive epochs. History object is returned by Keras
        `fit() <https://keras.io/api/models/model_training_apis/#fit-method>`_ method.

    metrics : List of str
        Metrics used.

    job_id : str
        Jod identifier.

    chartOutDir : str
        Path where the charts will be stored into.

    Examples
    --------
    +-----------------------------------------------+-----------------------------------------------+
    | .. figure:: ../../img/chart_loss.png          | .. figure:: ../../img/chart_jaccard_index.png |
    |   :width: 80%                                 |   :width: 80%                                 |
    |   :align: center                              |   :align: center                              |
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
    Create a plot with the different metric values binarizing the prediction with different thresholds, from ``0.1``
    to ``0.9``.

    Parameters
    ----------
    preds_test : 4D Numpy array
        Predictions made by the model. E.g. ``(num_of_images, y, x, channels)``.

    Y_test : 4D Numpy array
        Ground truth of the data. E.g. ``(num_of_images, y, x, channels)``.

    n_dig : int
        The number of digits used for encoding temporal indices (e.g. ``3``). Used by the DET calculation binary.

    job_id : str
        Id of the job.

    job_file : str
        Id and run number of the job.

    char_dir : str
        Path to store the charts generated.

    r_val : float, optional
        Threshold values to return.

    Returns
    -------
    t_jac : float
        Value of the Jaccard index when the threshold is ``r_val``.

    Examples
    --------
    ::

        jac, voc = threshold_plots(
            preds_test, Y_test, det_eval_ge_path, det_eval_path, det_bin,
            n_dig, args.job_id, '278_3', char_dir)

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
    Generates a weight map in order to make the U-Net learn better the borders of cells and distinguish individual
    cells that are tightly packed. These weight maps follow the methodology of the original U-Net paper.

    Based on `unet/py_files/helpers.py <https://github.com/deepimagej/python4deepimagej/blob/499955a264e1b66c4ed2c014cb139289be0e98a4/unet/py_files/helpers.py>`_.

    Parameters
    ----------

    label : 3D numpy array
       Corresponds to a label image. E.g. ``(y, x, channels)``.

    binary : bool, optional
       Corresponds to whether or not the labels are binary.

    w0 : float, optional
       Controls for the importance of separating tightly associated entities.

    sigma : int, optional
       Represents the standard deviation of the Gaussian used for the weight map.

    Examples
    --------

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
    Retrieves the label images, applies the weight-map algorithm and save the weight maps in a folder. Uses
    internally :meth:`util.make_weight_map`.

    Based on `deepimagejunet/py_files/helpers.py <https://github.com/deepimagej/python4deepimagej/blob/499955a264e1b66c4ed2c014cb139289be0e98a4/unet/py_files/helpers.py>`_.

    Parameters
    ----------
    labels : 4D numpy array
        Corresponds to given label images. E.g. ``(num_of_images, y, x, channels)``.

    path : str
        Refers to the path where the weight maps should be saved.

    binary : bool, optional
        Corresponds to whether or not the labels are binary.

    w0 : float, optional
        Controls for the importance of separating tightly associated entities.

    sigma : int, optional
        Represents the standard deviation of the Gaussian used for the weight
        map.
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
    Percentage of pixels that corresponds to the class in the given image.

    Parameters
    ----------
    mask : 2D Numpy array
        Image mask to analize.

    class_tag : int
        Class to find in the image.

    Returns
    -------
    x : float
        Percentage of pixels that corresponds to the class. Value between ``0``
        and ``1``.
    """

    c = 0
    for i in range(0, mask.shape[0]):
        for j in range(0, mask.shape[1]):
            if mask[i, j, 0] == class_tag:
                c = c + 1

    return c / (mask.shape[0] * mask.shape[1])


def divide_images_on_classes(data, data_mask, out_dir, num_classes=2, th=0.8):
    """
    Create a folder for each class where the images that have more pixels labeled as the class (in percentage) than
    the given threshold will be stored.

    Parameters
    ----------
    data : 4D numpy array
        Data to save as images. The first dimension must be the number of images. E. g.``(num_of_images, y, x, channels)``.

    data_mask : 4D numpy array
        Data mask to save as images.  The first dimension must be the number of images. E. g. ``(num_of_images, y, x, channels)``.

    out_dir : str
        Path to save the images.

    num_classes : int, optional
        Number of classes.

    th : float, optional
        Percentage of the pixels that must be labeled as a class to save it inside that class folder.
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
    Create an image of the filters learned by a convolutional layer. One can identify the layer with ``l_num`` or
    ``name`` args. If both are passed ``name`` will be prioritized.

    Inspired by https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks

    Parameters
    ----------
    model : Keras Model
        Model where the layers are stored.

    out_dir : str
        Path where the image will be stored.

    l_num : int, optional
        Number of the layer to extract filters from.

    name : str, optional
        Name of the layer to extract filters from.

    prefix : str, optional
        Prefix to add to the output image name.

    img_per_row : int, optional
        Filters per row on the image.

    Raises
    ------
    ValueError
        if ``l_num`` and ``name`` not provided.

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
    Ensures ``X`` shape is divisible by ``2`` times ``d_levels`` adding padding if necessary.

    Parameters
    ----------
    X : 4D Numpy array
        Data to check if its shape.  E.g. ``(10, 1000, 1000, 1)``.

    d_levels : int
        Levels of downsampling by ``2``.

    Returns
    -------
    X : 4D Numpy array
        Data divisible by 2 ``d_levels`` times.

    o_shape : 4 int tuple
        Original shape of ``X``. E.g. ``(10, 1000, 1000, 1)``.
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
    # Adapted from PyTorch for Connectomics:
    # https://github.com/zudi-lin/pytorch_connectomics/commit/6fbd5457463ae178ecd93b2946212871e9c617ee
    """
    Transform segmentation to 3D affinity graph.

    Parameters
    ----------
    img : Numpy array like
        3D indexed image, with each index corresponding to each segment.
    dz : int, optional
        Distance in voxels in the z direction to calculate affinity from.
    dy : int, optional
        Distance in voxels in the y direction to calculate affinity from.
    dx : int, optional
        Distance in voxels in the x direction to calculate affinity from.

    Returns
    -------
    ret : 4D Numpy array
        3D affinity graph (4D tensor), 3 channels for z, y, x direction.
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
    # Original code: https://github.com/torms3/DataProvider/blob/master/python/utils.py#L11
    """Ensure that data is numpy 3D array."""
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
    # Parameters
    M, N = A.shape
    # Get Starting block indices
    start_idx = np.arange(0, M - BSZ[0] + 1, stepsize)[:, None] * N + np.arange(0, N - BSZ[1] + 1, stepsize)
    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(BSZ[0])[:, None] * N + np.arange(BSZ[1])
    # Get all actual indices & index into input array for final output
    return np.take(A, start_idx.ravel()[:, None] + offset_idx.ravel())


def seg_widen_border(seg, tsz_h=1):
    # Kisuk Lee's thesis (A.1.4):
    # "we preprocessed the ground truth seg such that any voxel centered on a 3 × 3 × 1 window containing
    # more than one positive segment ID (zero is reserved for background) is marked as background."
    # seg=0: background
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
    h = sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(filename, "rb", buffering=0) as f:
        while n := f.readinto(mv):
            h.update(mv[:n])
    return h.hexdigest()
