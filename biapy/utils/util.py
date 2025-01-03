import os
import math
import numpy as np
import h5py
import zarr
import matplotlib.pyplot as plt
import scipy.ndimage
import copy
from PIL import Image
from tqdm import tqdm
from skimage.io import imsave
from skimage import measure
from hashlib import sha256

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


def save_tif(X, data_dir=None, filenames=None, verbose=True):
    """
    Save images in the given directory.

    Parameters
    ----------
    X : 4D/5D numpy array
        Data to save as images. The first dimension must be the number of images. E.g.
        ``(num_of_images, y, x, channels)`` or ``(num_of_images, z, y, x, channels)``.

    data_dir : str, optional
        Path to store X images.

    filenames : List, optional
        Filenames that should be used when saving each image.

    verbose : bool, optional
         To print saving information.
    """

    if verbose:
        s = X.shape if not isinstance(X, list) else X[0].shape
        print("Saving {} data as .tif in folder: {}".format(s, data_dir))

    os.makedirs(data_dir, exist_ok=True)
    if filenames is not None:
        if len(filenames) != len(X):
            raise ValueError(
                "Filenames array and length of X have different shapes: {} vs {}".format(len(filenames), len(X))
            )

    if not isinstance(X, list):
        _dtype = X.dtype if X.dtype in [np.uint8, np.uint16, np.float32] else np.float32
        ndims = X.ndim
    else:
        _dtype = X[0].dtype if X[0].dtype in [np.uint8, np.uint16, np.float32] else np.float32
        ndims = X[0].ndim

    d = len(str(len(X)))
    for i in tqdm(range(len(X)), leave=False, disable=not is_main_process()):
        if filenames is None:
            f = os.path.join(data_dir, str(i).zfill(d) + ".tif")
        else:
            f = os.path.join(data_dir, os.path.splitext(filenames[i])[0] + ".tif")
        if ndims == 4:
            if not isinstance(X, list):
                aux = np.expand_dims(np.expand_dims(X[i], 0).transpose((0, 3, 1, 2)), -1).astype(_dtype)
            else:
                aux = np.expand_dims(np.expand_dims(X[i][0], 0).transpose((0, 3, 1, 2)), -1).astype(_dtype)
        else:
            if not isinstance(X, list):
                aux = np.expand_dims(X[i].transpose((0, 3, 1, 2)), -1).astype(_dtype)
            else:
                aux = np.expand_dims(X[i][0].transpose((0, 3, 1, 2)), -1).astype(_dtype)
        try:
            imsave(
                f,
                np.expand_dims(aux, 0),
                imagej=True,
                metadata={"axes": "TZCYXS"},
                check_contrast=False,
                compression=("zlib", 1),
            )
        except:
            imsave(
                f,
                np.expand_dims(aux, 0),
                imagej=True,
                metadata={"axes": "TZCYXS"},
                check_contrast=False,
            )


def save_tif_pair_discard(X, Y, data_dir=None, suffix="", filenames=None, discard=True, verbose=True):
    """
    Save images in the given directory.

    Parameters
    ----------
    X : 4D/5D numpy array
        Data to save as images. The first dimension must be the number of images. E.g.
        ``(num_of_images, y, x, channels)`` or ``(num_of_images, z, y, x, channels)``.

    Y : 4D/5D numpy array
        Data mask to save. The first dimension must be the number of images. E.g.
        ``(num_of_images, y, x, channels)`` or ``(num_of_images, z, y, x, channels)``.

    data_dir : str, optional
        Path to store X images.

    suffix : str, optional
        Suffix to apply on output directory.

    filenames : List, optional
        Filenames that should be used when saving each image.

    discard : bool, optional
        Whether to discard image/mask pairs if the mask has no label information.

    verbose : bool, optional
         To print saving information.
    """

    if verbose:
        s = X.shape if not isinstance(X, list) else X[0].shape
        print("Saving {} data as .tif in folder: {}".format(s, data_dir))

    os.makedirs(os.path.join(data_dir, "x" + suffix), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "y" + suffix), exist_ok=True)
    if filenames is not None:
        if len(filenames) != len(X):
            raise ValueError(
                "Filenames array and length of X have different shapes: {} vs {}".format(len(filenames), len(X))
            )

    _dtype = X.dtype if X.dtype in [np.uint8, np.uint16, np.float32] else np.float32
    d = len(str(len(X)))
    for i in tqdm(range(X.shape[0]), leave=False, disable=not is_main_process()):
        if len(np.unique(Y[i])) >= 2 or not discard:
            if filenames is None:
                f1 = os.path.join(data_dir, "x" + suffix, str(i).zfill(d) + ".tif")
                f2 = os.path.join(data_dir, "y" + suffix, str(i).zfill(d) + ".tif")
            else:
                f1 = os.path.join(data_dir, "x" + suffix, os.path.splitext(filenames[i])[0] + ".tif")
                f2 = os.path.join(data_dir, "y" + suffix, os.path.splitext(filenames[i])[0] + ".tif")
            if X.ndim == 4:
                aux = np.expand_dims(np.expand_dims(X[i], 0).transpose((0, 3, 1, 2)), -1).astype(_dtype)
            else:
                aux = np.expand_dims(X[i].transpose((0, 3, 1, 2)), -1).astype(_dtype)
            imsave(
                f1,
                np.expand_dims(aux, 0),
                imagej=True,
                metadata={"axes": "TZCYXS"},
                check_contrast=False,
                compression=("zlib", 1),
            )
            if Y.ndim == 4:
                aux = np.expand_dims(np.expand_dims(Y[i], 0).transpose((0, 3, 1, 2)), -1).astype(_dtype)
            else:
                aux = np.expand_dims(Y[i].transpose((0, 3, 1, 2)), -1).astype(_dtype)
            imsave(
                f2,
                np.expand_dims(aux, 0),
                imagej=True,
                metadata={"axes": "TZCYXS"},
                check_contrast=False,
                compression=("zlib", 1),
            )


def save_img(
    X=None,
    data_dir=None,
    Y=None,
    mask_dir=None,
    scale_mask=True,
    prefix="",
    extension=".png",
    filenames=None,
):
    """
    Save images in the given directory.

    Parameters
    ----------
    X : 4D numpy array, optional
        Data to save as images. The first dimension must be the number of images. E.g. ``(num_of_images, y, x, channels)``.

    data_dir : str, optional
        Path to store X images.

    Y : 4D numpy array, optional
        Masks to save as images. The first dimension must be the number of images. E.g. ``(num_of_images, y, x, channels)``.

    scale_mask : bool, optional
        To allow mask be multiplied by 255.

    mask_dir : str, optional
        Path to store Y images.

    prefix : str, optional
        Path to store generated charts.

    filenames : list, optional
        Filenames that should be used when saving each image. If any provided each image should be named as:
        ``prefix + "_x_" + image_number + extension`` when ``X.ndim < 4`` and ``prefix + "_x_" + image_number +
        "_" + slice_numger + extension`` otherwise. E.g. ``prefix_x_000.png`` when ``X.ndim < 4`` or
        ``prefix_x_000_000.png`` when ``X.ndim >= 4``.  The same applies to ``Y``.
    """

    if prefix == "":
        p_x = "x_"
        p_y = "y_"
    else:
        p_x = prefix + "_x_"
        p_y = prefix + "_y_"

    if X is not None:
        if data_dir is not None:
            os.makedirs(data_dir, exist_ok=True)
        else:
            print("Not data_dir provided so no image will be saved!")
            return

        print("Saving images in {}".format(data_dir))

        v = 1 if np.max(X) > 2 else 255
        if X.ndim > 4:
            d = len(str(X.shape[0] * X.shape[3]))
            for i in tqdm(range(X.shape[0]), disable=not is_main_process()):
                for j in range(X.shape[3]):
                    if X.shape[-1] == 1:
                        im = Image.fromarray((X[i, :, :, j, 0] * v).astype(np.uint8))
                        im = im.convert("L")
                    else:
                        im = Image.fromarray((X[i, :, :, j] * v).astype(np.uint8), "RGB")

                    if filenames is None:
                        f = os.path.join(
                            data_dir,
                            p_x + str(i).zfill(d) + "_" + str(j).zfill(d) + extension,
                        )
                    else:
                        f = os.path.join(data_dir, filenames[(i * j) + j] + extension)
                    im.save(f)
        else:
            d = len(str(X.shape[0]))
            for i in tqdm(range(X.shape[0]), disable=not is_main_process()):
                if X.shape[-1] == 1:
                    im = Image.fromarray((X[i, :, :, 0] * v).astype(np.uint8))
                    im = im.convert("L")
                else:
                    im = Image.fromarray((X[i] * v).astype(np.uint8), "RGB")

                if filenames is None:
                    f = os.path.join(data_dir, p_x + str(i).zfill(d) + extension)
                else:
                    f = os.path.join(data_dir, filenames[i] + extension)
                im.save(f)

    if Y is not None:
        if mask_dir is not None:
            os.makedirs(mask_dir, exist_ok=True)
        else:
            print("Not mask_dir provided so no image will be saved!")
            return

        print("Saving images in {}".format(mask_dir))

        v = 1 if np.max(Y) > 2 or not scale_mask else 255
        if Y.ndim > 4:
            d = len(str(Y.shape[0] * Y.shape[3]))
            for i in tqdm(range(Y.shape[0]), disable=not is_main_process()):
                for j in range(Y.shape[3]):
                    for k in range(Y.shape[-1]):
                        im = Image.fromarray((Y[i, :, :, j, k] * v).astype(np.uint8))
                        im = im.convert("L")
                        if filenames is None:
                            c = "" if Y.shape[-1] == 1 else "_c" + str(j)
                            f = os.path.join(
                                mask_dir,
                                p_y + str(i).zfill(d) + "_" + str(j).zfill(d) + c + extension,
                            )
                        else:
                            f = os.path.join(data_dir, filenames[(i * j) + j] + extension)

                        im.save(f)
        else:
            d = len(str(Y.shape[0]))
            for i in tqdm(range(0, Y.shape[0]), disable=not is_main_process()):
                for j in range(Y.shape[-1]):
                    im = Image.fromarray((Y[i, :, :, j] * v).astype(np.uint8))
                    im = im.convert("L")

                    if filenames is None:
                        c = "" if Y.shape[-1] == 1 else "_c" + str(j)
                        f = os.path.join(mask_dir, p_y + str(i).zfill(d) + c + extension)
                    else:
                        f = os.path.join(mask_dir, filenames[i] + extension)

                    im.save(f)


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
        lab_multi = measure.label(lab, neighbors=8, background=0)
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
    components = np.unique(lab_multi)

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


def save_npy_files(X, data_dir=None, filenames=None, verbose=True):
    """
    Save images in the given directory.

    Parameters
    ----------
    X : 4D/5D numpy array
        Data to save as images. The first dimension must be the number of images. E.g.
        ``(num_of_images, y, x, channels)`` or ``(num_of_images, z, y, x, channels)``.

    data_dir : str, optional
        Path to store X images.

    filenames : List, optional
        Filenames that should be used when saving each image.

    verbose : bool, optional
         To print saving information.
    """

    if verbose:
        s = X.shape if not isinstance(X, list) else X[0].shape
        print("Saving {} data as .npy in folder: {}".format(s, data_dir))

    os.makedirs(data_dir, exist_ok=True)
    if filenames is not None:
        if len(filenames) != len(X):
            raise ValueError(
                "Filenames array and length of X have different shapes: {} vs {}".format(len(filenames), len(X))
            )

    d = len(str(len(X)))
    for i in tqdm(range(len(X)), leave=False, disable=not is_main_process()):
        if filenames is None:
            f = os.path.join(data_dir, str(i).zfill(d) + ".npy")
        else:
            f = os.path.join(data_dir, os.path.splitext(filenames[i])[0] + ".npy")
        if isinstance(X, list):
            np.save(f, X[i][0])
        else:
            np.save(f, X[i])


def read_chunked_data(filename):
    if isinstance(filename, str):
        if not os.path.exists(filename):
            raise ValueError(f"File {filename} does not exist.")
        
        if any(filename.endswith(x) for x in [".h5", ".hdf5", ".hdf"]):
            fid = h5py.File(filename, "r")
            data = fid[list(fid)[0]]
        elif filename.endswith(".zarr"):
            fid = zarr.open(filename, "r")
            if len(list((fid.group_keys()))) != 0:  # if the zarr has groups
                fid = fid[list(fid.group_keys())[0]]
            if len(list((fid.array_keys()))) != 0:  # if the zarr has arrays
                data = fid[list(fid.array_keys())[0]]
            else:
                data = fid
        else:
            raise ValueError(f"File extension {filename} not recognized")

        return fid, data
    else:
        raise ValueError("'filename' is expected to be a str")

    
def write_chunked_data(data, data_dir, filename, dtype_str="float32", verbose=True):
    """
    Save images in the given directory.

    Parameters
    ----------
    data : 5D numpy array
        Data to save. E.g. ``(1, z, y, x, channels)``.

    data_dir : str
        Path to store X images.

    filename : str
        Filename of the data to use.

    dtype_str : str, optional
        Data type to use when saving.

    verbose : bool, optional
        To print saving information.
    """
    if data.ndim != 5:
        raise ValueError(f"Expected data needs to have 5 dimensions (in 'TZYXC' order). Given data shape: {data.shape}")

    # Change to TZCYX
    data = data.transpose((0, 1, 4, 2, 3))

    ext = os.path.splitext(filename)[1]
    if verbose:
        print("Saving {} data as {} in folder: {}".format(data.shape, ext, data_dir))

    os.makedirs(data_dir, exist_ok=True)

    if ext in [".hdf5", ".hdf", ".h5"]:
        fid = h5py.File(os.path.join(data_dir, filename), "w")
        data = fid.create_dataset("data", data=data, dtype=dtype_str, compression="gzip")
    # Zarr
    else:
        fid = zarr.open_group(os.path.join(data_dir, filename), mode="w")
        data = fid.create_dataset("data", data=data, dtype=dtype_str)


def read_chunked_nested_data(file, data_path=""):
    """
    Find recursively raw and ground truth data within a H5/Zarr file.
    """
    if any(file.endswith(x) for x in [".h5", ".hdf5", ".hdf"]):
        return read_chunked_nested_h5(file, data_path)
    elif file.endswith(".zarr"):
        return read_chunked_nested_zarr(file, data_path)

def read_chunked_nested_zarr(zarrfile, data_path=""):
    """
    Find recursively raw and ground truth data within a Zarr file.
    """
    if not zarrfile.endswith(".zarr"):
        raise ValueError("Not implemented for other filetypes than Zarr")
    fid = zarr.open(zarrfile, "r")

    def find_obj(path, fid):
        obj = None
        rpath = path.split(".")
        if len(rpath) == 0:
            return None
        else:
            if len(rpath) > 1:
                groups = list(fid.group_keys())
                if rpath[0] not in groups:
                    return None
                obj = find_obj(".".join(rpath[1:]), fid[rpath[0]])
            else:
                arrays = list(fid.array_keys())
                if rpath[0] not in arrays:
                    return None
                return fid[rpath[0]]
        return obj

    data = find_obj(data_path, fid)

    if data is None and data_path != "":
        raise ValueError(f"'{data_path}' not found in Zarr: {zarrfile}.")

    return fid, data

def read_chunked_nested_h5(h5file, data_path=""):
    """
    Find recursively raw and ground truth data within a Zarr file.
    """
    if not any(h5file.endswith(x) for x in [".h5", ".hdf5", ".hdf"]):
        raise ValueError("Not implemented for other filetypes than H5")
    
    fid = h5py.File(h5file, "r")

    def find_obj(path, fid):
        obj = None
        rpath = path.split(".")
        if len(rpath) == 0:
            return None
        else:
            if len(rpath) > 1:
                groups = list(fid.keys())
                if rpath[0] not in groups:
                    return None
                obj = find_obj(".".join(rpath[1:]), fid[rpath[0]])
            else:
                arrays = list(fid.keys())
                if rpath[0] not in arrays:
                    return None
                return fid[rpath[0]]
        return obj
    data = find_obj(data_path, fid)    
    if data is None and data_path != "":
        raise ValueError(f"'{data_path}' not found in H5: {h5file}.")
    return fid, data

def seg2aff_pni(img, dz=1, dy=1, dx=1, dtype="float32"):
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


def create_file_sha256sum(filename):
    h = sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(filename, "rb", buffering=0) as f:
        while n := f.readinto(mv):
            h.update(mv[:n])
    return h.hexdigest()
