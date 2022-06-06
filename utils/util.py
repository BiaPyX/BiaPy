import os
import math
import numpy as np
import time
import random
import matplotlib
import matplotlib.pyplot as plt
import scipy.ndimage
import tensorflow as tf
import copy
from PIL import ImageEnhance, Image
from tqdm import tqdm
from skimage.io import imsave, imread
from skimage import measure
from skimage.transform import resize
from skimage.segmentation import clear_border, find_boundaries
from collections import namedtuple

from engine.metrics import jaccard_index, jaccard_index_numpy, voc_calculation, DET_calculation
from utils.matching import _safe_divide, precision, recall, accuracy, f1

matplotlib.use('pdf')

def limit_threads(threads_number='1'):
    """Limits the number of threads for a python process.

       Parameters
       ----------
       threads_number : int, optional
           Number of threads.
    """

    print("Python process limited to {} thread".format(threads_number))

    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ["MKL_DYNAMIC"]="FALSE";
    os.environ["NUMEXPR_NUM_THREADS"]='1';
    os.environ["VECLIB_MAXIMUM_THREADS"]='1';
    os.environ["OMP_NUM_THREADS"] = '1';


def set_seed(seedValue=42, determinism=False):
    """Sets the seed on multiple python modules to obtain results as reproducible as possible.

       Parameters
       ----------
       seedValue : int, optional
           Seed value.

       determinism : bool, optional
           To force determism.
    """

    random.seed = seedValue
    np.random.seed(seed=seedValue)
    tf.random.set_seed(seedValue)
    os.environ["PYTHONHASHSEED"]=str(seedValue);
    if determinism:
        os.environ['TF_DETERMINISTIC_OPS'] = '1'


def create_plots(results, job_id, chartOutDir, metric='jaccard_index'):
    """Create loss and main metric plots with the given results.

       Parameters
       ----------
       results : Keras History object
           Record of training loss values and metrics values at successive epochs. History object is returned by Keras
           `fit() <https://keras.io/api/models/model_training_apis/#fit-method>`_ method.

       job_id : str
           Jod identifier.

       chartOutDir : str
           Path where the charts will be stored into.

       metric : str, optional
           Metric used.

       Examples
       --------
       +--------------------------------------------+--------------------------------------------+
       | .. figure:: ../img/chart_loss.png          | .. figure:: ../img/chart_jaccard_index.png |
       |   :width: 80%                              |   :width: 80%                              |
       |   :align: center                           |   :align: center                           |
       |                                            |                                            |
       |   Loss values on each epoch                |   Jaccard index values on each epoch       |
       +--------------------------------------------+--------------------------------------------+
    """

    print("Creating training plots . . .")

    os.makedirs(chartOutDir, exist_ok=True)

    # For matplotlib errors in display
    os.environ['QT_QPA_PLATFORM']='offscreen'

    # Loss
    plt.plot(results.history['loss'])
    plt.plot(results.history['val_loss'])
    plt.title('Model JOBID=' + job_id + ' loss')
    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.legend(['Train loss', 'Val. loss'], loc='upper left')
    plt.savefig(os.path.join(chartOutDir, job_id + '_loss.png'))
    plt.clf()

    # Jaccard index
    plt.plot(results.history[metric])
    plt.plot(results.history['val_' + metric])
    plt.title('Model JOBID=' + job_id + " " + metric)
    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.legend(['Train metric', 'Val. metric'], loc='upper left')
    plt.savefig(os.path.join(chartOutDir, job_id + '_' + metric +'.png'))
    plt.clf()


def threshold_plots(preds_test, Y_test, det_eval_ge_path, det_eval_path, det_bin, n_dig, job_id, job_file, char_dir,
                    r_val=0.5):
    """Create a plot with the different metric values binarizing the prediction with different thresholds, from ``0.1``
       to ``0.9``.

       Parameters
       ----------
       preds_test : 4D Numpy array
           Predictions made by the model. E.g. ``(num_of_images, x, y, channels)``.

       Y_test : 4D Numpy array
           Ground truth of the data. E.g. ``(num_of_images, x, y, channels)``.

       det_eval_ge_path : str
           Path where the ground truth is stored for the DET calculation.

       det_eval_path : str
           Path where the evaluation of the metric will be done.

       det_bin : str
           Path to the DET binary.

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

       t_voc : float
           Value of VOC when the threshold is ``r_val``.

       t_det : float
           Value of DET when the threshold is ``r_val``.

       Examples
       --------
       ::

           jac, voc, det = threshold_plots(
               preds_test, Y_test, det_eval_ge_path, det_eval_path, det_bin,
               n_dig, args.job_id, '278_3', char_dir)

       Will generate 3 charts, one per each metric: IoU, VOC and DET. In the x axis represents the 9 different
       thresholds applied, that is: ``0.1, 0.2, 0.3, ..., 0.9``. The y axis is the value of the metric in each chart. For
       instance, the Jaccard/IoU chart will look like this:

       .. image:: ../img/278_3_threshold_Jaccard.png
           :width: 60%
           :align: center

       In this example, the best value, ``0.868``, is obtained with a threshold of ``0.4``.
    """

    char_dir = os.path.join(char_dir, "t_" + job_file)

    t_jac = np.zeros(9)
    t_voc = np.zeros(9)
    t_det = np.zeros(9)
    objects = []
    r_val_pos = 0

    for i, t in enumerate(np.arange(0.1,1.0,0.1)):

        if t == r_val:
            r_val_pos = i

        objects.append(str('%.2f' % float(t)))

        # Threshold images
        bin_preds_test = (preds_test > t).astype(np.uint8)

        # Metrics (Jaccard + VOC + DET)
        print("Calculate metrics . . .")
        t_jac[i] = jaccard_index_numpy(Y_test, bin_preds_test)
        t_voc[i] = voc_calculation(Y_test, bin_preds_test, t_jac[i])
        t_det[i] = DET_calculation(Y_test, bin_preds_test, det_eval_ge_path, det_eval_path, det_bin, n_dig, job_id)

        print("t_jac[{}]: {}".format(i, t_jac[i]))
        print("t_voc[{}]: {}".format(i, t_voc[i]))
        print("t_det[{}]: {}".format(i, t_det[i]))

    # For matplotlib errors in display
    os.environ['QT_QPA_PLATFORM']='offscreen'

    os.makedirs(char_dir, exist_ok=True)

    # Plot Jaccard values
    plt.clf()
    plt.plot(objects, t_jac)
    plt.title('Model JOBID=' + job_file + ' Jaccard', y=1.08)
    plt.ylabel('Value')
    plt.xlabel('Threshold')
    for k, point in enumerate(zip(objects, t_jac)):
        plt.text(point[0], point[1], '%.3f' % float(t_jac[k]))
    plt.savefig(os.path.join(char_dir, job_file + '_threshold_Jaccard.png'))
    plt.clf()

    # Plot VOC values
    plt.plot(objects, t_voc)
    plt.title('Model JOBID=' + job_file + ' VOC', y=1.08)
    plt.ylabel('Value')
    plt.xlabel('Threshold')
    for k, point in enumerate(zip(objects, t_voc)):
        plt.text(point[0], point[1], '%.3f' % float(t_voc[k]))
    plt.savefig(os.path.join(char_dir, job_file + '_threshold_VOC.png'))
    plt.clf()

    # Plot DET values
    plt.plot(objects, t_det)
    plt.title('Model JOBID=' + job_file + ' DET', y=1.08)
    plt.ylabel('Value')
    plt.xlabel('Threshold')
    for k, point in enumerate(zip(objects, t_det)):
        plt.text(point[0], point[1], '%.3f' % float(t_det[k]))
    plt.savefig(os.path.join(char_dir, job_file + '_threshold_DET.png'))
    plt.clf()

    return  t_jac[r_val_pos], t_voc[r_val_pos], t_det[r_val_pos]


def save_tif(X, data_dir=None, filenames=None, verbose=True):
    """Save images in the given directory.

       Parameters
       ----------
       X : 4D/5D numpy array
           Data to save as images. The first dimension must be the number of images. E.g.
           ``(num_of_images, x, y, channels)`` or ``(num_of_images, z, x, y, channels)``.

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
            raise ValueError("Filenames array and length of X have different shapes: {} vs {}".format(len(filenames),len(X)))

    _dtype = X.dtype if X.dtype in [np.uint8, np.uint16, np.float32] else np.float32
    d = len(str(len(X)))
    for i in tqdm(range(X.shape[0]), leave=False):
        if filenames is None:
            f = os.path.join(data_dir, str(i).zfill(d)+'.tif')
        else:
            f = os.path.join(data_dir, os.path.splitext(filenames[i])[0]+'.tif')
        if X.ndim == 4:
            aux = np.expand_dims(np.expand_dims(X[i],0).transpose((0,3,1,2)), -1).astype(_dtype)
        else:
            aux = np.expand_dims(X[i].transpose((0,3,1,2)), -1).astype(_dtype)
        imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False)


def save_tif_pair_discard(X, Y, data_dir=None, suffix="", filenames=None, discard=True, verbose=True):
    """Save images in the given directory.

       Parameters
       ----------
       X : 4D/5D numpy array
           Data to save as images. The first dimension must be the number of images. E.g.
           ``(num_of_images, x, y, channels)`` or ``(num_of_images, z, x, y, channels)``.

       Y : 4D/5D numpy array
           Data mask to save. The first dimension must be the number of images. E.g.
           ``(num_of_images, x, y, channels)`` or ``(num_of_images, z, x, y, channels)``.

       data_dir : str, optional
           Path to store X images.

       suffix : str, optional
           Suffix to apply on output directory.

       filenames : List, optional
           Filenames that should be used when saving each image.

       discard : bool, optional
           Wheter to discard image/mask pairs if the mask has no label information.

       verbose : bool, optional
            To print saving information.
    """

    if verbose:
        s = X.shape if not isinstance(X, list) else X[0].shape
        print("Saving {} data as .tif in folder: {}".format(s, data_dir))

    os.makedirs(os.path.join(data_dir, 'x'+suffix), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'y'+suffix), exist_ok=True)
    if filenames is not None:
        if len(filenames) != len(X):
            raise ValueError("Filenames array and length of X have different shapes: {} vs {}".format(len(filenames),len(X)))

    d = len(str(len(X)))
    for i in tqdm(range(X.shape[0]), leave=False):
        if len(np.unique(Y[i])) >= 2 or not discard:
            if filenames is None:
                f1 = os.path.join(data_dir, 'x'+suffix, str(i).zfill(d)+'.tif')
                f2 = os.path.join(data_dir, 'y'+suffix, str(i).zfill(d)+'.tif')
            else:

                f1 = os.path.join(data_dir, 'x'+suffix, os.path.splitext(filenames[i])[0]+'.tif')
                f2 = os.path.join(data_dir, 'y'+suffix, os.path.splitext(filenames[i])[0]+'.tif')
            if X.ndim == 4:
                aux = np.expand_dims(np.expand_dims(X[i],0).transpose((0,3,1,2)), -1).astype(np.float32)
            else:
                aux = np.expand_dims(X[i].transpose((0,3,1,2)), -1).astype(np.float32)
            imsave(f1, aux, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False)
            if Y.ndim == 4:
                aux = np.expand_dims(np.expand_dims(Y[i],0).transpose((0,3,1,2)), -1).astype(np.float32)
            else:
                aux = np.expand_dims(Y[i].transpose((0,3,1,2)), -1).astype(np.float32)
            imsave(f2, aux, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False)


def save_img(X=None, data_dir=None, Y=None, mask_dir=None, scale_mask=True,
             prefix="", extension=".png", filenames=None):
    """Save images in the given directory.

       Parameters
       ----------
       X : 4D numpy array, optional
           Data to save as images. The first dimension must be the number of images. E.g. ``(num_of_images, x, y, channels)``.

       data_dir : str, optional
           Path to store X images.

       Y : 4D numpy array, optional
           Masks to save as images. The first dimension must be the number of images. E.g. ``(num_of_images, x, y, channels)``.

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
            d = len(str(X.shape[0]*X.shape[3]))
            for i in tqdm(range(X.shape[0])):
                for j in range(X.shape[3]):
                    if X.shape[-1] == 1:
                        im = Image.fromarray((X[i,:,:,j,0]*v).astype(np.uint8))
                        im = im.convert('L')
                    else:
                        im = Image.fromarray((X[i,:,:,j]*v).astype(np.uint8), 'RGB')

                    if filenames is None:
                        f = os.path.join(data_dir, p_x + str(i).zfill(d) + "_" + str(j).zfill(d) + extension)
                    else:
                        f = os.path.join(data_dir, filenames[(i*j)+j] + extension)
                    im.save(f)
        else:
            d = len(str(X.shape[0]))
            for i in tqdm(range(X.shape[0])):
                if X.shape[-1] == 1:
                    im = Image.fromarray((X[i,:,:,0]*v).astype(np.uint8))
                    im = im.convert('L')
                else:
                    im = Image.fromarray((X[i]*v).astype(np.uint8), 'RGB')

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
            d = len(str(Y.shape[0]*Y.shape[3]))
            for i in tqdm(range(Y.shape[0])):
                for j in range(Y.shape[3]):
                    for k in range(Y.shape[-1]):
                        im = Image.fromarray((Y[i,:,:,j,k]*v).astype(np.uint8))
                        im = im.convert('L')
                        if filenames is None:
                            c = "" if Y.shape[-1] == 1 else "_c"+str(j)
                            f = os.path.join(mask_dir, p_y + str(i).zfill(d) + "_" + str(j).zfill(d)+c+extension)
                        else:
                            f = os.path.join(data_dir, filenames[(i*j)+j] + extension)

                        im.save(f)
        else:
            d = len(str(Y.shape[0]))
            for i in tqdm(range(0, Y.shape[0])):
                for j in range(Y.shape[-1]):
                    im = Image.fromarray((Y[i,:,:,j]*v).astype(np.uint8))
                    im = im.convert('L')

                    if filenames is None:
                        c = "" if Y.shape[-1] == 1 else "_c"+str(j)
                        f = os.path.join(mask_dir, p_y+str(i).zfill(d)+c+extension)
                    else:
                        f = os.path.join(mask_dir, filenames[i] + extension)

                    im.save(f)


def make_weight_map(label, binary = True, w0 = 10, sigma = 5):
    """Generates a weight map in order to make the U-Net learn better the borders of cells and distinguish individual
       cells that are tightly packed. These weight maps follow the methodology of the original U-Net paper.

       Based on `unet/py_files/helpers.py <https://github.com/deepimagej/python4deepimagej/blob/499955a264e1b66c4ed2c014cb139289be0e98a4/unet/py_files/helpers.py>`_.

       Parameters
       ----------

       label : 3D numpy array
          Corresponds to a label image. E.g. ``(x, y, channels)``.

       binary : bool, optional
          Corresponds to whether or not the labels are binary.

       w0 : float, optional
          Controls for the importance of separating tightly associated entities.

       sigma : int, optional
          Represents the standard deviation of the Gaussian used for the weight map.

       Example
       -------

       Notice that weight has been defined where the objects are almost touching
       each other.

       .. image:: ../img/weight_map.png
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
        lab_multi = measure.label(lab, neighbors = 8, background = 0)
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

    n_comp = len(components)-1

    maps = np.zeros((n_comp, rows, cols))

    map_weight = np.zeros((rows, cols))

    if n_comp >= 2:
        for i in range(n_comp):

            # Only keeps current object.
            tmp = (lab_multi == components[i+1])

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

        map_weight = w0*np.exp(-((d1+d2)**2)/(2*(sigma**2)) ) * (lab==0).astype(int);

    map_weight += w_c

    return map_weight


def do_save_wm(labels, path, binary = True, w0 = 10, sigma = 5):
    """Retrieves the label images, applies the weight-map algorithm and save the weight maps in a folder. Uses
       internally :meth:`util.make_weight_map`.

       Based on `deepimagejunet/py_files/helpers.py <https://github.com/deepimagej/python4deepimagej/blob/499955a264e1b66c4ed2c014cb139289be0e98a4/unet/py_files/helpers.py>`_.

       Parameters
       ----------
       labels : 4D numpy array
           Corresponds to given label images. E.g. ``(num_of_images, x, y, channels)``.

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
    """Percentage of pixels that corresponds to the class in the given image.

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

    return c/(mask.shape[0]*mask.shape[1])


def divide_images_on_classes(data, data_mask, out_dir, num_classes=2, th=0.8):
    """Create a folder for each class where the images that have more pixels labeled as the class (in percentage) than
       the given threshold will be stored.

       Parameters
       ----------
       data : 4D numpy array
           Data to save as images. The first dimension must be the number of images. E. g.``(num_of_images, x, y, channels)``.

       data_mask : 4D numpy array
           Data mask to save as images.  The first dimension must be the number of images. E. g. ``(num_of_images, x, y, channels)``.

       out_dir : str
           Path to save the images.

       num_classes : int, optional
           Number of classes.

       th : float, optional
           Percentage of the pixels that must be labeled as a class to save it inside that class folder.
    """

    # Create the directories
    for i in range(num_classes):
        os.makedirs(os.path.join(out_dir, "x", "class"+str(i)), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "y", "class"+str(i)), exist_ok=True)

    print("Dividing provided data into {} classes . . .".format(num_classes))
    d = len(str(data.shape[0]))
    for i in tqdm(range(data.shape[0])):
        # Assign the image to a class if it has, in percentage, more pixels of
        # that class than the given threshold
        for j in range(num_classes):
            t = foreground_percentage(data_mask[i], j)
            if t > th:
                im = Image.fromarray(data[i,:,:,0])
                im = im.convert('L')
                im.save(os.path.join(os.path.join(out_dir, "x", "class"+str(j)), "im_" + str(i).zfill(d) + ".png"))
                im = Image.fromarray(data_mask[i,:,:,0]*255)
                im = im.convert('L')
                im.save(os.path.join(os.path.join(out_dir, "y", "class"+str(j)), "mask_" + str(i).zfill(d) + ".png"))


def save_filters_of_convlayer(model, out_dir, l_num=None, name=None, prefix="", img_per_row=8):
    """Create an image of the filters learned by a convolutional layer. One can identify the layer with ``l_num`` or
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

       .. image:: ../img/save_filters.png
           :width: 60%
           :align: center
    """

    if l_num is None and name is None:
        raise ValueError("One between 'l_num' or 'name' must be provided")

    # For matplotlib errors in display
    os.environ['QT_QPA_PLATFORM']='offscreen'

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

    rows = int(math.floor(filters.shape[3]/img_per_row))
    i = 0
    for r in range(rows):
        for c in range(img_per_row):
            ax = plt.subplot(rows, img_per_row, i+1)
            ax.set_xticks([])
            ax.set_yticks([])
            f = filters[:,:,0,i]
            plt.imshow(filters[:,:,0,i], cmap='gray')

            i += 1

    prefix += "_" if prefix != "" else prefix
    plt.savefig(os.path.join(out_dir, prefix + 'f_layer' + str(l_num) + '.png'))
    plt.clf()


def calculate_2D_volume_prob_map(Y, Y_path=None, w_foreground=0.94, w_background=0.06, save_dir=None):
    """Calculate the probability map of the given 2D data.

       Parameters
       ----------
       Y : 4D Numpy array
           Data to calculate the probability map from. E. g. ``(num_of_images, x, y, channel)``

       Y_path : str, optional
           Path to load the data from in case ``Y=None``.

       w_foreground : float, optional
           Weight of the foreground. This value plus ``w_background`` must be equal ``1``.

       w_background : float, optional
           Weight of the background. This value plus ``w_foreground`` must be equal ``1``.

       save_dir : str, optional
           Path to the file where the probability map will be stored.

       Raises
       ------
       ValueError
           if ``Y`` does not have 4 dimensions.

       ValueError
           if ``w_foreground + w_background > 1``.

       Returns
       -------
       Array : Str or 4D Numpy array
           Path where the probability map/s is/are stored if ``Y_path`` was given and there are images of different
           shapes. Otherwise, an array that represents the probability map of ``Y`` or all loaded data files from
           ``Y_path`` will be returned.
    """

    if Y is not None:
        if Y.ndim != 4:
            raise ValueError("'Y' must be a 4D Numpy array")

    if Y is None and Y_path is None:
        raise ValueError("'Y' or 'Y_path' need to be provided")

    if w_foreground + w_background > 1:
        raise ValueError("'w_foreground' plus 'w_background' can not be greater than one")

    if Y is not None:
        prob_map = np.copy(Y).astype(np.float32)
        l = prob_map.shape[0]
        channels = prob_map.shape[-1]
        v = np.max(prob_map)
    else:
        prob_map, _, _ = load_data_from_dir(Y_path)
        l = len(prob_map)
        channels = prob_map[0].shape[-1]
        v = np.max(prob_map[0])

    if isinstance(prob_map, list):
        first_shape = prob_map[0][0].shape
    else:
        first_shape = prob_map[0].shape

    print("Connstructing the probability map . . .")
    maps = []
    diff_shape = False
    for i in tqdm(range(l)):
        if isinstance(prob_map, list):
            _map = prob_map[i][0].copy().astype(np.float32)
        else:
            _map = prob_map[i].copy().astype(np.float32)

        for k in range(channels):
            # Remove artifacts connected to image border
            _map[:,:,k] = clear_border(_map[:,:,k])

            foreground_pixels = (_map[:,:,k] == v).sum()
            background_pixels = (_map[:,:,k] == 0).sum()

            if foreground_pixels == 0:
                _map[:,:,k][np.where(_map[:,:,k] == v)] = 0
            else:
                _map[:,:,k][np.where(_map[:,:,k] == v)] = w_foreground/foreground_pixels
            if background_pixels == 0:
                _map[:,:,k][np.where(_map[:,:,k] == 0)] = 0
            else:
                _map[:,:,k][np.where(_map[:,:,k] == 0)] = w_background/background_pixels

            # Necessary to get all probs sum 1
            s = _map[:,:,k].sum()
            if s == 0:
                t = 1
                for x in _map[:,:,k].shape: t *=x
                _map[:,:,k].fill(1/t)
            else:
                _map[:,:,k] = _map[:,:,k]/_map[:,:,k].sum()

        if first_shape != _map.shape: diff_shape = True
        maps.append(_map)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        if not diff_shape:
            for i in range(len(maps)):
                maps[i] = np.expand_dims(maps[i], 0)
            maps = np.concatenate(maps)
            print("Saving the probability map in {}".format(save_dir))
            np.save(os.path.join(save_dir, 'prob_map.npy'), maps)
            return maps
        else:
            print("As the files loaded have different shapes, the probability map for each one will be stored"
                  " separately in {}".format(save_dir))
            d = len(str(l))
            for i in range(l):
                f = os.path.join(save_dir, 'prob_map'+str(i).zfill(d)+'.npy')
                np.save(f, maps[i])
            return save_dir


def calculate_3D_volume_prob_map(Y, Y_path=None, w_foreground=0.94, w_background=0.06, save_dir=None):
    """Calculate the probability map of the given 3D data.

       Parameters
       ----------
       Y : 5D Numpy array
           Data to calculate the probability map from. E. g. ``(num_subvolumes, x, y, z, channel)``

       Y_path : str, optional
           Path to load the data from in case ``Y=None``.

       w_foreground : float, optional
           Weight of the foreground. This value plus ``w_background`` must be equal ``1``.

       w_background : float, optional
           Weight of the background. This value plus ``w_foreground`` must be equal ``1``.

       save_dir : str, optional
           Path to the directory where the probability map will be stored.

       Returns
       -------
       Array : Str or 5D Numpy array
           Path where the probability map/s is/are stored if ``Y_path`` was given and there are images of different
           shapes. Otherwise, an array that represents the probability map of ``Y`` or all loaded data files from
           ``Y_path`` will be returned.

       Raises
       ------
       ValueError
           if ``Y`` does not have 5 dimensions.
       ValueError
           if ``w_foreground + w_background > 1``.
    """

    if Y is not None:
        if Y.ndim != 5:
            raise ValueError("'Y' must be a 5D Numpy array")

    if Y is None and Y_path is None:
        raise ValueError("'Y' or 'Y_path' need to be provided")

    if w_foreground + w_background > 1:
        raise ValueError("'w_foreground' plus 'w_background' can not be greater " "than one")

    if Y is not None:
        prob_map = np.copy(Y).astype(np.float32)
        l = prob_map.shape[0]
        channels = prob_map.shape[-1]
        v = np.max(prob_map)
    else:
        prob_map, _, _ = load_3d_images_from_dir(Y_path)
        l = len(prob_map)
        channels = prob_map[0].shape[-1]
        v = np.max(prob_map[0])

    if isinstance(prob_map, list):
        first_shape = prob_map[0][0].shape
    else:
        first_shape = prob_map[0].shape

    print("Constructing the probability map . . .")
    maps = []
    diff_shape = False
    for i in range(l):
        if isinstance(prob_map, list):
            _map = prob_map[i][0].copy().astype(np.float64)
        else:
            _map = prob_map[i].copy().astype(np.float64)

        for k in range(channels):
            for j in range(_map.shape[2]):
                # Remove artifacts connected to image border
                _map[:,:,j,k] = clear_border(_map[:,:,j,k])
            foreground_pixels = (_map[:,:,:,k] == v).sum()
            background_pixels = (_map[:,:,:,k] == 0).sum()

            if foreground_pixels == 0:
                _map[:,:,:,k][np.where(_map[:,:,:,k] == v)] = 0
            else:
                _map[:,:,:,k][np.where(_map[:,:,:,k] == v)] = w_foreground/foreground_pixels
            if background_pixels == 0:
                _map[:,:,:,k][np.where(_map[:,:,:,k] == 0)] = 0
            else:
                _map[:,:,:,k][np.where(_map[:,:,:,k] == 0)] = w_background/background_pixels

            # Necessary to get all probs sum 1
            s = _map[:,:,:,k].sum()
            if s == 0:
                t = 1
                for x in _map[:,:,:,k].shape: t *=x
                _map[:,:,:,k].fill(1/t)
            else:
                _map[:,:,:,k] = _map[:,:,:,k]/_map[:,:,:,k].sum()

        if first_shape != _map.shape: diff_shape = True
        maps.append(_map)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        if not diff_shape:
            for i in range(len(maps)):
                maps[i] = np.expand_dims(maps[i], 0)
            maps = np.concatenate(maps)
            print("Saving the probability map in {}".format(save_dir))
            np.save(os.path.join(save_dir, 'prob_map.npy'), maps)
            return maps
        else:
            print("As the files loaded have different shapes, the probability map for each one will be stored "
                  "separately in {}".format(save_dir))
            d = len(str(l))
            for i in range(l):
                f = os.path.join(save_dir, 'prob_map'+str(i).zfill(d)+'.npy')
                np.save(f, maps[i])
            return save_dir


def check_masks(path, n_classes=2):
    """Check wheter the data masks have the correct labels inspection a few random images of the given path. If the
       function gives no error one should assume that the masks are correct.

       Parameters
       ----------
       path : str
           Path to the data mask.

       n_classes : int, optional
           Maximum classes that the masks must contain.
    """

    print("Checking wheter the images in {} are binary . . .".format(path))

    ids = sorted(next(os.walk(path))[2])

    # Check only 4 random images or less if there are not as many
    num_sample = [4, len(ids)]
    numbers = random.sample(range(0, len(ids)), min(num_sample))
    for i in numbers:
        img = imread(os.path.join(path, ids[i]))
        values, _ = np.unique(img, return_counts=True)
        if len(values) > n_classes :
            raise ValueError("Error: given masks are not binary. Please correct the images before training. "
                             "(image: {})\nValues: {}".format(os.path.join(path, ids[i]), values))


def img_to_onehot_encoding(img, num_classes=2):
    """Converts image given into one-hot encode format.

       The opposite function is :func:`~onehot_encoding_to_img`.

       Parameters
       ----------
       img : Numpy 3D/4D array
           Image. E.g. ``(x, y, channels)`` or ``(x, y, z, channels)``.

       num_classes : int, optional
           Number of classes to distinguish.

       Returns
       -------
       one_hot_labels : Numpy 3D/4D array
           Data one-hot encoded. E.g. ``(x, y, num_classes)`` or ``(x, y, z, num_classes)``.
    """

    if img.ndim == 4:
        shape = img.shape[:3]+(num_classes,)
    else:
        shape = img.shape[:2]+(num_classes,)

    encoded_image = np.zeros(shape, dtype=np.int8)

    for i in range(num_classes):
        if img.ndim == 4:
            encoded_image[:,:,:,i] = np.all(img.reshape((-1,1)) == i, axis=1).reshape(shape[:3])
        else:
            encoded_image[:,:,i] = np.all(img.reshape((-1,1)) == i, axis=1).reshape(shape[:2])

    return encoded_image


def onehot_encoding_to_img(encoded_image):
    """Converts one-hot encode image into an image with jus tone channel and all the classes represented by an integer.

       The opposite function is :func:`~img_to_onehot_encoding`.

       Parameters
       ----------
       encoded_image : Numpy 3D/4D array
           Image. E.g. ``(x, y, channels)`` or ``(x, y, z, channels)``.

       Returns
       -------
       img : Numpy 3D/4D array
           Data one-hot encoded. E.g. ``(x, y, z, num_classes)``.
    """

    if encoded_image.ndim == 4:
        shape = encoded_image.shape[:3]+(1,)
    else:
        shape = encoded_image.shape[:2]+(1,)

    img = np.zeros(shape, dtype=np.int8)
    for i in range(img.shape[-1]):
        img[encoded_image[...,i] == 1] = i

    return img


def load_data_from_dir(data_dir, crop=False, crop_shape=None, overlap=(0,0), padding=(0,0), return_filenames=False,
                       reflect_to_complete_shape=False):
    """Load data from a directory. If ``crop=False`` all the data is suposed to have the same shape.

       Parameters
       ----------
       data_dir : str
           Path to read the data from.

       crop : bool, optional
           Crop each image into desired shape pointed by ``crop_shape``.

       crop_shape : Tuple of 3 ints, optional
           Shape of the crop to be made. E.g. ``(x, y, channels)``.

       overlap : Tuple of 2 floats, optional
           Amount of minimum overlap on x and y dimensions. The values must  be on range ``[0, 1)``, that is, ``0%`` or
           ``99%`` of overlap. E. g. ``(x, y)``.

       padding : Tuple of 2 ints, optional
           Size of padding to be added on each axis ``(x, y)``. E.g. ``(24, 24)``.

       return_filenames : bool, optional
           Return a list with the loaded filenames. Useful when you need to save them afterwards with the same names as
           the original ones.

       reflect_to_complete_shape : bool, optional
           Wheter to increase the shape of the dimension that have less size than selected patch size padding it with
           'reflect'.

       Returns
       -------
       data : 4D Numpy array or list of 3D Numpy arrays
           Data loaded. E.g. ``(num_of_images, y, x, channels)`` if all files have same shape, otherwise a list of
           ``(y, x, channels)`` arrays will be returned.

       data_shape : List of tuples
           Shapes of all 3D images readed. Useful to reconstruct the original images together with ``crop_shape``.

       crop_shape : List of tuples
           Shape of the loaded 3D images after cropping. Useful to reconstruct the original images together with
           ``data_shape``.

       filenames : List of str, optional
           Loaded filenames.

       Examples
       --------
       ::

           # EXAMPLE 1
           # Case where we need to load 165 images of shape (1024, 768)
           data_path = "data/train/x"

           load_data_from_dir(data_path)
           # The function will print the shape of the created array. In this example:
           #     *** Loaded data shape is (165, 768, 1024, 1)
           # Notice height and width swap because of Numpy ndarray terminology


           # EXAMPLE 2
           # Case where we need to load 165 images of shape (1024, 768) but
           # cropping them into (256, 256, 1) patches
           data_path = "data/train/x"
           crop_shape = (256, 256, 1)

           load_data_from_dir(data_path, crop=True, crop_shape=crop_shape)
           # The function will print the shape of the created array. In this example:
           #     *** Loaded data shape is (1980, 256, 256, 1)
    """

    if crop:
        from data.data_2D_manipulation import crop_data_with_overlap

    print("Loading data from {}".format(data_dir))
    ids = sorted(next(os.walk(data_dir))[2])
    data = []
    data_shape = []
    c_shape = []
    if return_filenames: filenames = []

    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        if id_.endswith('.npy'):
            img = np.load(os.path.join(data_dir, id_))
        else:
            img = imread(os.path.join(data_dir, id_))
        #img = np.asarray(Image.open(os.path.join(data_dir, id_)))

        if return_filenames: filenames.append(id_)

        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        else:
           if img.shape[0] <= 3: img = img.transpose((1,2,0))

        if reflect_to_complete_shape: img = pad_and_reflect(img, crop_shape, verbose=False)

        # Ensure uint8
        if img.dtype == np.uint16:
            if np.max(img) > 255:
                img = normalize(img, 0, 65535)
            else:
                img = img.astype(np.uint8)

        data_shape.append(img.shape)
        img = np.expand_dims(img, axis=0)
        if crop and img[0].shape != crop_shape[:2]+(img.shape[-1],):
            img = crop_data_with_overlap(img, crop_shape[:2]+(img.shape[-1],), overlap=overlap, padding=padding,
                                         verbose=False)
        c_shape.append(img.shape)
        data.append(img)

    same_shape = True
    s = data[0].shape
    for i in range(1,len(data)):
        if s != data[i].shape:
            same_shape = False
            break

    if crop or same_shape:
        data = np.concatenate(data)
        print("*** Loaded data shape is {}".format(data.shape))
    else:
        print("*** Loaded data[0] shape is {}".format(data[0].shape))

    if return_filenames:
        return data, data_shape, c_shape, filenames
    else:
        return data, data_shape, c_shape


def load_ct_data_from_dir(data_dir, shape=None):
    """Load CT data from a directory.

       Parameters
       ----------
       data_dir : str
           Path to read the data from.

       shape : 3D int tuple, optional
           Shape of the data to load. If is not provided the shape is calculated automatically looping over all data
           files and it will be  the maximum value found per axis. So, given the value the process should be faster.
           E.g. ``(x, y, channels)``.

       Returns
       -------
       data : 4D Numpy array
           Data loaded. E.g. ``(num_of_images, y, x, channels)``.

       Examples
       --------
       ::

           # EXAMPLE 1
           # Case where we need to load 165 images of shape (1024, 768)
           data_path = "data/train/x"
           data_shape = (1024, 768, 1)

           load_data_from_dir(data_path, data_shape)

           # The function will print list's first position array's shape. In this example:
           #     *** Loaded data[0] shape is (165, 768, 1024, 1)
           # Notice height and width swap because of Numpy ndarray terminology
    """
    import nibabel as nib

    print("Loading data from {}".format(data_dir))
    ids = sorted(next(os.walk(data_dir))[2])

    if shape is None:
        # Determine max in each dimension first
        max_x = 0
        max_y = 0
        max_z = 0
        for n, id_ in tqdm(enumerate(ids), total=len(ids)):
            img = nib.load(os.path.join(data_dir, id_))
            img = np.array(img.dataobj)

            max_x = img.shape[0] if max_x < img.shape[0] else max_x
            max_y = img.shape[1] if max_y < img.shape[1] else max_y
            max_z = img.shape[2] if max_z < img.shape[2] else max_z
        _shape = (max_x, max_y, max_z)
    else:
        _shape = shape

    # Create the array
    data = np.zeros((len(ids), ) + _shape, dtype=np.float32)
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        img = nib.load(os.path.join(data_dir, id_))
        img = np.array(img.dataobj)
        data[n,0:img.shape[0],0:img.shape[1],0:img.shape[2]] = img

    print("*** Loaded data shape is {}".format(data.shape))
    return data


def load_3d_images_from_dir(data_dir, crop=False, crop_shape=None, verbose=False, overlap=(0,0,0), padding=(0,0,0),
                            median_padding=False, reflect_to_complete_shape=False, return_filenames=False):
    """Load data from a directory.

       Parameters
       ----------
       data_dir : str
           Path to read the data from.

       crop : bool, optional
           Crop each 3D image when readed.

       crop_shape : Tuple of 4 ints, optional
           Shape of the subvolumes to create when cropping.  E.g. ``(x, y, z, channels)``.

       verbose : bool, optional
           Wheter to enable verbosity.

       overlap : Tuple of 3 floats, optional
           Amount of minimum overlap on x, y and z dimensions. The values must be on range ``[0, 1)``, that is, ``0%``
           or ``99%`` of overlap. E.g. ``(x, y, z)``.

       padding : Tuple of 3 ints, optional
           Size of padding to be added on each axis ``(x, y, z)``. E.g. ``(24, 24, 24)``.

       median_padding : bool, optional
           If ``True`` the padding value is the median value. If ``False``, the added values are zeroes.

       reflect_to_complete_shape : bool, optional
           Wheter to increase the shape of the dimension that have less size than selected patch size padding it with
           'reflect'.

       return_filenames : bool, optional
           Return a list with the loaded filenames. Useful when you need to save them afterwards with the same names as
           the original ones.

       Returns
       -------
       data : 5D Numpy array or list of 4D Numpy arrays
           Data loaded. E.g. ``(num_of_images, x, y, z, channels)`` if all files have same shape, otherwise a list of
           ``(1, x, y, z, channels)`` arrays will be returned.

       data_shape : List of tuples
           Shapes of all 3D images readed. Useful to reconstruct the original images together with ``crop_shape``.

       crop_shape : List of tuples
           Shape of the loaded 3D images after cropping. Useful to reconstruct the original images together with
           ``data_shape``.

       filenames : List of str, optional
           Loaded filenames.

       Examples
       --------
       ::

           # EXAMPLE 1
           # Case where we need to load 20 images of shape (1024, 1024, 91, 1)
           data_path = "data/train/x"

           data = load_data_from_dir(data_path)
           # The function will print list's first position array's shape. In this example:
           #     *** Loaded data[0] shape is (20, 91, 1024, 1024, 1)
           # Notice height, width and depth swap as skimage.io imread function
           # is used to load images

           # EXAMPLE 2
           # Same as example 1 but with unknown shape, cropping them into (256, 256, 40, 1) subvolumes with minimum
           # overlap and storing filenames.
           data_path = "data/train/x"

           X_test, orig_test_img_shapes, \
           crop_test_img_shapes, te_filenames = load_3d_images_from_dir(
               test_path, crop=True, crop_shape=(256, 256, 40, 1), overlap=(0,0,0), return_filenames=True)

           # The function will print the shape of the created array which its size is the concatenation in 0 axis of all
           # subvolumes created for each 3D image in the given path. For example:
           #     *** Loaded data shape is (350, 256, 256, 40, 1)
           # Notice height, width and depth swap as skimage.io imread function is used to load images.
    """
    if crop and crop_shape is None:
        raise ValueError("'crop_shape' must be provided when 'crop' is True")

    print("Loading data from {}".format(data_dir))
    ids = sorted(next(os.walk(data_dir))[2])

    if crop:
        from data.data_3D_manipulation import crop_3D_data_with_overlap

    data = []
    data_shape = []
    c_shape = []
    if return_filenames: filenames = []

    # Read images
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        if id_.endswith('.npy'):
            img = np.load(os.path.join(data_dir, id_))
        else:
            img = imread(os.path.join(data_dir, id_))
        img = np.squeeze(img)

        # Ensure uint8
        if img.dtype == np.uint16:
            if np.max(img) > 255:
                img = normalize(img, 0, 65535)
            else:
                img = img.astype(np.uint8)

        if return_filenames: filenames.append(id_)
        if len(img.shape) == 3: img = np.expand_dims(img, axis=-1)
        if reflect_to_complete_shape: img = pad_and_reflect(img, crop_shape, verbose=verbose)

        data_shape.append(img.shape)
        if crop and img.shape != crop_shape[:3]+(img.shape[-1],):
            img = crop_3D_data_with_overlap(img, crop_shape[:3]+(img.shape[-1],), overlap=overlap, padding=padding,
                                            median_padding=median_padding, verbose=verbose)
        else:
            img = np.transpose(img, (1,2,0,3))
            img = np.expand_dims(img, axis=0)

        c_shape.append(img.shape)
        data.append(img)

    same_shape = True
    s = data[0].shape
    for i in range(1,len(data)):
        if s != data[i].shape:
            same_shape = False
            break

    if crop or same_shape:
        data = np.concatenate(data)
        print("*** Loaded data shape is {}".format(data.shape))
    else:
        print("*** Loaded data[0] shape is {}".format(data[0].shape))

    if return_filenames:
        return data, data_shape, c_shape, filenames
    else:
        return data, data_shape, c_shape


def labels_into_bcd(data_mask, mode="BCD", fb_mode="outer", save_dir=None):
    """Create an array with 3 channels given semantic or instance segmentation data masks. These 3 channels are:
       semantic mask, contours and distance map.

       Parameters
       ----------
       data_mask : 5D Numpy array
           Data mask to create the new array from. It is expected to have just one channel. E.g. ``(10, 1000, 1000, 200, 1)``

       mode : str, optional
           Operation mode. Possible values: ``BC`` and ``BCD``.  ``BC`` corresponds to use binary segmentation+contour.
           ``BCD`` stands for binary segmentation+contour+distances.

       fb_mode : str, optional
          Mode of the find_boundaries function from ``scikit-image``. More info in:
          `find_boundaries() <https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.find_boundaries>`_.

       save_dir : str, optional
           Path to store samples of the created array just to debug it is correct.

       Returns
       -------
       new_mask : 5D Numpy array
           5D array with 3 channels instead of one. E.g. ``(10, 1000, 1000, 200, 3)``
    """

    assert mode in ['BC', 'BCM', 'BCD', 'BCDv2', 'Dv2']
    assert data_mask.ndim in [5, 4]

    d_shape = 4 if data_mask.ndim == 5 else 3
    if mode in ['BCDv2', 'Dv2']:
        c_number = 4
    elif mode in ['BCD', 'BCM']:
        c_number = 3
    elif mode == 'BC':
        c_number = 2


    new_mask = np.zeros(data_mask.shape[:d_shape] + (c_number,), dtype=np.float32)

    for img in tqdm(range(data_mask.shape[0])):
        vol = data_mask[img,...,0].astype(np.int64)
        l = np.unique(vol)

        # If only have background -> skip
        if len(l) != 1:
            vol_dist = np.zeros(vol.shape)

            if mode in ["BCD", "BCDv2", "Dv2"]:
                # For each nucleus
                for i in tqdm(range(1,len(l)), leave=False):
                    obj = l[i]
                    distance = scipy.ndimage.distance_transform_edt(vol==obj)
                    vol_dist += distance

                # Foreground distance
                new_mask[img,...,2] = vol_dist.copy()

                # Background distance
                if mode in ["BCDv2", "Dv2"]:
                    # Background distance
                    vol_b_dist = np.invert(vol>0)
                    vol_b_dist= scipy.ndimage.distance_transform_edt(vol_b_dist)
                    vol_b_dist = np.max(vol_b_dist)-vol_b_dist
                    new_mask[img,...,3] = vol_b_dist.copy()

            # Semantic mask
            if mode != "Dv2":
                new_mask[img,...,0] = (vol>0).copy().astype(np.uint8)

            # Contour
            if mode in ["BCD", "BCDv2", "BC", "BCM", "Dv2"]:
                new_mask[img,...,1] = find_boundaries(vol, mode=fb_mode).astype(np.uint8)
                # Remove contours from segmentation maps
                new_mask[img,...,0][np.where(new_mask[img,...,1] == 1)] = 0
                if mode == "BCM":
                    new_mask[img,...,2] = (vol>0).astype(np.uint8)

    # Normalize and merge distance channels
    if mode in ["BCDv2", "Dv2"]:
        f_min = np.min(new_mask[...,2])
        f_max = np.max(new_mask[...,2])
        b_min = np.min(new_mask[...,3])
        b_max = np.max(new_mask[...,3])

        # Normalize foreground and background separately
        new_mask[...,2] = (new_mask[...,2]-f_min)/(f_max-f_min)
        new_mask[...,3] = (new_mask[...,3]-b_min)/(b_max-b_min)

        new_mask[...,2] = new_mask[...,3] - new_mask[...,2]
        # The intersection of the channels is the contour channel, so set it to the maximum value 1
        new_mask[...,2][new_mask[...,1]>0] = 1
        new_mask = new_mask[...,:3]
        if mode == "Dv2":
            new_mask = np.expand_dims(new_mask[...,-1], -1)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        suffix = []
        if mode == "Dv2":
            suffix.append('_distance.tif')
        else:
            suffix.append('_semantic.tif')
        if mode in ["BC", "BCM", "BCD", "BCDv2"]:
            suffix.append('_contour.tif')
            if mode in ["BCD", "BCDv2"]:
                suffix.append('_distance.tif')
            elif mode == "BCM":
                suffix.append('_binary_mask.tif')

        for i in range(min(3,len(new_mask))):
            for j in range(len(suffix)):
                aux = np.transpose(new_mask[i,...,j],(2,0,1)) if data_mask.ndim == 5 else new_mask[i,...,j]
                aux = np.expand_dims(np.expand_dims(aux,-1),0)
                save_tif(aux, save_dir, filenames=['vol'+str(i)+suffix[j]], verbose=False)

    return new_mask


def check_downsample_division(X, d_levels):
    """Ensures ``X`` shape is divisible by ``2`` times ``d_levels`` adding padding if necessary.

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

    d_val = pow(2,d_levels)
    dy = math.ceil(X.shape[1]/d_val)
    dx = math.ceil(X.shape[2]/d_val)
    o_shape = X.shape
    if dy*d_val != X.shape[1] or dx*d_val != X.shape[2]:
        X = np.pad(X, ((0,0), (0,(dy*d_val)-X.shape[1]), (0,(dx*d_val)-X.shape[2]), (0,0)))
        print("Data has been padded to be downsampled {} times. Its shape now is: {}".format(d_levels, X.shape))
    return X, o_shape


def save_npy_files(X, data_dir=None, filenames=None, verbose=True):
    """Save images in the given directory.

       Parameters
       ----------
       X : 4D/5D numpy array
           Data to save as images. The first dimension must be the number of images. E.g.
           ``(num_of_images, x, y, channels)`` or ``(num_of_images, z, x, y, channels)``.

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
            raise ValueError("Filenames array and length of X have different shapes: {} vs {}".format(len(filenames),len(X)))

    d = len(str(len(X)))
    for i in tqdm(range(len(X)), leave=False):
        if filenames is None:
            f = os.path.join(data_dir, str(i).zfill(d)+'.npy')
        else:
            f = os.path.join(data_dir, os.path.splitext(filenames[i])[0]+'.npy')
        if isinstance(X, list):
            np.save(f, X[i][0])
        else:
            np.save(f, X[i])


def apply_binary_mask(X, bin_mask_dir):
    """Apply a binary mask to remove values outside it.

       Parameters
       ----------
       X : 4D Numpy array
           Data to apply the mask. E.g. ``(vol_number, x, y, channels)``

       bin_mask_dir : str, optional
           Directory where the binary mask are located.

       Returns
       -------
       X : 4D Numpy array
           Data with the mask applied. E.g. ``(vol_number, x, y, channels)``.
    """

    if X.ndim != 4:
        raise ValueError("'X' needs to have 4 dimensions and not {}".format(X.ndim))

    print("Applying binary mask(s) from {}".format(bin_mask_dir))

    ids = sorted(next(os.walk(bin_mask_dir))[2])

    if len(ids) == 1:
        one_file = True
        print("It is assumed that the mask found {} is valid for all 'X' data".format(os.path.join(bin_mask_dir, ids[0])))
    else:
        one_file = False

    if one_file:
        mask = imread(os.path.join(bin_mask_dir, ids[0]))
        mask = np.squeeze(mask)
        if mask.ndim != 2 and mask.ndim != 3:
            raise ValueError("Mask needs to have 2 or 3 dimensions and not {}".format(mask.ndim))

        for k in tqdm(range(X.shape[0])):
            if mask.ndim == 2:
                for c in range(X.shape[-1]):
                    X[k,:,:,c] = X[k,:,:,c]*(mask>0)
            else:
                X[k] = X[k]*(mask>0)
    else:
        for i in tqdm(range(len(ids))):
            mask = imread(os.path.join(bin_mask_dir, ids[i]))
            mask = np.squeeze(mask)
            if mask.ndim == 2:
                for c in range(X.shape[-1]):
                    X[i,:,:,c] = X[i,:,:,c]*(mask>0)
            else:
                X[i] = X[i]*(mask>0)

    return X


def pad_and_reflect(img, crop_shape, verbose=False):
    """Load data from a directory.

       Parameters
       ----------
       img : 3D/4D Numpy array
           Image to pad. E.g. ``(x, y, channels)`` or ``(x, y, z, channels)``.

       crop_shape : Tuple of 3/4 ints, optional
           Shape of the subvolumes to create when cropping.  E.g. ``(x, y, channels)`` or ``(x, y, z, channels)``.

       verbose : bool, optional
           Wheter to output information.

       Returns
       -------
       img : 3D/4D Numpy array
           Image padded (if needed). E.g. ``(x, y, channels)`` or ``(x, y, z, channels)``.
    """
    if img.ndim == 4 and len(crop_shape) != 4:
        raise ValueError("'crop_shape' needs to have 4 values as the input array has 4 dims")
    if img.ndim == 3 and len(crop_shape) != 3:
        raise ValueError("'crop_shape' needs to have 3 values as the input array has 3 dims")

    if img.ndim == 4:
        if img.shape[0] < crop_shape[2]:
            diff = crop_shape[2]-img.shape[0]
            o_shape = img.shape
            img = np.pad(img, ((diff,0),(0,0),(0,0),(0,0)), 'reflect')
            if verbose: print("Reflected from {} to {}".format(o_shape, img.shape))

        if img.shape[1] < crop_shape[0]:
            diff = crop_shape[0]-img.shape[1]
            o_shape = img.shape
            img = np.pad(img, ((0,0),(diff,0),(0,0),(0,0)), 'reflect')
            if verbose: print("Reflected from {} to {}".format(o_shape, img.shape))

        if img.shape[2] < crop_shape[1]:
            diff = crop_shape[1]-img.shape[2]
            o_shape = img.shape
            img = np.pad(img, ((0,0),(0,0),(diff,0),(0,0)), 'reflect')
            if verbose: print("Reflected from {} to {}".format(o_shape, img.shape))
    else:
        if img.shape[0] < crop_shape[1]:
            diff = crop_shape[1]-img.shape[0]
            o_shape = img.shape
            img = np.pad(img, ((diff,0),(0,0),(0,0)), 'reflect')
            if verbose: print("Reflected from {} to {}".format(o_shape, img.shape))

        if img.shape[1] < crop_shape[0]:
            diff = crop_shape[0]-img.shape[1]
            o_shape = img.shape
            img = np.pad(img, ((0,0),(diff,0),(0,0)), 'reflect')
            if verbose: print("Reflected from {} to {}".format(o_shape, img.shape))
    return img


def wrapper_matching_dataset_lazy(stats_all, thresh, criterion='iou', by_image=False):

    expected_keys = set(('fp', 'tp', 'fn', 'precision', 'recall', 'accuracy', 'f1', 'criterion', 'thresh', 'n_true', 'n_pred', 'mean_true_score', 'mean_matched_score', 'panoptic_quality'))

    # accumulate results over all images for each threshold separately
    n_images, n_threshs = len(stats_all), len(thresh)

    single_thresh = True if n_threshs == 1 else False
    accumulate = [{} for _ in range(n_threshs)]
    for stats in stats_all:
        for i,s in enumerate(stats):
            acc = accumulate[i]
            for k,v in s._asdict().items():
                if k == 'mean_true_score' and not bool(by_image):
                    # convert mean_true_score to "sum_matched_score"
                    acc[k] = acc.setdefault(k,0) + v * s.n_true
                else:
                    try:
                        acc[k] = acc.setdefault(k,0) + v
                    except TypeError:
                        pass

    # normalize/compute 'precision', 'recall', 'accuracy', 'f1'
    for thr,acc in zip(thresh,accumulate):
        set(acc.keys()) == expected_keys or _raise(ValueError("unexpected keys"))
        acc['criterion'] = criterion
        acc['thresh'] = thr
        acc['by_image'] = bool(by_image)
        if bool(by_image):
            for k in ('precision', 'recall', 'accuracy', 'f1', 'mean_true_score', 'mean_matched_score', 'panoptic_quality'):
                acc[k] /= n_images
        else:
            tp, fp, fn, n_true = acc['tp'], acc['fp'], acc['fn'], acc['n_true']
            sum_matched_score = acc['mean_true_score']

            mean_matched_score = _safe_divide(sum_matched_score, tp)
            mean_true_score    = _safe_divide(sum_matched_score, n_true)
            panoptic_quality   = _safe_divide(sum_matched_score, tp+fp/2+fn/2)

            acc.update(
                precision          = precision(tp,fp,fn),
                recall             = recall(tp,fp,fn),
                accuracy           = accuracy(tp,fp,fn),
                f1                 = f1(tp,fp,fn),
                mean_true_score    = mean_true_score,
                mean_matched_score = mean_matched_score,
                panoptic_quality   = panoptic_quality,
            )

    accumulate = tuple(namedtuple('DatasetMatching',acc.keys())(*acc.values()) for acc in accumulate)
    return accumulate[0] if single_thresh else accumulate

def wrapper_matching_segCompare(stats_all):
    expected_keys = ['number_of_cells', 'correct_segmentations', 'oversegmentation_rate', 'undersegmentation_rate', 'missing_rate']

    accumulated_values = dict.fromkeys(expected_keys, 0)

    for key in expected_keys:
        for stat in stats_all:
            accumulated_values[key] = accumulated_values[key] + stat[key]
        accumulated_values[key] = accumulated_values[key]/len(stats_all)
    return accumulated_values

def normalize(x, x_min, x_max, out_min=0, out_max=255, out_type=np.uint8):
    return ((np.array((x-x_min)/(x_max-x_min))*(out_max-out_min))+out_min).astype(out_type)

def check_value(value, range=(0,1)):
    """Checks if a value is within a range """
    if range[0] <= value <= range[1]:
        return True
    return False
