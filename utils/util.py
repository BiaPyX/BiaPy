import os
import math
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import scipy.ndimage
import tensorflow as tf
import copy
from PIL import Image
from tqdm import tqdm
from skimage.io import imsave, imread
from skimage import measure
from collections import namedtuple

from engine.metrics import jaccard_index_numpy, voc_calculation, DET_calculation
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

       metric : List of str, optional
           Metrics used.

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

    # Metric
    for i in range(len(metric)):
        plt.plot(results.history[metric[i]])
        plt.plot(results.history['val_' + metric[i]])
        plt.title('Model JOBID=' + job_id + " " + metric[i])
        plt.ylabel('Value')
        plt.xlabel('Epoch')
        plt.legend(['Train metric', 'Val. metric'], loc='upper left')
        plt.savefig(os.path.join(chartOutDir, job_id + '_' + metric[i] +'.png'))
        plt.clf()


def threshold_plots(preds_test, Y_test, det_eval_ge_path, det_eval_path, det_bin, n_dig, job_id, job_file, char_dir,
                    r_val=0.5):
    """Create a plot with the different metric values binarizing the prediction with different thresholds, from ``0.1``
       to ``0.9``.

       Parameters
       ----------
       preds_test : 4D Numpy array
           Predictions made by the model. E.g. ``(num_of_images, y, x, channels)``.

       Y_test : 4D Numpy array
           Ground truth of the data. E.g. ``(num_of_images, y, x, channels)``.

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

       .. image:: ../../img/278_3_threshold_Jaccard.png
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
            raise ValueError("Filenames array and length of X have different shapes: {} vs {}".format(len(filenames),len(X)))

    if not isinstance(X, list):
        _dtype = X.dtype if X.dtype in [np.uint8, np.uint16, np.float32] else np.float32
        ndims = X.ndim
    else:
        _dtype = X[0].dtype if X[0].dtype in [np.uint8, np.uint16, np.float32] else np.float32
        ndims = X[0].ndim

    d = len(str(len(X)))
    for i in tqdm(range(len(X)), leave=False):
        if filenames is None:
            f = os.path.join(data_dir, str(i).zfill(d)+'.tif')
        else:
            f = os.path.join(data_dir, os.path.splitext(filenames[i])[0]+'.tif')
        if ndims == 4:
            if not isinstance(X, list):
                aux = np.expand_dims(np.expand_dims(X[i],0).transpose((0,3,1,2)), -1).astype(_dtype)
            else:
                aux = np.expand_dims(np.expand_dims(X[i][0],0).transpose((0,3,1,2)), -1).astype(_dtype)
        else:
            if not isinstance(X, list):
                aux = np.expand_dims(X[i].transpose((0,3,1,2)), -1).astype(_dtype)
            else:
                aux = np.expand_dims(X[i][0].transpose((0,3,1,2)), -1).astype(_dtype)
        try:
            imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False, compression=('zlib', 1))
        except:
            imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False)


def save_tif_pair_discard(X, Y, data_dir=None, suffix="", filenames=None, discard=True, verbose=True):
    """Save images in the given directory.

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

    os.makedirs(os.path.join(data_dir, 'x'+suffix), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'y'+suffix), exist_ok=True)
    if filenames is not None:
        if len(filenames) != len(X):
            raise ValueError("Filenames array and length of X have different shapes: {} vs {}".format(len(filenames),len(X)))

    _dtype = X.dtype if X.dtype in [np.uint8, np.uint16, np.float32] else np.float32
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
                aux = np.expand_dims(np.expand_dims(X[i],0).transpose((0,3,1,2)), -1).astype(_dtype)
            else:
                aux = np.expand_dims(X[i].transpose((0,3,1,2)), -1).astype(_dtype)
            imsave(f1, aux, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False, compression=('zlib', 1))
            if Y.ndim == 4:
                aux = np.expand_dims(np.expand_dims(Y[i],0).transpose((0,3,1,2)), -1).astype(_dtype)
            else:
                aux = np.expand_dims(Y[i].transpose((0,3,1,2)), -1).astype(_dtype)
            imsave(f2, aux, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False, compression=('zlib', 1))


def save_img(X=None, data_dir=None, Y=None, mask_dir=None, scale_mask=True,
             prefix="", extension=".png", filenames=None):
    """Save images in the given directory.

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

       .. image:: ../../img/save_filters.png
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


def check_masks(path, n_classes=2):
    """Check Whether the data masks have the correct labels inspection a few random images of the given path. If the
       function gives no error one should assume that the masks are correct.

       Parameters
       ----------
       path : str
           Path to the data mask.

       n_classes : int, optional
           Maximum classes that the masks must contain.
    """

    print("Checking Whether the images in {} are binary . . .".format(path))

    ids = sorted(next(os.walk(path))[2])

    # Check only 4 random images or less if there are not as many
    num_sample = [4, len(ids)]
    numbers = random.sample(range(0, len(ids)), min(num_sample))
    for i in numbers:
        img = imread(os.path.join(path, ids[i]))
        values, _ = np.unique(img, return_counts=True)
        if len(values) > n_classes :
            raise ValueError("Error: given mask ({}) has more classes than specified in 'MODEL.N_CLASSES'."
                             "That variable value need to be set without counting with background class. " 
                             " E.g. if mask has [0,1,2] 'MODEL.N_CLASSES' should be 2.\n"
                             "Values found: {}".format(os.path.join(path, ids[i]), values))
        if not (values == range(len(values))).all() and len(values) > 2:
            raise ValueError("Mask values need to be consecutive. E.g. [0,1,2,3...]. Provided: {}"
                .format(values))

def img_to_onehot_encoding(img, num_classes=2):
    """Converts image given into one-hot encode format.

       The opposite function is :func:`~onehot_encoding_to_img`.

       Parameters
       ----------
       img : Numpy 3D/4D array
           Image. E.g. ``(y, x, channels)`` or ``(z, y, x, channels)``.

       num_classes : int, optional
           Number of classes to distinguish.

       Returns
       -------
       one_hot_labels : Numpy 3D/4D array
           Data one-hot encoded. E.g. ``(y, x, num_classes)`` or ``(z, y, x, num_classes)``.
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
           Image. E.g. ``(y, x, channels)`` or ``(z, y, x, channels)``.

       Returns
       -------
       img : Numpy 3D/4D array
           Data one-hot encoded. E.g. ``(z, y, x, num_classes)``.
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
                       reflect_to_complete_shape=False, check_channel=True):
    """Load data from a directory. If ``crop=False`` all the data is suposed to have the same shape.

       Parameters
       ----------
       data_dir : str
           Path to read the data from.

       crop : bool, optional
           Crop each image into desired shape pointed by ``crop_shape``.

       crop_shape : Tuple of 3 ints, optional
           Shape of the crop to be made. E.g. ``(y, x, channels)``.

       overlap : Tuple of 2 floats, optional
           Amount of minimum overlap on x and y dimensions. The values must  be on range ``[0, 1)``, that is, ``0%`` or
           ``99%`` of overlap. E. g. ``(y, x)``.

       padding : Tuple of 2 ints, optional
           Size of padding to be added on each axis ``(y, x)``. E.g. ``(24, 24)``.

       return_filenames : bool, optional
           Return a list with the loaded filenames. Useful when you need to save them afterwards with the same names as
           the original ones.

       reflect_to_complete_shape : bool, optional
           Whether to increase the shape of the dimension that have less size than selected patch size padding it with
           'reflect'.

       check_channel : bool, optional
           Whether to check if the crop_shape channel matches with the loaded images' one. 
           
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

    if len(ids) == 0:
        raise ValueError("No images found in dir {}".format(data_dir))

    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        if id_.endswith('.npy'):
            img = np.load(os.path.join(data_dir, id_))
        else:
            img = imread(os.path.join(data_dir, id_))

        if return_filenames: filenames.append(id_)

        if img.ndim == 2:
            img = np.expand_dims(img, -1)
        else:
            if img.shape[0] <= 3: img = img.transpose((1,2,0))  

        if reflect_to_complete_shape: img = pad_and_reflect(img, crop_shape, verbose=False)

        if crop_shape is not None and check_channel:
            if crop_shape[-1] != img.shape[-1]:
                raise ValueError("Channel of the patch size given {} does not correspond with the loaded image {}. "
                    "Please, check the channels of the images!".format(crop_shape[-1], img.shape[-1]))

        data_shape.append(img.shape)
        img = np.expand_dims(img, axis=0)
        if crop and img[0].shape != crop_shape[:2]+(img.shape[-1],):
            img = crop_data_with_overlap(img, crop_shape[:2]+(img.shape[-1],), overlap=overlap, padding=padding,
                                        verbose=False)
        c_shape.append(img.shape)
        data.append(img)

    same_shape = True
    s = data[0].shape
    dtype = data[0].dtype
    for i in range(1,len(data)):
        if dtype != data[i].dtype:
            raise ValueError("Data type mismatch {} and {} found in the dataset. Please check it and ensure all"
                             " images have same data type".format(dtype,data[i].dtype))
        if s != data[i].shape:
            same_shape = False

    if crop or same_shape:
        data = np.concatenate(data)
        print("*** Loaded data shape is {}".format(data.shape))
    else:
        print("*** Loaded data shape is {}".format((len(data),)+data[0].shape[1:]))

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
           E.g. ``(y, x, channels)``.

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
                            median_padding=False, reflect_to_complete_shape=False, check_channel=True,
                            return_filenames=False):
    """Load data from a directory.

       Parameters
       ----------
       data_dir : str
           Path to read the data from.

       crop : bool, optional
           Crop each 3D image when readed.

       crop_shape : Tuple of 4 ints, optional
           Shape of the subvolumes to create when cropping.  E.g. ``(z, y, x, channels)``.

       verbose : bool, optional
           Whether to enable verbosity.

       overlap : Tuple of 3 floats, optional
           Amount of minimum overlap on z, y and x dimensions. The values must be on range ``[0, 1)``, that is, ``0%``
           or ``99%`` of overlap. E.g. ``(z, y, x)``.

       padding : Tuple of 3 ints, optional
           Size of padding to be added on each axis ``(z, y, x)``. E.g. ``(24, 24, 24)``.

       median_padding : bool, optional
           If ``True`` the padding value is the median value. If ``False``, the added values are zeroes.

       reflect_to_complete_shape : bool, optional
           Whether to increase the shape of the dimension that have less size than selected patch size padding it with
           'reflect'.

       check_channel : bool, optional
           Whether to check if the crop_shape channel matches with the loaded images' one.
           
       return_filenames : bool, optional
           Return a list with the loaded filenames. Useful when you need to save them afterwards with the same names as
           the original ones.

       Returns
       -------
       data : 5D Numpy array or list of 4D Numpy arrays
           Data loaded. E.g. ``(num_of_images, z, y, x, channels)`` if all files have same shape, otherwise a list of
           ``(1, z, y, x, channels)`` arrays will be returned.

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
           #     *** Loaded data shape is (350, 40, 256, 256, 1)
           # Notice height, width and depth swap as skimage.io imread function is used to load images.
    """
    if crop and crop_shape is None:
        raise ValueError("'crop_shape' must be provided when 'crop' is True")

    print("Loading data from {}".format(data_dir))
    ids = sorted(next(os.walk(data_dir))[2])

    if len(ids) == 0:
        raise ValueError("No images found in dir {}".format(data_dir))

    if crop:
        from data.data_3D_manipulation import crop_3D_data_with_overlap

    data = []
    data_shape = []
    c_shape = []
    if return_filenames: filenames = []
    ax = None

    # Read images
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        if id_.endswith('.npy'):
            img = np.load(os.path.join(data_dir, id_))
        else:
            img = imread(os.path.join(data_dir, id_))
        img = np.squeeze(img)

        if img.ndim < 3:
            raise ValueError("Read image seems to be 2D: {}. Path: {}".format(img.shape, os.path.join(data_dir, id_)))

        if img.ndim == 3: 
            img = np.expand_dims(img, -1)
        else:
            min_val = min(img.shape)
            channel_pos = img.shape.index(min_val)
            if channel_pos != 3 and img.shape[channel_pos] <= 4:
                new_pos = [x for x in range(4) if x != channel_pos]+[channel_pos,]
                img = img.transpose(new_pos)

        if return_filenames: filenames.append(id_)
        if reflect_to_complete_shape: img = pad_and_reflect(img, crop_shape, verbose=verbose)
        
        if crop_shape is not None and check_channel:
            if crop_shape[-1] != img.shape[-1]:
                raise ValueError("Channel of the patch size given {} does not correspond with the loaded image {}. "
                    "Please, check the channels of the images!".format(crop_shape[-1], img.shape[-1]))

        data_shape.append(img.shape)
        if crop and img.shape != crop_shape[:3]+(img.shape[-1],):
            img = crop_3D_data_with_overlap(img, crop_shape[:3]+(img.shape[-1],), overlap=overlap, padding=padding,
                                            median_padding=median_padding, verbose=verbose)
        else:
            img = np.expand_dims(img, axis=0)
        
        c_shape.append(img.shape)
        data.append(img)

    same_shape = True
    s = data[0].shape
    dtype = data[0].dtype
    for i in range(1,len(data)):
        if dtype != data[i].dtype:
            raise ValueError("Data type mismatch {} and {} found in the dataset. Please check it and ensure all"
                             " images have same data type".format(dtype,data[i].dtype))
        if s != data[i].shape:
            same_shape = False

    if crop or same_shape:
        data = np.concatenate(data)
        print("*** Loaded data shape is {}".format(data.shape))
    else:
        print("*** Loaded data[0] shape is {}".format(data[0].shape))

    if return_filenames:
        return data, data_shape, c_shape, filenames
    else:
        return data, data_shape, c_shape


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

def pad_and_reflect(img, crop_shape, verbose=False):
    """Load data from a directory.

       Parameters
       ----------
       img : 3D/4D Numpy array
           Image to pad. E.g. ``(y, x, channels)`` or ``(z, y, x, c)``.

       crop_shape : Tuple of 3/4 ints, optional
           Shape of the subvolumes to create when cropping.  E.g. ``(y, x, channels)`` or ``(z, y, x, channels)``.

       verbose : bool, optional
           Whether to output information.

       Returns
       -------
       img : 3D/4D Numpy array
           Image padded (if needed). E.g. ``(y, x, channels)`` or ``(z, y, x, channels)``.
    """
    if img.ndim == 4 and len(crop_shape) != 4:
        raise ValueError("'crop_shape' needs to have 4 values as the input array has 4 dims")
    if img.ndim == 3 and len(crop_shape) != 3:
        raise ValueError("'crop_shape' needs to have 3 values as the input array has 3 dims")

    if img.ndim == 4:
        if img.shape[0] < crop_shape[0]:
            diff = crop_shape[0]-img.shape[0]
            o_shape = img.shape
            img = np.pad(img, ((diff,0),(0,0),(0,0),(0,0)), 'reflect')
            if verbose: print("Reflected from {} to {}".format(o_shape, img.shape))

        if img.shape[1] < crop_shape[1]:
            diff = crop_shape[1]-img.shape[1]
            o_shape = img.shape
            img = np.pad(img, ((0,0),(diff,0),(0,0),(0,0)), 'reflect')
            if verbose: print("Reflected from {} to {}".format(o_shape, img.shape))

        if img.shape[2] < crop_shape[2]:
            diff = crop_shape[2]-img.shape[2]
            o_shape = img.shape
            img = np.pad(img, ((0,0),(0,0),(diff,0),(0,0)), 'reflect')
            if verbose: print("Reflected from {} to {}".format(o_shape, img.shape))
    else:
        if img.shape[0] < crop_shape[0]:
            diff = crop_shape[0]-img.shape[0]
            o_shape = img.shape
            img = np.pad(img, ((diff,0),(0,0),(0,0)), 'reflect')
            if verbose: print("Reflected from {} to {}".format(o_shape, img.shape))

        if img.shape[1] < crop_shape[1]:
            diff = crop_shape[1]-img.shape[1]
            o_shape = img.shape
            img = np.pad(img, ((0,0),(diff,0),(0,0)), 'reflect')
            if verbose: print("Reflected from {} to {}".format(o_shape, img.shape))
    return img

    
def check_value(value, value_range=(0,1)):
    """Checks if a value is within a range """
    if isinstance(value, list) or isinstance(value, tuple):
        for i in range(len(value)):
            if not (value_range[0] <= value[i] <= value_range[1]):    
                return False
        return True 
    else:   
        if value_range[0] <= value <= value_range[1]:
            return True
        return False