import time
import os
import distutils
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from skimage import measure
from distutils import dir_util
from PIL import Image
from tensorflow.keras import losses


def jaccard_index_numpy(y_true, y_pred):
    """Define Jaccard index.

       Parameters
       ----------
       y_true : N dim Numpy array
           Ground truth masks. E.g. ``(num_of_images, x, y, channels)`` for 2D images or
           ``(volume_number, z, x, y, channels)`` for 3D volumes.

       y_pred : N dim Numpy array
           Predicted masks. E.g. ``(num_of_images, x, y, channels)`` for 2D images or
           ``(volume_number, z, x, y, channels)`` for 3D volumes.

       Returns
       -------
       jac : float
           Jaccard index value.
    """

    TP = np.count_nonzero(y_pred * y_true)
    FP = np.count_nonzero(y_pred * (y_true - 1))
    FN = np.count_nonzero((y_pred - 1) * y_true)

    if (TP + FP + FN) == 0:
        jac = 0
    else:
        jac = TP / (TP + FP + FN)

    return jac


def jaccard_index_numpy_without_background(y_true, y_pred):
    """Define Jaccard index excluding the background class (first channel).

       Parameters
       ----------
       y_true : N dim Numpy array
           Ground truth masks. E.g. ``(num_of_images, x, y, channels)`` for 2D images or
           ``(volume_number, z, x, y, channels)`` for 3D volumes.

       y_pred : N dim Numpy array
           Predicted masks. E.g. ``(num_of_images, x, y, channels)`` for 2D images or
           ``(volume_number, z, x, y, channels)`` for 3D volumes.

       Returns
       -------
       jac : float
           Jaccard index value.
    """

    TP = np.count_nonzero(y_pred[...,1:] * y_true[...,1:])
    FP = np.count_nonzero(y_pred[...,1:] * (y_true[...,1:] - 1))
    FN = np.count_nonzero((y_pred[...,1:] - 1) * y_true[...,1:])

    if (TP + FP + FN) == 0:
        jac = 0
    else:
        jac = TP / (TP + FP + FN)

    return jac


def jaccard_index(y_true, y_pred, t=0.5):
    """Define Jaccard index.

       Parameters
       ----------
       y_true : Tensor
           Ground truth masks.

       y_pred : Tensor
           Predicted masks.

       t : float, optional
           Threshold to be applied.

       Returns
       -------
       jac : Tensor
           Jaccard index value
    """

    y_pred_ = tf.cast(y_pred > t, dtype=tf.int32)
    y_true = tf.cast(y_true, dtype=tf.int32)

    TP = tf.math.count_nonzero(y_pred_ * y_true)
    FP = tf.math.count_nonzero(y_pred_ * (y_true - 1))
    FN = tf.math.count_nonzero((y_pred_ - 1) * y_true)

    jac = tf.cond(tf.greater((TP + FP + FN), 0), lambda: TP / (TP + FP + FN), lambda: K.cast(0.000, dtype='float64'))

    return jac


def jaccard_index_without_background(y_true, y_pred, t=0.5):
    """Define Jaccard index.

       Parameters
       ----------
       y_true : Tensor
           Ground truth masks.

       y_pred : Tensor
           Predicted masks.

       t : float, optional
           Threshold to be applied.

       Returns
       -------
       jac : Tensor
           Jaccard index value
    """

    _y_pred = tf.cast(y_pred[...,1:] > t, dtype=tf.int32)
    _y_true = tf.cast(y_true[...,1:], dtype=tf.int32)

    TP = tf.math.count_nonzero(_y_pred * _y_true)
    FP = tf.math.count_nonzero(_y_pred * (_y_true - 1))
    FN = tf.math.count_nonzero((_y_pred - 1) * _y_true)

    jac = tf.cond(tf.greater((TP + FP + FN), 0), lambda: TP / (TP + FP + FN), lambda: K.cast(0.000, dtype='float64'))

    return jac


def jaccard_index_softmax(y_true, y_pred, t=0.5):
    """Define Jaccard index.

       Parameters
       ----------
       y_true : Tensor
           Ground truth masks.

       y_pred : Tensor
           Predicted masks.

       t : float, optional
           Threshold to be applied.

       Returns
       -------
       jac : Tensor
           Jaccard index value
    """

    y_pred_ = tf.cast(y_pred > t, dtype=tf.int32)
    y_pred_ = tf.math.argmax(y_pred_, axis=-1)

    y_true_ = tf.cast(y_true, dtype=tf.int32)
    y_true_ = tf.math.argmax(y_true_, axis=-1)

    TP = tf.math.count_nonzero(y_pred_ * y_true_)
    FP = tf.math.count_nonzero(y_pred_ * (y_true_ - 1))
    FN = tf.math.count_nonzero((y_pred_ - 1) * y_true_)

    jac = tf.cond(tf.greater((TP + FP + FN), 0), lambda: TP / (TP + FP + FN), lambda: K.cast(0.000, dtype='float64'))

    return jac


def jaccard_index_instances(y_true, y_pred, t=0.5):
    """Define Jaccard index. It only applies for the first two segmentation
       channels.

       Parameters
       ----------
       y_true : Tensor
           Ground truth masks.

       y_pred : Tensor
           Predicted masks.

       t : float, optional
           Threshold to be applied.

       Returns
       -------
       jac : Tensor
           Jaccard index value
    """

    y_pred_ = tf.cast(y_pred[...,:2] > t, dtype=tf.int32)
    y_true_ = tf.cast(y_true[...,:2] > t, dtype=tf.int32)

    TP = tf.math.count_nonzero(y_pred_ * y_true_)
    FP = tf.math.count_nonzero(y_pred_ * (y_true_ - 1))
    FN = tf.math.count_nonzero((y_pred_ - 1) * y_true_)

    jac = tf.cond(tf.greater((TP + FP + FN), 0), lambda: TP / (TP + FP + FN), lambda: K.cast(0.000, dtype='float64'))

    return jac


def jaccard_loss(y_true, y_pred):
    """Define Jaccard index.

       Parameters
       ----------
       y_true Tensor
           Ground truth masks.

       y_pred Tensor
           Predicted masks.

       Returns
       -------
       jac : float
           Jaccard loss score.
    """

    numerator = tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred) - numerator

    jac =  numerator / (denominator + tf.keras.backend.epsilon())

    return 1 - jac


def dice_coeff(y_true, y_pred):
    """Dice coefficient.

       Based on `image_segmentation.ipynb <https://colab.research.google.com/github/tensorflow/models/blob/master/samples/outreach/blogs/segmentation_blogpost/image_segmentation.ipynb>`_.

       Parameters
       ----------
       y_true : Tensor
           Ground truth masks.

       y_pred : Tensor
           Predicted masks.

       Returns
       -------
       score : Tensor
           Dice coefficient value.
    """

    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred):
    """Dice loss.

       Based on `image_segmentation.ipynb <https://colab.research.google.com/github/tensorflow/models/blob/master/samples/outreach/blogs/segmentation_blogpost/image_segmentation.ipynb>`_.

       Parameters
       ----------
       y_true : Tensor
           Ground truth masks.

       y_pred : Tensor
           Predicted masks.

       Returns
       -------
       loss : Tensor
           Loss value.
    """

    loss = 1 - dice_coeff(y_true, y_pred)
    return loss


def bce_dice_loss(y_true, y_pred):
    """Loss function based on the combination of BCE and Dice.

       Based on `image_segmentation.ipynb <https://colab.research.google.com/github/tensorflow/models/blob/master/samples/outreach/blogs/segmentation_blogpost/image_segmentation.ipynb>`_.

       Parameters
       ----------
       y_true : Numpy array
           Ground truth.

       y_pred : Numpy array
           Predictions.

       Returns
       -------
       loss : Tensor
           Loss value.
    """
    loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


def weighted_bce_dice_loss(w_dice=0.5, w_bce=0.5):
    """Loss function based on the combination of BCE and Dice weighted.

       Inspired by `https://medium.com/@Bloomore post <https://medium.com/@Bloomore/how-to-write-a-custom-loss-function-with-additional-arguments-in-keras-5f193929f7a0>`_.

       Parameters
       ----------
       w_dice : float, optional
           Weight to be applied to Dice loss.

       w_bce : float, optional
           Weight to be applied to BCE.

       Returns
       -------
       loss : Tensor
           Loss value.
    """
    def loss(y_true, y_pred):
        return losses.binary_crossentropy(y_true, y_pred) * w_bce + dice_loss(y_true, y_pred) * w_dice
    return loss


def voc_calculation(y_true, y_pred, foreground):
    """Calculate VOC metric value.

       Parameters
       ----------
       y_true : 4D Numpy array
           Ground truth masks. E.g. ``(num_of_images, x, y, channels)``.

       y_pred : 4D Numpy array
           Predicted masks. E.g. ``(num_of_images, x, y, channels)``.

       foreground : float
           Foreground Jaccard index score.

       Returns
       -------
       voc : float
           VOC score value.
    """

    # Invert the arrays
    start_time = time.time()
    y_pred[y_pred == 0] = 2
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == 2] = 1

    y_true[y_true == 0] = 2
    y_true[y_true == 1] = 0
    y_true[y_true == 2] = 1

    background = jaccard_index_numpy(y_true, y_pred)
    voc = (float)(foreground + background)/2

    # Revert the changes
    y_pred[y_pred == 0] = 2
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == 2] = 1

    y_true[y_true == 0] = 2
    y_true[y_true == 1] = 0
    y_true[y_true == 2] = 1

    return voc


def DET_calculation(Y_test, preds_test, ge_path, eval_path, det_bin, n_dig, job_id="0"):
    """Cell tracking challenge detection accuracy (DET) calculation. This function uses the binary provided by the
       challenge to detect the cell and it needs to store the images into some folders.

       To obtain more info please visit the following link:

           http://celltrackingchallenge.net/evaluation-methodology/

       The name of the folders that are here created follow the conventions listed
       in this `link <https://public.celltrackingchallenge.net/documents/Naming%20and%20file%20content%20conventions.pdf>`_.

       Parameters
       ----------
       Y_test : 4D Numpy array
           Ground truth mask. E.g. ``(num_of_images, x, y, channels)``.

       preds_test : 4D Numpy array
           Predicted mask. E.g. ``(num_of_images, x, y, channels)``.

       ge_path : str
           Path where the ground truth is stored. If the folder does not exist it will be created with the ``Y_test``
           ground truth.

       eval_path : str
           Path where the evaluation of the metric will be done.

       det_bin : str
           Path to the DET binary provided by the cell tracking challenge.

       n_dig : int
           The number of digits used for encoding temporal indices (e.g., ``3``). Used by the DET calculation binary,
           more info `here <https://public.celltrackingchallenge.net/documents/Evaluation%20software.pdf>`_.

       job_id : str, optional
           Id of the job. Necessary to store the images on a location based on this string.

       Returns
       -------
       det : float
           DET accuracy.
    """

    # Create the ground truth directory to be reused in future runs if it is not
    # created yet
    if not os.path.exists(ge_path):
        print("No ground truth folder detected. Creating it . . .")
        os.makedirs(ge_path, exist_ok=True)

        gt_labels = measure.label(Y_test[:,:,:,0])
        for i in range(0,len(gt_labels)):
            i_dig = "{:0" + n_dig + "d}"
            i_dig  = i_dig.format(i)
            im = Image.fromarray(gt_labels[i].astype('uint8'))
            im = im.convert('I;16')
            im.save(os.path.join(ge_path, "man_track" + str(i_dig) + ".tif"))

    # Copy ground truth folder
    gt_eval_path = os.path.join(eval_path, job_id + '_GT', 'TRA')
    distutils.dir_util.copy_tree(ge_path, gt_eval_path)

    # Create results folder
    res_eval_path = os.path.join(eval_path, job_id + '_RES')
    os.makedirs(res_eval_path, exist_ok=True)

    res_labels = measure.label(preds_test[:,:,:,0])
    for i in range(0,len(res_labels)):
        i_dig = "{:0" + n_dig + "d}"
        i_dig  = i_dig.format(i)
        im = Image.fromarray(res_labels[i].astype('uint8'))
        im = im.convert('I;16')
        im.save(os.path.join(res_eval_path, "mask" + str(i_dig) + ".tif"))

    # Execute the metric with the given binary path
    det_cmd = det_bin + " " + eval_path +  " " + job_id + " " + n_dig
    det_out = os.popen(det_cmd).read()

    det = det_out.split()[2]
    return det


def binary_crossentropy_weighted(weights):
    """Custom binary cross entropy loss. The weights are used to multiply the results of the usual cross-entropy loss
       in order to give more weight to areas between cells close to one another.

       Based on `unet_weights.py <https://github.com/deepimagej/python4deepimagej/blob/master/unet/py_files/unet_weights.py>`_.

       Parameters
       ----------
       weights : float
           Weigth to multiply the BCE value by.

       Returns
       -------
       loss : Tensor
           Loss value.
    """
    def loss(y_true, y_pred):
        return K.mean(weights * K.binary_crossentropy(y_true, y_pred), axis=-1)

    return loss


def instance_segmentation_loss(weights=(1,0.2), out_channels="BC"):
    """Custom loss that mixed BCE and MSE depending on the ``out_channels`` variable.

       Parameters
       ----------
       weights : 2 float tuple, optional
           Weights to be applied to segmentation (binary and contours) and to distances respectively. E.g. ``(1, 0.2)``,
           ``1`` should be multipled by ``BCE`` for the first two channels and ``0.2`` to ``MSE`` for the last channel.

       out_channels : str, optional
           Channels to operate with. Possible values: ``BC`` and ``BCD``. ``BC`` corresponds to use binary
           segmentation+contour. ``BCD`` stands for binary segmentation+contour+distances.
    """

    def loss(y_true, y_pred):
        if out_channels == "BC":
            return weights[0]*losses.binary_crossentropy(tf.cast(y_true[...,:2], dtype=tf.float32), y_pred[...,:2])
        else:
            return weights[0]*losses.binary_crossentropy(tf.cast(y_true[...,:2], dtype=tf.float32), y_pred[...,:2])+\
                   weights[1]*masked_mse(tf.cast(y_true[...,2], dtype=tf.float32), y_pred[...,2], tf.cast(y_true[...,0], dtype=tf.float32))
    return loss


def masked_mse(y_true, y_pred, mask):
    """Apply MSE just in the pixels denoted by ``mask`` variable.

       Parameters
       ----------
       y_true : 4D Tensor
           Ground truth masks. E.g. ``(num_of_images, x, y, channels)``.

       y_pred : 4D Tensor
           Predicted masks. E.g. ``(num_of_images, x, y, channels)``.

       mask : 4F Tensor
           Mask with True values on the pixels that will be involved in MSE calculation.

       Returns
       -------
       value : Tensor
           MSE value.
    """

    return K.mean(tf.expand_dims(mask*K.square(y_true - y_pred), -1), axis=-1)

