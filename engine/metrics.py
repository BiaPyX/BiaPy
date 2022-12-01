import time
import os
import distutils
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from skimage import measure
from PIL import Image
from tensorflow.keras import losses
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment


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

    if y_true.ndim != y_pred.ndim:
        raise ValueError("Dimension mismatch: {} and {} provided".format(y_true.shape, y_pred.shape))

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

    if y_true.ndim != y_pred.ndim:
        raise ValueError("Dimension mismatch: {} and {} provided".format(y_true.shape, y_pred.shape))

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


def jaccard_index_softmax(y_true, y_pred, t=0.5):
    """Define Jaccard index. Assumes that the 0 channel is background so does not 
       compute the IoU on it. 

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

    tot_jac = K.cast(0.000, dtype='float64')
    for i in range(0,y_pred.shape[-1]):
        y_pred_[y_pred_== (i+1)] = 1
        y_true_[y_true_== (i+1)] = 1
        TP = tf.math.count_nonzero(y_pred_ * y_true_)
        FP = tf.math.count_nonzero(y_pred_ * (y_true_ - 1))
        FN = tf.math.count_nonzero((y_pred_ - 1) * y_true_)

    tot_jac += tf.cond(tf.greater((TP + FP + FN), 0), lambda: TP / (TP + FP + FN), lambda: K.cast(0.000, dtype='float64'))

    return tot_jac/(y_pred.shape[-1]-1)


def IoU_instances(t=0.5, binary_channels=2):
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

    def jaccard_index_instances(y_true, y_pred):
        y_pred_ = tf.cast(y_pred[...,:binary_channels] > t, dtype=tf.int32)
        y_true_ = tf.cast(y_true[...,:binary_channels] > t, dtype=tf.int32)

        TP = tf.math.count_nonzero(y_pred_ * y_true_)
        FP = tf.math.count_nonzero(y_pred_ * (y_true_ - 1))
        FN = tf.math.count_nonzero((y_pred_ - 1) * y_true_)

        jac = tf.cond(tf.greater((TP + FP + FN), 0), lambda: TP / (TP + FP + FN), lambda: K.cast(0.000, dtype='float64'))
        return jac

    return jaccard_index_instances


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
            return weights[0]*losses.binary_crossentropy(tf.expand_dims(tf.cast(y_true[...,0], dtype=tf.float32),-1), tf.expand_dims(y_pred[...,0],-1))+\
                    weights[1]*losses.binary_crossentropy(tf.expand_dims(tf.cast(y_true[...,1], dtype=tf.float32),-1), tf.expand_dims(y_pred[...,1],-1))
        elif out_channels == "BCM":
            return weights[0]*losses.binary_crossentropy(tf.expand_dims(tf.cast(y_true[...,0], dtype=tf.float32),-1), tf.expand_dims(y_pred[...,0],-1))+\
                    weights[1]*losses.binary_crossentropy(tf.expand_dims(tf.cast(y_true[...,1], dtype=tf.float32),-1), tf.expand_dims(y_pred[...,1],-1))+\
                    weights[2]*losses.binary_crossentropy(tf.expand_dims(tf.cast(y_true[...,2], dtype=tf.float32),-1), tf.expand_dims(y_pred[...,2],-1))   
        elif out_channels == "BCD":
            return weights[0]*losses.binary_crossentropy(tf.expand_dims(tf.cast(y_true[...,0], dtype=tf.float32),-1), tf.expand_dims(y_pred[...,0],-1))+\
                    weights[1]*losses.binary_crossentropy(tf.expand_dims(tf.cast(y_true[...,1], dtype=tf.float32),-1), tf.expand_dims(y_pred[...,1],-1))+\
                    weights[2]*masked_mse(tf.cast(y_true[...,2], dtype=tf.float32), y_pred[...,2], tf.cast(y_true[...,0], dtype=tf.float32))
        elif out_channels == "BCDv2":
            return weights[0]*losses.binary_crossentropy(tf.expand_dims(tf.cast(y_true[...,0], dtype=tf.float32),-1), tf.expand_dims(y_pred[...,0],-1))+\
                    weights[1]*losses.binary_crossentropy(tf.expand_dims(tf.cast(y_true[...,1], dtype=tf.float32),-1), tf.expand_dims(y_pred[...,1],-1))+\
                    weights[2]*K.mean(tf.expand_dims(K.square(tf.cast(y_true[...,2], dtype=tf.float32) - y_pred[...,2]), -1), axis=-1)
        elif out_channels == "BDv2":
            return weights[0]*losses.binary_crossentropy(tf.expand_dims(tf.cast(y_true[...,0], dtype=tf.float32),-1), tf.expand_dims(y_pred[...,0],-1))+\
                   weights[1]*masked_mse(tf.cast(y_true[...,1], dtype=tf.float32), y_pred[...,1], tf.cast(y_true[...,0], dtype=tf.float32))
        # Dv2
        else:
            return K.mean(K.square(tf.cast(y_true, dtype=tf.float32) - y_pred), axis=-1)
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


def detection_metrics(true, pred, tolerance=10, voxel_size=(1,1,1), verbose=False):
    """Calculate detection metrics based on

       Parameters
       ----------
       true : List of list
           List containing coordinates of ground truth points. E.g. ``[[5,3,2], [4,6,7]]``.

       pred : 4D Tensor
           List containing coordinates of predicted points. E.g. ``[[5,3,2], [4,6,7]]``.

       tolerance : optional, int
           Maximum distance far away from a GT point to consider a point as a true positive.

       voxel_size : List of floats
           Weights to be multiply by each axis. Useful when dealing with anysotropic data to reduce the distance value
           on the axis with less resolution. E.g. ``(1,1,0.5)``.

       Returns
       -------
       metrics : List of strings
           List containing precision, accuracy and F1 between the predicted points and ground truth.

    """

    _true = np.array(true, dtype=np.float32)
    _pred = np.array(pred, dtype=np.float32)
    
    # Multiply each axis for the its real value
    for i in range(len(voxel_size)):
        _true[:,i] *= voxel_size[i]
        _pred[:,i] *= voxel_size[i]

    # Create cost matrix
    distances = distance_matrix(_pred, _true)

    pred_ind, true_ind = linear_sum_assignment(distances)

    TP, FP, FN = 0, 0, 0
    for i in range(len(pred_ind)):
        if distances[pred_ind[i],true_ind[i]] < tolerance:
            TP += 1

    FN = len(_true) - TP
    FP = len(_pred) - TP

    try:
        precision = TP/(TP+FP)
    except:
        precision = 1
    try:
        recall = TP/(TP+FN)
    except:
        recall = 1
    try:
        F1 = 2*((precision*recall)/(precision+recall))
    except:
        F1 = 1

    if verbose:
    	print("Points in ground truth: {}, Points in prediction: {}".format(len(_true), len(_pred)))
    	print("True positives: {}, False positives: {}, False negatives: {}".format(TP, FP, FN))
        
    return ["Precision", precision, "Recall", recall, "F1", F1]

def masked_bce_loss( y_true, y_pred ):
    """Binary cross-entropy loss masking pixels of value 2 out.

       Based on `U-Net: deep learning for cell counting, detection, and morphometry <https://www.nature.com/articles/s41592-018-0261-2>`_.

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
    ring_value = tf.constant( [ 2.0 ], dtype=tf.float32 )
    exclusion_mask = tf.dtypes.cast( tf.math.less( y_true, ring_value ), tf.float32 )
    return losses.binary_crossentropy( y_true * exclusion_mask, y_pred * exclusion_mask )

def masked_jaccard_index(y_true, y_pred, t=0.5, mask_value=2.0):
    """Define Jaccard index masking out some pixels.

       Parameters
       ----------
       y_true : Tensor
           Ground truth masks.

       y_pred : Tensor
           Predicted masks.

       t : optional, float
           Threshold to be applied.

       mask_value : optional, float
           Value of the pixels to ommit.

       Returns
       -------
       jac : Tensor
           Jaccard index value
    """

    exclusion_mask = tf.cast( y_true < mask_value, dtype=tf.int32 )

    y_pred_ = exclusion_mask * tf.cast(y_pred > t, dtype=tf.int32)
    y_true = exclusion_mask * tf.cast(y_true, dtype=tf.int32)

    TP = tf.math.count_nonzero(y_pred_ * y_true)
    FP = tf.math.count_nonzero(y_pred_ * (y_true - 1))
    FN = tf.math.count_nonzero((y_pred_ - 1) * y_true)

    jac = tf.cond(tf.greater((TP + FP + FN), 0), lambda: TP / (TP + FP + FN), lambda: K.cast(0.000, dtype='float64'))

    return jac

def PSNR(super_resolution, high_resolution, max_val=255):
    """Compute the peak signal-to-noise ratio, measures quality of image."""
    psnr_value = tf.image.psnr(high_resolution, super_resolution, max_val=max_val)[0]
    return psnr_value

def n2v_loss_mse():
    def n2v_mse(y_true, y_pred):
        target, mask = tf.split(y_true, 2, axis=len(y_true.shape)-1)
        mask = tf.cast(mask, dtype=tf.float32)
        target = tf.cast(target, dtype=tf.float32)
        loss = tf.reduce_sum(K.square(target - y_pred*mask)) / tf.reduce_sum(mask)
        return loss

    return n2v_mse

