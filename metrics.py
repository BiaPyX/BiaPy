import time
import os
from keras import backend as K
import tensorflow as tf
import numpy as np
from skimage import measure
import distutils
from distutils import dir_util
from PIL import Image
from keras import losses

def jaccard_index_numpy(y_true, y_pred):
    """Define Jaccard index.

       Args:
            y_true (N dim Numpy array): ground truth masks.
            E.g. (image_number, x, y, channels) for 2D images or 
            (volume_number, z, x, y, channels) for 3D volumes.

            y_pred (N dim Numpy array): predicted masks.
            E.g. (image_number, x, y, channels) for 2D images or
            (volume_number, z, x, y, channels) for 3D volumes.

       Return:
            jac (float): Jaccard index value
    """

    TP = np.count_nonzero(y_pred * y_true)
    FP = np.count_nonzero(y_pred * (y_true - 1))
    FN = np.count_nonzero((y_pred - 1) * y_true)

    if (TP + FP + FN) == 0:
        jac = 0
    else:
        jac = TP / (TP + FP + FN)

    return jac


def jaccard_index(y_true, y_pred, t=0.5):
    """Define Jaccard index.

       Args:
            y_true (tensor): ground truth masks.

            y_pred (tensor): predicted masks.

            t (float, optional): threshold to be applied.

       Return:
            jac (tensor): Jaccard index value
    """

    y_pred_ = tf.to_int32(y_pred > t)
    y_true = tf.cast(y_true, dtype=tf.int32)

    TP = tf.count_nonzero(y_pred_ * y_true)
    FP = tf.count_nonzero(y_pred_ * (y_true - 1))
    FN = tf.count_nonzero((y_pred_ - 1) * y_true)

    jac = tf.cond(tf.greater((TP + FP + FN), 0), lambda: TP / (TP + FP + FN),
                  lambda: K.cast(0.000, dtype='float64'))

    return jac


def jaccard_index_softmax(y_true, y_pred, t=0.5):
    """Define Jaccard index.

       Args:
            y_true (tensor): ground truth masks.

            y_pred (tensor): predicted masks.

            t (float, optional): threshold to be applied.

       Return:
            jac (tensor): Jaccard index value
    """
    y_pred = tf.to_int32(y_pred > t)
    y_pred_ = tf.identity(y_pred)
    y_pred_ = tf.math.argmax(y_pred_, axis=-1)    
    
    y_true = tf.cast(y_true, dtype=tf.int32)
    y_true_ = tf.identity(y_true)
    y_true_ = tf.math.argmax(y_true_, axis=-1)

    TP = tf.count_nonzero(y_pred_ * y_true_)
    FP = tf.count_nonzero(y_pred_ * (y_true_ - 1))
    FN = tf.count_nonzero((y_pred_ - 1) * y_true_)

    jac = tf.cond(tf.greater((TP + FP + FN), 0), lambda: TP / (TP + FP + FN),
                  lambda: K.cast(0.000, dtype='float64'))

    return jac



def jaccard_loss(y_true, y_pred):
    """Define Jaccard index.

       Args:
            y_true (tensor): ground truth masks.

            y_pred (tensor): predicted masks.

       Return:
            jac (float): Jaccard loss score.
    """

    numerator = tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred) - numerator 

    jac =  numerator / (denominator + tf.keras.backend.epsilon())

    return 1 - jac

'''
def dice_loss(y_true, y_pred):
    """Define Dice loss.
       
       Args:
            y_true (tensor): ground truth masks.
            y_pred (tensor): predicted masks.

        Return:
            dice (float): Dice loss score.

        Based on:
        https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation
    """

    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + tf.square(y_pred))

    dice = numerator / (denominator + tf.keras.backend.epsilon())

    return (1 - dice)


def dice_loss2(y_true, y_pred):
    """Define Dice loss without squaring y_predi in the formula.

       Args:
            y_true (tensor): ground truth masks.
            y_pred (tensor): predicted masks.

        Return:
            dice (float): Dice loss score.

        Based on:
        https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation
    """

    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)

    dice = numerator / (denominator + tf.keras.backend.epsilon())

    return (1 - dice)
'''

### BCE DICE LOSS from https://colab.research.google.com/github/tensorflow/models/blob/master/samples/outreach/blogs/segmentation_blogpost/image_segmentation.ipynb
def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

###

# Weighted BCE+Dice
# Inspired by https://medium.com/@Bloomore/how-to-write-a-custom-loss-function-with-additional-arguments-in-keras-5f193929f7a0
def weighted_bce_dice_loss(w_dice=0.5, w_bce=0.5):
    def loss(y_true, y_pred):
        return losses.binary_crossentropy(y_true, y_pred) * w_bce + dice_loss(y_true, y_pred) * w_dice
    return loss


def voc_calculation(y_true, y_pred, foreground):
    """Calculate VOC metric value.

        Args:
            y_true (4D Numpy array): ground truth masks.
            E.g. (image_number, x, y, channels).

            y_pred (4D Numpy array): predicted masks.
            E.g. (image_number, x, y, channels).

            foreground (float): foreground Jaccard index score.

        Return:
            voc (float): VOC score value.
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


def DET_calculation(Y_test, preds_test, ge_path, eval_path, det_bin, n_dig,
                    job_id="0"):
    """Cell tracking challenge detection accuracy (DET) calculation. This
       function uses the binary provided by the challente to detect the cell
       (det_bin is the path to such binary). To obtain more info please visit
       the followin link:
           http://celltrackingchallenge.net/evaluation-methodology/

       The name of the folders that are here created follow the conventions listed
       in the following link:
           https://public.celltrackingchallenge.net/documents/Naming%20and%20file%20content%20conventions.pdf

       Args:
            Y_test (4D Numpy array): ground truth mask.  
            E.g. (image_number, x, y, channels).

            preds_test (4D Numpy array): predicted mask.
            E.g. (image_number, x, y, channels).

            ge_path (str): path where the ground truth is stored. If the folder
            does not exist it will be created with the Y_test ground truth.

            eval_path (str): path where the evaluation of the metric will be done.

            det_bin (str): path to the DET binary provided by the cell tracking
            challenge.      

            n_dig (int): The number of digits used for encoding temporal indices
            (e.g., 3). Used by the DET calculation binary, more info in: 
                https://public.celltrackingchallenge.net/documents/Evaluation%20software.pdf

            job_id (str, optional): id of the job. 
           
       Return:
            det (float): DET accuracy.
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
    """
    Based on:
        https://github.com/deepimagej/python4deepimagej/blob/master/unet/py_files/unet_weights.py

    Custom binary cross entropy loss. The weights are used to multiply
    the results of the usual cross-entropy loss in order to give more weight
    to areas between cells close to one another.
    
    The variable 'weights' refers to input weight-maps.
    """
    
    def loss(y_true, y_pred): 
        
        return K.mean(weights * K.binary_crossentropy(y_true, y_pred), axis=-1)
    
    return loss

