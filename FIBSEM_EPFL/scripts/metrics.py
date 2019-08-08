import time
import os
from keras import backend as K
import tensorflow as tf
import numpy as np
from util import Print
from skimage import measure
import distutils
from distutils import dir_util
from PIL import Image

def jaccard_index_numpy(y_true, y_pred):
    """Define Jaccard index.

       Args:
            y_true (numpy array): ground truth masks.
            y_pred (numpy array): predicted masks.

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


def mean_iou(y_true, y_pred):
    """Define IoU metric.

       Args:
            y_true (tensor): ground truth masks.
            y_pred (tensor): predicted masks.

       Return:
            meanIoU (tensor): mean IoU value.
    """

    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


def voc_calculation_meanIoU(y_true, y_pred, foreground_iou):
    """Calculate VOC metric value.

        Args:
            y_pred (array): predicted masks.
            y_true (array): ground truth masks.
            foreground_iou (float): foreground IoU score.

        Return:
            voc (float): VOC score value.
    """

    # Invert the arrays
    y_pred[y_pred == 0] = 2
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == 2] = 1

    y_true[y_true == 0] = 2
    y_true[y_true == 1] = 0
    y_true[y_true == 2] = 1

    with tf.Session() as sess:
        ypredT = tf.constant(np.argmax(y_pred, axis=-1))
        ytrueT = tf.constant(np.argmax(y_true, axis=-1))
        iou,conf_mat = tf.metrics.mean_iou(ytrueT, ypredT,
                                           num_classes=3)
        sess.run(tf.local_variables_initializer())
        sess.run([conf_mat])
        background_iou = sess.run([iou])

    voc = (float)(foreground_iou + background_iou)/2

    Print("Foreground IoU: " + str(foreground_iou))
    Print("Background IoU: " + str(background_iou))
    Print("VOC: " + str(voc))

    return voc

def voc_calculation(y_true, y_pred, foreground):
    """Calculate VOC metric value.

        Args:
            y_true (array): ground truth masks.
            y_pred (array): predicted masks.
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
           Y_test (numpy array): ground truth mask.  
           preds_test (numpy array): predicted mask.
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
        Print("No ground truth folder detected. Creating it . . .")
        os.makedirs(ge_path)
    
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
    if not os.path.exists(res_eval_path):
        os.makedirs(res_eval_path)
    
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
