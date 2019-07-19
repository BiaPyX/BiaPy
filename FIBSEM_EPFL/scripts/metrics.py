import time
from keras import backend as K
import tensorflow as tf
import numpy as np

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

    print("Foreground IoU: " + str(foreground_iou))
    print("Background IoU: " + str(background_iou))
    print("VOC: " + str(voc))

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
