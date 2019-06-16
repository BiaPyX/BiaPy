import time
from keras import backend as K
import tensorflow as tf

def jaccard_distance(y_true, y_pred, smooth=100):
    """Jaccard distance for semantic segmentation.
    Also known as the intersection-over-union loss.
    This loss is useful when you have unbalanced numbers of pixels 
    within an image because it gives all classes equal weight. However,
    it is not the defacto standard for image segmentation.
    For example, assume you are trying to predict if
    each pixel is cat, dog, or background.
    You have 80% background pixels, 10% dog, and 10% cat.
    If the model predicts 100% background
    should it be be 80% right (as with categorical cross entropy)
    or 30% (with this loss)?
    The loss has been modified to have a smooth gradient as it converges
    on zero.  This has been shifted so it converges on 0 and is smoothed
    to avoid exploding or disappearing gradient.
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    # Arguments
        y_true: The ground truth tensor.
        y_pred: The predicted tensor
        smooth: Smoothing factor. Default is 100.
    # Returns
        The Jaccard distance between the two tensors.
    # References
        - [What is a good evaluation measure for semantic segmentation?](
           http://www.bmva.org/bmvc/2013/Papers/paper0032/paper0032.pdf)
    # Source code obtained from:
        https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/losses/jaccard.py
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


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


def dice_loss(y_true, y_pred):
    """Define Dice loss.

       
       Args:
            y_true (tensor): ground truth masks.
            y_pred (tensor): predicted masks.

        Return:
            dice (tensor): Dice loss score.
    """

    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + tf.square(y_pred))

    dice = numerator / (denominator + tf.keras.backend.epsilon())

    return dice


def dice_loss2(y_true, y_pred):
    """Define Dice loss without squaring y_pred.

       Args:
            y_true (tensor): ground truth masks.
            y_pred (tensor): predicted masks.

       Return:
            dice (tensor): Dice loss score.
    """

    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)

    dice = numerator / (denominator + tf.keras.backend.epsilon())

    return dice


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
    elapsed_time = time.time() - start_time                             
    print("Time inverting arrays: " + str(elapsed_time))

    start_time = time.time()
    background = jaccard_index(y_true, y_pred)
    elapsed_time = time.time() - start_time                             
    print("Time calculating jaccard of the background: "
          + str(elapsed_time))

    sess = tf.InteractiveSession()
    background = background.eval(session=sess)
 
    voc = (float)(foreground + background)/2

    sess.close()

    return voc

