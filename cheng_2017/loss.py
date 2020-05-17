import tensorflow as tf
import numpy as np

def jaccard_loss_cheng2017(y_true, y_pred):
    """Define Jaccard index.

       Args:
            y_true (tensor): ground truth masks.

            y_pred (tensor): predicted masks.

       Return:
            Jaccard loss score.
    """
    C = 1
    numerator = tf.reduce_sum(y_true[...,1] * y_pred[...,1])
    denominator = tf.reduce_sum(y_true[...,1] + y_pred[...,1]) - numerator 

    jac =  (numerator + C)/(denominator + C)

    return 1 - jac

