import tensorflow as tf
import numpy as np

def jaccard_loss_cheng2017(y_true, y_pred):
    """Jaccard loss proposed by Cheng et al.

       Parameters
       ----------
       y_true : Tensor
           Ground truth masks.

       y_pred : tensor
           Predicted masks.

       Returns
       -------
       loss : tensor
           Jaccard loss score.
    """
    C = 1
    y_pred = tf.cast(y_pred , dtype=tf.float32)
    y_true = tf.cast(y_true, dtype=tf.float32)
    numerator = tf.reduce_sum(y_true[...,1] * y_pred[...,1])
    denominator = tf.reduce_sum(y_true[...,1] + y_pred[...,1]) - numerator

    jac =  (numerator + C)/(denominator + C)

    return 1 - jac

