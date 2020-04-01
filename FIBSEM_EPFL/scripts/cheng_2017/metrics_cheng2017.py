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

def jaccard_loss_cheng2017(y_true, y_pred):
    C = 1
    _y_pred = tf.cast((y_pred > 0.5), dtype=tf.float32)
    numerator = tf.reduce_sum(y_pred*(_y_pred*y_true)) + C
    denominator = tf.reduce_sum(y_pred + (_y_pred*y_true) - numerator) + 2*C
    
    return 1 - (numerator / denominator)
    
