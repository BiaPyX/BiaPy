import sys
sys.path.insert(0, '/data2/dfranco/experimentosTFM/FIBSEM_EPFL/scripts/')

from metrics import *
import numpy as np
from scipy import misc
import glob
import tensorflow as tf
import os

# Take arguments
gpu_selected=str(sys.argv[1])
y_true=str(sys.argv[2])
y_pred=str(sys.argv[3])

# Select GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_selected;

image_true = misc.imread(y_true)
image_pred = misc.imread(y_pred)

image_true = image_true / 255
image_pred = image_pred / 255

true_tf = tf.convert_to_tensor(image_true, np.float32)
pred_tf = tf.convert_to_tensor(image_pred, np.float32)

sess = tf.InteractiveSession()  
jac = jaccard_index(true_tf, pred_tf)
print(jac.eval(session=sess))

sess.close()
