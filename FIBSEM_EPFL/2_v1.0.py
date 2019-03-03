import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="3";

import sys
import random
import warnings

import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# Set some parameters
#IMG_WIDTH = 1024
#IMG_HEIGHT = 768
IMG_WIDTH = 512
IMG_HEIGHT = 384
IMG_CHANNELS = 1
TRAIN_PATH = 'data/train/x/'
TRAIN_MASK_PATH = 'data/train/y/'

TEST_PATH = 'data/test/x/'
TEST_MASK_PATH = 'data/test/y/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


#####################
#####################

# Load the model saved
model = load_model('model-v.1.0.fibsem.h5', custom_objects={'mean_iou': mean_iou})

# Predict on train, val and test
preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

# Threshold predictions
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

# Save results as image
from PIL import Image
outputDir='results'

for i in range(0,len(preds_train)):
    im = Image.fromarray(preds_train[i,:,:,0])
    im = im.convert('RGB')
    im.save(os.path.join(outputDir,"out" + str(i) + ".png"))

