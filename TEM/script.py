# Import all the necessary libraries
import os
import datetime
import glob
import random
import sys
from PIL import Image

import matplotlib.pyplot as plt
import skimage.io                                     #Used for imshow function
import skimage.transform                              #Used for resize function
from skimage.io import imread

import numpy as np
import pandas as pd
from tqdm import tqdm

import keras
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Conv2DTranspose
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.models import load_model, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.merge import add, concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import multi_gpu_model, plot_model
from keras import backend as K
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split

print('Python       :', sys.version.split('\n')[0])
print('Numpy        :', np.__version__)
print('Keras        :', keras.__version__)
print('Tensorflow   :', tf.__version__)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="3";
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

seedValue = 42
random.seed = seedValue
np.random.seed(seed=seedValue)

# Set some parameters
IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 1
TRAIN_PATH = 'data/train/x/'
TRAIN_MASK_PATH = 'data/train/y/'
batch_size=4

###############################
###############################

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

###############################
###############################

# Build U-Net model
inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = Lambda(lambda x: x / 255) (inputs)

c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
c1 = Dropout(0.1) (c1)
c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
c2 = Dropout(0.1) (c2)
c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
c3 = Dropout(0.2) (c3)
c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
c4 = Dropout(0.2) (c4)
c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
c5 = Dropout(0.3) (c5)
c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
c6 = Dropout(0.2) (c6)
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
c7 = Dropout(0.2) (c7)
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
c8 = Dropout(0.1) (c8)
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
c9 = Dropout(0.1) (c9)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
model.summary()

###############################
###############################

import matplotlib.pyplot as plt
from keras.callbacks import Callback
from IPython.display import clear_output
#from matplotlib.ticker import FormatStrFormatter

def translate_metric(x):
    translations = {'acc': "Accuracy", 'loss': "Log-loss (cost function)"}
    if x in translations:
        return translations[x]
    else:
        return x

class PlotLosses(Callback):
    def __init__(self, figsize=None):
        super(PlotLosses, self).__init__()
        self.figsize = figsize

    def on_train_begin(self, logs={}):
        self.base_metrics = [metric for metric in self.params['metrics'] if not metric.startswith('val_')]
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs.copy())

        clear_output(wait=True)
        plt.figure(figsize=self.figsize)
        
        for metric_id, metric in enumerate(self.base_metrics):
            plt.subplot(1, len(self.base_metrics), metric_id + 1)
            
            plt.plot(range(1, len(self.logs) + 1),
                     [log[metric] for log in self.logs],
                     label="training")
            if self.params['do_validation']:
                plt.plot(range(1, len(self.logs) + 1),
                         [log['val_' + metric] for log in self.logs],
                         label="validation")
            plt.title(translate_metric(metric))
            plt.xlabel('epoch')
            plt.legend(loc='center left')
        
        plt.tight_layout()
        plt.show();

plot_losses = PlotLosses(figsize=(16, 4))

###############################
###############################

train_ids = sorted(next(os.walk(TRAIN_PATH))[2])
train_mask_ids = sorted(next(os.walk(TRAIN_MASK_PATH))[2])

# Get and resize train images and masks
X_data = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_data = np.zeros((len(train_mask_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

print('Getting and resizing train images... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    img = imread(TRAIN_PATH + id_)
    img = np.expand_dims(img, axis=-1)
    X_data[n] = img


print('Getting and resizing train masks... ')
for n, id_ in tqdm(enumerate(train_mask_ids), total=len(train_mask_ids)):
    mask = imread(TRAIN_MASK_PATH + id_)
    mask = np.expand_dims(mask, axis=-1)
    Y_data[n] = mask

Y_data = Y_data/255

print("#################################")
print(train_ids)
print("#################################")
print(train_mask_ids)
print("#################################")
print(X_data.shape)
print(Y_data.shape)
print("#################################")

# Split the data 
validation_split=0.25
X_train, X_test, Y_train, Y_test = train_test_split(X_data,
                                                    Y_data,
                                                    train_size=1-validation_split,
                                                    test_size=validation_split,
                                                    random_state=seedValue)

print("#################################")
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
print("#################################")

# Image data generator distortion options
data_gen_args = dict(rotation_range=45.,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     shear_range=0.2,
                     zoom_range=0.2,
                     horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode='reflect')

# Train data, provide the same seed and keyword arguments to the fit and flow methods
X_datagen = ImageDataGenerator(**data_gen_args)
Y_datagen = ImageDataGenerator(**data_gen_args)
X_datagen.fit(X_train, augment=True, seed=seedValue)
Y_datagen.fit(Y_train, augment=True, seed=seedValue)

i=0
for batch in X_datagen.flow(X_train, save_to_dir="aug_x", batch_size=batch_size, shuffle=True, seed=seedValue, save_prefix='x', save_format='jpeg'):
    i += 1
    if i > 5:
        break

i=0
for batch in Y_datagen.flow(Y_train, save_to_dir="aug_y", batch_size=batch_size, shuffle=True, seed=seedValue, save_prefix='y', save_format='jpeg'):
    i += 1
    if i > 5:
        break

X_train_augmented = X_datagen.flow(X_train, batch_size=batch_size, shuffle=True, seed=seedValue)
Y_train_augmented = Y_datagen.flow(Y_train, batch_size=batch_size, shuffle=True, seed=seedValue)

# Test data, no data augmentation, but we create a generator anyway
X_datagen_val = ImageDataGenerator()
Y_datagen_val = ImageDataGenerator()
X_datagen_val.fit(X_test, augment=True, seed=seedValue)
Y_datagen_val.fit(Y_test, augment=True, seed=seedValue)
X_test_augmented = X_datagen_val.flow(X_test, batch_size=batch_size, shuffle=True, seed=seedValue)
Y_test_augmented = Y_datagen_val.flow(Y_test, batch_size=batch_size, shuffle=True, seed=seedValue)
    
# combine generators into one which yields image and masks
train_generator = zip(X_train_augmented, Y_train_augmented)
test_generator = zip(X_test_augmented, Y_test_augmented)

# Fit model
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model-tem_test1.h5', verbose=1, save_best_only=True)
model.fit_generator(train_generator, validation_data=test_generator,
                    validation_steps=batch_size/2, steps_per_epoch=len(X_train)/(batch_size*2),
                    epochs=50, callbacks=[earlystopper, checkpointer])

# Save the model weights
model.save_weights(filepath="model-weights.hdf5")

#####################
#####################

#model_loaded.load_weights("model-weights.hdf5")

# Use model to predict test labels
Y_hat = model.predict(X_test, verbose=1)
Y_hat.shape

# Save results as image
outputDir='results'

for i in range(0, len(Y_hat)):
    im = Image.fromarray(Y_hat[i,:,:,0]*255)
    im = im.convert('RGB')
    im.save(os.path.join(outputDir,"test_out" + str(i) + ".png"))

#####################
#####################

outputDir='prueba'
for i in range(0,X_train.shape[0]):
    im = Image.fromarray(X_train[i,:,:,0])
    im = im.convert('RGB')
    im.save(os.path.join(outputDir, str(i) + ".png"))

for i in range(0,Y_train.shape[0]):
    im = Image.fromarray(Y_train[i,:,:,0]*255)
    im = im.convert('RGB')
    im.save(os.path.join(outputDir, "y_" + str(i) + ".png"))

outputDir2='test'
for i in range(0,X_test.shape[0]):
    im = Image.fromarray(X_test[i,:,:,0])
    im = im.convert('RGB')
    im.save(os.path.join(outputDir2, str(i) + ".png"))

for i in range(0,Y_test.shape[0]):
    im = Image.fromarray(Y_test[i,:,:,0]*255)
    im = im.convert('RGB')
    im.save(os.path.join(outputDir2, "y_" + str(i) + ".png"))
