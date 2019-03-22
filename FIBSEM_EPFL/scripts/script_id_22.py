##########################
#        PREAMBLE        #
##########################

# Limit the number of threads
import os
import mkl
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ["MKL_DYNAMIC"]="FALSE";
os.environ["NUMEXPR_NUM_THREADS"]='1';
os.environ["VECLIB_MAXIMUM_THREADS"]='1';
os.environ["OMP_NUM_THREADS"] = '1';
mkl.set_num_threads(1)


##########################
#        IMPORTS         #
##########################

import numpy
import sys
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt
import keras
import time
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
from tensorflow import set_random_seed
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import math
import cv2


##########################
#    INITIAL VARIABLES   #
##########################

# Take arguments
if len(sys.argv) > 1:
    GPU_SELECTED=str(sys.argv[1])
    JOBID=str(sys.argv[2])
    TESTID=str(sys.argv[3])
    LOGFILE=str(sys.argv[4])
    HISTORYFILE=str(sys.argv[5])

# Try to generate the results as reproducible as possible
seedValue = 42
random.seed = seedValue
np.random.seed(seed=seedValue)
set_random_seed(seedValue)
os.environ["PYTHONHASHSEED"]=str(seedValue);

# Checks
print('JOBID :', JOBID)
print('GPU selected :', GPU_SELECTED)
print('Python       :', sys.version.split('\n')[0])
print('Numpy        :', np.__version__)
print('Keras        :', keras.__version__)
print('Tensorflow   :', tf.__version__)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
if len(sys.argv) > 1:
    os.environ["CUDA_VISIBLE_DEVICES"]=GPU_SELECTED;
else:
    os.environ["CUDA_VISIBLE_DEVICES"]="3";
os.environ['QT_QPA_PLATFORM']='offscreen' # For matplotlib errors in display
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

# Go to the experiments base dir
os.chdir("/data2/dfranco/experimentosTFM/FIBSEM_EPFL")

# Image dimensions
IMG_WIDTH = 1024
IMG_HEIGHT = 768
IMG_CHANNELS = 1

# Paths to put data and results 
TRAIN_PATH = 'data/train/x/'
TRAIN_MASK_PATH = 'data/train/y/'
TEST_PATH = 'data/test/x/'
TEST_MASK_PATH = 'data/test/y/'
outputDir='results/results_' + str(JOBID)
chartOutDir='charts'

# Additional variables
batch_size_value=4


##########################
#  ADDITIONAL FUNCTIONS  #
##########################

# Function to record each epoch time
class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
time_callback = TimeHistory()

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


##########################
#        LOAD DATA       #
##########################

train_ids = sorted(next(os.walk(TRAIN_PATH))[2])
train_mask_ids = sorted(next(os.walk(TRAIN_MASK_PATH))[2])

test_ids = sorted(next(os.walk(TEST_PATH))[2])
test_mask_ids = sorted(next(os.walk(TEST_MASK_PATH))[2])

# Get and resize train images and masks
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_mask_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

print('Getting and resizing train images... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    img = imread(TRAIN_PATH + id_)
    img = np.expand_dims(img, axis=-1)
    X_train[n] = img

print('Getting and resizing train masks... ')
for n, id_ in tqdm(enumerate(train_mask_ids), total=len(train_mask_ids)):
    mask = imread(TRAIN_MASK_PATH + id_)
    mask = np.expand_dims(mask, axis=-1)
    Y_train[n] = mask

Y_train = Y_train/255

# Get and resize test images and masks
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_test = np.zeros((len(test_mask_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

print('Getting and resizing test images... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    img = imread(TEST_PATH + id_)
    img = np.expand_dims(img, axis=-1)
    X_test[n] = img

print('Getting and resizing test masks... ')
for n, id_ in tqdm(enumerate(test_mask_ids), total=len(test_mask_ids)):
    mask = imread(TEST_MASK_PATH + id_)
    mask = np.expand_dims(mask, axis=-1)
    Y_test[n] = mask

Y_test = Y_test/255


# Split the data
validation_split=0.1
X_train, X_val, Y_train, Y_val = train_test_split(X_train,
                                                    Y_train,
                                                    train_size=1-validation_split,
                                                    test_size=validation_split,
                                                    random_state=seedValue)

# Check if training data looks all right
#ix = random.randint(0, len(next(os.walk(TRAIN_PATH))[2] )-1)
#imshow(X_train[ix,:,:,0])
#plt.show()
#imshow(np.squeeze(Y_train[ix]))


##########################
#    DATA AUGMENTATION   #
##########################

# Custom function to rotate the images with a fixed degree
def fixed_dregee(image):
    img = np.array(image)
    
    # get image height, width
    (h, w) = img.shape[:2]
    # calculate the center of the image
    center = (w / 2, h / 2) 
    
    M = cv2.getRotationMatrix2D(center, 180, 1.0)
    out_image = cv2.warpAffine(img, M, (w, h)) 
    out_image = np.expand_dims(out_image, axis=-1)
    return out_image

# Image data generator distortion options
data_gen_args = dict( horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode='reflect',
                     preprocessing_function=fixed_dregee)

# Train data, provide the same seed and keyword arguments to the fit and flow methods
X_datagen_train = ImageDataGenerator(**data_gen_args)
Y_datagen_train = ImageDataGenerator(**data_gen_args)
X_datagen_train.fit(X_train, augment=True, seed=seedValue)
Y_datagen_train.fit(Y_train, augment=True, seed=seedValue)

# Validation data, no data augmentation, but we create a generator anyway
X_datagen_val = ImageDataGenerator()
Y_datagen_val = ImageDataGenerator()
X_datagen_val.fit(X_val, augment=False, seed=seedValue)
Y_datagen_val.fit(Y_val, augment=False, seed=seedValue)

# Check a few of generated images 
i=0
for batch in X_datagen_train.flow(X_train, save_to_dir="aug_x", batch_size=batch_size_value, shuffle=True, seed=seedValue, save_prefix='x', save_format='jpeg'):
    i += 1
    if i > 5:
        break
i=0
for batch in Y_datagen_train.flow(Y_train, save_to_dir="aug_y", batch_size=batch_size_value, shuffle=True, seed=seedValue, save_prefix='y', save_format='jpeg'):
    i += 1
    if i > 5:
        break

X_train_augmented = X_datagen_train.flow(X_train, batch_size=batch_size_value, shuffle=False, seed=seedValue)
Y_train_augmented = Y_datagen_train.flow(Y_train, batch_size=batch_size_value, shuffle=False, seed=seedValue)
X_val_flow = X_datagen_val.flow(X_val, batch_size=batch_size_value, shuffle=False, seed=seedValue)
Y_val_flow = Y_datagen_val.flow(Y_val, batch_size=batch_size_value, shuffle=False, seed=seedValue)

# Combine generators into one which yields image and masks
train_generator = zip(X_train_augmented, Y_train_augmented)
val_generator = zip(X_val_flow, Y_val_flow)


##########################
#    BUILD THE NETWORK   #
##########################

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
adam= keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=[mean_iou])
model.summary()


# Fit model
earlystopper = EarlyStopping(patience=10, verbose=1)
checkpointer = ModelCheckpoint('model-v.1.0.fibsem.h5', verbose=1, save_best_only=True)

results = model.fit_generator(train_generator, validation_data=val_generator,
                    validation_steps=math.ceil(len(X_val)/batch_size_value), steps_per_epoch=math.ceil(len(X_train)/batch_size_value),
                    epochs=70, callbacks=[earlystopper, checkpointer, time_callback])

# Create some plots with the results
if len(sys.argv) > 1 and TESTID == "1":
    # Loss
    plt.plot(results.history['loss'])
    plt.plot(results.history['val_loss'])
    plt.title('Model JOBID=' + JOBID + ' loss')
    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.legend(['Train loss', 'Val. loss'], loc='upper left')
    if not os.path.exists(chartOutDir):
        os.makedirs(chartOutDir)
    plt.savefig(os.path.join(chartOutDir , str(JOBID) + '_loss.png'))
    plt.clf()

    # Mean_iou
    plt.plot(results.history['mean_iou'])
    plt.plot(results.history['val_mean_iou'])
    plt.title('Model JOBID=' + JOBID + ' mean_iou')
    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.legend(['Train mean_iou', 'Val. mean_iou'], loc='upper left') 
    plt.savefig(os.path.join(chartOutDir , str(JOBID) + '_meanIOU.png'))
    plt.clf()


#####################
#    PREDICTION     #
#####################

# Load the model saved
#model = load_model('model-v.1.0.fibsem.h5', custom_objects={'mean_iou': mean_iou})
print("Making the predictions on the new data . . .")

# Evaluate to obtain the loss and mean_iou
score = model.evaluate(X_test,Y_test, verbose=0)

# Predict on train, val and test
#preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
#preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

# Threshold predictions
#preds_train_t = (preds_train > 0.5).astype(np.uint8)
#preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

if not os.path.exists(outputDir):
    os.makedirs(outputDir)
# Save results as image
#for i in range(0,len(preds_train)):
#    im = Image.fromarray(preds_train[i,:,:,0]*255)
#    im = im.convert('RGB')
#    im.save(os.path.join(outputDir,"train_out" + str(i) + ".png"))
#
#for i in range(0,len(preds_val)):
#    im = Image.fromarray(preds_val[i,:,:,0]*255)
#    im = im.convert('RGB')
#    im.save(os.path.join(outputDir,"val_out" + str(i) + ".png"))

# Only the first test will be able to write the images
if len(sys.argv) > 1 and TESTID == "1":
    for i in range(0,len(preds_test)):
        im = Image.fromarray(preds_test[i,:,:,0]*255)
        im = im.convert('RGB')
        im.save(os.path.join(outputDir,"test_out" + str(i) + ".png"))


#####################
#      RESULTS      #
#####################

# Time
print("Epoch average time: ", np.mean(time_callback.times))
print("Train time (s):", np.sum(time_callback.times))

# Loss and metric
print("Train loss:", np.min(results.history['loss']))
print("Validation loss:", np.min(results.history['val_loss']))
print("Test loss:", score[0])
print("Train mean_iou:", np.max(results.history['mean_iou']))
print("Validation mean_iou:", np.max(results.history['val_mean_iou']))
print("Test mean_iou:", score[1])
print("Epoch number:", len(results.history['val_loss']))

# If we are running multiple tests store the results in a file to take them easy after
if len(sys.argv) > 1:

    try:
        os.remove(LOGFILE)
    except OSError:
        pass
    f = open(LOGFILE, 'x') 
    f.write(str(np.min(results.history['loss'])) + ' ' + str(np.min(results.history['val_loss'])) + ' ' + str(score[0]) \
    + ' ' + str(np.max(results.history['mean_iou'])) + ' ' + str(np.max(results.history['val_mean_iou'])) + ' ' + str(score[1]) \
    + ' ' + str(len(results.history['val_loss'])) + ' ' + str(np.mean(time_callback.times)) + ' ' + str(np.sum(time_callback.times)) + '\n')
    f.close()

    try:
        os.remove(HISTORYFILE)
    except OSError:
        pass
    f = open(HISTORYFILE, 'x')
    f.write('############## TRAIN LOSS ############## \n')
    f.write(str(results.history['loss']) + '\n')
    f.write('############## VALIDATION LOSS ############## \n')
    f.write(str(results.history['val_loss']) + '\n')
    f.write('############## TEST LOSS ############## \n')
    f.write(str(score[0]) + '\n')
    f.write('############## TRAIN MEAN_IOU ############## \n')
    f.write(str(results.history['mean_iou']) + '\n')
    f.write('############## VALIDATION MEAN_IOU ############## \n')
    f.write(str(results.history['val_mean_iou']) + '\n')
    f.write('############## TEST MEAN_IOU ############## \n')
    f.write(str(score[1]) + '\n')
     
