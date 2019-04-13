##########################
#        PREAMBLE        #
##########################
import sys
sys.path.insert(0, '/data2/dfranco/experimentosTFM/FIBSEM_EPFL/scripts/')

# Limit the number of threads
from util import *
limit_threads()

# Try to generate the results as reproducible as possible
set_seed(42)


##########################
#        IMPORTS         #
##########################

from data import *
from unet import *
import random
import numpy as np
import keras
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from PIL import Image
import math


##########################
#    INITIAL VARIABLES   #
##########################

# Take arguments
if len(sys.argv) > 1:
    gpu_selected=str(sys.argv[1])
    job_id=str(sys.argv[2])
    test_id=str(sys.argv[3])
    log_file=str(sys.argv[4])
    history_file=str(sys.argv[5])

# Checks
print('job_id :', job_id)
print('GPU selected :', gpu_selected)
print('Python       :', sys.version.split('\n')[0])
print('Numpy        :', np.__version__)
print('Keras        :', keras.__version__)
print('Tensorflow   :', tf.__version__)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
if len(sys.argv) > 1:
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu_selected;
else:
    os.environ["CUDA_VISIBLE_DEVICES"]="3";

# Go to the experiments base dir
os.chdir("/data2/dfranco/experimentosTFM/FIBSEM_EPFL")

# Image dimensions
img_width = 1024
img_height = 768
img_channels = 1

# Paths to put data and results 
TRAIN_PATH = 'data/train/x/'
TRAIN_MASK_PATH = 'data/train/y/'
TEST_PATH = 'data/test/x/'
TEST_MASK_PATH = 'data/test/y/'
RESULT_DIR='results/results_' + str(job_id)
CHAR_DIR='charts'

# Define time callback
time_callback = TimeHistory()

# Additional variables
batch_size_value=1
momentum_value=0.99


##########################
#       LOAD DATA        #
##########################

X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data(TRAIN_PATH,
                                             TRAIN_MASK_PATH,
                                             TEST_PATH,
                                             TEST_MASK_PATH,
                                             [img_height, img_width,
                                             img_channels])


##########################
#    DATA AUGMENTATION   #
##########################

train_generator, val_generator = da_generator(X_train, Y_train, X_val,
                                              Y_val, batch_size_value)
                                              

##########################
#    BUILD THE NETWORK   #
##########################

model = U_Net([img_height, img_width,img_channels])
adam = keras.optimizers.SGD(lr=0.001, momentum=momentum_value, decay=0.0,
                            nesterov=False)

model.compile(optimizer=adam, loss='binary_crossentropy',
              metrics=[mean_iou])
model.summary()

# Fit model
earlystopper = EarlyStopping(patience=50, verbose=1)
checkpointer = ModelCheckpoint('model.fibsem.h5', verbose=1,
                               save_best_only=True)

results = model.fit_generator(train_generator, 
                              validation_data=val_generator,
                              validation_steps=math.ceil(len(X_val) /batch_size_value),
                              steps_per_epoch=math.ceil(len(X_train)/batch_size_value),
                              epochs=230, callbacks=[earlystopper, 
                              checkpointer, time_callback])


#####################
#    PREDICTION     #
#####################

print("Making the predictions on the new data . . .")

# Evaluate to obtain the loss and mean_iou
score = model.evaluate(X_test,Y_test, verbose=0)

# Predict on test
preds_test = model.predict(X_test, verbose=1)

# Threshold predictions
preds_test_t = (preds_test > 0.5).astype(np.uint8)

# Save the resulting images 
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)
if len(sys.argv) > 1 and test_id == "1":
    for i in range(0,len(preds_test)):
        im = Image.fromarray(preds_test[i,:,:,0]*255)
        im = im.convert('L')
        im.save(os.path.join(RESULT_DIR,"test_out" + str(i) + ".png"))


#####################
#  SCORES OBTAINED  #
#####################

#VOC
voc = voc_calculation(preds_test_t, Y_test, score[1])

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
print("VOC: ", voc)
print("Epoch number:", len(results.history['val_loss']))

# If we are running multiple tests store the results
if len(sys.argv) > 1:

    store_history(results, score, voc, time_callback, log_file, history_file)

    if test_id == "1":
        create_plots(results, job_id, CHAR_DIR)

