##########################
#        PREAMBLE        #
##########################
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                '..'))

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
from metrics import *
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
    gpu_selected = str(sys.argv[1])                                       
    job_id = str(sys.argv[2])                                             
    test_id = str(sys.argv[3])                                            
    job_file = job_id + '_' + test_id                                     
    log_dir = os.path.join(str(sys.argv[4]), job_id)                   

# Checks
print('job_id :', job_id)
print('GPU selected :', gpu_selected)
print('Python       :', sys.version.split('\n')[0])
print('Numpy        :', np.__version__)
print('Keras        :', keras.__version__)
print('Tensorflow   :', tf.__version__)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
if len(sys.argv) > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_selected;
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "3";

# Go to the experiments base dir
os.chdir("/data2/dfranco/experimentosTFM/FIBSEM_EPFL")

# Image dimensions
img_width = 1024
img_height = 768
img_channels = 1

# Dimension to obtain in the crop
img_width_crop = 256
img_height_crop = 256
img_channels_crop = 1

# Paths to data and results                                             
TRAIN_PATH = os.path.join('data', 'train', 'x')                         
TRAIN_MASK_PATH = os.path.join('data', 'train', 'y')                    
TEST_PATH = os.path.join('data', 'test', 'x')                           
TEST_MASK_PATH = os.path.join('data', 'test', 'y')                      
RESULT_DIR = os.path.join('results', 'results_' + job_id)
CHAR_DIR='charts'
H5_DIR='h5_files'

# Define time callback
time_callback = TimeHistory()

# Additional variables
batch_size_value = 6
momentum_value = 0.99
learning_rate_value = 0.001
epochs_value = 10

##########################
#       LOAD DATA        #
##########################

X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data(TRAIN_PATH,
                                             TRAIN_MASK_PATH,
                                             TEST_PATH,
                                             TEST_MASK_PATH,
                                             [img_height, img_width,
                                              img_channels])

# Crop the data to the desired size
X_train, Y_train = crop_data(X_train, Y_train, img_width_crop,
                             img_height_crop)
X_val, Y_val = crop_data(X_val, Y_val, img_width_crop, img_height_crop)
X_test, Y_test = crop_data(X_test, Y_test, img_width_crop,
                             img_height_crop)
img_width = img_width_crop
img_height = img_height_crop
img_channels = img_channels_crop


##########################
#    DATA AUGMENTATION   #
##########################

train_generator, val_generator = da_generator(X_train, Y_train, X_val,
                                              Y_val, batch_size_value, 
                                              job_id)
                                              

##########################
#    BUILD THE NETWORK   #
##########################

model = U_Net([img_height, img_width, img_channels])
sdg = keras.optimizers.SGD(lr=learning_rate_value,
                           momentum=momentum_value, decay=0.0,
                           nesterov=False)

model.compile(optimizer=sdg, loss='binary_crossentropy',
              metrics=[jaccard_index])
model.summary()

# Fit model
earlystopper = EarlyStopping(patience=epochs_value, verbose=1,
                             restore_best_weights=True)

if not os.path.exists(H5_DIR):                                      
    os.makedirs(H5_DIR)
checkpointer = ModelCheckpoint(os.path.join(H5_DIR, 'model.fibsem_'     
                                                    + job_file +'.h5'), 
                               verbose=1, save_best_only=True)

results = model.fit_generator(train_generator, 
                              validation_data=val_generator,
                              validation_steps=math.ceil(len(X_val)/batch_size_value),
                              steps_per_epoch=math.ceil(len(X_train)/batch_size_value),
                              epochs=epochs_value, 
                              callbacks=[earlystopper, checkpointer,
                                         time_callback])


#####################
#    PREDICTION     #
#####################

# Evaluate to obtain the loss and jaccard index                         
print("Evaluating test data . . .")                                     
score = model.evaluate(X_test, Y_test, batch_size=batch_size_value,     
                       verbose=1)                                       
                                                                        
# Predict on test                                                       
print("Making the predictions on test data . . .")                      
preds_test = model.predict(X_test, batch_size=batch_size_value,         
                           verbose=1) 

# Threshold predictions
preds_test_t = (preds_test > 0.5).astype(np.uint8)

# Save the resulting images 
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)
if len(sys.argv) > 1 and test_id == "1":
    print("Saving predicted images . . .")
    for i in range(0,len(preds_test)):
        im = Image.fromarray(preds_test[i,:,:,0]*255)
        im = im.convert('L')
        im.save(os.path.join(RESULT_DIR,"test_out" + str(i) + ".png"))


#####################
#  SCORES OBTAINED  #
#####################

# VOC
print("Calculating VOC . . .")
voc = voc_calculation(Y_test, preds_test_t, score[1])

# Time
print("Epoch average time: ", np.mean(time_callback.times))
print("Train time (s):", np.sum(time_callback.times))

# Loss and metric
print("Train loss:", np.min(results.history['loss']))
print("Validation loss:", np.min(results.history['val_loss']))
print("Test loss:", score[0])
print("Train jaccard_index:", np.max(results.history['jaccard_index']))
print("Validation jaccard_index:",
      np.max(results.history['val_jaccard_index']))
print("Test jaccard_index:", score[1])
print("VOC: ", voc)
print("Epoch number:", len(results.history['val_loss']))

# If we are running multiple tests store the results
if len(sys.argv) > 1:

    store_history(results, score, voc, time_callback, log_dir, job_file)

    if test_id == "1":
        create_plots(results, job_id, CHAR_DIR)

