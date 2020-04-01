##########################
#        PREAMBLE        #
##########################

import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '..'))

# Limit the number of threads
from util import limit_threads, set_seed, create_plots, store_history,\
                 TimeHistory, Print, threshold_plots, save_img
limit_threads()

# Try to generate the results as reproducible as possible
set_seed(42)


##########################
#        IMPORTS         #
##########################

import random
import numpy as np
import keras
import math
import time
import tensorflow as tf
from data_manipulation import load_data, crop_data, merge_data_without_overlap,\
                              check_crops, crop_data_with_overlap,\
                              merge_data_with_overlap, check_binary_masks
from data_3D_generators import VoxelDataGenerator
from unet_3d import U_Net_3D
from metrics import jaccard_index, jaccard_index_numpy, voc_calculation,\
                    DET_calculation
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.morphology import label
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from PIL import Image
from tqdm import tqdm
from smooth_tiled_predictions import predict_img_with_smooth_windowing, \
                                     predict_img_with_overlap
from skimage.segmentation import clear_border


##########################
#   ARGS COMPROBATION    #
##########################

# Take arguments
gpu_selected = str(sys.argv[1])                                       
job_id = str(sys.argv[2])                                             
test_id = str(sys.argv[3])                                            
job_file = job_id + '_' + test_id                                     
base_work_dir = str(sys.argv[4])
log_dir = os.path.join(base_work_dir, 'logs', job_id)

# Checks
Print('job_id : ' + job_id)
Print('GPU selected : ' + gpu_selected)
Print('Python       : ' + sys.version.split('\n')[0])
Print('Numpy        : ' + np.__version__)
Print('Keras        : ' + keras.__version__)
Print('Tensorflow   : ' + tf.__version__)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_selected;

# Control variables 
crops_made = False

# Working dir
os.chdir(base_work_dir)


##########################                                                      
#  EXPERIMENT VARIABLES  #
##########################

### Dataset variables
# Main dataset data/mask paths
data_base_path = 'data'
train_path = os.path.join(data_base_path, 'train', 'x')
train_mask_path = os.path.join(data_base_path, 'train', 'y')
test_path = os.path.join(data_base_path, 'test', 'x')
test_mask_path = os.path.join(data_base_path, 'test', 'y')
# Percentage of the training data used as validation                            
perc_used_as_val = 0.1


### Dataset shape
# Note: train and test dimensions must be the same when training the network and
# making the predictions. Be sure to take care of this if you are not going to
# use "crop_data()" with the arg force_shape, as this function resolves the 
# problem creating always crops of the same dimension
img_train_shape = (1024, 768, 1)
img_test_shape = (1024, 768, 1)


### 3D volume variables
# Shape of the 3D subvolumes 
img_3d_desired_shape = (80, 256, 256, 1)
# Flag to use all the images to create the 3D subvolumes. If it is False random
# subvolumes from the whole data will be generated instead.
use_all_volume = True


### Normalization
# Flag to normalize the data dividing by the mean pixel value
normalize_data = False                                                          
# Force the normalization value to the given number instead of the mean pixel 
# value
norm_value_forced = -1                                                          


### Data augmentation (DA) variables
# Flag to activate DA
da = False
# Flag to shuffle the training data on every epoch 
shuffle_train_data_each_epoch = False
# Flag to shuffle the validation data on every epoch
shuffle_val_data_each_epoch = False
# Shift range to appply to the subvolumes 
shift_range = 0.0
# Range of rotation to the subvolumes
rotation_range = 0
# Flag to make flips on the subvolumes 
flips = False
# Flag to extract random subvolumnes during the DA. Not compatible with 
# 'use_all_volume' as it forces the data preparation into subvolumes
random_subvolumes_in_DA = False


### Load previously generated model weigths
# Flag to activate the load of a previous training weigths instead of train 
# the network again
load_previous_weights = True
# ID of the previous experiment to load the weigths from 
previous_job_weights = job_id
# Flag to activate the fine tunning
fine_tunning = False
# ID of the previous weigths to load the weigths from to make the fine tunning 
fine_tunning_weigths = "232"
# Prefix of the files where the weights are stored/loaded from
weight_files_prefix = 'model.fibsem_'
# Name of the folder where weights files will be stored/loaded from. This folder 
# must be located inside the directory pointed by "base_work_dir" variable. If
# there is no such directory, it will be created for the first time
h5_dir = 'h5_files'


### Experiment main parameters
# Batch size value
batch_size_value = 1
# Optimizer to use. Posible values: "sgd" or "adam"
optimizer = "sgd"
# Learning rate used by the optimization method
learning_rate_value = 0.001
# Number of epochs to train the network
epochs_value = 360
# Number of epochs to stop the training process after no improvement
patience = 50 
# Flag to activate the creation of a chart showing the loss and metrics fixing 
# different binarization threshold values, from 0.1 to 1. Useful to check a 
# correct threshold value (normally 0.5)
make_threshold_plots = False
# Define time callback                                                          
time_callback = TimeHistory()


### Network architecture specific parameters
# Number of channels in the first initial layer of the network
num_init_channels = 16
# Flag to activate the Spatial Dropout instead of use the "normal" dropout layer
spatial_dropout = False
# Fixed value to make the dropout. Ignored if the value is zero
fixed_dropout_value = 0.0 


### Post-processing
# Flag to activate the post-processing (Smoooth and Z-filtering)
post_process = True


### Paths of the results                                             
# Directory where predicted images of the segmentation will be stored
result_dir = os.path.join('results', 'results_' + job_id, job_file)
# Directory where binarized predicted images will be stored
result_bin_dir = os.path.join(result_dir, 'binarized')
# Directory where predicted images will be stored
result_no_bin_dir = os.path.join(result_dir, 'no_binarized')
# Directory where binarized predicted images with 50% of overlap will be stored
result_bin_dir_50ov = os.path.join(result_dir, 'binarized_50ov')
# Folder where the smoothed images will be stored
smooth_dir = os.path.join(result_dir, 'smooth')
# Folder where the images with the z-filter applied will be stored
zfil_dir = os.path.join(result_dir, 'zfil')
# Folder where the images with smoothing and z-filter applied will be stored
smoo_zfil_dir = os.path.join(result_dir, 'smoo_zfil')
# Name of the folder where the charts of the loss and metrics values while 
# training the network will be shown. This folder will be created under the
# folder pointed by "base_work_dir" variable 
char_dir = 'charts'


#####################
#   SANITY CHECKS   #
#####################

Print("#####################\n#   SANITY CHECKS   #\n#####################")

check_binary_masks(train_mask_path)
check_binary_masks(test_mask_path)


##########################                                                      
#       LOAD DATA        #                                                      
##########################

Print("##################\n#    LOAD DATA   #\n##################\n")

X_train, Y_train, \
X_val, Y_val, \
X_test, Y_test, \
norm_value, _ = load_data(train_path, train_mask_path, test_path,
                          test_mask_path, img_train_shape, img_test_shape,
                          val_split=perc_used_as_val, shuffle_val=False,
                          make_crops=False, prepare_subvolumes=use_all_volume, 
                          subvol_shape=img_3d_desired_shape)

# Normalize the data
if normalize_data == True:
    if norm_value_forced != -1: 
        Print("Forced normalization value to " + str(norm_value_forced))
        norm_value = norm_value_forced
    else:
        Print("Normalization value calculated: " + str(norm_value))
    X_train -= int(norm_value)
    X_val -= int(norm_value)
    X_test -= int(norm_value)
    

##########################
#    DATA AUGMENTATION   #
##########################

Print("##################\n#    DATA AUG    #\n##################\n")

train_generator = VoxelDataGenerator(X_train, Y_train,                                          
                                     random_subvolumes_in_DA=random_subvolumes_in_DA,           
                                     shuffle_each_epoch=shuffle_train_data_each_epoch,          
                                     batch_size=batch_size_value, da=da, 
                                     flip=flips, shift_range=shift_range, 
                                     rotation_range=rotation_range)

val_generator = VoxelDataGenerator(X_val, Y_val, random_subvolumes_in_DA=False,           
                                   shuffle_each_epoch=shuffle_val_data_each_epoch,        
                                   batch_size=batch_size_value, da=False)  
                                                                                

##########################
#    BUILD THE NETWORK   #
##########################

Print("###################\n#  TRAIN PROCESS  #\n###################\n")

Print("Creating the network . . .")
model = U_Net_3D(img_3d_desired_shape, numInitChannels=num_init_channels, 
                 spatial_dropout=spatial_dropout,
                 fixed_dropout=fixed_dropout_value,
                 optimizer=optimizer, lr=learning_rate_value)

model.summary()

if load_previous_weights == False:
    earlystopper = EarlyStopping(patience=patience, verbose=1, 
                                 restore_best_weights=True)
    
    if not os.path.exists(h5_dir):                                      
        os.makedirs(h5_dir)
    checkpointer = ModelCheckpoint(os.path.join(h5_dir, weight_files_prefix + job_file + '.h5'),
                                   verbose=1, save_best_only=True)
    
    if fine_tunning == True:                                                    
        h5_file=os.path.join(h5_dir, weight_files_prefix + fine_tunning_weigths 
                                     + '_' + test_id + '.h5')     
        Print("Fine-tunning: loading model weights from h5_file: " + h5_file)   
        model.load_weights(h5_file)                                             
   
    results = model.fit_generator(train_generator, validation_data=val_generator,
                                  validation_steps=math.ceil(len(X_val)/batch_size_value),
                                  steps_per_epoch=math.ceil(len(X_train)/batch_size_value),
                                  epochs=epochs_value, 
                                  callbacks=[earlystopper, checkpointer, time_callback])
else:
    h5_file=os.path.join(h5_dir, weight_files_prefix + previous_job_weights 
                                 + '_' + test_id + '.h5')
    Print("Loading model weights from h5_file: " + h5_file)
    model.load_weights(h5_file)


#####################
#     INFERENCE     #
#####################

Print("##################\n#    INFERENCE   #\n##################\n")

# Evaluate to obtain the loss value and the Jaccard index
Print("Evaluating test data . . .")
score = model.evaluate(X_test, Y_test, batch_size=batch_size_value, verbose=1)
jac_per_subvolume = score[1]

# Predict on test
Print("Making the predictions on test data . . .")
preds_test = model.predict(X_test, batch_size=batch_size_value, verbose=1)

# Threshold images
bin_preds_test = (preds_test > 0.5).astype(np.uint8)

Print("Saving predicted images . . .")
#reconstruct the images 
#save_img(Y=bin_preds_test, mask_dir=result_bin_dir, prefix="test_out_bin")
#save_img(Y=preds_test, mask_dir=result_no_bin_dir, prefix="test_out_no_bin")

Print("Calculate metrics . . .")
# Per image without overlap
score[1] = jaccard_index_numpy(Y_test, bin_preds_test)
voc = voc_calculation(Y_test, bin_preds_test, score[1])
#det = DET_calculation(Y_test, bin_preds_test, det_eval_ge_path,
#                      det_eval_path, det_bin, n_dig, job_id)
det = -1

# 50% overlap
jac_per_img_50ov = -1
voc_per_img_50ov = -1
det_per_img_50ov = -1

    
####################
#  POST-PROCESING  #
####################

Print("##################\n# POST-PROCESING #\n##################\n") 

Print("1) SMOOTH")
# not implemented
Print("2) Z-FILTERING")
# not implemented
Print("Finish post-processing") 


####################################
#  PRINT AND SAVE SCORES OBTAINED  #
####################################

if load_previous_weights == False:
    Print("Epoch average time: " + str(np.mean(time_callback.times)))
    Print("Epoch number: " + str(len(results.history['val_loss'])))
    Print("Train time (s): " + str(np.sum(time_callback.times)))
    Print("Train loss: " + str(np.min(results.history['loss'])))
    Print("Train jaccard_index: " + str(np.max(results.history['jaccard_index'])))
    Print("Validation loss: " + str(np.min(results.history['val_loss'])))
    Print("Validation jaccard_index: " + str(np.max(results.history['val_jaccard_index'])))

Print("Test loss: " + str(score[0]))
Print("Test jaccard_index (per subvolume): " + str(jac_per_subvolume))
Print("Test jaccard_index (per image without overlap): " + str(score[1]))
Print("Test jaccard_index (per image with 50% overlap): " + str(jac_per_img_50ov))
Print("VOC (per image without overlap): " + str(voc))
Print("VOC (per image with 50% overlap): " + str(voc_per_img_50ov))
Print("DET (per image without overlap): " + str(det))
Print("DET (per image with 50% overlap): " + str(det_per_img_50ov))
    
if load_previous_weights == False:
    smooth_score = -1 if 'smooth_score' not in globals() else smooth_score
    smooth_voc = -1 if 'smooth_voc' not in globals() else smooth_voc
    smooth_det = -1 if 'smooth_det' not in globals() else smooth_det
    zfil_score = -1 if 'zfil_score' not in globals() else zfil_score
    zfil_voc = -1 if 'zfil_voc' not in globals() else zfil_voc
    zfil_det = -1 if 'zfil_det' not in globals() else zfil_det
    smo_zfil_score = -1 if 'smo_zfil_score' not in globals() else smo_zfil_score
    smo_zfil_voc = -1 if 'smo_zfil_voc' not in globals() else smo_zfil_voc
    smo_zfil_det = -1 if 'smo_zfil_det' not in globals() else smo_zfil_det
    jac_per_subvolume = -1 if 'jac_per_subvolume' not in globals() else jac_per_subvolume

    store_history(results, jac_per_subvolume, score, jac_per_img_50ov, voc, 
                  voc_per_img_50ov, det, det_per_img_50ov, time_callback, log_dir,
                  job_file, smooth_score, smooth_voc, smooth_det, zfil_score,
                  zfil_voc, zfil_det, smo_zfil_score, smo_zfil_voc, smo_zfil_det)

    create_plots(results, job_id, test_id, char_dir)

Print("FINISHED JOB " + job_file + " !!")
