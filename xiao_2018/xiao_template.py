# Script based on 3D_template.py


##########################
#   ARGS COMPROBATION    #
##########################

import argparse
parser = argparse.ArgumentParser(
    description="Template based of template/template.py")
parser.add_argument("base_work_dir",
                    help="Path to code base dir , i.e ~/DeepLearning_EM")
parser.add_argument("data_dir", help="Path to data base dir")
parser.add_argument("result_dir",
                    help="Path to where the resulting output of the job will "\
                    "be stored")
parser.add_argument("-id", "--job_id", "--id", help="Job identifier",
                    default="unknown_job")
parser.add_argument("-rid","--run_id", "--rid", help="Run number of the same job",
                    type=int, default=0)
parser.add_argument("-gpu","--gpu", dest="gpu_selected",
                    help="GPU number according to 'nvidia-smi' command",
                    required=True)
args = parser.parse_args()


##########################
#        PREAMBLE        #
##########################

import os
import sys
sys.path.insert(0, args.base_work_dir)
sys.path.insert(0, os.path.join(args.base_work_dir, 'xiao_2018'))

# Working dir
os.chdir(args.base_work_dir)

# Limit the number of threads
from util import limit_threads, set_seed, create_plots, store_history,\
                 TimeHistory, threshold_plots, save_img
limit_threads()

# Try to generate the results as reproducible as possible
set_seed(42)

crops_made = False
job_identifier = args.job_id + '_' + str(args.run_id)


##########################
#        IMPORTS         #
##########################

import random
import numpy as np
import math
import time
import tensorflow as tf
from data_manipulation import load_and_prepare_3D_data, check_binary_masks, \
                              crop_3D_data_with_overlap, \
                              merge_3D_data_with_overlap
from data_3D_generators import VoxelDataGenerator
from unet_3d_xiao import U_Net_3D_Xiao
from metrics import jaccard_index, jaccard_index_numpy, voc_calculation,\
                    DET_calculation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tqdm import tqdm
from smooth_tiled_predictions import predict_img_with_smooth_windowing, \
                                     predict_img_with_overlap,\
                                     smooth_3d_predictions
from tensorflow.keras.utils import plot_model
from callbacks import ModelCheckpoint


############
#  CHECKS  #
############

print("Arguments: {}".format(args))
print("Python       : {}".format(sys.version.split('\n')[0]))
print("Numpy        : {}".format(np.__version__))
print("Keras        : {}".format(tf.keras.__version__))
print("Tensorflow   : {}".format(tf.__version__))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_selected;


##########################                                                      
#  EXPERIMENT VARIABLES  #
##########################

### Dataset variables
# Main dataset data/mask paths
train_path = os.path.join(args.data_dir, 'train', 'x')
train_mask_path = os.path.join(args.data_dir, 'train', 'y')
test_path = os.path.join(args.data_dir, 'test', 'x')
test_mask_path = os.path.join(args.data_dir, 'test', 'y')
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
# Train shape of the 3D subvolumes
train_3d_desired_shape = (20, 256, 256, 1)
# Train shape of the 3D subvolumes
test_3d_desired_shape = (20, 448, 576, 1)
# Flag to use all the images to create the 3D subvolumes. If it is False random
# subvolumes from the whole data will be generated instead.
use_all_volume = True
# Percentage of overlap made to create subvolumes of the defined shape based on
# test data. Fix in 0.0 to calculate the minimun overlap needed to satisfy the
# shape.
ov_test = 0.5


### Normalization
# Flag to normalize the data dividing by the mean pixel value
normalize_data = False                                                          
# Force the normalization value to the given number instead of the mean pixel 
# value
norm_value_forced = -1                                                          


### Data augmentation (DA) variables
# Flag to activate DA
da = True
# Create samples of the DA made. Useful to check the output images made.
aug_examples = True
# Flag to shuffle the training data on every epoch 
shuffle_train_data_each_epoch = True
# Flag to shuffle the validation data on every epoch
shuffle_val_data_each_epoch = False
# Shift range to appply to the subvolumes 
shift_range = 0.0
# Range of rotation to the subvolumes
rotation_range = 0
# Square rotations (90º, -90º and 180º) intead of using a range
square_rotations = True
# Flag to make flips on the subvolumes 
flips = True
# Flag to extract random subvolumnes during the DA. Not compatible with 
# 'use_all_volume' as it forces the data preparation into subvolumes
random_subvolumes_in_DA = False


### Extra train data generation
# Number of times to duplicate the train data. Useful when "random_crops_in_DA"
# is made, as more original train data can be cover
duplicate_train = 75
# Extra number of images to add to the train data. Applied after duplicate_train
extra_train_data = 0


### Load previously generated model weigths
# Flag to activate the load of a previous training weigths instead of train 
# the network again
load_previous_weights = False
# ID of the previous experiment to load the weigths from 
previous_job_weights = args.job_id
# Flag to activate the fine tunning
fine_tunning = False
# ID of the previous weigths to load the weigths from to make the fine tunning 
fine_tunning_weigths = args.job_id
# Prefix of the files where the weights are stored/loaded from
weight_files_prefix = 'model.fibsem_'
# Name of the folder where weights files will be stored/loaded from. This folder 
# must be located inside the directory pointed by "args.base_work_dir" variable. 
# If there is no such directory, it will be created for the first time
h5_dir = os.path.join(args.result_dir, 'h5_files')


### Experiment main parameters
# Batch size value
batch_size_value = 2
# Optimizer to use. Possible values: "sgd" or "adam"
optimizer = "adam"
# Learning rate used by the optimization method
learning_rate_value = 0.0001
# Number of epochs to train the network
epochs_value = 30
# Number of epochs to stop the training process after no improvement
patience = 10 


### Network architecture specific parameters
# Number of channels in the first initial layer of the network
num_init_channels = 32
# Flag to activate the Spatial Dropout instead of use the "normal" dropout layer
spatial_dropout = False
# Fixed value to make the dropout. Ignored if the value is zero
fixed_dropout_value = 0.0 
# Active flag if softmax is used as the last layer of the network
softmax_out = True


### Post-processing
# Flag to activate the post-processing (Smoooth and Z-filtering)
post_process = True


### Paths of the results                                             
# Directory where predicted images of the segmentation will be stored
result_dir = os.path.join(args.result_dir, 'results', job_identifier)
# Directory where binarized predicted images will be stored
result_bin_dir = os.path.join(result_dir, 'binarized')
# Directory where predicted images will be stored
result_no_bin_dir = os.path.join(result_dir, 'no_binarized')
# Folder where the smoothed images will be stored
smooth_dir = os.path.join(result_dir, 'smooth')
# Folder where the smoothed images (no binarized) will be stored
smooth_no_bin_dir = os.path.join(result_dir, 'smooth_no_bin')
# Name of the folder where the charts of the loss and metrics values while 
# training the network will be shown. This folder will be created under the
# folder pointed by "args.base_work_dir" variable 
char_dir = os.path.join(result_dir, 'charts')
# Folder where smaples of DA will be stored
da_samples_dir = os.path.join(result_dir, 'aug')


### Callbacks
# To measure the time
time_callback = TimeHistory()
# Stop early and restore the best model weights when finished the training
earlystopper = EarlyStopping(
    patience=patience, verbose=1, restore_best_weights=True)
# Save the best model into a h5 file in case one need again the weights learned
os.makedirs(h5_dir, exist_ok=True)
checkpointer = ModelCheckpoint(
    os.path.join(h5_dir, weight_files_prefix + job_identifier + '.h5'),
    verbose=1, save_best_only=True)


#####################
#   SANITY CHECKS   #
#####################

print("#####################\n#   SANITY CHECKS   #\n#####################")

check_binary_masks(train_mask_path)
check_binary_masks(test_mask_path)


##########################                                                      
#       LOAD DATA        #                                                      
##########################

print("##################\n#    LOAD DATA   #\n##################\n")

X_train, Y_train, X_val,\
Y_val, X_test, Y_test,\
orig_test_shape, norm_value = load_and_prepare_3D_data(
    train_path, train_mask_path, test_path, test_mask_path, img_train_shape,
    img_test_shape, val_split=perc_used_as_val, create_val=True,
    shuffle_val=False, train_subvol_shape=train_3d_desired_shape,
    test_subvol_shape=test_3d_desired_shape, ov_test=ov_test)

# Normalize the data
if normalize_data == True:
    if norm_value_forced != -1: 
        print("Forced normalization value to {}".format(norm_value_forced))
        norm_value = norm_value_forced
    else:
        print("Normalization value calculated: {}".format(norm_value))
    X_train -= int(norm_value)
    X_val -= int(norm_value)
    X_test -= int(norm_value)
    

#############################
#   EXTRA DATA GENERATION   #
#############################

# Duplicate train data N times
if duplicate_train != 0:
    print("##################\n# DUPLICATE DATA #\n##################\n")

    X_train = np.vstack([X_train]*duplicate_train)
    Y_train = np.vstack([Y_train]*duplicate_train)
    print("Train data replicated {} times. Its new shape is: {}"
          .format(duplicate_train, X_train.shape))

# Add extra train data generated with DA
if extra_train_data != 0:
    print("##################\n#   EXTRA DATA   #\n##################\n")

    extra_generator = VoxelDataGenerator(
        X_train, Y_train, random_subvolumes_in_DA=random_subvolumes_in_DA,
        shuffle_each_epoch=True, batch_size=batch_size_value,
        da=True, flip=True, shift_range=0.0, rotation_range=0)

    extra_x, extra_y = extra_generator.get_transformed_samples(extra_train_data)

    X_train = np.vstack((X_train, extra_x*255))
    Y_train = np.vstack((Y_train, extra_y*255))
    print("{} extra train data generated, the new shape of the train now is {}"\
          .format(extra_train_data, X_train.shape))


##########################
#    DATA AUGMENTATION   #
##########################

print("##################\n#    DATA AUG    #\n##################\n")

print("Preparing validation data generator . . .")
val_generator = VoxelDataGenerator(
    X_val, Y_val, random_subvolumes_in_DA=random_subvolumes_in_DA,
    shuffle_each_epoch=shuffle_val_data_each_epoch, batch_size=batch_size_value,
    da=False, softmax_out=softmax_out)
del X_val, Y_val

print("Preparing train data generator . . .")
train_generator = VoxelDataGenerator(
    X_train, Y_train, random_subvolumes_in_DA=random_subvolumes_in_DA,
    shuffle_each_epoch=shuffle_train_data_each_epoch, 
    batch_size=batch_size_value, da=da, flip=flips, shift_range=shift_range, 
    rotation_range=rotation_range, square_rotations=square_rotations,
    softmax_out=softmax_out)
del X_train, Y_train

# Create the test data generator without DA
print("Preparing test data generator . . .")
test_generator = VoxelDataGenerator(
    X_test, Y_test, random_subvolumes_in_DA=False, shuffle_each_epoch=False,
    batch_size=batch_size_value, da=False, softmax_out=softmax_out)

# Generate examples of data augmentation
if aug_examples == True:
    train_generator.get_transformed_samples(
        10, random_images=True, save_to_dir=True, out_dir=da_samples_dir)


##########################
#    BUILD THE NETWORK   #
##########################

print("###################\n#  TRAIN PROCESS  #\n###################\n")

print("Creating the network . . .")
model = U_Net_3D_Xiao(train_3d_desired_shape, lr=learning_rate_value)

# Check the network created
model.summary(line_length=150)
os.makedirs(char_dir, exist_ok=True)
model_name = os.path.join(char_dir, "model_plot_" + job_identifier + ".png")
#plot_model(model, to_file=model_name, show_shapes=True, show_layer_names=True)

if load_previous_weights == False:
    if fine_tunning == True:                                                    
        h5_file=os.path.join(h5_dir, weight_files_prefix + fine_tunning_weigths 
                             + '_' + str(args.run_id) + '.h5')     
        print("Fine-tunning: loading model weights from h5_file: {}"
              .format(h5_file))   
        model.load_weights(h5_file)                                             

    results = model.fit(x=train_generator, validation_data=val_generator,
        validation_steps=len(val_generator), steps_per_epoch=len(train_generator),
        epochs=epochs_value, callbacks=[earlystopper, checkpointer, time_callback])
else:
    h5_file=os.path.join(h5_dir, weight_files_prefix + previous_job_weights 
                                 + '_' + str(args.run_id) + '.h5')
    print("Loading model weights from h5_file: {}".format(h5_file))
    model.load_weights(h5_file)


#####################
#     INFERENCE     #
#####################

print("##################\n#    INFERENCE   #\n##################\n")

# Evaluate to obtain the loss value and the Jaccard index
print("Evaluating test data . . .")
score = model.evaluate_generator(test_generator, verbose=1)
jac_per_subvolume = score[1]

# Predict on test
print("Making the predictions on test data . . .")
preds_test = model.predict_generator(test_generator, verbose=1)

# Divide the test data into 255 if it is going to be used
Y_test /= 255 if np.max(Y_test) > 2 else Y_test
X_test /= 255 if np.max(X_test) > 2 else X_test

if softmax_out == True:
    preds_test = np.expand_dims(preds_test[...,1], -1)

# Merge the volumes and convert them into 2D data
recons_pred_test, Y_test = merge_3D_data_with_overlap(
    preds_test, orig_test_shape, data_mask=Y_test, overlap_z=ov_test)

print("Saving predicted images . . .")
save_img(Y=(recons_pred_test > 0.5).astype(np.uint8), mask_dir=result_bin_dir,
         prefix="test_out_bin")
save_img(Y=recons_pred_test, mask_dir=result_no_bin_dir, prefix="test_out_no_bin")

print("Calculate metrics . . .")
score[1] = jaccard_index_numpy(Y_test, (recons_pred_test > 0.5).astype(np.uint8))
voc = voc_calculation(Y_test, (recons_pred_test > 0.5).astype(np.uint8), score[1])
#det = DET_calculation(Y_test, bin_preds_test, det_eval_ge_path,
#                      det_eval_path, det_bin, n_dig, job_identifier)
det = -1

    
####################
#  POST-PROCESING  #
####################

if post_process == True:
    print("##################\n# POST-PROCESING #\n##################\n")
    print("1) SMOOTH")

    Y_test_smooth = np.zeros(X_test.shape, dtype=(np.float32))

    for i in tqdm(range(X_test.shape[0])):
        predictions_smooth = smooth_3d_predictions(X_test[i],
            pred_func=(lambda img_batch_subdiv: \
                           model.predict(img_batch_subdiv)))

        Y_test_smooth[i] = predictions_smooth

    # Merge the volumes and convert them into 2D data
    Y_test_smooth = merge_3D_data_with_overlap(
        Y_test_smooth, orig_test_shape, overlap_z=ov_test)

    print("Saving smooth predicted images . . .")
    save_img(Y=Y_test_smooth, mask_dir=smooth_no_bin_dir,
             prefix="test_out_smooth_no_bin")
    save_img(Y=(Y_test_smooth > 0.5).astype(np.uint8), mask_dir=smooth_dir,
             prefix="test_out_smooth")

    # Metrics (Jaccard + VOC + DET)
    print("Calculate metrics . . .")
    smooth_score = jaccard_index_numpy(
        Y_test, (Y_test_smooth > 0.5).astype(np.uint8))
    smooth_voc = voc_calculation(
        Y_test, (Y_test_smooth > 0.5).astype(np.uint8), smooth_score)

    print("Finish post-processing")


####################################
#  PRINT AND SAVE SCORES OBTAINED  #
####################################

if load_previous_weights == False:
    print("Epoch average time: {}".format(np.mean(time_callback.times)))
    print("Epoch number: {}".format(len(results.history['val_loss'])))
    print("Train time (s): {}".format(np.sum(time_callback.times)))
    print("Train loss: {}".format(np.min(results.history['loss'])))
    print("Train jaccard_index: {}"
          .format(np.max(results.history['jaccard_index_softmax'])))
    print("Validation loss: {}".format(np.min(results.history['val_loss'])))
    print("Validation jaccard_index: {}"
          .format(np.max(results.history['val_jaccard_index_softmax'])))

print("Test loss: {}".format(score[0]))
print("Test jaccard_index (per subvolume): {}".format(jac_per_subvolume))
print("Test jaccard_index (per image): {}".format(score[1]))
print("VOC (per image without overlap): {}".format(voc))
#print("DET (per image without overlap): {}".format(det))
#print("DET (per image with 50% overlap): {}".format(det_per_img_50ov))
    
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
    jac_per_img_50ov = -1 if 'jac_per_img_50ov' not in globals() else jac_per_img_50ov
    voc_per_img_50ov = -1 if 'voc_per_img_50ov' not in globals() else voc_per_img_50ov
    det_per_img_50ov = -1 if 'det_per_img_50ov' not in globals() else det_per_img_50ov

    store_history(
        results, jac_per_subvolume, score, jac_per_img_50ov, voc, 
        voc_per_img_50ov, det, det_per_img_50ov, time_callback, args.result_dir,
        job_identifier, smooth_score, smooth_voc, smooth_det, zfil_score, 
        zfil_voc, zfil_det, smo_zfil_score, smo_zfil_voc, smo_zfil_det,
        metric="jaccard_index_softmax")

    create_plots(results, job_identifier, char_dir, 
                 metric="jaccard_index_softmax")

if post_process == True:
    print("Post-process: SMOOTH - Test jaccard_index: {}".format(smooth_score))
    print("Post-process: SMOOTH - VOC: {}".format(smooth_voc))

print("FINISHED JOB {} !!".format(job_identifier))
