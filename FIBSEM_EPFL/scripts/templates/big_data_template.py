##########################
#        PREAMBLE        #
##########################

import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '..', 'code'))

# Limit the number of threads
from util import limit_threads, set_seed, create_plots, store_history,\
                 TimeHistory, threshold_plots, save_img
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
                              check_crops, crop_data_with_overlap, \
                              merge_data_with_overlap, check_binary_masks
from data_generators import keras_da_generator, ImageDataGenerator,\
                            keras_gen_samples, calculate_z_filtering,\
                            combine_generators
from unet import U_Net
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
                                     predict_img_with_overlap,\
                                     predict_img_with_overlap_weighted
from skimage.segmentation import clear_border
from keras.utils.vis_utils import plot_model


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
print("job_id : {}".format(job_id))
print("GPU selected : {}".format(gpu_selected))
print("Python       : {}".format(sys.version.split('\n')[0]))
print("Numpy        : {}".format(np.__version__))
print("Keras        : {}".format(keras.__version__))
print("Tensorflow   : {}".format(tf.__version__))
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
data_base_path = os.path.join('harvard_datasets', 'human_v2')
train_path = os.path.join(data_base_path, 'train', 'x')
train_mask_path = os.path.join(data_base_path, 'train', 'y')
val_path = os.path.join(data_base_path, 'val', 'x')
val_mask_path = os.path.join(data_base_path, 'val', 'y')
test_path = os.path.join(data_base_path, 'test', 'x')
test_mask_path = os.path.join(data_base_path, 'test', 'y')
complete_test_path = os.path.join('harvard_datasets', 'human_v2', 'test', 'x')
complete_test_mask_path = os.path.join('harvard_datasets', 'human_v2', 'test', 'y')


### Dataset shape
# Note: train and test dimensions must be the same when training the network and
# making the predictions. Be sure to take care of this if you are not going to
# use "crop_data()" with the arg force_shape, as this function resolves the
# problem creating always crops of the same dimension
img_train_shape =  [256, 256, 1] 
img_test_shape = [256, 256, 1]
original_test_shape = [4096, 4096, 1]


### Big data variables
data_paths = []
data_paths.append(train_path)
data_paths.append(train_mask_path)
data_paths.append(val_path)
data_paths.append(val_mask_path)
data_paths.append(test_path)
data_paths.append(test_mask_path)
data_paths.append(complete_test_path)
data_paths.append(complete_test_mask_path)


### Data augmentation (DA) variables
# Flag to decide which type of DA implementation will be used. Select False to
# use Keras API provided DA, otherwise, a custom implementation will be used
custom_da = False
# Create samples of the DA made. Useful to check the output images made.
# This option is available for both Keras and custom DA
aug_examples = True
# Flag to shuffle the training data on every epoch
#(Best options: Keras->False, Custom->True)
shuffle_train_data_each_epoch = custom_da
# Flag to shuffle the validation data on every epoch
# (Best option: False in both cases)
shuffle_val_data_each_epoch = False
# Make a bit of zoom in the images. Only available in Keras DA
keras_zoom = False
# width_shift_range (more details in Keras ImageDataGenerator class). Only
# available in Keras DA
w_shift_r = 0.0
# height_shift_range (more details in Keras ImageDataGenerator class). Only
# available in Keras DA
h_shift_r = 0.0
# shear_range (more details in Keras ImageDataGenerator class). Only
# available in Keras DA
shear_range = 0.0
# Range to pick a brightness value from to apply in the images. Available for
# both Keras and custom DA. Example of use: brightness_range = [1.0, 1.0]
brightness_range = None
# Range to pick a median filter size value from to apply in the images. Option
# only available in custom DA
median_filter_size = [0, 0]


### Load previously generated model weigths
# Flag to activate the load of a previous training weigths instead of train
# the network again
load_previous_weights = False
# ID of the previous experiment to load the weigths from
previous_job_weights = job_id
# Flag to activate the fine tunning
fine_tunning = False
# ID of the previous weigths to load the weigths from to make the fine tunning
fine_tunning_weigths = job_id
# Prefix of the files where the weights are stored/loaded from
weight_files_prefix = 'model.c_human_'
# Name of the folder where weights files will be stored/loaded from. This folder
# must be located inside the directory pointed by "base_work_dir" variable. If
# there is no such directory, it will be created for the first time
h5_dir = 'h5_files'


### Experiment main parameters
# Loss type, three options: "bce", "w_bce" or "w_bce_dice", which refers to 
# binary cross entropy (BCE), weighted BCE (based on a weight map) and 
# BCE and Dice with with a weight term on each one (that must sum 1) to 
# calculate the total loss value.
loss_type = "bce"
# Batch size value
batch_size_value = 6
# Optimizer to use. Possible values: "sgd" or "adam"
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
# If weights on data are going to be applied. To true when loss_type is 'w_bce'
weights_on_data = True if loss_type == "w_bce" else False


### Network architecture specific parameters
# Number of channels in the first initial layer of the network
num_init_channels = 32 
# Flag to activate the Spatial Dropout instead of use the "normal" dropout layer
spatial_dropout = False
# Fixed value to make the dropout. Ignored if the value is zero
fixed_dropout_value = 0.0 


### Post-processing
# Flag to activate the post-processing (Smoooth and Z-filtering)
post_process = True


### DET metric variables
# More info of the metric at http://celltrackingchallenge.net/evaluation-methodology/ 
# and https://public.celltrackingchallenge.net/documents/Evaluation%20software.pdf
# NEEDED CODE REFACTORING OF THIS VARIABLE
det_eval_ge_path = os.path.join('cell_challenge_eval', 'gen_' + job_file)
# Path where the evaluation of the metric will be done
det_eval_path = os.path.join('cell_challenge_eval', job_id, job_file)
# Path where the evaluation of the metric for the post processing methods will 
# be done
det_eval_post_path = os.path.join('cell_challenge_eval', job_id, job_file + '_s')
# Path were the binaries of the DET metric is stored
det_bin = os.path.join(script_dir, '..', 'cell_cha_eval' ,'Linux', 'DETMeasure')
# Number of digits used for encoding temporal indices of the DET metric
n_dig = "3"


### Paths of the results                                             
# Directory where predicted images of the segmentation will be stored
result_dir = os.path.join('results', 'results_' + job_id, job_file)
# Directory where binarized predicted images will be stored
result_bin_dir = os.path.join(result_dir, 'binarized')
# Directory where predicted images will be stored
result_no_bin_dir = os.path.join(result_dir, 'no_binarized')
# Directory where binarized predicted images with 50% of overlap will be stored
result_bin_dir_50ov = os.path.join(result_dir, 'binarized_50ov')
# Directory where predicted images with 50% of overlap will be stored
result_no_bin_dir_50ov = os.path.join(result_dir, 'no_binarized_50ov')
# Folder where the smoothed images will be stored
smooth_dir = os.path.join(result_dir, 'smooth')
# Folder where the images with the z-filter applied will be stored
zfil_dir = os.path.join(result_dir, 'zfil')
# Folder where the images with smoothing and z-filter applied will be stored
smoo_zfil_dir = os.path.join(result_dir, 'smoo_zfil')
# Name of the folder where the charts of the loss and metrics values while 
# training the network will be shown. This folder will be created under the
# folder pointed by "base_work_dir" variable 
char_dir = os.path.join('charts', job_id)
# Directory where weight maps will be stored
loss_weight_dir = os.path.join(base_work_dir, 'loss_weights', job_id)


#####################
#   SANITY CHECKS   #
#####################

print("#####################\n#   SANITY CHECKS   #\n#####################")

check_binary_masks(os.path.join(train_mask_path, 'y'))
check_binary_masks(os.path.join(val_mask_path, 'y'))
check_binary_masks(os.path.join(test_mask_path, 'y'))


##########################
#    DATA AUGMENTATION   #
##########################

print("##################\n#    DATA AUG    #\n##################\n")

if custom_da == False:                                                          
    print("Keras DA selected")

    # Keras Data Augmentation
    train_generator, val_generator, \
    X_test_augmented, Y_test_augmented, \
    W_test_augmented, X_complete_aug, \
    Y_complete_aug, W_complete_aug, \
    n_train_samples, n_val_samples, \
    n_test_samples  = keras_da_generator(                                       
        ld_img_from_disk=True, data_paths=data_paths,                           
        target_size=(img_train_shape[0], img_train_shape[1]),                   
        c_target_size=(original_test_shape[0], original_test_shape[1]),         
        batch_size_value=batch_size_value, save_examples=aug_examples,          
        job_id=job_id, shuffle_train=shuffle_train_data_each_epoch,             
        shuffle_val=shuffle_val_data_each_epoch, zoom=keras_zoom,               
        w_shift_r=w_shift_r, h_shift_r=h_shift_r, shear_range=shear_range,      
        brightness_range=brightness_range, weights_on_data=weights_on_data,     
        weights_path=loss_weight_dir)
else:                                                                           
    print("Custom DA selected")
    # NOT IMPLEMENTED YET #


##########################
#    BUILD THE NETWORK   #
##########################

print("###################\n#  TRAIN PROCESS  #\n###################\n")

print("Creating the network . . .")
model = U_Net(img_train_shape, numInitChannels=num_init_channels, 
              fixed_dropout=fixed_dropout_value, spatial_dropout=spatial_dropout,
              loss_type=loss_type, optimizer=optimizer, lr=learning_rate_value,
              fine_tunning=fine_tunning)
# Check the network created
model.summary(line_length=150)
os.makedirs(char_dir, exist_ok=True)
model_name = os.path.join(char_dir, "model_plot_" + job_file + ".png")
plot_model(model, to_file=model_name, show_shapes=True, show_layer_names=True)

if load_previous_weights == False:
    earlystopper = EarlyStopping(patience=patience, verbose=1,
                                 restore_best_weights=True)

    os.makedirs(h5_dir, exist_ok=True)
    checkpointer = ModelCheckpoint(
        os.path.join(h5_dir, weight_files_prefix + job_file + '.h5'),
        verbose=1, save_best_only=True)

    if fine_tunning == True:
        h5_file=os.path.join(h5_dir, weight_files_prefix + fine_tunning_weigths
                                     + '_' + test_id + '.h5')
        print("Fine-tunning: loading model weights from h5_file: {}".format(h5_file))
        model.load_weights(h5_file)

    results = model.fit_generator(
        train_generator, validation_data=val_generator,
        validation_steps=math.ceil(n_val_samples/batch_size_value),
        steps_per_epoch=math.ceil(n_train_samples/batch_size_value),
        epochs=epochs_value,
        callbacks=[earlystopper, checkpointer, time_callback])

else:
    h5_file=os.path.join(h5_dir, weight_files_prefix + previous_job_weights
                                 + '_' + test_id + '.h5')
    print("Loading model weights from h5_file: {}".format(h5_file))
    model.load_weights(h5_file)


#####################
#     INFERENCE     #
#####################

print("##################\n#    INFERENCE   #\n##################\n")

# Evaluate to obtain the loss value and the Jaccard index (per crop)
print("Evaluating test data . . .")
if loss_type == "w_bce":
    gen = combine_generators(X_test_augmented, Y_test_augmented, W_test_augmented)
    score = model.evaluate_generator(
        gen, steps=math.ceil(n_test_samples/batch_size_value), verbose=1)
else:
    score = model.evaluate_generator(
        zip(X_test_augmented, Y_test_augmented),
        steps=math.ceil(n_test_samples/batch_size_value), verbose=1)

jac_per_crop = score[1]

X_test_augmented.reset()
Y_test_augmented.reset()

# Predict on test
print("Making the predictions on test data . . .")
if loss_type == "w_bce":
    gen = combine_generators(X_test_augmented, Y_test_augmented, W_test_augmented)
    preds_test = model.predict_generator(
        gen, steps=math.ceil(n_test_samples/batch_size_value), verbose=1)
else:
    preds_test = model.predict_generator(
        zip(X_test_augmented, Y_test_augmented),
        steps=math.ceil(n_test_samples/batch_size_value), verbose=1)

# Load Y_test and reconstruct the original images
print("Loading test masks to make the predictions . . .")
test_mask_ids = sorted(next(os.walk(os.path.join(test_mask_path, 'y')))[2])
Y_test = np.zeros((len(test_mask_ids), img_test_shape[1], img_test_shape[0],
                   img_test_shape[2]), dtype=np.float32)

for n, id_ in tqdm(enumerate(test_mask_ids), total=len(test_mask_ids)):
  mask = imread(os.path.join(test_mask_path, 'y', id_))
  if len(mask.shape) == 2:
    mask = np.expand_dims(mask, axis=-1)
  Y_test[n,:,:,:] = mask
Y_test = Y_test / 255
Y_test = Y_test.astype(np.uint8)

# Calculate number of crops per dimension to reconstruct the full image
h_num = int(original_test_shape[0] / preds_test.shape[1]) \
        + (original_test_shape[0] % preds_test.shape[1] > 0)
v_num = int(original_test_shape[1] / preds_test.shape[2]) \
        + (original_test_shape[1] % preds_test.shape[2] > 0)

Y_test = merge_data_without_overlap(
    Y_test, math.ceil(Y_test.shape[0]/(h_num*v_num)), out_shape=[h_num, v_num], 
    grid=False)
preds_test = merge_data_without_overlap(
    preds_test, math.ceil(preds_test.shape[0]/(h_num*v_num)),
    out_shape=[h_num, v_num], grid=False)

print("Calculate metrics . . .")
# Threshold images                                                              
bin_preds_test = (preds_test > 0.5).astype(np.uint8)
# Per image without overlap
score[1] = jaccard_index_numpy(Y_test, bin_preds_test)
voc = voc_calculation(Y_test, bin_preds_test, score[1])
#det = DET_calculation(Y_test, bin_preds_test, det_eval_ge_path,
#                      det_eval_path, det_bin, n_dig, job_id)
det = -1

print("Saving predicted images . . .")
save_img(Y=bin_preds_test, mask_dir=result_bin_dir, prefix="test_out_bin")
save_img(Y=preds_test, mask_dir=result_no_bin_dir, prefix="test_out_no_bin")

# Per image with 50% overlap
Y_test_50ov = np.zeros(Y_test.shape, dtype=(np.float32))
cont = 0                                                                        
for i in tqdm(range(X_complete_aug.n)):                                         
    if cont == 0:                                                               
        images = next(X_complete_aug)                                           
                                                                                
        if loss_type == "w_bce":                                                
            masks = next(Y_complete_aug)                                        
            maps = next(W_complete_aug)                                         
                                                                                
    if weights_on_data == False:                                                
        predictions_smooth = predict_img_with_overlap(                          
            images[cont], window_size=img_train_shape[0], subdivisions=2,       
            nb_classes=1, pred_func=(                                           
                lambda img_batch_subdiv: model.predict(img_batch_subdiv)))      
    else:                                                                       
        predictions_smooth = predict_img_with_overlap_weighted(                 
            images[cont], masks[cont], maps[cont], batch_size_value,            
            window_size=img_train_shape[0], subdivisions=2, nb_classes=1,       
            pred_func=(                                                         
                lambda img_batch_subdiv,                                        
                       steps: model.predict_generator(img_batch_subdiv, steps)))
                                                                                
    Y_test_50ov[i] = predictions_smooth                                         
                                                                                
    cont += 1                                                                   
    if cont == batch_size_value:                                                
        cont = 0

print("Saving 50% overlap predicted images . . .")
save_img(Y=(Y_test_50ov > 0.5).astype(np.uint8), mask_dir=result_bin_dir_50ov, 
         prefix="test_out_bin_50ov")
save_img(Y=Y_test_50ov, mask_dir=result_no_bin_dir_50ov,
         prefix="test_out_no_bin_50ov")

X_complete_aug.reset()
if loss_type == "w_bce":
    Y_complete_aug.reset()
    W_complete_aug.reset()

print("Calculate metrics for 50% overlap images . . .")
jac_per_img_50ov = jaccard_index_numpy(
    Y_test, (Y_test_50ov > 0.5).astype(np.uint8))
voc_per_img_50ov = voc_calculation(
    Y_test, (Y_test_50ov > 0.5).astype(np.uint8), jac_per_img_50ov)
#det_per_img_50ov = DET_calculation(
#   Y_test, (Y_test_50ov > 0.5).astype(np.uint8), det_eval_ge_path,
#   det_eval_path, det_bin, n_dig, job_id)
det_per_img_50ov = -1

del Y_test_50ov


####################
#  POST-PROCESING  #
####################

print("##################\n# POST-PROCESING #\n##################\n") 

print("1) SMOOTH")
if post_process == True:

    print("Post processing active . . .")
    Y_test_smooth = np.zeros((X_complete_aug.n, original_test_shape[0],
                              original_test_shape[1], original_test_shape[2]),
                             dtype=np.uint8)

    # Extract the number of digits to create the image names
    d = len(str(X_complete_aug.n))

    os.makedirs(smooth_dir, exist_ok=True)

    print("Smoothing crops . . .")
    cont = 0                                                                    
    for i in tqdm(range(X_complete_aug.n)):                                     
        if cont == 0:                                                           
            images = next(X_complete_aug)                                       
            masks = next(Y_complete_aug)                                        
                                                                                
            if loss_type == "w_bce":                                            
                maps = next(W_complete_aug)                                     
                                                                                
        if weights_on_data == False:                                            
            predictions_smooth = predict_img_with_smooth_windowing(             
                images[cont], window_size=img_train_shape[0], subdivisions=2,   
                nb_classes=1, pred_func=(                                       
                    lambda img_batch_subdiv: model.predict(img_batch_subdiv)))  
        else:                                                                   
            predictions_smooth = predict_img_with_overlap_weighted(             
                images[cont], masks[cont], maps[cont], batch_size_value,        
                window_size=img_train_shape[0], subdivisions=2, nb_classes=1,   
                pred_func=(                                                     
                lambda img_batch_subdiv,                                        
                       steps: model.predict_generator(img_batch_subdiv, steps)))          
                                                                                
        Y_test_smooth[cont] = (predictions_smooth > 0.5).astype(np.uint8)       
                                                                                
        cont += 1                                                               
        if cont == batch_size_value:                                            
            cont = 0

        im = Image.fromarray(predictions_smooth[:,:,0]*255)
        im = im.convert('L')
        im.save(os.path.join(smooth_dir, "test_out_smooth_" 
                             + str(cont).zfill(d) + ".png"))

    # Metrics (Jaccard + VOC + DET)                                             
    print("Calculate metrics . . .")
    smooth_score = jaccard_index_numpy(Y_test, Y_test_smooth)
    smooth_voc = voc_calculation(Y_test, Y_test_smooth, smooth_score)
#    smooth_det = DET_calculation(Y_test, Y_test_smooth, det_eval_ge_path,
#                                 det_eval_post_path, det_bin, n_dig, job_id)

if post_process == True:
    print("2) Z-FILTERING")

    print("Applying Z-filter . . .")
    zfil_preds_test = calculate_z_filtering(bin_preds_test)

    print("Saving Z-filtered images . . .")
    save_img(Y=zfil_preds_test, mask_dir=zfil_dir, prefix="test_out_zfil")

    print("Calculate metrics for the Z-filtered data . . .")
    zfil_score = jaccard_index_numpy(Y_test, zfil_preds_test)
    zfil_voc = voc_calculation(Y_test, zfil_preds_test, zfil_score)
#    zfil_det = DET_calculation(Y_test, zfil_preds_test, det_eval_ge_path,
#                               det_eval_post_path, det_bin, n_dig, job_id)

    print("Applying Z-filter to the smoothed data . . .")
    smooth_zfil_preds_test = calculate_z_filtering(Y_test_smooth)

    print("Saving smoothed + Z-filtered images . . .")
    save_img(Y=smooth_zfil_preds_test, mask_dir=smoo_zfil_dir, 
             prefix="test_out_smoo_zfil")

    print("Calculate metrics for the smoothed + Z-filtered data . . .")
    smo_zfil_score = jaccard_index_numpy(Y_test, smooth_zfil_preds_test)
    smo_zfil_voc = voc_calculation(Y_test, smooth_zfil_preds_test,
                                   smo_zfil_score)
#    smo_zfil_det = DET_calculation(Y_test, smooth_zfil_preds_test,
#                                   det_eval_ge_path, det_eval_post_path,
#                                   det_bin, n_dig, job_id)

del Y_test
print("Finished post-processing!") 


####################################
#  PRINT AND SAVE SCORES OBTAINED  #
####################################

if load_previous_weights == False:
    print("Epoch average time: {}".format(np.mean(time_callback.times)))
    print("Epoch number: {}".format(len(results.history['val_loss'])))
    print("Train time (s): {}".format(np.sum(time_callback.times)))
    print("Train loss: {}".format(np.min(results.history['loss'])))
    print("Train jaccard_index: {}".format(np.max(results.history['jaccard_index'])))
    print("Validation loss: {}".format(np.min(results.history['val_loss'])))
    print("Validation jaccard_index: {}".format(np.max(results.history['val_jaccard_index'])))

print("Test loss: {}".format(score[0]))
print("Test jaccard_index (per crop): {}".format(jac_per_crop))
print("Test jaccard_index (per image without overlap): {}".format(score[1]))
print("Test jaccard_index (per image with 50% overlap): {}".format(jac_per_img_50ov))
print("VOC (per image without overlap): {}".format(voc))
print("VOC (per image with 50% overlap): {}".format(voc_per_img_50ov))
print("DET (per image without overlap): {}".format(det))
print("DET (per image with 50% overlap): {}".format(det_per_img_50ov))

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
    jac_per_crop = -1 if 'jac_per_crop' not in globals() else jac_per_crop

    store_history(
        results, jac_per_crop, score, jac_per_img_50ov, voc, voc_per_img_50ov, 
        det, det_per_img_50ov, time_callback, log_dir, job_file, smooth_score, 
        smooth_voc, smooth_det, zfil_score, zfil_voc, zfil_det, smo_zfil_score, 
        smo_zfil_voc, smo_zfil_det)

    create_plots(results, job_id, test_id, char_dir)

if post_process == True:
    print("Post-process: SMOOTH - Test jaccard_index: {}".format(smooth_score))
    print("Post-process: SMOOTH - VOC: {}".format(smooth_voc))
    print("Post-process: SMOOTH - DET: {}".format(smooth_det))
    print("Post-process: Z-filtering - Test jaccard_index: {}".format(zfil_score))
    print("Post-process: Z-filtering - VOC: {}".format(zfil_voc))
    print("Post-process: Z-filtering - DET: {}".format(zfil_det))
    print("Post-process: SMOOTH + Z-filtering - Test jaccard_index: "
          + str(smo_zfil_score))
    print("Post-process: SMOOTH + Z-filtering - VOC: {}".format(smo_zfil_voc))
    print("Post-process: SMOOTH + Z-filtering - DET: {}".format(smo_zfil_det))

print("FINISHED JOB {} !!".format(job_file))
