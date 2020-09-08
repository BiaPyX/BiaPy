# Script based on template.py

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

# Working dir
os.chdir(args.base_work_dir)

# Limit the number of threads
from util import limit_threads, set_seed, create_plots, store_history,\
                 TimeHistory, threshold_plots, save_img, \
                 calculate_2D_volume_prob_map
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
from data_manipulation import load_data_from_dir, merge_data_without_overlap,\
                              merge_data_with_overlap, check_binary_masks, \
                              img_to_onehot_encoding
from data_generators import keras_da_generator, ImageDataGenerator,\
                            keras_gen_samples
from networks.unet import U_Net_2D
from metrics import jaccard_index_numpy, voc_calculation, DET_calculation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from PIL import Image
from tqdm import tqdm
from smooth_tiled_predictions import predict_img_with_smooth_windowing, \
                                     predict_img_with_overlap,\
                                     ensemble8_2d_predictions
from tensorflow.keras.utils import plot_model
from callbacks import ModelCheckpoint
from post_processing import spuriuous_detection_filter, calculate_z_filtering,\
                            boundary_refinement_watershed2


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
val_path = os.path.join(args.data_dir, 'val', 'x')                              
val_mask_path = os.path.join(args.data_dir, 'val', 'y')                         
test_path = os.path.join(args.data_dir, 'test', 'x')                            
test_mask_path = os.path.join(args.data_dir, 'test', 'y')                       
complete_test_path = os.path.join(args.data_dir, '..', '..', 'test', 'x')       
complete_test_mask_path = os.path.join(args.data_dir, '..', '..', 'test', 'y')  
# Percentage of the training data used as validation                            
perc_used_as_val = 0.1
# Create the validation data with random images of the training data. If False
# the validation data will be the last portion of training images.
random_val_data = True


### Dataset shape
# Note: train and test dimensions must be the same when training the network and
# making the predictions. Be sure to take care of this if you are not going to
# use "crop_data()" with the arg force_shape, as this function resolves the 
# problem creating always crops of the same dimension
img_train_shape = (256, 256, 1)
img_test_shape = (256, 256, 1)
orig_test_shape = (4096, 4096, 1)                                           


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
# Flag to shuffle the training data on every epoch:
# (Best options: Keras->False, Custom->True)
shuffle_train_data_each_epoch = custom_da
# Flag to shuffle the validation data on every epoch:
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
# Range of rotation
rotation_range = 180
# Flag to make flips on the subvolumes. Available for both Keras and custom DA.
flips = True


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
weight_files_prefix = 'model.c_human_'
# Wheter to find the best learning rate plot. If this options is selected the
# training will stop when 5 epochs are done
use_LRFinder = False


### Experiment main parameters
# Loss type, three options: "bce" or "w_bce_dice", which refers to binary cross 
# entropy (BCE) and BCE and Dice with with a weight term on each one (that must 
# sum 1) to calculate the total loss value. NOTE: "w_bce" is not implemented on 
# this template type: please use big_data_template.py instead.
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
# If weights on data are going to be applied. To true when loss_type is 'w_bce' 
weights_on_data = True if loss_type == "w_bce" else False


### Network architecture specific parameters
# Number of feature maps on each level of the network. It's dimension must be 
# equal depth+1.
feature_maps = [32, 64, 128, 256, 512]
# Depth of the network
depth = 4
# Flag to activate the Spatial Dropout instead of use the "normal" dropout layer
spatial_dropout = False
# Values to make the dropout with. It's dimension must be equal depth+1. Set to
# None to prevent dropout 
dropout_values = [0.1, 0.1, 0.2, 0.2, 0.3]
# Flag to active batch normalization
batch_normalization = False
# Kernel type to use on convolution layers
kernel_init = 'he_normal'
# Activation function to use                                                    
activation = "elu" 
# Active flag if softmax or one channel per class is used as the last layer of
# the network. Custom DA needed.
softmax_out = False


### DET metric variables
# More info of the metric at http://celltrackingchallenge.net/evaluation-methodology/ 
# and https://public.celltrackingchallenge.net/documents/Evaluation%20software.pdf
# NEEDED CODE REFACTORING OF THIS VARIABLE
det_eval_ge_path = os.path.join(args.result_dir, "..", 'cell_challenge_eval',
                                 'gen_' + job_identifier)
# Path where the evaluation of the metric will be done
det_eval_path = os.path.join(args.result_dir, "..", 'cell_challenge_eval', 
                             args.job_id, job_identifier)
# Path where the evaluation of the metric for the post processing methods will 
# be done
det_eval_post_path = os.path.join(args.result_dir, "..", 'cell_challenge_eval', 
                                  args.job_id, job_identifier + '_s')
# Path were the binaries of the DET metric is stored
det_bin = os.path.join(args.base_work_dir, 'cell_cha_eval' ,'Linux', 'DETMeasure')
# Number of digits used for encoding temporal indices of the DET metric
n_dig = "3"


### Paths of the results                                             
result_dir = os.path.join(args.result_dir, 'results', job_identifier)

# Directory where binarized predicted images will be stored
result_bin_dir_per_image = os.path.join(result_dir, 'per_image_binarized')
# Directory where predicted images will be stored
result_no_bin_dir_per_image = os.path.join(result_dir, 'per_image_no_binarized')
# Folder where the smoothed images will be stored
smo_bin_dir_per_image = os.path.join(result_dir, 'per_image_smooth')
# Folder where the smoothed images (no binarized) will be stored
smo_no_bin_dir_per_image = os.path.join(result_dir, 'per_image_smooth_no_bin')
# Folder where the images with the z-filter applied will be stored
zfil_dir_per_image = os.path.join(result_dir, 'per_image_zfil')
# Folder where the images with smoothing and z-filter applied will be stored
smo_zfil_dir_per_image = os.path.join(result_dir, 'per_image_smo_zfil')

# Directory where binarized predicted images with 50% of overlap will be stored
result_bin_dir_50ov = os.path.join(result_dir, '50ov_binarized')
# Directory where predicted images with 50% of overlap will be stored
result_no_bin_dir_50ov = os.path.join(result_dir, '50ov_no_binarized')
# Folder where the images with the z-filter applied will be stored
zfil_dir_50ov = os.path.join(result_dir, '50ov_zfil')

# Directory where binarired predicted images obtained from feeding the full
# image will be stored
result_bin_dir_full = os.path.join(result_dir, 'full_binarized')
# Directory where predicted images obtained from feeding the full image will
# be stored
result_no_bin_dir_full = os.path.join(result_dir, 'full_no_binarized')
# Folder where the smoothed images will be stored
smo_bin_dir_full = os.path.join(result_dir, 'full_8ensemble')
# Folder where the smoothed images (no binarized) will be stored
smo_no_bin_dir_full = os.path.join(result_dir, 'full_8ensemble')
# Folder where the images with the z-filter applied will be stored
zfil_dir_full = os.path.join(result_dir, 'full_zfil')
# Folder where the images passed through the spurious detection filtering will
# be saved in
spu_dir_full = os.path.join(result_dir, 'full_spu')
# Folder where watershed debugging images will be placed in
wa_debug_dir_full = os.path.join(result_dir, 'full_watershed_debug')
# Folder where watershed output images will be placed in
wa_dir_full = os.path.join(result_dir, 'full_watershed')
# Folder where spurious detection + watershed + z-filter images' watershed
# markers will be placed in
spu_wa_zfil_wa_debug_dir = os.path.join(result_dir, 'full_wa_spu_zfil_wa_debug')
# Folder where spurious detection + watershed + z-filter images will be placed in
spu_wa_zfil_dir_full = os.path.join(result_dir, 'full_wa_spu_zfil')

# Name of the folder where the charts of the loss and metrics values while 
# training the network will be shown. This folder will be created under the
# folder pointed by "args.base_work_dir" variable 
char_dir = os.path.join(result_dir, 'charts')
# Directory where weight maps will be stored                                    
loss_weight_dir = os.path.join(result_dir, 'loss_weights', args.job_id)
# Folder where smaples of DA will be stored
da_samples_dir = os.path.join(result_dir, 'aug')
# Folder where crop samples will be stored
check_crop_path = os.path.join(result_dir, 'check_crop')
# Name of the folder where weights files will be stored/loaded from. This folder
# must be located inside the directory pointed by "args.base_work_dir" variable.
# If there is no such directory, it will be created for the first time
h5_dir = os.path.join(args.result_dir, 'h5_files')
# Name of the folder to store the probability map to avoid recalculating it on
# every run
prob_map_dir = os.path.join(args.result_dir, 'prob_map')
# Folder where LRFinder callback will store its plot
lrfinder_dir = os.path.join(result_dir, 'LRFinder')


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
# Check the best learning rate using the code from:
#  https://github.com/WittmannF/LRFinder
if use_LRFinder:
    lr_finder = LRFinder(min_lr=10e-9, max_lr=10e-3, lrfinder_dir=lrfinder_dir)
    os.makedirs(lrfinder_dir, exist_ok=True)


print("###################\n"
      "#  SANITY CHECKS  #\n"
      "###################\n")

check_binary_masks(train_mask_path)
check_binary_masks(test_mask_path)
if extra_datasets_mask_list: 
    for i in range(len(extra_datasets_mask_list)):
        check_binary_masks(extra_datasets_mask_list[i])


print("#######################\n"
      "#  DATA AUGMENTATION  #\n"
      "#######################\n")

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
        c_target_size=(orig_test_shape[0], orig_test_shape[1]),         
        batch_size_value=batch_size_value, save_examples=aug_examples,          
        out_dir=da_samples_dir, shuffle_train=shuffle_train_data_each_epoch,    
        shuffle_val=shuffle_val_data_each_epoch, zoom=keras_zoom,               
        rotation_range=rotation_range, w_shift_r=w_shift_r, h_shift_r=h_shift_r,
        shear_range=shear_range, brightness_range=brightness_range,             
        weights_on_data=weights_on_data, weights_path=loss_weight_dir,          
        hflip=flips, vflip=flips)                                               
else:                                                                           
    print("Custom DA selected")                                                 
    # NOT IMPLEMENTED YET #  


print("#################################\n"
      "#  BUILD AND TRAIN THE NETWORK  #\n"
      "#################################\n")

print("Creating the network . . .")
model = U_Net_2D([img_height, img_width, img_channels], activation=activation,
                 feature_maps=feature_maps, depth=depth, 
                 drop_values=dropout_values, spatial_dropout=spatial_dropout,
                 batch_norm=batch_normalization, k_init=kernel_init,
                 loss_type=loss_type, optimizer=optimizer, 
                 lr=learning_rate_value, fine_tunning=fine_tunning)

# Check the network created
model.summary(line_length=150)
os.makedirs(char_dir, exist_ok=True)
model_name = os.path.join(char_dir, "model_plot_" + job_identifier + ".png")
plot_model(model, to_file=model_name, show_shapes=True, show_layer_names=True)

if load_previous_weights == False:
    if fine_tunning:                                                    
        h5_file=os.path.join(h5_dir, weight_files_prefix + fine_tunning_weigths 
                             + '_' + args.run_id + '.h5')     
        print("Fine-tunning: loading model weights from h5_file: {}"
              .format(h5_file))
        model.load_weights(h5_file)                                             
   
    if use_LRFinder:
        print("Training just for 10 epochs . . .")
        results = model.fit_generator(
            train_generator, validation_data=val_generator,
            validation_steps=math.ceil(n_val_samples/batch_size_value),
            steps_per_epoch=math.ceil(n_train_samples/batch_size_value),
            epochs=epochs_value,
            callbacks=[earlystopper, checkpointer, time_callback])

        results = model.fit(x=train_generator, validation_data=val_generator,
            validation_steps=math.ceil(n_val_samples/batch_size_value),
            steps_per_epoch=math.ceil(n_train_samples/batch_size_value),
            epochs=5, callbacks=[lr_finder])

        print("Finish LRFinder. Check the plot in {}".format(lrfinder_dir))
        sys.exit(0)
    else:
        results = model.fit(x=train_generator, validation_data=val_generator,
            validation_steps=math.ceil(n_val_samples/batch_size_value),
            steps_per_epoch=math.ceil(n_train_samples/batch_size_value),
            epochs=epochs_value, callbacks=[earlystopper, checkpointer, time_callback])
else:
    h5_file=os.path.join(h5_dir, weight_files_prefix + previous_job_weights 
                         + '_' + str(args.run_id) + '.h5')
    print("Loading model weights from h5_file: {}".format(h5_file))
    model.load_weights(h5_file)


print("##########################\n"
      "#  INFERENCE (per crop)  #\n"
      "##########################\n")

print("Evaluating test data . . .")                                             
if loss_type == "w_bce":                                                        
    gen = combine_generators(X_test_augmented, Y_test_augmented, W_test_augmented)
    score_per_crop = model.evaluate(                                                     
        gen, steps=math.ceil(n_test_samples/batch_size_value), verbose=1)       
else:                                                                           
    score_per_crop = model.evaluate(                                                     
        zip(X_test_augmented, Y_test_augmented),                                
        steps=math.ceil(n_test_samples/batch_size_value), verbose=1)            
loss_per_crop = score_per_crop[0]                                               
jac_per_crop = score_per_crop[1]
                                                                                
X_test_augmented.reset()                                                        
Y_test_augmented.reset()                                                        
                                                                                
print("Making the predictions on test data . . .")                              
if loss_type == "w_bce":                                                        
    gen = combine_generators(X_test_augmented, Y_test_augmented, W_test_augmented)
    preds_test = model.predict(
        gen, steps=math.ceil(n_test_samples/batch_size_value), 
        batch_size=batch_size_value, verbose=1)       
else:                                                                           
    preds_test = model.predict(                                                 
        zip(X_test_augmented, Y_test_augmented), 
        steps=math.ceil(n_test_samples/batch_size_value), 
        batch_size=batch_size_value, verbose=1) 

if softmax_out:
    preds_test = np.expand_dims(preds_test[...,1], -1)


print("################\n"                              
      "#  Load Y_test #\n"
      "################\n")

# Load Y_test to calculate the metrics                                          
print("Loading test masks to make the predictions . . .")                       
Y_test = load_data_from_dir(os.path.join(test_mask_path, 'y'),                  
    (img_test_shape[1], img_test_shape[0], img_test_shape[2]))                  
Y_test = (Y_test/255).astype(np.uint8)   


print("########################################\n"
      "#  Metrics (per image, merging crops)  #\n"
      "########################################\n")

# Calculate number of crops per dimension to reconstruct the full images        
h_num = math.ceil(orig_test_shape[0]/preds_test.shape[1])                   
v_num = math.ceil(orig_test_shape[1]/preds_test.shape[2])                   
                                                                                
# Reconstruct the predict and Y_test images                                     
print("Reconstruct preds_test . . .")
preds_test = merge_data_without_overlap(
    preds_test, math.ceil(preds_test.shape[0]/(h_num*v_num)),
    out_shape=[h_num, v_num], grid=False)
print("Reconstruct Y_test . . .")
Y_test = merge_data_without_overlap(
    Y_test, math.ceil(Y_test.shape[0]/(h_num*v_num)),
    out_shape=[h_num, v_num], grid=False)

print("Saving predicted images . . .")
save_img(Y=(preds_test > 0.5).astype(np.uint8),
         mask_dir=result_bin_dir_per_image, prefix="test_out_bin")
save_img(Y=preds_test, mask_dir=result_no_bin_dir_per_image,
         prefix="test_out_no_bin")

print("Calculate metrics (per image) . . .")
jac_per_image = jaccard_index_numpy(Y_test, (preds_test > 0.5).astype(np.uint8))
voc_per_image = voc_calculation(
    Y_test, (preds_test > 0.5).astype(np.uint8), jac_per_image)
det_per_image = DET_calculation(
    Y_test, (preds_test > 0.5).astype(np.uint8), det_eval_ge_path, det_eval_path,
    det_bin, n_dig, args.job_id)

print("~~~~ Smooth (per image) ~~~~")
Y_test_smooth = np.zeros(X_test.shape, dtype=np.float32)
cont = 0
for i in tqdm(range(X_complete_aug.n)):
    if cont == 0:                                                               
        images = next(X_complete_aug)                                           
                                                                                
        if loss_type == "w_bce":                                                
            masks = next(Y_complete_aug)                                        
            maps = next(W_complete_aug)

    predictions_smooth = predict_img_with_smooth_windowing(
        X_test[i], window_size=crop_shape[0], subdivisions=2, nb_classes=1,
        pred_func=(lambda img_batch_subdiv: model.predict(img_batch_subdiv)),
        softmax=softmax_out)
    Y_test_smooth[i] = predictions_smooth

print("Saving smooth predicted images . . .")
save_img(Y=Y_test_smooth, mask_dir=smo_no_bin_dir_per_image,
         prefix="test_out_smo_no_bin")
save_img(Y=(Y_test_smooth > 0.5).astype(np.uint8), mask_dir=smo_bin_dir_per_image,
         prefix="test_out_smo")

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

print("Calculate metrics (smooth + per crop) . . .")
smo_score_per_image = jaccard_index_numpy(
    Y_test, (Y_test_smooth > 0.5).astype(np.uint8))
smo_voc_per_image = voc_calculation(
    Y_test, (Y_test_smooth > 0.5).astype(np.uint8), smo_score_per_image)
smo_det_per_image = DET_calculation(
    Y_test, (Y_test_smooth > 0.5).astype(np.uint8), det_eval_ge_path,
    det_eval_post_path, det_bin, n_dig, args.job_id)

print("~~~~ Z-Filtering (per image) ~~~~")
zfil_preds_test = calculate_z_filtering(preds_test)

print("Saving Z-filtered images . . .")
save_img(Y=zfil_preds_test, mask_dir=zfil_dir_per_image, prefix="test_out_zfil")

print("Calculate metrics (Z-filtering + per crop) . . .")
zfil_score_per_image = jaccard_index_numpy(
    Y_test, (zfil_preds_test > 0.5).astype(np.uint8))
zfil_voc_per_image = voc_calculation(
    Y_test, (zfil_preds_test > 0.5).astype(np.uint8), zfil_score_per_image)
zfil_det_per_image = DET_calculation(
    Y_test, (zfil_preds_test > 0.5).astype(np.uint8), det_eval_ge_path,
    det_eval_post_path, det_bin, n_dig, args.job_id)
del zfil_preds_test, preds_test

print("~~~~ Smooth + Z-Filtering (per image) ~~~~")
smo_zfil_preds_test = calculate_z_filtering(Y_test_smooth)

print("Saving smoothed + Z-filtered images . . .")
save_img(Y=smo_zfil_preds_test, mask_dir=smo_zfil_dir_per_image,
         prefix="test_out_smoo_zfil")

print("Calculate metrics (Smooth + Z-filtering per crop) . . .")
smo_zfil_score_per_image = jaccard_index_numpy(
    Y_test, (smo_zfil_preds_test > 0.5).astype(np.uint8))
smo_zfil_voc_per_image = voc_calculation(
    Y_test, (smo_zfil_preds_test > 0.5).astype(np.uint8),
    smo_zfil_score_per_image)
smo_zfil_det_per_image = DET_calculation(
    Y_test, (smo_zfil_preds_test > 0.5).astype(np.uint8),
    det_eval_ge_path, det_eval_post_path, det_bin, n_dig, args.job_id)

del Y_test_smooth, smo_zfil_preds_test


print("############################################################\n"
      "#  Metrics (per image, merging crops with 50% of overlap)  #\n"
      "############################################################\n")

Y_test_50ov = np.zeros(X_test.shape, dtype=np.float32)
for i in tqdm(range(X_test.shape[0])):
    predictions_smooth = predict_img_with_overlap(
        X_test[i], window_size=crop_shape[0], subdivisions=2,
        nb_classes=1, pred_func=(
            lambda img_batch_subdiv: model.predict(img_batch_subdiv)),
        softmax=softmax_out)
    Y_test_50ov[i] = predictions_smooth

print("Saving 50% overlap predicted images . . .")
save_img(Y=(Y_test_50ov > 0.5).astype(np.float32),
         mask_dir=result_bin_dir_50ov, prefix="test_out_bin_50ov")
save_img(Y=Y_test_50ov, mask_dir=result_no_bin_dir_50ov,
         prefix="test_out_no_bin_50ov")

print("Calculate metrics (50% overlap) . . .")
jac_50ov = jaccard_index_numpy(Y_test, (Y_test_50ov > 0.5).astype(np.float32))
voc_50ov = voc_calculation(
    Y_test, (Y_test_50ov > 0.5).astype(np.float32), jac_50ov)
det_50ov = DET_calculation(
    Y_test, (Y_test_50ov > 0.5).astype(np.float32), det_eval_ge_path,
    det_eval_path, det_bin, n_dig, args.job_id)

print("~~~~ Z-Filtering (50% overlap) ~~~~")
zfil_preds_test = calculate_z_filtering(Y_test_50ov)

print("Saving Z-filtered images . . .")
save_img(Y=zfil_preds_test, mask_dir=zfil_dir_50ov, prefix="test_out_zfil")

print("Calculate metrics (Z-filtering + 50% overlap) . . .")
zfil_score_50ov = jaccard_index_numpy(
    Y_test, (zfil_preds_test > 0.5).astype(np.uint8))
zfil_voc_50ov = voc_calculation(
    Y_test, (zfil_preds_test > 0.5).astype(np.uint8), zfil_score_50ov)
zfil_det_50ov = DET_calculation(
    Y_test, (zfil_preds_test > 0.5).astype(np.uint8), det_eval_ge_path,
    det_eval_post_path, det_bin, n_dig, args.job_id)
del Y_test_50ov, zfil_preds_test


print("########################\n"
      "# Metrics (full image) #\n"
      "########################\n")

print("Making the predictions on test data . . .")
preds_test_full = model.predict(X_test, batch_size=batch_size_value, verbose=1)

if softmax_out:
    preds_test_full = np.expand_dims(preds_test_full[...,1], -1)

print("Saving predicted images . . .")
save_img(Y=(preds_test_full > 0.5).astype(np.uint8),
         mask_dir=result_bin_dir_full, prefix="test_out_bin_full")
save_img(Y=preds_test_full, mask_dir=result_no_bin_dir_full,
         prefix="test_out_no_bin_full")

print("Calculate metrics (full image) . . .")
jac_full = jaccard_index_numpy(Y_test, (preds_test_full > 0.5).astype(np.uint8))
voc_full = voc_calculation(Y_test, (preds_test_full > 0.5).astype(np.uint8),
                           jac_full)
det_full = DET_calculation(
    Y_test, (preds_test_full > 0.5).astype(np.uint8), det_eval_ge_path,
    det_eval_path, det_bin, n_dig, args.job_id)

print("~~~~ 8-Ensemble (full image) ~~~~")
Y_test_smooth = np.zeros(X_test.shape, dtype=(np.float32))

for i in tqdm(range(X_test.shape[0])):
    predictions_smooth = ensemble8_2d_predictions(X_test[i],
        pred_func=(lambda img_batch_subdiv: model.predict(img_batch_subdiv)),
        softmax_output=softmax_out)
    Y_test_smooth[i] = predictions_smooth
del X_test

print("Saving smooth predicted images . . .")
save_img(Y=Y_test_smooth, mask_dir=smo_no_bin_dir_full,
         prefix="test_out_ens_no_bin")
save_img(Y=(Y_test_smooth > 0.5).astype(np.uint8), mask_dir=smo_bin_dir_full,
         prefix="test_out_ens")

print("Calculate metrics (8-Ensemble + full image) . . .")
smo_score_full = jaccard_index_numpy(
    Y_test, (Y_test_smooth > 0.5).astype(np.uint8))
smo_voc_full = voc_calculation(
    Y_test, (Y_test_smooth > 0.5).astype(np.uint8), smo_score_full)
smo_det_full = DET_calculation(
    Y_test, (Y_test_smooth > 0.5).astype(np.uint8), det_eval_ge_path,
    det_eval_path, det_bin, n_dig, args.job_id)
del Y_test_smooth

print("~~~~ Z-Filtering (full image) ~~~~")
zfil_preds_test = calculate_z_filtering(preds_test_full)

print("Saving Z-filtered images . . .")
save_img(Y=zfil_preds_test, mask_dir=zfil_dir_full, prefix="test_out_zfil")

print("Calculate metrics (Z-filtering + full image) . . .")
zfil_score_full = jaccard_index_numpy(
    Y_test, (zfil_preds_test > 0.5).astype(np.uint8))
zfil_voc_full = voc_calculation(
    Y_test, (zfil_preds_test > 0.5).astype(np.uint8), zfil_score_full)
zfil_det_full = DET_calculation(
    Y_test, (zfil_preds_test > 0.5).astype(np.uint8), det_eval_ge_path,
    det_eval_post_path, det_bin, n_dig, args.job_id)

del zfil_preds_test

print("~~~~ Spurious Detection (full image) ~~~~")
spu_preds_test = spuriuous_detection_filter(preds_test_full)

print("Saving spurious detection filtering resulting images . . .")
save_img(Y=spu_preds_test, mask_dir=spu_dir_full, prefix="test_out_spu")

print("Calculate metrics (Spurious + full image) . . .")
spu_score_full = jaccard_index_numpy(Y_test, spu_preds_test)
spu_voc_full = voc_calculation(Y_test, spu_preds_test, spu_score_full)
spu_det_full = DET_calculation(Y_test, spu_preds_test, det_eval_ge_path,
                               det_eval_post_path, det_bin, n_dig, args.job_id)

print("~~~~ Watershed (full image) ~~~~")
wa_preds_test = boundary_refinement_watershed2(
    preds_test_full, (preds_test_full> 0.5).astype(np.uint8),
    save_marks_dir=wa_debug_dir_full)
    #X_test, (preds_test> 0.5).astype(np.uint8), save_marks_dir=watershed_debug_dir)

print("Saving watershed resulting images . . .")
save_img(Y=(wa_preds_test).astype(np.uint8), mask_dir=wa_dir_full,
         prefix="test_out_wa")

print("Calculate metrics (Watershed + full image) . . .")
wa_score_full = jaccard_index_numpy(Y_test, wa_preds_test)
wa_voc_full = voc_calculation(Y_test, wa_preds_test, wa_score_full)
wa_det_full = DET_calculation(Y_test, wa_preds_test, det_eval_ge_path,
                              det_eval_post_path, det_bin, n_dig, args.job_id)
del preds_test_full, wa_preds_test

print("~~~~ Spurious Detection + Watershed + Z-filtering (full image) ~~~~")
# Use spu_preds_test
spu_wa_zfil_preds_test = boundary_refinement_watershed2(
    spu_preds_test, (spu_preds_test> 0.5).astype(np.uint8),
    save_marks_dir=spu_wa_zfil_wa_debug_dir)
    #X_test, (preds_test> 0.5).astype(np.uint8), save_marks_dir=watershed_debug_dir)

spu_wa_zfil_preds_test = calculate_z_filtering(spu_wa_zfil_preds_test)

print("Saving Z-filtered images . . .")
save_img(Y=spu_wa_zfil_preds_test, mask_dir=spu_wa_zfil_dir_full,
         prefix="test_out_spu_wa_zfil")

print("Calculate metrics (Z-filtering + full image) . . .")
spu_wa_zfil_score_full = jaccard_index_numpy(
    Y_test, (spu_wa_zfil_preds_test > 0.5).astype(np.uint8))
spu_wa_zfil_voc_full = voc_calculation(
    Y_test, (spu_wa_zfil_preds_test > 0.5).astype(np.uint8),
    spu_wa_zfil_score_full)
spu_wa_zfil_det_full = DET_calculation(
    Y_test, (spu_wa_zfil_preds_test > 0.5).astype(np.uint8), det_eval_ge_path,
    det_eval_post_path, det_bin, n_dig, args.job_id)
del spu_wa_zfil_preds_test, spu_preds_test, Y_test


print("####################################\n"
      "#  PRINT AND SAVE SCORES OBTAINED  #\n"
      "####################################\n")

if load_previous_weights == False:
    print("Epoch average time: {}".format(np.mean(time_callback.times)))
    print("Epoch number: {}".format(len(results.history['val_loss'])))
    print("Train time (s): {}".format(np.sum(time_callback.times)))
    print("Train loss: {}".format(np.min(results.history['loss'])))
    print("Train IoU: {}".format(np.max(results.history['jaccard_index'])))
    print("Validation loss: {}".format(np.min(results.history['val_loss'])))
    print("Validation IoU: {}"
          .format(np.max(results.history['val_jaccard_index'])))

print("Test loss: {}".format(loss_per_crop))
print("Test IoU (per crop): {}".format(jac_per_crop))

print("Test IoU (merge into complete image): {}".format(jac_per_image))
print("Test VOC (merge into complete image): {}".format(voc_per_image))
print("Test DET (merge into complete image): {}".format(det_per_image))
print("Post-process: Smooth - Test IoU (merge into complete image): {}".format(smo_score_per_image))
print("Post-process: Smooth - Test VOC (merge into complete image): {}".format(smo_voc_per_image))
print("Post-process: Smooth - Test DET (merge into complete image): {}".format(smo_det_per_image))
print("Post-process: Z-Filtering - Test IoU (merge into complete image): {}".format(zfil_score_per_image))
print("Post-process: Z-Filtering - Test VOC (merge into complete image): {}".format(zfil_voc_per_image))
print("Post-process: Z-Filtering - Test DET (merge into complete image): {}".format(zfil_det_per_image))
print("Post-process: Smooth + Z-Filtering - Test IoU (merge into complete image): {}".format(smo_zfil_score_per_image))
print("Post-process: Smooth + Z-Filtering - Test VOC (merge into complete image): {}".format(smo_zfil_voc_per_image))
print("Post-process: Smooth + Z-Filtering - Test DET (merge into complete image): {}".format(smo_zfil_det_per_image))

print("Test IoU (merge with 50% overlap): {}".format(jac_50ov))
print("Test VOC (merge with 50% overlap): {}".format(voc_50ov))
print("Test DET (merge with with 50% overlap): {}".format(det_50ov))
print("Post-process: Z-Filtering - Test IoU (merge with 50% overlap): {}".format(zfil_score_50ov))
print("Post-process: Z-Filtering - Test VOC (merge with 50% overlap): {}".format(zfil_voc_50ov))
print("Post-process: Z-Filtering - Test DET (merge with 50% overlap): {}".format(zfil_det_50ov))

print("Test IoU (full): {}".format(jac_full))
print("Test VOC (full): {}".format(voc_full))
print("Test DET (full): {}".format(det_full))
print("Post-process: Ensemble - Test IoU (full): {}".format(smo_score_full))
print("Post-process: Ensemble - Test VOC (full): {}".format(smo_voc_full))
print("Post-process: Ensemble - Test DET (full): {}".format(smo_det_full))
print("Post-process: Z-Filtering - Test IoU (full): {}".format(zfil_score_full))
print("Post-process: Z-Filtering - Test VOC (full): {}".format(zfil_voc_full))
print("Post-process: Z-Filtering - Test DET (full): {}".format(zfil_det_full))
print("Post-process: Spurious Detection - Test IoU (full): {}".format(spu_score_full))
print("Post-process: Spurious Detection - VOC (full): {}".format(spu_voc_full))
print("Post-process: Spurious Detection - DET (full): {}".format(spu_det_full))
print("Post-process: Watershed - Test IoU (full): {}".format(wa_score_full))
print("Post-process: Watershed - VOC (full): {}".format(wa_voc_full))
print("Post-process: Watershed - DET (full): {}".format(wa_det_full))
print("Post-process: Spurious + Watershed + Z-Filtering - Test IoU (full): {}".format(spu_wa_zfil_score_full))
print("Post-process: Spurious + Watershed + Z-Filtering - Test VOC (full): {}".format(spu_wa_zfil_voc_full))
print("Post-process: Spurious + Watershed + Z-Filtering - Test DET (full): {}".format(spu_wa_zfil_det_full))

if not load_previous_weights:
    scores = {}
    for name in dir():
        if not name.startswith('__') and ("_per_crop" in name or "_50ov" in name\
        or "_per_image" in name or "_full" in name):
            scores[name] = eval(name)

    store_history(results, scores, time_callback, args.result_dir, job_identifier)
    create_plots(results, job_identifier, char_dir)

print("FINISHED JOB {} !!".format(job_identifier))

