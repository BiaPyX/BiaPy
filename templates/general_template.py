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

import datetime
import random
import numpy as np
import math
import time
import tensorflow as tf
from data_manipulation import load_and_prepare_2D_data, crop_data,\
                              merge_data_without_overlap,\
                              crop_data_with_overlap, merge_data_with_overlap, \
                              check_binary_masks, img_to_onehot_encoding
from generators.custom_da_gen import ImageDataGenerator
from generators.keras_da_gen import keras_da_generator, keras_gen_samples
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
from aux.callbacks import ModelCheckpoint
from post_processing import spuriuous_detection_filter, calculate_z_filtering,\
                            boundary_refinement_watershed2


############
#  CHECKS  #
############

now = datetime.datetime.now()
print("Date : {}".format(now.strftime("%Y-%m-%d %H:%M:%S")))
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
# Create the validation data with random images of the training data. If False
# the validation data will be the last portion of training images.
random_val_data = True


### Dataset shape
# Note: train and test dimensions must be the same when training the network and
# making the predictions. Be sure to take care of this if you are not going to
# use "crop_data()" with the arg force_shape, as this function resolves the 
# problem creating always crops of the same dimension
img_train_shape = (1024, 768, 1)
img_test_shape = (1024, 768, 1)


### Extra datasets variables
# Paths, shapes and discard values for the extra dataset used together with the
# main train dataset, provided by train_path and train_mask_path variables, to 
# train the network with. If the shape of the datasets differ the best option
# To normalize them is to make crops ("make_crops" variable)
extra_datasets_data_list = []
extra_datasets_mask_list = []
extra_datasets_data_dim_list = []
extra_datasets_discard = []
### Example of use:
# Path to the data:
# extra_datasets_data_list.append(os.path.join('kasthuri_pp', 'reshaped_fibsem', 'train', 'x'))
# Path to the mask: 
# extra_datasets_mask_list.append(os.path.join('kasthuri_pp', 'reshaped_fibsem', 'train', 'y'))
# Shape of the images:
# extra_datasets_data_dim_list.append((877, 967, 1))
# Discard value to apply in the dataset (see "Discard variables" for more details):
# extra_datasets_discard.append(0.05)                                             
#
# Number of crop to take form each dataset to train the network. If 0, the      
# variable will be ignored                                                      
num_crops_per_dataset = 0


### Crop variables
# Shape of the crops
crop_shape = (256, 256, 1)
# To make crops on the train data
make_crops = True
# To check the crops. Useful to ensure that the crops have been made 
# correctly. Note: if "discard_cropped_images" is True only the run that 
# prepare the discarded data will check the crops, as the future runs only load 
# the crops stored by this first run
check_crop = True 
# Instead of make the crops before the network training, this flag activates
# the option to extract a random crop of each train image during data 
# augmentation (with a crop shape defined by "crop_shape" variable). This flag
# is not compatible with "make_crops" variable
random_crops_in_DA = False 
test_ov_crops = 16 # Only active with random_crops_in_DA
probability_map = False # Only active with random_crops_in_DA                       
w_foreground = 0.94 # Only active with probability_map
w_background = 0.06 # Only active with probability_map


### Discard variables
# To activate the discards in the main train data. Only active when 
# "make_crops" variable is True
discard_cropped_images = False
# Percentage of pixels labeled with the foreground class necessary to not 
# discard the image 
d_percentage_value = 0.05
# Path where the train discarded data will be stored to be loaded by future runs 
# instead of make again the process
train_crop_discard_path = \
    os.path.join(args.result_dir, 'data_d', job_identifier 
                 + str(d_percentage_value), 'train', 'x')
# Path where the train discarded masks will be stored                           
train_crop_discard_mask_path = \
    os.path.join(args.result_dir, 'data_d', job_identifier 
                 + str(d_percentage_value), 'train', 'y')
# The discards are NOT done in the test data, but this will store the test data,
# which will be cropped, into the pointed path to be loaded by future runs      
# Together with the train discarded data and masks                              
test_crop_discard_path = \
    os.path.join(args.result_dir, 'data_d', job_identifier 
                 + str(d_percentage_value), 'test', 'x')
test_crop_discard_mask_path = \
    os.path.join(args.result_dir, 'data_d', job_identifier 
                 + str(d_percentage_value), 'test', 'y')


### Normalization
# To normalize the data dividing by the mean pixel value
normalize_data = False                                                          
# Force the normalization value to the given number instead of the mean pixel 
# value
norm_value_forced = -1                                                          


### Data augmentation (DA) variables
# To decide which type of DA implementation will be used. Select False to 
# use Keras API provided DA, otherwise, a custom implementation will be used
custom_da = False
# Create samples of the DA made. Useful to check the output images made. 
# This option is available for both Keras and custom DA
aug_examples = True 
# To shuffle the training data on every epoch:
# (Best options: Keras->False, Custom->True)
shuffle_train_data_each_epoch = custom_da
# To shuffle the validation data on every epoch:
# (Best option: False in both cases)
shuffle_val_data_each_epoch = False

### Options available for Keras Data Augmentation
# widtk_h_shift_range (more details in Keras ImageDataGenerator class)
k_w_shift_r = 0.0
# height_shift_range (more details in Keras ImageDataGenerator class)
k_h_shift_r = 0.0
# k_shear_range (more details in Keras ImageDataGenerator class)
k_shear_range = 0.0 
# Range to pick a brightness value from to apply in the images. Available in 
# Keras. Example of use: k_brightness_range = [1.0, 1.0]
k_brightness_range = None 

### Options available for Custom Data Augmentation 
# Histogram equalization
hist_eq = False  
# Elastic transformations
elastic = False
# Median blur                                                             
median_blur = False
# Gaussian blur
g_blur = False                                                                  
# Gamma contrast
gamma_contrast = False      

### Options available for both, Custom and Kera Data Augmentation
# Rotation of 90, 180 or 270
rotation90 = False
# Range of rotation. Set to 0 to disable it
rotation_range = 180
# To make vertical flips 
vflips = True
# To make horizontal flips
hflips = True
# Make a bit of zoom in the images
zoom = False


### Extra train data generation
# Number of times to duplicate the train data. Useful when "random_crops_in_DA"
# is made, as more original train data can be cover
replicate_train = 0
# Extra number of images to add to the train data. Applied after replicate_train 
extra_train_data = 0


### Load previously generated model weigths
# To activate the load of a previous training weigths instead of train 
# the network again
load_previous_weights = False
# ID of the previous experiment to load the weigths from 
previous_job_weights = args.job_id
# To activate the fine tunning
fine_tunning = False
# ID of the previous weigths to load the weigths from to make the fine tunning 
fine_tunning_weigths = args.job_id
# Prefix of the files where the weights are stored/loaded from
weight_files_prefix = 'model.fibsem_'
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
# To activate the Spatial Dropout instead of use the "normal" dropout layer
spatial_dropout = False
# Values to make the dropout with. It's dimension must be equal depth+1. Set to
# 0 to prevent dropout 
dropout_values = [0.1, 0.1, 0.2, 0.2, 0.3]
# To active batch normalization
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


print("##########################################\n"
      "#  PREPARE DATASET IF DISCARD IS ACTIVE  #\n"
      "##########################################\n")

# The first time the dataset will be prepared for future runs if it is not 
# created yet
if discard_cropped_images and make_crops \
   and not os.path.exists(train_crop_discard_path):
    # Load data
    X_train, Y_train, \
    X_test, Y_test, \
    orig_test_shape, norm_value, \
    crops_made = load_and_prepare_2D_data(
        train_path, train_mask_path, test_path, test_mask_path, img_train_shape, 
        img_test_shape, create_val=False, job_id=args.job_id, crop_shape=crop_shape, 
        check_crop=check_crop, check_crop_path=check_crop_path, 
        d_percentage=d_percentage_value)

    # Create folders and save the images for future runs 
    print("Saving cropped images for future runs . . .")
    save_img(X=X_train, data_dir=train_crop_discard_path, Y=Y_train,            
             mask_dir=train_crop_discard_mask_path)                             
    save_img(X=X_test, data_dir=test_crop_discard_path, Y=Y_test,               
             mask_dir=test_crop_discard_mask_path)

    del X_train, Y_train, X_test, Y_test
   
    # Update shapes 
    img_train_shape = crop_shape
    img_test_shape = crop_shape
    discard_made_run = True
else:
    discard_made_run = False

# Disable the crops if the run is not the one that have prepared the discarded 
# data as it will work with cropped images instead of the original ones, 
# rewriting the needed images 
if discard_cropped_images and discard_made_run == False:
    check_crop = False

# For the rest of runs that are not the first that prepares the dataset when 
# discard is active some variables must be set as if it would made the crops
if make_crops and discard_cropped_images:
    train_path = train_crop_discard_path
    train_mask_path = train_crop_discard_mask_path
    test_path = test_crop_discard_path
    test_mask_path = test_crop_discard_mask_path
    img_train_shape = crop_shape
    img_test_shape = crop_shape
    crops_made = True


print("###############\n"
      "#  LOAD DATA  #\n"
      "###############\n")

X_train, Y_train, X_val,\
Y_val, X_test, Y_test,\
orig_test_shape, norm_value, crops_made = load_and_prepare_2D_data(
    train_path, train_mask_path, test_path, test_mask_path, img_train_shape, 
    img_test_shape, val_split=perc_used_as_val, shuffle_val=random_val_data,
    e_d_data=extra_datasets_data_list, e_d_mask=extra_datasets_mask_list, 
    e_d_data_dim=extra_datasets_data_dim_list, e_d_dis=extra_datasets_discard, 
    num_crops_per_dataset=num_crops_per_dataset, make_crops=make_crops, 
    crop_shape=crop_shape, check_crop=check_crop, 
    check_crop_path=check_crop_path)

# Normalize the data
if normalize_data:
    if norm_value_forced != -1: 
        print("Forced normalization value to {}".format(norm_value_forced))
        norm_value = norm_value_forced
    else:
        print("Normalization value calculated: {}".format(norm_value))
    X_train -= int(norm_value)
    X_val -= int(norm_value)
    X_test -= int(norm_value)
    
# Crop the data to the desired size
if (make_crops and crops_made) or random_crops_in_DA:
    img_width = crop_shape[0]
    img_height = crop_shape[1]
    img_channels = crop_shape[2]
else:                                                                           
    img_width = img_train_shape[0]
    img_height = img_train_shape[1]                                               
    img_channels = img_train_shape[2]


print("###########################\n"
      "#  EXTRA DATA GENERATION  #\n"
      "###########################\n")

# Calculate the steps_per_epoch value to train in case
if replicate_train != 0:
    steps_per_epoch_value = int((replicate_train*X_train.shape[0])/batch_size_value)
    print("Data doubled by {} ; Steps per epoch = {}".format(replicate_train,
          steps_per_epoch_value))
else:
    steps_per_epoch_value = int(X_train.shape[0]/batch_size_value)

# Add extra train data generated with DA
if extra_train_data != 0:
    if custom_da == False:
        # Keras DA generated extra data

        extra_x, extra_y = keras_gen_samples(
            extra_train_data, X_data=X_train, Y_data=Y_train, 
            batch_size_value=batch_size_value, zoom=zoom, 
            k_w_shift_r=k_w_shift_r, k_h_shift_r=k_h_shift_r, k_shear_range=k_shear_range,
            k_brightness_range=k_brightness_range, rotation_range=rotation_range,
            vflip=vflips, hflip=hflips, median_filter_size=median_filter_size, 
            hist_eq=hist_eq, elastic=elastic, g_blur=g_blur, 
            gamma_contrast=gamma_contrast)
    else:
        # Custom DA generated extra data
        extra_gen_args = dict(
            X=X_train, Y=Y_train, batch_size=batch_size_value,
            shape=(img_height,img_width,img_channels), shuffle=True, da=True, 
            hist_eq=hist_eq, rotation90=rotation90, rotation_range=rotation_range,                   
            vflip=vflips, hflip=hflips, elastic=elastic, g_blur=g_blur,             
            median_blur=median_blur, gamma_contrast=gamma_contrast, zoom=zoom,                
            random_crops_in_DA=random_crops_in_DA)

        extra_generator = ImageDataGenerator(**extra_gen_args)

        extra_x, extra_y = extra_generator.get_transformed_samples(
            extra_train_data, force_full_images=True)

    X_train = np.vstack((X_train, extra_x))
    Y_train = np.vstack((Y_train, extra_y))
    print("{} extra train data generated, the new shape of the train now is {}"\
          .format(extra_train_data, X_train.shape))


print("#######################\n"
      "#  DATA AUGMENTATION  #\n"
      "#######################\n")

if custom_da == False:                                                          
    print("Keras DA selected")

    # Keras Data Augmentation                                                   
    train_generator, \
    val_generator = keras_da_generator(
        X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val, 
        batch_size_value=batch_size_value, save_examples=aug_examples,
        out_dir=da_samples_dir, shuffle_train=shuffle_train_data_each_epoch, 
        shuffle_val=shuffle_val_data_each_epoch, zoom=zoom, 
        rotation_range=rotation_range, random_crops_in_DA=random_crops_in_DA,
        crop_length=crop_shape[0], w_shift_r=k_w_shift_r, h_shift_r=k_h_shift_r,    
        shear_range=k_shear_range, brightness_range=k_brightness_range,
        weights_on_data=weights_on_data, weights_path=loss_weight_dir,
        vflip=vflips, hflip=hflips)
else:                                                                           
    print("Custom DA selected")

    # Calculate the probability map per image
    train_prob = None
    if probability_map:
        prob_map_file = os.path.join(prob_map_dir, 'prob_map.npy')
        if os.path.exists(prob_map_dir):
            train_prob = np.load(prob_map_file)
        else:
            train_prob = calculate_2D_volume_prob_map(
                Y_train, w_foreground, w_background, save_file=prob_map_file)

    # Custom Data Augmentation                                                  
    data_gen_args = dict(
        X=X_train, Y=Y_train, batch_size=batch_size_value,     
        shape=(img_height,img_width,img_channels),
        shuffle=shuffle_train_data_each_epoch, da=True, hist_eq=hist_eq,
        rotation90=rotation90, rotation_range=rotation_range,
        vflip=vflips, hflip=hflips, elastic=elastic, g_blur=g_blur,
        median_blur=median_blur, gamma_contrast=gamma_contrast, zoom=zoom,
        random_crops_in_DA=random_crops_in_DA, prob_map=probability_map, 
        train_prob=train_prob, softmax_out=softmax_out,
        extra_data_factor=replicate_train)
    data_gen_val_args = dict(
        X=X_val, Y=Y_val, batch_size=batch_size_value, 
        shape=(img_height,img_width,img_channels), 
        shuffle=shuffle_val_data_each_epoch, da=False, 
        random_crops_in_DA=random_crops_in_DA, val=True, softmax_out=softmax_out)
    train_generator = ImageDataGenerator(**data_gen_args)                       
    val_generator = ImageDataGenerator(**data_gen_val_args)                     
                                                                                
    # Generate examples of data augmentation                                    
    if aug_examples:                                                    
        train_generator.get_transformed_samples(
            10, save_to_dir=True, train=False, out_dir=da_samples_dir)


print("#################################\n"
      "#  BUILD AND TRAIN THE NETWORK  #\n"
      "#################################\n")

print("Creating the network . . .")
model = U_Net_2D([img_height, img_width, img_channels], activation=activation,
                 feature_maps=feature_maps, depth=depth, 
                 drop_values=dropout_values, spatial_dropout=spatial_dropout,
                 batch_norm=batch_normalization, k_init=kernel_init,
                 loss_type=loss_type, optimizer=optimizer, 
                 lr=learning_rate_value)

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
        results = model.fit(x=train_generator, validation_data=val_generator,
            validation_steps=math.ceil(X_val.shape[0]/batch_size_value),
            steps_per_epoch=math.ceil(X_train.shape[0]/batch_size_value),
            epochs=5, callbacks=[lr_finder])

        print("Finish LRFinder. Check the plot in {}".format(lrfinder_dir))
        sys.exit(0)
    else:
        results = model.fit(train_generator, validation_data=val_generator,
            validation_steps=math.ceil(X_val.shape[0]/batch_size_value),
            steps_per_epoch=steps_per_epoch_value, epochs=epochs_value,
            callbacks=[earlystopper, checkpointer, time_callback])
else:
    h5_file=os.path.join(h5_dir, weight_files_prefix + previous_job_weights 
                         + '_' + str(args.run_id) + '.h5')
    print("Loading model weights from h5_file: {}".format(h5_file))
    model.load_weights(h5_file)


print("################################\n"
      "#  PREPARE DATA FOR INFERENCE  #\n"
      "################################\n")

# Prepare test data for its use
if np.max(Y_test) > n_classes:
    Y_test = Y_test.astype('float32')
    Y_test *= 1./255
if np.max(X_test) > 2:
    X_test = X_test.astype('float32')
    X_test *= 1./255

if softmax_out:
    Y_test_one_hot = np.zeros(Y_test.shape[:3] + (2,))
    for i in range(Y_test.shape[0]):
        Y_test_one_hot[i] = np.asarray(img_to_onehot_encoding(Y_test[i], n_classes))
    Y_test = Y_test_one_hot


print("##########################\n"
      "#  INFERENCE (per crop)  #\n"
      "##########################\n")

if random_crops_in_DA:
    X_test, Y_test = crop_data_with_overlap(
        X_test, Y_test, crop_shape[0], test_ov_crops)

print("Evaluating test data . . .")
score_per_crop = model.evaluate(
    X_test, Y_test, batch_size=batch_size_value, verbose=1)
loss_per_crop = score_per_crop[0]
jac_per_crop = score_per_crop[1]

print("Making the predictions on test data . . .")
preds_test = model.predict(X_test, batch_size=batch_size_value, verbose=1)

if softmax_out:
    preds_test = np.expand_dims(preds_test[...,1], -1)
    Y_test = np.expand_dims(Y_test[...,1], -1)


print("########################################\n"
      "#  Metrics (per image, merging crops)  #\n"
      "########################################\n")

# Merge crops
if make_crops or (random_crops_in_DA and test_ov_crops == 1):
    h_num = math.ceil(orig_test_shape[1]/preds_test.shape[1])
    v_num = math.ceil(orig_test_shape[2]/preds_test.shape[2])

    print("Reconstruct preds_test . . .")
    preds_test = merge_data_without_overlap(
        preds_test, math.ceil(preds_test.shape[0]/(h_num*v_num)),
        out_shape=[h_num, v_num], grid=False)
    print("Reconstruct X_test . . .")
    X_test = merge_data_without_overlap(
        X_test, math.ceil(X_test.shape[0]/(h_num*v_num)),
        out_shape=[h_num, v_num], grid=False)
    print("Reconstruct Y_test . . .")
    Y_test = merge_data_without_overlap(
        Y_test, math.ceil(Y_test.shape[0]/(h_num*v_num)),
        out_shape=[h_num, v_num], grid=False)
elif random_crops_in_DA and test_ov_crops > 1:
    print("Reconstruct X_test . . .")
    X_test = merge_data_with_overlap(
        X_test, orig_test_shape, crop_shape[0], test_ov_crops)
    print("Reconstruct Y_test . . .")
    Y_test = merge_data_with_overlap(
        Y_test, orig_test_shape, crop_shape[0], test_ov_crops)
    print("Reconstruct preds_test . . .")
    preds_test = merge_data_with_overlap(
        preds_test, orig_test_shape, crop_shape[0], test_ov_crops)

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
for i in tqdm(range(X_test.shape[0])):
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
del spu_wa_zfil_preds_test, spu_preds_test


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

