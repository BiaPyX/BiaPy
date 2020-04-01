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
from data import load_data, crop_data, merge_data_without_overlap, check_crops,\
                 keras_da_generator, ImageDataGenerator, crop_data_with_overlap,\
                 merge_data_with_overlap, calculate_z_filtering,\
                 check_binary_masks
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
train_path = os.path.join('data', 'train', 'x')
train_mask_path = os.path.join('data', 'train', 'y')
test_path = os.path.join('data', 'test', 'x')
test_mask_path = os.path.join('data', 'test', 'y')
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
img_train_shape = [1024, 768, 1]
img_test_shape = [1024, 768, 1]
original_test_shape = [img_test_shape[0], img_test_shape[1]]

### Extra datasets variables
# Paths, shapes and discard values for the extra dataset used together with the
# main train dataset, provided by train_path and train_mask_path variables, to 
# train the network with. If the shape of the datasets differ the best option
# to normalize them is to make crops ("make_crops" variable)
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
# extra_datasets_data_dim_list.append([877, 967, 1])
# Discard value to apply in the dataset (see "Discard variables" for more details):
# extra_datasets_discard.append(0.05)                                             
#
# Number of crop to take form each dataset to train the network. If 0, the      
# variable will be ignored                                                      
num_crops_per_dataset = 0

### Crop variables
# Shape of the crops
crop_shape = [256, 256, 1]
# Flag to make crops on the train data
make_crops = True
# Flag to check the crops. Useful to ensure that the crops have been made 
# correctly. Note: if "discard_cropped_images" is True only the run that 
# prepare the discarded data will check the crops, as the future runs only load 
# the crops stored by this first run
check_crop = True 
# Instead of make the crops before the network training, this flag activates
# the option to extract a random crop of each train image during data 
# augmentation (with a crop shape defined by "crop_shape" variable). This flag
# is not compatible with "make_crops" variable
random_crops_in_DA = False 
# NEEDED CODE REFACTORING OF THIS SECTION
test_ov_crops = 8 # Only active with random_crops_in_DA
probability_map = False # Only active with random_crops_in_DA                       
w_foreground = 0.94 # Only active with probability_map
w_background = 0.06 # Only active with probability_map

### Discard variables
# Flag to activate the discards in the main train data. Only active when 
# "make_crops" variable is True
discard_cropped_images = False
# Percentage of pixels labeled with the foreground class necessary to not 
# discard the image 
d_percentage_value = 0.05
# Path where the train discarded data will be stored to be loaded by future runs 
# instead of make again the process
train_crop_discard_path = os.path.join('data_d', job_id + str(d_percentage_value), job_file, 'train', 'x')
# Path where the train discarded masks will be stored                           
train_crop_discard_mask_path = os.path.join('data_d', job_id + str(d_percentage_value), job_file, 'train', 'y')
# The discards are NOT done in the test data, but this will store the test data,
# which will be cropped, into the pointed path to be loaded by future runs      
# together with the train discarded data and masks                              
test_crop_discard_path = os.path.join('data_d', job_id + str(d_percentage_value), job_file, 'test', 'x')
test_crop_discard_mask_path = os.path.join('data_d', job_id + str(d_percentage_value), job_file, 'test', 'y')

### Normalization
# Flag to normalize the data dividing by the mean pixel value
normalize_data = True
# Force the normalization value to the given number instead of the mean pixel 
# value
norm_value_forced = -1                                                          

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

### Extra train data generation
# Number of times to duplicate the train data. Useful when "random_crops_in_DA"
# is made, as more original train data can be cover
duplicate_train = 0
# Extra number of images to add to the train data. Applied after duplicate_train 
extra_train_data = 0

### Load previously generated model weigths
# Flag to activate the load of a previous training weigths instead of train 
# the network again
load_previous_weights = False
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
batch_size_value = 6
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
if extra_datasets_mask_list: 
    for i in range(len(extra_datasets_mask_list)):
        check_binary_masks(extra_datasets_mask_list[i])


#############################################
#    PREPARE DATASET IF DISCARD IS ACTIVE   #
#############################################

# The first time the dataset will be prepared for future runs if it is not 
# created yet
if discard_cropped_images == True and make_crops == True \
   and not os.path.exists(train_crop_discard_path):

    Print("##################\n#  DISCARD DATA  #\n##################\n") 

    # Load data
    X_train, Y_train, \
    X_test, Y_test, \
    norm_value, crops_made = load_data(train_path, train_mask_path, test_path,
                           test_mask_path, img_train_shape, img_test_shape,
                           create_val=False, job_id=job_id, 
                           crop_shape=crop_shape, check_crop=check_crop, 
                           d_percentage=d_percentage_value)

    # Create folders and save the images for future runs 
    Print("Saving cropped images for future runs . . .")
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
if discard_cropped_images == True and discard_made_run == False:
    check_crop = False

# For the rest of runs that are not the first that prepares the dataset when 
# discard is active some variables must be set as if it would made the crops
if make_crops == True and discard_cropped_images == True:
    train_path = train_crop_discard_path
    train_mask_path = train_crop_discard_mask_path
    test_path = test_crop_discard_path
    test_mask_path = test_crop_discard_mask_path
    img_train_shape = crop_shape
    img_test_shape = crop_shape
    crops_made = True


##########################                                                      
#       LOAD DATA        #                                                      
##########################

Print("##################\n#    LOAD DATA   #\n##################\n")

X_train, Y_train, \
X_val, Y_val, \
X_test, Y_test, \
norm_value, crops_made = load_data(train_path, train_mask_path, test_path,
                           test_mask_path, img_train_shape, img_test_shape,
                           val_split=perc_used_as_val, 
                           shuffle_val=random_val_data,
                           e_d_data=extra_datasets_data_list, job_id=job_id, 
                           e_d_mask=extra_datasets_mask_list,
                           e_d_data_dim=extra_datasets_data_dim_list,
                           e_d_dis=extra_datasets_discard,
                           num_crops_per_dataset=num_crops_per_dataset,
                           crop_shape=crop_shape, check_crop=check_crop)

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
    
# Crop the data to the desired size
if make_crops == True and crops_made == True:
    img_width = crop_shape[0]
    img_height = crop_shape[1]
    img_channels = crop_shape[2]
else:                                                                           
    img_width = img_train_shape[0]
    img_height = img_train_shape[1]                                               
    img_channels = img_train_shape[2]


#############################
#   EXTRA DATA GENERATION   #
#############################

# Duplicate train data N times
if duplicate_train != 0:
    Print("##################\n# DUPLICATE DATA #\n##################\n")

    X_train = np.vstack([X_train]*duplicate_train)
    Y_train = np.vstack([Y_train]*duplicate_train)
    Print("Train data replicated " + str(duplicate_train) + " times. Its new "
          + "shape is: " + str(X_train.shape))

# Add extra train data generated with DA
if extra_train_data != 0:
    Print("##################\n#   EXTRA DATA   #\n##################\n")

    if custom_da == False:
        # Keras DA generated extra data
        _, extra_x, extra_y = keras_da_generator(X_train=X_train, Y_train=Y_train,
                                                 batch_size_value=batch_size_value, 
                                                 val=False, job_id=job_id,
                                                 shuffle_train=True,
                                                 random_crops_in_DA=random_crops_in_DA,
                                                 crop_length=crop_shape[0],
                                                 extra_train_data=extra_train_data)
    else:
        # Custom DA generated extra data
        extra_gen_args = dict(X=X_train, Y=Y_train, batch_size=batch_size_value,
                              dim=(img_height,img_width), n_channels=1,
                              shuffle=True, da=True, e_prob=0.0, elastic=False,
                              vflip=True, hflip=True, rotation90=False,
                              rotation_range=0, 
                              random_crops_in_DA=random_crops_in_DA,
                              crop_length=crop_shape[0])
        extra_generator = ImageDataGenerator(**extra_gen_args)

        extra_x, extra_y = extra_generator.get_transformed_samples(extra_train_data)

    X_train = np.vstack((X_train, extra_x))
    Y_train = np.vstack((Y_train, extra_y))
    Print(str(extra_train_data) + " extra train data generated, the new shape "
          + "of the train now is " + str(X_train.shape))


##########################
#    DATA AUGMENTATION   #
##########################

Print("##################\n#    DATA AUG    #\n##################\n")

if custom_da == False:                                                          
    Print("Keras DA selected")

    # Keras Data Augmentation                                                   
    train_generator, \
    val_generator = keras_da_generator(X_train=X_train, Y_train=Y_train, 
                                       batch_size_value=batch_size_value,
                                       X_val=X_val, Y_val=Y_val,
                                       save_examples=aug_examples, job_id=job_id,          
                                       shuffle_train=shuffle_train_data_each_epoch, 
                                       shuffle_val=shuffle_val_data_each_epoch, 
                                       zoom=keras_zoom,        
                                       random_crops_in_DA=random_crops_in_DA,
                                       crop_length=crop_shape[0],
                                       w_shift_r=w_shift_r, h_shift_r=h_shift_r,    
                                       shear_range=shear_range,
                                       brightness_range=brightness_range)
else:                                                                           
    Print("Custom DA selected")

    # Calculate the probability map per image
    train_prob = None
    if probability_map == True:
        train_prob = np.copy(Y_train[:,:,:,0])
        train_prob = np.float32(train_prob)

        Print("Calculating the probability map . . .")
        for i in range(train_prob.shape[0]):
            pdf = train_prob[i]
        
            # Remove artifacts connected to image border
            pdf = clear_border(pdf)

            foreground_pixels = (pdf == 1).sum()
            background_pixels = (pdf == 0).sum()

            pdf[np.where(pdf == 1.0)] = w_foreground/foreground_pixels
            pdf[np.where(pdf == 0.0)] = w_background/background_pixels
            pdf /= pdf.sum() # Necessary to get all probs sum 1
            train_prob[i] = pdf

    # Custom Data Augmentation                                                  
    data_gen_args = dict(X=X_train, Y=Y_train, batch_size=batch_size_value,     
                         dim=(img_height,img_width), n_channels=1,              
                         shuffle=shuffle_train_data_each_epoch, da=True, 
                         e_prob=0.0, elastic=False, vflip=True, hflip=True, 
                         rotation90=False, rotation_range=180, 
                         brightness_range=brightness_range, 
                         median_filter_size=median_filter_size,
                         random_crops_in_DA=random_crops_in_DA, 
                         crop_length=crop_shape[0], prob_map=probability_map,
                         train_prob=train_prob)                            
                                                                                
    data_gen_val_args = dict(X=X_val, Y=Y_val, batch_size=batch_size_value,     
                             dim=(img_height,img_width), n_channels=1,          
                             shuffle=shuffle_val_data_each_epoch, da=False,                           
                             random_crops_in_DA=random_crops_in_DA,                   
                             crop_length=crop_shape[0], val=True)              
                                                                                
    train_generator = ImageDataGenerator(**data_gen_args)                       
    val_generator = ImageDataGenerator(**data_gen_val_args)                     
                                                                                
    # Generate examples of data augmentation                                    
    if aug_examples == True:                                                    
        train_generator.get_transformed_samples(10, save_to_dir=True,           
                                                job_id=os.path.join(job_id, test_id))
                                                                                
if random_crops_in_DA == True:
    img_width = crop_shape[0]
    img_height = crop_shape[1]


##########################
#    BUILD THE NETWORK   #
##########################

Print("###################\n#  TRAIN PROCESS  #\n###################\n")

Print("Creating the network . . .")
model = U_Net([img_height, img_width, img_channels], 
              numInitChannels=num_init_channels, spatial_dropout=spatial_dropout,
              fixed_dropout=fixed_dropout_value)

if optimizer == "sgd":
    opt = keras.optimizers.SGD(lr=learning_rate_value, momentum=0.99, decay=0.0, 
                               nesterov=False)
elif optimizer == "adam":    
    opt = keras.optimizers.Adam(lr=learning_rate_value, beta_1=0.9, beta_2=0.999,
                                epsilon=None, decay=0.0, amsgrad=False)
else:
    Print("Error: optimizer value must be 'sgd' or 'adam'")
    sys.exit(0)

model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[jaccard_index])
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

if random_crops_in_DA == False:
    # Evaluate to obtain the loss value and the Jaccard index (per crop)
    Print("Evaluating test data . . .")
    score = model.evaluate(X_test, Y_test, batch_size=batch_size_value, 
                           verbose=1)
    jac_per_crop = score[1]

    # Predict on test
    Print("Making the predictions on test data . . .")
    preds_test = model.predict(X_test, batch_size=batch_size_value, verbose=1)

    # Threshold images
    bin_preds_test = (preds_test > 0.5).astype(np.uint8)

    # Reconstruct the data to the original shape
    if make_crops == True:
        h_num = int(original_test_shape[0] / bin_preds_test.shape[1]) \
                + (original_test_shape[0] % bin_preds_test.shape[1] > 0)
        v_num = int(original_test_shape[1] / bin_preds_test.shape[2]) \
                + (original_test_shape[1] % bin_preds_test.shape[2] > 0)

        Y_test = merge_data_without_overlap(Y_test, math.ceil(Y_test.shape[0]/(h_num*v_num)),
                                            out_shape=[h_num, v_num], grid=False)
        Print("The shape of the test data reconstructed is " + str(Y_test.shape))
        
        # To calculate metrics (binarized)
        recons_preds_test = merge_data_without_overlap(bin_preds_test,
                                                       math.ceil(bin_preds_test.shape[0]/(h_num*v_num)),
                                                       out_shape=[h_num, v_num], 
                                                       grid=False)

        # To save the probabilities (no binarized)
        recons_no_bin_preds_test = merge_data_without_overlap(preds_test*255,
                                                              math.ceil(preds_test.shape[0]/(h_num*v_num)),
                                                              out_shape=[h_num, v_num], 
                                                              grid=False)
        recons_no_bin_preds_test = recons_no_bin_preds_test.astype(float)/255
        
        Print("Saving predicted images . . .")
        save_img(Y=recons_preds_test, mask_dir=result_bin_dir,
                 prefix="test_out_bin")
        save_img(Y=recons_no_bin_preds_test, mask_dir=result_no_bin_dir,
                 prefix="test_out_no_bin")
    else:
        Print("Saving predicted images . . .")
        save_img(Y=bin_preds_test, mask_dir=result_bin_dir,
                 prefix="test_out_bin")
        save_img(Y=preds_test, mask_dir=result_no_bin_dir,
                 prefix="test_out_no_bin")

    # Metric calculation
    if make_threshold_plots == True:
        Print("Calculate metrics with different thresholds . . .")
        score[1], voc, det = threshold_plots(preds_test, Y_test, original_test_shape,
                                score, det_eval_ge_path, det_eval_path, det_bin,
                                n_dig, job_id, job_file, char_dir)
    else:
        Print("Calculate metrics . . .")
        if make_crops == True:
            score[1] = jaccard_index_numpy(Y_test, recons_preds_test)
            voc = voc_calculation(Y_test, recons_preds_test, score[1])
            det = DET_calculation(Y_test, recons_preds_test, det_eval_ge_path,
                                  det_eval_path, det_bin, n_dig, job_id)
        else:
            score[1] = jaccard_index_numpy(Y_test, bin_preds_test)
            voc = voc_calculation(Y_test, bin_preds_test, score[1])
            det = DET_calculation(Y_test, bin_preds_test, det_eval_ge_path,
                                  det_eval_path, det_bin, n_dig, job_id)

else:
    ov_X_test, ov_Y_test = crop_data_with_overlap(X_test, Y_test, crop_shape[0], 
                                                  test_ov_crops)

    if check_crop == True:
        save_img(X=ov_X_test, data_dir=result_dir, Y=ov_Y_test, 
                 mask_dir=result_dir, prefix="ov_crop")

    Print("Evaluating overlapped test data . . .")
    score = model.evaluate(ov_X_test, ov_Y_test, batch_size=batch_size_value,
                           verbose=1)

    Print("Making the predictions on overlapped test data . . .")
    preds_test = model.predict(ov_X_test, batch_size=batch_size_value, verbose=1)

    bin_preds_test = (preds_test > 0.5).astype(np.uint8)
 
    Print("Calculate Jaccard for test (per crop). . .")
    jac_per_crop = jaccard_index_numpy(ov_Y_test, bin_preds_test)
    
    # Save output images
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if len(sys.argv) > 1 and test_id == "1":
        Print("Saving predicted images . . .")
        save_img(Y=bin_preds_test, mask_dir=result_dir, prefix="test_out_ov_bin")

    if test_ov_crops > 1:
        Print("Merging the overlapped predictions . . .")
        merged_preds_test = merge_data_with_overlap(bin_preds_test,
                                                    original_test_shape,
                                                    crop_shape[0],
                                                    test_ov_crops,
                                                    result_dir)

        Print("Calculate Jaccard for test (per image with overlap calculated). . .")
        score[1] = jaccard_index_numpy(Y_test, merged_preds_test)
        
        voc = voc_calculation(Y_test, merged_preds_test, score[1])
        det = DET_calculation(Y_test, merged_preds_test, det_eval_ge_path,
                              det_eval_path, det_bin, n_dig, job_id)
    else:
        Print("As the number of overlapped crops created is 1, we will obtain " 
              + "the (per image) Jaccard value overlapping 4 tiles with the " 
              + "predict_img_with_overlap function")                              
                                                                                
        Y_test_smooth = np.zeros(X_test.shape, dtype=(np.uint8))                
        for i in tqdm(range(0,len(X_test))):                                    
            predictions_smooth = predict_img_with_overlap(                      
                X_test[i,:,:,:],                                                
                window_size=crop_shape[0],                                     
                subdivisions=2,  
                nb_classes=1,                                                   
                pred_func=(                                                     
                    lambda img_batch_subdiv: model.predict(img_batch_subdiv)    
                )                                                               
            )                                                                   
            Y_test_smooth[i] = (predictions_smooth > 0.5).astype(np.uint8)      
                                                                                
        score[1] = jaccard_index_numpy(Y_test, Y_test_smooth)                   
        voc = voc_calculation(Y_test, Y_test_smooth, score[1])
        det = DET_calculation(Y_test, Y_test_smooth, det_eval_ge_path,
                              det_eval_path, det_bin, n_dig, job_id)
        del Y_test_smooth                                                       

    
####################
#  POST-PROCESING  #
####################

Print("##################\n# POST-PROCESING #\n##################\n") 

Print("1) SMOOTH")
if (post_process == True and make_crops == True) or (random_crops_in_DA == True):
    Print("Post processing active . . .")

    if random_crops_in_DA == False and make_crops == True:
        X_test = merge_data_without_overlap(X_test, math.ceil(X_test.shape[0]/(h_num*v_num)),
                                            out_shape=[h_num, v_num], grid=False)
    
    Y_test_smooth = np.zeros(X_test.shape, dtype=(np.uint8))

    # Extract the number of digits to create the image names
    d = len(str(X_test.shape[0]))

    if not os.path.exists(smooth_dir):
        os.makedirs(smooth_dir)

    Print("Smoothing crops . . .")
    for i in tqdm(range(0,len(X_test))):
        predictions_smooth = predict_img_with_smooth_windowing(
            X_test[i,:,:,:],
            window_size=crop_shape[0],
            subdivisions=2,  
            nb_classes=1,
            pred_func=(
                lambda img_batch_subdiv: model.predict(img_batch_subdiv)
            )
        )
        Y_test_smooth[i] = (predictions_smooth > 0.5).astype(np.uint8)

        im = Image.fromarray(predictions_smooth[:,:,0]*255)
        im = im.convert('L')
        im.save(os.path.join(smooth_dir,"test_out_smooth_" + str(i).zfill(d) 
                                        + ".png"))

    # Metrics (Jaccard + VOC + DET)
    Print("Calculate metrics . . .")
    smooth_score = jaccard_index_numpy(Y_test, Y_test_smooth)
    smooth_voc = voc_calculation(Y_test, Y_test_smooth, smooth_score)
    smooth_det = DET_calculation(Y_test, Y_test_smooth, det_eval_ge_path,
                                 det_eval_post_path, det_bin, n_dig, job_id)

zfil_preds_test = None
smooth_zfil_preds_test = None
if post_process == True and not extra_datasets_data_list:
    Print("2) Z-FILTERING")

    if random_crops_in_DA == False:
        Print("Applying Z-filter . . .")
        zfil_preds_test = calculate_z_filtering(recons_preds_test)
    else:
        if test_ov_crops > 1:
            Print("Applying Z-filter . . .")
            zfil_preds_test = calculate_z_filtering(merged_preds_test)

    if zfil_preds_test is not None:
        Print("Saving Z-filtered images . . .")
        save_img(Y=zfil_preds_test, mask_dir=zfil_dir, prefix="test_out_zfil")
 
        Print("Calculate metrics for the Z-filtered data . . .")
        zfil_score = jaccard_index_numpy(Y_test, zfil_preds_test)
        zfil_voc = voc_calculation(Y_test, zfil_preds_test, zfil_score)
        zfil_det = DET_calculation(Y_test, zfil_preds_test, det_eval_ge_path,
                                   det_eval_post_path, det_bin, n_dig, job_id)

    if Y_test_smooth is not None:
        Print("Applying Z-filter to the smoothed data . . .")
        smooth_zfil_preds_test = calculate_z_filtering(Y_test_smooth)

        Print("Saving smoothed + Z-filtered images . . .")
        save_img(Y=smooth_zfil_preds_test, mask_dir=smoo_zfil_dir, 
                 prefix="test_out_smoo_zfil")

        Print("Calculate metrics for the smoothed + Z-filtered data . . .")
        smo_zfil_score = jaccard_index_numpy(Y_test, smooth_zfil_preds_test)
        smo_zfil_voc = voc_calculation(Y_test, smooth_zfil_preds_test,
                                       smo_zfil_score)
        smo_zfil_det = DET_calculation(Y_test, smooth_zfil_preds_test,
                                       det_eval_ge_path, det_eval_post_path,
                                       det_bin, n_dig, job_id)

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
    
if random_crops_in_DA == False:    
    Print("Test (per crop) jaccard_index: " + str(jac_per_crop))
    Print("Test (per image) jaccard_index: " + str(score[1]))
    Print("VOC: " + str(voc))
    Print("DET: " + str(det))
else:
    Print("Test overlapped (per crop) jaccard_index: " + str(jac_per_crop))
    Print("Test overlapped (per image) jaccard_index: " + str(score[1]))
    if test_ov_crops > 1:
        Print("VOC: " + str(voc))
        Print("DET: " + str(det))
    
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

    store_history(results, jac_per_crop, score, voc, det, time_callback, log_dir,
                  job_file, smooth_score, smooth_voc, smooth_det, zfil_score,
                  zfil_voc, zfil_det, smo_zfil_score, smo_zfil_voc, smo_zfil_det)

    create_plots(results, job_id, test_id, char_dir)

if (post_process == True and make_crops == True) or (random_crops_in_DA == True):
    Print("Post-process: SMOOTH - Test jaccard_index: " + str(smooth_score))
    Print("Post-process: SMOOTH - VOC: " + str(smooth_voc))
    Print("Post-process: SMOOTH - DET: " + str(smooth_det))

if post_process == True and zfil_preds_test is not None:
    Print("Post-process: Z-filtering - Test jaccard_index: " + str(zfil_score))
    Print("Post-process: Z-filtering - VOC: " + str(zfil_voc))
    Print("Post-process: Z-filtering - DET: " + str(zfil_det))

if post_process == True and smooth_zfil_preds_test is not None:
    Print("Post-process: SMOOTH + Z-filtering - Test jaccard_index: "
          + str(smo_zfil_score))
    Print("Post-process: SMOOTH + Z-filtering - VOC: " + str(smo_zfil_voc))
    Print("Post-process: SMOOTH + Z-filtering - DET: " + str(smo_zfil_det))

Print("FINISHED JOB " + job_file + " !!")
