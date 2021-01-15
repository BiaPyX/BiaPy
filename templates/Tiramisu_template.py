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
                 calculate_2D_volume_prob_map, check_masks, \
                 img_to_onehot_encoding
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
from data_2D_manipulation import load_and_prepare_2D_data, crop_data_with_overlap,\
                                 merge_data_with_overlap
from generators.custom_da_gen import ImageDataGenerator
from generators.keras_da_gen import keras_da_generator, keras_gen_samples
from networks.fc_densenet103 import FC_DenseNet103
from metrics import jaccard_index_numpy, voc_calculation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from PIL import Image
from tqdm import tqdm
from smooth_tiled_predictions import predict_img_with_smooth_windowing
from tensorflow.keras.utils import plot_model
from aux.callbacks import ModelCheckpoint
from post_processing import spuriuous_detection_filter, calculate_z_filtering,\
                            boundary_refinement_watershed2,\
                            ensemble8_2d_predictions


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


### Data shape
# Note: train and test dimensions must be the same when training the network and
# making the predictions. Be sure to take care of this if you are not going to
# use "crop_data_with_overlap()" with the arg force_shape, as this function 
# resolves the problem creating always crops of the same dimension
img_train_shape = (1024, 768, 1)
img_test_shape = (1024, 768, 1)


### Crop variables
# Shape of the crops
crop_shape = (256, 256, 1)
# To make crops on the train data
make_crops = True
# To check the crops. Useful to ensure that the crops have been made correctly
check_crop = True 
# Percentage of overlap in (x, y) when cropping in test. Set to 0 to calculate
# the minimun overlap
overlap = (0,0)
# Use 'overlap' also in training data. If False overlap in train will be
# minimum: (0,0)
ov_train = False
# Instead of make the crops before the network training, this flag activates
# the option to extract a random crop of each train image during data 
# augmentation (with a crop shape defined by "crop_shape" variable). This flag
# is not compatible with "make_crops" variable
random_crops_in_DA = False 
# To extract the ramdom crop based on a probability map given. Only active with
# random_crops_in_DA and custom_da
probability_map = False
# Foreground weight to create the probability map
w_foreground = 0.94 
# Background weight to create the probability map
w_background = 0.06


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
# Range for random zoom
zoom = 0.0


### Extra train data generation
# Number of times to duplicate the train data. Useful when "random_crops_in_DA"
# is made, as more original train data can be cover
replicate_train = 0


### Load previously generated model weigths
# To activate the load of a previous training weigths instead of train 
# the network again
load_previous_weights = False
# ID of the previous experiment to load the weigths from 
previous_job_weights = args.job_id
# Prefix of the files where the weights are stored/loaded from
weight_files_prefix = 'model.fibsem_'


### Experiment main parameters
# Loss type, three options: "bce" or "w_bce_dice", which refers to binary cross 
# entropy (BCE) and BCE and Dice with with a weight term on each one (that must 
# sum 1) to calculate the total loss value. NOTE: "w_bce" is not implemented on 
# this template type: please use big_data_template.py instead.
loss_type = "bce"
# Batch size value
batch_size_value = 4
# Optimizer to use. Possible values: "sgd" or "adam"
optimizer = "adam"
# Learning rate used by the optimization method
learning_rate_value = 0.0005
# Number of epochs to train the network
epochs_value = 360
# Number of epochs to stop the training process after no improvement
patience = 200
# If weights on data are going to be applied. To true when loss_type is 'w_bce' 
weights_on_data = True if loss_type == "w_bce" else False


### Network architecture specific parameters
# Number of feature maps for the first conv layer 
n_filters_first_conv = 48
# Number of layers on each downsampling
n_layers_per_block = [4,5,7,10,12,15,12,10,7,5,4]
# Depth of the network                                                          
depth = 5
# Number of new feature maps created by each layer in a dense block
growth_rate = 16
# Values to make the dropout with. It's dimension must be equal depth+1. Set to
# 0 to prevent dropout 
dropout_value = 0.2
# Number of classes. To generate data with more than 1 channel custom DA need to
# be selected. It can be 1 or 2.                                                                   
n_classes = 1
# Adjust the metric used accordingly to the number of clases. This code is planned 
# to be used in a binary classification problem, so the function 'jaccard_index_softmax' 
# will only calculate the IoU for the foreground class (channel 1)
metric = "jaccard_index_softmax" if n_classes > 1 else "jaccard_index"
# To take only the last class of the predictions, which corresponds to the
# foreground in a binary problem. If n_classes > 2 this should be disabled to
# ensure all classes are preserved
last_class = True if n_classes <= 2 else False


### Paths of the results                                             
result_dir = os.path.join(args.result_dir, 'results', job_identifier)

# per-image directories
result_bin_dir_per_image = os.path.join(result_dir, 'per_image_binarized')
result_no_bin_dir_per_image = os.path.join(result_dir, 'per_image_no_binarized')
smo_bin_dir_per_image = os.path.join(result_dir, 'per_image_smooth')
smo_no_bin_dir_per_image = os.path.join(result_dir, 'per_image_smooth_no_bin')
zfil_dir_per_image = os.path.join(result_dir, 'per_image_zfil')
smo_zfil_dir_per_image = os.path.join(result_dir, 'per_image_smo_zfil')

# 50% overlap directories
result_bin_dir_50ov = os.path.join(result_dir, '50ov_binarized')
result_no_bin_dir_50ov = os.path.join(result_dir, '50ov_no_binarized')
ens_bin_dir_50ov = os.path.join(result_dir, '50ov_8ensemble_binarized')
ens_no_bin_dir_50ov = os.path.join(result_dir, '50ov_8ensemble_no_binarized')
ens_zfil_dir_50ov = os.path.join(result_dir, '50ov_8ensemble_zfil')

# Full image directories
result_bin_dir_full = os.path.join(result_dir, 'full_binarized')
result_no_bin_dir_full = os.path.join(result_dir, 'full_no_binarized')
smo_bin_dir_full = os.path.join(result_dir, 'full_8ensemble')
smo_no_bin_dir_full = os.path.join(result_dir, 'full_8ensemble')
zfil_dir_full = os.path.join(result_dir, 'full_zfil')
spu_dir_full = os.path.join(result_dir, 'full_spu')
wa_debug_dir_full = os.path.join(result_dir, 'full_watershed_debug')
wa_dir_full = os.path.join(result_dir, 'full_watershed')
spu_wa_zfil_wa_debug_dir = os.path.join(result_dir, 'full_wa_spu_zfil_wa_debug')
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


print("###################\n"
      "#  SANITY CHECKS  #\n"
      "###################\n")

check_masks(train_mask_path)
check_masks(test_mask_path)


print("###############\n"
      "#  LOAD DATA  #\n"
      "###############\n")

X_train, Y_train, X_val,\
Y_val, X_test, Y_test,\
orig_test_shape, norm_value, crops_made = load_and_prepare_2D_data(
    train_path, train_mask_path, test_path, test_mask_path, img_train_shape, 
    img_test_shape, val_split=perc_used_as_val, shuffle_val=random_val_data,
    make_crops=make_crops, crop_shape=crop_shape, ov=overlap,
    overlap_train=ov_train, check_crop=check_crop,
    check_crop_path=check_crop_path)

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

# Calculate the steps_per_epoch value to train in case we need to increase the 
# train data samples by multiplying the number of images with 'replicate_train'
if replicate_train != 0:
    steps_per_epoch_value = int((replicate_train*X_train.shape[0])/batch_size_value)
    print("Data doubled by {} ; Steps per epoch = {}".format(replicate_train,
          steps_per_epoch_value))
else:
    steps_per_epoch_value = int(X_train.shape[0]/batch_size_value)


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
        train_prob=train_prob, n_classes=n_classes,
        extra_data_factor=replicate_train)
    data_gen_val_args = dict(
        X=X_val, Y=Y_val, batch_size=batch_size_value, 
        shape=(img_height,img_width,img_channels), 
        shuffle=shuffle_val_data_each_epoch, da=False, 
        random_crops_in_DA=random_crops_in_DA, val=True, n_classes=n_classes)
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
model = FC_DenseNet103(
    [img_height, img_width, img_channels], 
    n_filters_first_conv=n_filters_first_conv, n_pool=depth, 
    growth_rate=growth_rate, n_layers_per_block=n_layers_per_block, 
    dropout_p=dropout_value, loss_type=loss_type, optimizer=optimizer, 
    lr=learning_rate_value)

# Check the network created
model.summary(line_length=150)
os.makedirs(char_dir, exist_ok=True)
model_name = os.path.join(char_dir, "model_plot_" + job_identifier + ".png")
plot_model(model, to_file=model_name, show_shapes=True, show_layer_names=True)

h5_file=os.path.join(h5_dir, weight_files_prefix + previous_job_weights     
                     + '_' + str(args.run_id) + '.h5')  

if load_previous_weights == False:
    results = model.fit(train_generator, validation_data=val_generator,
        validation_steps=math.ceil(X_val.shape[0]/batch_size_value),
        steps_per_epoch=steps_per_epoch_value, epochs=epochs_value,
        callbacks=[earlystopper, checkpointer, time_callback])

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

if n_classes > 1:
    Y_test_one_hot = np.zeros(Y_test.shape[:3] + (n_classes,))
    for i in range(Y_test.shape[0]):
        Y_test_one_hot[i] = np.asarray(img_to_onehot_encoding(Y_test[i], n_classes))
    Y_test = Y_test_one_hot
    del Y_test_one_hot


print("##########################\n"
      "#  INFERENCE (per crop)  #\n"
      "##########################\n")

if random_crops_in_DA:
    print("Cropping test data to have the same shape as the used during training")
    X_test, Y_test = crop_data_with_overlap(
        X_test, crop_shape, data_mask=Y_test, overlap=overlap)

print("Evaluating test data . . .")
score_per_crop = model.evaluate(
    X_test, Y_test, batch_size=batch_size_value, verbose=1)
loss_per_crop = score_per_crop[0]
jac_per_crop = score_per_crop[1]

print("Making the predictions on test data . . .")
preds_test = model.predict(X_test, batch_size=batch_size_value, verbose=1)

# Take only the foreground class                                                
if n_classes > 1:                                                               
    preds_test = np.expand_dims(preds_test[...,1], -1)                          
    Y_test = np.expand_dims(Y_test[...,1], -1) 


print("########################################\n"
      "#  Metrics (per image, merging crops)  #\n"
      "########################################\n")

# Merge crops
print("Reconstruct X_test/Y_test . . .")
X_test, Y_test = merge_data_with_overlap(
    X_test, orig_test_shape, Y_test, overlap=overlap)

print("Reconstruct preds_test . . .")
preds_test = merge_data_with_overlap(preds_test, orig_test_shape, overlap=overlap)

print("Saving predicted images . . .")
save_img(Y=(preds_test > 0.5).astype(np.uint8),
         mask_dir=result_bin_dir_per_image, prefix="test_out_bin")
save_img(Y=preds_test, mask_dir=result_no_bin_dir_per_image,
         prefix="test_out_no_bin")

print("Calculate metrics (per image) . . .")
jac_per_image = jaccard_index_numpy(Y_test, (preds_test > 0.5).astype(np.uint8))
voc_per_image = voc_calculation(
    Y_test, (preds_test > 0.5).astype(np.uint8), jac_per_image)

print("~~~~ Smooth (per image) ~~~~")
Y_test_smooth = np.zeros(X_test.shape, dtype=np.float32)
for i in tqdm(range(X_test.shape[0])):
    predictions_smooth = predict_img_with_smooth_windowing(
        X_test[i], window_size=crop_shape[0], subdivisions=2, n_classes=n_classes,
        pred_func=(lambda img_batch_subdiv: model.predict(img_batch_subdiv)))
    if n_classes > 1:
        Y_test_smooth[i] = np.expand_dims(predictions_smooth[...,1], axis=-1)
    else:
        Y_test_smooth[i] = predictions_smooth

print("Saving smooth predicted images . . .")
save_img(Y=Y_test_smooth, mask_dir=smo_no_bin_dir_per_image,
         prefix="test_out_smo_no_bin")
save_img(Y=(Y_test_smooth > 0.5).astype(np.uint8), mask_dir=smo_bin_dir_per_image,
         prefix="test_out_smo")

print("Calculate metrics (smooth + per crop) . . .")
smo_jac_per_image = jaccard_index_numpy(
    Y_test, (Y_test_smooth > 0.5).astype(np.uint8))
smo_voc_per_image = voc_calculation(
    Y_test, (Y_test_smooth > 0.5).astype(np.uint8), smo_jac_per_image)

print("~~~~ Z-Filtering (per image) ~~~~")
zfil_preds_test = calculate_z_filtering(preds_test)

print("Saving Z-filtered images . . .")
save_img(Y=zfil_preds_test, mask_dir=zfil_dir_per_image, prefix="test_out_zfil")

print("Calculate metrics (Z-filtering + per crop) . . .")
zfil_jac_per_image = jaccard_index_numpy(
    Y_test, (zfil_preds_test > 0.5).astype(np.uint8))
zfil_voc_per_image = voc_calculation(
    Y_test, (zfil_preds_test > 0.5).astype(np.uint8), zfil_jac_per_image)
del zfil_preds_test, preds_test

print("~~~~ Smooth + Z-Filtering (per image) ~~~~")
smo_zfil_preds_test = calculate_z_filtering(Y_test_smooth)

print("Saving smoothed + Z-filtered images . . .")
save_img(Y=smo_zfil_preds_test, mask_dir=smo_zfil_dir_per_image,
         prefix="test_out_smoo_zfil")

print("Calculate metrics (Smooth + Z-filtering per crop) . . .")
smo_zfil_jac_per_image = jaccard_index_numpy(
    Y_test, (smo_zfil_preds_test > 0.5).astype(np.uint8))
smo_zfil_voc_per_image = voc_calculation(
    Y_test, (smo_zfil_preds_test > 0.5).astype(np.uint8),
    smo_zfil_jac_per_image)
del Y_test_smooth, smo_zfil_preds_test


print("############################################################\n"
      "#  Metrics (per image, merging crops with 50% of overlap)  #\n"
      "############################################################\n")

print("Making the predictions on test data (50% overlap) . . .")                
X_test = crop_data_with_overlap(X_test, crop_shape, overlap=(0.5, 0.5))         
                                                                                
Y_test_50ov = model.predict(X_test, batch_size=batch_size_value, verbose=1)     
# Take only the foreground class                                                
if n_classes > 1:                                                               
    Y_test_50ov = np.expand_dims(Y_test_50ov[...,1], -1)                        
                                                                                
Y_test_50ov = merge_data_with_overlap(                                          
    Y_test_50ov, orig_test_shape, overlap=(0.5, 0.5))  

print("Saving 50% overlap predicted images . . .")
save_img(Y=(Y_test_50ov > 0.5).astype(np.uint8),
         mask_dir=result_bin_dir_50ov, prefix="test_out_bin_50ov")
save_img(Y=Y_test_50ov, mask_dir=result_no_bin_dir_50ov,
         prefix="test_out_no_bin_50ov")

print("Calculate metrics (50% overlap) . . .")
jac_50ov = jaccard_index_numpy(Y_test, (Y_test_50ov > 0.5).astype(np.float32))
voc_50ov = voc_calculation(
    Y_test, (Y_test_50ov > 0.5).astype(np.float32), jac_50ov)
del Y_test_50ov

print("~~~~ 8-Ensemble (50% overlap) ~~~~")
Y_test_50ov_ensemble = np.zeros(X_test.shape, dtype=(np.float32))
for i in tqdm(range(X_test.shape[0])):
    pred_ensembled = ensemble8_2d_predictions(X_test[i],
        pred_func=(lambda img_batch_subdiv: model.predict(img_batch_subdiv)),
        n_classes=n_classes, last_class=last_class)
    Y_test_50ov_ensemble[i] = pred_ensembled
Y_test_50ov_ensemble = merge_data_with_overlap(
    Y_test_50ov_ensemble, orig_test_shape, overlap=(0.5, 0.5))

print("Saving 50% overlap + 8-Ensemble predicted images . . .")
save_img(Y=(Y_test_50ov_ensemble > 0.5).astype(np.uint8),
         mask_dir=ens_bin_dir_50ov, prefix="test_out_bin_ens8_50ov")
save_img(Y=Y_test_50ov_ensemble, mask_dir=ens_no_bin_dir_50ov,
         prefix="test_out_no_bin_ens8_50ov")

print("Calculate metrics (50% overlap + 8-Ensemble) . . .")
ens_jac_50ov = jaccard_index_numpy(
    Y_test, (Y_test_50ov_ensemble > 0.5).astype(np.float32))
ens_voc_50ov = voc_calculation(
    Y_test, (Y_test_50ov_ensemble > 0.5).astype(np.float32), jac_50ov)

print("~~~~ 8-Ensemble + Z-Filtering (50% overlap) ~~~~")
zfil_preds_test = calculate_z_filtering(Y_test_50ov_ensemble)

print("Saving Z-filtered images . . .")
save_img(Y=zfil_preds_test, mask_dir=ens_zfil_dir_50ov, prefix="test_out_zfil")

print("Calculate metrics (50% overlap + 8-Ensemble + Z-filtering) . . .")
ens_zfil_jac_50ov = jaccard_index_numpy(
    Y_test, (zfil_preds_test > 0.5).astype(np.uint8))
ens_zfil_voc_50ov = voc_calculation(
    Y_test, (zfil_preds_test > 0.5).astype(np.uint8), ens_zfil_jac_50ov)
del Y_test_50ov_ensemble, zfil_preds_test

# Merge X_test 
X_test = merge_data_with_overlap(X_test, orig_test_shape, overlap=(0.5,0.5))      


print("########################\n"
      "# Metrics (full image) #\n"
      "########################\n")

print("Making the predictions on test data . . .")
preds_test_full = model.predict(X_test, batch_size=batch_size_value, verbose=1)

if n_classes > 1:
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

print("~~~~ 8-Ensemble (full image) ~~~~")
Y_test_ensemble = np.zeros(X_test.shape, dtype=(np.float32))

for i in tqdm(range(X_test.shape[0])):
    pred_ensembled = ensemble8_2d_predictions(X_test[i],
        pred_func=(lambda img_batch_subdiv: model.predict(img_batch_subdiv)),
        n_classes=n_classes, last_class=last_class)
    Y_test_ensemble[i] = pred_ensembled

print("Saving smooth predicted images . . .")
save_img(Y=Y_test_ensemble, mask_dir=smo_no_bin_dir_full,
         prefix="test_out_ens_no_bin")
save_img(Y=(Y_test_ensemble > 0.5).astype(np.uint8), mask_dir=smo_bin_dir_full,
         prefix="test_out_ens")

print("Calculate metrics (8-Ensemble + full image) . . .")
smo_jac_full = jaccard_index_numpy(
    Y_test, (Y_test_ensemble > 0.5).astype(np.uint8))
smo_voc_full = voc_calculation(
    Y_test, (Y_test_ensemble > 0.5).astype(np.uint8), smo_jac_full)

print("~~~~ 8-Ensemble + Z-Filtering (full image) ~~~~")
zfil_preds_test = calculate_z_filtering(Y_test_ensemble)

print("Saving Z-filtered images . . .")
save_img(Y=zfil_preds_test, mask_dir=zfil_dir_full, prefix="test_out_zfil")

print("Calculate metrics (8-Ensemble + Z-filtering + full image) . . .")
zfil_jac_full = jaccard_index_numpy(
    Y_test, (zfil_preds_test > 0.5).astype(np.uint8))
zfil_voc_full = voc_calculation(
    Y_test, (zfil_preds_test > 0.5).astype(np.uint8), zfil_jac_full)

del zfil_preds_test, Y_test_ensemble

print("~~~~ Spurious Detection (full image) ~~~~")
spu_preds_test = spuriuous_detection_filter(preds_test_full)

print("Saving spurious detection filtering resulting images . . .")
save_img(Y=spu_preds_test, mask_dir=spu_dir_full, prefix="test_out_spu")

print("Calculate metrics (Spurious + full image) . . .")
spu_jac_full = jaccard_index_numpy(Y_test, spu_preds_test)
spu_voc_full = voc_calculation(Y_test, spu_preds_test, spu_jac_full)

print("~~~~ Watershed (full image) ~~~~")
wa_preds_test = boundary_refinement_watershed2(
    preds_test_full, (preds_test_full> 0.5).astype(np.uint8),
    save_marks_dir=wa_debug_dir_full)
    #X_test, (preds_test> 0.5).astype(np.uint8), save_marks_dir=watershed_debug_dir)

print("Saving watershed resulting images . . .")
save_img(Y=(wa_preds_test).astype(np.uint8), mask_dir=wa_dir_full,
         prefix="test_out_wa")

print("Calculate metrics (Watershed + full image) . . .")
wa_jac_full = jaccard_index_numpy(Y_test, wa_preds_test)
wa_voc_full = voc_calculation(Y_test, wa_preds_test, wa_jac_full)
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
spu_wa_zfil_jac_full = jaccard_index_numpy(
    Y_test, (spu_wa_zfil_preds_test > 0.5).astype(np.uint8))
spu_wa_zfil_voc_full = voc_calculation(
    Y_test, (spu_wa_zfil_preds_test > 0.5).astype(np.uint8),
    spu_wa_zfil_jac_full)
del spu_wa_zfil_preds_test, spu_preds_test


print("####################################\n"
      "#  PRINT AND SAVE SCORES OBTAINED  #\n"
      "####################################\n")

if load_previous_weights == False:
    print("Epoch average time: {}".format(np.mean(time_callback.times)))
    print("Epoch number: {}".format(len(results.history['val_loss'])))
    print("Train time (s): {}".format(np.sum(time_callback.times)))
    print("Train loss: {}".format(np.min(results.history['loss'])))
    print("Train IoU: {}".format(np.max(results.history[metric])))
    print("Validation loss: {}".format(np.min(results.history['val_loss'])))
    print("Validation IoU: {}"
          .format(np.max(results.history['val_'+metric])))

print("Test loss: {}".format(loss_per_crop))
print("Test IoU (per crop): {}".format(jac_per_crop))

print("Test IoU (merge into complete image): {}".format(jac_per_image))
print("Test VOC (merge into complete image): {}".format(voc_per_image))
print("Post-process: Smooth - Test IoU (merge into complete image): {}".format(smo_jac_per_image))
print("Post-process: Smooth - Test VOC (merge into complete image): {}".format(smo_voc_per_image))
print("Post-process: Z-Filtering - Test IoU (merge into complete image): {}".format(zfil_jac_per_image))
print("Post-process: Z-Filtering - Test VOC (merge into complete image): {}".format(zfil_voc_per_image))
print("Post-process: Smooth + Z-Filtering - Test IoU (merge into complete image): {}".format(smo_zfil_jac_per_image))
print("Post-process: Smooth + Z-Filtering - Test VOC (merge into complete image): {}".format(smo_zfil_voc_per_image))
print("Test IoU (merge with 50% overlap): {}".format(jac_50ov))
print("Test VOC (merge with 50% overlap): {}".format(voc_50ov))
print("Post-process: Ensemble - Test IoU (merge with 50% overlap): {}".format(ens_jac_50ov))
print("Post-process: Ensemble - Test VOC (merge with 50% overlap): {}".format(ens_voc_50ov))
print("Post-process: Ensemble + Z-Filtering - Test IoU (merge with 50% overlap): {}".format(ens_zfil_jac_50ov))
print("Post-process: Ensemble + Z-Filtering - Test VOC (merge with 50% overlap): {}".format(ens_zfil_voc_50ov))
print("Test IoU (full): {}".format(jac_full))
print("Test VOC (full): {}".format(voc_full))
print("Post-process: Ensemble - Test IoU (full): {}".format(smo_jac_full))
print("Post-process: Ensemble - Test VOC (full): {}".format(smo_voc_full))
print("Post-process: Ensemble + Z-Filtering - Test IoU (full): {}".format(zfil_jac_full))
print("Post-process: Ensemble + Z-Filtering - Test VOC (full): {}".format(zfil_voc_full))
print("Post-process: Spurious Detection - Test IoU (full): {}".format(spu_jac_full))
print("Post-process: Spurious Detection - VOC (full): {}".format(spu_voc_full))
print("Post-process: Watershed - Test IoU (full): {}".format(wa_jac_full))
print("Post-process: Watershed - VOC (full): {}".format(wa_voc_full))
print("Post-process: Spurious + Watershed + Z-Filtering - Test IoU (full): {}".format(spu_wa_zfil_jac_full))
print("Post-process: Spurious + Watershed + Z-Filtering - Test VOC (full): {}".format(spu_wa_zfil_voc_full))

if not load_previous_weights:
    scores = {}
    for name in dir():
        if not name.startswith('__') and ("_per_crop" in name or "_50ov" in name\
        or "_per_image" in name or "_full" in name):
            scores[name] = eval(name)

    create_plots(results, job_identifier, char_dir, metric=metric)

print("FINISHED JOB {} !!".format(job_identifier))

