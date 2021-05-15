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

# Working dir
os.chdir(args.base_work_dir)

# Create job directory
os.makedirs(args.result_dir, exist_ok=True)

# Limit the number of threads
from util import limit_threads, set_seed, create_plots, store_history,\
                 TimeHistory, threshold_plots, save_img, \
                 calculate_3D_volume_prob_map, check_masks, labels_into_bcd
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
import h5py
import tensorflow as tf
from data_3D_manipulation import load_and_prepare_3D_data_v2,\
                                 merge_3D_data_with_overlap, \
                                 crop_3D_data_with_overlap,\
                                 load_3d_images_from_dir
from generators.data_3D_generators_v2 import VoxelDataGenerator
from networks.resunet_3d_instances import ResUNet_3D_instances
from metrics import jaccard_index_numpy, voc_calculation
from tensorflow.keras.callbacks import EarlyStopping
from aux.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tqdm import tqdm
from tensorflow.keras.utils import plot_model
from post_processing import ensemble16_3d_predictions, bc_watershed, bcd_watershed
from skimage.io import imsave, imread


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
#                                                                               
# IMPORTANT NOTE                                                                
# There are two possible options to work the data with:                         
#     1) Load the entire dataset into memory                                    
#     2) load every file when generating network input during DA.               
#                                                                               
# Choose one and set the following variable correspondly '1 == True', '2 == False'
# Some configuration depends on the method chosen. Notice the header, for instance,
# "When option (1) is chosen" refers to variables that only are supported when  
# option 1 is chosen. "Any option" means that those variables are supported in  
# bot options.                                                                  
in_memory = True


### Main dataset data/mask paths
train_path = os.path.join(args.data_dir, 'train', 'x')
train_mask_path = os.path.join(args.data_dir, 'train', 'y')
test_path = os.path.join(args.data_dir, 'test', 'x') 
test_mask_path = os.path.join(args.data_dir, 'test', 'y')


### Validation data                                                             
## ~~~ When option (1) is chosen ~~~~                                           
# Percentage of the training data used as validation                            
perc_used_as_val = 0.1
# Create the validation data with random images of the training data. If False  
# the validation data will be the last portion of training images.              
random_val_data = True                                                          
## ~~~~ When option (2) is chosen ~~~~                                          
# Validation should be stored in a directoy as the train and test               
val_path = os.path.join(args.data_dir, 'val', 'x')                              
val_mask_path = os.path.join(args.data_dir, 'val', 'y')                         
## ~~~~ Any option ~~~~                                                         
# Store the paths to create the generators later                                
data_paths = []                                                                 
data_paths.append(train_path)                                                   
data_paths.append(train_mask_path)                                              
data_paths.append(val_path)                                                     
data_paths.append(val_mask_path)                                                
data_paths.append(test_path)                                                    
data_paths.append(test_mask_path)                                               


### Crop variables: (x, y, z, channels)                                                              
## ~~~~ Any option ~~~~                                                         
# Shape of the 3D crops (channels for the masks are automatically selected)
crop_shape = (64, 64, 64, 1)
# Flag to extract random subvolumnes during the DA. If it is not True and 
# in_memory=False the samples loaded form disk are supposed to be of the same 
# size
random_subvolumes_in_DA = False
# Calculate probability map to make random subvolumes to be extracted with high 
# probability of having a mitochondria on the middle of it. Useful to avoid     
# extracting a subvolume which less mitochondria information.                   
probability_map = False # Only active with random_subvolumes_in_DA              
w_foreground = 0.98 # Weight to apply to foreground classes (probability_map=True)
w_background = 0.02 # Weight to apply to background class (probability_map=True)
## ~~~ When option (1) is chosen ~~~~
# Percentage of overlap in (x, y, z). Set to 0 to calculate the minimun overlap 
overlap = (0,0,0)
# Padding to be done in (x, y, z) when reconstructing test data. Useful to avoid
# patch 'border effect'.
padding = (16, 16, 0)
# Wheter to use median values to fill padded pixels or zeros
median_padding = False
# Make overlap on train data
ov_train = False
# Wheter to use the rest of the train data when there is no exact division between
# it and the subvolume shape needed (crop_shape). Only has sense when
# ov_train is False
use_rest_train = False


###############################################################                 
# From here on, all variables will be applied in both options #                 
###############################################################

### Data augmentation (DA) variables.
# Flag to activate DA
da = True
# Probability of each transformation
da_prob = 0.5
# Create samples of the DA made. Useful to check the output images made.
aug_examples = True
# Flag to shuffle the training data on every epoch 
shuffle_train_data_each_epoch = True
# Flag to shuffle the validation data on every epoch
shuffle_val_data_each_epoch = False
# Rotation of 90º to the subvolumes
rotation90 = False
# Random rotation between a defined range 
rand_rot = True
# Range of random rotations
rnd_rot_range = (-180, 180)
# Apply shear to images
shear = False
# Shear range
shear_range = (-20, 20)
# Apply zoom to images
zoom = False
# Zoom range
zoom_range = (0.8, 1.2)
# Apply shift 
shift = False   
# Shift range             
shift_range = (0.1, 0.2)
# Flag to make vertical flips
vflip = True
# Flag to make horizontal flips
hflip = True
# Flag to make flips in z
zflip = True
# Elastic transformations                                                       
elastic = False
# Strength of the distortion field
e_alpha = (240, 250)
# Standard deviation of the gaussian kernel used to smooth the distortion fields
e_sigma = 25
# Parameter that defines the handling of newly created pixels with the elastic
# transformation
e_mode = 'constant'
# Gaussian blur
g_blur = False
# Standard deviation of the gaussian kernel
g_sigma = (1.0, 2.0)
# To blur an image by computing median values over neighbourhoods
median_blur = False
# Median blur kernel size
mb_kernel = (3, 7)
# Blur images in a way that fakes camera or object movements
motion_blur = False
# Kernel size to use in motion blur
motb_k_range = (3, 8)
# Gamma contrast
gamma_contrast = False
# Exponent for the contrast adjustment. Higher values darken the image
gc_gamma = (1.25, 1.75)
# To apply brightness changes to images
brightness = False
# Strength of the brightness range, with valid values being 0 <= brightness_factor <= 1
brightness_factor = (0.1, 0.3)
# To apply contrast changes to images
contrast = False
# Strength of the contrast change range, with valid values being 0 <= contrast_factor <= 1
contrast_factor = (0.1, 0.3)
# Set a certain fraction of pixels in images to zero. Not get confuse with the
# dropout concept of neural networks, this is just for DA
dropout = False
# Range to take the probability to drop a pixel
drop_range = (0, 0.2)
# To fill one or more rectangular areas in an image using a fill mode
cutout = False
# Range of number of areas to fill the image with
cout_nb_iterations = (1, 3)
# Size of the areas in % of the corresponding image size
cout_size = (0.05, 0.4)
# Value to fill the area of cutout
cout_cval = 0
# Apply cutout to the segmentation masks
cout_apply_to_mask = False
# To apply cutblur operation
cutblur = False
# Size of the region to apply cutblur
cblur_size = (0.2, 0.4)
# Range of the downsampling to be made in cutblur
cblur_down_range = (2, 8)
# Wheter to apply cut-and-paste just LR into HR image. If False, HR to LR will
# be applied also (see Figure 1 of the paper https://arxiv.org/pdf/2004.00448.pdf)
cblur_inside = True
# Apply cutmix operation
cutmix = False
# Size of the region to apply cutmix
cmix_size = (0.2, 0.4)
# Apply noise to a region of the image                                          
cutnoise = False
# Scale of the random noise                                                     
cnoise_scale = (0.1, 0.2)                                                       
# Number of areas to fill with noise                                            
cnoise_nb_iterations = (1, 3)                                                   
# Size of the regions                                                           
cnoise_size = (0.2, 0.4) 
# Add miss-aligment augmentation                                                
misalignment = False
# Maximum pixel displacement in `xy`-plane for misalignment                     
ms_displacement = 16                                                            
# Ratio of rotation-based mis-alignment                                         
ms_rotate_ratio = 0.5                                                           
# Augment the image by creating a black line in a random position
missing_parts = False
# Iterations to dilate the missing line with
missp_iterations = (30, 40)


### Extra train data generation
# Number of times to duplicate the train data. Useful when 
# "random_subvolumes_in_DA" is made, as more original train data can be cover
replicate_train = 0


### Load previously generated model weigths
# Flag to activate the load of a previous training weigths instead of train 
# the network again
load_previous_weights = True
# ID of the previous experiment to load the weigths from 
previous_job_weights = args.job_id
# Prefix of the files where the weights are stored/loaded from
weight_files_prefix = 'model.fibsem_'
# Wheter to find the best learning rate plot. If this options is selected the 
# training will stop when 5 epochs are done
use_LRFinder = False


### Experiment main parameters
# Batch size value
batch_size_value = 2
# Optimizer to use. Possible values: "sgd" or "adam"
optimizer = "adam"
# Learning rate used by the optimization method
learning_rate_value = 0.0001
# Number of epochs to train the network
epochs_value = 200
# Number of epochs to stop the training process after no improvement
patience = 50


### Network architecture specific parameters
# Number of feature maps on each level of the network
#feature_maps = [28, 36, 48, 64]
feature_maps = [36, 48, 64]
# Depth of the network
depth = 2
# Flag to activate the Spatial Dropout instead of use the "normal" dropout layer
spatial_dropout = False
# Values to make the dropout with. It's dimension must be equal depth+1. Set to
# 0 to prevent dropout
dropout_values = [0, 0, 0]
# Flag to active batch normalization
batch_normalization = False
# Kernel type to use on convolution layers
kernel_init = 'he_normal'
# Activation function to use
activation = "elu"
# Downsampling to be made in Z. This value will be the third integer of the
# MaxPooling operation. When facing anysotropic datasets set it to get better
# performance
z_down = 1
# Adjust the metric used accordingly to the number of clases. This code is planned 
# to be used in a binary classification problem, so the function 'jaccard_index_softmax' 
# will only calculate the IoU for the foreground class (channel 1)              
metric = "jaccard_index_instances"


### Instances options
# Channels to operate with. Possible values: BC and BCD. BC corresponds to use
# binary segmentation+contour. BCD stands for binary segmentation+contour+distances.   
output_channels = "BCD"
# Weights to be applied to segmentation (binary and contours) and to distances
# respectively. E.g. (1, 0.2), 1 should be multipled by BCE for the first two
# channels and 0.2 to MSE for the last channel. 
channel_weights = (1, 0.2)


### Inference options
# To use ensemble for each patch, where a mean prediction of the 16 possible 
# rotations of each patch is used. It should improve the prediction but it will
# take much time
ensemble = False
# Path were the test stacks are placed. It can be the same as test_path
test_full_path = os.path.join(args.data_dir, 'test_full', 'x')
# Path were the test mask stacks are placed. It can be the same as test_mask_path
test_full_mask_path = os.path.join(args.data_dir, 'test_full', 'y')


### mAP calculation options
# Do not forgive to clone the repo:                                             
#       git clone https://github.com/danifranco/mAP_3Dvolume.git               
# 
# Change the branch:
#       git checkout grand-challenge                                            

# Folder where the mAP code should be placed 
mAP_folder = os.path.join(args.base_work_dir,'..', 'mAP_3Dvolume')              
# Path to the GT h5 files to calculate the mAP                                  
test_full_gt_h5 = os.path.join(args.data_dir, 'test_full', 'h5')


### Paths of the results                                             
# Directory where predicted images of the segmentation will be stored
result_dir = os.path.join(args.result_dir, 'results', job_identifier)

# per-image directories
result_no_bin_dir_per_image = os.path.join(result_dir, 'per_image_no_binarized')
ens_bin_dir_per_image = os.path.join(result_dir, 'per_image_ensemble')

# Name of the folder where the charts of the loss and metrics values while
# training the network will be shown. This folder will be created under the
# folder pointed by "args.base_work_dir" variable
char_dir = os.path.join(result_dir, 'charts')
# Directory where weight maps will be stored
loss_weight_dir = os.path.join(result_dir, 'loss_weights', args.job_id)
# Folder where smaples of DA will be stored
da_samples_dir = os.path.join(result_dir, 'aug')
# Name of the folder where weights files will be stored/loaded from. This folder
# must be located inside the directory pointed by "args.base_work_dir" variable.
# If there is no such directory, it will be created for the first time
h5_dir = os.path.join(args.result_dir, 'h5_files')
# Name of the folder to store the probability map to avoid recalculating it on
# every run
prob_map_dir = os.path.join(args.result_dir, 'prob_map')
# Watershed dubgging folder
watershed_dir = os.path.join(result_dir, 'watershed')
# To store h5 files needed for the mAP calculation
mAP_h5_dir = os.path.join(result_dir, 'mAP_h5_files')


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
# Necessary when using TF version < 2.2 as pointed out in:
#     https://github.com/tensorflow/tensorflow/issues/35911
class OnEpochEnd(tf.keras.callbacks.Callback):
  def __init__(self, callbacks):
    self.callbacks = callbacks

  def on_epoch_end(self, epoch, logs=None):
    for callback in self.callbacks:
      callback()


if in_memory:
    print("###############\n"
          "#  LOAD DATA  #\n"
          "###############\n")

    X_train, Y_train,\
    X_test, Y_test,\
    orig_test_shape, crop_test_shapes,\
    filenames = load_and_prepare_3D_data_v2(
        train_path, train_mask_path, test_path, test_mask_path, 
        val_split=perc_used_as_val, create_val=False, shuffle_val=random_val_data,
        random_subvolumes_in_DA=random_subvolumes_in_DA,
        test_subvol_shape=crop_shape, train_subvol_shape=crop_shape, ov=overlap,
        padding=padding, median_padding=median_padding)
      
    X_val, _, _ = load_3d_images_from_dir(val_path)
    Y_val, _, _ = load_3d_images_from_dir(val_mask_path)

    ## TRAIN
    aux_dir = os.path.join(args.result_dir, 'aux_train')
    if not os.path.isfile(os.path.join(args.result_dir, 'Y_train.npy')):
        Y_train = labels_into_bcd(Y_train, save_dir=aux_dir)                                    
        np.save(os.path.join(args.result_dir, 'Y_train.npy'), Y_train)
    else:
        Y_train = np.load(os.path.join(aux_dir, '../Y_train.npy'))

    ## VAL
    aux_dir = os.path.join(args.result_dir, 'aux_val')                        
    if not os.path.isfile(os.path.join(args.result_dir, 'Y_val.npy')):        
        Y_val = labels_into_bcd(Y_val, save_dir=aux_dir)                   
        np.save(os.path.join(args.result_dir, 'Y_val.npy'), Y_val)          
    else:                                                                       
        Y_val = np.load(os.path.join(aux_dir, '../Y_val.npy'))

    ## TEST
    aux_dir = os.path.join(args.result_dir, 'aux_test')
    if not os.path.isfile(os.path.join(args.result_dir, 'Y_test.npy')):
        np.save(os.path.join(args.result_dir, 'Y_test.npy'), (Y_test>0).astype(np.uint8))
    else:
        Y_test = np.load(os.path.join(args.result_dir, 'Y_test.npy'))
else:
    X_train = Y_train = X_val = Y_val = X_test = Y_test = None 


print("#######################\n"
      "#  DATA AUGMENTATION  #\n"
      "#######################\n")

# Calculate the probability map per image                                       
prob_map = None                                                                 
if probability_map and random_subvolumes_in_DA:                                 
    if os.path.exists(prob_map_dir):                                            
        prob_map_file = os.path.join(prob_map_dir, 'prob_map.npy')              
        num_files = len(next(os.walk(prob_map_dir))[2])                         
        prob_map = prob_map_dir if num_files > 1 else np.load(prob_map_file)    
    else:                                                                       
        prob_map = calculate_3D_volume_prob_map(                                
            Y_train, train_mask_path, w_foreground, w_background, 
            save_dir=prob_map_dir)

print("Preparing train data generator . . .")
train_generator = VoxelDataGenerator(                                           
    X_train, Y_train, in_memory=in_memory, data_paths=data_paths[0:2],
    random_subvolumes_in_DA=random_subvolumes_in_DA, 
    prob_map=prob_map,subvol_shape=crop_shape, 
    shuffle_each_epoch=shuffle_train_data_each_epoch, batch_size=batch_size_value,
    da=da, da_prob=da_prob, rotation90=rotation90, rand_rot=rand_rot, 
    rnd_rot_range=rnd_rot_range, shear=shear,shear_range=shear_range, zoom=zoom, 
    zoom_range=zoom_range, shift=shift, shift_range=shift_range, vflip=vflip, 
    hflip=hflip, zflip=zflip, elastic=elastic, e_alpha=e_alpha, e_sigma=e_sigma, 
    e_mode=e_mode, g_blur=g_blur, g_sigma=g_sigma, median_blur=median_blur, 
    mb_kernel=mb_kernel, motion_blur=motion_blur, motb_k_range=motb_k_range,
    gamma_contrast=gamma_contrast, gc_gamma=gc_gamma, brightness=brightness, 
    brightness_factor=brightness_factor, contrast=contrast,
    contrast_factor=contrast_factor, dropout=dropout, drop_range=drop_range,
    cutout=cutout, cout_nb_iterations=cout_nb_iterations, cout_size=cout_size,
    cout_cval=cout_cval, cout_apply_to_mask=cout_apply_to_mask, cutblur=cutblur, 
    cblur_size=cblur_size, cblur_down_range=cblur_down_range, 
    cblur_inside=cblur_inside, cutmix=cutmix, cmix_size=cmix_size,
    cutnoise=cutnoise, cnoise_size=cnoise_size,
    cnoise_nb_iterations=cnoise_nb_iterations, cnoise_scale=cnoise_scale,
    misalignment=misalignment, ms_displacement=ms_displacement, 
    ms_rotate_ratio=ms_rotate_ratio, missing_parts=missing_parts,
    missp_iterations=missp_iterations, extra_data_factor=replicate_train)
del X_train, Y_train

print("Preparing validation data generator . . .")                              
val_generator = VoxelDataGenerator(                                             
    X_val, Y_val, in_memory=in_memory, data_paths=data_paths[2:4],              
    random_subvolumes_in_DA=random_subvolumes_in_DA,                            
    subvol_shape=crop_shape, shuffle_each_epoch=shuffle_val_data_each_epoch,    
    batch_size=batch_size_value, da=False, val=True)       
del X_val, Y_val 

# Create the test data generator without DA
print("Preparing test data generator . . .")
test_generator = VoxelDataGenerator(
    X_test, Y_test, in_memory=in_memory, data_paths=data_paths[4:6],
    random_subvolumes_in_DA=False, shuffle_each_epoch=False,
    batch_size=batch_size_value, da=False)
del X_test, Y_test

# Generate examples of data augmentation
if aug_examples:
    train_generator.get_transformed_samples(
        5, random_images=False, save_to_dir=True, out_dir=da_samples_dir)


print("#################################\n"
      "#  BUILD AND TRAIN THE NETWORK  #\n"
      "#################################\n")

print("Creating the network . . .")
model = ResUNet_3D_instances(crop_shape, activation=activation, depth=depth,
    feature_maps=feature_maps, drop_values=dropout_values,
    batch_norm=batch_normalization, k_init=kernel_init, optimizer=optimizer,
    output_channels=output_channels, channel_weights=channel_weights, 
    lr=learning_rate_value, z_down=z_down)

# Check the network created
model.summary(line_length=150)
os.makedirs(char_dir, exist_ok=True)
model_name = os.path.join(char_dir, "model_plot_" + job_identifier + ".png")
plot_model(model, to_file=model_name, show_shapes=True, show_layer_names=True)

h5_file=os.path.join(h5_dir, weight_files_prefix + previous_job_weights     
                     + '_' + str(args.run_id) + '.h5')

if load_previous_weights == False:
    results = model.fit(x=train_generator, validation_data=val_generator,
        epochs=epochs_value, callbacks=[earlystopper, checkpointer, time_callback, 
        OnEpochEnd([train_generator.on_epoch_end])])

    create_plots(results, job_identifier, char_dir, metric=metric)

print("Loading model weights from h5_file: {}".format(h5_file))
model.load_weights(h5_file)


print("##########################\n"
      "#  INFERENCE (per crop)  #\n"
      "##########################\n")

## Overlap and padding prediction
X_test, _, _, t_filenames = load_3d_images_from_dir(test_full_path, return_filenames=True)
d = len(str(len(X_test)))
for im in range(len(X_test)):
    # Read and crop 
    X_test = (X_test[im]/255).transpose((2,0,1,3)) # Convert to (z, x, y, c)
    original_data_shape = X_test.shape
    X_test = crop_3D_data_with_overlap(
        X_test, crop_shape, overlap=overlap, padding=padding, verbose=True, 
        median_padding=median_padding)
    
    # Make prediction over each patch
    pred = []
    if ensemble:
        for i in tqdm(range(X_test.shape[0])):
            p = ensemble16_3d_predictions(X_test[i],                       
                pred_func=(lambda img_batch_subdiv: model.predict(img_batch_subdiv)),
                batch_size_value=batch_size_value)
            pred.append(np.expand_dims(p,0))
    else:
        l = int(math.ceil(X_test.shape[0]/batch_size_value))                       
        for k in tqdm(range(l)):
            top = (k+1)*batch_size_value if (k+1)*batch_size_value < X_test.shape[0] else X_test.shape[0]
            p = model.predict(X_test[k*batch_size_value:top])
            pred.append(p)
    del X_test                                                                      
    pred = np.concatenate(pred)
    
    # Merge and save prediction                                                     
    pred = merge_3D_data_with_overlap(pred, original_data_shape[:3]+(pred.shape[-1],),
                                      overlap=overlap, padding=padding, verbose=True)            
    if ensemble:
        os.makedirs(ens_bin_dir_per_image, exist_ok=True)
        f = os.path.join(ens_bin_dir_per_image, os.path.splitext(t_filenames[im])[0]+'.tif')                 
    else:
        os.makedirs(result_no_bin_dir_per_image, exist_ok=True)
        f = os.path.join(result_no_bin_dir_per_image, os.path.splitext(t_filenames[im])[0]+'.tif')
    aux = np.expand_dims(pred.astype(np.float32).transpose((0,3,1,2)), -1)      
    imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False)   
                                                                                  
    # Create instances 
    print("Creating instances with watershed . . .")
    if output_channels == "BC":
        pred = bc_watershed(pred, thres1=0.9, thres2=0.6, thres3=0.8, thres_small=5,
                            save_dir=watershed_dir)
    else:
        pred = bcd_watershed(pred, thres1=0.9, thres2=0.6, thres3=0.8, thres4=1.2, 
                             thres5=1, thres_small=5, save_dir=watershed_dir)
    
    
    print("####################\n"                                            
          "#  mAP Calculation #\n"
          "####################\n")  
    # Convert the prediction into an .h5 file   
    os.makedirs(mAP_h5_dir, exist_ok=True)
    h5file_name = os.path.join(mAP_h5_dir, os.path.splitext(t_filenames[im])[0]+'.h5')
    h5f = h5py.File(h5file_name, 'w')                                               
    h5f.create_dataset('dataset', data=pred, compression="lzf")             
    h5f.close()       
    
    # Prepare mAP call
    sys.path.insert(0, mAP_folder)                                                  
    from demo_modified import main as mAP_calculation    
    class Namespace:                                                                
        def __init__(self, **kwargs):                                               
            self.__dict__.update(kwargs)                                            
                                           
    # Create GT H5 file if it does not exist 
    gt_f = os.path.join(test_full_gt_h5, os.path.splitext(t_filenames[im])[0]+'.h5')
    test_file = os.path.join(test_full_mask_path, t_filenames[im])
    if not os.path.isfile(gt_f):
        print("GT .h5 file needed for mAP calculation is not found in {} so it "
              "will be created from its mask: {}".format(gt_f, test_file))

        if not os.path.isfile(test_file):
            raise ValueError("The mask is supossed to have the same name as the image")

        Y_test = imread(test_file).squeeze()
        
        print("Saving .h5 GT data from array shape: {}".format(Y_test.shape))
        os.makedirs(test_full_gt_h5, exist_ok=True)
        h5f = h5py.File(gt_f, 'w')                                           
        h5f.create_dataset('dataset', data=Y_test, compression="lzf")                 
        h5f.close() 
        del Y_test

    # Calculate mAP
    args = Namespace(gt_seg=gt_f, predict_seg=h5file_name, predict_score='',
                     threshold="5e3, 3e4", threshold_crumb=64, chunk_size=250, 
                     output_name=result_dir, do_txt=1, do_eval=1, slices="-1")
    mAP_calculation(args)


print("####################################\n"
      "#  PRINT AND SAVE SCORES OBTAINED  #\n"
      "####################################\n")

if load_previous_weights == False:
    print("Epoch average time: {}".format(np.mean(time_callback.times)))
    print("Epoch number: {}".format(len(results.history['val_loss'])))
    print("Train time (s): {}".format(np.sum(time_callback.times)))
    print("Train loss: {}".format(np.min(results.history['loss'])))
    print("Train Foreground IoU: {}".format(np.max(results.history[metric])))
    print("Validation loss: {}".format(np.min(results.history['val_loss'])))
    print("Validation Foreground IoU: {}".format(np.max(results.history['val_'+metric])))

print("FINISHED JOB {} !!".format(job_identifier))
