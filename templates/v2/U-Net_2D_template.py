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

# Create job directory
os.makedirs(args.result_dir, exist_ok=True)

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
from skimage.io import imsave
from data_2D_manipulation import load_and_prepare_2D_data, crop_data_with_overlap,\
                                 merge_data_with_overlap
from generators.custom_da_gen_v2 import ImageDataGenerator
from networks.unet import U_Net_2D
from metrics import jaccard_index_numpy, voc_calculation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from PIL import Image
from tqdm import tqdm
from smooth_tiled_predictions import predict_img_with_smooth_windowing
from tensorflow.keras.utils import plot_model
from aux.callbacks import ModelCheckpoint
from post_processing import calculate_z_filtering, ensemble8_2d_predictions


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


### Crop variables 
## ~~~~ Any option ~~~~
# Shape of the crops
crop_shape = (256, 256, 1)
# Flag to extract random subvolumnes during the DA
random_crops_in_DA = False
# Calculate probability map to make random subvolumes to be extracted with high
# probability of having a mitochondria on the middle of it. Useful to avoid
# extracting a subvolume which less foreground class information.
probability_map = False # Only active with random_crops_in_DA
w_foreground = 0.94 # Weight to apply to foreground classes (probability_map=True)
w_background = 0.06 # Weight to apply to background class (probability_map=True)
## ~~~ When option (1) is chosen ~~~~
# To check the crops. Useful to ensure that the crops have been made correctly.
check_crop = True 
# Percentage of overlap in (x, y) when cropping in test. Set to 0 to calculate
# the minimun overlap
overlap = (0,0)
# Use 'overlap' values also in training data. If False overlap in train will be
# minimum tho satisficy the 'crop_shape': (0,0)
ov_train = False
# Padding to be done in (x, y) when reconstructing test data. Useful to avoid
# patch 'border effect'.                                                        
padding = (0,0)


###############################################################
# From here on, all variables will be applied in both options #
###############################################################

### Data augmentation (DA) variables 
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
rand_rot = False
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
# Make vertical flips 
vflip = False
# Make horizontal flips                                                           
hflip = False
# Elastic transformations                                                       
elastic = False
# Strength of the distortion field
e_alpha = (12, 16)
# Standard deviation of the gaussian kernel used to smooth the distortion fields
e_sigma = 4
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
motb_k_range = (8, 12)
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
cout_size = (0.05, 0.3)
# Value to fill the area of cutout
cout_cval = 0
# Apply cutout to the segmentation mask
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
missp_iterations = (10, 30)


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
batch_size_value = 6
# Optimizer to use. Possible values: "sgd" or "adam"
optimizer = "sgd"
# Learning rate used by the optimization method
learning_rate_value = 0.002
# Number of epochs to train the network
epochs_value = 360
# Number of epochs to stop the training process after no improvement
patience = 50
# If weights on data are going to be applied. To true when loss_type is 'w_bce' 
weights_on_data = True if loss_type == "w_bce" else False


### Network architecture specific parameters
# Number of feature maps on each level of the network. It's dimension must be 
# equal depth+1.
feature_maps = [16, 32, 64, 128, 256]
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
# Number of classes
n_classes = 1
# Adjust the metric used accordingly to the number of clases. This code is planned 
# to be used in a binary classification problem, so the function 'jaccard_index_softmax' 
# will only calculate the IoU for the foreground class (channel 1)
metric = "jaccard_index_softmax" if n_classes > 1 else "jaccard_index"


### Paths of the results                                             
result_dir = os.path.join(args.result_dir, 'results', job_identifier)

# per-image directories
result_bin_dir_per_image = os.path.join(result_dir, 'per_image_binarized')
result_no_bin_dir_per_image = os.path.join(result_dir, 'per_image_no_binarized')
smo_bin_dir_per_image = os.path.join(result_dir, 'per_image_blending')
smo_no_bin_dir_per_image = os.path.join(result_dir, 'per_image_blending_no_bin')
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
# Necessary when using TF version < 2.2 as pointed out in:                      
#     https://github.com/tensorflow/tensorflow/issues/35911                     
class OnEpochEnd(tf.keras.callbacks.Callback):                                  
  def __init__(self, callbacks):                                                
    self.callbacks = callbacks                                                  
                                                                                
  def on_epoch_end(self, epoch, logs=None):                                     
    for callback in self.callbacks:                                             
      callback()     


print("###################\n"
      "#  SANITY CHECKS  #\n"
      "###################\n")

#check_masks(train_mask_path)
#check_masks(test_mask_path)


if in_memory:
    print("###############\n"                                                       
          "#  LOAD DATA  #\n"                                                       
          "###############\n")

    X_train, Y_train, X_val,\
    Y_val, X_test, Y_test,\
    orig_test_shape, crop_test_shapes,\
    filenames = load_and_prepare_2D_data(
        train_path, train_mask_path, test_path, test_mask_path,
        val_split=perc_used_as_val, shuffle_val=random_val_data,
        random_crops_in_DA=random_crops_in_DA, crop_shape=crop_shape,
        ov=overlap, padding=padding, overlap_train=ov_train, 
        check_crop=check_crop, check_crop_path=check_crop_path,
        crop_test=False)
else:
    X_train = Y_train = X_val = Y_val = X_test = Y_test = None


print("#######################\n"
      "#  DATA AUGMENTATION  #\n"
      "#######################\n")

# Calculate the probability map per image
prob_map = None                                                                 
if probability_map and random_crops_in_DA:                                      
    if os.path.exists(prob_map_dir):                                            
        print("Loading probability map")
        prob_map_file = os.path.join(prob_map_dir, 'prob_map.npy')              
        num_files = len(next(os.walk(prob_map_dir))[2])                         
        prob_map = prob_map_dir if num_files > 1 else np.load(prob_map_file)    
    else:                                                                       
        prob_map = calculate_2D_volume_prob_map(                                
            Y_train, train_mask_path, w_foreground, w_background, 
            save_dir=prob_map_dir)
     

# Custom Data Augmentation                                                  
train_generator = ImageDataGenerator(X=X_train, Y=Y_train, 
    batch_size=batch_size_value, shuffle=shuffle_train_data_each_epoch, 
    in_memory=in_memory, data_paths=data_paths[0:2], da=da, da_prob=da_prob, 
    rotation90=rotation90, rand_rot=rand_rot, rnd_rot_range=rnd_rot_range, 
    shear=shear, shear_range=shear_range, zoom=zoom, zoom_range=zoom_range, 
    shift=shift, shift_range=shift_range, vflip=vflip, hflip=hflip, 
    elastic=elastic, e_alpha=e_alpha, e_sigma=e_sigma, e_mode=e_mode, 
    g_blur=g_blur, g_sigma=g_sigma, median_blur=median_blur, mb_kernel=mb_kernel,
    motion_blur=motion_blur, motb_k_range=motb_k_range,
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
    missp_iterations=missp_iterations, shape=crop_shape,
    random_crops_in_DA=random_crops_in_DA, prob_map=prob_map,
    n_classes=n_classes, extra_data_factor=replicate_train)

val_generator = ImageDataGenerator(X=X_val, Y=Y_val, batch_size=batch_size_value, 
    shuffle=shuffle_val_data_each_epoch, in_memory=in_memory,                                                 
    data_paths=data_paths[2:4], da=False, shape=crop_shape,
    random_crops_in_DA=random_crops_in_DA, val=True, n_classes=n_classes)

test_generator = ImageDataGenerator(X=X_test, Y=Y_test, batch_size=batch_size_value,
    shuffle=shuffle_val_data_each_epoch, in_memory=in_memory,            
    data_paths=data_paths[4:6], da=False, shape=crop_shape,
    random_crops_in_DA=random_crops_in_DA, val=True, n_classes=n_classes)       
      
# Generate examples of data augmentation                                    
if aug_examples:                                                    
    train_generator.get_transformed_samples(
        10, save_to_dir=True, train=False, out_dir=da_samples_dir)

print("#################################\n"
      "#  BUILD AND TRAIN THE NETWORK  #\n"
      "#################################\n")

print("Creating the network . . .")
model = U_Net_2D(crop_shape, activation=activation, feature_maps=feature_maps, 
                 depth=depth, drop_values=dropout_values, 
                 spatial_dropout=spatial_dropout, batch_norm=batch_normalization, 
                 k_init=kernel_init, loss_type=loss_type, optimizer=optimizer, 
                 lr=learning_rate_value, n_classes=n_classes)

# Check the network created
model.summary(line_length=150)
os.makedirs(char_dir, exist_ok=True)
model_name = os.path.join(char_dir, "model_plot_" + job_identifier + ".png")
plot_model(model, to_file=model_name, show_shapes=True, show_layer_names=True)

h5_file=os.path.join(h5_dir, weight_files_prefix + previous_job_weights     
                     + '_' + str(args.run_id) + '.h5')  

if not load_previous_weights:
    results = model.fit(train_generator, validation_data=val_generator,
        validation_steps=len(val_generator), steps_per_epoch=len(train_generator),
        epochs=epochs_value, callbacks=[earlystopper, checkpointer, time_callback,
        OnEpochEnd([train_generator.on_epoch_end])])

    create_plots(results, job_identifier, char_dir, metric=metric)

print("Loading model weights from h5_file: {}".format(h5_file))
model.load_weights(h5_file)


print("##########################\n"
      "#  INFERENCE (per crop)  #\n"
      "##########################\n")

print("Making the predictions on test data . . .")
loss_per_crop = 0
iou_per_crop = 0
iou = 0
ov_iou = 0
os.makedirs(result_no_bin_dir_per_image, exist_ok=True)
it = iter(test_generator)
c1 = 0
c2 = 0
d = len(str(len(test_generator)*batch_size_value))
for i in tqdm(range(len(test_generator))):
    batch = next(it)

    X, Y = batch  
    for j in tqdm(range(X.shape[0]), leave=False):
        X_test, Y_test = crop_data_with_overlap(
            np.expand_dims(X[j],0), crop_shape, data_mask=np.expand_dims(Y[j],0), 
                           overlap=overlap, padding=padding, verbose=False)
       
        # Evaluate each patch
        l = int(math.ceil(X_test.shape[0]/batch_size_value))
        for k in tqdm(range(l), leave=False):
            top = (k+1)*batch_size_value if (k+1)*batch_size_value < X_test.shape[0] else X_test.shape[0]
            score_per_crop = model.evaluate(
                X_test[k*batch_size_value:top], Y_test[k*batch_size_value:top], verbose=0)
            loss_per_crop += score_per_crop[0]
            iou_per_crop += score_per_crop[1]
    
        # Predict each patch with ensembling
        pred = []
        for k in tqdm(range(X_test.shape[0]), leave=False):
            p = ensemble8_2d_predictions(X_test[k],
                pred_func=(lambda img_batch_subdiv: model.predict(img_batch_subdiv)),
                n_classes=n_classes)
            pred.append(np.expand_dims(p, 0))
            
        # Reconstruct the original shape 
        pred = np.concatenate(pred)
        if n_classes > 1:
            pred = np.expand_dims(np.argmax(pred,-1), -1)
            Y = np.expand_dims(np.argmax(Y,-1), -1)
        pred = merge_data_with_overlap(pred, (1,)+Y.shape[1:], padding=padding,
                                       overlap=overlap, verbose=False)
    
        c = (i*batch_size_value)+j                                              
        f = os.path.join(result_no_bin_dir_per_image, str(c).zfill(d)+'.tif')   
        aux = np.expand_dims(pred.transpose((0,3,1,2)), -1).astype(np.float32)  
        imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False)
          
        iou_per_image = jaccard_index_numpy((Y[j]>0.5).astype(np.uint8), (pred[0] > 0.5).astype(np.uint8))
        ov_iou_per_image = voc_calculation((Y[j]>0.5).astype(np.uint8), (pred[0] > 0.5).astype(np.uint8), 
                                           iou_per_image)
        iou += iou_per_image
        ov_iou += ov_iou_per_image
        c1 += X_test.shape[0]                                                    
    c2 += X.shape[0]

loss_per_crop = loss_per_crop / c1
iou_per_crop = iou_per_crop / c1
iou = iou / c2
ov_iou = ov_iou / c2


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
print("Test Foreground IoU (per crop): {}".format(iou_per_crop))
print("Test Foreground IoU (merge into complete image): {}".format(iou))
print("Test Overall IoU (merge into complete image): {}".format(ov_iou))

print("FINISHED JOB {} !!".format(job_identifier))

