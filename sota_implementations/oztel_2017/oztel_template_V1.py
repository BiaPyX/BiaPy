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
sys.path.insert(0, os.path.join("sota_implementations", "oztel_2017"))

# Working dir
os.chdir(args.base_work_dir)

# Limit the number of threads
from util import limit_threads, set_seed, create_plots, store_history,\
                 TimeHistory, threshold_plots, save_img, \
                 calculate_2D_volume_prob_map, check_masks, \
                 img_to_onehot_encoding
limit_threads()

# Try to generate the results as reproducible as possible
seed=42
set_seed(seed)

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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from cnn_oztel import create_oztel_model_V1
from metrics import jaccard_index_numpy, voc_calculation, jaccard_index
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from PIL import Image
from tqdm import tqdm
from smooth_tiled_predictions import predict_img_with_smooth_windowing
from tensorflow.keras.utils import plot_model
from aux.callbacks import ModelCheckpoint
from post_processing import calculate_z_filtering

from utils import create_oztel_patches, ensemble8_2d_predictions, \
                  spuriuous_detection_filter, watershed_refinement, \
                  improve_components
from tensorflow.keras.utils import to_categorical
from skimage.util import img_as_ubyte
from skimage import io
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from skimage import transform


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


### Data shape
# Note: train and test dimensions must be the same when training the network and
# making the predictions. Be sure to take care of this if you are not going to
# use "crop_data_with_overlap()" with the arg force_shape, as this function 
# resolves the problem creating always crops of the same dimension
img_train_shape = (1024, 768, 1)
img_test_shape = (1024, 768, 1)


### Data augmentation (DA) variables
### Options available for Keras Data Augmentation
# widtk_h_shift_range (more details in Keras ImageDataGenerator class)
k_w_shift_r = 0.2
# height_shift_range (more details in Keras ImageDataGenerator class)
k_h_shift_r = 0.2
# k_shear_range (more details in Keras ImageDataGenerator class)
k_shear_range = 0.2
# Range to pick a brightness value from to apply in the images. Available in 
# Keras. Example of use: k_brightness_range = [1.0, 1.0]
k_brightness_range = None 

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

### Load previously generated model weigths
# To activate the load of a previous training weigths instead of train 
# the network again
load_previous_weights = False
# ID of the previous experiment to load the weigths from 
previous_job_weights = args.job_id
# Prefix of the files where the weights are stored/loaded from
weight_files_prefix = 'model.fibsem_'


### Experiment main parameters
# Loss type
loss='categorical_crossentropy'
# Batch size value
batch_size_value = 128
# Learning rate used by the optimization method
learning_rate_value = 0.001
# Number of epochs to train the network
epochs_value = 45
# Number of epochs to stop the training process after no improvement
patience = 25


### Network architecture specific parameters
# Number of classes. To generate data with more than 1 channel custom DA need to
# be selected. It can be 1 or 2.                                                                   
n_classes = 2
# Metric used                                                                   
metric = "accuracy" 
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

###### TRAIN ########
# Read the list of file names
train_input_filenames = [x for x in os.listdir( train_path ) if x.endswith(".tif")]
train_input_filenames.sort()
train_label_filenames = [x for x in os.listdir( train_mask_path ) if x.endswith(".tif")]
train_label_filenames.sort()
print('Input images loaded: ' + str( len(train_input_filenames)))
print('Label images loaded: ' + str( len(train_label_filenames)))
# Read training images and labels
train_img = [ img_as_ubyte( io.imread( train_path + '/' + x ) ) for x in train_input_filenames ]
train_lbl = [ img_as_ubyte( io.imread( train_mask_path + '/' + x ) ) for x in train_label_filenames ]

[class0_img, class1_img] = create_oztel_patches( train_img, train_lbl )
print("Total number of background patches =", len(class0_img))
print("Total number of mitochondria patches =", len(class1_img))

###### VALIDATION ########
# Manually extract validation set
val_size = 0.2

# Shuffle lists of each class
random.Random(seed).shuffle( class0_img )
random.Random(seed).shuffle( class1_img )

# Select 'val_size' % for validation
index = int( len(class0_img) * val_size)
class0_img_val = class0_img[0:index]
class0_img_train = class0_img[index:]

index = int( len(class1_img) * val_size)
class1_img_val = class1_img[0:index]
class1_img_train = class1_img[index:]

# Resample mitochondria
resample = 20
aux = class1_img_train
for i in range(0,resample-1):
    class1_img_train = class1_img_train + aux

print('Resampled number of training mitochondria images: {}'.format(len(class1_img_train)) )
print('Number of training non-mitochondria images: {}'.format(len(class0_img_train)) )

class_names = ["non-mitochondria", 'mitochondria' ]
num_classes = len(class_names)

x_train = np.asarray(class0_img_train + class1_img_train, dtype=np.float32)
x_train = x_train / 255.0

y_0 = np.zeros(  len(class0_img_train)  )
y_1 = np.ones(  len(class1_img_train)  )
y_train = np.concatenate( (y_0, y_1) )

y_train = to_categorical(y_train, num_classes)

x_val = np.asarray(class0_img_val + class1_img_val, dtype=np.float32)
x_val = x_val / 255.0

y_0 = np.zeros(  len(class0_img_val)  )
y_1 = np.ones(  len(class1_img_val)  )
y_val = np.concatenate( (y_0, y_1) )

y_val = to_categorical(y_val, num_classes)

# Add two dimensions to the labels so they can be used for classification and later for segmentation as well
y_train = np.expand_dims( y_train, axis=1 )
y_val = np.expand_dims( y_val, axis=1 )

y_train = np.expand_dims( y_train, axis=1 )
y_val = np.expand_dims( y_val, axis=1 )

x_train = np.expand_dims( x_train, axis=-1 )
x_val = np.expand_dims( x_val, axis=-1 )

# Class weights to fight imbalancy
# Found in https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#class_weights

neg = len(class0_img_train)
pos = len(class1_img_train)
total = neg + pos

# Scaling by total/2 helps keep the loss to a similar magnitude.
# The sum of the weights of all examples stays the same.
weight_for_0 = (1 / neg)*(total)/2.0 
weight_for_1 = (1 / pos)*(total)/2.0

class_weight = {0: weight_for_0, 1: weight_for_1}

print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))


print("#######################\n"
      "#  DATA AUGMENTATION  #\n"
      "#######################\n")

datagen = ImageDataGenerator(
    horizontal_flip=hflips,
    vertical_flip=vflips,
    rotation_range=rotation_range,
    shear_range=k_shear_range,
    width_shift_range=k_w_shift_r,
    height_shift_range=k_h_shift_r,
    fill_mode='reflect')
datagen.fit(x_train, augment=True, seed=seed)


print("#################################\n"
      "#  BUILD AND TRAIN THE NETWORK  #\n"
      "#################################\n")

optim = tf.keras.optimizers.Adam(learning_rate=learning_rate_value)

# Create model
model = create_oztel_model_V1(optimizer=optim, loss=loss)

# Check the network created                                                     
model.summary(line_length=150)
os.makedirs(char_dir, exist_ok=True)                                            
model_name = os.path.join(char_dir, "model_plot_" + job_identifier + ".png")    
plot_model(model, to_file=model_name, show_shapes=True, show_layer_names=True)
h5_file=os.path.join(h5_dir, weight_files_prefix + previous_job_weights
                     + '_' + str(args.run_id) + '.h5')

if load_previous_weights == False:

    results = model.fit(
        datagen.flow(x_train, y_train, batch_size=batch_size_value, seed=seed, 
        shuffle=True), epochs=epochs_value, validation_data=(x_val, y_val), 
        callbacks=[earlystopper, time_callback])

    model.save(h5_file)

print("Loading model weights from h5_file: {}".format(h5_file))
model.load_weights(h5_file)

# Prediction over validation
Y_pred = model.predict( x_val )
y_pred = np.argmax(Y_pred, axis=3)
y_true = np.argmax( y_val, axis=3)

print('Confusion Matrix')
print(confusion_matrix( np.squeeze(y_true), np.squeeze(y_pred)))

print('Classification Report')
print(classification_report(np.squeeze(y_true), np.squeeze(y_pred), target_names=class_names))


print("##############################\n"
      "#  TEST ON FULL SIZE IMAGES  #\n"
      "##############################\n")

####### TEST #######
# Read the list of file names
test_input_filenames = [x for x in os.listdir( test_path ) if x.endswith(".tif")]
test_input_filenames.sort()

test_label_filenames = [x for x in os.listdir( test_mask_path ) if x.endswith(".tif")]
test_label_filenames.sort()

print( 'Test input images loaded: ' + str( len(test_input_filenames)) )
print( 'Test label images loaded: ' + str( len(test_label_filenames)) )

# Read test images
test_img = [ img_as_ubyte( io.imread( test_path + '/' + x ) ) for x in test_input_filenames ]
Y_test = [ img_as_ubyte( io.imread( test_mask_path + '/' + x ) ) for x in test_label_filenames ]
test_img = np.asarray(test_img, dtype=np.float32)
X_test = test_img/255 # normalize between 0 and 1
X_test = np.expand_dims( X_test, axis=-1 )
Y_test = np.asarray(Y_test, dtype=np.float32)
Y_test = Y_test/255
Y_test = np.expand_dims( Y_test, axis=-1 )

score_per_crop = -1
loss_per_crop = -1
jac_per_crop = -1

jac_per_image = -1
voc_per_image = -1

smo_jac_per_image = -1
smo_voc_per_image = -1

zfil_jac_per_image = -1
zfil_voc_per_image = -1

smo_zfil_jac_per_image = -1
smo_zfil_voc_per_image = -1

jac_50ov = -1
voc_50ov = -1 

ens_jac_50ov = -1
ens_voc_50ov = -1

ens_zfil_jac_50ov = -1
ens_zfil_voc_50ov = -1

print("########################\n"
      "# Metrics (full image) #\n"
      "########################\n")

print("Making the predictions on test data . . .")
preds = model.predict(X_test)
preds_test_full = np.zeros( [165,768,1024,2], dtype=np.float32)
for i in range( 0, len(preds) ):
    preds_test_full[i] = transform.resize( preds[i], [768,1024,2], order=3 )

print("Saving predicted images . . .")
save_img(Y=(np.expand_dims(preds_test_full[...,1],-1) > 0.5).astype(np.uint8),
         mask_dir=result_bin_dir_full, prefix="test_out_bin_full")
save_img(Y=np.expand_dims(preds_test_full[...,1],-1), mask_dir=result_no_bin_dir_full,
         prefix="test_out_no_bin_full")

print("Calculate metrics (full image) . . .")
jac_full = jaccard_index_numpy(Y_test, (np.expand_dims(preds_test_full[...,1],-1) > 0.5).astype(np.uint8))
voc_full = voc_calculation(Y_test, (np.expand_dims(preds_test_full[...,1],-1) > 0.5).astype(np.uint8),
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


####### OZTEL POST-PROCESSING #######
print("~~~~ Spurious Detection (full image) ~~~~")
spu_preds_test = spuriuous_detection_filter(preds_test_full, low_score_th=0.80)
spu_preds_test = (spu_preds_test).astype(np.uint8)

print("Saving spurious detection filtering resulting images . . .")
save_img(Y=spu_preds_test, mask_dir=spu_dir_full, prefix="test_out_spu")

print("Calculate metrics (Spurious + full image) . . .")
spu_jac_full = jaccard_index_numpy(Y_test, spu_preds_test)
spu_voc_full = voc_calculation(Y_test, spu_preds_test, spu_jac_full)

print("~~~~ Spurious Detection + Watershed (full image) ~~~~")
wa_preds_test = watershed_refinement(spu_preds_test, X_test, open_iter = 3, 
    dilate_iter = 11, erode_iter = 11, gauss_sigma = 0, canny_sigma = 2)

print("Saving watershed resulting images . . .")
save_img(Y=(wa_preds_test).astype(np.uint8), mask_dir=wa_dir_full,
         prefix="test_out_wa")

print("Calculate metrics (Watershed + full image) . . .")
wa_jac_full = jaccard_index_numpy(Y_test, wa_preds_test)
wa_voc_full = voc_calculation(Y_test, wa_preds_test, wa_jac_full)

print("~~~~ Spurious Detection + Watershed + Z-filtering (full image) ~~~~")
spu_wa_zfil_preds_test = improve_components(wa_preds_test, depth=9 ) 

print("Saving Z-filtered images . . .")
save_img(Y=spu_wa_zfil_preds_test, mask_dir=spu_wa_zfil_dir_full,
         prefix="test_out_spu_wa_zfil")

print("Calculate metrics (Z-filtering + full image) . . .")
spu_wa_zfil_jac_full = jaccard_index_numpy(
    Y_test, (spu_wa_zfil_preds_test > 0.5).astype(np.uint8))
spu_wa_zfil_voc_full = voc_calculation(
    Y_test, (spu_wa_zfil_preds_test > 0.5).astype(np.uint8),
    spu_wa_zfil_jac_full)


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
print("Post-process: Spurious Detection + Watershed - Test IoU (full): {}".format(wa_jac_full))
print("Post-process: Spurious Detection + Watershed - VOC (full): {}".format(wa_voc_full))
print("Post-process: Spurious Detection + Watershed + Z-Filtering - Test IoU (full): {}".format(spu_wa_zfil_jac_full))
print("Post-process: Spurious Detection + Watershed + Z-Filtering - VOC (full): {}".format(spu_wa_zfil_voc_full))

if not load_previous_weights:
    scores = {}
    for name in dir():
        if not name.startswith('__') and ("_per_crop" in name or "_50ov" in name\
        or "_per_image" in name or "_full" in name):
            scores[name] = eval(name)

    create_plots(results, job_identifier, char_dir, metric=metric)

print("FINISHED JOB {} !!".format(job_identifier))

