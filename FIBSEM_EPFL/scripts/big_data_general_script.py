##########################
#        PREAMBLE        #
##########################

import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '..', 'code'))

# Limit the number of threads
from util import limit_threads, set_seed, create_plots, store_history,\
                 TimeHistory, Print, threshold_plots
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
                 merge_data_with_overlap, calculate_z_filtering
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
#    INITIAL VARIABLES   #
##########################

#### VARAIBLES THAT SHOULD NOT BE MODIFIED ####
# Take arguments
gpu_selected = str(sys.argv[1])                                       
job_id = str(sys.argv[2])                                             
test_id = str(sys.argv[3])                                            
job_file = job_id + '_' + test_id                                     
base_work_dir = str(sys.argv[4])
log_dir = os.path.join(base_work_dir, 'logs', job_id)
h5_dir = os.path.join(base_work_dir, 'h5_files')

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
###############################################

# Working dir
os.chdir(base_work_dir)

# Dataset variables
train_path = os.path.join('casser_datasets', 'human', 'histogram_matching', 'toy', 'train', 'x')
train_mask_path = os.path.join('casser_datasets', 'human', 'histogram_matching', 'toy', 'train', 'y')
val_path = os.path.join('casser_datasets', 'human', 'histogram_matching', 'toy', 'val', 'x')
val_mask_path = os.path.join('casser_datasets', 'human', 'histogram_matching', 'toy', 'val', 'y')
test_path = os.path.join('casser_datasets', 'human', 'histogram_matching', 'toy', 'test', 'x')
test_mask_path = os.path.join('casser_datasets', 'human', 'histogram_matching', 'toy', 'test', 'y')
complete_test_path = os.path.join('casser_datasets', 'human', 'histogram_matching', 'toy', 'complete', 'x')
# Note: train and test dimensions must be the same when training the network and
# making the predictions. Be sure to take care of this if you are not going to
# use crop_data() with the arg force_shape, as this function resolves the 
# problem creating always crops of the same dimension
img_train_shape =  [256, 256, 1] 
img_test_shape = [256, 256, 1]
original_test_shape = [4096, 4096, 1]

# Big data variables
validation_percentage = 0.1
data_paths = []
data_paths.append(train_path)
data_paths.append(train_mask_path)
data_paths.append(val_path)
data_paths.append(val_mask_path)
data_paths.append(test_path)
data_paths.append(test_mask_path)
data_paths.append(complete_test_path)

# Extra datasets variables
extra_datasets_data_list = []
extra_datasets_mask_list = []
extra_datasets_data_dim_list = []
# Example of use:
#extra_datasets_data_list.append(os.path.join('kasthuri_pp', 'reshaped_fibsem', 'train', 'x'))
#extra_datasets_mask_list.append(os.path.join('kasthuri_pp', 'reshaped_fibsem', 'train', 'y'))
#extra_datasets_data_dim_list.append([877, 967, 1])

# Crop variables
#crop_shape = [256, 256, 1]
#make_crops = False
#check_crop = False
#random_crops_in_DA = False # No compatible with make_crops                                                        
#test_ov_crops = 8 # Only active with random_crops_in_DA
#probability_map = False # Only active with random_crops_in_DA                       
#w_foreground = 0.94 # Only active with probability_map
#w_background = 0.06 # Only active with probability_map

# Discard variables
#discard_cropped_images = False
#d_percentage_value = 0.05
#train_crop_discard_path = os.path.join('data_d', 'kas_' + str(d_percentage_value), 'train', 'x')
#train_crop_discard_mask_path = os.path.join('data_d', 'kas_' + str(d_percentage_value), 'train', 'y')
#test_crop_discard_path = os.path.join('data_d', 'kas_' + str(d_percentage_value), 'test', 'x')
#test_crop_discard_mask_path = os.path.join('data_d', 'kas_' + str(d_percentage_value), 'test', 'y')

# Data augmentation variables
#normalize_data = False
#norm_value_forced = -1
custom_da = False
aug_examples = True # Keras and Custom DA
keras_zoom = False # Only Keras DA
w_shift_r = 0.0 # Only Keras DA
h_shift_r = 0.0 # Only Keras DA
shear_range = 0.0 # Only Keras DA
brightness_range = [1.0, 1.0] # Keras and Custom DA
median_filter_size = [0, 0] # Only Custom DA

# Extra train data generation
#duplicate_train = 0
#extra_train_data = 0 # Applied after duplicate_train

# Load preoviously generated model weigths
load_previous_weights = False

# General parameters
batch_size_value = 6
momentum_value = 0.99
learning_rate_value = 0.001
epochs_value = 360
make_threshold_plots = False

# Define time callback                                                          
time_callback = TimeHistory()

# Post-processing
post_process = True

# DET metric variables
det_eval_ge_path = os.path.join('cell_challenge_eval', 'general_c_human_hismat')
det_eval_path = os.path.join('cell_challenge_eval', job_id, job_file)
det_eval_post_path = os.path.join('cell_challenge_eval', job_id, job_file + '_s')
det_bin = os.path.join(script_dir, '..', 'code', 'cell_cha_eval' ,'Linux', 'DETMeasure')
n_dig = "3"

# Paths of the results                                             
result_dir = os.path.join('results', 'results_' + job_id)
char_dir = 'charts'
h5_dir = 'h5_files'


##########################
#    DATA AUGMENTATION   #
##########################

Print("##################\n" + "#    DATA AUG    #\n" + "##################\n")

if custom_da == False:                                                          
    Print("Keras DA selected")

    # Keras Data Augmentation                                                   
    train_generator, val_generator, \
    test_generator, complete_generator, \
    n_train_samples, n_val_samples, \
    n_test_samples  = keras_da_generator(data_paths=data_paths, 
                                        target_size=(img_train_shape[0], img_train_shape[1]),
                                        c_target_size=(original_test_shape[0], original_test_shape[1]),
                                        batch_size_value=batch_size_value,
                                        val=True, save_examples=aug_examples, 
                                        job_id=job_id, shuffle=False, 
                                        zoom=keras_zoom, w_shift_r=w_shift_r, 
                                        h_shift_r=h_shift_r, 
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
                         shuffle=True, da=True, e_prob=0.0, elastic=False,      
                         vflip=True, hflip=True, rotation90=False,              
                         rotation_range=180, brightness_range=brightness_range,
                         median_filter_size=median_filter_size,
                         random_crops_in_DA=random_crops_in_DA, 
                         crop_length=crop_shape[0], prob_map=probability_map,
                         train_prob=train_prob)                            
                                                                                
    data_gen_val_args = dict(X=X_val, Y=Y_val, batch_size=batch_size_value,     
                             dim=(img_height,img_width), n_channels=1,          
                             shuffle=False, da=False,                           
                             random_crops_in_DA=random_crops_in_DA,                   
                             crop_length=crop_shape[0], val=True)              
                                                                                
    train_generator = ImageDataGenerator(**data_gen_args)                       
    val_generator = ImageDataGenerator(**data_gen_val_args)                     
                                                                                
    # Generate examples of data augmentation                                    
    if aug_examples == True:                                                    
        train_generator.get_transformed_samples(10, save_to_dir=True,           
                                                job_id=os.path.join(job_id, test_id))

##########################
#    BUILD THE NETWORK   #
##########################

Print("###################\n" + "#  TRAIN PROCESS  #\n" + "###################\n")

Print("Creating the network . . .")
model = U_Net(img_train_shape, numInitChannels=32)

sgd = keras.optimizers.SGD(lr=learning_rate_value, momentum=momentum_value,
                           decay=0.0, nesterov=False)

model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=[jaccard_index])
model.summary()

if load_previous_weights == False:
    earlystopper = EarlyStopping(patience=1, verbose=1, 
                                 restore_best_weights=True)
    
    if not os.path.exists(h5_dir):                                      
        os.makedirs(h5_dir)
    checkpointer = ModelCheckpoint(os.path.join(h5_dir, 'model.fibsem_' + job_file + '.h5'),
                                   verbose=1, save_best_only=True)
   
    Print("Training the model . . .")
    results = model.fit_generator(train_generator, validation_data=val_generator,
                                  validation_steps=math.ceil(n_val_samples/batch_size_value),
                                  steps_per_epoch=math.ceil(n_train_samples/batch_size_value),
                                  epochs=epochs_value, 
                                  callbacks=[earlystopper, checkpointer, time_callback])
else:
    h5_file=os.path.join(h5_dir, 'model.fibsem_' + job_id + '_' + test_id + '.h5')
    Print("Loading model weights from h5_file: " + h5_file)
    model.load_weights(h5_file)


#####################
#     INFERENCE     #
#####################

Print("##################\n" + "#    INFERENCE   #\n" + "##################\n")

# Evaluate to obtain the loss value and the Jaccard index (per crop)
Print("Evaluating test data . . .")
score = model.evaluate_generator(test_generator, steps=math.ceil(n_test_samples/batch_size_value), verbose=1)

# Predict on test
Print("Making the predictions on test data . . .")
preds_test = model.predict_generator(test_generator, steps=math.ceil(n_test_samples/batch_size_value),
                                     verbose=1)

# Threshold images
bin_preds_test = (preds_test > 0.5).astype(np.uint8)

# Load Y_test
Print("Loading test masks to make the predictions . . .")
test_mask_ids = sorted(next(os.walk(os.path.join(test_mask_path, 'y')))[2])

Y_test = np.zeros((len(test_mask_ids), img_test_shape[1], img_test_shape[0],
                   img_test_shape[2]), dtype=np.int16)

for n, id_ in tqdm(enumerate(test_mask_ids), total=len(test_mask_ids)):
  mask = imread(os.path.join(test_mask_path, 'y', id_))
  if len(mask.shape) == 2:
    mask = np.expand_dims(mask, axis=-1)
  Y_test[n,:,:,:] = mask

# Reconstruct the data to the original shape
Print("Calculate metrics . . .")
Print("n_test_samples: " + str(n_test_samples))
Print("Y_test.shape: " + str(Y_test.shape))
Print("bin_preds_test.shape: " + str(bin_preds_test.shape))
score[1] = jaccard_index_numpy(Y_test, bin_preds_test)
voc = voc_calculation(Y_test, bin_preds_test, score[1])
det = DET_calculation(Y_test, bin_preds_test, det_eval_ge_path,
                      det_eval_path, det_bin, n_dig, job_id)

# Save output images
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
if len(sys.argv) > 1 and test_id == "1":
    Print("Saving predicted images . . .")
    for i in range(0,len(bin_preds_test)):
        im = Image.fromarray(bin_preds_test[i,:,:,0]*255)
        im = im.convert('L')
        im.save(os.path.join(result_dir,"test_out" + str(i) + ".png"))

    
####################
#  POST-PROCESING  #
####################

Print("##################\n" + "# POST-PROCESING #\n" + "##################\n") 

Print("1) SMOOTH")
if post_process == True:

    Print("Post processing active . . .")

    Y_test_smooth = np.zeros((complete_generator.n, original_test_shape[0], 
                              original_test_shape[1], original_test_shape[2]), 
                             dtype=np.uint8)

    if not os.path.exists(result_dir):
            os.makedirs(result_dir)

    Print("Smoothing crops . . .")
    iterations = math.ceil(complete_generator.n/batch_size_value)
    cont = 0
    for i in tqdm(range(0,iterations)):
        batch = next(complete_generator)
        for im in batch:
            if cont >= complete_generator.n:
                break
            predictions_smooth = predict_img_with_smooth_windowing(
                im[0],
                window_size=img_train_shape[0],
                subdivisions=2,  
                nb_classes=1,
                pred_func=(
                    lambda img_batch_subdiv: model.predict(img_batch_subdiv)
                )
            )
            Y_test_smooth[cont] = (predictions_smooth > 0.5).astype(np.uint8)
            cont += 1

            if len(sys.argv) > 1 and test_id == "1":
                im = Image.fromarray(predictions_smooth[:,:,0])
                im = im.convert('L')
                im.save(os.path.join(result_dir,"test_out_smooth_" + str(i) + ".png"))

    # First crop the complete data 
    Y_test_smooth, _ = crop_data(Y_test_smooth, img_train_shape, tab="    ")

    # Metrics (Jaccard + VOC + DET)                                             
    Print("Calculate metrics . . .")
    smooth_score = jaccard_index_numpy(Y_test, Y_test_smooth)
    smooth_voc = voc_calculation(Y_test, Y_test_smooth, smooth_score)
    smooth_det = DET_calculation(Y_test, Y_test_smooth, det_eval_ge_path,
                                 det_eval_post_path, det_bin, n_dig, job_id)

Print("2) Z-FILTERING")
if post_process == True:

    Print("Applying Z-filter . . .")
    zfil_preds_test = calculate_z_filtering(bin_preds_test)

    Print("Calculate metrics for the Z-filtered data . . .")
    zfil_score = jaccard_index_numpy(Y_test, zfil_preds_test)
    zfil_voc = voc_calculation(Y_test, zfil_preds_test, zfil_score)
    zfil_det = DET_calculation(Y_test, zfil_preds_test, det_eval_ge_path,
                               det_eval_post_path, det_bin, n_dig, job_id)

    Print("Applying Z-filter to the smoothed data . . .")
    smooth_zfil_preds_test = calculate_z_filtering(Y_test_smooth)

    Print("Calculate metrics for the smoothed + Z-filtered data . . .")
    smo_zfil_score = jaccard_index_numpy(Y_test, smooth_zfil_preds_test)
    smo_zfil_voc = voc_calculation(Y_test, smooth_zfil_preds_test,
                                   smo_zfil_score)
    smo_zfil_det = DET_calculation(Y_test, smooth_zfil_preds_test,
                                   det_eval_ge_path, det_eval_post_path,
                                   det_bin, n_dig, job_id)

del Y_test
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
    
Print("Test (per crop) jaccard_index: " + str(score[1]))
Print("VOC: " + str(voc))
Print("DET: " + str(det))

if load_previous_weights == False:
    smooth_score = -1 if 'smooth_score' not in globals() else smooth_score
    smooth_voc = -1 if 'smooth_voc' not in globals() else smooth_voc
    smooth_det = -1 if 'smooth_det' not in globals() else smooth_det
    zfil_score = -1 if 'zfil_score' not in globals() else zfil_score
    zfil_voc = -1 if 'zfil_voc' not in globals() else zfil_voc
    zfil_det = -1 if 'zfil_det' not in globals() else zfil_det
    smo_zfil_score = -1 if 'zfil_score' not in globals() else smo_zfil_score
    smo_zfil_voc = -1 if 'zfil_voc' not in globals() else smo_zfil_voc
    smo_zfil_det = -1 if 'zfil_det' not in globals() else smo_zfil_det
    jac_per_crop = -1 if 'jac_per_crop' not in globals() else jac_per_crop

    store_history(results, jac_per_crop, score, voc, det, time_callback, log_dir,
                  job_file, smooth_score, smooth_voc, smooth_det, zfil_score,
                  zfil_voc, zfil_det, smo_zfil_score, smo_zfil_voc, smo_zfil_det)

    create_plots(results, job_id, test_id, char_dir)

if post_process == True:
    Print("Post-process: SMOOTH - Test jaccard_index: " + str(smooth_score))
    Print("Post-process: SMOOTH - VOC: " + str(smooth_voc))
    Print("Post-process: SMOOTH - DET: " + str(smooth_det))
    Print("Post-process: Z-filtering - Test jaccard_index: " + str(zfil_score))
    Print("Post-process: Z-filtering - VOC: " + str(zfil_voc))
    Print("Post-process: Z-filtering - DET: " + str(zfil_det))
    Print("Post-process: SMOOTH + Z-filtering - Test jaccard_index: "
          + str(smo_zfil_score))
    Print("Post-process: SMOOTH + Z-filtering - VOC: " + str(smo_zfil_voc))
    Print("Post-process: SMOOTH + Z-filtering - DET: " + str(smo_zfil_det))

Print("FINISHED JOB " + job_file + " !!")
