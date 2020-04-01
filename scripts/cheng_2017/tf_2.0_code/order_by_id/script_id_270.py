##########################
#        PREAMBLE        #
##########################

import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '..'))

# Limit the number of threads
from util import limit_threads, set_seed, create_plots, store_history,\
                 TimeHistory, Print
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
from data import load_data, crop_data, mix_data, check_crops, keras_da_generator, ImageDataGenerator
from unet import U_Net
from metrics import jaccard_index, jaccard_index_numpy, voc_calculation, DET_calculation
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.morphology import label
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from PIL import Image
from tqdm import tqdm
from smooth_tiled_predictions import predict_img_with_smooth_windowing


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
train_path = os.path.join('data', 'train', 'x')
train_mask_path = os.path.join('data', 'train', 'y')
test_path = os.path.join('data','test', 'x')
test_mask_path = os.path.join('data', 'test', 'y')
# Note: train and test dimensions must be the same when training the network and
# making the predictions. If you do not use crop_data() with the arg force_shape
# be sure to take care of this.
img_train_width = 1024
img_train_height = 768
img_train_channels = 1
img_test_width = 1024
img_test_height = 768
img_test_channels = 1
original_test_shape=[img_test_width, img_test_height]

# Crop variables
img_width_crop = 256                                                            
img_height_crop = 256                                                           
img_channels_crop = 1 
make_crops = True                                                               
check_crop = True

# Discard variables
discard_cropped_images = False
d_percentage_value = 0.05
train_crop_discard_path = os.path.join('data_d', 'kas_' + str(d_percentage_value), 'train', 'x')
train_crop_discard_mask_path = os.path.join('data_d', 'kas_' + str(d_percentage_value), 'train', 'y')
test_crop_discard_path = os.path.join('data_d', 'kas_' + str(d_percentage_value), 'test', 'x')
test_crop_discard_mask_path = os.path.join('data_d', 'kas_' + str(d_percentage_value), 'test', 'y')

# Data augmentation variables
normalize_data = False
norm_value_forced = -1
custom_da = True
aug_examples = True
keras_zoom = False

# Load preoviously generated model weigths
load_previous_weights = False

# General parameters
batch_size_value = 6
momentum_value = 0.99
learning_rate_value = 0.001
epochs_value = 360

# Define time callback                                                          
time_callback = TimeHistory()

# Post-processing
post_process = True

# DET metric variables
det_eval_ge_path = os.path.join('cell_challenge_eval', 'general_fib')
det_eval_path = os.path.join('cell_challenge_eval', job_id, job_file)
det_eval_post_path = os.path.join('cell_challenge_eval', job_id, job_file + '_s')
det_bin = os.path.join(script_dir, '..', 'cell_cha_eval' ,'Linux', 'DETMeasure')
n_dig = "3"

# Paths of the results                                             
result_dir = os.path.join('results', 'results_' + job_id)
char_dir = 'charts'
h5_dir = 'h5_files'


#############################################
#    PREPARE DATASET IF DISCARD IS ACTIVE   #
#############################################

# The first time the dataset will be prepared for future runs if it is not 
# created yet
if discard_cropped_images == True and make_crops == True \
   and not os.path.exists(train_crop_discard_path):

    crops_made = True

    # Load data
    X_train, Y_train, \
    X_test, Y_test, norm_value = load_data(train_path, train_mask_path, test_path,
                               test_mask_path, 
                               [img_train_width, img_train_height, img_train_channels],
                               [img_test_width, img_test_height, img_test_channels],
                               create_val=False)

    # Crop the data to the desired size
    X_train, Y_train, f_shape = crop_data(X_train, Y_train, img_width_crop, 
                                          img_height_crop, discard=True,     
                                          d_percentage=d_percentage_value)
    X_test, Y_test, _ = crop_data(X_test, Y_test, img_width_crop, img_height_crop,
                                  force_shape=f_shape)
    if check_crop == True:
        check_crops(X_train, [img_train_width, img_train_height], num_examples=3,
                    out_dir="check_crops", job_id=job_id, suffix="_x_", grid=True)
        check_crops(Y_train, [img_train_width, img_train_height], num_examples=3,
                    out_dir="check_crops", job_id=job_id, suffix="_y_", grid=True)
   
    # Create folders and save the images for future runs 
    Print("Saving cropped images for future runs . . .")
    os.makedirs(train_crop_discard_path)
    os.makedirs(train_crop_discard_mask_path)
    os.makedirs(test_crop_discard_path)
    os.makedirs(test_crop_discard_mask_path)
    for i in tqdm(range(0,X_train.shape[0])):                                          
        im = Image.fromarray(X_train[i,:,:,0])                           
        im = im.convert('L')                                                    
        im.save(os.path.join(train_crop_discard_path, "x_train_" + str(i) 
                             + ".png")) 

        im = Image.fromarray(Y_train[i,:,:,0]*255)                                  
        im = im.convert('L')                                                    
        im.save(os.path.join(train_crop_discard_mask_path, "y_train_" + str(i) 
                             + ".png"))

    for i in tqdm(range(0,X_test.shape[0])):                                         
        im = Image.fromarray(X_test[i,:,:,0])                                  
        im = im.convert('L')                                                    
        im.save(os.path.join(test_crop_discard_path, "x_test_" + str(i) + ".png"))                     

        im = Image.fromarray(Y_test[i,:,:,0]*255)                             
        im = im.convert('L')                                                    
        im.save(os.path.join(test_crop_discard_mask_path, "y_test_" + str(i) 
                             + ".png"))
    del X_train, Y_train, X_test, Y_test
   
    # Update shapes 
    img_train_width = img_width_crop
    img_train_height = img_height_crop
    img_train_channels = img_channels_crop
    img_test_width = img_width_crop
    img_test_height = img_height_crop
    img_test_channels = img_channels_crop

# For the rest of runs that are not the first that prepares the dataset when 
# discard is active some varaibles must be set as if it would made the crops
if make_crops == True and discard_cropped_images == True:
    train_path = train_crop_discard_path
    train_mask_path = train_crop_discard_mask_path
    test_path = test_crop_discard_path
    test_mask_path = test_crop_discard_mask_path
    img_train_width = img_width_crop
    img_train_height = img_height_crop
    img_train_channels = img_channels_crop
    img_test_width = img_width_crop
    img_test_height = img_height_crop
    img_test_channels = img_channels_crop
    crops_made = True


##########################                                                      
#       LOAD DATA        #                                                      
##########################

X_train, Y_train, \
X_val, Y_val, \
X_test, Y_test, norm_value = load_data(train_path, train_mask_path, test_path, 
                           test_mask_path, [img_train_width, img_train_height,
                           img_train_channels], [img_test_width, img_test_height,
                           img_test_channels])
# Nomalize the data
if normalize_data == True:
    if norm_value_forced != -1: 
        Print("Forced normalization value to " + str(norm_value_forced))
        norm_value = norm_value_forced
    else:
        Print("Normalization value calculated: " + str(norm_value_forced))
    X_train -= int(norm_value)
    X_val -= int(norm_value)
    X_test -= int(norm_value)
    
# Crop the data to the desired size
if make_crops == True and crops_made == False:
    X_train, Y_train, _ = crop_data(X_train, Y_train, img_width_crop,
                                    img_height_crop)
    X_val, Y_val, _ = crop_data(X_val, Y_val, img_width_crop, img_height_crop)
    X_test, Y_test, _ = crop_data(X_test, Y_test, img_width_crop, img_height_crop)

    if check_crop == True:
        check_crops(X_train, [img_train_width, img_train_height], num_examples=3,
                    out_dir="check_crops", job_id=job_id, suffix="_x_", grid=True)
        check_crops(Y_train, [img_train_width, img_train_height], num_examples=3,
                    out_dir="check_crops", job_id=job_id, suffix="_y_", grid=True)
    
    img_width = img_width_crop
    img_height = img_height_crop
    img_channels = img_channels_crop
else:                                                                           
    img_width = img_train_width                                                 
    img_height = img_train_height                                               
    img_channels = img_train_channels


##########################
#    DATA AUGMENTATION   #
##########################

if custom_da == False:
    train_generator, val_generator = keras_da_generator(X_train, Y_train, X_val,
                                                        Y_val, batch_size_value,
                                                        preproc_function=False,
                                                        save_examples=aug_examples,
                                                        job_id=job_id, 
                                                        zoom=keras_zoom)
else:
    data_gen_args = dict(X=X_train, Y=Y_train, batch_size=batch_size_value,
                         dim=(img_height,img_width), n_channels=1,
                         shuffle=True, da=True, e_prob=0.4, elastic=True,
                         vflip=True, hflip=True, rotation=True)

    data_gen_val_args = dict(X=X_val, Y=Y_val, batch_size=batch_size_value,
                             dim=(img_height,img_width), n_channels=1,
                             shuffle=False, da=False)

    train_generator = ImageDataGenerator(**data_gen_args)
    val_generator = ImageDataGenerator(**data_gen_val_args)

    # Generate examples of data augmentation
    if aug_examples == True:
        train_generator.flow_on_examples(10, job_id=job_id)


##########################
#    BUILD THE NETWORK   #
##########################

Print("Creating the network . . .")
model = U_Net([img_height, img_width, img_channels], numInitChannels=32)

sdg = keras.optimizers.SGD(lr=learning_rate_value, momentum=momentum_value,
                           decay=0.0, nesterov=False)

model.compile(optimizer=sdg, loss='binary_crossentropy', metrics=[jaccard_index])
model.summary()

if load_previous_weights == False:
    # Fit model
    earlystopper = EarlyStopping(patience=50, verbose=1, restore_best_weights=True)
    
    if not os.path.exists(h5_dir):                                      
        os.makedirs(h5_dir)
    checkpointer = ModelCheckpoint(os.path.join(h5_dir, 'model.fibsem_' + job_file 
                                                        +'.h5'),
                                   verbose=1, save_best_only=True)
    
    results = model.fit_generator(train_generator, validation_data=val_generator,
                                  validation_steps=math.ceil(len(X_val)/batch_size_value),
                                  steps_per_epoch=math.ceil(len(X_train)/batch_size_value),
                                  epochs=epochs_value, callbacks=[earlystopper, 
                                                                  checkpointer,
                                                                  time_callback])
else:
    h5_file=os.path.join(h5_dir, 'model.fibsem_' + job_id + '_' + test_id + '.h5')
    Print("Loading model weights from h5_file: " + h5_file)
    model.load_weights(h5_file)


#####################
#    PREDICTION     #
#####################

# Evaluate to obtain the loss value (the metric value will be discarded)
Print("Evaluating test data . . .")
score = model.evaluate(X_test, Y_test, batch_size=batch_size_value, verbose=1)

# Predict on test
Print("Making the predictions on test data . . .")
preds_test = model.predict(X_test, batch_size=batch_size_value, verbose=1)

# Threshold images
bin_preds_test = (preds_test > 0.5).astype(np.uint8)

# Reconstruct the data to the original shape and calculate Jaccard
h_num = int(original_test_shape[0] / bin_preds_test.shape[1]) \
        + (original_test_shape[0] % bin_preds_test.shape[1] > 0)
v_num = int(original_test_shape[1] / bin_preds_test.shape[2]) \
        + (original_test_shape[1] % bin_preds_test.shape[2] > 0)

# To calculate the jaccard (binarized)
recons_preds_test = mix_data(bin_preds_test,
                             math.ceil(bin_preds_test.shape[0]/(h_num*v_num)),
                             out_shape=[h_num, v_num], grid=False)

# To save the probabilities (no binarized)
recons_no_bin_preds_test = mix_data(preds_test*255,
                                    math.ceil(preds_test.shape[0]/(h_num*v_num)),
                                    out_shape=[h_num, v_num], grid=False)
recons_no_bin_preds_test = recons_no_bin_preds_test.astype(float)/255

Y_test = mix_data(Y_test, math.ceil(Y_test.shape[0]/(h_num*v_num)),
                  out_shape=[h_num, v_num], grid=False)
Print("The shape of the test data reconstructed is " + str(Y_test.shape))

# Metrics (jaccard + VOC + DET)
Print("Calculate metrics . . .")
score[1] = jaccard_index_numpy(Y_test, recons_preds_test)
voc = voc_calculation(Y_test, recons_preds_test, score[1])
det = DET_calculation(Y_test, recons_preds_test, det_eval_ge_path, det_eval_path,
                      det_bin, n_dig, job_id)

# Save output images
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
if len(sys.argv) > 1 and test_id == "1":
    Print("Saving predicted images . . .")
    for i in range(0,len(recons_no_bin_preds_test)):
        im = Image.fromarray(recons_no_bin_preds_test[i,:,:,0]*255)
        im = im.convert('L')
        im.save(os.path.join(result_dir,"test_out" + str(i) + ".png"))


####################
#  POST-PROCESING  #
####################

if post_process == True and make_crops == True:
    Print("Post processing active . . .")
    X_test = mix_data(X_test, math.ceil(X_test.shape[0]/(h_num*v_num)),
                  out_shape=[h_num, v_num], grid=False)

    Y_test_smooth = np.zeros(X_test.shape, dtype=(np.uint8))

    Print("1-Smoothing crops")
    start_time = time.time()
    for i in tqdm(range(0,len(X_test))):
        predictions_smooth = predict_img_with_smooth_windowing(
            X_test[i,:,:,:],
            window_size=img_width_crop,
            subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
            nb_classes=1,
            pred_func=(
                lambda img_batch_subdiv: model.predict(img_batch_subdiv)
            )
        )
        Y_test_smooth[i] = (predictions_smooth > 0.5).astype(np.uint8)

        if len(sys.argv) > 1 and test_id == "1":
            im = Image.fromarray(predictions_smooth[:,:,0]*255)
            im = im.convert('L')
            im.save(os.path.join(result_dir,"test_out_smooth_" + str(i) + ".png"))

    elapsed_time = time.time() - start_time                             
    Print("Time smoothing crops: " + str(elapsed_time))

    # Metrics (jaccard + VOC + DET)
    Print("Calculate metrics . . .")
    smooth_score = jaccard_index_numpy(Y_test, Y_test_smooth)
    smooth_voc = voc_calculation(Y_test, Y_test_smooth, smooth_score)
    smooth_det = DET_calculation(Y_test, Y_test_smooth, det_eval_ge_path,
                                 det_eval_post_path, det_bin, n_dig, job_id)

    Print("Finish post-processing")


####################################
#  PRINT AND SAVE SCORES OBTAINED  #
####################################

if load_previous_weights == False:
    # Time
    Print("Epoch average time: " + str(np.mean(time_callback.times)))
    Print("Epoch number: " +  str(len(results.history['val_loss'])))
    Print("Train time (s): " + str(np.sum(time_callback.times)))

    # Loss and metric
    Print("Train loss: " + str(np.min(results.history['loss'])))
    Print("Train jaccard_index: " + str(np.max(results.history['jaccard_index'])))
    Print("Validation loss: " + str(np.min(results.history['val_loss'])))
    Print("Validation jaccard_index: " + str(np.max(results.history['val_jaccard_index'])))

Print("Test loss: " + str(score[0]))
Print("Test jaccard_index: " + str(score[1]))
Print("VOC: " + str(voc))
Print("DET: " + str(det))

if load_previous_weights == False:
    # If we are running multiple tests store the results
    if len(sys.argv) > 1:

        if post_process == True:
            store_history(results, score, voc, det, time_callback, log_dir,
                          job_file, smooth_score=smooth_score, 
                          smooth_voc=smooth_voc, smooth_det=smooth_det)
        else:
            store_history(results, score, voc, det, time_callback, log_dir,
                          job_file)

        if test_id == "1":
            create_plots(results, job_id, char_dir)

if post_process == True and make_crops == True:
    Print("Post-process: SMOOTH - Test jaccard_index: " + str(smooth_score))
    Print("Post-process: SMOOTH - VOC: " + str(smooth_voc))
    Print("Post-process: SMOOTH - DET: " + str(smooth_det))

Print("FINISHED JOB " + job_file + " !!")
