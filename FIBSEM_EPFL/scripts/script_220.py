##########################
#        PREAMBLE        #
##########################
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Limit the number of threads
from util import *
limit_threads()

# Try to generate the results as reproducible as possible
set_seed(42)


##########################
#        IMPORTS         #
##########################

from data_kasthuri import *
from unet import *
from metrics import *
import random
import numpy as np
import keras
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from PIL import Image
import math


##########################
#    INITIAL VARIABLES   #
##########################

#### VARAIBLES THAT SHOULD NOT BE MODIFIED ####
# Take arguments
if len(sys.argv) > 1:
    gpu_selected = str(sys.argv[1])                                       
    job_id = str(sys.argv[2])                                             
    test_id = str(sys.argv[3])                                            
    job_file = job_id + '_' + test_id                                     
    log_dir = os.path.join(str(sys.argv[4]), job_id)                   

# Checks
print('job_id :', job_id)
print('GPU selected :', gpu_selected)
print('Python       :', sys.version.split('\n')[0])
print('Numpy        :', np.__version__)
print('Keras        :', keras.__version__)
print('Tensorflow   :', tf.__version__)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
if len(sys.argv) > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_selected;
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "3";

# Control variables 
crops_made = False
###############################################

# Working dir
os.chdir("/data2/dfranco/experimentosTFM/FIBSEM_EPFL")

# Image dimensions                                                              
# Note: train and test dimensions must be the same when training the network and
# making the predictions. If you do not use crop_data() with the arg force_shape
# be sure to take care of this.
img_train_width = 1024
img_train_height = 768
img_train_channels = 1
img_test_width = img_train_width
img_test_height = img_train_height
img_test_channels = img_train_channels

# Crop variables
img_width_crop = 256                                                            
img_height_crop = 256                                                           
img_channels_crop = 1 
make_crops = False
check_crop = True

# Discard variables
discard_cropped_images = False
d_percentage_value = 0.05

# Data augmentation variables
custom_da = True
aug_examples = True

# General parameters
batch_size_value = 1
momentum_value = 0.99
learning_rate_value = 0.01
epochs_value = 360

# Define time callback                                                          
time_callback = TimeHistory()

# Paths to data and results                                             
TRAIN_PATH = os.path.join('data','train', 'x')                         
TRAIN_MASK_PATH = os.path.join('data', 'train', 'y')                    
TEST_PATH = os.path.join('data', 'test', 'x')                           
TEST_MASK_PATH = os.path.join('data', 'test', 'y')                      

if make_crops == True:
    TRAIN_CROP_DISCARD_PATH = os.path.join('data_d', 'kas_'
                              + str(d_percentage_value), 'train', 'x')
    TRAIN_CROP_DISCARD_MASK_PATH = os.path.join('data_d', 'kas_'
                                   + str(d_percentage_value), 'train', 'y')
    TEST_CROP_DISCARD_PATH = os.path.join('data_d', 'kas_'
                             + str(d_percentage_value), 'test', 'x')
    TEST_CROP_DISCARD_MASK_PATH = os.path.join('data_d', 'kas_'
                                  + str(d_percentage_value), 'test', 'y')
RESULT_DIR = os.path.join('results', 'results_' + job_id)
CHAR_DIR='charts'
H5_DIR='h5_files'


#############################################
#    PREPARE DATASET IF DISCARD IS ACTIVE   #
#############################################

# The first time the dataset will be prepared for future runs if it is not 
# created yet
if discard_cropped_images == True and make_crops == True \
   and not os.path.exists(TRAIN_CROP_DISCARD_PATH):

    crops_made = True

    # Load data
    X_train, Y_train, \
    X_test, Y_test = load_data(TRAIN_PATH, TRAIN_MASK_PATH, TEST_PATH,
                               TEST_MASK_PATH, 
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
        check_crops(X_train, Y_train, [img_train_width, img_train_height],
                    num_examples=3, out_dir="check_crops", job_id=job_id, 
                    grid=True)
   
    # Create folders and save the images for future runs 
    print("\nSaving cropped images for future runs . . .", flush=True)
    os.makedirs(TRAIN_CROP_DISCARD_PATH)
    os.makedirs(TRAIN_CROP_DISCARD_MASK_PATH)
    os.makedirs(TEST_CROP_DISCARD_PATH)
    os.makedirs(TEST_CROP_DISCARD_MASK_PATH)
    for i in tqdm(range(0,X_train.shape[0])):                                          
        im = Image.fromarray(X_train[i,:,:,0])                           
        im = im.convert('L')                                                    
        im.save(os.path.join(TRAIN_CROP_DISCARD_PATH,
                             "x_train_" + str(i) + ".png")) 

        im = Image.fromarray(Y_train[i,:,:,0]*255)                                  
        im = im.convert('L')                                                    
        im.save(os.path.join(TRAIN_CROP_DISCARD_MASK_PATH,                           
                             "y_train_" + str(i) + ".png"))

    for i in tqdm(range(0,X_test.shape[0])):                                         
        im = Image.fromarray(X_test[i,:,:,0])                                  
        im = im.convert('L')                                                    
        im.save(os.path.join(TEST_CROP_DISCARD_PATH,                           
                             "x_test_" + str(i) + ".png"))                     

        im = Image.fromarray(Y_test[i,:,:,0]*255)                             
        im = im.convert('L')                                                    
        im.save(os.path.join(TEST_CROP_DISCARD_MASK_PATH,                      
                             "y_test_" + str(i) + ".png"))
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
    TRAIN_PATH = TRAIN_CROP_DISCARD_PATH
    TRAIN_MASK_PATH = TRAIN_CROP_DISCARD_MASK_PATH
    TEST_PATH = TEST_CROP_DISCARD_PATH
    TEST_MASK_PATH = TEST_CROP_DISCARD_MASK_PATH
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
X_test, Y_test = load_data(TRAIN_PATH, TRAIN_MASK_PATH, TEST_PATH, 
                           TEST_MASK_PATH, [img_train_width, img_train_height,
                           img_train_channels], [img_test_width, img_test_height,
                           img_test_channels])

# Crop the data to the desired size
if make_crops == True and crops_made == False:
    X_train, Y_train, _ = crop_data(X_train, Y_train, img_width_crop,
                                    img_height_crop)
    X_val, Y_val, _ = crop_data(X_val, Y_val, img_width_crop, img_height_crop)
    X_test, Y_test, _ = crop_data(X_test, Y_test, img_width_crop, img_height_crop)

    if check_crop == True:
        check_crops(X_train, Y_train, [img_train_width, img_train_height], 
                    num_examples=3, out_dir="check_crops", job_id=job_id, 
                    grid=True)
    
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
                                                        job_id=job_id)
else:
    data_gen_args = dict(X=X_train, Y=Y_train, batch_size=batch_size_value,
                         dim=(img_height,img_width), n_channels=1,
                         shuffle=True, da=True, e_prob=0.7, elastic=True,
                         vflip=False, hflip=False, rotation=True)

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

print("\nCreating the newtwok . . .", flush=True)
model = U_Net([img_height, img_width, img_channels], numInitChannels=32)

sdg = keras.optimizers.SGD(lr=learning_rate_value, momentum=momentum_value,
                           decay=0.0, nesterov=False)

model.compile(optimizer=sdg, loss='binary_crossentropy', metrics=[jaccard_index])
model.summary()

# Fit model
earlystopper = EarlyStopping(patience=50, verbose=1, restore_best_weights=True)

if not os.path.exists(H5_DIR):                                      
    os.makedirs(H5_DIR)
checkpointer = ModelCheckpoint(os.path.join(H5_DIR, 'model.fibsem_' + job_file 
                                                    +'.h5'),
                               verbose=1, save_best_only=True)

results = model.fit_generator(train_generator, validation_data=val_generator,
                              validation_steps=math.ceil(len(X_val)/batch_size_value),
                              steps_per_epoch=math.ceil(len(X_train)/batch_size_value),
                              epochs=epochs_value, callbacks=[earlystopper, 
                                                              checkpointer,
                                                              time_callback])


#####################
#    PREDICTION     #
#####################

# Evaluate to obtain the loss and jaccard index                         
print("Evaluating test data . . .")                                     
score = model.evaluate(X_test, Y_test, batch_size=batch_size_value, verbose=1)                                       
                                                                        
# Predict on test                                                       
print("Making the predictions on test data . . .")                      
preds_test = model.predict(X_test, batch_size=batch_size_value, verbose=1) 

# Threshold predictions
preds_test_t = (preds_test > 0.5).astype(np.uint8)

# Save the resulting images 
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)
if len(sys.argv) > 1 and test_id == "1":
    print("Saving predicted images . . .")
    for i in range(0,len(preds_test)):
        im = Image.fromarray(preds_test[i,:,:,0]*255)
        im = im.convert('L')
        im.save(os.path.join(RESULT_DIR,"test_out" + str(i) + ".png"))


#####################
#  SCORES OBTAINED  #
#####################

# VOC
print("Calculating VOC . . .")
voc = voc_calculation(Y_test, preds_test_t, score[1])

# Time
print("Epoch average time: ", np.mean(time_callback.times))
print("Train time (s):", np.sum(time_callback.times))

# Loss and metric
print("Train loss:", np.min(results.history['loss']))
print("Validation loss:", np.min(results.history['val_loss']))
print("Test loss:", score[0])
print("Train jaccard_index:", np.max(results.history['jaccard_index']))
print("Validation jaccard_index:", np.max(results.history['val_jaccard_index']))
print("Test jaccard_index:", score[1])
print("VOC: ", voc)
print("Epoch number:", len(results.history['val_loss']))

# If we are running multiple tests store the results
if len(sys.argv) > 1:

    store_history(results, score, voc, time_callback, log_dir, job_file)

    if test_id == "1":
        create_plots(results, job_id, CHAR_DIR)

