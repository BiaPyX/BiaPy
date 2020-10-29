import os
import math
#import mkl
import numpy as np
import time
import random
import matplotlib.pyplot as plt
from PIL import ImageEnhance, Image
from tqdm import tqdm
import copy
from skimage import measure
import scipy.ndimage
from skimage.segmentation import clear_border
import tensorflow as tf 
from metrics import jaccard_index, jaccard_index_numpy, voc_calculation, \
                    DET_calculation


def limit_threads(threads_number='1'):
    """Limits the number of threads for a python process.
       
       Parameters
       ---------- 
       threads_number : int, optional
           Number of threads.
    """

    print("Python process limited to {} thread".format(threads_number))

    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ["MKL_DYNAMIC"]="FALSE";
    os.environ["NUMEXPR_NUM_THREADS"]='1';
    os.environ["VECLIB_MAXIMUM_THREADS"]='1';
    os.environ["OMP_NUM_THREADS"] = '1';
    #mkl.set_num_threads(1)


def set_seed(seedValue=42, determinism=False):
    """Sets the seed on multiple python modules to obtain results as 
       reproducible as possible.

       Parameters
       ----------
       seedValue : int, optional
           Seed value.
        
       determinism : bool, optional
           To force determism. 
    """

    random.seed = seedValue
    np.random.seed(seed=seedValue)
    tf.random.set_seed(seedValue)
    os.environ["PYTHONHASHSEED"]=str(seedValue);
    if determinism:    
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
    

class TimeHistory(tf.keras.callbacks.Callback):
    """Class to record each epoch time.
    """

    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def create_plots(results, job_id, chartOutDir, metric='jaccard_index'):
    """Create loss and main metric plots with the given results.

       Parameters
       ----------
       results : Keras History object
           Record of training loss values and metrics values at successive 
           epochs. History object is returned by Keras `fit() 
           <https://keras.io/api/models/model_training_apis/#fit-method>`_ method.

       job_id : str
           Jod identifier.

       chartOutDir : str    
           Path where the charts will be stored into.
            
       metric : str, optional
           Metric used.

       Examples
       --------
       
       +-----------------------------------------+-----------------------------------------+
       | .. figure:: img/chart_loss.png          | .. figure:: img/chart_jaccard_index.png |
       |   :width: 80%                           |   :width: 80%                           |
       |   :align: center                        |   :align: center                        |
       |                                         |                                         |
       |   Loss values on each epoch             |   Jaccard index values on each epoch    |
       +-----------------------------------------+-----------------------------------------+
    """

    os.makedirs(chartOutDir, exist_ok=True)

    # For matplotlib errors in display
    os.environ['QT_QPA_PLATFORM']='offscreen'

    # Loss
    plt.plot(results.history['loss'])
    plt.plot(results.history['val_loss'])
    plt.title('Model JOBID=' + job_id + ' loss')
    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.legend(['Train loss', 'Val. loss'], loc='upper left')
    plt.savefig(os.path.join(chartOutDir, job_id + '_loss.png'))
    plt.clf()

    # Jaccard index
    plt.plot(results.history[metric])
    plt.plot(results.history['val_' + metric])
    plt.title('Model JOBID=' + job_id + " " + metric)
    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.legend(['Train metric', 'Val. metric'], loc='upper left')
    plt.savefig(os.path.join(chartOutDir, job_id + '_' + metric +'.png'))
    plt.clf()


def store_history(results, score, time_callback, log_dir, job_file, 
                  metric='jaccard_index'):
    """Stores the results obtained as csv to manipulate them later 
       and labeled in another file as historic results.

       Parameters
       ----------
       results : Keras History object
           Record of training loss values and metrics values at successive 
           epochs.

       score : Dictionary
           Contains all metrics values extracted from training and inference. 

       time_callback : ``util.TimeHistory``
           Time structure with the time of each epoch.
    """

    # Create folders and construct file names
    csv_file = os.path.join(log_dir, 'formatted', job_file)          
    history_file = os.path.join(log_dir, 'history_of_values', job_file)
    os.makedirs(os.path.join(log_dir, 'formatted'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'history_of_values'), exist_ok=True)
    
    # Store the results as csv
    try:
        os.remove(csv_file)
    except OSError:
        pass
    f = open(csv_file, 'x')

    s = ""
    s += str(np.min(results.history['loss'])) + ','
    s += str(np.min(results.history['val_loss'])) + ','
    s += str(score['loss_per_crop']) + ','
    s += str(np.max(results.history[metric])) + ','
    s += str(np.max(results.history['val_'+metric])) + ','
    s += str(score['jac_per_crop']) + ','
    s += str(score['jac_per_image']) + ','
    s += str(score['jac_50ov']) + ','
    s += str(score['jac_full']) + ','
    s += str(score['smo_score_per_image']) + ','
    s += str(score['zfil_score_per_image']) + ','
    s += str(score['smo_zfil_score_per_image']) + ','
    s += str(score['smo_score_full']) + ','
    s += str(score['zfil_score_full']) + ','
    s += str(score['spu_score_full']) + ','
    s += str(score['wa_score_full']) + ','
    s += str(score['spu_wa_zfil_score_full']) + ','
    s += str(score['voc_per_image']) + ','
    s += str(score['voc_50ov']) + ','
    s += str(score['voc_full']) + ','
    s += str(score['smo_voc_per_image']) + ','
    s += str(score['zfil_voc_per_image']) + ','
    s += str(score['smo_zfil_voc_per_image']) + ','
    s += str(score['smo_voc_full']) + ','
    s += str(score['zfil_voc_full']) + ','
    s += str(score['spu_voc_full']) + ','
    s += str(score['wa_voc_full']) + ','
    s += str(score['spu_wa_zfil_voc_full']) + ','
    s += str(score['det_per_image']) + ','
    s += str(score['det_50ov']) + ','
    s += str(score['det_full']) + ','
    s += str(score['smo_det_per_image']) + ','
    s += str(score['zfil_det_per_image']) + ','
    s += str(score['smo_zfil_det_per_image']) + ','
    s += str(score['smo_det_full']) + ','
    s += str(score['zfil_det_full']) + ','
    s += str(score['spu_det_full']) + ','
    s += str(score['wa_det_full']) + ','
    s += str(score['spu_wa_zfil_det_full']) + ','
    s += str(len(results.history['val_loss'])) + ','
    s += str(np.mean(time_callback.times)) + ','
    s += str(np.sum(time_callback.times)) + '\n'

    f.write(s)
    f.close()

    # Save all the values in case we need them in the future
    try:
        os.remove(history_file)
    except OSError:
        pass
    f = open(history_file, 'x')
    s = ""
    s += '### TRAIN LOSS ### \n'
    s += str(results.history['loss']) + '\n'
    s += '### VALIDATION LOSS ### \n'
    s += str(results.history['val_loss']) + '\n'
    s += '### TEST LOSS ### \n'
    s += str(score['loss_per_crop']) + '\n'
    s += '### TRAIN JACCARD INDEX ### \n'
    s += str(results.history[metric]) + '\n'
    s += '### VALIDATION JACCARD INDEX ### \n'
    s += str(results.history['val_'+metric]) + '\n'
    s += '### TEST JACCARD INDEX (per crop) ### \n'
    s += str(score['jac_per_crop']) + '\n'
    s += '### TEST JACCARD INDEX (per image) ### \n'
    s += str(score['jac_per_image']) + '\n'
    s += '### TEST JACCARD INDEX (per image with 50% ov) ### \n'
    s += str(score['jac_50ov']) + '\n'
    s += '### TEST JACCARD INDEX (full) ### \n'
    s += str(score['jac_full']) + '\n'
    s += '### TEST JACCARD INDEX SMOOTH (per image) ### \n'
    s += str(score['smo_score_per_image']) + '\n'
    s += '### TEST JACCARD INDEX Z-FILTERING (per image) ### \n'
    s += str(score['zfil_score_per_image']) + '\n'
    s += '### TEST JACCARD INDEX SMOOTH+Z-FILTERING (per image) ### \n'
    s += str(score['smo_zfil_score_per_image']) + '\n'
    s += '### TEST JACCARD INDEX 8-ENSEMBLE (full) ### \n'
    s += str(score['smo_score_full']) + '\n'
    s += '### TEST JACCARD INDEX Z-FILTERING (full) ### \n'
    s += str(score['zfil_score_full']) + '\n'
    s += '### TEST JACCARD INDEX SPURIOUS (full) ### \n'
    s += str(score['spu_score_full']) + '\n'
    s += '### TEST JACCARD INDEX WATERSHED (full) ### \n'
    s += str(score['wa_score_full']) + '\n'
    s += '### TEST JACCARD INDEX SPURIOUS+WATERSHED+Z-FILTERING (full) ### \n'
    s += str(score['spu_wa_zfil_score_full']) + '\n'
    s += '### VOC (per image) ### \n'
    s += str(score['voc_per_image']) + '\n'
    s += '### VOC (per image with 50% ov) ### \n'
    s += str(score['voc_50ov']) + '\n'
    s += '### VOC (full) ### \n'
    s += str(score['voc_full']) + '\n'
    s += '### VOC SMOOTH ### \n'
    s += str(score['smo_voc_per_image']) + '\n'
    s += '### VOC Z-FILTERING ### \n'
    s += str(score['zfil_voc_per_image']) + '\n'
    s += '### VOC SMOOTH+Z-FILTERING ### \n'
    s += str(score['smo_zfil_voc_per_image']) + '\n'
    s += '### VOC INDEX 8-ENSEMBLE (full) ### \n'
    s += str(score['smo_voc_full']) + '\n'
    s += '### VOC INDEX Z-FILTERING (full) ### \n'
    s += str(score['zfil_voc_full']) + '\n'
    s += '### VOC INDEX SPURIOUS (full) ### \n'
    s += str(score['spu_voc_full']) + '\n'
    s += '### VOC INDEX WATERSHED (full) ### \n'
    s += str(score['wa_voc_full']) + '\n'
    s += '### VOC INDEX SPURIOUS+WATERSHED+Z-FILTERING (full) ### \n'
    s += str(score['spu_wa_zfil_voc_full']) + '\n'
    s += '### DET (per image) ### \n'
    s += str(score['det_per_image']) + '\n'
    s += '### DET (per image with 50% ov) ### \n'
    s += str(score['det_50ov']) + '\n'
    s += '### DET SMOOTH ### \n'
    s += str(score['smo_det_per_image']) + '\n'
    s += '### DET Z-FILTERING ### \n'
    s += str(score['zfil_det_per_image']) + '\n'
    s += '### DET SMOOTH+Z-FILTERING ### \n'
    s += str(score['smo_zfil_det_per_image']) + '\n'
    s += '### DET INDEX 8-ENSEMBLE (full) ### \n'
    s += str(score['smo_det_full']) + '\n'
    s += '### DET INDEX Z-FILTERING (full) ### \n'
    s += str(score['zfil_det_full']) + '\n'
    s += '### DET INDEX SPURIOUS (full) ### \n'
    s += str(score['spu_det_full']) + '\n'
    s += '### DET INDEX WATERSHED (full) ### \n'
    s += str(score['wa_det_full']) + '\n'
    s += '### DET INDEX SPURIOUS+WATERSHED+Z-FILTERING (full) ### \n'
    s += str(score['spu_wa_zfil_det_full']) + '\n'
    f.write(s)
    f.close()


def threshold_plots(preds_test, Y_test, det_eval_ge_path, det_eval_path,
                    det_bin, n_dig, job_id, job_file, char_dir, r_val=0.5):
    """Create a plot with the different metric values binarizing the prediction
       with different thresholds, from 0.1 to 0.9.
                                                                                
       Parameters
       ----------                                                                    
       preds_test : 4D Numpy array
           Predictions made by the model. E.g. ``(image_number, x, y, channels)``.

       Y_test : 4D Numpy array
           Ground truth of the data. E.g. ``(image_number, x, y, channels)``.

       det_eval_ge_path : str
           Path where the ground truth is stored for the DET calculation.

       det_eval_path : str
           Path where the evaluation of the metric will be done.

       det_bin : str 
           Path to the DET binary.

       n_dig : int
           The number of digits used for encoding temporal indices (e.g. ``3``).
           Used by the DET calculation binary.

       job_id : str
           Id of the job.

       job_file : str
           Id and run number of the job.

       char_dir : str
           Path to store the charts generated.

       r_val : float, optional
           Threshold values to return. 

       Returns
       -------
       t_jac : float
           Value of the Jaccard index when the threshold is ``r_val``.

       t_voc : float
           Value of VOC when the threshold is ``r_val``.

       t_det : float
           Value of DET when the threshold is ``r_val``.

       Examples
       -------- 
       ::
       
           jac, voc, det = threshold_plots(
               preds_test, Y_test, det_eval_ge_path, det_eval_path, det_bin, 
               n_dig, args.job_id, '278_3', char_dir)

       Will generate 3 charts, one per each metric: IoU, VOC and DET. In the x
       axis represents the 9 different thresholds applied, that is: ``0.1, 0.2, 
       0.3, ..., 0.9``. The y axis is the value of the metric in each chart. For 
       instance, the Jaccard/IoU chart will look like this:

       .. image:: img/278_3_threshold_Jaccard.png
           :width: 60%
           :align: center

       In this example, the best value, ``0.868``, is obtained with a threshold 
       of ``0.4``.
    """

    char_dir = os.path.join(char_dir, "t_" + job_file)

    t_jac = np.zeros(9)                                                         
    t_voc = np.zeros(9)                                                         
    t_det = np.zeros(9)                                                         
    objects = []                                                                
    r_val_pos = 0
                                                                                
    for i, t in enumerate(np.arange(0.1,1.0,0.1)):                              
    
        if t == r_val:
            r_val_pos = i

        objects.append(str('%.2f' % float(t)))                                                  
                                                                                
        # Threshold images                                                      
        bin_preds_test = (preds_test > t).astype(np.uint8)                      
                                                                                
        # Metrics (Jaccard + VOC + DET)                                             
        print("Calculate metrics . . .")                                        
        t_jac[i] = jaccard_index_numpy(Y_test, bin_preds_test)
        t_voc[i] = voc_calculation(Y_test, bin_preds_test, t_jac[i])         
        t_det[i] = DET_calculation(
            Y_test, bin_preds_test, det_eval_ge_path, det_eval_path, 
            det_bin, n_dig, job_id)       
                                                                                
        print("t_jac[{}]: {}".format(i, t_jac[i]))                        
        print("t_voc[{}]: {}".format(i, t_voc[i]))                        
        print("t_det[{}]: {}".format(i, t_det[i]))

    # For matplotlib errors in display                                          
    os.environ['QT_QPA_PLATFORM']='offscreen'                                   
    
    os.makedirs(char_dir, exist_ok=True)
  
    # Plot Jaccard values   
    plt.clf()
    plt.plot(objects, t_jac)                                   
    plt.title('Model JOBID=' + job_file + ' Jaccard', y=1.08)                 
    plt.ylabel('Value')                                                         
    plt.xlabel('Threshold')                                                     
    for k, point in enumerate(zip(objects, t_jac)):
        plt.text(point[0], point[1], '%.3f' % float(t_jac[k]))
    plt.savefig(os.path.join(char_dir, job_file + '_threshold_Jaccard.png'))    
    plt.clf()                                                                   
                                                                                
    # Plot VOC values                                                           
    plt.plot(objects, t_voc)                                                    
    plt.title('Model JOBID=' + job_file + ' VOC', y=1.08)                   
    plt.ylabel('Value')                                                         
    plt.xlabel('Threshold')                                                     
    for k, point in enumerate(zip(objects, t_voc)):                              
        plt.text(point[0], point[1], '%.3f' % float(t_voc[k]))                  
    plt.savefig(os.path.join(char_dir, job_file + '_threshold_VOC.png'))    
    plt.clf()
                                                                                
    # Plot DET values                                                           
    plt.plot(objects, t_det)                                                    
    plt.title('Model JOBID=' + job_file + ' DET', y=1.08)                       
    plt.ylabel('Value')                                                         
    plt.xlabel('Threshold')                                                     
    for k, point in enumerate(zip(objects, t_det)):                              
        plt.text(point[0], point[1], '%.3f' % float(t_det[k]))                  
    plt.savefig(os.path.join(char_dir, job_file + '_threshold_DET.png'))        
    plt.clf()

    return  t_jac[r_val_pos], t_voc[r_val_pos], t_det[r_val_pos]


def array_to_img(x, data_format='channels_last', scale=True, dtype='float32'):
    """Converts a 3D Numpy array to a PIL Image instance.

       As the Keras array_to_img function in:

            `keras_preprocessing/image/utils.py <https://github.com/keras-team/keras-preprocessing/blob/28b8c9a57703b60ea7d23a196c59da1edf987ca0/keras_preprocessing/image/utils.py#L230>`_
    """
    if Image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    x = np.asarray(x, dtype=dtype)
    if x.ndim != 3:
        raise ValueError('Expected image array to have rank 3 (single image). '
                         'Got array with shape: %s' % (x.shape,))

    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Invalid data_format: %s' % data_format)

    # Original Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but target PIL image has format (width, height, channel)
    if data_format == 'channels_first':
        x = x.transpose(1, 2, 0)
    if scale:
        x = x - np.min(x)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    if x.shape[2] == 4:
        # RGBA
        return Image.fromarray(x.astype('uint8'), 'RGBA')
    elif x.shape[2] == 3:
        # RGB
        return Image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        # grayscale
        if np.max(x) > 255:
            # 32-bit signed integer grayscale image. PIL mode "I"
            return Image.fromarray(x[:, :, 0].astype('int32'), 'I')
        return Image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise ValueError('Unsupported channel number: %s' % (x.shape[2],))


def img_to_array(img, data_format='channels_last', dtype='float32'):
    """Converts a PIL Image instance to a Numpy array.

       It's a copy of the function `keras_preprocessing/image/utils.py <https://github.com/keras-team/keras-preprocessing/blob/28b8c9a57703b60ea7d23a196c59da1edf987ca0/keras_preprocessing/image/utils.py#L288>`_.
    """

    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: %s' % data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=dtype)
    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: %s' % (x.shape,))
    return x


def save_img(X=None, data_dir=None, Y=None, mask_dir=None, prefix=""):
    """Save images in the given directory. 

       Parameters
       ----------                                                                    
       X : 4D numpy array, optional
           Data to save as images. The first dimension must be the number of 
           images. E.g. ``(image_number, x, y, channels)``.
    
       data_dir : str, optional
           Path to store X images.

       Y : 4D numpy array, optional
           Masks to save as images. The first dimension must be the number of 
           images. E.g. ``(image_number, x, y, channels)``.

       mask_dir : str, optional
           Path to store Y images. 

       prefix : str, optional
           Path to store generated charts.
    """   

    if prefix is "":
        p_x = "x_"
        p_y = "y_"
    else:
        p_x = prefix + "_x_"
        p_y = prefix + "_y_"
 
    if X is not None:
        if data_dir is not None:                                                    
            os.makedirs(data_dir, exist_ok=True)
        else:       
            print("Not data_dir provided so no image will be saved!")
            return

        print("Saving images in {}".format(data_dir))

        v = 1 if np.max(X) > 2 else 255 
        if X.ndim > 4:
            d = len(str(X.shape[0]*X.shape[3]))
            for i in tqdm(range(X.shape[0])):
                for j in range(X.shape[3]):
                    im = Image.fromarray(X[i,:,:,j,0]*v)
                    im = im.convert('L')
                    im.save(os.path.join(data_dir, p_x + str(i).zfill(d) + "_" \
                                         + str(j).zfill(d) + ".png"))
        else:
            d = len(str(X.shape[0]))
            for i in tqdm(range(X.shape[0])):
                im = Image.fromarray(X[i,:,:,0]*v)         
                im = im.convert('L')                                                
                im.save(os.path.join(data_dir, p_x + str(i).zfill(d) + ".png")) 

    if Y is not None:
        if mask_dir is not None:                                                    
            os.makedirs(mask_dir, exist_ok=True)
        else:
            print("Not mask_dir provided so no image will be saved!")
            return
        
        print("Saving images in {}".format(mask_dir))

        v = 1 if np.max(Y) > 2 else 255
        if Y.ndim > 4:
            d = len(str(Y.shape[0]*Y.shape[3]))
            for i in tqdm(range(Y.shape[0])):
                for j in range(Y.shape[3]):
                    im = Image.fromarray(Y[i,:,:,j,0]*v)
                    im = im.convert('L')
                    im.save(os.path.join(mask_dir, p_y + str(i).zfill(d) + "_" \
                                         + str(j).zfill(d) + ".png"))
        else:
            d = len(str(Y.shape[0]))
            for i in tqdm(range(0, Y.shape[0])):
                im = Image.fromarray(Y[i,:,:,0]*v)         
                im = im.convert('L')                                                
                im.save(os.path.join(mask_dir, p_y + str(i).zfill(d) + ".png")) 
       

def make_weight_map(label, binary = True, w0 = 10, sigma = 5):
    """Generates a weight map in order to make the U-Net learn better the          
       borders of cells and distinguish individual cells that are tightly packed.  
       These weight maps follow the methodology of the original U-Net paper.

       Based on `unet/py_files/helpers.py <https://github.com/deepimagej/python4deepimagej/blob/499955a264e1b66c4ed2c014cb139289be0e98a4/unet/py_files/helpers.py>`_.
   
       Parameters
       ----------
   
       label : 3D numpy array
          Corresponds to a label image. E.g. ``(x, y, channels)``.
   
       binary : bool, optional
          Corresponds to whether or not the labels are binary.                                                             
                                                                                   
       w0 : float, optional
          Controls for the importance of separating tightly associated entities.                                                
                                                                                   
       sigma : int, optional
          Represents the standard deviation of the Gaussian used for the weight map.
   
       Example
       -------
   
       Notice that weight has been defined where the objects are almost touching 
       each other.
   
       .. image:: img/weight_map.png
           :width: 650
           :align: center
    """
    
    # Initialization.
    lab = np.array(label)
    lab_multi = lab
       
    if len(lab.shape) == 3:
        lab = lab[:, :, 0]

    # Get shape of label.
    rows, cols = lab.shape
    
    if binary:
        
        # Converts the label into a binary image with background = 0
        # and cells = 1.
        lab[lab == 255] = 1
        
        # Builds w_c which is the class balancing map. In our case, we want 
        # cells to have weight 2 as they are more important than background 
        # which is assigned weight 1.
        w_c = np.array(lab, dtype=float)
        w_c[w_c == 1] = 1
        w_c[w_c == 0] = 0.5
    
        # Converts the labels to have one class per object (cell).
        lab_multi = measure.label(lab, neighbors = 8, background = 0)
    else:
        
        # Converts the label into a binary image with background = 0.
        # and cells = 1.
        lab[lab > 0] = 1
        
        # Builds w_c which is the class balancing map. In our case, we want 
        # cells to have weight 2 as they are more important than background 
        # which is assigned weight 1.
        w_c = np.array(lab, dtype=float)
        w_c[w_c == 1] = 1
        w_c[w_c == 0] = 0.5
    components = np.unique(lab_multi)
    
    n_comp = len(components)-1
    
    maps = np.zeros((n_comp, rows, cols))
    
    map_weight = np.zeros((rows, cols))
    
    if n_comp >= 2:
        for i in range(n_comp):
            
            # Only keeps current object.
            tmp = (lab_multi == components[i+1])
            
            # Invert tmp so that it can have the correct distance.
            # transform
            tmp = ~tmp
            
            # For each pixel, computes the distance transform to
            # each object.
            maps[i][:][:] = scipy.ndimage.distance_transform_edt(tmp)
    
        maps = np.sort(maps, axis=0)
        
        # Get distance to the closest object (d1) and the distance to the second
        # object (d2).
        d1 = maps[0][:][:]
        d2 = maps[1][:][:]
        
        map_weight = w0*np.exp(-((d1+d2)**2)/(2*(sigma**2)) ) * (lab==0).astype(int);

    map_weight += w_c
    
    return map_weight


def do_save_wm(labels, path, binary = True, w0 = 10, sigma = 5):
    """Retrieves the label images, applies the weight-map algorithm and save the
       weight maps in a folder. Uses internally :meth:`util.make_weight_map`.
   
       Based on `deepimagejunet/py_files/helpers.py <https://github.com/deepimagej/python4deepimagej/blob/499955a264e1b66c4ed2c014cb139289be0e98a4/unet/py_files/helpers.py>`_.
   
       Parameters
       ----------
       labels : 4D numpy array
           Corresponds to given label images. E.g. ``(image_number, x, y, channels)``.
       
       path : str
           Refers to the path where the weight maps should be saved.
       
       binary : bool, optional
           Corresponds to whether or not the labels are binary. 
       
       w0 : float, optional
           Controls for the importance of separating tightly associated entities. 
           
       sigma : int, optional
           Represents the standard deviation of the Gaussian used for the weight 
           map.
    """
    
    # Copy labels.
    labels_ = copy.deepcopy(labels)
    
    # Perform weight maps.
    for i in range(len(labels_)):
        labels_[i] = make_weight_map(labels[i].copy(), binary, w0, sigma)
    
    maps = np.array(labels_)
    
    n, rows, cols = maps.shape
    
    # Resize correctly the maps so that it can be used in the model.
    maps = maps.reshape((n, rows, cols, 1))
    
    # Count number of digits in n. This is important for the number
    # of leading zeros in the name of the maps.
    n_digits = len(str(n))
    
    # Save path with correct leading zeros.
    path_to_save = path + "weight/{b:0" + str(n_digits) + "d}.npy"
    
    # Saving files as .npy files.
    for i in range(len(labels_)):
        np.save(path_to_save.format(b=i), labels_[i])
        
    return None


def foreground_percentage(mask, class_tag):
    """Percentage of pixels that corresponds to the class in the given image.

       Parameters
       ----------
       mask : 2D Numpy array
           Image mask to analize.

       class_tag : int
           Class to find in the image.

       Returns
       -------
       x : float
           Percentage of pixels that corresponds to the class. Value between ``0`` 
           and ``1``.
    """

    c = 0
    for i in range(0, mask.shape[0]):
        for j in range(0, mask.shape[1]):
            if mask[i, j, 0] == class_tag:
                c = c + 1

    return c/(mask.shape[0]*mask.shape[1])


def divide_images_on_classes(data, data_mask, out_dir, num_classes=2, th=0.8):
    """Create a folder for each class where the images that have more pixels 
       labeled as the class (in percentage) than the given threshold will be 
       stored. 
    
       Parameters
       ----------
       data : 4D numpy array
           Data to save as images. The first dimension must be the number of 
           images. ``E.g. (image_number, x, y, channels)``.

       data_mask : 4D numpy array
           Data mask to save as images.  The first dimension must be the number 
           of images. ``E.g. (image_number, x, y, channels)``.

       out_dir : str
           Path to save the images.

       num_classes : int, optional
           Number of classes. 

       th : float, optional
           Percentage of the pixels that must be labeled as a class to save it
           inside that class folder. 
    """
        
    # Create the directories
    for i in range(num_classes):
        os.makedirs(os.path.join(out_dir, "x", "class"+str(i)), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "y", "class"+str(i)), exist_ok=True)

    print("Dividing provided data into {} classes . . .".format(num_classes))
    d = len(str(data.shape[0]))
    for i in tqdm(range(data.shape[0])):
        # Assign the image to a class if it has, in percentage, more pixels of 
        # that class than the given threshold 
        for j in range(num_classes):
            t = foreground_percentage(data_mask[i], j)
            if t > th:
                im = Image.fromarray(data[i,:,:,0])
                im = im.convert('L')
                im.save(os.path.join(os.path.join(out_dir, "x", "class"+str(j)), 
                        "im_" + str(i).zfill(d) + ".png"))
                im = Image.fromarray(data_mask[i,:,:,0]*255)
                im = im.convert('L')
                im.save(os.path.join(os.path.join(out_dir, "y", "class"+str(j)), 
                        "mask_" + str(i).zfill(d) + ".png"))


def save_filters_of_convlayer(model, out_dir, l_num=None, name=None, prefix="",
                              img_per_row=8):
    """Create an image of the filters learned by a convolutional layer. One can 
       identify the layer with ``l_num`` or ``name`` args. If both are passed 
       ``name`` will be prioritized. 
    
       Inspired by https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks

       Parameters
       ----------
       model : Keras Model
           Model where the layers are stored.

       out_dir : str
           Path where the image will be stored.
        
       l_num : int, optional
           Number of the layer to extract filters from. 
            
       name : str, optional
           Name of the layer to extract filters from.

       prefix : str, optional
           Prefix to add to the output image name. 
        
       img_per_row : int, optional
           Filters per row on the image.

       Raises   
       ------
       ValueError
           if ``l_num`` and ``name`` not provided.     

       Examples
       --------
       To save the filters learned by the layer called ``conv1`` one can call 
       the function as follows ::

           save_filters_of_convlayer(model, char_dir, name="conv1", prefix="model")

       That will save in ``out_dir`` an image like this:

       .. image:: img/save_filters.png 
           :width: 60%
           :align: center 
    """

    if l_num is None and name is None:
        raise ValueError("One between 'l_num' or 'name' must be provided")
    
    # For matplotlib errors in display
    os.environ['QT_QPA_PLATFORM']='offscreen'

    # Find layer number of the layer named by 'name' variable
    if name is not None:
        pos = 0
        for layer in model.layers:
            if name == layer.name:
                break
            pos += 1
        l_num = pos

    filters, biases = model.layers[l_num].get_weights()
    
    # normalize filter values to 0-1 so we can visualize them
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    
    rows = int(math.floor(filters.shape[3]/img_per_row))
    i = 0
    for r in range(rows):
        for c in range(img_per_row):
            ax = plt.subplot(rows, img_per_row, i+1)
            ax.set_xticks([])
            ax.set_yticks([])
            f = filters[:,:,0,i]
            plt.imshow(filters[:,:,0,i], cmap='gray')

            i += 1

    prefix += "_" if prefix != "" else prefix
    plt.savefig(os.path.join(out_dir, prefix + 'f_layer' + str(l_num) + '.png'))
    plt.clf()


def calculate_2D_volume_prob_map(Y, w_foreground=0.94, w_background=0.06,
                                 save_file=None):
    """Calculate the probability map of the given 2D data.

       Parameters
       ----------
       Y : 4D Numpy array
           Data to calculate the probability map from. E. g. ``(image_number, x, 
           y, channel)``

       w_foreground : float, optional
           Weight of the foreground. This value plus ``w_background`` must be 
           equal ``1``.

       w_background : float, optional
           Weight of the background. This value plus ``w_foreground`` must be 
           equal ``1``.

       save_file : str, optional
           Path to the file where the probability map will be stored.

       Raises
       ------
       ValueError
           if ``Y`` does not have 4 dimensions.                     

       ValueError
           if ``w_foreground + w_background > 1``.

       Returns
       -------
       Array : 4D Numpy array
           Probability map of the given data.
    """

    if Y.ndim != 4:
        raise ValueError("'Y' must be a 4D Numpy array")

    if w_foreground + w_background > 1:
        raise ValueError("'w_foreground' plus 'w_background' can not be greater "
                         "than one")

    prob_map = np.copy(Y[...,0])

    print("Constructing the probability map . . .")
    for i in tqdm(range(prob_map.shape[0])):
        pdf = prob_map[i]
    
        # Remove artifacts connected to image border
        pdf = clear_border(pdf)

        foreground_pixels = (pdf == 255).sum()
        background_pixels = (pdf == 0).sum()

        pdf[np.where(pdf == 255)] = w_foreground/foreground_pixels
        pdf[np.where(pdf == 0)] = w_background/background_pixels
        pdf /= pdf.sum() # Necessary to get all probs sum 1
        prob_map[i] = pdf

    if save_file is not None:
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        print("Saving the probability map in {}".format(save_file))
        np.save(save_file, prob_map)

    return prob_map


def calculate_3D_volume_prob_map(Y, w_foreground=0.94, w_background=0.06,
                                 save_file=None):
    """Calculate the probability map of the given 3D data.
       
       Parameters
       ---------- 
       Y : 5D Numpy array
           Data to calculate the probability map from. E. g. ``(num_subvolumes,
           x, y, z, channel)``

       w_foreground : float, optional
           Weight of the foreground. This value plus ``w_background`` must be 
           equal ``1``.

       w_background : float, optional
           Weight of the background. This value plus ``w_foreground`` must be 
           equal ``1``.

       save_file : str, optional
           Path to the file where the probability map will be stored.

       Returns
       -------
       Array : 5D Numpy array
           Probability map of the given data.

       Raises
       ------
       ValueError
           if ``Y`` does not have 5 dimensions.
       ValueError
           if ``w_foreground + w_background > 1``.
    """

    if Y.ndim != 5:
        raise ValueError("'Y' must be a 5D Numpy array")

    if w_foreground + w_background > 1:
        raise ValueError("'w_foreground' plus 'w_background' can not be greater "
                         "than one")

    prob_map = np.copy(Y[..., 0])
    
    print("Constructing the probability map . . .")
    for i in range(prob_map.shape[0]):
        for j in tqdm(range(prob_map[i].shape[2])):
            # Remove artifacts connected to image border
            prob_map[i,:,:,j] = clear_border(prob_map[i,:,:,j])

        foreground_pixels = (prob_map[i] == 255).sum()
        background_pixels = (prob_map[i] == 0).sum()

        prob_map[i][np.where(prob_map[i] == 255)] = w_foreground/foreground_pixels
        prob_map[i][np.where(prob_map[i] == 0)] = w_background/background_pixels
        prob_map[i] /= prob_map[i].sum() # Necessary to get all probs sum 1

    if save_file is not None:
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        print("Saving the probability map in {}".format(save_file))
        np.save(save_file, prob_map)

    return np.expand_dims(prob_map, -1)


def grayscale_2D_image_to_3D(X, Y, th=127):
    """Creates 3D surface from each image in X based on the grayscale of each
       image.
    
       Parameters
       ----------
       X : 4D numpy array
           Data that contains the images to create the surfaces from. E.g. 
           ``(image_number, x, y, channels)``.

       Y : 4D numpy array
           Data mask of the same shape of X that will be converted into 3D volume,
           stacking multiple times each image. Useful if you need the two data
           arrays to be of the same shape. E.g. ``(image_number, x, y, channels)``.

       th : int, optional
           Values to ommit when creating the surfaces. Useful to reduce the
           amount of data in z to be created and reduce computational time.

       Returns
       -------
       Array : 5D numpy array
           3D surface of each image provided. E.g. ``(image_number, z, x, y, 
           channels)``.

       Array : 5D numpy array
           3D stack of each mask provided. E.g. ``(image_number, z, x, y, 
           channels)``.
    """

    print("Creating 3D surface for each image . . .")

    _th = 255 - th
    X_3D = np.zeros((X.shape[0], X.shape[1], X.shape[2], _th, X.shape[3]), 
                    dtype=np.int32) 
    Y_3D = np.zeros((Y.shape[0], Y.shape[1], Y.shape[2], _th, Y.shape[3]),
                    dtype=np.int32)

    for i in tqdm(range(X.shape[0])): 
        for x in range(X.shape[1]):
            for y in range(X.shape[2]):
                pos = int(X[i, x, y, 0])-_th if int(X[i, x, y, 0]) >_th else 0
                X_3D[i, x, y, 0:pos, 0] = 1
                pos = int(Y[i, x, y, 0:pos, 0])*255
                Y_3D[i, x, y, 0:pos, 0] = 1

    print("*** New surface 3D data shape is now: {}".format(X_3D.shape))

    return X_3D, Y_3D
