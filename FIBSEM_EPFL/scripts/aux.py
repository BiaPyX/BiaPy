import os
import mkl
import numpy as np
from keras import backend as K
import tensorflow as tf
from tensorflow import set_random_seed
import time
import keras
import random
import matplotlib.pyplot as plt

def limit_threads(threads_number='1'):
    """Limit the number of threads for a python process.
       
       Args: 
            threads_number (int, optional): number of threads.
    """

    print("Python process limited to " + threads_number + " thread")

    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ["MKL_DYNAMIC"]="FALSE";
    os.environ["NUMEXPR_NUM_THREADS"]='1';
    os.environ["VECLIB_MAXIMUM_THREADS"]='1';
    os.environ["OMP_NUM_THREADS"] = '1';
    mkl.set_num_threads(1)


def set_seed(seedValue=42):
    """Sets the seed on multiple python modules to obtain results as 
       reproducible as possible.

       Args:
           seedValue (int, optional): seed value.
    """
    random.seed = seedValue
    np.random.seed(seed=seedValue)
    set_random_seed(seedValue)
    os.environ["PYTHONHASHSEED"]=str(seedValue);


class TimeHistory(keras.callbacks.Callback):
    """Class to record each epoch time.
    """

    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def mean_iou(y_true, y_pred):
    """Define IoU metric 

       Args:
            y_true (array): ground truth masks
            y_pred (array): predicted masks
    """

    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


def create_plots(results, job_id, chartOutDir):
    """Create loss and mean_iou plots with the given matrix results

       Args:
            results (history object): record of training loss values 
            and metrics values at successive epochs.
            job_id (str): job number. 
            chartOutDir (str): path where the charts will be stored 
            into.
    """
    
    # For matplotlib errors in display
    os.environ['QT_QPA_PLATFORM']='offscreen'

    # Loss
    plt.plot(results.history['loss'])
    plt.plot(results.history['val_loss'])
    plt.title('Model JOBID=' + job_id + ' loss')
    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.legend(['Train loss', 'Val. loss'], loc='upper left')
    if not os.path.exists(chartOutDir):
        os.makedirs(chartOutDir)
    plt.savefig(os.path.join(chartOutDir , str(job_id) + '_loss.png'))
    plt.clf()

    # Mean_iou
    plt.plot(results.history['mean_iou'])
    plt.plot(results.history['val_mean_iou'])
    plt.title('Model JOBID=' + job_id + ' mean_iou')
    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.legend(['Train mean_iou', 'Val. mean_iou'], loc='upper left')
    plt.savefig(os.path.join(chartOutDir , str(job_id) 
                + '_meanIOU.png'))
    plt.clf()
    plt.clf()


def store_history(results, test_score, time_callback, csv_file,
                  history_file, metric='mean_iou'):
    """Stores the results obtained as csv to manipulate them later 
       and labeled in another file as historic results.

       Args:
            results (history object): record of training loss values
            and metrics values at successive epochs.
            test_score (array of 2 int): loss and mean_iou obtained with
            the test data.
            time_callback: time structure with the time of each epoch.
            csv_file (str): path where the csv file will be stored.
            history_file (str): path where the historic results will be
            stored.
            metric (str, optional): metric used (e.g. mean_iou).
    """
    # Store the results as csv
    try:
        os.remove(csv_file)
    except OSError:
        pass
    f = open(csv_file, 'x')
    f.write(str(np.min(results.history['loss'])) + ','
            + str(np.min(results.history['val_loss'])) + ','
            + str(test_score[0]) + ',' 
            + str(np.max(results.history[metric])) + ',' 
            + str(np.max(results.history['val_' + metric])) + ',' 
            + str(test_score[1]) + ',' 
            + str(len(results.history['val_loss'])) + ',' 
            + str(np.mean(time_callback.times)) + ','
            + str(np.sum(time_callback.times)) + '\n')
    f.close()

    # Save all the values in case we need them in the future
    try:
        os.remove(history_file)
    except OSError:
        pass
    f = open(history_file, 'x')
    f.write('############## TRAIN LOSS ############## \n')
    f.write(str(results.history['loss']) + '\n')
    f.write('############## VALIDATION LOSS ############## \n')
    f.write(str(results.history['val_loss']) + '\n')
    f.write('############## TEST LOSS ############## \n')
    f.write(str(test_score[0]) + '\n')
    f.write('############## TRAIN ' + metric.upper() 
            + ' ############## \n')
    f.write(str(results.history[metric]) + '\n')
    f.write('############## VALIDATION ' + metric.upper()
            + ' ############## \n')
    f.write(str(results.history[metric]) + '\n')
    f.write('############## TEST ' + metric.upper()
            + ' ############## \n')
    f.write(str(test_score[1]) + '\n')


