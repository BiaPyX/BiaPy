import os
import math
import mkl
import numpy as np
from tensorflow import set_random_seed
import time
import keras
import random
import matplotlib.pyplot as plt

def Print(s):
    """ Just a print """
    print("\n" + s, flush=True)

def limit_threads(threads_number='1'):
    """Limit the number of threads for a python process.
       
       Args: 
            threads_number (int, optional): number of threads.
    """

    Print("Python process limited to " + threads_number + " thread")

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


def create_plots(results, job_id, chartOutDir):
    """Create loss and jaccard_index plots with the given matrix results

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

    # Jaccard index
    plt.plot(results.history['jaccard_index'])
    plt.plot(results.history['val_jaccard_index'])
    plt.title('Model JOBID=' + job_id + ' jaccard_index')
    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.legend(['Train jaccard_index', 'Val. jaccard_index'], 
               loc='upper left')
    plt.savefig(os.path.join(chartOutDir , str(job_id) 
                + '_jaccard_index.png'))
    plt.clf()


def store_history(results, test_score, voc, det, time_callback, log_dir, 
                  job_file, metric='jaccard_index', smooth_score=0, smooth_voc=0,   
                  smooth_det=0):
    """Stores the results obtained as csv to manipulate them later 
       and labeled in another file as historic results.

       Args:
            results (history object): record of training loss values
            and metrics values at successive epochs.
            test_score (array of 2 int): loss and jaccard_index obtained
            with the test data.
            voc (float): VOC score value.
            det (float): DET score value.
            time_callback: time structure with the time of each epoch.
            csv_file (str): path where the csv file will be stored.
            history_file (str): path where the historic results will be
            stored.
            metric (str, optional): metric used (e.g. jaccard_index).
            smooth_score (float, optional): main metric obtained with smooth
            results.
            smooth_voc (float, optional): VOC metric obtained with smooth
            results.
            smooth_det (float, optional): DET metric obtained with smooth
            results.
    """

    # Create folders and construct file names
    csv_file = os.path.join(log_dir, 'formatted', job_file)          
    history_file = os.path.join(log_dir, 'history_of_values', job_file)
    if not os.path.exists(os.path.join(log_dir, 'formatted')):
        os.makedirs(os.path.join(log_dir, 'formatted'))
    if not os.path.exists(os.path.join(log_dir, 'history_of_values')):
        os.makedirs(os.path.join(log_dir, 'history_of_values'))
    
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
            + str(smooth_score) + ','
            + str(voc) + ','
            + str(smooth_voc) + ','
            + str(det) + ','
            + str(smooth_det) + ','
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
    f.write('############## TEST ' + metric.upper()
            + ' SMOOTH ############## \n')
    f.write(str(smooth_score) + '\n')
    f.write('############## VOC ############## \n')
    f.write(str(voc) + '\n')
    f.write('############## VOC SMOOTH ############## \n')
    f.write(str(smooth_voc) + '\n')
    f.write('############## DET ############## \n')
    f.write(str(det) + '\n')
    f.write('############## DET SMOOTH ############## \n')
    f.write(str(smooth_det) + '\n')
    f.close()


def threshold_plots(preds_test, Y_test, o_test_shape, j_score, det_eval_ge_path,
                    det_eval_path, det_bin, n_dig, job_id, job_file, char_dir):
    """Create a plot with the different metric values binarizing the prediction
       with different thresholds, from 0.1 to 0.9.
                                                                                
       Args:                                                                    
            preds_test (numpy array): predictions made by the model.
            Y_test (numpy array): ground truth of the data.
            o_test_shape (tuple): original shape of the data without crops, 
            necessary to reconstruct the images. 
            j_score (float): foreground jaccard score to calculate VOC.
            det_eval_ge_path (str): path where the ground truth is stored for 
            the DET calculation.
            det_eval_path (str): path where the evaluation of the metric will be done.
            det_bin (str): path to the DET binary.
            n_dig (int): The number of digits used for encoding temporal indices
            (e.g., 3). Used by the DET calculation binary.
            job_id (str): id of the job.
            job_file (str): id and run number of the job.
            char_dir (str): path to store the charts generated.

        Returns:
            t_jac[4] (float): value of the Jaccard index when the threshold 
            is 0.5.
            t_voc[4] (float): value of VOC when the threshold is 0.5.
            t_det[4] (float): value of DET when the threshold is 0.5.
    """

    from data import mix_data
    from metrics import jaccard_index, jaccard_index_numpy, voc_calculation, DET_calculation

    char_dir = os.path.join(char_dir, "t_" + job_file)

    t_jac = np.zeros(9)                                                         
    t_voc = np.zeros(9)                                                         
    t_det = np.zeros(9)                                                         
    objects = []                                                                
                                                                                
    for i, t in enumerate(np.arange(0.1,1.0,0.1)):                              
        objects.append(str(t))                                                  
                                                                                
        # Threshold images                                                      
        bin_preds_test = (preds_test > t).astype(np.uint8)                      
                                                                                
        # Reconstruct the data to the original shape and calculate Jaccard      
        h_num = int(o_test_shape[0] / bin_preds_test.shape[1]) \
                + (o_test_shape[0] % bin_preds_test.shape[1] > 0)        
        v_num = int(o_test_shape[1] / bin_preds_test.shape[2]) \
                + (o_test_shape[1] % bin_preds_test.shape[2] > 0)        
                                                                                
        # To calculate the Jaccard (binarized)                                  
        recons_preds_test = mix_data(bin_preds_test,                            
                                     math.ceil(bin_preds_test.shape[0]/(h_num*v_num)),
                                     out_shape=[h_num, v_num], grid=False)      
                                                                                
        # Metrics (Jaccard + VOC + DET)                                             
        Print("Calculate metrics . . .")                                        
        t_jac[i] = jaccard_index_numpy(Y_test, recons_preds_test)               
        t_voc[i] = voc_calculation(Y_test, recons_preds_test, j_score[1])         
        t_det[i] = DET_calculation(Y_test, recons_preds_test, det_eval_ge_path, 
                                   det_eval_path, det_bin, n_dig, job_id)       
                                                                                
        Print("t_jac[" + str(i) + "]: " + str(t_jac[i]))                        
        Print("t_voc[" + str(i) + "]: " + str(t_voc[i]))                        
        Print("t_det[" + str(i) + "]: " + str(t_det[i]))

    # For matplotlib errors in display                                          
    os.environ['QT_QPA_PLATFORM']='offscreen'                                   
    y_pos = np.arange(len(objects))                                             
    Print("t_jac.shape: " + str(t_jac.shape))                                   
    Print("y_pos.shape: " + str(y_pos.shape))                                   
   
                                                                             
    # Plot Jaccard values   
    fig, ax1 = plt.subplots()                                                    
    p1 = ax1.bar(y_pos, t_jac, align='center')                                   
    plt.title('Model JOBID=' + job_file + ' Jaccard', y=1.08)                 
    plt.ylabel('Value')                                                         
    plt.xlabel('Threshold')                                                     
    if not os.path.exists(char_dir):                                            
        os.makedirs(char_dir)                                                   
    for p in p1:                                                                
        ax1.text(p.get_x() + p.get_width()/2., 0.95*p.get_height(),              
        '%.3f' % float(p.get_height()), ha='center', va='bottom')               
    plt.savefig(os.path.join(char_dir, job_file + '_threshold_Jaccard.png'))    
    plt.clf()                                                                   
                                                                                
    # Plot VOC values                                                           
    fig, ax2 = plt.subplots()
    p2 = ax2.bar(y_pos, t_voc, align='center') 
    plt.title('Model JOBID=' + job_file + ' VOC', y=1.08)                               
    plt.ylabel('Value')                                                         
    plt.xlabel('Threshold')                                                     
    if not os.path.exists(char_dir):                                            
        os.makedirs(char_dir)                                                   
    for p in p2:                                                                
        ax2.text(p.get_x() + p.get_width()/2., 0.95*p.get_height(),              
        '%.3f' % float(p.get_height()), ha='center', va='bottom')
    plt.savefig(os.path.join(char_dir, job_file + '_threshold_VOC.png'))        
    plt.clf()                                                                   
                                                                                
    # Plot DET values                                                           
    fig, ax3 = plt.subplots()
    p3 = ax3.bar(y_pos, t_det, align='center')
    plt.title('Model JOBID=' + job_file + ' DET', y=1.08)                               
    plt.ylabel('Value')                                                         
    plt.xlabel('Threshold')                                                     
    if not os.path.exists(char_dir):                                            
        os.makedirs(char_dir)                                                   
    for p in p3:                                                                
        ax3.text(p.get_x() + p.get_width()/2., 0.95*p.get_height(),              
        '%.3f' % float(p.get_height()), ha='center', va='bottom')
    plt.savefig(os.path.join(char_dir, job_file + '_threshold_DET.png'))        
    plt.clf()

    return  t_jac[4], t_voc[4], t_det[4]
