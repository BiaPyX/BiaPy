import numpy as np
from scipy.ndimage.filters import median_filter

from engine.metrics import jaccard_index_numpy, voc_calculation
from data.post_processing.post_processing import calculate_z_filtering


def apply_post_processing(cfg, data, Y=None):
    """Create training and validation generators.                                                                       
                                                                                                                        
       Parameters                                                                                                       
       ----------                                                                                                       
       cfg : config.config (Config class)                                                                               
           Configuration class.                                                                                         
                                                                                                                        
       data : 4D Numpy array                                                                                               
           Data to apply post_proccessing. E.g. ``(num_of_images, x, y, channels)``.                                                              
                                                                                                                        
       Y : 4D Numpy array, optional                                                                                               
           Data GT to calculate the metrics. E.g. ``(num_of_images, x, y, channels)``. 
                                                                                                                        
       Returns                                                                                                          
       -------                                                                                                          
       iou_post : float
           Foreground IoU of ``data`` compared with ``Y`` after post-processing.

       ov_iou_post : float                                                                                                 
           Overall IoU of ``data`` compared with ``Y`` after post-processing.                                                   
    """  

    print("Applying post-processing . . .")

    if cfg.TEST.POST_PROCESSING.BLENDING:
        ens_zfil_preds_test = calculate_z_filtering(data)

    if cfg.TEST.POST_PROCESSING.YZ_FILTERING:
        data = calculate_z_filtering(data, cfg.TEST.POST_PROCESSING.YZ_FILTERING_SIZE)

    if cfg.TEST.POST_PROCESSING.Z_FILTERING:
        data = median_filter(data, size=(cfg.TEST.POST_PROCESSING.Z_FILTERING_SIZE,1,1,1))

    if Y is not None:
        iou_post = jaccard_index_numpy((Y>0.5).astype(np.uint8), (data>0.5).astype(np.uint8))
        ov_iou_post = voc_calculation((Y>0.5).astype(np.uint8), (data>0.5).astype(np.uint8), iou_post)
    else:
        iou_post, ov_iou_post = 0, 0 

    return iou_post, ov_iou_post
