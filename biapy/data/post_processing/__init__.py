import numpy as np

from biapy.engine.metrics import jaccard_index_numpy, voc_calculation
from biapy.data.post_processing.post_processing import calculate_zy_filtering, calculate_z_filtering


def apply_post_processing(cfg, data, Y=None):
    """Create training and validation generators.                                                                       
                                                                                                                        
       Parameters                                                                                                       
       ----------                                                                                                       
       cfg : YACS CN object                                                                               
           Configuration.                                                                                         
                                                                                                                        
       data : 4D Numpy array                                                                                               
           Data to apply post_proccessing. E.g. ``(num_of_images, y, x, channels)``.                                                              
                                                                                                                        
       Y : 4D Numpy array, optional                                                                                               
           Data GT to calculate the metrics. E.g. ``(num_of_images, y, x, channels)``. 
                                                                                                                        
       Returns                                                                                                          
       -------                                                                                                          
       iou_post : float
           Foreground IoU of ``data`` compared with ``Y`` after post-processing.

       ov_iou_post : float                                                                                                 
           Overall IoU of ``data`` compared with ``Y`` after post-processing.                                                   
    """  

    print("Applying post-processing . . .")

    if cfg.TEST.POST_PROCESSING.YZ_FILTERING:
        data = calculate_zy_filtering(data, cfg.TEST.POST_PROCESSING.YZ_FILTERING_SIZE)

    if cfg.TEST.POST_PROCESSING.Z_FILTERING:
        data = calculate_z_filtering(data, cfg.TEST.POST_PROCESSING.Z_FILTERING_SIZE)

    if Y is not None:
        iou_post = jaccard_index_numpy((Y>0.5).astype(np.uint8), (data>0.5).astype(np.uint8))
        ov_iou_post = voc_calculation((Y>0.5).astype(np.uint8), (data>0.5).astype(np.uint8), iou_post)
    else:
        iou_post, ov_iou_post = 0, 0 

    return data, iou_post, ov_iou_post