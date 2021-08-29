import os
import numpy as np

from utils.util import load_data_from_dir, load_3d_images_from_dir, labels_into_bcd


def create_train_val_instance_channels(cfg):
    """Create training and validation new data with appropiate channels based on ``DATA.CHANNELS`` for instance
       segmentation.
                                                                                                                        
       Parameters                                                                                                       
       ----------                                                                                                       
       cfg : YACS CN object                                                                               
           Configuration.                                                                                         
                                                                                                                        
       Returns                                                                                                          
       -------                                                                                                          
       train_filenames: List of str
           Training image paths.
    """                  

    f_name = load_data_from_dir if cfg.PROBLEM.NDIM == '2D' else load_3d_images_from_dir                        
    X_train, _, _, train_filenames = f_name(cfg.DATA.TRAIN.PATH, crop=True, crop_shape=cfg.DATA.PATCH_SIZE,
                                            overlap=cfg.DATA.TRAIN.OVERLAP, padding=cfg.DATA.TRAIN.PADDING,
                                            return_filenames=True)                    
    os.makedirs(cfg.DATA.TRAIN.INSTANCE_CHANNELS_DIR, exist_ok=True)                                            
    np.save(cfg.DATA.TRAIN.INSTANCE_CHANNELS_FILE, X_train)                                                     

    Y_train, _, _ = f_name(cfg.DATA.TRAIN.MASK_PATH, crop=True, crop_shape=cfg.DATA.PATCH_SIZE,
                           overlap=cfg.DATA.TRAIN.OVERLAP, padding=cfg.DATA.TRAIN.PADDING)                                                            
    Y_train = labels_into_bcd(Y_train, mode=cfg.DATA.CHANNELS, save_dir=cfg.PATHS.TRAIN_INSTANCE_CHANNELS_CHECK,
                              fb_mode=cfg.DATA.CONTOUR_MODE)                                                    
    os.makedirs(cfg.DATA.TRAIN.INSTANCE_CHANNELS_MASK_DIR, exist_ok=True)                                       
    np.save(cfg.DATA.TRAIN.INSTANCE_CHANNELS_MASK_FILE, Y_train)                                                
                                                                                                                
    if not cfg.DATA.VAL.FROM_TRAIN:
        X_val, _, _ = f_name(cfg.DATA.VAL.PATH, crop=True, crop_shape=cfg.DATA.PATCH_SIZE,
                             overlap=cfg.DATA.VAL.OVERLAP, padding=cfg.DATA.VAL.PADDING)
        os.makedirs(cfg.DATA.VAL.INSTANCE_CHANNELS_DIR, exist_ok=True)                                          
        np.save(cfg.DATA.VAL.INSTANCE_CHANNELS_FILE, X_val)                                                     

        Y_val, _, _ = f_name(cfg.DATA.VAL.MASK_PATH, crop=True, crop_shape=cfg.DATA.PATCH_SIZE,
                             overlap=cfg.DATA.VAL.OVERLAP, padding=cfg.DATA.VAL.PADDING)
        Y_val = labels_into_bcd(Y_val, mode=cfg.DATA.CHANNELS, save_dir=cfg.PATHS.VAL_INSTANCE_CHANNELS_CHECK,  
                                fb_mode=cfg.DATA.CONTOUR_MODE)                                                  
        os.makedirs(cfg.DATA.VAL.INSTANCE_CHANNELS_MASK_DIR, exist_ok=True)                                     
        np.save(cfg.DATA.VAL.INSTANCE_CHANNELS_MASK_FILE, Y_val)                                                
    return train_filenames


def create_test_instance_channels(cfg):
    """Create test new data with appropiate channels based on ``DATA.CHANNELS`` for instance segmentation.
                                                                                                                        
       Parameters                                                                                                       
       ----------                                                                                                       
       cfg : YACS CN object                                                                               
           Configuration.                                                                                         
                                                                                                                        
       Returns                                                                                                          
       -------                                                                                                          
       test_filenames: List of str                                                                                     
           Test image paths.                                                                                        
    """  

    f_name = load_data_from_dir if cfg.PROBLEM.NDIM == '2D' else load_3d_images_from_dir                        
    X_test, _, _, test_filenames = f_name(cfg.DATA.TEST.PATH, crop=True, crop_shape=cfg.DATA.PATCH_SIZE,
                                          overlap=cfg.DATA.TEST.OVERLAP, padding=cfg.DATA.TEST.PADDING, 
                                          return_filenames=True)                      
    os.makedirs(cfg.DATA.TEST.INSTANCE_CHANNELS_DIR, exist_ok=True)                                            
    np.save(cfg.DATA.TEST.INSTANCE_CHANNELS_FILE, X_test)                                                      
    
    if cfg.DATA.TEST.LOAD_GT:                                                                                   
        Y_test, _, _ = f_name(cfg.DATA.TEST.MASK_PATH)                                                          
        Y_test = labels_into_bcd(Y_test, mode=cfg.DATA.CHANNELS, save_dir=cfg.PATHS.TEST_INSTANCE_CHANNELS_CHECK,
                                 fb_mode=cfg.DATA.CONTOUR_MODE)                                                
        os.makedirs(cfg.DATA.TEST.INSTANCE_CHANNELS_MASK_DIR, exist_ok=True)                                    
        np.save(cfg.DATA.TEST.INSTANCE_CHANNELS_MASK_FILE, Y_test)                                              

    return test_filenames
