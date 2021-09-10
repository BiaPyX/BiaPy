import os
import numpy as np
from tqdm import tqdm

from utils.util import load_data_from_dir, load_3d_images_from_dir, labels_into_bcd, save_npy_files


def create_instance_channels(cfg, data_type='train'):
    """Create training and validation new data with appropiate channels based on ``DATA.CHANNELS`` for instance
       segmentation.

       Parameters
       ----------
       cfg : YACS CN object
           Configuration.

	   data_type: str, optional
		   Wheter to create training or validation instance channels.

       Returns
       -------
       filenames: List of str
           Image paths.
    """

    assert data_type in ['train', 'val']

    f_name = load_data_from_dir if cfg.PROBLEM.NDIM == '2D' else load_3d_images_from_dir
    tag = "TRAIN" if data_type == "train" else "VAL"
    X, _, _, filenames = f_name(getattr(cfg.DATA, tag).PATH, return_filenames=True)
    print("Creating X_{} channels . . .".format(data_type))
    if isinstance(X, list):
        for i in tqdm(range(len(X))):
            X[i] = X[i].transpose((0,3,1,2,4))
    else:
        X = X.transpose((0,3,1,2,4))
    save_npy_files(X, data_dir=getattr(cfg.DATA, tag).INSTANCE_CHANNELS_DIR, filenames=filenames,
                   verbose=cfg.TEST.VERBOSE)

    Y, _, _ = f_name(getattr(cfg.DATA, tag).MASK_PATH)
    print("Creating Y_{} channels . . .".format(data_type))
    if isinstance(Y, list):
        for i in tqdm(range(len(Y))):
            Y[i] = labels_into_bcd(Y[i], mode=cfg.DATA.CHANNELS, save_dir=getattr(cfg.PATHS, tag+'_INSTANCE_CHANNELS_CHECK'),
                                   fb_mode=cfg.DATA.CONTOUR_MODE)
            Y[i] = Y[i].transpose((0,3,1,2,4))
    else:
        Y = labels_into_bcd(Y, mode=cfg.DATA.CHANNELS, save_dir=getattr(cfg.PATHS, tag+'_INSTANCE_CHANNELS_CHECK'),
                            fb_mode=cfg.DATA.CONTOUR_MODE)
        Y = Y.transpose((0,3,1,2,4))
    save_npy_files(Y, data_dir=getattr(cfg.DATA, tag).INSTANCE_CHANNELS_MASK_DIR, filenames=filenames,
                   verbose=cfg.TEST.VERBOSE)
    return filenames


def create_test_instance_channels(cfg):
    """Create test new data with appropiate channels based on ``DATA.CHANNELS`` for instance segmentation.

       Parameters
       ----------
       cfg : YACS CN object
           Configuration.
    """

    f_name = load_data_from_dir if cfg.PROBLEM.NDIM == '2D' else load_3d_images_from_dir
    X_test, _, _, test_filenames = f_name(cfg.DATA.TEST.PATH, return_filenames=True)
    print("Creating X_test channels . . .")
    if isinstance(X_test, list):
        for i in tqdm(range(len(X_test))):
            X_test[i] = X_test[i].transpose((0,3,1,2,4))
    else:
        X_test = X_test.transpose((0,3,1,2,4))
    save_npy_files(X_test, data_dir=cfg.DATA.TEST.INSTANCE_CHANNELS_DIR, filenames=test_filenames,
                   verbose=cfg.TEST.VERBOSE)

    if cfg.DATA.TEST.LOAD_GT and cfg.TEST.EVALUATE:
        Y_test, _, _ = f_name(cfg.DATA.TEST.MASK_PATH)
        print("Creating Y_test channels . . .")
        if isinstance(Y_test, list):
            for i in tqdm(range(len(Y_test))):
                Y_test[i] = labels_into_bcd(Y_test[i], mode=cfg.DATA.CHANNELS, save_dir=cfg.PATHS.TEST_INSTANCE_CHANNELS_CHECK,
                                            fb_mode=cfg.DATA.CONTOUR_MODE)
                Y_test[i] = Y_test[i].transpose((0,3,1,2,4))
        else:
            Y_test = labels_into_bcd(Y_test, mode=cfg.DATA.CHANNELS, save_dir=cfg.PATHS.TEST_INSTANCE_CHANNELS_CHECK,
                                     fb_mode=cfg.DATA.CONTOUR_MODE)
            Y_test = Y_test.transpose((0,3,1,2,4))
        save_npy_files(Y_test, data_dir=cfg.DATA.TEST.INSTANCE_CHANNELS_MASK_DIR, filenames=test_filenames,
                       verbose=cfg.TEST.VERBOSE)
