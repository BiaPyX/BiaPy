import os
import numpy as np

from utils.util import load_data_from_dir, load_3d_images_from_dir, labels_into_bcd, save_npy_files


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
    X_train, _, _, train_filenames = f_name(cfg.DATA.TRAIN.PATH, return_filenames=True)
    if isinstance(X_train, list):
        for i in tqdm(range(len(X_train))):
            X_train[i] = X_train[i].transpose((0,3,1,2,4))
    else:
        X_train = X_train.transpose((0,3,1,2,4))
    save_npy_files(X_train, data_dir=cfg.DATA.TRAIN.INSTANCE_CHANNELS_DIR, filenames=train_filenames,
                   verbose=cfg.TEST.VERBOSE)

    Y_train, _, _ = f_name(cfg.DATA.TRAIN.MASK_PATH)
    if isinstance(Y_train, list):
        for i in tqdm(range(len(Y_train))):
            Y_train[i] = labels_into_bcd(Y_train[i], mode=cfg.DATA.CHANNELS, save_dir=cfg.PATHS.TEST_INSTANCE_CHANNELS_CHECK,
                                        fb_mode=cfg.DATA.CONTOUR_MODE)
            Y_train[i] = Y_train[i].transpose((0,3,1,2,4))
    else:
        Y_train = labels_into_bcd(Y_train, mode=cfg.DATA.CHANNELS, save_dir=cfg.PATHS.TRAIN_INSTANCE_CHANNELS_CHECK,
                              fb_mode=cfg.DATA.CONTOUR_MODE)
        Y_train = Y_train.transpose((0,3,1,2,4))
    save_npy_files(Y_train, data_dir=cfg.DATA.TRAIN.INSTANCE_CHANNELS_MASK_DIR, filenames=train_filenames,
                   verbose=cfg.TEST.VERBOSE)

    if not cfg.DATA.VAL.FROM_TRAIN:
        X_val, _, _, val_filenames = f_name(cfg.DATA.VAL.PATH, return_filenames=True)
        if isinstance(X_val, list):
            for i in tqdm(range(len(X_val))):
                X_val[i] = X_val[i].transpose((0,3,1,2,4))
        else:
            X_val = X_val.transpose((0,3,1,2,4))
        save_npy_files(X_val, data_dir=cfg.DATA.VAL.INSTANCE_CHANNELS_DIR, filenames=val_filenames,
                       verbose=cfg.TEST.VERBOSE)

        Y_val, _, _ = f_name(cfg.DATA.VAL.MASK_PATH)
        if isinstance(Y_val, list):
            for i in tqdm(range(len(Y_val))):
                Y_val[i] = labels_into_bcd(Y_val[i], mode=cfg.DATA.CHANNELS, save_dir=cfg.PATHS.VAL_INSTANCE_CHANNELS_CHECK,
                                           fb_mode=cfg.DATA.CONTOUR_MODE)
                Y_val[i] = Y_val[i].transpose((0,3,1,2,4))
        else:
            Y_val = Y_val.transpose((0,3,1,2,4))
        save_npy_files(Y_val, data_dir=cfg.DATA.VAL.INSTANCE_CHANNELS_MASK_DIR, filenames=val_filenames,
                       verbose=cfg.TEST.VERBOSE)
    return train_filenames


def create_test_instance_channels(cfg):
    """Create test new data with appropiate channels based on ``DATA.CHANNELS`` for instance segmentation.

       Parameters
       ----------
       cfg : YACS CN object
           Configuration.
    """

    f_name = load_data_from_dir if cfg.PROBLEM.NDIM == '2D' else load_3d_images_from_dir
    X_test, _, _, test_filenames = f_name(cfg.DATA.TEST.PATH, return_filenames=True)
    if isinstance(X_test, list):
        for i in tqdm(range(len(X_test))):
            X_test[i] = X_test[i].transpose((0,3,1,2,4))
    else:
        X_test = X_test.transpose((0,3,1,2,4))
    save_npy_files(X_test, data_dir=cfg.DATA.TEST.INSTANCE_CHANNELS_DIR, filenames=test_filenames,
                   verbose=cfg.TEST.VERBOSE)

    if cfg.DATA.TEST.LOAD_GT:
        Y_test, _, _ = f_name(cfg.DATA.TEST.MASK_PATH)
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
