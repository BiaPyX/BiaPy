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
    Y, _, _, filenames = f_name(getattr(cfg.DATA, tag).MASK_PATH, return_filenames=True)
    print("Creating Y_{} channels . . .".format(data_type))
    if isinstance(Y, list):
        for i in tqdm(range(len(Y))):
            Y[i] = labels_into_bcd(Y[i], mode=cfg.DATA.CHANNELS, save_dir=getattr(cfg.PATHS, tag+'_INSTANCE_CHANNELS_CHECK'),
                          fb_mode=cfg.DATA.CONTOUR_MODE)
    else:
        Y = labels_into_bcd(Y, mode=cfg.DATA.CHANNELS, save_dir=getattr(cfg.PATHS, tag+'_INSTANCE_CHANNELS_CHECK'),
                   fb_mode=cfg.DATA.CONTOUR_MODE)
    save_npy_files(Y, data_dir=getattr(cfg.DATA, tag).INSTANCE_CHANNELS_MASK_DIR, filenames=filenames,
                   verbose=cfg.TEST.VERBOSE)
    X, _, _, filenames = f_name(getattr(cfg.DATA, tag).PATH, return_filenames=True)
    print("Creating X_{} channels . . .".format(data_type))
    save_npy_files(X, data_dir=getattr(cfg.DATA, tag).INSTANCE_CHANNELS_DIR, filenames=filenames,
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

    if cfg.DATA.TEST.LOAD_GT and cfg.TEST.EVALUATE:
        Y_test, _, _, test_filenames = f_name(cfg.DATA.TEST.MASK_PATH, return_filenames=True)
        print("Creating Y_test channels . . .")
        if isinstance(Y_test, list):
            for i in tqdm(range(len(Y_test))):
                Y_test[i] = labels_into_bcd(Y_test[i], mode=cfg.DATA.CHANNELS, save_dir=cfg.PATHS.TEST_INSTANCE_CHANNELS_CHECK,
                                            fb_mode=cfg.DATA.CONTOUR_MODE)
        else:
            Y_test = labels_into_bcd(Y_test, mode=cfg.DATA.CHANNELS, save_dir=cfg.PATHS.TEST_INSTANCE_CHANNELS_CHECK,
                                     fb_mode=cfg.DATA.CONTOUR_MODE)
        save_npy_files(Y_test, data_dir=cfg.DATA.TEST.INSTANCE_CHANNELS_MASK_DIR, filenames=test_filenames,
                       verbose=cfg.TEST.VERBOSE)

    print("Creating X_test channels . . .")
    X_test, _, _, test_filenames = f_name(cfg.DATA.TEST.PATH, return_filenames=True)
    save_npy_files(X_test, data_dir=cfg.DATA.TEST.INSTANCE_CHANNELS_DIR, filenames=test_filenames,
                   verbose=cfg.TEST.VERBOSE)


def data_checks(cfg):
    """Checks data variables so no error is thrown if the user forgets setting some variables. 
    """
    opts = []
    if cfg.PROBLEM.NDIM == '3D':
        if cfg.DATA.TRAIN.OVERLAP == (0,0):
            opts.extend(['DATA.TRAIN.OVERLAP', (0,0,0)])
        if cfg.DATA.TRAIN.PADDING == (0,0):
            opts.extend(['DATA.TRAIN.PADDING', (0,0,0)])
        if cfg.DATA.VAL.OVERLAP == (0,0):
            opts.extend(['DATA.VAL.OVERLAP', (0,0,0)])
        if cfg.DATA.VAL.PADDING == (0,0):
            opts.extend(['DATA.VAL.PADDING', (0,0,0)])
        if cfg.DATA.TEST.OVERLAP == (0,0):
            opts.extend(['DATA.TEST.OVERLAP', (0,0,0)])
        if cfg.DATA.TEST.PADDING == (0,0):
            opts.extend(['DATA.TEST.PADDING', (0,0,0)])
    if len(opts) > 0:
        cfg.merge_from_list(opts)

    count = 2 if cfg.PROBLEM.NDIM == '2D' else 3
    if len(cfg.DATA.TRAIN.OVERLAP) != count:
        raise ValueError("When PROBLEM.NDIM == {} DATA.TRAIN.OVERLAP tuple must be lenght {}, given {}."
                         .format(cfg.PROBLEM.NDIM, count, cfg.DATA.TRAIN.OVERLAP))
    if len(cfg.DATA.TRAIN.PADDING) != count:
        raise ValueError("When PROBLEM.NDIM == {} DATA.TRAIN.PADDING tuple must be lenght {}, given {}."
                         .format(cfg.PROBLEM.NDIM, count, cfg.DATA.TRAIN.PADDING))
    if len(cfg.DATA.TEST.OVERLAP) != count:
        raise ValueError("When PROBLEM.NDIM == {} DATA.TEST.OVERLAP tuple must be lenght {}, given {}."
                         .format(cfg.PROBLEM.NDIM, count, cfg.DATA.TEST.OVERLAP))
    if len(cfg.DATA.TEST.PADDING) != count:
        raise ValueError("When PROBLEM.NDIM == {} DATA.TEST.PADDING tuple must be lenght {}, given {}."
                         .format(cfg.PROBLEM.NDIM, count, cfg.DATA.TEST.PADDING))
    if len(cfg.DATA.PATCH_SIZE) != count+1:
        raise ValueError("When PROBLEM.NDIM == {} DATA.PATCH_SIZE tuple must be lenght {}, given {}."
                         .format(cfg.PROBLEM.NDIM, count+1, cfg.DATA.PATCH_SIZE))
    if len(cfg.DATA.TRAIN.RESOLUTION) != 1 and len(cfg.DATA.TRAIN.RESOLUTION) != count:
        raise ValueError("When PROBLEM.NDIM == {} DATA.TRAIN.RESOLUTION tuple must be lenght {}, given {}."
                         .format(cfg.PROBLEM.NDIM, count, cfg.DATA.TRAIN.RESOLUTION))
    if len(cfg.DATA.VAL.RESOLUTION) != 1 and len(cfg.DATA.VAL.RESOLUTION) != count:
        raise ValueError("When PROBLEM.NDIM == {} DATA.VAL.RESOLUTION tuple must be lenght {}, given {}."
                         .format(cfg.PROBLEM.NDIM, count, cfg.DATA.VAL.RESOLUTION))
    if len(cfg.DATA.TEST.RESOLUTION) != 1 and len(cfg.DATA.TEST.RESOLUTION) != count:
        raise ValueError("When PROBLEM.NDIM == {} DATA.TEST.RESOLUTION tuple must be lenght {}, given {}."
                         .format(cfg.PROBLEM.NDIM, count, cfg.DATA.TEST.RESOLUTION))
    
    if cfg.TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS:
        if len(cfg.DATA.TEST.RESOLUTION) == 1:
            raise ValueError("'DATA.TEST.RESOLUTION' must be set when using 'TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS'")
        if len(cfg.DATA.TEST.RESOLUTION) != count:
            raise ValueError("'DATA.TEST.RESOLUTION' must match in length to {}, which is the number of "
                             "dimensions".format(count))