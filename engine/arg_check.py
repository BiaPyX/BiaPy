
def arg_check(cfg):
    
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
        
    assert cfg.PROBLEM.NDIM in ['2D', '3D']
    assert cfg.PROBLEM.TYPE in ['SEMANTIC_SEG', 'INSTANCE_SEG', 'CLASSIFICATION', 'DETECTION', 'DENOISING', 'SUPER_RESOLUTION', 'SELF_SUPERVISED']

    if not cfg.TEST.STATS.PER_PATCH and not cfg.TEST.STATS.FULL_IMG:
        raise ValueError("One between 'TEST.STATS.PER_PATCH' or 'TEST.STATS.FULL_IMG' need to be True")

    if cfg.PROBLEM.NDIM == '3D' and not cfg.TEST.STATS.PER_PATCH and not cfg.TEST.STATS.MERGE_PATCHES:
        raise ValueError("One between 'TEST.STATS.PER_PATCH' or 'TEST.STATS.MERGE_PATCHES' need to be True when 'PROBLEM.NDIM'=='3D'")

    if cfg.TEST.MAP and not os.path.isdir(cfg.PATHS.MAP_CODE_DIR):
        raise ValueError("mAP calculation code not found. Please set 'PATHS.MAP_CODE_DIR' variable with the path of the "
                         "Github repo 'mAP_3Dvolume': 0) git clone https://github.com/danifranco/mAP_3Dvolume.git ; "
                         "1) git checkout grand-challenge ")

    if cfg.PROBLEM.NDIM == '3D' and cfg.TEST.STATS.FULL_IMG:
        print("WARNING: TEST.STATS.FULL_IMG == True while using PROBLEM.NDIM == '3D'. As 3D images are usually 'huge'"
              ", full image statistics will be disabled to avoid GPU memory overflow")

    if cfg.PROBLEM.TYPE == 'INSTANCE_SEG':
        if cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS == "B":
            raise ValueError("PROBLEM.INSTANCE_SEG.DATA_CHANNELS must be 'BC' or 'BCD' when PROBLEM.TYPE == 'INSTANCE_SEG'")
        else:
            if cfg.MODEL.N_CLASSES > 1:
                raise ValueError("Not implemented pipeline option")
    elif cfg.PROBLEM.TYPE in ['SUPER_RESOLUTION']:
        if not cfg.DATA.EXTRACT_RANDOM_PATCH:
            raise ValueError("'DATA.EXTRACT_RANDOM_PATCH' need to be True for 'SUPER_RESOLUTION'")
        if cfg.AUGMENTOR.RANDOM_CROP_SCALE == 1:
            raise ValueError("Resolution scale must be provided with 'AUGMENTOR.RANDOM_CROP_SCALE' variable")
        if cfg.PROBLEM.NDIM == '3D':
            raise NotImplementedError
    elif cfg.PROBLEM.TYPE in ['DENOISING']:
        if cfg.DATA.TEST.IN_MEMORY:
            raise ValueError("DATA.TEST.IN_MEMORY==True not supported in DENOISING. Please change it to False")
        if cfg.DATA.TEST.LOAD_GT:
            raise NotImplementedError
    if cfg.DATA.VAL.FROM_TRAIN and not cfg.DATA.VAL.CROSS_VAL and cfg.DATA.VAL.SPLIT_TRAIN <= 0:
        raise ValueError("'DATA.VAL.SPLIT_TRAIN' needs to be > 0 when 'DATA.VAL.FROM_TRAIN' == True")
    if cfg.DATA.VAL.FROM_TRAIN and not cfg.DATA.TRAIN.IN_MEMORY:
        raise ValueError("Validation can be extracted from train while 'DATA.TRAIN.IN_MEMORY' == False. Please set"
                         "'DATA.VAL.FROM_TRAIN' to False")
                         

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