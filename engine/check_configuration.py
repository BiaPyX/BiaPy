import os
import numpy as np

from utils.util import check_value

def check_configuration(cfg):
    """
    Check if the configuration is good. 
    """

    dim_count = 2 if cfg.PROBLEM.NDIM == '2D' else 3

    # Adjust overlap and padding in the default setting if it was not set
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

    # Adjust channel weights 
    if cfg.PROBLEM.TYPE == 'INSTANCE_SEG':
        if not 'Dv2' in cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS:
            channels_provided = len(cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS)
        else:
            channels_provided = len(cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS.replace('Dv2',''))+1
        if len(cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNEL_WEIGHTS) != channels_provided:
            if cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNEL_WEIGHTS == (1, 1):
                opts.extend(['PROBLEM.INSTANCE_SEG.DATA_CHANNEL_WEIGHTS', (1,)*channels_provided])    
        
    # Adjust dropout to feature maps
    if cfg.MODEL.ARCHITECTURE in ['ViT', 'unetr', 'tiramisu', 'mae']:
        if all(x == 0 for x in cfg.MODEL.DROPOUT_VALUES):
            opts.extend(['MODEL.DROPOUT_VALUES', (0.,)])
        elif len(cfg.MODEL.DROPOUT_VALUES) != 1:
            raise ValueError("'MODEL.DROPOUT_VALUES' must be list of an unique number when 'MODEL.ARCHITECTURE' is one among ['ViT', 'mae', 'unetr', 'tiramisu']")
        elif not check_value(cfg.MODEL.DROPOUT_VALUES[0]):
            raise ValueError("'MODEL.DROPOUT_VALUES' not in [0, 1] range")
    else:
        if len(cfg.MODEL.FEATURE_MAPS) != len(cfg.MODEL.DROPOUT_VALUES):
            if all(x == 0 for x in cfg.MODEL.DROPOUT_VALUES):
                opts.extend(['MODEL.DROPOUT_VALUES', (0.,)*len(cfg.MODEL.FEATURE_MAPS)])
            elif any(not check_value(x) for x in cfg.MODEL.DROPOUT_VALUES):
                raise ValueError("'MODEL.DROPOUT_VALUES' not in [0, 1] range")
            else:
                raise ValueError("'MODEL.FEATURE_MAPS' and 'MODEL.DROPOUT_VALUES' lengths must be equal")

    # Adjust Z_DOWN values to feature maps
    if len(cfg.MODEL.FEATURE_MAPS)-1 != len(cfg.MODEL.Z_DOWN):
        if all(x == 0 for x in cfg.MODEL.Z_DOWN):
            opts.extend(['MODEL.Z_DOWN', (2,)*(len(cfg.MODEL.FEATURE_MAPS)-1)])
        elif any([False for x in cfg.MODEL.Z_DOWN if x != 1 and x != 2]):
            raise ValueError("'MODEL.Z_DOWN' need to be 1 or 2")
        else:
            raise ValueError("'MODEL.FEATURE_MAPS' length minus one and 'MODEL.Z_DOWN' length must be equal")

    if cfg.DATA.TRAIN.MINIMUM_FOREGROUND_PER != -1:
        if not check_value(cfg.DATA.TRAIN.MINIMUM_FOREGROUND_PER):
            raise ValueError("DATA.TRAIN.MINIMUM_FOREGROUND_PER not in [0, 1] range")
        if cfg.PROBLEM.TYPE not in ['SEMANTIC_SEG', 'INSTANCE_SEG', 'DETECTION']:
            raise ValueError("'DATA.TRAIN.MINIMUM_FOREGROUND_PER' can only be set in 'SEMANTIC_SEG', 'INSTANCE_SEG' and 'DETECTION' workflows")

    if len(cfg.DATA.TRAIN.RESOLUTION) == 1 and cfg.DATA.TRAIN.RESOLUTION[0] == -1:
        opts.extend(['DATA.TRAIN.RESOLUTION', (1,)*dim_count])
    if len(cfg.DATA.VAL.RESOLUTION) == 1 and cfg.DATA.VAL.RESOLUTION[0] == -1:
        opts.extend(['DATA.VAL.RESOLUTION', (1,)*dim_count])
    if len(cfg.DATA.TEST.RESOLUTION) == 1 and cfg.DATA.TEST.RESOLUTION[0] == -1:
        opts.extend(['DATA.TEST.RESOLUTION', (1,)*dim_count])

    if cfg.TEST.POST_PROCESSING.DET_WATERSHED and cfg.PROBLEM.TYPE != 'DETECTION':
        raise ValueError("'TEST.POST_PROCESSING.DET_WATERSHED' can only be set when 'PROBLEM.TYPE' is 'DETECTION'")
    if cfg.TEST.POST_PROCESSING.DET_WATERSHED:
        for x in cfg.TEST.POST_PROCESSING.DET_WATERSHED_FIRST_DILATION:
            if not isinstance(x, list):
                raise ValueError("'TEST.POST_PROCESSING.DET_WATERSHED_FIRST_DILATION' need to be a list of list") 
            if any(y == -1 for y in x):
                raise ValueError("Please set 'TEST.POST_PROCESSING.DET_WATERSHED_FIRST_DILATION' when using 'TEST.POST_PROCESSING.DET_WATERSHED_FIRST_DILATION'")
            if len(x) != dim_count:
                raise ValueError("'TEST.POST_PROCESSING.DET_WATERSHED_FIRST_DILATION' need to be of dimension {} for {} problem".format(dim_count, cfg.PROBLEM.NDIM))
        if cfg.TEST.POST_PROCESSING.DET_WATERSHED_DONUTS_CLASSES != [-1]:
            if len(cfg.TEST.POST_PROCESSING.DET_WATERSHED_DONUTS_CLASSES) > cfg.MODEL.N_CLASSES:
                raise ValueError("'TEST.POST_PROCESSING.DET_WATERSHED_DONUTS_CLASSES' length can't be greater than 'MODEL.N_CLASSES'")
            if np.max(cfg.TEST.POST_PROCESSING.DET_WATERSHED_DONUTS_CLASSES) > cfg.MODEL.N_CLASSES:
                raise ValueError("'TEST.POST_PROCESSING.DET_WATERSHED_DONUTS_CLASSES' can not have a class number greater than 'MODEL.N_CLASSES'")
            min_class = np.min(cfg.TEST.POST_PROCESSING.DET_WATERSHED_DONUTS_CLASSES)
            if not all(cfg.TEST.POST_PROCESSING.DET_WATERSHED_DONUTS_CLASSES == np.array(range(min_class,len(cfg.TEST.POST_PROCESSING.DET_WATERSHED_DONUTS_CLASSES)+1))):
                raise ValueError("'TEST.POST_PROCESSING.DET_WATERSHED_DONUTS_CLASSES' must be consecutive, e.g [1,2,3,4..]") 
            if len(cfg.TEST.POST_PROCESSING.DET_WATERSHED_DONUTS_PATCH) != dim_count:
                raise ValueError("'TEST.POST_PROCESSING.DET_WATERSHED_DONUTS_PATCH' need to be of dimension {} for {} problem".format(dim_count, cfg.PROBLEM.NDIM))

    if cfg.TEST.POST_PROCESSING.WATERSHED_CIRCULARITY != -1:
        if not check_value(cfg.TEST.POST_PROCESSING.WATERSHED_CIRCULARITY):
            raise ValueError("'TEST.POST_PROCESSING.WATERSHED_CIRCULARITY' not in [0, 1] range")
    if cfg.PROBLEM.TYPE != 'INSTANCE_SEG':
        if cfg.TEST.POST_PROCESSING.WATERSHED_CIRCULARITY != -1 and cfg.PROBLEM.TYPE != 'DETECTION':
            raise ValueError("'TEST.POST_PROCESSING.WATERSHED_CIRCULARITY' can only be set in 'DETECTION' or 'INSTANCE_SEG' problems")    
        if cfg.TEST.POST_PROCESSING.VORONOI_ON_MASK:
            raise ValueError("'TEST.POST_PROCESSING.VORONOI_ON_MASK' can only be enabled in a 'INSTANCE_SEG' problem")
    if cfg.TEST.POST_PROCESSING.DET_WATERSHED and cfg.PROBLEM.TYPE != 'DETECTION':
        raise ValueError("'TEST.POST_PROCESSING.DET_WATERSHED' can only be set when 'PROBLEM.TYPE' is 'DETECTION'")

    if cfg.PROBLEM.NDIM == "2D":
        if (cfg.TEST.POST_PROCESSING.YZ_FILTERING or cfg.TEST.POST_PROCESSING.Z_FILTERING) \
            and not cfg.TEST.ANALIZE_2D_IMGS_AS_3D_STACK:
            raise ValueError("'TEST.POST_PROCESSING.YZ_FILTERING' and 'TEST.POST_PROCESSING.Z_FILTERING' is done only if"
                " 'TEST.ANALIZE_2D_IMGS_AS_3D_STACK' is enabled. Enable this last or disable those post-processing methods "
                "because it can not be applied to 2D images")
    if (cfg.TEST.POST_PROCESSING.YZ_FILTERING or cfg.TEST.POST_PROCESSING.Z_FILTERING) \
        and cfg.PROBLEM.TYPE not in ['SEMANTIC_SEG', 'INSTANCE_SEG', 'DETECTION']:
        raise ValueError("'TEST.POST_PROCESSING.YZ_FILTERING' or 'TEST.POST_PROCESSING.Z_FILTERING' can only be enabled "
            "when 'PROBLEM.TYPE' is among ['SEMANTIC_SEG', 'INSTANCE_SEG', 'DETECTION']")
 

    if len(opts) > 0:
        cfg.merge_from_list(opts)

    #### General checks ####
    assert cfg.PROBLEM.NDIM in ['2D', '3D'], "Problem need to be '2D' or '3D'"
    assert cfg.PROBLEM.TYPE in ['SEMANTIC_SEG', 'INSTANCE_SEG', 'CLASSIFICATION', 'DETECTION', 'DENOISING', 'SUPER_RESOLUTION', 'SELF_SUPERVISED'],\
        "PROBLEM.TYPE not in ['SEMANTIC_SEG', 'INSTANCE_SEG', 'CLASSIFICATION', 'DETECTION', 'DENOISING', 'SUPER_RESOLUTION', 'SELF_SUPERVISED']"

    if cfg.PROBLEM.NDIM == "2D" and not cfg.TEST.STATS.PER_PATCH and not cfg.TEST.STATS.FULL_IMG:
        raise ValueError("At least one between 'TEST.STATS.PER_PATCH' or 'TEST.STATS.FULL_IMG' need to be True")

    if cfg.PROBLEM.NDIM == '3D':
        if not cfg.TEST.STATS.PER_PATCH and not cfg.TEST.STATS.MERGE_PATCHES and cfg.PROBLEM.TYPE != "CLASSIFICATION":
            raise ValueError("At least one between 'TEST.STATS.PER_PATCH' or 'TEST.STATS.MERGE_PATCHES' need to be True when 'PROBLEM.NDIM'=='3D'")
        if cfg.TEST.STATS.FULL_IMG:
            print("WARNING: TEST.STATS.FULL_IMG == True while using PROBLEM.NDIM == '3D'. As 3D images are usually 'huge'"
                ", full image statistics will be disabled to avoid GPU memory overflow")

    if cfg.LOSS.TYPE != "CE" and cfg.PROBLEM.TYPE not in ['SEMANTIC_SEG', 'DETECTION']:
        raise ValueError("Not implemented pipeline option: LOSS.TYPE != 'CE' only available in 'SEMANTIC_SEG' and 'DETECTION'")
    
    if cfg.TEST.ENABLE and cfg.TEST.ANALIZE_2D_IMGS_AS_3D_STACK and cfg.PROBLEM.NDIM == "3D":
        raise ValueError("'TEST.ANALIZE_2D_IMGS_AS_3D_STACK' makes no sense when the problem is 3D. Disable it.")

    #### Semantic segmentation ####
    if cfg.PROBLEM.TYPE == 'SEMANTIC_SEG':
        if cfg.MODEL.N_CLASSES == 0:
            raise ValueError("'MODEL.N_CLASSES' can not be 0")
        if cfg.LOSS.TYPE == "MASKED_BCE":
            if cfg.MODEL.N_CLASSES > 1:
                raise ValueError("Not implemented pipeline option: N_CLASSES > 1 and MASKED_BCE")
                    
    #### Instance segmentation ####
    if cfg.PROBLEM.TYPE == 'INSTANCE_SEG':
        assert cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS in ['BC', 'BCM', 'BCD', 'BCDv2', 'Dv2', 'BDv2', 'BP', 'BD'],\
            "PROBLEM.INSTANCE_SEG.DATA_CHANNELS not in ['BC', 'BCM', 'BCD', 'BCDv2', 'Dv2', 'BDv2', 'BP', 'BD']"
        if cfg.MODEL.N_CLASSES > 1:
            raise ValueError("Not implemented pipeline option for INSTANCE_SEGMENTATION")
        if len(cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNEL_WEIGHTS) != channels_provided:
            raise ValueError("'PROBLEM.INSTANCE_SEG.DATA_CHANNEL_WEIGHTS' needs to be of the same length as the channels selected in 'PROBLEM.INSTANCE_SEG.DATA_CHANNELS'. "
                            "E.g. 'PROBLEM.INSTANCE_SEG.DATA_CHANNELS'='BC' 'PROBLEM.INSTANCE_SEG.DATA_CHANNEL_WEIGHTS'=[1,0.5]. "
                            "'PROBLEM.INSTANCE_SEG.DATA_CHANNELS'='BCD' 'PROBLEM.INSTANCE_SEG.DATA_CHANNEL_WEIGHTS'=[0.5,0.5,1]")
        if cfg.TEST.POST_PROCESSING.VORONOI_ON_MASK:
            if cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS not in ['BC', 'BCM', 'BCD', 'BCDv2']:
                raise ValueError("'PROBLEM.INSTANCE_SEG.DATA_CHANNELS' need to be one between ['BC', 'BCM', 'BCD', 'BCDv2'] "
                                "when 'TEST.POST_PROCESSING.VORONOI_ON_MASK' is enabled")
            if not check_value(cfg.TEST.POST_PROCESSING.VORONOI_TH):
                raise ValueError("'TEST.POST_PROCESSING.VORONOI_TH' not in [0, 1] range")   
        if cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS not in ["BC", "BCM", "BCD", "BP"] and cfg.PROBLEM.INSTANCE_SEG.ERODE_AND_DILATE_FOREGROUND:
            raise ValueError("'PROBLEM.INSTANCE_SEG.ERODE_AND_DILATE_FOREGROUND' can only be used with 'BC', 'BCM', 'BP' or 'BCD' channels")
        for morph_operation in cfg.PROBLEM.INSTANCE_SEG.SEED_MORPH_SEQUENCE:
            if morph_operation != "dilate" and morph_operation != "erode":
                raise ValueError("'PROBLEM.INSTANCE_SEG.SEED_MORPH_SEQUENCE' can only be a sequence with 'dilate' or 'erode' operations. "
                    "{} given".format(cfg.PROBLEM.INSTANCE_SEG.SEED_MORPH_SEQUENCE))
        if len(cfg.PROBLEM.INSTANCE_SEG.SEED_MORPH_SEQUENCE) != len(cfg.PROBLEM.INSTANCE_SEG.SEED_MORPH_RADIUS):
            raise ValueError("'PROBLEM.INSTANCE_SEG.SEED_MORPH_SEQUENCE' length and 'PROBLEM.INSTANCE_SEG.SEED_MORPH_RADIUS' length need to be the same")
        if cfg.PROBLEM.INSTANCE_SEG.DATA_CONTOUR_MODE not in ['thick', 'inner', 'outer', 'subpixel', 'dense']:
            raise ValueError("'PROBLEM.INSTANCE_SEG.DATA_CONTOUR_MODE' must be one between ['thick', 'inner', 'outer', 'subpixel', 'dense']")
        if cfg.PROBLEM.INSTANCE_SEG.DATA_CONTOUR_MODE == 'dense' and cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS == "BCM":
            raise ValueError("'PROBLEM.INSTANCE_SEG.DATA_CONTOUR_MODE' can not be 'dense' when 'PROBLEM.INSTANCE_SEG.DATA_CHANNELS' is 'BCM'"
                " as it does not have sense")

    #### Detection ####
    if cfg.PROBLEM.TYPE == 'DETECTION':
        if cfg.MODEL.N_CLASSES == 0:
            raise ValueError("'MODEL.N_CLASSES' can not be 0")
        if cfg.TEST.POST_PROCESSING.DET_WATERSHED:
            if any(len(x) != dim_count for x in cfg.TEST.POST_PROCESSING.DET_WATERSHED_FIRST_DILATION):
                raise ValueError("Each structure object defined in 'TEST.POST_PROCESSING.DET_WATERSHED_FIRST_DILATION' "
                                 "need to be of {} dimension".format(dim_count))
            if cfg.TEST.POST_PROCESSING.WATERSHED_CIRCULARITY == -1:
                raise ValueError("'TEST.POST_PROCESSING.WATERSHED_CIRCULARITY' need to be set when 'TEST.POST_PROCESSING.DET_WATERSHED' is enabled")

    #### Super-resolution ####
    elif cfg.PROBLEM.TYPE == 'SUPER_RESOLUTION':
        if cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING == 1:
            raise ValueError("Resolution scale must be provided with 'PROBLEM.SUPER_RESOLUTION.UPSCALING' variable")
        assert cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING in [2, 4], "PROBLEM.SUPER_RESOLUTION.UPSCALING not in [2, 4]"

    #### Self-supervision ####
    elif cfg.PROBLEM.TYPE == 'SELF_SUPERVISED':
        if cfg.PROBLEM.SELF_SUPERVISED.PRETEXT_TASK == "crappify":
            if cfg.PROBLEM.SELF_SUPERVISED.RESIZING_FACTOR not in [2,4,6]:
                raise ValueError("PROBLEM.SELF_SUPERVISED.RESIZING_FACTOR not in [2,4,6]")
            if not check_value(cfg.PROBLEM.SELF_SUPERVISED.NOISE):
                raise ValueError("PROBLEM.SELF_SUPERVISED.NOISE not in [0, 1] range")
        elif cfg.PROBLEM.SELF_SUPERVISED.PRETEXT_TASK == "masking":
            if cfg.MODEL.ARCHITECTURE != 'mae':
                raise ValueError("'MODEL.ARCHITECTURE' need to be 'mae' when 'PROBLEM.SELF_SUPERVISED.PRETEXT_TASK' is 'masking'")  
        else:
            raise ValueError("'PROBLEM.SELF_SUPERVISED.PRETEXT_TASK' need to be among these options: ['crappify', 'masking']")
    #### Denoising ####
    elif cfg.PROBLEM.TYPE == 'DENOISING':
        if cfg.DATA.TEST.LOAD_GT:
            raise ValueError("Denoising is made in an unsupervised way so there is no ground truth required. Disable 'DATA.TEST.LOAD_GT'")
        if not cfg.DATA.TRAIN.IN_MEMORY or not cfg.DATA.VAL.IN_MEMORY:
            raise NotImplementedError
        if not check_value(cfg.PROBLEM.DENOISING.N2V_PERC_PIX):
            raise ValueError("PROBLEM.DENOISING.N2V_PERC_PIX not in [0, 1] range")
           

    ### Pre-processing ###
    if cfg.DATA.EXTRACT_RANDOM_PATCH and cfg.DATA.PROBABILITY_MAP:
        if cfg.DATA.W_FOREGROUND+cfg.DATA.W_BACKGROUND != 1:
            raise ValueError("cfg.DATA.W_FOREGROUND+cfg.DATA.W_BACKGROUND need to sum 1. E.g. 0.94 and 0.06 respectively.")

    #### Data #### 
    if cfg.TRAIN.ENABLE:
        if not os.path.exists(cfg.DATA.TRAIN.PATH):
            raise ValueError("Train data dir not found: {}".format(cfg.DATA.TRAIN.PATH))
        if not os.path.exists(cfg.DATA.TRAIN.GT_PATH) and cfg.PROBLEM.TYPE not in ['DENOISING', "CLASSIFICATION", "SELF_SUPERVISED"]:
            raise ValueError("Train mask data dir not found: {}".format(cfg.DATA.TRAIN.GT_PATH))
        if not cfg.DATA.VAL.FROM_TRAIN and not cfg.DATA.VAL.IN_MEMORY:
            if not os.path.exists(cfg.DATA.VAL.PATH):
                raise ValueError("Validation data dir not found: {}".format(cfg.DATA.VAL.PATH))
            if not os.path.exists(cfg.DATA.VAL.GT_PATH) and cfg.PROBLEM.TYPE not in ['DENOISING', "CLASSIFICATION", "SELF_SUPERVISED"]:
                raise ValueError("Validation mask data dir not found: {}".format(cfg.DATA.VAL.GT_PATH))
    if cfg.TEST.ENABLE and not cfg.DATA.TEST.USE_VAL_AS_TEST:
        if not os.path.exists(cfg.DATA.TEST.PATH):
            raise ValueError("Test data not found: {}".format(cfg.DATA.TEST.PATH))
        if cfg.DATA.TEST.LOAD_GT and not os.path.exists(cfg.DATA.TEST.GT_PATH) and cfg.PROBLEM.TYPE not in ["CLASSIFICATION", "SELF_SUPERVISED"]:
            raise ValueError("Test data mask not found: {}".format(cfg.DATA.TEST.GT_PATH))

    if cfg.DATA.EXTRACT_RANDOM_PATCH and cfg.DATA.PROBABILITY_MAP:
        if not cfg.PROBLEM.TYPE == 'SEMANTIC_SEG':
            raise ValueError("'DATA.PROBABILITY_MAP' can only be selected when 'PROBLEM.TYPE' is 'SEMANTIC_SEG'")

    if cfg.DATA.VAL.FROM_TRAIN and not cfg.DATA.VAL.CROSS_VAL and cfg.DATA.VAL.SPLIT_TRAIN <= 0:
        raise ValueError("'DATA.VAL.SPLIT_TRAIN' needs to be > 0 when 'DATA.VAL.FROM_TRAIN' == True")
    
    if cfg.DATA.VAL.FROM_TRAIN and not cfg.DATA.TRAIN.IN_MEMORY:
        raise ValueError("Validation can not be extracted from train when 'DATA.TRAIN.IN_MEMORY' == False. Please set"
                         " 'DATA.VAL.FROM_TRAIN' to False and configure 'DATA.VAL.PATH'/'DATA.VAL.GT_PATH'")
    if cfg.DATA.VAL.CROSS_VAL: 
        if not cfg.DATA.VAL.FROM_TRAIN:
            raise ValueError("'DATA.VAL.CROSS_VAL' can only be used when 'DATA.VAL.FROM_TRAIN' is True")
        if cfg.DATA.VAL.CROSS_VAL_NFOLD < cfg.DATA.VAL.CROSS_VAL_FOLD:
            raise ValueError("'DATA.VAL.CROSS_VAL_NFOLD' can not be less than 'DATA.VAL.CROSS_VAL_FOLD'")
        if not cfg.DATA.VAL.IN_MEMORY:
            print("WARNING: ignoring 'DATA.VAL.IN_MEMORY' as it is always True when 'DATA.VAL.CROSS_VAL' is enabled")
    if cfg.DATA.TEST.USE_VAL_AS_TEST and not cfg.DATA.VAL.CROSS_VAL:
        raise ValueError("'DATA.TEST.USE_VAL_AS_TEST' can only be used when 'DATA.VAL.CROSS_VAL' is selected")
    if cfg.DATA.TEST.USE_VAL_AS_TEST and not cfg.TRAIN.ENABLE and cfg.DATA.TEST.IN_MEMORY:
        print("WARNING: 'DATA.TEST.IN_MEMORY' is disabled when 'DATA.TEST.USE_VAL_AS_TEST' is enabled")
    if len(cfg.DATA.TRAIN.RESOLUTION) != dim_count:
        raise ValueError("Train resolution needs to be a tuple with {} values".format(dim_count))
    if len(cfg.DATA.VAL.RESOLUTION) != dim_count:
        raise ValueError("Validation resolution needs to be a tuple with {} values".format(dim_count))
    if len(cfg.DATA.TEST.RESOLUTION) != dim_count:
        raise ValueError("Test resolution needs to be a tuple with {} values".format(dim_count))

    if len(cfg.DATA.TRAIN.OVERLAP) != dim_count:
        raise ValueError("When PROBLEM.NDIM == {} DATA.TRAIN.OVERLAP tuple must be lenght {}, given {}."
                         .format(cfg.PROBLEM.NDIM, dim_count, cfg.DATA.TRAIN.OVERLAP))
    if any(not check_value(x) for x in cfg.DATA.TRAIN.OVERLAP):
            raise ValueError("DATA.TRAIN.OVERLAP not in [0, 1] range")
    if len(cfg.DATA.TRAIN.PADDING) != dim_count:
        raise ValueError("When PROBLEM.NDIM == {} DATA.TRAIN.PADDING tuple must be lenght {}, given {}."
                         .format(cfg.PROBLEM.NDIM, dim_count, cfg.DATA.TRAIN.PADDING))
    if len(cfg.DATA.VAL.OVERLAP) != dim_count:
        raise ValueError("When PROBLEM.NDIM == {} DATA.VAL.OVERLAP tuple must be lenght {}, given {}."
                         .format(cfg.PROBLEM.NDIM, dim_count, cfg.DATA.VAL.OVERLAP))
    if any(not check_value(x) for x in cfg.DATA.VAL.OVERLAP):
            raise ValueError("DATA.VAL.OVERLAP not in [0, 1] range")
    if len(cfg.DATA.VAL.PADDING) != dim_count:
        raise ValueError("When PROBLEM.NDIM == {} DATA.VAL.PADDING tuple must be lenght {}, given {}."
                         .format(cfg.PROBLEM.NDIM, dim_count, cfg.DATA.VAL.PADDING))
    if len(cfg.DATA.TEST.OVERLAP) != dim_count:
        raise ValueError("When PROBLEM.NDIM == {} DATA.TEST.OVERLAP tuple must be lenght {}, given {}."
                         .format(cfg.PROBLEM.NDIM, dim_count, cfg.DATA.TEST.OVERLAP))
    if any(not check_value(x) for x in cfg.DATA.TEST.OVERLAP):
            raise ValueError("DATA.TEST.OVERLAP not in [0, 1] range")
    if len(cfg.DATA.TEST.PADDING) != dim_count:
        raise ValueError("When PROBLEM.NDIM == {} DATA.TEST.PADDING tuple must be lenght {}, given {}."
                         .format(cfg.PROBLEM.NDIM, dim_count, cfg.DATA.TEST.PADDING))
    if len(cfg.DATA.PATCH_SIZE) != dim_count+1:
        raise ValueError("When PROBLEM.NDIM == {} DATA.PATCH_SIZE tuple must be lenght {}, given {}."
                         .format(cfg.PROBLEM.NDIM, dim_count+1, cfg.DATA.PATCH_SIZE))
    if len(cfg.DATA.TRAIN.RESOLUTION) != 1 and len(cfg.DATA.TRAIN.RESOLUTION) != dim_count:
        raise ValueError("When PROBLEM.NDIM == {} DATA.TRAIN.RESOLUTION tuple must be lenght {}, given {}."
                         .format(cfg.PROBLEM.NDIM, dim_count, cfg.DATA.TRAIN.RESOLUTION))
    if len(cfg.DATA.VAL.RESOLUTION) != 1 and len(cfg.DATA.VAL.RESOLUTION) != dim_count:
        raise ValueError("When PROBLEM.NDIM == {} DATA.VAL.RESOLUTION tuple must be lenght {}, given {}."
                         .format(cfg.PROBLEM.NDIM, dim_count, cfg.DATA.VAL.RESOLUTION))
    if len(cfg.DATA.TEST.RESOLUTION) != 1 and len(cfg.DATA.TEST.RESOLUTION) != dim_count:
        raise ValueError("When PROBLEM.NDIM == {} DATA.TEST.RESOLUTION tuple must be lenght {}, given {}."
                         .format(cfg.PROBLEM.NDIM, dim_count, cfg.DATA.TEST.RESOLUTION))
    assert cfg.DATA.NORMALIZATION.TYPE in ['div', 'custom'], "DATA.NORMALIZATION.TYPE not in ['div', 'custom']"
    if cfg.DATA.NORMALIZATION.TYPE == 'custom':
        if cfg.DATA.NORMALIZATION.CUSTOM_MEAN == -1 and cfg.DATA.NORMALIZATION.CUSTOM_STD == -1:
            if not os.path.exists(cfg.PATHS.MEAN_INFO_FILE) or not os.path.exists(cfg.PATHS.STD_INFO_FILE):
                if not cfg.DATA.TRAIN.IN_MEMORY:
                    raise ValueError("If no 'DATA.NORMALIZATION.CUSTOM_MEAN' and 'DATA.NORMALIZATION.CUSTOM_STD' were provided "
                        "when DATA.NORMALIZATION.TYPE == 'custom', DATA.TRAIN.IN_MEMORY need to be True")

    ### Model ###
    assert cfg.MODEL.ARCHITECTURE in ['unet', 'resunet', 'attention_unet', 'fcn32', 'fcn8', 'tiramisu', 'mnet',
                                      'multiresunet', 'seunet', 'simple_cnn', 'EfficientNetB0', 'unetr', 'edsr',
                                      'rcan', 'dfcan', 'wdsr', 'ViT', 'mae'],\
        "MODEL.ARCHITECTURE not in ['unet', 'resunet', 'attention_unet', 'fcn32', 'fcn8', 'tiramisu', 'mnet',\
        'multiresunet', 'seunet', 'simple_cnn', 'EfficientNetB0', 'unetr', 'edsr', 'rcan', 'dfcan', \
        'wdsr', 'ViT', 'mae']"
    if cfg.MODEL.ARCHITECTURE not in ['unet', 'resunet', 'seunet', 'attention_unet', 'unetr', 'ViT', 'mae'] and cfg.PROBLEM.NDIM == '3D' and cfg.PROBLEM.TYPE != "CLASSIFICATION":
        raise ValueError("For 3D these models are available: {}".format(['unet', 'resunet', 'seunet', 'attention_unet']))
    if cfg.MODEL.N_CLASSES > 1 and cfg.PROBLEM.TYPE != "CLASSIFICATION" and cfg.MODEL.ARCHITECTURE not in ['unet', 'resunet', 'seunet', 'attention_unet']:
        raise ValueError("'MODEL.N_CLASSES' > 1 can only be used with 'MODEL.ARCHITECTURE' in ['unet', 'resunet', 'seunet', 'attention_unet']")
    if cfg.MODEL.LAST_ACTIVATION not in ['softmax', 'sigmoid', 'linear']:
        raise ValueError("'MODEL.LAST_ACTIVATION' need to be in ['softmax','sigmoid','linear']. Provided {}"
                         .format(cfg.MODEL.LAST_ACTIVATION))
    if cfg.MODEL.UPSAMPLE_LAYER.lower() not in ["upsampling", "convtranspose"]:
        raise ValueError("cfg.MODEL.UPSAMPLE_LAYER' need to be one between ['upsampling', 'convtranspose']. Provided {}"
                          .format(cfg.MODEL.UPSAMPLE_LAYER))
    if cfg.PROBLEM.TYPE == "SEMANTIC_SEG" and cfg.MODEL.ARCHITECTURE not in ['unet', 'resunet', 'attention_unet', 'fcn32', \
        'fcn8', 'tiramisu', 'mnet', 'multiresunet', 'seunet', 'unetr']:
        raise ValueError("Not implemented pipeline option: semantic segmentation models are ['unet', 'resunet', "
                         "'attention_unet', 'fcn32', 'fcn8', 'tiramisu', 'mnet', 'multiresunet', 'seunet', 'unetr']")
    if cfg.PROBLEM.TYPE == "INSTANCE_SEG" and cfg.MODEL.ARCHITECTURE not in ['unet', 'resunet', 'seunet', 'attention_unet', 'unetr']:
        raise ValueError("Not implemented pipeline option: instance segmentation models are ['unet', 'resunet', 'seunet', 'attention_unet']")    
    if cfg.PROBLEM.TYPE in ['DETECTION', 'DENOISING'] and \
        cfg.MODEL.ARCHITECTURE not in ['unet', 'resunet', 'seunet', 'attention_unet']:
        raise ValueError("Architectures available for {} are: ['unet', 'resunet', 'seunet', 'attention_unet']"
                         .format(cfg.PROBLEM.TYPE))
    if cfg.PROBLEM.TYPE == 'SUPER_RESOLUTION':
        if cfg.PROBLEM.NDIM == '2D' and cfg.MODEL.ARCHITECTURE not in ['edsr', 'rcan', 'dfcan', 'wdsr', 'unet', 'resunet', 'seunet', 'attention_unet']:
            raise ValueError("Architectures available for 2D 'SUPER_RESOLUTION' are: ['edsr', 'rcan', 'dfcan', 'wdsr', 'unet', 'resunet', 'seunet', 'attention_unet']")
        elif cfg.PROBLEM.NDIM == '3D':
            if cfg.MODEL.ARCHITECTURE not in ['unet', 'resunet', 'seunet', 'attention_unet']:
                raise ValueError("Architectures available for 3D 'SUPER_RESOLUTION' are: ['unet', 'resunet', 'seunet', 'attention_unet']")
            assert cfg.MODEL.UNET_SR_UPSAMPLE_POSITION in ["pre", "post"], "'MODEL.UNET_SR_UPSAMPLE_POSITION' not in ['pre', 'post']"
    if cfg.PROBLEM.TYPE == 'CLASSIFICATION' and cfg.MODEL.ARCHITECTURE not in ['simple_cnn', 'EfficientNetB0', 'ViT']:
        raise ValueError("Architectures available for 'CLASSIFICATION' are: ['simple_cnn', 'EfficientNetB0', 'ViT']")
    if cfg.MODEL.ARCHITECTURE in ['unetr', 'ViT', 'mae']:    
        if cfg.MODEL.VIT_HIDDEN_SIZE % cfg.MODEL.VIT_NUM_HEADS != 0:
            raise ValueError("'MODEL.VIT_HIDDEN_SIZE' should be divisible by 'MODEL.VIT_NUM_HEADS'")
        if not all([i == cfg.DATA.PATCH_SIZE[0] for i in cfg.DATA.PATCH_SIZE[:-1]]):      
            raise ValueError("'unetr', 'ViT' 'mae' models need to have same shape in all dimensions (e.g. DATA.PATCH_SIZE = (80,80,80,1) )")

    ### Train ###
    assert cfg.TRAIN.OPTIMIZER in ['SGD', 'ADAM', 'ADAMW'], "TRAIN.OPTIMIZER not in ['SGD', 'ADAM', 'ADAMW']"
    assert cfg.LOSS.TYPE in ['CE', 'W_CE_DICE', 'MASKED_BCE'], "LOSS.TYPE not in ['CE', 'W_CE_DICE', 'MASKED_BCE']"
    if cfg.TRAIN.LR_SCHEDULER.NAME != '':
        if cfg.TRAIN.LR_SCHEDULER.NAME not in ['reduceonplateau', 'warmupcosine', 'onecycle']:
            raise ValueError("'TRAIN.LR_SCHEDULER.NAME' must be one between ['reduceonplateau', 'warmupcosine', 'onecycle']")
        if cfg.TRAIN.LR_SCHEDULER.MIN_LR == -1. and cfg.TRAIN.LR_SCHEDULER.NAME != 'onecycle':
            raise ValueError("'TRAIN.LR_SCHEDULER.MIN_LR' need to be set when 'TRAIN.LR_SCHEDULER.NAME' is between ['reduceonplateau', 'warmupcosine']")

        if cfg.TRAIN.LR_SCHEDULER.NAME == 'reduceonplateau':
            if cfg.TRAIN.LR_SCHEDULER.REDUCEONPLATEAU_PATIENCE == -1:
                raise ValueError("'TRAIN.LR_SCHEDULER.REDUCEONPLATEAU_PATIENCE' need to be set when 'TRAIN.LR_SCHEDULER.NAME' is 'reduceonplateau'")
            if cfg.TRAIN.LR_SCHEDULER.REDUCEONPLATEAU_PATIENCE >= cfg.TRAIN.PATIENCE:
                raise ValueError("'TRAIN.LR_SCHEDULER.REDUCEONPLATEAU_PATIENCE' need to be less than 'TRAIN.PATIENCE' ")
      
        if cfg.TRAIN.LR_SCHEDULER.NAME == 'warmupcosine':
            if cfg.TRAIN.LR_SCHEDULER.WARMUP_COSINE_DECAY_LR == -1.:
                raise ValueError("'TRAIN.LR_SCHEDULER.WARMUP_COSINE_DECAY_LR' need to be set when 'TRAIN.LR_SCHEDULER.NAME' is 'warmupcosine'")
            if cfg.TRAIN.LR_SCHEDULER.WARMUP_COSINE_DECAY_LR >= cfg.TRAIN.LR:
                raise ValueError("'TRAIN.LR_SCHEDULER.WARMUP_COSINE_DECAY_LR' need to be lower than 'TRAIN.LR'")
            if cfg.TRAIN.LR_SCHEDULER.WARMUP_COSINE_DECAY_HOLD_EPOCHS == -1:
                raise ValueError("'TRAIN.LR_SCHEDULER.WARMUP_COSINE_DECAY_HOLD_EPOCHS' need to be set when 'TRAIN.LR_SCHEDULER.NAME' is 'warmupcosine'")
            if cfg.TRAIN.LR_SCHEDULER.WARMUP_COSINE_DECAY_EPOCHS == -1:
                raise ValueError("'TRAIN.LR_SCHEDULER.WARMUP_COSINE_DECAY_EPOCHS' need to be set when 'TRAIN.LR_SCHEDULER.NAME' is 'warmupcosine'")
             
    #### Augmentation ####
    if cfg.AUGMENTOR.ENABLE:
        if not check_value(cfg.AUGMENTOR.DA_PROB):
            raise ValueError("AUGMENTOR.DA_PROB not in [0, 1] range")
        if cfg.AUGMENTOR.RANDOM_ROT:
            if not check_value(cfg.AUGMENTOR.RANDOM_ROT_RANGE, (-360,360)):
                raise ValueError("AUGMENTOR.RANDOM_ROT_RANGE values need to be between [-360,360]")
        if cfg.AUGMENTOR.SHEAR:
            if not check_value(cfg.AUGMENTOR.SHEAR_RANGE, (-360,360)):
                raise ValueError("AUGMENTOR.SHEAR_RANGE values need to be between [-360,360]")
        if cfg.AUGMENTOR.ELASTIC:
            if cfg.AUGMENTOR.E_MODE not in ['constant', 'nearest', 'reflect', 'wrap']:
                raise ValueError("AUGMENTOR.E_MODE not in ['constant', 'nearest', 'reflect', 'wrap']")
        if cfg.AUGMENTOR.BRIGHTNESS: 
            if cfg.AUGMENTOR.BRIGHTNESS_MODE not in ['2D', '3D'] and cfg.PROBLEM.NDIM == "3D":
                raise ValueError("AUGMENTOR.BRIGHTNESS_MODE not in ['2D', '3D']")
        if cfg.AUGMENTOR.CONTRAST:
            if cfg.AUGMENTOR.CONTRAST_MODE not in ['2D', '3D'] and cfg.PROBLEM.NDIM == "3D":
                raise ValueError("AUGMENTOR.CONTRAST_MODE not in ['2D', '3D']")
        if cfg.AUGMENTOR.BRIGHTNESS_EM:
            if cfg.AUGMENTOR.BRIGHTNESS_EM_MODE not in ['2D', '3D'] and cfg.PROBLEM.NDIM == "3D":
                raise ValueError("AUGMENTOR.BRIGHTNESS_EM_MODE not in ['2D', '3D']")
        if cfg.AUGMENTOR.CONTRAST_EM:
            if cfg.AUGMENTOR.CONTRAST_EM_MODE not in ['2D', '3D'] and cfg.PROBLEM.NDIM == "3D":
                raise ValueError("AUGMENTOR.CONTRAST_EM_MODE not in ['2D', '3D']")
        if cfg.AUGMENTOR.DROPOUT:
            if not check_value(cfg.AUGMENTOR.DROP_RANGE):
                raise ValueError("AUGMENTOR.DROP_RANGE values not in [0, 1] range")
        if cfg.AUGMENTOR.CUTOUT:
            if not check_value(cfg.AUGMENTOR.COUT_SIZE):
                raise ValueError("AUGMENTOR.COUT_SIZE values not in [0, 1] range")
        if cfg.AUGMENTOR.CUTBLUR:
            if not check_value(cfg.AUGMENTOR.CBLUR_SIZE):
                raise ValueError("AUGMENTOR.CBLUR_SIZE values not in [0, 1] range")
            if not check_value(cfg.AUGMENTOR.CBLUR_DOWN_RANGE, (1,8)):
                raise ValueError("AUGMENTOR.CBLUR_DOWN_RANGE values not in [1, 8] range")
        if cfg.AUGMENTOR.CUTMIX:
            if not check_value(cfg.AUGMENTOR.CMIX_SIZE):
                raise ValueError("AUGMENTOR.CMIX_SIZE values not in [0, 1] range")
        if cfg.AUGMENTOR.CUTNOISE:
            if not check_value(cfg.AUGMENTOR.CNOISE_SCALE):
                raise ValueError("AUGMENTOR.CNOISE_SCALE values not in [0, 1] range")
            if not check_value(cfg.AUGMENTOR.CNOISE_SIZE):
                raise ValueError("AUGMENTOR.CNOISE_SIZE values not in [0, 1] range")
        if cfg.AUGMENTOR.GRIDMASK:
            if not check_value(cfg.AUGMENTOR.GRID_RATIO):
                raise ValueError("AUGMENTOR.GRID_RATIO not in [0, 1] range")
            if cfg.AUGMENTOR.GRID_D_RANGE[0] >= cfg.AUGMENTOR.GRID_D_RANGE[1]:
                raise ValueError("cfg.AUGMENTOR.GRID_D_RANGE[0] needs to be larger than cfg.AUGMENTOR.GRID_D_RANGE[1]"
                                "Provided {}".format(cfg.AUGMENTOR.GRID_D_RANGE))
            if not check_value(cfg.AUGMENTOR.GRID_D_RANGE):
                raise ValueError("cfg.AUGMENTOR.GRID_D_RANGE values not in [0, 1] range")
            if not check_value(cfg.AUGMENTOR.GRID_ROTATE):
                raise ValueError("AUGMENTOR.GRID_ROTATE not in [0, 1] range")
                             
    #### Post-processing ####
    if cfg.TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS:
        if len(cfg.DATA.TEST.RESOLUTION) == 1:
            raise ValueError("'DATA.TEST.RESOLUTION' must be set when using 'TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS'")
        if len(cfg.DATA.TEST.RESOLUTION) != dim_count:
            raise ValueError("'DATA.TEST.RESOLUTION' must match in length to {}, which is the number of "
                             "dimensions".format(dim_count))
        if cfg.TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS_RADIUS[0] == -1:
            raise ValueError("'TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS' need to be set when 'TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS' is True")   
