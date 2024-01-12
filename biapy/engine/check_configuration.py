import os
import numpy as np
import collections
import requests
from pathlib import Path
from biapy.utils.misc import get_checkpoint_path
from biapy.utils.util import check_value

def check_configuration(cfg, jobname, check_data_paths=True):
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
                raise ValueError("'TEST.POST_PROCESSING.DET_WATERSHED_FIRST_DILATION' needs to be a list of list") 
            if any(y == -1 for y in x):
                raise ValueError("Please set 'TEST.POST_PROCESSING.DET_WATERSHED_FIRST_DILATION' when using 'TEST.POST_PROCESSING.DET_WATERSHED_FIRST_DILATION'")
            if len(x) != dim_count:
                raise ValueError("'TEST.POST_PROCESSING.DET_WATERSHED_FIRST_DILATION' needs to be of dimension {} for {} problem".format(dim_count, cfg.PROBLEM.NDIM))
        if cfg.TEST.POST_PROCESSING.DET_WATERSHED_DONUTS_CLASSES != [-1]:
            if len(cfg.TEST.POST_PROCESSING.DET_WATERSHED_DONUTS_CLASSES) > cfg.MODEL.N_CLASSES:
                raise ValueError("'TEST.POST_PROCESSING.DET_WATERSHED_DONUTS_CLASSES' length can't be greater than 'MODEL.N_CLASSES'")
            if np.max(cfg.TEST.POST_PROCESSING.DET_WATERSHED_DONUTS_CLASSES) > cfg.MODEL.N_CLASSES:
                raise ValueError("'TEST.POST_PROCESSING.DET_WATERSHED_DONUTS_CLASSES' can not have a class number greater than 'MODEL.N_CLASSES'")
            min_class = np.min(cfg.TEST.POST_PROCESSING.DET_WATERSHED_DONUTS_CLASSES)
            if not all(cfg.TEST.POST_PROCESSING.DET_WATERSHED_DONUTS_CLASSES == np.array(range(min_class,len(cfg.TEST.POST_PROCESSING.DET_WATERSHED_DONUTS_CLASSES)+1))):
                raise ValueError("'TEST.POST_PROCESSING.DET_WATERSHED_DONUTS_CLASSES' must be consecutive, e.g [1,2,3,4..]")
            if len(cfg.TEST.POST_PROCESSING.DET_WATERSHED_DONUTS_PATCH) != dim_count:
                raise ValueError("'TEST.POST_PROCESSING.DET_WATERSHED_DONUTS_PATCH' needs to be of dimension {} for {} problem".format(dim_count, cfg.PROBLEM.NDIM))

    if not (len(cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS) == \
        len(cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.VALUES) == \
        len(cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.SIGN)):
        raise ValueError("'TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS', 'TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.VALUES' and "
            "'TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.SIGN' need to have same length")
            
    if cfg.PROBLEM.TYPE == 'DETECTION':
        if cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.ENABLE and \
            cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.ENABLE:
            for i in range(len(cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS)):
                if len(cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS[i]) > 1:
                    raise ValueError("In DETECTION 'TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS' can only be used for filtering 'circularity'.")
                if len(cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS[i]) == 1 and cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS[i][0] != 'circularity':  
                    raise ValueError("In DETECTION 'TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS' can only be used for filtering 'circularity'.")
    
    if cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.ENABLE and \
        cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.ENABLE:
        if cfg.PROBLEM.TYPE not in ['INSTANCE_SEG', 'DETECTION']:
            raise ValueError("'TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS' can only be used in INSTANCE_SEG and DETECTION workflows")

        if len(cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS) == 0:
            raise ValueError("'TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS' can not be an empty list when "
                "'TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.ENABLE' is enabled")

        for i in range(len(cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS)):
            if not isinstance(cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS[i], list):
                raise ValueError("'TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS' need to be a list of list. E.g. [ ['circularity'], ['area', 'diameter'] ]")
            if not isinstance(cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.VALUES[i], list):
                raise ValueError("'TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.VALUES' need to be a list of list. E.g. [ [10], [15, 3] ]")
            if not isinstance(cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.SIGN[i], list):
                raise ValueError("'TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.SIGN' need to be a list of list. E.g. [ ['gt'], ['le', 'gt'] ]")

            if not (len(cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS[i]) == \
                len(cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.VALUES[i]) == \
                len(cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.SIGN[i])):
                raise ValueError("'TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS', 'TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.VALUES' and "
                    "'TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.SIGN' need to have same length")

            # Check for unique values 
            if len([item for item, count in collections.Counter(cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS[i]).items() if count > 1]) > 0:
                raise ValueError("Non repeated values are allowed in 'TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES'")
            for j in range(len(cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS[i])):
                if cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS[i][j] not in ['circularity', 'npixels', 'area', 'diameter', 'elongation', 'sphericity', 'perimeter']:
                    raise ValueError("'TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS' can only be one among these: ['circularity', 'npixels', 'area', 'diameter', 'elongation', 'sphericity', 'perimeter']")
                if cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS[i][j] in ["circularity", "elongation"] and cfg.PROBLEM.NDIM != '2D':
                    raise ValueError("'circularity' or 'elongation' properties can only be measured in 2D images. Delete them from 'TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS'")
                if cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS[i][j] == "sphericity" and cfg.PROBLEM.NDIM != '3D':
                    raise ValueError("'sphericity' property can only be measured in 3D images. Delete it from 'TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS'")
                if cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.SIGN[i][j] not in ['gt', 'ge', 'lt', 'le']:
                    raise ValueError("'TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.SIGN' can only be one among these: ['gt', 'ge', 'lt', 'le']")
                if cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS[i][j] == "circularity" and not check_value(cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.VALUES[i][j]):
                    raise ValueError("Circularity can only have values in [0, 1] range (check  'TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.VALUES' values)")
                    
    if cfg.PROBLEM.TYPE != 'INSTANCE_SEG':  
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

    # First update is done here as some checks from this point need to have those updates
    if len(opts) > 0:
        cfg.merge_from_list(opts)
        opts = []

    #### General checks ####
    assert cfg.PROBLEM.NDIM in ['2D', '3D'], "Problem needs to be '2D' or '3D'"
    assert cfg.PROBLEM.TYPE in ['SEMANTIC_SEG', 'INSTANCE_SEG', 'CLASSIFICATION', 'DETECTION', 'DENOISING', 'SUPER_RESOLUTION', 'SELF_SUPERVISED'],\
        "PROBLEM.TYPE not in ['SEMANTIC_SEG', 'INSTANCE_SEG', 'CLASSIFICATION', 'DETECTION', 'DENOISING', 'SUPER_RESOLUTION', 'SELF_SUPERVISED']"

    if cfg.PROBLEM.NDIM == "2D" and not cfg.TEST.STATS.PER_PATCH and not cfg.TEST.STATS.FULL_IMG:
        raise ValueError("At least one between 'TEST.STATS.PER_PATCH' or 'TEST.STATS.FULL_IMG' needs to be True")

    if cfg.PROBLEM.NDIM == '3D':
        if not cfg.TEST.STATS.PER_PATCH and not cfg.TEST.STATS.MERGE_PATCHES and cfg.PROBLEM.TYPE != "CLASSIFICATION":
            raise ValueError("At least one between 'TEST.STATS.PER_PATCH' or 'TEST.STATS.MERGE_PATCHES' needs to be True when 'PROBLEM.NDIM'=='3D'")
        if cfg.TEST.STATS.FULL_IMG:
            print("WARNING: TEST.STATS.FULL_IMG == True while using PROBLEM.NDIM == '3D'. As 3D images are usually 'huge'"
                ", full image statistics will be disabled to avoid GPU memory overflow")

    if cfg.LOSS.TYPE != "CE" and cfg.PROBLEM.TYPE not in ['SEMANTIC_SEG', 'DETECTION']:
        raise ValueError("Not implemented pipeline option: LOSS.TYPE != 'CE' only available in 'SEMANTIC_SEG' and 'DETECTION'")

    if cfg.TEST.ENABLE and cfg.TEST.ANALIZE_2D_IMGS_AS_3D_STACK and cfg.PROBLEM.NDIM == "3D":
        raise ValueError("'TEST.ANALIZE_2D_IMGS_AS_3D_STACK' makes no sense when the problem is 3D. Disable it.")

    if cfg.MODEL.SOURCE not in ["biapy", "bmz", "torchvision"]:
        raise ValueError("'MODEL.SOURCE' needs to be one between ['biapy', 'bmz', 'torchvision']")

    if cfg.MODEL.SOURCE == "bmz":
        if cfg.TRAIN.ENABLE:
            raise ValueError("Currently not supported to train a BMZ model")
        if cfg.MODEL.BMZ.SOURCE_MODEL_DOI == "":
            raise ValueError("'MODEL.BMZ.SOURCE_MODEL_DOI' needs to be configured when 'MODEL.SOURCE' is 'bmz'")

        # Check if the model exists
        url = 'http://www.doi.org/'+cfg.MODEL.BMZ.SOURCE_MODEL_DOI
        r = requests.get(url, stream=True, verify=True)
        if r.status_code >= 200 and r.status_code < 400:
            print(f'BMZ model DOI: {cfg.MODEL.BMZ.SOURCE_MODEL_DOI} found')
        else:
            raise ValueError(f'BMZ model DOI: {cfg.MODEL.BMZ.SOURCE_MODEL_DOI} not found. Aborting!')

    if cfg.MODEL.BMZ.EXPORT_MODEL.ENABLE:
        if cfg.MODEL.BMZ.EXPORT_MODEL.NAME == "":
            raise ValueError("'MODEL.BMZ.EXPORT_MODEL.NAME' can not be empty when 'MODEL.BMZ.EXPORT_MODEL.ENABLE' is True")
        if cfg.MODEL.BMZ.EXPORT_MODEL.DESCRIPTION == "":
            raise ValueError("'MODEL.BMZ.EXPORT_MODEL.DESCRIPTION' can not be empty when 'MODEL.BMZ.EXPORT_MODEL.ENABLE' is True")
        # Authors of the model. Need to be a list of dicts, e.g. authors=[{"name": "Daniel"}]
        if not isinstance(cfg.MODEL.BMZ.EXPORT_MODEL.AUTHORS, list):
            raise ValueError("'MODEL.BMZ.EXPORT_MODEL.AUTHORS' needs to be a list of dicts. E.g. [{'name': 'Daniel'}]")
        else:
            if len(cfg.MODEL.BMZ.EXPORT_MODEL.AUTHORS) == 0:
                raise ValueError("'MODEL.BMZ.EXPORT_MODEL.AUTHORS' can not be empty when 'MODEL.BMZ.EXPORT_MODEL.ENABLE' is True")
            for d in cfg.MODEL.BMZ.EXPORT_MODEL.AUTHORS:
                if not isinstance(d, dict):
                    raise ValueError("'MODEL.BMZ.EXPORT_MODEL.AUTHORS' must be a list of dicts. E.g. [{'name': 'Daniel'}]")
        if cfg.MODEL.BMZ.EXPORT_MODEL.LICENSE == "":
                raise ValueError("'MODEL.BMZ.EXPORT_MODEL.LICENSE' can not be empty. E.g. 'CC-BY-4.0'")
        if not isinstance(cfg.MODEL.BMZ.EXPORT_MODEL.TAGS, list):
            raise ValueError("'MODEL.BMZ.EXPORT_MODEL.TAGS' needs to be a list of strings. E.g. [{'modality': 'electron-microscopy', 'content': 'mitochondria'}]")
        else:
            if len(cfg.MODEL.BMZ.EXPORT_MODEL.TAGS) == 0:
                raise ValueError("'MODEL.BMZ.EXPORT_MODEL.TAGS' can not be empty when 'MODEL.BMZ.EXPORT_MODEL.ENABLE' is True")
            for d in cfg.MODEL.BMZ.EXPORT_MODEL.TAGS:
                if not isinstance(d, dict):
                    raise ValueError("'MODEL.BMZ.EXPORT_MODEL.TAGS' must be a list of strings. E.g. [{'modality': 'electron-microscopy', 'content': 'mitochondria'}]")
        if not isinstance(cfg.MODEL.BMZ.EXPORT_MODEL.CITE, list):
            raise ValueError("'MODEL.BMZ.EXPORT_MODEL.CITE' needs to be a list of dicts. E.g. [{'text': 'Gizmo et al.', 'doi': 'doi:10.1002/xyzacab123'}]")
        else:
            for d in cfg.MODEL.BMZ.EXPORT_MODEL.CITE:
                if not isinstance(d, dict):
                    raise ValueError("'MODEL.BMZ.EXPORT_MODEL.CITE' needs to be a list of dicts. E.g. [{'text': 'Gizmo et al.', 'doi': 'doi:10.1002/xyzacab123'}]")
        if cfg.MODEL.BMZ.EXPORT_MODEL.DOCUMENTATION == "":
            raise ValueError("'MODEL.BMZ.EXPORT_MODEL.DOCUMENTATION' can not be empty. E.g. '/home/user/my-model/doc.md'")
        else:
            if not os.path.exists(cfg.MODEL.BMZ.EXPORT_MODEL.DOCUMENTATION):
                raise ValueError("'MODEL.BMZ.EXPORT_MODEL.DOCUMENTATION' file does not exist!")

    elif cfg.MODEL.SOURCE == "torchvision":
        if cfg.MODEL.TORCHVISION_MODEL_NAME == "":
            raise ValueError("'MODEL.TORCHVISION_MODEL_NAME' needs to be configured when 'MODEL.SOURCE' is 'torchvision'")
        if cfg.TEST.AUGMENTATION:
            print("WARNING: 'TEST.AUGMENTATION' is not available using TorchVision models")

    if cfg.TEST.AUGMENTATION and cfg.TEST.REDUCE_MEMORY:
        raise ValueError("'TEST.AUGMENTATION' and 'TEST.REDUCE_MEMORY' are incompatible as the function used to make the rotation "
            "does not support float16 data type.")

    model_arch = cfg.MODEL.ARCHITECTURE.lower()
    #### Semantic segmentation ####
    if cfg.PROBLEM.TYPE == 'SEMANTIC_SEG':
        if cfg.MODEL.SOURCE == "biapy":
            if cfg.MODEL.N_CLASSES < 2:
                raise ValueError("'MODEL.N_CLASSES' needs to be greater or equal 2 (binary case)")
            if cfg.LOSS.TYPE == "MASKED_BCE":
                if cfg.MODEL.N_CLASSES > 2:
                    raise ValueError("Not implemented pipeline option: N_CLASSES > 2 and MASKED_BCE")
        elif cfg.MODEL.SOURCE == "torchvision":
            if cfg.MODEL.TORCHVISION_MODEL_NAME not in ['deeplabv3_mobilenet_v3_large', 'deeplabv3_resnet101', 'deeplabv3_resnet50', \
                'fcn_resnet101', 'fcn_resnet50', 'lraspp_mobilenet_v3_large']:
                raise ValueError("'MODEL.SOURCE' must be one between ['deeplabv3_mobilenet_v3_large', 'deeplabv3_resnet101', "
                    "'deeplabv3_resnet50', 'fcn_resnet101', 'fcn_resnet50', 'lraspp_mobilenet_v3_large' ]")
            if cfg.MODEL.TORCHVISION_MODEL_NAME in ['deeplabv3_mobilenet_v3_large'] and cfg.DATA.PATCH_SIZE[-1] != 3:
                raise ValueError("'deeplabv3_mobilenet_v3_large' model expects 3 channel data (RGB). "
                    f"'DATA.PATCH_SIZE' set is {cfg.DATA.PATCH_SIZE}")
            if cfg.PROBLEM.NDIM == '3D':
                raise ValueError("TorchVision model's for semantic segmentation are only available for 2D images")
            if not cfg.TEST.STATS.FULL_IMG:
                raise ValueError("With TorchVision models for semantic segmentation workflow only 'TEST.STATS.FULL_IMG' setting is available, so "
                    "please set it.")
            if cfg.TEST.STATS.PER_PATCH or cfg.TEST.STATS.MERGE_PATCHES:
                raise ValueError("With TorchVision models for semantic segmentation workflow only 'TEST.STATS.FULL_IMG' setting is available, so "
                    "please enable it and disable 'TEST.STATS.PER_PATCH' and 'TEST.STATS.MERGE_PATCHES'")
            if cfg.LOSS.TYPE != "CE":
                raise ValueError("Only 'LOSS.TYPE' = 'CE' is available in semantic segmentation workflow using TorchVision models")

    #### Instance segmentation ####
    if cfg.PROBLEM.TYPE == 'INSTANCE_SEG':
        assert cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS in ['BC', 'BCM', 'BCD', 'BCDv2', 'Dv2', 'BDv2', 'BP', 'BD'],\
            "PROBLEM.INSTANCE_SEG.DATA_CHANNELS not in ['BC', 'BCM', 'BCD', 'BCDv2', 'Dv2', 'BDv2', 'BP', 'BD']"
        if len(cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNEL_WEIGHTS) != channels_provided:
            raise ValueError("'PROBLEM.INSTANCE_SEG.DATA_CHANNEL_WEIGHTS' needs to be of the same length as the channels selected in 'PROBLEM.INSTANCE_SEG.DATA_CHANNELS'. "
                            "E.g. 'PROBLEM.INSTANCE_SEG.DATA_CHANNELS'='BC' 'PROBLEM.INSTANCE_SEG.DATA_CHANNEL_WEIGHTS'=[1,0.5]. "
                            "'PROBLEM.INSTANCE_SEG.DATA_CHANNELS'='BCD' 'PROBLEM.INSTANCE_SEG.DATA_CHANNEL_WEIGHTS'=[0.5,0.5,1]")
        if cfg.TEST.POST_PROCESSING.VORONOI_ON_MASK:
            if cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS not in ['BC', 'BCM', 'BCD', 'BCDv2']:
                raise ValueError("'PROBLEM.INSTANCE_SEG.DATA_CHANNELS' needs to be one between ['BC', 'BCM', 'BCD', 'BCDv2'] "
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
            raise ValueError("'PROBLEM.INSTANCE_SEG.SEED_MORPH_SEQUENCE' length and 'PROBLEM.INSTANCE_SEG.SEED_MORPH_RADIUS' length needs to be the same")
        if cfg.PROBLEM.INSTANCE_SEG.DATA_CONTOUR_MODE not in ['thick', 'inner', 'outer', 'subpixel', 'dense']:
            raise ValueError("'PROBLEM.INSTANCE_SEG.DATA_CONTOUR_MODE' must be one between ['thick', 'inner', 'outer', 'subpixel', 'dense']")
        if cfg.PROBLEM.INSTANCE_SEG.DATA_CONTOUR_MODE == 'dense' and cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS == "BCM":
            raise ValueError("'PROBLEM.INSTANCE_SEG.DATA_CONTOUR_MODE' can not be 'dense' when 'PROBLEM.INSTANCE_SEG.DATA_CHANNELS' is 'BCM'"
                " as it does not have sense")
        if cfg.MODEL.N_CLASSES > 2:
            raise ValueError("'MODEL.N_CLASSES' greater than 2 is not allowed in Instance Segmentation workflow")
        if cfg.MODEL.SOURCE == "torchvision":
            if cfg.MODEL.TORCHVISION_MODEL_NAME not in ['maskrcnn_resnet50_fpn', 'maskrcnn_resnet50_fpn_v2']:
                raise ValueError("'MODEL.SOURCE' must be one between ['maskrcnn_resnet50_fpn', 'maskrcnn_resnet50_fpn_v2']")
            if cfg.PROBLEM.NDIM == '3D':
                raise ValueError("TorchVision model's for instance segmentation are only available for 2D images")
            if cfg.TRAIN.ENABLE:
                raise NotImplementedError # require bbox generator etc.
            if cfg.TEST.ANALIZE_2D_IMGS_AS_3D_STACK:
                raise ValueError("'TEST.ANALIZE_2D_IMGS_AS_3D_STACK' can not be activated with TorchVision models for instance segmentation "
                    "workflow")
            if not cfg.TEST.STATS.FULL_IMG:
                raise ValueError("With TorchVision models for instance segmentation workflow only 'TEST.STATS.FULL_IMG' setting is available, so "
                    "please set it.")
            if cfg.TEST.STATS.PER_PATCH or cfg.TEST.STATS.MERGE_PATCHES:
                raise ValueError("With TorchVision models for instance segmentation workflow only 'TEST.STATS.FULL_IMG' setting is available, so "
                    "please enable it and disable 'TEST.STATS.PER_PATCH' and 'TEST.STATS.MERGE_PATCHES'")

    #### Detection ####
    if cfg.PROBLEM.TYPE == 'DETECTION':
        if cfg.MODEL.SOURCE == "biapy" and cfg.MODEL.N_CLASSES < 2:
            raise ValueError("'MODEL.N_CLASSES' needs to be greater or equal 2 (binary case)")
        if cfg.TEST.POST_PROCESSING.DET_WATERSHED:
            if any(len(x) != dim_count for x in cfg.TEST.POST_PROCESSING.DET_WATERSHED_FIRST_DILATION):
                raise ValueError("Each structure object defined in 'TEST.POST_PROCESSING.DET_WATERSHED_FIRST_DILATION' "
                                 "needs to be of {} dimension".format(dim_count))
            if not cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.ENABLE or \
                not cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.ENABLE:
                raise ValueError("'TEST.POST_PROCESSING.MEASURE_PROPERTIES.ENABLE' and "
                    "'TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.ENABLE' needs to be set when 'TEST.POST_PROCESSING.DET_WATERSHED' is enabled")
            if len(cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS[0]) != 0: 
                raise ValueError("'TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS' needs to be set to 'circularity' or 'sphericity' filtering "
                    "when 'TEST.POST_PROCESSING.DET_WATERSHED' is enabled")
            if cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS[0][0] not in ['circularity', 'sphericity']: 
                raise ValueError("'TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS' needs to be set to 'circularity' or 'sphericity' filtering "
                    "when 'TEST.POST_PROCESSING.DET_WATERSHED' is enabled")
        if cfg.TEST.DET_POINT_CREATION_FUNCTION not in ['peak_local_max', 'blob_log']:
            raise ValueError("'TEST.DET_POINT_CREATION_FUNCTION' must be one between: ['peak_local_max', 'blob_log']")
        if cfg.MODEL.SOURCE == "torchvision":
            if cfg.MODEL.TORCHVISION_MODEL_NAME not in ['fasterrcnn_mobilenet_v3_large_320_fpn', 'fasterrcnn_mobilenet_v3_large_fpn', \
                'fasterrcnn_resnet50_fpn', 'fasterrcnn_resnet50_fpn_v2', 'fcos_resnet50_fpn', 'ssd300_vgg16', 'ssdlite320_mobilenet_v3_large', \
                'retinanet_resnet50_fpn', 'retinanet_resnet50_fpn_v2']:
                raise ValueError("'MODEL.SOURCE' must be one between ['fasterrcnn_mobilenet_v3_large_320_fpn', 'fasterrcnn_mobilenet_v3_large_fpn', "
                    "'fasterrcnn_resnet50_fpn', 'fasterrcnn_resnet50_fpn_v2', 'fcos_resnet50_fpn', 'ssd300_vgg16', 'ssdlite320_mobilenet_v3_large', "
                    "'retinanet_resnet50_fpn', 'retinanet_resnet50_fpn_v2']")
            if cfg.PROBLEM.NDIM == '3D':
                raise ValueError("TorchVision model's for detection are only available for 2D images")
            if cfg.TRAIN.ENABLE:
                raise NotImplementedError # require bbox generator etc.
            if not cfg.TEST.STATS.FULL_IMG:
                raise ValueError("With TorchVision models for detection workflow only 'TEST.STATS.FULL_IMG' setting is available, so "
                    "please set it.")
            if cfg.TEST.STATS.PER_PATCH or cfg.TEST.STATS.MERGE_PATCHES:
                raise ValueError("With TorchVision models for detection workflow only 'TEST.STATS.FULL_IMG' setting is available, so "
                    "please enable it and disable 'TEST.STATS.PER_PATCH' and 'TEST.STATS.MERGE_PATCHES'")

    #### Super-resolution ####
    elif cfg.PROBLEM.TYPE == 'SUPER_RESOLUTION':
        if cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING == 1:
            raise ValueError("Resolution scale must be provided with 'PROBLEM.SUPER_RESOLUTION.UPSCALING' variable")
        assert cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING in [2, 4], "PROBLEM.SUPER_RESOLUTION.UPSCALING not in [2, 4]"
        if cfg.MODEL.SOURCE == "torchvision":
            raise ValueError("'MODEL.SOURCE' as 'torchvision' is not available in super-resolution workflow")

    #### Self-supervision ####
    elif cfg.PROBLEM.TYPE == 'SELF_SUPERVISED':
        if cfg.PROBLEM.SELF_SUPERVISED.PRETEXT_TASK == "crappify":
            if cfg.PROBLEM.SELF_SUPERVISED.RESIZING_FACTOR not in [2,4,6]:
                raise ValueError("PROBLEM.SELF_SUPERVISED.RESIZING_FACTOR not in [2,4,6]")
            if not check_value(cfg.PROBLEM.SELF_SUPERVISED.NOISE):
                raise ValueError("PROBLEM.SELF_SUPERVISED.NOISE not in [0, 1] range")
        elif cfg.PROBLEM.SELF_SUPERVISED.PRETEXT_TASK == "masking":
            if model_arch != 'mae':
                raise ValueError("'MODEL.ARCHITECTURE' needs to be 'mae' when 'PROBLEM.SELF_SUPERVISED.PRETEXT_TASK' is 'masking'")  
        else:
            raise ValueError("'PROBLEM.SELF_SUPERVISED.PRETEXT_TASK' needs to be among these options: ['crappify', 'masking']")
        if cfg.MODEL.SOURCE == "torchvision":
            raise ValueError("'MODEL.SOURCE' as 'torchvision' is not available in super-resolution workflow")

    #### Denoising ####
    elif cfg.PROBLEM.TYPE == 'DENOISING':
        if cfg.DATA.TEST.LOAD_GT:
            raise ValueError("Denoising is made in an unsupervised way so there is no ground truth required. Disable 'DATA.TEST.LOAD_GT'")
        if not check_value(cfg.PROBLEM.DENOISING.N2V_PERC_PIX):
            raise ValueError("PROBLEM.DENOISING.N2V_PERC_PIX not in [0, 1] range")
        if cfg.MODEL.SOURCE == "torchvision":
            raise ValueError("'MODEL.SOURCE' as 'torchvision' is not available in super-resolution workflow")

    #### Classification ####
    elif cfg.PROBLEM.TYPE == 'CLASSIFICATION':
        if cfg.TEST.BY_CHUNKS.ENABLE:
            raise ValueError("'TEST.BY_CHUNKS.ENABLE' can not be activated for CLASSIFICATION workflow")
        if cfg.MODEL.SOURCE == "torchvision":
            if cfg.MODEL.TORCHVISION_MODEL_NAME not in [
                'alexnet', 'convnext_base', 'convnext_large', 'convnext_small', 'convnext_tiny', 'densenet121', 'densenet161', \
                'densenet169', 'densenet201', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', \
                'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 'efficientnet_v2_l', 'efficientnet_v2_m', \
                'efficientnet_v2_s', 'googlenet', 'inception_v3', 'maxvit_t', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3', \
                'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small',  'quantized_googlenet', 'quantized_inception_v3', \
                'quantized_mobilenet_v2', 'quantized_mobilenet_v3_large', 'quantized_resnet18', 'quantized_resnet50', \
                'quantized_resnext101_32x8d', 'quantized_resnext101_64x4d', 'quantized_shufflenet_v2_x0_5', 'quantized_shufflenet_v2_x1_0', \
                'quantized_shufflenet_v2_x1_5', 'quantized_shufflenet_v2_x2_0', 'regnet_x_16gf', 'regnet_x_1_6gf', 'regnet_x_32gf', \
                'regnet_x_3_2gf', 'regnet_x_400mf', 'regnet_x_800mf', 'regnet_x_8gf', 'regnet_y_128gf', 'regnet_y_16gf', 'regnet_y_1_6gf', \
                'regnet_y_32gf', 'regnet_y_3_2gf', 'regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_8gf', 'resnet101', 'resnet152', \
                'resnet18', 'resnet34', 'resnet50', 'resnext101_32x8d', 'resnext101_64x4d', 'resnext50_32x4d', 'retinanet_resnet50_fpn', \
                'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', \
                'squeezenet1_0', 'squeezenet1_1', 'swin_b', 'swin_s', 'swin_t', 'swin_v2_b', 'swin_v2_s', 'swin_v2_t', \
                'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'vit_b_16', 'vit_b_32', \
                'vit_h_14', 'vit_l_16', 'vit_l_32', 'wide_resnet101_2', 'wide_resnet50_2']:
                raise ValueError("'MODEL.SOURCE' must be one between [ "
                    "'alexnet', 'convnext_base', 'convnext_large', 'convnext_small', 'convnext_tiny', 'densenet121', 'densenet161', "
                    "'densenet169', 'densenet201', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', "
                    "'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 'efficientnet_v2_l', 'efficientnet_v2_m', "
                    "'efficientnet_v2_s', 'googlenet', 'inception_v3', 'maxvit_t', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3', "
                    "'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small',  'quantized_googlenet', 'quantized_inception_v3', "
                    "'quantized_mobilenet_v2', 'quantized_mobilenet_v3_large', 'quantized_resnet18', 'quantized_resnet50', "
                    "'quantized_resnext101_32x8d', 'quantized_resnext101_64x4d', 'quantized_shufflenet_v2_x0_5', 'quantized_shufflenet_v2_x1_0', "
                    "'quantized_shufflenet_v2_x1_5', 'quantized_shufflenet_v2_x2_0', 'regnet_x_16gf', 'regnet_x_1_6gf', 'regnet_x_32gf', "
                    "'regnet_x_3_2gf', 'regnet_x_400mf', 'regnet_x_800mf', 'regnet_x_8gf', 'regnet_y_128gf', 'regnet_y_16gf', 'regnet_y_1_6gf', "
                    "'regnet_y_32gf', 'regnet_y_3_2gf', 'regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_8gf', 'resnet101', 'resnet152', "
                    "'resnet18', 'resnet34', 'resnet50', 'resnext101_32x8d', 'resnext101_64x4d', 'resnext50_32x4d', 'retinanet_resnet50_fpn', "
                    "'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', "
                    "'squeezenet1_0', 'squeezenet1_1', 'swin_b', 'swin_s', 'swin_t', 'swin_v2_b', 'swin_v2_s', 'swin_v2_t', "
                    "'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'vit_b_16', 'vit_b_32', "
                    "'vit_h_14', 'vit_l_16', 'vit_l_32', 'wide_resnet101_2', 'wide_resnet50_2' "
                    "]")
            if cfg.PROBLEM.NDIM == '3D':
                raise ValueError("TorchVision model's for classification are only available for 2D images")

    ### Pre-processing ###
    if cfg.DATA.EXTRACT_RANDOM_PATCH and cfg.DATA.PROBABILITY_MAP:
        if cfg.DATA.W_FOREGROUND+cfg.DATA.W_BACKGROUND != 1:
            raise ValueError("cfg.DATA.W_FOREGROUND+cfg.DATA.W_BACKGROUND need to sum 1. E.g. 0.94 and 0.06 respectively.")

    #### Data ####
    if cfg.TRAIN.ENABLE and check_data_paths:
        if not os.path.exists(cfg.DATA.TRAIN.PATH):
            raise ValueError("Train data dir not found: {}".format(cfg.DATA.TRAIN.PATH))
        if not os.path.exists(cfg.DATA.TRAIN.GT_PATH) and cfg.PROBLEM.TYPE not in ['DENOISING', "CLASSIFICATION", "SELF_SUPERVISED"]:
            raise ValueError("Train mask data dir not found: {}".format(cfg.DATA.TRAIN.GT_PATH))
        if not cfg.DATA.VAL.FROM_TRAIN and not cfg.DATA.VAL.IN_MEMORY:
            if not os.path.exists(cfg.DATA.VAL.PATH):
                raise ValueError("Validation data dir not found: {}".format(cfg.DATA.VAL.PATH))
            if not os.path.exists(cfg.DATA.VAL.GT_PATH) and cfg.PROBLEM.TYPE not in ['DENOISING', "CLASSIFICATION", "SELF_SUPERVISED"]:
                raise ValueError("Validation mask data dir not found: {}".format(cfg.DATA.VAL.GT_PATH))
    if cfg.TEST.ENABLE and not cfg.DATA.TEST.USE_VAL_AS_TEST and check_data_paths:
        if not os.path.exists(cfg.DATA.TEST.PATH):
            raise ValueError("Test data not found: {}".format(cfg.DATA.TEST.PATH))
        if cfg.DATA.TEST.LOAD_GT and not os.path.exists(cfg.DATA.TEST.GT_PATH) and cfg.PROBLEM.TYPE not in ["CLASSIFICATION", "SELF_SUPERVISED"]:
            raise ValueError("Test data mask not found: {}".format(cfg.DATA.TEST.GT_PATH))
    if cfg.TEST.BY_CHUNKS.ENABLE:
        if cfg.PROBLEM.NDIM == '2D':
            raise ValueError("'TEST.BY_CHUNKS' can not be activated when 'PROBLEM.NDIM' is 2D")
        assert cfg.TEST.BY_CHUNKS.FORMAT.lower() in ["h5", "zarr"], "'TEST.BY_CHUNKS.FORMAT' needs to be one between ['H5', 'Zarr']"
        opts.extend(['TEST.BY_CHUNKS.FORMAT', cfg.TEST.BY_CHUNKS.FORMAT.lower()])
        if cfg.TEST.BY_CHUNKS.WORKFLOW_PROCESS.ENABLE:
            assert cfg.TEST.BY_CHUNKS.WORKFLOW_PROCESS.TYPE in ["chunk_by_chunk", "entire_pred"], \
                "'TEST.BY_CHUNKS.WORKFLOW_PROCESS.TYPE' needs to be one between ['chunk_by_chunk', 'entire_pred']"

    if cfg.TRAIN.ENABLE:
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
                        "when DATA.NORMALIZATION.TYPE == 'custom', DATA.TRAIN.IN_MEMORY needs to be True")

    ### Model ###
    if cfg.MODEL.SOURCE == "biapy":
        assert model_arch in ['unet', 'resunet', 'resunet++', 'attention_unet', 'multiresunet', 'seunet', 'simple_cnn', 'efficientnet_b0',
            'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4','efficientnet_b5','efficientnet_b6','efficientnet_b7',
            'unetr', 'edsr', 'rcan', 'dfcan', 'wdsr', 'vit', 'mae'],\
            "MODEL.ARCHITECTURE not in ['unet', 'resunet', 'resunet++', 'attention_unet', 'multiresunet', 'seunet', 'simple_cnn', 'efficientnet_b[0-7]', 'unetr', 'edsr', 'rcan', 'dfcan', 'wdsr', 'vit', 'mae']"
        if model_arch not in ['unet', 'resunet', 'resunet++', 'seunet', 'attention_unet', 'multiresunet', 'unetr', 'vit', 'mae'] and cfg.PROBLEM.NDIM == '3D' and cfg.PROBLEM.TYPE != "CLASSIFICATION":
            raise ValueError("For 3D these models are available: {}".format(['unet', 'resunet', 'resunet++', 'seunet', 'multiresunet', 'attention_unet', 'unetr', 'vit', 'mae']))
        if cfg.MODEL.N_CLASSES > 2 and cfg.PROBLEM.TYPE != "CLASSIFICATION" and model_arch not in ['unet', 'resunet', 'resunet++', 'seunet', 'attention_unet', 'multiresunet']:
            raise ValueError("'MODEL.N_CLASSES' > 2 can only be used with 'MODEL.ARCHITECTURE' in ['unet', 'resunet', 'resunet++', 'seunet', 'attention_unet', 'multiresunet']")

        assert len(cfg.MODEL.FEATURE_MAPS) > 2, "'MODEL.FEATURE_MAPS' needs to have at least 3 values"

        # Adjust dropout to feature maps
        if model_arch in ['vit', 'unetr', 'mae']:
            if all(x == 0 for x in cfg.MODEL.DROPOUT_VALUES):
                opts.extend(['MODEL.DROPOUT_VALUES', (0.,)])
            elif len(cfg.MODEL.DROPOUT_VALUES) != 1:
                raise ValueError("'MODEL.DROPOUT_VALUES' must be list of an unique number when 'MODEL.ARCHITECTURE' is one among ['vit', 'mae', 'unetr']")
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
        if all(x == 0 for x in cfg.MODEL.Z_DOWN):
            if cfg.PROBLEM.TYPE == 'SUPER_RESOLUTION' and cfg.PROBLEM.NDIM == '3D':
                opts.extend(['MODEL.Z_DOWN', (1,)*(len(cfg.MODEL.FEATURE_MAPS)-1)])
            else:
                opts.extend(['MODEL.Z_DOWN', (2,)*(len(cfg.MODEL.FEATURE_MAPS)-1)])
        elif (cfg.PROBLEM.TYPE == 'SUPER_RESOLUTION' and cfg.PROBLEM.NDIM == '3D') and \
            any(x != 1 for x in cfg.MODEL.Z_DOWN):
            raise ValueError("'MODEL.Z_DOWN' != 1 not allowed in super-resolution workflow")
        elif any([False for x in cfg.MODEL.Z_DOWN if x != 1 and x != 2]):
            raise ValueError("'MODEL.Z_DOWN' needs to be 1 or 2")
        else:
            if model_arch == 'multiresunet' and len(cfg.MODEL.Z_DOWN) != 4:
                raise ValueError("'MODEL.Z_DOWN' length must be 4 when using 'multiresunet'")
            elif len(cfg.MODEL.FEATURE_MAPS)-1 != len(cfg.MODEL.Z_DOWN):
                raise ValueError("'MODEL.FEATURE_MAPS' length minus one and 'MODEL.Z_DOWN' length must be equal")

    if len(opts) > 0:
        cfg.merge_from_list(opts)

    if cfg.MODEL.SOURCE == "biapy":
        assert cfg.MODEL.LAST_ACTIVATION.lower() in ["relu", "tanh", "leaky_relu", "elu", "gelu", "silu", "sigmoid","softmax", "linear", "none"], \
            "Get unknown activation key {}".format(activation)

        if cfg.MODEL.UPSAMPLE_LAYER.lower() not in ["upsampling", "convtranspose"]:
            raise ValueError("cfg.MODEL.UPSAMPLE_LAYER' needs to be one between ['upsampling', 'convtranspose']. Provided {}"
                            .format(cfg.MODEL.UPSAMPLE_LAYER))
        if cfg.PROBLEM.TYPE == "SEMANTIC_SEG" and model_arch not in ['unet', 'resunet', 'resunet++', 'attention_unet', \
            'multiresunet', 'seunet', 'unetr']:
            raise ValueError("Not implemented pipeline option: semantic segmentation models are ['unet', 'resunet', 'resunet++', "
                            "'attention_unet', 'multiresunet', 'seunet', 'unetr']")
        if cfg.PROBLEM.TYPE == "INSTANCE_SEG" and model_arch not in ['unet', 'resunet', 'resunet++', 'seunet', 'attention_unet', 'unetr', 'multiresunet']:
            raise ValueError("Not implemented pipeline option: instance segmentation models are ['unet', 'resunet', 'resunet++', 'seunet', 'attention_unet', 'unetr', 'multiresunet']")
        if cfg.PROBLEM.TYPE in ['DETECTION', 'DENOISING'] and \
            model_arch not in ['unet', 'resunet', 'resunet++', 'seunet', 'attention_unet']:
            raise ValueError("Architectures available for {} are: ['unet', 'resunet', 'resunet++', 'seunet', 'attention_unet']"
                            .format(cfg.PROBLEM.TYPE))
        if cfg.PROBLEM.TYPE == 'SUPER_RESOLUTION':
            if cfg.PROBLEM.NDIM == '2D' and model_arch not in ['edsr', 'rcan', 'dfcan', 'wdsr', 'unet', 'resunet', 'resunet++', 'seunet', 'attention_unet', 'multiresunet']:
                raise ValueError("Architectures available for 2D 'SUPER_RESOLUTION' are: ['edsr', 'rcan', 'dfcan', 'wdsr', 'unet', 'resunet', 'resunet++', 'seunet', 'attention_unet', 'multiresunet']")
            elif cfg.PROBLEM.NDIM == '3D':
                if model_arch not in ['unet', 'resunet', 'resunet++', 'seunet', 'attention_unet', 'multiresunet']:
                    raise ValueError("Architectures available for 3D 'SUPER_RESOLUTION' are: ['unet', 'resunet', 'resunet++', 'seunet', 'attention_unet', 'multiresunet']")
                assert cfg.MODEL.UNET_SR_UPSAMPLE_POSITION in ["pre", "post"], "'MODEL.UNET_SR_UPSAMPLE_POSITION' not in ['pre', 'post']"
        if cfg.PROBLEM.TYPE == 'SELF_SUPERVISED':
            if model_arch not in ['unet', 'resunet', 'resunet++', 'attention_unet', 'multiresunet', 'seunet',
                'unetr', 'edsr', 'rcan', 'dfcan', 'wdsr', 'vit', 'mae']:
                raise ValueError("'SELF_SUPERVISED' models available are these: ['unet', 'resunet', 'resunet++', 'attention_unet', 'multiresunet', 'seunet', "
                    "'unetr', 'edsr', 'rcan', 'dfcan', 'wdsr', 'vit', 'mae']")
        if cfg.PROBLEM.TYPE == 'CLASSIFICATION' and model_arch not in ['simple_cnn', 'vit'] and \
            'efficientnet' not in model_arch:
            raise ValueError("Architectures available for 'CLASSIFICATION' are: ['simple_cnn', 'efficientnet_b[0-7]', 'vit']")
        if model_arch in ['unetr', 'vit', 'mae']:
            if model_arch == 'mae' and cfg.PROBLEM.TYPE != 'SELF_SUPERVISED':
                raise ValueError("'mae' model can only be used in 'SELF_SUPERVISED' workflow")
            if cfg.MODEL.VIT_EMBED_DIM % cfg.MODEL.VIT_NUM_HEADS != 0:
                raise ValueError("'MODEL.VIT_EMBED_DIM' should be divisible by 'MODEL.VIT_NUM_HEADS'")
            if not all([i == cfg.DATA.PATCH_SIZE[0] for i in cfg.DATA.PATCH_SIZE[:-1]]):
                raise ValueError("'unetr', 'vit' 'mae' models need to have same shape in all dimensions (e.g. DATA.PATCH_SIZE = (80,80,80,1) )")
        # Check that the input patch size is divisible in every level of the U-Net's like architectures, as the model
        # will throw an error not very clear for users
        if model_arch in ['unet', 'resunet', 'resunet++', 'seunet', 'attention_unet', 'multiresunet']:
            for i in range(len(cfg.MODEL.FEATURE_MAPS)-1):
                if cfg.MODEL.Z_DOWN[i] == 1 or (cfg.PROBLEM.TYPE == 'SUPER_RESOLUTION' and cfg.PROBLEM.NDIM == '3D'):
                    sizes = cfg.DATA.PATCH_SIZE[1:-1]
                else:
                    sizes = cfg.DATA.PATCH_SIZE[:-1]
                if not all([False for x in sizes if x%(np.power(2,(i+1))) != 0 or x == 0]):
                    m = "The 'DATA.PATCH_SIZE' provided is not divisible by 2 in each of the U-Net's levels. You can:\n 1) Reduce the number " + \
                            "of levels (by reducing 'cfg.MODEL.FEATURE_MAPS' array's length)\n 2) Increase 'DATA.PATCH_SIZE'"
                    if cfg.PROBLEM.NDIM == '3D':
                        m += "\n 3) If the Z axis is the problem, as the patch size is normally less than in other axis due to resolution, you " + \
                            "can tune 'MODEL.Z_DOWN' variable to not downsample the image in all U-Net levels"
                    raise ValueError(m)

    if cfg.MODEL.LOAD_CHECKPOINT and check_data_paths:
        if not os.path.exists(get_checkpoint_path(cfg, jobname)):
            raise FileNotFoundError(f"Model checkpoint not found at {get_checkpoint_path(cfg, jobname)}")

    ### Train ###
    assert cfg.TRAIN.OPTIMIZER in ['SGD', 'ADAM', 'ADAMW'], "TRAIN.OPTIMIZER not in ['SGD', 'ADAM', 'ADAMW']"
    assert cfg.LOSS.TYPE in ['CE', 'W_CE_DICE', 'MASKED_BCE'], "LOSS.TYPE not in ['CE', 'W_CE_DICE', 'MASKED_BCE']"
    if cfg.TRAIN.LR_SCHEDULER.NAME != '':
        if cfg.TRAIN.LR_SCHEDULER.NAME not in ['reduceonplateau', 'warmupcosine', 'onecycle']:
            raise ValueError("'TRAIN.LR_SCHEDULER.NAME' must be one between ['reduceonplateau', 'warmupcosine', 'onecycle']")
        if cfg.TRAIN.LR_SCHEDULER.MIN_LR == -1. and cfg.TRAIN.LR_SCHEDULER.NAME != 'onecycle':
            raise ValueError("'TRAIN.LR_SCHEDULER.MIN_LR' needs to be set when 'TRAIN.LR_SCHEDULER.NAME' is between ['reduceonplateau', 'warmupcosine']")

        if cfg.TRAIN.LR_SCHEDULER.NAME == 'reduceonplateau':
            if cfg.TRAIN.LR_SCHEDULER.REDUCEONPLATEAU_PATIENCE == -1:
                raise ValueError("'TRAIN.LR_SCHEDULER.REDUCEONPLATEAU_PATIENCE' needs to be set when 'TRAIN.LR_SCHEDULER.NAME' is 'reduceonplateau'")
            if cfg.TRAIN.LR_SCHEDULER.REDUCEONPLATEAU_PATIENCE >= cfg.TRAIN.PATIENCE:
                raise ValueError("'TRAIN.LR_SCHEDULER.REDUCEONPLATEAU_PATIENCE' needs to be less than 'TRAIN.PATIENCE' ")

        if cfg.TRAIN.LR_SCHEDULER.NAME == 'warmupcosine':
            if cfg.TRAIN.LR_SCHEDULER.WARMUP_COSINE_DECAY_EPOCHS == -1:
                raise ValueError("'TRAIN.LR_SCHEDULER.WARMUP_COSINE_DECAY_EPOCHS' needs to be set when 'TRAIN.LR_SCHEDULER.NAME' is 'warmupcosine'")

    #### Augmentation ####
    if cfg.AUGMENTOR.ENABLE:
        if not check_value(cfg.AUGMENTOR.DA_PROB):
            raise ValueError("AUGMENTOR.DA_PROB not in [0, 1] range")
        if cfg.AUGMENTOR.RANDOM_ROT:
            if not check_value(cfg.AUGMENTOR.RANDOM_ROT_RANGE, (-360,360)):
                raise ValueError("AUGMENTOR.RANDOM_ROT_RANGE values needs to be between [-360,360]")
        if cfg.AUGMENTOR.SHEAR:
            if not check_value(cfg.AUGMENTOR.SHEAR_RANGE, (-360,360)):
                raise ValueError("AUGMENTOR.SHEAR_RANGE values needs to be between [-360,360]")
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
            raise ValueError("'TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS' needs to be set when 'TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS' is True")