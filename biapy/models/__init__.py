import importlib
import os
import json
from pathlib import Path
import pooch
import yaml
import torch
import numpy as np
import torch.nn as nn
from torchinfo import summary 
from marshmallow import missing
from bioimageio.spec.utils import download
from bioimageio.core.model_adapters._pytorch_model_adapter import PytorchModelAdapter
from bioimageio.spec._internal.io import RelativePath

from biapy.utils.misc import is_main_process
from biapy.engine import prepare_optimizer
from biapy.models.blocks import get_activation

def build_model(cfg, job_identifier, device):
    """
    Build selected model

    Parameters
    ----------
    cfg : YACS CN object
        Configuration.

    job_identifier: str
        Job name.

    device : Torch device
        Using device. Most commonly "cpu" or "cuda" for GPU, but also potentially "mps", 
        "xpu", "xla" or "meta". 
    Returns
    -------
    model : Keras model
        Selected model.
    """
    # Import the model
    if 'efficientnet' in cfg.MODEL.ARCHITECTURE.lower():
        modelname = 'efficientnet'
    else:
        modelname = str(cfg.MODEL.ARCHITECTURE).lower()
    mdl = importlib.import_module('biapy.models.'+modelname)
    model_file = os.path.abspath(mdl.__file__)
    names = [x for x in mdl.__dict__ if not x.startswith("_")]
    globals().update({k: getattr(mdl, k) for k in names})

    ndim = 3 if cfg.PROBLEM.NDIM == "3D" else 2

    # Model building
    if modelname in ['unet', 'resunet', 'resunet++', 'seunet', 'resunet_se', 'attention_unet', 'unext_v1']:
        args = dict(image_shape=cfg.DATA.PATCH_SIZE, activation=cfg.MODEL.ACTIVATION.lower(), feature_maps=cfg.MODEL.FEATURE_MAPS, 
            drop_values=cfg.MODEL.DROPOUT_VALUES, normalization=cfg.MODEL.NORMALIZATION, k_size=cfg.MODEL.KERNEL_SIZE,
            upsample_layer=cfg.MODEL.UPSAMPLE_LAYER, z_down=cfg.MODEL.Z_DOWN)
        if modelname == 'unet':
            callable_model = U_Net
        elif modelname == 'resunet':
            callable_model = ResUNet
            args['isotropy'] = cfg.MODEL.ISOTROPY
            args['larger_io'] = cfg.MODEL.LARGER_IO
        elif modelname == 'resunet++':
            callable_model = ResUNetPlusPlus
        elif modelname == 'attention_unet':
            callable_model = Attention_U_Net
        elif modelname == 'seunet':
            callable_model = SE_U_Net
            args['isotropy'] = cfg.MODEL.ISOTROPY
            args['larger_io'] = cfg.MODEL.LARGER_IO
        elif modelname == 'resunet_se':
            callable_model = ResUNet_SE
            args['isotropy'] = cfg.MODEL.ISOTROPY
            args['larger_io'] = cfg.MODEL.LARGER_IO
        elif modelname == 'unext_v1':
            args = dict(image_shape=cfg.DATA.PATCH_SIZE, feature_maps=cfg.MODEL.FEATURE_MAPS, 
                upsample_layer=cfg.MODEL.UPSAMPLE_LAYER, z_down=cfg.MODEL.Z_DOWN, cn_layers=cfg.MODEL.CONVNEXT_LAYERS,
                layer_scale=cfg.MODEL.CONVNEXT_LAYER_SCALE, stochastic_depth_prob=cfg.MODEL.CONVNEXT_SD_PROB)
            callable_model = U_NeXt_V1
        args['output_channels'] = cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS if cfg.PROBLEM.TYPE == 'INSTANCE_SEG' else None        
        if cfg.PROBLEM.TYPE == 'SUPER_RESOLUTION':
            args['upsampling_factor'] = cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING
            args['upsampling_position'] = cfg.MODEL.UNET_SR_UPSAMPLE_POSITION
            args['n_classes'] = cfg.DATA.PATCH_SIZE[-1]
        else:
            args['n_classes'] = cfg.MODEL.N_CLASSES if cfg.PROBLEM.TYPE != 'DENOISING' else cfg.DATA.PATCH_SIZE[-1]
        model = callable_model(**args)
    else:
        if modelname == 'simple_cnn':
            args = dict(image_shape=cfg.DATA.PATCH_SIZE, activation=cfg.MODEL.ACTIVATION.lower(), n_classes=cfg.MODEL.N_CLASSES)
            model = simple_CNN(**args)
            callable_model = simple_CNN
        elif 'efficientnet' in modelname:
            shape = (224, 224)+(cfg.DATA.PATCH_SIZE[-1],) if cfg.DATA.PATCH_SIZE[:-1] != (224, 224) else cfg.DATA.PATCH_SIZE
            args = dict(cfg.MODEL.ARCHITECTURE.lower(), shape, n_classes=cfg.MODEL.N_CLASSES)
            model = efficientnet(**args)
            callable_model = efficientnet
        elif modelname == 'vit':
            args = dict(img_size=cfg.DATA.PATCH_SIZE[0], patch_size=cfg.MODEL.VIT_TOKEN_SIZE, in_chans=cfg.DATA.PATCH_SIZE[-1],  
                ndim=ndim, num_classes=cfg.MODEL.N_CLASSES, norm_layer=partial(nn.LayerNorm, eps=1e-6))
            if cfg.MODEL.VIT_MODEL == "custom":
                args2 = dict(embed_dim=cfg.MODEL.VIT_EMBED_DIM, depth=cfg.MODEL.VIT_NUM_LAYERS, num_heads=cfg.MODEL.VIT_NUM_HEADS, 
                    mlp_ratio=cfg.MODEL.VIT_MLP_RATIO, drop_rate=cfg.MODEL.DROPOUT_VALUES[0])
                args.update(args2)
                model = VisionTransformer(**args)
                callable_model = VisionTransformer
            else:
                model = eval(cfg.MODEL.VIT_MODEL)(**args)
                callable_model = eval(cfg.MODEL.VIT_MODEL)
        elif modelname == 'multiresunet':
            args = dict(input_channels=cfg.DATA.PATCH_SIZE[-1], ndim=ndim, alpha=1.67, z_down=cfg.MODEL.Z_DOWN)
            args['output_channels'] = cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS if cfg.PROBLEM.TYPE == 'INSTANCE_SEG' else None
            if cfg.PROBLEM.TYPE == 'SUPER_RESOLUTION':
                args['upsampling_factor'] = cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING
                args['upsampling_position'] = cfg.MODEL.UNET_SR_UPSAMPLE_POSITION
                args['n_classes'] = cfg.DATA.PATCH_SIZE[-1]
            else:
                args['n_classes'] = cfg.MODEL.N_CLASSES if cfg.PROBLEM.TYPE != 'DENOISING' else cfg.DATA.PATCH_SIZE[-1]
            model = MultiResUnet(**args)
            callable_model = MultiResUnet
        elif modelname == 'unetr':
            args = dict(input_shape=cfg.DATA.PATCH_SIZE, patch_size=cfg.MODEL.VIT_TOKEN_SIZE, embed_dim=cfg.MODEL.VIT_EMBED_DIM,
                depth=cfg.MODEL.VIT_NUM_LAYERS, num_heads=cfg.MODEL.VIT_NUM_HEADS, mlp_ratio=cfg.MODEL.VIT_MLP_RATIO, 
                num_filters=cfg.MODEL.UNETR_VIT_NUM_FILTERS, n_classes=cfg.MODEL.N_CLASSES, 
                decoder_activation=cfg.MODEL.UNETR_DEC_ACTIVATION, ViT_hidd_mult=cfg.MODEL.UNETR_VIT_HIDD_MULT, 
                normalization=cfg.MODEL.NORMALIZATION, dropout=cfg.MODEL.DROPOUT_VALUES[0], k_size=cfg.MODEL.UNETR_DEC_KERNEL_SIZE)
            args['output_channels'] = cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS if cfg.PROBLEM.TYPE == 'INSTANCE_SEG' else None
            model = UNETR(**args)
            callable_model = UNETR
        elif modelname == 'edsr':
            args = dict(ndim=ndim, num_filters=64, num_of_residual_blocks=16, upsampling_factor=cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING, 
                num_channels=cfg.DATA.PATCH_SIZE[-1])
            model = EDSR(args)
            callable_model = EDSR
        elif modelname == 'rcan':
            scale = cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING
            if type(scale) is tuple:
                scale = scale[0]
            args = dict(ndim=ndim, filters=16, scale=scale, n_sub_block=int(np.log2(scale)), num_channels=cfg.DATA.PATCH_SIZE[-1])
            model = rcan(**args)
            callable_model = rcan
        elif modelname == 'dfcan':
            args = dict(ndim=ndim, input_shape=cfg.DATA.PATCH_SIZE, scale=cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING, n_ResGroup = 4, n_RCAB = 4)
            model = DFCAN(**args)
            callable_model = DFCAN
        elif modelname == 'wdsr':
            args = dict(scale=cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING, num_filters=32, num_res_blocks=8, res_block_expansion=6, 
                num_channels=cfg.DATA.PATCH_SIZE[-1])
            model = wdsr(**args)
            callable_model = wdsr
        elif modelname == 'mae':
            args = dict(img_size=cfg.DATA.PATCH_SIZE[0], patch_size=cfg.MODEL.VIT_TOKEN_SIZE, in_chans=cfg.DATA.PATCH_SIZE[-1],  
                ndim=ndim, norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_dim=cfg.MODEL.VIT_EMBED_DIM, 
                depth=cfg.MODEL.VIT_NUM_LAYERS, num_heads=cfg.MODEL.VIT_NUM_HEADS, decoder_embed_dim=512, decoder_depth=8, 
                decoder_num_heads=16, mlp_ratio=cfg.MODEL.VIT_MLP_RATIO, masking_type=cfg.MODEL.MAE_MASK_TYPE, 
                mask_ratio=cfg.MODEL.MAE_MASK_RATIO, device=device)
            model = MaskedAutoencoderViT(**args)
            callable_model = MaskedAutoencoderViT
    # Check the network created
    model.to(device)
    if cfg.PROBLEM.NDIM == '2D':
        sample_size = (1,cfg.DATA.PATCH_SIZE[2], cfg.DATA.PATCH_SIZE[0], cfg.DATA.PATCH_SIZE[1])
    else:
        sample_size = (1,cfg.DATA.PATCH_SIZE[3], cfg.DATA.PATCH_SIZE[0], cfg.DATA.PATCH_SIZE[1], cfg.DATA.PATCH_SIZE[2])
    summary(model, input_size=sample_size, col_names=("input_size", "output_size", "num_params"), depth=10, device=device.type)
    
    model_file += ":"+str(callable_model.__name__)
    model_name = model_file.rsplit(':', 1)[-1]
    return model, model_file, model_name, args

def build_bmz_model(cfg, model, device):
    model_instance = PytorchModelAdapter.get_network(model.weights.pytorch_state_dict)
    model_instance = model_instance.to(device)
    state = torch.load(download(model.weights.pytorch_state_dict).path, map_location=device)
    model_instance.load_state_dict(state)

    # Check the network created
    if cfg.PROBLEM.NDIM == '2D':
        sample_size = (1,cfg.DATA.PATCH_SIZE[2], cfg.DATA.PATCH_SIZE[0], cfg.DATA.PATCH_SIZE[1])
    else:
        sample_size = (1,cfg.DATA.PATCH_SIZE[3], cfg.DATA.PATCH_SIZE[0], cfg.DATA.PATCH_SIZE[1], cfg.DATA.PATCH_SIZE[2])
    summary(model_instance, input_size=sample_size, col_names=("input_size", "output_size", "num_params"), depth=10, device=device.type)    
    return model_instance

def get_bmz_model_info(model, spec_version="v0_4"):
    """
    Gather model info depending on its spec version.
    """
    assert model.weights.pytorch_state_dict is not None, \
        "Seems that the original BMZ model has no pytorch_state_dict object. Aborting"
    
    if spec_version == "v0_5":
        arch = model.weights.pytorch_state_dict.architecture
        if isinstance(arch, ArchitectureFromFileDescr):
            arch_file_path = download(arch.source, sha256=arch.sha256).path
            arch_file_sha256 = arch.sha256
            arch_name = arch.callable
            arch_kwargs = arch.kwargs

            pytorch_architecture = ArchitectureFromFileDescr(
                source=arch_file_path,
                sha256=arch_file_sha256,
                callable=arch_name,
                kwargs=arch_kwargs
            )
        else:
            # For a model architecture that is published in a Python package
            # Make sure to include the Python library referenced in `import_from` in the weights entry's `dependencies`
            pytorch_architecture = ArchitectureFromLibraryDescr(
                callable=arch.callable,
                kwargs=arch.kwargs,
                import_from=arch.import_from,
            )
        state_dict_source = model.weights.pytorch_state_dict.source
        state_dict_sha256 = model.weights.pytorch_state_dict.sha256
    else: #v0_4
        weights = model.weights.pytorch_state_dict
        arch_file_sha256 = weights.architecture_sha256
        
        arch_file_path = download(RelativePath("unet.py"), sha256=arch_file_sha256).path
        # arch_file_path = download(weights.architecture, sha256=arch_file_sha256).path
        arch_name = weights.architecture # 'unet.py:UNet2d'
        pytorch_architecture = ArchitectureFromFileDescr(
            source=arch_file_path,
            sha256=arch_file_sha256,
            callable=arch_name,
            kwargs=weights.kwargs,
        )
        state_dict_source = model.weights.pytorch_state_dict.source
        state_dict_sha256 = model.weights.pytorch_state_dict.sha256

    return state_dict_source, state_dict_sha256, pytorch_architecture

def check_bmz_model_compatibility(cfg):
    # Checking BMZ model compatibility using the available model list provided by BMZ
    # COLLECTION_URL = "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/collection.json"
    COLLECTION_URL = "https://raw.githubusercontent.com/bioimage-io/collection-bioimage-io/gh-pages/collection.json"
    collection_path = Path(pooch.retrieve(COLLECTION_URL, known_hash=None))
    with collection_path.open() as f:
        collection = json.load(f)

    # Find the model among all 
    model_doi = cfg.MODEL.BMZ.SOURCE_MODEL_DOI
    model_urls = [entry["rdf_source"] for entry in collection["collection"] if entry["type"] == "model" and model_doi in entry["rdf_source"]]
    if len(model_urls) == 0:
        raise ValueError(f"No model found with the provided DOI: {model_doi}")
    if len(model_urls) > 1:
        raise ValueError("More than one model found with the provided DOI. Contact BiaPy team.")
    with open(Path(pooch.retrieve(model_urls[0], known_hash=None))) as stream:
        try:
            model_rdf = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # Check axes, preprocessing functions used and postprocessing to ensure BiaPy can handle it
    implemented = True
    reason_message = ""
    preproc_names = []
    if "pytorch_state_dict" in model_rdf["weights"] and len(model_rdf['inputs']) == 1:
        # Check problem type 
        model_problem_match = False
        if cfg.PROBLEM.TYPE == 'SEMANTIC_SEG' and \
            ('semantic-segmentation' in model_rdf['tags'] or 'segmentation' in model_rdf['tags']): 
            model_problem_match = True

            # Check number of classes
            classes = -1
            if 'kwargs' in model_rdf['weights']['pytorch_state_dict']:
                if 'out_channels' in model_rdf['weights']['pytorch_state_dict']['kwargs']:
                    classes = model_rdf['weights']['pytorch_state_dict']['kwargs']['out_channels']
                elif 'classes' in model_rdf['weights']['pytorch_state_dict']['kwargs']:
                    classes = model_rdf['weights']['pytorch_state_dict']['kwargs']['classes']
                elif 'n_classes' in model_rdf['weights']['pytorch_state_dict']['kwargs']:
                    classes = model_rdf['weights']['pytorch_state_dict']['kwargs']['n_classes']
            if classes != -1:
                if classes > 2 and cfg.MODEL.N_CLASSES != classes:
                    raise ValueError("'MODEL.N_CLASSES' does not match network's output classes. Please check it!")
                else:
                    print("BiaPy works only with one channel for the binary case (classes <= 2) so will adapt the output to match "
                        "the ground truth data")
            else:
                print("WARNING: couldn't find the classes this model is returning so please be aware to match it")

        elif cfg.PROBLEM.TYPE == 'INSTANCE_SEG' and 'instance-segmentation' in model_rdf['tags']:
            model_problem_match = True
        elif cfg.PROBLEM.TYPE == 'DETECTION' and 'detection' in model_rdf['tags']:
            model_problem_match = True
        elif cfg.PROBLEM.TYPE == 'DENOISING' and 'denoising' in model_rdf['tags']:
            model_problem_match = True
        elif cfg.PROBLEM.TYPE == 'SUPER_RESOLUTION' and 'super-resolution' in model_rdf['tags']:
            model_problem_match = True
        elif cfg.PROBLEM.TYPE == 'SELF_SUPERVISED' and 'self-supervision' in model_rdf['tags']:
            model_problem_match = True
        elif cfg.PROBLEM.TYPE == 'CLASSIFICATION' and 'classification' in model_rdf['tags']:
            model_problem_match = True
        elif cfg.PROBLEM.TYPE == 'IMAGE_TO_IMAGE' and \
            ('pix2pix' in model_rdf['tags'] or "image-reconstruction" in model_rdf['tags'] or \
                "image-to-image" in model_rdf['tags'] or 'image-restoration' in model_rdf['tags']):
            model_problem_match = True

        if not model_problem_match:
            implemented = False
            reason_message = f"Selected model is not for {cfg.PROBLEM.TYPE} problem"

        # Check axes and dimension
        if cfg.PROBLEM.NDIM == '2D':
            if model_rdf['inputs'][0]['axes'] != "bcyx":
                implemented = False 
                reason_message = f"In a 2D problem the axes need to be 'bcyx', found {model_rdf['inputs'][0]['axes']}"
            elif "2d" not in model_rdf['tags']:
                implemented = False 
                reason_message = f"Selected model seems to not be 2D"
        elif cfg.PROBLEM.NDIM == '3D':
            if model_rdf['inputs'][0]['axes'] != "bczyx":
                implemented = False 
                reason_message = f"In a 3D problem the axes need to be 'bczyx', found {model_rdf['inputs'][0]['axes']}"
            elif "3d" not in model_rdf['tags']:
                implemented = False 
                reason_message = f"Selected model seems to not be 3D"

        # Check preprocessing
        if implemented and 'preprocessing' in model_rdf['inputs'][0]:
            for preprocs in model_rdf['inputs'][0]['preprocessing']:
                if preprocs['name'] not in ["zero_mean_unit_variance", "scale_range"]: # TODO: scale_linear
                    implemented = False
                    reason_message = f"Not recognized preprocessing found: {preprocs['name']}"
                    break
                preproc_names = preprocs

        # Check post-processing
        if implemented and 'postprocessing' in model_rdf['weights']['pytorch_state_dict']['kwargs'] and \
            model_rdf['weights']['pytorch_state_dict']['kwargs']['postprocessing'] is not None:
            implemented = False
            reason_message = f"Currently no postprocessing is supported. Found: {model_rdf['weights']['pytorch_state_dict']['kwargs']['postprocessing']}"
    else:
        implemented = False
        reason_message = "pytorch_state_dict not found in model RDF"

    if not implemented:
        raise ValueError(f"Model {model_doi} can not be used in BiaPy. Message: {reason_message}")

    return preproc_names

def build_torchvision_model(cfg, device):
    # Find model in TorchVision
    if 'quantized_' in cfg.MODEL.TORCHVISION_MODEL_NAME:
        mdl = importlib.import_module('torchvision.models.quantization', cfg.MODEL.TORCHVISION_MODEL_NAME)
        w_prefix = "_quantizedweights"
        tc_model_name = cfg.MODEL.TORCHVISION_MODEL_NAME.replace('quantized_','')
        mdl_weigths = importlib.import_module('torchvision.models', cfg.MODEL.TORCHVISION_MODEL_NAME)
    else:
        w_prefix = "_weights"
        tc_model_name = cfg.MODEL.TORCHVISION_MODEL_NAME
        if cfg.PROBLEM.TYPE == 'CLASSIFICATION':
            mdl = importlib.import_module('torchvision.models', cfg.MODEL.TORCHVISION_MODEL_NAME)
        elif cfg.PROBLEM.TYPE == 'SEMANTIC_SEG':
            mdl = importlib.import_module('torchvision.models.segmentation', cfg.MODEL.TORCHVISION_MODEL_NAME)
        elif cfg.PROBLEM.TYPE in ['INSTANCE_SEG', 'DETECTION']:
            mdl = importlib.import_module('torchvision.models.detection', cfg.MODEL.TORCHVISION_MODEL_NAME)
        mdl_weigths = mdl

    # Import model and weights
    names = [x for x in mdl.__dict__ if not x.startswith("_")]
    for weight_name in names:
        if tc_model_name+w_prefix in weight_name.lower():
            break 
    weight_name = weight_name.replace('Quantized','')     
    print(f"Pytorch model selected: {tc_model_name} (weights: {weight_name})")
    globals().update(
        {
            tc_model_name: getattr(mdl, tc_model_name), 
            weight_name: getattr(mdl_weigths, weight_name)
        })

    # Load model and weights 
    model_torchvision_weights = eval(weight_name).DEFAULT
    args = {}
    model = eval(tc_model_name)(weights=model_torchvision_weights)

    # Create new head
    sample_size = None
    out_classes = cfg.MODEL.N_CLASSES if cfg.MODEL.N_CLASSES > 2 else 1
    if cfg.PROBLEM.TYPE == 'CLASSIFICATION':
        if cfg.MODEL.N_CLASSES != 1000: # 1000 classes are the ones by default in ImageNet, which are the weights loaded by default
            print(f"WARNING: Model's head changed from 1000 to {out_classes} so a finetunning is required to have good results")
            if cfg.MODEL.TORCHVISION_MODEL_NAME in ['squeezenet1_0', 'squeezenet1_1']:
                head = torch.nn.Conv2d(model.classifier[1].in_channels, out_classes, kernel_size=1, stride=1)
                model.classifier[1] = head
            else:
                if hasattr(model, 'fc'):
                    layer = "fc"
                elif hasattr(model, 'classifier'):
                    layer = 'classifier'
                else: 
                    layer = "head"
                if isinstance(getattr(model, layer), list) or isinstance(getattr(model, layer), torch.nn.modules.container.Sequential):
                    head = torch.nn.Linear(getattr(model, layer)[-1].in_features, out_classes, bias=True)
                    getattr(model, layer)[-1] = head
                else:
                    head = torch.nn.Linear(getattr(model, layer).in_features, out_classes, bias=True)
                    setattr(model, layer, head)
            
            # Fix sample input shape as required by some models
            if cfg.MODEL.TORCHVISION_MODEL_NAME in ['maxvit_t']:
                sample_size = (1, 3, 224, 224)
    elif cfg.PROBLEM.TYPE == 'SEMANTIC_SEG':        
        head = torch.nn.Conv2d(model.classifier[-1].in_channels, out_classes, kernel_size=1, stride=1)
        model.classifier[-1] = head
        head = torch.nn.Conv2d(model.aux_classifier[-1].in_channels, out_classes, kernel_size=1, stride=1)
        model.aux_classifier[-1] = head
    elif cfg.PROBLEM.TYPE == 'INSTANCE_SEG':
        # MaskRCNN
        if cfg.MODEL.N_CLASSES != 91: # 91 classes are the ones by default in MaskRCNN
            cls_score = torch.nn.Linear(in_features=1024, out_features=out_classes, bias=True)
            model.roi_heads.box_predictor.cls_score = cls_score
            mask_fcn_logits = torch.nn.Conv2d(model.roi_heads.mask_predictor.mask_fcn_logits.in_channels, out_classes, kernel_size=1, stride=1)
            model.roi_heads.mask_predictor.mask_fcn_logits = mask_fcn_logits
            print(f"Model's head changed from 91 to {out_classes} so a finetunning is required")

    # Check the network created
    model.to(device)
    if sample_size is None:
        if cfg.PROBLEM.NDIM == '2D':
            sample_size = (1,cfg.DATA.PATCH_SIZE[2], cfg.DATA.PATCH_SIZE[0], cfg.DATA.PATCH_SIZE[1])
        else:
            sample_size = (1,cfg.DATA.PATCH_SIZE[3], cfg.DATA.PATCH_SIZE[0], cfg.DATA.PATCH_SIZE[1], cfg.DATA.PATCH_SIZE[2])

    summary(model, input_size=sample_size, col_names=("input_size", "output_size", "num_params"), depth=10, device=device.type)

    return model, model_torchvision_weights.transforms()
