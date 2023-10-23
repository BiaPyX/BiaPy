import importlib
import os
import torch
import numpy as np
import torch.nn as nn
from torchinfo import summary 

from utils.misc import is_main_process
from engine import prepare_optimizer
from models.blocks import get_activation

def build_model(cfg, job_identifier, device):
    """Build selected model

       Parameters
       ----------
       cfg : YACS CN object
           Configuration.

       job_identifier: str
           Job name.

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
    mdl = importlib.import_module('models.'+modelname)
    names = [x for x in mdl.__dict__ if not x.startswith("_")]
    globals().update({k: getattr(mdl, k) for k in names})

    ndim = 3 if cfg.PROBLEM.NDIM == "3D" else 2

    # Model building
    if modelname in ['unet', 'resunet', 'resunet++', 'seunet', 'attention_unet']:
        args = dict(image_shape=cfg.DATA.PATCH_SIZE, activation=cfg.MODEL.ACTIVATION.lower(), feature_maps=cfg.MODEL.FEATURE_MAPS, 
            drop_values=cfg.MODEL.DROPOUT_VALUES, batch_norm=cfg.MODEL.BATCH_NORMALIZATION, k_size=cfg.MODEL.KERNEL_SIZE,
            upsample_layer=cfg.MODEL.UPSAMPLE_LAYER, z_down=cfg.MODEL.Z_DOWN)
        if modelname == 'unet':
            f_name = U_Net
        elif modelname == 'resunet':
            f_name = ResUNet
        elif modelname == 'resunet++':
            f_name = ResUNetPlusPlus
        elif modelname == 'attention_unet':
            f_name = Attention_U_Net
        elif modelname == 'seunet':
            f_name = SE_U_Net

        args['output_channels'] = cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS if cfg.PROBLEM.TYPE == 'INSTANCE_SEG' else None        
        if cfg.PROBLEM.TYPE == 'SUPER_RESOLUTION':
            args['upsampling_factor'] = cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING
            args['upsampling_position'] = cfg.MODEL.UNET_SR_UPSAMPLE_POSITION
            args['n_classes'] = cfg.DATA.PATCH_SIZE[-1]
        else:
            args['n_classes'] = cfg.MODEL.N_CLASSES if cfg.PROBLEM.TYPE != 'DENOISING' else cfg.DATA.PATCH_SIZE[-1]
        model = f_name(**args)
    else:
        if modelname == 'simple_cnn':
            model = simple_CNN(image_shape=cfg.DATA.PATCH_SIZE, activation=cfg.MODEL.ACTIVATION.lower(), n_classes=cfg.MODEL.N_CLASSES)
        elif 'efficientnet' in modelname:
            shape = (224, 224)+(cfg.DATA.PATCH_SIZE[-1],) if cfg.DATA.PATCH_SIZE[:-1] != (224, 224) else cfg.DATA.PATCH_SIZE
            model = efficientnet(cfg.MODEL.ARCHITECTURE.lower(), shape, n_classes=cfg.MODEL.N_CLASSES)
        elif modelname == 'vit':
            args = dict(img_size=cfg.DATA.PATCH_SIZE[0], patch_size=cfg.MODEL.VIT_TOKEN_SIZE, in_chans=cfg.DATA.PATCH_SIZE[-1],  
                ndim=ndim, num_classes=cfg.MODEL.N_CLASSES, norm_layer=partial(nn.LayerNorm, eps=1e-6))
            if cfg.MODEL.VIT_MODEL == "custom":
                args2 = dict(embed_dim=cfg.MODEL.VIT_EMBED_DIM, depth=cfg.MODEL.VIT_NUM_LAYERS, num_heads=cfg.MODEL.VIT_NUM_HEADS, 
                    mlp_ratio=cfg.MODEL.VIT_MLP_RATIO, drop_rate=cfg.MODEL.DROPOUT_VALUES[0])
                args.update(args2)
                model = VisionTransformer(**args)
            else:
                model = eval(cfg.MODEL.VIT_MODEL)(**args)
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
        elif modelname == 'unetr':
            args = dict(input_shape=cfg.DATA.PATCH_SIZE, patch_size=cfg.MODEL.VIT_TOKEN_SIZE, embed_dim=cfg.MODEL.VIT_EMBED_DIM,
                depth=cfg.MODEL.VIT_NUM_LAYERS, transformer_layers=cfg.MODEL.VIT_NUM_LAYERS, num_heads=cfg.MODEL.VIT_NUM_HEADS, 
                mlp_ratio=cfg.MODEL.VIT_MLP_RATIO, num_filters=cfg.MODEL.UNETR_VIT_NUM_FILTERS, n_classes=cfg.MODEL.N_CLASSES, 
                decoder_activation=cfg.MODEL.UNETR_DEC_ACTIVATION, ViT_hidd_mult=cfg.MODEL.UNETR_VIT_HIDD_MULT, 
                batch_norm=cfg.MODEL.BATCH_NORMALIZATION, dropout=cfg.MODEL.DROPOUT_VALUES[0])
            args['output_channels'] = cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS if cfg.PROBLEM.TYPE == 'INSTANCE_SEG' else None
            model = UNETR(**args)
        elif modelname == 'edsr':
            model = EDSR(ndim=ndim, num_filters=64, num_of_residual_blocks=16, upsampling_factor=cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING, 
                num_channels=cfg.DATA.PATCH_SIZE[-1])
        elif modelname == 'rcan':
            model = rcan(ndim=ndim, filters=16, n_sub_block=int(np.log2(cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING)), num_channels=cfg.DATA.PATCH_SIZE[-1])
        elif modelname == 'dfcan':
            model = DFCAN(ndim=ndim, input_shape=cfg.DATA.PATCH_SIZE, scale=cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING, n_ResGroup = 4, n_RCAB = 4)
        elif modelname == 'wdsr':
            model = wdsr(scale=cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING, num_filters=32, num_res_blocks=8, res_block_expansion=6, 
                num_channels=cfg.DATA.PATCH_SIZE[-1])
        elif modelname == 'mae':
            model = MaskedAutoencoderViT(
                img_size=cfg.DATA.PATCH_SIZE[0], patch_size=cfg.MODEL.VIT_TOKEN_SIZE, in_chans=cfg.DATA.PATCH_SIZE[-1],  
                ndim=ndim, norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_dim=cfg.MODEL.VIT_EMBED_DIM, 
                depth=cfg.MODEL.VIT_NUM_LAYERS, num_heads=cfg.MODEL.VIT_NUM_HEADS, decoder_embed_dim=512, decoder_depth=8, 
                decoder_num_heads=16, mlp_ratio=cfg.MODEL.VIT_MLP_RATIO, mask_ratio=cfg.MODEL.MAE_MASK_RATIO)
                 
    # Check the network created
    model.to(device)
    if cfg.PROBLEM.NDIM == '2D':
        sample_size = (1,cfg.DATA.PATCH_SIZE[2], cfg.DATA.PATCH_SIZE[0], cfg.DATA.PATCH_SIZE[1])
    else:
        sample_size = (1,cfg.DATA.PATCH_SIZE[3], cfg.DATA.PATCH_SIZE[0], cfg.DATA.PATCH_SIZE[1], cfg.DATA.PATCH_SIZE[2])
    summary(model, input_size=sample_size, col_names=("input_size", "output_size", "num_params"), depth=10,
        device="cpu" if "cuda" not in device.type else "cuda")
    return model
