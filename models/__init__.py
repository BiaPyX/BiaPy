import importlib
import os
import numpy as np
from tensorflow.keras.utils import plot_model


def build_model(cfg, job_identifier):
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
    if cfg.MODEL.ARCHITECTURE in ['fcn32', 'fcn8']:
        modelname = 'fcn_vgg'
    else:
        modelname = str(cfg.MODEL.ARCHITECTURE).lower()
    mdl = importlib.import_module('models.'+modelname)
    names = [x for x in mdl.__dict__ if not x.startswith("_")]
    globals().update({k: getattr(mdl, k) for k in names})

    ndim = 3 if cfg.PROBLEM.NDIM == "3D" else 2

    # Model building
    if cfg.MODEL.ARCHITECTURE in ['unet', 'resunet', 'seunet', 'attention_unet']:
        args = dict(image_shape=cfg.DATA.PATCH_SIZE, activation=cfg.MODEL.ACTIVATION, feature_maps=cfg.MODEL.FEATURE_MAPS,
            drop_values=cfg.MODEL.DROPOUT_VALUES, spatial_dropout=cfg.MODEL.SPATIAL_DROPOUT,
            batch_norm=cfg.MODEL.BATCH_NORMALIZATION, k_init=cfg.MODEL.KERNEL_INIT, k_size=cfg.MODEL.KERNEL_SIZE,
            upsample_layer=cfg.MODEL.UPSAMPLE_LAYER, last_act=cfg.MODEL.LAST_ACTIVATION)
        if cfg.MODEL.ARCHITECTURE == 'unet':
            f_name = U_Net
        elif cfg.MODEL.ARCHITECTURE == 'resunet':
            f_name = ResUNet
        elif cfg.MODEL.ARCHITECTURE == 'attention_unet':
            f_name = Attention_U_Net
        elif cfg.MODEL.ARCHITECTURE == 'seunet':
            f_name = SE_U_Net

        args['output_channels'] = cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS if cfg.PROBLEM.TYPE == 'INSTANCE_SEG' else None
        args['n_classes'] = cfg.MODEL.N_CLASSES if cfg.PROBLEM.TYPE != 'DENOISING' else cfg.DATA.PATCH_SIZE[-1]
        if cfg.PROBLEM.NDIM == '3D':
            args['z_down'] = cfg.MODEL.Z_DOWN

        model = f_name(**args)
    else:
        if cfg.MODEL.ARCHITECTURE == 'simple_cnn':
            model = simple_CNN(image_shape=cfg.DATA.PATCH_SIZE, ndim=ndim, n_classes=cfg.MODEL.N_CLASSES)
        elif cfg.MODEL.ARCHITECTURE == 'EfficientNetB0':
            shape = (224, 224)+(cfg.DATA.PATCH_SIZE[-1],) if cfg.DATA.PATCH_SIZE[:-1] != (224, 224) else cfg.DATA.PATCH_SIZE
            model = efficientnetb0(shape, n_classes=cfg.MODEL.N_CLASSES)
        elif cfg.MODEL.ARCHITECTURE == 'ViT':
            if cfg.PROBLEM.NDIM == '3D':
                num_patches = (cfg.DATA.PATCH_SIZE[0]//cfg.MODEL.VIT_TOKEN_SIZE)**3
                transformer_units = [cfg.MODEL.VIT_HIDDEN_SIZE * 3, cfg.MODEL.VIT_HIDDEN_SIZE]  
            else:
                num_patches = (cfg.DATA.PATCH_SIZE[0]//cfg.MODEL.VIT_TOKEN_SIZE)**2
                transformer_units = [cfg.MODEL.VIT_HIDDEN_SIZE * 2, cfg.MODEL.VIT_HIDDEN_SIZE] 
            args = dict(input_shape=cfg.DATA.PATCH_SIZE, patch_size=cfg.MODEL.VIT_TOKEN_SIZE, num_patches=num_patches,
                hidden_size=cfg.MODEL.VIT_HIDDEN_SIZE, transformer_layers=cfg.MODEL.VIT_NUM_LAYERS, num_heads=cfg.MODEL.VIT_NUM_HEADS,
                transformer_units=transformer_units, mlp_head_units=cfg.MODEL.VIT_MLP_DIMS, num_classes=cfg.MODEL.N_CLASSES, 
                dropout=cfg.MODEL.DROPOUT_VALUES)
            model = ViT(**args)
        elif cfg.MODEL.ARCHITECTURE == 'fcn32':
            model = FCN32_VGG16(cfg.DATA.PATCH_SIZE, n_classes=cfg.MODEL.N_CLASSES)
        elif cfg.MODEL.ARCHITECTURE == 'fcn8':
            model = FCN8_VGG16(cfg.DATA.PATCH_SIZE, n_classes=cfg.MODEL.N_CLASSES)
        elif cfg.MODEL.ARCHITECTURE == 'tiramisu':
            model = FC_DenseNet103(cfg.DATA.PATCH_SIZE, n_filters_first_conv=cfg.MODEL.FEATURE_MAPS[0],
                n_pool=cfg.MODEL.TIRAMISU_DEPTH, growth_rate=12, n_layers_per_block=5,
                dropout_p=cfg.MODEL.DROPOUT_VALUES[0])
        elif cfg.MODEL.ARCHITECTURE == 'mnet':
            model = MNet((None, None, cfg.DATA.PATCH_SIZE[-1]))
        elif cfg.MODEL.ARCHITECTURE == 'multiresunet':
            model = MultiResUnet(None, None, cfg.DATA.PATCH_SIZE[-1])
        elif cfg.MODEL.ARCHITECTURE == 'unetr':
            if cfg.PROBLEM.NDIM == '3D':
                num_patches = (cfg.DATA.PATCH_SIZE[0]//cfg.MODEL.VIT_TOKEN_SIZE)**3
                transformer_units = [cfg.MODEL.VIT_HIDDEN_SIZE * 3, cfg.MODEL.VIT_HIDDEN_SIZE] 
            else:
                num_patches = (cfg.DATA.PATCH_SIZE[0]//cfg.MODEL.VIT_TOKEN_SIZE)**2
                transformer_units = [cfg.MODEL.VIT_HIDDEN_SIZE * 2, cfg.MODEL.VIT_HIDDEN_SIZE] 
            args = dict(input_shape=cfg.DATA.PATCH_SIZE, patch_size=cfg.MODEL.VIT_TOKEN_SIZE, num_patches=num_patches,
                hidden_size=cfg.MODEL.VIT_HIDDEN_SIZE, transformer_layers=cfg.MODEL.VIT_NUM_LAYERS, num_heads=cfg.MODEL.VIT_NUM_HEADS,
                transformer_units=transformer_units, mlp_head_units=cfg.MODEL.VIT_MLP_DIMS, 
                num_filters=cfg.MODEL.UNETR_VIT_NUM_FILTERS, num_classes=cfg.MODEL.N_CLASSES, 
                decoder_activation=cfg.MODEL.UNETR_DEC_ACTIVATION, decoder_kernel_init=cfg.MODEL.UNETR_DEC_KERNEL_INIT,
                ViT_hidd_mult=cfg.MODEL.UNETR_VIT_HIDD_MULT, batch_norm=cfg.MODEL.BATCH_NORMALIZATION, dropout=cfg.MODEL.DROPOUT_VALUES,
                last_act=cfg.MODEL.LAST_ACTIVATION)
            args['output_channels'] = cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS if cfg.PROBLEM.TYPE == 'INSTANCE_SEG' else None
            model = UNETR(**args)
        elif cfg.MODEL.ARCHITECTURE == 'edsr':
            model = EDSR(num_filters=64, num_of_residual_blocks=16, upsampling_factor=cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING, 
                num_channels=cfg.DATA.PATCH_SIZE[-1])
        elif cfg.MODEL.ARCHITECTURE == 'srunet':
            model = preResUNet(cfg.DATA.PATCH_SIZE, activation='elu', kernel_initializer='he_normal',
                dropout_value=0.2, batchnorm=cfg.MODEL.BATCH_NORMALIZATION, 
                maxpooling=True, separable=False, numInitChannels=16, depth=4, upsampling_factor=2,
                upsample_method='UpSampling2D', final_activation=None)
        elif cfg.MODEL.ARCHITECTURE == 'rcan':
            model = rcan(filters=16, n_sub_block=int(np.log2(cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING)), out_channels=1, use_mish=False)
        elif cfg.MODEL.ARCHITECTURE == 'dfcan':
            model = DFCAN(cfg.DATA.PATCH_SIZE, scale=cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING, n_ResGroup = 4, n_RCAB = 4)
        elif cfg.MODEL.ARCHITECTURE == 'wdsr':
            model = wdsr_b(scale=cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING, num_filters=32, num_res_blocks=8, res_block_expansion=6, 
                res_block_scaling=None, out_channels=1)

    # Check the network created
    model.summary(line_length=150)
    os.makedirs(cfg.PATHS.CHARTS, exist_ok=True)
    model_name = os.path.join(cfg.PATHS.CHARTS, "model_plot_" + job_identifier + ".png")
    if cfg.MODEL.MAKE_PLOT:
        plot_model(model, to_file=model_name, show_shapes=True, show_layer_names=True)

    return model
