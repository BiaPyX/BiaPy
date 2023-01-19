import importlib
import os
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
    elif cfg.MODEL.ARCHITECTURE in ['tiramisu', 'nnunet', 'mnet', 'multiresunet', 'simple_cnn', 'EfficientNetB0']:
        modelname = cfg.MODEL.ARCHITECTURE
    else:
        modelname = cfg.MODEL.ARCHITECTURE if cfg.PROBLEM.NDIM == '2D' else cfg.MODEL.ARCHITECTURE + '_3d'
    if cfg.PROBLEM.TYPE == 'INSTANCE_SEG': modelname = modelname+"_instances"
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
            f_name = U_Net_3D if cfg.PROBLEM.NDIM == '3D' else U_Net_2D
        elif cfg.MODEL.ARCHITECTURE == 'resunet':
            f_name = ResUNet_3D if cfg.PROBLEM.NDIM == '3D' else ResUNet_2D
        elif cfg.MODEL.ARCHITECTURE == 'attention_unet':
            f_name = Attention_U_Net_3D if cfg.PROBLEM.NDIM == '3D' else Attention_U_Net_2D
        elif cfg.MODEL.ARCHITECTURE == 'seunet':
            f_name = SE_U_Net_3D if cfg.PROBLEM.NDIM == '3D' else SE_U_Net_2D

        if cfg.PROBLEM.TYPE == 'INSTANCE_SEG':
            args['output_channels'] = cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS
            del args['last_act']
        else:
            args['n_classes'] = cfg.MODEL.N_CLASSES
        if cfg.PROBLEM.NDIM == '3D':
            args['z_down'] = cfg.MODEL.Z_DOWN

        model = f_name(**args)
    else:
        if cfg.MODEL.ARCHITECTURE == 'simple_cnn':
            model = simple_CNN(image_shape=cfg.DATA.PATCH_SIZE, ndim=ndim, n_classes=cfg.MODEL.N_CLASSES)
        elif cfg.MODEL.ARCHITECTURE == 'EfficientNetB0':
            shape = (224, 224)+(cfg.DATA.PATCH_SIZE[-1],) if cfg.DATA.PATCH_SIZE[:-1] != (224, 224) else cfg.DATA.PATCH_SIZE
            model = efficientnetb0(shape, n_classes=cfg.MODEL.N_CLASSES)
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
            num_patches = (cfg.DATA.PATCH_SIZE[0]//cfg.MODEL.UNETR_TOKEN_SIZE)**2
            args = dict(input_shape=cfg.DATA.PATCH_SIZE, patch_size=cfg.MODEL.UNETR_TOKEN_SIZE, num_patches=num_patches,
                projection_dim=cfg.MODEL.UNETR_EMBED_DIM, transformer_layers=cfg.MODEL.UNETR_DEPTH, num_heads=cfg.MODEL.UNETR_NUM_HEADS,
                transformer_units=cfg.MODEL.UNETR_MLP_HIDDEN_UNITS, data_augmentation = None, num_filters = 16, 
                num_classes=cfg.MODEL.N_CLASSES, decoder_activation = 'relu', decoder_kernel_init = 'he_normal',
                ViT_hidd_mult = cfg.MODEL.UNETR_VIT_HIDD_MULT, batch_norm=cfg.MODEL.BATCH_NORMALIZATION, dropout=cfg.MODEL.DROPOUT_VALUES)
            model = UNETR_2D(**args)
        elif cfg.MODEL.ARCHITECTURE == 'edsr':
            model = EDSR(num_filters=64, num_of_residual_blocks=16, upsampling_factor=cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING, 
                num_channels=cfg.DATA.PATCH_SIZE[-1])

    # Check the network created
    model.summary(line_length=150)
    os.makedirs(cfg.PATHS.CHARTS, exist_ok=True)
    model_name = os.path.join(cfg.PATHS.CHARTS, "model_plot_" + job_identifier + ".png")
    plot_model(model, to_file=model_name, show_shapes=True, show_layer_names=True)

    return model
