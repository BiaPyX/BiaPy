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

    assert cfg.MODEL.ARCHITECTURE in ['unet', 'resunet', 'attention_unet', 'fcn32', 'fcn8', 'nnunet', 'tiramisu', 'mnet',
                                      'multiresunet', 'seunet']

    if cfg.PROBLEM.TYPE == 'INSTANCE_SEG' and cfg.MODEL.ARCHITECTURE != 'unet' and cfg.MODEL.ARCHITECTURE != 'resunet':
        raise ValueError("Not implemented pipeline option: instance segmentation models adapted are 'unet' or 'resunet'")

    # Import the model
    if cfg.MODEL.ARCHITECTURE == 'fcn32' or cfg.MODEL.ARCHITECTURE == 'fcn8':
        modelname = 'fcn_vgg'
    elif cfg.MODEL.ARCHITECTURE == 'tiramisu' or cfg.MODEL.ARCHITECTURE == 'nnunet' or cfg.MODEL.ARCHITECTURE == 'mnet'\
         or cfg.MODEL.ARCHITECTURE == 'multiresunet':
        modelname = cfg.MODEL.ARCHITECTURE
    else:
        modelname = cfg.MODEL.ARCHITECTURE if cfg.PROBLEM.NDIM == '2D' else cfg.MODEL.ARCHITECTURE + '_3d'
    if cfg.PROBLEM.TYPE == 'INSTANCE_SEG': modelname = modelname+"_instances"
    mdl = importlib.import_module('models.'+modelname)
    names = [x for x in mdl.__dict__ if not x.startswith("_")]
    globals().update({k: getattr(mdl, k) for k in names})

    # Model building
    if cfg.MODEL.ARCHITECTURE == 'unet' or cfg.MODEL.ARCHITECTURE == 'resunet' or cfg.MODEL.ARCHITECTURE == 'seunet' or \
       cfg.MODEL.ARCHITECTURE == 'attention_unet':
        args = dict(image_shape=cfg.DATA.PATCH_SIZE, activation=cfg.MODEL.ACTIVATION, feature_maps=cfg.MODEL.FEATURE_MAPS,
                drop_values=cfg.MODEL.DROPOUT_VALUES, spatial_dropout=cfg.MODEL.SPATIAL_DROPOUT,
                batch_norm=cfg.MODEL.BATCH_NORMALIZATION, k_init=cfg.MODEL.KERNEL_INIT)
        if cfg.MODEL.ARCHITECTURE == 'unet':
            f_name = U_Net_3D if cfg.PROBLEM.NDIM == '3D' else U_Net_2D
        elif cfg.MODEL.ARCHITECTURE == 'resunet':
            f_name = ResUNet_3D if cfg.PROBLEM.NDIM == '3D' else ResUNet_2D
        elif cfg.MODEL.ARCHITECTURE == 'attention_unet':
            f_name = Attention_U_Net_3D if cfg.PROBLEM.NDIM == '3D' else Attention_U_Net_2D
        elif cfg.MODEL.ARCHITECTURE == 'seunet':
            f_name = SE_U_Net_3D if cfg.PROBLEM.NDIM == '3D' else SE_U_Net_2D

        if cfg.PROBLEM.TYPE == 'INSTANCE_SEG':
            args['output_channels'] = cfg.DATA.CHANNELS
            args['channel_weights'] = cfg.DATA.CHANNEL_WEIGHTS
        else:
            args['n_classes'] = cfg.MODEL.N_CLASSES
        if cfg.PROBLEM.NDIM == '3D':
            args['z_down'] = cfg.MODEL.Z_DOWN

        model = f_name(**args)

    elif cfg.MODEL.ARCHITECTURE == 'fcn32':
        if cfg.PROBLEM.NDIM == '2D':
            model = FCN32_VGG16(cfg.DATA.PATCH_SIZE, n_classes=cfg.MODEL.N_CLASSES)
        else:
            raise ValueError("Not implemented pipeline option")

    elif cfg.MODEL.ARCHITECTURE == 'fcn8':
        if cfg.PROBLEM.NDIM == '2D':
            model = FCN8_VGG16(cfg.DATA.PATCH_SIZE, n_classes=cfg.MODEL.N_CLASSES)
        else:
            raise ValueError("Not implemented pipeline option")

    elif cfg.MODEL.ARCHITECTURE == 'tiramisu':
        if cfg.PROBLEM.NDIM == '2D':
            model = FC_DenseNet103(cfg.DATA.PATCH_SIZE, n_filters_first_conv=n_filters_first_conv, n_pool=cfg.MODEL.DEPTH,
                growth_rate=growth_rate, n_layers_per_block=n_layers_per_block, dropout_p=dropout_value)
        else:
            raise ValueError("Not implemented pipeline option")

    elif cfg.MODEL.ARCHITECTURE == 'mnet':
        if cfg.PROBLEM.NDIM == '2D':
            model = MNet((None, None, cfg.DATA.PATCH_SIZE[-1]))
        else:
            raise ValueError("Not implemented pipeline option")

    elif cfg.MODEL.ARCHITECTURE == 'multiresunet':
        if cfg.PROBLEM.NDIM == '2D':
            model = MultiResUnet(None, None, cfg.DATA.PATCH_SIZE[-1])
        else:
            raise ValueError("Not implemented pipeline option")

    # Check the network created
    model.summary(line_length=150)
    os.makedirs(cfg.PATHS.CHARTS, exist_ok=True)
    model_name = os.path.join(cfg.PATHS.CHARTS, "model_plot_" + job_identifier + ".png")
    plot_model(model, to_file=model_name, show_shapes=True, show_layer_names=True)

    return model
