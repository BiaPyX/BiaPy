from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout, Concatenate, Add, BatchNormalization, ELU,  Activation, SpatialDropout2D


def ResUNet(image_shape, activation='elu', feature_maps=[16,32,64,128,256], drop_values=[0.1,0.1,0.1,0.1,0.1],
    spatial_dropout=False, batch_norm=False, z_down=[2,2,2,2,2], k_init='he_normal', k_size=3, upsample_layer="convtranspose", 
    n_classes=1, last_act='sigmoid', output_channels="BC", upsampling_factor=1, upsampling_position="pre"):

    """
    Create 2D/3D Residual_U-Net. 

    Parameters
    ----------
    image_shape : 3D/4D tuple
        Dimensions of the input image. E.g. ``(y, x, channels)`` or ``(z, y, x, channels)``.

    activation : str, optional
        Keras available activation type.

    feature_maps : array of ints, optional
        Feature maps to use on each level.

    drop_values : array of floats, optional
        Dropout value to be fixed.

    spatial_dropout : bool, optional
        Use spatial dropout instead of the `normal` dropout.

    batch_norm : bool, optional
        Make batch normalization.

    z_down : List of ints, optional
        Downsampling used in z dimension. Set it to ``1`` if the dataset is not isotropic.

    k_init : str, optional
        Keras available kernel initializer type.

    k_size : int, optional
        Kernel size.

    upsample_layer : str, optional
        Type of layer to use to make upsampling. Two options: "convtranspose" or "upsampling". 

    n_classes: int, optional
        Number of classes.

    last_act : str, optional
        Name of the last activation layer.

    output_channels : str, optional
        Channels to operate with. Possible values: ``BC``, ``BCD``, ``BP``, ``BCDv2``,
        ``BDv2``, ``Dv2`` and ``BCM``.
    
    upsampling_factor : int, optional
        Factor of upsampling for super resolution workflow. 

    upsampling_position : str, optional
        Whether the upsampling is going to be made previously (``pre`` option) to the model 
        or after the model (``post`` option).

    Returns
    -------
    Model : Keras model
        Model containing the U-Net.


    Calling this function with its default parameters returns the following network:

    .. image:: ../../img/models/resunet.png
        :width: 100%
        :align: center

    Where each green layer represents a residual block as the following:

    .. image:: ../../img/models/res_block.png
        :width: 45%
        :align: center

    Images created with `PlotNeuralNet <https://github.com/HarisIqbal88/PlotNeuralNet>`_.
    """

    depth = len(feature_maps)-1
    fm = feature_maps[::-1]
    dp = drop_values[::-1]
    zd = z_down[::-1]

    global ndim, conv, convtranspose, maxpooling, zeropadding, upsampling
    if len(image_shape) == 4:
        from tensorflow.keras.layers import (Conv3D, Conv3DTranspose, MaxPooling3D, ZeroPadding3D, UpSampling3D)
        conv = Conv3D
        convtranspose = Conv3DTranspose
        maxpooling = MaxPooling3D
        zeropadding = ZeroPadding3D
        upsampling = UpSampling3D
        ndim = 3
    else:
        from tensorflow.keras.layers import (Conv2D, Conv2DTranspose, MaxPooling2D, ZeroPadding2D, UpSampling2D)
        conv = Conv2D
        convtranspose = Conv2DTranspose
        maxpooling = MaxPooling2D
        zeropadding = ZeroPadding2D
        upsampling = UpSampling2D
        ndim = 2

    dinamic_dim = (None,)*(len(image_shape)-1) + (image_shape[-1],)
    inputs = Input(dinamic_dim)

    if upsampling_factor > 1 and upsampling_position =="pre":
        mpool = (1, 2, 2) if len(image_shape) == 4 else (2, 2)
        s = convtranspose(fm[-1], 2, strides=mpool, padding='same') (inputs)
        x = s 
    else:
        x = inputs 
        
    x = level_block(x, depth, fm, k_size, activation, k_init, dp, spatial_dropout, batch_norm, True,
        upsample_layer, zd)

    if upsampling_factor > 1:
        if upsampling_position =="pre":
            x = Add()([s,x]) # long shortcut
        else:
            mpool = (1, 2, 2) if len(image_shape) == 4 else (2, 2)
            x = convtranspose(fm[-1], 2, strides=mpool, padding='same') (x)

    # Instance segmentation
    if output_channels is not None:
        if output_channels == "Dv2":
            outputs = conv(1, 2, activation="linear", padding='same') (x)
        elif output_channels in ["BC", "BP"]:
            outputs = conv(2, 2, activation="sigmoid", padding='same') (x)
        elif output_channels == "BCM":
            outputs = conv(3, 2, activation="sigmoid", padding='same') (x)
        elif output_channels in ["BDv2", "BD"]:
            seg = conv(1, 2, activation="sigmoid", padding='same') (x)
            dis = conv(1, 2, activation="linear", padding='same') (x)
            outputs = Concatenate()([seg, dis])
        elif output_channels in ["BCD", "BCDv2"]:
            seg = conv(2, 2, activation="sigmoid", padding='same') (x)
            dis = conv(1, 2, activation="linear", padding='same') (x)
            outputs = Concatenate()([seg, dis])
    # Other
    else:
        outputs = conv(n_classes, 1, activation=last_act) (x)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model


def level_block(x, depth, f_maps, filter_size, activation, k_init, drop_values, spatial_dropout, batch_norm, first_block,
                upsample_layer, z_down):
    """
    Produces a level of the network. It calls itself recursively.

    Parameters
    ----------
    x : Keras layer
        Input layer of the block.

    depth : int
        Depth of the network. This value determines how many times the function will be called recursively.

    f_maps : array of ints
        Feature maps to use.

    filter_size : 3 int tuple
        Height, width and depth of the convolution window.

    activation : str
        Keras available activation type.

    k_init : str
        Keras available kernel initializer type.

    drop_values : array of floats
        Dropout value to be fixed.

    spatial_dropout : bool
        Use spatial dropout instead of the `normal` dropout.

    batch_norm : bool
        Use batch normalization.

    first_block : float
        To advice the function that it is the first residual block of the network, which avoids Full Pre-Activation
        layers (more info of Full Pre-Activation in `Identity Mappings in Deep Residual Networks
        <https://arxiv.org/pdf/1603.05027.pdf>`_).

    upsample_layer : str
        Type of layer to use to make upsampling. Two options: "convtranspose" or "upsampling". 

    z_down : List of ints
        Downsampling used in z dimension. Set it to ``1`` if the dataset is not isotropic.

    Returns
    -------
    x : Keras layer
        last layer of the levels.
    """

    if depth > 0:
        r = residual_block(x, f_maps[depth], filter_size, activation, k_init, drop_values[depth], spatial_dropout,
                           batch_norm, first_block)
        mpool = (z_down[depth-1], 2, 2) if ndim == 3 else (2, 2)
        x = maxpooling(mpool) (r)
        x = level_block(x, depth-1, f_maps, filter_size, activation, k_init, drop_values, spatial_dropout, batch_norm,
                        False, upsample_layer, z_down)
        if upsample_layer == "convtranspose":
            x = convtranspose(f_maps[depth], 2, strides=mpool, padding='same') (x)
        else:
            x = upsampling(size=mpool) (x)
        x = Concatenate()([x, r])

        x = residual_block(x, f_maps[depth], filter_size, activation, k_init, drop_values[depth], spatial_dropout,
                           batch_norm, False)
    else:
        x = residual_block(x, f_maps[depth], filter_size, activation, k_init, drop_values[depth], spatial_dropout,
                           batch_norm, False)
    return x


def residual_block(x, f_maps, filter_size, activation='elu', k_init='he_normal', drop_value=0.0, spatial_dropout=False,
                   bn=False, first_block=False):
    """Residual block.

       Parameters
       ----------
       x : Keras layer
           iInput layer of the block.

       f_maps : array of ints
           Feature maps to use.

       filter_size : 3 int tuple
           Height, width and depth of the convolution window.

       activation : str, optional
           Keras available activation type.

       k_init : str, optional
           Keras available kernel initializer type.

       drop_value : float, optional
           Dropout value to be fixed.

       spatial_dropout : bool, optional
           Use spatial dropout instead of the `normal` dropout.

       bn : bool, optional
           Use batch normalization.

       first_block : float, optional
           To advice the function that it is the first residual block of the network, which avoids Full Pre-Activation
           layers (more info of Full Pre-Activation in `Identity Mappings in Deep Residual Networks
           <https://arxiv.org/pdf/1603.05027.pdf>`_).

       Returns
       -------
       x : Keras layer
           Last layer of the block.
    """

    # Create shorcut
    shortcut = conv(f_maps, activation=None, kernel_size=1, kernel_initializer=k_init)(x)

    # Main path
    if not first_block:
        x = BatchNormalization()(x) if bn else x
        if activation == "elu":
            x = ELU(alpha=1.0) (x)
        else:
            x = Activation(activation) (x)
    x = conv(f_maps, filter_size, activation=None, kernel_initializer=k_init, padding='same') (x)

    if drop_value > 0:
        if spatial_dropout:
            x = SpatialDropout2D(drop_value) (x)
        else:
            x = Dropout(drop_value) (x)
    x = BatchNormalization()(x) if bn else x
    if activation == "elu":
        x = ELU(alpha=1.0) (x)
    else:
        x = Activation(activation) (x)
    x = conv(f_maps, filter_size, activation=None, kernel_initializer=k_init, padding='same') (x)
    x = BatchNormalization()(x) if bn else x

    # Add shortcut value to main path
    x = Add()([shortcut, x])

    return x
