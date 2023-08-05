from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (Dropout, Concatenate, BatchNormalization, Activation, Add, Multiply, SpatialDropout2D)


def Attention_U_Net(image_shape, activation='elu', feature_maps=[32, 64, 128, 256], drop_values=[0.1,0.1,0.1,0.1],
    spatial_dropout=False, batch_norm=False, k_init='he_normal', k_size=3, upsample_layer="convtranspose", 
    z_down=[2,2,2,2], n_classes=1, last_act='sigmoid', output_channels="BC", upsampling_factor=1,
    upsampling_position="pre"):
    """
    Create 2D/3D U-Net with Attention blocks. Based on `Attention U-Net: Learning Where to Look for the 
    Pancreas <https://arxiv.org/abs/1804.03999>`_.

    Parameters
    ----------
    image_shape : 3D/4D tuple
        Dimensions of the input image. E.g. ``(y, x, channels)`` or ``(z, y, x, channels)``.

    activation : str, optional
        Keras available activation type.

    feature_maps : array of ints, optional
        Feature maps to use on each level.

    drop_values : float, optional
        Dropout value to be fixed.

    spatial_dropout : bool, optional
        Use spatial dropout instead of the `normal` dropout.

    batch_norm : bool, optional
        Make batch normalization.

    k_init : string, optional
        Kernel initialization for convolutional layers.

    k_size : int, optional
        Kernel size.

    upsample_layer : str, optional
        Type of layer to use to make upsampling. Two options: "convtranspose" or "upsampling". 
                    
    z_down : List of ints, optional
        Downsampling used in z dimension. Set it to ``1`` if the dataset is not isotropic.

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
    model : Keras model
        Model containing the Attention U-Net.


    Calling this function with its default parameters returns the following network:


    Examples
    --------

    Calling this function with its default parameters returns the following network:

    .. image:: ../../img/models/unet.png
        :width: 100%
        :align: center

    Image created with `PlotNeuralNet <https://github.com/HarisIqbal88/PlotNeuralNet>`_.

    That networks incorporates in skip connecions Attention Gates (AG), which can be seen as follows:

    .. image:: ../../img/models/attention_gate.png
        :width: 100%
        :align: center

    Image extracted from `Attention U-Net: Learning Where to Look for the Pancreas <https://arxiv.org/abs/1804.03999>`_.
    """
    depth = len(feature_maps)-1

    dinamic_dim = (None,)*(len(image_shape)-1) + (image_shape[-1],)
    x = Input(dinamic_dim)
    inputs = x

    global conv, convtranspose, maxpooling, zeropadding, upsampling
    if len(image_shape) == 4:
        from tensorflow.keras.layers import (Conv3D, Conv3DTranspose, MaxPooling3D, ZeroPadding3D, UpSampling3D)
        conv = Conv3D
        convtranspose = Conv3DTranspose
        maxpooling = MaxPooling3D
        zeropadding = ZeroPadding3D
        upsampling = UpSampling3D
    else:
        from tensorflow.keras.layers import (Conv2D, Conv2DTranspose, MaxPooling2D, ZeroPadding2D, UpSampling2D)
        conv = Conv2D
        convtranspose = Conv2DTranspose
        maxpooling = MaxPooling2D
        zeropadding = ZeroPadding2D
        upsampling = UpSampling2D

    # List used to access layers easily to make the skip connections of the U-Net
    l=[]

    if upsampling_factor > 1 and upsampling_position =="pre":
        x = convtranspose(image_shape[-1], 2, strides=2, padding='same') (x)

    # ENCODER
    for i in range(depth):
        x = conv(feature_maps[i], k_size, activation=None, kernel_initializer=k_init, padding='same') (x)
        x = BatchNormalization() (x) if batch_norm else x
        x = Activation(activation) (x)

        if spatial_dropout and drop_values[i] > 0:
            x = SpatialDropout2D(drop_values[i]) (x)
        elif drop_values[i] > 0 and not spatial_dropout:
            x = Dropout(drop_values[i]) (x)

        x = conv(feature_maps[i], k_size, activation=None, kernel_initializer=k_init, padding='same') (x)
        x = BatchNormalization() (x) if batch_norm else x
        x = Activation(activation) (x)

        l.append(x)

        mpool = (z_down[i], 2, 2) if len(image_shape) == 4 else (2, 2)
        x = maxpooling(mpool)(x)

    # BOTTLENECK
    x = conv(feature_maps[depth], k_size, activation=None, kernel_initializer=k_init, padding='same')(x)
    x = BatchNormalization() (x) if batch_norm else x
    x = Activation(activation) (x)

    if spatial_dropout and drop_values[depth] > 0:
        x = SpatialDropout2D(drop_values[depth]) (x)
    elif drop_values[depth] > 0 and not spatial_dropout:
        x = Dropout(drop_values[depth]) (x)
    x = conv(feature_maps[depth], k_size, activation=None, kernel_initializer=k_init, padding='same') (x)
    x = BatchNormalization() (x) if batch_norm else x
    x = Activation(activation) (x)

    # DECODER
    for i in range(depth-1, -1, -1):
        mpool = (z_down[i], 2, 2) if len(image_shape) == 4 else (2, 2)
        if upsample_layer == "convtranspose":
            x = convtranspose(feature_maps[i], 2, strides=mpool, padding='same') (x)
        else:
            x = upsampling(size=mpool) (x)
        attn = AttentionBlock(x, l[i], feature_maps[i], batch_norm)
        x = Concatenate()([x, attn])
        x = conv(feature_maps[i], k_size, activation=None, kernel_initializer=k_init, padding='same') (x)
        x = BatchNormalization() (x) if batch_norm else x
        x = Activation(activation) (x)

        if spatial_dropout and drop_values[i] > 0:
            x = SpatialDropout2D(drop_values[i]) (x)
        elif drop_values[i] > 0 and not spatial_dropout:
            x = Dropout(drop_values[i]) (x)

        x = conv(feature_maps[i], k_size, activation=None, kernel_initializer=k_init, padding='same') (x)
        x = BatchNormalization() (x) if batch_norm else x
        x = Activation(activation) (x)

    if upsampling_factor > 1 and upsampling_position == "post":
        x = convtranspose(n_classes, 2, strides=2, padding='same') (x)

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



def AttentionBlock(x, shortcut, filters, batch_norm):
    """Attention block.

       Extracted from `Kaggle <https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/64367>`_.

       Parameters
       ----------
       x : Keras layer
           Input layer.

       shortcut : Keras layer
           Input skip connection.

       filters : int
           Feature maps to define on the Conv layers.

       batch_norm : bool, optional
           To use batch normalization.

       Returns
       -------
       out : Keras layer
           Last layer of the Attention block.
    """

    g1 = conv(filters, kernel_size = 1)(shortcut)
    g1 = BatchNormalization() (g1) if batch_norm else g1
    x1 = conv(filters, kernel_size = 1)(x)
    x1 = BatchNormalization() (x1) if batch_norm else x1

    g1_x1 = Add()([g1,x1])
    psi = Activation('relu')(g1_x1)
    psi = conv(1, kernel_size = 1)(psi)
    psi = BatchNormalization() (psi) if batch_norm else psi
    psi = Activation('sigmoid')(psi)
    x = Multiply()([x,psi])
    return x
