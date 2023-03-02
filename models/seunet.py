from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Dropout, concatenate, BatchNormalization, Activation, Reshape, Dense, multiply, Permute, 
                                    SpatialDropout2D)
from tensorflow.keras import Model, Input


def SE_U_Net(image_shape, activation='elu', feature_maps=[32, 64, 128, 256], drop_values=[0.1,0.1,0.1,0.1],
    spatial_dropout=False, batch_norm=False, k_init='he_normal', k_size=3, upsample_layer="convtranspose", 
    z_down=2, n_classes=1, last_act='sigmoid', output_channels="BC"):
    """
    Create 2D/3D U-Net with squeeze-excite blocks. Reference `Squeeze and Excitation 
    Networks <https://arxiv.org/abs/1709.01507>`_.

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

    z_down : int, optional
        Downsampling used in z dimension. Set it to ``1`` if the dataset is not isotropic.

    n_classes: int, optional                                                 
        Number of classes.    

    last_act : str, optional
        Name of the last activation layer.

    output_channels : str, optional
        Channels to operate with. Possible values: ``BC``, ``BCD``, ``BP``, ``BCDv2``,
        ``BDv2``, ``Dv2`` and ``BCM``.

    Returns
    -------
    model : Keras model
        Model containing the U-Net.


    Calling this function with its default parameters returns the following
    network:

    .. image:: ../../img/unet_3d.png
        :width: 100%
        :align: center

    Image created with `PlotNeuralNet <https://github.com/HarisIqbal88/PlotNeuralNet>`_.
    """

    depth = len(feature_maps)-1
    x = Input(image_shape)
    inputs = x

    global conv, convtranspose, maxpooling, mpool, zeropadding, upsampling, globalaveragepooling
    if len(image_shape) == 4:
        from tensorflow.keras.layers import (Conv3D, Conv3DTranspose, MaxPooling3D, ZeroPadding3D, UpSampling3D, GlobalAveragePooling3D)
        conv = Conv3D
        convtranspose = Conv3DTranspose
        maxpooling = MaxPooling3D
        mpool = (z_down, 2, 2)
        zeropadding = ZeroPadding3D
        upsampling = UpSampling3D
        globalaveragepooling = GlobalAveragePooling3D
    else:
        from tensorflow.keras.layers import (Conv2D, Conv2DTranspose, MaxPooling2D, ZeroPadding2D, UpSampling2D, GlobalAveragePooling2D)
        conv = Conv2D
        convtranspose = Conv2DTranspose
        maxpooling = MaxPooling2D
        mpool = (2, 2)
        zeropadding = ZeroPadding2D
        upsampling = UpSampling2D
        globalaveragepooling = GlobalAveragePooling2D

    # List used to access layers easily to make the skip connections of the U-Net
    l=[]

    # ENCODER
    for i in range(depth):
        x = conv(feature_maps[i], k_size, activation=None, kernel_initializer=k_init, padding='same') (x)
        x = BatchNormalization() (x) if batch_norm else x 
        x = Activation(activation) (x)
        x = squeeze_excite_block(x)
        if spatial_dropout and drop_values[i] > 0:
            x = SpatialDropout2D(drop_values[i]) (x)
        elif drop_values[i] > 0 and not spatial_dropout:
            x = Dropout(drop_values[i]) (x)

        x = conv(feature_maps[i], k_size, activation=None, kernel_initializer=k_init, padding='same') (x)
        x = BatchNormalization() (x) if batch_norm else x 
        x = Activation(activation) (x)
        x = squeeze_excite_block(x)

        l.append(x)
    
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
        if upsample_layer == "convtranspose":
            x = convtranspose(feature_maps[i], 2, strides=mpool, padding='same') (x)
        else:
            x = upsampling(size=mpool) (x)
        x = concatenate([x, l[i]])
        x = conv(feature_maps[i], k_size, activation=None, kernel_initializer=k_init, padding='same') (x)
        x = BatchNormalization() (x) if batch_norm else x
        x = Activation(activation) (x)
        x = squeeze_excite_block(x)
        if spatial_dropout and drop_values[i] > 0:
            x = SpatialDropout2D(drop_values[i]) (x)
        elif drop_values[i] > 0 and not spatial_dropout:
            x = Dropout(drop_values[i]) (x)
        x = conv(feature_maps[i], k_size, activation=None, kernel_initializer=k_init, padding='same') (x)
        x = BatchNormalization() (x) if batch_norm else x
        x = Activation(activation) (x)
        x = squeeze_excite_block(x)

    # Instance segmentation
    if output_channels is not None:
        if output_channels == "Dv2":
            outputs = conv(1, 2, activation="linear", padding='same') (x)
        elif output_channels in ["BC", "BP"]:
            outputs = conv(2, 2, activation="sigmoid", padding='same') (x)
        elif output_channels == "BCM":
            outputs = conv(3, 2, activation="sigmoid", padding='same') (x)
        elif output_channels == "BDv2":
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

def squeeze_excite_block(input, ratio=16):
    """Create a channel-wise squeeze-excite block.

       Code fully extracted from `keras-squeeze-excite-network <https://github.com/titu1994/keras-squeeze-excite-network>`_.

       Parameters
       ----------
       input : Keras layer
           Input Keras layer

       ratio : int
           Reduction fatio. See the paper for more info.

       Returns
       -------
       x : Keras tensor
           The last layer after applayng the SE block
    """
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = globalaveragepooling()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x


