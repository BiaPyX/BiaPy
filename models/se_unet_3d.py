import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Dropout, SpatialDropout3D, Conv3D, Conv3DTranspose, MaxPooling3D, concatenate,
                                     ELU, BatchNormalization, Activation, ZeroPadding3D, GlobalAveragePooling3D,
                                     Reshape, Dense, multiply, Permute)
from tensorflow.keras import Model, Input


def SE_U_Net_3D(image_shape, activation='elu', feature_maps=[32, 64, 128, 256], drop_values=[0.1,0.1,0.1,0.1],
                spatial_dropout=False, batch_norm=False, k_init='he_normal', n_classes=1):
    """Create 3D U-Net with squeeze-excite blocks.

       Reference `Squeeze and Excitation Networks <https://arxiv.org/abs/1709.01507>`_.

       Parameters
       ----------
       image_shape : 3D tuple
           Dimensions of the input image.

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

       n_classes: int, optional                                                 
           Number of classes.    

       Returns
       -------
       model : Keras model
           Model containing the U-Net.

    
       Calling this function with its default parameters returns the following
       network:

       .. image:: ../img/unet_3d.png
           :width: 100%
           :align: center

       Image created with `PlotNeuralNet <https://github.com/HarisIqbal88/PlotNeuralNet>`_.
    """

    if len(feature_maps) != len(drop_values):
        raise ValueError("'feature_maps' dimension must be equal 'drop_values' dimension")
    depth = len(feature_maps)-1

    #dinamic_dim = (None,)*(len(image_shape)-1) + (image_shape[-1],)
    #inputs = Input(dinamic_dim)
    x = Input(image_shape)
    inputs = x
        
    # List used to access layers easily to make the skip connections of the U-Net
    l=[]

    # ENCODER
    for i in range(depth):
        x = Conv3D(feature_maps[i], (3, 3, 3), activation=None, kernel_initializer=k_init, padding='same') (x)
        x = BatchNormalization() (x) if batch_norm else x 
        x = Activation(activation) (x)
        x = squeeze_excite_block(x)
        if spatial_dropout and drop_values[i] > 0:
            x = SpatialDropout3D(drop_values[i]) (x)
        elif drop_values[i] > 0 and not spatial_dropout:
            x = Dropout(drop_values[i]) (x)

        x = Conv3D(feature_maps[i], (3, 3, 3), activation=None, kernel_initializer=k_init, padding='same') (x)
        x = BatchNormalization() (x) if batch_norm else x 
        x = Activation(activation) (x)
        x = squeeze_excite_block(x)

        l.append(x)
    
        x = MaxPooling3D((2, 2, 2))(x)

    # BOTTLENECK
    x = Conv3D(feature_maps[depth], (3, 3, 3), activation=None, kernel_initializer=k_init, padding='same')(x)
    x = BatchNormalization() (x) if batch_norm else x
    x = Activation(activation) (x)
    if spatial_dropout and drop_values[depth] > 0:
        x = SpatialDropout3D(drop_values[depth]) (x)
    elif drop_values[depth] > 0 and not spatial_dropout:
        x = Dropout(drop_values[depth]) (x)

    x = Conv3D(feature_maps[depth], (3, 3, 3), activation=None, kernel_initializer=k_init, padding='same') (x)
    x = BatchNormalization() (x) if batch_norm else x
    x = Activation(activation) (x)

    # DECODER
    for i in range(depth-1, -1, -1):
        x = Conv3DTranspose(feature_maps[i], (2, 2, 2), strides=(2, 2, 2), padding='same') (x)
        x = concatenate([x, l[i]])
        x = Conv3D(feature_maps[i], (3, 3, 3), activation=None, kernel_initializer=k_init, padding='same') (x)
        x = BatchNormalization() (x) if batch_norm else x
        x = Activation(activation) (x)
        x = squeeze_excite_block(x)
        if spatial_dropout and drop_values[i] > 0:
            x = SpatialDropout3D(drop_values[i]) (x)
        elif drop_values[i] > 0 and not spatial_dropout:
            x = Dropout(drop_values[i]) (x)
        x = Conv3D(feature_maps[i], (3, 3, 3), activation=None, kernel_initializer=k_init, padding='same') (x)
        x = BatchNormalization() (x) if batch_norm else x
        x = Activation(activation) (x)
        x = squeeze_excite_block(x)

    outputs = Conv3D(n_classes, (1, 1, 1), activation='sigmoid') (x)
    
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

    se = GlobalAveragePooling3D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x


