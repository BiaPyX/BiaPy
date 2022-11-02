from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (Dropout, SpatialDropout2D, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate,
                                     BatchNormalization, Activation, Add, Multiply, UpSampling2D)


def Attention_U_Net_2D(image_shape, activation='elu', feature_maps=[16, 32, 64, 128, 256],
                       drop_values=[0.1,0.1,0.2,0.2,0.3], spatial_dropout=False, batch_norm=False,
                       k_init='he_normal', k_size=3, upsample_layer="convtranspose", n_classes=1, 
                       last_act='sigmoid'):
    """Create 2D U-Net with Attention blocks.

       Based on `Attention U-Net: Learning Where to Look for the Pancreas <https://arxiv.org/abs/1804.03999>`_.

       Parameters
       ----------
       image_shape : 2D tuple
           Dimensions of the input image.

       activation : str, optional
           Keras available activation type.

       feature_maps : array of ints, optional
           Feature maps to use on each level.

       drop_values : float, optional
           Dropout value to be fixed. If no value is provided the default behaviour will be to select a piramidal value
           starting from ``0.1`` and reaching ``0.3`` value.

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
                      
       n_classes: int, optional
           Number of classes.

       last_act : str, optional
           Name of the last activation layer.
           
       Returns
       -------
       model : Keras model
           Model containing the Attention U-Net.


       Example
       -------

       Calling this function with its default parameters returns the following network:

       .. image:: ../img/unet.png
           :width: 100%
           :align: center

       Image created with `PlotNeuralNet <https://github.com/HarisIqbal88/PlotNeuralNet>`_.

       That networks incorporates in skip connecions Attention Gates (AG), which
       can be seen as follows:

       .. image:: ../img/attention_gate.png
           :width: 100%
           :align: center

       Image extracted from `Attention U-Net: Learning Where to Look for the Pancreas <https://arxiv.org/abs/1804.03999>`_.
    """

    depth = len(feature_maps)-1

    dinamic_dim = (None,)*(len(image_shape)-1) + (image_shape[-1],)
    x = Input(dinamic_dim)
    #x = Input(image_shape)
    inputs = x

    # List used to access layers easily to make the skip connections of the U-Net
    l=[]

    # ENCODER
    for i in range(depth):
        x = Conv2D(feature_maps[i], k_size, activation=None, kernel_initializer=k_init, padding='same') (x)
        x = BatchNormalization() (x) if batch_norm else x
        x = Activation(activation) (x)
        if drop_values is not None:
            if spatial_dropout:
                x = SpatialDropout2D(drop_values[i]) (x)
            else:
                x = Dropout(drop_values[i]) (x)
        x = Conv2D(feature_maps[i], k_size, activation=None, kernel_initializer=k_init, padding='same') (x)
        x = BatchNormalization() (x) if batch_norm else x
        x = Activation(activation) (x)

        l.append(x)

        x = MaxPooling2D((2, 2))(x)

    # BOTTLENECK
    x = Conv2D(feature_maps[depth], k_size, activation=None, kernel_initializer=k_init, padding='same')(x)
    x = BatchNormalization() (x) if batch_norm else x
    x = Activation(activation) (x)
    if drop_values is not None:
            if spatial_dropout:
                x = SpatialDropout2D(drop_values[depth]) (x)
            else:
                x = Dropout(drop_values[depth]) (x)
    x = Conv2D(feature_maps[depth], k_size, activation=None, kernel_initializer=k_init, padding='same') (x)
    x = BatchNormalization() (x) if batch_norm else x
    x = Activation(activation) (x)

    # DECODER
    for i in range(depth-1, -1, -1):
        if upsample_layer == "convtranspose":
            x = Conv2DTranspose(feature_maps[i], (2, 2), strides=(2, 2), padding='same') (x)
        else:
            x = UpSampling2D() (x)
        attn = AttentionBlock(x, l[i], feature_maps[i], batch_norm)
        x = concatenate([x, attn])
        x = Conv2D(feature_maps[i], k_size, activation=None, kernel_initializer=k_init, padding='same') (x)
        x = BatchNormalization() (x) if batch_norm else x
        x = Activation(activation) (x)
        if drop_values is not None:
            if spatial_dropout:
                x = SpatialDropout2D(drop_values[i]) (x)
            else:
                x = Dropout(drop_values[i]) (x)

        x = Conv2D(feature_maps[i], k_size, activation=None, kernel_initializer=k_init, padding='same') (x)
        x = BatchNormalization() (x) if batch_norm else x
        x = Activation(activation) (x)

    outputs = Conv2D(n_classes, (1, 1), activation=last_act) (x)

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

    g1 = Conv2D(filters, kernel_size = 1)(shortcut)
    g1 = BatchNormalization() (g1) if batch_norm else g1
    x1 = Conv2D(filters, kernel_size = 1)(x)
    x1 = BatchNormalization() (x1) if batch_norm else x1

    g1_x1 = Add()([g1,x1])
    psi = Activation('relu')(g1_x1)
    psi = Conv2D(1, kernel_size = 1)(psi)
    psi = BatchNormalization() (psi) if batch_norm else psi
    psi = Activation('sigmoid')(psi)
    x = Multiply()([x,psi])
    return x
