from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (Dropout, SpatialDropout2D, Conv2D, Conv2DTranspose, MaxPooling2D,
                                     Concatenate, Add, BatchNormalization, ELU, Activation, UpSampling2D)


def ResUNet_2D(image_shape, activation='elu', feature_maps=[16,32,64,128,256], drop_values=[0.1,0.1,0.1,0.1,0.1],
               spatial_dropout=False, batch_norm=False, k_init='he_normal', k_size=3,
               upsample_layer="convtranspose",  output_channels="BC"):
    """Create 2D Residual_U-Net.

       Parameters
       ----------
       image_shape : 3D tuple
           Dimensions of the input image. E.g. ``(y, x, channels)``

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

       k_init : str, optional
           Kernel initialization for convolutional layers.

       k_size : int, optional
           Kernel size.

       upsample_layer : str, optional
           Type of layer to use to make upsampling. Two options: "convtranspose" or "upsampling".    

       output_channels : str, optional
           Channels to operate with. Possible values: ``B``, ``BC`` and ``BCD``. ``B`` stands for binary segmentation.
           ``BC`` corresponds to use binary segmentation+contour. ``BCD`` stands for binary segmentation+contour+distances.

       Returns
       -------
       model : Keras model
           Model containing the U-Net.


       Calling this function with its default parameters returns the following network:

       .. image:: ../img/resunet.png
           :width: 100%
           :align: center

       Where each green layer represents a residual block as the following:

       .. image:: ../img/res_block.png
           :width: 45%
           :align: center

       Images created with `PlotNeuralNet <https://github.com/HarisIqbal88/PlotNeuralNet>`_.
    """

    depth = len(feature_maps)-1

    fm = feature_maps[::-1]

    dinamic_dim = (None,)*(len(image_shape)-1) + (image_shape[-1],)
    inputs = Input(dinamic_dim)

    x = level_block(inputs, depth, fm, k_size, activation, k_init, drop_values, spatial_dropout, batch_norm, True,
        upsample_layer)

    if output_channels == "Dv2":
        outputs = Conv2D(1, (2, 2), activation="linear", padding='same') (x)
    elif output_channels == "BC":
        outputs = Conv2D(2, (2, 2), activation="sigmoid", padding='same') (x)
    elif output_channels == "BCM":
        outputs = Conv2D(3, (2, 2), activation="sigmoid", padding='same') (x)
    elif output_channels == "BDv2":
        seg = Conv2D(1, (2, 2), activation="sigmoid", padding='same') (x)
        dis = Conv2D(1, (2, 2), activation="linear", padding='same') (x)
        outputs = Concatenate()([seg, dis])
    elif output_channels in ["BCD", "BCDv2"]:
        seg = Conv2D(2, (2, 2), activation="sigmoid", padding='same') (x)
        dis = Conv2D(1, (2, 2), activation="linear", padding='same') (x)
        outputs = Concatenate()([seg, dis])

    model = Model(inputs=[inputs], outputs=[outputs])

    return model


def level_block(x, depth, f_maps, f_size, activation, k_init, drop_values, spatial_dropout, batch_norm, first_block,
                upsample_layer):
    """Produces a level of the network. It calls itself recursively.

       Parameters
       ----------
       x : Keras layer
           Input layer of the block.

       depth :int
           Depth of the network. This value determines how many times the function will be called recursively.

       f_maps : array of ints
           Feature maps to use.

       f_size : int
           Convolution window.

       activation : str, optional
           Keras available activation type.

       k_init : str, optional
           Keras available kernel initializer type.

       drop_values : array of floats, optional
           Dropout value to be fixed.

       spatial_dropout : bool, optional
           Use spatial dropout instead of the `normal` dropout.

       batch_norm : bool, optional
           Use batch normalization.

       first_block : float, optional
           To advice the function that it is the first residual block of the network, which avoids Full Pre-Activation
           layers (more info of Full Pre-Activation in `Identity Mappings in Deep Residual Networks
           <https://arxiv.org/pdf/1603.05027.pdf>`_).

       upsample_layer : str, optional
           Type of layer to use to make upsampling. Two options: "convtranspose" or "upsampling".    

       Returns
       -------
       x : Keras layer
           last layer of the levels.
    """

    if depth > 0:
        r = residual_block(x, f_maps[depth], f_size, activation, k_init, drop_values[depth], spatial_dropout, batch_norm, first_block)
        x = MaxPooling2D((2, 2)) (r)
        x = level_block(x, depth-1, f_maps, f_size, activation, k_init, drop_values, spatial_dropout, batch_norm, False,
            upsample_layer)
        if upsample_layer == "convtranspose":
            x = Conv2DTranspose(f_maps[depth], (2, 2), strides=(2, 2), padding='same') (x)
        else:
            x = UpSampling2D() (x)
        x = Concatenate()([r, x])
        x = residual_block(x, f_maps[depth], f_size, activation, k_init, drop_values[depth], spatial_dropout, batch_norm, False)
    else:
        x = residual_block(x, f_maps[depth], f_size, activation, k_init, drop_values[depth], spatial_dropout, batch_norm, False)
    return x


def residual_block(x, f_maps, f_size, activation='elu', k_init='he_normal', drop_value=0.0, spatial_dropout=False,
                   bn=False, first_block=False):
    """Residual block.

       Parameters
       ----------

       x : Keras layer
           Input layer of the block.

       f_maps : array of ints
           Feature maps to use.

       f_size : int
           Convolution window.

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
    shortcut = Conv2D(f_maps, activation=None, kernel_size=(1, 1), kernel_initializer=k_init)(x)

    # Main path
    if not first_block:
        x = BatchNormalization()(x) if bn else x
        if activation == 'elu':
            x = ELU(alpha=1.0) (x)
        else:
            x = Activation(activation) (x)
    x = Conv2D(f_maps, f_size, activation=None, kernel_initializer=k_init, padding='same') (x)

    if drop_value > 0:
        if spatial_dropout:
            x = SpatialDropout2D(drop_value) (x)
        else:
            x = Dropout(drop_value) (x)

    x = BatchNormalization()(x) if bn else x
    if activation == 'elu':
        x = ELU(alpha=1.0) (x)
    else:
        x = Activation(activation) (x)
    x = Conv2D(f_maps, f_size, activation=None, kernel_initializer=k_init, padding='same') (x)
    x = BatchNormalization()(x) if bn else x

    # Add shortcut value to main path
    x = Add()([shortcut, x])

    return x
