from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (Dropout, Conv3D, Conv3DTranspose, MaxPooling3D, Concatenate, Add,
                                     BatchNormalization, ELU, Activation, ZeroPadding3D, SpatialDropout2D,
                                     UpSampling3D)


def ResUNet_3D(image_shape, activation='elu', feature_maps=[16,32,64,128,256], drop_values=[0.1,0.1,0.1,0.1,0.1],
               spatial_dropout=False, batch_norm=False, z_down=2, k_init='he_normal', k_size=3, reduced_decoder=False,
               upsample_layer="convtranspose", output_channels="BC"):
    """Create 3D Residual_U-Net for instance segmentation. It can be output up to ``3`` channels in the following order:
       binary segmentation, contour segmentation and distance transform. Binary cross entropy (BCE) will be used for the
       first two channels and mean squared error (MSE) to the last one. The channel number is controlled with
       ``output_channels`` arg.

       Parameters
       ----------
       image_shape : 4D tuple
           Dimensions of the input image. E.g. ``(z, y, x, channels)``

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

       z_down : int, optional
           Downsampling used in z dimension. Set it to ``1`` if the dataset is not isotropic.

       k_init : str, optional
           Keras available kernel initializer type.

       k_size : int, optional
           Kernel size.
           
       reduced_decoder : bool, optional
           Reduce the feature maps of the decoder using the first feature size in ``feature_maps``. 
           E.g. if ``feature_maps=[32,64,128]`` in feature used in the decoder convolutions will 
           be ``32`` always.

       upsample_layer : str, optional
           Type of layer to use to make upsampling. Two options: "convtranspose" or "upsampling". 
                      
       out_channels : str, optional
           Channels to operate with. Possible values: ``BC`` and ``BCD``.  ``BC`` corresponds to use binary
           segmentation+contour. ``BCD`` stands for binary segmentation+contour+distances.

       Returns
       -------
       Model : Keras model
            Model containing the U-Net.


       Calling this function with its default parameters returns the following network:

       .. image:: img/resunet_3d.png
           :width: 100%
           :align: center

       Where each green layer represents a residual block as the following:

       .. image:: img/res_block.png
           :width: 45%
           :align: center

       Images created with `PlotNeuralNet <https://github.com/HarisIqbal88/PlotNeuralNet>`_.
    """

    if len(feature_maps) != len(drop_values):
        raise ValueError("'feature_maps' dimension must be equal 'drop_values' dimension")
    depth = len(feature_maps)-1

    assert output_channels in ['BC', 'BCM', 'BCD', 'BCDv2', 'BDv2', 'Dv2']
    
    fm = feature_maps[::-1]

    inputs = Input(image_shape)

    x = level_block(inputs, depth, fm, k_size, activation, k_init, drop_values, spatial_dropout, batch_norm, True, z_down,
        reduced_decoder, upsample_layer)

    if output_channels == "BC":
        outputs = Conv3D(2, (2, 2, 2), activation="sigmoid", padding='same') (x)
    elif output_channels == "BCM":
        outputs = Conv3D(3, (2, 2, 2), activation="sigmoid", padding='same') (x)
    elif output_channels in ["BCD", "BCDv2"]:
        seg = Conv3D(2, (2, 2, 2), activation="sigmoid", padding='same') (x)
        dis = Conv3D(1, (2, 2, 2), activation="linear", padding='same') (x)
        outputs = Concatenate()([seg, dis])
    elif output_channels == "BDv2":
        seg = Conv3D(1, (2, 2, 2), activation="sigmoid", padding='same') (x)
        dis = Conv3D(1, (2, 2, 2), activation="linear", padding='same') (x)
        outputs = Concatenate()([seg, dis])
    else: # Dv2
        outputs = Conv3D(1, (2, 2, 2), activation="linear", padding='same') (x)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model


def level_block(x, depth, f_maps, filter_size, activation, k_init, drop_values, spatial_dropout, batch_norm, first_block,
                z_down, reduced_decoder, upsample_layer):
    """Produces a level of the network. It calls itself recursively.

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

       z_down : int, optional
           Downsampling used in z dimension. Set it to 1 if the dataset is not isotropic.

       reduced_decoder : bool, optional
           Reduce the feature maps of the decoder using the first feature size in ``feature_maps``. 
           E.g. if ``feature_maps=[32,64,128]`` in feature used in the decoder convolutions will 
           be ``32`` always.

       upsample_layer : str, optional
           Type of layer to use to make upsampling. Two options: "convtranspose" or "upsampling". 

       Returns
       -------
       x : Keras layer
           last layer of the levels.
    """

    if depth > 0:
        r = residual_block(x, f_maps[depth], filter_size, activation, k_init, drop_values[depth], spatial_dropout,
                           batch_norm, first_block)
        x = MaxPooling3D((z_down, 2, 2)) (r)
        x = level_block(x, depth-1, f_maps, filter_size, activation, k_init, drop_values, spatial_dropout, batch_norm,
                        False, z_down, reduced_decoder, upsample_layer)
        d = 0 if reduced_decoder else depth
        if upsample_layer == "convtranspose":
            x = Conv3DTranspose(f_maps[d], (2, 2, 2), strides=(z_down, 2, 2), padding='same') (x)
        else:
            x = UpSampling3D() (x)

        # Adjust shape introducing zero padding to allow the concatenation
        a = x.shape[3]
        b = r.shape[3]
        s = a - b
        if s > 0:
            r = ZeroPadding3D(padding=((0,0), (0,0), (s,0))) (r)
        elif s < 0:
            x = ZeroPadding3D(padding=((0,0), (0,0), (abs(s),0))) (x)
        x = Concatenate()([x, r])

        x = residual_block(x, f_maps[d], filter_size, activation, k_init, drop_values[depth], spatial_dropout,
                           batch_norm, False)
    else:
        d = depth-1 if reduced_decoder else depth
        x = residual_block(x, f_maps[d], filter_size, activation, k_init, drop_values[depth], spatial_dropout,
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
    shortcut = Conv3D(f_maps, activation=None, kernel_size=(1, 1, 1), kernel_initializer=k_init)(x)

    # Main path
    if not first_block:
        x = BatchNormalization()(x) if bn else x
        if activation == "elu":
            x = ELU(alpha=1.0) (x)
        else:
            x = Activation(activation) (x)
    x = Conv3D(f_maps, filter_size, activation=None, kernel_initializer=k_init, padding='same') (x)

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
    x = Conv3D(f_maps, filter_size, activation=None, kernel_initializer=k_init, padding='same') (x)
    x = BatchNormalization()(x) if bn else x

    # Add shortcut value to main path
    x = Add()([shortcut, x])

    return x
