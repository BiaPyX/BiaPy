from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (Dropout, SpatialDropout3D, Conv3D, Conv3DTranspose, MaxPooling3D, Concatenate,
                                     BatchNormalization, Activation, UpSampling3D)


def U_Net_3D(image_shape, activation='elu', feature_maps=[32, 64, 128, 256], drop_values=[0.1,0.1,0.1,0.1],
             spatial_dropout=False, batch_norm=False, k_init='he_normal', k_size=3, upsample_layer="convtranspose", 
             z_down=2, output_channels="BC"):
    """Create 3D U-Net for instance segmentation.

       Parameters
       ----------
       image_shape : 4D tuple
           Dimensions of the input image. E.g. ``(z, y, x, channels)``

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

       output_channels : str, optional
           Channels to operate with. Possible values: ``BC`` and ``BCD``.  ``BC`` corresponds to use binary
           segmentation+contour. ``BCD`` stands for binary segmentation+contour+distances.

       Returns
       -------
       model : Keras model
           Model containing the U-Net.


       Calling this function with its default parameters returns the following network:

       .. image:: ../img/unet_3d.png
           :width: 100%
           :align: center

       Image created with `PlotNeuralNet <https://github.com/HarisIqbal88/PlotNeuralNet>`_.
    """

    depth = len(feature_maps)-1

    # dinamic_dim = (None,)*(len(image_shape)-1) + (image_shape[-1],)
    # inputs = Input(dinamic_dim)
    # x = inputs
    x = Input(image_shape)
    inputs = x

    # List used to access layers easily to make the skip connections of the U-Net
    l=[]

    # ENCODER
    for i in range(depth):
        x = Conv3D(feature_maps[i], k_size, activation=None, kernel_initializer=k_init, padding='same') (x)
        x = BatchNormalization() (x) if batch_norm else x
        x = Activation(activation) (x)

        if spatial_dropout and drop_values[i] > 0:
            x = SpatialDropout3D(drop_values[i]) (x)
        elif drop_values[i] > 0 and not spatial_dropout:
            x = Dropout(drop_values[i]) (x)

        x = Conv3D(feature_maps[i], k_size, activation=None, kernel_initializer=k_init, padding='same') (x)
        x = BatchNormalization() (x) if batch_norm else x
        x = Activation(activation) (x)

        l.append(x)

        x = MaxPooling3D((z_down, 2, 2))(x)

    # BOTTLENECK
    x = Conv3D(feature_maps[depth], k_size, activation=None, kernel_initializer=k_init, padding='same')(x)
    x = BatchNormalization() (x) if batch_norm else x
    x = Activation(activation) (x)

    if spatial_dropout and drop_values[depth] > 0:
        x = SpatialDropout3D(drop_values[depth]) (x)
    elif drop_values[depth] > 0 and not spatial_dropout:
        x = Dropout(drop_values[depth]) (x)
    x = Conv3D(feature_maps[depth], k_size, activation=None, kernel_initializer=k_init, padding='same') (x)
    x = BatchNormalization() (x) if batch_norm else x
    x = Activation(activation) (x)

    # DECODER
    for i in range(depth-1, -1, -1):
        if upsample_layer == "convtranspose":
            x = Conv3DTranspose(feature_maps[i], (2, 2, 2), strides=(z_down, 2, 2), padding='same') (x)
        else:
            x = UpSampling3D(size=(z_down, 2, 2)) (x)
        
        x = Concatenate()([x, l[i]])

        x = Conv3D(feature_maps[i], k_size, activation=None, kernel_initializer=k_init, padding='same') (x)
        x = BatchNormalization() (x) if batch_norm else x
        x = Activation(activation) (x)

        if spatial_dropout and drop_values[i] > 0:
            x = SpatialDropout3D(drop_values[i]) (x)
        elif drop_values[i] > 0 and not spatial_dropout:
            x = Dropout(drop_values[i]) (x)

        x = Conv3D(feature_maps[i], k_size, activation=None, kernel_initializer=k_init, padding='same') (x)
        x = BatchNormalization() (x) if batch_norm else x
        x = Activation(activation) (x)

    if output_channels == "Dv2":
        outputs = Conv3D(1, (2, 2, 2), activation="linear", padding='same') (x)
    elif output_channels in ["BC", "BP"]:
        outputs = Conv3D(2, (2, 2, 2), activation="sigmoid", padding='same') (x)
    elif output_channels == "BCM":
        outputs = Conv3D(3, (2, 2, 2), activation="sigmoid", padding='same') (x)
    elif output_channels == "BDv2":
        seg = Conv3D(1, (2, 2, 2), activation="sigmoid", padding='same') (x)
        dis = Conv3D(1, (2, 2, 2), activation="linear", padding='same') (x)
        outputs = Concatenate()([seg, dis])
    elif output_channels in ["BCD", "BCDv2"]:
        seg = Conv3D(2, (2, 2, 2), activation="sigmoid", padding='same') (x)
        dis = Conv3D(1, (2, 2, 2), activation="linear", padding='same') (x)
        outputs = Concatenate()([seg, dis])

    model = Model(inputs=[inputs], outputs=[outputs])

    return model
