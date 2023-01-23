import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Add, Cropping2D, Dropout


def FCN32_VGG16(image_shape, n_classes=2):
    """Create FCN32 network based on a VGG16.

       Parameters
       ----------
       image_shape : 2D tuple
           Dimensions of the input image.

       n_classes: int, optional
           Number of classes.

       Returns
       -------
       model : Keras model
           Model containing the FCN32.


       Calling this function with its default parameters returns the following network:

       .. image:: ../../img/fcn32.png
           :width: 100%
           :align: center

       Image created with `PlotNeuralNet <https://github.com/HarisIqbal88/PlotNeuralNet>`_.
    """

    
    dinamic_dim = (None,)*(len(image_shape)-1) + (image_shape[-1],)
    inputs = Input(dinamic_dim, name="input")

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Last convolutional block
    x = Conv2D(4096, (3, 3), activation='relu', padding='same', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(n_classes, (1, 1), kernel_initializer='he_normal', activation='sigmoid', padding='valid', strides=(1, 1))(x)

    outputs = tf.keras.layers.UpSampling2D(size=(32, 32), interpolation='bilinear')(x)
    
    model_fcn = Model(inputs=[inputs], outputs=[outputs])

    return model_fcn


def FCN8_VGG16(image_shape, n_classes=2):
    """Create FCN8 network based on a VGG16.

       Parameters
       ----------
       image_shape : 2D tuple
           Dimensions of the input image.

       n_classes: int, optional
           Number of classes.

       Returns
       -------
       model : Keras model
           Model containing the FCN8.


       Calling this function with its default parameters returns the following network:

       .. image:: ../../img/fcn8.png
           :width: 100%
           :align: center

       Image created with `PlotNeuralNet <https://github.com/HarisIqbal88/PlotNeuralNet>`_.
    """
    
    dinamic_dim = (None,)*(len(image_shape)-1) + (1,)
    inputs = Input(dinamic_dim, name="input")

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    p3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(p3)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    p4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(p4)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Last convolutional block
    x = Conv2D(4096, (3, 3), activation='relu', padding='same', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(n_classes, (1, 1), kernel_initializer='he_normal')(x)

    u1 = Conv2DTranspose(n_classes, kernel_size=(4, 4), strides=(2, 2), use_bias=False) (x)
    u1 = Cropping2D(cropping=((0, 2), (0, 2))) (u1)
    u_p4 = Conv2D(n_classes, (1, 1), kernel_initializer='he_normal')(p4)
    o1 = Add() ([u1, u_p4])

    u2 = Conv2DTranspose(n_classes, kernel_size=(4, 4), strides=(2, 2), use_bias=False) (o1)
    u2 = Cropping2D(cropping=((0, 2), (0, 2))) (u2)
    u_p3 = Conv2D(n_classes, (1, 1), kernel_initializer='he_normal')(p3)
    o2 = Add() ([u2, u_p3])

    outputs = Conv2DTranspose(n_classes, kernel_size=(16, 16), strides=(8, 8), use_bias=False, padding='same',
        activation="sigmoid") (o2)

    model_fcn = Model(inputs=[inputs], outputs=[outputs])

    return model_fcn
