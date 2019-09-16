from keras.models import Model
from keras.layers import Input
from keras.layers.core import SpatialDropout2D, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose, SeparableConv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Activation, Add, Concatenate, BatchNormalization


def residual_block(x, dim, filter_size, activation='elu', 
                   kernel_initializer='he_normal', dropout_value=0.2, bn=False,
                   first_conv_strides=1, separable_conv=False, firstBlock=False):

    # Create shorcut
    shortcut = Conv2D(dim, activation=None, kernel_size=(1, 1), 
                      strides=first_conv_strides)(x)
    
    # Main path
    if firstBlock == False:
        x = BatchNormalization()(x) if bn else x
        x = Activation( activation )(x)
    if firstBlock == True or separable_conv == False:
        x = Conv2D(dim, filter_size, strides=first_conv_strides, activation=None,
                kernel_initializer=kernel_initializer, padding='same') (x)
    else:
        x = SeparableConv2D(dim, filter_size, strides=first_conv_strides, 
                            activation=None, kernel_initializer=kernel_initializer,
                            padding='same') (x)
    x = SpatialDropout2D( dropout_value ) (x) if dropout_value else x
    x = BatchNormalization()(x) if bn else x
    x = Activation( activation )(x)
      
    if separable_conv == False:
        x = Conv2D(dim, filter_size, activation=None,
                kernel_initializer=kernel_initializer, padding='same') (x)
    else:
        x = SeparableConv2D(dim, filter_size, activation=None,
                kernel_initializer=kernel_initializer, padding='same') (x)

    # Add shortcut value to main path
    x = Add()([shortcut, x])
    
    # Add the last activation
    x = Activation( activation )(x)

    return x


def level_block(x, depth, dim, fs, ac, k, d, bn, fcs, sc, fb, mp):

    if depth > 0:
        r = residual_block(x, dim, fs, ac, k, d, bn, fcs, sc, fb)
        x = MaxPooling2D((2, 2)) (r) if mp else r
        x = level_block(x, depth-1, (dim*2), fs, ac, k, d, bn, fcs, sc, False, mp) 
        x = Conv2DTranspose(dim, (2, 2), strides=(2, 2), padding='same') (x)
        x = Concatenate()([r, x])
        x = residual_block(x, dim, fs, ac, k, d, bn, fcs, sc, False)
    else:
        x = residual_block(x, dim, fs, ac, k, d, bn, fcs, sc, False)
    return x


def ResUNet(image_shape, activation='elu', kernel_initializer='he_normal',
            dropout_value=0.2, batchnorm=False, maxpooling=True, separable=False,
            numInitChannels=16, depth=4):

    """Create the ResU-Net (with proper residual add).

       Args:
            image_shape (array of 3 int): dimensions of the input image.
            activation (str, optional): Keras available activation type.
            kernel_initializer (str, optional): Keras available kernel 
            initializer type.
            dropout_value (real value, optional): dropout value
            batchnorm (bool, optional): use batch normalization
            maxpooling (bool, optional): use max-pooling between U-Net levels 
            (otherwise use stride of 2x2).
            separable (bool, optional): use SeparableConv2D instead of Conv2D
            numInitChannels (integer value, optional): number of channels at the
            first level of U-Net

       Returns:
            model (Keras model): model containing the U-Net created.
    """

    inputs = Input((image_shape[0], image_shape[1], image_shape[2]))
    s = Lambda(lambda x: x / 255) (inputs)

    conv_strides = (1,1) if maxpooling else (2,2)

    x = level_block(s, 4, numInitChannels, 3, activation, kernel_initializer,
                    dropout_value, batchnorm, conv_strides, separable, True,
                    maxpooling)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (x)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model

