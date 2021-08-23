"""
Code fully extracted from `MultiResUNet <https://github.com/nibtehaz/MultiResUNet>`_.
"""

from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, 
                                     Activation, add)
from tensorflow.keras.models import Model


def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', name=None):
    """2D Convolutional layers.
    
       Parameters
       ----------
       x : Keras layer
           Input layer.

       filters : int
           Number of filters.

       num_row : int
           Number of rows in filters.

       num_col : int
           Number of columns in filters.

       padding : str, optional
           Mode of padding. 

       strides : tuple, optional
           Stride of convolution operation.

       activation : str, optional
           Activation function.
  
       name : str, optional
           Name of the layer.
    
       Returns
       -------
       x : Keras layer
           Output layer.
    """

    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(x)
    x = BatchNormalization(axis=3, scale=False)(x)

    if(activation == None):
        return x

    x = Activation(activation, name=name)(x)

    return x


def trans_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(2, 2), name=None):
    """2D Transposed Convolutional layers.
    
       Parameters
       ----------
       x : keras layer
           Input layer.

       filters : int                                                            
           Number of filters.                                                   
                                                                                
       num_row : int                                                            
           Number of rows in filters.                                           
                                                                                
       num_col : int                                                            
           Number of columns in filters.                                        
                                                                                
       padding : str, optional                                                  
           Mode of padding.                                                     
                                                                                
       strides : tuple, optional                                                
           Stride of convolution operation.                                     
                                                                                
       name : str, optional                                                     
           Name of the layer.
    
       Returns
       -------
       x : Keras layer
           Output layer.
    """

    x = Conv2DTranspose(filters, (num_row, num_col), strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3, scale=False)(x)
    
    return x


def MultiResBlock(U, inp, alpha = 1.67):
    """MultiRes Block.
    
       Parameters
       ----------
       U : int
           Number of filters in a corrsponding UNet stage.

       inp : Keras layer
           Input layer.

       Returns
       -------
       out : Keras layer
           Output layer.
    """

    W = alpha * U

    shortcut = inp

    shortcut = conv2d_bn(shortcut, int(W*0.167) + int(W*0.333) +
                         int(W*0.5), 1, 1, activation=None, padding='same')

    conv3x3 = conv2d_bn(inp, int(W*0.167), 3, 3,
                        activation='relu', padding='same')

    conv5x5 = conv2d_bn(conv3x3, int(W*0.333), 3, 3,
                        activation='relu', padding='same')

    conv7x7 = conv2d_bn(conv5x5, int(W*0.5), 3, 3,
                        activation='relu', padding='same')

    out = concatenate([conv3x3, conv5x5, conv7x7], axis=3)
    out = BatchNormalization(axis=3)(out)

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)

    return out


def ResPath(filters, length, inp):
    """ResPath.
    
       Parameters
       ----------
       filters : int
           Description.

       length : int
           Length of ResPath.

       inp : Keras layer
           Input layer.
    
       Returns
       -------
       out : Keras layer
           Output layer.
    """

    shortcut = inp
    shortcut = conv2d_bn(shortcut, filters, 1, 1,
                         activation=None, padding='same')

    out = conv2d_bn(inp, filters, 3, 3, activation='relu', padding='same')

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)

    for i in range(length-1):

        shortcut = out
        shortcut = conv2d_bn(shortcut, filters, 1, 1,
                             activation=None, padding='same')

        out = conv2d_bn(out, filters, 3, 3, activation='relu', padding='same')

        out = add([shortcut, out])
        out = Activation('relu')(out)
        out = BatchNormalization(axis=3)(out)

    return out


def MultiResUnet(height, width, n_channels):
    """MultiResUNet.
    
       Parameters
       ----------
       height : int
           Height of image.

       width : int
           Width of image.

       n_channels : int
           Number of channels in image.
    
       Returns
       -------
       model : Keras model
           MultiResUNet model.
    """

    inputs = Input((height, width, n_channels))

    mresblock1 = MultiResBlock(32, inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(mresblock1)
    mresblock1 = ResPath(32, 4, mresblock1)

    mresblock2 = MultiResBlock(32*2, pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(mresblock2)
    mresblock2 = ResPath(32*2, 3, mresblock2)

    mresblock3 = MultiResBlock(32*4, pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(mresblock3)
    mresblock3 = ResPath(32*4, 2, mresblock3)

    mresblock4 = MultiResBlock(32*8, pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(mresblock4)
    mresblock4 = ResPath(32*8, 1, mresblock4)

    mresblock5 = MultiResBlock(32*16, pool4)

    up6 = concatenate([Conv2DTranspose(
        32*8, (2, 2), strides=(2, 2), padding='same')(mresblock5), mresblock4], axis=3)
    mresblock6 = MultiResBlock(32*8, up6)

    up7 = concatenate([Conv2DTranspose(
        32*4, (2, 2), strides=(2, 2), padding='same')(mresblock6), mresblock3], axis=3)
    mresblock7 = MultiResBlock(32*4, up7)

    up8 = concatenate([Conv2DTranspose(
        32*2, (2, 2), strides=(2, 2), padding='same')(mresblock7), mresblock2], axis=3)
    mresblock8 = MultiResBlock(32*2, up8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(
        2, 2), padding='same')(mresblock8), mresblock1], axis=3)
    mresblock9 = MultiResBlock(32, up9)

    conv10 = conv2d_bn(mresblock9, 1, 1, 1, activation='sigmoid')
    
    model = Model(inputs=[inputs], outputs=[conv10])

    return model
   
