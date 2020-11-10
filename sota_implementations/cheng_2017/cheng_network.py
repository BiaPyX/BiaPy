import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization, Dropout, Lambda,\
                                    SpatialDropout2D, Conv2D, Conv2DTranspose, \
                                    MaxPooling2D, concatenate, Add
from tensorflow.keras.layers import PReLU
from tensorflow.keras.regularizers import l2
from metrics import jaccard_index_softmax
from loss import jaccard_loss_cheng2017
from StochasticDownsampling2D import StochasticDownsampling2D


def asymmetric_network(image_shape, numInitChannels=16, fixed_dropout=0.0, 
                       optimizer="sgd", lr=0.001, t_downsmp_layer=4):
    """Create the assymetric network proposed in Cheng et al.

       Parameters
       ----------
       image_shape : array of 3 int
           Dimensions of the input image.

       numInitChannels : int, optional
           Number of convolution channels to start with. In each
           downsampling/upsampling the number of filters are multiplied/divided 
           by ``2``.

       fixed_dropout : float, optional
           Dropout value to be fixed. If no value is provided the default
           behaviour will be to select a piramidal value stating from 0.1 and
           reaching 0.3 value.

       optimizer : str, optional
           Optimizer used to minimize the loss function. Posible options: ``sgd``
           or ``adam``.

       lr : float, optional
           Learning rate value.
        
       t_downsmp_layer : int, optional
           Degree of randomness in the sampling pattern which corresponds to the
           ``t`` value defined in the paper for the proposed stochastic 
           downsampling layer. 

       Returns
       -------
       model : Keras model
          Asymmetric network proposed in Cheng et al. model.


       Here is a picture of the network extracted from the original paper:
                                                                                
       .. image:: img/cheng_network.png                                         
           :width: 90%                                                         
           :align: center
    """

    dinamic_dim = (None,)*(len(image_shape)-1) + (1,)
    inputs = Input(dinamic_dim)
        
    # Input block
    channels = numInitChannels
    c1 = Conv2D(channels, (3, 3), activation=None, strides=(2, 2),
                kernel_initializer='he_normal', padding='same',
                kernel_regularizer=l2(0.01)) (inputs)
    m1 = MaxPooling2D(pool_size=(2, 2))(inputs)
    x = concatenate([c1,m1])
       
    # First encode block sequence
    for i in range(2):
        channels += 8
        x = encode_block(x, channels, t_downsmp_layer=t_downsmp_layer, 
                         fixed_dropout=fixed_dropout)

    # 1st downsample block
    channels += 8
    x = encode_block(
        x, channels, downsample=True, t_downsmp_layer=t_downsmp_layer, 
        fixed_dropout=fixed_dropout)

    # Second encode block sequence
    for i in range(3):
        channels += 8
        x = encode_block(x, channels, t_downsmp_layer=t_downsmp_layer,
                         fixed_dropout=fixed_dropout)

    # 2nd downsample block
    channels += 8
    x = encode_block(
        x, channels, downsample=True, t_downsmp_layer=t_downsmp_layer,
        fixed_dropout=fixed_dropout)
    
    # Third encode block sequence
    for i in range(6):
        channels += 8
        x = encode_block(x, channels, t_downsmp_layer=t_downsmp_layer,
                         fixed_dropout=fixed_dropout)

    # 1st upsample block 
    channels = 64
    x = decode_block(x, channels, upsample=True) 

    # First decode block sequence
    for i in range(4):
        x = decode_block(x, channels)

    # 2nd upsample block
    channels = int(channels/2)
    x = decode_block(x, channels, upsample=True)

    # Second decode block sequence 
    for i in range(2):                                                          
        x = decode_block(x, channels)

    # Last transpose conv 
    outputs = Conv2DTranspose(2, (2, 2), activation="softmax", strides=(2, 2),
                              kernel_regularizer=l2(0.01)) (x)

    model = Model(inputs=[inputs], outputs=[outputs])
    
    if optimizer == "sgd":
        opt = tf.keras.optimizers.SGD(lr=lr, momentum=0.90, decay=0.0, 
                                      nesterov=False)
    elif optimizer == "adam":
        opt = tf.keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, 
                                       epsilon=None, decay=0.0, amsgrad=False)
    else:
        raise ValueError("Error: optimizer value must be 'sgd' or 'adam'")

    model.compile(optimizer=opt, loss=jaccard_loss_cheng2017,
                  metrics=[jaccard_index_softmax])

    return model


def pad_depth(x, desired_channels):
    """ Zero padding to use in case the feature map changes in shortcut 
        connections.
    """
    y = K.zeros_like(x)
    new_channels = desired_channels - x.shape.as_list()[-1]
    y = y[:,:,:,:new_channels]
    return concatenate([x,y])


def encode_block(inp_layer, channels, t_downsmp_layer=4, downsample=False, 
                 fixed_dropout=0.1):
    """Encode block defined in Cheng et al.     
        
       Parameters
       ----------   
       inp_layer : Keras layer
           Input layer.
       
       channels : int, optional
           Feature maps to define in Conv layers.
        
       t_downsmp_layer : int, optional
           ``t`` value defined in the paper for the proposed stochastic 
           downsampling layer. 
    
       downsample : bool, optional
           To make a downsampling. Blue blocks in the encoding part. 

       fixed_dropout : float, optional
           Dropout value.

       Returns
       -------
       out : Keras layer
           Last layer of the block.
    """
    if downsample == True:
        if inp_layer.shape[1] is None:
            tf.print("MaxPooling2D")
            print("MaxPooling2D")
            shortcut_padded = MaxPooling2D((2,2)) (inp_layer)
        else:
            tf.print("StochasticDownsampling2D")
            print("StochasticDownsampling2D")
            shortcut_padded = StochasticDownsampling2D() (inp_layer, t_downsmp_layer)

        shortcut_padded = Conv2D(channels, (1, 1), activation=None, 
                                 kernel_regularizer=l2(0.01)) (shortcut_padded)
    else:
        shortcut_padded = Lambda(
            pad_depth, arguments={'desired_channels':channels})(inp_layer)
   
    x = BatchNormalization()(inp_layer)
    x = PReLU(shared_axes=[1, 2]) (x)
    if downsample == True:
        c1 = Conv2D(int(channels/2), (1, 3), activation=None, strides=(2, 2),
                   kernel_initializer='he_normal', padding='same',
                   kernel_regularizer=l2(0.01)) (x)
        c2 = Conv2D(int(channels/2), (3, 1), activation=None, strides=(2, 2),
                   kernel_initializer='he_normal', padding='same',
                   kernel_regularizer=l2(0.01)) (x)
        x = concatenate([c1,c2])
    else:
        c1 = Conv2D(int(channels/2), (1, 3), activation=None,
                   kernel_initializer='he_normal', padding='same',
                   kernel_regularizer=l2(0.01)) (x)
        c2 = Conv2D(int(channels/2), (3, 1), activation=None,
                   kernel_initializer='he_normal', padding='same',
                   kernel_regularizer=l2(0.01)) (x)
        x = concatenate([c1,c2])
    
    x = Dropout(fixed_dropout)(x)
    x = BatchNormalization()(x)
    x = PReLU(shared_axes=[1, 2]) (x)

    c1 = Conv2D(int(channels/2), (1, 3), activation=None,
               kernel_initializer='he_normal', padding='same',
               kernel_regularizer=l2(0.01)) (x)
    c2 = Conv2D(int(channels/2), (3, 1), activation=None,
               kernel_initializer='he_normal', padding='same',
               kernel_regularizer=l2(0.01)) (x)
    x = concatenate([c1,c2])

    x = Add()([shortcut_padded, x])
    return x


def decode_block(inp_layer, channels, upsample=False):
    """Encode block defined in Cheng et al.                                     
                                                                                
       Parameters                                                               
       ----------                                                               
       inp_layer : Keras layer                                                  
           Input layer.                                                         
                                                                                
       channels : int, optional                                                 
           Feature maps to define in Conv layers.                               
                                                                                
       upsample : bool, optional                                              
           To make an upsampling. Blue blocks in the decoding part.
                                                                                
       Returns                                                                  
       -------                                                                  
       out : Keras layer                                                        
           Last layer of the block.                                             
    """ 
    if upsample == True:    
        x = Conv2DTranspose(channels, (3, 3), activation=None, 
                            strides=(2, 2), padding='same',
                            kernel_regularizer=l2(0.01)) (inp_layer)
    else:
        shortcut = Conv2D(channels, kernel_size=(1, 1), padding='same',
                          kernel_regularizer=l2(0.01))(inp_layer)
        x = Conv2D(int(channels/4), (1, 1), activation=None,
                   kernel_initializer='he_normal', padding='same',
                   kernel_regularizer=l2(0.01))(inp_layer)
        x = Conv2D(int(channels/4), (3, 3), activation=None,                           
                   kernel_initializer='he_normal', padding='same',
                   kernel_regularizer=l2(0.01))(x)
        x = Conv2D(channels, (1, 1), activation=None,                           
                   kernel_initializer='he_normal', padding='same',
                   kernel_regularizer=l2(0.01))(x)
        x = Add()([shortcut, x])           
    return x 

