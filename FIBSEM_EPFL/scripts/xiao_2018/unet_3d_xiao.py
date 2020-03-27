import tensorflow as tf
from keras.models import Model
from keras.layers import Input, ELU, UpSampling3D, Add, Dense, Activation, Reshape, Flatten, Permute
from keras.layers.core import Dropout, Lambda, SpatialDropout3D
from keras.layers.convolutional import Conv3D, Conv3DTranspose
from keras.layers.pooling import MaxPooling3D
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
import keras 
from metrics import binary_crossentropy_weighted, jaccard_index, jaccard_index_softmax
from loss import custom_loss


def U_Net_3D_Xiao(image_shape, activation='elu', numInitChannels=16, 
             fixed_dropout=0.0, spatial_dropout=False, optimizer="sgd", 
             weighted_loss=False, lr=0.0001, num_classes=2):
    """Create 3D U-Net.

       Args:
            image_shape (array of 3 int): dimensions of the input image.

            activation (str, optional): Keras available activation type.

            numInitChannels (int, optional): number of convolution channels to 
            start with. In each downsampling/upsampling the number of filters 
            are multiplied/divided by 2.

            fixed_dropout (float, optional): dropout value to be fixed. If no 
            value is provided the default behaviour will be to select a 
            piramidal value stating from 0.1 and reaching 0.3 value.

            spatial_dropout (bool, optional): use spatial dropout instead of the
            "normal" dropout. 

            optimizer (str, optional): optimizer used to minimize the loss 
            function. Posible options: 'sgd' or 'adam'.
        
            weighted_loss (bool, optional): flag to control if a weighted loss 
            is going to be used.

            lr (float, optional): learning rate value.
    
       Returns:
            model (Keras model): model containing the U-Net created.
    """

    dinamic_dim = (None,)*(len(image_shape)-1) + (1,)

    #inputs = Input(image_shape)
    inputs = Input(dinamic_dim)
        
    x = Conv3D(3, (3, 3, 3), activation=None,
               kernel_initializer='he_normal', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ELU() (x)

    # Encoder
    s1 = residual_block(x, 32)
    x = MaxPooling3D((1, 2, 2), padding='same') (s1)
    s2 = residual_block(x, 48)
    x = MaxPooling3D((1, 2, 2), padding='same') (s2)
    s3 = residual_block(x, 64)
    x = MaxPooling3D((1, 2, 2), padding='same') (s3)
    x = residual_block(x, 80) 
   
    # Decoder 
    x = UpSampling3D((1, 2, 2)) (x)
    x = Conv3D(64, (1, 1, 1), activation=None, kernel_initializer='he_normal', 
               padding='same')(x)
    x = BatchNormalization()(x)
    x = ELU() (x)
    x = Add()([s3, x])

    x = residual_block(x, 64)
    x = UpSampling3D((1, 2, 2)) (x)
        
    # Auxiliary ouput 1 
    #a1 = UpSampling3D((1, 2, 2)) (x)    
    a1 = UpSampling3D((1, 2, 2)) (x)    
    a1 = Conv3D(num_classes, (1, 1, 1), activation=None,
               kernel_initializer='he_normal', padding='same')(a1)
    a1 = Activation('softmax')(a1)


    x = Conv3D(48, (1, 1, 1), activation=None, kernel_initializer='he_normal', 
               padding='same')(x)
    x = BatchNormalization()(x)
    x = ELU() (x)
    x = Add()([s2, x])

    x = residual_block(x, 48)
    x = UpSampling3D((1, 2, 2)) (x)
    x = Conv3D(32, (1, 1, 1), activation=None, kernel_initializer='he_normal', 
               padding='same')(x)
    x = BatchNormalization()(x)
    x = ELU() (x)

    # Auxiliary ouput 2
    #a2 = UpSampling3D((1, 2, 2)) (x)
    a2 = Conv3D(num_classes, (1, 1, 1), activation=None,
               kernel_initializer='he_normal', padding='same')(x)
    a2 = Activation('softmax')(a2)


    x = Add()([s1, x])
    x = residual_block(x, 32)
    x = Conv3D(3, (3, 3, 3), activation=None, kernel_initializer='he_normal', 
               padding='same')(x)
    x = BatchNormalization()(x)
    x = ELU() (x)

    # Adapt the output to use softmax pixel-wise 
    x = Conv3D(num_classes, (1, 1, 1), activation=None, 
               kernel_initializer='he_normal', padding='same')(x)
    outputs = Activation('softmax')(x)
   
    model = Model(inputs=[inputs], outputs=[outputs])
    
    opt = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, 
                                epsilon=1e-8, decay=0.0, amsgrad=False)
        
    model.compile(optimizer=opt, loss=custom_loss(a1, a2),
                  metrics=[jaccard_index_softmax])

    return model

def residual_block(inp_layer, channels):

    a = Conv3D(channels, (1, 1, 1), activation=None,
           kernel_initializer='he_normal', padding='same')(inp_layer)
    a = BatchNormalization()(a)
    a = ELU() (a)

    b = Conv3D(channels, (3, 3, 3), activation=None,
               kernel_initializer='he_normal', padding='same')(inp_layer)
    b = BatchNormalization()(b)
    b = ELU() (b)
    b = Conv3D(channels, (3, 3, 3), activation=None,
               kernel_initializer='he_normal', padding='same')(b)
    b = BatchNormalization()(b)
    b = ELU() (b)

    return Add()([a, b])

