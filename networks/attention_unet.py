from keras import backend as K
from keras.models import Model
from keras.layers import Input, Activation, UpSampling2D, BatchNormalization
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate, add, multiply


def AttnGatingBlock( x, g, inter_shape ):
    shape_x = K.int_shape(x)  # 32
    shape_g = K.int_shape(g)  # 16

    theta_x = Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)  # 16
    shape_theta_x = K.int_shape(theta_x)

    phi_g = Conv2D(inter_shape, (1, 1), padding='same')(g)
    upsample_g = Conv2DTranspose(inter_shape, (3, 3),strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),padding='same')(phi_g)  # 16

    concat_xg = add([upsample_g, theta_x])
    act_xg = Activation('relu')(concat_xg)
    psi = Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32

    # my_repeat=Lambda(lambda xinput:K.repeat_elements(xinput[0],shape_x[1],axis=1))
    # upsample_psi=my_repeat([upsample_psi])
    #upsample_psi = self.expend_as(upsample_psi, shape_x[3])
    upsample_psi = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': shape_x[3]})(upsample_psi)
    y = multiply([upsample_psi, x])

    # print(K.is_keras_tensor(upsample_psi))

    result = Conv2D(shape_x[3], (1, 1), padding='same')(y)
    result_bn = BatchNormalization()(result)
    return result_bn

def UnetGatingSignal( input, is_batchnorm=False):
    shape = K.int_shape(input)
    x = Conv2D(shape[3] * 2, (1, 1), strides=(1, 1), padding="same")(input)
    if is_batchnorm:
	    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x
    
def Attention_U_Net_BN(image_shape, activation='elu', numInitChannels=16):
    """Create the Attention U-Net

       Args:
            image_shape (array of 3 int): dimensions of the input image.
            activation (str, optional): Keras available activation type.

       Returns:
            model (Keras model): model containing the U-Net created.
    """
    inputs = Input((image_shape[0], image_shape[1], image_shape[2]))
    
    c1 = Conv2D(numInitChannels, (3, 3), activation=None,
                kernel_initializer='he_normal', padding='same') (inputs)
    c1 = BatchNormalization()(c1)
    c1 = Activation( activation )(c1)
    c1 = Conv2D(numInitChannels, (3, 3), activation=None,
                kernel_initializer='he_normal', padding='same') (c1)
    c1 = BatchNormalization()(c1)
    c1 = Activation( activation )(c1)            
    p1 = MaxPooling2D((2, 2)) (c1)
    
    c2 = Conv2D(numInitChannels*2, (3, 3), activation=None,
                kernel_initializer='he_normal', padding='same') (p1)
    c2 = BatchNormalization()(c2)
    c2 = Activation( activation )(c2)
    c2 = Conv2D(numInitChannels*2, (3, 3), activation=None,
                kernel_initializer='he_normal', padding='same') (c2)
    c2 = BatchNormalization()(c2)
    c2 = Activation( activation )(c2)
    p2 = MaxPooling2D((2, 2)) (c2)
    
    c3 = Conv2D(numInitChannels*4, (3, 3), activation=None,
                kernel_initializer='he_normal', padding='same') (p2)
    c3 = BatchNormalization()(c3)
    c3 = Activation( activation )(c3)
    c3 = Conv2D(numInitChannels*4, (3, 3), activation=None,
                kernel_initializer='he_normal', padding='same') (c3)
    c3 = BatchNormalization()(c3)
    c3 = Activation( activation )(c3)
    p3 = MaxPooling2D((2, 2)) (c3)
    
    c4 = Conv2D(numInitChannels*8, (3, 3), activation=None,
                kernel_initializer='he_normal', padding='same') (p3)
    c4 = BatchNormalization()(c4)
    c4 = Activation( activation )(c4)            
    
    c4 = Conv2D(numInitChannels*8, (3, 3), activation=None,
                kernel_initializer='he_normal', padding='same') (c4)
    c4 = BatchNormalization()(c4)
    c4 = Activation( activation )(c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    
    # bridge
    c5 = Conv2D(numInitChannels*16, (3, 3), activation=None,
                kernel_initializer='he_normal', padding='same') (p4)
    c5 = BatchNormalization()(c5)
    c5 = Activation( activation )(c5)
    c5 = Conv2D(numInitChannels*16, (3, 3), activation=None,
                kernel_initializer='he_normal', padding='same') (c5)
    c5 = BatchNormalization()(c5)
    c5 = Activation( activation )(c5)
    
    # up path
    
    gating = UnetGatingSignal( c5, is_batchnorm=True)
    attn_1 = AttnGatingBlock( c4, gating, numInitChannels*16)
    
    u6 = Conv2DTranspose(numInitChannels*8, (2, 2), strides=(2, 2),
                         padding='same') (c5)
    u6 = concatenate([u6, attn_1])
    
  
    c6 = Conv2D(numInitChannels*8, (3, 3), activation=None,
                kernel_initializer='he_normal', padding='same') (u6)
    c6 = BatchNormalization()(c6)
    c6 = Activation( activation )(c6)
    c6 = Conv2D(numInitChannels*8, (3, 3), activation=None,
                kernel_initializer='he_normal', padding='same') (c6)
    c6 = BatchNormalization()(c6)
    c6 = Activation( activation )(c6)
    
    gating = UnetGatingSignal( u6, is_batchnorm=True)
    attn_2 = AttnGatingBlock( c3, gating, numInitChannels*8)
    
    u7 = Conv2DTranspose(numInitChannels*4, (2, 2), strides=(2, 2),
                         padding='same') (c6)
    u7 = concatenate([u7, attn_2])
    
        
    c7 = Conv2D(numInitChannels*4, (3, 3), activation=None,
                kernel_initializer='he_normal', padding='same') (u7)
    c7 = BatchNormalization()(c7)
    c7 = Activation( activation )(c7)
    c7 = Conv2D(numInitChannels*4, (3, 3), activation=None,
                kernel_initializer='he_normal', padding='same') (c7)
    c7 = BatchNormalization()(c7)
    c7 = Activation( activation )(c7)
    
    gating = UnetGatingSignal( u7, is_batchnorm=True)
    attn_3 = AttnGatingBlock( c2, gating, numInitChannels*4)
    
    u8 = Conv2DTranspose(numInitChannels*2, (2, 2), strides=(2, 2),
                         padding='same') (c7)
    u8 = concatenate([u8, attn_3])
    
    
    
    c8 = Conv2D(numInitChannels*2, (3, 3), activation=None,
                kernel_initializer='he_normal', padding='same') (u8)
    c8 = BatchNormalization()(c8)
    c8 = Activation( activation )(c8)
    c8 = Conv2D(numInitChannels*2, (3, 3), activation=None,
                kernel_initializer='he_normal', padding='same') (c8)
    c8 = BatchNormalization()(c8)
    c8 = Activation( activation )(c8)
    
    gating = UnetGatingSignal( u8, is_batchnorm=True)
    attn_4 = AttnGatingBlock( c1, gating, numInitChannels*2)
    
    u9 = Conv2DTranspose(numInitChannels, (2, 2), strides=(2, 2),
                         padding='same') (c8)
    u9 = concatenate([u9, attn_4], axis=3)
    
    c9 = Conv2D(numInitChannels, (3, 3), activation=None,
                kernel_initializer='he_normal', padding='same') (u9)
    c9 = BatchNormalization()(c9)
    c9 = Activation( activation )(c9)
    c9 = Conv2D(numInitChannels, (3, 3), activation=None,
                kernel_initializer='he_normal', padding='same') (c9)
    c9 = BatchNormalization()(c9)
    c9 = Activation( activation )(c9)
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model

def Attention_U_Net(image_shape, activation='elu', numInitChannels=16):
    """Create the Attention U-Net

       Args:
            image_shape (array of 3 int): dimensions of the input image.
            activation (str, optional): Keras available activation type.

       Returns:
            model (Keras model): model containing the U-Net created.
    """
    inputs = Input((image_shape[0], image_shape[1], image_shape[2]))
    s = Lambda(lambda x: x / 255) (inputs)
    
    c1 = Conv2D(numInitChannels, (3, 3), activation=activation,
                kernel_initializer='he_normal', padding='same') (s)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(numInitChannels, (3, 3), activation=activation,
                kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)
    
    c2 = Conv2D(numInitChannels*2, (3, 3), activation=activation,
                kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(numInitChannels*2, (3, 3), activation=activation,
                kernel_initializer='he_normal', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)
    
    c3 = Conv2D(numInitChannels*4, (3, 3), activation=activation,
                kernel_initializer='he_normal', padding='same') (p2)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(numInitChannels*4, (3, 3), activation=activation,
                kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)
    
    c4 = Conv2D(numInitChannels*8, (3, 3), activation=activation,
                kernel_initializer='he_normal', padding='same') (p3)
    c4 = Dropout(0.2) (c4)
    c4 = Conv2D(numInitChannels*8, (3, 3), activation=activation,
                kernel_initializer='he_normal', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    
    # bridge
    c5 = Conv2D(numInitChannels*16, (3, 3), activation=activation,
                kernel_initializer='he_normal', padding='same') (p4)
    c5 = Dropout(0.3) (c5)
    c5 = Conv2D(numInitChannels*16, (3, 3), activation=activation,
                kernel_initializer='he_normal', padding='same') (c5)
    
    # up path
    
    gating = UnetGatingSignal( c5, is_batchnorm=True)
    attn_1 = AttnGatingBlock( c4, gating, numInitChannels*16)
    
    u6 = Conv2DTranspose(numInitChannels*8, (2, 2), strides=(2, 2),
                         padding='same') (c5)
    u6 = concatenate([u6, attn_1])
    
  
    c6 = Conv2D(numInitChannels*8, (3, 3), activation=activation,
                kernel_initializer='he_normal', padding='same') (u6)
    c6 = Dropout(0.2) (c6)
    c6 = Conv2D(numInitChannels*8, (3, 3), activation=activation,
                kernel_initializer='he_normal', padding='same') (c6)
    
    gating = UnetGatingSignal( u6, is_batchnorm=True)
    attn_2 = AttnGatingBlock( c3, gating, numInitChannels*8)
    
    u7 = Conv2DTranspose(numInitChannels*4, (2, 2), strides=(2, 2),
                         padding='same') (c6)
    u7 = concatenate([u7, attn_2])
    
        
    c7 = Conv2D(numInitChannels*4, (3, 3), activation=activation,
                kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv2D(numInitChannels*4, (3, 3), activation=activation,
                kernel_initializer='he_normal', padding='same') (c7)
    
    gating = UnetGatingSignal( u7, is_batchnorm=True)
    attn_3 = AttnGatingBlock( c2, gating, numInitChannels*4)
    
    u8 = Conv2DTranspose(numInitChannels*2, (2, 2), strides=(2, 2),
                         padding='same') (c7)
    u8 = concatenate([u8, attn_3])
    
    
    
    c8 = Conv2D(numInitChannels*2, (3, 3), activation=activation,
                kernel_initializer='he_normal', padding='same') (u8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv2D(numInitChannels*2, (3, 3), activation=activation,
                kernel_initializer='he_normal', padding='same') (c8)
    
    gating = UnetGatingSignal( u8, is_batchnorm=True)
    attn_4 = AttnGatingBlock( c1, gating, numInitChannels*2)
    
    u9 = Conv2DTranspose(numInitChannels, (2, 2), strides=(2, 2),
                         padding='same') (c8)
    u9 = concatenate([u9, attn_4], axis=3)
    
    c9 = Conv2D(numInitChannels, (3, 3), activation=activation,
                kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv2D(numInitChannels, (3, 3), activation=activation,
                kernel_initializer='he_normal', padding='same') (c9)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model

