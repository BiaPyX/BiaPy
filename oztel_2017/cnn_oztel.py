from keras.models import Model
from keras.layers import Input, Dense, Cropping2D, Add, Activation
from keras.layers.core import Dropout, Lambda, SpatialDropout2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
import keras 
from metrics import binary_crossentropy_weighted, jaccard_index, \
                    weighted_bce_dice_loss

import tensorflow as tf 
def cnn_oztel_2017(image_shape, activation='relu',lr=0.1):
    """Create the U-Net

       Args:
            image_shape (array of 3 int): dimensions of the input image.

            activation (str, optional): Keras available activation type.

            lr (float, optional): learning rate value.
        
       Returns:
            model (Keras model): model containing the U-Net created.
    """
    dinamic_dim = (None,)*(len(image_shape)-1) + (1,)
    inputs = Input(dinamic_dim)
    #inputs = Input((image_shape[0], image_shape[1], image_shape[2]))
        
    s = Lambda(lambda x: x / 255) (inputs)

    o = Conv2D(32, (5, 5), activation=activation,
                kernel_initializer='he_normal', padding='same') (s)
    p1 = MaxPooling2D((2, 2)) (o)
    o = Conv2D(32, (5, 5), activation=activation,
                kernel_initializer='he_normal', padding='same') (p1)
    p2 = MaxPooling2D((2, 2)) (o)
    o = Conv2D(64, (5, 5), activation=activation,
                kernel_initializer='he_normal', padding='same') (p2)
    p3 = MaxPooling2D((2, 2)) (o)
    o = Conv2D(64, (4, 4), activation=activation,
               kernel_initializer='he_normal', padding='same') (p3)

    o = Conv2D(2, (1 ,1), activation=activation, kernel_initializer='he_normal', padding='same')(o) 
    o = Conv2DTranspose(2, kernel_size=(4,4), strides=(2,2), kernel_initializer='he_normal')(o)
    o = Cropping2D(((1, 1), (1, 1)))(o)

    o2 = Conv2D(2,  (1, 1) ,activation = 'relu')(p2)
    o = Add()([o , o2])

    o = Conv2DTranspose(2 , kernel_size=(4,4) ,  strides=(2,2) , use_bias=False )(o)
    o = Cropping2D(((1, 1), (1, 1)))(o)
    
    o1 = Conv2D(2,  (1, 1) ,activation = 'relu')(p1)
    o = Add()([o , o1])

    o = Conv2DTranspose(2, kernel_size=(2,2) ,  strides=(2,2) , use_bias=False )(o)
    #o = Cropping2D(((4, 4), (4, 4)))(o)

    outputs = Activation('softmax')(o)
 
    # Loss type 
    model = Model(inputs=[inputs], outputs=[outputs])
    
    # Select the optimizer
    opt = keras.optimizers.SGD(lr=lr, momentum=0.99, decay=0.0, nesterov=False)
        
    # Compile the model
    model.compile(optimizer=opt, loss='binary_crossentropy', 
                  metrics=['accuracy'])

    return model
