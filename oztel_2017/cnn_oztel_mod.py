from keras.models import Model
from keras.layers import Input, Dense
from keras.layers.core import Dropout, Lambda, SpatialDropout2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
import keras 
from metrics import binary_crossentropy_weighted, jaccard_index, \
                    weighted_bce_dice_loss

def cnn_oztel_2017(image_shape, activation='relu',lr=0.1):
    """Create the U-Net

       Args:
            image_shape (array of 3 int): dimensions of the input image.

            activation (str, optional): Keras available activation type.

            lr (float, optional): learning rate value.
        
       Returns:
            model (Keras model): model containing the U-Net created.
    """

    inputs = Input((image_shape[0], image_shape[1], image_shape[2]))
        
    s = Lambda(lambda x: x / 255) (inputs)

    o = Conv2D(32, (5, 5), activation=activation,
                kernel_initializer='he_normal', padding='same') (s)
    print(o.output_shape)
    p1 = MaxPooling2D((2, 2)) (o)
    o = Conv2D(32, (5, 5), activation=activation,
                kernel_initializer='he_normal', padding='same') (p1)
    print(o.output_shape)
    p2 = MaxPooling2D((2, 2)) (o)
    o = Conv2D(64, (5, 5), activation=activation,
                kernel_initializer='he_normal', padding='same') (p2)
    print(o.output_shape)
    p3 = MaxPooling2D((2, 2)) (o)
    o = Conv2D(64, (4, 4), activation=activation,
                kernel_initializer='he_normal', padding='same') (p3)
    print(o.output_shape)
    o = Dense(2, activation=activation) (o)
    print(o.output_shape)

    o = Conv2DTranspose( n_classes , kernel_size=(4,4) ,  strides=(8,8) , use_bias=False )(o)
    print(o.output_shape)
    o = Cropping2D(((1, 1), (1, 1)))(o)
    print(o.output_shape)
    outputs = Softmax(axis=3)(o)
    print(outputs.output_shape)
 
    # Loss type 
    model = Model(inputs=[inputs], outputs=[outputs])
    
    # Select the optimizer
    opt = keras.optimizers.SGD(lr=lr, momentum=0.99, decay=0.0, nesterov=False)
        
    # Compile the model
    model.compile(optimizer=opt, loss='binary_crossentropy', 
                  metrics=['accuracy'])

    return model
