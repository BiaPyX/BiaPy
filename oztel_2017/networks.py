import tensorflow as tf
import sys
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Lambda, Conv2D, Conv2DTranspose, \
                                    MaxPooling2D, concatenate, Add
from metrics import jaccard_index_softmax, jaccard_index
from BilinearUpSampling import *
from tensorflow.keras.applications import VGG16
from tensorflow.keras.regularizers import l2


def FCN_VGG16(image_shape, activation='relu', lr=0.1, 
              weight_decay=0.01, optimizer="adam"):

    dinamic_dim = (None,)*(len(image_shape)-1) + (1,)
    inputs = Input(dinamic_dim, name="input")
    #inputs = Input((image_shape[0], image_shape[1], image_shape[2]))

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', 
               name='block1_conv1')(inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', 
               name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', 
               name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', 
               name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', 
               name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', 
               name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', 
               name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', 
               name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', 
               name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', 
               name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', 
               name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', 
               name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Last convolutional block
    x = Conv2D(4096, (3, 3), activation='relu', padding='same', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(1, (1, 1), kernel_initializer='he_normal', activation='sigmoid',
               padding='valid', strides=(1, 1))(x)

    outputs = BilinearUpSampling2D(target_size=image_shape)(x)

    model_fcn = Model(inputs=[inputs], outputs=[outputs])

#    # Load the weights of VGG model
#    for layer in model_vgg.layers:
#        name = layer.name
#        
#        # Break when the last convolutional block is reached, as it differs with
#        # the VGG and any weights could be loaded
#        if 'fc1' == name: break
#
#        if 'conv' in name:
#            model_fcn.get_layer(name).set_weights(
#                model_vgg.get_layer(name).get_weights())

    # Select the optimizer
    if optimizer == "sgd":
        opt = tf.keras.optimizers.SGD(lr=lr, momentum=0.99, decay=0.0,
                                      nesterov=False)
    elif optimizer == "adam":
        opt = tf.keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999,
                                       epsilon=None, decay=0.0, amsgrad=False)
    else:
        raise ValueError("Error: optimizer value must be 'sgd' or 'adam'")

    # Compile the model
    model_fcn.compile(optimizer=opt, loss='binary_crossentropy',
                      metrics=[jaccard_index])

    return model_fcn
