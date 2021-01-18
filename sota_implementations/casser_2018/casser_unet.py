import tensorflow as tf
from tensorflow.keras.layers import Dropout, SpatialDropout2D, Conv2D,\
                                    Conv2DTranspose, MaxPooling2D, concatenate,\
                                    ELU, BatchNormalization, Activation, \
                                    ZeroPadding2D, UpSampling2D
from tensorflow.keras import Model, Input
from metrics import binary_crossentropy_weighted, jaccard_index, \
                    jaccard_index_softmax, weighted_bce_dice_loss


def U_Net_2D(image_shape, start_filters=16, dr_rate=0.2, optimizer="adam",
             lr=0.0005):
    """Create 2D U-Net.
                                                                                
       Copied from https://github.com/mpsych/mitochondria

       Parameters
       ----------
       image_shape : 2D tuple
           Dimensions of the input image.              
                                                                                
       start_filters : int, optional
           Feature maps to start with in the first level of the unet. It will 
           be doubled on each downsampling.
                                                                           
       drop_value : float, optional
           Dropout value to be fixed.
                                                                           
       optimizer : str, optional
           Optimizer used to minimize the loss function. Posible options: 
           ``sgd`` or ``adam``.                 
                                                                           
       lr : float, optional
           Learning rate value.                          
        
       Returns
       -------                                                                 
       model : Keras model
           Model containing the U-Net.              
    """

    dinamic_dim = (None,)*(len(image_shape)-1) + (image_shape[-1],)                           
    x = Input(dinamic_dim)                                                      
    #inputs = Input((img_rows, img_cols, MEMORY))
    inputs = x
    
        
    # ENCODER
    conv1 = Conv2D(start_filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(start_filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(start_filters*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(start_filters*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(start_filters*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(start_filters*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(start_filters*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(start_filters*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(dr_rate)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    # BOTTLENECK
    conv5 = Conv2D(start_filters*16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(start_filters*16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(dr_rate)(conv5)

    # DECODER
    up6 = Conv2D(start_filters*8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6])
    conv6 = Conv2D(start_filters*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(start_filters*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(start_filters*4, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7])
    conv7 = Conv2D(start_filters*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(start_filters*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(start_filters*4, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8])
    conv8 = Conv2D(start_filters*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(start_filters*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(start_filters, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9])
    conv9 = Conv2D(start_filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(start_filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    outputs = Conv2D(1, 1, activation = 'sigmoid')(conv9)
    
    model = Model(inputs=[inputs], outputs=[outputs])

    # Select the optimizer
    if optimizer == "sgd":
        opt = tf.keras.optimizers.SGD(
            lr=lr, momentum=0.99, decay=0.0, nesterov=False)
    elif optimizer == "adam":
        opt = tf.keras.optimizers.Adam(
            lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,
            amsgrad=False)
    else:
        raise ValueError("Error: optimizer value must be 'sgd' or 'adam'")

    # Compile the model
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[jaccard_index])

    return model

