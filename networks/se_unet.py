import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dropout, SpatialDropout2D, Conv2D,\
                                    Conv2DTranspose, MaxPooling2D, concatenate,\
                                    ELU, BatchNormalization, Activation, \
                                    ZeroPadding2D, GlobalAveragePooling2D, \
                                    Reshape, Dense, multiply, add, Permute
from tensorflow.keras import Model, Input
from metrics import binary_crossentropy_weighted, jaccard_index, \
                    weighted_bce_dice_loss


def SE_U_Net_2D(image_shape, activation='elu', feature_maps=[32, 64, 128, 256, 512], 
             depth=4, drop_values=[0.1,0.1,0.2,0.2,0.3], spatial_dropout=False, 
             batch_norm=False, k_init='he_normal', loss_type="bce", 
             optimizer="sgd", lr=0.001, fine_tunning=False):
    """Create 2D U-Net.                                                         
                                                                                
       Args:                                                                    
            image_shape (2D tuple): dimensions of the input image.              
                                                                                
            activation (str, optional): Keras available activation type.        
                                                                                
            feature_maps (array of ints, optional): feature maps to use on each 
                level. Must have the same length as the ``depth+1``.            
                                                                                
            depth (int, optional): depth of the network.                        
                                                                                
            drop_values (float, optional): dropout value to be fixed. If no     
                value is provided the default behaviour will be to select a     
                piramidal value starting from ``0.1`` and reaching ``0.3`` value.
                                                                                
            spatial_dropout (bool, optional): use spatial dropout instead of the
                `normal` dropout.                                               
                                                                                
            batch_norm (bool, optional): flag to make batch normalization.      
                                                                                
            k_init (string, optional): kernel initialization for convolutional  
                layers.                                                         
                                                                                
            loss_type (str, optional): loss type to use, three type available:  
                ``bce`` (Binary Cross Entropy) , ``w_bce`` (Weighted BCE, based 
                on weigth maps) and ``w_bce_dice`` (Weighted loss: ``weight1*BCE 
                + weight2*Dice``).                                              
                                                                                
            optimizer (str, optional): optimizer used to minimize the loss      
                function. Posible options: ``sgd`` or ``adam``.                 
                                                                                
            lr (float, optional): learning rate value.                          
                                                                                
            fine_tunning (bool, optional): flag to freeze the encoder part for  
                fine tuning.                                                    
                                                                                
       Returns:                                                                 
            Keras model: model containing the U-Net.              

       Calling this function with its default parameters returns the following  
       network:                                                                 
                                                                                
       .. image:: img/unet.png                                                  
           :width: 100%                                                         
           :align: center                                                       
                                                                                
       Image created with `PlotNeuralNet <https://github.com/HarisIqbal88/PlotNeuralNet>`_.
    """

    if len(feature_maps) != depth+1:
        raise ValueError("feature_maps dimension must be equal depth+1")
    if len(drop_values) != depth+1:
        raise ValueError("'drop_values' dimension must be equal depth+1")

    dinamic_dim = (None,)*(len(image_shape)-1) + (1,)                           
    x = Input(dinamic_dim)                                                      
    #x = Input(image_shape)                                                     
    inputs = x
        
    if loss_type == "w_bce":
        weights = Input(image_shape)

    # List used to access layers easily to make the skip connections of the U-Net
    l=[]

    # ENCODER
    for i in range(depth):
        x = Conv2D(feature_maps[i], (3, 3), activation=None,
                   kernel_initializer=k_init, padding='same') (x)
        x = BatchNormalization() (x) if batch_norm else x 
        x = Activation(activation) (x)
        if drop_values is not None:
            if spatial_dropout:
                x = SpatialDropout2D(drop_values[i]) (x)
            else:
                x = Dropout(drop_values[i]) (x)
        x = Conv2D(feature_maps[i], (3, 3), activation=None,
                   kernel_initializer=k_init, padding='same') (x)
        x = BatchNormalization() (x) if batch_norm else x 
        x = Activation(activation) (x)
        x = squeeze_excite_block(x)

        l.append(x)
    
        x = MaxPooling2D((2, 2))(x)

    # BOTTLENECK
    x = Conv2D(feature_maps[depth], (3, 3), activation=None,
               kernel_initializer=k_init, padding='same')(x)
    x = BatchNormalization() (x) if batch_norm else x
    x = Activation(activation) (x)
    if drop_values is not None:
            if spatial_dropout:
                x = SpatialDropout2D(drop_values[depth]) (x)
            else:
                x = Dropout(drop_values[depth]) (x)
    x = Conv2D(feature_maps[depth], (3, 3), activation=None,
               kernel_initializer=k_init, padding='same') (x)
    x = BatchNormalization() (x) if batch_norm else x
    x = Activation(activation) (x)

    # DECODER
    for i in range(depth-1, -1, -1):
        x = Conv2DTranspose(feature_maps[i], (2, 2), 
                            strides=(2, 2), padding='same') (x)
        x = concatenate([x, l[i]])
        x = Conv2D(feature_maps[i], (3, 3), activation=None,
                   kernel_initializer=k_init, padding='same') (x)
        x = BatchNormalization() (x) if batch_norm else x
        x = Activation(activation) (x)
        if drop_values is not None:
            if spatial_dropout:
                x = SpatialDropout2D(drop_values[i]) (x)
            else:
                x = Dropout(drop_values[i]) (x)
        x = Conv2D(feature_maps[i], (3, 3), activation=None,
                   kernel_initializer=k_init, padding='same') (x)
        x = BatchNormalization() (x) if batch_norm else x
        x = Activation(activation) (x)
        x = squeeze_excite_block(x)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (x)
    
    # Loss type
    if loss_type == "w_bce":
        model = Model(inputs=[inputs, weights], outputs=[outputs])
    else:
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

    # Fine tunning: freeze the enconder part
    if fine_tunning == True:
        print("Freezing the contracting path of the U-Net for fine tunning . . .")
        for layer in model.layers[:20]:
            layer.trainable = False
        for layer in model.layers:
            print("{}: {}".format(layer, layer.trainable))

    # Compile the model
    if loss_type == "bce":
        model.compile(optimizer=opt, loss='binary_crossentropy',
                      metrics=[jaccard_index])
    elif loss_type == "w_bce":
        model.compile(optimizer=opt, loss=binary_crossentropy_weighted(weights),
                      metrics=[jaccard_index])
    elif loss_type == "w_bce_dice":
        model.compile(optimizer=opt,
                      loss=weighted_bce_dice_loss(w_dice=0.66, w_bce=0.33),
                      metrics=[jaccard_index])
    else:
        raise ValueError("'loss_type' must be 'bce', 'w_bce' or 'w_bce_dice'")

    return model


def squeeze_excite_block(input, ratio=16):
    ''' Create a channel-wise squeeze-excite block
        Code fully extracted from `keras-squeeze-excite-network <https://github.com/titu1994/keras-squeeze-excite-network>_`.

    Args:
        input: input tensor
        filters: number of output filters

    Returns: a keras tensor

    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x


def spatial_squeeze_excite_block(input):
    ''' Create a spatial squeeze-excite block
        Code fully extracted from `keras-squeeze-excite-network <https://github.com/titu1994/keras-squeeze-excite-network>_`.

    Args:
        input: input tensor

    Returns: a keras tensor

    References
    -   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
    '''

    se = Conv2D(1, (1, 1), activation='sigmoid', use_bias=False,
                kernel_initializer='he_normal')(input)

    x = multiply([input, se])
    return x


def channel_spatial_squeeze_excite(input, ratio=16):
    ''' Create a spatial squeeze-excite block.
        Code fully extracted from `keras-squeeze-excite-network <https://github.com/titu1994/keras-squeeze-excite-network>_`.

    Args:
        input: input tensor
        filters: number of output filters

    Returns: a keras tensor

    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    -   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
    '''

    cse = squeeze_excite_block(input, ratio)
    sse = spatial_squeeze_excite_block(input)

    x = add([cse, sse])
    return x

