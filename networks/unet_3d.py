import tensorflow as tf
from tensorflow.keras.layers import Dropout, Lambda, SpatialDropout3D, Conv3D,\
                                    Conv3DTranspose, MaxPooling3D, concatenate,\
                                    ELU, BatchNormalization, Activation, \
                                    ZeroPadding3D
from tensorflow.keras import Model, Input
from tensorflow.keras.activations import relu
from metrics import binary_crossentropy_weighted, jaccard_index, \
                    weighted_bce_dice_loss


def U_Net_3D(image_shape, activation='elu', feature_maps=[32, 64, 128, 256], 
             depth=3, drop_value=0.0, spatial_dropout=False, batch_norm=False, 
             k_init='he_normal', loss_type="bce", optimizer="sgd", lr=0.001):
             
    """Create the U-Net3D.

       Args:
            image_shape (3D tuple): dimensions of the input image.

            activation (str, optional): Keras available activation type.

            feature_maps (int, optional): feature maps to use on each level. Must
            have the same length as the depth+1.
        
            depth (int, optional): depth of the network.

            drop_value (float, optional): dropout value to be fixed. If no
            value is provided the default behaviour will be to select a
            piramidal value stating from 0.1 and reaching 0.3 value.

            spatial_dropout (bool, optional): use spatial dropout instead of the
            "normal" dropout.

            batch_norm (bool, optional): flag to make batch normalization.
    
            k_init (string, optional): kernel initialization for convolutional 
            layers.

            loss_type (str, optional): loss type to use, three type available: 
            "bce" (Binary Cross Entropy) , "w_bce" (Weighted BCE, based on
            weigth maps) and "w_bce_dice" (Weighted loss: weight1*BCE + weight2*Dice). 

            optimizer (str, optional): optimizer used to minimize the loss
            function. Posible options: 'sgd' or 'adam'.

            lr (float, optional): learning .dd(Input(image_shape))

       Returns:
            model (Keras model): model containing the U-Net created.
    """

    if len(feature_maps) != depth+1:
        raise ValueError("feature_maps dimension must be equal depth+1")

    #dinamic_dim = (None,)*(len(image_shape)-1) + (1,)
    #inputs = Input(dinamic_dim)
    x = Input(image_shape)
    inputs = x
        
    if loss_type == "w_bce":
        weights = Input(image_shape)

    # List used to access layers easily to make the skip connections of the U-Net
    l=[]

    # ENCODER
    for i in range(depth):
        x = Conv3D(feature_maps[i], (3, 3, 3), activation=None,
                   kernel_initializer=k_init, padding='same') (x)
        x = BatchNormalization() (x) if batch_norm else x 
        x = Activation(activation) (x)
        if spatial_dropout and drop_value > 0:
            x = SpatialDropout3D(drop_value) (x)
        elif drop_value > 0 and not spatial_dropout:
            x = Dropout(drop_value) (x)

        x = Conv3D(feature_maps[i], (3, 3, 3), activation=None,
                   kernel_initializer=k_init, padding='same') (x)
        x = BatchNormalization() (x) if batch_norm else x 
        x = Activation(activation) (x)

        l.append(x)
    
        x = MaxPooling3D((2, 2, 2))(x)

    # BOTTLENECK
    x = Conv3D(feature_maps[depth], (3, 3, 3), activation=None,
               kernel_initializer=k_init, padding='same')(x)
    x = BatchNormalization() (x) if batch_norm else x
    x = Activation(activation) (x)
    if spatial_dropout and drop_value > 0:
        x = SpatialDropout3D(drop_value) (x)
    elif drop_value > 0 and not spatial_dropout:
        x = Dropout(drop_value) (x)

    x = Conv3D(feature_maps[depth], (3, 3, 3), activation=None,
               kernel_initializer=k_init, padding='same') (x)
    x = BatchNormalization() (x) if batch_norm else x
    x = Activation(activation) (x)

    # DECODER
    for i in range(depth-1, -1, -1):
        x = Conv3DTranspose(feature_maps[i], (2, 2, 2), 
                            strides=(2, 2, 2), padding='same') (x)

        # Adjust shape if need introducing zero padding to allow the concatenation
        a = x.shape[1]
        b = l[i].shape[1]
        s = a - b
        if s > 0:
            l[i] = ZeroPadding3D(padding=((s,0), (s,0), (s,0))) (l[i])
        elif s < 0:
            x = ZeroPadding3D(padding=((abs(s),0), (abs(s),0), (abs(s),0))) (x)
        x = concatenate([x, l[i]])

        x = Conv3D(feature_maps[i], (3, 3, 3), activation=None,
                   kernel_initializer=k_init, padding='same') (x)
        x = BatchNormalization() (x) if batch_norm else x
        x = Activation(activation) (x)
        if spatial_dropout and drop_value > 0:
            x = SpatialDropout3D(drop_value) (x)
        elif drop_value > 0 and not spatial_dropout:
            x = Dropout(drop_value) (x)
        x = Conv3D(feature_maps[i], (3, 3, 3), activation=None,
                   kernel_initializer=k_init, padding='same') (x)
        x = BatchNormalization() (x) if batch_norm else x
        x = Activation(activation) (x)

    outputs = Conv3D(1, (1, 1, 1), activation='sigmoid') (x)
    
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
