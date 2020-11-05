import tensorflow as tf
from tensorflow.keras.layers import Dropout, SpatialDropout3D, Conv3D,\
                                    Conv3DTranspose, MaxPooling3D, concatenate,\
                                    ELU, BatchNormalization, Activation, \
                                    ZeroPadding3D
from tensorflow.keras import Model, Input
from metrics import binary_crossentropy_weighted, jaccard_index, \
                    jaccard_index_softmax, weighted_bce_dice_loss


def Vanilla_U_Net_3D(image_shape, activation='relu', 
                     feature_maps=[32, 64, 128, 256, 512], depth=3, 
                     k_init='he_normal', loss_type="bce", optimizer="sgd",
                     lr=0.001, n_classes=1):
    """Create Vanilla 3D U-Net.

       Parameters
       ----------
       image_shape : 3D tuple
           Dimensions of the input image.

       activation : str, optional
           Keras available activation type.

       feature_maps : array of ints, optional
           Feature maps to use on each level. Must have the same length as the 
           ``depth+2``.
   
       depth : int, optional
           Depth of the network.

       k_init : string, optional
           Kernel initialization for convolutional layers.

       loss_type : str, optional
           Loss type to use, three type available: ``bce`` (Binary Cross Entropy) 
           , ``w_bce`` (Weighted BCE, based on weigth maps) and ``w_bce_dice`` 
           (Weighted loss: ``weight1*BCE + weight2*Dice``). 

       optimizer : str, optional
           Optimizer used to minimize the loss function. Posible options: ``sgd``
           or ``adam``.

       lr : float, optional
           Learning rate value.

       n_classes: int, optional                                                 
           Number of classes.    

       Returns
       -------
       model : Keras model
           Model containing the U-Net.

    
       Calling this function with its default parameters returns the following
       network:

       .. image:: img/unet_3d.png
           :width: 100%
           :align: center

       Image created with `PlotNeuralNet <https://github.com/HarisIqbal88/PlotNeuralNet>`_.
    """

    if len(feature_maps) != depth+2:                                            
        raise ValueError("feature_maps dimension must be equal depth+2")        

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
        x = BatchNormalization() (x) 
        x = Activation(activation) (x)

        x = Conv3D(feature_maps[i+1], (3, 3, 3), activation=None,
                   kernel_initializer=k_init, padding='same') (x)
        x = BatchNormalization() (x)
        x = Activation(activation) (x)

        l.append(x)
    
        x = MaxPooling3D((2, 2, 2))(x)

    # BOTTLENECK
    x = Conv3D(feature_maps[depth], (3, 3, 3), activation=None,
               kernel_initializer=k_init, padding='same')(x)
    x = BatchNormalization() (x)
    x = Activation(activation) (x)

    x = Conv3D(feature_maps[depth+1], (3, 3, 3), activation=None,
               kernel_initializer=k_init, padding='same') (x)
    x = BatchNormalization() (x)
    x = Activation(activation) (x)

    # DECODER
    for i in range(depth-1, -1, -1):
        x = Conv3DTranspose(feature_maps[i+2], (2, 2, 2), 
                            strides=(2, 2, 2), padding='same') (x)
        x = concatenate([x, l[i]])

        x = Conv3D(feature_maps[i+1], (3, 3, 3), activation=None,
                   kernel_initializer=k_init, padding='same') (x)
        x = BatchNormalization() (x)
        x = Activation(activation) (x)

        x = Conv3D(feature_maps[i+1], (3, 3, 3), activation=None,
                   kernel_initializer=k_init, padding='same') (x)
        x = BatchNormalization() (x) 
        x = Activation(activation) (x)

    outputs = Conv3D(n_classes, (1, 1, 1), activation='sigmoid') (x)
    
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
        if n_classes > 1:                                                       
            model.compile(optimizer=opt, loss='categorical_crossentropy',       
                          metrics=[jaccard_index_softmax])                      
        else:                                                                   
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
