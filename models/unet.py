import tensorflow as tf
from tensorflow.keras.layers import (Dropout, SpatialDropout2D, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate,
                                     ELU, BatchNormalization, Activation, ZeroPadding2D)
from tensorflow.keras import Model, Input


def U_Net_2D(image_shape, activation='elu', feature_maps=[16, 32, 64, 128, 256], drop_values=[0.1,0.1,0.2,0.2,0.3],
             spatial_dropout=False, batch_norm=False, k_init='he_normal', n_classes=1):
    """Create 2D U-Net.
                                                                                
       Parameters
       ----------
       image_shape : 2D tuple
           Dimensions of the input image.              
                                                                                
       activation : str, optional
           Keras available activation type.        
                                                                           
       feature_maps : array of ints, optional
           Feature maps to use on each level. 
                                                                           
       drop_values : float, optional
           Dropout value to be fixed. If no value is provided the default behaviour will be to select a piramidal value
           starting from ``0.1`` and reaching ``0.3`` value.
                                                                           
       spatial_dropout : bool, optional
           Use spatial dropout instead of the `normal` dropout.                                               
                                                                           
       batch_norm : bool, optional
           Make batch normalization.      
                                                                           
       k_init : string, optional
           Kernel initialization for convolutional layers.                                                         
                                                                           
       n_classes: int, optional
           Number of classes.
                                                                           
       Returns
       -------                                                                 
       model : Keras model
           Model containing the U-Net.              


       Calling this function with its default parameters returns the following  network:                                                                 
                                                                                
       .. image:: ../img/unet.png                                                  
           :width: 100%                                                         
           :align: center                                                       
                                                                                
       Image created with `PlotNeuralNet <https://github.com/HarisIqbal88/PlotNeuralNet>`_.
    """

    if len(feature_maps) != len(drop_values):
        raise ValueError("'feature_maps' dimension must be equal 'drop_values' dimension")
    depth = len(feature_maps)-1

    dinamic_dim = (None,)*(len(image_shape)-1) + (image_shape[-1],)                           
    x = Input(dinamic_dim)                                                      
    #x = Input(image_shape)                                                     
    inputs = x
        
    # List used to access layers easily to make the skip connections of the U-Net
    l=[]

    # ENCODER
    for i in range(depth):
        x = Conv2D(feature_maps[i], (3, 3), activation=None, kernel_initializer=k_init, padding='same') (x)
        x = BatchNormalization() (x) if batch_norm else x 
        x = Activation(activation) (x)
        if drop_values is not None:
            if spatial_dropout:
                x = SpatialDropout2D(drop_values[i]) (x)
            else:
                x = Dropout(drop_values[i]) (x)
        x = Conv2D(feature_maps[i], (3, 3), activation=None, kernel_initializer=k_init, padding='same') (x)
        x = BatchNormalization() (x) if batch_norm else x 
        x = Activation(activation) (x)

        l.append(x)
    
        x = MaxPooling2D((2, 2))(x)

    # BOTTLENECK
    x = Conv2D(feature_maps[depth], (3, 3), activation=None, kernel_initializer=k_init, padding='same')(x)
    x = BatchNormalization() (x) if batch_norm else x
    x = Activation(activation) (x)
    if drop_values is not None:
            if spatial_dropout:
                x = SpatialDropout2D(drop_values[depth]) (x)
            else:
                x = Dropout(drop_values[depth]) (x)
    x = Conv2D(feature_maps[depth], (3, 3), activation=None, kernel_initializer=k_init, padding='same') (x)
    x = BatchNormalization() (x) if batch_norm else x
    x = Activation(activation) (x)

    # DECODER
    for i in range(depth-1, -1, -1):
        x = Conv2DTranspose(feature_maps[i], (2, 2), strides=(2, 2), padding='same') (x)
        x = concatenate([x, l[i]])
        x = Conv2D(feature_maps[i], (3, 3), activation=None, kernel_initializer=k_init, padding='same') (x)
        x = BatchNormalization() (x) if batch_norm else x
        x = Activation(activation) (x)
        if drop_values is not None:
            if spatial_dropout:
                x = SpatialDropout2D(drop_values[i]) (x)
            else:
                x = Dropout(drop_values[i]) (x)
        x = Conv2D(feature_maps[i], (3, 3), activation=None, kernel_initializer=k_init, padding='same') (x)
        x = BatchNormalization() (x) if batch_norm else x
        x = Activation(activation) (x)

    outputs = Conv2D(n_classes, (1, 1), activation='sigmoid') (x)
    
    model = Model(inputs=[inputs], outputs=[outputs])

    return model

