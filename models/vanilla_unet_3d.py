from tensorflow.keras.layers import (Conv3D, Conv3DTranspose, MaxPooling3D, concatenate, BatchNormalization, Activation)
from tensorflow.keras import Model, Input


def Vanilla_U_Net_3D(image_shape, activation='relu', feature_maps=[32, 64, 128, 256, 512], k_init='he_normal',
                     n_classes=1):
    """Create Vanilla 3D U-Net.

       Based on `U-Net 3D <https://arxiv.org/abs/1606.06650>`_.

       Parameters
       ----------
       image_shape : 3D tuple
           Dimensions of the input image.

       activation : str, optional
           Keras available activation type.

       feature_maps : array of ints, optional
           Feature maps to use on each level. 
   
       k_init : string, optional
           Kernel initialization for convolutional layers.

       n_classes: int, optional                                                 
           Number of classes.    

       Returns
       -------
       model : Keras model
           Model containing the U-Net.

    
       Calling this function with its default parameters returns the following network:

       .. image:: ../img/vanilla_unet_3d.png
           :width: 100%
           :align: center

       Image created with `PlotNeuralNet <https://github.com/HarisIqbal88/PlotNeuralNet>`_.
    """

    depth = len(feature_maps)-1

    #dinamic_dim = (None,)*(len(image_shape)-1) + (image_shape[-1],)
    #inputs = Input(dinamic_dim)
    x = Input(image_shape)
    inputs = x
        
    # List used to access layers easily to make the skip connections of the U-Net
    l=[]

    # ENCODER
    for i in range(depth):
        x = Conv3D(feature_maps[i], (3, 3, 3), activation=None, kernel_initializer=k_init, padding='same') (x)
        x = BatchNormalization() (x) 
        x = Activation(activation) (x)

        x = Conv3D(feature_maps[i+1], (3, 3, 3), activation=None, kernel_initializer=k_init, padding='same') (x)
        x = BatchNormalization() (x)
        x = Activation(activation) (x)

        l.append(x)
    
        x = MaxPooling3D((2, 2, 2))(x)

    # BOTTLENECK
    x = Conv3D(feature_maps[depth], (3, 3, 3), activation=None, kernel_initializer=k_init, padding='same')(x)
    x = BatchNormalization() (x)
    x = Activation(activation) (x)

    x = Conv3D(feature_maps[depth+1], (3, 3, 3), activation=None, kernel_initializer=k_init, padding='same') (x)
    x = BatchNormalization() (x)
    x = Activation(activation) (x)

    # DECODER
    for i in range(depth-1, -1, -1):
        x = Conv3DTranspose(feature_maps[i+2], (2, 2, 2), strides=(2, 2, 2), padding='same') (x)
        x = concatenate([x, l[i]])

        x = Conv3D(feature_maps[i+1], (3, 3, 3), activation=None, kernel_initializer=k_init, padding='same') (x)
        x = BatchNormalization() (x)
        x = Activation(activation) (x)

        x = Conv3D(feature_maps[i+1], (3, 3, 3), activation=None, kernel_initializer=k_init, padding='same') (x)
        x = BatchNormalization() (x) 
        x = Activation(activation) (x)

    outputs = Conv3D(n_classes, (1, 1, 1), activation='sigmoid') (x)
    
    # Loss type 
    model = Model(inputs=[inputs], outputs=[outputs]) 
    
    return model
