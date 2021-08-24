import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (Dropout, Lambda, SpatialDropout3D, Conv3D, Conv3DTranspose, MaxPooling3D, 
                                     Concatenate, Add, BatchNormalization, ELU, ZeroPadding3D)


def ResUNet_3D(image_shape, activation='elu', k_init='he_normal', drop_values=[0.1,0.1,0.1,0.1,0.1], batch_norm=False, 
               feature_maps=[16,32,64,128,256], z_down=2, n_classes=1):
    """Create 3D Residual_U-Net.

       Parameters
       ----------
       image_shape : 3D tuple
           Dimensions of the input image.

       activation : str, optional
           Keras available activation type.

       k_init : str, optional
           Keras available kernel initializer type.

       drop_values : array of floats, optional
           Dropout value to be fixed.

       batch_norm : bool, optional
           Use batch normalization.

       feature_maps : array of ints, optional
           Feature maps to use on each level. 
       
       z_down : int, optional
           Downsampling used in z dimension. Set it to ``1`` if the dataset is not isotropic.

       n_classes: int, optional                                                 
           Number of classes.    

       Returns  
       -------
       Model : Keras model
            Model containing the U-Net.


       Calling this function with its default parameters returns the following network:

       .. image:: ../img/resunet_3d.png
           :width: 100%
           :align: center

       Where each green layer represents a residual block as the following:

       .. image:: ../img/res_block.png
           :width: 45%
           :align: center

       Images created with `PlotNeuralNet <https://github.com/HarisIqbal88/PlotNeuralNet>`_.
    """

    if len(feature_maps) != len(drop_values):
        raise ValueError("'feature_maps' dimension must be equal 'drop_values' dimension")
    depth = len(feature_maps)-1

    fm = feature_maps[::-1]

    inputs = Input(image_shape)

    x = level_block(inputs, depth, fm, 3, activation, k_init, drop_values, batch_norm, True)

    outputs = Conv3D(n_classes, (1, 1, 1), activation='sigmoid') (x)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model


def level_block(x, depth, f_maps, filter_size, activation, k_init, drop_values, batch_norm, first_block):                                       
    """Produces a level of the network. It calls itself recursively.
                                                                                
       Parameters
       ----------
       x : Keras layer
           Input layer of the block.                          
                                                                           
       depth : int
           Depth of the network. This value determines how many times the function will be called recursively.

       f_maps : array of ints
           Feature maps to use.                        
                                                                           
       filter_size : 3 int tuple
           Height, width and depth of the convolution window.                                                             
                                                                           
       activation : str, optional
           Keras available activation type.        
                                                                           
       k_init : str, optional
           Keras available kernel initializer type.    
                                                                           
       drop_values : array of floats, optional
           Dropout value to be fixed.
                                                                           
       batch_norm : bool, optional
           Use batch normalization.                       
                                                                           
       first_block : float, optional
           To advice the function that it is the first residual block of the network, which avoids Full Pre-Activation
           layers (more info of Full Pre-Activation in `Identity Mappings in Deep Residual Networks
           <https://arxiv.org/pdf/1603.05027.pdf>`_).
           
       Returns
       -------
       x : Keras layer
           last layer of the levels.
    """  
                                                                                
    if depth > 0:                                                               
        r = residual_block(x, f_maps[depth], filter_size, activation, k_init, drop_values[depth], batch_norm, first_block)                 
        x = MaxPooling3D((2, 2, z_down)) (r)                                         
        x = level_block(x, depth-1, f_maps, filter_size, activation, k_init, drop_values, batch_norm, False)                          
        x = Conv3DTranspose(f_maps[depth], (2, 2, 2), strides=(2, 2, z_down), padding='same') (x)

        # Adjust shape introducing zero padding to allow the concatenation
        a = x.shape[3]
        b = r.shape[3]
        s = a - b
        if s > 0:
            r = ZeroPadding3D(padding=((0,0), (0,0), (s,0))) (r)
        elif s < 0:
            x = ZeroPadding3D(padding=((0,0), (0,0), (abs(s),0))) (x)
        x = Concatenate()([x, r])                                               

        x = residual_block(x, f_maps[depth], filter_size, activation, k_init, drop_values[depth], batch_norm, False)                       
    else:                                                                       
        x = residual_block(x, f_maps[depth], filter_size, activation, k_init, drop_values[depth], batch_norm, False)                       
    return x 


def residual_block(x, f_maps, filter_size, activation='elu', k_init='he_normal', drop_value=0.0, bn=False, 
                   first_block=False):                                          
    """Residual block.
                                                                                
       Parameters
       ----------
       x : Keras layer
           Input layer of the block.
                                                                           
       f_maps : array of ints
           Feature maps to use.
       
       filter_size : 3 int tuple
           Height, width and depth of the convolution window. 

       activation : str, optional
           Keras available activation type.        
                                                                           
       k_init : str, optional
           Keras available kernel initializer type.    
                                                                           
       drop_value : float, optional
           Dropout value to be fixed.
                                                                           
       bn : bool, optional
           Use batch normalization.               
                                                                           
       first_block : float, optional
           To advice the function that it is the first residual block of the network, which avoids Full Pre-Activation
           layers (more info of Full Pre-Activation in `Identity Mappings in Deep Residual Networks                    
           <https://arxiv.org/pdf/1603.05027.pdf>`_).
                                                                                
       Returns
       -------
       x : Keras layer
           Last layer of the block.
    """ 
                                                                                
    # Create shorcut                                                            
    shortcut = Conv3D(f_maps, activation=None, kernel_size=(1, 1, 1), kernel_initializer=k_init)(x)                             
                                                                                
    # Main path                                                                 
    if not first_block:                                                         
        x = BatchNormalization()(x) if bn else x                                
        if activation == "elu":
            x = ELU(alpha=1.0) (x)                                                  
                                                                                
    x = Conv3D(f_maps, filter_size, activation=None, kernel_initializer=k_init, padding='same') (x)                   
                                                                                
    x = Dropout(drop_value) (x) if drop_value > 0 else x                        
    x = BatchNormalization()(x) if bn else x                                    
    if activation == "elu":
        x = ELU(alpha=1.0) (x)                                                      
                                                                                
    x = Conv3D(f_maps, filter_size, activation=None, kernel_initializer=k_init, padding='same') (x)                   
    x = BatchNormalization()(x) if bn else x                                    
                                                                                
    # Add shortcut value to main path                                           
    x = Add()([shortcut, x])                                                    
                                                                                
    return x         