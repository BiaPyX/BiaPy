import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout, Lambda, SpatialDropout2D, Conv2D, \
                                    Conv2DTranspose, MaxPooling2D, Concatenate,\
                                    Add, BatchNormalization, ELU
from metrics import jaccard_index, instance_segmentation_loss, \
                    jaccard_index_instances


def ResUNet_2D(image_shape, activation='elu', k_init='he_normal',
               drop_values=[0.1,0.1,0.1,0.1,0.1], batch_norm=False, 
               feature_maps=[16,32,64,128,256], depth=4, optimizer="sgd",
               lr=0.001, n_classes=1, output_channels="BC", 
               channel_weights=(1,0.2)):
    """Create 2D Residual_U-Net.

       Parameters
       ----------
       image_shape : array of 3 int
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
           Feature maps to use on each level. Must have the same length as the 
           ``depth+1``.
       
       depth : int, optional
           Depth of the network.                        
                                                                           
       optimizer : str, optional
           Optimizer used to minimize the loss function. Posible options: 
           ``sgd`` or ``adam``.                         
                                                                           
       lr : float, optional
           Learning rate value.

       n_classes: int, optional                                                 
           Number of classes.  
     
       out_channels : str, optional                                             
           Channels to operate with. Possible values: ``B``, ``BC`` and ``BCD``.
           ``B`` stands for binary segmentation. ``BC`` corresponds to use
           binary segmentation+contour. ``BCD`` stands for binary
           segmentation+contour+distances.                               
                                                                                
       channel_weights : 2 float tuple, optional                                
           Weights to be applied to segmentation (binary and contours) and to   
           distances respectively. E.g. ``(1, 0.2)``, ``1`` should be multipled 
           by ``BCE`` for the first two channels and ``0.2`` to ``MSE`` for the 
           last channel.   

       Returns
       -------
       model : Keras model
           Model containing the U-Net.


       Calling this function with its default parameters returns the following
       network:

       .. image:: img/resunet.png
           :width: 100%
           :align: center

       Where each green layer represents a residual block as the following:
        
       .. image:: img/res_block.png
           :width: 45%
           :align: center

       Images created with `PlotNeuralNet <https://github.com/HarisIqbal88/PlotNeuralNet>`_.
    """

    assert output_channels in ['B', 'BC', 'BCD']                                     
    if len(channel_weights) != 2:                                               
        raise ValueError("Channel weights need to be len(2) and not {}"         
                         .format(len(channel_weights))) 

    if len(feature_maps) != depth+1:                                            
        raise ValueError("feature_maps dimension must be equal depth+1")
    if len(drop_values) != depth+1:
        raise ValueError("'drop_values' dimension must be equal depth+1")

    fm = feature_maps[::-1]

    dinamic_dim = (None,)*(len(image_shape)-1) + (image_shape[-1],)
    inputs = Input(dinamic_dim)

    x = level_block(inputs, depth, fm, 3, activation, k_init, drop_values, 
                    batch_norm, True)

    if output_channels == "B":                                                 
        outputs = Conv2D(1, (2, 2), activation="sigmoid", padding='same') (x)
        loss = "binary_crossentropy"
        jac = [jaccard_index]
    elif output_channels == "BC":                                             
        jac = [jaccard_index]
        loss = "binary_crossentropy"
        outputs = Conv2D(2, (2, 2), activation="sigmoid", padding='same') (x)
    else:
        loss = instance_segmentation_loss(channel_weights, output_channels)
        jac = [jaccard_index_instances]
        seg = Conv2D(2, (2, 2), activation="sigmoid", padding='same') (x)    
        dis = Conv2D(1, (2, 2), activation="linear", padding='same') (x)     
        outputs = Concatenate()([seg, dis])    

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

    model.compile(optimizer=opt, loss=loss, metrics=jac)

    return model


def level_block(x, depth, f_maps, f_size, activation, k_init, drop_values,   
                batch_norm, first_block):                                       

    """Produces a level of the network. It calls itself recursively.
                                                                                
       Parameters
       ----------
       x : Keras layer
           Input layer of the block.                          
                                                                           
       depth :int
           Depth of the network. This value determines how many times the
           function will be called recursively.

       f_maps : array of ints
           Feature maps to use.                        
                                                                           
       f_size : int
           Convolution window.                                                             
                                                                           
       activation : str, optional
           Keras available activation type.        
                                                                           
       k_init : str, optional
           Keras available kernel initializer type.    
                                                                           
       drop_values : array of floats, optional
           Dropout value to be fixed.
                                                                           
       batch_norm : bool, optional
           Use batch normalization.                       
                                                                           
       first_block : float, optional
           To advice the function that it is the first residual block of the
           network, which avoids Full Pre-Activation layers.                                              
                                                                                
       Returns
       -------                                                                 
       x : Keras layer
           last layer of the levels.
    """  
                                                                                
    if depth > 0:                                                               
        r = residual_block(x, f_maps[depth], f_size, activation, k_init,           
                           drop_values[depth], batch_norm, first_block)                 
        x = MaxPooling2D((2, 2)) (r)                                         
        x = level_block(x, depth-1, f_maps, f_size, activation, k_init, 
                        drop_values, batch_norm, False)                          
        x = Conv2DTranspose(f_maps[depth], (2, 2), strides=(2, 2), padding='same') (x)
        x = Concatenate()([r, x])                                               
        x = residual_block(x, f_maps[depth], f_size, activation, k_init,           
                           drop_values[depth], batch_norm, False)                       
    else:                                                                       
        x = residual_block(x, f_maps[depth], f_size, activation, k_init,           
                           drop_values[depth], batch_norm, False)                       
    return x 


def residual_block(x, f_maps, f_size, activation='elu', k_init='he_normal', 
                   drop_value=0.0, bn=False, first_block=False):                                          

    """Residual block.
                                                                                
       Parameters
       ----------

       x : Keras layer
           Input layer of the block.
                                                                                
       f_maps : array of ints
           Feature maps to use.
            
       f_size : int
           Convolution window. 

       activation : str, optional
           Keras available activation type.        
                                                                                
       k_init : str, optional
           Keras available kernel initializer type.    
                                                                                
       drop_value : float, optional
           Dropout value to be fixed. 
                                                                                
       bn : bool, optional
           Use batch normalization.               
                                                                                
       first_block : float, optional
           To advice the function that it is the first residual block of the
           network, which avoids Full Pre-Activation layers.
                                                                                
       Returns
       -------
       x : Keras layer
           Last layer of the block.
    """ 
                                                                                
    # Create shorcut                                                            
    shortcut = Conv2D(f_maps, activation=None, kernel_size=(1, 1),           
                      kernel_initializer=k_init)(x)                             
                                                                                
    # Main path                                                                 
    if not first_block:                                                         
        x = BatchNormalization()(x) if bn else x                                
        if activation == 'elu':
            x = ELU(alpha=1.0) (x)                                                  
                                                                                
    x = Conv2D(f_maps, f_size, activation=None,                            
               kernel_initializer=k_init, padding='same') (x)                   
                                                                                
    x = Dropout(drop_value) (x) if drop_value > 0 else x                        
    x = BatchNormalization()(x) if bn else x                                    
    if activation == 'elu':
        x = ELU(alpha=1.0) (x)                                                      
                                                                                
    x = Conv2D(f_maps, f_size, activation=None,                            
               kernel_initializer=k_init, padding='same') (x)                   
    x = BatchNormalization()(x) if bn else x                                    
                                                                                
    # Add shortcut value to main path                                           
    x = Add()([shortcut, x])                                                    
                                                                                
    return x         
