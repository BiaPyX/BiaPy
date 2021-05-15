import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout, Conv3D, Conv3DTranspose,\
                                    MaxPooling3D, Concatenate, Add,\
                                    BatchNormalization, ELU, ZeroPadding3D
from tensorflow.keras import backend as K                                       
from tensorflow.keras import losses  
from metrics import jaccard_index_instances, instance_segmentation_loss 


def ResUNet_3D_instances(image_shape, activation='elu', k_init='he_normal', 
                         drop_values=[0.1,0.1,0.1,0.1,0.1], batch_norm=False, 
                         feature_maps=[16,32,64,128,256], depth=4, z_down=2,
                         output_channels="BC", channel_weights=(1,0.2),
                         loss_type="bce", optimizer="sgd", lr=0.001):
    """Create 3D Residual_U-Net for instance segmentation. It can be output up to
       ``3`` channels in the following order: binary segmentation, contour 
       segmentation and distance transform. Binary cross entropy (BCE) will be 
       used for the first two channels and mean squared error (MSE) to the last 
       one. The channel number is controlled with ``output_channels`` arg. 

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
           Feature maps to use on each level. Must have the same length as the 
           ``depth+1``.
       
       depth : int, optional
           Depth of the network.                        
                                                                           
       z_down : int, optional
           Downsampling used in z dimension. Set it to ``1`` if the dataset is 
           not isotropic.

       out_channels : str, optional                                             
           Channels to operate with. Possible values: ``BC`` and ``BCD``.  ``BC``
           corresponds to use binary segmentation+contour. ``BCD`` stands for   
           binary segmentation+contour+distances. 

       channel_weights : 2 float tuple, optional
           Weights to be applied to segmentation (binary and contours) and to 
           distances respectively. E.g. ``(1, 0.2)``, ``1`` should be multipled
           by ``BCE`` for the first two channels and ``0.2`` to ``MSE`` for the 
           last channel. 
 
       loss_type : str, optional
           Loss type to use, three type available: ``bce`` (Binary Cross Entropy),
           ``w_bce`` (Weighted BCE, based on weigth maps) and ``w_bce_dice``
           (Weighted loss: ``weight1*BCE + weight2*Dice``). 
                                                                           
       optimizer : str, optional
           Optimizer used to minimize the loss function. Posible options: ``sgd``
           or ``adam``.                         
                                                                           
       lr : float, optional
           Learning rate value.

       Returns  
       -------
       Model : Keras model
            Model containing the U-Net.


       Calling this function with its default parameters returns the following
       network:

       .. image:: img/resunet_3d.png
           :width: 100%
           :align: center

       Where each green layer represents a residual block as the following:

       .. image:: img/res_block.png
           :width: 45%
           :align: center

       Images created with `PlotNeuralNet <https://github.com/HarisIqbal88/PlotNeuralNet>`_.
    """

    if len(feature_maps) != depth+1:                                            
        raise ValueError("feature_maps dimension must be equal depth+1")
    if len(drop_values) != depth+1:
        raise ValueError("'drop_values' dimension must be equal depth+1")
    assert output_channels in ['BC', 'BCD']  
    if len(channel_weights) != 2:                                                 
        raise ValueError("Channel weights need to be len(2) and not {}"
                         .format(len(channel_weights))) 

    fm = feature_maps[::-1]

    inputs = Input(image_shape)

    x = level_block(inputs, depth, fm, 3, activation, k_init, drop_values, 
                    batch_norm, True, z_down)

    if output_channels == "BC":
        outputs = Conv3D(2, (2, 2, 2), activation="sigmoid", padding='same') (x)        
    else:   
        seg = Conv3D(2, (2, 2, 2), activation="sigmoid", padding='same') (x)
        dis = Conv3D(1, (2, 2, 2), activation="linear", padding='same') (x)
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
                                                                                
    # Compile the model                                                         
    model.compile(
        optimizer=opt, loss=instance_segmentation_loss(channel_weights, output_channels), 
        metrics=[jaccard_index_instances])
                            
    return model


def level_block(x, depth, f_maps, filter_size, activation, k_init, drop_values,   
                batch_norm, first_block, z_down):                                       
    """Produces a level of the network. It calls itself recursively.
                                                                                
       Parameters
       ----------
       x : Keras layer
           Input layer of the block.                          
                                                                           
       depth : int
           Depth of the network. This value determines how many times the 
           function will be called recursively.

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
           To advice the function that it is the first residual block of the
           network, which avoids Full Pre-Activation layers.                                              

       z_down : int, optional                                                   
           Downsampling used in z dimension. Set it to 1 if the dataset is not  
           isotropic.                                                           
      
       Returns
       -------
       x : Keras layer
           last layer of the levels.
    """  
                                                                                
    if depth > 0:                                                               
        r = residual_block(x, f_maps[depth], filter_size, activation, k_init,           
                           drop_values[depth], batch_norm, first_block)                 
        x = MaxPooling3D((2, 2, z_down)) (r)                                         
        x = level_block(x, depth-1, f_maps, filter_size, activation, k_init, 
                        drop_values, batch_norm, False, z_down)                          
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

        x = residual_block(x, f_maps[depth], filter_size, activation, k_init,           
                           drop_values[depth], batch_norm, False)                       
    else:                                                                       
        x = residual_block(x, f_maps[depth], filter_size, activation, k_init,           
                           drop_values[depth], batch_norm, False)                       
    return x 


def residual_block(x, f_maps, filter_size, activation='elu', k_init='he_normal', 
                   drop_value=0.0, bn=False, first_block=False):                                          
    """Residual block.
                                                                                
       Parameters
       ----------
       x : Keras layer
           iInput layer of the block.
                                                                           
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
           To advice the function that it is the first residual block of the 
           network, which avoids Full Pre-Activation layers.
                                                                                
       Returns
       -------
       x : Keras layer
           Last layer of the block.
    """ 
                                                                                
    # Create shorcut                                                            
    shortcut = Conv3D(f_maps, activation=None, kernel_size=(1, 1, 1),           
                      kernel_initializer=k_init)(x)                             
                                                                                
    # Main path                                                                 
    if not first_block:                                                         
        x = BatchNormalization()(x) if bn else x                                
        if activation == "elu":
            x = ELU(alpha=1.0) (x)                                                  
                                                                                
    x = Conv3D(f_maps, filter_size, activation=None,                            
               kernel_initializer=k_init, padding='same') (x)                   
                                                                                
    x = Dropout(drop_value) (x) if drop_value > 0 else x                        
    x = BatchNormalization()(x) if bn else x                                    
    if activation == "elu":
        x = ELU(alpha=1.0) (x)                                                      
                                                                                
    x = Conv3D(f_maps, filter_size, activation=None,                            
               kernel_initializer=k_init, padding='same') (x)                   
    x = BatchNormalization()(x) if bn else x                                    
                                                                                
    # Add shortcut value to main path                                           
    x = Add()([shortcut, x])                                                    
                                                                                
    return x         
