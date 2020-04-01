import tensorflow as tf
import tensorflow.keras
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dropout, Lambda, SpatialDropout2D, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Add
from metrics import binary_crossentropy_weighted, weighted_bce_dice_loss
from metrics_cheng2017 import jaccard_loss_cheng2017
from tensorflow.keras.layers import PReLU
import tensorflow as tensorflow
import numpy as np

def asymmetric_U_Net(image_shape, activation='elu', numInitChannels=16, 
                     fixed_dropout=0.0, spatial_dropout=False, loss_type="bce", 
                     optimizer="sgd", lr=0.001, t_downsmp_layer=4):
    """Create the U-Net

       Args:
            image_shape (array of 3 int): dimensions of the input image.

            activation (str, optional): Keras available activation type.

            numInitChannels (int, optional): number of convolution channels to
            start with. In each downsampling/upsampling the number of filters
            are multiplied/divided by 2.

            fixed_dropout (float, optional): dropout value to be fixed. If no
            value is provided the default behaviour will be to select a
            piramidal value stating from 0.1 and reaching 0.3 value.

            spatial_dropout (bool, optional): use spatial dropout instead of the
            "normal" dropout.

            loss_type (str, optional): loss type to use, three type available: 
            "bce" (Binary Cross Entropy) , "w_bce" (Weighted BCE, based on
            weigth maps) and "w_bce_dice" (Weighted loss: weight1*BCE + weight2*Dice). 

            optimizer (str, optional): optimizer used to minimize the loss
            function. Posible options: 'sgd' or 'adam'.

            lr (float, optional): learning rate value.

       Returns:
            model (Keras model): model containing the U-Net created.
    """
    inputs = Input((image_shape[0], image_shape[1], image_shape[2]))
        
    s = Lambda(lambda x: x / 255) (inputs)

    # Input block
    channels = numInitChannels
    c1 = Conv2D(channels, (3, 3), activation=None, strides=(2, 2),
                      kernel_initializer='he_normal', padding='same') (s)
    m1 = MaxPooling2D(pool_size=(2, 2))(s)
    x = concatenate([c1,m1])
       
    # First encode block sequence
    for i in range(2):
        channels += 8
        x = encode_block(x, channels, t_downsmp_layer=t_downsmp_layer)

    # 1st downsample block
    channels += 8
    x = encode_block(x, channels, downsample=True, 
                     t_downsmp_layer=t_downsmp_layer)

    # Second encode block sequence
    for i in range(3):
        channels += 8
        x = encode_block(x, channels, t_downsmp_layer=t_downsmp_layer)

    # 2nd downsample block
    x = encode_block(x, channels, downsample=True, 
                     t_downsmp_layer=t_downsmp_layer)
    
    # Third encode block sequence
    for i in range(6):
        channels += 8
        x = encode_block(x, channels, t_downsmp_layer=t_downsmp_layer)
   
    # 1st upsample block 
    channels = int(channels/2)
    x = decode_block(x, channels, upsample=True) 

    # First decode block sequence
    for i in range(4):
        x = decode_block(x, channels)

    # 2nd upsample block
    channels = int(channels/2)
    x = decode_block(x, channels, upsample=True)

    # Second decode block sequence 
    for i in range(2):                                                          
        x = decode_block(x, channels)

    # Last transpose conv 
    outputs = Conv2DTranspose(1, (2, 2), activation=None, strides=(2, 2)) (x)

    if loss_type == "w_bce":
        model = Model(inputs=[inputs, weights], outputs=[outputs]) 
    else:
        model = Model(inputs=[inputs], outputs=[outputs])
    
    if optimizer == "sgd":
        opt = tensorflow.keras.optimizers.SGD(lr=lr, momentum=0.90, decay=0.0, 
                                   nesterov=False)
    elif optimizer == "adam":
        opt = tensorflow.keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, 
                                    epsilon=None, decay=0.0, amsgrad=False)
    else:
        raise ValueError("Error: optimizer value must be 'sgd' or 'adam'")

    def scheduler(epoch):
        if epoch < 10:
            return 0.001
        else:
            return 0.001 * tf.math.exp(0.1 * (10 - epoch))

    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)    
        
    if loss_type == "bce":
        model.compile(optimizer=opt, loss='binary_crossentropy', 
                      metrics=[jaccard_loss_cheng2017])
    elif loss_type == "w_bce":
        model.compile(optimizer=opt, loss=binary_crossentropy_weighted(weights), 
                      metrics=[jaccard_loss_cheng2017])
    elif loss_type == "w_bce_dice":
        model.compile(optimizer=opt, 
                      loss=weighted_bce_dice_loss(w_dice=0.66, w_bce=0.33),
                      metrics=[jaccard_loss_cheng2017])
    else:
        raise ValueError("'loss_type' must be 'bce', 'w_bce' or 'w_bce_dice'")

    return model

def pad_depth(x, desired_channels):
    y = K.zeros_like(x)
    new_channels = desired_channels - x.shape.as_list()[-1]
    y = y[:,:,:,:new_channels]
    return concatenate([x,y])

def encode_block(inp_layer, channels, t_downsmp_layer=4, downsample=False):
        if downsample == True:
            shortcut_padded = Lambda(
                sto_downsampling2d, 
                arguments={'t':t_downsmp_layer})(inp_layer)
            shortcut_padded = Conv2D(1, (1, 1), activation=None) (shortcut_padded)
        else:
            shortcut_padded = Lambda(
                pad_depth, arguments={'desired_channels':channels})(inp_layer)
    
        x = BatchNormalization()(inp_layer)
        x = PReLU() (x)
        if downsample == True:
            x = Conv2D(channels, (3, 3), activation=None, strides=(2, 2),
                       kernel_initializer='he_normal', padding='same') (x)
        else:
            #x = Conv2D(channels, (3, 3), activation=None,
            #           kernel_initializer='he_normal', padding='same') (x)

            # Factorized kernels
            x = Conv2D(channels, (1, 3), activation=None,
                       kernel_initializer='he_normal', padding='same') (x)
            x = Conv2D(channels, (3, 1), activation=None,
                       kernel_initializer='he_normal', padding='same') (x)
    
        x = Dropout(0.1)(x)
        x = BatchNormalization()(x)
        x = PReLU() (x)
        x = Conv2D(channels, (3, 3), activation=None,
                    kernel_initializer='he_normal', padding='same') (x)
        x = Add()([shortcut_padded, x])
        return x

def decode_block(inp_layer, channels, upsample=False):
   
    if upsample == True:    
        x = Conv2DTranspose(int(channels), (3, 3), activation=None, 
                            strides=(2, 2), padding='same') (inp_layer)
    else:
        x = Conv2D(int(channels/4), (1, 1), activation=None,
               kernel_initializer='he_normal', padding='same')(inp_layer)
        x = Conv2D(int(channels/4), (3, 3), activation=None,                           
                   kernel_initializer='he_normal', padding='same')(x)
        x = Conv2D(channels, (1, 1), activation=None,                           
               kernel_initializer='he_normal', padding='same')(x)
        x = Add()([inp_layer, x])           
    return x 


def sto_downsampling2d(x, t=4):                                                 
    N = x.shape[0]
    H = x.shape[1]                                                              
    W = x.shape[2]                                                              
    C = x.shape[3]                                                              
                                                                                
    sv_h = int(H//t)                                                            
    sv_w = int(W//t)                                                            
    elem = int(t/2)                                                             
                                                                                
    # Select random rows and columns                                            
    c_rows = np.zeros((sv_h*elem), dtype=np.int32)                              
    c_cols = np.zeros((sv_w*elem), dtype=np.int32)                              
    for i in range(0, sv_h*elem, elem):                                         
        nums = np.sort(np.random.choice(t, elem, replace=False))                
        for j in range(elem):                                                   
            c_rows[i+j] = nums[j] + int(i/elem)*t                               
                                                                                
    for i in range(0, sv_w*elem, elem):                                         
        nums = np.sort(np.random.choice(t, elem, replace=False))                
        for j in range(elem):                                                   
            c_cols[i+j] = nums[j] + int(i/elem)*t                               
                                                                                
    tc_rows = tensorflow.constant(c_rows, dtype=tensorflow.int32)                               
    tc_cols = tensorflow.constant(c_cols, dtype=tensorflow.int32)                               
    
    if N is None:                                                               
        x = MaxPooling2D((2, 2)) (x)                                            
        return x                                                                
    else:                                                                       
        print("AQUI: N: {}".format(N))
        a = np.array([ [ [(c_rows[i], c_cols[j]) for j in range(sv_w*elem)] for i in range(sv_h*elem) ] for j in range(N) ])
        ta = tf.constant(a, dtype=tf.int32)                                     
        ta = tf.transpose(tf.stack([ta for i in range(C)]), [1, 2, 3, 0, 4])    
        ta = tf.pad(ta, [[0,0], [0,0], [0,0], [0,0], [ 1, 1 ]])                 
        return tf.gather_nd(x, ta)

def sto_downsampling2d_mod(x, t=4):
    N = x.shape[0]                                                              
    H = x.shape[1]                                                              
    W = x.shape[2]                                                              
    C = x.shape[3]                                                              
                                                                                
    sv_h = int(H//t)                                                            
    sv_w = int(W//t)                                                            
    elem = int(t/2)                                                             
                                                                                
    # Select random rows and columns                                            
    c_rows = np.zeros((sv_h*elem), dtype=np.int32)                              
    c_cols = np.zeros((sv_w*elem), dtype=np.int32)                              
    for i in range(0, sv_h*elem, elem):                                         
        nums = np.sort(np.random.choice(t, elem, replace=False))                
        for j in range(elem):                                                   
            c_rows[i+j] = nums[j] + int(i/elem)*t                               
                                                                                
    for i in range(0, sv_w*elem, elem):                                         
        nums = np.sort(np.random.choice(t, elem, replace=False))                
        for j in range(elem):                                                   
            c_cols[i+j] = nums[j] + int(i/elem)*t                               
                                                                                
    tc_rows = tensorflow.constant(c_rows, dtype=tensorflow.int32)                               
    tc_cols = tensorflow.constant(c_cols, dtype=tensorflow.int32)                               
                                                                                
    a = np.array([ [ [(c_rows[i], c_cols[j]) for j in range(sv_w*elem)] for i in range(sv_h*elem) ] for j in range(N) ])
                                                                                
    ta = tensorflow.constant(a, dtype=tensorflow.int32)                                         
    ta = tensorflow.transpose(tensorflow.stack([ta for i in range(C)]), [1, 2, 3, 0, 4])        
    ta = tensorflow.pad(ta, [[0,0], [0,0], [0,0], [0,0], [ 1, 1 ]])                     
    return tensorflow.gather_nd(x, ta)
