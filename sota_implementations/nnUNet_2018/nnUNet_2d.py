import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D,Conv2DTranspose, concatenate,\
                                    BatchNormalization, LeakyReLU, ZeroPadding2D
from tensorflow.keras import Model, Input
from metrics import jaccard_index_softmax


def nnUNet_2D(image_shape, feature_maps=32, max_fa=480, num_pool=8, 
              k_init='he_normal', optimizer="sgd", lr=0.002, n_classes=1):
    """Create nnU-Net 2D. This implementations tries to be a Keras version of the
       original nnU-Net 2D presented in `nnU-Net Github <https://github.com/MIC-DKFZ/nnUNet>`_.
                                                                                
       Parameters
       ----------
       image_shape : 2D tuple
           Dimensions of the input image.              
                                                                                
       feature_map : ints, optional
           Feature maps to start with in the first level of the U-Net (will be 
           duplicated on each level). 

       max_fa : int, optional
           Number of maximum feature maps allowed to used in conv layers.
        
       num_pool : int, optional
           number of pooling (downsampling) operations to do.

       k_init : string, optional
           Kernel initialization for convolutional layers.                                                         
                                                                           
       optimizer : str, optional
           Optimizer used to minimize the loss function. Posible options: 
           ``sgd`` or ``adam``.                 
                                                                           
       lr : float, optional
           Learning rate value.                          
        
       n_classes: int, optional
           Number of classes.
                                                                           
       Returns
       -------                                                                 
       model : Keras model
           Model containing the U-Net.              
    """

    dinamic_dim = (None,)*(len(image_shape)-1) + (image_shape[-1],)                           
    x = Input(dinamic_dim)                                                      
    #x = Input(image_shape)                                                     
    inputs = x
        
    l=[]
    seg_outputs = []
    fa_save = []
    fa = feature_maps

    # ENCODER
    x = StackedConvLayers(x, fa, k_init, first_conv_stride=1)
    fa_save.append(fa)
    fa = fa*2 if fa*2 < max_fa else max_fa
    l.append(x)

    # conv_blocks_context
    for i in range(num_pool-1):
        x = StackedConvLayers(x, fa, k_init)
        fa_save.append(fa)
        fa = fa*2 if fa*2 < max_fa else max_fa
        l.append(x)

    # BOTTLENECK
    x = StackedConvLayers(x, fa, k_init, first_conv_stride=(1,2))

    # DECODER
    for i in range(len(fa_save)):
        # tu
        if i == 0:
            x = Conv2DTranspose(fa_save[-(i+1)], (1, 2), use_bias=False,
                                strides=(1, 2), padding='valid') (x)
        else:
            x = Conv2DTranspose(fa_save[-(i+1)], (2, 2), use_bias=False,
                                strides=(2, 2), padding='valid') (x)
        x = concatenate([x, l[-(i+1)]])

        # conv_blocks_localization
        x = StackedConvLayers(x, fa_save[-(i+1)], k_init, first_conv_stride=1)
        seg_outputs.append(Conv2D(n_classes, (1, 1), use_bias=False, activation="softmax") (x))   

    outputs = seg_outputs
    
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


    # Calculate the weigts as nnUNet does
    ################# Here we wrap the loss for deep supervision ############
    # we need to know the number of outputs of the network
    net_numpool = num_pool

    # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
    # this gives higher resolution outputs more weight in the loss
    weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

    # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
    mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
    weights[~mask] = 0
    weights = weights / weights.sum()
    weights = weights[::-1] 
    ################# END ###################

    # Compile the model
    model.compile(optimizer=opt, loss='binary_crossentropy',
                  metrics=[jaccard_index_softmax], loss_weights=list(weights))

    return model


def StackedConvLayers(x, feature_maps, k_init, first_conv_stride=2):
    x = ConvDropoutNormNonlin(x, feature_maps, k_init, first_conv_stride=first_conv_stride)
    x = ConvDropoutNormNonlin(x, feature_maps, k_init)
    return x
    
def ConvDropoutNormNonlin(x, feature_maps, k_init, first_conv_stride=1):
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(feature_maps, (3, 3), strides=first_conv_stride, activation=None,
               kernel_initializer=k_init, padding='valid') (x)
    x = BatchNormalization(epsilon=1e-05, momentum=0.1) (x)
    x = LeakyReLU(alpha=0.01) (x)
    return x
