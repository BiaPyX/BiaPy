import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (Dropout, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Activation, 
                                     BatchNormalization)


def FC_DenseNet103(image_shape, n_filters_first_conv=48, n_pool=4, growth_rate=12, n_layers_per_block=5, dropout_p=0.2):
    """Create FC-DenseNet103 (Tiramisu network) proposed in `The One Hundred Layers Tiramisu: Fully Convolutional
       DenseNets for Semantic Segmentation <https://arxiv.org/pdf/1611.09326.pdf>`_ . Code copied from `FC-DenseNet103
       <https://github.com/SimJeg/FC-DenseNet/blob/master/FC-DenseNet.py>`_ and just adapted from Lasagne to Keras.

       The network consist of a downsampling path, where dense blocks and transition down are applied, followed by an
       upsampling path where transition up and dense blocks are applied. Skip connections are used between the
       downsampling path and the upsampling path Each layer is a composite function of BN - ReLU - Conv and the last
       layer is a softmax layer.
        
       Parameters
       ----------
       image_shape : array of 3 int
           Dimensions of the input image.

       n_filters_first_conv : int, optional
           Number of filters for the first convolution applied.

       n_pool : int, optional
           Number of pooling layers = number of transition down = number of transition up.

       growth_rate : int, optional
           Number of new feature maps created by each layer in a dense block.
       
       n_layers_per_block : array of ints, optional
           Number of layers per block. Can be an int or a list of size ``(2*n_pool)+1``.

       dropout_p : float, optional
           Dropout rate applied after each convolution (``0.0`` for not using).

       Returns
       -------
       model : Keras model
           Model containing the FC_DenseNet103.
    """
    
    if type(n_layers_per_block) == list:
        assert (len(n_layers_per_block) == 2 * n_pool + 1)
    elif type(n_layers_per_block) == int:
        n_layers_per_block = [n_layers_per_block] * (2 * n_pool + 1)
    else:
        raise ValueError

    dinamic_dim = (None,)*(len(image_shape)-1) + (image_shape[-1],)
    inputs = Input(dinamic_dim)
    #inputs = Input(image_shape)

    #####################
    # First Convolution #
    #####################

    # We perform a first convolution. All the features maps will be stored in the tensor called stack (the Tiramisu)
    stack = Conv2D(n_filters_first_conv, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(inputs)

    # The number of feature maps in the stack is stored in the variable n_filters
    n_filters = n_filters_first_conv

    #####################
    # Downsampling path #
    #####################

    skip_connection_list = []

    for i in range(n_pool):
        # Dense Block
        for j in range(n_layers_per_block[i]):
            # Compute new feature maps
            l = BN_ReLU_Conv(stack, growth_rate, dropout_p=dropout_p)
            # And stack it : the Tiramisu is growing
            stack = concatenate([stack, l])
            n_filters += growth_rate
        # At the end of the dense block, the current stack is stored in the skip_connections list
        skip_connection_list.append(stack)

        # Transition Down
        stack = TransitionDown(stack, n_filters, dropout_p)

    skip_connection_list = skip_connection_list[::-1]

    #####################
    #     Bottleneck    #
    #####################

    # We store now the output of the next dense block in a list. We will only upsample these new feature maps
    block_to_upsample = []

    # Dense Block
    for j in range(n_layers_per_block[n_pool]):
        l = BN_ReLU_Conv(stack, growth_rate, dropout_p=dropout_p)
        block_to_upsample.append(l)
        stack = concatenate([stack, l])

    #######################
    #   Upsampling path   #
    #######################

    for i in range(n_pool):
        # Transition Up ( Upsampling + concatenation with the skip connection)
        n_filters_keep = growth_rate * n_layers_per_block[n_pool + i]
        stack = TransitionUp(skip_connection_list[i], block_to_upsample, n_filters_keep)

        # Dense Block
        block_to_upsample = []
        for j in range(n_layers_per_block[n_pool + i + 1]):
            l = BN_ReLU_Conv(stack, growth_rate, dropout_p=dropout_p)
            block_to_upsample.append(l)
            stack = concatenate([stack, l])

    # Changed from the original code as there is only one class in the data used 
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (stack)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model


def BN_ReLU_Conv(inputs, n_filters, filter_size=3, dropout_p=0.2):
    """Apply successivly BatchNormalization, ReLu nonlinearity, Convolution and Dropout (if dropout_p > 0) on the inputs. """
    
    l = BatchNormalization()(inputs)
    l = Activation("relu")(l)
    l = Conv2D(n_filters, filter_size, activation=None, padding='same', kernel_initializer='he_uniform') (l)
    if dropout_p != 0.0:
        l = Dropout(dropout_p)(l)
    return l


def TransitionDown(inputs, n_filters, dropout_p=0.2):
    """Apply first a BN_ReLu_conv layer with filter size = 1, and a max pooling with a factor 2. """

    l = BN_ReLU_Conv(inputs, n_filters, filter_size=1, dropout_p=dropout_p)
    l = MaxPooling2D((2,2))(l)

    return l
    # Note : network accuracy is quite similar with average pooling or without 
    # BN - ReLU. We can also reduce the number of parameters reducing n_filters 
    # in the 1x1 convolution


def TransitionUp(skip_connection, block_to_upsample, n_filters_keep):
    """Performs upsampling on block_to_upsample by a factor 2 and concatenates it with the skip_connection. """

    # Upsample
    l = concatenate(block_to_upsample)
    l = Conv2DTranspose(n_filters_keep, (3,3), strides=(2, 2), padding='same', kernel_initializer='he_uniform') (l)

    # concatenate with skip connection
    l = concatenate([l, skip_connection])

    return l
    # Note : we also tried Subpixel Deconvolution without seeing any improvements.
    # We can reduce the number of parameters reducing n_filters_keep in the 
    # Deconvolution
