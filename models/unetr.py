import math
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.layers import Dropout, concatenate, BatchNormalization, Activation, Reshape

from .vit import ViT 

'''
UNETR BLOCKS

To make easier to read, same blocks described in the UNETR architecture are defined below:
    `UNETR paper <https://openaccess.thecvf.com/content/WACV2022/papers/Hatamizadeh_UNETR_Transformers_for_3D_Medical_Image_Segmentation_WACV_2022_paper.pdf>`__.
'''

def basic_yellow_block(x, filters, conv, activation='relu', kernel_initializer='glorot_uniform', batch_norm=True, 
    dropout=0.0):
    """
    This function takes in an input tensor, applies a convolutional layer with the specified number of
    filters, applies batch normalization, applies an activation function, and applies dropout if
    specified.
    
    Parameters
    ----------
    x : Tensor
        Input tensor.

    filters : int
        Number of filters in the convolutional layer.

    conv : Tensorflow's convolution layer
        Convolution to be made (2D or 3D).

    activation : str, optional 
        Activation function to use.

    kernel_initializer : int, optional
        Initializer for the kernel weights matrix. 

    batch_norm : bool, optional
        Whether to use batch normalization or not.

    dropout : float, optional
        Dropout rate. 
    
    Returns
    -------
    x : Tensor
        Last layer in the block.
    """
    x = conv(filters, 3, padding = 'same', kernel_initializer = kernel_initializer)(x)
    x = BatchNormalization() (x) if batch_norm else x
    x = Activation(activation) (x)
    x = Dropout(dropout)(x) if dropout > 0.0 else x
    return x

def up_green_block(x, filters, convtranspose, name=None):
    """
    This function takes in a tensor and a number of filters and returns a tensor that is the result of
    applying a 2x2 transpose convolution with the given number of filters.
    
    Parameters
    ----------
    x : Tensor
        Input tensor.

    filters : int
        Number of filters for the transpose convolutional layer.

    convtranspose : Tensorflow's tranpose convolution layer
        Convolution to be made (2D or 3D).

    name : str, optional
        Name of the layer.
    
    Returns
    -------
      The output of the convolution layer.
    """
    x = convtranspose(filters, 2, strides=2, padding='same', name=name) (x)
    return x

def mid_blue_block(x, filters, conv, convtranspose, activation='relu', kernel_initializer='glorot_uniform', batch_norm=True, 
    dropout=0.0):
    """
    This function takes in an input tensor and returns an output tensor after applying a transpose convolution 
    which upscale x2 the spatial size, and applies a convolutional layer.
    
    Parameters
    ----------
    x : Tensor
        Input tensor.

    filters : int
        Number of filters in the convolutional layers

    conv : Tensorflow's convolution layer
        Convolution to be made (2D or 3D).

    convtranspose : Tensorflow's tranpose convolution layer
        Convolution to be made (2D or 3D).

    activation: str, optional
        Activation function to use. 

    kernel_initializer: str, optional
        Initializer for the convolutional kernel weights matrix.

    batch_norm : bool, optional
        Whether to use batch normalization or not.

    dropout : float, optional
        Dropout rate.
    
    Returns
    -------
    x : Tensor
        The output of the last layer of the block.
    """
    x = up_green_block(x, filters, convtranspose)
    x = basic_yellow_block(x, filters, conv, activation=activation, kernel_initializer=kernel_initializer, 
        batch_norm=batch_norm, dropout=dropout)
    return x
    
def two_yellow(x, filters, conv, activation='relu', kernel_initializer='glorot_uniform', batch_norm=True, 
    dropout=0.0):
    """
    This function takes in an input tensor, and returns an output tensor that is the result of
    applying two basic yellow blocks to the input tensor.
    
    Parameters
    ----------
    x : Tensor
        Input tensor.

    filters : int, optional
        Number of filters in the convolutional layer.

    conv : Tensorflow's convolution layer
        Convolution to be made (2D or 3D).

    activation : str, optional
        Activation function to use.

    kernel_initializer : str, optionañ
        Initializer for the kernel weights matrix.

    batch_norm : bool, optional
        Whether to use batch normalization or not.

    dropout: bool, optional
        The dropout rate.
    
    Returns
    -------
    x : Tensor
        The output of the second basic_yellow_block.
    """
    x = basic_yellow_block(x, filters, conv, activation=activation, kernel_initializer=kernel_initializer, 
        batch_norm=batch_norm, dropout=dropout)
    x = basic_yellow_block(x, filters, conv, activation=activation, kernel_initializer=kernel_initializer, 
        batch_norm=batch_norm, dropout=0.0)
    return x


def UNETR(input_shape, patch_size, hidden_size, transformer_layers, num_heads, mlp_head_units, num_filters = 16, n_classes = 1, 
          decoder_activation = 'relu', decoder_kernel_init = 'he_normal', ViT_hidd_mult = 3, batch_norm = True, dropout = 0.0, 
          last_act='sigmoid', output_channels="BC"):
    """
    UNETR architecture. It combines a ViT with U-Net, replaces the convolutional encoder 
    with the ViT and adapt each skip connection signal to their layer's spatial dimensionality. 

    Note: Unlike the original UNETR, the sigmoid activation function is used in the last convolutional layer.

   `UNETR paper <https://openaccess.thecvf.com/content/WACV2022/papers/Hatamizadeh_UNETR_Transformers_for_3D_Medical_Image_Segmentation_WACV_2022_paper.pdf>`__.

    Parameters
    ----------
    input_shape : 3D/4D tuple
        Dimensions of the input image. E.g. ``(y, x, channels)`` or ``(z, y, x, channels)``.
        
    patch_size : int
        Size of the patches that are extracted from the input image. As an example, to use ``16x16`` 
        patches, set ``patch_size = 16``.

    hidden_size : int
        Dimension of the embedding space.

    transformer_layers : int
        Number of transformer encoder layers.

    num_heads : int
        Number of heads in the multi-head attention layer.

    mlp_head_units : 2D tuple
        Size of the dense layers of the final classifier. 

    num_filters: int, optional
        Number of filters in the first UNETR's layer of the decoder. In each layer the previous number of filters is 
        doubled.

    n_classes : int, optional
        Number of classes to predict. Is the number of channels in the output tensor.

    decoder_activation : str, optional
        Activation function for the decoder.

    decoder_kernel_init : str, optional
        Initializer for the kernel weights matrix of the convolutional layers in the decoder.

    ViT_hidd_mult : int, optional
        Multiple of the transformer encoder layers from of which the skip connection signal is going to be extracted.
        E.g. if we have ``12`` transformer encoder layers, and we set ``ViT_hidd_mult = 3``, we are going to take
        ``[1*ViT_hidd_mult, 2*ViT_hidd_mult, 3*ViT_hidd_mult]`` -> ``[Z3, Z6, Z9]`` encoder's signals. 

    batch_norm : bool, optional
        Whether to use batch normalization or not.

    dropout : bool, optional
        Dropout rate for the decoder (can be a list of dropout rates for each layer).
    
    last_act : str, optional
        Name of the last activation layer.

    output_channels : str, optional
        Channels to operate with. Possible values: ``BC``, ``BCD``, ``BP``, ``BCDv2``,
        ``BDv2``, ``Dv2`` and ``BCM``.

    Returns
    -------
    model : Keras model
        Model containing the UNETR .
    """
    
    global conv, convtranspose, maxpooling, zeropadding, upsampling
    if len(input_shape) == 4:
        ndim = 3
        from tensorflow.keras.layers import Conv3D, Conv3DTranspose
        conv = Conv3D
        convtranspose = Conv3DTranspose
    else:
        ndim = 2
        from tensorflow.keras.layers import Conv2D, Conv2DTranspose
        conv = Conv2D
        convtranspose = Conv2DTranspose

    vit_input, hidden_states_out, encoded_patches = ViT(input_shape, patch_size, hidden_size, transformer_layers, num_heads, mlp_head_units, 
        dropout=dropout, include_top=False, include_class_token=False, use_as_backbone=True)

    # UNETR Part (bottom_up, from the bottle-neck, to the output)
    total_upscale_factor = int(math.log2(patch_size))
    # make a list of dropout values if needed
    if type( dropout ) is float: 
        dropout = [dropout,]*total_upscale_factor
    
    # bottleneck
    if ndim == 2:
        z = Reshape([ input_shape[0]//patch_size, input_shape[1]//patch_size, hidden_size ])(encoded_patches) 
    else:
        z = Reshape([ input_shape[0]//patch_size, input_shape[1]//patch_size, input_shape[2]//patch_size, hidden_size ])(encoded_patches) 
    x = up_green_block(z, num_filters * (2**(total_upscale_factor-1)), convtranspose)

    for layer in reversed(range(1, total_upscale_factor)):
        # skips (with blue blocks)
        if ndim == 2:
            z = Reshape([ input_shape[0]//patch_size, input_shape[1]//patch_size, hidden_size ])( hidden_states_out[ (ViT_hidd_mult * layer) - 1 ] )
        else:
            z = Reshape([ input_shape[0]//patch_size, input_shape[1]//patch_size, input_shape[2]//patch_size, \
                hidden_size ])( hidden_states_out[ (ViT_hidd_mult * layer) - 1 ] )
        for _ in range(total_upscale_factor - layer):
            z = mid_blue_block(z, num_filters * (2**layer), conv, convtranspose, activation=decoder_activation, kernel_initializer=decoder_kernel_init, 
                batch_norm=batch_norm, dropout=dropout[layer])
        # decoder
        x = concatenate([x, z])
        x = two_yellow(x, num_filters * (2**(layer)), conv, activation=decoder_activation, kernel_initializer=decoder_kernel_init, 
            batch_norm=batch_norm, dropout=dropout[layer])
        x = up_green_block(x, num_filters * (2**(layer-1)), convtranspose)

    # first skip connection (out of transformer)
    first_skip = two_yellow(vit_input, num_filters, conv, activation=decoder_activation, kernel_initializer=decoder_kernel_init, 
        batch_norm=batch_norm, dropout=dropout[0]) 
    x = concatenate([first_skip, x])

    # UNETR output 
    x = two_yellow(x, num_filters, conv, activation=decoder_activation, kernel_initializer=decoder_kernel_init, batch_norm=batch_norm, 
        dropout=dropout[0] )

    # Instance segmentation
    if output_channels is not None:
        if output_channels == "Dv2":
            outputs = conv(1, 2, activation="linear", padding='same') (x)
        elif output_channels in ["BC", "BP"]:
            outputs = conv(2, 2, activation="sigmoid", padding='same') (x)
        elif output_channels == "BCM":
            outputs = conv(3, 2, activation="sigmoid", padding='same') (x)
        elif output_channels in ["BDv2", "BD"]:
            seg = conv(1, 2, activation="sigmoid", padding='same') (x)
            dis = conv(1, 2, activation="linear", padding='same') (x)
            outputs = Concatenate()([seg, dis])
        elif output_channels in ["BCD", "BCDv2"]:
            seg = conv(2, 2, activation="sigmoid", padding='same') (x)
            dis = conv(1, 2, activation="linear", padding='same') (x)
            outputs = Concatenate()([seg, dis])
    # Other
    else:
        outputs = conv(n_classes, 1, activation=last_act) (x)

    model = Model(inputs=vit_input, outputs=outputs)
    return model