import math
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model

from tensorflow.keras.layers import (Dropout, Conv2D, Conv2DTranspose, concatenate,
                                     BatchNormalization, Activation, Reshape)


# Transformer utilities

class Patches(layers.Layer):
    # It takes a batch of images and returns a batch of patches
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'patch_size': self.patch_size,
        })
        return config

class PatchEncoder(layers.Layer):
    # It takes patches and projects them into a `projection_dim` dimensional space, then the position embedding is added
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_patches': self.num_patches,
            'projection': self.projection,
            'position_embedding': self.position_embedding,
        })
        return config
        
def mlp(x, hidden_units, dropout_rate):
    """
    It takes an input tensor and returns a tensor that is the result of applying a transformer multi-layer
    perceptron (MLP) block to the input
    
    Args:
      x: The input layer.
      hidden_units: A list of integers, the number of units for each mlp hidden layer. 
                    It defines the dimensionality of the output space at each mlp layer
      dropout_rate: The dropout rate to use.
    
    Returns:
      The output of the last layer.
    """
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

'''
UNETR_2D BLOCKS

To make easier to read, same blocks described in the UNETR architecture are defined below, but using 2D operations.
    UNETR paper:
        https://openaccess.thecvf.com/content/WACV2022/papers/Hatamizadeh_UNETR_Transformers_for_3D_Medical_Image_Segmentation_WACV_2022_paper.pdf
'''

def basic_yellow_block(x, filters, activation='relu', kernel_initializer='glorot_uniform', batch_norm=True, dropout=0.0):
    """
    This function takes in an input tensor, applies a convolutional layer with the specified number of
    filters, applies batch normalization, applies an activation function, and applies dropout if
    specified.
    
    Args:
      x: the input tensor
      filters: the number of filters in the convolutional layer
      activation: the activation function to use. Defaults to relu
      kernel_initializer: This is the initializer for the kernel weights matrix (see initializers).
                          Defaults to glorot_uniform
      batch_norm: Whether to use batch normalization or not. Defaults to False
      dropout: the dropout rate
    
    Returns:
      The output of the last layer in the block.
    """
    x = Conv2D(filters, (3,3), padding = 'same', kernel_initializer = kernel_initializer)(x)
    x = BatchNormalization() (x) if batch_norm else x
    x = Activation(activation) (x)
    x = Dropout(dropout)(x) if dropout > 0.0 else x
    return x

def up_green_block(x, filters, name=None):
    """
    This function takes in a tensor and a number of filters and returns a tensor that is the result of
    applying a 2x2 transpose convolution with the given number of filters.
    
    Args:
      x: the input tensor
      filters: The number of filters for the transpose convolutional layer.
      name: The name of the layer (optional).
    
    Returns:
      The output of the Conv2DTranspose layer.
    """
    x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same', name=name) (x)
    return x

def mid_blue_block(x, filters, activation='relu', kernel_initializer='glorot_uniform', batch_norm=True, dropout=0.0):
    """
    This function takes in an input tensor and returns an output tensor after applying a transpose convolution which upscale x2 the spatial size, 
    and applies a convolutional layer.
    
    Args:
      x: the input tensor
      filters: number of filters in the convolutional layers
      activation: The activation function to use. Defaults to relu
      kernel_initializer: Initializer for the convolutional kernel weights matrix (see initializers).
                          Defaults to glorot_uniform
      batch_norm: Whether to use batch normalization or not. Defaults to False
      dropout: The dropout rate.
    
    Returns:
      The output of the last layer of the block.
    """
    x = up_green_block(x, filters)
    x = basic_yellow_block(x, filters, activation=activation, kernel_initializer=kernel_initializer, batch_norm=batch_norm, dropout=dropout)
    return x
    
def two_yellow(x, filters, activation='relu', kernel_initializer='glorot_uniform', batch_norm=True, dropout=0.0):
    """
    This function takes in an input tensor, and returns an output tensor that is the result of
    applying two basic yellow blocks to the input tensor.
    
    Args:
      x: the input tensor
      filters: number of filters in the convolutional layer
      activation: The activation function to use. Defaults to relu
      kernel_initializer: Initializer for the kernel weights matrix (see initializers). 
                          Defaults to glorot_uniform
      batch_norm: Whether to use batch normalization or not. Defaults to False
      dropout: The dropout rate.
    
    Returns:
      The output of the second basic_yellow_block.
    """
    x = basic_yellow_block(x, filters, activation=activation, kernel_initializer=kernel_initializer, batch_norm=batch_norm, dropout=dropout)
    x = basic_yellow_block(x, filters, activation=activation, kernel_initializer=kernel_initializer, batch_norm=batch_norm, dropout=0.0)
    return x


def UNETR_2D(
            input_shape,
            patch_size,
            num_patches,
            projection_dim,
            transformer_layers,
            num_heads,
            transformer_units,
            data_augmentation = None,
            num_filters = 16, 
            num_classes = 1,
            decoder_activation = 'relu',
            decoder_kernel_init = 'he_normal',
            ViT_hidd_mult = 3,
            batch_norm = True,
            dropout = 0.0
        ):

    """
    UNETR architecture adapted for 2D operations. It combines a ViT with U-Net, replaces the convolutional encoder with the ViT
    and adapt each skip connection signal to their layer's spatial dimensionality. 

    Note: Unlike the original UNETR, the sigmoid activation function is used in the last convolutional layer.

    The ViT implementation is based on keras implementation:
        https://keras.io/examples/vision/image_classification_with_vision_transformer/
    Only code:
        https://github.com/keras-team/keras-io/blob/master/examples/vision/image_classification_with_vision_transformer.py

    UNETR paper:
        https://openaccess.thecvf.com/content/WACV2022/papers/Hatamizadeh_UNETR_Transformers_for_3D_Medical_Image_Segmentation_WACV_2022_paper.pdf
    
    Args:
      input_shape: the shape of the input image.
      patch_size: the size of the patches that are extracted from the input image. As an example, to use 16x16 patches, set patch_size = 16.
                  As each layer doubles the spatial size, this value must be 2^x.
      num_patches: number of patches to extract from the image. Take into account that each patch must be of specified patch_size.
      projection_dim: the dimension of the embedding space.
      transformer_layers: number of transformer encoder layers
      num_heads: number of heads in the multi-head attention layer.
      transformer_units: number of units in the MLP blocks.
      data_augmentation: a function that takes an input tensor and returns an augmented tensor. 
                         To make use of tensorflow additional data augmentation layers 
                         (use tf layer, if multiple layers, then use sequential() and add them, 
                         and use the resulting sequential layer)
      num_filters: number of filters in the first UNETR's layer of the decoder. In each layer the previous number of filters is doubled. Defaults to 16.
      num_classes: number of classes to predict. Is the number of channels in the output tensor. Defaults to 1.
      decoder_activation: activation function for the decoder. Defaults to relu.
      decoder_kernel_init: Initializer for the kernel weights matrix of the convolutional layers in the
                           decoder. Defaults to he_normal
      ViT_hidd_mult: the multiple of the transformer encoder layers from of which the skip connection signal is going to be extracted.
                     As an example, if we have 12 transformer encoder layers, and we set ViT_hidd_mult = 3, we are going to take
                     [1*ViT_hidd_mult, 2*ViT_hidd_mult, 3*ViT_hidd_mult] -> [Z3, Z6, Z9] encoder's signals. Defaults to 3.
      batch_norm: whether to use batch normalization or not. Defaults to True.
      dropout: dropout rate for the decoder (can be a list of dropout rates for each layer).
    
    Returns:
      A UNETR_2D Keras model.
    """
    
    # ViT
    inputs = layers.Input(shape=input_shape)
    # Augment data.
    augmented = data_augmentation(inputs) if data_augmentation != None else inputs
    # Create patches.
    patches = Patches(patch_size)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Hidden states
    hidden_states_out = []

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)

        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)

        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

        # save hidden state
        hidden_states_out.append(encoded_patches)

    # UNETR Part (bottom_up, from the bottle-neck, to the output)
    total_upscale_factor = int(math.log2(patch_size))
    # make a list of dropout values if needed
    if type( dropout ) is float: 
        dropout = [dropout,]*total_upscale_factor

    # bottleneck
    z = Reshape([ input_shape[0]//patch_size, input_shape[1]//patch_size, projection_dim ])(encoded_patches) # use the encoder output (easier to try different things)
    x = up_green_block(z, num_filters * (2**(total_upscale_factor-1)) )

    for layer in reversed(range(1, total_upscale_factor)):
        # skips (with blue blocks)
        z = Reshape([ input_shape[0]//patch_size, input_shape[1]//patch_size, projection_dim ])( hidden_states_out[ (ViT_hidd_mult * layer) - 1 ] )
        for _ in range(total_upscale_factor - layer):
            z = mid_blue_block(z, num_filters * (2**layer), activation=decoder_activation, kernel_initializer=decoder_kernel_init, batch_norm=batch_norm, dropout=dropout[layer])
        # decoder
        x = concatenate([x, z])
        x = two_yellow(x, num_filters * (2**(layer)), activation=decoder_activation, kernel_initializer=decoder_kernel_init, batch_norm=batch_norm, dropout=dropout[layer])
        x = up_green_block(x, num_filters * (2**(layer-1)))

    # first skip connection (out of transformer)
    first_skip = two_yellow(augmented, num_filters, activation=decoder_activation, kernel_initializer=decoder_kernel_init, batch_norm=batch_norm, dropout=dropout[0]) 
    x = concatenate([first_skip, x])

    # UNETR_2D output 
    x = two_yellow(x, num_filters, activation=decoder_activation, kernel_initializer=decoder_kernel_init, batch_norm=batch_norm, dropout=dropout[0] )
    output = Conv2D( num_classes, (1, 1), activation='sigmoid', name="mask") (x) # semantic segmentation -- ORIGINAL: softmax

    # Create the Keras model.
    model = Model(inputs=inputs, outputs=output)
    return model