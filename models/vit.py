import math
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model, Input

from tensorflow.keras.layers import (Dropout, Conv2D, Conv2DTranspose, concatenate,
                                     BatchNormalization, Activation, Reshape)

from .mlp import mlp 
from .tr_patch_mgmnt import Patches, PatchEncoder

def ViT(input_shape, patch_size, num_patches, projection_dim, transformer_layers, num_heads,
        transformer_units, mlp_head_units, num_classes=1, dropout=0.0, use_as_backbone=False):
    """
    ViT architecture. `ViT paper <https://arxiv.org/abs/2010.11929>`__.

    Parameters
    ----------
    input_shape : 3D/4D tuple
        Dimensions of the input image. E.g. ``(y, x, channels)`` or ``(z, y, x, channels)``.
        
    patch_size : int
        Size of the patches that are extracted from the input image. As an example, to use ``16x16`` 
        patches, set ``patch_size = 16``.

    num_patches : int
        Number of patches to extract from the image. Take into account that each patch must be of specified patch_size.

    projection_dim : int
        Dimension of the embedding space.

    transformer_layers : int
        Number of transformer encoder layers.

    num_heads : int
        Number of heads in the multi-head attention layer.

    transformer_units : int
        Number of units in the MLP blocks.

    mlp_head_units : 2D tuple
        Size of the dense layers of the final classifier. 

    num_classes : int, optional
        Number of classes to predict. Is the number of channels in the output tensor.

    dropout : bool, optional
        Dropout rate for the decoder (can be a list of dropout rates for each layer).

    batch_norm : bool, optional
        Whether to use the model as a backbone so its components are returned instead of a composed model.
    
    Returns
    -------
    model : Keras model, optional
        Model containing the ViT .
        
    inputs : Tensorflow layer, optional
        Input layer.

    hidden_states_out : List of Tensorflow layers, optional 
        Layers of the transformer. 

    encoded_patches : PatchEncoder, optional 
        Patch enconder.
    """
    inputs = layers.Input(shape=input_shape)
    if len(input_shape) == 4:
        dims = 3   
        patch_dims = patch_size*patch_size*patch_size*input_shape[-1]
    else:
        dims = 2
        patch_dims = patch_size*patch_size*input_shape[-1]

    patches = Patches(patch_size, patch_dims, dims)(inputs)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    if use_as_backbone:
        hidden_states_out = []

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])
        if use_as_backbone:
            hidden_states_out.append(encoded_patches)
    if use_as_backbone:
        return inputs, hidden_states_out, encoded_patches

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(num_classes)(features)
    # Create the Keras model.
    model = Model(inputs=inputs, outputs=logits)

    return model