import math
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model, Input

from tensorflow.keras.layers import (Dropout, Conv3D, Conv2D, concatenate,
                                     BatchNormalization, Activation, Reshape)

from .mlp import mlp 
from .tr_layers import TransformerBlock, ClassToken, AddPositionEmbs


def ViT(input_shape, patch_size, num_patches, hidden_size, transformer_layers, num_heads,
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

    hidden_size : int
        Dimension of the embedding space.

    transformer_layers : int
        Number of transformer encoder layers.

    num_heads : int
        Number of heads in the multi-head attention layer.

    transformer_units : int
        Number of units in the MLP blocks.

    mlp_head_units : int
        Size of the dense layer of the final classifier. 

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
        conv = Conv3D
    else:
        dims = 2
        patch_dims = patch_size*patch_size*input_shape[-1]
        conv = Conv2D

    # Patch creation 
    y = conv(filters=hidden_size, kernel_size=patch_size, strides=patch_size, padding="valid", name="embedding")(inputs)
    if dims == 2:
        y = tf.keras.layers.Reshape((y.shape[1] * y.shape[2], hidden_size))(y)
    else:
        y = tf.keras.layers.Reshape((y.shape[1] * y.shape[2]* y.shape[3], hidden_size))(y)

    if not use_as_backbone:
        y = ClassToken(name="class_token")(y)
    y = AddPositionEmbs(name="Transformer/posembed_input")(y)

    if use_as_backbone:
        hidden_states_out = []

    # Create multiple layers of the Transformer block.
    for i in range(transformer_layers):
        y, _ = TransformerBlock(
            num_heads=num_heads,
            mlp_dim=mlp_head_units,
            dropout=0.1,
            name=f"Transformer/encoderblock_{i}",
        )(y)

        if use_as_backbone:
            hidden_states_out.append(y)

    if use_as_backbone:
        return inputs, hidden_states_out, y

    # Create a [batch_size, hidden_size] tensor.
    y = layers.LayerNormalization(epsilon=1e-6)(y)
    y = tf.keras.layers.Lambda(lambda v: v[:, 0], name="ExtractToken")(y)
    y = tf.keras.layers.Dense(hidden_size, name="pre_logits", activation="tanh")(y)
    logits = tf.keras.layers.Dense(num_classes, name="head", activation="linear")(y)

    model = Model(inputs=inputs, outputs=logits)

    return model
