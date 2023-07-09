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
    inputs = layers.Input(shape=input_shape)

    patches = Patches(patch_size, patch_size*patch_size*input_shape[-1])(inputs)
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

    if use_as_backbone:
        return model, hidden_states_out, encoded_patches
    else:
        return model