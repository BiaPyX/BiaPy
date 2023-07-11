import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model, Input

from tensorflow.keras.layers import (Dropout, Conv3D, Conv2D, concatenate,
                                     BatchNormalization, Activation, Reshape)

from .mlp import mlp 
from .tr_layers import TransformerBlock, ClassToken, AddPositionEmbs


def ViT(input_shape, patch_size, hidden_size, transformer_layers, num_heads, mlp_head_units, n_classes=1, 
        dropout=0.0, include_class_token=True, representation_size=None, include_top=True, 
        use_as_backbone=False):
    """
    ViT architecture. `ViT paper <https://arxiv.org/abs/2010.11929>`__.

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

    mlp_head_units : int
        Size of the dense layer of the final classifier. 

    n_classes : int, optional
        Number of classes to predict. Is the number of channels in the output tensor.

    dropout : bool, optional
        Dropout rate for the decoder (can be a list of dropout rates for each layer).

    include_class_token : bool, optional
        Whether to include or not the class token.

    representation_size : int, optional
        The size of the representation prior to the classification layer. If None, no Dense layer is inserted.
        Not used but added to mimic vit-keras. 

    include_top : bool, optional
        Whether to include the final classification layer. If not, the output will have dimensions 
        ``(batch_size, hidden_size)``.

    use_as_backbone : bool, optional
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
    # 2D: (B, patch_size, patch_size, projection_dim)
    # 3D: (B, patch_size, patch_size, patch_size, projection_dim)

    if dims == 2:
        y = tf.keras.layers.Reshape((y.shape[1] * y.shape[2], hidden_size))(y)
    else:
        y = tf.keras.layers.Reshape((y.shape[1] * y.shape[2]* y.shape[3], hidden_size))(y)
    # 2D: (B, patch_size^2, projection_dim)
    # 3D: (B, patch_size^3, projection_dim)

    if include_class_token:
        y = ClassToken(name="class_token")(y)
        # 2D: (B, (patch_size^2)+1, projection_dim)
        # 3D: (B, (patch_size^3)+1, projection_dim)

    y = AddPositionEmbs(name="Transformer/posembed_input")(y)
    # 2D: (B, patch_size^2, projection_dim)
    # 3D: (B, patch_size^3, projection_dim)

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
        # 2D: (B, patch_size^2, projection_dim)
        # 3D: (B, patch_size^3, projection_dim)

        if use_as_backbone:
            hidden_states_out.append(y)

    if use_as_backbone:
        return inputs, hidden_states_out, y

    # Create a [batch_size, hidden_size] tensor.
    y = layers.LayerNormalization(epsilon=1e-6)(y)
    if include_class_token:
        y = tf.keras.layers.Lambda(lambda v: v[:, 0], name="ExtractToken")(y)
    if representation_size is not None:
        y = tf.keras.layers.Dense(hidden_size, name="pre_logits", activation="tanh")(y)
    if include_top:
        y = tf.keras.layers.Dense(n_classes, name="head", activation="linear")(y)
    
    model = Model(inputs=inputs, outputs=y)

    return model
