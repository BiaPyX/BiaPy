import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import matplotlib.pyplot as plt

'''Code adapted from https://keras.io/examples/vision/masked_image_modeling and https://github.com/faustomorales/vit-keras/tree/master'''

class Patches(layers.Layer):
    def __init__(self, patch_size, patch_dims, ndim, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.patch_dims = patch_dims
        self.ndim = ndim

    def call(self, images):
        batch_size = tf.shape(images)[0]
        if self.ndim == 2:
            patches = tf.image.extract_patches(
                images=images,
                sizes=[1, self.patch_size, self.patch_size, 1],
                strides=[1, self.patch_size, self.patch_size, 1],
                rates=[1, 1, 1, 1],
                padding="VALID",
            )
        else:
            patches = tf.extract_volume_patches(
                input=images,
                ksizes=[1, self.patch_size, self.patch_size, self.patch_size, 1],
                strides=[1, self.patch_size, self.patch_size, self.patch_size, 1],
                padding="VALID",
            )
        # num_patches = images.shape[0]//patch_size
        # 2D: (B, num_patches, num_patches, patch_size*patch_size*images.shape[-1])
        # 3D: (B, num_patches, num_patches, num_patches, patch_size*patch_size*patch_size*images.shape[-1])

        patches = tf.reshape(patches, [
            batch_size if batch_size is not None else -1, 
            -1,
            np.prod(self.patch_dims)])
        # 2D: (B, num_patches*num_patches, patch_size*patch_size*images.shape[-1])
        # 3D: (B, num_patches*num_patches*num_patches, patch_size*patch_size*patch_size*images.shape[-1])
        return patches

    # taken from https://stackoverflow.com/a/58082878/10319735
    def reconstruct_from_patch(self, patch):
        # This utility function takes patches from a *single* image and
        # reconstructs it back into the image. This is useful for the train
        # monitor callback.
        num_patches = patch.shape[0]
        n = int(np.sqrt(num_patches))
        patch = tf.reshape(patch, (num_patches,) + self.patch_dims)
        rows = tf.split(patch, n, axis=0)
        rows = [tf.concat(tf.unstack(x), axis=1) for x in rows]
        reconstructed = tf.concat(rows, axis=0)
        return reconstructed

@tf.keras.utils.register_keras_serializable()
class PatchEncoder(layers.Layer):
    def __init__(self, *args, num_patches, hidden_size, patch_dims=None, mask_proportion=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_patches = num_patches
        self.hidden_size = hidden_size
        self.mask_proportion = mask_proportion
        if patch_dims is not None:
            self.patch_dims_dot = np.prod(patch_dims)

        # Create the projection layer for the patches.
        self.projection = layers.Dense(units=self.hidden_size)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=hidden_size
        )

        # This is a trainable mask token initialized randomly from a normal distribution.
        if mask_proportion != 0:
            self.mask_token = tf.Variable(tf.random.normal([1, self.patch_dims_dot]), trainable=True)
            self.num_mask = int(self.mask_proportion * self.num_patches)

    def call(self, patches):
        # Get the positional embeddings.
        batch_size = tf.shape(patches)[0]
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        pos_embeddings = self.position_embedding(positions[tf.newaxis, ...])
        pos_embeddings = tf.tile(
            pos_embeddings, [batch_size, 1, 1]
        )  # (B, num_patches, hidden_size)

        # Embed the patches.
        patch_embeddings = (
            self.projection(patches) + pos_embeddings
        )  # (B, num_patches, hidden_size)

        # No masking for ViT
        if self.mask_proportion == 0:
            return patch_embeddings

        # patch_embeddings = self.projection(patches)
        mask_indices, unmask_indices = self.get_random_indices(batch_size)
        # The encoder input is the unmasked patch embeddings. Here we gather
        # all the patches that should be unmasked.
        unmasked_embeddings = tf.gather(
            patch_embeddings, unmask_indices, axis=1, batch_dims=1
        )  # (B, unmask_numbers, hidden_size)

        # Get the unmasked and masked position embeddings. We will need them
        # for the decoder.
        unmasked_positions = tf.gather(
            pos_embeddings, unmask_indices, axis=1, batch_dims=1
        )  # (B, unmask_numbers, hidden_size)
        masked_positions = tf.gather(
            pos_embeddings, mask_indices, axis=1, batch_dims=1
        )  # (B, mask_numbers, hidden_size)

        # Repeat the mask token number of mask times.
        # Mask tokens replace the masks of the image.
        mask_tokens = tf.repeat(self.mask_token, repeats=self.num_mask, axis=0)
        mask_tokens = tf.repeat(
            mask_tokens[tf.newaxis, ...], repeats=batch_size, axis=0
        )

        # Get the masked embeddings for the tokens.
        masked_embeddings = self.projection(mask_tokens) + masked_positions
        return (
            unmasked_embeddings,  # Input to the encoder.
            masked_embeddings,  # First part of input to the decoder.
            unmasked_positions,  # Added to the encoder outputs.
            mask_indices,  # The indices that were masked.
            unmask_indices,  # The indices that were unmaksed.
        )

    def get_random_indices(self, batch_size):
        # Create random indices from a uniform distribution and then split
        # it into mask and unmask indices.
        rand_indices = tf.argsort(
            tf.random.uniform(shape=(batch_size, self.num_patches)), axis=-1
        )
        mask_indices = rand_indices[:, : self.num_mask]
        unmask_indices = rand_indices[:, self.num_mask :]
        return mask_indices, unmask_indices

    def generate_masked_image(self, patches, unmask_indices):
        # Choose a random patch and it corresponding unmask index.
        idx = np.random.choice(patches.shape[0])
        patch = patches[idx]
        unmask_index = unmask_indices[idx]

        # Build a numpy array of same shape as patch.
        new_patch = np.zeros_like(patch)

        # Iterate of the new_patch and plug the unmasked patches.
        count = 0
        for i in range(unmask_index.shape[0]):
            new_patch[unmask_index[i]] = patch[unmask_index[i]]
        return new_patch, idx

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class ClassToken(layers.Layer):
    """Append a class token to an input layer."""

    def build(self, input_shape):
        cls_init = tf.zeros_initializer()
        self.hidden_size = input_shape[-1]
        self.cls = tf.Variable(
            name="cls",
            initial_value=cls_init(shape=(1, 1, self.hidden_size), dtype="float32"),
            trainable=True,
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        cls_broadcasted = tf.cast(
            tf.broadcast_to(self.cls, [batch_size, 1, self.hidden_size]),
            dtype=inputs.dtype,
        )
        return tf.concat([cls_broadcasted, inputs], 1)

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable()
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, *args, num_heads, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads

    def build(self, input_shape):
        hidden_size = input_shape[-1]
        num_heads = self.num_heads
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {hidden_size} should be divisible by number of heads = {num_heads}"
            )
        self.hidden_size = hidden_size
        self.projection_dim = hidden_size // num_heads
        self.query_dense = layers.Dense(hidden_size, name="query")
        self.key_dense = layers.Dense(hidden_size, name="key")
        self.value_dense = layers.Dense(hidden_size, name="value")
        self.combine_heads = layers.Dense(hidden_size, name="out")

    # pylint: disable=no-self-use
    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], score.dtype)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.hidden_size))
        output = self.combine_heads(concat_attention)
        return output, weights

    def get_config(self):
        config = super().get_config()
        config.update({"num_heads": self.num_heads})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# pylint: disable=too-many-instance-attributes
@tf.keras.utils.register_keras_serializable()
class TransformerBlock(layers.Layer):
    """Implements a Transformer block."""

    def __init__(self, *args, num_heads, mlp_dim, dropout, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout

    def build(self, input_shape):
        self.att = MultiHeadSelfAttention(
            num_heads=self.num_heads,
            name="MultiHeadDotProductAttention_1",
        )
        self.mlpblock = tf.keras.Sequential(
            [
                layers.Dense(
                    self.mlp_dim,
                    activation="linear",
                    name=f"{self.name}/Dense_0",
                ),
                layers.Lambda(
                    lambda x: tf.keras.activations.gelu(x, approximate=False)
                ),
                layers.Dropout(self.dropout),
                layers.Dense(input_shape[-1], name=f"{self.name}/Dense_1"),
                layers.Dropout(self.dropout),
            ],
            name="MlpBlock_3",
        )
        self.layernorm1 = layers.LayerNormalization(
            epsilon=1e-6, name="LayerNorm_0"
        )
        self.layernorm2 = layers.LayerNormalization(
            epsilon=1e-6, name="LayerNorm_2"
        )
        self.dropout_layer = layers.Dropout(self.dropout)

    def call(self, inputs, training):
        x = self.layernorm1(inputs)
        x, weights = self.att(x)        
        x = self.dropout_layer(x, training=training)
        x = x + inputs
        y = self.layernorm2(x)
        y = self.mlpblock(y)
        return x + y, weights

    def get_config(self):
        config = super().get_config()
        
        config.update(
            {
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "dropout": self.dropout,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
