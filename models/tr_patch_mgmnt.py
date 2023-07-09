import tensorflow as tf
from tensorflow.keras import layers

class Patches(layers.Layer):
    # It takes a batch of images and returns a batch of patches
    def __init__(self, patch_size, patch_dims):
        super(Patches, self).__init__()
        self.patch_size = patch_size
        self.patch_dims = patch_dims

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patches = tf.reshape(patches, [
            batch_size if batch_size is not None else -1, 
            -1,
            self.patch_dims])
        return patches

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'patch_size': self.patch_size,
            'patch_dims': self.patch_dims,
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
        