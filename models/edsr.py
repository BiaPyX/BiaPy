import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

class EDSRModel(tf.keras.Model):
    """
    Code copied from https://keras.io/examples/vision/edsr
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass 

            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        y_pred = self(x, training=False)
        
        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, x):
        super_resolution_img = self(x, training=False)  
        super_resolution_img = tf.clip_by_value(super_resolution_img, 0, 1)     
        return super_resolution_img 
 
    def set_dtype(self, img):
        if not self.dtype_not_set:
            if img.dtype == np.uint16:
                self.max_value = 65535
                self.out_dtype = tf.uint16
            self.dtype_not_set = True

# Residual Block
def ResBlock(inputs):
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.Add()([inputs, x])
    return x


# Upsampling Block
def Upsampling(inputs, factor=2, **kwargs):
    f = 2 if factor == 4 else factor
    x = layers.Conv2D(64 * (f ** 2), 3, padding="same", **kwargs)(inputs)
    x = tf.nn.depth_to_space(x, block_size=f)
    if factor == 4:
        x = layers.Conv2D(64 * (f ** 2), 3, padding="same", **kwargs)(x)
        x = tf.nn.depth_to_space(x, block_size=f)
    return x


def EDSR(num_filters, num_of_residual_blocks, upsampling_factor, num_channels):
    # Flexible Inputs to input_layer
    input_layer = layers.Input(shape=(None, None, num_channels))
    x = x_new = layers.Conv2D(num_filters, 3, padding="same")(input_layer)

    # 16 residual blocks
    for _ in range(num_of_residual_blocks):
        x_new = ResBlock(x_new)

    x_new = layers.Conv2D(num_filters, 3, padding="same")(x_new)
    x = layers.Add()([x, x_new])

    x = Upsampling(x, factor=upsampling_factor)
    output_layer = layers.Conv2D(num_channels, 3, padding="same")(x)
    return EDSRModel(input_layer, output_layer)
