import tensorflow as tf
from tensorflow.keras.layers import Add, Input, Lambda
from tensorflow.keras.models import Model
import tensorflow_addons as tfa


class WDSRModel(tf.keras.Model): 
    """
    Code adapted from https://keras.io/examples/vision/edsr
    """
    def __init__(self, x_norm, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_norm = x_norm

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass 

            # Denormalization to calculate PSNR with original range values 
            if self.x_norm['type'] == 'div':
                y_pred = y_pred*255 if len([x for x in list(self.x_norm.keys()) if not 'reduced' in x]) > 0 else y_pred*65535
            else:
                y_pred = (y_pred * self.x_norm['std']) + self.x_norm['mean']
                y_pred = tf.round(y_pred)                                                                 
                y_pred = y_pred+abs(tf.reduce_min(y_pred))
                    
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

        # Denormalization to calculate PSNR with original range values 
        if self.x_norm['type'] == 'div':
            y_pred = y_pred*255 if len([x for x in list(self.x_norm.keys()) if not 'reduced' in x]) > 0 else y_pred*65535
        else:
            y_pred = (y_pred * self.x_norm['std']) + self.x_norm['mean']
            y_pred = tf.round(y_pred)                                                                 
            y_pred = y_pred+abs(tf.reduce_min(y_pred))

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

## RCAN network definition. We follow the code from:
### [Martin Krasser](http://krasserm.github.io/2019/09/04/super-resolution/).

def subpixel_conv2d(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)


def wdsr_a(x_norm, scale, num_filters=32, num_res_blocks=8, res_block_expansion=4, res_block_scaling=None, num_channels=1):
    return wdsr(x_norm, scale, num_filters, num_res_blocks, res_block_expansion, res_block_scaling, res_block_a, num_channels)


def wdsr_b(x_norm, scale, num_filters=32, num_res_blocks=8, res_block_expansion=6, res_block_scaling=None, num_channels=1):
    return wdsr(x_norm ,scale, num_filters, num_res_blocks, res_block_expansion, res_block_scaling, res_block_b, num_channels)


def wdsr(x_norm, scale, num_filters, num_res_blocks, res_block_expansion, res_block_scaling, res_block, num_channels=1):
    x_in = Input(shape=(None, None, num_channels))
    x = x_in
    # main branch
    m = conv2d_weightnorm(num_filters, 3, padding='same')(x)
    for i in range(num_res_blocks):
        m = res_block(m, num_filters, res_block_expansion, kernel_size=3, scaling=res_block_scaling)
    m = conv2d_weightnorm(num_channels * scale ** 2, 3, padding='same', name=f'conv2d_main_scale_{scale}')(m)
    m = Lambda(subpixel_conv2d(scale))(m)

    # skip branch
    s = conv2d_weightnorm(num_channels * scale ** 2, 5, padding='same', name=f'conv2d_skip_scale_{scale}')(x)
    s = Lambda(subpixel_conv2d(scale))(s)

    x = Add()([m, s])
    # final convolution with sigmoid activation ?
    #x = Conv2D(num_channels, 3, padding='same', activation='sigmoid')(x)

    return WDSRModel(x_norm=x_norm, inputs=x_in, outputs=x, name="wdsr")


def res_block_a(x_in, num_filters, expansion, kernel_size, scaling):
    x = conv2d_weightnorm(num_filters * expansion, kernel_size, padding='same', activation='relu')(x_in)
    x = conv2d_weightnorm(num_filters, kernel_size, padding='same')(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_in, x])
    return x


def res_block_b(x_in, num_filters, expansion, kernel_size, scaling):
    linear = 0.8
    x = conv2d_weightnorm(num_filters * expansion, 1, padding='same', activation='relu')(x_in)
    x = conv2d_weightnorm(int(num_filters * linear), 1, padding='same')(x)
    x = conv2d_weightnorm(num_filters, kernel_size, padding='same')(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_in, x])
    return x

def conv2d_weightnorm(filters, kernel_size, padding='same', activation=None, **kwargs):
    return tfa.layers.WeightNormalization(tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, activation=activation, **kwargs), data_init=False)
