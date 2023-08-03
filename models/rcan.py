import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Activation, Add, Lambda, GlobalAveragePooling2D, Multiply, Dense, Reshape
from tensorflow.keras.models import Model

# In order to avoid RecursionError: maximum recursion depth exceeded
import sys
sys.setrecursionlimit(10000)
import warnings
warnings.filterwarnings('ignore')

class RCANModel(tf.keras.Model):
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

## RCAN network definition. We follow the code from:
### [Hoang Trung Hieu](https://github.com/hieubkset/Keras-Image-Super-Resolution/blob/master/model/rcan.py).

https://github.com/hieubkset/Keras-Image-Super-Resolution/blob/master/model/rcan.py

class Mish(tf.keras.layers.Layer):
  '''
  Mish Activation Function.
  .. math::
      mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
  Shape:
      - Input: Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.
      - Output: Same shape as the input.
  Examples:
      >>> X_input = Input(input_shape)
      >>> X = Mish()(X_input)
  '''

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def call(self, inputs):
    return inputs * tf.math.tanh(tf.math.softplus(inputs))

  def get_config(self):
    base_config = super().get_config()
    return {**base_config}

  def compute_output_shape(self, input_shape):
    return input_shape


def sub_pixel_conv2d(scale=2, **kwargs):
  return Lambda(lambda x: tf.nn.depth_to_space(x, scale), **kwargs)


def upsample(input_tensor, filters, use_mish=False):
  x = Conv2D(filters=filters * 4, kernel_size=3, strides=1, padding='same')(input_tensor)
  x = sub_pixel_conv2d(scale=2)(x)
  if use_mish:
      x = Mish()(x)
  else:
      x = Activation('relu')(x)
  return x


def ca(input_tensor, filters, reduce=16, use_mish=False):
  x = GlobalAveragePooling2D()(input_tensor)
  x = Reshape((1, 1, filters))(x)
  if use_mish:
    x = Dense(filters/reduce, kernel_initializer='he_normal', use_bias=False)(x)
    x = Mish()(x)
  else:
    x = Dense(filters/reduce,  activation='relu', kernel_initializer='he_normal', use_bias=False)(x)
  x = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(x)
  x = Multiply()([x, input_tensor])
  return x


def rcab(input_tensor, filters, scale=0.1, use_mish=False):
  x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(input_tensor)
  if use_mish:
    x = Mish()(x)
  else:
    x = Activation('relu')(x)
  x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
  x = ca(x, filters, use_mish=use_mish)
  if scale:
      x = Lambda(lambda t: t * scale)(x)
  x = Add()([x, input_tensor])

  return x


def rg(input_tensor, filters, n_rcab=20, use_mish=False):
  x = input_tensor
  for _ in range(n_rcab):
    x = rcab(x, filters, use_mish=use_mish)
  x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
  x = Add()([x, input_tensor])

  return x


def rir(input_tensor, filters, n_rg=10, use_mish=False):
    x = input_tensor
    for _ in range(n_rg):
        x = rg(x, filters=filters, use_mish=use_mish)
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    x = Add()([x, input_tensor])

    return x


def rcan(filters=64, n_sub_block=2, out_channels=1, use_mish=False):
  inputs = Input(shape=(None, None, out_channels))

  x = x_1 = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(inputs)
  x = rir(x, filters=filters, use_mish=use_mish)
  x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
  x = Add()([x_1, x])

  for _ in range(n_sub_block):
    x = upsample(x, filters, use_mish=use_mish)
  x = Conv2D(filters=out_channels, kernel_size=3, strides=1, padding='same')(x)

  return RCANModel(inputs=inputs, outputs=x)


