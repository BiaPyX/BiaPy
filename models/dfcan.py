import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

class DFCANModel(tf.keras.Model):
    """
    	Code copied from https://keras.io/examples/vision/edsr
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dtype_not_set = False
        self.max_value = 255
        self.out_dtype = tf.uint8

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        if not self.dtype_not_set:
            if y.dtype == tf.uint16:
                self.max_value = 65535 
            self.out_dtype = y.dtype 
            self.dtype_not_set = True

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass 
            y_pred = y_pred*255

            # Compare always in [0-255] range
            if y.dtype == tf.uint16:
                y = tf.cast((y/65535)*255, tf.uint8)
            
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
        y_pred = y_pred*255

        # Compare always in [0-255] range
        if y.dtype == tf.uint16:
            y = tf.cast((y/65535)*255, tf.uint8)

        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, x):
        # Always return the same image dtype and value range of the ground truth
        # if gt is uint16 -> return uint16 image with values between [0,65535]
        # if gt is uint8 -> return uint8 image with values between [0,255]

        # Passing low resolution image to model
        super_resolution_img = self(x, training=False)  
        super_resolution_img = tf.clip_by_value(super_resolution_img, 0, 1)
        super_resolution_img = super_resolution_img*self.max_value
        super_resolution_img = tf.cast(super_resolution_img, self.out_dtype)
        
        return super_resolution_img 
 
    def set_dtype(self, img):
        if not self.dtype_not_set:
            if img.dtype == np.uint16:
                self.max_value = 65535
                self.out_dtype = tf.uint16
            self.dtype_not_set = True

## Loss function definition used in the paper from nature methods
def loss_dfcan(y_true, y_pred):
  mse = tf.keras.losses.MeanSquaredError()
  ssim = tf.image.ssim_multiscale(y_true, y_pred, max_val=1)
  res = mse(y_true, y_pred) + 0.1*(1-ssim)
  return res

## DFCAN network definition. We follow the code from:
### [Chang Qiao](https://github.com/qc17-THU/DL-SR/tree/main/src) (MIT license).
#### Common methods for both DFCAN and DFGAN adapted from `common.py`:


def gelu(x):
    cdf = 0.5 * (1.0 + tf.math.erf(x / tf.sqrt(2.0)))
    return x * cdf


def fft2d(input, gamma=0.1):
    temp = K.permute_dimensions(input, (0, 3, 1, 2))
    fft = tf.signal.fft2d(tf.complex(temp, tf.zeros_like(temp)))
    absfft = tf.math.pow(tf.math.abs(fft)+1e-8, gamma)
    output = K.permute_dimensions(absfft, (0, 2, 3, 1))
    return output


def fft3d(input, gamma=0.1):
    input = apodize3d(input, napodize=5)
    temp = K.permute_dimensions(input, (0, 4, 1, 2, 3))
    fft = tf.fft3d(tf.complex(temp, tf.zeros_like(temp)))
    absfft = tf.math.pow(tf.math.abs(fft) + 1e-8, gamma)
    output = K.permute_dimensions(absfft, (0, 2, 3, 4, 1))
    return output


def fftshift2d(input, size_psc):
    bs, h, w, ch = input.get_shape().as_list()
    fs11 = input[:, -h // 2:h, -w // 2:w, :]
    fs12 = input[:, -h // 2:h, 0:w // 2, :]
    fs21 = input[:, 0:h // 2, -w // 2:w, :]
    fs22 = input[:, 0:h // 2, 0:w // 2, :]
    output = tf.concat([tf.concat([fs11, fs21], axis=1), tf.concat([fs12, fs22], axis=1)], axis=2)
    output = tf.image.resize(output, (size_psc, size_psc))
    return output


def fftshift3d(input, size_psc=64):
    bs, h, w, z, ch = input.get_shape().as_list()
    fs111 = input[:, -h // 2:h, -w // 2:w, -z // 2 + 1:z, :]
    fs121 = input[:, -h // 2:h, 0:w // 2, -z // 2 + 1:z, :]
    fs211 = input[:, 0:h // 2, -w // 2:w, -z // 2 + 1:z, :]
    fs221 = input[:, 0:h // 2, 0:w // 2, -z // 2 + 1:z, :]
    fs112 = input[:, -h // 2:h, -w // 2:w, 0:z // 2 + 1, :]
    fs122 = input[:, -h // 2:h, 0:w // 2, 0:z // 2 + 1, :]
    fs212 = input[:, 0:h // 2, -w // 2:w, 0:z // 2 + 1, :]
    fs222 = input[:, 0:h // 2, 0:w // 2, 0:z // 2 + 1, :]
    output1 = tf.concat([tf.concat([fs111, fs211], axis=1), tf.concat([fs121, fs221], axis=1)], axis=2)
    output2 = tf.concat([tf.concat([fs112, fs212], axis=1), tf.concat([fs122, fs222], axis=1)], axis=2)
    output0 = tf.concat([output1, output2], axis=3)
    output = []
    for iz in range(z):
        output.append(tf.image.resize(output0[:, :, :, iz, :], (size_psc, size_psc)))
    output = tf.stack(output, axis=3)
    return output


def apodize2d(img, napodize=10):
    bs, ny, nx, ch = img.get_shape().as_list()
    img_apo = img[:, napodize:ny-napodize, :, :]

    imageUp = img[:, 0:napodize, :, :]
    imageDown = img[:, ny-napodize:, :, :]
    diff = (imageDown[:, -1::-1, :, :] - imageUp) / 2
    l = np.arange(napodize)
    fact_raw = 1 - np.sin((l + 0.5) / napodize * np.pi / 2)
    fact = fact_raw[np.newaxis, :, np.newaxis, np.newaxis]
    fact = tf.convert_to_tensor(fact, dtype=tf.float32)
    fact = tf.tile(fact, [tf.shape(img)[0], 1, nx, ch])
    factor = diff * fact
    imageUp = tf.add(imageUp, factor)
    imageDown = tf.subtract(imageDown, factor[:, -1::-1, :, :])
    img_apo = tf.concat([imageUp, img_apo, imageDown], axis=1)

    imageLeft = img_apo[:, :, 0:napodize, :]
    imageRight = img_apo[:, :, nx-napodize:, :]
    img_apo = img_apo[:, :, napodize:nx-napodize, :]
    diff = (imageRight[:, :, -1::-1, :] - imageLeft) / 2
    fact = fact_raw[np.newaxis, np.newaxis, :, np.newaxis]
    fact = tf.convert_to_tensor(fact, dtype=tf.float32)
    fact = tf.tile(fact, [tf.shape(img)[0], ny, 1, ch])
    factor = diff * fact
    imageLeft = tf.add(imageLeft, factor)
    imageRight = tf.subtract(imageRight, factor[:, :, -1::-1, :])
    img_apo = tf.concat([imageLeft, img_apo, imageRight], axis=2)

    return img_apo


def apodize3d(img, napodize=5):
    bs, ny, nx, nz, ch = img.get_shape().as_list()
    img_apo = img[:, napodize:ny-napodize, :, :, :]

    imageUp = img[:, 0:napodize, :, :, :]
    imageDown = img[:, ny-napodize:, :, :, :]
    diff = (imageDown[:, -1::-1, :, :, :] - imageUp) / 2
    l = np.arange(napodize)
    fact_raw = 1 - np.sin((l + 0.5) / napodize * np.pi / 2)
    fact = fact_raw[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
    fact = tf.convert_to_tensor(fact, dtype=tf.float32)
    fact = tf.tile(fact, [tf.shape(img)[0], 1, nx, nz, ch])
    factor = diff * fact
    imageUp = tf.add(imageUp, factor)
    imageDown = tf.subtract(imageDown, factor[:, -1::-1, :, :, :])
    img_apo = tf.concat([imageUp, img_apo, imageDown], axis=1)

    imageLeft = img_apo[:, :, 0:napodize, :, :]
    imageRight = img_apo[:, :, nx-napodize:, :, :]
    img_apo = img_apo[:, :, napodize:nx-napodize, :, :]
    diff = (imageRight[:, :, -1::-1, :, :] - imageLeft) / 2
    fact = fact_raw[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
    fact = tf.convert_to_tensor(fact, dtype=tf.float32)
    fact = tf.tile(fact, [tf.shape(img)[0], ny, 1, nz, ch])
    factor = diff * fact
    imageLeft = tf.add(imageLeft, factor)
    imageRight = tf.subtract(imageRight, factor[:, :, -1::-1, :, :])
    img_apo = tf.concat([imageLeft, img_apo, imageRight], axis=2)

    return img_apo


def pixel_shuffle(layer_in, scale):
    return tf.nn.depth_to_space(layer_in, block_size=scale)


def global_average_pooling2d(layer_in):
    return tf.reduce_mean(layer_in, axis=(1, 2), keepdims=True)


def global_average_pooling3d(layer_in):
    return tf.reduce_mean(layer_in, axis=(1, 2, 3), keepdims=True)


def conv_block2d(input, channel_size):
    conv = layers.Conv2D(channel_size[0], kernel_size=3, padding='same')(input)
    conv = layers.LeakyReLU(alpha=0.1)(conv)
    conv = layers.Conv2D(channel_size[1], kernel_size=3, padding='same')(conv)
    conv = layers.LeakyReLU(alpha=0.1)(conv)
    return conv


def conv_block3d(input, channel_size):
    conv = layers.Conv3D(channel_size[0], kernel_size=3, padding='same')(input)
    conv = layers.LeakyReLU(alpha=0.1)(conv)
    conv = layers.Conv3D(channel_size[1], kernel_size=3, padding='same')(conv)
    conv = layers.LeakyReLU(alpha=0.1)(conv)
    return conv


## DFCAN specific methods:

def FCALayer(input, channel, size_psc, reduction=16):
    absfft1 = layers.Lambda(fft2d, arguments={'gamma': 0.8})(input)
    absfft1 = layers.Lambda(fftshift2d, arguments={'size_psc': size_psc})(absfft1)
    absfft2 = layers.Conv2D(channel, kernel_size=3, activation='relu', padding='same')(absfft1)
    W = layers.Lambda(global_average_pooling2d)(absfft2)
    W = layers.Conv2D(channel // reduction, kernel_size=1, activation='relu', padding='same')(W)
    W = layers.Conv2D(channel, kernel_size=1, activation='sigmoid', padding='same')(W)
    mul = layers.multiply([input, W])
    return mul


def FCAB(input, channel, size_psc):
    conv = layers.Conv2D(channel, kernel_size=3, padding='same')(input)
    conv = layers.Lambda(gelu)(conv)
    conv = layers.Conv2D(channel, kernel_size=3, padding='same')(conv)
    conv = layers.Lambda(gelu)(conv)
    att = FCALayer(conv, channel, size_psc=size_psc, reduction=16)
    output = layers.add([att, input])
    return output


def ResidualGroup(input, channel, size_psc, n_RCAB = 4):
    conv = input
    for _ in range(n_RCAB):
        conv = FCAB(conv, channel=channel, size_psc=size_psc)
    conv = layers.add([conv, input])
    return conv


def DFCAN(input_shape, scale=4, n_ResGroup = 4, n_RCAB = 4, pretrained_weights=None):
    inputs = layers.Input(input_shape)
    size_psc = input_shape[0]
    conv = layers.Conv2D(64, kernel_size=3, padding='same')(inputs)
    conv = layers.Lambda(gelu)(conv)
    for _ in range(n_ResGroup):
        conv = ResidualGroup(conv, 64, size_psc, n_RCAB = 4)
    conv = layers.Conv2D(64 * (scale ** 2), kernel_size=3, padding='same')(conv)
    conv = layers.Lambda(gelu)(conv)
    upsampled = layers.Lambda(pixel_shuffle, arguments={'scale': scale})(conv)
    conv = layers.Conv2D(1, kernel_size=3, padding='same')(upsampled)
    output = layers.Activation('sigmoid')(conv)
    model = DFCANModel(inputs=inputs, outputs=output)
    return model


