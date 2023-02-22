import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Dropout, BatchNormalization, SpatialDropout2D
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, SeparableConv2D
from tensorflow.keras.layers import Multiply
from tensorflow.keras.layers import Concatenate, Add, concatenate, Lambda
from tensorflow.keras.layers import Input, UpSampling2D, Activation
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.nn import depth_to_space

class UnetModel(tf.keras.Model):
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


# Sub-pixel layer for learnable upsampling
# From: https://github.com/twairball/keras-subpixel-conv/blob/master/subpixel.py
def SubpixelConv2D(input_shape, scale=4):
    """
    Keras layer to do subpixel convolution.
    NOTE: Tensorflow backend only. Uses tf.depth_to_space
    Ref:
        [1] Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
            Shi et Al.
            https://arxiv.org/abs/1609.05158
    :param input_shape: tensor shape, (batch, height, width, channel)
    :param scale: upsampling scale. Default=4
    :return:
    """
    # upsample using depth_to_space
    def subpixel_shape(input_shape):
        dims = [input_shape[0],
                input_shape[1] * scale,
                input_shape[2] * scale,
                int(input_shape[3] / (scale ** 2))]
        output_shape = tuple(dims)
        return output_shape

    def subpixel(x):
        return depth_to_space(x, scale)


    #return Lambda(subpixel, output_shape=subpixel_shape, name='subpixel')
    return Lambda(subpixel, output_shape=subpixel_shape)

def upsample(x, out_channels=16, method='Upsampling2D', upsampling_factor=2,
             input_shape=None):
    if method == 'Conv2DTranspose':
        if input_shape is None:
            x = Conv2DTranspose(out_channels, (2, 2),
                                strides=(upsampling_factor, upsampling_factor),
                                padding='same') (x)
        else:
            x = Conv2DTranspose(out_channels, (2, 2),
                                strides=(upsampling_factor, upsampling_factor),
                                padding='same', input_shape=input_shape) (x)
    elif method == 'Upsampling2D':
        x = UpSampling2D( size=(upsampling_factor, upsampling_factor) )( x )
    elif method == 'SubpixelConv2D':
        x = Conv2D(out_channels * upsampling_factor ** 2, (3, 3),
                   padding='same')(x)
        x = SubpixelConv2D( input_shape, scale=upsampling_factor )(x)
    else:
        x = UpSampling2D( size=(upsampling_factor, upsampling_factor) )( x )

    return x

def preUNet4(filters=16, input_size = (128,128,1), upsampling_factor=2,
          spatial_dropout=False, upsample_method='UpSampling2D'):

  inputs = Input( input_size )
  
  s = upsample( inputs, out_channels=1, method=upsample_method,
                     upsampling_factor=upsampling_factor,
                     input_shape=(input_size[0], input_size[1], input_size[2]))


  conv1 = Conv2D(filters, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(s)
  conv1 = SpatialDropout2D(0.1)(conv1) if spatial_dropout else Dropout(0.1) (conv1)
  conv1 = Conv2D(filters, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
  pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)
  
  conv2 = Conv2D(filters*2, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
  conv2 = SpatialDropout2D(0.1)(conv2) if spatial_dropout else Dropout(0.1) (conv2)
  conv2 = Conv2D(filters*2, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
  pool2 = AveragePooling2D(pool_size=(2, 2))(conv2)
  
  conv3 = Conv2D(filters*4, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
  conv3 = SpatialDropout2D(0.2)(conv3) if spatial_dropout else Dropout(0.2) (conv3)
  conv3 = Conv2D(filters*4, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
  pool3 = AveragePooling2D(pool_size=(2, 2))(conv3)
  
  conv4 = Conv2D(filters*8, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
  conv4 = SpatialDropout2D(0.2)(conv4) if spatial_dropout else Dropout(0.2)(conv4)
  conv4 = Conv2D(filters*8, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
  pool4 = AveragePooling2D(pool_size=(2, 2))(conv4)

  conv5 = Conv2D(filters*16, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
  conv5 = SpatialDropout2D(0.3)(conv5) if spatial_dropout else Dropout(0.3)(conv5)
  conv5 = Conv2D(filters*16, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
  
  up6 = Conv2DTranspose(filters*8, (2, 2), strides=(2, 2), padding='same') (conv5)
  merge6 = concatenate([conv4,up6], axis = 3)
  conv6 = Conv2D(filters*8, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
  conv6 = SpatialDropout2D(0.2)(conv6) if spatial_dropout else Dropout(0.2)(conv6)
  conv6 = Conv2D(filters*8, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

  up7 = Conv2DTranspose(filters*4, (2, 2), strides=(2, 2), padding='same') (conv6)
  merge7 = concatenate([conv3,up7], axis = 3)
  conv7 = Conv2D(filters*4, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
  conv7 = SpatialDropout2D(0.2)(conv7) if spatial_dropout else Dropout(0.2)(conv7)
  conv7 = Conv2D(filters*4, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

  up8 = Conv2DTranspose(filters*2, (2, 2), strides=(2, 2), padding='same') (conv7)
  merge8 = concatenate([conv2,up8], axis = 3)
  conv8 = Conv2D(filters*2, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
  conv8 = SpatialDropout2D(0.1)(conv8) if spatial_dropout else Dropout(0.1)(conv8)
  conv8 = Conv2D(filters*2, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

  up9 = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same') (conv8)
  merge9 = concatenate([conv1,up9], axis = 3)
  conv9 = Conv2D(filters, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
  conv9 = SpatialDropout2D(0.1)(conv9) if spatial_dropout else Dropout(0.1)(conv9)
  conv9 = Conv2D(filters, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

  #outputs = Conv2D(1, (1, 1), activation='sigmoid') (conv9)
  outputs = Conv2D(1, (1, 1)) (conv9)
  
  model = UnetModel(inputs=[inputs], outputs=[outputs])
  return model

def postUNet4(filters=16, input_size = (128,128,1), upsampling_factor=2,
          spatial_dropout=False, upsample_method='UpSampling2D'):

  inputs = Input( input_size )
  
  conv1 = Conv2D(filters, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
  conv1 = SpatialDropout2D(0.1)(conv1) if spatial_dropout else Dropout(0.1) (conv1)
  conv1 = Conv2D(filters, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
  pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)
  
  conv2 = Conv2D(filters*2, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
  conv2 = SpatialDropout2D(0.1)(conv2) if spatial_dropout else Dropout(0.1) (conv2)
  conv2 = Conv2D(filters*2, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
  pool2 = AveragePooling2D(pool_size=(2, 2))(conv2)
  
  conv3 = Conv2D(filters*4, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
  conv3 = SpatialDropout2D(0.2)(conv3) if spatial_dropout else Dropout(0.2) (conv3)
  conv3 = Conv2D(filters*4, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
  pool3 = AveragePooling2D(pool_size=(2, 2))(conv3)
  
  conv4 = Conv2D(filters*8, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
  conv4 = SpatialDropout2D(0.2)(conv4) if spatial_dropout else Dropout(0.2)(conv4)
  conv4 = Conv2D(filters*8, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
  pool4 = AveragePooling2D(pool_size=(2, 2))(conv4)

  conv5 = Conv2D(filters*16, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
  conv5 = SpatialDropout2D(0.3)(conv5) if spatial_dropout else Dropout(0.3)(conv5)
  conv5 = Conv2D(filters*16, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
  
  up6 = Conv2DTranspose(filters*8, (2, 2), strides=(2, 2), padding='same') (conv5)
  merge6 = concatenate([conv4,up6], axis = 3)
  conv6 = Conv2D(filters*8, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
  conv6 = SpatialDropout2D(0.2)(conv6) if spatial_dropout else Dropout(0.2)(conv6)
  conv6 = Conv2D(filters*8, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

  up7 = Conv2DTranspose(filters*4, (2, 2), strides=(2, 2), padding='same') (conv6)
  merge7 = concatenate([conv3,up7], axis = 3)
  conv7 = Conv2D(filters*4, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
  conv7 = SpatialDropout2D(0.2)(conv7) if spatial_dropout else Dropout(0.2)(conv7)
  conv7 = Conv2D(filters*4, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

  up8 = Conv2DTranspose(filters*2, (2, 2), strides=(2, 2), padding='same') (conv7)
  merge8 = concatenate([conv2,up8], axis = 3)
  conv8 = Conv2D(filters*2, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
  conv8 = SpatialDropout2D(0.1)(conv8) if spatial_dropout else Dropout(0.1)(conv8)
  conv8 = Conv2D(filters*2, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

  up9 = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same') (conv8)
  merge9 = concatenate([conv1,up9], axis = 3)
  conv9 = Conv2D(filters, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
  conv9 = SpatialDropout2D(0.1)(conv9) if spatial_dropout else Dropout(0.1)(conv9)
  conv9 = Conv2D(filters, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

  conv9 = upsample( conv9, out_channels=1, method=upsample_method,
                    upsampling_factor=upsampling_factor,
                    input_shape=(input_size[0], input_size[1], input_size[2]))

  #outputs = Conv2D(1, (1, 1), activation='sigmoid') (conv9)
  outputs = Conv2D(1, (1, 1)) (conv9)

  model = UnetModel(inputs=[inputs], outputs=[outputs])
  return model

# == Residual U-Net ==

def residual_block(x, dim, filter_size, activation='elu', 
                   kernel_initializer='he_normal', dropout_value=0.2, bn=False,
                   first_conv_strides=1, separable_conv=False, firstBlock=False):

    # Create shorcut
    if firstBlock == False:
        shortcut = Conv2D(dim, activation=None, kernel_size=(1, 1), 
                      strides=first_conv_strides)(x)
    else:
        shortcut = Conv2D(dim, activation=None, kernel_size=(1, 1), 
                      strides=1)(x)
    
    # Main path
    if firstBlock == False:
        x = BatchNormalization()(x) if bn else x
        x = Activation( activation )(x)
    if separable_conv == False:
        if firstBlock == True:
            x = Conv2D(dim, filter_size, strides=1, activation=None,
                kernel_initializer=kernel_initializer, padding='same') (x)
        else:
            x = Conv2D(dim, filter_size, strides=first_conv_strides, activation=None,
                kernel_initializer=kernel_initializer, padding='same') (x)
    else:
        x = SeparableConv2D(dim, filter_size, strides=first_conv_strides, 
                            activation=None, kernel_initializer=kernel_initializer,
                            padding='same') (x)
    x = SpatialDropout2D( dropout_value ) (x) if dropout_value else x
    x = BatchNormalization()(x) if bn else x
    x = Activation( activation )(x)
      
    if separable_conv == False:
        x = Conv2D(dim, filter_size, activation=None,
                kernel_initializer=kernel_initializer, padding='same') (x)
    else:
        x = SeparableConv2D(dim, filter_size, activation=None,
                kernel_initializer=kernel_initializer, padding='same') (x)

    # Add shortcut value to main path
    x = Add()([shortcut, x])
    return x

def level_block(x, depth, dim, fs, ac, k, d, bn, fcs, sc, fb, mp):

    if depth > 0:
        r = residual_block(x, dim, fs, ac, k, d, bn, fcs, sc, fb)
        x = MaxPooling2D((2, 2)) (r) if mp else r
        x = level_block(x, depth-1, (dim*2), fs, ac, k, d, bn, fcs, sc, False, mp) 
        x = Conv2DTranspose(dim, (2, 2), strides=(2, 2), padding='same') (x)
        x = Concatenate()([r, x])
        x = residual_block(x, dim, fs, ac, k, d, bn, 1, sc, False)
    else:
        x = residual_block(x, dim, fs, ac, k, d, bn, fcs, sc, False)
    return x


def preResUNet(image_shape, activation='elu', kernel_initializer='he_normal',
            dropout_value=0.2, batchnorm=False, maxpooling=True, separable=False,
            numInitChannels=16, depth=4, upsampling_factor=2,
            upsample_method='UpSampling2D', final_activation=None):

    """Create the pre-upsampling ResU-Net for super-resolution
       Args:
            image_shape (array of 3 int): dimensions of the input image.
            activation (str, optional): Keras available activation type.
            kernel_initializer (str, optional): Keras available kernel 
            initializer type.
            dropout_value (real value, optional): dropout value
            batchnorm (bool, optional): use batch normalization
            maxpooling (bool, optional): use max-pooling between U-Net levels 
            (otherwise use stride of 2x2).
            separable (bool, optional): use SeparableConv2D instead of Conv2D
            numInitChannels (int, optional): number of channels at the
            first level of U-Net
            depth (int, optional): number of U-Net levels
            upsampling_factor (int, optional): initial image upsampling factor
            upsample_method (str, optional): upsampling method to use
            ('UpSampling2D', 'Conv2DTranspose', or 'SubpixelConv2D')
            final_activation (str, optional): activation function for the last
            layer
       Returns:
            model (Keras model): model containing the ResUNet created.
    """

    inputs = Input((None, None, image_shape[2]))

    s = upsample( inputs, out_channels=numInitChannels, method=upsample_method,
                  upsampling_factor=upsampling_factor,
                  input_shape=(image_shape[0], image_shape[1], image_shape[2]))

    conv_strides = (1,1) if maxpooling else (2,2)

    x = level_block(s, depth, numInitChannels, 3, activation, kernel_initializer,
                    dropout_value, batchnorm, conv_strides, separable, True,
                    maxpooling)

    #outputs = Conv2D(1, (1, 1), activation='sigmoid') (x)

    x = Add()([s,x]) # long shortcut

    outputs = Conv2D(image_shape[2], (1, 1), activation=final_activation) (x)

    model = UnetModel(inputs=[inputs], outputs=[outputs])

    return model
def postResUNet(image_shape, activation='elu', kernel_initializer='he_normal',
            dropout_value=0.2, batchnorm=False, maxpooling=True, separable=False,
            numInitChannels=16, depth=4, upsampling_factor=2,
            upsample_method='UpSampling2D', final_activation=None ):

    """Create the post-upsampling ResU-Net for super-resolution
       Args:
            image_shape (array of 3 int): dimensions of the input image.
            activation (str, optional): Keras available activation type.
            kernel_initializer (str, optional): Keras available kernel 
            initializer type.
            dropout_value (real value, optional): dropout value
            batchnorm (bool, optional): use batch normalization
            maxpooling (bool, optional): use max-pooling between U-Net levels 
            (otherwise use stride of 2x2).
            separable (bool, optional): use SeparableConv2D instead of Conv2D
            numInitChannels (int, optional): number of channels at the
            first level of U-Net
            depth (int, optional): number of U-Net levels
            upsampling_factor (int, optional): initial image upsampling factor
            upsample_method (str, optional): upsampling method to use
            ('UpSampling2D', 'Conv2DTranspose', or 'SubpixelConv2D')
            final_activation (str, optional): activation function for the last
            layer
       Returns:
            model (Keras model): model containing the ResUNet created.
    """

    inputs = Input((None, None, image_shape[2]))

    conv_strides = (1,1) if maxpooling else (2,2)

    x = level_block(inputs, depth, numInitChannels, 3, activation, kernel_initializer,
                    dropout_value, batchnorm, conv_strides, separable, True,
                    maxpooling)

    x = upsample( x, out_channels=numInitChannels, method=upsample_method,
                  upsampling_factor=upsampling_factor,
                  input_shape=(image_shape[0], image_shape[1], image_shape[2]))
    

    #outputs = Conv2D(1, (1, 1), activation='sigmoid') (x)
    outputs = Conv2D(image_shape[2], (1, 1), activation=final_activation) (x)

    model = UnetModel(inputs=[inputs], outputs=[outputs])

    return model

def preAttention_U_Net_2D(image_shape = (None,None,1), activation='elu', feature_maps=[16, 32, 64, 128, 256],
                       drop_values=[0.1,0.1,0.2,0.2,0.3], spatial_dropout=False, batch_norm=False,
                       k_init='he_normal',num_outputs=1,pre_load_weights=False,pretrained_model=None,
                       train_encoder=True,bottleneck_train=True,skip_connection_train=True,
                       upsampling_factor=2, upsample_method='UpSampling2D'):
    """Create pre-upsampling 2D U-Net with Attention blocks.
       Based on `Attention U-Net: Learning Where to Look for the Pancreas <https://arxiv.org/abs/1804.03999>`_.
       Parameters
       ----------
       image_shape : 2D tuple
           Dimensions of the input image.
       activation : str, optional
           Keras available activation type.
       feature_maps : array of ints, optional
           Feature maps to use on each level.
       drop_values : float, optional
           Dropout value to be fixed. If no value is provided the default behaviour will be to select a piramidal value
           starting from ``0.1`` and reaching ``0.3`` value.
       spatial_dropout : bool, optional
           Use spatial dropout instead of the `normal` dropout.
       batch_norm : bool, optional
           Make batch normalization.
       k_init : string, optional
           Kernel initialization for convolutional layers.
       n_classes: int, optional
           Number of classes.
       Returns
       -------
       model : Keras model
           Model containing the Attention U-Net.
       Example
       -------
       Calling this function with its default parameters returns the following network:
       .. image:: ../img/unet.png
           :width: 100%
           :align: center
       Image created with `PlotNeuralNet <https://github.com/HarisIqbal88/PlotNeuralNet>`_.
       That networks incorporates in skip connecions Attention Gates (AG), which
       can be seen as follows:
       .. image:: ../img/attention_gate.png
           :width: 100%
           :align: center
       Image extracted from `Attention U-Net: Learning Where to Look for the Pancreas <https://arxiv.org/abs/1804.03999>`_.
    """

    if len(feature_maps) != len(drop_values):
        raise ValueError("'feature_maps' dimension must be equal 'drop_values' dimension")
    depth = len(feature_maps)-1

    inputs = Input((None, None, image_shape[2]))

    x = upsample( inputs, out_channels=num_outputs, method=upsample_method,
                  upsampling_factor=upsampling_factor,
                  input_shape=(image_shape[0], image_shape[1], image_shape[2]))
    
    # List used to access layers easily to make the skip connections of the U-Net
    l=[]

    # ENCODER
    for i in range(depth):
        x = Conv2D(feature_maps[i], (3, 3), activation=None, kernel_initializer=k_init, padding='same',trainable=train_encoder) (x)
        x = BatchNormalization(trainable=train_encoder) (x) if batch_norm else x
        x = Activation(activation,trainable=train_encoder) (x)
        if drop_values is not None:
            if spatial_dropout:
                x = SpatialDropout2D(drop_values[i],trainable=train_encoder) (x)
            else:
                x = Dropout(drop_values[i],trainable=train_encoder) (x)
        x = Conv2D(feature_maps[i], (3, 3), activation=None, kernel_initializer=k_init, padding='same',trainable=train_encoder) (x)
        x = BatchNormalization(trainable=train_encoder) (x) if batch_norm else x
        x = Activation(activation,trainable=train_encoder) (x)

        l.append(x)

        x = MaxPooling2D((2, 2),trainable=train_encoder)(x)
      
    
    # BOTTLENECK
    x = Conv2D(feature_maps[depth], (3, 3), activation=None, kernel_initializer=k_init, padding='same',trainable=bottleneck_train)(x)
    x = BatchNormalization(trainable=bottleneck_train) (x) if batch_norm else x
    x = Activation(activation,trainable=bottleneck_train) (x)
    if drop_values is not None:
            if spatial_dropout:
                x = SpatialDropout2D(drop_values[depth],trainable=bottleneck_train) (x)
            else:
                x = Dropout(drop_values[depth],trainable=bottleneck_train) (x)
    x = Conv2D(feature_maps[depth], (3, 3), activation=None, kernel_initializer=k_init, padding='same',trainable=bottleneck_train) (x)
    x = BatchNormalization(trainable=bottleneck_train) (x) if batch_norm else x
    x = Activation(activation,trainable=bottleneck_train) (x)
   
    # DECODER
    for i in range(depth-1, -1, -1):
        x = Conv2DTranspose(feature_maps[i], (2, 2), strides=(2, 2), padding='same') (x)
        attn = AttentionBlock(x, l[i], feature_maps[i], batch_norm,trainable=skip_connection_train)
        x = concatenate([x, attn])
        x = Conv2D(feature_maps[i], (3, 3), activation=None, kernel_initializer=k_init, padding='same') (x)
        x = BatchNormalization() (x) if batch_norm else x
        x = Activation(activation) (x)
        if drop_values is not None:
            if spatial_dropout:
                x = SpatialDropout2D(drop_values[i]) (x)
            else:
                x = Dropout(drop_values[i]) (x)

        x = Conv2D(feature_maps[i], (3, 3), activation=None, kernel_initializer=k_init, padding='same') (x)
        x = BatchNormalization() (x) if batch_norm else x
        x = Activation(activation) (x)


    outputs = Conv2D( num_outputs, (1, 1), activation='linear') (x)
    '''
    if num_outputs==1:
         outputs = Conv2D( num_outputs, (1, 1), activation='sigmoid') (x)
    else:
         outputs = Conv2D( num_outputs, (1, 1), activation='softmax') (x)
    '''
    

    model = UnetModel(inputs=[inputs], outputs=[outputs])
    if pre_load_weights:
        #Loading weights layer by layer except from the last layer whose structure would change 
    
        for i in range((len(model.layers)-1)):
            model.get_layer(index=i).set_weights(pretrained_model.get_layer(index=i).get_weights())

    return model


def AttentionBlock(x, shortcut, filters, batch_norm,trainable=False):
    """Attention block.
       Extracted from `Kaggle <https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/64367>`_.
       Parameters
       ----------
       x : Keras layer
           Input layer.
       shortcut : Keras layer
           Input skip connection.
       filters : int
           Feature maps to define on the Conv layers.
       batch_norm : bool, optional
           To use batch normalization.
       Returns
       -------
       out : Keras layer
           Last layer of the Attention block.
    """
    g1 = Conv2D(filters, kernel_size = 1,trainable=trainable)(shortcut)
    g1 = BatchNormalization(trainable=trainable) (g1) if batch_norm else g1
    x1 = Conv2D(filters, kernel_size = 1,trainable=trainable)(x)
    x1 = BatchNormalization(trainable=trainable) (x1) if batch_norm else x1

    g1_x1 = Add(trainable=trainable)([g1,x1])
    psi = Activation('relu',trainable=trainable)(g1_x1)
    psi = Conv2D(1, kernel_size = 1,trainable=trainable)(psi)
    psi = BatchNormalization(trainable=trainable) (psi) if batch_norm else psi
    psi = Activation('sigmoid',trainable=trainable)(psi)
    x = Multiply(trainable=trainable)([x,psi])
    
    return x    

######

def pad_images_for_Unet(lr_imgs, hr_imgs, depth_Unet, is_pre, scale):
  
  lr_height = lr_imgs[0].shape[0]
  lr_width = lr_imgs[0].shape[1]

  if is_pre:
    lr_height *= scale
    lr_width *= scale

  if lr_width%2**depth_Unet != 0 or lr_height%2**depth_Unet != 0:
    height_gap = ((lr_height//2**depth_Unet) + 1) * 2**depth_Unet - lr_height
    width_gap = ((lr_width//2**depth_Unet) + 1) * 2**depth_Unet - lr_width

    if is_pre:
      height_gap //= 2
      width_gap //= 2

    height_padding = (height_gap//2 + height_gap%2, height_gap//2)
    width_padding = (width_gap//2 + width_gap%2, width_gap//2)

    if is_pre:
      if height_gap == 1:
        height_padding = (height_gap, 0)
      if width_gap == 1:
        width_padding = (width_gap, 0)

    lr_imgs = [np.pad(x, (height_padding, width_padding,(0,0)), mode="constant", constant_values=0) for x in lr_imgs]

    hr_height_padding = (height_padding[0] * 2, height_padding[1] * 2)
    hr_width_padding = (width_padding[0] * 2, width_padding[1] * 2)

    hr_imgs = [np.pad(x, (hr_height_padding, hr_width_padding, (0,0)), mode="constant", constant_values=0) for x in hr_imgs]

  return np.array(lr_imgs), np.array(hr_imgs)

