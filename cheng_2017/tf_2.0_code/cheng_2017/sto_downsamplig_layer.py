class _Pooling2D(Layer):
    """Abstract class for different pooling 2D layers.
    """

    def __init__(self, pool_size=(2, 2), strides=None, padding='valid',
                 data_format=None, **kwargs):
        super(_Pooling2D, self).__init__(**kwargs)
        if strides is None:
            strides = pool_size
        self.pool_size = conv_utils.normalize_tuple(pool_size, 2, 'pool_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = K.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            rows = input_shape[2]
            cols = input_shape[3]
        elif self.data_format == 'channels_last':
            rows = input_shape[1]
            cols = input_shape[2]
        rows = conv_utils.conv_output_length(rows, self.pool_size[0],
                                             self.padding, self.strides[0])
        cols = conv_utils.conv_output_length(cols, self.pool_size[1],
                                             self.padding, self.strides[1])
        if self.data_format == 'channels_first':
            return (input_shape[0], input_shape[1], rows, cols)
        elif self.data_format == 'channels_last':
            return (input_shape[0], rows, cols, input_shape[3])

    def _pooling_function(self, inputs, pool_size, strides,
                          padding, data_format):
        raise NotImplementedError

    def call(self, inputs):
        output = self._pooling_function(inputs=inputs,
                                        pool_size=self.pool_size,
                                        strides=self.strides,
                                        padding=self.padding,
                                        data_format=self.data_format)
        return output

    def get_config(self):
        config = {'pool_size': self.pool_size,
                  'padding': self.padding,
                  'strides': self.strides,
                  'data_format': self.data_format}
        base_config = super(_Pooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class StoDownsampling2D(_Pooling2D):
    """Max pooling operation for spatial data.
    # Arguments
        pool_size: integer or tuple of 2 integers,
            factors by which to downscale (vertical, horizontal).
            (2, 2) will halve the input in both spatial dimension.
            If only one integer is specified, the same window length
            will be used for both dimensions.
        strides: Integer, tuple of 2 integers, or None.
            Strides values.
            If None, it will default to `pool_size`.
        padding: One of `"valid"` or `"same"` (case-insensitive).
        data_format: A string,
            one of `"channels_last"` (default) or `"channels_first"`.
            The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch, height, width, channels)` while `"channels_first"`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be `"channels_last"`.
    # Input shape
        - If `data_format='channels_last'`:
            4D tensor with shape:
            `(batch_size, rows, cols, channels)`
        - If `data_format='channels_first'`:
            4D tensor with shape:
            `(batch_size, channels, rows, cols)`
    # Output shape
        - If `data_format='channels_last'`:
            4D tensor with shape:
            `(batch_size, pooled_rows, pooled_cols, channels)`
        - If `data_format='channels_first'`:
            4D tensor with shape:
            `(batch_size, channels, pooled_rows, pooled_cols)`
    """

    @interfaces.legacy_pooling2d_support
    def __init__(self, pool_size=(2, 2), strides=None, padding='valid',
                 data_format=None, **kwargs):
        super(MaxPooling2D, self).__init__(pool_size, strides, padding,
                                           data_format, **kwargs)

    def _pooling_function(self, inputs, pool_size, strides,
                          padding, data_format):
        output = K.pool2d(inputs, pool_size, strides,
                          padding, data_format,
                          pool_mode='max')
        return output


def pool2d(x, pool_size, strides=(1, 1),
           padding='valid', data_format=None,
           pool_mode='max'):
    """2D Pooling.
    # Arguments
        x: Tensor or variable.
        pool_size: tuple of 2 integers.
        strides: tuple of 2 integers.
        padding: string, `"same"` or `"valid"`.
        data_format: string, `"channels_last"` or `"channels_first"`.
        pool_mode: string, `"max"` or `"avg"`.
    # Returns
        A tensor, result of 2D pooling.
    # Raises
        ValueError: if `data_format` is
        neither `"channels_last"` or `"channels_first"`.
        ValueError: if `pool_mode` is neither `"max"` or `"avg"`.
    """
    data_format = normalize_data_format(data_format)

    x, tf_data_format = _preprocess_conv2d_input(x, data_format)
    padding = _preprocess_padding(padding)
    if tf_data_format == 'NHWC':
        strides = (1,) + strides + (1,)
        pool_size = (1,) + pool_size + (1,)
    else:
        strides = (1, 1) + strides
        pool_size = (1, 1) + pool_size

    if pool_mode == 'max':
        x = tf.nn.max_pool(x, pool_size, strides,
                           padding=padding,
                           data_format=tf_data_format)
    elif pool_mode == 'avg':
        x = tf.nn.avg_pool(x, pool_size, strides,
                           padding=padding,
                           data_format=tf_data_format)
    else:
        raise ValueError('Invalid pool_mode: ' + str(pool_mode))

    if data_format == 'channels_first' and tf_data_format == 'NHWC':
        x = tf.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
    return x


@tf_export(v1=["nn.max_pool"])
def max_pool(value,
             ksize,
             strides,
             padding,
             data_format="NHWC",
             name=None,
             input=None):  # pylint: disable=redefined-builtin
  """Performs the max pooling on the input.
  Args:
    value: A 4-D `Tensor` of the format specified by `data_format`.
    ksize: An int or list of `ints` that has length `1`, `2` or `4`.
      The size of the window for each dimension of the input tensor.
    strides: An int or list of `ints` that has length `1`, `2` or `4`.
      The stride of the sliding window for each dimension of the input tensor.
    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm.
      See the "returns" section of `tf.nn.convolution` for details.
    data_format: A string. 'NHWC', 'NCHW' and 'NCHW_VECT_C' are supported.
    name: Optional name for the operation.
    input: Alias for value.
  Returns:
    A `Tensor` of format specified by `data_format`.
    The max pooled output tensor.
  """
  value = deprecation.deprecated_argument_lookup("input", input, "value", value)
  with ops.name_scope(name, "MaxPool", [value]) as name:
    if data_format is None:
      data_format = "NHWC"
    channel_index = 1 if data_format.startswith("NC") else 3

    ksize = _get_sequence(ksize, 2, channel_index, "ksize")
    strides = _get_sequence(strides, 2, channel_index, "strides")
    if ((np.isscalar(ksize) and ksize == 0) or
        (isinstance(ksize,
                    (list, tuple, np.ndarray)) and any(v == 0 for v in ksize))):
      raise ValueError("ksize cannot be zero.")

    return gen_nn_ops.max_pool(
        value,
        ksize=ksize,
        strides=strides,
        padding=padding,
        data_format=data_format,
        name=name)

