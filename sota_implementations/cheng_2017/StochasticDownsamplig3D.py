import tensorflow as tf
import numpy as np


class StochasticDownsampling3D(tf.keras.layers.Layer):
    """Stochastic downsampling layer based on the following work:

        https://www.cs.umd.edu/~varshney/papers/Cheng_VolumeSegmentationUsingConvolutionalNeuralNetworksWithLimitedTrainingData_ICIP2017.pdf
        
        Specifically, this layer is explained in 2.3 of their paper. An example
        can be shown in Fig. 2.
    """
    def __init__(self):
        super(StochasticDownsampling3D, self).__init__()

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        shape[1] /= 2
        shape[2] /= 2
        shape[3] /= 2
        return tuple(shape)

    def call(self, inputs, t=4):
        N = tf.shape(inputs)[0]
        D = inputs.shape[1]
        H = inputs.shape[2]
        W = inputs.shape[3]
        C = inputs.shape[4]

        sv_d = int(D//t)
        sv_h = int(H//t)
        sv_w = int(W//t)
        elem = int(t/2)

        c_rows, c_cols, c_z = select_random_positions(sv_h*elem, sv_w*elem, sv_d*elem, elem, t)

        # Create the indexes which will be picked up on the layer 
        ind = [[ [(c_z[k], c_rows[i], c_cols[j]) for j in range(sv_w*elem)] for i in range(sv_h*elem)] for k in range(sv_d*elem)]
        ind_t = generate_indexes(ind, N)

        # Transform the indexes to call gather_nd function
        ta = tf.transpose(tf.stack([ind_t for i in range(C)]), [1, 2, 3, 4, 0, 5])
        ta = tf.pad(ta, [[0,0], [0,0], [0,0], [0,0], [0,0], [1, 1]])
       
        # Pick up the selected indexes to form the new layer
        r = tf.gather_nd(inputs, ta)
        
        # The shape is correctly done but this tf.reshape function is needed to
        # have a None batch size, which is needed to match the next layers 
        r = tf.reshape(r,[-1,int(D/2),int(H/2),int(W/2),C])
        return r

@tf.function
def select_random_positions(a, b, z, c, d):
    """Select random rows and columns"""
    c_rows = []
    c_cols = []
    c_z = []
    # Select random rows
    for i in range(0, a, c):
        nums = tf.sort(tf.py_function(np.random.choice, [d, c, False], tf.int32))
        for j in range(c):
            c_rows.append(nums[j] + int(i/c)*d)
    # Select random columns
    for i in range(0, b, c):
        nums = tf.sort(tf.py_function(np.random.choice, [d, c, False], tf.int32))
        for j in range(c):
            c_cols.append(nums[j] + int(i/c)*d)
    # Select random z indexes
    for i in range(0, z, c):
        nums = tf.sort(tf.py_function(np.random.choice, [d, c, False], tf.int32))
        for j in range(c):
            c_z.append(nums[j] + int(i/c)*d)
    return c_rows, c_cols, c_z

@tf.function
def generate_indexes(ind, N):
    """Concatenates given indexes N times. This is done in a separated function
       because '@tf.function' decoration is needed to deal with 'None' 
       batch_size dimension.
    """
    ind_t = tf.expand_dims(tf.identity(ind), axis=0)
    for i in tf.range(1,N):
        ind_t = tf.concat([ind_t, tf.expand_dims(ind, axis=0)], axis=0)
    return ind_t


