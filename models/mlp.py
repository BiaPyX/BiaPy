import tensorflow as tf
from tensorflow.keras import layers

def mlp(x, hidden_units, dropout_rate):
    """
    It takes an input tensor and returns a tensor that is the result of applying a transformer multi-layer
    perceptron (MLP) block to the input
    
    Args:
      x: The input layer.
      hidden_units: A list of integers, the number of units for each mlp hidden layer. 
                    It defines the dimensionality of the output space at each mlp layer
      dropout_rate: The dropout rate to use.
    
    Returns:
      The output of the last layer.
    """
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x