"""
This module implements a simple Convolutional Neural Network (CNN) for image classification tasks. It is designed to be a straightforward and adaptable model for both 2D and 3D image inputs.

The `simple_CNN` class constructs a network composed of two main convolutional
blocks, each followed by batch normalization, activation, pooling, and dropout.
A final dense layer with Softmax activation is used for classification.

The architecture is flexible, automatically adapting to 2D or 3D input based
on the provided `image_shape`.

Classes:
--------
- simple_CNN: The main class for creating the simple CNN model.

This module uses a helper function `get_activation` from `biapy.models.blocks`
to dynamically select the activation function.
"""
import torch.nn as nn
from typing import Dict

from biapy.models.blocks import get_activation


class simple_CNN(nn.Module):
    """
    Create a simple Convolutional Neural Network (CNN) model.

    This CNN architecture is designed for classification tasks and can handle
    both 2D and 3D image inputs. It consists of two main convolutional blocks
    followed by pooling and dropout, culminating in a fully connected layer
    for classification.

    Parameters
    ----------
    image_shape : Tuple[int, ...]
        Dimensions of the input image.
        - For 2D: `(height, width, channels)`
        - For 3D: `(depth, height, width, channels)`
        The last element `image_shape[-1]` should be the number of input channels.

    activation : str, optional
        Name of the activation layer to use within the convolutional blocks.
        Defaults to "ReLU".

    n_classes : int, optional
        Number of output classes for the classification task. Defaults to 2.

    Returns
    -------
    model : nn.Module
        The constructed simple CNN model.
    """

    def __init__(self, image_shape, activation="ReLU", n_classes=2):
        """
        Initialize the simple CNN model.

        Sets up the convolutional layers, batch normalization, pooling, dropout,
        and the final classification head based on the input image dimensions
        and specified parameters. It dynamically selects 2D or 3D layers.

        Parameters
        ----------
        image_shape : Tuple[int, ...]
            Dimensions of the input image.
            - For 2D: `(height, width, channels)`
            - For 3D: `(depth, height, width, channels)`
            The last element is the number of input channels.
        activation : str, optional
            Name of the activation layer to use (e.g., "ReLU", "ELU", "SiLU").
            Defaults to "ReLU".
        n_classes : int, optional
            Number of output classes for the classification task. Defaults to 2.
        """
        super(simple_CNN, self).__init__()
        self.ndim = 3 if len(image_shape) == 4 else 2

        if self.ndim == 3:
            conv = nn.Conv3d
            batchnorm_layer = nn.BatchNorm3d
            pool = nn.MaxPool3d
        else:
            conv = nn.Conv2d
            batchnorm_layer = nn.BatchNorm2d
            pool = nn.MaxPool2d

        firt_block_features = 32
        second_block_features = 64

        # Block 1
        activation = get_activation(activation)
        self.block1 = nn.Sequential(
            conv(image_shape[-1], firt_block_features, kernel_size=3, padding="same"),
            batchnorm_layer(firt_block_features),
            activation,
            conv(firt_block_features, firt_block_features, kernel_size=3, padding="same"),
            batchnorm_layer(firt_block_features),
            activation,
            conv(firt_block_features, firt_block_features, kernel_size=5, padding="same"),
            pool(2),
            batchnorm_layer(firt_block_features),
            activation,
            nn.Dropout(0.4),
        )

        # Block 2
        self.block2 = nn.Sequential(
            conv(
                firt_block_features,
                second_block_features,
                kernel_size=3,
                padding="same",
            ),
            activation,
            batchnorm_layer(second_block_features),
            conv(
                second_block_features,
                second_block_features,
                kernel_size=3,
                padding="same",
            ),
            activation,
            batchnorm_layer(second_block_features),
            conv(
                second_block_features,
                second_block_features,
                kernel_size=5,
                padding="same",
            ),
            pool(2),
            activation,
            batchnorm_layer(second_block_features),
            nn.Dropout(0.4),
        )

        # Last convolutional block
        if self.ndim == 2:
            h = image_shape[0] // 4
            w = image_shape[1] // 4
            f = h * w * second_block_features
        else:
            z = image_shape[0] // 4
            h = image_shape[1] // 4
            w = image_shape[2] // 4
            f = z * h * w * second_block_features

        self.last_block = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(f, n_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x) -> Dict:
        """
        Perform the forward pass of the simple CNN model.

        The input `x` passes sequentially through `block1`, `block2`, and then
        the `last_block` which flattens the features and applies a linear layer
        with Softmax for classification.

        Parameters
        ----------
        x : torch.Tensor
            The input image tensor.
            Expected shape for 2D: `(batch_size, channels, height, width)`.
            Expected shape for 3D: `(batch_size, channels, depth, height, width)`.

        Returns
        -------
        Dict
            A dictionary containing the classification probabilities.
            The key is typically 'out' or similar, mapping to a `torch.Tensor`
            of shape `(batch_size, n_classes)`.
        """
        out = self.block1(x)
        out = self.block2(out)
        out = self.last_block(out)
        return out
