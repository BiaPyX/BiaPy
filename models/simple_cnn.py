import torch
import torch.nn as nn
from models.blocks import get_activation

class simple_CNN(nn.Module):
    def __init__(self, image_shape, activation="ReLU", n_classes=2):
        """
        Create simple CNN.

        Parameters
        ----------
        image_shape : 2D tuple
            Dimensions of the input image.
            
        activation : str, optional
            Activation layer to use in the model.  

        n_classes: int, optional
            Number of classes.

        Returns
        -------
        model : Torch model
            Model containing the simple CNN.
        """
        super(simple_CNN, self).__init__()
        self.ndim = 3 if len(image_shape) == 4 else 2 

        if self.ndim == 3:
            conv = nn.Conv3d
            convtranspose = nn.ConvTranspose3d
            batchnorm_layer = nn.BatchNorm3d
            pool = nn.MaxPool3d
        else:
            conv = nn.Conv2d
            convtranspose = nn.ConvTranspose2d
            batchnorm_layer = nn.BatchNorm2d
            pool = nn.MaxPool2d

        firt_block_features = 32
        second_block_features = 64 

        # Block 1
        activation = get_activation(activation)
        self.block1 = nn.Sequential(
            conv(image_shape[-1], firt_block_features, kernel_size=3, padding='same'),
            batchnorm_layer(firt_block_features),
            activation,
            conv(firt_block_features, firt_block_features, kernel_size=3, padding='same'),
            batchnorm_layer(firt_block_features),
            activation,
            conv(firt_block_features, firt_block_features, kernel_size=5, padding='same'),
            pool(2),
            batchnorm_layer(firt_block_features),
            activation,
            nn.Dropout(0.4)
        )

        # Block 2
        self.block2 = nn.Sequential(
            conv(firt_block_features, second_block_features, kernel_size=3, padding='same'),
            activation,
            batchnorm_layer(second_block_features),
            conv(second_block_features, second_block_features, kernel_size=3, padding='same'),
            activation,
            batchnorm_layer(second_block_features),
            conv(second_block_features, second_block_features, kernel_size=5, padding='same'),
            pool(2),
            activation,
            batchnorm_layer(second_block_features),
            nn.Dropout(0.4)
        )

        # Last convolutional block
        if self.ndim == 2:
            h = image_shape[0]//4
            w = image_shape[1]//4
            f = h*w*second_block_features
        else:
            z = image_shape[0]//4
            h = image_shape[1]//4
            w = image_shape[2]//4
            f=z*h*w*second_block_features

        self.last_block = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(f, n_classes), 
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.last_block(out)
        return out


