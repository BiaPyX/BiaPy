import torch
import torch.nn as nn
import numpy as np

class EDSR(nn.Module):
    """
    Enhanced Deep Residual Networks for Single Image Super-Resolution (EDSR) model. 

    Reference: `Enhanced Deep Residual Networks for Single Image Super-Resolution <https://arxiv.org/abs/1707.02921>`_.

    Code adapted from https://keras.io/examples/vision/edsr
    """
    def __init__(self, ndim=2, num_filters=64, num_of_residual_blocks=16, upsampling_factor=2, num_channels=3):
        super(EDSR, self).__init__()
        self.ndim = ndim

        if self.ndim == 3:
            conv = nn.Conv3d
        else:
            conv = nn.Conv2d

        self.first_conv_of_block = conv(num_channels, num_filters, kernel_size=3, padding='same')

        self.resblock = nn.Sequential()
        # 16 residual blocks
        for i in range(num_of_residual_blocks):
            self.resblock.append( 
                SR_convblock(conv, num_filters)
            )
        
        self.last_conv_of_block = conv(num_filters, num_filters, kernel_size=3, padding='same')
        self.last_block = nn.Sequential(
            SR_upsampling(conv, num_filters, upsampling_factor),
            conv(num_filters, num_channels, kernel_size=3, padding='same')
        )

    def forward(self, x):
        out = x_new = self.first_conv_of_block(x)
        out = self.resblock(out)
        x_new = self.last_conv_of_block(x_new)
        out = out + x_new
        out = self.last_block(out)    

        return out

class SR_convblock(nn.Module):
    """
    Super-resolution upsampling block.

    Parameters
    ----------
    conv : Torch convolutional layer
        Convolutional layer to use.

    num_filters : Int
        Number of filter to apply in the convolutional layer. 
    """ 
    def __init__(self, conv, num_filters):
        super(SR_convblock, self).__init__()
        self.conv1 = conv(num_filters, num_filters, kernel_size=3, padding='same')
        self.conv2 = conv(num_filters, num_filters, kernel_size=3, padding='same')

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + x 
        return out

class SR_upsampling(nn.Module):
    """
    Super-resolution upsampling block.

    Parameters
    ----------
    conv : Torch convolutional layer
        Convolutional layer to use.

    num_filters : Int
        Number of filter to apply in the convolutional layer. 

    factor : int, optional
        Upscaling factor to be made to the input image. 
    """ 
    def __init__(self, conv, num_filters, factor=2):
        super(SR_upsampling, self).__init__()
        self.f = 2 if factor == 4 else factor
        self.conv1 = conv(num_filters, num_filters * (self.f ** 2), kernel_size=3, padding='same')
        self.conv2 = None
        if factor == 4:
            self.conv2 = conv(num_filters, num_filters * (self.f ** 2), kernel_size=3, padding='same')

    def forward(self, x):
        out = self.conv1(x)
        out = torch.nn.functional.pixel_shuffle(out, upscale_factor=self.f)
        if self.conv2 is not None:
            out = self.conv2(out)
            out = torch.nn.functional.pixel_shuffle(out, upscale_factor=self.f)
        return out