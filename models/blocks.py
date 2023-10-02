import torch 
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, conv, in_size, out_size, k_size, act=None, batch_norm=None, dropout=0, se_block=False):
        """
        Convolutional block.

        Parameters
        ----------
        conv : Torch convolutional layer
            Convolutional layer to use in the residual block. 

        in_size : array of ints
            Input feature maps of the convolutional layers.

        out_size : str, optional
            Output feature maps of the convolutional layers.

        k_size : 3 int tuple
            Height, width and depth of the convolution window.

        act : str, optional
            Activation layer to use. 

        batch_norm : nn.BatchNorm Torch layer, optional
            Batch normalization layer to use. 

        drop_value : float, optional
            Dropout value to be fixed.
        
        se_block : boolean, optional
            Whether to add Squeeze-and-Excitation blocks or not. 

        """
        super(ConvBlock, self).__init__()
        block = []

        block.append(conv(in_size, out_size, kernel_size=k_size, padding="same"))
        if batch_norm is not None:
            block.append(batch_norm(out_size))
        if act is not None:
            block.append(get_activation(act))
        if dropout > 0:
            block.append(nn.Dropout(dropout))
        if se_block:
            block.append(SqExBlock(out_size, ndim=2 if conv == nn.Conv2d else 3))

        self.block = nn.Sequential(*block)
        
    def forward(self, x):
        out = self.block(x)
        return out

class DoubleConvBlock(nn.Module):
    def __init__(self, conv, in_size, out_size, k_size, act=None, batch_norm=None, dropout=0, se_block=False):
        """
        Convolutional block.

        Parameters
        ----------
        conv : Torch convolutional layer
            Convolutional layer to use in the residual block. 

        in_size : array of ints
            Input feature maps of the convolutional layers.

        out_size : str, optional
            Output feature maps of the convolutional layers.

        k_size : 3 int tuple
            Height, width and depth of the convolution window.

        act : str, optional
            Activation layer to use. 

        batch_norm : nn.BatchNorm Torch layer, optional
            Batch normalization layer to use. 

        drop_value : float, optional
            Dropout value to be fixed.
        
        se_block : boolean, optional
            Whether to add Squeeze-and-Excitation blocks or not. 

        """
        super(DoubleConvBlock, self).__init__()
        block = []
        block.append(ConvBlock(conv=conv, in_size=in_size, out_size=out_size, k_size=k_size, act=act, 
            batch_norm=batch_norm, dropout=dropout, se_block=se_block))
        block.append(ConvBlock(conv=conv, in_size=out_size, out_size=out_size, k_size=k_size, act=act, 
            batch_norm=batch_norm, dropout=dropout, se_block=se_block))
        self.block = nn.Sequential(*block)
        
    def forward(self, x):
        out = self.block(x)
        return out

class UpBlock(nn.Module):
    def __init__(self, ndim, convtranspose, in_size, out_size, z_down, up_mode, conv, k_size, 
        act=None, batch_norm=None, dropout=0, attention_gate=False, se_block=False):
        """
        Convolutional upsampling block.

        Parameters
        ----------
        ndim : Torch convolutional layer
            Number of dimensions of the input data. 

        convtranspose : Torch convolutional layer
            Transpose convolutional layer to use. Only used if ``up_mode`` is ``'convtranspose'``. 

        in_size : array of ints
            Input feature maps of the convolutional layers.

        out_size : str, optional
            Output feature maps of the convolutional layers.

        z_down : int, optional
            Downsampling used in z dimension. 

        up_mode : str, optional
            Upsampling mode between ``'convtranspose'`` and ``'upsampling'``, which refers respectively
            to make an upsampling by appliying a transpose convolution (nn.ConvTranspose) or 
            upsampling layer (nn.Upsample). 

        conv : Torch convolutional layer
            Convolutional layer to use in the residual block. 

        k_size : 3 int tuple
            Height, width and depth of the convolution window.

        act : str, optional
            Activation layer to use. 

        batch_norm : nn.BatchNorm Torch layer, optional
            Batch normalization layer to use. 

        drop_value : float, optional
            Dropout value to be fixed.

        se_block : boolean, optional
            Whether to add Squeeze-and-Excitation blocks or not. 
        """
        super(UpBlock, self).__init__()
        self.ndim = ndim
        block = []
        if up_mode == 'convtranspose':
            mpool = (z_down, 2, 2) if ndim == 3 else (2, 2)
            block.append(convtranspose(in_size, out_size, kernel_size=mpool, stride=mpool))
        elif up_mode == 'upsampling':
            block.append(nn.Upsample(mode='bilinear' if ndim==2 else 'trilinear', scale_factor=2))
            block.append(conv(in_size, out_size, kernel_size=1))
        if batch_norm is not None:
            block.append(batch_norm(out_size))
        if act is not None:
            block.append(get_activation(act))
        self.up = nn.Sequential(*block)

        if attention_gate:
            self.attention_gate = AttentionBlock(conv=conv, in_size=out_size, out_size=out_size//2, batch_norm=batch_norm)
        else:
            self.attention_gate = None
        self.conv_block = DoubleConvBlock(conv=conv, in_size=out_size*2, out_size=out_size, k_size=k_size, 
            act=act, batch_norm=batch_norm, dropout=dropout, se_block=se_block)

    def forward(self, x, bridge):
        up = self.up(x)
        if self.attention_gate is not None:
            attn = self.attention_gate(up, bridge)
            out = torch.cat([up, attn], 1)
        else:
            out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out

class AttentionBlock(nn.Module):
    def __init__(self, conv, in_size, out_size, batch_norm=None):
        """
        Attention block.

        Reference: `Attention U-Net: Learning Where to Look for the Pancreas <https://arxiv.org/abs/1804.03999>`_.

        Parameters
        ----------
        conv : Torch convolutional layer
            Convolutional layer to use in the residual block. 

        in_size : array of ints
            Input feature maps of the convolutional layers.

        out_size : str, optional
            Output feature maps of the convolutional layers.

        batch_norm : bool, optional
            To use batch normalization.
        """
        super(AttentionBlock, self).__init__()
        w_g = []
        w_g.append(conv(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=True))
        if batch_norm is not None:
            w_g.append(batch_norm(out_size)) 
        self.w_g = nn.Sequential(*w_g)
        
        w_x = []
        w_x.append(conv(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=True))
        if batch_norm is not None:
            w_x.append(batch_norm(out_size))
        self.w_x = nn.Sequential(*w_x)

        psi = []
        psi.append(conv(out_size, 1, kernel_size=1, stride=1, padding=0, bias=True))
        if batch_norm is not None:
            psi.append(batch_norm(1))
        psi.append(nn.Sigmoid())
        self.psi = nn.Sequential(*psi)

        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        return psi*x

class SqExBlock(nn.Module):
    """
    Squeeze-and-Excitation block from `Squeeze and Excitation Networks <https://arxiv.org/abs/1709.01507>`_.

    Credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    """
    def __init__(self, c, r=16, ndim=2):
        super().__init__()
        self.ndim = ndim
        self.squeeze = nn.AdaptiveAvgPool2d(1) if ndim == 2 else nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs = x.shape[0]
        c = x.shape[1]
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y)
        if self.ndim == 2:
            y = y.view(bs, c, 1, 1)
        else:
            y = y.view(bs, c, 1, 1, 1)
        return x * y.expand_as(x)

class ResConvBlock(nn.Module):
    def __init__(self, conv, in_size, out_size, k_size, act=None, batch_norm=None, dropout=0, skip_k_size=1,
        skip_batch_norm=None, first_block=False):
        """
        Residual block.

        Parameters
        ----------
        conv : Torch convolutional layer
            Convolutional layer to use in the residual block. 

        in_size : array of ints
            Input feature maps of the convolutional layers.

        out_size : str, optional
            Output feature maps of the convolutional layers.

        k_size : 3 int tuple
            Height, width and depth of the convolution window.

        act : str, optional
            Activation layer to use. 

        batch_norm : nn.BatchNorm Torch layer, optional
            Batch normalization layer to use. 

        drop_value : float, optional
            Dropout value to be fixed.

        skip_k_size : int, optional
            Kernel size for the skip connection convolution. Used in resunet++.
        
        skip_batch_norm : nn.BatchNorm Torch layer, optional
            Batch normalization layer to use in the skip connection. Used in resunet++.

        first_block : float, optional
            To advice the function that it is the first residual block of the network, which avoids Full Pre-Activation
            layers (more info of Full Pre-Activation in `Identity Mappings in Deep Residual Networks
            <https://arxiv.org/pdf/1603.05027.pdf>`_).
        """
        super(ResConvBlock, self).__init__()
        block = []

        if not first_block:
            if batch_norm is not None:
                block.append(batch_norm(in_size))
            if act is not None:
                block.append(get_activation(act))

        block.append(ConvBlock(conv=conv, in_size=in_size, out_size=out_size, k_size=k_size, 
            act=act, batch_norm=batch_norm, dropout=dropout))
        block.append(ConvBlock(conv=conv, in_size=out_size, out_size=out_size, k_size=k_size))

        self.block = nn.Sequential(*block)

        block = []
        block.append(conv(in_size, out_size, kernel_size=skip_k_size, padding='same'))
        if skip_batch_norm is not None:
            block.append(skip_batch_norm(out_size))
        self.shortcut = nn.Sequential(*block)
    def forward(self, x):
        out = self.block(x) + self.shortcut(x)
        return out

class ResUpBlock(nn.Module):
    def __init__(self, ndim, convtranspose, in_size, out_size, in_size_bridge, z_down, up_mode, conv, k_size, 
        act=None, batch_norm=None, skip_k_size=1, skip_batch_norm=None, dropout=0):
        """
        Residual upsampling block.

        Parameters
        ----------
        ndim : Torch convolutional layer
            Number of dimensions of the input data. 

        convtranspose : Torch convolutional layer
            Transpose convolutional layer to use. Only used if ``up_mode`` is ``'convtranspose'``. 

        in_size : array of ints
            Input feature maps of the convolutional layers.

        out_size : int, optional
            Output feature maps of the convolutional layers.

        in_size_bridge : int, optional
            Output feature maps of the skip connection input. 

        z_down : int, optional
            Downsampling used in z dimension. 

        up_mode : str, optional
            Upsampling mode between ``'convtranspose'`` and ``'upsampling'``, which refers respectively
            to make an upsampling by appliying a transpose convolution (nn.ConvTranspose) or 
            upsampling layer (nn.Upsample). 

        conv : Torch convolutional layer
            Convolutional layer to use in the residual block. 

        k_size : 3 int tuple
            Height, width and depth of the convolution window.

        act : str, optional
            Activation layer to use. 

        batch_norm : nn.BatchNorm Torch layer, optional
            Batch normalization layer to use. 

        skip_k_size : int, optional
            Kernel size for the skip connection convolution. Used in resunet++.
        
        skip_batch_norm : nn.BatchNorm Torch layer, optional
            Batch normalization layer to use in the skip connection. Used in resunet++.

        drop_value : float, optional
            Dropout value to be fixed.
        """
        super(ResUpBlock, self).__init__()
        self.ndim = ndim
        mpool = (z_down, 2, 2) if ndim == 3 else (2, 2)
        if up_mode == 'convtranspose':
            self.up = convtranspose(in_size, in_size, kernel_size=mpool, stride=mpool)
        elif up_mode == 'upsampling':
            self.up = nn.Upsample(mode='bilinear' if ndim==2 else 'trilinear', scale_factor=mpool)
            
        self.conv_block = ResConvBlock(conv=conv, in_size=in_size+in_size_bridge, out_size=out_size, 
            k_size=k_size, act=act, batch_norm=batch_norm, dropout=dropout, skip_k_size=skip_k_size,
            skip_batch_norm=skip_batch_norm)

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out

def get_activation(activation: str = 'relu') -> nn.Module:
    """
    Get the specified activation layer.

    Parameters
    ----------
    activation : str, optional
        One of ``'relu'``, ``'tanh'``, ``'leaky_relu'``, ``'elu'``, ``'gelu'``,
        ``'silu'``, ``'sigmoid'``, ``'softmax'``,``'swish'``, 'efficient_swish'``, 
        ``'linear'`` and ``'none'``.
    """
    assert activation in ["relu", "tanh", "leaky_relu", "elu", "gelu", "silu", "sigmoid",
        "softmax", "linear", "none"], "Get unknown activation key {}".format(activation)
    activation_dict = {
        "relu": nn.ReLU(),
        'tanh': nn.Tanh(),
        "leaky_relu": nn.LeakyReLU(negative_slope=0.2),
        "elu": nn.ELU(alpha=1.0),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
        'sigmoid': nn.Sigmoid(),
        'softmax': nn.Softmax(dim=1),
        "linear": nn.Identity(),
        "none": nn.Identity()
    }
    return activation_dict[activation]


class ResUNetPlusPlus_AttentionBlock(nn.Module):
    """ Adapted from `here <https://github.com/rishikksh20/ResUnet/blob/master/core/modules.py>`_.
    """
    def __init__(self, conv, maxpool, input_encoder, input_decoder, output_dim, z_down=2, batch_norm=False):
        super(ResUNetPlusPlus_AttentionBlock, self).__init__()

        block = []
        if batch_norm is not None:
            block.append(batch_norm(input_encoder))
        block += [
            nn.ReLU(),
            conv(input_encoder, output_dim, 3, padding=1),
            maxpool((2, 2)) if conv == nn.Conv2d else maxpool((z_down, 2, 2))
        ]
        self.conv_encoder = nn.Sequential(*block)

        block = []
        if batch_norm is not None:
            block.append(batch_norm(input_decoder))
        block += [
            nn.ReLU(),
            conv(input_decoder, output_dim, 3, padding=1)
        ]
        self.conv_decoder = nn.Sequential(*block)

        block = []
        if batch_norm is not None:
            block.append(batch_norm(output_dim))
        block += [
            nn.ReLU(),
            conv(output_dim, 1, 1)
        ]
        self.conv_attn = nn.Sequential(*block)

    def forward(self, x1, x2):
        out = self.conv_encoder(x1) + self.conv_decoder(x2)
        out = self.conv_attn(out)
        return out * x2

class ASPP(nn.Module):
    """ Adapted from `here <https://github.com/rishikksh20/ResUnet/blob/master/core/modules.py>`_.
    """
    def __init__(self, conv, in_dims, out_dims, batch_norm=None, rate=[6, 12, 18]):
        super(ASPP, self).__init__()

        block = [
            conv(
                in_dims, out_dims, 3, stride=1, padding=rate[0], dilation=rate[0]
            ),
            nn.ReLU(inplace=True)
        ]
        if batch_norm is not None:
            block.append(batch_norm(out_dims))
        self.aspp_block1 = nn.Sequential(*block)
        block = [
            conv(
                in_dims, out_dims, 3, stride=1, padding=rate[1], dilation=rate[1]
            ),
            nn.ReLU(inplace=True),
        ]
        if batch_norm is not None:
            block.append(batch_norm(out_dims))    
        self.aspp_block2 = nn.Sequential(*block)
        block = [
            conv(
                in_dims, out_dims, 3, stride=1, padding=rate[2], dilation=rate[2]
            ),
            nn.ReLU(inplace=True)
        ]
        if batch_norm is not None:
            block.append(batch_norm(out_dims))
        self.aspp_block3 = nn.Sequential(*block)

        self.output = conv(len(rate) * out_dims, out_dims, 1)

    def forward(self, x):
        x1 = self.aspp_block1(x)
        x2 = self.aspp_block2(x)
        x3 = self.aspp_block3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        return self.output(out)