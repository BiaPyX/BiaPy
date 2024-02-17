# Adapted from https://github.com/nibtehaz/MultiResUNet

import torch
import torch.nn as nn

class Conv_batchnorm(torch.nn.Module):
    """
    Convolutional layers. 

    Parameters
    ----------
    conv : Torch conv layer
        Convolutional layer to use.

    batchnorm : Torch batch normalization layer
        Convolutional layer to use.

    num_in_filters : int
        Number of input filters.

    num_out_filters : int
        Number of output filters.

    kernel_size : Tuple of ints
        Size of the convolving kernel.

    stride : Tuple of ints, optional
        Stride of the convolution.

    activation : str, optional
        Activation function.
    """
    def __init__(self, conv, batchnorm, num_in_filters, num_out_filters, kernel_size, stride = 1, activation = 'relu'):
        super().__init__()
        self.activation = activation
        self.conv1 = conv(in_channels=num_in_filters, out_channels=num_out_filters, kernel_size=kernel_size, stride=stride, padding = 'same')
        self.batchnorm = batchnorm(num_out_filters)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.batchnorm(x)
        
        if self.activation == 'relu':
            return torch.nn.functional.relu(x)
        else:
            return x

class Multiresblock(torch.nn.Module):
    """
    MultiRes Block.

    Parameters
    ----------
    conv : Torch conv layer
        Convolutional layer to use.

    batchnorm : Torch batch normalization layer
        Convolutional layer to use.

    num_in_channels : int
        Number of channels coming into multires block

    num_filters : int
        Number of output filters.

    alpha : str, optional
        Alpha hyperparameter.
    """
    def __init__(self, conv, batchnorm, num_in_channels, num_filters, alpha=1.67):
        super().__init__()
        self.alpha = alpha
        self.W = num_filters * alpha
        
        filt_cnt_3x3 = int(self.W*0.167)
        filt_cnt_5x5 = int(self.W*0.333)
        filt_cnt_7x7 = int(self.W*0.5)
        num_out_filters = filt_cnt_3x3 + filt_cnt_5x5 + filt_cnt_7x7
        
        self.shortcut = Conv_batchnorm(conv, batchnorm, num_in_channels, num_out_filters, kernel_size = 1, activation='None')

        self.conv_3x3 = Conv_batchnorm(conv, batchnorm,  num_in_channels, filt_cnt_3x3, kernel_size = 3, activation='relu')

        self.conv_5x5 = Conv_batchnorm(conv, batchnorm,  filt_cnt_3x3, filt_cnt_5x5, kernel_size = 3, activation='relu')
        
        self.conv_7x7 = Conv_batchnorm(conv, batchnorm,  filt_cnt_5x5, filt_cnt_7x7, kernel_size = 3, activation='relu')

        self.batch_norm1 = batchnorm(num_out_filters)
        self.batch_norm2 = batchnorm(num_out_filters)

    def forward(self,x):
        shrtct = self.shortcut(x)
        
        a = self.conv_3x3(x) 
        b = self.conv_5x5(a)
        c = self.conv_7x7(b)

        x = torch.cat([a,b,c],dim=1)
        x = self.batch_norm1(x)

        x = x + shrtct
        x = self.batch_norm2(x)
        x = torch.nn.functional.relu(x)
    
        return x

class Respath(torch.nn.Module):
    """
    ResPath.
    
    Parameters
    ----------
    conv : Torch conv layer
        Convolutional layer to use.

    batchnorm : Torch batch normalization layer
        Convolutional layer to use.

    num_in_filters : int
        Number of output filters.

    num_out_filters : int
        Number of filters going out the respath.

    respath_length : str, optional
        length of ResPath.
    """
    def __init__(self, conv, batchnorm, num_in_filters, num_out_filters, respath_length):
        super().__init__()

        self.respath_length = respath_length
        self.shortcuts = torch.nn.ModuleList([])
        self.convs = torch.nn.ModuleList([])
        self.bns = torch.nn.ModuleList([])

        for i in range(self.respath_length):
            if(i==0):
                self.shortcuts.append(Conv_batchnorm(conv, batchnorm, num_in_filters, num_out_filters, kernel_size = 1, activation='None'))
                self.convs.append(Conv_batchnorm(conv, batchnorm, num_in_filters, num_out_filters, kernel_size = 3,activation='relu'))
            else:
                self.shortcuts.append(Conv_batchnorm(conv, batchnorm, num_out_filters, num_out_filters, kernel_size = 1, activation='None'))
                self.convs.append(Conv_batchnorm(conv, batchnorm, num_out_filters, num_out_filters, kernel_size = 3, activation='relu'))

            self.bns.append(batchnorm(num_out_filters))
        
    
    def forward(self,x):
        for short, conv, bn in zip(self.shortcuts,self.convs,self.bns):

            shortcut = short(x)

            x = conv(x)
            x = bn(x)
            x = torch.nn.functional.relu(x)

            x = x + shortcut
            x = bn(x)
            x = torch.nn.functional.relu(x)

        return x

class MultiResUnet(torch.nn.Module):
    """
    Create 2D/3D MultiResUNet model. 

    Reference: `MultiResUNet : Rethinking the U-Net Architecture for Multimodal Biomedical Image 
    Segmentation <https://arxiv.org/abs/1902.04049>`_.
    
    Parameters
    ----------
    ndim : int
        Number of dimensions of the input data.

    input_channels: int
        Number of channels in image.

    alpha: float, optional
        Alpha hyperparameter (default: 1.67)

    n_classes: int, optional
        Number of segmentation classes.

    z_down : List of ints, optional
        Downsampling used in z dimension. Set it to ``1`` if the dataset is not isotropic.

    output_channels : str, optional
        Channels to operate with. Possible values: ``BC``, ``BCD``, ``BP``, ``BCDv2``,
        ``BDv2``, ``Dv2`` and ``BCM``.

    upsampling_factor : int, optional
        Factor of upsampling for super resolution workflow. 

    upsampling_position : str, optional
        Whether the upsampling is going to be made previously (``pre`` option) to the model 
        or after the model (``post`` option).
    """
    def __init__(self, ndim, input_channels, alpha=1.67, n_classes=1, z_down=[2,2,2,2], output_channels="BC", 
        upsampling_factor=1, upsampling_position="pre"): 
        super().__init__()
        self.ndim = ndim
        self.alpha = alpha
        self.n_classes = 1 if n_classes <= 2 else n_classes
        self.multiclass = True if n_classes > 2 and output_channels is not None else False

        if self.ndim == 3:
            conv = nn.Conv3d
            convtranspose = nn.ConvTranspose3d
            batchnorm_layer = nn.BatchNorm3d 
            pooling = nn.MaxPool3d
        else:
            conv = nn.Conv2d
            convtranspose = nn.ConvTranspose2d
            batchnorm_layer = nn.BatchNorm2d 
            pooling = nn.MaxPool2d

        # Super-resolution   
        self.pre_upsampling = None
        if upsampling_factor > 1 and upsampling_position == "pre":
            self.pre_upsampling = convtranspose(input_channels, input_channels, kernel_size=upsampling_factor, stride=upsampling_factor)

        # Encoder Path
        self.multiresblock1 = Multiresblock(conv, batchnorm_layer, input_channels,32)
        self.in_filters1 = int(32*self.alpha*0.167)+int(32*self.alpha*0.333)+int(32*self.alpha* 0.5)
        mpool = (z_down[0], 2, 2) if self.ndim == 3 else (2, 2)
        self.pool1 = pooling(mpool)
        self.respath1 = Respath(conv, batchnorm_layer, self.in_filters1,32,respath_length=4)

        self.multiresblock2 = Multiresblock(conv, batchnorm_layer, self.in_filters1,32*2)
        self.in_filters2 = int(32*2*self.alpha*0.167)+int(32*2*self.alpha*0.333)+int(32*2*self.alpha* 0.5)
        mpool = (z_down[1], 2, 2) if self.ndim == 3 else (2, 2)
        self.pool2 = pooling(mpool)
        self.respath2 = Respath(conv, batchnorm_layer,  self.in_filters2,32*2,respath_length=3)
    
        self.multiresblock3 =  Multiresblock(conv, batchnorm_layer, self.in_filters2,32*4)
        self.in_filters3 = int(32*4*self.alpha*0.167)+int(32*4*self.alpha*0.333)+int(32*4*self.alpha* 0.5)
        mpool = (z_down[2], 2, 2) if self.ndim == 3 else (2, 2)
        self.pool3 = pooling(mpool)
        self.respath3 = Respath(conv, batchnorm_layer, self.in_filters3,32*4,respath_length=2)
    
        self.multiresblock4 = Multiresblock(conv, batchnorm_layer, self.in_filters3,32*8)
        self.in_filters4 = int(32*8*self.alpha*0.167)+int(32*8*self.alpha*0.333)+int(32*8*self.alpha* 0.5)
        mpool = (z_down[3], 2, 2) if self.ndim == 3 else (2, 2)
        self.pool4 = pooling(mpool)
        self.respath4 = Respath(conv, batchnorm_layer, self.in_filters4,32*8,respath_length=1)
     
        self.multiresblock5 = Multiresblock(conv, batchnorm_layer, self.in_filters4,32*16)
        self.in_filters5 = int(32*16*self.alpha*0.167)+int(32*16*self.alpha*0.333)+int(32*16*self.alpha* 0.5)
     
        # Decoder path
        mpool = (z_down[3], 2, 2) if self.ndim == 3 else (2, 2)
        self.upsample6 = convtranspose(self.in_filters5,32*8,kernel_size=mpool,stride=mpool)  
        self.concat_filters1 = 32*8 *2
        self.multiresblock6 = Multiresblock(conv, batchnorm_layer, self.concat_filters1,32*8)
        self.in_filters6 = int(32*8*self.alpha*0.167)+int(32*8*self.alpha*0.333)+int(32*8*self.alpha* 0.5)

        mpool = (z_down[2], 2, 2) if self.ndim == 3 else (2, 2)
        self.upsample7 = convtranspose(self.in_filters6,32*4,kernel_size=mpool,stride=mpool)  
        self.concat_filters2 = 32*4 *2
        self.multiresblock7 = Multiresblock(conv, batchnorm_layer, self.concat_filters2,32*4)
        self.in_filters7 = int(32*4*self.alpha*0.167)+int(32*4*self.alpha*0.333)+int(32*4*self.alpha* 0.5)
    
        mpool = (z_down[1], 2, 2) if self.ndim == 3 else (2, 2)
        self.upsample8 = convtranspose(self.in_filters7,32*2,kernel_size=mpool,stride=mpool)
        self.concat_filters3 = 32*2 *2
        self.multiresblock8 = Multiresblock(conv, batchnorm_layer, self.concat_filters3,32*2)
        self.in_filters8 = int(32*2*self.alpha*0.167)+int(32*2*self.alpha*0.333)+int(32*2*self.alpha* 0.5)
    
        mpool = (z_down[0], 2, 2) if self.ndim == 3 else (2, 2)
        self.upsample9 = convtranspose(self.in_filters8,32,kernel_size=mpool,stride=mpool)
        self.concat_filters4 = 32 *2
        self.multiresblock9 = Multiresblock(conv, batchnorm_layer, self.concat_filters4,32)
        self.in_filters9 = int(32*self.alpha*0.167)+int(32*self.alpha*0.333)+int(32*self.alpha* 0.5)

        # Super-resolution
        self.post_upsampling = None
        if upsampling_factor > 1 and upsampling_position == "post":
            self.post_upsampling = convtranspose(self.in_filters9, self.n_classes, kernel_size=upsampling_factor, stride=upsampling_factor)

        # Instance segmentation
        if output_channels is not None:
            if output_channels == "Dv2":
                self.last_block = conv(self.in_filters9, 1, kernel_size=1, padding='same')
            elif output_channels in ["BC", "BP"]:
                self.last_block = conv(self.in_filters9, 2, kernel_size=1, padding='same')
            elif output_channels in ["BDv2", "BD"]:
                self.last_block = conv(self.in_filters9, 2, kernel_size=1, padding='same')
            elif output_channels in ["BCM", "BCD", "BCDv2"]:
                self.last_block = conv(self.in_filters9, 3, kernel_size=1, padding='same')
        # Other
        else:
            self.last_block = Conv_batchnorm(conv, batchnorm_layer, self.in_filters9, self.n_classes, kernel_size = 1, activation='None')

        # Multi-head: instances + classification
        if self.multiclass:
            self.last_class_head = conv(self.in_filters9, self.n_classes, kernel_size=1, padding='same')

    def forward(self, x : torch.Tensor)-> torch.Tensor:
        # Super-resolution
        if self.pre_upsampling is not None:
            x = self.pre_upsampling(x)

        x_multires1 = self.multiresblock1(x)
        x_pool1 = self.pool1(x_multires1)
        x_multires1 = self.respath1(x_multires1)
        
        x_multires2 = self.multiresblock2(x_pool1)
        x_pool2 = self.pool2(x_multires2)
        x_multires2 = self.respath2(x_multires2)

        x_multires3 = self.multiresblock3(x_pool2)
        x_pool3 = self.pool3(x_multires3)
        x_multires3 = self.respath3(x_multires3)

        x_multires4 = self.multiresblock4(x_pool3)
        x_pool4 = self.pool4(x_multires4)
        x_multires4 = self.respath4(x_multires4)

        x_multires5 = self.multiresblock5(x_pool4)

        up6 = torch.cat([self.upsample6(x_multires5),x_multires4],dim=1)
        x_multires6 = self.multiresblock6(up6)

        up7 = torch.cat([self.upsample7(x_multires6),x_multires3],dim=1)
        x_multires7 = self.multiresblock7(up7)

        up8 = torch.cat([self.upsample8(x_multires7),x_multires2],dim=1)
        x_multires8 = self.multiresblock8(up8)

        up9 = torch.cat([self.upsample9(x_multires8),x_multires1],dim=1)
        x_multires9 = self.multiresblock9(up9)

        # Super-resolution
        if self.post_upsampling is not None:
            x_multires9 = self.post_upsampling(x_multires9)

        class_head_out = torch.empty(())    
        if self.multiclass:
            class_head_out = self.last_class_head(x_multires9) 

        out =  self.last_block(x_multires9)
        
        # Clip values in SR
        if self.pre_upsampling is not None or self.post_upsampling is not None:
            out = torch.clamp(out, min=0, max=1)
            
        if class_head_out is not None:
            return [out, class_head_out]
        else:
            return out