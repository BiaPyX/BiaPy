import torch
import torch.nn as nn

from models.blocks import ResConvBlock, ResUpBlock, SqExBlock, ASPP, ResUNetPlusPlus_AttentionBlock

class ResUNetPlusPlus(nn.Module):
    """
    Create 2D/3D ResUNet++.

    Reference: `ResUNet++: An Advanced Architecture for Medical Image Segmentation <https://arxiv.org/pdf/1911.07067.pdf>`_.

    Parameters
    ----------
    image_shape : 3D/4D tuple
        Dimensions of the input image. E.g. ``(y, x, channels)`` or ``(z, y, x, channels)``.

    activation : str, optional
        Activation layer.

    feature_maps : array of ints, optional
        Feature maps to use on each level.

    drop_values : float, optional
        Dropout value to be fixed.

    batch_norm : bool, optional
        Make batch normalization.

    k_size : int, optional
        Kernel size.

    upsample_layer : str, optional
        Type of layer to use to make upsampling. Two options: "convtranspose" or "upsampling". 

    z_down : List of ints, optional
        Downsampling used in z dimension. Set it to ``1`` if the dataset is not isotropic.

    n_classes: int, optional
        Number of classes.

    output_channels : str, optional
        Channels to operate with. Possible values: ``BC``, ``BCD``, ``BP``, ``BCDv2``,
        ``BDv2``, ``Dv2`` and ``BCM``.

    upsampling_factor : int, optional
        Factor of upsampling for super resolution workflow. 

    upsampling_position : str, optional
        Whether the upsampling is going to be made previously (``pre`` option) to the model 
        or after the model (``post`` option).

    Returns
    -------
    model : Torch model
        ResUNet++ model.


    Calling this function with its default parameters returns the following network:

    .. image:: ../../img/models/unet.png
        :width: 100%
        :align: center

    Image created with `PlotNeuralNet <https://github.com/HarisIqbal88/PlotNeuralNet>`_.
    """
    def __init__(self, image_shape=(256, 256, 1), activation="ELU", feature_maps=[32, 64, 128, 256], drop_values=[0.1,0.1,0.1,0.1],
        batch_norm=False, k_size=3, upsample_layer="convtranspose", z_down=[2,2,2,2], n_classes=1, 
        output_channels="BC", upsampling_factor=1, upsampling_position="pre"):
        super(ResUNetPlusPlus, self).__init__()

        self.depth = len(feature_maps)-2
        self.ndim = 3 if len(image_shape) == 4 else 2 
        self.z_down = z_down
        self.n_classes = 1 if n_classes <= 2 else n_classes
        if self.ndim == 3:
            conv = nn.Conv3d
            convtranspose = nn.ConvTranspose3d
            batchnorm_layer = nn.BatchNorm3d if batch_norm else None
            self.pooling = nn.MaxPool3d
        else:
            conv = nn.Conv2d
            convtranspose = nn.ConvTranspose2d
            batchnorm_layer = nn.BatchNorm2d if batch_norm else None
            self.pooling = nn.MaxPool2d
            
        # Super-resolution
        self.pre_upsampling = None
        if upsampling_factor > 1 and upsampling_position == "pre":
            mpool = (1, 2, 2) if self.ndim == 3 else (2, 2)
            self.pre_upsampling = convtranspose(image_shape[-1], image_shape[-1], kernel_size=mpool, stride=mpool)

        self.down_path = nn.ModuleList()

        # ENCODER
        self.down_path = nn.ModuleList()
        self.sqex_blocks = nn.ModuleList()
        self.down_path.append( 
                ResConvBlock(conv=conv, in_size=image_shape[-1], out_size=feature_maps[0], k_size=k_size, act=activation, 
                    batch_norm=batchnorm_layer, dropout=drop_values[0], skip_k_size=k_size, skip_batch_norm=batchnorm_layer, 
                    first_block=True)
            )
        self.sqex_blocks.append(SqExBlock(feature_maps[0], ndim=self.ndim))
        in_channels = feature_maps[0]
        for i in range(self.depth):
            self.down_path.append( 
                ResConvBlock(conv=conv, in_size=in_channels, out_size=feature_maps[i+1], k_size=k_size, act=activation, 
                    batch_norm=batchnorm_layer, dropout=drop_values[i], skip_k_size=k_size, skip_batch_norm=batchnorm_layer, 
                    first_block=False)
            )
            in_channels = feature_maps[i+1]
            if i != self.depth-1:
                self.sqex_blocks.append(SqExBlock(in_channels, ndim=self.ndim))

        self.aspp_bridge = ASPP(conv=conv, in_dims=in_channels, out_dims=feature_maps[-1], batch_norm=batchnorm_layer)

        # DECODER
        self.up_path = nn.ModuleList()
        self.attentions = nn.ModuleList()
        for i in range(self.depth-1, -1, -1):
            self.attentions.append(
                ResUNetPlusPlus_AttentionBlock(conv=conv, maxpool=self.pooling, input_encoder=feature_maps[i], 
                    input_decoder=feature_maps[i+2], output_dim=feature_maps[i+2], batch_norm=batchnorm_layer,
                    z_down=z_down[i+1])
            )
            self.up_path.append( 
                ResUpBlock(ndim=self.ndim, convtranspose=convtranspose, in_size=feature_maps[i+2], out_size=feature_maps[i+1], 
                    in_size_bridge=feature_maps[i], z_down=z_down[i+1], up_mode=upsample_layer, 
                    conv=conv, k_size=k_size, act=activation, batch_norm=batchnorm_layer, dropout=drop_values[i+2], 
                    skip_k_size=k_size, skip_batch_norm=batchnorm_layer)
            )
        self.aspp_out = ASPP(conv=conv, in_dims=feature_maps[1], out_dims=feature_maps[0], batch_norm=batchnorm_layer)
        
        # Super-resolution
        self.post_upsampling = None
        if upsampling_factor > 1 and upsampling_position == "post":
            mpool = (1, 2, 2) if self.ndim == 3 else (2, 2)
            self.post_upsampling = convtranspose(feature_maps[0], self.n_classes, kernel_size=mpool, stride=mpool)

        # Instance segmentation
        if output_channels is not None:
            if output_channels == "Dv2":
                self.last_block = conv(feature_maps[0], 1, kernel_size=1, padding='same')
            elif output_channels in ["BC", "BP"]:
                self.last_block = conv(feature_maps[0], 2, kernel_size=1, padding='same')
            elif output_channels in ["BDv2", "BD"]:
                self.last_block = conv(feature_maps[0], 2, kernel_size=1, padding='same')
            elif output_channels in ["BCM", "BCD", "BCDv2"]:
                self.last_block = conv(feature_maps[0], 3, kernel_size=1, padding='same')
        # Other
        else:
            self.last_block = conv(feature_maps[0], self.n_classes, kernel_size=1, padding='same')

        self.apply(self._init_weights)

    def forward(self, x):
        # Super-resolution
        if self.pre_upsampling is not None:
            x = self.pre_upsampling(x)

        # Down
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i < len(self.down_path)-1: #Avoid last block
                x = self.sqex_blocks[i](x)
            if i != len(self.down_path):
                mpool = (self.z_down[i], 2, 2) if self.ndim == 3 else (2, 2)
                if i != 0: # First level is not downsampled
                    x = self.pooling(mpool)(x) 
                blocks.append(x)

        x = self.aspp_bridge(x) 

        # Up
        for i, up in enumerate(self.up_path):
            x = self.attentions[i](blocks[-i - 2], x)
            x = up(x, blocks[-i - 2])

        x = self.aspp_out(x)

        # Super-resolution
        if self.post_upsampling is not None:
            x = self.post_upsampling(x)

        x = self.last_block(x)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)