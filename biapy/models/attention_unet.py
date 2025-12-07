"""
Attention U-Net Implementation.

This module implements a 2D/3D U-Net architecture with attention gates for improved 
feature learning in medical image segmentation and other computer vision tasks.

The implementation is based on the paper:
"Attention U-Net: Learning Where to Look for the Pancreas" 
https://arxiv.org/abs/1804.03999

The attention mechanism helps the model focus on relevant spatial regions while 
suppressing irrelevant background activations, leading to improved segmentation 
performance especially in medical imaging applications.

Classes:
    Attention_U_Net: Main U-Net architecture with attention gates
"""
import torch
import torch.nn as nn

from biapy.models.blocks import DoubleConvBlock, UpBlock, get_norm_2d, get_norm_3d
from typing import Dict
from biapy.models.heads import ProjectionHead

class Attention_U_Net(nn.Module):
    """
    2D/3D U-Net architecture with attention gates for enhanced feature learning.
    
    This implementation provides a flexible U-Net with attention mechanisms that can handle
    both 2D and 3D inputs, supports various normalization techniques, dropout regularization,
    and optional contrastive learning capabilities.
    
    The attention gates are integrated into the skip connections to help the model focus
    on relevant spatial regions while suppressing background noise.
    
    Attributes:
        depth (int): Number of encoder/decoder levels
        ndim (int): Number of spatial dimensions (2 or 3)
        z_down (list): Downsampling factors for z-dimension
        output_channels (list): Number of output channels for each head
        multihead (bool): Whether the model has multiple output heads
        contrast (bool): Whether contrastive learning is enabled
        down_path (nn.ModuleList): Encoder blocks
        mpooling_layers (nn.ModuleList): Max pooling layers
        bottleneck (DoubleConvBlock): Bottleneck layer at the deepest level
        up_path (nn.ModuleList): Decoder blocks with attention gates
        last_block (nn.Module): Final output layer
        proj_head (ProjectionHead, optional): Projection head for contrastive learning
        last_class_head (nn.Module, optional): Classification head for multi-head setup
        pre_upsampling (nn.Module, optional): Pre-processing upsampling layer
        post_upsampling (nn.Module, optional): Post-processing upsampling layer
    """

    def __init__(
        self,
        image_shape=(256, 256, 1),
        activation="ELU",
        feature_maps=[32, 64, 128, 256],
        drop_values=[0.1, 0.1, 0.1, 0.1],
        normalization="none",
        k_size=3,
        upsample_layer="convtranspose",
        z_down=[2, 2, 2, 2],
        output_channels=[1],
        upsampling_factor=(),
        upsampling_position="pre",
        contrast: bool = False,
        contrast_proj_dim: int = 256,
    ):
        """
        Create 2D/3D U-Net with Attention blocks.

        Reference: `Attention U-Net: Learning Where to Look for the Pancreas <https://arxiv.org/abs/1804.03999>`_.

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

        normalization : str, optional
            Normalization layer (one of ``'bn'``, ``'sync_bn'`` ``'in'``, ``'gn'`` or ``'none'``).

        k_size : int, optional
            Kernel size.

        upsample_layer : str, optional
            Type of layer to use to make upsampling. Two options: "convtranspose" or "upsampling".

        z_down : List of ints, optional
            Downsampling used in z dimension. Set it to ``1`` if the dataset is not isotropic.

        output_channels : list of int, optional
            Output channels of the network. It must be a list of lenght ``1`` or ``2``. When two
            numbers are provided two task to be done is expected (multi-head). Possible scenarios are:
            
                * instances + classification on instance segmentation
                * points + classification in detection.

        upsampling_factor : tuple of ints, optional
            Factor of upsampling for super resolution workflow for each dimension.

        upsampling_position : str, optional
            Whether the upsampling is going to be made previously (``pre`` option) to the model
            or after the model (``post`` option).

        contrast : bool, optional
            Whether to add contrastive learning head to the model. Default is ``False``.

        contrast_proj_dim : int, optional
            Dimension of the projection head for contrastive learning. Default is ``256``.

        Returns
        -------
        model : Torch model
            Attention U-Net model.


        Calling this function with its default parameters returns the following network:

        .. image:: ../../img/models/unet.png
            :width: 100%
            :align: center

        Image created with `PlotNeuralNet <https://github.com/HarisIqbal88/PlotNeuralNet>`_.

        That networks incorporates in skip connecions Attention Gates (AG), which can be seen as follows:

        .. image:: ../../img/models/attention_gate.png
            :width: 100%
            :align: center

        Image extracted from `Attention U-Net: Learning Where to Look for the Pancreas <https://arxiv.org/abs/1804.03999>`_.
        """
        super(Attention_U_Net, self).__init__()

        if len(output_channels) == 0:
            raise ValueError("'output_channels' needs to has at least one value")
        if len(output_channels) != 1 and len(output_channels) != 2:
            raise ValueError(f"'output_channels' must be a list of one or two values at max, not {output_channels}")

        self.depth = len(feature_maps) - 1
        self.ndim = 3 if len(image_shape) == 4 else 2
        self.z_down = z_down
        self.output_channels = output_channels
        self.multihead = len(output_channels) == 2
        self.contrast = contrast
        if self.ndim == 3:
            conv = nn.Conv3d
            convtranspose = nn.ConvTranspose3d
            pooling = nn.MaxPool3d
            norm_func = get_norm_3d
            dropout = nn.Dropout3d
        else:
            conv = nn.Conv2d
            convtranspose = nn.ConvTranspose2d
            pooling = nn.MaxPool2d
            norm_func = get_norm_2d
            dropout = nn.Dropout2d

        # Super-resolution
        self.pre_upsampling = None
        if len(upsampling_factor) > 1 and upsampling_position == "pre":
            self.pre_upsampling = convtranspose(
                image_shape[-1],
                image_shape[-1],
                kernel_size=upsampling_factor,
                stride=upsampling_factor,
            )

        # ENCODER
        self.down_path = nn.ModuleList()
        self.mpooling_layers = nn.ModuleList()
        in_channels = image_shape[-1]
        for i in range(self.depth):
            self.down_path.append(
                DoubleConvBlock(
                    conv,
                    in_channels,
                    feature_maps[i],
                    k_size,
                    activation,
                    normalization,
                    drop_values[i],
                )
            )
            mpool = (self.z_down[i], 2, 2) if self.ndim == 3 else (2, 2)
            self.mpooling_layers.append(pooling(mpool))
            in_channels = feature_maps[i]

        self.bottleneck = DoubleConvBlock(
            conv,
            in_channels,
            feature_maps[-1],
            k_size,
            activation,
            normalization,
            drop_values[-1],
        )

        # DECODER
        self.up_path = nn.ModuleList()
        in_channels = feature_maps[-1]
        for i in range(self.depth - 1, -1, -1):
            self.up_path.append(
                UpBlock(
                    self.ndim,
                    convtranspose,
                    in_channels,
                    feature_maps[i],
                    z_down[i],
                    upsample_layer,
                    conv,
                    k_size,
                    activation,
                    normalization,
                    drop_values[i],
                    attention_gate=True,
                )
            )
            in_channels = feature_maps[i]

        # Super-resolution
        self.post_upsampling = None
        if len(upsampling_factor) > 1 and upsampling_position == "post":
            self.post_upsampling = convtranspose(
                feature_maps[0],
                feature_maps[0],
                kernel_size=upsampling_factor,
                stride=upsampling_factor,
            )

        if self.contrast:
            # extra added layers
            self.last_block = nn.Sequential(
                conv(feature_maps[0], feature_maps[0], kernel_size=3, stride=1, padding=1),
                norm_func(normalization, feature_maps[0]),
                dropout(0.10),
                conv(feature_maps[0], output_channels[0], kernel_size=1, stride=1, padding=0, bias=False),
            )

            self.proj_head = ProjectionHead(ndim=self.ndim, in_channels=feature_maps[0], proj_dim=contrast_proj_dim)
        else:
            self.last_block = conv(feature_maps[0], output_channels[0], kernel_size=1, padding="same")

        # Multi-head:
        #   Instance segmentation: instances + classification
        #   Detection: points + classification
        self.last_class_head = None
        if self.multihead == 2:
            self.last_class_head = conv(feature_maps[0], output_channels[1], kernel_size=1, padding="same")

        self.apply(self._init_weights)

    def forward(self, x) -> Dict | torch.Tensor:
        r"""
        Forward pass through the Attention U-Net.

        Processes input through encoder-decoder architecture with attention gates
        in skip connections. Returns either a single tensor for basic segmentation
        or a dictionary of outputs for multi-head or contrastive learning setups.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape:

            - 2D: (batch_size, channels, height, width)
            - 3D: (batch_size, channels, depth, height, width)

        Returns
        -------
        torch.Tensor or Dict[str, torch.Tensor]
            For single-head without contrastive learning:
                torch.Tensor of shape (batch_size, output_channels, \*spatial_dims)
            
            For multi-head or contrastive learning:
                Dictionary containing:

                - "pred": Main prediction tensor
                - "embed": Feature embeddings (if contrast=True)
                - "class": Classification output (if multihead=True)

        Notes
        -----
        The attention gates in the decoder help focus on relevant features
        from the encoder skip connections, improving segmentation quality
        especially for small or complex structures.
        """
        # Super-resolution
        if self.pre_upsampling:
            x = self.pre_upsampling(x)

        # Down
        blocks = []
        for i, layers in enumerate(zip(self.down_path, self.mpooling_layers)):
            down, pool = layers
            x = down(x)
            if i != len(self.down_path):
                blocks.append(x)
                x = pool(x)

        x = self.bottleneck(x)

        # Up
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        feats = x
        # Super-resolution
        if self.post_upsampling:
            feats = self.post_upsampling(feats)

        # Regular output
        out = self.last_block(feats)
        out_dict = {
            "pred": out,
        }

        # Contrastive learning head
        if self.contrast:
            out_dict["embed"] = self.proj_head(feats)

        # Multi-head output
        #   Instance segmentation: instances + classification
        #   Detection: points + classification
        if self.multihead and self.last_class_head:
            out_dict["class"] = self.last_class_head(feats)

        if len(out_dict.keys()) == 1:
            return out_dict["pred"]
        else:
            return out_dict

    def _init_weights(self, m):
        """
        Initialize model weights using Xavier uniform initialization.

        Applies Xavier uniform initialization to convolutional and linear layers,
        and sets appropriate initial values for normalization layers.

        Parameters
        ----------
        m : nn.Module
            PyTorch module to initialize.

        Notes
        -----
        Initialization strategy:
        - Conv2d/Conv3d: Xavier uniform for weights, zero for biases
        - Linear: Xavier uniform for weights, zero for biases  
        - LayerNorm: Zero for biases, one for weights

        Xavier initialization helps maintain consistent variance of activations
        and gradients throughout the network depth.
        """
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
