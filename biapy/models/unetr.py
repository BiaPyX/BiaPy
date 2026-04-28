"""
This module implements the UNETR (U-Net TRansformers) architecture, a hybrid deep learning model that combines the strengths of Vision Transformers (ViT) with the U-Net's skip-connection mechanism.

UNETR replaces the traditional convolutional encoder of a U-Net with a ViT,
allowing it to capture long-range dependencies effectively. The ViT's latent
representations are then integrated into a convolutional decoder via skip
connections, adapting their spatial dimensionality to match the decoder's
levels. This design is particularly well-suited for 3D medical image
segmentation.

Classes:

- ``UNETR``: The main UNETR model, integrating a ViT encoder with a U-Net-like decoder.

This module leverages components from `biapy.models.blocks` such as `DoubleConvBlock`,
`ConvBlock`, `ProjectionHead`, and normalization helpers (`get_norm_2d`, `get_norm_3d`),
as well as `PatchEmbed` from `biapy.models.tr_layers`.

Reference:
`UNETR: Transformers for 3D Medical Image Segmentation
<https://openaccess.thecvf.com/content/WACV2022/html/Hatamizadeh_UNETR_Transformers_for_3D_Medical_Image_Segmentation_WACV_2022_paper.html>`_.
"""

import math
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
from typing import Dict, List

from biapy.models.blocks import (
    DoubleConvBlock, 
    ConvBlock, 
    get_norm_2d, 
    get_norm_3d, 
    prepare_activation_layers, 
    init_weights
)
from biapy.models.tr_layers import PatchEmbed
from biapy.models.heads import ProjectionHead


class UNETR(nn.Module):
    """
    UNETR (U-Net TRansformers) architecture.

    This model combines a Vision Transformer (ViT) as an encoder with a
    U-Net-like convolutional decoder. The ViT processes input images as
    sequences of patches, capturing global context, while the decoder
    reconstructs the output by upsampling and integrating skip connections
    from the ViT's intermediate layers.

    Reference: `UNETR: Transformers for 3D Medical Image Segmentation
    <https://openaccess.thecvf.com/content/WACV2022/html/Hatamizadeh_UNETR_Transformers_for_3D_Medical_Image_Segmentation_WACV_2022_paper.html>`_.
    """

    def __init__(
        self,
        input_shape,
        patch_size,
        embed_dim,
        depth,
        num_heads,
        mlp_ratio=4.0,
        num_filters=16,
        norm_layer=nn.LayerNorm,
        output_channels=[1],
        output_channel_info=["F"],
        explicit_activations: bool = False,
        head_activations: List[str] = ["ce_sigmoid"],
        decoder_activation="relu",
        ViT_hidd_mult=3,
        normalization="bn",
        dropout=0.0,
        k_size=3,
        contrast: bool = False,
        contrast_proj_dim: int = 256,
        return_one_tensor: bool = False,
    ):
        """
        Initialize the UNETR model.

        Sets up the Vision Transformer (ViT) encoder, including patch embedding,
        positional embeddings, and transformer blocks. It then constructs the
        U-Net-like convolutional decoder, which includes a bottleneck layer,
        upsampling layers, and convolutional blocks that integrate skip
        connections from the ViT encoder. Optional contrastive learning and
        multi-head outputs are also configured.

        
        Parameters
        ----------
        input_shape : Tuple[int, ...]
            Dimensions of the input image. E.g., `(y, x, channels)` for 2D or
            `(z, y, x, channels)` for 3D. The last element `input_shape[-1]`
            should be the number of input channels.

        patch_size : int
            Size of the square/cubic patches that are extracted from the input image.
            For example, to use `16x16` patches, set `patch_size = 16`.

        embed_dim : int
            Dimension of the embedding space for the Vision Transformer. This is
            the dimensionality of the patch tokens.

        depth : int
            Number of transformer encoder layers (blocks) in the ViT backbone.

        num_heads : int
            Number of attention heads in the multi-head attention layer of the ViT.

        mlp_ratio : float, optional
            Ratio to multiply `embed_dim` to obtain the hidden dimension of the
            MLP block within each Transformer block. Defaults to 4.0.

        num_filters : int, optional
            Number of filters in the first layer of the UNETR's convolutional decoder.
            In subsequent decoder layers, the number of filters is typically doubled
            or halved depending on the stage. Defaults to 16.

        norm_layer : Callable, optional
            Normalization layer constructor to use in the ViT backbone (e.g., `nn.LayerNorm`).
            Defaults to `nn.LayerNorm`.

        output_channels : list of int, optional
            Output channels of the network. If one value is provided, the model will have a single output head. 
            If two values are provided, the model will have two output heads (e.g. for multi-task learning with 
            instance segmentation and classification).

        output_channel_info : list of str, optional
            Output channels of the network. If one value is provided, the model will have a single output head. 
            If two values are provided, the model will have two output heads (e.g. for multi-task learning with 
            instance segmentation and classification).

        output_channel_info : list of str, optional
            Information about the type of output channels. Possible values are:
            - "X": where X is a letter, e.g. "F" for foreground, "D" for distance, "R" for rays, "C" for cpntours, etc.
            - "class": classification (e.g. for multi-task learning)

        explicit_activations : bool, optional
            If True, uses explicit activation functions in the last layers.
        
        head_activations : List[str], optional
            Activation functions to apply to each output head if `explicit_activations` is True.

        decoder_activation : str, optional
            Activation function for the convolutional decoder blocks (e.g., "relu", "elu").
            Defaults to "relu".

        ViT_hidd_mult : int, optional
            Multiplier to select which intermediate transformer encoder layers' outputs
            are used as skip connections for the decoder. For example, if `depth` is 12
            and `ViT_hidd_mult = 3`, skip connections will be taken from layers 3, 6, and 9.
            Defaults to 3.

        normalization : str, optional
            Normalization layer type for the convolutional decoder (one of `'bn'`,
            `'sync_bn'`, `'in'`, `'gn'`, or `'none'`). Defaults to "bn".

        dropout : float or List[float], optional
            Dropout rate for the decoder. Can be a single float applied uniformly
            or a list of dropout rates for each decoder layer. Defaults to 0.0.

        k_size : int, optional
            Kernel size for the convolutional layers in the decoder. Defaults to 3.

        contrast : bool, optional
            Whether to add a contrastive learning projection head to the model.
            If True, an additional output `embed` will be available in the forward pass.
            Defaults to `False`.

        contrast_proj_dim : int, optional
            Dimension of the projection head for contrastive learning, if `contrast` is True.
            Defaults to `256`.

        return_one_tensor : bool, optional
            If True, concatenates all outputs into a single tensor along the channel dimension
            in the forward pass. Defaults to `False`.

        Returns
        -------
        model : nn.Module
            The constructed UNETR model.
        """
        super().__init__()

        if len(output_channels) == 0:
            raise ValueError("'output_channels' needs to has at least one value")
        if contrast and len(output_channels) > 2:
            raise ValueError("If 'contrast' is True, 'output_channels' can only have two values at max: one for the main output and one for the class.")
        print("Selected output channels:")        
        for i, info in enumerate(output_channel_info):
            print(f"  - {i} channel for {info} output")

        self.input_shape = input_shape
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.ViT_hidd_mult = ViT_hidd_mult
        self.ndim = 3 if len(input_shape) == 4 else 2
        self.output_channels = output_channels
        self.output_channel_info = output_channel_info
        self.return_class = True if "class" in output_channel_info else False
        self.k_size = k_size
        self.contrast = contrast
        self.explicit_activations = explicit_activations
        self.return_one_tensor = return_one_tensor
        if self.explicit_activations:
            assert len(head_activations) == sum(output_channels), "If 'explicit_activations' is True, 'head_activations' needs to "
            "have the same number of values as 'output_channels'"
            self.head_activations, self.class_head_activations = prepare_activation_layers(head_activations, output_channel_info, output_channels)
            if self.return_class and self.class_head_activations is None:
                raise ValueError("If 'return_class' is True, 'head_activations' must be provided.")
        if self.ndim == 3:
            conv = nn.Conv3d
            convtranspose = nn.ConvTranspose3d
            self.reshape_shape = (
                self.input_shape[0] // self.patch_size,
                self.input_shape[1] // self.patch_size,
                self.input_shape[2] // self.patch_size,
                self.embed_dim,
            )
            self.permutation = (0, 4, 1, 2, 3)
            norm_func = get_norm_3d
            dropout_layer = nn.Dropout3d
        else:
            conv = nn.Conv2d
            convtranspose = nn.ConvTranspose2d
            self.reshape_shape = (
                self.input_shape[0] // self.patch_size,
                self.input_shape[1] // self.patch_size,
                self.embed_dim,
            )
            self.permutation = (0, 3, 1, 2)
            norm_func = get_norm_2d
            dropout_layer = nn.Dropout2d

        # ViT part
        self.patch_embed = PatchEmbed(
            img_size=input_shape[0],
            patch_size=patch_size,
            in_chans=input_shape[-1],
            ndim=self.ndim,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        # UNETR Part (bottom_up, from the bottle-neck, to the output)
        self.total_upscale_factor = int(math.log2(patch_size))
        # make a list of dropout values if needed
        if type(dropout) is float or type(dropout) is int:
            dropout = [
                dropout,
            ] * self.total_upscale_factor

        self.bottleneck = convtranspose(
            embed_dim,
            num_filters * (2 ** (self.total_upscale_factor - 1)),
            kernel_size=2,
            stride=2,
            bias=False,
        )

        self.mid_blue_block = nn.ModuleList()
        self.two_yellow_layers = nn.ModuleList()
        self.up_green_layers = nn.ModuleList()
        for layer in reversed(range(1, self.total_upscale_factor)):
            block = []
            in_size = embed_dim
            for _ in range(self.total_upscale_factor - layer):
                block.append(
                    convtranspose(
                        in_size,
                        num_filters * (2**layer),
                        kernel_size=2,
                        stride=2,
                        bias=False,
                    )
                )
                block.append(
                    ConvBlock(
                        conv,
                        in_size=num_filters * (2**layer),
                        out_size=num_filters * (2**layer),
                        k_size=k_size,
                        act=decoder_activation,
                        norm=normalization,
                        dropout=dropout[layer],
                    )
                )
                in_size = num_filters * (2**layer)
            self.mid_blue_block.append(nn.Sequential(*block))
            self.two_yellow_layers.append(
                DoubleConvBlock(
                    conv,
                    in_size * 2,
                    in_size,
                    k_size=k_size,
                    act=decoder_activation,
                    norm=normalization,
                    dropout=dropout[layer],
                )
            )
            self.up_green_layers.append(
                convtranspose(
                    in_size,
                    num_filters * (2 ** (layer - 1)),
                    kernel_size=2,
                    stride=2,
                    bias=False,
                )
            )

        # Last two yellow block for the first skip connection
        self.two_yellow_layers.append(
            DoubleConvBlock(
                conv,
                input_shape[-1],
                num_filters,
                k_size=k_size,
                act=decoder_activation,
                norm=normalization,
                dropout=dropout[0],
            )
        )

        # Last convolutions
        self.two_yellow_layers.append(
            DoubleConvBlock(
                conv,
                num_filters * 2,
                num_filters,
                k_size=k_size,
                act=decoder_activation,
                norm=normalization,
                dropout=dropout[0],
            )
        )

        if self.contrast:
            # extra added layers
            self.heads = nn.Sequential(
                conv(num_filters, num_filters, kernel_size=3, stride=1, padding=1),
                norm_func(normalization, num_filters),
                dropout_layer(0.10),
                conv(num_filters, output_channels[0], kernel_size=1, stride=1, padding=0, bias=False),
            )

            self.proj_head = ProjectionHead(ndim=self.ndim, in_channels=num_filters, proj_dim=contrast_proj_dim)
        else:
            self.heads = nn.Sequential()
            for i, out_ch in enumerate(output_channels):
                self.heads.append(conv(num_filters, out_ch, kernel_size=1, padding="same"))

        init_weights(self)


    def proj_feat(self, x):
        """
        Reshape and permute the flattened ViT feature tensor back into a spatial feature map format suitable for convolutional layers.

        Parameters
        ----------
        x : torch.Tensor
            Flattened feature tensor from the ViT encoder, typically
            ` (batch_size, num_patches, embed_dim)`.

        Returns
        -------
        torch.Tensor
            Reshaped and permuted feature tensor,
            e.g., `(batch_size, embed_dim, D, H, W)` for 3D or
            `(batch_size, embed_dim, H, W)` for 2D.
        """
        x = x.view((x.size(0),) + self.reshape_shape)
        x = x.permute(self.permutation).contiguous()
        return x

    def forward(self, input) -> Dict | torch.Tensor:
        """
        Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width) for 2D or (batch_size, channels, depth, height, width) for 3D.

        Returns
        -------
        Dict or torch.Tensor
            Model output. Returns a dictionary if multi-head or contrastive outputs are enabled,
            otherwise returns the main prediction tensor.
        """
        B = input.shape[0] # batch size
        # ViT Encoder
        x = self.patch_embed(input)

        # Add class token and positional embeddings
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        # Collect skip connections from ViT blocks
        skip_connection_index = [self.ViT_hidd_mult * layer for layer in range(1, self.total_upscale_factor)]
        skip_connections = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if (i + 1) in skip_connection_index:
                skip_connections.insert(0, x[:, 1:, :])

        # CNN Decoder
        # Bottleneck: Reshape ViT output (excluding class token) and apply transposed conv
        x = self.bottleneck(self.proj_feat(x[:, 1:, :]))

        # Decoder's upsampling path
        for i, layers in enumerate(zip(self.mid_blue_block, self.two_yellow_layers, self.up_green_layers)):
            blue, yellow, green = layers

            # Process ViT skip connection (blue block)
            z = self.proj_feat(skip_connections[i])
            z = blue(z)

            # Concatenate current decoder feature with processed skip connection
            x = torch.cat([x, z], dim=1)

            # Apply DoubleConvBlock (yellow block)
            x = yellow(x)

            # Apply transposed conv for upsampling (green block)
            x = green(x)

        # First skip connection (from original input image)
        # This connects the raw input to the first decoder stage
        first_skip = self.two_yellow_layers[-2](input)
        x = torch.cat([first_skip, x], dim=1)

        # Final UNETR output block before prediction heads
        x = self.two_yellow_layers[-1](x)

        feats = x

        out_dict = {}

        # Pass the features through the output heads
        class_outs, outs = [], []
        for i, head in enumerate(self.heads):
            if "class" not in self.output_channel_info[i]:
                outs.append(head(feats))
            else:
                class_outs.append(head(feats))  
        outs = torch.cat(outs, dim=1)

        # Apply activations to the output heads if explicit_activations is True
        if self.explicit_activations:
            # If there is only one activation, apply it to the whole tensor
            if len(self.head_activations) == 1:
                outs = self.head_activations[0](outs)
            else:
                for i, act in enumerate(self.head_activations):
                    outs[:, i:i+1] = act(outs[:, i:i+1])

            if self.return_class and self.class_head_activations is not None:
                for i, act in enumerate(self.class_head_activations):
                    class_outs[i] = act(class_outs[i])

        out_dict = {
            "pred": outs,
        }
        if self.return_class:
            out_dict["class"] = torch.cat(class_outs, dim=1)

        # Contrastive learning head
        if self.contrast:
            out_dict["embed"] = self.proj_head(feats)

        if self.return_one_tensor:
            # Concatenate all outputs into a single tensor along the channel dimension
            return torch.cat((out_dict["pred"], torch.argmax(out_dict["class"], dim=1).unsqueeze(1)), dim=1)
        else:
            if len(out_dict.keys()) == 1:
                return out_dict["pred"]
            else:
                return out_dict
