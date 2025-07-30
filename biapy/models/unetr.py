"""
This module implements the UNETR (U-Net TRansformers) architecture, a hybrid deep learning model that combines the strengths of Vision Transformers (ViT) with the U-Net's skip-connection mechanism.

UNETR replaces the traditional convolutional encoder of a U-Net with a ViT,
allowing it to capture long-range dependencies effectively. The ViT's latent
representations are then integrated into a convolutional decoder via skip
connections, adapting their spatial dimensionality to match the decoder's
levels. This design is particularly well-suited for 3D medical image
segmentation.

Classes:
--------
- UNETR: The main UNETR model, integrating a ViT encoder with a U-Net-like decoder.

This module leverages components from `biapy.models.blocks` such as `DoubleConvBlock`,
`ConvBlock`, `ProjectionHead`, and normalization helpers (`get_norm_2d`, `get_norm_3d`),
as well as `PatchEmbed` from `biapy.models.tr_layers`.

Reference:
----------
`UNETR: Transformers for 3D Medical Image Segmentation
<https://openaccess.thecvf.com/content/WACV2022/html/Hatamizadeh_UNETR_Transformers_for_3D_Medical_Image_Segmentation_WACV_2022_paper.html>`_.
"""

import math
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
from typing import Dict

from biapy.models.blocks import DoubleConvBlock, ConvBlock, ProjectionHead,  get_norm_2d,  get_norm_3d
from biapy.models.tr_layers import PatchEmbed


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

    output_channels : List[int], optional
        Output channels of the network. It must be a list of length 1 for a single
        output task (e.g., semantic segmentation) or length 2 for multi-head tasks
        (e.g., instances + classification in instance segmentation, or points +
        classification in detection). Defaults to `[1]`.

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

    Returns
    -------
    model : nn.Module
        The constructed UNETR model.
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
        decoder_activation="relu",
        ViT_hidd_mult=3,
        normalization="bn",
        dropout=0.0,
        k_size=3,
        contrast: bool = False,
        contrast_proj_dim: int = 256,
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
            `(z, y, x, channels)` for 3D. The last element is the number of input channels.
        patch_size : int
            Size of the patches.
        embed_dim : int
            Dimension of the embedding space for the ViT.
        depth : int
            Number of transformer encoder layers.
        num_heads : int
            Number of attention heads.
        mlp_ratio : float, optional
            Ratio for MLP hidden dimension. Defaults to 4.0.
        num_filters : int, optional
            Number of filters in the first decoder layer. Defaults to 16.
        norm_layer : Callable, optional
            Normalization layer constructor for ViT. Defaults to `nn.LayerNorm`.
        output_channels : List[int], optional
            Output channels for the network's prediction head(s). Can be length 1 or 2.
            Defaults to `[1]`.
        decoder_activation : str, optional
            Activation function for decoder. Defaults to "relu".
        ViT_hidd_mult : int, optional
            Multiplier for selecting ViT hidden layer outputs as skip connections.
            Defaults to 3.
        normalization : str, optional
            Normalization layer type for decoder. Defaults to "bn".
        dropout : float | List[float], optional
            Dropout rate(s) for the decoder. Defaults to 0.0.
        k_size : int, optional
            Kernel size for decoder convolutions. Defaults to 3.
        contrast : bool, optional
            Whether to add a contrastive learning head. Defaults to `False`.
        contrast_proj_dim : int, optional
            Dimension of the contrastive projection head. Defaults to `256`.

        Raises
        ------
        ValueError
            If 'output_channels' is empty or has more than two values.
        """
        super().__init__()

        if len(output_channels) == 0:
            raise ValueError("'output_channels' needs to has at least one value")
        if len(output_channels) != 1 and len(output_channels) != 2:
            raise ValueError(f"'output_channels' must be a list of one or two values at max, not {output_channels}")

        self.input_shape = input_shape
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.ViT_hidd_mult = ViT_hidd_mult
        self.ndim = 3 if len(input_shape) == 4 else 2
        self.output_channels = output_channels
        self.multihead = len(output_channels) == 2
        self.k_size = k_size
        self.contrast = contrast
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
        self.norm = norm_layer(embed_dim)

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
            self.last_block = nn.Sequential(
                conv(num_filters, num_filters, kernel_size=3, stride=1, padding=1),
                norm_func(normalization, num_filters),
                dropout_layer(0.10),
                conv(num_filters, output_channels[0], kernel_size=1, stride=1, padding=0, bias=False),
            )

            self.proj_head = ProjectionHead(ndim=self.ndim, in_channels=num_filters, proj_dim=contrast_proj_dim)
        else:
            self.last_block = conv(num_filters, output_channels[0], kernel_size=1, padding="same")

        # Multi-head:
        #   Instance segmentation: instances + classification
        #   Detection: points + classification
        self.last_class_head = None
        if self.multihead:
            self.last_class_head = conv(num_filters, output_channels[1], kernel_size=1, padding="same")

        self.apply(self._init_weights)

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
        Perform the complete forward pass of the UNETR model.

        The input `input` first goes through the ViT encoder, which produces
        a latent representation and extracts intermediate features for skip
        connections. These features are then fed into the convolutional decoder,
        which upsamples and reconstructs the output, integrating the ViT's
        hierarchical representations via skip connections. Finally, the
        output passes through the prediction head(s).

        Parameters
        ----------
        input : torch.Tensor
            The input image tensor.
            Expected shape for 2D: `(batch_size, input_channels, height, width)`.
            Expected shape for 3D: `(batch_size, input_channels, depth, height, width)`.

        Returns
        -------
        Dict | torch.Tensor
            If `contrast` is True or `multihead` is True, returns a dictionary
            containing different output tensors (e.g., "pred", "embed", "class").
            Otherwise, returns only the primary prediction tensor.
            The primary prediction tensor ("pred") will have `output_channels[0]`
            channels and spatial dimensions matching the input `input`.
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

        # Primary output (e.g., segmentation mask)
        out = self.last_block(feats)
        out_dict = {
            "pred": out,
        }

        # Optional: Contrastive learning projection head
        if self.contrast:
            out_dict["embed"] = self.proj_head(feats)

        # Multi-head output
        #   Instance segmentation: instances + classification
        #   Detection: points + classification
        if self.multihead and self.last_class_head:
            out_dict["class"] = self.last_class_head(feats)

        # Return format based on whether multiple outputs are generated
        if len(out_dict.keys()) == 1:
            return out_dict["pred"]
        else:
            return out_dict

    def _init_weights(self, m):
        """
        Initialize the weights of convolutional, linear, and LayerNorm layers.

        Applies Xavier uniform initialization to convolutional and linear layer weights
        (with bias set to 0 if present). For LayerNorm, weights are set to 1.0 and
        biases to 0. This method is typically called using `model.apply()`.

        Parameters
        ----------
        m : nn.Module
            The module whose weights are to be initialized.
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
