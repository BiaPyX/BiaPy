import math
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
from typing import List

from biapy.models.blocks import DoubleConvBlock, ConvBlock
from biapy.models.tr_layers import PatchEmbed


class UNETR(nn.Module):
    """
    UNETR architecture. It combines a ViT with U-Net, replaces the convolutional encoder
    with the ViT and adapt each skip connection signal to their layer's spatial dimensionality.

    Reference: `UNETR: Transformers for 3D Medical Image Segmentation
    <https://openaccess.thecvf.com/content/WACV2022/html/Hatamizadeh_UNETR_Transformers_for_3D_Medical_Image_Segmentation_WACV_2022_paper.html>`_.

    Parameters
    ----------
    input_shape : 3D/4D tuple
        Dimensions of the input image. E.g. ``(y, x, channels)`` or ``(z, y, x, channels)``.

    patch_size : int
        Size of the patches that are extracted from the input image. As an example, to use ``16x16``
        patches, set ``patch_size = 16``.

    embed_dim : int
        Dimension of the embedding space.

    depth : int
        Number of transformer encoder layers.

    num_heads : int
        Number of heads in the multi-head attention layer.

    mlp_ratio : float, optional
        Ratio to multiply ``embed_dim`` to obtain the dense layers of the final classifier.

    num_filters: int, optional
        Number of filters in the first UNETR's layer of the decoder. In each layer the previous number of filters is
        doubled.

    norm_layer : Torch layer, optional
        Normalization layer to use in ViT backbone.

    output_channels : list of int, optional
        Output channels of the network. It must be a list of lenght ``1`` or ``2``. When two
        numbers are provided two task to be done is expected (multi-head). Possible scenarios are:
            * instances + classification on instance segmentation
            * points + classification in detection.

    decoder_activation : str, optional
        Activation function for the decoder.

    ViT_hidd_mult : int, optional
        Multiple of the transformer encoder layers from of which the skip connection signal is going to be extracted.
        E.g. if we have ``12`` transformer encoder layers, and we set ``ViT_hidd_mult = 3``, we are going to take
        ``[1*ViT_hidd_mult, 2*ViT_hidd_mult, 3*ViT_hidd_mult]`` -> ``[Z3, Z6, Z9]`` encoder's signals.

    normalization : str, optional
        Normalization layer (one of ``'bn'``, ``'sync_bn'`` ``'in'``, ``'gn'`` or ``'none'``).

    dropout : bool, optional
        Dropout rate for the decoder (can be a list of dropout rates for each layer).

    k_size : int, optional
        Decoder convolutions' kernel size.

    Returns
    -------
    model : Torch model
        UNETR model.
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
    ):
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
        else:
            conv = nn.Conv2d
            convtranspose = nn.ConvTranspose2d
            self.reshape_shape = (
                self.input_shape[0] // self.patch_size,
                self.input_shape[1] // self.patch_size,
                self.embed_dim,
            )
            self.permutation = (0, 3, 1, 2)

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

        self.last_block = conv(num_filters, output_channels[0], kernel_size=1, padding="same")
        # Multi-head:
        #   Instance segmentation: instances + classification
        #   Detection: points + classification
        self.last_class_head = None
        if self.multihead:
            self.last_class_head = conv(num_filters, output_channels[1], kernel_size=1, padding="same")

        self.apply(self._init_weights)

    def proj_feat(self, x):
        x = x.view((x.size(0),) + self.reshape_shape)
        x = x.permute(self.permutation).contiguous()
        return x

    def forward(self, input) -> torch.Tensor | List[torch.Tensor]:
        # Vit part
        B = input.shape[0]
        x = self.patch_embed(input)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        skip_connection_index = [self.ViT_hidd_mult * layer for layer in range(1, self.total_upscale_factor)]
        skip_connections = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if (i + 1) in skip_connection_index:
                skip_connections.insert(0, x[:, 1:, :])

        # CNN Decoder
        x = self.bottleneck(self.proj_feat(x[:, 1:, :]))

        for i, layers in enumerate(zip(self.mid_blue_block, self.two_yellow_layers, self.up_green_layers)):
            blue, yellow, green = layers
            z = self.proj_feat(skip_connections[i])
            z = blue(z)
            x = torch.cat([x, z], dim=1)
            x = yellow(x)
            x = green(x)

        # first skip connection (out of transformer)
        first_skip = self.two_yellow_layers[-2](input)
        x = torch.cat([first_skip, x], dim=1)

        # UNETR output
        x = self.two_yellow_layers[-1](x)
        class_head_out = torch.empty(())
        if self.multihead and self.last_class_head:
            class_head_out = self.last_class_head(x)
        x = self.last_block(x)

        if self.multihead:
            return [x, class_head_out]
        else:
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
