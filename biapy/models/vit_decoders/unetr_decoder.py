"""UNETR's convolutional decoder (Hatamizadeh et al., WACV 2022), with skip connections from intermediate blocks."""

import math
from typing import Dict, List

import torch
import torch.nn as nn

from biapy.models.blocks import ConvBlock, DoubleConvBlock, get_norm_2d, get_norm_3d, prepare_activation_layers
from biapy.models.heads import ProjectionHead


class UNETRDecoderMixin:
    """UNETR's decoder and output heads."""

    def _build_unetr_decoder(
        self,
        input_shape,
        embed_dim: int,
        patch_size: int,
        num_filters: int,
        output_channels: List[int],
        output_channel_info: List[str],
        explicit_activations: bool,
        head_activations: List[str],
        decoder_activation: str,
        normalization: str,
        dropout,
        k_size: int,
        contrast: bool,
        contrast_proj_dim: int,
        return_one_tensor: bool,
    ):
        """
        Build the decoder and the output heads.

        Parameters
        ----------
        input_shape : Tuple[int, ...]
            ``(y, x, channels)`` or ``(z, y, x, channels)``.

        embed_dim, patch_size : int
            Token channels and token size (a power of two).

        num_filters : int
            Filters of the last level (doubled on each coarser one).

        decoder_activation, normalization, dropout, k_size :
            Decoder block settings (``dropout``: one value or one per level).

        contrast, contrast_proj_dim : bool, int
            Add a contrastive projection head (``embed`` output) and its dimension.

        output_channels, output_channel_info, explicit_activations, head_activations, return_one_tensor :
            Output heads configuration.
        """
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
                input_shape[0] // patch_size,
                input_shape[1] // patch_size,
                input_shape[2] // patch_size,
                embed_dim,
            )
            self.permutation = (0, 4, 1, 2, 3)
            norm_func = get_norm_3d
            dropout_layer = nn.Dropout3d
        else:
            conv = nn.Conv2d
            convtranspose = nn.ConvTranspose2d
            self.reshape_shape = (
                input_shape[0] // patch_size,
                input_shape[1] // patch_size,
                embed_dim,
            )
            self.permutation = (0, 3, 1, 2)
            norm_func = get_norm_2d
            dropout_layer = nn.Dropout2d

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

    def _unetr_decode(self, x: torch.Tensor, skip_connections: List[torch.Tensor], input: torch.Tensor) -> Dict | torch.Tensor:
        """
        Patch tokens of the last block (``x``) and of the skip blocks (deepest first) plus the input images, to the
        model output.
        """
        # Bottleneck: Reshape ViT output and apply transposed conv
        x = self.bottleneck(self.proj_feat(x))

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

        if len(out_dict.keys()) == 1:
            return out_dict["pred"]
        else:
            if self.return_one_tensor:
                if "class" in out_dict:
                    return torch.cat((out_dict["pred"], torch.argmax(out_dict["class"], dim=1).unsqueeze(1)), dim=1)
                else:
                    return out_dict["pred"]
            return out_dict
