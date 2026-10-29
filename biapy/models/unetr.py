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

Encoder: `biapy.models.vit_backbones.encoder`. Decoder: `biapy.models.vit_decoders.unetr_decoder`.

Reference:
`UNETR: Transformers for 3D Medical Image Segmentation
<https://openaccess.thecvf.com/content/WACV2022/html/Hatamizadeh_UNETR_Transformers_for_3D_Medical_Image_Segmentation_WACV_2022_paper.html>`_.
"""

import math
from typing import Dict, List

import torch
import torch.nn as nn

from biapy.models.blocks import init_weights
from biapy.models.vit_backbones.common import UNETR_PATCH_SIZE, VIT_BACKBONES
from biapy.models.vit_backbones.encoder import ViTEncoderMixin
from biapy.models.vit_decoders.unetr_decoder import UNETRDecoderMixin


class UNETR(ViTEncoderMixin, UNETRDecoderMixin, nn.Module):
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
        vit_model="custom",
        num_filters=16,
        norm_layer=nn.LayerNorm,
        output_channels=[1],
        output_channel_info=["F"],
        explicit_activations: bool = False,
        head_activations: List[str] = ["ce_sigmoid"],
        decoder_activation="relu",
        ViT_hidd_mult=-1,
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
            For example, to use `16x16` patches, set `patch_size = 16`. As the decoder
            upsamples the ViT features by a factor of two on each of its levels, it needs
            to be a power of two. Ignored unless `vit_model` is "custom".

        embed_dim : int
            Dimension of the embedding space for the Vision Transformer. This is
            the dimensionality of the patch tokens. Ignored unless `vit_model` is "custom".

        depth : int
            Number of transformer encoder layers (blocks) in the ViT backbone.
            Ignored unless `vit_model` is "custom".

        num_heads : int
            Number of attention heads in the multi-head attention layer of the ViT.
            Ignored unless `vit_model` is "custom".

        mlp_ratio : float, optional
            Ratio to multiply `embed_dim` to obtain the hidden dimension of the
            MLP block within each Transformer block. Defaults to 4.0. Ignored unless
            `vit_model` is "custom".

        vit_model : str, optional
            "custom" (built from `patch_size`, `embed_dim`, `depth`, `num_heads` and `mlp_ratio`) or one
            of `VIT_BACKBONES` (`biapy.models.vit_backbones.common`), which fix those values ("sam3_vit"
            uses 16x16 tokens, as the decoder needs a power of two).

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
            Set it to ``-1`` (default) to space them evenly along the encoder, which takes
            one every ``depth // log2(patch_size)`` blocks.

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

        # Fixed by predefined backbones
        if vit_model != "custom" and vit_model in VIT_BACKBONES:
            patch_size = UNETR_PATCH_SIZE.get(vit_model, VIT_BACKBONES[vit_model]["patch_size"])
            depth = VIT_BACKBONES[vit_model]["depth"]

        # The decoder recovers the input resolution upsampling the ViT features by a factor of two on each
        # of its levels, so there must be an exact number of levels, i.e. log2(patch_size) must be an integer
        if patch_size < 2 or 2 ** int(math.log2(patch_size)) != patch_size:
            raise ValueError(
                "UNETR's patch size needs to be a power of two greater than one, as its decoder upsamples "
                "the ViT features by a factor of two on each level. Provided: {}{}".format(
                    patch_size,
                    f" (set by the '{vit_model}' ViT model)" if vit_model != "custom" else "",
                )
            )

        # The decoder takes one skip connection from the transformer at every level but the last one, i.e.
        # from the blocks 'ViT_hidd_mult', '2 * ViT_hidd_mult' ... So the multiplier must be small enough for
        # all of them to exist, and spreading them evenly along the encoder means one every 'depth // levels'
        # blocks. That is what is used when no multiplier is provided, which with the classic 12-block ViT
        # and 16x16 tokens gives the 3 that used to be the default.
        decoder_levels = int(math.log2(patch_size))
        if ViT_hidd_mult <= 0:
            ViT_hidd_mult = max(1, depth // decoder_levels)
            print(
                f"Extracting UNETR's skip connections every {ViT_hidd_mult} transformer blocks, i.e. from the "
                f"blocks {[ViT_hidd_mult * i for i in range(1, decoder_levels)]} of the {depth} of the encoder"
            )
        elif ViT_hidd_mult * (decoder_levels - 1) > depth:
            raise ValueError(
                "'ViT_hidd_mult' is too large for this encoder: UNETR takes its skip connections from the "
                f"transformer blocks {[ViT_hidd_mult * i for i in range(1, decoder_levels)]}, but the encoder "
                f"only has {depth} blocks. Reduce it to {max(1, depth // decoder_levels)} or below, or set it to "
                "-1 ('MODEL.UNETR_VIT_HIDD_MULT') to let BiaPy space them evenly along the encoder."
            )

        self.patch_size = patch_size
        self.ViT_hidd_mult = ViT_hidd_mult
        self.ndim = 3 if len(input_shape) == 4 else 2

        # No final norm, fixed position embedding, 'ln_pre' (identity but for SAM 3)
        self._build_vit_encoder(
            "unetr",
            input_shape,
            patch_size,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            vit_model=vit_model,
            norm_layer=norm_layer,
            final_norm=False,
            pre_norm=True,
            trainable_pos_embed=False,
            preset_patch_size=UNETR_PATCH_SIZE.get(vit_model),
        )
        if self.reg_token is not None:
            nn.init.normal_(self.reg_token, std=0.02)

        self._build_unetr_decoder(
            input_shape,
            self.embed_dim,
            patch_size,
            num_filters,
            output_channels,
            output_channel_info,
            explicit_activations,
            head_activations,
            decoder_activation,
            normalization,
            dropout,
            k_size,
            contrast,
            contrast_proj_dim,
            return_one_tensor,
        )

        init_weights(self)

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
        # Skip connections: patch tokens of intermediate blocks, deepest first
        num_prefix = self.num_prefix_tokens
        skip_connection_index = [self.ViT_hidd_mult * layer for layer in range(1, self.total_upscale_factor)]
        layers = sorted(set(skip_connection_index + [self.depth]))
        outs = dict(zip(layers, self._run_blocks(self.embed_tokens(input), layers)))
        skip_connections = [outs[i][:, num_prefix:, :] for i in reversed(skip_connection_index)]

        return self._unetr_decode(outs[self.depth][:, num_prefix:, :], skip_connections, input)
