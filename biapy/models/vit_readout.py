"""
ViT-Readout: a plain Vision Transformer whose last-layer tokens are read out directly into pixels.

The design follows the models of Cellpose 4 (``CPSAM`` / ``CPDINO`` in https://github.com/MouseLand/cellpose,
Pachitariu et al., "Cellpose-SAM: superhuman generalization for cellular segmentation", 2025). There is no
convolutional encoder-decoder: all the spatial reasoning is left to a (pretrained, fully fine-tuned) ViT, and

1. the tokens are made smaller than the patch embedding by moving it with a smaller stride (e.g. 16x16
   pretrained patches every 8 pixels, doubling the token grid while keeping the pretrained weights);
2. only the output of the last block (after the final norm) is used;
3. each token is linearly projected to ``features * stride^2`` values that are rearranged (pixel shuffle) into its
   own ``stride x stride`` block of pixels.

On top of Cellpose's readout, a small convolutional refinement stage can be added at full resolution, optionally
fed with the input image too. It smooths the seams between the blocks predicted by neighbouring tokens and brings
back pixel-level detail from the input. With ``refine_layers=0`` the model is Cellpose's linear readout.
"""

from typing import Dict, List

import torch
import torch.nn as nn

from biapy.models.vit_backbones.encoder import ViTEncoderMixin
from biapy.models.vit_decoders.output_heads import OutputHeadsMixin
from biapy.models.vit_decoders.readout_decoder import ReadoutDecoderMixin


class ViTReadout(ViTEncoderMixin, ReadoutDecoderMixin, OutputHeadsMixin, nn.Module):
    """ViT encoder + per-token pixel-shuffle readout + optional convolutional refinement (Cellpose 4 style)."""

    def __init__(
        self,
        input_shape,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        vit_model: str = "custom",
        token_stride: int = -1,
        drop_path_rate: float = 0.0,
        grad_checkpointing: bool = False,
        readout_features: int = 32,
        refine_layers: int = 2,
        input_skip: bool = True,
        normalization: str = "bn",
        decoder_activation: str = "relu",
        k_size: int = 3,
        dropout: float = 0.0,
        output_channels: List[int] = [1],
        output_channel_info: List[str] = ["F"],
        explicit_activations: bool = False,
        head_activations: List[str] = ["ce_sigmoid"],
        return_one_tensor: bool = False,
    ):
        """
        Initialize the ViT-Readout model.

        Parameters
        ----------
        input_shape : Tuple[int, ...]
            Input shape, ``(y, x, channels)``. 2D square inputs only.

        patch_size, embed_dim, depth, num_heads, mlp_ratio : int, int, int, int, float, optional
            ViT hyperparameters. Ignored unless ``vit_model`` is "custom".

        vit_model : str, optional
            "custom" or one of ``VIT_BACKBONES`` (``biapy.models.vit_backbones.common``). "cellpose_sam_vit" with
            ``readout_features=3`` and ``refine_layers=0`` is the whole Cellpose-SAM network.

        token_stride : int, optional
            Stride of the patch embedding, i.e. the side of the pixel block each token predicts. ``-1`` uses the
            token size. Cellpose uses 8 with 16x16 pretrained patches.

        drop_path_rate : float, optional
            Stochastic depth of the last block, increasing linearly from 0 (Cellpose uses 0.4).

        grad_checkpointing : bool, optional
            Recompute the transformer blocks in the backward pass to save memory.

        readout_features, refine_layers, input_skip, normalization, decoder_activation, k_size, dropout : optional
            Decoder settings (see ``ReadoutDecoderMixin._build_readout_decoder``).

        output_channels, output_channel_info, explicit_activations, head_activations, return_one_tensor : optional
            Output heads configuration, as in the rest of BiaPy's models.
        """
        super().__init__()
        if len(input_shape) != 3:
            raise ValueError("ViTReadout only supports 2D data, i.e. 'input_shape' as (y, x, channels)")
        if input_shape[0] != input_shape[1]:
            raise ValueError(f"ViTReadout needs square inputs. Provided: {input_shape}")
        self._build_vit_encoder(
            "vit_readout",
            input_shape,
            patch_size,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            vit_model=vit_model,
            token_stride=token_stride,
            drop_path_rate=drop_path_rate,
            grad_checkpointing=grad_checkpointing,
        )
        out_features = self._build_readout_decoder(
            input_shape[-1],
            readout_features,
            refine_layers,
            input_skip,
            normalization,
            decoder_activation,
            k_size,
            dropout,
        )
        self._build_heads(
            out_features,
            output_channels,
            output_channel_info,
            explicit_activations,
            head_activations,
            return_one_tensor,
        )
        self._init_vit_encoder_weights()

    def forward(self, input: torch.Tensor) -> Dict | torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        input : torch.Tensor
            Input images, ``(B, C, H, W)``.

        Returns
        -------
        Dict or torch.Tensor
            Model output, as in the rest of BiaPy's models.
        """
        return self._format_output(self._readout_decode(self.forward_features(input), input))
