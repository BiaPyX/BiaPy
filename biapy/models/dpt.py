"""
DPT (Dense Prediction Transformer) for BiaPy.

Reference: `Vision Transformers for Dense Prediction <https://arxiv.org/abs/2103.13413>`_ (Ranftl, Bochkovskiy and
Koltun, ICCV 2021), and the implementation in https://github.com/isl-org/DPT. It is also the decoder DINOv2 uses for
its dense (depth) evaluation.

Unlike UNETR, which builds a separate stack of transposed convolutions per skip connection, DPT:

1. **Reads** four ViT layers (evenly spaced along the encoder by default), folding the class token into the patch
   tokens ("ignore", "add" or "project").
2. **Reassembles** each of them into an image-like feature map and resamples it to a different scale, forming a
   pyramid at 4x, 2x, 1x and 0.5x the token-grid resolution (1/4, 1/8, 1/16 and 1/32 of the input for 16x16 tokens).
3. **Fuses** the pyramid from coarse to fine with residual convolutional units (RefineNet-style fusion blocks),
   upsampling by two at each step.
4. A small **head** brings the fused features to the input resolution.
"""

from typing import Dict, List

import torch
import torch.nn as nn

from biapy.models.vit_backbones.encoder import ViTEncoderMixin
from biapy.models.vit_decoders.dpt_decoder import DPTDecoderMixin
from biapy.models.vit_decoders.output_heads import OutputHeadsMixin


class DPT(ViTEncoderMixin, DPTDecoderMixin, OutputHeadsMixin, nn.Module):
    """DPT: ViT encoder + reassemble/fusion convolutional decoder (Ranftl et al., 2021)."""

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
        layers: List[int] = [],
        features: int = 256,
        reassemble_channels: List[int] = [],
        readout: str = "project",
        normalization: str = "none",
        decoder_activation: str = "relu",
        output_channels: List[int] = [1],
        output_channel_info: List[str] = ["F"],
        explicit_activations: bool = False,
        head_activations: List[str] = ["ce_sigmoid"],
        return_one_tensor: bool = False,
    ):
        """
        Initialize the DPT model.

        Parameters
        ----------
        input_shape : Tuple[int, ...]
            Input shape, ``(y, x, channels)``. 2D square inputs only.

        patch_size, embed_dim, depth, num_heads, mlp_ratio : int, int, int, int, float, optional
            ViT hyperparameters. Ignored unless ``vit_model`` is "custom".

        vit_model : str, optional
            "custom" or one of ``VIT_BACKBONES`` (``biapy.models.vit_backbones.common``).

        token_stride : int, optional
            Stride of the patch embedding. ``-1`` uses the token size (non-overlapping tokens, as in DPT).

        drop_path_rate : float, optional
            Stochastic depth of the last block, increasing linearly from 0.

        grad_checkpointing : bool, optional
            Recompute the transformer blocks in the backward pass to save memory.

        layers, features, reassemble_channels, readout, normalization, decoder_activation : optional
            Decoder settings (see ``DPTDecoderMixin._build_dpt_decoder``).

        output_channels, output_channel_info, explicit_activations, head_activations, return_one_tensor : optional
            Output heads configuration, as in the rest of BiaPy's models.
        """
        super().__init__()
        if len(input_shape) != 3:
            raise ValueError("DPT only supports 2D data, i.e. 'input_shape' as (y, x, channels)")
        if input_shape[0] != input_shape[1]:
            raise ValueError(f"DPT needs square inputs. Provided: {input_shape}")
        self._build_vit_encoder(
            "dpt",
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
        out_features = self._build_dpt_decoder(
            layers, features, reassemble_channels, readout, normalization, decoder_activation
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
        tokens = self.forward_tokens(input, self.layers)
        return self._format_output(self._dpt_decode(tokens, input.shape[-2:]))
