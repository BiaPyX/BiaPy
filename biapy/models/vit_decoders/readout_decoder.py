"""
Cellpose 4's per-token readout: each last-block token is projected to ``features * stride^2`` values, pixel-shuffled
into its ``stride x stride`` block, optionally followed by full-resolution 3x3 refinement convolutions.
"""

import torch
import torch.nn as nn

from biapy.models.blocks import ConvBlock


class ReadoutDecoderMixin:
    """Readout decoder (needs the encoder's ``feature_dim`` and ``token_stride``)."""

    def _build_readout_decoder(
        self,
        in_chans: int,
        readout_features: int,
        refine_layers: int,
        input_skip: bool,
        normalization: str,
        decoder_activation: str,
        k_size: int,
        dropout: float,
    ) -> int:
        """
        Build the readout and return its output channels.

        Parameters
        ----------
        in_chans : int
            Input image channels (concatenated before the refinement with ``input_skip``).

        readout_features : int
            Channels per pixel from the readout.

        refine_layers : int
            3x3 refinement blocks (0 = Cellpose's linear readout).

        input_skip : bool
            Concatenate the input image before the refinement.

        normalization, decoder_activation, k_size, dropout : str, str, int, float
            Refinement block settings.
        """
        self.readout_features = readout_features
        self.refine_layers = refine_layers
        self.input_skip = input_skip and refine_layers > 0

        s = self.token_stride
        self.readout = nn.Linear(self.feature_dim, readout_features * s * s)
        self.pixel_shuffle = nn.PixelShuffle(s)

        refine = []
        in_size = readout_features + (in_chans if self.input_skip else 0)
        for _ in range(refine_layers):
            refine.append(
                ConvBlock(
                    nn.Conv2d,
                    in_size=in_size,
                    out_size=readout_features,
                    k_size=k_size,
                    act=decoder_activation,
                    norm=normalization,
                    dropout=dropout,
                )
            )
            in_size = readout_features
        self.refine = nn.Sequential(*refine)
        return readout_features

    def _readout_decode(self, feats: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        """``(B, feature_dim, h, w)`` features to full resolution."""
        x = self.pixel_shuffle(self.readout(feats.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        if self.input_skip:
            x = torch.cat([x, input], dim=1)
        return self.refine(x)
