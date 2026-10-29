"""DPT's reassemble/fusion decoder (Ranftl et al., ICCV 2021, https://arxiv.org/abs/2103.13413)."""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from biapy.models.blocks import get_activation, get_norm_2d


def dpt_activation(name: str) -> nn.Module:
    """BiaPy activation, never in-place (inputs are reused by the residual connections)."""
    act = get_activation(name)
    if hasattr(act, "inplace"):
        act.inplace = False
    return act


class ResidualConvUnit(nn.Module):
    """DPT's residual convolutional unit: ``x + conv(act(conv(act(x))))``."""

    def __init__(self, features: int, normalization: str, activation: str):
        super().__init__()
        bias = normalization == "none"
        self.act1 = dpt_activation(activation)
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, padding=1, bias=bias)
        self.norm1 = get_norm_2d(normalization, features)
        self.act2 = dpt_activation(activation)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1, bias=bias)
        self.norm2 = get_norm_2d(normalization, features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.norm1(self.conv1(self.act1(x)))
        out = self.norm2(self.conv2(self.act2(out)))
        return x + out


class FeatureFusionBlock(nn.Module):
    """
    DPT's fusion block: adds the (refined) skip features to the incoming ones, refines the sum and upsamples it.

    The coarsest block has no skip features (``has_skip=False``), so it does not build the unit that refines them:
    unused parameters would break DistributedDataParallel.
    """

    def __init__(self, features: int, normalization: str, activation: str, has_skip: bool = True):
        super().__init__()
        self.res_conv_unit1 = ResidualConvUnit(features, normalization, activation) if has_skip else None
        self.res_conv_unit2 = ResidualConvUnit(features, normalization, activation)
        self.out_conv = nn.Conv2d(features, features, kernel_size=1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None, size) -> torch.Tensor:
        if skip is not None:
            if self.res_conv_unit1 is None:
                raise ValueError("This fusion block was built without skip features ('has_skip=False')")
            x = x + self.res_conv_unit1(skip)
        x = self.res_conv_unit2(x)
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=True)
        return self.out_conv(x)


class DPTDecoderMixin:
    """DPT's decoder (needs the encoder's ``embed_dim`` and ``depth``)."""

    def _build_dpt_decoder(
        self,
        layers: List[int],
        features: int,
        reassemble_channels: List[int],
        readout: str,
        normalization: str,
        decoder_activation: str,
    ) -> int:
        """
        Build the decoder and return its output channels.

        Parameters
        ----------
        layers : List[int]
            Four increasing ViT blocks (1-indexed) to read, finest level first. Empty = evenly spaced.

        features : int
            Fusion channels (256 in DPT).

        reassemble_channels : List[int]
            Channels of the four reassembled maps. Empty = DPT's (``[256, 512, 1024, 1024]`` for ViT-L/H,
            ``[D/8, D/4, D/2, D]`` otherwise).

        readout : str
            Class token folding: "ignore", "add" or "project" (DPT's).

        normalization, decoder_activation : str
            Normalization and activation of the residual units.
        """
        D = self.embed_dim
        if len(layers) == 0:
            layers = [self.depth * (i + 1) // 4 for i in range(4)]
        if len(layers) != 4 or any(b >= a for a, b in zip(layers[1:], layers[:-1])) or not 1 <= layers[0] <= layers[-1] <= self.depth:
            raise ValueError(
                f"DPT needs 4 increasing ViT blocks in [1, {self.depth}] to read from. Provided: {layers}"
            )
        if len(reassemble_channels) == 0:
            reassemble_channels = [256, 512, 1024, 1024] if D >= 1024 else [D // 8, D // 4, D // 2, D]
        if len(reassemble_channels) != 4:
            raise ValueError(f"'reassemble_channels' needs 4 values. Provided: {reassemble_channels}")
        if readout not in ["ignore", "add", "project"]:
            raise ValueError(f"'readout' must be 'ignore', 'add' or 'project'. Provided: '{readout}'")
        self.layers = list(layers)
        self.readout = readout
        print(f"DPT reading the ViT blocks {self.layers} (readout '{readout}'), reassembled into {reassemble_channels} channels")

        # 1. Readout: fold the class token into the patch tokens
        if readout == "project":
            self.readout_projects = nn.ModuleList([nn.Sequential(nn.Linear(2 * D, D), nn.GELU()) for _ in range(4)])

        # 2. Reassemble: tokens -> maps at 4x, 2x, 1x and 0.5x the token grid resolution
        resample = [
            nn.ConvTranspose2d(reassemble_channels[0], reassemble_channels[0], kernel_size=4, stride=4),
            nn.ConvTranspose2d(reassemble_channels[1], reassemble_channels[1], kernel_size=2, stride=2),
            nn.Identity(),
            nn.Conv2d(reassemble_channels[3], reassemble_channels[3], kernel_size=3, stride=2, padding=1),
        ]
        self.reassemble = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(D, c, kernel_size=1), r) for c, r in zip(reassemble_channels, resample)]
        )
        # Project every level of the pyramid to the fusion channels
        self.layer_rn = nn.ModuleList(
            [nn.Conv2d(c, features, kernel_size=3, padding=1, bias=False) for c in reassemble_channels]
        )

        # 3. Fusion, from the coarsest level to the finest one (the coarsest, fusion[3], has no skip features)
        self.fusion = nn.ModuleList(
            [FeatureFusionBlock(features, normalization, decoder_activation, has_skip=i < 3) for i in range(4)]
        )

        # 4. Head (as DPT's depth head): to the input resolution
        self.head_conv1 = nn.Conv2d(features, features // 2, kernel_size=3, padding=1)
        self.head_conv2 = nn.Conv2d(features // 2, 32, kernel_size=3, padding=1)
        self.head_act = dpt_activation(decoder_activation)
        return 32

    def _fold_readout(self, tokens: torch.Tensor, i: int) -> torch.Tensor:
        """Fold the class token of ``(B, 1 + N, D)`` tokens into the ``N`` patch tokens."""
        cls, patches = tokens[:, :1], tokens[:, 1:]
        if self.readout == "add":
            return patches + cls
        if self.readout == "project":
            return self.readout_projects[i](torch.cat([patches, cls.expand_as(patches)], dim=-1))
        return patches

    def _dpt_decode(self, tokens: List[torch.Tensor], out_size: Tuple[int, int]) -> torch.Tensor:
        """Tokens of the ``layers`` blocks (class token first) to ``out_size`` features."""
        pyramid = []
        for i, t in enumerate(tokens):
            x = self.reassemble[i](self.tokens_to_map(self._fold_readout(t, i)))
            pyramid.append(self.layer_rn[i](x))

        # Coarse to fine: each fusion block upsamples to the size of the next (finer) level
        x = self.fusion[3](pyramid[3], None, pyramid[2].shape[-2:])
        x = self.fusion[2](x, pyramid[2], pyramid[1].shape[-2:])
        x = self.fusion[1](x, pyramid[1], pyramid[0].shape[-2:])
        x = self.fusion[0](x, pyramid[0], tuple(2 * s for s in pyramid[0].shape[-2:]))

        x = self.head_conv1(x)
        x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=True)
        return self.head_act(self.head_conv2(x))
