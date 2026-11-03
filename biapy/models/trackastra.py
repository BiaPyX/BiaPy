"""
Trackastra's linking transformer (https://github.com/weigertlab/trackastra), used by the tracking workflow.

Inference-only port of ``trackastra.model`` (``TrackingTransformer`` with rotary positional encoding and its
causal output normalization). Module names match Trackastra's, so its released weights load unchanged.

Copyright (c) 2024, Benjamin Gallusser, Martin Weigert. BSD 3-Clause License:

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
   disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
   following disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
   products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import math
import os
import zipfile
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
import yaml
from torch import nn

from biapy.models.vit_backbones.weights import fetch_vit_weights

PRETRAINED_TRACKASTRA = {
    "trackastra_general_2d": "https://github.com/weigertlab/trackastra-models/releases/download/v0.3.0/general_2d.zip",
    "trackastra_ctc": "https://github.com/weigertlab/trackastra-models/releases/download/v0.3.0/ctc.zip",
}


def _fourier_freqs(cutoff: float, n: int, cutoff_start: float = 1) -> torch.Tensor:
    return torch.exp(torch.linspace(-math.log(cutoff_start), -math.log(cutoff), n)).unsqueeze(0).unsqueeze(0)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.unflatten(-1, (-1, 2)).unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, expand: float = 2):
        super().__init__()
        self.fc1 = nn.Linear(d_model, int(d_model * expand))
        self.fc2 = nn.Linear(int(d_model * expand), d_model)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class PositionalEncoding(nn.Module):
    """Fourier encoding of each coordinate (or feature) with ``n_pos[i]`` frequencies up to ``cutoffs[i]``."""

    def __init__(self, cutoffs: Tuple[float, ...], n_pos: Tuple[int, ...], cutoffs_start=None):
        super().__init__()
        if cutoffs_start is None:
            cutoffs_start = (1,) * len(cutoffs)
        self.freqs = nn.ParameterList(
            [nn.Parameter(_fourier_freqs(c, n // 2, s)) for c, n, s in zip(cutoffs, n_pos, cutoffs_start)]
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            tuple(
                torch.cat(
                    (torch.sin(0.5 * math.pi * x.unsqueeze(-1) * f), torch.cos(0.5 * math.pi * x.unsqueeze(-1) * f)),
                    dim=-1,
                )
                for x, f in zip(coords.moveaxis(-1, 0), self.freqs)
            ),
            dim=-1,
        )


class RotaryPositionalEncoding(nn.Module):
    def __init__(self, cutoffs: Tuple[float, ...], n_pos: Tuple[int, ...]):
        super().__init__()
        self.freqs = nn.ParameterList(
            [
                nn.Parameter(torch.exp(torch.linspace(0, -math.log(c), n // 2)).unsqueeze(0).unsqueeze(0))
                for c, n in zip(cutoffs, n_pos)
            ]
        )

    def forward(self, q: torch.Tensor, k: torch.Tensor, coords: torch.Tensor):
        angles = [0.5 * math.pi * x.unsqueeze(-1) * f for x, f in zip(coords.moveaxis(-1, 0), self.freqs)]
        co = torch.cat([torch.cos(a) for a in angles], dim=-1)
        si = torch.cat([torch.sin(a) for a in angles], dim=-1)
        co = co.unsqueeze(1).repeat_interleave(2, dim=-1)
        si = si.unsqueeze(1).repeat_interleave(2, dim=-1)
        return q * co + _rotate_half(q) * si, k * co + _rotate_half(k) * si


class RelativePositionalAttention(nn.Module):
    def __init__(
        self,
        coord_dim: int,
        embed_dim: int,
        n_head: int,
        cutoff_spatial: float,
        cutoff_temporal: float,
        attn_dist_mode: str = "v0",
    ):
        super().__init__()
        self.q_pro = nn.Linear(embed_dim, embed_dim)
        self.k_pro = nn.Linear(embed_dim, embed_dim)
        self.v_pro = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.n_head = n_head
        self.cutoff_spatial = cutoff_spatial
        self.attn_dist_mode = attn_dist_mode
        n_split = 2 * (embed_dim // (2 * (coord_dim + 1) * n_head))
        self.rot_pos_enc = RotaryPositionalEncoding(
            cutoffs=(cutoff_temporal,) + (cutoff_spatial,) * coord_dim,
            n_pos=(embed_dim // n_head - coord_dim * n_split,) + (n_split,) * coord_dim,
        )

    def forward(self, query, key, value, coords):
        B, N, D = query.size()
        q, k, v = (
            p(x).view(B, N, self.n_head, D // self.n_head).transpose(1, 2)
            for p, x in ((self.q_pro, query), (self.k_pro, key), (self.v_pro, value))
        )
        spatial_dist = torch.cdist(coords[..., 1:], coords[..., 1:])
        attn_mask = torch.zeros((B, self.n_head, N, N), device=query.device, dtype=q.dtype)
        attn_mask.masked_fill_((spatial_dist > self.cutoff_spatial).unsqueeze(1), -1e3)
        q, k = self.rot_pos_enc(q, k, coords)
        if self.attn_dist_mode == "v0":
            attn_mask += torch.exp(-0.1 * torch.cdist(coords, coords).unsqueeze(1))
        else:
            attn_mask += torch.exp(-5 * spatial_dist.unsqueeze(1) / self.cutoff_spatial)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.proj(y.transpose(1, 2).contiguous().view(B, N, D))


class EncoderLayer(nn.Module):
    def __init__(self, coord_dim, d_model, num_heads, window, cutoff_spatial, attn_dist_mode):
        super().__init__()
        self.attn = RelativePositionalAttention(coord_dim, d_model, num_heads, cutoff_spatial, window, attn_dist_mode)
        self.mlp = FeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, coords):
        x = self.norm1(x)
        x = x + self.attn(x, x, x, coords)
        return x + self.mlp(self.norm2(x))


class DecoderLayer(nn.Module):
    def __init__(self, coord_dim, d_model, num_heads, window, cutoff_spatial, attn_dist_mode):
        super().__init__()
        self.attn = RelativePositionalAttention(coord_dim, d_model, num_heads, cutoff_spatial, window, attn_dist_mode)
        self.mlp = FeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, y, coords):
        x = self.norm1(x)
        y = self.norm2(y)
        x = x + self.attn(x, y, y, coords)
        return x + self.mlp(self.norm3(x))


def _blockwise_reduce(A: torch.Tensor, timepoints: torch.Tensor, dim: int, reduce: str = "sum") -> torch.Tensor:
    """Reduce ``A`` along ``dim`` within each block of equal timepoints, broadcast back to ``A``'s shape."""
    A = A.transpose(dim, 0)
    ts = timepoints - timepoints.min() + 1
    index = ts.unsqueeze(1).expand(-1, len(ts))
    out = torch.zeros((int(ts.max()) + 1, A.shape[1]), device=A.device, dtype=A.dtype)
    return torch.scatter_reduce(out, 0, index, A, reduce=reduce)[ts].transpose(0, dim)


def blockwise_causal_norm(A: torch.Tensor, timepoints: torch.Tensor, mask_invalid: torch.Tensor, eps: float = 1e-6):
    """
    Trackastra's "quiet softmax": for each object, softmax over the objects of each other frame, with an extra 1 in
    the denominator so all the scores of a row/column can be close to 0 (no link).

    Parameters
    ----------
    A : torch.Tensor
        ``(N, N)`` association logits.

    timepoints : torch.Tensor
        ``(N,)`` frame of each object.

    mask_invalid : torch.Tensor
        ``(N, N)`` pairs that can not be linked.

    Returns
    -------
    A : torch.Tensor
        ``(N, N)`` association scores in ``[0, 1]``.
    """
    A = A.clone()
    A[mask_invalid] = -torch.inf
    with torch.no_grad():
        ma0 = _blockwise_reduce(A, timepoints, dim=0, reduce="amax")
        ma1 = _blockwise_reduce(A, timepoints, dim=1, reduce="amax")
    u0 = torch.exp(A - ma0)
    u1 = torch.exp(A - ma1)
    u0_sum = _blockwise_reduce(u0, timepoints, dim=0) + eps + torch.exp(-ma0)
    u1_sum = _blockwise_reduce(u1, timepoints, dim=1) + eps + torch.exp(-ma1)
    mask0 = timepoints.unsqueeze(0) > timepoints.unsqueeze(1)
    return torch.clamp(mask0 * u0 / u0_sum + ~mask0 * u1 / u1_sum, 0, 1)


class TrackingTransformer(nn.Module):
    """
    Association scores between the objects of a window of frames.

    Each object is a token made of its position ``(t, y, x)`` and its region features. The output ``A[i, j]`` is how
    likely object ``j`` is object ``i`` (or one of its daughters) in a later frame.
    """

    def __init__(
        self,
        coord_dim: int = 2,
        feat_dim: int = 7,
        d_model: int = 320,
        nhead: int = 4,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        pos_embed_per_dim: int = 32,
        feat_embed_per_dim: int = 8,
        window: int = 4,
        spatial_pos_cutoff: int = 256,
        attn_positional_bias: str = "rope",
        attn_positional_bias_n_spatial: int = 16,
        causal_norm: str = "quiet_softmax",
        attn_dist_mode: str = "v0",
        dropout: float = 0.0,
    ):
        super().__init__()
        if attn_positional_bias != "rope" or causal_norm != "quiet_softmax" or feat_embed_per_dim <= 1:
            raise NotImplementedError(
                "Only Trackastra models with 'rope' positional bias, 'quiet_softmax' normalization and embedded "
                "features are supported"
            )
        self.coord_dim = coord_dim
        self.window = window
        self.spatial_pos_cutoff = spatial_pos_cutoff
        self.proj = nn.Linear((1 + coord_dim) * pos_embed_per_dim + feat_dim * feat_embed_per_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        layer_args = (coord_dim, d_model, nhead, window, spatial_pos_cutoff, attn_dist_mode)
        self.encoder = nn.ModuleList([EncoderLayer(*layer_args) for _ in range(num_encoder_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(*layer_args) for _ in range(num_decoder_layers)])
        self.head_x = FeedForward(d_model)
        self.head_y = FeedForward(d_model)
        self.feat_embed = PositionalEncoding(
            cutoffs=(1000,) * feat_dim, n_pos=(feat_embed_per_dim,) * feat_dim, cutoffs_start=(0.01,) * feat_dim
        )
        self.pos_embed = PositionalEncoding(
            cutoffs=(window,) + (spatial_pos_cutoff,) * coord_dim, n_pos=(pos_embed_per_dim,) * (1 + coord_dim)
        )

    def forward(self, coords: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        coords : torch.Tensor
            ``(1, N, 1 + coord_dim)`` frame and position of each object.

        features : torch.Tensor
            ``(1, N, feat_dim)`` region features.

        Returns
        -------
        A : torch.Tensor
            ``(1, N, N)`` association scores in ``[0, 1]``.
        """
        timepoints = coords[0, :, 0].long()
        coords = coords - coords[:, :, :1].min(dim=1, keepdim=True).values
        x = self.norm(self.proj(torch.cat((self.pos_embed(coords), self.feat_embed(features)), dim=-1)))
        y = x
        for enc in self.encoder:
            x = enc(x, coords)
        for dec in self.decoder:
            y = dec(y, x, coords)
        A = torch.einsum("bnd,bmd->bnm", self.head_x(x), self.head_y(y))[0]
        invalid = torch.cdist(coords[0, :, 1:], coords[0, :, 1:]) > self.spatial_pos_cutoff
        return blockwise_causal_norm(A, timepoints, invalid).unsqueeze(0)


def load_trackastra(name: str, device: torch.device) -> TrackingTransformer:
    """
    Pretrained Trackastra model, downloaded the first time.

    Parameters
    ----------
    name : str
        Key of ``PRETRAINED_TRACKASTRA``.

    device : torch.device
        Device to place the model in.

    Returns
    -------
    model : TrackingTransformer
        Model in evaluation mode.
    """
    zip_path = fetch_vit_weights(name, named_urls=PRETRAINED_TRACKASTRA, cache_subdir="trackastra")
    folder = os.path.splitext(zip_path)[0]
    if not os.path.isdir(folder):
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(os.path.dirname(zip_path))
    with open(os.path.join(folder, "config.yaml")) as f:
        config: Dict = yaml.safe_load(f)
    model = TrackingTransformer(**config)
    model.load_state_dict(torch.load(os.path.join(folder, "model.pt"), map_location="cpu", weights_only=True))
    print(f"Loaded the Trackastra linker '{name}' ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters)")
    return model.to(device).eval()
