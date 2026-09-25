"""
Wavelettention: Hybrid Attention Transformer for super-resolution, trained with an
additional wavelet-subband loss (see biapy.engine.swt_loss.SWTLoss).

Adapted from https://github.com/mandalinadagi/Wavelettention.git.
"""
import math
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from biapy.models.blocks import SqExBlock, get_activation


def to_2tuple(x):
    return x if isinstance(x, tuple) else (x, x)

def to_ntuple(x, n):
    return x if isinstance(x, tuple) else (x,) * n

def window_partition(x, window_size):
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows

def window_partition_3d(x, window_size):
    b, d, h, w, c = x.shape
    x = x.view(b, d // window_size, window_size, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size, window_size, window_size, c)
    return windows

def window_reverse(windows, window_size, h, w):
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x

def window_reverse_3d(windows, window_size, d, h, w):
    b = int(windows.shape[0] / (d * h * w / window_size**3))
    x = windows.view(b, d // window_size, h // window_size, w // window_size, window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, -1)
    return x

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True, ndim=2):
    conv = nn.Conv3d if ndim == 3 else nn.Conv2d
    return conv(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), stride=stride, bias=bias)


class BasicBlock(nn.Sequential):
    def __init__(self, conv, in_channels, out_channels, kernel_size, stride=1, bias=True, bn=False, act=nn.PReLU(), ndim=2):
        m = [conv(in_channels, out_channels, kernel_size, bias=bias, ndim=ndim)]
        if bn:
            m.append(nn.BatchNorm3d(out_channels) if ndim == 3 else nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)
        super().__init__(*m)


def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))


class NonLocalSparseAttention(nn.Module):
    """LSH-based non-local sparse attention, from https://github.com/HarukiYqM/Non-Local-Sparse-Attention."""

    def __init__(self, n_hashes=4, channels=64, k_size=3, reduction=4, chunk_size=144, res_scale=1, ndim=2):
        super().__init__()
        self.chunk_size = chunk_size
        self.n_hashes = n_hashes
        self.reduction = reduction
        self.res_scale = res_scale
        self.ndim = ndim
        self.conv_match = BasicBlock(default_conv, channels, channels // reduction, k_size, bn=False, act=None, ndim=ndim)
        self.conv_assembly = BasicBlock(default_conv, channels, channels, 1, bn=False, act=None, ndim=ndim)

    def LSH(self, hash_buckets, x):
        N = x.shape[0]
        device = x.device
        rotations_shape = (1, x.shape[-1], self.n_hashes, hash_buckets // 2)
        random_rotations = torch.randn(rotations_shape, dtype=x.dtype, device=device).expand(N, -1, -1, -1)
        rotated_vecs = torch.einsum("btf,bfhi->bhti", x, random_rotations)
        rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)
        hash_codes = torch.argmax(rotated_vecs, dim=-1)
        offsets = torch.arange(self.n_hashes, device=device)
        offsets = torch.reshape(offsets * hash_buckets, (1, -1, 1))
        hash_codes = torch.reshape(hash_codes + offsets, (N, -1))
        return hash_codes

    def add_adjacent_buckets(self, x):
        x_extra_back = torch.cat([x[:, :, -1:, ...], x[:, :, :-1, ...]], dim=2)
        x_extra_forward = torch.cat([x[:, :, 1:, ...], x[:, :, :1, ...]], dim=2)
        return torch.cat([x, x_extra_back, x_extra_forward], dim=3)

    def forward(self, input):
        if self.ndim == 3:
            N, _, D, H, W = input.shape
            spatial_shape = (D, H, W)
        else: 
            N, _, H, W = input.shape
            spatial_shape = (H, W)
        L_total = 1
        for s in spatial_shape:
            L_total *= s

        x_embed = self.conv_match(input).view(N, -1, L_total).contiguous().permute(0, 2, 1)
        y_embed = self.conv_assembly(input).view(N, -1, L_total).contiguous().permute(0, 2, 1)
        L, C = x_embed.shape[-2:]

        hash_buckets = min(L // self.chunk_size + (L // self.chunk_size) % 2, 128)
        hash_codes = self.LSH(hash_buckets, x_embed)
        hash_codes = hash_codes.detach()

        _, indices = hash_codes.sort(dim=-1)
        _, undo_sort = indices.sort(dim=-1)
        mod_indices = indices % L
        x_embed_sorted = batched_index_select(x_embed, mod_indices)
        y_embed_sorted = batched_index_select(y_embed, mod_indices)

        padding = self.chunk_size - L % self.chunk_size if L % self.chunk_size != 0 else 0
        x_att_buckets = torch.reshape(x_embed_sorted, (N, self.n_hashes, -1, C))
        y_att_buckets = torch.reshape(y_embed_sorted, (N, self.n_hashes, -1, C * self.reduction))
        if padding:
            pad_x = x_att_buckets[:, :, -padding:, :].clone()
            pad_y = y_att_buckets[:, :, -padding:, :].clone()
            x_att_buckets = torch.cat([x_att_buckets, pad_x], dim=2)
            y_att_buckets = torch.cat([y_att_buckets, pad_y], dim=2)

        x_att_buckets = torch.reshape(x_att_buckets, (N, self.n_hashes, -1, self.chunk_size, C))
        y_att_buckets = torch.reshape(y_att_buckets, (N, self.n_hashes, -1, self.chunk_size, C * self.reduction))

        x_match = F.normalize(x_att_buckets, p=2, dim=-1, eps=5e-5)
        x_match = self.add_adjacent_buckets(x_match)
        y_att_buckets = self.add_adjacent_buckets(y_att_buckets)

        raw_score = torch.einsum("bhkie,bhkje->bhkij", x_att_buckets, x_match)
        bucket_score = torch.logsumexp(raw_score, dim=-1, keepdim=True)
        score = torch.exp(raw_score - bucket_score)
        bucket_score = torch.reshape(bucket_score, [N, self.n_hashes, -1])

        ret = torch.einsum("bukij,bukje->bukie", score, y_att_buckets)
        ret = torch.reshape(ret, (N, self.n_hashes, -1, C * self.reduction))
        if padding:
            ret = ret[:, :, :-padding, :].clone()
            bucket_score = bucket_score[:, :, :-padding].clone()

        ret = torch.reshape(ret, (N, -1, C * self.reduction))
        bucket_score = torch.reshape(bucket_score, (N, -1))
        ret = batched_index_select(ret, undo_sort)
        bucket_score = bucket_score.gather(1, undo_sort)

        ret = torch.reshape(ret, (N, self.n_hashes, L, C * self.reduction))
        bucket_score = torch.reshape(bucket_score, (N, self.n_hashes, L, 1))
        probs = nn.functional.softmax(bucket_score, dim=1)
        ret = torch.sum(ret * probs, dim=1)

        ret = ret.permute(0, 2, 1).view(N, -1, *spatial_shape).contiguous() * self.res_scale + input
        return ret


class CAB(nn.Module):
    """Convolutional Attention Block: conv-GELU-conv + channel attention.

    Uses BiaPy's SqExBlock instead of the paper's own ChannelAttention (same
    squeeze-excitation mechanism, and SqExBlock already supports ndim=2/3).
    """

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30, ndim=2):
        super().__init__()
        conv = nn.Conv3d if ndim == 3 else nn.Conv2d
        self.cab = nn.Sequential(
            conv(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            conv(num_feat // compress_ratio, num_feat, 3, 1, 1),
            SqExBlock(num_feat, r=squeeze_factor, ndim=ndim),
        )

    def forward(self, x):
        return self.cab(x)


class WindowAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) with relative position bias."""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0, ndim=2):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.ndim = ndim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        table_size = 1
        for w in window_size:
            table_size *= 2 * w - 1
        self.relative_position_bias_table = nn.Parameter(torch.zeros(table_size, num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, rpi, mask=None):
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        window_tokens = 1
        for w in self.window_size:
            window_tokens *= w
        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(window_tokens, window_tokens, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class HAB(nn.Module):
    r"""Hybrid Attention Block: window attention + convolutional attention branch (CAB)."""

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        ndim=2,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.ndim = ndim
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_ntuple(self.window_size, ndim),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            ndim=ndim,
        )

        self.conv_scale = conv_scale
        self.conv_block = CAB(num_feat=dim, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor, ndim=ndim)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, x_size, rpi_sa, attn_mask):
        if self.ndim == 3:
            d, h, w = x_size
            b, _, c = x.shape

            shortcut = x
            x = self.norm1(x)
            x = x.view(b, d, h, w, c)

            conv_x = self.conv_block(x.permute(0, 4, 1, 2, 3))
            conv_x = conv_x.permute(0, 2, 3, 4, 1).contiguous().view(b, d * h * w, c)

            if self.shift_size > 0:
                shifted_x = torch.roll(x, shifts=(-self.shift_size,) * 3, dims=(1, 2, 3))
            else:
                shifted_x = x
                attn_mask = None

            x_windows = window_partition_3d(shifted_x, self.window_size)
            x_windows = x_windows.view(-1, self.window_size**3, c)

            attn_windows = self.attn(x_windows, rpi=rpi_sa, mask=attn_mask)

            attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.window_size, c)
            shifted_x = window_reverse_3d(attn_windows, self.window_size, d, h, w)

            if self.shift_size > 0:
                attn_x = torch.roll(shifted_x, shifts=(self.shift_size,) * 3, dims=(1, 2, 3))
            else:
                attn_x = shifted_x
            attn_x = attn_x.view(b, d * h * w, c)

            x = shortcut + self.drop_path(attn_x) + conv_x * self.conv_scale
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

        # Original 2D path, unchanged
        h, w = x_size
        b, _, c = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        conv_x = self.conv_block(x.permute(0, 3, 1, 2))
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)

        attn_windows = self.attn(x_windows, rpi=rpi_sa, mask=attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)

        if self.shift_size > 0:
            attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_x = shifted_x
        attn_x = attn_x.view(b, h * w, c)

        x = shortcut + self.drop_path(attn_x) + conv_x * self.conv_scale
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class OCAB(nn.Module):
    """Overlapping cross-attention block."""

    def __init__(
        self, dim, input_resolution, window_size, overlap_ratio, num_heads,
        qkv_bias=True, qk_scale=None, mlp_ratio=2, norm_layer=nn.LayerNorm, ndim=2,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.ndim = ndim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size

        self.norm1 = norm_layer(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        if ndim == 2:
            self.unfold = nn.Unfold(
                kernel_size=(self.overlap_win_size, self.overlap_win_size),
                stride=window_size,
                padding=(self.overlap_win_size - window_size) // 2,
            )
        # ndim == 3: no nn.Unfold3d exists -- see _unfold_3d, called manually in forward()

        table_size = (window_size + self.overlap_win_size - 1) ** ndim
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(table_size, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

        self.proj = nn.Linear(dim, dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU)

    def _unfold_3d(self, kv):
        """Manual 3D equivalent of nn.Unfold (which only supports 4D input), verified against
        nn.Unfold's 2D behaviour before generalizing."""
        n, c2 = kv.shape[0], kv.shape[1]
        # Asymmetric padding (floor on one side, ceil on the other) instead of a single
        # symmetric `pad` value: with an odd (overlap_win_size - window_size), a plain
        # `//2` on each side under-pads by 1, which only bites when there's a single
        # window along that axis (e.g. a volume with very few Z slices) -- there just
        # isn't a neighbouring window to make up the shortfall.
        extra = self.overlap_win_size - self.window_size
        pad_before = extra // 2
        pad_after = extra - pad_before
        kv = F.pad(kv, (pad_before, pad_after, pad_before, pad_after, pad_before, pad_after))
        kv = kv.unfold(2, self.overlap_win_size, self.window_size).unfold(3, self.overlap_win_size, self.window_size).unfold(4, self.overlap_win_size, self.window_size)
        nd, nh, nw = kv.shape[2:5]
        kv = kv.permute(0, 1, 5, 6, 7, 2, 3, 4).contiguous().view(n, c2 * self.overlap_win_size**3, nd * nh * nw)
        return kv

    def forward(self, x, x_size, rpi):
        if self.ndim == 3 :
            d, h, w = x_size
            b, _, c = x.shape
            ws = self.window_size

            shortcut = x
            x = self.norm1(x)
            x = x.view(b, d, h, w, c)

            qkv = self.qkv(x).reshape(b, d, h, w, 3, c).permute(4, 0, 5, 1, 2, 3)
            q = qkv[0].permute(0, 2, 3, 4, 1)
            kv = torch.cat((qkv[1], qkv[2]), dim=1)

            q_windows = window_partition_3d(q, ws)
            q_windows = q_windows.view(-1, ws * ws * ws, c)

            kv_windows = self._unfold_3d(kv)
            kv_windows = rearrange(
                kv_windows, "b (nc ch owd owh oww) nw -> nc (b nw) (owd owh oww) ch",
                nc=2, ch=c, owd=self.overlap_win_size, owh=self.overlap_win_size, oww=self.overlap_win_size,
            ).contiguous()
            k_windows, v_windows = kv_windows[0], kv_windows[1]

            b_, nq, _ = q_windows.shape
            _, n, _ = k_windows.shape
            dh = self.dim // self.num_heads
            q = q_windows.reshape(b_, nq, self.num_heads, dh).permute(0, 2, 1, 3)
            k = k_windows.reshape(b_, n, self.num_heads, dh).permute(0, 2, 1, 3)
            v = v_windows.reshape(b_, n, self.num_heads, dh).permute(0, 2, 1, 3)

            q = q * self.scale
            attn = q @ k.transpose(-2, -1)

            relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
                ws * ws * ws, self.overlap_win_size**3, -1
            )
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)

            attn = self.softmax(attn)
            attn_windows = (attn @ v).transpose(1, 2).reshape(b_, nq, self.dim)

            attn_windows = attn_windows.view(-1, ws, ws, ws, self.dim)
            x = window_reverse_3d(attn_windows, ws, d, h, w)
            x = x.view(b, d * h * w, self.dim)

            x = self.proj(x) + shortcut
            x = x + self.mlp(self.norm2(x))
            return x

        # Original 2D path, unchanged
        h, w = x_size
        b, _, c = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        qkv = self.qkv(x).reshape(b, h, w, 3, c).permute(3, 0, 4, 1, 2)
        q = qkv[0].permute(0, 2, 3, 1)
        kv = torch.cat((qkv[1], qkv[2]), dim=1)

        q_windows = window_partition(q, self.window_size)
        q_windows = q_windows.view(-1, self.window_size * self.window_size, c)

        kv_windows = self.unfold(kv)
        kv_windows = rearrange(
            kv_windows, "b (nc ch owh oww) nw -> nc (b nw) (owh oww) ch",
            nc=2, ch=c, owh=self.overlap_win_size, oww=self.overlap_win_size,
        ).contiguous()
        k_windows, v_windows = kv_windows[0], kv_windows[1]

        b_, nq, _ = q_windows.shape
        _, n, _ = k_windows.shape
        d = self.dim // self.num_heads
        q = q_windows.reshape(b_, nq, self.num_heads, d).permute(0, 2, 1, 3)
        k = k_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3)
        v = v_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size * self.window_size, self.overlap_win_size * self.overlap_win_size, -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)
        attn_windows = (attn @ v).transpose(1, 2).reshape(b_, nq, self.dim)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.dim)
        x = window_reverse(attn_windows, self.window_size, h, w)
        x = x.view(b, h * w, self.dim)

        x = self.proj(x) + shortcut
        x = x + self.mlp(self.norm2(x))
        return x


class AttenBlocks(nn.Module):
    """A series of HAB blocks for one RHAG, followed by an OCAB."""

    def __init__(
        self, dim, input_resolution, depth, num_heads, window_size, compress_ratio, squeeze_factor,
        conv_scale, overlap_ratio, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0,
        drop_path=0.0, norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False, ndim=2,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList(
            [
                HAB(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    compress_ratio=compress_ratio,
                    squeeze_factor=squeeze_factor,
                    conv_scale=conv_scale,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    ndim=ndim,
                )
                for i in range(depth)
            ]
        )

        self.overlap_attn = OCAB(
            dim=dim,
            input_resolution=input_resolution,
            window_size=window_size,
            overlap_ratio=overlap_ratio,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            ndim=ndim,
        )

        self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer) if downsample is not None else None

    def forward(self, x, x_size, params):
        for blk in self.blocks:
            x = blk(x, x_size, params["rpi_sa"], params["attn_mask"])
        x = self.overlap_attn(x, x_size, params["rpi_oca"])
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchEmbed(nn.Module):
    """Flatten (B, C, H, W) -> (B, H*W, C). NOT the same as biapy.models.tr_layers.PatchEmbed
    (which projects with a strided conv) -- here patch_size is always 1, projection already
    happened via conv_first before this is called."""

    def __init__(self, img_size=64, patch_size=1, in_chans=3, embed_dim=96, norm_layer=None, ndim=2):
        super().__init__()
        img_size = to_ntuple(img_size, ndim)
        patch_size = to_ntuple(patch_size, ndim)
        patches_resolution = [img_size[i] // patch_size[i] for i in range(ndim)]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = 1
        for r in patches_resolution:
            self.num_patches *= r
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    """Reverse of PatchEmbed: (B, H*W, C) -> (B, C, H, W)."""

    def __init__(self, img_size=64, patch_size=1, in_chans=3, embed_dim=96, norm_layer=None, ndim=2):
        super().__init__()
        img_size = to_ntuple(img_size, ndim)
        patch_size = to_ntuple(patch_size, ndim)
        patches_resolution = [img_size[i] // patch_size[i] for i in range(ndim)]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = 1
        for r in patches_resolution:
            self.num_patches *= r
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).contiguous().view(x.shape[0], self.embed_dim, *x_size)
        return x


class RHAG(nn.Module):
    """Residual Hybrid Attention Group."""

    def __init__(
        self, dim, input_resolution, depth, num_heads, window_size, compress_ratio, squeeze_factor,
        conv_scale, overlap_ratio, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0,
        drop_path=0.0, norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
        img_size=224, patch_size=4, resi_connection="1conv", ndim=2,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = AttenBlocks(
            dim=dim, input_resolution=input_resolution, depth=depth, num_heads=num_heads,
            window_size=window_size, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor,
            conv_scale=conv_scale, overlap_ratio=overlap_ratio, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer,
            downsample=downsample, use_checkpoint=use_checkpoint, ndim=ndim,
        )

        if resi_connection == "1conv":
            conv = nn.Conv3d if ndim == 3 else nn.Conv2d
            self.conv = conv(dim, dim, 3, 1, 1)
        elif resi_connection == "identity":
            self.conv = nn.Identity()

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None, ndim=ndim)
        self.patch_unembed = PatchUnEmbed(img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None, ndim=ndim)

    def forward(self, x, x_size, params):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size, params), x_size))) + x


class PixelShuffle3D(nn.Module):
    """Upscales only the last two spatial dims (H, W) of a 5D (N, C, D, H, W) tensor by `r`,
    leaving D untouched. nn.PixelShuffle treats the last 3 dims as (C, H, W), so it can't be
    applied directly to 5D input -- fold D into the batch dim, run standard 2D pixel shuffle
    per D-slice, then unfold."""

    def __init__(self, r):
        super().__init__()
        self.r = r

    def forward(self, x):
        n, c, d, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(n * d, c, h, w)
        x = F.pixel_shuffle(x, self.r)
        _, c2, h2, w2 = x.shape
        x = x.view(n, d, c2, h2, w2).permute(0, 2, 1, 3, 4).contiguous()
        return x


class Upsample(nn.Sequential):
    """Pixel-shuffle upsampling. Only supports scale = 2**n or 3."""

    def __init__(self, scale, num_feat, ndim=2):
        conv = nn.Conv3d if ndim == 3 else nn.Conv2d
        shuffle = PixelShuffle3D if ndim == 3 else nn.PixelShuffle
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(shuffle(2))
        elif scale == 3:
            m.append(conv(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(shuffle(3))
        else:
            raise ValueError(f"scale {scale} is not supported. Supported scales: 2^n and 3.")
        super().__init__(*m)


class Wavelettention(nn.Module):
    r"""Hybrid Attention Transformer with NLSA blocks, trained with an additional wavelet loss
    (see biapy.engine.swt_loss.SWTLoss). 2D only.

    Reference: `Training Transformer Models by Wavelet Losses Improves Quantitative and Visual
    Performance in Single Image Super-Resolution`. Based on SwinIR and HAT.
    """

    def __init__(
        self,
        img_size=64,
        patch_size=1,
        in_chans=3,
        embed_dim=96,
        depths=(6, 6, 6, 6),
        num_heads=(6, 6, 6, 6),
        window_size=7,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        upscale=2,
        img_range=1.0,
        upsampler="pixelshuffle",
        resi_connection="1conv",
        ndim=2,
        **kwargs,
    ):
        super().__init__()

        # Sanitize upscale: build_model may pass PROBLEM.SUPER_RESOLUTION.UPSCALING as a tuple.
        if not isinstance(upscale, int) and isinstance(upscale, Sequence):
            upscale = upscale[0]
        # img_size is left as-is (int or per-axis tuple) -- to_ntuple() and every PatchEmbed/
        # PatchUnEmbed below already handle both forms correctly, so collapsing a real
        # per-axis tuple down to its first element here would silently throw away the Y/X
        # shape for anisotropic (e.g. 3D) inputs.

        self.window_size = window_size
        self.shift_size = window_size // 2
        self.overlap_ratio = overlap_ratio

        # Auto-shrink the (cubic) window so it evenly divides every spatial axis of the
        # input -- e.g. a volume with only a handful of Z slices that doesn't share a large
        # common divisor with the Y/X extent. Using the largest divisor of gcd(D, H, W) that
        # is <= the requested window_size (instead of just min(D, H, W)) means the user's
        # PATCH_SIZE doesn't need to be hand-picked to be a multiple of anything: whatever
        # shape they give, a valid cubic window always exists. Also keeps the precomputed
        # position tables (calculate_rpi_sa/calculate_rpi_oca, built right below) consistent
        # with what every HAB/OCAB block will actually use.
        _img_size_t = to_ntuple(img_size, ndim)
        _patch_size_t = to_ntuple(patch_size, ndim)
        _resolution = [_img_size_t[i] // _patch_size_t[i] for i in range(ndim)]
        _gcd = _resolution[0]
        for _r in _resolution[1:]:
            _gcd = math.gcd(_gcd, _r)
        _w = min(self.window_size, _gcd)
        while _gcd % _w != 0:
            _w -= 1
        if _w != self.window_size:
            self.window_size = _w
            self.shift_size = self.window_size // 2

        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            mean_shape = (1, 3, 1, 1, 1) if ndim == 3 else (1, 3, 1, 1)
            self.mean = torch.Tensor(rgb_mean).view(*mean_shape)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.ndim = ndim
        self.upsampler = upsampler

        relative_position_index_SA = self.calculate_rpi_sa_3d() if ndim == 3 else self.calculate_rpi_sa()
        relative_position_index_OCA = self.calculate_rpi_oca_3d() if ndim == 3 else self.calculate_rpi_oca()
        self.register_buffer("relative_position_index_SA", relative_position_index_SA)
        self.register_buffer("relative_position_index_OCA", relative_position_index_OCA)

        conv = nn.Conv3d if ndim == 3 else nn.Conv2d
        self.conv_first = conv(num_in_ch, embed_dim, 3, 1, 1)

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None, ndim=ndim,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None, ndim=ndim,
        )

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            nn.init.trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.nlsa1 = NonLocalSparseAttention(channels=embed_dim, chunk_size=36, n_hashes=4, reduction=4, res_scale=1, ndim=ndim)
        self.nlsa2 = NonLocalSparseAttention(channels=embed_dim, chunk_size=36, n_hashes=4, reduction=4, res_scale=1, ndim=ndim)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RHAG(
                dim=embed_dim,
                input_resolution=tuple(patches_resolution),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=self.window_size,
                compress_ratio=compress_ratio,
                squeeze_factor=squeeze_factor,
                conv_scale=conv_scale,
                overlap_ratio=overlap_ratio,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
                ndim=ndim,
            )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        self.nlsa3 = NonLocalSparseAttention(channels=embed_dim, chunk_size=36, n_hashes=4, reduction=4, res_scale=1, ndim=ndim)
        self.nlsa4 = NonLocalSparseAttention(channels=embed_dim, chunk_size=36, n_hashes=4, reduction=4, res_scale=1, ndim=ndim)

        if resi_connection == "1conv":
            self.conv_after_body = conv(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == "identity":
            self.conv_after_body = nn.Identity()

        # Only 'pixelshuffle' is implemented -- required for BiaPy's build_model call.
        if self.upsampler == "pixelshuffle":
            self.conv_before_upsample = nn.Sequential(
                conv(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True)
            )
            self.upsample = Upsample(upscale, num_feat, ndim=ndim)
            self.conv_last = conv(num_feat, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def calculate_rpi_sa(self):
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index

    def calculate_rpi_sa_3d(self):  
        ws = self.window_size
        coords_d = torch.arange(ws)
        coords_h = torch.arange(ws)
        coords_w = torch.arange(ws)
        coords = torch.stack(torch.meshgrid([coords_d, coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += ws - 1
        relative_coords[:, :, 1] += ws - 1
        relative_coords[:, :, 2] += ws - 1
        relative_coords[:, :, 0] *= (2 * ws - 1) ** 2
        relative_coords[:, :, 1] *= (2 * ws - 1)
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index

    def calculate_rpi_oca(self):
        window_size_ori = self.window_size
        window_size_ext = self.window_size + int(self.overlap_ratio * self.window_size)

        coords_h = torch.arange(window_size_ori)
        coords_w = torch.arange(window_size_ori)
        coords_ori = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_ori_flatten = torch.flatten(coords_ori, 1)

        coords_h = torch.arange(window_size_ext)
        coords_w = torch.arange(window_size_ext)
        coords_ext = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_ext_flatten = torch.flatten(coords_ext, 1)

        relative_coords = coords_ext_flatten[:, None, :] - coords_ori_flatten[:, :, None]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size_ori - window_size_ext + 1
        relative_coords[:, :, 1] += window_size_ori - window_size_ext + 1
        relative_coords[:, :, 0] *= window_size_ori + window_size_ext - 1
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index

    def calculate_rpi_oca_3d(self):
        window_size_ori = self.window_size
        window_size_ext = self.window_size + int(self.overlap_ratio * self.window_size)

        coords_d = torch.arange(window_size_ori)
        coords_h = torch.arange(window_size_ori)
        coords_w = torch.arange(window_size_ori)
        coords_ori = torch.stack(torch.meshgrid([coords_d, coords_h, coords_w]))
        coords_ori_flatten = torch.flatten(coords_ori, 1)

        coords_d = torch.arange(window_size_ext)
        coords_h = torch.arange(window_size_ext)
        coords_w = torch.arange(window_size_ext)
        coords_ext = torch.stack(torch.meshgrid([coords_d, coords_h, coords_w]))
        coords_ext_flatten = torch.flatten(coords_ext, 1)

        relative_coords = coords_ext_flatten[:, None, :] - coords_ori_flatten[:, :, None]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size_ori - window_size_ext + 1
        relative_coords[:, :, 1] += window_size_ori - window_size_ext + 1
        relative_coords[:, :, 2] += window_size_ori - window_size_ext + 1
        relative_coords[:, :, 0] *= (window_size_ori + window_size_ext - 1) ** 2
        relative_coords[:, :, 1] *= window_size_ori + window_size_ext - 1
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index


    def calculate_mask(self, x_size):
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def calculate_mask_3d(self, x_size):
        d, h, w = x_size
        img_mask = torch.zeros((1, d, h, w, 1))
        d_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for d_ in d_slices:
            for h_ in h_slices:
                for w_ in w_slices:
                    img_mask[:, d_, h_, w_, :] = cnt
                    cnt += 1

        mask_windows = window_partition_3d(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size**3)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask


    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward_features(self, x):
        if self.ndim == 3:
            x_size = (x.shape[2], x.shape[3], x.shape[4])
            attn_mask = self.calculate_mask_3d(x_size).to(x.device)
        else:
            x_size = (x.shape[2], x.shape[3])
            attn_mask = self.calculate_mask(x_size).to(x.device)

        params = {"attn_mask": attn_mask, "rpi_sa": self.relative_position_index_SA, "rpi_oca": self.relative_position_index_OCA}

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size, params)

        x = self.norm(x)
        x = self.patch_unembed(x, x_size)
        return x

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == "pixelshuffle":
            x = self.conv_first(x)
            nlsa_st = self.nlsa2(self.nlsa1(x))
            ff = self.forward_features(nlsa_st)
            x = self.conv_after_body(ff) + x
            nlsa_fn = self.nlsa4(self.nlsa3(x))
            x = self.conv_before_upsample(nlsa_fn)
            x = self.conv_last(self.upsample(x))

        x = x / self.img_range + self.mean
        return x

