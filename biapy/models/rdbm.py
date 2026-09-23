"""Residual Diffusion Bridge Model (RDBM) for paired image-to-image translation.

Paper: Wang et al., "Residual Diffusion Bridge Model for Image Restoration", CVPR 2026.
https://arxiv.org/abs/2510.23116
Reference code: https://github.com/MiliLab/RDBM (files ``rdbm.py``, ``networks.py``).

Standalone re-implementation: no einops/accelerate/ema_pytorch, just torch. Deviates from the
reference in three ways:
- ``_bounded`` uses a straight-through clamp instead of a plain ``torch.clamp`` (avoids a
  zero-gradient dead zone outside [-1, 1]).
- 2D only; 3D is rejected by ``check_configuration.py``.
- the U-Net runs under gradient checkpointing per level (numerically identical, lower peak
  memory).

Helper classes are prefixed with ``_`` so BiaPy's model loader, which pulls every public name
from this module into ``biapy.models``' namespace, only picks up ``RDBM`` itself.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _grad_checkpoint


def _default(val, d):
    if val is not None:
        return val
    return d() if callable(d) else d


def _extract(a, t, x_shape):
    out = a.gather(-1, t)
    return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))


def _betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999):
    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


class _SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None].float() * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class _WeightStandardizedConv2d(nn.Conv2d):
    """Weight standardization (https://arxiv.org/abs/1903.10520); pairs with GroupNorm."""

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        weight = self.weight
        mean = weight.mean(dim=(1, 2, 3), keepdim=True)
        var = weight.var(dim=(1, 2, 3), unbiased=False, keepdim=True)
        normalized_weight = (weight - mean) * (var + eps).rsqrt()
        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class _LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class _PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = _LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))


class _Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class _Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = _WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.norm(self.proj(x))
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return self.act(x)


class _ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
        self.block1 = _Block(dim, dim_out, groups=groups)
        self.block2 = _Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        scale_shift = self.mlp(time_emb)[:, :, None, None].chunk(2, dim=1)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class _LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), _LayerNorm(dim))

    def forward(self, x):
        b, _, H, W = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = (t.reshape(b, self.heads, -1, H * W) for t in qkv)
        q = q.softmax(dim=-2) * self.scale
        k = k.softmax(dim=-1)
        v = v / (H * W)
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = out.reshape(b, -1, H, W)
        return self.to_out(out)


def _upsample_2d(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, _default(dim_out, dim), 3, padding=1),
    )


def _downsample_2d(dim, dim_out=None):
    return nn.Conv2d(dim, _default(dim_out, dim), 4, 2, 1)


class _Attention(nn.Module):
    """Full (quadratic) self-attention; used only at the bottleneck."""

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, _, H, W = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = (t.reshape(b, self.heads, -1, H * W) for t in qkv)
        q = q * self.scale
        sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        attn = sim.softmax(dim=-1)
        out = torch.einsum("b h i j, b h d j -> b h i d", attn, v)
        out = out.reshape(b, self.heads, H, W, -1).permute(0, 1, 4, 2, 3).reshape(b, -1, H, W)
        return self.to_out(out)


class _RDBMUNet(nn.Module):
    """Conditional, time-embedded U-Net: the bridge's denoising network. 2D only."""

    def __init__(self, dim, dim_mults, channels):
        super().__init__()
        self.depth = len(dim_mults)

        self.init_conv = nn.Conv2d(channels * 2, dim, 7, padding=3)
        dims = [dim, *[dim * m for m in dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            _SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= num_resolutions - 1
            self.downs.append(
                nn.ModuleList(
                    [
                        _ResnetBlock(dim_in, dim_in, time_dim),
                        _ResnetBlock(dim_in, dim_in, time_dim),
                        _Residual(_PreNorm(dim_in, _LinearAttention(dim_in))),
                        _downsample_2d(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = _ResnetBlock(mid_dim, mid_dim, time_dim)
        self.mid_attn = _Residual(_PreNorm(mid_dim, _Attention(mid_dim)))
        self.mid_block2 = _ResnetBlock(mid_dim, mid_dim, time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == num_resolutions - 1
            self.ups.append(
                nn.ModuleList(
                    [
                        _ResnetBlock(dim_out + dim_in, dim_out, time_dim),
                        _ResnetBlock(dim_out + dim_in, dim_out, time_dim),
                        _Residual(_PreNorm(dim_out, _LinearAttention(dim_out))),
                        _upsample_2d(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.final_res_block = _ResnetBlock(dim * 2, dim, time_dim)
        self.final_conv = nn.Conv2d(dim, channels, 1)

    def _pad_to_multiple(self, x):
        s = 2**self.depth
        H, W = x.shape[-2:]
        pad_h, pad_w = (s - H % s) % s, (s - W % s) % s
        return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect"), H, W

    def forward(self, x_t, mu, time):
        x = torch.cat((x_t, mu), dim=1)
        x, H, W = self._pad_to_multiple(x)
        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)

        use_ckpt = torch.is_grad_enabled()

        def _run(fn, *args):
            return _grad_checkpoint(fn, *args, use_reentrant=False) if use_ckpt else fn(*args)

        h = []
        for block1, block2, attn, downsample in self.downs:
            def _level(x, t, block1=block1, block2=block2, attn=attn):
                x = block1(x, t)
                s1 = x
                x = block2(x, t)
                x = attn(x)
                return x, s1, x

            x, s1, s2 = _run(_level, x, t)
            h.append(s1)
            h.append(s2)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            skip2, skip1 = h.pop(), h.pop()

            def _level(x, skip2, skip1, t, block1=block1, block2=block2, attn=attn):
                x = torch.cat((x, skip2), dim=1)
                x = block1(x, t)
                x = torch.cat((x, skip1), dim=1)
                x = block2(x, t)
                return attn(x)

            x = _run(_level, x, skip2, skip1, t)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = _run(self.final_res_block, x, t)
        x = self.final_conv(x)
        x = x[..., :H, :W].contiguous()
        return x + mu


class RDBM(nn.Module):
    """Residual Diffusion Bridge Model: a stochastic bridge between input and target images.

    Parameters
    ----------
    image_shape : tuple
        Unused (fully convolutional); kept for BiaPy's model-builder calling convention.
    output_channels : list of int
        Number of image channels to predict; summed to get the bridge's channel count.
    base_dim : int, optional
        Base channel width of the U-Net (``MODEL.RDBM.BASE_DIM``).
    dim_mults : tuple of int, optional
        Channel multiplier per resolution level (``MODEL.RDBM.DIM_MULTS``).
    timesteps : int, optional
        Discretization steps of the forward bridge process (``MODEL.RDBM.TIMESTEPS``).
    sampling_timesteps : int, optional
        Reverse-process steps at validation/test time (``MODEL.RDBM.SAMPLING_TIMESTEPS``); must
        be <= ``timesteps``.
    lamb : float, optional
        Noise scale of the bridge's stochastic term (``MODEL.RDBM.LAMB``).
    """

    def __init__(
        self,
        image_shape=None,
        output_channels=(1,),
        base_dim: int = 64,
        dim_mults=(1, 2, 4, 8),
        timesteps: int = 100,
        sampling_timesteps: int = 10,
        lamb: float = 1e-4,
        **kwargs,
    ):
        super().__init__()
        channels = int(sum(output_channels))
        self.channels = channels
        self.network = _RDBMUNet(dim=base_dim, dim_mults=tuple(dim_mults), channels=channels)

        self.num_timesteps = int(timesteps)
        self.sampling_timesteps = min(int(sampling_timesteps), self.num_timesteps)

        thetas = _betas_for_alpha_bar(self.num_timesteps)
        thetas_cumsum_0_to_t = thetas.cumsum(dim=0)
        thetas_cumsum_0_to_T = thetas_cumsum_0_to_t[-1]
        thetas_cumsum_t_to_T = thetas_cumsum_0_to_T - thetas_cumsum_0_to_t

        sinh_0_to_t = torch.sinh(thetas_cumsum_0_to_t)
        sinh_0_to_T = torch.sinh(thetas_cumsum_0_to_T)
        sinh_t_to_T = torch.sinh(thetas_cumsum_t_to_T)

        # Theta(0)=1, Sigma(0)=0 -> x_0 == x_start; Theta(T)=0, Sigma(T)=0 -> x_T == mu.
        Theta = sinh_t_to_T / sinh_0_to_T
        Sigma2 = 2 * lamb * sinh_0_to_t * sinh_t_to_T / sinh_0_to_T
        Sigma = torch.sqrt(Sigma2)

        self.register_buffer("Theta", Theta.to(torch.float32))
        self.register_buffer("Sigma", Sigma.to(torch.float32))

    def _bounded(self, x):
        return x + (torch.clamp(x, -1.0, 1.0) - x).detach()

    def q_sample(self, x_start, mu, t, noise=None):
        noise = _default(noise, lambda: torch.randn_like(x_start))
        return (
            mu
            + (x_start - mu) * _extract(self.Theta, t, x_start.shape)
            + _extract(self.Sigma, t, x_start.shape) * noise
        )

    def forward_loss(self, mu01, target01, recon_loss_fn):
        """Random-timestep training loss. ``mu01``/``target01`` in [0, 1]; ``recon_loss_fn``:
        ``(pred, target) -> scalar``, e.g. a ``WeightedCompositeLoss`` from ``metrics.py``.
        """
        mu = mu01 * 2 - 1
        x_start = target01 * 2 - 1
        b = x_start.shape[0]
        t = torch.randint(0, self.num_timesteps, (b,), device=x_start.device).long()
        x_t = self.q_sample(x_start, mu, t)
        pred_x_start = self._bounded(self.network(x_t, mu, t))
        return recon_loss_fn((pred_x_start + 1) * 0.5, target01)

    @torch.no_grad()
    def _sample(self, mu):
        """DDIM-style deterministic reverse process, ``sampling_timesteps`` steps, x_T = mu."""
        b = mu.shape[0]
        times = torch.linspace(-1, self.num_timesteps - 1, steps=self.sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        img = mu
        x_start = mu
        for time, time_next in time_pairs:
            t = torch.full((b,), time, device=mu.device, dtype=torch.long)
            x_start = self._bounded(self.network(img, mu, t))
            if time_next < 0:
                img = x_start
                continue

            Theta_now, Theta_next = self.Theta[time], self.Theta[time_next]
            Sigma_now, Sigma_next = self.Sigma[time], self.Sigma[time_next]
            if time == self.num_timesteps - 1:
                img = mu + Theta_next * (x_start - mu)
            else:
                img = (
                    mu
                    + (Sigma_next / Sigma_now) * (img - mu)
                    + (Theta_next - Theta_now * Sigma_next / Sigma_now) * (x_start - mu)
                )
        return img

    def forward(self, inputs):
        """Returns ``{"pred": image in [0, 1], "cond": inputs}``. Training: cheap single-step
        estimate at t=T-1, no grad (the real loss is ``forward_loss``). Eval: full reverse
        sample.
        """
        mu = inputs * 2 - 1
        if self.training:
            with torch.no_grad():
                t = torch.full((mu.shape[0],), self.num_timesteps - 1, device=mu.device, dtype=torch.long)
                pred = self._bounded(self.network(mu, mu, t))
        else:
            pred = self._sample(mu)
        return {"pred": (pred + 1) * 0.5, "cond": inputs}
