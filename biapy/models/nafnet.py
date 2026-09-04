"""NAFNet model (2D/3D) and its PatchGAN discriminator wiring.

Reference: `Simple Baselines for Image Restoration <https://arxiv.org/abs/2204.04676>`_.
Adapted from https://github.com/GolpedeRemo37/NafNet-in-AI4Life-Microscopy-Supervised-Denoising-Challenge
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from biapy.models.blocks import get_activation
from biapy.models.patchgan import PatchGANDiscriminator


def get_level_kernel(k_size: int, is_isotropic: bool, ndim: int) -> tuple:
    """Per-level kernel size, anisotropic (1, k, k) for non-isotropic 3D levels."""
    if ndim == 2:
        return (k_size, k_size)
    return (k_size, k_size, k_size) if is_isotropic else (1, k_size, k_size)


class SimpleGate(nn.Module):
    """Splits channels in half and multiplies both halves element-wise."""

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class PixelShuffle3D(nn.Module):
    """3D pixel shuffle: (B, C*r^3, D, H, W) -> (B, C, D*r, H*r, W*r)."""

    def __init__(self, upscale_factor=2):
        super().__init__()
        self.r = upscale_factor

    def forward(self, x):
        B, C_in, D, H, W = x.size()
        C_out = C_in // (self.r**3)

        x = x.view(B, C_out, self.r, self.r, self.r, D, H, W)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)
        x = x.reshape(B, C_out, D * self.r, H * self.r, W * self.r)

        return x


class SlicePixelShuffle2D(nn.Module):
    """Applies 2D PixelShuffle independently to each Z slice of a (B, C, D, H, W) tensor."""

    def __init__(self, upscale_factor=2):
        super().__init__()
        self.ps = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
        x = self.ps(x)
        _, C_out, H_out, W_out = x.shape
        return x.view(B, D, C_out, H_out, W_out).permute(0, 2, 1, 3, 4)


class LayerNorm2d(nn.Module):
    """Per-position layer normalization over the channel dimension, 2D."""

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + self.eps).sqrt()
        y = self.weight.view(1, C, 1, 1) * y + self.bias.view(1, C, 1, 1)
        return y


class LayerNorm3d(nn.Module):
    """Per-position layer normalization over the channel dimension, 3D."""

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm3d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + self.eps).sqrt()
        y = self.weight.view(1, -1, 1, 1, 1) * y + self.bias.view(1, -1, 1, 1, 1)
        return y


class NAFBlock(nn.Module):
    """NAF residual block: norm -> depthwise conv -> SimpleGate -> channel attention -> FFN."""

    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., ndim=2, k_size=3):
        super().__init__()
        conv = nn.Conv3d if ndim == 3 else nn.Conv2d
        pool = nn.AdaptiveAvgPool3d if ndim == 3 else nn.AdaptiveAvgPool2d

        if isinstance(k_size, (list, tuple)):
            kernel_size = tuple(k_size)[-ndim:]
        else:
            kernel_size = (k_size,) * ndim
        padding = tuple(k // 2 for k in kernel_size)

        dw_channel = c * DW_Expand
        self.conv1 = conv(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = conv(in_channels=dw_channel, out_channels=dw_channel, kernel_size=kernel_size, padding=padding, stride=1, groups=dw_channel, bias=True)
        self.conv3 = conv(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.sca = nn.Sequential(
            pool(1),
            conv(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
        )

        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = conv(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = conv(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        norm_layer = LayerNorm3d if ndim == 3 else LayerNorm2d
        self.norm1 = norm_layer(c)
        self.norm2 = norm_layer(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        shape = [1, c] + [1] * ndim
        self.beta = nn.Parameter(torch.zeros(shape), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros(shape), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

class NAFNet(nn.Module):
    """NAFNet encoder-decoder (intro/ending convs, encoder/bottleneck/decoder, PixelShuffle upsampling).

    Supports 2D and 3D via the same ``MODEL.FEATURE_MAPS`` / ``MODEL.DROPOUT_VALUES`` /
    ``MODEL.KERNEL_SIZE`` / ``MODEL.Z_DOWN`` / ``MODEL.YX_DOWN`` / ``MODEL.ISOTROPY`` /
    ``MODEL.LARGER_IO`` conventions used by the other U-Net family models.
    """

    def __init__(
        self,
        img_channel=3,
        ndim=2,
        feature_maps=[16, 32, 64, 128, 256],
        drop_values=[0.0, 0.0, 0.0, 0.0, 0.0],
        k_size=3,
        z_down=[2, 2, 2, 2],
        yx_down=[2, 2, 2, 2],
        isotropy=True,
        larger_io=False,
        middle_blk_num=1,
        enc_blk_nums=[],
        dec_blk_nums=[],
        dw_expand=2,
        ffn_expand=2,
        discriminator_arch=None,
        patchgan_base_filters=64,
        out_channels=None,
        head_activations: Optional[List[str]] = None,
    ):
        super().__init__()
        self.ndim = ndim
        self.depth = len(feature_maps) - 1

        if out_channels is None:
            out_channels = img_channel
        act_name = (head_activations[0] if head_activations else "linear").lower().removeprefix("ce_")
        self.output_activation = get_activation(act_name)

        if isinstance(isotropy, bool):
            isotropy = [isotropy] * len(feature_maps)

        conv = nn.Conv3d if ndim == 3 else nn.Conv2d

        # Padding target: product of the per-level downsampling factors, so the input
        # divides evenly through every encoder stage.
        if ndim == 3:
            pad_d = 1
            pad_h = 1
            for i in range(self.depth):
                pad_d *= z_down[i]
                pad_h *= yx_down[i]
            self.padder_size = (pad_d, pad_h, pad_h)
        else:
            pad_h = 1
            for i in range(self.depth):
                pad_h *= yx_down[i]
            self.padder_size = (pad_h, pad_h)

        io_k_size = k_size + 2 if larger_io else k_size
        intro_k = get_level_kernel(io_k_size, isotropy[0], ndim)
        intro_p = tuple(k // 2 for k in intro_k)

        self.intro = conv(in_channels=img_channel, out_channels=feature_maps[0], kernel_size=intro_k, padding=intro_p, stride=1, groups=1, bias=True)
        self.ending = conv(in_channels=feature_maps[0], out_channels=out_channels, kernel_size=intro_k, padding=intro_p, stride=1, groups=1, bias=True)
        # Learned projection for the residual skip when in/out channel counts differ
        self.skip_proj = (
            conv(img_channel, out_channels, kernel_size=1, bias=False)
            if out_channels != img_channel else None
        )

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        for i in range(self.depth):
            in_c = feature_maps[i]
            out_c = feature_maps[i + 1]
            level_k = get_level_kernel(k_size, isotropy[i], ndim)

            self.encoders.append(
                nn.Sequential(*[
                    NAFBlock(in_c, DW_Expand=dw_expand, FFN_Expand=ffn_expand,
                             drop_out_rate=drop_values[i], ndim=ndim, k_size=level_k)
                    for _ in range(enc_blk_nums[i])
                ])
            )

            dz = z_down[i] if ndim == 3 else 1
            dyx = yx_down[i]
            stride_tuple = (dz, dyx, dyx) if ndim == 3 else (dyx, dyx)
            self.downs.append(conv(in_c, out_c, kernel_size=stride_tuple, stride=stride_tuple))

        bot_k = get_level_kernel(k_size, isotropy[-1], ndim)
        self.middle_blks = nn.Sequential(
            *[NAFBlock(feature_maps[-1], DW_Expand=dw_expand, FFN_Expand=ffn_expand,
                       drop_out_rate=drop_values[-1], ndim=ndim, k_size=bot_k) for _ in range(middle_blk_num)]
        )

        for i in range(self.depth - 1, -1, -1):
            in_c = feature_maps[i + 1]
            out_c = feature_maps[i]
            level_k = get_level_kernel(k_size, isotropy[i], ndim)
            dz = z_down[i] if ndim == 3 else 1
            dyx = yx_down[i]

            if ndim == 3:
                # dz == 1: upsample Y/X only, one 2D PixelShuffle per Z slice; otherwise
                # upsample all three axes with the native 3D shuffle.
                scale = (dyx**2) if dz == 1 else (dz * dyx * dyx)
                shuffle_layer = SlicePixelShuffle2D(dyx) if dz == 1 else PixelShuffle3D(dz)
                self.ups.append(nn.Sequential(
                    conv(in_c, out_c * scale, kernel_size=1, bias=False),
                    shuffle_layer,
                ))
            else:
                self.ups.append(nn.Sequential(
                    conv(in_c, out_c * (dyx**2), kernel_size=1, bias=False),
                    nn.PixelShuffle(dyx),
                ))

            self.decoders.append(
                nn.Sequential(*[
                    NAFBlock(out_c, DW_Expand=dw_expand, FFN_Expand=ffn_expand,
                             drop_out_rate=drop_values[i], ndim=ndim, k_size=level_k)
                    for _ in range(dec_blk_nums[i])
                ])
            )

        discriminator = None
        if discriminator_arch == "patchgan":
            discriminator = PatchGANDiscriminator(
                in_channels=out_channels,
                ndim=ndim,
                base_filters=patchgan_base_filters,
            )

        self.discriminator = discriminator

    @property
    def param_groups(self):
        """``[generator_params, discriminator_params]`` when a discriminator is present, else a single group."""
        if self.discriminator is not None:
            gen_params = [p for n, p in self.named_parameters() if not n.startswith("discriminator.")]
            return [gen_params, list(self.discriminator.parameters())]
        return [list(self.parameters())]

    def forward(self, inp):
        """Returns the restored tensor, or ``{"pred": tensor}`` when a discriminator is active."""
        if self.ndim == 3:
            B, C, D, H, W = inp.shape
        else:
            B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        skip = self.skip_proj(inp) if self.skip_proj is not None else inp
        x = x + skip

        if self.ndim == 3:
            x = x[:, :, :D, :H, :W]
        else:
            x = x[:, :, :H, :W]
        pred = self.output_activation(x)
        if self.discriminator is not None and isinstance(self.output_activation, nn.Identity):
            # Straight-through logit clamp + sigmoid, only when no explicit head activation was
            # given: bounds output to (0, 1) while keeping gradient nonzero at extreme saturation.
            logit = pred + (torch.clamp(pred, -15.0, 15.0) - pred).detach()
            pred = torch.sigmoid(logit)

        if self.discriminator is not None:
            return {"pred": pred}
        return pred

    def forward_loss(self, pred, targets, loss_fn):
        """Returns ``(loss_generator, loss_discriminator)``, or ``None`` without a discriminator."""
        if self.discriminator is None:
            return None

        # Defense-in-depth straight-through clamp in case forward() didn't already bound pred.
        fake_img = pred + (torch.clamp(pred, 0, 1) - pred).detach()

        for p in self.discriminator.parameters():
            p.requires_grad_(False)
        d_fake_for_g = self.discriminator(fake_img)
        loss_g = loss_fn.forward_generator(fake_img, targets, d_fake_for_g, last_layer=self.last_layer())
        for p in self.discriminator.parameters():
            p.requires_grad_(True)

        # Grad-tracking leaf so loss_fn can compute the R1 penalty (d(d_real)/d(real_img)) when
        # LOSS.GAN.R1_GAMMA > 0; a no-op otherwise (forward_discriminator skips it when disabled).
        real_img = targets.detach().requires_grad_(True)
        d_real = self.discriminator(real_img)
        d_fake = self.discriminator(fake_img.detach())
        loss_d = loss_fn.forward_discriminator(d_real, d_fake, real_images=real_img)

        return (loss_g, loss_d)

    def check_image_size(self, x):
        """Pads spatial dims up to a multiple of ``padder_size``, so downsampling divides evenly."""
        if self.ndim == 3:
            _, _, d, h, w = x.size()
            pad_d, pad_h, pad_w = self.padder_size
            mod_pad_d = (pad_d - d % pad_d) % pad_d
            mod_pad_h = (pad_h - h % pad_h) % pad_h
            mod_pad_w = (pad_w - w % pad_w) % pad_w
            return F.pad(x, (0, mod_pad_w, 0, mod_pad_h, 0, mod_pad_d))
        else:
            _, _, h, w = x.size()
            pad_h, pad_w = self.padder_size
            mod_pad_h = (pad_h - h % pad_h) % pad_h
            mod_pad_w = (pad_w - w % pad_w) % pad_w
            return F.pad(x, (0, mod_pad_w, 0, mod_pad_h))

    def last_layer(self) -> torch.nn.Parameter:
        """Weight of the final conv (pre output-activation), for VQGAN-style adaptive GAN weighting."""
        return self.ending.weight
