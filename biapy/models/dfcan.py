# Adapted from https://github.com/L0-zhang/DFCAN-pytorch

import torch
import torch.nn as nn
import torch.fft


def fftshift2d(img, size_psc=128):
    bs, ch, h, w = img.shape
    fs11 = img[:, :, h // 2 :, w // 2 :]
    fs12 = img[:, :, h // 2 :, : w // 2]
    fs21 = img[:, :, : h // 2, w // 2 :]
    fs22 = img[:, :, : h // 2, : w // 2]
    output = torch.cat([torch.cat([fs11, fs21], dim=2), torch.cat([fs12, fs22], dim=2)], dim=3)
    return output


def fftshift3d(img, size_psc=128):
    bs, ch, z, h, w = img.shape
    fs111 = img[:, :, z // 2 :, h // 2 :, w // 2 :]
    fs121 = img[:, :, z // 2 :, h // 2 :, : w // 2]
    fs211 = img[:, :, z // 2 :, : h // 2, w // 2 :]
    fs221 = img[:, :, z // 2 :, : h // 2, : w // 2]
    fs112 = img[:, :, : z // 2, h // 2 :, w // 2 :]
    fs122 = img[:, :, : z // 2, h // 2 :, : w // 2]
    fs212 = img[:, :, : z // 2, : h // 2, w // 2 :]
    fs222 = img[:, :, : z // 2, : h // 2, : w // 2]
    output1 = torch.cat([torch.cat([fs111, fs211], dim=3), torch.cat([fs121, fs221], dim=3)], dim=4)
    output2 = torch.cat([torch.cat([fs112, fs212], dim=3), torch.cat([fs122, fs222], dim=3)], dim=4)
    return torch.cat([output1, output2], dim=2)


class RCAB(nn.Module):
    def __init__(self, size_psc=128, conv=nn.Conv2d):  # size_psc：crop_size input_shape：depth
        super().__init__()
        self.size_psc = size_psc
        self.conv_gelu1 = nn.Sequential(conv(64, 64, kernel_size=3, stride=1, padding="same"), nn.GELU())
        self.conv_gelu2 = nn.Sequential(conv(64, 64, kernel_size=3, stride=1, padding="same"), nn.GELU())
        self.conv_relu1 = nn.Sequential(conv(64, 64, kernel_size=3, stride=1, padding="same"), nn.ReLU())
        self.avg_pool = nn.AdaptiveAvgPool2d(1) if conv == nn.Conv2d else nn.AdaptiveAvgPool3d(1)
        self.conv_relu2 = nn.Sequential(conv(64, 4, kernel_size=1, stride=1, padding=0), nn.ReLU())
        self.conv_sigmoid = nn.Sequential(conv(4, 64, kernel_size=1, stride=1, padding=0), nn.Sigmoid())
        self.fftshiffunc = fftshift2d if conv == nn.Conv2d else fftshift3d

    def forward(self, x, gamma=0.8):
        x0 = x.to(torch.float32)
        x = self.conv_gelu1(x)
        x = self.conv_gelu2(x)
        x1 = x.to(torch.float32)
        x = torch.fft.fftn(x.to(torch.float32), dim=(2, 3))
        x = torch.pow(torch.abs(x) + 1e-8, gamma)  # abs
        x = self.fftshiffunc(x, self.size_psc)
        x = self.conv_relu1(x)
        x = self.avg_pool(x)
        x = self.conv_relu2(x)
        x = self.conv_sigmoid(x)
        x = x1 * x
        x = x0 + x
        return x


class ResGroup(nn.Module):
    def __init__(self, n_RCAB=4, size_psc=128, conv=nn.Conv2d):
        super().__init__()
        RCABs = []
        for _ in range(n_RCAB):
            RCABs.append(RCAB(size_psc, conv=conv))
        self.RCABs = nn.Sequential(*RCABs)

    def forward(self, x):
        x0 = x
        x = self.RCABs(x)
        x = x0 + x
        return x


class DFCAN(nn.Module):
    """
    Fourier channel attention network (DFCAN) for super-resolution.

    References: `Evaluation and development of deep neural networks for image super-resolution in optical
    microscopy <https://www.nature.com/articles/s41592-020-01048-5>`_.
    """

    def __init__(self, ndim, input_shape, scale=2, n_ResGroup=4, n_RCAB=4):
        super().__init__()
        if type(scale) is tuple:
            scale = scale[0]
        self.ndim = ndim
        size_psc = input_shape[0]
        conv = nn.Conv3d if self.ndim == 3 else nn.Conv2d

        self.input = nn.Sequential(
            conv(input_shape[-1], 64, kernel_size=3, stride=1, padding="same"),
            nn.GELU(),
        )
        ResGroups = []
        for _ in range(n_ResGroup):
            ResGroups.append(ResGroup(n_RCAB=n_RCAB, size_psc=size_psc, conv=conv))
        self.RGs = nn.Sequential(*ResGroups)
        self.conv_gelu = nn.Sequential(
            conv(64, 64 * (scale**2), kernel_size=3, stride=1, padding="same"),
            nn.GELU(),
        )
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.conv_sigmoid = nn.Sequential(
            conv(64, input_shape[-1], kernel_size=3, stride=1, padding="same"),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.input(x)
        x = self.RGs(x)
        x = self.conv_gelu(x)
        x = self.pixel_shuffle(x)  # upsampling
        x = self.conv_sigmoid(x)
        return x
