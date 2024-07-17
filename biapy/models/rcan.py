from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, num_features, reduction):
        super(ChannelAttention, self).__init__()
        self.module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, num_features // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // reduction, num_features, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.module(x)


class RCAB(nn.Module):
    def __init__(self, num_features, reduction):
        super(RCAB, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding="same"),
            ChannelAttention(num_features, reduction),
        )

    def forward(self, x):
        return x + self.module(x)


class RG(nn.Module):
    def __init__(self, num_features, num_rcab, reduction):
        super(RG, self).__init__()
        self.module = [RCAB(num_features, reduction) for _ in range(num_rcab)]
        self.module.append(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding="same")
        )
        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return x + self.module(x)


class rcan(nn.Module):
    """
    Deep residual channel attention networks (RCAN) model.

    Reference: `Image Super-Resolution Using Very Deep Residual Channel Attention Networks
    <https://openaccess.thecvf.com/content_ECCV_2018/html/Yulun_Zhang_Image_Super-Resolution_Using_ECCV_2018_paper.html>`_.

    Adapted from `here <https://github.com/yjn870/RCAN-pytorch>`_.
    """

    def __init__(
        self,
        ndim,
        num_channels=3,
        filters=64,
        scale=2,
        n_sub_block=2,
        num_rcab=20,
        reduction=16,
    ):
        super(rcan, self).__init__()
        if type(scale) is tuple:
            scale = scale[0]
        self.ndim = ndim
        self.sf = nn.Conv2d(num_channels, filters, kernel_size=3, padding="same")
        self.rgs = nn.Sequential(
            *[RG(filters, num_rcab, reduction) for _ in range(n_sub_block)]
        )
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding="same")
        self.upscale = nn.Sequential(
            nn.Conv2d(filters, filters * (scale**2), kernel_size=3, padding="same"),
            nn.PixelShuffle(scale),
        )
        self.conv2 = nn.Conv2d(filters, num_channels, kernel_size=3, padding="same")

    def forward(self, x):
        x = self.sf(x)
        residual = x
        x = self.rgs(x)
        x = self.conv1(x)
        x += residual
        x = self.upscale(x)
        x = self.conv2(x)
        return x
