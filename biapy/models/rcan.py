from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, num_features, reduction, ndim=2):
        super(ChannelAttention, self).__init__()
        if ndim == 2:
            conv = nn.Conv2d
            avg_pool = nn.AdaptiveAvgPool2d
        else:
            conv = nn.Conv3d
            avg_pool = nn.AdaptiveAvgPool3d
        self.module = nn.Sequential(
            avg_pool(1),
            conv(num_features, num_features // reduction, kernel_size=1),
            nn.SiLU(inplace=True),
            conv(num_features // reduction, num_features, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.module(x)


class RCAB(nn.Module):
    def __init__(self, num_features, reduction, ndim=2):
        super(RCAB, self).__init__()
        if ndim == 2:
            conv = nn.Conv2d
        else:
            conv = nn.Conv3d
            
        self.module = nn.Sequential(
            conv(num_features, num_features, kernel_size=3, padding="same"),
            nn.SiLU(inplace=True),
            conv(num_features, num_features, kernel_size=3, padding="same"),
            ChannelAttention(num_features, reduction, ndim=ndim),
        )

    def forward(self, x):
        return x + self.module(x)


class RG(nn.Module):
    def __init__(self, num_features, num_rcab, reduction, ndim=2):
        super(RG, self).__init__()
        if ndim == 2:
            conv = nn.Conv2d
        else:
            conv = nn.Conv3d
        self.module = [RCAB(num_features, reduction, ndim=ndim) for _ in range(num_rcab)]
        self.module.append(conv(num_features, num_features, kernel_size=3, padding="same"))
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
        num_rg=10,
        num_rcab=20,
        reduction=16,
        upscaling_layer=True,
    ):
        super(rcan, self).__init__()
        if type(scale) is tuple:
            scale = scale[0]
        self.ndim = ndim
        self.upscaling_layer = upscaling_layer
        if ndim == 2:
            conv = nn.Conv2d
        else:
            conv = nn.Conv3d
        self.sf = conv(num_channels, filters, kernel_size=3, padding="same")
        self.rgs = nn.Sequential(*[RG(filters, num_rcab, reduction, ndim=ndim) for _ in range(num_rg)])
        self.conv1 = conv(filters, filters, kernel_size=3, padding="same")
        if upscaling_layer:
            self.upscale = nn.Sequential(
                conv(filters, filters * (scale**2), kernel_size=3, padding="same"),
                nn.PixelShuffle(scale),
            )
        self.conv2 = conv(filters, num_channels, kernel_size=3, padding="same")

    def forward(self, x) -> dict:
        x = self.sf(x)
        residual = x
        x = self.rgs(x)
        x = self.conv1(x)
        x += residual
        if self.upscaling_layer:
            x = self.upscale(x)
        x = self.conv2(x)
        return {"pred": x}
