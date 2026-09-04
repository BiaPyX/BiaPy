"""PatchGAN discriminator: convolutional discriminator scoring realism per patch instead of globally."""

import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """Four strided conv+BatchNorm+LeakyReLU blocks, then a final conv to a 1-channel patch-logits map."""

    def __init__(self, in_channels=1, ndim=2, base_filters=64):
        super(PatchGANDiscriminator, self).__init__()
        conv = nn.Conv3d if ndim == 3 else nn.Conv2d
        norm = nn.BatchNorm3d if ndim == 3 else nn.BatchNorm2d
        kernel_size = (4,) * ndim
        down_stride = (2,) * ndim
        unit_stride = (1,) * ndim
        padding = (1,) * ndim

        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [conv(in_filters, out_filters, kernel_size, stride=down_stride, padding=padding)]
            if normalization:
                layers.append(norm(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, base_filters, normalization=False),
            *discriminator_block(base_filters, base_filters * 2),
            *discriminator_block(base_filters * 2, base_filters * 4),
            *discriminator_block(base_filters * 4, base_filters * 8),
            conv(base_filters * 8, 1, kernel_size, stride=unit_stride, padding=padding)
        )

    def forward(self, img):
        return self.model(img)
