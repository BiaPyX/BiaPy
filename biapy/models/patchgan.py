import torch.nn as nn

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=1, base_filters=64):
        super(PatchGANDiscriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, base_filters, normalization=False),
            *discriminator_block(base_filters, base_filters * 2),
            *discriminator_block(base_filters * 2, base_filters * 4),
            *discriminator_block(base_filters * 4, base_filters * 8),
            nn.Conv2d(base_filters * 8, 1, 4, stride=1, padding=1)  
        )

    def forward(self, img):
        return self.model(img)