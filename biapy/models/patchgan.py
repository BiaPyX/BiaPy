"""PatchGAN discriminator model used in image-to-image GAN training.

This module implements a convolutional discriminator that predicts realism at
the patch level instead of producing a single global score. Patch-level
classification is commonly used in conditional GAN pipelines because it
emphasizes local texture and edge consistency, which is especially useful in
restoration and translation tasks.

Classes
-------
PatchGANDiscriminator
    Multi-layer convolutional discriminator with strided downsampling blocks and
    a final 1-channel logits map.

Notes
-----
The output tensor shape is `(N, 1, H_patch, W_patch)`, where each spatial value
acts as a local real/fake logit for a receptive-field patch in the input image.

Implementation adapted for this project from:
https://github.com/GolpedeRemo37/NafNet-in-AI4Life-Microscopy-Supervised-Denoising-Challenge

"""

import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """PatchGAN discriminator based on strided convolutional blocks.

    Parameters
    ----------
    in_channels : int, optional
        Number of channels in the input image.
    base_filters : int, optional
        Number of filters in the first discriminator block. Each subsequent
        block doubles this value.

    Notes
    -----
    The architecture follows a typical PatchGAN design:
    1. Four convolutional downsampling blocks.
    2. Batch normalization on all blocks except the first one.
    3. LeakyReLU activations.
    4. Final convolution producing a patch-logits map.
    """

    def __init__(self, in_channels=1, base_filters=64):
        super(PatchGANDiscriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, normalization=True):
            """Create one discriminator stage.

            Parameters
            ----------
            in_filters : int
                Number of input channels.
            out_filters : int
                Number of output channels.
            normalization : bool, optional
                Whether to include BatchNorm after convolution.

            Returns
            -------
            list[nn.Module]
                Layers composing one stage of the discriminator.
            """
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
        """Run a forward pass through the discriminator.

        Parameters
        ----------
        img : torch.Tensor
            Input tensor with shape `(N, C, H, W)`.

        Returns
        -------
        torch.Tensor
            Patch-wise realism logits with shape `(N, 1, H_patch, W_patch)`.
        """
        return self.model(img)