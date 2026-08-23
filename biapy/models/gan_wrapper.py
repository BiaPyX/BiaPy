"""Generic GAN wrapper: attaches NAFNet's PatchGAN discriminator to any generator module."""

from typing import Optional

import torch
import torch.nn as nn

from biapy.models.patchgan import PatchGANDiscriminator


class GANGeneratorWrapper(nn.Module):
    """Wrap a generator module with an optional PatchGAN discriminator.

    Parameters
    ----------
    generator : torch.nn.Module
        Backbone network. May return a plain tensor or a dict with a ``"pred"`` key.

    discriminator_arch : str, optional
        Only ``"patchgan"`` is supported; leave ``None`` to disable adversarial training.

    patchgan_base_filters : int, optional
        Number of filters in the first PatchGAN discriminator block.

    out_channels : int, optional
        Number of channels the generator outputs.
    """

    def __init__(
        self,
        generator: nn.Module,
        discriminator_arch: Optional[str] = None,
        patchgan_base_filters: int = 64,
        out_channels: int = 1,
    ):
        super().__init__()
        self.generator = generator

        discriminator = None
        if discriminator_arch == "patchgan":
            discriminator = PatchGANDiscriminator(
                in_channels=out_channels,
                base_filters=patchgan_base_filters,
            )
        self.discriminator = discriminator

    @property
    def param_groups(self):
        """``[generator_params, discriminator_params]``, or a single group without a discriminator."""
        if self.discriminator is not None:
            gen_params = [p for n, p in self.named_parameters() if not n.startswith("discriminator.")]
            return [gen_params, list(self.discriminator.parameters())]
        return [list(self.parameters())]

    def forward(self, inp):
        """Return ``{"pred": tensor}`` when a discriminator is active, else the plain tensor."""
        pred = self.generator(inp)
        if isinstance(pred, dict):
            pred = pred["pred"]

        if self.discriminator is not None:
            return {"pred": pred}
        return pred

    def forward_loss(self, pred, targets, loss_fn):
        """Compute ``(loss_generator, loss_discriminator)`` via the discriminator and ``loss_fn``."""
        if self.discriminator is None:
            return None

        fake_img = torch.clamp(pred, 0, 1)

        for p in self.discriminator.parameters():
            p.requires_grad_(False)
        d_fake_for_g = self.discriminator(fake_img)
        loss_g = loss_fn.forward_generator(fake_img, targets, d_fake_for_g)
        for p in self.discriminator.parameters():
            p.requires_grad_(True)

        d_real = self.discriminator(targets)
        d_fake = self.discriminator(fake_img.detach())
        loss_d = loss_fn.forward_discriminator(d_real, d_fake)

        return (loss_g, loss_d)
