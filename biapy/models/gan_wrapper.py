"""Generic GAN wrapper: attaches NAFNet's PatchGAN discriminator to any generator module."""

from typing import Optional

import torch
import torch.nn as nn

from biapy.models.patchgan import PatchGANDiscriminator


class GANGeneratorWrapper(nn.Module):
    """Wraps a generator module (plain tensor or dict with a ``"pred"`` key) with an optional PatchGAN discriminator."""

    def __init__(
        self,
        generator: nn.Module,
        discriminator_arch: Optional[str] = None,
        ndim: int = 2,
        patchgan_base_filters: int = 64,
        out_channels: int = 1,
    ):
        super().__init__()
        self.generator = generator

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
        # Straight-through clamp the logit to [-15,15] then sigmoid: bounds output and keeps
        # gradient nonzero even at extreme saturation (plain sigmoid grad underflows to 0 past ~20).
        logit = pred + (torch.clamp(pred, -15.0, 15.0) - pred).detach()
        pred = torch.sigmoid(logit)

        if self.discriminator is not None:
            return {"pred": pred}
        return pred

    def last_layer(self) -> torch.nn.Parameter:
        """Weight of the wrapped generator's final conv, for VQGAN-style adaptive GAN weighting."""
        return self.generator.last_layer()

    def forward_loss(self, pred, targets, loss_fn):
        """Compute ``(loss_generator, loss_discriminator)`` via the discriminator and ``loss_fn``."""
        if self.discriminator is None:
            return None

        # `pred` is already bounded to (0, 1) by forward()'s sigmoid.
        fake_img = pred

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
