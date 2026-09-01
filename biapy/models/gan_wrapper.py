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
        # Bound the output with a real saturating activation, not a hard clamp: the wrapped
        # backbone (e.g. STUNet) has no output activation of its own, so `pred` is otherwise
        # unbounded. A straight-through clamp on the output (constant gradient=1 regardless of
        # distance from [0, 1]) fixes getting permanently stuck outside range, but not a *different*
        # failure: once a pixel saturates, nothing discourages the pre-activation logit from
        # drifting arbitrarily far past the boundary, since the clamped output can't get any "more
        # correct" -- observed as MAE/MSE climbing for many epochs with the loss itself looking
        # healthy. Sigmoid's gradient shrinks as it saturates, giving a genuine brake on that drift
        # -- but plain sigmoid's own gradient underflows to exact float32 zero once the logit's
        # magnitude reaches ~20 (verified), reproducing the same dead zone a hard clamp caused,
        # just requiring a larger excursion to trigger. Straight-through clamp the *logit* (not the
        # final output) to a range where sigmoid's gradient is still meaningfully nonzero (at +/-15,
        # ~3.6e-7) before applying it, so the two mechanisms cover each other: guaranteed nonzero
        # gradient always (from the logit clamp) and a properly bounded, braking output always
        # (from sigmoid).
        logit = pred + (torch.clamp(pred, -15.0, 15.0) - pred).detach()
        pred = torch.sigmoid(logit)

        if self.discriminator is not None:
            return {"pred": pred}
        return pred

    def forward_loss(self, pred, targets, loss_fn):
        """Compute ``(loss_generator, loss_discriminator)`` via the discriminator and ``loss_fn``."""
        if self.discriminator is None:
            return None

        # `pred` is already bounded to (0, 1) by forward()'s sigmoid.
        fake_img = pred

        for p in self.discriminator.parameters():
            p.requires_grad_(False)
        d_fake_for_g = self.discriminator(fake_img)
        loss_g = loss_fn.forward_generator(fake_img, targets, d_fake_for_g)
        for p in self.discriminator.parameters():
            p.requires_grad_(True)

        # Grad-tracking leaf so loss_fn can compute the R1 penalty (d(d_real)/d(real_img)) when
        # LOSS.CYCLEGAN.R1_GAMMA > 0; a no-op otherwise (forward_discriminator skips it when disabled).
        real_img = targets.detach().requires_grad_(True)
        d_real = self.discriminator(real_img)
        d_fake = self.discriminator(fake_img.detach())
        loss_d = loss_fn.forward_discriminator(d_real, d_fake, real_images=real_img)

        return (loss_g, loss_d)
