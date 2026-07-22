"""
EmbedSeg post-processing utilities for BiaPy.

This module gathers the EmbedSeg-specific inference code (spatial-embedding clustering) used by the
``E_offset``/``E_sigma``/``E_seediness`` instance-segmentation channels, kept separate from the
generic ``post_processing`` module.

Reference: `EmbedSeg: Embedding-based Instance Segmentation for Biomedical Microscopy Data
<https://www.sciencedirect.com/science/article/pii/S1361841522001700>`_.

Code adapted from: `EmbedSeg <https://github.com/juglab/EmbedSeg>`_.
"""
import math
import torch
import numpy as np
from typing import List, Tuple, Callable
from numpy.typing import NDArray

from biapy.utils.misc import to_numpy_format, to_pytorch_format


class Embedding_cluster:
    def __init__(
        self,
        device: torch.device,
        anisotropy: List[float | int] = [1,1,1],
        ndims: int = 2,
        grid_size: int = 1024,
    ):
        """
        Embedding-based clustering for instance segmentation inspired by the EmbedSeg method.

        Reference: `EmbedSeg: Embedding-based Instance Segmentation for Biomedical Microscopy Data <https://www.sciencedirect.com/science/article/pii/S1361841522001700>`_.

        Code adapted from: `Embedseg <https://github.com/juglab/EmbedSeg>`_.

        Parameters
        ----------
        device : torch.device
            Device to run the computations on.

        anisotropy : List of float/int, optional
            Voxel spacing ``(z, y, x)``; only the ratios matter (z carries the anisotropy).

        ndims : int, optional
            Number of dimensions of the input data. 2 for 2D images, 3 for 3D volumes.

        grid_size : int, optional
            Size of the canonical coordinate grid (per-pixel coordinate step ``1 / (grid_size - 1)``).
            MUST match the value used by the training loss (``SpatialEmbLoss``) so inference and
            training share the same coordinate scale.
        """
        self.device = device
        self.ndims = ndims

        # Canonical coordinate grid, kept identical to SpatialEmbLoss: fixed per-pixel step
        # ``ratio_axis / (grid_size - 1)`` (independent of patch/image size), with z carrying the
        # voxel anisotropy. This is the original EmbedSeg convention (grid = dataset max image size,
        # pixel = 1) and is what makes the learned offsets/sigmas mean the same thing at train & test.
        self.grid_size = int(grid_size)
        self.norm = float(self.grid_size - 1)
        pixel_z = anisotropy[0] if ndims == 3 else 1
        pixel_y = anisotropy[1]
        pixel_x = anisotropy[2]
        self.ratio_x = float(pixel_x) / float(pixel_y)
        self.ratio_y = 1.0
        self.ratio_z = float(pixel_z) / float(pixel_y)

    def _build_coords(self, Z: int, H: int, W: int) -> "torch.Tensor":
        """
        Build the coordinate map for the exact input size (no large pre-allocated grid).

        Coordinate at index ``i`` along an axis is ``i * ratio_axis / self.norm`` (per-pixel step
        ``ratio_axis / (grid_size - 1)``), so it is constant for any image size and matches the
        training loss (``SpatialEmbLoss``) exactly -- images larger than ``grid_size`` keep the same
        scale instead of being rescaled.
        """
        ym = (torch.arange(H, device=self.device, dtype=torch.float32) * (self.ratio_y / self.norm)).view(1, -1, 1)
        xm = (torch.arange(W, device=self.device, dtype=torch.float32) * (self.ratio_x / self.norm)).view(1, 1, -1)
        if self.ndims == 2:
            return torch.cat((xm.expand(1, H, W), ym.expand(1, H, W)), 0)  # (2, H, W) as (x, y)
        zm = (torch.arange(Z, device=self.device, dtype=torch.float32) * (self.ratio_z / self.norm)).view(-1, 1, 1)
        return torch.stack((xm.expand(Z, H, W), ym.expand(Z, H, W), zm.expand(Z, H, W)), 0)  # (3, Z, H, W)

    def create_instances(
        self,
        pred: NDArray,
        fg_thresh: float = 0.5,
        seed_thresh: float = 0.9,
        min_mask_sum: int = 0,
        min_unclustered_sum: int = 0,
        min_object_size: int = 0,
    ) -> NDArray:
        """
        Create instances from predicted offsets, per-axis sigmas, and seediness using the
        EmbedSeg method (faithful port of ``EmbedSeg.utils.utils.Cluster.cluster``, the
        ``cluster_fast=True`` path that the original uses for its reported results).

        Parameters
        ----------
        pred : NDArray
            Model predictions with shape:
            - 2D: (Y, X, C=5)  with channels [off_y, off_x, sig_y, sig_x, seed]
            - 3D: (Z, Y, X, C=7) with channels [off_z, off_y, off_x, sig_z, sig_y, sig_x, seed]

        fg_thresh : float, optional
            Seediness threshold defining the foreground pixels that get clustered into objects.

        seed_thresh : float, optional
            Seediness threshold a pixel must reach to seed a new object; clustering stops once no
            remaining foreground pixel exceeds it.

        min_mask_sum : int, optional
            Minimum number of foreground pixels required to perform clustering.

        min_unclustered_sum : int, optional
            Minimum number of unclustered foreground pixels to continue clustering.

        min_object_size : int, optional
            Minimum size (in pixels) a proposal must have to be kept. Defaults to 0 (keep all): small
            objects are instead removed by the ``TEST.POST_PROCESSING.INSTANCE_REFINEMENT``
            (``remove_small_objects``) step, so it is not exposed as a config option.

        Returns
        -------
        NDArray
            Instance labels with shape ``(*spatial,)``, dtype uint8 if max(label) ≤ 255 else uint16.
        """
        assert pred.ndim in (3, 4), f"Expected (Y,X,C) or (Z,Y,X,C); got {pred.shape}"
        spatial = pred.shape[:-1]
        D = len(spatial)
        C = pred.shape[-1]
        expected_C = 2 * D + 1
        assert C == expected_C, f"Expected {expected_C} channels (offset(D)+sigma(D)+seed); got {C}"

        pred = torch.from_numpy(np.moveaxis(pred, -1, 0)).to(self.device)  # (C, *spatial)

        # Build the coordinate map for this image at the fixed per-pixel step (constant train/test).
        if D == 2:
            H, W = pred.shape[1], pred.shape[2]
            coords = self._build_coords(1, H, W).contiguous()
        else:
            Z, H, W = pred.shape[1], pred.shape[2], pred.shape[3]
            coords = self._build_coords(Z, H, W).contiguous()

        # offset (tanh) and seed (sigmoid) activations are already applied by the model head; exp(sigma*10)
        # further below is part of the distance, not an activation.
        offsets  = pred[:D]      # (D, *spatial)
        sigma  = pred[D:2*D]     # (D, *spatial)  per-axis σ
        seed_map = pred[2*D]     # (*spatial)

        # embeddings e(x) = x + o(x)
        spatial_emb = offsets + coords  # (D, *spatial)

        count = 1
        labels = np.zeros(spatial, dtype=np.int64)
        mask_fg = seed_map > fg_thresh
        if mask_fg.sum() > min_mask_sum:
            # Every foreground pixel is a seed candidate; the highest-seediness unclustered pixel
            # centres the next object, and clustering stops once that peak drops below seed_thresh.
            spatial_emb_masked = spatial_emb[mask_fg.expand_as(spatial_emb)].view(D, -1)
            sigma_masked = sigma[mask_fg.expand_as(sigma)].view(D, -1)
            seed_map_masked = seed_map[mask_fg].view(1, -1)

            unclustered = torch.ones(mask_fg.sum()).short().to(self.device)
            labels_masked = torch.zeros(mask_fg.sum()).short().to(self.device)
            while unclustered.sum() > min_unclustered_sum:
                scores = seed_map_masked * unclustered.float()
                seed = scores.argmax().item()
                if scores.max().item() < seed_thresh:
                    break
                center = spatial_emb_masked[:, seed : seed + 1]
                unclustered[seed] = 0
                s = torch.exp(sigma_masked[:, seed : seed + 1] * 10)
                dist = torch.exp(
                    -1 * torch.sum(torch.pow(spatial_emb_masked - center, 2) * s, 0)
                )
                proposal = (dist > 0.5).squeeze()
                if proposal.sum() > min_object_size:
                    if (
                        unclustered[proposal].sum().float() / proposal.sum().float()
                        > 0.5
                    ):
                        labels_masked[proposal.squeeze()] = count
                        count += 1
                # Mark every pixel of the proposal clustered, even if the object was rejected, so it
                # cannot re-seed on the next iteration (matches the original).
                unclustered[proposal] = 0
            labels[mask_fg.squeeze().cpu()] = labels_masked.cpu()
        return labels


def ensemble_embedseg_predictions(
    o_img: NDArray,
    pred_func: Callable,
    axes_order_back: Tuple[int, ...],
    axes_order: Tuple[int, ...],
    device: torch.device,
    ndim: int = 2,
    batch_size_value: int = 1,
    mode: str = "mean",
) -> "torch.Tensor":
    """
    EmbedSeg-specific test-time augmentation for 2D/3D spatial-embedding predictions.

    Faithful port of ``EmbedSeg.utils.test_time_augmentation.apply_tta_2d`` / ``apply_tta_3d``: the
    image is predicted in every orientation of the XY-plane dihedral group crossed with the axis flips
    -- ``4 rotations × Y-flip`` in 2D (8 orientations) and ``4 rotations × Y-flip × Z-flip`` in 3D
    (16 orientations). Each prediction is spatially un-transformed **and** its vector channels are
    remapped back to the canonical frame, then all orientations are reduced with ``mode``.

    Unlike the generic :func:`ensemble8_2d_predictions`, the offset and sigma channels are not scalar
    fields: an XY-plane rotation rotates the offset vector ``(off_x, off_y)`` by the rotation matrix
    ``[cosθ sinθ; -sinθ cosθ]`` (channel swap + sign changes) and swaps the per-axis sigmas
    ``(sig_x, sig_y)``; a Y-flip negates ``off_y`` and a Z-flip negates ``off_z``. ``off_z``/``sig_z``
    are untouched by the XY rotation (rotations are only in-plane, as in the original). The physical
    channel order is ``[off_x, off_y, (off_z,) sig_x, sig_y, (sig_z,) seed]`` -- identical between the
    training loss and clustering (both build coordinates as ``cat((xm, ym(, zm)))``), which is what
    makes the remap valid.

    Parameters
    ----------
    o_img : Numpy array
        Input image ``(y, x, channels)`` (2D) or ``(z, y, x, channels)`` (3D) -- the raw intensity
        channels fed to the model.

    pred_func : function
        Function to make predictions.

    axes_order_back : tuple
        Axis order to convert from tensor to numpy. E.g. ``(0, 3, 1, 2)`` (2D) or ``(0, 4, 1, 2, 3)`` (3D).

    axes_order : tuple
        Axis order to convert from numpy to tensor.

    device : Torch device
        Device used.

    ndim : int
        Number of spatial dimensions of ``o_img`` (2 or 3).

    batch_size_value : int, optional
        Batch size value.

    mode : str, optional
        Ensemble mode. Possible options: "mean", "min", "max" ("mean" is the original behaviour).

    Returns
    -------
    out : torch.Tensor
        Model output with the ``pred`` key assembled (matching :func:`ensemble8_2d_predictions`).
    """
    assert mode in ["mean", "min", "max"], "Get unknown ensemble mode {}".format(mode)
    assert ndim in (2, 3), "ndim must be 2 or 3, got {}".format(ndim)
    assert o_img.ndim == ndim + 1, "Expected {}D input (spatial..., channels); got {}".format(ndim, o_img.shape)

    # Per-sample spatial axes in the (spatial..., ch) layout: rotations act in the Y-X plane, Y/Z are
    # the flip axes (Z only in 3D). Channel roles follow the coord order [x, y(, z)] used everywhere.
    if ndim == 2:
        y_ax, x_ax, z_ax = 0, 1, None
        off_x, off_y, off_z = 0, 1, None
        sig_x, sig_y = 2, 3
    else:
        z_ax, y_ax, x_ax = 0, 1, 2
        off_x, off_y, off_z = 0, 1, 2
        sig_x, sig_y = 3, 4
    yx_fwd = (y_ax, x_ax)
    yx_inv = (x_ax, y_ax)  # reversed axes => inverse rotation

    # Reflect-pad the Y-X plane to a square so the 90° rotations keep a single stackable shape (same
    # trick as ensemble8_2d_predictions); Z is left untouched. Cropped off the predictions at the end.
    pad_to_square = o_img.shape[y_ax] - o_img.shape[x_ax]
    pad_w = [(0, 0)] * o_img.ndim
    if pad_to_square < 0:
        pad_w[y_ax] = (abs(pad_to_square), 0)
    elif pad_to_square > 0:
        pad_w[x_ax] = (pad_to_square, 0)
    img = np.pad(o_img, pad_w, "reflect")

    # Forward augmentations: rot90(k) in the Y-X plane, then optional Z- and Y-flips.
    fz_opts = [0, 1] if ndim == 3 else [0]
    combos = [(k, fy, fz) for fz in fz_opts for fy in (0, 1) for k in range(4)]

    def _augment(a, k, fy, fz):
        a = np.rot90(a, k, yx_fwd)
        if fz:
            a = np.flip(a, z_ax)
        if fy:
            a = np.flip(a, y_ax)
        return np.ascontiguousarray(a)

    aug_img = np.stack([_augment(img, *c) for c in combos], axis=0)  # (N, spatial..., ch)

    # Predict (batched, like ensemble8_2d_predictions).
    preds = []
    l = int(math.ceil(aug_img.shape[0] / batch_size_value))
    for i in range(l):
        top = min((i + 1) * batch_size_value, aug_img.shape[0])
        r_aux = pred_func(aug_img[i * batch_size_value : top])
        if isinstance(r_aux, dict):
            r_aux = r_aux["pred"]
        preds.append(to_numpy_format(r_aux, axes_order_back))
    pred = np.concatenate(preds, axis=0).astype(np.float32)  # (N, spatial..., C)
    C = pred.shape[-1]
    exp_C = 2 * ndim + 1
    assert C == exp_C, "EmbedSeg {}D expects {} output channels; got {}".format(ndim, exp_C, C)

    def _rotate_channels(p, k):
        """Rotate the offset vector (off_x, off_y) by k*90° and swap (sig_x, sig_y); rest untouched."""
        out = p.copy()
        if k == 1:
            out[..., off_x], out[..., off_y] = -p[..., off_y], p[..., off_x]
            out[..., sig_x], out[..., sig_y] = p[..., sig_y], p[..., sig_x]
        elif k == 2:
            out[..., off_x], out[..., off_y] = -p[..., off_x], -p[..., off_y]
        elif k == 3:
            out[..., off_x], out[..., off_y] = p[..., off_y], -p[..., off_x]
            out[..., sig_x], out[..., sig_y] = p[..., sig_y], p[..., sig_x]
        return out

    def _restore(p, k, fy, fz):
        # Un-transform spatially (reverse order: un-flip Y, un-flip Z, then un-rotate).
        if fy:
            p = np.flip(p, y_ax)
        if fz:
            p = np.flip(p, z_ax)
        p = np.rot90(p, k, yx_inv).copy()
        # Channel corrections: flip negations first (off_y / off_z), then the rotation remap.
        if fy:
            p[..., off_y] *= -1.0
        if fz and off_z is not None:
            p[..., off_z] *= -1.0
        if k:
            p = _rotate_channels(p, k)
        return p

    corrected = np.stack([_restore(pred[n], *c) for n, c in enumerate(combos)], axis=0)

    funct = {"mean": np.mean, "min": np.min, "max": np.max}[mode]
    out = funct(corrected, axis=0)  # (spatial..., C)

    # Crop the square padding back off (inverse of the reflect-pad above).
    if pad_to_square < 0:
        sl = [slice(None)] * out.ndim
        sl[y_ax] = slice(abs(pad_to_square), None)
        out = out[tuple(sl)]
    elif pad_to_square > 0:
        sl = [slice(None)] * out.ndim
        sl[x_ax] = slice(pad_to_square, None)
        out = out[tuple(sl)]

    out = np.expand_dims(out, 0)
    return to_pytorch_format(out, axes_order, device)
