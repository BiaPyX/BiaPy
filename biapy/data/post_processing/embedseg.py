"""
EmbedSeg post-processing utilities for BiaPy.

This module gathers the EmbedSeg-specific inference code (spatial-embedding clustering) used by the
``E_offset``/``E_sigma``/``E_seediness`` instance-segmentation channels, kept separate from the
generic ``post_processing`` module. Test-time augmentation lives in
``biapy.data.post_processing.tta`` and covers EmbedSeg through its channel spec, so there is no
EmbedSeg-specific ensembler here.

Reference: `EmbedSeg: Embedding-based Instance Segmentation for Biomedical Microscopy Data
<https://www.sciencedirect.com/science/article/pii/S1361841522001700>`_.

Code adapted from: `EmbedSeg <https://github.com/juglab/EmbedSeg>`_.
"""
import torch
import numpy as np
from typing import List
from numpy.typing import NDArray


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
