"""
Supervoxel-based structure-aware affinity loss, for PROBLEM.IMAGE_TO_IMAGE.MEMBRANE_REPAIR's
``LOSS.TYPE == "MEMBRANE_REPAIR_AFFINITY"``.

Ref: Grim, A., Chandrashekar, J., Sumbul, U., "Efficient Connectivity-Preserving Instance
Segmentation with Supervoxel-Based Loss Function", AAAI 2025 (https://arxiv.org/abs/2501.01022).
Adapted from https://github.com/AllenNeuralDynamics/supervoxel-loss (MIT License).

Method:

1. Binarize prediction and target into a foreground/background mask. There's no true background
   class here, so foreground ("cell interior") is derived as ``mean(affinity, channel) > 0.5``
   (high = same cell; low = membrane/boundary).
2. Compute the false-negative mask (target foreground the prediction missed) and, symmetrically,
   the false-positive mask.
3. For each mask, grow every connected mistake region and check whether fixing it would change the
   connectivity of the "correct" foreground it's embedded in -- a false split (false-negative
   direction) or false merge (false-positive direction). Only these "critical" regions get the
   extra structure-level penalty.
4. Loss = ``(1 - alpha) * voxel_loss + alpha * critical_weight * voxel_loss``, ``critical_weight``
   a ``beta``/``(1 - beta)`` blend of the split-risk and merge-risk critical masks.

``_detect_critical`` is a plain Python BFS (not JIT-compiled), run once per batch sample on CPU.
"""
import itertools
from collections import deque
from typing import Dict, List, Tuple

import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn as nn
from numpy.typing import NDArray


def _neighbor_offsets(ndim: int) -> List[Tuple[int, ...]]:
    """Full-connectivity neighbor offsets (8 in 2D, 26 in 3D)."""
    return [off for off in itertools.product((-1, 0, 1), repeat=ndim) if any(off)]


def _extract_component(
    y_target: NDArray,
    mistakes: NDArray,
    minus_mistakes: NDArray,
    root: Tuple[int, ...],
    nbs: List[Tuple[int, ...]],
    shape: Tuple[int, ...],
) -> Tuple[NDArray, "set[Tuple[int, ...]]", bool]:
    """
    BFS-grow the mistake region reachable from ``root`` within a single ``y_target`` component, and
    decide whether resolving it would change that component's connectivity ("critical").

    Parameters
    ----------
    y_target : NDArray of int
        Instance-labeled "ground truth" for this call (``0`` = background).
    mistakes : NDArray of bool
        False-negative mask (``y_target`` foreground the counterpart prediction missed).
    minus_mistakes : NDArray of int
        Connected components of ``y_target``'s foreground with ``mistakes`` removed -- the
        "correctly classified" sub-components ``mistakes`` may or may not be reconnecting.
    root : tuple of int
        Coordinate of the mistake voxel to grow the BFS from.
    nbs : list of tuple of int
        Neighbor offsets (see :func:`_neighbor_offsets`).
    shape : tuple of int
        Spatial shape.

    Returns
    -------
    mask : NDArray of bool
        The extracted mistake region.
    visited : set of tuple of int
        Every voxel the BFS looked at (mistake and non-mistake), so the caller can remove them from
        the pool of unprocessed mistake voxels.
    is_critical : bool
        Whether fixing ``mask`` would change ``y_target``'s connectivity.
    """
    mask = np.zeros(shape, dtype=bool)
    root_label = y_target[root]
    first_component = None
    is_critical = False
    queue = deque([root])
    visited = {root}
    while queue:
        cur = queue.popleft()
        mask[cur] = True
        for off in nbs:
            nb = tuple(c + o for c, o in zip(cur, off))
            if not all(0 <= n < s for n, s in zip(nb, shape)):
                continue
            if nb in visited or y_target[nb] != root_label:
                continue
            visited.add(nb)
            if mistakes[nb]:
                queue.append(nb)
            elif not is_critical:
                comp_id = minus_mistakes[nb]
                if first_component is None:
                    first_component = comp_id
                elif comp_id != first_component:
                    is_critical = True
    if first_component is None:
        # Whole reachable same-instance region is mistakes -- no correctly-classified anchor found,
        # so this GT component is trivially critical.
        is_critical = True
    return mask, visited, is_critical


def _detect_critical(y_target: NDArray, y_pred: NDArray) -> NDArray:
    """
    Detect "critical" false-negative regions in ``y_target`` relative to ``y_pred``: connected
    mistake components whose correction would change ``y_target``'s connectivity (i.e. currently
    causing a false split of one of its instances).

    Parameters
    ----------
    y_target : NDArray of int
        Instance-labeled array (``0`` = background), the side whose connectivity is being checked.
    y_pred : NDArray of bool (or 0/1)
        Binary foreground mask of the other side -- only foreground/background status matters here.

    Returns
    -------
    critical_mask : NDArray of bool, same shape as ``y_target``
        Voxels belonging to a critical mistake region.
    """
    shape = y_target.shape
    nbs = _neighbor_offsets(y_target.ndim)

    mistakes = (y_target > 0) & (~y_pred.astype(bool))
    minus_mistakes, _ = ndi.label(np.where(mistakes, 0, y_target > 0))

    critical_mask = np.zeros(shape, dtype=bool)
    remaining = set(map(tuple, np.argwhere(mistakes)))
    while remaining:
        root = next(iter(remaining))
        mask, visited, is_critical = _extract_component(y_target, mistakes, minus_mistakes, root, nbs, shape)
        remaining -= visited
        if is_critical:
            critical_mask |= mask
    return critical_mask


class SupervoxelLoss(nn.Module):
    """
    Supervoxel-based structure-aware affinity loss (see the module docstring for the full
    algorithm, its provenance, and the background-free domain adaptation).
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.5, threshold: float = 0.5):
        """
        Initialize the supervoxel loss.

        Parameters
        ----------
        alpha : float, optional
            Blend between the plain voxel-wise BCE (``alpha=0``) and the fully
            critical-weighted term (``alpha=1``).

        beta : float, optional
            Blend between the split-risk (``beta=1``) and merge-risk (``beta=0``) critical masks.

        threshold : float, optional
            Probability threshold used to binarize the derived cell-interior probability into a
            foreground mask before connected-component labeling.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def _critical_weight(self, pred_prob_np: NDArray, target_np: NDArray) -> NDArray:
        """Compute the spatial critical-weight map (independent of ``alpha``) for one sample."""
        target_interior = target_np.mean(axis=0) > self.threshold
        pred_interior = pred_prob_np.mean(axis=0) > self.threshold

        target_labels, _ = ndi.label(target_interior)
        pred_labels, _ = ndi.label(pred_interior)

        split_risk = _detect_critical(target_labels, pred_interior)
        merge_risk = _detect_critical(pred_labels, target_interior)
        return (self.beta * split_risk + (1.0 - self.beta) * merge_risk).astype(np.float32)

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the supervoxel loss.

        Parameters
        ----------
        pred_logits : torch.Tensor
            Raw (pre-sigmoid) affinity logits, ``(B, C, ...)``.

        target : torch.Tensor
            GT affinities (0/1), same shape as ``pred_logits``.

        Returns
        -------
        loss : torch.Tensor
            Scalar loss.
        """
        pred_prob = torch.sigmoid(pred_logits)
        voxel_loss = self.bce(pred_logits.float(), target.float())  # (B, C, ...), no reduction

        sample_losses = []
        for b in range(pred_prob.shape[0]):
            pred_prob_np = pred_prob[b].detach().cpu().numpy()
            target_np = target[b].detach().cpu().numpy()
            weight_np = self._critical_weight(pred_prob_np, target_np)
            weight_t = torch.from_numpy(weight_np).to(voxel_loss.device, voxel_loss.dtype).unsqueeze(0)

            combined = (1.0 - self.alpha) * voxel_loss[b] + self.alpha * weight_t * voxel_loss[b]
            sample_losses.append(combined.mean())
        return torch.stack(sample_losses).mean()
