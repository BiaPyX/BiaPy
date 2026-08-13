"""
Malis-weighted affinity loss, for PROBLEM.IMAGE_TO_IMAGE.MEMBRANE_REPAIR's
``LOSS.TYPE == "MEMBRANE_REPAIR_AFFINITY"``.

Ref: Turaga et al., "Maximin Affinity Learning of Image Segmentation"
(https://arxiv.org/abs/0911.5372).

For every affinity edge, two max-spanning-tree passes (Kruskal + union-find, edges sorted by
predicted affinity descending) count how many pixel pairs that edge is responsible for merging
correctly or incorrectly:

- Positive (constrained) pass: restricted to edges within the same GT instance; every merge is a
  should-connect pair, credited to the responsible edge.
- Negative (unconstrained) pass: every edge eligible; a merge across two different GT instances is
  a should-not-connect pair, tracked via a per-component GT-id histogram.

The resulting per-edge pair counts are used as fixed (detached) weights in a differentiable
weighted MSE against 1/0, avoiding a custom autograd backward pass.

GT instance labels are reconstructed from the target's short-range (unit-offset) affinity
channels via connected components (the generator drops the raw instance-label channel before it
reaches the model). Requires at least one unit-offset channel in
``DATA_CHANNELS_EXTRA_OPTS["A"]``, and ``widen_borders: 0`` for the ``"A"`` channel
(``labels_into_channels``'s default is 1) -- otherwise the reconstruction fragments into many more
components than the true instance count. ``Membrane_Repair_Workflow`` validates this when MALIS is
enabled (``LOSS.WEIGHTS[1] > 0``).

The MST passes are numba-jitted (``nogil=True``); the batch loop dispatches samples to a
``ThreadPoolExecutor`` so they run concurrently.
"""
from concurrent.futures import ThreadPoolExecutor
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from numba import njit
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


@njit(cache=True, nogil=True)
def _uf_find(parent: NDArray, x: int) -> int:
    """Union-find ``find`` with path compression."""
    root = x
    while parent[root] != root:
        root = parent[root]
    while parent[x] != root:
        nxt = parent[x]
        parent[x] = root
        x = nxt
    return root


@njit(cache=True, nogil=True)
def _uf_union(parent: NDArray, rank: NDArray, a: int, b: int) -> int:
    """Union-find ``union`` by rank. Returns the new root (always ``a`` or ``b``'s root)."""
    ra, rb = _uf_find(parent, a), _uf_find(parent, b)
    if ra == rb:
        return ra
    if rank[ra] < rank[rb]:
        ra, rb = rb, ra
    parent[rb] = ra
    if rank[ra] == rank[rb]:
        rank[ra] += 1
    return ra


def _build_edges(shape: Tuple[int, ...], offsets: Sequence[Tuple[int, ...]]):
    """
    Build the flat-pixel-index edge list for a set of offsets.

    Parameters
    ----------
    shape : tuple of int
        Spatial shape, ``(y, x)`` in 2D or ``(z, y, x)`` in 3D.

    offsets : sequence of tuple of int
        One offset per affinity channel, same order/length as the channel axis.

    Returns
    -------
    edge_u, edge_v : (E,) int64 Numpy arrays
        Flat pixel indices (into a ``shape``-shaped array) of each edge's two endpoints.

    edge_channel : (E,) int64 Numpy array
        Which offset/channel each edge belongs to.
    """
    ndim = len(shape)
    idx = np.arange(int(np.prod(shape)), dtype=np.int64).reshape(shape)
    u_list, v_list, c_list = [], [], []
    for c, off in enumerate(offsets):
        slices_u = tuple(slice(max(0, -o), shp - max(0, o)) for o, shp in zip(off, shape))
        u = idx[slices_u].reshape(-1)
        slices_v = tuple(slice(max(0, o), shp - max(0, -o)) for o, shp in zip(off, shape))
        v = idx[slices_v].reshape(-1)
        u_list.append(u)
        v_list.append(v)
        c_list.append(np.full(u.shape[0], c, dtype=np.int64))
    return np.concatenate(u_list), np.concatenate(v_list), np.concatenate(c_list)


def _reconstruct_gt_labels(target_np: NDArray, offsets: List[Tuple[int, ...]], shape: Tuple[int, ...]) -> NDArray:
    """
    Reconstruct the GT instance-label volume from the target's short-range affinity channels.

    Two adjacent pixels share an instance iff their unit-offset target affinity is 1, so connected
    components of that adjacency graph recover the true instance labels exactly. Every pixel gets
    a component id >= 1 (no separate background id).

    Parameters
    ----------
    target_np : (C, *shape) Numpy array
        Target affinities.

    offsets : list of tuple of int
        One offset per channel, same order as ``target_np``'s channel axis.

    shape : tuple of int
        Spatial shape.

    Returns
    -------
    labels : Numpy array of int64, shape ``shape``
        Reconstructed instance labels, ``1..K``.
    """
    n = int(np.prod(shape))
    rows, cols = [], []
    for c, off in enumerate(offsets):
        if sum(abs(o) for o in off) != 1:
            continue
        # seg2aff_pni (biapy.utils.util) stores each offset's affinity at the higher-index
        # (target) position -- offsets here are always positive unit offsets, so read at edge_v.
        edge_u, edge_v, edge_c = _build_edges(shape, [off])
        aff_at_v = target_np[c].reshape(-1)[edge_v]
        connected = aff_at_v > 0.5
        rows.append(edge_u[connected])
        cols.append(edge_v[connected])
    if not rows:
        raise ValueError(
            "MalisLoss requires at least one unit-offset ('short-range') affinity channel in "
            "PROBLEM.IMAGE_TO_IMAGE.MEMBRANE_REPAIR.DATA_CHANNELS_EXTRA_OPTS['A'] to reconstruct "
            "GT instance labels."
        )
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    adj = coo_matrix((np.ones(rows.shape[0]), (rows, cols)), shape=(n, n))
    _, labels_flat = connected_components(adj, directed=False)
    return (labels_flat.reshape(shape) + 1).astype(np.int64)


@njit(cache=True, nogil=True)
def _mst_pass_constrained_jit(
    edge_u: NDArray, edge_v: NDArray, order: NDArray, gt_label_flat: NDArray
) -> NDArray:
    """
    Positive (constrained) Kruskal max-spanning-tree pass, fully compiled.

    Restricted to edges connecting pixels in the same GT instance; every merge is a should-connect
    pair, so no histogram is needed (unlike the unconstrained pass).

    Parameters
    ----------
    edge_u, edge_v : (E,) int64
        Flat pixel indices of every edge's endpoints (all channels concatenated).

    order : (E,) int64
        Edge indices (into ``edge_u``/``edge_v``) sorted by predicted affinity, descending.

    gt_label_flat : (N,) int64
        Flat, reconstructed GT instance labels.

    Returns
    -------
    pair_count : (E,) float64
        Per-edge pair count (indexed the same as ``edge_u``/``edge_v``, not ``order``); zero for
        edges that never caused a merge.
    """
    n = gt_label_flat.shape[0]
    parent = np.arange(n, dtype=np.int64)
    rank = np.zeros(n, dtype=np.int64)
    size = np.ones(n, dtype=np.int64)
    pair_count = np.zeros(edge_u.shape[0], dtype=np.float64)

    for idx in range(order.shape[0]):
        e = order[idx]
        u = edge_u[e]
        v = edge_v[e]
        # Cheap integer check first, before paying for two find() calls, since most edges in a
        # real volume cross instances and get skipped here.
        if gt_label_flat[u] != gt_label_flat[v]:
            continue
        ru = _uf_find(parent, u)
        rv = _uf_find(parent, v)
        if ru == rv:
            continue

        su = size[ru]
        sv = size[rv]
        pair_count[e] = su * sv

        new_root = _uf_union(parent, rank, ru, rv)
        size[new_root] = su + sv

    return pair_count


@njit(cache=True, nogil=True)
def _mst_pass_unconstrained_jit(
    edge_u: NDArray, edge_v: NDArray, order: NDArray, gt_label_flat: NDArray, n_gt: int
) -> NDArray:
    """
    Negative (unconstrained) Kruskal max-spanning-tree pass, fully compiled.

    Every edge is eligible; a merge across two different GT instances is a should-not-connect
    pair, counted via a per-component GT-id histogram (``(n_pixels, n_gt + 1)`` int64 array
    indexed by GT id; column 0 unused, ids are ``1..n_gt``).

    Parameters
    ----------
    edge_u, edge_v : (E,) int64
        Flat pixel indices of every edge's endpoints (all channels concatenated).

    order : (E,) int64
        Edge indices (into ``edge_u``/``edge_v``) sorted by predicted affinity, descending.

    gt_label_flat : (N,) int64
        Flat, reconstructed GT instance labels, ``1..n_gt``.

    n_gt : int
        Number of distinct GT instances (``gt_label_flat.max()``).

    Returns
    -------
    pair_count : (E,) float64
        Per-edge pair count (indexed the same as ``edge_u``/``edge_v``, not ``order``); zero for
        edges that never caused a merge.
    """
    n = gt_label_flat.shape[0]
    parent = np.arange(n, dtype=np.int64)
    rank = np.zeros(n, dtype=np.int64)
    size = np.ones(n, dtype=np.int64)
    pair_count = np.zeros(edge_u.shape[0], dtype=np.float64)

    hist = np.zeros((n, n_gt + 1), dtype=np.int64)
    for i in range(n):
        hist[i, gt_label_flat[i]] = 1

    for idx in range(order.shape[0]):
        e = order[idx]
        u = edge_u[e]
        v = edge_v[e]
        ru = _uf_find(parent, u)
        rv = _uf_find(parent, v)
        if ru == rv:
            continue

        su = size[ru]
        sv = size[rv]
        total_pairs = su * sv

        matching = 0
        for g in range(1, n_gt + 1):
            matching += hist[ru, g] * hist[rv, g]
        pair_count[e] = total_pairs - matching  # wrong (cross-instance) pairs merged

        new_root = _uf_union(parent, rank, ru, rv)
        other_root = rv if new_root == ru else ru
        size[new_root] = su + sv
        for g in range(1, n_gt + 1):
            hist[new_root, g] += hist[other_root, g]

    return pair_count


class MalisLoss(nn.Module):
    """
    Malis-weighted affinity loss (see the module docstring for the full algorithm and scope).
    """

    def __init__(self, offsets: List[Tuple[int, ...]], eps: float = 1e-8):
        """
        Initialize the Malis-weighted affinity loss.

        Parameters
        ----------
        offsets : list of tuple of int
            One offset per affinity channel (``(dz, dy, dx)`` in 3D, ``(dy, dx)`` in 2D), in the
            same order as the model's affinity output channels -- matches
            ``PROBLEM.IMAGE_TO_IMAGE.MEMBRANE_REPAIR.DATA_CHANNELS_EXTRA_OPTS[0]["A"]``'s
            interleaved ``(z_affinities, y_affinities, x_affinities)`` channel order.

        eps : float, optional
            Denominator floor.
        """
        super().__init__()
        self.offsets = list(offsets)
        self.eps = eps

    def _compute_sample_weights(
        self,
        pred_np: NDArray,
        target_np: NDArray,
        edge_u: NDArray,
        edge_v: NDArray,
        edge_c: NDArray,
        shape: Tuple[int, ...],
    ) -> Tuple[NDArray, NDArray]:
        """
        Run both MST passes for one sample and scatter the per-edge pair counts back to
        per-(channel, position) weight arrays. Called per-sample from a thread pool in ``forward``.

        Returns
        -------
        weight_pos, weight_neg : Numpy arrays, same shape as ``pred_np``
        """
        gt_label = _reconstruct_gt_labels(target_np, self.offsets, shape).reshape(-1)
        n_gt = int(gt_label.max())
        flat_pred = pred_np.reshape(pred_np.shape[0], -1)
        # Predicted/target affinity for offset c is stored at the *target* (edge_v) position,
        # matching seg2aff_pni's convention (see _reconstruct_gt_labels).
        edge_affinity = flat_pred[edge_c, edge_v]
        order = np.argsort(-edge_affinity, kind="stable")

        pos_pairs = _mst_pass_constrained_jit(edge_u, edge_v, order, gt_label)
        neg_pairs = _mst_pass_unconstrained_jit(edge_u, edge_v, order, gt_label, n_gt)

        weight_pos = np.zeros(flat_pred.shape, dtype=np.float64)
        weight_neg = np.zeros(flat_pred.shape, dtype=np.float64)
        np.add.at(weight_pos, (edge_c, edge_v), pos_pairs)
        np.add.at(weight_neg, (edge_c, edge_v), neg_pairs)
        return weight_pos.reshape(pred_np.shape), weight_neg.reshape(pred_np.shape)

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the Malis-weighted affinity loss.

        Parameters
        ----------
        pred_logits : torch.Tensor
            Raw (pre-sigmoid) affinity logits, ``(B, C, ...)``.

        target : torch.Tensor
            GT affinities (0/1), ``(B, C, ...)``, same shape as ``pred_logits``.

        Returns
        -------
        loss : torch.Tensor
            Scalar loss.
        """
        pred_prob = torch.sigmoid(pred_logits)
        shape = tuple(pred_prob.shape[2:])
        edge_u, edge_v, edge_c = _build_edges(shape, self.offsets)

        batch_size = pred_prob.shape[0]
        pred_np_list = [pred_prob[b].detach().cpu().numpy().astype(np.float64) for b in range(batch_size)]
        target_np_list = [target[b].detach().cpu().numpy().astype(np.float64) for b in range(batch_size)]

        if batch_size == 1:
            weights = [self._compute_sample_weights(pred_np_list[0], target_np_list[0], edge_u, edge_v, edge_c, shape)]
        else:
            with ThreadPoolExecutor(max_workers=batch_size) as pool:
                weights = list(
                    pool.map(
                        lambda i: self._compute_sample_weights(
                            pred_np_list[i], target_np_list[i], edge_u, edge_v, edge_c, shape
                        ),
                        range(batch_size),
                    )
                )

        batch_losses = []
        for b in range(batch_size):
            weight_pos, weight_neg = weights[b]
            weight_pos_t = torch.from_numpy(weight_pos).to(pred_prob.device, pred_prob.dtype)
            weight_neg_t = torch.from_numpy(weight_neg).to(pred_prob.device, pred_prob.dtype)

            p = pred_prob[b]
            denom = (weight_pos_t.sum() + weight_neg_t.sum()).clamp_min(self.eps)
            sample_loss = (weight_pos_t * (1.0 - p) ** 2 + weight_neg_t * p**2).sum() / denom
            batch_losses.append(sample_loss)

        return torch.stack(batch_losses).mean()
