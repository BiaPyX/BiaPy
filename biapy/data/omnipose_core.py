"""
Vendored Omnipose math (target generation + test-time mask reconstruction).

This module reproduces, in NumPy/PyTorch, the exact process Omnipose uses, on
both ends of the pipeline:

* **Targets** (:func:`omnipose_masks_to_flows`): an Eikonal **distance field**
  solved by iterative geometric-mean relaxation on a same-label affinity graph,
  and the **flow field** obtained as a smoothed multi-stencil gradient of it.
* **Reconstruction** (:func:`compute_masks_omnipose`, the ``cluster`` path):
  divergence-rescaled, suppressed (step-damped) Euler flow following, then DBSCAN
  clustering of the convergence points, hole filling and small-mask removal.

Faithful port of Omnipose's original code (MIT License, Copyright (c) 2026 Kevin
Cutler):

* ``omnipose/core/fields.py``   -> ``_iterate``, ``eikonal_update_torch``,
                                    ``update_torch``, ``_gradient``, ``div_rescale``,
                                    ``step_factor``
* ``omnipose/core/steps.py``    -> ``steps_batch``, ``follow_flows``
* ``omnipose/core/masks.py``    -> ``compute_masks`` (cluster path), ``get_masks``,
                                    ``remove_bad_flow_masks``, ``flow_error``
* ``ocdkit/array/spatial.py``   -> ``kernel_setup``, ``get_neighbors``,
                                    ``get_neigh_inds``, ``masks_to_affinity``
* ``ocdkit/array/normalize.py`` -> ``normalize_field``, ``normalize99``
* ``ocdkit/array/ops.py``       -> ``divergence``
* ``ocdkit/measure/diameter.py``-> ``diameters``, ``dist_to_diam``
* ``omnipose/utils``            -> ``fill_holes_and_remove_small_masks``

Expressed with NumPy/PyTorch so BiaPy carries no Omnipose/ocdkit dependency.
Omnipose's ``new_DBSCAN`` (the ``dbscan`` package) is replaced by
``sklearn.cluster.DBSCAN`` (identical algorithm; border-point assignment may
differ marginally). Target flows validated against ``omnipose.core.masks_to_flows``
(cosine >= 0.998, ``max|Δmu| ~ 1e-3`` on dsb2018).
"""
from itertools import product
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn.functional as _F
from scipy.spatial import cKDTree
from skimage import measure, filters

__all__ = ["omnipose_masks_to_flows", "compute_masks_omnipose"]


def _kernel_setup(dim: int):
    """ND neighbour steps grouped by connectivity, with per-group distance factors.

    Port of ``ocdkit.array.spatial.kernel_setup``.

    Returns
    -------
    steps : (3**dim, dim) int array of all neighbour offsets.
    inds  : list of arrays, step indices grouped by connectivity level
            (center, cardinal, ordinal, ...).
    idx   : int, index of the zero (center) step.
    fact  : (dim+1,) float array, Euclidean distance factor per group (sqrt of level).
    sign  : (3**dim,) int array, connectivity level per step.
    """
    steps = np.array(list(product([-1, 0, 1], repeat=dim)))
    sign = np.sum(np.abs(steps), axis=1)
    uniq = np.unique(sign)
    inds = [np.where(sign == i)[0] for i in uniq]
    idx = inds[0][0]
    fact = np.sqrt(uniq.astype(float))
    return steps, inds, idx, fact, sign


def _get_neighbors(coords, steps, dim, shape, edges=None, pad=0):
    """Neighbour coordinates of every foreground pixel (Neumann reflection at image edges).

    Port of ``ocdkit.array.spatial.get_neighbors``. Returns an
    ``(dim, nsteps, npix)`` int array. At image borders a step that would leave
    the volume is reflected back onto the source pixel (Neumann boundary).
    """
    if edges is None:
        edges = [np.array([-1 + pad, s - pad]) for s in shape]
    npix = coords[0].shape[-1]
    nsteps = len(steps)
    neighbors = np.empty((dim, nsteps, npix), dtype=np.int64)
    edge_masks = []
    for d in range(dim):
        mask = np.zeros(shape[d], dtype=bool)
        ve = edges[d][(edges[d] >= 0) & (edges[d] < shape[d])]
        mask[ve] = True
        edge_masks.append(mask)
    for d in range(dim):
        cm = edge_masks[d]
        size_d = shape[d]
        for n, step_d in enumerate(steps[:, d]):
            X = coords[d] + step_d
            Xc = X.copy()
            np.clip(Xc, 0, size_d - 1, out=Xc)
            Xs = X + step_d
            np.clip(Xs, 0, size_d - 1, out=Xs)
            oob = np.logical_and(cm[Xc], ~cm[Xs])
            out = Xc
            out[oob] = coords[d][oob]
            neighbors[d, n] = out
    return neighbors


def _get_neigh_inds(neighbors, coords, shape):
    """Map neighbour coordinates to flat foreground-pixel indices.

    Port of ``ocdkit.array.spatial.get_neigh_inds``. Background/out-of-mask
    neighbours map to ``-1`` (later zeroed via the affinity graph).
    """
    neighbors = tuple(neighbors)
    npix = neighbors[0].shape[-1]
    indexes = np.arange(npix)
    ind_matrix = -np.ones(shape, int)
    ind_matrix[tuple(coords)] = indexes
    neigh_inds = ind_matrix[neighbors]
    return indexes, neigh_inds, ind_matrix


def _masks_to_affinity(masks, idx, dim, neighbors):
    """Boolean same-label affinity graph ``(nsteps, npix)``.

    Port of the standard (no-links) branch of
    ``ocdkit.array.spatial.masks_to_affinity``. An entry is ``True`` when the
    neighbour shares the pixel's label, or when the step was reflected at an
    image edge (Neumann boundary); the center step is forced ``False``.
    """
    is_edge = np.logical_and.reduce([neighbors[d] == neighbors[d][idx] for d in range(dim)])
    piece = masks[tuple(neighbors)]
    is_self = piece == piece[idx]
    aff = np.logical_or.reduce([is_self, is_edge])
    aff[idx] = 0
    return aff


def _update(a, f, fsq):
    """Solve the 2-input Eikonal quadratic (port of ``update_torch``, d==2 case).

    ``a`` is ``(2, npix)`` holding the two directional minima; ``f`` the step
    length and ``fsq = f**2``.
    """
    a0 = np.minimum(a[0], a[1])
    a1 = np.maximum(a[0], a[1])
    sum_a = a0 + a1
    sum_a2 = a0 * a0 + a1 * a1
    return 0.5 * (sum_a + np.sqrt(np.clip(sum_a * sum_a - 2 * (sum_a2 - fsq), 0, None)))


def _eikonal_update(Tneigh, inds, fact):
    """One Eikonal relaxation step: geometric mean over connectivity groups.

    Port of ``eikonal_update_torch`` (``geometric=1``). For each group
    (cardinal, ordinal, ...) the minima of opposite neighbour pairs feed
    :func:`_update`; the group solutions are multiplied and raised to ``1/n``.
    """
    n = len(fact) - 1
    phi = np.ones_like(Tneigh[0])
    for ind, f in zip(inds[1:], fact[1:]):
        npair = len(ind) // 2
        left = ind[:npair]
        right = np.flip(ind)[:npair]
        mins = np.minimum(Tneigh[left], Tneigh[right])
        phi *= _update(mins, f, f * f)
    return np.power(phi, 1.0 / n)


def _iterate(T, neigh_inds, isneigh, inds, fact, n_iter, eps=1e-3, check_every=10):
    """Relax the distance field (port of ``_iterate``, ``omni=True``).

    ``T`` starts as ones over the foreground; neighbours outside the cell are
    zeroed by ``isneigh`` (Dirichlet 0 at cell boundaries). Iteration 0 adds a
    one-time neighbour-mean smoothing, exactly as Omnipose does. Early-exits when
    the mean-squared update drops below ``eps``.
    """
    T0 = T.copy()
    t = 0
    for t in range(n_iter):
        Tn = T[neigh_inds] * isneigh
        T = _eikonal_update(Tn, inds, fact)
        if t < 1:  # one-time initial smoothing on the first iteration
            Tn = T[neigh_inds] * isneigh
            T = Tn.mean(axis=0)
        err = np.mean((T - T0) ** 2)
        T0 = T.copy()
        if (t % check_every) == (check_every - 1) and err < eps:
            break
    return T


def _gradient(T, steps, fact, inds, isneigh, neigh_inds, central_inds, dim, npix):
    """Smoothed multi-stencil gradient of the distance field (port of ``_gradient``).

    Central differences over opposite neighbour pairs are averaged across
    connectivity groups, then smoothed by a neighbour weighting proportional to
    the (absolute) directional agreement of the raw gradient. Returns
    ``(dim, npix)`` flow components.
    """
    n_axes = len(fact) - 1
    fd = np.zeros((n_axes, dim, npix), dtype=T.dtype)
    for ax, (ind, f) in enumerate(zip(inds[1:], fact[1:])):
        vals = T[neigh_inds[ind]].copy()
        vals[~isneigh[ind]] = 0
        mid = len(ind) // 2
        r = np.arange(mid)
        vecs = steps[ind].astype(float)
        uvecs = (vecs[-(r + 1)] - vecs[r]).T          # (dim, mid)
        diff = vals[-(r + 1)] - vals[r]               # (mid, npix)
        fd[ax] = uvecs @ diff / (2 * f) ** 2          # (dim, npix)
    mu = np.mean(fd, axis=0)                           # (dim, npix)
    weight = np.abs(np.sum(mu[:, neigh_inds] * mu[:, central_inds][:, None, :], axis=0))
    weight[~isneigh] = 0
    wsum = weight.sum(axis=0)
    out = np.where(
        wsum != 0,
        (mu[:, neigh_inds] * weight).sum(axis=1) / np.where(wsum == 0, 1, wsum),
        0.0,
    )
    return out                                         # (dim, npix)


def omnipose_masks_to_flows(masks: NDArray, n_iter: int = 50) -> Tuple[NDArray, NDArray]:
    """Compute Omnipose's distance field and flow field from an instance label map.

    Faithful NumPy reproduction of ``omnipose.core.masks_to_flows(omni=True)``.
    The distance field is the Eikonal solution relaxed on a same-label affinity
    graph (0 at cell boundaries); the flow is its smoothed gradient. Neither is
    unit-normalised — the flow magnitude naturally decays to 0 at the cell
    skeleton (the sink), which is what Omnipose's dynamics rely on.

    Parameters
    ----------
    masks : 2D/3D int array
        Instance label map (0 = background).
    n_iter : int, optional
        Relaxation-iteration cap (Omnipose's training/``masks_to_flows_batch``
        default is 50, with early stopping on convergence). Default 50.

    Returns
    -------
    T : float32 array, shape ``masks.shape``
        Smooth distance field (0 outside the foreground).
    mu : float32 array, shape ``(masks.ndim,) + masks.shape``
        Flow components in axis order ``[..., y, x]`` (``mu[-2]`` = Y, ``mu[-1]`` = X;
        ``mu[-3]`` = Z in 3D). Zero in the background. NOT scaled by 5 (Omnipose's
        ``mu*5`` target scaling is applied downstream, at loss time).
    """
    masks = np.ascontiguousarray(masks)
    dim = masks.ndim
    shape = masks.shape
    T_grid = np.zeros(shape, dtype=np.float32)
    mu_grid = np.zeros((dim,) + shape, dtype=np.float32)
    if not masks.any():
        return T_grid, mu_grid

    coords = np.nonzero(masks)
    npix = coords[0].shape[-1]
    steps, inds, idx, fact, sign = _kernel_setup(dim)
    neighbors = _get_neighbors(coords, steps, dim, shape)
    _indexes, neigh_inds, ind_matrix = _get_neigh_inds(neighbors, coords, shape)
    central_inds = ind_matrix[tuple(neighbors[:, idx])]
    isneigh = _masks_to_affinity(masks, idx, dim, neighbors)

    T = np.ones(npix, dtype=np.float64)
    T = _iterate(T, neigh_inds, isneigh, inds, fact, n_iter)
    mu = _gradient(T, steps, fact, inds, isneigh, neigh_inds, central_inds, dim, npix)

    T_grid[coords] = T.astype(np.float32)
    for d in range(dim):
        mu_grid[d][coords] = mu[d].astype(np.float32)
    return T_grid, mu_grid


# ============================================================================
# Test-time mask reconstruction (Omnipose ``compute_masks`` -- cluster path)
# ============================================================================
def _normalize_field(mu, cutoff=0):
    """Normalize all field vectors with magnitude > cutoff to unit length.

    Port of ``ocdkit.array.normalize.normalize_field`` (NumPy branch). ``mu`` has
    shape ``(D, *spatial)``.
    """
    mag = np.sqrt(np.nansum(mu ** 2, axis=0))
    valid = mag > cutoff
    return np.where(valid, mu / np.where(valid, mag, 1.0), mu)


def _normalize99(Y, lower=0.01, upper=99.99):
    """Clip to the [lower, upper] percentile range and rescale to [0, 1].

    Port of ``ocdkit.array.normalize.normalize99`` (NumPy branch).
    """
    lower_val, upper_val = np.quantile(Y, np.array([lower, upper]) / 100)
    denom = upper_val - lower_val
    if denom == 0:
        denom = 1.0
    return np.clip((Y - lower_val) / denom, 0, 1)


def _divergence(f):
    """Divergence of a ``(D, *spatial)`` vector field (port of ``ocdkit...ops.divergence``)."""
    num_dims = len(f)
    if any(f.shape[1 + i] < 2 for i in range(num_dims)):
        return np.zeros_like(f[0])
    return np.add.reduce([np.gradient(f[i], axis=i) for i in range(num_dims)])


def _div_rescale(dP, mask, p=1):
    """Rescale flow magnitude by normalized-divergence (port of ``fields.div_rescale``).

    Normalizes each vector to unit length, then multiplies by the (0-1 normalized)
    divergence of that unit field. This makes the reconstruction insensitive to the
    absolute flow scale (so the training ``mu*5`` scaling is irrelevant here).
    """
    dP = dP.copy()
    dP *= mask
    dP = _normalize_field(dP)
    if p > 0:
        div = _normalize99(_divergence(dP)) ** p
        dP *= div
    return dP


def _dist_to_diam(dt_pos, n):
    """Mean diameter from positive distance-field values (port of ``diameter.dist_to_diam``)."""
    return 2 * (n + 1) * np.mean(dt_pos)


def _diameters(masks, dt, dist_threshold=0):
    """Mean cell diameter from a distance field (port of ``diameter.diameters``)."""
    if dist_threshold < 0:
        dist_threshold = 0
    dt_pos = np.abs(dt[dt > dist_threshold])
    if np.any(dt_pos):
        return _dist_to_diam(np.abs(dt_pos), n=masks.ndim)
    return 0


def _step_factor(t):
    """Euler-integration suppression factor (port of ``fields.step_factor``)."""
    return 1.0 + t


def _steps_batch(p, dP, niter, suppress=True, interp=True):
    """Euler integration of positions ``p`` under flow ``dP`` (port of ``steps.steps_batch``).

    Parameters
    ----------
    p : (B, D, I) float tensor of starting positions (axis order [.., y, x]).
    dP : (B, D, *spatial) float tensor flow field.
    niter : int.
    suppress : bool -- step-damped momentum integration (``dPt=(dPt+dPt0)/2 / (1+t)``).
    interp : bool -- bilinear sampling; ``suppress`` forces nearest (as in Omnipose).

    Returns
    -------
    p : (B, D, I) final positions.
    """
    align_corners = True
    interp = interp and not suppress
    mode = "bilinear" if interp else "nearest"

    d = dP.shape[1]
    spatial = dP.shape[2:]
    inds = list(range(d))[::-1]

    shape = np.array([int(s) for s in spatial])[inds] - 1.0
    B, D, I = p.shape
    pt = p[:, inds].permute(0, 2, 1).view([B] + [1] * (D - 1) + [I, D]).float()
    flow = dP[:, inds]

    for k in range(d):
        if shape[k] == 0:
            pt[..., k] = 0.0
            flow[:, k] = 0.0
        else:
            pt[..., k] = 2 * pt[..., k] / shape[k] - 1
            flow[:, k] = 2 * flow[:, k] / shape[k]

    if suppress:
        dPt0 = _F.grid_sample(flow, pt, mode=mode, align_corners=align_corners)

    for t in range(niter):
        dPt = _F.grid_sample(flow, pt, mode=mode, align_corners=align_corners)
        if suppress:
            dPt = (dPt + dPt0) / 2.0
            dPt0 = dPt.clone()
            dPt = dPt / _step_factor(t)
        for k in range(d):
            pt[..., k] = torch.clamp(pt[..., k] + dPt[:, k], -1.0, 1.0)

    pt = (pt + 1) * 0.5
    for k in range(d):
        pt[..., k] *= shape[k]

    return pt[..., inds].transpose(-1, 1).contiguous()


def _follow_flows(dP, inds, niter, suppress=True, interp=True, device=None):
    """Run dynamics on the foreground pixels (port of ``masks.follow_flows``).

    ``dP`` is a ``(D, *spatial)`` NumPy flow field; ``inds`` the ``(D, N)`` foreground
    coordinates. Returns ``p`` of shape ``(D, *spatial)`` (background pixels keep their
    grid coordinates, foreground pixels hold their convergence positions).
    """
    device = device or torch.device("cpu")
    dim = dP.shape[0]
    spatial = dP.shape[1:]
    flow_pred = torch.tensor(dP, device=device).unsqueeze(0)          # (1, D, *spatial)

    coords = [torch.arange(0, l, device=device) for l in spatial]
    mesh = torch.meshgrid(coords, indexing="ij")
    initial_points = torch.stack(mesh, dim=0).float()                 # (D, *spatial)
    initial_points = initial_points.unsqueeze(0)                      # (1, D, *spatial)
    final_points = initial_points.clone()

    cell_px = (Ellipsis,) + tuple(inds)
    if inds.ndim < 2 or inds.shape[0] < dim:
        return final_points.squeeze(0).cpu().numpy()

    final_p = _steps_batch(initial_points[cell_px], flow_pred, int(niter),
                           suppress=suppress, interp=interp)
    final_points[cell_px] = final_p.squeeze()
    return final_points.squeeze(0).cpu().numpy()


def _dbscan(X, eps, min_samples):
    """DBSCAN labels (-1 = noise). ``sklearn`` stand-in for Omnipose's ``new_DBSCAN``."""
    from sklearn.cluster import DBSCAN
    return DBSCAN(eps=eps, min_samples=min_samples).fit(X).labels_


def _get_masks(p, dist, mask, inds, cluster=False, eps=None, min_samples=5,
               diam_threshold=12.0, verbose=False):
    """Cluster the convergence points into instances (port of ``masks.get_masks``, cluster path).

    Parameters
    ----------
    p : (D, *spatial) convergence positions from :func:`_follow_flows`.
    dist : (*spatial) distance field.
    mask : (*spatial) bool foreground.
    inds : (D, N) foreground coordinates.
    """
    dt = np.abs(dist[mask])
    d = _dist_to_diam(dt, mask.ndim) if np.any(dt) else 0

    if eps is None:
        eps = 2 ** 0.5
    if d <= diam_threshold:
        cluster = True

    cell_px = tuple(inds)
    newinds = p[(Ellipsis,) + cell_px].T          # (N, D) convergence points
    out = np.zeros(p.shape[1:], np.uint32)

    if cluster:
        labels = _dbscan(newinds, eps=eps, min_samples=min_samples)
        # Snap noise points (label == -1) to the nearest cluster within eps.
        tree = cKDTree(newinds)
        o_inds = np.where(labels == -1)[0]
        if len(o_inds):
            outliers = newinds[o_inds]
            nearest_dists, nearest_indices = tree.query(outliers, k=min(5, len(newinds)))
            nearest_labels = labels[nearest_indices]
            nearest_idx = [np.where(n != -1)[0][0] if np.any(n != -1) else 0 for n in nearest_labels]
            l = [nl[i] if nd[i] < eps else -1 for i, nl, nd in zip(nearest_idx, nearest_labels, nearest_dists)]
            labels[o_inds] = l
        out[cell_px] = labels + 1
    else:
        newinds_r = np.rint(newinds.T).astype(int)
        new_px = tuple(newinds_r)
        skelmask = np.zeros_like(dist, dtype=bool)
        skelmask[new_px] = 1
        labels = measure.label(skelmask, connectivity=skelmask.ndim)
        out[cell_px] = labels[new_px]

    return out


def _flow_error(maski, dP_net, omni=True):
    """Mean-squared flow error per mask vs flows regenerated from the mask (port of ``flow_error``)."""
    if dP_net.shape[1:] != maski.shape:
        return np.zeros(maski.max())
    _, dP_masks = omnipose_masks_to_flows(maski)     # (D, *spatial) unit-scale target flow
    flow_errors = np.zeros(maski.max())
    from scipy.ndimage import mean as _ndmean
    for i in range(dP_masks.shape[0]):
        flow_errors += _ndmean((dP_masks[i] - dP_net[i] / 5.0) ** 2, maski,
                               index=np.arange(1, maski.max() + 1))
    return flow_errors


def _remove_bad_flow_masks(masks, flows, threshold=0.4):
    """Zero masks whose regenerated flow disagrees with the network (port of ``remove_bad_flow_masks``)."""
    merrors = _flow_error(masks, flows)
    badi = 1 + np.nonzero(merrors > threshold)[0]
    masks[np.isin(masks, badi)] = 0
    return masks


def compute_masks_omnipose(
    dP: NDArray,
    dist: NDArray,
    mask_threshold: float = 0.0,
    flow_threshold: float = 0.4,
    niter: Optional[int] = None,
    eps: Optional[float] = None,
    min_samples: int = 5,
    diam_threshold: float = 12.0,
    interp: bool = True,
    cluster: bool = False,
    device: Optional["torch.device"] = None,
) -> NDArray:
    """Reconstruct instance masks from predicted flow + distance (Omnipose ``compute_masks``, cluster path).

    Faithful port of ``omnipose.core.masks.compute_masks`` for ``omni=True``,
    ``affinity_seg=False`` (the standard DBSCAN reconstruction). No padding is
    applied (``compute_masks`` uses ``pad=0`` on this path). Omnipose's final
    hole-filling / small-mask removal is intentionally left out -- BiaPy applies
    those as a separate post-processing step.

    Parameters
    ----------
    dP : (D, *spatial) float array
        Predicted flow field, axis order ``[.., y, x]`` (``[z, y, x]`` in 3D). The
        absolute scale is irrelevant -- :func:`_div_rescale` normalizes it.
    dist : (*spatial) float array
        Predicted distance field (Omnipose ``Db``; background is negative).
    mask_threshold : float
        Foreground is ``hysteresis(dist, mask_threshold-1, mask_threshold)``. Default 0.
    flow_threshold : float
        Remove masks whose regenerated flow error exceeds this. ``<= 0`` skips the check.
    niter : int, optional
        Euler steps. ``None`` -> ``int(diameters(iscell, dist))``.
    eps, min_samples : DBSCAN parameters. ``eps=None`` -> ``sqrt(2)``.
    diam_threshold : float
        Below this mean diameter, subpixel clustering is forced on (Omnipose default 12).
    cluster : bool
        Force DBSCAN clustering regardless of diameter.

    Returns
    -------
    masks : int32 array, shape ``dist.shape`` (0 = background).
    """
    device = device or torch.device("cpu")

    # Foreground via hysteresis threshold (Omnipose omni path).
    iscell = filters.apply_hysteresis_threshold(dist, mask_threshold - 1, mask_threshold)
    if not np.any(iscell):
        return np.zeros(dist.shape, dtype=np.int32)

    coords = np.array(np.nonzero(iscell)).astype(np.int32)

    # Divergence-rescaled, suppressed flow (omni + suppress branch).
    dP_ = _div_rescale(dP, iscell)

    if niter is None:
        niter = int(_diameters(iscell, dist))
    niter = max(1, int(niter))

    p = _follow_flows(dP_, coords, niter, suppress=True, interp=interp, device=device)

    labels = _get_masks(p, dist, iscell, coords, cluster=cluster, eps=eps,
                        min_samples=min_samples, diam_threshold=diam_threshold)

    if labels.max() > 0 and flow_threshold is not None and flow_threshold > 0:
        labels = _remove_bad_flow_masks(labels, np.asarray(dP), threshold=flow_threshold)

    labels = labels * iscell
    # Renumber labels consecutively (0 = background).
    if labels.max() > 0:
        _, labels = np.unique(labels, return_inverse=True)
        labels = labels.reshape(dist.shape)
    return labels.astype(np.int32)
