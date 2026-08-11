"""
Affinity-based region agglomeration.

Turns predicted affinities into instances by oversegmenting into small fragments, then greedily
merging fragment pairs by a quantile of their affinity histogram (accumulated across all offsets,
not just the first short-range triple) until a merge threshold is reached.

Public entry points:

- ``affinity_agglomeration``: merges an existing fragment labeling using affinities.
- ``watershed_and_agglomerate_affinities``: builds fragments via ``watershed_by_channels`` and
  agglomerates them in one call.
"""
import heapq
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

from biapy.data.post_processing.post_processing import watershed_by_channels

_HIST_BINS = 256
_DEFAULT_QUANTILE = 50.0


def _offset_pair_slices(offset: Tuple[int, ...], ndim: int) -> Tuple[Tuple[slice, ...], Tuple[slice, ...], Tuple[slice, ...]]:
    """
    Build the (lower-index, higher-index, value) slice triple for one affinity offset.

    Matches ``biapy.utils.util.seg2aff_pni``'s storage convention: for a signed per-axis offset
    ``d``, voxel pair ``(p, p + |d|)`` is compared, and the result is stored at the pair's
    higher-index voxel when ``d > 0`` and its lower-index voxel when ``d < 0``.

    Parameters
    ----------
    offset : tuple of int
        Signed per-axis shift (e.g. ``(1, 0, 0)``). One entry per affinity channel, at most one
        nonzero axis per entry -- the format ``biapy.engine.membrane_repair._affinity_offsets``/
        ``MalisLoss`` use.

    ndim : int
        Number of spatial dimensions (``2`` or ``3``).

    Returns
    -------
    lo_slices, hi_slices, val_slices : tuple of slice
        ``fragments[lo_slices]``/``fragments[hi_slices]`` give element-aligned arrays of both
        endpoints of every pair; ``val_slices`` indexes the matching affinity value.
    """
    lo = [slice(None)] * ndim
    hi = [slice(None)] * ndim
    val = [slice(None)] * ndim
    for axis, d in enumerate(offset):
        if d == 0:
            continue
        ad = abs(d)
        lo[axis] = slice(0, -ad)
        hi[axis] = slice(ad, None)
        val[axis] = slice(ad, None) if d > 0 else slice(0, -ad)
    return tuple(lo), tuple(hi), tuple(val)


def _edge_score(hist: NDArray, quantile: float) -> float:
    """
    Score a fragment-pair edge as a quantile of its accumulated affinity histogram.

    Parameters
    ----------
    hist : NDArray
        ``(_HIST_BINS,)`` int array, one bin count per ``1 / _HIST_BINS``-wide affinity interval.

    quantile : float
        Percentile in ``[0, 100]`` (``50`` = median).

    Returns
    -------
    score : float
        Bin-center value, in ``[0, 1]``, of the bin containing the requested percentile.
    """
    total = int(hist.sum())
    if total == 0:
        return 0.0
    target = quantile / 100.0 * total
    bin_idx = int(np.searchsorted(np.cumsum(hist), target, side="left"))
    bin_idx = min(bin_idx, _HIST_BINS - 1)
    return (bin_idx + 0.5) / _HIST_BINS


def _fragment_adjacency_edges(
    fragments: NDArray, affinities: NDArray, offsets: List[Tuple[int, ...]]
) -> Dict[Tuple[int, int], NDArray]:
    """
    Build the fragment-adjacency graph's edge histograms from raw affinities.

    For every offset/channel, finds voxel pairs that straddle two different (nonzero) fragments
    and bins their affinity into a running histogram per unordered fragment-id pair, across all
    offsets at once.

    Parameters
    ----------
    fragments : NDArray
        Integer label volume, ``0`` reserved for background.

    affinities : NDArray
        ``(..., C)`` float array, ``C == len(offsets)``, matching ``fragments``'s spatial shape.

    offsets : list of tuple of int
        One signed offset per affinity channel (see ``_offset_pair_slices``).

    Returns
    -------
    edges : dict
        ``{(lo_id, hi_id): histogram}``, ``lo_id < hi_id``, ``histogram`` a ``(_HIST_BINS,)`` int64
        array.
    """
    ndim = fragments.ndim
    edges: Dict[Tuple[int, int], NDArray] = {}
    for c, offset in enumerate(offsets):
        if not any(offset):
            continue
        lo_sl, hi_sl, val_sl = _offset_pair_slices(offset, ndim)
        frag_lo = fragments[lo_sl]
        frag_hi = fragments[hi_sl]
        vals = affinities[val_sl + (c,)]

        mask = (frag_lo != frag_hi) & (frag_lo > 0) & (frag_hi > 0)
        if not np.any(mask):
            continue

        a = frag_lo[mask].astype(np.int64)
        b = frag_hi[mask].astype(np.int64)
        v = vals[mask]
        pairs = np.stack([np.minimum(a, b), np.maximum(a, b)], axis=1)

        uniq_pairs, inverse = np.unique(pairs, axis=0, return_inverse=True)
        n_pairs = len(uniq_pairs)
        bins = np.clip((v * _HIST_BINS).astype(np.int64), 0, _HIST_BINS - 1)
        flat_idx = inverse * _HIST_BINS + bins
        hist = np.bincount(flat_idx, minlength=n_pairs * _HIST_BINS).reshape(n_pairs, _HIST_BINS)

        for (p_lo, p_hi), h in zip(uniq_pairs.tolist(), hist):
            key = (p_lo, p_hi)
            if key in edges:
                edges[key] += h
            else:
                edges[key] = h.copy()
    return edges


class _UnionFind:
    """Disjoint-set over fragment ids, path-halved and merged by size."""

    __slots__ = ("parent", "size")

    def __init__(self, ids: List[int]):
        self.parent = {i: i for i in ids}
        self.size = {i: 1 for i in ids}

    def find(self, x: int) -> int:
        """Return ``x``'s current component root, path-halving along the way."""
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> int:
        """Merge the components of ``a`` and ``b`` (smaller into larger) and return the new root."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return ra


def _apply_merge(
    uf: _UnionFind, neighbors: Dict[int, Dict[int, NDArray]], ra: int, rb: int
) -> Tuple[int, List[Tuple[int, NDArray]]]:
    """
    Union ``ra``/``rb`` and fold the absorbed root's edge histograms into the surviving one.

    The absorbed root's ``neighbors`` entry is removed; each of its other neighbours gets its edge
    into the new root updated in place (summed bin-by-bin with any edge already there).

    Returns
    -------
    new_root : int

    updated : list of (neighbour_id, histogram)
        Every neighbour of the absorbed root whose edge into ``new_root`` changed value.
    """
    new_root = uf.union(ra, rb)
    old_root = rb if new_root == ra else ra

    old_neighbors = neighbors.pop(old_root, {})
    updated: List[Tuple[int, NDArray]] = []
    for nb, hist in old_neighbors.items():
        if nb == new_root:
            continue
        nb_entry = neighbors.setdefault(nb, {})
        nb_entry.pop(old_root, None)

        existing = neighbors[new_root].get(nb)
        if existing is not None:
            existing += hist
        else:
            neighbors[new_root][nb] = hist
        nb_entry[new_root] = neighbors[new_root][nb]
        updated.append((nb, neighbors[new_root][nb]))
    return new_root, updated


def _greedy_merge(
    ids: List[int],
    edges: Dict[Tuple[int, int], NDArray],
    threshold: float,
    quantile: float = _DEFAULT_QUANTILE,
    min_edge_voxels: int = 5,
    verbose: bool = True,
) -> _UnionFind:
    """
    Greedily merge fragments by descending edge score.

    Repeatedly merges the fragment pair with the highest current edge score while it is ``>=
    threshold``, using a lazily-invalidated max-heap: an edge's score changes whenever either
    endpoint merges, so heap entries are re-validated against ``neighbors`` (kept current by
    ``_apply_merge``) when popped. The first validated, sufficiently-supported pop at any point is
    the current global-max eligible edge, so the loop stops as soon as one pops below ``threshold``.

    ``min_edge_voxels`` guards against an edge with only a handful of supporting voxel-pairs (where
    even a median gives little protection against an outlier) triggering a bad, irreversible merge.

    Parameters
    ----------
    ids : list of int
        All fragment ids to consider (background/``0`` excluded by the caller).

    edges : dict
        Output of ``_fragment_adjacency_edges``.

    threshold : float
        Minimum edge score (see ``quantile``) to merge two fragments.

    quantile : float, optional
        Percentile of each edge's affinity histogram used as its score. ``50`` (default) = median.

    min_edge_voxels : int, optional
        An edge needs at least this many supporting voxel-pairs before it's eligible for merging.
        ``1`` effectively disables this.

    verbose : bool, optional
        Whether to print a one-line merge summary.

    Returns
    -------
    uf : _UnionFind
        Final component state; look up any fragment's merged instance id via ``uf.find(fragment_id)``.
    """
    uf = _UnionFind(ids)
    neighbors: Dict[int, Dict[int, NDArray]] = {i: {} for i in ids}
    heap: List[Tuple[float, int, int]] = []
    for (a, b), hist in edges.items():
        neighbors[a][b] = hist
        neighbors[b][a] = hist
        heapq.heappush(heap, (-_edge_score(hist, quantile), a, b))

    n_merges = 0
    while heap:
        neg_w, a, b = heapq.heappop(heap)
        ra, rb = uf.find(a), uf.find(b)
        if ra == rb:
            continue
        cur = neighbors.get(ra, {}).get(rb)
        if cur is None:
            continue
        if int(cur.sum()) < min_edge_voxels:
            continue
        cur_w = _edge_score(cur, quantile)
        if abs(cur_w - (-neg_w)) > 1e-9:
            heapq.heappush(heap, (-cur_w, ra, rb))
            continue
        if cur_w < threshold:
            break

        new_root, updated = _apply_merge(uf, neighbors, ra, rb)
        n_merges += 1
        for nb, hist in updated:
            heapq.heappush(heap, (-_edge_score(hist, quantile), new_root, nb))

    if verbose:
        print(
            f"Affinity agglomeration: {len(ids)} fragments, {n_merges} threshold merges "
            f"(Q{quantile:g} >= {threshold})"
        )
    return uf


def affinity_agglomeration(
    fragments: NDArray,
    affinities: NDArray,
    offsets: List[Tuple[int, ...]],
    threshold: float = 0.5,
    quantile: float = _DEFAULT_QUANTILE,
    min_edge_voxels: int = 5,
    verbose: bool = True,
) -> NDArray:
    """
    Merge an oversegmented fragment labeling into instances using predicted affinities.

    Use ``watershed_and_agglomerate_affinities`` instead if you don't already have fragments.

    Parameters
    ----------
    fragments : NDArray
        Initial oversegmentation, integer labels, ``0`` reserved for background. Must be
        boundary-safe (no fragment straddling a true instance boundary).

    affinities : NDArray
        ``(..., C)`` predicted affinities (post-sigmoid, ``[0, 1]``), ``C == len(offsets)``, spatial
        shape matching ``fragments``.

    offsets : list of tuple of int
        One signed per-axis offset per affinity channel, e.g.
        ``biapy.engine.membrane_repair._affinity_offsets``'s output.

    threshold : float, optional
        Minimum edge score (see ``quantile``) for two fragments to merge.

    quantile : float, optional
        Percentile (``0``-``100``) of each edge's affinity histogram used as its merge score.
        ``50`` (default) is the median.

    min_edge_voxels : int, optional
        Minimum voxel-pair support an edge needs before it's eligible to trigger a merge (see
        ``_greedy_merge``). ``1`` effectively disables this.

    verbose : bool, optional
        Whether to print progress/summary information.

    Returns
    -------
    instances : NDArray
        Same shape as ``fragments``, ``uint32``, background ``0``, merged fragments sharing one
        sequential instance id (``1..N``).
    """
    if affinities.shape[-1] != len(offsets):
        raise ValueError(
            f"'affinities' has {affinities.shape[-1]} channels but {len(offsets)} offsets were "
            "given -- affinity_agglomeration needs exactly one offset tuple per affinity channel."
        )
    if affinities.shape[:-1] != fragments.shape:
        raise ValueError(
            f"'affinities' spatial shape {affinities.shape[:-1]} does not match 'fragments' shape "
            f"{fragments.shape}"
        )

    ids = np.unique(fragments)
    ids = ids[ids > 0].tolist()
    if len(ids) <= 1:
        return fragments.astype(np.uint32)

    edges = _fragment_adjacency_edges(fragments, affinities, offsets)
    uf = _greedy_merge(ids, edges, threshold, quantile=quantile, min_edge_voxels=min_edge_voxels, verbose=verbose)

    id_to_root = {i: uf.find(i) for i in ids}
    root_to_label = {r: k + 1 for k, r in enumerate(sorted(set(id_to_root.values())))}
    if verbose:
        print(f"Affinity agglomeration: {len(ids)} fragments -> {len(root_to_label)} instances")

    lut = np.zeros(int(fragments.max()) + 1, dtype=np.uint32)
    for i, r in id_to_root.items():
        lut[i] = root_to_label[r]
    return lut[fragments]


def watershed_and_agglomerate_affinities(
    data: NDArray,
    offsets: List[Tuple[int, ...]],
    fragment_seed_th: float = 0.9,
    fragment_growth_th: float = 0.1,
    merge_threshold: float = 0.5,
    merge_quantile: float = _DEFAULT_QUANTILE,
    min_edge_voxels: int = 5,
    resolution: List[float] = [1.0, 1.0, 1.0],
    watershed_by_2d_slices: bool = False,
    verbose: bool = True,
) -> NDArray:
    """
    Predicted affinities -> instances, in one call: oversegment, then agglomerate.

    Step 1 calls ``watershed_by_channels``'s ``"A"``-only branch with fragment-oriented
    thresholds: ``fragment_seed_th`` high (few, confident seeds -> small, boundary-safe fragments)
    and ``fragment_growth_th`` low (fragments cover all foreground, no gaps). Only the first
    ``(z, y, x)`` affinity triple is used here.

    Step 2 (``affinity_agglomeration``) merges those fragments using every offset in ``offsets``,
    including long-range ones step 1 ignores.

    Parameters
    ----------
    data : NDArray
        ``(..., C)`` predicted affinities (post-sigmoid, ``[0, 1]``), ``C == len(offsets)``.

    offsets : list of tuple of int
        One signed per-axis offset per affinity channel (see ``affinity_agglomeration``).

    fragment_seed_th, fragment_growth_th : float, optional
        Thresholds forwarded to ``watershed_by_channels`` for fragment generation.

    merge_threshold : float, optional
        Forwarded to ``affinity_agglomeration`` as ``threshold``.

    merge_quantile : float, optional
        Forwarded to ``affinity_agglomeration`` as ``quantile``.

    min_edge_voxels : int, optional
        Forwarded to ``affinity_agglomeration``.

    resolution : list of float, optional
        Forwarded to ``watershed_by_channels``.

    watershed_by_2d_slices : bool, optional
        Forwarded to ``watershed_by_channels``.

    verbose : bool, optional
        Whether to print progress/summary information for both steps.

    Returns
    -------
    instances : NDArray
        Same spatial shape as ``data``, ``uint32``, background ``0``.
    """
    fragments = watershed_by_channels(
        data=data,
        channels=["A"],
        seed_channels=["A"],
        seed_channel_ths=[fragment_seed_th],
        topo_surface_channel="",
        growth_mask_channels=["A"],
        growth_mask_channel_ths=[fragment_growth_th],
        resolution=resolution,
        watershed_by_2d_slices=watershed_by_2d_slices,
        save_dir=None,
        verbose=verbose,
    ).astype(np.uint32)

    return affinity_agglomeration(
        fragments=fragments,
        affinities=data,
        offsets=offsets,
        threshold=merge_threshold,
        quantile=merge_quantile,
        min_edge_voxels=min_edge_voxels,
        verbose=verbose,
    )
