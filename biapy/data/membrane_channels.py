"""
Derived input channels for the membrane-repair problem (PROBLEM.IMAGE_TO_IMAGE.MEMBRANE_REPAIR).

Given the raw, on-disk "source" class channels (a binary membrane mask and, optionally, further
GMM classes such as "mito"), this module computes the canonical, dataset-agnostic channels the
repair network actually consumes:

- A clamped Euclidean distance transform of the per-slice membrane skeleton (``skeleton_dt``),
  which is dense (unlike the ~99%-zero skeleton itself) and thickness-/staining-independent.
- A Hessian-eigenvalue-based "blobness" response (``hessian_blob``), which separates laminar
  membrane (one dominant curvature) from dense blob-like structures (mito/synapse/vesicles),
  independent of any GMM cluster-identity alignment.
- A standardised, multi-scale Meijering ridge response (``meijering``), a complementary ridge cue.

All channels are derived from the class maps only -- never from the raw image -- and are always
recomputed from the (possibly augmented/corrupted) source channels rather than warped directly,
so a geometric transform or a corruption augmentor can never leave a stale derived field behind.
"""
from typing import Dict, List, Sequence, Tuple

import edt
import numpy as np
from numpy.typing import NDArray
from skimage.filters import meijering
from skimage.feature import hessian_matrix
from skimage.morphology import skeletonize


def _iter_slices(volume: NDArray, ndim: int, per_slice: bool):
    """
    Yield either the ``(z, slice_2d)`` pairs of a 3D volume or a single ``(None, volume)`` pair.

    Parameters
    ----------
    volume : 2D/3D Numpy array
        Spatial-only array, i.e. ``(y, x)`` in 2D or ``(z, y, x)`` in 3D (no channel axis).

    ndim : int
        Number of spatial dimensions of the problem (``2`` or ``3``).

    per_slice : bool
        Whether a 3D volume should be walked one z-slice at a time. Ignored when ``ndim == 2``.

    Yields
    ------
    z : int or None
        Slice index (3D, ``per_slice=True``) or ``None`` (2D, or 3D processed as a whole).

    slice_2d : 2D/3D Numpy array
        The array to process.
    """
    if ndim == 2 or not per_slice:
        yield None, volume
    else:
        for z in range(volume.shape[0]):
            yield z, volume[z]


def _normalize_percentile(response: NDArray, low: float = 1.0, high: float = 99.0) -> NDArray:
    """Rescale ``response`` to ``[0, 1]`` using percentile clipping (robust to outliers)."""
    lo, hi = np.percentile(response, [low, high])
    if hi <= lo:
        return np.zeros_like(response, dtype=np.float32)
    out = (response.astype(np.float32) - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def clamped_skeleton_dt(
    membrane: NDArray,
    clamp_px: int = 10,
    ndim: int = 3,
    resolution: Sequence[float] = (1.0, 1.0, 1.0),
    per_slice: bool = True,
    threshold: float = 0.5,
) -> NDArray:
    """
    Compute the clamped, normalized Euclidean distance transform of the membrane skeleton.

    The membrane mask is skeletonized to a 1-pixel-wide line, and the Euclidean distance transform
    of its complement is taken (dense everywhere, unlike the ~99%-zero skeleton), clamped to
    ``clamp_px`` canonical pixels and normalized to ``[0, 1]`` (``0`` on the skeleton, ``1`` at
    ``clamp_px`` or further).

    Parameters
    ----------
    membrane : 2D/3D Numpy array
        Binary membrane mask (``0``/``1``), ``(y, x)`` in 2D or ``(z, y, x)`` in 3D.

    clamp_px : int, optional
        Distance (in canonical pixels) at which the transform is clamped.

    ndim : int, optional
        Number of spatial dimensions of the problem (``2`` or ``3``).

    resolution : sequence of float, optional
        Physical voxel spacing, ``(y, x)`` in 2D or ``(z, y, x)`` in 3D, used as the ``edt``
        anisotropy so the clamp means a consistent physical distance across datasets.

    per_slice : bool, optional
        Whether to skeletonize/derive the DT per z-slice (2D skeleton within a 3D stack) rather
        than on the full 3D volume. Ignored when ``ndim == 2``.

    threshold : float, optional
        Threshold applied before skeletonization.

    Returns
    -------
    dt : 2D/3D Numpy array of float32
        Same spatial shape as ``membrane``, values in ``[0, 1]``.
    """
    binary = membrane > threshold
    dt = np.zeros(membrane.shape, dtype=np.float32)
    anisotropy_2d = tuple(resolution[-2:])
    anisotropy_3d = tuple(resolution[-3:])

    for z, sub in _iter_slices(binary, ndim, per_slice):
        skel = skeletonize(sub)
        aniso = anisotropy_2d if (ndim == 2 or per_slice) else anisotropy_3d
        sub_dt = edt.edt(~skel, anisotropy=aniso, parallel=1)
        if z is None:
            dt = sub_dt
        else:
            dt[z] = sub_dt

    dt = np.clip(dt, 0.0, float(clamp_px)) / float(clamp_px)
    return dt.astype(np.float32)


def hessian_blobness(
    channel_map: NDArray,
    sigma_range: Tuple[float, float] = (1.0, 3.0),
    ndim: int = 3,
    n_scales: int = 3,
) -> NDArray:
    """
    Multi-scale, scale-normalized determinant-of-Hessian "blobness" response.

    Separates laminar membrane (Hessian dominated by one large-magnitude eigenvalue, so the
    determinant is small/negative) from dense blob-like structures such as mitochondria, synapses
    and vesicles (both eigenvalues large with the same sign, so the determinant is large and
    positive). Computed per z-slice in 3D so it stays a purely 2D, alignment-free shape cue.

    Parameters
    ----------
    channel_map : 2D/3D Numpy array
        Class map to probe (typically the membrane channel), ``(y, x)`` in 2D or ``(z, y, x)``
        in 3D.

    sigma_range : tuple of 2 floats, optional
        ``(min, max)`` Gaussian scales probed for the Hessian.

    ndim : int, optional
        Number of spatial dimensions of the problem (``2`` or ``3``).

    n_scales : int, optional
        Number of scales sampled (log-spaced) within ``sigma_range``.

    Returns
    -------
    blobness : 2D/3D Numpy array of float32
        Same spatial shape as ``channel_map``, values in ``[0, 1]``.
    """
    sigmas = np.geomspace(sigma_range[0], sigma_range[1], num=max(1, n_scales))
    response = np.zeros(channel_map.shape, dtype=np.float32)

    for z, sub in _iter_slices(channel_map, ndim, per_slice=True):
        best = np.full(sub.shape, -np.inf, dtype=np.float32)
        for sigma in sigmas:
            hrr, hrc, hcc = hessian_matrix(sub, sigma=float(sigma), order="rc", use_gaussian_derivatives=False)
            det = (hrr * hcc - hrc**2) * (float(sigma) ** 4)  # gamma-normalized (Lindeberg)
            best = np.maximum(best, det.astype(np.float32))
        best = np.clip(best, 0.0, None)
        if z is None:
            response = best
        else:
            response[z] = best

    return _normalize_percentile(response)


def meijering_ridge(
    channel_map: NDArray,
    sigma_range: Tuple[float, float] = (1.0, 4.0),
    ndim: int = 3,
    n_scales: int = 4,
    standardize: bool = True,
) -> NDArray:
    """
    Standardised, multi-scale Meijering ridge response.

    Helps discriminate laminar membrane from solid artifacts; computed per z-slice in 3D. Does not
    help with blob-shaped structures (synapses), so it complements rather than replaces
    ``hessian_blobness``.

    Parameters
    ----------
    channel_map : 2D/3D Numpy array
        Class map to probe (typically the membrane channel), ``(y, x)`` in 2D or ``(z, y, x)``
        in 3D.

    sigma_range : tuple of 2 floats, optional
        ``(min, max)`` Gaussian scales probed for the ridge filter.

    ndim : int, optional
        Number of spatial dimensions of the problem (``2`` or ``3``).

    n_scales : int, optional
        Number of scales sampled (linearly) within ``sigma_range``.

    standardize : bool, optional
        Whether to percentile-normalize the response to ``[0, 1]`` for cross-dataset
        comparability.

    Returns
    -------
    ridge : 2D/3D Numpy array of float32
        Same spatial shape as ``channel_map``.
    """
    sigmas = np.linspace(sigma_range[0], sigma_range[1], num=max(1, n_scales))
    response = np.zeros(channel_map.shape, dtype=np.float32)

    for z, sub in _iter_slices(channel_map, ndim, per_slice=True):
        ridge = meijering(sub, sigmas=sigmas, black_ridges=False)
        if z is None:
            response = ridge.astype(np.float32)
        else:
            response[z] = ridge

    if standardize:
        response = _normalize_percentile(response)
    return response.astype(np.float32)


_DERIVED_CHANNEL_FNS = {
    "skeleton_dt": clamped_skeleton_dt,
    "hessian_blob": hessian_blobness,
    "meijering": meijering_ridge,
}


def source_channel_offsets(source_channels: List[str], derived_channels: List[str]) -> Dict[str, int]:
    """
    Map every source and derived channel name to its physical index in the assembled X stack.

    The assembled stack is always ``[*source_channels, *derived_channels]``.

    Parameters
    ----------
    source_channels : list of str
        Ordered raw, on-disk channel names (e.g. ``["membrane"]`` or ``["membrane", "mito"]``).

    derived_channels : list of str
        Ordered derived channel names (e.g. ``["skeleton_dt", "hessian_blob", "meijering"]``).

    Returns
    -------
    offsets : dict of str to int
        Channel name -> physical index in the assembled ``[*source, *derived]`` stack.
    """
    offsets = {}
    for i, name in enumerate(source_channels):
        offsets[name] = i
    base = len(source_channels)
    for i, name in enumerate(derived_channels):
        offsets[name] = base + i
    return offsets


def derive_membrane_input_channels(
    source_stack: NDArray,
    source_channels: List[str],
    derived_channels: List[str],
    derived_channels_extra_opts: Dict[str, Dict],
    ndim: int = 3,
    resolution: Sequence[float] = (1.0, 1.0, 1.0),
) -> NDArray:
    """
    Assemble the full network input by appending derived channels to the raw source channels.

    Parameters
    ----------
    source_stack : 3D/4D Numpy array
        Raw source channels, ``(y, x, len(source_channels))`` in 2D or
        ``(z, y, x, len(source_channels))`` in 3D. Channel 0 is always the binary membrane mask.

    source_channels : list of str
        Ordered raw channel names, matching ``source_stack``'s channel axis.

    derived_channels : list of str
        Ordered derived channel names to compute and append (see ``_DERIVED_CHANNEL_FNS`` for
        the supported options: ``"skeleton_dt"``, ``"hessian_blob"``, ``"meijering"``).

    derived_channels_extra_opts : dict of str to dict
        Per-channel options, e.g. ``{"skeleton_dt": {"clamp_px": 10}}``.

    ndim : int, optional
        Number of spatial dimensions of the problem (``2`` or ``3``).

    resolution : sequence of float, optional
        Physical voxel spacing, ``(y, x)`` in 2D or ``(z, y, x)`` in 3D.

    Returns
    -------
    stack : 3D/4D Numpy array of float32
        ``source_stack`` with ``len(derived_channels)`` extra channels appended, i.e.
        ``(..., len(source_channels) + len(derived_channels))``.
    """
    assert source_stack.shape[-1] == len(source_channels), (
        f"source_stack has {source_stack.shape[-1]} channels but {len(source_channels)} "
        f"source_channels were given"
    )
    membrane = source_stack[..., 0]

    derived = []
    for name in derived_channels:
        opts = dict(derived_channels_extra_opts.get(name, {}))
        if name == "skeleton_dt":
            chan = clamped_skeleton_dt(membrane, ndim=ndim, resolution=resolution, **opts)
        elif name in ("hessian_blob", "meijering"):
            chan = _DERIVED_CHANNEL_FNS[name](membrane, ndim=ndim, **opts)
        else:
            raise ValueError(f"Unknown derived channel '{name}'")
        derived.append(np.expand_dims(chan.astype(np.float32), -1))

    if not derived:
        return source_stack.astype(np.float32)
    return np.concatenate([source_stack.astype(np.float32)] + derived, axis=-1)
