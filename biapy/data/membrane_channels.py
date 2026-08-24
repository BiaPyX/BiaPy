"""
Derived input channels for the membrane-repair problem (PROBLEM.IMAGE_TO_IMAGE.MEMBRANE_REPAIR).

SOURCE_CHANNELS (binary membrane/GMM-class maps, plus optionally "raw") are never fed to the
model directly -- only these derived channels are:

- ``skeleton_dt``: clamped EDT of the per-slice membrane skeleton.
- ``hessian_blob``: Hessian-eigenvalue "blobness" response.
- ``meijering``: standardised, multi-scale Meijering ridge response.

Channels are resolved by name, not position: ``skeleton_dt``/``hessian_blob`` need "membrane" in
SOURCE_CHANNELS, ``meijering`` needs "raw". SOURCE_CHANNELS may be e.g. ``["membrane"]``,
``["raw"]``, ``["membrane", "raw"]`` or ``["raw", "membrane"]``.
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

    Computed per z-slice in 3D. Must be run on the raw intensity image, not the binary membrane
    mask (which would just reproduce the mask's own information).

    Parameters
    ----------
    channel_map : 2D/3D Numpy array
        Raw intensity image, ``(y, x)`` in 2D or ``(z, y, x)`` in 3D. Must NOT be the binary
        membrane mask.

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
        ridge = meijering(sub, sigmas=sigmas, black_ridges=True)
        if z is None:
            response = ridge.astype(np.float32)
        else:
            response[z] = ridge

    if standardize:
        response = _normalize_percentile(response)
    return response.astype(np.float32)


def source_channel_offsets(source_channels: List[str], derived_channels: List[str]) -> Dict[str, int]:
    """
    Map channel names to indices. Only the ``source_channels`` entries are physically meaningful
    (used by corruption augmentors on the pre-derivation working array); ``derived_channels``
    entries are just offset by ``len(source_channels)`` for a collision-free namespace, since
    source and derived channels never coexist in the same array.

    Parameters
    ----------
    source_channels : list of str
        Ordered raw, on-disk channel names (e.g. ``["membrane"]`` or ``["membrane", "raw"]``).

    derived_channels : list of str
        Ordered derived channel names (e.g. ``["skeleton_dt", "hessian_blob", "meijering"]``).

    Returns
    -------
    offsets : dict of str to int
        Channel name -> index.
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
    Compute the network input (DERIVED_CHANNELS only) from the raw source channels.

    Parameters
    ----------
    source_stack : 3D/4D Numpy array
        Raw source channels (already warped/augmented), ``(y, x, len(source_channels))`` in 2D or
        ``(z, y, x, len(source_channels))`` in 3D.

    source_channels : list of str
        Ordered raw channel names, matching ``source_stack``'s channel axis. Channels are looked
        up by name ("membrane", "raw"), not position.

    derived_channels : list of str
        Ordered derived channel names to compute. Must be non-empty. Supported: ``"skeleton_dt"``,
        ``"hessian_blob"`` (from ``"membrane"``, which must be present in ``source_channels``),
        ``"meijering"`` (from ``"raw"``, which must then be present in ``source_channels``).

    derived_channels_extra_opts : dict of str to dict
        Per-channel options, e.g. ``{"skeleton_dt": {"clamp_px": 10}}``.

    ndim : int, optional
        Number of spatial dimensions of the problem (``2`` or ``3``).

    resolution : sequence of float, optional
        Physical voxel spacing, ``(y, x)`` in 2D or ``(z, y, x)`` in 3D.

    Returns
    -------
    stack : 3D/4D Numpy array of float32
        ``(..., len(derived_channels))``. Does NOT include ``source_channels``.
    """
    assert source_stack.shape[-1] == len(source_channels), (
        f"source_stack has {source_stack.shape[-1]} channels but {len(source_channels)} "
        f"source_channels were given"
    )
    if not derived_channels:
        raise ValueError("DERIVED_CHANNELS is empty -- at least one derived channel is required.")
    membrane_idx = source_channels.index("membrane") if "membrane" in source_channels else None
    raw_idx = source_channels.index("raw") if "raw" in source_channels else None

    derived = []
    for name in derived_channels:
        opts = dict(derived_channels_extra_opts.get(name, {}))
        if name == "skeleton_dt":
            if membrane_idx is None:
                raise ValueError(
                    "'skeleton_dt' is in DERIVED_CHANNELS but SOURCE_CHANNELS has no 'membrane' "
                    f"entry (SOURCE_CHANNELS={source_channels})."
                )
            chan = clamped_skeleton_dt(source_stack[..., membrane_idx], ndim=ndim, resolution=resolution, **opts)
        elif name == "hessian_blob":
            if membrane_idx is None:
                raise ValueError(
                    "'hessian_blob' is in DERIVED_CHANNELS but SOURCE_CHANNELS has no 'membrane' "
                    f"entry (SOURCE_CHANNELS={source_channels})."
                )
            chan = hessian_blobness(source_stack[..., membrane_idx], ndim=ndim, **opts)
        elif name == "meijering":
            if raw_idx is None:
                raise ValueError(
                    "'meijering' is in DERIVED_CHANNELS but SOURCE_CHANNELS has no 'raw' entry "
                    f"(SOURCE_CHANNELS={source_channels})."
                )
            raw = source_stack[..., raw_idx]
            chan = meijering_ridge(raw, ndim=ndim, **opts)
        else:
            raise ValueError(f"Unknown derived channel '{name}'")
        derived.append(np.expand_dims(chan.astype(np.float32), -1))

    return np.concatenate(derived, axis=-1)
