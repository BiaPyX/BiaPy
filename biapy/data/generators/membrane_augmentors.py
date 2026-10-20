"""
Corruption augmentors for the membrane-repair problem (PROBLEM.IMAGE_TO_IMAGE.MEMBRANE_REPAIR).

Synthetic gaps (under-segmentation), spurious bridges (over-segmentation), a skeleton-geometry
perturbation, a slice-dropout augmentor, a heavy acquisition-artifact augmentor, fake internal
membranes (e.g. mitochondria) and membrane dust around the membranes, applied directly to the raw
source channels.

All functions take and return the full ``image`` array (``(y, x, C)`` in 2D, ``(z, y, x, C)`` in
3D) and a ``membrane_idx`` physical channel index (see
``biapy.data.membrane_channels.source_channel_offsets``); ``artifact_corruption``'s blob artifact
also affects every non-membrane channel identically.
"""
import random
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    distance_transform_edt,
    find_objects,
    gaussian_filter,
    gaussian_filter1d,
    grey_dilation,
    label,
)
from scipy.special import ndtr
from skimage.draw import disk, line
from skimage.measure import regionprops
from skimage.morphology import skeletonize


def _prob_mask(n: int, prob: float) -> NDArray:
    """Boolean array of length ``n``, each entry independently ``True`` with probability ``prob``."""
    return np.random.random(n) < prob


def _iter_prob_slices(image: NDArray, ndim: int, prob: float):
    """
    Yield ``(z, slice_2d)`` for each z-slice independently selected with probability ``prob`` (3D),
    or ``(None, image)`` once with probability ``prob`` (2D, no z-slice concept).
    """
    if ndim == 2:
        if random.random() < prob:
            yield None, image
    else:
        for z in np.flatnonzero(_prob_mask(image.shape[0], prob)):
            yield int(z), image[z]


def _line_rect_clip(pos: float, direction: float, lo: float, hi: float) -> Tuple[float, float]:
    """
    ``t`` range for which ``pos + t * direction`` stays within ``[lo, hi]``. ``(-inf, inf)`` if
    ``direction`` is ~0 (the line runs parallel to this axis, so it never leaves the band).
    """
    if abs(direction) < 1e-9:
        return -np.inf, np.inf
    t0, t1 = (lo - pos) / direction, (hi - pos) / direction
    return min(t0, t1), max(t0, t1)


def _random_band_mask(
    h: int,
    w: int,
    length_range: Tuple[float, float],
    thickness_range: Tuple[float, float],
) -> NDArray:
    """
    Boolean ``(h, w)`` mask of a straight band: a line segment at a random angle and random
    position, a random fraction of that line's border-to-border chord, dilated to a random
    thickness. ``length_range=(1.0, 1.0)`` always reaches exactly from one border to the other.

    Parameters
    ----------
    h, w : int
        Height and width of the slice the band is drawn onto.

    length_range : tuple of 2 floats
        ``(min, max)`` fraction (``0``-``1``) of the line's border-to-border chord length.

    thickness_range : tuple of 2 floats
        ``(min, max)`` thickness (in pixels) of the band.

    Returns
    -------
    mask : 2D Numpy array of bool
        ``(h, w)`` band mask.
    """
    theta = random.uniform(0, np.pi)
    dy, dx = np.sin(theta), np.cos(theta)
    py, px = random.uniform(0, h - 1), random.uniform(0, w - 1)

    t_ylo, t_yhi = _line_rect_clip(py, dy, 0, h - 1)
    t_xlo, t_xhi = _line_rect_clip(px, dx, 0, w - 1)
    t_lo, t_hi = max(t_ylo, t_xlo), min(t_yhi, t_xhi)

    chord = t_hi - t_lo
    length = min(random.uniform(*length_range) * chord, chord)
    start_t = random.uniform(t_lo, t_hi - length)
    end_t = start_t + length

    r0 = int(np.clip(round(py + start_t * dy), 0, h - 1))
    c0 = int(np.clip(round(px + start_t * dx), 0, w - 1))
    r1 = int(np.clip(round(py + end_t * dy), 0, h - 1))
    c1 = int(np.clip(round(px + end_t * dx), 0, w - 1))

    rr, cc = line(r0, c0, r1, c1)
    mask = np.zeros((h, w), dtype=bool)
    mask[rr, cc] = True

    thickness = random.uniform(*thickness_range)
    radius = max(0, int(round((thickness - 1) / 2)))
    struct = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    return binary_dilation(mask, structure=struct)


def _edit_raw(raw: NDArray, mask: NDArray, mode: str, strength: float) -> NDArray:
    """
    Weaken (``"fade"``: towards the local bright level) or darken (``"dark"``: smooth dark blob) ``raw`` under a
    soft version of ``mask`` (2D ``(y, x)`` or 3D ``(z, y, x)``, smoothed in-plane only).
    """
    sig = (0, 2.0, 2.0) if mask.ndim == 3 else 2.0
    if mode == "dark":
        w = np.clip(gaussian_filter(binary_dilation(mask, iterations=2).astype(np.float32), sig), 0, 1)
        return raw * (1 - strength * w)
    w = np.clip(gaussian_filter(mask.astype(np.float32), sig), 0, 1)
    size = (1, 7, 7) if raw.ndim == 3 else (7, 7)
    bright = gaussian_filter(grey_dilation(raw, size=size), sig)
    return raw + strength * w * (bright - raw)


def _paint_random_bands(
    image: NDArray,
    membrane_idx: int,
    ndim: int,
    prob: float,
    length_range: Tuple[float, float],
    thickness_range: Tuple[float, float],
    n_lines: Tuple[int, int],
    fill_value: float,
    raw_idx: Optional[int] = None,
    raw_prob: float = 0.0,
    raw_strength_range: Tuple[float, float] = (0.5, 1.0),
    raw_modes: Tuple[str, ...] = ("dark",),
) -> NDArray:
    """
    Paint ``fill_value`` into one or more random bands across the membrane channel (see
    ``_random_band_mask``), on each z-slice independently with probability ``prob``. Shared by
    ``synthetic_gap`` (``fill_value=0``) and ``spurious_bridge`` (``fill_value=1``).

    Parameters
    ----------
    image : 3D/4D Numpy array
        ``(y, x, C)`` in 2D or ``(z, y, x, C)`` in 3D.

    membrane_idx : int
        Physical channel index of the membrane class.

    ndim : int
        Number of spatial dimensions (``2`` or ``3``).

    prob : float
        Independent probability of augmenting each z-slice (2D: the whole image).

    length_range : tuple of 2 floats
        ``(min, max)`` fraction (``0``-``1``) of the image's border-to-border extent along the
        band's angle. ``1.0`` reaches from one border to the other.

    thickness_range : tuple of 2 floats
        ``(min, max)`` thickness (in pixels) of each band.

    n_lines : tuple of 2 ints
        ``(min, max)`` number of bands drawn per augmented slice.

    fill_value : float
        Value written into the membrane channel along the band(s).

    raw_idx : int, optional
        Physical channel index of the raw image, needed for ``raw_prob``.

    raw_prob : float, optional
        Probability, per painted slice, of also editing the raw image under the band(s) with one of
        ``raw_modes`` (see ``_edit_raw``), so the raw ridge response agrees with the membrane change.

    raw_strength_range : tuple of 2 floats, optional
        ``(min, max)`` strength (``0``-``1``) of the raw edit.

    raw_modes : tuple of str, optional
        Raw edits drawn from: ``"fade"`` and/or ``"dark"``.

    Returns
    -------
    image : 3D/4D Numpy array
        ``image`` with the band(s) applied (copy).
    """
    out = image.copy()
    for z, sub in _iter_prob_slices(out, ndim, prob):
        h, w = sub.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        for _ in range(random.randint(*n_lines)):
            mask |= _random_band_mask(h, w, length_range, thickness_range)
        if mask.any():
            membrane = sub[..., membrane_idx].copy()
            membrane[mask] = fill_value
            sub[..., membrane_idx] = membrane
            if raw_prob > 0 and raw_idx is not None and random.random() < raw_prob:
                mode = random.choice(raw_modes)
                sub[..., raw_idx] = _edit_raw(sub[..., raw_idx], mask, mode, random.uniform(*raw_strength_range))
    return out


def synthetic_gap(
    image: NDArray,
    membrane_idx: int,
    ndim: int,
    prob: float = 0.5,
    length_range: Tuple[float, float] = (0.3, 1.0),
    thickness_range: Tuple[float, float] = (4, 9),
    n_lines: Tuple[int, int] = (1, 3),
    raw_idx: Optional[int] = None,
    raw_prob: float = 0.0,
    raw_strength_range: Tuple[float, float] = (0.5, 1.0),
) -> NDArray:
    """
    Black out one or more straight bands across the membrane channel (merge/under-segmentation
    error). See ``_paint_random_bands``. With ``raw_prob``, the raw membrane under the gap is also faded or
    covered by a dark blob, as where the upstream model misses real membranes (faint membranes, synapses).

    Parameters
    ----------
    image : 3D/4D Numpy array
        ``(y, x, C)`` in 2D or ``(z, y, x, C)`` in 3D.

    membrane_idx : int
        Physical channel index of the membrane class.

    ndim : int
        Number of spatial dimensions (``2`` or ``3``).

    prob : float, optional
        Independent probability of augmenting each z-slice (2D: the whole image).

    length_range : tuple of 2 floats, optional
        ``(min, max)`` fraction (``0``-``1``) of the image's border-to-border extent along the
        band's angle. ``1.0`` reaches from one border to the other.

    thickness_range : tuple of 2 floats, optional
        ``(min, max)`` thickness (in pixels) of each band.

    n_lines : tuple of 2 ints, optional
        ``(min, max)`` number of bands drawn per augmented slice.

    raw_idx, raw_prob, raw_strength_range : optional
        Raw edit under the gap (``"fade"`` or ``"dark"``), see ``_paint_random_bands``.

    Returns
    -------
    image : 3D/4D Numpy array
        ``image`` with the band(s) applied (copy).
    """
    return _paint_random_bands(
        image, membrane_idx, ndim, prob, length_range, thickness_range, n_lines, fill_value=0,
        raw_idx=raw_idx, raw_prob=raw_prob, raw_strength_range=raw_strength_range, raw_modes=("fade", "dark"),
    )


def spurious_bridge(
    image: NDArray,
    membrane_idx: int,
    ndim: int,
    prob: float = 0.3,
    length_range: Tuple[float, float] = (0.3, 1.0),
    thickness_range: Tuple[float, float] = (4, 9),
    n_lines: Tuple[int, int] = (1, 3),
    fill_value: float = 1.0,
    raw_idx: Optional[int] = None,
    raw_prob: float = 0.0,
    raw_strength_range: Tuple[float, float] = (0.2, 0.5),
) -> NDArray:
    """
    Paint one or more straight bands of fake membrane across the channel (split error). Inverse of
    ``synthetic_gap``: same band geometry, adds membrane instead of erasing it. See
    ``_paint_random_bands``.

    Parameters
    ----------
    image : 3D/4D Numpy array
        ``(y, x, C)`` in 2D or ``(z, y, x, C)`` in 3D.

    membrane_idx : int
        Physical channel index of the membrane class.

    ndim : int
        Number of spatial dimensions (``2`` or ``3``).

    prob : float, optional
        Independent probability of augmenting each z-slice (2D: the whole image).

    length_range : tuple of 2 floats, optional
        ``(min, max)`` fraction (``0``-``1``) of the image's border-to-border extent along the
        band's angle. ``1.0`` reaches from one border to the other.

    thickness_range : tuple of 2 floats, optional
        ``(min, max)`` thickness (in pixels) of each band.

    n_lines : tuple of 2 ints, optional
        ``(min, max)`` number of bands drawn per augmented slice.

    fill_value : float, optional
        Value written into the membrane channel along the band(s).

    raw_idx, raw_prob, raw_strength_range : optional
        Darken the raw image under the bridge (``"dark"``), see ``_paint_random_bands``.

    Returns
    -------
    image : 3D/4D Numpy array
        ``image`` with the bridge band(s) applied (copy).
    """
    return _paint_random_bands(
        image, membrane_idx, ndim, prob, length_range, thickness_range, n_lines, fill_value=fill_value,
        raw_idx=raw_idx, raw_prob=raw_prob, raw_strength_range=raw_strength_range, raw_modes=("dark",),
    )


def _random_blob_mask(
    h: int,
    w: int,
    size_range: Tuple[float, float],
    n_blobs_range: Tuple[int, int],
) -> NDArray:
    """
    Boolean ``(h, w)`` mask of one or more filled disks at random positions/radii. ``size_range``
    is a fraction (``0``-``1``) of ``min(h, w)``, so blob size scales with the image.
    """
    mask = np.zeros((h, w), dtype=bool)
    min_dim = min(h, w)
    for _ in range(random.randint(*n_blobs_range)):
        radius = random.uniform(*size_range) * min_dim
        center = (random.uniform(0, h - 1), random.uniform(0, w - 1))
        rr, cc = disk(center, radius, shape=(h, w))
        mask[rr, cc] = True
    return mask


def artifact_corruption(
    image: NDArray,
    membrane_idx: int,
    ndim: int,
    prob: float = 0.1,
    band_prob: float = 0.5,
    band_thickness_range: Tuple[float, float] = (50, 70),
    band_membrane_fill: float = 1.0,
    band_other_fill: float = 0.0,
    blob_size_range: Tuple[float, float] = (0.1, 0.3),
    blob_n_range: Tuple[int, int] = (1, 3),
    blob_fill: float = 0.0,
) -> NDArray:
    """
    Simulate a heavy acquisition-style artifact, on each z-slice independently with probability
    ``prob``. One artifact type is picked per augmented slice:

    - Band (probability ``band_prob``): border-to-border band at a random angle/position (see
      ``_random_band_mask``). ``band_membrane_fill`` inside the band on the membrane channel,
      ``band_other_fill`` on every other channel; untouched outside the band.
    - Blobs (otherwise): cluster of filled disks (see ``_random_blob_mask``) set to ``blob_fill``
      in every channel; untouched elsewhere.

    Parameters
    ----------
    image : 3D/4D Numpy array
        ``(y, x, C)`` in 2D or ``(z, y, x, C)`` in 3D.

    membrane_idx : int
        Physical channel index of the membrane class.

    ndim : int
        Number of spatial dimensions (``2`` or ``3``).

    prob : float, optional
        Independent probability of augmenting each z-slice (2D: the whole image).

    band_prob : float, optional
        Probability of picking the band artifact over the blob artifact when a slice is augmented.

    band_thickness_range : tuple of 2 floats, optional
        ``(min, max)`` thickness (in pixels) of the band.

    band_membrane_fill : float, optional
        Value written into the membrane channel inside the band.

    band_other_fill : float, optional
        Value written into every non-membrane channel inside the band.

    blob_size_range : tuple of 2 floats, optional
        ``(min, max)`` radius of each blob, as a fraction (``0``-``1``) of ``min(h, w)``.

    blob_n_range : tuple of 2 ints, optional
        ``(min, max)`` number of blobs drawn per augmented slice.

    blob_fill : float, optional
        Value written into every channel inside the blob(s).

    Returns
    -------
    image : 3D/4D Numpy array
        ``image`` with the artifact applied (copy).
    """
    out = image.copy()
    other_idxs = [c for c in range(image.shape[-1]) if c != membrane_idx]
    for z, sub in _iter_prob_slices(out, ndim, prob):
        h, w = sub.shape[:2]
        if random.random() < band_prob:
            band = _random_band_mask(h, w, (1.0, 1.0), band_thickness_range)
            membrane = sub[..., membrane_idx].copy()
            membrane[band] = band_membrane_fill
            sub[..., membrane_idx] = membrane
            for idx in other_idxs:
                channel = sub[..., idx].copy()
                channel[band] = band_other_fill
                sub[..., idx] = channel
        else:
            blobs = _random_blob_mask(h, w, blob_size_range, blob_n_range)
            sub[blobs] = blob_fill
    return out


def skeleton_perturbation(
    image: NDArray,
    membrane_idx: int,
    ndim: int,
    prob: float = 0.3,
    radius_range: Tuple[int, int] = (1, 2),
    threshold: float = 0.5,
    n_spurs: Tuple[int, int] = (0, 2),
    spur_length_range: Tuple[int, int] = (2, 4),
) -> NDArray:
    """
    Randomly dilate/erode the membrane channel and add short spurs, so downstream derived
    channels don't overfit to exact skeleton geometry. Applied to each z-slice independently
    with probability ``prob``.

    Parameters
    ----------
    image : 3D/4D Numpy array
        ``(y, x, C)`` in 2D or ``(z, y, x, C)`` in 3D.

    membrane_idx : int
        Physical channel index of the membrane class.

    ndim : int
        Number of spatial dimensions (``2`` or ``3``).

    prob : float, optional
        Independent probability of augmenting each z-slice (2D: the whole image).

    radius_range : tuple of 2 ints, optional
        ``(min, max)`` footprint radius (pixels) for the dilation/erosion.

    threshold : float, optional
        Threshold applied before the morphological op.

    n_spurs : tuple of 2 ints, optional
        ``(min, max)`` number of short skeleton spurs added per augmented slice.

    spur_length_range : tuple of 2 ints, optional
        ``(min, max)`` length (pixels) of each spur.

    Returns
    -------
    image : 3D/4D Numpy array
        ``image`` with the perturbation applied (copy).
    """
    out = image.copy()
    for z, sub in _iter_prob_slices(out, ndim, prob):
        h, w = sub.shape[:2]
        binary = sub[..., membrane_idx] > threshold

        radius = random.randint(*radius_range)
        struct = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
        binary = binary_dilation(binary, structure=struct) if random.random() < 0.5 else binary_erosion(binary, structure=struct)

        skel = skeletonize(binary)
        endpoints = _skeleton_endpoints(skel)
        n = random.randint(*n_spurs)
        if len(endpoints) > 0:
            for _ in range(min(n, len(endpoints))):
                r0, c0 = endpoints[random.randrange(len(endpoints))]
                length = random.randint(*spur_length_range)
                angle = random.uniform(0, 2 * np.pi)
                r1 = int(np.clip(r0 + length * np.sin(angle), 0, h - 1))
                c1 = int(np.clip(c0 + length * np.cos(angle), 0, w - 1))
                rr, cc = line(r0, c0, r1, c1)
                binary = binary.copy()
                binary[rr, cc] = True

        sub[..., membrane_idx] = binary.astype(sub.dtype)
    return out


def _skeleton_endpoints(skel: NDArray) -> list:
    """Skeleton pixels with exactly one 8-connected skeleton neighbor."""
    if not skel.any():
        return []
    padded = np.pad(skel.astype(np.int32), 1)
    total = np.zeros_like(skel, dtype=np.int32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            total += padded[1 + dy : 1 + dy + skel.shape[0], 1 + dx : 1 + dx + skel.shape[1]]
    ys, xs = np.nonzero(skel & (total == 1))
    return list(zip(ys.tolist(), xs.tolist()))


def slice_dropout(image: NDArray, droppable_idxs: Tuple[int, ...], prob: float, ndim: int = 3) -> NDArray:
    """
    Zero out each channel in ``droppable_idxs``, per z-slice in 3D, independently with probability
    ``prob`` (e.g. ``prob=0.3`` over 10 z-slices zeroes ~3 of them). In 2D there is no z-slice, so the
    whole channel is zeroed with probability ``prob``.

    If any dropped channel feeds a derived channel computed downstream, call this before that
    derivation -- otherwise the derived channel keeps the dropped channel's information intact.

    Parameters
    ----------
    image : 3D/4D Numpy array
        ``(y, x, C)`` in 2D or ``(z, y, x, C)`` in 3D.

    droppable_idxs : tuple of int
        Physical channel indices eligible to be zeroed.

    prob : float
        Independent probability of zeroing each channel (2D) or each channel's z-slice (3D).

    ndim : int, optional
        Number of spatial dimensions (``2`` or ``3``).

    Returns
    -------
    image : 3D/4D Numpy array
        ``image`` with the sampled channels/slices zeroed (copy).
    """
    if prob <= 0 or not droppable_idxs:
        return image
    out = image.copy()
    if ndim == 3:
        for idx in droppable_idxs:
            out[_prob_mask(image.shape[0], prob), ..., idx] = 0
    else:
        for idx in droppable_idxs:
            if random.random() < prob:
                out[..., idx] = 0
    return out
def _inner_distance(labels: NDArray) -> NDArray:
    """Per z-slice distance of each instance voxel to the nearest voxel with a different label (0 on background)."""
    out = np.zeros(labels.shape, dtype=np.float32)
    for z in range(labels.shape[0]):
        lab = labels[z]
        edge = lab == 0
        edge[:-1] |= lab[:-1] != lab[1:]
        edge[1:] |= lab[:-1] != lab[1:]
        edge[:, :-1] |= lab[:, :-1] != lab[:, 1:]
        edge[:, 1:] |= lab[:, :-1] != lab[:, 1:]
        out[z] = distance_transform_edt(~edge)
    return out


def ridge_membrane_response(raw: NDArray, z_sigma: float = 1.0) -> NDArray:
    """
    Standardised (``[0, 1]``) Meijering ridge response of a ``(z, y, x)`` raw volume, smoothed along z with a
    Gaussian of ``z_sigma`` slices so ridges persisting across slices stand out. Shared by
    ``internal_membranes`` and ``ridge_membranes``. Two scales (sigma 1.5-3) keep it ~5x cheaper than the
    derived 'meijering' channel with nearly the same response.
    """
    from biapy.data.membrane_channels import _normalize_percentile, meijering_ridge

    response = meijering_ridge(raw, sigma_range=(1.5, 3.0), ndim=3, n_scales=2, standardize=False)
    if raw.shape[0] > 1 and z_sigma > 0:
        response = gaussian_filter1d(response, z_sigma, axis=0)
    return _normalize_percentile(response)


def _round_footprint(radius: int) -> NDArray:
    """Boolean disk footprint of the given radius."""
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    return yy**2 + xx**2 <= radius**2


def _ridge_enclosed_objects(
    response: NDArray,
    raw: NDArray,
    labels: NDArray,
    thresholds: Tuple[float, ...],
    gap: int,
    radius_range: Tuple[float, float],
    darker: float,
    max_fraction: float = 0.5,
    min_solidity: float = 0.7,
    min_ridge_size: int = 40,
) -> NDArray:
    """
    Mito-like objects per z-slice: areas enclosed by the ridge ``response`` inside a single instance. The ridges
    of the instance boundaries are kept, so objects touching the membrane are closed by it, and the image border
    also closes them, so objects cut by the patch border are kept. For each
    threshold the ridges are closed over gaps of up to ``gap`` px and their enclosed holes kept if compact
    (``min_solidity``), of equivalent radius within ``radius_range``, smaller than ``max_fraction`` of the
    instance's cross-section (excludes the instance itself) and darker than the rest of the instance in the
    ``raw`` image by at least ``darker`` (mitochondria are darker than cytoplasm). Isolated ridge specks smaller
    than ``min_ridge_size`` px (e.g. texture inside mitochondria) are ignored so they don't fragment the interior. The union over thresholds catches each object at
    the level where its ring is closed and its interior (cristae) not fragmented.
    """
    footprint = _round_footprint(gap)
    objects = np.zeros(labels.shape, dtype=bool)
    for z in range(labels.shape[0]):
        lab = labels[z]
        lab_int = lab.astype(np.int64).ravel()
        inst_area = np.bincount(lab_int)
        inst_sum = np.bincount(lab_int, weights=raw[z].ravel())
        for thr in thresholds:
            hot, n = label(response[z] > thr)
            hot = (np.bincount(hot.ravel(), minlength=n + 1) >= min_ridge_size)[hot] & (hot > 0)
            closed = binary_closing(hot, structure=footprint) | hot
            filled = binary_fill_holes(np.pad(closed, 1, constant_values=True))[1:-1, 1:-1]
            holes, _ = label(filled & ~closed & (lab > 0))
            for p in regionprops(holes):
                sl, m = p.slice, holes[p.slice] == p.label
                ids = np.unique(lab[sl][m])
                if len(ids) != 1 or ids[0] == 0:
                    continue
                i = int(ids[0])
                r = np.sqrt(p.area / np.pi)
                if not (
                    radius_range[0] <= r <= radius_range[1]
                    and p.area <= max_fraction * inst_area[i] and p.solidity >= min_solidity
                ):
                    continue
                hole_sum = raw[z][sl][m].sum()
                rest_mean = (inst_sum[i] - hole_sum) / max(1, inst_area[i] - p.area)
                if hole_sum / p.area <= rest_mean - darker:
                    objects[z][sl][m] = True
        # Merge fragments of the same interior
        objects[z] = binary_fill_holes(binary_closing(objects[z]))
    return objects


def internal_membranes(
    image: NDArray,
    membrane_idx: int,
    raw_idx: int,
    labels: NDArray,
    ndim: int,
    prob: float = 0.5,
    n_range: Tuple[int, int] = (20, 40),
    thresholds: Tuple[float, ...] = (0.5, 0.55, 0.6, 0.65, 0.7),
    gap: int = 10,
    radius_range: Tuple[float, float] = (4, 50),
    darker: float = 0.05,
    z_range: Tuple[int, int] = (3, 8),
    thickness_range: Tuple[int, int] = (2, 4),
    keep_range: Tuple[float, float] = (0.5, 1.0),
    response: Optional[NDArray] = None,
) -> NDArray:
    """
    Add fake mitochondria membranes inside instances, tracing the real ridge rings of the raw image. Mito-like
    objects (areas enclosed by ridges inside one instance, see ``_ridge_enclosed_objects``) are detected and
    linked across z-slices; for each painted object, the ridge pixels around its outline are thinned, thickened
    and painted over a run of consecutive slices, so the fake follows the exact C or ring shape of the ridge.
    The targets are left untouched, so the model learns to ignore them.

    Parameters
    ----------
    image : 3D/4D Numpy array
        ``(y, x, C)`` in 2D or ``(z, y, x, C)`` in 3D.

    membrane_idx : int
        Physical channel index of the membrane class.

    raw_idx : int
        Physical channel index of the raw image.

    labels : 2D/3D Numpy array
        Instance labels aligned with ``image``.

    ndim : int
        Number of spatial dimensions (``2`` or ``3``).

    prob : float, optional
        Probability of augmenting the sample.

    n_range : tuple of 2 ints, optional
        ``(min, max)`` number of detected objects painted per augmented sample (all if fewer are found).

    thresholds : tuple of floats, optional
        Ridge response thresholds (``[0, 1]``) used to detect the objects. The lowest one also defines the
        painted ridge pixels.

    gap : int, optional
        Largest gap (pixels) in a ridge ring that is bridged to detect its enclosed area. Bridged gaps are not
        painted, so open (C-shaped) rings stay open.

    radius_range : tuple of 2 floats, optional
        ``(min, max)`` equivalent radius (pixels) of the enclosed area of a detected object.

    darker : float, optional
        Minimum drop of the mean raw intensity (raw in ``[0, 1]``) inside a detected object relative to the rest
        of its instance, as mitochondria are darker than cytoplasm.

    z_range : tuple of 2 ints, optional
        ``(min, max)`` number of consecutive z-slices painted per object (objects spanning fewer slices than
        the minimum are skipped).

    thickness_range : tuple of 2 ints, optional
        ``(min, max)`` thickness (pixels) of the painted outline.

    keep_range : tuple of 2 floats, optional
        ``(min, max)`` fraction of the outline kept: an angular sector around the object centre, the same in all
        its slices, is left unpainted.

    response : 3D Numpy array, optional
        Precomputed ``ridge_membrane_response`` of the raw channel, ``(z, y, x)`` (``(1, y, x)`` in 2D).

    Returns
    -------
    image : 3D/4D Numpy array
        ``image`` with the fake membranes added (copy).
    """
    if random.random() >= prob or not (labels > 0).any():
        return image
    out = image.copy()
    membrane = out[..., membrane_idx] if ndim == 3 else out[None, ..., membrane_idx]
    labels = labels if ndim == 3 else labels[None]
    raw = out[..., raw_idx] if ndim == 3 else out[None, ..., raw_idx]
    if response is None:
        response = ridge_membrane_response(raw)
    depth = labels.shape[0]
    z_min = min(z_range[0], depth)

    objects, _ = label(
        _ridge_enclosed_objects(response, raw, labels, thresholds, gap, radius_range, darker),
        structure=np.ones((3, 3, 3)),
    )
    boxes = [(i, sl) for i, sl in enumerate(find_objects(objects), start=1) if sl[0].stop - sl[0].start >= z_min]
    random.shuffle(boxes)
    hot = response > min(thresholds)
    for i, sl in boxes[: random.randint(*n_range)]:
        zs = np.arange(sl[0].start, sl[0].stop)
        run = min(random.randint(z_min, z_range[1]), len(zs))
        zs = zs[random.randint(0, len(zs) - run) :][:run]
        gap_angle = (1.0 - random.uniform(*keep_range)) * 2 * np.pi
        gap_start = random.uniform(-np.pi, np.pi)
        grow = int(round((random.randint(*thickness_range) - 1) / 2))
        margin = 6
        win = tuple(slice(max(0, s.start - margin), min(dim, s.stop + margin)) for s, dim in zip(sl[1:], labels.shape[1:]))
        for z in zs:
            obj = objects[z][win] == i
            if not obj.any():
                continue
            lab = labels[z][win]
            inst = np.bincount(lab[obj].astype(np.int64)).argmax()
            # Ridge pixels right around the object's area (a fixed band, so nearby unrelated ridges are left out)
            outline = skeletonize(binary_dilation(obj, iterations=4) & ~obj & hot[z][win])
            if gap_angle > 0:
                cy, cx = np.argwhere(obj).mean(axis=0)
                yy, xx = np.mgrid[0 : obj.shape[0], 0 : obj.shape[1]]
                outline &= ((np.arctan2(yy - cy, xx - cx) - gap_start) % (2 * np.pi)) >= gap_angle
            if grow > 0:
                outline = binary_dilation(outline, iterations=grow)
            membrane[z][win][outline & (lab == inst)] = 1
    out[..., membrane_idx] = membrane if ndim == 3 else membrane[0]
    return out


def ridge_membranes(
    image: NDArray,
    membrane_idx: int,
    raw_idx: int,
    labels: NDArray,
    ndim: int,
    prob: float = 0.3,
    n_range: Tuple[int, int] = (6, 12),
    box_z_range: Tuple[int, int] = (8, 24),
    box_xy_range: Tuple[int, int] = (48, 160),
    threshold_range: Tuple[float, float] = (0.45, 0.6),
    z_range: Tuple[int, int] = (3, 8),
    thickness_range: Tuple[int, int] = (1, 3),
    z_sigma: float = 1.0,
    min_area: float = 20,
    response: Optional[NDArray] = None,
) -> NDArray:
    """
    Add fake membranes inside instances where the raw image looks like a membrane (e.g. mitochondria or other
    organelles), within random 3D boxes. The ridge response of the raw channel (see
    ``ridge_membrane_response``) is thresholded inside instances, and its 3D components spanning ``z_range``
    slices are painted as lines of ``thickness_range`` pixels. The targets are left untouched.

    Parameters
    ----------
    image : 3D/4D Numpy array
        ``(y, x, C)`` in 2D or ``(z, y, x, C)`` in 3D.

    membrane_idx : int
        Physical channel index of the membrane class.

    raw_idx : int
        Physical channel index of the raw image.

    labels : 2D/3D Numpy array
        Instance labels aligned with ``image``.

    ndim : int
        Number of spatial dimensions (``2`` or ``3``).

    prob : float, optional
        Probability of augmenting the sample.

    n_range : tuple of 2 ints, optional
        ``(min, max)`` number of boxes per augmented sample.

    box_z_range, box_xy_range : tuple of 2 ints, optional
        ``(min, max)`` box size in z and in y/x.

    threshold_range : tuple of 2 floats, optional
        ``(min, max)`` threshold on the standardised ridge response.

    z_range : tuple of 2 ints, optional
        ``(min, max)`` number of z-slices a ridge component must span to be painted.

    thickness_range : tuple of 2 ints, optional
        ``(min, max)`` thickness (pixels) of the painted lines.

    z_sigma : float, optional
        Gaussian sigma (slices) used to smooth the ridge response along z. ``0`` disables it. Ignored if
        ``response`` is given.

    min_area : float, optional
        Minimum mean area (pixels per z-slice) of a ridge component; removes small specks.

    response : 3D Numpy array, optional
        Precomputed ``ridge_membrane_response`` of the raw channel, ``(z, y, x)`` (``(1, y, x)`` in 2D).

    Returns
    -------
    image : 3D/4D Numpy array
        ``image`` with the fake membranes added (copy).
    """
    if random.random() >= prob or not (labels > 0).any():
        return image
    out = image.copy()
    membrane = out[..., membrane_idx] if ndim == 3 else out[None, ..., membrane_idx]
    labels = labels if ndim == 3 else labels[None]
    if response is None:
        response = ridge_membrane_response(out[..., raw_idx] if ndim == 3 else out[None, ..., raw_idx], z_sigma)
    inner_dist = _inner_distance(labels)
    depth, h, w = labels.shape
    z_min, z_max = min(z_range[0], depth), z_range[1]

    for _ in range(random.randint(*n_range)):
        bz = min(random.randint(*box_z_range), depth)
        by, bx = min(random.randint(*box_xy_range), h), min(random.randint(*box_xy_range), w)
        z0, y0, x0 = random.randint(0, depth - bz), random.randint(0, h - by), random.randint(0, w - bx)
        box = (slice(z0, z0 + bz), slice(y0, y0 + by), slice(x0, x0 + bx))

        hot = (response[box] > random.uniform(*threshold_range)) & (inner_dist[box] > 2)
        comps, n = label(hot, structure=np.ones((3, 3, 3)))
        if n == 0:
            continue
        sizes = np.bincount(comps.ravel())
        keep = np.zeros(n + 1, dtype=bool)
        for i, sl in enumerate(find_objects(comps), start=1):
            zext = sl[0].stop - sl[0].start
            keep[i] = z_min <= zext <= z_max and sizes[i] / zext >= min_area
        hot = keep[comps]
        grow = int(round((random.randint(*thickness_range) - 1) / 2))
        sub_mem, sub_lab = membrane[box], labels[box]
        for z in range(bz):
            if hot[z].any():
                lines = skeletonize(hot[z])
                if grow > 0:
                    lines = binary_dilation(lines, iterations=grow)
                sub_mem[z][lines & (sub_lab[z] > 0)] = 1

    out[..., membrane_idx] = membrane if ndim == 3 else membrane[0]
    return out


def _uniform_noise(shape: Tuple[int, ...], sigma) -> NDArray:
    """Gaussian-smoothed white noise (``sigma`` px, scalar or per axis) mapped to a uniform ``[0, 1]`` distribution."""
    noise = gaussian_filter(np.random.standard_normal(shape).astype(np.float32), sigma)
    return ndtr(noise / max(float(noise.std()), 1e-6))


def _dust_mask(
    mem: NDArray,
    n_regions: Tuple[int, int],
    radius_range: Tuple[float, float],
    z_radius_range: Tuple[float, float],
    density_range: Tuple[float, float],
    decay_range: Tuple[float, float],
    cloud_sigma_range: Tuple[float, float],
    speck_sigma_range: Tuple[float, float],
    noise_z_sigma_range: Tuple[float, float],
    speck_density: float,
    ring_density: float,
) -> NDArray:
    """Dust of ``membrane_dust`` for a ``(z, y, x)`` membrane mask; ``z`` is 1 in the per-slice mode."""
    d, h, w = mem.shape
    dist = np.stack([distance_transform_edt(~m) if m.any() else np.full((h, w), np.inf) for m in mem])
    seeds = np.argwhere(mem)
    zz, yy, xx = np.ogrid[:d, :h, :w]
    near_p = np.zeros((d, h, w), dtype=np.float32)
    far_p = np.zeros((d, h, w), dtype=np.float32)
    for _ in range(random.randint(*n_regions)):
        cz, cy, cx = seeds[random.randrange(len(seeds))]
        radius = random.uniform(*radius_range)
        density = random.uniform(*density_range)
        if d > 1:
            rz = random.uniform(*z_radius_range)
            # Only within 3 sigma of the centre (negligible beyond)
            box = (
                slice(max(0, int(cz - 3 * rz)), int(cz + 3 * rz) + 1),
                slice(max(0, int(cy - 3 * radius)), int(cy + 3 * radius) + 1),
                slice(max(0, int(cx - 3 * radius)), int(cx + 3 * radius) + 1),
            )
            env = density * np.exp(
                -((yy[:, box[1]] - cy) ** 2 + (xx[..., box[2]] - cx) ** 2) / (2 * radius**2)
                - ((zz[box[0]] - cz) ** 2) / (2 * rz**2)
            )
        else:
            box = (slice(None),) * 3
            env = density * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * radius**2))
        decay = random.uniform(*decay_range)
        near_p[box] = np.maximum(near_p[box], env * np.exp(-dist[box] / decay))
        far_p[box] = np.maximum(far_p[box], env * np.exp(-dist[box] / (2 * decay)))

    zs = random.uniform(*noise_z_sigma_range) if d > 1 else 0
    dust = _uniform_noise((d, h, w), (zs, *(2 * [random.uniform(*cloud_sigma_range)]))) < near_p
    if speck_density > 0:
        dust |= _uniform_noise((d, h, w), (zs, *(2 * [random.uniform(*speck_sigma_range)]))) < speck_density * far_p
    if ring_density > 0:
        blobs, n = label(_uniform_noise((d, h, w), (zs, 2.0, 2.0)) < 0.05)
        if n > 0:
            # One draw per blob, so rings are kept whole (along z too)
            p = np.bincount(blobs.ravel(), weights=far_p.ravel(), minlength=n + 1)
            p /= np.maximum(np.bincount(blobs.ravel(), minlength=n + 1), 1)
            keep = np.random.random(n + 1) < ring_density * p
            keep[0] = False
            blobs = keep[blobs]
            dust |= binary_dilation(blobs, structure=np.array([[[0, 1, 0], [1, 1, 1], [0, 1, 0]]], bool)) & ~blobs
    return dust


def membrane_dust(
    image: NDArray,
    membrane_idx: int,
    ndim: int,
    prob: float = 0.5,
    n_regions: Tuple[int, int] = (2, 6),
    radius_range: Tuple[float, float] = (15, 50),
    density_range: Tuple[float, float] = (0.3, 0.8),
    decay_range: Tuple[float, float] = (2, 8),
    cloud_sigma_range: Tuple[float, float] = (1.5, 3.0),
    speck_sigma_range: Tuple[float, float] = (0.7, 1.2),
    speck_density: float = 0.3,
    ring_density: float = 0.3,
    z_radius_range: Tuple[float, float] = (0, 0),
    noise_z_sigma_range: Tuple[float, float] = (1.0, 3.0),
    raw_idx: Optional[int] = None,
    raw_prob: float = 0.0,
    raw_strength_range: Tuple[float, float] = (0.2, 0.5),
) -> NDArray:
    """
    Add membrane "dust" around the membranes: the porous clouds bleeding off the membranes, the specks and the
    small rings the upstream membrane model predicts in ambiguous areas. The targets are left untouched, so the
    model learns to ignore them.

    By default each z-slice is augmented independently with probability ``prob``, with ``n_regions`` dusty
    regions. In 3D with ``z_radius_range`` above 0, the dust is coherent across slices instead: the whole volume is
    augmented with probability ``prob``, its ``n_regions`` regions also extend along z (Gaussian of
    ``z_radius_range`` slices) and the noise is smoothed along z (``noise_z_sigma_range`` slices).

    Each region is centred on a random membrane pixel, a Gaussian envelope of ``radius_range`` px with its own
    ``density`` and ``decay``. Inside them, a pixel at distance ``d`` from the membrane (in its slice) is painted
    with probability:

    - clouds: ``density * exp(-d / decay)``, thresholding coarse smoothed noise (``cloud_sigma_range``), so the
      membrane grows into thick ragged masses with holes;
    - specks: ``speck_density * density * exp(-d / (2 * decay))``, thresholding fine smoothed noise
      (``speck_sigma_range``), isolated specks that reach further into the instances;
    - rings: ``ring_density * density * exp(-d / (2 * decay))`` per ring, outlines of small blobs.

    Membrane pixels are never removed.

    Parameters
    ----------
    image : 3D/4D Numpy array
        ``(y, x, C)`` in 2D or ``(z, y, x, C)`` in 3D.

    membrane_idx : int
        Physical channel index of the membrane class.

    ndim : int
        Number of spatial dimensions (``2`` or ``3``).

    prob : float, optional
        Independent probability of augmenting each z-slice (2D: the whole image; z-coherent mode: the volume).

    n_regions : tuple of 2 ints, optional
        ``(min, max)`` number of dusty regions per augmented slice (z-coherent mode: per volume).

    radius_range : tuple of 2 floats, optional
        ``(min, max)`` radius (Gaussian sigma, pixels) of each dusty region.

    density_range : tuple of 2 floats, optional
        ``(min, max)`` probability (``0``-``1``) of painting a cloud pixel right next to the membrane, per region.

    decay_range : tuple of 2 floats, optional
        ``(min, max)`` distance (pixels) from the membrane over which the cloud painting probability drops by
        ``e``, per region (twice that for specks and rings).

    cloud_sigma_range, speck_sigma_range : tuple of 2 floats, optional
        ``(min, max)`` smoothing (pixels) of the noise the clouds/specks are thresholded from; larger means
        bigger blobs.

    speck_density, ring_density : float, optional
        Painting probability of specks/rings relative to the clouds. ``0`` disables them.

    z_radius_range : tuple of 2 floats, optional
        ``(min, max)`` z-extent (Gaussian sigma, slices) of each region. ``(0, 0)``: per-slice dust.

    noise_z_sigma_range : tuple of 2 floats, optional
        ``(min, max)`` smoothing (slices) of the noise along z in the z-coherent mode.

    raw_idx : int, optional
        Physical channel index of the raw image, needed for ``raw_prob``.

    raw_prob : float, optional
        Probability, per augmented slice (z-coherent mode: volume), of also darkening the raw image under the dust
        (see ``_edit_raw``), as the upstream model's false membranes sit on darker texture.

    raw_strength_range : tuple of 2 floats, optional
        ``(min, max)`` darkening strength (``0``-``1``).

    Returns
    -------
    image : 3D/4D Numpy array
        ``image`` with the dust added (copy).
    """
    opts = (n_regions, radius_range, z_radius_range, density_range, decay_range, cloud_sigma_range,
            speck_sigma_range, noise_z_sigma_range, speck_density, ring_density)
    out = image.copy()
    if ndim == 3 and z_radius_range[1] > 0:
        mem = out[..., membrane_idx] > 0.5
        if random.random() < prob and mem.any() and not mem.all():
            dust = _dust_mask(mem, *opts)
            out[..., membrane_idx][dust & ~mem] = 1
            if raw_prob > 0 and raw_idx is not None and random.random() < raw_prob:
                out[..., raw_idx] = _edit_raw(out[..., raw_idx], dust & ~mem, "dark", random.uniform(*raw_strength_range))
        return out
    for z, sub in _iter_prob_slices(out, ndim, prob):
        membrane = sub[..., membrane_idx]
        mem = membrane > 0.5
        if not mem.any() or mem.all():
            continue
        dust = _dust_mask(mem[None], *opts)[0]
        if dust.any():
            membrane = membrane.copy()
            membrane[dust & ~mem] = 1
            sub[..., membrane_idx] = membrane
            if raw_prob > 0 and raw_idx is not None and random.random() < raw_prob:
                sub[..., raw_idx] = _edit_raw(sub[..., raw_idx], dust & ~mem, "dark", random.uniform(*raw_strength_range))
    return out
