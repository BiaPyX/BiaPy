"""
Corruption augmentors for the membrane-repair problem (PROBLEM.IMAGE_TO_IMAGE.MEMBRANE_REPAIR).

Synthetic gaps (under-segmentation), spurious bridges (over-segmentation), a skeleton-geometry
perturbation, a slice-dropout augmentor, and a heavy acquisition-artifact augmentor, applied
directly to the raw source channels, per z-slice.

All functions take and return the full ``image`` array (``(y, x, C)`` in 2D, ``(z, y, x, C)`` in
3D) and a ``membrane_idx`` physical channel index (see
``biapy.data.membrane_channels.source_channel_offsets``); ``artifact_corruption``'s blob artifact
also affects every non-membrane channel identically.
"""
import random
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_dilation, binary_erosion
from skimage.draw import disk, line
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


def _paint_random_bands(
    image: NDArray,
    membrane_idx: int,
    ndim: int,
    prob: float,
    length_range: Tuple[float, float],
    thickness_range: Tuple[float, float],
    n_lines: Tuple[int, int],
    fill_value: float,
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
    return out


def synthetic_gap(
    image: NDArray,
    membrane_idx: int,
    ndim: int,
    prob: float = 0.5,
    length_range: Tuple[float, float] = (0.3, 1.0),
    thickness_range: Tuple[float, float] = (4, 9),
    n_lines: Tuple[int, int] = (1, 3),
) -> NDArray:
    """
    Black out one or more straight bands across the membrane channel (merge/under-segmentation
    error). See ``_paint_random_bands``.

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

    Returns
    -------
    image : 3D/4D Numpy array
        ``image`` with the band(s) applied (copy).
    """
    return _paint_random_bands(image, membrane_idx, ndim, prob, length_range, thickness_range, n_lines, fill_value=0)


def spurious_bridge(
    image: NDArray,
    membrane_idx: int,
    ndim: int,
    prob: float = 0.3,
    length_range: Tuple[float, float] = (0.3, 1.0),
    thickness_range: Tuple[float, float] = (4, 9),
    n_lines: Tuple[int, int] = (1, 3),
    fill_value: float = 1.0,
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

    Returns
    -------
    image : 3D/4D Numpy array
        ``image`` with the bridge band(s) applied (copy).
    """
    return _paint_random_bands(
        image, membrane_idx, ndim, prob, length_range, thickness_range, n_lines, fill_value=fill_value
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
