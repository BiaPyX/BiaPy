"""
Corruption augmentors for the membrane-repair problem (PROBLEM.IMAGE_TO_IMAGE.MEMBRANE_REPAIR).

Synthetic gaps (under-segmentation), spurious bridges/islands (over-segmentation), a
skeleton-geometry perturbation, and a slice-dropout augmentor, applied directly to the membrane
(and, where noted, mito) source channel(s), per z-slice.

All functions take and return the full ``image`` array (``(y, x, C)`` in 2D, ``(z, y, x, C)`` in
3D) and a ``membrane_idx``/``mito_idx`` physical channel index (see
``biapy.data.membrane_channels.source_channel_offsets``).
"""
import random
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_dilation, binary_erosion
from skimage.draw import disk, line
from skimage.morphology import skeletonize
from skimage.segmentation import find_boundaries


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


def _geodesic_segment(binary_mask: NDArray, length_px: float) -> NDArray:
    """
    Grow a connected patch of ``binary_mask`` of geodesic radius ``length_px`` from a random seed.

    Used to select "a stretch of membrane/skeleton/boundary around a random point" without
    needing an explicit graph representation: repeatedly dilating a seed pixel and re-intersecting
    with ``binary_mask`` grows exactly along the mask's connected structure.

    Parameters
    ----------
    binary_mask : 2D Numpy array of bool
        Structure to walk along (e.g. a skeleton or a boundary mask).

    length_px : float
        Approximate geodesic radius (in pixels) of the returned patch.

    Returns
    -------
    patch : 2D Numpy array of bool
        Subset of ``binary_mask`` within ``length_px`` geodesic steps of a random seed pixel.
        All-``False`` if ``binary_mask`` is empty.
    """
    ys, xs = np.nonzero(binary_mask)
    if len(ys) == 0:
        return np.zeros_like(binary_mask, dtype=bool)
    i = random.randrange(len(ys))
    patch = np.zeros_like(binary_mask, dtype=bool)
    patch[ys[i], xs[i]] = True
    struct = np.ones((3, 3), dtype=bool)
    for _ in range(max(1, int(round(length_px)))):
        grown = binary_dilation(patch, structure=struct) & binary_mask
        if grown.sum() == patch.sum():
            break
        patch = grown
    return patch


def _random_band_mask(
    h: int,
    w: int,
    length_range: Tuple[float, float],
    thickness_range: Tuple[float, float],
) -> NDArray:
    """
    Boolean ``(h, w)`` mask of a straight band: a line segment at a random angle and random
    position, at a random fraction of the image's border-to-border extent along that angle,
    dilated to a random thickness.

    A fraction of ``1.0`` reproduces the chord of the image along the sampled angle -- e.g. a
    perfectly horizontal/vertical line spans the full width/height of the image; an oblique line
    spans the (longer) corner-to-corner extent along its own angle.

    Parameters
    ----------
    h, w : int
        Height and width of the slice the band is drawn onto.

    length_range : tuple of 2 floats
        ``(min, max)`` fraction (``0``-``1``) of the image's border-to-border extent along the
        sampled angle.

    thickness_range : tuple of 2 floats
        ``(min, max)`` thickness (in pixels) of the band.

    Returns
    -------
    mask : 2D Numpy array of bool
        ``(h, w)`` band mask.
    """
    theta = random.uniform(0, np.pi)
    dy, dx = np.sin(theta), np.cos(theta)

    eps = 1e-9
    dist_y = (h / 2) / abs(dy) if abs(dy) > eps else np.inf
    dist_x = (w / 2) / abs(dx) if abs(dx) > eps else np.inf
    chord = 2 * min(dist_y, dist_x)

    length = random.uniform(*length_range) * chord
    cy, cx = random.uniform(0, h - 1), random.uniform(0, w - 1)
    r0 = int(np.clip(cy - length / 2 * dy, 0, h - 1))
    c0 = int(np.clip(cx - length / 2 * dx, 0, w - 1))
    r1 = int(np.clip(cy + length / 2 * dy, 0, h - 1))
    c1 = int(np.clip(cx + length / 2 * dx, 0, w - 1))

    rr, cc = line(r0, c0, r1, c1)
    mask = np.zeros((h, w), dtype=bool)
    mask[rr, cc] = True

    thickness = random.uniform(*thickness_range)
    radius = max(0, int(round((thickness - 1) / 2)))
    struct = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    return binary_dilation(mask, structure=struct)


def synthetic_gap(
    image: NDArray,
    membrane_idx: int,
    ndim: int,
    prob: float = 0.5,
    length_range: Tuple[float, float] = (0.1, 0.5),
    thickness_range: Tuple[float, float] = (2, 8),
    n_lines: Tuple[int, int] = (1, 3),
) -> NDArray:
    """
    Black out one or more straight bands across the membrane channel to synthesize a merge/
    under-segmentation error affecting many membrane instances at once, on each z-slice
    independently with probability ``prob``.

    Each band is a line segment at a random angle and random position (see
    ``_random_band_mask``), dilated to a random thickness.

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
    out = image.copy()
    for z, sub in _iter_prob_slices(out, ndim, prob):
        h, w = sub.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        for _ in range(random.randint(*n_lines)):
            mask |= _random_band_mask(h, w, length_range, thickness_range)
        if mask.any():
            membrane = sub[..., membrane_idx].copy()
            membrane[mask] = 0
            sub[..., membrane_idx] = membrane
    return out


def spurious_bridge(
    image: NDArray,
    membrane_idx: int,
    ndim: int,
    prob: float = 0.3,
    length_range: Tuple[float, float] = (3, 15),
    line_width: int = 1,
    fill_value: float = 1.0,
) -> NDArray:
    """
    Paint a short spurious bridge of fake membrane between two nearby points (a split error at a
    multi-instance junction), on each z-slice independently with probability ``prob``.

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
        ``(min, max)`` length (in pixels) of the bridge.

    line_width : int, optional
        Dilation radius (pixels) applied to the drawn line.

    fill_value : float, optional
        Value written into the membrane channel along the bridge.

    Returns
    -------
    image : 3D/4D Numpy array
        ``image`` with the bridge applied (copy).
    """
    out = image.copy()
    struct = np.ones((2 * line_width + 1, 2 * line_width + 1), dtype=bool)
    for z, sub in _iter_prob_slices(out, ndim, prob):
        h, w = sub.shape[:2]
        length = random.uniform(*length_range)
        angle = random.uniform(0, 2 * np.pi)
        r0, c0 = random.randrange(h), random.randrange(w)
        r1 = int(np.clip(r0 + length * np.sin(angle), 0, h - 1))
        c1 = int(np.clip(c0 + length * np.cos(angle), 0, w - 1))
        rr, cc = line(r0, c0, r1, c1)
        mask = np.zeros((h, w), dtype=bool)
        mask[rr, cc] = True
        mask = binary_dilation(mask, structure=struct)
        membrane = sub[..., membrane_idx].copy()
        membrane[mask] = fill_value
        sub[..., membrane_idx] = membrane
    return out


def spurious_island(
    image: NDArray,
    membrane_idx: int,
    ndim: int,
    prob: float = 0.3,
    size_range: Tuple[float, float] = (2, 6),
    fill_value: float = 1.0,
) -> NDArray:
    """
    Paint a small spurious island of fake membrane (split error inside an instance), on each
    z-slice independently with probability ``prob``.

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

    size_range : tuple of 2 floats, optional
        ``(min, max)`` radius (in pixels) of the island.

    fill_value : float, optional
        Value written into the membrane channel inside the island.

    Returns
    -------
    image : 3D/4D Numpy array
        ``image`` with the island applied (copy).
    """
    out = image.copy()
    for z, sub in _iter_prob_slices(out, ndim, prob):
        h, w = sub.shape[:2]
        radius = random.uniform(*size_range)
        center = (random.randrange(h), random.randrange(w))
        rr, cc = disk(center, radius, shape=(h, w))
        membrane = sub[..., membrane_idx].copy()
        membrane[rr, cc] = fill_value
        sub[..., membrane_idx] = membrane
    return out


def mito_border_erasure(
    image: NDArray,
    membrane_idx: int,
    mito_idx: int,
    ndim: int,
    prob: float = 0.3,
    length_range: Tuple[float, float] = (5, 20),
    threshold: float = 0.5,
    erase_radius: int = 1,
) -> NDArray:
    """
    Erase membrane along a random stretch of the mito channel's boundary, on each z-slice
    independently with probability ``prob``.

    Parameters
    ----------
    image : 3D/4D Numpy array
        ``(y, x, C)`` in 2D or ``(z, y, x, C)`` in 3D.

    membrane_idx : int
        Physical channel index of the membrane class.

    mito_idx : int
        Physical channel index of the mito class.

    ndim : int
        Number of spatial dimensions (``2`` or ``3``).

    prob : float, optional
        Independent probability of augmenting each z-slice (2D: the whole image).

    length_range : tuple of 2 floats, optional
        ``(min, max)`` geodesic length (in pixels) of the erased boundary stretch.

    threshold : float, optional
        Threshold applied to the (soft) mito map before finding its boundary.

    erase_radius : int, optional
        Dilation radius (pixels) applied to the erased stretch, so it reaches membrane pixels
        immediately adjacent to (not just exactly on) the mito boundary.

    Returns
    -------
    image : 3D/4D Numpy array
        ``image`` with the erasure applied (copy).
    """
    out = image.copy()
    struct = np.ones((2 * erase_radius + 1, 2 * erase_radius + 1), dtype=bool)
    for z, sub in _iter_prob_slices(out, ndim, prob):
        mito = sub[..., mito_idx] > threshold
        if not mito.any():
            continue
        boundary = find_boundaries(mito.astype(np.uint8), mode="outer")
        if not boundary.any():
            continue
        length = random.uniform(*length_range)
        erase = _geodesic_segment(boundary, length)
        if erase.any():
            erase = binary_dilation(erase, structure=struct)
            membrane = sub[..., membrane_idx].copy()
            membrane[erase] = 0
            sub[..., membrane_idx] = membrane
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
