import numpy as np
from scipy.ndimage import map_coordinates, maximum_filter1d, find_objects
from skimage.measure import label as cc_label
from skimage.segmentation import relabel_sequential
from typing import List, Optional
from numpy.typing import NDArray


def normalize99(x, eps=1e-6):
    """
    Normalize using 1st and 99th percentiles, similar to Cellpose.
    """
    x = x.astype(np.float32)
    p1, p99 = np.percentile(x, (1, 99))

    if p99 - p1 < eps:
        return np.zeros_like(x, dtype=np.float32)

    return (x - p1) / (p99 - p1)


def flows_to_cellpose_rgb(dy, dx, dz=None, z_mode="brightness"):
    """
    Convert 2D or 3D flow fields to a Cellpose-like RGB visualization.

    Parameters
    ----------
    dy : ndarray
        Vertical / y flow, shape (..., H, W) or (H, W).

    dx : ndarray
        Horizontal / x flow, same shape as dy.

    dz : ndarray, optional
        Z flow, same shape as dy and dx.
        If None, standard 2D Cellpose-style visualization is used.

    z_mode : {"brightness", "saturation", "none"}
        How to include dz when provided.

        "brightness":
            3D magnitude controls brightness.
            XY direction controls color.
            Large dz makes vectors brighter even if XY is small.

        "saturation":
            XY direction controls color.
            Vectors pointing mostly in Z become whiter/desaturated.
            This makes out-of-plane flow visually distinct.

        "none":
            Ignore dz in the color mapping, equivalent to 2D visualization.

    Returns
    -------
    rgb : uint8 ndarray
        RGB visualization with shape dy.shape + (3,).
    """
    dy = np.asarray(dy, dtype=np.float32)
    dx = np.asarray(dx, dtype=np.float32)

    if dy.shape != dx.shape:
        raise ValueError(f"dy and dx must have the same shape, got {dy.shape} and {dx.shape}")

    if dz is not None:
        dz = np.asarray(dz, dtype=np.float32)
        if dz.shape != dy.shape:
            raise ValueError(f"dz must have the same shape as dy/dx, got {dz.shape} and {dy.shape}")

    # XY angle: Cellpose convention is arctan2(dx, dy), not arctan2(dy, dx)
    angles = np.arctan2(dx, dy) + np.pi

    mag_xy = np.sqrt(dx**2 + dy**2)

    if dz is None or z_mode == "none":
        mag = mag_xy
    else:
        mag_3d = np.sqrt(dx**2 + dy**2 + dz**2)

        if z_mode in {"brightness", "saturation"}:
            mag = mag_3d
        else:
            raise ValueError(
                f"Unknown z_mode={z_mode!r}. Use 'brightness', 'saturation', or 'none'."
            )

    mag = 255 * np.clip(normalize99(mag), 0, 1)
    mag = mag / 2

    rgb_float = np.zeros((*dy.shape, 3), dtype=np.float32)

    rgb_float[..., 0] = mag * (np.cos(angles) + 1)
    rgb_float[..., 1] = mag * (np.cos(angles + 2 * np.pi / 3) + 1)
    rgb_float[..., 2] = mag * (np.cos(angles + 4 * np.pi / 3) + 1)

    if dz is not None and z_mode == "saturation":
        # Fraction of the vector lying in the XY plane.
        # 1.0 = fully in-plane, standard Cellpose color.
        # 0.0 = mostly Z, shown closer to white/gray.
        mag_3d_raw = np.sqrt(dx**2 + dy**2 + dz**2)
        xy_fraction = mag_xy / (mag_3d_raw + 1e-6)

        # Convert toward grayscale/white as Z dominates.
        gray = np.mean(rgb_float, axis=-1, keepdims=True)
        white = np.full_like(rgb_float, 255.0)

        # Mostly-Z vectors become brighter and less hue-specific.
        desaturated = 0.5 * gray + 0.5 * white
        rgb_float = (
            xy_fraction[..., None] * rgb_float
            + (1 - xy_fraction[..., None]) * desaturated
        )

    rgb = np.clip(rgb_float, 0, 255).astype(np.uint8)
    return rgb


# ─────────────────────────────────────────────────────────────────────────────
# Cellpose / Omnipose post-processing: flow fields → instance labels
# ─────────────────────────────────────────────────────────────────────────────


def _interpolate_flow(flow_components: List[NDArray], positions: NDArray) -> NDArray:
    """
    Bilinearly interpolate a multi-component flow field at sub-pixel positions.

    Parameters
    ----------
    flow_components : list of ndarray
        One array per spatial axis, each with shape (Y, X) or (Z, Y, X).
        Order must be (Gv, Gh) for 2D or (Gz, Gv, Gh) for 3D.
    positions : (N, ndim) float array
        Query positions in voxel index space.

    Returns
    -------
    (N, ndim) float array of interpolated flow values.
    """
    coords = [positions[:, d] for d in range(len(flow_components))]
    return np.stack(
        [map_coordinates(f, coords, order=1, mode="nearest") for f in flow_components],
        axis=1,
    )


def _euler_integrate(
    flow_components: List[NDArray],
    fg_coords: NDArray,
    n_steps: int,
    res_per_axis: NDArray,
) -> NDArray:
    """
    Euler-integrate the flow field from each foreground pixel position.

    Flows are passed in already scaled to ``≈ ±1 px`` magnitude (raw network
    output ``≈ ±5`` divided by 5, background zeroed — identical to Cellpose).
    Each step displaces a pixel by ``flow / res`` voxels; no additional ``dt``
    factor is applied.

    Parameters
    ----------
    flow_components : list of ndarray
        Spatial flow components in axis order: (Gv, Gh) for 2D or (Gz, Gv, Gh)
        for 3D.  Each array has shape (Y, X) or (Z, Y, X).
    fg_coords : (N, ndim) int array
        Starting voxel coordinates of every foreground pixel.
    n_steps : int
        Number of integration steps.
    res_per_axis : (ndim,) float array
        Physical voxel size per axis ``[z, y, x]`` (or ``[y, x]`` for 2D).
        Used to convert a physical displacement to a pixel displacement.

    Returns
    -------
    pos : (N, ndim) float array
        Final positions in voxel index space after integration.
    """
    shape = flow_components[0].shape
    ndim = len(flow_components)
    res = np.asarray(res_per_axis, dtype=np.float32)

    pos = fg_coords.astype(np.float32).copy()

    for _ in range(n_steps):
        # Bilinear interpolation of each flow component at current positions
        flow_at = _interpolate_flow(flow_components, pos)          # (N, ndim)

        # Flows are already divided by 5 (≈ ±1 px/step); convert physical step to pixels via resolution.
        pos = pos + (flow_at / res[np.newaxis, :])

        # Keep positions inside the volume
        for d in range(ndim):
            pos[:, d] = np.clip(pos[:, d], 0.0, float(shape[d] - 1))

    return pos


def _cluster_to_instances(
    final_pos: NDArray,
    fg_coords: NDArray,
    shape: tuple,
) -> NDArray:
    """
    Build an instance label map using Cellpose's exact histogram-peak + expansion strategy.

    1. Truncate convergence positions to integers (Cellpose: ``.astype("int32")``).
    2. Build a padded 1-px convergence histogram.
    3. Find peaks: local maxima within a 5-bin window per axis with **h > 10**
       (Cellpose hardcoded threshold).
    4. Sort seeds ascending by convergence count so the strongest seed wins
       when two expanded regions overlap (mirrors the torch version).
    5. Expand each seed independently for exactly **5 iterations** of 3×3
       (2D) or 3×3×3 (3D) neighbourhood growth, keeping only bins where
       **h > 2** (Cellpose hardcoded expansion gate).
    6. Build the histogram label map M; look up each fg pixel's label via its
       truncated integer convergence position, then relabel sequentially. One
       peak = one instance (matching Cellpose's ``get_masks``, which does not
       split pixels sharing a peak by connectivity).

    Parameters
    ----------
    final_pos : (N, ndim) float array
        Post-integration convergence positions.
    fg_coords : (N, ndim) int array
        Source foreground pixel voxel coordinates.
    shape : tuple of int
        Spatial shape of the volume.

    Returns
    -------
    labels : (shape) int32 ndarray
        Instance label map. 0 = background.
    """
    ndim = final_pos.shape[1]
    rpad = 20

    # 1. Truncate convergence positions to integers (Cellpose: .astype("int32"))
    pflows = [final_pos[:, d].astype(np.int32) for d in range(ndim)]

    # 2. Build padded convergence histogram
    edges = [np.arange(-0.5 - rpad, shape[d] + 0.5 + rpad, 1.0) for d in range(ndim)]
    h, _ = np.histogramdd(tuple(pflows), bins=edges)

    # 3. Find peaks: local maxima with h > 10 (Cellpose hardcoded seed threshold)
    hmax = h.copy()
    for d in range(ndim):
        hmax = maximum_filter1d(hmax, 5, axis=d)
    seeds_idx = np.nonzero(np.logical_and(h - hmax > -1e-6, h > 10))

    out = np.zeros(shape, dtype=np.int32)
    if len(seeds_idx[0]) == 0:
        return out

    # 4. Sort seeds ascending by convergence count (strongest last → wins on overlap)
    Nmax = h[seeds_idx]
    isort = np.argsort(Nmax)  # ascending
    seeds_idx = tuple(s[isort] for s in seeds_idx)
    n_seeds = len(seeds_idx[0])

    # 5. Expand each seed for exactly 5 iterations of 3×3 (or 3×3×3) neighbourhood,
    #    keeping only bins where h > 2 (Cellpose hardcoded expansion gate).
    #    This mirrors get_masks_torch's iterative max-pool on local 11×11 patches.
    if ndim == 3:
        offsets = np.array(list(np.ndindex(3, 3, 3))) - 1   # (27, 3)
    else:
        offsets = np.array(list(np.ndindex(3, 3))) - 1      # (9, 2)
    hshape = np.array(h.shape)

    # Initialise each seed as a single-bin region (shape (1, ndim))
    pix = [np.array([[seeds_idx[d][k] for d in range(ndim)]]) for k in range(n_seeds)]

    for _ in range(5):
        for k in range(n_seeds):
            curr = pix[k]   # (N, ndim)
            if len(curr) == 0:
                continue
            # All 3×3 (or 3×3×3) neighbours of every current bin: (N·9, ndim)
            neighbours = (
                curr[:, np.newaxis, :] + offsets[np.newaxis, :, :]
            ).reshape(-1, ndim)
            # In-bounds filter (rpad=20 ensures we never truly leave h, but kept for safety)
            inbounds = np.all(
                (neighbours >= 0) & (neighbours < hshape), axis=1
            )
            neighbours = neighbours[inbounds]
            if len(neighbours) == 0:
                pix[k] = neighbours
                continue
            # Gate: only keep bins where h > 2
            valid = h[tuple(neighbours[:, d] for d in range(ndim))] > 2
            neighbours = neighbours[valid]
            if len(neighbours) > 0:
                pix[k] = np.unique(neighbours, axis=0)
            else:
                pix[k] = neighbours

    # 6. Build histogram label map and look up each fg pixel's label
    M = np.zeros(h.shape, dtype=np.int32)
    for k in range(n_seeds):
        if len(pix[k]) > 0:
            M[tuple(pix[k][:, d] for d in range(ndim))] = k + 1

    bin_idx = tuple(
        np.clip(pflows[d] + rpad, 0, h.shape[d] - 1) for d in range(ndim)
    )
    pk_labels = np.zeros(shape, dtype=np.int32)
    pk_labels[tuple(fg_coords[:, d] for d in range(ndim))] = M[bin_idx]

    # 7. Relabel sequentially so instance ids are contiguous. One peak = one instance; we deliberately
    #    do NOT split pixels sharing a peak by connectivity, matching Cellpose's get_masks.
    final_labels, _, _ = relabel_sequential(pk_labels)
    return final_labels.astype(np.int32)


def _dbscan_cluster(
    final_pos: NDArray,
    fg_coords: NDArray,
    shape: tuple,
    eps: float,
) -> NDArray:
    """
    Build an instance label map by clustering convergence positions with DBSCAN.

    This is the Omnipose-style clustering strategy: because Omnipose flows point
    toward the cell skeleton (EDT gradient), all pixels in one cell converge to a
    small neighbourhood, forming a tight DBSCAN cluster.  No histogram peak
    detection is needed.

    Algorithm:
    1. Run DBSCAN on ``final_pos`` with ``eps`` as the neighbourhood radius and
       ``min_samples=1`` (every pixel is a potential cluster seed).
    2. Map DBSCAN cluster IDs back to foreground pixel coordinates.
    3. Split any disconnected region sharing the same DBSCAN label into separate
       instances (preserves fine-grained boundaries).

    Parameters
    ----------
    final_pos : (N, ndim) float array
        Post-integration convergence positions (from :func:`_euler_integrate`).
    fg_coords : (N, ndim) int array
        Source foreground pixel voxel coordinates.
    shape : tuple of int
        Spatial shape of the volume.
    eps : float
        DBSCAN neighbourhood radius in pixels.  Equivalent to ``max_cluster_dist``
        in :func:`_cluster_to_instances`.

    Returns
    -------
    labels : (shape) int32 ndarray
        Instance label map.  0 = background.
    """
    from sklearn.cluster import DBSCAN

    ndim = final_pos.shape[1]
    db = DBSCAN(eps=eps, min_samples=1, n_jobs=-1).fit(final_pos)
    db_labels = db.labels_  # -1 = noise

    # Map DBSCAN labels to the spatial grid (0 = unassigned / noise)
    pk_labels = np.zeros(shape, dtype=np.int32)
    for i in range(len(fg_coords)):
        lbl = int(db_labels[i])
        if lbl >= 0:
            pk_labels[tuple(fg_coords[i])] = lbl + 1  # shift so 0 stays background

    # Split disconnected regions sharing the same DBSCAN cluster
    final_labels = np.zeros(shape, dtype=np.int32)
    current_id = 0
    n_clusters = int(pk_labels.max())
    for pk in range(1, n_clusters + 1):
        mask = pk_labels == pk
        if not mask.any():
            continue
        cc = cc_label(mask, connectivity=ndim)
        n = int(cc.max())
        if n > 0:
            final_labels[mask] = cc[mask] + current_id
            current_id += n

    final_labels, _, _ = relabel_sequential(final_labels)
    return final_labels.astype(np.int32)


def _mask_to_unit_flow(
    mask: NDArray,
    resolution: NDArray,
) -> tuple:
    """
    Regenerate the unit flow field of a single mask via heat diffusion.

    Reproduces the *same* per-cell diffusion used to build the training targets
    in :func:`biapy.data.pre_processing.labels_into_channels` (``gradient_type
    ="cellpose"``): heat is injected at the mask pixel closest to the centroid
    and diffused with a Moore-neighbourhood average for ``2 * (sum(shape) + 4)``
    (2D) / ``6 * (sum(shape) + 3)`` (3D) iterations; the normalised gradient of
    that field (no ``log``) is the flow.  Because the
    generation matches the ground-truth exactly, a *correct* predicted mask
    yields a regenerated flow that matches the network flow (error ≈ 0), while a
    spurious fragment of a larger cell produces a flow pointing at the fragment's
    own centre — very different from the network flow, giving a large error.

    Parameters
    ----------
    mask : (y, x) or (z, y, x) bool array
        Cropped binary mask of a single instance.
    resolution : (ndim,) float array
        Physical voxel size per axis, used for an anisotropy-correct gradient.

    Returns
    -------
    unit_flow : (ndim, N) float array
        Unit-normalised flow at the mask's foreground pixels, axis order
        (Gv, Gh) in 2D or (Gz, Gv, Gh) in 3D.
    coords : tuple of arrays
        ``np.nonzero(mask)`` in the cropped frame — the pixel order the flow
        columns correspond to.
    """
    coords = np.nonzero(mask)
    centroid = np.array([c.mean() for c in coords])
    coord_stack = np.stack(coords, axis=1).astype(np.float32)
    idx = int(np.argmin(np.sum((coord_stack - centroid) ** 2, axis=1)))
    centers = tuple(int(c[idx]) for c in coords)

    pad = 1
    heat = np.zeros(tuple(s + 2 * pad for s in mask.shape), dtype=np.float64)
    p_coords = tuple(c + pad for c in coords)
    p_center = tuple(c + pad for c in centers)
    # Same diffusion-step count as the training-target generator (2D: 2*(sum+4), 3D: 6*(sum+3)) so the
    # flow-error comparison is not biased.
    if mask.ndim == 3:
        n_iter = 6 * (int(sum(mask.shape)) + 3)
    else:
        n_iter = 2 * (int(sum(mask.shape)) + 4)

    if mask.ndim == 2:
        y_c, x_c = p_coords
        ymed, xmed = p_center
        for _ in range(n_iter):
            heat[ymed, xmed] += 1.0
            heat[y_c, x_c] = (1.0 / 9.0) * (
                heat[y_c, x_c] +
                heat[y_c - 1, x_c] + heat[y_c + 1, x_c] +
                heat[y_c, x_c - 1] + heat[y_c, x_c + 1] +
                heat[y_c - 1, x_c - 1] + heat[y_c - 1, x_c + 1] +
                heat[y_c + 1, x_c - 1] + heat[y_c + 1, x_c + 1]
            )
        # Gradient of the RAW diffusion field (no log), matching the training-target generation; a
        # log(1+T) would tilt the flow direction at boundary pixels.
        grads = np.gradient(heat, resolution[0], resolution[1])
    else:
        z_c, y_c, x_c = p_coords
        zmed, ymed, xmed = p_center
        for _ in range(n_iter):
            heat[zmed, ymed, xmed] += 1.0
            neigh = np.zeros(len(z_c), dtype=np.float64)
            for dz in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        neigh += heat[z_c + dz, y_c + dy, x_c + dx]
            heat[z_c, y_c, x_c] = neigh / 27.0
        # Gradient of the RAW diffusion field (no log); see the 2D branch above.
        grads = np.gradient(heat, resolution[0], resolution[1], resolution[2])

    # Epsilon outside the sqrt and negligible (1e-60), matching the training-target generator: one
    # inside would floor the magnitude and zero out the small-but-valid gradients at border pixels.
    mag = np.sqrt(sum(g[p_coords] ** 2 for g in grads)) + 1e-60
    unit_flow = np.stack([g[p_coords] / mag for g in grads], axis=0)  # (ndim, N)
    return unit_flow, coords


def _flow_error(
    Gv: NDArray,
    Gh: NDArray,
    labels: NDArray,
    Gz: Optional[NDArray] = None,
    resolution: NDArray = None,
    exempt_border_cells: bool = True,
) -> dict:
    """
    Per-instance flow error, matching Cellpose ``metrics.flow_error``.

    For every predicted mask the flow field is *regenerated from the mask*
    (:func:`_mask_to_unit_flow`) and compared, as a mean squared error, against
    the network-predicted flow at the same pixels.  This is Cellpose's exact
    quality metric (``dynamics.remove_bad_flow_masks`` → ``metrics.flow_error``)
    and is the mechanism that removes the spurious fragments produced when a
    large cell is over-segmented into several instances: each fragment's own
    diffusion flow disagrees with the network flow (which points to the true
    cell centre), so its error exceeds the threshold and it is discarded.

    The network flows ``Gv``/``Gh``/``Gz`` are expected already divided by 5
    (as done in :func:`flows_to_instances`), i.e. ≈ unit magnitude, matching the
    unit ``dP_masks`` regenerated here — the same convention as Cellpose's
    ``dP_masks - dP_net / 5``.  In 3D the Z term is down-weighted by 0.5, exactly
    as Cellpose does.

    Parameters
    ----------
    Gv, Gh : (Y, X) or (Z, Y, X) float arrays
        Network y / x flow components (already scaled to ≈ ±1).
    labels : same-shape int array
        Instance label map (0 = background).
    Gz : (Z, Y, X) float array, optional
        Network z flow component (3D only).
    resolution : (ndim,) float array, optional
        Physical voxel size per axis.  Defaults to isotropic ``1``.
    exempt_border_cells : bool, optional
        When ``True`` (default), instances whose mask touches the image border are assigned error
        ``0`` (never removed): their flow legitimately points off-frame and cannot be reproduced by
        an in-frame regeneration.  Set ``False`` to score them like any other instance (Cellpose's
        behaviour).

    Returns
    -------
    errors : dict {label_id (int): mean squared flow error (float)}
    """
    ndim = labels.ndim
    is_3d = Gz is not None
    if resolution is None:
        res = np.ones(ndim, dtype=np.float32)
    else:
        res = np.asarray(resolution[-ndim:], dtype=np.float32)

    errors: dict = {}
    slices = find_objects(labels)
    for i, slc in enumerate(slices):
        lab = i + 1
        if slc is None:
            continue

        # Border-touching instances are exempt: a cell cut by the field of view has flows pointing to
        # an off-frame centre that no in-frame regeneration can reproduce, so it would always score a
        # large error. Diverges from Cellpose's remove_bad_flow_masks, which has no such guard.
        if exempt_border_cells and any(slc[d].start == 0 or slc[d].stop == labels.shape[d] for d in range(ndim)):
            errors[lab] = 0.0
            continue

        sub = labels[slc] == lab

        # Cells too small to diffuse (1-px wide along any axis) get error 0, never removed.
        if sub.sum() < 2 or any(s < 2 for s in sub.shape):
            errors[lab] = 0.0
            continue

        unit_flow, coords = _mask_to_unit_flow(sub, res)  # (ndim, N), local coords

        # Network flow at the same pixels, same (C-order) ordering as `coords`.
        if is_3d:
            net = np.stack(
                [Gz[slc][sub], Gv[slc][sub], Gh[slc][sub]], axis=0
            )  # (3, N)
            err = (
                0.5 * (unit_flow[0] - net[0]) ** 2
                + (unit_flow[1] - net[1]) ** 2
                + (unit_flow[2] - net[2]) ** 2
            ).mean()
        else:
            net = np.stack([Gv[slc][sub], Gh[slc][sub]], axis=0)  # (2, N)
            err = (
                (unit_flow[0] - net[0]) ** 2 + (unit_flow[1] - net[1]) ** 2
            ).mean()

        errors[lab] = float(err)

    return errors


def _remove_bad_flow_masks(
    labels: NDArray,
    errors: dict,
    flow_threshold: float,
) -> NDArray:
    """
    Zero-out instances whose flow consistency error exceeds ``flow_threshold``.

    Parameters
    ----------
    labels : int32 ndarray
        Instance label map (not modified in place).
    errors : dict {label_id: float}
        Per-instance flow error from :func:`_flow_error`.
    flow_threshold : float
        Instances with ``error > flow_threshold`` are removed.

    Returns
    -------
    labels : int32 ndarray
        Filtered label map, relabelled sequentially.
    """
    bad = {lab for lab, err in errors.items() if err > flow_threshold}
    if not bad:
        return labels

    out = labels.copy()
    for lab in bad:
        out[out == lab] = 0

    out, _, _ = relabel_sequential(out)
    return out.astype(np.int32)


def _resize_spatial(arr: NDArray, out_shape: tuple, order: int) -> NDArray:
    """
    Resize a 2D/3D array to ``out_shape`` (Cellpose ``transforms.resize_image``).

    ``order=1`` (bilinear) is used for flow fields / probabilities and ``order=0``
    (nearest, ``cv2.INTER_NEAREST`` in Cellpose) for label maps so integer ids are
    preserved.  Anti-aliasing is disabled to match Cellpose's plain interpolation.
    """
    from skimage.transform import resize as _sk_resize

    if tuple(arr.shape) == tuple(out_shape):
        return arr
    return _sk_resize(
        arr, out_shape, order=order, preserve_range=True,
        anti_aliasing=False, mode="edge",
    )


def _estimate_cell_radius(
    fg_mask: NDArray,
    is_3d: bool,
    percentile: float = 50.0,
    min_components: int = 3,
) -> tuple:
    """
    Estimate a representative cell radius (pixels) from a binary foreground mask.

    Connected components of ``fg_mask`` are measured and the equivalent radius at
    ``percentile`` of the per-object size distribution is returned.  With the
    default ``percentile=50`` this is the *median* object radius, matching
    Cellpose's median-diameter convention (``utils.diameters``); the diameter
    (``2 * radius``) can be fed straight into the rescaling of
    :func:`flows_to_instances`.  This lets a test set whose images have different
    cell scales be handled without a fixed value — Cellpose's ``diameter=0`` path.

    .. note::
        The estimate is taken from the *foreground* connected components, so cells
        that touch merge into one component and inflate the radius.  For heavily
        touching cells prefer an explicit ``diameter``, or measure it from an
        actual segmentation.

    Parameters
    ----------
    fg_mask : (Y, X) or (Z, Y, X) bool array
        Binary foreground mask.
    is_3d : bool
        Whether the data is 3-D (sphere-equivalent radius) or 2-D (disc).
    percentile : float, optional
        Percentile of the per-object size distribution used as the representative
        size.  50 = median (Cellpose convention).  Default 50.
    min_components : int, optional
        Minimum number of components required for a reliable estimate.  Below this
        the function returns ``(None, None)``.  Default 3.

    Returns
    -------
    radius : float or None
        Representative object radius in pixels, or None if unreliable.
    stats : dict or None
        Diagnostic fields (component count, size range), or None.
    """
    labeled, n_labels = cc_label(fg_mask, return_num=True)
    if n_labels < min_components:
        return None, None

    sizes = np.bincount(labeled.ravel())[1:]          # drop background (label 0)
    sizes = sizes[sizes > 0].astype(np.float64)
    if len(sizes) < min_components:
        return None, None

    # Per-object equivalent radius, then take the requested percentile (median by
    # default) — Cellpose measures the median of per-object equivalent sizes.
    if is_3d:
        radii = (3.0 * sizes / (4.0 * np.pi)) ** (1.0 / 3.0)
    else:
        radii = np.sqrt(sizes / np.pi)
    radius = float(np.percentile(radii, percentile))

    stats = {
        "n_components": int(len(sizes)),
        "min_area": int(sizes.min()),
        "max_area": int(sizes.max()),
        "median_area": float(np.median(sizes)),
        "percentile": percentile,
    }
    return radius, stats


def flows_to_instances(
    pred: NDArray,
    channels: List[str],
    flow_type: str = "cellpose",
    fg_channel: str = "",
    fg_thresh: float = 0.5,
    flow_threshold: float = 0.4,
    n_steps: int = 200,
    max_cluster_dist: float = 5.0,
    resolution: List[float] = [1.0, 1.0, 1.0],
    diameter: float = 30.0,
    diam_mean: float = 30.0,
    already_rescaled: bool = False,
    exempt_border_cells: bool = True,
) -> NDArray:
    """
    Convert predicted Cellpose / Omnipose flow fields into an instance label map.

    ``flow_type="cellpose"``: flows are the gradient of a per-instance heat-diffusion potential.
    Euler-integrate each foreground pixel (≈1 px/step), detect histogram peaks and grow them with a
    5-iteration 3x3 expansion; an optional flow-error check removes masks whose diffusion-regenerated
    flow disagrees with the network.

    ``flow_type="omnipose"``: flows are the gradient of the per-cell EDT. Euler-integrate, then DBSCAN
    the convergence positions. The flow-error check is skipped.

    Parameters
    ----------
    pred : (Y, X, C) or (Z, Y, X, C) float ndarray
        Full model prediction (spatial dims + channel dim last).
    channels : list of str
        Channel names matching the last axis of ``pred``,
        e.g. ``["F", "Gv", "Gh"]``, ``["B", "Gv", "Gh", "Gz"]``.
    flow_type : {"cellpose", "omnipose"}, optional
        Post-processing strategy. Default "cellpose".
    fg_channel : str, optional
        Channel thresholded for the foreground mask (``"B"`` inverts it). Empty derives the mask from
        the flow magnitude.
    fg_thresh : float, optional
        Sigmoid-space threshold applied to ``fg_channel``. Default 0.5.
    flow_threshold : float, optional
        Cellpose only. Mean-squared flow error above which an instance is removed; ``<= 0`` skips the
        check. Default 0.4.
    n_steps : int, optional
        Ignored; the step count is derived from the diameter. Default 200.
    max_cluster_dist : float, optional
        Omnipose only. DBSCAN ``eps`` radius in pixels. Default 5.0.
    resolution : list of float, optional
        Physical voxel size ``[z, y, x]`` for converting flow steps to pixels. Default [1, 1, 1].
    diameter : float, optional
        Cellpose only. Expected cell diameter (px); flows are rescaled by ``diam_mean / diameter`` so
        cells become ~``diam_mean``. ``diam_mean`` disables rescaling; ``<= 0`` auto-estimates it from
        the median object size. Default 30.0.
    diam_mean : float, optional
        Cellpose only. Diameter the model was trained at (30 cyto, 17 nuclei). Default 30.0.
    already_rescaled : bool, optional
        Cellpose only. ``True`` when the input was rescaled before the network and the flows resized
        back, so the flow field is not rescaled again here. Default ``False``.
    exempt_border_cells : bool, optional
        Cellpose only. When ``True``, instances touching the image border are never removed by the
        flow-error check (their flows point to an off-frame centre). Default ``True``.

    Returns
    -------
    labels : int32 ndarray, shape (Y, X) or (Z, Y, X)
        Instance label map.  0 = background.
    """
    if flow_type not in ("cellpose", "omnipose"):
        raise ValueError(f"flow_type must be 'cellpose' or 'omnipose', got '{flow_type}'")
    is_3d = pred.ndim == 4  # (Z, Y, X, C)
    spatial_shape = pred.shape[:-1]
    ndim = len(spatial_shape)
    ch = list(channels)

    # ── 1. Extract flow channels ──────────────────────────────────────────
    if "Gv" not in ch or "Gh" not in ch:
        raise ValueError("'pred' must contain at least 'Gv' and 'Gh' channels")

    Gv = pred[..., ch.index("Gv")].astype(np.float32)
    Gh = pred[..., ch.index("Gh")].astype(np.float32)
    Gz: Optional[NDArray] = None
    if is_3d and "Gz" in ch:
        Gz = pred[..., ch.index("Gz")].astype(np.float32)

    # ── 2. Foreground mask ────────────────────────────────────────────────
    if fg_channel and fg_channel in ch:
        fg_raw = pred[..., ch.index(fg_channel)].astype(np.float32)
        # B=1 means background → invert
        fg_mask = (fg_raw < fg_thresh) if fg_channel == "B" else (fg_raw > fg_thresh)
    else:
        # Background flow is identically zero by construction (GT masking ensures
        # this); any pixel with non-zero magnitude is foreground.
        mag = np.sqrt(Gv ** 2 + Gh ** 2 + (Gz ** 2 if Gz is not None else 0.0))
        fg_mask = mag > 0.0

    fg_mask = fg_mask.astype(bool)
    if not fg_mask.any():
        return np.zeros(spatial_shape, dtype=np.int32)

    # ── 2b. Diameter: auto-estimate per image when not provided ───────────
    # BiaPy has no SizeModel, so it measures the median object size from the foreground mask. An
    # explicit positive diameter skips estimation.
    if flow_type == "cellpose" and diameter <= 0:
        radius, stats = _estimate_cell_radius(fg_mask, is_3d)
        if radius is not None and radius > 0:
            diameter = 2.0 * radius
            print(f"  Auto-estimated diameter: {diameter:.1f} px "
                  f"(median of {stats['n_components']} foreground components)")
        else:
            diameter = diam_mean
            print(f"  Diameter auto-estimate unreliable (too few objects); "
                  f"falling back to diam_mean={diam_mean}")

    # ── 2c. Cellpose diameter rescaling ───────────────────────────────────
    # Run the dynamics with cells at ~diam_mean. When the input was already rescaled before the network
    # (``already_rescaled``), skip it here to avoid resampling the flows twice; otherwise rescale the
    # flow field by diam_mean/diameter and resize the labels back afterwards. Cellpose only (Omnipose
    # clusters convergence positions directly). Rescale is uniform per axis (approximate for anisotropic
    # 3D).
    rescale = float(diam_mean) / float(diameter) if diameter > 0 else 1.0
    do_rescale = flow_type == "cellpose" and not already_rescaled and abs(rescale - 1.0) > 1e-3
    if do_rescale:
        work_shape = tuple(max(1, int(round(s * rescale))) for s in spatial_shape)
        Gv = _resize_spatial(Gv, work_shape, order=1).astype(np.float32)
        Gh = _resize_spatial(Gh, work_shape, order=1).astype(np.float32)
        if Gz is not None:
            Gz = _resize_spatial(Gz, work_shape, order=1).astype(np.float32)
        fg_mask = _resize_spatial(fg_mask.astype(np.float32), work_shape, order=1) > 0.5
        if not fg_mask.any():
            return np.zeros(spatial_shape, dtype=np.int32)
    else:
        work_shape = spatial_shape

    # ── 2d. Integration step count ────────────────────────────────────────
    # Depends on the dynamics resolution: 200 steps when integrating at ~diam_mean (do_rescale),
    # (1/rescale)*200 when integrating at native resolution (Cellpose's niter rule).
    if flow_type == "cellpose":
        n_steps = 200 if do_rescale else max(1, int(round((1.0 / rescale) * 200)))
    else:
        n_steps = 200

    # ── 3. Scale and mask flows ───────────────────────────────────────────
    # Raw network output ≈ ±5 (5x targets); divide by 5 for ≈ ±1 px/step, then zero background.
    Gv = (Gv * fg_mask / 5.0).astype(np.float32)
    Gh = (Gh * fg_mask / 5.0).astype(np.float32)
    if Gz is not None:
        Gz = (Gz * fg_mask / 5.0).astype(np.float32)

    # ── 4. Foreground pixel coordinates ──────────────────────────────────
    fg_coords = np.stack(np.nonzero(fg_mask), axis=1)  # (N, ndim)

    # ── 5. Resolution slice matching ndim (stored as [z, y, x]) ───────────
    res = np.array(resolution[-ndim:], dtype=np.float32)

    # Ordered flow tuple: (Gz, Gv, Gh) for 3D, (Gv, Gh) for 2D
    flow_comps: List[NDArray] = ([Gz, Gv, Gh] if Gz is not None else [Gv, Gh])  # type: ignore[list-item]

    # ── 6. Euler integration ──────────────────────────────────────────────
    print(f"  Flow integration ({flow_type}): {len(fg_coords):,} foreground pixels, {n_steps} steps "
          f"(rescale={rescale:.3f}, work_shape={work_shape}) ...")
    final_pos = _euler_integrate(flow_comps, fg_coords, n_steps, res)

    # ── 7. Cluster convergence positions into instances ───────────────────
    print("  Clustering convergence positions ...")
    if flow_type == "omnipose":
        # DBSCAN on convergence positions: each cluster = one cell.
        # eps = max_cluster_dist (how close positions must be to belong to the same instance).
        labels = _dbscan_cluster(final_pos, fg_coords, work_shape, max_cluster_dist)
    else:
        # Cellpose: histogram peak detection + 5-step 3×3 expansion.
        labels = _cluster_to_instances(final_pos, fg_coords, work_shape)

    # ── 8. Optional flow error check (Cellpose only) ──────────────────────
    # Regenerates each mask's flow by diffusion and removes masks whose flow disagrees with the network
    # (over-segmentation fragments). Skipped for Omnipose.
    if flow_type == "cellpose" and flow_threshold > 0.0 and int(labels.max()) > 0:
        print(f"  Flow error check (threshold={flow_threshold}) ...")
        errors = _flow_error(Gv, Gh, labels, Gz, resolution=res, exempt_border_cells=exempt_border_cells)
        n_before = int(labels.max())
        labels = _remove_bad_flow_masks(labels, errors, flow_threshold)
        n_removed = n_before - int(labels.max())
        if n_removed:
            print(f"    Removed {n_removed} instance(s) with inconsistent flow.")

    # ── 9. Resize labels back to native resolution (Cellpose: resize_image
    #      with nearest-neighbour interpolation) ───────────────────────────
    if do_rescale:
        labels = _resize_spatial(labels, spatial_shape, order=0).astype(np.int32)

    return labels