import numpy as np
from scipy.ndimage import map_coordinates, maximum_filter1d
from scipy.spatial import cKDTree
from skimage.measure import label as cc_label
from skimage.morphology import remove_small_objects
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


def _estimate_cell_radius(
    fg_channel: NDArray,
    fg_thresh: float,
    is_3d: bool,
    min_components: int = 3,
    size_percentile: float = 15.0,
) -> Optional[float]:
    """
    Estimate a representative small-cell radius from the foreground channel.

    Thresholds ``fg_channel``, labels connected components, and returns the
    equivalent radius at ``size_percentile`` of the area/volume distribution.
    Used to auto-tune ``min_size`` and ``max_cluster_dist`` for Cellpose
    post-processing when cells of mixed scales are present.

    Parameters
    ----------
    fg_channel : ndarray
        2-D (Y, X) or 3-D (Z, Y, X) foreground probability map (sigmoid space).
    fg_thresh : float
        Threshold applied to ``fg_channel`` to obtain the binary mask.
    is_3d : bool
        Whether the data is 3-D.
    min_components : int
        Minimum number of connected components required to return an estimate.
        If fewer are found the function returns None.
    size_percentile : float
        Percentile of the component-size distribution used to represent
        "small cells". Lower values are more sensitive to the smallest cells.

    Returns
    -------
    radius : float or None
        Equivalent radius (pixels) at ``size_percentile``, or None if the
        estimate is unreliable (too few components).
    stats : dict or None
        Dictionary with diagnostic fields:
        ``n_components``, ``min_area``, ``max_area``, ``median_area``,
        ``percentile_area`` (the area used for radius), ``size_percentile``.
        None when radius is None.
    """
    binary = (fg_channel > fg_thresh).astype(bool)
    labeled, n_labels = cc_label(binary, return_num=True)
    if n_labels < min_components:
        return None, None

    sizes = np.bincount(labeled.ravel())[1:]   # skip background (label 0)
    sizes = sizes[sizes > 0]
    if len(sizes) < min_components:
        return None, None

    small_area = float(np.percentile(sizes, size_percentile))
    if is_3d:
        radius = (3.0 * small_area / (4.0 * np.pi)) ** (1.0 / 3.0)
    else:
        radius = np.sqrt(small_area / np.pi)

    stats = {
        "n_components": int(len(sizes)),
        "min_area": int(sizes.min()),
        "max_area": int(sizes.max()),
        "median_area": float(np.median(sizes)),
        "percentile_area": small_area,
        "size_percentile": size_percentile,
    }
    return radius, stats


def _euler_integrate(
    flow_components: List[NDArray],
    fg_coords: NDArray,
    n_steps: int,
    dt: float,
    suppressed: bool,
    res_per_axis: NDArray,
) -> NDArray:
    """
    Euler-integrate the flow field from each foreground pixel position.

    Flow vectors are defined in physical space (unit magnitude after normalisation).
    The integration converts each physical step to voxel-index steps using the
    provided resolution so that anisotropic data is handled correctly.

    Parameters
    ----------
    flow_components : list of ndarray
        Spatial flow components in axis order: (Gv, Gh) for 2D or (Gz, Gv, Gh)
        for 3D.  Each array has shape (Y, X) or (Z, Y, X).
    fg_coords : (N, ndim) int array
        Starting voxel coordinates of every foreground pixel.
    n_steps : int
        Number of integration steps.
    dt : float
        Physical-unit step size per iteration.
    suppressed : bool
        If True, decay the step size as ``dt / (t + 1)``, giving a total travel
        distance of approximately ``5.9 * dt`` pixels (harmonic series). Only
        useful for very small cells or as a convergence aid for very noisy flows.
        If False (Cellpose 4.x default), use a constant step ``dt`` per iteration;
        total travel is ``n_steps * dt`` pixels, allowing convergence for any cell
        size.
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

    for t in range(n_steps):
        # Bilinear interpolation of each flow component at current positions
        flow_at = _interpolate_flow(flow_components, pos)          # (N, ndim)

        # Convert physical step → pixel step: Δpix[d] = Δphys[d] / res[d]
        factor = dt / (t + 1) if suppressed else dt
        pos = pos + factor * (flow_at / res[np.newaxis, :])

        # Keep positions inside the volume
        for d in range(ndim):
            pos[:, d] = np.clip(pos[:, d], 0.0, float(shape[d] - 1))

    return pos


def _attractor_peaks(
    final_pos: NDArray,
    shape: tuple,
    rpad: int = 20,
    min_count: int = 5,
) -> NDArray:
    """
    Detect attractor peaks in the histogram of post-integration convergence positions.

    Follows Cellpose's histogram-maximum strategy: a position is a peak if it is
    a local maximum (within a 5-bin window per axis) with at least ``min_count``
    pixels converging to it.

    Parameters
    ----------
    final_pos : (N, ndim) float array
        Convergence positions after Euler integration.
    shape : tuple of int
        Spatial shape of the volume (without channel axis).
    rpad : int
        Padding added around the volume in histogram bins to capture pixels
        that drifted slightly outside bounds.
    min_count : int
        Minimum number of converging pixels for a position to be a peak.

    Returns
    -------
    peaks : (K, ndim) float array
        Peak positions in image voxel coordinates.  May be empty (shape (0, ndim)).
    """
    ndim = final_pos.shape[1]
    edges = [np.arange(-0.5 - rpad, shape[d] + 0.5 + rpad, 1.0) for d in range(ndim)]
    h, _ = np.histogramdd(final_pos, bins=edges)

    # Local-maximum filter with window 5 along each axis
    hmax = h.copy()
    for d in range(ndim):
        hmax = maximum_filter1d(hmax, 5, axis=d)

    peak_mask = (h - hmax > -1e-6) & (h >= min_count)
    # Histogram indices → image coordinates (subtract padding offset)
    peak_hist_idx = np.stack(np.nonzero(peak_mask), axis=1).astype(float)
    peaks = peak_hist_idx - rpad

    return peaks  # (K, ndim)


def _cluster_to_instances(
    final_pos: NDArray,
    fg_coords: NDArray,
    shape: tuple,
    min_size: int,
    max_cluster_dist: float,
) -> NDArray:
    """
    Build an instance label map by assigning foreground pixels to attractor peaks.

    Algorithm:
    1. Find peaks in the histogram of ``final_pos`` (Cellpose strategy).
    2. Assign each foreground pixel to its nearest peak within ``max_cluster_dist``.
    3. Split any disconnected region sharing the same peak label into separate
       instances (handles the rare case where noise causes two far-apart pixels to
       converge to the same attractor).
    4. Relabel sequentially and remove objects smaller than ``min_size``.

    Parameters
    ----------
    final_pos : (N, ndim) float array
        Post-integration convergence positions.
    fg_coords : (N, ndim) int array
        Source foreground pixel voxel coordinates.
    shape : tuple of int
        Spatial shape of the volume.
    min_size : int
        Minimum instance size in voxels.
    max_cluster_dist : float
        Maximum pixel distance from a convergence position to a valid attractor
        peak.  Pixels farther away are treated as background noise.

    Returns
    -------
    labels : (shape) int32 ndarray
        Instance label map. 0 = background.
    """
    ndim = final_pos.shape[1]

    # 1. Find attractor peaks
    peaks = _attractor_peaks(final_pos, shape, min_count=max(1, min_size // 2))

    labels = np.zeros(shape, dtype=np.int32)
    if peaks.shape[0] == 0:
        return labels

    # 2. Assign each fg pixel to its nearest valid peak
    tree = cKDTree(peaks)
    dists, peak_ids = tree.query(final_pos, k=1)
    valid = dists < max_cluster_dist

    # Peak-assignment map: value = peak_id + 1 (so 0 means unassigned)
    pk_labels = np.zeros(shape, dtype=np.int32)
    for i in range(len(fg_coords)):
        if valid[i]:
            pk_labels[tuple(fg_coords[i])] = int(peak_ids[i]) + 1

    # 3. Split disconnected regions sharing the same peak label
    #    Each unique peak label is processed independently so adjacent regions
    #    with *different* peak labels are never merged.
    final_labels = np.zeros(shape, dtype=np.int32)
    current_id = 0
    for pk in range(1, int(peaks.shape[0]) + 1):
        mask = pk_labels == pk
        if not mask.any():
            continue
        cc = cc_label(mask, connectivity=ndim)
        n = int(cc.max())
        if n > 0:
            final_labels[mask] = cc[mask] + current_id
            current_id += n

    # 4. Remove small objects and relabel
    if min_size > 0:
        final_labels = remove_small_objects(final_labels, min_size=min_size)
    final_labels, _, _ = relabel_sequential(final_labels)
    return final_labels.astype(np.int32)


def _dbscan_cluster(
    final_pos: NDArray,
    fg_coords: NDArray,
    shape: tuple,
    min_size: int,
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
    4. Relabel sequentially and remove objects smaller than ``min_size``.

    Parameters
    ----------
    final_pos : (N, ndim) float array
        Post-integration convergence positions (from :func:`_euler_integrate`).
    fg_coords : (N, ndim) int array
        Source foreground pixel voxel coordinates.
    shape : tuple of int
        Spatial shape of the volume.
    min_size : int
        Minimum instance size in voxels.
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

    if min_size > 0:
        final_labels = remove_small_objects(final_labels, min_size=min_size)
    final_labels, _, _ = relabel_sequential(final_labels)
    return final_labels.astype(np.int32)


def _flow_consistency_error(
    Gv: NDArray,
    Gh: NDArray,
    labels: NDArray,
    Gz: Optional[NDArray] = None,
) -> dict:
    """
    Compute a per-instance flow consistency error.

    For each instance the *expected* flow is approximated as the unit vector
    pointing from every foreground pixel toward the instance centroid.  The
    error is ``1 - cosine_similarity(predicted_flow, expected_flow)`` averaged
    over the instance's foreground pixels, so:

    * 0  → predicted flow points perfectly toward the centroid (ideal).
    * 1  → predicted flow is orthogonal to the centroid direction.
    * 2  → predicted flow points away from the centroid (spurious instance).

    .. note::
        This is an approximation.  For elongated or non-convex cells the true
        Cellpose / Omnipose flow does not point toward the centroid, so the
        error may be inflated.  For Omnipose (EDT-gradient flows) prefer a
        lower threshold or disable the check entirely.

    Parameters
    ----------
    Gv, Gh : (Y, X) or (Z, Y, X) float arrays
        Predicted y / x flow components.
    labels : same-shape int array
        Instance label map (0 = background).
    Gz : (Z, Y, X) float array, optional
        Predicted z flow component (3D only).

    Returns
    -------
    errors : dict {label_id (int): error (float)}
    """
    is_3d = Gz is not None
    errors: dict = {}

    for lab in np.unique(labels):
        if lab == 0:
            continue
        coords = np.stack(np.nonzero(labels == lab), axis=1).astype(float)  # (N, ndim)
        centroid = coords.mean(axis=0)

        # Expected unit vector: each pixel → centroid
        delta = centroid[np.newaxis, :] - coords
        norms = np.linalg.norm(delta, axis=1, keepdims=True) + 1e-8
        expected = delta / norms  # (N, ndim)

        # Predicted flow at the same integer pixel coordinates
        ci = coords.astype(int)
        idx = tuple(ci[:, d] for d in range(ci.shape[1]))
        if is_3d:
            pred_flow = np.stack([Gz[idx], Gv[idx], Gh[idx]], axis=1)
        else:
            pred_flow = np.stack([Gv[idx], Gh[idx]], axis=1)

        pred_norms = np.linalg.norm(pred_flow, axis=1, keepdims=True) + 1e-8
        pred_unit = pred_flow / pred_norms

        cosine_sim = float((expected * pred_unit).sum(axis=1).mean())
        errors[int(lab)] = 1.0 - cosine_sim

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
        Per-instance flow error from :func:`_flow_consistency_error`.
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


def flows_to_instances(
    pred: NDArray,
    channels: List[str],
    flow_type: str = "cellpose",
    fg_channel: str = "",
    fg_thresh: float = 0.5,
    flow_threshold: float = 0.4,
    n_steps: int = 200,
    dt: float = 1.0,
    suppressed: bool = False,
    min_size: int = 15,
    max_cluster_dist: float = 5.0,
    resolution: List[float] = [1.0, 1.0, 1.0],
    auto_adjust: bool = False,
) -> NDArray:
    """
    Convert predicted Cellpose / Omnipose flow fields into an instance label map.

    Two modes are supported, selected by ``flow_type``:

    **Cellpose** (``flow_type="cellpose"``, default):
      Flows are the gradient of a per-instance heat-diffusion potential.
      Post-processing uses time-suppressed Euler integration (step = dt/(t+1))
      followed by histogram peak detection and cKDTree assignment.  A flow
      consistency check then discards instances whose predicted flow diverges
      from a centroid-pointing approximation.

    **Omnipose** (``flow_type="omnipose"``):
      Flows are the gradient of the per-cell Euclidean Distance Transform (EDT).
      Post-processing uses constant-step Euler integration (``suppressed`` is
      forced to False) followed by DBSCAN clustering on convergence positions.
      The flow consistency check is skipped because Omnipose flows are accurate
      by construction (they always point toward the EDT maximum / skeleton).

    Common pipeline for both modes:

    1. Extract Gv / Gh / [Gz] from ``pred``.
    2. Build the foreground mask from a dedicated channel or from the flow magnitude.
    3. Re-normalise predicted flow vectors to unit length.
    4. Euler-integrate each foreground pixel along the flow field (anisotropy-aware).
    5. *Cellpose*: detect attractor peaks via histogram + ``maximum_filter1d``; assign
       pixels to their nearest peak via cKDTree.
       *Omnipose*: cluster convergence positions with DBSCAN (``eps = max_cluster_dist``).
    6. Split disconnected regions that share a cluster label into separate instances.
    7. *Cellpose only*: optionally remove instances with inconsistent flow.
    8. Remove small objects and relabel sequentially.

    Parameters
    ----------
    pred : (Y, X, C) or (Z, Y, X, C) float ndarray
        Full model prediction (spatial dims + channel dim last).
    channels : list of str
        Channel names matching the last axis of ``pred``,
        e.g. ``["F", "Gv", "Gh"]``, ``["B", "Gv", "Gh", "Gz"]``.
    flow_type : {"cellpose", "omnipose"}, optional
        Post-processing strategy.  "cellpose" uses histogram peak detection and
        step suppression; "omnipose" uses DBSCAN clustering and constant step
        size.  Default "cellpose".
    fg_channel : str, optional
        Name of a channel to threshold for the foreground mask.
        Use ``"B"`` for the background channel (mask is inverted).
        Leave empty to derive the mask from the flow magnitude (background
        flow is identically zero by construction).
    fg_thresh : float, optional
        Sigmoid-space threshold applied to ``fg_channel``.  Default 0.5.
    flow_threshold : float, optional
        *Cellpose only.*  Flow consistency error above which an instance is
        removed. The error is ``1 - cosine_similarity(predicted, centroid-pointing)``,
        so 0 = perfect, 2 = opposite.  Set ≤ 0 to skip this check.
        Default 0.4 (matches Cellpose default).  Ignored when ``flow_type="omnipose"``.
    n_steps : int, optional
        Number of Euler integration steps.  Default 200.
    dt : float, optional
        Physical-unit step size per iteration.  Default 1.0.
    suppressed : bool, optional
        *Cellpose only.*  If False (default, matches Cellpose 4.x), use a
        constant step ``dt`` per iteration — total travel = ``n_steps * dt``
        pixels, sufficient for cells of any size.  If True, decay the step as
        ``dt / (t + 1)`` — total travel ≈ ``5.9 * dt`` pixels, which can be
        too short for cells larger than ~12 px diameter.  Always forced False
        for Omnipose.  Default False.
    min_size : int, optional
        Minimum instance size in voxels; smaller instances are discarded.
        Default 15.
    max_cluster_dist : float, optional
        *Cellpose*: maximum pixel distance from a pixel's convergence position
        to a valid attractor peak; pixels farther away are treated as noise.
        *Omnipose*: DBSCAN ``eps`` neighbourhood radius in pixels.
        Default 5.0.
    resolution : list of float, optional
        Physical voxel size ``[z, y, x]`` used to convert physical flow steps to
        pixel steps during integration.  For isotropic data use ``[1, 1, 1]``.
        Default [1.0, 1.0, 1.0].
    auto_adjust : bool, optional
        *Cellpose only.*  If True, analyse the foreground channel (``fg_channel``
        must be set) to estimate a representative small-cell radius for this
        image and override ``min_size`` and ``max_cluster_dist`` accordingly:

        * ``min_size``  ← max(1, int(0.5 × estimated_small_cell_area))
        * ``max_cluster_dist`` ← max(5.0, estimated_radius)

        This adapts post-processing to images that contain cells of mixed
        scales without requiring manual parameter tuning.  If the foreground
        channel is absent or too few connected components are detected, the
        provided values of ``min_size`` and ``max_cluster_dist`` are kept.
        Ignored when ``flow_type="omnipose"``.  Default False.

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

        # ── 2a. Auto-adjust min_size / max_cluster_dist (Cellpose only) ───
        if auto_adjust and flow_type == "cellpose":
            _fg_for_estimate = 1.0 - fg_raw if fg_channel == "B" else fg_raw
            radius, stats = _estimate_cell_radius(_fg_for_estimate, fg_thresh, is_3d)
            if radius is not None:
                small_area = (4.0 / 3.0 * np.pi * radius ** 3) if is_3d else (np.pi * radius ** 2)
                adj_min_size = max(1, int(0.5 * small_area))
                adj_max_cluster_dist = max(5.0, radius)
                print(
                    f"  [auto_adjust] foreground component analysis ({stats['n_components']} components detected):\n"
                    f"    area  — min={stats['min_area']} px, "
                    f"median={stats['median_area']:.0f} px, "
                    f"max={stats['max_area']} px\n"
                    f"    {stats['size_percentile']:.0f}th-percentile area={stats['percentile_area']:.1f} px "
                    f"→ equivalent radius={radius:.1f} px\n"
                    f"    min_size:        {min_size} → {adj_min_size}\n"
                    f"    max_cluster_dist: {max_cluster_dist:.1f} → {adj_max_cluster_dist:.1f}"
                )
                min_size = adj_min_size
                max_cluster_dist = adj_max_cluster_dist
            else:
                print(
                    "  [auto_adjust] too few foreground components to estimate cell radius "
                    "(need ≥ 3 separated instances in the foreground channel); "
                    "keeping provided values: "
                    f"min_size={min_size}, max_cluster_dist={max_cluster_dist:.1f}"
                )
    else:
        # Background flow is identically zero by construction (GT masking ensures
        # this); any pixel with non-zero magnitude is foreground.
        mag = np.sqrt(Gv ** 2 + Gh ** 2 + (Gz ** 2 if Gz is not None else 0.0))
        fg_mask = mag > 0.0

    fg_mask = fg_mask.astype(bool)
    if not fg_mask.any():
        return np.zeros(spatial_shape, dtype=np.int32)

    # ── 3. Re-normalise flow to unit vectors ──────────────────────────────
    #    The network output (tanh) is already near unit length but not exact.
    if Gz is not None:
        mag = np.sqrt(Gv ** 2 + Gh ** 2 + Gz ** 2) + 1e-8
        Gz = (Gz / mag).astype(np.float32)
    else:
        mag = np.sqrt(Gv ** 2 + Gh ** 2) + 1e-8
    Gv = (Gv / mag).astype(np.float32)
    Gh = (Gh / mag).astype(np.float32)

    # ── 4. Foreground pixel coordinates ──────────────────────────────────
    fg_coords = np.stack(np.nonzero(fg_mask), axis=1)  # (N, ndim)

    # ── 5. Resolution slice matching ndim ────────────────────────────────
    #    resolution is always stored as [z, y, x]; for 2D take the last two.
    res = np.array(resolution[-ndim:], dtype=np.float32)

    # Ordered flow tuple: (Gz, Gv, Gh) for 3D, (Gv, Gh) for 2D
    flow_comps: List[NDArray] = ([Gz, Gv, Gh] if Gz is not None else [Gv, Gh])  # type: ignore[list-item]

    # ── 6. Euler integration ──────────────────────────────────────────────
    # Omnipose always uses constant step size; Cellpose uses time-suppressed steps.
    _suppressed = False if flow_type == "omnipose" else suppressed
    print(f"  Flow integration ({flow_type}): {len(fg_coords):,} foreground pixels, {n_steps} steps ...")
    final_pos = _euler_integrate(flow_comps, fg_coords, n_steps, dt, _suppressed, res)

    # ── 7. Cluster convergence positions into instances ───────────────────
    print("  Clustering convergence positions ...")
    if flow_type == "omnipose":
        # DBSCAN on convergence positions: each cluster = one cell.
        # eps = max_cluster_dist (same semantic: how close positions must be to
        # belong to the same instance).
        labels = _dbscan_cluster(final_pos, fg_coords, spatial_shape, min_size, max_cluster_dist)
    else:
        # Cellpose: histogram peak detection + nearest-peak assignment.
        labels = _cluster_to_instances(final_pos, fg_coords, spatial_shape, min_size, max_cluster_dist)

    # ── 8. Optional flow consistency check (Cellpose only) ────────────────
    # Omnipose flows point toward the EDT skeleton by construction, so the
    # centroid-based consistency approximation is not meaningful there.
    if flow_type == "cellpose" and flow_threshold > 0.0 and int(labels.max()) > 0:
        print(f"  Flow consistency check (threshold={flow_threshold}) ...")
        errors = _flow_consistency_error(Gv, Gh, labels, Gz)
        n_before = int(labels.max())
        labels = _remove_bad_flow_masks(labels, errors, flow_threshold)
        n_removed = n_before - int(labels.max())
        if n_removed:
            print(f"    Removed {n_removed} instance(s) with inconsistent flow.")

    return labels