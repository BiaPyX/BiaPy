# SPDX-License-Identifier: BSD-3-Clause
# Adapted in part from StarDist (https://github.com/stardist/stardist)
# StarDist is licensed under the BSD 3-Clause License:
#
# Copyright (c) 2018, Uwe Schmidt, Martin Weigert, and contributors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DAMAGES ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE.

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, Sequence
from scipy.ndimage import maximum_filter
from scipy.spatial import ConvexHull
from skimage.draw import polygon as sk_polygon 

from biapy.data.pre_processing import generate_rays

# ---------------------------
# Generic helpers (ndim-aware)
# ---------------------------

def _ensure_tuple(v, ndim):
    """
    Ensure input is a tuple of the specified dimensionality.
    If v is a single value, replicate it ndim times.    
    If v is a list/tuple/ndarray, ensure its length is ndim.

    Parameters
    ----------
    v : scalar or sequence
        Input value(s).
    ndim : int
        Desired dimensionality of the output tuple.

    Returns
    -------
    out : tuple of float
        Tuple of length ndim.
    """
    if isinstance(v, (list, tuple, np.ndarray)):
        assert len(v) == ndim, f"Expected {ndim} values, got {v}"
        return tuple(float(x) for x in v)
    return tuple([float(v)] * ndim)

def local_maxima_nd(prob: np.ndarray, footprint: int | Sequence[int] = 3, thresh: float = 0.5):
    """
    Find local maxima in n-dimensional probability map.

    Parameters
    ----------
    prob : ndarray
        n-dimensional probability map.
    footprint : int or tuple of int
        Size of the maximum filter. Use an odd integer or a tuple of odd integers (one per spatial dim).
    thresh : float
        Minimum value to be considered a peak.  

    Returns
    -------
    coords : (K,ndim) ndarray of int
        Coordinates of K peaks in axis order.
    """
    size = footprint if isinstance(footprint, (list, tuple)) else (footprint,) * prob.ndim
    mf = maximum_filter(prob, size=size, mode='nearest')
    peaks = (prob == mf) & (prob >= thresh)
    coords = np.stack(np.nonzero(peaks), axis=1)  # (K, ndim)
    return coords

def build_vertices(center: Sequence[float], dists: np.ndarray, rays: np.ndarray, anisotropy: Sequence[float]):
    """
    Build vertices (R,D) from center (D,), dists (R,), rays (R,D) in Cartesian order,
    and anisotropy (D,) in axis order.  

    Parameters
    ----------
    center : sequence of float
        Center of the shape in axis order (y,x) or (z,y,x).
    dists : (R,) ndarray
        Radial distances for R rays.
    rays : (R,D) ndarray
        Unit rays in Cartesian order (dx,dy[,dz]).
    anisotropy : tuple of float
        Voxel anisotropy (z,y,x) for 3D or (y,x) for 2D.    

    Returns
    -------
    verts : (R,D) ndarray
        Vertices of the shape in axis order (y,x) or (z,y,x).
    """
    center = np.asarray(center, dtype=np.float32)
    dists  = np.asarray(dists,  dtype=np.float32)  # (R,)
    rays   = np.asarray(rays,   dtype=np.float32)  # (R,D)
    D = rays.shape[1]
    an = np.asarray(_ensure_tuple(anisotropy, D), dtype=np.float32)

    # Map Cartesian rays (dx,dy[,dz]) to axis order (y,x) or (z,y,x)
    if D == 2:
        # axis order (y,x) = (center[0], center[1])
        dy = rays[:, 1]; dx = rays[:, 0]
        verts = np.stack([center[0] + dists * dy * an[0],
                          center[1] + dists * dx * an[1]], axis=1)
    elif D == 3:
        dz = rays[:, 2]; dy = rays[:, 1]; dx = rays[:, 0]
        verts = np.stack([center[0] + dists * dz * an[0],
                          center[1] + dists * dy * an[1],
                          center[2] + dists * dx * an[2]], axis=1)
    else:
        raise ValueError("Only 2D/3D supported.")
    return verts.astype(np.float32)

def _roi_from_vertices(verts: np.ndarray, shape: Tuple[int, ...], pad: int = 2):
    """
    Compute tight ROI from vertices with padding and clipping to shape.

    Parameters
    ----------
    verts : (R,D) ndarray
        Vertices of the shape in axis order (y,x) or (z,y,x).
    shape : tuple of int
        Shape of the image/volume for clipping.
    pad : int
        Padding in voxels for the ROI. Default is 2.

    Returns
    -------
    mins : (D,) ndarray of int
        Minimum coordinates of the ROI (inclusive).
    maxs : (D,) ndarray of int
        Maximum coordinates of the ROI (inclusive).
    """
    mins = np.maximum(np.floor(verts.min(axis=0)).astype(int) - pad, 0)
    maxs = np.minimum(np.ceil (verts.max(axis=0)).astype(int) + pad, np.array(shape) - 1)
    return mins, maxs  # inclusive

def _rasterize_2d(verts_axis: np.ndarray, shape: Tuple[int,int]) -> np.ndarray:
    """
    Rasterize a 2D shape given its vertices.

    Parameters
    ----------
    verts_axis : (R,2) ndarray
        Vertices of the shape in axis order (y,x).
    shape : tuple of int
        Shape of the image for clipping.

    Returns
    -------
    mask : (H,W) ndarray of bool
        Binary mask of the rasterized shape.
    """
    ys = np.clip(verts_axis[:, 0], 0, shape[0]-1)
    xs = np.clip(verts_axis[:, 1], 0, shape[1]-1)
    rr, cc = sk_polygon(ys, xs, shape=shape)
    mask = np.zeros(shape, dtype=bool)
    mask[rr, cc] = True
    return mask

def _rasterize_3d_convex(verts_axis: np.ndarray, shape: Tuple[int,int,int], pad: int = 2) -> np.ndarray:
    """
    Rasterize a 3D convex shape given its vertices. 

    Parameters
    ----------
    verts_axis : (R,3) ndarray
        Vertices of the shape in axis order (z,y,x).
    shape : tuple of int
        Shape of the image/volume for clipping.
    pad : int
        Padding in voxels for rasterization of the shape. Default is 2.   

    Returns
    -------
    mask : (Z,Y,X) ndarray of bool
        Binary mask of the rasterized shape.
    """
    # tight ROI
    mins, maxs = _roi_from_vertices(verts_axis, shape, pad=pad)
    if np.any(mins > maxs):  # empty
        return np.zeros(shape, dtype=bool)
    (z0,y0,x0), (z1,y1,x1) = mins, maxs
    sub_shape = (z1-z0+1, y1-y0+1, x1-x0+1)

    # shift verts into ROI coords
    v = verts_axis.copy()
    v[:,0] -= z0; v[:,1] -= y0; v[:,2] -= x0

    hull = ConvexHull(v)  # (R,3)
    A = hull.equations[:, :3]; d = hull.equations[:, 3:4]  # (F,3), (F,1)

    zz, yy, xx = np.mgrid[0:sub_shape[0], 0:sub_shape[1], 0:sub_shape[2]].astype(np.float32)
    pts = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=1)  # (P,3)
    inside = np.all(A @ pts.T + d <= 1e-6, axis=0)
    mask_sub = inside.reshape(sub_shape)

    mask = np.zeros(shape, dtype=bool)
    mask[z0:z1+1, y0:y1+1, x0:x1+1] = mask_sub
    return mask

def iou_vertices(va: np.ndarray, vb: np.ndarray, shape: Tuple[int,...], pad: int = 2) -> float:
    """
    Compute IoU of two shapes given by their vertices (R,D) in axis order (y,x) or (z,y,x).
    Shapes are rasterized in a joint tight ROI. Return 0.0 if no overlap.   

    Parameters  
    ----------
    va : (R,D) ndarray
        Vertices of shape A in axis order (y,x) or (z,y,x).
    vb : (R,D) ndarray
        Vertices of shape B in axis order (y,x) or (z,y,x).
    shape : tuple of int
        Shape of the image/volume for clipping.
    pad : int
        Padding in voxels for rasterization of shapes. Default is 2.    
    Returns
    -------
    iou : float
        Intersection-over-union of the two shapes.
    """
    D = va.shape[1]
    if D == 2:
        mins_a, maxs_a = _roi_from_vertices(va, shape, pad=pad)
        mins_b, maxs_b = _roi_from_vertices(vb, shape, pad=pad)
        mins = np.maximum(mins_a, mins_b); maxs = np.minimum(maxs_a, maxs_b)
        if np.any(mins > maxs):
            return 0.0
        (y0,x0), (y1,x1) = mins, maxs
        sub_shape = (y1-y0+1, x1-x0+1)
        va_r = va.copy(); vb_r = vb.copy()
        va_r[:,0] -= y0; va_r[:,1] -= x0
        vb_r[:,0] -= y0; vb_r[:,1] -= x0
        ma = _rasterize_2d(va_r, sub_shape)
        mb = _rasterize_2d(vb_r, sub_shape)
    elif D == 3:
        mins_a, maxs_a = _roi_from_vertices(va, shape, pad=pad)
        mins_b, maxs_b = _roi_from_vertices(vb, shape, pad=pad)
        mins = np.maximum(mins_a, mins_b); maxs = np.minimum(maxs_a, maxs_b)
        if np.any(mins > maxs):
            return 0.0
        (z0,y0,x0), (z1,y1,x1) = mins, maxs
        sub_shape = (z1-z0+1, y1-y0+1, x1-x0+1)
        va_r = va.copy(); vb_r = vb.copy()
        va_r[:,0] -= z0; va_r[:,1] -= y0; va_r[:,2] -= x0
        vb_r[:,0] -= z0; vb_r[:,1] -= y0; vb_r[:,2] -= x0
        # voxelize convex hulls in joint ROI
        hull_a = ConvexHull(va_r); hull_b = ConvexHull(vb_r)
        A_a, d_a = hull_a.equations[:, :3], hull_a.equations[:, 3:4]
        A_b, d_b = hull_b.equations[:, :3], hull_b.equations[:, 3:4]
        zz, yy, xx = np.mgrid[0:sub_shape[0], 0:sub_shape[1], 0:sub_shape[2]].astype(np.float32)
        pts = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=1)
        in_a = np.all(A_a @ pts.T + d_a <= 1e-6, axis=0).reshape(sub_shape)
        in_b = np.all(A_b @ pts.T + d_b <= 1e-6, axis=0).reshape(sub_shape)
        ma, mb = in_a, in_b
    else:
        raise ValueError("Only 2D/3D supported.")

    inter = np.count_nonzero(ma & mb)
    if inter == 0: return 0.0
    union = np.count_nonzero(ma | mb)
    return float(inter) / float(union) if union > 0 else 0.0

def dist_to_coord_nd(dist: np.ndarray, points: np.ndarray, rays: np.ndarray, anisotropy=(1,1,1)):
    """
    StarDist-like: (N,R) distances -> (N,D,R) coordinates in axis order.
    For D=2: rows [ys, xs]; for D=3: [zs, ys, xs].

    Parameters
    ----------
    dist : (N,R) ndarray
        Radial distances for N shapes and R rays.
    points : (N,D) ndarray
        Shape centers in axis order (y,x) or (z,y,x).
    rays : (R,D) ndarray
        Unit rays in Cartesian order (dx,dy[,dz]).
    anisotropy : tuple of float
        Voxel anisotropy (z,y,x) for 3D or (y,x) for 2D. Default is isotropic (1,1[,1]).

    Returns
    -------
    coord : (N,D,R) ndarray
        Coordinates of N shapes with R vertices in axis order (y,x) or (z,y,x).
    """
    dist   = np.asarray(dist, dtype=np.float32)   # (N,R)
    points = np.asarray(points, dtype=np.float32) # (N,D) axis order
    rays   = np.asarray(rays,  dtype=np.float32)  # (R,D) Cartesian order
    N, R = dist.shape; D = points.shape[1]
    an = np.asarray(_ensure_tuple(anisotropy, D), dtype=np.float32)

    # Build offsets in axis order from Cartesian rays
    if D == 2:
        dy = rays[:,1]; dx = rays[:,0]
        off = np.stack([dy, dx], axis=0)  # (2,R)
    elif D == 3:
        dz = rays[:,2]; dy = rays[:,1]; dx = rays[:,0]
        off = np.stack([dz, dy, dx], axis=0)  # (3,R)
    else:
        raise ValueError("Only 2D/3D supported.")

    coord = dist[:, None, :] * off[None, :, :]        # (N,D,R)
    coord *= an.reshape(1, D, 1)
    coord += points[..., None]
    return coord

def _shapes_to_label_coord(coord: np.ndarray, shape: Tuple[int,...], labels: Optional[np.ndarray]=None, pad_3d: int = 2):
    """
    Rasterize shapes given coord (N,D,R) into label image/volume.
    If labels is provided, id = labels[i] + 1 (StarDist-like).

    Parameters
    ----------
    coord : (N,D,R) ndarray
        Coordinates of N shapes with R vertices in axis order (y,x) or (z,y,x).
    shape : tuple of int
        Shape of the output label image/volume.
    labels : (N,) ndarray or None
        Labels for the N shapes. If None, labels are 0..N-1.
    pad_3d : int
        Padding in voxels for rasterization of 3D shapes. Default is 2.
    
    Returns
    -------
    out : ndarray of int
        Label image/volume with painted shapes.
    """
    N, D, R = coord.shape
    if labels is None:
        labels = np.arange(N, dtype=int)
    out = np.zeros(shape, np.int32)
    for i, c in zip(labels, coord):
        if D == 2:
            mask = _rasterize_2d(np.stack([c[0], c[1]], axis=1), shape)
        elif D == 3:
            mask = _rasterize_3d_convex(np.stack([c[0], c[1], c[2]], axis=1), shape, pad=pad_3d)
        else:
            raise ValueError("Only 2D/3D supported.")
        out[mask] = int(i) + 1
    return out

def shapes_to_label(dist: np.ndarray, points: np.ndarray, rays: np.ndarray,
                    shape: Tuple[int,...], prob: Optional[np.ndarray]=None,
                    thr: float=-np.inf, anisotropy=(1,1,1), pad_3d: int=2):
    """
    StarDist-like painter: filter by prob>thr, sort by prob ascending, then paint.

    Parameters
    ----------
    dist : (N,R) ndarray
        Radial distances for N shapes and R rays.
    points : (N,D) ndarray
        Shape centers in axis order (y,x) or (z,y,x).
    rays : (R,D) ndarray
        Unit rays in Cartesian order (dx,dy[,dz]).
    shape : tuple of int
        Shape of the output label image/volume.
    prob : (N,) ndarray or None
        Probabilities for the N shapes. If None, all shapes are used.
    thr : float
        Probability threshold for shape inclusion.
    anisotropy : tuple of float
        Voxel anisotropy (z,y,x) for 3D or (y,x) for 2D. Default is isotropic (1,1[,1]).
    pad_3d : int
        Padding in voxels for rasterization of 3D shapes. Default is 2.

    Returns
    -------
    labels : ndarray of int
        Label image/volume with painted shapes.
    """
    if prob is None:
        prob = np.inf * np.ones(points.shape[0], dtype=np.float32)
    ok = prob > thr
    dist, points, prob = dist[ok], points[ok], prob[ok]
    order = np.argsort(prob, kind='stable')  # ascending
    dist, points, prob = dist[order], points[order], prob[order]
    coord = dist_to_coord_nd(dist, points, rays, anisotropy=anisotropy)
    return _shapes_to_label_coord(coord, shape=shape, labels=order, pad_3d=pad_3d)


# ---------------------------------
# Greedy IoU-NMS + full pipeline
# ---------------------------------
def stardist_instances_from_prediction(
    prob: np.ndarray,            # (Hc,Wc) or (Zc,Yc,Xc) on lattice
    dist: np.ndarray,            # (Hc,Wc,R) or (Zc,Yc,Xc,R) on lattice
    prob_thresh: float = 0.5,
    nms_iou_thresh: float = 0.3,
    anisotropy: Sequence[float] = (1.0, 1.0, 1.0),
    peak_footprint: int | Sequence[int] = 3,
    max_proposals: Optional[int] = None,
    thr_for_painter: float = -np.inf,
    return_shapes: bool = False,
    pad_vox: int = 2,
    grid: Sequence[int] | int = (1, 1, 1),
):
    """
    Convert StarDist probability and radial-distance predictions into instance labels
    via polygon/polyhedron IoU-NMS and StarDist-style rasterization.

    Now supports *lattice outputs* via `grid`. If the network predicts on a coarse lattice
    (e.g., StarDist pretrained models with grid=(2,2)), pass that grid here so candidate
    centers are placed at full-resolution coordinates while distances remain in pixel units.

    Parameters
    ----------
    prob : 2D or 3D Numpy array (float32)
        Probability map on the output lattice:
        - 2D: ``(Hc, Wc)`` where Hc = H/gy, Wc = W/gx
        - 3D: ``(Zc, Yc, Xc)``

    dist : 3D or 4D Numpy array (float32)
        Radial distances on the output lattice. The last dim is the number of rays ``R``.
        - 2D: ``(Hc, Wc, R)``
        - 3D: ``(Zc, Yc, Xc, R)``

    prob_thresh : float, optional
        Probability threshold for candidate detection.

    nms_iou_thresh : float, optional
        IoU threshold for greedy NMS.

    anisotropy : tuple of float, optional
        Voxel anisotropy (z,y,x) for 3D or (y,x) for 2D. Default is isotropic (1,1[,1]).

    peak_footprint : int or tuple of int, optional
        Size of the maximum filter for peak detection   in ``prob``.
        Use an odd integer or a tuple of odd integers (one per spatial dim). 

    max_proposals : int or None, optional
        If not None, consider at most this many candidates (highest ``prob``) for NMS.
        Default is None (all candidates above ``prob_thresh``). 
    
    thr_for_painter : float, optional
        Probability threshold for the StarDist-like painter. Default is -inf (all shapes painted).  
    
    return_shapes : bool, optional
        If True, return the vertices of the shapes kept after NMS in ``details["shapes_vertices"]``.
    
    pad_vox : int, optional
        Padding in voxels for rasterization of 3D shapes. Default is 2. 

    grid : tuple of int, optional
        Lattice stride in pixels (axis order). For 2D use ``(gy, gx)``, for 3D ``(gz, gy, gx)``.
        Use ``(1,1[,1])`` if predictions are already at full resolution.

    Returns
    -------
    labels : Numpy array (int32)
        Full-resolution label image/volume. Shape is ``(Hc*gy, Wc*gx)`` in 2D or
        ``(Zc*gz, Yc*gy, Xc*gx)`` in 3D.

    details : dict
        - ``points``: full-res integer centers kept after NMS (axis order).
        - ``scores``: probabilities at those centers.
        - optionally shapes in ``shapes_vertices`` if ``return_shapes=True``.
    """
    D = prob.ndim
    assert dist.shape[:D] == prob.shape, "dist spatial dims must match prob"
    R = dist.shape[-1]
    anisotropy = _ensure_tuple(anisotropy, D)
    grid = _ensure_tuple(grid, D)  # e.g., (gy,gx) or (gz,gy,gx)
    rays = generate_rays(R, ndim=D)

    # full-resolution output shape
    shape_coarse = prob.shape
    shape_full = tuple(int(shape_coarse[d] * grid[d]) for d in range(D))

    # 1) candidate centers on the *coarse* lattice, then map to *full-res* coords
    coords_c = local_maxima_nd(prob, footprint=peak_footprint, thresh=prob_thresh)  # (K,D) lattice indices
    if coords_c.size == 0:
        labels = np.zeros(shape_full, dtype=np.int32)
        return labels, {"points": np.zeros((0, D), int), "scores": np.zeros((0,), np.float32)}
    scores = prob[tuple(coords_c.T)].astype(np.float32)
    order = np.argsort(-scores)
    coords_c, scores = coords_c[order], scores[order]
    if max_proposals is not None and len(scores) > max_proposals:
        coords_c, scores = coords_c[:max_proposals], scores[:max_proposals]

    # map lattice indices -> full-res pixel centers (axis order)
    coords_full = coords_c.astype(np.float32) * np.asarray(grid, np.float32)[None, :]

    # 2) build vertices for NMS in full-res coordinates
    verts_list = []
    for p_c, p_full in zip(coords_c, coords_full):
        d = dist[tuple(p_c)][...]  # (R,)
        verts = build_vertices(p_full, d, rays, anisotropy)  # full-res vertices
        verts_list.append(verts)

    # 3) greedy IoU-NMS (in full-res space)
    keep = []
    for i in range(len(scores)):
        vi = verts_list[i]
        ok = True
        for j in keep:
            vj = verts_list[j]
            if iou_vertices(vi, vj, shape_full, pad=pad_vox) > nms_iou_thresh:
                ok = False
                break
        if ok:
            keep.append(i)
    keep = np.asarray(keep, dtype=int)

    kept_points_full = np.rint(coords_full[keep]).astype(int)  # integer centers in full-res
    kept_scores = scores[keep]
    kept_dists  = dist[tuple(coords_c[keep].T)].reshape(len(keep), R)

    # 4) StarDist-like painting at full resolution
    labels = shapes_to_label(
        dist=kept_dists,
        points=kept_points_full,
        rays=rays,
        shape=shape_full,
        prob=kept_scores,
        thr=thr_for_painter,
        anisotropy=anisotropy,
        pad_3d=pad_vox
    )

    details = {"points": kept_points_full, "scores": kept_scores}
    if return_shapes:
        details["shapes_vertices"] = [verts_list[i] for i in keep]
    details["grid"] = grid
    return labels, details

