#!/usr/bin/env python3
"""
stardist_wheels.py
------------------
Draw StarDist-style "wheel" overlays (dashed outline + radial spokes) for every
instance in a labeled image.

Usage
-----
python stardist_wheels.py \
  --labels path/to/labels.png \
  --background path/to/background.png \        # optional (grayscale or RGB)
  --n_rays 32 \
  --out overlay.png

Notes
-----
* --labels can be either:
    (a) a 2-D integer label image (0 background; 1..N instances), or
    (b) an RGB color label map (each instance a unique color; black = background).
* If no --background is given, the label image is used as the base.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.color import rgb2gray
from skimage.measure import label as cc_label


# ---------- I/O ----------

def read_image_any(path):
    """Return np.ndarray, preserving channels if present."""
    arr = np.array(Image.open(path))
    return arr


def load_instances_from_labels(path):
    """Load instances from labels (2-D int labels or RGB color map).

    Returns
    -------
    instances : list of boolean masks (H, W)
    H, W      : ints
    """
    L = read_image_any(path)
    if L.ndim == 2:  # integer label image
        H, W = L.shape
        ids = [i for i in np.unique(L) if i != 0]
        instances = [(L == i) for i in ids]
        return instances, H, W

    if L.ndim == 3 and L.shape[2] in (3, 4):  # color label map
        rgb = L[..., :3]
        H, W, _ = rgb.shape
        uniq = np.unique(rgb.reshape(-1, 3), axis=0)
        instances = []
        for col in uniq:
            if np.all(col == [0, 0, 0]):  # background
                continue
            comp = np.all(rgb == col, axis=-1)
            # split color into connected components to handle reused colors
            lab, n = ndi.label(comp)
            for k in range(1, n + 1):
                m = (lab == k)
                if m.sum() >= 10:
                    instances.append(m)
        return instances, H, W

    raise ValueError("Unsupported label image format.")


# ---------- Geometry (StarDist-like) ----------

def pick_center(mask):
    """Choose a center guaranteed to lie inside: maximum of distance transform."""
    dt = ndi.distance_transform_edt(mask)
    cy, cx = np.unravel_index(np.argmax(dt), mask.shape)
    return float(cy), float(cx)


def radial_distances(mask, center, n_rays=32, step=0.5):
    """Cast n_rays equiangular rays from center and march to the boundary.

    Returns
    -------
    angles : (n_rays,) array
    dists  : (n_rays,) array of distances in pixels
    """
    cy, cx = center
    H, W = mask.shape
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    dists = np.zeros(n_rays, dtype=np.float32)

    for i, a in enumerate(angles):
        r = 0.0
        while True:
            y = cy + (r + step) * np.sin(a)
            x = cx + (r + step) * np.cos(a)
            if x < 0 or x >= W or y < 0 or y >= H:
                break
            yi = int(np.floor(y))
            xi = int(np.floor(x))
            if mask[yi, xi]:
                r += step
            else:
                break
        dists[i] = r
    return angles, dists


def polygon_from_rays(center, angles, dists):
    cy, cx = center
    ys = cy + dists * np.sin(angles)
    xs = cx + dists * np.cos(angles)
    return xs, ys


# ---------- Drawing (StarDist-style) ----------

def draw_polygons(ax, base_image, polygons, centers, colors=None,
                  dash=(0, (6, 6)), lw_outline=0.6, lw_spoke=0.25,
                  center_size=6, alpha_spoke=0.85):
    """Mimic the StarDist notebook `_draw_polygons` look.

    Parameters
    ----------
    ax          : matplotlib Axes
    base_image  : HxWx{1,3} array for background
    polygons    : list of (xs, ys)
    centers     : list of (cx, cy)
    colors      : list of RGB triples in [0,1], one per polygon (optional)
    """
    H, W = base_image.shape[:2]
    if base_image.ndim == 2:
        base = np.dstack([base_image]*3)
    else:
        base = base_image

    ax.imshow(base, extent=[0, W, H, 0])
    ax.set_axis_off()

    if colors is None:
        rng = np.random.default_rng(0)
        colors = rng.random((len(polygons), 3))*0.6 + 0.35

    for (xs, ys), (cx, cy), c in zip(polygons, centers, colors):
        # spokes
        for x, y in zip(xs, ys):
            ax.plot([cx, x], [cy, y], lw=lw_spoke, color=c, alpha=alpha_spoke)
        # dashed outline
        ax.plot(np.r_[xs, xs[0]], np.r_[ys, ys[0]], lw=lw_outline, color=c, linestyle=dash)
        # center
        ax.scatter([cx], [cy], s=center_size, color=c, edgecolor="k", linewidth=0.3, zorder=3)

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)


# ---------- Main CLI ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="Path to label image (int labels or RGB color map).")
    ap.add_argument("--background", default=None, help="Optional background (grayscale/RGB) to draw on.")
    ap.add_argument("--n_rays", type=int, default=32, help="Number of equiangular rays.")
    ap.add_argument("--step", type=float, default=0.5, help="Ray marching step size in pixels.")
    ap.add_argument("--out", default="overlay.png", help="Output path for overlay image.")
    ap.add_argument("--lw_outline", type=float, default=0.6, help="Dashed contour linewidth.")
    ap.add_argument("--lw_spoke", type=float, default=0.25, help="Radial spoke linewidth.")
    args = ap.parse_args()

    # calling example: python /data/dfranco/BiaPy/biapy/utils/scripts/stardist_wheels.py --labels y/mito_labels.tif --out wheels.png  --lw_outline 0.8 --lw_spoke 0.5

    instances, H, W = load_instances_from_labels(args.labels)

    # choose base image
    if args.background is None:
        base = read_image_any(args.labels)
        # if labels are 2-D ints, turn into a soft grayscale for nicer look
        if base.ndim == 2:
            base = (base > 0).astype(np.float32)
            # soft vignette per object
            soft = np.zeros((H, W), dtype=np.float32)
            for m in instances:
                dt = ndi.distance_transform_edt(m)
                if dt.max() > 0:
                    soft += (dt / dt.max()) * 0.5
            base = np.clip(soft, 0, 1)
    else:
        base = read_image_any(args.background)
        if base.ndim == 2:
            base = (base - base.min()) / (base.ptp() + 1e-9)

    # compute polygons
    centers, polygons = [], []
    for m in instances:
        cy, cx = pick_center(m)
        ang, dist = radial_distances(m, (cy, cx), n_rays=args.n_rays, step=args.step)
        xs, ys = polygon_from_rays((cy, cx), ang, dist)
        centers.append((cx, cy))
        polygons.append((xs, ys))

    # draw
    dpi = 200
    fig = plt.figure(figsize=(W/dpi, H/dpi), dpi=dpi)
    ax = plt.axes([0, 0, 1, 1])
    draw_polygons(ax, base, polygons, centers, lw_outline=args.lw_outline, lw_spoke=args.lw_spoke)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=dpi)
    plt.close(fig)
    print(f"Saved wheel overlay to: {args.out}")


if __name__ == "__main__":
    main()
