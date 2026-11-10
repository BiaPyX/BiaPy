"""
Preview of the data selected in the wizard: a few raw images next to their targets, paired as BiaPy pairs them, as PNG
thumbnails, and warnings about anything that looks wrong for the workflow.
"""
import base64
import io
import os
import re

import numpy as np

from biapy.wizard.data_check import folder_images, preview_image
from biapy.wizard.fingerprint import sample_indices


ROWS = 3  # pairs shown
CHECKED = 6  # pairs inspected for the warnings
THUMB = 220  # max. side of the thumbnails, in pixels
POINTS_CROP = 128  # side of the full-resolution crops where the Wizard draws the points with the radius chosen
POINTS_CROP_DZ = 10  # points kept up to this many slices away from the slice of a crop (for the Z radius)
# More labels than this, each a single connected object, look like instance masks
INSTANCE_LIKE_LABELS = 10
# A label split in more objects than this, with few labels, looks like a semantic/binary mask
SEMANTIC_LIKE_OBJECTS = 3

IMAGE_TARGETS = ("DENOISING", "SUPER_RESOLUTION", "IMAGE_TO_IMAGE")
TARGET_KIND = {
    "SEMANTIC_SEG": "semantic masks",
    "INSTANCE_SEG": "instance masks",
    "DETECTION": "CSV files with the points",
    "DENOISING": "clean images",
    "SUPER_RESOLUTION": "high-resolution images",
    "IMAGE_TO_IMAGE": "target images",
}


###############
# Rendering   #
###############
def _png(rgb):
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _resize(img, order):
    from skimage.transform import resize

    h, w = img.shape[:2]
    s = min(1.0, THUMB / max(h, w))
    if s == 1.0:
        return img
    shape = (max(1, int(round(h * s))), max(1, int(round(w * s)))) + img.shape[2:]
    return resize(img, shape, order=order, preserve_range=True, anti_aliasing=order > 0).astype(img.dtype)


def _plane(img):
    """2D view of an image read by preview_image(): (y, x, c), the middle slice for 3D."""
    return img[img.shape[0] // 2] if img.ndim == 4 else img


# Intensities clipped to these percentiles and scaled to [0, 1] for display (any data type, e.g. uint16 or float)
DISPLAY_PERCENTILES = (0.1, 99.8)


def _intensity_rgb(plane):
    """Grey (1 channel), green/magenta (2) or RGB (3+), clipped to DISPLAY_PERCENTILES and scaled to [0, 1]."""
    p = plane.astype(np.float32)
    lo, hi = np.percentile(p, DISPLAY_PERCENTILES)
    p = np.clip((p - lo) / (hi - lo if hi > lo else 1), 0, 1)
    c = p.shape[-1]
    if c == 1:
        rgb = np.repeat(p, 3, axis=-1)
    elif c == 2:
        rgb = np.stack([p[..., 1], p[..., 0], p[..., 1]], -1)
    else:
        rgb = p[..., :3]
    return (rgb * 255).astype(np.uint8)


def _label_colors(n, seed):
    rng = np.random.default_rng(seed)
    colors = rng.uniform(0.25, 1, (n + 1, 3))
    colors[0] = 0
    return (colors * 255).astype(np.uint8)


def _labels_rgb(plane, semantic):
    """Labels as colors: one fixed color per class (semantic) or random ones per instance."""
    lab = plane[..., 0]
    vals, inv = np.unique(lab, return_inverse=True)
    inv = inv.reshape(lab.shape)
    if semantic:
        palette = np.array([[0, 0, 0], [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
                            [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60]], np.uint8)
        idx = np.where(vals[inv] == 0, 0, 1 + (inv - (1 if vals[0] == 0 else 0)) % (len(palette) - 1))
        return palette[idx]
    colors = _label_colors(len(vals), seed=7)
    rgb = colors[inv]
    rgb[lab == 0] = 0
    return rgb


def _points_rgb(raw_plane, points, z_mid):
    """Raw image with the points drawn (3D: the points near the middle slice)."""
    from skimage.draw import disk

    rgb = _intensity_rgb(raw_plane)
    r = max(2, int(round(max(rgb.shape[:2]) / THUMB * 2)))
    for p in points:
        if z_mid is not None and abs(p[0] - z_mid) > 2:
            continue
        y, x = p[-2], p[-1]
        rr, cc = disk((y, x), r, shape=rgb.shape[:2])
        rgb[rr, cc] = (255, 40, 40)
    return rgb


def points_crop(raw, pts, is_3d):
    """
    A POINTS_CROP square of ``raw`` (as read by preview_image()) at full resolution, around the point with the most
    points near it (3D: in its slice), and the points in it as [dz, y, x] crop pixels (truncated, as BiaPy paints
    them), for the Wizard to draw the training targets of a point radius. None without points.
    """
    p = pts.astype(int)
    if not is_3d:
        p = np.concatenate([np.zeros((len(p), 1), int), p], axis=1)
        raw = raw[None]
    p = p[((p >= 0) & (p < raw.shape[:3])).all(1)]
    if len(p) == 0:
        return None
    half = POINTS_CROP // 2
    rng = np.random.default_rng(0)
    cand = p[rng.choice(len(p), min(len(p), 200), replace=False)]
    ref = p[rng.choice(len(p), min(len(p), 5000), replace=False)]  # bounds the memory of the counts
    near = (np.abs(ref[None, :, 1:] - cand[:, None, 1:]) < half).all(-1)
    near &= np.abs(ref[None, :, 0] - cand[:, None, 0]) <= POINTS_CROP_DZ
    z, cy, cx = cand[near.sum(1).argmax()]
    h, w = raw.shape[1:3]
    y0, x0 = int(np.clip(cy - half, 0, max(0, h - POINTS_CROP))), int(np.clip(cx - half, 0, max(0, w - POINTS_CROP)))
    crop = raw[z, y0 : y0 + POINTS_CROP, x0 : x0 + POINTS_CROP]
    rel = p - [z, y0, x0]
    keep = (np.abs(rel[:, 0]) <= POINTS_CROP_DZ) & (rel[:, 1:] >= 0).all(1)
    keep &= (rel[:, 1] < crop.shape[0]) & (rel[:, 2] < crop.shape[1])
    return {"points_crop_png": _png(_intensity_rgb(crop)), "points_crop": rel[keep].tolist()}


def _read_points(path, is_3d):
    from biapy.data.data_manipulation import read_points_csv

    cols = ["axis-0", "axis-1", "axis-2"] if is_3d else ["axis-0", "axis-1"]
    return read_points_csv(path, is_3d=is_3d)[cols].to_numpy(dtype=float)


###############
# Checks      #
###############
def _numbers(name):
    return [int(v) for v in re.findall(r"\d+", os.path.splitext(name)[0])]


def _check_names(pairs, warnings):
    """Files are paired by order: warn when the names of a pair have nothing in common."""
    odd = [(a, b) for a, b in pairs if os.path.splitext(a)[0] != os.path.splitext(b)[0] and _numbers(a) != _numbers(b)]
    if odd:
        a, b = odd[0]
        warnings.append(
            "Raw images and targets are paired by file order, but their names differ: e.g. '{}' is paired with '{}'. "
            "Check that each raw image has its target in the same position.".format(a, b)
        )


def _check_raw(raws, warnings):
    if raws and all(len(np.unique(r)) <= 2 for r in raws):
        warnings.append(
            "The raw images contain only two values, like masks. Are the raw and target folders swapped?"
        )
    if any(r.max() == r.min() for r in raws):
        warnings.append("Some raw images are completely flat (a single value).")


def _check_label_targets(workflow, targets, warnings, notes):
    if not targets:
        return
    from scipy import ndimage

    if any(not np.allclose(t, np.round(t)) for t in targets):
        warnings.append(
            "The targets contain non-integer values, so they are not label images. The {} workflow expects {}.".format(
                workflow.replace("_", " ").lower(), TARGET_KIND[workflow])
        )
        return
    empty = sum(1 for t in targets if not (t[..., 0] != 0).any())
    if empty == len(targets):
        warnings.append("The targets inspected are empty (all background).")
    elif empty:
        notes.append("{} of the {} targets inspected are empty (all background).".format(empty, len(targets)))
    instance_like, semantic_like = [], []
    for t in targets:
        lab = t[..., 0].astype(np.int64)
        labels = [int(v) for v in np.unique(lab) if v != 0]
        if not labels:
            continue
        boxes = ndimage.find_objects(lab)
        objects = {v: ndimage.label(lab[boxes[v - 1]] == v)[1] for v in labels[:200] if boxes[v - 1] is not None}
        if len(labels) > INSTANCE_LIKE_LABELS and np.median(list(objects.values())) <= 1:
            instance_like.append(len(labels))
        many = max(objects.values())
        if len(labels) <= 3 and many > SEMANTIC_LIKE_OBJECTS:
            semantic_like.append(many)
    if workflow == "SEMANTIC_SEG" and len(instance_like) > len(targets) / 2:
        warnings.append(
            "The targets look like instance masks: each object has its own label (up to {} labels per image). "
            "Semantic segmentation expects one label per class (e.g. 0 background, 1 foreground). If you want to "
            "separate the objects, choose the instance segmentation workflow instead.".format(max(instance_like))
        )
    if workflow == "INSTANCE_SEG" and len(semantic_like) > len(targets) / 2:
        warnings.append(
            "The targets look like semantic/binary masks: up to {} separate objects share the same label. Instance "
            "segmentation expects a different label for each object. If you only need to separate classes, choose "
            "the semantic segmentation workflow instead.".format(max(semantic_like))
        )


def _check_image_targets(workflow, targets, warnings):
    if targets and all(len(np.unique(t)) <= 2 for t in targets):
        warnings.append(
            "The targets contain only two values, like masks. The {} workflow expects {}.".format(
                workflow.replace("_", " ").lower(), TARGET_KIND[workflow])
        )


def _check_shapes(workflow, shapes, warnings, notes):
    """``shapes``: (raw name, raw shape, target shape) of each pair, spatial shapes."""
    from biapy.data.data_manipulation import mismatched_raw_target_files, upscaling_of

    raw = [(name, tuple(rs)) for name, rs, _ in shapes]
    targets = [(name, tuple(ts)) for name, _, ts in shapes]
    upscaling = (1,) * len(raw[0][1])
    if workflow == "SUPER_RESOLUTION":
        upscaling = upscaling_of(raw[0][1], targets[0][1]) or upscaling
    bad = mismatched_raw_target_files(raw, targets, upscaling)
    if not bad:
        if upscaling != (1,) * len(upscaling):
            notes.append("Upscaling between raw images and targets: x{}.".format("x".join(map(str, upscaling))))
        return
    name, _, rs, ts, expected = bad[0]
    if workflow == "SUPER_RESOLUTION" and any(t < r for r, t in zip(rs, ts)):
        warnings.append(
            "The target of '{}' ({}) is smaller than its raw image ({}): in super-resolution the raw images are the "
            "low-resolution ones and the targets the high-resolution ones. Are the raw and target folders swapped?".format(
                name, "x".join(map(str, ts)), "x".join(map(str, rs)))
        )
    else:
        warnings.append("The target of '{}' is {} but {} was expected.".format(
            name, "x".join(map(str, ts)), "x".join(map(str, expected))))


###############
# Preview     #
###############
def _classification_preview(raw_folder, is_3d):
    from biapy.utils.misc import os_walk_clean

    classes = next(os_walk_clean(raw_folder))[1]
    samples, notes = [], []
    for c in classes[:ROWS]:
        files = folder_images(os.path.join(raw_folder, c))
        if not files:
            continue
        img = preview_image(os.path.join(raw_folder, c, files[0]), is_3d)[0]
        samples.append({"raw_name": os.path.join(c, files[0]), "raw_png": _png(_resize(_intensity_rgb(_plane(img)), 1)),
                        "target_name": "class '{}'".format(c), "target_text": c})
    notes.append("{} classes: {}".format(len(classes), ", ".join(classes[:10]) + (" ..." if len(classes) > 10 else "")))
    return {"samples": samples, "warnings": [], "notes": notes, "columns": ["Image", "Class"]}


def preview(workflow, ndim, raw_folder=None, target_folder=None, target_data_path=""):
    """
    Few raw/target pairs as PNG thumbnails (base64) and warnings. ``target_data_path``: path of the targets within
    the Zarr files of ``target_folder``, e.g. "labels.mito" for the label images of OME-Zarr images.

    Returns
    -------
    result : dict
        ``samples`` (list of ``raw_name``, ``raw_png``, ``target_name``, ``target_png`` and, in detection,
        ``points_crop_png`` and ``points_crop``: see points_crop()), ``warnings``, ``notes`` and ``columns`` (titles).
    """
    is_3d = ndim == "3D"
    if workflow == "CLASSIFICATION" and raw_folder:
        return _classification_preview(raw_folder, is_3d)
    warnings, notes = [], []
    raw_files = folder_images(raw_folder) if raw_folder else []
    tgt_files = folder_images(target_folder) if target_folder else []
    if raw_files and tgt_files and len(raw_files) != len(tgt_files):
        warnings.append(
            "There are {} raw images but {} targets: each raw image needs its target.".format(len(raw_files), len(tgt_files))
        )
    # (raw image, target) of each sample, as BiaPy pairs them
    if raw_files and tgt_files and workflow == "DETECTION":
        from biapy.data.data_manipulation import pair_csv_with_images

        matched = pair_csv_with_images(raw_files, tgt_files)
        pairs = [(img, csv) for (img, _), csv in zip(matched, tgt_files)]
        by_position = [csv for (_, by_name), csv in zip(matched, tgt_files) if not by_name]
        if by_position:
            warnings.append(
                "{} CSV files have no image with the same name (e.g. '{}'), so they are paired with the image in the "
                "same position. Check that it is right.".format(len(by_position), by_position[0])
            )
    elif raw_files and tgt_files:
        pairs = list(zip(raw_files, tgt_files))
    else:
        pairs = [(f, None) for f in raw_files] if raw_files else [(None, f) for f in tgt_files]
    idx = sample_indices(len(pairs), CHECKED)
    shown = {idx[j] for j in sample_indices(len(idx), ROWS)}
    if raw_files and tgt_files and workflow != "DETECTION":
        _check_names([pairs[i] for i in idx], warnings)

    samples, raws, targets, shapes = [], [], [], []
    cropped = False  # large Zarr/H5 images read on a central block
    for i in idx:
        raw_name, tgt_name = pairs[i]
        raw, raw_shape = preview_image(os.path.join(raw_folder, raw_name), is_3d) if raw_name else (None, None)
        raw_cropped = raw is not None and tuple(raw.shape) != tuple(raw_shape)
        cropped |= raw_cropped
        s = {}
        if raw is not None:
            raws.append(raw)
            s.update(raw_name=raw_name, raw_png=_png(_resize(_intensity_rgb(_plane(raw)), 1)))
        if tgt_name:
            path = os.path.join(target_folder, tgt_name)
            s["target_name"] = tgt_name
            if workflow == "DETECTION":
                pts = _read_points(path, is_3d)
                if raw is not None:
                    spatial = raw_shape[:-1]
                    out = np.sum(np.any((pts < 0) | (pts >= np.array(spatial)), axis=1)) if len(pts) else 0
                    if out:
                        warnings.append("{} points of '{}' are outside its image ({}).".format(
                            out, tgt_name, "x".join(map(str, spatial))))
                    if raw_cropped:  # the points are not drawn on a part of the image
                        s["target_text"] = "{} points".format(len(pts))
                    else:
                        s["target_png"] = _png(
                            _resize(_points_rgb(_plane(raw), pts, raw.shape[0] // 2 if is_3d else None), 1)
                        )
                        if i in shown:
                            s.update(points_crop(raw, pts, is_3d) or {})
                else:
                    s["target_text"] = "{} points".format(len(pts))
                if len(pts) == 0:
                    notes.append("'{}' has no points.".format(tgt_name))
            else:
                tgt, tgt_shape = preview_image(path, is_3d, target_data_path)
                cropped |= tuple(tgt.shape) != tuple(tgt_shape)
                targets.append(tgt)
                if raw is not None:
                    shapes.append((raw_name, raw_shape[:-1], tgt_shape[:-1]))
                if workflow in ("SEMANTIC_SEG", "INSTANCE_SEG"):
                    rgb = _labels_rgb(_plane(tgt), semantic=workflow == "SEMANTIC_SEG")
                    s["target_png"] = _png(_resize(rgb, 0))
                else:
                    s["target_png"] = _png(_resize(_intensity_rgb(_plane(tgt)), 1))
        if i in shown:
            samples.append(s)

    _check_raw(raws, warnings)
    if workflow in ("SEMANTIC_SEG", "INSTANCE_SEG"):
        _check_label_targets(workflow, targets, warnings, notes)
    elif workflow in IMAGE_TARGETS:
        _check_image_targets(workflow, targets, warnings)
    if shapes:
        _check_shapes(workflow, shapes, warnings, notes)
    if is_3d:
        notes.append("3D images: the middle slice is shown.")
    if cropped:
        notes.append("Large Zarr/H5 images: a central part of them is shown and inspected.")
    columns = ["Raw image", "Target ({})".format(TARGET_KIND.get(workflow, "target"))]
    return {"samples": samples, "warnings": warnings, "notes": notes, "columns": columns}
