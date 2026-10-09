"""
Dataset fingerprint: properties of the data of a folder that the wizard's strategies use to configure BiaPy, in
the spirit of nnU-Net's dataset fingerprint (https://github.com/MIC-DKFZ/nnUNet).

The fingerprints are accumulated while the folder is checked (:mod:`biapy.wizard.data_check`), which reads
every file once with BiaPy's readers, so the data is not read again here:

* raw images (:class:`ImagesFingerprint`): shapes, voxel size (from the image metadata) and intensities.
* semantic masks (:class:`SemanticMasksFingerprint`): labels, how much of the data each class covers, the
  foreground fraction of each image and the size of the objects of each class along each axis.
* instance masks (:class:`InstanceMasksFingerprint`): number, size and shape of the instances (solidity,
  elongation, thickness, branching, star-convexity) and how often they touch each other.
* detection points (:class:`PointsFingerprint`): points per image and the distance from each point to its
  nearest one.

Detailed statistics are computed on up to :data:`MAX_FILES` files evenly spread over the folder.
"""
import time

import numpy as np

# Files analyzed in detail (evenly spaced along the folder). Shapes and voxel sizes come from all files.
MAX_FILES = 64
# Voxels sampled per file and channel to estimate the intensity statistics
INTENSITY_SAMPLES_PER_FILE = 10000
# Masks with more labels than this are not label images (e.g. a raw image selected by mistake)
MAX_LABELS = 1000
# Instances measured in detail (solidity, star-convexity, skeleton...), spread over the analyzed images
MAX_INSTANCES = 500
# Pixels per instance whose straight path to the instance center is checked for star-convexity
STAR_SAMPLES = 32
# An instance is star-convex if all these sampled pixels see its center (StarDist's assumption)
STAR_VISIBLE_FRACTION = 1.0
# Elongated / thin instances: their skeleton is checked for branches
ELONGATED = 3.0
THIN = 0.25
# Larger instances are downsampled before measuring their shape (e.g. neurons spanning the whole volume)
MAX_CROP_VOXELS = {2: 256**2, 3: 64**3}
# Time for the detailed shape measures of a folder (sizes, foreground and touching use all analyzed images)
SHAPE_TIME_BUDGET = 20.0
# Nearest-neighbour displacements kept to plan detection (the distances depend on the voxel size, which is
# known only when the raw images are analyzed)
MAX_NN_VECTORS = 5000


def sample_indices(n, max_n=MAX_FILES):
    """Up to ``max_n`` indices evenly spaced in ``range(n)``."""
    if n <= max_n:
        return list(range(n))
    return sorted(set(np.linspace(0, n - 1, max_n).round().astype(int).tolist()))


def _shape_stats(spatial_shapes):
    shapes = np.array(spatial_shapes, dtype=float)
    return {
        "median_shape": [int(round(v)) for v in np.median(shapes, axis=0)],
        "min_shape": [int(v) for v in shapes.min(axis=0)],
        "max_shape": [int(v) for v in shapes.max(axis=0)],
        "total_voxels": float(np.prod(shapes, axis=1).sum()),
    }


def _percentiles(values, axis=0):
    return {
        "extent_p10": [float(v) for v in np.percentile(values, 10, axis=axis)],
        "extent_median": [float(v) for v in np.median(values, axis=axis)],
        "extent_p90": [float(v) for v in np.percentile(values, 90, axis=axis)],
    }


class ImagesFingerprint:
    """Raw images: shapes, voxel size and intensity statistics per channel."""

    def __init__(self, n_files, is_3d, seed=0):
        self.n_files = n_files
        self.is_3d = is_3d
        self.analyze = set(sample_indices(n_files))
        self.rng = np.random.default_rng(seed)
        self.shapes, self.voxel_sizes, self.values, self.image_means, self.dtypes = [], [], [], [], set()
        self.channels = None

    def add(self, index, img, meta=None):
        """``img`` as read by BiaPy (``(z, )y, x, c``) and its metadata (``read_img_as_ndarray(load_meta=True)``)."""
        from biapy.data.data_manipulation import voxel_size_from_meta

        self.shapes.append(tuple(img.shape[:-1]))
        self.channels = int(img.shape[-1])
        vs = voxel_size_from_meta(meta, self.is_3d)
        if vs is not None:
            self.voxel_sizes.append(vs)
        if index in self.analyze:
            self.dtypes.add(str(img.dtype))
            flat = img.reshape(-1, img.shape[-1])
            pick = self.rng.integers(0, len(flat), size=min(len(flat), INTENSITY_SAMPLES_PER_FILE))
            self.values.append(flat[pick].astype(np.float64))
            self.image_means.append(flat.mean(axis=0, dtype=np.float64))

    def result(self):
        fp = {"kind": "images", "n_files": self.n_files, "channels": self.channels, "n_analyzed": len(self.values)}
        fp.update(_shape_stats(self.shapes))
        found = np.array(self.voxel_sizes, dtype=float)
        fp["n_calibrated"] = int(len(found))
        fp["spacing"] = None
        fp["spacing_consistent"] = True
        if len(found):
            median = np.median(found, axis=0)
            fp["spacing"] = [float(v) for v in median]
            fp["spacing_consistent"] = bool(np.all(np.abs(found / median - 1) < 0.1))
        fp["dtype"] = sorted(self.dtypes)
        values = np.concatenate(self.values)
        means = np.array(self.image_means)
        fp["intensity"] = []
        for c in range(values.shape[1]):
            v, m = values[:, c], means[:, c]
            fp["intensity"].append(
                {
                    "mean": float(v.mean()),
                    "std": float(v.std()),
                    "min": float(v.min()),
                    "max": float(v.max()),
                    "p00_5": float(np.percentile(v, 0.5)),
                    "p99_5": float(np.percentile(v, 99.5)),
                    # How much the brightness changes from image to image
                    "image_mean_cv": float(m.std() / abs(m.mean())) if m.mean() != 0 else 0.0,
                }
            )
        return fp


def _object_extents(mask):
    """Extent along each axis of the connected components of a boolean mask."""
    from scipy import ndimage

    labeled, n = ndimage.label(mask)
    if n == 0:
        return np.zeros((0, mask.ndim), dtype=int)
    return np.array([[s.stop - s.start for s in sl] for sl in ndimage.find_objects(labeled)])


class SemanticMasksFingerprint:
    """
    Semantic segmentation masks (one channel, a label per class). Masks in 0/255 are taken as 0/1, as BiaPy
    does.
    """

    def __init__(self, n_files, is_3d):
        self.n_files = n_files
        self.analyze = set(sample_indices(n_files))
        self.shapes = []
        self.counts, self.present, self.extents = {}, {}, {}
        self.fg_fractions = []
        self.total = 0
        self.n_analyzed = 0
        self.error = None

    def add(self, index, mask):
        self.shapes.append(tuple(mask.shape[:-1]))
        if index not in self.analyze or self.error:
            return
        m = mask[..., 0]
        labels, n = np.unique(m, return_counts=True)
        if len(set(self.counts) | set(labels.tolist())) > MAX_LABELS:
            self.error = "More than {} different values found".format(MAX_LABELS)
            return
        self.n_analyzed += 1
        self.total += m.size
        self.fg_fractions.append(float(n[labels != 0].sum()) / m.size)
        for lab, cnt in zip(labels.tolist(), n.tolist()):
            self.counts[lab] = self.counts.get(lab, 0) + cnt
            self.present[lab] = self.present.get(lab, 0) + 1
            if lab != 0:
                self.extents.setdefault(lab, []).append(_object_extents(m == lab))

    def result(self):
        fp = {"kind": "semantic_masks", "n_files": self.n_files, "n_analyzed": self.n_analyzed}
        fp.update(_shape_stats(self.shapes))
        if self.error:
            fp["error"] = self.error
            return fp
        counts, present, extents = dict(self.counts), dict(self.present), dict(self.extents)
        binary_255 = set(counts) <= {0, 255} and 255 in counts
        if binary_255:
            for d in (counts, present, extents):
                if 255 in d:
                    d[1] = d.pop(255)
        fp["binary_255"] = bool(binary_255)
        fp["labels"] = sorted(counts)
        fp["class_voxel_fraction"] = {str(k): counts[k] / self.total for k in sorted(counts)}
        fp["class_image_fraction"] = {str(k): present[k] / self.n_analyzed for k in sorted(present)}
        fg = np.array(self.fg_fractions)
        fp["foreground_fraction"] = {
            "mean": float(fg.mean()), "median": float(np.median(fg)), "min": float(fg.min()), "max": float(fg.max())
        }
        fp["objects"] = {}
        for lab, ext in extents.items():
            ext = np.concatenate(ext)
            if len(ext):
                fp["objects"][str(lab)] = {"count": int(len(ext)), **_percentiles(ext)}
        return fp


def _touching_labels(lab):
    """Labels of the instances that touch another instance (face connectivity)."""
    touching = set()
    for ax in range(lab.ndim):
        a = np.moveaxis(lab, ax, 0)
        x, y = a[:-1], a[1:]
        m = (x != y) & (x > 0) & (y > 0)
        touching.update(np.unique(x[m]).tolist())
        touching.update(np.unique(y[m]).tolist())
    return touching


def _downsample(crop):
    """Crop strided to at most MAX_CROP_VOXELS, and the stride."""
    s = int(np.ceil((crop.size / MAX_CROP_VOXELS[crop.ndim]) ** (1 / crop.ndim)))
    if s <= 1:
        return crop, 1
    return np.pad(crop[tuple(slice(None, None, s) for _ in range(crop.ndim))], 1), s


def instance_shape(crop, rng):
    """
    Shape descriptors of one instance (boolean crop with a 1-voxel margin).

    * ``inscribed_radius``: radius of the largest inscribed circle/sphere (thickness).
    * ``elongation``: largest over middle principal axis (in 3D the middle one, so that objects flattened by
      the Z anisotropy are not taken as elongated).
    * ``solidity``: volume over the volume of its convex hull.
    * ``star_convex``: whether the straight path from its innermost point (max. of the distance transform,
      as StarDist) to sampled pixels stays inside.
    * ``centroid_inside``: whether its centroid falls inside it (central point representations).
    * ``endpoints``: endpoints of its skeleton, only for elongated or thin instances (else None).
    """
    from scipy import ndimage
    from skimage.measure import regionprops
    from skimage.morphology import skeletonize

    crop, stride = _downsample(crop)
    if not crop.any():
        return None
    props = regionprops(crop.astype(np.uint8))[0]
    edt = ndimage.distance_transform_edt(crop)
    r_in = float(edt.max()) * stride
    center = np.unravel_index(np.argmax(edt), crop.shape)

    coords = np.argwhere(crop)
    if len(coords) >= crop.ndim + 1:
        ev = np.sort(np.linalg.eigvalsh(np.atleast_2d(np.cov((coords - coords.mean(0)).T))))[::-1]
        ev = np.sqrt(np.clip(ev, 1e-6, None))
        elongation = float(ev[0] / ev[1])
        major = 4 * ev[0] * stride  # full length of the main axis (as skimage's axis_major_length)
    else:
        elongation, major = 1.0, 1.0

    try:
        solidity = float(props.solidity)
    except Exception:  # degenerate hull (flat or tiny instances)
        solidity = 1.0

    pick = coords[rng.choice(len(coords), size=min(STAR_SAMPLES, len(coords)), replace=False)]
    visible = 0
    for p in pick:
        n = int(np.ceil(np.abs(p - center).max())) + 1
        line = np.rint(np.linspace(center, p, n)).astype(int)
        visible += bool(crop[tuple(line.T)].all())
    star_convex = visible >= STAR_VISIBLE_FRACTION * len(pick)

    centroid_inside = bool(crop[tuple(int(round(c)) for c in props.centroid)])

    endpoints = None
    thinness = 2 * r_in / major if major > 0 else 1.0
    if elongation >= ELONGATED or thinness < THIN:
        skel = skeletonize(crop).astype(bool)
        neighbors = ndimage.convolve(skel.astype(np.uint8), np.ones((3,) * crop.ndim, np.uint8), mode="constant")
        endpoints = int(((neighbors == 2) & skel).sum())
    return {
        "inscribed_radius": r_in,
        "elongation": elongation,
        "thinness": thinness,
        "solidity": solidity,
        "star_convex": bool(star_convex),
        "centroid_inside": centroid_inside,
        "endpoints": endpoints,
    }


class InstanceMasksFingerprint:
    """
    Instance segmentation masks (instance IDs in the first channel and, optionally, their classes in the
    second): number, size and shape of the instances, and how often they touch.
    """

    def __init__(self, n_files, is_3d, seed=0):
        self.n_files = n_files
        self.analyze = set(sample_indices(n_files))
        self.per_image = max(1, MAX_INSTANCES // len(self.analyze)) if self.analyze else 1
        self.rng = np.random.default_rng(seed)
        self.start = time.time()
        self.shapes, self.counts, self.fg_fractions, self.extents, self.touching = [], [], [], [], []
        self.shape_stats = []
        self.n_analyzed = 0

    def add(self, index, mask):
        from scipy import ndimage

        self.shapes.append(tuple(mask.shape[:-1]))
        if index not in self.analyze:
            return
        lab = mask[..., 0]
        if not np.issubdtype(lab.dtype, np.integer):  # e.g. float32 label images
            lab = lab.astype(np.int64)
        self.n_analyzed += 1
        ids = np.unique(lab)
        ids = ids[ids != 0]
        self.counts.append(len(ids))
        self.fg_fractions.append(float((lab != 0).sum()) / lab.size)
        if len(ids) == 0:
            return
        objs = ndimage.find_objects(lab)
        touch = _touching_labels(lab)
        for inst in ids:
            sl = objs[int(inst) - 1] if int(inst) - 1 < len(objs) else None
            if sl is not None:
                self.extents.append([s.stop - s.start for s in sl])
        self.touching.append(sum(1 for inst in ids if int(inst) in touch) / len(ids))
        for inst in self.rng.choice(ids, size=min(self.per_image, len(ids)), replace=False):
            if time.time() - self.start > SHAPE_TIME_BUDGET:
                break
            sl = objs[int(inst) - 1]
            if sl is None:
                continue
            st = instance_shape(np.pad(lab[sl] == inst, 1), self.rng)
            if st is not None:
                self.shape_stats.append(st)

    def result(self):
        fp = {"kind": "instance_masks", "n_files": self.n_files, "n_analyzed": self.n_analyzed}
        fp.update(_shape_stats(self.shapes))
        fp["instances_per_image"] = {"median": float(np.median(self.counts)), "max": int(max(self.counts))}
        fg = np.array(self.fg_fractions)
        fp["foreground_fraction"] = {
            "mean": float(fg.mean()), "median": float(np.median(fg)), "min": float(fg.min()), "max": float(fg.max())
        }
        if not self.extents:
            fp["n_instances"] = 0
            return fp
        ext = np.array(self.extents)
        fp["n_instances"] = int(len(ext))
        fp["objects"] = {"instances": {"count": int(len(ext)), **_percentiles(ext)}}
        fp["touching_fraction"] = float(np.mean(self.touching))
        st = self.shape_stats
        n = len(st)
        if n == 0:
            return fp
        fp["morphology"] = {
            "n_measured": n,
            "inscribed_radius_median": float(np.median([s["inscribed_radius"] for s in st])),
            "solidity_median": float(np.median([s["solidity"] for s in st])),
            "elongation_median": float(np.median([s["elongation"] for s in st])),
            "elongated_fraction": sum(s["elongation"] >= ELONGATED for s in st) / n,
            "thin_fraction": sum(s["thinness"] < THIN for s in st) / n,
            "star_convex_fraction": sum(s["star_convex"] for s in st) / n,
            "centroid_inside_fraction": sum(s["centroid_inside"] for s in st) / n,
            # Skeletons with more than two endpoints branch
            "branched_fraction": sum(1 for s in st if s["endpoints"] is not None and s["endpoints"] > 2) / n,
        }
        return fp


class PointsFingerprint:
    """Detection points: points per image and the displacement from each point to its nearest one (voxels)."""

    def __init__(self, n_files, seed=0):
        self.n_files = n_files
        self.rng = np.random.default_rng(seed)
        self.counts, self.vectors, self.classes = [], [], set()

    def add(self, points, classes=None):
        """``points``: ``(n, 2|3)`` coordinates (axis-0, axis-1[, axis-2]); ``classes``: their classes, if any."""
        from scipy.spatial import cKDTree

        self.counts.append(len(points))
        if classes is not None:
            self.classes.update(int(c) for c in classes)
        if len(points) >= 2:
            _, nn = cKDTree(points).query(points, k=2)
            self.vectors.append(points[nn[:, 1]] - points)

    def result(self):
        fp = {
            "kind": "detection_points",
            "n_files": self.n_files,
            "points_per_image": {
                "median": float(np.median(self.counts)), "max": int(max(self.counts)), "total": int(sum(self.counts))
            },
            "classes": sorted(self.classes),
        }
        if self.vectors:
            v = np.concatenate(self.vectors)
            if len(v) > MAX_NN_VECTORS:
                v = v[self.rng.choice(len(v), MAX_NN_VECTORS, replace=False)]
            fp["nn_vectors"] = np.abs(v).round(2).tolist()
        return fp
