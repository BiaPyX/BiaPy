"""
Check of a data folder selected in the wizard: its images are read with BiaPy's readers, in BiaPy's order, to verify
that BiaPy can use them, to deduce some variables (channels, classes...) and to build their fingerprint
(:mod:`biapy.wizard.fingerprint`).
"""
import math
import os

import numpy as np

from biapy.wizard import fingerprint as fpr

# Labels above this are not taken as class numbers
MAX_CLASSES = 256
# Zarr/H5 data larger than this is checked on a random part of it, chosen by the user (see folder_size())
LARGE_DATA_BYTES = 2 * 1024**3
SUGGESTED_CHECK_BYTES = 1024**3
# Blocks read from large Zarr/H5 images: axes longer than BLOCK_MIN_SIDE are cut in proportion, so the block keeps
# the order of axis sizes BiaPy guesses the axes from (when they are not stored, as in OME-Zarr)
BLOCK_VOXELS = 4 * 1024**2
BLOCK_MIN_SIDE = 64
PREVIEW_VOXELS = 16 * 1024**2
# data_range() values from the narrowest: an image read in blocks spans the widest of them
RANGES = ("01 range", "uint8 range", "uint16 range", "none_range")


def _result(error, message="", constraints=None, sample_info=None):
    return {
        "error": bool(error),
        "error_message": str(message),
        "data_constraints": constraints or {},
        "sample_info": sample_info or {},
    }


def folder_images(folder):
    """Images of a folder as BiaPy lists them, or none."""
    from biapy.data.data_manipulation import list_data_files

    try:
        return list_data_files(folder)[0]
    except ValueError:
        return []


def _open_chunked(path, data_path=""):
    """Zarr/H5 data, not read yet, and its OME-Zarr axes (None when BiaPy guesses them from the shape)."""
    from biapy.data.data_manipulation import load_img_data

    data, _, meta = load_img_data(path, data_within_zarr_path=data_path or None, load_meta=True)
    return data, meta["ome_zarr"].axes if meta else None, meta


def _block_shape(shape, voxels):
    """Block of about ``voxels`` voxels: the axes longer than BLOCK_MIN_SIDE cut in proportion to their length."""
    long_axes = [i for i, v in enumerate(shape) if v > BLOCK_MIN_SIDE]
    if not long_axes:
        return tuple(shape)
    r = min(1.0, (voxels / float(np.prod(shape))) ** (1.0 / len(long_axes)))
    return tuple(max(BLOCK_MIN_SIDE, int(math.ceil(v * r))) if i in long_axes else v for i, v in enumerate(shape))


def _blocks(shape, fraction, rng):
    """Random blocks (slices) covering about ``fraction`` of an image, or the whole image if that is not less."""
    block = _block_shape(shape, BLOCK_VOXELS)
    count = int(math.ceil(fraction * np.prod(shape) / np.prod(block)))
    if fraction >= 1 or count * np.prod(block) >= np.prod(shape):
        return [tuple(slice(None) for _ in shape)]
    starts = [rng.integers(0, v - b + 1, count) for v, b in zip(shape, block)]
    return [tuple(slice(int(st[k]), int(st[k]) + b) for st, b in zip(starts, block)) for k in range(count)]


def _biapy_shape(data, axes, is_3d, path):
    """Shape of Zarr/H5 data as BiaPy reads it, without reading it."""
    from biapy.data.data_manipulation import to_biapy_shape

    return tuple(to_biapy_shape(np.broadcast_to(np.zeros((), np.uint8), data.shape), is_3d, path, ome_axes=axes).shape)


def image_parts(path, is_3d, fraction=1.0, data_path="", rng=None):
    """
    Image as BiaPy reads it or, for a Zarr/H5 image and ``fraction`` < 1, random blocks covering that fraction of it.

    Parameters
    ----------
    path : str
        Path to the image.
    is_3d : bool
        Whether the image is 3D.
    fraction : float, optional
        Fraction of a Zarr/H5 image to read.
    data_path : str, optional
        Path of the image within a Zarr/H5 file, e.g. "labels.mito" for a label image of an OME-Zarr image.
    rng : numpy.random.Generator, optional
        Generator to choose the blocks with.

    Returns
    -------
    parts : list of NDArray
        Image or blocks, ``(z, )y, x, c``.
    shape : tuple of int
        Shape of the whole image, ``(z, )y, x, c``.
    meta : dict or None
        Metadata of the image (see read_img_as_ndarray).
    """
    from biapy.data.data_3D_manipulation import is_chunked
    from biapy.data.data_manipulation import read_img_as_ndarray, to_biapy_shape

    if not is_chunked(path):
        img, meta = read_img_as_ndarray(path, is_3d=is_3d, load_meta=True)
        return [img], tuple(img.shape), meta
    data, axes, meta = _open_chunked(path, data_path)
    if not is_3d and axes is not None and "Z" in axes and data.shape[axes.index("Z")] > 1:
        raise ValueError(
            f"The image is 3D (axes {axes}, shape {tuple(data.shape)}), but the images were said to be 2D: answer "
            "'Yes' to the question 'Are your images in 3D?'"
        )
    rng = rng or np.random.default_rng(0)
    parts = [to_biapy_shape(np.asarray(data[sl]), is_3d, path, ome_axes=axes) for sl in _blocks(data.shape, fraction, rng)]
    return parts, _biapy_shape(data, axes, is_3d, path), meta


def preview_image(path, is_3d, data_path=""):
    """Image to preview as BiaPy reads it (a central block of a large Zarr/H5 image) and the whole image's shape."""
    from biapy.data.data_3D_manipulation import is_chunked
    from biapy.data.data_manipulation import to_biapy_shape

    if not is_chunked(path):
        img = image_parts(path, is_3d)[0][0]
        return img, img.shape
    data, axes, _ = _open_chunked(path, data_path)
    block = _block_shape(data.shape, PREVIEW_VOXELS)
    center = tuple(slice((v - b) // 2, (v - b) // 2 + b) for v, b in zip(data.shape, block))
    return to_biapy_shape(np.asarray(data[center]), is_3d, path, ome_axes=axes), _biapy_shape(data, axes, is_3d, path)


def folder_size(folder):
    """
    Size of the Zarr/H5 images of a folder, to decide how much of them to check.

    Returns
    -------
    size : dict
        ``files`` and ``chunked_files`` (Zarr/H5), ``chunked_bytes`` (their size uncompressed), ``large`` (whether
        only a part of them should be checked) and ``suggested_percentage`` (about SUGGESTED_CHECK_BYTES of them).
    """
    from biapy.data.data_3D_manipulation import is_chunked

    files = folder_images(folder)
    chunked, total = 0, 0
    for f in files:
        path = os.path.join(folder, f)
        if not is_chunked(path):
            continue
        try:
            data = _open_chunked(path)[0]
        except Exception:  # the check will tell what is wrong
            continue
        chunked += 1
        total += int(np.prod(data.shape)) * np.dtype(data.dtype).itemsize
    percentage = 100 if total <= SUGGESTED_CHECK_BYTES else max(1, int(100 * SUGGESTED_CHECK_BYTES / total))
    return {
        "files": len(files),
        "chunked_files": chunked,
        "chunked_bytes": total,
        "large": total > LARGE_DATA_BYTES,
        "suggested_percentage": percentage,
    }


def ome_zarr_labels(folder):
    """Label images (e.g. the ground truth) that all the images of a folder have, if they are OME-Zarr images."""
    from biapy.data.ome_zarr import image_info

    common = None
    for f in folder_images(folder):
        info = image_info(os.path.join(folder, f))
        if info is None:
            return []
        common = [x for x in info.labels if common is None or x in common]
    return common or []


def check_folder(folder, key, workflow, ndim, progress=None, percentage=100, data_path="", tracking=False,
                 train_channels=None):
    """
    Checks the data of a folder selected in the wizard.

    Parameters
    ----------
    folder : str
        Folder to check.
    key : str
        Variable the folder is for, e.g. "DATA.TRAIN.PATH" or "DATA.TRAIN.GT_PATH" (targets).
    workflow : str
        ``PROBLEM.TYPE``, e.g. "SEMANTIC_SEG".
    ndim : str
        "2D" or "3D".
    progress : callable, optional
        Called as ``progress(done, total, name)`` after each file is checked (``name`` is the file's), and as
        ``progress(total, total, None)`` when all are read and the data is analyzed (it may take a while).
    percentage : float, optional
        Percentage of each Zarr/H5 image to check, in random blocks (see :func:`folder_size`). The rest of the
        images are always read whole.
    data_path : str, optional
        Path of the targets within the Zarr files of ``folder``, e.g. "labels.mito" to use the label images of
        OME-Zarr images as targets (see :func:`ome_zarr_labels`).
    tracking : bool, optional
        Whether the objects are tracked (``PROBLEM.TRACKING.ENABLE``): the test raw images are then the frames of one
        time-lapse, which must be single-channel images of the same size, one file per frame.
    train_channels : int, optional
        Channels of the training images, when checking other raw images (e.g. the test ones): the model takes that
        number of channels, so they must have it too. None: not compared.

    Returns
    -------
    result : dict
        * ``error`` and ``error_message``: whether BiaPy can't use the data, and why.
        * ``data_constraints``: variables deduced from the data, e.g. ``{"DATA.PATCH_SIZE_C": 1,
          "DATA.N_CLASSES": 3}``, and, under ``key``, ``key + "_path"`` and ``key + "_path_shapes"``, the
          number of files, the folder and the shape of each image.
        * ``sample_info``: ``dir_name`` (``key``), the folder's ``fingerprint`` and, for raw images, the label
          images of the OME-Zarr images (``ome_zarr_labels``).
    """
    is_3d = ndim == "3D"
    is_gt = "GT_PATH" in key
    print(f"Checking data in {folder}", flush=True)
    if not os.path.isdir(folder):
        return _result(True, f"The folder does not exist:\n{folder}")
    progress = progress or (lambda done, total, name: None)
    if workflow == "CLASSIFICATION" and not is_gt:
        return same_channels(_check_classification(folder, key, is_3d, progress), train_channels)
    if workflow == "DETECTION" and is_gt:
        return _check_points(folder, key, is_3d, progress)
    kind = None
    if is_gt:
        kind = {"SEMANTIC_SEG": "semantic", "INSTANCE_SEG": "instance"}.get(workflow, "image")
    frames = tracking and key == "DATA.TEST.PATH"
    if frames:
        from biapy.data.data_manipulation import list_data_files
        from biapy.data.tracks_io import frames_problem

        try:
            names, chunked = list_data_files(folder)
        except ValueError as e:
            return _result(True, e)
        if chunked:
            return _result(True, f"Tracking reads one time-lapse, one image file (e.g. TIFF) per frame, but this folder "
                           f"has Zarr/H5 images or subfolders. If each subfolder is a time-lapse, select one:\n{folder}")
        problem = frames_problem(names)
        if problem:
            return _result(True, f"{problem}\n{folder}")
    res = _check_images(folder, key, workflow, is_3d, kind, progress, percentage / 100.0, data_path)
    if res["error"]:
        return res
    if frames:
        problem = frames_problem(names, res["data_constraints"][key + "_path_shapes"])
        if problem:
            return _result(True, f"{problem}\n{folder}")
    if is_gt:
        # Targets within the raw images' Zarr files
        phase = key.split(".")[1]
        res["data_constraints"].update({
            "DATA.{}.INPUT_ZARR_MULTIPLE_DATA".format(phase): bool(data_path),
            "DATA.{}.INPUT_ZARR_MULTIPLE_DATA_RAW_PATH".format(phase): "",
            "DATA.{}.INPUT_ZARR_MULTIPLE_DATA_GT_PATH".format(phase): data_path,
        })
    else:
        res["sample_info"]["ome_zarr_labels"] = ome_zarr_labels(folder)
    return same_channels(res, train_channels)


def same_channels(res, train_channels):
    """
    The result of a data check (see :func:`check_folder`), as an error if its images don't have ``train_channels``
    channels, those of the training images (None: not compared).
    """
    channels = res["data_constraints"].get("DATA.PATCH_SIZE_C")
    if res["error"] or train_channels is None or channels is None or channels == train_channels:
        return res
    return _result(
        True,
        f"These images have {channels} channel(s), but the training images have {train_channels}: the model takes "
        f"{train_channels} channel(s), so these images must have the same number (e.g. 1 for grayscale, 3 for RGB). "
        f"Convert them and check them again.",
    )


def _check_images(folder, key, workflow, is_3d, kind, progress, fraction=1.0, data_path=""):
    """
    ``kind``: None for raw images, "semantic"/"instance" masks or "image" targets. ``fraction``: part of each
    Zarr/H5 image read, in random blocks that are analyzed as separate images. ``data_path``: path of the images
    within the Zarr/H5 files.
    """
    from biapy.data.data_3D_manipulation import is_chunked
    from biapy.data.data_manipulation import check_image_consistency, data_range

    files = folder_images(folder)
    if not files:
        return _result(True, f"No images found in folder:\n{folder}")
    if data_path:
        missing = [f for f in files if not is_chunked(f)]
        if missing:
            return _result(True, f"'{data_path}' can only be read from Zarr/H5 files, but this is not one:\n"
                           f"{os.path.join(folder, missing[0])}")
    n = len(files)
    rng = np.random.default_rng(0)
    n_parts = 0  # blocks are added to the fingerprint as separate images
    for f in files:
        path = os.path.join(folder, f)
        try:
            n_parts += len(_blocks(_open_chunked(path, data_path)[0].shape, fraction, rng)) if is_chunked(path) else 1
        except Exception as e:
            return _result(True, f"Couldn't load image:\n{path}\n{e}")
    rng = np.random.default_rng(0)
    if kind == "semantic":
        acc = fpr.SemanticMasksFingerprint(n_parts, is_3d)
    elif kind == "instance":
        acc = fpr.InstanceMasksFingerprint(n_parts, is_3d)
    elif kind is None and workflow != "CLASSIFICATION":
        acc = fpr.ImagesFingerprint(n_parts, is_3d)
    else:
        acc = None

    expected, shapes, labels, inst_classes, part = None, [], set(), 0, 0
    for i, f in enumerate(files):
        path = os.path.join(folder, f)
        try:
            parts, shape, meta = image_parts(path, is_3d, fraction, data_path, rng)
            drange = None
            if kind in (None, "image"):
                drange = (RANGES[max(RANGES.index(data_range(img)) for img in parts)], parts[0].dtype)
            # The wizard's normalizations don't depend on the data range
            expected = check_image_consistency(path, shape[-1], expected, drange=drange, strict=False)
        except Exception as e:
            return _result(True, f"Couldn't load image:\n{path}\n{e}")
        channels = shape[-1]
        shapes.append(shape)

        if kind == "semantic":
            if channels != 1:
                return _result(True, f"Semantic masks are expected to have just one channel. Image analized:\n{path}")
            for img in parts:
                if len(labels) <= fpr.MAX_LABELS:
                    labels.update(np.unique(img).tolist())
        elif kind == "instance":
            if channels not in (1, 2):
                return _result(
                    True,
                    "Instance masks are expected to have one or two channels. In case two channels are provided the "
                    f"first one must have the instance IDs and one their corresponding semantic (class) labels. Image "
                    f"analized:\n{path}",
                )
            if channels == 2:
                inst_classes = max([inst_classes] + [len(np.unique(img[..., 1])) for img in parts])

        for j, img in enumerate(parts):
            if acc is not None:
                if kind is None:
                    acc.add(part, img, meta)
                else:
                    acc.add(part, img)
            part += 1
            progress(i + (j + 1 == len(parts)), n, f if len(parts) == 1 else f"{f} (part {j + 1} of {len(parts)})")

    constraints = {key: n, key + "_path": folder, key + "_path_shapes": shapes}
    if kind is None:
        constraints["DATA.PATCH_SIZE_C"] = channels
    if kind == "semantic":
        constraints["DATA.N_CLASSES"] = semantic_classes(labels)
    elif kind == "instance":
        constraints["DATA.N_CLASSES"] = max(2, inst_classes)

    sample_info = {"dir_name": key}
    if acc is not None:
        progress(n, n, None)
        try:
            fp = acc.result()
            # The size of the images, not of the blocks read of them
            fp.update(fpr.shape_stats([sh[:-1] for sh in shapes]))
            if kind == "instance":
                from biapy.wizard.instance_seg import describe_morphology, propose_representations

                fp["representations"] = propose_representations(fp, 3 if is_3d else 2)
                fp["morphology_summary"] = describe_morphology(fp)
            sample_info["fingerprint"] = fp
        except Exception as e:  # the analysis is not essential: the data can be used anyway
            print(f"Could not analyze the data in detail ({type(e).__name__}: {e})", flush=True)
    return _result(False, "", constraints, sample_info)


def semantic_classes(labels):
    """Classes of semantic masks from the labels found in all of them (0/255 masks are binary, as for BiaPy)."""
    labels = set(labels)
    if labels <= {0, 255}:
        return 2
    if all(float(v).is_integer() for v in labels) and max(labels) < MAX_CLASSES:
        return max(2, int(max(labels)) + 1)
    return max(2, len(labels))


def _check_points(folder, key, is_3d, progress):
    """CSV files with the points to detect (columns axis-0, axis-1[, axis-2] and optionally class)."""
    from biapy.data.data_manipulation import read_points_csv

    files = folder_images(folder)
    if not files:
        return _result(True, f"No CSV files found in folder:\n{folder}")
    acc = fpr.PointsFingerprint(len(files))
    cols = ["axis-0", "axis-1", "axis-2"] if is_3d else ["axis-0", "axis-1"]
    with_class = False
    for i, f in enumerate(files):
        path = os.path.join(folder, f)
        try:
            df = read_points_csv(path, is_3d=is_3d)
        except Exception as e:
            return _result(True, f"Couldn't load CSV file:\n{path}\n{e}")
        classes = df["class"].astype(int).to_numpy() if "class" in df.columns else None
        with_class |= classes is not None
        acc.add(df[cols].to_numpy(dtype=float), classes)
        progress(i + 1, len(files), f)
    constraints = {key: len(files), key + "_path": folder}
    fp = acc.result()
    if with_class:
        constraints["DATA.N_CLASSES"] = len(fp["classes"])
    return _result(False, "", constraints, {"dir_name": key, "fingerprint": fp})


def _check_classification(folder, key, is_3d, progress):
    """One subfolder per class, with its images."""
    from biapy.data.data_manipulation import check_image_consistency, data_range, read_img_as_ndarray
    from biapy.utils.misc import os_walk_clean

    try:
        class_names = next(os_walk_clean(folder))[1]
    except StopIteration:
        class_names = []
    if not class_names:
        return _result(True, f"There is no folder/class in folder:\n{folder}")
    class_files = {c: next(os_walk_clean(os.path.join(folder, c)))[2] for c in class_names}
    for c, files in class_files.items():
        if not files:
            return _result(True, f"There are no images in class folder:\n{os.path.join(folder, c)}")
    total = sum(len(files) for files in class_files.values())
    expected, done = None, 0
    for c, files in class_files.items():
        for f in files:
            path = os.path.join(folder, c, f)
            try:
                img = read_img_as_ndarray(path, is_3d=is_3d)
                expected = check_image_consistency(
                    path, img.shape[-1], expected, drange=(data_range(img), img.dtype), strict=False
                )
            except Exception as e:
                return _result(True, f"Couldn't load image:\n{path}\n{e}")
            done += 1
            progress(done, total, os.path.join(c, f))
    constraints = {"DATA.PATCH_SIZE_C": expected[0], "DATA.N_CLASSES": len(class_names), key: total, key + "_path": folder}
    return _result(False, "", constraints, {"dir_name": key})
