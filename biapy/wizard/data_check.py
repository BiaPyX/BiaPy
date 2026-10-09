"""
Check of a data folder selected in the wizard: the files are read with BiaPy's readers, in the order BiaPy lists
(and pairs) them, to verify that BiaPy can use them, to deduce some variables (channels, classes...) and to
build the folder's fingerprint (:mod:`biapy.wizard.fingerprint`) in the same pass.
"""
import os

import numpy as np

from biapy.wizard import fingerprint as fpr

# Labels above this are not taken as class numbers
MAX_CLASSES = 256


def _result(error, message="", constraints=None, sample_info=None):
    return {
        "error": bool(error),
        "error_message": str(message),
        "data_constraints": constraints or {},
        "sample_info": sample_info or {},
    }


def list_files(folder):
    from biapy.utils.misc import os_walk_clean

    try:
        return next(os_walk_clean(folder))[2]
    except StopIteration:
        return []


def _channels_message(expected, found, path):
    return (
        "All images need to have the same number of channels and represent same information to ensure the deep "
        f"learning model can be trained correctly. However, the following image (with {found} channels) appears "
        f"to have a different number of channels than the first image (with {expected} channels) in the folder:\n{path}"
    )


def check_folder(folder, key, workflow, ndim, progress=None):
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

    Returns
    -------
    result : dict
        * ``error`` and ``error_message``: whether BiaPy can't use the data, and why.
        * ``data_constraints``: variables deduced from the data, e.g. ``{"DATA.PATCH_SIZE_C": 1,
          "DATA.N_CLASSES": 3}``, and, under ``key``, ``key + "_path"`` and ``key + "_path_shapes"``, the
          number of files, the folder and the shape of each image.
        * ``sample_info``: ``dir_name`` (``key``) and the folder's ``fingerprint``.
    """
    is_3d = ndim == "3D"
    is_gt = "GT_PATH" in key
    print(f"Checking data in {folder}", flush=True)
    if not os.path.isdir(folder):
        return _result(True, f"The folder does not exist:\n{folder}")
    progress = progress or (lambda done, total, name: None)
    if workflow == "CLASSIFICATION" and not is_gt:
        return _check_classification(folder, key, is_3d, progress)
    if workflow == "DETECTION" and is_gt:
        return _check_points(folder, key, is_3d, progress)
    kind = None
    if is_gt:
        kind = {"SEMANTIC_SEG": "semantic", "INSTANCE_SEG": "instance"}.get(workflow, "image")
    return _check_images(folder, key, workflow, is_3d, kind, progress)


def _check_images(folder, key, workflow, is_3d, kind, progress):
    """``kind``: None for raw images, "semantic"/"instance" masks or "image" targets."""
    from biapy.data.data_manipulation import data_range, read_img_as_ndarray

    files = list_files(folder)
    if not files:
        return _result(True, f"No images found in folder:\n{folder}")
    n = len(files)
    if kind == "semantic":
        acc = fpr.SemanticMasksFingerprint(n, is_3d)
    elif kind == "instance":
        acc = fpr.InstanceMasksFingerprint(n, is_3d)
    elif kind is None and workflow != "CLASSIFICATION":
        acc = fpr.ImagesFingerprint(n, is_3d)
    else:
        acc = None

    channels, drange, shapes, labels, inst_classes = None, None, [], set(), 0
    for i, f in enumerate(files):
        path = os.path.join(folder, f)
        try:
            img, meta = read_img_as_ndarray(path, is_3d=is_3d, load_meta=True)
        except Exception as e:
            return _result(True, f"Couldn't load image:\n{path}\n{e}")
        if channels is None:
            channels = img.shape[-1]
        if img.shape[-1] != channels:
            return _result(True, _channels_message(channels, img.shape[-1], path))
        shapes.append(tuple(img.shape))

        if kind in (None, "image"):
            r = data_range(img)
            if drange is None:
                drange = r
            if r != drange:
                return _result(
                    True,
                    f"All images must be within the same data range. However, the following image (with a range of "
                    f"{r}) appears to be in a different data range than the first image (with a range of {drange}) in "
                    f"the folder:\n{path}",
                )
        elif kind == "semantic":
            if channels != 1:
                return _result(True, f"Semantic masks are expected to have just one channel. Image analized:\n{path}")
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
                inst_classes = max(inst_classes, len(np.unique(img[..., 1])))

        if acc is not None:
            if kind is None:
                acc.add(i, img, meta)
            else:
                acc.add(i, img)
        progress(i + 1, n, f)

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

    files = list_files(folder)
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
    from biapy.data.data_manipulation import data_range, read_img_as_ndarray
    from biapy.utils.misc import os_walk_clean

    try:
        class_names = next(os_walk_clean(folder))[1]
    except StopIteration:
        class_names = []
    if not class_names:
        return _result(True, f"There is no folder/class in folder:\n{folder}")
    class_files = {c: list_files(os.path.join(folder, c)) for c in class_names}
    for c, files in class_files.items():
        if not files:
            return _result(True, f"There are no images in class folder:\n{os.path.join(folder, c)}")
    total = sum(len(files) for files in class_files.values())
    channels, drange, done = None, None, 0
    for c, files in class_files.items():
        for f in files:
            path = os.path.join(folder, c, f)
            try:
                img = read_img_as_ndarray(path, is_3d=is_3d)
            except Exception as e:
                return _result(True, f"Couldn't load image:\n{path}\n{e}")
            if channels is None:
                channels = img.shape[-1]
            if img.shape[-1] != channels:
                return _result(True, _channels_message(channels, img.shape[-1], path))
            r = data_range(img)
            if drange is None:
                drange = r
            if r != drange:
                return _result(
                    True,
                    f"All images must be within the same data range. However, the following image (with a range of "
                    f"{r}) appears to be in a different data range than the first image (with a range of {drange}) in "
                    f"the folder:\n{path}",
                )
            done += 1
            progress(done, total, os.path.join(c, f))
    constraints = {"DATA.PATCH_SIZE_C": channels, "DATA.N_CLASSES": len(class_names), key: total, key + "_path": folder}
    return _result(False, "", constraints, {"dir_name": key})
