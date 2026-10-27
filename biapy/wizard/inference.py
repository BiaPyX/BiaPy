"""
Inference that fits in the computer's memory (RAM). From the shape of the largest test image, the peak memory of
BiaPy's test phase is estimated for its ways of predicting an image, and the first one that fits is used:

* whole images (BiaPy's default): the image, all its patches and their predictions (float32), merged into the full
  prediction, then post-processed (e.g. the watershed of instance segmentation) in memory.
* TEST.REDUCE_MEMORY: patches cropped and merged separately and float16 predictions. Less memory, but slower.
* TEST.BY_CHUNKS (3D semantic segmentation, instance segmentation and detection): the patches are predicted and
  written to a Zarr file one by one and post-processed tile by tile, so memory hardly grows with the image. Only the
  image itself is held in memory when it is not an H5/Zarr file (those are read by chunks too).

Whole (and with REDUCE_MEMORY), BiaPy holds the image, all its patches and their predictions, so the memory grows with
the voxels of the image, its channels, the channels the model predicts and how much the patches overlap: ``f``, the
voxels of all the patches over the voxels of the image (with DATA.TEST.PADDING of 1/6 of the patch, as the Wizard
sets, ``f`` is about 3.3-5.6; without padding about 1.3). Measured (peak PSS of BiaPy and its workers) with 3D
semantic (1 output channel) and instance (F, C) models on volumes of 124 to 495 MiB, 1 and 3 input channels, with and
without padding, batch size 1, the peak memory is, within 5% of every measurement::

    whole:          1.3 GiB + voxels * (4 * f * (inputs + outputs) + 11 * outputs + 7 * inputs) bytes
    REDUCE_MEMORY:  1.6 GiB + voxels * (1.5 * f * (inputs + outputs) + 12 * outputs + 8 * inputs) bytes

(the first term: the patches and their predictions, float32 or float16; the others: the image and the full
prediction). By chunks it took 1.6-2.4 GiB, plus the image when it is read whole.

E.g. a 145x1364x1364 volume with 3 channels and an 18x160x160 patch (f = 3.75) needs about 33 GB to segment its
instances whole and 22 GB with REDUCE_MEMORY, while 1-channel estimates said 13 GB (a 32 GB computer ran out of
memory).
"""
import os

import numpy as np

from . import common

GIB = 1024**3
# The models above: fixed bytes, and bytes per voxel per f * (inputs + outputs), per output and per input channel
MEMORY_MODEL = {
    "whole": (1.3 * GIB, 4.0, 11.0, 7.0),
    "reduce": (1.6 * GIB, 1.5, 12.0, 8.0),
}
# By chunks: BiaPy, the tiles being filled and post-processed, and the image when it is read whole (twice its size)
CHUNKS_BASE_BYTES = 1.5 * GIB
# Writing the result of the chunks as a TIF too (TEST.BY_CHUNKS.SAVE_OUT_TIF) reads it whole
SAVE_TIF_BYTES_PER_VOXEL = 16
# Share of the RAM BiaPy may take: Fiji, the system and other programs need the rest
RAM_FRACTION = 0.6
CHUNK_WORKFLOWS = ("SEMANTIC_SEG", "INSTANCE_SEG", "DETECTION")
CHUNKED_FORMATS = (".h5", ".hdf5", ".hdf", ".zarr", ".n5")


def host_ram_bytes():
    """Total RAM of this computer (BiaPy runs on it), or None if unknown."""
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except Exception:
        pass
    try:  # Linux and macOS
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, ValueError, OSError):
        pass
    try:  # Windows
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
                (n, ctypes.c_ulonglong)
                for n in ("total", "avail", "totalPage", "availPage", "totalVirtual", "availVirtual", "availExt")
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.total)
    except Exception:
        pass
    return None


def output_channels(cfg):
    """Channels the model predicts per voxel (approximate for the channels that expand, e.g. rays)."""
    workflow = cfg["PROBLEM"]["TYPE"]
    n_classes = int(cfg["DATA"].get("N_CLASSES", 2))
    classes = n_classes if n_classes > 2 else 0
    if workflow == "SEMANTIC_SEG":
        return max(1, classes)
    if workflow == "DETECTION":
        return 1 + classes
    if workflow == "INSTANCE_SEG":
        inst = cfg["PROBLEM"].get("INSTANCE_SEG", {})
        opts = (inst.get("DATA_CHANNELS_EXTRA_OPTS") or [{}])[0]
        ndim = 3 if cfg["PROBLEM"]["NDIM"] == "3D" else 2
        n = 0
        for ch in inst.get("DATA_CHANNELS", ["F", "C"]):
            if ch in ("I", "We"):
                continue
            if ch == "R":
                n += opts.get("R", {}).get("nrays", 32 if ndim == 2 else 96)
            elif ch in ("E_offset", "E_sigma"):
                n += ndim
            elif ch == "Db" and opts.get("Db", {}).get("val_type") == "discretize":
                n += 11
            elif ch == "A":
                n += ndim
            else:
                n += 1
        return max(1, n) + classes
    return 1


def patch_factor(shape, patch, padding, overlap=None):
    """
    Voxels of all the patches BiaPy cuts an image of ``shape`` into, over the voxels of the image (``f`` above), as
    its crop_data_with_overlap / crop_3D_data_with_overlap count them: ``ceil(size / step)`` patches per axis, with a
    step of ``(patch - 2 * padding) * (1 - overlap)``. Images smaller than the patch are reflected up to it.
    """
    overlap = overlap or [0] * len(shape)
    total = 1.0
    for size, p, pad, ov in zip(shape, patch, padding, overlap):
        step = max(1, int((p - 2 * pad) * (1 - ov if ov else 1)))
        total *= np.ceil(max(size, p) / step) * p
    return total / float(np.prod(shape))


def estimate_bytes(mode, voxels, in_channels, out_channels, factor, image_bytes):
    """
    Peak memory of predicting an image of ``voxels`` voxels ("whole", "reduce" or "chunks"), with ``in_channels`` and
    ``out_channels`` (the model's) and patches covering ``factor`` times its voxels (:func:`patch_factor`).
    ``image_bytes``: what the image takes when read whole (0 if it is read by chunks, i.e. H5/Zarr by chunks).
    """
    if mode == "chunks":
        return CHUNKS_BASE_BYTES + 2 * image_bytes
    fixed, per_patch, per_out, per_in = MEMORY_MODEL[mode]
    per_voxel = per_patch * factor * (in_channels + out_channels) + per_out * out_channels + per_in * in_channels
    # Measured with uint8 images: wider types take more to read them
    return fixed + voxels * per_voxel + max(0.0, image_bytes - voxels * in_channels)


def _tuple(value, n, default=0):
    """A (z, )y, x tuple from the configuration (a string like "(5, 21, 32)", a list, or nothing)."""
    import ast

    if isinstance(value, str):
        value = ast.literal_eval(value)
    if not value:
        return (default,) * n
    return tuple(value)[:n]


def test_files(folder):
    try:
        return [f for f in os.listdir(folder) if not f.startswith(".")]
    except OSError:
        return []


def plan_inference(cfg, sample_info, lines, ram_bytes=None):
    """
    Sets how the test images are predicted so that BiaPy fits in the RAM (see the module's description), and adds
    what was decided to ``lines``.
    """
    if not cfg.get("TEST", {}).get("ENABLE"):
        return
    fp = common.fingerprint(sample_info, "DATA.TEST.PATH", "images")
    if fp is None:
        return
    ram = host_ram_bytes() if ram_bytes is None else ram_bytes
    if not ram:
        lines.append("Test: the memory of this computer could not be read, so the test images are predicted whole")
        return
    budget = ram * RAM_FRACTION
    workflow = cfg["PROBLEM"]["TYPE"]
    shape = fp["max_shape"]
    voxels = float(np.prod(shape))
    in_ch = int(fp.get("channels") or 1)
    itemsize = max([np.dtype(d).itemsize for d in fp.get("dtype") or ["uint8"]] or [1])
    image_bytes = voxels * itemsize * in_ch
    out_ch = output_channels(cfg)
    # How much the patches overlap: the patch size and test padding are set by now (planned, from the checkpoint...)
    n = len(shape)
    patch = _tuple(cfg["DATA"].get("PATCH_SIZE"), n, default=None)
    factor = 1.0
    if patch and all(p is not None and p > 0 for p in patch):
        test_data = cfg["DATA"].get("TEST", {})
        factor = patch_factor(shape, patch, _tuple(test_data.get("PADDING"), n), _tuple(test_data.get("OVERLAP"), n))
    size = "largest test image {} with {} channel{} ({:.2g} GB)".format(
        common.fmt(shape), in_ch, "s" if in_ch > 1 else "", image_bytes / 1e9
    )

    def gib(b):
        return "{:.1f} GB".format(b / 1e9)

    files = test_files(cfg["DATA"].get("TEST", {}).get("PATH", ""))
    chunked_files = bool(files) and all(f.lower().endswith(CHUNKED_FORMATS) for f in files)
    chunks_possible = cfg["PROBLEM"]["NDIM"] == "3D" and workflow in CHUNK_WORKFLOWS

    whole = estimate_bytes("whole", voxels, in_ch, out_ch, factor, image_bytes)
    if whole <= budget and not (chunks_possible and chunked_files):
        lines.append("Test: images predicted whole ({}: about {} of the {} of RAM)".format(size, gib(whole), gib(ram)))
        return
    test = cfg["TEST"]
    reduce = estimate_bytes("reduce", voxels, in_ch, out_ch, factor, image_bytes)
    if (reduce <= budget and not chunked_files) or not chunks_possible:
        test["REDUCE_MEMORY"] = True
        lines.append(
            "Test: TEST.REDUCE_MEMORY, as predicting the {} whole would take about {} of the {} of RAM (about {} "
            "with it)".format(size, gib(whole), gib(ram), gib(reduce))
        )
        if reduce > budget:
            lines.append(
                "WARNING: the test images may not fit in the memory even so: crop them into smaller images, or use "
                "a computer with more RAM"
            )
        return

    chunks = estimate_bytes("chunks", voxels, in_ch, out_ch, factor, 0 if chunked_files else image_bytes)
    by_chunks = test.setdefault("BY_CHUNKS", {})
    by_chunks["ENABLE"] = True
    by_chunks.setdefault("WORKFLOW_PROCESS", {})["TYPE"] = "chunk_by_chunk"
    # The result is a Zarr file; a TIF too (the front-ends show TIFs) if it fits: read whole to write it, which took
    # 16 bytes per voxel more (instances of a 248 MiB volume: 6.0 GiB instead of 1.9)
    by_chunks["SAVE_OUT_TIF"] = bool(chunks + voxels * SAVE_TIF_BYTES_PER_VOXEL <= budget)
    # Images that can't be read by chunks (e.g. TIF) are read whole, in (Z, Y, X, C) order
    if files and not chunked_files:
        cfg["DATA"]["TEST"]["INPUT_IMG_AXES_ORDER"] = "ZYXC"
        if cfg["DATA"]["TEST"].get("LOAD_GT"):
            cfg["DATA"]["TEST"]["INPUT_MASK_AXES_ORDER"] = "ZYXC"
    why = (
        "the test images are H5/Zarr files" if chunked_files
        else "predicting the {} whole would take about {} of the {} of RAM ({} with TEST.REDUCE_MEMORY)".format(
            size, gib(whole), gib(ram), gib(reduce)
        )
    )
    lines.append(
        "Test: by chunks (TEST.BY_CHUNKS), as {}; by chunks about {}. The results are saved as Zarr{}".format(
            why, gib(chunks), " and TIF" if by_chunks["SAVE_OUT_TIF"] else " (too large to save them as TIF too)"
        )
    )
    if chunks > budget:
        lines.append(
            "WARNING: even by chunks the test images may not fit in the memory, as TIF images are read whole: save "
            "them as Zarr or H5, which BiaPy reads by chunks"
        )
