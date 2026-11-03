"""
Test mode that fits in the RAM: whole images, TEST.REDUCE_MEMORY or TEST.BY_CHUNKS (3D semantic segmentation,
instance segmentation and detection), the first whose estimated peak memory fits. With ``f`` the voxels of all the
patches over the image's (3.3-5.6 with the wizard's padding), measured within 5%::

    whole:          1.3 GiB + voxels * (4 * f * (inputs + outputs) + 11 * outputs + 7 * inputs) bytes
    REDUCE_MEMORY:  1.6 GiB + voxels * (1.5 * f * (inputs + outputs) + 12 * outputs + 8 * inputs) bytes

By chunks: 1.6-2.4 GiB, plus the image when it is not Zarr/H5.
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


def plan_inference(cfg, sample_info, lines, ram_bytes=None):
    """
    Sets how the test images are predicted so that BiaPy fits in the RAM (see the module's description), and adds
    what was decided to ``lines``.

    Returns
    -------
    estimate : dict or None
        ``mode`` ("whole", "reduce" or "chunks") and ``bytes``, the RAM estimated for it; None if not estimated.
    """
    if not cfg.get("TEST", {}).get("ENABLE"):
        return None
    fp = common.fingerprint(sample_info, "DATA.TEST.PATH", "images")
    if fp is None:
        return None
    ram = host_ram_bytes() if ram_bytes is None else ram_bytes
    if not ram:
        lines.append("Test: the memory of this computer could not be read, so the test images are predicted whole")
        return None
    budget = ram * RAM_FRACTION
    workflow = cfg["PROBLEM"]["TYPE"]
    shape = fp["max_shape"]
    voxels = float(np.prod(shape))
    in_ch = int(fp.get("channels") or 1)
    itemsize = max([np.dtype(d).itemsize for d in fp.get("dtype") or ["uint8"]] or [1])
    image_bytes = voxels * itemsize * in_ch
    out_ch = common.output_channels(cfg)
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

    from biapy.wizard.data_check import folder_images
    from biapy.data.data_3D_manipulation import is_chunked

    files = folder_images(cfg["DATA"].get("TEST", {}).get("PATH", ""))
    chunked_files = bool(files) and all(is_chunked(f) for f in files)
    # Tracking links the instances of the frames held in memory, so it can't run by chunks
    tracking = cfg["PROBLEM"].get("TRACKING", {}).get("ENABLE", False)
    chunks_possible = cfg["PROBLEM"]["NDIM"] == "3D" and workflow in CHUNK_WORKFLOWS and not tracking

    whole = estimate_bytes("whole", voxels, in_ch, out_ch, factor, image_bytes)
    if whole <= budget and not (chunks_possible and chunked_files):
        lines.append("Test: images predicted whole ({}: about {} of the {} of RAM)".format(size, gib(whole), gib(ram)))
        return {"mode": "whole", "bytes": whole}
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
        return {"mode": "reduce", "bytes": reduce}

    chunks = estimate_bytes("chunks", voxels, in_ch, out_ch, factor, 0 if chunked_files else image_bytes)
    by_chunks = test.setdefault("BY_CHUNKS", {})
    by_chunks["ENABLE"] = True
    by_chunks.setdefault("WORKFLOW_PROCESS", {})["TYPE"] = "chunk_by_chunk"
    # A TIF too (the front-ends show TIFs) if it fits: writing it reads the result whole
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
    return {"mode": "chunks", "bytes": chunks}
