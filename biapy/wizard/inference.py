"""
Inference that fits in the computer's memory (RAM). From the shape of the largest test image, the peak memory of
BiaPy's test phase is estimated for its ways of predicting an image, and the first one that fits is used:

* whole images (BiaPy's default): the image, all its patches and their predictions (float32), merged into the full
  prediction, then post-processed (e.g. the watershed of instance segmentation) in memory.
* TEST.REDUCE_MEMORY: patches cropped and merged separately and float16 predictions. Less memory, but slower.
* TEST.BY_CHUNKS (3D semantic segmentation, instance segmentation and detection): the patches are predicted and
  written to a Zarr file one by one and post-processed tile by tile, so memory hardly grows with the image. Only the
  image itself is held in memory when it is not an H5/Zarr file (those are read by chunks too).

The peak memory (PSS of BiaPy and its workers) was measured with the Wizard's 3D configurations on volumes of 124 to
495 MiB (uint8, one channel), batch size 1, and fits a line, a fixed part plus a part per voxel of the image:

=====================  =========================  =========================  ===========================
workflow (outputs)     whole images               TEST.REDUCE_MEMORY         TEST.BY_CHUNKS
=====================  =========================  =========================  ===========================
semantic (1 channel)   1.3 GiB + 43 bytes/voxel   1.5 GiB + 28 bytes/voxel   1.6-2.4 GiB
instance (F, C)        2.9 GiB + 80 bytes/voxel   2.1 GiB + 51 bytes/voxel   1.8-1.9 GiB
=====================  =========================  =========================  ===========================

E.g. a 1.5 GB uint8 volume needs about 120 GB to segment its instances whole, 80 GB with REDUCE_MEMORY and 4-5 GB
by chunks.
"""
import os

import numpy as np

from . import common

GIB = 1024**3
# (bytes per voxel of the largest test image, fixed bytes, output channels measured with) of each way of predicting,
# from the table above. The part per voxel grows with the output channels (float32 predictions per channel).
MEMORY_MODEL = {
    "whole": {"SEMANTIC_SEG": (43, 1.3 * GIB, 1), "INSTANCE_SEG": (80, 2.9 * GIB, 2)},
    "reduce": {"SEMANTIC_SEG": (28, 1.5 * GIB, 1), "INSTANCE_SEG": (51, 2.1 * GIB, 2)},
}
# Workflows not measured: detection is predicted as semantic segmentation (one channel per class), the rest too
SAME_AS = "SEMANTIC_SEG"
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


def estimate_bytes(mode, workflow, voxels, out_channels, image_bytes):
    """
    Peak memory of predicting an image of ``voxels`` voxels ("whole", "reduce" or "chunks"). ``image_bytes``: what the
    image takes when read whole (0 if it is read by chunks, i.e. H5/Zarr by chunks).
    """
    if mode == "chunks":
        return CHUNKS_BASE_BYTES + 2 * image_bytes
    per_voxel, fixed, measured_channels = MEMORY_MODEL[mode].get(workflow, MEMORY_MODEL[mode][SAME_AS])
    return fixed + voxels * per_voxel * max(1.0, out_channels / measured_channels)


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
    voxels = float(np.prod(fp["max_shape"]))
    itemsize = max([np.dtype(d).itemsize for d in fp.get("dtype") or ["uint8"]] or [1])
    image_bytes = voxels * itemsize * (fp.get("channels") or 1)
    out_ch = output_channels(cfg)
    size = "largest test image {} ({:.2g} GB)".format(common.fmt(fp["max_shape"]), image_bytes / 1e9)

    def gib(b):
        return "{:.1f} GB".format(b / 1e9)

    files = test_files(cfg["DATA"].get("TEST", {}).get("PATH", ""))
    chunked_files = bool(files) and all(f.lower().endswith(CHUNKED_FORMATS) for f in files)
    chunks_possible = cfg["PROBLEM"]["NDIM"] == "3D" and workflow in CHUNK_WORKFLOWS

    whole = estimate_bytes("whole", workflow, voxels, out_ch, image_bytes)
    if whole <= budget and not (chunks_possible and chunked_files):
        lines.append("Test: images predicted whole ({}: about {} of the {} of RAM)".format(size, gib(whole), gib(ram)))
        return
    test = cfg["TEST"]
    reduce = estimate_bytes("reduce", workflow, voxels, out_ch, image_bytes)
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

    chunks = estimate_bytes("chunks", workflow, voxels, out_ch, 0 if chunked_files else image_bytes)
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
