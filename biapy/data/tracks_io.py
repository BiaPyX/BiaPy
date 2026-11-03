"""
Reading and writing of tracks: Cell Tracking Challenge (CTC) format and GEFF (https://github.com/live-image-tracking-tools/geff).

CTC: one label mask per frame, whose labels are the track labels, plus a text file with one ``L B E P`` row per track
(label, first frame, last frame, parent track label or 0).
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import tifffile
import zarr
from numpy.typing import NDArray
from skimage.util import map_array

from biapy.data.data_manipulation import read_img_as_ndarray
from biapy.utils.misc import os_walk_clean

GEFF_VERSION = "1.3"


def write_ctc_tracks(out_dir: str, masks: NDArray, tracks: NDArray, label_maps: List[Dict[int, int]]) -> NDArray:
    """
    Write the tracks in CTC result format: ``mask<t>.tif`` per frame and ``res_track.txt``.

    Parameters
    ----------
    out_dir : str
        Output folder.

    masks : NDArray
        ``(T, (Z), Y, X)`` instance labels of the frames.

    tracks : NDArray
        ``(K, 4)`` ``L B E P`` rows.

    label_maps : list of dict
        Per frame, ``{instance label: track label}``. Instances not in it are left out.

    Returns
    -------
    tracked : NDArray
        ``(T, (Z), Y, X)`` masks labelled with the track labels.
    """
    os.makedirs(out_dir, exist_ok=True)
    tracked = np.zeros(masks.shape, dtype=np.uint16 if len(tracks) < 2**16 else np.uint32)
    digits = max(3, len(str(len(masks) - 1)))
    for t, (mask, label_map) in enumerate(zip(masks, label_maps)):
        if label_map:
            map_array(mask, np.array(list(label_map)), np.array(list(label_map.values()), dtype=tracked.dtype), out=tracked[t])
        tifffile.imwrite(os.path.join(out_dir, f"mask{t:0{digits}d}.tif"), tracked[t], compression="zlib")
    np.savetxt(os.path.join(out_dir, "res_track.txt"), tracks, fmt="%d")
    return tracked


def read_frames(folder: str, is_3d: bool) -> NDArray:
    """``(T, (Z), Y, X)`` single-channel images of a folder, one per frame in name order (``.txt`` files skipped)."""
    files = [f for f in next(os_walk_clean(folder))[2] if not f.endswith(".txt")]
    frames = [read_img_as_ndarray(os.path.join(folder, f), is_3d=is_3d) for f in files]
    if any(f.shape[-1] != 1 for f in frames):
        raise ValueError(f"Tracking needs single-channel frames: {folder}")
    return np.stack([f[..., 0] for f in frames])


def read_ctc_tracks(folder: str, is_3d: bool) -> Tuple[NDArray, NDArray]:
    """
    Read CTC tracks, e.g. a ground truth ``TRA`` folder: label masks (one per frame, in name order) and the ``.txt``
    track file (``man_track.txt`` or ``res_track.txt``).

    Parameters
    ----------
    folder : str
        CTC folder.

    is_3d : bool
        Whether the frames are 3D.

    Returns
    -------
    masks : NDArray
        ``(T, (Z), Y, X)`` track labels.

    tracks : NDArray
        ``(K, 4)`` ``L B E P`` rows.
    """
    txt = [f for f in next(os_walk_clean(folder))[2] if f.endswith(".txt")]
    if len(txt) != 1:
        raise ValueError(f"Expected one '.txt' track file in {folder}, found {txt}")
    tracks = np.loadtxt(os.path.join(folder, txt[0]), dtype=np.int64, ndmin=2).reshape(-1, 4)
    return read_frames(folder, is_3d), tracks


def write_geff(
    path: str,
    links: List[Tuple[int, int]],
    timepoints: NDArray,
    coords: NDArray,
    track_ids: Dict[int, int],
):
    """
    Write the tracking graph as GEFF (Zarr v2): one node per linked instance, with its frame, centroid and track label
    (``seg_id``, its label in the CTC masks, and ``track_id``, the GEFF tracklet), and one edge per link.

    Parameters
    ----------
    path : str
        Output ``.geff`` folder.

    links : list of tuple
        ``(i, j)`` links between instance indexes.

    timepoints : NDArray
        ``(N,)`` frame of each instance.

    coords : NDArray
        ``(N, ndim)`` centroid ``((z), y, x)`` of each instance, in pixels.

    track_ids : dict
        ``{instance index: track label}`` of the linked instances.
    """
    nodes = np.array(list(track_ids), dtype=np.int64)
    space = ["z", "y", "x"][-coords.shape[1] :]
    props = {"t": timepoints[nodes].astype(np.int64)}
    props.update({a: coords[nodes, i].astype(np.float32) for i, a in enumerate(space)})
    props["seg_id"] = props["track_id"] = np.array([track_ids[n] for n in nodes], dtype=np.int64)

    group = zarr.open_group(path, mode="w", zarr_format=2)
    group.attrs["geff"] = {
        "geff_version": GEFF_VERSION,
        "directed": True,
        "axes": [{"name": "t", "type": "time"}] + [{"name": a, "type": "space"} for a in space],
        "node_props_metadata": {k: {"identifier": k, "dtype": str(v.dtype)} for k, v in props.items()},
        "edge_props_metadata": {},
        "track_node_props": {"tracklet": "track_id"},
    }
    group.create_array("nodes/ids", data=nodes)
    for k, v in props.items():
        group.create_array(f"nodes/props/{k}/values", data=v)
    group.create_array("edges/ids", data=np.array(links, dtype=np.int64).reshape(-1, 2))
