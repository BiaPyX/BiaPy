"""
Linking of 2D instances across frames with a Trackastra model (https://github.com/weigertlab/trackastra).

Same steps as ``Trackastra.track(..., mode="greedy")``: region features per instance, association scores over sliding
windows of frames, greedy linking and conversion to Cell Tracking Challenge (CTC) tracks.
"""

from collections import deque
from typing import Dict, List, Tuple

import numpy as np
import torch
from numpy.typing import NDArray
from scipy import ndimage as ndi
from scipy.spatial.distance import cdist
from skimage.measure import regionprops_table

REGION_FEATURES = ("equivalent_diameter_area", "intensity_mean", "inertia_tensor")


def normalize_movie(frames: NDArray) -> NDArray:
    """Percentile (1, 99.8) normalization of the whole ``(T, Y, X)`` movie, subsampled 4x in Y/X if large."""
    frames = frames.astype(np.float32)
    sample = frames[:, ::4, ::4] if all(s > 256 for s in frames.shape[-2:]) else frames
    mi, ma = np.percentile(sample, (1, 99.8)).astype(np.float32)
    return (frames - mi) / (ma - mi + 1e-8)


def _border_dist(mask: NDArray, labels: NDArray, cutoff: int = 5) -> NDArray:
    """Per instance, closeness to the image border: 1 touching it, 0 at ``cutoff`` pixels or more."""
    ramps = [np.minimum(np.minimum(np.arange(s), np.arange(s)[::-1]).astype(np.float32) / cutoff, 1) for s in mask.shape]
    return np.asarray(ndi.maximum(1 - np.minimum.outer(*ramps), labels=mask, index=labels), dtype=np.float32)


def object_features(masks: NDArray, frames: NDArray) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    One token per instance of each frame.

    Parameters
    ----------
    masks : NDArray
        ``(T, Y, X)`` instance labels.

    frames : NDArray
        ``(T, Y, X)`` normalized images.

    Returns
    -------
    timepoints : NDArray
        ``(N,)`` frame of each instance.

    labels : NDArray
        ``(N,)`` label of each instance in its frame.

    coords : NDArray
        ``(N, 2)`` centroid ``(y, x)``.

    features : NDArray
        ``(N, 7)`` equivalent diameter, mean intensity, inertia tensor (4) and closeness to the border.
    """
    timepoints, labels, coords, features = [], [], [], []
    for t, (mask, img) in enumerate(zip(masks, frames)):
        props = regionprops_table(mask, intensity_image=img, properties=("label", "centroid") + REGION_FEATURES)
        timepoints.append(np.full(len(props["label"]), t, dtype=np.int32))
        labels.append(props["label"].astype(np.int32))
        coords.append(np.stack([props["centroid-0"], props["centroid-1"]], -1).astype(np.float32))
        feats = [props[k] for k in props if k.startswith(REGION_FEATURES)]
        features.append(np.stack(feats + [_border_dist(mask, props["label"])], -1).astype(np.float32))
    return np.concatenate(timepoints), np.concatenate(labels), np.concatenate(coords), np.concatenate(features)


def association_scores(
    model: torch.nn.Module,
    timepoints: NDArray,
    coords: NDArray,
    features: NDArray,
    device: torch.device,
    edge_threshold: float = 0.05,
) -> Dict[Tuple[int, int], np.float32]:
    """
    Scores of linking each instance to the instances of the next frame, averaged over the windows that see both.

    Parameters
    ----------
    model : torch.nn.Module
        Trackastra model (``biapy.models.trackastra.TrackingTransformer``).

    timepoints, coords, features : NDArray
        Instances, as returned by :func:`object_features`.

    device : torch.device
        Device of the model.

    edge_threshold : float, optional
        Scores below it are not kept.

    Returns
    -------
    scores : dict
        ``{(i, j): score}`` with ``i`` and ``j`` instance indexes, ``j`` in the frame after ``i``'s.
    """
    n_frames = int(timepoints.max()) + 1
    window = min(model.window, n_frames)
    acc: Dict[Tuple[int, int], List] = {}
    # shortcut: one window per forward pass (no padding/batching), batch them if linking speed matters
    for t1 in range(n_frames - window + 1):
        idx = np.flatnonzero((timepoints >= t1) & (timepoints < t1 + window))
        if len(idx) == 0:
            continue
        ts = timepoints[idx]
        c = torch.from_numpy(np.concatenate([ts[:, None].astype(np.float32), coords[idx]], axis=1))[None]
        with torch.no_grad():
            A = model(c.to(device), torch.from_numpy(features[idx])[None].to(device))[0].cpu().numpy()
        A[ts[None, :] - ts[:, None] != 1] = 0
        for i, j in zip(*np.nonzero(A >= edge_threshold)):
            s = acc.setdefault((int(idx[i]), int(idx[j])), [np.float32(0), np.float32(0)])
            s[0] += A[i, j]
            s[1] += np.float32(1)
    return {k: s / c for k, (s, c) in acc.items()}


def greedy_linking(
    scores: Dict[Tuple[int, int], np.float32],
    timepoints: NDArray,
    coords: NDArray,
    allow_divisions: bool = True,
    threshold: float = 0.5,
    max_distance: float = 256,
    max_neighbors: int = 10,
) -> List[Tuple[int, int]]:
    """
    Links taken from the highest score down, skipping those that would merge two instances or give more than two
    daughters (one if ``allow_divisions`` is False).

    Parameters
    ----------
    scores : dict
        Output of :func:`association_scores`.

    timepoints, coords : NDArray
        Instances, as returned by :func:`object_features`.

    allow_divisions : bool, optional
        Whether an instance can be linked to two instances of the next frame.

    threshold : float, optional
        Minimum score of a link.

    max_distance : float, optional
        Maximum centroid distance of a link.

    max_neighbors : int, optional
        Candidate links per instance: its closest instances of the next frame that have a score.

    Returns
    -------
    links : list of tuple
        ``(i, j)`` links, in the order they were taken.
    """
    candidates = []
    for t in range(int(timepoints.max())):
        ni, nj = np.flatnonzero(timepoints == t), np.flatnonzero(timepoints == t + 1)
        if len(ni) == 0 or len(nj) == 0:
            continue
        dists = cdist(coords[ni], coords[nj])
        for a, i in enumerate(ni):
            n = 0
            for b in np.argsort(dists[a]):
                if n >= max_neighbors:
                    break
                if dists[a, b] <= max_distance and (int(i), int(nj[b])) in scores:
                    candidates.append((int(i), int(nj[b]), scores[(int(i), int(nj[b]))]))
                    n += 1

    max_children = 2 if allow_divisions else 1
    n_children: Dict[int, int] = {}
    has_parent = set()
    links = []
    for i, j, s in sorted(candidates, key=lambda e: e[2], reverse=True):
        if s < threshold:
            break
        if j in has_parent or n_children.get(i, 0) >= max_children:
            continue
        links.append((i, j))
        has_parent.add(j)
        n_children[i] = n_children.get(i, 0) + 1
    return links


def links_to_ctc(
    links: List[Tuple[int, int]], timepoints: NDArray, labels: NDArray, n_frames: int
) -> Tuple[NDArray, List[Dict[int, int]]]:
    """
    CTC tracks of the linked instances. Instances without any link are left out.

    Parameters
    ----------
    links : list of tuple
        Output of :func:`greedy_linking`.

    timepoints, labels : NDArray
        Instances, as returned by :func:`object_features`.

    n_frames : int
        Number of frames.

    Returns
    -------
    tracks : NDArray
        ``(K, 4)`` CTC rows ``L B E P``: track label, first frame, last frame and parent track label (0 if none).

    label_maps : list of dict
        Per frame, ``{instance label: track label}``.
    """
    children: Dict[int, List[int]] = {}
    nodes: Dict[int, None] = {}
    for i, j in links:
        nodes.setdefault(i)
        nodes.setdefault(j)
        children.setdefault(i, []).append(j)
    has_parent = {j for _, j in links}

    starts = deque((p, d) for p in nodes for d in children.get(p, []) if len(children[p]) == 2)
    starts.extend((-1, n) for n in nodes if n not in has_parent)
    tracklets = []
    while starts:
        parent, node = starts.popleft()
        chain = [node]
        while len(children.get(chain[-1], [])) == 1:
            chain.append(children[chain[-1]][0])
        tracklets.append((int(timepoints[node]), parent, chain))
    tracklets.sort(key=lambda tr: (tr[0], tr[1]))

    track_of_end = {-1: 0}
    label_maps: List[Dict[int, int]] = [{} for _ in range(n_frames)]
    rows = []
    for k, (start, parent, chain) in enumerate(tracklets, start=1):
        track_of_end[chain[-1]] = k
        for n in chain:
            label_maps[timepoints[n]][int(labels[n])] = k
        rows.append([k, start, int(timepoints[chain[-1]]), track_of_end[parent]])
    return np.array(rows, dtype=np.int64).reshape(-1, 4), label_maps
