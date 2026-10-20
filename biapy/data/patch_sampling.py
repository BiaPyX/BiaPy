"""
Foreground oversampling of the training patches (``DATA.TRAIN.PATCH_SAMPLING = "foreground_oversampling"``), as
nnU-Net does: a share of the patches of each batch is centred on a voxel of a class chosen at random among the classes
of the image, the rest are taken at random.

To centre them without reading the whole mask every time, the voxels of each class are stored once per image as a
regular grid (with a random offset) that covers the class evenly with at most ``max_locations`` points, plus one voxel
of each object (connected component, or instance) so none is missed however small. When a patch is centred, the point
is moved at random within its grid cell and the patch is shifted so the point is not always in its middle.
"""
import json
import math
import os
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from numpy.typing import NDArray
from scipy import ndimage

PATCH_SAMPLING_MODES = ("tiles", "random", "foreground_oversampling")
# Workflows whose targets have classes to oversample
FOREGROUND_OVERSAMPLING_WORKFLOWS = ("SEMANTIC_SEG", "INSTANCE_SEG", "DETECTION")
CACHE_VERSION = 2


def class_locations_from_mask(
    mask: NDArray, workflow: str, max_locations: int = 10000, rng: Optional[np.random.Generator] = None
) -> Dict:
    """
    Locations of the voxels of each class of a target, to centre patches on them.

    Parameters
    ----------
    mask : NDArray
        Target as BiaPy reads it, ``(y, x, c)`` or ``(z, y, x, c)``. Semantic segmentation: class of each voxel in the
        first channel. Instance segmentation: instance ids in the first channel and, optionally, their class in the
        second. Detection: the points' masks in the first channel and, optionally, their class in the second.
    workflow : str
        ``PROBLEM.TYPE``.
    max_locations : int, optional
        Approximate maximum number of grid points per class (one voxel per object is added to them).
    rng : numpy Generator, optional
        For the offset of the grid.

    Returns
    -------
    locations : dict
        ``shape`` of the image and, per class (key: class id as str), ``coords`` (``(n, ndim)`` int32), ``objects``
        (``(n,)`` int32, the object each point belongs to) and ``step`` (grid step, in voxels).
    """
    rng = rng if rng is not None else np.random.default_rng()
    first = mask[..., 0]
    if workflow == "SEMANTIC_SEG":
        classes_map = first
        objects_map = None  # connected components, per class
    else:
        foreground = first > 0
        classes_map = np.where(foreground, mask[..., 1], 0) if mask.shape[-1] > 1 else foreground.astype(np.uint8)
        objects_map = first if workflow == "INSTANCE_SEG" else None
    out: Dict = {"shape": list(first.shape), "classes": {}}
    for c in np.unique(classes_map):
        if c <= 0:
            continue
        in_class = classes_map == c
        if objects_map is not None:
            objects = np.where(in_class, objects_map, 0).astype(np.int64)
        else:
            objects, _ = ndimage.label(in_class)
        n = int(in_class.sum())
        step = max(1, int(math.ceil((n / max(1, max_locations)) ** (1.0 / in_class.ndim))))
        # Regular grid with a random offset, so different images (and runs) don't share the same grid
        offset = tuple(int(rng.integers(0, step)) for _ in range(in_class.ndim))
        grid = in_class[tuple(slice(o, None, step) for o in offset)]
        coords = np.argwhere(grid) * step + np.array(offset)
        if len(coords) > max_locations:  # unevenly shaped classes may exceed it slightly
            coords = coords[np.sort(rng.choice(len(coords), max_locations, replace=False))]
        # One voxel of every object, so objects smaller than the grid step are not missed. They are not grid points:
        # patches are centred exactly on them (moving them within a grid cell could leave the object)
        flat = objects.ravel()
        ids, first_index = np.unique(flat, return_index=True)
        keep = ids > 0
        extra = np.stack(np.unravel_index(first_index[keep], objects.shape), axis=1)
        on_grid = np.concatenate([np.ones(len(coords), bool), np.zeros(len(extra), bool)])
        coords = np.concatenate([coords, extra]) if len(coords) else extra
        coords, unique_index = np.unique(coords, axis=0, return_index=True)
        coords = coords.astype(np.int32)
        out["classes"][str(int(c))] = {
            "coords": coords,
            "objects": objects[tuple(coords.T)].astype(np.int32),
            "on_grid": on_grid[unique_index],
            "step": step,
        }
    return out


def pick_center(
    locations: Dict, rng: np.random.Generator, balance_objects: bool = False
) -> Optional[Tuple[int, ...]]:
    """
    A voxel to centre a patch on: a class chosen uniformly among the ones in the image, then (``balance_objects``) one of
    its objects uniformly, then one of its points, moved at random within its grid cell. None if the image has no
    foreground.
    """
    classes = [k for k, v in locations["classes"].items() if len(v["coords"])]
    if not classes:
        return None
    cls = locations["classes"][classes[int(rng.integers(len(classes)))]]
    coords, objects, step = cls["coords"], cls["objects"], int(cls["step"])
    if balance_objects:
        ids = np.unique(objects)
        chosen = ids[int(rng.integers(len(ids)))]
        candidates = np.flatnonzero(objects == chosen)
        k = candidates[int(rng.integers(len(candidates)))]
    else:
        k = int(rng.integers(len(coords)))
    point = coords[k]
    if step > 1 and cls.get("on_grid", np.ones(len(coords), bool))[k]:
        # Anywhere in its grid cell, so the centres are not limited to the grid points
        point = point + rng.integers(-(step // 2), step - step // 2, size=len(point))
    shape = locations["shape"]
    return tuple(int(min(max(p, 0), s - 1)) for p, s in zip(point, shape))


def shift_center(
    center: Sequence[int], patch_size: Sequence[int], max_shift: float, rng: np.random.Generator
) -> Tuple[int, ...]:
    """
    Centre of the patch so that ``center`` lands at a random place within ``max_shift`` (fraction of the patch size) of
    the patch's middle, always inside the patch.
    """
    out = []
    for c, p in zip(center, patch_size):
        limit = min(int(max_shift * p), max(0, p // 2 - 1))
        out.append(int(c + (rng.integers(-limit, limit + 1) if limit > 0 else 0)))
    return tuple(out)


def forced_in_batch(batch_len: int, percent: float, rng: Optional[np.random.Generator] = None) -> List[bool]:
    """
    Which positions of a batch must contain foreground: the last ones, as nnU-Net (e.g. 1 of 2, 4 of 12). When that
    leaves none (e.g. batches of 1), each position is forced with probability ``percent``.
    """
    forced = batch_len - int(round(batch_len * (1 - percent)))
    if forced == 0 and percent > 0:
        rng = rng if rng is not None else np.random.default_rng()
        return [bool(rng.random() < percent) for _ in range(batch_len)]
    return [k >= batch_len - forced for k in range(batch_len)]


class ForegroundOversamplingBatchSampler(torch.utils.data.Sampler):
    """
    Batches of ``(sample index, force foreground)`` pairs: the samples in a random order (each epoch), and in each batch
    the last ``batch - round(batch * (1 - percent))`` forced to contain foreground (see :func:`forced_in_batch`). With several processes
    (distributed training) each takes its share of the samples, as ``DistributedSampler`` does, and applies the rule to
    its own batches.
    """

    def __init__(
        self,
        num_samples: int,
        batch_size: int,
        percent: float,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.percent = percent
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = 0
        self.per_replica = int(math.ceil(num_samples / num_replicas))

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _local_indices(self) -> List[int]:
        order = (
            np.random.default_rng(self.seed + self.epoch).permutation(self.num_samples)
            if self.shuffle
            else np.arange(self.num_samples)
        )
        # The same order in every process (same seed): padded to a multiple of the processes, then split
        total = self.per_replica * self.num_replicas
        if total > len(order):
            order = np.concatenate([order, order[: total - len(order)]])
        return [int(i) for i in order[self.rank : total : self.num_replicas]]

    def __iter__(self) -> Iterator[List[Tuple[int, bool]]]:
        local = self._local_indices()
        rng = np.random.default_rng((self.seed, self.epoch, self.rank))
        for b in range(0, len(local), self.batch_size):
            batch = local[b : b + self.batch_size]
            if self.drop_last and len(batch) < self.batch_size:
                return
            yield list(zip(batch, forced_in_batch(len(batch), self.percent, rng)))

    def __len__(self) -> int:
        if self.drop_last:
            return self.per_replica // self.batch_size
        return int(math.ceil(self.per_replica / self.batch_size))


class ClassLocations:
    """
    The class locations of every sample of a dataset, stored in a folder (one file per sample) so they are computed
    once, shared by the data-loading workers without copies and reused when the job runs again on the same data.
    """

    def __init__(self, folder: str):
        self.folder = folder
        self._cache: Dict[int, Dict] = {}

    def path(self, index: int) -> str:
        return os.path.join(self.folder, "{:07d}.npz".format(index))

    def get(self, index: int) -> Dict:
        if index not in self._cache:
            with np.load(self.path(index)) as f:
                loc: Dict = {"shape": [int(v) for v in f["shape"]], "classes": {}}
                for c in [str(v) for v in f["classes"]]:
                    loc["classes"][c] = {
                        "coords": f["coords_" + c],
                        "objects": f["objects_" + c],
                        "on_grid": f["on_grid_" + c],
                        "step": int(f["step_" + c]),
                    }
            # Few samples are kept: the dataset may be large
            if len(self._cache) > 256:
                self._cache.clear()
            self._cache[index] = loc
        return self._cache[index]

    @staticmethod
    def save(path: str, loc: Dict):
        arrays = {"shape": np.array(loc["shape"]), "classes": np.array(list(loc["classes"].keys()), dtype=str)}
        for c, v in loc["classes"].items():
            arrays["coords_" + c] = v["coords"]
            arrays["objects_" + c] = v["objects"]
            arrays["on_grid_" + c] = v["on_grid"]
            arrays["step_" + c] = np.array(v["step"])
        np.savez(path, **arrays)


def compute_class_locations(Y, workflow: str, is_3d: bool, folder: str, max_locations: int = 10000, seed: int = 0) -> ClassLocations:
    """
    Class locations of every sample of the targets ``Y`` (a :class:`~biapy.data.dataset.BiaPyDataset` with one sample per
    image), stored in ``folder``. Already stored locations of the same files and settings are reused. With several
    processes, each computes its share of the samples.
    """
    from tqdm import tqdm

    from biapy.data.data_manipulation import read_img_as_ndarray
    from biapy.utils.misc import is_dist_avail_and_initialized, is_main_process

    os.makedirs(folder, exist_ok=True)
    n = len(Y.sample_list)
    identity = []
    for s in Y.sample_list:
        if s.coords is not None:
            raise ValueError(
                "DATA.TRAIN.PATCH_SAMPLING = 'foreground_oversampling' needs one sample per image (the targets can't be "
                "read by patches, e.g. from Zarr/H5 files)"
            )
        path = Y.dataset_info[s.fid].path
        st = os.stat(path) if os.path.isfile(path) else None
        identity.append([path, st.st_size if st else -1, int(st.st_mtime) if st else -1])
    manifest = {
        "version": CACHE_VERSION,
        "workflow": workflow,
        "max_locations": max_locations,
        "seed": seed,
        "files": identity,
    }
    manifest_path = os.path.join(folder, "manifest.json")
    try:
        with open(manifest_path) as f:
            reusable = json.load(f) == json.loads(json.dumps(manifest))
    except (OSError, ValueError):
        reusable = False
    store = ClassLocations(folder)
    if reusable and all(os.path.isfile(store.path(i)) for i in range(n)):
        print("Reusing the class locations stored in {}".format(folder))
        return store

    print("Finding the class locations to oversample the foreground . . .")
    from biapy.data.data_manipulation import _rank_local_indices

    local = _rank_local_indices(n)
    for i in tqdm(local, total=len(local), disable=not is_main_process()):
        i = int(i)
        sample = Y.sample_list[i]
        if sample.img_is_loaded():
            mask = sample.img
        else:
            mask = read_img_as_ndarray(Y.dataset_info[sample.fid].path, is_3d=is_3d)
        loc = class_locations_from_mask(mask, workflow, max_locations, np.random.default_rng(seed + i))
        ClassLocations.save(store.path(i), loc)
    if is_dist_avail_and_initialized():
        import torch.distributed as dist

        dist.barrier()
    if is_main_process():
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
    return store
