"""
Tests of the training patch sampling (DATA.TRAIN.PATCH_SAMPLING), in particular nnU-Net's foreground oversampling
(biapy/data/patch_sampling.py): the class locations found in the targets, the centres picked from them, the batches that
force foreground and the crops centred on the chosen voxel.

How to run
----------
No GPU, data or network needed. From the repository root, with the BiaPy environment active::

    pytest tests/test_patch_sampling.py -q
"""
import numpy as np
import pytest
from scipy import ndimage

from biapy.data.generators.augmentors import random_3D_crop_pair, random_crop_pair
from biapy.data.patch_sampling import (
    ClassLocations,
    ForegroundOversamplingBatchSampler,
    class_locations_from_mask,
    compute_class_locations,
    forced_in_batch,
    pick_center,
    shift_center,
)


def _semantic_mask():
    """A large square of class 1, a tiny object of class 1 far from it and a few voxels of class 2."""
    m = np.zeros((300, 300, 1), np.uint8)
    m[20:220, 20:220] = 1
    m[280:282, 280:282] = 1
    m[250:253, 10:13] = 2
    return m


def test_locations_cover_the_classes_evenly():
    loc = class_locations_from_mask(_semantic_mask(), "SEMANTIC_SEG", max_locations=100, rng=np.random.default_rng(0))
    assert set(loc["classes"]) == {"1", "2"}
    c1 = loc["classes"]["1"]
    # About max_locations grid points plus one voxel per object; all inside the class
    assert len(c1["coords"]) <= 100 + 2
    mask = _semantic_mask()[..., 0]
    assert np.all(mask[tuple(c1["coords"].T)] == 1)
    # Even coverage: every voxel of the square is within one grid step (per axis) of a point
    grid = np.zeros((300, 300), bool)
    grid[tuple(c1["coords"].T)] = True
    d = ndimage.distance_transform_cdt(~grid, metric="chessboard")
    assert d[20:220, 20:220].max() <= c1["step"]
    # The tiny object is not missed, and small classes keep all their voxels
    assert ((c1["coords"] >= 280).all(axis=1)).any()
    assert len(loc["classes"]["2"]["coords"]) == 9 and loc["classes"]["2"]["step"] == 1


def test_locations_instances_and_detection():
    lab = np.zeros((64, 64, 2), np.uint16)
    lab[5:15, 5:15, 0], lab[5:15, 5:15, 1] = 1, 1
    lab[30:60, 30:60, 0], lab[30:60, 30:60, 1] = 2, 2
    loc = class_locations_from_mask(lab, "INSTANCE_SEG", 10000)
    assert set(loc["classes"]) == {"1", "2"}
    assert set(loc["classes"]["2"]["objects"].tolist()) == {2}
    # Without a class channel, all the instances are one class
    loc = class_locations_from_mask(lab[..., :1], "INSTANCE_SEG", 10000)
    assert set(loc["classes"]) == {"1"} and set(loc["classes"]["1"]["objects"].tolist()) == {1, 2}
    pts = np.zeros((32, 32, 1), np.uint8)
    pts[4:7, 4:7] = pts[20:23, 20:23] = 1
    loc = class_locations_from_mask(pts, "DETECTION", 10000)
    assert len(np.unique(loc["classes"]["1"]["objects"])) == 2  # one object per point


def test_pick_center_balances_classes_and_objects():
    rng = np.random.default_rng(0)
    m = np.zeros((200, 200, 1), np.uint8)
    m[0:150, 0:150] = 1  # a large object
    m[190:193, 190:193] = 1  # a small one of the same class
    m[170:173, 10:13] = 2  # a rare class
    loc = class_locations_from_mask(m, "SEMANTIC_SEG", 500, rng)
    picks = [pick_center(loc, rng) for _ in range(2000)]
    classes = [m[p][0] for p in picks]
    # Classes chosen uniformly, whatever their size
    assert 0.4 < np.mean(np.array(classes) == 2) < 0.6
    small = lambda p: p[0] >= 185 and p[1] >= 185  # noqa: E731
    # Within class 1, the small object is rarely chosen by voxel, as often as the large one by object
    by_voxel = np.mean([small(p) for p, c in zip(picks, classes) if c == 1])
    picks = [pick_center(loc, rng, balance_objects=True) for _ in range(2000)]
    by_object = np.mean([small(p) for p in picks if m[p][0] == 1])
    assert by_voxel < 0.05 and 0.4 < by_object < 0.6


def test_pick_center_no_foreground():
    assert pick_center(class_locations_from_mask(np.zeros((16, 16, 1), np.uint8), "SEMANTIC_SEG"), np.random.default_rng()) is None


def test_shift_keeps_the_voxel_in_the_patch():
    rng = np.random.default_rng(0)
    patch = (64, 128)
    shifts = np.array([np.subtract(shift_center((100, 100), patch, 0.25, rng), (100, 100)) for _ in range(500)])
    assert np.all(np.abs(shifts) <= np.array(patch) * 0.25)
    assert np.abs(shifts).max() > 0  # the voxel is not always in the middle
    # Never beyond the patch, whatever the jitter
    shifts = np.array([np.subtract(shift_center((100,), (10,), 0.5, rng), (100,)) for _ in range(200)])
    assert np.abs(shifts).max() <= 4


def test_forced_positions_as_nnunet():
    assert forced_in_batch(2, 0.33) == [False, True]
    assert sum(forced_in_batch(12, 0.33)) == 4 and forced_in_batch(12, 0.33)[-4:] == [True] * 4
    # Batches of 1: forced with probability PERCENT
    rng = np.random.default_rng(0)
    assert 0.25 < np.mean([forced_in_batch(1, 0.33, rng)[0] for _ in range(1000)]) < 0.41


def test_batch_sampler_single_and_distributed():
    s = ForegroundOversamplingBatchSampler(10, 4, 0.33, seed=1)
    batches = list(s)
    assert len(batches) == len(s) == 3
    assert sorted(i for b in batches for i, _ in b) == list(range(10))
    assert [f for _, f in batches[0]] == [False, False, False, True]
    s.set_epoch(1)
    assert [i for b in s for i, _ in b] != [i for b in batches for i, _ in b]  # another order each epoch
    # Two processes: disjoint shares covering every sample, same order in both
    parts = [ForegroundOversamplingBatchSampler(10, 2, 0.33, num_replicas=2, rank=r, seed=1) for r in (0, 1)]
    seen = [i for p in parts for b in p for i, _ in b]
    assert set(seen) == set(range(10)) and len(seen) == 10
    full = [[x[1] for x in b] for p in parts for b in p if len(b) == 2]
    assert full and all(f == [False, True] for f in full)


def test_crop_centred_on_voxel():
    img = np.random.rand(100, 120, 1)
    mask = np.zeros((100, 120, 1))
    mask[90, 5] = 1
    _, m = random_crop_pair(img, mask, (32, 32), center=(90, 5))
    assert m.sum() == 1  # clamped inside the image, the voxel still in the crop
    vol = np.random.rand(20, 60, 60, 1)
    vmask = np.zeros((20, 60, 60, 1))
    vmask[10, 30, 30] = 1
    _, vm = random_3D_crop_pair(vol, vmask, (8, 16, 16), center=(10, 30, 30))
    assert vm.sum() == 1 and vm[4, 8, 8, 0] == 1


class _Sample:
    def __init__(self, img):
        self.img, self.coords, self.fid = img, None, 0

    def img_is_loaded(self):
        return True


class _Dataset:
    def __init__(self, masks, tmp_path):
        self.sample_list = [_Sample(m) for m in masks]
        info = type("I", (), {"path": str(tmp_path / "missing.tif")})
        self.dataset_info = [info]


def test_locations_stored_and_reused(tmp_path):
    masks = [_semantic_mask(), _semantic_mask()[::-1]]
    store = compute_class_locations(_Dataset(masks, tmp_path), "SEMANTIC_SEG", False, str(tmp_path / "loc"), 100)
    loc = store.get(1)
    assert set(loc["classes"]) == {"1", "2"} and loc["shape"] == [300, 300]
    t = (tmp_path / "loc" / "0000000.npz").stat().st_mtime_ns
    store = compute_class_locations(_Dataset(masks, tmp_path), "SEMANTIC_SEG", False, str(tmp_path / "loc"), 100)
    assert (tmp_path / "loc" / "0000000.npz").stat().st_mtime_ns == t  # reused
    assert isinstance(store, ClassLocations)


def _config(**train):
    from yacs.config import CfgNode as CN

    from biapy.config.config import Config, update_dependencies

    manager = Config("/tmp", "patch_sampling")
    manager._C.merge_from_other_cfg(CN({"PROBLEM": {"TYPE": "SEMANTIC_SEG"}, "TRAIN": {"ENABLE": True},
                                        "DATA": {"TRAIN": {"PATH": "/tmp", "GT_PATH": "/tmp", **train}}}))
    update_dependencies(manager)
    return manager.get_cfg_defaults()


def test_old_key_converted():
    from biapy.engine.check_configuration import convert_old_model_cfg_to_current_version

    cfg = convert_old_model_cfg_to_current_version({"DATA": {"TRAIN": {"EXTRACT_RANDOM_PATCH": True}}})
    assert cfg["DATA"]["TRAIN"] == {"PATCH_SAMPLING": "random"}
    cfg = convert_old_model_cfg_to_current_version({"DATA": {"EXTRACT_RANDOM_PATCH": False, "TRAIN": {}}})
    assert cfg["DATA"]["TRAIN"] == {"PATCH_SAMPLING": "tiles"}


@pytest.mark.parametrize(
    "train,message",
    [
        ({"PATCH_SAMPLING": "everywhere"}, "must be one of"),
        ({"PATCH_SAMPLING": "foreground_oversampling", "PROBABILITY_MAP": True}, "PROBABILITY_MAP"),
        ({"PATCH_SAMPLING": "foreground_oversampling", "FOREGROUND_OVERSAMPLING": {"PERCENT": 1.5}}, "PERCENT"),
    ],
)
def test_invalid_settings_rejected(train, message):
    from biapy.engine.check_configuration import check_configuration

    with pytest.raises(ValueError, match=message):
        check_configuration(_config(**train), "patch_sampling", check_data_paths=False)
