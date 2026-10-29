"""
Tests of the configuration wizard's logic (:mod:`biapy.wizard`): data checks and fingerprints, the strategies
that adapt the configuration to the data, the instance representations proposed, the data preview and the
configuration built from the wizard's answers (validated with BiaPy's own configuration check).

How to run
----------
No GPU and no real data are needed: the datasets are small synthetic ones written to a temporary folder.
From the repository root, with the BiaPy environment active (``conda activate BiaPy_env``)::

    pytest tests/test_wizard.py -q                  # whole suite (~1 min)
    pytest tests/test_wizard.py -k representation   # only the instance representations
    pytest tests/test_wizard.py -k config -v        # configurations built from answers

Without activating the environment, call its interpreter directly::

    ~/miniforge3/envs/BiaPy_env/bin/python -m pytest tests/test_wizard.py -q

The memory model of the U-Net plans is fitted with ``tests/calibrate_unet_memory.py`` (needs a GPU).
"""
import os

import numpy as np
import pytest

from biapy.wizard import detection, instance_seg, unet
from biapy.wizard.config_builder import WizardError, build_config, tuples_to_lists, write_config
from biapy.wizard.data_check import check_folder, semantic_classes
from biapy.wizard.planning import plan_config
from biapy.wizard.preview import preview

GB = 1024**3
GPU = [{"index": 0, "name": "test", "memory_mb": 8000}]


################
# Data helpers #
################
def _write_2d(path, arr):
    import tifffile

    tifffile.imwrite(path, arr)


def _write_stack(path, arr, spacing=None):
    import tifffile

    if spacing is None:
        tifffile.imwrite(path, arr)
    else:
        z, y, x = spacing
        tifffile.imwrite(
            path, arr, imagej=True, resolution=(1 / x, 1 / y), metadata={"spacing": z, "unit": "micron", "axes": "ZYX"}
        )


def _semantic_3d(tmp_path, n=12, spacing=(1.0, 0.25, 0.25), labels=(1, 2)):
    rng = np.random.default_rng(0)
    xdir, ydir = tmp_path / "x", tmp_path / "y"
    xdir.mkdir()
    ydir.mkdir()
    for i in range(n):
        mask = np.zeros((16, 64, 64), np.uint8)
        for j, lab in enumerate(labels):
            mask[4:8, 10 + 20 * j : 22 + 20 * j, 10:22] = lab
        raw = (rng.normal(100, 10, mask.shape) + 50 * mask).clip(0, 255).astype(np.uint8)
        _write_stack(str(xdir / "{:02d}.tif".format(i)), raw, spacing)
        _write_stack(str(ydir / "{:02d}.tif".format(i)), mask, spacing)
    return str(xdir), str(ydir)


def _instances(tmp_path, kind, n=12):
    """2D instance masks: 'round' blobs, 'rods' or 'c_shapes'."""
    from skimage.draw import disk, rectangle

    d = tmp_path / kind
    d.mkdir()
    for i in range(n):
        lab = np.zeros((128, 128), np.uint16)
        for j in range(4):
            r, c = 20 + 30 * j, 30 + 20 * (i % 3)
            if kind == "round":
                lab[disk((r, c), 10, shape=lab.shape)] = j + 1
            elif kind == "rods":
                rr, cc = rectangle((r - 3, c - 25), extent=(6, 70), shape=lab.shape)
                lab[rr, cc] = j + 1
            else:  # C shapes: a ring with a gap
                m = np.zeros_like(lab, bool)
                m[disk((r, c + 30), 12, shape=lab.shape)] = True
                m[disk((r, c + 30), 7, shape=lab.shape)] = False
                m[r - 3 : r + 4, c + 30 : c + 45] = False
                lab[m] = j + 1
        _write_2d(str(d / "{:02d}.tif".format(i)), lab)
    return str(d)


def _raw_like(tmp_path, folder):
    raw = tmp_path / "raw"
    raw.mkdir()
    rng = np.random.default_rng(0)
    for f in sorted(os.listdir(folder)):
        _write_2d(str(raw / f), rng.integers(0, 255, (128, 128), dtype=np.uint8))
    return str(raw)


def _dense_3d(tmp_path, n=3):
    """3D instances filling the whole volume (EM neuron-like)."""
    from scipy import ndimage

    d = tmp_path / "dense"
    d.mkdir()
    rng = np.random.default_rng(1)
    for i in range(n):
        seeds = np.zeros((16, 64, 64), np.int32)
        for j, (z, y, x) in enumerate(rng.uniform((0, 0, 0), (16, 64, 64), (30, 3)).astype(int), start=1):
            seeds[z, y, x] = j
        _, idx = ndimage.distance_transform_edt(seeds == 0, return_indices=True)
        _write_stack(str(d / "{:02d}.tif".format(i)), seeds[tuple(idx)].astype(np.uint16))
    return str(d)


def _points(tmp_path, is_3d, step=20):
    d = tmp_path / "csv"
    d.mkdir()
    g = np.arange(10, 200, step)
    for i in range(3):
        if is_3d:
            zz, yy, xx = np.meshgrid(np.arange(2, 40, step // 4), g, g, indexing="ij")
            pts = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], 1)
            header = "axis-0,axis-1,axis-2"
        else:
            yy, xx = np.meshgrid(g, g, indexing="ij")
            pts = np.stack([yy.ravel(), xx.ravel()], 1)
            header = "axis-0,axis-1"
        np.savetxt(str(d / "{}.csv".format(i)), pts, delimiter=",", header=header, comments="", fmt="%d")
    return str(d)


def _check(folder, key, workflow, ndim):
    r = check_folder(folder, key, workflow, ndim)
    assert not r["error"], r["error_message"]
    return r


def _fp(folder, key, workflow, ndim):
    return _check(folder, key, workflow, ndim)["sample_info"]["fingerprint"]


#############
# U-Net     #
#############
def test_topology_isotropic_2d():
    t = unet.get_topology((1, 1), (512, 512))
    # Downsampled while the feature maps are at least 8 px long: 512 -> 4 px
    assert len(t["pools"]) == 7
    assert all(p == [2, 2] for p in t["pools"])
    assert t["divisor"] == [128, 128]
    assert all(k == [3, 3] for k in t["conv_kernels"])


def test_topology_anisotropic_3d():
    # Z voxels 4 times larger: XY downsampled alone twice, then all axes
    t = unet.get_topology((4, 1, 1), (32, 256, 256))
    assert t["pools"][:3] == [[1, 2, 2], [1, 2, 2], [2, 2, 2]]
    # Z kernels are 1 wide until the XY spacing is within a factor 2 of Z's
    assert t["conv_kernels"][0] == [1, 3, 3]
    assert t["conv_kernels"][1] == [1, 3, 3]
    assert t["conv_kernels"][2] == [3, 3, 3]
    assert all(p % d == 0 for p, d in zip(t["patch_size"], t["divisor"]))


def test_topology_y_and_x_apart():
    # X (16 px) is downsampled twice down to 4 px; Y goes on alone, written as [y, x] pairs in MODEL.YX_DOWN
    t = unet.get_topology((1, 1), (256, 16))
    assert t["pools"] == [[2, 2], [2, 2], [2, 1], [2, 1], [2, 1]]
    plan = unet.plan_unet((1, 1), (256, 16), 1, 1, 1000 * 256 * 16, 11 * GB)
    assert unet.plan_to_biapy(plan)["YX_DOWN"] == [2, 2, [2, 1], [2, 1], [2, 1]]


@pytest.mark.parametrize(
    "spacing,shape,vram",
    [
        ((1, 1), (1024, 1024), 11 * GB),
        ((1, 1), (300, 5000), 6 * GB),
        ((1, 1, 1), (200, 512, 512), 11 * GB),
        ((5, 1, 1), (30, 1024, 1024), 24 * GB),
        ((1, 1, 1), (64, 64, 64), 4 * GB),
    ],
)
def test_plan_fits(spacing, shape, vram):
    plan = unet.plan_unet(spacing, shape, 1, 1, 50 * float(np.prod(shape)), vram)
    assert plan["vram_estimate_gb"] <= plan["vram_budget_gb"] or plan["batch_size"] == unet.MIN_BATCH
    assert plan["batch_size"] >= unet.MIN_BATCH
    assert len(plan["feature_maps"]) == len(plan["pools"]) + 1
    assert max(plan["feature_maps"]) <= unet.MAX_FEATURES[len(spacing)]
    # BiaPy needs the patch to be divisible by the downsampling and longer than 2 at every level
    size = np.array(plan["patch_size"])
    for p in plan["pools"]:
        assert np.all(size % p == 0) and np.all(size > 2)
        size = size // p


# Output of nnU-Net v2.8.1's get_pool_and_conv_props(spacing, patch, 4, 999999), the rules the planner follows:
# (spacing, patch, downsampling per level, kernel per level, patch padded).
NNUNET_TOPOLOGIES = [
    ((1, 1), (320, 2048), "2x2 2x2 2x2 2x2 2x2 2x2 1x2 1x2", "3x3 3x3 3x3 3x3 3x3 3x3 3x3 3x3 3x3", (320, 2048)),
    ((1, 1), (64, 640), "2x2 2x2 2x2 2x2 1x2 1x2", "3x3 3x3 3x3 3x3 3x3 3x3 3x3", (64, 640)),
    ((1, 1, 1), (48, 48, 256), "2x2x2 2x2x2 2x2x2 1x1x2 1x1x2", "3x3x3 3x3x3 3x3x3 3x3x3 3x3x3 3x3x3", (48, 48, 256)),
    ((4, 1, 1), (16, 64, 512), "1x2x2 1x2x2 2x2x2 2x2x2 1x1x2 1x1x2", "1x3x3 1x3x3 3x3x3 3x3x3 3x3x3 3x3x3 3x3x3", (16, 64, 512)),
    ((1, 1), (512, 512), "2x2 2x2 2x2 2x2 2x2 2x2 2x2", "3x3 3x3 3x3 3x3 3x3 3x3 3x3 3x3", (512, 512)),
    ((1, 1), (768, 1024), "2x2 2x2 2x2 2x2 2x2 2x2 2x2", "3x3 3x3 3x3 3x3 3x3 3x3 3x3 3x3", (768, 1024)),
    ((1, 1), (96, 96), "2x2 2x2 2x2 2x2", "3x3 3x3 3x3 3x3 3x3", (96, 96)),
    ((1, 1), (100, 100), "2x2 2x2 2x2 2x2", "3x3 3x3 3x3 3x3 3x3", (112, 112)),
    ((1, 1), (1280, 1024), "2x2 2x2 2x2 2x2 2x2 2x2 2x2 2x2", "3x3 3x3 3x3 3x3 3x3 3x3 3x3 3x3 3x3", (1280, 1024)),
    ((1, 1, 1), (128, 128, 128), "2x2x2 2x2x2 2x2x2 2x2x2 2x2x2", "3x3x3 3x3x3 3x3x3 3x3x3 3x3x3 3x3x3", (128, 128, 128)),
    ((1, 1, 1), (48, 192, 256), "2x2x2 2x2x2 2x2x2 1x2x2 1x2x2", "3x3x3 3x3x3 3x3x3 3x3x3 3x3x3 3x3x3", (48, 192, 256)),
    ((2, 1, 1), (56, 192, 192), "1x2x2 2x2x2 2x2x2 2x2x2 1x2x2", "1x3x3 3x3x3 3x3x3 3x3x3 3x3x3 3x3x3", (56, 192, 192)),
    ((4, 1, 1), (24, 256, 256), "1x2x2 1x2x2 2x2x2 2x2x2 1x2x2 1x2x2", "1x3x3 1x3x3 3x3x3 3x3x3 3x3x3 3x3x3 3x3x3", (24, 256, 256)),
    ((5, 1, 1), (12, 384, 384), "1x2x2 1x2x2 2x2x2 1x2x2 1x2x2 1x2x2", "1x3x3 1x3x3 3x3x3 3x3x3 3x3x3 3x3x3 3x3x3", (12, 384, 384)),
    ((10, 1, 1), (20, 256, 256), "1x2x2 1x2x2 1x2x2 2x2x2 2x2x2 1x2x2", "1x3x3 1x3x3 1x3x3 3x3x3 3x3x3 3x3x3 3x3x3", (20, 256, 256)),
    ((3.3, 1, 1), (20, 256, 384), "1x2x2 2x2x2 2x2x2 1x2x2 1x2x2 1x2x2", "1x3x3 3x3x3 3x3x3 3x3x3 3x3x3 3x3x3 3x3x3", (20, 256, 384)),
    ((1, 0.25, 0.25), (16, 64, 64), "1x2x2 1x2x2 2x2x2 2x2x2", "1x3x3 1x3x3 3x3x3 3x3x3 3x3x3", (16, 64, 64)),
    ((1, 1, 1), (40, 128, 120), "2x2x2 2x2x2 2x2x2 1x2x2 1x2x2", "3x3x3 3x3x3 3x3x3 3x3x3 3x3x3 3x3x3", (40, 128, 128)),
    ((0.5, 1, 1), (64, 64, 64), "2x1x1 2x2x2 2x2x2 2x2x2 1x2x2", "3x1x1 3x3x3 3x3x3 3x3x3 3x3x3 3x3x3", (64, 64, 64)),
]


@pytest.mark.parametrize("spacing,patch,pools,kernels,padded", NNUNET_TOPOLOGIES)
def test_topology_as_nnunet(spacing, patch, pools, kernels, padded):
    def dec(s):
        return [[int(v) for v in p.split("x")] for p in s.split()]

    t = unet.get_topology(spacing, patch)
    assert t["pools"] == dec(pools)
    assert t["conv_kernels"] == dec(kernels)
    assert tuple(t["patch_size"]) == padded


def test_large_patch_preferred_over_batch():
    # As nnU-Net, the patch is only shrunk until the minimum batch size fits, also in 2D (where nnU-Net's
    # reference batch size is 12): 768x1024 images fit whole with a batch of 2 in 11 GB
    plan = unet.plan_unet((1, 1), (768, 1024), 1, 1, 165 * 768 * 1024, 11 * GB)
    assert plan["patch_size"] == [768, 1024] and plan["batch_size"] == unet.MIN_BATCH


def test_batch_limited_by_dataset_size():
    # A small dataset: no more than 5% of it in a batch, but at least 2
    plan = unet.plan_unet((1, 1), (256, 256), 1, 1, 10 * 256 * 256, 40 * GB)
    assert plan["batch_size"] == unet.MIN_BATCH


def test_plan_to_biapy_3d():
    plan = unet.plan_unet((4, 1, 1), (40, 512, 512), 1, 1, 1e9, 11 * GB)
    m = unet.plan_to_biapy(plan)
    n = len(m["FEATURE_MAPS"])
    assert len(m["Z_DOWN"]) == len(m["YX_DOWN"]) == n - 1
    assert len(m["ISOTROPY"]) == len(m["CONV_LAYERS"]) == n
    assert m["Z_DOWN"][0] == 1 and m["ISOTROPY"][0] is False


###############
# Data checks #
###############
def test_voxel_size_from_metadata(tmp_path):
    from biapy.data.data_manipulation import read_img_as_ndarray, voxel_size_from_meta

    arr = np.zeros((10, 32, 48), np.uint8)
    _write_stack(str(tmp_path / "a.tif"), arr, (2.0, 0.5, 0.5))
    _write_stack(str(tmp_path / "b.tif"), arr)
    _, meta = read_img_as_ndarray(str(tmp_path / "a.tif"), is_3d=True, load_meta=True)
    assert voxel_size_from_meta(meta, is_3d=True) == pytest.approx((2.0, 0.5, 0.5))
    _, meta = read_img_as_ndarray(str(tmp_path / "b.tif"), is_3d=True, load_meta=True)
    assert voxel_size_from_meta(meta, is_3d=True) is None


def test_check_semantic_folders(tmp_path):
    x, y = _semantic_3d(tmp_path)
    rx = _check(x, "DATA.TRAIN.PATH", "SEMANTIC_SEG", "3D")
    assert rx["data_constraints"]["DATA.PATCH_SIZE_C"] == 1
    assert rx["data_constraints"]["DATA.TRAIN.PATH"] == 12
    fx = rx["sample_info"]["fingerprint"]
    assert fx["spacing"] == pytest.approx([1.0, 0.25, 0.25])
    assert fx["n_calibrated"] == 12 and fx["median_shape"] == [16, 64, 64]

    ry = _check(y, "DATA.TRAIN.GT_PATH", "SEMANTIC_SEG", "3D")
    assert ry["data_constraints"]["DATA.N_CLASSES"] == 3
    fy = ry["sample_info"]["fingerprint"]
    assert fy["labels"] == [0, 1, 2]
    assert fy["objects"]["1"]["extent_median"] == [4.0, 12.0, 12.0]
    assert fy["foreground_fraction"]["median"] == pytest.approx(2 * 4 * 12 * 12 / (16 * 64 * 64))


def test_check_progress(tmp_path):
    folder = _instances(tmp_path, "round", n=5)
    calls = []
    _check_ok = check_folder(folder, "DATA.TRAIN.GT_PATH", "INSTANCE_SEG", "2D", progress=lambda *a: calls.append(a))
    assert not _check_ok["error"]
    # One call per file, then one when the data is analyzed
    assert [c[:2] for c in calls] == [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (5, 5)]
    assert calls[0][2] == "00.tif" and calls[-1][2] is None
    calls.clear()
    check_folder(_points(tmp_path, False), "DATA.TRAIN.GT_PATH", "DETECTION", "2D", progress=lambda *a: calls.append(a))
    assert [c[:2] for c in calls] == [(1, 3), (2, 3), (3, 3)]


def test_semantic_classes():
    assert semantic_classes({0, 255}) == 2
    assert semantic_classes({0, 1, 3}) == 4  # a missing class is still a class
    assert semantic_classes({0.0, 0.5, 1.0}) == 3


def test_check_errors(tmp_path):
    assert "does not exist" in check_folder(str(tmp_path / "nope"), "DATA.TRAIN.PATH", "SEMANTIC_SEG", "2D")["error_message"]
    (tmp_path / "empty").mkdir()
    assert check_folder(str(tmp_path / "empty"), "DATA.TRAIN.PATH", "SEMANTIC_SEG", "2D")["error"]
    d = tmp_path / "mixed"
    d.mkdir()
    _write_2d(str(d / "a.tif"), np.zeros((32, 32), np.uint8))
    _write_2d(str(d / "b.tif"), np.zeros((32, 32, 3), np.uint8))
    r = check_folder(str(d), "DATA.TRAIN.PATH", "SEMANTIC_SEG", "2D")
    assert r["error"] and "channels" in r["error_message"]


def test_check_points(tmp_path):
    r = _check(_points(tmp_path, False), "DATA.TRAIN.GT_PATH", "DETECTION", "2D")
    assert r["data_constraints"]["DATA.TRAIN.GT_PATH"] == 3
    assert r["sample_info"]["fingerprint"]["kind"] == "detection_points"


def test_check_points_header_spaces(tmp_path):
    # Column names are stripped, as BiaPy does when it creates the detection masks
    d = tmp_path / "csv"
    d.mkdir()
    (d / "a.csv").write_text(" axis-0 , axis-1 ,class\n10,10,1\n30,40,2\n")
    r = _check(str(d), "DATA.TRAIN.GT_PATH", "DETECTION", "2D")
    assert r["data_constraints"]["DATA.N_CLASSES"] == 2


def test_check_points_missing_column(tmp_path):
    d = tmp_path / "csv"
    d.mkdir()
    (d / "a.csv").write_text("axis-0,y\n1,2\n")
    r = check_folder(str(d), "DATA.TRAIN.GT_PATH", "DETECTION", "2D")
    assert r["error"] and "axis-1" in r["error_message"]


def test_pair_csv_with_images():
    from biapy.data.data_manipulation import pair_csv_with_images

    # One image per CSV file, in the order of the CSV files
    assert pair_csv_with_images(["a.tif", "b.tif"], ["b.csv", "a.csv"]) == [("b.tif", True), ("a.tif", True)]
    assert pair_csv_with_images(["a.tif"], ["x.csv"]) == [("a.tif", False)]


##############
# Strategies #
##############
def _cfg(ndim="3D"):
    return {
        "PROBLEM": {"TYPE": "SEMANTIC_SEG", "NDIM": ndim},
        "MODEL": {"SOURCE": "biapy", "ARCHITECTURE": "resunet", "Z_DOWN": [1, 1, 1, 1]},
        "DATA": {"N_CLASSES": 2, "PATCH_SIZE": "(20, 128, 128, 1)", "VAL": {"SPLIT_TRAIN": 0.1}, "TRAIN": {}, "TEST": {}},
        "TRAIN": {"ENABLE": True, "BATCH_SIZE": 1},
        "TEST": {"ENABLE": True},
    }


def test_plan_semantic_seg(tmp_path):
    x, y = _semantic_3d(tmp_path, labels=(1, 3))
    ry = _check(y, "DATA.TRAIN.GT_PATH", "SEMANTIC_SEG", "3D")
    si = {
        "DATA.TRAIN.PATH": {"fingerprint": _fp(x, "DATA.TRAIN.PATH", "SEMANTIC_SEG", "3D")},
        "DATA.TRAIN.GT_PATH": ry["sample_info"],
    }
    cfg = _cfg()
    cfg["DATA"]["N_CLASSES"] = ry["data_constraints"]["DATA.N_CLASSES"]
    report = plan_config(cfg, si, GPU)
    assert report["applied"], report
    assert cfg["MODEL"]["ARCHITECTURE"] == "unet"
    assert cfg["DATA"]["N_CLASSES"] == 4  # labels 0..3, though 2 is missing
    assert cfg["MODEL"]["Z_DOWN"][0] == 1  # Z spacing 4 times XY's
    assert cfg["LOSS"]["TYPE"] == ["DICE", "CE"]
    assert cfg["AUGMENTOR"]["ELASTIC"] is False
    # Enough images: random patches, a third of them centred on the foreground (nnU-Net's sampling)
    assert cfg["DATA"]["TRAIN"]["PATCH_SAMPLING"] == "foreground_oversampling"
    assert cfg["DATA"]["TRAIN"]["FOREGROUND_OVERSAMPLING"]["PERCENT"] == 0.33
    assert any("labels [2]" in line for line in report["lines"])


def test_quick_training(tmp_path):
    x, y = _semantic_3d(tmp_path)
    answers = _answers("SEMANTIC_SEG", "3D", [("DATA.TRAIN.PATH", x), ("DATA.TRAIN.GT_PATH", y)], training_length="quick")
    cfg, report = build_config(answers, GPU)
    assert "training_length" not in cfg  # not a BiaPy variable
    # 20 epochs of about 250 iterations, without early stopping, the learning rate warming up in the first one
    line = next(l for l in report["lines"] if "training:" in l)
    assert line.startswith("Quick training: {} epochs".format(cfg["TRAIN"]["EPOCHS"]))
    iterations = int(line.split(" epochs of ")[1].split()[0])
    assert 0.9 * 5000 <= cfg["TRAIN"]["EPOCHS"] * iterations <= 1.1 * 5000
    assert cfg["TRAIN"]["PATIENCE"] == -1 and cfg["TRAIN"]["LR_SCHEDULER"]["WARMUP_COSINE_DECAY_EPOCHS"] == 1
    assert cfg["TRAIN"]["SAVE_TRAIN_PREDS_FREQ"] == 1
    _biapy_check(cfg, tmp_path)


def test_quick_training_not_planned(tmp_path):
    # Workflows without a strategy (and pretrained models): a fifth of the default epochs
    x = tmp_path / "x"
    x.mkdir()
    for i in range(3):
        _write_2d(str(x / "{}.tif".format(i)), np.zeros((64, 64), np.uint8))
    answers = _answers("DENOISING", "2D", [("DATA.TRAIN.PATH", str(x))], training_length="quick")
    cfg, report = build_config(answers, GPU)
    assert cfg["TRAIN"]["EPOCHS"] == 30 and cfg["TRAIN"]["PATIENCE"] == -1
    assert "Quick training: 30 epochs" in report["lines"]
    _, full = build_config(_answers("DENOISING", "2D", [("DATA.TRAIN.PATH", str(x))]), GPU)
    assert not any("Quick" in l for l in full.get("lines", []))


def _test_images(tmp_path, names, shape, ndim="3D"):
    """A test folder (empty files with these names) and the fingerprint of its images, as checked by the wizard."""
    folder = tmp_path / "test_x"
    folder.mkdir()
    for n in names:
        (folder / n).write_bytes(b"")
    fp = {"kind": "images", "n_files": len(names), "channels": 1, "max_shape": list(shape), "dtype": ["uint8"]}
    return str(folder), {"DATA.TEST.PATH": {"fingerprint": fp}}


def test_inference_fits_in_memory(tmp_path):
    from biapy.wizard.inference import plan_inference

    gb = 1e9
    # Small images: predicted whole, as BiaPy does by default
    folder, si = _test_images(tmp_path, ["a.tif"], (100, 512, 512))
    cfg = _cfg()
    cfg["DATA"]["TEST"]["PATH"] = folder
    lines = []
    plan_inference(cfg, si, lines, ram_bytes=32 * gb)
    assert "BY_CHUNKS" not in cfg["TEST"] and "REDUCE_MEMORY" not in cfg["TEST"]
    assert lines[0].startswith("Test: images predicted whole")

    # A 1.5 GB volume on a 32 GB laptop (which ran out of memory): by chunks, read whole as it is a TIF
    cfg = _cfg()
    cfg["DATA"]["TEST"]["PATH"] = folder
    si["DATA.TEST.PATH"]["fingerprint"]["max_shape"] = [1500, 1000, 1000]
    lines = []
    plan_inference(cfg, si, lines, ram_bytes=32 * gb)
    assert cfg["TEST"]["BY_CHUNKS"]["ENABLE"] and cfg["TEST"]["BY_CHUNKS"]["WORKFLOW_PROCESS"]["TYPE"] == "chunk_by_chunk"
    assert cfg["DATA"]["TEST"]["INPUT_IMG_AXES_ORDER"] == "ZYXC"
    # About 4.6 GB by chunks, but 24 GB more to write the result as TIF too: only the Zarr
    assert not cfg["TEST"]["BY_CHUNKS"]["SAVE_OUT_TIF"]
    assert lines[0].startswith("Test: by chunks") and not any(l.startswith("WARNING") for l in lines)
    # A smaller volume is saved as TIF too (the front-ends show TIFs)
    cfg = _cfg()
    cfg["DATA"]["TEST"]["PATH"] = folder
    si["DATA.TEST.PATH"]["fingerprint"]["max_shape"] = [900, 1000, 1000]
    plan_inference(cfg, si, [], ram_bytes=32 * gb)
    assert cfg["TEST"]["BY_CHUNKS"]["ENABLE"] and cfg["TEST"]["BY_CHUNKS"]["SAVE_OUT_TIF"]
    si["DATA.TEST.PATH"]["fingerprint"]["max_shape"] = [1500, 1000, 1000]
    # With more RAM, REDUCE_MEMORY is enough: semantic about 41 GB whole and 36 GB with it, 38 GB usable of 64
    cfg = _cfg()
    cfg["DATA"]["TEST"]["PATH"] = folder
    plan_inference(cfg, si, [], ram_bytes=64 * gb)
    assert cfg["TEST"].get("REDUCE_MEMORY") and "BY_CHUNKS" not in cfg["TEST"]


def test_inference_channels_and_overlap(tmp_path):
    from biapy.wizard.inference import patch_factor, plan_inference

    # BiaPy cuts the image into overlapping patches: (8, 9, 8) patches of 32x128x192 for a 165x768x1024 volume with a
    # padding of (5, 21, 32), i.e. 3.49 times its voxels; without padding (6, 6, 6), 1.31 times
    assert round(patch_factor((165, 768, 1024), (32, 128, 192), (5, 21, 32)), 2) == 3.49
    assert round(patch_factor((165, 768, 1024), (32, 128, 192), (0, 0, 0)), 2) == 1.31

    # A 145x1364x1364 volume with 3 uint16 channels, 18x160x160 patches and the Wizard's padding (1/6 of the patch),
    # which a 33 GB computer could not predict whole (estimated 13 GB counting one channel and no overlap)
    folder, si = _test_images(tmp_path, ["a.tif"], (145, 1364, 1364))
    si["DATA.TEST.PATH"]["fingerprint"].update(channels=3, dtype=["uint16"])
    cfg = _cfg()
    cfg["DATA"]["PATCH_SIZE"] = "(18, 160, 160, 3)"
    cfg["DATA"]["TEST"].update(PATH=folder, PADDING="(3, 26, 26)")
    lines = []
    plan_inference(cfg, si, lines, ram_bytes=33.3e9)
    assert cfg["TEST"]["REDUCE_MEMORY"] and "BY_CHUNKS" not in cfg["TEST"]  # about 27 GB whole, 18 GB reduced
    assert "with 3 channels" in lines[0]
    # Instance segmentation (F, C): 34 GB whole, 23 GB reduced, so by chunks
    cfg["PROBLEM"]["TYPE"] = "INSTANCE_SEG"
    cfg["PROBLEM"]["INSTANCE_SEG"] = {"DATA_CHANNELS": ["F", "C"]}
    cfg["TEST"] = {"ENABLE": True}
    plan_inference(cfg, si, [], ram_bytes=33.3e9)
    assert cfg["TEST"]["BY_CHUNKS"]["ENABLE"]


def test_resources_estimated(tmp_path):
    # What the job needs, for the front-ends to show before running it (and in the YAML's header)
    x, y = _semantic_3d(tmp_path)
    answers = _answers("SEMANTIC_SEG", "3D", [("DATA.TRAIN.PATH", x), ("DATA.TRAIN.GT_PATH", y), ("DATA.TEST.PATH", x)])
    answers["TEST.ENABLE"] = True
    cfg, report = build_config(answers, GPU)
    r = report["resources"]
    assert r["ram_train"] > 0 and r["ram_test"] > 0 and r["gpu_train"] > 0
    what = [d["what"] for d in r["disk"]]
    assert {"model checkpoint", "test predictions", "test segmentations"} <= set(what)
    assert r["disk_total"] == sum(d["bytes"] for d in r["disk"])
    assert [l for l in report["lines"] if l.startswith("Estimated resources: Disk")]
    # The checkpoint: 12 bytes per parameter of the planned U-Net
    from biapy.wizard import unet
    from biapy.wizard.inference import output_channels

    params = unet.count_parameters(report["plan"], 1, output_channels(cfg), report["plan"]["feature_maps"])
    assert next(d["bytes"] for d in r["disk"] if d["what"] == "model checkpoint") == 12 * params


def test_training_ram_measured():
    from biapy.wizard.resources import estimate_resources

    # The training peaks measured on Lucchi++ volumes (uint8, 3 workers): 9.1 GiB to read the statistics of a 495 MiB
    # volume (tiles), 3.2 GiB with twelve 124 MiB ones (foreground oversampling)
    def ram(shape, n, sampling):
        fp = {"kind": "images", "n_files": n, "channels": 1, "max_shape": list(shape), "dtype": ["uint8"]}
        si = {"DATA.TRAIN.PATH": {"fingerprint": fp}, "DATA.TRAIN.GT_PATH": {"fingerprint": {"kind": "semantic_masks"}}}
        cfg = _cfg()
        cfg["TEST"]["ENABLE"] = False
        cfg["SYSTEM"] = {"NUM_WORKERS": 3}
        cfg["DATA"]["TRAIN"]["PATCH_SAMPLING"] = sampling
        return estimate_resources(cfg, si, {})["ram_train"] / 2**30

    assert 9.1 <= ram((330, 1536, 1024), 1, "tiles") <= 9.1 * 1.1
    assert 3.16 <= ram((165, 768, 1024), 12, "foreground_oversampling") <= 3.16 * 1.1


def test_inference_by_chunks_for_h5_zarr(tmp_path):
    from biapy.wizard.inference import plan_inference

    # BiaPy reads H5/Zarr test images only by chunks, whatever their size; their axes order is left to the user
    folder, si = _test_images(tmp_path, ["a.zarr", "b.h5"], (20, 64, 64))
    cfg = _cfg()
    cfg["DATA"]["TEST"]["PATH"] = folder
    plan_inference(cfg, si, [], ram_bytes=32e9)
    assert cfg["TEST"]["BY_CHUNKS"]["ENABLE"] and "INPUT_IMG_AXES_ORDER" not in cfg["DATA"]["TEST"]


def test_inference_2d_reduce_memory(tmp_path):
    from biapy.wizard.inference import plan_inference

    # No chunks in 2D: REDUCE_MEMORY, with a warning if even so it does not fit
    folder, si = _test_images(tmp_path, ["a.tif"], (40000, 40000), "2D")
    cfg = _cfg("2D")
    cfg["DATA"]["TEST"]["PATH"] = folder
    lines = []
    plan_inference(cfg, si, lines, ram_bytes=32e9)
    assert cfg["TEST"]["REDUCE_MEMORY"] and "BY_CHUNKS" not in cfg["TEST"]
    assert any(l.startswith("WARNING") for l in lines)


def test_config_by_chunks_is_valid(tmp_path):
    from biapy.wizard.inference import plan_inference

    x, y = _semantic_3d(tmp_path)
    answers = _answers("SEMANTIC_SEG", "3D", [("DATA.TRAIN.PATH", x), ("DATA.TRAIN.GT_PATH", y)])
    cfg, report = build_config(answers, GPU)
    folder, si = _test_images(tmp_path, ["a.tif"], (1500, 1000, 1000))
    cfg["TEST"]["ENABLE"] = True
    cfg["DATA"].setdefault("TEST", {})["PATH"] = folder
    plan_inference(cfg, si, [], ram_bytes=32e9)
    assert cfg["TEST"]["BY_CHUNKS"]["ENABLE"]
    _biapy_check(cfg, tmp_path)


def test_no_plan_for_pretrained_models():
    cfg = _cfg()
    cfg["MODEL"]["SOURCE"] = "bmz"
    assert not plan_config(cfg, {}, [])["applied"]
    assert cfg["MODEL"]["ARCHITECTURE"] == "resunet"


def test_unplanned_patch_from_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    ckpt = str(tmp_path / "model-checkpoint-best.pth")
    torch.save({"cfg": {"DATA": {"PATCH_SIZE": (24, 160, 192, 1)}}}, ckpt)
    cfg = _cfg()
    cfg["TRAIN"]["ENABLE"] = False
    cfg["MODEL"]["LOAD_CHECKPOINT"] = True
    cfg["PATHS"] = {"CHECKPOINT_FILE": ckpt}
    cfg["DATA"]["PATCH_SIZE"] = "(-1, 1)"  # as the wizard does not ask for it
    report = plan_config(cfg, {}, [])
    assert not report["applied"] and report["patch_size_source"] == "the checkpoint"
    assert cfg["DATA"]["PATCH_SIZE"] == "(24, 160, 192, 1)"
    assert cfg["DATA"]["TEST"]["PADDING"] == "(4, 26, 32)"


def test_unplanned_patch_default():
    cfg = _cfg("2D")
    cfg["DATA"]["PATCH_SIZE"] = "(-1, 3)"
    report = plan_config(cfg, {}, [])  # data not analyzed
    assert not report["applied"] and report["patch_size_source"] == "fixed default"
    assert cfg["DATA"]["PATCH_SIZE"] == "(256, 256, 3)"


def test_fixed_patch_for_workflows_without_strategy():
    cfg = _cfg("3D")
    cfg["PROBLEM"]["TYPE"] = "INSTANCE_SEG"
    cfg["DATA"]["PATCH_SIZE"] = "(-1, 2)"
    report = plan_config(cfg, {}, [])
    assert not report["applied"] and report["patch_size_source"] == "fixed default"
    assert cfg["DATA"]["PATCH_SIZE"] == "(20, 256, 256, 2)"
    assert cfg["MODEL"]["ARCHITECTURE"] == "resunet"


def test_workflow_patch_kept():
    # Denoising and super-resolution get their patch size in set_default_config(), before planning
    cfg = _cfg("2D")
    cfg["PROBLEM"]["TYPE"] = "DENOISING"
    cfg["DATA"]["PATCH_SIZE"] = "(64, 64, 1)"
    plan_config(cfg, {}, [])
    assert cfg["DATA"]["PATCH_SIZE"] == "(64, 64, 1)"


###########################
# Instance representations #
###########################
def _proposal(tmp_path, kind):
    fp = _fp(_instances(tmp_path, kind), "DATA.TRAIN.GT_PATH", "INSTANCE_SEG", "2D")
    return fp, {r["id"]: r for r in fp["representations"]}


def test_representations_round(tmp_path):
    fp, reps = _proposal(tmp_path, "round")
    assert fp["morphology"]["star_convex_fraction"] == 1.0
    assert reps["Db_R"]["recommended"] and reps["F_P"]["available"] and reps["F_HV"]["available"]
    assert not reps["F_S"]["available"]
    assert sum(r["recommended"] for r in reps.values()) == 1
    assert fp["morphology_summary"]


def test_representations_rods(tmp_path):
    fp, reps = _proposal(tmp_path, "rods")
    assert fp["morphology"]["elongated_fraction"] == 1.0
    assert reps["F_S"]["recommended"]
    assert not reps["F_P"]["available"]


def test_representations_c_shapes(tmp_path):
    fp, reps = _proposal(tmp_path, "c_shapes")
    assert fp["morphology"]["star_convex_fraction"] == 0.0
    assert not reps["Db_R"]["available"] and not reps["F_P"]["available"]
    assert reps["F_C"]["available"]


def test_representations_affinities_dense_3d(tmp_path):
    fp = _fp(_dense_3d(tmp_path), "DATA.TRAIN.GT_PATH", "INSTANCE_SEG", "3D")
    reps = {r["id"]: r for r in fp["representations"]}
    assert reps["A"]["available"] and reps["A"]["recommended"]
    # Contours can't separate instances that fill the volume
    assert not reps["F_C"]["available"]
    assert instance_seg.representation_to_biapy("A", 3)[0] == ["A"]
    assert instance_seg.output_channels(["A"], 3, 2) == 3


def test_representations_dense_2d(tmp_path):
    # Dense instances in 2D: no contours nor affinities, and the recommended one is offered
    folder = tmp_path / "dense2d"
    folder.mkdir()
    from scipy import ndimage

    rng = np.random.default_rng(2)
    for i in range(3):
        seeds = np.zeros((128, 128), np.int32)
        for j, (y, x) in enumerate(rng.uniform((0, 0), (128, 128), (40, 2)).astype(int), start=1):
            seeds[y, x] = j
        _, idx = ndimage.distance_transform_edt(seeds == 0, return_indices=True)
        _write_2d(str(folder / "{}.tif".format(i)), seeds[tuple(idx)].astype(np.uint16))
    reps = {r["id"]: r for r in _fp(str(folder), "DATA.TRAIN.GT_PATH", "INSTANCE_SEG", "2D")["representations"]}
    assert not reps["F_C"]["available"] and not reps["A"]["available"]
    rec = [r for r in reps.values() if r["recommended"]]
    assert len(rec) == 1 and rec[0]["available"]


def test_representations_no_affinities_2d(tmp_path):
    _, reps = _proposal(tmp_path, "round")
    assert not reps["A"]["available"]


def test_representation_channels():
    assert instance_seg.representation_to_biapy("F_G", 3)[0] == ["F", "Gv", "Gh", "Gz"]
    assert instance_seg.representation_to_biapy("F_S", 2) == (["F", "P"], [{"P": {"type": "skeleton"}}])
    assert instance_seg.representation_of(("F", "P"), ({"P": {"type": "skeleton"}},), 2) == "F_S"
    assert instance_seg.output_channels(["Db", "R"], 2, 2) == 33


def test_plan_instance_seg(tmp_path):
    folder = _instances(tmp_path, "round")
    si = {
        "DATA.TRAIN.PATH": {"fingerprint": _fp(_raw_like(tmp_path, folder), "DATA.TRAIN.PATH", "INSTANCE_SEG", "2D")},
        "DATA.TRAIN.GT_PATH": {"fingerprint": _fp(folder, "DATA.TRAIN.GT_PATH", "INSTANCE_SEG", "2D")},
    }
    cfg = _cfg("2D")
    cfg["PROBLEM"]["TYPE"] = "INSTANCE_SEG"
    cfg["PROBLEM"]["INSTANCE_SEG"] = {"DATA_CHANNELS": ["Db", "R"], "DATA_CHANNELS_EXTRA_OPTS": [{}]}
    report = plan_config(cfg, si, GPU)
    assert report["applied"], report
    assert cfg["MODEL"]["ARCHITECTURE"] == "unet"
    assert any("StarDist" in line for line in report["lines"])


#############
# Detection #
#############
def test_detection_distances_2d(tmp_path):
    fp = _fp(_points(tmp_path, False), "DATA.TRAIN.GT_PATH", "DETECTION", "2D")
    dist = detection.detection_distances(fp, [1.0, 1.0])
    assert dist["min_sep"] == pytest.approx(20)
    assert dist["radius"] == 10 and dist["dilation"] == [5, 5]


def test_detection_distances_anisotropic_3d(tmp_path):
    # Points 5 slices apart in Z with Z voxels 4 times larger: 20 px apart in physical terms
    fp = _fp(_points(tmp_path, True), "DATA.TRAIN.GT_PATH", "DETECTION", "3D")
    dist = detection.detection_distances(fp, [4.0, 1.0, 1.0])
    assert dist["min_sep"] == pytest.approx(20)
    assert dist["dilation"] == [1, 5, 5]


###########
# Preview #
###########
def _pairs(tmp_path, raw_names, tgt_names, kind="semantic", tgt_shape=(64, 64)):
    from skimage.draw import disk

    rd, td = tmp_path / "raw", tmp_path / "tgt"
    rd.mkdir()
    td.mkdir()
    rng = np.random.default_rng(0)
    for rn, tn in zip(raw_names, tgt_names):
        _write_2d(str(rd / rn), rng.integers(0, 255, (64, 64), dtype=np.uint8))
        lab = np.zeros(tgt_shape, np.uint16)
        for j in range(12):
            r, c = 8 + 16 * (j // 4), 8 + 16 * (j % 4)
            lab[disk((r, c), 5, shape=lab.shape)] = 1 if kind == "semantic" else j + 1
        _write_2d(str(td / tn), lab)
    return str(rd), str(td)


def test_preview_ok(tmp_path):
    raw, tgt = _pairs(tmp_path, ["img_01.tif", "img_02.tif"], ["mask_001.tif", "mask_002.tif"])
    r = preview("SEMANTIC_SEG", "2D", raw, tgt)
    assert r["warnings"] == []
    assert len(r["samples"]) == 2 and "raw_png" in r["samples"][0] and "target_png" in r["samples"][0]


def test_preview_instance_masks_for_semantic(tmp_path):
    raw, tgt = _pairs(tmp_path, ["a.tif", "b.tif"], ["a.tif", "b.tif"], kind="instance")
    warnings = preview("SEMANTIC_SEG", "2D", raw, tgt)["warnings"]
    assert any("look like instance masks" in w for w in warnings)


def test_preview_semantic_masks_for_instance(tmp_path):
    raw, tgt = _pairs(tmp_path, ["a.tif", "b.tif"], ["a.tif", "b.tif"], kind="semantic")
    warnings = preview("INSTANCE_SEG", "2D", raw, tgt)["warnings"]
    assert any("semantic/binary masks" in w for w in warnings)


def test_preview_swapped_and_names(tmp_path):
    raw, tgt = _pairs(tmp_path, ["a1.tif", "a2.tif"], ["b7.tif", "b8.tif"])
    warnings = preview("SEMANTIC_SEG", "2D", tgt, raw)["warnings"]  # folders swapped
    assert any("swapped" in w for w in warnings)
    assert any("names differ" in w for w in warnings)


def test_preview_shape_mismatch(tmp_path):
    raw, tgt = _pairs(tmp_path, ["a.tif"], ["a.tif"], tgt_shape=(32, 64))
    warnings = preview("SEMANTIC_SEG", "2D", raw, tgt)["warnings"]
    assert any("should be the size of the raw image" in w for w in warnings)


def test_preview_super_resolution_targets_smaller(tmp_path):
    raw, tgt = _sr_pairs(tmp_path, "test", 64, 32, n=1)
    warnings = preview("SUPER_RESOLUTION", "2D", raw, tgt)["warnings"]
    assert any("smaller than its raw image (64x64)" in w and "swapped" in w for w in warnings)


def test_preview_detection_points_outside(tmp_path):
    raw, _ = _pairs(tmp_path, ["a.tif"], ["a.tif"])
    csv = tmp_path / "csv"
    csv.mkdir()
    (csv / "a.csv").write_text("axis-0,axis-1\n10,10\n70,5\n")
    r = preview("DETECTION", "2D", raw, str(csv))
    assert any("outside its image" in w for w in r["warnings"])
    assert "target_png" in r["samples"][0]


###################################
# Configuration from the answers  #
###################################
def _answers(workflow, ndim, checks, **extra):
    """Answers as a wizard collects them: the variables asked plus what the data checks gathered."""
    answers = {
        "PROBLEM.TYPE": workflow,
        "PROBLEM.NDIM": ndim,
        "TRAIN.ENABLE": True,
        "TEST.ENABLE": False,
        "MODEL.SOURCE": "biapy",
        "MODEL.LOAD_CHECKPOINT": False,
        "data_constraints": {},
        "sample_info": {},
    }
    for key, folder in checks:
        r = _check(folder, key, workflow, ndim)
        answers[key] = folder
        answers["data_constraints"].update(r["data_constraints"])
        answers["sample_info"][key] = r["sample_info"]
    answers.update(extra)
    return answers


def _biapy_check(cfg, tmp_path):
    """Loads the configuration as BiaPy does and runs its configuration check."""
    import copy

    import yaml
    from yacs.config import CfgNode as CN

    from biapy.config.config import Config, update_dependencies
    from biapy.engine.check_configuration import check_configuration, convert_old_model_cfg_to_current_version

    yaml_file = str(tmp_path / "job.yaml")
    write_config(cfg, {"applied": True, "lines": ["test"]}, yaml_file)
    with open(yaml_file) as f:
        assert f.readline().startswith("# Configuration adapted to the data")
        loaded = yaml.safe_load(f)
    # Written with the current variables: nothing for BiaPy to translate
    def no_empty(d):
        return {k: no_empty(v) if isinstance(v, dict) else v for k, v in d.items() if v != {}}

    assert no_empty(convert_old_model_cfg_to_current_version(copy.deepcopy(loaded))) == no_empty(loaded)
    manager = Config(str(tmp_path / "out"), "job")
    manager._C.merge_from_other_cfg(CN(loaded))
    update_dependencies(manager)
    check_configuration(manager.get_cfg_defaults(), "job", check_data_paths=False)


def test_config_semantic_seg(tmp_path):
    x, y = _semantic_3d(tmp_path)
    answers = _answers("SEMANTIC_SEG", "3D", [("DATA.TRAIN.PATH", x), ("DATA.TRAIN.GT_PATH", y)])
    cfg, report = build_config(answers, GPU)
    assert report["applied"], report
    assert cfg["MODEL"]["ARCHITECTURE"] == "unet" and cfg["DATA"]["N_CLASSES"] == 3
    assert cfg["TRAIN"]["SAVE_TRAIN_PREDS_NUM_SAMPLES"] == 2
    # nnU-Net's amount of training (1000 epochs of 250 iterations) whatever the size of the data, and the predictions
    # the front-ends show saved at most 100 times
    epochs = cfg["TRAIN"]["EPOCHS"]
    assert any(line.startswith("Complete training: {} epochs".format(epochs)) for line in report["lines"])
    iterations = int(next(l for l in report["lines"] if "training:" in l).split(" epochs of ")[1].split()[0])
    assert 0.9 * 250000 <= epochs * iterations <= 1.1 * 250000
    assert cfg["TRAIN"]["PATIENCE"] == 100
    assert cfg["TRAIN"]["SAVE_TRAIN_PREDS_FREQ"] == -(-epochs // 100)
    _biapy_check(cfg, tmp_path)


def test_config_instance_seg_representation(tmp_path):
    folder = _instances(tmp_path, "rods")
    raw = _raw_like(tmp_path, folder)
    answers = _answers(
        "INSTANCE_SEG", "2D", [("DATA.TRAIN.PATH", raw), ("DATA.TRAIN.GT_PATH", folder)], instance_representation="F_S"
    )
    cfg, report = build_config(answers, GPU)
    assert report["applied"], report
    inst = cfg["PROBLEM"]["INSTANCE_SEG"]
    assert tuple(inst["DATA_CHANNELS"]) == ("F", "P")
    assert inst["INSTANCE_CREATION_PROCESS"]
    _biapy_check(cfg, tmp_path)


@pytest.mark.parametrize(
    "representation, augmentor, weights",
    [
        ("F_G", {"AFFINE_MODE": "constant", "ZOOM_RANGE": "(0.75, 1.25)", "ZOOM_PROB": 1.0}, [1, 0.25, 0.25]),
        ("Db_R", {"ROT90": True, "RANDOM_ROT": False, "ZOOM": False}, [1, 0.2]),
        ("F_HV", {"AFFINE_MODE": "reflect", "ZOOM_RANGE": "(0.5, 1.5)", "ZOOM_PROB": 1.0}, [1, 0.25, 0.25]),
        ("F_C", {"AFFINE_MODE": "reflect", "ZOOM": True}, None),
    ],
)
def test_config_representation_settings(tmp_path, representation, augmentor, weights):
    """What the representation needs as in the original method (the sanity checks of the paper), valid for BiaPy."""
    folder = _instances(tmp_path, "round")
    raw = _raw_like(tmp_path, folder)
    answers = _answers(
        "INSTANCE_SEG", "2D", [("DATA.TRAIN.PATH", raw), ("DATA.TRAIN.GT_PATH", folder)],
        instance_representation=representation,
    )
    cfg, report = build_config(answers, GPU)
    assert report["applied"], report
    for key, value in augmentor.items():
        assert cfg["AUGMENTOR"][key] == value, key
    inst = cfg["PROBLEM"]["INSTANCE_SEG"]
    assert inst.get("DATA_CHANNEL_WEIGHTS") == weights
    assert any("as the original method" in line for line in report["lines"]) == (weights is not None)
    _biapy_check(cfg, tmp_path)


def test_representation_settings_bmz_cellpose(tmp_path):
    """A BioImage Model Zoo model predicting Cellpose's flows, fine-tuned: the same settings as the representation."""
    folder = _instances(tmp_path, "round")
    raw = _raw_like(tmp_path, folder)
    restrictions = {
        "PROBLEM.INSTANCE_SEG.DATA_CHANNELS": ["F", "Gh", "Gv"],
        "PROBLEM.INSTANCE_SEG.DATA_CHANNEL_WEIGHTS": [1, 1],
        "PROBLEM.INSTANCE_SEG.DATA_CHANNELS_LOSSES": [],
        "PROBLEM.INSTANCE_SEG.INSTANCE_CREATION_PROCESS": "",
        "PROBLEM.INSTANCE_SEG.DATA_CHANNELS_EXTRA_OPTS": [{}],
        "DATA.PATCH_SIZE": [256, 256, 1],
    }
    answers = _answers(
        "INSTANCE_SEG", "2D", [("DATA.TRAIN.PATH", raw), ("DATA.TRAIN.GT_PATH", folder)],
        model_restrictions=restrictions, **{"MODEL.SOURCE": "bmz", "MODEL.BMZ.SOURCE_MODEL_ID": "some-model"},
    )
    cfg, report = build_config(answers, GPU)
    assert cfg["AUGMENTOR"]["AFFINE_MODE"] == "constant" and cfg["AUGMENTOR"]["ZOOM_RANGE"] == "(0.75, 1.25)"
    inst = cfg["PROBLEM"]["INSTANCE_SEG"]
    assert inst["DATA_CHANNEL_WEIGHTS"] == [1, 0.25, 0.25]
    assert inst["CLASS_REBALANCE_WITHIN_CHANNELS"] is False
    assert any(line.startswith("Cellpose flows") for line in report["lines"])


@pytest.mark.parametrize("channels,forced", [(["F", "Gv", "Gh"], True), (["F", "H", "V"], None)])
def test_representation_settings_force_padding(channels, forced):
    """Inference: Cellpose's flows pad also the images of the patch size (DATA.TEST.FORCE_PADDING), the rest don't."""
    cfg = _cfg("2D")
    cfg["PROBLEM"]["TYPE"] = "INSTANCE_SEG"
    cfg["PROBLEM"]["INSTANCE_SEG"] = {"DATA_CHANNELS": channels, "DATA_CHANNELS_EXTRA_OPTS": [{}]}
    cfg["TRAIN"]["ENABLE"] = False
    cfg["MODEL"]["LOAD_CHECKPOINT"] = True
    plan_config(cfg, {}, [])
    assert cfg.get("DATA", {}).get("TEST", {}).get("FORCE_PADDING") == forced


def test_representation_settings_omnipose_checkpoint(tmp_path):
    """Omnipose's flows (from a BiaPy checkpoint) are filled by reflection, unlike Cellpose's."""
    torch = pytest.importorskip("torch")
    ckpt = str(tmp_path / "model-checkpoint-best.pth")
    extra = [{"Db": {"val_type": "omnipose"}, "Gh": {"gradient_type": "omnipose"}}]
    inst = {"DATA_CHANNELS": ["Db", "Gh", "Gv"], "DATA_CHANNELS_EXTRA_OPTS": extra}
    torch.save({"cfg": {"PROBLEM": {"TYPE": "INSTANCE_SEG", "INSTANCE_SEG": inst}}}, ckpt)
    cfg = _cfg("2D")
    cfg["PROBLEM"]["TYPE"] = "INSTANCE_SEG"
    cfg["MODEL"]["LOAD_CHECKPOINT"] = True
    cfg["PATHS"] = {"CHECKPOINT_FILE": ckpt}
    cfg["DATA"]["PATCH_SIZE"] = "(256, 256, 1)"
    report = plan_config(cfg, {}, [])
    assert cfg["AUGMENTOR"]["AFFINE_MODE"] == "reflect" and cfg["AUGMENTOR"]["ZOOM_RANGE"] == "(0.5, 1.5)"
    assert cfg["PROBLEM"]["INSTANCE_SEG"]["DATA_CHANNELS_LOSSES"] == ["mse"] * 3
    assert any(line.startswith("Omnipose flows") for line in report["lines"])


def test_representation_settings_hovernet_3d_inference(tmp_path):
    """Inference only: HoVer-Net's watershed thresholds, with the Z distances in 3D, and no training settings."""
    cfg = _cfg("3D")
    cfg["PROBLEM"]["TYPE"] = "INSTANCE_SEG"
    cfg["PROBLEM"]["INSTANCE_SEG"] = {"DATA_CHANNELS": ["F", "H", "V", "Z"], "DATA_CHANNELS_EXTRA_OPTS": [{}]}
    cfg["TRAIN"]["ENABLE"] = False
    cfg["MODEL"]["LOAD_CHECKPOINT"] = True
    report = plan_config(cfg, {}, [])
    ws = cfg["PROBLEM"]["INSTANCE_SEG"]["WATERSHED"]
    assert ws == {"SEED_CHANNELS_THRESH": [0.5, 0.4, 0.4, 0.4], "GROWTH_MASK_CHANNELS_THRESH": [0.5]}
    assert "AUGMENTOR" not in cfg and "DATA_CHANNEL_WEIGHTS" not in cfg["PROBLEM"]["INSTANCE_SEG"]
    assert any(line.startswith("HoVer-Net") for line in report["lines"])
    # BiaPy sets the seed channels (F first) and keeps the thresholds, without warning that they are custom
    import warnings

    from biapy.config.config import Config
    from biapy.engine.check_configuration import check_configuration
    from yacs.config import CfgNode as CN

    manager = Config(str(tmp_path / "out"), "job")
    manager._C.merge_from_other_cfg(CN(tuples_to_lists(cfg)))
    biapy_cfg = manager.get_cfg_defaults()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        check_configuration(biapy_cfg, "job", check_data_paths=False)
    ws = biapy_cfg.PROBLEM.INSTANCE_SEG.WATERSHED
    assert list(ws.SEED_CHANNELS) == ["F", "Z", "V", "H"] and list(ws.SEED_CHANNELS_THRESH) == [0.5, 0.4, 0.4, 0.4]
    assert list(ws.GROWTH_MASK_CHANNELS) == ["F"] and list(ws.GROWTH_MASK_CHANNELS_THRESH) == [0.5]


def test_representation_family():
    family = instance_seg.representation_family
    assert family(("F", "Gv", "Gh", "Gz"), ({},)) == "cellpose"
    assert family(["Gv", "Gh", "F", "I"], None) == "cellpose"
    assert family(["Db", "Gv", "Gh"], [{"Gv": {"gradient_type": "omnipose"}}]) == "omnipose"
    assert family(["Db", "R"], [{}]) == "stardist"
    assert family(["E_offset", "E_sigma", "E_seediness"], [{}]) == "embedseg"
    assert family(["F", "H", "V"], [{}]) == "hovernet"
    assert family(["Db", "H", "V"], [{}]) is None  # HoVer-Net's thresholds are for its foreground
    assert family(["F", "C"], [{}]) is None and family("BC", None) is None


def test_config_detection(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for i in range(3):
        _write_2d(str(raw / "{}.tif".format(i)), np.zeros((200, 200), np.uint8))
    answers = _answers("DETECTION", "2D", [("DATA.TRAIN.PATH", str(raw)), ("DATA.TRAIN.GT_PATH", _points(tmp_path, False))])
    cfg, report = build_config(answers, GPU)
    assert report["applied"], report
    _biapy_check(cfg, tmp_path)


def test_config_renamed_variables(tmp_path):
    # Variables of the questions that BiaPy renamed are translated
    x, y = _semantic_3d(tmp_path)
    answers = _answers(
        "SEMANTIC_SEG", "3D", [("DATA.TRAIN.PATH", x), ("DATA.TRAIN.GT_PATH", y)], **{"MODEL.LOAD_MODEL_FROM_CHECKPOINT": False}
    )
    cfg, _ = build_config(answers, GPU)
    assert "LOAD_MODEL_FROM_CHECKPOINT" not in cfg["MODEL"]
    _biapy_check(cfg, tmp_path)


def test_config_elongated_images(tmp_path):
    # Very elongated images: X is downsampled alone in the deepest levels ([1, 2] pairs in MODEL.YX_DOWN)
    rng = np.random.default_rng(0)
    xdir, ydir = tmp_path / "x", tmp_path / "y"
    xdir.mkdir()
    ydir.mkdir()
    for i in range(20):
        _write_2d(str(xdir / "{}.tif".format(i)), rng.integers(0, 255, (64, 640), dtype=np.uint8))
        mask = np.zeros((64, 640), np.uint8)
        mask[20:40, 100:300] = 1
        _write_2d(str(ydir / "{}.tif".format(i)), mask)
    answers = _answers("SEMANTIC_SEG", "2D", [("DATA.TRAIN.PATH", str(xdir)), ("DATA.TRAIN.GT_PATH", str(ydir))])
    cfg, report = build_config(answers, GPU)
    assert report["applied"], report
    assert [1, 2] in [list(v) if isinstance(v, (list, tuple)) else v for v in cfg["MODEL"]["YX_DOWN"]]
    _biapy_check(cfg, tmp_path)


def test_config_mismatched_folders(tmp_path):
    x, y = _semantic_3d(tmp_path)
    answers = _answers("SEMANTIC_SEG", "3D", [("DATA.TRAIN.PATH", x), ("DATA.TRAIN.GT_PATH", y)])
    answers["data_constraints"]["DATA.TRAIN.GT_PATH"] = 5
    with pytest.raises(WizardError, match="do not match"):
        build_config(answers, GPU)


def _sr_pairs(tmp_path, name, raw_size, target_size, n=2):
    raw, target = tmp_path / (name + "_raw"), tmp_path / (name + "_target")
    raw.mkdir()
    target.mkdir()
    rng = np.random.default_rng(0)
    for i in range(n):
        _write_2d(str(raw / "{}.tif".format(i)), rng.integers(0, 255, (raw_size, raw_size), dtype=np.uint8))
        _write_2d(str(target / "{}.tif".format(i)), rng.integers(0, 255, (target_size, target_size), dtype=np.uint8))
    return str(raw), str(target)


def _sr_answers(tmp_path, train, test):
    checks = [("DATA.TRAIN.PATH", train[0]), ("DATA.TRAIN.GT_PATH", train[1])]
    if test:
        checks += [("DATA.TEST.PATH", test[0]), ("DATA.TEST.GT_PATH", test[1])]
    return _answers("SUPER_RESOLUTION", "2D", checks, **{"TEST.ENABLE": bool(test), "DATA.TEST.LOAD_GT": bool(test)})


def test_config_super_resolution(tmp_path):
    train, test = _sr_pairs(tmp_path, "train", 64, 128), _sr_pairs(tmp_path, "test", 64, 128)
    cfg, _ = build_config(_sr_answers(tmp_path, train, test), GPU)
    assert cfg["PROBLEM"]["SUPER_RESOLUTION"]["UPSCALING"] == "(2, 2)"


@pytest.mark.parametrize(
    "test_sizes,message",
    [
        # High-resolution test images as raw images: their targets look smaller
        ((128, 64), r"test targets are smaller than their raw images:\n    - raw image 0.tif: 128x128"),
        ((64, 96), r"target is x1.5 larger"),
        # High-resolution images in both test folders: x1 instead of the training's x2
        ((128, 128), r"It is x2 in the training data \(e.g. raw image 0.tif: 64x64, its target: 128x128\), but x1"),
    ],
)
def test_config_super_resolution_wrong_test_data(tmp_path, test_sizes, message):
    train, test = _sr_pairs(tmp_path, "train", 64, 128), _sr_pairs(tmp_path, "test", *test_sizes)
    with pytest.raises(WizardError, match=message):
        build_config(_sr_answers(tmp_path, train, test), GPU)


def test_bmz_blacklist():
    from biapy.models import bmz_model_blacklist_reason, check_bmz_model_compatibility

    # As BMZ lists wild-whale (only tagged "segmentation"): excluded from semantic segmentation
    manifest = {
        "id": "10.5281/zenodo.6079314/7695872",
        "config": {"bioimageio": {"nickname": "wild-whale"}},
        "tags": ["segmentation", "unet"],
        "weights": {"pytorch_state_dict": {"kwargs": {"out_channels": 8}}},
        "inputs": [{"axes": "bcyx", "shape": {"min": [1, 1, 64, 64]}}],
    }
    listed = {"nickname": "wild-whale", "alias": "wild-whale", "raw": {"manifest": manifest}}
    for model in (listed, manifest):  # as find_bmz_models() lists it and its RDF alone
        assert "wild-whale" in bmz_model_blacklist_reason(model, "SEMANTIC_SEG")
        assert bmz_model_blacklist_reason(model, "INSTANCE_SEG") is None
        r = check_bmz_model_compatibility(model, {"workflow_type": "SEMANTIC_SEG", "ndim": "2D", "nclasses": "all"})
        assert r["error"] and "excluded" in r["reason_message"]
    manifest["config"]["bioimageio"]["nickname"] = "other-model"
    r = check_bmz_model_compatibility(manifest, {"workflow_type": "SEMANTIC_SEG", "ndim": "2D", "nclasses": "all"})
    assert not r["error"], r["reason_message"]
