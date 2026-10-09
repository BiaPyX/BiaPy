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
from biapy.wizard.config_builder import WizardError, build_config, write_config
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
    # Scarce foreground and enough images: patches sampled around it
    assert cfg["DATA"]["TRAIN"]["PROBABILITY_MAP"] is True
    assert any("labels [2]" in line for line in report["lines"])


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
    assert instance_seg.representation_to_biapy("A", 3)[0] == ["A"]
    assert instance_seg.output_channels(["A"], 3, 2) == 3


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
