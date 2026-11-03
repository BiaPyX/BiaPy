"""
Tests of the data checks shared by BiaPy's loaders and the wizard (biapy.data): listing a data folder, the consistency
of its images, raw/target shapes, the instance output channels and the frames of a time-lapse.

    pytest tests/test_data_checks.py -q
"""
import numpy as np
import pytest
import tifffile

from biapy.data.data_manipulation import (
    check_image_consistency,
    list_data_files,
    mismatched_raw_target_files,
    raw_target_mismatch_message,
    upscaling_of,
)
from biapy.data.pre_processing import instance_channel_names
from biapy.data.tracks_io import frames_problem


def test_list_data_files(tmp_path):
    tifffile.imwrite(str(tmp_path / "b.tif"), np.zeros((4, 4), np.uint8))
    tifffile.imwrite(str(tmp_path / "a.tif"), np.zeros((4, 4), np.uint8))
    (tmp_path / "sub.zarr").mkdir()
    assert list_data_files(str(tmp_path)) == (["a.tif", "b.tif"], False)
    zarrs = tmp_path / "zarrs"
    (zarrs / "x.zarr").mkdir(parents=True)
    assert list_data_files(str(zarrs)) == (["x.zarr"], True)
    with pytest.raises(ValueError, match="No images"):
        list_data_files(str(zarrs / "x.zarr"))
    with pytest.raises(ValueError, match="does not exist"):
        list_data_files(str(tmp_path / "nope"))


def test_check_image_consistency():
    first = check_image_consistency("a", 1, drange=("uint8 range", np.uint8))
    assert check_image_consistency("b", 1, first, patch_channels=1, drange=("uint8 range", np.uint8)) == first
    with pytest.raises(ValueError, match="this image has 3 channels and the first image of the folder 1"):
        check_image_consistency("c", 3, first)
    with pytest.raises(ValueError, match="patch size given 3"):
        check_image_consistency("d", 1, first, patch_channels=3)
    # Same integer dtype: consistent unless the normalization depends on the data range
    check_image_consistency("e", 1, first, drange=("01 range", np.uint8), strict=False)
    with pytest.raises(ValueError, match="data range"):
        check_image_consistency("f", 1, first, drange=("01 range", np.uint8))


def test_raw_target_shapes():
    raw = [("a", (32, 32)), ("b", (32, 32)), ("c", None)]
    assert mismatched_raw_target_files(raw, [("a", (64, 64)), ("b", (64, 64)), ("c", (1, 1))], (2, 2)) == []
    bad = mismatched_raw_target_files(raw, [("a", (64, 64)), ("b", (96, 96)), ("c", None)], (2, 2))
    assert bad == [("b", "b", (32, 32), (96, 96), (64, 64))]
    assert upscaling_of((32, 32), (96, 64)) == (3, 2)
    assert upscaling_of((32, 32), (48, 64)) is None and upscaling_of((32, 32), (16, 32)) is None

    msg = raw_target_mismatch_message(raw[:2], [("a", (96, 96)), ("b", (96, 96))], "x", "y", (2, 2),
                                      super_resolution=True, phase="TEST")
    assert "test pairs do not match" in msg and "set PROBLEM.SUPER_RESOLUTION.UPSCALING to (3, 3)" in msg
    msg = raw_target_mismatch_message(raw[:2], [("a", (16, 16)), ("b", (16, 16))], "x", "y", (2, 2), super_resolution=True)
    assert "swapped" in msg
    assert "There are 2 training raw images but 1 masks" in raw_target_mismatch_message(
        raw[:2], [("a", (32, 32))], "x", "y", (1, 1), is_y_mask=True
    )


def test_instance_channel_names():
    assert instance_channel_names(["F", "C", "I", "We"], {}, 2) == [("F", "F"), ("C", "C")]
    names = instance_channel_names(["Db", "R"], {"Db": {"val_type": "discretize"}, "R": {"nrays": 4}}, 2)
    assert [n for _, n in names] == ["Db_bin{}".format(i) for i in range(11)] + ["R_0", "R_1", "R_2", "R_3"]
    assert len(instance_channel_names(["E_offset", "E_sigma", "E_seediness"], {}, 3)) == 7
    assert [n for _, n in instance_channel_names(["A"], {}, 2)] == ["Az_1", "Ay_1", "Ax_1"]


def test_frames_problem():
    assert "only 1" in frames_problem(["t0.tif"])
    assert frames_problem(["t0", "t1"]) is None
    assert frames_problem(["t0", "t1"], [(8, 8, 1), (8, 8, 1)]) is None
    assert "single-channel" in frames_problem(["t0", "t1"], [(8, 8, 3), (8, 8, 3)])
    assert "same size" in frames_problem(["t0", "t1"], [(8, 8, 1), (8, 9, 1)])


def test_multiple_raw_images_need_external_validation(tmp_path):
    from biapy.config.config import Config
    from biapy.engine.check_configuration import check_configuration

    cfg = Config(str(tmp_path), "job").get_cfg_defaults()
    cfg.merge_from_list(["PROBLEM.TYPE", "IMAGE_TO_IMAGE", "PROBLEM.IMAGE_TO_IMAGE.MULTIPLE_RAW_ONE_TARGET_LOADER", True,
                         "DATA.PATCH_SIZE", (64, 64, 1), "TRAIN.ENABLE", True, "TEST.ENABLE", False])
    with pytest.raises(ValueError, match="DATA.VAL.FROM_TRAIN"):
        check_configuration(cfg, "job", check_data_paths=False)
    cfg.merge_from_list(["DATA.VAL.FROM_TRAIN", False])
    check_configuration(cfg, "job", check_data_paths=False)
