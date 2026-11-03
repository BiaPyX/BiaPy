"""Tracking: CTC conversion of the links, tracking metrics and GEFF output (synthetic data, CPU)."""

import numpy as np
import pytest
import zarr

from biapy.data.post_processing.tracking import greedy_linking, links_to_ctc
from biapy.data.tracks_io import write_geff
from biapy.engine.metrics import tracking_metrics

# Ground truth: cell 1 (frames 0-1) divides into 2 and 3 (frames 2-3); cell 4 lives in frames 0-3.
# 10 nodes, 8 edges (6 within tracks + 2 parent-daughter).
GT_TRACKS = np.array([[1, 0, 1, 0], [2, 2, 3, 1], [3, 2, 3, 1], [4, 0, 3, 0]])


def _gt_masks():
    m = np.zeros((4, 20, 20), dtype=np.uint16)
    m[:2, 2:6, 2:6] = 1
    m[2:, 2:6, 2:6] = 2
    m[2:, 10:14, 2:6] = 3
    m[:, 10:14, 12:16] = 4
    return m


def test_metrics_perfect():
    gt = _gt_masks()
    relabel = np.array([0, 7, 5, 6, 9], dtype=np.uint16)
    pred_tracks = np.array([[7, 0, 1, 0], [5, 2, 3, 7], [6, 2, 3, 7], [9, 0, 3, 0]])
    m = tracking_metrics(gt, GT_TRACKS, relabel[gt], pred_tracks)
    assert m["DET"] == m["TRA"] == m["link_f1"] == m["division_f1"] == 1
    assert m["NS"] == m["FN"] == m["FP"] == m["ED"] == m["EA"] == m["EC"] == 0


def test_metrics_missing_cell_and_division():
    gt = _gt_masks()
    pred = gt.copy()
    pred[3][pred[3] == 4] = 0
    pred_tracks = np.array([[1, 0, 1, 0], [2, 2, 3, 0], [3, 2, 3, 0], [4, 0, 2, 0]])
    m = tracking_metrics(gt, GT_TRACKS, pred, pred_tracks)
    assert (m["FN"], m["FP"], m["NS"], m["ED"], m["EA"], m["EC"]) == (1, 0, 0, 0, 3, 0)
    assert m["DET"] == pytest.approx(1 - 10 / 100)
    assert m["TRA"] == pytest.approx(1 - (10 + 1.5 * 3) / (100 + 1.5 * 8))
    assert (m["link_precision"], m["link_recall"]) == (1, 5 / 8)
    assert m["division_recall"] == m["division_f1"] == 0


def test_metrics_merged_cells():
    gt = _gt_masks()
    pred = gt.copy()
    pred[3, 2:14, 2:6] = 2
    pred_tracks = np.array([[1, 0, 1, 0], [2, 2, 3, 1], [3, 2, 2, 1], [4, 0, 3, 0]])
    m = tracking_metrics(gt, GT_TRACKS, pred, pred_tracks)
    assert (m["NS"], m["FN"], m["FP"]) == (1, 0, 0)


def test_links_to_ctc_division():
    timepoints = np.array([0, 1, 1])
    coords = np.array([[5.0, 5.0], [4.0, 5.0], [6.0, 5.0]])
    scores = {(0, 1): np.float32(0.9), (0, 2): np.float32(0.8)}

    links = greedy_linking(scores, timepoints, coords, allow_divisions=True)
    tracks, label_maps = links_to_ctc(links, timepoints, np.array([3, 1, 2]), 2)
    assert links == [(0, 1), (0, 2)]
    assert tracks.tolist() == [[1, 0, 0, 0], [2, 1, 1, 1], [3, 1, 1, 1]]
    assert label_maps == [{3: 1}, {1: 2, 2: 3}]

    links = greedy_linking(scores, timepoints, coords, allow_divisions=False)
    tracks, _ = links_to_ctc(links, timepoints, np.array([3, 1, 2]), 2)
    assert tracks.tolist() == [[1, 0, 1, 0]]


def test_write_geff(tmp_path):
    path = str(tmp_path / "tracks.geff")
    links = [(0, 1), (0, 2)]
    write_geff(path, links, np.array([0, 1, 1]), np.array([[5.0, 5.0], [4.0, 5.0], [6.0, 5.0]]), {0: 1, 1: 2, 2: 3})

    group = zarr.open_group(path, mode="r")
    assert group.attrs["geff"]["track_node_props"] == {"tracklet": "track_id"}
    assert group["nodes/ids"][:].tolist() == [0, 1, 2]
    assert group["nodes/props/t/values"][:].tolist() == [0, 1, 1]
    assert group["nodes/props/track_id/values"][:].tolist() == [1, 2, 3]
    assert group["edges/ids"][:].tolist() == [[0, 1], [0, 2]]

    geff = pytest.importorskip("geff")
    from geff.core_io._base_read import read_to_memory
    from geff.validate.data import ValidationConfig

    geff.validate_structure(path)
    read_to_memory(path, data_validation=ValidationConfig(graph=True, tracklet=True, lineage=False))
