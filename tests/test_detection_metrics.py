import warnings

import pytest

from biapy.engine.metrics import detection_metrics


def test_detection_metrics_maximizes_valid_matches_within_tolerance():
    """Valid matches within tolerance must not be sacrificed for shorter distances."""
    # Both pairs (pred[0]->gt[0] and pred[1]->gt[1]) are at distance 9.5 < 10.
    # Assigning pred[0]->gt[1] (distance 0.5) forces pred[1]->gt[0] (distance ~13.8 >= 10),
    # yielding only 1 TP instead of 2.
    true_points = [[0, 0, 0], [0, 10, 0]]
    pred_points = [[0, 9.5, 0], [0, 10, 9.5]]

    metrics, df, df_fp = detection_metrics(true_points, pred_points, tolerance=10)

    assert metrics["TP"] == 2
    assert metrics["FP"] == 0
    assert metrics["FN"] == 0
    assert metrics["Precision"] == pytest.approx(1.0)
    assert metrics["Recall"] == pytest.approx(1.0)
    assert metrics["F1"] == pytest.approx(1.0)
    assert list(df["tag"]) == ["TP", "TP"]
    assert len(df_fp) == 0


def test_detection_metrics_maximizes_matches_with_large_tolerance():
    """Matching remains cardinality-optimal at larger physical tolerances."""
    true_points = [[0, 0, 0], [0, 20, 0]]
    pred_points = [[0, 18, 0], [0, 20, 18]]

    metrics, _, _ = detection_metrics(true_points, pred_points, tolerance=20)

    assert metrics["TP"] == 2
    assert metrics["FP"] == 0
    assert metrics["FN"] == 0
    assert metrics["F1"] == pytest.approx(1.0)


def test_detection_metrics_unmatched_distant_gt_does_not_flip_closest_tp():
    """Distances of invalid (>= tolerance) pairs must not affect which valid match is chosen."""
    # Only gt[0]=[0, 0, 0] has valid predictions within tolerance=10:
    # pred[0]=[0, 4, 0] (distance 4) and pred[1]=[0, -6, 0] (distance 6).
    # gt[1]=[0, 50, 0] is out of tolerance for both predictions and must not bias the match.
    true_points = [[0, 0, 0], [0, 50, 0]]
    pred_points = [[0, 4, 0], [0, -6, 0]]

    metrics, df, df_fp = detection_metrics(
        true_points,
        pred_points,
        true_classes=[1, 2],
        pred_classes=[1, 2],
        tolerance=10,
    )

    assert metrics["TP"] == 1
    assert metrics["FP"] == 1
    assert metrics["FN"] == 1
    assert df.loc[0, "pred_id"] == 1
    assert df.loc[0, "distance"] == pytest.approx(4.0)
    assert metrics["TP (class)"] == 1
    assert metrics["FN (class)"] == 0
    assert list(df_fp["pred_id"]) == [2]


def test_detection_metrics_empty_predictions_no_divide_warning():
    """Empty predictions with non-empty ground truth produce zero TP without warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        metrics, df, df_fp = detection_metrics([[0, 0, 0]], [], tolerance=10)

    assert metrics["TP"] == 0
    assert metrics["FP"] == 0
    assert metrics["FN"] == 1
    assert len(df) == 1
    assert len(df_fp) == 0
    assert not any("divide" in str(w.message).lower() for w in caught)
