"""
This module provides functions for evaluating detection and instance segmentation metrics, primarily based on object matching between ground truth and predicted label images. The core logic is adapted from the StarDist project.

It includes utilities for:
- Checking and manipulating label arrays.
- Calculating overlap matrices between label images.
- Defining various matching criteria (e.g., Intersection over Union - IoU).
- Computing standard metrics such as True Positives (TP), False Positives (FP),
  False Negatives (FN), Precision, Recall, Accuracy, F1-score, and Panoptic Quality.
- Aggregating matching statistics across multiple images.

The code is modified from the StarDist project, and the original license
(BSD 3-Clause License) is included below.

Code modified from https://github.com/stardist/stardist

BSD 3-Clause License

Copyright (c) 2018-2023, Uwe Schmidt, Martin Weigert
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np

from scipy.optimize import linear_sum_assignment
from collections import namedtuple
from skimage.segmentation import relabel_sequential

from typing import (
    Tuple,
    Literal,
    Dict,
)

# Dictionary to store different matching criteria functions
matching_criteria = dict()

# Copied from https://github.com/CSBDeep/CSBDeep/blob/master/csbdeep/utils/utils.py
def _raise(e):
    """
    Raise an exception.

    Parameters
    ----------
    e : Exception or str
        The exception instance to raise, or a string message to raise as a ValueError.

    Raises
    ------
    Exception
        The provided exception instance.
    ValueError
        If `e` is a string.
    """
    if isinstance(e, BaseException):
        raise e
    else:
        raise ValueError(e)


def label_are_sequential(y):
    """
    Check if a label array contains sequential labels starting from 1.

    Labels are considered sequential if, excluding the background label (0),
    they form a continuous sequence from 1 up to the maximum label value.

    Parameters
    ----------
    y : np.ndarray
        The input label image (integer-valued).

    Returns
    -------
    bool
        True if labels are sequential (1, 2, ..., max_label), False otherwise.
    """
    labels = np.unique(y)
    return (set(labels) - {0}) == set(range(1, 1 + labels.max()))


def is_array_of_integers(y):
    """
    Check if the input is a NumPy array and its data type is an integer type.

    Parameters
    ----------
    y : np.ndarray
        The array to check.

    Returns
    -------
    bool
        True if `y` is a NumPy array with an integer dtype, False otherwise.
    """
    return isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.integer)


def _check_label_array(y, name=None, check_sequential=False):
    """
    Validate a label array.

    Ensures that the input `y` is a NumPy array of non-negative integers.
    Optionally checks if the labels are sequential (starting from 1 for non-zero labels).

    Parameters
    ----------
    y : np.ndarray
        The label array to validate.
    name : Optional[str], optional
        A descriptive name for the label array, used in error messages. Defaults to None.
    check_sequential : bool, optional
        If True, also checks if the labels are sequential (1, 2, ..., max_label).
        Defaults to False.

    Returns
    -------
    bool
        True if the array passes all checks.

    Raises
    ------
    ValueError
        If the array is not a NumPy array of integers, contains negative values,
        or if `check_sequential` is True and labels are not sequential.
    """
    err = ValueError(
        "{label} must be an array of {integers}.".format(
            label="labels" if name is None else name,
            integers=("sequential " if check_sequential else "") + "non-negative integers",
        )
    )
    is_array_of_integers(y) or _raise(err)
    if len(y) == 0:
        return True
    if check_sequential:
        label_are_sequential(y) or _raise(err)
    else:
        y.min() >= 0 or _raise(err)
    return True


def label_overlap(x, y, check=True):
    """
    Compute the overlap matrix between two label images.

    The overlap matrix `O` has dimensions `(1+max(x_label), 1+max(y_label))`,
    where `O[i, j]` is the number of pixels where label `i` in `x` overlaps
    with label `j` in `y`. Background label (0) is included.

    Parameters
    ----------
    x : np.ndarray
        The first label image (integer-valued).
    y : np.ndarray
        The second label image (integer-valued).
    check : bool, optional
        If True, performs rigorous checks on input arrays (shape, type, sequential labels).
        Defaults to True.

    Returns
    -------
    np.ndarray
        The overlap matrix, where `overlap[i,j]` is the number of pixels where
        label `i` in `x` and label `j` in `y` overlap.

    Raises
    ------
    ValueError
        If `check` is True and inputs are invalid (e.g., different shapes, non-integer labels).
    """
    if check:
        _check_label_array(x, "x", True)
        _check_label_array(y, "y", True)
        x.shape == y.shape or _raise(ValueError("x and y must have the same shape"))
    return _label_overlap(x, y)


def _label_overlap(x, y):
    """
    Compute the overlap matrix without input checks.

    Parameters
    ----------
    x : np.ndarray
        The first label image (integer-valued).
    y : np.ndarray
        The second label image (integer-valued).

    Returns
    -------
    np.ndarray
        The overlap matrix.
    """
    x = x.ravel()
    y = y.ravel()
    overlap = np.zeros((1 + x.max(), 1 + y.max()), dtype=np.uint)
    for i in range(len(x)):
        overlap[x[i], y[i]] += 1
    return overlap


def _safe_divide(x, y, eps=1e-10):
    """
    Compute a safe division, returning 0 if the denominator is close to zero.

    Handles both scalar and array inputs.

    Parameters
    ----------
    x : np.ndarray or float
        Numerator.
    y : np.ndarray or float
        Denominator.
    eps : float, optional
        Small epsilon value to check if denominator is close to zero. Defaults to 1e-10.

    Returns
    -------
    np.ndarray or float
        Result of the division. Returns 0 (or array of zeros) where `y` is close to zero.
    """
    if np.isscalar(x) and np.isscalar(y):
        return x / y if np.abs(y) > eps else 0.0
    else:
        out = np.zeros(np.broadcast(x, y).shape, np.float32)
        np.divide(x, y, out=out, where=np.abs(y) > eps)
        return out


def intersection_over_union(overlap):
    """
    Calculate the Intersection over Union (IoU) matrix from an overlap matrix.

    IoU for two objects is defined as `(intersection) / (union)`.
    `union = n_pixels_pred + n_pixels_true - overlap`.

    Parameters
    ----------
    overlap : np.ndarray
        The overlap matrix, typically generated by `label_overlap`.

    Returns
    -------
    np.ndarray
        The IoU matrix, where `iou[i,j]` is the IoU between object `i` in true
        and object `j` in prediction. Returns an empty matrix if total overlap is 0.
    """
    _check_label_array(overlap, "overlap")
    if np.sum(overlap) == 0:
        return overlap
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    return _safe_divide(overlap, (n_pixels_pred + n_pixels_true - overlap))

# Register IoU as a matching criterion
matching_criteria["iou"] = intersection_over_union

def intersection_over_true(overlap):
    """
    Calculate the Intersection over True (IoT) matrix from an overlap matrix.

    IoT for two objects is defined as `(intersection) / (number of pixels in true object)`.

    Parameters
    ----------
    overlap : np.ndarray
        The overlap matrix, typically generated by `label_overlap`.

    Returns
    -------
    np.ndarray
        The IoT matrix. Returns an empty matrix if total overlap is 0.
    """
    _check_label_array(overlap, "overlap")
    if np.sum(overlap) == 0:
        return overlap
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    return _safe_divide(overlap, n_pixels_true)

# Register IoT as a matching criterion
matching_criteria["iot"] = intersection_over_true


def intersection_over_pred(overlap):
    """
    Calculate the Intersection over Pred (IoP) matrix from an overlap matrix.

    IoP for two objects is defined as `(intersection) / (number of pixels in predicted object)`.

    Parameters
    ----------
    overlap : np.ndarray
        The overlap matrix, typically generated by `label_overlap`.

    Returns
    -------
    np.ndarray
        The IoP matrix. Returns an empty matrix if total overlap is 0.
    """
    _check_label_array(overlap, "overlap")
    if np.sum(overlap) == 0:
        return overlap
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    return _safe_divide(overlap, n_pixels_pred)

# Register IoP as a matching criterion
matching_criteria["iop"] = intersection_over_pred


def precision(tp, fp, fn):
    """
    Calculate precision.

    Precision = `TP / (TP + FP)`.

    Parameters
    ----------
    tp : int
        Number of true positives.
    fp : int
        Number of false positives.
    fn : int
        Number of false negatives. (Not used in precision calculation but included for consistency).

    Returns
    -------
    float
        The precision score. Returns 0 if `tp` is 0.
    """
    return tp / (tp + fp) if tp > 0 else 0


def recall(tp, fp, fn):
    """
    Calculate recall.

    Recall = `TP / (TP + FN)`.

    Parameters
    ----------
    tp : int
        Number of true positives.
    fp : int
        Number of false positives. (Not used in recall calculation but included for consistency).
    fn : int
        Number of false negatives.

    Returns
    -------
    float
        The recall score. Returns 0 if `tp` is 0.
    """
    return tp / (tp + fn) if tp > 0 else 0


def accuracy(tp, fp, fn):
    """
    Calculate the accuracy score.

    This metric is also known as "average precision" in some contexts (e.g., Data Science Bowl 2018).
    Accuracy is defined as `TP / (TP + FP + FN)`.

    Parameters
    ----------
    tp : int
        Number of true positives.
    fp : int
        Number of false positives.
    fn : int
        Number of false negatives.

    Returns
    -------
    float
        The accuracy score. Returns 0 if `tp` is 0.
    """
    return tp / (tp + fp + fn) if tp > 0 else 0


def f1(tp, fp, fn):
    """
    Calculate the F1-score (also known as "Dice coefficient").

    F1-score = `(2 * TP) / (2 * TP + FP + FN)`.

    Parameters
    ----------
    tp : int
        Number of true positives.
    fp : int
        Number of false positives.
    fn : int
        Number of false negatives.

    Returns
    -------
    float
        The F1-score. Returns 0 if `tp` is 0.
    """
    return (2 * tp) / (2 * tp + fp + fn) if tp > 0 else 0


def matching(y_true, y_pred, thresh=0.5, criterion="iou", report_matches=False) -> Dict | Tuple[Dict]:
    """
    Calculate detection/instance segmentation metrics between ground truth and predicted label images.

    This function performs an optimal (Hungarian algorithm-based) matching
    between objects in `y_true` and `y_pred` based on the specified `criterion`
    and `thresh`. It then computes various standard metrics.

    Currently, the following metrics are implemented:
    'fp', 'tp', 'fn', 'precision', 'recall', 'accuracy', 'f1', 'criterion',
    'thresh', 'n_true', 'n_pred', 'mean_true_score', 'mean_matched_score',
    'panoptic_quality'.

    - `tp` (True Positives): Number of matched ground truth objects.
    - `fp` (False Positives): Number of predicted objects that do not match any ground truth.
    - `fn` (False Negatives): Number of ground truth objects that are not matched by any prediction.
    - `precision`: `TP / (TP + FP)`
    - `recall`: `TP / (TP + FN)`
    - `accuracy`: `TP / (TP + FP + FN)` (often referred to as Detection Quality or Average Precision in some contexts)
    - `f1`: `(2 * TP) / (2 * TP + FP + FN)` (Dice coefficient)
    - `mean_matched_score`: The mean score (based on `criterion`) of matched true positives.
    - `mean_true_score`: The sum of scores of matched true positives, normalized by the total number of ground truth objects.
    - `panoptic_quality`: Defined as `(sum_matched_score) / (TP + 0.5*FP + 0.5*FN)` as in Eq. 1 of Kirillov et al. "Panoptic Segmentation", CVPR 2019.
    - `n_true`: Total number of ground truth objects.
    - `n_pred`: Total number of predicted objects.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth label image (integer valued). Objects are labeled with
        unique positive integers, background is 0.
    y_pred : np.ndarray
        Predicted label image (integer valued). Objects are labeled with
        unique positive integers, background is 0.
    thresh : float or List[float], optional
        Threshold(s) for the matching criterion. An object pair is considered
        a match if their score (based on `criterion`) is greater than or equal to `thresh`.
        If a list is provided, metrics are computed for each threshold. Defaults to 0.5.
    criterion : Literal["iou", "iot", "iop"], optional
        The matching criterion to use. Options are "iou" (Intersection over Union),
        "iot" (Intersection over True), or "iop" (Intersection over Pred).
        Defaults to "iou".
    report_matches : bool, optional
        If True, the returned dictionary (or dictionaries) will additionally
        include `matched_pairs` (list of (true_id, pred_id) for all matched objects,
        even if below `thresh`), `matched_scores` (corresponding scores),
        `matched_tps` (indices of true positives within `matched_pairs`),
        `pred_ids` (all unique predicted labels), and `gt_ids` (all unique ground truth labels).
        Defaults to False.

    Returns
    -------
    Dict or Tuple[Dict, ...]
        If `thresh` is a single float, returns a dictionary of metrics.
        If `thresh` is a list of floats, returns a tuple of dictionaries,
        one for each threshold.

    Examples
    --------
    >>> y_true = np.zeros((10,10), np.uint16)
    >>> y_true[1:3,1:3] = 1
    >>> y_true[6:8,6:8] = 2
    >>> y_pred = np.zeros((10,10), np.uint16)
    >>> y_pred[1:3,1:3] = 1 # Perfect match for object 1
    >>> y_pred[5:7,5:7] = 2 # Shifted match for object 2 (partial overlap)
    >>> y_pred[8:9,8:9] = 3 # False Positive

    >>> stats = matching(y_true, y_pred, thresh=0.5, criterion="iou")
    >>> print(stats['tp'], stats['fp'], stats['fn'], stats['precision'], stats['recall'])
    1 1 1 0.5 0.5

    >>> stats_multiple_thresh = matching(y_true, y_pred, thresh=[0.1, 0.5, 0.9], criterion="iou")
    >>> for s in stats_multiple_thresh: print(f"Thresh: {s['thresh']}, TP: {s['tp']}")
    Thresh: 0.1, TP: 2
    Thresh: 0.5, TP: 1
    Thresh: 0.9, TP: 1
    """
    _check_label_array(y_true, "y_true")
    _check_label_array(y_pred, "y_pred")
    y_true.shape == y_pred.shape or _raise(
        ValueError(
            "y_true ({y_true.shape}) and y_pred ({y_pred.shape}) have different shapes".format(
                y_true=y_true, y_pred=y_pred
            )
        )
    )
    criterion in matching_criteria or _raise(ValueError("Matching criterion '%s' not supported." % criterion))
    if thresh is None:
        thresh = 0
    thresh = float(thresh) if np.isscalar(thresh) else map(float, thresh)

    y_true, _, map_rev_true = relabel_sequential(y_true)
    y_pred, _, map_rev_pred = relabel_sequential(y_pred)

    map_rev_true = np.array(map_rev_true)
    map_rev_pred = np.array(map_rev_pred)

    overlap = label_overlap(y_true, y_pred, check=False)
    scores = matching_criteria[criterion](overlap)
    assert 0 <= np.min(scores) <= np.max(scores) <= 1

    # ignoring background
    scores = scores[1:, 1:]
    n_true, n_pred = scores.shape
    n_matched = min(n_true, n_pred)

    def _single(thr):
        not_trivial = n_matched > 0 and np.any(scores >= thr)
        if not_trivial:
            # compute optimal matching with scores as tie-breaker
            costs = -(scores >= thr).astype(float) - scores / (2 * n_matched)
            true_ind, pred_ind = linear_sum_assignment(costs)
            assert n_matched == len(true_ind) == len(pred_ind)
            match_ok = scores[true_ind, pred_ind] >= thr
            tp = np.count_nonzero(match_ok)
        else:
            tp = 0
        fp = n_pred - tp
        fn = n_true - tp
        # assert tp+fp == n_pred
        # assert tp+fn == n_true

        # the score sum over all matched objects (tp)
        sum_matched_score = np.sum(scores[true_ind, pred_ind][match_ok]) if not_trivial else 0.0

        # the score average over all matched objects (tp)
        mean_matched_score = _safe_divide(sum_matched_score, tp)
        # the score average over all gt/true objects
        mean_true_score = _safe_divide(sum_matched_score, n_true)
        panoptic_quality = _safe_divide(sum_matched_score, tp + fp / 2 + fn / 2)

        stats_dict = dict(
            criterion=criterion,
            thresh=thr,
            fp=fp,
            tp=tp,
            fn=fn,
            precision=precision(tp, fp, fn),
            recall=recall(tp, fp, fn),
            accuracy=accuracy(tp, fp, fn),
            f1=f1(tp, fp, fn),
            n_true=n_true,
            n_pred=n_pred,
            mean_true_score=mean_true_score,
            mean_matched_score=mean_matched_score,
            panoptic_quality=panoptic_quality,
        )
        if bool(report_matches):
            if not_trivial:
                stats_dict.update(
                    # int() to be json serializable
                    matched_pairs=tuple(
                        (int(map_rev_true[i]), int(map_rev_pred[j])) for i, j in zip(1 + true_ind, 1 + pred_ind)
                    ),
                    matched_scores=tuple(scores[true_ind, pred_ind]),
                    matched_tps=tuple(map(int, np.flatnonzero(match_ok))),
                    pred_ids=tuple(map_rev_pred),
                    gt_ids=tuple(map_rev_true),
                )
            else:
                stats_dict.update(
                    matched_pairs=(),
                    matched_scores=(),
                    matched_tps=(),
                    pred_ids=(),
                    gt_ids=(),
                )
        return stats_dict

    return _single(thresh) if np.isscalar(thresh) else tuple(map(_single, thresh))


def wrapper_matching_dataset_lazy(stats_all, thresh, criterion="iou", by_image=False):
    """
    Aggregate matching statistics across multiple images in a dataset.

    This function takes a list of per-image matching results (each potentially
    for multiple thresholds) and computes overall dataset-level metrics.
    It can either average metrics per image (`by_image=True`) or sum up
    TP/FP/FN counts across all images (`by_image=False`) before calculating
    the final aggregated metrics.

    Parameters
    ----------
    stats_all : List[Tuple[Dict, ...]]
        A list where each element is the result of a `matching` call for a single image.
        Each element can be a single dictionary (if `matching` was called with one threshold)
        or a tuple of dictionaries (if `matching` was called with multiple thresholds).
    thresh : List[float]
        A list of thresholds for which the statistics were originally computed.
        This must match the thresholds used when calling `matching` for individual images.
    criterion : Literal["iou", "iot", "iop"], optional
        The matching criterion used for individual image statistics. Defaults to "iou".
    by_image : bool, optional
        If True, metrics (precision, recall, etc.) are averaged across images.
        If False, TP, FP, FN counts are summed across all images, and then
        metrics are calculated from these summed counts. Defaults to False.

    Returns
    -------
    Dict or Tuple[Dict, ...]
        If `thresh` contains a single threshold, returns a single dictionary
        of aggregated metrics. If `thresh` contains multiple thresholds,
        returns a tuple of dictionaries, one for each threshold.
        The returned dictionaries contain the same keys as the output of `matching`.
    """
    expected_keys = set(
        (
            "fp",
            "tp",
            "fn",
            "precision",
            "recall",
            "accuracy",
            "f1",
            "criterion",
            "thresh",
            "n_true",
            "n_pred",
            "mean_true_score",
            "mean_matched_score",
            "panoptic_quality",
        )
    )

    # accumulate results over all images for each threshold separately
    n_images, n_threshs = len(stats_all), len(thresh)
    single_thresh = True if n_threshs == 1 else False
    accumulate = [{} for _ in range(n_threshs)]
    for stats in stats_all:
        for i, s in enumerate(stats):
            acc = accumulate[i]
            for item in s.items():
                k, v = item
                if k == "mean_true_score" and not bool(by_image):
                    # convert mean_true_score to "sum_matched_score"
                    acc[k] = acc.setdefault(k, 0) + v * s["n_true"]
                else:
                    try:
                        acc[k] = acc.setdefault(k, 0) + v
                    except TypeError:
                        pass

    # normalize/compute 'precision', 'recall', 'accuracy', 'f1'
    for thr, acc in zip(thresh, accumulate):
        acc["criterion"] = criterion
        acc["thresh"] = thr
        acc["by_image"] = bool(by_image)
        if bool(by_image):
            for k in (
                "precision",
                "recall",
                "accuracy",
                "f1",
                "mean_true_score",
                "mean_matched_score",
                "panoptic_quality",
            ):
                acc[k] /= n_images
        else:
            tp, fp, fn, n_true = acc["tp"], acc["fp"], acc["fn"], acc["n_true"]
            sum_matched_score = acc["mean_true_score"]

            mean_matched_score = _safe_divide(sum_matched_score, tp)
            mean_true_score = _safe_divide(sum_matched_score, n_true)
            panoptic_quality = _safe_divide(sum_matched_score, tp + fp / 2 + fn / 2)

            acc.update(
                precision=precision(tp, fp, fn),
                recall=recall(tp, fp, fn),
                accuracy=accuracy(tp, fp, fn),
                f1=f1(tp, fp, fn),
                mean_true_score=mean_true_score,
                mean_matched_score=mean_matched_score,
                panoptic_quality=panoptic_quality,
            )

    accumulate = tuple(namedtuple("DatasetMatching", acc.keys())(*acc.values()) for acc in accumulate)
    return accumulate[0] if single_thresh else accumulate
