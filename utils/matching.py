"""Copied from https://github.com/stardist/stardist"""

import numpy as np

from numba import jit
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from collections import namedtuple
import pandas as pd
import networkx as nx

matching_criteria = dict()

# Copied from https://github.com/CSBDeep/CSBDeep/blob/master/csbdeep/utils/utils.py
def _raise(e):
    if isinstance(e, BaseException):
        raise e
    else:
        raise ValueError(e)

def label_are_sequential(y):
    """ returns true if y has only sequential labels from 1... """
    labels = np.unique(y)
    return (set(labels)-{0}) == set(range(1,1+labels.max()))


def is_array_of_integers(y):
    return isinstance(y,np.ndarray) and np.issubdtype(y.dtype, np.integer)


def _check_label_array(y, name=None, check_sequential=False):
    err = ValueError("{label} must be an array of {integers}.".format(
        label = 'labels' if name is None else name,
        integers = ('sequential ' if check_sequential else '') + 'non-negative integers',
    ))
    is_array_of_integers(y) or _raise(err)
    if len(y) == 0:
        return True
    if check_sequential:
        label_are_sequential(y) or _raise(err)
    else:
        y.min() >= 0 or _raise(err)
    return True


def label_overlap(x, y, check=True):
    if check:
        _check_label_array(x,'x',True)
        _check_label_array(y,'y',True)
        x.shape == y.shape or _raise(ValueError("x and y must have the same shape"))
    return _label_overlap(x, y)

@jit(nopython=True)
def _label_overlap(x, y):
    x = x.ravel()
    y = y.ravel()
    overlap = np.zeros((1+x.max(),1+y.max()), dtype=np.uint)
    for i in range(len(x)):
        overlap[x[i],y[i]] += 1
    return overlap

def _safe_divide(x,y, eps=1e-10):
    """computes a safe divide which returns 0 if y is zero"""
    if np.isscalar(x) and np.isscalar(y):
        return x/y if np.abs(y)>eps else 0.0
    else:
        out = np.zeros(np.broadcast(x,y).shape, np.float32)
        np.divide(x,y, out=out, where=np.abs(y)>eps)
        return out


def intersection_over_union(overlap):
    _check_label_array(overlap,'overlap')
    if np.sum(overlap) == 0:
        return overlap
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    return _safe_divide(overlap, (n_pixels_pred + n_pixels_true - overlap))

matching_criteria['iou'] = intersection_over_union


def intersection_over_true(overlap):
    _check_label_array(overlap,'overlap')
    if np.sum(overlap) == 0:
        return overlap
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    return _safe_divide(overlap, n_pixels_true)

matching_criteria['iot'] = intersection_over_true


def intersection_over_pred(overlap):
    _check_label_array(overlap,'overlap')
    if np.sum(overlap) == 0:
        return overlap
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    return _safe_divide(overlap, n_pixels_pred)

matching_criteria['iop'] = intersection_over_pred


def precision(tp,fp,fn):
    return tp/(tp+fp) if tp > 0 else 0
def recall(tp,fp,fn):
    return tp/(tp+fn) if tp > 0 else 0
def accuracy(tp,fp,fn):
    # also known as "average precision" (?)
    # -> https://www.kaggle.com/c/data-science-bowl-2018#evaluation
    return tp/(tp+fp+fn) if tp > 0 else 0
def f1(tp,fp,fn):
    # also known as "dice coefficient"
    return (2*tp)/(2*tp+fp+fn) if tp > 0 else 0


def matching(y_true, y_pred, thresh=0.5, criterion='iou', report_matches=False):
    """Calculate detection/instance segmentation metrics between ground truth and predicted label images.

    Currently, the following metrics are implemented:

    'fp', 'tp', 'fn', 'precision', 'recall', 'accuracy', 'f1', 'criterion', 'thresh', 'n_true', 'n_pred', 'mean_true_score', 'mean_matched_score', 'panoptic_quality'

    Corresponding objects of y_true and y_pred are counted as true positives (tp), false positives (fp), and false negatives (fn)
    whether their intersection over union (IoU) >= thresh (for criterion='iou', which can be changed)

    * mean_matched_score is the mean IoUs of matched true positives

    * mean_true_score is the mean IoUs of matched true positives but normalized by the total number of GT objects

    * panoptic_quality defined as in Eq. 1 of Kirillov et al. "Panoptic Segmentation", CVPR 2019

    Parameters
    ----------
    y_true: ndarray
        ground truth label image (integer valued)
    y_pred: ndarray
        predicted label image (integer valued)
    thresh: float
        threshold for matching criterion (default 0.5)
    criterion: string
        matching criterion (default IoU)
    report_matches: bool
        if True, additionally calculate matched_pairs and matched_scores (note, that this returns even gt-pred pairs whose scores are below  'thresh')

    Returns
    -------
    Matching object with different metrics as attributes

    Examples
    --------
    >>> y_true = np.zeros((100,100), np.uint16)
    >>> y_true[10:20,10:20] = 1
    >>> y_pred = np.roll(y_true,5,axis = 0)

    >>> stats = matching(y_true, y_pred)
    >>> print(stats)
    Matching(criterion='iou', thresh=0.5, fp=1, tp=0, fn=1, precision=0, recall=0, accuracy=0, f1=0, n_true=1, n_pred=1, mean_true_score=0.0, mean_matched_score=0.0, panoptic_quality=0.0)

    """
    _check_label_array(y_true,'y_true')
    _check_label_array(y_pred,'y_pred')
    y_true.shape == y_pred.shape or _raise(ValueError("y_true ({y_true.shape}) and y_pred ({y_pred.shape}) have different shapes".format(y_true=y_true, y_pred=y_pred)))
    criterion in matching_criteria or _raise(ValueError("Matching criterion '%s' not supported." % criterion))
    if thresh is None: thresh = 0
    thresh = float(thresh) if np.isscalar(thresh) else map(float,thresh)

    y_true, _, map_rev_true = relabel_sequential(y_true)
    y_pred, _, map_rev_pred = relabel_sequential(y_pred)

    overlap = label_overlap(y_true, y_pred, check=False)
    scores = matching_criteria[criterion](overlap)
    assert 0 <= np.min(scores) <= np.max(scores) <= 1

    # ignoring background
    scores = scores[1:,1:]
    n_true, n_pred = scores.shape
    n_matched = min(n_true, n_pred)

    def _single(thr):
        not_trivial = n_matched > 0 and np.any(scores >= thr)
        if not_trivial:
            # compute optimal matching with scores as tie-breaker
            costs = -(scores >= thr).astype(float) - scores / (2*n_matched)
            true_ind, pred_ind = linear_sum_assignment(costs)
            assert n_matched == len(true_ind) == len(pred_ind)
            match_ok = scores[true_ind,pred_ind] >= thr
            tp = np.count_nonzero(match_ok)
        else:
            tp = 0
        fp = n_pred - tp
        fn = n_true - tp
        # assert tp+fp == n_pred
        # assert tp+fn == n_true

        # the score sum over all matched objects (tp)
        sum_matched_score = np.sum(scores[true_ind,pred_ind][match_ok]) if not_trivial else 0.0

        # the score average over all matched objects (tp)
        mean_matched_score = _safe_divide(sum_matched_score, tp)
        # the score average over all gt/true objects
        mean_true_score    = _safe_divide(sum_matched_score, n_true)
        panoptic_quality   = _safe_divide(sum_matched_score, tp+fp/2+fn/2)

        stats_dict = dict (
            criterion          = criterion,
            thresh             = thr,
            fp                 = fp,
            tp                 = tp,
            fn                 = fn,
            precision          = precision(tp,fp,fn),
            recall             = recall(tp,fp,fn),
            accuracy           = accuracy(tp,fp,fn),
            f1                 = f1(tp,fp,fn),
            n_true             = n_true,
            n_pred             = n_pred,
            mean_true_score    = mean_true_score,
            mean_matched_score = mean_matched_score,
            panoptic_quality   = panoptic_quality,
        )
        if bool(report_matches):
            if not_trivial:
                stats_dict.update (
                    # int() to be json serializable
                    matched_pairs  = tuple((int(map_rev_true[i]),int(map_rev_pred[j])) for i,j in zip(1+true_ind,1+pred_ind)),
                    matched_scores = tuple(scores[true_ind,pred_ind]),
                    matched_tps    = tuple(map(int,np.flatnonzero(match_ok))),
                    pred_ids       = tuple(map_rev_pred),        
                    gt_ids         = tuple(map_rev_true),
                )
            else:
                stats_dict.update (
                    matched_pairs  = (),
                    matched_scores = (),
                    matched_tps    = (),
                    pred_ids       = (), 
                    gt_ids         = (),
                )
        return stats_dict

    return _single(thresh) if np.isscalar(thresh) else tuple(map(_single,thresh))


def wrapper_matching_dataset_lazy(stats_all, thresh, criterion='iou', by_image=False):

    expected_keys = set(('fp', 'tp', 'fn', 'precision', 'recall', 'accuracy', 'f1', 'criterion', 'thresh', 'n_true', 'n_pred', 'mean_true_score', 'mean_matched_score', 'panoptic_quality'))

    # accumulate results over all images for each threshold separately
    n_images, n_threshs = len(stats_all), len(thresh)
    single_thresh = True if n_threshs == 1 else False
    accumulate = [{} for _ in range(n_threshs)]
    for stats in stats_all:
        for i, s in enumerate(stats):
            acc = accumulate[i]
            for item in s.items():
                k, v = item
                if k == 'mean_true_score' and not bool(by_image):
                    # convert mean_true_score to "sum_matched_score"
                    acc[k] = acc.setdefault(k,0) + v * s['n_true']
                else:
                    try:
                        acc[k] = acc.setdefault(k,0) + v
                    except TypeError:
                        pass

    # normalize/compute 'precision', 'recall', 'accuracy', 'f1'
    for thr,acc in zip(thresh,accumulate):
        acc['criterion'] = criterion
        acc['thresh'] = thr
        acc['by_image'] = bool(by_image)
        if bool(by_image):
            for k in ('precision', 'recall', 'accuracy', 'f1', 'mean_true_score', 'mean_matched_score', 'panoptic_quality'):
                acc[k] /= n_images
        else:
            tp, fp, fn, n_true = acc['tp'], acc['fp'], acc['fn'], acc['n_true']
            sum_matched_score = acc['mean_true_score']

            mean_matched_score = _safe_divide(sum_matched_score, tp)
            mean_true_score    = _safe_divide(sum_matched_score, n_true)
            panoptic_quality   = _safe_divide(sum_matched_score, tp+fp/2+fn/2)

            acc.update(
                precision          = precision(tp,fp,fn),
                recall             = recall(tp,fp,fn),
                accuracy           = accuracy(tp,fp,fn),
                f1                 = f1(tp,fp,fn),
                mean_true_score    = mean_true_score,
                mean_matched_score = mean_matched_score,
                panoptic_quality   = panoptic_quality,
            )

    accumulate = tuple(namedtuple('DatasetMatching',acc.keys())(*acc.values()) for acc in accumulate)
    return accumulate[0] if single_thresh else accumulate

def wrapper_matching_segCompare(stats_all):
    expected_keys = ['number_of_cells', 'correct_segmentations', 'oversegmentation_rate', 'undersegmentation_rate', 'missing_rate']

    accumulated_values = dict.fromkeys(expected_keys, 0)

    for key in expected_keys:
        for stat in stats_all:
            accumulated_values[key] = accumulated_values[key] + stat[key]
        accumulated_values[key] = accumulated_values[key]/len(stats_all)
    return accumulated_values

# copied from scikit-image master for now (remove when part of a release)
def relabel_sequential(label_field, offset=1):
    """Relabel arbitrary labels to {`offset`, ... `offset` + number_of_labels}.

    This function also returns the forward map (mapping the original labels to
    the reduced labels) and the inverse map (mapping the reduced labels back
    to the original ones).

    Parameters
    ----------
    label_field : numpy array of int, arbitrary shape
        An array of labels, which must be non-negative integers.
    offset : int, optional
        The return labels will start at `offset`, which should be
        strictly positive.

    Returns
    -------
    relabeled : numpy array of int, same shape as `label_field`
        The input label field with labels mapped to
        {offset, ..., number_of_labels + offset - 1}.
        The data type will be the same as `label_field`, except when
        offset + number_of_labels causes overflow of the current data type.
    forward_map : numpy array of int, shape ``(label_field.max() + 1,)``
        The map from the original label space to the returned label
        space. Can be used to re-apply the same mapping. See examples
        for usage. The data type will be the same as `relabeled`.
    inverse_map : 1D numpy array of int, of length offset + number of labels
        The map from the new label space to the original space. This
        can be used to reconstruct the original label field from the
        relabeled one. The data type will be the same as `relabeled`.

    Notes
    -----
    The label 0 is assumed to denote the background and is never remapped.

    The forward map can be extremely big for some inputs, since its
    length is given by the maximum of the label field. However, in most
    situations, ``label_field.max()`` is much smaller than
    ``label_field.size``, and in these cases the forward map is
    guaranteed to be smaller than either the input or output images.

    Examples
    --------
    >>> from skimage.segmentation import relabel_sequential
    >>> label_field = np.array([1, 1, 5, 5, 8, 99, 42])
    >>> relab, fw, inv = relabel_sequential(label_field)
    >>> relab
    array([1, 1, 2, 2, 3, 5, 4])
    >>> fw
    array([0, 1, 0, 0, 0, 2, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5])
    >>> inv
    array([ 0,  1,  5,  8, 42, 99])
    >>> (fw[label_field] == relab).all()
    True
    >>> (inv[relab] == label_field).all()
    True
    >>> relab, fw, inv = relabel_sequential(label_field, offset=5)
    >>> relab
    array([5, 5, 6, 6, 7, 9, 8])
    """
    offset = int(offset)
    if offset <= 0:
        raise ValueError("Offset must be strictly positive.")
    if np.min(label_field) < 0:
        raise ValueError("Cannot relabel array that contains negative values.")
    max_label = int(label_field.max()) # Ensure max_label is an integer
    if not np.issubdtype(label_field.dtype, np.integer):
        new_type = np.min_scalar_type(max_label)
        label_field = label_field.astype(new_type)
    labels = np.unique(label_field)
    labels0 = labels[labels != 0]
    new_max_label = offset - 1 + len(labels0)
    new_labels0 = np.arange(offset, new_max_label + 1)
    output_type = label_field.dtype
    required_type = np.min_scalar_type(new_max_label)
    if np.dtype(required_type).itemsize > np.dtype(label_field.dtype).itemsize:
        output_type = required_type
    forward_map = np.zeros(max_label + 1, dtype=output_type)
    forward_map[labels0] = new_labels0
    inverse_map = np.zeros(new_max_label + 1, dtype=output_type)
    inverse_map[offset:] = labels0
    relabeled = forward_map[label_field]
    return relabeled, forward_map, inverse_map
    
    
def segmentation_state(row):
    """Function from https://mosaic.gitlabpages.inria.fr/publications/seg_compare/
    used in match_using_segCompare function                                                    

       Inputs
       ----------
       row : pandas dataframe row
           
       Returns
       -------
       string with the segmentation state of a cell based on the row

    """

    if len(row.reference) == 0:
        return 'background'
    elif len(row.target) == 0:
        return 'missing'
    elif len(row.target) == 1:
        if len(row.reference) == 1:
            return 'one-to-one'
        else:
            return 'under-segmentation'
    else:
        if len(row.reference) == 1:
            return 'over-segmentation'
        else:
            return 'misc.'
    
def match_using_segCompare(y_true, y_pred, BACKGROUND_LABEL=0, thresh_background=0.5):

    """Calcualte over-segmentation and under-segmentation rates based on the paper
    under-segmentation rates based on the paper entitled "Assessment of deep learning algorithms
    for 3D instance segmentation of confocal image datasets" by A. Kar et al 
    (https://doi.org/10.1101/2021.06.09.447748)
     
    Their code is available at https://mosaic.gitlabpages.inria.fr/publications/seg_compare/
    but uses some libraries that are not compatible with the python version used in this 
    repository.                                                      

       Inputs
       ----------
       y_true : 4D Numpy array
           Ground Truth data. E.g. ``(img_number, x, y, channels)``.
           
       y_pred : 4D Numpy array
           Predicted data  E.g. ``(img_number, x, y, channels)``.

      Returns
      -------
      stats_dict: Dictionary with the following info:
          % correct segmentations: float
              one-to-one associations
          % undersegmentation: float
              one-to-many associations
          % oversegmentation: float
              many-to-one associations
          % misc: float
              many-to-many associations
          % missing: float
              if ground truth cell is associated is not in predicted image
          % background: float
              if predicted cell is associated with ground truth background

    """
    
    #Number of labels
    groundTruthLabelsNum = np.size(np.unique(y_true))
    predictedCellsNum = np.size(np.unique(y_pred))

    #Initialize variables
    groundTruthCellVolume = np.zeros([groundTruthLabelsNum])
    predictedCellVolume = np.zeros([predictedCellsNum])

    #Relabel (we assume that background is the lowest value in image)
    if min(np.unique(y_true)) != 0:
        y_true = relabel_sequential(y_true, 1)[0]-1
    else:
        y_true = relabel_sequential(y_true, 1)[0]
    if min(np.unique(y_pred)) != 0:
        y_pred = relabel_sequential(y_pred, 1)[0]-1
    else:
        y_pred = relabel_sequential(y_pred, 1)[0]  

    predictedCells = np.unique(y_pred)
    groundTruthCells = np.unique(y_true)

    #Calculate unions, intersections, jaccardIndex, volumes and assymetric inclusion indices
    #VJI is not strictly neccessary, could be removed or maybe hidden w/ an if statement.

    for predictedCell in predictedCells:
        #Calculate GroundTruth cell and predictedCellVolume Volume
        predictedCellVolume[predictedCell] = np.count_nonzero(y_pred==predictedCell)

    df_target = pd.DataFrame(columns = ['target', 'reference', 'target_in_reference'])
    df_reference = pd.DataFrame(columns = ['target', 'reference', 'reference_in_target'])

    for groundTruthCell in groundTruthCells:
        gtCell = y_true==groundTruthCell

        validSlices = [np.count_nonzero(gtCell[slice, :, :])>0 for slice in range(np.shape(gtCell)[0])]
        
        gtCell = gtCell[validSlices, :, :]
        groundTruthCellVolume[groundTruthCell] = np.count_nonzero(y_true[validSlices, :, :]==groundTruthCell)
        matchingCells = np.unique(y_pred[(y_true==groundTruthCell)])

        for predictedCell in matchingCells:
        
            pCell =  y_pred[validSlices, :, :]==predictedCell
        
            #Calculate union and intersection
            intersection = (gtCell & pCell).sum()

            df_target.loc[len(df_target.index)] = [predictedCell, groundTruthCell, intersection/predictedCellVolume[predictedCell]]
            df_reference.loc[len(df_reference.index)] = [predictedCell, groundTruthCell, intersection/groundTruthCellVolume[groundTruthCell]]


    ### - Solve the background associations - ###
    # target --> reference background
    target_background = df_target.target.loc[(df_target.reference == BACKGROUND_LABEL) &
                                             (df_target.target_in_reference >= thresh_background)].to_list()

    # reference --> target background
    reference_background = df_reference.reference.loc[(df_reference.target == BACKGROUND_LABEL) &
                                                      (df_reference.reference_in_target >= thresh_background)].to_list()

    # - Add the background (make sure we remove them for the clique associations)
    target_background.append(BACKGROUND_LABEL)
    reference_background.append(BACKGROUND_LABEL)

    # - Remove the cells associated with the background + backgrounds
    df_target = df_target.loc[~((df_target.target.isin(target_background)) |
                            (df_target.reference == BACKGROUND_LABEL))].copy()
    df_reference = df_reference.loc[~((df_reference.reference.isin(reference_background)) |
                                  (df_reference.target == BACKGROUND_LABEL))].copy()
                       
    ### - Get the 1 <--> 1 associations - ###
    # - Associate each reference/target cell with the target/reference cell in which it is the most included
    df_target = df_target.loc[df_target.groupby('target')['target_in_reference'].idxmax()]
    df_reference = df_reference.loc[df_reference.groupby('reference')['reference_in_target'].idxmax()]

    # - Convert in dict
    target_in_reference = df_target[['target', 'reference']].set_index('target').to_dict()['reference']
    reference_in_target = df_reference[['reference', 'target']].set_index('reference').to_dict()['target']

    ### - Build the associations (bijection, over-segmentation,...) - ###
    # - Create a bipartite graph where nodes represent the A labels (left) and B labels (right)
    #   and the edges the associations obtained using the max/min methods

    # - Reindex the labels
    target_labels = list(set(df_target.target.values) | set(df_reference.target.values))
    reference_labels = list(set(df_target.reference.values) | set(df_reference.reference.values))

    label_tp_list = [(m, 'l') for m in target_labels] + [(d, 'r') for d in reference_labels]
    lg2nid = dict(zip(label_tp_list, range(len(label_tp_list))))

    # - Create the graph
    G = nx.Graph()

    G.add_nodes_from([(nid, {'label': lab, 'group': g}) for (lab, g), nid in lg2nid.items()])

    target_to_ref_list = [(lg2nid[(i, 'l')], lg2nid[(j, 'r')]) for i, j in target_in_reference.items()]
    G.add_edges_from(target_to_ref_list)

    ref_to_target_list = [(lg2nid[(i, 'r')], lg2nid[(j, 'l')]) for i, j in reference_in_target.items()]
    G.add_edges_from(ref_to_target_list)

    # - Overlap analysis
    # - Get the  target_cells <--> reference_cells from the connected subgraph in G
    connected_graph = [list(G.subgraph(c)) for c in nx.connected_components(G)]

    # - Gather all the connected subgraph and reindex according to the image labels
    nid2lg = {v: k for k, v in lg2nid.items()}

    out_results = []
    for c in connected_graph:
        if len(c) > 1:  # at least two labels
            target, reference = [], []
            for nid in c:
                if nid2lg[nid][1] == 'l':
                    target.append(nid2lg[nid][0])  # label from target image
                else:
                    reference.append(nid2lg[nid][0])  # label from reference image

            out_results.append({'target': target, 'reference': reference})

    # - Add the background associations
    # - target --> reference background
    for lab in target_background:
        if lab != BACKGROUND_LABEL:  # ignore target background
            out_results.append({'target': [lab], 'reference': []})

    # - reference --> target background
    for lab in reference_background:
        if lab != BACKGROUND_LABEL:  # ignore reference background
            out_results.append({'target': [], 'reference': [lab]})

    out_results = pd.DataFrame(out_results)
        
    out_results['segmentation_state'] = out_results.apply(segmentation_state, axis=1) # add name for each type of association

    ignore_background = True

    cell_statistics = {'one-to-one': 0, 'over-segmentation': 0, 'under-segmentation': 0, 'misc.': 0}

    if not ignore_background:
        cell_statistics['background'] = 0 # add background

    state_target = {lab: state for list_lab, state
                    in zip(out_results.target.values, out_results.segmentation_state.values)
                    for lab in list_lab if state in cell_statistics}

    for lab, state in state_target.items():
        cell_statistics[state] += 1

    total_cells = len(state_target)
    cell_statistics = {state: np.around(val / total_cells * 100, 2) for state, val in cell_statistics.items()}

    # - add the missing  cells : percentage of reference cells that are missing in the predicted segmentation
    total_reference_cells = len(np.unique([item for sublist in out_results.reference.values for item in sublist]))
    missing_cells = [item for sublist in out_results.reference.loc[out_results.segmentation_state == 'missing'].values for item in sublist]

    total_missing = len(missing_cells)
    cell_statistics['missing'] = np.around(total_missing/total_reference_cells * 100, 2)

    stats_dict = dict (
        number_of_cells = total_cells,
        correct_segmentations = cell_statistics['one-to-one'],
        oversegmentation_rate = cell_statistics['over-segmentation'],
        undersegmentation_rate = cell_statistics['under-segmentation'],
        missing_rate = cell_statistics['missing']
    )

    return stats_dict
    
