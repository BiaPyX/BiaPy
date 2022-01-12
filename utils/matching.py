"""Copied from https://github.com/stardist/stardist"""

import numpy as np

from numba import jit
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from collections import namedtuple

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
                )
            else:
                stats_dict.update (
                    matched_pairs  = (),
                    matched_scores = (),
                    matched_tps    = (),
                )
        return namedtuple('Matching',stats_dict.keys())(*stats_dict.values())

    return _single(thresh) if np.isscalar(thresh) else tuple(map(_single,thresh))



def matching_dataset(y_true, y_pred, thresh=0.5, criterion='iou', by_image=False, show_progress=True, parallel=False):
    """matching metrics for list of images, see `stardist.matching.matching`
    """
    len(y_true) == len(y_pred) or _raise(ValueError("y_true and y_pred must have the same length."))
    return matching_dataset_lazy (
        tuple(zip(y_true,y_pred)), thresh=thresh, criterion=criterion, by_image=by_image, show_progress=show_progress, parallel=parallel,
    )



def matching_dataset_lazy(y_gen, thresh=0.5, criterion='iou', by_image=False, show_progress=True, parallel=False):

    expected_keys = set(('fp', 'tp', 'fn', 'precision', 'recall', 'accuracy', 'f1', 'criterion', 'thresh', 'n_true', 'n_pred', 'mean_true_score', 'mean_matched_score', 'panoptic_quality'))

    single_thresh = False
    if np.isscalar(thresh):
        single_thresh = True
        thresh = (thresh,)

    tqdm_kwargs = {}
    tqdm_kwargs['disable'] = not bool(show_progress)
    if int(show_progress) > 1:
        tqdm_kwargs['total'] = int(show_progress)

    # compute matching stats for every pair of label images
    if parallel:
        from concurrent.futures import ThreadPoolExecutor
        fn = lambda pair: matching(*pair, thresh=thresh, criterion=criterion, report_matches=False)
        with ThreadPoolExecutor() as pool:
            stats_all = tuple(pool.map(fn, tqdm(y_gen,**tqdm_kwargs)))
    else:
        stats_all = tuple (
            matching(y_t, y_p, thresh=thresh, criterion=criterion, report_matches=False)
            for y_t,y_p in tqdm(y_gen,**tqdm_kwargs)
        )

    # accumulate results over all images for each threshold separately
    n_images, n_threshs = len(stats_all), len(thresh)
    accumulate = [{} for _ in range(n_threshs)]
    for stats in stats_all:
        for i,s in enumerate(stats):
            acc = accumulate[i]
            for k,v in s._asdict().items():
                if k == 'mean_true_score' and not bool(by_image):
                    # convert mean_true_score to "sum_matched_score"
                    acc[k] = acc.setdefault(k,0) + v * s.n_true
                else:
                    try:
                        acc[k] = acc.setdefault(k,0) + v
                    except TypeError:
                        pass

    # normalize/compute 'precision', 'recall', 'accuracy', 'f1'
    for thr,acc in zip(thresh,accumulate):
        set(acc.keys()) == expected_keys or _raise(ValueError("unexpected keys"))
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
    
def match_using_VJI_and_PAI(y_true, y_pred):

    """Calcualte Volume Averaged Jaccard Index (VJI) metric as well as over-segmentation and 
    under-segmentation rates based on the paper entitled "Assessment of deep learning algorithms
    for 3D instance segmentation of confocal image datasets" by A. Kar et al 
    (https://doi.org/10.1101/2021.06.09.447748)
     
    The numbers next to the calculations correspond to equation number in the aforementioned
    paper.                                                             

       Inputs
       ----------
       y_true : 4D Numpy array
           Ground Truth data. E.g. ``(img_number, x, y, channels)``.
           
       y_pred : 4D Numpy array
           Predicted data  E.g. ``(img_number, x, y, channels)``.

      Returns
      -------
      stats_dict: Dictionary with the following info:
		  VJI: float
		      Volume Averaged Jaccard Index. As defined in https://doi.org/10.1101/2021.06.09.447748
		  backgroundRate: float
		      Percentage of predicted cells pair-associated with the ground truth background
		  oversegmentationRate: float
		      Percentage of ground truth cells pair-associated with more than one predicted cell.
		  undersegmentationRate: float
		      Percentage of predicted cells pair-associated with more than one ground truth cell.
		  bijectionRate: float
		      Percentage of predicted cells pair-associated with just one ground truth cell.

    """

    ground_truth_cells = np.unique(y_true)
    predicted_cells = np.unique(y_pred)

    ground_truth_cells_num = np.size(ground_truth_cells)
    predicted_cells_num = np.size(predicted_cells)
    
    #Initialize variables
    jaccard_index = np.zeros([ground_truth_cells_num, predicted_cells_num])
    ground_truth_cellVolume = np.zeros([ground_truth_cells_num])
    predicted_cellVolume = np.zeros([predicted_cells_num])
    asymetric_inclusion_GC = np.zeros([ground_truth_cells_num, predicted_cells_num])
    asymetric_inclusion_PC = np.zeros([predicted_cells_num, ground_truth_cells_num])
    A = np.zeros([ground_truth_cells_num], dtype=np.int)
    B = np.zeros([ground_truth_cells_num], dtype=np.int)
    Bprime = np.zeros([predicted_cells_num], dtype=np.int)

    #Relabel
    new_label = 0
    for ground_truth_cell in ground_truth_cells:
        y_true[y_true==ground_truth_cell] = new_label
        new_label = new_label + 1
    ground_truth_cells = np.unique(y_true)

    new_label = 0
    for predicted_cell in predicted_cells:
        y_pred[y_pred==predicted_cell] = new_label
        new_label = new_label + 1
    predicted_cells = np.unique(y_pred)

    #Calculate unions, intersections, jaccard_index, volumes and assymetric inclusion indices
    for predicted_cell in predicted_cells:
        #Calculate GroundTruth cell and predicted_cellVolume Volume
        predicted_cellVolume[predicted_cell] = np.count_nonzero(y_pred==predicted_cell)
    matchingCells = np.unique(y_pred[(y_true==3) & (y_pred>0)])
    
    for ground_truth_cell in ground_truth_cells:
        gt_cell = y_true==ground_truth_cell
        #Crop to speed up
        validSlices = [np.count_nonzero(gt_cell[slice, :, :])>0 for slice in range(np.shape(gt_cell)[0])]
        gt_cell = gt_cell[validSlices, :, :]
    
        matchingCells = np.unique(y_pred[(y_true==ground_truth_cell) & (y_pred>0)])

        #Calculate GroundTruth cell and predicted_cellVolume Volume
        ground_truth_cellVolume[ground_truth_cell] = np.count_nonzero(y_true==ground_truth_cells[ground_truth_cell])
            
        for predicted_cell in matchingCells:
            p_cell =  y_pred==predicted_cell
            p_cell = p_cell[validSlices, :, :]
        
            #Calculate union and intersection
            intersection = (gt_cell & p_cell).sum()

            union = ground_truth_cellVolume[ground_truth_cell]+predicted_cellVolume[predicted_cell]-intersection
        
            #Calculate Jaccard Index (1)
            jaccard_index[ground_truth_cell, predicted_cell] = intersection/union

            #Calculate Assymetric Inclusion Index (2)
            asymetric_inclusion_GC[ground_truth_cell, predicted_cell] = intersection/ground_truth_cellVolume[ground_truth_cell]
            asymetric_inclusion_PC[predicted_cell, ground_truth_cell] = intersection/predicted_cellVolume[predicted_cell]

        #Calculate A (3) and B (4)
        A[ground_truth_cell] = np.argmax(jaccard_index[ground_truth_cell, :])
        B[ground_truth_cell] = np.argmax(asymetric_inclusion_GC[ground_truth_cell, :])

    #Calculate B prime (5)
    for predicted_cell in predicted_cells:
        Bprime[predicted_cell] = np.argmax(asymetric_inclusion_PC[predicted_cell, :])

    #Calculate Volume Averaged Jaccard (VJI) (6)
    upperPart = 0
    lowerPart = 0

    for ground_truth_cell in ground_truth_cells[1:]:
        upperPart = upperPart + ground_truth_cellVolume[ground_truth_cell]*jaccard_index[ground_truth_cell,predicted_cells[A[ground_truth_cell]]]
        lowerPart = lowerPart + ground_truth_cellVolume[ground_truth_cell]

    VJI = upperPart/lowerPart

    print('VJI = {}'.format(VJI))

    #Under and over segmentation rates
    pair_asociated_indices = [0, 0];
    for ground_truth_cell in ground_truth_cells:
        for predicted_cell in predicted_cells:
	        if(B[ground_truth_cell]==predicted_cell or Bprime[predicted_cell]==ground_truth_cell):
	            pair_asociated_indices = np.vstack([pair_asociated_indices, [ground_truth_cell, predicted_cell]])
	        
    pair_asociated_indices = pair_asociated_indices[1:]

    oversegmented = 0;
    for ground_truth_cell in ground_truth_cells[1:]:
        if(np.shape(pair_asociated_indices[pair_asociated_indices[:, 0]==ground_truth_cell, :])[0]>1):
            for predicted_cell in pair_asociated_indices[pair_asociated_indices[:, 0]==ground_truth_cell, 1]:
                if ground_truth_cell in pair_asociated_indices[pair_asociated_indices[:, 1]==predicted_cell, 0]:
                    oversegmented = oversegmented + 1
       
    background = 0
    ground_truth_cell = ground_truth_cells[0]
    if(np.shape(pair_asociated_indices[pair_asociated_indices[:, 0]==ground_truth_cell, :])[0]>1):
        for predicted_cell in pair_asociated_indices[pair_asociated_indices[:, 0]==ground_truth_cell, 1]:
            if ground_truth_cell in pair_asociated_indices[pair_asociated_indices[:, 1]==predicted_cell, 0]:
                background = background + 1
    
    bijection = 0;
    for predicted_cell in predicted_cells:
        if(np.shape(pair_asociated_indices[pair_asociated_indices[:, 1]==predicted_cell, :])[0]==1):
            for ground_truth_cell in pair_asociated_indices[pair_asociated_indices[:, 1]==predicted_cell, 0]:
                if predicted_cells in pair_asociated_indices[pair_asociated_indices[:, 0]==ground_truth_cell, 1]:
                    bijection = bijection + 1
                
    undersegmented = 0;
    for predicted_cell in predicted_cells:
        if(np.shape(pair_asociated_indices[pair_asociated_indices[:, 1]==predicted_cell, :])[0]>1):
            undersegmented_aux = 0;
            for ground_truth_cell in pair_asociated_indices[pair_asociated_indices[:, 1]==predicted_cell, 0]:
                if predicted_cell in pair_asociated_indices[pair_asociated_indices[:, 0]==ground_truth_cell, 1]:
                    undersegmented_aux = undersegmented_aux + 1
                if undersegmented_aux == len(pair_asociated_indices[pair_asociated_indices[:, 0]==ground_truth_cell, 1]):
                    undersegmented = undersegmented + 1
                    
    validCells = len(np.unique(pair_asociated_indices[:, 1]))

    background_rate = (background/validCells)
    oversegmentation_rate = (oversegmented/validCells)
    undersegmentation_rate = (undersegmented/validCells)
    bijection_rate = (bijection/validCells)

    stats_dict = dict (
        VJI = VJI,
        background_rate = background_rate,
        oversegmentation_rate = oversegmentation_rate,
        undersegmentation_rate = undersegmentation_rate,
        bijection_rate = bijection_rate,
    )

    return stats_dict
    
