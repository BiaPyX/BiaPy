import time
import os
import torch
import distutils
import numpy as np
import pandas as pd
from skimage import measure
from PIL import Image
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
from torchmetrics import JaccardIndex
from torchmetrics.image import StructuralSimilarityIndexMeasure

def jaccard_index_numpy(y_true, y_pred):
    """Define Jaccard index.

       Parameters
       ----------
       y_true : N dim Numpy array
           Ground truth masks. E.g. ``(num_of_images, x, y, channels)`` for 2D images or
           ``(volume_number, z, x, y, channels)`` for 3D volumes.

       y_pred : N dim Numpy array
           Predicted masks. E.g. ``(num_of_images, x, y, channels)`` for 2D images or
           ``(volume_number, z, x, y, channels)`` for 3D volumes.

       Returns
       -------
       jac : float
           Jaccard index value.
    """

    if y_true.ndim != y_pred.ndim:
        raise ValueError("Dimension mismatch: {} and {} provided".format(y_true.shape, y_pred.shape))

    TP = np.count_nonzero(y_pred * y_true)
    FP = np.count_nonzero(y_pred * (y_true - 1))
    FN = np.count_nonzero((y_pred - 1) * y_true)

    if (TP + FP + FN) == 0:
        jac = 0
    else:
        jac = TP / (TP + FP + FN)

    return jac


def jaccard_index_numpy_without_background(y_true, y_pred):
    """Define Jaccard index excluding the background class (first channel).

       Parameters
       ----------
       y_true : N dim Numpy array
           Ground truth masks. E.g. ``(num_of_images, x, y, channels)`` for 2D images or
           ``(volume_number, z, x, y, channels)`` for 3D volumes.

       y_pred : N dim Numpy array
           Predicted masks. E.g. ``(num_of_images, x, y, channels)`` for 2D images or
           ``(volume_number, z, x, y, channels)`` for 3D volumes.

       Returns
       -------
       jac : float
           Jaccard index value.
    """

    if y_true.ndim != y_pred.ndim:
        raise ValueError("Dimension mismatch: {} and {} provided".format(y_true.shape, y_pred.shape))

    TP = np.count_nonzero(y_pred[...,1:] * y_true[...,1:])
    FP = np.count_nonzero(y_pred[...,1:] * (y_true[...,1:] - 1))
    FN = np.count_nonzero((y_pred[...,1:] - 1) * y_true[...,1:])

    if (TP + FP + FN) == 0:
        jac = 0
    else:
        jac = TP / (TP + FP + FN)

    return jac


def jaccard_index(y_pred, y_true, device, t=0.5, num_classes=2, first_not_binary_channel=2):
    """
    Define Jaccard index.

    Parameters
    ---------- 
    y_pred : Tensor
        Predicted masks.

    y_true : Tensor
        Ground truth masks.

    t : float, optional
        Threshold to be applied.

    num_classes : int, optional
        Number of classes.

    first_not_binary_channel : int, optional
        First channel not binary to not apply IoU in. 
        
    Returns
    -------
    jac : Tensor
        Jaccard index value
    """
    task = "multiclass" if num_classes > 2 else "binary"
    jaccard = JaccardIndex(task=task, threshold=t, num_classes=num_classes).to(device, non_blocking=True)
    return jaccard(y_pred[:,:first_not_binary_channel], y_true[:,:first_not_binary_channel])

def dice_loss(y_true, y_pred):
    """Dice loss.

       Based on `image_segmentation.ipynb <https://colab.research.google.com/github/tensorflow/models/blob/master/samples/outreach/blogs/segmentation_blogpost/image_segmentation.ipynb>`_.

       Parameters
       ----------
       y_true : Tensor
           Ground truth masks.

       y_pred : Tensor
           Predicted masks.

       Returns
       -------
       loss : Tensor
           Loss value.
    """

    loss = 1 - dice_coeff(y_true, y_pred)
    return loss


def bce_dice_loss(y_true, y_pred):
    """Loss function based on the combination of BCE and Dice.

       Based on `image_segmentation.ipynb <https://colab.research.google.com/github/tensorflow/models/blob/master/samples/outreach/blogs/segmentation_blogpost/image_segmentation.ipynb>`_.

       Parameters
       ----------
       y_true : Numpy array
           Ground truth.

       y_pred : Numpy array
           Predictions.

       Returns
       -------
       loss : Tensor
           Loss value.
    """
    loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


def weighted_bce_dice_loss(w_dice=0.5, w_bce=0.5):
    """Loss function based on the combination of BCE and Dice weighted.

       Inspired by `https://medium.com/@Bloomore post <https://medium.com/@Bloomore/how-to-write-a-custom-loss-function-with-additional-arguments-in-keras-5f193929f7a0>`_.

       Parameters
       ----------
       w_dice : float, optional
           Weight to be applied to Dice loss.

       w_bce : float, optional
           Weight to be applied to BCE.

       Returns
       -------
       loss : Tensor
           Loss value.
    """
    def loss(y_true, y_pred):
        return losses.binary_crossentropy(y_true, y_pred) * w_bce + dice_loss(y_true, y_pred) * w_dice
    return loss


def voc_calculation(y_true, y_pred, foreground):
    """Calculate VOC metric value.

       Parameters
       ----------
       y_true : 4D Numpy array
           Ground truth masks. E.g. ``(num_of_images, x, y, channels)``.

       y_pred : 4D Numpy array
           Predicted masks. E.g. ``(num_of_images, x, y, channels)``.

       foreground : float
           Foreground Jaccard index score.

       Returns
       -------
       voc : float
           VOC score value.
    """

    # Invert the arrays
    start_time = time.time()
    y_pred[y_pred == 0] = 2
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == 2] = 1

    y_true[y_true == 0] = 2
    y_true[y_true == 1] = 0
    y_true[y_true == 2] = 1

    background = jaccard_index_numpy(y_true, y_pred)
    voc = (float)(foreground + background)/2

    # Revert the changes
    y_pred[y_pred == 0] = 2
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == 2] = 1

    y_true[y_true == 0] = 2
    y_true[y_true == 1] = 0
    y_true[y_true == 2] = 1

    return voc

def binary_crossentropy_weighted(weights):
    """Custom binary cross entropy loss. The weights are used to multiply the results of the usual cross-entropy loss
       in order to give more weight to areas between cells close to one another.

       Based on `unet_weights.py <https://github.com/deepimagej/python4deepimagej/blob/master/unet/py_files/unet_weights.py>`_.

       Parameters
       ----------
       weights : float
           Weigth to multiply the BCE value by.

       Returns
       -------
       loss : Tensor
           Loss value.
    """
    def loss(y_true, y_pred):
        return K.mean(weights * K.binary_crossentropy(y_true, y_pred), axis=-1)

    return loss


def instance_segmentation_loss(weights=(1,0.2), out_channels="BC", mask_distance_channel=True):
    """
    Custom loss that mixed BCE and MSE depending on the ``out_channels`` variable.

    Parameters
    ----------
    weights : 2 float tuple, optional
        Weights to be applied to segmentation (binary and contours) and to distances respectively. E.g. ``(1, 0.2)``,
        ``1`` should be multipled by ``BCE`` for the first two channels and ``0.2`` to ``MSE`` for the last channel.

    out_channels : str, optional
        Channels to operate with. 

    mask_distance_channel : bool, optional
        Whether to mask the distance channel to only calculate the loss in those regions where the binary mask
        defined by B channel is present. 
    """
    binary_channels_loss = torch.nn.BCEWithLogitsLoss()
    distance_channels_loss = torch.nn.L1Loss()
    def loss(y_pred, y_true):
        if "D" in out_channels and out_channels != "Dv2":
            if mask_distance_channel:  
                D = y_pred[:,-1] * y_true[:,0]
            else:
                D = y_pred[:,-1] 
        if out_channels == "BC":
            return weights[0]*binary_channels_loss(y_pred[:,0], y_true[:,0])+\
                   weights[1]*binary_channels_loss(y_pred[:,1], y_true[:,1])
        elif out_channels == "BCM":
            return weights[0]*binary_channels_loss(y_pred[:,0], y_true[:,0])+\
                   weights[1]*binary_channels_loss(y_pred[:,1], y_true[:,1])+\
                   weights[2]*binary_channels_loss(y_pred[:,2], y_true[:,2])   
        elif out_channels == "BCD":
            return weights[0]*binary_channels_loss(y_pred[:,0], y_true[:,0])+\
                   weights[1]*binary_channels_loss(y_pred[:,1], y_true[:,1])+\
                   weights[2]*distance_channels_loss(D, y_true[:,2]) 
        elif out_channels == "BCDv2":
            return weights[0]*binary_channels_loss(y_pred[:,0], y_true[:,0])+\
                   weights[1]*binary_channels_loss(y_pred[:,1], y_true[:,1])+\
                   weights[2]*distance_channels_loss(D, y_true[:,2]) 
        elif out_channels in ["BDv2", "BD"]:
            return weights[0]*binary_channels_loss(y_pred[:,0], y_true[:,0])+\
                   weights[1]*distance_channels_loss(D, y_true[:,1])
        elif out_channels == "BP":
            return weights[0]*binary_channels_loss(y_pred[:,0], y_true[:,0])+\
                   weights[1]*binary_channels_loss(y_pred[:,1], y_true[:,1])
        # Dv2
        else:
            return weights[0]*distance_channels_loss(y_pred, y_true)
    return loss

def detection_metrics(true, pred, tolerance=10, voxel_size=(1,1,1), return_assoc=False, verbose=False):
    """Calculate detection metrics based on

       Parameters
       ----------
       true : List of list
           List containing coordinates of ground truth points. E.g. ``[[5,3,2], [4,6,7]]``.

       pred : 4D Tensor
           List containing coordinates of predicted points. E.g. ``[[5,3,2], [4,6,7]]``.

       tolerance : optional, int
           Maximum distance far away from a GT point to consider a point as a true positive.

       voxel_size : List of floats
           Weights to be multiply by each axis. Useful when dealing with anysotropic data to reduce the distance value
           on the axis with less resolution. E.g. ``(1,1,0.5)``.

       return_assoc : bool, optional
           To return two dataframes containing the gt points association and the FP. 

       verbose : bool, optional
            To print extra information.

       Returns
       -------
       metrics : List of strings
           List containing precision, accuracy and F1 between the predicted points and ground truth.
    """

    _true = np.array(true, dtype=np.float32)
    _pred = np.array(pred, dtype=np.float32)
    
    # Multiply each axis for the its real value
    for i in range(len(voxel_size)):
        _true[:,i] *= voxel_size[i]
        _pred[:,i] *= voxel_size[i]

    # Create cost matrix
    distances = distance_matrix(_pred, _true)
    n_matched = min(len(_true), len(_pred))
    costs = -(distances >= tolerance).astype(float) - distances / (2*n_matched)
    pred_ind, true_ind = linear_sum_assignment(-costs)

    TP, FP, FN = 0, 0, 0
    tag = ["FN" for x in _true]
    fp_preds = list(range(1,len(_pred)+1))
    dis = [-1 for x in _true]
    pred_id_assoc = [-1 for x in _true]

    # Analyse which associations are below the tolerance to consider them TP
    for i in range(len(pred_ind)):
        if distances[pred_ind[i],true_ind[i]] < tolerance:
            TP += 1
            tag[true_ind[i]] = "TP"
            fp_preds.remove(pred_ind[i]+1)

        dis[true_ind[i]] = distances[pred_ind[i],true_ind[i]]
        pred_id_assoc[true_ind[i]] = pred_ind[i]+1

    FN = len(_true) - TP
    FP = len(_pred) - TP

    # Create tow dataframes with the GT and prediction points association made and another one with the FPs
    if return_assoc:
        _true = np.array(true, dtype=np.float32)
        _pred = np.array(pred, dtype=np.float32)

        # Capture FP coords
        fp_coords = np.zeros((len(fp_preds),_pred.shape[-1]))
        for i in range(len(fp_preds)):
            fp_coords[i] = _pred[fp_preds[i]-1]

        # Capture prediction coords
        pred_coords = np.zeros( (len(pred_id_assoc), 3), dtype=np.float32)
        for i in range(len(pred_id_assoc)):
            if pred_id_assoc[i] != -1:
                pred_coords[i] = _pred[pred_id_assoc[i]-1]
            else:
                pred_coords[i] = [0,0,0]

        df = pd.DataFrame(zip(list(range(1,len(_true)+1)), pred_id_assoc, dis, tag, _true[...,0], 
            _true[...,1], _true[...,2], pred_coords[...,0], pred_coords[...,1], pred_coords[...,2]), 
            columns =['gt_id', 'pred_id', 'distance', 'tag', 'axis-0', 'axis-1', 'axis-2', 'pred_axis-0', 
            'pred_axis-1', 'pred_axis-2'])
        df_fp = pd.DataFrame(zip(fp_preds, fp_coords[...,0], fp_coords[...,1], fp_coords[...,2]), 
            columns =['pred_id', 'axis-0', 'axis-1', 'axis-2'])

    try:
        precision = TP/(TP+FP)
    except:
        precision = 0
    try:
        recall = TP/(TP+FN)
    except:
        recall = 0
    try:
        F1 = 2*((precision*recall)/(precision+recall))
    except:
        F1 = 0

    if verbose:
    	print("Points in ground truth: {}, Points in prediction: {}".format(len(_true), len(_pred)))
    	print("True positives: {}, False positives: {}, False negatives: {}".format(TP, FP, FN))
    
    if return_assoc:
        return ["Precision", precision, "Recall", recall, "F1", F1], df, df_fp
    else:
        return ["Precision", precision, "Recall", recall, "F1", F1]

## Loss function definition used in the paper from nature methods:
### [Chang Qiao](https://github.com/qc17-THU/DL-SR/tree/main/src) (MIT license).
class dfcan_loss(torch.nn.Module):
    def __init__(self, device, max_val=1):
        super(dfcan_loss, self).__init__()
        self.max_val = max_val
        self.mse = torch.nn.MSELoss().to(device, non_blocking=True)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=max_val).to(device, non_blocking=True)
        
    def forward(self, input, target):
        return self.mse(input, target) + 0.1*(1-self.ssim(input, target))

def n2v_loss_mse(y_pred, y_true):
    target = y_true[:,0].squeeze()
    mask = y_true[:,1].squeeze()
    loss = torch.sum(torch.square(target - y_pred.squeeze()*mask)) / torch.sum(mask)
    return loss

def MaskedAutoencoderViT_loss(y_pred, y_true, model):
    """
    imgs: [N, 3, H, W]
    pred: [N, L, p*p*3]
    mask: [N, L], 0 is keep, 1 is remove, 
    """
    target = model.patchify(imgs)
    if self.norm_pix_loss:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.e-6)**.5

    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

    loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
    return loss
