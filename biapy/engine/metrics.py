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
from torchvision.transforms.functional import resize
import torchvision.transforms as T
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

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


def weight_binary_ratio(target):
    if torch.max(target) == torch.min(target):
        return torch.ones_like(target, dtype=torch.float32)
    
    # Generate weight map by balancing the foreground and background.
    min_ratio = 5e-2
    label = target.clone() # copy of target label

    label = (label != 0).double()  # foreground

    ww = label.sum() / torch.prod(torch.tensor(label.shape, dtype=torch.double))
    
    ww = torch.clamp(ww, min=min_ratio, max=1-min_ratio)
    
    weight_factor = max(ww, 1-ww) / min(ww, 1-ww)
        
    # Case 1 -- Affinity Map
    # In that case, ww is large (i.e., ww > 1 - ww), which means the high weight
    # factor should be applied to background pixels.

    # Case 2 -- Contour Map
    # In that case, ww is small (i.e., ww < 1 - ww), which means the high weight
    # factor should be applied to foreground pixels.

    if ww > 1 - ww:
        # Switch when foreground is the dominant class.
        label = 1 - label
    weight = weight_factor * label + (1 - label)

    return weight.float()


class jaccard_index():
    def __init__(self,  num_classes, device, t=0.5, torchvision_models=False):
        """
        Define Jaccard index.

        Parameters
        ---------- 
        num_classes : int
            Number of classes.

        device : Torch device
            Using device. Most commonly "cpu" or "cuda" for GPU, but also potentially "mps", 
            "xpu", "xla" or "meta". 

        t : float, optional
            Threshold to be applied.

        torchvision_models : bool, optional
            Whether the workflow is using a TorchVision model or not. In that case the GT could be 
            resized and normalized, as it was done so with TorchVision preprocessing for the X data.
        """
        self.torchvision_models = torchvision_models
        self.loss = torch.nn.CrossEntropyLoss()
        self.device = device 
        self.num_classes = num_classes
        self.t = t 

        if self.num_classes > 2:
            self.jaccard = JaccardIndex(task="multiclass", threshold=self.t, num_classes=self.num_classes).to(self.device, non_blocking=True)
        else:
            self.jaccard = JaccardIndex(task="binary", threshold=self.t, num_classes=self.num_classes).to(self.device, non_blocking=True)

    def __call__(self, y_pred, y_true):
        """
        Calculate CrossEntropyLoss.

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
        # If image shape has changed due to TorchVision or BMZ preprocessing then the mask needs
        # to be resized too
        if self.torchvision_models:
            if y_pred.shape[-2:] != y_true.shape[-2:]:    
                y_true = resize(y_true, size=y_pred.shape[-2:], interpolation=T.InterpolationMode("nearest"))
            if torch.max(y_true) > 1 and self.num_classes <= 2: 
                y_true = (y_true/255).type(torch.long)
        
        if self.num_classes > 2:
            return self.jaccard(y_pred, y_true.squeeze() if y_true.shape[0] > 1 else y_true.squeeze().unsqueeze(0))
        else:
            return self.jaccard(y_pred, y_true)

class instance_metrics():
    def __init__(self, num_classes, metric_names, device, torchvision_models=False):
        """
        Define instance segmentation workflow metrics.

        Parameters
        ---------- 
        num_classes : int
            Number of classes.

        metric_names : list of str
            Names of the metrics to use.  
        
        device : Torch device
            Using device. Most commonly "cpu" or "cuda" for GPU, but also potentially "mps", 
            "xpu", "xla" or "meta". 

        torchvision_models : bool, optional
            Whether the workflow is using a TorchVision model or not. In that case the GT could be 
            resized and normalized, as it was done so with TorchVision preprocessing for the X data.
        """

        self.num_classes = num_classes
        self.metric_names = metric_names
        self.device = device 
        self.torchvision_models = torchvision_models
        
        self.jaccard = None
        self.jaccard_multi = None
        self.l1loss = None
        self.multihead = False
        self.metric_func = []
        for i in range(len(metric_names)):
            if "jaccard_index_classes" in metric_names[i] and self.jaccard_multi is None: 
                self.jaccard_multi = JaccardIndex(task="multiclass", threshold=0.5, num_classes=self.num_classes).to(self.device, non_blocking=True)
                self.multihead = True 
                loss_func = self.jaccard_multi
            elif "jaccard_index" in metric_names[i] and self.jaccard is None:
                self.jaccard = JaccardIndex(task="binary", threshold=0.5, num_classes=2).to(self.device, non_blocking=True)
                loss_func = self.jaccard
            elif metric_names[i] == "L1_distance_channel" and self.l1loss is None:   
                self.l1loss = torch.nn.L1Loss()
                loss_func = self.l1loss

            self.metric_func.append(loss_func)

    def __call__(self, y_pred, y_true):
        """
        Calculate metrics.

        Parameters
        ----------
        y_true : Tensor
            Ground truth masks.

        y_pred : Tensor or list of Tensors
            Prediction.

        Returns
        -------
        dict : dict
            Metrics and their values.
        """
        # Check multi-head 
        if isinstance(y_pred, list):
            num_channels = y_pred[0].shape[1]+1
            _y_pred = y_pred[0]
            _y_pred_class = torch.argmax(y_pred[1], axis=1)
        else:
            num_channels = y_pred.shape[1]
            _y_pred = y_pred
            assert "jaccard_index_classes" not in self.metric_names, "'jaccard_index_classes' can only be used with multi-head predictions"

        # If image shape has changed due to TorchVision or BMZ preprocessing then the mask needs
        # to be resized too
        if self.torchvision_models:
            if _y_pred.shape[-2:] != y_true.shape[-2:]:    
                y_true = resize(y_true, size=_y_pred.shape[-2:], interpolation=T.InterpolationMode("nearest"))
            if torch.max(y_true) > 1 and self.num_classes <= 2: 
                y_true = (y_true/255).type(torch.long)

        res_metrics = {}       
        for i in range(num_channels):
            if self.metric_names[i] not in res_metrics:
                res_metrics[self.metric_names[i]] = []
            # Measure metric 
            if self.metric_names[i] == "jaccard_index_classes":
                res_metrics[self.metric_names[i]].append(self.metric_func[i](_y_pred_class, y_true[:,1]))
            else:
                res_metrics[self.metric_names[i]].append(self.metric_func[i](_y_pred[:,i], y_true[:,0]))
        
        # Mean of same metric values 
        for key, value in res_metrics.items():
            if len(value) > 1:
                res_metrics[key] = torch.mean(torch.as_tensor(value))
            else:
                res_metrics[key] = torch.as_tensor(value[0])
        return res_metrics


class CrossEntropyLoss_wrapper():
    def __init__(self, num_classes, torchvision_models=False, class_rebalance=False):
        """
        Wrapper to Pytorch's CrossEntropyLoss. 

        Parameters
        ----------
        torchvision_models : bool, optional
            Whether the workflow is using a TorchVision model or not. In that case the GT could be 
            resized and normalized, as it was done so with TorchVision preprocessing for the X data.

        class_rebalance: bool, optional
            Whether to reweight classes (inside loss function) or not.
        """
        self.torchvision_models = torchvision_models
        self.num_classes = num_classes
        self.class_rebalance = class_rebalance
        if num_classes <= 2:
            self.loss = torch.nn.BCEWithLogitsLoss()
        else:
            self.loss = torch.nn.CrossEntropyLoss()

    def __call__(self, y_pred, y_true):
        """
        Calculate CrossEntropyLoss.

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
        # If image shape has changed due to TorchVision or BMZ preprocessing then the mask needs
        # to be resized too
        if self.torchvision_models:
            if y_pred.shape[-2:] != y_true.shape[-2:]:    
                y_true = resize(y_true, size=y_pred.shape[-2:], interpolation=T.InterpolationMode("nearest"))
            if torch.max(y_true) > 1 and self.num_classes <= 2: 
                y_true = (y_true/255).type(torch.float32)

        if self.class_rebalance:
            weight_mask = weight_binary_ratio(y_true)
            if num_classes <= 2:
                loss_fn = torch.nn.BCEWithLogitsLoss(weight=weight_mask)
            else:
                loss_fn = torch.nn.CrossEntropyLoss(weight=weight_mask)
        else:
            loss_fn = self.loss 

        if self.num_classes <= 2:
            return loss_fn(y_pred, y_true)
        else:
            return loss_fn(y_pred, y_true[:,0].type(torch.long))

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


class instance_segmentation_loss():
    def __init__(self, weights=(1,0.2), out_channels="BC", mask_distance_channel=True, n_classes=2, class_rebalance=False):
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

        class_rebalance: bool, optional
            Whether to reweight classes (inside loss function) or not.
        """
        self.weights = weights
        self.out_channels = out_channels
        self.mask_distance_channel = mask_distance_channel 
        self.n_classes = n_classes
        self.d_channel = -2 if n_classes > 2 else -1 
        self.class_rebalance = class_rebalance

        self.binary_channels_loss = torch.nn.BCEWithLogitsLoss()
        self.distance_channels_loss = torch.nn.L1Loss()
        self.class_channel_loss = torch.nn.CrossEntropyLoss()

    def __call__(self, y_pred, y_true):
        """
        Calculate instance segmentation loss.

        Parameters
        ----------
        y_true : Tensor
            Ground truth masks.

        y_pred : Tensor or list of Tensors
            Predictions.

        Returns
        -------
        loss : Tensor
            Loss value.
        """
        if isinstance(y_pred, list):
            _y_pred = y_pred[0]
            _y_pred_class = y_pred[1]
        else:
            _y_pred = y_pred

        if "D" in self.out_channels and self.out_channels != "Dv2":
            if self.mask_distance_channel:  
                D = _y_pred[:,self.d_channel] * y_true[:,0]
            else:
                D = _y_pred[:,self.d_channel] 

        loss = 0
        if self.out_channels == "BC":
            assert y_true.shape[1] == 2, f"Seems that the GT loaded doesn't have 2 channels as expected in BC. GT shape: {y_true.shape}"
            if self.class_rebalance:
                B_weight_mask = weight_binary_ratio(y_true[:,0])
                B_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=B_weight_mask)
                C_weight_mask = weight_binary_ratio(y_true[:,1])
                C_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=C_weight_mask)
            else:
                B_binary_channels_loss = self.binary_channels_loss
                C_binary_channels_loss = self.binary_channels_loss
            loss = self.weights[0]*B_binary_channels_loss(_y_pred[:,0], y_true[:,0])+\
                   self.weights[1]*C_binary_channels_loss(_y_pred[:,1], y_true[:,1])
        elif self.out_channels == "BCM":
            assert y_true.shape[1] == 3, f"Seems that the GT loaded doesn't have 3 channels as expected in BCM. GT shape: {y_true.shape}"
            if self.class_rebalance:
                B_weight_mask = weight_binary_ratio(y_true[:,0])
                B_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=B_weight_mask)
                C_weight_mask = weight_binary_ratio(y_true[:,1])
                C_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=C_weight_mask)
                M_weight_mask = weight_binary_ratio(y_true[:,2])
                M_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=M_weight_mask)
            else:
                B_binary_channels_loss = self.binary_channels_loss
                C_binary_channels_loss = self.binary_channels_loss
                M_binary_channels_loss = self.binary_channels_loss
            loss = self.weights[0]*B_binary_channels_loss(_y_pred[:,0], y_true[:,0])+\
                   self.weights[1]*C_binary_channels_loss(_y_pred[:,1], y_true[:,1])+\
                   self.weights[2]*M_binary_channels_loss(_y_pred[:,2], y_true[:,2])   
        elif self.out_channels == "BCD":
            assert y_true.shape[1] == 3, f"Seems that the GT loaded doesn't have 3 channels as expected in BCD. GT shape: {y_true.shape}"
            if self.class_rebalance:
                B_weight_mask = weight_binary_ratio(y_true[:,0])
                B_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=B_weight_mask)
                C_weight_mask = weight_binary_ratio(y_true[:,1])
                C_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=C_weight_mask)
            else:
                B_binary_channels_loss = self.binary_channels_loss
                C_binary_channels_loss = self.binary_channels_loss
            loss = self.weights[0]*B_binary_channels_loss(_y_pred[:,0], y_true[:,0])+\
                   self.weights[1]*C_binary_channels_loss(_y_pred[:,1], y_true[:,1])+\
                   self.weights[2]*self.distance_channels_loss(D, y_true[:,2]) 
        elif self.out_channels == "BCDv2":
            assert y_true.shape[1] == 3, f"Seems that the GT loaded doesn't have 3 channels as expected in BCDv2. GT shape: {y_true.shape}"
            if self.class_rebalance:
                B_weight_mask = weight_binary_ratio(y_true[:,0])
                B_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=B_weight_mask)
                C_weight_mask = weight_binary_ratio(y_true[:,1])
                C_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=C_weight_mask)
            else:
                B_binary_channels_loss = self.binary_channels_loss
                C_binary_channels_loss = self.binary_channels_loss
            loss = self.weights[0]*B_binary_channels_loss(_y_pred[:,0], y_true[:,0])+\
                   self.weights[1]*C_binary_channels_loss(_y_pred[:,1], y_true[:,1])+\
                   self.weights[2]*self.distance_channels_loss(D, y_true[:,2]) 
        elif self.out_channels in ["BDv2", "BD"]:
            assert y_true.shape[1] == 2, f"Seems that the GT loaded doesn't have 2 channels as expected in BD/BDv2. GT shape: {y_true.shape}"
            if self.class_rebalance:
                B_weight_mask = weight_binary_ratio(y_true[:,0])
                B_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=B_weight_mask)
            else:
                B_binary_channels_loss = self.binary_channels_loss
            loss = self.weights[0]*B_binary_channels_loss(_y_pred[:,0], y_true[:,0])+\
                   self.weights[1]*self.distance_channels_loss(D, y_true[:,1])
        elif self.out_channels == "BP":
            assert y_true.shape[1] == 2, f"Seems that the GT loaded doesn't have 2 channels as expected in BP. GT shape: {y_true.shape}"
            if self.class_rebalance:
                B_weight_mask = weight_binary_ratio(y_true[:,0])
                B_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=B_weight_mask)
                P_weight_mask = weight_binary_ratio(y_true[:,1])
                P_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=P_weight_mask)
            else:
                B_binary_channels_loss = self.binary_channels_loss
                P_binary_channels_loss = self.binary_channels_loss
            loss = self.weights[0]*B_binary_channels_loss(_y_pred[:,0], y_true[:,0])+\
                   self.weights[1]*P_binary_channels_loss(_y_pred[:,1], y_true[:,1])
        elif self.out_channels == "C":
            if self.class_rebalance:
                C_weight_mask = weight_binary_ratio(y_true)
                C_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=C_weight_mask)
            else:
                C_binary_channels_loss = self.binary_channels_loss
            loss = C_binary_channels_loss(_y_pred, y_true)
        elif self.out_channels in ["A"]:
            if self.class_rebalance:
                A_weight_mask = weight_binary_ratio(y_true)
                A_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=A_weight_mask)
            else:
                A_binary_channels_loss = self.binary_channels_loss
            loss = A_binary_channels_loss(_y_pred, y_true)
        # Dv2
        else:
            loss = self.weights[0]*self.distance_channels_loss(_y_pred, y_true)

        if self.n_classes > 2:
            loss += self.weights[-1]*self.class_channel_loss(_y_pred_class, y_true[:,-1].type(torch.long))
        
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

    TP, FP, FN = 0, 0, 0
    tag = ["FN" for x in _true]
    fp_preds = list(range(1,len(_pred)+1))
    dis = [-1 for x in _true]
    pred_id_assoc = [-1 for x in _true]    

    if len(_true) > 0:
        # Multiply each axis for the its real value
        for i in range(len(voxel_size)):
            _true[:,i] *= voxel_size[i]
            _pred[:,i] *= voxel_size[i]

        # Create cost matrix
        distances = distance_matrix(_pred, _true)
        n_matched = min(len(_true), len(_pred))
        costs = -(distances >= tolerance).astype(float) - distances / (2*n_matched)
        pred_ind, true_ind = linear_sum_assignment(-costs)

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
    df, df_fp = None, None
    if return_assoc and len(_true) > 0:
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
    
    r_dict = {"Precision": precision, "Recall": recall, "F1": F1, "TP": TP, "FP": FP, "FN": FN}
    if return_assoc:
        return r_dict, df, df_fp
    else:
        return r_dict

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


class L1_wrapper():
    def __init__(self):
        """
        Wrapper to L1 loss function. 
        """
        self.loss = torch.nn.L1Loss(reduction='mean')

    def __call__(self, y_pred, y_true):
        """
        Calculate instance segmentation loss.

        Parameters
        ----------
        y_true : Tensor
            Ground truth masks.

        y_pred : Tensor or list of Tensors
            Predictions.

        Returns
        -------
        loss : Tensor
            Loss value.
        """
        return self.loss(y_pred, y_true) 

class MSE_wrapper():
    def __init__(self):
        """
        Wrapper to MSE loss function. 
        """
        self.loss = torch.nn.MSELoss(reduction='mean')

    def __call__(self, y_pred, y_true):
        """
        Calculate instance segmentation loss.

        Parameters
        ----------
        y_true : Tensor
            Ground truth masks.

        y_pred : Tensor or list of Tensors
            Predictions.

        Returns
        -------
        loss : Tensor
            Loss value.
        """
        return self.loss(y_pred, y_true) 

class SSIM_wrapper():
    def __init__(self):
        """
        Wrapper to SSIM loss function. 
        """
        self.loss = SSIM(data_range=1, size_average=True, channel=1)

    def __call__(self, y_pred, y_true):
        """
        Calculate instance segmentation loss.

        Parameters
        ----------
        y_true : Tensor
            Ground truth masks.

        y_pred : Tensor or list of Tensors
            Predictions.

        Returns
        -------
        loss : Tensor
            Loss value.
        """
        return 1-self.loss(y_pred, y_true)