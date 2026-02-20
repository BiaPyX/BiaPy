"""
Metrics and loss functions for BiaPy.

This module provides a variety of metrics and loss functions for evaluating and training
deep learning models in BiaPy. It includes implementations for Jaccard index (IoU),
Dice loss, BCE, Cross-Entropy, contrastive losses, instance segmentation losses,
detection metrics, and wrappers for SSIM, MSE, and MAE-based losses. Both PyTorch and
NumPy-based metrics are supported for 2D and 3D biomedical image analysis.
"""
import torch
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
from torchmetrics import JaccardIndex
from torchmetrics.image import StructuralSimilarityIndexMeasure
from pytorch_msssim import SSIM
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional, List, Tuple, Dict

def jaccard_index_numpy(y_true, y_pred):
    """
    Compute the Jaccard index (Intersection over Union) between ground truth and prediction.

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
    """
    Compute Jaccard index excluding the background class (first channel).

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

    TP = np.count_nonzero(y_pred[..., 1:] * y_true[..., 1:])
    FP = np.count_nonzero(y_pred[..., 1:] * (y_true[..., 1:] - 1))
    FN = np.count_nonzero((y_pred[..., 1:] - 1) * y_true[..., 1:])

    if (TP + FP + FN) == 0:
        jac = 0
    else:
        jac = TP / (TP + FP + FN)

    return jac


def weight_binary_ratio(target):
    """
    Compute a weight map to balance foreground and background pixels.

    Parameters
    ----------
    target : torch.Tensor
        Target tensor.

    Returns
    -------
    weight : torch.Tensor
        Weight map.
    """
    if torch.max(target) == torch.min(target):
        return torch.ones_like(target, dtype=torch.float32)

    # Generate weight map by balancing the foreground and background.
    min_ratio = 5e-2
    label = target.clone()  # copy of target label

    label = (label != 0).double()  # foreground

    ww = label.sum() / torch.prod(torch.tensor(label.shape, dtype=torch.double))

    ww = torch.clamp(ww, min=min_ratio, max=1 - min_ratio)

    weight_factor = max(ww, 1 - ww) / min(ww, 1 - ww)  # type: ignore

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


class jaccard_index:
    """
    Jaccard index (IoU) metric for PyTorch tensors.

    Supports binary and multiclass segmentation, with optional thresholding and ignore index.
    """

    def __init__(
        self,
        num_classes: int,
        device: torch.device,
        t: float = 0.5,
        model_source: str = "biapy",
        ndim: int = 2,
        ignore_index: int = -1,
    ):
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

        model_source : str, optional
            Source of the model. It can be "biapy", "bmz" or "torchvision".

        ndim : int, optional
            Number of dimensions of the input data. 2 for 2D images, 3 for 3D volumes.

        ignore_index : int, optional
            Value to ignore in the loss calculation. If not provided, no value will be ignored.
        """
        self.model_source = model_source
        self.loss = torch.nn.CrossEntropyLoss()
        self.device = device
        self.num_classes = num_classes
        self.t = t
        self.ndim = ndim
        self.ignore_index = ignore_index if ignore_index != -1 else None
        if self.num_classes > 2:
            self.jaccard = JaccardIndex(
                task="multiclass", threshold=self.t, num_classes=self.num_classes, ignore_index=ignore_index
            ).to(self.device, non_blocking=True)
        else:
            self.jaccard = JaccardIndex(
                task="binary", threshold=self.t, num_classes=self.num_classes, ignore_index=ignore_index
            ).to(self.device, non_blocking=True)

    def __call__(self, y_pred, y_true):
        """
        Calculate Jaccard index (intersection over union).

        Parameters
        ----------
        y_true : torch.Tensor
            Ground truth masks.

        y_pred : torch.Tensor
            Predicted masks.

        Returns
        -------
        jaccard : torch.Tensor
            Jaccard index value.
        """
        _y_pred = y_pred["pred"] if isinstance(y_pred, dict) and "pred" in y_pred else y_pred

        # For those cases that are predicting 2 channels (binary case) we adapt the GT to match.
        # It's supposed to have 0 value as background and 1 as foreground
        if self.model_source == "bmz" and self.num_classes <= 2 and _y_pred.shape[1] != y_true.shape[1]:
            y_true = torch.cat((1 - y_true, y_true), 1)

        if not isinstance(_y_pred, list):
            _y_pred = [_y_pred]

        iou = 0
        for j, pd in enumerate(_y_pred):
            _y_true = scale_target(y_true, pd.shape[-self.ndim :]) if pd.shape[-self.ndim :] != y_true.shape[-self.ndim :] else y_true

            if self.num_classes > 2:
                if pd.shape[1] > 1:
                    _y_true = _y_true.squeeze()
                if len(pd.shape) - 2 == len(_y_true.shape):
                    _y_true = _y_true.unsqueeze(0)
            iou += self.jaccard(pd, _y_true)

        return iou/len(_y_pred)


class multiple_metrics:
    """
    Compute multiple metrics for instance segmentation workflows.

    Supports IoU, L1, and other metrics for multi-head or multi-channel outputs.
    """

    def __init__(
        self,
        num_classes: int,
        metric_names: List[str],
        device: torch.device,
        out_channels: Optional[List[str]]=["F"],
        channel_extra_opts: Optional[Dict]={},
        ignore_index: int = -1,
        model_source: str = "biapy",
        ndim: int = 2,
    ):
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

        out_channels : list of str, optional
            Output channels to be predicted. E.g. ["F", "C"] for foreground and class channels.

        channel_extra_opts : dict, optional
            Additional options for each output channel (e.g., {"B": {"mask_values": True}}).

        ignore_index : int, optional
            Value to ignore in the loss calculation. If not provided, no value will be ignored.

        model_source : str, optional
            Source of the model. It can be "biapy", "bmz" or "torchvision".

        ndim : int, optional
            Number of dimensions of the input data. 2 for 2D images, 3 for 3D volumes.
        """
        self.num_classes = num_classes
        self.metric_names = metric_names
        self.device = device
        self.out_channels = out_channels.copy() if out_channels is not None else [".",]*len(metric_names)
        if self.num_classes > 2:
            self.out_channels += ["class"]
        self.out_channels = [x for x in self.out_channels if x != "We"]  # Ignore weight extra channel
        self.channel_extra_opts = channel_extra_opts
        self.model_source = model_source
        self.ignore_index = ignore_index if ignore_index != -1 else None
        self.ndim = ndim

        self.metric_func = []
        for i in range(len(metric_names)):
            if "IoU (classes)" in metric_names[i]:
                loss_func = JaccardIndex(
                    task="multiclass", threshold=0.5, num_classes=self.num_classes, ignore_index=self.ignore_index
                ).to(self.device, non_blocking=True)
            elif "IoU" in metric_names[i]:
                loss_func = JaccardIndex(
                    task="binary", threshold=0.5, num_classes=2, ignore_index=self.ignore_index
                ).to(self.device, non_blocking=True)
            elif "L1" in metric_names[i]:
                loss_func = torch.nn.L1Loss()
            else:
                raise ValueError(f"Metric {metric_names[i]} not recognized.")

            self.metric_func.append(loss_func)

    def __call__(self, y_pred, y_true):
        """
        Calculate metrics.

        Parameters
        ----------
        y_true : torch.Tensor
            Ground truth masks.

        y_pred : torch.Tensor or list of Tensors
            Prediction.

        Returns
        -------
        dict : dict
            Metrics and their values.
        """
        if isinstance(y_pred, dict):
            _y_pred = y_pred["pred"]
        else:
            _y_pred = y_pred

        # Check multi-head
        if isinstance(y_pred, dict) and "class" in y_pred:
            _y_pred_class = torch.argmax(y_pred["class"], dim=1)
        else:      
            # Just take the last channel as class prediction from the first output, which is assumed to be the main one
            if isinstance(_y_pred, list):    
                _y_pred_class = _y_pred[0][:, -1] 
            else: 
                _y_pred_class = _y_pred[:, -1]
            
        if not isinstance(_y_pred, list):
            _y_pred = [_y_pred]

        res_metrics = {}
        for pd in _y_pred:
            _y_true = scale_target(y_true, pd.shape[-self.ndim :]) if pd.shape[-self.ndim :] != y_true.shape[-self.ndim :] else y_true
 
            db_val_type = ""
            for pred_ch_start, channel in enumerate(self.out_channels):
                gt_ch_start = pred_ch_start
                if channel == "A":
                    assert self.channel_extra_opts is not None and "A" in self.channel_extra_opts, "Affinity channel options must be provided."
                    pred_ch_end = len(self.channel_extra_opts["A"].get("y_affinities", [1])) + pred_ch_start
                    gt_ch_end = pred_ch_end
                elif channel == "R":
                    assert self.channel_extra_opts is not None and "R" in self.channel_extra_opts, "Rays channel options must be provided."
                    pred_ch_end = self.channel_extra_opts["R"].get("nrays", 32) + pred_ch_start
                    gt_ch_end = pred_ch_end
                elif channel == "Db":
                    assert self.channel_extra_opts is not None and "Db" in self.channel_extra_opts, "Distance to border channel options must be provided."
                    db_val_type = self.channel_extra_opts.get("Db", {}).get("val_type", "norm")
                    if db_val_type == "discretize":
                        db_dis_bin_size = self.channel_extra_opts.get("Db", {}).get("bin_size", 0.1)
                        db_dis_K = int(round(1.0 / db_dis_bin_size))  # 10
                        db_channels = db_dis_K + 1   
                    else:
                        db_channels = 1
                    pred_ch_end = pred_ch_start + db_channels
                    gt_ch_end = pred_ch_end
                else:
                    pred_ch_end = pred_ch_start + 1
                    gt_ch_end = pred_ch_end

                if self.metric_names[pred_ch_start] not in res_metrics:
                    res_metrics[self.metric_names[pred_ch_start]] = []

                # Measure metric
                if self.metric_names[pred_ch_start] == "IoU (classes)":
                    res_metrics[self.metric_names[pred_ch_start]].append(self.metric_func[pred_ch_start](_y_pred_class, _y_true[:, 1]))
                else:
                    y_pred_slice = pd[:, pred_ch_start:pred_ch_end]
                    y_true_slice = _y_true[:, gt_ch_start:gt_ch_end].float()
                    if y_pred_slice.shape[1] != y_true_slice.shape[1] and "Db" == channel and db_val_type == "discretize":
                        y_pred_slice = torch.argmax(y_pred_slice, dim=1).unsqueeze(1).float()
                        y_true_slice = y_true_slice.float()
                    res_metrics[self.metric_names[pred_ch_start]].append(self.metric_func[pred_ch_start](y_pred_slice, y_true_slice))

        # Mean of same metric values
        for key, value in res_metrics.items():
            if len(value) > 1:
                res_metrics[key] = torch.mean(torch.as_tensor(value))
            else:
                res_metrics[key] = torch.as_tensor(value[0])
        return res_metrics


def scale_target(targets_: torch.Tensor, scaled_size: Tuple[int, ...]) -> torch.Tensor:
    """
    Scale the target masks to match the size of the predictions.

    Parameters
    ----------
    targets_ : torch.Tensor
        Ground truth masks.

    scaled_size : tuple
        Size to scale the masks to.

    Returns
    -------
    targets : torch.Tensor
        Scaled ground truth masks.
    """
    targets = F.interpolate(targets_.clone(), size=scaled_size, mode="nearest")
    return targets

class loss_encapsulation(nn.Module):
    """Just a wrapper to any other common loss deataching the prediction from the dict given by the model."""

    def __init__(self, loss):
        """
        Initialize the loss_encapsulation module.

        Parameters
        ----------
        loss : nn.Module or callable
            The loss function to wrap.
        """
        super(loss_encapsulation, self).__init__()
        self.loss = loss 

    def forward(self, inputs, targets):
        """
        Forward pass for the encapsulated loss.

        Parameters
        ----------
        inputs : torch.Tensor or dict
            Model predictions. If a dict, expects the prediction under the "pred" key.
        targets : torch.Tensor
            Ground truth targets.

        Returns
        -------
        loss : torch.Tensor
            Computed loss value.
        """
        if isinstance(inputs, dict):
            inputs = inputs["pred"]
        return self.loss(inputs, targets)
       
class CrossEntropyLoss_wrapper:
    """
    Wrapper for PyTorch's CrossEntropyLoss and BCEWithLogitsLoss.

    Supports multi-head, class rebalancing, and ignore index.
    """

    def __init__(
        self,
        num_classes: int,
        ndim: int = 2,
        multihead: bool = False,
        model_source: str = "biapy",
        class_rebalance: str = "none",
        class_weights: List[float] = [],
        ignore_index: int = -1,
        device=None,
    ):
        """
        Initialize wrapper to Pytorch's CrossEntropyLoss.

        Parameters
        ----------
        num_classes : int
            Number of classes.

        ndim : int, optional
            Number of dimensions of the input data. 2 for 2D images, 3 for 3D volumes.

        multihead : bool, optional
            For multihead predictions e.g. points + classification in detection.

        model_source : str, optional
            Source of the model. It can be "biapy", "bmz" or "torchvision".

        class_rebalance: str, optional
            Whether to reweight classes (inside loss function) or not. Options are: "none", "auto" and "manual".

        ignore_index : int, optional
            Value to ignore in the loss calculation. If not provided, no value will be ignored.
        
        device : Torch device, optional
            Using device. Most commonly "cpu" or "cuda" for GPU, but also potentially "mps".
        """
        self.ndim = ndim
        self.model_source = model_source
        self.multihead = multihead
        self.num_classes = num_classes
        self.class_rebalance = class_rebalance
        self.class_weights = None
        self.ignore_index = ignore_index if ignore_index != -1 else -100  # Default ignore index for CrossEntropyLoss
        self.device = device if device is not None else torch.device("cpu")

        # For intermediate outputs weighting
        self.gamma = 0.5

        if self.class_rebalance == "manual":
            self.class_weights = torch.tensor(class_weights, device=device, dtype=torch.float32)

        if num_classes <= 2:
            self.loss = torch.nn.BCEWithLogitsLoss()
        else:
            self.loss = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index, weight=self.class_weights)
        if self.multihead:
            self.class_channel_loss = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index)

    def __call__(self, y_pred, y_true):
        """
        Calculate CrossEntropyLoss.

        Parameters
        ----------
        y_true : torch.Tensor
            Ground truth masks.

        y_pred : torch.Tensor
            Predicted masks.

        Returns
        -------
        loss : torch.Tensor
            Loss value.
        """
        if self.multihead:
            _y_pred = y_pred["pred"]
            _y_pred_class = y_pred["class"]
            assert (
                y_true.shape[1] == 2
            ), f"In multihead setting the ground truth is expected to have 2 channels. Provided {y_true.shape}"
        else:
            _y_pred = y_pred["pred"] if isinstance(y_pred, dict) and "pred" in y_pred else y_pred

        # For those cases that are predicting 2 channels (binary case) we adapt the GT to match.
        # It's supposed to have 0 value as background and 1 as foreground
        if self.model_source == "bmz" and self.num_classes <= 2 and _y_pred.shape[1] != y_true.shape[1]:
            y_true = torch.cat((1 - y_true, y_true), 1)

        if not isinstance(_y_pred, list):
            _y_pred = [_y_pred]
            inter_output_weights = [1.0]
        else:
            w = [self.gamma**i for i in range(len(_y_pred))]
            s = sum(w)
            inter_output_weights = [x / s for x in w]

        loss = 0
        for j, pd in enumerate(_y_pred):
            _y_true = scale_target(y_true, pd.shape[-self.ndim :]) if pd.shape[-self.ndim :] != y_true.shape[-self.ndim :] else y_true

            if self.class_rebalance == "auto":
                if self.multihead:
                    weight_mask = weight_binary_ratio(_y_true[:, 0])
                    loss_fn = torch.nn.BCEWithLogitsLoss(weight=weight_mask)
                else:
                    if self.num_classes <= 2:
                        weight_mask = weight_binary_ratio(_y_true)
                        loss_fn = torch.nn.BCEWithLogitsLoss(weight=weight_mask)
                    else:
                        loss_fn = self.loss
            else:
                loss_fn = self.loss

            if self.multihead:
                _loss = loss_fn(pd[:, 0], _y_true[:, 0]) + self.class_channel_loss(
                    _y_pred_class, _y_true[:, -1].type(torch.long)
                )
            else:
                if self.num_classes <= 2:
                    _loss = loss_fn(pd, _y_true.type(torch.float32))
                else:
                    _loss = loss_fn(pd, _y_true[:, 0].type(torch.long))
            
            loss += _loss * inter_output_weights[j]

        return loss


class DiceLoss(nn.Module):
    """
    Dice loss for binary segmentation.

    Based on `Kaggle implementation <https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch>`_.
    """

    def __init__(self):
        """Initialize the DiceLoss module."""
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        """
        Compute the Dice loss.

        Parameters
        ----------
        inputs : torch.Tensor or dict
            Predicted logits. If a dict, expects the prediction under the "pred" key.
        targets : torch.Tensor
            Ground truth masks.
        smooth : float, optional
            Smoothing factor to avoid division by zero (default: 1).

        Returns
        -------
        loss : torch.Tensor
            Dice loss value (1 - Dice coefficient).
        """
        if isinstance(inputs, dict):
            inputs = inputs["pred"]
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class DiceBCELoss(nn.Module):
    """
    Combined Dice and BCE loss for binary segmentation.
     
    Based on `Kaggle implementation <https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch>`_ implementation.
    """

    def __init__(self, w_dice=0.5, w_bce=0.5):
        """
        Initialize the DiceBCELoss module.

        Parameters
        ----------
        w_dice : float, optional
            Weight for the Dice loss component (default: 0.5).
        w_bce : float, optional
            Weight for the BCE loss component (default: 0.5).
        """
        super(DiceBCELoss, self).__init__()
        self.w_dice = w_dice
        self.w_bce = w_bce

    def forward(self, inputs, targets, smooth=1):
        """
        Compute the weighted sum of Dice loss and BCE loss.

        Parameters
        ----------
        inputs : torch.Tensor or dict
            Predicted logits. If a dict, expects the prediction under the "pred" key.
        targets : torch.Tensor
            Ground truth masks.
        smooth : float, optional
            Smoothing factor to avoid division by zero (default: 1).

        Returns
        -------
        loss : torch.Tensor
            Weighted sum of Dice loss and BCE loss.
        """
        if isinstance(inputs, dict):
            inputs = inputs["pred"]
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction="mean")
        Dice_BCE = (BCE * self.w_bce) + (dice_loss * self.w_dice)

        return Dice_BCE


class ContrastCELoss(nn.Module):
    """
    Contrastive Cross Entropy Loss for semantic segmentation tasks. It mixes the main loss function and the constrastive loss.

    Parameters
    ----------
    main_loss : nn.Module
        The main loss function to be used for the segmentation task.

    ndim : int, optional
        Number of dimensions of the input data. 2 for 2D images, 3 for 3D volumes. Default is 2.

    weight : float, optional
        Weight for the contrastive loss. Default is 1.0.
        This weight is used to balance the contribution of the contrastive loss in the final loss calculation
        and can be adjusted based on the specific requirements of the task.

    ignore_index : int, optional
        Label to ignore in the loss calculation. Default is -1.
    """

    def __init__(
        self,
        main_loss: nn.Module,
        ndim: int = 2,
        weight: float = 1.0,
        ignore_index: int = -1,
    ):
        """
        Initialize the ContrastCELoss module.

        Parameters
        ----------
        main_loss : nn.Module
            The main loss function to be used for the segmentation task.

        ndim : int, optional
            Number of dimensions of the input data. 2 for 2D images, 3 for 3D volumes. Default is 2.

        weight : float, optional
            Weight for the contrastive loss. Default is 1.0.

        ignore_index : int, optional
            Label to ignore in the loss calculation. Default is -1.
        """
        super(ContrastCELoss, self).__init__()
        self.ndim = ndim
        self.main_loss = main_loss
        self.contrast_criterion = PixelContrastLoss(ignore_index=ignore_index, ndim=ndim)
        self.loss_weight = weight

    def forward(self, preds, target, with_embed=False):
        """
        Forward pass of the Contrastive Cross Entropy Loss.

        Parameters
        ----------
        preds : dict
            Dictionary containing the predictions from the model. It should contain:
            - "pred": Segmentation predictions.
            - "embed": Embedding predictions.
            - "segment_queue": Segment queues for contrastive learning.
            - "pixel_queue": Pixel queues for contrastive learning.

        target : torch.Tensor
            Ground truth segmentation masks.

        with_embed : bool, optional
            Whether to include the embedding in the loss calculation. Default is False.
        """
        assert "pred" in preds, "Segmentation prediction is missing in the input dictionary."
        assert "embed" in preds, "Embedding prediction is missing in the input dictionary."

        seg = preds["pred"]
        embedding = preds["embed"]

        segment_queue = preds["segment_queue"] if "segment_queue" in preds else None
        pixel_queue = preds["pixel_queue"] if "pixel_queue" in preds else None

        if seg.shape[-self.ndim :] != target.shape[-self.ndim :]:
            mode = "bilinear" if self.ndim == 2 else "trilinear"
            pred = F.interpolate(input=seg, size=target.shape[-self.ndim :], mode=mode, align_corners=True)
        else:
            pred = seg

        loss = self.main_loss(pred, target)

        loss_contrast = 0
        if segment_queue is not None and pixel_queue is not None:
            queue = torch.cat((segment_queue, pixel_queue), dim=1)

            # When the classes are less or equal 2 the background class channel is not added in BiaPy
            # so can't apply directly an argmax/max operation
            if seg.shape[1] <= 2:
                _, predict = seg.max(dim=1)

                if predict.ndim == 3:
                    offsets = torch.tensor([1, 2], device=seg.device).view(1, 2, 1, 1)
                else:
                    offsets = torch.tensor([1, 2], device=seg.device).view(1, 2, 1, 1, 1)
                predict = predict * offsets
                predict, _ = predict.max(dim=1)
            else:
                predict = torch.argmax(seg, 1)

            loss_contrast += self.contrast_criterion(
                embedding,
                labels=target,
                predict=predict,
                queue=queue,
            )
        else:
            loss_contrast += 0

        if with_embed:
            return loss + self.loss_weight * loss_contrast

        return loss + 0 * loss_contrast  # just a trick to avoid errors in distributed training


class PixelContrastLoss(nn.Module):
    """
    Pixel Contrastive Loss for semantic segmentation tasks.

    Supports hard anchor sampling and negative sampling for contrastive learning.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
        ignore_index: int = -1,
        max_samples: int = 1024,
        max_views: int = 1,
        ndim: int = 2,
    ):
        """
        Initialize the Pixel Contrastive Loss for semantic segmentation tasks.

        Parameters
        ----------
        temperature : float, optional
            Temperature parameter for the contrastive loss. Default is 0.07.

        base_temperature : float, optional
            Base temperature for the contrastive loss. Default is 0.07.

        ignore_index : int, optional
            Label to ignore in the loss calculation. Default is -1.

        max_samples : int, optional
            Maximum number of samples to consider for the contrastive loss. Default is 1024.

        max_views : int, optional
            Maximum number of views to consider for the contrastive loss. Default is 1.

        ndim : int, optional
            Number of dimensions of the input data. 2 for 2D images, 3 for 3D volumes. Default is 2.
        """
        super(PixelContrastLoss, self).__init__()

        self.temperature = temperature
        self.base_temperature = base_temperature
        self.ignore_index = ignore_index
        self.max_samples = max_samples
        self.max_views = max_views
        self.ndim = ndim

    def _hard_anchor_sampling(
        self, X: torch.Tensor, y_hat: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample hard anchors from the input features and their corresponding labels.

        Parameters
        ----------
        X : torch.Tensor
            Input features of shape (batch_size, num_samples, feature_dim).
            E.g. (2, 32768, 256) for a batch size of 2, 32768 samples and 256 features.

        y_hat : torch.Tensor
            Ground truth labels of shape (batch_size, num_samples).
            E.g. (2, 32768) for a batch size of 2 and 32768 samples.

        y : torch.Tensor
            Predicted labels of shape (batch_size, num_samples).
            E.g. (2, 32768) for a batch size of 2 and 32768 samples.

        Returns
        -------
        X_ : torch.Tensor
            Sampled features of shape (total_classes, self.max_views, feature_dim).
            E.g. (82, 1, 256) for 82 classes (this can vary depeding on the classes
            found in the ground truth), and 256 features.

        y_ : torch.Tensor
            Sampled labels of shape (total_classes,). E.g. (82,) for 82 classes (this
            can vary depeding on the classes found in the ground truth).
        """
        batch_size, feat_dim = X.shape[0], X.shape[-1]

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x != self.ignore_index]
            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]

            classes.append(this_classes)
            total_classes += len(this_classes)

        if total_classes == 0:
            return None, None  # type: ignore

        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)

        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()

        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]

            for cls_id in this_classes:
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    raise Exception("this shoud be never touched! {} {} {}".format(num_hard, num_easy, n_view))

                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)

                X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                y_[X_ptr] = cls_id
                X_ptr += 1

        return X_, y_

    def _sample_negative(self, Q: torch.Tensor):
        """
        Sample negative examples from the queue.

        The queue is expected to be of shape (class_num, cache_size, feat_size), where:
            - class_num is the number of classes,
            - cache_size is the number of samples per class,
            - feat_size is the size of the feature vector.

        Parameters
        ----------
        Q : torch.Tensor
            Queue of shape (class_num, cache_size, feat_size).
            E.g. (2, 60, 256) for 2 classes, 60 samples per class and 256 features.

        Returns
        -------
        X_ : torch.Tensor
            Sampled negative examples of shape (class_num * cache_size, feat_size).
            E.g. (120, 256) for 2 classes, 60 samples per class and 256 features.

        y_ : torch.Tensor

        """
        class_num, cache_size, feat_size = Q.shape

        X_ = torch.zeros((class_num * cache_size, feat_size)).float().cuda()
        y_ = torch.zeros((class_num * cache_size, 1)).float().cuda()
        sample_ptr = 0
        for ii in range(class_num):
            if ii == 0: continue
            this_q = Q[ii, :cache_size, :]

            X_[sample_ptr : sample_ptr + cache_size, ...] = this_q
            y_[sample_ptr : sample_ptr + cache_size, ...] = ii
            sample_ptr += cache_size

        return X_, y_

    def _contrastive(
        self,
        X_anchor: torch.Tensor,
        y_anchor: torch.Tensor,
        queue: Optional[torch.Tensor] = None,
    ):
        """
        Contrastive loss calculation.

        Parameters
        ----------
        X_anchor : torch.Tensor
            Anchor features of shape (total_classes, self.max_views, feature_dim).
            E.g. (82, 1, 256) for 82 classes (this can vary depeding on the classes
            found in the ground truth), and 256 features.

        y_anchor : torch.Tensor
            Anchor labels of shape (total_classes,). E.g. (82,) for 82 classes (this
            can vary depeding on the classes found in the ground truth).

        queue : torch.Tensor, optional
            Queue of negative examples of shape (class_num, cache_size, feat_size).
            E.g. (19, 10000, 256) for 19 classes, 10000 samples per class and 256 features.
            If not provided, the contrastive loss will be calculated using the anchor features only.
        """
        anchor_num, n_view = X_anchor.shape[0], X_anchor.shape[1]

        y_anchor = y_anchor.contiguous().view(-1, 1)
        anchor_count = n_view
        anchor_feature = torch.cat(torch.unbind(X_anchor, dim=1), dim=0)

        if queue is not None:
            X_contrast, y_contrast = self._sample_negative(queue)
            y_contrast = y_contrast.contiguous().view(-1, 1)
            contrast_count = 1
            contrast_feature = X_contrast
        else:
            y_contrast = y_anchor
            contrast_count = n_view
            contrast_feature = torch.cat(torch.unbind(X_anchor, dim=1), dim=0)

        mask = torch.eq(y_anchor, y_contrast.T).float().cuda()

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)

        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask

        logits_mask = torch.ones_like(mask).scatter_(1, torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(), 0)

        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def forward(
        self,
        feats: torch.Tensor,
        labels: torch.Tensor,
        predict: torch.Tensor,
        queue: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of the Pixel Contrastive Loss.

        Parameters
        ----------
        feats : torch.Tensor
            Input features of shape (batch_size, feat_size, H, W) or (batch_size, feat_size, D, H, W).
            E.g. (2, 256, 128, 256) for a batch size of 2, 256 features and a spatial size of 128x256.

        labels : torch.Tensor
            Ground truth labels of shape (batch_size, C, H, W) or (batch_size, C, D, H, W).
            E.g. (2, 1, 128, 256) for a batch size of 2, 1 channel and a
            spatial size of 128x256. 

        predict : torch.Tensor
            Predicted labels of shape (batch_size, H, W) or (batch_size, D, H, W).
            E.g. (2, 128, 256) for a batch size of 2 and a spatial size of 128x256.

        queue : torch.Tensor, optional
            Queue of negative examples of shape (class_num, cache_size, feat_size).
            E.g. (2, 60, 256) for 2 classes, 60 samples per class and 256 features.
            If not provided, the contrastive loss will be calculated using the anchor features only.

        Returns
        -------
        loss : torch.Tensor
            Contrastive loss value.
        """
        labels = torch.nn.functional.interpolate(labels.float().clone(), feats.shape[-self.ndim :], mode="nearest")

        # When working in instance segmentation the channels are more than 1 so we need to merge then into 
        # just one channel.
        if labels.shape[1] != 1:
            if labels.ndim == 4:
                offsets = torch.tensor([1, 2], device=labels.device).view(1, 2, 1, 1)
            else:
                offsets = torch.tensor([1, 2], device=labels.device).view(1, 2, 1, 1, 1)
            labels = labels * offsets
            labels, _ = labels.max(dim=1)
        # In semantic the target is already compressed into one channel
        else:  
            labels = labels.squeeze(1)
        labels = labels.long()
        
        assert labels.shape[-1] == feats.shape[-1], "Labels ({}) and features ({}) does not match in shape".format(
            labels.shape, feats.shape
        )

        batch_size = feats.shape[0]
        labels = labels.contiguous().view(batch_size, -1)
        predict = predict.contiguous().view(batch_size, -1)

        if feats.ndim == 4:
            feats = feats.permute(0, 2, 3, 1)
        else:
            feats = feats.permute(0, 2, 3, 4, 1)
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])

        
        feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict)

        if feats_ is not None and labels_ is not None:
            loss = self._contrastive(feats_, labels_, queue=queue)
            return loss
        else:
            return 0


class instance_segmentation_loss:
    """
    Custom loss for instance segmentation tasks in BiaPy.

    This loss combines different loss functions (e.g., BCE, L1, CrossEntropy) for multiple output channels,
    such as binary masks, contours, distances, and class channels. It supports class rebalancing, masking
    of distance channels, and different instance segmentation output types (e.g., "regular", "synapses").
    The loss is configurable for various output channel combinations and can handle multi-class and 
    multi-head settings.

    Parameters
    ----------
    weights : tuple of float, optional
        Weights to be applied to each output channel loss. E.g. (1, 0.2).

    out_channels : List of str, optional
        String specifying the output channels (e.g., ["F", "C"], ["B", "C", "P"], ["B","C","D"], etc.).

    losses_to_use : list of str, optional
        List of loss functions to use for each output channel (e.g., ["BCE", "MSE"]).

    channel_extra_opts : dict, optional
        Additional options for each output channel (e.g., {"D": {"mask_values": True}}).
    
    gt_channels_expected : int, optional
        Number of channels expected in the ground truth (default: 1).   

    n_classes : int, optional
        Number of classes for the class channel (default: 2).

    class_rebalance : str, optional
        Whether to reweight classes (inside loss function) or not. Options are: "none" and "auto".

    class_weights : List[float], optional
        Weights for each class to be used in the loss calculation (default: None).

    ignore_index : int, optional
        Value to ignore in the loss calculation (default: -1).

    Usage
    -----
    loss_fn = instance_segmentation_loss(weights=(1, 0.2), out_channels=["F", "C"])
    loss = loss_fn(y_pred, y_true)
    """

    def __init__(
        self,
        weights=(1, 0.2),
        ndim: int = 2,
        out_channels=["F", "C"],
        losses_to_use=[],
        channel_extra_opts={},
        gt_channels_expected: int = 1,
        n_classes=2,
        class_rebalance: str = "none",
        class_weights: List[float] = [],
        ignore_index: int = -1,
    ):
        """
        Initialize the custom loss that mixed BCE and MSE depending on the ``out_channels`` variable.

        Parameters
        ----------
        weights : 2 float tuple, optional
            Weights to be applied to segmentation (binary and contours) and to distances respectively. E.g. ``(1, 0.2)``,
            ``1`` should be multipled by ``BCE`` for the first two channels and ``0.2`` to ``MSE`` for the last channel.

        ndim : int, optional
            Number of dimensions of the input data. 2 for 2D images, 3 for 3D volumes.

        out_channels : List of str, optional
            Channels to operate with.

        channel_extra_opts : dict, optional
            Additional options for each output channel (e.g., {"B": {"mask_values": True}}).

        class_rebalance: str, optional
            Whether to reweight classes (inside loss function) or not. Options are: "none", "auto" and "manual".

        class_weights : List of float, optional
            Weights for each class to be used in the loss calculation.

        ignore_index : int, optional
            Value to ignore in the loss calculation.
        """
        self.weights = weights
        self.ndim = ndim
        self.out_channels = [x for x in out_channels if x != "We"]
        self.extra_weight_in_borders = out_channels.count("We") > 0
        self.gt_channels_expected = gt_channels_expected if not self.extra_weight_in_borders else gt_channels_expected + 1
        self.channel_extra_opts = channel_extra_opts
        self.n_classes = n_classes
        if self.n_classes > 2:
           self.gt_channels_expected += 1  # for the class channel
        self.class_rebalance = class_rebalance
        self.class_weights = class_weights if class_rebalance == "manual" else None
        self.ignore_index = ignore_index
        self.ignore_values = True if ignore_index != -1 else False
        self.losses_to_use = losses_to_use
        self.class_channel_loss = torch.nn.CrossEntropyLoss() if self.n_classes > 2 else None
        self.gamma = 0.5  # for intermediate outputs weighting

    def __call__(self, y_pred, y_true):
        """
        Calculate instance segmentation loss.

        Parameters
        ----------
        y_true : torch.Tensor
            Ground truth masks.

        y_pred : torch.Tensor or list of Tensors
            Predictions.

        Returns
        -------
        loss : torch.Tensor
            Loss value.
        """
        if isinstance(y_pred, dict):
            _y_pred = y_pred["pred"]
        else:
            _y_pred = y_pred
    
        if self.n_classes > 2 and isinstance(y_pred, dict) and "class" in y_pred:
            _y_pred_class = y_pred["class"]

        assert (y_true.shape[1] == self.gt_channels_expected), (
            "Seems that the GT loaded doesn't have {} channels as expected in {}. GT shape: {}".format(
                self.gt_channels_expected, self.out_channels, y_true.shape
            )
        )

        w_borders = None
        if self.extra_weight_in_borders:
            w_borders = y_true[:, -1]

        if not isinstance(_y_pred, list):
            _y_pred = [_y_pred]
            inter_output_weights = [1.0]
        else:
            w = [self.gamma**i for i in range(len(_y_pred))]
            s = sum(w)
            inter_output_weights = [x / s for x in w]

        loss = 0
        for idx, pd in enumerate(_y_pred):
            inter_output_loss = 0
            for i, channel in enumerate(self.out_channels):
                pred_ch_start = self.out_channels.index(channel)
                gt_ch_start = pred_ch_start
                if channel == "A":
                    pred_ch_end = len(self.channel_extra_opts["A"].get("y_affinities", [1])) + pred_ch_start
                    gt_ch_end = pred_ch_end
                elif channel == "R":
                    pred_ch_end = self.channel_extra_opts["R"].get("nrays", 32) + pred_ch_start
                    gt_ch_end = pred_ch_end
                elif channel == "Db":
                    val_type = self.channel_extra_opts.get("Db", {}).get("val_type", "norm")
                    if val_type == "discretize":
                        db_dis_bin_size = self.channel_extra_opts.get("Db", {}).get("bin_size", 0.1)
                        db_dis_K = int(round(1.0 / db_dis_bin_size))  # 10
                        db_channels = db_dis_K + 1   
                    else:
                        db_channels = 1
                    pred_ch_end = pred_ch_start + db_channels
                    gt_ch_end = pred_ch_start + 1
                else:
                    pred_ch_end = pred_ch_start + 1
                    gt_ch_end = pred_ch_end

                y_pred_slice = pd[:, pred_ch_start:pred_ch_end]
                y_true_slice = y_true[:, gt_ch_start:gt_ch_end].float()

                # element-wise mask you wanted to use (float on same device)
                mask_vals = self.channel_extra_opts.get(channel, {}).get("mask_values", False)
                mask = None
                if mask_vals:
                    mask = (y_true_slice != 0).float()

                if y_pred_slice.shape[-self.ndim :] != y_true_slice.shape[-self.ndim :]:
                    y_true_slice = scale_target(y_true_slice, y_pred_slice.shape[-self.ndim :])

                # class-rebalance / ignore_index weights for BCE
                weight = None
                if self.losses_to_use[i] in ["bce", "ce"] and channel in ["B","F","P","C","T","A","M","F_pre","F_post"]:
                    if self.class_rebalance == "auto":
                        weight = weight_binary_ratio(y_true_slice).float()
                    elif self.class_rebalance == "manual" and self.class_weights is not None:
                        weight = torch.tensor(self.class_weights, device=y_true.device).float()
                    if self.ignore_values:
                        ignore_mask = (y_true_slice != self.ignore_index).float()
                        weight = ignore_mask if weight is None else weight * ignore_mask

                # instantiate criterion with no reduction so we can mask safely
                if self.losses_to_use[i] == "bce":
                    crit = torch.nn.BCEWithLogitsLoss(weight=weight, reduction="none")
                elif self.losses_to_use[i] == "ce":
                    crit = torch.nn.CrossEntropyLoss(weight=weight, reduction="none")
                    y_true_slice = y_true_slice.long().squeeze(1)
                elif self.losses_to_use[i] in ["l1", "mae"]:
                    crit = torch.nn.L1Loss(reduction="none")
                elif self.losses_to_use[i] == "mse":
                    crit = torch.nn.MSELoss(reduction="none")
                else:
                    raise ValueError("Loss function {} not recognized".format(self.losses_to_use[i]))

                if self.losses_to_use[i] != "ce":
                    y_pred_slice = y_pred_slice.float()
                    y_true_slice = y_true_slice.float()

                loss_tensor = crit(y_pred_slice, y_true_slice)  # same shape as slice
                    
                # multiply by spatial border weights after crit
                if w_borders is not None:
                    loss_tensor = loss_tensor * w_borders

                # apply optional element mask AFTER computing the per-element loss
                if mask is not None:
                    loss_tensor = loss_tensor * mask
                    denom = mask.sum().clamp_min(1.0)
                else:
                    denom = torch.tensor(loss_tensor.numel(), device=loss_tensor.device, dtype=loss_tensor.dtype)

                channel_loss_val = loss_tensor.sum() / denom
                inter_output_loss += self.weights[i] * channel_loss_val
            
            loss += inter_output_weights[idx] * inter_output_loss

        if self.n_classes > 2 and isinstance(y_pred, dict) and "class" in y_pred:
            loss += self.weights[-1] * self.class_channel_loss(_y_pred_class, y_true[:, -1].type(torch.long))

        return loss

def detection_metrics(
    true,
    pred,
    true_classes=None,
    pred_classes=None,
    tolerance=10,
    resolution: List[int | float] = [1, 1, 1],
    bbox_to_consider=[],
    verbose=False,
):
    """
    Calculate detection metrics (precision, recall, F1) for point-based object detection.

    Parameters
    ----------
    true : List of list
        List containing coordinates of ground truth points. E.g. ``[[5,3,2], [4,6,7]]``.

    pred : 4D Tensor
        List containing coordinates of predicted points. E.g. ``[[5,3,2], [4,6,7]]``.

    true_classes : List of ints, optional
        Classes of each ground truth points.

    pred_classes : List of ints, optional
        Classes of each predicted points.

    tolerance : optional, int
        Maximum distance far away from a GT point to consider a point as a true positive.

    resolution : List of int/float
        Weights to be multiply by each axis. Useful when dealing with anysotropic data to reduce the distance value
        on the axis with less resolution. E.g. ``(1,1,0.5)``.

    bbox_to_consider : List of tuple/list, optional
        To not take into account during metric calculation to those points outside the bounding box defined with
        this variable. Order is: ``[[z_min, z_max], [y_min, y_max], [x_min, x_max]]``. For example, using an image
        of ``10x100x200`` to not take into account points on the first/last slices and with a border of ``15`` pixel
        for ``x`` and ``y`` axes, this variable could be defined as follows: ``[[1, 9], [15, 85], [15, 185]]``.

    verbose : bool, optional
        To print extra information.

    Returns
    -------
    metrics : List of strings
        List containing precision, accuracy and F1 between the predicted points and ground truth.
    """
    if len(bbox_to_consider) > 0:
        assert len(bbox_to_consider) == 3, "'bbox_to_consider' need to be of length 3"
        assert [len(x) == 2 for x in bbox_to_consider], (
            "'bbox_to_consider' needs to be a list of " "two element array/tuple. E.g. [[1,1],[15,100],[10,200]]"
        )
    if true_classes is not None and pred_classes is None:
        raise ValueError("'pred_classes' must be provided when 'true_classes' is set")

    if true_classes is not None and pred_classes is not None:
        if len(true_classes) != len(true):
            raise ValueError("'true' and 'true_classes' length must be the same")
        if len(pred_classes) != len(pred_classes):
            raise ValueError("'pred' and 'pred_classes' length must be the same")
        class_metrics = True
    else:
        class_metrics = False

    _true = np.array(true, dtype=np.float32)
    _pred = np.array(pred, dtype=np.float32)

    TP, FP, FN = 0, 0, 0
    tag = ["FN" for x in _true]
    fp_preds = list(range(1, len(_pred) + 1))
    dis = [-1 for x in _true]
    pred_id_assoc = [-1 for x in _true]

    TP_not_considered = 0
    if len(_true) > 0:
        # Multiply each axis for the its real value
        for i in range(len(resolution)):
            _true[:, i] *= resolution[i]
            _pred[:, i] *= resolution[i]

        # Create cost matrix
        distances = distance_matrix(_pred, _true)
        n_matched = min(len(_true), len(_pred))
        costs = -(distances >= tolerance).astype(float) - distances / (2 * n_matched)
        pred_ind, true_ind = linear_sum_assignment(-costs)

        # Analyse which associations are below the tolerance to consider them TP
        for i in range(len(pred_ind)):
            # Filter out those point outside the defined bounding box
            consider_point = False
            if len(bbox_to_consider) > 0:
                point = true[true_ind[i]]
                if (
                    bbox_to_consider[0][0] <= point[0] <= bbox_to_consider[0][1]
                    and bbox_to_consider[1][0] <= point[1] <= bbox_to_consider[1][1]
                    and bbox_to_consider[2][0] <= point[2] <= bbox_to_consider[2][1]
                ):
                    consider_point = True
            else:
                consider_point = True

            if distances[pred_ind[i], true_ind[i]] < tolerance:
                if consider_point:
                    TP += 1
                    tag[true_ind[i]] = "TP"
                else:
                    tag[true_ind[i]] = "NC"
                    TP_not_considered += 1
                fp_preds.remove(pred_ind[i] + 1)

            dis[true_ind[i]] = distances[pred_ind[i], true_ind[i]]
            pred_id_assoc[true_ind[i]] = pred_ind[i] + 1

        if TP_not_considered > 0:
            print(f"{TP_not_considered} TPs not considered due to filtering")
        FN = len(_true) - TP - TP_not_considered

    # FP filtering
    FP_not_considered = 0
    fp_tags = ["FP" for x in fp_preds]
    if len(bbox_to_consider) > 0:
        for i in range(len(fp_preds)):
            point = pred[fp_preds[i] - 1]
            if not (
                bbox_to_consider[0][0] <= point[0] <= bbox_to_consider[0][1]
                and bbox_to_consider[1][0] <= point[1] <= bbox_to_consider[1][1]
                and bbox_to_consider[2][0] <= point[2] <= bbox_to_consider[2][1]
            ):
                FP_not_considered += 1
                fp_tags[i] = "NC"

        print(f"{FP_not_considered} FPs not considered due to filtering")
    FP = len(fp_preds) - FP_not_considered

    # Create two dataframes with the GT and prediction points association made and another one with the FPs
    df, df_fp = None, None
    if len(_true) > 0:
        _true = np.array(true, dtype=np.float32)
        _pred = np.array(pred, dtype=np.float32)

        # Capture FP coords
        fp_coords = np.zeros((len(fp_preds), _pred.shape[-1]))
        pred_fp_class = [-1] * len(fp_preds)
        for i in range(len(fp_preds)):
            fp_coords[i] = _pred[fp_preds[i] - 1]
            if class_metrics:
                assert pred_classes is not None
                pred_fp_class[i] = int(pred_classes[fp_preds[i] - 1])

        # Capture prediction coords
        pred_coords = np.zeros((len(pred_id_assoc), 3), dtype=np.float32)
        pred_class = [-1] * len(pred_id_assoc)
        if not class_metrics:
            true_classes = [-1] * len(pred_id_assoc)
        for i in range(len(pred_id_assoc)):
            if pred_id_assoc[i] != -1:
                pred_coords[i] = _pred[pred_id_assoc[i] - 1]
                if class_metrics:
                    assert pred_classes is not None
                    pred_class[i] = int(pred_classes[pred_id_assoc[i] - 1])
            else:
                pred_coords[i] = [0, 0, 0]

        df = pd.DataFrame(
            zip(
                list(range(1, len(_true) + 1)),
                pred_id_assoc,
                dis,
                tag,
                _true[..., 0],
                _true[..., 1],
                _true[..., 2],
                true_classes,  # type: ignore
                pred_coords[..., 0],
                pred_coords[..., 1],
                pred_coords[..., 2],
                pred_class,
            ),  # type: ignore
            columns=[
                "gt_id",
                "pred_id",
                "distance",
                "tag",
                "axis-0",
                "axis-1",
                "axis-2",
                "gt_class",
                "pred_axis-0",
                "pred_axis-1",
                "pred_axis-2",
                "pred_class",
            ],
        )
        df_fp = pd.DataFrame(
            zip(
                fp_preds,
                fp_coords[..., 0],
                fp_coords[..., 1],
                fp_coords[..., 2],
                fp_tags,
                pred_fp_class,
            ),
            columns=["pred_id", "axis-0", "axis-1", "axis-2", "tag", "pred_class"],
        )

    try:
        precision = TP / (TP + FP)
    except:
        precision = 0
    try:
        recall = TP / (TP + FN)
    except:
        recall = 0
    try:
        F1 = 2 * ((precision * recall) / (precision + recall))
    except:
        F1 = 0

    if not class_metrics:
        if df is not None:
            df = df.drop(columns=["gt_class", "pred_class"])
        if df_fp is not None:
            df_fp = df_fp.drop(columns=["pred_class"])
    else:
        if df is not None:
            gt_matched_classes = df["gt_class"].tolist()
            pred_matched_classes = df["pred_class"].tolist()
            TP_classes = len([1 for x, y in zip(gt_matched_classes, pred_matched_classes) if x == y])
            FN_classes = len([1 for x, y in zip(gt_matched_classes, pred_matched_classes) if x != y])
        else:
            TP_classes = 0
            FN_classes = 0

        try:
            precision_classes = TP_classes / (TP_classes + FP)
        except:
            precision_classes = 0
        try:
            recall_classes = TP_classes / (TP_classes + FN_classes)
        except:
            recall_classes = 0
        try:
            F1_classes = 2 * ((precision_classes * recall_classes) / (precision_classes + recall_classes))
        except:
            F1_classes = 0

    if verbose:
        if len(bbox_to_consider) > 0:
            print(
                "Points in ground truth: {} ({} total but {} not considered), Points in prediction: {} "
                "({} total but {} not considered)".format(
                    len(_true),
                    len(true),
                    TP_not_considered,
                    len(_pred),
                    len(pred),
                    FP_not_considered,
                )
            )
        else:
            print("Points in ground truth: {}, Points in prediction: {}".format(len(_true), len(_pred)))
        print("True positives: {}, False positives: {}, False negatives: {}".format(int(TP), int(FP), int(FN)))
        if class_metrics:
            print("True positives (class): {}, False negatives (class): {}".format(int(TP_classes), int(FN_classes)))

    if not class_metrics:
        r_dict = {
            "Precision": precision,
            "Recall": recall,
            "F1": F1,
            "TP": int(TP),
            "FP": int(FP),
            "FN": int(FN),
        }
    else:
        r_dict = {
            "Precision": precision,
            "Recall": recall,
            "F1": F1,
            "TP": int(TP),
            "FP": int(FP),
            "FN": int(FN),
            "Precision (class)": precision_classes,
            "Recall (class)": recall_classes,
            "F1 (class)": F1_classes,
            "TP (class)": int(TP_classes),
            "FN (class)": int(FN_classes),
        }
    return r_dict, df, df_fp


class SSIM_loss(torch.nn.Module):
    """SSIM loss using torchmetrics StructuralSimilarityIndexMeasure."""

    def __init__(self, data_range, device):
        """
        Initialize the SSIM_loss module.

        Parameters
        ----------
        data_range : float
            The value range of the input images (e.g., 1.0 or 255).
        device : torch.device
            Device to use for computation.
        """
        super(SSIM_loss, self).__init__()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=data_range).to(device, non_blocking=True)

    def forward(self, input, target):
        """
        Compute the SSIM loss.

        Parameters
        ----------
        input : torch.Tensor or dict
            Predicted images. If a dict, expects the prediction under the "pred" key.
        target : torch.Tensor
            Ground truth images.

        Returns
        -------
        loss : torch.Tensor
            1 minus the SSIM value (so that lower is better).
        """
        if isinstance(input, dict):
            input = input["pred"]
        return 1 - self.ssim(input, target)


class W_MAE_SSIM_loss(torch.nn.Module):
    """
    Weighted combination of MAE and SSIM loss.

    This loss combines Mean Absolute Error (MAE) and Structural Similarity Index Measure (SSIM)
    for image regression tasks, allowing the user to balance pixel-wise and perceptual similarity.
    """

    def __init__(self, data_range, device, w_mae=0.5, w_ssim=0.5):
        """
        Initialize the W_MAE_SSIM_loss module.

        Parameters
        ----------
        data_range : float
            The value range of the input images (e.g., 1.0 or 255).
        device : torch.device
            Device to use for computation.
        w_mae : float, optional
            Weight for the MAE loss component (default: 0.5).
        w_ssim : float, optional
            Weight for the SSIM loss component (default: 0.5).
        """
        super(W_MAE_SSIM_loss, self).__init__()
        self.w_mae = w_mae
        self.w_ssim = w_ssim
        self.mse = torch.nn.L1Loss().to(device, non_blocking=True)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=data_range).to(device, non_blocking=True)

    def forward(self, input, target):
        """
        Compute the weighted sum of MAE and SSIM loss.

        Parameters
        ----------
        input : torch.Tensor or dict
            Predicted images. If a dict, expects the prediction under the "pred" key.
        target : torch.Tensor
            Ground truth images.

        Returns
        -------
        loss : torch.Tensor
            Weighted sum of MAE and (1 - SSIM) loss.
        """
        if isinstance(input, dict):
            input = input["pred"]
        return (self.mse(input, target) * self.w_mae) + ((1 - self.ssim(input, target)) * self.w_ssim)


class W_MSE_SSIM_loss(torch.nn.Module):
    """
    Weighted combination of MSE and SSIM loss.

    This loss combines Mean Squared Error (MSE) and Structural Similarity Index Measure (SSIM)
    for image regression tasks, allowing the user to balance pixel-wise and perceptual similarity.
    """

    def __init__(self, data_range, device, w_mse=0.5, w_ssim=0.5):
        """
        Initialize the W_MSE_SSIM_loss module.

        Parameters
        ----------
        data_range : float
            The value range of the input images (e.g., 1.0 or 255).
        device : torch.device
            Device to use for computation.
        w_mse : float, optional
            Weight for the MSE loss component (default: 0.5).
        w_ssim : float, optional
            Weight for the SSIM loss component (default: 0.5).
        """
        super(W_MSE_SSIM_loss, self).__init__()
        self.w_mse = w_mse
        self.w_ssim = w_ssim
        self.mse = torch.nn.MSELoss().to(device, non_blocking=True)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=data_range).to(device, non_blocking=True)

    def forward(self, input, target):
        """
        Compute the weighted sum of MSE and SSIM loss.

        Parameters
        ----------
        input : torch.Tensor or dict
            Predicted images. If a dict, expects the prediction under the "pred" key.
        target : torch.Tensor
            Ground truth images.

        Returns
        -------
        loss : torch.Tensor
            Weighted sum of MSE and (1 - SSIM) loss.
        """
        if isinstance(input, dict):
            input = input["pred"]
        return (self.mse(input, target) * self.w_mse) + ((1 - self.ssim(input, target)) * self.w_ssim)


def n2v_loss_mse(y_pred, y_true):
    """
    Noise2Void MSE loss for self-supervised denoising.

    Parameters
    ----------
    y_pred : torch.Tensor or dict
        Predicted output.
    y_true : torch.Tensor
        Ground truth and mask.

    Returns
    -------
    loss : torch.Tensor
        Loss value.
    """
    if isinstance(y_pred, dict):
        y_pred = y_pred["pred"]
    target = y_true[:, :y_pred.shape[1]]
    mask = y_true[:, y_pred.shape[1]:]
    loss = torch.sum(torch.square(target - y_pred * mask)) / torch.sum(mask)
    return loss


class SSIM_wrapper:
    """Wrapper for SSIM loss using pytorch_msssim."""

    def __init__(self):
        """Initiate wrapper to SSIM loss function."""
        self.loss = SSIM(data_range=1, size_average=True, channel=1)

    def __call__(self, y_pred, y_true):
        """
        Calculate instance segmentation loss.

        Parameters
        ----------
        y_true : torch.Tensor
            Ground truth masks.

        y_pred : torch.Tensor or list of Tensors
            Predictions.

        Returns
        -------
        loss : torch.Tensor
            Loss value.
        """
        if isinstance(y_pred, dict):
            y_pred = y_pred["pred"]
        return 1 - self.loss(y_pred, y_true)


def lovasz_hinge(logits: torch.Tensor,
                 labels: torch.Tensor,
                 per_image: bool = True,
                 ignore_index: int | None = None) -> torch.Tensor:
    """
    Single-function binary Lovász hinge loss.
    - logits: unnormalized scores, same shape as labels (e.g. (1,H,W) or (1,D,H,W))
    - labels: {0,1} or bool tensor, same shape as logits
    - per_image: average loss per-item if a batch dim exists
    - ignore_index: label value to ignore (optional)
    """
    if logits.shape != labels.shape:
        raise ValueError(f"Shape mismatch: logits {logits.shape} vs labels {labels.shape}")

    # Handle per-image averaging if a batch dim is present
    if per_image and logits.dim() >= 2 and logits.size(0) > 1:
        losses = []
        for li, yi in zip(logits, labels):
            # Flatten and optionally filter ignore_index
            l_flat = li.reshape(-1)
            y_flat = yi.to(dtype=torch.long, device=li.device).reshape(-1)
            if ignore_index is not None:
                valid = (y_flat != ignore_index)
                l_flat = l_flat[valid]
                y_flat = y_flat[valid]
            if l_flat.numel() == 0:
                continue

            # Signs in {-1,+1}, hinge errors, sort desc
            signs = y_flat.float() * 2 - 1
            errors = 1 - l_flat * signs
            errs_sorted, perm = torch.sort(errors, descending=True)
            y_sorted = y_flat[perm].float()

            # Lovász gradient (Jaccard) in-place, no helpers
            p = y_sorted.numel()
            if p == 0:
                continue
            gts = y_sorted.sum()
            inter = gts - y_sorted.cumsum(0)
            union = gts + (1 - y_sorted).cumsum(0)
            jacc = 1.0 - inter / torch.clamp_min(union, 1.0)
            if p > 1:
                jacc[1:p] = jacc[1:p] - jacc[0:p-1]

            losses.append(F.relu(errs_sorted) @ jacc)

        return (torch.stack(losses).mean() if len(losses) else logits.new_tensor(0.0))

    # Single item (or per_image=False): same steps without the loop
    l_flat = logits.reshape(-1)
    y_flat = labels.to(dtype=torch.long, device=logits.device).reshape(-1)
    if ignore_index is not None:
        valid = (y_flat != ignore_index)
        l_flat = l_flat[valid]
        y_flat = y_flat[valid]
    if l_flat.numel() == 0:
        return logits.new_tensor(0.0)

    signs = y_flat.float() * 2 - 1
    errors = 1 - l_flat * signs
    errs_sorted, perm = torch.sort(errors, descending=True)
    y_sorted = y_flat[perm].float()

    p = y_sorted.numel()
    gts = y_sorted.sum()
    inter = gts - y_sorted.cumsum(0)
    union = gts + (1 - y_sorted).cumsum(0)
    jacc = 1.0 - inter / torch.clamp_min(union, 1.0)
    if p > 1:
        jacc[1:p] = jacc[1:p] - jacc[0:p-1]

    return F.relu(errs_sorted) @ jacc


class SpatialEmbLoss(nn.Module):
    """
    Spatial Embedding Loss for 2D and 3D inspired by `EmbedSeg <https://github.com/juglab/EmbedSeg/tree/main>`__.

    Parameters
    ----------
    patch_size : List of int, optional
        Patch size used during training (used to build coordinate map buffer).  
    anisotropy : List of float or int, optional
        Anisotropy factors for each axis (z,y,x).
    ndims : int, optional
        Number of spatial dimensions (2 or 3).
    center_mode : str, optional
        Method to compute object center: "centroid" or "medoid".
    medoid_max_points : int, optional
        Maximum number of points to use when computing medoid (to avoid O(N^2) complexity).
    weights : List of float, optional
        Weights for the different loss components: [foreground, instance, variance, seed].
    """

    def __init__(
        self,
        patch_size: List[int] = [32, 1024, 1024], 
        anisotropy: List[float | int] = [1,1,1],
        ndims: int = 2, 
        center_mode: str = "centroid",      # "centroid" or "medoid"
        medoid_max_points: Optional[int] = 10000,  # cap to avoid O(N^2) on huge objects
        weights: List[float] = [1.0, 1.0, 1.0],
    ):
        super().__init__()

        self.ndims = ndims
        self.center_mode = center_mode
        self.medoid_max_points = medoid_max_points
        
        # Grid sizes (used to build the coordinate map buffer; sliced to input size on forward)
        grid_z = patch_size[0] if ndims == 3 else 1
        grid_y = patch_size[-3]
        grid_x = patch_size[-2]
        # Pixel sizes (coordinate extents)
        pixel_z = anisotropy[0] if ndims == 3 else 1
        pixel_y = anisotropy[1]
        pixel_x = anisotropy[2]

        self.weights = weights
        self.foreground_weight = self.weights[0]
        self.w_inst = self.weights[1]
        self.w_var = self.weights[2]
        self.w_seed = self.weights[3]

        # Build max-size 3D coordinate grid buffer; for 2D we will slice z=1.
        # This lets one class handle both 2D (uses x,y) and 3D (uses x,y,z).
        xm = (
            torch.linspace(0, pixel_x, grid_x)
            .view(1, 1, 1, -1)
            .expand(1, grid_z, grid_y, grid_x)
        )
        ym = (
            torch.linspace(0, pixel_y, grid_y)
            .view(1, 1, -1, 1)
            .expand(1, grid_z, grid_y, grid_x)
        )
        zm = (
            torch.linspace(0, pixel_z, grid_z)
            .view(1, -1, 1, 1)
            .expand(1, grid_z, grid_y, grid_x)
        )
        # Stack as (3, Z, Y, X); for 2D we’ll slice to (2, Y, X) at forward time.
        xyzm = torch.cat((xm, ym, zm), 0)  # (3, Z, Y, X)
        self.register_buffer("xyzm", xyzm)

    def _calculate_binary_iou(self, pred, label):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | (pred == 1)).sum()
        if not union:
            return 0
        else:
            iou = intersection.item() / union.item()
            return iou
    
    @torch.no_grad()
    def _center_from_mask(
        self,
        coords: torch.Tensor,   # (D, ...)
        in_mask: torch.Tensor,  # (1, ...)
    ) -> torch.Tensor:
        """
        Compute object center from binary mask using centroid or medoid.

        Parameters
        ----------
        coords : torch.Tensor
            Coordinate grid tensor of shape (D, ...), where D is the number of spatial dimensions
        in_mask : torch.Tensor
            Binary mask tensor of shape (1, ...), indicating the object pixels/voxels.  

        Returns
        -------
        center : torch.Tensor
            Computed center coordinates of shape (D, 1, ..., 1).
        """
        D = coords.size(0)
        # Extract coordinates of all pixels/voxels in the instance: (N, D)
        pts = coords[in_mask.expand_as(coords)].view(D, -1).t().contiguous()  # (N, D)
        if pts.numel() == 0:
            # No pixels: fall back to zeros
            return torch.zeros(D, *([1] * (coords.dim() - 1)), device=coords.device, dtype=coords.dtype)

        if self.center_mode == "centroid" or pts.shape[0] == 1:
            c = pts.mean(0)  # (D,)
        else:
            # MEDOID: minimize sum of Euclidean distances to all other points
            # Optionally sub-sample to keep cdist tractable
            if self.medoid_max_points is not None and pts.shape[0] > self.medoid_max_points:
                idx = torch.randperm(pts.shape[0], device=pts.device)[: self.medoid_max_points]
                pts_sub = pts[idx]
                dist = torch.cdist(pts_sub, pts_sub, p=2)  # (M, M)
                sums = dist.sum(dim=1)
                best = torch.argmin(sums)
                c = pts_sub[best]  # approximate medoid
            else:
                dist = torch.cdist(pts, pts, p=2)  # (N, N)
                sums = dist.sum(dim=1)
                best = torch.argmin(sums)
                c = pts[best]  # exact medoid
        return c.view(D, *([1] * (coords.dim() - 1)))  # (D, 1, ..., 1)

    def forward(
        self,
        prediction: torch.Tensor,   # (B, C, H, W) or (B, C, D, H, W)
        instances: torch.Tensor,    # (B, H, W) or (B, D, H, W)
    ) -> Tuple[torch.Tensor, float, str]:
        if prediction.dim() not in (4, 5):
            raise ValueError(
                f"Unsupported prediction tensor dimensionality {prediction.dim()}. "
                "Expected 4D (B,C,H,W) or 5D (B,C,D,H,W)."
            )

        B = prediction.size(0)
        D = prediction.dim() - 2  # number of spatial dims (2 or 3)
        assert D in (2, 3), "Only 2D or 3D supported"
        assert D == self.ndims, f"Model ndims={D} does not match loss ndims={self.ndims}"

        # Spatial sizes
        if D == 2:
            H, W = prediction.size(2), prediction.size(3)
            # coords: (2, H, W) from self.xyzm (3, Z, Y, X)
            coords = self.xyzm[:2, 0, :H, :W].contiguous()
            total_voxels = H * W
        else:
            Z, H, W = prediction.size(2), prediction.size(3), prediction.size(4)
            # coords: (3, Z, H, W)
            coords = self.xyzm[:3, :Z, :H, :W].contiguous()
            total_voxels = Z * H * W

        # Remove the extra channel dimension in instances
        instances = instances[:, 0]

        # Channel partition
        emb_ch = D                   # 2 for 2D, 3 for 3D
        sig_ch = self.ndims          # equal to D
        seed_ch = 1

        loss = prediction.new_tensor(0.0)

        for b in range(B):
            # Spatial embedding (tanh) + coordinate grid
            spatial_emb = torch.tanh(prediction[b, :emb_ch]) + coords  # (D, ...)
            sigma = prediction[b, emb_ch:emb_ch + sig_ch]              # (D, ...)
            seed_map = torch.sigmoid(
                prediction[b, emb_ch + sig_ch: emb_ch + sig_ch + seed_ch]
            )  # (1, ...)

            var_loss = prediction.new_tensor(0.0)
            instance_loss = prediction.new_tensor(0.0)
            seed_loss = prediction.new_tensor(0.0)
            iou = prediction.new_tensor(0.0)
            obj_count = 0

            instance = instances[b].unsqueeze(0)       # (1, ...)
            instance_ids = instance.unique()
            instance_ids = instance_ids[instance_ids != 0]

            # Regress background seeds to zero
            bg_mask = (instances[b] == 0).unsqueeze(0)  # (1, ...)
            if bg_mask.sum() > 0:
                seed_loss = seed_loss + torch.sum(torch.pow(seed_map[bg_mask] - 0, 2))

            for idv in instance_ids:
                in_mask = instance.eq(idv)  # (1, ...)

                center = self._center_from_mask(coords, in_mask)

                # Sigma stats on object pixels/voxels
                sigma_in = sigma[in_mask.expand_as(sigma)].view(sig_ch, -1)  # (D, N)
                s_mean = sigma_in.mean(1).view(sig_ch, 1)                    # (D, 1)

                # Variance loss (before exp), detaching mean to match originals
                var_loss = var_loss + torch.mean(torch.pow(sigma_in - s_mean.detach(), 2))

                # Distance field
                s = torch.exp(s_mean.view(sig_ch, *([1] * D)) * 10)          # (D, 1...1)
                dist = torch.exp(
                    -1 * torch.sum(torch.pow(spatial_emb - center, 2) * s, dim=0, keepdim=True)
                )  # (1, ...)

                # Instance (Lovász hinge) loss on the soft mask
                instance_loss = instance_loss + lovasz_hinge(dist * 2 - 1, in_mask)

                # Seed regression loss towards distance field (fg only)
                seed_loss = seed_loss + self.foreground_weight * torch.sum(
                    torch.pow(seed_map[in_mask] - dist[in_mask].detach(), 2)
                )

                # Measure IoU at 0.5 threshold
                iou += self._calculate_binary_iou(dist > 0.5, in_mask)

                obj_count += 1

            if obj_count > 0:
                instance_loss = instance_loss / obj_count
                var_loss = var_loss / obj_count
                iou = iou / obj_count

            seed_loss = seed_loss / total_voxels

            loss = loss + (self.w_inst * instance_loss + self.w_var * var_loss + self.w_seed * seed_loss)

        loss = loss / B
        iou = iou / B
        return loss + prediction.sum() * 0, float(iou), "IoU" # keep graph identical to originals