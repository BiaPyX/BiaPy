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
from typing import Optional, List, Tuple


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
        if isinstance(y_pred, dict):
            y_pred = y_pred["pred"]

        # For those cases that are predicting 2 channels (binary case) we adapt the GT to match.
        # It's supposed to have 0 value as background and 1 as foreground
        if self.model_source == "bmz" and self.num_classes <= 2 and y_pred.shape[1] != y_true.shape[1]:
            y_true = torch.cat((1 - y_true, y_true), 1)
        else:
            if y_pred.shape[-self.ndim :] != y_true.shape[-self.ndim :]:
                y_true = scale_target(y_true, y_pred.shape[-self.ndim :])

        if self.num_classes > 2:
            if y_pred.shape[1] > 1:
                y_true = y_true.squeeze()
            if len(y_pred.shape) - 2 == len(y_true.shape):
                y_true = y_true.unsqueeze(0)

        return self.jaccard(y_pred, y_true)


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
            elif metric_names[i] == "L1 (distance channel)":
                loss_func = torch.nn.L1Loss()

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
        num_channels = _y_pred.shape[1]

        # Check multi-head
        if isinstance(y_pred, dict) and "class" in y_pred:
            num_channels +=  1
            _y_pred_class = torch.argmax(y_pred["class"], dim=1)
        else:            
            _y_pred_class = _y_pred[:, -1]

        if _y_pred.shape[-self.ndim :] != y_true.shape[-self.ndim :]:
            y_true = scale_target(y_true, _y_pred.shape[-self.ndim :])

        res_metrics = {}
        for i in range(num_channels):
            if self.metric_names[i] not in res_metrics:
                res_metrics[self.metric_names[i]] = []
            # Measure metric
            if self.metric_names[i] == "IoU (classes)":
                res_metrics[self.metric_names[i]].append(self.metric_func[i](_y_pred_class, y_true[:, 1]))
            else:
                res_metrics[self.metric_names[i]].append(self.metric_func[i](_y_pred[:, i], y_true[:, 0]))

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
    targets = targets_.clone().float()
    targets = F.interpolate(targets, size=scaled_size, mode="nearest")
    return targets.long()

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
        class_rebalance: bool = False,
        ignore_index: int = -1,
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

        class_rebalance: bool, optional
            Whether to reweight classes (inside loss function) or not.

        ignore_index : int, optional
            Value to ignore in the loss calculation. If not provided, no value will be ignored.
        """
        self.ndim = ndim
        self.model_source = model_source
        self.multihead = multihead
        self.num_classes = num_classes
        self.class_rebalance = class_rebalance
        self.ignore_index = ignore_index if ignore_index != -1 else -100  # Default ignore index for CrossEntropyLoss

        if num_classes <= 2:
            self.loss = torch.nn.BCEWithLogitsLoss()
        else:
            self.loss = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index)
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
        else:
            if _y_pred.shape[-self.ndim :] != y_true.shape[-self.ndim :]:
                y_true = scale_target(y_true, _y_pred.shape[-self.ndim :])

        if self.class_rebalance:
            if self.multihead:
                weight_mask = weight_binary_ratio(y_true[:, 0])
                loss_fn = torch.nn.BCEWithLogitsLoss(weight=weight_mask)
            else:
                weight_mask = weight_binary_ratio(y_true)
                if self.num_classes <= 2:
                    loss_fn = torch.nn.BCEWithLogitsLoss(weight=weight_mask)
                else:
                    loss_fn = torch.nn.CrossEntropyLoss(weight=weight_mask)
        else:
            loss_fn = self.loss

        if self.multihead:
            loss = loss_fn(_y_pred[:, 0], y_true[:, 0]) + self.class_channel_loss(
                _y_pred_class, y_true[:, -1].type(torch.long)
            )
        else:
            if self.num_classes <= 2:
                loss = loss_fn(_y_pred, y_true)
            else:
                loss = loss_fn(_y_pred, y_true[:, 0].type(torch.long))

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
    The loss is configurable for various output channel combinations (e.g., "BC", "BCP", "BCD", etc.)
    and can handle multi-class and multi-head settings.

    Parameters
    ----------
    weights : tuple of float, optional
        Weights to be applied to each output channel loss. E.g. (1, 0.2).
    out_channels : str, optional
        String specifying the output channels (e.g., "BC", "BCP", "BCD", etc.).
    mask_distance_channel : bool, optional
        Whether to mask the distance channel loss to only calculate it where the binary mask is present.
    n_classes : int, optional
        Number of classes for the class channel (default: 2).
    class_rebalance : bool, optional
        Whether to reweight classes inside the loss function.
    instance_type : str, optional
        Type of instance segmentation ("regular" or "synapses").
    ignore_index : int, optional
        Value to ignore in the loss calculation (default: -1).

    Usage
    -----
    loss_fn = instance_segmentation_loss(weights=(1, 0.2), out_channels="BC")
    loss = loss_fn(y_pred, y_true)
    """

    def __init__(
        self,
        weights=(1, 0.2),
        out_channels="BC",
        mask_distance_channel=True,
        n_classes=2,
        class_rebalance=False,
        instance_type="regular",
        ignore_index: int = -1,
    ):
        """
        Initialize the custom loss that mixed BCE and MSE depending on the ``out_channels`` variable.

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

        instance_type : str, optional
            Type of instances expected. Options are: ["regular", "synapses"]

        ignore_index : int, optional
            Value to ignore in the loss calculation.
        """
        assert instance_type in ["regular", "synapses"]

        self.weights = weights
        self.out_channels = out_channels
        self.mask_distance_channel = mask_distance_channel
        self.n_classes = n_classes
        self.d_channel = -2 if n_classes > 2 else -1
        self.class_rebalance = class_rebalance
        self.instance_type = instance_type
        self.ignore_index = ignore_index
        self.ignore_values = True if ignore_index != -1 else False
        self.binary_channels_loss = torch.nn.BCEWithLogitsLoss()
        self.distance_channels_loss = torch.nn.L1Loss()
        self.class_channel_loss = torch.nn.CrossEntropyLoss()

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
        extra_channels = 0
        if isinstance(y_pred, dict) and "class" in y_pred:
            _y_pred_class = y_pred["class"]
            extra_channels = 1

        if self.instance_type == "regular" and "D" in self.out_channels and self.out_channels != "Dv2":
            if self.mask_distance_channel:
                D = _y_pred[:, self.d_channel] * y_true[:, 0]
            else:
                D = _y_pred[:, self.d_channel]

        loss = 0
        if self.instance_type == "regular":
            if self.out_channels == "BC":
                assert (
                    y_true.shape[1] == 2 + extra_channels
                ), f"Seems that the GT loaded doesn't have 2 channels as expected in BC. GT shape: {y_true.shape}"
                if self.class_rebalance:
                    B_weight_mask = weight_binary_ratio(y_true[:, 0])
                    if self.ignore_values:
                        B_weight_mask = B_weight_mask * (y_true[:, 0] != self.ignore_index)
                    B_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=B_weight_mask)
                    C_weight_mask = weight_binary_ratio(y_true[:, 1])
                    if self.ignore_values:
                        C_weight_mask = C_weight_mask * (y_true[:, 1] != self.ignore_index)
                    C_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=C_weight_mask)
                else:
                    if self.ignore_values:
                        B_weight_mask = torch.ones((y_true[:, 0].shape)) * (y_true[:, 0] != self.ignore_index)
                        B_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=B_weight_mask)
                        C_weight_mask = torch.ones((y_true[:, 1].shape)) * (y_true[:, 1] != self.ignore_index)
                        C_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=C_weight_mask)
                    else:
                        B_binary_channels_loss = self.binary_channels_loss
                        C_binary_channels_loss = self.binary_channels_loss

                loss = self.weights[0] * B_binary_channels_loss(_y_pred[:, 0], y_true[:, 0]) + self.weights[
                    1
                ] * C_binary_channels_loss(_y_pred[:, 1], y_true[:, 1])

            elif self.out_channels == "BCP":
                assert (
                    y_true.shape[1] == 3 + extra_channels
                ), f"Seems that the GT loaded doesn't have 3 channels as expected in BCP. GT shape: {y_true.shape}"
                if self.class_rebalance:
                    B_weight_mask = weight_binary_ratio(y_true[:, 0])
                    if self.ignore_values:
                        B_weight_mask = B_weight_mask * (y_true[:, 0] != self.ignore_index)
                    B_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=B_weight_mask)

                    C_weight_mask = weight_binary_ratio(y_true[:, 1])
                    if self.ignore_values:
                        C_weight_mask = C_weight_mask * (y_true[:, 1] != self.ignore_index)
                    C_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=C_weight_mask)

                    P_weight_mask = weight_binary_ratio(y_true[:, 2])
                    if self.ignore_values:
                        P_weight_mask = P_weight_mask * (y_true[:, 2] != self.ignore_index)
                    P_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=P_weight_mask)
                else:
                    if self.ignore_values:
                        B_weight_mask = torch.ones((y_true[:, 0].shape)) * (y_true[:, 0] != self.ignore_index)
                        B_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=B_weight_mask)
                        C_weight_mask = torch.ones((y_true[:, 1].shape)) * (y_true[:, 1] != self.ignore_index)
                        C_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=C_weight_mask)
                        P_weight_mask = torch.ones((y_true[:, 2].shape)) * (y_true[:, 2] != self.ignore_index)
                        P_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=P_weight_mask)
                    else:
                        B_binary_channels_loss = self.binary_channels_loss
                        C_binary_channels_loss = self.binary_channels_loss
                        P_binary_channels_loss = self.binary_channels_loss

                loss = (
                    self.weights[0] * B_binary_channels_loss(_y_pred[:, 0], y_true[:, 0])
                    + self.weights[1] * C_binary_channels_loss(_y_pred[:, 1], y_true[:, 1])
                    + self.weights[2] * P_binary_channels_loss(_y_pred[:, 2], y_true[:, 2])
                )
            elif self.out_channels == "BCM":
                assert (
                    y_true.shape[1] == 3 + extra_channels
                ), f"Seems that the GT loaded doesn't have 3 channels as expected in BCM. GT shape: {y_true.shape}"
                if self.class_rebalance:
                    B_weight_mask = weight_binary_ratio(y_true[:, 0])
                    B_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=B_weight_mask)
                    C_weight_mask = weight_binary_ratio(y_true[:, 1])
                    C_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=C_weight_mask)
                    M_weight_mask = weight_binary_ratio(y_true[:, 2])
                    M_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=M_weight_mask)
                else:
                    B_binary_channels_loss = self.binary_channels_loss
                    C_binary_channels_loss = self.binary_channels_loss
                    M_binary_channels_loss = self.binary_channels_loss
                loss = (
                    self.weights[0] * B_binary_channels_loss(_y_pred[:, 0], y_true[:, 0])
                    + self.weights[1] * C_binary_channels_loss(_y_pred[:, 1], y_true[:, 1])
                    + self.weights[2] * M_binary_channels_loss(_y_pred[:, 2], y_true[:, 2])
                )
            elif self.out_channels == "BCD":
                assert (
                    y_true.shape[1] == 3 + extra_channels
                ), f"Seems that the GT loaded doesn't have 3 channels as expected in BCD. GT shape: {y_true.shape}"
                if self.class_rebalance:
                    B_weight_mask = weight_binary_ratio(y_true[:, 0])
                    B_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=B_weight_mask)
                    C_weight_mask = weight_binary_ratio(y_true[:, 1])
                    C_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=C_weight_mask)
                else:
                    B_binary_channels_loss = self.binary_channels_loss
                    C_binary_channels_loss = self.binary_channels_loss
                loss = (
                    self.weights[0] * B_binary_channels_loss(_y_pred[:, 0], y_true[:, 0])
                    + self.weights[1] * C_binary_channels_loss(_y_pred[:, 1], y_true[:, 1])
                    + self.weights[2] * self.distance_channels_loss(D, y_true[:, 2])
                )
            elif self.out_channels == "BCDv2":
                assert (
                    y_true.shape[1] == 3 + extra_channels
                ), f"Seems that the GT loaded doesn't have 3 channels as expected in BCDv2. GT shape: {y_true.shape}"
                if self.class_rebalance:
                    B_weight_mask = weight_binary_ratio(y_true[:, 0])
                    B_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=B_weight_mask)
                    C_weight_mask = weight_binary_ratio(y_true[:, 1])
                    C_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=C_weight_mask)
                else:
                    B_binary_channels_loss = self.binary_channels_loss
                    C_binary_channels_loss = self.binary_channels_loss
                loss = (
                    self.weights[0] * B_binary_channels_loss(_y_pred[:, 0], y_true[:, 0])
                    + self.weights[1] * C_binary_channels_loss(_y_pred[:, 1], y_true[:, 1])
                    + self.weights[2] * self.distance_channels_loss(D, y_true[:, 2])
                )
            elif self.out_channels in ["BDv2", "BD"]:
                assert (
                    y_true.shape[1] == 2 + extra_channels
                ), f"Seems that the GT loaded doesn't have 2 channels as expected in BD/BDv2. GT shape: {y_true.shape}"
                if self.class_rebalance:
                    B_weight_mask = weight_binary_ratio(y_true[:, 0])
                    B_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=B_weight_mask)
                else:
                    B_binary_channels_loss = self.binary_channels_loss
                loss = self.weights[0] * B_binary_channels_loss(_y_pred[:, 0], y_true[:, 0]) + self.weights[
                    1
                ] * self.distance_channels_loss(D, y_true[:, 1])
            elif self.out_channels == "BP":
                assert (
                    y_true.shape[1] == 2 + extra_channels
                ), f"Seems that the GT loaded doesn't have 2 channels as expected in BP. GT shape: {y_true.shape}"
                if self.class_rebalance:
                    B_weight_mask = weight_binary_ratio(y_true[:, 0])
                    B_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=B_weight_mask)
                    P_weight_mask = weight_binary_ratio(y_true[:, 1])
                    P_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=P_weight_mask)
                else:
                    B_binary_channels_loss = self.binary_channels_loss
                    P_binary_channels_loss = self.binary_channels_loss
                loss = self.weights[0] * B_binary_channels_loss(_y_pred[:, 0], y_true[:, 0]) + self.weights[
                    1
                ] * P_binary_channels_loss(_y_pred[:, 1], y_true[:, 1])
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
                loss = self.weights[0] * self.distance_channels_loss(_y_pred, y_true)

            if self.n_classes > 2:
                loss += self.weights[-1] * self.class_channel_loss(_y_pred_class, y_true[:, -1].type(torch.long))
        else:
            if self.out_channels == "BF":
                if self.class_rebalance:
                    B_weight_mask = weight_binary_ratio(y_true[:, 0])
                    B_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=B_weight_mask)
                else:
                    B_binary_channels_loss = self.binary_channels_loss
                loss = self.weights[0] * B_binary_channels_loss(_y_pred[:, 0], y_true[:, 0])
                # Depending the dimensions more or less channels are present (2 for 2D and 3 for 3D)
                for c in range(1, y_true.shape[1]):
                    if self.mask_distance_channel:
                        loss += self.weights[c] * self.distance_channels_loss(
                            _y_pred[:, c] * (y_true[:, c] != 0), y_true[:, c]
                        )
                    else:
                        loss += self.weights[c] * self.distance_channels_loss(_y_pred[:, c], y_true[:, c])
            elif self.out_channels == "B":
                if self.class_rebalance:
                    B_weight_mask = weight_binary_ratio(y_true[:, 0])
                    B_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=B_weight_mask)
                    BB_weight_mask = weight_binary_ratio(y_true[:, 1])
                    BB_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=BB_weight_mask)
                else:
                    B_binary_channels_loss = self.binary_channels_loss
                    BB_binary_channels_loss = self.binary_channels_loss
                loss = self.weights[0] * B_binary_channels_loss(_y_pred[:, 0], y_true[:, 0]) + self.weights[
                    1
                ] * BB_binary_channels_loss(_y_pred[:, 1], y_true[:, 1])
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
    target = y_true[:, 0].squeeze()
    mask = y_true[:, 1].squeeze()
    loss = torch.sum(torch.square(target - y_pred.squeeze() * mask)) / torch.sum(mask)
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
