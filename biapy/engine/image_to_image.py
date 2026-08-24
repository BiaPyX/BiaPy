"""
Image-to-image workflow for BiaPy.

This module defines the Image_to_Image_Workflow class, which implements the
training, validation, and inference pipeline for image-to-image regression tasks
in BiaPy. It supports metrics such as PSNR, SSIM, FID, IS, LPIPS, and handles
data loading, model setup, predictions, and result saving for 2D and 3D images.
"""
import torch
import numpy as np
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, PearsonCorrCoef
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from typing import Dict, Optional
from numpy.typing import NDArray
import copy

from biapy.engine.metrics import SSIM_loss, W_MAE_SSIM_loss, W_MSE_SSIM_loss, loss_encapsulation, CycleGanLoss
from biapy.engine.base_workflow import Base_Workflow
from biapy.utils.misc import (
    to_pytorch_format,
    crop_border_tensor,
    MetricLogger,
    is_dist_avail_and_initialized,
    get_world_size,
)
from biapy.data.data_2D_manipulation import (
    crop_data_with_overlap,
    merge_data_with_overlap,
)
from biapy.data.data_3D_manipulation import (
    crop_3D_data_with_overlap,
    merge_3D_data_with_overlap,
)
from biapy.data.data_manipulation import save_tif
from biapy.data.norm import undo_image_norm, resolve_fixed_norm_info

class Image_to_Image_Workflow(Base_Workflow):
    """
    Image to image workflow where the goal is ..

    Parameters
    ----------
    cfg : YACS configuration
        Running configuration.

    Job_identifier : str
        Complete name of the running job.

    device : Torch device
        Device used.

    args : argpase class
        Arguments used in BiaPy's call.
    """

    def __init__(self, cfg, job_identifier, device, system_dict, args, **kwargs):
        """
        Initialize the Image_to_Image_Workflow.

        Sets up configuration, device, job identifier, and initializes
        workflow-specific attributes for image-to-image tasks.

        Parameters
        ----------
        cfg : YACS configuration
            Running configuration.
        job_identifier : str
            Complete name of the running job.
        device : torch.device
            Device used.
        args : argparse.Namespace
            Arguments used in BiaPy's call.
        **kwargs : dict
            Additional keyword arguments.
        """
        super(Image_to_Image_Workflow, self).__init__(cfg, job_identifier, device, system_dict, args, **kwargs)
        # From now on, no modification of the cfg will be allowed
        self.cfg.freeze()

        self.mask_path = cfg.DATA.TRAIN.GT_PATH
        self.is_y_mask = False

        self.norm_module["target_type"] = "image"
        self.test_norm_module["target_type"] = "image"
        if self.norm_module.get("target_norm_override") is not None:
            self.norm_module["target_norm_override"]["target_type"] = "image"
        if self.test_norm_module.get("target_norm_override") is not None:
            self.test_norm_module["target_norm_override"]["target_type"] = "image"

        # Per-study balanced validation metrics (see '_val_batch_group_ids' / '_grouped_metric_update'
        # in 'metric_calculation' below). Only relevant when MULTIPLE_RAW_ONE_TARGET_LOADER is on, where
        # each validation "sample" is one of several raw/out-of-focus versions of the same study.
        self.multiple_raw_one_target = bool(cfg.PROBLEM.IMAGE_TO_IMAGE.MULTIPLE_RAW_ONE_TARGET_LOADER)
        self._val_study_group_ids = None
        self._val_study_cursor = 0
        self._val_study_metric_sums = {}
        self._val_study_metric_counts = {}
        self._warned_balance_by_study_dist = False
        # Lazily-resolved fixed norm_info used to un-normalize predictions at test time when
        # 'DATA.NORMALIZATION.TARGET' is enabled (see 'process_test_sample').
        self._resolved_target_norm_info = None

    def define_activations_and_channels(self):
        """
        Define the activations to be applied to the model output and the channels that the model will output.

        This function must define the following variables:

        self.model_output_channels : List of int
            Number of channels for each output head of the model. E.g. [3] for a model with one head outputting 3 channels, 
            [1, 5] for a model with two heads outputting 1 and 5 channels respectively, etc.

        self.model_output_channel_info : List of str
            Information about the output channels. A value per output head of the model must be defined. 

        self.separated_class_channel : bool
            Whether if we should expect a separated output channel for classification.

        self.head_activations : List of str
            Activations to be applied to the model output. A value per output channel (not output head) of the model must be defined.
            "linear" and "ce_sigmoid" will not be applied. E.g. ["linear"] for a model with one channel, ["linear", "sigmoid"] for a
            model with two channels, etc.

        Example of a correct definition of the function for a model with two output heads: 1) the first one will be predicting foreground
        and contours; 2) the second one will classify into 3 classes the predicted objects. In this case the following definition would
        be correct:
            self.model_output_channels = [1, 3]
            self.model_output_channel_info = ["mask", "class"]
            self.separated_class_channel = True
            self.head_activations = ["ce_sigmoid", "ce_sigmoid", "ce_softmax", "ce_softmax", "ce_softmax"]
        """
        if self.cfg.PROBLEM.IMAGE_TO_IMAGE.CHANNELS_PER_HEAD_INFO != []:
            self.model_output_channels = []
            for head_channels in self.cfg.PROBLEM.IMAGE_TO_IMAGE.CHANNELS_PER_HEAD_INFO:
                self.model_output_channels.append(head_channels)
        else:
            self.model_output_channels = [self.cfg.PROBLEM.IMAGE_TO_IMAGE.OUTPUT_CHANNELS]
        
        self.model_output_channel_info = ["pred{}".format(i) for i in range(len(self.model_output_channels))]
        self.gt_channels_expected = self.cfg.PROBLEM.IMAGE_TO_IMAGE.OUTPUT_CHANNELS
        self.separated_class_channel = False
        if self.cfg.PROBLEM.IMAGE_TO_IMAGE.OUTPUT_CHANNEL_ACT != []:
            assert len(self.cfg.PROBLEM.IMAGE_TO_IMAGE.OUTPUT_CHANNEL_ACT) == self.cfg.PROBLEM.IMAGE_TO_IMAGE.OUTPUT_CHANNELS, "The number of activations defined in cfg.PROBLEM.IMAGE_TO_IMAGE.OUTPUT_CHANNEL_ACT must be the same as the number of output channels defined in cfg.PROBLEM.IMAGE_TO_IMAGE.OUTPUT_CHANNELS or cfg.PROBLEM.IMAGE_TO_IMAGE.CHANNELS_PER_HEAD_INFO"    
            self.head_activations = self.cfg.PROBLEM.IMAGE_TO_IMAGE.OUTPUT_CHANNEL_ACT
        else:
            self.head_activations = ["linear"] * self.cfg.PROBLEM.IMAGE_TO_IMAGE.OUTPUT_CHANNELS

        super().define_activations_and_channels()

    def define_metrics(self):
        """
        Define the metrics to be used during training and test.

        This function must define the following variables:

        self.train_metrics : List of functions
            Metrics to be calculated during model's training.

        self.train_metric_names : List of str
            Names of the metrics calculated during training.

        self.train_metric_best : List of str
            To know which value should be considered as the best one. Options must be: "max" or "min".

        self.test_metrics : List of functions
            Metrics to be calculated during model's test/inference.

        self.test_metric_names : List of str
            Names of the metrics calculated during test/inference.

        self.loss : Function
            Loss function used during training and test.
        """
        data_range = (0, 1) if self.cfg.DATA.NORMALIZATION.TYPE in ["div", "scale_range"] else None
        self.train_metrics = []
        self.train_metric_names = []
        self.train_metric_best = []
        for metric in list(set(self.cfg.TRAIN.METRICS)):
            if metric == "psnr":
                # No fixed 'data_range': it must reflect the *actual* range the normalized data
                # lands in, which depends on the source dtype/scale and isn't always [0, 1] or
                # [0, 255] (see 'metric_calculation'). Letting torchmetrics infer it per call from
                # the real target range keeps this consistent with the test-time PSNR metric below.
                self.train_metrics.append(PeakSignalNoiseRatio().to(self.device))
                self.train_metric_names.append("PSNR")
                self.train_metric_best.append("max")
            elif metric == "mse":
                self.train_metrics.append(MeanSquaredError().to(self.device))
                self.train_metric_names.append("MSE")
                self.train_metric_best.append("min")
            elif metric == "mae":
                self.train_metrics.append(MeanAbsoluteError().to(self.device))
                self.train_metric_names.append("MAE")
                self.train_metric_best.append("min")
            elif metric == "ssim":
                self.train_metrics.append(StructuralSimilarityIndexMeasure(data_range=data_range).to(self.device))
                self.train_metric_names.append("SSIM")
                self.train_metric_best.append("max")
            elif metric == "fid":
                self.train_metrics.append(FrechetInceptionDistance(normalize=True).to(self.device))
                self.train_metric_names.append("FID")
                self.train_metric_best.append("min")
            elif metric == "is":
                self.train_metrics.append(InceptionScore(normalize=True).to(self.device))
                self.train_metric_names.append("IS")
                self.train_metric_best.append("max")
            elif metric == "lpips":
                self.train_metrics.append(
                    LearnedPerceptualImagePatchSimilarity(net_type="squeeze", normalize=True).to(self.device)
                )
                self.train_metric_names.append("LPIPS")
                self.train_metric_best.append("min")
            elif metric == "pcc":
                self.train_metrics.append(PearsonCorrCoef().to(self.device))
                self.train_metric_names.append("PCC")
                self.train_metric_best.append("max")

        self.test_metrics = []
        self.test_metric_names = []
        for metric in list(set(self.cfg.TEST.METRICS)):
            if metric == "psnr":
                self.test_metrics.append(PeakSignalNoiseRatio().to(self.test_device))
                self.test_metric_names.append("PSNR")
            elif metric == "mse":
                self.test_metrics.append(MeanSquaredError().to(self.test_device))
                self.test_metric_names.append("MSE")
            elif metric == "mae":
                self.test_metrics.append(MeanAbsoluteError().to(self.test_device))
                self.test_metric_names.append("MAE")
            elif metric == "ssim":
                self.test_metrics.append(StructuralSimilarityIndexMeasure().to(self.test_device))
                self.test_metric_names.append("SSIM")
            elif metric == "fid":
                self.test_metrics.append(FrechetInceptionDistance(normalize=True).to(self.test_device))
                self.test_metric_names.append("FID")
            elif metric == "is":
                self.test_metrics.append(InceptionScore(normalize=True).to(self.test_device))
                self.test_metric_names.append("IS")
            elif metric == "lpips":
                self.test_metrics.append(
                    LearnedPerceptualImagePatchSimilarity(net_type="squeeze", normalize=True).to(self.test_device)
                )
                self.test_metric_names.append("LPIPS")
            elif metric == "pcc":
                self.test_metrics.append(PearsonCorrCoef().to(self.test_device))
                self.test_metric_names.append("PCC")

        if self.cfg.LOSS.TYPE == "MSE":
            self.loss = loss_encapsulation(torch.nn.MSELoss().to(self.device))
        elif self.cfg.LOSS.TYPE == "MAE":
            self.loss = loss_encapsulation(torch.nn.L1Loss().to(self.device))
        elif self.cfg.LOSS.TYPE == "SSIM":
            self.loss = SSIM_loss(data_range=data_range, device=self.device)
        elif self.cfg.LOSS.TYPE == "W_MAE_SSIM":
            self.loss = W_MAE_SSIM_loss(
                data_range=data_range,
                device=self.device,
                w_mae=self.cfg.LOSS.WEIGHTS[0],
                w_ssim=self.cfg.LOSS.WEIGHTS[1],
            )
        elif self.cfg.LOSS.TYPE == "W_MSE_SSIM":
            self.loss = W_MSE_SSIM_loss(
                data_range=data_range,
                device=self.device,
                w_mse=self.cfg.LOSS.WEIGHTS[0],
                w_ssim=self.cfg.LOSS.WEIGHTS[1],
            )
        elif self.cfg.LOSS.TYPE == "CYCLEGAN":
            self.cyclegan_loss = CycleGanLoss(cfg=self.cfg, device=self.device)
            self.loss = self.GAN_loss_wrapper
            if "loss_discriminator" not in self.loss_names:
                self.loss_names.append("loss_discriminator")

        super().define_metrics()

    def GAN_loss_wrapper(self, output, targets):
        """Mirrors ``Denoising_Workflow.NAFNetGan_loss_wrapper`` for the image-to-image workflow."""
        if isinstance(output, dict):
            pred = output["pred"]
        else:
            pred = output
        loss_g, loss_d = self.model_without_ddp.forward_loss(pred, targets, self.cyclegan_loss)
        return {"losses": [loss_g, loss_d]}

    def metric_calculation(
        self,
        output: NDArray | torch.Tensor,
        targets: NDArray | torch.Tensor,
        train: bool = True,
        metric_logger: Optional[MetricLogger] = None,
    ) -> Dict:
        """
        Calculate the metrics defined in :func:`~define_metrics` function.

        Parameters
        ----------
        output : Torch Tensor
            Prediction of the model.

        targets : Torch Tensor
            Ground truth to compare the prediction with.

        train : bool, optional
            Whether to calculate train or test metrics.

        metric_logger : MetricLogger, optional
            Class to be updated with the new metric(s) value(s) calculated.

        Returns
        -------
        out_metrics : dict
            Value of the metrics for the given prediction.
        """
        if isinstance(output, dict):
            output = output["pred"]
        if isinstance(output, np.ndarray):
            _output = to_pytorch_format(
                output.copy(),
                self.axes_order,
                self.device if train else self.test_device,
                dtype=self.loss_dtype,
            )
        else:  # torch.Tensor
            if not train:
                _output = output.clone()
            else:
                _output = output

        if isinstance(targets, np.ndarray):
            _targets = to_pytorch_format(
                targets.copy(),
                self.axes_order,
                self.device if train else self.test_device,
                dtype=self.loss_dtype,
            )
        else:  # torch.Tensor
            if not train:
                _targets = targets.clone()
            else:
                _targets = targets

        # Exclude the border region from the metric computation (see 'TEST.EVAL_BORDER_CROP').
        # Train-time patches are small already and never carry this crop.
        if not train and self.cfg.TEST.EVAL_BORDER_CROP:
            border = list(self.cfg.TEST.EVAL_BORDER_CROP)
            _output = crop_border_tensor(_output, border)
            _targets = crop_border_tensor(_targets, border)

        out_metrics = {}
        list_to_use = self.train_metrics if train else self.test_metrics
        list_names_to_use = self.train_metric_names if train else self.test_metric_names
        list_names_to_use_lower = [x.lower() for x in list_names_to_use]

        # Per-study balanced validation metrics: with MULTIPLE_RAW_ONE_TARGET_LOADER, each validation
        # "sample" batch item is one of several raw/out-of-focus versions of the same study, and studies
        # can hold anywhere from 1 to dozens of raw versions. A flat mean over samples would let studies
        # with many raw versions dominate the epoch metric, so instead we average within each study first
        # and then across studies. See '_val_batch_group_ids' / '_grouped_metric_update' below.
        balance_by_study, group_ids_this_batch = self._val_batch_group_ids(
            metric_logger=metric_logger, batch_size=_output.shape[0]
        )

        # First metrics that do not require normalization, e.g. MAE and MSE
        metrics_without_norm = ["mae", "mse"] if train else ["mae", "mse", "ssim"]
        not_norm_metrics_pos = [
            list_names_to_use_lower.index(x) for x in metrics_without_norm if x in list_names_to_use_lower
        ]
        not_norm_metrics = [list_to_use[i] for i in not_norm_metrics_pos]
        not_norm_metrics_names = [list_names_to_use_lower[i] for i in not_norm_metrics_pos]
        with torch.no_grad():
            for i, metric in enumerate(not_norm_metrics):
                m_name = not_norm_metrics_names[i]
                m_name_real = list_names_to_use[not_norm_metrics_pos[i]]
                if m_name not in ["mse", "mae", "ssim"]:
                    raise NotImplementedError

                if balance_by_study and m_name in ("mse", "mae"):
                    val = self._grouped_metric_update(
                        metric, m_name_real, _output, _targets, group_ids_this_batch, metric_logger
                    )
                else:
                    val = metric(_output.contiguous(), _targets.contiguous())
                    val = val.item() if not torch.isnan(val) else 0  # type: ignore
                    if metric_logger:
                        metric_logger.meters[m_name_real].update(val)

                out_metrics[m_name_real] = val

        # Ensure values between 0 and 1 in training. For test it is  not done as the values are calculated
        # with the original test image values and the unnormalized prediction
        if train and isinstance(_output, torch.Tensor) and isinstance(_targets, torch.Tensor):
            if self.cfg.DATA.NORMALIZATION.TYPE in ["div", "scale_range"]:
                _output = torch.clamp(_output, min=0, max=1)
                try:
                    _targets = torch.clamp(_targets, min=0, max=1)
                except Exception as e:
                    _targets = _targets.to(torch.float32).clamp(min=0, max=1)
            elif self.cfg.DATA.NORMALIZATION.TYPE == "zero_mean_unit_variance":
                _output = (_output - torch.min(_output)) / (torch.max(_output) - torch.min(_output) + 1e-8)
                _targets = (_targets - torch.min(_targets)) / (torch.max(_targets) - torch.min(_targets) + 1e-8)

        metrics_with_norm = ["ssim", "psnr", "is", "lpips", "fid"] if train else ["psnr", "is", "lpips", "fid"]
        norm_metrics_pos = [list_names_to_use_lower.index(x) for x in metrics_with_norm if x in list_names_to_use_lower]
        norm_metrics = [list_to_use[i] for i in norm_metrics_pos]
        norm_metrics_names = [list_names_to_use_lower[i] for i in norm_metrics_pos]
        with torch.no_grad():
            for i, metric in enumerate(norm_metrics):
                m_name = norm_metrics_names[i]
                m_name_real = list_names_to_use[norm_metrics_pos[i]]
                if m_name == "ssim":
                    if balance_by_study:
                        val = self._grouped_metric_update(
                            metric, m_name_real, _output, _targets, group_ids_this_batch, metric_logger
                        )
                    else:
                        val = metric(_output, _targets)
                        val = val.item() if not torch.isnan(val) else 0  # type: ignore
                        if metric_logger:
                            metric_logger.meters[m_name_real].update(val)
                    out_metrics[m_name_real] = val
                elif m_name == "psnr":
                    # 'metric' was built with the same 'data_range' as the normalized data (see
                    # 'define_metrics'), so no rescale is needed here. Rescaling by a hardcoded 255
                    # assumed normalized data always came from an 8-bit image, which silently inflated
                    # PSNR by up to ~48 dB whenever that wasn't true (e.g. sources already in [0,1]).
                    if balance_by_study:
                        val = self._grouped_metric_update(
                            metric,
                            m_name_real,
                            _output,
                            _targets,
                            group_ids_this_batch,
                            metric_logger,
                        )
                    else:
                        val = metric(_output, _targets)
                        val = val.item() if not torch.isnan(val) else 0  # type: ignore
                        if metric_logger:
                            metric_logger.meters[m_name_real].update(val)
                    out_metrics[m_name_real] = val
                elif m_name in ["is", "lpips", "fid"]:
                    # As these metrics are going to be calculated at the end we can modify _output and _targets
                    assert isinstance(_output, torch.Tensor) and isinstance(
                        _targets, torch.Tensor
                    ), "'is', 'lpips', 'fid' inputs are expected to be tensors"
                    if _output.shape[1] == 1:
                        _output = torch.cat([_output, _output, _output], dim=1)
                    if _targets.shape[1] == 1:
                        _targets = torch.cat([_targets, _targets, _targets], dim=1)

                    if m_name == "fid":
                        metric.update(_output, real=True)
                        metric.update(_targets, real=False)
                    elif m_name == "is":
                        metric.update(_targets)
                    else:  # lpips
                        metric.update(_output, _targets)
                else:
                    raise NotImplementedError

        # Pearson Correlation Coefficient. Always computed per sample (never as one batch-pooled flatten,
        # which would correlate pixels across unrelated images) - see '_per_sample_metric_mean'.
        pcc_pos = [i for i, x in enumerate(list_names_to_use_lower) if x == "pcc"]
        with torch.no_grad():
            for i in pcc_pos:
                metric = list_to_use[i]
                m_name_real = list_names_to_use[i]
                if balance_by_study:
                    val = self._grouped_metric_update(
                        metric, m_name_real, _output, _targets, group_ids_this_batch, metric_logger, flatten=True
                    )
                else:
                    val = self._per_sample_metric_mean(metric, _output, _targets)
                    if metric_logger:
                        metric_logger.meters[m_name_real].update(val)
                out_metrics[m_name_real] = val

        return out_metrics

    def _val_batch_group_ids(self, metric_logger: Optional[MetricLogger], batch_size: int):
        """
        Work out whether the current batch's metrics should be balanced per study, and, if so, the
        study id (``gt_associated_id``) of each sample in the batch, in batch order.

        Balancing is only attempted for the per-epoch validation loop, identified via
        ``self.in_val_epoch`` (set by :func:`~biapy.engine.base_workflow.Base_Workflow` around its call
        to ``train_engine.evaluate()``) - both the training and validation loops call
        ``metric_calculation`` the exact same way otherwise (same ``metric_logger``, ``train=True`` by
        omission), so ``train_one_epoch`` calls are correctly left alone. It also requires
        ``PROBLEM.IMAGE_TO_IMAGE.MULTIPLE_RAW_ONE_TARGET_LOADER`` to be enabled, and falls back to the
        flat per-image average under distributed evaluation with more than one process, since each rank
        would otherwise only see a non-contiguous shard of the study order.

        Parameters
        ----------
        metric_logger : MetricLogger, optional
            The logger passed to ``metric_calculation``; used only to tell whether this call is inside
            a batched, per-epoch loop at all (test-time calls never pass one).

        batch_size : int
            Number of samples in the current batch (``output.shape[0]``).

        Returns
        -------
        balance_by_study : bool
            Whether grouped, per-study balancing should be used for this call.

        group_ids : list of int or None
            Study id for each sample in the batch, in order. ``None`` when ``balance_by_study`` is False.
        """
        if not self.in_val_epoch or metric_logger is None or not self.multiple_raw_one_target:
            return False, None
        if getattr(self, "X_val", None) is None or not hasattr(self.X_val, "sample_list"):
            return False, None
        if is_dist_avail_and_initialized() and get_world_size() > 1:
            if not self._warned_balance_by_study_dist:
                print(
                    "WARNING: per-study balanced validation metrics are not supported under distributed "
                    "evaluation (more than one process); falling back to the flat per-image average."
                )
                self._warned_balance_by_study_dist = True
            return False, None

        if self._val_study_group_ids is None:
            gids = []
            for i, sample in enumerate(self.X_val.sample_list):
                gid = sample.get_gt_associated_id()
                gids.append(gid if gid is not None else i)
            self._val_study_group_ids = gids

        total = len(self._val_study_group_ids)
        if self._val_study_cursor >= total:
            # Starting a new validation epoch: forget the previous one's partial accumulators.
            self._val_study_cursor = 0
            self._val_study_metric_sums = {}
            self._val_study_metric_counts = {}

        start = self._val_study_cursor
        end = min(start + batch_size, total)
        group_ids = self._val_study_group_ids[start:end]
        self._val_study_cursor = end

        if len(group_ids) != batch_size:
            # Out of sync with the val dataset order (e.g. an unexpected sampler) - bail out rather
            # than risk grouping samples under the wrong study.
            return False, None

        return True, group_ids

    def _grouped_metric_update(
        self,
        metric,
        m_name_real: str,
        output: torch.Tensor,
        targets: torch.Tensor,
        group_ids: list,
        metric_logger: Optional[MetricLogger],
        scale: float = 1.0,
        flatten: bool = False,
    ) -> float:
        """
        Compute ``metric`` per sample, accumulate it under its study id, and, once the validation epoch
        is complete, fold every study's average into a single balanced mean (mean over studies of the
        mean over that study's raw/out-of-focus versions) written directly into ``metric_logger`` so
        ``evaluate()``'s returned ``global_avg`` reports the balanced value instead of a flat per-image
        one.

        Parameters
        ----------
        metric : torchmetrics.Metric
            Metric to evaluate. Called once per sample (forward() returns a value scoped to just that
            call's input, so per-sample calls do not leak into each other).

        m_name_real : str
            Metric name as registered in ``metric_logger.meters``.

        output, targets : torch.Tensor
            Batch tensors, shape ``(B, C, ...)``.

        group_ids : list of int
            Study id for each sample in the batch, in order (from :func:`_val_batch_group_ids`).

        metric_logger : MetricLogger
            Logger to update.

        scale : float, optional
            Multiplier applied to both ``output`` and ``targets`` before computing the metric (used to
            reproduce PSNR's 0-255 rescale).

        flatten : bool, optional
            Flatten each sample to a 1D vector before calling ``metric`` (needed for
            ``PearsonCorrCoef``, which expects paired 1D observations rather than a spatial tensor).

        Returns
        -------
        float
            The last per-sample value computed in this call (matches the previous per-batch return
            convention closely enough for the ``out_metrics`` dict, which isn't read for validation).
        """
        sums = self._val_study_metric_sums.setdefault(m_name_real, {})
        counts = self._val_study_metric_counts.setdefault(m_name_real, {})

        last_val = 0.0
        for b, gid in enumerate(group_ids):
            pred_b = output[b : b + 1]
            targ_b = targets[b : b + 1]
            if scale != 1.0:
                pred_b = pred_b * scale
                targ_b = targ_b * scale
            if flatten:
                v = metric(pred_b.reshape(-1), targ_b.reshape(-1))
            else:
                v = metric(pred_b.contiguous(), targ_b.contiguous())
            v = v.item() if not torch.isnan(v) else 0.0
            sums[gid] = sums.get(gid, 0.0) + v
            counts[gid] = counts.get(gid, 0) + 1
            last_val = v

        if self._val_study_cursor >= len(self._val_study_group_ids):
            # Last batch of the epoch: every study has now contributed at least one value, so finalize
            # the balanced mean and overwrite the meter with it directly.
            study_means = [sums[g] / counts[g] for g in sums]
            balanced = float(np.mean(study_means)) if study_means else 0.0
            meter = metric_logger.meters[m_name_real]
            meter.total = balanced
            meter.count = 1
            meter.deque.clear()
            meter.deque.append(balanced)
            return balanced
        else:
            # Mid-epoch: keep the meter alive with a plain running value for the progress printout;
            # it will be overwritten with the balanced value on the epoch's last batch.
            metric_logger.meters[m_name_real].update(last_val)
            return last_val

    @staticmethod
    def _per_sample_metric_mean(metric, output: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Compute ``metric`` on each sample separately (flattened to 1D) and return the plain mean across
        the batch. Used for PCC, whose value over a whole batch flattened together would correlate
        pixels across unrelated images instead of within each image, so it must never be computed as one
        batch-pooled call the way MAE/MSE/SSIM/PSNR are.

        Parameters
        ----------
        metric : torchmetrics.Metric
            Metric to evaluate (e.g. ``PearsonCorrCoef``).

        output, targets : torch.Tensor
            Batch tensors, shape ``(B, C, ...)``.

        Returns
        -------
        float
            Mean of the per-sample values.
        """
        vals = []
        for b in range(output.shape[0]):
            v = metric(output[b : b + 1].reshape(-1), targets[b : b + 1].reshape(-1))
            vals.append(v.item() if not torch.isnan(v) else 0.0)
        return float(np.mean(vals)) if vals else 0.0

    def process_test_sample(self):
        """Process a sample in the inference phase."""
        assert self.model
        # Skip processing image
        if "discard" in self.current_sample and self.current_sample["discard"]:
            return True

        original_data_shape = self.current_sample["X"].shape

        # Crop if necessary
        if self.current_sample["X"].shape[1:-1] != self.cfg.DATA.PATCH_SIZE[:-1]:
            if self.cfg.PROBLEM.NDIM == "2D":
                obj = crop_data_with_overlap(
                    self.current_sample["X"],
                    self.cfg.DATA.PATCH_SIZE,
                    data_mask=self.current_sample["Y"],
                    overlap=self.cfg.DATA.TEST.OVERLAP,
                    padding=self.cfg.DATA.TEST.PADDING,
                    verbose=self.cfg.TEST.VERBOSE,
                )
                if self.current_sample["Y"] is not None:
                    self.current_sample["X"], self.current_sample["Y"], _ = obj  # type: ignore
                else:
                    self.current_sample["X"], _ = obj  # type: ignore
                del obj
            else:
                if self.current_sample["Y"] is not None:
                    self.current_sample["Y"] = self.current_sample["Y"][0]
                if self.cfg.TEST.REDUCE_MEMORY:
                    self.current_sample["X"], _ = crop_3D_data_with_overlap(  # type: ignore
                        self.current_sample["X"][0],
                        self.cfg.DATA.PATCH_SIZE,
                        overlap=self.cfg.DATA.TEST.OVERLAP,
                        padding=self.cfg.DATA.TEST.PADDING,
                        verbose=self.cfg.TEST.VERBOSE,
                        median_padding=self.cfg.DATA.TEST.MEDIAN_PADDING,
                    )
                    self.current_sample["Y"], _ = crop_3D_data_with_overlap(  # type: ignore
                        self.current_sample["Y"],
                        self.cfg.DATA.PATCH_SIZE,
                        overlap=self.cfg.DATA.TEST.OVERLAP,
                        padding=self.cfg.DATA.TEST.PADDING,
                        verbose=self.cfg.TEST.VERBOSE,
                        median_padding=self.cfg.DATA.TEST.MEDIAN_PADDING,
                    )
                else:
                    obj = crop_3D_data_with_overlap(
                        self.current_sample["X"][0],
                        self.cfg.DATA.PATCH_SIZE,
                        data_mask=self.current_sample["Y"],
                        overlap=self.cfg.DATA.TEST.OVERLAP,
                        padding=self.cfg.DATA.TEST.PADDING,
                        verbose=self.cfg.TEST.VERBOSE,
                        median_padding=self.cfg.DATA.TEST.MEDIAN_PADDING,
                    )
                    if self.current_sample["Y"] is not None:
                        self.current_sample["X"], self.current_sample["Y"], _ = obj  # type: ignore
                    else:
                        self.current_sample["X"], _ = obj  # type: ignore
                    del obj

        pred = self.predict_batches_in_test(self.current_sample["X"], self.current_sample["Y"])
        del self.current_sample["X"]

        # Reconstruct the predictions
        if original_data_shape[1:-1] != self.cfg.DATA.PATCH_SIZE[:-1]:
            if self.cfg.PROBLEM.NDIM == "3D":
                original_data_shape = original_data_shape[1:]
            f_name = merge_data_with_overlap if self.cfg.PROBLEM.NDIM == "2D" else merge_3D_data_with_overlap

            if self.cfg.TEST.REDUCE_MEMORY:
                pred = f_name(
                    pred,
                    original_data_shape[:-1] + (pred.shape[-1],),
                    padding=self.cfg.DATA.TEST.PADDING,
                    overlap=self.cfg.DATA.TEST.OVERLAP,
                    verbose=self.cfg.TEST.VERBOSE,
                )
                self.current_sample["Y"] = f_name(
                    self.current_sample["Y"],
                    original_data_shape[:-1] + (self.current_sample["Y"].shape[-1],),
                    padding=self.cfg.DATA.TEST.PADDING,
                    overlap=self.cfg.DATA.TEST.OVERLAP,
                    verbose=self.cfg.TEST.VERBOSE,
                )
            else:
                obj = f_name(
                    pred,
                    original_data_shape[:-1] + (pred.shape[-1],),
                    data_mask=self.current_sample["Y"],
                    padding=self.cfg.DATA.TEST.PADDING,
                    overlap=self.cfg.DATA.TEST.OVERLAP,
                    verbose=self.cfg.TEST.VERBOSE,
                )
                if self.current_sample["Y"] is not None:
                    pred, self.current_sample["Y"] = obj
                else:
                    pred = obj
                del obj

            assert isinstance(pred, np.ndarray)
            if self.cfg.PROBLEM.NDIM == "3D":
                pred = np.expand_dims(pred, 0)
                if self.current_sample["Y"] is not None:
                    self.current_sample["Y"] = np.expand_dims(self.current_sample["Y"], 0)

        if self.cfg.DATA.REFLECT_TO_COMPLETE_SHAPE:
            reflected_orig_shape = (1,) + self.current_sample["reflected_orig_shape"]
            if reflected_orig_shape != pred.shape:
                if self.cfg.PROBLEM.NDIM == "2D":
                    pred = pred[:, -reflected_orig_shape[1] :, -reflected_orig_shape[2] :]  # type: ignore
                    if self.current_sample["Y"] is not None:
                        self.current_sample["Y"] = self.current_sample["Y"][
                            :, -reflected_orig_shape[1] :, -reflected_orig_shape[2] :
                        ]
                else:
                    pred = pred[
                        :,
                        -reflected_orig_shape[1] :,
                        -reflected_orig_shape[2] :,
                        -reflected_orig_shape[3] :,
                    ]  # type: ignore
                    if self.current_sample["Y"] is not None:
                        self.current_sample["Y"] = self.current_sample["Y"][
                            :,
                            -reflected_orig_shape[1] :,
                            -reflected_orig_shape[2] :,
                            -reflected_orig_shape[3] :,
                        ]

        # Undo normalization
        target_norm_override = self.test_norm_module.get("target_norm_override")
        if target_norm_override is not None:
            if getattr(self, "_resolved_target_norm_info", None) is None:
                self._resolved_target_norm_info = resolve_fixed_norm_info(
                    target_norm_override,
                    self.cfg.PROBLEM.IMAGE_TO_IMAGE.OUTPUT_CHANNELS,
                    orig_dtype=self.current_sample["X_norm"]["orig_dtype"],
                )
            adjusted_norm = self._resolved_target_norm_info
        else:
            adjusted_norm = copy.deepcopy(self.current_sample["X_norm"])
            if self.cfg.PROBLEM.IMAGE_TO_IMAGE.OUTPUT_CHANNELS != len(self.current_sample["X_norm"]["per_channel_info"]):
                for i in range(len(self.current_sample["X_norm"]["per_channel_info"]), self.cfg.PROBLEM.IMAGE_TO_IMAGE.OUTPUT_CHANNELS):
                    adjusted_norm["per_channel_info"][str(i)] = copy.deepcopy(self.current_sample["X_norm"]["per_channel_info"]["0"])

        pred = undo_image_norm(pred, adjusted_norm)
        assert isinstance(pred, np.ndarray)

        if self.return_prediction:
            self._predictions.append({"role": "raw", "data": np.array(pred)})

        # Save image
        if self.cfg.PATHS.RESULT_DIR.PER_IMAGE != "" and self.cfg.TEST.SAVE_MODEL_RAW_OUTPUT:
            if self.save_to_disk:
                save_tif(
                    pred,
                    self.cfg.PATHS.RESULT_DIR.PER_IMAGE,
                    [self.current_sample["X_filename"]],
                    verbose=self.cfg.TEST.VERBOSE,
                    meta=self.current_sample.get("img_meta"),
                )

        # Calculate metrics
        if pred.dtype == np.dtype("uint16"):
            pred = pred.astype(np.float32)

        if self.current_sample["Y"] is not None:
            if self.current_sample["Y"].dtype == np.dtype("uint16"):
                self.current_sample["Y"] = self.current_sample["Y"].astype(np.float32)

            metric_values = self.metric_calculation(output=pred, targets=self.current_sample["Y"], train=False)
            for metric in metric_values:
                if str(metric).lower() not in self.stats["merge_patches"]:
                    self.stats["merge_patches"][str(metric).lower()] = 0
                self.stats["merge_patches"][str(metric).lower()] += metric_values[metric]
                self.current_sample_metrics[str(metric).lower()] = metric_values[metric]

    def torchvision_model_call(self, in_img: torch.Tensor, is_train: bool = False) -> torch.Tensor | None:
        """
        Call a regular Pytorch model.

        Parameters
        ----------
        in_img : torch.Tensor
            Input image to pass through the model.

        is_train : bool, optional
            Whether if the call is during training or inference.

        Returns
        -------
        prediction : torch.Tensor
            Image prediction.
        """
        pass

    def after_merge_patches(self, pred):
        """
        Execute steps needed after merging all predicted patches into the original image.

        Parameters
        ----------
        pred : Torch Tensor
            Model prediction.
        """
        pass

    def after_full_image(self, pred: NDArray):
        """
        Execute steps needed after generating the prediction by supplying the entire image to the model.

        Parameters
        ----------
        pred : NDArray
            Model prediction.
        """
        pass

    def after_all_images(self):
        """Execute steps needed after predicting all images."""
        # FID, IS and LPIPS need to be computed for all the images
        if self.use_gt:
            for i, metric in enumerate(self.test_metrics):
                m_name = self.test_metric_names[i].lower()
                if m_name in ["fid", "is", "lpips"]:
                    # label = "full_image" if not self.cfg.TEST.FULL_IMG or self.cfg.PROBLEM.NDIM == "3D" else "merge_patches"
                    label = "merge_patches"
                    if m_name == "is":
                        val = metric.compute()[0]  # It returns a the mean and the std, we only need the mean
                    else:
                        val = metric.compute()
                    val = val.item() if not torch.isnan(val) else 0
                    self.stats[label][m_name] = val

        super().after_all_images()

    def after_all_chunk_prediction_workflow_process(self):
        """
        Place any code that needs to be done after predicting all patches in "by chunks" setting.
        This function is called on all ranks.
        """
        pass

    def after_all_chunk_prediction_workflow_process_master_rank(self):
        """
        Place any code that needs to be done after predicting all patches in "by chunks" setting, but only on the master rank.
        This function is called only on the master rank.
        """
        pass
    