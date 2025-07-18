import os
import torch
import math
import numpy as np
from tqdm import tqdm
import torch.distributed as dist
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from typing import Dict, Optional, Tuple, Any
from numpy.typing import NDArray
from biapy.data.dataset import PatchCoords


from biapy.data.data_2D_manipulation import (
    crop_data_with_overlap,
    merge_data_with_overlap,
)
from biapy.data.data_3D_manipulation import (
    crop_3D_data_with_overlap,
    merge_3D_data_with_overlap,
)
from biapy.data.post_processing.post_processing import (
    ensemble8_2d_predictions,
    ensemble16_3d_predictions,
)
from biapy.data.data_manipulation import save_tif
from biapy.utils.misc import (
    to_pytorch_format,
    to_numpy_format,
    is_main_process,
    is_dist_avail_and_initialized,
    MetricLogger,
    os_walk_clean,
)
from biapy.engine.base_workflow import Base_Workflow
from biapy.data.pre_processing import create_ssl_source_data_masks
from biapy.engine.metrics import SSIM_loss, W_MAE_SSIM_loss, W_MSE_SSIM_loss


class Self_supervised_Workflow(Base_Workflow):
    """
    Self supervised workflow where the goal is to pretrain the backbone model by solving a so-called
    pretext task without labels. This way, the model learns a representation that can be later transferred
    to solve a downstream task in a labeled (but smaller) dataset. More details in `our documentation
    <https://biapy.readthedocs.io/en/latest/workflows/self_supervision.html>`_.

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

    def __init__(self, cfg, job_identifier, device, args, **kwargs):
        super(Self_supervised_Workflow, self).__init__(cfg, job_identifier, device, args, **kwargs)

        self.prepare_ssl_data()

        # From now on, no modification of the cfg will be allowed
        self.cfg.freeze()

        # Workflow specific training variables
        self.mask_path = None
        self.is_y_mask = False
        if cfg.PROBLEM.SELF_SUPERVISED.PRETEXT_TASK == "masking":
            self.load_Y_val = False
        else:
            self.mask_path = cfg.DATA.TRAIN.GT_PATH
            self.load_Y_val = True

        self.norm_module.mask_norm = "as_image"
        self.test_norm_module.mask_norm = "as_image"

    def define_activations_and_channels(self):
        """
        This function must define the following variables:

        self.model_output_channels : List of functions
            Metrics to be calculated during model's training.

        self.multihead : bool
            Whether if the output of the model has more than one head.

        self.activations : List of dicts
            Activations to be applied to the model output. Each dict will
            match an output channel of the model. If ':' is used the activation
            will be applied to all channels at once. "Linear" and "CE_Sigmoid"
            will not be applied. E.g. [{":": "Linear"}].
        """
        self.model_output_channels = {
            "type": "image",
            "channels": [self.cfg.DATA.PATCH_SIZE[-1]],
        }
        self.real_classes = self.model_output_channels["channels"][0]
        self.multihead = False
        self.activations = [{":": "Linear"}]

        super().define_activations_and_channels()

    def define_metrics(self):
        """
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
                self.train_metrics.append(PeakSignalNoiseRatio(data_range=(0, 255)).to(self.device))
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

        self.test_metrics = []
        self.test_metric_names = []
        for metric in list(set(self.cfg.TEST.METRICS)):
            if metric == "psnr":
                self.test_metrics.append(PeakSignalNoiseRatio().to(self.device))
                self.test_metric_names.append("PSNR")
            elif metric == "mse":
                self.test_metrics.append(MeanSquaredError().to(self.device))
                self.test_metric_names.append("MSE")
            elif metric == "mae":
                self.test_metrics.append(MeanAbsoluteError().to(self.device))
                self.test_metric_names.append("MAE")
            elif metric == "ssim":
                self.test_metrics.append(StructuralSimilarityIndexMeasure().to(self.device))
                self.test_metric_names.append("SSIM")
            elif metric == "fid":
                self.test_metrics.append(FrechetInceptionDistance(normalize=True).to(self.device))
                self.test_metric_names.append("FID")
            elif metric == "is":
                self.test_metrics.append(InceptionScore(normalize=True).to(self.device))
                self.test_metric_names.append("IS")
            elif metric == "lpips":
                self.test_metrics.append(
                    LearnedPerceptualImagePatchSimilarity(net_type="squeeze", normalize=True).to(self.device)
                )
                self.test_metric_names.append("LPIPS")

        if self.cfg.MODEL.ARCHITECTURE.lower() == "mae":
            print("Overriding 'LOSS.TYPE' to set it to MSE loss (masking patches)")
            self.loss = self.MaskedAutoencoderViT_loss_wrapper
        else:
            if self.cfg.LOSS.TYPE == "MSE":
                self.loss = torch.nn.MSELoss().to(self.device)
            elif self.cfg.LOSS.TYPE == "MAE":
                self.loss = torch.nn.L1Loss().to(self.device)
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

        super().define_metrics()

    def MaskedAutoencoderViT_loss_wrapper(self, output, targets):
        """
        Unravel MAE loss.
        """
        # Targets not used because the loss has been already calculated
        loss, pred, mask = output
        return loss

    def metric_calculation(
        self,
        output: NDArray | torch.Tensor,
        targets: NDArray | torch.Tensor,
        train: bool = True,
        metric_logger: Optional[MetricLogger] = None,
    ) -> Dict:
        """
        Execution of the metrics defined in :func:`~define_metrics` function.

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
        if self.cfg.PROBLEM.SELF_SUPERVISED.PRETEXT_TASK.lower() == "masking":
            _, _output, _ = output
            assert self.model_without_ddp
            _output = self.model_without_ddp.unpatchify(_output)
        else:
            _output = output

        if isinstance(_output, np.ndarray):
            _output = to_pytorch_format(
                _output.copy(),
                self.axes_order,
                self.device,
                dtype=self.loss_dtype,
            )
        else:  # torch.Tensor
            if not train:
                _output = _output.clone()
            else:
                _output = _output

        if isinstance(targets, np.ndarray):
            _targets = to_pytorch_format(
                targets.copy(),
                self.axes_order,
                self.device,
                dtype=self.loss_dtype,
            )
        else:  # torch.Tensor
            if not train:
                _targets = targets.clone()
            else:
                _targets = targets

        out_metrics = {}
        list_to_use = self.train_metrics if train else self.test_metrics
        list_names_to_use = self.train_metric_names if train else self.test_metric_names
        list_names_to_use_lower = [x.lower() for x in list_names_to_use]

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
                if m_name in ["mse", "mae"]:
                    val = metric(_output, _targets)
                elif m_name == "ssim":
                    val = metric(_output, _targets)
                else:
                    raise NotImplementedError

                if m_name in ["mse", "mae", "ssim", "psnr"]:
                    val = val.item() if not torch.isnan(val) else 0  # type: ignore
                    out_metrics[m_name_real] = val

                if metric_logger:
                    metric_logger.meters[m_name_real].update(val)

        # Ensure values between 0 and 1 in training. For test it is  not done as the values are calculated
        # with the original test image values and the unnormalized prediction
        if train and isinstance(_output, torch.Tensor) and isinstance(_targets, torch.Tensor):
            if self.cfg.DATA.NORMALIZATION.TYPE in ["div", "scale_range"]:
                _output = torch.clamp(_output, min=0, max=1)
                _targets = torch.clamp(_targets, min=0, max=1)
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
                    val = metric(_output, _targets)
                elif m_name == "psnr":
                    if train:
                        # Set values to be between 0-255 range so PSNR value its more meaningful
                        val = metric(_output * 255, _targets * 255)
                    else:
                        # In test the values against the original values are calculated
                        val = metric(_output, _targets)
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

                if m_name in ["mse", "mae", "ssim", "psnr"]:
                    val = val.item() if not torch.isnan(val) else 0  # type: ignore
                    out_metrics[m_name_real] = val

                if metric_logger:
                    metric_logger.meters[m_name_real].update(val)

        return out_metrics

    def prepare_targets(self, targets, batch):
        """
        Location to perform any necessary data transformations to ``targets``
        before calculating the loss.

        Parameters
        ----------
        targets : Torch Tensor
            Ground truth to compare the prediction with.

        batch : Torch Tensor
            Prediction of the model.

        Returns
        -------
        targets : Torch tensor
            Resulting targets.
        """
        if self.cfg.PROBLEM.SELF_SUPERVISED.PRETEXT_TASK == "masking":
            # Swap with original images so we can calculate PSNR metric afterwards
            return to_pytorch_format(batch, self.axes_order, self.device, dtype=self.loss_dtype)
        else:
            return to_pytorch_format(targets, self.axes_order, self.device, dtype=self.loss_dtype)

    def process_test_sample(self):
        """
        Function to process a sample in the inference phase.
        """
        assert self.model and self.model_without_ddp
        # Skip processing image
        if "discard" in self.current_sample["X"] and self.current_sample["X"]["discard"]:
            return True

        original_data_shape = self.current_sample["X"].shape

        # Crop if necessary
        if self.current_sample["X"].shape[1:-1] != self.cfg.DATA.PATCH_SIZE[:-1]:
            if self.cfg.PROBLEM.NDIM == "2D":
                self.current_sample["X"], _ = crop_data_with_overlap(  # type: ignore
                    self.current_sample["X"],
                    self.cfg.DATA.PATCH_SIZE,
                    overlap=self.cfg.DATA.TEST.OVERLAP,
                    padding=self.cfg.DATA.TEST.PADDING,
                    verbose=self.cfg.TEST.VERBOSE,
                )
            else:
                self.current_sample["X"], _ = crop_3D_data_with_overlap(  # type: ignore
                    self.current_sample["X"][0],
                    self.cfg.DATA.PATCH_SIZE,
                    overlap=self.cfg.DATA.TEST.OVERLAP,
                    padding=self.cfg.DATA.TEST.PADDING,
                    verbose=self.cfg.TEST.VERBOSE,
                    median_padding=self.cfg.DATA.TEST.MEDIAN_PADDING,
                )

        # Predict each patch
        if self.cfg.TEST.AUGMENTATION:
            for k in tqdm(range(self.current_sample["X"].shape[0]), leave=False, disable=not is_main_process()):
                if self.cfg.PROBLEM.NDIM == "2D":
                    p = ensemble8_2d_predictions(
                        self.current_sample["X"][k],
                        axes_order_back=self.axes_order_back,
                        axes_order=self.axes_order,
                        device=self.device,
                        pred_func=self.model_call_func,
                    )
                else:
                    p = ensemble16_3d_predictions(
                        self.current_sample["X"][k],
                        batch_size_value=self.cfg.TRAIN.BATCH_SIZE,
                        axes_order_back=self.axes_order_back,
                        axes_order=self.axes_order,
                        device=self.device,
                        pred_func=self.model_call_func,
                    )
                p = to_numpy_format(p, self.axes_order_back)
                if "pred" not in locals():
                    pred = np.zeros((self.current_sample["X"].shape[0],) + p.shape[1:], dtype=self.dtype)
                pred[k] = p
        else:
            l = int(math.ceil(self.current_sample["X"].shape[0] / self.cfg.TRAIN.BATCH_SIZE))
            for k in tqdm(range(l), leave=False, disable=not is_main_process()):
                top = (
                    (k + 1) * self.cfg.TRAIN.BATCH_SIZE
                    if (k + 1) * self.cfg.TRAIN.BATCH_SIZE < self.current_sample["X"].shape[0]
                    else self.current_sample["X"].shape[0]
                )
                p = self.model_call_func(
                    self.current_sample["X"][k * self.cfg.TRAIN.BATCH_SIZE : top], 
                    apply_act=False,
                )
                if self.cfg.PROBLEM.SELF_SUPERVISED.PRETEXT_TASK == "masking":
                    loss, p, mask = p
                    p = self.apply_model_activations(p)
                    p, m, pv = self.model.save_images(
                        to_pytorch_format(
                            self.current_sample["X"][k * self.cfg.TRAIN.BATCH_SIZE : top],
                            self.axes_order,
                            self.device,
                        ),
                        p,
                        mask,
                        self.dtype,
                    )
                else:
                    p = self.apply_model_activations(p)
                    p = to_numpy_format(p, self.axes_order_back)

                if "pred" not in locals():
                    pred = np.zeros((self.current_sample["X"].shape[0],) + p.shape[1:], dtype=self.dtype)
                    if self.cfg.PROBLEM.SELF_SUPERVISED.PRETEXT_TASK == "masking":
                        pred_mask = np.zeros((self.current_sample["X"].shape[0],) + p.shape[1:], dtype=self.dtype)
                        pred_visi = np.zeros((self.current_sample["X"].shape[0],) + p.shape[1:], dtype=self.dtype)
                pred[k * self.cfg.TRAIN.BATCH_SIZE : top] = p
                if self.cfg.PROBLEM.SELF_SUPERVISED.PRETEXT_TASK == "masking":
                    pred_mask[k * self.cfg.TRAIN.BATCH_SIZE : top] = m
                    pred_visi[k * self.cfg.TRAIN.BATCH_SIZE : top] = pv

        # Delete self.current_sample["X"] as in 3D there is no full image
        if self.cfg.PROBLEM.NDIM == "3D":
            del self.current_sample["X"], p

        # Reconstruct the predictions
        if original_data_shape[1:-1] != self.cfg.DATA.PATCH_SIZE[:-1]:
            if self.cfg.PROBLEM.NDIM == "3D":
                original_data_shape = original_data_shape[1:]
            f_name = merge_data_with_overlap if self.cfg.PROBLEM.NDIM == "2D" else merge_3D_data_with_overlap
            pred = f_name(
                pred,
                original_data_shape[:-1] + (pred.shape[-1],),
                padding=self.cfg.DATA.TEST.PADDING,
                overlap=self.cfg.DATA.TEST.OVERLAP,
                verbose=self.cfg.TEST.VERBOSE,
            )
            if self.cfg.PROBLEM.SELF_SUPERVISED.PRETEXT_TASK == "masking":
                pred_mask = f_name(
                    pred_mask,
                    original_data_shape[:-1] + (pred_mask.shape[-1],),
                    padding=self.cfg.DATA.TEST.PADDING,
                    overlap=self.cfg.DATA.TEST.OVERLAP,
                    verbose=self.cfg.TEST.VERBOSE,
                )
                pred_visi = f_name(
                    pred_visi,
                    original_data_shape[:-1] + (pred_visi.shape[-1],),
                    padding=self.cfg.DATA.TEST.PADDING,
                    overlap=self.cfg.DATA.TEST.OVERLAP,
                    verbose=self.cfg.TEST.VERBOSE,
                )
                if self.cfg.PROBLEM.NDIM == "3D":
                    assert isinstance(pred_mask, np.ndarray) and isinstance(pred_visi, np.ndarray)
                    pred_mask = np.expand_dims(pred_mask, 0)
                    pred_visi = np.expand_dims(pred_visi, 0)

            assert isinstance(pred, np.ndarray)
            if self.cfg.PROBLEM.NDIM == "3D":
                pred = np.expand_dims(pred, 0)

        if self.cfg.DATA.REFLECT_TO_COMPLETE_SHAPE:
            reflected_orig_shape = (1,) + self.current_sample["reflected_orig_shape"]
            if reflected_orig_shape != pred.shape:
                if self.cfg.PROBLEM.NDIM == "2D":
                    pred = pred[:, -reflected_orig_shape[1] :, -reflected_orig_shape[2] :]  # type: ignore
                    if self.current_sample["Y"] is not None:
                        self.current_sample["Y"] = self.current_sample["Y"][
                            :, -reflected_orig_shape[1] :, -reflected_orig_shape[2] :
                        ]
                    if self.cfg.PROBLEM.SELF_SUPERVISED.PRETEXT_TASK == "masking":
                        pred_mask = pred_mask[:, -reflected_orig_shape[1] :, -reflected_orig_shape[2] :]  # type: ignore
                        pred_visi = pred_visi[:, -reflected_orig_shape[1] :, -reflected_orig_shape[2] :]  # type: ignore
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
                    if self.cfg.PROBLEM.SELF_SUPERVISED.PRETEXT_TASK == "masking":
                        pred_mask = pred_mask[
                            :,
                            -reflected_orig_shape[1] :,
                            -reflected_orig_shape[2] :,
                            -reflected_orig_shape[3] :,
                        ]  # type: ignore
                        pred_visi = pred_visi[
                            :,
                            -reflected_orig_shape[1] :,
                            -reflected_orig_shape[2] :,
                            -reflected_orig_shape[3] :,
                        ]  # type: ignore

        # Undo normalization
        pred = self.norm_module.undo_image_norm(pred, self.current_sample["X_norm"])
        assert isinstance(pred, np.ndarray)

        # Save image
        if self.cfg.PATHS.RESULT_DIR.PER_IMAGE != "":
            fname, fext = os.path.splitext(self.current_sample["filename"])
            save_tif(
                pred,
                self.cfg.PATHS.RESULT_DIR.PER_IMAGE,
                [self.current_sample["filename"]],
                verbose=self.cfg.TEST.VERBOSE,
            )
            if self.cfg.PROBLEM.SELF_SUPERVISED.PRETEXT_TASK == "masking":
                assert isinstance(pred_mask, np.ndarray) and isinstance(pred_visi, np.ndarray)
                save_tif(
                    pred_mask,
                    self.cfg.PATHS.RESULT_DIR.PER_IMAGE,
                    [fname + "_masked.tif"],
                    verbose=self.cfg.TEST.VERBOSE,
                )
                save_tif(
                    pred_visi,
                    self.cfg.PATHS.RESULT_DIR.PER_IMAGE,
                    [fname + "_reconstruction_and_visible.tif"],
                    verbose=self.cfg.TEST.VERBOSE,
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
        Steps need to be done after merging all predicted patches into the original image.

        Parameters
        ----------
        pred : Torch Tensor
            Model prediction.
        """
        pass

    def after_full_image(self, pred: NDArray):
        """
        Steps that must be executed after generating the prediction by supplying the entire image to the model.

        Parameters
        ----------
        pred : NDArray
            Model prediction.
        """
        pass

    def after_all_images(self):
        """
        Steps that must be done after predicting all images.
        """
        # FID, IS and LPIPS need to be computed for all the images
        if self.current_sample["Y"] is not None:
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

    def prepare_ssl_data(self):
        """
        Creates self supervised "ground truth" images, if ``crappify`` was selected, to train the model based
        on the input images provided. They will be saved in a separate folder in the root path of the inout images.
        """
        if self.cfg.PROBLEM.SELF_SUPERVISED.PRETEXT_TASK == "masking":
            print("No SSL data needs to be prepared for masking, as it will be generated on the fly")
            return

        if is_main_process():
            print("############################")
            print("#  PREPARE DETECTION DATA  #")
            print("############################")

            # Create selected channels for train data
            if self.cfg.TRAIN.ENABLE:
                create_mask = False
                if not os.path.isdir(self.cfg.DATA.TRAIN.SSL_SOURCE_DIR):
                    print(
                        "You select to create detection masks from given .csv files but no file is detected in {}. "
                        "So let's prepare the data. Notice that, if you do not modify 'DATA.TRAIN.SSL_SOURCE_DIR' "
                        "path, this process will be done just once!".format(self.cfg.DATA.TRAIN.SSL_SOURCE_DIR)
                    )
                    create_mask = True
                else:
                    if len(next(os_walk_clean(self.cfg.DATA.TRAIN.SSL_SOURCE_DIR))[2]) != len(
                        next(os_walk_clean(self.cfg.DATA.TRAIN.PATH))[2]
                    ):
                        print(
                            "Different number of files found in {} and {}. Trying to create the the rest again".format(
                                self.cfg.DATA.TRAIN.GT_PATH,
                                self.cfg.DATA.TRAIN.SSL_SOURCE_DIR,
                            )
                        )
                        create_mask = True
                    else:
                        print("Train source data found in {}".format(self.cfg.DATA.TRAIN.SSL_SOURCE_DIR))
                if create_mask:
                    create_ssl_source_data_masks(self.cfg, data_type="train")

            # Create selected channels for val data
            if self.cfg.TRAIN.ENABLE and not self.cfg.DATA.VAL.FROM_TRAIN:
                create_mask = False
                if not os.path.isdir(self.cfg.DATA.VAL.SSL_SOURCE_DIR):
                    print(
                        "You select to create detection masks from given .csv files but no file is detected in {}. "
                        "So let's prepare the data. Notice that, if you do not modify 'DATA.VAL.SSL_SOURCE_DIR' "
                        "path, this process will be done just once!".format(self.cfg.DATA.VAL.SSL_SOURCE_DIR)
                    )
                    create_mask = True
                else:
                    if len(next(os_walk_clean(self.cfg.DATA.VAL.SSL_SOURCE_DIR))[2]) != len(
                        next(os_walk_clean(self.cfg.DATA.VAL.PATH))[2]
                    ):
                        print(
                            "Different number of files found in {} and {}. Trying to create the the rest again".format(
                                self.cfg.DATA.VAL.GT_PATH,
                                self.cfg.DATA.VAL.SSL_SOURCE_DIR,
                            )
                        )
                        create_mask = True
                    else:
                        print("Validation source data found in {}".format(self.cfg.DATA.VAL.SSL_SOURCE_DIR))
                if create_mask:
                    create_ssl_source_data_masks(self.cfg, data_type="val")

            # Create selected channels for test data
            if self.cfg.TEST.ENABLE:
                create_mask = False
                if not os.path.isdir(self.cfg.DATA.TEST.SSL_SOURCE_DIR):
                    print(
                        "You select to create detection masks from given .csv files but no file is detected in {}. "
                        "So let's prepare the data. Notice that, if you do not modify 'DATA.TEST.SSL_SOURCE_DIR' "
                        "path, this process will be done just once!".format(self.cfg.DATA.TEST.SSL_SOURCE_DIR)
                    )
                    create_mask = True
                else:
                    if len(next(os_walk_clean(self.cfg.DATA.TEST.SSL_SOURCE_DIR))[2]) != len(
                        next(os_walk_clean(self.cfg.DATA.TEST.PATH))[2]
                    ):
                        print(
                            "Different number of files found in {} and {}. Trying to create the the rest again".format(
                                self.cfg.DATA.TEST.GT_PATH,
                                self.cfg.DATA.TEST.SSL_SOURCE_DIR,
                            )
                        )
                        create_mask = True
                    else:
                        print("Test source data found in {}".format(self.cfg.DATA.TEST.SSL_SOURCE_DIR))
                if create_mask:
                    create_ssl_source_data_masks(self.cfg, data_type="test")

        if is_dist_avail_and_initialized():
            dist.barrier()

        opts = []
        if self.cfg.TRAIN.ENABLE or self.cfg.DATA.TEST.USE_VAL_AS_TEST:
            print(
                "DATA.TRAIN.PATH changed from {} to {}".format(
                    self.cfg.DATA.TRAIN.PATH, self.cfg.DATA.TRAIN.SSL_SOURCE_DIR
                )
            )
            print(
                "DATA.TRAIN.GT_PATH changed from {} to {}".format(self.cfg.DATA.TRAIN.GT_PATH, self.cfg.DATA.TRAIN.PATH)
            )
            opts.extend(
                [
                    "DATA.TRAIN.PATH",
                    self.cfg.DATA.TRAIN.SSL_SOURCE_DIR,
                    "DATA.TRAIN.GT_PATH",
                    self.cfg.DATA.TRAIN.PATH,
                ]
            )
            if not self.cfg.DATA.VAL.FROM_TRAIN:
                print(
                    "DATA.VAL.PATH changed from {} to {}".format(
                        self.cfg.DATA.VAL.PATH, self.cfg.DATA.VAL.SSL_SOURCE_DIR
                    )
                )
                print(
                    "DATA.VAL.GT_PATH changed from {} to {}".format(self.cfg.DATA.VAL.GT_PATH, self.cfg.DATA.VAL.PATH)
                )
                opts.extend(
                    [
                        "DATA.VAL.PATH",
                        self.cfg.DATA.VAL.SSL_SOURCE_DIR,
                        "DATA.VAL.GT_PATH",
                        self.cfg.DATA.VAL.PATH,
                    ]
                )
        if self.cfg.TEST.ENABLE:
            print(
                "DATA.TEST.PATH changed from {} to {}".format(
                    self.cfg.DATA.TEST.PATH, self.cfg.DATA.TEST.SSL_SOURCE_DIR
                )
            )
            print("DATA.TEST.GT_PATH changed from {} to {}".format(self.cfg.DATA.TEST.GT_PATH, self.cfg.DATA.TEST.PATH))
            opts.extend(
                [
                    "DATA.TEST.PATH",
                    self.cfg.DATA.TEST.SSL_SOURCE_DIR,
                    "DATA.TEST.GT_PATH",
                    self.cfg.DATA.TEST.PATH,
                ]
            )
        self.cfg.merge_from_list(opts)
