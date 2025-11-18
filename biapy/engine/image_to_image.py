"""
Image-to-image workflow for BiaPy.

This module defines the Image_to_Image_Workflow class, which implements the
training, validation, and inference pipeline for image-to-image regression tasks
in BiaPy. It supports metrics such as PSNR, SSIM, FID, IS, LPIPS, and handles
data loading, model setup, predictions, and result saving for 2D and 3D images.
"""
import torch
import numpy as np
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from typing import Dict, Optional
from numpy.typing import NDArray


from biapy.engine.metrics import SSIM_loss, W_MAE_SSIM_loss, W_MSE_SSIM_loss, loss_encapsulation
from biapy.engine.base_workflow import Base_Workflow
from biapy.utils.misc import to_pytorch_format, MetricLogger
from biapy.data.data_2D_manipulation import (
    crop_data_with_overlap,
    merge_data_with_overlap,
)
from biapy.data.data_3D_manipulation import (
    crop_3D_data_with_overlap,
    merge_3D_data_with_overlap,
)
from biapy.data.data_manipulation import save_tif


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

    def __init__(self, cfg, job_identifier, device, args, **kwargs):
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
        super(Image_to_Image_Workflow, self).__init__(cfg, job_identifier, device, args, **kwargs)
        # From now on, no modification of the cfg will be allowed
        self.cfg.freeze()

        self.mask_path = cfg.DATA.TRAIN.GT_PATH
        self.is_y_mask = False

        self.norm_module.mask_norm = "as_image"
        self.test_norm_module.mask_norm = "as_image"

    def define_activations_and_channels(self):
        """
        Define the activations and output channels of the model.

        This function must define the following variables:

        self.model_output_channels : List of functions
            Metrics to be calculated during model's training.

        self.multihead : bool
            Whether if the output of the model has more than one head.

        self.activations : List of lists of str
            Activations to be applied to the model output. Each dict will
            match an output channel of the model. "linear" and "ce_sigmoid"
            will not be applied. E.g. ["linear"].
        """
        self.model_output_channels = {
            "type": "image",
            "channels": [self.cfg.DATA.PATCH_SIZE[-1]],
        }
        self.real_classes = self.model_output_channels["channels"][0]
        self.multihead = False
        for _ in range(self.model_output_channels["channels"][0]):
            self.activations.append("linear")
        self.activations = [self.activations]

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

        super().define_metrics()

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
        pred = self.norm_module.undo_image_norm(pred, self.current_sample["X_norm"])
        assert isinstance(pred, np.ndarray)

        # Save image
        if self.cfg.PATHS.RESULT_DIR.PER_IMAGE != "" and self.cfg.TEST.SAVE_MODEL_RAW_OUTPUT:
            save_tif(
                pred,
                self.cfg.PATHS.RESULT_DIR.PER_IMAGE,
                [self.current_sample["X_filename"]],
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
