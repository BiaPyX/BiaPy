import math
import torch
import numpy as np
from tqdm import tqdm
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

from biapy.engine.base_workflow import Base_Workflow
from biapy.utils.util import save_tif, pad_and_reflect
from biapy.utils.misc import to_pytorch_format, to_numpy_format
from biapy.data.pre_processing import undo_norm_range01, denormalize
from biapy.data.post_processing.post_processing import (
    ensemble8_2d_predictions,
    ensemble16_3d_predictions,
)
from biapy.data.data_2D_manipulation import (
    crop_data_with_overlap,
    merge_data_with_overlap,
)
from biapy.data.data_3D_manipulation import (
    crop_3D_data_with_overlap,
    merge_3D_data_with_overlap,
)
from biapy.engine.metrics import MSE_wrapper


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
        super(Image_to_Image_Workflow, self).__init__(cfg, job_identifier, device, args, **kwargs)
        # From now on, no modification of the cfg will be allowed
        self.cfg.freeze()

        # Activations for each output channel:
        # channel number : 'activation'
        self.activations = [{":": "Linear"}]

        self.mask_path = cfg.DATA.TRAIN.GT_PATH

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
        self.train_metrics = []
        self.train_metric_names = []
        self.train_metric_best = []
        for metric in list(set(self.cfg.TRAIN.METRICS)):
            if metric == "psnr":
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
                self.train_metrics.append(StructuralSimilarityIndexMeasure().to(self.device))
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

        print("Overriding 'LOSS.TYPE' to set it to MSE")
        self.loss = MSE_wrapper()

        super().define_metrics()

    def metric_calculation(self, output, targets, train=True, metric_logger=None):
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
        out_metrics = {}
        list_to_use = self.train_metrics if train else self.test_metrics
        list_names_to_use = self.train_metric_names if train else self.test_metric_names

        with torch.no_grad():
            for i, metric in enumerate(list_to_use):
                m_name = list_names_to_use[i].lower()
                if m_name in ["mse", "mae"]:
                    val = metric(output, targets)
                elif m_name == "ssim":
                    val = metric(output.to(torch.float32), targets.to(torch.float32))
                elif m_name == "psnr":
                    # Normalize values to be between 0-255 range so PSNR value its more meaningful
                    norm_output = ((output - torch.min(output)) / (torch.max(output) - torch.min(output) + 1e-8)) * 255
                    norm_targets = (
                        (targets - torch.min(targets)) / (torch.max(targets) - torch.min(targets) + 1e-8)
                    ) * 255
                    val = metric(norm_output, norm_targets)
                elif m_name in ["is", "lpips", "fid"]:
                    # These metrics need to have normalized (between 0 and 1) images with 3 channels
                    norm_output = (output - torch.min(output)) / (torch.max(output) - torch.min(output) + 1e-8)
                    norm_targets = (targets - torch.min(targets)) / (torch.max(targets) - torch.min(targets) + 1e-8)
                    norm_3c_output = torch.cat([norm_output, norm_output, norm_output], dim=1)
                    norm_3c_targets = torch.cat([norm_targets, norm_targets, norm_targets], dim=1)
                    if m_name == "fid":
                        metric.update(norm_3c_output, real=True)
                        metric.update(norm_3c_targets, real=False)
                    elif m_name == "is":
                        metric.update(norm_3c_targets)
                    else:  # lpips
                        metric.update(norm_3c_output, norm_3c_targets)
                else:
                    raise NotImplementedError

                if m_name in ["mse", "mae", "ssim", "psnr"]:
                    val = val.item() if not torch.isnan(val) else 0
                    out_metrics[m_name] = val

                if metric_logger is not None:
                    metric_logger.meters[list_names_to_use[i]].update(val)
        return out_metrics

    def process_sample(self, norm):
        """
        Function to process a sample in the inference phase.

        Parameters
        ----------
        norm : List of dicts
            Normalization used during training. Required to denormalize the predictions of the model.
        """
        # Save test_input if the user wants to export the model to BMZ later
        if "test_input" not in self.bmz_config:
            if self.cfg.PROBLEM.NDIM == "2D":
                self.bmz_config["test_input"] = self._X[0][
                    : self.cfg.DATA.PATCH_SIZE[0], : self.cfg.DATA.PATCH_SIZE[1]
                ].copy()
            else:
                self.bmz_config["test_input"] = self._X[0][
                    : self.cfg.DATA.PATCH_SIZE[0], : self.cfg.DATA.PATCH_SIZE[1], : self.cfg.DATA.PATCH_SIZE[2]
                ].copy()

        # Reflect data to complete the needed shape
        if self.cfg.DATA.REFLECT_TO_COMPLETE_SHAPE:
            reflected_orig_shape = self._X.shape
            self._X = np.expand_dims(
                pad_and_reflect(self._X[0], self.cfg.DATA.PATCH_SIZE, verbose=self.cfg.TEST.VERBOSE),
                0,
            )
            if self._Y is not None:
                self._Y = np.expand_dims(
                    pad_and_reflect(
                        self._Y[0],
                        self.cfg.DATA.PATCH_SIZE,
                        verbose=self.cfg.TEST.VERBOSE,
                    ),
                    0,
                )

        original_data_shape = self._X.shape

        # Crop if necessary
        if self._X.shape[1:-1] != self.cfg.DATA.PATCH_SIZE[:-1]:
            if self.cfg.PROBLEM.NDIM == "2D":
                obj = crop_data_with_overlap(
                    self._X,
                    self.cfg.DATA.PATCH_SIZE,
                    data_mask=self._Y,
                    overlap=self.cfg.DATA.TEST.OVERLAP,
                    padding=self.cfg.DATA.TEST.PADDING,
                    verbose=self.cfg.TEST.VERBOSE,
                )
                if self._Y is not None:
                    self._X, self._Y = obj
                else:
                    self._X = obj
                del obj
            else:
                if self._Y is not None:
                    self._Y = self._Y[0]
                if self.cfg.TEST.REDUCE_MEMORY:
                    self._X = crop_3D_data_with_overlap(
                        self._X[0],
                        self.cfg.DATA.PATCH_SIZE,
                        overlap=self.cfg.DATA.TEST.OVERLAP,
                        padding=self.cfg.DATA.TEST.PADDING,
                        verbose=self.cfg.TEST.VERBOSE,
                        median_padding=self.cfg.DATA.TEST.MEDIAN_PADDING,
                    )
                    self._Y = crop_3D_data_with_overlap(
                        self._Y,
                        self.cfg.DATA.PATCH_SIZE,
                        overlap=self.cfg.DATA.TEST.OVERLAP,
                        padding=self.cfg.DATA.TEST.PADDING,
                        verbose=self.cfg.TEST.VERBOSE,
                        median_padding=self.cfg.DATA.TEST.MEDIAN_PADDING,
                    )
                else:
                    obj = crop_3D_data_with_overlap(
                        self._X[0],
                        self.cfg.DATA.PATCH_SIZE,
                        data_mask=self._Y,
                        overlap=self.cfg.DATA.TEST.OVERLAP,
                        padding=self.cfg.DATA.TEST.PADDING,
                        verbose=self.cfg.TEST.VERBOSE,
                        median_padding=self.cfg.DATA.TEST.MEDIAN_PADDING,
                    )
                    if self._Y is not None:
                        self._X, self._Y = obj
                    else:
                        self._X = obj
                    del obj

        # Predict each patch
        if self.cfg.TEST.AUGMENTATION:
            for k in tqdm(range(self._X.shape[0]), leave=False):
                if self.cfg.PROBLEM.NDIM == "2D":
                    p = ensemble8_2d_predictions(
                        self._X[k],
                        axis_order_back=self.axis_order_back,
                        pred_func=self.model_call_func,
                        axis_order=self.axis_order,
                        device=self.device,
                    )
                else:
                    p = ensemble16_3d_predictions(
                        self._X[k],
                        batch_size_value=self.cfg.TRAIN.BATCH_SIZE,
                        axis_order_back=self.axis_order_back,
                        pred_func=self.model_call_func,
                        axis_order=self.axis_order,
                        device=self.device,
                    )
                p = self.apply_model_activations(p)
                p = to_numpy_format(p, self.axis_order_back)
                if "pred" not in locals():
                    pred = np.zeros((self._X.shape[0],) + p.shape[1:], dtype=self.dtype)
                pred[k] = p
        else:
            self._X = to_pytorch_format(self._X, self.axis_order, self.device)
            l = int(math.ceil(self._X.shape[0] / self.cfg.TRAIN.BATCH_SIZE))
            for k in tqdm(range(l), leave=False):
                top = (
                    (k + 1) * self.cfg.TRAIN.BATCH_SIZE
                    if (k + 1) * self.cfg.TRAIN.BATCH_SIZE < self._X.shape[0]
                    else self._X.shape[0]
                )
                with torch.cuda.amp.autocast():
                    p = self.model(self._X[k * self.cfg.TRAIN.BATCH_SIZE : top])
                p = to_numpy_format(self.apply_model_activations(p), self.axis_order_back)
                if "pred" not in locals():
                    pred = np.zeros((self._X.shape[0],) + p.shape[1:], dtype=self.dtype)
                pred[k * self.cfg.TRAIN.BATCH_SIZE : top] = p
        del self._X, p

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
                self._Y = f_name(
                    self._Y,
                    original_data_shape[:-1] + (self._Y.shape[-1],),
                    padding=self.cfg.DATA.TEST.PADDING,
                    overlap=self.cfg.DATA.TEST.OVERLAP,
                    verbose=self.cfg.TEST.VERBOSE,
                )
            else:
                obj = f_name(
                    pred,
                    original_data_shape[:-1] + (pred.shape[-1],),
                    data_mask=self._Y,
                    padding=self.cfg.DATA.TEST.PADDING,
                    overlap=self.cfg.DATA.TEST.OVERLAP,
                    verbose=self.cfg.TEST.VERBOSE,
                )
                if self._Y is not None:
                    pred, self._Y = obj
                else:
                    pred = obj
                del obj

            if self.cfg.PROBLEM.NDIM == "3D":
                pred = np.expand_dims(pred, 0)
                if self._Y is not None:
                    self._Y = np.expand_dims(self._Y, 0)

        if self.cfg.DATA.REFLECT_TO_COMPLETE_SHAPE:
            if self.cfg.PROBLEM.NDIM == "2D":
                pred = pred[:, -reflected_orig_shape[1] :, -reflected_orig_shape[2] :]
                if self._Y is not None:
                    self._Y = self._Y[:, -reflected_orig_shape[1] :, -reflected_orig_shape[2] :]
            else:
                pred = pred[
                    :,
                    -reflected_orig_shape[1] :,
                    -reflected_orig_shape[2] :,
                    -reflected_orig_shape[3] :,
                ]
                if self._Y is not None:
                    self._Y = self._Y[
                        :,
                        -reflected_orig_shape[1] :,
                        -reflected_orig_shape[2] :,
                        -reflected_orig_shape[3] :,
                    ]

        # Undo normalization
        x_norm = norm[0]
        if x_norm["type"] == "div":
            pred = undo_norm_range01(pred, x_norm)
        elif x_norm["type"] == "scale_range":
            pred = undo_norm_range01(pred, x_norm, x_norm["min_val_scale"], x_norm["max_val_scale"])
        else:
            pred = denormalize(pred, x_norm["mean"], x_norm["std"])

            if x_norm["orig_dtype"] not in [
                np.dtype("float64"),
                np.dtype("float32"),
                np.dtype("float16"),
            ]:
                pred = np.round(pred)
                minpred = np.min(pred)
                pred = pred + abs(minpred)

            pred = pred.astype(x_norm["orig_dtype"])

        # Save image
        if self.cfg.PATHS.RESULT_DIR.PER_IMAGE != "":
            save_tif(
                pred,
                self.cfg.PATHS.RESULT_DIR.PER_IMAGE,
                self.processing_filenames,
                verbose=self.cfg.TEST.VERBOSE,
            )

        # Calculate metrics
        if pred.dtype == np.dtype("uint16"):
            pred = pred.astype(np.float32)

        if self._Y is not None:
            if self._Y.dtype == np.dtype("uint16"):
                self._Y = self._Y.astype(np.float32)

            metric_values = self.metric_calculation(
                to_pytorch_format(pred, self.axis_order, self.device),
                to_pytorch_format(
                    self._Y,
                    self.axis_order,
                    self.device,
                    dtype=self.loss_dtype,
                ),
                train=False,
            )
            for metric in metric_values:
                if str(metric).lower() not in self.stats["merge_patches"]:
                    self.stats["merge_patches"][str(metric).lower()] = 0
                self.stats["merge_patches"][str(metric).lower()] += metric_values[metric]

        # Save test_output if the user wants to export the model to BMZ later
        if "test_output" not in self.bmz_config:
            if self.cfg.PROBLEM.NDIM == "2D":
                self.bmz_config["test_output"] = pred[0][
                    : self.cfg.DATA.PATCH_SIZE[0], : self.cfg.DATA.PATCH_SIZE[1]
                ].copy()
            else:
                self.bmz_config["test_output"] = pred[0][
                    : self.cfg.DATA.PATCH_SIZE[0], : self.cfg.DATA.PATCH_SIZE[1], : self.cfg.DATA.PATCH_SIZE[2]
                ].copy()

    def torchvision_model_call(self, in_img, is_train=False):
        """
        Call a regular Pytorch model.

        Parameters
        ----------
        in_img : Tensor
            Input image to pass through the model.

        is_train : bool, optional
            Whether if the call is during training or inference.

        Returns
        -------
        prediction : Tensor
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

    def after_merge_patches_by_chunks_proccess_patch(self, filename):
        """
        Place any code that needs to be done after merging all predicted patches into the original image
        but in the process made chunk by chunk. This function will operate patch by patch defined by
        ``DATA.PATCH_SIZE``.

        Parameters
        ----------
        filename : List of str
            Filename of the predicted image H5/Zarr.
        """
        pass

    def after_full_image(self, pred):
        """
        Steps that must be executed after generating the prediction by supplying the entire image to the model.

        Parameters
        ----------
        pred : Torch Tensor
            Model prediction.
        """
        pass

    def after_all_images(self):
        """
        Steps that must be done after predicting all images.
        """
        # FID, IS and LPIPS need to be computed for all the images
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
