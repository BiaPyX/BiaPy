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
)
from biapy.engine.base_workflow import Base_Workflow
from biapy.data.pre_processing import create_ssl_source_data_masks, undo_sample_normalization


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

        self.norm_dict["mask_norm"] = "as_image"
        self.test_norm_dict["mask_norm"] = "none" 

    def define_activations_and_channels(self):
        """
        This function must define the following variables:

        self.model_output_channels : List of functions
            Metrics to be calculated during model's training.

        self.multihead : List of str
            Names of the metrics calculated during training.

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

        if self.cfg.MODEL.ARCHITECTURE.lower() == "mae":
            print("Overriding 'LOSS.TYPE' to set it to MSE loss (masking patches)")
            self.loss = self.MaskedAutoencoderViT_loss_wrapper
        else:
            if self.cfg.LOSS.TYPE == "MSE":
                self.loss = torch.nn.MSELoss().to(self.device)
            elif self.cfg.LOSS.TYPE == "MAE":
                self.loss = torch.nn.L1Loss().to(self.device)

        super().define_metrics()

    def MaskedAutoencoderViT_loss_wrapper(self, output, targets):
        """
        Unravel MAE loss.
        """
        # Targets not used because the loss has been already calculated
        loss, pred, mask = output
        return loss

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
        if self.cfg.PROBLEM.SELF_SUPERVISED.PRETEXT_TASK.lower() == "masking":
            _, pred, _ = output
            pred = self.model_without_ddp.unpatchify(pred)
        else:
            pred = output

        out_metrics = {}
        list_to_use = self.train_metrics if train else self.test_metrics
        list_names_to_use = self.train_metric_names if train else self.test_metric_names

        with torch.no_grad():
            for i, metric in enumerate(list_to_use):
                if self.cfg.DATA.NORMALIZATION.TYPE in ["div", "scale_range"]:
                    pred = torch.clamp(pred, min=0, max=1)
                    targets = torch.clamp(targets, min=0, max=1)

                m_name = list_names_to_use[i].lower()
                if m_name in ["mse", "mae"]:
                    val = metric(pred, targets)
                elif m_name == "ssim":
                    val = metric(pred.to(torch.float32), targets.to(torch.float32))
                elif m_name == "psnr":
                    # Normalize values to be between 0-255 range so PSNR value its more meaningful
                    if self.cfg.DATA.NORMALIZATION.TYPE not in ["div", "scale_range"]:
                        norm_output = (pred - torch.min(pred)) / (torch.max(pred) - torch.min(pred) + 1e-8)
                        norm_targets = (targets - torch.min(targets)) / (torch.max(targets) - torch.min(targets) + 1e-8)
                    else:
                        norm_output = pred
                        norm_targets = targets
                    norm_output *= 255
                    norm_targets *= 255
                    val = metric(norm_output, norm_targets)
                elif m_name in ["is", "lpips", "fid"]:
                    # These metrics need to have normalized (between 0 and 1) images with 3 channels
                    if self.cfg.DATA.NORMALIZATION.TYPE not in ["div", "scale_range"]:
                        norm_output = (pred - torch.min(pred)) / (torch.max(pred) - torch.min(pred) + 1e-8)
                        norm_targets = (targets - torch.min(targets)) / (torch.max(targets) - torch.min(targets) + 1e-8)
                    else:
                        norm_output = pred
                        norm_targets = targets
                    norm_3c_pred = torch.cat([norm_output, norm_output, norm_output], dim=1)
                    norm_3c_targets = torch.cat([norm_targets, norm_targets, norm_targets], dim=1)
                    if m_name == "fid":
                        metric.update(norm_3c_pred, real=True)
                        metric.update(norm_3c_targets, real=False)
                    elif m_name == "is":
                        metric.update(norm_3c_targets)
                    else:  # lpips
                        metric.update(norm_3c_pred, norm_3c_targets)
                else:
                    raise NotImplementedError

                if m_name in ["mse", "mae", "ssim", "psnr"]:
                    val = val.item() if not torch.isnan(val) else 0
                    out_metrics[m_name] = val

                if metric_logger is not None:
                    metric_logger.meters[list_names_to_use[i]].update(val)
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
            return to_pytorch_format(batch, self.axis_order, self.device, dtype=self.loss_dtype)
        else:
            return to_pytorch_format(targets, self.axis_order, self.device, dtype=self.loss_dtype)

    def process_test_sample(self):
        """
        Function to process a sample in the inference phase.
        """
        # Skip processing image
        if "discard" in self.current_sample["X"] and self.current_sample["X"]["discard"]:
            return True

        # Save BMZ input/output so the user could export the model to BMZ later
        if "test_output" not in self.bmz_config:
            self.prepare_bmz_data(self.current_sample["X"].transpose(self.axis_order))

        original_data_shape = self.current_sample["X"].shape

        # Crop if necessary
        if self.current_sample["X"].shape[1:-1] != self.cfg.DATA.PATCH_SIZE[:-1]:
            if self.cfg.PROBLEM.NDIM == "2D":
                self.current_sample["X"], _ = crop_data_with_overlap(
                    self.current_sample["X"],
                    self.cfg.DATA.PATCH_SIZE,
                    overlap=self.cfg.DATA.TEST.OVERLAP,
                    padding=self.cfg.DATA.TEST.PADDING,
                    verbose=self.cfg.TEST.VERBOSE,
                )
            else:
                self.current_sample["X"], _ = crop_3D_data_with_overlap(
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
                        axis_order_back=self.axis_order_back,
                        pred_func=self.model_call_func,
                        axis_order=self.axis_order,
                        device=self.device,
                    )
                else:
                    p = ensemble16_3d_predictions(
                        self.current_sample["X"][k],
                        batch_size_value=self.cfg.TRAIN.BATCH_SIZE,
                        axis_order_back=self.axis_order_back,
                        pred_func=self.model_call_func,
                        axis_order=self.axis_order,
                        device=self.device,
                    )
                p = self.apply_model_activations(p)
                p = to_numpy_format(p, self.axis_order_back)
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
                p = self.model(
                    to_pytorch_format(
                        self.current_sample["X"][k * self.cfg.TRAIN.BATCH_SIZE : top],
                        self.axis_order,
                        self.device,
                    )
                )
                if self.cfg.PROBLEM.SELF_SUPERVISED.PRETEXT_TASK == "masking":
                    loss, p, mask = p
                    p = self.apply_model_activations(p)
                    p, m, pv = self.model_without_ddp.save_images(
                        to_pytorch_format(
                            self.current_sample["X"][k * self.cfg.TRAIN.BATCH_SIZE : top],
                            self.axis_order,
                            self.device,
                        ),
                        p,
                        mask,
                        self.dtype,
                    )
                else:
                    p = self.apply_model_activations(p)
                    p = to_numpy_format(p, self.axis_order_back)

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
                    pred_mask = np.expand_dims(pred_mask, 0)
                    pred_visi = np.expand_dims(pred_visi, 0)

            if self.cfg.PROBLEM.NDIM == "3D":
                pred = np.expand_dims(pred, 0)

        if self.cfg.DATA.REFLECT_TO_COMPLETE_SHAPE:
            reflected_orig_shape = (1,) + self.current_sample["reflected_orig_shape"]
            if reflected_orig_shape != pred.shape:
                if self.cfg.PROBLEM.NDIM == "2D":
                    pred = pred[:, -reflected_orig_shape[1] :, -reflected_orig_shape[2] :]
                    if self.current_sample["Y"] is not None:
                        self.current_sample["Y"] = self.current_sample["Y"][
                            :, -reflected_orig_shape[1] :, -reflected_orig_shape[2] :
                        ]
                    if self.cfg.PROBLEM.SELF_SUPERVISED.PRETEXT_TASK == "masking":
                        pred_mask = pred_mask[:, -reflected_orig_shape[1] :, -reflected_orig_shape[2] :]
                        pred_visi = pred_visi[:, -reflected_orig_shape[1] :, -reflected_orig_shape[2] :]
                else:
                    pred = pred[
                        :,
                        -reflected_orig_shape[1] :,
                        -reflected_orig_shape[2] :,
                        -reflected_orig_shape[3] :,
                    ]
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
                        ]
                        pred_visi = pred_visi[
                            :,
                            -reflected_orig_shape[1] :,
                            -reflected_orig_shape[2] :,
                            -reflected_orig_shape[3] :,
                        ]

        # Undo normalization
        pred = undo_sample_normalization(pred, self.current_sample["X_norm"])

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

            metric_values = self.metric_calculation(
                to_pytorch_format(pred, self.axis_order, self.device),
                to_pytorch_format(
                    self.current_sample["Y"],
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
                    if len(next(os.walk(self.cfg.DATA.TRAIN.SSL_SOURCE_DIR))[2]) != len(
                        next(os.walk(self.cfg.DATA.TRAIN.PATH))[2]
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
                    if len(next(os.walk(self.cfg.DATA.VAL.SSL_SOURCE_DIR))[2]) != len(
                        next(os.walk(self.cfg.DATA.VAL.PATH))[2]
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
                    if len(next(os.walk(self.cfg.DATA.TEST.SSL_SOURCE_DIR))[2]) != len(
                        next(os.walk(self.cfg.DATA.TEST.PATH))[2]
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
