import torch
import numpy as np
from skimage.transform import resize
from typing import (
    Dict,
    Optional
)
from numpy.typing import NDArray


from biapy.data.post_processing.post_processing import apply_binary_mask
from biapy.engine.base_workflow import Base_Workflow
from biapy.data.data_manipulation import check_masks, save_tif
from biapy.utils.misc import to_pytorch_format, to_numpy_format, to_pytorch_format, MetricLogger
from biapy.engine.metrics import (
    jaccard_index,
    CrossEntropyLoss_wrapper,
    DiceBCELoss,
    DiceLoss,
)


class Semantic_Segmentation_Workflow(Base_Workflow):
    """
    Semantic segmentation workflow where the goal is to assign a class to each pixel of the input image.
    More details in `our documentation <https://biapy.readthedocs.io/en/latest/workflows/semantic_segmentation.html>`_.

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
        super(Semantic_Segmentation_Workflow, self).__init__(cfg, job_identifier, device, args, **kwargs)

        if cfg.TRAIN.ENABLE and cfg.DATA.TRAIN.CHECK_DATA:
            check_masks(cfg.DATA.TRAIN.GT_PATH, n_classes=cfg.MODEL.N_CLASSES, is_3d=(self.cfg.PROBLEM.NDIM == "3D"))
            if not cfg.DATA.VAL.FROM_TRAIN:
                check_masks(cfg.DATA.VAL.GT_PATH, n_classes=cfg.MODEL.N_CLASSES, is_3d=(self.cfg.PROBLEM.NDIM == "3D"))
        if cfg.TEST.ENABLE and cfg.DATA.TEST.LOAD_GT and cfg.DATA.TEST.CHECK_DATA:
            check_masks(cfg.DATA.TEST.GT_PATH, n_classes=cfg.MODEL.N_CLASSES, is_3d=(self.cfg.PROBLEM.NDIM == "3D"))

        # From now on, no modification of the cfg will be allowed
        self.cfg.freeze()

        # Workflow specific training variables
        self.mask_path = cfg.DATA.TRAIN.GT_PATH
        self.is_y_mask = True

        self.load_Y_val = True
        self.loss_dtype = torch.float32

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
            "type": "mask",
            "channels": [1 if self.cfg.MODEL.N_CLASSES <= 2 else self.cfg.MODEL.N_CLASSES],
        }
        self.multihead = False 
        self.activations = [{":": "CE_Sigmoid"}]

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
            if metric in ["iou", "jaccard_index"]:
                self.train_metrics.append(
                    jaccard_index(
                        num_classes=self.cfg.MODEL.N_CLASSES,
                        device=self.device,
                        model_source=self.cfg.MODEL.SOURCE,
                    )
                )
                self.train_metric_names.append("IoU")
                self.train_metric_best.append("max")

        self.test_metrics = []
        self.test_metric_names = []
        for metric in list(set(self.cfg.TEST.METRICS)):
            if metric in ["iou", "jaccard_index"]:
                self.test_metrics.append(
                    jaccard_index(
                        num_classes=self.cfg.MODEL.N_CLASSES,
                        device=self.device,
                        model_source=self.cfg.MODEL.SOURCE,
                    )
                )
                self.test_metric_names.append("IoU")

        if self.cfg.LOSS.TYPE == "CE":
            self.loss = CrossEntropyLoss_wrapper(
                num_classes=self.cfg.MODEL.N_CLASSES,
                model_source=self.cfg.MODEL.SOURCE,
                class_rebalance=self.cfg.LOSS.CLASS_REBALANCE,
            )
        elif self.cfg.LOSS.TYPE == "DICE":
            self.loss = DiceLoss()
        elif self.cfg.LOSS.TYPE == "W_CE_DICE":
            self.loss = DiceBCELoss(w_dice=self.cfg.LOSS.WEIGHTS[0], w_bce=self.cfg.LOSS.WEIGHTS[1])

        super().define_metrics()

    def process_test_sample(self):
        """
        Function to process a sample in the inference phase.
        """
        if self.cfg.MODEL.SOURCE != "torchvision":
            super().process_test_sample()
        else:
            # Skip processing image
            if "discard" in self.current_sample["X"] and self.current_sample["X"]["discard"]:
                return True

            ##################
            ### FULL IMAGE ###
            ##################
            # Make the prediction
            pred = self.model_call_func(self.current_sample["X"])
            pred = to_numpy_format(pred, self.axis_order_back)
            del self.current_sample["X"]

            if self.cfg.DATA.REFLECT_TO_COMPLETE_SHAPE:
                reflected_orig_shape = (1,) + self.current_sample["reflected_orig_shape"]
                if reflected_orig_shape != pred.shape:
                    if self.cfg.PROBLEM.NDIM == "2D":
                        pred = pred[:, -reflected_orig_shape[1] :, -reflected_orig_shape[2] :]
                    else:
                        pred = pred[
                            :,
                            -reflected_orig_shape[1] :,
                            -reflected_orig_shape[2] :,
                            -reflected_orig_shape[3] :,
                        ]

            if self.cfg.TEST.POST_PROCESSING.APPLY_MASK:
                pred = apply_binary_mask(pred, self.cfg.DATA.TEST.BINARY_MASKS)

            if self.current_sample["Y"] is not None:
                if pred.shape != self.current_sample["Y"].shape:
                    self.current_sample["Y"] = resize(self.current_sample["Y"], pred.shape, order=0)

                metric_values = self.metric_calculation(output=pred, targets=self.current_sample["Y"], train=False)
                for metric in metric_values:
                    if str(metric).lower() not in self.stats["full_image"]:
                        self.stats["full_image"][str(metric).lower()] = 0
                    self.stats["full_image"][str(metric).lower()] += metric_values[metric]

    def torchvision_model_call(
        self, 
        in_img: torch.Tensor, 
        is_train=False
    ) -> torch.Tensor:
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
        assert self.torchvision_preprocessing and self.model

        # Convert first to 0-255 range if uint16
        if in_img.dtype == torch.float32:
            if torch.max(in_img) > 1:
                in_img = (self.torchvision_norm.apply_image_norm(in_img)[0] * 255).to(torch.uint8)
            in_img = in_img.to(torch.uint8)

        # Apply TorchVision pre-processing
        in_img = self.torchvision_preprocessing(in_img)

        pred = self.model(in_img)
        key = "aux" if "aux" in pred else "out"

        # Save masks
        if not is_train:
            masks = np.expand_dims(np.argmax(pred[key].cpu().numpy().transpose(0, 2, 3, 1), axis=-1), -1)
            save_tif(
                masks,
                self.cfg.PATHS.RESULT_DIR.FULL_IMAGE,
                [self.current_sample["filename"]],
                verbose=self.cfg.TEST.VERBOSE,
            )

        return pred[key]

    def metric_calculation(
        self, 
        output: NDArray | torch.Tensor, 
        targets: NDArray | torch.Tensor, 
        train: bool=True, 
        metric_logger: Optional[MetricLogger]=None
    ) -> Dict :
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
        if isinstance(output, np.ndarray):
            _output = to_pytorch_format(
                output.copy(),
                self.axis_order,
                self.device,
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
                self.axis_order,
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

        with torch.no_grad():
            for i, metric in enumerate(list_to_use):
                val = metric(_output, _targets)
                val = val.item() if not torch.isnan(val) else 0
                out_metrics[list_names_to_use[i]] = val

                if metric_logger:
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
            Prediction of the model. Only used in SSL workflow.

        Returns
        -------
        targets : Torch tensor
            Resulting targets.
        """
        # We do not use 'batch' input but in SSL workflow
        return to_pytorch_format(targets, self.axis_order, self.device, dtype=self.loss_dtype)

    def after_merge_patches(self, pred):
        """
        Steps need to be done after merging all predicted patches into the original image.

        Parameters
        ----------
        pred : Torch Tensor
            Model prediction.
        """
        # Save simple binarization of predictions
        if self.cfg.MODEL.N_CLASSES <= 2:
            pred = (pred > 0.5).astype(np.uint8)
        save_tif(
            pred,
            self.cfg.PATHS.RESULT_DIR.PER_IMAGE_BIN,
            [self.current_sample["filename"]],
            verbose=self.cfg.TEST.VERBOSE,
        )

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
        # Save simple binarization of predictions
        save_tif(
            (pred > 0.5).astype(np.uint8),
            self.cfg.PATHS.RESULT_DIR.FULL_IMAGE_BIN,
            [self.current_sample["filename"]],
            verbose=self.cfg.TEST.VERBOSE,
        )

    def after_all_images(self):
        """
        Steps that must be done after predicting all images.
        """
        super().after_all_images()
