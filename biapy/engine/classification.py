"""
Classification workflow for BiaPy.

This module defines the Classification_Workflow class, which implements the
training, validation, and inference pipeline for image classification tasks in BiaPy.
It handles data loading, model setup, metrics, predictions, and result saving for
single-label classification problems.
"""
import os
import torch
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torchmetrics import Accuracy
from typing import Dict, Optional
from numpy.typing import NDArray

from biapy.engine.base_workflow import Base_Workflow
from biapy.data.pre_processing import preprocess_data
from biapy.data.data_manipulation import load_and_prepare_train_data_cls, load_and_prepare_cls_test_data
from biapy.utils.misc import is_main_process, MetricLogger
from biapy.engine.metrics import loss_encapsulation


class Classification_Workflow(Base_Workflow):
    """
    Classification workflow where the goal of this workflow is to assing a label to the input image.

    More details in `our documentation <https://biapy.readthedocs.io/en/latest/workflows/classification.html>`_.

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
        Initialize the Classification_Workflow.

        Sets up configuration, device, job identifier, and initializes
        workflow-specific attributes for classification tasks.

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
        super(Classification_Workflow, self).__init__(cfg, job_identifier, device, args, **kwargs)
        self.all_pred = []
        if self.cfg.DATA.TEST.LOAD_GT:
            self.all_gt = []
        self.test_filenames = None
        self.class_names = None

        # From now on, no modification of the cfg will be allowed
        self.cfg.freeze()

        # Workflow specific training variables
        self.mask_path = cfg.DATA.TRAIN.GT_PATH
        self.is_y_mask = False
        self.load_Y_val = True

    def define_activations_and_channels(self):
        """
        Define the activations and output channels of the model.

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
            "channels": [self.cfg.DATA.N_CLASSES],
        }
        self.real_classes = self.model_output_channels["channels"][0]
        self.multihead = False
        self.activations = [{":": "Linear"}]

        super().define_activations_and_channels()

    def define_metrics(self):
        """
        Define the metrics to be used during training and test/inference.

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
            if metric == "accuracy":
                self.train_metrics.append(
                    Accuracy(task="multiclass", num_classes=self.cfg.DATA.N_CLASSES).to(self.device),
                )
                self.train_metric_names.append("Accuracy")
                self.train_metric_best.append("max")
            elif metric == "top-5-accuracy" and self.cfg.DATA.N_CLASSES > 5:
                self.train_metrics.append(
                    Accuracy(task="multiclass", num_classes=self.cfg.DATA.N_CLASSES, top_k=5).to(self.device),
                )
                self.train_metric_names.append("Top 5 accuracy")
                self.train_metric_best.append("max")

        self.test_metrics = []
        self.test_metric_names = []
        for metric in list(set(self.cfg.TEST.METRICS)):
            if metric == "accuracy":
                self.test_metrics.append(
                    accuracy_score,
                )
                self.test_metric_names.append("Accuracy")

        self.test_metrics.append(confusion_matrix)
        self.test_metric_names.append("Confusion matrix")

        if self.cfg.LOSS.TYPE == "CE":
            self.loss = loss_encapsulation(torch.nn.CrossEntropyLoss())

        super().define_metrics()

    def metric_calculation(
        self,
        output: NDArray | torch.Tensor,
        targets: NDArray | torch.Tensor,
        train: bool = True,
        metric_logger: Optional[MetricLogger] = None,
    ) -> Dict:
        """
        Execute the calculation of metrics defined in :func:`~define_metrics` function.

        Parameters
        ----------
        output : Torch Tensor/List of ints
            Prediction of the model.

        targets : Torch Tensor/List of ints
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
                if isinstance(output, dict):
                    output = output["pred"]
                val = metric(output, targets)
                if torch.is_tensor(val):
                    val = val.item() if not torch.isnan(val) else 0
                out_metrics[list_names_to_use[i]] = val

                if metric_logger:
                    metric_logger.meters[list_names_to_use[i]].update(val)
        return out_metrics

    def prepare_targets(self, targets, batch):
        """
        Perform any necessary data transformations to ``targets`` before calculating the loss.

        Parameters
        ----------
        targets : Torch Tensor
            Ground truth to compare the prediction with.

        batch : Torch Tensor
            Prediction of the model. Not used here.

        Returns
        -------
        targets : Torch tensor
            Resulting targets.
        """
        return targets.to(self.device, non_blocking=True)

    def load_train_data(self):
        """Load training and validation data."""
        (
            self.X_train,
            self.X_val,
            self.cross_val_samples_ids,
        ) = load_and_prepare_train_data_cls(
            train_path=self.cfg.DATA.TRAIN.PATH,
            train_in_memory=self.cfg.DATA.TRAIN.IN_MEMORY,
            val_path=self.cfg.DATA.VAL.PATH,
            val_in_memory=self.cfg.DATA.VAL.IN_MEMORY,
            expected_classes=self.cfg.DATA.N_CLASSES,
            cross_val=self.cfg.DATA.VAL.CROSS_VAL,
            cross_val_nsplits=self.cfg.DATA.VAL.CROSS_VAL_NFOLD,
            cross_val_fold=self.cfg.DATA.VAL.CROSS_VAL_FOLD,
            val_split=self.cfg.DATA.VAL.SPLIT_TRAIN if self.cfg.DATA.VAL.FROM_TRAIN else 0.0,
            seed=self.cfg.SYSTEM.SEED,
            shuffle_val=self.cfg.DATA.VAL.RANDOM,
            train_preprocess_f=preprocess_data if self.cfg.DATA.PREPROCESS.TRAIN else None,
            train_preprocess_cfg=self.cfg.DATA.PREPROCESS if self.cfg.DATA.PREPROCESS.TRAIN else None,
            train_filter_props=(
                self.cfg.DATA.TRAIN.FILTER_SAMPLES.PROPS if self.cfg.DATA.TRAIN.FILTER_SAMPLES.ENABLE else []
            ),
            train_filter_vals=(
                self.cfg.DATA.TRAIN.FILTER_SAMPLES.VALUES if self.cfg.DATA.TRAIN.FILTER_SAMPLES.ENABLE else []
            ),
            train_filter_signs=(
                self.cfg.DATA.TRAIN.FILTER_SAMPLES.SIGNS if self.cfg.DATA.TRAIN.FILTER_SAMPLES.ENABLE else []
            ),
            val_preprocess_f=preprocess_data if self.cfg.DATA.PREPROCESS.VAL else None,
            val_preprocess_cfg=self.cfg.DATA.PREPROCESS if self.cfg.DATA.PREPROCESS.VAL else None,
            val_filter_props=(
                self.cfg.DATA.VAL.FILTER_SAMPLES.PROPS if self.cfg.DATA.VAL.FILTER_SAMPLES.ENABLE else []
            ),
            val_filter_vals=(
                self.cfg.DATA.VAL.FILTER_SAMPLES.VALUES if self.cfg.DATA.VAL.FILTER_SAMPLES.ENABLE else []
            ),
            val_filter_signs=(
                self.cfg.DATA.VAL.FILTER_SAMPLES.SIGNS if self.cfg.DATA.VAL.FILTER_SAMPLES.ENABLE else []
            ),
            crop_shape=self.cfg.DATA.PATCH_SIZE,
            reflect_to_complete_shape=self.cfg.DATA.REFLECT_TO_COMPLETE_SHAPE,
            convert_to_rgb=self.cfg.DATA.FORCE_RGB,
            is_3d=(self.cfg.PROBLEM.NDIM == "3D"),
            norm_before_filter=self.cfg.DATA.TRAIN.FILTER_SAMPLES.NORM_BEFORE,
            norm_module=self.norm_module,
        )
        self.Y_train, self.Y_val = None, None

    def load_test_data(self):
        """Load test data."""
        if self.cfg.TEST.ENABLE:
            print("######################")
            print("#   LOAD TEST DATA   #")
            print("######################")
            use_val_as_test_info = None
            if self.cfg.DATA.TEST.USE_VAL_AS_TEST:
                use_val_as_test_info = {
                    "cross_val_samples_ids": self.cross_val_samples_ids,
                    "train_path": self.cfg.DATA.TRAIN.PATH,
                    "selected_fold": self.cfg.DATA.VAL.CROSS_VAL_FOLD,
                    "n_splits": self.cfg.DATA.VAL.CROSS_VAL_NFOLD,
                    "shuffle": self.cfg.DATA.VAL.RANDOM,
                    "seed": self.cfg.SYSTEM.SEED,
                }

            self.Y_test = None
            (
                self.X_test,
                self.test_filenames,
            ) = load_and_prepare_cls_test_data(
                test_path=self.cfg.DATA.TEST.PATH,
                norm_module=self.norm_module,
                use_val_as_test=self.cfg.DATA.TEST.USE_VAL_AS_TEST,
                expected_classes=self.cfg.DATA.N_CLASSES if self.use_gt else 1,
                crop_shape=self.cfg.DATA.PATCH_SIZE,
                is_3d=(self.cfg.PROBLEM.NDIM == "3D"),
                reflect_to_complete_shape=self.cfg.DATA.REFLECT_TO_COMPLETE_SHAPE,
                convert_to_rgb=self.cfg.DATA.FORCE_RGB,
                use_val_as_test_info=use_val_as_test_info,
            )

    def process_test_sample(self):
        """Process a sample in the inference phase."""
        assert isinstance(self.all_pred, list) and isinstance(self.all_gt, list)
        # Skip processing image
        if "discard" in self.current_sample["X"] and self.current_sample["X"]["discard"]:
            return True

        # Predict each patch
        l = int(math.ceil(self.current_sample["X"].shape[0] / self.cfg.TRAIN.BATCH_SIZE))
        for k in tqdm(range(l), leave=False, disable=not is_main_process()):
            top = (
                (k + 1) * self.cfg.TRAIN.BATCH_SIZE
                if (k + 1) * self.cfg.TRAIN.BATCH_SIZE < self.current_sample["X"].shape[0]
                else self.current_sample["X"].shape[0]
            )
            p = self.model_call_func(self.current_sample["X"][k * self.cfg.TRAIN.BATCH_SIZE : top])
            if isinstance(p, dict):
                p = p["pred"]
            p = p.cpu().numpy()
            p = np.argmax(p, axis=1)
            self.all_pred.append(p)

        if self.current_sample["Y"] is not None and self.all_gt is not None:
            self.all_gt.append(self.current_sample["Y"])

    def torchvision_model_call(self, in_img: torch.Tensor, is_train: bool = False) -> torch.Tensor | None:
        """
        Call a regular Pytorch model.

        Parameters
        ----------
        in_img : torch.Tensors
            Input image to pass through the model.

        is_train : bool, optional
            Whether if the call is during training or inference.

        Returns
        -------
        prediction : torch.Tensor
            Image prediction.
        """
        # Convert first to 0-255 range if uint16
        if in_img.dtype == torch.float32:
            if torch.max(in_img) > 255:
                in_img = (self.torchvision_norm.apply_image_norm(in_img)[0] * 255).to(torch.uint8)  # type: ignore
            in_img = in_img.to(torch.uint8)

        # Apply TorchVision pre-processing
        assert self.torchvision_preprocessing and self.model
        in_img = self.torchvision_preprocessing(in_img)

        return self.model(in_img)

    def after_all_images(self):
        """Execute steps that are needed after predicting all images."""
        self.all_pred = np.array(self.all_pred).squeeze()
        if self.cfg.DATA.TEST.LOAD_GT and self.all_gt is not None:
            self.all_gt = np.array(self.all_gt).squeeze()

        # Save predictions in a csv file
        assert self.test_filenames is not None, "Test filenames must be defined before saving predictions."
        t_filename = [x.path for x in self.test_filenames]
        df = pd.DataFrame(t_filename, columns=["filename"])
        df["class"] = self.all_pred
        f = os.path.join(self.cfg.PATHS.RESULT_DIR.PATH, "predictions.csv")
        os.makedirs(self.cfg.PATHS.RESULT_DIR.PATH, exist_ok=True)
        df.to_csv(f, index=False, header=True)

        # Calculate the metrics
        if self.cfg.DATA.TEST.LOAD_GT and self.all_gt is not None:
            metric_values = self.metric_calculation(
                self.all_pred,
                self.all_gt,  # type: ignore
                train=False,
            )
            for metric in metric_values:
                self.stats["full_image"][str(metric).lower()] = metric_values[metric]
                self.current_sample_metrics[str(metric).lower()] = metric_values[metric]

    def print_stats(self, image_counter):
        """
        Print statistics.

        Parameters
        ----------
        image_counter : int
            Number of images to call ``normalize_stats``.
        """
        if len(self.stats["full_image"]) > 0:
            for metric in self.test_metric_names:
                if metric.lower() in self.stats["full_image"]:
                    if metric == "Confusion matrix":
                        print("Confusion matrix: ")
                        print(self.stats["full_image"][metric.lower()])
                        if self.class_names:
                            display_labels = [
                                "Category {} ({})".format(i, self.class_names[i])
                                for i in range(self.cfg.DATA.N_CLASSES)
                            ]
                        else:
                            display_labels = ["Category {}".format(i) for i in range(self.cfg.DATA.N_CLASSES)]
                        print("\n" + classification_report(self.all_gt, self.all_pred, target_names=display_labels))  # type: ignore
                    else:
                        print(
                            "Test {}: {}".format(
                                metric,
                                self.stats["full_image"][metric.lower()],
                            )
                        )

    def after_merge_patches(self, pred):
        """
        Execute steps that are needed after merging all predicted patches into the original image.

        Parameters
        ----------
        pred : Torch Tensor
            Model prediction.
        """
        pass

    def after_full_image(self, pred: NDArray):
        """
        Execute steps that are needed after generating the prediction by supplying the entire image to the model.

        Parameters
        ----------
        pred : NDArray
            Model prediction.
        """
        pass
