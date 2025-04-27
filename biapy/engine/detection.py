import os
import torch
import torch.distributed as dist
import numpy as np
import pandas as pd
from skimage.feature import peak_local_max, blob_log
from skimage.morphology import disk, dilation
from typing import Dict, Optional, List
from numpy.typing import NDArray
from skimage.filters import threshold_otsu

from biapy.data.post_processing.post_processing import (
    remove_close_points,
    detection_watershed,
    measure_morphological_props_and_filter,
)
from biapy.utils.misc import is_main_process, is_dist_avail_and_initialized, to_pytorch_format, MetricLogger
from biapy.engine.metrics import (
    detection_metrics,
    multiple_metrics,
    DiceBCELoss,
    DiceLoss,
    CrossEntropyLoss_wrapper,
)
from biapy.data.pre_processing import create_detection_masks
from biapy.engine.base_workflow import Base_Workflow
from biapy.data.data_3D_manipulation import order_dimensions, write_chunked_data
from biapy.data.data_manipulation import save_tif
from biapy.data.dataset import PatchCoords


class Detection_Workflow(Base_Workflow):
    """
    Detection workflow where the goal is to localize objects in the input image, not requiring a pixel-level class.
    More details in `our documentation <https://biapy.readthedocs.io/en/latest/workflows/detection.html>`_.

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
        super(Detection_Workflow, self).__init__(cfg, job_identifier, device, args, **kwargs)

        self.original_test_mask_path = self.prepare_detection_data()

        if self.use_gt:
            self.csv_files = sorted(next(os.walk(self.original_test_mask_path))[2])

        # From now on, no modification of the cfg will be allowed
        self.cfg.freeze()

        # Workflow specific training variables
        self.mask_path = cfg.DATA.TRAIN.GT_PATH
        self.is_y_mask = True
        self.load_Y_val = True

        # Workflow specific test variables
        if self.cfg.TEST.POST_PROCESSING.DET_WATERSHED or self.cfg.TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS:
            self.post_processing["detection_post"] = True
        else:
            self.post_processing["detection_post"] = False

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
            "channels": 1,
        }

        # Multi-head: points + classification
        if self.cfg.MODEL.N_CLASSES > 2:
            self.activations = [{"0": "CE_Sigmoid"}, {"0": "Linear"}]
            self.model_output_channels["channels"] = [self.model_output_channels["channels"], self.cfg.MODEL.N_CLASSES]
            self.multihead = True
        else:
            self.activations = [{"0": "CE_Sigmoid"}]
            self.model_output_channels["channels"] = [self.model_output_channels["channels"]]
            self.multihead = False

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
                self.train_metric_names.append("IoU")
                self.train_metric_best.append("max")

        # Multi-head: detection + classification
        if self.multihead:
            self.train_metric_names.append("IoU (classes)")
            self.train_metric_best += ["max"]

        self.train_metrics.append(
            multiple_metrics(
                num_classes=self.cfg.MODEL.N_CLASSES,
                metric_names=self.train_metric_names,
                device=self.device,
                model_source=self.cfg.MODEL.SOURCE,
            )
        )

        self.test_metrics = []
        self.test_metric_names = []
        for metric in list(set(self.cfg.TEST.METRICS)):
            if metric in ["iou", "jaccard_index"]:
                self.test_metric_names.append("IoU")

        # Multi-head: detection + classification
        if self.multihead:
            self.test_metric_names.append("IoU (classes)")

        self.test_metrics.append(
            multiple_metrics(
                num_classes=self.cfg.MODEL.N_CLASSES,
                metric_names=self.test_metric_names,
                device=self.device,
                model_source=self.cfg.MODEL.SOURCE,
            )
        )

        # Workflow specific metrics calculated in a different way than calling metric_calculation(). These metrics are
        # always calculated
        self.test_extra_metrics = ["Precision", "Recall", "F1", "TP", "FP", "FN"]
        if self.multihead:
            self.test_extra_metrics += ["Precision (class)", "Recall (class)", "F1 (class)", "TP (class)", "FN (class)"]
        self.test_metric_names += self.test_extra_metrics

        if self.cfg.LOSS.TYPE == "CE":
            self.loss = CrossEntropyLoss_wrapper(
                num_classes=self.cfg.MODEL.N_CLASSES,
                multihead=self.multihead,
                model_source=self.cfg.MODEL.SOURCE,
                class_rebalance=self.cfg.LOSS.CLASS_REBALANCE,
            )
        elif self.cfg.LOSS.TYPE == "DICE":
            self.loss = DiceLoss()
        elif self.cfg.LOSS.TYPE == "W_CE_DICE":
            self.loss = DiceBCELoss(w_dice=self.cfg.LOSS.WEIGHTS[0], w_bce=self.cfg.LOSS.WEIGHTS[1])

        super().define_metrics()

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
        if isinstance(output, np.ndarray):
            _output = to_pytorch_format(
                output.copy(),
                self.axes_order,
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

        with torch.no_grad():
            k = 0
            for i, metric in enumerate(list_to_use):
                val = metric(_output, _targets)
                if isinstance(val, dict):
                    for m in val:
                        v = val[m].item() if not torch.isnan(val[m]) else 0
                        out_metrics[list_names_to_use[k]] = v
                        if metric_logger:
                            metric_logger.meters[list_names_to_use[k]].update(v)
                        k += 1
                else:
                    val = val.item() if not torch.isnan(val) else 0  # type: ignore
                    out_metrics[list_names_to_use[i]] = val
                    if metric_logger:
                        metric_logger.meters[list_names_to_use[i]].update(val)
        return out_metrics

    def detection_process(
        self,
        pred: NDArray,
        inference_type: str = "full_image",
        patch_pos: Optional[PatchCoords] = None,
    ):
        """
        Detection workflow engine for test/inference. Process model's prediction to prepare detection output and
        calculate metrics.

        Parameters
        ----------
        pred : 4D Torch tensor
            Model predictions. E.g. ``(z, y, x, channels)`` for both 2D and 3D.

        inference_type : str, optional
            Type of inference. Options: ["per_crop", "merge_patches", "as_3D_stack", "full_image"].

        patch_pos : PatchCoords, optional
            Position of the patch to analize. By setting this the function will take only into account the GT points
            corresponding to the patch at hand.
        """
        assert inference_type in ["per_crop", "merge_patches", "as_3D_stack", "full_image"]
        assert pred.ndim == 4, f"Prediction doesn't have 4 dim: {pred.shape}"

        # Multi-head: points + classification
        if self.multihead:
            class_channel = np.expand_dims(pred[..., -1], -1)
            pred = pred[..., :-1]

        pred_shape = pred.shape
        if self.cfg.TEST.VERBOSE and not self.cfg.TEST.BY_CHUNKS.ENABLE:
            print("Capturing the local maxima ")

        # Find points
        if self.cfg.TEST.DET_TH_TYPE == "auto":
            threshold_abs = threshold_otsu(pred[..., 0])
        else: # manual
            threshold_abs = self.cfg.TEST.DET_MIN_TH_TO_BE_PEAK

        if self.cfg.TEST.DET_POINT_CREATION_FUNCTION == "peak_local_max":
            pred_points = peak_local_max(
                pred[..., 0].astype(np.float32),
                min_distance=self.cfg.TEST.DET_PEAK_LOCAL_MAX_MIN_DISTANCE,
                threshold_abs=threshold_abs,
                exclude_border=self.cfg.TEST.DET_EXCLUDE_BORDER,
            )
        else:
            pred_points = blob_log(
                pred[..., 0] * 255,
                min_sigma=self.cfg.TEST.DET_BLOB_LOG_MIN_SIGMA,
                max_sigma=self.cfg.TEST.DET_BLOB_LOG_MAX_SIGMA,
                num_sigma=self.cfg.TEST.DET_BLOB_LOG_NUM_SIGMA,
                threshold=threshold_abs,
                exclude_border=self.cfg.TEST.DET_EXCLUDE_BORDER,
            )
            pred_points = pred_points[:, :3].astype(int)  # Remove sigma

        # Remove close points per class as post-processing method
        out_dir = self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK
        if self.cfg.TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS and not self.cfg.TEST.BY_CHUNKS.ENABLE:
            out_dir = self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK_POST_PROCESSING
            pred_points = remove_close_points(
                pred_points,
                self.cfg.TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS_RADIUS,
                self.resolution,
                ndim=self.dims,
            )
            assert isinstance(pred_points, list)

        # Decide the class for each point
        pred_points_classes = []
        if self.multihead:
            for point in pred_points:
                if self.dims == 3:
                    point_area = class_channel[
                        max(0, point[0] - 1) : min(pred.shape[0], point[0] + 1),
                        max(0, point[1] - 1) : min(pred.shape[1], point[1] + 1),
                        max(0, point[2] - 1) : min(pred.shape[2], point[2] + 1),
                    ]
                else:
                    point_area = class_channel[
                        max(0, point[0] - 1) : min(pred.shape[0], point[0] + 1),
                        max(0, point[1] - 1) : min(pred.shape[1], point[1] + 1),
                    ]
                instance_classes, instance_classes_count = np.unique(point_area, return_counts=True)
                # Remove background
                if instance_classes[0] == 0:
                    instance_classes = instance_classes[1:]
                    instance_classes_count = instance_classes_count[1:]

                if len(instance_classes) > 0:
                    label_selected = int(instance_classes[np.argmax(instance_classes_count)])
                else:  # Label by default with class 1 in case there was no class info
                    label_selected = 1

                pred_points_classes.append(label_selected)
        else:
            pred_points_classes = [0] * len(pred_points)

        # Create a file with detected point and other image with predictions ids (if GT given)
        if not self.cfg.TEST.BY_CHUNKS.ENABLE:
            file_ext = os.path.splitext(self.current_sample["filename"])[1]
            if self.cfg.TEST.VERBOSE:
                print("Creating the images with detected points . . .")
            points_pred_mask = np.zeros(pred.shape[:-1], dtype=np.uint8)

            if len(pred_points) > 0:
                # Paint the points
                for n, coord in enumerate(pred_points):
                    z, y, x = coord
                    points_pred_mask[z, y, x] = n + 1

                # Dilate and save the detected point image
                for i in range(points_pred_mask.shape[0]):
                    points_pred_mask[i] = dilation(points_pred_mask[i], disk(3))

                if self.multihead:
                    class_channel = np.zeros(points_pred_mask.shape, dtype=np.uint8)
                    for n in range(len(pred_points)):
                        class_channel = np.where(points_pred_mask == n + 1, pred_points_classes[n], class_channel)

                    points_pred_mask = np.concatenate(
                        [
                            np.expand_dims(points_pred_mask, -1),
                            np.expand_dims(class_channel, -1),
                        ],
                        axis=-1,
                    )
                else:
                    points_pred_mask = np.expand_dims(points_pred_mask, -1)

                if file_ext in [".hdf5", ".hdf", ".h5", ".zarr"]:
                    write_chunked_data(
                        np.expand_dims(points_pred_mask, 0),
                        out_dir,
                        self.current_sample["filename"],
                        dtype_str="uint8",
                        verbose=self.cfg.TEST.VERBOSE,
                    )
                else:
                    save_tif(
                        np.expand_dims(points_pred_mask, 0),
                        out_dir,
                        [self.current_sample["filename"]],
                        verbose=self.cfg.TEST.VERBOSE,
                    )

                if self.multihead:
                    points_pred_mask = points_pred_mask[..., 0]
                else:
                    points_pred_mask = points_pred_mask.squeeze()

            # Detection watershed
            if self.cfg.TEST.POST_PROCESSING.DET_WATERSHED:
                data_filename = os.path.join(self.cfg.DATA.TEST.PATH, self.current_sample["filename"])
                w_dir = os.path.join(self.cfg.PATHS.WATERSHED_DIR, self.current_sample["filename"])
                check_wa = w_dir if self.cfg.PROBLEM.DETECTION.DATA_CHECK_MW else None
                assert isinstance(pred_points, list)
                points_pred_mask = detection_watershed(
                    points_pred_mask,
                    pred_points,
                    data_filename,
                    self.cfg.TEST.POST_PROCESSING.DET_WATERSHED_FIRST_DILATION,
                    save_dir=check_wa,
                    ndim=self.dims,
                    donuts_classes=self.cfg.TEST.POST_PROCESSING.DET_WATERSHED_DONUTS_CLASSES,
                    donuts_patch=self.cfg.TEST.POST_PROCESSING.DET_WATERSHED_DONUTS_PATCH,
                    donuts_nucleus_diameter=self.cfg.TEST.POST_PROCESSING.DET_WATERSHED_DONUTS_NUCLEUS_DIAMETER,
                )

                # Instance filtering by properties
                points_pred_mask, d_result = measure_morphological_props_and_filter(
                    points_pred_mask,
                    self.resolution,
                    properties=self.cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS,
                    prop_values=self.cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.VALUES,
                    comp_signs=self.cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.SIGNS,
                )

                if file_ext in [".hdf5", ".hdf", ".h5", ".zarr"]:
                    write_chunked_data(
                        np.expand_dims(np.expand_dims(points_pred_mask, -1), 0),
                        self.cfg.PATHS.RESULT_DIR.DET_ASSOC_POINTS,
                        self.current_sample["filename"],
                        dtype_str="uint8",
                        verbose=self.cfg.TEST.VERBOSE,
                    )
                else:
                    save_tif(
                        np.expand_dims(np.expand_dims(points_pred_mask, 0), -1),
                        self.cfg.PATHS.RESULT_DIR.PER_IMAGE_POST_PROCESSING,
                        [self.current_sample["filename"]],
                        verbose=self.cfg.TEST.VERBOSE,
                    )
            del points_pred_mask

        # Save coords in a couple of csv files
        df = None
        if len(pred_points) > 0:
            aux = np.array(pred_points)
            if self.cfg.PROBLEM.NDIM == "3D":
                prob = pred[aux[:, 0], aux[:, 1], aux[:, 2], 0]
                if self.cfg.TEST.POST_PROCESSING.DET_WATERSHED:
                    df = pd.DataFrame(
                        zip(
                            d_result["labels"],
                            list(aux[:, 0]),
                            list(aux[:, 1]),
                            list(aux[:, 2]),
                            list(prob),
                            list(pred_points_classes),
                            d_result["npixels"],
                            d_result["areas"],
                            d_result["sphericities"],
                            d_result["diameters"],
                            d_result["perimeters"],
                            d_result["comment"],
                            d_result["conditions"],
                        ),
                        columns=[
                            "pred_id",
                            "axis-0",
                            "axis-1",
                            "axis-2",
                            "probability",
                            "class",
                            "npixels",
                            "volume",
                            "sphericity",
                            "diameter",
                            "perimeter (surface area)",
                            "comment",
                            "conditions",
                        ],
                    )
                else:
                    labels = np.array(range(1, len(pred_points) + 1))
                    df = pd.DataFrame(
                        zip(
                            labels,
                            list(aux[:, 0]),
                            list(aux[:, 1]),
                            list(aux[:, 2]),
                            list(prob),
                            list(pred_points_classes),
                        ),
                        columns=[
                            "pred_id",
                            "axis-0",
                            "axis-1",
                            "axis-2",
                            "probability",
                            "class",
                        ],
                    )
            else:
                prob = pred[aux[:, 0], aux[:, 1], 0]
                if self.cfg.TEST.POST_PROCESSING.DET_WATERSHED:
                    df = pd.DataFrame(
                        zip(
                            d_result["labels"],
                            list(aux[:, 0]),
                            list(aux[:, 1]),
                            list(prob),
                            list(pred_points_classes),
                            d_result["npixels"],
                            d_result["areas"],
                            d_result["circularities"],
                            d_result["diameters"],
                            d_result["perimeters"],
                            d_result["elongations"],
                            d_result["comment"],
                            d_result["conditions"],
                        ),
                        columns=[
                            "pred_id",
                            "axis-0",
                            "axis-1",
                            "probability",
                            "class",
                            "npixels",
                            "area",
                            "circularity",
                            "diameter",
                            "perimeter",
                            "elongation",
                            "comment",
                            "conditions",
                        ],
                    )
                else:
                    labels = np.array(range(1, len(pred_points) + 1))
                    df = pd.DataFrame(
                        zip(
                            labels,
                            list(aux[:, 0]),
                            list(aux[:, 1]),
                            list(prob),
                            list(pred_points_classes),
                        ),
                        columns=["pred_id", "axis-0", "axis-1", "probability", "class"],
                    )
            df = df.sort_values(by=["pred_id"])
            del aux

            if not self.multihead:
                df = df.drop(columns=["class"])

            if not self.cfg.TEST.BY_CHUNKS.ENABLE:
                # Save just the points and their probabilities
                os.makedirs(out_dir, exist_ok=True)
                df.to_csv(
                    os.path.join(
                        out_dir,
                        os.path.splitext(self.current_sample["filename"])[0] + "_points.csv",
                    ),
                    index=False,
                )

        # Calculate detection metrics
        if self.use_gt and not self.cfg.TEST.BY_CHUNKS.ENABLE:
            all_channel_d_metrics = [0, 0, 0, 0, 0, 0]
            if self.multihead:
                all_channel_d_metrics += [0, 0, 0, 0, 0]

            # Read the GT coordinates from the CSV file
            csv_filename = os.path.join(
                self.original_test_mask_path, os.path.splitext(self.current_sample["filename"])[0] + ".csv"
            )
            if not os.path.exists(csv_filename):
                if self.cfg.TEST.VERBOSE:
                    print(
                        "WARNING: The CSV file seems to have different name than image. Using the CSV file "
                        "with the same position as the CSV in the directory. Check if it is correct!"
                    )
                csv_filename = os.path.join(self.original_test_mask_path, self.csv_files[self.f_numbers[0]])
                if self.cfg.TEST.VERBOSE:
                    print("Its respective CSV file seems to be: {}".format(csv_filename))
            if self.cfg.TEST.VERBOSE:
                print("Reading GT data from: {}".format(csv_filename))
            df_gt = pd.read_csv(csv_filename)
            df_gt = df_gt.rename(columns=lambda x: x.strip())
            zcoords = df_gt["axis-0"].tolist()
            ycoords = df_gt["axis-1"].tolist()
            if self.cfg.PROBLEM.NDIM == "3D":
                xcoords = df_gt["axis-2"].tolist()
                gt_coordinates = [[z, y, x] for z, y, x in zip(zcoords, ycoords, xcoords)]
            else:
                gt_coordinates = [[0, y, x] for y, x in zip(zcoords, ycoords)]

            if self.cfg.MODEL.N_CLASSES > 2:
                if "class" not in df_gt:
                    raise ValueError("MODEL.N_CLASSES > 2 but no class specified in the CSV file")
            gt_points_classes = None
            if self.multihead:
                if "class" not in df_gt:
                    raise ValueError("'class' column not present in the CSV file")
                gt_points_classes = df_gt["class"].tolist()

            # Take only into account the GT points corresponding to the patch at hand
            if patch_pos:
                patch_gt_coordinates = []
                for j, cor in enumerate(gt_coordinates):
                    z, y, x = cor
                    z, y, x = int(z), int(y), int(x)
                    if (
                        patch_pos.z_start <= z < patch_pos.z_end
                        and patch_pos.y_start <= y < patch_pos.y_end
                        and patch_pos.x_start <= x < patch_pos.x_end
                    ):
                        z = z - patch_pos.z_start
                        y = y - patch_pos.y_start
                        x = x - patch_pos.x_start
                        patch_gt_coordinates.append([z, y, x])
                        if z >= pred_shape[0] or y >= pred_shape[1] or x >= pred_shape[2]:
                            raise ValueError(f"Point [{z},{y},{x}] outside image with shape {pred_shape}")
                gt_coordinates = patch_gt_coordinates.copy()

            roi_to_consider = []
            if self.cfg.TEST.DET_IGNORE_POINTS_OUTSIDE_BOX:
                if self.cfg.PROBLEM.NDIM == "2D":
                    roi_to_consider = [
                        [
                            self.cfg.TEST.DET_IGNORE_POINTS_OUTSIDE_BOX[0],
                            max(
                                pred_shape[0] - self.cfg.TEST.DET_IGNORE_POINTS_OUTSIDE_BOX[0],
                                0,
                            ),
                        ],
                        [
                            self.cfg.TEST.DET_IGNORE_POINTS_OUTSIDE_BOX[1],
                            max(
                                pred_shape[1] - self.cfg.TEST.DET_IGNORE_POINTS_OUTSIDE_BOX[1],
                                0,
                            ),
                        ],
                    ]
                else:
                    roi_to_consider = [
                        [
                            self.cfg.TEST.DET_IGNORE_POINTS_OUTSIDE_BOX[0],
                            max(
                                pred_shape[0] - self.cfg.TEST.DET_IGNORE_POINTS_OUTSIDE_BOX[0],
                                0,
                            ),
                        ],
                        [
                            self.cfg.TEST.DET_IGNORE_POINTS_OUTSIDE_BOX[1],
                            max(
                                pred_shape[1] - self.cfg.TEST.DET_IGNORE_POINTS_OUTSIDE_BOX[1],
                                0,
                            ),
                        ],
                        [
                            self.cfg.TEST.DET_IGNORE_POINTS_OUTSIDE_BOX[2],
                            max(
                                pred_shape[2] - self.cfg.TEST.DET_IGNORE_POINTS_OUTSIDE_BOX[2],
                                0,
                            ),
                        ],
                    ]

            # Calculate detection metrics
            fp, gt_assoc = None, None
            if len(pred_points) > 0:
                d_metrics, gt_assoc, fp = detection_metrics(
                    gt_coordinates,
                    pred_points,
                    true_classes=gt_points_classes,
                    pred_classes=pred_points_classes,
                    tolerance=self.cfg.TEST.DET_TOLERANCE,
                    resolution=self.resolution,
                    bbox_to_consider=roi_to_consider,
                    verbose=self.cfg.TEST.VERBOSE,
                )
                if self.cfg.TEST.VERBOSE:
                    print("Detection metrics: {}".format(d_metrics))

                all_channel_d_metrics[0] += d_metrics["Precision"]
                all_channel_d_metrics[1] += d_metrics["Recall"]
                all_channel_d_metrics[2] += d_metrics["F1"]
                all_channel_d_metrics[3] += d_metrics["TP"]
                all_channel_d_metrics[4] += d_metrics["FP"]
                all_channel_d_metrics[5] += d_metrics["FN"]
                if self.multihead:
                    all_channel_d_metrics[6] += d_metrics["Precision (class)"]
                    all_channel_d_metrics[7] += d_metrics["Recall (class)"]
                    all_channel_d_metrics[8] += d_metrics["F1 (class)"]
                    all_channel_d_metrics[9] += d_metrics["TP (class)"]
                    all_channel_d_metrics[9] += d_metrics["FN (class)"]

                # Save csv files with the associations between GT points and predicted ones
                if gt_assoc is not None:
                    gt_assoc_orig = gt_assoc.copy()
                if fp is not None:
                    fp_orig = fp.copy()
                if self.cfg.PROBLEM.NDIM == "2D":
                    if gt_assoc is not None:
                        gt_assoc = gt_assoc.drop(columns=["axis-0"])
                        gt_assoc = gt_assoc.rename(columns={"axis-1": "axis-0", "axis-2": "axis-1"})
                    if fp is not None:
                        fp = fp.drop(columns=["axis-0"])
                        fp = fp.rename(columns={"axis-1": "axis-0", "axis-2": "axis-1"})
                if gt_assoc is not None:
                    os.makedirs(self.cfg.PATHS.RESULT_DIR.DET_ASSOC_POINTS, exist_ok=True)
                    gt_assoc.to_csv(
                        os.path.join(
                            self.cfg.PATHS.RESULT_DIR.DET_ASSOC_POINTS,
                            os.path.splitext(self.current_sample["filename"])[0] + "_gt_assoc.csv",
                        ),
                        index=False,
                    )
                if fp is not None:
                    os.makedirs(self.cfg.PATHS.RESULT_DIR.DET_ASSOC_POINTS, exist_ok=True)
                    fp.to_csv(
                        os.path.join(
                            self.cfg.PATHS.RESULT_DIR.DET_ASSOC_POINTS,
                            os.path.splitext(self.current_sample["filename"])[0] + "_fp.csv",
                        ),
                        index=False,
                    )
                if gt_assoc is not None:
                    gt_assoc = gt_assoc_orig
                if fp is not None:
                    fp = fp_orig
            else:
                if self.cfg.TEST.VERBOSE:
                    print("No point found to calculate the metrics!")

            if not self.cfg.TEST.BY_CHUNKS.ENABLE:
                for n, metric in enumerate(self.test_extra_metrics):
                    if str(metric).lower() not in self.stats[inference_type]:
                        self.stats[inference_type][str(metric.lower())] = 0
                    self.stats[inference_type][str(metric).lower()] += all_channel_d_metrics[n]

                if self.cfg.TEST.VERBOSE:
                    if len(gt_coordinates) == 0:
                        print("No points found in GT!")
                    print("Creating the image with a summary of detected points and false positives with colors . . .")

                points_pred_mask_color = np.zeros(pred_shape[:-1] + (3,), dtype=np.uint8)

                # TP and FN
                gt_id_img = np.zeros(pred_shape[:-1], dtype=np.uint32)
                for j, cor in enumerate(gt_coordinates):
                    z, y, x = cor
                    z, y, x = int(z), int(y), int(x)
                    if gt_assoc is not None:
                        if gt_assoc[gt_assoc["gt_id"] == j + 1]["tag"].iloc[0] == "TP":
                            points_pred_mask_color[z, y, x] = (0, 255, 0)  # Green
                        elif gt_assoc[gt_assoc["gt_id"] == j + 1]["tag"].iloc[0] == "NC":
                            points_pred_mask_color[z, y, x] = (150, 150, 150)  # Gray
                        else:
                            points_pred_mask_color[z, y, x] = (255, 0, 0)  # Red
                    else:
                        points_pred_mask_color[z, y, x] = (255, 0, 0)  # Red

                    gt_id_img[z, y, x] = j + 1

                # Dilate and save the GT ids for the current class
                for i in range(gt_id_img.shape[0]):
                    gt_id_img[i] = dilation(gt_id_img[i], disk(3))
                if file_ext in [".hdf5", ".hdf", ".h5", ".zarr"]:
                    write_chunked_data(
                        np.expand_dims(np.expand_dims(gt_id_img, -1), 0),
                        self.cfg.PATHS.RESULT_DIR.DET_ASSOC_POINTS,
                        os.path.splitext(self.current_sample["filename"])[0] + "_gt_ids" + file_ext,
                        dtype_str="uint32",
                        verbose=self.cfg.TEST.VERBOSE,
                    )
                else:
                    save_tif(
                        np.expand_dims(np.expand_dims(gt_id_img, 0), -1),
                        self.cfg.PATHS.RESULT_DIR.DET_ASSOC_POINTS,
                        [os.path.splitext(self.current_sample["filename"])[0] + "_gt_ids" + file_ext],
                        verbose=self.cfg.TEST.VERBOSE,
                    )

                # FP
                if fp is not None:
                    for cor in zip(
                        fp["axis-0"].tolist(),
                        fp["axis-1"].tolist(),
                        fp["axis-2"].tolist(),
                    ):
                        z, y, x = cor
                        z, y, x = int(z), int(y), int(x)
                        points_pred_mask_color[z, y, x] = (0, 0, 255)  # Blue

                # Dilate and save the predicted points for the current class
                for i in range(points_pred_mask_color.shape[0]):
                    for j in range(points_pred_mask_color.shape[-1]):
                        points_pred_mask_color[i, ..., j] = dilation(points_pred_mask_color[i, ..., j], disk(3))
                if file_ext in [".hdf5", ".hdf", ".h5", ".zarr"]:
                    write_chunked_data(
                        np.expand_dims(points_pred_mask_color, 0),
                        self.cfg.PATHS.RESULT_DIR.DET_ASSOC_POINTS,
                        self.current_sample["filename"],
                        dtype_str="uint8",
                        verbose=self.cfg.TEST.VERBOSE,
                    )
                else:
                    save_tif(
                        np.expand_dims(points_pred_mask_color, 0),
                        self.cfg.PATHS.RESULT_DIR.DET_ASSOC_POINTS,
                        [self.current_sample["filename"]],
                        verbose=self.cfg.TEST.VERBOSE,
                    )

        return df

    def after_merge_patches(self, pred):
        """
        Steps need to be done after merging all predicted patches into the original image.

        Parameters
        ----------
        pred : Torch Tensor
            Model prediction.
        """
        if pred.shape[0] == 1:
            if self.cfg.PROBLEM.NDIM == "3D":
                pred = pred[0]
            self.detection_process(
                pred,
                inference_type="merge_patches",
            )
        else:
            raise NotImplementedError

    def after_one_patch_prediction_by_chunks(
        self, patch_id: int, patch: NDArray, patch_in_data: PatchCoords, added_pad: List[List[int]]
    ):
        """
        Place any code that needs to be done after predicting one patch in "by chunks" setting.

        Parameters
        ----------
        patch_id: int
            Patch identifier.

        patch : NDArray
            Predicted patch.

        patch_in_data : PatchCoords
            Global coordinates of the patch.

        added_pad: List of list of ints
            Padding added to the patch that should be not taken into account when processing the patch.
        """
        df_patch = self.detection_process(patch, patch_pos=patch_in_data)

        if df_patch is not None and len(df_patch) > 0:
            # Remove possible points in the padded area
            df_patch = df_patch[df_patch["axis-0"] < patch.shape[0] - added_pad[0][1]]
            df_patch = df_patch[df_patch["axis-1"] < patch.shape[1] - added_pad[1][1]]
            df_patch = df_patch[df_patch["axis-2"] < patch.shape[2] - added_pad[2][1]]
            df_patch["axis-0"] = df_patch["axis-0"] - added_pad[0][0]
            df_patch["axis-1"] = df_patch["axis-1"] - added_pad[1][0]
            df_patch["axis-2"] = df_patch["axis-2"] - added_pad[2][0]
            df_patch = df_patch[df_patch["axis-0"] >= 0]
            df_patch = df_patch[df_patch["axis-1"] >= 0]
            df_patch = df_patch[df_patch["axis-2"] >= 0]

            # Add the patch shift to the detected coordinates so they represent global coords
            df_patch["axis-0"] = df_patch["axis-0"] + patch_in_data.z_start
            df_patch["axis-1"] = df_patch["axis-1"] + patch_in_data.y_start
            df_patch["axis-2"] = df_patch["axis-2"] + patch_in_data.x_start

            # Save the csv file
            output_dir = (
                self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK_POST_PROCESSING
                if self.post_processing["detection_post"]
                else self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK
            )
            os.makedirs(output_dir, exist_ok=True)
            _filename, _ = os.path.splitext(os.path.basename(self.current_sample["filename"]))
            df_patch.to_csv(
                os.path.join(
                    output_dir,
                    _filename + "_patch" + str(patch_id).zfill(len(str(len(self.test_generator)))) + "_points.csv",
                ),
                index=False,
            )

    def after_all_patch_prediction_by_chunks(self):
        """
        Place any code that needs to be done after predicting all the patches, one by one, in the "by chunks" setting.
        """
        assert isinstance(self.all_pred, list)
        filename, _ = os.path.splitext(self.current_sample["filename"])
        input_dir = (
            self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK_POST_PROCESSING
            if self.post_processing["detection_post"]
            else self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK
        )
        all_pred_files = sorted(next(os.walk(input_dir))[2])
        all_pred_files = [x for x in all_pred_files if filename + "_patch" in x]
        all_pred_files = [x for x in all_pred_files if "_points.csv" in x and "all_points.csv" not in x]
        if len(all_pred_files) > 0:
            point_counter = 0
            for pred_file in all_pred_files:
                pred_file_path = os.path.join(input_dir, pred_file)
                pred_df = pd.read_csv(pred_file_path, index_col=False)
                pred_df["pred_id"] = pred_df["pred_id"] + point_counter
                point_counter += len(pred_df)
                self.all_pred.append(pred_df)

        if len(self.all_pred) > 0:
            df = pd.concat(self.all_pred, ignore_index=True)

            # Take point coords
            pred_coordinates = []
            if df is None:
                print("No points created, skipping evaluation . . .")
                return
            coordz = df["axis-0"].tolist()
            coordy = df["axis-1"].tolist()
            coordx = df["axis-2"].tolist()
            for z, y, x in zip(coordz, coordy, coordx):
                pred_coordinates.append([z, y, x])

            # Apply post-processing of removing points
            out_dir = self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK
            if self.cfg.TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS:
                out_dir = self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK_POST_PROCESSING
                pred_coordinates, dropped_pos = remove_close_points(  # type: ignore
                    pred_coordinates,
                    self.cfg.TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS_RADIUS,
                    self.resolution,
                    ndim=self.dims,
                    return_drops=True,
                )
                # Remove points from dataframe
                df = df.drop(dropped_pos)

            t_dim, z_dim, y_dim, x_dim, c_dim = order_dimensions(
                self.cfg.DATA.PREPROCESS.ZOOM.ZOOM_FACTOR,
                input_order=self.cfg.DATA.TEST.INPUT_IMG_AXES_ORDER,
                output_order="TZYXC",
                default_value=1,
            )

            df["axis-0"] = df["axis-0"] / z_dim  # type: ignore
            df["axis-1"] = df["axis-1"] / y_dim  # type: ignore
            df["axis-2"] = df["axis-2"] / x_dim  # type: ignore
            os.makedirs(out_dir, exist_ok=True)
            df.to_csv(
                os.path.join(
                    out_dir,
                    filename + "_all_points.csv",
                ),
                index=False,
            )

            # Calculate metrics with all the points
            if self.use_gt:
                print("Calculating detection metrics with all the points found . . .")

                # Read the GT coordinates from the CSV file
                csv_filename = os.path.join(self.original_test_mask_path, os.path.splitext(filename[0])[0] + ".csv")
                if not os.path.exists(csv_filename):
                    if self.cfg.TEST.VERBOSE:
                        print(
                            "WARNING: The CSV file seems to have different name than image. Using the CSV file "
                            "with the same position as the CSV in the directory. Check if it is correct!"
                        )
                    csv_filename = os.path.join(self.original_test_mask_path, self.csv_files[self.f_numbers[0]])
                    if self.cfg.TEST.VERBOSE:
                        print("Its respective CSV file seems to be: {}".format(csv_filename))
                if self.cfg.TEST.VERBOSE:
                    print("Reading GT data from: {}".format(csv_filename))
                df_gt = pd.read_csv(csv_filename)
                df_gt = df_gt.rename(columns=lambda x: x.strip())
                gt_coordinates = [
                    [z, y, x]
                    for z, y, x in zip(
                        df_gt["axis-0"].tolist(),
                        df_gt["axis-1"].tolist(),
                        df_gt["axis-2"].tolist(),
                    )
                ]

                # Measure metrics
                roi_to_consider = []
                if self.cfg.TEST.DET_IGNORE_POINTS_OUTSIDE_BOX:
                    roi_to_consider = [
                        [
                            self.cfg.TEST.DET_IGNORE_POINTS_OUTSIDE_BOX[0],
                            max(
                                self.parallel_data_shape[0] - self.cfg.TEST.DET_IGNORE_POINTS_OUTSIDE_BOX[0],
                                0,
                            ),
                        ],
                        [
                            self.cfg.TEST.DET_IGNORE_POINTS_OUTSIDE_BOX[1],
                            max(
                                self.parallel_data_shape[1] - self.cfg.TEST.DET_IGNORE_POINTS_OUTSIDE_BOX[1],
                                0,
                            ),
                        ],
                        [
                            self.cfg.TEST.DET_IGNORE_POINTS_OUTSIDE_BOX[2],
                            max(
                                self.parallel_data_shape[2] - self.cfg.TEST.DET_IGNORE_POINTS_OUTSIDE_BOX[2],
                                0,
                            ),
                        ],
                    ]
                d_metrics, gt_assoc, fp = detection_metrics(
                    gt_coordinates,
                    pred_coordinates,
                    tolerance=self.cfg.TEST.DET_TOLERANCE,
                    resolution=self.resolution,
                    bbox_to_consider=roi_to_consider,
                    verbose=self.cfg.TEST.VERBOSE,
                )
                print("Detection metrics: {}".format(d_metrics))

                if gt_assoc is not None:
                    os.makedirs(self.cfg.PATHS.RESULT_DIR.DET_ASSOC_POINTS, exist_ok=True)
                    gt_assoc.to_csv(
                        os.path.join(
                            self.cfg.PATHS.RESULT_DIR.DET_ASSOC_POINTS,
                            filename + "_gt_assoc.csv",
                        ),
                        index=False,
                    )
                if fp is not None:
                    os.makedirs(self.cfg.PATHS.RESULT_DIR.DET_ASSOC_POINTS, exist_ok=True)
                    fp.to_csv(
                        os.path.join(
                            self.cfg.PATHS.RESULT_DIR.DET_ASSOC_POINTS,
                            filename + "_fp.csv",
                        ),
                        index=False,
                    )

                for metric in self.test_extra_metrics:
                    if str(metric).lower() not in self.stats["merge_patches"]:
                        self.stats["merge_patches"][str(metric).lower()] = 0
                    self.stats["merge_patches"][str(metric).lower()] += d_metrics[str(metric)]
        else:
            print("No points created for the given sample")

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
            del self.current_sample["X"]

            # In Torchvision the output is a collection of bboxes so there is nothing else to do here
            if self.cfg.MODEL.SOURCE == "torchvision" and pred is None:
                return

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
        assert self.model and self.torchvision_preprocessing
        filename, file_extension = os.path.splitext(self.current_sample["filename"])

        # Convert first to 0-255 range if uint16
        if in_img.dtype == torch.float32:
            if torch.max(in_img) > 1:
                in_img = (self.torchvision_norm.apply_image_norm(in_img)[0] * 255).to(torch.uint8)  # type: ignore
            in_img = in_img.to(torch.uint8)

        # Apply TorchVision pre-processing
        in_img = self.torchvision_preprocessing(in_img)

        pred = self.model(in_img)

        bboxes = pred[0]["boxes"].cpu().numpy()
        if not is_train and len(bboxes) != 0:
            # Extract each output from prediction
            labels = pred[0]["labels"].cpu().numpy()
            scores = pred[0]["scores"].cpu().numpy()

            # Save all info in a csv file
            df = pd.DataFrame(
                zip(
                    labels,
                    scores,
                    bboxes[:, 0],
                    bboxes[:, 1],
                    bboxes[:, 2],
                    bboxes[:, 3],
                ),
                columns=["label", "scores", "x1", "y1", "x2", "y2"],
            )
            df = df.sort_values(by=["label"])
            df.to_csv(
                os.path.join(self.cfg.PATHS.RESULT_DIR.FULL_IMAGE, filename + ".csv"),
                index=False,
            )

        return None

    def after_full_image(self, pred: NDArray):
        """
        Steps that must be executed after generating the prediction by supplying the entire image to the model.

        Parameters
        ----------
        pred : NDArray
            Model prediction.
        """
        if pred.shape[0] == 1:
            if self.cfg.PROBLEM.NDIM == "3D":
                pred = pred[0]
            self.detection_process(
                pred,
                inference_type="full_image",
            )
        else:
            raise NotImplementedError

    def after_all_images(self):
        """
        Steps that must be done after predicting all images.
        """
        super().after_all_images()

    def prepare_detection_data(self) -> str:
        """
        Creates detection ground truth images to train the model based on the ground truth coordinates provided.
        They will be saved in a separate folder in the root path of the ground truth.
        """
        original_test_mask_path = self.cfg.DATA.TEST.GT_PATH
        create_mask = False

        if is_main_process():
            print("############################")
            print("#  PREPARE DETECTION DATA  #")
            print("############################")

            # Create selected channels for train data
            if self.cfg.TRAIN.ENABLE or self.cfg.DATA.TEST.USE_VAL_AS_TEST:
                create_mask = False
                if not os.path.isdir(self.cfg.DATA.TRAIN.DETECTION_MASK_DIR):
                    print(
                        "You select to create detection masks from given .csv files but no file is detected in {}. "
                        "So let's prepare the data. Notice that, if you do not modify 'DATA.TRAIN.DETECTION_MASK_DIR' "
                        "path, this process will be done just once!".format(self.cfg.DATA.TRAIN.DETECTION_MASK_DIR)
                    )
                    create_mask = True
                else:
                    if len(next(os.walk(self.cfg.DATA.TRAIN.DETECTION_MASK_DIR))[2]) != len(
                        next(os.walk(self.cfg.DATA.TRAIN.GT_PATH))[2]
                    ) and len(next(os.walk(self.cfg.DATA.TRAIN.DETECTION_MASK_DIR))[1]) != len(
                        next(os.walk(self.cfg.DATA.TRAIN.GT_PATH))[2]
                    ):
                        print(
                            "Different number of files found in {} and {}. Trying to create the the rest again".format(
                                self.cfg.DATA.TRAIN.GT_PATH,
                                self.cfg.DATA.TRAIN.DETECTION_MASK_DIR,
                            )
                        )
                        create_mask = True

                if create_mask:
                    create_detection_masks(self.cfg)

            # Create selected channels for val data
            if self.cfg.TRAIN.ENABLE and not self.cfg.DATA.VAL.FROM_TRAIN:
                create_mask = False
                if not os.path.isdir(self.cfg.DATA.VAL.DETECTION_MASK_DIR):
                    print(
                        "You select to create detection masks from given .csv files but no file is detected in {}. "
                        "So let's prepare the data. Notice that, if you do not modify 'DATA.VAL.DETECTION_MASK_DIR' "
                        "path, this process will be done just once!".format(self.cfg.DATA.VAL.DETECTION_MASK_DIR)
                    )
                    create_mask = True
                else:
                    if len(next(os.walk(self.cfg.DATA.VAL.DETECTION_MASK_DIR))[2]) != len(
                        next(os.walk(self.cfg.DATA.VAL.GT_PATH))[2]
                    ) and len(next(os.walk(self.cfg.DATA.VAL.DETECTION_MASK_DIR))[1]) != len(
                        next(os.walk(self.cfg.DATA.VAL.GT_PATH))[2]
                    ):
                        print(
                            "Different number of files found in {} and {}. Trying to create the the rest again".format(
                                self.cfg.DATA.VAL.GT_PATH,
                                self.cfg.DATA.VAL.DETECTION_MASK_DIR,
                            )
                        )
                        create_mask = True

                if create_mask:
                    create_detection_masks(self.cfg, data_type="val")

            # Create selected channels for test data once
            if self.cfg.TEST.ENABLE and self.cfg.DATA.TEST.LOAD_GT and not self.cfg.DATA.TEST.USE_VAL_AS_TEST:
                create_mask = False
                if not os.path.isdir(self.cfg.DATA.TEST.DETECTION_MASK_DIR):
                    print(
                        "You select to create detection masks from given .csv files but no file is detected in {}. "
                        "So let's prepare the data. Notice that, if you do not modify 'DATA.TEST.DETECTION_MASK_DIR' "
                        "path, this process will be done just once!".format(self.cfg.DATA.TEST.DETECTION_MASK_DIR)
                    )
                    create_mask = True
                else:
                    if len(next(os.walk(self.cfg.DATA.TEST.DETECTION_MASK_DIR))[2]) != len(
                        next(os.walk(self.cfg.DATA.TEST.GT_PATH))[2]
                    ) and len(next(os.walk(self.cfg.DATA.TEST.DETECTION_MASK_DIR))[1]) != len(
                        next(os.walk(self.cfg.DATA.TEST.GT_PATH))[2]
                    ):
                        print(
                            "Different number of files found in {} and {}. Trying to create the the rest again".format(
                                self.cfg.DATA.TEST.GT_PATH,
                                self.cfg.DATA.TEST.DETECTION_MASK_DIR,
                            )
                        )
                        create_mask = True
                if create_mask:
                    create_detection_masks(self.cfg, data_type="test")

        if is_dist_avail_and_initialized():
            dist.barrier()

        opts = []
        if self.cfg.TRAIN.ENABLE:
            print(
                "DATA.TRAIN.GT_PATH changed from {} to {}".format(
                    self.cfg.DATA.TRAIN.GT_PATH, self.cfg.DATA.TRAIN.DETECTION_MASK_DIR
                )
            )
            opts.extend(["DATA.TRAIN.GT_PATH", self.cfg.DATA.TRAIN.DETECTION_MASK_DIR])

            out_data_order = self.cfg.DATA.TRAIN.INPUT_IMG_AXES_ORDER
            if "C" not in self.cfg.DATA.TRAIN.INPUT_IMG_AXES_ORDER:
                out_data_order += "C"
            print(
                "DATA.TRAIN.INPUT_MASK_AXES_ORDER changed from {} to {}".format(
                    self.cfg.DATA.TRAIN.INPUT_MASK_AXES_ORDER, out_data_order
                )
            )
            opts.extend([f"DATA.TRAIN.INPUT_MASK_AXES_ORDER", out_data_order])

        if not self.cfg.DATA.VAL.FROM_TRAIN:
            print(
                "DATA.VAL.GT_PATH changed from {} to {}".format(
                    self.cfg.DATA.VAL.GT_PATH, self.cfg.DATA.VAL.DETECTION_MASK_DIR
                )
            )
            opts.extend(["DATA.VAL.GT_PATH", self.cfg.DATA.VAL.DETECTION_MASK_DIR])

            out_data_order = self.cfg.DATA.VAL.INPUT_IMG_AXES_ORDER
            if "C" not in self.cfg.DATA.VAL.INPUT_IMG_AXES_ORDER:
                out_data_order += "C"
            print(
                "DATA.VAL.INPUT_MASK_AXES_ORDER changed from {} to {}".format(
                    self.cfg.DATA.VAL.INPUT_MASK_AXES_ORDER, out_data_order
                )
            )
            opts.extend([f"DATA.VAL.INPUT_MASK_AXES_ORDER", out_data_order])

        if self.cfg.TEST.ENABLE and self.cfg.DATA.TEST.LOAD_GT:
            print(
                "DATA.TEST.GT_PATH changed from {} to {}".format(
                    self.cfg.DATA.TEST.GT_PATH, self.cfg.DATA.TEST.DETECTION_MASK_DIR
                )
            )
            opts.extend(["DATA.TEST.GT_PATH", self.cfg.DATA.TEST.DETECTION_MASK_DIR])

            out_data_order = self.cfg.DATA.TEST.INPUT_IMG_AXES_ORDER
            if "C" not in self.cfg.DATA.TEST.INPUT_IMG_AXES_ORDER:
                out_data_order += "C"
            print(
                "DATA.TEST.INPUT_MASK_AXES_ORDER changed from {} to {}".format(
                    self.cfg.DATA.TEST.INPUT_MASK_AXES_ORDER, out_data_order
                )
            )
            opts.extend([f"DATA.TEST.INPUT_MASK_AXES_ORDER", out_data_order])
        self.cfg.merge_from_list(opts)

        return original_test_mask_path
