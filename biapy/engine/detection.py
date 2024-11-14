import os
import math
import torch
import torch.distributed as dist
import numpy as np
import pandas as pd
from skimage.feature import peak_local_max, blob_log
from skimage.morphology import disk, dilation
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from biapy.data.post_processing.post_processing import (
    remove_close_points,
    detection_watershed,
    measure_morphological_props_and_filter,
)
from biapy.utils.misc import (
    is_main_process,
    is_dist_avail_and_initialized,
)
from biapy.utils.util import (
    save_tif,
    write_chunked_data,
    read_chunked_data,
)
from biapy.engine.metrics import (
    detection_metrics,
    jaccard_index,
    DiceBCELoss,
    DiceLoss,
    CrossEntropyLoss_wrapper,
)
from biapy.data.pre_processing import create_detection_masks, norm_range01
from biapy.engine.base_workflow import Base_Workflow
from biapy.data.data_3D_manipulation import order_dimensions


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

        # Activations for each output channel:
        # channel number : 'activation'
        self.activations = [{"0": "CE_Sigmoid"}]

        # Workflow specific training variables
        self.mask_path = cfg.DATA.TRAIN.GT_PATH
        self.is_y_mask = True
        self.load_Y_val = True

        # Workflow specific test variables
        if self.cfg.TEST.POST_PROCESSING.DET_WATERSHED or self.cfg.TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS:
            self.post_processing["detection_post"] = True
        else:
            self.post_processing["detection_post"] = False

        if self.cfg.PROBLEM.NDIM == "3D":
            self.v_size = (
                self.cfg.DATA.TEST.RESOLUTION[0],
                self.cfg.DATA.TEST.RESOLUTION[1],
                self.cfg.DATA.TEST.RESOLUTION[2],
            )
        else:
            self.v_size = (
                1,
                self.cfg.DATA.TEST.RESOLUTION[0],
                self.cfg.DATA.TEST.RESOLUTION[1],
            )

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

        # Workflow specific metrics calculated in a different way than calling metric_calculation(). These metrics are
        # always calculated
        self.test_extra_metrics = ["Precision", "Recall", "F1", "TP", "FP", "FN"]
        self.test_metric_names += self.test_extra_metrics

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
                val = metric(output, targets)
                val = val.item() if not torch.isnan(val) else 0
                out_metrics[list_names_to_use[i]] = val

                if metric_logger is not None:
                    metric_logger.meters[list_names_to_use[i]].update(val)
        return out_metrics

    def detection_process(self, pred, filenames, inference_type="full_image", patch_pos=None):
        """
        Detection workflow engine for test/inference. Process model's prediction to prepare detection output and
        calculate metrics.

        Parameters
        ----------
        pred : 4D Torch tensor
            Model predictions. E.g. ``(z, y, x, channels)`` for both 2D and 3D.

        filenames : List of str
            Predicted image's filenames.

        inference_type : str
            Type of inference. Options: ["per_crop", "merge_patches", "as_3D_stack", "full_image", "by_chunks"].

        patch_pos : List of tuples of ints
            Position of the patch to analize. By setting this the function will take only into account the GT points
            corresponding to the patch at hand.
        """
        assert inference_type in ["per_crop", "merge_patches", "as_3D_stack", "full_image", "by_chunks"]
        assert pred.ndim == 4, f"Prediction doesn't have 4 dim: {pred.shape}"

        file_ext = os.path.splitext(filenames[0])[1]
        ndim = 2 if self.cfg.PROBLEM.NDIM == "2D" else 3
        pred_shape = pred.shape
        if self.cfg.TEST.VERBOSE:
            print("Capturing the local maxima ")
        all_points = []
        all_classes = []
        for channel in range(pred.shape[-1]):
            if self.cfg.TEST.VERBOSE:
                print("Class {}".format(channel + 1))
            if len(self.cfg.TEST.DET_MIN_TH_TO_BE_PEAK) == 1:
                min_th_peak = self.cfg.TEST.DET_MIN_TH_TO_BE_PEAK[0]
            else:
                min_th_peak = self.cfg.TEST.DET_MIN_TH_TO_BE_PEAK[channel]

            # Find points
            if self.cfg.TEST.DET_POINT_CREATION_FUNCTION == "peak_local_max":
                pred_coordinates = peak_local_max(
                    pred[..., channel].astype(np.float32),
                    min_distance=self.cfg.TEST.DET_PEAK_LOCAL_MAX_MIN_DISTANCE,
                    threshold_abs=min_th_peak,
                    exclude_border=self.cfg.TEST.DET_EXCLUDE_BORDER,
                )
            else:
                pred_coordinates = blob_log(
                    pred[..., channel] * 255,
                    min_sigma=self.cfg.TEST.DET_BLOB_LOG_MIN_SIGMA,
                    max_sigma=self.cfg.TEST.DET_BLOB_LOG_MAX_SIGMA,
                    num_sigma=self.cfg.TEST.DET_BLOB_LOG_NUM_SIGMA,
                    threshold=min_th_peak,
                    exclude_border=self.cfg.TEST.DET_EXCLUDE_BORDER,
                )
                pred_coordinates = pred_coordinates[:, :3].astype(int)  # Remove sigma

            # Remove close points per class as post-processing method
            if self.cfg.TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS and not self.by_chunks:
                if len(self.cfg.TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS_RADIUS) == 1:
                    radius = self.cfg.TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS_RADIUS[0]
                else:
                    radius = self.cfg.TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS_RADIUS[channel]

                pred_coordinates = remove_close_points(
                    pred_coordinates, radius, self.cfg.DATA.TEST.RESOLUTION, ndim=ndim
                )

            all_points.append(pred_coordinates)
            c_size = 1 if len(pred_coordinates) == 0 else len(pred_coordinates)
            all_classes.append(np.full(c_size, channel))

        # Remove close points again seeing all classes together, as it can be that a point is detected in both classes
        # if there is not clear distinction between them
        classes = 1 if self.cfg.MODEL.N_CLASSES <= 2 else self.cfg.MODEL.N_CLASSES
        if self.cfg.TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS and classes > 1 and not self.by_chunks:
            if self.cfg.TEST.VERBOSE:
                print("All classes together")
            radius = np.min(self.cfg.TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS_RADIUS)

            all_points = np.concatenate(all_points, axis=0)
            all_classes = np.concatenate(all_classes, axis=0)

            new_points, all_classes = remove_close_points(
                all_points,
                radius,
                self.cfg.DATA.TEST.RESOLUTION,
                classes=all_classes,
                ndim=ndim,
            )

            # Create again list of arrays of all points
            all_points = []
            for i in range(classes):
                all_points.append([])
            for i, c in enumerate(all_classes):
                all_points[c].append(new_points[i])
            del new_points

        # Create a file with detected point and other image with predictions ids (if GT given)
        if not self.by_chunks:
            if self.cfg.TEST.VERBOSE:
                print("Creating the images with detected points . . .")
            points_pred = np.zeros(pred.shape[:-1], dtype=np.uint8)
            for n, pred_coordinates in enumerate(all_points):
                if self.use_gt:
                    pred_id_img = np.zeros(pred_shape[:-1], dtype=np.uint32)
                for j, coord in enumerate(pred_coordinates):
                    z, y, x = coord
                    points_pred[z, y, x] = n + 1
                    if self.use_gt:
                        pred_id_img[z, y, x] = j + 1

                # Dilate and save the prediction ids for the current class
                if self.use_gt:
                    for i in range(pred_id_img.shape[0]):
                        pred_id_img[i] = dilation(pred_id_img[i], disk(3))
                    if file_ext in [".hdf5", ".h5", ".zarr"]:
                        write_chunked_data(
                            np.expand_dims(np.expand_dims(pred_id_img, -1), 0),
                            self.cfg.PATHS.RESULT_DIR.DET_ASSOC_POINTS,
                            os.path.splitext(filenames[0])[0] + "_class" + str(n + 1) + "_pred_ids" + file_ext,
                            dtype_str="uint32",
                            verbose=self.cfg.TEST.VERBOSE,
                        )
                    else:
                        save_tif(
                            np.expand_dims(np.expand_dims(pred_id_img, 0), -1),
                            self.cfg.PATHS.RESULT_DIR.DET_ASSOC_POINTS,
                            [os.path.splitext(filenames[0])[0] + "_class" + str(n + 1) + "_pred_ids.tif"],
                            verbose=self.cfg.TEST.VERBOSE,
                        )

            if self.use_gt:
                del pred_id_img

            # Dilate and save the detected point image
            if len(pred_coordinates) > 0:
                for i in range(points_pred.shape[0]):
                    points_pred[i] = dilation(points_pred[i], disk(3))
            if file_ext in [".hdf5", ".h5", ".zarr"]:
                write_chunked_data(
                    np.expand_dims(np.expand_dims(points_pred, -1), 0),
                    self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK,
                    filenames[0],
                    dtype_str="uint8",
                    verbose=self.cfg.TEST.VERBOSE,
                )
            else:
                save_tif(
                    np.expand_dims(np.expand_dims(points_pred, 0), -1),
                    self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK,
                    filenames,
                    verbose=self.cfg.TEST.VERBOSE,
                )

            # Detection watershed
            if self.cfg.TEST.POST_PROCESSING.DET_WATERSHED:
                data_filename = os.path.join(self.cfg.DATA.TEST.PATH, filenames[0])
                w_dir = os.path.join(self.cfg.PATHS.WATERSHED_DIR, filenames[0])
                check_wa = w_dir if self.cfg.PROBLEM.DETECTION.DATA_CHECK_MW else None
                points_pred = detection_watershed(
                    points_pred,
                    all_points,
                    data_filename,
                    self.cfg.TEST.POST_PROCESSING.DET_WATERSHED_FIRST_DILATION,
                    nclasses=classes,
                    ndim=ndim,
                    donuts_classes=self.cfg.TEST.POST_PROCESSING.DET_WATERSHED_DONUTS_CLASSES,
                    donuts_patch=self.cfg.TEST.POST_PROCESSING.DET_WATERSHED_DONUTS_PATCH,
                    donuts_nucleus_diameter=self.cfg.TEST.POST_PROCESSING.DET_WATERSHED_DONUTS_NUCLEUS_DIAMETER,
                    save_dir=check_wa,
                )

                # Instance filtering by properties
                points_pred, d_result = measure_morphological_props_and_filter(
                    points_pred,
                    self.cfg.DATA.TEST.RESOLUTION,
                    properties=self.cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS,
                    prop_values=self.cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.VALUES,
                    comp_signs=self.cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.SIGNS,
                )

                if file_ext in [".hdf5", ".h5", ".zarr"]:
                    write_chunked_data(
                        np.expand_dims(np.expand_dims(points_pred, -1), 0),
                        self.cfg.PATHS.RESULT_DIR.DET_ASSOC_POINTS,
                        filenames[0],
                        dtype_str="uint8",
                        verbose=self.cfg.TEST.VERBOSE,
                    )
                else:
                    save_tif(
                        np.expand_dims(np.expand_dims(points_pred, 0), -1),
                        self.cfg.PATHS.RESULT_DIR.PER_IMAGE_POST_PROCESSING,
                        filenames,
                        verbose=self.cfg.TEST.VERBOSE,
                    )
            del points_pred

        # Save coords in a couple of csv files
        aux = np.concatenate(all_points, axis=0)
        df = None
        if len(aux) != 0:
            if self.cfg.PROBLEM.NDIM == "3D":
                prob = pred[aux[:, 0], aux[:, 1], aux[:, 2], all_classes]
                prob = np.concatenate(prob, axis=0)
                all_classes = np.concatenate(all_classes, axis=0)
                if self.cfg.TEST.POST_PROCESSING.DET_WATERSHED:
                    df = pd.DataFrame(
                        zip(
                            d_result["labels"],
                            list(aux[:, 0]),
                            list(aux[:, 1]),
                            list(aux[:, 2]),
                            list(prob),
                            list(all_classes),
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
                    df = df.sort_values(by=["pred_id"])
                else:
                    labels = []
                    for i, pred_coordinates in enumerate(all_points):
                        for j in range(len(pred_coordinates)):
                            labels.append(j + 1)

                    df = pd.DataFrame(
                        zip(
                            labels,
                            list(aux[:, 0]),
                            list(aux[:, 1]),
                            list(aux[:, 2]),
                            list(prob),
                            list(all_classes),
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
                    df = df.sort_values(by=["pred_id"])
            else:
                aux = aux[:, 1:]
                prob = pred[0, aux[:, 0], aux[:, 1], all_classes]
                prob = np.concatenate(prob, axis=0)
                all_classes = np.concatenate(all_classes, axis=0)
                if self.cfg.TEST.POST_PROCESSING.DET_WATERSHED:
                    df = pd.DataFrame(
                        zip(
                            d_result["labels"],
                            list(aux[:, 0]),
                            list(aux[:, 1]),
                            list(prob),
                            list(all_classes),
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
                    df = df.sort_values(by=["pred_id"])
                else:
                    df = pd.DataFrame(
                        zip(
                            list(aux[:, 0]),
                            list(aux[:, 1]),
                            list(prob),
                            list(all_classes),
                        ),
                        columns=["axis-0", "axis-1", "probability", "class"],
                    )
                    df = df.sort_values(by=["axis-0"])
            del aux

            # Save just the points and their probabilities
            os.makedirs(self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK, exist_ok=True)
            df.to_csv(
                os.path.join(
                    self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK,
                    os.path.splitext(filenames[0])[0] + "_points.csv",
                )
            )

        # Calculate detection metrics
        if self.use_gt:
            all_channel_d_metrics = [0, 0, 0, 0, 0, 0]
            dfs = []
            gt_all_coords = []

            # Read the GT coordinates from the CSV file
            csv_filename = os.path.join(self.original_test_mask_path, os.path.splitext(filenames[0])[0] + ".csv")
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
            class_info = None
            if self.cfg.PROBLEM.NDIM == "3D":
                xcoords = df_gt["axis-2"].tolist()
                gt_coordinates = [[z, y, x] for z, y, x in zip(zcoords, ycoords, xcoords)]
            else:
                gt_coordinates = [[0, y, x] for y, x in zip(zcoords, ycoords)]

            if classes > 1:
                if "class" not in df_gt:
                    raise ValueError("MODEL.N_CLASSES > 1 but no class specified in CSV file")
                else:
                    class_info = df_gt["class"].tolist()

            # Take only into account the GT points corresponding to the patch at hand
            if patch_pos is not None:
                patch_gt_coordinates = []
                for j, cor in enumerate(gt_coordinates):
                    z, y, x = cor
                    z, y, x = int(z), int(y), int(x)
                    if (
                        patch_pos[0][0] <= z < patch_pos[0][1]
                        and patch_pos[1][0] <= y < patch_pos[1][1]
                        and patch_pos[2][0] <= x < patch_pos[2][1]
                    ):
                        z = z - patch_pos[0][0]
                        y = y - patch_pos[1][0]
                        x = x - patch_pos[2][0]
                        patch_gt_coordinates.append([z, y, x])
                        if z >= pred_shape[0] or y >= pred_shape[1] or x >= pred_shape[2]:
                            raise ValueError(f"Point [{z},{y},{x}] outside image with shape {pred_shape}")
                gt_coordinates = patch_gt_coordinates.copy()
            gt_all_coords.append(gt_coordinates)

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
            for ch, pred_coordinates in enumerate(all_points):
                # If there was class info take only the points related to the class at hand
                if class_info is not None:
                    class_points = []
                    for i in range(len(gt_coordinates)):
                        if int(class_info[i]) == ch:
                            class_points.append(gt_coordinates[i])
                    gt_coordinates = class_points.copy()
                    del class_points

                # Calculate detection metrics
                if len(pred_coordinates) > 0:
                    if self.cfg.TEST.VERBOSE:
                        print("Detection (class " + str(ch + 1) + ")")
                    d_metrics, gt_assoc, fp = detection_metrics(
                        gt_coordinates,
                        pred_coordinates,
                        tolerance=self.cfg.TEST.DET_TOLERANCE[ch],
                        voxel_size=self.v_size,
                        return_assoc=True,
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

                    # Save csv files with the associations between GT points and predicted ones
                    if gt_assoc is not None and fp is not None:
                        dfs.append([gt_assoc.copy(), fp.copy()])
                    else:
                        if gt_assoc is not None:
                            dfs.append([gt_assoc.copy(), None])
                        if fp is not None:
                            dfs.append([None, fp.copy()])
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
                                os.path.splitext(filenames[0])[0] + "_class" + str(ch + 1) + "_gt_assoc.csv",
                            )
                        )
                    if fp is not None:
                        os.makedirs(self.cfg.PATHS.RESULT_DIR.DET_ASSOC_POINTS, exist_ok=True)
                        fp.to_csv(
                            os.path.join(
                                self.cfg.PATHS.RESULT_DIR.DET_ASSOC_POINTS,
                                os.path.splitext(filenames[0])[0] + "_class" + str(ch + 1) + "_fp.csv",
                            )
                        )
                else:
                    if self.cfg.TEST.VERBOSE:
                        print("No point found to calculate the metrics!")

            if self.cfg.TEST.VERBOSE:
                if len(gt_coordinates) == 0:
                    print("No points found in GT!")
                print("All classes " + str(ch + 1))
            for k in range(len(all_channel_d_metrics)):
                all_channel_d_metrics[k] = all_channel_d_metrics[k] / len(all_points)
            if self.cfg.TEST.VERBOSE:
                print(
                    "Detection metrics: {}".format(
                        [
                            "Precision",
                            all_channel_d_metrics[0],
                            "Recall",
                            all_channel_d_metrics[1],
                            "F1",
                            all_channel_d_metrics[2],
                        ]
                    )
                )

            if not self.by_chunks:
                for n, metric in enumerate(self.test_extra_metrics):
                    if str(metric).lower() not in self.stats[inference_type]:
                        self.stats[inference_type][str(metric.lower())] = 0
                    self.stats[inference_type][str(metric).lower()] += all_channel_d_metrics[n]

            if self.cfg.TEST.VERBOSE:
                print("Creating the image with a summary of detected points and false positives with colors . . .")
            if not self.by_chunks:
                points_pred = np.zeros(pred_shape[:-1] + (3,), dtype=np.uint8)
                for ch, gt_coords in enumerate(gt_all_coords):
                    # if gt_assoc is None:
                    gt_assoc, fp = None, None
                    if len(dfs) > 0 and len(dfs[ch]) > 0:
                        if dfs[ch][0] is not None:
                            gt_assoc = dfs[ch][0]
                        if dfs[ch][1] is not None:
                            fp = dfs[ch][1]

                    # TP and FN
                    gt_id_img = np.zeros(pred_shape[:-1], dtype=np.uint32)
                    for j, cor in enumerate(gt_coords):
                        z, y, x = cor
                        z, y, x = int(z), int(y), int(x)
                        if gt_assoc is not None:
                            if gt_assoc[gt_assoc["gt_id"] == j + 1]["tag"].iloc[0] == "TP":
                                points_pred[z, y, x] = (0, 255, 0)  # Green
                            elif gt_assoc[gt_assoc["gt_id"] == j + 1]["tag"].iloc[0] == "NC":
                                points_pred[z, y, x] = (150, 150, 150)  # Gray
                            else:
                                points_pred[z, y, x] = (255, 0, 0)  # Red
                        else:
                            points_pred[z, y, x] = (255, 0, 0)  # Red

                        gt_id_img[z, y, x] = j + 1

                    # Dilate and save the GT ids for the current class
                    for i in range(gt_id_img.shape[0]):
                        gt_id_img[i] = dilation(gt_id_img[i], disk(3))
                    if file_ext in [".hdf5", ".h5", ".zarr"]:
                        write_chunked_data(
                            np.expand_dims(np.expand_dims(gt_id_img, -1), 0),
                            self.cfg.PATHS.RESULT_DIR.DET_ASSOC_POINTS,
                            os.path.splitext(filenames[0])[0] + "_class" + str(ch + 1) + "_gt_ids" + file_ext,
                            dtype_str="uint32",
                            verbose=self.cfg.TEST.VERBOSE,
                        )
                    else:
                        save_tif(
                            np.expand_dims(np.expand_dims(gt_id_img, 0), -1),
                            self.cfg.PATHS.RESULT_DIR.DET_ASSOC_POINTS,
                            [os.path.splitext(filenames[0])[0] + "_class" + str(ch + 1) + "_gt_ids.csv"],
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
                            points_pred[z, y, x] = (0, 0, 255)  # Blue

                # Dilate and save the predicted points for the current class
                for i in range(points_pred.shape[0]):
                    for j in range(points_pred.shape[-1]):
                        points_pred[i, ..., j] = dilation(points_pred[i, ..., j], disk(3))
                if file_ext in [".hdf5", ".h5", ".zarr"]:
                    write_chunked_data(
                        np.expand_dims(points_pred, 0),
                        self.cfg.PATHS.RESULT_DIR.DET_ASSOC_POINTS,
                        filenames[0],
                        dtype_str="uint8",
                        verbose=self.cfg.TEST.VERBOSE,
                    )
                else:
                    save_tif(
                        np.expand_dims(points_pred, 0),
                        self.cfg.PATHS.RESULT_DIR.DET_ASSOC_POINTS,
                        filenames,
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
                [self.current_sample["filename"]],
                inference_type="merge_patches",
            )
        else:
            raise NotImplementedError

    def process_patch(
        self,
        z,
        y,
        x,
        _filename,
        total_patches,
        c,
        pred,
        d,
        file_ext,
        z_dim,
        y_dim,
        x_dim,
    ):
        """
        Process a patch for the detection workflow.

        Parameters
        ----------
        z : int
            Patch z index.

        y : int
            Patch y index.

        x : int
            Patch x index.

        _filename : str
            Filename of the predicted image H5/Zarr.

        total_patches : int
            Total number of patches.

        c : int
            Current patch number.

        pred : 4D numpy array
            Model prediction.

        d : int
            Number of digits of the total patches.

        file_ext : str
            File extension of the predicted image.

        z_dim : int
            Dimension of the z axis.

        y_dim : int
            Dimension of the y axis.

        x_dim : int
            Dimension of the x axis.

        Returns
        -------
        df_patch : DataFrame
            Detected points in the patch.

        fname : str
            Filename of the patch.

        """
        print("Processing patch {}/{} of image".format(c, total_patches))
        if self.cfg.TEST.VERBOSE:
            print(
                "D: z: {}-{}, y: {}-{}, x: {}-{}".format(
                    max(
                        0,
                        z * self.cfg.DATA.PATCH_SIZE[0] - self.cfg.DATA.TEST.PADDING[0],
                    ),
                    min(
                        z_dim,
                        self.cfg.DATA.PATCH_SIZE[0] * (z + 1) + self.cfg.DATA.TEST.PADDING[0],
                    ),
                    max(
                        0,
                        y * self.cfg.DATA.PATCH_SIZE[1] - self.cfg.DATA.TEST.PADDING[1],
                    ),
                    min(
                        y_dim,
                        self.cfg.DATA.PATCH_SIZE[1] * (y + 1) + self.cfg.DATA.TEST.PADDING[1],
                    ),
                    max(
                        0,
                        x * self.cfg.DATA.PATCH_SIZE[2] - self.cfg.DATA.TEST.PADDING[2],
                    ),
                    min(
                        x_dim,
                        self.cfg.DATA.PATCH_SIZE[2] * (x + 1) + self.cfg.DATA.TEST.PADDING[2],
                    ),
                )
            )

        fname = _filename + "_patch" + str(c).zfill(d) + file_ext

        slices = (
            slice(None),
            slice(
                max(0, z * self.cfg.DATA.PATCH_SIZE[0] - self.cfg.DATA.TEST.PADDING[0]),
                min(
                    z_dim,
                    self.cfg.DATA.PATCH_SIZE[0] * (z + 1) + self.cfg.DATA.TEST.PADDING[0],
                ),
            ),
            slice(
                max(0, y * self.cfg.DATA.PATCH_SIZE[1] - self.cfg.DATA.TEST.PADDING[1]),
                min(
                    y_dim,
                    self.cfg.DATA.PATCH_SIZE[1] * (y + 1) + self.cfg.DATA.TEST.PADDING[1],
                ),
            ),
            slice(
                max(0, x * self.cfg.DATA.PATCH_SIZE[2] - self.cfg.DATA.TEST.PADDING[2]),
                min(
                    x_dim,
                    self.cfg.DATA.PATCH_SIZE[2] * (x + 1) + self.cfg.DATA.TEST.PADDING[2],
                ),
            ),
            slice(None),  # Channel
        )

        data_ordered_slices = order_dimensions(
            slices,
            input_order="TZYXC",
            output_order=self.cfg.TEST.BY_CHUNKS.INPUT_IMG_AXES_ORDER,
            default_value=0,
        )

        if "C" not in self.cfg.TEST.BY_CHUNKS.INPUT_IMG_AXES_ORDER:
            expected_out_data_order = self.cfg.TEST.BY_CHUNKS.INPUT_IMG_AXES_ORDER + "C"
        else:
            expected_out_data_order = self.cfg.TEST.BY_CHUNKS.INPUT_IMG_AXES_ORDER

        current_order = np.array(range(len(pred.shape)))
        transpose_order = order_dimensions(
            current_order,
            input_order=expected_out_data_order,
            output_order="TZYXC",
            default_value=np.nan,
        )
        transpose_order = [x for x in transpose_order if not np.isnan(x)]
        transpose_order = current_order[np.array(transpose_order)]
        raw_patch = pred[data_ordered_slices]
        patch = raw_patch.transpose(transpose_order)

        patch_pos = [(k.start, k.stop) for k in slices[1:]]
        df_patch = self.detection_process(patch, [fname], patch_pos=patch_pos)
        if df_patch is not None:  # if there is at least one point detected

            if z * self.cfg.DATA.PATCH_SIZE[0] - self.cfg.DATA.TEST.PADDING[0] >= 0:  # if a patch was added
                df_patch["axis-0"] = (
                    df_patch["axis-0"] - self.cfg.DATA.TEST.PADDING[0]
                )  # shift the coordinates to the correct patch position
            if y * self.cfg.DATA.PATCH_SIZE[1] - self.cfg.DATA.TEST.PADDING[1] >= 0:
                df_patch["axis-1"] = df_patch["axis-1"] - self.cfg.DATA.TEST.PADDING[1]
            if x * self.cfg.DATA.PATCH_SIZE[2] - self.cfg.DATA.TEST.PADDING[2] >= 0:
                df_patch["axis-2"] = df_patch["axis-2"] - self.cfg.DATA.TEST.PADDING[2]

            df_patch = df_patch[df_patch["axis-0"] >= 0]  # remove all coordinate from the previous patch
            df_patch = df_patch[
                df_patch["axis-0"] < self.cfg.DATA.PATCH_SIZE[0]
            ]  # remove all coordinate from the next patch
            df_patch = df_patch[df_patch["axis-1"] >= 0]
            df_patch = df_patch[df_patch["axis-1"] < self.cfg.DATA.PATCH_SIZE[1]]
            df_patch = df_patch[df_patch["axis-2"] >= 0]
            df_patch = df_patch[df_patch["axis-2"] < self.cfg.DATA.PATCH_SIZE[2]]

            df_patch = df_patch.reset_index(drop=True)

            # add the patch shift to the detected coordinates
            shift = np.array(
                [
                    z * self.cfg.DATA.PATCH_SIZE[0],
                    y * self.cfg.DATA.PATCH_SIZE[1],
                    x * self.cfg.DATA.PATCH_SIZE[2],
                ]
            )

            df_patch["axis-0"] = df_patch["axis-0"] + shift[0]
            df_patch["axis-1"] = df_patch["axis-1"] + shift[1]
            df_patch["axis-2"] = df_patch["axis-2"] + shift[2]

            if not df_patch.empty:
                # save the detected points in the patch
                os.makedirs(self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK, exist_ok=True)
                df_patch.to_csv(
                    os.path.join(
                        self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK,
                        os.path.splitext(fname)[0] + "_points_global.csv",
                    )
                )

                return df_patch, fname

        return None, None

    def after_merge_patches_by_chunks_proccess_patch(self, filename):
        """
        Place any code that needs to be done after merging all predicted patches into the original image
        but in the process made chunk by chunk. This function will operate patch by patch defined by
        ``DATA.PATCH_SIZE`` + ``DATA.PADDING``.

        Parameters
        ----------
        filename : List of str
            Filename of the predicted image H5/Zarr.
        """

        _filename, file_ext = os.path.splitext(os.path.basename(filename))
        print("Detection workflow pipeline continues for image {}".format(_filename))

        # Load H5/Zarr
        pred_file, pred = read_chunked_data(filename)

        t_dim, z_dim, c_dim, y_dim, x_dim = order_dimensions(pred.shape, self.cfg.TEST.BY_CHUNKS.INPUT_IMG_AXES_ORDER)
        pred_shape = [z_dim, y_dim, x_dim]

        # Fill the new data
        z_vols = math.ceil(z_dim / self.cfg.DATA.PATCH_SIZE[0])
        y_vols = math.ceil(y_dim / self.cfg.DATA.PATCH_SIZE[1])
        x_vols = math.ceil(x_dim / self.cfg.DATA.PATCH_SIZE[2])
        total_patches = z_vols * y_vols * x_vols
        d = len(str(total_patches))
        c = 1

        workers = self.cfg.SYSTEM.NUM_WORKERS if self.cfg.SYSTEM.NUM_WORKERS > 0 else None
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for z in tqdm(range(z_vols), disable=not is_main_process()):
                for y in range(y_vols):
                    for x in range(x_vols):
                        futures.append(
                            executor.submit(
                                self.process_patch,
                                z,
                                y,
                                x,
                                _filename,
                                total_patches,
                                c,
                                pred,
                                d,
                                file_ext,
                                z_dim,
                                y_dim,
                                x_dim,
                            )
                        )
                        c += 1

            df = None
            all_patches = []
            for future in as_completed(futures):
                try:
                    data = future.result()
                    df_patch, fname = data
                    if df_patch is not None:
                        df_patch["file"] = fname
                        all_patches.append(df_patch)
                        print("Current total patch with detection: {}".format(len(all_patches)))
                except Exception as e:
                    print("Error while detecting patch", e)

        df = pd.concat(all_patches, ignore_index=True)

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
        if self.cfg.TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS and self.by_chunks:
            radius = self.cfg.TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS_RADIUS[0]
            pred_coordinates, dropped_pos = remove_close_points(
                pred_coordinates,
                radius,
                self.cfg.DATA.TEST.RESOLUTION,
                ndim=3,
                return_drops=True,
            )

            # Remove points from dataframe
            df = df.drop(dropped_pos)

        # Save large csv with all point of all patches
        df = df.sort_values(by=["file"])

        t_dim, z_dim, y_dim, x_dim, c_dim = order_dimensions(
            self.cfg.DATA.PREPROCESS.ZOOM.ZOOM_FACTOR,
            input_order=self.cfg.TEST.BY_CHUNKS.INPUT_IMG_AXES_ORDER,
            output_order="TZYXC",
            default_value=1,
        )

        df["axis-0"] = df["axis-0"] / z_dim
        df["axis-1"] = df["axis-1"] / y_dim
        df["axis-2"] = df["axis-2"] / x_dim
        df.to_csv(
            os.path.join(
                self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK,
                _filename + "_all_points.csv",
            )
        )

        if self.cfg.TEST.BY_CHUNKS.FORMAT == "h5":
            pred_file.close()

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
            d_metrics, gt_assoc, fp = detection_metrics(
                gt_coordinates,
                pred_coordinates,
                tolerance=self.cfg.TEST.DET_TOLERANCE[0],
                voxel_size=self.v_size,
                return_assoc=True,
                bbox_to_consider=roi_to_consider,
                verbose=self.cfg.TEST.VERBOSE,
            )
            print("Detection metrics: {}".format(d_metrics))

            for metric in self.test_extra_metrics:
                if str(metric).lower() not in self.stats["by_chunks"]:
                    self.stats["by_chunks"][str(metric).lower()] = 0
                self.stats["by_chunks"][str(metric).lower()] += d_metrics[str(metric)]

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
            
            # Save test_output if the user wants to export the model to BMZ later
            if "test_input" not in self.bmz_config:
                if self.cfg.PROBLEM.NDIM == "2D":
                    self.bmz_config["test_input"] = self.current_sample["X"][0][
                        : self.cfg.DATA.PATCH_SIZE[0], : self.cfg.DATA.PATCH_SIZE[1]
                    ].copy()
                else:
                    self.bmz_config["test_input"] = self.current_sample["X"][0][
                        : self.cfg.DATA.PATCH_SIZE[0], : self.cfg.DATA.PATCH_SIZE[1], : self.cfg.DATA.PATCH_SIZE[2]
                    ].copy()


            ##################
            ### FULL IMAGE ###
            ##################
            # Make the prediction
            with torch.cuda.amp.autocast():
                pred = self.model_call_func(self.current_sample["X"])
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
                
                # Check activations to be inserted as postprocessing in BMZ
                self.bmz_config["postprocessing"] = []
                act = list(self.activations[0].values())
                for ac in act:
                    if ac in ["CE_Sigmoid","Sigmoid"]:
                        self.bmz_config["postprocessing"].append("sigmoid")

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
        filename, file_extension = os.path.splitext(self.current_sample["filename"])

        # Convert first to 0-255 range if uint16
        if in_img.dtype == torch.float32:
            if torch.max(in_img) > 1:
                in_img = (norm_range01(in_img, torch.uint8)[0] * 255).to(torch.uint8)
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

    def after_full_image(self, pred):
        """
        Steps that must be executed after generating the prediction by supplying the entire image to the model.

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
                [self.current_sample["filename"]],
                inference_type="full_image",
            )
        else:
            raise NotImplementedError

    def after_all_images(self):
        """
        Steps that must be done after predicting all images.
        """
        super().after_all_images()

    def prepare_detection_data(self):
        """
        Creates detection ground truth images to train the model based on the ground truth coordinates provided.
        They will be saved in a separate folder in the root path of the ground truth.
        """
        original_test_mask_path = None
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
            if not self.cfg.DATA.VAL.FROM_TRAIN:
                print(
                    "DATA.VAL.GT_PATH changed from {} to {}".format(
                        self.cfg.DATA.VAL.GT_PATH, self.cfg.DATA.VAL.DETECTION_MASK_DIR
                    )
                )
                opts.extend(["DATA.VAL.GT_PATH", self.cfg.DATA.VAL.DETECTION_MASK_DIR])
            if create_mask and self.cfg.DATA.TRAIN.INPUT_MASK_AXES_ORDER != "TZCYX":
                print(
                    f"DATA.TRAIN.INPUT_MASK_AXES_ORDER changed from '{self.cfg.DATA.TRAIN.INPUT_MASK_AXES_ORDER}' to 'TZCYX'. Remember to set that value "
                    " in future runs if you reuse the mask created."
                )
                opts.extend(["DATA.TRAIN.INPUT_MASK_AXES_ORDER", "TZCYX"])
        if self.cfg.TEST.ENABLE and self.cfg.DATA.TEST.LOAD_GT:
            print(
                "DATA.TEST.GT_PATH changed from {} to {}".format(
                    self.cfg.DATA.TEST.GT_PATH, self.cfg.DATA.TEST.DETECTION_MASK_DIR
                )
            )
            opts.extend(["DATA.TEST.GT_PATH", self.cfg.DATA.TEST.DETECTION_MASK_DIR])
            original_test_mask_path = self.cfg.DATA.TEST.GT_PATH
        self.cfg.merge_from_list(opts)

        return original_test_mask_path
