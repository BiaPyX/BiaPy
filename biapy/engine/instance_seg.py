import os
import torch
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.segmentation import clear_border
from skimage.transform import resize
from skimage.morphology import disk, dilation
import torch.distributed as dist
from typing import Dict, Optional, List, Tuple, Any
from numpy.typing import NDArray
from scipy.spatial import distance_matrix


from biapy.data.post_processing.post_processing import (
    watershed_by_channels,
    voronoi_on_mask,
    measure_morphological_props_and_filter,
    repare_large_blobs,
    apply_binary_mask,
    create_synapses,
    remove_close_points,
)
from biapy.data.pre_processing import create_instance_channels
from biapy.utils.matching import matching, wrapper_matching_dataset_lazy
from biapy.engine.metrics import (
    jaccard_index,
    instance_segmentation_loss,
    multiple_metrics,
    detection_metrics,
)
from biapy.engine.base_workflow import Base_Workflow
from biapy.utils.misc import (
    is_main_process,
    is_dist_avail_and_initialized,
    to_pytorch_format,
    to_numpy_format,
    MetricLogger,
)
from biapy.data.data_manipulation import read_img_as_ndarray, save_tif
from biapy.data.data_3D_manipulation import read_chunked_data, read_chunked_nested_data, ensure_3d_shape
from biapy.data.dataset import PatchCoords


class Instance_Segmentation_Workflow(Base_Workflow):
    """
    Instance segmentation workflow where the goal is to assign an unique id, i.e. integer, to each object of the input image.
    More details in `our documentation <https://biapy.readthedocs.io/en/latest/workflows/instance_segmentation.html>`_.

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
        super(Instance_Segmentation_Workflow, self).__init__(cfg, job_identifier, device, args, **kwargs)

        self.original_train_input_mask_axes_order = self.cfg.DATA.TRAIN.INPUT_MASK_AXES_ORDER
        self.original_test_path, self.original_test_mask_path = self.prepare_instance_data()

        # Merging the image
        self.all_matching_stats_merge_patches = []
        self.all_matching_stats_merge_patches_post = []
        self.stats["inst_stats_merge_patches"] = None
        self.stats["inst_stats_merge_patches_post"] = None
        # Multi-head: instances + classification
        if self.multihead:
            self.all_class_stats_merge_patches = []
            self.all_class_stats_merge_patches_post = []
            self.stats["class_stats_merge_patches"] = None
            self.stats["class_stats_merge_patches_post"] = None

        # As 3D stack
        self.all_matching_stats_as_3D_stack = []
        self.all_matching_stats_as_3D_stack_post = []
        self.stats["inst_stats_as_3D_stack"] = None
        self.stats["inst_stats_as_3D_stack_post"] = None
        # Multi-head: instances + classification
        if self.multihead:
            self.all_class_stats_as_3D_stack = []
            self.all_class_stats_as_3D_stack_post = []
            self.stats["class_stats_as_3D_stack"] = None
            self.stats["class_stats_as_3D_stack_post"] = None

        # Full image
        self.all_matching_stats = []
        self.all_matching_stats_post = []
        self.stats["inst_stats"] = None
        self.stats["inst_stats_post"] = None
        # Multi-head: instances + classification
        if self.multihead:
            self.all_class_stats = []
            self.all_class_stats_post = []
            self.stats["class_stats"] = None
            self.stats["class_stats_post"] = None

        self.instance_ths = {}
        self.instance_ths["TYPE"] = self.cfg.PROBLEM.INSTANCE_SEG.DATA_MW_TH_TYPE
        self.instance_ths["TH_BINARY_MASK"] = self.cfg.PROBLEM.INSTANCE_SEG.DATA_MW_TH_BINARY_MASK
        self.instance_ths["TH_CONTOUR"] = self.cfg.PROBLEM.INSTANCE_SEG.DATA_MW_TH_CONTOUR
        self.instance_ths["TH_FOREGROUND"] = self.cfg.PROBLEM.INSTANCE_SEG.DATA_MW_TH_FOREGROUND
        self.instance_ths["TH_DISTANCE"] = self.cfg.PROBLEM.INSTANCE_SEG.DATA_MW_TH_DISTANCE
        self.instance_ths["TH_POINTS"] = self.cfg.PROBLEM.INSTANCE_SEG.DATA_MW_TH_POINTS

        # From now on, no modification of the cfg will be allowed
        self.cfg.freeze()

        # Workflow specific training variables
        self.mask_path = cfg.DATA.TRAIN.GT_PATH
        self.is_y_mask = True
        self.load_Y_val = True
        if self.cfg.TEST.ENABLE and self.cfg.DATA.TEST.LOAD_GT:
            self.test_gt_filenames = sorted(next(os.walk(self.original_test_mask_path))[2])
            if len(self.test_gt_filenames) == 0:
                self.test_gt_filenames = sorted(next(os.walk(self.original_test_mask_path))[1])

        # Specific instance segmentation post-processing
        if (
            self.cfg.TEST.POST_PROCESSING.VORONOI_ON_MASK
            or self.cfg.TEST.POST_PROCESSING.CLEAR_BORDER
            or self.cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.ENABLE
            or self.cfg.TEST.POST_PROCESSING.REPARE_LARGE_BLOBS_SIZE != -1
        ):
            self.post_processing["instance_post"] = True
        else:
            self.post_processing["instance_post"] = False
        self.instances_already_created = False

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
        self.activations = {}
        self.model_output_channels = {"type": "mask", "channels": 1}
        if self.cfg.PROBLEM.INSTANCE_SEG.TYPE == "regular":
            if self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS == "C":
                self.activations = {"0": "CE_Sigmoid"}
                self.model_output_channels["channels"] = 1
            if self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS == "Dv2":
                self.activations = {"0": "Linear"}
                self.model_output_channels["channels"] = 2
            elif self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS == "BC":
                self.activations = {":": "CE_Sigmoid"}
                self.model_output_channels["channels"] = 2
            elif self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS == "BP":
                self.activations = {":": "CE_Sigmoid"}
                self.model_output_channels["channels"] = 2
            elif self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS == "BCM":
                self.activations = {":": "CE_Sigmoid"}
                self.model_output_channels["channels"] = 3
            elif self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS == "A":
                self.activations = {":": "CE_Sigmoid"}
                self.model_output_channels["channels"] = self.dims
            elif self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS in ["BDv2", "BD"]:
                self.activations = {"0": "CE_Sigmoid", "1": "Linear"}
                self.model_output_channels["channels"] = 2
            elif self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS in ["BCD", "BCDv2"]:
                self.activations = {"0": "CE_Sigmoid", "1": "CE_Sigmoid", "2": "Linear"}
                self.model_output_channels["channels"] = 3
        else:  # synapses
            if self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS == "B":
                self.activations = {":": "CE_Sigmoid"}
                self.model_output_channels["channels"] = 2
            elif self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS == "BF":
                self.activations = {"0": "CE_Sigmoid", "1": "Linear"}
                self.model_output_channels["channels"] = 1 + self.dims

        if len(self.activations) == 0:
            raise ValueError("Something wrong happen during instance seg. channel configuration. Contact BiaPy team")

        # Multi-head: instances + classification
        if self.cfg.MODEL.N_CLASSES > 2:
            self.activations = [self.activations, {"0": "Linear"}]
            self.model_output_channels["channels"] = [self.model_output_channels["channels"], self.cfg.MODEL.N_CLASSES]
            self.multihead = True
        else:
            self.activations = [self.activations]
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
        if self.cfg.PROBLEM.INSTANCE_SEG.TYPE == "regular":
            if self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS == "BC":
                self.train_metric_names = ["IoU (B channel)", "IoU (C channel)"]
                self.train_metric_best += ["max", "max"]
            elif self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS == "C":
                self.train_metric_names = ["IoU (C channel)"]
                self.train_metric_best += ["max"]
            elif self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS in ["BCM"]:
                self.train_metric_names = [
                    "IoU (B channel)",
                    "IoU (C channel)",
                    "IoU (M channel)",
                ]
                self.train_metric_best += ["max", "max", "max"]
            elif self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS in ["A"]:
                if self.cfg.PROBLEM.NDIM == "3D":
                    self.train_metric_names = [
                        "IoU (affinity Z)",
                        "IoU (affinity Y)",
                        "IoU (affinity X)",
                    ]
                    self.train_metric_best += ["max", "max", "max"]
                else:
                    self.train_metric_names = ["IoU (affinity Y)", "IoU (affinity X)"]
                    self.train_metric_best += ["max", "max"]
            elif self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS == "BP":
                self.train_metric_names = ["IoU (B channel)", "IoU (P channel)"]
                self.train_metric_best += ["max", "max"]
            elif self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS == "BD":
                self.train_metric_names = ["IoU (B channel)", "L1 (distance channel)"]
                self.train_metric_best += ["max", "min"]
            elif self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS in ["BCD", "BCDv2"]:
                self.train_metric_names = [
                    "IoU (B channel)",
                    "IoU (C channel)",
                    "L1 (distance channel)",
                ]
                self.train_metric_best += ["max", "max", "min"]
            elif self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS == "BDv2":
                self.train_metric_names = ["IoU (B channel)", "L1 (distance channel)"]
                self.train_metric_best += ["max", "min"]
            elif self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS == "Dv2":
                self.train_metric_names = ["L1 (distance channel)"]
                self.train_metric_best += ["min"]

            # Multi-head: instances + classification
            if self.multihead:
                self.train_metric_names.append("IoU (classes)")
                self.train_metric_best += ["max"]
                # Used to calculate IoU with the classification results
                self.jaccard_index_matching = jaccard_index(device=self.device, num_classes=self.cfg.MODEL.N_CLASSES)
        else:  # synapses
            if self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS == "BF":
                self.train_metric_names = ["IoU (B channel)"]
                self.train_metric_best = ["max"]
                if self.cfg.PROBLEM.NDIM == "3D":
                    self.train_metric_names += [
                        "L1 (Z distance)",
                        "L1 (Y distance)",
                        "L1 (X distance)",
                    ]
                    self.train_metric_best += ["max", "max", "max"]
                else:
                    self.train_metric_names += ["L1 (Y distance)", "L1 (X distance)"]
                    self.train_metric_best += ["max", "max"]
            elif self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS == "B":
                self.train_metric_names = ["IoU (pre-sites)", "IoU (post-sites)"]
                self.train_metric_best = ["max", "max"]

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
        if self.cfg.PROBLEM.INSTANCE_SEG.TYPE == "regular":
            if self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS == "BC":
                self.test_metric_names = ["IoU (B channel)", "IoU (C channel)"]
            if self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS == "C":
                self.test_metric_names = ["IoU (C channel)"]
            elif self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS in ["BCM"]:
                self.test_metric_names = [
                    "IoU (B channel)",
                    "IoU (C channel)",
                    "IoU (M channel)",
                ]
            elif self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS in ["A"]:
                if self.cfg.PROBLEM.NDIM == "3D":
                    self.test_metric_names = [
                        "IoU (affinity Z)",
                        "IoU (affinity Y)",
                        "IoU (affinity X)",
                    ]
                else:
                    self.test_metric_names = ["IoU (affinity Y)", "IoU (affinity X)"]
            elif self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS == "BP":
                self.test_metric_names = ["IoU (B channel)", "IoU (P channel)"]
            elif self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS == "BD":
                self.test_metric_names = ["IoU (B channel)", "L1 (distance channel)"]
            elif self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS in ["BCD", "BCDv2"]:
                self.test_metric_names = [
                    "IoU (B channel)",
                    "IoU (C channel)",
                    "L1 (distance channel)",
                ]
            elif self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS == "BDv2":
                self.test_metric_names = ["IoU (B channel)", "L1 (distance channel)"]
            elif self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS == "Dv2":
                self.test_metric_names = ["L1 (distance channel)"]
        else:  # synapses
            if self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS == "BF":
                self.test_metric_names = ["IoU (B channel)"]
                if self.cfg.PROBLEM.NDIM == "3D":
                    self.test_metric_names += [
                        "L1 (Z distance)",
                        "L1 (Y distance)",
                        "L1 (X distance)",
                    ]
                else:
                    self.test_metric_names += ["L1 (Y distance)", "L1 (X distance)"]
            elif self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS == "B":
                self.test_metric_names = ["IoU (pre-sites)", "IoU (post-sites)"]

        # Multi-head: instances + classification
        if self.multihead:
            self.test_metric_names.append("IoU (classes)")
            # Used to calculate IoU with the classification results
            self.jaccard_index_matching = jaccard_index(device="cpu", num_classes=self.cfg.MODEL.N_CLASSES)

        self.test_metrics.append(
            multiple_metrics(
                num_classes=self.cfg.MODEL.N_CLASSES,
                metric_names=self.test_metric_names,
                device=self.device,
                model_source=self.cfg.MODEL.SOURCE,
            )
        )

        self.loss = instance_segmentation_loss(
            self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNEL_WEIGHTS,
            self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS,
            self.cfg.PROBLEM.INSTANCE_SEG.DISTANCE_CHANNEL_MASK,
            self.cfg.MODEL.N_CLASSES,
            class_rebalance=self.cfg.LOSS.CLASS_REBALANCE,
            instance_type=self.cfg.PROBLEM.INSTANCE_SEG.TYPE,
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
                    val = val.item() if not torch.isnan(val) else 0
                    out_metrics[list_names_to_use[i]] = val
                    if metric_logger:
                        metric_logger.meters[list_names_to_use[i]].update(val)
        return out_metrics

    def instance_seg_process(self, pred, filenames, out_dir, out_dir_post_proc, calculate_metrics: bool = True):
        """
        Instance segmentation workflow engine for test/inference. Process model's prediction to prepare
        instance segmentation output and calculate metrics.

        Parameters
        ----------
        pred : 4D/5D Torch tensor
            Model predictions. E.g. ``(z, y, x, channels)`` for both 2D and 3D.

        filenames : List of str
            Predicted image's filenames.

        out_dir : path
            Output directory to save the instances.

        out_dir_post_proc : path
            Output directory to save the post-processed instances.

        calculate_metrics : bool, optional
            Whether to calculate or not the metrics.
        """
        assert pred.ndim == 4, f"Prediction doesn't have 4 dim: {pred.shape}"

        #############################
        ### INSTANCE SEGMENTATION ###
        #############################
        if not self.instances_already_created:
            # Multi-head: instances + classification
            if self.multihead:
                class_channel = np.expand_dims(pred[..., -1], -1)
                pred = pred[..., :-1]

            print("Creating instances with watershed . . .")
            w_dir = os.path.join(self.cfg.PATHS.WATERSHED_DIR, filenames[0])
            check_wa = w_dir if self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHECK_MW else None

            w_pred = watershed_by_channels(
                pred,
                self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS,
                ths=self.instance_ths,
                remove_before=self.cfg.PROBLEM.INSTANCE_SEG.DATA_REMOVE_BEFORE_MW,
                thres_small_before=self.cfg.PROBLEM.INSTANCE_SEG.DATA_REMOVE_SMALL_OBJ_BEFORE,
                seed_morph_sequence=self.cfg.PROBLEM.INSTANCE_SEG.SEED_MORPH_SEQUENCE,
                seed_morph_radius=self.cfg.PROBLEM.INSTANCE_SEG.SEED_MORPH_RADIUS,
                erode_and_dilate_foreground=self.cfg.PROBLEM.INSTANCE_SEG.ERODE_AND_DILATE_FOREGROUND,
                fore_erosion_radius=self.cfg.PROBLEM.INSTANCE_SEG.FORE_EROSION_RADIUS,
                fore_dilation_radius=self.cfg.PROBLEM.INSTANCE_SEG.FORE_DILATION_RADIUS,
                rmv_close_points=self.cfg.TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS,
                remove_close_points_radius=self.cfg.TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS_RADIUS,
                resolution=self.resolution,
                save_dir=check_wa,
                watershed_by_2d_slices=self.cfg.PROBLEM.INSTANCE_SEG.WATERSHED_BY_2D_SLICES,
            )

            # Multi-head: instances + classification
            if self.multihead:
                print("Adapting class channel . . .")
                labels = np.unique(w_pred)[1:]
                new_class_channel = np.zeros(w_pred.shape, dtype=w_pred.dtype)
                # Classify each instance counting the most prominent class of all the pixels that compose it
                for l in labels:
                    instance_classes, instance_classes_count = np.unique(class_channel[w_pred == l], return_counts=True)

                    # Remove background
                    if instance_classes[0] == 0:
                        instance_classes = instance_classes[1:]
                        instance_classes_count = instance_classes_count[1:]

                    if len(instance_classes) > 0:
                        label_selected = int(instance_classes[np.argmax(instance_classes_count)])
                    else:  # Label by default with class 1 in case there was no class info
                        label_selected = 1
                    new_class_channel = np.where(w_pred == l, label_selected, new_class_channel)

                class_channel = new_class_channel
                class_channel = class_channel.squeeze()
                del new_class_channel
                save_tif(
                    np.expand_dims(
                        np.concatenate(
                            [
                                np.expand_dims(w_pred.squeeze(), -1),
                                np.expand_dims(class_channel, -1),
                            ],
                            axis=-1,
                        ),
                        0,
                    ),
                    out_dir,
                    filenames,
                    verbose=self.cfg.TEST.VERBOSE,
                )
            else:
                save_tif(
                    np.expand_dims(np.expand_dims(w_pred, -1), 0),
                    out_dir,
                    filenames,
                    verbose=self.cfg.TEST.VERBOSE,
                )

            # Add extra dimension if working in 2D
            if w_pred.ndim == 2:
                w_pred = np.expand_dims(w_pred, 0)
        else:
            w_pred = pred.squeeze()
            if w_pred.ndim == 2:
                w_pred = np.expand_dims(w_pred, 0)

        results = None
        results_class = None
        if (
            calculate_metrics
            and self.cfg.TEST.MATCHING_STATS
            and (self.cfg.DATA.TEST.LOAD_GT or self.cfg.DATA.TEST.USE_VAL_AS_TEST)
        ):
            print("Calculating matching stats . . .")

            # Need to load instance labels, as Y are binary channels used for IoU calculation
            if self.cfg.TEST.ANALIZE_2D_IMGS_AS_3D_STACK and len(self.test_filenames) == w_pred.shape[0]:
                del self.current_sample["Y"]
                _Y = np.zeros(w_pred.shape, dtype=w_pred.dtype)
                for i in range(len(self.test_filenames)):
                    test_file = os.path.join(self.original_test_mask_path, self.test_filenames[i])
                    _Y[i] = read_img_as_ndarray(test_file, is_3d=False).squeeze()
            else:
                test_file = os.path.join(self.original_test_mask_path, self.test_filenames[self.f_numbers[0]])
                if not os.path.exists(test_file):
                    print(
                        "WARNING: The image seems to have different name than its mask file. Using the mask file that's "
                        "in the same spot (within the mask files list) where the image is in its own list of images. Check if it is correct!"
                    )
                    test_file = os.path.join(
                        self.original_test_mask_path,
                        self.test_gt_filenames[self.f_numbers[0]],
                    )
                print(f"Its respective image seems to be: {test_file}")

                if test_file.endswith(".zarr"):
                    if self.cfg.DATA.TEST.INPUT_ZARR_MULTIPLE_DATA:
                        _, _Y = read_chunked_nested_data(
                            test_file,
                            self.cfg.DATA.TEST.INPUT_ZARR_MULTIPLE_DATA_GT_PATH,
                        )
                    else:
                        _, _Y = read_chunked_data(test_file)
                    _Y = np.array(_Y).squeeze()
                else:
                    _Y = read_img_as_ndarray(test_file, is_3d=self.cfg.PROBLEM.NDIM == "3D").squeeze()

            # Multi-head: instances + classification
            if self.multihead:
                # Channel check
                error_shape = None
                if self.cfg.PROBLEM.NDIM == "2D" and _Y.ndim != 3:
                    error_shape = (256, 256, 2)
                elif self.cfg.PROBLEM.NDIM == "3D" and _Y.ndim != 4:
                    error_shape = (40, 256, 256, 2)
                if error_shape:
                    raise ValueError(
                        f"Image {test_file} wrong dimension. In instance segmentation, when 'MODEL.N_CLASSES' are "
                        f"more than 2 labels need to have two channels, e.g. {error_shape}, containing the instance "
                        "segmentation map (first channel) and classification map (second channel)."
                    )

                # Separate instance and classification channels
                _Y_classes = _Y[..., 1]  # Classes
                _Y = _Y[..., 0]  # Instances

                # Measure class IoU
                class_iou = self.jaccard_index_matching(
                    torch.as_tensor(class_channel.squeeze().astype(np.int32)),
                    torch.as_tensor(_Y_classes.squeeze().astype(np.int32)),
                )
                class_iou = class_iou.item() if not torch.isnan(class_iou) else 0
                print(f"Class IoU: {class_iou}")
                results_class = class_iou

            if _Y.ndim == 2:
                _Y = np.expand_dims(_Y, 0)

            # For torchvision models that resize need to rezise the images
            if w_pred.shape != _Y.shape:
                _Y = resize(_Y, w_pred.shape, order=0)

            # Convert instances to integer
            if _Y.dtype == np.float32:
                _Y = _Y.astype(np.uint32)
            if _Y.dtype == np.float64:
                _Y = _Y.astype(np.uint64)

            diff_ths_colored_img = abs(
                len(self.cfg.TEST.MATCHING_STATS_THS_COLORED_IMG) - len(self.cfg.TEST.MATCHING_STATS_THS)
            )
            colored_img_ths = self.cfg.TEST.MATCHING_STATS_THS_COLORED_IMG + [-1] * diff_ths_colored_img

            results = matching(_Y, w_pred, thresh=self.cfg.TEST.MATCHING_STATS_THS, report_matches=True)
            for i in range(len(results)):
                # Extract TPs, FPs and FNs from the resulting matching data structure
                r_stats = results[i]
                thr = r_stats["thresh"]

                # TP and FN
                gt_ids = r_stats["gt_ids"][1:]
                matched_pairs = r_stats["matched_pairs"]
                gt_match = [x[0] for x in matched_pairs]
                gt_unmatch = [x for x in gt_ids if x not in gt_match]
                matched_scores = list(r_stats["matched_scores"]) + [0 for _ in gt_unmatch]
                pred_match = [x[1] for x in matched_pairs] + [-1 for _ in gt_unmatch]
                tag = ["TP" if score >= thr else "FN" for score in matched_scores]

                # FPs
                pred_ids = r_stats["pred_ids"][1:]
                fp_instances = [x for x in pred_ids if x not in pred_match]
                fp_instances += [pred_id for score, pred_id in zip(matched_scores, pred_match) if score < thr]

                # Save csv files
                df = pd.DataFrame(
                    zip(gt_match + gt_unmatch, pred_match, matched_scores, tag),
                    columns=["gt_id", "pred_id", "iou", "tag"],
                )
                df = df.sort_values(by=["gt_id"])
                df_fp = pd.DataFrame(zip(fp_instances), columns=["pred_id"])

                os.makedirs(self.cfg.PATHS.RESULT_DIR.INST_ASSOC_POINTS, exist_ok=True)
                df.to_csv(
                    os.path.join(
                        self.cfg.PATHS.RESULT_DIR.INST_ASSOC_POINTS,
                        os.path.splitext(filenames[0])[0] + "_th_{}_gt_assoc.csv".format(thr),
                    ),
                    index=False,
                )
                df_fp.to_csv(
                    os.path.join(
                        self.cfg.PATHS.RESULT_DIR.INST_ASSOC_POINTS,
                        os.path.splitext(filenames[0])[0] + "_th_{}_fp.csv".format(thr),
                    ),
                    index=False,
                )
                del r_stats["matched_scores"]
                del r_stats["matched_tps"]
                del r_stats["matched_pairs"]
                del r_stats["pred_ids"]
                del r_stats["gt_ids"]
                print("DatasetMatching: {}".format(r_stats))

                if colored_img_ths[i] != -1 and colored_img_ths[i] == thr:
                    print("Creating the image with a summary of detected points and false positives with colors . . .")
                    colored_result = np.zeros(w_pred.shape + (3,), dtype=np.uint8)

                    print("Painting TPs and FNs . . .")
                    for j in tqdm(range(len(gt_match)), disable=not is_main_process()):
                        color = (0, 255, 0) if tag[j] == "TP" else (255, 0, 0)  # Green or red
                        colored_result[np.where(_Y == gt_match[j])] = color
                    for j in tqdm(range(len(gt_unmatch)), disable=not is_main_process()):
                        colored_result[np.where(_Y == gt_unmatch[j])] = (
                            255,
                            0,
                            0,
                        )  # Red

                    print("Painting FPs . . .")
                    for j in tqdm(range(len(fp_instances)), disable=not is_main_process()):
                        colored_result[np.where(w_pred == fp_instances[j])] = (
                            0,
                            0,
                            255,
                        )  # Blue

                    save_tif(
                        np.expand_dims(colored_result, 0),
                        self.cfg.PATHS.RESULT_DIR.INST_ASSOC_POINTS,
                        [os.path.splitext(filenames[0])[0] + "_th_{}.tif".format(thr)],
                        verbose=self.cfg.TEST.VERBOSE,
                    )
                    del colored_result

        ###################
        # Post-processing #
        ###################
        if self.cfg.TEST.POST_PROCESSING.REPARE_LARGE_BLOBS_SIZE != -1:
            if self.cfg.PROBLEM.NDIM == "2D":
                w_pred = w_pred[0]
            w_pred = repare_large_blobs(w_pred[0], self.cfg.TEST.POST_PROCESSING.REPARE_LARGE_BLOBS_SIZE)
            if self.cfg.PROBLEM.NDIM == "2D":
                w_pred = np.expand_dims(w_pred, 0)

        if self.cfg.TEST.POST_PROCESSING.VORONOI_ON_MASK:
            w_pred = voronoi_on_mask(
                w_pred,
                pred,
                th=self.cfg.TEST.POST_PROCESSING.VORONOI_TH,
                verbose=self.cfg.TEST.VERBOSE,
            )
        del pred

        if self.cfg.TEST.POST_PROCESSING.CLEAR_BORDER:
            print("Clearing borders . . .")
            if self.cfg.PROBLEM.NDIM == "2D":
                w_pred = w_pred[0]
            w_pred = clear_border(w_pred)
            if self.cfg.PROBLEM.NDIM == "2D":
                w_pred = np.expand_dims(w_pred, 0)

        if (
            self.cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.ENABLE
            or self.cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.ENABLE
        ):
            if self.cfg.PROBLEM.NDIM == "2D":
                w_pred = w_pred[0]
            w_pred, d_result = measure_morphological_props_and_filter(
                w_pred,
                self.resolution,
                filter_instances=self.cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.ENABLE,
                properties=self.cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS,
                prop_values=self.cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.VALUES,
                comp_signs=self.cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.SIGNS,
            )
            if self.cfg.PROBLEM.NDIM == "2D":
                w_pred = np.expand_dims(w_pred, 0)

            # Save all instance stats
            if self.cfg.PROBLEM.NDIM == "2D":
                df = pd.DataFrame(
                    zip(
                        np.array(d_result["labels"], dtype=np.uint64),
                        list(d_result["centers"][:, 0]),
                        list(d_result["centers"][:, 1]),
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
                        "label",
                        "axis-0",
                        "axis-1",
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
                df = pd.DataFrame(
                    zip(
                        np.array(d_result["labels"], dtype=np.uint64),
                        list(d_result["centers"][:, 0]),
                        list(d_result["centers"][:, 1]),
                        list(d_result["centers"][:, 2]),
                        d_result["npixels"],
                        d_result["areas"],
                        d_result["sphericities"],
                        d_result["diameters"],
                        d_result["perimeters"],
                        d_result["elongations"],
                        d_result["comment"],
                        d_result["conditions"],
                    ),
                    columns=[
                        "label",
                        "axis-0",
                        "axis-1",
                        "axis-2",
                        "npixels",
                        "volume",
                        "sphericity",
                        "diameter",
                        "perimeter (surface area)",
                        "elongation (P2A)",
                        "comment",
                        "conditions",
                    ],
                )
            df = df.sort_values(by=["label"])
            df.to_csv(
                os.path.join(out_dir, os.path.splitext(filenames[0])[0] + "_full_stats.csv"),
                index=False,
            )
            # Save only remain instances stats
            df = df[df["comment"].str.contains("Strange") == False]
            os.makedirs(out_dir_post_proc, exist_ok=True)
            df.to_csv(
                os.path.join(
                    out_dir_post_proc,
                    os.path.splitext(filenames[0])[0] + "_filtered_stats.csv",
                ),
                index=False,
            )
            del df

        results_post_proc = None
        results_class_post_proc = None
        if self.post_processing["instance_post"]:
            if self.cfg.PROBLEM.NDIM == "2D":
                w_pred = w_pred[0]

            # Multi-head: instances + classification
            if self.multihead:
                class_channel = np.where(w_pred > 0, class_channel, 0)  # Adapt changes to post-processed w_pred
                save_tif(
                    np.expand_dims(
                        np.concatenate(
                            [
                                np.expand_dims(w_pred, -1),
                                np.expand_dims(class_channel, -1),
                            ],
                            axis=-1,
                        ),
                        0,
                    ),
                    out_dir_post_proc,
                    filenames,
                    verbose=self.cfg.TEST.VERBOSE,
                )
            else:
                save_tif(
                    np.expand_dims(np.expand_dims(w_pred, -1), 0),
                    out_dir_post_proc,
                    filenames,
                    verbose=self.cfg.TEST.VERBOSE,
                )

            if (
                calculate_metrics
                and self.cfg.TEST.MATCHING_STATS
                and (self.cfg.DATA.TEST.LOAD_GT or self.cfg.DATA.TEST.USE_VAL_AS_TEST)
            ):
                # Multi-head: instances + classification
                if self.multihead:
                    # Measure class IoU
                    class_iou = self.jaccard_index_matching(
                        torch.as_tensor(class_channel.squeeze().astype(np.int32)),
                        torch.as_tensor(_Y_classes.squeeze().astype(np.int32)),
                    )
                    class_iou = class_iou.item() if not torch.isnan(class_iou) else 0
                    print(f"Class IoU (post-processing): {class_iou}")
                    results_class_post_proc = class_iou

                if self.cfg.PROBLEM.NDIM == "2D":
                    w_pred = np.expand_dims(w_pred, 0)

                print("Calculating matching stats after post-processing . . .")
                results_post_proc = matching(
                    _Y,
                    w_pred,
                    thresh=self.cfg.TEST.MATCHING_STATS_THS,
                    report_matches=True,
                )

                for i in range(len(results_post_proc)):
                    # Extract TPs, FPs and FNs from the resulting matching data structure
                    r_stats = results_post_proc[i]
                    thr = r_stats["thresh"]

                    # TP and FN
                    gt_ids = r_stats["gt_ids"][1:]
                    matched_pairs = r_stats["matched_pairs"]
                    gt_match = [x[0] for x in matched_pairs]
                    gt_unmatch = [x for x in gt_ids if x not in gt_match]
                    matched_scores = list(r_stats["matched_scores"]) + [0 for _ in gt_unmatch]
                    pred_match = [x[1] for x in matched_pairs] + [-1 for _ in gt_unmatch]
                    tag = ["TP" if score >= thr else "FN" for score in matched_scores]

                    # FPs
                    pred_ids = r_stats["pred_ids"][1:]
                    fp_instances = [x for x in pred_ids if x not in pred_match]
                    fp_instances += [pred_id for score, pred_id in zip(matched_scores, pred_match) if score < thr]

                    # Save csv files
                    df = pd.DataFrame(
                        zip(gt_match + gt_unmatch, pred_match, matched_scores, tag),
                        columns=["gt_id", "pred_id", "iou", "tag"],
                    )
                    df = df.sort_values(by=["gt_id"])
                    df_fp = pd.DataFrame(zip(fp_instances), columns=["pred_id"])

                    os.makedirs(self.cfg.PATHS.RESULT_DIR.INST_ASSOC_POINTS, exist_ok=True)
                    df.to_csv(
                        os.path.join(
                            self.cfg.PATHS.RESULT_DIR.INST_ASSOC_POINTS,
                            os.path.splitext(filenames[0])[0] + "_post-proc_th_{}_gt_assoc.csv".format(thr),
                        ),
                        index=False,
                    )
                    df_fp.to_csv(
                        os.path.join(
                            self.cfg.PATHS.RESULT_DIR.INST_ASSOC_POINTS,
                            os.path.splitext(filenames[0])[0] + "_post-proc_th_{}_fp.csv".format(thr),
                        ),
                        index=False,
                    )
                    del r_stats["matched_scores"]
                    del r_stats["matched_tps"]
                    del r_stats["matched_pairs"]
                    del r_stats["pred_ids"]
                    del r_stats["gt_ids"]
                    print("DatasetMatching: {}".format(r_stats))

                    if colored_img_ths[i] != -1 and colored_img_ths[i] == thr:
                        print(
                            "Creating the image with a summary of detected points and false positives with colors . . ."
                        )
                        colored_result = np.zeros(w_pred.shape + (3,), dtype=np.uint8)

                        print("Painting TPs and FNs . . .")
                        for j in tqdm(range(len(gt_match)), disable=not is_main_process()):
                            color = (0, 255, 0) if tag[j] == "TP" else (255, 0, 0)  # Green or red
                            colored_result[np.where(_Y == gt_match[j])] = color
                        for j in tqdm(range(len(gt_unmatch)), disable=not is_main_process()):
                            colored_result[np.where(_Y == gt_unmatch[j])] = (
                                255,
                                0,
                                0,
                            )  # Red

                        print("Painting FPs . . .")
                        for j in tqdm(range(len(fp_instances)), disable=not is_main_process()):
                            colored_result[np.where(w_pred == fp_instances[j])] = (
                                0,
                                0,
                                255,
                            )  # Blue

                        save_tif(
                            np.expand_dims(colored_result, 0),
                            self.cfg.PATHS.RESULT_DIR.INST_ASSOC_POINTS,
                            [os.path.splitext(filenames[0])[0] + "_post-proc_th_{}.tif".format(thr)],
                            verbose=self.cfg.TEST.VERBOSE,
                        )
                        del colored_result

        return results, results_post_proc, results_class, results_class_post_proc

    def synapse_seg_process(
        self,
        pred: NDArray,
        filenames: Optional[List[str]] = None,
        out_dir: Optional[str] = None,
        out_dir_post_proc: Optional[str] = None,
        calculate_metrics: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Synapse segmentation workflow engine for test/inference. Process model's prediction to prepare
        synapse segmentation output and calculate metrics.

        Parameters
        ----------
        pred : 4D/5D Torch tensor
            Model predictions. E.g. ``(z, y, x, channels)`` for both 2D and 3D.

        filenames : List of str
            Predicted image's filenames.

        out_dir : path
            Output directory to save the instances.

        out_dir_post_proc : str
            Output directory to save the post-processed instances.

        calculate_metrics : bool, optional
            Whether to calculate or not the metrics.
        """
        assert pred.ndim == 4, f"Prediction doesn't have 4 dim: {pred.shape}"
        #############################
        ### INSTANCE SEGMENTATION ###
        #############################
        pred, d_result = create_synapses(
            data=pred,
            channels=self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS,
            point_creation_func=self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.POINT_CREATION_FUNCTION,
            min_th_to_be_peak=self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.MIN_TH_TO_BE_PEAK,
            min_distance=self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.PEAK_LOCAL_MAX_MIN_DISTANCE,
            min_sigma=self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.BLOB_LOG_MIN_SIGMA,
            max_sigma=self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.BLOB_LOG_MAX_SIGMA,
            num_sigma=self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.BLOB_LOG_NUM_SIGMA,
            exclude_border=self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.EXCLUDE_BORDER,
        )
        if out_dir is not None:
            save_tif(
                np.expand_dims(pred, 0),
                out_dir,
                filenames,
                verbose=self.cfg.TEST.VERBOSE,
            )

        total_pre_points = len([x for x in d_result["tag"] if x == "pre"])
        pre_points = np.array(d_result["points"][total_pre_points:])
        pre_points_df = pd.DataFrame(
            zip(
                d_result["ids"][total_pre_points:],
                list(pre_points[:, 0]),
                list(pre_points[:, 1]),
                list(pre_points[:, 2]),
                d_result["probabilities"][total_pre_points:],
            ),
            columns=[
                "pre_id",
                "axis-0",
                "axis-1",
                "axis-2",
                "probability",
            ],
        )

        # Save just the points and their probabilities
        if out_dir is not None:
            pre_points_df.to_csv(
                os.path.join(
                    out_dir,
                    "pred_pre_locations.csv",
                ),
                index=False,
            )

        total_post_points = len(d_result) - total_pre_points
        post_points = np.array(d_result["points"][:total_post_points])
        post_points_df = pd.DataFrame(
            zip(
                d_result["ids"][:total_post_points],
                list(post_points[:, 0]),
                list(post_points[:, 1]),
                list(post_points[:, 2]),
                d_result["probabilities"][:total_post_points],
            ),
            columns=[
                "post_id",
                "axis-0",
                "axis-1",
                "axis-2",
                "probability",
            ],
        )

        # Save just the points and their probabilities
        if out_dir is not None:
            post_points_df.to_csv(
                os.path.join(
                    out_dir,
                    "pred_post_locations.csv",
                ),
                index=False,
            )

        # Create coordinate arrays
        pre_points, post_points = [], []
        for coord in zip(pre_points_df["axis-0"], pre_points_df["axis-1"], pre_points_df["axis-2"]):
            pre_points.append(list(coord))
        for coord in zip(post_points_df["axis-0"], post_points_df["axis-1"], post_points_df["axis-2"]):
            post_points.append(list(coord))
        pre_points = np.array(pre_points)
        post_points = np.array(post_points)

        pre_post_mapping = {}
        pres, posts = [], []
        pre_ids = pre_points_df["pre_id"].to_list()
        if len(pre_points) > 0:
            for i in range(len(pre_points)):
                pre_post_mapping[pre_ids[i]] = []

            # Match each post with a pre
            distances = distance_matrix(post_points, pre_points)
            for i in range(len(post_points)):
                closest_pre_point = np.argmax(distances[i])
                closest_pre_point = pre_ids[closest_pre_point]
                pre_post_mapping[closest_pre_point].append(i)

            # Create pre/post lists so we can create the final dataframe
            for i in pre_post_mapping.keys():
                if len(pre_post_mapping[i]) > 0:
                    for post_site in pre_post_mapping[i]:
                        pres.append(i)
                        posts.append(post_site)
                else:
                    # For those pre points that do not have any post points assigned just put a -1 value
                    pres.append(i)
                    posts.append(-1)
        else:
            if self.cfg.TEST.VERBOSE:
                print("No pre synaptic points found!")

        # Create a mapping dataframe
        pre_post_map_df = pd.DataFrame(
            zip(
                pres,
                posts,
            ),
            columns=[
                "pre_id",
                "post_id",
            ],
        )
        if out_dir is not None:
            pre_post_map_df.to_csv(
                os.path.join(
                    out_dir,
                    "pre_post_mapping.csv",
                ),
                index=False,
            )

        if calculate_metrics and self.cfg.DATA.TEST.LOAD_GT or self.cfg.DATA.TEST.USE_VAL_AS_TEST:
            print("Calculating synapse detection stats . . .")
            locations_path = self.cfg.DATA.TEST.INPUT_ZARR_MULTIPLE_DATA_LOCATIONS_PATH
            resolution_path = self.cfg.DATA.TEST.INPUT_ZARR_MULTIPLE_DATA_RESOLUTION_PATH
            partners_path = self.cfg.DATA.TEST.INPUT_ZARR_MULTIPLE_DATA_PARTNERS_PATH
            id_path = self.cfg.DATA.TEST.INPUT_ZARR_MULTIPLE_DATA_ID_PATH
            filename = os.path.join(self.current_sample["dir"], self.current_sample["filename"])
            file, ids = read_chunked_nested_data(filename, id_path)
            ids = list(np.array(ids))
            _, partners = read_chunked_nested_data(filename, partners_path)
            partners = np.array(partners)
            _, locations = read_chunked_nested_data(filename, locations_path)
            locations = np.array(locations)
            _, resolution = read_chunked_nested_data(filename, resolution_path)
            try:
                resolution = resolution.attrs["resolution"]
            except:
                raise ValueError(
                    "There is no 'resolution' attribute in '{}'. Add it like: data['{}'].attrs['resolution'] = (8,8,8)".format(
                        resolution_path, resolution_path
                    )
                )
            gt_pre_points, gt_post_points = {}, {}
            for i in tqdm(range(len(partners)), disable=not is_main_process()):
                pre_id, post_id = partners[i]
                pre_position = ids.index(pre_id)
                post_position = ids.index(post_id)
                pre_coord = locations[pre_position] // resolution
                post_coord = locations[post_position] // resolution
                if str(pre_coord) not in gt_pre_points:
                    gt_pre_points[str(pre_coord)] = pre_coord
                if str(post_coord) not in gt_post_points:
                    gt_post_points[str(post_coord)] = post_coord
            gt_pre_points = list(gt_pre_points.values())
            gt_post_points = list(gt_post_points.values())

            if isinstance(file, h5py.File):
                file.close()

            # Calculate detection metrics
            if len(pre_points) > 0:
                d_metrics, gt_assoc, fp = detection_metrics(
                    gt_pre_points,
                    pre_points,
                    true_classes=None,
                    pred_classes=[],
                    tolerance=self.cfg.TEST.DET_TOLERANCE,
                    resolution=resolution,
                    bbox_to_consider=[],
                    verbose=True,
                )
                print("Synapse detection (pre points) metrics: {}".format(d_metrics))

                # Save csv files with the associations between GT points and predicted ones
                if out_dir:
                    gt_assoc.to_csv(
                        os.path.join(
                            out_dir,
                            "pred_pre_locations_gt_assoc.csv",
                        )
                    )
                    fp.to_csv(
                        os.path.join(
                            out_dir,
                            "pred_pre_locations_fp.csv",
                        )
                    )

                d_metrics, gt_assoc, fp = detection_metrics(
                    gt_post_points,
                    post_points,
                    true_classes=None,
                    pred_classes=[],
                    tolerance=self.cfg.TEST.DET_TOLERANCE,
                    resolution=resolution,
                    bbox_to_consider=[],
                    verbose=True,
                )
                print("Synapse detection (post points) metrics: {}".format(d_metrics))

                # Save csv files with the associations between GT points and predicted ones
                if out_dir:
                    gt_assoc.to_csv(
                        os.path.join(
                            out_dir,
                            "pred_post_locations_gt_assoc.csv",
                        )
                    )
                    fp.to_csv(
                        os.path.join(
                            out_dir,
                            "pred_post_locations_fp.csv",
                        )
                    )

        ###################
        # Post-processing #
        ###################
        if self.post_processing["instance_post"]:
            print("TODO: post-processing")
            if self.cfg.PROBLEM.NDIM == "2D":
                pred = pred[0]

            if out_dir_post_proc is not None:
                save_tif(
                    np.expand_dims(np.expand_dims(pred, -1), 0),
                    out_dir_post_proc,
                    filenames,
                    verbose=self.cfg.TEST.VERBOSE,
                )

            if calculate_metrics and self.cfg.DATA.TEST.LOAD_GT or self.cfg.DATA.TEST.USE_VAL_AS_TEST:
                print("Calculating synapse detection stats . . .")
                # Calculate detection metrics
                if len(pre_points) > 0:
                    d_metrics, gt_assoc, fp = detection_metrics(
                        gt_pre_points,
                        pre_points,
                        true_classes=None,
                        pred_classes=[],
                        tolerance=self.cfg.TEST.DET_TOLERANCE,
                        resolution=resolution,
                        bbox_to_consider=[],
                        verbose=True,
                    )
                    print("Synapse detection (pre points) metrics (post-processing): {}".format(d_metrics))

                    # Save csv files with the associations between GT points and predicted ones
                    if out_dir_post_proc:
                        gt_assoc.to_csv(
                            os.path.join(
                                out_dir_post_proc,
                                "pred_pre_locations_gt_assoc.csv",
                            )
                        )
                        fp.to_csv(
                            os.path.join(
                                out_dir_post_proc,
                                "pred_pre_locations_fp.csv",
                            )
                        )

                    d_metrics, gt_assoc, fp = detection_metrics(
                        gt_post_points,
                        post_points,
                        true_classes=None,
                        pred_classes=[],
                        tolerance=self.cfg.TEST.DET_TOLERANCE,
                        resolution=resolution,
                        bbox_to_consider=[],
                        verbose=True,
                    )
                    print("Synapse detection (post points) metrics (post-processing): {}".format(d_metrics))

                    # Save csv files with the associations between GT points and predicted ones
                    if out_dir_post_proc:
                        gt_assoc.to_csv(
                            os.path.join(
                                out_dir_post_proc,
                                "pred_post_locations_gt_assoc.csv",
                            )
                        )
                        fp.to_csv(
                            os.path.join(
                                out_dir_post_proc,
                                "pred_post_locations_fp.csv",
                            )
                        )

        return pre_points_df, post_points_df

    def process_test_sample(self):
        """
        Function to process a sample in the inference phase.
        """
        if self.cfg.MODEL.SOURCE != "torchvision":
            self.instances_already_created = False
            super().process_test_sample()
        else:
            # Skip processing image
            if "discard" in self.current_sample["X"] and self.current_sample["X"]["discard"]:
                return True

            self.instances_already_created = True

            ##################
            ### FULL IMAGE ###
            ##################
            # Make the prediction
            pred = self.model_call_func(self.current_sample["X"])
            pred = to_numpy_format(pred, self.axes_order_back)
            del self.current_sample["X"]

            # In Torchvision the output is a collection of bboxes so there is nothing else to do here
            if self.cfg.MODEL.SOURCE == "torchvision":
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

            if self.cfg.TEST.POST_PROCESSING.APPLY_MASK:
                pred = apply_binary_mask(pred, self.cfg.DATA.TEST.BINARY_MASKS)

            self.after_full_image(pred)

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
            if not self.cfg.TEST.ANALIZE_2D_IMGS_AS_3D_STACK:
                if self.cfg.PROBLEM.INSTANCE_SEG.TYPE == "regular":
                    r, r_post, rcls, rcls_post = self.instance_seg_process(
                        pred,
                        [self.current_sample["filename"]],
                        self.cfg.PATHS.RESULT_DIR.PER_IMAGE_INSTANCES,
                        self.cfg.PATHS.RESULT_DIR.PER_IMAGE_POST_PROCESSING,
                    )
                    if r:
                        self.all_matching_stats_merge_patches.append(r)
                    if r_post:
                        self.all_matching_stats_merge_patches_post.append(r_post)
                    if rcls:
                        self.all_class_stats_merge_patches.append(rcls)
                    if rcls_post:
                        self.all_class_stats_merge_patches_post.append(rcls_post)
                else:  # synapses
                    self.synapse_seg_process(
                        pred,
                        [self.current_sample["filename"]],
                        self.cfg.PATHS.RESULT_DIR.PER_IMAGE_INSTANCES,
                        self.cfg.PATHS.RESULT_DIR.PER_IMAGE_POST_PROCESSING,
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
        if self.cfg.PROBLEM.INSTANCE_SEG.TYPE == "regular":
            pass
            # Important to maintain calculate_metrics=False in the future call here
            # pre_points_df, post_points_df = self.instance_seg_process(patch, filenames, out_dir, out_dir_post_proc, calculate_metrics=False)

        else:  # synapses
            pre_points_df, post_points_df = self.synapse_seg_process(patch, calculate_metrics=False)

            if pre_points_df is not None:
                # Add the patch shift to the detected coordinates so they represent global coords
                pre_points_df["axis-0"] = pre_points_df["axis-0"] + patch_in_data.z_start
                pre_points_df["axis-1"] = pre_points_df["axis-1"] + patch_in_data.y_start
                pre_points_df["axis-2"] = pre_points_df["axis-2"] + patch_in_data.x_start

            if post_points_df is not None:
                # Add the patch shift to the detected coordinates so they represent global coords
                post_points_df["axis-0"] = post_points_df["axis-0"] + patch_in_data.z_start
                post_points_df["axis-1"] = post_points_df["axis-1"] + patch_in_data.y_start
                post_points_df["axis-2"] = post_points_df["axis-2"] + patch_in_data.x_start

            assert isinstance(self.all_pred, list)
            self.all_pred.append([pre_points_df, post_points_df])

    def after_all_patch_prediction_by_chunks(self):
        """
        Place any code that needs to be done after predicting all the patches, one by one, in the "by chunks" setting.
        """
        assert isinstance(self.all_pred, list) and isinstance(self.all_gt, list)
        if self.cfg.PROBLEM.INSTANCE_SEG.TYPE == "regular":
            if self.cfg.TEST.BY_CHUNKS.WORKFLOW_PROCESS.TYPE == "chunk_by_chunk":
                raise NotImplementedError
            else:
                # Load H5/Zarr and convert it into numpy array
                fpath = os.path.join(
                    self.cfg.PATHS.RESULT_DIR.PER_IMAGE, os.path.splitext(self.current_sample["filename"])[0] + ".zarr"
                )
                pred_file, pred = read_chunked_data(fpath)
                pred = np.squeeze(np.array(pred, dtype=self.dtype))
                if isinstance(pred_file, h5py.File):
                    pred_file.close()

                pred = ensure_3d_shape(pred, fpath)

                self.after_merge_patches(np.expand_dims(pred, 0))
        else:
            # In this case we need only to merge all local points so it will be done by the main thread. The rest will wait
            if not self.cfg.TEST.REUSE_PREDICTIONS:
                # For synapses we need to map the pre to the post points. It needs to be done here and not patch by patch as
                # some pre points may lay in other chunks of the data.
                all_pre_dfs, all_post_dfs = [], []
                pre_id_counter, post_id_counter = 0, 0
                for pre_df, post_dt in self.all_pred:
                    if len(pre_df["pre_id"]) > 0:
                        pre_df["pre_id"] = pre_df["pre_id"] + pre_id_counter
                        pre_id_counter += len(pre_df["pre_id"])
                        all_pre_dfs.append(pre_df)
                    if len(post_dt["post_id"]) > 0:
                        post_dt["post_id"] = post_dt["post_id"] + post_id_counter
                        post_id_counter += len(post_dt["post_id"])
                        all_post_dfs.append(post_dt)

                pre_points_df = pd.concat(all_pre_dfs, ignore_index=True)
                post_points_df = pd.concat(all_post_dfs, ignore_index=True)

                # Save then the pre and post sites separately
                os.makedirs(self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK, exist_ok=True)
                pre_points_df.to_csv(
                    os.path.join(
                        self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK,
                        "pred_pre_locations.csv",
                    ),
                    index=False,
                )
                post_points_df.to_csv(
                    os.path.join(
                        self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK,
                        "pred_post_locations.csv",
                    ),
                    index=False,
                )

                # Create coordinate arrays
                pre_points, post_points = [], []
                for coord in zip(pre_points_df["axis-0"], pre_points_df["axis-1"], pre_points_df["axis-2"]):
                    pre_points.append(list(coord))
                for coord in zip(post_points_df["axis-0"], post_points_df["axis-1"], post_points_df["axis-2"]):
                    post_points.append(list(coord))
                pre_points = np.array(pre_points)
                post_points = np.array(post_points)

                pre_post_mapping = {}
                pres, posts = [], []
                pre_ids = pre_points_df["pre_id"].to_list()
                if len(pre_points) > 0:
                    for i in range(len(pre_points)):
                        pre_post_mapping[pre_ids[i]] = []

                    # Match each post with a pre
                    distances = distance_matrix(post_points, pre_points)
                    for i in range(len(post_points)):
                        closest_pre_point = np.argmax(distances[i])
                        closest_pre_point = pre_ids[closest_pre_point]
                        pre_post_mapping[closest_pre_point].append(i)

                    # Create pre/post lists so we can create the final dataframe
                    for i in pre_post_mapping.keys():
                        if len(pre_post_mapping[i]) > 0:
                            for post_site in pre_post_mapping[i]:
                                pres.append(i)
                                posts.append(post_site)
                        else:
                            # For those pre points that do not have any post points assigned just put a -1 value
                            pres.append(i)
                            posts.append(-1)
                else:
                    if self.cfg.TEST.VERBOSE:
                        print("No pre synaptic points found!")

                # Create a mapping dataframe
                pre_post_map_df = pd.DataFrame(
                    zip(
                        pres,
                        posts,
                    ),
                    columns=[
                        "pre_id",
                        "post_id",
                    ],
                )

                pre_post_map_df.to_csv(
                    os.path.join(
                        self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK,
                        "pre_post_mapping.csv",
                    ),
                    index=False,
                )
            else:
                # Read the dataframes
                pre_points_df = pd.read_csv(
                    os.path.join(
                        self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK,
                        "pred_pre_locations.csv",
                    )
                )
                post_points_df = pd.read_csv(
                    os.path.join(
                        self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK,
                        "pred_post_locations.csv",
                    )
                )
                pre_post_map_df = pd.read_csv(
                    os.path.join(
                        self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK,
                        "pre_post_mapping.csv",
                    )
                )

                # Create coordinate arrays
                pre_points, post_points = [], []
                for coord in zip(pre_points_df["axis-0"], pre_points_df["axis-1"], pre_points_df["axis-2"]):
                    pre_points.append(list(coord))
                for coord in zip(post_points_df["axis-0"], post_points_df["axis-1"], post_points_df["axis-2"]):
                    post_points.append(list(coord))
                pre_points = np.array(pre_points)
                post_points = np.array(post_points)

            if self.cfg.DATA.TEST.LOAD_GT or self.cfg.DATA.TEST.USE_VAL_AS_TEST:
                print("Calculating synapse detection stats . . .")
                locations_path = self.cfg.DATA.TEST.INPUT_ZARR_MULTIPLE_DATA_LOCATIONS_PATH
                resolution_path = self.cfg.DATA.TEST.INPUT_ZARR_MULTIPLE_DATA_RESOLUTION_PATH
                partners_path = self.cfg.DATA.TEST.INPUT_ZARR_MULTIPLE_DATA_PARTNERS_PATH
                id_path = self.cfg.DATA.TEST.INPUT_ZARR_MULTIPLE_DATA_ID_PATH
                filename = os.path.join(self.current_sample["dir"], self.current_sample["filename"])
                file, ids = read_chunked_nested_data(filename, id_path)
                ids = list(np.array(ids))
                _, partners = read_chunked_nested_data(filename, partners_path)
                partners = np.array(partners)
                _, locations = read_chunked_nested_data(filename, locations_path)
                locations = np.array(locations)
                _, resolution = read_chunked_nested_data(filename, resolution_path)
                try:
                    resolution = resolution.attrs["resolution"]
                except:
                    raise ValueError(
                        "There is no 'resolution' attribute in '{}'. Add it like: data['{}'].attrs['resolution'] = (8,8,8)".format(
                            resolution_path, resolution_path
                        )
                    )
                gt_pre_points, gt_post_points = {}, {}
                for i in tqdm(range(len(partners)), disable=not is_main_process()):
                    pre_id, post_id = partners[i]
                    pre_position = ids.index(pre_id)
                    post_position = ids.index(post_id)
                    pre_coord = locations[pre_position] // resolution
                    post_coord = locations[post_position] // resolution
                    if str(pre_coord) not in gt_pre_points:
                        gt_pre_points[str(pre_coord)] = pre_coord
                    if str(post_coord) not in gt_post_points:
                        gt_post_points[str(post_coord)] = post_coord
                gt_pre_points = list(gt_pre_points.values())
                gt_post_points = list(gt_post_points.values())

                if isinstance(file, h5py.File):
                    file.close()

                # Calculate detection metrics
                if len(pre_points) > 0:
                    d_metrics, gt_assoc, fp = detection_metrics(
                        gt_pre_points,
                        pre_points,
                        true_classes=None,
                        pred_classes=[],
                        tolerance=self.cfg.TEST.DET_TOLERANCE,
                        resolution=resolution,
                        bbox_to_consider=[],
                        verbose=True,
                    )
                    print("Synapse detection (pre points) metrics: {}".format(d_metrics))

                    # Save csv files with the associations between GT points and predicted ones
                    gt_assoc.to_csv(
                        os.path.join(
                            self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK,
                            "pred_pre_locations_gt_assoc.csv",
                        ),
                        index=False,
                    )
                    fp.to_csv(
                        os.path.join(
                            self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK,
                            "pred_pre_locations_fp.csv",
                        ),
                        index=False,
                    )

                    d_metrics, gt_assoc, fp = detection_metrics(
                        gt_post_points,
                        post_points,
                        true_classes=None,
                        pred_classes=[],
                        tolerance=self.cfg.TEST.DET_TOLERANCE,
                        resolution=resolution,
                        bbox_to_consider=[],
                        verbose=True,
                    )
                    print("Synapse detection (post points) metrics: {}".format(d_metrics))

                    # Save csv files with the associations between GT points and predicted ones
                    gt_assoc.to_csv(
                        os.path.join(
                            self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK,
                            "pred_post_locations_gt_assoc.csv",
                        ),
                        index=False,
                    )
                    fp.to_csv(
                        os.path.join(
                            self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK,
                            "pred_post_locations_fp.csv",
                        ),
                        index=False,
                    )

            # Remove close points
            post_proc = False
            if self.cfg.TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS:
                post_proc = True
                pre_points, pre_dropped_pos = remove_close_points(  # type: ignore
                    pre_points,
                    self.cfg.TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS_RADIUS,
                    self.resolution,
                    ndim=self.dims,
                    return_drops=True,
                )
                pre_points_df.drop(pre_points_df.index[pre_dropped_pos], inplace=True)  # type: ignore
                post_points, post_dropped_pos = remove_close_points(  # type: ignore
                    post_points,
                    self.cfg.TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS_RADIUS,
                    self.resolution,
                    ndim=self.dims,
                    return_drops=True,
                )
                post_points_df.drop(post_points_df.index[post_dropped_pos], inplace=True)  # type: ignore

                # Save filtered stats
                os.makedirs(self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK_POST_PROCESSING, exist_ok=True)
                pre_points_df.to_csv(
                    os.path.join(
                        self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK_POST_PROCESSING,
                        "pred_pre_locations.csv",
                    ),
                    index=False,
                )
                post_points_df.to_csv(
                    os.path.join(
                        self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK_POST_PROCESSING,
                        "pred_post_locations.csv",
                    ),
                    index=False,
                )

                pre_post_mapping = {}
                pres, posts = [], []
                pre_ids = pre_points_df["pre_id"].to_list()
                if len(pre_points) > 0:
                    for i in range(len(pre_points)):
                        pre_post_mapping[pre_ids[i]] = []

                    # Match each post with a pre
                    distances = distance_matrix(post_points, pre_points)
                    for i in range(len(post_points)):
                        closest_pre_point = np.argmax(distances[i])
                        closest_pre_point = pre_ids[closest_pre_point]
                        pre_post_mapping[closest_pre_point].append(i)

                    # Create pre/post lists so we can create the final dataframe
                    for i in pre_post_mapping.keys():
                        if len(pre_post_mapping[i]) > 0:
                            for post_site in pre_post_mapping[i]:
                                pres.append(i)
                                posts.append(post_site)
                        else:
                            # For those pre points that do not have any post points assigned just put a -1 value
                            pres.append(i)
                            posts.append(-1)
                else:
                    if self.cfg.TEST.VERBOSE:
                        print("No pre synaptic points found!")

                # Create a mapping dataframe
                pre_post_map_df = pd.DataFrame(
                    zip(
                        pres,
                        posts,
                    ),
                    columns=[
                        "pre_id",
                        "post_id",
                    ],
                )
                pre_post_map_df.to_csv(
                    os.path.join(
                        self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK_POST_PROCESSING,
                        "pre_post_mapping.csv",
                    ),
                    index=False,
                )

                if self.cfg.DATA.TEST.LOAD_GT or self.cfg.DATA.TEST.USE_VAL_AS_TEST:
                    print("Calculating synapse detection stats after post-processing . . .")
                    # Calculate detection metrics
                    if len(pre_points) > 0:
                        d_metrics, gt_assoc, fp = detection_metrics(
                            gt_pre_points,
                            pre_points,
                            true_classes=None,
                            pred_classes=[],
                            tolerance=self.cfg.TEST.DET_TOLERANCE,
                            resolution=resolution,
                            bbox_to_consider=[],
                            verbose=True,
                        )
                        print("Synapse detection (pre points) metrics (post-processing): {}".format(d_metrics))

                        # Save csv files with the associations between GT points and predicted ones
                        gt_assoc.to_csv(
                            os.path.join(
                                self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK_POST_PROCESSING,
                                "pred_pre_locations_gt_assoc.csv",
                            ),
                            index=False,
                        )
                        fp.to_csv(
                            os.path.join(
                                self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK_POST_PROCESSING,
                                "pred_pre_locations_fp.csv",
                            ),
                            index=False,
                        )

                        d_metrics, gt_assoc, fp = detection_metrics(
                            gt_post_points,
                            post_points,
                            true_classes=None,
                            pred_classes=[],
                            tolerance=self.cfg.TEST.DET_TOLERANCE,
                            resolution=resolution,
                            bbox_to_consider=[],
                            verbose=True,
                        )
                        print("Synapse detection (post points) metrics (post-processing): {}".format(d_metrics))

                        # Save csv files with the associations between GT points and predicted ones
                        gt_assoc.to_csv(
                            os.path.join(
                                self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK_POST_PROCESSING,
                                "pred_post_locations_gt_assoc.csv",
                            ),
                            index=False,
                        )
                        fp.to_csv(
                            os.path.join(
                                self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK_POST_PROCESSING,
                                "pred_post_locations_fp.csv",
                            ),
                            index=False,
                        )

            sshape = list(self.current_sample["X"].shape)
            assert len(sshape) >= 3
            if len(sshape) == 3:
                sshape += [
                    2,
                ]
            else:
                sshape[-1] = 2
            mask = np.zeros(sshape, dtype=np.uint8)

            # Paint pre points
            pre_ids = pre_points_df["pre_id"].to_list()
            assert len(pre_points) == len(pre_ids)
            for j, cor in enumerate(pre_points):
                z, y, x = cor  # type: ignore
                z, y, x = int(z), int(y), int(x)
                mask[z, y, x, 0] = pre_ids[j]

            # Paint post points
            post_ids = post_points_df["post_id"].to_list()
            for j, cor in enumerate(post_points):
                z, y, x = cor  # type: ignore
                z, y, x = int(z), int(y), int(x)
                mask[z, y, x, 1] = post_ids[j]

            # Dilate and save the predicted points for the current class
            for i in range(mask.shape[0]):
                for c in range(mask.shape[-1]):
                    mask[i, ..., c] = dilation(mask[i, ..., c], disk(3))

            out_dir = (
                self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK_POST_PROCESSING
                if post_proc
                else self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK
            )
            save_tif(
                np.expand_dims(mask, 0),
                out_dir,
                [os.path.splitext(self.current_sample["filename"])[0] + "_points.tif"],
                verbose=self.cfg.TEST.VERBOSE,
            )

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
            if not self.cfg.TEST.ANALIZE_2D_IMGS_AS_3D_STACK:
                if self.cfg.PROBLEM.INSTANCE_SEG.TYPE == "regular":
                    r, r_post, rcls, rcls_post = self.instance_seg_process(
                        pred,
                        [self.current_sample["filename"]],
                        self.cfg.PATHS.RESULT_DIR.FULL_IMAGE_INSTANCES,
                        self.cfg.PATHS.RESULT_DIR.FULL_IMAGE_POST_PROCESSING,
                    )
                    if r:
                        self.all_matching_stats.append(r)
                    if r_post:
                        self.all_matching_stats_post.append(r_post)
                    if rcls:
                        self.all_class_stats.append(rcls)
                    if rcls_post:
                        self.all_class_stats_post.append(rcls_post)
                else:  # synapses
                    self.synapse_seg_process(
                        pred,
                        [self.current_sample["filename"]],
                        self.cfg.PATHS.RESULT_DIR.FULL_IMAGE_INSTANCES,
                        self.cfg.PATHS.RESULT_DIR.FULL_IMAGE_POST_PROCESSING,
                    )
        else:
            raise NotImplementedError

    def after_all_images(self):
        """
        Steps that must be done after predicting all images.
        """
        super().after_all_images()
        assert isinstance(self.all_pred, list) and isinstance(self.all_gt, list)
        if self.cfg.TEST.ANALIZE_2D_IMGS_AS_3D_STACK:
            print("Analysing all images as a 3D stack . . .")
            self.all_pred = np.concatenate(self.all_pred)
            if self.cfg.PROBLEM.INSTANCE_SEG.TYPE == "regular":
                r, r_post, rcls, rcls_post = self.instance_seg_process(
                    self.all_pred,
                    ["3D_stack_instances.tif"],
                    self.cfg.PATHS.RESULT_DIR.AS_3D_STACK,
                    self.cfg.PATHS.RESULT_DIR.AS_3D_STACK_POST_PROCESSING,
                )
                if r:
                    self.all_matching_stats_as_3D_stack.append(r)
                if r_post:
                    self.all_matching_stats_as_3D_stack_post.append(r_post)
                if rcls:
                    self.all_class_stats_as_3D_stack.append(rcls)
                if rcls_post:
                    self.all_class_stats_as_3D_stack_post.append(rcls_post)
            else:  # synapses
                self.synapse_seg_process(
                    self.all_pred,
                    ["3D_stack_instances.tif"],
                    self.cfg.PATHS.RESULT_DIR.AS_3D_STACK,
                    self.cfg.PATHS.RESULT_DIR.AS_3D_STACK_POST_PROCESSING,
                )

    def normalize_stats(self, image_counter):
        """
        Normalize statistics.

        Parameters
        ----------
        image_counter : int
            Number of images to average the metrics.
        """
        super().normalize_stats(image_counter)

        if self.cfg.DATA.TEST.LOAD_GT or self.cfg.DATA.TEST.USE_VAL_AS_TEST:
            if self.cfg.TEST.MATCHING_STATS:
                # Merge patches
                if len(self.all_matching_stats_merge_patches) > 0:
                    self.stats["inst_stats_merge_patches"] = wrapper_matching_dataset_lazy(
                        self.all_matching_stats_merge_patches,
                        self.cfg.TEST.MATCHING_STATS_THS,
                    )
                # As 3D stack
                if len(self.all_matching_stats_as_3D_stack) > 0:
                    self.stats["inst_stats_as_3D_stack"] = wrapper_matching_dataset_lazy(
                        self.all_matching_stats_as_3D_stack,
                        self.cfg.TEST.MATCHING_STATS_THS,
                    )
                # Full image
                if len(self.all_matching_stats) > 0:
                    self.stats["inst_stats"] = wrapper_matching_dataset_lazy(
                        self.all_matching_stats, self.cfg.TEST.MATCHING_STATS_THS
                    )
                if self.post_processing["instance_post"]:
                    # Merge patches
                    if len(self.all_matching_stats_merge_patches_post) > 0:
                        self.stats["inst_stats_merge_patches_post"] = wrapper_matching_dataset_lazy(
                            self.all_matching_stats_merge_patches_post,
                            self.cfg.TEST.MATCHING_STATS_THS,
                        )
                    # As 3D stack
                    if len(self.all_matching_stats_as_3D_stack_post) > 0:
                        self.stats["inst_stats_as_3D_stack_post"] = wrapper_matching_dataset_lazy(
                            self.all_matching_stats_as_3D_stack_post,
                            self.cfg.TEST.MATCHING_STATS_THS,
                        )
                    # Full image
                    if len(self.all_matching_stats_post) > 0:
                        self.stats["inst_stats_post"] = wrapper_matching_dataset_lazy(
                            self.all_matching_stats_post,
                            self.cfg.TEST.MATCHING_STATS_THS,
                        )

                # Multi-head: instances + classification
                if self.multihead:
                    # Merge patches
                    if len(self.all_class_stats_merge_patches) > 0:
                        self.stats["class_stats_merge_patches"] = np.mean(self.all_class_stats_merge_patches)
                    # As 3D stack
                    if len(self.all_class_stats_as_3D_stack) > 0:
                        self.stats["class_stats_as_3D_stack"] = np.mean(self.all_class_stats_as_3D_stack)
                    # Full image
                    if len(self.all_class_stats) > 0:
                        self.stats["class_stats"] = np.mean(self.all_class_stats)

                    if self.post_processing["instance_post"]:
                        # Merge patches
                        if len(self.all_class_stats_merge_patches_post) > 0:
                            self.stats["class_stats_merge_patches_post"] = np.mean(
                                self.all_class_stats_merge_patches_post
                            )
                        # As 3D stack
                        if len(self.all_class_stats_as_3D_stack_post) > 0:
                            self.stats["class_stats_as_3D_stack_post"] = np.mean(self.all_class_stats_as_3D_stack_post)
                        # Full image
                        if len(self.all_class_stats_post) > 0:
                            self.stats["class_stats_post"] = np.mean(self.all_class_stats_post)

    def print_stats(self, image_counter):
        """
        Print statistics.

        Parameters
        ----------
        image_counter : int
            Number of images to call ``normalize_stats``.
        """
        if self.cfg.MODEL.SOURCE != "torchvision":
            super().print_stats(image_counter)

            print("Instance segmentation specific metrics:")
            if self.cfg.TEST.MATCHING_STATS and (self.cfg.DATA.TEST.LOAD_GT or self.cfg.DATA.TEST.USE_VAL_AS_TEST):
                for i in range(len(self.cfg.TEST.MATCHING_STATS_THS)):
                    print("IoU TH={}".format(self.cfg.TEST.MATCHING_STATS_THS[i]))
                    # Merge patches
                    if self.stats["inst_stats_merge_patches"]:
                        print("      Merge patches:")
                        print(f"      {self.stats['inst_stats_merge_patches'][i]}")
                    # As 3D stack
                    if self.stats["inst_stats_as_3D_stack"]:
                        print("      As 3D stack:")
                        print(f"      {self.stats['inst_stats_as_3D_stack'][i]}")
                    # Full image
                    if self.stats["inst_stats"]:
                        print("      Full image:")
                        print(f"      {self.stats['inst_stats'][i]}")
                    if self.post_processing["instance_post"]:
                        print("IoU (post-processing) TH={}".format(self.cfg.TEST.MATCHING_STATS_THS[i]))
                        # Merge patches
                        if self.stats["inst_stats_merge_patches_post"]:
                            print("      Merge patches (post-processing):")
                            print(f"      {self.stats['inst_stats_merge_patches_post'][i]}")
                        # As 3D stack
                        if self.stats["inst_stats_as_3D_stack_post"]:
                            print("      As 3D stack (post-processing):")
                            print(f"      {self.stats['inst_stats_as_3D_stack_post'][i]}")
                        # Full image
                        if self.stats["inst_stats_post"]:
                            print("      Full image (post-processing):")
                            print(f"      {self.stats['inst_stats_post'][i]}")

                # Multi-head: instances + classification
                if self.multihead:
                    # Merge patches
                    if self.stats["class_stats_merge_patches"]:
                        print(f"      Merge patches classification IoU: {self.stats['class_stats_merge_patches']}")
                    # As 3D stack
                    if self.stats["class_stats_as_3D_stack"]:
                        print(f"      As 3D stack classification IoU: {self.stats['class_stats_as_3D_stack']}")
                    # Full image
                    if self.stats["class_stats"]:
                        print(f"      Full image classification IoU: {self.stats['class_stats']}")

                    if self.post_processing["instance_post"]:
                        # Merge patches
                        if self.stats["class_stats_merge_patches_post"]:
                            print(
                                f"      Merge patches classification IoU (post-processing): {self.stats['class_stats_merge_patches_post']}"
                            )
                        # As 3D stack
                        if self.stats["class_stats_as_3D_stack_post"]:
                            print(
                                f"      As 3D stack classification IoU (post-processing): {self.stats['class_stats_as_3D_stack_post']}"
                            )
                        # Full image
                        if self.stats["class_stats_post"]:
                            print(
                                f"      Full image classification IoU (post-processing): {self.stats['class_stats_post']}"
                            )

    def prepare_instance_data(self):
        """
        Creates instance segmentation ground truth images to train the model based on the ground truth instances provided.
        They will be saved in a separate folder in the root path of the ground truth.
        """
        original_test_path, original_test_mask_path = None, None
        train_channel_mask_dir = self.cfg.DATA.TRAIN.INSTANCE_CHANNELS_MASK_DIR
        val_channel_mask_dir = self.cfg.DATA.VAL.INSTANCE_CHANNELS_MASK_DIR

        if not self.cfg.DATA.TEST.INPUT_ZARR_MULTIPLE_DATA:
            test_instance_mask_dir = self.cfg.DATA.TEST.GT_PATH
            test_channel_mask_dir = self.cfg.DATA.TEST.INSTANCE_CHANNELS_MASK_DIR
        else:
            test_instance_mask_dir = self.cfg.DATA.TEST.PATH
            if self.cfg.PROBLEM.INSTANCE_SEG.TYPE == "synapses":
                test_channel_mask_dir = self.cfg.DATA.TEST.INSTANCE_CHANNELS_MASK_DIR
            else:
                test_channel_mask_dir = self.cfg.DATA.TEST.PATH

        opts = []
        print("###########################")
        print("#  PREPARE INSTANCE DATA  #")
        print("###########################")

        # Create selected channels for train data
        if self.cfg.TRAIN.ENABLE or self.cfg.DATA.TEST.USE_VAL_AS_TEST:
            if not os.path.isdir(train_channel_mask_dir) and is_main_process():
                print(
                    "You select to create {} channels from given instance labels and no file is detected in {}. "
                    "So let's prepare the data. This process will be done just once!".format(
                        self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS, train_channel_mask_dir
                    )
                )
                create_instance_channels(self.cfg)

            # Change the value of DATA.TRAIN.INPUT_MASK_AXES_ORDER as we have created the instance mask and maybe the user doesn't
            # know the data order that is created.
            if self.cfg.DATA.TRAIN.INPUT_ZARR_MULTIPLE_DATA:
                out_data_order = self.cfg.DATA.TRAIN.INPUT_IMG_AXES_ORDER
                if "C" not in self.cfg.DATA.TRAIN.INPUT_IMG_AXES_ORDER:
                    out_data_order += "C"
                print(
                    "DATA.TRAIN.INPUT_MASK_AXES_ORDER changed from {} to {}".format(
                        self.cfg.DATA.TRAIN.INPUT_MASK_AXES_ORDER, out_data_order
                    )
                )
                opts.extend([f"DATA.TRAIN.INPUT_MASK_AXES_ORDER", out_data_order])

        # Create selected channels for val data
        if self.cfg.TRAIN.ENABLE and not self.cfg.DATA.VAL.FROM_TRAIN:
            if not os.path.isdir(val_channel_mask_dir) and is_main_process():
                print(
                    "You select to create {} channels from given instance labels and no file is detected in {}. "
                    "So let's prepare the data. This process will be done just once!".format(
                        self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS, val_channel_mask_dir
                    )
                )
                create_instance_channels(self.cfg, data_type="val")

            # Change the value of DATA.VAL.INPUT_MASK_AXES_ORDER as we have created the instance mask and maybe the user doesn't
            # know the data order that is created.
            if self.cfg.DATA.VAL.INPUT_ZARR_MULTIPLE_DATA:
                out_data_order = self.cfg.DATA.VAL.INPUT_IMG_AXES_ORDER
                if "C" not in self.cfg.DATA.VAL.INPUT_IMG_AXES_ORDER:
                    out_data_order += "C"
                print(
                    "DATA.VAL.INPUT_MASK_AXES_ORDER changed from {} to {}".format(
                        self.cfg.DATA.VAL.INPUT_MASK_AXES_ORDER, out_data_order
                    )
                )
                opts.extend([f"DATA.VAL.INPUT_MASK_AXES_ORDER", out_data_order])

        # Create selected channels for test data once
        if self.cfg.TEST.ENABLE and not self.cfg.DATA.TEST.USE_VAL_AS_TEST and self.cfg.DATA.TEST.LOAD_GT:
            if not os.path.isdir(test_channel_mask_dir) and is_main_process():
                print(
                    "You select to create {} channels from given instance labels and no file is detected in {}. "
                    "So let's prepare the data. This process will be done just once!".format(
                        self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS,
                        test_channel_mask_dir,
                    )
                )
                create_instance_channels(self.cfg, data_type="test")

            # Change the value of DATA.TEST.INPUT_MASK_AXES_ORDER as we have created the instance mask and maybe the user doesn't
            # know the data order that is created.
            if self.cfg.DATA.TEST.INPUT_ZARR_MULTIPLE_DATA:
                out_data_order = self.cfg.DATA.TEST.INPUT_IMG_AXES_ORDER
                if "C" not in self.cfg.DATA.TEST.INPUT_IMG_AXES_ORDER:
                    out_data_order += "C"
                print(
                    "DATA.TEST.INPUT_MASK_AXES_ORDER changed from {} to {}".format(
                        self.cfg.DATA.TEST.INPUT_MASK_AXES_ORDER, out_data_order
                    )
                )
                opts.extend([f"DATA.TEST.INPUT_MASK_AXES_ORDER", out_data_order])

        if is_dist_avail_and_initialized():
            dist.barrier()

        if self.cfg.TRAIN.ENABLE:
            if self.cfg.DATA.TRAIN.GT_PATH != train_channel_mask_dir:
                print(
                    "DATA.TRAIN.GT_PATH changed from {} to {}".format(
                        self.cfg.DATA.TRAIN.GT_PATH, train_channel_mask_dir
                    )
                )
            opts.extend(
                [
                    "DATA.TRAIN.GT_PATH",
                    train_channel_mask_dir,
                ]
            )
        if not self.cfg.DATA.VAL.FROM_TRAIN:
            if self.cfg.DATA.VAL.GT_PATH != val_channel_mask_dir:
                print("DATA.VAL.GT_PATH changed from {} to {}".format(self.cfg.DATA.VAL.GT_PATH, val_channel_mask_dir))
            opts.extend(
                [
                    "DATA.VAL.GT_PATH",
                    val_channel_mask_dir,
                ]
            )
        if self.cfg.TEST.ENABLE and not self.cfg.DATA.TEST.USE_VAL_AS_TEST:
            if self.cfg.DATA.TEST.LOAD_GT:
                if self.cfg.DATA.TEST.GT_PATH != test_channel_mask_dir:
                    print(
                        "DATA.TEST.GT_PATH changed from {} to {}".format(
                            self.cfg.DATA.TEST.GT_PATH, test_channel_mask_dir
                        )
                    )
                    opts.extend(["DATA.TEST.GT_PATH", test_channel_mask_dir])
        original_test_path = self.cfg.DATA.TEST.PATH
        original_test_mask_path = test_instance_mask_dir
        self.cfg.merge_from_list(opts)

        return original_test_path, original_test_mask_path

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
        assert self.torchvision_preprocessing and self.model
        filename, file_extension = os.path.splitext(self.current_sample["filename"])

        # Convert first to 0-255 range if uint16
        if in_img.dtype == torch.float32:
            if torch.max(in_img) > 1:
                in_img = (self.torchvision_norm.apply_image_norm(in_img)[0] * 255).to(torch.uint8)  # type: ignore
            in_img = in_img.to(torch.uint8)

        # Apply TorchVision pre-processing
        in_img = self.torchvision_preprocessing(in_img)
        pred = self.model(in_img)
        masks = pred[0]["masks"].cpu().numpy().transpose(0, 2, 3, 1)
        if masks.shape[0] != 0:
            masks = np.argmax(pred[0]["masks"].cpu().numpy().transpose(0, 2, 3, 1), axis=0)
        else:
            masks = torch.ones(
                (1,) + pred[0]["masks"].cpu().numpy().transpose(0, 2, 3, 1).shape[1:],
                dtype=torch.uint8,
            )

        if not is_train and masks.shape[0] != 0:
            # Extract each output from MaskRCNN
            bboxes = pred[0]["boxes"].cpu().numpy().astype(np.uint16)
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

            # Save masks
            save_tif(
                np.expand_dims(masks, 0),
                self.cfg.PATHS.RESULT_DIR.FULL_IMAGE,
                [self.current_sample["filename"]],
                verbose=self.cfg.TEST.VERBOSE,
            )

        return None
