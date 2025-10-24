"""
Instance segmentation workflow for BiaPy.

This module defines the Instance_Segmentation_Workflow class, which implements the
training, validation, and inference pipeline for instance segmentation tasks in BiaPy.
It handles data preparation, model setup, metrics, predictions, post-processing,
and result saving for assigning unique IDs to each object in 2D and 3D images.
"""
import os
import torch
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.segmentation import clear_border
from skimage.transform import resize
from skimage.morphology import ball, dilation
import torch.distributed as dist
from typing import Dict, Optional, List, Tuple
from numpy.typing import NDArray
from scipy.spatial import distance_matrix
from skimage.filters import threshold_otsu


from biapy.data.post_processing.post_processing import (
    watershed_by_channels,
    voronoi_on_mask,
    measure_morphological_props_and_filter,
    repare_large_blobs,
    apply_binary_mask,
    create_synapses,
    remove_close_points,
    remove_close_points_by_mask,
    fill_label_holes,
    embedseg_instances,
)
from biapy.data.post_processing.polygon_nms_postprocessing import stardist_instances_from_prediction
from biapy.data.pre_processing import create_instance_channels
from biapy.utils.matching import matching, wrapper_matching_dataset_lazy
from biapy.engine.metrics import (
    jaccard_index,
    instance_segmentation_loss,
    multiple_metrics,
    detection_metrics,
    ContrastCELoss,
    SpatialEmbLoss,
)
from biapy.engine.base_workflow import Base_Workflow
from biapy.utils.misc import (
    is_main_process,
    is_dist_avail_and_initialized,
    to_pytorch_format,
    to_numpy_format,
    MetricLogger,
    os_walk_clean,
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
        """
        Initialize the Instance_Segmentation_Workflow.

        Sets up configuration, device, job identifier, and initializes
        workflow-specific attributes for instance segmentation tasks.

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

        # From now on, no modification of the cfg will be allowed
        self.cfg.freeze()

        # Workflow specific training variables
        self.mask_path = cfg.DATA.TRAIN.GT_PATH
        self.is_y_mask = True
        self.load_Y_val = True
        if self.cfg.TEST.ENABLE and self.cfg.DATA.TEST.LOAD_GT:
            self.test_gt_filenames = sorted(next(os_walk_clean(self.original_test_mask_path))[2])
            if len(self.test_gt_filenames) == 0:
                self.test_gt_filenames = sorted(next(os_walk_clean(self.original_test_mask_path))[1])

        # Specific instance segmentation post-processing
        if (
            self.cfg.TEST.POST_PROCESSING.VORONOI_ON_MASK
            or self.cfg.TEST.POST_PROCESSING.CLEAR_BORDER
            or self.cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.ENABLE
            or self.cfg.TEST.POST_PROCESSING.REPARE_LARGE_BLOBS_SIZE != -1
            or self.cfg.TEST.POST_PROCESSING.FILL_HOLES
        ):
            self.post_processing["instance_post"] = True
        else:
            self.post_processing["instance_post"] = False
        
        if self.cfg.PROBLEM.INSTANCE_SEG.TYPE == "synapses":
            if (
                self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.REMOVE_CLOSE_PRE_POINTS_RADIUS > 0 
                or self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.REMOVE_CLOSE_POST_POINTS_RADIUS > 0
            ):
                self.post_processing['per_image'] = True

        self.instances_already_created = False

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
        self.activations = []
        self.model_output_channels = {"type": "mask", "channels": 0}
        dst = self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS_EXTRA_OPTS[0]
        for i, channel in enumerate(self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS):
                if channel in ["B", "F", "P", "C", "T", "M", "F_pre", "F_post"]:
                    self.activations.append("ce_sigmoid")
                    self.model_output_channels["channels"] += 1
                elif channel in ["Db", "Dc", "Dn", "D", "H", "V", "Z"]:
                    if dst.get(channel, {}).get("norm", True) and self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS_LOSSES[i] not in ["mse", "l1", "mae"]:
                        self.activations.append("ce_sigmoid")
                    else:
                        self.activations.append("linear")
                    self.model_output_channels["channels"] += 1
                elif channel == "D":
                    self.activations.append(dst.get("D", {}).get("act", "linear"))
                    self.model_output_channels["channels"] += 1
                elif channel == "A":
                    for _ in range(len(dst.get("A", {}).get("y_affinities", [1]))):
                        self.model_output_channels["channels"] += 1
                        self.activations.append("ce_sigmoid")
                elif channel == "R":
                    for _ in range(dst.get("R", {}).get("nrays", 32 if self.dims == 2 else 96)):
                        self.activations.append("linear")
                        self.model_output_channels["channels"] += 1
                elif channel == "E_offset":
                    for _ in range(self.dims):
                        self.activations.append("ce_sigmoid")
                        self.model_output_channels["channels"] += 1
                elif channel == "E_sigma":
                    for _ in range(self.dims):
                        self.activations.append("ce_sigmoid")
                        self.model_output_channels["channels"] += 1
                elif channel == "E_seediness":
                    self.activations.append("ce_sigmoid")
                    self.model_output_channels["channels"] += 1
                elif channel == "We":
                    continue
                else:
                    raise ValueError("Unknown channel: {}".format(channel))

        # Multi-head: instances + classification
        if self.cfg.DATA.N_CLASSES > 2:
            self.activations = [self.activations, ["linear",]*self.cfg.DATA.N_CLASSES]
            self.model_output_channels["channels"] = [self.model_output_channels["channels"], self.cfg.DATA.N_CLASSES]
            self.multihead = True
        else:
            self.activations = [self.activations]
            self.model_output_channels["channels"] = [self.model_output_channels["channels"]]
            self.multihead = False

        self.real_classes = self.model_output_channels["channels"][0]

        super().define_activations_and_channels()

        self.stardist_grid = (1,1)

    def define_metrics(self):
        """
        Define the metrics to be used in the instance segmentation workflow.

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
        
        for channel in self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS:
            if channel in ["B", "F", "P", "C", "T", "A", "M"]:
                m = "IoU ({} channel)".format(channel) if channel != "A" else "IoU ({} channels)".format(channel)
                self.train_metric_names += [m]
                self.train_metric_best += ["max"]
            elif channel in ["Db", "Dc", "Dn", "D", "H", "V", "Z", "R"]:
                m = "L1 ({} channel)".format(channel) if channel != "R" else "L1 ({} channels)".format(channel)
                self.train_metric_names += ["L1 ({} channel)".format(channel)]
                self.train_metric_best += ["min"]
            elif channel == "E_offset":
                self.train_metric_names += ["IoU"]
                self.train_metric_best += ["max"]
            elif channel in ["E_sigma", "E_seediness"]:
               continue  # No metrics for these channels
            elif channel == "We":
                continue

            # Extra channels for the synapse detection branch
            elif channel == "F_pre":
                self.train_metric_names += ["IoU (pre-sites)"]
                self.train_metric_best += ["max"]
            elif channel == "F_post":
                self.train_metric_names += ["IoU (post-sites)"]
                self.train_metric_best += ["max"]
            else:
                raise ValueError("Unknown channel: {}".format(channel))
        
        # Multi-head: instances + classification
        if self.cfg.PROBLEM.INSTANCE_SEG.TYPE == "regular" and self.multihead:
            self.train_metric_names += ["IoU (classes)"]
            self.train_metric_best += ["max"]
            # Used to calculate IoU with the classification results
            self.jaccard_index_matching = jaccard_index(
                device=self.device, 
                num_classes=self.cfg.DATA.N_CLASSES,
                ndim=self.dims,
                ignore_index=self.cfg.LOSS.IGNORE_INDEX,
            )
        
        if "E_offset" in self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS:
            # No metric for the embedding representation during training as the IoU is calculated together with the loss
            self.train_metrics.append("none") 
        else:
            self.train_metrics.append(
                multiple_metrics(
                    num_classes=self.cfg.DATA.N_CLASSES,
                    metric_names=self.train_metric_names,
                    device=self.device,
                    out_channels=self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS,
                    channel_extra_opts = self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS_EXTRA_OPTS[0],
                    model_source=self.cfg.MODEL.SOURCE,
                    ignore_index=self.cfg.LOSS.IGNORE_INDEX,
                    ndim=self.dims,
                )
            )
    
        self.test_metrics = []
        self.test_metric_names = self.train_metric_names.copy()

        # Multi-head: instances + classification
        if self.multihead:
            self.test_metric_names.append("IoU (classes)")
            # Used to calculate IoU with the classification results
            self.jaccard_index_matching = jaccard_index(
                device=self.device, 
                num_classes=self.cfg.DATA.N_CLASSES,
                ndim=self.dims,
                ignore_index=self.cfg.LOSS.IGNORE_INDEX,
            )

        if "E_offset" in self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS:
            # No metric for the embedding representation during training as the IoU is calculated together with the loss
            self.test_metrics.append("none")
        else:
            self.test_metrics.append(
                multiple_metrics(
                    num_classes=self.cfg.DATA.N_CLASSES,
                    metric_names=self.test_metric_names,
                    device=self.device,
                    out_channels=self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS,
                    channel_extra_opts = self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS_EXTRA_OPTS[0],
                    model_source=self.cfg.MODEL.SOURCE,
                    ndim=self.dims,
                    ignore_index=self.cfg.LOSS.IGNORE_INDEX,
                )
            )
        
        if self.cfg.PROBLEM.INSTANCE_SEG.TYPE == "synapses":
            self.test_extra_metrics = ["Precision (pre-points)", "Recall (pre-points)", "F1 (pre-points)", "TP (pre-points)", "FP (pre-points)", "FN (pre-points)"]
            self.test_extra_metrics += ["Precision (post-points)", "Recall (post-points)", "F1 (post-points)", "TP (post-points)", "FP (post-points)", "FN (post-points)"]
            self.test_metric_names += self.test_extra_metrics

        if "E_offset" in self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS:
            instance_loss = SpatialEmbLoss(
                patch_size=self.cfg.DATA.PATCH_SIZE,
                ndims=self.dims,
                anisotropy=self.resolution,
                weights=self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNEL_WEIGHTS,
                center_mode=self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS_EXTRA_OPTS[0].get("E_offset", {}).get("center_mode", "centroid"),
                medoid_max_points=self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS_EXTRA_OPTS[0].get("E_offset", {}).get("medoid_max_points", 10000),
            ).to(self.device, non_blocking=True)
        else:
            instance_loss = instance_segmentation_loss(
                weights = self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNEL_WEIGHTS,
                out_channels = self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS,
                losses_to_use = self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS_LOSSES,
                channel_extra_opts = self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS_EXTRA_OPTS[0],
                channels_expected = self.real_classes,
                n_classes=self.cfg.DATA.N_CLASSES,
                class_rebalance=self.cfg.LOSS.CLASS_REBALANCE,
                class_weights=self.cfg.LOSS.CLASS_WEIGHTS,
                ignore_index=self.cfg.LOSS.IGNORE_INDEX
            )
        
        if self.cfg.LOSS.CONTRAST.ENABLE: 
            self.loss = ContrastCELoss(
                main_loss=instance_loss, # type: ignore
                ndim=self.dims,
                ignore_index=self.cfg.LOSS.IGNORE_INDEX,
            )
        else:
            self.loss = instance_loss

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
                        if isinstance(val[m], torch.Tensor):
                            v = val[m].item() if not torch.isnan(val[m]) else 0
                        else:
                            v = val[m]
                        out_metrics[list_names_to_use[k]] = v
                        if metric_logger:
                            metric_logger.meters[list_names_to_use[k]].update(v)
                        k += 1
                else:
                    if isinstance(val[m], torch.Tensor):
                        val = val.item() if not torch.isnan(val) else 0
                    else:
                        v = val[m]
                    out_metrics[list_names_to_use[i]] = val
                    if metric_logger:
                        metric_logger.meters[list_names_to_use[i]].update(val)
        return out_metrics

    def instance_seg_process(self, pred, filenames, out_dir, out_dir_post_proc, calculate_metrics: bool = True):
        """
        Instance segmentation workflow engine for test/inference.
        
        Process model's prediction to prepare
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

            w_dir = os.path.join(self.cfg.PATHS.WATERSHED_DIR, filenames[0])
            check_wa = w_dir if self.cfg.PROBLEM.INSTANCE_SEG.WATERSHED.DATA_CHECK_MW else None

            if "R" in self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS:
                print("Creating instances with Stardist procedure . . .")
                pred_labels, _ = stardist_instances_from_prediction(
                    pred[..., :self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS.index("R")].squeeze(), 
                    pred[..., self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS.index("R"):].squeeze(), 
                    prob_thresh=self.cfg.PROBLEM.INSTANCE_SEG.STARDIST.PROB_THRESH, 
                    nms_iou_thresh=self.cfg.PROBLEM.INSTANCE_SEG.STARDIST.NMS_IOU_THRESH, 
                    anisotropy=self.resolution[-self.dims:], # as a 1 is added at the beginning for 2D
                    grid=self.stardist_grid, 
                )
            elif "E_offset" in self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS:
                pred_labels = embedseg_instances(
                    y_pred=pred,
                    s_fg=self.cfg.PROBLEM.INSTANCE_SEG.EMBEDSEG.SEED_THRESH,  
                    s_min=self.cfg.PROBLEM.INSTANCE_SEG.EMBEDSEG.MIN_SIZE,
                    assign_thresh=self.cfg.PROBLEM.INSTANCE_SEG.EMBEDSEG.ASSIGN_THRESH,
                )
            else:
                print("Creating instances with watershed . . .")
                pred_labels = watershed_by_channels(
                    data=pred,
                    channels=self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS,
                    seed_channels=self.cfg.PROBLEM.INSTANCE_SEG.WATERSHED.SEED_CHANNELS,
                    seed_channel_ths=self.cfg.PROBLEM.INSTANCE_SEG.WATERSHED.SEED_CHANNELS_THRESH,
                    topo_surface_channel=self.cfg.PROBLEM.INSTANCE_SEG.WATERSHED.TOPOGRAPHIC_SURFACE_CHANNEL,
                    growth_mask_channels=self.cfg.PROBLEM.INSTANCE_SEG.WATERSHED.GROWTH_MASK_CHANNELS,
                    growth_mask_channel_ths=self.cfg.PROBLEM.INSTANCE_SEG.WATERSHED.GROWTH_MASK_CHANNELS_THRESH,
                    remove_before=self.cfg.PROBLEM.INSTANCE_SEG.WATERSHED.DATA_REMOVE_BEFORE_MW,
                    thres_small_before=self.cfg.PROBLEM.INSTANCE_SEG.WATERSHED.DATA_REMOVE_SMALL_OBJ_BEFORE,
                    seed_morph_sequence=self.cfg.PROBLEM.INSTANCE_SEG.WATERSHED.SEED_MORPH_SEQUENCE,
                    seed_morph_radius=self.cfg.PROBLEM.INSTANCE_SEG.WATERSHED.SEED_MORPH_RADIUS,
                    erode_and_dilate_growth_mask=self.cfg.PROBLEM.INSTANCE_SEG.WATERSHED.ERODE_AND_DILATE_GROWTH_MASK,
                    fore_erosion_radius=self.cfg.PROBLEM.INSTANCE_SEG.WATERSHED.FORE_EROSION_RADIUS,
                    fore_dilation_radius=self.cfg.PROBLEM.INSTANCE_SEG.WATERSHED.FORE_DILATION_RADIUS,
                    resolution=self.resolution,
                    save_dir=check_wa,
                    watershed_by_2d_slices=self.cfg.PROBLEM.INSTANCE_SEG.WATERSHED.BY_2D_SLICES,
                )

            # Multi-head: instances + classification
            if self.multihead:
                print("Adapting class channel . . .")
                labels = np.unique(pred_labels)[1:]
                new_class_channel = np.zeros(pred_labels.shape, dtype=pred_labels.dtype)
                # Classify each instance counting the most prominent class of all the pixels that compose it
                for l in labels:
                    instance_classes, instance_classes_count = np.unique(class_channel[pred_labels == l], return_counts=True)

                    # Remove background
                    if instance_classes[0] == 0:
                        instance_classes = instance_classes[1:]
                        instance_classes_count = instance_classes_count[1:]

                    if len(instance_classes) > 0:
                        label_selected = int(instance_classes[np.argmax(instance_classes_count)])
                    else:  # Label by default with class 1 in case there was no class info
                        label_selected = 1
                    new_class_channel = np.where(pred_labels == l, label_selected, new_class_channel)

                class_channel = new_class_channel
                class_channel = class_channel.squeeze()
                del new_class_channel
                save_tif(
                    np.expand_dims(
                        np.concatenate(
                            [
                                np.expand_dims(pred_labels.squeeze(), -1),
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
                    np.expand_dims(np.expand_dims(pred_labels, -1), 0),
                    out_dir,
                    filenames,
                    verbose=self.cfg.TEST.VERBOSE,
                )

            # Add extra dimension if working in 2D
            if pred_labels.ndim == 2:
                pred_labels = np.expand_dims(pred_labels, 0)
        else:
            pred_labels = pred.squeeze()
            if pred_labels.ndim == 2:
                pred_labels = np.expand_dims(pred_labels, 0)

        results = None
        results_class = None
        if (
            calculate_metrics
            and self.cfg.TEST.MATCHING_STATS
            and (self.cfg.DATA.TEST.LOAD_GT or self.cfg.DATA.TEST.USE_VAL_AS_TEST)
        ):
            print("Calculating matching stats . . .")

            # Need to load instance labels, as Y are binary channels used for IoU calculation
            if self.cfg.TEST.ANALIZE_2D_IMGS_AS_3D_STACK and len(self.test_filenames) == pred_labels.shape[0]:
                del self.current_sample["Y"]
                _Y = np.zeros(pred_labels.shape, dtype=pred_labels.dtype)
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

                if test_file.endswith(".zarr") or test_file.endswith(".n5"):
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
                        f"Image {test_file} wrong dimension. In instance segmentation, when 'DATA.N_CLASSES' are "
                        f"more than 2 labels need to have two channels, e.g. {error_shape}, containing the instance "
                        "segmentation map (first channel) and classification map (second channel)."
                    )

                # Separate instance and classification channels
                _Y_classes = _Y[..., 1]  # Classes
                _Y = _Y[..., 0]  # Instances

                # Measure class IoU
                class_iou = self.jaccard_index_matching(
                    torch.as_tensor(class_channel.squeeze().astype(np.int32)).to(self.device, non_blocking=True),
                    torch.as_tensor(_Y_classes.squeeze().astype(np.int32)).to(self.device, non_blocking=True),
                )
                class_iou = class_iou.item() if not torch.isnan(class_iou) else 0
                print(f"Class IoU: {class_iou}")
                results_class = class_iou

            if _Y.ndim == 2:
                _Y = np.expand_dims(_Y, 0)

            # For torchvision models that resize need to rezise the images
            if pred_labels.shape != _Y.shape:
                pred_labels = resize(pred_labels, _Y.shape, order=0)

            # Convert instances to integer
            if _Y.dtype == np.float32:
                _Y = _Y.astype(np.uint32)
            if _Y.dtype == np.float64:
                _Y = _Y.astype(np.uint64)

            diff_ths_colored_img = abs(
                len(self.cfg.TEST.MATCHING_STATS_THS_COLORED_IMG) - len(self.cfg.TEST.MATCHING_STATS_THS)
            )
            colored_img_ths = self.cfg.TEST.MATCHING_STATS_THS_COLORED_IMG + [-1] * diff_ths_colored_img

            results = matching(_Y, pred_labels, thresh=self.cfg.TEST.MATCHING_STATS_THS, report_matches=True)
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
                    colored_result = np.zeros(pred_labels.shape + (3,), dtype=np.uint8)

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
                        colored_result[np.where(pred_labels == fp_instances[j])] = (
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
        if self.cfg.TEST.POST_PROCESSING.FILL_HOLES:
            pred_labels = fill_label_holes(pred_labels)
            
        if self.cfg.TEST.POST_PROCESSING.REPARE_LARGE_BLOBS_SIZE != -1:
            if self.cfg.PROBLEM.NDIM == "2D":
                pred_labels = pred_labels[0]
            pred_labels = repare_large_blobs(pred_labels[0], self.cfg.TEST.POST_PROCESSING.REPARE_LARGE_BLOBS_SIZE)
            if self.cfg.PROBLEM.NDIM == "2D":
                pred_labels = np.expand_dims(pred_labels, 0)

        if self.cfg.TEST.POST_PROCESSING.VORONOI_ON_MASK:
            erode_size = 0
            if "M" in self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS:
                ch_pos = self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS.index("M")
                pred = pred[...,ch_pos]
            elif "F" in self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS:
                pred = pred[...,self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS.index("F")]
                if "C" in self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS:
                    pred += pred[...,self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS.index("C")]
            elif "B" in self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS:
                pred = 1 - pred[...,self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS.index("B")]    
            elif "C" in self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS:
                pred = pred[...,self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS.index("C")] > self.cfg.TEST.POST_PROCESSING.VORONOI_TH
                erode_size = 2 # As the contours are thicker we erode a little bit

            pred_labels = voronoi_on_mask(
                pred_labels,
                pred,
                th=self.cfg.TEST.POST_PROCESSING.VORONOI_TH,
                verbose=self.cfg.TEST.VERBOSE,
                erode_size=erode_size,
            )
        del pred

        if self.cfg.TEST.POST_PROCESSING.CLEAR_BORDER:
            print("Clearing borders . . .")
            if self.cfg.PROBLEM.NDIM == "2D":
                pred_labels = pred_labels[0]
            pred_labels = clear_border(pred_labels)
            if self.cfg.PROBLEM.NDIM == "2D":
                pred_labels = np.expand_dims(pred_labels, 0)

        if (
            self.cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.ENABLE
            or self.cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.ENABLE
        ):
            if self.cfg.PROBLEM.NDIM == "2D":
                pred_labels = pred_labels[0]
            pred_labels, d_result = measure_morphological_props_and_filter(
                pred_labels,
                self.resolution,
                filter_instances=self.cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.ENABLE,
                properties=self.cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS,
                prop_values=self.cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.VALUES,
                comp_signs=self.cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.SIGNS,
            )
            if self.cfg.PROBLEM.NDIM == "2D":
                pred_labels = np.expand_dims(pred_labels, 0)

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
                pred_labels = pred_labels[0]

            # Multi-head: instances + classification
            if self.multihead:
                class_channel = np.where(pred_labels > 0, class_channel, 0)  # Adapt changes to post-processed pred_labels
                save_tif(
                    np.expand_dims(
                        np.concatenate(
                            [
                                np.expand_dims(pred_labels, -1),
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
                    np.expand_dims(np.expand_dims(pred_labels, -1), 0),
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
                    pred_labels = np.expand_dims(pred_labels, 0)

                print("Calculating matching stats after post-processing . . .")
                results_post_proc = matching(
                    _Y,
                    pred_labels,
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
                        colored_result = np.zeros(pred_labels.shape + (3,), dtype=np.uint8)

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
                            colored_result[np.where(pred_labels == fp_instances[j])] = (
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
        Synapse segmentation workflow engine for test/inference.
        
        Process model's prediction to prepare
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
        threshold_abs = []
        for c in range(pred.shape[-1]): 
            if self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.TH_TYPE == "auto":
                threshold_abs.append(threshold_otsu(pred[..., c]))
            else: # "manual", "relative_by_patch", "relative"
                threshold_abs.append(self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.MIN_TH_TO_BE_PEAK)

        pred, d_result = create_synapses(
            data=pred,
            channels=self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS,
            point_creation_func=self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.POINT_CREATION_FUNCTION,
            min_th_to_be_peak=threshold_abs,
            min_distance=self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.PEAK_LOCAL_MAX_MIN_DISTANCE,
            min_sigma=self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.BLOB_LOG_MIN_SIGMA,
            max_sigma=self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.BLOB_LOG_MAX_SIGMA,
            num_sigma=self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.BLOB_LOG_NUM_SIGMA,
            exclude_border=self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.EXCLUDE_BORDER,
            relative_th_value=self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.TH_TYPE in ["relative", "relative_by_patch"], 
        )
        if out_dir is not None:
            save_tif(
                np.expand_dims(pred, 0),
                out_dir,
                filenames,
                verbose=self.cfg.TEST.VERBOSE,
            )

        total_pre_points = len([x for x in d_result["tag"] if x == "pre"])
        pre_points = np.array(d_result["points"][:total_pre_points])
        pre_points_df = pd.DataFrame(
            zip(
                d_result["ids"][:total_pre_points],
                list(pre_points[:, 0]),
                list(pre_points[:, 1]),
                list(pre_points[:, 2]),
                d_result["probabilities"][:total_pre_points],
                [threshold_abs[0],]*total_pre_points,
            ),
            columns=[
                "pre_id",
                "axis-0",
                "axis-1",
                "axis-2",
                "probability",
                "pre th",
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

        post_points = np.array(d_result["points"][total_pre_points:])
        post_points_df = pd.DataFrame(
            zip(
                d_result["ids"][total_pre_points:],
                list(post_points[:, 0]),
                list(post_points[:, 1]),
                list(post_points[:, 2]),
                d_result["probabilities"][total_pre_points:],
                [threshold_abs[1],]*len(post_points),
            ),
            columns=[
                "post_id",
                "axis-0",
                "axis-1",
                "axis-2",
                "probability",
                "post th",
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
        post_ids = post_points_df["post_id"].to_list()
        if len(pre_points) > 0 and len(post_points) > 0:
            for i in range(len(pre_points)):
                pre_post_mapping[pre_ids[i]] = []

            # Match each post with a pre
            distances = distance_matrix(post_points, pre_points)
            for i in range(len(post_points)):
                closest_pre_point = np.argmin(distances[i])
                closest_pre_point = pre_ids[closest_pre_point]
                pre_post_mapping[closest_pre_point].append(post_ids[i])

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
            if self.cfg.TEST.VERBOSE and not self.cfg.TEST.BY_CHUNKS.ENABLE:
                print("No pre/post synaptic points found!")

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
                for n, item in enumerate(d_metrics.items()):
                    metric = self.test_extra_metrics[n]
                    if str(metric).lower() not in self.stats["merge_patches"]:
                        self.stats["merge_patches"][str(metric.lower())] = 0
                    self.stats["merge_patches"][str(metric).lower()] += item[1]
                    self.current_sample_metrics[str(metric).lower() + " (pre points)"] = item[1]

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
                previous_pre_keys_num = len(d_metrics)
                for n, item in enumerate(d_metrics.items()):
                    metric = self.test_extra_metrics[n+previous_pre_keys_num]
                    if str(metric).lower() not in self.stats["merge_patches"]:
                        self.stats["merge_patches"][str(metric.lower())] = 0
                    self.stats["merge_patches"][str(metric).lower()] += item[1]
                    self.current_sample_metrics[str(metric).lower() + " (post points)"] = item[1]
                
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
                    for n, item in enumerate(d_metrics.items()):
                        metric = self.test_extra_metrics[n]
                        if str(metric).lower() not in self.stats["merge_patches_post"]:
                            self.stats["merge_patches_post"][str(metric.lower())] = 0
                        self.stats["merge_patches_post"][str(metric).lower()] += item[1]
                        self.current_sample_metrics[str(metric).lower() + " (pre points, post-processing)"] = item[1]
                        
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
                    previous_pre_keys_num = len(d_metrics)
                    for n, item in enumerate(d_metrics.items()):
                        metric = self.test_extra_metrics[n+previous_pre_keys_num]
                        if str(metric).lower() not in self.stats["merge_patches_post"]:
                            self.stats["merge_patches_post"][str(metric.lower())] = 0
                        self.stats["merge_patches_post"][str(metric).lower()] += item[1]
                        self.current_sample_metrics[str(metric).lower() + " (post points, post-processing)"] = item[1]
                        
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
        """Process a sample in the inference phase."""
        if self.cfg.MODEL.SOURCE != "torchvision":
            self.instances_already_created = False
            super().process_test_sample()
        else:
            # Skip processing image
            if "discard" in self.current_sample and self.current_sample["discard"]:
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
        Execute steps needed after merging all predicted patches into the original image.

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
                        for i, r_per_th in enumerate(r):
                            prefix = str(r_per_th['thresh']) + " TH " if 'thresh' in r_per_th else f"TH{i} "
                            for mkey in ['fp','tp','fn','precision','recall','accuracy','f1','n_true','n_pred','mean_true_score','mean_matched_score','panoptic_quality']:
                                if mkey in r_per_th:
                                    self.current_sample_metrics[prefix + mkey] = r_per_th[mkey]
                    if r_post:
                        self.all_matching_stats_merge_patches_post.append(r_post)
                        for i, r_per_th in enumerate(r_post):
                            prefix = str(r_per_th['thresh']) + " TH (post)" if 'thresh' in r_per_th else f"TH{i} (post) "
                            for mkey in ['fp','tp','fn','precision','recall','accuracy','f1','n_true','n_pred','mean_true_score','mean_matched_score','panoptic_quality']:
                                if mkey in r_per_th:
                                    self.current_sample_metrics[prefix + mkey] = r_per_th[mkey]
                    if rcls:
                        self.all_class_stats_merge_patches.append(rcls)
                        self.current_sample_metrics["class iou"] = rcls
                    if rcls_post:
                        self.all_class_stats_merge_patches_post.append(rcls_post)
                        self.current_sample_metrics["class iou (post)"] = rcls_post
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

            _filename, _ = os.path.splitext(os.path.basename(self.current_sample["filename"]))
            if pre_points_df is not None and len(pre_points_df) > 0:
                # Remove possible points in the padded area
                pre_points_df = pre_points_df[pre_points_df["axis-0"] < patch.shape[0] - added_pad[0][1]]
                pre_points_df = pre_points_df[pre_points_df["axis-1"] < patch.shape[1] - added_pad[1][1]]
                pre_points_df = pre_points_df[pre_points_df["axis-2"] < patch.shape[2] - added_pad[2][1]]
                pre_points_df["axis-0"] = pre_points_df["axis-0"] - added_pad[0][0]
                pre_points_df["axis-1"] = pre_points_df["axis-1"] - added_pad[1][0]
                pre_points_df["axis-2"] = pre_points_df["axis-2"] - added_pad[2][0]
                pre_points_df = pre_points_df[pre_points_df["axis-0"] >= 0]
                pre_points_df = pre_points_df[pre_points_df["axis-1"] >= 0]
                pre_points_df = pre_points_df[pre_points_df["axis-2"] >= 0]
                
                # Add the patch shift to the detected coordinates so they represent global coords
                pre_points_df["axis-0"] = pre_points_df["axis-0"] + patch_in_data.z_start
                pre_points_df["axis-1"] = pre_points_df["axis-1"] + patch_in_data.y_start
                pre_points_df["axis-2"] = pre_points_df["axis-2"] + patch_in_data.x_start

                # Save the csv file
                if len(pre_points_df) > 0:
                    os.makedirs(self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK, exist_ok=True)
                    pre_points_df.to_csv(
                        os.path.join(
                            self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK,
                            _filename + "_patch" + str(patch_id).zfill(len(str(len(self.test_generator)))) + "_pre_points.csv",
                        ),
                        index=False,
                    )
                
            if post_points_df is not None and len(post_points_df) > 0:
                # Remove possible points in the padded area
                post_points_df = post_points_df[post_points_df["axis-0"] < patch.shape[0] - added_pad[0][1]]
                post_points_df = post_points_df[post_points_df["axis-1"] < patch.shape[1] - added_pad[1][1]]
                post_points_df = post_points_df[post_points_df["axis-2"] < patch.shape[2] - added_pad[2][1]]
                post_points_df["axis-0"] = post_points_df["axis-0"] - added_pad[0][0]
                post_points_df["axis-1"] = post_points_df["axis-1"] - added_pad[1][0]
                post_points_df["axis-2"] = post_points_df["axis-2"] - added_pad[2][0]
                post_points_df = post_points_df[post_points_df["axis-0"] >= 0]
                post_points_df = post_points_df[post_points_df["axis-1"] >= 0]
                post_points_df = post_points_df[post_points_df["axis-2"] >= 0]

                # Add the patch shift to the detected coordinates so they represent global coords
                post_points_df["axis-0"] = post_points_df["axis-0"] + patch_in_data.z_start
                post_points_df["axis-1"] = post_points_df["axis-1"] + patch_in_data.y_start
                post_points_df["axis-2"] = post_points_df["axis-2"] + patch_in_data.x_start

                # Save the csv file
                if len(post_points_df) > 0:
                    os.makedirs(self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK, exist_ok=True)
                    post_points_df.to_csv(
                        os.path.join(
                            self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK,
                            _filename + "_patch" + str(patch_id).zfill(len(str(len(self.test_generator)))) + "_post_points.csv",
                        ),
                        index=False,
                    )

    def after_all_patch_prediction_by_chunks(self):
        """Execute steps needed after merging all predicted patches into the original image in "by chunks" setting."""
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
            filename = os.path.splitext(self.current_sample["filename"])[0]
            pre_points_df, post_points_df, pre_post_map_df = None, None, None
            if not self.cfg.TEST.REUSE_PREDICTIONS:
                # For synapses we need to map the pre to the post points. It needs to be done here and not patch by patch as
                # some pre points may lay in other chunks of the data.
                input_dir = self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK
                all_pred_files = sorted(next(os_walk_clean(input_dir))[2])
                all_pred_files = [x for x in all_pred_files if filename + "_patch" in x]
                all_pre_point_files = [x for x in all_pred_files if "_pre_points.csv" in x and "all_points.csv" not in x]
                all_post_point_files = [x for x in all_pred_files if "_post_points.csv" in x and "all_points.csv" not in x]
                all_pre_dfs, all_post_dfs = [], []

                # Collect pre dataframes 
                if len(all_pre_point_files) > 0:
                    for pre_file in all_pre_point_files:
                        pre_file_path = os.path.join(input_dir, pre_file)
                        pred_pre_df = pd.read_csv(pre_file_path, index_col=False)
                        if len(pred_pre_df) > 0:
                            pred_pre_df = pred_pre_df.drop(columns=["pre th"])
                            all_pre_dfs.append(pred_pre_df)

                # Collect post dataframes 
                if len(all_post_point_files) > 0:
                    for post_file in all_post_point_files:
                        post_file_path = os.path.join(input_dir, post_file)
                        pred_post_df = pd.read_csv(post_file_path, index_col=False)
                        if len(pred_post_df) > 0:
                            pred_post_df = pred_post_df.drop(columns=["post th"])
                            all_post_dfs.append(pred_post_df)      

                # Save then the pre and post sites separately
                if len(all_pre_dfs) > 0:
                    pre_points_df = pd.concat(all_pre_dfs, ignore_index=True)
                    pre_points_df.sort_values(by=["axis-0"])

                    pre_th_global = self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.MIN_TH_TO_BE_PEAK
                    if self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.TH_TYPE == "manual":
                        print(f"Using global threshold (pre-points): {pre_th_global}")
                    elif self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.TH_TYPE == "relative":
                        max_val = max(pre_points_df["probability"])
                        pre_th_global = max_val*self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.MIN_TH_TO_BE_PEAK
                        print(f"Using global threshold (pre-points): {pre_th_global} (max: {max_val})")
                    pre_points_df = pre_points_df[pre_points_df["probability"] > pre_th_global]

                    pre_points_df["pre_id"] = list(range(1, len(pre_points_df)+1))
                    os.makedirs(self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK, exist_ok=True)
                    pre_points_df.to_csv(
                        os.path.join(
                            self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK,
                            filename+"_pred_pre_locations.csv",
                        ),
                        index=False,
                    )
                if len(all_post_dfs) > 0:
                    post_points_df = pd.concat(all_post_dfs, ignore_index=True)
                    post_points_df.sort_values(by=["axis-0"])

                    post_th_global = self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.MIN_TH_TO_BE_PEAK
                    if self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.TH_TYPE == "manual":
                        print(f"Using global threshold (post-points): {post_th_global}")
                    elif self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.TH_TYPE == "relative":
                        max_val = max(post_points_df["probability"])
                        post_th_global = max_val*self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.MIN_TH_TO_BE_PEAK
                        print(f"Using global threshold (post-points): {post_th_global} (max: {max_val})")
                    post_points_df = post_points_df[post_points_df["probability"] > post_th_global]

                    post_points_df["post_id"] = list(range(1, len(post_points_df)+1))
                    os.makedirs(self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK, exist_ok=True)
                    post_points_df.to_csv(
                        os.path.join(
                            self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK,
                            filename+"_pred_post_locations.csv",
                        ),
                        index=False,
                    )

                # Create coordinate arrays
                pre_points, post_points = [], []
                if len(all_pre_dfs) > 0 and pre_points_df is not None:
                    for coord in zip(pre_points_df["axis-0"], pre_points_df["axis-1"], pre_points_df["axis-2"]):
                        pre_points.append(list(coord))
                if len(all_post_dfs) > 0 and post_points_df is not None:
                    for coord in zip(post_points_df["axis-0"], post_points_df["axis-1"], post_points_df["axis-2"]):
                        post_points.append(list(coord))
                pre_points = np.array(pre_points)
                post_points = np.array(post_points)

                pre_post_mapping = {}
                pres, posts = [], []
                if len(pre_points) > 0 and pre_points_df is not None and len(post_points) > 0 and post_points_df is not None:
                    pre_ids = pre_points_df["pre_id"].to_list()
                    if len(post_points) > 0 and post_points_df is not None:
                        post_ids = post_points_df["post_id"].to_list()
                    for i in range(len(pre_points)):
                        pre_post_mapping[pre_ids[i]] = []

                    # Match each post with a pre
                    distances = distance_matrix(post_points, pre_points)
                    for i in range(len(post_points)):
                        closest_pre_point = np.argmin(distances[i])
                        closest_pre_point = pre_ids[closest_pre_point]
                        pre_post_mapping[closest_pre_point].append(post_ids[i])

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
                        if len(pre_points) == 0:
                            print("No pre synaptic points found!")
                        if len(post_points) == 0:
                            print("No post synaptic points found!")

                if len(pres) > 0 and len(posts) > 0:
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
                    pre_post_map_df.sort_values(by=["pre_id", "post_id"])
                    pre_post_map_df.to_csv(
                        os.path.join(
                            self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK,
                            filename+"_pre_post_mapping.csv",
                        ),
                        index=False,
                    )
            else:
                # Read the dataframes
                pre_points_df = pd.read_csv(
                    os.path.join(
                        self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK,
                        filename+"_pred_pre_locations.csv",
                    )
                )
                post_points_df = pd.read_csv(
                    os.path.join(
                        self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK,
                        filename+"_pred_post_locations.csv",
                    )
                )
                pre_post_map_df = pd.read_csv(
                    os.path.join(
                        self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK,
                        filename+"_pre_post_mapping.csv",
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

                post_th_global = self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.MIN_TH_TO_BE_PEAK
                pre_th_global = self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.MIN_TH_TO_BE_PEAK
                if self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.TH_TYPE == "manual":
                    print(f"Using global threshold (pre-points): {pre_th_global}")
                    print(f"Using global threshold (post-points): {post_th_global}")
                elif self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.TH_TYPE == "relative":
                    max_val = max(pre_points_df["probability"]) if len(pre_points_df) > 0 else 1
                    pre_th_global = max_val*self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.MIN_TH_TO_BE_PEAK
                    print(f"Using global threshold (pre-points): {pre_th_global} (max: {max_val})")
                    max_val = max(post_points_df["probability"]) if len(post_points_df) > 0 else 1
                    post_th_global = max_val*self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.MIN_TH_TO_BE_PEAK
                    print(f"Using global threshold (post-points): {post_th_global} (max: {max_val})")

            if self.cfg.DATA.TEST.LOAD_GT or self.cfg.DATA.TEST.USE_VAL_AS_TEST:
                print("Calculating synapse detection stats . . .")
                locations_path = self.cfg.DATA.TEST.INPUT_ZARR_MULTIPLE_DATA_LOCATIONS_PATH
                resolution_path = self.cfg.DATA.TEST.INPUT_ZARR_MULTIPLE_DATA_RESOLUTION_PATH
                partners_path = self.cfg.DATA.TEST.INPUT_ZARR_MULTIPLE_DATA_PARTNERS_PATH
                id_path = self.cfg.DATA.TEST.INPUT_ZARR_MULTIPLE_DATA_ID_PATH
                data_filename = os.path.join(self.current_sample["dir"], self.current_sample["filename"])
                file, ids = read_chunked_nested_data(data_filename, id_path)
                ids = list(np.array(ids))
                _, partners = read_chunked_nested_data(data_filename, partners_path)
                partners = np.array(partners)
                _, locations = read_chunked_nested_data(data_filename, locations_path)
                locations = np.array(locations)
                _, resolution = read_chunked_nested_data(data_filename, resolution_path)
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
                    d_metrics, pre_gt_assoc, pre_fp = detection_metrics(
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
                    for n, item in enumerate(d_metrics.items()):
                        metric = self.test_extra_metrics[n]
                        if str(metric).lower() not in self.stats["merge_patches"]:
                            self.stats["merge_patches"][str(metric.lower())] = 0
                        self.stats["merge_patches"][str(metric).lower()] += item[1]
                        self.current_sample_metrics[str(metric).lower() + " (pre points)"] = item[1]

                    # Save csv files with the associations between GT points and predicted ones
                    pre_gt_assoc.to_csv(
                        os.path.join(
                            self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK,
                            filename+"_pred_pre_locations_gt_assoc.csv",
                        ),
                        index=False,
                    )
                    pre_fp.to_csv(
                        os.path.join(
                            self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK,
                            filename+"_pred_pre_locations_fp.csv",
                        ),
                        index=False,
                    )
                 
                if len(post_points):
                    d_metrics, post_gt_assoc, post_fp = detection_metrics(
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
                    previous_pre_keys_num = len(d_metrics)
                    for n, item in enumerate(d_metrics.items()):
                        metric = self.test_extra_metrics[n+previous_pre_keys_num]
                        if str(metric).lower() not in self.stats["merge_patches"]:
                            self.stats["merge_patches"][str(metric.lower())] = 0
                        self.stats["merge_patches"][str(metric).lower()] += item[1]
                        self.current_sample_metrics[str(metric).lower() + " (post points)"] = item[1]

                    # Save csv files with the associations between GT points and predicted ones
                    post_gt_assoc.to_csv(
                        os.path.join(
                            self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK,
                            filename+"_pred_post_locations_gt_assoc.csv",
                        ),
                        index=False,
                    )
                    post_fp.to_csv(
                        os.path.join(
                            self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK,
                            filename+"_pred_post_locations_fp.csv",
                        ),
                        index=False,
                    )

            # Remove close points
            if self.post_processing['per_image']:
                if (
                    len(pre_points) > 0 
                    and pre_points_df is not None 
                    and self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.REMOVE_CLOSE_PRE_POINTS_RADIUS > 0
                ):
                    if self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.REMOVE_CLOSE_POINTS_RADIUS_BY_MASK:    
                        # Load H5/Zarr and convert it into numpy array
                        fpath = os.path.join(
                            self.cfg.PATHS.RESULT_DIR.PER_IMAGE, os.path.splitext(self.current_sample["filename"])[0] + ".zarr"
                        )
                        pred_file, pred = read_chunked_data(fpath) 
                        
                        pre_points, pre_dropped_pos = remove_close_points_by_mask(  # type: ignore
                            points=pre_points,
                            radius=self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.REMOVE_CLOSE_PRE_POINTS_RADIUS,
                            raw_predictions=pred,
                            bin_th=pre_th_global,
                            resolution=resolution,
                            channel_to_look_into=1, # post channel
                            ndim=self.dims,
                            return_drops=True,
                        )

                        if isinstance(pred_file, h5py.File):
                            pred_file.close()
                    else:
                        pre_points, pre_dropped_pos = remove_close_points(  # type: ignore
                            pre_points,
                            self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.REMOVE_CLOSE_PRE_POINTS_RADIUS,
                            resolution,
                            ndim=self.dims,
                            return_drops=True,
                        )
                    pre_points_df.drop(pre_points_df.index[pre_dropped_pos], inplace=True)  # type: ignore
                    os.makedirs(self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK_POST_PROCESSING, exist_ok=True)
                    pre_points_df.to_csv(
                        os.path.join(
                            self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK_POST_PROCESSING,
                            filename+"_pred_pre_locations.csv",
                        ),
                        index=False,
                    )

                if (
                    len(post_points) > 0 
                    and post_points_df is not None 
                    and self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.REMOVE_CLOSE_POST_POINTS_RADIUS > 0
                ):   
                    if self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.REMOVE_CLOSE_POINTS_RADIUS_BY_MASK:    
                        # Load H5/Zarr and convert it into numpy array
                        fpath = os.path.join(
                            self.cfg.PATHS.RESULT_DIR.PER_IMAGE, os.path.splitext(self.current_sample["filename"])[0] + ".zarr"
                        )
                        pred_file, pred = read_chunked_data(fpath) 
                        
                        post_points, post_dropped_pos = remove_close_points_by_mask(  # type: ignore
                            points=post_points,
                            radius=self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.REMOVE_CLOSE_POST_POINTS_RADIUS,
                            raw_predictions=pred,
                            bin_th=post_th_global,
                            resolution=resolution,
                            channel_to_look_into=1, # post channel
                            ndim=self.dims,
                            return_drops=True,
                        )

                        if isinstance(pred_file, h5py.File):
                            pred_file.close()
                    else:
                        post_points, post_dropped_pos = remove_close_points(  # type: ignore
                            post_points,
                            self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.REMOVE_CLOSE_POST_POINTS_RADIUS,
                            resolution,
                            ndim=self.dims,
                            return_drops=True,
                        )
                    post_points_df.drop(post_points_df.index[post_dropped_pos], inplace=True)  # type: ignore

                    os.makedirs(self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK_POST_PROCESSING, exist_ok=True)
                    post_points_df.to_csv(
                        os.path.join(
                            self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK_POST_PROCESSING,
                            filename+"_pred_post_locations.csv",
                        ),
                        index=False,
                    )

                pre_post_mapping = {}
                pres, posts = [], []
                if len(pre_points) > 0 and pre_points_df is not None and len(post_points) > 0 and post_points_df is not None:
                    pre_ids = pre_points_df["pre_id"].to_list()
                    if len(post_points) > 0 and post_points_df is not None:
                        post_ids = post_points_df["post_id"].to_list()
                    for i in range(len(pre_points)):
                        pre_post_mapping[pre_ids[i]] = []

                    # Match each post with a pre
                    distances = distance_matrix(post_points, pre_points)
                    for i in range(len(post_points)):
                        closest_pre_point = np.argmin(distances[i])
                        closest_pre_point = pre_ids[closest_pre_point]
                        pre_post_mapping[closest_pre_point].append(post_ids[i])

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
                        if len(pre_points) == 0:
                            print("No pre synaptic points found!")
                        if len(post_points) == 0:
                            print("No post synaptic points found!")

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
                        filename+"_pre_post_mapping.csv",
                    ),
                    index=False,
                )

            if self.cfg.DATA.TEST.LOAD_GT or self.cfg.DATA.TEST.USE_VAL_AS_TEST:
                print("Calculating synapse detection stats after post-processing . . .")
                # Calculate detection metrics
                if len(pre_points) > 0:
                    d_metrics, pre_gt_assoc, pre_fp = detection_metrics(
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
                    for n, item in enumerate(d_metrics.items()):
                        metric = self.test_extra_metrics[n]
                        if str(metric).lower() not in self.stats["merge_patches_post"]:
                            self.stats["merge_patches_post"][str(metric.lower())] = 0
                        self.stats["merge_patches_post"][str(metric).lower()] += item[1]
                        self.current_sample_metrics[str(metric).lower() + " (pre points, post-processing)"] = item[1]
                        
                    # Save csv files with the associations between GT points and predicted ones
                    pre_gt_assoc.to_csv(
                        os.path.join(
                            self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK_POST_PROCESSING,
                            filename+"_pred_pre_locations_gt_assoc.csv",
                        ),
                        index=False,
                    )
                    pre_fp.to_csv(
                        os.path.join(
                            self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK_POST_PROCESSING,
                            filename+"_pred_pre_locations_fp.csv",
                        ),
                        index=False,
                    )
                if len(post_points):
                    d_metrics, post_gt_assoc, post_fp = detection_metrics(
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
                    previous_pre_keys_num = len(d_metrics)
                    for n, item in enumerate(d_metrics.items()):
                        metric = self.test_extra_metrics[n+previous_pre_keys_num]
                        if str(metric).lower() not in self.stats["merge_patches_post"]:
                            self.stats["merge_patches_post"][str(metric.lower())] = 0
                        self.stats["merge_patches_post"][str(metric).lower()] += item[1]
                        self.current_sample_metrics[str(metric).lower() + " (post points, post-processing)"] = item[1]

                    # Save csv files with the associations between GT points and predicted ones
                    post_gt_assoc.to_csv(
                        os.path.join(
                            self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK_POST_PROCESSING,
                            filename+"_pred_post_locations_gt_assoc.csv",
                        ),
                        index=False,
                    )
                    post_fp.to_csv(
                        os.path.join(
                            self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK_POST_PROCESSING,
                            filename+"_pred_post_locations_fp.csv",
                        ),
                        index=False,
                    )

            if self.cfg.TEST.BY_CHUNKS.SAVE_OUT_TIF:
                print("Preparing prediction and GT tiffs as auxiliary images for checking the output. . .")
                sshape = list(self.current_sample["X"].shape)
                assert len(sshape) >= 3
                if len(sshape) == 3:
                    sshape += [
                        2,
                    ]
                else:
                    sshape[-1] = 2

                aux_tif = np.zeros(sshape, dtype=np.uint16)
                # Paint pre points
                if pre_points_df is not None:
                    pre_ids = pre_points_df["pre_id"].to_list()
                    assert len(pre_points) == len(pre_ids)
                    for j, cor in enumerate(pre_points):
                        z, y, x = cor  # type: ignore
                        z, y, x = int(z), int(y), int(x)
                        aux_tif[z, y, x, 0] = pre_ids[j]
                        aux_tif[z, y, x, 0] = pre_ids[j]

                # Paint post points
                if post_points_df is not None:
                    post_ids = post_points_df["post_id"].to_list()
                    assert len(post_points) == len(post_ids)
                    for j, cor in enumerate(post_points):
                        z, y, x = cor  # type: ignore
                        z, y, x = int(z), int(y), int(x)
                        aux_tif[z, y, x, 1] = post_ids[j]

                out_dir = (
                    self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK_POST_PROCESSING
                    if self.post_processing['per_image']
                    else self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK
                )
                if pre_points_df is not None or post_points_df is not None:
                    # Dilate and save the predicted points
                    for c in range(aux_tif.shape[-1]):
                        aux_tif[..., c] = dilation(aux_tif[..., c], ball(3))

                    save_tif(
                        np.expand_dims(aux_tif, 0),
                        out_dir,
                        [filename + "_points.tif"],
                        verbose=self.cfg.TEST.VERBOSE,
                    )

                aux_tif = np.zeros(sshape, dtype=np.uint16)
                for j, cor in enumerate(gt_pre_points):
                    z, y, x = cor  # type: ignore
                    z, y, x = int(z)-1, int(y)-1, int(x)-1
                    try:
                        aux_tif[z, y, x, 0] = j+1
                    except:
                        pass
                for j, cor in enumerate(gt_post_points):
                    z, y, x = cor  # type: ignore
                    z, y, x = int(z)-1, int(y)-1, int(x)-1
                    try:
                        aux_tif[z, y, x, 1] = j+1
                    except:
                        pass
                
                for c in range(aux_tif.shape[-1]):
                    aux_tif[..., c] = dilation(aux_tif[..., c], ball(3))
                    
                save_tif(
                    np.expand_dims(aux_tif, 0),
                    out_dir,
                    [filename + "_gt_ids.tif"],
                    verbose=self.cfg.TEST.VERBOSE,
                )        
                if (self.cfg.DATA.TEST.LOAD_GT or self.cfg.DATA.TEST.USE_VAL_AS_TEST):
                    if len(pre_points) > 0:   
                        print(
                            "Creating the image with a summary of detected points and false positives with colors (pre-points) . . ."
                        )
                        aux_tif = np.zeros(sshape[:-1] + [3,], dtype=np.uint8)
                        
                        print("Painting TPs and FNs (pre-points) . . .")
                        for j, cor in tqdm(enumerate(gt_pre_points), total=len(gt_pre_points)):
                            z, y, x = cor  # type: ignore
                            z, y, x = int(z)-1, int(y)-1, int(x)-1
                            tag = pre_gt_assoc[pre_gt_assoc["gt_id"]==j+1]["tag"].iloc[0]
                            color = (0, 255, 0) if tag == "TP" else (255, 0, 0)  # Green or red
                            try:
                                aux_tif[z, y, x] = color
                            except:
                                pass

                        print("Painting FPs (pre-points) . . .")
                        for index, row in tqdm(pre_fp.iterrows(), total=len(pre_fp)):
                            z,y,x = int(row['axis-0']), int(row['axis-1']), int(row['axis-2'])
                            try:
                                aux_tif[z, y, x] = (0,0,255) # Blue
                            except:
                                pass
                        
                        print("Dilating points (pre-points) . . .")
                        for c in range(aux_tif.shape[-1]):
                            aux_tif[..., c] = dilation(aux_tif[..., c], ball(3))

                        save_tif(
                            np.expand_dims(aux_tif, 0),
                            out_dir,
                            [filename + "_pre_point_assoc.tif"],
                            verbose=self.cfg.TEST.VERBOSE,
                        )   
                    if len(post_points) > 0:
                        print(
                            "Creating the image with a summary of detected points and false positives with colors (post-points) . . ."
                        )
                        aux_tif = np.zeros(sshape[:-1] + [3,], dtype=np.uint8)
                        
                        print("Painting TPs and FNs (post-points) . . .")
                        for j, cor in tqdm(enumerate(gt_post_points), total=len(gt_post_points)):
                            z, y, x = cor  # type: ignore
                            z, y, x = int(z)-1, int(y)-1, int(x)-1
                            tag = post_gt_assoc[post_gt_assoc["gt_id"]==j+1]["tag"].iloc[0]
                            color = (0, 255, 0) if tag == "TP" else (255, 0, 0)  # Green or red
                            try:
                                aux_tif[z, y, x] = color
                            except:
                                pass

                        print("Painting FPs (post-points) . . .")
                        for index, row in tqdm(post_fp.iterrows(), total=len(post_fp)):
                            z,y,x = int(row['axis-0']), int(row['axis-1']), int(row['axis-2'])
                            try:
                                aux_tif[z, y, x] = (0,0,255) # Blue
                            except:
                                pass
                            
                        print("Dilating points (post-points) . . .")
                        for c in range(aux_tif.shape[-1]):
                            aux_tif[..., c] = dilation(aux_tif[..., c], ball(3))

                        save_tif(
                            np.expand_dims(aux_tif, 0),
                            out_dir,
                            [filename + "_post_point_assoc.tif"],
                            verbose=self.cfg.TEST.VERBOSE,
                        )   
                        del aux_tif

    def after_full_image(self, pred: NDArray):
        """
        Execute steps needed after generating the prediction by supplying the entire image to the model.

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
        """Execute steps needed after predicting all images."""
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

            if self.cfg.PROBLEM.INSTANCE_SEG.TYPE == "regular":
                print("Instance segmentation specific metrics:")
            if self.cfg.TEST.MATCHING_STATS and (self.cfg.DATA.TEST.LOAD_GT or self.cfg.DATA.TEST.USE_VAL_AS_TEST):
                for i in range(len(self.cfg.TEST.MATCHING_STATS_THS)):
                    if self.cfg.PROBLEM.INSTANCE_SEG.TYPE == "regular":
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
        Create instance segmentation ground truth images to train the model based on the ground truth instances provided.

        They will be saved in a separate folder in the root path of the ground truth.
        """
        original_test_path, original_test_mask_path = None, None
        train_channel_mask_dir = self.cfg.DATA.TRAIN.INSTANCE_CHANNELS_MASK_DIR
        val_channel_mask_dir = self.cfg.DATA.VAL.INSTANCE_CHANNELS_MASK_DIR
        test_channel_mask_dir = self.cfg.DATA.TEST.INSTANCE_CHANNELS_MASK_DIR

        if not self.cfg.DATA.TEST.INPUT_ZARR_MULTIPLE_DATA:
            test_instance_mask_dir = self.cfg.DATA.TEST.GT_PATH
        else:
            test_instance_mask_dir = self.cfg.DATA.TEST.PATH            

        opts = []
        print("###########################")
        print("#  PREPARE INSTANCE DATA  #")
        print("###########################")

        # Create selected channels for train data
        if self.cfg.TRAIN.ENABLE or self.cfg.DATA.TEST.USE_VAL_AS_TEST:
            if not os.path.isdir(train_channel_mask_dir):
                # Barrier need as some of the threads may check the existence of the folder after it is created
                if is_dist_avail_and_initialized():
                    dist.barrier()
                print(
                    "You select to create {} channels from given instance labels and no file is detected in {} . "
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
            if not os.path.isdir(val_channel_mask_dir):
                # Barrier need as some of the threads may check the existence of the folder after it is created
                if is_dist_avail_and_initialized():
                    dist.barrier()
                print(
                    "You select to create {} channels from given instance labels and no file is detected in {} . "
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
            if not os.path.isdir(test_channel_mask_dir):
                # Barrier need as some of the threads may check the existence of the folder after it is created
                if is_dist_avail_and_initialized():
                    dist.barrier()
                print(
                    "You select to create {} channels from given instance labels and no file is detected in {} . "
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
