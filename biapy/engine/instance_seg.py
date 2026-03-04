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
    create_synapses_from_point_probs,
    extract_points_in_predictions,
    remove_close_points,
    remove_close_points_by_mask,
    Embedding_cluster,
    apply_label_refinement,
    extract_synapse_connectivity,
    collect_point_type_csv_files,
    extract_synful_synapses,
    connect_pre_post_synapse_points_by_distance,
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
from biapy.data.data_3D_manipulation import read_chunked_data, read_chunked_nested_data, ensure_3d_shape, load_synapse_gt_points
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

    def __init__(self, cfg, job_identifier, device, system_dict, args, **kwargs):
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
        super(Instance_Segmentation_Workflow, self).__init__(cfg, job_identifier, device, system_dict, args, **kwargs)

        self.original_train_input_mask_axes_order = self.cfg.DATA.TRAIN.INPUT_MASK_AXES_ORDER
        self.original_test_path, self.original_test_mask_path = self.prepare_instance_data()

        # Merging the image
        self.all_matching_stats_merge_patches = []
        self.all_matching_stats_merge_patches_post = []
        self.stats["inst_stats_merge_patches"] = None
        self.stats["inst_stats_merge_patches_post"] = None
        # Multi-head: instances + classification
        if self.separated_class_channel:
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
        if self.separated_class_channel:
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
        if self.separated_class_channel:
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
            self.test_gt_filenames = next(os_walk_clean(self.original_test_mask_path))[2]
            if len(self.test_gt_filenames) == 0:
                self.test_gt_filenames = next(os_walk_clean(self.original_test_mask_path))[1]

        # Specific instance segmentation post-processing
        self.post_processing["instance_post"] = False
        if self.cfg.PROBLEM.INSTANCE_SEG.TYPE == "regular":
            if (
                self.cfg.TEST.POST_PROCESSING.VORONOI_ON_MASK
                or self.cfg.TEST.POST_PROCESSING.INSTANCE_REFINEMENT.ENABLE
                or self.cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.ENABLE
                or self.cfg.TEST.POST_PROCESSING.REPARE_LARGE_BLOBS_SIZE != -1
            
            ):
                self.post_processing["instance_post"] = True
        else:  # synapses        
            if (
                self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.REMOVE_CLOSE_PRE_POINTS_RADIUS > 0 
                or self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.REMOVE_CLOSE_POST_POINTS_RADIUS > 0
            ):
                # The "instance_post" is related to matching metrics aftwerwards, so it is more related 
                # to the regular instance segmentation workflow than to the synapse detection one, where 
                # we have specific metrics for the synapse detection performance.
                self.post_processing["per_image"] = True

            self.synapse_method = ""
            if all(ch in self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS for ch in ["F_pre", "F_post"]) and len(self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS) == 2:
                self.synapse_method = "simpsyn"
            elif all(ch in self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS for ch in ["F_post", "Z", "V", "H"]) and len(self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS) == 4:
                self.synapse_method = "synful"
            elif all(ch in self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS for ch in ["F_cleft"]) and len(self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS) == 1:
                self.synapse_method = "cleft"
            elif all(ch in self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS for ch in ["F_post"]) and len(self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS) == 1:
                self.synapse_method = "F_post_only"
            else:
                raise ValueError("Unknown synapse prediction method for the given channels. Please check the documentation for more details.")

        self.instances_already_created = False

    def define_activations_and_channels(self):
        """
        Define the activations to be applied to the model output and the channels that the model will output.

        This function must define the following variables:

        self.model_output_channels : List of int
            Number of channels for each output head of the model. E.g. [3] for a model with one head outputting 3 channels, 
            [1, 5] for a model with two heads outputting 1 and 5 channels respectively, etc.

        self.model_output_channel_info : List of str
            Information about the output channels.

        self.separated_class_channel : bool
            Whether if we should expect a separated output channel for classification.

        self.head_activations : List of str
            Activations to be applied to the model output. Each dict will match an output channel of the model. "linear" and "ce_sigmoid"
            will not be applied. E.g. ["linear"] for a model with one head, ["linear", "sigmoid"] for a model with two heads, etc.
        """
        if self.cfg.PROBLEM.INSTANCE_SEG.CHANNELS_PER_HEAD_INFO != []:
            set_model_output_channels = False
            self.model_output_channels = []
            count = 0
            for head_channels in self.cfg.PROBLEM.INSTANCE_SEG.CHANNELS_PER_HEAD_INFO:
                self.model_output_channels.append(head_channels)
                self.model_output_channel_info.append("+".join(self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS[count:count+head_channels]))
                count += head_channels
        else:
            self.model_output_channels = [0]
            self.model_output_channel_info = [""]
            set_model_output_channels = True

        dst = self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS_EXTRA_OPTS[0]
        for i, channel in enumerate(self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS):
                if channel in ["B", "F", "P", "C", "T", "M", "F_pre", "F_post", "F_cleft"]:
                    self.head_activations.append("ce_sigmoid")
                    if set_model_output_channels:
                        self.model_output_channels[0] += 1
                        self.model_output_channel_info[0] += "+" + channel
                elif channel in ["Dc", "Dn", "D", "Z", "V", "H"]:
                    if self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS_LOSSES[i] not in ["mse", "l1", "mae"] or dst.get(channel, {}).get("act", "") == "sigmoid":
                        self.head_activations.append("ce_sigmoid")
                    else:
                        self.head_activations.append("linear")
                    if set_model_output_channels:
                        self.model_output_channels[0] += 1
                        self.model_output_channel_info[0] += "+" + channel
                elif channel == "Db":
                    val_type = dst.get(channel, {}).get("val_type", "norm")
                    if val_type == "discretize":
                        for i in range(11):  # Default 10 bins + background
                            self.head_activations.append("ce_softmax")
                            if set_model_output_channels:
                                self.model_output_channels[0] += 1
                                self.model_output_channel_info[0] += "+" + channel+"_bin{}".format(i)
                    elif self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS_LOSSES[i] not in ["mse", "l1", "mae"] or dst.get(channel, {}).get("act", "") == "sigmoid":
                        self.head_activations.append("ce_sigmoid")
                        if set_model_output_channels:
                            self.model_output_channels[0] += 1
                            self.model_output_channel_info[0] += "+" + channel
                    else:
                        self.head_activations.append("linear")
                        if set_model_output_channels:
                            self.model_output_channels[0] += 1
                            self.model_output_channel_info[0] += "+" + channel
                elif channel == "D":
                    self.head_activations.append(dst.get("D", {}).get("act", "linear"))
                    if set_model_output_channels:
                        self.model_output_channels[0] += 1
                        self.model_output_channel_info[0] += "+" + channel
                elif channel == "A":
                    for i in range(len(dst.get("A", {}).get("z_affinities", [1]))):
                        if set_model_output_channels:
                            self.model_output_channels[0] += 1
                            self.model_output_channel_info[0] += "+" + channel+"_{}".format(i)
                        self.head_activations.append("ce_sigmoid")
                    for i in range(len(dst.get("A", {}).get("y_affinities", [1]))):
                        if set_model_output_channels:
                            self.model_output_channels[0] += 1
                            self.model_output_channel_info[0] += "+" + channel+"_{}".format(i)
                        self.head_activations.append("ce_sigmoid")
                    for i in range(len(dst.get("A", {}).get("x_affinities", [1]))):
                        if set_model_output_channels:
                            self.model_output_channels[0] += 1
                            self.model_output_channel_info[0] += "+" + channel+"_{}".format(i)
                        self.head_activations.append("ce_sigmoid")
                elif channel == "R":
                    for i in range(dst.get("R", {}).get("nrays", 32 if self.dims == 2 else 96)):
                        self.head_activations.append("linear")
                        if set_model_output_channels:
                            self.model_output_channels[0] += 1
                            self.model_output_channel_info[0] += "+" + channel+"_{}".format(i)
                elif channel == "E_offset":
                    for i in range(self.dims):
                        self.head_activations.append("ce_sigmoid")
                        if set_model_output_channels:
                            self.model_output_channels[0] += 1
                            self.model_output_channel_info[0] += "+" + channel+"_{}".format(i)
                elif channel == "E_sigma":
                    for i in range(self.dims):
                        self.head_activations.append("ce_sigmoid")
                        if set_model_output_channels:
                            self.model_output_channels[0] += 1
                            self.model_output_channel_info[0] += "+" + channel+"_{}".format(i)
                elif channel == "E_seediness":
                    self.head_activations.append("ce_sigmoid")
                    if set_model_output_channels:
                        self.model_output_channels[0] += 1
                        self.model_output_channel_info[0] += "+" + channel
                elif channel == "We":
                    continue
                else:
                    raise ValueError("Unknown channel: {}".format(channel))

        for i in range(len(self.model_output_channel_info)):
            self.model_output_channel_info[i] = self.model_output_channel_info[i].lstrip("+")

        # Multi-head: instances + classification
        self.gt_channels_expected = len(self.head_activations)
        if self.cfg.DATA.N_CLASSES > 2:
            self.head_activations += ["ce_softmax"] * self.cfg.DATA.N_CLASSES
            self.model_output_channels += [self.cfg.DATA.N_CLASSES,]
            self.model_output_channel_info += ["class"]
            self.gt_channels_expected += 1
            self.separated_class_channel = True
        else:
            self.separated_class_channel = False

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
            elif channel in ["Db", "Dc", "Dn", "D", "Z", "V", "H", "R"]:
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
            elif channel == "F_cleft":
                self.train_metric_names += ["IoU (clefts)"]
                self.train_metric_best += ["max"]
            else:
                raise ValueError("Unknown channel: {}".format(channel))
        
        # Multi-head: instances + classification
        if self.cfg.PROBLEM.INSTANCE_SEG.TYPE == "regular" and self.separated_class_channel:
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
        if self.separated_class_channel:
            self.test_metric_names.append("IoU (classes)")
            # Used to calculate IoU with the classification results
            self.jaccard_index_matching = jaccard_index(
                device=self.test_device, 
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
                    device=self.test_device,
                    out_channels=self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS,
                    channel_extra_opts = self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS_EXTRA_OPTS[0],
                    model_source=self.cfg.MODEL.SOURCE,
                    ndim=self.dims,
                    ignore_index=self.cfg.LOSS.IGNORE_INDEX,
                )
            )
        
        if self.cfg.PROBLEM.INSTANCE_SEG.TYPE == "synapses":
            self.test_extra_metrics = []
            for x in ["pre", "post", "cleft"]:
                self.test_extra_metrics.append(f"Precision ({x}-points)")
                self.test_extra_metrics.append(f"Recall ({x}-points)")
                self.test_extra_metrics.append(f"F1 ({x}-points)")
                self.test_extra_metrics.append(f"TP ({x}-points)")
                self.test_extra_metrics.append(f"FP ({x}-points)")
                self.test_extra_metrics.append(f"FN ({x}-points)")
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
            self.embedding_cluster = Embedding_cluster(
                device=self.test_device,
                patch_size=self.cfg.DATA.PATCH_SIZE,
                ndims=self.dims,
                anisotropy=self.resolution,
            )
        else:
            instance_loss = instance_segmentation_loss(
                weights = self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNEL_WEIGHTS,
                ndim = self.dims,
                out_channels = self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS,
                losses_to_use = self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS_LOSSES,
                channel_extra_opts = self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS_EXTRA_OPTS[0],
                gt_channels_expected = self.gt_channels_expected,
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

        with torch.no_grad():
            k = 0
            for i, metric in enumerate(list_to_use):
                if metric == "none":
                    continue
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
                    if isinstance(val, torch.Tensor):
                        v = val.item() if not torch.isnan(val) else 0
                    else:
                        v = val
                    out_metrics[list_names_to_use[i]] = v
                    if metric_logger:
                        metric_logger.meters[list_names_to_use[i]].update(v)
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
            if self.separated_class_channel:
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
                pred_labels = self.embedding_cluster.create_instances(
                    pred=pred if self.dims == 3 else pred[0],
                    fg_thresh=self.cfg.PROBLEM.INSTANCE_SEG.EMBEDSEG.SEED_THRESH,
                    min_mask_sum=self.cfg.PROBLEM.INSTANCE_SEG.EMBEDSEG.MIN_MASK_SUM,
                    min_unclustered_sum=self.cfg.PROBLEM.INSTANCE_SEG.EMBEDSEG.MIN_UNCLUSTERED_SUM,
                    min_object_size=self.cfg.PROBLEM.INSTANCE_SEG.EMBEDSEG.MIN_OBJECT_SIZE
                )
                if self.dims == 2:
                    pred_labels = np.expand_dims(pred_labels, 0)
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
            if self.separated_class_channel:
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
            if self.separated_class_channel:
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
                    torch.as_tensor(class_channel.squeeze().astype(np.uint8)).to(self.test_device, non_blocking=True),
                    torch.as_tensor(_Y_classes.squeeze().astype(np.uint8)).to(self.test_device, non_blocking=True),
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
        if self.cfg.TEST.POST_PROCESSING.INSTANCE_REFINEMENT.ENABLE:
            pred_labels = apply_label_refinement(
                pred_labels, 
                is_3d=self.cfg.PROBLEM.NDIM=="3D",
                operations=self.cfg.TEST.POST_PROCESSING.INSTANCE_REFINEMENT.OPERATIONS, 
                values=self.cfg.TEST.POST_PROCESSING.INSTANCE_REFINEMENT.VALUES, 
            )

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
                if "C" in self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS:
                    pred = pred[...,self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS.index("F")] + pred[...,self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS.index("C")]
                else:
                    pred = pred[...,self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS.index("F")]
            elif "B" in self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS:
                pred = 1 - pred[...,self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS.index("B")]    
            elif "C" in self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS:
                pred = pred[...,self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS.index("C")]
                erode_size = 2 # As the contours are thicker we erode a little bit

            pred_labels = voronoi_on_mask(
                pred_labels,
                pred,
                th=self.cfg.TEST.POST_PROCESSING.VORONOI_TH,
                verbose=self.cfg.TEST.VERBOSE,
                erode_size=erode_size,
            )
        del pred

        if (
            self.cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.ENABLE
            or self.cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.ENABLE
        ):
            if self.cfg.PROBLEM.NDIM == "2D":
                pred_labels = pred_labels[0]
            pred_labels, d_result = measure_morphological_props_and_filter(
                pred_labels,
                intensity_image=self.current_sample["X"][0],
                resolution=self.resolution,
                extra_props=self.cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.EXTRA_PROPS,
                filter_instances=self.cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.ENABLE,
                properties=self.cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS,
                prop_values=self.cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.VALUES,
                comp_signs=self.cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.SIGNS,
            )
            extra_properties_keys = self.cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.EXTRA_PROPS
            if self.cfg.PROBLEM.NDIM == "2D":
                pred_labels = np.expand_dims(pred_labels, 0)

            # Save all instance stats
            if self.cfg.PROBLEM.NDIM == "2D":
                # Base properties that are always included
                base_data_series = [
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
                ]

                # Base column names
                base_columns = [
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
                ]
            else:
                base_data_series = [
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
                ]
                base_columns =[
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
                ]
            extra_properties_keys = [key for key in extra_properties_keys if key not in base_columns and key in d_result]
            extra_data_series = [d_result[key] for key in extra_properties_keys if key in d_result]
            all_data_series = base_data_series + extra_data_series
            all_columns = base_columns + extra_properties_keys

            df = pd.DataFrame(
                zip(*all_data_series),
                columns=all_columns,
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
            if self.separated_class_channel:
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
                if self.separated_class_channel:
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
        calculate_metrics: bool = True,
        do_post_processing: bool = True,
    ) -> Dict:
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
            Whether to calculate or not the metrics. Normally we disable it when doring inference per chunks
            as the metrics are calculated at the end on the whole image.

        do_post_processing : bool
            Whether to do or not the post-processing step. Normally we disable it when doring inference per chunks
            as the post-processing is done at the end on the whole image.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the predicted synapse-related points.
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

        points_available = {}
        if self.synapse_method == "synful":
            pre_points_df, pre_points, post_points_df, post_points = extract_synful_synapses(
                data=pred,
                channels=self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS,
                threshold_abs=0.2,
                min_distance=1,
                cluster_distance=5.0,
                out_dir=out_dir,
                verbose=self.cfg.TEST.VERBOSE,
            )
            points_available["pre"] = {"points": pre_points, "df": pre_points_df}
            points_available["post"] = {"points": post_points, "df": post_points_df}
        elif self.synapse_method == "simpsyn":
            pre_points_df, pre_points, post_points_df, post_points = create_synapses_from_point_probs(
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
                out_dir=out_dir,
                filenames = filenames,
                verbose=self.cfg.TEST.VERBOSE,
            )
            points_available["pre"] = {"points": pre_points, "df": pre_points_df}
            points_available["post"] = {"points": post_points, "df": post_points_df}
        elif self.synapse_method == "cleft":
            cleft_points_df, cleft_points = extract_points_in_predictions(
                data=pred[...,0],
                point_type="cleft",
                point_creation_func=self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.POINT_CREATION_FUNCTION,
                min_th_to_be_peak=threshold_abs[0],
                min_distance=self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.PEAK_LOCAL_MAX_MIN_DISTANCE,
                min_sigma=self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.BLOB_LOG_MIN_SIGMA,
                max_sigma=self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.BLOB_LOG_MAX_SIGMA,
                num_sigma=self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.BLOB_LOG_NUM_SIGMA,
                exclude_border=self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.EXCLUDE_BORDER,
                relative_th_value=self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.TH_TYPE in ["relative", "relative_by_patch"], 
                out_dir=out_dir,
                filenames = filenames,
                verbose=self.cfg.TEST.VERBOSE,
            )
            points_available["cleft"] = {"points": cleft_points, "df": cleft_points_df}
        elif self.synapse_method == "F_post_only":
            post_points_df, post_points = extract_points_in_predictions(
                data=pred[...,0],
                point_type="post",
                point_creation_func=self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.POINT_CREATION_FUNCTION,
                min_th_to_be_peak=threshold_abs[0],
                min_distance=self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.PEAK_LOCAL_MAX_MIN_DISTANCE,
                min_sigma=self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.BLOB_LOG_MIN_SIGMA,
                max_sigma=self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.BLOB_LOG_MAX_SIGMA,
                num_sigma=self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.BLOB_LOG_NUM_SIGMA,
                exclude_border=self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.EXCLUDE_BORDER,
                relative_th_value=self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.TH_TYPE in ["relative", "relative_by_patch"], 
                out_dir=out_dir,
                filenames = filenames,
                verbose=self.cfg.TEST.VERBOSE,
            )
            points_available["post"] = {"points": post_points, "df": post_points_df}
        else:
            raise ValueError(f"Synapse method {self.synapse_method} not recognized.")

        if calculate_metrics and self.cfg.DATA.TEST.LOAD_GT or self.cfg.DATA.TEST.USE_VAL_AS_TEST:
            print("Calculating synapse detection stats . . .")
            gt_info = load_synapse_gt_points(
                locations_path = self.cfg.DATA.TEST.INPUT_ZARR_MULTIPLE_DATA_LOCATIONS_PATH,
                resolution_path = self.cfg.DATA.TEST.INPUT_ZARR_MULTIPLE_DATA_RESOLUTION_PATH,
                partners_path = self.cfg.DATA.TEST.INPUT_ZARR_MULTIPLE_DATA_PARTNERS_PATH,
                id_path = self.cfg.DATA.TEST.INPUT_ZARR_MULTIPLE_DATA_ID_PATH,
                data_filename = os.path.join(self.current_sample["X_dir"], self.current_sample["X_filename"]),
            )
            assert out_dir is not None, "Output directory must be provided to save the synapse detection metrics results."

            # Calculate detection metrics for each type of points if they are available
            for key in points_available:
                if key not in ["pre", "post", "cleft"]:
                    raise ValueError(f"Unknown point type {key} found in points_available. Expected 'pre', 'post' or 'cleft'.")
                points_available[key]["gt"] = gt_info[key]

                assert "points" in points_available[key], f"Points not found for key {key} in points_available. Found keys: {points_available[key].keys()}"
                assert "gt" in points_available[key], f"GT not found for key {key} in points_available. Found keys: {points_available[key].keys()}"
                points_available[key]["gt_assoc"], points_available[key]["fps"] = self.calculate_synapse_det_metrics_on_points(
                    points_available[key]["gt"], 
                    points_available[key]["points"], 
                    gt_info["resolution"], 
                    self.current_sample["X_filename"], 
                    out_dir, 
                    point_type=key
                )

        ###################
        # Post-processing #
        ###################
        if do_post_processing and self.post_processing["per_image"]:
            print("TODO: post-processing")

        return points_available

    def calculate_synapse_det_metrics_on_points(self, 
        gt_points: NDArray | List[int], 
        pred_points: NDArray, 
        resolution: List[int | float], 
        filename: str, 
        out_dir: str, 
        point_type: str ="pre", 
        post_processing: bool=False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate synapse detection metrics on the predicted points and save the associations between GT points and predicted ones.

        Parameters
        ----------
        gt_points : np.array or list of int
            Ground truth synapse points.

        pred_points : np.array
            Predicted synapse points.

        resolution : list
            Image resolution.

        filename : str
            Filename of the predicted image.

        out_dir : str
            Output directory to save the csv files with the associations between GT points and predicted ones.

        point_type : str
            Type of synaptic point to calculate the metrics on. E.g. "pre" or "post".
        
        post_processing : bool
            Whether the predicted points are from the post-processing step or not. Used for printing and saving the results.

        Returns
        -------
        gt_assoc : pd.DataFrame
            DataFrame with the associations between GT points and predicted ones.
        
        fps : pd.DataFrame
            DataFrame with the false positive predicted points.
        """
        d_metrics, gt_assoc, fps = detection_metrics(
            true_points=gt_points,
            pred_points=pred_points,
            true_classes=None,
            pred_classes=[],
            tolerance=self.cfg.TEST.DET_TOLERANCE,
            resolution=resolution,
            bbox_to_consider=[],
            verbose=True,
        )
        point_metrics = [x for x in self.test_extra_metrics if point_type in str(x).lower()]
        stat_key = "merge_patches" if not post_processing else "merge_patches_post"
        print("Synapse detection ({} points) metrics{}: {}".format(point_type, " (post-processing)" if post_processing else "", d_metrics))
        for n, item in enumerate(d_metrics.items()):
            metric = point_metrics[n]
            if str(metric).lower() not in self.stats[stat_key]:
                self.stats[stat_key][str(metric.lower())] = 0
            self.stats[stat_key][str(metric).lower()] += item[1]
            self.current_sample_metrics[str(metric).lower() + f" ({point_type} points{(', post-processing' if post_processing else '')})"] = item[1]

        # Save csv files with the associations between GT points and predicted ones
        gt_assoc.to_csv(
            os.path.join(
                out_dir,
                filename+f"_pred_{point_type}_locations_gt_assoc.csv",
            ),
            index=False,
        )
        fps.to_csv(
            os.path.join(
                out_dir,
                filename+f"_pred_{point_type}_locations_fp.csv",
            ),
            index=False,
        )
        return gt_assoc, fps

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
                        [self.current_sample["X_filename"]],
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
                        [self.current_sample["X_filename"]],
                        self.cfg.PATHS.RESULT_DIR.PER_IMAGE_INSTANCES,
                        self.cfg.PATHS.RESULT_DIR.PER_IMAGE_POST_PROCESSING,
                    )
        else:
            raise NotImplementedError

    def after_one_chunk_raw_prediction(
        self, chunk_id: int, chunk: NDArray, chunk_in_data: PatchCoords, added_pad: List[List[int]]
    ):
        """
        Place any code that needs to be done after predicting one chunk of data in "by chunks" setting.

        Parameters
        ----------
        chunk_id: int
            Chunk identifier.

        chunk : NDArray
            Predicted chunk

        chunk_in_data : PatchCoords
            Global coordinates of the chunk.
        
        added_pad: List of list of ints
            Padding added to the chunk in each dimension. The order of dimensions is the same as the input 
            image, and the order of the list is: [[pad_before_dim1, pad_after_dim1], [pad_before_dim2, pad_after_dim2], .... 
        """
        if self.cfg.PROBLEM.INSTANCE_SEG.TYPE == "regular":
            pass
            # Important to maintain calculate_metrics=False in the future call here
            # pre_points_df, post_points_df = self.instance_seg_process(chunk, filenames, out_dir, out_dir_post_proc, calculate_metrics=False)
        else:  # synapses
            if self.synapse_method == "synful":
                return

            # "simpsyn", "cleft" or "F_post_only"
            points_available = self.synapse_seg_process(chunk, calculate_metrics=False, do_post_processing=False)
            _filename, _ = os.path.splitext(os.path.basename(self.current_sample["X_filename"]))

            npatches = len(str(len(self.test_generator)))
            for key in points_available:
                assert key in ["pre", "post", "cleft"], f"Unknown point type {key} found in points_available. Expected 'pre', 'post' or 'cleft'."
                assert "df" in points_available[key], f"'df' key not found for {key} in points_available. Found keys: {points_available[key].keys()}"
                point_df = points_available[key]["df"]
                # Remove possible points in the padded area
                point_df = point_df[point_df["axis-0"] < chunk.shape[0] - added_pad[0][1]]
                point_df = point_df[point_df["axis-1"] < chunk.shape[1] - added_pad[1][1]]
                point_df = point_df[point_df["axis-2"] < chunk.shape[2] - added_pad[2][1]]
                point_df["axis-0"] = point_df["axis-0"] - added_pad[0][0]
                point_df["axis-1"] = point_df["axis-1"] - added_pad[1][0]
                point_df["axis-2"] = point_df["axis-2"] - added_pad[2][0]
                point_df = point_df[point_df["axis-0"] >= 0]
                point_df = point_df[point_df["axis-1"] >= 0]
                point_df = point_df[point_df["axis-2"] >= 0]
                
                # Add the chunk shift to the detected coordinates so they represent global coords
                point_df["axis-0"] = point_df["axis-0"] + chunk_in_data.z_start
                point_df["axis-1"] = point_df["axis-1"] + chunk_in_data.y_start
                point_df["axis-2"] = point_df["axis-2"] + chunk_in_data.x_start

                # Save the csv file
                os.makedirs(self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK, exist_ok=True)
                point_df.to_csv(
                    os.path.join(
                        self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK,
                        _filename + "_patch" + str(chunk_id).zfill(npatches) + "_" + key + "_points.csv",
                    ),
                    index=False,
                )

    def after_all_chunk_prediction_workflow_process(self):
        """
        Place any code that needs to be done after predicting all patches in "by chunks" setting.
        This function is called on all ranks.
        """
        if self.cfg.PROBLEM.INSTANCE_SEG.TYPE == "regular":
            if self.cfg.TEST.BY_CHUNKS.WORKFLOW_PROCESS.TYPE == "chunk_by_chunk":
                raise NotImplementedError
            else: 
                pass
        else: # synapses
            pass

    def after_all_chunk_prediction_workflow_process_master_rank(self):
        """Execute steps needed after merging all predicted patches into the original image in "by chunks" setting."""
        assert isinstance(self.all_pred, list) and isinstance(self.all_gt, list)
        filename = os.path.basename(self.current_sample["X_filename"])

        points_available = {}
        if self.cfg.PROBLEM.INSTANCE_SEG.TYPE == "regular":
            if self.cfg.TEST.BY_CHUNKS.WORKFLOW_PROCESS.TYPE == "chunk_by_chunk":
                raise NotImplementedError
            else:
                # Load H5/Zarr and convert it into numpy array
                fpath = os.path.join(
                    self.cfg.PATHS.RESULT_DIR.PER_IMAGE, os.path.splitext(filename)[0] + ".zarr"
                )
                pred_file, pred = read_chunked_data(fpath)
                pred = np.squeeze(np.array(pred, dtype=self.dtype))
                if isinstance(pred_file, h5py.File):
                    pred_file.close()

                pred = ensure_3d_shape(pred, fpath)

                self.after_merge_patches(np.expand_dims(pred, 0))
        else:
            if self.synapse_method == "synful":
                if self.cfg.TEST.BY_CHUNKS.WORKFLOW_PROCESS.TYPE == "chunk_by_chunk":
                    raise NotImplementedError
                else:
                    # Load H5/Zarr and convert it into numpy array
                    fpath = os.path.join(
                        self.cfg.PATHS.RESULT_DIR.PER_IMAGE, os.path.splitext(filename)[0] + ".zarr"
                    )
                    pred_file, pred = read_chunked_data(fpath)
                    pred = np.squeeze(np.array(pred, dtype=self.dtype))
                    if isinstance(pred_file, h5py.File):
                        pred_file.close()

                    pred = ensure_3d_shape(pred, fpath)

                    self.after_merge_patches(np.expand_dims(pred, 0))
                    print("TODO: synful support")
                    return 
            elif self.synapse_method in ["F_post_only", "cleft"]:
                p_type = "post" if self.synapse_method == "F_post_only" else "cleft"
                point_info = collect_point_type_csv_files(
                    filename=os.path.splitext(filename)[0],
                    point_type=p_type,
                    csv_dir=self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK,
                    min_th_to_be_peak=self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.MIN_TH_TO_BE_PEAK,
                    th_type=self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.TH_TYPE,
                )
                points = []
                if isinstance(point_info["df"], pd.DataFrame) and len(point_info["df"]) > 0:
                    for coord in zip(point_info["df"]["axis-0"], point_info["df"]["axis-1"], point_info["df"]["axis-2"]):
                        points.append(list(coord))
                points_available[p_type] = {
                    "points": points,
                    "df": point_info["df"]
                }
            elif self.synapse_method == "simpsyn":
                pre_points_df, pre_points, pre_th_global, post_points_df, post_points, post_th_global = extract_synapse_connectivity(
                    filename=os.path.splitext(filename)[0],
                    reuse_predictions=self.cfg.TEST.REUSE_PREDICTIONS,
                    csv_dir=self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK,
                    min_th_to_be_peak=self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.MIN_TH_TO_BE_PEAK,
                    th_type=self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.TH_TYPE,
                    verbose=self.cfg.TEST.VERBOSE,
                )
                points_available["pre"] = {"points": pre_points, "df": pre_points_df}
                points_available["post"] = {"points": post_points, "df": post_points_df}

            # Calculate synapse detection metrics if GT is available
            if self.cfg.DATA.TEST.LOAD_GT or self.cfg.DATA.TEST.USE_VAL_AS_TEST:
                print("Calculating synapse detection stats . . .")
                gt_info = load_synapse_gt_points(
                    locations_path = self.cfg.DATA.TEST.INPUT_ZARR_MULTIPLE_DATA_LOCATIONS_PATH,
                    resolution_path = self.cfg.DATA.TEST.INPUT_ZARR_MULTIPLE_DATA_RESOLUTION_PATH,
                    partners_path = self.cfg.DATA.TEST.INPUT_ZARR_MULTIPLE_DATA_PARTNERS_PATH,
                    id_path = self.cfg.DATA.TEST.INPUT_ZARR_MULTIPLE_DATA_ID_PATH,
                    data_filename = os.path.join(self.current_sample["X_dir"], filename),
                )

                # Calculate detection metrics for each type of points if they are available
                for key in points_available:
                    if key not in ["pre", "post", "cleft"]:
                        raise ValueError(f"Unknown point type {key} found in points_available. Expected 'pre', 'post' or 'cleft'.")
                    points_available[key]["gt"] = gt_info[key]

                    assert "points" in points_available[key], f"Points not found for key {key} in points_available. Found keys: {points_available[key].keys()}"
                    assert "gt" in points_available[key], f"GT not found for key {key} in points_available. Found keys: {points_available[key].keys()}"
                    points_available[key]["gt_assoc"], points_available[key]["fps"] = self.calculate_synapse_det_metrics_on_points(
                        points_available[key]["gt"], 
                        points_available[key]["points"], 
                        gt_info["resolution"], 
                        filename, 
                        self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK, 
                        point_type=key
                    )

            # Post-processing: remove points that are too close to each other based on a radius threshold defined in the config. 
            for key in points_available:
                if key not in ["pre", "post", "cleft"]:
                    raise ValueError(f"Unknown point type {key} found in points_available. Expected 'pre', 'post' or 'cleft'.")
                assert "points" in points_available[key], f"Points not found for key {key} in points_available. Found keys: {points_available[key].keys()}"
                assert "df" in points_available[key], f"Dataframe not found for key {key} in points_available. Found keys: {points_available[key].keys()}"
                if self.post_processing["per_image"]:
                    if self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.REMOVE_CLOSE_PRE_POINTS_RADIUS > 0:
                        if key == "pre" :
                            pradius = self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.REMOVE_CLOSE_PRE_POINTS_RADIUS
                            th_global = pre_th_global
                        else:
                            pradius = self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.REMOVE_CLOSE_POST_POINTS_RADIUS
                            th_global = post_th_global
                        if self.cfg.PROBLEM.INSTANCE_SEG.SYNAPSES.REMOVE_CLOSE_POINTS_RADIUS_BY_MASK:    
                            # Load H5/Zarr and convert it into numpy array
                            fpath = os.path.join(
                                self.cfg.PATHS.RESULT_DIR.PER_IMAGE, os.path.splitext(filename)[0] + ".zarr"
                            )
                            pred_file, pred = read_chunked_data(fpath) 

                            points_available[key]["points"], pre_dropped_pos = remove_close_points_by_mask(  # type: ignore
                                points=points_available[key]["points"],
                                radius=pradius,
                                raw_predictions=pred,
                                bin_th=th_global,
                                resolution=gt_info["resolution"],
                                channel_to_look_into=1, # post channel
                                ndim=self.dims,
                                return_drops=True,
                            )

                            if isinstance(pred_file, h5py.File):
                                pred_file.close()
                        else:
                            points_available[key]["points"], pre_dropped_pos = remove_close_points(  # type: ignore
                                points_available[key]["points"],
                                pradius,
                                gt_info["resolution"],
                                ndim=self.dims,
                                return_drops=True,
                            )
                        points_available[key]["df"].drop(points_available[key]["df"].index[pre_dropped_pos], inplace=True)  # type: ignore
                        os.makedirs(self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK_POST_PROCESSING, exist_ok=True)
                        points_available[key]["df"].to_csv(
                            os.path.join(
                                self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK_POST_PROCESSING,
                                str(filename)+"_pred_"+key+"_locations.csv",
                            ),
                            index=False,
                        )

                    # After removing close points, calculate again the detection metrics to see the effect of this post-processing step on the metrics.
                    if self.cfg.DATA.TEST.LOAD_GT or self.cfg.DATA.TEST.USE_VAL_AS_TEST:
                        points_available[key]["gt_assoc"], points_available[key]["fps"] = self.calculate_synapse_det_metrics_on_points(
                            points_available[key]["gt"], 
                            points_available[key]["points"], 
                            gt_info["resolution"], 
                            filename, 
                            self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK_POST_PROCESSING, 
                            point_type=key, 
                            post_processing=True
                            )

            # If both pre and post points are available and the post-processing step was done we need to connect again the points to create the
            # synapse connectivity.
            if self.post_processing["per_image"] and "pre" in points_available and "post" in points_available:
                connect_pre_post_synapse_points_by_distance(
                    pre_points_df=points_available["pre"]["df"],
                    pre_points=points_available["pre"]["points"],
                    post_points_df=points_available["post"]["df"],
                    post_points=points_available["post"]["points"],
                    out_dir=self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK_POST_PROCESSING,
                    verbose=self.cfg.TEST.VERBOSE,
                )
                
            if self.cfg.TEST.BY_CHUNKS.SAVE_OUT_TIF:
                print("Preparing prediction and GT tiffs as auxiliary images for checking the output. . .")
                sshape = list(self.current_sample["X"].shape)
                channels = len(points_available) if len(points_available) > 0 else 1
                assert len(sshape) >= 3
                if len(sshape) == 3:
                    sshape += [channels]
                else:
                    sshape[-1] = channels

                # Create a tif with the predicted points, coloring them based on their ID in the dataframe (if available) and dilating them to make them more visible. 
                # We create one channel per type of point (pre, post, cleft).
                aux_tif = np.zeros(sshape, dtype=np.uint16)
                for i, key in enumerate(points_available):
                    point_df = points_available[key]["df"]
                    # Paint points
                    if point_df is not None:
                        point_ids = point_df[f"{key}_id"].to_list() 
                        assert len(points_available[key]["points"]) == len(point_ids)
                        for j, cor in enumerate(points_available[key]["points"]):
                            z, y, x = int(cor[0]), int(cor[1]), int(cor[2])
                            aux_tif[z, y, x, i] = point_ids[j]
                            aux_tif[z, y, x, i] = point_ids[j]

                        # Dilate points to make them more visible in the tif
                        aux_tif[..., i] = dilation(aux_tif[..., i], ball(3))

                out_dir = (
                    self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK_POST_PROCESSING
                    if self.post_processing["per_image"]
                    else self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK
                )
                save_tif(
                    np.expand_dims(aux_tif, 0),
                    out_dir,
                    [str(filename) + "_points.tif"],
                    verbose=self.cfg.TEST.VERBOSE,
                )

                # Create another tif with the GT points, coloring them based on their ID in the dataframe (if available) and dilating them to make them more visible.
                aux_tif = np.zeros(sshape, dtype=np.uint16)
                for i, key in enumerate(points_available):
                    points = points_available[key]["gt"]
                    if len(points) > 0:
                        for j, coord in enumerate(points):
                            z, y, x = int(coord[0])-1, int(coord[1])-1, int(coord[2])-1
                            aux_tif[z, y, x, i] = j+1
                    aux_tif[..., i] = dilation(aux_tif[..., i], ball(3))
                    
                save_tif(
                    np.expand_dims(aux_tif, 0),
                    out_dir,
                    [str(filename) + "_gt_ids.tif"],
                    verbose=self.cfg.TEST.VERBOSE,
                )        

                # Create another tif with the predicted points colored in green if they are TP and in red if they are FP, and with the GT points colored in blue. 
                # This is useful to visually check the quality of the predictions and the errors. We do this only if GT is available, otherwise we don't know which 
                # predicted points are TP or FP. We create one image per type of point (pre, post, cleft).
                if (self.cfg.DATA.TEST.LOAD_GT or self.cfg.DATA.TEST.USE_VAL_AS_TEST):
                    for i, key in enumerate(points_available):
                        if key not in ["pre", "post", "cleft"]:
                            raise ValueError(f"Unknown point type {key} found in points_available. Expected 'pre', 'post' or 'cleft'.")
                        assert "gt" in points_available[key], f"GT not found for key {key} in points_available. Found keys: {points_available[key].keys()}"
                        assert "gt_assoc" in points_available[key], f"GT association not found for key {key} in points_available. Found keys: {points_available[key].keys()}"
                        assert "fps" in points_available[key], f"FPs not found for key {key} in points_available. Found keys: {points_available[key].keys()}"
                        aux_tif = np.zeros(sshape[:-1] + [3,], dtype=np.uint8)    
                        print(f"Creating the image with a summary of detected points and false positives with colors ({key}-points) . . .")
                        
                        print(f"Painting TPs and FNs ({key}-points) . . .")
                        for j, cor in tqdm(enumerate(points_available[key]["gt"]), total=len(points_available[key]["gt"])):
                            z, y, x = int(cor[0])-1, int(cor[1])-1, int(cor[2])-1
                            tag = points_available[key]["gt_assoc"][points_available[key]["gt_assoc"]["gt_id"]==j+1]["tag"].iloc[0]
                            color = (0, 255, 0) if tag == "TP" else (255, 0, 0)  # Green or red
                            try:
                                aux_tif[z, y, x] = color
                            except:
                                pass

                        print(f"Painting FPs ({key}-points) . . .")
                        for index, row in tqdm(points_available[key]["fps"].iterrows(), total=len(points_available[key]["fps"])):
                            z,y,x = int(row['axis-0']), int(row['axis-1']), int(row['axis-2'])
                            try:
                                aux_tif[z, y, x] = (0,0,255) # Blue
                            except:
                                pass
                        
                        print(f"Dilating points ({key}-points) . . .")
                        for c in range(aux_tif.shape[-1]):
                            aux_tif[..., c] = dilation(aux_tif[..., c], ball(3))

                        save_tif(
                            np.expand_dims(aux_tif, 0),
                            out_dir,
                            [str(filename) + f"_{key}_point_assoc.tif"],
                            verbose=self.cfg.TEST.VERBOSE,
                        )   

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
                        [self.current_sample["X_filename"]],
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
                        [self.current_sample["X_filename"]],
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
                if self.separated_class_channel:
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
                        if self.separated_class_channel:
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
        filename, file_extension = os.path.splitext(self.current_sample["X_filename"])

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
                [self.current_sample["X_filename"]],
                verbose=self.cfg.TEST.VERBOSE,
            )

        return None
