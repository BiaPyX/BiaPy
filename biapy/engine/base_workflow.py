import math
import os
import datetime
import time
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from abc import ABCMeta, abstractmethod
import torch.distributed as dist
from typing import Dict, Optional, List
from numpy.typing import NDArray
from yacs.config import CfgNode as CN
import pandas as pd

import biapy
from bioimageio.core import create_prediction_pipeline
from bioimageio.spec import load_description

from biapy.config.config import update_dependencies
from biapy.models import (
    build_model,
    build_torchvision_model,
    build_bmz_model,
    check_bmz_args,
    check_model_restrictions,
)
from biapy.engine import prepare_optimizer, build_callbacks
from biapy.data.generators import (
    create_train_val_augmentors,
    create_test_generator,
    create_chunked_test_generator,
    check_generator_consistence,
)
from biapy.utils.misc import (
    get_world_size,
    get_rank,
    is_main_process,
    save_model,
    time_text,
    load_model_checkpoint,
    TensorboardLogger,
    MetricLogger,
    to_pytorch_format,
    to_numpy_format,
    is_dist_avail_and_initialized,
    setup_for_distributed,
    update_dict_with_existing_keys,
)
from biapy.engine.check_configuration import (
    convert_old_model_cfg_to_current_version,
    diff_between_configs,
    compare_configurations_without_model,
    check_configuration,
)
from biapy.utils.util import (
    create_plots,
    check_downsample_division,
)
from biapy.engine.train_engine import train_one_epoch, evaluate
from biapy.data.data_2D_manipulation import (
    crop_data_with_overlap,
    merge_data_with_overlap,
)
from biapy.data.data_3D_manipulation import crop_3D_data_with_overlap, merge_3D_data_with_overlap, order_dimensions
from biapy.data.data_manipulation import (
    load_and_prepare_train_data,
    load_and_prepare_test_data,
    read_img_as_ndarray,
    save_tif,
    resize,
)
from biapy.data.post_processing.post_processing import (
    ensemble8_2d_predictions,
    ensemble16_3d_predictions,
    apply_binary_mask,
)
from biapy.data.post_processing import apply_post_processing
from biapy.data.pre_processing import preprocess_data
from biapy.data.norm import Normalization
from biapy.data.generators.chunked_test_pair_data_generator import chunked_test_pair_data_generator
from biapy.data.dataset import PatchCoords
from biapy.models.memory_bank import MemoryBank


class Base_Workflow(metaclass=ABCMeta):
    """
    Base workflow class. A new workflow should extend this class.

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

    def __init__(
        self,
        cfg: CN,
        job_identifier: str,
        device: torch.device,
        args: argparse.Namespace,
    ):
        self.cfg = cfg
        self.args = args
        self.job_identifier = job_identifier
        self.device = device
        self.original_test_mask_path = None
        self.test_mask_filenames = None
        self.cross_val_samples_ids = None
        self.post_processing = {}
        self.post_processing["per_image"] = False
        self.post_processing["as_3D_stack"] = False
        self.data_norm = None
        self.model = None
        self.model_build_kwargs = None
        self.checkpoint_path = None
        self.optimizer = None
        self.model_prepared = False
        self.dtype = np.float32 if not self.cfg.TEST.REDUCE_MEMORY else np.float16
        self.dtype_str = "float32" if not self.cfg.TEST.REDUCE_MEMORY else "float16"
        self.loss_dtype = torch.float32
        self.dims = 2 if self.cfg.PROBLEM.NDIM == "2D" else 3

        self.use_gt = False
        if self.cfg.DATA.TEST.LOAD_GT or self.cfg.DATA.TEST.USE_VAL_AS_TEST:
            self.use_gt = True

        # Save paths in case we need them in a future
        self.orig_train_path = self.cfg.DATA.TRAIN.PATH
        self.orig_train_mask_path = self.cfg.DATA.TRAIN.GT_PATH
        self.orig_val_path = self.cfg.DATA.VAL.PATH
        self.orig_val_mask_path = self.cfg.DATA.VAL.GT_PATH

        self.all_pred = []
        self.all_gt = []

        self.stats = {}

        # Per crop
        self.stats["per_crop"] = {}
        self.stats["patch_by_batch_counter"] = 0

        # Merging the image
        self.stats["merge_patches"] = {}
        self.stats["merge_patches_post"] = {}

        # As 3D stack
        self.stats["as_3D_stack"] = {}
        self.stats["as_3D_stack_post"] = {}

        # Full image
        self.stats["full_image"] = {}
        self.stats["full_image_post"] = {}

        # To store all the metrics for each test file in order to create a final csv file with the results
        self.metrics_per_test_file = [] 

        self.mask_path = ""
        self.is_y_mask = False
        self.model_output_channels = {}
        self.activations = []
        self.multihead = None
        self.train_metrics = []
        self.train_metric_best = []
        self.train_metric_names = []
        self.test_metrics = []
        self.test_metric_best = []
        self.test_metric_names = []
        self.loss = None
        self.memory_bank = None

        self.resolution: List[int | float] = list(self.cfg.DATA.TEST.RESOLUTION)
        if self.cfg.PROBLEM.NDIM == "2D":
            self.resolution = [
                1,
            ] + self.resolution

        self.world_size = get_world_size()
        self.global_rank = get_rank()

        # Test variables
        if self.cfg.TEST.POST_PROCESSING.MEDIAN_FILTER:
            if self.cfg.PROBLEM.NDIM == "2D":
                if self.cfg.TEST.ANALIZE_2D_IMGS_AS_3D_STACK:
                    self.post_processing["as_3D_stack"] = True
                else:
                    self.post_processing["per_image"] = True
            else:
                self.post_processing["per_image"] = True

        # Define permute shapes to pass from Numpy axis order (Y,X,C) to Pytorch's (C,Y,X)
        self.axes_order = (0, 3, 1, 2) if self.cfg.PROBLEM.NDIM == "2D" else (0, 4, 1, 2, 3)
        self.axes_order_back = (0, 2, 3, 1) if self.cfg.PROBLEM.NDIM == "2D" else (0, 2, 3, 4, 1)

        # Tochvision variables
        self.torchvision_preprocessing = None

        # Load BioImage Model Zoo pretrained model information
        self.bmz_config = {}
        self.bmz_pipeline = None
        if self.cfg.MODEL.SOURCE == "bmz":
            self.bmz_config["preprocessing"] = check_bmz_args(self.cfg.MODEL.BMZ.SOURCE_MODEL_ID, self.cfg)

            print("Loading BioImage Model Zoo pretrained model . . .")
            self.bmz_config["original_bmz_config"] = load_description(self.cfg.MODEL.BMZ.SOURCE_MODEL_ID)

            opts = check_model_restrictions(self.cfg, self.bmz_config, {"workflow_type": cfg.PROBLEM.TYPE})

            self.cfg.merge_from_list(opts)
            update_dependencies(self.cfg)

        # Save number of channels to be created by the model
        self.define_activations_and_channels()

        # Define metrics
        self.define_metrics()

        # Normalization checks
        print("Creating normalization module . . .")
        self.norm_module = Normalization(
            type=cfg.DATA.NORMALIZATION.TYPE,
            measure_by=cfg.DATA.NORMALIZATION.MEASURE_BY,
            mask_norm="as_mask",
            out_dtype="float32" if not cfg.TEST.REDUCE_MEMORY else "float16",
            percentile_clip=cfg.DATA.NORMALIZATION.PERC_CLIP.ENABLE,
            per_lower_bound=cfg.DATA.NORMALIZATION.PERC_CLIP.LOWER_PERC,
            per_upper_bound=cfg.DATA.NORMALIZATION.PERC_CLIP.UPPER_PERC,
            lower_bound_val=cfg.DATA.NORMALIZATION.PERC_CLIP.LOWER_VALUE,
            upper_bound_val=cfg.DATA.NORMALIZATION.PERC_CLIP.UPPER_VALUE,
            mean=cfg.DATA.NORMALIZATION.ZERO_MEAN_UNIT_VAR.MEAN_VAL,
            std=cfg.DATA.NORMALIZATION.ZERO_MEAN_UNIT_VAR.STD_VAL,
        )
        self.test_norm_module = self.norm_module.copy()
        self.test_norm_module.train_normalization = False
        if self.cfg.MODEL.SOURCE == "torchvision":
            print("Creating normalization module . . .")
            self.torchvision_norm = Normalization(
                type="scale_range",
                measure_by="image",
                mask_norm="as_mask",
                out_dtype="float32" if not cfg.TEST.REDUCE_MEMORY else "float16",
                percentile_clip=cfg.DATA.NORMALIZATION.PERC_CLIP.ENABLE,
                per_lower_bound=cfg.DATA.NORMALIZATION.PERC_CLIP.LOWER_PERC,
                per_upper_bound=cfg.DATA.NORMALIZATION.PERC_CLIP.UPPER_PERC,
                lower_bound_val=cfg.DATA.NORMALIZATION.PERC_CLIP.LOWER_VALUE,
                upper_bound_val=cfg.DATA.NORMALIZATION.PERC_CLIP.UPPER_VALUE,
            )

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
        if not self.model_output_channels:
            raise ValueError(
                "'model_output_channels' needs to be defined. Correct define_activations_and_channels() function"
            )
        else:
            if not isinstance(self.model_output_channels, dict):
                raise ValueError("'self.model_output_channels' must be a dict")
            if "type" not in self.model_output_channels:
                raise ValueError("'self.model_output_channels' must have 'type' key")
            if "channels" not in self.model_output_channels:
                raise ValueError("'self.model_output_channels' must have 'channels' key")
        if self.multihead is None:
            raise ValueError("'multihead' needs to be defined. Correct define_activations_and_channels() function")
        if not self.activations:
            raise ValueError("'activations' needs to be defined. Correct define_activations_and_channels() function")
        else:
            if not isinstance(self.activations, list):
                raise ValueError("'self.activations' must be a list of dicts")
            for x in self.activations:
                if not isinstance(x, dict):
                    raise ValueError("'self.activations' must be a list of dicts")

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
        if not self.train_metrics:
            raise ValueError("'train_metrics' needs to be defined. Correct define_metrics() function")
        if not self.train_metric_names:
            raise ValueError("'train_metric_names' needs to be defined. Correct define_metrics() function")
        if not self.train_metric_best:
            raise ValueError("'train_metric_best' needs to be defined. Correct define_metrics() function")
        else:
            assert all(
                [True if x in ["max", "min"] else False for x in self.train_metric_best]
            ), "'train_metric_best' needs to be one between ['max', 'min']"
        if not self.test_metrics:
            raise ValueError("'test_metrics' needs to be defined. Correct define_metrics() function")
        if not self.test_metric_names:
            raise ValueError("'test_metric_names' needs to be defined. Correct define_metrics() function")
        if self.loss is None:
            raise ValueError("'loss' needs to be defined. Correct define_metrics() function")

    @abstractmethod
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
        value : float
            Value of the metric for the given prediction.
        """
        raise NotImplementedError

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
        return to_pytorch_format(targets, self.axes_order, self.device)

    def load_train_data(self):
        """
        Load training and validation data.
        """
        print("##########################")
        print("#   LOAD TRAINING DATA   #")
        print("##########################")
        train_zarr_data_information = {
            "raw_path": self.cfg.DATA.TRAIN.INPUT_ZARR_MULTIPLE_DATA_RAW_PATH,
            "gt_path": self.cfg.DATA.TRAIN.INPUT_ZARR_MULTIPLE_DATA_GT_PATH,
            "use_gt_path": self.cfg.PROBLEM.TYPE != "INSTANCE_SEG",
            "multiple_data_within_zarr": self.cfg.DATA.TRAIN.INPUT_ZARR_MULTIPLE_DATA,
            "input_img_axes": self.cfg.DATA.TRAIN.INPUT_IMG_AXES_ORDER,
            "input_mask_axes": self.cfg.DATA.TRAIN.INPUT_MASK_AXES_ORDER,
        }
        val_zarr_data_information = {
            "raw_path": self.cfg.DATA.VAL.INPUT_ZARR_MULTIPLE_DATA_RAW_PATH,
            "gt_path": self.cfg.DATA.VAL.INPUT_ZARR_MULTIPLE_DATA_GT_PATH,
            "use_gt_path": self.cfg.PROBLEM.TYPE != "INSTANCE_SEG",
            "multiple_data_within_zarr": self.cfg.DATA.VAL.INPUT_ZARR_MULTIPLE_DATA,
            "input_img_axes": self.cfg.DATA.VAL.INPUT_IMG_AXES_ORDER,
            "input_mask_axes": self.cfg.DATA.VAL.INPUT_MASK_AXES_ORDER,
        }
        (
            self.X_train,
            self.Y_train,
            self.X_val,
            self.Y_val,
        ) = load_and_prepare_train_data(
            train_path=self.cfg.DATA.TRAIN.PATH,
            train_mask_path=self.mask_path,
            train_in_memory=self.cfg.DATA.TRAIN.IN_MEMORY,
            train_ov=self.cfg.DATA.TRAIN.OVERLAP,
            train_padding=self.cfg.DATA.TRAIN.PADDING,
            val_path=self.cfg.DATA.VAL.PATH,
            val_mask_path=self.cfg.DATA.VAL.GT_PATH,
            val_in_memory=self.cfg.DATA.VAL.IN_MEMORY,
            val_ov=self.cfg.DATA.VAL.OVERLAP,
            val_padding=self.cfg.DATA.VAL.PADDING,
            norm_module=self.norm_module,
            crop_shape=self.cfg.DATA.PATCH_SIZE,
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
            random_crops_in_DA=self.cfg.DATA.EXTRACT_RANDOM_PATCH,
            filter_by_entire_image=self.cfg.DATA.FILTER_BY_IMAGE,
            norm_before_filter=self.cfg.DATA.TRAIN.FILTER_SAMPLES.NORM_BEFORE,
            y_upscaling=self.cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING,
            reflect_to_complete_shape=self.cfg.DATA.REFLECT_TO_COMPLETE_SHAPE,
            convert_to_rgb=self.cfg.DATA.FORCE_RGB,
            is_y_mask=self.is_y_mask,
            is_3d=(self.cfg.PROBLEM.NDIM == "3D"),
            train_zarr_data_information=train_zarr_data_information,
            val_zarr_data_information=val_zarr_data_information,
            multiple_raw_images=(
                self.cfg.PROBLEM.TYPE == "IMAGE_TO_IMAGE"
                and self.cfg.PROBLEM.IMAGE_TO_IMAGE.MULTIPLE_RAW_ONE_TARGET_LOADER
            ),
            save_filtered_images=self.cfg.DATA.SAVE_FILTERED_IMAGES,
            save_filtered_images_dir=self.cfg.PATHS.FIL_SAMPLES_DIR,
            save_filtered_images_num=self.cfg.DATA.SAVE_FILTERED_IMAGES_NUM,
        )

        # Ensure all the processes have read the data
        if is_dist_avail_and_initialized():
            print("Waiting until all processes have read the data . . .")
            dist.barrier()

    def destroy_train_data(self):
        """
        Delete training variable to release memory.
        """
        print("Releasing memory . . .")
        if "X_train" in locals() or "X_train" in globals():
            del self.X_train
        if "Y_train" in locals() or "Y_train" in globals():
            del self.Y_train
        if "X_val" in locals() or "X_val" in globals():
            del self.X_val
        if "Y_val" in locals() or "Y_val" in globals():
            del self.Y_val
        if "train_generator" in locals() or "train_generator" in globals():
            del self.train_generator
        if "val_generator" in locals() or "val_generator" in globals():
            del self.val_generator

    def prepare_train_generators(self):
        """
        Build train and val generators.
        """
        if self.cfg.TRAIN.ENABLE:
            print("##############################")
            print("#  PREPARE TRAIN GENERATORS  #")
            print("##############################")
            (
                self.train_generator,
                self.val_generator,
                self.data_norm,
                self.num_training_steps_per_epoch,
                self.bmz_config["test_input"],
            ) = create_train_val_augmentors(
                self.cfg,
                X_train=self.X_train,
                X_val=self.X_val,
                Y_train=self.Y_train,
                Y_val=self.Y_val,
                norm_module=self.norm_module,
            )
            if self.cfg.DATA.CHECK_GENERATORS and self.cfg.PROBLEM.TYPE != "CLASSIFICATION":
                check_generator_consistence(
                    self.train_generator,
                    self.cfg.PATHS.GEN_CHECKS + "_train",
                    self.cfg.PATHS.GEN_MASK_CHECKS + "_train",
                )
                check_generator_consistence(
                    self.val_generator,
                    self.cfg.PATHS.GEN_CHECKS + "_val",
                    self.cfg.PATHS.GEN_MASK_CHECKS + "_val",
                )

    def bmz_model_call(self, in_img, is_train=False):
        """
        Call BioImage Model Zoo model.

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
        # ##### OPTION 1: we need batch size information as apply_preprocessing fails if the batch is not the same as the
        # ##### one fixed for the model. Last batch of the epoch can have less samples than batch size.
        # ##### Check torch.utils.data.DataLoader() drop last arg.
        # # Convert from Numpy to xarray.DataArray
        # self.bmz_axes = self.bmz_config['original_bmz_config'].inputs[0].axes
        # in_img = xr.DataArray(in_img.cpu().numpy(), dims=tuple(self.bmz_axes))

        # # Apply pre-processing
        # in_img = dict(zip([ipt.name for ipt in self.bmz_pipeline.input_specs], (in_img,)))
        # self.bmz_computed_measures = {}
        # self.bmz_pipeline.apply_preprocessing(in_img, self.bmz_computed_measures)
        # # print(f"in_img: {in_img['input0'].shape} {in_img['input0'].min()} {in_img['input0'].max()}")
        # # Predict
        # prediction = self.model(torch.from_numpy(np.array(in_img['input0'])).to(self.device))

        # # Apply post-processing (if any)
        # if bool(self.bmz_pipeline.output_specs[0].postprocessing):
        #     prediction = xr.DataArray(prediction.cpu().numpy(), dims=tuple(self.bmz_axes))
        #     prediction = dict(zip([out.name for out in self.bmz_pipeline.output_specs], prediction))
        #     self.bmz_pipeline.apply_postprocessing(prediction, self.bmz_computed_measures)

        #     # Convert back to Tensor
        #     prediction = torch.from_numpy(np.array(prediction)).to(self.device)

        ##### OPTION 2: Just a normal model call, but the pre and post need to be done in BiaPy
        assert self.model
        prediction = self.model(in_img)

        return prediction

    @abstractmethod
    def torchvision_model_call(self, in_img: torch.Tensor, is_train=False) -> torch.Tensor:
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
        raise NotImplementedError

    def model_call_func(
        self, in_img: NDArray | torch.Tensor, is_train: bool = False, apply_act: bool = True
    ) -> torch.Tensor:
        """
        Call a regular Pytorch model.

        Parameters
        ----------
        in_img : torch.Tensor
            Input image to pass through the model.

        is_train : bool, optional
            Whether if the call is during training or inference.

        apply_act : bool, optional
            Whether to apply activations or not.

        Returns
        -------
        prediction : torch.Tensor
            Image prediction.
        """
        in_img = to_pytorch_format(in_img, self.axes_order, self.device)
        assert isinstance(in_img, torch.Tensor)

        if self.cfg.MODEL.SOURCE == "biapy":
            assert self.model
            p = self.model(in_img)

            if (
                not (self.cfg.PROBLEM.TYPE == "SELF_SUPERVISED" and self.cfg.PROBLEM.SELF_SUPERVISED.PRETEXT_TASK.lower() == "masking") 
                and self.cfg.PROBLEM.TYPE not in ["CLASSIFICATION", "SUPER_RESOLUTION"]
            ):
                # Recover the original shape of the input, as not all the model return a prediction
                # of the same size as the input image
                if isinstance(p, dict):
                    if p["pred"].shape[2:] != in_img.shape[2:]:
                        p["pred"] = resize(p["pred"], in_img.shape, mode="bilinear")
                        if "class" in p:
                            p["class"] = resize(p["class"], in_img.shape, mode="nearest")
                else:
                    if p.shape[2:] != in_img.shape[2:]:
                        p = resize(p, in_img.shape, mode="bilinear")
            if apply_act:
                p = self.apply_model_activations(p, training=is_train)
        elif self.cfg.MODEL.SOURCE == "bmz":
            p = self.bmz_model_call(in_img, is_train)
        elif self.cfg.MODEL.SOURCE == "torchvision":
            p = self.torchvision_model_call(in_img, is_train)
        return p

    def prepare_model(self):
        """
        Build the model.
        """
        if self.model_prepared:
            print("Model already prepared!")
            return

        print("###############")
        print("# Build model #")
        print("###############")
        if self.cfg.MODEL.SOURCE == "biapy":
            # Obtain model spec from checkpoint
            if self.cfg.MODEL.LOAD_CHECKPOINT and self.cfg.MODEL.LOAD_MODEL_FROM_CHECKPOINT:
                # Take cfg from the checkpoint
                saved_cfg, biapy_ckpt_version = load_model_checkpoint(
                    cfg=self.cfg,
                    jobname=self.job_identifier,
                    model_without_ddp=None,
                    device=self.device,
                    just_extract_checkpoint_info=True,
                    skip_unmatched_layers=self.cfg.MODEL.SKIP_UNMATCHED_LAYERS,
                )
                if saved_cfg:
                    # Checks that this config and previous represent same workflow
                    header_message = "There is an inconsistency between the configuration loaded from checkpoint and the actual one. Error:\n"
                    tmp_cfg = convert_old_model_cfg_to_current_version(saved_cfg.clone())
                    compare_configurations_without_model(
                        self.cfg, tmp_cfg, header_message, old_cfg_version=biapy_ckpt_version
                    )

                    # Override model specs
                    if self.cfg.PROBLEM.PRINT_OLD_KEY_CHANGES:
                        print("The following changes were made in order to adapt the loaded input configuration from checkpoint into the current configuration version:")
                        diff_between_configs(saved_cfg, tmp_cfg)
                    update_dict_with_existing_keys(self.cfg["MODEL"], tmp_cfg["MODEL"])

                    # Check if the merge is coherent
                    self.cfg["MODEL"]["LOAD_CHECKPOINT"] = True
                    self.cfg["MODEL"]["LOAD_MODEL_FROM_CHECKPOINT"] = False
                    check_configuration(self.cfg, self.job_identifier)
            (
                self.model,
                self.bmz_config["model_file"],
                self.bmz_config["model_name"],
                self.model_build_kwargs,
                self.network_stride,
            ) = build_model(self.cfg, self.model_output_channels["channels"], self.device)
        elif self.cfg.MODEL.SOURCE == "torchvision":
            self.model, self.torchvision_preprocessing = build_torchvision_model(self.cfg, self.device)
        # BioImage Model Zoo pretrained models
        elif self.cfg.MODEL.SOURCE == "bmz":
            # Create a bioimage pipeline to create predictions
            try:
                self.bmz_pipeline = create_prediction_pipeline(
                    self.bmz_config["original_bmz_config"],
                    devices=None,
                    weight_format="pytorch_state_dict",
                )
            except Exception as e:
                print(f"The error thrown during the BMZ model load was:\n{e}")
                raise ValueError(
                    "An error ocurred when creating the BMZ model (see above). "
                    "BiaPy only supports models prepared with pytorch_state_dict."
                )

            if self.args.distributed:
                raise ValueError("DDP can not be activated when loading a BMZ pretrained model")

            self.model = build_bmz_model(self.cfg, self.bmz_config["original_bmz_config"], self.device)

        self.model_without_ddp = self.model
        if self.args.distributed:
            if self.cfg.MODEL.ARCHITECTURE.lower() in ["unetr", "resunet_se"]:
                find_unused_parameters = True
            else:
                find_unused_parameters = False
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.args.gpu],
                find_unused_parameters=find_unused_parameters,
            )
            self.model_without_ddp = self.model.module
        self.model_prepared = True

        # Load checkpoint if necessary
        if self.cfg.MODEL.SOURCE == "biapy" and self.cfg.MODEL.LOAD_CHECKPOINT:
            self.start_epoch, self.checkpoint_path = load_model_checkpoint(
                cfg=self.cfg,
                jobname=self.job_identifier,
                model_without_ddp=self.model_without_ddp,
                device=self.device,
                optimizer=self.optimizer,
                skip_unmatched_layers=self.cfg.MODEL.SKIP_UNMATCHED_LAYERS,
            )
        else:
            self.start_epoch = 0

    def prepare_logging_tool(self):
        """
        Prepare looging tool.
        """
        print("#######################")
        print("# Prepare logging tool #")
        print("#######################")
        # To start the logging
        now = datetime.datetime.now()
        now = now.strftime("%Y_%m_%d_%H_%M_%S")
        self.log_file = os.path.join(
            self.cfg.LOG.LOG_DIR,
            self.cfg.LOG.LOG_FILE_PREFIX + "_log_" + str(now) + ".txt",
        )
        if self.global_rank == 0:
            os.makedirs(self.cfg.LOG.LOG_DIR, exist_ok=True)
            os.makedirs(self.cfg.PATHS.CHECKPOINT, exist_ok=True)
            self.log_writer = TensorboardLogger(log_dir=self.cfg.LOG.TENSORBOARD_LOG_DIR)
        else:
            self.log_writer = None

        self.plot_values = {}
        self.plot_values["loss"] = []
        self.plot_values["val_loss"] = []
        for i in range(len(self.train_metric_names)):
            self.plot_values[self.train_metric_names[i]] = []
            self.plot_values["val_" + self.train_metric_names[i]] = []

    def train(self):
        """
        Training phase.
        """
        self.load_train_data()
        if not self.model_prepared:
            self.prepare_model()
        self.prepare_train_generators()
        self.prepare_logging_tool()
        self.early_stopping = build_callbacks(self.cfg)

        assert (
            self.start_epoch is not None and self.model is not None and self.model_without_ddp is not None and self.loss
        )
        self.optimizer, self.lr_scheduler = prepare_optimizer(
            self.cfg, self.model_without_ddp, len(self.train_generator)
        )

        contrast_init_iter = 0
        if self.cfg.LOSS.CONTRAST.ENABLE:
            self.memory_bank = MemoryBank(
                num_classes=self.cfg.DATA.N_CLASSES,
                memory_size = self.cfg.LOSS.CONTRAST.MEMORY_SIZE,
                feature_dims = self.cfg.LOSS.CONTRAST.PROJ_DIM,
                network_stride = self.network_stride,
                pixel_update_freq=self.cfg.LOSS.CONTRAST.PIXEL_UPD_FREQ,
                device = self.device,
                ignore_index = self.cfg.LOSS.IGNORE_INDEX,
            )
            self.memory_bank.to(self.device)
            # When to activate the contrastive loss
            contrast_init_iter = self.cfg.LOSS.CONTRAST.MEMORY_SIZE
            if self.cfg.TRAIN.LR_SCHEDULER.NAME == "warmupcosine":
                contrast_init_iter += self.cfg.TRAIN.LR_SCHEDULER.WARMUP_COSINE_DECAY_EPOCHS
        
        print("#####################")
        print("#  TRAIN THE MODEL  #")
        print("#####################")

        print(f"Start training in epoch {self.start_epoch+1} - Total: {self.cfg.TRAIN.EPOCHS}")
        start_time = time.time()
        self.val_best_metric = np.zeros(len(self.train_metric_names), dtype=np.float32)
        self.val_best_loss = np.inf
        total_iters = 0
        for epoch in range(self.start_epoch, self.cfg.TRAIN.EPOCHS):
            print("~~~ Epoch {}/{} ~~~\n".format(epoch + 1, self.cfg.TRAIN.EPOCHS))
            e_start = time.time()

            if self.args.distributed:
                self.train_generator.sampler.set_epoch(epoch)  # type: ignore
            if self.log_writer:
                self.log_writer.set_step(epoch * self.num_training_steps_per_epoch)

            # Train
            train_stats, iterations_done = train_one_epoch(
                self.cfg,
                model=self.model,
                model_call_func=self.model_call_func,
                loss_function=self.loss,
                metric_function=self.metric_calculation,
                prepare_targets=self.prepare_targets,
                data_loader=self.train_generator,
                optimizer=self.optimizer,
                device=self.device,
                epoch=epoch,
                log_writer=self.log_writer,
                lr_scheduler=self.lr_scheduler,
                verbose=self.cfg.TRAIN.VERBOSE,
                memory_bank=self.memory_bank,
                total_iters=total_iters,
                contrast_warmup_iters=contrast_init_iter,
            )
            total_iters += iterations_done

            # Save checkpoint
            if self.cfg.MODEL.SAVE_CKPT_FREQ != -1:
                if (
                    (epoch + 1) % self.cfg.MODEL.SAVE_CKPT_FREQ == 0
                    or epoch + 1 == self.cfg.TRAIN.EPOCHS
                    and is_main_process()
                ):
                    save_model(
                        cfg=self.cfg,
                        biapy_version=biapy.__version__,
                        jobname=self.job_identifier,
                        model_without_ddp=self.model_without_ddp,
                        optimizer=self.optimizer,
                        epoch=epoch + 1,
                        model_build_kwargs=self.model_build_kwargs,
                    )

            # Validation
            if self.val_generator:
                test_stats = evaluate(
                    self.cfg,
                    model=self.model,
                    model_call_func=self.model_call_func,
                    loss_function=self.loss,
                    metric_function=self.metric_calculation,
                    prepare_targets=self.prepare_targets,
                    epoch=epoch,
                    data_loader=self.val_generator,
                    lr_scheduler=self.lr_scheduler,
                    memory_bank=self.memory_bank,
                )

                # Save checkpoint is val loss improved
                if test_stats["loss"] < self.val_best_loss:
                    f = os.path.join(
                        self.cfg.PATHS.CHECKPOINT,
                        "{}-checkpoint-best.pth".format(self.job_identifier),
                    )
                    print(
                        "Val loss improved from {} to {}, saving model to {}".format(
                            self.val_best_loss, test_stats["loss"], f
                        )
                    )
                    m = " "
                    for i in range(len(self.val_best_metric)):
                        self.val_best_metric[i] = test_stats[self.train_metric_names[i]]
                        m += f"{self.train_metric_names[i]}: {self.val_best_metric[i]:.4f} "
                    self.val_best_loss = test_stats["loss"]

                    if is_main_process():
                        self.checkpoint_path = save_model(
                            cfg=self.cfg,
                            biapy_version=biapy.__version__,
                            jobname=self.job_identifier,
                            model_without_ddp=self.model_without_ddp,
                            optimizer=self.optimizer,
                            epoch="best",
                            model_build_kwargs=self.model_build_kwargs,
                        )
                print(f"[Val] best loss: {self.val_best_loss:.4f} best " + m)

                # Store validation stats
                if self.log_writer:
                    self.log_writer.update(test_loss=test_stats["loss"], head="perf", step=epoch)
                    for i in range(len(self.train_metric_names)):
                        self.log_writer.update(
                            test_iou=test_stats[self.train_metric_names[i]],
                            head="perf",
                            step=epoch,
                        )

                log_stats = {
                    **{f"train_{k}": v for k, v in train_stats.items()},
                    **{f"test_{k}": v for k, v in test_stats.items()},
                    "epoch": epoch,
                }
            else:
                log_stats = {
                    **{f"train_{k}": v for k, v in train_stats.items()},
                    "epoch": epoch,
                }

            # Write statistics in the logging file
            if is_main_process():
                # Log epoch stats
                if self.log_writer:
                    self.log_writer.flush()
                with open(self.log_file, mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # Create training plot
                self.plot_values["loss"].append(train_stats["loss"])
                if self.val_generator:
                    self.plot_values["val_loss"].append(test_stats["loss"])
                for i in range(len(self.train_metric_names)):
                    self.plot_values[self.train_metric_names[i]].append(train_stats[self.train_metric_names[i]])
                    if self.val_generator:
                        self.plot_values["val_" + self.train_metric_names[i]].append(
                            test_stats[self.train_metric_names[i]]
                        )
                if (epoch + 1) % self.cfg.LOG.CHART_CREATION_FREQ == 0:
                    create_plots(
                        self.plot_values,
                        self.train_metric_names,
                        self.job_identifier,
                        self.cfg.PATHS.CHARTS,
                    )

            if self.val_generator and self.early_stopping:
                self.early_stopping(test_stats["loss"])
                if self.early_stopping.early_stop:
                    print("Early stopping")
                    break

            e_end = time.time()
            t_epoch = e_end - e_start
            print(
                "[Time] {} {}/{}\n".format(
                    time_text(t_epoch),
                    time_text(e_end - start_time),
                    time_text((e_end - start_time) + (t_epoch * (self.cfg.TRAIN.EPOCHS - epoch))),
                )
            )

        total_time = time.time() - start_time
        self.total_training_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time: {}".format(self.total_training_time_str))

        print("Train loss: {}".format(train_stats["loss"]))
        for i in range(len(self.train_metric_names)):
            print("Train {}: {}".format(self.train_metric_names[i], train_stats[self.train_metric_names[i]]))
        if self.val_generator:
            print("Validation loss: {}".format(self.val_best_loss))
            for i in range(len(self.train_metric_names)):
                print("Validation {}: {}".format(self.train_metric_names[i], self.val_best_metric[i]))

        print("Finished Training")

        if is_dist_avail_and_initialized():
            print(f"[Rank {get_rank()} ({os.getpid()})] Process waiting (train finished, step 1) . . . ")
            dist.barrier()

        # Save output sample to export the model to BMZ
        if "test_output" not in self.bmz_config:
            assert self.model_without_ddp
            self.model_without_ddp.eval()
            # Load best checkpoint on validation to ensure it
            _ = load_model_checkpoint(
                cfg=self.cfg,
                jobname=self.job_identifier,
                model_without_ddp=self.model_without_ddp,
                device=self.device,
                skip_unmatched_layers=self.cfg.MODEL.SKIP_UNMATCHED_LAYERS,
            )

            # Save BMZ input/output so the user could export the model to BMZ later
            self.prepare_bmz_data(self.bmz_config["test_input"])

        self.destroy_train_data()

    def load_test_data(self):
        """
        Load test data.
        """
        print("######################")
        print("#   LOAD TEST DATA   #")
        print("######################")

        self.X_test, self.Y_test = None, None
        if self.cfg.DATA.TEST.USE_VAL_AS_TEST:
            print("Loading train data information to extract the validation to be used as test")
            self.cfg.merge_from_list(["DATA.TRAIN.IN_MEMORY", False, "DATA.VAL.IN_MEMORY", False])
            self.load_train_data()
            self.X_test = self.X_val.copy()
            if self.Y_val:
                self.Y_test = self.Y_val.copy()
        else:
            # Paths to the raw and gt within the Zarr file. Only used when 'DATA.TEST.INPUT_ZARR_MULTIPLE_DATA' is True.
            test_zarr_data_information = None
            if self.cfg.DATA.TEST.INPUT_ZARR_MULTIPLE_DATA:
                use_gt_path = True
                if self.cfg.PROBLEM.TYPE != "INSTANCE_SEG" and self.cfg.PROBLEM.INSTANCE_SEG.TYPE != "synapses":
                    use_gt_path = False
                test_zarr_data_information = {
                    "raw_path": self.cfg.DATA.TEST.INPUT_ZARR_MULTIPLE_DATA_RAW_PATH,
                    "gt_path": self.cfg.DATA.TEST.INPUT_ZARR_MULTIPLE_DATA_GT_PATH,
                    "use_gt_path": use_gt_path,
                }

            (
                self.X_test,
                self.Y_test,
                self.test_filenames,
            ) = load_and_prepare_test_data(
                test_path=self.cfg.DATA.TEST.PATH,
                test_mask_path=self.cfg.DATA.TEST.GT_PATH if self.use_gt else None,
                multiple_raw_images=(
                    self.cfg.PROBLEM.TYPE == "IMAGE_TO_IMAGE"
                    and self.cfg.PROBLEM.IMAGE_TO_IMAGE.MULTIPLE_RAW_ONE_TARGET_LOADER
                ),
                test_zarr_data_information=test_zarr_data_information,
            )

    def destroy_test_data(self):
        """
        Delete test variable to release memory.
        """
        print("Releasing memory . . .")
        if "X_test" in locals() or "X_test" in globals():
            del self.X_test
        if "Y_test" in locals() or "Y_test" in globals():
            del self.Y_test
        if "test_generator" in locals() or "test_generator" in globals():
            del self.test_generator
        if "current_sample" in locals() or "current_sample" in globals():
            del self.current_sample

    def prepare_test_generators(self):
        """
        Prepare test data generator.
        """
        if self.cfg.TEST.ENABLE:
            print("############################")
            print("#  PREPARE TEST GENERATOR  #")
            print("############################")
            (self.test_generator, self.data_norm, test_input) = create_test_generator(
                self.cfg,
                self.X_test,
                self.Y_test,
                norm_module=self.test_norm_module,
            )
            # Only save it if it was not done before
            if "test_input" not in self.bmz_config:
                self.bmz_config["test_input"] = test_input

    def apply_model_activations(self, pred: torch.Tensor, training=False) -> torch.Tensor:
        """
        Function that apply the last activation (if any) to the model's output.

        Parameters
        ----------
        pred : Torch Tensor
            Predictions of the model.

        training : bool, optional
            To advice the function if this is being applied during training of inference. During training,
            ``CE_Sigmoid`` activations will NOT be applied, as ``torch.nn.BCEWithLogitsLoss`` will apply
            ``Sigmoid`` automatically in a way that is more stable numerically
            (`ref <https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html>`_).

        Returns
        -------
        pred : Torch tensor
            Resulting predictions after applying last activation(s).
        """
        # Not apply the activation, as it will be done in the BMZ model
        if self.cfg.MODEL.SOURCE == "bmz":
            return pred

        def __apply_acts(prediction, acts):
            for key, value in acts.items():
                # Ignore CE_Sigmoid as torch.nn.BCEWithLogitsLoss will apply Sigmoid automatically in a way
                # that is more stable numerically (ref: https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
                if (training and value not in ["Linear", "CE_Sigmoid"]) or (not training and value != "Linear"):
                    value = "Sigmoid" if value == "CE_Sigmoid" else value
                    act = getattr(torch.nn, value)()
                    if key == ":":
                        prediction = act(prediction)
                    else:
                        prediction[:, int(key), ...] = act(prediction[:, int(key), ...])
            return prediction

        if isinstance(pred, dict):
            pred["pred"] = __apply_acts(pred["pred"], self.activations[0])
            if "class" in pred:
                pred["class"] = __apply_acts(pred["class"], self.activations[1])
        else:
            pred = __apply_acts(pred, self.activations[0])
        return pred

    @torch.no_grad()
    def test(self):
        """
        Test/Inference step.
        """
        self.load_test_data()
        if not self.model_prepared:
            self.prepare_model()
        self.prepare_test_generators()

        # Switch to evaluation mode
        assert self.model_without_ddp
        if self.cfg.MODEL.SOURCE != "bmz":
            self.model_without_ddp.eval()

        # When not training was done
        if "test_output" not in self.bmz_config:
            # Save BMZ input/output so the user could export the model to BMZ later
            self.prepare_bmz_data(self.bmz_config["test_input"])

        # Load best checkpoint on validation
        if self.cfg.TRAIN.ENABLE:
            self.start_epoch, self.checkpoint_path = load_model_checkpoint(
                cfg=self.cfg,
                jobname=self.job_identifier,
                model_without_ddp=self.model_without_ddp,
                device=self.device,
                skip_unmatched_layers=self.cfg.MODEL.SKIP_UNMATCHED_LAYERS,
            )

        # Check possible checkpoint problems
        if self.start_epoch == -1:
            raise ValueError("There was a problem loading the checkpoint. Test phase aborted!")

        image_counter = 0

        print("###############")
        print("#  INFERENCE  #")
        print("###############")
        print("Making predictions on test data . . .")

        # Reactivate prints to see each rank progress
        if self.cfg.TEST.BY_CHUNKS.ENABLE and self.cfg.PROBLEM.NDIM == "3D":
            setup_for_distributed(True)

        # Process all the images
        for i, self.current_sample in enumerate(self.test_generator):  # type: ignore
            self.current_sample_metrics = {"file": self.current_sample["filename"]}
            self.f_numbers = [i]
            if "Y" not in self.current_sample:
                self.current_sample["Y"] = None

            # Decide whether to infer by chunks or not
            discarded = False
            _, file_extension = os.path.splitext(self.current_sample["filename"])
            if self.cfg.TEST.BY_CHUNKS.ENABLE and self.cfg.PROBLEM.NDIM == "3D":
                by_chunks = True
                if file_extension not in [".hdf5", ".hdf", ".h5", ".zarr", ".n5"]:
                    print(
                        "WARNING: You are not using an image format that can extract patches without loading it entirely into memory. "
                        "The image formats that support this are: '.hdf5', '.hdf', '.h5', '.zarr' and '.n5'. "
                    )
            else:
                by_chunks = False

            if by_chunks:
                print(
                    "[Rank {} ({})] Processing image (by chunks): {}".format(
                        get_rank(), os.getpid(), self.current_sample["filename"]
                    )
                )
                self.process_test_sample_by_chunks()
            else:
                if is_main_process():
                    if self.cfg.PROBLEM.TYPE != "CLASSIFICATION":
                        print("Processing image: {}".format(self.current_sample["filename"]))
                    discarded = self.process_test_sample()

            # If process_test_sample() returns True means that the sample was skipped due to filter set
            # up with: DATA.TEST.FILTER_SAMPLES
            if discarded:
                print(" Skipping image: {}".format(self.current_sample["filename"]))
            else:
                image_counter += 1

            self.metrics_per_test_file.append(self.current_sample_metrics)
            
        # Only enable print for the main rank again
        if self.cfg.TEST.BY_CHUNKS.ENABLE and self.cfg.PROBLEM.NDIM == "3D":
            setup_for_distributed(is_main_process())

        self.destroy_test_data()

        if is_main_process():
            self.after_all_images()

            print("#############")
            print("#  RESULTS  #")
            print("#############")
            print("The values below represent the averages across all test samples")
            if self.cfg.TRAIN.ENABLE:
                print("Epoch number: {}".format(len(self.plot_values["val_loss"])))
                print("Train time (s): {}".format(self.total_training_time_str))
                print("Train loss: {}".format(np.min(self.plot_values["loss"])))

                for i in range(len(self.train_metric_names)):
                    metric_name = (
                        "Foreground IoU" if self.train_metric_names[i] == "IoU" else self.train_metric_names[i]
                    )
                    print(
                        "Train {}: {}".format(
                            metric_name,
                            (
                                np.max(self.plot_values[self.train_metric_names[i]])
                                if self.train_metric_best[i] == "max"
                                else np.min(self.plot_values[self.train_metric_names[i]])
                            ),
                        )
                    )
                print("Validation loss: {}".format(self.val_best_loss))
                for i in range(len(self.train_metric_names)):
                    metric_name = (
                        "Foreground IoU" if self.train_metric_names[i] == "IoU" else self.train_metric_names[i]
                    )
                    print(
                        "Validation {}: {}".format(
                            metric_name,
                            self.val_best_metric[i],
                        )
                    )
            self.print_stats(image_counter)

    def predict_batches_in_test(
        self, x_batch: NDArray, y_batch: Optional[NDArray], stats_name="per_crop", disable_tqdm: bool = False
    ) -> NDArray:
        """
        Predict a batch of data for the test phase.

        Parameters
        ----------
        x_batch : NDArray
            X batch data. Expected axes are: ``(batch, z, y, x, channels)`` for 3D and
            ``(batch, y, x, channels)`` for 2D.

        y_batch: NDArray
            Y batch data. Expected axes are: ``(batch, z, y, x, channels)`` for 3D and
            ``(batch, y, x, channels)`` for 2D.

        stats_name : str, optional
            Name of the statistics to save.

        disable_tqdm : bool, optional
            Whether to disable tqdm or not.

        Returns
        -------
        pred : NDArray
            Predicted batch.
        """
        if self.cfg.TEST.AUGMENTATION:
            for k in tqdm(range(x_batch.shape[0]), leave=False, disable=disable_tqdm):
                if self.cfg.PROBLEM.NDIM == "2D":
                    p = ensemble8_2d_predictions(
                        x_batch[k],
                        axes_order_back=self.axes_order_back,
                        axes_order=self.axes_order,
                        device=self.device,
                        pred_func=self.model_call_func,
                        mode=self.cfg.TEST.AUGMENTATION_MODE,
                    )
                else:
                    p = ensemble16_3d_predictions(
                        x_batch[k],
                        batch_size_value=self.cfg.TRAIN.BATCH_SIZE,
                        axes_order_back=self.axes_order_back,
                        axes_order=self.axes_order,
                        device=self.device,
                        pred_func=self.model_call_func,
                        mode=self.cfg.TEST.AUGMENTATION_MODE,
                    )

                # Multi-head concatenation
                if isinstance(p, dict):
                    if "class" in p:
                        p = torch.cat((p["pred"], torch.argmax(p["class"], dim=1).unsqueeze(1)), dim=1)
                    else:
                        p = p["pred"]

                # Calculate the metrics
                if y_batch is not None:                        
                    metric_values = self.metric_calculation(output=p, targets=np.expand_dims(y_batch[k],0), train=False)
                    for metric in metric_values:
                        if str(metric).lower() not in self.stats[stats_name]:
                            self.stats[stats_name][str(metric).lower()] = 0
                        self.stats[stats_name][str(metric).lower()] += metric_values[metric]
                self.stats["patch_by_batch_counter"] += 1

                p = to_numpy_format(p, self.axes_order_back)
                if "pred" not in locals():
                    pred = np.zeros((x_batch.shape[0],) + p.shape[1:], dtype=self.dtype)
                pred[k] = p
        else:
            l = int(math.ceil(x_batch.shape[0] / self.cfg.TRAIN.BATCH_SIZE))
            for k in tqdm(range(l), leave=False, disable=disable_tqdm):
                top = (
                    (k + 1) * self.cfg.TRAIN.BATCH_SIZE
                    if (k + 1) * self.cfg.TRAIN.BATCH_SIZE < x_batch.shape[0]
                    else x_batch.shape[0]
                )
                p = self.model_call_func(x_batch[k * self.cfg.TRAIN.BATCH_SIZE : top])

                # Multi-head concatenation
                if isinstance(p, dict):
                    if "class" in p:
                        p = torch.cat((p["pred"], torch.argmax(p["class"], dim=1).unsqueeze(1)), dim=1)
                    else:
                        p = p["pred"]

                # Calculate the metrics
                if y_batch is not None:
                    metric_values = self.metric_calculation(
                        output=p,
                        targets=y_batch[k * self.cfg.TRAIN.BATCH_SIZE : top],
                        train=False,
                    )
                    for metric in metric_values:
                        if str(metric).lower() not in self.stats[stats_name]:
                            self.stats[stats_name][str(metric).lower()] = 0
                        self.stats[stats_name][str(metric).lower()] += metric_values[metric]
                self.stats["patch_by_batch_counter"] += 1

                p = to_numpy_format(p, self.axes_order_back)
                if "pred" not in locals():
                    pred = np.zeros((x_batch.shape[0],) + p.shape[1:], dtype=self.dtype)
                pred[k * self.cfg.TRAIN.BATCH_SIZE : top] = p

        return pred

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
        raise NotImplementedError

    def process_test_sample_by_chunks(self):
        """
        Function to process a sample in the inference phase. A final H5/Zarr file is created in "TZCYX" or "TZYXC" order
        depending on ``DATA.TEST.INPUT_IMG_AXES_ORDER`` ('T' is always included).
        """
        if not self.cfg.TEST.REUSE_PREDICTIONS:
            # Create the generator
            self.test_generator = create_chunked_test_generator(
                self.cfg,
                current_sample=self.current_sample,
                norm_module=self.norm_module,
                out_dir=self.cfg.PATHS.RESULT_DIR.PER_IMAGE,
                dtype_str=self.dtype_str,
            )
            tgen: chunked_test_pair_data_generator = self.test_generator.dataset  # type: ignore

            # Get parallel data shape is ZYX
            _, z_dim, _, y_dim, x_dim = order_dimensions(
                tgen.X_parallel_data.shape, self.cfg.DATA.TEST.INPUT_IMG_AXES_ORDER
            )
            self.parallel_data_shape = [z_dim, y_dim, x_dim]
            samples_visited = {}
            for obj_list in self.test_generator:
                sampler_ids, img, mask, patch_in_data, added_pad, norm_extra_info = obj_list

                if self.cfg.TEST.VERBOSE:
                    print(
                        "[Rank {} ({})] Patch number {} processing patches {} from {}".format(
                            get_rank(), os.getpid(), sampler_ids, patch_in_data, self.parallel_data_shape
                        )
                    )

                # Pass the batch through the model
                pred = self.predict_batches_in_test(img, mask, disable_tqdm=True)

                lbreaked = False
                for i in range(pred.shape[0]):
                    # Break the loop as those samples were created just to complete the last batch
                    if sampler_ids[i] < sampler_ids[0] or sampler_ids[i] in samples_visited:
                        print(
                            "[Rank {} ({})] Patch {} discarded".format(
                                get_rank(),
                                os.getpid(),
                                sampler_ids[i],
                            )
                        )
                        lbreaked = True
                        break

                    single_pred = pred[i]
                    single_pred_pad = added_pad[i]
                    single_patch_in_data = patch_in_data[i]
                    self.after_one_patch_prediction_by_chunks(
                        sampler_ids[i], single_pred, single_patch_in_data, single_pred_pad
                    )

                    # Remove padding if added
                    single_pred = single_pred[
                        single_pred_pad[0][0] : single_pred.shape[0] - single_pred_pad[0][1],
                        single_pred_pad[1][0] : single_pred.shape[1] - single_pred_pad[1][1],
                        single_pred_pad[2][0] : single_pred.shape[2] - single_pred_pad[2][1],
                    ]
                    # Insert into the data
                    tgen.insert_patch_in_file(single_pred, single_patch_in_data)

                    samples_visited[sampler_ids[i]] = True

                if lbreaked and sampler_ids[i] in samples_visited:
                    print(
                        "[Rank {} ({})] Finishing the loop. Seems that the patches are starting to repeat".format(
                            get_rank(),
                            os.getpid(),
                        )
                    )
                    break

            # Wait until all threads are done so the main thread can create the full size image
            if self.cfg.SYSTEM.NUM_GPUS > 1 and is_dist_avail_and_initialized():
                if self.cfg.TEST.VERBOSE:
                    print(
                        f"[Rank {get_rank()} ({os.getpid()})] Finished predicting patches. Waiting for all ranks . . ."
                    )
                dist.barrier()

                if is_main_process():
                    tgen.merge_zarr_parts_into_one()

                if self.cfg.TEST.VERBOSE:
                    print(
                        f"[Rank {get_rank()} ({os.getpid()})] Waiting for master rank to create the final Zarr from all the parts . . ."
                    )
                dist.barrier()

            tgen.close_open_files()

            if self.cfg.TEST.BY_CHUNKS.SAVE_OUT_TIF and is_main_process():
                tgen.save_parallel_data_as_tif()

        if is_main_process():
            self.after_all_patch_prediction_by_chunks()

        # Wait until all threads are done so the main thread can create the full size image
        if self.cfg.SYSTEM.NUM_GPUS > 1 and is_dist_avail_and_initialized():
            if self.cfg.TEST.VERBOSE:
                print(f"[Rank {get_rank()} ({os.getpid()})] Finished predicting sample. Waiting for all ranks . . .")
            dist.barrier()

    def prepare_bmz_data(self, img):
        """
        Prepare required data for exporting a model into BMZ.

        Parameters
        ----------
        img : 4D/5D Numpy array
            Image to save (unnormalized). The axes must be in Torch format already, i.e. ``(b,c,y,x)`` for 2D or
            ``(b,c,z,y,x)`` for 3D.
        """

        def _prepare_bmz_sample(sample_key, img, apply_norm=True):
            """
            Prepare a sample from the given ``img`` using the patch size in the configuration. It also saves
            the sample in ``self.bmz_config`` using the ``sample_key``.

            Parameters
            ----------
            sample_key : str
                Key to store the sample into. Must be one between: ``["test_input", "test_output"]``

            img : 4D/5D Numpy array
                Image to extract the sample from. The axes must be in Torch format already, i.e.
                ``(b,c,y,x)`` for 2D or ``(b,c,z,y,x)`` for 3D.
            """
            img = img.astype(np.float32)
            if len(img.shape) == 2:  # Classification
                self.bmz_config[sample_key] = img[0]
            else:
                if self.cfg.PROBLEM.NDIM == "2D":
                    self.bmz_config[sample_key] = img[
                        0, :, : self.cfg.DATA.PATCH_SIZE[0], : self.cfg.DATA.PATCH_SIZE[1]
                    ].copy()
                else:
                    self.bmz_config[sample_key] = img[
                        0,
                        :,
                        : self.cfg.DATA.PATCH_SIZE[0],
                        : self.cfg.DATA.PATCH_SIZE[1],
                        : self.cfg.DATA.PATCH_SIZE[2],
                    ].copy()

            # Ensure dimensions
            if self.cfg.PROBLEM.NDIM == "2D":
                if self.bmz_config[sample_key].ndim == 3:
                    self.bmz_config[sample_key] = np.expand_dims(self.bmz_config[sample_key], 0)
            else:  # 3D
                if self.bmz_config[sample_key].ndim == 4:
                    self.bmz_config[sample_key] = np.expand_dims(self.bmz_config[sample_key], 0)

            # Apply normalization
            if apply_norm:
                self.norm_module.set_stats_from_image(self.bmz_config[sample_key])
                self.bmz_config[sample_key], _ = self.norm_module.apply_image_norm(self.bmz_config[sample_key])
                self.bmz_config[sample_key] = self.bmz_config[sample_key].astype(np.float32)

        # Save test_input without the normalization if not already saved
        if "test_input" not in self.bmz_config:
            _prepare_bmz_sample("test_input", img, apply_norm=False)

        # Save test_input with the normalization
        if "test_input_norm" not in self.bmz_config:
            _prepare_bmz_sample("test_input_norm", img)

        # Model prediction
        assert self.model and self.model_without_ddp
        assert isinstance(self.bmz_config["test_input_norm"], np.ndarray)
        pred = self.model(torch.from_numpy(self.bmz_config["test_input_norm"]).to(self.device))

        # MAE
        if isinstance(pred, dict) and "mask" in pred:
            pred = self.apply_model_activations(pred)
            mask = pred["mask"]
            pred = pred["pred"]
            pred, _, _ = self.model_without_ddp.save_images(
                torch.from_numpy(self.bmz_config["test_input_norm"]).to(self.device),
                pred,
                mask,
                self.dtype,
            )
            pred = torch.from_numpy(pred)
        else:
            pred = self.apply_model_activations(pred)
            # Multi-head concatenation
            if isinstance(pred, dict):
                if "class" in pred:
                    pred = torch.cat((pred["pred"], torch.argmax(pred["class"], dim=1).unsqueeze(1)), dim=1)
                else:
                    pred = pred["pred"]

        # Save output
        _prepare_bmz_sample("test_output", pred.clone().cpu().detach().numpy().astype(np.float32), apply_norm=False)

        self.bmz_config["postprocessing"] = []
        if self.cfg.MODEL.SOURCE == "biapy":
            # Check activations to be inserted as postprocessing in BMZ
            act = list(self.activations[0].values())
            for ac in act:
                if ac in ["CE_Sigmoid", "Sigmoid"]:
                    self.bmz_config["postprocessing"].append("sigmoid")

    def process_test_sample(self):
        """
        Function to process a sample in the inference phase.
        """
        # Skip processing image
        if "discard" in self.current_sample["X"] and self.current_sample["X"]["discard"]:
            return True

        #################
        ### PER PATCH ###
        #################
        if not self.cfg.TEST.FULL_IMG or self.cfg.PROBLEM.NDIM == "3D":
            if not self.cfg.TEST.REUSE_PREDICTIONS:
                original_data_shape = self.current_sample["X"].shape

                # Crop if necessary
                if self.current_sample["X"].shape[1:-1] != self.cfg.DATA.PATCH_SIZE[:-1]:
                    # Copy X to be used later in full image
                    if self.cfg.PROBLEM.NDIM != "3D":
                        X_original = self.current_sample["X"].copy()

                    if (
                        self.current_sample["Y"] is not None
                        and self.current_sample["X"].shape[:-1] != self.current_sample["Y"].shape[:-1]
                    ):
                        raise ValueError(
                            "Image {} and mask {} differ in shape (without considering the channels, i.e. last dimension)".format(
                                self.current_sample["X"].shape, self.current_sample["Y"].shape
                            )
                        )

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
                        if self.cfg.TEST.REDUCE_MEMORY:
                            self.current_sample["X"], _ = crop_3D_data_with_overlap(  # type: ignore
                                self.current_sample["X"][0],
                                self.cfg.DATA.PATCH_SIZE,
                                overlap=self.cfg.DATA.TEST.OVERLAP,
                                padding=self.cfg.DATA.TEST.PADDING,
                                verbose=self.cfg.TEST.VERBOSE,
                                median_padding=self.cfg.DATA.TEST.MEDIAN_PADDING,
                            )
                            if self.current_sample["Y"] is not None:
                                self.current_sample["Y"], _ = crop_3D_data_with_overlap(  # type: ignore
                                    self.current_sample["Y"][0],
                                    self.cfg.DATA.PATCH_SIZE[:-1] + (self.current_sample["Y"].shape[-1],),
                                    overlap=self.cfg.DATA.TEST.OVERLAP,
                                    padding=self.cfg.DATA.TEST.PADDING,
                                    verbose=self.cfg.TEST.VERBOSE,
                                    median_padding=self.cfg.DATA.TEST.MEDIAN_PADDING,
                                )
                        else:
                            if self.current_sample["Y"] is not None:
                                self.current_sample["Y"] = self.current_sample["Y"][0]
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

                # Delete self.current_sample["X"] as in 3D there is no full image
                if self.cfg.PROBLEM.NDIM == "3D":
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
                        if self.current_sample["Y"] is not None:
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
                    if self.cfg.PROBLEM.NDIM != "3D":
                        self.current_sample["X"] = X_original.copy()
                        del X_original
                    else:
                        pred = np.expand_dims(pred, 0)
                        if self.current_sample["Y"] is not None:
                            self.current_sample["Y"] = np.expand_dims(self.current_sample["Y"], 0)

                if self.cfg.DATA.REFLECT_TO_COMPLETE_SHAPE:
                    reflected_orig_shape = (1,) + self.current_sample["reflected_orig_shape"]
                    if reflected_orig_shape != pred.shape:
                        if self.cfg.PROBLEM.NDIM == "2D":
                            pred = pred[:, -reflected_orig_shape[1] :, -reflected_orig_shape[2] :]
                            if self.current_sample["Y"] is not None:
                                self.current_sample["Y"] = self.current_sample["Y"][
                                    :,
                                    -reflected_orig_shape[1] :,
                                    -reflected_orig_shape[2] :,
                                ]
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

                # Apply mask
                if self.cfg.TEST.POST_PROCESSING.APPLY_MASK:
                    pred = np.expand_dims(apply_binary_mask(pred[0], self.cfg.DATA.TEST.BINARY_MASKS), 0)

                # Save image
                if self.cfg.PATHS.RESULT_DIR.PER_IMAGE != "":
                    save_tif(
                        pred,
                        self.cfg.PATHS.RESULT_DIR.PER_IMAGE,
                        [self.current_sample["filename"]],
                        verbose=self.cfg.TEST.VERBOSE,
                    )

                # Argmax if needed
                if self.cfg.DATA.N_CLASSES > 2 and self.cfg.DATA.TEST.ARGMAX_TO_OUTPUT and not self.multihead:
                    _type = np.uint8 if self.cfg.DATA.N_CLASSES < 255 else np.uint16
                    pred = np.expand_dims(np.argmax(pred, -1), -1).astype(_type)

                # Calculate the metrics
                if self.current_sample["Y"] is not None:
                    metric_values = self.metric_calculation(output=pred, targets=self.current_sample["Y"], train=False)
                    for metric in metric_values:
                        if str(metric).lower() not in self.stats["merge_patches"]:
                            self.stats["merge_patches"][str(metric).lower()] = 0
                        self.stats["merge_patches"][str(metric).lower()] += metric_values[metric]
                        self.current_sample_metrics[str(metric).lower()] = metric_values[metric]

                ############################
                ### POST-PROCESSING (3D) ###
                ############################
                if self.post_processing["per_image"]:
                    pred = apply_post_processing(self.cfg, pred)

                    # Calculate the metrics
                    if self.current_sample["Y"] is not None:
                        metric_values = self.metric_calculation(
                            output=pred, targets=self.current_sample["Y"], train=False
                        )
                        for metric in metric_values:
                            if str(metric).lower() not in self.stats["merge_patches_post"]:
                                self.stats["merge_patches_post"][str(metric).lower()] = 0
                            self.stats["merge_patches_post"][str(metric).lower()] += metric_values[metric]
                            self.current_sample_metrics[str(metric).lower() + " (post-processing)"] = metric_values[metric]
                    save_tif(
                        pred,
                        self.cfg.PATHS.RESULT_DIR.PER_IMAGE_POST_PROCESSING,
                        [self.current_sample["filename"]],
                        verbose=self.cfg.TEST.VERBOSE,
                    )
            else:
                # Load prediction from file
                folder = (
                    self.cfg.PATHS.RESULT_DIR.PER_IMAGE_POST_PROCESSING
                    if self.post_processing["per_image"]
                    else self.cfg.PATHS.RESULT_DIR.PER_IMAGE
                )
                # read file created by 'save_tif' (it always has .tif extension)
                test_file = os.path.join(folder, os.path.splitext(self.current_sample["filename"])[0]+'.tif')
                pred = read_img_as_ndarray(test_file, is_3d=self.cfg.PROBLEM.NDIM == "3D")
                pred = np.expand_dims(pred, 0)  # expand dimensions to include "batch"

                # Calculate the metrics
                if self.current_sample["Y"] is not None:
                    metric_values = self.metric_calculation(output=pred, targets=self.current_sample["Y"], train=False)
                    for metric in metric_values:
                        if str(metric).lower() not in self.stats["merge_patches"]:
                            self.stats["merge_patches"][str(metric).lower()] = 0
                        self.stats["merge_patches"][str(metric).lower()] += metric_values[metric]
                        self.current_sample_metrics[str(metric).lower()] = metric_values[metric]

            self.after_merge_patches(pred)

            if self.cfg.TEST.ANALIZE_2D_IMGS_AS_3D_STACK:
                assert isinstance(self.all_pred, list) and isinstance(self.all_gt, list)
                self.all_pred.append(pred)
                if self.current_sample["Y"] is not None and self.all_gt is not None:
                    self.all_gt.append(self.current_sample["Y"])

        ##################
        ### FULL IMAGE ###
        ##################
        if self.cfg.TEST.FULL_IMG and self.cfg.PROBLEM.NDIM == "2D":
            self.current_sample["X"], o_test_shape = check_downsample_division(
                self.current_sample["X"], len(self.cfg.MODEL.FEATURE_MAPS) - 1
            )
            if not self.cfg.TEST.REUSE_PREDICTIONS:
                if self.current_sample["Y"] is not None:
                    self.current_sample["Y"], _ = check_downsample_division(
                        self.current_sample["Y"], len(self.cfg.MODEL.FEATURE_MAPS) - 1
                    )

                # Make the prediction
                if self.cfg.TEST.AUGMENTATION:
                    pred = ensemble8_2d_predictions(
                        self.current_sample["X"][0],
                        axes_order_back=self.axes_order_back,
                        pred_func=self.model_call_func,
                        axes_order=self.axes_order,
                        device=self.device,
                        mode=self.cfg.TEST.AUGMENTATION_MODE,
                    )
                else:
                    pred = self.model_call_func(self.current_sample["X"])

                # Multi-head concatenationç
                if isinstance(pred, dict):
                    if "class" in pred:
                        pred = torch.cat((pred["pred"], torch.argmax(pred["class"], dim=1).unsqueeze(1)), dim=1)
                    else:
                        pred = pred["pred"]

                pred = to_numpy_format(pred, self.axes_order_back)
                del self.current_sample["X"]

                # Recover original shape if padded with check_downsample_division
                pred = pred[:, : o_test_shape[1], : o_test_shape[2]]
                if self.current_sample["Y"] is not None:
                    self.current_sample["Y"] = self.current_sample["Y"][:, : o_test_shape[1], : o_test_shape[2]]

                # Save image
                save_tif(
                    pred,
                    self.cfg.PATHS.RESULT_DIR.FULL_IMAGE,
                    [self.current_sample["filename"]],
                    verbose=self.cfg.TEST.VERBOSE,
                )

                # Argmax if needed
                if self.cfg.DATA.N_CLASSES > 2 and self.cfg.DATA.TEST.ARGMAX_TO_OUTPUT and not self.multihead:
                    _type = np.uint8 if self.cfg.DATA.N_CLASSES < 255 else np.uint16
                    pred = np.expand_dims(np.argmax(pred, -1), -1).astype(_type)

                if self.cfg.TEST.POST_PROCESSING.APPLY_MASK:
                    pred = apply_binary_mask(pred, self.cfg.DATA.TEST.BINARY_MASKS)

            else:
                # load prediction from file
                # read file created by 'save_tif' (it always has .tif extension)
                test_file = os.path.join(self.cfg.PATHS.RESULT_DIR.FULL_IMAGE, os.path.splitext(self.current_sample["filename"])[0]+'.tif')
                pred = read_img_as_ndarray(test_file, is_3d=self.cfg.PROBLEM.NDIM == "3D")
                pred = np.expand_dims(pred, 0)  # expand dimensions to include "batch"

            # Calculate the metrics
            if self.current_sample["Y"] is not None:
                metric_values = self.metric_calculation(output=pred, targets=self.current_sample["Y"], train=False)
                for metric in metric_values:
                    if str(metric).lower() not in self.stats["full_image"]:
                        self.stats["full_image"][str(metric).lower()] = 0
                    self.stats["full_image"][str(metric).lower()] += metric_values[metric]
                    self.current_sample_metrics[str(metric).lower()] = metric_values[metric]

            if self.cfg.TEST.ANALIZE_2D_IMGS_AS_3D_STACK:
                assert isinstance(self.all_pred, list) and isinstance(self.all_gt, list)
                self.all_pred.append(pred)
                if self.current_sample["Y"] is not None and self.all_gt is not None:
                    self.all_gt.append(self.current_sample["Y"])

            self.after_full_image(pred)

    def normalize_stats(self, image_counter):
        """
        Normalize statistics.

        Parameters
        ----------
        image_counter : int
            Number of images to average the metrics.
        """
        # Per crop
        for metric in self.stats["per_crop"]:
            self.stats["per_crop"][metric] = (
                self.stats["per_crop"][metric] / self.stats["patch_by_batch_counter"]
                if self.stats["patch_by_batch_counter"] != 0
                else 0
            )

        # Merge patches
        for metric in self.stats["merge_patches"]:
            self.stats["merge_patches"][metric] = (
                self.stats["merge_patches"][metric] / image_counter if image_counter != 0 else 0
            )

        # Full image
        for metric in self.stats["full_image"]:
            self.stats["full_image"][metric] = (
                self.stats["full_image"][metric] / image_counter if image_counter != 0 else 0
            )

        if self.post_processing["per_image"]:
            for metric in self.stats["merge_patches_post"]:
                self.stats["merge_patches_post"][metric] = (
                    self.stats["merge_patches_post"][metric] / image_counter if image_counter != 0 else 0
                )

    def print_stats(self, image_counter):
        """
        Print statistics.

        Parameters
        ----------
        image_counter : int
            Number of images to call ``normalize_stats``.
        """
        self.normalize_stats(image_counter)
        if self.cfg.DATA.TEST.LOAD_GT:
            if not self.cfg.TEST.FULL_IMG or (len(self.stats["per_crop"]) > 0 or len(self.stats["merge_patches"]) > 0):
                if len(self.stats["per_crop"]) > 0:
                    for metric in self.test_metric_names:
                        if metric.lower() in self.stats["per_crop"]:
                            metric_name = "Foreground IoU" if metric == "IoU" else metric
                            print(
                                "Test {} (per patch): {}".format(
                                    metric_name,
                                    self.stats["per_crop"][metric.lower()],
                                )
                            )

                if len(self.stats["merge_patches"]) > 0:
                    for metric in self.test_metric_names:
                        if metric.lower() in self.stats["merge_patches"]:
                            metric_name = "Foreground IoU" if metric == "IoU" else metric
                            print(
                                "Test {} (merge patches): {}".format(
                                    metric_name,
                                    self.stats["merge_patches"][metric.lower()],
                                )
                            )
            else:
                if len(self.stats["full_image"]) > 0:
                    for metric in self.test_metric_names:
                        if metric.lower() in self.stats["full_image"]:
                            metric_name = "Foreground IoU" if metric == "IoU" else metric
                            print(
                                "Test {} (per image): {}".format(
                                    metric_name,
                                    self.stats["full_image"][metric.lower()],
                                )
                            )

            print(" ")

            if self.post_processing["per_image"] and len(self.stats["merge_patches_post"]) > 0:
                for metric in self.test_metric_names:
                    if metric.lower() in self.stats["merge_patches_post"]:
                        metric_name = "Foreground IoU" if metric == "IoU" else metric
                        print(
                            "Test {} (merge patches - post-processing): {}".format(
                                metric_name,
                                self.stats["merge_patches_post"][metric.lower()],
                            )
                        )
                print(" ")

            if self.post_processing["as_3D_stack"] and len(self.stats["as_3D_stack_post"]) > 0:
                for metric in self.test_metric_names:
                    if metric.lower() in self.stats["as_3D_stack_post"]:
                        metric_name = "Foreground IoU" if metric == "IoU" else metric
                        print(
                            "Test {} (as 3D stack - post-processing): {}".format(
                                metric_name,
                                self.stats["as_3D_stack_post"][metric.lower()],
                            )
                        )
                print(" ")

            df_metrics = pd.DataFrame(self.metrics_per_test_file) 
            os.makedirs(self.cfg.PATHS.RESULT_DIR.PATH, exist_ok=True)
            df_metrics.to_csv(
                os.path.join(
                    self.cfg.PATHS.RESULT_DIR.PATH,
                    "test_results_metrics.csv",
                ),
                index=False,
            )

    @abstractmethod
    def after_merge_patches(self, pred):
        """
        Place any code that needs to be done after merging all predicted patches into the original image.

        Parameters
        ----------
        pred : Torch Tensor
            Model prediction.
        """
        raise NotImplementedError

    @abstractmethod
    def after_full_image(self, pred: NDArray):
        """
        Place here any code that must be executed after generating the prediction by supplying the entire image to the model.
        To enable this, the model should be convolutional, and the image(s) should be in a 2D format. Using 3D images as
        direct inputs to the model is not feasible due to their large size.

        Parameters
        ----------
        pred : NDArray
            Model prediction.
        """
        raise NotImplementedError

    def after_all_images(self):
        """
        Place here any code that must be done after predicting all images.
        """
        ############################
        ### POST-PROCESSING (2D) ###
        ############################
        if self.post_processing["as_3D_stack"]:
            self.all_pred = np.expand_dims(np.concatenate(self.all_pred), 0)
            if self.cfg.DATA.TEST.LOAD_GT and self.all_gt is not None:
                self.all_gt = np.expand_dims(np.concatenate(self.all_gt), 0)

            save_tif(
                self.all_pred,
                self.cfg.PATHS.RESULT_DIR.AS_3D_STACK,
                verbose=self.cfg.TEST.VERBOSE,
            )
            save_tif(
                (self.all_pred > 0.5).astype(np.uint8),
                self.cfg.PATHS.RESULT_DIR.AS_3D_STACK_BIN,
                verbose=self.cfg.TEST.VERBOSE,
            )

            self.all_pred = apply_post_processing(self.cfg, self.all_pred)

            # Calculate the metrics
            if self.cfg.DATA.TEST.LOAD_GT:
                metric_values = self.metric_calculation(output=self.all_pred[0], targets=self.all_gt[0], train=False)
                for metric in metric_values:
                    self.stats["as_3D_stack_post"][str(metric).lower()] = metric_values[metric]
                    self.current_sample_metrics[str(metric).lower() + " as 3D stack (post-processing)"] = metric_values[metric]

            save_tif(
                self.all_pred,
                self.cfg.PATHS.RESULT_DIR.AS_3D_STACK_POST_PROCESSING,
                verbose=self.cfg.TEST.VERBOSE,
            )

    def after_all_patch_prediction_by_chunks(self):
        """
        Place any code that needs to be done after predicting all the patches, one by one, in the "by chunks" setting.
        """
        raise NotImplementedError