import math
import os
import datetime
import time
import json
import torch
import h5py
import zarr
import numpy as np
from tqdm import tqdm
from abc import ABCMeta, abstractmethod
from sklearn.model_selection import StratifiedKFold
import torch.multiprocessing as mp
import torch.distributed as dist

from biapy.models import build_model, build_torchvision_model
from biapy.engine import prepare_optimizer, build_callbacks
from biapy.data.generators import create_train_val_augmentors, create_test_augmentor, check_generator_consistence
from biapy.utils.misc import (get_world_size, get_rank, is_main_process, save_model, time_text, load_model_checkpoint, TensorboardLogger,
    to_pytorch_format, to_numpy_format, is_dist_avail_and_initialized, setup_for_distributed)
from biapy.utils.util import (load_data_from_dir, load_3d_images_from_dir, create_plots, pad_and_reflect, save_tif, check_downsample_division,
    read_chunked_data, order_dimensions)
from biapy.engine.train_engine import train_one_epoch, evaluate
from biapy.data.data_2D_manipulation import crop_data_with_overlap, merge_data_with_overlap, load_and_prepare_2D_train_data
from biapy.data.data_3D_manipulation import (crop_3D_data_with_overlap, merge_3D_data_with_overlap, load_and_prepare_3D_data, 
    load_and_prepare_3D_efficient_format_data, load_3D_efficient_files, extract_3D_patch_with_overlap_yield)
from biapy.data.post_processing.post_processing import ensemble8_2d_predictions, ensemble16_3d_predictions, apply_binary_mask
from biapy.engine.metrics import jaccard_index_numpy, voc_calculation
from biapy.data.post_processing import apply_post_processing
from biapy.data.pre_processing import preprocess_data


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
    def __init__(self, cfg, job_identifier, device, args):
        self.cfg = cfg
        self.args = args
        self.job_identifier = job_identifier
        self.device = device
        self.original_test_path = None
        self.original_test_mask_path = None
        self.test_mask_filenames = None
        self.cross_val_samples_ids = None
        self.post_processing = {}
        self.post_processing['per_image'] = False
        self.post_processing['as_3D_stack'] = False
        self.test_filenames = None 
        self.metrics = []
        self.data_norm = None
        self.model = None
        self.optimizer = None
        self.loss_scaler = None
        self.model_prepared = False 
        self.dtype = np.float32 if not self.cfg.TEST.REDUCE_MEMORY else np.float16 
        self.dtype_str = "float32" if not self.cfg.TEST.REDUCE_MEMORY else "float16" 
        self.loss_dtype = torch.float32 

        # Save paths in case we need them in a future
        self.orig_train_path = self.cfg.DATA.TRAIN.PATH
        self.orig_train_mask_path = self.cfg.DATA.TRAIN.GT_PATH
        self.orig_val_path = self.cfg.DATA.VAL.PATH
        self.orig_val_mask_path = self.cfg.DATA.VAL.GT_PATH

        self.all_pred = []
        self.all_gt = []

        self.stats = {}

        # Per crop
        self.stats['loss_per_crop'] = 0
        self.stats['iou_per_crop'] = 0
        self.stats['patch_counter'] = 0

        # Merging the image
        self.stats['iou_merge_patches'] = 0
        self.stats['ov_iou_merge_patches'] = 0
        self.stats['iou_merge_patches_post'] = 0
        self.stats['ov_iou_merge_patches_post'] = 0

        # As 3D stack
        self.stats['iou_as_3D_stack_post'] = 0
        self.stats['ov_iou_as_3D_stack_post'] = 0

        # Full image
        self.stats['loss'] = 0
        self.stats['iou'] = 0
        self.stats['ov_iou'] = 0

        self.world_size = get_world_size()
        self.global_rank = get_rank()
        if self.cfg.TEST.BY_CHUNKS.ENABLE and self.cfg.PROBLEM.NDIM == '3D':
            maxsize = min(10,self.cfg.SYSTEM.NUM_GPUS*10)
            self.output_queue = mp.Queue(maxsize=maxsize)
            self.input_queue = mp.Queue(maxsize=maxsize)
            self.extract_info_queue = mp.Queue()

        # Test variables
        if self.cfg.TEST.ANALIZE_2D_IMGS_AS_3D_STACK and self.cfg.PROBLEM.NDIM == "2D":
            if self.cfg.TEST.POST_PROCESSING.YZ_FILTERING or self.cfg.TEST.POST_PROCESSING.Z_FILTERING:
                self.post_processing['as_3D_stack'] = True
        elif self.cfg.PROBLEM.NDIM == "3D":
            if self.cfg.TEST.POST_PROCESSING.YZ_FILTERING or self.cfg.TEST.POST_PROCESSING.Z_FILTERING:
                self.post_processing['per_image'] = True

        # Define permute shapes to pass from Numpy axis order (Y,X,C) to Pytorch's (C,Y,X)
        self.axis_order = (0,3,1,2) if self.cfg.PROBLEM.NDIM == "2D" else (0,4,1,2,3)
        self.axis_order_back = (0,2,3,1) if self.cfg.PROBLEM.NDIM == "2D" else (0,2,3,4,1)

        # Define metrics
        self.define_metrics()

        # Load Bioimage model Zoo pretrained model information
        self.torchvision_preprocessing = None
        self.bmz_test_input = None
        self.bmz_test_output = None
        self.bmz_model_resource = None
        if self.cfg.MODEL.SOURCE == "bmz":
            import bioimageio.core
            import xarray as xr

            print("Loading Bioimage Model Zoo pretrained model . . .")
            self.bmz_model_resource = bioimageio.core.load_resource_description(self.cfg.MODEL.BMZ.SOURCE_MODEL_DOI)
        
            # Change PATCH_SIZE with the one stored in the RDF
            input_image = np.load(self.bmz_model_resource.test_inputs[0])
            opts = ["DATA.PATCH_SIZE", input_image.shape[2:]+(input_image.shape[1],)]
            print("[BMZ] Changed 'DATA.PATCH_SIZE' from {} to {} as defined in the RDF"
                  .format(self.cfg.DATA.PATCH_SIZE,opts[1]))
            self.cfg.merge_from_list(opts)

            
    @abstractmethod
    def define_metrics(self):
        """
        This function must define the following variables:

        self.metrics : List of functions
            Metrics to be calculated during model's training and inference. 

        self.metric_names : List of str
            Names of the metrics calculated. 
    
        self.loss : Function
            Loss function used during training. 
        """
        NotImplementedError

    @abstractmethod
    def metric_calculation(self, output, targets, metric_logger=None):
        """
        Execution of the metrics defined in :func:`~define_metrics` function. 

        Parameters
        ----------
        output : Torch Tensor
            Prediction of the model. 

        targets : Torch Tensor
            Ground truth to compare the prediction with. 

        metric_logger : MetricLogger, optional
            Class to be updated with the new metric(s) value(s) calculated. 
        
        Returns
        -------
        value : float
            Value of the metric for the given prediction. 
        """
        NotImplementedError

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
        return to_pytorch_format(targets, self.axis_order, self.device)
        
    def load_train_data(self):
        """ 
        Load training and validation data.
        """
        if self.cfg.TRAIN.ENABLE:
            print("##########################")
            print("#   LOAD TRAINING DATA   #")
            print("##########################")
            if self.cfg.DATA.TRAIN.IN_MEMORY:
                val_split = self.cfg.DATA.VAL.SPLIT_TRAIN if self.cfg.DATA.VAL.FROM_TRAIN else 0.
                f_name = load_and_prepare_2D_train_data if self.cfg.PROBLEM.NDIM == '2D' else load_and_prepare_3D_data
                preprocess_cfg = self.cfg.DATA.PREPROCESS if self.cfg.DATA.PREPROCESS.TRAIN else None
                preprocess_fn = preprocess_data if self.cfg.DATA.PREPROCESS.TRAIN else None
                is_y_mask = self.cfg.PROBLEM.TYPE in ['SEMANTIC_SEG', 'INSTANCE_SEG']
                objs = f_name(self.cfg.DATA.TRAIN.PATH, self.mask_path, cross_val=self.cfg.DATA.VAL.CROSS_VAL, 
                    cross_val_nsplits=self.cfg.DATA.VAL.CROSS_VAL_NFOLD, cross_val_fold=self.cfg.DATA.VAL.CROSS_VAL_FOLD, 
                    val_split=val_split, seed=self.cfg.SYSTEM.SEED, shuffle_val=self.cfg.DATA.VAL.RANDOM, 
                    random_crops_in_DA=self.cfg.DATA.EXTRACT_RANDOM_PATCH, crop_shape=self.cfg.DATA.PATCH_SIZE, 
                    y_upscaling=self.cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING, ov=self.cfg.DATA.TRAIN.OVERLAP, 
                    padding=self.cfg.DATA.TRAIN.PADDING, minimum_foreground_perc=self.cfg.DATA.TRAIN.MINIMUM_FOREGROUND_PER,
                    reflect_to_complete_shape=self.cfg.DATA.REFLECT_TO_COMPLETE_SHAPE, convert_to_rgb=self.cfg.DATA.FORCE_RGB,
                    preprocess_cfg = preprocess_cfg, is_y_mask = is_y_mask, preprocess_f=preprocess_fn)
            
                if self.cfg.DATA.VAL.FROM_TRAIN:
                    if self.cfg.DATA.VAL.CROSS_VAL:
                        self.X_train, self.Y_train, self.X_val, self.Y_val, self.train_filenames, self.cross_val_samples_ids = objs
                    else:
                        self.X_train, self.Y_train, self.X_val, self.Y_val, self.train_filenames = objs
                else:
                    self.X_train, self.Y_train, self.train_filenames = objs
                del objs
            else:
                # Checking if the user inputted Zarr/H5 files 
                zarr_files = sorted(next(os.walk(self.cfg.DATA.TRAIN.PATH))[1])
                h5_files = sorted(next(os.walk(self.cfg.DATA.TRAIN.PATH))[2])
                if self.cfg.PROBLEM.NDIM == '3D' and (len(zarr_files) > 0 and '.zarr' in zarr_files[0]) or \
                    (len(h5_files) > 0 and '.h5' in h5_files[0]):
                    val_split = self.cfg.DATA.VAL.SPLIT_TRAIN if self.cfg.DATA.VAL.FROM_TRAIN else 0.

                    if len(zarr_files) > 0 and '.zarr' in zarr_files[0]:
                        print("Working with Zarr files . . .")
                        img_files = [os.path.join(self.cfg.DATA.TRAIN.PATH, x) for x in zarr_files]
                        mask_files = [os.path.join(self.mask_path, x) for x in sorted(next(os.walk(self.mask_path))[1])]
                    elif len(h5_files) > 0 and '.h5' in h5_files[0]:
                        print("Working with H5 files . . .")
                        img_files = [os.path.join(self.cfg.DATA.TRAIN.PATH, x) for x in h5_files]
                        mask_files = [os.path.join(self.mask_path, x) for x in sorted(next(os.walk(self.mask_path))[2])]
                    del zarr_files, h5_files

                    if self.cfg.DATA.EXTRACT_RANDOM_PATCH:
                        print("WARNING: 'DATA.EXTRACT_RANDOM_PATCH' not taken into account when working with Zarr/H5 images")
                    if self.cfg.DATA.FORCE_RGB:
                        print("WARNING: 'DATA.FORCE_RGB' not taken into account when working with Zarr/H5 images")

                    objs = load_and_prepare_3D_efficient_format_data(
                        img_files, mask_files, input_img_axes=self.cfg.DATA.TRAIN.INPUT_IMG_AXES_ORDER, 
                        input_mask_axes=self.cfg.DATA.TRAIN.INPUT_MASK_AXES_ORDER, 
                        cross_val=self.cfg.DATA.VAL.CROSS_VAL, cross_val_nsplits=self.cfg.DATA.VAL.CROSS_VAL_NFOLD, 
                        cross_val_fold=self.cfg.DATA.VAL.CROSS_VAL_FOLD, val_split=val_split, seed=self.cfg.SYSTEM.SEED, 
                        shuffle_val=self.cfg.DATA.VAL.RANDOM, crop_shape=self.cfg.DATA.PATCH_SIZE, 
                        y_upscaling=self.cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING, 
                        ov=self.cfg.DATA.TRAIN.OVERLAP, padding=self.cfg.DATA.TRAIN.PADDING, 
                        minimum_foreground_perc=self.cfg.DATA.TRAIN.MINIMUM_FOREGROUND_PER)
                    
                    if self.cfg.DATA.VAL.FROM_TRAIN:
                        if self.cfg.DATA.VAL.CROSS_VAL:
                            self.X_train, self.Y_train, self.X_val, self.Y_val, self.cross_val_samples_ids = objs
                        else:
                            self.X_train, self.Y_train, self.X_val, self.Y_val = objs
                    else:
                        self.X_train, self.Y_train = objs
                    del objs
                    
                else:
                    self.X_train, self.Y_train = None, None

            ##################
            ### VALIDATION ###
            ##################
            if not self.cfg.DATA.VAL.FROM_TRAIN:
                if self.cfg.DATA.VAL.IN_MEMORY:
                    print("1) Loading validation images . . .")
                    f_name = load_data_from_dir if self.cfg.PROBLEM.NDIM == '2D' else load_3d_images_from_dir
                    preprocess_cfg = self.cfg.DATA.PREPROCESS if self.cfg.DATA.PREPROCESS.VAL else None
                    preprocess_fn = preprocess_data if self.cfg.DATA.PREPROCESS.VAL else None
                    is_y_mask = self.cfg.PROBLEM.TYPE in ['SEMANTIC_SEG', 'INSTANCE_SEG']
                    self.X_val, _, _ = f_name(self.cfg.DATA.VAL.PATH, crop=True, crop_shape=self.cfg.DATA.PATCH_SIZE,
                        overlap=self.cfg.DATA.VAL.OVERLAP, padding=self.cfg.DATA.VAL.PADDING,
                        reflect_to_complete_shape=self.cfg.DATA.REFLECT_TO_COMPLETE_SHAPE,
                        convert_to_rgb=self.cfg.DATA.FORCE_RGB, preprocess_cfg = preprocess_cfg, 
                        is_mask = False, preprocess_f=preprocess_fn)

                    if self.cfg.PROBLEM.NDIM == '2D':
                        crop_shape = (self.cfg.DATA.PATCH_SIZE[0]*self.cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING[0],
                            self.cfg.DATA.PATCH_SIZE[1]*self.cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING[1], self.cfg.DATA.PATCH_SIZE[2])
                    else:
                        crop_shape = (self.cfg.DATA.PATCH_SIZE[0], self.cfg.DATA.PATCH_SIZE[1]*self.cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING[0],
                            self.cfg.DATA.PATCH_SIZE[2]*self.cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING[1],
                            self.cfg.DATA.PATCH_SIZE[3]*self.cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING[2])
                    if self.load_Y_val:
                        self.Y_val, _, _ = f_name(self.cfg.DATA.VAL.GT_PATH, crop=True, crop_shape=crop_shape,
                            overlap=self.cfg.DATA.VAL.OVERLAP, padding=self.cfg.DATA.VAL.PADDING,
                            reflect_to_complete_shape=self.cfg.DATA.REFLECT_TO_COMPLETE_SHAPE,
                            check_channel=False, check_drange=False,convert_to_rgb=self.cfg.DATA.FORCE_RGB, 
                            preprocess_cfg = preprocess_cfg, is_mask = is_y_mask, preprocess_f=preprocess_fn)                            
                    else:
                        self.Y_val = None
                    if self.Y_val is not None and len(self.X_val) != len(self.Y_val):
                        raise ValueError("Different number of raw and ground truth items ({} vs {}). "
                            "Please check the data!".format(len(self.X_val), len(self.Y_val)))
                else:
                    # Checking if the user inputted Zarr/H5 files 
                    zarr_files = sorted(next(os.walk(self.cfg.DATA.VAL.PATH))[1])
                    h5_files = sorted(next(os.walk(self.cfg.DATA.VAL.PATH))[2])
                    if self.cfg.PROBLEM.NDIM == '3D' and (len(zarr_files) > 0 and '.zarr' in zarr_files[0]) or \
                        (len(h5_files) > 0 and '.h5' in h5_files[0]):
                        print("1) Loading validation image information . . .")
                        if len(zarr_files) > 0 and '.zarr' in zarr_files[0]:
                            print("Working with Zarr files . . .")
                            img_files = [os.path.join(self.cfg.DATA.VAL.PATH, x) for x in zarr_files]
                            mask_files = [os.path.join(self.mask_path, x) for x in sorted(next(os.walk(self.mask_path))[1])]
                        elif len(h5_files) > 0 and '.h5' in h5_files[0]:
                            print("Working with H5 files . . .")
                            img_files = [os.path.join(self.cfg.DATA.VAL.PATH, x) for x in h5_files]
                            mask_files = [os.path.join(self.mask_path, x) for x in sorted(next(os.walk(self.mask_path))[2])]
                        del zarr_files, h5_files

                        if self.cfg.DATA.FORCE_RGB:
                            print("WARNING: 'DATA.FORCE_RGB' not taken into account when working with Zarr/H5 images")

                        self.X_val, _ = load_3D_efficient_files(data_path=img_files, input_axes=self.cfg.DATA.VAL.INPUT_IMG_AXES_ORDER,
                            crop_shape=self.cfg.DATA.PATCH_SIZE, overlap=self.cfg.DATA.VAL.OVERLAP, padding=self.cfg.DATA.VAL.PADDING)

                        if self.cfg.PROBLEM.NDIM == '2D':
                            crop_shape = (self.cfg.DATA.PATCH_SIZE[0]*self.cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING[0],
                                self.cfg.DATA.PATCH_SIZE[1]*self.cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING[1], self.cfg.DATA.PATCH_SIZE[2])
                        else:
                            crop_shape = (self.cfg.DATA.PATCH_SIZE[0], self.cfg.DATA.PATCH_SIZE[1]*self.cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING[0],
                                self.cfg.DATA.PATCH_SIZE[2]*self.cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING[1],
                                self.cfg.DATA.PATCH_SIZE[3]*self.cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING[2])

                        if self.load_Y_val:
                            print("1) Loading validation GT information . . .")
                            self.Y_val, _ = load_3D_efficient_files(data_path=img_files, input_axes=self.cfg.DATA.VAL.INPUT_IMG_AXES_ORDER,
                                crop_shape=crop_shape, overlap=self.cfg.DATA.VAL.OVERLAP, padding=self.cfg.DATA.VAL.PADDING, check_channel=False)                          
                        else:
                            self.Y_val = None
                        if self.Y_val is not None and len(self.X_val) != len(self.Y_val):
                            raise ValueError("Different number of raw and ground truth items ({} vs {}). "
                                "Please check the data!".format(len(self.X_val), len(self.Y_val)))

                    else:        
                        self.X_val, self.Y_val = None, None

        # Ensure all the processes have read the data                 
        if is_dist_avail_and_initialized():
            print("Waiting until all processes have read the data . . .")
            dist.barrier()

    def destroy_train_data(self):
        """
        Delete training variable to release memory.
        """
        print("Releasing memory . . .")
        if 'X_train' in locals() or 'X_train' in globals():
            del self.X_train
        if 'Y_train' in locals() or 'Y_train' in globals():
            del self.Y_train
        if 'X_val' in locals() or 'X_val' in globals():
            del self.X_val
        if 'Y_val' in locals() or 'Y_val' in globals():
            del self.Y_val
        if 'train_generator' in locals() or 'train_generator' in globals():
            del self.train_generator
        if 'val_generator' in locals() or 'val_generator' in globals():
            del self.val_generator

    def prepare_train_generators(self):
        """
        Build train and val generators.
        """
        if self.cfg.TRAIN.ENABLE:
            print("##############################")
            print("#  PREPARE TRAIN GENERATORS  #")
            print("##############################")
            self.train_generator, \
            self.val_generator, \
            self.data_norm, \
            self.num_training_steps_per_epoch = create_train_val_augmentors(self.cfg, self.X_train, self.Y_train, 
                self.X_val, self.Y_val, self.world_size, self.global_rank, self.args.distributed)
            if self.cfg.DATA.CHECK_GENERATORS and self.cfg.PROBLEM.TYPE != 'CLASSIFICATION':
                check_generator_consistence(
                    self.train_generator, self.cfg.PATHS.GEN_CHECKS+"_train", self.cfg.PATHS.GEN_MASK_CHECKS+"_train")
                check_generator_consistence(
                    self.val_generator, self.cfg.PATHS.GEN_CHECKS+"_val", self.cfg.PATHS.GEN_MASK_CHECKS+"_val")

    def bmz_model_call(self, in_img, is_train=False):
        """
        Call Bioimage model zoo model.

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
        # Convert from Numpy to xarray.DataArray
        if self.cfg.PROBLEM.NDIM == '2D': 
            self.bmz_axes = ('b', 'c', 'y', 'x')
        else:
            self.bmz_axes = ('b', 'c', 'z', 'y', 'x')
        in_img = xr.DataArray(in_img.cpu().numpy(), dims=tuple(self.bmz_axes))

        # Apply pre-processing
        in_img = dict(zip([ipt.name for ipt in self.model.input_specs], (in_img,)))
        self.bmz_computed_measures = {}
        self.model.apply_preprocessing(in_img, self.bmz_computed_measures)

        # Predict
        prediction_tensors = self.model.predict(*list(in_img.values()))

        # Apply post-processing
        prediction = dict(zip([out.name for out in self.model.output_specs], prediction_tensors))
        self.model.apply_postprocessing(prediction, self.bmz_computed_measures)

        # Convert back to Tensor 
        prediction = torch.from_numpy(prediction['output0'].to_numpy())

        return prediction

    @abstractmethod
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
        raise NotImplementedError

    def model_call_func(self, in_img, to_pytorch=True, is_train=False):
        """
        Call a regular Pytorch model.

        Parameters
        ----------
        in_img : Tensor
            Input image to pass through the model.

        to_pytorch : bool, optional
            Whether if the input image needs to be converted into pytorch format or not.
        
        is_train : bool, optional
            Whether if the call is during training or inference. 

        Returns
        -------
        prediction : Tensor 
            Image prediction. 
        """
        if to_pytorch:
            in_img = to_pytorch_format(in_img, self.axis_order, self.device)
        if self.cfg.MODEL.SOURCE == "biapy":
            p = self.model(in_img)
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
            self.model = build_model(self.cfg, self.job_identifier, self.device)
        elif self.cfg.MODEL.SOURCE == "torchvision":
            self.model, self.torchvision_preprocessing = build_torchvision_model(self.cfg, self.device)
        # Bioimage Model Zoo pretrained models
        elif self.cfg.MODEL.SOURCE == "bmz":
            # Create a bioimage pipeline to create predictions
            try:
                self.model = bioimageio.core.create_prediction_pipeline(
                    self.bmz_model_resource, devices=None, 
                    weight_format="torchscript",
                )
            except Exception as e:
                print(f"The error thrown during the BMZ model load was:\n{e}")
                raise ValueError("An error ocurred when creating the BMZ model (see above). "
                    "BiaPy only supports models prepared with Torchscript.")

            if self.args.distributed:
                raise ValueError("DDP can not be activated when loading a BMZ pretrained model")

        self.model_without_ddp = self.model
        if self.args.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.args.gpu], 
                find_unused_parameters=False)
            self.model_without_ddp = self.model.module
        self.model_prepared = True

        # Load checkpoint if necessary
        if self.cfg.MODEL.SOURCE == "biapy" and self.cfg.MODEL.LOAD_CHECKPOINT:
            self.start_epoch = load_model_checkpoint(cfg=self.cfg, jobname=self.job_identifier, model_without_ddp=self.model_without_ddp,
                    device=self.device, optimizer=self.optimizer, loss_scaler=self.loss_scaler)
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
        self.log_file = os.path.join(self.cfg.LOG.LOG_DIR, self.cfg.LOG.LOG_FILE_PREFIX + "_log_"+str(now)+".txt")
        if self.global_rank == 0:
            os.makedirs(self.cfg.LOG.LOG_DIR, exist_ok=True)
            os.makedirs(self.cfg.PATHS.CHECKPOINT, exist_ok=True)
            self.log_writer = TensorboardLogger(log_dir=self.cfg.LOG.TENSORBOARD_LOG_DIR)
        else:
            self.log_writer = None

        self.plot_values = {}
        self.plot_values['loss'] = []
        self.plot_values['val_loss'] = []
        for i in range(len(self.metric_names)):
            self.plot_values[self.metric_names[i]] = []
            self.plot_values['val_'+self.metric_names[i]] = []

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
        
        self.optimizer, self.lr_scheduler, self.loss_scaler = prepare_optimizer(self.cfg, self.model_without_ddp, 
            len(self.train_generator))    

        print("#####################")
        print("#  TRAIN THE MODEL  #")
        print("#####################")
        
        print(f"Start training in epoch {self.start_epoch+1} - Total: {self.cfg.TRAIN.EPOCHS}")
        start_time = time.time()
        val_best_metric = np.zeros(len(self.metric_names), dtype=np.float32)
        val_best_loss = np.Inf
        for epoch in range(self.start_epoch, self.cfg.TRAIN.EPOCHS):
            print("~~~ Epoch {}/{} ~~~\n".format(epoch+1, self.cfg.TRAIN.EPOCHS))
            e_start = time.time()

            if self.args.distributed:
                self.train_generator.sampler.set_epoch(epoch)
            if self.log_writer is not None:
                self.log_writer.set_step(epoch * self.num_training_steps_per_epoch)

            # Train
            train_stats = train_one_epoch(self.cfg, model=self.model, model_call_func=self.model_call_func, loss_function=self.loss, 
                activations=self.apply_model_activations, metric_function=self.metric_calculation, prepare_targets=self.prepare_targets, 
                data_loader=self.train_generator, optimizer=self.optimizer, device=self.device, loss_scaler=self.loss_scaler, epoch=epoch, 
                log_writer=self.log_writer, lr_scheduler=self.lr_scheduler, start_steps=epoch * self.num_training_steps_per_epoch,
                verbose=self.cfg.TRAIN.VERBOSE)

            # Save checkpoint
            if self.cfg.MODEL.SAVE_CKPT_FREQ != -1:
                if (epoch + 1) % self.cfg.MODEL.SAVE_CKPT_FREQ == 0 or epoch + 1 == self.cfg.TRAIN.EPOCHS and is_main_process():
                    save_model(cfg=self.cfg, jobname=self.job_identifier, model=self.model, model_without_ddp=self.model_without_ddp, 
                        optimizer=self.optimizer, loss_scaler=self.loss_scaler, epoch=epoch+1)
                
            # Validation
            if self.val_generator is not None:
                test_stats = evaluate(self.cfg, model=self.model, model_call_func=self.model_call_func, loss_function=self.loss, 
                    activations=self.apply_model_activations, metric_function=self.metric_calculation, prepare_targets=self.prepare_targets, 
                    epoch=epoch, data_loader=self.val_generator, lr_scheduler=self.lr_scheduler)

                # Save checkpoint is val loss improved 
                if test_stats['loss'] < val_best_loss:
                    f = os.path.join(self.cfg.PATHS.CHECKPOINT,"{}-checkpoint-best.pth".format(self.job_identifier))
                    print("Val loss improved from {} to {}, saving model to {}".format(val_best_loss, test_stats['loss'], f))
                    m = " "
                    for i in range(len(val_best_metric)):
                        val_best_metric[i] = test_stats[self.metric_names[i]]
                        m += f"{self.metric_names[i]}: {val_best_metric[i]:.4f} "
                    val_best_loss = test_stats['loss']

                    if is_main_process():
                        save_model(cfg=self.cfg, jobname=self.job_identifier, model=self.model, model_without_ddp=self.model_without_ddp, 
                            optimizer=self.optimizer, loss_scaler=self.loss_scaler, epoch="best")
                print(f'[Val] best loss: {val_best_loss:.4f} best '+m)

                # Store validation stats 
                if self.log_writer is not None:
                    self.log_writer.update(test_loss=test_stats['loss'], head="perf", step=epoch)
                    for i in range(len(self.metric_names)):
                        self.log_writer.update(test_iou=test_stats[self.metric_names[i]], head="perf", step=epoch)
                    
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            **{f'test_{k}': v for k, v in test_stats.items()},
                            'epoch': epoch}
            else:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             'epoch': epoch}

            # Write statistics in the logging file
            if is_main_process():
                # Log epoch stats
                if self.log_writer is not None:
                    self.log_writer.flush()
                with open(self.log_file, mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # Create training plot
                self.plot_values['loss'].append(train_stats['loss'])
                if self.val_generator is not None:
                    self.plot_values['val_loss'].append(test_stats['loss'])
                for i in range(len(self.metric_names)):
                    self.plot_values[self.metric_names[i]].append(train_stats[self.metric_names[i]])
                    if self.val_generator is not None:
                        self.plot_values['val_'+self.metric_names[i]].append(test_stats[self.metric_names[i]])
                if (epoch+1) % self.cfg.LOG.CHART_CREATION_FREQ == 0:
                    create_plots(self.plot_values, self.metric_names, self.job_identifier, self.cfg.PATHS.CHARTS)

                if self.val_generator is not None and self.early_stopping is not None:
                    self.early_stopping(test_stats['loss'])
                    if self.early_stopping.early_stop:
                        print("Early stopping")
                        break
                        
            e_end = time.time()
            t_epoch = e_end - e_start
            print("[Time] {} {}/{}\n".format(time_text(t_epoch), time_text(e_end - start_time),
                                             time_text((e_end - start_time)+(t_epoch*(self.cfg.TRAIN.EPOCHS-epoch)))))
            
        total_time = time.time() - start_time
        self.total_training_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(self.total_training_time_str))

        print("Train loss: {}".format(train_stats['loss']))
        for i in range(len(self.metric_names)):
            print("Train {}: {}".format(self.metric_names[i], train_stats[self.metric_names[i]]))
        if self.val_generator is not None:
            print("Val loss: {}".format(val_best_loss))
            for i in range(len(self.metric_names)):
                print("Val {}: {}".format(self.metric_names[i], val_best_metric[i]))

        print('Finished Training')

        # Save two samples to export the model to BMZ 
        if self.bmz_test_input is None:
            sample = next(enumerate(self.train_generator))
            self.bmz_test_input = sample[1][0][0]
            self.bmz_test_output = sample[1][1]
            if not isinstance(self.bmz_test_output, int):
                self.bmz_test_output = self.bmz_test_output[0]

        self.destroy_train_data()

    def load_test_data(self):
        """
        Load test data.
        """
        if self.cfg.TEST.ENABLE:
            print("######################")
            print("#   LOAD TEST DATA   #")
            print("######################")
            if not self.cfg.DATA.TEST.USE_VAL_AS_TEST:
                if self.cfg.DATA.TEST.IN_MEMORY:
                    print("2) Loading test images . . .")
                    f_name = load_data_from_dir if self.cfg.PROBLEM.NDIM == '2D' else load_3d_images_from_dir
                    preprocess_cfg = self.cfg.DATA.PREPROCESS if self.cfg.DATA.PREPROCESS.TEST else None
                    preprocess_fn = preprocess_data if self.cfg.DATA.PREPROCESS.TEST else None
                    is_y_mask = self.cfg.PROBLEM.TYPE in ['SEMANTIC_SEG', 'INSTANCE_SEG']
                    self.X_test, _, _ = f_name(self.cfg.DATA.TEST.PATH, convert_to_rgb=self.cfg.DATA.FORCE_RGB,
                        preprocess_cfg=preprocess_cfg, is_mask=False, preprocess_f=preprocess_fn)
                    if self.cfg.DATA.TEST.LOAD_GT:
                        print("3) Loading test masks . . .")
                        self.Y_test, _, _ = f_name(self.cfg.DATA.TEST.GT_PATH, check_channel=False, check_drange=False, 
                                                   preprocess_cfg=preprocess_cfg, is_mask=is_y_mask, preprocess_f=preprocess_fn)
                        if len(self.X_test) != len(self.Y_test):
                            raise ValueError("Different number of raw and ground truth items ({} vs {}). "
                                "Please check the data!".format(len(self.X_test), len(self.Y_test)))
                    else:
                        self.Y_test = None
                else:
                    self.X_test, self.Y_test = None, None

                if self.original_test_path is None:
                    self.test_filenames = sorted(next(os.walk(self.cfg.DATA.TEST.PATH))[2])
                else:
                    self.test_filenames = sorted(next(os.walk(self.original_test_path))[2])
            else:
                # The test is the validation, and as it is only available when validation is obtained from train and when 
                # cross validation is enabled, the test set files reside in the train folder
                self.test_filenames = sorted(next(os.walk(self.cfg.DATA.TRAIN.PATH))[2])
                self.X_test, self.Y_test = None, None
                if self.cross_val_samples_ids is None:                      
                    # Split the test as it was the validation when train is not enabled 
                    skf = StratifiedKFold(n_splits=self.cfg.DATA.VAL.CROSS_VAL_NFOLD, shuffle=self.cfg.DATA.VAL.RANDOM,
                        random_state=self.cfg.SYSTEM.SEED)
                    fold = 1
                    test_index = None
                    A = B = np.zeros(len(self.test_filenames))  
                
                    for _, te_index in skf.split(A, B):
                        if self.cfg.DATA.VAL.CROSS_VAL_FOLD == fold:
                            self.cross_val_samples_ids = te_index.copy()
                            break
                        fold += 1
                    if len(self.cross_val_samples_ids) > 5:
                        print("Fold number {} used for test data. Printing the first 5 ids: {}".format(fold, self.cross_val_samples_ids[:5]))
                    else:
                        print("Fold number {}. Indexes used in cross validation: {}".format(fold, self.cross_val_samples_ids))
                
                self.test_filenames = [x for i, x in enumerate(self.test_filenames) if i in self.cross_val_samples_ids]
                self.original_test_path = self.orig_train_path
                self.original_test_mask_path = self.orig_train_mask_path  

    def destroy_test_data(self):
        """
        Delete test variable to release memory.
        """
        print("Releasing memory . . .")
        if 'X_test' in locals() or 'X_test' in globals():
            del self.self.X_test
        if 'Y_test' in locals() or 'Y_test' in globals():
            del self.self.Y_test
        if 'test_generator' in locals() or 'test_generator' in globals():
            del self.test_generator
        if '_X' in locals() or '_X' in globals():
            del self._X
        if '_Y' in locals() or '_Y' in globals():
            del self._Y

    def prepare_test_generators(self):
        """
        Prepare test data generator.
        """
        if self.cfg.TEST.ENABLE:
            print("############################")
            print("#  PREPARE TEST GENERATOR  #")
            print("############################")
            self.test_generator, self.data_norm = create_test_augmentor(self.cfg, self.X_test, self.Y_test, self.cross_val_samples_ids)

    def apply_model_activations(self, pred, training=False):
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
        if not isinstance(pred, list):
            multiple_heads = False
            pred = [pred]
        else: 
            multiple_heads = True
            assert len(pred) == len(self.activations), "Activations length need to match prediction list length in multiple heads setting"

        for out_heads in range(len(pred)):
            for key, value in self.activations[out_heads].items():
                # Ignore CE_Sigmoid as torch.nn.BCEWithLogitsLoss will apply Sigmoid automatically in a way 
                # that is more stable numerically (ref: https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
                if (training and value not in ["Linear", "CE_Sigmoid"]) or (not training and value != "Linear"):
                    value = "Sigmoid" if value == "CE_Sigmoid" else value
                    act = getattr(torch.nn, value)()
                    if key == ':':
                        pred[out_heads] = act(pred[out_heads])
                    else:
                        pred[out_heads][:,int(key),...] = act(pred[out_heads][:,int(key),...])

        if not multiple_heads:
            return pred[0]
        else:
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
        if self.cfg.MODEL.SOURCE != "bmz":
            self.model_without_ddp.eval()    

        # Check possible checkpoint problems
        if self.start_epoch == -1:
            raise ValueError("There was a problem loading the checkpoint. Test phase aborted!")

        image_counter = 0
        
        print("###############")
        print("#  INFERENCE  #")
        print("###############")
        print("Making predictions on test data . . .")

        # Reactivate prints to see each rank progress
        if self.cfg.TEST.BY_CHUNKS.ENABLE and self.cfg.PROBLEM.NDIM == '3D':
            setup_for_distributed(True)

        # Process all the images
        for i, batch in tqdm(enumerate(self.test_generator), total=len(self.test_generator), disable=not is_main_process()):
            if self.cfg.DATA.TEST.LOAD_GT and self.cfg.PROBLEM.TYPE not in ["SELF_SUPERVISED"]:
                X, X_norm, Y, Y_norm = batch
            else:
                X, X_norm = batch
                Y, Y_norm = None, None
            del batch

            if self.cfg.TEST.BY_CHUNKS.ENABLE and self.cfg.PROBLEM.NDIM == '3D':
                if type(X) is tuple:
                    self._X = X[0]
                    if self.cfg.DATA.TEST.LOAD_GT and self.cfg.PROBLEM.TYPE not in ["SELF_SUPERVISED"]:
                        self._Y = Y[0]  
                    else:
                        self._Y = None
                else:
                    self._X = X
                    self._Y = Y if self.cfg.DATA.TEST.LOAD_GT else None  

                if len(self.test_filenames) == 0:
                    self.test_filenames = sorted(next(os.walk(self.cfg.DATA.TEST.PATH))[1])  

                self.processing_filenames = self.test_filenames[i]
                if is_main_process():
                    print("Processing image: {}".format(self.processing_filenames))

                # Process each image separately
                self.f_numbers = [i]
                self.process_sample_by_chunks(self.processing_filenames)
            else:
                # Process all the images in the batch, sample by sample
                l_X = len(X)
                for j in tqdm(range(l_X), leave=False, disable=not is_main_process()):
                    self.processing_filenames = self.test_filenames[(i*l_X)+j:(i*l_X)+j+1]
                    if is_main_process():
                        print(f"[Rank {get_rank()} ({os.getpid()})] Processing image(s): {self.processing_filenames}")
                        
                        if self.cfg.PROBLEM.TYPE != 'CLASSIFICATION':
                            if type(X) is tuple:
                                self._X = X[j]
                                if self.cfg.DATA.TEST.LOAD_GT and self.cfg.PROBLEM.TYPE not in ["SELF_SUPERVISED"]:
                                    self._Y = Y[j]  
                                else:
                                    self._Y = None
                            else:
                                self._X = np.expand_dims(X[j],0)
                                if self.cfg.DATA.TEST.LOAD_GT and self.cfg.PROBLEM.TYPE not in ["SELF_SUPERVISED"]:
                                    self._Y = np.expand_dims(Y[j],0)  
                                else:
                                    self._Y = None
                        else:
                            self._X = np.expand_dims(X[j], 0)                    
                            self._Y = np.expand_dims(Y, 0) if self.cfg.DATA.TEST.LOAD_GT else None

                        # Process each image separately
                        self.f_numbers = list(range((i*l_X)+j,(i*l_X)+j+1)) 
                        self.process_sample(norm=(X_norm, Y_norm))                        
            
            image_counter += 1

        self.destroy_test_data()

        if is_main_process():
            self.after_all_images()

            print("#############")
            print("#  RESULTS  #")
            print("#############")

            if self.cfg.TRAIN.ENABLE:
                print("Epoch number: {}".format(len(self.plot_values['val_loss'])))
                print("Train time (s): {}".format(self.total_training_time_str))
                print("Train loss: {}".format(np.min(self.plot_values['loss'])))
                for i in range(len(self.metric_names)):
                    if self.metric_names[i] == "IoU":
                        print("Train Foreground {}: {}".format(self.metric_names[i], np.max(self.plot_values[self.metric_names[i]])))
                    else:
                        print("Train {}: {}".format(self.metric_names[i], np.max(self.plot_values[self.metric_names[i]])))
                print("Validation loss: {}".format(np.min(self.plot_values['val_loss'])))
                for i in range(len(self.metric_names)):
                    if self.metric_names[i] == "IoU":
                        print("Validation Foreground {}: {}".format(self.metric_names[i], np.max(self.plot_values['val_'+self.metric_names[i]])))
                    else:
                        print("Validation {}: {}".format(self.metric_names[i], np.max(self.plot_values['val_'+self.metric_names[i]])))
            self.print_stats(image_counter)
        
    def process_sample_by_chunks(self, filenames):
        """
        Function to process a sample in the inference phase. A final H5/Zarr file is created in "TZCYX" or "TZYXC" order
        depending on ``TEST.BY_CHUNKS.INPUT_IMG_AXES_ORDER`` ('T' is always included).

        Parameters
        ----------
        filenames : List of str
            Filenames fo the samples to process. 
        """
        filename, file_extension = os.path.splitext(filenames)
        if file_extension not in ['.hdf5', '.h5', ".zarr"]:
            print("WARNING: you could have saved more memory by converting input test images into H5 file format (.h5) "
                  "or Zarr (.zarr) as with 'TEST.BY_CHUNKS.ENABLE' option enabled H5/Zarr files will be processed by chunks")
        # Load data
        if file_extension in ['.hdf5', '.h5', ".zarr"]:
            self._X_file, self._X = read_chunked_data(self._X)
        else: # Numpy array
            if self._X.ndim == 3:
                c_pos = -1 if self.cfg.TEST.BY_CHUNKS.INPUT_IMG_AXES_ORDER[-1] == 'C' else 1
                self._X = np.expand_dims(self._X, c_pos)

        if is_main_process():
            print(f"Loaded image shape is {self._X.shape}")

        data_shape = self._X.shape

        if self._X.ndim < 3:
            raise ValueError("Loaded image need to have at least 3 dimensions: {} (ndim: {})".format(self._X.shape, self._X.ndim))
        
        if len(self.cfg.TEST.BY_CHUNKS.INPUT_IMG_AXES_ORDER) != self._X.ndim:
            raise ValueError("'TEST.BY_CHUNKS.INPUT_IMG_AXES_ORDER' value {} does not match the number of dimensions of the loaded H5/Zarr "
                "file {} (ndim: {})".format(self.cfg.TEST.BY_CHUNKS.INPUT_IMG_AXES_ORDER, self._X.shape, self._X.ndim))

        # Data paths
        os.makedirs(self.cfg.PATHS.RESULT_DIR.PER_IMAGE, exist_ok=True)
        ext = ".h5" if self.cfg.TEST.BY_CHUNKS.FORMAT == "h5" else ".zarr"
        if self.cfg.SYSTEM.NUM_GPUS > 1:
            out_data_filename = os.path.join(self.cfg.PATHS.RESULT_DIR.PER_IMAGE, filename+"_part"+str(get_rank())+ext)
            out_data_mask_filename = os.path.join(self.cfg.PATHS.RESULT_DIR.PER_IMAGE, filename+"_part"+str(get_rank())+"_mask"+ext)
        else:
            out_data_filename = os.path.join(self.cfg.PATHS.RESULT_DIR.PER_IMAGE, filename+"_nodiv"+ext)
            out_data_mask_filename = os.path.join(self.cfg.PATHS.RESULT_DIR.PER_IMAGE, filename+"_mask"+ext)
        out_data_div_filename = os.path.join(self.cfg.PATHS.RESULT_DIR.PER_IMAGE, filename+ext)
        in_data = self._X

        # Process in charge of processing one predicted patch
        output_handle_proc = mp.Process(target=insert_patch_into_dataset, args=(out_data_filename, out_data_mask_filename, 
            data_shape, self.output_queue, self.extract_info_queue, self.cfg, self.dtype_str, self.dtype, 
            self.cfg.TEST.BY_CHUNKS.FORMAT, self.cfg.TEST.VERBOSE))
        output_handle_proc.daemon=True
        output_handle_proc.start()
        
        # Process in charge of loading part of the data 
        load_data_process = mp.Process(target=extract_patch_from_dataset, args=(in_data, self.cfg, self.input_queue, 
            self.extract_info_queue, self.cfg.TEST.VERBOSE))
        load_data_process.daemon=True
        load_data_process.start()

        if '_X_file' in locals() and isinstance(self._X_file, h5py.File):
            self._X_file.close()
        del self._X, in_data
 
        # Lock the thread inferring until no more patches 
        if self.cfg.TEST.VERBOSE and self.cfg.SYSTEM.NUM_GPUS > 1:
            print(f"[Rank {get_rank()} ({os.getpid()})] Doing inference ")
        while True:
            obj = self.input_queue.get(timeout=60)
            if obj == None: break

            img, patch_coords = obj
            img, _ = self.test_generator.norm_X(img)
            if self.cfg.TEST.AUGMENTATION:
                p = ensemble16_3d_predictions(img[0], batch_size_value=self.cfg.TRAIN.BATCH_SIZE,
                    axis_order_back=self.axis_order_back, pred_func=self.model_call_func, 
                    axis_order=self.axis_order, device=self.device)
            else:
                with torch.cuda.amp.autocast():
                    p = self.model_call_func(img)
            p = self.apply_model_activations(p)
            # Multi-head concatenation
            if isinstance(p, list):
                p = torch.cat((p[0], torch.argmax(p[1], axis=1).unsqueeze(1)), dim=1)
            p = to_numpy_format(p, self.axis_order_back)

            # Create a mask with the overlap. Calculate the exact part of the patch that will be inserted in the 
            # final H5/Zarr file
            p = p[0, self.cfg.DATA.TEST.PADDING[0]:p.shape[1]-self.cfg.DATA.TEST.PADDING[0],
                self.cfg.DATA.TEST.PADDING[1]:p.shape[2]-self.cfg.DATA.TEST.PADDING[1],
                self.cfg.DATA.TEST.PADDING[2]:p.shape[3]-self.cfg.DATA.TEST.PADDING[2]]
            m = np.ones(p.shape, dtype=np.uint8)

            # Put the prediction into queue
            self.output_queue.put([p, m, patch_coords])         

        # Get some auxiliar variables
        self.stats['patch_counter'] = self.extract_info_queue.get(timeout=60)
        if is_main_process():
            z_vol_info = self.extract_info_queue.get(timeout=60)
            list_of_vols_in_z  = self.extract_info_queue.get(timeout=60)
        load_data_process.join()
        output_handle_proc.join()

        # Wait until all threads are done so the main thread can create the full size image 
        if self.cfg.SYSTEM.NUM_GPUS > 1 :
            if self.cfg.TEST.VERBOSE:
                print(f"[Rank {get_rank()} ({os.getpid()})] Finish sample inference ")
            if is_dist_avail_and_initialized():
                dist.barrier()

        # Create the final H5/Zarr file that contains all the individual parts 
        if is_main_process():
            if "C" not in self.cfg.TEST.BY_CHUNKS.INPUT_IMG_AXES_ORDER:
                out_data_order = self.cfg.TEST.BY_CHUNKS.INPUT_IMG_AXES_ORDER + "C"
                c_index = -1
            else:
                out_data_order = self.cfg.TEST.BY_CHUNKS.INPUT_IMG_AXES_ORDER
                c_index = out_data_order.index("C")
                
            if self.cfg.SYSTEM.NUM_GPUS > 1:
                # Obtain parts of the data created by all GPUs
                if self.cfg.TEST.BY_CHUNKS.FORMAT == "h5":
                    data_parts_filenames = sorted(next(os.walk(self.cfg.PATHS.RESULT_DIR.PER_IMAGE))[2])
                else: 
                    data_parts_filenames = sorted(next(os.walk(self.cfg.PATHS.RESULT_DIR.PER_IMAGE))[1])
                parts = []
                mask_parts = []
                for x in data_parts_filenames:
                    if filename+"_part" in x and x.endswith(self.cfg.TEST.BY_CHUNKS.FORMAT):
                        if "_mask" not in x:
                            parts.append(x)
                        else:
                            mask_parts.append(x)
                data_parts_filenames = parts 
                data_parts_mask_filenames = mask_parts
                del parts, mask_parts

                if max(1,self.cfg.SYSTEM.NUM_GPUS) != len(data_parts_filenames) != len(list_of_vols_in_z):
                    raise ValueError("Number of data parts is not the same as number of GPUs")

                # Compose the large image 
                for i, data_part_fname in enumerate(data_parts_filenames):
                    print("Reading {}".format(os.path.join(self.cfg.PATHS.RESULT_DIR.PER_IMAGE, data_part_fname)))
                    data_part_file, data_part = read_chunked_data(os.path.join(self.cfg.PATHS.RESULT_DIR.PER_IMAGE, data_part_fname))
                    data_mask_part_file, data_mask_part = read_chunked_data(os.path.join(self.cfg.PATHS.RESULT_DIR.PER_IMAGE, data_parts_mask_filenames[i]))

                    if 'data' not in locals():
                        all_data_filename = os.path.join(self.cfg.PATHS.RESULT_DIR.PER_IMAGE, filename+ext)
                        if self.cfg.TEST.BY_CHUNKS.FORMAT == "h5":
                            allfile = h5py.File(all_data_filename,'w')
                            data = allfile.create_dataset("data", data_part.shape, dtype=self.dtype_str, compression="gzip")
                        else:
                            allfile = zarr.open_group(all_data_filename, mode="w")
                            data = allfile.create_dataset("data", shape=data_part.shape, dtype=self.dtype_str, compression="gzip")

                    for j, k in enumerate(list_of_vols_in_z[i]):
                        
                        slices = (
                            slice(z_vol_info[k][0],z_vol_info[k][1]), # z (only z axis is distributed across GPUs)
                            slice(None), # y
                            slice(None), # x
                            slice(None), # Channel
                        )
                        
                        data_ordered_slices = order_dimensions(
                            slices,
                            input_order="ZYXC",
                            output_order=out_data_order,
                            default_value=0)

                        if self.cfg.TEST.VERBOSE:
                            print(f"Filling {k} [{z_vol_info[k][0]}:{z_vol_info[k][1]}]")
                        if len(self.cfg.TEST.BY_CHUNKS.INPUT_IMG_AXES_ORDER) == 4 and 'T' not in self.cfg.TEST.BY_CHUNKS.INPUT_IMG_AXES_ORDER:
                            data_ordered_slices = (slice(None, None, None),)+data_ordered_slices
                        data[data_ordered_slices] = data_part[data_ordered_slices] / data_mask_part[data_ordered_slices]

                        if self.cfg.TEST.BY_CHUNKS.FORMAT == "h5":
                            allfile.flush() 

                    if self.cfg.TEST.BY_CHUNKS.FORMAT == "h5":
                        data_part_file.close()
                        data_mask_part_file.close()

                # Save image
                if self.cfg.TEST.BY_CHUNKS.SAVE_OUT_TIF and self.cfg.PATHS.RESULT_DIR.PER_IMAGE != "":
                    current_order = np.array(range(len(data.shape)))
                    transpose_order = order_dimensions(current_order, input_order=out_data_order,
                        output_order="TZYXC", default_value=np.nan)
                    transpose_order = [x for x in transpose_order if not np.isnan(x)]
                    data = np.array(data, dtype=self.dtype).transpose(transpose_order)
                    if "T" not in out_data_order:
                        data = np.expand_dims(data,0)

                    save_tif(data, self.cfg.PATHS.RESULT_DIR.PER_IMAGE, [filename+".tif"], verbose=self.cfg.TEST.VERBOSE)

                if self.cfg.TEST.BY_CHUNKS.FORMAT == "h5":
                    allfile.close()      

            # Just make the division with the overlap
            else:
                # Load predictions and overlapping mask
                pred_file, pred = read_chunked_data(out_data_filename)
                mask_file, mask = read_chunked_data(out_data_mask_filename)

                # Create new file
                if self.cfg.TEST.BY_CHUNKS.FORMAT == "h5":
                    fid_div = h5py.File(out_data_div_filename,'w')
                    pred_div = fid_div.create_dataset("data", pred.shape, dtype=pred.dtype, compression="gzip")
                else:
                    fid_div = zarr.open_group(out_data_div_filename, mode="w")
                    pred_div = fid_div.create_dataset("data", shape=pred.shape, dtype=pred.dtype)
                    
                t_dim, z_dim, c_dim, y_dim, x_dim = order_dimensions(
                    data_shape, self.cfg.TEST.BY_CHUNKS.INPUT_IMG_AXES_ORDER)
                
                # Fill the new data
                z_vols = math.ceil(z_dim/self.cfg.DATA.PATCH_SIZE[0])
                y_vols = math.ceil(y_dim/self.cfg.DATA.PATCH_SIZE[1])
                x_vols = math.ceil(x_dim/self.cfg.DATA.PATCH_SIZE[2])
                for z in tqdm(range(z_vols), disable=not is_main_process()):
                    for y in range(y_vols):
                        for x in range(x_vols):

                            slices = (
                                slice(z*self.cfg.DATA.PATCH_SIZE[0], min(z_dim,self.cfg.DATA.PATCH_SIZE[0]*(z+1))),
                                slice(y*self.cfg.DATA.PATCH_SIZE[1], min(y_dim,self.cfg.DATA.PATCH_SIZE[1]*(y+1))),
                                slice(x*self.cfg.DATA.PATCH_SIZE[2], min(x_dim,self.cfg.DATA.PATCH_SIZE[2]*(x+1))),
                                slice(0,pred.shape[c_index]), # Channel
                            )

                            data_ordered_slices = order_dimensions(
                                slices,
                                input_order = "ZYXC",
                                output_order = out_data_order,
                                default_value = 0,
                                )
                            pred_div[data_ordered_slices] = pred[data_ordered_slices] / mask[data_ordered_slices]

                    if self.cfg.TEST.BY_CHUNKS.FORMAT == "h5":
                        fid_div.flush()

                # Save image
                if self.cfg.TEST.BY_CHUNKS.SAVE_OUT_TIF and self.cfg.PATHS.RESULT_DIR.PER_IMAGE != "":
                    current_order = np.array(range(len(pred_div.shape)))
                    transpose_order = order_dimensions(current_order, input_order=out_data_order,
                        output_order="TZYXC", default_value=np.nan)
                    transpose_order = [x for x in transpose_order if not np.isnan(x)]
                    pred_div = np.array(pred_div, dtype=self.dtype).transpose(transpose_order)
                    if "T" not in out_data_order:
                        pred_div = np.expand_dims(pred_div,0)

                    save_tif(pred_div, self.cfg.PATHS.RESULT_DIR.PER_IMAGE, [os.path.join(self.cfg.PATHS.RESULT_DIR.PER_IMAGE, filename+".tif")],
                        verbose=self.cfg.TEST.VERBOSE)

                if self.cfg.TEST.BY_CHUNKS.FORMAT == "h5":
                    pred_file.close()
                    mask_file.close()
                    fid_div.close()

            if self.cfg.TEST.BY_CHUNKS.WORKFLOW_PROCESS:
                if self.cfg.TEST.BY_CHUNKS.WORKFLOW_PROCESS.TYPE == "chunk_by_chunk":
                    self.after_merge_patches_by_chunks_proccess_patch(out_data_div_filename) 
                else:            
                    self.after_merge_patches_by_chunks_proccess_entire_pred(out_data_div_filename) 
                    
        # Wait until the main thread is done to predict the next sample
        if self.cfg.SYSTEM.NUM_GPUS > 1 :
            if self.cfg.TEST.VERBOSE:
                print(f"[Rank {get_rank()} ({os.getpid()})] Process waiting . . . ")
            if is_dist_avail_and_initialized():
                dist.barrier()
            if self.cfg.TEST.VERBOSE:
                print(f"[Rank {get_rank()} ({os.getpid()})] Synched with main thread. Go for the next sample")

    def process_sample(self, norm):
        """
        Function to process a sample in the inference phase. 

        Parameters
        ----------
        norm : List of dicts
            Normalization used during training. Required to denormalize the predictions of the model.
        """
        # Data channel check
        if self.cfg.DATA.PATCH_SIZE[-1] != self._X.shape[-1]:
            raise ValueError("Channel of the DATA.PATCH_SIZE given {} does not correspond with the loaded image {}. "
                "Please, check the channels of the images!".format(self.cfg.DATA.PATCH_SIZE[-1], self._X.shape[-1]))
                
        #################
        ### PER PATCH ###
        #################
        if not self.cfg.TEST.FULL_IMG or self.cfg.PROBLEM.NDIM == '3D':
            # Reflect data to complete the needed shape
            if self.cfg.DATA.REFLECT_TO_COMPLETE_SHAPE:
                reflected_orig_shape = self._X.shape
                self._X = np.expand_dims(pad_and_reflect(self._X[0], self.cfg.DATA.PATCH_SIZE, verbose=self.cfg.TEST.VERBOSE),0)
                if self.cfg.DATA.TEST.LOAD_GT:
                    self._Y = np.expand_dims(pad_and_reflect(self._Y[0], self.cfg.DATA.PATCH_SIZE, verbose=self.cfg.TEST.VERBOSE),0)

            original_data_shape = self._X.shape
            
            # Crop if necessary
            if self._X.shape[1:-1] != self.cfg.DATA.PATCH_SIZE[:-1]:
                # Copy X to be used later in full image 
                if self.cfg.PROBLEM.NDIM != '3D': 
                    X_original = self._X.copy()

                if self.cfg.DATA.TEST.LOAD_GT and self._X.shape[:-1] != self._Y.shape[:-1]:
                    raise ValueError("Image {} and mask {} differ in shape (without considering the channels, i.e. last dimension)"
                                     .format(self._X.shape,self._Y.shape))

                if self.cfg.PROBLEM.NDIM == '2D':
                    obj = crop_data_with_overlap(self._X, self.cfg.DATA.PATCH_SIZE, data_mask=self._Y, overlap=self.cfg.DATA.TEST.OVERLAP, 
                        padding=self.cfg.DATA.TEST.PADDING, verbose=self.cfg.TEST.VERBOSE)
                    if self.cfg.DATA.TEST.LOAD_GT:
                        self._X, self._Y = obj
                    else:
                        self._X = obj
                    del obj
                else:
                    if self.cfg.TEST.REDUCE_MEMORY:
                        self._X = crop_3D_data_with_overlap(self._X[0], self.cfg.DATA.PATCH_SIZE, overlap=self.cfg.DATA.TEST.OVERLAP, 
                            padding=self.cfg.DATA.TEST.PADDING, verbose=self.cfg.TEST.VERBOSE, 
                            median_padding=self.cfg.DATA.TEST.MEDIAN_PADDING)
                        if self.cfg.DATA.TEST.LOAD_GT:
                            self._Y = crop_3D_data_with_overlap(self._Y[0], self.cfg.DATA.PATCH_SIZE[:-1]+(self._Y.shape[-1],), overlap=self.cfg.DATA.TEST.OVERLAP, 
                                padding=self.cfg.DATA.TEST.PADDING, verbose=self.cfg.TEST.VERBOSE, 
                                median_padding=self.cfg.DATA.TEST.MEDIAN_PADDING)
                    else:
                        if self.cfg.DATA.TEST.LOAD_GT: self._Y = self._Y[0]
                        obj = crop_3D_data_with_overlap(self._X[0], self.cfg.DATA.PATCH_SIZE, data_mask=self._Y, overlap=self.cfg.DATA.TEST.OVERLAP, 
                            padding=self.cfg.DATA.TEST.PADDING, verbose=self.cfg.TEST.VERBOSE, 
                            median_padding=self.cfg.DATA.TEST.MEDIAN_PADDING)
                        if self.cfg.DATA.TEST.LOAD_GT:
                            self._X, self._Y = obj
                        else:
                            self._X = obj
                        del obj

            # Evaluate each patch
            if self.cfg.DATA.TEST.LOAD_GT and self.cfg.TEST.EVALUATE:
                l = int(math.ceil(self._X.shape[0]/self.cfg.TRAIN.BATCH_SIZE))
                for k in tqdm(range(l), leave=False, disable=not is_main_process()):
                    top = (k+1)*self.cfg.TRAIN.BATCH_SIZE if (k+1)*self.cfg.TRAIN.BATCH_SIZE < self._X.shape[0] else self._X.shape[0]
                    with torch.cuda.amp.autocast():
                        output = self.apply_model_activations(self.model_call_func(self._X[k*self.cfg.TRAIN.BATCH_SIZE:top]))
                        loss = self.loss(output, to_pytorch_format(self._Y[k*self.cfg.TRAIN.BATCH_SIZE:top], self.axis_order, self.device, dtype=self.loss_dtype))

                    # Calculate the metrics
                    train_iou = self.metric_calculation(output, to_pytorch_format(self._Y[k*self.cfg.TRAIN.BATCH_SIZE:top], self.axis_order, self.device, dtype=self.loss_dtype))
                    
                    self.stats['loss_per_crop'] += loss.item()
                    self.stats['iou_per_crop'] += train_iou
                    
                del output    

            self.stats['patch_counter'] += self._X.shape[0]

            # Predict each patch
            if self.cfg.TEST.AUGMENTATION:
                for k in tqdm(range(self._X.shape[0]), leave=False, disable=not is_main_process()):
                    if self.cfg.PROBLEM.NDIM == '2D':
                        p = ensemble8_2d_predictions(self._X[k], axis_order_back=self.axis_order_back,
                            pred_func=self.model_call_func, axis_order=self.axis_order, device=self.device)
                    else:
                        p = ensemble16_3d_predictions(self._X[k], batch_size_value=self.cfg.TRAIN.BATCH_SIZE,
                            axis_order_back=self.axis_order_back, pred_func=self.model_call_func, 
                            axis_order=self.axis_order, device=self.device)
                    p = self.apply_model_activations(p)
                    # Multi-head concatenation
                    if isinstance(p, list):
                        p = torch.cat((p[0], p[1]), dim=1)
                    p = to_numpy_format(p, self.axis_order_back)
                    if 'pred' not in locals():
                        pred = np.zeros((self._X.shape[0],)+p.shape[1:], dtype=self.dtype)
                    pred[k] = p
            else:
                l = int(math.ceil(self._X.shape[0]/self.cfg.TRAIN.BATCH_SIZE))
                for k in tqdm(range(l), leave=False, disable=not is_main_process()):
                    top = (k+1)*self.cfg.TRAIN.BATCH_SIZE if (k+1)*self.cfg.TRAIN.BATCH_SIZE < self._X.shape[0] else self._X.shape[0]
                    with torch.cuda.amp.autocast():
                        p = self.apply_model_activations(self.model_call_func(self._X[k*self.cfg.TRAIN.BATCH_SIZE:top]))
                        # Multi-head concatenation
                        if isinstance(p, list):
                            p = torch.cat((p[0], p[1]), dim=1)
                        p = to_numpy_format(p, self.axis_order_back)
                    if 'pred' not in locals():
                        pred = np.zeros((self._X.shape[0],)+p.shape[1:], dtype=self.dtype)
                    pred[k*self.cfg.TRAIN.BATCH_SIZE:top] = p

            # Delete self._X as in 3D there is no full image
            if self.cfg.PROBLEM.NDIM == '3D':
                del self._X, p

            # Reconstruct the predictions
            if original_data_shape[1:-1] != self.cfg.DATA.PATCH_SIZE[:-1]:
                if self.cfg.PROBLEM.NDIM == '3D': original_data_shape = original_data_shape[1:]
                f_name = merge_data_with_overlap if self.cfg.PROBLEM.NDIM == '2D' else merge_3D_data_with_overlap

                if self.cfg.TEST.REDUCE_MEMORY:
                    pred = f_name(pred, original_data_shape[:-1]+(pred.shape[-1],), padding=self.cfg.DATA.TEST.PADDING, 
                        overlap=self.cfg.DATA.TEST.OVERLAP, verbose=self.cfg.TEST.VERBOSE)
                    if self.cfg.DATA.TEST.LOAD_GT:
                        self._Y = f_name(self._Y, original_data_shape[:-1]+(self._Y.shape[-1],), padding=self.cfg.DATA.TEST.PADDING, 
                            overlap=self.cfg.DATA.TEST.OVERLAP, verbose=self.cfg.TEST.VERBOSE)
                else:
                    obj = f_name(pred, original_data_shape[:-1]+(pred.shape[-1],), data_mask=self._Y,
                        padding=self.cfg.DATA.TEST.PADDING, overlap=self.cfg.DATA.TEST.OVERLAP,
                        verbose=self.cfg.TEST.VERBOSE)
                    if self.cfg.DATA.TEST.LOAD_GT:
                        pred, self._Y = obj
                    else:
                        pred = obj
                    del obj
                if self.cfg.PROBLEM.NDIM != '3D': 
                    self._X = X_original.copy()
                    del X_original
            else:
                pred = pred[0]
                if self._Y is not None: self._Y = self._Y[0]

            if self.cfg.DATA.REFLECT_TO_COMPLETE_SHAPE: 
                if self.cfg.PROBLEM.NDIM == '2D':
                    pred = pred[-reflected_orig_shape[1]:,-reflected_orig_shape[2]:]
                    if self._Y is not None:
                        self._Y = self._Y[-reflected_orig_shape[1]:,-reflected_orig_shape[2]:]
                else:
                    pred = pred[-reflected_orig_shape[1]:,-reflected_orig_shape[2]:,-reflected_orig_shape[3]:]
                    if self._Y is not None:
                        self._Y = self._Y[-reflected_orig_shape[1]:,-reflected_orig_shape[2]:,-reflected_orig_shape[3]:]

            # Argmax if needed
            if self.cfg.MODEL.N_CLASSES > 2 and self.cfg.DATA.TEST.ARGMAX_TO_OUTPUT:
                # Multi-head case of instance segmentation
                if pred.shape[-1] > self.cfg.MODEL.N_CLASSES:
                    pred = np.concatenate([pred[...,:-(self.cfg.MODEL.N_CLASSES)], 
                        np.expand_dims(np.argmax(pred[...,-(self.cfg.MODEL.N_CLASSES):],-1), -1)], axis=-1)
                else:
                    pred = np.expand_dims(np.argmax(pred,-1), -1)
                if self.cfg.DATA.TEST.LOAD_GT: self._Y = np.expand_dims(np.argmax(self._Y,-1), -1)

            # Apply mask
            if self.cfg.TEST.POST_PROCESSING.APPLY_MASK:
                pred = apply_binary_mask(pred, self.cfg.DATA.TEST.BINARY_MASKS)

            # Save image
            if self.cfg.PATHS.RESULT_DIR.PER_IMAGE != "":
                save_tif(np.expand_dims(pred,0), self.cfg.PATHS.RESULT_DIR.PER_IMAGE, self.processing_filenames, 
                    verbose=self.cfg.TEST.VERBOSE)

            if self.cfg.DATA.TEST.LOAD_GT and self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS != "Dv2":
                if self._Y.ndim > pred.ndim: self._Y = self._Y[0]
                if self.cfg.LOSS.TYPE != 'MASKED_BCE':
                    _iou_merge_patches = jaccard_index_numpy((self._Y>0.5).astype(np.uint8), (pred>0.5).astype(np.uint8))
                    _ov_iou_merge_patches = voc_calculation((self._Y>0.5).astype(np.uint8), (pred>0.5).astype(np.uint8),
                                                    _iou_merge_patches)
                else:
                    exclusion_mask = self._Y < 2
                    binY = self._Y * exclusion_mask.astype( float )
                    _iou_merge_patches = jaccard_index_numpy((binY>0.5).astype(np.uint8), (pred>0.5).astype(np.uint8))
                    _ov_iou_merge_patches = voc_calculation((binY>0.5).astype(np.uint8), (pred>0.5).astype(np.uint8),
                                                    _iou_merge_patches)
                self.stats['iou_merge_patches'] += _iou_merge_patches
                self.stats['ov_iou_merge_patches'] += _ov_iou_merge_patches

            ############################
            ### POST-PROCESSING (3D) ###
            ############################
            if self.post_processing['per_image']:
                pred, _iou_post, _ov_iou_post = apply_post_processing(self.cfg, pred, self._Y)
                self.stats['iou_merge_patches_post'] += _iou_post
                self.stats['ov_iou_merge_patches_post'] += _ov_iou_post
                if pred.ndim == 4:
                    save_tif(np.expand_dims(pred,0), self.cfg.PATHS.RESULT_DIR.PER_IMAGE_POST_PROCESSING,
                        self.processing_filenames, verbose=self.cfg.TEST.VERBOSE)
                else:
                    save_tif(pred, self.cfg.PATHS.RESULT_DIR.PER_IMAGE_POST_PROCESSING, self.processing_filenames,
                        verbose=self.cfg.TEST.VERBOSE)

            self.after_merge_patches(pred)
            
            if self.cfg.TEST.ANALIZE_2D_IMGS_AS_3D_STACK:
                self.all_pred.append(pred)
                if self.cfg.DATA.TEST.LOAD_GT: self.all_gt.append(self._Y)            

        ##################
        ### FULL IMAGE ###
        ##################
        if self.cfg.TEST.FULL_IMG and self.cfg.PROBLEM.NDIM == '2D':
            self._X, o_test_shape = check_downsample_division(self._X, len(self.cfg.MODEL.FEATURE_MAPS)-1)
            if self.cfg.DATA.TEST.LOAD_GT:
                self._Y, _ = check_downsample_division(self._Y, len(self.cfg.MODEL.FEATURE_MAPS)-1)

            # Evaluate each img
            if self.cfg.DATA.TEST.LOAD_GT:
                with torch.cuda.amp.autocast():
                    output = self.model_call_func(self._X)
                    loss = self.loss(output, to_pytorch_format(self._Y, self.axis_order, self.device, dtype=self.loss_dtype))
                self.stats['loss'] += loss.item()
                del output

            # Make the prediction
            if self.cfg.TEST.AUGMENTATION:
                pred = ensemble8_2d_predictions(self._X[0], axis_order_back=self.axis_order_back, 
                    pred_func=self.model_call_func, axis_order=self.axis_order, device=self.device)
            else:
                with torch.cuda.amp.autocast():
                    pred = self.model_call_func(self._X)
            pred = self.apply_model_activations(pred)
            # Multi-head concatenation
            if isinstance(pred, list):
                pred = torch.cat((pred[0], torch.argmax(pred[1], axis=1).unsqueeze(1)), dim=1)  
            pred = to_numpy_format(pred, self.axis_order_back)  
            if self.cfg.TEST.AUGMENTATION: pred = np.expand_dims(pred, 0)
            del self._X 

            # Recover original shape if padded with check_downsample_division
            pred = pred[:,:o_test_shape[1],:o_test_shape[2]]
            if self.cfg.DATA.TEST.LOAD_GT: self._Y = self._Y[:,:o_test_shape[1],:o_test_shape[2]]

            # Save image
            if pred.ndim == 4 and self.cfg.PROBLEM.NDIM == '3D':
                save_tif(np.expand_dims(pred,0), self.cfg.PATHS.RESULT_DIR.FULL_IMAGE, self.processing_filenames,
                    verbose=self.cfg.TEST.VERBOSE)
            else:
                save_tif(pred, self.cfg.PATHS.RESULT_DIR.FULL_IMAGE, self.processing_filenames, verbose=self.cfg.TEST.VERBOSE)

            # Argmax if needed
            if self.cfg.MODEL.N_CLASSES > 2 and self.cfg.DATA.TEST.ARGMAX_TO_OUTPUT:
                pred = np.expand_dims(np.argmax(pred,-1), -1)
                if self.cfg.DATA.TEST.LOAD_GT: self._Y = np.expand_dims(np.argmax(self._Y,-1), -1)

            if self.cfg.TEST.POST_PROCESSING.APPLY_MASK:
                pred = apply_binary_mask(pred, self.cfg.DATA.TEST.BINARY_MASKS)
                
            if self.cfg.DATA.TEST.LOAD_GT:
                score = jaccard_index_numpy((self._Y>0.5).astype(np.uint8), (pred>0.5).astype(np.uint8))
                self.stats['iou'] += score
                self.stats['ov_iou'] += voc_calculation((self._Y>0.5).astype(np.uint8), (pred>0.5).astype(np.uint8), score)

            if self.cfg.TEST.ANALIZE_2D_IMGS_AS_3D_STACK:
                self.all_pred.append(pred)
                if self.cfg.DATA.TEST.LOAD_GT: self.all_gt.append(self._Y)

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
        self.stats['loss_per_crop'] = self.stats['loss_per_crop'] / self.stats['patch_counter'] if self.stats['patch_counter'] != 0 else 0
        self.stats['iou_per_crop'] = self.stats['iou_per_crop'] / self.stats['patch_counter'] if self.stats['patch_counter'] != 0 else 0

        # Merge patches
        self.stats['iou_merge_patches'] = self.stats['iou_merge_patches'] / image_counter
        self.stats['ov_iou_merge_patches'] = self.stats['ov_iou_merge_patches'] / image_counter

        # Full image
        self.stats['iou'] = self.stats['iou'] / image_counter
        self.stats['loss'] = self.stats['loss'] / image_counter
        self.stats['ov_iou'] = self.stats['ov_iou'] / image_counter

        if self.post_processing['per_image']:
            self.stats['iou_merge_patches_post'] = self.stats['iou_merge_patches_post'] / image_counter
            self.stats['ov_iou_merge_patches_post'] = self.stats['ov_iou_merge_patches_post'] / image_counter
        if self.post_processing['as_3D_stack']:
            self.stats['iou_as_3D_stack_post'] = self.stats['iou_as_3D_stack_post'] / image_counter
            self.stats['ov_iou_as_3D_stack_post'] = self.stats['ov_iou_as_3D_stack_post'] / image_counter
            
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
            if not self.cfg.TEST.FULL_IMG:
                print("Loss (per patch): {}".format(self.stats['loss_per_crop']))
                print("Test Foreground IoU (per patch): {}".format(self.stats['iou_per_crop']))
                print(" ")
                print("Test Foreground IoU (merge patches): {}".format(self.stats['iou_merge_patches']))
                print("Test Overall IoU (merge patches): {}".format(self.stats['ov_iou_merge_patches']))
                print(" ")
            else:
                print("Loss (per image): {}".format(self.stats['loss']))
                print("Test Foreground IoU (per image): {}".format(self.stats['iou']))
                print("Test Overall IoU (per image): {}".format(self.stats['ov_iou']))
                print(" ")

    def print_post_processing_stats(self):
        """
        Print post-processing statistics.
        """
        if self.post_processing['per_image']:
            print("Test Foreground IoU (merge patches - post-processing): {}".format(self.stats['iou_merge_patches_post']))
            print("Test Overall IoU (merge patches - post-processing): {}".format(self.stats['ov_iou_merge_patches_post']))
            print(" ")
        if self.post_processing['as_3D_stack']:
            print("Test Foreground IoU (as 3D stack - post-processing): {}".format(self.stats['iou_as_3D_stack_post']))
            print("Test Overall IoU (as 3D stack - post-processing): {}".format(self.stats['ov_iou_as_3D_stack_post']))
            print(" ")     

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

    def after_merge_patches_by_chunks_proccess_entire_pred(self, filename):
        """
        Place any code that needs to be done after merging all predicted patches into the original image
        but in the process made chunk by chunk. This function will operate over the entire predicted
        image.

        Parameters
        ----------
        filename : List of str
            Filename of the predicted image H5/Zarr.  
        """
        # Load H5/Zarr and convert it into numpy array
        pred_file, pred = read_chunked_data(filename)
        pred = np.squeeze(np.array(pred, dtype=self.dtype))
        if self.cfg.TEST.BY_CHUNKS.FORMAT == "h5":
            pred_file.close()

        # Adjust shape
        if pred.ndim < 3:
            raise ValueError("Read image seems to be 2D: {}. Path: {}".format(pred.shape, filename))
        if pred.ndim == 3: 
            pred = np.expand_dims(pred, -1)
        else:
            min_val = min(pred.shape)
            channel_pos = pred.shape.index(min_val)
            if channel_pos != 3 and pred.shape[channel_pos] <= 4:
                new_pos = [x for x in range(4) if x != channel_pos]+[channel_pos,]
                pred = pred.transpose(new_pos)

        fname, file_extension = os.path.splitext(os.path.basename(filename))
        self.processing_filenames = [fname+".tif"]
        self.after_merge_patches(pred)

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def after_full_image(self, pred):
        """
        Place here any code that must be executed after generating the prediction by supplying the entire image to the model. 
        To enable this, the model should be convolutional, and the image(s) should be in a 2D format. Using 3D images as 
        direct inputs to the model is not feasible due to their large size.
        
        Parameters
        ----------
        pred : Torch Tensor
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
        if self.post_processing['as_3D_stack']:
            self.all_pred = np.concatenate(self.all_pred)
            self.all_gt = np.concatenate(self.all_gt) if self.cfg.DATA.TEST.LOAD_GT else None
            self.all_pred, self.stats['iou_as_3D_stack_post'], self.stats['ov_iou_as_3D_stack_post'] = apply_post_processing(self.cfg, self.all_pred, self.all_gt)
            save_tif(np.expand_dims(self.all_pred,0), self.cfg.PATHS.RESULT_DIR.AS_3D_STACK_POST_PROCESSING, verbose=self.cfg.TEST.VERBOSE)

def extract_patch_from_dataset(data, cfg, input_queue, extract_info_queue, verbose=False):
    """
    Extract patches from data and put them into a queue read by each GPU inference process.
    This function will be run by a child process created for every test sample.  

    Parameters
    ----------
    data : Str or Numpy array
        If str it will be consider a path to load a H5/Zarr file. If not, it will be considered as the 
        data to extract patches from. 

    cfg : YACS configuration
        Running configuration.

    input_queue : Multiprocessing queue 
        Queue to put each extracted patch into.

    extract_info_queue : Multiprocessing queue 
        Auxiliary queue to pass information between processes. 
    
    verbose : bool, optional
        To print useful information for debugging.  
    """
    if verbose and cfg.SYSTEM.NUM_GPUS > 1:
        if isinstance(data, str):
            print(f"[Rank {get_rank()} ({os.getpid()})] In charge of extracting patch from data from {data}")
        else:
            print(f"[Rank {get_rank()} ({os.getpid()})] In charge of extracting patch from data from Numpy array {data.shape}")

    # Load H5/Zarr in case we need it
    if isinstance(data, str):
        data_file, data = read_chunked_data(data)

    # Process of extracting each patch
    patch_counter = 0
    for obj in extract_3D_patch_with_overlap_yield(data, cfg.DATA.PATCH_SIZE, cfg.TEST.BY_CHUNKS.INPUT_IMG_AXES_ORDER,
        overlap=cfg.DATA.TEST.OVERLAP, padding=cfg.DATA.TEST.PADDING, total_ranks=max(1,cfg.SYSTEM.NUM_GPUS), 
        rank=get_rank(), verbose=verbose):

        if is_main_process():
            img, patch_coords, total_vol, z_vol_info, list_of_vols_in_z = obj
        else: 
            img, patch_coords, total_vol = obj

        img = np.expand_dims(img,0)
        input_queue.put([img, patch_coords])

        if patch_counter == 0:
            # This goes for the child process in charge of inserting data patches (insert_patch_into_dataset function)
            extract_info_queue.put(total_vol)
        patch_counter += 1

    # Send a sentinel so the main thread knows that there is no more data
    input_queue.put(None)  

    # Send to the main thread patch_counter
    extract_info_queue.put(patch_counter)
    if is_main_process():
        extract_info_queue.put(z_vol_info)
        extract_info_queue.put(list_of_vols_in_z)

    if verbose and cfg.SYSTEM.NUM_GPUS > 1:
        if isinstance(data, str):
            print(f"[Rank {get_rank()} ({os.getpid()})] Finish extracting patches from data {data}")
        else:
            print(f"[Rank {get_rank()} ({os.getpid()})] Finish extracting patches from data {data.shape}")

    if 'data_file' in locals() and cfg.TEST.BY_CHUNKS.FORMAT == "h5":
        data_file.close()

def insert_patch_into_dataset(data_filename, data_filename_mask, data_shape, output_queue, extract_info_queue, cfg, 
    dtype_str, dtype, file_type, verbose=False):
    """
    Insert predicted patches (in ``output_queue``) in its original position in a H5/Zarr file. Each GPU will create
    a file containing the part it has processed (as we can not write the same H5/Zarr file ar the same time). Then, 
    the main rank will create the final image. This function will be run by a child process created for every 
    test sample.  

    Parameters
    ----------
    data_filename : Str or Numpy array
        If str it will be consider a path to load a H5/Zarr file. If not, it will be considered as the 
        data to extract patches from. 

    data_shape : YACS configuration
        Shape of the H5/Zarr file dataset to create. 

    output_queue : Multiprocessing queue 
        Queue to get each prediction from.

    extract_info_queue : Multiprocessing queue 
        Auxiliary queue to pass information between processes. 
    
    cfg : YACS configuration
        Running configuration.

    dtype_str : str
        Type of the H5/Zarr dataset to create.

    dtype : Numpy dtype
        Type of the H5/Zarr dataset to create. Only used if a TIF file is created by selected to do so
        with ``TEST.BY_CHUNKS.SAVE_OUT_TIF`` variable. 

    verbose : bool, optional
        To print useful information for debugging. 
    """
    if verbose and cfg.SYSTEM.NUM_GPUS > 1:
        print(f"[Rank {get_rank()} ({os.getpid()})] In charge of inserting patches into data . . .")
    
    if file_type == "h5":
        fid = h5py.File(data_filename, "w") 
        fid_mask = h5py.File(data_filename_mask, "w") 
    else:
        fid = zarr.open_group(data_filename, mode="w")
        fid_mask = zarr.open_group(data_filename_mask, mode="w")
      
    filename, file_extension = os.path.splitext(os.path.basename(data_filename))
    
    # Obtain the total patches so we can display it for the user
    total_patches = extract_info_queue.get(timeout=60)
    for i in tqdm(range(total_patches), disable=not is_main_process()):
        p, m, patch_coords = output_queue.get(timeout=60)

        if 'data' not in locals():
            # Channel dimension should be equal to the number of channel of the prediction
            out_data_shape = tuple(data_shape)
            if "C" not in cfg.TEST.BY_CHUNKS.INPUT_IMG_AXES_ORDER:
                out_data_shape = tuple(out_data_shape) + (p.shape[-1],)
                out_data_order = cfg.TEST.BY_CHUNKS.INPUT_IMG_AXES_ORDER + "C"
            else:
                out_data_shape = tuple(out_data_shape[:-1]) + (p.shape[-1],)
                out_data_order = cfg.TEST.BY_CHUNKS.INPUT_IMG_AXES_ORDER

            if file_type == "h5":
                data = fid.create_dataset("data", out_data_shape, dtype=dtype_str, compression="gzip")
                mask = fid_mask.create_dataset("data", out_data_shape, dtype=dtype_str, compression="gzip")
            else:
                data = fid.create_dataset("data", shape=out_data_shape, dtype=dtype_str)
                mask = fid_mask.create_dataset("data", shape=out_data_shape, dtype=dtype_str)

        # Adjust slices to calculate where to insert the predicted patch. This slice does not have into account the 
        # channel so any of them can be inserted 
        slices = (slice(patch_coords[0][0],patch_coords[0][1]),slice(patch_coords[1][0],patch_coords[1][1]),
            slice(patch_coords[2][0],patch_coords[2][1]), slice(None))
        data_ordered_slices = tuple(order_dimensions(slices, input_order="ZYXC", output_order=cfg.TEST.BY_CHUNKS.INPUT_IMG_AXES_ORDER,
            default_value=0))

        # Adjust patch slice to transpose it before inserting intop the final data 
        current_order = np.array(range(len(p.shape)))
        transpose_order = order_dimensions(current_order, input_order="ZYXC", output_order=out_data_order,
            default_value=np.nan)
        transpose_order = [x for x in transpose_order if not np.isnan(x)]

        data[data_ordered_slices] += p.transpose(transpose_order)
        mask[data_ordered_slices] += m.transpose(transpose_order)

        # Force flush after some iterations
        if i % cfg.TEST.BY_CHUNKS.FLUSH_EACH == 0 and file_type == "h5":
            fid.flush() 
            fid_mask.flush() 

    # Save image
    if cfg.TEST.BY_CHUNKS.SAVE_OUT_TIF and cfg.PATHS.RESULT_DIR.PER_IMAGE != "":
        current_order = np.array(range(len(data.shape)))
        transpose_order = order_dimensions(current_order, input_order=out_data_order,
            output_order="TZYXC", default_value=np.nan)
        transpose_order = [x for x in transpose_order if not np.isnan(x)]
        data = np.array(data, dtype=dtype).transpose(transpose_order)
        mask = np.array(mask, dtype=dtype).transpose(transpose_order)
        if "T" not in out_data_order:
            data = np.expand_dims(data,0)
            mask = np.expand_dims(mask,0)
        save_tif(data, cfg.PATHS.RESULT_DIR.PER_IMAGE, [filename+".tif"], verbose=verbose)
        save_tif(mask, cfg.PATHS.RESULT_DIR.PER_IMAGE, [filename+"_mask.tif"], verbose=verbose)
    if file_type == "h5":
        fid.close()        
        fid_mask.close()

    if verbose and cfg.SYSTEM.NUM_GPUS > 1:
        print(f"[Rank {get_rank()} ({os.getpid()})] Finish inserting patches into data . . .")
