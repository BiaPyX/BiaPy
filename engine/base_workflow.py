import math
import os
import datetime
import time
import json
import torch
import numpy as np
from tqdm import tqdm
from abc import ABCMeta, abstractmethod
from sklearn.model_selection import StratifiedKFold

from models import build_model
from engine import prepare_optimizer, build_callbacks
from data.generators import create_train_val_augmentors, create_test_augmentor, check_generator_consistence
from utils.misc import (get_world_size, get_rank, is_main_process, save_model, time_text, load_model_checkpoint, TensorboardLogger,
    to_pytorch_format, to_numpy_format)
from utils.util import load_data_from_dir, load_3d_images_from_dir, create_plots, pad_and_reflect, save_tif, check_downsample_division
from engine.train_engine import train_one_epoch, evaluate
from data.data_2D_manipulation import crop_data_with_overlap, merge_data_with_overlap, load_and_prepare_2D_train_data
from data.data_3D_manipulation import crop_3D_data_with_overlap, merge_3D_data_with_overlap, load_and_prepare_3D_data
from data.post_processing.post_processing import ensemble8_2d_predictions, ensemble16_3d_predictions, apply_binary_mask
from engine.metrics import jaccard_index_numpy, voc_calculation
from data.post_processing import apply_post_processing


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
        self.post_processing['all_images'] = False
        self.test_filenames = None 
        self.metrics = []
        self.data_norm = None
        self.model_prepared = False 

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
        self.stats['iou_per_image'] = 0
        self.stats['ov_iou_per_image'] = 0

        # Full image
        self.stats['loss'] = 0
        self.stats['iou'] = 0
        self.stats['ov_iou'] = 0

        # Post processing
        self.stats['iou_post'] = 0
        self.stats['ov_iou_post'] = 0

        self.world_size = get_world_size()
        self.global_rank = get_rank()
        
        # Test variables
        if self.cfg.TEST.ANALIZE_2D_IMGS_AS_3D_STACK and self.cfg.PROBLEM.NDIM == "2D":
            if self.cfg.TEST.POST_PROCESSING.YZ_FILTERING or self.cfg.TEST.POST_PROCESSING.Z_FILTERING:
                self.post_processing['all_images'] = True
        elif self.cfg.PROBLEM.NDIM == "3D":
            if self.cfg.TEST.POST_PROCESSING.YZ_FILTERING or self.cfg.TEST.POST_PROCESSING.Z_FILTERING:
                self.post_processing['per_image'] = True

        # Define permute shapes to pass from Numpy axis order (Y,X,C) to Pytorch's (C,Y,X)
        self.axis_order = (0,3,1,2) if self.cfg.PROBLEM.NDIM == "2D" else (0,4,1,2,3)
        self.axis_order_back = (0,2,3,1) if self.cfg.PROBLEM.NDIM == "2D" else (0,2,3,4,1)

        # Define metrics
        self.define_metrics()

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
        before inputting it into the model.

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
                objs = f_name(self.cfg.DATA.TRAIN.PATH, self.mask_path, cross_val=self.cfg.DATA.VAL.CROSS_VAL, 
                    cross_val_nsplits=self.cfg.DATA.VAL.CROSS_VAL_NFOLD, cross_val_fold=self.cfg.DATA.VAL.CROSS_VAL_FOLD, 
                    val_split=val_split, seed=self.cfg.SYSTEM.SEED, shuffle_val=self.cfg.DATA.VAL.RANDOM, 
                    random_crops_in_DA=self.cfg.DATA.EXTRACT_RANDOM_PATCH, crop_shape=self.cfg.DATA.PATCH_SIZE, 
                    y_upscaling=self.cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING, ov=self.cfg.DATA.TRAIN.OVERLAP, 
                    padding=self.cfg.DATA.TRAIN.PADDING, minimum_foreground_perc=self.cfg.DATA.TRAIN.MINIMUM_FOREGROUND_PER,
                    reflect_to_complete_shape=self.cfg.DATA.REFLECT_TO_COMPLETE_SHAPE)
            
                if self.cfg.DATA.VAL.FROM_TRAIN:
                    if self.cfg.DATA.VAL.CROSS_VAL:
                        self.X_train, self.Y_train, self.X_val, self.Y_val, self.train_filenames, self.cross_val_samples_ids = objs
                    else:
                        self.X_train, self.Y_train, self.X_val, self.Y_val, self.train_filenames = objs
                else:
                    self.X_train, self.Y_train, self.train_filenames = objs
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
                    self.X_val, _, _ = f_name(self.cfg.DATA.VAL.PATH, crop=True, crop_shape=self.cfg.DATA.PATCH_SIZE,
                                        overlap=self.cfg.DATA.VAL.OVERLAP, padding=self.cfg.DATA.VAL.PADDING,
                                        reflect_to_complete_shape=self.cfg.DATA.REFLECT_TO_COMPLETE_SHAPE)

                    if self.cfg.PROBLEM.NDIM == '2D':
                        crop_shape = (self.cfg.DATA.PATCH_SIZE[0]*self.cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING,
                            self.cfg.DATA.PATCH_SIZE[1]*self.cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING, self.cfg.DATA.PATCH_SIZE[2])
                    else:
                        crop_shape = (self.cfg.DATA.PATCH_SIZE[0], self.cfg.DATA.PATCH_SIZE[1]*self.cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING,
                            self.cfg.DATA.PATCH_SIZE[2]*self.cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING, self.cfg.DATA.PATCH_SIZE[3])
                    if self.load_Y_val:
                        self.Y_val, _, _ = f_name(self.cfg.DATA.VAL.GT_PATH, crop=True, crop_shape=crop_shape,
                                            overlap=self.cfg.DATA.VAL.OVERLAP, padding=self.cfg.DATA.VAL.PADDING,
                                            reflect_to_complete_shape=self.cfg.DATA.REFLECT_TO_COMPLETE_SHAPE,
                                            check_channel=False, check_drange=False)                            
                    else:
                        self.Y_val = None
                    if self.Y_val is not None and len(self.X_val) != len(self.Y_val):
                        raise ValueError("Different number of raw and ground truth items ({} vs {}). "
                            "Please check the data!".format(len(self.X_val), len(self.Y_val)))
                else:
                    self.X_val, self.Y_val = None, None

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
                    
    def prepare_model(self):
        """
        Build the model.
        """
        print("###############")
        print("# Build model #")
        print("###############")
        self.model = build_model(self.cfg, self.job_identifier, self.device)
        self.model_without_ddp = self.model

        if self.args.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.args.gpu], 
                find_unused_parameters=True)
            self.model_without_ddp = self.model.module
        self.model_prepared = True

    def prepare_logging_tool(self):
        """
        Prepare looging tool.
        """
        print("#######################")
        print("# Prepare loggin tool #")
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
        self.prepare_train_generators()
        self.prepare_logging_tool()
        self.early_stopping = build_callbacks(self.cfg)
        self.prepare_model()
        self.optimizer, self.lr_scheduler, self.loss_scaler = prepare_optimizer(self.cfg, self.model_without_ddp, 
            len(self.train_generator))
 
        # Load checkpoint if necessary
        if self.cfg.MODEL.LOAD_CHECKPOINT:
            self.start_epoch = load_model_checkpoint(cfg=self.cfg, jobname=self.job_identifier, model_without_ddp=self.model_without_ddp,
                    optimizer=self.optimizer, loss_scaler=self.loss_scaler)
        else:
            self.start_epoch = 0      

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
            train_stats = train_one_epoch(self.cfg, model=self.model, loss_function=self.loss, activations=self.apply_model_activations, 
                metric_function=self.metric_calculation, prepare_targets=self.prepare_targets, data_loader=self.train_generator, 
                optimizer=self.optimizer, device=self.device, loss_scaler=self.loss_scaler, epoch=epoch, log_writer=self.log_writer, 
                lr_scheduler=self.lr_scheduler, start_steps=epoch * self.num_training_steps_per_epoch, axis_order=self.axis_order)

            # Save checkpoint
            if self.cfg.MODEL.SAVE_CKPT_FREQ != -1:
                if (epoch + 1) % self.cfg.MODEL.SAVE_CKPT_FREQ == 0 or epoch + 1 == self.cfg.TRAIN.EPOCHS and is_main_process():
                    save_model(cfg=self.cfg, jobname=self.job_identifier, model=self.model, model_without_ddp=self.model_without_ddp, 
                        optimizer=self.optimizer, loss_scaler=self.loss_scaler, epoch=epoch+1)

            # Apply warmup cosine decay scheduler
            if epoch % self.cfg.TRAIN.ACCUM_ITER == 0 and self.cfg.TRAIN.LR_SCHEDULER.NAME == 'warmupcosine':
                self.lr_scheduler.adjust_learning_rate(self.optimizer, epoch / len(self.train_generator) + epoch)
                
            # Validation
            if self.val_generator is not None:
                test_stats = evaluate(self.cfg, model=self.model, loss_function=self.loss, activations=self.apply_model_activations,
                    metric_function=self.metric_calculation, prepare_targets=self.prepare_targets, epoch=epoch, 
                    data_loader=self.val_generator, device=self.device, lr_scheduler=self.lr_scheduler, axis_order=self.axis_order)

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

                if self.val_generator is not None:
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
                print("Val {}: {}".format(self.metric_names[i], val_best_metric))

        print('Finished Training')

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
                    self.X_test, _, _ = f_name(self.cfg.DATA.TEST.PATH)
                    if self.cfg.DATA.TEST.LOAD_GT:
                        print("3) Loading test masks . . .")
                        self.Y_test, _, _ = f_name(self.cfg.DATA.TEST.GT_PATH, check_channel=False, check_drange=False)
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
        for key, value in self.activations.items():
            # Ignore CE_Sigmoid as torch.nn.BCEWithLogitsLoss will apply Sigmoid automatically in a way 
            # that is more stable numerically (ref: https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
            if (training and value not in ["Linear", "CE_Sigmoid"]) or (not training and value != "Linear"):
                value = "Sigmoid" if value == "CE_Sigmoid" else value
                act = getattr(torch.nn, value)()
                if key == ':':
                    pred = act(pred)
                else:
                    pred[:,int(key),...] = act(pred[:,int(key),...])
        return pred

    @torch.no_grad()
    def test(self):
        """
        Test/Inference step.
        """
        self.load_test_data()
        self.prepare_test_generators()
        if not self.model_prepared:
            self.prepare_model()

        # Load checkpoint
        self.start_epoch = load_model_checkpoint(cfg=self.cfg, jobname=self.job_identifier, model_without_ddp=self.model_without_ddp)
        if self.start_epoch == -1:
            raise ValueError("There was a problem loading the checkpoint. Test phase aborted!")

        image_counter = 0
        
        print("###############")
        print("#  INFERENCE  #")
        print("###############")
        print("Making predictions on test data . . .")
        # Process all the images
        for i, batch in tqdm(enumerate(self.test_generator), total=len(self.test_generator)):
            if self.cfg.DATA.TEST.LOAD_GT or \
                (self.cfg.PROBLEM.TYPE == 'SELF_SUPERVISED' and self.cfg.PROBLEM.SELF_SUPERVISED.PRETEXT_TASK != 'masking'):
                X, X_norm, Y, Y_norm = batch
            else:
                X, X_norm = batch
                Y, Y_norm = None, None
            del batch

            # Process all the images in the batch, sample by sample
            l_X = len(X)
            for j in tqdm(range(l_X), leave=False):
                print("Processing image(s): {}".format(self.test_filenames[(i*l_X)+j:(i*l_X)+j+1]))

                if self.cfg.PROBLEM.TYPE != 'CLASSIFICATION':
                    if type(X) is tuple:
                        self._X = X[j]
                        if self.cfg.DATA.TEST.LOAD_GT or \
                            (self.cfg.PROBLEM.TYPE == 'SELF_SUPERVISED' and self.cfg.PROBLEM.SELF_SUPERVISED.PRETEXT_TASK != 'masking'):
                            self._Y = Y[j]  
                        else:
                            self._Y = None
                    else:
                        self._X = np.expand_dims(X[j],0)
                        if self.cfg.DATA.TEST.LOAD_GT or \
                            (self.cfg.PROBLEM.TYPE == 'SELF_SUPERVISED' and self.cfg.PROBLEM.SELF_SUPERVISED.PRETEXT_TASK != 'masking'):
                            self._Y = np.expand_dims(Y[j],0)  
                        else:
                            self._Y = None
                else:
                    self._X = np.expand_dims(X[j], 0)                    
                    self._Y = np.expand_dims(Y, 0) if self.cfg.DATA.TEST.LOAD_GT else None

                # Process each image separately
                self.f_numbers = list(range((i*l_X)+j,(i*l_X)+j+1)) 
                self.process_sample(self.test_filenames[(i*l_X)+j:(i*l_X)+j+1], norm=(X_norm, Y_norm))

                image_counter += 1

        self.destroy_test_data()

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

    def process_sample(self, filenames, norm):
        """
        Function to process a sample in the inference phase. 

        Parameters
        ----------
        filenames : List of str
            Filenames fo the samples to process. 

        norm : List of dicts
            Normalization used during training. Required to denormalize the predictions of the model.
        """
        #################
        ### PER PATCH ###
        #################
        if self.cfg.TEST.STATS.PER_PATCH:
            # Reflect data to complete the needed shape
            if self.cfg.DATA.REFLECT_TO_COMPLETE_SHAPE:
                reflected_orig_shape = self._X.shape
                self._X = np.expand_dims(pad_and_reflect(self._X[0], self.cfg.DATA.PATCH_SIZE, verbose=self.cfg.TEST.VERBOSE),0)
                if self.cfg.DATA.TEST.LOAD_GT:
                    self._Y = np.expand_dims(pad_and_reflect(self._Y[0], self.cfg.DATA.PATCH_SIZE, verbose=self.cfg.TEST.VERBOSE),0)

            original_data_shape = self._X.shape
            
            # Crop if necessary
            if self._X.shape[1:-1] != self.cfg.DATA.PATCH_SIZE[:-1] or any(x == 0 for x in self.cfg.DATA.TEST.PADDING)\
                or any(x == 0 for x in self.cfg.DATA.TEST.OVERLAP):
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
                self._X = to_pytorch_format(self._X, self.axis_order, self.device)
                self._Y = to_pytorch_format(self._Y, self.axis_order, self.device)
                l = int(math.ceil(self._X.shape[0]/self.cfg.TRAIN.BATCH_SIZE))
                for k in tqdm(range(l), leave=False):
                    top = (k+1)*self.cfg.TRAIN.BATCH_SIZE if (k+1)*self.cfg.TRAIN.BATCH_SIZE < self._X.shape[0] else self._X.shape[0]
                    with torch.cuda.amp.autocast():
                        output = self.apply_model_activations(self.model(self._X[k*self.cfg.TRAIN.BATCH_SIZE:top]))
                        loss = self.loss(output, self._Y[k*self.cfg.TRAIN.BATCH_SIZE:top])

                    # Calculate the metrics
                    train_iou = self.metric_calculation(output, self._Y[k*self.cfg.TRAIN.BATCH_SIZE:top])
                    
                    self.stats['loss_per_crop'] += loss.item()
                    self.stats['iou_per_crop'] += train_iou
                    
                del output    
                # Restore array and axis order
                self._Y = to_numpy_format(self._Y, self.axis_order_back)
                self._X = to_numpy_format(self._X, self.axis_order_back)

            self.stats['patch_counter'] += self._X.shape[0]

            # Predict each patch
            if self.cfg.TEST.AUGMENTATION:
                for k in tqdm(range(self._X.shape[0]), leave=False):
                    if self.cfg.PROBLEM.NDIM == '2D':
                        p = ensemble8_2d_predictions(self._X[k], n_classes=self.cfg.MODEL.N_CLASSES,
                            pred_func=(
                                lambda img_batch_subdiv: 
                                    to_numpy_format(
                                        self.apply_model_activations(
                                            self.model(to_pytorch_format(img_batch_subdiv, self.axis_order, self.device)),
                                            ), 
                                        self.axis_order_back
                                    )
                            )
                        )
                    else:
                        p = ensemble16_3d_predictions(self._X[k], batch_size_value=self.cfg.TRAIN.BATCH_SIZE,
                            pred_func=(
                                lambda img_batch_subdiv: 
                                    to_numpy_format(
                                        self.apply_model_activations(
                                            self.model(to_pytorch_format(img_batch_subdiv, self.axis_order, self.device)),
                                            ), 
                                        self.axis_order_back
                                    )
                            )
                        )
                    if 'pred' not in locals():
                        pred = np.zeros((self._X.shape[0],)+p.shape, dtype=np.float32)
                    pred[k] = p
            else:
                self._X = to_pytorch_format(self._X, self.axis_order, self.device)
                l = int(math.ceil(self._X.shape[0]/self.cfg.TRAIN.BATCH_SIZE))
                for k in tqdm(range(l), leave=False):
                    top = (k+1)*self.cfg.TRAIN.BATCH_SIZE if (k+1)*self.cfg.TRAIN.BATCH_SIZE < self._X.shape[0] else self._X.shape[0]
                    with torch.cuda.amp.autocast():
                        p = self.model(self._X[k*self.cfg.TRAIN.BATCH_SIZE:top])
                        p = to_numpy_format(self.apply_model_activations(p), self.axis_order_back)
                    if 'pred' not in locals():
                        pred = np.zeros((self._X.shape[0],)+p.shape[1:], dtype=np.float32)
                    pred[k*self.cfg.TRAIN.BATCH_SIZE:top] = p

                # Restore array and axis order
                self._X = to_numpy_format(self._X, self.axis_order_back)

            # Delete self._X as in 3D there is no full image
            if self.cfg.PROBLEM.NDIM == '3D':
                del self._X, p

            # Reconstruct the predictions
            if original_data_shape[1:-1] != self.cfg.DATA.PATCH_SIZE[:-1] or any(x == 0 for x in self.cfg.DATA.TEST.PADDING)\
                or any(x == 0 for x in self.cfg.DATA.TEST.OVERLAP):
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

            if self.cfg.DATA.REFLECT_TO_COMPLETE_SHAPE and self.cfg.PROBLEM.NDIM == '3D':
                pred = pred[-reflected_orig_shape[1]:,-reflected_orig_shape[2]:,-reflected_orig_shape[3]:]
                if self._Y is not None:
                    self._Y = self._Y[-reflected_orig_shape[1]:,-reflected_orig_shape[2]:,-reflected_orig_shape[3]:]

            # Argmax if needed
            if self.cfg.MODEL.N_CLASSES > 2 and self.cfg.DATA.TEST.ARGMAX_TO_OUTPUT:
                pred = np.expand_dims(np.argmax(pred,-1), -1)
                if self.cfg.DATA.TEST.LOAD_GT: self._Y = np.expand_dims(np.argmax(self._Y,-1), -1)

            # Apply mask
            if self.cfg.TEST.POST_PROCESSING.APPLY_MASK:
                pred = apply_binary_mask(pred, self.cfg.DATA.TEST.BINARY_MASKS)

            # Save image
            if self.cfg.PATHS.RESULT_DIR.PER_IMAGE != "":
                save_tif(np.expand_dims(pred,0), self.cfg.PATHS.RESULT_DIR.PER_IMAGE, filenames, verbose=self.cfg.TEST.VERBOSE)

            #########################
            ### MERGE PATCH STATS ###
            #########################
            if self.cfg.TEST.STATS.MERGE_PATCHES:
                if self.cfg.DATA.TEST.LOAD_GT and self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS != "Dv2":
                    if self._Y.ndim > pred.ndim: self._Y = self._Y[0]
                    if self.cfg.LOSS.TYPE != 'MASKED_BCE':
                        _iou_per_image = jaccard_index_numpy((self._Y>0.5).astype(np.uint8), (pred>0.5).astype(np.uint8))
                        _ov_iou_per_image = voc_calculation((self._Y>0.5).astype(np.uint8), (pred>0.5).astype(np.uint8),
                                                        _iou_per_image)
                    else:
                        exclusion_mask = self._Y < 2
                        binY = self._Y * exclusion_mask.astype( float )
                        _iou_per_image = jaccard_index_numpy((binY>0.5).astype(np.uint8), (pred>0.5).astype(np.uint8))
                        _ov_iou_per_image = voc_calculation((binY>0.5).astype(np.uint8), (pred>0.5).astype(np.uint8),
                                                        _iou_per_image)
                    self.stats['iou_per_image'] += _iou_per_image
                    self.stats['ov_iou_per_image'] += _ov_iou_per_image

                ############################
                ### POST-PROCESSING (3D) ###
                ############################
                if self.post_processing['per_image']:
                    pred, _iou_post, _ov_iou_post = apply_post_processing(self.cfg, pred, self._Y)
                    self.stats['iou_post'] += _iou_post
                    self.stats['ov_iou_post'] += _ov_iou_post
                    if pred.ndim == 4:
                        save_tif(np.expand_dims(pred,0), self.cfg.PATHS.RESULT_DIR.PER_IMAGE_POST_PROCESSING,
                                    filenames, verbose=self.cfg.TEST.VERBOSE)
                    else:
                        save_tif(pred, self.cfg.PATHS.RESULT_DIR.PER_IMAGE_POST_PROCESSING, filenames,
                                    verbose=self.cfg.TEST.VERBOSE)

            self.after_merge_patches(pred, filenames)
            
            if not self.cfg.TEST.STATS.FULL_IMG and self.cfg.TEST.ANALIZE_2D_IMGS_AS_3D_STACK:
                self.all_pred.append(pred)
                if self.cfg.DATA.TEST.LOAD_GT: self.all_gt.append(self._Y)            

        ##################
        ### FULL IMAGE ###
        ##################
        if self.cfg.TEST.STATS.FULL_IMG and self.cfg.PROBLEM.NDIM == '2D':
            self._X, o_test_shape = check_downsample_division(self._X, len(self.cfg.MODEL.FEATURE_MAPS)-1)
            if self.cfg.DATA.TEST.LOAD_GT:
                self._Y, _ = check_downsample_division(self._Y, len(self.cfg.MODEL.FEATURE_MAPS)-1)

            # Evaluate each img
            self._X = to_pytorch_format(self._X, self.axis_order, self.device)
            if self.cfg.DATA.TEST.LOAD_GT:
                self._Y = to_pytorch_format(self._Y, self.axis_order, self.device)
                with torch.cuda.amp.autocast():
                    output = self.model(self._X)
                    loss = self.loss(output, self._Y)
                self.stats['loss'] += loss.item()
                del output
            self._X = to_numpy_format(self._X, self.axis_order_back)

            # Make the prediction
            if self.cfg.TEST.AUGMENTATION:
                pred = ensemble8_2d_predictions(
                    self._X[0],  
                    pred_func=(
                            lambda img_batch_subdiv: 
                            self.model(
                                to_numpy_format(self.apply_model_activations(img_batch_subdiv), self.axis_order_back)
                            )
                        ),
                    n_classes=self.cfg.MODEL.N_CLASSES)
                pred = np.expand_dims(pred, 0)
            else:
                self._X = to_pytorch_format(self._X, self.axis_order, self.device)
                with torch.cuda.amp.autocast():
                    pred = self.model(self._X)
                pred = to_numpy_format(self.apply_model_activations(pred), self.axis_order_back)    
            del self._X 

            # Restore array and axis order
            self._Y = to_numpy_format(self._Y, self.axis_order_back)

            # Recover original shape if padded with check_downsample_division
            pred = pred[:,:o_test_shape[1],:o_test_shape[2]]
            if self.cfg.DATA.TEST.LOAD_GT: self._Y = self._Y[:,:o_test_shape[1],:o_test_shape[2]]

            # Save image
            if pred.ndim == 4 and self.cfg.PROBLEM.NDIM == '3D':
                save_tif(np.expand_dims(pred,0), self.cfg.PATHS.RESULT_DIR.FULL_IMAGE, filenames,
                            verbose=self.cfg.TEST.VERBOSE)
            else:
                save_tif(pred, self.cfg.PATHS.RESULT_DIR.FULL_IMAGE, filenames, verbose=self.cfg.TEST.VERBOSE)

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

            self.after_full_image(pred, filenames)

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
        self.stats['iou_per_image'] = self.stats['iou_per_image'] / image_counter
        self.stats['ov_iou_per_image'] = self.stats['ov_iou_per_image'] / image_counter

        # Full image
        self.stats['iou'] = self.stats['iou'] / image_counter
        self.stats['loss'] = self.stats['loss'] / image_counter
        self.stats['ov_iou'] = self.stats['ov_iou'] / image_counter

        if self.post_processing['per_image'] or self.post_processing['all_images']:
            self.stats['iou_post'] = self.stats['iou_post'] / image_counter
            self.stats['ov_iou_post'] = self.stats['ov_iou_post'] / image_counter

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
            if self.cfg.TEST.STATS.PER_PATCH:
                print("Loss (per patch): {}".format(self.stats['loss_per_crop']))
                print("Test Foreground IoU (per patch): {}".format(self.stats['iou_per_crop']))
                print(" ")
                if self.cfg.TEST.STATS.MERGE_PATCHES:
                    print("Test Foreground IoU (merge patches): {}".format(self.stats['iou_per_image']))
                    print("Test Overall IoU (merge patches): {}".format(self.stats['ov_iou_per_image']))
                    print(" ")
            if self.cfg.TEST.STATS.FULL_IMG:
                print("Loss (per image): {}".format(self.stats['loss']))
                print("Test Foreground IoU (per image): {}".format(self.stats['iou']))
                print("Test Overall IoU (per image): {}".format(self.stats['ov_iou']))
                print(" ")

    def print_post_processing_stats(self):
        """
        Print post-processing statistics.
        """
        if self.post_processing['per_image'] or self.post_processing['all_images']:
            print("Test Foreground IoU (post-processing): {}".format(self.stats['iou_post']))
            print("Test Overall IoU (post-processing): {}".format(self.stats['ov_iou_post']))
            print(" ")


    @abstractmethod
    def after_merge_patches(self, pred, filenames):
        """
        Place any code that needs to be done after merging all predicted patches into the original image.

        Parameters
        ----------
        pred : Torch Tensor
            Model prediction.

        filenames : List of str
            Filenames of the predicted images.  
        """
        raise NotImplementedError

    @abstractmethod
    def after_full_image(self, pred, filenames):
        """
        Place here any code that must be executed after generating the prediction by supplying the entire image to the model. 
        To enable this, the model should be convolutional, and the image(s) should be in a 2D format. Using 3D images as 
        direct inputs to the model is not feasible due to their large size.
        
        Parameters
        ----------
        pred : Torch Tensor
            Model prediction.

        filenames : List of str
            Filenames of the predicted images.  
        """
        raise NotImplementedError

    def after_all_images(self):
        """
        Place here any code that must be done after predicting all images. 
        """
        ############################
        ### POST-PROCESSING (2D) ###
        ############################
        if self.post_processing['all_images']:
            self.all_pred = np.concatenate(self.all_pred)
            self.all_gt = np.concatenate(self.all_gt) if self.cfg.DATA.TEST.LOAD_GT else None
            self.all_pred, self.stats['iou_post'], self.stats['ov_iou_post'] = apply_post_processing(self.cfg, self.all_pred, self.all_gt)
            save_tif(np.expand_dims(self.all_pred,0), self.cfg.PATHS.RESULT_DIR.AS_3D_STACK_POST_PROCESSING, verbose=self.cfg.TEST.VERBOSE)

