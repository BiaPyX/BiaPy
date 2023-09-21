import os
import torch
import math
import numpy as np
from tqdm import tqdm
from torchmetrics.image import PeakSignalNoiseRatio

from data.data_2D_manipulation import crop_data_with_overlap, merge_data_with_overlap
from data.data_3D_manipulation import crop_3D_data_with_overlap, merge_3D_data_with_overlap
from data.post_processing.post_processing import ensemble8_2d_predictions, ensemble16_3d_predictions
from utils.util import save_tif
from utils.misc import to_pytorch_format, to_numpy_format
from engine.base_workflow import Base_Workflow
from data.pre_processing import create_ssl_source_data_masks, denormalize
from engine.metrics import MaskedAutoencoderViT_loss

class Self_supervised_Workflow(Base_Workflow):
    def __init__(self, cfg, job_identifier, device, rank, **kwargs):
        super(Self_supervised_Workflow, self).__init__(cfg, job_identifier, device, rank, **kwargs)
        
        self.stats['psnr_per_image'] = 0

        self.prepare_ssl_data()

        # From now on, no modification of the cfg will be allowed
        self.cfg.freeze()

        # Activations for each output channel:
        # channel number : 'activation'
        self.activations = {':': 'Linear'}

        # Workflow specific training variables
        self.mask_path = None
        if cfg.PROBLEM.SELF_SUPERVISED.PRETEXT_TASK == 'masking':
            self.load_Y_val = False
        else:
            self.load_Y_val = True

    def prepare_targets(self, targets, batch):
        # Swap with original images so we can calculate PSNR metric afterwards
        return batch

    def define_metrics(self):
        self.metrics = [PeakSignalNoiseRatio()]
        self.metric_names = ["PSNR"]
        if self.cfg.MODEL.ARCHITECTURE == 'mae':
            print("Overriding 'LOSS.TYPE' to set it to MSE loss (masking patches)")
            self.loss = self.MaskedAutoencoderViT_loss_wrapper
        else:
            print("Overriding 'LOSS.TYPE' to set it to L1 loss")
            self.loss = torch.nn.L1Loss()

    def MaskedAutoencoderViT_loss_wrapper(self, output, targets):
        # Targets not used because the loss has been already calculated
        loss, pred, mask = output
        return loss

    def metric_calculation(self, output, targets, device=None, metric_logger=None):
        # Calculate PSNR 
        _, pred, _ = output
        pred = self.model_without_ddp.unpatchify(pred).to(torch.float32).detach().cpu()
        targets = targets.to(torch.float32).detach().cpu()
        with torch.no_grad():
            train_psnr = self.metrics[0](pred, targets)
            train_psnr = train_psnr.item() if not torch.isnan(train_psnr) else 0
            if metric_logger is not None:
                metric_logger.meters[self.metric_names[0]].update(train_psnr)
            else:
                return train_psnr

    def process_sample(self, filenames, norm): 
        original_data_shape = self._X.shape
    
        # Crop if necessary
        if self._X.shape[1:-1] != self.cfg.DATA.PATCH_SIZE[:-1]:
            if self.cfg.PROBLEM.NDIM == '2D':
                self._X, self._Y = crop_data_with_overlap(self._X, self.cfg.DATA.PATCH_SIZE, data_mask=self._Y, overlap=self.cfg.DATA.TEST.OVERLAP, 
                    padding=self.cfg.DATA.TEST.PADDING, verbose=self.cfg.TEST.VERBOSE)
            else:
                self._Y = self._Y[0]
                if self.cfg.TEST.REDUCE_MEMORY:
                    self._X = crop_3D_data_with_overlap(self._X[0], self.cfg.DATA.PATCH_SIZE, overlap=self.cfg.DATA.TEST.OVERLAP, 
                        padding=self.cfg.DATA.TEST.PADDING, verbose=self.cfg.TEST.VERBOSE, 
                        median_padding=self.cfg.DATA.TEST.MEDIAN_PADDING)
                    self._Y = crop_3D_data_with_overlap(self._Y, self.cfg.DATA.PATCH_SIZE, overlap=self.cfg.DATA.TEST.OVERLAP, 
                        padding=self.cfg.DATA.TEST.PADDING, verbose=self.cfg.TEST.VERBOSE, 
                        median_padding=self.cfg.DATA.TEST.MEDIAN_PADDING)
                else:
                    self._X, self._Y = crop_3D_data_with_overlap(self._X[0], self.cfg.DATA.PATCH_SIZE, data_mask=self._Y, overlap=self.cfg.DATA.TEST.OVERLAP, 
                        padding=self.cfg.DATA.TEST.PADDING, verbose=self.cfg.TEST.VERBOSE, 
                        median_padding=self.cfg.DATA.TEST.MEDIAN_PADDING)

        # Predict each patch
        pred = []
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
                    p = ensemble16_3d_predictions(self._X[k], batch_size_value=1,
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
                pred.append(p)
        else:
            self._X = to_pytorch_format(self._X, self.axis_order, self.device)
            l = int(math.ceil(self._X.shape[0]/self.cfg.TRAIN.BATCH_SIZE))
            for k in tqdm(range(l), leave=False):
                top = (k+1)*self.cfg.TRAIN.BATCH_SIZE if (k+1)*self.cfg.TRAIN.BATCH_SIZE < self._X.shape[0] else self._X.shape[0]                
                with torch.cuda.amp.autocast():
                    p = self.model(self._X[k*self.cfg.TRAIN.BATCH_SIZE:top])
                    p = to_numpy_format(self.apply_model_activations(p), self.axis_order_back)
                pred.append(p)
        del self._X, p

        # Reconstruct the predictions
        pred = np.concatenate(pred)
        if original_data_shape[1:-1] != self.cfg.DATA.PATCH_SIZE[:-1]:
            if self.cfg.PROBLEM.NDIM == '3D': original_data_shape = original_data_shape[1:]
            f_name = merge_data_with_overlap if self.cfg.PROBLEM.NDIM == '2D' else merge_3D_data_with_overlap

            if self.cfg.TEST.REDUCE_MEMORY:
                pred = f_name(pred, original_data_shape[:-1]+(pred.shape[-1],), padding=self.cfg.DATA.TEST.PADDING, 
                    overlap=self.cfg.DATA.TEST.OVERLAP, verbose=self.cfg.TEST.VERBOSE)
                self._Y = f_name(self._Y, original_data_shape[:-1]+(self._Y.shape[-1],), padding=self.cfg.DATA.TEST.PADDING, 
                    overlap=self.cfg.DATA.TEST.OVERLAP, verbose=self.cfg.TEST.VERBOSE)
            else:
                pred, self._Y = f_name(pred, original_data_shape[:-1]+(pred.shape[-1],), data_mask=self._Y,
                    padding=self.cfg.DATA.TEST.PADDING, overlap=self.cfg.DATA.TEST.OVERLAP,
                    verbose=self.cfg.TEST.VERBOSE)
        else:
            pred = pred[0]

        # Undo normalization
        x_norm = norm[0]
        if x_norm['type'] == 'div':
            pred = pred*255
            if 'reduced_uint16' in x_norm:
                pred = (pred*65535).astype(np.uint16)
        else:
            pred = denormalize(pred, x_norm['mean'], x_norm['std'])  
            
        # Save image
        if self.cfg.PATHS.RESULT_DIR.PER_IMAGE != "":
            save_tif(np.expand_dims(pred,0), self.cfg.PATHS.RESULT_DIR.PER_IMAGE, filenames, verbose=self.cfg.TEST.VERBOSE)
    
        # Calculate PSNR
        psnr_per_image = self.metrics[0](torch.from_numpy(pred), torch.from_numpy(self._Y))
        self.stats['psnr_per_image'] += psnr_per_image

    def after_merge_patches(self, pred,filenames):
        pass

    def after_full_image(self, pred, filenames):
        pass

    def after_all_images(self):
        pass
    
    def normalize_stats(self, image_counter):
        self.stats['psnr_per_image'] = self.stats['psnr_per_image'] / image_counter

    def print_stats(self, image_counter):
        self.normalize_stats(image_counter)

        if self.cfg.DATA.TEST.LOAD_GT or self.cfg.DATA.TEST.USE_VAL_AS_TEST:
            print("Test PSNR (merge patches): {}".format(self.stats['psnr_per_image']))
            print(" ")


    def prepare_ssl_data(self):
        if self.cfg.PROBLEM.SELF_SUPERVISED.PRETEXT_TASK == "masking":
            print("No SSL data needs to be prepared for masking, as it will be generated on the fly")
            return

        print("############################")
        print("#  PREPARE DETECTION DATA  #")
        print("############################")

        # Create selected channels for train data
        if self.cfg.TRAIN.ENABLE:
            create_mask = False
            if not os.path.isdir(self.cfg.DATA.TRAIN.SSL_SOURCE_DIR):
                print("You select to create detection masks from given .csv files but no file is detected in {}. "
                    "So let's prepare the data. Notice that, if you do not modify 'DATA.TRAIN.SSL_SOURCE_DIR' "
                    "path, this process will be done just once!".format(self.cfg.DATA.TRAIN.SSL_SOURCE_DIR))
                create_mask = True
            else:
                if len(next(os.walk(self.cfg.DATA.TRAIN.SSL_SOURCE_DIR))[2]) != len(next(os.walk(self.cfg.DATA.TRAIN.PATH))[2]):
                    print("Different number of files found in {} and {}. Trying to create the the rest again"
                        .format(self.cfg.DATA.TRAIN.GT_PATH, self.cfg.DATA.TRAIN.SSL_SOURCE_DIR))
                    create_mask = True 
                else:
                    print("Train source data found in {}".format(self.cfg.DATA.TRAIN.SSL_SOURCE_DIR))   
            if create_mask:
                create_ssl_source_data_masks(self.cfg, data_type='train')

        # Create selected channels for val data
        if self.cfg.TRAIN.ENABLE and not self.cfg.DATA.VAL.FROM_TRAIN:
            create_mask = False
            if not os.path.isdir(self.cfg.DATA.VAL.SSL_SOURCE_DIR):
                print("You select to create detection masks from given .csv files but no file is detected in {}. "
                    "So let's prepare the data. Notice that, if you do not modify 'DATA.VAL.SSL_SOURCE_DIR' "
                    "path, this process will be done just once!".format(self.cfg.DATA.VAL.SSL_SOURCE_DIR))
                create_mask = True
            else:
                if len(next(os.walk(self.cfg.DATA.VAL.SSL_SOURCE_DIR))[2]) != len(next(os.walk(self.cfg.DATA.VAL.PATH))[2]):
                    print("Different number of files found in {} and {}. Trying to create the the rest again"
                        .format(self.cfg.DATA.VAL.GT_PATH, self.cfg.DATA.VAL.SSL_SOURCE_DIR))
                    create_mask = True   
                else:
                    print("Validation source data found in {}".format(self.cfg.DATA.VAL.SSL_SOURCE_DIR)) 
            if create_mask:         
                create_ssl_source_data_masks(self.cfg, data_type='val')

        # Create selected channels for test data
        if self.cfg.TEST.ENABLE:
            create_mask = False
            if not os.path.isdir(self.cfg.DATA.TEST.SSL_SOURCE_DIR):
                print("You select to create detection masks from given .csv files but no file is detected in {}. "
                    "So let's prepare the data. Notice that, if you do not modify 'DATA.TEST.SSL_SOURCE_DIR' "
                    "path, this process will be done just once!".format(self.cfg.DATA.TEST.SSL_SOURCE_DIR))
                create_mask = True
            else:
                if len(next(os.walk(self.cfg.DATA.TEST.SSL_SOURCE_DIR))[2]) != len(next(os.walk(self.cfg.DATA.TEST.PATH))[2]):
                    print("Different number of files found in {} and {}. Trying to create the the rest again"
                        .format(self.cfg.DATA.TEST.GT_PATH, self.cfg.DATA.TEST.SSL_SOURCE_DIR))
                    create_mask = True    
                else:
                    print("Test source data found in {}".format(self.cfg.DATA.TEST.SSL_SOURCE_DIR))
            if create_mask:
                create_ssl_source_data_masks(self.cfg, data_type='test')

        opts = []
        if self.cfg.TRAIN.ENABLE:
            print("DATA.TRAIN.PATH changed from {} to {}".format(self.cfg.DATA.TRAIN.PATH, self.cfg.DATA.TRAIN.SSL_SOURCE_DIR))
            print("DATA.TRAIN.GT_PATH changed from {} to {}".format(self.cfg.DATA.TRAIN.GT_PATH, self.cfg.DATA.TRAIN.PATH))
            opts.extend(['DATA.TRAIN.PATH', self.cfg.DATA.TRAIN.SSL_SOURCE_DIR, 'DATA.TRAIN.GT_PATH', self.cfg.DATA.TRAIN.PATH])
            if not self.cfg.DATA.VAL.FROM_TRAIN:
                print("DATA.VAL.PATH changed from {} to {}".format(self.cfg.DATA.VAL.PATH, self.cfg.DATA.VAL.SSL_SOURCE_DIR))
                print("DATA.VAL.GT_PATH changed from {} to {}".format(self.cfg.DATA.VAL.GT_PATH, self.cfg.DATA.VAL.PATH))
                opts.extend(['DATA.VAL.PATH', self.cfg.DATA.VAL.SSL_SOURCE_DIR, 'DATA.VAL.GT_PATH', self.cfg.DATA.VAL.PATH])
        if self.cfg.TEST.ENABLE:
            print("DATA.TEST.PATH changed from {} to {}".format(self.cfg.DATA.TEST.PATH, self.cfg.DATA.TEST.SSL_SOURCE_DIR))
            print("DATA.TEST.GT_PATH changed from {} to {}".format(self.cfg.DATA.TEST.GT_PATH, self.cfg.DATA.TEST.PATH))
            opts.extend(['DATA.TEST.PATH', self.cfg.DATA.TEST.SSL_SOURCE_DIR, 'DATA.TEST.GT_PATH', self.cfg.DATA.TEST.PATH]) 
        self.cfg.merge_from_list(opts)
