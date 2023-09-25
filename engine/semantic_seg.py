import torch
import numpy as np

from engine.base_workflow import Base_Workflow
from utils.util import save_tif, check_masks
from utils.misc import to_pytorch_format
from engine.metrics import jaccard_index, weighted_bce_dice_loss

class Semantic_Segmentation_Workflow(Base_Workflow):
    def __init__(self, cfg, job_identifier, device, rank, **kwargs):
        super(Semantic_Segmentation_Workflow, self).__init__(cfg, job_identifier, device, rank, **kwargs)

        print("####################")      
        print("#  PRE-PROCESSING  #")
        print("####################")
        if cfg.TRAIN.ENABLE and cfg.DATA.TRAIN.CHECK_DATA:
            if cfg.LOSS.TYPE == 'MASKED_BCE':
                check_masks(cfg.DATA.TRAIN.GT_PATH, n_classes=3)
            else:
                check_masks(cfg.DATA.TRAIN.GT_PATH, n_classes=cfg.MODEL.N_CLASSES+1)
            if not cfg.DATA.VAL.FROM_TRAIN:
                if cfg.LOSS.TYPE == 'MASKED_BCE':
                    check_masks(cfg.DATA.VAL.GT_PATH, n_classes=3)
                else:
                    check_masks(cfg.DATA.VAL.GT_PATH, n_classes=cfg.MODEL.N_CLASSES+1)
        if cfg.TEST.ENABLE and cfg.DATA.TEST.LOAD_GT and cfg.DATA.TEST.CHECK_DATA:
            if cfg.LOSS.TYPE == 'MASKED_BCE':
                check_masks(cfg.DATA.TEST.GT_PATH, n_classes=3)
            else:
                check_masks(cfg.DATA.TEST.GT_PATH, n_classes=cfg.MODEL.N_CLASSES+1)

        # From now on, no modification of the cfg will be allowed
        self.cfg.freeze()

        # Activations for each output channel:
        # channel number : 'activation'
        self.activations = {':': 'BCE_Sigmoid'}

        # Workflow specific training variables
        self.mask_path = cfg.DATA.TRAIN.GT_PATH
        self.load_Y_val = True

    def define_metrics(self):
        if self.cfg.LOSS.TYPE == "CE": 
            self.metrics = [jaccard_index]
            self.metric_names = ["jaccard_index"]
            if self.cfg.MODEL.N_CLASSES <= 2:
                self.loss = torch.nn.BCEWithLogitsLoss()
            else:
                self.loss = torch.nn.CrossEntropyLoss()
        elif self.cfg.LOSS.TYPE == "W_CE_DICE":
            self.metrics = [jaccard_index]
            self.metric_names = ["jaccard_index"]
            self.loss = weighted_bce_dice_loss(w_dice=0.66, w_bce=0.33)

    def metric_calculation(self, output, targets, device, metric_logger=None):
        with torch.no_grad():
            train_iou = self.metrics[0](output, targets, device, num_classes=self.cfg.MODEL.N_CLASSES)
            train_iou = train_iou.item() if not torch.isnan(train_iou) else 0
            if metric_logger is not None:
                metric_logger.meters[self.metric_names[0]].update(train_iou)
            else:
                return train_iou

    def after_merge_patches(self, pred, filenames):
        # Save simple binarization of predictions
        if pred.ndim == 4 and self.cfg.PROBLEM.NDIM == '3D':
            save_tif(np.expand_dims((pred>0.5).astype(np.uint8),0), self.cfg.PATHS.RESULT_DIR.PER_IMAGE_BIN,
                     filenames, verbose=self.cfg.TEST.VERBOSE)
        else:
            save_tif((pred>0.5).astype(np.uint8), self.cfg.PATHS.RESULT_DIR.PER_IMAGE_BIN, filenames,
                        verbose=self.cfg.TEST.VERBOSE)

    def after_full_image(self, pred, filenames):
        # Save simple binarization of predictions
        if pred.ndim == 4 and self.cfg.PROBLEM.NDIM == '3D':
            save_tif(np.expand_dims((pred>0.5).astype(np.uint8),0), self.cfg.PATHS.RESULT_DIR.FULL_IMAGE_BIN,
                     filenames, verbose=self.cfg.TEST.VERBOSE)
        else:
            save_tif((pred>0.5).astype(np.uint8), self.cfg.PATHS.RESULT_DIR.FULL_IMAGE_BIN, filenames,
                        verbose=self.cfg.TEST.VERBOSE)

    def after_all_images(self):
        super().after_all_images()

    def normalize_stats(self, image_counter):
        super().normalize_stats(image_counter)

    def print_stats(self, image_counter):
        super().print_stats(image_counter)
        super().print_post_processing_stats()


        