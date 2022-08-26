import math
import numpy as np
from tqdm import tqdm

from data.data_2D_manipulation import crop_data_with_overlap, merge_data_with_overlap
from data.data_3D_manipulation import crop_3D_data_with_overlap, merge_3D_data_with_overlap
from data.post_processing.post_processing import ensemble8_2d_predictions, ensemble16_3d_predictions
from utils.util import pad_and_reflect, save_tif
from engine.base_workflow import Base_Workflow
from engine.metrics import PSNR

class Super_resolution(Base_Workflow):
    def __init__(self, cfg, model, post_processing=False):
        super().__init__(cfg, model, post_processing)
        self.stats['psnr_per_image'] = 0

    def process_sample(self, X, Y, filenames): 
        _X = X.copy()
        _Y = Y.copy() if self.cfg.DATA.TEST.LOAD_GT else None

        original_data_shape = (_X.shape[0], _X.shape[1]*self.cfg.AUGMENTOR.RANDOM_CROP_SCALE,
                               _X.shape[2]*self.cfg.AUGMENTOR.RANDOM_CROP_SCALE, _X.shape[3])
    
        # Crop if necessary
        if self.cfg.PROBLEM.NDIM == '2D':
            t_patch_size = self.cfg.DATA.PATCH_SIZE
        else:
            t_patch_size = tuple(self.cfg.DATA.PATCH_SIZE[i] for i in [2, 1, 0, 3])
        if _X.shape[1:] != t_patch_size:
            if self.cfg.PROBLEM.NDIM == '2D':
                _X = crop_data_with_overlap(_X, self.cfg.DATA.PATCH_SIZE, overlap=self.cfg.DATA.TEST.OVERLAP, 
                    padding=self.cfg.DATA.TEST.PADDING, verbose=self.cfg.TEST.VERBOSE)
            else:
                _X = crop_3D_data_with_overlap(_X[0], self.cfg.DATA.PATCH_SIZE, 
                    overlap=self.cfg.DATA.TEST.OVERLAP, padding=self.cfg.DATA.TEST.PADDING,
                    verbose=self.cfg.TEST.VERBOSE, median_padding=self.cfg.DATA.TEST.MEDIAN_PADDING)

        # Predict each patch
        pred = []
        if self.cfg.TEST.AUGMENTATION:
            for k in tqdm(range(_X.shape[0]), leave=False):
                if self.cfg.PROBLEM.NDIM == '2D':
                    p = ensemble8_2d_predictions(_X[k], n_classes=self.cfg.MODEL.N_CLASSES,
                            pred_func=(lambda img_batch_subdiv: self.model.predict(img_batch_subdiv)))
                else:
                    p = ensemble16_3d_predictions(_X[k], batch_size_value=1,
                            pred_func=(lambda img_batch_subdiv: self.model.predict(img_batch_subdiv)))
                pred.append(np.expand_dims(p, 0))
        else:
            for k in tqdm(range(_X.shape[0]), leave=False):
                p = self.model.predict(_X[k], verbose=0)
                pred.append(np.expand_dims(p, 0))

        # Reconstruct the predictions
        pred = np.concatenate(pred)
        if original_data_shape[1:] != t_patch_size:
            if self.cfg.PROBLEM.NDIM == '3D': original_data_shape = original_data_shape[1:]
            f_name = merge_data_with_overlap if self.cfg.PROBLEM.NDIM == '2D' else merge_3D_data_with_overlap
            pad = tuple(p*self.cfg.AUGMENTOR.RANDOM_CROP_SCALE for p in self.cfg.DATA.TEST.PADDING)
            ov = tuple(o*self.cfg.AUGMENTOR.RANDOM_CROP_SCALE for o in self.cfg.DATA.TEST.OVERLAP) 
            pred = f_name(pred, original_data_shape[:-1]+(pred.shape[-1],), padding=pad, 
                          overlap=ov, verbose=self.cfg.TEST.VERBOSE)
        else:
            pred = pred[0]

        # Save image
        if self.cfg.PATHS.RESULT_DIR.PER_IMAGE != "":
            save_tif(np.expand_dims(pred,0), self.cfg.PATHS.RESULT_DIR.PER_IMAGE, filenames, verbose=self.cfg.TEST.VERBOSE)
    
        # Calculate PSNR
        if self.cfg.DATA.TEST.LOAD_GT:
            _Y = _Y[0]
            psnr_per_image = PSNR(pred, _Y)
            self.stats['psnr_per_image'] += psnr_per_image

    def after_merge_patches(self, pred, Y, filenames):
        pass

    def after_full_image(self, pred, Y, filenames):
        pass

    def after_all_images(self, Y):
        super().after_all_images(None)

    def normalize_stats(self, image_counter):
        self.stats['psnr_per_image'] = self.stats['psnr_per_image'] / image_counter

    def print_stats(self, image_counter):
        self.normalize_stats(image_counter)

        if self.cfg.DATA.TEST.LOAD_GT:
            print("Test PSNR (merge patches): {}".format(self.stats['psnr_per_image']))
            print(" ")


        