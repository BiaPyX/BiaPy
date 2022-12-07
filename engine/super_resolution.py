import math
import numpy as np
from tqdm import tqdm

from data.data_2D_manipulation import crop_data_with_overlap, merge_data_with_overlap
from data.data_3D_manipulation import crop_3D_data_with_overlap, merge_3D_data_with_overlap
from data.post_processing.post_processing import ensemble8_2d_predictions, ensemble16_3d_predictions
from utils.util import save_tif
from engine.base_workflow import Base_Workflow
from engine.metrics import PSNR
from data.pre_processing import denormalize

class Super_resolution(Base_Workflow):
    def __init__(self, cfg, model, post_processing=False):
        super().__init__(cfg, model, post_processing)
        self.stats['psnr_per_image'] = 0

    def process_sample(self, X, Y, filenames, norm): 
        original_data_shape= (X.shape[0], X.shape[1]*self.cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING,
                              X.shape[2]*self.cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING, X.shape[3])

        # Crop if necessary
        if X.shape[1:-1] != self.cfg.DATA.PATCH_SIZE[:-1]:
            if self.cfg.PROBLEM.NDIM == '2D':
                X = crop_data_with_overlap(X, self.cfg.DATA.PATCH_SIZE, overlap=self.cfg.DATA.TEST.OVERLAP, 
                    padding=self.cfg.DATA.TEST.PADDING, verbose=self.cfg.TEST.VERBOSE)
            else:
                X = crop_3D_data_with_overlap(X, self.cfg.DATA.PATCH_SIZE, overlap=self.cfg.DATA.TEST.OVERLAP, 
                    padding=self.cfg.DATA.TEST.PADDING, verbose=self.cfg.TEST.VERBOSE)

        # Predict each patch
        self.model.set_dtype(Y[0])
        pred = []
        if self.cfg.TEST.AUGMENTATION:
            for k in tqdm(range(X.shape[0]), leave=False):
                if self.cfg.PROBLEM.NDIM == '2D':
                    p = ensemble8_2d_predictions(X[k], n_classes=self.cfg.MODEL.N_CLASSES,
                            pred_func=(lambda img_batch_subdiv: self.model.predict(img_batch_subdiv)))
                else:
                    p = ensemble16_3d_predictions(X[k], batch_size_value=1,
                            pred_func=(lambda img_batch_subdiv: self.model.predict(img_batch_subdiv)))
                pred.append(np.expand_dims(p,0))
        else:
            l = int(math.ceil(X.shape[0]/self.cfg.TRAIN.BATCH_SIZE))
            for k in tqdm(range(l), leave=False):
                top = (k+1)*self.cfg.TRAIN.BATCH_SIZE if (k+1)*self.cfg.TRAIN.BATCH_SIZE < X.shape[0] else X.shape[0]
                p = self.model.predict(X[k*self.cfg.TRAIN.BATCH_SIZE:top], verbose=0)
                pred.append(p)
        del X, p

        # Reconstruct the predictions
        pred = np.concatenate(pred)
        if original_data_shape[1:-1] != self.cfg.DATA.PATCH_SIZE[:-1]:
            if self.cfg.PROBLEM.NDIM == '3D': original_data_shape = original_data_shape[1:]
            f_name = merge_data_with_overlap if self.cfg.PROBLEM.NDIM == '2D' else merge_3D_data_with_overlap
            pad = tuple(p*self.cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING for p in self.cfg.DATA.TEST.PADDING)
            ov = tuple(o*self.cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING for o in self.cfg.DATA.TEST.OVERLAP) 
            pred = f_name(pred, original_data_shape[:-1]+(pred.shape[-1],), padding=pad, 
                overlap=ov, verbose=self.cfg.TEST.VERBOSE)
        else:
            pred = pred[0]
            
        # Save image
        if self.cfg.PATHS.RESULT_DIR.PER_IMAGE != "":
            save_tif(np.expand_dims(pred,0), self.cfg.PATHS.RESULT_DIR.PER_IMAGE, filenames, verbose=self.cfg.TEST.VERBOSE)
    
        # Calculate PSNR
        if self.cfg.DATA.TEST.LOAD_GT:
            m_val = 255 if Y.dtype != np.uint16 else 65535
            psnr_per_image = PSNR(pred, Y, m_val)
            self.stats['psnr_per_image'] += psnr_per_image

    def after_merge_patches(self, pred, Y, filenames):
        pass

    def after_full_image(self, pred, Y, filenames):
        pass

    def after_all_images(self):
        pass

    def normalize_stats(self, image_counter):
        self.stats['psnr_per_image'] = self.stats['psnr_per_image'] / image_counter

    def print_stats(self, image_counter):
        self.normalize_stats(image_counter)

        if self.cfg.DATA.TEST.LOAD_GT:
            print("Test PSNR (merge patches): {}".format(self.stats['psnr_per_image']))
            print(" ")


        