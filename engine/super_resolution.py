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
        original_data_shape = X.shape
    
        # Crop if necessary
        if X.shape[1:-1] != self.cfg.DATA.PATCH_SIZE[:-1]:
            if self.cfg.PROBLEM.NDIM == '2D':
                obj = crop_data_with_overlap(X, self.cfg.DATA.PATCH_SIZE, data_mask=Y, overlap=self.cfg.DATA.TEST.OVERLAP, 
                    padding=self.cfg.DATA.TEST.PADDING, verbose=self.cfg.TEST.VERBOSE)
                if self.cfg.DATA.TEST.LOAD_GT:
                    X, Y = obj
                else:
                    X = obj
                del obj
            else:
                if self.cfg.DATA.TEST.LOAD_GT: Y = Y[0]
                if self.cfg.TEST.REDUCE_MEMORY:
                    X = crop_3D_data_with_overlap(X[0], self.cfg.DATA.PATCH_SIZE, overlap=self.cfg.DATA.TEST.OVERLAP, 
                        padding=self.cfg.DATA.TEST.PADDING, verbose=self.cfg.TEST.VERBOSE, 
                        median_padding=self.cfg.DATA.TEST.MEDIAN_PADDING)
                    Y = crop_3D_data_with_overlap(Y, self.cfg.DATA.PATCH_SIZE, overlap=self.cfg.DATA.TEST.OVERLAP, 
                        padding=self.cfg.DATA.TEST.PADDING, verbose=self.cfg.TEST.VERBOSE, 
                        median_padding=self.cfg.DATA.TEST.MEDIAN_PADDING)
                else:
                    obj = crop_3D_data_with_overlap(X[0], self.cfg.DATA.PATCH_SIZE, data_mask=Y, overlap=self.cfg.DATA.TEST.OVERLAP, 
                        padding=self.cfg.DATA.TEST.PADDING, verbose=self.cfg.TEST.VERBOSE, 
                        median_padding=self.cfg.DATA.TEST.MEDIAN_PADDING)
                    if self.cfg.DATA.TEST.LOAD_GT:
                        X, Y = obj
                    else:
                        X = obj
                    del obj

        # Predict each patch
        pred = []
        if self.cfg.TEST.AUGMENTATION:
            for k in tqdm(range(X.shape[0]), leave=False):
                if self.cfg.PROBLEM.NDIM == '2D':
                    p = ensemble8_2d_predictions(X[k], n_classes=self.cfg.MODEL.N_CLASSES,
                            pred_func=(lambda img_batch_subdiv: self.model.predict(img_batch_subdiv)))
                else:
                    p = ensemble16_3d_predictions(X[k], batch_size_value=1,
                            pred_func=(lambda img_batch_subdiv: self.model.predict(img_batch_subdiv)))
                pred.append(p)
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

            if self.cfg.TEST.REDUCE_MEMORY:
                pred = f_name(pred, original_data_shape[:-1]+(pred.shape[-1],), padding=self.cfg.DATA.TEST.PADDING, 
                    overlap=self.cfg.DATA.TEST.OVERLAP, verbose=self.cfg.TEST.VERBOSE)
                Y = f_name(Y, original_data_shape[:-1]+(Y.shape[-1],), padding=self.cfg.DATA.TEST.PADDING, 
                    overlap=self.cfg.DATA.TEST.OVERLAP, verbose=self.cfg.TEST.VERBOSE)
            else:
                obj = f_name(pred, original_data_shape[:-1]+(pred.shape[-1],), data_mask=Y,
                    padding=self.cfg.DATA.TEST.PADDING, overlap=self.cfg.DATA.TEST.OVERLAP,
                    verbose=self.cfg.TEST.VERBOSE)
                if self.cfg.DATA.TEST.LOAD_GT:
                    pred, Y = obj
                else:
                    pred = obj
                del obj
        else:
            pred = pred[0]

        # Undo normalization
        x_norm = norm[0][0]
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
        if self.cfg.DATA.TEST.LOAD_GT:
            psnr_per_image = PSNR(pred, Y)
            self.stats['psnr_per_image'] += psnr_per_image

    def after_merge_patches(self, pred, Y, filenames):
        pass

    def after_full_image(self, pred, Y, filenames):
        pass

    def after_all_images(self, Y):
        pass

    def normalize_stats(self, image_counter):
        self.stats['psnr_per_image'] = self.stats['psnr_per_image'] / image_counter

    def print_stats(self, image_counter):
        self.normalize_stats(image_counter)

        if self.cfg.DATA.TEST.LOAD_GT:
            print("Test PSNR (merge patches): {}".format(self.stats['psnr_per_image']))
            print(" ")


        