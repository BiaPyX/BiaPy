import numpy as np

from engine.base_workflow import Base_Workflow
from utils.util import save_tif

class Semantic_Segmentation(Base_Workflow):
    def __init__(self, cfg, model, post_processing=False):
        super().__init__(cfg, model, post_processing)
        
    def after_merge_patches(self, pred, Y, filenames):
        # Save simple binarization of predictions
        if pred.ndim == 4 and self.cfg.PROBLEM.NDIM == '3D':
            save_tif(np.expand_dims((pred>0.5).astype(np.uint8),0), self.cfg.PATHS.RESULT_DIR.PER_IMAGE_BIN,
                     filenames, verbose=self.cfg.TEST.VERBOSE)
        else:
            save_tif((pred>0.5).astype(np.uint8), self.cfg.PATHS.RESULT_DIR.PER_IMAGE_BIN, filenames,
                        verbose=self.cfg.TEST.VERBOSE)

    def after_full_image(self, pred, Y, filenames):
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


        