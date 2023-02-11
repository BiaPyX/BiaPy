import math
import numpy as np
from tqdm import tqdm
from abc import ABCMeta, abstractmethod

from utils.util import pad_and_reflect, save_tif, check_downsample_division
from data.data_2D_manipulation import crop_data_with_overlap, merge_data_with_overlap
from data.data_3D_manipulation import crop_3D_data_with_overlap, merge_3D_data_with_overlap
from data.post_processing.post_processing import ensemble8_2d_predictions, ensemble16_3d_predictions, apply_binary_mask
from engine.metrics import jaccard_index_numpy, voc_calculation
from data.post_processing import apply_post_processing


class Base_Workflow(metaclass=ABCMeta):
    def __init__(self, cfg, model, post_processing={}):
        self.cfg = cfg
        self.model = model
        self.post_processing = post_processing

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


    def process_sample(self, X, Y, filenames, f_numbers, norm):
        #################
        ### PER PATCH ###
        #################
        if self.cfg.TEST.STATS.PER_PATCH:
            # Reflect data to complete the needed shape
            if self.cfg.DATA.REFLECT_TO_COMPLETE_SHAPE:
                reflected_orig_shape = X.shape
                X = np.expand_dims(pad_and_reflect(X[0], self.cfg.DATA.PATCH_SIZE, verbose=self.cfg.TEST.VERBOSE),0)
                if self.cfg.DATA.TEST.LOAD_GT:
                    Y = np.expand_dims(pad_and_reflect(Y[0], self.cfg.DATA.PATCH_SIZE, verbose=self.cfg.TEST.VERBOSE),0)

            original_data_shape = X.shape
            
            # Crop if necessary
            if X.shape[1:-1] != self.cfg.DATA.PATCH_SIZE[:-1] or any(x == 0 for x in self.cfg.DATA.TEST.PADDING)\
                or any(x == 0 for x in self.cfg.DATA.TEST.OVERLAP):
                # Copy X to be used later in full image 
                if self.cfg.PROBLEM.NDIM != '3D': 
                    X_original = X.copy()

                if self.cfg.PROBLEM.NDIM == '2D':
                    obj = crop_data_with_overlap(X, self.cfg.DATA.PATCH_SIZE, data_mask=Y, overlap=self.cfg.DATA.TEST.OVERLAP, 
                        padding=self.cfg.DATA.TEST.PADDING, verbose=self.cfg.TEST.VERBOSE)
                    if self.cfg.DATA.TEST.LOAD_GT:
                        X, Y = obj
                        if X.shape[:-1] != Y.shape[:-1]:
                            raise ValueError("Image {} and mask {} differ in shape".format(X.shape,Y.shape))
                    else:
                        X = obj
                    del obj
                else:
                    if self.cfg.DATA.TEST.LOAD_GT and X.shape[:-1] != Y.shape[:-1]:
                        raise ValueError("Image {} and mask {} differ in shape (without considering the channels, last dimension)"
                            .format(X[0].shape,Y[0].shape))
                    if self.cfg.TEST.REDUCE_MEMORY:
                        X = crop_3D_data_with_overlap(X[0], self.cfg.DATA.PATCH_SIZE, overlap=self.cfg.DATA.TEST.OVERLAP, 
                            padding=self.cfg.DATA.TEST.PADDING, verbose=self.cfg.TEST.VERBOSE, 
                            median_padding=self.cfg.DATA.TEST.MEDIAN_PADDING)
                        if self.cfg.DATA.TEST.LOAD_GT:
                            Y = crop_3D_data_with_overlap(Y[0], self.cfg.DATA.PATCH_SIZE[:-1]+(Y.shape[-1],), overlap=self.cfg.DATA.TEST.OVERLAP, 
                                padding=self.cfg.DATA.TEST.PADDING, verbose=self.cfg.TEST.VERBOSE, 
                                median_padding=self.cfg.DATA.TEST.MEDIAN_PADDING)
                    else:
                        if self.cfg.DATA.TEST.LOAD_GT: Y = Y[0]
                        obj = crop_3D_data_with_overlap(X[0], self.cfg.DATA.PATCH_SIZE, data_mask=Y, overlap=self.cfg.DATA.TEST.OVERLAP, 
                            padding=self.cfg.DATA.TEST.PADDING, verbose=self.cfg.TEST.VERBOSE, 
                            median_padding=self.cfg.DATA.TEST.MEDIAN_PADDING)
                        if self.cfg.DATA.TEST.LOAD_GT:
                            X, Y = obj
                        else:
                            X = obj
                        del obj
            
            # Evaluate each patch
            if self.cfg.DATA.TEST.LOAD_GT and self.cfg.TEST.EVALUATE:
                l = int(math.ceil(X.shape[0]/self.cfg.TRAIN.BATCH_SIZE))
                for k in tqdm(range(l), leave=False):
                    top = (k+1)*self.cfg.TRAIN.BATCH_SIZE if (k+1)*self.cfg.TRAIN.BATCH_SIZE < X.shape[0] else X.shape[0]
                    score = self.model.evaluate(
                        X[k*self.cfg.TRAIN.BATCH_SIZE:top], Y[k*self.cfg.TRAIN.BATCH_SIZE:top], verbose=0)
                    self.stats['loss_per_crop'] += score[0]
                    self.stats['iou_per_crop'] += score[1]
            self.stats['patch_counter'] += X.shape[0]

            # Predict each patch
            if self.cfg.TEST.AUGMENTATION:
                for k in tqdm(range(X.shape[0]), leave=False):
                    if self.cfg.PROBLEM.NDIM == '2D':
                        p = ensemble8_2d_predictions(X[k], n_classes=self.cfg.MODEL.N_CLASSES,
                                pred_func=(lambda img_batch_subdiv: self.model.predict(img_batch_subdiv)))
                    else:
                        p = ensemble16_3d_predictions(X[k], batch_size_value=self.cfg.TRAIN.BATCH_SIZE,
                                pred_func=(lambda img_batch_subdiv: self.model.predict(img_batch_subdiv)))
                    if 'pred' not in locals():
                        pred = np.zeros((X.shape[0],)+p.shape, dtype=np.float32)
                    pred[k] = p
            else:
                l = int(math.ceil(X.shape[0]/self.cfg.TRAIN.BATCH_SIZE))
                for k in tqdm(range(l), leave=False):
                    top = (k+1)*self.cfg.TRAIN.BATCH_SIZE if (k+1)*self.cfg.TRAIN.BATCH_SIZE < X.shape[0] else X.shape[0]
                    p = self.model.predict(X[k*self.cfg.TRAIN.BATCH_SIZE:top], verbose=0)
                    if 'pred' not in locals():
                        pred = np.zeros((X.shape[0],)+p.shape[1:], dtype=np.float32)
                    pred[k*self.cfg.TRAIN.BATCH_SIZE:top] = p

            # Delete X as in 3D there is no full image
            if self.cfg.PROBLEM.NDIM == '3D':
                del X, p

            # Reconstruct the predictions
            if original_data_shape[1:-1] != self.cfg.DATA.PATCH_SIZE[:-1] or any(x == 0 for x in self.cfg.DATA.TEST.PADDING)\
                or any(x == 0 for x in self.cfg.DATA.TEST.OVERLAP):
                if self.cfg.PROBLEM.NDIM == '3D': original_data_shape = original_data_shape[1:]
                f_name = merge_data_with_overlap if self.cfg.PROBLEM.NDIM == '2D' else merge_3D_data_with_overlap

                if self.cfg.TEST.REDUCE_MEMORY:
                    pred = f_name(pred, original_data_shape[:-1]+(pred.shape[-1],), padding=self.cfg.DATA.TEST.PADDING, 
                        overlap=self.cfg.DATA.TEST.OVERLAP, verbose=self.cfg.TEST.VERBOSE)
                    if self.cfg.DATA.TEST.LOAD_GT:
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
                if self.cfg.PROBLEM.NDIM != '3D': 
                    X = X_original.copy()
                    del X_original
            else:
                pred = pred[0]

            if self.cfg.DATA.REFLECT_TO_COMPLETE_SHAPE and self.cfg.PROBLEM.NDIM == '3D':
                pred = pred[-reflected_orig_shape[1]:,-reflected_orig_shape[2]:,-reflected_orig_shape[3]:]
                if Y is not None:
                    Y = Y[-reflected_orig_shape[1]:,-reflected_orig_shape[2]:,-reflected_orig_shape[3]:]

            # Argmax if needed
            if self.cfg.MODEL.N_CLASSES > 1 and self.cfg.DATA.TEST.ARGMAX_TO_OUTPUT:
                pred = np.expand_dims(np.argmax(pred,-1), -1)
                if self.cfg.DATA.TEST.LOAD_GT: Y = np.expand_dims(np.argmax(Y,-1), -1)

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
                    if Y.ndim > pred.ndim: Y = Y[0]
                    if self.cfg.LOSS.TYPE != 'MASKED_BCE':
                        _iou_per_image = jaccard_index_numpy((Y>0.5).astype(np.uint8), (pred>0.5).astype(np.uint8))
                        _ov_iou_per_image = voc_calculation((Y>0.5).astype(np.uint8), (pred>0.5).astype(np.uint8),
                                                        _iou_per_image)
                    else:
                        exclusion_mask = Y < 2
                        binY = Y * exclusion_mask.astype( float )
                        _iou_per_image = jaccard_index_numpy((binY>0.5).astype(np.uint8), (pred>0.5).astype(np.uint8))
                        _ov_iou_per_image = voc_calculation((binY>0.5).astype(np.uint8), (pred>0.5).astype(np.uint8),
                                                        _iou_per_image)
                    self.stats['iou_per_image'] += _iou_per_image
                    self.stats['ov_iou_per_image'] += _ov_iou_per_image

                ############################
                ### POST-PROCESSING (3D) ###
                ############################
                if self.post_processing['per_image']:
                    pred, _iou_post, _ov_iou_post = apply_post_processing(self.cfg, pred, Y)
                    self.stats['iou_post'] += _iou_post
                    self.stats['ov_iou_post'] += _ov_iou_post
                    if pred.ndim == 4:
                        save_tif(np.expand_dims(pred,0), self.cfg.PATHS.RESULT_DIR.PER_IMAGE_POST_PROCESSING,
                                    filenames, verbose=self.cfg.TEST.VERBOSE)
                    else:
                        save_tif(pred, self.cfg.PATHS.RESULT_DIR.PER_IMAGE_POST_PROCESSING, filenames,
                                    verbose=self.cfg.TEST.VERBOSE)

            self.after_merge_patches(pred, Y, filenames, f_numbers)
            
            if not self.cfg.TEST.STATS.FULL_IMG and self.cfg.TEST.ANALIZE_2D_IMGS_AS_3D_STACK:
                self.all_pred.append(pred)
                if self.cfg.DATA.TEST.LOAD_GT: self.all_gt.append(Y)            

        ##################
        ### FULL IMAGE ###
        ##################
        if self.cfg.TEST.STATS.FULL_IMG and self.cfg.PROBLEM.NDIM == '2D':
            X, o_test_shape = check_downsample_division(X, len(self.cfg.MODEL.FEATURE_MAPS)-1)
            if self.cfg.DATA.TEST.LOAD_GT:
                Y, _ = check_downsample_division(Y, len(self.cfg.MODEL.FEATURE_MAPS)-1)

            # Evaluate each img
            if self.cfg.DATA.TEST.LOAD_GT:
                score = self.model.evaluate(X, Y, verbose=0)
                self.stats['loss'] += score[0]

            # Make the prediction
            if self.cfg.TEST.AUGMENTATION:
                pred = ensemble8_2d_predictions(
                    X[0], pred_func=(lambda img_batch_subdiv: self.model.predict(img_batch_subdiv)),
                    n_classes=self.cfg.MODEL.N_CLASSES)
                pred = np.expand_dims(pred, 0)
            else:
                pred = self.model.predict(X, verbose=0)

            # Recover original shape if padded with check_downsample_division
            pred = pred[:,:o_test_shape[1],:o_test_shape[2]]
            if self.cfg.DATA.TEST.LOAD_GT: Y = Y[:,:o_test_shape[1],:o_test_shape[2]]

            # Save image
            if pred.ndim == 4 and self.cfg.PROBLEM.NDIM == '3D':
                save_tif(np.expand_dims(pred,0), self.cfg.PATHS.RESULT_DIR.FULL_IMAGE, filenames,
                            verbose=self.cfg.TEST.VERBOSE)
            else:
                save_tif(pred, self.cfg.PATHS.RESULT_DIR.FULL_IMAGE, filenames, verbose=self.cfg.TEST.VERBOSE)

            # Argmax if needed
            if self.cfg.MODEL.N_CLASSES > 1 and self.cfg.DATA.TEST.ARGMAX_TO_OUTPUT:
                pred = np.expand_dims(np.argmax(pred,-1), -1)
                if self.cfg.DATA.TEST.LOAD_GT: Y = np.expand_dims(np.argmax(Y,-1), -1)

            if self.cfg.TEST.POST_PROCESSING.APPLY_MASK:
                pred = apply_binary_mask(pred, self.cfg.DATA.TEST.BINARY_MASKS)
                
            if self.cfg.DATA.TEST.LOAD_GT:
                score[1] = jaccard_index_numpy((Y>0.5).astype(np.uint8), (pred>0.5).astype(np.uint8))
                self.stats['iou'] += score[1]
                self.stats['ov_iou'] += voc_calculation((Y>0.5).astype(np.uint8), (pred>0.5).astype(np.uint8), score[1])

            if self.post_processing['all_images']:
                self.all_pred.append(pred)
                if self.cfg.DATA.TEST.LOAD_GT: self.all_gt.append(Y)

            self.after_full_image(pred, Y, filenames)

    def normalize_stats(self, image_counter):
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
        if self.post_processing['per_image'] or self.post_processing['all_images']:
            print("Test Foreground IoU (post-processing): {}".format(self.stats['iou_post']))
            print("Test Overall IoU (post-processing): {}".format(self.stats['ov_iou_post']))
            print(" ")


    @abstractmethod
    def after_merge_patches(self, pred, Y, filenames, f_numbers):
        raise NotImplementedError

    @abstractmethod
    def after_full_image(self):
        raise NotImplementedError

    def after_all_images(self):
        ############################
        ### POST-PROCESSING (2D) ###
        ############################
        if self.post_processing['all_images']:
            self.all_pred = np.concatenate(self.all_pred)
            self.all_gt = np.concatenate(self.all_gt) if self.cfg.DATA.TEST.LOAD_GT else None
            self.all_pred, self.stats['iou_post'], self.stats['ov_iou_post'] = apply_post_processing(self.cfg, self.all_pred, self.all_gt)
            save_tif(self.all_pred, self.cfg.PATHS.RESULT_DIR.AS_3D_STACK_POST_PROCESSING, verbose=self.cfg.TEST.VERBOSE)

