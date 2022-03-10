import os
import math
import h5py
import numpy as np
import csv
import pandas as pd
from tqdm import tqdm
from skimage.io import imread
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from skimage.feature import peak_local_max
from scipy.ndimage.morphology import binary_dilation
from skimage.measure import label, regionprops_table

from utils.util import (check_masks, check_downsample_division, create_plots, save_tif, load_data_from_dir,
                        load_3d_images_from_dir, apply_binary_mask, pad_and_reflect, wrapper_matching_dataset_lazy, wrapper_matching_VJI_and_PAI)
from utils.matching import matching, match_using_VJI_and_PAI
from data import create_instance_channels, create_test_instance_channels
from data.data_2D_manipulation import (load_and_prepare_2D_train_data, crop_data_with_overlap, merge_data_with_overlap,
                                       load_data_classification)
from data.data_3D_manipulation import load_and_prepare_3D_data, crop_3D_data_with_overlap, merge_3D_data_with_overlap
from data.generators import create_train_val_augmentors, create_test_augmentor, check_generator_consistence
from data.post_processing import apply_post_processing
from data.post_processing.post_processing import (ensemble8_2d_predictions, ensemble16_3d_predictions, bc_watershed,
                                                  bcd_watershed, bdv2_watershed, calculate_optimal_mw_thresholds,
                                                  voronoi_on_mask_2)
from models import build_model
from engine import build_callbacks, prepare_optimizer
from engine.metrics import jaccard_index_numpy, voc_calculation, detection_metrics


class Trainer(object):

    def __init__(self, cfg, job_identifier):
        self.cfg = cfg
        self.job_identifier = job_identifier
        self.original_test_path = None
        self.original_test_mask_path = None
        self.test_mask_filenames = None

        # Save paths in case we need them in a future
        self.orig_train_path = cfg.DATA.TRAIN.PATH
        self.orig_train_mask_path = cfg.DATA.TRAIN.MASK_PATH
        self.orig_val_path = cfg.DATA.VAL.PATH
        self.orig_val_mask_path = cfg.DATA.VAL.MASK_PATH

        if cfg.PROBLEM.TYPE in ['SEMANTIC_SEG', 'DETECTION']:
            print("###################\n"
                  "#  SANITY CHECKS  #\n"
                  "###################\n")
            if cfg.TRAIN.ENABLE:
                check_masks(cfg.DATA.TRAIN.MASK_PATH)
                if not cfg.DATA.VAL.FROM_TRAIN:
                    check_masks(cfg.DATA.VAL.MASK_PATH)
            if cfg.TEST.ENABLE and cfg.DATA.TEST.LOAD_GT:
                check_masks(cfg.DATA.TEST.MASK_PATH)

            # Adjust the metric used accordingly to the number of classes. This code is planned to be used in a binary
            # classification problem, so the function 'jaccard_index_softmax' will only calculate the IoU for the
            # foreground class (channel 1)
            self.metric = "jaccard_index_softmax" if cfg.MODEL.N_CLASSES > 1 else "jaccard_index"
        elif cfg.PROBLEM.TYPE == 'INSTANCE_SEG':
            if cfg.DATA.CHANNELS in ["BC", "BCM"]:
                self.metric = "jaccard_index"
            elif  cfg.DATA.CHANNELS == 'Dv2':
                self.metric = "mse"
            else:
                self.metric = "jaccard_index_instances"
        # CLASSIFICATION
        else:
            self.metric = "accuracy"


        if cfg.PROBLEM.TYPE == 'INSTANCE_SEG':
            print("###########################\n"
                  "#  PREPARE INSTANCE DATA  #\n"
                  "###########################\n")
            # Create selected channels for train data
            if cfg.TRAIN.ENABLE and not os.path.isdir(cfg.DATA.TRAIN.INSTANCE_CHANNELS_DIR):
                print("You select to create {} channels from given instance labels and no file is detected in {}. "
                      "So let's prepare the data. Notice that, if you do not modify 'DATA.TRAIN.INSTANCE_CHANNELS_DIR' "
                      "path, this process will be done just once!".format(cfg.DATA.CHANNELS,
                      cfg.DATA.TRAIN.INSTANCE_CHANNELS_DIR))
                self.train_filenames = create_instance_channels(cfg)

            # Create selected channels for val data
            if cfg.TRAIN.ENABLE and not cfg.DATA.VAL.FROM_TRAIN and not os.path.isdir(cfg.DATA.VAL.INSTANCE_CHANNELS_DIR):
                print("You select to create {} channels from given instance labels and no file is detected in {}. "
                      "So let's prepare the data. Notice that, if you do not modify 'DATA.VAL.INSTANCE_CHANNELS_DIR' "
                      "path, this process will be done just once!".format(cfg.DATA.CHANNELS,
                      cfg.DATA.VAL.INSTANCE_CHANNELS_DIR))
                create_instance_channels(cfg, data_type='val')

            # Create selected channels for test data once
            if cfg.TEST.ENABLE and not os.path.isdir(cfg.DATA.TEST.INSTANCE_CHANNELS_DIR):
                print("You select to create {} channels from given instance labels and no file is detected in {}. "
                      "So let's prepare the data. Notice that, if you do not modify 'DATA.TEST.INSTANCE_CHANNELS_DIR' "
                      "path, this process will be done just once!".format(cfg.DATA.CHANNELS,
                      cfg.DATA.TEST.INSTANCE_CHANNELS_DIR))
                create_test_instance_channels(cfg)

            opts = []
            print("DATA.TRAIN.PATH changed from {} to {}".format(cfg.DATA.TRAIN.PATH, cfg.DATA.TRAIN.INSTANCE_CHANNELS_DIR))
            print("DATA.TRAIN.MASK_PATH changed from {} to {}".format(cfg.DATA.TRAIN.MASK_PATH, cfg.DATA.TRAIN.INSTANCE_CHANNELS_MASK_DIR))
            opts.extend(['DATA.TRAIN.PATH', cfg.DATA.TRAIN.INSTANCE_CHANNELS_DIR,
                         'DATA.TRAIN.MASK_PATH', cfg.DATA.TRAIN.INSTANCE_CHANNELS_MASK_DIR])
            if not cfg.DATA.VAL.FROM_TRAIN:
                print("DATA.VAL.PATH changed from {} to {}".format(cfg.DATA.VAL.PATH, cfg.DATA.VAL.INSTANCE_CHANNELS_DIR))
                print("DATA.VAL.MASK_PATH changed from {} to {}".format(cfg.DATA.VAL.MASK_PATH, cfg.DATA.VAL.INSTANCE_CHANNELS_MASK_DIR))
                opts.extend(['DATA.VAL.PATH', cfg.DATA.VAL.INSTANCE_CHANNELS_DIR,
                             'DATA.VAL.MASK_PATH', cfg.DATA.VAL.INSTANCE_CHANNELS_MASK_DIR])
            if cfg.TEST.ENABLE:
                self.original_test_path = cfg.DATA.TEST.PATH
                print("DATA.TEST.PATH changed from {} to {}".format(cfg.DATA.TEST.PATH, cfg.DATA.TEST.INSTANCE_CHANNELS_DIR))
                opts.extend(['DATA.TEST.PATH', cfg.DATA.TEST.INSTANCE_CHANNELS_DIR])
                self.original_test_mask_path = cfg.DATA.TEST.MASK_PATH
                if cfg.DATA.TEST.LOAD_GT and cfg.TEST.EVALUATE:
                    print("DATA.TEST.MASK_PATH changed from {} to {}".format(cfg.DATA.TEST.MASK_PATH, cfg.DATA.TEST.INSTANCE_CHANNELS_MASK_DIR))
                    opts.extend(['DATA.TEST.MASK_PATH', cfg.DATA.TEST.INSTANCE_CHANNELS_MASK_DIR])
            cfg.merge_from_list(opts)

        # From now on, no modification of the cfg will be allowed
        cfg.freeze()


        if (cfg.TRAIN.ENABLE and cfg.DATA.TRAIN.IN_MEMORY) or (cfg.TEST.ENABLE and cfg.DATA.TEST.IN_MEMORY):
            print("#################\n"
                  "### LOAD DATA ###\n"
                  "#################\n")

        #############
        ### TRAIN ###
        #############
        if cfg.TRAIN.ENABLE:
            if cfg.PROBLEM.TYPE in ['SEMANTIC_SEG', 'INSTANCE_SEG', 'DETECTION']:
                if cfg.DATA.TRAIN.IN_MEMORY:
                    if cfg.PROBLEM.NDIM == '2D':
                        objs = load_and_prepare_2D_train_data(cfg.DATA.TRAIN.PATH, cfg.DATA.TRAIN.MASK_PATH,
                            val_split=cfg.DATA.VAL.SPLIT_TRAIN, seed=cfg.SYSTEM.SEED, shuffle_val=cfg.DATA.VAL.RANDOM,
                            random_crops_in_DA=cfg.DATA.EXTRACT_RANDOM_PATCH, crop_shape=cfg.DATA.PATCH_SIZE,
                            ov=cfg.DATA.TRAIN.OVERLAP, padding=cfg.DATA.TRAIN.PADDING, check_crop=cfg.DATA.TRAIN.CHECK_CROP,
                            check_crop_path=cfg.PATHS.CROP_CHECKS, reflect_to_complete_shape=cfg.DATA.REFLECT_TO_COMPLETE_SHAPE)
                    else:
                        objs = load_and_prepare_3D_data(cfg.DATA.TRAIN.PATH, cfg.DATA.TRAIN.MASK_PATH,
                            val_split=cfg.DATA.VAL.SPLIT_TRAIN, seed=cfg.SYSTEM.SEED, shuffle_val=cfg.DATA.VAL.RANDOM,
                            random_crops_in_DA=cfg.DATA.EXTRACT_RANDOM_PATCH, crop_shape=cfg.DATA.PATCH_SIZE,
                            ov=cfg.DATA.TRAIN.OVERLAP, padding=cfg.DATA.TRAIN.PADDING,
                            reflect_to_complete_shape=cfg.DATA.REFLECT_TO_COMPLETE_SHAPE)

                    if cfg.DATA.VAL.FROM_TRAIN:
                        X_train, Y_train, X_val, Y_val, self.train_filenames = objs
                    else:
                        X_train, Y_train, self.train_filenames = objs
                    del objs
                else:
                    if not os.path.exists(cfg.DATA.TRAIN.PATH):
                        raise ValueError("Train data dir not found: {}".format(cfg.DATA.TRAIN.PATH))
                    if not os.path.exists(cfg.DATA.TRAIN.MASK_PATH):
                        raise ValueError("Train mask data dir not found: {}".format(cfg.DATA.TRAIN.MASK_PATH))
                    X_train, Y_train = None, None

                ##################
                ### VALIDATION ###
                ##################
                if not cfg.DATA.VAL.FROM_TRAIN:
                    if cfg.DATA.VAL.IN_MEMORY:
                        f_name = load_data_from_dir if cfg.PROBLEM.NDIM == '2D' else load_3d_images_from_dir
                        X_val, _, _ = f_name(cfg.DATA.VAL.PATH, crop=True, crop_shape=cfg.DATA.PATCH_SIZE,
                                             overlap=cfg.DATA.VAL.OVERLAP, padding=cfg.DATA.VAL.PADDING,
                                             reflect_to_complete_shape=cfg.DATA.REFLECT_TO_COMPLETE_SHAPE)
                        Y_val, _, _ = f_name(cfg.DATA.VAL.MASK_PATH, crop=True, crop_shape=cfg.DATA.PATCH_SIZE,
                                             overlap=cfg.DATA.VAL.OVERLAP, padding=cfg.DATA.VAL.PADDING,
                                             reflect_to_complete_shape=cfg.DATA.REFLECT_TO_COMPLETE_SHAPE)
                    else:
                        if not os.path.exists(cfg.DATA.VAL.PATH):
                            raise ValueError("Validation data dir not found: {}".format(cfg.DATA.VAL.PATH))
                        if not os.path.exists(cfg.DATA.VAL.MASK_PATH):
                            raise ValueError("Validation mask data dir not found: {}".format(cfg.DATA.VAL.MASK_PATH))
                        X_val, Y_val = None, None

            # CLASSIFICATION
            else:
                if cfg.DATA.TRAIN.IN_MEMORY:
                    X_train, Y_train, X_val, Y_val = load_data_classification(cfg)
                else:
                    X_train, Y_train = None, None
                    if not os.path.exists(cfg.DATA.TRAIN.PATH):
                        raise ValueError("Train data dir not found: {}".format(cfg.DATA.TRAIN.PATH))

                    X_val, Y_val = None, None
                    if not os.path.exists(cfg.DATA.VAL.PATH):
                        raise ValueError("Validation data dir not found: {}".format(cfg.DATA.VAL.PATH))

        ############
        ### TEST ###
        ############
        if cfg.TEST.ENABLE:
            if cfg.PROBLEM.TYPE in ['SEMANTIC_SEG', 'INSTANCE_SEG', 'DETECTION']:
                # Path comprobations
                if not os.path.exists(cfg.DATA.TEST.PATH):
                    raise ValueError("Test data not found: {}".format(cfg.DATA.TEST.PATH))
                if cfg.DATA.TEST.LOAD_GT and not os.path.exists(cfg.DATA.TEST.MASK_PATH):
                        raise ValueError("Test data mask not found: {}".format(cfg.DATA.TEST.MASK_PATH))

                if cfg.DATA.TEST.IN_MEMORY:
                    print("2) Loading test images . . .")
                    f_name = load_data_from_dir if cfg.PROBLEM.NDIM == '2D' else load_3d_images_from_dir
                    X_test, _, _ = f_name(cfg.DATA.TEST.PATH)
                    if cfg.DATA.TEST.LOAD_GT:
                        print("3) Loading test masks . . .")
                        Y_test, _, _ = f_name(cfg.DATA.TEST.MASK_PATH)
                    else:
                        Y_test = None
                else:
                    X_test, Y_test = None, None

                if self.original_test_path is None:
                    self.test_filenames = sorted(next(os.walk(cfg.DATA.TEST.PATH))[2])
                    if cfg.TEST.MAP and cfg.DATA.TEST.LOAD_GT:
                        self.test_mask_filenames = sorted(next(os.walk(cfg.DATA.TEST.MASK_PATH))[2])
                else:
                    self.test_filenames = sorted(next(os.walk(self.original_test_path))[2])
                    if cfg.TEST.MAP and cfg.DATA.TEST.LOAD_GT:
                        self.test_mask_filenames = sorted(next(os.walk(self.original_test_mask_path))[2])
            # CLASSIFICATION
            else:
                X_test, Y_test, self.test_filenames = load_data_classification(cfg, test=True)


        print("########################\n"
              "#  PREPARE GENERATORS  #\n"
              "########################\n")
        if cfg.TRAIN.ENABLE:
            self.train_generator, self.val_generator = create_train_val_augmentors(cfg, X_train, Y_train, X_val, Y_val)
            if cfg.DATA.CHECK_GENERATORS:
                check_generator_consistence(
                    self.train_generator, cfg.PATHS.GEN_CHECKS+"_train", cfg.PATHS.GEN_MASK_CHECKS+"_train")
                check_generator_consistence(
                    self.val_generator, cfg.PATHS.GEN_CHECKS+"_val", cfg.PATHS.GEN_MASK_CHECKS+"_val")
        if cfg.TEST.ENABLE:
            self.test_generator = create_test_augmentor(cfg, X_test, Y_test)

        print("#################\n"
              "#  BUILD MODEL  #\n"
              "#################\n")
        self.model = build_model(cfg, self.job_identifier)
        prepare_optimizer(cfg, self.model)


    def train(self):
        print("#####################\n"
              "#  TRAIN THE MODEL  #\n"
              "#####################\n")
        if self.cfg.MODEL.LOAD_CHECKPOINT:
            print("Loading model weights from h5_file: {}".format(self.cfg.PATHS.CHECKPOINT_FILE))
            self.model.load_weights(self.cfg.PATHS.CHECKPOINT_FILE)

        self.callbacks = build_callbacks(self.cfg)
        self.results = self.model.fit(self.train_generator, validation_data=self.val_generator,
            validation_steps=len(self.val_generator), steps_per_epoch=len(self.train_generator),
            epochs=self.cfg.TRAIN.EPOCHS, callbacks=self.callbacks)

        create_plots(self.results, self.job_identifier, self.cfg.PATHS.CHARTS, metric=self.metric)


    def test(self):
        print("Loading model weights from h5_file: {}".format(self.cfg.PATHS.CHECKPOINT_FILE))
        self.model.load_weights(self.cfg.PATHS.CHECKPOINT_FILE)

        print("###############\n"
              "#  INFERENCE  #\n"
              "###############\n")

        print("Making predictions on test data . . .")
        if self.cfg.TEST.STATS.PER_PATCH or self.cfg.PROBLEM.TYPE == 'CLASSIFICATION':
           loss_per_crop, iou_per_crop, patch_counter = 0, 0, 0
        if self.cfg.TEST.STATS.PER_PATCH and self.cfg.PROBLEM.TYPE == 'DETECTION':
            d_precision, d_recall, d_f1 = 0, 0, 0
        if self.cfg.TEST.STATS.MERGE_PATCHES:
           loss_per_imag, iou_per_image, ov_iou_per_image = 0, 0, 0
        if self.cfg.TEST.STATS.FULL_IMG:
           loss, iou, ov_iou = 0, 0, 0
        if self.cfg.TEST.MAP and self.cfg.PROBLEM.TYPE == 'INSTANCE_SEG' and self.cfg.DATA.TEST.LOAD_GT:
            mAP_50_total = 0
            mAP_75_total = 0
            if self.cfg.TEST.VORONOI_ON_MASK:
                mAP_50_total_vor = 0
                mAP_75_total_vor = 0
        if self.cfg.TEST.MATCHING_STATS and self.cfg.PROBLEM.TYPE == 'INSTANCE_SEG' and self.cfg.DATA.TEST.LOAD_GT:
            all_matching_stats = []
            if self.cfg.TEST.VORONOI_ON_MASK:
                all_matching_stats_voronoi = []

        if self.cfg.TEST.MATCHING_VJI_PAI and self.cfg.PROBLEM.TYPE == 'INSTANCE_SEG' and self.cfg.DATA.TEST.LOAD_GT:
            all_matching_stats_VJI = []
            if self.cfg.TEST.VORONOI_ON_MASK:
                all_matching_stats_voronoi_VJI = []

        if self.cfg.PROBLEM.TYPE == 'INSTANCE_SEG':
            if self.cfg.DATA.MW_OPTIMIZE_THS and self.cfg.DATA.CHANNELS != "BCDv2":
                if self.cfg.TEST.APPLY_MASK and os.path.isdir(self.cfg.DATA.VAL.BINARY_MASKS):
                    bin_mask = self.cfg.DATA.VAL.BINARY_MASKS
                else:
                    bin_mask = None
                obj = calculate_optimal_mw_thresholds(self.model, self.cfg.DATA.VAL.PATH,
                    self.orig_val_mask_path, self.cfg.DATA.PATCH_SIZE, self.cfg.DATA.CHANNELS,
                    self.cfg.DATA.VAL.MASK_PATH, self.cfg.DATA.REMOVE_SMALL_OBJ, bin_mask,
                    chart_dir=self.cfg.PATHS.CHARTS, verbose=self.cfg.TEST.VERBOSE)
                if self.cfg.DATA.CHANNELS == "BCD":
                    th1_opt, th2_opt, th3_opt, th4_opt, th5_opt = obj
                else:
                    th1_opt, th2_opt, th3_opt = obj
                    th4_opt, th5_opt = self.cfg.DATA.MW_TH4, self.cfg.DATA.MW_TH5
            else:
                th1_opt, th2_opt, th3_opt = self.cfg.DATA.MW_TH1, self.cfg.DATA.MW_TH2, self.cfg.DATA.MW_TH3
                th4_opt, th5_opt = self.cfg.DATA.MW_TH4, self.cfg.DATA.MW_TH5

        image_counter = 0

        if self.cfg.TEST.POST_PROCESSING.BLENDING or self.cfg.TEST.POST_PROCESSING.YZ_FILTERING or \
           self.cfg.TEST.POST_PROCESSING.Z_FILTERING:
            post_processing, iou_post, ov_iou_post = True, 0, 0
        else:
            post_processing = False

        if (self.cfg.TEST.STATS.FULL_IMG and self.cfg.PROBLEM.NDIM == '2D' and post_processing) or \
            self.cfg.PROBLEM.TYPE == 'CLASSIFICATION':
            all_pred = []
            all_gt = []

        it = iter(self.test_generator)
        for i in tqdm(range(len(self.test_generator))):
            batch = next(it)

            if self.cfg.DATA.TEST.LOAD_GT:
                X, Y = batch
            else:
                X = batch
                Y = None
            del batch

            l_X = len(X)
            for j in tqdm(range(l_X), leave=False):
                print("Processing image(s): {}".format(self.test_filenames[(i*l_X)+j:(i*l_X)+j+1]))

                if self.cfg.PROBLEM.TYPE != 'CLASSIFICATION':
                    if type(X) is tuple:
                        _X = X[j]
                        _Y = Y[j] if self.cfg.DATA.TEST.LOAD_GT else None
                    else:
                        _X = np.expand_dims(X[j],0)
                        _Y = np.expand_dims(Y[j],0) if self.cfg.DATA.TEST.LOAD_GT else None
                    if self.cfg.PROBLEM.NDIM == '3D':
                        # Convert to (num_images, z, x, y, c)
                        _X = _X.transpose((0,3,1,2,4))
                        if self.cfg.DATA.TEST.LOAD_GT: _Y = _Y.transpose((0,3,1,2,4))
                else:
                    _X = np.expand_dims(X[j], 0)
                    _Y = np.expand_dims(Y[j], 0)


                #################
                ### PER PATCH ###
                #################
                if self.cfg.TEST.STATS.PER_PATCH or self.cfg.PROBLEM.TYPE == 'CLASSIFICATION':
                    if self.cfg.DATA.REFLECT_TO_COMPLETE_SHAPE:
                        reflected_orig_shape = _X.shape
                        _X = np.expand_dims(pad_and_reflect(_X[0], self.cfg.DATA.PATCH_SIZE, verbose=self.cfg.TEST.VERBOSE),0)
                        if _Y is not None:
                            _Y = np.expand_dims(pad_and_reflect(_Y[0], self.cfg.DATA.PATCH_SIZE, verbose=self.cfg.TEST.VERBOSE),0)

                    if self.cfg.PROBLEM.TYPE != 'CLASSIFICATION':
                        original_data_shape = _X.shape if self.cfg.PROBLEM.NDIM == '2D' else _X.shape[1:]
                    else:
                        original_data_shape = _X.shape[1:]

                    if _X.shape[1:] != self.cfg.DATA.PATCH_SIZE:
                        if self.cfg.PROBLEM.TYPE == 'CLASSIFICATION':
                            raise ValueError("For classification the images provided need to be of the selected "
                                             "'DATA.PATCH_SIZE', {} given".format(_X.shape[1:]))

                        if self.cfg.PROBLEM.NDIM == '2D':
                            obj = crop_data_with_overlap(_X, self.cfg.DATA.PATCH_SIZE, data_mask=_Y,
                                overlap=self.cfg.DATA.TEST.OVERLAP, padding=self.cfg.DATA.TEST.PADDING,
                                verbose=self.cfg.TEST.VERBOSE)
                            if self.cfg.DATA.TEST.LOAD_GT:
                                _X, _Y = obj
                            else:
                                _X = obj
                        else:
                            if self.cfg.DATA.TEST.LOAD_GT: _Y = _Y[0]
                            obj = crop_3D_data_with_overlap(_X[0], self.cfg.DATA.PATCH_SIZE, data_mask=_Y,
                                overlap=self.cfg.DATA.TEST.OVERLAP, padding=self.cfg.DATA.TEST.PADDING,
                                verbose=self.cfg.TEST.VERBOSE, median_padding=self.cfg.DATA.TEST.MEDIAN_PADDING)
                            if self.cfg.DATA.TEST.LOAD_GT:
                                _X, _Y = obj
                            else:
                                _X = obj

                    # Evaluate each patch
                    if self.cfg.DATA.TEST.LOAD_GT and self.cfg.TEST.EVALUATE and self.cfg.PROBLEM.TYPE != 'CLASSIFICATION':
                        l = int(math.ceil(_X.shape[0]/self.cfg.TRAIN.BATCH_SIZE))
                        for k in tqdm(range(l), leave=False):
                            top = (k+1)*self.cfg.TRAIN.BATCH_SIZE if (k+1)*self.cfg.TRAIN.BATCH_SIZE < _X.shape[0] else _X.shape[0]
                            score = self.model.evaluate(
                                _X[k*self.cfg.TRAIN.BATCH_SIZE:top], _Y[k*self.cfg.TRAIN.BATCH_SIZE:top], verbose=0)
                            loss_per_crop += score[0]
                            iou_per_crop += score[1]
                    patch_counter += _X.shape[0]

                    # Predict each patch
                    pred = []
                    if self.cfg.TEST.AUGMENTATION and self.cfg.PROBLEM.TYPE != 'CLASSIFICATION':
                        for k in tqdm(range(_X.shape[0]), leave=False):
                            if self.cfg.PROBLEM.NDIM == '2D':
                                p = ensemble8_2d_predictions(_X[k], n_classes=self.cfg.MODEL.N_CLASSES,
                                        pred_func=(lambda img_batch_subdiv: self.model.predict(img_batch_subdiv)))
                            else:
                                p = ensemble16_3d_predictions(_X[k], batch_size_value=self.cfg.TRAIN.BATCH_SIZE,
                                        pred_func=(lambda img_batch_subdiv: self.model.predict(img_batch_subdiv)))
                            pred.append(np.expand_dims(p, 0))
                    else:
                        l = int(math.ceil(_X.shape[0]/self.cfg.TRAIN.BATCH_SIZE))
                        for k in tqdm(range(l), leave=False):
                            top = (k+1)*self.cfg.TRAIN.BATCH_SIZE if (k+1)*self.cfg.TRAIN.BATCH_SIZE < _X.shape[0] else _X.shape[0]
                            p = self.model.predict(_X[k*self.cfg.TRAIN.BATCH_SIZE:top], verbose=0)
                            if self.cfg.PROBLEM.TYPE == 'CLASSIFICATION':
                                p = np.argmax(p, axis=1)
                                all_pred.append(pred)
                            pred.append(p)

                    # Reconstruct the predictions
                    pred = np.concatenate(pred)
                    if original_data_shape[1:] != self.cfg.DATA.PATCH_SIZE and self.cfg.PROBLEM.TYPE != 'CLASSIFICATION':
                        f_name = merge_data_with_overlap if self.cfg.PROBLEM.NDIM == '2D' else merge_3D_data_with_overlap
                        obj = f_name(pred, original_data_shape[:-1]+(pred.shape[-1],), data_mask=_Y,
                                      padding=self.cfg.DATA.TEST.PADDING, overlap=self.cfg.DATA.TEST.OVERLAP,
                                      verbose=self.cfg.TEST.VERBOSE)
                        if self.cfg.DATA.TEST.LOAD_GT:
                            pred, _Y = obj
                        else:
                            pred = obj
                    else:
                        pred = pred[0]

                    if self.cfg.DATA.REFLECT_TO_COMPLETE_SHAPE and self.cfg.PROBLEM.NDIM == '3D':
                        pred = pred[-reflected_orig_shape[1]:,-reflected_orig_shape[2]:,-reflected_orig_shape[3]:]
                        if _Y is not None:
                            _Y = _Y[-reflected_orig_shape[1]:,-reflected_orig_shape[2]:,-reflected_orig_shape[3]:]

                    if self.cfg.PROBLEM.TYPE != 'CLASSIFICATION':
                        # Argmax if needed
                        if self.cfg.MODEL.N_CLASSES > 1:
                            pred = np.expand_dims(np.argmax(pred,-1), -1)
                            if self.cfg.DATA.TEST.LOAD_GT: _Y = np.expand_dims(np.argmax(_Y,-1), -1)

                        # Apply mask
                        if self.cfg.TEST.APPLY_MASK:
                            pred = apply_binary_mask(pred, self.cfg.DATA.TEST.BINARY_MASKS)

                        # Save image
                        filenames = self.test_filenames[(i*l_X)+j:(i*l_X)+j+1]
                        if self.cfg.PATHS.RESULT_DIR.PER_IMAGE != "":
                            save_tif(np.expand_dims(pred,0), self.cfg.PATHS.RESULT_DIR.PER_IMAGE, filenames, verbose=self.cfg.TEST.VERBOSE)


                    #####################
                    ### MERGE PATCHES ###
                    #####################
                    if self.cfg.TEST.STATS.MERGE_PATCHES and self.cfg.PROBLEM.TYPE != 'CLASSIFICATION':
                        if self.cfg.DATA.TEST.LOAD_GT and self.cfg.DATA.CHANNELS != "Dv2":
                            if pred.ndim == 3: _Y = _Y[0]
                            _iou_per_image = jaccard_index_numpy((_Y>0.5).astype(np.uint8), (pred>0.5).astype(np.uint8))
                            _ov_iou_per_image = voc_calculation((_Y>0.5).astype(np.uint8), (pred>0.5).astype(np.uint8),
                                                                _iou_per_image)
                            iou_per_image += _iou_per_image
                            ov_iou_per_image += _ov_iou_per_image

                        ############################
                        ### POST-PROCESSING (3D) ###
                        ############################
                        if post_processing and self.cfg.PROBLEM.NDIM == '3D':
                            _iou_post, _ov_iou_post = apply_post_processing(self.cfg, pred, _Y)
                            iou_post += _iou_post
                            ov_iou_post += _ov_iou_post
                            if pred.ndim == 4 and self.cfg.PROBLEM.NDIM == '3D':
                                save_tif(np.expand_dims(pred,0), self.cfg.PATHS.RESULT_DIR.PER_IMAGE_POST_PROCESSING,
                                         filenames, verbose=self.cfg.TEST.VERBOSE)
                            else:
                                save_tif(pred, self.cfg.PATHS.RESULT_DIR.PER_IMAGE_POST_PROCESSING, filenames,
                                         verbose=self.cfg.TEST.VERBOSE)

                    # Detection in 3D
                    if self.cfg.TEST.DET_LOCAL_MAX_COORDS and self.cfg.PROBLEM.TYPE == 'DETECTION' and self.cfg.PROBLEM.NDIM == '3D':
                        print("Capturing the local maxima ")
                        pred_coordinates = peak_local_max(pred[...,0], threshold_rel=0.2, min_distance=10, exclude_border=False)

                        props = regionprops_table(label(_Y[...,0]), properties=('area','centroid'))
                        gt_coordinates = []
                        for n in range(len(props['centroid-0'])):
                            gt_coordinates.append([props['centroid-0'][n], props['centroid-1'][n], props['centroid-2'][n]])
                        gt_coordinates = np.array(gt_coordinates)

                        # Create a file that represent the local maxima
                        points_pred = np.zeros((pred[...,0].shape + (1,)), dtype=np.uint8)
                        for n, coord in enumerate(pred_coordinates):
                            z,x,y = coord
                            points_pred[z,x,y,0] = 255
                        for z_index in range(len(points_pred)):
                            points_pred[z_index,...,0] = binary_dilation(points_pred[z_index,...,0], iterations=2)

                        save_tif(np.expand_dims(points_pred,0), self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK,
                                 filenames, verbose=self.cfg.TEST.VERBOSE)
                        del points_pred

                        # Save coords in csv file
                        f = os.path.join(self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK, os.path.splitext(filenames[0])[0]+'.csv')
                        with open(f, 'w', newline="") as file:
                            csvwriter = csv.writer(file)
                            csvwriter.writerow(['index', 'axis-0', 'axis-1', 'axis-2'])
                            for nr in range(len(pred_coordinates)):
                                csvwriter.writerow([nr+1] + pred_coordinates[nr].tolist())

                        v_size = (self.cfg.TEST.DET_VOXEL_SIZE[2], self.cfg.TEST.DET_VOXEL_SIZE[1], self.cfg.TEST.DET_VOXEL_SIZE[0])
                        d_metrics = detection_metrics(gt_coordinates, pred_coordinates, tolerance=self.cfg.TEST.DET_TOLERANCE,
                                                      voxel_size=v_size)
                        d_precision += d_metrics[1]
                        d_recall += d_metrics[3]
                        d_f1 += d_metrics[5]
                        print("Detection metrics: {}".format(d_metrics))


                    #############################
                    ### INSTANCE SEGMENTATION ###
                    #############################
                    if self.cfg.PROBLEM.TYPE == 'INSTANCE_SEG':
                        print("Creating instances with watershed . . .")
                        w_dir = os.path.join(self.cfg.PATHS.WATERSHED_DIR, filenames[0])
                        check_wa = w_dir if self.cfg.DATA.CHECK_MW else None
                        if self.cfg.DATA.CHANNELS in ["BC", "BCM"]:
                            w_pred = bc_watershed(pred, thres1=th1_opt, thres2=th2_opt, thres3=th3_opt,
                                thres_small=self.cfg.DATA.REMOVE_SMALL_OBJ, remove_before=self.cfg.DATA.REMOVE_BEFORE_MW,
                                save_dir=check_wa)
                        elif self.cfg.DATA.CHANNELS == "BCD":
                            w_pred = bcd_watershed(pred, thres1=th1_opt, thres2=th2_opt, thres3=th3_opt, thres4=th4_opt,
                                thres5=th5_opt, thres_small=self.cfg.DATA.REMOVE_SMALL_OBJ,
                                remove_before=self.cfg.DATA.REMOVE_BEFORE_MW, save_dir=check_wa)
                        else: # "BCDv2"
                            w_pred = bdv2_watershed(pred, bin_th=th1_opt, thres_small=self.cfg.DATA.REMOVE_SMALL_OBJ,
                                remove_before=self.cfg.DATA.REMOVE_BEFORE_MW, save_dir=check_wa)

                        save_tif(np.expand_dims(np.expand_dims(w_pred,-1),0), self.cfg.PATHS.RESULT_DIR.PER_IMAGE_INSTANCES,
                                 filenames, verbose=self.cfg.TEST.VERBOSE)

                        if self.cfg.TEST.VORONOI_ON_MASK:
                            vor_pred = voronoi_on_mask_2(np.expand_dims(w_pred,0), np.expand_dims(pred,0),
                                self.cfg.PATHS.RESULT_DIR.PER_IMAGE_INST_VORONOI, filenames, verbose=self.cfg.TEST.VERBOSE)[0]

                        # Add extra dimension if working in 2D
                        if w_pred.ndim == 2:
                            w_pred = np.expand_dims(w_pred,0)

                    if self.cfg.TEST.MAP and self.cfg.PROBLEM.TYPE == 'INSTANCE_SEG' and self.cfg.DATA.TEST.LOAD_GT:
                        print("####################\n"
                              "#  mAP Calculation #\n"
                              "####################\n")

                        # Convert the prediction into an .h5 file
                        os.makedirs(self.cfg.PATHS.MAP_H5_DIR, exist_ok=True)
                        filenames = self.test_mask_filenames[(i*l_X)+j:(i*l_X)+j+1]
                        h5file_name = os.path.join(self.cfg.PATHS.MAP_H5_DIR, os.path.splitext(filenames[0])[0]+'.h5')
                        print("Creating prediction h5 file to calculate mAP: {}".format(h5file_name))
                        h5f = h5py.File(h5file_name, 'w')
                        h5f.create_dataset('dataset', data=w_pred, compression="lzf")
                        h5f.close()

                        # Prepare mAP call
                        import sys
                        sys.path.insert(0, self.cfg.PATHS.MAP_CODE_DIR)
                        from demo_modified import main as mAP_calculation
                        class Namespace:
                            def __init__(self, **kwargs):
                                self.__dict__.update(kwargs)

                        # Create GT H5 file if it does not exist
                        gt_f = os.path.join(self.cfg.PATHS.TEST_FULL_GT_H5, os.path.splitext(filenames[0])[0]+'.h5')
                        test_file = os.path.join(self.original_test_mask_path, filenames[0])
                        if not os.path.isfile(gt_f):
                            print("GT .h5 file needed for mAP calculation is not found in {} so it will be created "
                                  "from its mask: {}".format(gt_f, test_file))

                            if not os.path.isfile(test_file):
                                raise ValueError("The mask is supossed to have the same name as the image")
                            _Y = imread(test_file).squeeze()

                            # If multiple-channel data then only capture the first channel that is assumed to be instance labels
                            if (self.cfg.PROBLEM.NDIM == '2D' and (_Y.shape[0] > 1 and _Y.ndim == 3)) or\
                               (self.cfg.PROBLEM.NDIM == '3D' and (_Y.shape[0] > 1 and _Y.ndim == 4)):
                                _Y =  _Y[0]

                            # As the mAP code is prepared for 3D we need an extra z dimension and change dtype ot int
                            if _Y.dtype == np.float32: _Y = _Y.astype(np.int32)
                            if _Y.dtype == np.float64: _Y = _Y.astype(np.int64)
                            if _Y.ndim == 2: _Y = np.expand_dims(_Y,0)

                            print("Saving .h5 GT data from array shape: {}".format(_Y.shape))
                            os.makedirs(self.cfg.PATHS.TEST_FULL_GT_H5, exist_ok=True)
                            h5f = h5py.File(gt_f, 'w')
                            h5f.create_dataset('dataset', data=_Y, compression="lzf")
                            h5f.close()

                        # In case the GT has no labels in this image
                        gt_num_labels = len(np.unique(imread(test_file)))
                        if gt_num_labels > 1:
                            # Calculate mAP
                            args = Namespace(gt_seg=gt_f, predict_seg=h5file_name, predict_score='', threshold="5e3, 3e4",
                                             threshold_crumb=-1, chunk_size=250, output_name=w_dir, do_txt=1, do_eval=1,
                                             slices="-1")
                            mAP_calculation(args)

                            # Save metric
                            with open(os.path.join(w_dir, 'nucmm_map.txt'), "r") as read_obj:
                                for line in read_obj:
                                    if 'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] =' in line:
                                        mAP_50_total += float(line.split()[-1])
                                    elif 'Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] =' in line:
                                        mAP_75_total += float(line.split()[-1])
                        else:
                            print("No labels found in {} file. Skipping sample from mAP calculation . . .".format(test_file))

                        if self.cfg.TEST.VORONOI_ON_MASK:
                            print("mAP with Voronoi")
                            # As the mAP code is prepared for 3D we need an extra z dimension
                            if vor_pred.ndim == 2:
                                vor_pred = np.expand_dims(vor_pred,0)

                            h5file_name_vor = os.path.join(self.cfg.PATHS.MAP_H5_DIR, os.path.splitext(filenames[0])[0]+'_voronoi.h5')
                            print("Creating prediction h5 file to calculate mAP: {}".format(h5file_name_vor))
                            h5f = h5py.File(h5file_name_vor, 'w')
                            h5f.create_dataset('dataset', data=vor_pred, compression="lzf")
                            h5f.close()

                            if gt_num_labels > 1:
                                w_dir = os.path.join(self.cfg.PATHS.WATERSHED_DIR, filenames[0], "voronoi")
                                os.makedirs(w_dir, exist_ok=True)
                                # Calculate mAP
                                args = Namespace(gt_seg=gt_f, predict_seg=h5file_name_vor, predict_score='', threshold="5e3, 3e4",
                                                 threshold_crumb=-1, chunk_size=250, output_name=w_dir, do_txt=1, do_eval=1,
                                                 slices="-1")
                                mAP_calculation(args)

                                # Save metric
                                with open(os.path.join(w_dir, 'nucmm_map.txt'), "r") as read_obj:
                                    for line in read_obj:
                                        if 'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] =' in line:
                                            mAP_50_total_vor += float(line.split()[-1])
                                        elif 'Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] =' in line:
                                            mAP_75_total_vor += float(line.split()[-1])
                            else:
                                print("No labels found in {} file. Skipping sample from mAP calculation (Voronoi). . .".format(test_file))

                    if self.cfg.TEST.MATCHING_STATS and self.cfg.PROBLEM.TYPE == 'INSTANCE_SEG' and self.cfg.DATA.TEST.LOAD_GT:
                        print("Calculating matching stats . . .")
                        test_file = os.path.join(self.original_test_mask_path, filenames[0])
                        if not os.path.isfile(test_file):
                            raise ValueError("The mask is supossed to have the same name as the image")
                        _Y = imread(test_file).squeeze()

                        # If multiple-channel data then only capture the first channel that is assumed to be instance labels
                        if (self.cfg.PROBLEM.NDIM == '2D' and (_Y.shape[0] > 1 and _Y.ndim == 3)) or\
                           (self.cfg.PROBLEM.NDIM == '3D' and (_Y.shape[0] > 1 and _Y.ndim == 4)):
                            _Y =  _Y[0]

                        # As the mAP code is prepared for 3D we need an extra z dimension and change dtype ot int
                        if _Y.dtype == np.float32: _Y = _Y.astype(np.int32)
                        if _Y.dtype == np.float64: _Y = _Y.astype(np.int64)
                        if _Y.ndim == 2: _Y = np.expand_dims(_Y,0)

                        # Convert instances to integer
                        if _Y.dtype == np.float32: _Y = _Y.astype(np.int32)
                        if _Y.dtype == np.float64: _Y = _Y.astype(np.int64)

                        r_stats = matching(_Y, w_pred, thresh=self.cfg.TEST.MATCHING_STATS_THS, report_matches=False)
                        print(r_stats)
                        all_matching_stats.append(r_stats)

                        if self.cfg.TEST.VORONOI_ON_MASK:
                            r_stats = matching(_Y, vor_pred, thresh=self.cfg.TEST.MATCHING_STATS_THS, report_matches=False)
                            print("Stats with Voronoi")
                            print(r_stats)
                            all_matching_stats_voronoi.append(r_stats)

                    if self.cfg.TEST.MATCHING_VJI_PAI and self.cfg.PROBLEM.TYPE == 'INSTANCE_SEG' and self.cfg.DATA.TEST.LOAD_GT:
                        print("Calculating matching stats using VJI and PAI. . .")
                        test_file = os.path.join(self.cfg.DATA.TEST.MASK_PATH, filenames[0])
                        if not os.path.isfile(test_file):
                            raise ValueError("The mask is supossed to have the same name as the image")
                        _Y = imread(test_file).squeeze()
                        r_stats_VJI = match_using_VJI_and_PAI(_Y, w_pred)
                        print(r_stats_VJI)
                        all_matching_stats_VJI.append(r_stats_VJI)

                        if self.cfg.TEST.VORONOI_ON_MASK:
                            r_stats_VJI = match_using_VJI_and_PAI(_Y, vor_pred)
                            print("Stats with Voronoi")
                            print(r_stats_VJI)
                            all_matching_stats_voronoi_VJI.append(r_stats_VJI)


                ##################
                ### FULL IMAGE ###
                ##################
                if self.cfg.TEST.STATS.FULL_IMG and self.cfg.PROBLEM.NDIM == '2D' and self.cfg.PROBLEM.TYPE != 'CLASSIFICATION':
                    if type(X) is tuple:
                        _X = X[j]
                        _Y = Y[j] if self.cfg.DATA.TEST.LOAD_GT else None
                    else:
                        _X = np.expand_dims(X[j],0)
                        _Y = np.expand_dims(Y[j],0) if self.cfg.DATA.TEST.LOAD_GT else None

                    _X, o_test_shape = check_downsample_division(_X, len(self.cfg.MODEL.FEATURE_MAPS)-1)
                    if self.cfg.DATA.TEST.LOAD_GT:
                        _Y, _ = check_downsample_division(_Y, len(self.cfg.MODEL.FEATURE_MAPS)-1)

                    # Evaluate each img
                    if self.cfg.DATA.TEST.LOAD_GT:
                        score = self.model.evaluate(_X, _Y, verbose=0)
                        loss += score[0]
                        iou += score[1]

                    # Make the prediction
                    if self.cfg.TEST.AUGMENTATION:
                        pred = ensemble8_2d_predictions(
                            _X[0], pred_func=(lambda img_batch_subdiv: self.model.predict(img_batch_subdiv)),
                            n_classes=self.cfg.MODEL.N_CLASSES)
                        pred = np.expand_dims(pred, 0)
                    else:
                        pred = self.model.predict(_X, verbose=0)

                    # Recover original shape if padded with check_downsample_division
                    pred = pred[:,:o_test_shape[1],:o_test_shape[2]]
                    if self.cfg.DATA.TEST.LOAD_GT: _Y = _Y[:,:o_test_shape[1],:o_test_shape[2]]

                    # Save image
                    filenames = self.test_filenames[(i*len(_X))+j:(i*len(_X))+j+1]
                    if pred.ndim == 4 and self.cfg.PROBLEM.NDIM == '3D':
                        save_tif(np.expand_dims(pred,0), self.cfg.PATHS.RESULT_DIR.FULL_IMAGE, filenames,
                                 verbose=self.cfg.TEST.VERBOSE)
                    else:
                        save_tif(pred, self.cfg.PATHS.RESULT_DIR.FULL_IMAGE, filenames, verbose=self.cfg.TEST.VERBOSE)

                    # Argmax if needed
                    if self.cfg.MODEL.N_CLASSES > 1:
                        pred = np.expand_dims(np.argmax(pred,-1), -1)
                        if self.cfg.DATA.TEST.LOAD_GT: _Y = np.expand_dims(np.argmax(_Y,-1), -1)

                    if self.cfg.DATA.TEST.LOAD_GT:
                        ov_iou += voc_calculation((_Y>0.5).astype(np.uint8), (pred>0.5).astype(np.uint8), score[1])

                    if self.cfg.TEST.STATS.FULL_IMG and self.cfg.PROBLEM.NDIM == '2D' and post_processing:
                        all_pred.append(pred)
                        if self.cfg.DATA.TEST.LOAD_GT: all_gt.append(_Y)

                image_counter += 1

        del pred, _X, _Y

        if self.cfg.PROBLEM.TYPE == 'CLASSIFICATION':
            all_pred = np.concatenate(all_pred)

            # Save predictions in a csv file
            df = pd.DataFrame(self.test_filenames, columns=['Nuclei file'])
            df['pred_class'] = np.array(all_pred).squeeze()
            f= os.path.join(self.cfg.PATHS.RESULT_DIR.PER_IMAGE, "..", "predictions.csv")
            os.makedirs(self.cfg.PATHS.RESULT_DIR.PER_IMAGE, exist_ok=True)
            df.to_csv(f, index=False, header=True)

            if self.cfg.DATA.TEST.LOAD_GT and self.cfg.TEST.EVALUATE:
                display_labels = ["Category {}".format(i) for i in range(self.cfg.MODEL.N_CLASSES)]
                test_accuracy = accuracy_score(np.argmax(Y, axis=-1), all_pred)
                cm = confusion_matrix(np.argmax(Y, axis=-1), all_pred)


        ############################
        ### POST-PROCESSING (2D) ###
        ############################
        if self.cfg.TEST.STATS.FULL_IMG and self.cfg.PROBLEM.NDIM == '2D' and post_processing and self.cfg.PROBLEM.TYPE != 'CLASSIFICATION':
            all_pred = np.concatenate(all_pred)
            if self.cfg.DATA.TEST.LOAD_GT:
                all_gt = np.concatenate(all_gt)
                iou_post, ov_iou_post = apply_post_processing(self.cfg, all_pred, all_gt)
            else:
                iou_post, ov_iou_post = 0, 0
            save_tif(all_pred, self.cfg.PATHS.RESULT_DIR.FULL_POST_PROCESSING, verbose=self.cfg.TEST.VERBOSE)
            del all_pred

        if post_processing and self.cfg.PROBLEM.NDIM == '3D':
            iou_post = iou_post / image_counter
            ov_iou_post = ov_iou_post / image_counter
        if self.cfg.TEST.STATS.PER_PATCH and self.cfg.PROBLEM.TYPE != 'CLASSIFICATION':
            loss_per_crop = loss_per_crop / patch_counter
            iou_per_crop = iou_per_crop / patch_counter
            if self.cfg.TEST.STATS.MERGE_PATCHES:
                iou_per_image = iou_per_image / image_counter
                ov_iou_per_image = ov_iou_per_image / image_counter
            if self.cfg.TEST.MAP and self.cfg.PROBLEM.TYPE == 'INSTANCE_SEG' and self.cfg.DATA.TEST.LOAD_GT:
                mAP_50_total = mAP_50_total / image_counter
                mAP_75_total = mAP_75_total / image_counter
                if self.cfg.TEST.VORONOI_ON_MASK:
                    mAP_50_total_vor = mAP_50_total_vor / image_counter
                    mAP_75_total_vor = mAP_75_total_vor / image_counter

        if self.cfg.TEST.STATS.FULL_IMG and self.cfg.PROBLEM.NDIM == '2D' and self.cfg.PROBLEM.TYPE != 'CLASSIFICATION':
            iou = iou / image_counter
            loss = loss / image_counter
            ov_iou = ov_iou / image_counter

        if self.cfg.TEST.MATCHING_STATS and self.cfg.PROBLEM.TYPE == 'INSTANCE_SEG' and self.cfg.DATA.TEST.LOAD_GT:
            stats = wrapper_matching_dataset_lazy(all_matching_stats, self.cfg.TEST.MATCHING_STATS_THS)
            if self.cfg.TEST.VORONOI_ON_MASK:
                stats_vor = wrapper_matching_dataset_lazy(all_matching_stats_voronoi, self.cfg.TEST.MATCHING_STATS_THS)


        if self.cfg.TEST.MATCHING_VJI_PAI and self.cfg.PROBLEM.TYPE == 'INSTANCE_SEG' and self.cfg.DATA.TEST.LOAD_GT:
            stats_VJI = wrapper_matching_VJI_and_PAI(all_matching_stats_VJI)
            if self.cfg.TEST.VORONOI_ON_MASK:
                stats_vor_VJI = wrapper_matching_VJI_and_PAI(all_matching_stats_voronoi_VJI)

        if self.cfg.TEST.STATS.PER_PATCH and self.cfg.PROBLEM.TYPE == 'DETECTION':
            d_precision = d_precision / image_counter
            d_recall = d_recall / image_counter
            d_f1 = d_f1 / image_counter


        print("#############\n"
              "#  RESULTS  #\n"
              "#############\n")

        if self.cfg.TRAIN.ENABLE:
            print("Epoch average time: {}".format(np.mean(self.callbacks[0].times)))
            print("Epoch number: {}".format(len(self.results.history['val_loss'])))
            print("Train time (s): {}".format(np.sum(self.callbacks[0].times)))
            print("Train loss: {}".format(np.min(self.results.history['loss'])))
            print("Train Foreground IoU: {}".format(np.max(self.results.history[self.metric])))
            print("Validation loss: {}".format(np.min(self.results.history['val_loss'])))
            print("Validation Foreground IoU: {}".format(np.max(self.results.history['val_'+self.metric])))

        if self.cfg.DATA.TEST.LOAD_GT:
            if self.cfg.PROBLEM.TYPE == 'CLASSIFICATION':
                if self.cfg.DATA.TEST.LOAD_GT and self.cfg.TEST.EVALUATE:
                    print('Test Accuracy: ', round((test_accuracy * 100), 2), "%")
                    print("Confusion matrix: ")
                    print(cm)
                    print(classification_report(np.argmax(Y, axis=-1), all_pred, target_names=display_labels))
            else:
                if self.cfg.TEST.STATS.PER_PATCH:
                    print("Loss (per patch): {}".format(loss_per_crop))
                    print("Test Foreground IoU (per patch): {}".format(iou_per_crop))
                    print(" ")
                    if self.cfg.TEST.STATS.MERGE_PATCHES:
                        print("Test Foreground IoU (merge patches): {}".format(iou_per_image))
                        print("Test Overall IoU (merge patches): {}".format(ov_iou_per_image))
                        print(" ")
                    if self.cfg.PROBLEM.TYPE == 'DETECTION':
                        print("Detection - Test Precision: {}".format(d_precision))
                        print("Detection - Test Recall: {}".format(d_recall))
                        print("Detection - Test F1: {}".format(d_f1))

                if self.cfg.PROBLEM.TYPE == 'INSTANCE_SEG':
                    if self.cfg.TEST.MAP:
                        print("Test Average Precision (AP) - IoU=0.50 : {}".format(mAP_50_total))
                        print("Test Average Precision (AP) - IoU=0.75 : {}".format(mAP_75_total))
                        print(" ")
                        if self.cfg.TEST.VORONOI_ON_MASK:
                            print("Test Average Precision (AP) (Voronoi) - IoU=0.50 : {}".format(mAP_50_total_vor))
                            print("Test Average Precision (AP) (Voronoi) - IoU=0.75 : {}".format(mAP_75_total_vor))
                            print(" ")
                    if self.cfg.TEST.MATCHING_STATS:
                        for i in range(len(self.cfg.TEST.MATCHING_STATS_THS)):
                            print("IoU TH={}".format(self.cfg.TEST.MATCHING_STATS_THS[i]))
                            print(stats[i])
                            if self.cfg.TEST.VORONOI_ON_MASK:
                                print("IoU (Voronoi) TH={}".format(self.cfg.TEST.MATCHING_STATS_THS[i]))
                                print(stats_vor[i])

                    if self.cfg.TEST.MATCHING_VJI_PAI:
                        print("Volume Averaged Jaccard Index (VJI) and segmentation rates:")
                        print(stats_VJI)
                        if self.cfg.TEST.VORONOI_ON_MASK:
                            print("Volume Averaged Jaccard Index (VJI) and segmentation rates (voronoi):")
                            print(stats_vor_VJI)

                if self.cfg.TEST.STATS.FULL_IMG:
                    print("Loss (per image): {}".format(loss))
                    print("Test Foreground IoU (per image): {}".format(iou))
                    print("Test Overall IoU (per image): {}".format(ov_iou))
                    print(" ")

                if post_processing:
                    print("Test Foreground IoU (post-processing): {}".format(iou_post))
                    print("Test Overall IoU (post-processing): {}".format(ov_iou_post))
                    print(" ")
