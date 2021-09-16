import os
import math
import sys
import h5py
import numpy as np
from tqdm import tqdm
from skimage.io import imread

from utils.util import (check_masks, check_downsample_division, create_plots, save_tif, load_data_from_dir,
                        load_3d_images_from_dir)
from data import create_instance_channels, create_test_instance_channels
from data.data_2D_manipulation import load_and_prepare_2D_train_data, crop_data_with_overlap, merge_data_with_overlap
from data.data_3D_manipulation import load_and_prepare_3D_data, crop_3D_data_with_overlap, merge_3D_data_with_overlap
from data.generators import create_train_val_augmentors, create_test_augmentor, check_generator_consistence
from data.post_processing import apply_post_processing
from data.post_processing.post_processing import (ensemble8_2d_predictions, ensemble16_3d_predictions, bc_watershed,
                                                  bcd_watershed, calculate_optimal_mw_thresholds)
from models import build_model
from engine import build_callbacks, prepare_optimizer
from engine.metrics import jaccard_index_numpy, voc_calculation


class Trainer(object):

    def __init__(self, cfg, job_identifier):
        self.cfg = cfg
        self.job_identifier = job_identifier
        original_test_path = None
        self.test_mask_filenames = None

        # Save paths in case we need them in a future
        self.orig_train_path = cfg.DATA.TRAIN.PATH
        self.orig_train_mask_path = cfg.DATA.TRAIN.MASK_PATH
        self.orig_val_path = cfg.DATA.VAL.PATH
        self.orig_val_mask_path = cfg.DATA.VAL.MASK_PATH

        if cfg.PROBLEM.TYPE == 'SEMANTIC_SEG':
            print("###################\n"
                  "#  SANITY CHECKS  #\n"
                  "###################\n")
            check_masks(cfg.DATA.TRAIN.MASK_PATH)
            check_masks(cfg.DATA.TEST.MASK_PATH)

            # Adjust the metric used accordingly to the number of classes. This code is planned to be used in a binary
            # classification problem, so the function 'jaccard_index_softmax' will only calculate the IoU for the
            # foreground class (channel 1)
            self.metric = "jaccard_index_softmax" if cfg.MODEL.N_CLASSES > 1 else "jaccard_index"
        else:
            self.metric = "jaccard_index_instances" if cfg.DATA.CHANNELS == "BCD" else "jaccard_index"


        if cfg.PROBLEM.TYPE == 'INSTANCE_SEG' and cfg.DATA.CHANNELS != 'B':
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
            if cfg.TRAIN.ENABLE:
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
                original_test_path = cfg.DATA.TEST.PATH
                print("DATA.TEST.PATH changed from {} to {}".format(cfg.DATA.TEST.PATH, cfg.DATA.TEST.INSTANCE_CHANNELS_DIR))
                opts.extend(['DATA.TEST.PATH', cfg.DATA.TEST.INSTANCE_CHANNELS_DIR])
                if cfg.DATA.TEST.LOAD_GT and cfg.TEST.EVALUATE:
                    original_test_mask_path = cfg.DATA.TEST.MASK_PATH
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
            if cfg.DATA.TRAIN.IN_MEMORY:
                if cfg.PROBLEM.NDIM == '2D':
                    objs = load_and_prepare_2D_train_data(cfg.DATA.TRAIN.PATH, cfg.DATA.TRAIN.MASK_PATH,
                        val_split=cfg.DATA.VAL.SPLIT_TRAIN, seed=cfg.SYSTEM.SEED, shuffle_val=cfg.DATA.VAL.RANDOM,
                        random_crops_in_DA=cfg.DATA.EXTRACT_RANDOM_PATCH, crop_shape=cfg.DATA.PATCH_SIZE,
                        ov=cfg.DATA.TRAIN.OVERLAP, padding=cfg.DATA.TRAIN.PADDING, check_crop=cfg.DATA.TRAIN.CHECK_CROP,
                        check_crop_path=cfg.PATHS.CROP_CHECKS)
                else:
                    objs = load_and_prepare_3D_data(cfg.DATA.TRAIN.PATH, cfg.DATA.TRAIN.MASK_PATH,
                        val_split=cfg.DATA.VAL.SPLIT_TRAIN, seed=cfg.SYSTEM.SEED, shuffle_val=cfg.DATA.VAL.RANDOM,
                        random_crops_in_DA=cfg.DATA.EXTRACT_RANDOM_PATCH, crop_shape=cfg.DATA.PATCH_SIZE,
                        ov=cfg.DATA.TRAIN.OVERLAP, padding=cfg.DATA.TRAIN.PADDING)

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
                                         overlap=cfg.DATA.VAL.OVERLAP, padding=cfg.DATA.VAL.PADDING)
                    Y_val, _, _ = f_name(cfg.DATA.VAL.MASK_PATH, crop=True, crop_shape=cfg.DATA.PATCH_SIZE,
                                         overlap=cfg.DATA.VAL.OVERLAP, padding=cfg.DATA.VAL.PADDING)
                else:
                    if not os.path.exists(cfg.DATA.VAL.PATH):
                        raise ValueError("Validation data dir not found: {}".format(cfg.DATA.VAL.PATH))
                    if not os.path.exists(cfg.DATA.VAL.MASK_PATH):
                        raise ValueError("Validation mask data dir not found: {}".format(cfg.DATA.VAL.MASK_PATH))
                    X_val, Y_val = None, None

        ############
        ### TEST ###
        ############
        if cfg.TEST.ENABLE:
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
                    if cfg.PROBLEM.NDIM == '2D':
                        raise ValueError("Not implemented pipeline option: no test data labels when PROBLEM.NDIM == '2D'")
                    else:
                        Y_test = None
            else:
                X_test, Y_test = None, None

            if original_test_path is None:
                self.test_filenames = sorted(next(os.walk(cfg.DATA.TEST.PATH))[2])
                if cfg.TEST.MAP and cfg.DATA.TEST.LOAD_GT:
                    self.test_mask_filenames = sorted(next(os.walk(cfg.DATA.TEST.MASK_PATH))[2])
            else:
                self.test_filenames = sorted(next(os.walk(original_test_path))[2])
                if cfg.TEST.MAP and cfg.DATA.TEST.LOAD_GT:
                    self.test_mask_filenames = sorted(next(os.walk(original_test_path))[2])


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
        iou, ov_iou, c1, c2  = 0, 0, 0, 0

        if self.cfg.TEST.STATS.PER_PATCH:
           loss_per_crop, iou_per_crop, patch_counter = 0, 0, 0
        if self.cfg.TEST.STATS.MERGE_PATCHES:
           loss_per_imag, iou_per_image, ov_iou_per_image = 0, 0, 0
        if self.cfg.TEST.STATS.FULL_IMG:
           loss, iou, ov_iou = 0, 0, 0
        image_counter = 0

        if self.cfg.TEST.POST_PROCESSING.BLENDING or self.cfg.TEST.POST_PROCESSING.YZ_FILTERING or \
           self.cfg.TEST.POST_PROCESSING.Z_FILTERING:
            post_processing, iou_post, ov_iou_post = True, 0, 0
        else:
            post_processing = False

        if self.cfg.TEST.STATS.FULL_IMG and self.cfg.PROBLEM.NDIM == '2D' and post_processing:
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
                if not self.cfg.TEST.VERBOSE:
                    print("Processing image(s): {}".format(self.test_filenames[(i*l_X)+j:(i*l_X)+j+1]))

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

                #################
                ### PER PATCH ###
                #################
                if self.cfg.TEST.STATS.PER_PATCH:
                    original_data_shape = _X.shape if self.cfg.PROBLEM.NDIM == '2D' else _X.shape[1:]
                    if _X.shape[1:] != self.cfg.DATA.PATCH_SIZE:
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
                    if self.cfg.DATA.TEST.LOAD_GT and self.cfg.TEST.EVALUATE:
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
                    if self.cfg.TEST.AUGMENTATION:
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
                            pred.append(p)

                    # Reconstruct the predictions
                    pred = np.concatenate(pred)
                    if original_data_shape != self.cfg.DATA.PATCH_SIZE:
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

                    # Argmax if needed
                    if self.cfg.MODEL.N_CLASSES > 1:
                        pred = np.expand_dims(np.argmax(pred,-1), -1)
                        if self.cfg.DATA.TEST.LOAD_GT: _Y = np.expand_dims(np.argmax(_Y,-1), -1)

                    # Save image
                    filenames = self.test_filenames[(i*l_X)+j:(i*l_X)+j+1]
                    if pred.ndim == 4 and self.cfg.PROBLEM.NDIM == '3D':
                        save_tif(np.expand_dims(pred,0), self.cfg.PATHS.RESULT_DIR.PER_IMAGE, filenames,
                                 verbose=self.cfg.TEST.VERBOSE)
                    else:
                        save_tif(pred, self.cfg.PATHS.RESULT_DIR.PER_IMAGE, filenames, verbose=self.cfg.TEST.VERBOSE)


                    #####################
                    ### MERGE PATCHES ###
                    #####################
                    if self.cfg.TEST.STATS.MERGE_PATCHES:
                        if self.cfg.DATA.TEST.LOAD_GT:
                            _iou_per_image = jaccard_index_numpy((_Y>0.5).astype(np.uint8), (pred[0] > 0.5).astype(np.uint8))
                            _ov_iou_per_image = voc_calculation((_Y>0.5).astype(np.uint8), (pred[0] > 0.5).astype(np.uint8),
                                                                iou_per_image)
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

                    #############################
                    ### INSTANCE SEGMENTATION ###
                    #############################
                    if self.cfg.PROBLEM.TYPE == 'INSTANCE_SEG':
                        if self.cfg.DATA.MW_OPTIMIZE_THS:
                            obj = calculate_optimal_mw_thresholds(self.model, self.cfg.DATA.VAL.PATH,
                                self.cfg.DATA.VAL.MASK_PATH, self.cfg.DATA.CHANNELS, self.cfg.DATA.REMOVE_SMALL_OBJ,
                                verbose=self.cfg.TEST.VERBOSE)
                            if self.cfg.DATA.CHANNELS == "BCD":
                                th1_opt, th2_opt, th3_opt, th4_opt, th5_opt = obj
                            else:
                                th1_opt, th2_opt, th3_opt = obj
                                th4_opt, th5_opt = self.cfg.DATA.MW_TH4, self.cfg.DATA.MW_TH5
                        else:
                            th1_opt, th2_opt, th3_opt = self.cfg.DATA.MW_TH1, self.cfg.DATA.MW_TH2, self.cfg.DATA.MW_TH3
                            th4_opt, th5_opt = self.cfg.DATA.MW_TH4, self.cfg.DATA.MW_TH5

                        # Create instances
                        print("Creating instances with watershed . . .")
                        w_dir = os.path.join(self.cfg.PATHS.WATERSHED_DIR, filenames[0])
                        if self.cfg.DATA.CHANNELS == "BC":
                            pred = bc_watershed(pred, thres1=th1_opt, thres2=th2_opt, thres3=th3_opt,
                                thres_small=self.cfg.DATA.REMOVE_SMALL_OBJ, remove_before=self.cfg.DATA.REMOVE_BEFORE_MW,
                                save_dir=w_dir)
                        else:
                            pred = bcd_watershed(pred, thres1=th1_opt, thres2=th2_opt, thres3=th3_opt, thres4=th4_opt,
                                thres5=th5_opt, thres_small=self.cfg.DATA.REMOVE_SMALL_OBJ,
                                remove_before=self.cfg.DATA.REMOVE_BEFORE_MW, save_dir=w_dir)
                        save_tif(np.expand_dims(np.expand_dims(pred,-1),0), self.cfg.PATHS.RESULT_DIR.PER_IMAGE_INSTANCES,
                                 filenames, verbose=self.cfg.TEST.VERBOSE)

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
                        h5f.create_dataset('dataset', data=pred, compression="lzf")
                        h5f.close()

                        # Prepare mAP call
                        sys.path.insert(0, self.cfg.PATHS.MAP_CODE_DIR)
                        from demo_modified import main as mAP_calculation
                        class Namespace:
                            def __init__(self, **kwargs):
                                self.__dict__.update(kwargs)

                        # Create GT H5 file if it does not exist
                        gt_f = os.path.join(self.cfg.PATHS.TEST_FULL_GT_H5, os.path.splitext(filenames[0])[0]+'.h5')
                        test_file = os.path.join(self.cfg.PATHS.TEST_FULL_GT_H5, filenames[0])
                        if not os.path.isfile(gt_f):
                            print("GT .h5 file needed for mAP calculation is not found in {} so it will be created "
                                  "from its mask: {}".format(gt_f, test_file))

                            if not os.path.isfile(test_file):
                                raise ValueError("The mask is supossed to have the same name as the image")

                            _Y = imread(test_file).squeeze()

                            print("Saving .h5 GT data from array shape: {}".format(_Y.shape))
                            os.makedirs(self.cfg.PATHS.TEST_FULL_GT_H5, exist_ok=True)
                            h5f = h5py.File(gt_f, 'w')
                            h5f.create_dataset('dataset', data=_Y, compression="lzf")
                            h5f.close()

                        # Calculate mAP
                        args = Namespace(gt_seg=gt_f, predict_seg=h5file_name, predict_score='', threshold="5e3, 3e4",
                                         threshold_crumb=64, chunk_size=250, output_name=w_dir, do_txt=1, do_eval=1,
                                         slices="-1")
                        mAP_calculation(args)

                ##################
                ### FULL IMAGE ###
                ##################
                if self.cfg.TEST.STATS.FULL_IMG and self.cfg.PROBLEM.NDIM == '2D':
                    if type(X) is tuple:
                        _X = X[j]
                        _Y = Y[j] if self.cfg.DATA.TEST.LOAD_GT else None
                    else:
                        _X = np.expand_dims(X[j],0)
                        _Y = np.expand_dims(Y[j],0) if self.cfg.DATA.TEST.LOAD_GT else None

                    _X, _ = check_downsample_division(_X, len(self.cfg.MODEL.FEATURE_MAPS)-1)
                    if self.cfg.DATA.TEST.LOAD_GT:
                        _Y, o_test_shape = check_downsample_division(_Y, len(self.cfg.MODEL.FEATURE_MAPS)-1)

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
                        ov_iou += voc_calculation((_Y>0.5).astype(np.uint8), (pred[0]>0.5).astype(np.uint8), score[1])

                    if self.cfg.TEST.STATS.FULL_IMG and self.cfg.PROBLEM.NDIM == '2D' and post_processing:
                        all_pred.append(pred)
                        if self.cfg.DATA.TEST.LOAD_GT: all_gt.append(_Y)

                image_counter += 1
        del pred, _X, _Y

        ############################
        ### POST-PROCESSING (2D) ###
        ############################
        if self.cfg.TEST.STATS.FULL_IMG and self.cfg.PROBLEM.NDIM == '2D' and post_processing:
            all_pred = np.concatenate(all_pred)
            if self.cfg.DATA.TEST.LOAD_GT:
                all_gt = np.concatenate(all_gt)
                iou_post, ov_iou_post = apply_post_processing(self.cfg, all_pred, all_gt)
            else:
                iou_post, ov_iou_post = 0, 0
            save_tif(all_pred, self.cfg.PATHS.RESULT_DIR.FULL_POST_PROCESSING, verbose=self.cfg.TEST.VERBOSE)

        if post_processing and self.cfg.PROBLEM.NDIM == '3D':
            iou_post = iou_post / image_counter
            ov_iou_post = ov_iou_post / image_counter
        if self.cfg.TEST.STATS.PER_PATCH:
            loss_per_crop = loss_per_crop / patch_counter
            iou_per_crop = iou_per_crop / patch_counter
            if self.cfg.TEST.STATS.MERGE_PATCHES:
                iou_per_image = iou_per_image / image_counter
                ov_iou_per_image = ov_iou_per_image / image_counter
        iou = iou / image_counter
        ov_iou = ov_iou / image_counter


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
            if self.cfg.TEST.STATS.PER_PATCH:
                print("Loss (per patch): {}".format(loss_per_crop))
                print("Test Foreground IoU (per patch): {}".format(iou_per_crop))
                print(" ")
                if self.cfg.TEST.STATS.MERGE_PATCHES:
                    print("Test Foreground IoU (merge patches): {}".format(iou_per_image))
                    print("Test Overall IoU (merge patches): {}".format(ov_iou_per_image))
                    print(" ")

            if self.cfg.TEST.STATS.FULL_IMG:
                print("Loss (per image): {}".format(iou))
                print("Test Foreground IoU (per image): {}".format(iou))
                print("Test Overall IoU (per image): {}".format(ov_iou))
                print(" ")

            if post_processing:
                print("Test Foreground IoU (post-processing): {}".format(iou_post))
                print("Test Overall IoU (post-processing): {}".format(ov_iou_post))
                print(" ")

