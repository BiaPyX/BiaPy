import os
import numpy as np
from tqdm import tqdm

from utils.util import check_masks, create_plots, load_data_from_dir, load_3d_images_from_dir
from data import data_checks
from data.data_2D_manipulation import load_and_prepare_2D_train_data, load_data_classification
from data.data_3D_manipulation import load_and_prepare_3D_data
from data.generators import create_train_val_augmentors, create_test_augmentor, check_generator_consistence
from models import build_model
from engine import build_callbacks, prepare_optimizer
from engine.semantic_seg import Semantic_Segmentation
from engine.instance_seg import prepare_instance_data, Instance_Segmentation
from engine.detection import Detection
from engine.classification import Classification
from engine.super_resolution import Super_resolution

class Engine(object):

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
            if cfg.TRAIN.ENABLE and cfg.DATA.TRAIN.CHECK_DATA:
                if cfg.LOSS.TYPE == 'MASKED_BCE':
                    check_masks(cfg.DATA.TRAIN.MASK_PATH, n_classes=3)
                else:
                    check_masks(cfg.DATA.TRAIN.MASK_PATH)
                if not cfg.DATA.VAL.FROM_TRAIN:
                    if cfg.LOSS.TYPE == 'MASKED_BCE':
                        check_masks(cfg.DATA.VAL.MASK_PATH, n_classes=3)
                    else:
                        check_masks(cfg.DATA.VAL.MASK_PATH)
            if cfg.TEST.ENABLE and cfg.DATA.TEST.LOAD_GT and cfg.DATA.TEST.CHECK_DATA:
                if cfg.LOSS.TYPE == 'MASKED_BCE':
                    check_masks(cfg.DATA.TEST.MASK_PATH, n_classes=3)
                else:
                    check_masks(cfg.DATA.TEST.MASK_PATH)

            # Adjust the metric used accordingly to the number of classes. This code is planned to be used in a binary
            # classification problem, so the function 'jaccard_index_softmax' will only calculate the IoU for the
            # foreground class (channel 1)
            self.metric = "jaccard_index_softmax" if cfg.MODEL.N_CLASSES > 1 else "jaccard_index"
            if cfg.LOSS.TYPE == 'MASKED_BCE':
                self.metric = "masked_jaccard_index"
        elif cfg.PROBLEM.TYPE == 'INSTANCE_SEG':
            if  cfg.DATA.CHANNELS == 'Dv2':
                self.metric = "mse"
            else:
                self.metric = "jaccard_index_instances"
        elif cfg.PROBLEM.TYPE == 'CLASSIFICATION':
            self.metric = "accuracy"
        elif cfg.PROBLEM.TYPE == 'SUPER_RESOLUTION':
            self.metric = "PSNR"
        else:
            raise ValueError("Undefined 'PROBLEM.TYPE' {}".format(cfg.PROBLEM.TYPE))

        if cfg.PROBLEM.TYPE == 'INSTANCE_SEG':
            train_filenames = prepare_instance_data(cfg)
            self.train_filenames = train_filenames

        # Adjust overlap and padding for the problem type
        data_checks(cfg)

        # From now on, no modification of the cfg will be allowed
        cfg.freeze()

        if (cfg.TRAIN.ENABLE and cfg.DATA.TRAIN.IN_MEMORY) or (cfg.TEST.ENABLE and cfg.DATA.TEST.IN_MEMORY):
            print("#################\n"
                  "#   LOAD DATA   #\n"
                  "#################\n")

        #############
        ### TRAIN ###
        #############
        if cfg.TRAIN.ENABLE:
            if cfg.PROBLEM.TYPE in ['SEMANTIC_SEG', 'INSTANCE_SEG', 'DETECTION', 'SUPER_RESOLUTION']:
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
            if cfg.PROBLEM.TYPE in ['SEMANTIC_SEG', 'INSTANCE_SEG', 'DETECTION', 'SUPER_RESOLUTION']:
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
            elif cfg.PROBLEM.TYPE == 'CLASSIFICATION':
                X_test, Y_test, self.test_filenames = load_data_classification(cfg, test=True)
            else:
                raise ValueError("Undefined 'PROBLEM.TYPE' {}".format(cfg.PROBLEM.TYPE))


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

        image_counter = 0
        if self.cfg.TEST.POST_PROCESSING.BLENDING or self.cfg.TEST.POST_PROCESSING.YZ_FILTERING or \
           self.cfg.TEST.POST_PROCESSING.Z_FILTERING:
            post_processing = True
        else:
            post_processing = False

        # Initialize the workflow
        if self.cfg.PROBLEM.TYPE == 'SEMANTIC_SEG':
            workflow = Semantic_Segmentation(self.cfg, self.model, post_processing)
        elif self.cfg.PROBLEM.TYPE == 'INSTANCE_SEG':
            workflow = Instance_Segmentation(self.cfg, self.model, post_processing)
        elif self.cfg.PROBLEM.TYPE == 'DETECTION':
            workflow = Detection(self.cfg, self.model, post_processing)
        elif self.cfg.PROBLEM.TYPE == 'CLASSIFICATION':
            workflow = Classification(self.cfg, self.model, post_processing)
        elif self.cfg.PROBLEM.TYPE == 'SUPER_RESOLUTION':
            workflow = Super_resolution(self.cfg, self.model, post_processing)
        else:
            raise ValueError("Undefined 'PROBLEM.TYPE' {}".format(self.cfg.PROBLEM.TYPE))

        print("###############\n"
              "#  INFERENCE  #\n"
              "###############\n")
        print("Making predictions on test data . . .")

        # Process all the images
        it = iter(self.test_generator)
        for i in tqdm(range(len(self.test_generator))):
            batch = next(it)
            if self.cfg.DATA.TEST.LOAD_GT:
                X, Y = batch
            else:
                X, Y = batch, None
            del batch

            # Process all the images in the batch
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
                else:
                    _X = np.expand_dims(X[j], 0)
                    _Y = np.expand_dims(Y[j], 0)

                # Save memory if possible
                if l_X == 1: 
                    del X
                    if self.cfg.DATA.TEST.LOAD_GT:
                        del Y

                # Process each image separately
                workflow.process_sample(_X, _Y, self.test_filenames[(i*l_X)+j:(i*l_X)+j+1])

                image_counter += 1
        del _X, _Y

        workflow.after_all_images(Y)

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

        workflow.print_stats(image_counter)


