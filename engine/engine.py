import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

from utils.util import check_masks, create_plots, load_data_from_dir, load_3d_images_from_dir
from data.data_2D_manipulation import load_and_prepare_2D_train_data, load_data_classification
from data.data_3D_manipulation import load_and_prepare_3D_data, load_3d_data_classification
from data.generators import create_train_val_augmentors, create_test_augmentor, check_generator_consistence
from models import build_model
from engine import build_callbacks, prepare_optimizer
from engine.semantic_seg import Semantic_Segmentation
from engine.instance_seg import prepare_instance_data, Instance_Segmentation
from engine.detection import prepare_detection_data, Detection
from engine.classification import Classification
from engine.super_resolution import Super_resolution
from engine.denoising import Denoising
from engine.self_supervised import prepare_ssl_data, Self_supervised

class Engine(object):

    def __init__(self, cfg, job_identifier, num_gpus):
        self.cfg = cfg
        self.job_identifier = job_identifier
        self.original_test_path = None
        self.original_test_mask_path = None
        self.test_mask_filenames = None
        self.cross_val_samples_ids = None
        self.post_processing = {}
        self.post_processing['per_image'] = False
        self.post_processing['all_images'] = False
        self.test_filenames = None 
        self.metric = []
        self.data_norm = None

        # Save paths in case we need them in a future
        self.orig_train_path = cfg.DATA.TRAIN.PATH
        self.orig_train_mask_path = cfg.DATA.TRAIN.GT_PATH
        self.orig_val_path = cfg.DATA.VAL.PATH
        self.orig_val_mask_path = cfg.DATA.VAL.GT_PATH

        print("####################\n"
              "#  PRE-PROCESSING  #\n"
              "####################\n")
        if cfg.PROBLEM.TYPE in ['SEMANTIC_SEG']:
            
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
        elif cfg.PROBLEM.TYPE == 'INSTANCE_SEG':
            self.original_test_path, self.original_test_mask_path = prepare_instance_data(cfg)
        elif cfg.PROBLEM.TYPE == 'DETECTION':
            self.original_test_mask_path = prepare_detection_data(cfg)
        elif cfg.PROBLEM.TYPE == 'SELF_SUPERVISED':
            prepare_ssl_data(cfg)

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
            if cfg.DATA.TRAIN.IN_MEMORY:
                mask_path = cfg.DATA.TRAIN.GT_PATH if cfg.PROBLEM.TYPE != 'DENOISING' else None
                val_split = cfg.DATA.VAL.SPLIT_TRAIN if cfg.DATA.VAL.FROM_TRAIN else 0.
                if cfg.PROBLEM.TYPE != "CLASSIFICATION":
                    f_name = load_and_prepare_2D_train_data if cfg.PROBLEM.NDIM == '2D' else load_and_prepare_3D_data
                    objs = f_name(cfg.DATA.TRAIN.PATH, mask_path, cross_val=cfg.DATA.VAL.CROSS_VAL, 
                        cross_val_nsplits=cfg.DATA.VAL.CROSS_VAL_NFOLD, cross_val_fold=cfg.DATA.VAL.CROSS_VAL_FOLD, 
                        val_split=val_split, seed=cfg.SYSTEM.SEED, shuffle_val=cfg.DATA.VAL.RANDOM, 
                        random_crops_in_DA=cfg.DATA.EXTRACT_RANDOM_PATCH, crop_shape=cfg.DATA.PATCH_SIZE, 
                        y_upscaling=cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING, ov=cfg.DATA.TRAIN.OVERLAP, 
                        padding=cfg.DATA.TRAIN.PADDING, minimum_foreground_perc=cfg.DATA.TRAIN.MINIMUM_FOREGROUND_PER,
                        reflect_to_complete_shape=cfg.DATA.REFLECT_TO_COMPLETE_SHAPE)
                else: # CLASSSIFICATION
                    f_name = load_data_classification if cfg.PROBLEM.NDIM == '2D' else load_3d_data_classification
                    print("0) Loading train images . . .")
                    objs = f_name(cfg.DATA.TRAIN.PATH, cfg.MODEL.N_CLASSES, cross_val=cfg.DATA.VAL.CROSS_VAL, 
                        cross_val_nsplits=cfg.DATA.VAL.CROSS_VAL_NFOLD, cross_val_fold=cfg.DATA.VAL.CROSS_VAL_FOLD, 
                        val_split=val_split, seed=cfg.SYSTEM.SEED, shuffle_val=cfg.DATA.VAL.RANDOM)

                if cfg.DATA.VAL.FROM_TRAIN:
                    if cfg.DATA.VAL.CROSS_VAL:
                        X_train, Y_train, X_val, Y_val, self.train_filenames, self.cross_val_samples_ids  = objs
                    else:
                        X_train, Y_train, X_val, Y_val, self.train_filenames = objs
                else:
                    X_train, Y_train, self.train_filenames = objs
                del objs
            else:
                X_train, Y_train = None, None

            ##################
            ### VALIDATION ###
            ##################
            if not cfg.DATA.VAL.FROM_TRAIN:
                if cfg.DATA.VAL.IN_MEMORY:
                    print("1) Loading validation images . . .")
                    if cfg.PROBLEM.TYPE != "CLASSIFICATION":
                        f_name = load_data_from_dir if cfg.PROBLEM.NDIM == '2D' else load_3d_images_from_dir
                        X_val, _, _ = f_name(cfg.DATA.VAL.PATH, crop=True, crop_shape=cfg.DATA.PATCH_SIZE,
                                            overlap=cfg.DATA.VAL.OVERLAP, padding=cfg.DATA.VAL.PADDING,
                                            reflect_to_complete_shape=cfg.DATA.REFLECT_TO_COMPLETE_SHAPE)
                        if cfg.PROBLEM.TYPE != 'DENOISING':
                            if cfg.PROBLEM.NDIM == '2D':
                                crop_shape = (cfg.DATA.PATCH_SIZE[0]*cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING,
                                    cfg.DATA.PATCH_SIZE[1]*cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING, cfg.DATA.PATCH_SIZE[2])
                            else:
                                crop_shape = (cfg.DATA.PATCH_SIZE[0], cfg.DATA.PATCH_SIZE[1]*cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING,
                                    cfg.DATA.PATCH_SIZE[2]*cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING, cfg.DATA.PATCH_SIZE[3])
                            Y_val, _, _ = f_name(cfg.DATA.VAL.GT_PATH, crop=True, crop_shape=crop_shape,
                                                overlap=cfg.DATA.VAL.OVERLAP, padding=cfg.DATA.VAL.PADDING,
                                                reflect_to_complete_shape=cfg.DATA.REFLECT_TO_COMPLETE_SHAPE,
                                                check_channel=False)
                        else:
                            Y_val = np.zeros(X_val.shape, dtype=np.float32) # Fake mask val
                    else: # Classification
                        f_name = load_data_classification if cfg.PROBLEM.NDIM == '2D' else load_3d_data_classification
                        X_val, Y_val, _ = f_name(cfg.DATA.VAL.PATH, cfg.MODEL.N_CLASSES, val_split=0) 
                else:
                    X_val, Y_val = None, None

        ############
        ### TEST ###
        ############
        if cfg.TEST.ENABLE:
            if not cfg.DATA.TEST.USE_VAL_AS_TEST:
                if cfg.DATA.TEST.IN_MEMORY:
                    print("2) Loading test images . . .")
                    if cfg.PROBLEM.TYPE != "CLASSIFICATION":
                        f_name = load_data_from_dir if cfg.PROBLEM.NDIM == '2D' else load_3d_images_from_dir
                        X_test, _, _ = f_name(cfg.DATA.TEST.PATH)
                        if cfg.DATA.TEST.LOAD_GT or cfg.PROBLEM.TYPE == 'SELF_SUPERVISED':
                            print("3) Loading test masks . . .")
                            Y_test, _, _ = f_name(cfg.DATA.TEST.GT_PATH, check_channel=False)
                        else:
                            Y_test = None
                    else: # CLASSIFICATION
                        f_name = load_data_classification if cfg.PROBLEM.NDIM == '2D' else load_3d_data_classification
                        X_test, Y_test, self.test_filenames = f_name(cfg.DATA.TEST.PATH,  
                            cfg.MODEL.N_CLASSES if cfg.DATA.TEST.LOAD_GT else None, val_split=0)
                        self.class_names = sorted(next(os.walk(cfg.DATA.TEST.PATH))[1])
                else:
                    X_test, Y_test = None, None

                if cfg.PROBLEM.TYPE != "CLASSIFICATION":
                    if self.original_test_path is None:
                        self.test_filenames = sorted(next(os.walk(cfg.DATA.TEST.PATH))[2])
                    else:
                        self.test_filenames = sorted(next(os.walk(self.original_test_path))[2])
                else:
                    self.class_names = sorted(next(os.walk(cfg.DATA.TEST.PATH))[1])
                    if self.test_filenames is None:
                        self.test_filenames = []
                        for c_num, folder in enumerate(self.class_names):
                            self.test_filenames += sorted(next(os.walk(os.path.join(cfg.DATA.TEST.PATH, folder)))[2])
            else:
                # The test is the validation, and as it is only available when validation is obtained from train and when 
                # cross validation is enabled, the test set files reside in the train folder
                
                X_test, Y_test = None, None
                if self.cross_val_samples_ids is None:                      
                    # Split the test as it was the validation when train is not enabled 
                    skf = StratifiedKFold(n_splits=cfg.DATA.VAL.CROSS_VAL_NFOLD, shuffle=cfg.DATA.VAL.RANDOM,
                        random_state=cfg.SYSTEM.SEED)
                    fold = 1
                    test_index = None
                    if cfg.PROBLEM.TYPE != "CLASSIFICATION":
                        self.test_filenames = sorted(next(os.walk(cfg.DATA.TRAIN.PATH))[2])
                        A, B = np.zeros(len(self.test_filenames))  
                    else :
                        self.class_names = sorted(next(os.walk(cfg.DATA.TRAIN.PATH))[2])
                        self.test_filenames = []
                        B = []
                        for c_num, folder in enumerate(self.class_names):
                            ids += sorted(next(os.walk(os.path.join(cfg.DATA.TRAIN.PATH,folder)))[2])
                            B.append((c_num,)*len(ids))
                            self.test_filenames += ids
                        A = np.zeros(len(self.test_filenames)) 
                        B = np.concatenate(B, 0)
                    for _, te_index in skf.split(A, B):
                        if cfg.DATA.VAL.CROSS_VAL_FOLD == fold:
                            self.cross_val_samples_ids = te_index.copy()
                            break
                        fold += 1
                    if len(self.cross_val_samples_ids) > 5:
                        print("Fold number {} used for test data. Printing the first 5 ids: {}".format(fold, self.cross_val_samples_ids[:5]))
                    else:
                        print("Fold number {}. Indexes used in cross validation: {}".format(fold, self.cross_val_samples_ids))
                
                self.test_filenames = [x for i, x in enumerate(self.test_filenames) if i in self.cross_val_samples_ids]
                self.original_test_path = self.orig_train_path
                self.original_test_mask_path = self.orig_train_mask_path                

        print("########################\n"
              "#  PREPARE GENERATORS  #\n"
              "########################\n")
        if cfg.TRAIN.ENABLE:
            self.train_generator, self.val_generator, self.data_norm = create_train_val_augmentors(cfg, X_train, Y_train, X_val, Y_val, num_gpus)
            if cfg.DATA.CHECK_GENERATORS and cfg.PROBLEM.TYPE != 'CLASSIFICATION':
                check_generator_consistence(
                    self.train_generator, cfg.PATHS.GEN_CHECKS+"_train", cfg.PATHS.GEN_MASK_CHECKS+"_train")
                check_generator_consistence(
                    self.val_generator, cfg.PATHS.GEN_CHECKS+"_val", cfg.PATHS.GEN_MASK_CHECKS+"_val")
        if cfg.TEST.ENABLE:
            self.test_generator, self.data_norm = create_test_augmentor(cfg, X_test, Y_test, self.cross_val_samples_ids)

        print("#################\n"
              "#  BUILD MODEL  #\n"
              "#################\n")
        if num_gpus > 1:
            strategy = tf.distribute.MirroredStrategy()
            print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

            # Open a strategy scope.
            with strategy.scope():
                self.model = build_model(cfg, self.job_identifier, self.data_norm)
                self.metric = prepare_optimizer(cfg, self.model)
        else:
            self.model = build_model(cfg, self.job_identifier, self.data_norm)
            self.metric = prepare_optimizer(cfg, self.model)

    def train(self):
        print("#####################\n"
              "#  TRAIN THE MODEL  #\n"
              "#####################\n")
        if self.cfg.MODEL.LOAD_CHECKPOINT:
            print("Loading model weights from h5_file: {}".format(self.cfg.PATHS.CHECKPOINT_FILE))
            self.model.load_weights(self.cfg.PATHS.CHECKPOINT_FILE)

        self.callbacks = build_callbacks(self.cfg, len(self.train_generator)*self.cfg.TRAIN.EPOCHS)
        self.results = self.model.fit(self.train_generator, validation_data=self.val_generator,
            epochs=self.cfg.TRAIN.EPOCHS, callbacks=self.callbacks)

        create_plots(self.results, self.job_identifier, self.cfg.PATHS.CHARTS, metric=self.metric)


    def test(self):
        print("Loading model weights from h5_file: {}".format(self.cfg.PATHS.CHECKPOINT_FILE))
        self.model.load_weights(self.cfg.PATHS.CHECKPOINT_FILE)

        image_counter = 0
        if self.cfg.TEST.ANALIZE_2D_IMGS_AS_3D_STACK and self.cfg.PROBLEM.NDIM == "2D":
            if self.cfg.TEST.POST_PROCESSING.YZ_FILTERING or self.cfg.TEST.POST_PROCESSING.Z_FILTERING:
                self.post_processing['all_images'] = True
        elif self.cfg.PROBLEM.NDIM == "3D":
            if self.cfg.TEST.POST_PROCESSING.YZ_FILTERING or self.cfg.TEST.POST_PROCESSING.Z_FILTERING:
                self.post_processing['per_image'] = True
                
        # Initialize the workflow
        if self.cfg.PROBLEM.TYPE == 'SEMANTIC_SEG':
            workflow = Semantic_Segmentation(self.cfg, self.model, self.post_processing)
        elif self.cfg.PROBLEM.TYPE == 'INSTANCE_SEG':
            # Specific instance segmentation post-processing
            if self.cfg.TEST.POST_PROCESSING.VORONOI_ON_MASK or self.cfg.TEST.POST_PROCESSING.WATERSHED_CIRCULARITY != -1 or\
                self.cfg.TEST.POST_PROCESSING.REPARE_LARGE_BLOBS_SIZE != -1:
                self.post_processing['instance_post'] = True
            else:
                self.post_processing['instance_post'] = False
            workflow = Instance_Segmentation(self.cfg, self.model, self.post_processing, self.original_test_mask_path)
        elif self.cfg.PROBLEM.TYPE == 'DETECTION':
            # Specific detection post-processing
            if self.cfg.TEST.POST_PROCESSING.DET_WATERSHED or self.cfg.TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS:
                self.post_processing['detection_post'] = True
            else:
                self.post_processing['detection_post'] = False
            workflow = Detection(self.cfg, self.model, self.post_processing, self.original_test_mask_path)
        elif self.cfg.PROBLEM.TYPE == 'CLASSIFICATION':
            workflow = Classification(self.cfg, self.model, self.class_names, self.post_processing)
        elif self.cfg.PROBLEM.TYPE == 'SUPER_RESOLUTION':
            workflow = Super_resolution(self.cfg, self.model, self.post_processing)
        elif self.cfg.PROBLEM.TYPE == 'DENOISING':
            workflow = Denoising(self.cfg, self.model, self.post_processing)
        elif self.cfg.PROBLEM.TYPE == 'SELF_SUPERVISED':
            workflow = Self_supervised(self.cfg, self.model, self.post_processing)

        print("###############\n"
              "#  INFERENCE  #\n"
              "###############\n")
        print("Making predictions on test data . . .")

        # Process all the images
        it = iter(self.test_generator)
        for i in tqdm(range(len(self.test_generator))):
            batch = next(it)
            if self.cfg.DATA.TEST.LOAD_GT or self.cfg.PROBLEM.TYPE == 'SELF_SUPERVISED':
                X, X_norm, Y, Y_norm = batch
            else:
                X, X_norm = batch
                Y, Y_norm = None, None
            del batch

            # Process all the images in the batch
            l_X = len(X)
            for j in tqdm(range(l_X), leave=False):
                print("Processing image(s): {}".format(self.test_filenames[(i*l_X)+j:(i*l_X)+j+1]))

                if self.cfg.PROBLEM.TYPE != 'CLASSIFICATION':
                    if type(X) is tuple:
                        _X = X[j]
                        if self.cfg.DATA.TEST.LOAD_GT or self.cfg.PROBLEM.TYPE == 'SELF_SUPERVISED':
                            _Y = Y[j]  
                        else:
                            _Y = None
                    else:
                        _X = np.expand_dims(X[j],0)
                        if self.cfg.DATA.TEST.LOAD_GT or self.cfg.PROBLEM.TYPE == 'SELF_SUPERVISED':
                            _Y = np.expand_dims(Y[j],0)  
                        else:
                            _Y = None
                else:
                    _X = np.expand_dims(X[j], 0)                    
                    _Y = np.expand_dims(Y, 0) if self.cfg.DATA.TEST.LOAD_GT else None

                # Process each image separately
                numbers = list(range((i*l_X)+j,(i*l_X)+j+1)) if self.cross_val_samples_ids is None else self.cross_val_samples_ids[(i*l_X)+j:(i*l_X)+j+1]
                workflow.process_sample(_X, _Y, self.test_filenames[(i*l_X)+j:(i*l_X)+j+1], f_numbers=numbers, 
                    norm=(X_norm, Y_norm))

                image_counter += 1
        del _X, _Y

        workflow.after_all_images()

        print("#############\n"
              "#  RESULTS  #\n"
              "#############\n")

        if self.cfg.TRAIN.ENABLE:
            print("Epoch average time: {}".format(np.mean(self.callbacks[0].times)))
            print("Epoch number: {}".format(len(self.results.history['val_loss'])))
            print("Train time (s): {}".format(np.sum(self.callbacks[0].times)))
            print("Train loss: {}".format(np.min(self.results.history['loss'])))
            for i in range(len(self.metric)):
                if self.metric[i] == "IoU":
                    print("Train Foreground {}: {}".format(self.metric[i], np.max(self.results.history[self.metric[i]])))
                else:
                    print("Train {}: {}".format(self.metric[i], np.max(self.results.history[self.metric[i]])))
            print("Validation loss: {}".format(np.min(self.results.history['val_loss'])))
            for i in range(len(self.metric)):
                if self.metric[i] == "IoU":
                    print("Validation Foreground {}: {}".format(self.metric[i], np.max(self.results.history['val_'+self.metric[i]])))
                else:
                    print("Validation {}: {}".format(self.metric[i], np.max(self.results.history['val_'+self.metric[i]])))
        workflow.print_stats(image_counter)


