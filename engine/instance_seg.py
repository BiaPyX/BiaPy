import os
import h5py
import numpy as np
import pandas as pd
from skimage.io import imread

from data.post_processing.post_processing import (watershed_by_channels, calculate_optimal_mw_thresholds, voronoi_on_mask_2, 
                                                  remove_instance_by_circularity_central_slice)
from data.pre_processing import create_instance_channels, create_test_instance_channels
from utils.util import save_tif, wrapper_matching_dataset_lazy
from utils.matching import matching

from engine.base_workflow import Base_Workflow

class Instance_Segmentation(Base_Workflow):
    def __init__(self, cfg, model, post_processing=False, original_test_mask_path=None):
        super().__init__(cfg, model, post_processing)

        self.original_test_mask_path = original_test_mask_path     
        self.all_matching_stats = []
        self.post_processing = post_processing

        if post_processing:
            self.all_matching_stats_post_processing = []            

        self.instance_ths = {}
        self.instance_ths['TH1'] = self.cfg.PROBLEM.INSTANCE_SEG.DATA_MW_TH1
        self.instance_ths['TH2'] = self.cfg.PROBLEM.INSTANCE_SEG.DATA_MW_TH2
        self.instance_ths['TH3'] = self.cfg.PROBLEM.INSTANCE_SEG.DATA_MW_TH3
        self.instance_ths['TH4'] = self.cfg.PROBLEM.INSTANCE_SEG.DATA_MW_TH4
        self.instance_ths['TH5'] = self.cfg.PROBLEM.INSTANCE_SEG.DATA_MW_TH5
        self.instance_ths['TH_POINTS'] = self.cfg.PROBLEM.INSTANCE_SEG.DATA_MW_TH_POINTS

        if self.cfg.PROBLEM.INSTANCE_SEG.DATA_MW_OPTIMIZE_THS and self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS != "BCDv2":
            if self.cfg.TEST.POST_PROCESSING.APPLY_MASK and os.path.isdir(self.cfg.DATA.VAL.BINARY_MASKS):
                bin_mask = self.cfg.DATA.VAL.BINARY_MASKS
            else:
                bin_mask = None
            obj = calculate_optimal_mw_thresholds(self.model, self.cfg.DATA.VAL.PATH,
                self.orig_val_mask_path, self.cfg.DATA.PATCH_SIZE, self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS,
                self.cfg.DATA.VAL.MASK_PATH, self.cfg.PROBLEM.INSTANCE_SEG.DATA_REMOVE_SMALL_OBJ, bin_mask,
                chart_dir=self.cfg.PATHS.CHARTS, verbose=self.cfg.TEST.VERBOSE)
            if self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS == "BCD":
                self.instance_ths['TH1'], self.instance_ths['TH2'], self.instance_ths['TH3'], self.instance_ths['TH4'], self.instance_ths['TH5'] = obj
            else:
                self.instance_ths['TH1'], self.instance_ths['TH2'], self.instance_ths['TH3'] = obj
            
    def after_merge_patches(self, pred, Y, filenames):
        #############################
        ### INSTANCE SEGMENTATION ###
        #############################
        print("Creating instances with watershed . . .")
        w_dir = os.path.join(self.cfg.PATHS.WATERSHED_DIR, filenames[0])
        check_wa = w_dir if self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHECK_MW else None
        
        w_pred = watershed_by_channels(pred, self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS, ths=self.instance_ths, 
            thres_small=self.cfg.PROBLEM.INSTANCE_SEG.DATA_REMOVE_SMALL_OBJ, remove_before=self.cfg.PROBLEM.INSTANCE_SEG.DATA_REMOVE_BEFORE_MW, 
            seed_morph_sequence=self.cfg.PROBLEM.INSTANCE_SEG.SEED_MORPH_SEQUENCE, seed_morph_radius=self.cfg.PROBLEM.INSTANCE_SEG.SEED_MORPH_RADIUS, 
            erode_and_dilate_foreground=self.cfg.PROBLEM.INSTANCE_SEG.ERODE_AND_DILATE_FOREGROUND, fore_erosion_radius=self.cfg.PROBLEM.INSTANCE_SEG.FORE_EROSION_RADIUS, 
            fore_dilation_radius=self.cfg.PROBLEM.INSTANCE_SEG.FORE_DILATION_RADIUS, save_dir=check_wa)

        save_tif(np.expand_dims(np.expand_dims(w_pred,-1),0), self.cfg.PATHS.RESULT_DIR.PER_IMAGE_INSTANCES,
            filenames, verbose=self.cfg.TEST.VERBOSE)
   
        # Add extra dimension if working in 2D
        if w_pred.ndim == 2:
            w_pred = np.expand_dims(w_pred,0)

        if self.cfg.TEST.MATCHING_STATS and self.cfg.DATA.TEST.LOAD_GT:
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
            self.all_matching_stats.append(r_stats)


        ###################
        # Post-processing #
        ###################
        if self.cfg.TEST.POST_PROCESSING.WATERSHED_CIRCULARITY != -1:
            w_pred, labels, npixels, areas, circularities, diameters, comment = remove_instance_by_circularity_central_slice(w_pred, self.cfg.DATA.TEST.RESOLUTION, 
                circularity_th=self.cfg.TEST.POST_PROCESSING.WATERSHED_CIRCULARITY)
            # Save stats
            size_measure = 'area' if w_pred.ndim == 2 else 'volume'
            df = pd.DataFrame(zip(labels, npixels, areas, circularities, diameters, comment), 
                columns=['label','npixels', size_measure, 'circularity', 'diameter', 'comment'])
            df = df.sort_values(by=['label'])   
            df.to_csv(os.path.join(self.cfg.PATHS.RESULT_DIR.PER_IMAGE_INSTANCES, os.path.splitext(filenames[0])[0]+'_stats.csv'))
            del df
            
        if self.cfg.TEST.POST_PROCESSING.VORONOI_ON_MASK:
            w_pred = voronoi_on_mask_2(np.expand_dims(w_pred,0), np.expand_dims(pred,0),
                self.cfg.PATHS.RESULT_DIR.PER_IMAGE_INST_VORONOI, filenames, verbose=self.cfg.TEST.VERBOSE)[0]

        if self.post_processing:
            save_tif(np.expand_dims(np.expand_dims(w_pred,-1),0), self.cfg.PATHS.RESULT_DIR.PER_IMAGE_POST_PROCESSING,
                filenames, verbose=self.cfg.TEST.VERBOSE)

            if self.cfg.TEST.MATCHING_STATS and self.cfg.DATA.TEST.LOAD_GT:
                print("Calculating matching stats after post-processing . . .")
                r_stats = matching(_Y, w_pred, thresh=self.cfg.TEST.MATCHING_STATS_THS, report_matches=False)
                print(r_stats)
                self.all_matching_stats_post_processing.append(r_stats)


    def after_full_image(self, pred, Y, filenames):
        pass

    def after_all_images(self):
        super().after_all_images()

    def normalize_stats(self, image_counter): 
        super().normalize_stats(image_counter) 

        if self.cfg.DATA.TEST.LOAD_GT:
            if self.cfg.TEST.MATCHING_STATS:
                self.stats['inst_stats'] = wrapper_matching_dataset_lazy(self.all_matching_stats, self.cfg.TEST.MATCHING_STATS_THS)
                if self.post_processing:
                    self.stats['inst_stats_vor'] = wrapper_matching_dataset_lazy(self.all_matching_stats_post_processing, self.cfg.TEST.MATCHING_STATS_THS)

    def print_stats(self, image_counter):
        super().print_stats(image_counter)

        if self.cfg.DATA.TEST.LOAD_GT:
            if self.cfg.TEST.MATCHING_STATS:
                for i in range(len(self.cfg.TEST.MATCHING_STATS_THS)):
                    print("IoU TH={}".format(self.cfg.TEST.MATCHING_STATS_THS[i]))
                    print(self.stats['inst_stats'][i])
                    if self.post_processing:
                        print("IoU (post-processing) TH={}".format(self.cfg.TEST.MATCHING_STATS_THS[i]))
                        print(self.stats['inst_stats_vor'][i])

        super().print_post_processing_stats()

def prepare_instance_data(cfg):
    print("###########################\n"
           "#  PREPARE INSTANCE DATA  #\n"
           "###########################\n")
    original_test_path, original_test_mask_path = None, None

    # Create selected channels for train data
    if cfg.TRAIN.ENABLE and not os.path.isdir(cfg.DATA.TRAIN.INSTANCE_CHANNELS_DIR):
        print("You select to create {} channels from given instance labels and no file is detected in {}. "
                "So let's prepare the data. Notice that, if you do not modify 'DATA.TRAIN.INSTANCE_CHANNELS_DIR' "
                "path, this process will be done just once!".format(cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS,
                cfg.DATA.TRAIN.INSTANCE_CHANNELS_DIR))
        create_instance_channels(cfg)

    # Create selected channels for val data
    if cfg.TRAIN.ENABLE and not cfg.DATA.VAL.FROM_TRAIN and not os.path.isdir(cfg.DATA.VAL.INSTANCE_CHANNELS_DIR):
        print("You select to create {} channels from given instance labels and no file is detected in {}. "
                "So let's prepare the data. Notice that, if you do not modify 'DATA.VAL.INSTANCE_CHANNELS_DIR' "
                "path, this process will be done just once!".format(cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS,
                cfg.DATA.VAL.INSTANCE_CHANNELS_DIR))
        create_instance_channels(cfg, data_type='val')

    # Create selected channels for test data once
    if cfg.TEST.ENABLE and not os.path.isdir(cfg.DATA.TEST.INSTANCE_CHANNELS_DIR):
        print("You select to create {} channels from given instance labels and no file is detected in {}. "
                "So let's prepare the data. Notice that, if you do not modify 'DATA.TEST.INSTANCE_CHANNELS_DIR' "
                "path, this process will be done just once!".format(cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS,
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
        print("DATA.TEST.PATH changed from {} to {}".format(cfg.DATA.TEST.PATH, cfg.DATA.TEST.INSTANCE_CHANNELS_DIR))
        opts.extend(['DATA.TEST.PATH', cfg.DATA.TEST.INSTANCE_CHANNELS_DIR])
        original_test_path = cfg.DATA.TEST.PATH
        if cfg.DATA.TEST.LOAD_GT:
            print("DATA.TEST.MASK_PATH changed from {} to {}".format(cfg.DATA.TEST.MASK_PATH, cfg.DATA.TEST.INSTANCE_CHANNELS_MASK_DIR))
            opts.extend(['DATA.TEST.MASK_PATH', cfg.DATA.TEST.INSTANCE_CHANNELS_MASK_DIR])
        original_test_mask_path = cfg.DATA.TEST.MASK_PATH
    cfg.merge_from_list(opts)

    return original_test_path, original_test_mask_path

