import os
import h5py
import numpy as np
from skimage.io import imread

from data.post_processing.post_processing import (bc_watershed,bcd_watershed, bdv2_watershed, calculate_optimal_mw_thresholds,
                                                  voronoi_on_mask_2)
from data.pre_processing import create_instance_channels, create_test_instance_channels
from utils.util import save_tif, wrapper_matching_dataset_lazy, wrapper_matching_segCompare
from utils.matching import matching, match_using_segCompare

from engine.base_workflow import Base_Workflow

class Instance_Segmentation(Base_Workflow):
    def __init__(self, cfg, model, post_processing=False, original_test_mask_path=None):
        super().__init__(cfg, model, post_processing)

        self.original_test_mask_path = original_test_mask_path
        self.stats['mAP_50_total'] = 0
        self.stats['mAP_75_total'] = 0
        self.stats['mAP_50_total_vor'] = 0
        self.stats['mAP_75_total_vor'] = 0      
        self.all_matching_stats = []
        self.all_matching_stats_voronoi = []
        self.all_matching_stats_segCompare = []
        self.all_matching_stats_voronoi_segCompare = []                   

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
                self.th1_opt, self.th2_opt, self.th3_opt, self.th4_opt, self.th5_opt = obj
            else:
                self.th1_opt, self.th2_opt, self.th3_opt = obj
                self.th4_opt, self.th5_opt = self.cfg.PROBLEM.INSTANCE_SEG.DATA_MW_TH4, self.cfg.PROBLEM.INSTANCE_SEG.DATA_MW_TH5
        else:
            self.th1_opt, self.th2_opt, self.th3_opt = self.cfg.PROBLEM.INSTANCE_SEG.DATA_MW_TH1, self.cfg.PROBLEM.INSTANCE_SEG.DATA_MW_TH2, self.cfg.PROBLEM.INSTANCE_SEG.DATA_MW_TH3
            self.th4_opt, self.th5_opt = self.cfg.PROBLEM.INSTANCE_SEG.DATA_MW_TH4, self.cfg.PROBLEM.INSTANCE_SEG.DATA_MW_TH5
            
    def after_merge_patches(self, pred, Y, filenames):
        #############################
        ### INSTANCE SEGMENTATION ###
        #############################
        print("Creating instances with watershed . . .")
        w_dir = os.path.join(self.cfg.PATHS.WATERSHED_DIR, filenames[0])
        check_wa = w_dir if self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHECK_MW else None
        if self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS in ["BC", "BCM"]:
            w_pred = bc_watershed(pred, thres1=self.th1_opt, thres2=self.th2_opt, thres3=self.th3_opt,
                thres_small=self.cfg.PROBLEM.INSTANCE_SEG.DATA_REMOVE_SMALL_OBJ, remove_before=self.cfg.PROBLEM.INSTANCE_SEG.DATA_REMOVE_BEFORE_MW,
                save_dir=check_wa)
        elif self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS == "BCD":
            w_pred = bcd_watershed(pred, thres1=self.th1_opt, thres2=self.th2_opt, thres3=self.th3_opt, thres4=self.th4_opt,
                thres5=self.th5_opt, thres_small=self.cfg.PROBLEM.INSTANCE_SEG.DATA_REMOVE_SMALL_OBJ,
                remove_before=self.cfg.PROBLEM.INSTANCE_SEG.DATA_REMOVE_BEFORE_MW, save_dir=check_wa)
        else: # "BCDv2"
            w_pred = bdv2_watershed(pred, bin_th=self.th1_opt, thres_small=self.cfg.PROBLEM.INSTANCE_SEG.DATA_REMOVE_SMALL_OBJ,
                remove_before=self.cfg.PROBLEM.INSTANCE_SEG.DATA_REMOVE_BEFORE_MW, save_dir=check_wa)

        save_tif(np.expand_dims(np.expand_dims(w_pred,-1),0), self.cfg.PATHS.RESULT_DIR.PER_IMAGE_INSTANCES,
                    filenames, verbose=self.cfg.TEST.VERBOSE)

        if self.cfg.TEST.POST_PROCESSING.VORONOI_ON_MASK:
            vor_pred = voronoi_on_mask_2(np.expand_dims(w_pred,0), np.expand_dims(pred,0),
                self.cfg.PATHS.RESULT_DIR.PER_IMAGE_INST_VORONOI, filenames, verbose=self.cfg.TEST.VERBOSE)[0]

        # Add extra dimension if working in 2D
        if w_pred.ndim == 2:
            w_pred = np.expand_dims(w_pred,0)

        if self.cfg.TEST.MAP and self.cfg.DATA.TEST.LOAD_GT:
            print("####################\n"
                  "#  mAP Calculation #\n"
                  "####################\n")

            # Convert the prediction into an .h5 file
            os.makedirs(self.cfg.PATHS.MAP_H5_DIR, exist_ok=True)
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

            if self.cfg.TEST.POST_PROCESSING.VORONOI_ON_MASK:
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

            if self.cfg.TEST.POST_PROCESSING.VORONOI_ON_MASK:
                r_stats = matching(_Y, vor_pred, thresh=self.cfg.TEST.MATCHING_STATS_THS, report_matches=False)
                print("Stats with Voronoi")
                print(r_stats)
                self.all_matching_stats_voronoi.append(r_stats)

        if self.cfg.TEST.MATCHING_SEGCOMPARE and self.cfg.DATA.TEST.LOAD_GT:
            print("Calculating matching stats using segCompare. . .")
            test_file = os.path.join(self.cfg.DATA.TEST.MASK_PATH, filenames[0])
            if not os.path.isfile(test_file):
                raise ValueError("The mask is supossed to have the same name as the image")
            _Y = imread(test_file).squeeze()
            r_stats_segCompare = match_using_segCompare(_Y, w_pred)
            print(r_stats_segCompare)
            self.all_matching_stats_segCompare.append(r_stats_segCompare)

            if self.cfg.TEST.POST_PROCESSING.VORONOI_ON_MASK:
                r_stats_segCompare = match_using_segCompare(_Y, vor_pred)
                print("Stats with Voronoi")
                print(r_stats_segCompare)
                self.all_matching_stats_voronoi_segCompare.append(r_stats_segCompare)

    def after_full_image(self, pred, Y, filenames):
        pass

    def after_all_images(self, Y):
        super().after_all_images(None)

    def normalize_stats(self, image_counter): 
        super().normalize_stats(image_counter) 

        if self.cfg.DATA.TEST.LOAD_GT:
            if self.cfg.TEST.MAP: 
                self.stats['mAP_50_total'] = self.stats['mAP_50_total'] / image_counter
                self.stats['mAP_75_total'] = self.stats['mAP_75_total'] / image_counter
                if self.cfg.TEST.POST_PROCESSING.VORONOI_ON_MASK:
                    self.stats['mAP_50_total_vor'] = self.stats['mAP_50_total_vor'] / image_counter
                    self.stats['mAP_75_total_vor'] = self.stats['mAP_75_total_vor'] / image_counter

            if self.cfg.TEST.MATCHING_STATS:
                self.stats['inst_stats'] = wrapper_matching_dataset_lazy(self.all_matching_stats, self.cfg.TEST.MATCHING_STATS_THS)
                if self.cfg.TEST.POST_PROCESSING.VORONOI_ON_MASK:
                    self.stats['inst_stats_vor'] = wrapper_matching_dataset_lazy(self.all_matching_stats_voronoi, self.cfg.TEST.MATCHING_STATS_THS)

            if self.cfg.TEST.MATCHING_SEGCOMPARE:
                self.stats['inst_stats_segCompare'] = wrapper_matching_segCompare(self.all_matching_stats_segCompare)
                if self.cfg.TEST.POST_PROCESSING.VORONOI_ON_MASK:
                    self.stats['inst_stats_vor_segCompare'] = wrapper_matching_segCompare(self.all_matching_stats_voronoi_segCompare)

    def print_stats(self, image_counter):
        super().print_stats(image_counter)

        if self.cfg.DATA.TEST.LOAD_GT:
            if self.cfg.TEST.MAP:
                print("Test Average Precision (AP) - IoU=0.50 : {}".format(self.stats['mAP_50_total']))
                print("Test Average Precision (AP) - IoU=0.75 : {}".format(self.stats['mAP_75_total']))
                print(" ")
                if self.cfg.TEST.POST_PROCESSING.VORONOI_ON_MASK:
                    print("Test Average Precision (AP) (Voronoi) - IoU=0.50 : {}".format(self.stats['inst_mAP_50_total_vor']))
                    print("Test Average Precision (AP) (Voronoi) - IoU=0.75 : {}".format(self.stats['mAP_75_total_vor']))
                    print(" ")
            if self.cfg.TEST.MATCHING_STATS:
                for i in range(len(self.cfg.TEST.MATCHING_STATS_THS)):
                    print("IoU TH={}".format(self.cfg.TEST.MATCHING_STATS_THS[i]))
                    print(self.stats['inst_stats'][i])
                    if self.cfg.TEST.POST_PROCESSING.VORONOI_ON_MASK:
                        print("IoU (Voronoi) TH={}".format(self.cfg.TEST.MATCHING_STATS_THS[i]))
                        print(self.stats['inst_stats_vor'][i])

            if self.cfg.TEST.MATCHING_SEGCOMPARE:
                print("segCompare segmentation rates:")
                print(self.stats['inst_stats_segCompare'])
                if self.cfg.TEST.POST_PROCESSING.VORONOI_ON_MASK:
                    print("segCompare segmentation rates (voronoi):")
                    print(self.stats['inst_stats_vor_segCompare'])
                    
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

