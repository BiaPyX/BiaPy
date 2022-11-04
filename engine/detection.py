import os
import csv
import numpy as np
from skimage.feature import peak_local_max
from scipy.ndimage.morphology import grey_dilation
from skimage.measure import label, regionprops_table
from data.post_processing.post_processing import remove_close_points
from data.pre_processing import create_detection_masks

from utils.util import save_tif
from engine.metrics import detection_metrics
from engine.base_workflow import Base_Workflow

class Detection(Base_Workflow):
    def __init__(self, cfg, model, post_processing=False):
        super().__init__(cfg, model, post_processing)

        self.cell_count_file = os.path.join(self.cfg.PATHS.RESULT_DIR.PATH, 'cell_counter.csv')
        self.cell_count_lines = []

        self.stats['d_precision'] = 0
        self.stats['d_recall'] = 0
        self.stats['d_f1'] = 0

        self.stats['d_precision_per_crop'] = 0
        self.stats['d_recall_per_crop'] = 0
        self.stats['d_f1_per_crop'] = 0

    def detection_process(self, pred, Y, filenames, metric_names=[]):
        if self.cfg.TEST.DET_LOCAL_MAX_COORDS:
            print("Capturing the local maxima ")
            all_channel_coord = []
            for channel in range(pred.shape[-1]):
                if len(self.cfg.TEST.DET_MIN_TH_TO_BE_PEAK) == 1:
                    min_th_peak = self.cfg.TEST.DET_MIN_TH_TO_BE_PEAK[0]
                else:
                    min_th_peak = self.cfg.TEST.DET_MIN_TH_TO_BE_PEAK[channel]
                pred_coordinates = peak_local_max(pred[...,channel], threshold_abs=min_th_peak, exclude_border=False)

                # Remove close points as post-processing method
                if self.cfg.TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS:
                    ndim = 2 if self.cfg.PROBLEM.NDIM == "2D" else 3
                    if len(self.cfg.TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS_RADIUS) == 1:
                        radius = self.cfg.TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS_RADIUS[0]
                    else:
                        radius = self.cfg.TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS_RADIUS[channel]
                    pred_coordinates = remove_close_points(pred_coordinates, radius, self.cfg.DATA.TEST.RESOLUTION,
                        ndim=ndim)

                all_channel_coord.append(pred_coordinates)   

            # Create a file that represent the local maxima
            points_pred = np.zeros((pred.shape[:-1] + (1,)), dtype=np.uint8)
            for n, pred_coordinates in enumerate(all_channel_coord):
                for coord in pred_coordinates:
                        z,y,x = coord
                        points_pred[z,y,x,0] = n+1
                self.cell_count_lines.append([filenames, len(pred_coordinates)])

            if self.cfg.PROBLEM.NDIM == '3D':
                for z_index in range(len(points_pred)):
                    points_pred[z_index] = grey_dilation(points_pred[z_index], size=(3,3,1))
            else:
                points_pred = grey_dilation(points_pred, size=(3,3))

            save_tif(np.expand_dims(points_pred,0), self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK,
                     filenames, verbose=self.cfg.TEST.VERBOSE)
            del points_pred

            all_channel_d_metrics = [0,0,0]
            for ch, pred_coordinates in enumerate(all_channel_coord):
                # Save coords in a couple of csv files
                f1 = os.path.join(self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK, 
                                  os.path.splitext(filenames[0])[0]+'_class'+str(ch+1)+'.csv')
                f2 = os.path.join(self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK, 
                                  os.path.splitext(filenames[0])[0]+'_class'+str(ch+1)+'_prob.csv')
                file1 = open(f1, 'w', newline="")
                file2 = open(f2, 'w', newline="")
                csvwriter1 = csv.writer(file1)
                csvwriter2 = csv.writer(file2)
                csvwriter1.writerow(['index', 'axis-0', 'axis-1', 'axis-2'])
                csvwriter2.writerow(['index', 'axis-0', 'axis-1', 'axis-2', 'probability'])
                for nr in range(len(pred_coordinates)):
                    csvwriter1.writerow([nr+1] + pred_coordinates[nr].tolist())
                    prob = pred[pred_coordinates[nr][0],pred_coordinates[nr][1],pred_coordinates[nr][2],ch]
                    csvwriter2.writerow([nr+1] + pred_coordinates[nr].tolist() + [prob])
                file1.close()
                file2.close()

                # Calculate detection metrics
                if self.cfg.DATA.TEST.LOAD_GT:
                    exclusion_mask = Y[...,ch] > 0
                    bin_Y = Y[...,ch] * exclusion_mask.astype( float )
                    props = regionprops_table(label( bin_Y ), properties=('area','centroid'))
                    gt_coordinates = []
                    for n in range(len(props['centroid-0'])):
                        gt_coordinates.append([props['centroid-0'][n], props['centroid-1'][n], props['centroid-2'][n]])
                    gt_coordinates = np.array(gt_coordinates)

                    if self.cfg.PROBLEM.NDIM == '3D':
                        v_size = (self.cfg.DATA.TEST.RESOLUTION[2], self.cfg.DATA.TEST.RESOLUTION[1], self.cfg.DATA.TEST.RESOLUTION[0])
                    else:
                        v_size = (1,self.cfg.DATA.TEST.RESOLUTION[1], self.cfg.DATA.TEST.RESOLUTION[0])
                    print("Detection (class "+str(ch+1)+")")
                    d_metrics = detection_metrics(gt_coordinates, pred_coordinates, tolerance=self.cfg.TEST.DET_TOLERANCE[ch],
                                                  voxel_size=v_size, verbose=self.cfg.TEST.VERBOSE)
                    print("Detection metrics: {}".format(d_metrics))
                    all_channel_d_metrics[0] += d_metrics[1]
                    all_channel_d_metrics[1] += d_metrics[3]
                    all_channel_d_metrics[2] += d_metrics[5]

            if self.cfg.DATA.TEST.LOAD_GT:
                print("All classes "+str(ch+1))
                all_channel_d_metrics[0] = all_channel_d_metrics[0]/Y.shape[-1]
                all_channel_d_metrics[1] = all_channel_d_metrics[1]/Y.shape[-1]
                all_channel_d_metrics[2] = all_channel_d_metrics[2]/Y.shape[-1]
                print("Detection metrics: {}".format(["Precision", all_channel_d_metrics[0],
                                                        "Recall", all_channel_d_metrics[1],
                                                        "F1", all_channel_d_metrics[2]]))

                self.stats[metric_names[0]] += all_channel_d_metrics[0]
                self.stats[metric_names[1]] += all_channel_d_metrics[1]
                self.stats[metric_names[2]] += all_channel_d_metrics[2]
            
    def normalize_stats(self, image_counter):
        super().normalize_stats(image_counter)

        with open(self.cell_count_file, 'w', newline="") as file:
            csvwriter = csv.writer(file)
            csvwriter.writerow(['filename', 'cells'])
            for nr in range(len(self.cell_count_lines)):
                csvwriter.writerow([nr+1] + self.cell_count_lines[nr])
        if self.cfg.DATA.TEST.LOAD_GT:
            if self.cfg.TEST.STATS.PER_PATCH:
                self.stats['d_precision_per_crop'] = self.stats['d_precision_per_crop'] / image_counter
                self.stats['d_recall_per_crop'] = self.stats['d_recall_per_crop'] / image_counter
                self.stats['d_f1_per_crop'] = self.stats['d_f1_per_crop'] / image_counter
            if self.cfg.TEST.STATS.FULL_IMG:
                self.stats['d_precision'] = self.stats['d_precision'] / image_counter
                self.stats['d_recall'] = self.stats['d_recall'] / image_counter
                self.stats['d_f1'] = self.stats['d_f1'] / image_counter

    def after_merge_patches(self, pred, Y, filenames):
        self.detection_process(pred, Y, filenames, ['d_precision_per_crop', 'd_recall_per_crop', 'd_f1_per_crop'])

    def after_full_image(self, pred, Y, filenames):
        self.detection_process(pred, Y, filenames, ['d_precision', 'd_recall', 'd_f1'])

    def after_all_images(self, Y):
        super().after_all_images(None)

    def print_stats(self, image_counter):
        super().print_stats(image_counter)

        if self.cfg.DATA.TEST.LOAD_GT:
            if self.cfg.TEST.STATS.PER_PATCH:
                print("Detection - Test Precision (merge patches): {}".format(self.stats['d_precision_per_crop']))
                print("Detection - Test Recall (merge patches): {}".format(self.stats['d_recall_per_crop']))
                print("Detection - Test F1 (merge patches): {}".format(self.stats['d_f1_per_crop']))
            if self.cfg.TEST.STATS.FULL_IMG:
                print("Detection - Test Precision (per image): {}".format(self.stats['d_precision']))
                print("Detection - Test Recall (per image): {}".format(self.stats['d_recall']))
                print("Detection - Test F1 (per image): {}".format(self.stats['d_f1']))

        super().print_post_processing_stats()


def prepare_detection_data(cfg):
    print("#############################\n"
          "#  PREPARE DETECTION DATA  #\n"
          "############################\n")

    # Create selected channels for train data
    if cfg.TRAIN.ENABLE:
        create_mask = False
        if not os.path.isdir(cfg.DATA.TRAIN.DETECTION_MASK_DIR):
            print("You select to create detection masks from given .csv files but no file is detected in {}. "
                  "So let's prepare the data. Notice that, if you do not modify 'DATA.TRAIN.DETECTION_MASK_DIR' "
                  "path, this process will be done just once!".format(cfg.DATA.TRAIN.DETECTION_MASK_DIR))
            create_mask = True
        else:
            if len(next(os.walk(cfg.DATA.TRAIN.DETECTION_MASK_DIR))[2]) != len(next(os.walk(cfg.DATA.TRAIN.MASK_PATH))[2]):
                print("Different number of files found in {} and {}. Trying to create the the rest again"
                       .format(cfg.DATA.TRAIN.MASK_PATH,cfg.DATA.TRAIN.DETECTION_MASK_DIR))
                create_mask = True    

        if create_mask:
            create_detection_masks(cfg)

    # Create selected channels for val data
    if cfg.TRAIN.ENABLE and not cfg.DATA.VAL.FROM_TRAIN:
        create_mask = False
        if not os.path.isdir(cfg.DATA.VAL.DETECTION_MASK_DIR):
            print("You select to create detection masks from given .csv files but no file is detected in {}. "
                "So let's prepare the data. Notice that, if you do not modify 'DATA.VAL.DETECTION_MASK_DIR' "
                "path, this process will be done just once!".format(cfg.DATA.VAL.DETECTION_MASK_DIR))
            create_mask = True
        else:
            if len(next(os.walk(cfg.DATA.VAL.DETECTION_MASK_DIR))[2]) != len(next(os.walk(cfg.DATA.VAL.MASK_PATH))[2]):
                print("Different number of files found in {} and {}. Trying to create the the rest again"
                       .format(cfg.DATA.VAL.MASK_PATH,cfg.DATA.VAL.DETECTION_MASK_DIR))
                create_mask = True 
                
        if create_mask:
            create_detection_masks(cfg, data_type='val')

    # Create selected channels for test data once
    if cfg.TEST.ENABLE and cfg.DATA.TEST.LOAD_GT and cfg.TEST.EVALUATE:
        create_mask = False
        if not os.path.isdir(cfg.DATA.TEST.DETECTION_MASK_DIR):
            print("You select to create detection masks from given .csv files but no file is detected in {}. "
                "So let's prepare the data. Notice that, if you do not modify 'DATA.TEST.DETECTION_MASK_DIR' "
                "path, this process will be done just once!".format(cfg.DATA.TEST.DETECTION_MASK_DIR))
            create_mask = True
        else:
            if len(next(os.walk(cfg.DATA.TEST.DETECTION_MASK_DIR))[2]) != len(next(os.walk(cfg.DATA.TEST.MASK_PATH))[2]):
                print("Different number of files found in {} and {}. Trying to create the the rest again"
                       .format(cfg.DATA.TEST.MASK_PATH,cfg.DATA.TEST.DETECTION_MASK_DIR))
                create_mask = True 
        if create_mask:
            create_detection_masks(cfg, data_type='test')

    opts = []
    if cfg.TRAIN.ENABLE:
        print("DATA.TRAIN.MASK_PATH changed from {} to {}".format(cfg.DATA.TRAIN.MASK_PATH, cfg.DATA.TRAIN.DETECTION_MASK_DIR))
        opts.extend(['DATA.TRAIN.MASK_PATH', cfg.DATA.TRAIN.DETECTION_MASK_DIR])
        if not cfg.DATA.VAL.FROM_TRAIN:
            print("DATA.VAL.MASK_PATH changed from {} to {}".format(cfg.DATA.VAL.MASK_PATH, cfg.DATA.VAL.DETECTION_MASK_DIR))
            opts.extend(['DATA.VAL.MASK_PATH', cfg.DATA.VAL.DETECTION_MASK_DIR])
    if cfg.TEST.ENABLE and cfg.DATA.TEST.LOAD_GT:
        print("DATA.TEST.MASK_PATH changed from {} to {}".format(cfg.DATA.TEST.MASK_PATH, cfg.DATA.TEST.DETECTION_MASK_DIR))
        opts.extend(['DATA.TEST.MASK_PATH', cfg.DATA.TEST.DETECTION_MASK_DIR])
    cfg.merge_from_list(opts)
