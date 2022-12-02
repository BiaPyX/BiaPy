import os
import csv
import numpy as np
import pandas as pd
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops_table
from data.post_processing.post_processing import remove_close_points, detection_watershed, check_instances_by_prop
from data.pre_processing import create_detection_masks
from skimage.morphology import disk, dilation                                                    

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
        ndim = 2 if self.cfg.PROBLEM.NDIM == "2D" else 3

        if self.cfg.TEST.DET_LOCAL_MAX_COORDS:
            print("Capturing the local maxima ")
            all_points = []
            all_classes = []
            for channel in range(pred.shape[-1]):
                print("Class {}".format(channel+1))
                if len(self.cfg.TEST.DET_MIN_TH_TO_BE_PEAK) == 1:
                    min_th_peak = self.cfg.TEST.DET_MIN_TH_TO_BE_PEAK[0]
                else:
                    min_th_peak = self.cfg.TEST.DET_MIN_TH_TO_BE_PEAK[channel]
                pred_coordinates = peak_local_max(pred[...,channel], threshold_abs=min_th_peak, exclude_border=False)

                # Remove close points per class as post-processing method
                if self.cfg.TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS:
                    if len(self.cfg.TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS_RADIUS) == 1:
                        radius = self.cfg.TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS_RADIUS[0]
                    else:
                        radius = self.cfg.TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS_RADIUS[channel]
    
                    pred_coordinates = remove_close_points(pred_coordinates, radius, self.cfg.DATA.TEST.RESOLUTION,
                        ndim=ndim)
                        
                all_points.append(pred_coordinates)   
                all_classes.append(np.full(len(pred_coordinates), channel))

            # Remove close points again seeing all classes together
            if self.cfg.TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS:
                print("All classes together")
                radius = np.min(self.cfg.TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS_RADIUS)
 
                all_points = np.concatenate(all_points, axis=0)
                all_classes = np.concatenate(all_classes, axis=0)

                new_points, all_classes = remove_close_points(all_points, radius, self.cfg.DATA.TEST.RESOLUTION,
                    classes=all_classes, ndim=ndim)
                
                # Create again list of arrays of all points
                all_points = []
                for i in range(self.cfg.MODEL.N_CLASSES):
                    all_points.append([])
                for i, c in enumerate(all_classes):
                    all_points[c].append(new_points[i])
                del new_points

            # Create a file that represent the local maxima
            print("Creating the image with detected points . . .")
            all_labels = []
            points_pred = np.zeros(pred.shape[:-1], dtype=np.uint8)
            for n, pred_coordinates in enumerate(all_points):
                for coord in pred_coordinates:
                        z,y,x = coord
                        points_pred[z,y,x] = n+1
                self.cell_count_lines.append([filenames, len(pred_coordinates)])

            if len(pred_coordinates) > 0:
                for i in range(points_pred.shape[0]):                                                                                  
                    points_pred[i] = dilation(points_pred[i], disk(3)) 

            save_tif(np.expand_dims(np.expand_dims(points_pred,0),-1), self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK,
                     filenames, verbose=self.cfg.TEST.VERBOSE)

            # Detection watershed
            if self.cfg.TEST.POST_PROCESSING.DET_WATERSHED:
                data_filename = os.path.join(self.cfg.DATA.TEST.PATH, filenames[0])
                w_dir = os.path.join(self.cfg.PATHS.WATERSHED_DIR, filenames[0])
                check_wa = w_dir if self.cfg.PROBLEM.DETECTION.DATA_CHECK_MW else None

                points_pred = detection_watershed(points_pred, all_points, data_filename, self.cfg.TEST.POST_PROCESSING.DET_WATERSHED_FIRST_DILATION,
                    self.cfg.MODEL.N_CLASSES, ndim=ndim, donuts_classes=self.cfg.TEST.POST_PROCESSING.DET_WATERSHED_DONUTS_CLASSES,
                    donuts_patch=self.cfg.TEST.POST_PROCESSING.DET_WATERSHED_DONUTS_PATCH, 
                    donuts_nucleus_diameter=self.cfg.TEST.POST_PROCESSING.DET_WATERSHED_DONUTS_NUCLEUS_DIAMETER, save_dir=check_wa)
                
                # Advice user if instance     
                labels, npixels, areas, circularities, comment = check_instances_by_prop(points_pred, self.cfg.DATA.TEST.RESOLUTION, 
                    np.concatenate(all_points, axis=0), circularity_th=self.cfg.TEST.POST_PROCESSING.DET_WATERSHED_CIRCULARITY)

                save_tif(np.expand_dims(np.expand_dims(points_pred,0),-1), self.cfg.PATHS.RESULT_DIR.PER_IMAGE_POST_PROCESSING,
                    filenames, verbose=self.cfg.TEST.VERBOSE)
            del points_pred

            # Save coords in a couple of csv files            
            aux = np.concatenate(all_points, axis=0)
            prob = pred[aux[:,0], aux[:,1], aux[:,2], all_classes]

            if self.cfg.TEST.POST_PROCESSING.DET_WATERSHED:
                df = pd.DataFrame(zip(labels, list(aux[:,0]), list(aux[:,1]), list(aux[:,2]), list(prob), list(all_classes),\
                    npixels, areas, circularities, comment), columns =['label', 'axis-0', 'axis-1', 'axis-2', 'probability', \
                    'class', 'npixels','area', 'circularity', 'comment'])
                df = df.sort_values(by=['label'])   
            else:
                df = pd.DataFrame(zip(list(aux[:,0]), list(aux[:,1]), list(aux[:,2]), list(prob), list(all_classes)), 
                    columns =['axis-0', 'axis-1', 'axis-2', 'probability', 'class'])
            del aux 

            df.to_csv(os.path.join(self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK, os.path.splitext(filenames[0])[0]+'_full_info.csv'))
            if self.cfg.TEST.POST_PROCESSING.DET_WATERSHED:
                df = df.drop(columns=['class', 'label', 'npixels', 'area', 'circularity', 'comment'])
            else:
                df = df.drop(columns=['class'])
            df.to_csv(os.path.join(self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK, os.path.splitext(filenames[0])[0]+'_prob.csv'))

            # Calculate detection metrics
            if self.cfg.DATA.TEST.LOAD_GT:
                all_channel_d_metrics = [0,0,0]
                for ch, pred_coordinates in enumerate(all_points):
                    exclusion_mask = Y[...,ch] > 0
                    bin_Y = Y[...,ch] * exclusion_mask.astype( float )
                    props = regionprops_table(label( bin_Y ), properties=('area','centroid'))
                    gt_coordinates = []
                    for n in range(len(props['centroid-0'])):
                        gt_coordinates.append([props['centroid-0'][n], props['centroid-1'][n], props['centroid-2'][n]])
                    gt_coordinates = np.array(gt_coordinates)

                    if self.cfg.PROBLEM.NDIM == '3D':
                        v_size = (self.cfg.DATA.TEST.RESOLUTION[0], self.cfg.DATA.TEST.RESOLUTION[1], self.cfg.DATA.TEST.RESOLUTION[2])
                    else:
                        v_size = (1,self.cfg.DATA.TEST.RESOLUTION[0], self.cfg.DATA.TEST.RESOLUTION[1])

                    if len(pred_coordinates) > 0:
                        print("Detection (class "+str(ch+1)+")")
                        d_metrics = detection_metrics(gt_coordinates, pred_coordinates, tolerance=self.cfg.TEST.DET_TOLERANCE[ch],
                                                      voxel_size=v_size, verbose=self.cfg.TEST.VERBOSE)
                        print("Detection metrics: {}".format(d_metrics))
                        all_channel_d_metrics[0] += d_metrics[1]
                        all_channel_d_metrics[1] += d_metrics[3]
                        all_channel_d_metrics[2] += d_metrics[5]
                    else:
                        print("No point found to calculate the metrics!")

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
    print("############################\n"
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
    if cfg.TEST.ENABLE and cfg.DATA.TEST.LOAD_GT:
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
