import os
import csv
import torch 
import torch.nn.functional as F
import numpy as np
import pandas as pd
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops_table
from skimage.morphology import disk, dilation        

from data.data_2D_manipulation import load_and_prepare_2D_train_data
from data.data_3D_manipulation import load_and_prepare_3D_data

from data.post_processing.post_processing import (remove_close_points, detection_watershed, 
    remove_instance_by_circularity_central_slice)
from data.pre_processing import create_detection_masks
from utils.util import save_tif
from engine.metrics import detection_metrics, jaccard_index, weighted_bce_dice_loss
from engine.base_workflow import Base_Workflow

class Detection_Workflow(Base_Workflow):
    def __init__(self, cfg, job_identifier, device, rank, **kwargs):
        super(Detection_Workflow, self).__init__(cfg, job_identifier, device, rank, **kwargs)

        # Detection stats
        self.stats['d_precision'] = 0
        self.stats['d_recall'] = 0
        self.stats['d_f1'] = 0

        self.stats['d_precision_per_crop'] = 0
        self.stats['d_recall_per_crop'] = 0
        self.stats['d_f1_per_crop'] = 0

        print("####################\n"
              "#  PRE-PROCESSING  #\n"
              "####################\n")

        self.original_test_mask_path = self.prepare_detection_data()

        if self.cfg.DATA.TEST.LOAD_GT or self.cfg.DATA.TEST.USE_VAL_AS_TEST:
            self.csv_files = sorted(next(os.walk(self.original_test_mask_path))[2])
        self.cell_count_file = os.path.join(self.cfg.PATHS.RESULT_DIR.PATH, 'cell_counter.csv')
        self.cell_count_lines = []

        # From now on, no modification of the cfg will be allowed
        self.cfg.freeze()

        # Activations for each output channel:
        # channel number : 'activation'
        self.activations = {'0': 'CE_Sigmoid'}

        # Workflow specific training variables
        self.mask_path = cfg.DATA.TRAIN.GT_PATH
        self.load_Y_val = True

        # Workflow specific test variables
        if self.cfg.TEST.POST_PROCESSING.DET_WATERSHED or self.cfg.TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS:
            self.post_processing['detection_post'] = True
        else:
            self.post_processing['detection_post'] = False    

    def define_metrics(self):
        if self.cfg.LOSS.TYPE == "CE": 
            self.metrics = [jaccard_index]
            self.metric_names = ["jaccard_index"]
            self.loss = torch.nn.BCEWithLogitsLoss()
        elif self.cfg.LOSS.TYPE == "W_CE_DICE":
            self.metrics = [jaccard_index]
            self.metric_names = ["jaccard_index"]
            self.loss = weighted_bce_dice_loss(w_dice=0.66, w_bce=0.33)

    def metric_calculation(self, output, targets, device, metric_logger=None):
        with torch.no_grad():
            train_iou = self.metrics[0](output, targets, device, num_classes=self.cfg.MODEL.N_CLASSES)
            train_iou = train_iou.item() if not torch.isnan(train_iou) else 0
            if metric_logger is not None:
                metric_logger.meters[self.metric_names[0]].update(train_iou)
            else:
                return train_iou

    def detection_process(self, pred, filenames, metric_names=[], f_numbers=0):
        ndim = 2 if self.cfg.PROBLEM.NDIM == "2D" else 3
        pred_shape = pred.shape
        
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
                pred_coordinates = peak_local_max(pred[...,channel].astype(np.float32), threshold_abs=min_th_peak, exclude_border=False)

                # Remove close points per class as post-processing method
                if self.cfg.TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS:
                    if len(self.cfg.TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS_RADIUS) == 1:
                        radius = self.cfg.TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS_RADIUS[0]
                    else:
                        radius = self.cfg.TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS_RADIUS[channel]
    
                    pred_coordinates = remove_close_points(pred_coordinates, radius, self.cfg.DATA.TEST.RESOLUTION,
                        ndim=ndim)
                     
                all_points.append(pred_coordinates)   
                c_size = 1 if len(pred_coordinates) == 0 else len(pred_coordinates)
                all_classes.append(np.full(c_size, channel))

            # Remove close points again seeing all classes together, as it can be that a point is detected in both classes
            # if there is not clear distinction between them
            classes = 1 if self.cfg.MODEL.N_CLASSES <= 2 else self.cfg.MODEL.N_CLASSES
            if self.cfg.TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS and classes > 1:
                print("All classes together")
                radius = np.min(self.cfg.TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS_RADIUS)
 
                all_points = np.concatenate(all_points, axis=0)
                all_classes = np.concatenate(all_classes, axis=0)

                new_points, all_classes = remove_close_points(all_points, radius, self.cfg.DATA.TEST.RESOLUTION,
                    classes=all_classes, ndim=ndim)
                
                # Create again list of arrays of all points
                all_points = []
                for i in range(classes):
                    all_points.append([])
                for i, c in enumerate(all_classes):
                    all_points[c].append(new_points[i])
                del new_points
            # Create a file with detected point and other image with predictions ids (if GT given)
            print("Creating the images with detected points . . .")   
            points_pred = np.zeros(pred.shape[:-1], dtype=np.uint8)
            for n, pred_coordinates in enumerate(all_points):
                if self.cfg.DATA.TEST.LOAD_GT or self.cfg.DATA.TEST.USE_VAL_AS_TEST:
                    pred_id_img = np.zeros(pred_shape[:-1], dtype=np.uint32)
                for j, coord in enumerate(pred_coordinates):
                    z,y,x = coord
                    points_pred[z,y,x] = n+1
                    if self.cfg.DATA.TEST.LOAD_GT or self.cfg.DATA.TEST.USE_VAL_AS_TEST:
                        pred_id_img[z,y,x] = j+1
                
                # Dilate and save the prediction ids for the current class 
                if self.cfg.DATA.TEST.LOAD_GT or self.cfg.DATA.TEST.USE_VAL_AS_TEST:
                    for i in range(pred_id_img.shape[0]):                                                                                  
                        pred_id_img[i] = dilation(pred_id_img[i], disk(3))
                    save_tif(np.expand_dims(np.expand_dims(pred_id_img,0),-1), self.cfg.PATHS.RESULT_DIR.DET_ASSOC_POINTS,
                        [os.path.splitext(filenames[0])[0]+'_class'+str(n+1)+'_pred_ids.tif'], verbose=self.cfg.TEST.VERBOSE)

                self.cell_count_lines.append([filenames, len(pred_coordinates)])

            if self.cfg.DATA.TEST.LOAD_GT or self.cfg.DATA.TEST.USE_VAL_AS_TEST: del pred_id_img

            # Dilate and save the detected point image
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
                    clases, ndim=ndim, donuts_classes=self.cfg.TEST.POST_PROCESSING.DET_WATERSHED_DONUTS_CLASSES,
                    donuts_patch=self.cfg.TEST.POST_PROCESSING.DET_WATERSHED_DONUTS_PATCH, 
                    donuts_nucleus_diameter=self.cfg.TEST.POST_PROCESSING.DET_WATERSHED_DONUTS_NUCLEUS_DIAMETER, save_dir=check_wa)
                
                # Advice user if instance     
                points_pred, labels, npixels, areas, circularities, diameters, comment = remove_instance_by_circularity_central_slice(points_pred, self.cfg.DATA.TEST.RESOLUTION, 
                    np.concatenate(all_points, axis=0), circularity_th=self.cfg.TEST.POST_PROCESSING.WATERSHED_CIRCULARITY)

                save_tif(np.expand_dims(np.expand_dims(points_pred,0),-1), self.cfg.PATHS.RESULT_DIR.PER_IMAGE_POST_PROCESSING,
                    filenames, verbose=self.cfg.TEST.VERBOSE)
            del points_pred

            # Save coords in a couple of csv files            
            aux = np.concatenate(all_points, axis=0)
            if len(aux) != 0:
                if self.cfg.PROBLEM.NDIM == "3D":
                    prob = pred[aux[:,0], aux[:,1], aux[:,2], all_classes]
                    prob = np.concatenate(prob, axis=0)
                    all_classes = np.concatenate(all_classes, axis=0)
                    if self.cfg.TEST.POST_PROCESSING.DET_WATERSHED:
                        size_measure = 'area' if ndim == 2 else 'volume'
                        df = pd.DataFrame(zip(labels, list(aux[:,0]), list(aux[:,1]), list(aux[:,2]), list(prob), list(all_classes),\
                            npixels, areas, circularities, diameters, comment), columns =['pred_id', 'axis-0', 'axis-1', 'axis-2', 'probability', \
                            'class', 'npixels', size_measure, 'circularity', 'diameters', 'comment'])
                        df = df.sort_values(by=['pred_id'])   
                    else:
                        labels = []
                        for i, pred_coordinates in enumerate(all_points):
                            for j in range(len(pred_coordinates)):
                                labels.append(j+1)

                        df = pd.DataFrame(zip(labels, list(aux[:,0]), list(aux[:,1]), list(aux[:,2]), list(prob), list(all_classes)), 
                            columns =['pred_id', 'axis-0', 'axis-1', 'axis-2', 'probability', 'class'])
                        df = df.sort_values(by=['pred_id'])
                else:
                    aux = aux[:,1:]
                    prob = pred[0,aux[:,0], aux[:,1], all_classes]
                    prob = np.concatenate(prob, axis=0)
                    all_classes = np.concatenate(all_classes, axis=0)
                    if self.cfg.TEST.POST_PROCESSING.DET_WATERSHED:
                        size_measure = 'area' if ndim == 2 else 'volume'
                        df = pd.DataFrame(zip(labels, list(aux[:,0]), list(aux[:,1]), list(prob), list(all_classes),\
                            npixels, areas, circularities, diameters, comment), columns =['pred_id', 'axis-0', 'axis-1', 'probability', \
                            'class', 'npixels', size_measure, 'circularity', 'diameters', 'comment'])
                        df = df.sort_values(by=['pred_id'])   
                    else:
                        df = pd.DataFrame(zip(list(aux[:,0]), list(aux[:,1]), list(prob), list(all_classes)), 
                            columns =['axis-0', 'axis-1', 'probability', 'class'])
                        df = df.sort_values(by=['axis-0'])
                del aux 

                df.to_csv(os.path.join(self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK, os.path.splitext(filenames[0])[0]+'_full_info.csv'))
                if self.cfg.TEST.POST_PROCESSING.DET_WATERSHED:
                    df = df.drop(columns=['class', 'pred_id', 'npixels', size_measure, 'circularity', 'comment'])
                else:
                    df = df.drop(columns=['class'])
                df.to_csv(os.path.join(self.cfg.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK, os.path.splitext(filenames[0])[0]+'_prob.csv'))
                del df

            # Calculate detection metrics
            if self.cfg.DATA.TEST.LOAD_GT or self.cfg.DATA.TEST.USE_VAL_AS_TEST:
                all_channel_d_metrics = [0,0,0]
                dfs = []
                gt_all_coords = []
                for ch, pred_coordinates in enumerate(all_points):

                    # Read the GT coordinates from the CSV file
                    csv_filename = os.path.join(self.original_test_mask_path, os.path.splitext(filenames[0])[0]+'.csv')
                    if not os.path.exists(csv_filename):
                        print("WARNING: The CSV file seems to have different name than iamge. Using the CSV file "
                              "with the same position as the CSV in the directory. Check if it is correct!")
                        csv_filename = os.path.join(self.original_test_mask_path, self.csv_files[f_numbers[0]])
                        print("Its respective CSV file seems to be: {}".format(csv_filename))
                    print("Reading GT data from: {}".format(csv_filename))
                    df = pd.read_csv(csv_filename, index_col=0)     
                    zcoords = df['axis-0'].tolist()
                    ycoords = df['axis-1'].tolist()
                    if self.cfg.PROBLEM.NDIM == '3D': 
                        xcoords = df['axis-2'].tolist()
                        gt_coordinates = [[z,y,x] for z,y,x in zip(zcoords,ycoords,xcoords)]
                    else:
                        gt_coordinates = [[0,y,x] for y,x in zip(zcoords,ycoords)]
                    gt_all_coords.append(gt_coordinates)

                    if self.cfg.PROBLEM.NDIM == '3D':
                        v_size = (self.cfg.DATA.TEST.RESOLUTION[0], self.cfg.DATA.TEST.RESOLUTION[1], self.cfg.DATA.TEST.RESOLUTION[2])
                    else:
                        v_size = (1,self.cfg.DATA.TEST.RESOLUTION[0], self.cfg.DATA.TEST.RESOLUTION[1])

                    # Calculate detection metrics 
                    if len(pred_coordinates) > 0:
                        print("Detection (class "+str(ch+1)+")")
                        d_metrics, gt_assoc, fp = detection_metrics(gt_coordinates, pred_coordinates, tolerance=self.cfg.TEST.DET_TOLERANCE[ch],
                            voxel_size=v_size, return_assoc=True, verbose=self.cfg.TEST.VERBOSE)
                        print("Detection metrics: {}".format(d_metrics))
                        all_channel_d_metrics[0] += d_metrics[1]
                        all_channel_d_metrics[1] += d_metrics[3]
                        all_channel_d_metrics[2] += d_metrics[5]

                        # Save csv files with the associations between GT points and predicted ones 
                        dfs.append([gt_assoc.copy(),fp.copy()])
                        if self.cfg.PROBLEM.NDIM == "2D":
                            gt_assoc = gt_assoc.drop(columns=['axis-0'])
                            fp = fp.drop(columns=['axis-0'])
                            gt_assoc = gt_assoc.rename(columns={'axis-1': 'axis-0', 'axis-2': 'axis-1'})
                            fp = fp.rename(columns={'axis-1': 'axis-0', 'axis-2': 'axis-1'})
                        gt_assoc.to_csv(os.path.join(self.cfg.PATHS.RESULT_DIR.DET_ASSOC_POINTS, os.path.splitext(filenames[0])[0]+'_class'+str(ch+1)+'_gt_assoc.csv'))
                        fp.to_csv(os.path.join(self.cfg.PATHS.RESULT_DIR.DET_ASSOC_POINTS, os.path.splitext(filenames[0])[0]+'_class'+str(ch+1)+'_fp.csv'))             
                    else:
                        print("No point found to calculate the metrics!")

                print("All classes "+str(ch+1))
                all_channel_d_metrics[0] = all_channel_d_metrics[0]/len(all_points)
                all_channel_d_metrics[1] = all_channel_d_metrics[1]/len(all_points)
                all_channel_d_metrics[2] = all_channel_d_metrics[2]/len(all_points)
                print("Detection metrics: {}".format(["Precision", all_channel_d_metrics[0],
                    "Recall", all_channel_d_metrics[1], "F1", all_channel_d_metrics[2]]))

                self.stats[metric_names[0]] += all_channel_d_metrics[0]
                self.stats[metric_names[1]] += all_channel_d_metrics[1]
                self.stats[metric_names[2]] += all_channel_d_metrics[2]
                
                print("Creating the image with a summary of detected points and false positives with colors . . .")
                points_pred = np.zeros(pred_shape[:-1]+(3,), dtype=np.uint8)
                for ch, gt_coords in enumerate(gt_all_coords):
                    if len(dfs) > 0:
                        gt_assoc, fp = dfs[ch]

                    # TP and FN
                    gt_id_img = np.zeros(pred_shape[:-1], dtype=np.uint32)
                    for j, cor in enumerate(gt_coords):
                        z,y,x = cor
                        z,y,x = int(z),int(y),int(x)
                        if len(dfs) > 0:
                            if gt_assoc[gt_assoc['gt_id'] == j+1]["tag"].iloc[0] == "TP":
                                points_pred[z,y,x] = (0,255,0)# Green
                            else:   
                                points_pred[z,y,x] = (255,0,0)# Red
                        else:                           
                            points_pred[z,y,x] = (255,0,0)# Red

                        gt_id_img[z,y,x] = j+1

                    # Dilate and save the GT ids for the current class 
                    for i in range(gt_id_img.shape[0]):      
                        gt_id_img[i] = dilation(gt_id_img[i], disk(3))
                    save_tif(np.expand_dims(np.expand_dims(gt_id_img,0),-1), self.cfg.PATHS.RESULT_DIR.DET_ASSOC_POINTS,
                        [os.path.splitext(filenames[0])[0]+'_class'+str(ch+1)+'_gt_ids.csv'], verbose=self.cfg.TEST.VERBOSE)

                    # FP
                    if len(dfs) > 0:
                        for cor in zip(fp['axis-0'].tolist(),fp['axis-1'].tolist(),fp['axis-2'].tolist()):
                            z, y, x =  cor  
                            z,y,x = int(z),int(y),int(x)
                            points_pred[z,y,x] = (0,0,255) # Blue

                # Dilate and save the predicted points for the current class 
                for i in range(points_pred.shape[0]):      
                    for j in range(points_pred.shape[-1]):                                                                              
                        points_pred[i,...,j] = dilation(points_pred[i,...,j], disk(3)) 
                save_tif(np.expand_dims(points_pred,0), self.cfg.PATHS.RESULT_DIR.DET_ASSOC_POINTS,
                        filenames, verbose=self.cfg.TEST.VERBOSE)          

    def normalize_stats(self, image_counter):
        super().normalize_stats(image_counter)

        with open(self.cell_count_file, 'w', newline="") as file:
            csvwriter = csv.writer(file)
            csvwriter.writerow(['filename', 'cells'])
            for nr in range(len(self.cell_count_lines)):
                csvwriter.writerow([nr+1] + self.cell_count_lines[nr])
        if self.cfg.DATA.TEST.LOAD_GT or self.cfg.DATA.TEST.USE_VAL_AS_TEST:
            if self.cfg.TEST.STATS.PER_PATCH:
                self.stats['d_precision_per_crop'] = self.stats['d_precision_per_crop'] / image_counter
                self.stats['d_recall_per_crop'] = self.stats['d_recall_per_crop'] / image_counter
                self.stats['d_f1_per_crop'] = self.stats['d_f1_per_crop'] / image_counter
            if self.cfg.TEST.STATS.FULL_IMG:
                self.stats['d_precision'] = self.stats['d_precision'] / image_counter
                self.stats['d_recall'] = self.stats['d_recall'] / image_counter
                self.stats['d_f1'] = self.stats['d_f1'] / image_counter

    def after_merge_patches(self, pred, filenames):
        self.detection_process(pred, filenames, ['d_precision_per_crop', 'd_recall_per_crop', 'd_f1_per_crop'])

    def after_full_image(self, pred, filenames):
        self.detection_process(pred, filenames, ['d_precision', 'd_recall', 'd_f1'])

    def after_all_images(self):
        super().after_all_images()

    def print_stats(self, image_counter):
        super().print_stats(image_counter)
        super().print_post_processing_stats()

        print("Detection specific metrics:")
        if self.cfg.DATA.TEST.LOAD_GT or self.cfg.DATA.TEST.USE_VAL_AS_TEST:
            if self.cfg.TEST.STATS.PER_PATCH:
                print("Detection - Test Precision (merge patches): {}".format(self.stats['d_precision_per_crop']))
                print("Detection - Test Recall (merge patches): {}".format(self.stats['d_recall_per_crop']))
                print("Detection - Test F1 (merge patches): {}".format(self.stats['d_f1_per_crop']))
            if self.cfg.TEST.STATS.FULL_IMG:
                print("Detection - Test Precision (per image): {}".format(self.stats['d_precision']))
                print("Detection - Test Recall (per image): {}".format(self.stats['d_recall']))
                print("Detection - Test F1 (per image): {}".format(self.stats['d_f1']))

    def prepare_detection_data(self):
        print("############################")
        print("#  PREPARE DETECTION DATA  #")
        print("############################")
        original_test_mask_path = None

        # Create selected channels for train data
        if self.cfg.TRAIN.ENABLE or self.cfg.DATA.TEST.USE_VAL_AS_TEST:
            create_mask = False
            if not os.path.isdir(self.cfg.DATA.TRAIN.DETECTION_MASK_DIR):
                print("You select to create detection masks from given .csv files but no file is detected in {}. "
                    "So let's prepare the data. Notice that, if you do not modify 'DATA.TRAIN.DETECTION_MASK_DIR' "
                    "path, this process will be done just once!".format(self.cfg.DATA.TRAIN.DETECTION_MASK_DIR))
                create_mask = True
            else:
                if len(next(os.walk(self.cfg.DATA.TRAIN.DETECTION_MASK_DIR))[2]) != len(next(os.walk(self.cfg.DATA.TRAIN.GT_PATH))[2]):
                    print("Different number of files found in {} and {}. Trying to create the the rest again"
                        .format(self.cfg.DATA.TRAIN.GT_PATH,self.cfg.DATA.TRAIN.DETECTION_MASK_DIR))
                    create_mask = True    

            if create_mask:
                create_detection_masks(self.cfg)

        # Create selected channels for val data
        if self.cfg.TRAIN.ENABLE and not self.cfg.DATA.VAL.FROM_TRAIN:
            create_mask = False
            if not os.path.isdir(self.cfg.DATA.VAL.DETECTION_MASK_DIR):
                print("You select to create detection masks from given .csv files but no file is detected in {}. "
                    "So let's prepare the data. Notice that, if you do not modify 'DATA.VAL.DETECTION_MASK_DIR' "
                    "path, this process will be done just once!".format(self.cfg.DATA.VAL.DETECTION_MASK_DIR))
                create_mask = True
            else:
                if len(next(os.walk(self.cfg.DATA.VAL.DETECTION_MASK_DIR))[2]) != len(next(os.walk(self.cfg.DATA.VAL.GT_PATH))[2]):
                    print("Different number of files found in {} and {}. Trying to create the the rest again"
                        .format(self.cfg.DATA.VAL.GT_PATH,self.cfg.DATA.VAL.DETECTION_MASK_DIR))
                    create_mask = True 
                    
            if create_mask:
                create_detection_masks(self.cfg, data_type='val')

        # Create selected channels for test data once
        if self.cfg.TEST.ENABLE and self.cfg.DATA.TEST.LOAD_GT and not self.cfg.DATA.TEST.USE_VAL_AS_TEST:
            create_mask = False
            if not os.path.isdir(self.cfg.DATA.TEST.DETECTION_MASK_DIR):
                print("You select to create detection masks from given .csv files but no file is detected in {}. "
                    "So let's prepare the data. Notice that, if you do not modify 'DATA.TEST.DETECTION_MASK_DIR' "
                    "path, this process will be done just once!".format(self.cfg.DATA.TEST.DETECTION_MASK_DIR))
                create_mask = True
            else:
                if len(next(os.walk(self.cfg.DATA.TEST.DETECTION_MASK_DIR))[2]) != len(next(os.walk(self.cfg.DATA.TEST.GT_PATH))[2]):
                    print("Different number of files found in {} and {}. Trying to create the the rest again"
                        .format(self.cfg.DATA.TEST.GT_PATH,self.cfg.DATA.TEST.DETECTION_MASK_DIR))
                    create_mask = True 
            if create_mask:
                create_detection_masks(self.cfg, data_type='test')

        opts = []
        if self.cfg.TRAIN.ENABLE:
            print("DATA.TRAIN.GT_PATH changed from {} to {}".format(self.cfg.DATA.TRAIN.GT_PATH, self.cfg.DATA.TRAIN.DETECTION_MASK_DIR))
            opts.extend(['DATA.TRAIN.GT_PATH', self.cfg.DATA.TRAIN.DETECTION_MASK_DIR])
            if not self.cfg.DATA.VAL.FROM_TRAIN:
                print("DATA.VAL.GT_PATH changed from {} to {}".format(self.cfg.DATA.VAL.GT_PATH, self.cfg.DATA.VAL.DETECTION_MASK_DIR))
                opts.extend(['DATA.VAL.GT_PATH', self.cfg.DATA.VAL.DETECTION_MASK_DIR])
        if self.cfg.TEST.ENABLE and self.cfg.DATA.TEST.LOAD_GT:
            print("DATA.TEST.GT_PATH changed from {} to {}".format(self.cfg.DATA.TEST.GT_PATH, self.cfg.DATA.TEST.DETECTION_MASK_DIR))
            opts.extend(['DATA.TEST.GT_PATH', self.cfg.DATA.TEST.DETECTION_MASK_DIR])
            original_test_mask_path = self.cfg.DATA.TEST.GT_PATH
        self.cfg.merge_from_list(opts)
        

        return original_test_mask_path