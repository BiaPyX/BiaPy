import os
import torch
import numpy as np
import pandas as pd
from skimage.io import imread
from tqdm import tqdm
from skimage.segmentation import clear_border
from skimage.transform import resize

from biapy.data.post_processing.post_processing import (watershed_by_channels, voronoi_on_mask, 
    measure_morphological_props_and_filter, repare_large_blobs, apply_binary_mask)
from biapy.data.pre_processing import create_instance_channels, create_test_instance_channels, norm_range01
from biapy.utils.util import save_tif, pad_and_reflect
from biapy.utils.matching import matching, wrapper_matching_dataset_lazy
from biapy.engine.metrics import jaccard_index, instance_segmentation_loss
from biapy.engine.base_workflow import Base_Workflow

class Instance_Segmentation_Workflow(Base_Workflow):
    """
    Instance segmentation workflow where the goal is to assign an unique id, i.e. integer, to each object of the input image.
    More details in `our documentation <https://biapy.readthedocs.io/en/latest/workflows/instance_segmentation.html>`_.  

    Parameters
    ----------
    cfg : YACS configuration
        Running configuration.
    
    Job_identifier : str
        Complete name of the running job.

    device : Torch device
        Device used. 

    args : argpase class
        Arguments used in BiaPy's call. 
    """
    def __init__(self, cfg, job_identifier, device, args, **kwargs):
        super(Instance_Segmentation_Workflow, self).__init__(cfg, job_identifier, device, args, **kwargs)

        self.original_test_path, self.original_test_mask_path = self.prepare_instance_data()

        self.all_matching_stats = []

        self.instance_ths = {}
        self.instance_ths['TYPE'] = self.cfg.PROBLEM.INSTANCE_SEG.DATA_MW_TH_TYPE
        self.instance_ths['TH_BINARY_MASK'] = self.cfg.PROBLEM.INSTANCE_SEG.DATA_MW_TH_BINARY_MASK
        self.instance_ths['TH_CONTOUR'] = self.cfg.PROBLEM.INSTANCE_SEG.DATA_MW_TH_CONTOUR
        self.instance_ths['TH_FOREGROUND'] = self.cfg.PROBLEM.INSTANCE_SEG.DATA_MW_TH_FOREGROUND
        self.instance_ths['TH_DISTANCE'] = self.cfg.PROBLEM.INSTANCE_SEG.DATA_MW_TH_DISTANCE
        self.instance_ths['TH_POINTS'] = self.cfg.PROBLEM.INSTANCE_SEG.DATA_MW_TH_POINTS 

        # From now on, no modification of the cfg will be allowed
        self.cfg.freeze()

        # Activations for each output channel:
        # channel number : 'activation'
        if self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS == "Dv2":
            self.activations = {'0': 'Linear'}
        elif self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS in ["BC", "BP", "BCM"]:
            self.activations = {':': 'CE_Sigmoid'}
        elif self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS in ["BDv2", "BD"]:
            self.activations = {'0': 'CE_Sigmoid', '1': 'Linear'}
        elif self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS in ["BCD", "BCDv2"]:
            self.activations = {'0': 'CE_Sigmoid', '1': 'CE_Sigmoid', '2': 'Linear'}

        # Workflow specific training variables
        self.mask_path = cfg.DATA.TRAIN.GT_PATH
        self.load_Y_val = True

        # Specific instance segmentation post-processing
        if self.cfg.TEST.POST_PROCESSING.VORONOI_ON_MASK or \
            self.cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.ENABLE or \
            self.cfg.TEST.POST_PROCESSING.REPARE_LARGE_BLOBS_SIZE != -1:
            self.post_processing['instance_post'] = True
            self.all_matching_stats_post_processing = []   
        else:
            self.post_processing['instance_post'] = False            
        self.instances_already_created = False 

    def define_metrics(self):
        """
        Definition of self.metrics, self.metric_names and self.loss variables.
        """
        self.metrics = []
        self.metric_names = []
        self.loss = instance_segmentation_loss(self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNEL_WEIGHTS,
            self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS, 
            self.cfg.PROBLEM.INSTANCE_SEG.DISTANCE_CHANNEL_MASK)
        if self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS in ["BC", "BCM", "BP"]:
            self.first_not_binary_channel = len(self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS)
            self.metrics.append(
                jaccard_index(device=self.device, num_classes=1, 
                    first_not_binary_channel=self.first_not_binary_channel, 
                    torchvision_models=True if self.cfg.MODEL.SOURCE == "torchvision" else False)
            )
            self.metric_names.append("jaccard_index")
        elif self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS == "BD":
            self.first_not_binary_channel = 1 
            self.metrics.append(
                jaccard_index(device=self.device, num_classes=1, 
                    first_not_binary_channel=self.first_not_binary_channel, 
                    torchvision_models=True if self.cfg.MODEL.SOURCE == "torchvision" else False)
            ) 
            self.metric_names.append("jaccard_index")
            self.metrics.append(torch.nn.L1Loss())
            self.metric_names.append("L1_distance_channel")
        elif self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS in ["BCD", 'BCDv2']:
            self.first_not_binary_channel = 2
            self.metrics.append(
                jaccard_index(device=self.device, num_classes=1, 
                    first_not_binary_channel=self.first_not_binary_channel, 
                    torchvision_models=True if self.cfg.MODEL.SOURCE == "torchvision" else False)
            )
            self.metric_names.append("jaccard_index")
            self.metrics.append(torch.nn.L1Loss())
            self.metric_names.append("L1_distance_channel")
        elif self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS == "BDv2":
            self.first_not_binary_channel = 1
            self.metrics.append(
                jaccard_index(device=self.device, num_classes=1, 
                    first_not_binary_channel=self.first_not_binary_channel, 
                    torchvision_models=True if self.cfg.MODEL.SOURCE == "torchvision" else False)
            )
            self.metrics.append(torch.nn.L1Loss())
            self.metric_names.append("jaccard_index")
            self.metric_names.append("L1_distance_channel")
        elif self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS == "Dv2":
            self.metrics.append(torch.nn.L1Loss())
            self.metric_names.append("L1_distance_channel")

    def metric_calculation(self, output, targets, metric_logger=None):
        """
        Execution of the metrics defined in :func:`~define_metrics` function. 

        Parameters
        ----------
        output : Torch Tensor
            Prediction of the model. 

        targets : Torch Tensor
            Ground truth to compare the prediction with. 

        metric_logger : MetricLogger, optional
            Class to be updated with the new metric(s) value(s) calculated. 
        
        Returns
        -------
        value : float
            Value of the metric for the given prediction. 
        """
        with torch.no_grad():
            train_iou = []
            train_dis = []
            for i in range(len(self.metric_names)):
                if self.metric_names[i] == "jaccard_index":
                    iou = self.metrics[i](output, targets)
                    iou = iou.item() if not torch.isnan(iou) else 0
                    train_iou.append(iou)
                elif self.metric_names[i] == "L1_distance_channel":
                    train_dis.append(self.metrics[i](output, targets).item())
            train_iou = np.mean(train_iou)
            train_dis = np.mean(train_dis)
            if metric_logger is not None:
                metric_logger.meters[self.metric_names[0]].update(train_iou)
                if len(self.metric_names) > 1:
                    metric_logger.meters[self.metric_names[1]].update(train_dis)
            else:
                return train_iou

    def instance_seg_process(self, pred, filenames):
        """
        Instance segmentation workflow engine for test/inference. Process model's prediction to prepare 
        instance segmentation output and calculate metrics. 

        Parameters
        ----------
        pred : 4D/5D Torch tensor
            Model predictions. E.g. ``(num_of_images, y, x, channels)`` for 2D or 
            ``(num_of_images, z, y, x, channels)`` for 3D.

        filenames : List of str
            Predicted image's filenames.
        """
        #############################
        ### INSTANCE SEGMENTATION ###
        #############################
        if not self.instances_already_created: 
            print("Creating instances with watershed . . .")
            w_dir = os.path.join(self.cfg.PATHS.WATERSHED_DIR, filenames[0])
            check_wa = w_dir if self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHECK_MW else None
            
            w_pred = watershed_by_channels(pred, self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS, ths=self.instance_ths, 
                remove_before=self.cfg.PROBLEM.INSTANCE_SEG.DATA_REMOVE_BEFORE_MW, thres_small_before=self.cfg.PROBLEM.INSTANCE_SEG.DATA_REMOVE_SMALL_OBJ_BEFORE,
                seed_morph_sequence=self.cfg.PROBLEM.INSTANCE_SEG.SEED_MORPH_SEQUENCE, seed_morph_radius=self.cfg.PROBLEM.INSTANCE_SEG.SEED_MORPH_RADIUS, 
                erode_and_dilate_foreground=self.cfg.PROBLEM.INSTANCE_SEG.ERODE_AND_DILATE_FOREGROUND, fore_erosion_radius=self.cfg.PROBLEM.INSTANCE_SEG.FORE_EROSION_RADIUS, 
                fore_dilation_radius=self.cfg.PROBLEM.INSTANCE_SEG.FORE_DILATION_RADIUS, rmv_close_points=self.cfg.TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS, 
                remove_close_points_radius=self.cfg.TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS_RADIUS[0], resolution=self.cfg.DATA.TEST.RESOLUTION, save_dir=check_wa)

            save_tif(np.expand_dims(np.expand_dims(w_pred,-1),0), self.cfg.PATHS.RESULT_DIR.PER_IMAGE_INSTANCES,
                filenames, verbose=self.cfg.TEST.VERBOSE)

            # Add extra dimension if working in 2D
            if w_pred.ndim == 2:
                w_pred = np.expand_dims(w_pred,0)
        else:
            w_pred = pred.squeeze()
            if w_pred.ndim == 2: w_pred = np.expand_dims(w_pred,0)

        if self.cfg.TEST.MATCHING_STATS and (self.cfg.DATA.TEST.LOAD_GT or self.cfg.DATA.TEST.USE_VAL_AS_TEST):
            print("Calculating matching stats . . .")

            # Need to load instance labels, as Y are binary channels used for IoU calculation
            if self.cfg.TEST.ANALIZE_2D_IMGS_AS_3D_STACK:
                del self.Y
                _Y = np.zeros(w_pred.shape, dtype=w_pred.dtype)
                for i in range(len(self.test_filenames)):
                    test_file = os.path.join(self.original_test_mask_path, self.test_filenames[i])
                    _Y[i] = imread(test_file).squeeze()
            else:
                test_file = os.path.join(self.original_test_mask_path, self.test_filenames[self.f_numbers[0]])
                _Y = imread(test_file).squeeze()

            if _Y.ndim == 2: _Y = np.expand_dims(_Y,0)

            # For torchvision models that resize need to rezise the images 
            if w_pred.shape != self._Y.shape:
                _Y = resize(_Y, w_pred.shape, order=0)

            # Convert instances to integer
            if _Y.dtype == np.float32: _Y = _Y.astype(np.uint32)
            if _Y.dtype == np.float64: _Y = _Y.astype(np.uint64)

            diff_ths_colored_img = abs(len(self.cfg.TEST.MATCHING_STATS_THS_COLORED_IMG) - len(self.cfg.TEST.MATCHING_STATS_THS))
            colored_img_ths = self.cfg.TEST.MATCHING_STATS_THS_COLORED_IMG+[-1]*diff_ths_colored_img

            results = matching(_Y, w_pred, thresh=self.cfg.TEST.MATCHING_STATS_THS, report_matches=True)
            for i in range(len(results)):
                # Extract TPs, FPs and FNs from the resulting matching data structure 
                r_stats = results[i] 
                thr = r_stats['thresh']

                # TP and FN
                gt_ids = r_stats['gt_ids'][1:]
                matched_pairs = r_stats['matched_pairs']
                gt_match = [x[0] for x in matched_pairs]
                gt_unmatch = [x for x in gt_ids if x not in gt_match]
                matched_scores = list(r_stats['matched_scores'])+[0 for _ in gt_unmatch]
                pred_match = [x[1] for x in matched_pairs]+[-1 for _ in gt_unmatch]
                tag = ["TP" if score >= thr else "FN" for score in matched_scores]

                # FPs
                pred_ids = r_stats['pred_ids'][1:]
                fp_instances = [x for x in pred_ids if x not in pred_match]
                fp_instances += [pred_id for score, pred_id in zip(matched_scores, pred_match) if score < thr]

                # Save csv files
                df = pd.DataFrame(zip(gt_match+gt_unmatch, pred_match, matched_scores, tag), columns =['gt_id', 'pred_id', 'iou', 'tag'])
                df = df.sort_values(by=['gt_id'])  
                df_fp = pd.DataFrame(zip(fp_instances), columns =['pred_id'])

                os.makedirs(self.cfg.PATHS.RESULT_DIR.INST_ASSOC_POINTS, exist_ok=True)
                df.to_csv(os.path.join(self.cfg.PATHS.RESULT_DIR.INST_ASSOC_POINTS, os.path.splitext(filenames[0])[0]+'_th_{}_gt_assoc.csv'.format(thr)), index=False)
                df_fp.to_csv(os.path.join(self.cfg.PATHS.RESULT_DIR.INST_ASSOC_POINTS, os.path.splitext(filenames[0])[0]+'_th_{}_fp.csv'.format(thr)), index=False)
                del r_stats['matched_scores']; del r_stats['matched_tps']; del r_stats['matched_pairs']; del r_stats['pred_ids']; del r_stats['gt_ids']
                print("DatasetMatching: {}".format(r_stats))

                if colored_img_ths[i] != -1 and colored_img_ths[i] == thr:
                    print("Creating the image with a summary of detected points and false positives with colors . . .")
                    colored_result = np.zeros(w_pred.shape+(3,), dtype=np.uint8)

                    print("Painting TPs and FNs . . .")
                    for j in tqdm(range(len(gt_match))):
                        color = (0,255,0) if tag[j] == "TP" else (255,0,0) # Green or red
                        colored_result[np.where(_Y == gt_match[j])] = color
                    for j in tqdm(range(len(gt_unmatch))):
                        colored_result[np.where(_Y == gt_unmatch[j])] = (255,0,0) # Red

                    print("Painting FPs . . .")
                    for j in tqdm(range(len(fp_instances))):
                        colored_result[np.where(w_pred == fp_instances[j])] = (0,0,255) # Blue

                    save_tif(np.expand_dims(colored_result,0), self.cfg.PATHS.RESULT_DIR.INST_ASSOC_POINTS,
                            [os.path.splitext(filenames[0])[0]+'_th_{}.tif'.format(thr)], verbose=self.cfg.TEST.VERBOSE)          
                    del colored_result
            self.all_matching_stats.append(results)


        ###################
        # Post-processing #
        ###################
        if self.cfg.TEST.POST_PROCESSING.REPARE_LARGE_BLOBS_SIZE != -1:
            w_pred = repare_large_blobs(w_pred, self.cfg.TEST.POST_PROCESSING.REPARE_LARGE_BLOBS_SIZE)

        if self.cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.ENABLE or \
            self.cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.ENABLE:
            w_pred, d_result = measure_morphological_props_and_filter(w_pred.squeeze(), self.cfg.DATA.TEST.RESOLUTION, 
                filter_instances=self.cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.ENABLE,
                properties=self.cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS, 
                prop_values=self.cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.VALUES,
                comp_signs=self.cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.SIGN)

            # Save all instance stats            
            if self.cfg.PROBLEM.NDIM == "2D":
                df = pd.DataFrame(zip(np.array(d_result['labels'], dtype=np.uint64), list(d_result['centers'][:,0]), list(d_result['centers'][:,1]), 
                    d_result['npixels'], d_result['areas'], d_result['circularities'], d_result['diameters'], d_result['perimeters'], d_result['elongations'],
                    d_result['comment'], d_result['conditions']), columns=['label', 'axis-0', 'axis-1', 'npixels', 'area', 'circularity', 'diameter', 
                    'perimeter', 'elongation', 'comment', 'conditions'])
            else:
                df = pd.DataFrame(zip(np.array(d_result['labels'], dtype=np.uint64), list(d_result['centers'][:,0]), list(d_result['centers'][:,1]), 
                    list(d_result['centers'][:,2]), d_result['npixels'], d_result['areas'], d_result['circularities'], d_result['diameters'], 
                    d_result['perimeters'], d_result['comment'], d_result['conditions']), columns=['label', 'axis-0', 'axis-1', 
                    'axis-2', 'npixels', 'volume', 'sphericity', 'diameter', 'perimeter (surface area)', 'comment', 'conditions'])
            df = df.sort_values(by=['label'])   
            df.to_csv(os.path.join(self.cfg.PATHS.RESULT_DIR.PER_IMAGE_INSTANCES, os.path.splitext(filenames[0])[0]+'_full_stats.csv'), index=False)
            # Save only remain instances stats
            df = df[df["comment"].str.contains("Strange")==False] 
            os.makedirs(self.cfg.PATHS.RESULT_DIR.PER_IMAGE_POST_PROCESSING, exist_ok=True)
            df.to_csv(os.path.join(self.cfg.PATHS.RESULT_DIR.PER_IMAGE_POST_PROCESSING, os.path.splitext(filenames[0])[0]+'_filtered_stats.csv'), index=False)
            del df

        if self.cfg.TEST.POST_PROCESSING.VORONOI_ON_MASK:
            w_pred = voronoi_on_mask(w_pred, pred, th=self.cfg.TEST.POST_PROCESSING.VORONOI_TH, verbose=self.cfg.TEST.VERBOSE)
            del pred

        if self.cfg.TEST.POST_PROCESSING.CLEAR_BORDER:
            print("Clearing borders . . .")
            w_pred = clear_border(w_pred)

        if self.post_processing['instance_post']:
            save_tif(np.expand_dims(np.expand_dims(w_pred,-1),0), self.cfg.PATHS.RESULT_DIR.PER_IMAGE_POST_PROCESSING,
                filenames, verbose=self.cfg.TEST.VERBOSE)

            if self.cfg.TEST.MATCHING_STATS and (self.cfg.DATA.TEST.LOAD_GT or self.cfg.DATA.TEST.USE_VAL_AS_TEST):
                print("Calculating matching stats after post-processing . . .")
                results = matching(_Y, w_pred, thresh=self.cfg.TEST.MATCHING_STATS_THS, report_matches=True)
                
                for i in range(len(results)):
                    # Extract TPs, FPs and FNs from the resulting matching data structure 
                    r_stats = results[i] 
                    thr = r_stats['thresh']

                    # TP and FN
                    gt_ids = r_stats['gt_ids'][1:]
                    matched_pairs = r_stats['matched_pairs']
                    gt_match = [x[0] for x in matched_pairs]
                    gt_unmatch = [x for x in gt_ids if x not in gt_match]
                    matched_scores = list(r_stats['matched_scores'])+[0 for _ in gt_unmatch]
                    pred_match = [x[1] for x in matched_pairs]+[-1 for _ in gt_unmatch]
                    tag = ["TP" if score >= thr else "FN" for score in matched_scores]

                    # FPs
                    pred_ids = r_stats['pred_ids'][1:]
                    fp_instances = [x for x in pred_ids if x not in pred_match]
                    fp_instances += [pred_id for score, pred_id in zip(matched_scores, pred_match) if score < thr]

                    # Save csv files
                    df = pd.DataFrame(zip(gt_match+gt_unmatch, pred_match, matched_scores, tag), columns =['gt_id', 'pred_id', 'iou', 'tag'])
                    df = df.sort_values(by=['gt_id'])  
                    df_fp = pd.DataFrame(zip(fp_instances), columns =['pred_id'])

                    os.makedirs(self.cfg.PATHS.RESULT_DIR.INST_ASSOC_POINTS, exist_ok=True)
                    df.to_csv(os.path.join(self.cfg.PATHS.RESULT_DIR.INST_ASSOC_POINTS, os.path.splitext(filenames[0])[0]+'_post-proc_th_{}_gt_assoc.csv'.format(thr)), index=False)
                    df_fp.to_csv(os.path.join(self.cfg.PATHS.RESULT_DIR.INST_ASSOC_POINTS, os.path.splitext(filenames[0])[0]+'_post-proc_th_{}_fp.csv'.format(thr)), index=False)
                    del r_stats['matched_scores']; del r_stats['matched_tps']; del r_stats['matched_pairs']; del r_stats['pred_ids']; del r_stats['gt_ids']
                    print("DatasetMatching: {}".format(r_stats))

                    if colored_img_ths[i] != -1 and colored_img_ths[i] == thr:
                        print("Creating the image with a summary of detected points and false positives with colors . . .")
                        colored_result = np.zeros(w_pred.shape+(3,), dtype=np.uint8)

                        print("Painting TPs and FNs . . .")
                        for j in tqdm(range(len(gt_match))):
                            color = (0,255,0) if tag[j] == "TP" else (255,0,0) # Green or red
                            colored_result[np.where(_Y == gt_match[j])] = color
                        for j in tqdm(range(len(gt_unmatch))):
                            colored_result[np.where(_Y == gt_unmatch[j])] = (255,0,0) # Red
                            
                        print("Painting FPs . . .")
                        for j in tqdm(range(len(fp_instances))):
                            colored_result[np.where(w_pred == fp_instances[j])] = (0,0,255) # Blue

                        save_tif(np.expand_dims(colored_result,0), self.cfg.PATHS.RESULT_DIR.INST_ASSOC_POINTS,
                                [os.path.splitext(filenames[0])[0]+'_post-proc_th_{}.tif'.format(thr)], verbose=self.cfg.TEST.VERBOSE)          
                        del colored_result
                self.all_matching_stats_post_processing.append(results)

    def process_sample(self, norm):
        """
        Function to process a sample in the inference phase. 

        Parameters
        ----------
        norm : List of dicts
            Normalization used during training. Required to denormalize the predictions of the model.
        """
        if self.cfg.MODEL.SOURCE != "torchvision":
            self.instances_already_created = False
            super().process_sample(norm)
        else:
            self.instances_already_created = True
            # Data channel check
            if self.cfg.DATA.PATCH_SIZE[-1] != self._X.shape[-1]:
                raise ValueError("Channel of the DATA.PATCH_SIZE given {} does not correspond with the loaded image {}. "
                    "Please, check the channels of the images!".format(self.cfg.DATA.PATCH_SIZE[-1], self._X.shape[-1]))

            ##################
            ### FULL IMAGE ###
            ##################
            if self.cfg.TEST.STATS.FULL_IMG:
                # Make the prediction
                with torch.cuda.amp.autocast():
                    pred = self.model_call_func(self._X)
                del self._X 

                if self.cfg.TEST.POST_PROCESSING.APPLY_MASK:
                    pred = apply_binary_mask(pred, self.cfg.DATA.TEST.BINARY_MASKS)
    
                self.after_full_image(pred)

    def after_merge_patches(self, pred):
        """
        Steps need to be done after merging all predicted patches into the original image.

        Parameters
        ----------
        pred : Torch Tensor
            Model prediction.
        """
        if not self.cfg.TEST.ANALIZE_2D_IMGS_AS_3D_STACK:
            self.instance_seg_process(pred, self.processing_filenames)        

    def after_merge_patches_by_chunks_proccess_patch(self, filename):
        """
        Place any code that needs to be done after merging all predicted patches into the original image
        but in the process made chunk by chunk. This function will operate patch by patch defined by 
        ``DATA.PATCH_SIZE``.

        Parameters
        ----------
        filename : List of str
            Filename of the predicted image H5/Zarr.  
        """
        pass

    def after_full_image(self, pred):
        """
        Steps that must be executed after generating the prediction by supplying the entire image to the model.

        Parameters
        ----------
        pred : Torch Tensor
            Model prediction.
        """
        filename, file_extension = os.path.splitext(self.processing_filenames[0])
        self.instance_seg_process(pred, [filename+"_full_image"+file_extension])  

    def after_all_images(self):
        """
        Steps that must be done after predicting all images. 
        """
        super().after_all_images()
        if self.cfg.TEST.ANALIZE_2D_IMGS_AS_3D_STACK:
            print("Analysing all images as a 3D stack . . .")    
            if type(self.all_pred) is list:
                self.all_pred = np.concatenate(self.all_pred)
            self.instance_seg_process(self.all_pred, ["3D_stack.tif"])

    def normalize_stats(self, image_counter): 
        """
        Normalize statistics.  

        Parameters
        ----------
        image_counter : int
            Number of images to average the metrics.
        """
        super().normalize_stats(image_counter) 

        if (self.cfg.DATA.TEST.LOAD_GT or self.cfg.DATA.TEST.USE_VAL_AS_TEST):
            if self.cfg.TEST.MATCHING_STATS:
                self.stats['inst_stats'] = wrapper_matching_dataset_lazy(self.all_matching_stats, self.cfg.TEST.MATCHING_STATS_THS)
                if self.post_processing['instance_post']:
                    self.stats['inst_stats_vor'] = wrapper_matching_dataset_lazy(self.all_matching_stats_post_processing, self.cfg.TEST.MATCHING_STATS_THS)

    def print_stats(self, image_counter):
        """
        Print statistics.  

        Parameters
        ----------
        image_counter : int
            Number of images to call ``normalize_stats``.
        """
        if self.cfg.MODEL.SOURCE != "torchvision":
            super().print_stats(image_counter)
            super().print_post_processing_stats()

            print("Instance segmentation specific metrics:")
            if self.cfg.TEST.MATCHING_STATS and (self.cfg.DATA.TEST.LOAD_GT or self.cfg.DATA.TEST.USE_VAL_AS_TEST) :
                for i in range(len(self.cfg.TEST.MATCHING_STATS_THS)):
                    print("IoU TH={}".format(self.cfg.TEST.MATCHING_STATS_THS[i]))
                    print(self.stats['inst_stats'][i])
                    if self.post_processing['instance_post']:
                        print("IoU (post-processing) TH={}".format(self.cfg.TEST.MATCHING_STATS_THS[i]))
                        print(self.stats['inst_stats_vor'][i])

    def prepare_instance_data(self):
        """
        Creates instance segmentation ground truth images to train the model based on the ground truth instances provided.
        They will be saved in a separate folder in the root path of the ground truth. 
        """
        print("###########################")
        print("#  PREPARE INSTANCE DATA  #")
        print("###########################")
        original_test_path, original_test_mask_path = None, None

        # Create selected channels for train data
        if (self.cfg.TRAIN.ENABLE or self.cfg.DATA.TEST.USE_VAL_AS_TEST) and (not os.path.isdir(self.cfg.DATA.TRAIN.INSTANCE_CHANNELS_DIR) or \
            not os.path.isdir(self.cfg.DATA.TRAIN.INSTANCE_CHANNELS_MASK_DIR)):
            print("You select to create {} channels from given instance labels and no file is detected in {}. "
                    "So let's prepare the data. Notice that, if you do not modify 'DATA.TRAIN.INSTANCE_CHANNELS_DIR' "
                    "path, this process will be done just once!".format(self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS,
                    self.cfg.DATA.TRAIN.INSTANCE_CHANNELS_DIR))
            create_instance_channels(self.cfg)

        # Create selected channels for val data
        if self.cfg.TRAIN.ENABLE and not self.cfg.DATA.VAL.FROM_TRAIN and (not os.path.isdir(self.cfg.DATA.VAL.INSTANCE_CHANNELS_DIR) or \
            not os.path.isdir(self.cfg.DATA.VAL.INSTANCE_CHANNELS_MASK_DIR)):
            print("You select to create {} channels from given instance labels and no file is detected in {}. "
                    "So let's prepare the data. Notice that, if you do not modify 'DATA.VAL.INSTANCE_CHANNELS_DIR' "
                    "path, this process will be done just once!".format(self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS,
                    self.cfg.DATA.VAL.INSTANCE_CHANNELS_DIR))
            create_instance_channels(self.cfg, data_type='val')

        # Create selected channels for test data once
        if self.cfg.TEST.ENABLE and not self.cfg.DATA.TEST.USE_VAL_AS_TEST and (not os.path.isdir(self.cfg.DATA.TEST.INSTANCE_CHANNELS_DIR) or \
            (not os.path.isdir(self.cfg.DATA.TEST.INSTANCE_CHANNELS_MASK_DIR) and self.cfg.DATA.TEST.LOAD_GT)) and not self.cfg.TEST.BY_CHUNKS.ENABLE:
            print("You select to create {} channels from given instance labels and no file is detected in {}. "
                    "So let's prepare the data. Notice that, if you do not modify 'DATA.TEST.INSTANCE_CHANNELS_DIR' "
                    "path, this process will be done just once!".format(self.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS,
                    self.cfg.DATA.TEST.INSTANCE_CHANNELS_MASK_DIR))
            create_test_instance_channels(self.cfg)

        opts = []
        if self.cfg.TRAIN.ENABLE:
            print("DATA.TRAIN.PATH changed from {} to {}".format(self.cfg.DATA.TRAIN.PATH, self.cfg.DATA.TRAIN.INSTANCE_CHANNELS_DIR))
            print("DATA.TRAIN.GT_PATH changed from {} to {}".format(self.cfg.DATA.TRAIN.GT_PATH, self.cfg.DATA.TRAIN.INSTANCE_CHANNELS_MASK_DIR))
            opts.extend(['DATA.TRAIN.PATH', self.cfg.DATA.TRAIN.INSTANCE_CHANNELS_DIR,
                        'DATA.TRAIN.GT_PATH', self.cfg.DATA.TRAIN.INSTANCE_CHANNELS_MASK_DIR])
        if not self.cfg.DATA.VAL.FROM_TRAIN:
            print("DATA.VAL.PATH changed from {} to {}".format(self.cfg.DATA.VAL.PATH, self.cfg.DATA.VAL.INSTANCE_CHANNELS_DIR))
            print("DATA.VAL.GT_PATH changed from {} to {}".format(self.cfg.DATA.VAL.GT_PATH, self.cfg.DATA.VAL.INSTANCE_CHANNELS_MASK_DIR))
            opts.extend(['DATA.VAL.PATH', self.cfg.DATA.VAL.INSTANCE_CHANNELS_DIR,
                        'DATA.VAL.GT_PATH', self.cfg.DATA.VAL.INSTANCE_CHANNELS_MASK_DIR])
        if self.cfg.TEST.ENABLE and not self.cfg.DATA.TEST.USE_VAL_AS_TEST and not self.cfg.TEST.BY_CHUNKS.ENABLE:
            print("DATA.TEST.PATH changed from {} to {}".format(self.cfg.DATA.TEST.PATH, self.cfg.DATA.TEST.INSTANCE_CHANNELS_DIR))
            opts.extend(['DATA.TEST.PATH', self.cfg.DATA.TEST.INSTANCE_CHANNELS_DIR])
            if self.cfg.DATA.TEST.LOAD_GT:
                print("DATA.TEST.GT_PATH changed from {} to {}".format(self.cfg.DATA.TEST.GT_PATH, self.cfg.DATA.TEST.INSTANCE_CHANNELS_MASK_DIR))
                opts.extend(['DATA.TEST.GT_PATH', self.cfg.DATA.TEST.INSTANCE_CHANNELS_MASK_DIR])
        original_test_path = self.cfg.DATA.TEST.PATH
        original_test_mask_path = self.cfg.DATA.TEST.GT_PATH
        self.cfg.merge_from_list(opts)

        return original_test_path, original_test_mask_path

    def torchvision_model_call(self, in_img, is_train=False):
        """
        Call a regular Pytorch model.

        Parameters
        ----------
        in_img : Tensor
            Input image to pass through the model.

        is_train : bool, optional
            Whether if the call is during training or inference. 

        Returns
        -------
        prediction : Tensor 
            Image prediction. 
        """
        filename, file_extension = os.path.splitext(self.processing_filenames[0])

        # Convert first to 0-255 range if uint16
        if in_img.dtype == torch.float32:
            if torch.max(in_img) > 1:
                in_img = (norm_range01(in_img, torch.uint8)[0]*255).to(torch.uint8)
            in_img = in_img.to(torch.uint8)
        
        # Apply TorchVision pre-processing
        in_img = self.torchvision_preprocessing(in_img)
        pred = self.model(in_img)
        masks = pred[0]['masks'].cpu().numpy().transpose(0,2,3,1)
        if masks.shape[0] != 0:
            masks = np.argmax(pred[0]['masks'].cpu().numpy().transpose(0,2,3,1), axis=0)
        else:
            masks = torch.ones((1,)+pred[0]['masks'].cpu().numpy().transpose(0,2,3,1).shape[1:], dtype=torch.uint8)

        if not is_train and masks.shape[0] != 0:
            # Extract each output from MaskRCNN
            bboxes = pred[0]['boxes'].cpu().numpy().astype(np.uint16)
            labels = pred[0]['labels'].cpu().numpy()
            scores = pred[0]['scores'].cpu().numpy()
            
            # Save all info in a csv file
            df = pd.DataFrame(zip(labels, scores, bboxes[:,0],bboxes[:,1],bboxes[:,2],bboxes[:,3]), 
                columns = ['label', 'scores', 'x1', 'y1', 'x2', 'y2'])
            df = df.sort_values(by=['label']) 
            df.to_csv(os.path.join(self.cfg.PATHS.RESULT_DIR.FULL_IMAGE, filename+".csv"), index=False)

            # Save masks
            save_tif(np.expand_dims(masks,0), self.cfg.PATHS.RESULT_DIR.FULL_IMAGE, self.processing_filenames, 
                verbose=self.cfg.TEST.VERBOSE)    

        # Actually to allow training this should return a Torch Tensor and not a Numpy array. For instance, 
        # segmentation training is disabled, due to the absence of generators that contain bboxes, so this can be left 
        # returning a Numpy array. This will only be called in process_sample inference function and for full image setting
        return np.expand_dims(masks.squeeze(),0) 