import math
import torch
import numpy as np
from tqdm import tqdm
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance 
from torchmetrics.image.inception import InceptionScore


from biapy.data.data_2D_manipulation import crop_data_with_overlap, merge_data_with_overlap
from biapy.data.data_3D_manipulation import crop_3D_data_with_overlap, merge_3D_data_with_overlap
from biapy.data.post_processing.post_processing import ensemble8_2d_predictions, ensemble16_3d_predictions
from biapy.utils.util import save_tif
from biapy.utils.misc import to_pytorch_format, to_numpy_format, is_main_process
from biapy.engine.base_workflow import Base_Workflow
from biapy.engine.metrics import dfcan_loss
from biapy.data.pre_processing import normalize, denormalize, norm_range01, undo_norm_range01

class Super_resolution_Workflow(Base_Workflow):
    """
    Semantic segmentation workflow where the goal is to assign a class to each pixel of the input image. 
    More details in `our documentation <https://biapy.readthedocs.io/en/latest/workflows/super_resolution.html>`_.  

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
        super(Super_resolution_Workflow, self).__init__(cfg, job_identifier, device, args, **kwargs)
        
        self.stats['psnr_merge_patches'] = 0
        self.stats['mse_merge_patches'] = 0
        self.stats['mae_merge_patches'] = 0
        self.stats['ssim_merge_patches'] = 0

        self.stats['fid_merge_patches'] = 0
        self.stats['iscore_merge_patches'] = 0
        self.stats['lpips_merge_patches'] = 0

        # From now on, no modification of the cfg will be allowed
        self.cfg.freeze()

        # Activations for each output channel:
        # channel number : 'activation'
        self.activations = [{':': 'Linear'}]

        # Workflow specific training variables
        self.mask_path = cfg.DATA.TRAIN.GT_PATH
        self.load_Y_val = True

    def define_metrics(self):
        """
        Definition of self.metrics, self.metric_names, self.test_metrics, self.test_metric_names and self.loss variables.
        """
        self.metrics = [PeakSignalNoiseRatio(), 
                        MeanSquaredError(),
                        MeanAbsoluteError(),
                        StructuralSimilarityIndexMeasure()]
        
        self.metric_names = ["PSNR", 
                             "MSE",
                             "MAE",
                             "SSIM"]
        
        self.test_metrics = [FrechetInceptionDistance(normalize=True),
                             InceptionScore(normalize=True),
                             LearnedPerceptualImagePatchSimilarity(net_type='squeeze', normalize=True)]
        
        self.test_metric_names = ["FID",
                                  "IS",
                                  "LPIPS"]
        
        if self.cfg.MODEL.ARCHITECTURE == 'dfcan':
            print("Overriding 'LOSS.TYPE' to set it to DFCAN loss")
            self.loss = dfcan_loss(self.device)
        else:
            print("Overriding 'LOSS.TYPE' to set it to MAE")
            self.loss = torch.nn.L1Loss()

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
        # Denormalization to calculate PSNR with original range values 
        output = output.to(torch.float32).detach().cpu()
        targets = targets.to(torch.float32).detach().cpu()
        if self.data_norm['type'] == 'div':
            if len([x for x in list(self.data_norm.keys()) if not 'reduced' in x]) > 0:
                output = torch.clip(output*255, 0, 255) 
                targets = torch.round(targets*255) 
            else:
                output = torch.clip(output*65535, 0, 65535)
                targets = torch.round(targets*65535) 
            output = torch.round(output)
        else:
            output = (output * self.data_norm['std']) + self.data_norm['mean']
            output = torch.round(output)                                                                 
            output = output+abs(torch.min(output))

            targets = (targets * self.data_norm['std']) + self.data_norm['mean']
            targets = torch.round(targets)                                                                 
            targets = targets+abs(torch.min(targets))

        # Reshape (in case its necessary) to follow PyTorch format (B, C, H, W)
        if output.shape[-1] == self.cfg.DATA.PATCH_SIZE[-1]:
            output = output.permute(0, 3, 1, 2)
        if targets.shape[-1] == self.cfg.DATA.PATCH_SIZE[-1]:
            targets = targets.permute(0, 3, 1, 2)

        with torch.no_grad():
            # Calculate PSNR
            train_psnr = self.metrics[0](output, targets)
            train_psnr = train_psnr.item() if not torch.isnan(train_psnr) else 0

            # Calculate MSE
            train_mse = self.metrics[1](output, targets)
            train_mse = train_mse.item() if not torch.isnan(train_mse) else 0

            # Calculate MAE
            train_mae = self.metrics[2](output, targets)
            train_mae = train_mae.item() if not torch.isnan(train_mae) else 0

            # Calculate SSIM
            train_ssim = self.metrics[3](output, targets)
            train_ssim = train_ssim.item() if not torch.isnan(train_ssim) else 0

            if metric_logger is not None:
                # Metrics computed here, will only be calculated during training

                metric_logger.meters[self.metric_names[0]].update(train_psnr)
                metric_logger.meters[self.metric_names[1]].update(train_mse)
                metric_logger.meters[self.metric_names[2]].update(train_mae)
                metric_logger.meters[self.metric_names[3]].update(train_ssim)
            else:
                # Metrics computed here, will only be calculated during testing

                # The metrcis below need to have normalized (between 0 and 1) images with 3 channels
                norm_output = (output - torch.min(output))/(torch.max(output) - torch.min(output) + 1e-8)
                norm_targets = (targets - torch.min(targets))/(torch.max(targets) - torch.min(targets) + 1e-8)
                norm_3c_output = torch.cat([norm_output, norm_output, norm_output], dim=1)
                norm_3c_targets = torch.cat([norm_targets, norm_targets, norm_targets], dim=1)

                # Update FID (it will be computed on self.after_all_images())
                self.test_metrics[0].update(norm_3c_output, real=True)
                self.test_metrics[0].update(norm_3c_targets, real=False)

                # Update IS (it will be computed on self.after_all_images())
                self.test_metrics[1].update(norm_3c_targets)

                # Update LPIPS (it will be computed on self.after_all_images())
                self.test_metrics[2].update(norm_3c_output, norm_3c_targets)
                
                return train_psnr, train_mse, train_mae, train_ssim

    def process_sample(self, norm): 
        """
        Function to process a sample in the inference phase. 

        Parameters
        ----------
        norm : List of dicts
            Normalization used during training. Required to denormalize the predictions of the model.
        """
        if self.cfg.PROBLEM.NDIM == '2D':
            original_data_shape = (self._X.shape[0],
                                   self._X.shape[1]*self.cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING[0],
                                   self._X.shape[2]*self.cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING[1],
                                   self._X.shape[3])
        else:
            original_data_shape = (self._X.shape[0],
                                   self._X.shape[1]*self.cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING[0],
                                   self._X.shape[2]*self.cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING[1],
                                   self._X.shape[3]*self.cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING[2],
                                   self._X.shape[4])

        # Crop if necessary
        if self._X.shape[1:-1] != self.cfg.DATA.PATCH_SIZE[:-1]:
            if self.cfg.PROBLEM.NDIM == '2D':
                self._X = crop_data_with_overlap(self._X, self.cfg.DATA.PATCH_SIZE, overlap=self.cfg.DATA.TEST.OVERLAP, 
                    padding=self.cfg.DATA.TEST.PADDING, verbose=self.cfg.TEST.VERBOSE)
            else:
                self._X = crop_3D_data_with_overlap(self._X[0], self.cfg.DATA.PATCH_SIZE, overlap=self.cfg.DATA.TEST.OVERLAP, 
                    padding=self.cfg.DATA.TEST.PADDING, verbose=self.cfg.TEST.VERBOSE)

        # Predict each patch
        if self.cfg.TEST.AUGMENTATION:
            for k in tqdm(range(self._X.shape[0]), leave=False, disable=not is_main_process()):
                if self.cfg.PROBLEM.NDIM == '2D':
                    p = ensemble8_2d_predictions(self._X[k], axis_order_back=self.axis_order_back,
                            pred_func=self.model_call_func, axis_order=self.axis_order, device=self.device)
                else:
                    p = ensemble16_3d_predictions(self._X[k], batch_size_value=self.cfg.TRAIN.BATCH_SIZE,
                            axis_order_back=self.axis_order_back, pred_func=self.model_call_func, 
                            axis_order=self.axis_order, device=self.device)
                p = self.apply_model_activations(p)
                p = to_numpy_format(p, self.axis_order_back)
                if 'pred' not in locals():
                    pred = np.zeros((self._X.shape[0],)+p.shape[1:], dtype=self.dtype)
                pred[k] = p
        else:
            self._X = to_pytorch_format(self._X, self.axis_order, self.device)
            l = int(math.ceil(self._X.shape[0]/self.cfg.TRAIN.BATCH_SIZE))
            for k in tqdm(range(l), leave=False, disable=not is_main_process()):
                top = (k+1)*self.cfg.TRAIN.BATCH_SIZE if (k+1)*self.cfg.TRAIN.BATCH_SIZE < self._X.shape[0] else self._X.shape[0]
                with torch.cuda.amp.autocast():
                    p = self.model(self._X[k*self.cfg.TRAIN.BATCH_SIZE:top])
                p = to_numpy_format(self.apply_model_activations(p), self.axis_order_back)
                if 'pred' not in locals():
                    pred = np.zeros((self._X.shape[0],)+p.shape[1:], dtype=self.dtype)
                pred[k*self.cfg.TRAIN.BATCH_SIZE:top] = p
        del self._X, p

        # Reconstruct the predictions
        if original_data_shape[1:-1] != self.cfg.DATA.PATCH_SIZE[:-1]:
            if self.cfg.PROBLEM.NDIM == '3D': original_data_shape = original_data_shape[1:]
            f_name = merge_data_with_overlap if self.cfg.PROBLEM.NDIM == '2D' else merge_3D_data_with_overlap
            if self.cfg.PROBLEM.NDIM == '2D':
                pad = tuple(p*self.cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING[0] for p in self.cfg.DATA.TEST.PADDING)
                ov = tuple(o*self.cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING[1] for o in self.cfg.DATA.TEST.OVERLAP)
            else:
                pad = (self.cfg.DATA.TEST.PADDING[0]*self.cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING[0],
                       self.cfg.DATA.TEST.PADDING[1]*self.cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING[1],
                       self.cfg.DATA.TEST.PADDING[2]*self.cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING[2])
                ov = (self.cfg.DATA.TEST.OVERLAP[0]*self.cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING[0],
                      self.cfg.DATA.TEST.OVERLAP[1]*self.cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING[1],
                      self.cfg.DATA.TEST.OVERLAP[2]*self.cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING[2])
            pred = f_name(pred, original_data_shape[:-1]+(pred.shape[-1],), padding=pad, 
                overlap=ov, verbose=self.cfg.TEST.VERBOSE)

            if self.cfg.PROBLEM.NDIM == '3D': 
                pred = np.expand_dims(pred,0)
                if self._Y is not None:  self._Y = np.expand_dims(self._Y,0)

        # Undo normalization
        x_norm = norm[0]
        if x_norm['type'] == 'div':
            pred = undo_norm_range01(pred, x_norm)
        elif x_norm['type'] == 'scale_range':
            pred = undo_norm_range01(pred, x_norm, x_norm['min_val_scale'], x_norm['max_val_scale'])
        else:
            pred = denormalize(pred, x_norm['mean'], x_norm['std'])  
            
            if x_norm['orig_dtype'] not in [np.dtype('float64'), np.dtype('float32'), np.dtype('float16')]:
                pred = np.round(pred)
                minpred = np.min(pred)                                                                                                
                pred = pred+abs(minpred)

            pred = pred.astype(x_norm['orig_dtype'])

        # Save image
        if self.cfg.PATHS.RESULT_DIR.PER_IMAGE != "":
            save_tif(pred, self.cfg.PATHS.RESULT_DIR.PER_IMAGE, self.processing_filenames, 
                verbose=self.cfg.TEST.VERBOSE)
    
        # Calculate metrics
        if pred.dtype == np.dtype('uint16'):
            pred = pred.astype(np.float32)
        
        if self.cfg.DATA.TEST.LOAD_GT or self.cfg.DATA.TEST.USE_VAL_AS_TEST:
            if self._Y.dtype == np.dtype('uint16'):
                self._Y = self._Y.astype(np.float32)
                
            psnr, mse, mae, ssim = self.metric_calculation(torch.from_numpy(self._Y), 
                                                            torch.from_numpy(pred), 
                                                            metric_logger=None)

            self.stats['psnr_merge_patches'] += psnr
            self.stats['mse_merge_patches'] += mse
            self.stats['mae_merge_patches'] += mae
            self.stats['ssim_merge_patches'] += ssim
            
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
        pass

    def after_merge_patches(self, pred):
        """
        Steps need to be done after merging all predicted patches into the original image.

        Parameters
        ----------
        pred : Torch Tensor
            Model prediction.
        """
        pass

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
        pass

    def after_all_images(self):
        """
        Steps that must be done after predicting all images. 
        """
        # FID, IS and LPIPS need to be computed for all the images
        train_fid = self.test_metrics[0].compute()
        train_fid = train_fid.item() if not torch.isnan(train_fid) else 0
        self.stats['fid_merge_patches'] = train_fid
        
        train_is =  self.test_metrics[1].compute()[0] # It returns a the mean and the std, we only need the mean
        train_is = train_is.item() if not torch.isnan(train_is) else 0
        self.stats['iscore_merge_patches'] = train_is
        
        train_lpips = self.test_metrics[2].compute()
        train_lpips = train_lpips.item() if not torch.isnan(train_lpips) else 0
        self.stats['lpips_merge_patches'] = train_lpips

        super(Super_resolution_Workflow, self)

    def normalize_stats(self, image_counter):
        """
        Normalize statistics.  

        Parameters
        ----------
        image_counter : int
            Number of images to average the metrics.
        """
        self.stats['psnr_merge_patches'] = self.stats['psnr_merge_patches'] / image_counter
        self.stats['mse_merge_patches'] = self.stats['mse_merge_patches'] / image_counter
        self.stats['mae_merge_patches'] = self.stats['mae_merge_patches'] / image_counter
        self.stats['ssim_merge_patches'] = self.stats['ssim_merge_patches'] / image_counter

        # FID, IS and LPIPS are already normalized

    def print_stats(self, image_counter):
        """
        Print statistics.  

        Parameters
        ----------
        image_counter : int
            Number of images to call ``normalize_stats``.
        """
        self.normalize_stats(image_counter)

        if self.cfg.DATA.TEST.LOAD_GT or self.cfg.DATA.TEST.USE_VAL_AS_TEST:
            print("Test PSNR (merge patches): {}".format(self.stats['psnr_merge_patches']))
            print("Test MSE (merge patches): {}".format(self.stats['mse_merge_patches']))
            print("Test MAE (merge patches): {}".format(self.stats['mae_merge_patches']))
            print("Test SSIM (merge patches): {}".format(self.stats['ssim_merge_patches']))
            print("Test FID (merge patches): {}".format(self.stats['fid_merge_patches']))
            print("Test IS (merge patches): {}".format(self.stats['iscore_merge_patches']))
            print("Test LPIPS (merge patches): {}".format(self.stats['lpips_merge_patches']))
            print(" ")
