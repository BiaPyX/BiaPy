import os
import math
import torch
import numpy as np
from skimage.transform import resize
from torchmetrics.image import PeakSignalNoiseRatio
from tqdm import tqdm

from biapy.engine.base_workflow import Base_Workflow
from biapy.utils.util import save_tif, check_masks, pad_and_reflect
from biapy.utils.misc import to_pytorch_format, to_numpy_format
from biapy.data.pre_processing import norm_range01, undo_norm_range01, denormalize
from biapy.data.post_processing.post_processing import ensemble8_2d_predictions, ensemble16_3d_predictions
from biapy.data.data_2D_manipulation import crop_data_with_overlap, merge_data_with_overlap
from biapy.data.data_3D_manipulation import crop_3D_data_with_overlap, merge_3D_data_with_overlap
from biapy.engine.metrics import weighted_L1

class Image_to_Image_Workflow(Base_Workflow):
    """
    Image to image workflow where the goal is .. 

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
        super(Image_to_Image_Workflow, self).__init__(cfg, job_identifier, device, args, **kwargs)
        self.stats['psnr_merge_patches'] = 0
        
        # From now on, no modification of the cfg will be allowed
        self.cfg.freeze()

        # Activations for each output channel:
        # channel number : 'activation'
        self.activations = [{':': 'Linear'}]
        
        self.mask_path = cfg.DATA.TRAIN.GT_PATH
        
    def define_metrics(self):
        """
        Definition of self.metrics, self.metric_names and self.loss variables.
        """
        self.metrics = [PeakSignalNoiseRatio().to(self.device), torch.nn.MSELoss()]
        self.metric_names = ["PSNR", "MSE"]
        print("Overriding 'LOSS.TYPE' to set it to MAE")
        # self.loss = torch.nn.L1Loss()
        self.loss = weighted_L1()

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
            train_psnr = self.metrics[0](output.squeeze(), targets[0].squeeze())
            train_psnr = train_psnr.item() if not torch.isnan(train_psnr) else 0

            train_mse = self.metrics[1](output.squeeze(), targets[0].squeeze())
            train_mse = train_mse.item() if not torch.isnan(train_mse) else 0

            if metric_logger is not None:
                metric_logger.meters[self.metric_names[0]].update(train_psnr)
                metric_logger.meters[self.metric_names[1]].update(train_mse)

            return train_psnr, train_mse

    def prepare_targets(self, targets, batch):
        """
        Location to perform any necessary data transformations to ``targets``
        before calculating the loss.

        Parameters
        ----------
        targets : Torch Tensor
            Ground truth to compare the prediction with.

        batch : Torch Tensor
            Prediction of the model. Only used in SSL workflow. 

        Returns
        -------
        targets : Torch tensor
            Resulting targets. 
        """
        # We do not use 'batch' input but in SSL workflow
        return [to_pytorch_format(targets[0], self.axis_order, self.device), targets[1].to(self.device, non_blocking=True)]

    def process_sample(self, norm): 
        """
        Function to process a sample in the inference phase. 

        Parameters
        ----------
        norm : List of dicts
            Normalization used during training. Required to denormalize the predictions of the model.
        """

        # Reflect data to complete the needed shape
        if self.cfg.DATA.REFLECT_TO_COMPLETE_SHAPE:
            reflected_orig_shape = self._X.shape
            self._X = np.expand_dims(pad_and_reflect(self._X[0], self.cfg.DATA.PATCH_SIZE, verbose=self.cfg.TEST.VERBOSE),0)
            if self.cfg.DATA.TEST.LOAD_GT:
                self._Y = np.expand_dims(pad_and_reflect(self._Y[0], self.cfg.DATA.PATCH_SIZE, verbose=self.cfg.TEST.VERBOSE),0)
        
        original_data_shape = self._X.shape

        # Crop if necessary
        if self._X.shape[1:-1] != self.cfg.DATA.PATCH_SIZE[:-1]:
            if self.cfg.PROBLEM.NDIM == '2D':
                obj = crop_data_with_overlap(self._X, self.cfg.DATA.PATCH_SIZE, data_mask=self._Y, overlap=self.cfg.DATA.TEST.OVERLAP, 
                    padding=self.cfg.DATA.TEST.PADDING, verbose=self.cfg.TEST.VERBOSE)
                if self.cfg.DATA.TEST.LOAD_GT or self.cfg.DATA.TEST.USE_VAL_AS_TEST:
                    self._X, self._Y = obj
                else:
                    self._X = obj
                del obj
            else:
                if self.cfg.DATA.TEST.LOAD_GT or self.cfg.DATA.TEST.USE_VAL_AS_TEST: self._Y = self._Y[0]
                if self.cfg.TEST.REDUCE_MEMORY:
                    self._X = crop_3D_data_with_overlap(self._X[0], self.cfg.DATA.PATCH_SIZE, overlap=self.cfg.DATA.TEST.OVERLAP, 
                        padding=self.cfg.DATA.TEST.PADDING, verbose=self.cfg.TEST.VERBOSE, 
                        median_padding=self.cfg.DATA.TEST.MEDIAN_PADDING)
                    self._Y = crop_3D_data_with_overlap(self._Y, self.cfg.DATA.PATCH_SIZE, overlap=self.cfg.DATA.TEST.OVERLAP, 
                        padding=self.cfg.DATA.TEST.PADDING, verbose=self.cfg.TEST.VERBOSE, 
                        median_padding=self.cfg.DATA.TEST.MEDIAN_PADDING)
                else:
                    obj = crop_3D_data_with_overlap(self._X[0], self.cfg.DATA.PATCH_SIZE, data_mask=self._Y, overlap=self.cfg.DATA.TEST.OVERLAP, 
                        padding=self.cfg.DATA.TEST.PADDING, verbose=self.cfg.TEST.VERBOSE, 
                        median_padding=self.cfg.DATA.TEST.MEDIAN_PADDING)
                    if self.cfg.DATA.TEST.LOAD_GT or self.cfg.DATA.TEST.USE_VAL_AS_TEST:
                        self._X, self._Y = obj
                    else:
                        self._X = obj
                    del obj

        # Predict each patch
        if self.cfg.TEST.AUGMENTATION:
            for k in tqdm(range(self._X.shape[0]), leave=False):
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
            for k in tqdm(range(l), leave=False):
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

            if self.cfg.TEST.REDUCE_MEMORY:
                pred = f_name(pred, original_data_shape[:-1]+(pred.shape[-1],), padding=self.cfg.DATA.TEST.PADDING, 
                    overlap=self.cfg.DATA.TEST.OVERLAP, verbose=self.cfg.TEST.VERBOSE)
                self._Y = f_name(self._Y, original_data_shape[:-1]+(self._Y.shape[-1],), padding=self.cfg.DATA.TEST.PADDING, 
                    overlap=self.cfg.DATA.TEST.OVERLAP, verbose=self.cfg.TEST.VERBOSE)
            else:
                obj = f_name(pred, original_data_shape[:-1]+(pred.shape[-1],), data_mask=self._Y,
                    padding=self.cfg.DATA.TEST.PADDING, overlap=self.cfg.DATA.TEST.OVERLAP,
                    verbose=self.cfg.TEST.VERBOSE)
                if self.cfg.DATA.TEST.LOAD_GT or self.cfg.DATA.TEST.USE_VAL_AS_TEST:
                    pred, self._Y = obj
                else:
                    pred = obj
                del obj

            if self.cfg.PROBLEM.NDIM == '3D': 
                pred = np.expand_dims(pred,0)
                if self._Y is not None:  self._Y = np.expand_dims(self._Y,0)

        if self.cfg.DATA.REFLECT_TO_COMPLETE_SHAPE: 
            if self.cfg.PROBLEM.NDIM == '2D':
                pred = pred[:,-reflected_orig_shape[1]:,-reflected_orig_shape[2]:]
                if self._Y is not None:
                    self._Y = self._Y[:,-reflected_orig_shape[1]:,-reflected_orig_shape[2]:]
            else:
                pred = pred[:,-reflected_orig_shape[1]:,-reflected_orig_shape[2]:,-reflected_orig_shape[3]:]
                if self._Y is not None:
                    self._Y = self._Y[:,-reflected_orig_shape[1]:,-reflected_orig_shape[2]:,-reflected_orig_shape[3]:]

        # Undo normalization
        x_norm = norm[0]
        if x_norm['type'] == 'div':
            pred = undo_norm_range01(pred, x_norm)
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

        # Calculate PSNR
        if self.cfg.DATA.TEST.LOAD_GT or self.cfg.DATA.TEST.USE_VAL_AS_TEST:
            if pred.dtype == np.dtype('uint16'):
                pred = pred.astype(np.float32)
            if self._Y.dtype == np.dtype('uint16'):
                self._Y = self._Y.astype(np.float32)
            psnr_merge_patches = self.metrics[0](torch.from_numpy(pred), torch.from_numpy(self._Y))
            self.stats['psnr_merge_patches'] += psnr_merge_patches
        
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
        super().after_all_images()

    def normalize_stats(self, image_counter):
        """
        Normalize statistics.  

        Parameters
        ----------
        image_counter : int
            Number of images to average the metrics.
        """
        self.stats['psnr_merge_patches'] = self.stats['psnr_merge_patches'] / image_counter

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
            print(" ")


        