import math
import torch
import numpy as np
from tqdm import tqdm
from torchmetrics.image import PeakSignalNoiseRatio

from biapy.data.data_2D_manipulation import crop_data_with_overlap, merge_data_with_overlap
from biapy.data.data_3D_manipulation import crop_3D_data_with_overlap, merge_3D_data_with_overlap
from biapy.data.post_processing.post_processing import ensemble8_2d_predictions, ensemble16_3d_predictions
from biapy.utils.util import save_tif
from biapy.utils.misc import to_pytorch_format, to_numpy_format, is_main_process
from biapy.engine.base_workflow import Base_Workflow
from biapy.engine.metrics import dfcan_loss
from biapy.data.pre_processing import normalize, denormalize, undo_norm_range01

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
        Definition of self.metrics, self.metric_names and self.loss variables.
        """
        self.metrics = [PeakSignalNoiseRatio()]
        self.metric_names = ["PSNR"]
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

        with torch.no_grad():
            train_psnr = self.metrics[0](output, targets)
            train_psnr = train_psnr.item() if not torch.isnan(train_psnr) else 0
            if metric_logger is not None:
                metric_logger.meters[self.metric_names[0]].update(train_psnr)
            else:
                return train_psnr

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
        if pred.dtype == np.dtype('uint16'):
            pred = pred.astype(np.float32)
        
        if self.cfg.DATA.TEST.LOAD_GT or self.cfg.DATA.TEST.USE_VAL_AS_TEST:
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
        pass

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

