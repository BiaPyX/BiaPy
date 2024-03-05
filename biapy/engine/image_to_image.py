import os
import torch
import numpy as np
from skimage.transform import resize

from biapy.engine.base_workflow import Base_Workflow
from biapy.utils.util import save_tif, check_masks
from biapy.utils.misc import to_pytorch_format, to_numpy_format
from biapy.engine.metrics import jaccard_index, CrossEntropyLoss_wrapper, weighted_bce_dice_loss, jaccard_index_numpy, voc_calculation
from biapy.data.pre_processing import norm_range01
from biapy.data.post_processing.post_processing import ensemble8_2d_predictions

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
        print("Overriding 'LOSS.TYPE' to set it to N2V loss (masked MSE)")
        self.metrics = [torch.nn.MSELoss()]
        self.metric_names = ["MSE"]
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
        with torch.no_grad():
            train_mse = self.metrics[0](output.squeeze(), targets.squeeze())
            train_mse = train_mse.item() if not torch.isnan(train_mse) else 0
            if metric_logger is not None:
                metric_logger.meters[self.metric_names[0]].update(train_mse)
            else:
                return train_mse

    def process_sample(self, norm):
        """
        Function to process a sample in the inference phase. 

        Parameters
        ----------
        norm : List of dicts
            Normalization used during training. Required to denormalize the predictions of the model.
        """
        if self.cfg.MODEL.SOURCE != "torchvision":
            super().process_sample(norm)
        else:
            NotImplementedError

                
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
        pass

    def print_stats(self, image_counter):
        """
        Print statistics.  

        Parameters
        ----------
        image_counter : int
            Number of images to call ``normalize_stats``.
        """
        self.normalize_stats(image_counter)


        