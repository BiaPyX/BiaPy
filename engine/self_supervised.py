import os
import torch
import math
import numpy as np
from tqdm import tqdm
from torchmetrics.image import PeakSignalNoiseRatio

from data.data_2D_manipulation import crop_data_with_overlap, merge_data_with_overlap
from data.data_3D_manipulation import crop_3D_data_with_overlap, merge_3D_data_with_overlap
from data.post_processing.post_processing import ensemble8_2d_predictions, ensemble16_3d_predictions
from utils.util import save_tif
from utils.misc import to_pytorch_format, to_numpy_format
from engine.base_workflow import Base_Workflow
from data.pre_processing import create_ssl_source_data_masks, denormalize, undo_norm_range01
from engine.metrics import MaskedAutoencoderViT_loss

class Self_supervised_Workflow(Base_Workflow):
    """
    Self supervised workflow where the goal is to pretrain the backbone model by solving a so-called 
    pretext task without labels. This way, the model learns a representation that can be later transferred 
    to solve a downstream task in a labeled (but smaller) dataset. More details in `our documentation 
    <https://biapy.readthedocs.io/en/latest/workflows/self_supervision.html>`_.  

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
        super(Self_supervised_Workflow, self).__init__(cfg, job_identifier, device, args, **kwargs)
        
        self.prepare_ssl_data()

        # From now on, no modification of the cfg will be allowed
        self.cfg.freeze()

        # Activations for each output channel:
        # channel number : 'activation'
        self.activations = {':': 'Linear'}

        # Workflow specific training variables
        self.mask_path = None
        if cfg.PROBLEM.SELF_SUPERVISED.PRETEXT_TASK == 'masking':
            self.load_Y_val = False
        else:
            self.load_Y_val = True

    def define_metrics(self):
        """
        Definition of self.metrics, self.metric_names and self.loss variables.
        """
        self.metrics = [PeakSignalNoiseRatio()]
        self.metric_names = ["PSNR"]
        if self.cfg.MODEL.ARCHITECTURE == 'mae':
            print("Overriding 'LOSS.TYPE' to set it to MSE loss (masking patches)")
            self.loss = self.MaskedAutoencoderViT_loss_wrapper
        else:
            print("Overriding 'LOSS.TYPE' to set it to L1 loss")
            self.loss = torch.nn.L1Loss()

    def MaskedAutoencoderViT_loss_wrapper(self, output, targets):
        """
        Unravel MAE loss.
        """
        # Targets not used because the loss has been already calculated
        loss, pred, mask = output
        return loss

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
        # Calculate PSNR
        if self.cfg.PROBLEM.SELF_SUPERVISED.PRETEXT_TASK == 'masking':
            _, pred, _ = output
            pred = self.model_without_ddp.unpatchify(pred).to(torch.float32).detach().cpu()
        else:
            pred = output.to(torch.float32).detach().cpu()
        targets = targets.to(torch.float32).detach().cpu()
        with torch.no_grad():
            train_psnr = self.metrics[0](pred, targets)
            train_psnr = train_psnr.item() if not torch.isnan(train_psnr) else 0
            if metric_logger is not None:
                metric_logger.meters[self.metric_names[0]].update(train_psnr)
            else:
                return train_psnr

    def prepare_targets(self, targets, batch):
        """
        Location to perform any necessary data transformations to ``targets``
        before inputting it into the model.

        Parameters
        ----------
        targets : Torch Tensor
            Ground truth to compare the prediction with.

        batch : Torch Tensor
            Prediction of the model. 

        Returns
        -------
        targets : Torch tensor
            Resulting targets. 
        """
        # Swap with original images so we can calculate PSNR metric afterwards
        return batch

    def process_sample(self, filenames, norm): 
        """
        Function to process a sample in the inference phase. 

        Parameters
        ----------
        filenames : List of str
            Filenames fo the samples to process. 

        norm : List of dicts
            Normalization used during training. Required to denormalize the predictions of the model.
        """
        original_data_shape = self._X.shape
    
        # Crop if necessary
        if self._X.shape[1:-1] != self.cfg.DATA.PATCH_SIZE[:-1]:
            if self.cfg.PROBLEM.NDIM == '2D':
                self._X = crop_data_with_overlap(self._X, self.cfg.DATA.PATCH_SIZE, overlap=self.cfg.DATA.TEST.OVERLAP, 
                    padding=self.cfg.DATA.TEST.PADDING, verbose=self.cfg.TEST.VERBOSE)
            else:
                self._X = crop_3D_data_with_overlap(self._X[0], self.cfg.DATA.PATCH_SIZE, overlap=self.cfg.DATA.TEST.OVERLAP, 
                    padding=self.cfg.DATA.TEST.PADDING, verbose=self.cfg.TEST.VERBOSE, 
                    median_padding=self.cfg.DATA.TEST.MEDIAN_PADDING)

        # Predict each patch
        if self.cfg.TEST.AUGMENTATION:
            for k in tqdm(range(self._X.shape[0]), leave=False):
                if self.cfg.PROBLEM.NDIM == '2D':
                    p = ensemble8_2d_predictions(self._X[k], n_classes=self.cfg.MODEL.N_CLASSES,
                            pred_func=(
                                lambda img_batch_subdiv: 
                                    to_numpy_format(
                                        self.apply_model_activations(
                                            self.model(to_pytorch_format(img_batch_subdiv, self.axis_order, self.device)),
                                            ), 
                                        self.axis_order_back
                                    )
                            )
                        )
                else:
                    p = ensemble16_3d_predictions(self._X[k], batch_size_value=self.cfg.TRAIN.BATCH_SIZE,
                            pred_func=(
                                lambda img_batch_subdiv: 
                                    to_numpy_format(
                                        self.apply_model_activations(
                                            self.model(to_pytorch_format(img_batch_subdiv, self.axis_order, self.device)),
                                            ), 
                                        self.axis_order_back
                                    )
                            )
                        )
                if 'pred' not in locals():
                    pred = np.zeros((self._X.shape[0],)+p.shape, dtype=self.dtype)
                pred[k] = p
        else:
            l = int(math.ceil(self._X.shape[0]/self.cfg.TRAIN.BATCH_SIZE))
            for k in tqdm(range(l), leave=False):
                top = (k+1)*self.cfg.TRAIN.BATCH_SIZE if (k+1)*self.cfg.TRAIN.BATCH_SIZE < self._X.shape[0] else self._X.shape[0]                
                with torch.cuda.amp.autocast():
                    p = self.model(to_pytorch_format(self._X[k*self.cfg.TRAIN.BATCH_SIZE:top], self.axis_order, self.device))
                    if self.cfg.PROBLEM.SELF_SUPERVISED.PRETEXT_TASK == "masking":
                        loss, p, mask = p
                        p = self.apply_model_activations(p)
                        p = self.model.save_images(to_pytorch_format(self._X[k*self.cfg.TRAIN.BATCH_SIZE:top], self.axis_order, self.device), 
                            p, mask, self.cfg.PATHS.MAE_OUT_DIR, filenames[0], k*self.cfg.TRAIN.BATCH_SIZE, self.dtype)
                    else:
                        p = self.apply_model_activations(p)
                        p = to_numpy_format(p, self.axis_order_back)
                    
                if 'pred' not in locals():
                    pred = np.zeros((self._X.shape[0],)+p.shape[1:], dtype=self.dtype)
                pred[k*self.cfg.TRAIN.BATCH_SIZE:top] = p

        # Delete self._X as in 3D there is no full image
        if self.cfg.PROBLEM.NDIM == '3D':
            del self._X, p

        # Reconstruct the predictions
        if original_data_shape[1:-1] != self.cfg.DATA.PATCH_SIZE[:-1]:
            if self.cfg.PROBLEM.NDIM == '3D': original_data_shape = original_data_shape[1:]
            f_name = merge_data_with_overlap if self.cfg.PROBLEM.NDIM == '2D' else merge_3D_data_with_overlap
            pred = f_name(pred, original_data_shape[:-1]+(pred.shape[-1],), padding=self.cfg.DATA.TEST.PADDING, 
                overlap=self.cfg.DATA.TEST.OVERLAP, verbose=self.cfg.TEST.VERBOSE)
        else:
            pred = pred[0]

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
            save_tif(np.expand_dims(pred,0), self.cfg.PATHS.RESULT_DIR.PER_IMAGE, filenames, verbose=self.cfg.TEST.VERBOSE)

    def after_merge_patches(self, pred,filenames):
        """
        Steps need to be done after merging all predicted patches into the original image.

        Parameters
        ----------
        pred : Torch Tensor
            Model prediction.

        filenames : List of str
            Filenames of the predicted images.  
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

    def after_full_image(self, pred, filenames):
        """
        Steps that must be executed after generating the prediction by supplying the entire image to the model.

        Parameters
        ----------
        pred : Torch Tensor
            Model prediction.

        filenames : List of str
            Filenames of the predicted images.  
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


    def prepare_ssl_data(self):
        """
        Creates self supervised "ground truth" images, if ``crappify`` was selected, to train the model based 
        on the input images provided. They will be saved in a separate folder in the root path of the inout images. 
        """
        if self.cfg.PROBLEM.SELF_SUPERVISED.PRETEXT_TASK == "masking":
            print("No SSL data needs to be prepared for masking, as it will be generated on the fly")
            return

        print("############################")
        print("#  PREPARE DETECTION DATA  #")
        print("############################")

        # Create selected channels for train data
        if self.cfg.TRAIN.ENABLE:
            create_mask = False
            if not os.path.isdir(self.cfg.DATA.TRAIN.SSL_SOURCE_DIR):
                print("You select to create detection masks from given .csv files but no file is detected in {}. "
                    "So let's prepare the data. Notice that, if you do not modify 'DATA.TRAIN.SSL_SOURCE_DIR' "
                    "path, this process will be done just once!".format(self.cfg.DATA.TRAIN.SSL_SOURCE_DIR))
                create_mask = True
            else:
                if len(next(os.walk(self.cfg.DATA.TRAIN.SSL_SOURCE_DIR))[2]) != len(next(os.walk(self.cfg.DATA.TRAIN.PATH))[2]):
                    print("Different number of files found in {} and {}. Trying to create the the rest again"
                        .format(self.cfg.DATA.TRAIN.GT_PATH, self.cfg.DATA.TRAIN.SSL_SOURCE_DIR))
                    create_mask = True 
                else:
                    print("Train source data found in {}".format(self.cfg.DATA.TRAIN.SSL_SOURCE_DIR))   
            if create_mask:
                create_ssl_source_data_masks(self.cfg, data_type='train')

        # Create selected channels for val data
        if self.cfg.TRAIN.ENABLE and not self.cfg.DATA.VAL.FROM_TRAIN:
            create_mask = False
            if not os.path.isdir(self.cfg.DATA.VAL.SSL_SOURCE_DIR):
                print("You select to create detection masks from given .csv files but no file is detected in {}. "
                    "So let's prepare the data. Notice that, if you do not modify 'DATA.VAL.SSL_SOURCE_DIR' "
                    "path, this process will be done just once!".format(self.cfg.DATA.VAL.SSL_SOURCE_DIR))
                create_mask = True
            else:
                if len(next(os.walk(self.cfg.DATA.VAL.SSL_SOURCE_DIR))[2]) != len(next(os.walk(self.cfg.DATA.VAL.PATH))[2]):
                    print("Different number of files found in {} and {}. Trying to create the the rest again"
                        .format(self.cfg.DATA.VAL.GT_PATH, self.cfg.DATA.VAL.SSL_SOURCE_DIR))
                    create_mask = True   
                else:
                    print("Validation source data found in {}".format(self.cfg.DATA.VAL.SSL_SOURCE_DIR)) 
            if create_mask:         
                create_ssl_source_data_masks(self.cfg, data_type='val')

        # Create selected channels for test data
        if self.cfg.TEST.ENABLE:
            create_mask = False
            if not os.path.isdir(self.cfg.DATA.TEST.SSL_SOURCE_DIR):
                print("You select to create detection masks from given .csv files but no file is detected in {}. "
                    "So let's prepare the data. Notice that, if you do not modify 'DATA.TEST.SSL_SOURCE_DIR' "
                    "path, this process will be done just once!".format(self.cfg.DATA.TEST.SSL_SOURCE_DIR))
                create_mask = True
            else:
                if len(next(os.walk(self.cfg.DATA.TEST.SSL_SOURCE_DIR))[2]) != len(next(os.walk(self.cfg.DATA.TEST.PATH))[2]):
                    print("Different number of files found in {} and {}. Trying to create the the rest again"
                        .format(self.cfg.DATA.TEST.GT_PATH, self.cfg.DATA.TEST.SSL_SOURCE_DIR))
                    create_mask = True    
                else:
                    print("Test source data found in {}".format(self.cfg.DATA.TEST.SSL_SOURCE_DIR))
            if create_mask:
                create_ssl_source_data_masks(self.cfg, data_type='test')

        opts = []
        if self.cfg.TRAIN.ENABLE:
            print("DATA.TRAIN.PATH changed from {} to {}".format(self.cfg.DATA.TRAIN.PATH, self.cfg.DATA.TRAIN.SSL_SOURCE_DIR))
            print("DATA.TRAIN.GT_PATH changed from {} to {}".format(self.cfg.DATA.TRAIN.GT_PATH, self.cfg.DATA.TRAIN.PATH))
            opts.extend(['DATA.TRAIN.PATH', self.cfg.DATA.TRAIN.SSL_SOURCE_DIR, 'DATA.TRAIN.GT_PATH', self.cfg.DATA.TRAIN.PATH])
            if not self.cfg.DATA.VAL.FROM_TRAIN:
                print("DATA.VAL.PATH changed from {} to {}".format(self.cfg.DATA.VAL.PATH, self.cfg.DATA.VAL.SSL_SOURCE_DIR))
                print("DATA.VAL.GT_PATH changed from {} to {}".format(self.cfg.DATA.VAL.GT_PATH, self.cfg.DATA.VAL.PATH))
                opts.extend(['DATA.VAL.PATH', self.cfg.DATA.VAL.SSL_SOURCE_DIR, 'DATA.VAL.GT_PATH', self.cfg.DATA.VAL.PATH])
        if self.cfg.TEST.ENABLE:
            print("DATA.TEST.PATH changed from {} to {}".format(self.cfg.DATA.TEST.PATH, self.cfg.DATA.TEST.SSL_SOURCE_DIR))
            print("DATA.TEST.GT_PATH changed from {} to {}".format(self.cfg.DATA.TEST.GT_PATH, self.cfg.DATA.TEST.PATH))
            opts.extend(['DATA.TEST.PATH', self.cfg.DATA.TEST.SSL_SOURCE_DIR, 'DATA.TEST.GT_PATH', self.cfg.DATA.TEST.PATH]) 
        self.cfg.merge_from_list(opts)
