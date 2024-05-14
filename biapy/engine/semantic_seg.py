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

class Semantic_Segmentation_Workflow(Base_Workflow):
    """
    Semantic segmentation workflow where the goal is to assign a class to each pixel of the input image. 
    More details in `our documentation <https://biapy.readthedocs.io/en/latest/workflows/semantic_segmentation.html>`_.  

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
        super(Semantic_Segmentation_Workflow, self).__init__(cfg, job_identifier, device, args, **kwargs)

        if cfg.TRAIN.ENABLE and cfg.DATA.TRAIN.CHECK_DATA:
            if cfg.LOSS.TYPE == 'MASKED_BCE':
                check_masks(cfg.DATA.TRAIN.GT_PATH, n_classes=3)
            else:
                check_masks(cfg.DATA.TRAIN.GT_PATH, n_classes=cfg.MODEL.N_CLASSES+1)
            if not cfg.DATA.VAL.FROM_TRAIN:
                if cfg.LOSS.TYPE == 'MASKED_BCE':
                    check_masks(cfg.DATA.VAL.GT_PATH, n_classes=3)
                else:
                    check_masks(cfg.DATA.VAL.GT_PATH, n_classes=cfg.MODEL.N_CLASSES+1)
        if cfg.TEST.ENABLE and cfg.DATA.TEST.LOAD_GT and cfg.DATA.TEST.CHECK_DATA:
            if cfg.LOSS.TYPE == 'MASKED_BCE':
                check_masks(cfg.DATA.TEST.GT_PATH, n_classes=3)
            else:
                check_masks(cfg.DATA.TEST.GT_PATH, n_classes=cfg.MODEL.N_CLASSES+1)

        # From now on, no modification of the cfg will be allowed
        self.cfg.freeze()

        # Activations for each output channel:
        # channel number : 'activation'
        self.activations = [{':': 'CE_Sigmoid'}]

        # Workflow specific training variables
        self.mask_path = cfg.DATA.TRAIN.GT_PATH
        self.load_Y_val = True
        self.loss_dtype = torch.float32
        
    def define_metrics(self):
        """
        Definition of self.metrics, self.metric_names and self.loss variables.
        """
        self.metrics = [
            jaccard_index(num_classes=self.cfg.MODEL.N_CLASSES, device=self.device, 
                torchvision_models=True if self.cfg.MODEL.SOURCE == "torchvision" else False)
        ]
        self.metric_names = ["jaccard_index"]
        if self.cfg.LOSS.TYPE == "CE":    
            self.loss = CrossEntropyLoss_wrapper(num_classes=self.cfg.MODEL.N_CLASSES,
                torchvision_models=True if self.cfg.MODEL.SOURCE == "torchvision" else False)
        elif self.cfg.LOSS.TYPE == "W_CE_DICE":
            self.loss = weighted_bce_dice_loss(w_dice=0.66, w_bce=0.33)

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
            # Data channel check
            if self.cfg.DATA.PATCH_SIZE[-1] != self._X.shape[-1]:
                raise ValueError("Channel of the DATA.PATCH_SIZE given {} does not correspond with the loaded image {}. "
                    "Please, check the channels of the images!".format(self.cfg.DATA.PATCH_SIZE[-1], self._X.shape[-1]))

            ##################
            ### FULL IMAGE ###
            ##################
            if self.cfg.TEST.FULL_IMG:
                resized_Y = False
                # Evaluate each img
                if self.cfg.DATA.TEST.LOAD_GT:
                    with torch.cuda.amp.autocast():
                        output = self.model_call_func(self._X)

                        # Resize target if it was done due to model restrictions (applied with TorchVision preprocessing provided)
                        if output.shape != self._Y.shape:
                            self._Y = self._Y.transpose((self.axis_order))
                            s = list(output.shape)
                            s[1] = self._Y.shape[1]
                            self._Y = resize(self._Y, s, order=0)
                            self._Y = self._Y.transpose((self.axis_order_back))
                            resized_Y = True

                        loss = self.loss(output, to_pytorch_format(self._Y, self.axis_order, self.device, dtype=self.loss_dtype))
                    self.stats['loss'] += loss.item()
                    del output

                # Make the prediction
                with torch.cuda.amp.autocast():
                    pred = self.model_call_func(self._X)
                pred = to_numpy_format(pred, self.axis_order_back)
                del self._X 

                if self.cfg.TEST.POST_PROCESSING.APPLY_MASK:
                    pred = apply_binary_mask(pred, self.cfg.DATA.TEST.BINARY_MASKS)
                    
                if self.cfg.DATA.TEST.LOAD_GT:
                    if not resized_Y and pred.shape != self._Y.shape:
                        self._Y = resize(self._Y, pred.shape, order=0)
                    score = jaccard_index_numpy((self._Y>0.5).astype(np.uint8), (pred>0.5).astype(np.uint8))
                    self.stats['iou'] += score
                    self.stats['ov_iou'] += voc_calculation((self._Y>0.5).astype(np.uint8), (pred>0.5).astype(np.uint8), score)

                
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
        # Convert first to 0-255 range if uint16
        if in_img.dtype == torch.float32:
            if torch.max(in_img) > 1:
                in_img = (norm_range01(in_img, torch.uint8)[0]*255).to(torch.uint8)
            in_img = in_img.to(torch.uint8)
        
        # Apply TorchVision pre-processing
        in_img = self.torchvision_preprocessing(in_img)

        pred = self.model(in_img)
        key = "aux" if "aux" in pred else "out"
        
        # Save masks
        if not is_train:
            masks = np.expand_dims(np.argmax(pred[key].cpu().numpy().transpose(0,2,3,1), axis=-1),-1)
            save_tif(masks, self.cfg.PATHS.RESULT_DIR.FULL_IMAGE, self.processing_filenames, 
                verbose=self.cfg.TEST.VERBOSE)    

        return pred[key]
        
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
            train_iou = self.metrics[0](output, targets)
            train_iou = train_iou.item() if not torch.isnan(train_iou) else 0
            if metric_logger is not None:
                metric_logger.meters[self.metric_names[0]].update(train_iou)
            else:
                return train_iou

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
        return to_pytorch_format(targets, self.axis_order, self.device, dtype=self.loss_dtype)

    def after_merge_patches(self, pred):
        """
        Steps need to be done after merging all predicted patches into the original image.

        Parameters
        ----------
        pred : Torch Tensor
            Model prediction.
        """
        # Save simple binarization of predictions
        if self.cfg.MODEL.N_CLASSES > 2:
            _type = np.uint8 if self.cfg.MODEL.N_CLASSES < 255 else np.uint16
            pred = np.expand_dims(np.argmax(pred, axis=-1),-1).astype(_type)        
        else:
            pred = (pred>0.5).astype(np.uint8)        
        save_tif(pred, self.cfg.PATHS.RESULT_DIR.PER_IMAGE_BIN, self.processing_filenames, verbose=self.cfg.TEST.VERBOSE)

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
        # Save simple binarization of predictions
        save_tif((pred>0.5).astype(np.uint8), self.cfg.PATHS.RESULT_DIR.FULL_IMAGE_BIN, self.processing_filenames,
            verbose=self.cfg.TEST.VERBOSE)

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
        super().normalize_stats(image_counter)

    def print_stats(self, image_counter):
        """
        Print statistics.  

        Parameters
        ----------
        image_counter : int
            Number of images to call ``normalize_stats``.
        """
        super().print_stats(image_counter)
        super().print_post_processing_stats()


        