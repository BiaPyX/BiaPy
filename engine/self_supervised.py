import math
import numpy as np
from tqdm import tqdm
from skimage.transform import resize

from data.data_2D_manipulation import crop_data_with_overlap, merge_data_with_overlap
from data.data_3D_manipulation import crop_3D_data_with_overlap, merge_3D_data_with_overlap
from data.post_processing.post_processing import ensemble8_2d_predictions, ensemble16_3d_predictions
from utils.util import denormalize, save_tif
from engine.base_workflow import Base_Workflow
from engine.metrics import PSNR

class Self_supervised(Base_Workflow):
    def __init__(self, cfg, model, post_processing=False):
        super().__init__(cfg, model, post_processing)
        self.stats['psnr_per_image'] = 0

    def process_sample(self, X, Y, filenames, norm): 
        original_data_shape = X.shape
    
        # Crop if necessary
        if X.shape[1:-1] != self.cfg.DATA.PATCH_SIZE[:-1]:
            if self.cfg.PROBLEM.NDIM == '2D':
                obj = crop_data_with_overlap(X, self.cfg.DATA.PATCH_SIZE, data_mask=Y, overlap=self.cfg.DATA.TEST.OVERLAP, 
                    padding=self.cfg.DATA.TEST.PADDING, verbose=self.cfg.TEST.VERBOSE)
                if self.cfg.DATA.TEST.LOAD_GT:
                    X, Y = obj
                else:
                    X = obj
                del obj
            else:
                if self.cfg.DATA.TEST.LOAD_GT: Y = Y[0]
                if self.cfg.TEST.REDUCE_MEMORY:
                    X = crop_3D_data_with_overlap(X[0], self.cfg.DATA.PATCH_SIZE, overlap=self.cfg.DATA.TEST.OVERLAP, 
                        padding=self.cfg.DATA.TEST.PADDING, verbose=self.cfg.TEST.VERBOSE, 
                        median_padding=self.cfg.DATA.TEST.MEDIAN_PADDING)
                    Y = crop_3D_data_with_overlap(Y, self.cfg.DATA.PATCH_SIZE, overlap=self.cfg.DATA.TEST.OVERLAP, 
                        padding=self.cfg.DATA.TEST.PADDING, verbose=self.cfg.TEST.VERBOSE, 
                        median_padding=self.cfg.DATA.TEST.MEDIAN_PADDING)
                else:
                    obj = crop_3D_data_with_overlap(X[0], self.cfg.DATA.PATCH_SIZE, data_mask=Y, overlap=self.cfg.DATA.TEST.OVERLAP, 
                        padding=self.cfg.DATA.TEST.PADDING, verbose=self.cfg.TEST.VERBOSE, 
                        median_padding=self.cfg.DATA.TEST.MEDIAN_PADDING)
                    if self.cfg.DATA.TEST.LOAD_GT:
                        X, Y = obj
                    else:
                        X = obj
                    del obj

        # Predict each patch
        pred = []
        if self.cfg.TEST.AUGMENTATION:
            for k in tqdm(range(X.shape[0]), leave=False):
                if self.cfg.PROBLEM.NDIM == '2D':
                    p = ensemble8_2d_predictions(X[k], n_classes=self.cfg.MODEL.N_CLASSES,
                            pred_func=(lambda img_batch_subdiv: self.model.predict(img_batch_subdiv)))
                else:
                    p = ensemble16_3d_predictions(X[k], batch_size_value=1,
                            pred_func=(lambda img_batch_subdiv: self.model.predict(img_batch_subdiv)))
                pred.append(p)
        else:
            l = int(math.ceil(X.shape[0]/self.cfg.TRAIN.BATCH_SIZE))
            for k in tqdm(range(l), leave=False):
                top = (k+1)*self.cfg.TRAIN.BATCH_SIZE if (k+1)*self.cfg.TRAIN.BATCH_SIZE < X.shape[0] else X.shape[0]
                p = self.model.predict(X[k*self.cfg.TRAIN.BATCH_SIZE:top], verbose=0)
                pred.append(p)
        del X, p

        # Reconstruct the predictions
        pred = np.concatenate(pred)
        if original_data_shape[1:-1] != self.cfg.DATA.PATCH_SIZE[:-1]:
            if self.cfg.PROBLEM.NDIM == '3D': original_data_shape = original_data_shape[1:]
            f_name = merge_data_with_overlap if self.cfg.PROBLEM.NDIM == '2D' else merge_3D_data_with_overlap

            if self.cfg.TEST.REDUCE_MEMORY:
                pred = f_name(pred, original_data_shape[:-1]+(pred.shape[-1],), padding=self.cfg.DATA.TEST.PADDING, 
                    overlap=self.cfg.DATA.TEST.OVERLAP, verbose=self.cfg.TEST.VERBOSE)
                Y = f_name(Y, original_data_shape[:-1]+(Y.shape[-1],), padding=self.cfg.DATA.TEST.PADDING, 
                    overlap=self.cfg.DATA.TEST.OVERLAP, verbose=self.cfg.TEST.VERBOSE)
            else:
                obj = f_name(pred, original_data_shape[:-1]+(pred.shape[-1],), data_mask=Y,
                    padding=self.cfg.DATA.TEST.PADDING, overlap=self.cfg.DATA.TEST.OVERLAP,
                    verbose=self.cfg.TEST.VERBOSE)
                if self.cfg.DATA.TEST.LOAD_GT:
                    pred, Y = obj
                else:
                    pred = obj
                del obj
        else:
            pred = pred[0]

        # Undo normalization
        x_norm = norm[0]
        if x_norm['type'] == 'div':
            pred = pred*255
        else:
            pred = denormalize(pred, x_norm['mean'], x_norm['std'])  
            
        # Save image
        if self.cfg.PATHS.RESULT_DIR.PER_IMAGE != "":
            save_tif(np.expand_dims(pred,0), self.cfg.PATHS.RESULT_DIR.PER_IMAGE, filenames, verbose=self.cfg.TEST.VERBOSE)
    
        # Calculate PSNR
        if self.cfg.DATA.TEST.LOAD_GT:
            psnr_per_image = PSNR(pred, Y)
            self.stats['psnr_per_image'] += psnr_per_image

    def after_merge_patches(self, pred, Y, filenames):
        pass

    def after_full_image(self, pred, Y, filenames):
        pass

    def after_all_images(self, Y):
        pass
    
    def normalize_stats(self, image_counter):
        self.stats['psnr_per_image'] = self.stats['psnr_per_image'] / image_counter

    def print_stats(self, image_counter):
        self.normalize_stats(image_counter)

        if self.cfg.DATA.TEST.LOAD_GT:
            print("Test PSNR (merge patches): {}".format(self.stats['psnr_per_image']))
            print(" ")


def crappify(data, resizing_factor, add_noise=True, noise_level=None, Down_up=True):
    """
    Crappifies input data by adding Gaussian noise and downsampling and upsampling it so the resolution
    gets worsen. 

    data : 4D/5D Numpy array
        Data to be modified. E.g. ``(num_of_images, y, x, channels)`` if working with 2D images or
        ``(num_of_images, z, y, x, channels)`` if working with 3D.

    resizing_factor : floats
        Downsizing factor to divide the number of pixels with.

    add_noise : boolean, optional
        Indicating whether to add gaussian noise before applying the resizing.

    noise_level: float, optional
        Number between ``[0,1]`` indicating the std of the Gaussian noise N(0,std).

    Down_up : bool, optional
        Indicating whether to perform a final upsampling operation to obtain an image of the 
        same size as the original but with the corresponding loss of quality of downsizing and 
        upsizing.

    Returns
    -------
    data : 4D/5D Numpy array
        Same input data but normalized.

    new_data : 4D/5D Numpy array
        Train images. E.g. ``(num_of_images, y, x, channels)`` if working with 2D images or
        ``(num_of_images, z, y, x, channels)`` if working with 3D.
    """
    if data.ndim == 4:
        w, h, c = data[0].shape
        org_sz = (w, h)
    else:
        d, w, h, c = data[0].shape
        org_sz = (d, w, h)
        new_d = int(d / np.sqrt(resizing_factor))

    new_w = int(w / np.sqrt(resizing_factor))
    new_h = int(h / np.sqrt(resizing_factor))

    if data.ndim == 4:
        targ_sz = (new_w, new_h)
    else:
        targ_sz = (new_d, new_w, new_h)

    # Normalize data
    if np.max(data) > 100:
        print("WARNING: in SELF_SUPERVISED data is normalized between [0,1] even if custom normalization was selected!")
        data = data/255        

    new_data = []
    for i in tqdm(range(data.shape[0]), leave=False):
        img = data[i]
        if add_noise:
            img = add_gaussian_noise(img, noise_level)

        img = resize(img, targ_sz, order=1, mode='reflect',
                     clip=True, preserve_range=True, anti_aliasing=False)

        if Down_up:
            img = resize(img, org_sz, order=1, mode='reflect',
                         clip=True, preserve_range=True, anti_aliasing=False)
        new_data.append(img)

    return data, np.array(new_data)

def add_gaussian_noise(image, percentage_of_noise):
    """
    Adds Gaussian noise to an input image. 

    Parameters
    ----------
    image : 3D Numpy array
        Image to be added Gaussian Noise with 0 mean and a certain std. E.g. ``(y, x, channels)``.

    percentage_of_noise : float
        percentage of the maximum value of the image that will be used as the std of the Gaussian Noise 
        distribution.

    Returns
    -------
    out : 3D Numpy array
        Transformed image. E.g. ``(y, x, channels)``.
    """
    max_value=np.max(image)
    noise_level=percentage_of_noise*max_value
    noise = np.random.normal(loc=0, scale=noise_level, size=image.shape)
    noisy_img=np.clip(image+noise, 0, max_value) 
    return noisy_img
