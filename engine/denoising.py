import math
import numpy as np
import numpy.ma as ma
from tqdm import tqdm

from data.data_2D_manipulation import crop_data_with_overlap, merge_data_with_overlap
from data.data_3D_manipulation import crop_3D_data_with_overlap, merge_3D_data_with_overlap
from data.post_processing.post_processing import ensemble8_2d_predictions, ensemble16_3d_predictions
from engine.base_workflow import Base_Workflow
from utils.util import save_tif
from data.pre_processing import denormalize, undo_norm_range01


class Denoising(Base_Workflow):
    def __init__(self, cfg, model, post_processing={}):
        super().__init__(cfg, model, post_processing)
    
    def process_sample(self, X, Y, filenames, f_numbers, norm): 
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
        x_norm = norm[0][0]
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

    def after_merge_patches(self, pred, Y, filenames, f_numbers):
        pass

    def after_full_image(self, pred, Y, filenames):
        pass

    def after_all_images(self):
        super().after_all_images()

    def normalize_stats(self, image_counter):
        return

    def print_stats(self, image_counter):
        self.normalize_stats(image_counter)


####################################
# Adapted from N2V code:           #
#   https://github.com/juglab/n2v  #
####################################

def get_subpatch(patch, coord, local_sub_patch_radius, crop_patch=True):
    crop_neg, crop_pos = 0, 0
    if crop_patch:
        start = np.array(coord) - local_sub_patch_radius
        end = start + local_sub_patch_radius * 2 + 1

        # compute offsets left/up ...
        crop_neg = np.minimum(start, 0)
        # and right/down
        crop_pos = np.maximum(0, end-patch.shape)

        # correct for offsets, patch size shrinks if crop_*!=0
        start -= crop_neg
        end -= crop_pos
    else:
        start = np.maximum(0, np.array(coord) - local_sub_patch_radius)
        end = start + local_sub_patch_radius * 2 + 1

        shift = np.minimum(0, patch.shape - end)

        start += shift
        end += shift

    slices = [slice(s, e) for s, e in zip(start, end)]

    # return crop vectors for deriving correct center pixel locations later
    return patch[tuple(slices)], crop_neg, crop_pos


def random_neighbor(shape, coord):
    rand_coords = sample_coords(shape, coord)
    while np.any(rand_coords == coord):
        rand_coords = sample_coords(shape, coord)

    return rand_coords


def sample_coords(shape, coord, sigma=4):
    return [normal_int(c, sigma, s) for c, s in zip(coord, shape)]


def normal_int(mean, sigma, w):
    return int(np.clip(np.round(np.random.normal(mean, sigma)), 0, w - 1))


def mask_center(local_sub_patch_radius, ndims=2):
    size = local_sub_patch_radius*2 + 1
    patch_wo_center = np.ones((size, ) * ndims)
    if ndims == 2:
        patch_wo_center[local_sub_patch_radius, local_sub_patch_radius] = 0
    elif ndims == 3:
        patch_wo_center[local_sub_patch_radius,
        local_sub_patch_radius, local_sub_patch_radius] = 0
    else:
        raise NotImplementedError()
    return ma.make_mask(patch_wo_center)


def pm_normal_withoutCP(local_sub_patch_radius):
    def normal_withoutCP(patch, coords, dims, structN2Vmask=None):
        vals = []
        for coord in zip(*coords):
            rand_coords = random_neighbor(patch.shape, coord)
            vals.append(patch[tuple(rand_coords)])
        return vals

    return normal_withoutCP


def pm_mean(local_sub_patch_radius):
    def patch_mean(patch, coords, dims, structN2Vmask=None):
        patch_wo_center = mask_center(local_sub_patch_radius, ndims=dims)
        vals = []
        for coord in zip(*coords):
            sub_patch, crop_neg, crop_pos = get_subpatch(patch, coord, local_sub_patch_radius)
            slices = [slice(-n, s-p) for n, p, s in zip(crop_neg, crop_pos, patch_wo_center.shape)]
            sub_patch_mask = (structN2Vmask or patch_wo_center)[tuple(slices)]
            vals.append(np.mean(sub_patch[sub_patch_mask]))
        return vals

    return patch_mean


def pm_median(local_sub_patch_radius):
    def patch_median(patch, coords, dims, structN2Vmask=None):
        patch_wo_center = mask_center(local_sub_patch_radius, ndims=dims)
        vals = []
        for coord in zip(*coords):
            sub_patch, crop_neg, crop_pos = get_subpatch(patch, coord, local_sub_patch_radius)
            slices = [slice(-n, s-p) for n, p, s in zip(crop_neg, crop_pos, patch_wo_center.shape)]
            sub_patch_mask = (structN2Vmask or patch_wo_center)[tuple(slices)]
            vals.append(np.median(sub_patch[sub_patch_mask]))
        return vals

    return patch_median


def pm_uniform_withCP(local_sub_patch_radius):
    def random_neighbor_withCP_uniform(patch, coords, dims, structN2Vmask=None):
        vals = []
        for coord in zip(*coords):
            sub_patch, _, _ = get_subpatch(patch, coord, local_sub_patch_radius)
            rand_coords = [np.random.randint(0, s) for s in sub_patch.shape[0:dims]]
            vals.append(sub_patch[tuple(rand_coords)])
        return vals

    return random_neighbor_withCP_uniform


def pm_uniform_withoutCP(local_sub_patch_radius):
    def random_neighbor_withoutCP_uniform(patch, coords, dims, structN2Vmask=None):
        patch_wo_center = mask_center(local_sub_patch_radius, ndims=dims)
        vals = []
        for coord in zip(*coords):
            sub_patch, crop_neg, crop_pos = get_subpatch(patch, coord, local_sub_patch_radius)
            slices = [slice(-n, s-p) for n, p, s in zip(crop_neg, crop_pos, patch_wo_center.shape)]
            sub_patch_mask = (structN2Vmask or patch_wo_center)[tuple(slices)]
            vals.append(np.random.permutation(sub_patch[sub_patch_mask])[0])
        return vals

    return random_neighbor_withoutCP_uniform


def pm_normal_additive(pixel_gauss_sigma):
    def pixel_gauss(patch, coords, dims, structN2Vmask=None):
        vals = []
        for coord in zip(*coords):
            vals.append(np.random.normal(patch[tuple(coord)], pixel_gauss_sigma))
        return vals

    return pixel_gauss


def pm_normal_fitted(local_sub_patch_radius):
    def local_gaussian(patch, coords, dims, structN2Vmask=None):
        vals = []
        for coord in zip(*coords):
            sub_patch, _, _ = get_subpatch(patch, coord, local_sub_patch_radius)
            axis = tuple(range(dims))
            vals.append(np.random.normal(np.mean(sub_patch, axis=axis), np.std(sub_patch, axis=axis)))
        return vals

    return local_gaussian


def pm_identity(local_sub_patch_radius):
    def identity(patch, coords, dims, structN2Vmask=None):
        vals = []
        for coord in zip(*coords):
            vals.append(patch[coord])
        return vals

    return identity


def get_stratified_coords2D(box_size, shape):
    box_count_Y = int(np.ceil(shape[0] / box_size))
    box_count_X = int(np.ceil(shape[1] / box_size))
    x_coords = []
    y_coords = []
    for i in range(box_count_Y):
        for j in range(box_count_X):
            y, x = np.random.rand() * box_size, np.random.rand() * box_size
            y = int(i * box_size + y)
            x = int(j * box_size + x)
            if (y < shape[0] and x < shape[1]):
                y_coords.append(y)
                x_coords.append(x)
    return (y_coords, x_coords)

def get_stratified_coords3D(box_size, shape):
        box_count_z = int(np.ceil(shape[0] / box_size))
        box_count_Y = int(np.ceil(shape[1] / box_size))
        box_count_X = int(np.ceil(shape[2] / box_size))
        x_coords = []
        y_coords = []
        z_coords = []
        for i in range(box_count_z):
            for j in range(box_count_Y):
                for k in range(box_count_X):
                    z, y, x = np.random.rand() * box_size, np.random.rand() * box_size, np.random.rand() * box_size
                    z = int(i * box_size + z)
                    y = int(j * box_size + y)
                    x = int(k * box_size + x)
                    if (z < shape[0] and y < shape[1] and x < shape[2]):
                        z_coords.append(z)
                        y_coords.append(y)
                        x_coords.append(x)
        return (z_coords, y_coords, x_coords)

def apply_structN2Vmask(patch, coords, mask):
    """
    each point in coords corresponds to the center of the mask.
    then for point in the mask with value=1 we assign a random value
    """
    coords = np.array(coords).astype(np.int)
    ndim = mask.ndim
    center = np.array(mask.shape)//2
    ## leave the center value alone
    mask[tuple(center.T)] = 0
    ## displacements from center
    dx = np.indices(mask.shape)[:,mask==1] - center[:,None]
    ## combine all coords (ndim, npts,) with all displacements (ncoords,ndim,)
    mix = (dx.T[...,None] + coords[None])
    mix = mix.transpose([1,0,2]).reshape([ndim,-1]).T
    ## stay within patch boundary
    mix = mix.clip(min=np.zeros(ndim),max=np.array(patch.shape)-1).astype(np.uint)
    ## replace neighbouring pixels with random values from flat dist
    patch[tuple(mix.T)] = np.random.rand(mix.shape[0])*4 - 2

def apply_structN2Vmask3D(patch, coords, mask):
    """
    each point in coords corresponds to the center of the mask.
    then for point in the mask with value=1 we assign a random value
    """
    z_coords = coords[0]
    coords = coords[1:]
    for z in z_coords:
        coords = np.array(coords).astype(np.int)
        ndim = mask.ndim
        center = np.array(mask.shape)//2
        ## leave the center value alone
        mask[tuple(center.T)] = 0
        ## displacements from center
        dx = np.indices(mask.shape)[:,mask==1] - center[:,None]
        ## combine all coords (ndim, npts,) with all displacements (ncoords,ndim,)
        mix = (dx.T[...,None] + coords[None])
        mix = mix.transpose([1,0,2]).reshape([ndim,-1]).T
        ## stay within patch boundary
        mix = mix.clip(min=np.zeros(ndim),max=np.array(patch.shape[1:])-1).astype(np.uint)
        ## replace neighbouring pixels with random values from flat dist
        patch[z][tuple(mix.T)] = np.random.rand(mix.shape[0])*4 - 2

def manipulate_val_data(X_val, Y_val, perc_pix=0.198, shape=(64, 64), value_manipulation=pm_uniform_withCP(5)):
    dims = len(shape)
    if dims == 2:
        box_size = np.round(np.sqrt(100 / perc_pix)).astype(np.int)
        get_stratified_coords = get_stratified_coords2D
    elif dims == 3:
        box_size = np.round(np.sqrt(100 / perc_pix)).astype(np.int)
        get_stratified_coords = get_stratified_coords3D

    n_chan = X_val.shape[-1]

    Y_val *= 0
    for j in tqdm(range(X_val.shape[0]), desc='Preparing validation data: '):
        coords = get_stratified_coords(box_size=box_size,
                                       shape=np.array(X_val.shape)[1:-1])
        for c in range(n_chan):
            indexing = (j,) + coords + (c,)
            indexing_mask = (j,) + coords + (c + n_chan,)
            y_val = X_val[indexing]
            x_val = value_manipulation(X_val[j, ..., c], coords, dims)

            Y_val[indexing] = y_val
            Y_val[indexing_mask] = 1
            X_val[indexing] = x_val

def get_value_manipulation(n2v_manipulator, n2v_neighborhood_radius):
    return eval('pm_{0}({1})'.format(n2v_manipulator, str(n2v_neighborhood_radius)))