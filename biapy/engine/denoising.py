import math
import torch
import numpy as np
import numpy.ma as ma
from tqdm import tqdm
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError

from biapy.data.data_2D_manipulation import (
    crop_data_with_overlap,
    merge_data_with_overlap,
)
from biapy.data.data_3D_manipulation import (
    crop_3D_data_with_overlap,
    merge_3D_data_with_overlap,
)
from biapy.data.post_processing.post_processing import (
    ensemble8_2d_predictions,
    ensemble16_3d_predictions,
)
from biapy.engine.base_workflow import Base_Workflow
from biapy.utils.util import save_tif, pad_and_reflect
from biapy.utils.misc import to_pytorch_format, to_numpy_format, is_main_process
from biapy.data.pre_processing import denormalize, undo_norm_range01
from biapy.engine.metrics import n2v_loss_mse


class Denoising_Workflow(Base_Workflow):
    """
    Denoising workflow where the goal is to remove noise from an image. More details in
    `our documentation <https://biapy.readthedocs.io/en/latest/workflows/denoising.html>`_.

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
        super(Denoising_Workflow, self).__init__(cfg, job_identifier, device, args, **kwargs)

        # From now on, no modification of the cfg will be allowed
        self.cfg.freeze()

        # Activations for each output channel:
        # channel number : 'activation'
        self.activations = [{":": "Linear"}]

        # Workflow specific training variables
        self.mask_path = None
        self.load_Y_val = False

    def define_metrics(self):
        """
        This function must define the following variables:

        self.train_metrics : List of functions
            Metrics to be calculated during model's training.

        self.train_metric_names : List of str
            Names of the metrics calculated during training.

        self.train_metric_best : List of str
            To know which value should be considered as the best one. Options must be: "max" or "min".

        self.test_metrics : List of functions
            Metrics to be calculated during model's test/inference.

        self.test_metric_names : List of str
            Names of the metrics calculated during test/inference.

        self.loss : Function
            Loss function used during training and test.
        """
        self.train_metrics = []
        self.train_metric_names = []
        self.train_metric_best = []
        for metric in list(set(self.cfg.TRAIN.METRICS)):
            if metric in ["mse"]:
                self.train_metrics.append(
                    MeanSquaredError().to(self.device),
                )
                self.train_metric_names.append("MSE")
                self.train_metric_best.append("min")
            elif metric == "mae":
                self.train_metrics.append(
                    MeanAbsoluteError().to(self.device),
                )
                self.train_metric_names.append("MAE")
                self.train_metric_best.append("min")

        self.test_metrics = []
        self.test_metric_names = []
        for metric in list(set(self.cfg.TEST.METRICS)):
            if metric in ["mse"]:
                self.test_metrics.append(
                    MeanSquaredError().to(self.device),
                )
                self.test_metric_names.append("MSE")
            elif metric == "mae":
                self.test_metrics.append(
                    MeanAbsoluteError().to(self.device),
                )
                self.test_metric_names.append("MAE")

        print("Overriding 'LOSS.TYPE' to set it to N2V loss (masked MSE)")
        self.loss = n2v_loss_mse

        super().define_metrics()

    def metric_calculation(self, output, targets, train=True, metric_logger=None):
        """
        Execution of the metrics defined in :func:`~define_metrics` function.

        Parameters
        ----------
        output : Torch Tensor
            Prediction of the model.

        targets : Torch Tensor
            Ground truth to compare the prediction with.

        train : bool, optional
            Whether to calculate train or test metrics.

        metric_logger : MetricLogger, optional
            Class to be updated with the new metric(s) value(s) calculated.

        Returns
        -------
        out_metrics : dict
            Value of the metrics for the given prediction.
        """
        out_metrics = {}
        list_to_use = self.train_metrics if train else self.test_metrics
        list_names_to_use = self.train_metric_names if train else self.test_metric_names

        with torch.no_grad():
            for i, metric in enumerate(list_to_use):
                val = metric(output.squeeze(), targets[:, 0].squeeze())
                val = val.item() if not torch.isnan(val) else 0
                out_metrics[list_names_to_use[i]] = val

                if metric_logger is not None:
                    metric_logger.meters[list_names_to_use[i]].update(val)
        return out_metrics

    def process_sample(self, norm):
        """
        Function to process a sample in the inference phase.

        Parameters
        ----------
        norm : List of dicts
            Normalization used during training. Required to denormalize the predictions of the model.
        """
        # Save test_output if the user wants to export the model to BMZ later
        if "test_input" not in self.bmz_config:
            if self.cfg.PROBLEM.NDIM == "2D":
                self.bmz_config["test_input"] = self._X[0][
                    : self.cfg.DATA.PATCH_SIZE[0], : self.cfg.DATA.PATCH_SIZE[1]
                ].copy()
            else:
                self.bmz_config["test_input"] = self._X[0][
                    : self.cfg.DATA.PATCH_SIZE[0], : self.cfg.DATA.PATCH_SIZE[1], : self.cfg.DATA.PATCH_SIZE[2]
                ].copy()

        # Reflect data to complete the needed shape
        if self.cfg.DATA.REFLECT_TO_COMPLETE_SHAPE:
            reflected_orig_shape = self._X.shape
            self._X = np.expand_dims(
                pad_and_reflect(self._X[0], self.cfg.DATA.PATCH_SIZE, verbose=self.cfg.TEST.VERBOSE),
                0,
            )

        original_data_shape = self._X.shape

        # Crop if necessary
        if self._X.shape[1:-1] != self.cfg.DATA.PATCH_SIZE[:-1]:
            if self.cfg.PROBLEM.NDIM == "2D":
                obj = crop_data_with_overlap(
                    self._X,
                    self.cfg.DATA.PATCH_SIZE,
                    overlap=self.cfg.DATA.TEST.OVERLAP,
                    padding=self.cfg.DATA.TEST.PADDING,
                    verbose=self.cfg.TEST.VERBOSE,
                )
                self._X = obj
                del obj
            else:
                if self.cfg.TEST.REDUCE_MEMORY:
                    self._X = crop_3D_data_with_overlap(
                        self._X[0],
                        self.cfg.DATA.PATCH_SIZE,
                        overlap=self.cfg.DATA.TEST.OVERLAP,
                        padding=self.cfg.DATA.TEST.PADDING,
                        verbose=self.cfg.TEST.VERBOSE,
                        median_padding=self.cfg.DATA.TEST.MEDIAN_PADDING,
                    )
                else:
                    obj = crop_3D_data_with_overlap(
                        self._X[0],
                        self.cfg.DATA.PATCH_SIZE,
                        overlap=self.cfg.DATA.TEST.OVERLAP,
                        padding=self.cfg.DATA.TEST.PADDING,
                        verbose=self.cfg.TEST.VERBOSE,
                        median_padding=self.cfg.DATA.TEST.MEDIAN_PADDING,
                    )
                    self._X = obj
                    del obj

        # Predict each patch
        if self.cfg.TEST.AUGMENTATION:
            for k in tqdm(range(self._X.shape[0]), leave=False, disable=not is_main_process()):
                if self.cfg.PROBLEM.NDIM == "2D":
                    p = ensemble8_2d_predictions(
                        self._X[k],
                        axis_order_back=self.axis_order_back,
                        pred_func=self.model_call_func,
                        axis_order=self.axis_order,
                        device=self.device,
                    )
                else:
                    p = ensemble16_3d_predictions(
                        self._X[k],
                        batch_size_value=self.cfg.TRAIN.BATCH_SIZE,
                        axis_order_back=self.axis_order_back,
                        pred_func=self.model_call_func,
                        axis_order=self.axis_order,
                        device=self.device,
                    )
                p = self.apply_model_activations(p)
                p = to_numpy_format(p, self.axis_order_back)
                if "pred" not in locals():
                    pred = np.zeros((self._X.shape[0],) + p.shape[1:], dtype=self.dtype)
                pred[k] = p
        else:
            self._X = to_pytorch_format(self._X, self.axis_order, self.device)
            l = int(math.ceil(self._X.shape[0] / self.cfg.TRAIN.BATCH_SIZE))
            for k in tqdm(range(l), leave=False, disable=not is_main_process()):
                top = (
                    (k + 1) * self.cfg.TRAIN.BATCH_SIZE
                    if (k + 1) * self.cfg.TRAIN.BATCH_SIZE < self._X.shape[0]
                    else self._X.shape[0]
                )
                with torch.cuda.amp.autocast():
                    p = self.model(self._X[k * self.cfg.TRAIN.BATCH_SIZE : top])
                p = to_numpy_format(self.apply_model_activations(p), self.axis_order_back)
                if "pred" not in locals():
                    pred = np.zeros((self._X.shape[0],) + p.shape[1:], dtype=self.dtype)
                pred[k * self.cfg.TRAIN.BATCH_SIZE : top] = p
        del self._X, p

        # Reconstruct the predictions
        if original_data_shape[1:-1] != self.cfg.DATA.PATCH_SIZE[:-1]:
            if self.cfg.PROBLEM.NDIM == "3D":
                original_data_shape = original_data_shape[1:]
            f_name = merge_data_with_overlap if self.cfg.PROBLEM.NDIM == "2D" else merge_3D_data_with_overlap

            if self.cfg.TEST.REDUCE_MEMORY:
                pred = f_name(
                    pred,
                    original_data_shape[:-1] + (pred.shape[-1],),
                    padding=self.cfg.DATA.TEST.PADDING,
                    overlap=self.cfg.DATA.TEST.OVERLAP,
                    verbose=self.cfg.TEST.VERBOSE,
                )
            else:
                obj = f_name(
                    pred,
                    original_data_shape[:-1] + (pred.shape[-1],),
                    padding=self.cfg.DATA.TEST.PADDING,
                    overlap=self.cfg.DATA.TEST.OVERLAP,
                    verbose=self.cfg.TEST.VERBOSE,
                )
                pred = obj
                del obj

            if self.cfg.PROBLEM.NDIM == "3D":
                pred = np.expand_dims(pred, 0)

        if self.cfg.DATA.REFLECT_TO_COMPLETE_SHAPE:
            if self.cfg.PROBLEM.NDIM == "2D":
                pred = pred[:, -reflected_orig_shape[1] :, -reflected_orig_shape[2] :]
            else:
                pred = pred[
                    :,
                    -reflected_orig_shape[1] :,
                    -reflected_orig_shape[2] :,
                    -reflected_orig_shape[3] :,
                ]

        # Undo normalization
        x_norm = norm[0]
        if x_norm["type"] == "div":
            pred = undo_norm_range01(pred, x_norm)
        elif x_norm["type"] == "scale_range":
            pred = undo_norm_range01(pred, x_norm, x_norm["min_val_scale"], x_norm["max_val_scale"])
        else:
            pred = denormalize(pred, x_norm["mean"], x_norm["std"])

            if x_norm["orig_dtype"] not in [
                np.dtype("float64"),
                np.dtype("float32"),
                np.dtype("float16"),
            ]:
                pred = np.round(pred)
                minpred = np.min(pred)
                pred = pred + abs(minpred)

            pred = pred.astype(x_norm["orig_dtype"])

        # Save image
        if self.cfg.PATHS.RESULT_DIR.PER_IMAGE != "":
            save_tif(
                pred,
                self.cfg.PATHS.RESULT_DIR.PER_IMAGE,
                self.processing_filenames,
                verbose=self.cfg.TEST.VERBOSE,
            )

        # Save test_output if the user wants to export the model to BMZ later
        if "test_output" not in self.bmz_config:
            if self.cfg.PROBLEM.NDIM == "2D":
                self.bmz_config["test_output"] = pred[0][
                    : self.cfg.DATA.PATCH_SIZE[0], : self.cfg.DATA.PATCH_SIZE[1]
                ].copy()
            else:
                self.bmz_config["test_output"] = pred[0][
                    : self.cfg.DATA.PATCH_SIZE[0], : self.cfg.DATA.PATCH_SIZE[1], : self.cfg.DATA.PATCH_SIZE[2]
                ].copy()

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
        crop_pos = np.maximum(0, end - patch.shape)

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
    size = local_sub_patch_radius * 2 + 1
    patch_wo_center = np.ones((size,) * ndims)
    if ndims == 2:
        patch_wo_center[local_sub_patch_radius, local_sub_patch_radius] = 0
    elif ndims == 3:
        patch_wo_center[local_sub_patch_radius, local_sub_patch_radius, local_sub_patch_radius] = 0
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
            slices = [slice(-n, s - p) for n, p, s in zip(crop_neg, crop_pos, patch_wo_center.shape)]
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
            slices = [slice(-n, s - p) for n, p, s in zip(crop_neg, crop_pos, patch_wo_center.shape)]
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
            slices = [slice(-n, s - p) for n, p, s in zip(crop_neg, crop_pos, patch_wo_center.shape)]
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
            if y < shape[0] and x < shape[1]:
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
                z, y, x = (
                    np.random.rand() * box_size,
                    np.random.rand() * box_size,
                    np.random.rand() * box_size,
                )
                z = int(i * box_size + z)
                y = int(j * box_size + y)
                x = int(k * box_size + x)
                if z < shape[0] and y < shape[1] and x < shape[2]:
                    z_coords.append(z)
                    y_coords.append(y)
                    x_coords.append(x)
    return (z_coords, y_coords, x_coords)


def apply_structN2Vmask(patch, coords, mask):
    """
    each point in coords corresponds to the center of the mask.
    then for point in the mask with value=1 we assign a random value
    """
    coords = np.array(coords, dtype=int)
    ndim = mask.ndim
    center = np.array(mask.shape) // 2
    ## leave the center value alone
    mask[tuple(center.T)] = 0
    ## displacements from center
    dx = np.indices(mask.shape)[:, mask == 1] - center[:, None]
    ## combine all coords (ndim, npts,) with all displacements (ncoords,ndim,)
    mix = dx.T[..., None] + coords[None]
    mix = mix.transpose([1, 0, 2]).reshape([ndim, -1]).T
    ## stay within patch boundary
    mix = mix.clip(min=np.zeros(ndim), max=np.array(patch.shape) - 1).astype(np.uint)
    ## replace neighbouring pixels with random values from flat dist
    patch[tuple(mix.T)] = np.random.rand(mix.shape[0]) * 4 - 2


def apply_structN2Vmask3D(patch, coords, mask):
    """
    each point in coords corresponds to the center of the mask.
    then for point in the mask with value=1 we assign a random value
    """
    z_coords = coords[0]
    coords = coords[1:]
    for z in z_coords:
        coords = np.array(coords, dtype=int)
        ndim = mask.ndim
        center = np.array(mask.shape) // 2
        ## leave the center value alone
        mask[tuple(center.T)] = 0
        ## displacements from center
        dx = np.indices(mask.shape)[:, mask == 1] - center[:, None]
        ## combine all coords (ndim, npts,) with all displacements (ncoords,ndim,)
        mix = dx.T[..., None] + coords[None]
        mix = mix.transpose([1, 0, 2]).reshape([ndim, -1]).T
        ## stay within patch boundary
        mix = mix.clip(min=np.zeros(ndim), max=np.array(patch.shape[1:]) - 1).astype(np.uint)
        ## replace neighbouring pixels with random values from flat dist
        patch[z][tuple(mix.T)] = np.random.rand(mix.shape[0]) * 4 - 2


def manipulate_val_data(
    X_val,
    Y_val,
    perc_pix=0.198,
    shape=(64, 64),
    value_manipulation=pm_uniform_withCP(5),
):
    dims = len(shape)
    if dims == 2:
        box_size = np.round(np.sqrt(100 / perc_pix), dtype=int)
        get_stratified_coords = get_stratified_coords2D
    elif dims == 3:
        box_size = np.round(np.sqrt(100 / perc_pix), dtype=int)
        get_stratified_coords = get_stratified_coords3D

    n_chan = X_val.shape[-1]

    Y_val *= 0
    for j in tqdm(
        range(X_val.shape[0]),
        desc="Preparing validation data: ",
        disable=not is_main_process(),
    ):
        coords = get_stratified_coords(box_size=box_size, shape=np.array(X_val.shape)[1:-1])
        for c in range(n_chan):
            indexing = (j,) + coords + (c,)
            indexing_mask = (j,) + coords + (c + n_chan,)
            y_val = X_val[indexing]
            x_val = value_manipulation(X_val[j, ..., c], coords, dims)

            Y_val[indexing] = y_val
            Y_val[indexing_mask] = 1
            X_val[indexing] = x_val


def get_value_manipulation(n2v_manipulator, n2v_neighborhood_radius):
    return eval("pm_{0}({1})".format(n2v_manipulator, str(n2v_neighborhood_radius)))
