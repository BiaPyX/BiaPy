"""
Denoising workflow and utilities for BiaPy.

This module provides the Denoising_Workflow class for training and inference on image denoising tasks,
as well as utility functions for patch manipulation, stratified coordinate sampling, and structN2V masking.
It supports both 2D and 3D data, and includes implementations of various pixel manipulation strategies
used in self-supervised denoising approaches such as Noise2Void (N2V).
"""
import math
import torch
import numpy as np
import numpy.ma as ma
from tqdm import tqdm
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError
from typing import Tuple, Callable, Dict, Optional
from numpy.typing import NDArray

from biapy.data.data_2D_manipulation import (
    crop_data_with_overlap,
    merge_data_with_overlap,
)
from biapy.data.data_3D_manipulation import (
    crop_3D_data_with_overlap,
    merge_3D_data_with_overlap,
)
from biapy.engine.base_workflow import Base_Workflow
from biapy.data.data_manipulation import save_tif
from biapy.utils.misc import to_pytorch_format, is_main_process, to_pytorch_format, MetricLogger
from biapy.engine.metrics import n2v_loss_mse, loss_encapsulation


class Denoising_Workflow(Base_Workflow):
    """
    Denoising workflow where the goal is to remove noise from an image.

    More details in `our documentation <https://biapy.readthedocs.io/en/latest/workflows/denoising.html>`_.

    Parameters
    ----------
    cfg : YACS configuration
        Running configuration.
    job_identifier : str
        Complete name of the running job.
    device : torch.device
        Device used.
    args : argparse.Namespace
        Arguments used in BiaPy's call.
    """

    def __init__(self, cfg, job_identifier, device, args, **kwargs):
        """
        Initialize the Denoising_Workflow.

        Sets up configuration, device, job identifier, and initializes
        workflow-specific attributes for denoising tasks.

        Parameters
        ----------
        cfg : YACS configuration
            Running configuration.
        job_identifier : str
            Complete name of the running job.
        device : torch.device
            Device used.
        args : argparse.Namespace
            Arguments used in BiaPy's call.
        **kwargs : dict
            Additional keyword arguments.
        """
        super(Denoising_Workflow, self).__init__(cfg, job_identifier, device, args, **kwargs)

        # From now on, no modification of the cfg will be allowed
        self.cfg.freeze()

        # Workflow specific training variables
        self.mask_path = cfg.DATA.TRAIN.GT_PATH if cfg.PROBLEM.DENOISING.LOAD_GT_DATA else None
        self.is_y_mask = False
        self.load_Y_val = cfg.PROBLEM.DENOISING.LOAD_GT_DATA

        self.norm_module.mask_norm = "as_image"
        self.test_norm_module.mask_norm = "as_image"

    def define_activations_and_channels(self):
        """
        Define the activations and output channels of the model.

        This function must define the following variables:

        self.model_output_channels : List of functions
            Metrics to be calculated during model's training.

        self.multihead : bool
            Whether if the output of the model has more than one head.

        self.activations : List of dicts
            Activations to be applied to the model output. Each dict will
            match an output channel of the model. If ':' is used the activation
            will be applied to all channels at once. "Linear" and "CE_Sigmoid"
            will not be applied. E.g. [{":": "Linear"}].
        """
        self.model_output_channels = {
            "type": "image",
            "channels": [self.cfg.DATA.PATCH_SIZE[-1]],
        }
        self.real_classes = self.model_output_channels["channels"][0]
        self.multihead = False
        self.activations = [{":": "Linear"}]

        super().define_activations_and_channels()

    def define_metrics(self):
        """
        Define the metrics to be used during training and test/inference.

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

        # print("Overriding 'LOSS.TYPE' to set it to N2V loss (masked MSE)")
        if self.cfg.LOSS.TYPE == "MSE":
            self.loss = loss_encapsulation(n2v_loss_mse)

        super().define_metrics()

    def metric_calculation(
        self,
        output: NDArray | torch.Tensor,
        targets: NDArray | torch.Tensor,
        train: bool = True,
        metric_logger: Optional[MetricLogger] = None,
    ) -> Dict:
        """
        Execute the calculation of metrics defined in :func:`~define_metrics` function.

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
        if isinstance(output, dict):
            output = output["pred"]
        if isinstance(output, np.ndarray):
            _output = to_pytorch_format(
                output.copy(),
                self.axes_order,
                self.device,
                dtype=self.loss_dtype,
            )
        else:  # torch.Tensor
            if not train:
                _output = output.clone()
            else:
                _output = output

        if isinstance(targets, np.ndarray):
            _targets = to_pytorch_format(
                targets.copy(),
                self.axes_order,
                self.device,
                dtype=self.loss_dtype,
            )
        else:  # torch.Tensor
            if not train:
                _targets = targets.clone()
            else:
                _targets = targets

        out_metrics = {}
        list_to_use = self.train_metrics if train else self.test_metrics
        list_names_to_use = self.train_metric_names if train else self.test_metric_names

        with torch.no_grad():
            for i, metric in enumerate(list_to_use):
                val = metric(_output.squeeze(), _targets[:, 0].squeeze())
                val = val.item() if not torch.isnan(val) else 0
                out_metrics[list_names_to_use[i]] = val

                if metric_logger:
                    metric_logger.meters[list_names_to_use[i]].update(val)
        return out_metrics

    def process_test_sample(self):
        """Process a sample in the test/inference phase."""
        assert self.model is not None
        # Skip processing image
        if "discard" in self.current_sample["X"] and self.current_sample["X"]["discard"]:
            return True

        original_data_shape = self.current_sample["X"].shape

        # Crop if necessary
        if self.current_sample["X"].shape[1:-1] != self.cfg.DATA.PATCH_SIZE[:-1]:
            if self.cfg.PROBLEM.NDIM == "2D":
                self.current_sample["X"], _ = crop_data_with_overlap(  # type: ignore
                    self.current_sample["X"],
                    self.cfg.DATA.PATCH_SIZE,
                    overlap=self.cfg.DATA.TEST.OVERLAP,
                    padding=self.cfg.DATA.TEST.PADDING,
                    verbose=self.cfg.TEST.VERBOSE,
                )
            else:
                self.current_sample["X"], _ = crop_3D_data_with_overlap(  # type: ignore
                    self.current_sample["X"][0],
                    self.cfg.DATA.PATCH_SIZE,
                    overlap=self.cfg.DATA.TEST.OVERLAP,
                    padding=self.cfg.DATA.TEST.PADDING,
                    verbose=self.cfg.TEST.VERBOSE,
                    median_padding=self.cfg.DATA.TEST.MEDIAN_PADDING,
                )

        pred = self.predict_batches_in_test(self.current_sample["X"], None)
        del self.current_sample["X"]

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
                assert isinstance(pred, np.ndarray)
                pred = np.expand_dims(pred, 0)

        if self.cfg.DATA.REFLECT_TO_COMPLETE_SHAPE:
            reflected_orig_shape = (1,) + self.current_sample["reflected_orig_shape"]
            if reflected_orig_shape != pred.shape:
                if self.cfg.PROBLEM.NDIM == "2D":
                    pred = pred[:, -reflected_orig_shape[1] :, -reflected_orig_shape[2] :]  # type: ignore
                else:
                    pred = pred[
                        :,
                        -reflected_orig_shape[1] :,
                        -reflected_orig_shape[2] :,
                        -reflected_orig_shape[3] :,
                    ]  # type: ignore

        # Undo normalization
        assert isinstance(pred, np.ndarray)
        pred = self.norm_module.undo_image_norm(pred, self.current_sample["X_norm"])

        # Save image
        if self.cfg.PATHS.RESULT_DIR.PER_IMAGE != "":
            assert isinstance(pred, np.ndarray)
            save_tif(
                pred,
                self.cfg.PATHS.RESULT_DIR.PER_IMAGE,
                [self.current_sample["filename"]],
                verbose=self.cfg.TEST.VERBOSE,
            )

    def torchvision_model_call(self, in_img: torch.Tensor, is_train: bool = False) -> torch.Tensor | None:
        """
        Call a regular Pytorch model.

        Parameters
        ----------
        in_img : torch.Tensor
            Input image to pass through the model.

        is_train : bool, optional
            Whether if the call is during training or inference.

        Returns
        -------
        prediction : torch.Tensor
            Image prediction.
        """
        pass

    def after_merge_patches(self, pred: torch.Tensor):
        """
        Execute steps needed after merging all predicted patches into the original image.

        Parameters
        ----------
        pred : Torch Tensor
            Model prediction.
        """
        pass

    def after_full_image(self, pred: NDArray):
        """
        Execute steps needed after generating the prediction by supplying the entire image to the model.

        Parameters
        ----------
        pred : NDArray
            Model prediction.
        """
        pass

    def after_all_images(self):
        """Excute steps that must be done after predicting all images."""
        super().after_all_images()


####################################
# Adapted from N2V code:           #
#   https://github.com/juglab/n2v  #
####################################


def get_subpatch(patch, coord, local_sub_patch_radius, crop_patch=True):
    """
    Extract a subpatch centered at a given coordinate, handling border cropping.

    Parameters
    ----------
    patch : np.ndarray
        Input patch.
    coord : tuple of int
        Center coordinate for the subpatch.
    local_sub_patch_radius : int
        Radius of the subpatch to extract.
    crop_patch : bool, optional
        Whether to crop the patch at the borders (default: True).

    Returns
    -------
    subpatch : np.ndarray
        Extracted subpatch.
    crop_neg : int
        Negative crop offset.
    crop_pos : int
        Positive crop offset.
    """
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
    """
    Sample a random neighbor coordinate different from the given coordinate.

    Parameters
    ----------
    shape : tuple of int
        Shape of the patch.
    coord : tuple of int
        Center coordinate.

    Returns
    -------
    rand_coords : list of int
        Random neighbor coordinate.
    """
    rand_coords = sample_coords(shape, coord)
    while np.any(rand_coords == coord):
        rand_coords = sample_coords(shape, coord)

    return rand_coords


def sample_coords(shape, coord, sigma=4):
    """
    Sample random coordinates from a normal distribution centered at coord.

    Parameters
    ----------
    shape : tuple of int
        Shape of the patch.
    coord : tuple of int
        Center coordinate.
    sigma : float, optional
        Standard deviation for the normal distribution (default: 4).

    Returns
    -------
    coords : list of int
        Sampled coordinates.
    """
    return [normal_int(c, sigma, s) for c, s in zip(coord, shape)]


def normal_int(mean, sigma, w):
    """
    Sample an integer from a normal distribution and clip to valid range.

    Parameters
    ----------
    mean : float
        Mean of the normal distribution.
    sigma : float
        Standard deviation.
    w : int
        Maximum allowed value (exclusive).

    Returns
    -------
    int
        Sampled and clipped integer.
    """
    return int(np.clip(np.round(np.random.normal(mean, sigma)), 0, w - 1))


def mask_center(local_sub_patch_radius, ndims=2):
    """
    Create a mask with the center pixel set to zero.

    Parameters
    ----------
    local_sub_patch_radius : int
        Radius of the patch.
    ndims : int, optional
        Number of dimensions (default: 2).

    Returns
    -------
    mask : np.ndarray
        Boolean mask with center pixel set to zero.
    """
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
    """
    Return a function that samples a random neighbor from a normal distribution (without center pixel).

    Parameters
    ----------
    local_sub_patch_radius : int
        Radius of the local subpatch.

    Returns
    -------
    Callable
        Function that takes (patch, coords, dims, structN2Vmask) and returns values from random neighbors.
    """
    def normal_withoutCP(patch, coords, dims, structN2Vmask=None):
        vals = []
        for coord in zip(*coords):
            rand_coords = random_neighbor(patch.shape, coord)
            vals.append(patch[tuple(rand_coords)])
        return vals

    return normal_withoutCP


def pm_mean(local_sub_patch_radius):
    """
    Return a function that computes the mean of the local neighborhood (excluding center pixel).

    Parameters
    ----------
    local_sub_patch_radius : int
        Radius of the local subpatch.

    Returns
    -------
    Callable
        Function that takes (patch, coords, dims, structN2Vmask) and returns mean values.
    """
    def patch_mean(patch, coords, dims, structN2Vmask=None):
        patch_wo_center = mask_center(local_sub_patch_radius, ndims=dims)
        vals = []
        for coord in zip(*coords):
            sub_patch, crop_neg, crop_pos = get_subpatch(patch, coord, local_sub_patch_radius)
            slices = [slice(-n, s - p) for n, p, s in zip(crop_neg, crop_pos, patch_wo_center.shape)]  # type: ignore
            sub_patch_mask = (structN2Vmask or patch_wo_center)[tuple(slices)]
            vals.append(np.mean(sub_patch[sub_patch_mask]))
        return vals

    return patch_mean


def pm_median(local_sub_patch_radius):
    """
    Return a function that computes the median of the local neighborhood (excluding center pixel).

    Parameters
    ----------
    local_sub_patch_radius : int
        Radius of the local subpatch.

    Returns
    -------
    Callable
        Function that takes (patch, coords, dims, structN2Vmask) and returns median values.
    """
    def patch_median(patch, coords, dims, structN2Vmask=None):
        patch_wo_center = mask_center(local_sub_patch_radius, ndims=dims)
        vals = []
        for coord in zip(*coords):
            sub_patch, crop_neg, crop_pos = get_subpatch(patch, coord, local_sub_patch_radius)
            slices = [slice(-n, s - p) for n, p, s in zip(crop_neg, crop_pos, patch_wo_center.shape)]  # type: ignore
            sub_patch_mask = (structN2Vmask or patch_wo_center)[tuple(slices)]
            vals.append(np.median(sub_patch[sub_patch_mask]))
        return vals

    return patch_median


def pm_uniform_withCP(local_sub_patch_radius):
    """
    Return a function that samples a random value from the local neighborhood (including center pixel).

    Parameters
    ----------
    local_sub_patch_radius : int
        Radius of the local subpatch.

    Returns
    -------
    Callable
        Function that takes (patch, coords, dims, structN2Vmask) and returns random values.
    """
    def random_neighbor_withCP_uniform(patch, coords, dims, structN2Vmask=None):
        vals = []
        for coord in zip(*coords):
            sub_patch, _, _ = get_subpatch(patch, coord, local_sub_patch_radius)
            rand_coords = [np.random.randint(0, s) for s in sub_patch.shape[0:dims]]
            vals.append(sub_patch[tuple(rand_coords)])
        return vals

    return random_neighbor_withCP_uniform


def pm_uniform_withoutCP(local_sub_patch_radius):
    """
    Return a function that samples a random value from the local neighborhood (excluding center pixel).

    Parameters
    ----------
    local_sub_patch_radius : int
        Radius of the local subpatch.

    Returns
    -------
    Callable
        Function that takes (patch, coords, dims, structN2Vmask) and returns random values.
    """
    def random_neighbor_withoutCP_uniform(patch, coords, dims, structN2Vmask=None):
        patch_wo_center = mask_center(local_sub_patch_radius, ndims=dims)
        vals = []
        for coord in zip(*coords):
            sub_patch, crop_neg, crop_pos = get_subpatch(patch, coord, local_sub_patch_radius)
            slices = [slice(-n, s - p) for n, p, s in zip(crop_neg, crop_pos, patch_wo_center.shape)]  # type: ignore
            sub_patch_mask = (structN2Vmask or patch_wo_center)[tuple(slices)]
            vals.append(np.random.permutation(sub_patch[sub_patch_mask])[0])
        return vals

    return random_neighbor_withoutCP_uniform


def pm_normal_additive(pixel_gauss_sigma):
    """
    Return a function that adds Gaussian noise to the center pixel.

    Parameters
    ----------
    pixel_gauss_sigma : float
        Standard deviation of the Gaussian noise.

    Returns
    -------
    Callable
        Function that takes (patch, coords, dims, structN2Vmask) and returns noisy values.
    """
    def pixel_gauss(patch, coords, dims, structN2Vmask=None):
        vals = []
        for coord in zip(*coords):
            vals.append(np.random.normal(patch[tuple(coord)], pixel_gauss_sigma))
        return vals

    return pixel_gauss


def pm_normal_fitted(local_sub_patch_radius):
    """
    Return a function that samples from a Gaussian fitted to the local neighborhood.

    Parameters
    ----------
    local_sub_patch_radius : int
        Radius of the local subpatch.

    Returns
    -------
    Callable
        Function that takes (patch, coords, dims, structN2Vmask) and returns sampled values.
    """
    def local_gaussian(patch, coords, dims, structN2Vmask=None):
        vals = []
        for coord in zip(*coords):
            sub_patch, _, _ = get_subpatch(patch, coord, local_sub_patch_radius)
            axis = tuple(range(dims))
            vals.append(np.random.normal(np.mean(sub_patch, axis=axis), np.std(sub_patch, axis=axis)))
        return vals

    return local_gaussian


def pm_identity(local_sub_patch_radius):
    """
    Return a function that simply returns the center pixel value (identity).

    Parameters
    ----------
    local_sub_patch_radius : int
        Radius of the local subpatch (unused).

    Returns
    -------
    Callable
        Function that takes (patch, coords, dims, structN2Vmask) and returns the center pixel value.
    """
    def identity(patch, coords, dims, structN2Vmask=None):
        vals = []
        for coord in zip(*coords):
            vals.append(patch[coord])
        return vals

    return identity


def get_stratified_coords2D(box_size, shape):
    """
    Generate stratified random coordinates for 2D patches.

    Parameters
    ----------
    box_size : int
        Size of the box for stratification.
    shape : tuple of int
        Shape of the 2D image.

    Returns
    -------
    tuple of lists
        (y_coords, x_coords) for sampled points.
    """
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
    """
    Generate stratified random coordinates for 3D patches.

    Parameters
    ----------
    box_size : int
        Size of the box for stratification.
    shape : tuple of int
        Shape of the 3D image.

    Returns
    -------
    tuple of lists
        (z_coords, y_coords, x_coords) for sampled points.
    """
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
    Apply a structN2V mask to a 2D patch.

    Each point in coords corresponds to the center of the mask.
    For each point in the mask with value=1, assign a random value.

    Parameters
    ----------
    patch : np.ndarray
        Input patch to modify.
    coords : np.ndarray or list
        Coordinates of mask centers.
    mask : np.ndarray
        Binary mask to apply.
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
    Apply a structN2V mask to a 3D patch.

    Each point in coords corresponds to the center of the mask.
    For each point in the mask with value=1, assign a random value.

    Parameters
    ----------
    patch : np.ndarray
        Input 3D patch to modify.
    coords : np.ndarray or list
        Coordinates of mask centers (z, y, x).
    mask : np.ndarray
        Binary mask to apply.
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
    X_val: NDArray,
    Y_val: NDArray,
    perc_pix: float = 0.198,
    shape: Tuple[int, ...] = (64, 64),
    value_manipulation: Callable = pm_uniform_withCP(5),
):
    """
    Manipulate validation data for self-supervised denoising.

    Applies a value manipulation strategy (e.g., uniform, mean, median) to a percentage of pixels
    in the validation set, as used in Noise2Void/structN2V validation.

    Parameters
    ----------
    X_val : NDArray
        Validation input data.
    Y_val : NDArray
        Validation target data (will be overwritten).
    perc_pix : float, optional
        Percentage of pixels to manipulate (default: 0.198).
    shape : tuple of int, optional
        Shape of the patch (default: (64, 64)).
    value_manipulation : Callable, optional
        Function to manipulate pixel values (default: pm_uniform_withCP(5)).
    """
    dims = len(shape)
    if dims == 2:
        box_size = np.round(np.sqrt(100 / perc_pix), dtype=int)  # type: ignore
        get_stratified_coords = get_stratified_coords2D
    elif dims == 3:
        box_size = np.round(np.sqrt(100 / perc_pix), dtype=int)  # type: ignore
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
    """
    Return a value manipulation function for N2V/structN2V based on the given strategy.

    Parameters
    ----------
    n2v_manipulator : str
        Name of the manipulation strategy (e.g., 'uniform_withCP').
    n2v_neighborhood_radius : int
        Neighborhood radius for the manipulation.

    Returns
    -------
    Callable
        Value manipulation function.
    """
    return eval("pm_{0}({1})".format(n2v_manipulator, str(n2v_neighborhood_radius)))
