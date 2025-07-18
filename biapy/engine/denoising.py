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
from biapy.engine.metrics import n2v_loss_mse
from biapy.data.dataset import PatchCoords


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

        # Workflow specific training variables
        self.mask_path = cfg.DATA.TRAIN.GT_PATH if cfg.PROBLEM.DENOISING.LOAD_GT_DATA else None
        self.is_y_mask = False
        self.load_Y_val = cfg.PROBLEM.DENOISING.LOAD_GT_DATA

        self.norm_module.mask_norm = "as_image"
        self.test_norm_module.mask_norm = "as_image"

    def define_activations_and_channels(self):
        """
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
            self.loss = n2v_loss_mse

        super().define_metrics()

    def metric_calculation(
        self,
        output: NDArray | torch.Tensor,
        targets: NDArray | torch.Tensor,
        train: bool = True,
        metric_logger: Optional[MetricLogger] = None,
    ) -> Dict:
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
        """
        Function to process a sample in the inference phase.
        """
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
        Steps need to be done after merging all predicted patches into the original image.

        Parameters
        ----------
        pred : Torch Tensor
            Model prediction.
        """
        pass

    def after_full_image(self, pred: NDArray):
        """
        Steps that must be executed after generating the prediction by supplying the entire image to the model.

        Parameters
        ----------
        pred : NDArray
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
            slices = [slice(-n, s - p) for n, p, s in zip(crop_neg, crop_pos, patch_wo_center.shape)]  # type: ignore
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
            slices = [slice(-n, s - p) for n, p, s in zip(crop_neg, crop_pos, patch_wo_center.shape)]  # type: ignore
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
            slices = [slice(-n, s - p) for n, p, s in zip(crop_neg, crop_pos, patch_wo_center.shape)]  # type: ignore
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
    X_val: NDArray,
    Y_val: NDArray,
    perc_pix: float = 0.198,
    shape: Tuple[int, ...] = (64, 64),
    value_manipulation: Callable = pm_uniform_withCP(5),
):
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
    return eval("pm_{0}({1})".format(n2v_manipulator, str(n2v_neighborhood_radius)))
