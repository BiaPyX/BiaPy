from __future__ import annotations
import torch
from torch.utils.data import IterableDataset, DistributedSampler
import h5py
import os
import math
import zarr
import numpy as np
from typing import Tuple, Optional, Dict, List, Callable
from numpy.typing import NDArray

from biapy.data.data_3D_manipulation import (
    extract_patch_from_efficient_file,
    insert_patch_in_efficient_file,
    order_dimensions,
)
from biapy.data.data_manipulation import sample_satisfy_conds, save_tif
from biapy.data.data_3D_manipulation import ensure_3d_shape
from biapy.utils.misc import get_world_size, get_rank
from biapy.data.dataset import PatchCoords
from biapy.data.norm import Normalization


class chunked_test_pair_data_generator(IterableDataset):
    """
    Image data generator without data augmentation. Used only for test data.

    Parameters
    ----------
    sample_to_process : dict
        Sample to process. Expected keys are:
        * ``"X"``, Zarr/H5 data: X data to process
        * ``"img_file_to_close"``, Zarr/H5 file: X data file pointer
        * ``"Y"``, Zarr/H5 data (optional): Y data to process
        * ``"mask_file_to_close"``, Zarr/H5 file (optional): Y data file pointer

    norm_module : Normalization
        Normalization module that defines the normalization steps to apply.

    input_axes : str
        Input axes expected in the X data to be load.

    mask_input_axes : str
        Mask input axes expected in the Y data to be load.

    crop_shape : tuple of int
        Shape of the patches to extract.

    padding : tuple of int
        Padding to be applied to avoid border effects.

    path_to_gt_data : str
        Path to the ground truth data.

    out_dir: str
        Output directory to save the predicted data into.

    dtype_str : str
        Data type to be used in the Zarr/H5 created.

    convert_to_rgb : bool, optional
        Whether to convert images into 3-channel, i.e. RGB, by using the information of the first channel.

    filter_props : list of lists of str, optional
        Filter conditions to be applied to the data. The three variables, ``filter_props``, ``filter_vals`` and ``filter_vals``
        will compose a list of conditions to remove the samples from the list. They are list of list of conditions. For instance, the
        conditions can be like this: ``[['A'], ['B','C']]``. Then, if the sample satisfies the first list of conditions, only 'A'
        in this first case (from ['A'] list), or satisfy 'B' and 'C' (from ['B','C'] list) it will be removed. In each sublist all the
        conditions must be satisfied. Available properties are: [``'foreground'``, ``'mean'``, ``'min'``, ``'max'``].
        Each property descrition:
          * ``'foreground'`` is defined as the mask foreground percentage.
          * ``'mean'`` is defined as the mean value.
          * ``'min'`` is defined as the min value.
          * ``'max'`` is defined as the max value.

    filter_vals : list of int/float, optional
        Represent the values of the properties listed in ``filter_props`` that the images need to satisfy to not be dropped.

    filter_signs : list of list of str, optional
        Signs to do the comparison for data filtering. Options: [``'gt'``, ``'ge'``, ``'lt'``, ``'le'``] that corresponds to
        "greather than", e.g. ">", "greather equal", e.g. ">=", "less than", e.g. "<", and "less equal" e.g. "<=" comparisons.

    preprocess_data : Callable, optional
        Preprocessing function to apply.

    preprocess_cfg : dict, optional
        Configuration of the preprocessing.
    """

    def __init__(
        self,
        sample_to_process: Dict,
        norm_module: Normalization,
        input_axes: str,
        mask_input_axes: str,
        crop_shape: Tuple[int, ...],
        padding: Tuple[int, ...],
        out_dir: str,
        dtype_str: str = "float32",
        convert_to_rgb: bool = False,
        filter_props: List[List[str]] = [],
        filter_vals: Optional[List[List[float | int]]] = None,
        filter_signs: Optional[List[List[str]]] = None,
        preprocess_data: Optional[Callable] = None,
        preprocess_cfg: Optional[Dict] = None,
    ):
        super(chunked_test_pair_data_generator).__init__()
        self.sample_to_process = sample_to_process
        self.X_parallel_data = sample_to_process["X"]
        self.X_parallel_file = (
            sample_to_process["img_file_to_close"] if "img_file_to_close" in sample_to_process else None
        )
        if sample_to_process["Y"] is not None:
            self.Y_parallel_data = sample_to_process["Y"]
            self.Y_parallel_file = (
                sample_to_process["mask_file_to_close"] if "mask_file_to_close" in sample_to_process else None
            )
        else:
            self.Y_parallel_data = None
            self.Y_parallel_file = None
        self.filename = self.sample_to_process["filename"]
        self.file_type = "h5" if any(self.filename.endswith(x) for x in [".h5", ".hdf5", ".hdf"]) else "zarr"
        self.dir = self.sample_to_process["dir"]
        self.norm_module = norm_module
        self.input_axes = input_axes
        self.mask_input_axes = mask_input_axes
        self.dtype_str = dtype_str
        self.out_dir = out_dir
        self.convert_to_rgb = convert_to_rgb
        self.filter_samples = True if len(filter_props) > 0 else False
        self.filter_props = filter_props
        self.filter_vals = filter_vals
        self.filter_signs = filter_signs
        self.preprocess_data = preprocess_data
        self.preprocess_cfg = preprocess_cfg

        # Modify crop_shape with the channel
        c_index = -1
        try:
            c_index = input_axes.index("C")
            crop_shape = crop_shape[:-1] + (self.X_parallel_data.shape[c_index],)
        except:
            pass
        self.crop_shape = crop_shape
        self.padding = padding

        self.out_data = None
        self.out_file = None
        # Channel dimension should be equal to the number of channel of the prediction
        if "C" not in self.input_axes:
            self.out_data_order = self.input_axes + "C"
        else:
            self.out_data_order = self.input_axes

        # Ensure the out axes match with the ground truth
        if sample_to_process["Y"] is not None:
            assert (
                self.mask_input_axes == self.out_data_order
            ), f"The expected mask axes do not match the order of the output data axes to be created ({self.mask_input_axes} vs {self.out_data_order})"

        # Information about the dataset to work with
        _, self.z_dim, _, self.y_dim, self.x_dim = order_dimensions(self.X_parallel_data.shape, self.input_axes)
        assert isinstance(self.z_dim, int) and isinstance(self.x_dim, int) and isinstance(self.y_dim, int)
        if self.crop_shape[0] > self.z_dim:
            raise ValueError(
                "Z Axis problem: {} greater than {} (you can reduce 'DATA.PATCH_SIZE' in that axis)".format(
                    self.crop_shape[0], self.z_dim
                )
            )
        if self.crop_shape[1] > self.y_dim:
            raise ValueError(
                "Y Axis problem: {} greater than {} (you can reduce 'DATA.PATCH_SIZE' in that axis)".format(
                    self.crop_shape[1], self.y_dim
                )
            )
        if self.crop_shape[2] > self.x_dim:
            raise ValueError(
                "X Axis problem: {} greater than {} (you can reduce 'DATA.PATCH_SIZE' in that axis)".format(
                    self.crop_shape[2], self.x_dim
                )
            )

        # Z
        self.step_z = self.crop_shape[0] - (self.padding[0] * 2)
        self.vols_per_z = math.ceil(self.z_dim / self.step_z)

        # Y
        self.step_y = self.crop_shape[1] - (self.padding[1] * 2)
        self.vols_per_y = math.ceil(self.y_dim / self.step_y)

        # X
        self.step_x = self.crop_shape[2] - (self.padding[2] * 2)
        self.vols_per_x = math.ceil(self.x_dim / self.step_x)

        self.len = self.vols_per_z * self.vols_per_y * self.vols_per_x

    def extract_and_prepare_sample(
        self, z: int, y: int, x: int, patch_coords: PatchCoords, extract: str = "image"
    ) -> Tuple[NDArray, List[List[int]]]:
        """
        Extract and prepare the data sample from the parallel data.

        Parameters
        ----------
        z : int
            Number of samples processed in Z axis.

        y : int
            Number of samples processed in Y axis.

        x : int
            Number of samples processed in X axis.

        patch_coords : PatchCoords
            Coordinates of the patch to extract.

        extract : str, optional
            Whether to extract the image or the mask from the parallel data.
            Options: ``["image", "mask"]``

        Returns
        -------
        data : NDArray
            Extracted patch.

        added_pad : list of list of ints
            Added pad on each dimension. E.g. [ [10, 10], [5,5], [0,5]]
        """
        assert extract in ["image", "mask"]
        if extract == "image":
            input_axes = self.input_axes
            var_tag = "DATA.TEST.INPUT_IMG_AXES_ORDER"
        else:
            input_axes = self.mask_input_axes
            var_tag = "DATA.TEST.INPUT_MASK_AXES_ORDER"

        # Extact the patch
        if extract == "image":
            data = extract_patch_from_efficient_file(self.X_parallel_data, patch_coords, input_axes)
        else:  # mask
            data = extract_patch_from_efficient_file(self.Y_parallel_data, patch_coords, input_axes)

        # Ensure the shape of the extracted patch is as the crop_shape
        pad_z_left = self.padding[0] if z * self.step_z - self.padding[0] < 0 else 0
        pad_z_right = self.crop_shape[0] - (patch_coords.z_end - patch_coords.z_start) - pad_z_left
        pad_y_left = self.padding[1] if y * self.step_y - self.padding[1] < 0 else 0
        pad_y_right = self.crop_shape[1] - (patch_coords.y_end - patch_coords.y_start) - pad_y_left
        pad_x_left = self.padding[2] if x * self.step_x - self.padding[2] < 0 else 0
        pad_x_right = self.crop_shape[2] - (patch_coords.x_end - patch_coords.x_start) - pad_x_left
        pad_to_add = [
            [pad_z_left, pad_z_right],
            [pad_y_left, pad_y_right],
            [pad_x_left, pad_x_right],
        ]
        if data.ndim == 3:
            data = np.pad(data, pad_to_add, "reflect")
            data = np.expand_dims(data, -1)
        else:
            pad_to_add += [
                [0, 0],
            ]
            data = np.pad(data, pad_to_add, "reflect")

        # Save real padding info
        pad_to_add[0][0] = max(pad_to_add[0][0], self.padding[0])
        pad_to_add[0][1] = max(pad_to_add[0][1], self.padding[0])
        pad_to_add[1][0] = max(pad_to_add[1][0], self.padding[1])
        pad_to_add[1][1] = max(pad_to_add[1][1], self.padding[1])
        pad_to_add[2][0] = max(pad_to_add[2][0], self.padding[2])
        pad_to_add[2][1] = max(pad_to_add[2][1], self.padding[2])

        assert (
            data.shape[:-1] == self.crop_shape[:-1]
        ), f"Image shape and expected shape differ: {data.shape} vs {self.crop_shape}. " \
           f"Double check that the data is following '{input_axes}' axis order (set in '{var_tag}')"

        if self.convert_to_rgb:
            if extract == "image" or (extract == "mask" and self.norm_module.mask_norm == "as_image"):
                if data.shape[-1] == 1:
                    data = np.repeat(data, 3, axis=-1)

        return data, pad_to_add

    def __iter__(self):
        """
        Function to iterate over the generator.

        Returns
        -------
        img : NDArray
            X patch of data to process.

        mask : NDArray
            Y yatch of data to process.

        real_patch_in_data : PatchCoords
            Coordinates of the patch in the data.

        pad_to_add : List of list of ints
            Padding added to the patch in order to satisfy the crop shape expected.

        norm_extra_info : dict
            Extra information of the normalization applied ``img``.
        """
        assert isinstance(self.z_dim, int) and isinstance(self.x_dim, int) and isinstance(self.y_dim, int)
        worker_info = torch.utils.data.get_worker_info()  # type: ignore
        n_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        world_size = get_world_size()
        process_rank = get_rank()

        sampler = DistributedSampler(
            self, num_replicas=(n_workers * world_size), rank=(process_rank * n_workers + worker_id), shuffle=False
        )

        for i in sampler:
            if worker_info is None:
                self.z_vols_to_process = list(range(self.vols_per_z))
            else:
                units = n_workers * world_size
                vols_per_worker = self.vols_per_z // units
                first_block = vols_per_worker * i
                if self.vols_per_z % units > i:
                    vols_per_worker += 1
                    extra_z_vols = i
                else:
                    extra_z_vols = self.vols_per_z % units
                self.z_vols_to_process = list(
                    range(
                        first_block + extra_z_vols,
                        min(first_block + extra_z_vols + vols_per_worker, self.vols_per_z),
                    )
                )

            mask = None
            for z in self.z_vols_to_process:
                for y in range(self.vols_per_y):
                    for x in range(self.vols_per_x):
                        start_z = max(0, z * self.step_z - self.padding[0])
                        finish_z = min((z + 1) * self.step_z + self.padding[0], self.z_dim)
                        start_y = max(0, y * self.step_y - self.padding[1])
                        finish_y = min((y + 1) * self.step_y + self.padding[1], self.y_dim)
                        start_x = max(0, x * self.step_x - self.padding[2])
                        finish_x = min((x + 1) * self.step_x + self.padding[2], self.x_dim)
                        patch_to_extract = PatchCoords(
                            y_start=start_y,
                            y_end=finish_y,
                            x_start=start_x,
                            x_end=finish_x,
                            z_start=start_z,
                            z_end=finish_z,
                        )

                        # The real data that is being processed. This doesn't take into account the adding pad.
                        # This coordinates are useful to know after where to insert the data
                        real_patch_in_data = PatchCoords(
                            z_start=z * self.step_z,
                            z_end=min((z + 1) * self.step_z, self.z_dim),
                            y_start=y * self.step_y,
                            y_end=min((y + 1) * self.step_y, self.y_dim),
                            x_start=x * self.step_x,
                            x_end=min((x + 1) * self.step_x, self.x_dim),
                        )

                        img, added_pad = self.extract_and_prepare_sample(z, y, x, patch_to_extract)
                        if self.Y_parallel_data is not None:
                            mask, _ = self.extract_and_prepare_sample(z, y, x, patch_to_extract, extract="mask")

                        # Skip processing image
                        discard = False
                        if self.filter_samples:
                            foreground_filter_requested = False
                            for cond in self.filter_props:
                                if (
                                    "foreground" in cond
                                    or "diff" in cond
                                    or "diff_by_min_max_ratio" in cond
                                    or "diff_by_target_min_max_ratio" in cond
                                    or "target_mean" in cond
                                    or "target_min" in cond
                                    or "target_max" in cond
                                ):
                                    foreground_filter_requested = True
                            assert self.filter_vals and self.filter_signs
                            discard = sample_satisfy_conds(
                                img,
                                self.filter_props,
                                self.filter_vals,
                                self.filter_signs,
                                mask=mask if foreground_filter_requested else None,
                            )

                        if not discard:
                            # Preprocess test data
                            if self.preprocess_data:
                                img = self.preprocess_data(
                                    self.preprocess_cfg,
                                    x_data=[img],
                                    is_2d=(img.ndim == 3),
                                )[0]
                                if self.Y_parallel_data:
                                    mask = self.preprocess_data(
                                        self.preprocess_cfg,
                                        y_data=[mask],
                                        is_2d=(img.ndim == 3),
                                        is_y_mask=True,
                                    )[0]

                            # Normalization
                            self.norm_module.set_stats_from_image(img)
                            img, norm_extra_info = self.norm_module.apply_image_norm(img)
                            if mask is not None:
                                mask = np.array(mask)
                                self.norm_module.set_stats_from_mask(mask)
                                mask, _ = self.norm_module.apply_mask_norm(mask)
                                assert isinstance(mask, np.ndarray)

                            yield img, mask, real_patch_in_data, added_pad, norm_extra_info

    def insert_patch_in_file(self, patch: NDArray, patch_coords: PatchCoords):
        """
        Insert patch into the output parallel file. It always creates a Zarr dataset using ``self.crop_shape``
        as chunk size.

        Parameters
        ----------
        patch : int
            Sample index counter.

        patch_coords : PatchCoords
            Whether its the first time a sample is loaded to prevent normalizing it.

        """
        # Create the data container if it was not created
        if not self.out_file:
            # Channel dimension should be equal to the number of channel of the prediction
            out_data_shape = np.array(self.X_parallel_data.shape)
            if "C" not in self.input_axes:
                out_data_shape = tuple(out_data_shape) + (patch.shape[-1],)
            else:
                out_data_shape[self.input_axes.index("C")] = patch.shape[-1]
                out_data_shape = tuple(out_data_shape)

            data_filename = os.path.join(self.out_dir, os.path.splitext(self.filename)[0] + ".zarr")
            os.makedirs(self.out_dir, exist_ok=True)
            self.out_file = data_filename
            self.out_data = zarr.open_array(
                data_filename,
                shape=out_data_shape,
                mode="w",
                chunks=self.crop_shape,  # type: ignore
                dtype=self.dtype_str,
            )

        insert_patch_in_efficient_file(
            data=self.out_data,
            patch=patch,
            patch_coords=patch_coords,
            data_axes_order=self.out_data_order,
            patch_axes_order="ZYXC",
        )

    def save_parallel_data_as_tif(self):
        """
        Saves the parallel data (``self.out_data``) as a tiff file.
        """
        data = np.array(self.out_data)
        data = ensure_3d_shape(data)
        save_tif(np.expand_dims(data, 0), self.out_dir, [os.path.splitext(self.filename)[0] + ".tif"], verbose=True)

    def close_open_files(self):
        # Input data files
        if self.X_parallel_file is not None and isinstance(self.X_parallel_file, h5py.File):
            self.X_parallel_file.close()
        if self.Y_parallel_file is not None and isinstance(self.Y_parallel_file, h5py.File):
            self.Y_parallel_file.close()
        # Output data file
        if isinstance(self.out_file, h5py.File):
            self.out_file.close()

    def __len__(self):
        return self.len
