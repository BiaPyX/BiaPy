"""
Chunked test pair data generator for BiaPy.

This module provides an IterableDataset for generating test data pairs from chunked
Zarr/HDF5 files, including patch extraction, normalization, filtering, and saving
results. It is designed for efficient inference on large volumetric datasets.
"""
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
from tqdm import tqdm

from biapy.data.data_3D_manipulation import (
    extract_patch_from_efficient_file,
    ensure_3d_shape,
    insert_patch_in_efficient_file,
    order_dimensions,
    looks_like_hdf5,
)
from biapy.data.data_manipulation import sample_satisfy_conds, save_tif, extract_patch_within_image
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
        n_classes: int = 1,
        ignore_index: Optional[int] = None,
        instance_problem: bool = False,
    ):
        """
        Initialize the chunked_test_pair_data_generator.

        Parameters
        ----------
        sample_to_process : dict
            Dictionary containing sample data and file pointers.
        norm_module : Normalization
            Normalization module to apply.
        input_axes : str
            Axes order for input data.
        mask_input_axes : str
            Axes order for mask data.
        crop_shape : tuple of int
            Shape of the patches to extract.
        padding : tuple of int
            Padding to apply to patches.
        out_dir : str
            Output directory for results.
        dtype_str : str, optional
            Data type for output.
        convert_to_rgb : bool, optional
            Convert single-channel images to RGB.
        filter_props : list of list of str, optional
            Properties for filtering samples.
        filter_vals : list of list of float or int, optional
            Values for filtering samples.
        filter_signs : list of list of str, optional
            Comparison signs for filtering.
        preprocess_data : Callable, optional
            Preprocessing function.
        preprocess_cfg : dict, optional
            Preprocessing configuration.
        n_classes : int, optional
            Number of classes.
        ignore_index : int, optional
            Index to ignore in mask.
        instance_problem : bool, optional
            Whether the problem is instance segmentation.
        """
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
        self.filename = self.sample_to_process["X_filename"]
        self.file_type = "h5" if looks_like_hdf5(self.filename) else "zarr"
        self.dir = self.sample_to_process["X_dir"]
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
        self.n_classes = n_classes 
        self.instance_problem = instance_problem
        self.ignore_index = ignore_index

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
        for i, p in enumerate(self.padding):
            if p >= self.crop_shape[i] // 2:
                raise ValueError(
                    "'Padding' can not be greater than half of 'crop_shape'. Max value for the given input shape {} is {}".format(
                        self.crop_shape, ((self.crop_shape[0] // 2) - 1, (self.crop_shape[1] // 2) - 1, (self.crop_shape[2] // 2) - 1)
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
        data_to_process = self.X_parallel_data if extract == "image" else self.Y_parallel_data
        if not isinstance(data_to_process, np.ndarray):
            data = extract_patch_from_efficient_file(data_to_process, patch_coords, input_axes)
        else:
            data = extract_patch_within_image(data_to_process, patch_coords, is_3d=True)

        # Ensure the shape of the extracted patch is as the crop_shape
        pad_z_left = abs(z * self.step_z - self.padding[0]) if z * self.step_z - self.padding[0] < 0 else 0
        pad_z_right = self.crop_shape[0] - (patch_coords.z_end - patch_coords.z_start) - pad_z_left
        pad_y_left = abs(y * self.step_y - self.padding[1]) if y * self.step_y - self.padding[1] < 0 else 0
        pad_y_right = self.crop_shape[1] - (patch_coords.y_end - patch_coords.y_start) - pad_y_left
        pad_x_left = abs(x * self.step_x - self.padding[2]) if x * self.step_x - self.padding[2] < 0 else 0
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

        assert data.shape[:-1] == self.crop_shape[:-1], (
            f"Image shape and expected shape differ: {data.shape} vs {self.crop_shape}. "
            f"Double check that the data is following '{input_axes}' axis order (set in '{var_tag}')"
        )

        if self.convert_to_rgb:
            if extract == "image" or (extract == "mask" and self.norm_module.mask_norm == "as_image"):
                if data.shape[-1] == 1:
                    data = np.repeat(data, 3, axis=-1)

        return data, pad_to_add

    def __iter__(self):
        """
        Iterate over the generator.

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

        for vol_id in sampler:
            mask = None

            z, y, x = np.unravel_index(vol_id, (self.vols_per_z, self.vols_per_y, self.vols_per_x))
            z = int(z)
            y = int(y)
            x = int(x)

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
                norm_extra_info = {}
                if mask is not None:
                    mask = np.array(mask)
                    self.norm_module.set_stats_from_mask(mask, n_classes=self.n_classes, ignore_index=self.ignore_index, instance_problem=self.instance_problem)
                    mask, _ = self.norm_module.apply_mask_norm(mask)
                    assert isinstance(mask, np.ndarray)

                yield vol_id, img, mask, real_patch_in_data, added_pad, norm_extra_info

    def insert_patch_in_file(self, patch: NDArray, patch_coords: PatchCoords):
        """
        Insert patch into the output parallel file.
        
        It always creates a Zarr dataset using ``self.crop_shape`` as chunk size.

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
            self.out_data_shape = out_data_shape

            if get_world_size() > 1:
                data_filename = os.path.join(
                    self.out_dir, f"rank{get_rank()}_" + os.path.splitext(self.filename)[0] + ".zarr"
                )
            else:
                data_filename = os.path.join(
                    self.out_dir, os.path.splitext(self.filename)[0] + ".zarr"
                )

            # Adapt the crop_shape into the dataset axes order
            chunk_shape = order_dimensions(
                self.crop_shape,
                input_order="ZYXC",
                output_order=self.out_data_order,
                default_value=np.nan,
            )
            chunk_shape = tuple([int(val) if not np.isnan(val) else out_data_shape[i] for i, val in enumerate(chunk_shape)]) # type: ignore

            os.makedirs(self.out_dir, exist_ok=True)
            self.out_file = data_filename
            self.out_data = zarr.open_array(
                data_filename,
                shape=out_data_shape,
                mode="w",
                chunks=chunk_shape,  # type: ignore
                dtype=self.dtype_str,
            )

        insert_patch_in_efficient_file(
            data=self.out_data,
            patch=patch,
            patch_coords=patch_coords,
            data_axes_order=self.out_data_order,
            patch_axes_order="ZYXC",
        )

    def merge_zarr_parts_into_one(self):
        """Merge all parts of the Zarr data, created by each rank, into just one file."""
        # Creates the final Zarr dataset
        data_filename = os.path.join(self.out_dir, os.path.splitext(self.filename)[0] + ".zarr")
        final_data = zarr.open_array(
            data_filename,
            shape=self.out_data_shape,
            mode="w",
            chunks=self.crop_shape,  # type: ignore
            dtype=self.dtype_str,
        )

        for i in tqdm(range(get_world_size())):
            zarr_of_rank_filename = os.path.join(
                self.out_dir, f"rank{i}_" + os.path.splitext(self.filename)[0] + ".zarr"
            )
            print("[Rank {} ({})] Reading file {}".format(get_rank(), os.getpid(), zarr_of_rank_filename))
            data = zarr.open_array(zarr_of_rank_filename, mode="r")

            for z in range(math.ceil(data.shape[0] / self.crop_shape[0])):
                for y in range(math.ceil(data.shape[1] / self.crop_shape[1])):
                    for x in range(math.ceil(data.shape[2] / self.crop_shape[2])):
                        coords = PatchCoords(
                            z_start=z * self.crop_shape[0],
                            z_end=min((z + 1) * self.crop_shape[0], data.shape[0]),
                            y_start=y * self.crop_shape[1],
                            y_end=min((y + 1) * self.crop_shape[1], data.shape[1]),
                            x_start=x * self.crop_shape[2],
                            x_end=min((x + 1) * self.crop_shape[2], data.shape[2]),
                        )
                        patch = data[
                            coords.z_start : coords.z_end,
                            coords.y_start : coords.y_end,
                            coords.x_start : coords.x_end,
                        ]
                        patch = np.array(patch)

                        # If the patch contains something
                        if patch.max() != patch.min():
                            insert_patch_in_efficient_file(
                                data=final_data,
                                patch=patch,
                                patch_coords=coords,
                                data_axes_order=self.out_data_order,
                                patch_axes_order="ZYXC",
                                mode="add",
                            )

    def save_parallel_data_as_tif(self):
        """Save the final zarr into a tiff file."""
        final_zarr_file = os.path.join(self.out_dir, os.path.splitext(self.filename)[0] + ".zarr")
        if not os.path.exists(final_zarr_file):
            print(f"Couldn't load Zarr data for saving. File {final_zarr_file} not found!")
        else:
            data = np.array(zarr.open_array(final_zarr_file, mode="r"))
            data = ensure_3d_shape(data)
            save_tif(np.expand_dims(data, 0), self.out_dir, [os.path.splitext(self.filename)[0] + ".tif"], verbose=True)

    def close_open_files(self):
        """Close all files that may be open in the generator."""
        # Input data files
        if self.X_parallel_file is not None and isinstance(self.X_parallel_file, h5py.File):
            self.X_parallel_file.close()
        if self.Y_parallel_file is not None and isinstance(self.Y_parallel_file, h5py.File):
            self.Y_parallel_file.close()
        # Output data file
        if isinstance(self.out_file, h5py.File):
            self.out_file.close()

    def __len__(self):
        """
        Return the number of patches in the dataset.

        Returns
        -------
        int
            Number of patches.
        """
        return self.len
