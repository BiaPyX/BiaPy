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
import time
from scipy.ndimage import zoom as ndi_zoom
from typing import Tuple, Optional, Dict, List, Callable, Sequence
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
from biapy.utils.misc import get_world_size, get_rank, is_main_process
from biapy.data.dataset import PatchCoords
from biapy.data.norm import normalize_image, normalize_mask
from biapy.data.roi_mask import load_roi_mask

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

    norm_module : Dict
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

    zoom_factor : sequence of float, optional
        Per-axis zoom factor (``DATA.PREPROCESS.ZOOM.ZOOM_FACTOR``) matching ``input_axes``, applied to
        each patch with ``scipy.ndimage.zoom`` before it is fed to the model. Useful when the input data
        has a different resolution than the one used in training. ``None`` (the default) disables it.
    """

    def __init__(
        self,
        sample_to_process: Dict,
        norm_module: Dict,
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
        z_start: int = -1,
        z_end: int = -1,
        roi_mask_path: str = "",
        roi_mask_axes_order: str = "",
        patches_per_tile: Tuple[int, int, int] = (1, 1, 1),
        zoom_factor: Optional[Sequence[float]] = None,
    ):
        """
        Initialize the chunked_test_pair_data_generator.

        Parameters
        ----------
        sample_to_process : dict
            Dictionary containing sample data and file pointers.
        norm_module : Dict
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
        z_start : int, optional
            First Z slice (inclusive) to process. -1 means start from the beginning.
        z_end : int, optional
            Last Z slice (exclusive) to process. -1 means process until the end.
        roi_mask_path : str, optional
            Path to a region of interest mask. Patches not overlapping it are not extracted nor
            predicted. It does not need to match the data shape.
        roi_mask_axes_order : str, optional
            Order of the axes of the ROI mask. Defaults to the axes order of the image.
        patches_per_tile : tuple of int, optional
            Patches grouped into each workflow process tile, on each axis.
        zoom_factor : sequence of float, optional
            Per-axis zoom factor matching ``input_axes``, applied to each patch before inference.
        """
        super(chunked_test_pair_data_generator).__init__()
        self.zoom_enable = zoom_factor is not None
        self.zoom_zyxc = (
            tuple(order_dimensions(tuple(zoom_factor), input_order=input_axes, output_order="ZYXC", default_value=1))
            if self.zoom_enable
            else (1, 1, 1, 1)
        )
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
        # The declared axes order (e.g. DATA.TEST.INPUT_IMG_AXES_ORDER) describes the raw axes of a
        # chunked H5/Zarr dataset, which may include a leading 'T' dimension. Formats that can't be
        # read in chunks (e.g. .tif) are instead loaded fully into memory and normalized to a plain
        # (Z, Y, X, C) array with no 'T' axis (see ensure_3d_shape), so reconcile the declared axes
        # with the actual data rank the same way ensure_3d_shape does, or order_dimensions() below
        # will index past the end of the shape tuple.
        if "T" in input_axes and len(input_axes) != self.X_parallel_data.ndim:
            input_axes = input_axes.replace("T", "")
        if (
            self.Y_parallel_data is not None
            and "T" in mask_input_axes
            and len(mask_input_axes) != self.Y_parallel_data.ndim
        ):
            mask_input_axes = mask_input_axes.replace("T", "")
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
                "Z Axis problem: {} greater than {} (you can reduce 'DATA.PATCH_SIZE' in that axis). Shape provided: {} (axis order: {})".format(
                    self.crop_shape[0], self.z_dim, self.X_parallel_data.shape, self.input_axes
                )
            )
        if self.crop_shape[1] > self.y_dim:
            raise ValueError(
                "Y Axis problem: {} greater than {} (you can reduce 'DATA.PATCH_SIZE' in that axis). Data shape provided: {} (axis order: {})".format(
                    self.crop_shape[1], self.y_dim, self.X_parallel_data.shape, self.input_axes
                )
            )
        if self.crop_shape[2] > self.x_dim:
            raise ValueError(
                "X Axis problem: {} greater than {} (you can reduce 'DATA.PATCH_SIZE' in that axis). Shape provided: {} (axis order: {})".format(
                    self.crop_shape[2], self.x_dim, self.X_parallel_data.shape, self.input_axes
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

        self.z_dim_out, self.y_dim_out, self.x_dim_out = self._scale_zyx((self.z_dim, self.y_dim, self.x_dim))
        self.step_z_out, self.step_y_out, self.step_x_out = self._scale_zyx((self.step_z, self.step_y, self.step_x))

        # Clamp Z range to valid chunk indices
        effective_z_start = 0 if z_start == -1 else z_start
        effective_z_end = self.z_dim if z_end == -1 else z_end
        self.z_vol_start = math.ceil(effective_z_start / self.step_z)
        self.z_vol_end = min(math.ceil(effective_z_end / self.step_z), self.vols_per_z)
        self.vols_per_z_effective = self.z_vol_end - self.z_vol_start

        self.total_vols = self.vols_per_z_effective * self.vols_per_y * self.vols_per_x

        # Filter the tiles here, and not while iterating, so the ones left are evenly spread among
        # ranks/workers by the sampler.
        self.roi_mask = (
            load_roi_mask(
                roi_mask_path,
                data_shape_zyx=(self.z_dim, self.y_dim, self.x_dim),
                sample_filename=self.filename,
                axes_order=roi_mask_axes_order if roi_mask_axes_order else self.input_axes,
            )
            if roi_mask_path
            else None
        )
        roi_msg = ""
        if self.roi_mask is None:
            self.vol_ids = list(range(self.total_vols))
        else:
            self.vol_ids = [
                vol_id
                for vol_id in range(self.total_vols)
                if self.roi_mask.patch_is_inside(self._patch_coords(vol_id)[4])
            ]
            if len(self.vol_ids) == 0:
                raise ValueError(
                    f"No patch of sample {self.filename} overlaps the ROI mask set in 'DATA.TEST.ROI_MASK.PATH', so "
                    "there is nothing to predict. Check that the mask is not empty and that it covers the same field "
                    "of view as the image (its shape is mapped to the image shape by scaling each axis)."
                )
            roi_msg = (
                f"ROI mask: {len(self.vol_ids)} of {self.total_vols} patches to process "
                f"({self.total_vols - len(self.vol_ids)} discarded as outside the ROI). "
            )

        self.len = len(self.vol_ids)

        # Group the patches into the tiles the workflow process works on. Tiles are groups of consecutive
        # patches of the global grid, so each patch belongs to exactly one tile.
        self.patches_per_tile = tuple(max(1, int(x)) for x in patches_per_tile)
        self.tile_step = (
            self.step_z * self.patches_per_tile[0],
            self.step_y * self.patches_per_tile[1],
            self.step_x * self.patches_per_tile[2],
        )
        self.tile_step_out = self._scale_zyx(self.tile_step)
        self.tiles_per_z = math.ceil(self.vols_per_z / self.patches_per_tile[0])
        self.tiles_per_y = math.ceil(self.vols_per_y / self.patches_per_tile[1])
        self.tiles_per_x = math.ceil(self.vols_per_x / self.patches_per_tile[2])

        self.patches_of_tile: Dict[int, List[int]] = {}
        for vol_id in self.vol_ids:
            z_local, y, x = np.unravel_index(vol_id, (self.vols_per_z_effective, self.vols_per_y, self.vols_per_x))
            tile_id = int(
                np.ravel_multi_index(
                    (
                        (int(z_local) + self.z_vol_start) // self.patches_per_tile[0],
                        int(y) // self.patches_per_tile[1],
                        int(x) // self.patches_per_tile[2],
                    ),
                    (self.tiles_per_z, self.tiles_per_y, self.tiles_per_x),
                )
            )
            self.patches_of_tile.setdefault(tile_id, []).append(vol_id)
        self.tile_ids = sorted(self.patches_of_tile.keys())

        if is_main_process():
            z_range_msg = ""
            if z_start != -1 or z_end != -1:
                z_range_msg = (
                    f"Z range: slices [{effective_z_start}, {effective_z_end}) → "
                    f"chunks [{self.z_vol_start}, {self.z_vol_end}) of {self.vols_per_z} total. "
                )
            print(
                f"Initialized chunked_test_pair_data_generator with sample {self.filename} and shape {self.X_parallel_data.shape}.\n"
                f"Crop shape: {self.crop_shape}, padding: {self.padding}. Input axes: {self.input_axes}. Mask input axes: {self.mask_input_axes}.\n"
                f"Output data axes order: {self.out_data_order}. {z_range_msg}{roi_msg}\n"
                f"Workflow process tile: {self.tile_step} ({self.patches_per_tile} patches), "
                f"{len(self.tile_ids)} tiles to process."
                ""
            )

    def _scale_zyx(self, values: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Scale a ``(z, y, x)`` tuple from input-data to output-data resolution via ``zoom_factor``."""
        if not self.zoom_enable:
            return values
        return tuple(int(round(v * f)) for v, f in zip(values, self.zoom_zyxc[:3]))

    def _to_output_coords(self, coords: PatchCoords) -> PatchCoords:
        """Scale a :class:`PatchCoords`, given in input-data resolution, to the output-data one."""
        if not self.zoom_enable:
            return coords
        zz, zy, zx = self.zoom_zyxc[:3]
        return PatchCoords(
            z_start=int(round(coords.z_start * zz)),
            z_end=int(round(coords.z_end * zz)),
            y_start=int(round(coords.y_start * zy)),
            y_end=int(round(coords.y_end * zy)),
            x_start=int(round(coords.x_start * zx)),
            x_end=int(round(coords.x_end * zx)),
        )

    def data_shape_for_output(self) -> Tuple[int, ...]:
        """
        Return ``X_parallel_data``'s shape with its Z/Y/X dimensions scaled to the output resolution.

        Matches ``DATA.PREPROCESS.ZOOM.ZOOM_FACTOR`` when enabled; identical to the input shape otherwise.
        """
        shape = list(self.X_parallel_data.shape)
        for axis_char, out_dim in zip("ZYX", (self.z_dim_out, self.y_dim_out, self.x_dim_out)):
            if axis_char in self.input_axes:
                shape[self.input_axes.index(axis_char)] = out_dim
        return tuple(int(v) for v in shape)

    def tile_coords(self, tile_id: int) -> PatchCoords:
        """
        Return the coordinates of the region a tile is responsible for, i.e. without the padding.

        Coordinates are given in output-data resolution (see :meth:`_to_output_coords`).

        Parameters
        ----------
        tile_id : int
            Tile identifier within the ``(tiles_per_z, tiles_per_y, tiles_per_x)`` grid.

        Returns
        -------
        PatchCoords
            Coordinates of the tile within the data.
        """
        assert isinstance(self.z_dim, int) and isinstance(self.x_dim, int) and isinstance(self.y_dim, int)
        z, y, x = np.unravel_index(tile_id, (self.tiles_per_z, self.tiles_per_y, self.tiles_per_x))
        z0 = int(z) * self.tile_step[0]
        y0 = int(y) * self.tile_step[1]
        x0 = int(x) * self.tile_step[2]
        return self._to_output_coords(
            PatchCoords(
                z_start=z0,
                z_end=min(z0 + self.tile_step[0], self.z_dim),
                y_start=y0,
                y_end=min(y0 + self.tile_step[1], self.y_dim),
                x_start=x0,
                x_end=min(x0 + self.tile_step[2], self.x_dim),
            )
        )

    def rank_workload(self, num_workers: int, world_size: int, rank: int) -> Tuple[int, int]:
        """
        Return the patches and tiles that a given rank will process, to report the progress.

        It follows the same distribution the sampler does in :meth:`__iter__`, where the tiles are
        shared among the workers of every rank, repeating some of them when they do not divide evenly.

        Parameters
        ----------
        num_workers : int
            Workers of the data loader that reads from this generator.

        world_size : int
            Number of processes among which the tiles are distributed.

        rank : int
            Process to return the workload of.

        Returns
        -------
        patches, tiles : int
            Patches and tiles the given rank processes.
        """
        workers = max(1, int(num_workers))
        replicas = workers * max(1, int(world_size))
        total = len(self.tile_ids)
        if total == 0:
            return 0, 0

        padded = math.ceil(total / replicas) * replicas
        order = [i % total for i in range(padded)]
        mine = set()
        for worker in range(workers):
            mine.update(order[rank * workers + worker :: replicas])

        return sum(len(self.patches_of_tile[self.tile_ids[i]]) for i in mine), len(mine)

    def _patch_coords(self, vol_id: int) -> Tuple[int, int, int, PatchCoords, PatchCoords]:
        """
        Translate a tile identifier into the coordinates of the patch it represents.

        Parameters
        ----------
        vol_id : int
            Tile identifier within the ``(vols_per_z_effective, vols_per_y, vols_per_x)`` grid.

        Returns
        -------
        z, y, x : int
            Position of the tile in the grid.

        patch_to_extract : PatchCoords
            Coordinates of the region to read, padding included and clipped to the image limits.

        real_patch_in_data : PatchCoords
            Coordinates of the region without the padding, where the prediction is written back.
        """
        assert isinstance(self.z_dim, int) and isinstance(self.x_dim, int) and isinstance(self.y_dim, int)
        z_local, y, x = np.unravel_index(vol_id, (self.vols_per_z_effective, self.vols_per_y, self.vols_per_x))
        z = int(z_local) + self.z_vol_start
        y = int(y)
        x = int(x)

        patch_to_extract = PatchCoords(
            z_start=max(0, z * self.step_z - self.padding[0]),
            z_end=min((z + 1) * self.step_z + self.padding[0], self.z_dim),
            y_start=max(0, y * self.step_y - self.padding[1]),
            y_end=min((y + 1) * self.step_y + self.padding[1], self.y_dim),
            x_start=max(0, x * self.step_x - self.padding[2]),
            x_end=min((x + 1) * self.step_x + self.padding[2], self.x_dim),
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

        return z, y, x, patch_to_extract, real_patch_in_data


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
            if extract == "image" or (extract == "mask" and self.norm_module["target_type"] == "image"):
                if data.shape[-1] == 1:
                    data = np.repeat(data, 3, axis=-1)

        if self.zoom_enable:
            data = ndi_zoom(data, self.zoom_zyxc, order=0, mode="nearest")
            pad_to_add = [
                [int(round(side * self.zoom_zyxc[axis])) for side in sides]
                for axis, sides in enumerate(pad_to_add)
            ]

        return data, pad_to_add

    def __iter__(self):
        """
        Iterate over the generator.

        Returns
        -------
        vol_id : int
            Patch identifier.

        tile_info : tuple of int
            Identifier of the tile the patch belongs to and number of patches of that tile.

        img : NDArray
            X patch of data to process.

        mask : NDArray
            Y yatch of data to process.

        real_patch_in_data : PatchCoords
            Coordinates of the patch in the data.

        pad_to_add : List of list of ints
            Padding added to the patch in order to satisfy the crop shape expected.

        xnorm_info : dict
            Extra information of the normalization applied to ``img``.
        """
        assert isinstance(self.z_dim, int) and isinstance(self.x_dim, int) and isinstance(self.y_dim, int)
        worker_info = torch.utils.data.get_worker_info()  # type: ignore
        n_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        world_size = get_world_size()
        process_rank = get_rank()

        # Tiles, and not patches, are distributed, so all the patches of a tile are predicted by the same
        # process and it can be post-processed as soon as its last patch is predicted.
        sampler = DistributedSampler(
            self.tile_ids,
            num_replicas=(n_workers * world_size),
            rank=(process_rank * n_workers + worker_id),
            shuffle=False,
        )

        for sampler_id in sampler:
            tile_id = self.tile_ids[int(sampler_id)]
            patches_of_tile = self.patches_of_tile[tile_id]
            for vol_id in patches_of_tile:
                yield from self._prepare_patch(vol_id, (tile_id, len(patches_of_tile)))

    def _prepare_patch(self, vol_id: int, tile_info: Tuple[int, int]):
        """
        Read, filter and normalize one patch, yielding nothing when it is filtered out.

        Parameters
        ----------
        vol_id : int
            Patch identifier.

        tile_info : tuple of int
            Identifier of the tile the patch belongs to and number of patches of that tile.
        """
        mask = None
        z, y, x, patch_to_extract, real_patch_in_data = self._patch_coords(vol_id)

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
            img, xnorm_info = normalize_image(img, norm_module=self.norm_module)
            if mask is not None:
                mask, _ = normalize_mask(
                    np.array(mask),
                    norm_module=self.norm_module,
                    n_classes=self.n_classes,
                    ignore_index=self.ignore_index,
                    instance_problem=self.instance_problem
                )
                assert isinstance(mask, np.ndarray)

            yield vol_id, tile_info, img, mask, self._to_output_coords(real_patch_in_data), added_pad, xnorm_info

    def _shared_zarr_path(self) -> str:
        base = os.path.splitext(self.filename)[0]
        return os.path.join(self.out_dir, f"{base}.zarr")

    def _compute_out_shape(self, patch: NDArray) -> Tuple[int, ...]:
        out_shape = list(self.data_shape_for_output())

        if "C" not in self.input_axes:
            out_shape = list(out_shape) + [patch.shape[-1]]
        else:
            out_shape[self.input_axes.index("C")] = patch.shape[-1]

        return tuple(int(v) for v in out_shape)

    def _compute_out_chunks(self, out_data_shape: Tuple[int, ...], patch: NDArray) -> Tuple[int, ...]:
        """
        Compute the chunk shape for the output Zarr file. Chunking is aligned to output tiles 
        (step sizes) to guarantee safe concurrent writes.

        Parameters
        ----------
        out_data_shape : tuple of int
            Shape of the output data.
        
        patch : NDArray
            Sample patch to process.
        
        Returns
        -------
        tuple of int
            Chunk shape for the output data.
        """
        write_tile_zyxc = (
            self.step_z_out,
            self.step_y_out,
            self.step_x_out,
            patch.shape[-1],
        )

        # Adapt into dataset axes order (e.g. ZYXC -> whatever self.out_data_order is)
        chunk_shape = order_dimensions(
            write_tile_zyxc,
            input_order="ZYXC",
            output_order=self.out_data_order,
            default_value=np.nan,
        )
        chunk_shape = tuple(
            int(v) if not np.isnan(v) else int(out_data_shape[i])
            for i, v in enumerate(chunk_shape)
        )
        return tuple(int(v) for v in chunk_shape)

    def _open_or_create_shared_out(self, out_path: str, out_shape: Tuple[int, ...], out_chunks: Tuple[int, ...]):
        """
        Open or create the shared output Zarr file.

        Parameters
        ----------
        out_path : str
            Path to the output Zarr file.
            
        out_shape : tuple of int
            Shape of the output data.

        out_chunks : tuple of int
            Chunk shape of the output data.
        """
        os.makedirs(self.out_dir, exist_ok=True)

        # Fast path: already opened in this worker process
        if self.out_data is not None:
            return

        # Try a few times to survive races where another process is creating metadata
        last_err = None
        for _ in range(20):
            try:
                # Create new (fail if exists)
                self.out_data = zarr.open(
                    out_path,
                    mode="w-",
                    shape=out_shape,
                    chunks=out_chunks,
                    dtype=self.dtype_str,
                    zarr_format=3,
                )
                self.out_file = out_path
                return
            except Exception as e:
                last_err = e
                # If it already exists (or creation raced), open read/write
                try:
                    existing = zarr.open(out_path, mode="r+", zarr_format=3)
                    if tuple(existing.shape) != tuple(out_shape):
                        raise RuntimeError(
                            f"Existing Zarr at {out_path} has shape {existing.shape}, but this run expects "
                            f"{out_shape}. It is likely left over from a previous run with a different "
                            "'DATA.PATCH_SIZE' or 'DATA.PREPROCESS.ZOOM.ZOOM_FACTOR'. Remove it and re-run."
                        )
                    self.out_data = existing
                    self.out_file = out_path
                    return
                except FileNotFoundError:
                    # Possibly metadata not fully written yet by creator
                    time.sleep(0.05)

        raise RuntimeError(f"Could not create/open shared Zarr at {out_path}. Last error: {last_err}")

    def insert_patch_in_file(self, patch: NDArray, patch_coords: PatchCoords):
        """
        Insert patch into the output parallel file. Chunking is aligned to output tiles 
        (step sizes) to guarantee safe concurrent writes.
        
        Parameters
        ----------
        patch : int
            Sample index counter.

        patch_coords : PatchCoords
            Whether its the first time a sample is loaded to prevent normalizing it.

        """
        out_path = self._shared_zarr_path()

        if self.out_file is None or self.out_data is None:
            out_shape = self._compute_out_shape(patch)
            out_chunks = self._compute_out_chunks(out_shape, patch)
            self.out_data_shape = out_shape
            self._open_or_create_shared_out(out_path, out_shape, out_chunks)

        # Insert the patch
        insert_patch_in_efficient_file(
            data=self.out_data,
            patch=patch,
            patch_coords=patch_coords,
            data_axes_order=self.out_data_order,
            patch_axes_order="ZYXC",
            mode="replace",
        )

    def save_parallel_data_as_tif(self):
        """Save the final zarr into a tiff file."""
        final_zarr_file = self._shared_zarr_path()
        if not os.path.exists(final_zarr_file):
            print(f"Couldn't load Zarr data for saving. File {final_zarr_file} not found!")
        else:
            data = np.array(zarr.open(final_zarr_file, mode="r"))
            data = ensure_3d_shape(data)
            save_tif(
                np.expand_dims(data, 0),
                self.out_dir,
                [os.path.splitext(self.filename)[0] + ".tif"],
                verbose=True,
            )


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
