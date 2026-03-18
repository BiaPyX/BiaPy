"""
Chunked workflow data generator for processing data without augmentation.

This module contains the `chunked_workflow_process_generator` class, which is an
iterable dataset designed to handle large 3D image data by processing it in smaller
chunks.
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
from typing import Tuple 
from numpy.typing import NDArray

from biapy.data.data_3D_manipulation import (
    extract_patch_from_efficient_file,
    ensure_3d_shape,
    insert_patch_in_efficient_file,
    order_dimensions,
    looks_like_hdf5,
)
from biapy.data.data_manipulation import load_img_data, save_tif, extract_patch_within_image
from biapy.utils.misc import get_world_size, get_rank, is_main_process
from biapy.data.dataset import PatchCoords

class chunked_workflow_process_generator(IterableDataset):
    """
    Chunked generator for post-processing already-predicted volumetric data
    WITHOUT padding/overlap. Tiles are exactly `crop_shape` (except edge tiles
    which may be smaller). This is designed to be faster than the original
    prediction chunking pipeline.

    Typical use:
      - Read raw prediction Zarr/H5 (model_predictions)
      - For each tile: load raw logits/probs patch, postprocess (e.g. binarize),
        and write it into a new output Zarr aligned to the same tile grid.

    Parameters
    ----------
    model_predictions : str
        Path to the model predictions to process.

    input_axes : str
        Input axes expected in the X data to be load.

    crop_shape : tuple of int
        Shape of the patches to extract.

    path_to_gt_data : str
        Path to the ground truth data.

    out_dir: str
        Output directory to save the predicted data into.

    dtype_str : str
        Data type to be used in the Zarr/H5 created.
    """

    def __init__(
        self,
        model_predictions: str,
        input_axes: str,
        crop_shape: Tuple[int, ...],
        out_dir: str,
        dtype_str: str = "float32",
    ):
        """
        Initialize the chunked_workflow_process_generator.

        Parameters
        ----------
        model_predictions : str
            Path to the model predictions to process.
        input_axes : str
            Axes order for input data.
        crop_shape : tuple of int
            Shape of the patches to extract.
        out_dir : str
            Output directory for results.
        dtype_str : str, optional
            Data type for output.
        """
        super(chunked_workflow_process_generator).__init__()
        self.model_predictions = model_predictions
        self.filename = os.path.basename(self.model_predictions)
        self.file_type = "h5" if looks_like_hdf5(self.filename) else "zarr"
        self.dir = os.path.dirname(self.model_predictions)
        
        self.X_parallel_data, self.X_parallel_file = load_img_data(
            self.model_predictions,
            is_3d=True,
            data_within_zarr_path=None,
        )

        self.input_axes = input_axes
        self.dtype_str = dtype_str
        self.out_dir = out_dir

        # ---- Normalize crop_shape to include correct channel dimension if needed ----
        # We only tile over Z/Y/X. Channel is carried through as-is.
        # If crop_shape includes a trailing channel entry, we ignore it for tiling.
        crop_shape = tuple(int(v) for v in crop_shape)
        if len(crop_shape) >= 4:
            crop_shape_zyx = crop_shape[:3]
        else:
            crop_shape_zyx = crop_shape
        if len(crop_shape_zyx) != 3:
            raise ValueError(f"crop_shape must be 3D (Z,Y,X) or 4D (...,C). Got: {crop_shape}")
        
        self.crop_shape_zyx = crop_shape_zyx

        # Output zarr handle (shared among workers/processes via filesystem)
        self.out_data = None
        self.out_file = None

        # Output data order: ensure there is a channel dim in output ordering
        # (predictions typically have channels; if not, C is appended)
        if "C" not in self.input_axes:
            self.out_data_order = self.input_axes + "C"
        else:
            self.out_data_order = self.input_axes

        # ---- Dimensions ----
        # We assume data is at least 3D in Z/Y/X. If it has extra axes (T, etc),
        # order_dimensions will place missing axes as np.nan. We only use Z/Y/X.
        _, self.z_dim, _, self.y_dim, self.x_dim = order_dimensions(self.X_parallel_data.shape, self.input_axes)
        assert isinstance(self.z_dim, int) and isinstance(self.y_dim, int) and isinstance(self.x_dim, int)

        if self.crop_shape_zyx[0] > self.z_dim:
            raise ValueError(f"Z Axis problem: {self.crop_shape_zyx[0]} greater than {self.z_dim}")
        if self.crop_shape_zyx[1] > self.y_dim:
            raise ValueError(f"Y Axis problem: {self.crop_shape_zyx[1]} greater than {self.y_dim}")
        if self.crop_shape_zyx[2] > self.x_dim:
            raise ValueError(f"X Axis problem: {self.crop_shape_zyx[2]} greater than {self.x_dim}")

        # ---- No-overlap tiling ----
        self.step_z = self.crop_shape_zyx[0]
        self.step_y = self.crop_shape_zyx[1]
        self.step_x = self.crop_shape_zyx[2]

        self.vols_per_z = math.ceil(self.z_dim / self.step_z)
        self.vols_per_y = math.ceil(self.y_dim / self.step_y)
        self.vols_per_x = math.ceil(self.x_dim / self.step_x)

        self.len = self.vols_per_z * self.vols_per_y * self.vols_per_x

        if is_main_process():
            print(
                f"Initialized chunked_workflow_process_generator with sample {self.filename} and shape {self.X_parallel_data.shape}.\n"
                f"Crop shape: {self.crop_shape_zyx}. Input axes: {self.input_axes}. Output data axes order: {self.out_data_order}. "
                ""
            )


    def _extract_patch(self, patch_coords: PatchCoords) -> NDArray:
        """
        Extract a patch from X_parallel_data according to patch_coords.
        No padding is applied.
        """
        data_to_process = self.X_parallel_data
        if not isinstance(data_to_process, np.ndarray):
            patch = extract_patch_from_efficient_file(data_to_process, patch_coords, self.input_axes)
        else:
            patch = extract_patch_within_image(data_to_process, patch_coords, is_3d=True)

        # Ensure channel last if input has no channel (add singleton)
        if patch.ndim == 3:
            patch = np.expand_dims(patch, -1)

        return patch

    def __iter__(self):
        """
        Yield:
          vol_id: int
          patch: NDArray (Z,Y,X,C) patch (edge patches may be smaller on Z/Y/X)
          patch_in_data: PatchCoords (coords where this patch belongs)
        """
        worker_info = torch.utils.data.get_worker_info()  # type: ignore
        n_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        world_size = get_world_size()
        process_rank = get_rank()

        sampler = DistributedSampler(
            self,
            num_replicas=(n_workers * world_size),
            rank=(process_rank * n_workers + worker_id),
            shuffle=False,
        )
        assert isinstance(self.z_dim, int) and isinstance(self.y_dim, int) and isinstance(self.x_dim, int)
        
        for vol_id in sampler:
            z, y, x = np.unravel_index(vol_id, (self.vols_per_z, self.vols_per_y, self.vols_per_x))
            z = int(z)
            y = int(y)
            x = int(x)

            z0 = z * self.step_z
            y0 = y * self.step_y
            x0 = x * self.step_x

            z1 = min(z0 + self.step_z, self.z_dim)
            y1 = min(y0 + self.step_y, self.y_dim)
            x1 = min(x0 + self.step_x, self.x_dim)

            patch_in_data = PatchCoords(
                z_start=z0, z_end=z1,
                y_start=y0, y_end=y1,
                x_start=x0, x_end=x1,
            )

            patch = self._extract_patch(patch_in_data)
            yield vol_id, patch, patch_in_data

    def _shared_zarr_path(self) -> str:
        """
        Output path for the Zarr file created by this generator.
        You can change naming here if you want a suffix like _bin.zarr.
        """
        base = os.path.splitext(self.filename)[0]
        return os.path.join(self.out_dir, f"{base}.zarr")

    def _compute_out_shape(self, patch: NDArray) -> Tuple[int, ...]:
        """
        Output shape mirrors input shape, but ensures the channel dim matches patch.shape[-1].
        """
        out_shape = list(self.X_parallel_data.shape)

        if "C" not in self.input_axes:
            out_shape = list(out_shape) + [patch.shape[-1]]
        else:
            out_shape[self.input_axes.index("C")] = patch.shape[-1]

        return tuple(int(v) for v in out_shape)

    def _compute_out_chunks(self, out_data_shape: Tuple[int, ...], out_channels: int) -> Tuple[int, ...]:
        """
        Chunk shape aligned to the tile grid: (step_z, step_y, step_x, C) in ZYXC,
        then mapped to out_data_order.
        """
        write_tile_zyxc = (self.step_z, self.step_y, self.step_x, int(out_channels))

        chunk_shape = order_dimensions(
            write_tile_zyxc,
            input_order="ZYXC",
            output_order=self.out_data_order,
            default_value=np.nan,
        )

        # Any axes not present get full size along that axis
        chunk_shape = tuple(
            int(v) if not np.isnan(v) else int(out_data_shape[i])
            for i, v in enumerate(chunk_shape)
        )
        return tuple(int(v) for v in chunk_shape)

    def _open_or_create_shared_out(self, out_path: str, out_shape: Tuple[int, ...], out_chunks: Tuple[int, ...]):
        os.makedirs(self.out_dir, exist_ok=True)

        # Already opened in this worker
        if self.out_data is not None:
            return

        last_err = None
        for _ in range(20):
            try:
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
                # If exists or raced, try open r+
                try:
                    self.out_data = zarr.open(out_path, mode="r+", zarr_format=3)
                    self.out_file = out_path
                    return
                except Exception:
                    time.sleep(0.05)

        raise RuntimeError(f"Could not create/open shared Zarr at {out_path}. Last error: {last_err}")

    def insert_patch_in_file(self, patch: NDArray, patch_coords: PatchCoords):
        """
        Insert patch into the output Zarr at patch_coords (no padding removal required).
        `patch` must be ZYXC.
        """
        out_path = self._shared_zarr_path()

        if self.out_file is None or self.out_data is None:
            out_shape = self._compute_out_shape(patch)
            out_chunks = self._compute_out_chunks(out_shape, out_channels=patch.shape[-1])
            self.out_data_shape = out_shape
            self._open_or_create_shared_out(out_path, out_shape, out_chunks)

        if self.out_data is not None:
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
            return

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
        if self.X_parallel_file is not None and isinstance(self.X_parallel_file, h5py.File):
            self.X_parallel_file.close()
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
