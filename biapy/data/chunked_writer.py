"""Zarr writer shared by the processes that create a large output file patch by patch."""
from __future__ import annotations

import os
import time
import zarr
import numpy as np
from typing import Optional, Sequence, Tuple
from numpy.typing import NDArray

from biapy.data.data_3D_manipulation import (
    ensure_3d_shape,
    insert_patch_in_efficient_file,
    order_dimensions,
)
from biapy.data.data_manipulation import save_tif
from biapy.data.dataset import PatchCoords


class ChunkedZarrWriter:
    """
    Write patches into a Zarr file shared by all the ranks/workers of a job.

    The file is created with the first patch inserted, as its number of channels defines the output
    shape, and its chunks are aligned to the write tile so concurrent writes never touch the same
    chunk.

    Parameters
    ----------
    out_dir : str
        Directory to create the Zarr file into.

    filename : str
        Name of the sample being processed. The Zarr file takes its name with the extension changed.

    data_shape : tuple of int
        Shape of the input data, in ``data_axes_order`` order.

    data_axes_order : str
        Order of the axes of the input data, e.g. ``'TZCYX'``.

    dtype_str : str
        Data type of the Zarr file to create.

    write_tile_zyx : tuple of int
        Shape, in ``(z, y, x)`` order, of the region written on each call to :meth:`insert`.
    """

    def __init__(
        self,
        out_dir: str,
        filename: str,
        data_shape: Sequence[int],
        data_axes_order: str,
        dtype_str: str,
        write_tile_zyx: Sequence[int],
    ):
        """Initialize the writer without creating the output file yet."""
        self.out_dir = out_dir
        self.filename = filename
        self.data_shape = tuple(int(x) for x in data_shape)
        self.data_axes_order = data_axes_order if "C" in data_axes_order else data_axes_order + "C"
        self.dtype_str = dtype_str
        self.write_tile_zyx = tuple(int(x) for x in write_tile_zyx)
        self.data = None
        self.path = os.path.join(out_dir, os.path.splitext(filename)[0] + ".zarr")

    def _out_shape(self, patch: NDArray) -> Tuple[int, ...]:
        """Return the shape of the file to create, taking the channels from the given patch."""
        out_shape = list(self.data_shape)
        if "C" not in self.data_axes_order[: len(self.data_shape)] or len(self.data_shape) < len(
            self.data_axes_order
        ):
            out_shape = out_shape + [patch.shape[-1]]
        else:
            out_shape[self.data_axes_order.index("C")] = patch.shape[-1]
        return tuple(int(v) for v in out_shape)

    def _out_chunks(self, out_shape: Tuple[int, ...], patch: NDArray) -> Tuple[int, ...]:
        """Return the chunk shape, aligned to the write tile so concurrent writes are safe."""
        chunk_shape = order_dimensions(
            self.write_tile_zyx + (patch.shape[-1],),
            input_order="ZYXC",
            output_order=self.data_axes_order,
            default_value=np.nan,
        )
        return tuple(
            int(v) if not np.isnan(v) else int(out_shape[i]) for i, v in enumerate(chunk_shape)
        )

    def _create_or_open(self, patch: NDArray):
        """Create the output file or open the one another rank created."""
        if self.data is not None:
            return

        os.makedirs(self.out_dir, exist_ok=True)
        out_shape = self._out_shape(patch)
        out_chunks = self._out_chunks(out_shape, patch)

        last_err = None
        for _ in range(20):
            try:
                self.data = zarr.open(
                    self.path,
                    mode="w-",
                    shape=out_shape,
                    chunks=out_chunks,
                    dtype=self.dtype_str,
                    zarr_format=3,
                )
                return
            except Exception as e:
                last_err = e
                try:
                    self.data = zarr.open(self.path, mode="r+", zarr_format=3)
                    return
                except Exception:
                    # Possibly metadata not fully written yet by the creator
                    time.sleep(0.05)

        raise RuntimeError(f"Could not create/open shared Zarr at {self.path}. Last error: {last_err}")

    def insert(self, patch: NDArray, patch_coords: PatchCoords):
        """
        Write a patch into the output file.

        Parameters
        ----------
        patch : NDArray
            Patch to write, in ``(z, y, x, c)`` order.

        patch_coords : PatchCoords
            Coordinates the patch must be written into.
        """
        self._create_or_open(patch)
        insert_patch_in_efficient_file(
            data=self.data,
            patch=patch,
            patch_coords=patch_coords,
            data_axes_order=self.data_axes_order,
            patch_axes_order="ZYXC",
            mode="replace",
        )

    def save_as_tif(self, verbose: bool = True):
        """Save the whole output file as a TIF. It needs to fit in memory."""
        if not os.path.exists(self.path):
            print(f"Couldn't load Zarr data for saving. File {self.path} not found!")
            return
        data = ensure_3d_shape(np.array(zarr.open(self.path, mode="r")))
        save_tif(
            np.expand_dims(data, 0),
            self.out_dir,
            [os.path.splitext(self.filename)[0] + ".tif"],
            verbose=verbose,
        )

    def created(self) -> bool:
        """Whether the output file was created, i.e. at least one patch was written."""
        return self.data is not None or os.path.exists(self.path)
