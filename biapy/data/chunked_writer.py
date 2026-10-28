"""Zarr writer shared by the processes that create a large output file patch by patch."""
from __future__ import annotations

import os
import shutil
import numpy as np
from typing import Optional, Sequence
from numpy.typing import NDArray

from biapy.data.data_3D_manipulation import ensure_3d_shape, insert_patch_in_efficient_file
from biapy.data.data_manipulation import save_tif
from biapy.data.dataset import PatchCoords
from biapy.data import ome_zarr


def output_zarr_path(out_dir: str, filename: str) -> str:
    """Path of the Zarr file created in ``out_dir`` for the sample ``filename``."""
    return os.path.join(out_dir, os.path.splitext(filename)[0] + ".zarr")


def finish_output(path: str, axes: str, pyramid_levels: int = 1, save_as_tif: bool = False, verbose: bool = True):
    """
    Finish a Zarr file once all its data is written: create its lower resolution levels and, if asked, save it as a
    TIF next to it.

    Parameters
    ----------
    path : str
        Path to the Zarr file.

    axes : str
        Axes of its data, e.g. ``'CZYX'``.

    pyramid_levels : int, optional
        Number of resolution levels of the OME-Zarr image (``DATA.OME_ZARR.PYRAMID_LEVELS``). Not used for files
        written by older BiaPy versions, which are not OME-Zarr images.

    save_as_tif : bool, optional
        Whether to save the file as a TIF too (``TEST.BY_CHUNKS.SAVE_OUT_TIF``). It needs to fit in memory.

    verbose : bool, optional
        Whether to print where the TIF is saved.
    """
    if not os.path.exists(path):
        # Some processes do not write a file (e.g. detection saves the points as CSV files)
        if save_as_tif:
            print(f"Couldn't load Zarr data for saving. File {path} not found!")
        return
    if ome_zarr.image_info(path) is not None:
        ome_zarr.build_pyramid(path, pyramid_levels)
    if save_as_tif:
        data = ensure_3d_shape(np.array(ome_zarr.open_array(path)), data_axes_order=axes)
        save_tif(
            np.expand_dims(data, 0),
            os.path.dirname(path),
            [os.path.splitext(os.path.basename(path))[0] + ".tif"],
            verbose=verbose,
        )


class ChunkedZarrWriter:
    """
    Write patches into an OME-Zarr image shared by all the ranks/workers of a job.

    The file is created with the first patch inserted, as its number of channels defines the output
    shape, and its chunks are aligned to the write tile so concurrent writes never touch the same
    chunk. Its axes are those of the input data ordered as OME-Zarr requires, with a channel axis
    (e.g. ``'ZYX'`` -> ``'CZYX'``).

    Parameters
    ----------
    out_dir : str
        Directory to create the Zarr file into.

    filename : str
        Name of the sample being processed. The Zarr file takes its name with the extension changed.

    data_shape : tuple of int
        Shape of the output data, in ``data_axes_order`` order. Its channels are taken from the patches.

    data_axes_order : str
        Order of the axes of ``data_shape``, e.g. ``'TZCYX'``.

    dtype_str : str
        Data type of the Zarr file to create.

    write_tile_zyx : tuple of int
        Shape, in ``(z, y, x)`` order, of the region written on each call to :meth:`insert`.

    source : OmeZarrImage, optional
        OME-Zarr image the output is created from, whose voxel size is carried over.

    ome_version : str or float, optional
        OME-Zarr version to write (``DATA.OME_ZARR.VERSION``).
    """

    def __init__(
        self,
        out_dir: str,
        filename: str,
        data_shape: Sequence[int],
        data_axes_order: str,
        dtype_str: str,
        write_tile_zyx: Sequence[int],
        source: Optional[ome_zarr.OmeZarrImage] = None,
        ome_version: str | float = ome_zarr.DEFAULT_VERSION,
    ):
        """Initialize the writer without creating the output file yet."""
        self.out_dir = out_dir
        self.filename = filename
        self.data_shape = dict(zip(data_axes_order, (int(x) for x in data_shape)))
        self.axes_order = ome_zarr.ngff_axes_order(data_axes_order)
        self.dtype_str = dtype_str
        self.write_tile_zyx = tuple(int(x) for x in write_tile_zyx)
        self.source = source
        self.ome_version = ome_version
        self.data = None
        self.path = output_zarr_path(out_dir, filename)

    def _create_or_open(self, patch: NDArray):
        """Create the output file or open the one another rank created."""
        if self.data is not None:
            return
        os.makedirs(self.out_dir, exist_ok=True)
        sizes = dict(self.data_shape, C=int(patch.shape[-1]))
        out_shape = tuple(sizes.get(a, 1) for a in self.axes_order)
        # Chunks aligned to the write tile (whole channels), so concurrent writes are safe
        tile = dict(zip("ZYX", self.write_tile_zyx), C=int(patch.shape[-1]))
        out_chunks = tuple(tile.get(a, 1) for a in self.axes_order)
        scale, units = ome_zarr.output_scale(self.axes_order, out_shape, self.source)
        self.data = ome_zarr.create_image(
            self.path,
            out_shape,
            self.axes_order,
            self.dtype_str,
            out_chunks,
            scale=scale,
            units=units,
            version=self.ome_version,
        )

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
            data_axes_order=self.axes_order,
            patch_axes_order="ZYXC",
            mode="replace",
        )

    def remove_if_misaligned(self):
        """
        Remove the output file left by a previous run if its chunks are not aligned to the tiles this run writes (e.g.
        the instances were created while predicting, and now from the saved predictions, which uses other tiles). The
        ranks write different tiles at once, so tiles sharing a chunk would overwrite each other's data. Call it on the
        main rank before any rank writes.
        """
        info = ome_zarr.image_info(self.path)
        if info is None:  # not there, or written by an older BiaPy version (create_image reports it)
            return
        chunks = ome_zarr.open_array(self.path).chunks
        tile = dict(zip("ZYX", self.write_tile_zyx))
        if all(chunks[info.axes.index(a)] == tile[a] for a in "ZYX" if a in info.axes):
            return
        print(
            f"Removing {self.path}, left over from a previous run: its chunks {tuple(chunks)} are not aligned to the "
            f"tiles written now {self.write_tile_zyx} (z, y, x)"
        )
        shutil.rmtree(self.path)

    def created(self) -> bool:
        """Whether the output file was created, i.e. at least one patch was written."""
        return self.data is not None or os.path.exists(self.path)
