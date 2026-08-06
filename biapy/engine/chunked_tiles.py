"""Post-processing of the "by chunks" inference while the patches are being predicted."""
from __future__ import annotations

import os
import math
import numpy as np
from typing import Dict, List, Optional, Tuple
from numpy.typing import NDArray

from biapy.data.chunked_writer import ChunkedZarrWriter
from biapy.data.dataset import PatchCoords
from biapy.utils.misc import get_rank, is_main_process


class ChunkedTileProcessor:
    """
    Group the predicted patches into tiles and post-process each tile as soon as it is complete.

    The prediction of a tile is kept in memory together with the ``DATA.TEST.PADDING`` of its border
    patches, which gives the workflow process the context of the neighbouring tiles. Once processed,
    that context is removed and only the tile is written, so the raw predictions never need to be
    stored.

    Parameters
    ----------
    workflow : Base_Workflow
        Workflow being run. It provides the per-tile processing and the output description.

    tgen : chunked_test_pair_data_generator
        Generator that yields the patches, which defines the tile grid.
    """

    def __init__(self, workflow, tgen):
        """Initialize the tile buffers and the writer of the post-processed output."""
        self.workflow = workflow
        self.cfg = workflow.cfg
        self.tgen = tgen
        self.padding = tuple(int(x) for x in self.cfg.DATA.TEST.PADDING)
        self.dims = (int(tgen.z_dim), int(tgen.y_dim), int(tgen.x_dim))
        self.open_tiles: Dict[int, Dict] = {}
        self.processed_tiles = 0

        out_vars = workflow.test_chunked_workflow_process_vars
        self.writer = ChunkedZarrWriter(
            out_dir=out_vars["out_dir"],
            filename=tgen.filename,
            data_shape=tgen.X_parallel_data.shape,
            data_axes_order=tgen.out_data_order,
            dtype_str=out_vars["dtype_str"],
            write_tile_zyx=tgen.tile_step,
        )

        # Consumed by the workflow steps that run after the prediction (e.g. instance merging)
        workflow.chunked_output_info = {
            "zarr_path": self.writer.path,
            "axes_order": tgen.out_data_order,
            "dims": self.dims,
            "tile_step": tuple(int(x) for x in tgen.tile_step),
            "tiles": (int(tgen.tiles_per_z), int(tgen.tiles_per_y), int(tgen.tiles_per_x)),
        }

        # Each worker fills one tile at a time, and each process (i.e. each GPU) has its own set of workers
        self.gpus = max(1, int(self.cfg.SYSTEM.NUM_GPUS))
        tiles_per_gpu = math.ceil(len(tgen.tile_ids) / self.gpus)
        self.tiles_in_flight = max(1, min(int(getattr(workflow.test_generator, "num_workers", 0)), tiles_per_gpu))
        # Biggest tile buffer, i.e. the one of a tile that is not at the border of the image
        self.max_buffer_shape = tuple(
            min(int(tgen.tile_step[i]) + 2 * self.padding[i], self.dims[i]) for i in range(3)
        )
        self._reported = False

    def _tile_buffer(self, tile_id: int, remaining: int, channels: int, dtype) -> Dict:
        """Return the buffer of a tile, allocating it the first time one of its patches arrives."""
        if tile_id in self.open_tiles:
            return self.open_tiles[tile_id]

        tile = self.tgen.tile_coords(tile_id)
        starts = (
            max(0, tile.z_start - self.padding[0]),
            max(0, tile.y_start - self.padding[1]),
            max(0, tile.x_start - self.padding[2]),
        )
        ends = (
            min(self.dims[0], tile.z_end + self.padding[0]),
            min(self.dims[1], tile.y_end + self.padding[1]),
            min(self.dims[2], tile.x_end + self.padding[2]),
        )
        shape = tuple(ends[i] - starts[i] for i in range(3))

        if not self._reported and is_main_process():
            self._reported = True
            per_tile = int(np.prod(self.max_buffer_shape)) * channels * np.dtype(dtype).itemsize
            # Post-processing a tile takes about two more tiles of working memory, and BiaPy itself around
            # 3 GiB per GPU. Measured to be a slight over-estimate, which is the safe side to be on.
            estimate = self.gpus * ((self.tiles_in_flight + 2) * per_tile + 3 * 2**30)
            print(
                f"Post-processing tiles of {tuple(int(x) for x in self.tgen.tile_step)} with a context of "
                f"{self.padding}: {per_tile / 2**20:.0f} MiB each ({self.max_buffer_shape} voxels, {channels} "
                f"channels). Up to {self.tiles_in_flight} of them are filled at the same time per GPU, so this job "
                f"needs around {estimate / 2**30:.1f} GiB of RAM with {self.gpus} GPU(s). "
                f"Output written to {self.writer.path}"
            )

        self.open_tiles[tile_id] = {
            "tile": tile,
            "starts": starts,
            "ends": ends,
            "data": np.zeros(shape + (channels,), dtype=dtype),
            # Voxels already taken from the patch they belong to, which no other patch can overwrite
            "owned": np.zeros(shape, dtype=bool),
            "remaining": remaining,
        }
        return self.open_tiles[tile_id]

    def _overlap(self, src_start: Tuple[int, ...], src_shape: Tuple[int, ...], buf: Dict):
        """Return the slices to copy a region into a tile buffer, or None if they do not overlap."""
        src_slices, dst_slices = [], []
        for axis in range(3):
            start = max(src_start[axis], buf["starts"][axis])
            end = min(src_start[axis] + src_shape[axis], buf["ends"][axis])
            if end <= start:
                return None, None
            src_slices.append(slice(start - src_start[axis], end - src_start[axis]))
            dst_slices.append(slice(start - buf["starts"][axis], end - buf["starts"][axis]))
        return tuple(src_slices), tuple(dst_slices)

    def add_patch(self, pred: NDArray, patch_coords: PatchCoords, tile_info: Tuple[int, int]):
        """
        Place the prediction of one patch into the buffer of its tile.

        The region the patch is responsible for is always taken from it, while the padding around it
        only fills the parts of the buffer that no patch of the tile is responsible for, i.e. the
        context around the tile.

        Parameters
        ----------
        pred : NDArray
            Prediction of the patch, padding included, in ``(z, y, x, c)`` order.

        patch_coords : PatchCoords
            Region the patch is responsible for.

        tile_info : tuple of int
            Identifier of the tile the patch belongs to and number of patches of that tile.
        """
        tile_id, patches_in_tile = int(tile_info[0]), int(tile_info[1])
        buf = self._tile_buffer(tile_id, patches_in_tile, pred.shape[-1], pred.dtype)

        # The prediction covers the region of the patch expanded by the padding, out of the data limits
        # at the borders of the image, where it was created by reflection
        padded_start = (
            patch_coords.z_start - self.padding[0],
            patch_coords.y_start - self.padding[1],
            patch_coords.x_start - self.padding[2],
        )
        src, dst = self._overlap(padded_start, pred.shape[:3], buf)
        if src is not None:
            free = ~buf["owned"][dst]
            buf["data"][dst][free] = pred[src][free]

        inner_start = (patch_coords.z_start, patch_coords.y_start, patch_coords.x_start)
        inner_shape = (
            patch_coords.z_end - patch_coords.z_start,
            patch_coords.y_end - patch_coords.y_start,
            patch_coords.x_end - patch_coords.x_start,
        )
        src, dst = self._overlap(inner_start, inner_shape, buf)
        if src is not None:
            src = tuple(
                slice(s.start + self.padding[axis], s.stop + self.padding[axis]) for axis, s in enumerate(src)
            )
            buf["data"][dst] = pred[src]
            buf["owned"][dst] = True

        buf["remaining"] -= 1
        if buf["remaining"] <= 0:
            self.flush(tile_id)

    def flush(self, tile_id: int):
        """Post-process a tile, write it and free its buffer."""
        buf = self.open_tiles.pop(tile_id, None)
        if buf is None:
            return

        self.processed_tiles += 1
        tile = buf["tile"]
        context = [
            [tile.z_start - buf["starts"][0], buf["ends"][0] - tile.z_end],
            [tile.y_start - buf["starts"][1], buf["ends"][1] - tile.y_end],
            [tile.x_start - buf["starts"][2], buf["ends"][2] - tile.x_end],
        ]

        if self.cfg.TEST.VERBOSE:
            print(f"[Rank {get_rank()} ({os.getpid()})] Processing tile {tile_id} {tile}")

        processed = self.workflow.after_one_chunk_workflow_process([buf["data"]], [tile], added_pad=[context])
        if processed is None or processed[0] is None:
            return

        out = processed[0]
        out = out[
            context[0][0] : out.shape[0] - context[0][1],
            context[1][0] : out.shape[1] - context[1][1],
            context[2][0] : out.shape[2] - context[2][1],
        ]
        self.writer.insert(out, tile)

    def flush_all(self):
        """Post-process the tiles that are still open, i.e. those missing some discarded patch."""
        for tile_id in list(self.open_tiles.keys()):
            self.flush(tile_id)

    def save_as_tif(self):
        """Save the post-processed output as a TIF."""
        self.writer.save_as_tif()
