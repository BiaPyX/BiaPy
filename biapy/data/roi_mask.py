"""
Region of interest (ROI) mask support for inference (``DATA.TEST.ROI_MASK``).

Provides :class:`CoarseROIMask`, which restricts the inference to a region defined by a mask that
does not need to match the image shape: it is kept at its own resolution and the image coordinates
are mapped into it. Used to discard the patches outside the ROI in the "by chunks" inference
(:meth:`CoarseROIMask.patch_is_inside`) and to zero the prediction outside it in the normal one
(:meth:`CoarseROIMask.apply`).
"""
from __future__ import annotations

import os
import math
import numpy as np
from typing import Optional, Sequence, Tuple
from numpy.typing import NDArray

from biapy.data.dataset import PatchCoords
from biapy.data.data_3D_manipulation import looks_like_hdf5
from biapy.utils.misc import os_walk_clean, is_main_process


def mask_to_zyx(mask: NDArray, axes_order: str, name: str = "") -> NDArray:
    """
    Reorder a mask into ``(z, y, x)``, collapsing the ``'T'`` and ``'C'`` axes if present.

    Parameters
    ----------
    mask : NDArray
        Mask as read from disk.

    axes_order : str
        Order of the axes of ``mask``, e.g. ``'ZYX'``, ``'XYZ'`` or ``'TZCYX'`` in 3D and ``'YX'``
        in 2D. Axes declared but not present in the array (usually ``'T'``, and ``'C'`` for a mask
        saved without channels) are ignored, following the same convention used for the test images.

    name : str, optional
        Path/description of the mask. Only used in the error messages.

    Returns
    -------
    NDArray
        Boolean mask in ``(z, y, x)`` order. A 2D mask gets a singleton ``z`` axis.
    """
    where = f" ({name})" if name else ""
    axes = axes_order.upper()

    # The declared order describes the file format and may include axes the mask does not have
    for optional_axis in ("T", "C"):
        if len(axes) > mask.ndim and optional_axis in axes:
            axes = axes.replace(optional_axis, "")

    if len(axes) != mask.ndim:
        raise ValueError(
            f"The ROI mask{where} has {mask.ndim} dimensions (shape {mask.shape}) but the axes order given for it "
            f"('{axes_order}') describes {len(axes)}. Set 'DATA.TEST.ROI_MASK.AXES_ORDER' accordingly."
        )

    # Collapse the axes that carry no spatial information
    if "T" in axes:
        mask = mask.take(0, axis=axes.index("T"))
        axes = axes.replace("T", "")
    if "C" in axes:
        mask = mask.max(axis=axes.index("C"))
        axes = axes.replace("C", "")

    # A 2D mask is handled as a 3D one with a single Z slice
    if "Z" not in axes:
        mask = np.expand_dims(mask, 0)
        axes = "Z" + axes

    if sorted(axes) != ["X", "Y", "Z"]:
        raise ValueError(
            f"The axes order given for the ROI mask{where} ('{axes_order}') must contain 'Y' and 'X' axes, plus 'Z' "
            "in 3D (and, optionally, 'T' and 'C')"
        )

    return (mask > 0).transpose(axes.index("Z"), axes.index("Y"), axes.index("X"))


class CoarseROIMask:
    """
    Region of interest defined by a (possibly coarse) binary mask.

    The mask is mapped to the image grid by axis-wise scaling factors, so it can be of any shape:
    coarser, equal or finer than the data.

    Parameters
    ----------
    mask : NDArray
        Mask where anything ``> 0`` counts as ROI.

    data_shape_zyx : tuple of int
        Shape of the data the mask refers to, in ``(z, y, x)`` order. In 2D, use ``(1, y, x)``.

    axes_order : str, optional
        Order of the axes of ``mask``, e.g. ``'ZYX'``, ``'XYZ'`` or ``'TZCYX'`` in 3D and ``'YX'``
        in 2D. ``'T'`` and ``'C'`` axes, if present, are collapsed.

    is_3d : bool, optional
        Whether the data the mask is applied to is 3D. Only used by :meth:`apply`, to know the
        spatial axes of the arrays it receives.

    name : str, optional
        Description of where the mask comes from. Only used in the printed messages.
    """

    def __init__(
        self,
        mask: NDArray,
        data_shape_zyx: Sequence[int],
        axes_order: str = "ZYX",
        is_3d: bool = True,
        name: str = "",
    ):
        """Initialize the ROI mask and precompute the image -> mask scaling factors."""
        mask = mask_to_zyx(np.asarray(mask), axes_order, name=name)
        if len(data_shape_zyx) != 3:
            raise ValueError(f"'data_shape_zyx' must have 3 values and not {len(data_shape_zyx)}")

        self.is_3d = is_3d
        self.mask = mask
        self.mask_shape = tuple(int(x) for x in self.mask.shape)
        self.data_shape = tuple(int(x) for x in data_shape_zyx)
        self.name = name
        # Image voxels -> mask voxels. Kept as floats: the mask does not need to divide the data shape exactly.
        self.scales = tuple(m / d for m, d in zip(self.mask_shape, self.data_shape))

    def block_for(self, coords: PatchCoords) -> Tuple[slice, slice, slice]:
        """
        Return the slices of the mask covered by the given patch coordinates.

        The block holds at least one mask voxel per axis, so patches smaller than a mask voxel are
        resolved against the one they fall in.
        """
        starts = (coords.z_start, coords.y_start, coords.x_start)
        ends = (coords.z_end, coords.y_end, coords.x_end)

        block = []
        for axis in range(3):
            scale = self.scales[axis]
            size = self.mask_shape[axis]
            lo = int(math.floor(starts[axis] * scale))
            hi = int(math.ceil(ends[axis] * scale))
            lo = max(0, min(lo, size - 1))
            hi = max(lo + 1, min(hi, size))
            block.append(slice(lo, hi))
        return block[0], block[1], block[2]

    def patch_is_inside(self, coords: PatchCoords) -> bool:
        """Whether the patch at ``coords`` covers any foreground voxel of the region of interest."""
        zs, ys, xs = self.block_for(coords)
        return bool(self.mask[zs, ys, xs].any())

    def expanded_to(self, shape_zyx: Sequence[int]) -> NDArray:
        """
        Return the mask resampled (nearest neighbour) to the given ``(z, y, x)`` shape.

        The mask is stretched over the shape given, which does not need to be a multiple of it.
        """
        index_per_axis = [
            np.minimum((np.arange(size) * m) // max(size, 1), m - 1)
            for size, m in zip(shape_zyx, self.mask_shape)
        ]
        return self.mask[np.ix_(*index_per_axis)]

    def apply(self, data: NDArray) -> NDArray:
        """
        Zero out everything outside the region of interest.

        Parameters
        ----------
        data : NDArray
            Data to mask: ``(y, x, c)``/``(b, y, x, c)`` in 2D and ``(z, y, x, c)``/``(b, z, y, x, c)``
            in 3D. The mask is stretched over its spatial shape, so it may be of any size.

        Returns
        -------
        NDArray
            ``data`` with the values outside the ROI set to 0.
        """
        spatial_ndim = 3 if self.is_3d else 2
        if data.ndim == spatial_ndim + 1:
            spatial_shape = data.shape[:-1]
        elif data.ndim == spatial_ndim + 2:
            spatial_shape = data.shape[1:-1]
        else:
            raise ValueError(
                "Data to mask with the ROI mask has {} dimensions (shape {}) but {} or {} were expected for {} data "
                "(spatial axes plus channel, with an optional leading batch axis)".format(
                    data.ndim, data.shape, spatial_ndim + 1, spatial_ndim + 2, "3D" if self.is_3d else "2D"
                )
            )

        mask = self.expanded_to((1,) + tuple(spatial_shape) if not self.is_3d else tuple(spatial_shape))
        if not self.is_3d:
            mask = mask[0]

        return data * np.expand_dims(mask, -1)

    def __repr__(self) -> str:
        """Return a short description of the mask and the mapping it defines."""
        return (
            f"CoarseROIMask(mask_shape={self.mask_shape}, data_shape={self.data_shape}, "
            f"image_voxels_per_mask_voxel={tuple(round(1 / s, 2) for s in self.scales)}, "
            f"foreground={self.mask.mean() * 100:.2f}%)"
        )


def find_roi_mask_file(path: str, sample_filename: Optional[str] = None) -> str:
    """
    Resolve which mask file to use for a given test sample.

    Parameters
    ----------
    path : str
        Path to a mask file or to a directory holding the masks.

    sample_filename : str, optional
        File name of the test sample the mask is for. Used to pick the right mask when ``path``
        is a directory with more than one file.

    Returns
    -------
    str
        Path of the mask file to load.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"ROI mask path not found: {path}")

    if os.path.isfile(path) or path.endswith((".zarr", ".n5")):
        return path

    _, dirs, files = next(os_walk_clean(path))
    # Zarr/N5 masks are directories, so they must be collected apart from the plain files
    candidates = sorted(
        [os.path.join(path, x) for x in files]
        + [os.path.join(path, x) for x in dirs if x.endswith((".zarr", ".n5"))]
    )

    if len(candidates) == 0:
        raise FileNotFoundError(f"No ROI mask file found in {path}")
    if len(candidates) == 1:
        return candidates[0]

    # More than one mask: use the one named as the sample
    if sample_filename:
        sample_base = os.path.splitext(os.path.basename(sample_filename))[0]
        for candidate in candidates:
            if os.path.splitext(os.path.basename(candidate))[0] == sample_base:
                return candidate

    raise ValueError(
        "Found {} files in the ROI mask directory {} but none of them is named as the sample being "
        "processed ({}). Provide one mask per sample, named as the sample, or a single mask file to "
        "be used for all the samples.".format(len(candidates), path, sample_filename)
    )


def load_roi_mask(
    path: str,
    data_shape_zyx: Sequence[int],
    sample_filename: Optional[str] = None,
    axes_order: str = "ZYX",
    is_3d: bool = True,
    verbose: bool = True,
) -> CoarseROIMask:
    """
    Load the ROI mask to use for a test sample.

    Parameters
    ----------
    path : str
        Path to a mask file or to a directory holding the masks.

    data_shape_zyx : tuple of int
        Shape, in ``(z, y, x)`` order, of the data the mask must be mapped onto.

    sample_filename : str, optional
        File name of the test sample the mask is for.

    axes_order : str, optional
        Order of the axes of the mask file.

    is_3d : bool, optional
        Whether the data the mask is applied to is 3D.

    verbose : bool, optional
        Whether to print the loaded mask information.

    Returns
    -------
    CoarseROIMask
        Mask ready to filter patches or to be applied to a prediction.
    """
    # Imported here to avoid a circular import (data_manipulation imports from this package tree)
    from biapy.data.data_manipulation import imread

    mask_file = find_roi_mask_file(path, sample_filename)
    # Read the mask as it is stored: read_img_as_ndarray() guesses the axes order of 3D data from the
    # shape (smallest axis first), which would silently transpose a coarse mask.
    if mask_file.endswith((".zarr", ".n5")) or looks_like_hdf5(mask_file):
        from biapy.data.data_3D_manipulation import read_chunked_data

        _, mask = read_chunked_data(mask_file)
        mask = np.array(mask)
    else:
        mask = imread(mask_file)[0]
    assert isinstance(mask, np.ndarray)
    roi_mask = CoarseROIMask(mask, data_shape_zyx, axes_order=axes_order, is_3d=is_3d, name=mask_file)

    if verbose and is_main_process():
        print(f"ROI mask loaded from {mask_file}: {roi_mask}")

    return roi_mask
