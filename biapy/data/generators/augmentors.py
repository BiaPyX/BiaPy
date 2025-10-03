"""
Augmentation utilities for image and mask data in deep learning workflows.

This module provides a variety of data augmentation functions for images and masks,
including cutout, cutblur, cutmix, cutnoise, misalignment, cropping, flipping,
rotation, zoom, gamma/contrast adjustment, blurring, dropout, elastic deformation,
shear, shift, and more. These augmentations are designed to improve model robustness
and generalization for both 2D and 3D data formats.
"""

import cv2
import random
import math
import numpy as np
from PIL import Image
from skimage.transform import resize
from skimage.draw import line
from skimage.exposure import adjust_gamma
from skimage.filters import gaussian
from scipy.ndimage import binary_dilation as binary_dilation_scipy
from scipy.ndimage import rotate
from typing import Tuple, Union, Optional, List
from numpy.typing import NDArray
from scipy.ndimage import median_filter, shift as shift_nd
from skimage.transform import AffineTransform, ProjectiveTransform, warp


def cutout(
    img: NDArray,
    mask: NDArray,
    z_size: int,
    nb_iterations: Tuple[int, int] = (1, 3),
    size: Tuple[float, float] = (0.2, 0.4),
    cval: int = 0,
    res_relation: Tuple[float, ...] = (1.0, 1.0),
    apply_to_mask: bool = False,
) -> Tuple[NDArray, NDArray]:
    """
    Apply augmentation using Cutout technique.

    Cutout data augmentation presented in `Improved Regularization of Convolutional Neural Networks with Cutout <https://arxiv.org/pdf/1708.04552.pdf>`_.

    Parameters
    ----------
    img : Numpy array
        Image to transform. E.g. ``(y, x, channels)`` for ``2D`` or ``(y, x, z, channels)`` for ``3D``.

    mask : Numpy array
        Mask to transform. E.g. ``(y, x, channels)`` for ``2D`` or ``(y, x, z, channels)`` for ``3D``.

    z_size : int
        Size of z dimension. Used for 3D images as the z axis has been merged with the channels. Set to -1 to when
        do not want to be applied.

    nb_iterations : tuple of ints, optional
        Number of areas to fill the image with. E.g. ``(1, 3)``.

    size : tuple of floats, optional
        Range to choose the size of the areas to create.

    cval : int, optional
        Value to fill the area with.

    res_relation: tuple of floats, optional
        Relation between axis resolution in ``(x,y,z)``. E.g. ``(1,1,0.27)`` for anisotropic data of
        8umx8umx30um resolution.

    apply_to_mask : boolean, optional
        To apply cutout to the mask.

    Returns
    -------
    out : Numpy array
        Transformed image. E.g. ``(y, x, channels)`` for ``2D`` or ``(y, x, z, channels)`` for ``3D``.

    mask : Numpy array
        Transformed mask. E.g. ``(y, x, channels)`` for ``2D`` or ``(y, x, z, channels)`` for ``3D``.

    Examples
    --------
    Calling this function with ``nb_iterations=(1,3)``, ``size=(0.05,0.3)``,
    ``apply_to_mask=False`` may result in:

    +----------------------------------------------+----------------------------------------------+
    | .. figure:: ../../../img/orig_cutout.png     | .. figure:: ../../../img/orig_cutout_mask.png|
    |   :width: 80%                                |   :width: 80%                                |
    |   :align: center                             |   :align: center                             |
    |                                              |                                              |
    |   Input image                                |   Corresponding mask                         |
    +----------------------------------------------+----------------------------------------------+
    | .. figure:: ../../../img/cutout.png          | .. figure:: ../../../img/cutout_mask.png     |
    |   :width: 80%                                |   :width: 80%                                |
    |   :align: center                             |   :align: center                             |
    |                                              |                                              |
    |   Augmented image                            |   Augmented mask                             |
    +----------------------------------------------+----------------------------------------------+

    The grid is painted for visualization purposes.
    """
    assert img.ndim in [3, 4], f"Image must be 3D or 4D, got shape {img.shape}"
    assert mask.ndim in [3, 4], f"Mask must be 3D or 4D, got shape {mask.shape}"
    assert len(nb_iterations) == 2 and nb_iterations[0] <= nb_iterations[1]
    assert len(size) == 2 and 0.0 < size[0] <= size[1] <= 1.0

    # ensure (x,y,z) factors available
    rx = float(res_relation[0]) if len(res_relation) >= 1 else 1.0
    ry = float(res_relation[1]) if len(res_relation) >= 2 else 1.0
    rz = float(res_relation[2]) if len(res_relation) >= 3 else 1.0

    out = img.copy()
    m_out = mask.copy()

    # spatial dims
    if img.ndim == 3:            # (y, x, c)
        H, W = img.shape[:2]
        Z = None
    else:                        # (z, y, x, c)
        Z, H, W = img.shape[0], img.shape[1], img.shape[2]

    # how many cutouts
    it = int(np.random.randint(nb_iterations[0], nb_iterations[1] + 1))

    # helper to clamp sizes to [1, max]
    def _clamp_size(n, mx):
        return max(1, min(int(n), int(mx)))

    # fill values cast to dtype
    fill_img = np.array(cval, dtype=img.dtype)
    fill_mask = np.array(0, dtype=mask.dtype)

    for _ in range(it):
        frac = random.uniform(size[0], size[1])

        # rectangle size in (y, x)
        y_size = _clamp_size(round(H * frac * ry), H)
        x_size = _clamp_size(round(W * frac * rx), W)

        # random top-left in-bounds (inclusive)
        cy = 0 if H == y_size else np.random.randint(0, H - y_size + 1)
        cx = 0 if W == x_size else np.random.randint(0, W - x_size + 1)

        if img.ndim == 4:
            # decide z extent
            if z_size != -1:
                # sample a z block size scaled by rz
                assert Z is not None, "Z dimension not found in 4D image"
                z_block = _clamp_size(round(Z * frac * rz), Z)
                z0 = 0 if Z == z_block else np.random.randint(0, Z - z_block + 1)
                z_slice = slice(z0, z0 + z_block)
            else:
                z_slice = slice(None)

            # apply to image & (optionally) mask
            out[z_slice, cy:cy + y_size, cx:cx + x_size, :] = fill_img
            if apply_to_mask:
                m_out[z_slice, cy:cy + y_size, cx:cx + x_size, :] = fill_mask

        else:
            # 2D: apply across all channels
            out[cy:cy + y_size, cx:cx + x_size, :] = fill_img
            if apply_to_mask:
                m_out[cy:cy + y_size, cx:cx + x_size, :] = fill_mask

    return out, m_out


def cutblur(
    img: NDArray,
    size: Tuple[float, float] = (0.2, 0.4),
    down_ratio_range: Tuple[int, int] = (2, 8),
    only_inside: bool = True,
) -> NDArray:
    """
    Apply CutBlur data augmentation.

    CutBlur data augmentation introduced in `Rethinking Data Augmentation for Image Super-resolution: 
    A Comprehensive Analysis and a New Strategy <https://arxiv.org/pdf/2004.00448.pdf>`_ and adapted 
    from https://github.com/clovaai/cutblur .

    Parameters
    ----------
    img : Numpy array
        Image to transform. E.g. ``(y, x, channels)`` for ``2D`` or ``(y, x, z, channels)`` for ``3D``.

    size : float, optional
        Size of the region to transform.

    down_ratio_range : tuple of ints, optional
        Downsampling ratio range to be applied. E.g. ``(2, 8)``.

    only_inside : bool, optional
        If ``True`` only the region inside will be modified (cut LR into HR image). If ``False`` the ``50%`` of the
        times the region inside will be modified (cut LR into HR image) and the other ``50%`` the inverse will be
        done (cut HR into LR image). See Figure 1 of the official `paper <https://arxiv.org/pdf/2004.00448.pdf>`_.

    Returns
    -------
    out : Numpy array
        Transformed image. E.g. ``(y, x, channels)`` for ``2D`` or ``(y, x, z, channels)`` for ``3D``.

    Examples
    --------
    Calling this function with ``size=(0.2,0.4)``, ``down_ratio_range=(2,8)``,
    ``only_inside=True`` may result in:

    +--------------------------------------------+--------------------------------------------+
    | .. figure:: ../../../img/orig_cutblur.png  | .. figure:: ../../../img/cutblur.png       |
    |   :width: 80%                              |   :width: 80%                              |
    |   :align: center                           |   :align: center                           |
    |                                            |                                            |
    |   Input image                              |   Augmented image                          |
    +--------------------------------------------+--------------------------------------------+
    | .. figure:: ../../../img/orig_cutblur2.png | .. figure:: ../../../img/cutblur2.png      |
    |   :width: 80%                              |   :width: 80%                              |
    |   :align: center                           |   :align: center                           |
    |                                            |                                            |
    |   Input image                              |   Augmented image                          |
    +--------------------------------------------+--------------------------------------------+

    The grid and the red square are painted for visualization purposes.
    """
    assert img.ndim in (3, 4), f"Image must be 3D or 4D, got shape {img.shape}"
    assert len(size) == 2 and 0.0 < size[0] <= size[1] <= 1.0, f"Invalid size range: {size}"
    assert len(down_ratio_range) == 2 and down_ratio_range[0] >= 1 and down_ratio_range[0] <= down_ratio_range[1], \
        f"Invalid down_ratio_range: {down_ratio_range}"

    if img.size == 0:
        return img

    # Spatial dims (H, W) and channel count
    if img.ndim == 3:      # (y, x, c)
        H, W, C = img.shape
        Z = 1
    else:                  # (z, y, x, c)
        Z, H, W, C = img.shape

    # Sample patch fraction and clamp to at least 1 pixel
    frac = float(random.uniform(size[0], size[1]))
    y_size = max(1, int(round(H * frac)))
    x_size = max(1, int(round(W * frac)))

    # Random top-left (inclusive) so the patch stays inside bounds
    cy = 0 if H == y_size else np.random.randint(0, H - y_size + 1)
    cx = 0 if W == x_size else np.random.randint(0, W - x_size + 1)

    # Downsample ratio and shapes
    down_ratio = int(np.random.randint(down_ratio_range[0], down_ratio_range[1] + 1))
    dsH_full, dsW_full = max(1, H // down_ratio), max(1, W // down_ratio)
    dsH_patch, dsW_patch = max(1, y_size // down_ratio), max(1, x_size // down_ratio)

    # inside flag
    inside = True if only_inside else (random.uniform(0, 1) < 0.5)

    out = img.copy()
    orig_dtype = img.dtype

    def _resize(arr: NDArray, out_shape_hw_c: Tuple[int, int, int], order: int, aa: bool) -> NDArray:
        # skimage.transform.resize expects (H, W, C)
        res = resize(
            arr, out_shape_hw_c, order=order, mode="reflect",
            clip=True, preserve_range=True, anti_aliasing=aa
        )
        # cast back to original dtype when needed
        if orig_dtype.kind != "f":
            res = res.astype(orig_dtype, copy=False)
        return res

    if img.ndim == 3:
        if inside:
            # LR->HR only for the selected patch
            patch = img[cy:cy + y_size, cx:cx + x_size, :]
            down = _resize(patch, (dsH_patch, dsW_patch, C), order=1, aa=True)
            up   = _resize(down,  (y_size,     x_size,     C), order=0, aa=False)
            out[cy:cy + y_size, cx:cx + x_size, :] = up
        else:
            # Whole image to LR->HR, then paste original HR patch back
            down = _resize(img, (dsH_full, dsW_full, C), order=1, aa=True)
            up   = _resize(down, (H, W, C),           order=0, aa=False)
            out = up
            out[cy:cy + y_size, cx:cx + x_size, :] = img[cy:cy + y_size, cx:cx + x_size, :]
        return out

    # 4D: apply per z-slice with shared region and ratio
    for z in range(Z):
        if inside:
            patch = img[z, cy:cy + y_size, cx:cx + x_size, :]
            down = _resize(patch, (dsH_patch, dsW_patch, C), order=1, aa=True)
            up   = _resize(down,  (y_size,     x_size,     C), order=0, aa=False)
            out[z, cy:cy + y_size, cx:cx + x_size, :] = up
        else:
            full = img[z]
            down = _resize(full, (dsH_full, dsW_full, C), order=1, aa=True)
            up   = _resize(down, (H, W, C),              order=0, aa=False)
            out[z] = up
            out[z, cy:cy + y_size, cx:cx + x_size, :] = img[z, cy:cy + y_size, cx:cx + x_size, :]

    return out


def cutmix(
    im1: NDArray,
    im2: NDArray,
    mask1: NDArray,
    mask2: NDArray,
    heat1: NDArray | None,
    heat2: NDArray | None,
    size: Tuple[float, float] = (0.2, 0.4),
) -> Tuple[NDArray, NDArray, NDArray | None]:
    """
    Apply Cutmix data augmentation.

    Cutmix augmentation introduced in `CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features <https://arxiv.org/abs/1905.04899>`_. 
    With this augmentation a region of the image sample is filled with a given second image. This implementation is used for semantic segmentation so the masks of 
    the images are also needed. It assumes that the images are of the same shape.

    Parameters
    ----------
    im1 : Numpy array
        Image to transform. E.g. ``(y, x, channels)`` for ``2D`` or ``(y, x, z, channels)`` for ``3D``.

    im2 : Numpy array
        Image to paste into the region of ``im1``. E.g. ``(y, x, channels)`` for ``2D`` or ``(y, x, z, channels)`` for ``3D``.

    mask1 : Numpy array
        Mask to transform (belongs to ``im1``). E.g. ``(y, x, channels)`` for ``2D`` or ``(y, x, z, channels)`` for ``3D``.

    mask2 : Numpy array
        Mask to paste into the region of ``mask1``. E.g. ``(y, x, channels)`` for ``2D`` or ``(y, x, z, channels)`` for ``3D``.

    heat1 : Numpy array or None
        Heatmap to transform (belongs to ``im1``). E.g. ``(y, x, channels)`` for ``2D`` or ``(y, x, z, channels)`` for ``3D``.
        If ``None``, no heatmap is used.
    
    heat2 : Numpy array or None
        Heatmap to paste into the region of ``heat1``. E.g. ``(y, x, channels)`` for ``2D`` or ``(y, x, z, channels)`` for ``3D``.
        If ``None``, no heatmap is used.

    size : tuple of floats, optional
        Range to choose the size of the areas to transform. E.g. ``(0.2, 0.4)``.

    Returns
    -------
    out : Numpy array
        Transformed image. E.g. ``(y, x, channels)`` for ``2D`` or ``(y, x, z, channels)`` for ``3D``.

    m_out : Numpy array
        Transformed mask. E.g. ``(y, x, channels)`` for ``2D`` or ``(y, x, z, channels)`` for ``3D``.

    h_out : Numpy array or None
        Transformed heatmap. E.g. ``(y, x, channels)`` for ``2D`` or ``(y, x, z, channels)`` for ``3D``.

    Examples
    --------
    Calling this function with ``size=(0.2,0.4)`` may result in:

    +----------------------------------------------+----------------------------------------------+
    | .. figure:: ../../../img/orig_cutmix.png     | .. figure:: ../../../img/orig_cutmix_mask.png|
    |   :width: 80%                                |   :width: 80%                                |
    |   :align: center                             |   :align: center                             |
    |                                              |                                              |
    |   Input image                                |   Corresponding mask                         |
    +----------------------------------------------+----------------------------------------------+
    | .. figure:: ../../../img/cutmix.png          | .. figure:: ../../../img/cutmix_mask.png     |
    |   :width: 80%                                |   :width: 80%                                |
    |   :align: center                             |   :align: center                             |
    |                                              |                                              |
    |   Augmented image                            |   Augmented mask                             |
    +----------------------------------------------+----------------------------------------------+

    The grid is painted for visualization purposes.
    """
    assert im1.ndim in (3, 4) and im2.ndim == im1.ndim, f"Shape mismatch: {im1.shape} vs {im2.shape}"
    assert mask1.ndim == im1.ndim and mask2.ndim == im1.ndim, "Mask dims must match image dims"
    assert im1.shape[:-1] == im2.shape[:-1] == mask1.shape[:-1] == mask2.shape[:-1], "All inputs must share shape"
    assert len(size) == 2 and 0.0 < size[0] <= size[1] <= 1.0, f"Invalid size range: {size}"

    if im1.size == 0:
        return im1, mask1, heat1

    # Spatial dims (H, W)
    if im1.ndim == 3:  # (y, x, c)
        H, W, C = im1.shape
        Z = 1
    else:              # (z, y, x, c)
        Z, H, W, C = im1.shape

    # Sample patch size (at least 1×1)
    frac = float(random.uniform(size[0], size[1]))
    y_size = max(1, int(round(H * frac)))
    x_size = max(1, int(round(W * frac)))

    # Random top-left corners (inclusive bounds so the patch fits)
    im1cy = 0 if H == y_size else np.random.randint(0, H - y_size + 1)
    im1cx = 0 if W == x_size else np.random.randint(0, W - x_size + 1)
    im2cy = 0 if H == y_size else np.random.randint(0, H - y_size + 1)
    im2cx = 0 if W == x_size else np.random.randint(0, W - x_size + 1)

    out = im1.copy()
    m_out = mask1.copy()
    h_out = heat1.copy() if heat1 is not None else None

    if im1.ndim == 3:
        # Vectorized over channels
        out[im1cy:im1cy + y_size, im1cx:im1cx + x_size, :] = \
            im2[im2cy:im2cy + y_size, im2cx:im2cx + x_size, :]

        m_out[im1cy:im1cy + y_size, im1cx:im1cx + x_size, :] = \
            mask2[im2cy:im2cy + y_size, im2cx:im2cx + x_size, :]

        if h_out is not None and heat2 is not None:
            h_out[im1cy:im1cy + y_size, im1cx:im1cx + x_size, :] = \
                heat2[im2cy:im2cy + y_size, im2cx:im2cx + x_size, :]
    else:
        # Apply the same (y,x) patch across all z-slices
        out[:, im1cy:im1cy + y_size, im1cx:im1cx + x_size, :] = \
            im2[:, im2cy:im2cy + y_size, im2cx:im2cx + x_size, :]

        m_out[:, im1cy:im1cy + y_size, im1cx:im1cx + x_size, :] = \
            mask2[:, im2cy:im2cy + y_size, im2cx:im2cx + x_size, :]
        
        if h_out is not None and heat2 is not None:
            h_out[:, im1cy:im1cy + y_size, im1cx:im1cx + x_size, :] = \
                heat2[:, im2cy:im2cy + y_size, im2cx:im2cx + x_size, :]

    return out, m_out, h_out


def cutnoise(
    img: NDArray,
    scale: Tuple[float, float] = (0.1, 0.2),
    nb_iterations: Tuple[int, int] = (1, 3),
    size: Tuple[float, float] = (0.2, 0.4),
) -> NDArray:
    """
    Apply Cutnoise data augmentation.

    Cutnoise data augmentation. Randomly add noise to a cuboid region in the image to force the model to learn
    denoising when making predictions.

    Parameters
    ----------
    img : Numpy array
        Image to transform. E.g. ``(y, x, channels)`` for ``2D`` or ``(y, x, z, channels)`` for ``3D``.

    scale : tuple of floats, optional
        Scale of the random noise. E.g. ``(0.1, 0.2)``.

    nb_iterations : tuple of ints, optional
        Number of areas with noise to create. E.g. ``(1, 3)``.

    size : boolean, optional
        Range to choose the size of the areas to transform. E.g. ``(0.2, 0.4)``.

    Returns
    -------
    out : Numpy array
        Transformed image. E.g. ``(y, x, channels)`` for ``2D`` or ``(y, x, z, channels)`` for ``3D``.

    Examples
    --------
    Calling this function with ``scale=(0.1,0.2)``, ``nb_iterations=(1,3)`` and
    ``size=(0.2,0.4)`` may result in:

    +---------------------------------------------+---------------------------------------------+
    | .. figure:: ../../../img/orig_cutnoise.png  | .. figure:: ../../../img/cutnoise.png       |
    |   :width: 80%                               |   :width: 80%                               |
    |   :align: center                            |   :align: center                            |
    |                                             |                                             |
    |   Input image                               |   Augmented image                           |
    +---------------------------------------------+---------------------------------------------+
    | .. figure:: ../../../img/orig_cutnoise2.png | .. figure:: ../../../img/cutnoise2.png      |
    |   :width: 80%                               |   :width: 80%                               |
    |   :align: center                            |   :align: center                            |
    |                                             |                                             |
    |   Input image                               |   Augmented image                           |
    +---------------------------------------------+---------------------------------------------+

    The grid and the red squares are painted for visualization purposes.
    """
    assert img.ndim in (3, 4), f"Image must be 3D or 4D, got {img.shape}"
    assert len(scale) == 2 and scale[0] <= scale[1], f"Invalid scale range: {scale}"
    assert len(size) == 2 and 0.0 < size[0] <= size[1] <= 1.0, f"Invalid size range: {size}"
    if img.size == 0:
        return img

    out = img.copy()
    orig_dtype = img.dtype
    is_float = np.issubdtype(orig_dtype, np.floating)

    # Spatial dims
    if img.ndim == 3:      # (H, W, C)
        H, W, C = img.shape
        Z = None
    else:                  # (Z, H, W, C)
        Z, H, W, C = img.shape

    # how many patches (inclusive upper bound)
    it = int(np.random.randint(nb_iterations[0], nb_iterations[1] + 1))

    # helpers
    def _clamp_int(n, lo, hi):
        return max(lo, min(int(n), hi))

    # dtype-safe add with clipping for integer arrays
    def _add_noise_inplace(target_view: NDArray, noise_arr: NDArray):
        # noise_arr shapes: (H,W) or (Z,H,W), we’ll broadcast over C with [..., None]
        if is_float:
            target_view += noise_arr[..., None]
        else:
            info = np.iinfo(orig_dtype)
            tmp = target_view.astype(np.float32, copy=False) + noise_arr[..., None].astype(np.float32, copy=False)
            np.clip(tmp, info.min, info.max, out=tmp)
            target_view[...] = tmp.astype(orig_dtype, copy=False)

    for _ in range(it):
        frac = float(random.uniform(size[0], size[1]))
        # patch size in (y, x)
        y_size = _clamp_int(round(H * frac), 1, H)
        x_size = _clamp_int(round(W * frac), 1, W)
        # top-left (inclusive bounds)
        cy = 0 if H == y_size else np.random.randint(0, H - y_size + 1)
        cx = 0 if W == x_size else np.random.randint(0, W - x_size + 1)

        # amplitude (keep same semantics as your original: scale * img.max())
        amp = float(random.uniform(scale[0], scale[1])) * float(img.max())

        if img.ndim == 3:
            # noise shape (y, x)
            noise = np.random.normal(loc=0.0, scale=amp, size=(y_size, x_size)).astype(np.float32)
            view = out[cy:cy + y_size, cx:cx + x_size, :]
            _add_noise_inplace(view, noise)

        else:
            # z-block proportional to frac (at least 1)
            assert Z is not None, "Z dimension not found in 4D image"
            z_size = _clamp_int(round((Z if Z is not None else 1) * frac), 1, Z)
            z0 = 0 if Z == z_size else np.random.randint(0, Z - z_size + 1)

            # noise shape (z, y, x)
            noise = np.random.normal(loc=0.0, scale=amp, size=(z_size, y_size, x_size)).astype(np.float32)
            view = out[z0:z0 + z_size, cy:cy + y_size, cx:cx + x_size, :]
            _add_noise_inplace(view, noise)

    return out


def misalignment(
    img: NDArray,
    mask: NDArray,
    displacement: int = 16,
    rotate_ratio: float = 0.0,
) -> Tuple[NDArray, NDArray]:
    """
    Apply mis-alignment data augmentation.

    Mis-alignment data augmentation of image stacks. This augmentation is applied to both images and masks.

    Implementation based on `PyTorch Connectomics' misalign.py
    <https://github.com/zudi-lin/pytorch_connectomics/blob/master/connectomics/data/augmentation/misalign.py>`_.

    Parameters
    ----------
    img : Numpy array
        Image to transform. E.g. ``(y, x, channels)`` for ``2D`` or ``(y, x, z, channels)`` for ``3D``.

    mask : Numpy array
        Mask to transform. E.g. ``(y, x, channels)`` for ``2D`` or ``(y, x, z, channels)`` for ``3D``.

    displacement : int, optional
        Maximum pixel displacement in ``xy``-plane.

    rotate_ratio : float, optional
        Ratio of rotation-based mis-alignment.

    Returns
    -------
    out : Numpy array
        Transformed image. E.g. ``(y, x, channels)`` for ``2D`` or ``(y, x, z, channels)`` for ``3D``.

    m_out : Numpy array
        Transformed mask. E.g. ``(y, x, channels)`` for ``2D`` or ``(y, x, z, channels)`` for ``3D``.

    Examples
    --------
    Calling this function with ``displacement=16`` and ``rotate_ratio=0.5`` may result in:

    +---------------------------------------------+---------------------------------------------+
    | .. figure:: ../../../img/orig_miss.png      | .. figure:: ../../../img/orig_miss_mask.png |
    |   :width: 80%                               |   :width: 80%                               |
    |   :align: center                            |   :align: center                            |
    |                                             |                                             |
    |   Input image                               |   Corresponding mask                        |
    +---------------------------------------------+---------------------------------------------+
    | .. figure:: ../../../img/miss.png           | .. figure:: ../../../img/miss_mask.png      |
    |   :width: 80%                               |   :width: 80%                               |
    |   :align: center                            |   :align: center                            |
    |                                             |                                             |
    |   Augmented image                           |   Augmented mask                            |
    +---------------------------------------------+---------------------------------------------+

    The grid is painted for visualization purposes.
    """
    assert img.ndim in [3, 4], f"Image must be 3D or 4D, got shape {img.shape}"
    assert mask.ndim in [3, 4], "Mask is supposed to be 3 or 4 dimensions but provided {} mask shape instead".format(
        mask.shape
    )
    out = np.zeros(img.shape, img.dtype)
    m_out = np.zeros(mask.shape, mask.dtype)

    def _randomrotate_matrix(height: int, displacement: int):
        """ Generate random rotation matrix."""
        x = displacement / 2.0
        y = ((height - displacement) / 2.0) * 1.42
        angle = math.asin(x / y) * 2.0 * 57.2958  # convert radians to degrees
        rand_angle = (random.uniform(0, 1) - 0.5) * 2.0 * angle
        M = cv2.getRotationMatrix2D((height / 2, height / 2), rand_angle, 1)
        return M

    # 2D
    if img.ndim == 3:
        oy = np.random.randint(1, img.shape[0] - 1)
        d = np.random.randint(0, displacement)
        if random.uniform(0, 1) < rotate_ratio:
            # Apply misalignment to all channels
            for i in range(img.shape[-1]):
                out[:oy, :, i] = img[:oy, :, i]
                out[oy:, : img.shape[1] - d, i] = img[oy:, d:, i]
            for i in range(mask.shape[-1]):
                m_out[:oy, :, i] = mask[:oy, :, i]
                m_out[oy:, : mask.shape[1] - d, i] = mask[oy:, d:, i]
        else:
            H, W = img.shape[:2]
            M = _randomrotate_matrix(H, displacement)
            H = H - oy
            # Apply misalignment to all channels
            for i in range(img.shape[-1]):
                out[:oy, :, i] = img[:oy, :, i]
                out[oy:, :, i] = cv2.warpAffine(
                    img[oy:, :, i],
                    M,
                    (W, H),
                    1.0,  # type: ignore
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                )
            for i in range(mask.shape[-1]):
                m_out[:oy, :, i] = mask[:oy, :, i]
                m_out[oy:, :, i] = cv2.warpAffine(
                    mask[oy:, :, i],
                    M,
                    (W, H),
                    1.0,  # type: ignore
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                )
    # 3D
    else:
        # img, mask: (z, y, x, c)
        Z, H, W, C_img = img.shape
        C_mask = mask.shape[-1]

        # spatial crop size after displacement
        out_h = H - displacement
        out_w = W - displacement

        mode = "slip" if random.uniform(0, 1) < 0.5 else "translation"
        print("MODEEE: {}".format(mode))
        # pick a z-plane (used as the “affected slice” for slip, or split point for translation)
        z_idx = np.random.randint(1, Z - 1) if Z >= 3 else 0
        print("z_idx: {}".format(z_idx))
        if random.uniform(0, 1) < rotate_ratio:
            # start from a copy; we’ll overwrite affected slices
            out = img.copy()
            m_out = mask.copy()

            # rotate in the (y, x) plane
            M = _randomrotate_matrix(H, displacement)

            if mode == "slip":
                # only transform the selected z-slice, across ALL channels
                for c in range(C_img):
                    out[z_idx, :, :, c] = 0
                    out[z_idx, :, :, c] = cv2.warpAffine(
                        img[z_idx, :, :, c],
                        M,
                        (W, H),
                        1.0,  # type: ignore
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                    )
                for c in range(C_mask):
                    m_out[z_idx, :, :, c] = 0
                    m_out[z_idx, :, :, c] = cv2.warpAffine(
                        mask[z_idx, :, :, c],
                        M,
                        (W, H),
                        1.0,  # type: ignore
                        flags=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT,
                    )
            else:
                # transform all slices from z_idx onwards, across ALL channels
                for z in range(z_idx, Z):
                    for c in range(C_img):
                        out[z, :, :, c] = 0
                        out[z, :, :, c] = cv2.warpAffine(
                            img[z, :, :, c],
                            M,
                            (W, H),
                            1.0,  # type: ignore
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                        )
                for z in range(z_idx, Z):
                    for c in range(C_mask):
                        m_out[z, :, :, c] = 0
                        m_out[z, :, :, c] = cv2.warpAffine(
                            mask[z, :, :, c],
                            M,
                            (W, H),
                            1.0,  # type: ignore
                            flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT,
                        )
        else:
            # xy translations via cropping/paste
            rng = np.random.RandomState()
            x0 = rng.randint(displacement)
            y0 = rng.randint(displacement)
            x1 = rng.randint(displacement)
            y1 = rng.randint(displacement)

            if mode == "slip":
                # copy whole volume once
                out[:, y0:y0 + out_h, x0:x0 + out_w, :] = img[
                    :, y0:y0 + out_h, x0:x0 + out_w, :
                ]
                m_out[:, y0:y0 + out_h, x0:x0 + out_w, :] = mask[
                    :, y0:y0 + out_h, x0:x0 + out_w, :
                ]

                # then overwrite only the chosen z-slice (all channels) with a different offset
                out[z_idx, :, :, :] = 0
                out[z_idx, y1:y1 + out_h, x1:x1 + out_w, :] = img[
                    z_idx, y1:y1 + out_h, x1:x1 + out_w, :
                ]
                m_out[z_idx, :, :, :] = 0
                m_out[z_idx, y1:y1 + out_h, x1:x1 + out_w, :] = mask[
                    z_idx, y1:y1 + out_h, x1:x1 + out_w, :
                ]
            else:
                # split volume along z at z_idx (all channels)
                out[:z_idx, y0:y0 + out_h, x0:x0 + out_w, :] = img[
                    :z_idx, y0:y0 + out_h, x0:x0 + out_w, :
                ]
                out[z_idx:, y1:y1 + out_h, x1:x1 + out_w, :] = img[
                    z_idx:, y1:y1 + out_h, x1:x1 + out_w, :
                ]
                m_out[:z_idx, y0:y0 + out_h, x0:x0 + out_w, :] = mask[
                    :z_idx, y0:y0 + out_h, x0:x0 + out_w, :
                ]
                m_out[z_idx:, y1:y1 + out_h, x1:x1 + out_w, :] = mask[
                    z_idx:, y1:y1 + out_h, x1:x1 + out_w, :
                ]

    return out, m_out


def brightness(
    image: NDArray,
    brightness_factor: Tuple[float, float] = (0, 0),
) -> NDArray:
    """
    Randomly adjust brightness between a range.

    Parameters
    ----------
    image : Numpy array
        Image to transform. E.g. ``(y, x, channels)`` for ``2D`` or ``(y, x, z, channels)`` for ``3D``.

    brightness_factor : tuple of 2 floats
        Range of brightness' intensity. E.g. ``(0.1, 0.3)``.

    Returns
    -------
    image : Numpy array
        Transformed image. E.g. ``(y, x, channels)`` for ``2D`` or ``(y, x, z, channels)`` for ``3D``.

    Examples
    --------
    Calling this function with ``brightness_factor=(0.1,0.3)``, ``mode='mix'``, ``invert=False`` and ``invert_p=0``
    may result in:

    +---------------------------------------------+---------------------------------------------+
    | .. figure:: ../../../img/orig_bright.png    | .. figure:: ../../../img/bright.png         |
    |   :width: 80%                               |   :width: 80%                               |
    |   :align: center                            |   :align: center                            |
    |                                             |                                             |
    |   Input image                               |   Augmented image                           |
    +---------------------------------------------+---------------------------------------------+
    | .. figure:: ../../../img/orig_bright2.png   | .. figure:: ../../../img/bright2.png        |
    |   :width: 80%                               |   :width: 80%                               |
    |   :align: center                            |   :align: center                            |
    |                                             |                                             |
    |   Input image                               |   Augmented image                           |
    +---------------------------------------------+---------------------------------------------+

    The grid is painted for visualization purposes.
    """
    assert image.ndim in (3, 4), f"Image must be 3D or 4D, got {image.shape}"

    lo, hi = float(brightness_factor[0]), float(brightness_factor[1])
    if lo == 0.0 and hi == 0.0:
        return image
    if lo > hi:
        lo, hi = hi, lo

    if image.size == 0:
        return image

    delta = float(np.random.uniform(lo, hi))
    out = image.copy()

    if np.issubdtype(out.dtype, np.floating):
        out += delta
        return out

    # integer dtype: add in float and clip back
    info = np.iinfo(out.dtype)
    tmp = out.astype(np.float32, copy=False) + delta
    np.clip(tmp, info.min, info.max, out=tmp)
    return tmp.astype(out.dtype, copy=False)


def contrast(image: NDArray, contrast_factor: Tuple[float, float] = (0, 0)) -> NDArray:
    """
    Contrast augmentation.

    Parameters
    ----------
    image : Numpy array
        Image to transform. E.g. ``(y, x, channels)`` for ``2D`` or ``(y, x, z, channels)`` for ``3D``.

    contrast_factor : tuple of 2 floats
        Range of contrast's intensity. E.g. ``(0.1, 0.3)``.

    Returns
    -------
    image : Numpy array
        Transformed image. E.g. ``(y, x, channels)`` for ``2D`` or ``(y, x, z, channels)`` for ``3D``.

    Examples
    --------
    Calling this function with ``contrast_factor=(0.1,0.3)``, ``mode='mix'``, ``invert=False`` and ``invert_p=0``
    may result in:

    +---------------------------------------------+---------------------------------------------+
    | .. figure:: ../../../img/orig_contrast.png  | .. figure:: ../../../img/contrast.png       |
    |   :width: 80%                               |   :width: 80%                               |
    |   :align: center                            |   :align: center                            |
    |                                             |                                             |
    |   Input image                               |   Augmented image                           |
    +---------------------------------------------+---------------------------------------------+
    | .. figure:: ../../../img/orig_contrast2.png | .. figure:: ../../../img/contrast2.png      |
    |   :width: 80%                               |   :width: 80%                               |
    |   :align: center                            |   :align: center                            |
    |                                             |                                             |
    |   Input image                               |   Augmented image                           |
    +---------------------------------------------+---------------------------------------------+

    The grid is painted for visualization purposes.
    """
    assert image.ndim in (3, 4), f"Image must be 3D or 4D, got {image.shape}"

    lo, hi = float(contrast_factor[0]), float(contrast_factor[1])
    if lo == 0.0 and hi == 0.0:
        return image
    if lo > hi:
        lo, hi = hi, lo

    if image.size == 0:
        return image

    scale = 1.0 + float(np.random.uniform(lo, hi))
    out = image.copy()

    if np.issubdtype(out.dtype, np.floating):
        out *= scale
        return out

    # integer dtype: multiply in float and clip back
    info = np.iinfo(out.dtype)
    tmp = out.astype(np.float32, copy=False) * scale
    np.clip(tmp, info.min, info.max, out=tmp)
    return tmp.astype(out.dtype, copy=False)


def missing_sections(img: NDArray, iterations: Tuple[int, int] = (30, 40), channel_prob: float = 0.5) -> NDArray:
    """
    Augment the image by creating a black line in a random position.

    Implementation based on `PyTorch Connectomics' missing_parts.py
    <https://github.com/zudi-lin/pytorch_connectomics/blob/master/connectomics/data/augmentation/missing_parts.py>`_.

    Parameters
    ----------
    img : Numpy array
        Image to transform. E.g. ``(y, x, channels)`` for ``2D`` or ``(y, x, z, channels)`` for ``3D``.

    iterations : tuple of 2 ints, optional
        Iterations to dilate the missing line with. E.g. ``(30, 40)``.

    channel_prob : float, optional
        Probability of applying a missing section to each channel individually.

    Returns
    -------
    out : Numpy array
        Transformed image. E.g. ``(y, x, channels)`` for ``2D`` or ``(y, x, z, channels)`` for ``3D``.

    Examples
    --------
    Calling this function with ``iterations=(30,40)`` may result in:

    +---------------------------------------------+---------------------------------------------+
    | .. figure:: ../../../img/orig_missing.png   | .. figure:: ../../../img/missing.png        |
    |   :width: 80%                               |   :width: 80%                               |
    |   :align: center                            |   :align: center                            |
    |                                             |                                             |
    |   Input image                               |   Augmented image                           |
    +---------------------------------------------+---------------------------------------------+
    | .. figure:: ../../../img/orig_missing2.png  | .. figure:: ../../../img/missing2.png       |
    |   :width: 80%                               |   :width: 80%                               |
    |   :align: center                            |   :align: center                            |
    |                                             |                                             |
    |   Input image                               |   Augmented image                           |
    +---------------------------------------------+---------------------------------------------+

    The grid is painted for visualization purposes.
    """
    assert img.ndim in (3, 4), f"Image must be 3D or 4D, got shape {img.shape}"
    it = int(np.random.randint(iterations[0], iterations[1]))

    out = img.copy()

    def _prepare_deform_slice(slice_shape: Tuple[int, int], iterations: int) -> NDArray:
        """Build a boolean mask (H,W) for a dilated random line to 'remove'."""
        H, W = slice_shape
        if H < 3 or W < 3:
            # too small to draw a line with the given sampling; return empty mask
            return np.zeros((H, W), dtype=bool)

        # randomly choose fixed x or fixed y with p=1/2
        fixed_x = (np.random.rand() < 0.5)
        if fixed_x:
            x0, y0 = 0, np.random.randint(1, W - 1)
            x1, y1 = H - 1, np.random.randint(1, W - 1)
        else:
            x0, y0 = np.random.randint(1, H - 1), 0
            x1, y1 = np.random.randint(1, H - 1), W - 1

        # base line mask
        line_mask = np.zeros((H, W), dtype=bool)
        rr, cc = line(x0, y0, x1, y1)
        rr = np.clip(rr, 0, H - 1)
        cc = np.clip(cc, 0, W - 1)
        line_mask[rr, cc] = True

        # (legacy leftover: normal/labels not used for the final effect)
        # dilate to thicken the missing section
        line_mask = binary_dilation_scipy(line_mask, iterations=iterations)  # type: ignore
        return line_mask

    if img.ndim == 3:
        # (y, x, c)  -> operate per channel
        H, W, C = img.shape
        slice_shape = (H, W)

        transforms = {}
        i = 0
        while i < C:
            if np.random.rand() < channel_prob:
                transforms[i] = _prepare_deform_slice(slice_shape, it)
                i += 2  # enforce gap: at most one mod in any consecutive 3
            i += 1

        for c in transforms.keys():
            line_mask = transforms[c]
            sl = out[..., c]
            mean_val = sl.mean()
            sl[line_mask] = mean_val
            out[..., c] = sl

    else:
        # (z, y, x, c) -> operate along z for each channel independently
        Z, H, W, C = img.shape
        slice_shape = (H, W)

        for c in range(C):
            transforms = {}
            i = 0
            while i < Z:
                if np.random.rand() < channel_prob:
                    transforms[i] = _prepare_deform_slice(slice_shape, it)
                    i += 2  # enforce gap along z for this channel
                i += 1

            for z, line_mask in transforms.items():
                sl = out[z, :, :, c]
                mean_val = sl.mean()
                sl[line_mask] = mean_val
                out[z, :, :, c] = sl

    return out



def shuffle_channels(img: NDArray) -> NDArray:
    """
    Augment the image by shuffling its channels.

    Parameters
    ----------
    img : 3D/4D Numpy array
        Image to transform. E.g. ``(y, x, channels)`` for ``2D`` or ``(y, x, z, channels)`` for ``3D``.

    Returns
    -------
    out : 3D/4D Numpy array
        Transformed image. E.g. ``(y, x, channels)`` for ``2D`` or ``(y, x, z, channels)`` for ``3D``.

    Examples
    --------
    +-----------------------------------------------+-----------------------------------------------+
    | .. figure:: ../../../img/orig_chshuffle.png   | .. figure:: ../../../img/chshuffle.png        |
    |   :width: 80%                                 |   :width: 80%                                 |
    |   :align: center                              |   :align: center                              |
    |                                               |                                               |
    |   Input image                                 |   Augmented image                             |
    +-----------------------------------------------+-----------------------------------------------+

    The grid is painted for visualization purposes.
    """
    assert img.ndim in (3, 4), f"Image must be 3D or 4D, got {img.shape}"
    new_channel_order = np.random.permutation(img.shape[-1])
    return img[..., new_channel_order]


def grayscale(img: NDArray) -> NDArray:
    """
    Augment the image by converting it into grayscale.

    Parameters
    ----------
    img : 3D/4D Numpy array
        Image to transform. E.g. ``(y, x, channels)`` for ``2D`` or ``(y, x, z, channels)`` for ``3D``.

    Returns
    -------
    out : 3D/4D Numpy array
        Transformed image. E.g. ``(y, x, channels)`` for ``2D`` or ``(y, x, z, channels)`` for ``3D``.

    Examples
    --------
    +-----------------------------------------------+-----------------------------------------------+
    | .. figure:: ../../../img/orig_grayscale.png   | .. figure:: ../../../img/grayscale.png        |
    |   :width: 80%                                 |   :width: 80%                                 |
    |   :align: center                              |   :align: center                              |
    |                                               |                                               |
    |   Input image                                 |   Augmented image                             |
    +-----------------------------------------------+-----------------------------------------------+

    The grid is painted for visualization purposes.
    """
    assert img.ndim in [3, 4], f"Image must be 3D or 4D, got shape {img.shape}"
    if img.shape[-1] != 3:
        raise ValueError(
            "Image is supposed to have 3 channels (RGB). Provided {} image shape instead".format(img.shape)
        )

    return np.tile(np.expand_dims(np.mean(img, -1), -1), 3)


def GridMask(
    img: NDArray,
    z_size: int,
    ratio: float = 0.6,
    d_range: Tuple[float, ...] = (30.0, 60.0),
    rotate: int = 1,
    invert: bool = False,
) -> NDArray:
    """
    Apply GridMask data augmentation presented in `GridMask Data Augmentation <https://arxiv.org/abs/2001.04086v1>`_.

    GridMask is a data augmentation technique that randomly masks out grid-like regions in the image, which helps the model to learn more robust features by forcing it to focus on different parts of the image Code adapted from `<https://github.com/dvlab-research/GridMask/blob/master/imagenet_grid/utils/grid.py>`_.

    Parameters
    ----------
    img : Numpy array
        Image to transform. E.g. ``(y, x, channels)`` for ``2D`` or ``(y, x, z, channels)`` for ``3D``.

    z_size : int
        Size of z dimension. Used for 3D images as the z axis has been merged with the channels. Set to -1 to when
        do not want to be applied.

    ratio : tuple of floats, optional
        Range to choose the size of the areas to create.

    d_range : tuple of floats, optional
        Range to choose the ``d`` value in the original paper.

    rotate : float, optional
        Rotation of the mask in GridMask. Needs to be between ``[0,1]`` where 1 is 360 degrees.

    invert : bool, optional
        Whether to invert the mask.

    Returns
    -------
    out : Numpy array
        Transformed image. E.g. ``(y, x, channels)`` for ``2D`` or ``(y, x, z, channels)`` for ``3D``.

    Examples
    --------
    Calling this function with the default settings may result in:

    +----------------------------------------------+----------------------------------------------+
    | .. figure:: ../../../img/orig_GridMask.png   | .. figure:: ../../../img/GridMask.png        |
    |   :width: 80%                                |   :width: 80%                                |
    |   :align: center                             |   :align: center                             |
    |                                              |                                              |
    |   Input image                                |   Augmented image                            |
    +----------------------------------------------+----------------------------------------------+

    The grid is painted for visualization purposes.
    """
    assert img.ndim in [3, 4], f"Image must be 3D or 4D, got shape {img.shape}"
    assert 0 <= rotate <= 1, "Rotate should be between 0 and 1. Provided {}".format(rotate)
    assert 0 < ratio < 1, "Ratio should be between 0 and 1. Provided {}".format(ratio)

    # Get spatial dims (h, w) regardless of 2D or 3D input
    if img.ndim == 3:
        h, w = img.shape[0], img.shape[1]
    else:  # (z, y, x, c)
        h, w = img.shape[1], img.shape[2]

    # Minimum square that fully covers the image after rotation
    hh = int(math.ceil(math.sqrt(h * h + w * w)))

    # Grid parameters
    d = np.random.randint(int(d_range[0]), int(d_range[1]))  # grid period
    l = int(math.ceil(d * ratio))                            # mask square size per period

    # Build base mask
    mask = np.ones((hh, hh), np.float32)
    st_h = np.random.randint(d)
    st_w = np.random.randint(d)

    # Horizontal stripes
    for i in range(-1, hh // d + 1):
        s = d * i + st_h
        t = s + l
        s = max(min(s, hh), 0)
        t = max(min(t, hh), 0)
        if s < t:
            mask[s:t, :] *= 0

    # Vertical stripes
    for i in range(-1, hh // d + 1):
        s = d * i + st_w
        t = s + l
        s = max(min(s, hh), 0)
        t = max(min(t, hh), 0)
        if s < t:
            mask[:, s:t] *= 0

    # Rotation: interpret rotate in [0,1] as a fraction of 360°
    max_deg = max(1, int(round(rotate * 360)))
    r = np.random.randint(max_deg)
    mask = Image.fromarray(np.uint8(mask * 255))
    mask = mask.rotate(r)
    mask = np.asarray(mask, dtype=np.float32) / 255.0

    # Center-crop back to (h, w)
    y0 = (hh - h) // 2
    x0 = (hh - w) // 2
    mask = mask[y0:y0 + h, x0:x0 + w]

    if not invert:
        mask = 1.0 - mask  # keep same semantics as your original

    # Apply
    if img.ndim == 3:
        # (y, x, c): broadcast mask over channel dim
        return img * mask[:, :, None]
    else:
        # (z, y, x, c)
        Z, H, W, C = img.shape
        mask_4d = mask[None, :, :, None]  # (1, H, W, 1)

        out = img.copy()

        # If z_size != -1, apply to a random contiguous z-block (size sampled if possible)
        if z_size != -1:
            # If provided, use d_range[2:4] as the z-block size range; else default to [1, Z]
            if len(d_range) >= 4:
                z_min = max(1, int(d_range[2]))
                z_max = max(z_min, int(d_range[3]))
                z_max = min(z_max, Z)
            else:
                z_min, z_max = 1, Z

            block = np.random.randint(z_min, z_max + 1)  # inclusive upper bound
            if block > Z:
                block = Z
            start = np.random.randint(0, max(1, Z - block + 1))
            end = start + block

            out[start:end, :, :, :] = out[start:end, :, :, :] * mask_4d
            return out

        # Else: apply to all z-slices
        out = out * mask_4d
        return out


def random_crop_pair(
    image: NDArray,
    mask: NDArray,
    random_crop_size: Tuple[int, ...],
    val: bool = False,
    draw_prob_map_points: bool = False,
    img_prob: Optional[NDArray] = None,
    weight_map: Optional[NDArray] = None,
    scale: Tuple[int, ...] = (1, 1),
) -> Union[
    Tuple[NDArray, NDArray],
    Tuple[NDArray, NDArray, NDArray],
    Tuple[NDArray, NDArray, int, int, int, int],
]:
    """
    Apply random crop for an image and its mask.

    No crop is done in those dimensions that ``random_crop_size`` is greater than
    the input image shape in those dimensions. For instance, if an input image is ``400x150`` and ``random_crop_size`` is ``224x224`` the resulting image will be ``224x150``.

    Parameters
    ----------
    image : Numpy 3D array
        Image. E.g. ``(y, x, channels)``.

    mask : Numpy 3D array
        Image mask. E.g. ``(y, x, channels)``.

    random_crop_size : 2 int tuple
        Size of the crop. E.g. ``(height, width)``.

    val : bool, optional
        If the image provided is going to be used in the validation data. This forces to crop from the origin,
        e. g. ``(0, 0)`` point.

    draw_prob_map_points : bool, optional
        To return the pixel chosen to be the center of the crop.

    img_prob : Numpy 3D array, optional
        Probability of each pixel to be chosen as the center of the crop. E. .g. ``(y, x, channels)``.

    weight_map : bool, optional
        Weight map of the given image. E.g. ``(y, x, channels)``.

    scale : tuple of 2 ints, optional
        Scale factor the second image given. E.g. ``(2,2)``.

    Returns
    -------
    img : 2D Numpy array
        Crop of the given image. E.g. ``(y, x, channels)``.

    weight_map : 2D Numpy array, optional
        Crop of the given image's weigth map. E.g. ``(y, x, channels)``.

    ox : int, optional
        X coordinate in the complete image of the chose central pixel to make the crop.

    oy : int, optional
        Y coordinate in the complete image of the chose central pixel to make the crop.

    x : int, optional
        X coordinate in the complete image where the crop starts.

    y : int, optional
        Y coordinate in the complete image where the crop starts.
    """
    assert image.ndim == 3, f"Image must be 3D, got {image.shape}"
    assert mask.ndim == 3, f"Mask must be 3D, got {mask.shape}"
    assert len(random_crop_size) == 2, f"Random crop size must have 2 elements, got {random_crop_size}"
    if weight_map is not None:
        img, we = image
    else:
        img = image

    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size[0], random_crop_size[1]
    if val:
        y, x, oy, ox = 0, 0, 0, 0
    else:
        if img_prob is not None:
            prob = img_prob.ravel()

            # Generate the random coordinates based on the distribution
            choices = np.prod(img_prob.shape)
            index = np.random.choice(choices, size=1, p=prob)
            coordinates = np.unravel_index(index, img_prob.shape)
            x = int(coordinates[1][0])
            y = int(coordinates[0][0])
            ox = int(coordinates[1][0])
            oy = int(coordinates[0][0])

            # Adjust the coordinates to be the origin of the crop and control to
            # not be out of the image
            if y < int(random_crop_size[0] / 2):
                y = 0
            elif y > img.shape[0] - int(random_crop_size[0] / 2):
                y = img.shape[0] - random_crop_size[0]
            else:
                y -= int(random_crop_size[0] / 2)

            if x < int(random_crop_size[1] / 2):
                x = 0
            elif x > img.shape[1] - int(random_crop_size[1] / 2):
                x = img.shape[1] - random_crop_size[1]
            else:
                x -= int(random_crop_size[1] / 2)
        else:
            oy, ox = 0, 0
            x = np.random.randint(0, width - dx + 1) if width - dx + 1 > 0 else 0
            y = np.random.randint(0, height - dy + 1) if height - dy + 1 > 0 else 0

    # Super-resolution check
    if any([x != 1 for x in scale]):
        img_out_shape = img[y : (y + dy), x : (x + dx)].shape
        mask_out_shape = mask[y * scale[0] : (y + dy) * scale[0], x * scale[1] : (x + dx) * scale[1]].shape
        s = [img_out_shape[0] * scale[0], img_out_shape[1] * scale[1]]
        if all(x != y for x, y in zip(s, mask_out_shape)):
            raise ValueError(
                "Images can not be cropped to a PATCH_SIZE of {}. Inputs: LR image shape={} "
                "and HR image shape={}. When cropping the output shapes are {} and {}, for LR and HR images respectively. "
                "Try to reduce DATA.PATCH_SIZE".format(
                    random_crop_size,
                    img.shape,
                    mask.shape,
                    img_out_shape,
                    mask_out_shape,
                )
            )

    if draw_prob_map_points:
        return (
            img[y : (y + dy), x : (x + dx)],
            mask[y * scale[0] : (y + dy) * scale[0], x * scale[1] : (x + dx) * scale[1]],
            oy,
            ox,
            y,
            x,
        )
    else:
        if weight_map is not None:
            return (
                img[y : (y + dy), x : (x + dx)],
                mask[
                    y * scale[0] : (y + dy) * scale[0],
                    x * scale[1] : (x + dx) * scale[1],
                ],
                weight_map[y : (y + dy), x : (x + dx)],
            )
        else:
            return (
                img[y : (y + dy), x : (x + dx)],
                mask[
                    y * scale[0] : (y + dy) * scale[0],
                    x * scale[1] : (x + dx) * scale[1],
                ],
            )


def random_3D_crop_pair(
    image: NDArray,
    mask: NDArray,
    random_crop_size: Tuple[int, ...],
    val: bool = False,
    img_prob: Optional[NDArray] = None,
    weight_map: Optional[NDArray] = None,
    draw_prob_map_points: bool = False,
    scale: Tuple[int, ...] = (1, 1, 1),
) -> Union[
    Tuple[NDArray, NDArray],
    Tuple[NDArray, NDArray, NDArray],
    Tuple[NDArray, NDArray, int, int, int, int, int, int],
]:
    """
    Extract a random 3D patch from the given image and mask.

    No crop is done in those dimensions that ``random_crop_size`` is
    greater than the input image shape in those dimensions. For instance, if an input image is ``10x400x150`` and ``random_crop_size``
    is ``10x224x224`` the resulting image will be ``10x224x150``.

    Parameters
    ----------
    image : 4D Numpy array
        Data to extract the patch from. E.g. ``(z, y, x, channels)``.

    mask : 4D Numpy array
        Data mask to extract the patch from. E.g. ``(z, y, x, channels)``.

    random_crop_size : 3D int tuple
        Shape of the patches to create. E.g. ``(z, y, x)``.

    val : bool, optional
        If the image provided is going to be used in the validation data. This forces to crop from the origin, e.g.
        ``(0, 0)`` point.

    img_prob : Numpy 4D array, optional
        Probability of each pixel to be chosen as the center of the crop. E. g. ``(z, y, x, channels)``.

    weight_map : bool, optional
        Weight map of the given image. E.g. ``(z, y, x, channels)``.

    draw_prob_map_points : bool, optional
        To return the voxel chosen to be the center of the crop.

    scale : tuple of 3 ints, optional
        Scale factor the second image given. E.g. ``(2,4,4)``.

    Returns
    -------
    img : 4D Numpy array
        Crop of the given image. E.g. ``(z, y, x, channels)``.

    weight_map : 4D Numpy array, optional
        Crop of the given image's weigth map. E.g. ``(z, y, x, channels)``.

    oz : int, optional
        Z coordinate in the complete image of the chose central pixel to
        make the crop.

    oy : int, optional
        Y coordinate in the complete image of the chose central pixel to
        make the crop.

    ox : int, optional
        X coordinate in the complete image of the chose central pixel to
        make the crop.

    z : int, optional
        Z coordinate in the complete image where the crop starts.

    y : int, optional
        Y coordinate in the complete image where the crop starts.

    x : int, optional
        X coordinate in the complete image where the crop starts.
    """
    assert image.ndim == 4, f"Image must be 4D, got {image.shape}"
    assert mask.ndim == 4, f"Mask must be 4D, got {mask.shape}"
    assert len(random_crop_size) == 3, f"Random crop size must have 3 elements, got {random_crop_size}"
    if weight_map is not None:
        vol, we = image
    else:
        vol = image

    deep, cols, rows = vol.shape[0], vol.shape[1], vol.shape[2]
    dz, dy, dx = random_crop_size
    if val:
        x, y, z, ox, oy, oz = 0, 0, 0, 0, 0, 0
    else:
        if img_prob is not None:
            prob = img_prob.ravel()

            # Generate the random coordinates based on the distribution
            choices = np.prod(img_prob.shape)
            index = np.random.choice(choices, size=1, p=prob)
            coordinates = np.unravel_index(index, shape=img_prob.shape)
            x = int(coordinates[2])
            y = int(coordinates[1])
            z = int(coordinates[0])
            ox = int(coordinates[2])
            oy = int(coordinates[1])
            oz = int(coordinates[0])

            # Adjust the coordinates to be the origin of the crop and control to
            # not be out of the volume
            if z < int(random_crop_size[0] / 2):
                z = 0
            elif z > vol.shape[0] - int(random_crop_size[0] / 2):
                z = vol.shape[0] - random_crop_size[0]
            else:
                z -= int(random_crop_size[0] / 2)

            if y < int(random_crop_size[1] / 2):
                y = 0
            elif y > vol.shape[1] - int(random_crop_size[1] / 2):
                y = vol.shape[1] - random_crop_size[1]
            else:
                y -= int(random_crop_size[1] / 2)

            if x < int(random_crop_size[2] / 2):
                x = 0
            elif x > vol.shape[2] - int(random_crop_size[2] / 2):
                x = vol.shape[2] - random_crop_size[2]
            else:
                x -= int(random_crop_size[2] / 2)
        else:
            ox = 0
            oy = 0
            oz = 0
            z = np.random.randint(0, deep - dz + 1) if deep - dz + 1 > 0 else 0
            y = np.random.randint(0, cols - dy + 1) if cols - dy + 1 > 0 else 0
            x = np.random.randint(0, rows - dx + 1) if rows - dx + 1 > 0 else 0

    # Super-resolution check
    if any([x != 1 for x in scale]):
        img_out_shape = vol[z : (z + dz), y : (y + dy), x : (x + dx)].shape
        mask_out_shape = mask[
            z * scale[0] : (z + dz) * scale[0],
            y * scale[1] : (y + dy) * scale[1],
            x * scale[2] : (x + dx) * scale[2],
        ].shape
        s = [
            img_out_shape[0] * scale[0],
            img_out_shape[1] * scale[1],
            img_out_shape[2] * scale[2],
        ]
        if all(x != y for x, y in zip(s, mask_out_shape)):
            raise ValueError(
                "Images can not be cropped to a PATCH_SIZE of {}. Inputs: LR image shape={} "
                "and HR image shape={}. When cropping the output shapes are {} and {}, for LR and HR images respectively. "
                "Try to reduce DATA.PATCH_SIZE".format(
                    random_crop_size,
                    vol.shape,
                    mask.shape,
                    img_out_shape,
                    mask_out_shape,
                )
            )

    if draw_prob_map_points:
        return (
            vol[z : (z + dz), y : (y + dy), x : (x + dx)],
            mask[
                z * scale[0] : (z + dz) * scale[0],
                y * scale[1] : (y + dy) * scale[1],
                x * scale[2] : (x + dx) * scale[2],
            ],
            oz,
            oy,
            ox,
            z,
            y,
            x,
        )
    else:
        if weight_map is not None:
            return (
                vol[z : (z + dz), y : (y + dy), x : (x + dx)],
                mask[
                    z * scale[0] : (z + dz) * scale[0],
                    y * scale[1] : (y + dy) * scale[1],
                    x * scale[2] : (x + dx) * scale[2],
                ],
                weight_map[z : (z + dz), y : (y + dy), x : (x + dx)],
            )
        else:
            return (
                vol[z : (z + dz), y : (y + dy), x : (x + dx)],
                mask[z : (z + dz), y : (y + dy), x : (x + dx)],
            )


def random_crop_single(
    image: NDArray,
    random_crop_size: Tuple[int, ...],
    val: bool = False,
    draw_prob_map_points: bool = False,
    weight_map: Optional[NDArray] = None,
) -> Union[
    NDArray,
    Tuple[NDArray, NDArray],
    Tuple[NDArray, int, int, int, int],
]:
    """
    Random crop for a single image.

    No crop is done in those dimensions that ``random_crop_size`` is greater than
    the input image shape in those dimensions. For instance, if an input image is ``400x150`` and ``random_crop_size`` is ``224x224`` the resulting image will be ``224x150``.

    Parameters
    ----------
    image : Numpy 3D array
        Image. E.g. ``(y, x, channels)`` for ``2D`` or ``(y, x, z, channels)`` for ``3D``.

    random_crop_size : 2 int tuple
        Size of the crop. E.g. ``(y, x)``.

    val : bool, optional
        If the image provided is going to be used in the validation data. This forces to crop from the origin,
        e. g. ``(0, 0)`` point.

    draw_prob_map_points : bool, optional
        To return the pixel chosen to be the center of the crop.

    weight_map : bool, optional
        Weight map of the given image. E.g. ``(y, x, channels)`` for ``2D`` or ``(y, x, z, channels)`` for ``3D``.

    Returns
    -------
    img : 2D Numpy array
        Crop of the given image. E.g. ``(y, x)``.

    weight_map : 2D Numpy array, optional
        Crop of the given image's weigth map. E.g. ``(y, x)``.

    oy : int, optional
        Y coordinate in the complete image of the chose central pixel to make the crop.

    ox : int, optional
        X coordinate in the complete image of the chose central pixel to make the crop.

    y : int, optional
        Y coordinate in the complete image where the crop starts.

    y : int, optional
        X coordinate in the complete image where the crop starts.
    """
    assert image.ndim == 3, f"Image must be 3D, got {image.shape}"
    if weight_map is not None:
        img, we = image
    else:
        img = image

    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    if val:
        x, y, z, ox, oy, oz = 0, 0, 0, 0, 0, 0
    else:
        oy, ox = 0, 0
        x = np.random.randint(0, width - dx + 1) if width - dx + 1 > 0 else 0
        y = np.random.randint(0, height - dy + 1) if height - dy + 1 > 0 else 0

    if draw_prob_map_points:
        return img[y : (y + dy), x : (x + dx)], ox, oy, x, y
    else:
        if weight_map is not None:
            return (
                img[y : (y + dy), x : (x + dx)],
                weight_map[y : (y + dy), x : (x + dx)],
            )
        else:
            return img[y : (y + dy), x : (x + dx)]


def random_3D_crop_single(
    image: NDArray,
    random_crop_size: Tuple[int, ...],
    val: bool = False,
    draw_prob_map_points: bool = False,
    weight_map: Optional[NDArray] = None,
) -> Union[
    NDArray,
    Tuple[NDArray, NDArray],
    Tuple[NDArray, int, int, int, int, int, int],
]:
    """
    Random crop for a single image.

    No crop is done in those dimensions that ``random_crop_size`` is greater than
    the input image shape in those dimensions. For instance, if an input image is ``50x400x150`` and ``random_crop_size`` is ``30x224x224`` the resulting image will be ``30x224x150``.

    Parameters
    ----------
    image : Numpy 3D array
        Image. E.g. ``(z, y, x, channels)``.

    random_crop_size : 2 int tuple
        Size of the crop. E.g. ``(z, y, x)``.

    val : bool, optional
        If the image provided is going to be used in the validation data. This forces to crop from the origin,
        e. g. ``(0, 0)`` point.

    draw_prob_map_points : bool, optional
        To return the pixel chosen to be the center of the crop.

    weight_map : bool, optional
        Weight map of the given image. E.g. ``(z, y, x, channels)``.

    Returns
    -------
    img : 2D Numpy array
        Crop of the given image. E.g. ``(z, y, x)``.

    weight_map : 2D Numpy array, optional
        Crop of the given image's weigth map. E.g. ``(z, y, x)``.

    ox : int, optional
        Z coordinate in the complete image of the chose central pixel to make the crop.

    oy : int, optional
        Y coordinate in the complete image of the chose central pixel to make the crop.

    ox : int, optional
        X coordinate in the complete image of the chose central pixel to make the crop.

    z : int, optional
        Z coordinate in the complete image where the crop starts.

    y : int, optional
        Y coordinate in the complete image where the crop starts.

    x : int, optional
        X coordinate in the complete image where the crop starts.
    """
    assert image.ndim == 3, f"Image must be 3D, got {image.shape}"

    if weight_map is not None:
        img, we = image
    else:
        img = image

    deep, cols, rows = img.shape[0], img.shape[1], img.shape[2]
    dz, dy, dx = random_crop_size
    if val:
        x, y, z, ox, oy, oz = 0, 0, 0, 0, 0, 0
    else:
        ox = 0
        oy = 0
        oz = 0
        z = np.random.randint(0, deep - dz + 1) if deep - dz + 1 > 0 else 0
        y = np.random.randint(0, cols - dy + 1) if cols - dy + 1 > 0 else 0
        x = np.random.randint(0, rows - dx + 1) if rows - dx + 1 > 0 else 0

    if draw_prob_map_points:
        return img[z : (z + dz), y : (y + dy), x : (x + dx)], oz, oy, ox, z, y, x
    else:
        if weight_map is not None:
            return (
                img[z : (z + dz), y : (y + dy), x : (x + dx)],
                weight_map[z : (z + dz), y : (y + dy), x : (x + dx)],
            )
        else:
            return img[z : (z + dz), y : (y + dy), x : (x + dx)]


def center_crop_single(
    img: NDArray,
    crop_shape: Tuple[int, ...],
) -> NDArray:
    """
    Extract the central patch from a single image.

    Parameters
    ----------
    img : 3D/4D array
        Image. E.g. ``(y, x, channels)`` or ``(z, y, x, channels)``.

    crop_shape : 2/3 int tuple
        Size of the crop. E.g. ``(y, x)`` or ``(z, y, x)``.

    Returns
    -------
    img : 3D/4D Numpy array
        Center crop of the given image. E.g. ``(y, x, channels)`` or ``(z, y, x, channels)``.
    """
    assert img.ndim in [3, 4], f"Image must be 3D or 4D, got shape {img.shape}"
    if img.ndim == 4:
        z, y, x, c = img.shape
        startz = max(z // 2 - crop_shape[0] // 2, 0)
        starty = max(y // 2 - crop_shape[1] // 2, 0)
        startx = max(x // 2 - crop_shape[2] // 2, 0)
        return img[
            startz : startz + crop_shape[0],
            starty : starty + crop_shape[1],
            startx : startx + crop_shape[2],
        ]
    else:
        y, x, c = img.shape
        starty = max(y // 2 - crop_shape[0] // 2, 0)
        startx = max(x // 2 - crop_shape[1] // 2, 0)
        return img[starty : starty + crop_shape[0], startx : startx + crop_shape[1]]


def resize_img(img: NDArray, shape: Tuple[int, ...]) -> NDArray:
    """
    Resize input image to given shape.

    Parameters
    ----------
    img : 3D/4D Numpy array
        Data to extract the patch from. E.g. ``(y, x, channels)`` for ``2D`` or  ``(z, y, x, channels)`` for ``3D``.

    shape : 2D/3D int tuple
        Shape to resize the image to. E.g.  ``(y, x)`` for ``2D`` ``(z, y, x)`` for ``3D``.

    Returns
    -------
    img : 3D/4D Numpy array
        Resized image. E.g. ``(y, x, channels)`` for ``2D`` or  ``(z, y, x, channels)`` for ``3D``.
    """
    assert img.ndim in [3, 4], f"Image must be 3D or 4D, got shape {img.shape}"
    assert (len(shape) == 2 and img.ndim == 3) or (len(shape) == 3 and img.ndim == 4), (
        "Shape is supposed to have 2 elements for 2D images and 3 elements for 3D images. "
        "Provided {} shape for {} image instead".format(shape, img.shape)
    )
    return resize(
        img,
        shape,
        order=1,
        mode="reflect",
        clip=True,
        preserve_range=True,
        anti_aliasing=True,
    )


def rotation(
    img: NDArray,
    mask: Optional[NDArray] = None,
    heat: Optional[NDArray] = None,
    angles: Union[Tuple[int, int], List[int]] = [],
    mode: str = "reflect",
    mask_type: str = "as_mask",
) -> Union[
    NDArray,
    Tuple[NDArray, Optional[NDArray], Optional[NDArray]],
]:
    """
    Apply a rotation to input ``image`` and ``mask`` (if provided).

    Parameters
    ----------
    img : 3D/4D Numpy array
        Image to rotate. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    mask : 3D/4D Numpy array, optional
        Mask to rotate. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    heat : 3D/4D Numpy array, optional
        Heatmap (float mask) to rotate. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    angles : List of ints, optional
        List of angles to choose the rotation to be made. E.g. [90,180,360].

    mode : str, optional
        How to fill up the new values created. Options: ``constant``, ``reflect``, ``wrap``, ``symmetric``.

    mask_type : str, optional
        How to treat the mask during interpolation. Either as "as_mask" (order 0) or "as_image" (order 1).

    Returns
    -------
    img : 3D/4D Numpy array
        Rotated image. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    mask : 3D/4D Numpy array, optional
        Rotated mask. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    heat : 3D/4D Numpy array, optional
        Rotated heatmap. Returned if ``mask`` is provided. E.g. ``(y, x, channels)`` for ``2D`` or
        ``(y, x, z, channels)`` for ``3D``.
    """
    assert img.ndim in (3, 4), f"Image must be 3D or 4D, got shape {img.shape}"
    if mask is not None:
        assert mask.ndim in (3, 4), f"Mask must be 3D or 4D, got shape {mask.shape}"
    if heat is not None:
        assert heat.ndim in (3, 4), f"Heat must be 3D or 4D, got shape {heat.shape}"

    # --- pick angle ---
    if not angles:
        angle = float(np.random.uniform(0.0, 360.0))
    elif isinstance(angles, tuple):
        assert len(angles) == 2, "If a tuple is provided it must have length 2"
        lo, hi = float(angles[0]), float(angles[1])
        if lo > hi:
            lo, hi = hi, lo
        angle = float(np.random.uniform(lo, hi))
    elif isinstance(angles, list):
        angle = float(random.choice(angles))
    else:
        raise ValueError("angles must be a list or a tuple")

    # Map "symmetric" to SciPy's "mirror"
    _mode = "mirror" if mode == "symmetric" else mode

    # axes for (y, x) rotation
    axes_img = (1, 0) if img.ndim == 3 else (2, 1)

    def _rotate(arr: NDArray, axes: Tuple[int, int], order: int) -> NDArray:
        orig_dtype = arr.dtype
        out = rotate(
            arr,
            angle=angle,
            axes=axes,
            reshape=False,
            order=order,
            mode=_mode,
        )
        # Cast back to original dtype
        if np.issubdtype(orig_dtype, np.floating):
            return out.astype(orig_dtype, copy=False)
        info = np.iinfo(orig_dtype)
        out = np.clip(out, info.min, info.max)
        return out.astype(orig_dtype, copy=False)

    # Image (bilinear)
    img_out = _rotate(img, axes_img, order=1)

    # Mask
    mask_out = None
    if mask is not None:
        order_mask = 0 if mask_type == "as_mask" else 1
        mask_out = _rotate(mask, axes_img, order=order_mask)

    # Heat
    heat_out = None
    if heat is not None:
        heat_out = _rotate(heat, axes_img, order=1)

    return img_out if mask is None and heat is None else (img_out, mask_out, heat_out)

def zoom(
    img: NDArray,
    zoom_range: Tuple[float, ...],
    mask: Optional[NDArray] = None,
    heat: Optional[NDArray] = None,
    zoom_in_z: bool = False,
    mode: str = "reflect",
    mask_type: str = "as_mask",
) -> Union[
    NDArray,
    Tuple[NDArray, Optional[NDArray], Optional[NDArray]],
]:
    """
    Apply zoom to input ``image`` and ``mask`` (if provided).

    Parameters
    ----------
    img : 3D/4D Numpy array
        Image to rotate. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    zoom_range : tuple of floats
        Defines minimum and maximum factors to scale the images. E.g. (0.8, 1.2).

    mask : 3D/4D Numpy array, optional
        Mask to rotate. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    heat : 3D/4D Numpy array, optional
        Heatmap (float mask) to rotate. E.g. ``(y, x, channels)`` for ``2D`` or
        ``(y, x, z, channels)`` for ``3D``.

    zoom_in_z: bool, optional
        Whether to apply or not zoom in Z axis.

    mode : str, optional
        How to fill up the new values created. Options: ``constant``, ``reflect``, ``wrap``, ``symmetric``.

    mask_type : str, optional
        How to treat the mask during interpolation. Either as "as_mask" (order 0) or "as_image" (order 1).

    Returns
    -------
    img : 3D/4D Numpy array
        Zoomed image. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    mask : 3D/4D Numpy array, optional
        Zoomed mask. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    heat : 3D/4D Numpy array, optional
        Zoomed heatmap. Returned if ``mask`` is provided. E.g. ``(y, x, channels)`` for ``2D`` or
        ``(y, x, z, channels)`` for ``3D``.
    """
    assert img.ndim in [3, 4], f"Image must be 3D or 4D, got shape {img.shape}"
    if mask is not None:
        assert mask.ndim in [3, 4], f"Mask must be 3D or 4D, got shape {mask.shape}"
    if heat is not None:
        assert heat.ndim in [3, 4], f"Heatmap must be 3D or 4D, got shape {heat.shape}"
    assert len(zoom_range) == 2, f"Zoom range is supposed to have 2 elements but provided {zoom_range} instead"
    assert zoom_range[0] <= zoom_range[1], "First element of zoom range must be lower than the second one"
    assert zoom_range[0] > 0, "Zoom range values must be greater than 0"

    zoom_selected = random.uniform(zoom_range[0], zoom_range[1])
    mask_order = 0 if mask_type == "as_mask" else 1
    if img.ndim == 4:
        z_zoom = zoom_selected if zoom_in_z else 1
        img_shape = [
            int(img.shape[0] * zoom_selected),
            int(img.shape[1] * zoom_selected),
            int(img.shape[2] * z_zoom),
        ]
        if mask is not None:
            mask_shape = [
                int(mask.shape[0] * zoom_selected),
                int(mask.shape[1] * zoom_selected),
                int(mask.shape[2] * z_zoom),
            ]
    else:
        img_shape = [
            int(img.shape[0] * zoom_selected),
            int(img.shape[1] * zoom_selected),
        ]
        if mask is not None:
            mask_shape = [
                int(mask.shape[0] * zoom_selected),
                int(mask.shape[1] * zoom_selected),
            ]

    img_shape += [img.shape[-1],]
    if mask is not None:
        mask_shape += [mask.shape[-1],]  # type: ignore
    
    if img_shape != img.shape:
        img_orig_shape = img.shape
        img = resize(
            img,
            img_shape,
            order=1,
            mode=mode,
            clip=True,
            preserve_range=True,
            anti_aliasing=True,
        )
        if mask is not None:
            mask_orig_shape = mask.shape
            mask = resize(
                mask,
                mask_shape,
                order=mask_order,
                mode=mode,
                clip=True,
                preserve_range=True,
                anti_aliasing=True,
            )
        if heat is not None:
            heat = resize(
                heat,
                img_shape[:-1],
                order=1,
                mode=mode,
                clip=True,
                preserve_range=True,
                anti_aliasing=True,
            )

        if zoom_selected >= 1:
            img = center_crop_single(img, img_orig_shape)
            if mask is not None:
                mask = center_crop_single(mask, mask_orig_shape)
            if heat is not None:
                heat = center_crop_single(heat, img_orig_shape[:-1])
        else:
            if img.ndim == 4:
                img_pad_tup = (
                    (
                        int((img_orig_shape[0] - img_shape[0]) // 2),
                        math.ceil((img_orig_shape[0] - img_shape[0]) / 2),
                    ),
                    (
                        int((img_orig_shape[1] - img_shape[1]) // 2),
                        math.ceil((img_orig_shape[1] - img_shape[1]) / 2),
                    ),
                    (
                        int((img_orig_shape[2] - img_shape[2]) // 2),
                        math.ceil((img_orig_shape[2] - img_shape[2]) / 2),
                    ),
                    (0, 0),
                )
                if mask is not None:
                    mask_pad_tup = (
                        (
                            int((mask_orig_shape[0] - mask_shape[0]) // 2),
                            math.ceil((mask_orig_shape[0] - mask_shape[0]) / 2),
                        ),
                        (
                            int((mask_orig_shape[1] - mask_shape[1]) // 2),
                            math.ceil((mask_orig_shape[1] - mask_shape[1]) / 2),
                        ),
                        (
                            int((mask_orig_shape[2] - mask_shape[2]) // 2),
                            math.ceil((mask_orig_shape[2] - mask_shape[2]) / 2),
                        ),
                        (0, 0),
                    )
            else:
                img_pad_tup = (
                    (
                        int((img_orig_shape[0] - img_shape[0]) // 2),
                        math.ceil((img_orig_shape[0] - img_shape[0]) / 2),
                    ),
                    (
                        int((img_orig_shape[1] - img_shape[1]) // 2),
                        math.ceil((img_orig_shape[1] - img_shape[1]) / 2),
                    ),
                    (0, 0),
                )
                if mask is not None:
                    mask_pad_tup = (
                        (
                            int((mask_orig_shape[0] - mask_shape[0]) // 2),
                            math.ceil((mask_orig_shape[0] - mask_shape[0]) / 2),
                        ),
                        (
                            int((mask_orig_shape[1] - mask_shape[1]) // 2),
                            math.ceil((mask_orig_shape[1] - mask_shape[1]) / 2),
                        ),
                        (0, 0),
                    )

            img = np.pad(img, img_pad_tup, mode)  # type: ignore
            if mask is not None:
                mask = np.pad(mask, mask_pad_tup, mode)  # type: ignore
            if heat is not None:
                heat = np.pad(heat, img_pad_tup, mode)  # type: ignore

    if mask is None:
        return img
    else:
        return img, mask, heat


def gamma_contrast(img: NDArray, gamma: Tuple[float, float] = (0, 1)) -> NDArray:
    """
    Apply gamma contrast to input ``image``.

    Parameters
    ----------
    img : Numpy array
        Image to transform. E.g. ``(y, x, channels)`` for ``2D`` or ``(y, x, z, channels)`` for ``3D``.

    gamma : tuple of 2 floats, optional
        Range of gamma intensity. E.g. ``(0.8, 1.3)``.

    Returns
    -------
    img : Numpy array
        Transformed image. E.g. ``(y, x, channels)`` for ``2D`` or ``(y, x, z, channels)`` for ``3D``.
    """
    assert img.ndim in [3, 4], f"Image must be 3D or 4D, got shape {img.shape}"
    assert len(gamma) == 2, "Gamma is supposed to have 2 elements but provided {} instead".format(gamma)
    assert gamma[0] <= gamma[1], "First element of gamma must be lower than the second one"
    assert gamma[0] > 0, "Gamma values must be greater than 0"
    _gamma = random.uniform(gamma[0], gamma[1])

    return adjust_gamma(np.clip(img, 0, 1), gamma=_gamma)  # type: ignore


def shear(
    image: NDArray,
    shear: tuple,
    mask: Optional[NDArray] = None,
    heat: Optional[NDArray] = None,
    cval: float = 0,
    mask_type: str = "as_mask",
    mode: str = "constant",
):
    """
    Apply a shear transformation to an image (and optional mask/heatmap).

    Parameters
    ----------
    image : 3D/4D Numpy array
        Image to shear. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    mask : 3D/4D Numpy array, optional
        Mask to shear. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    heat : 3D/4D Numpy array, optional
        Heatmap (float mask) to shear. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    shear : tuple
        Shear range (min, max) in degrees for both x and y directions.

    cval : float
        Value used for points outside the boundaries.

    mask_type : str
        How to treat the mask during interpolation. Either as "as_mask" (order 0) or "as_image" (order 1).

    mode : str
        Points outside boundaries are filled according to this mode.

    Returns
    -------
    img : 3D/4D Numpy array
        Sheared image. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    mask : 3D/4D Numpy array, optional
        Sheared mask. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    heat : 3D/4D Numpy array, optional
        Sheared heatmap. E.g. ``(y, x, channels)`` for ``2D`` or ``(y, x, z, channels)`` for ``3D``.

    """
    # Random shear (deg)
    shear_x = random.randint(shear[0], shear[1])
    shear_y = random.randint(shear[0], shear[1])

    def _restore_channels(original, warped):
        # If a single-channel input comes back as (H,W), expand to (H,W,1)
        if original is not None and original.ndim >= 3 and original.shape[-1] == 1 and warped.ndim == 2:
            return warped[..., np.newaxis]
        return warped

    def _warp_hwc(arr_hwc: NDArray, tform, order: int, cval: float, mode: str) -> NDArray:
        H, W = arr_hwc.shape[:2]
        orig_dtype = arr_hwc.dtype
        out = warp(
            arr_hwc,
            inverse_map=tform,
            cval=cval,
            mode=mode,
            order=order,
            output_shape=(H, W),
            preserve_range=True,     # keep original value range
        )
        # For masks (nearest/bilinear), cast back to original dtype
        if orig_dtype.kind != "f":
            out = out.astype(orig_dtype, copy=False)
        # Ensure 3D with channel axis if a single channel was reduced
        out = _restore_channels(arr_hwc, out)
        return out

    # Get spatial size and build transform on (H,W)
    if image.ndim == 3:         # (y, x, c)
        H_img, W_img = image.shape[:2]
        tform = _build_shear_matrix_skimage((H_img, W_img), np.deg2rad(shear_x), np.deg2rad(shear_y))
        img = _warp_hwc(image, tform, order=3, cval=cval, mode=mode)

        m = None
        if mask is not None:
            H_mask, W_mask = mask.shape[:2]
            # same transform (center/size differs? -> rebuild with mask size)
            tform_m = _build_shear_matrix_skimage((H_mask, W_mask), np.deg2rad(shear_x), np.deg2rad(shear_y))
            mask_order = 0 if mask_type == "as_mask" else 1
            m = _warp_hwc(mask, tform_m, order=mask_order, cval=cval, mode=mode)

        h = None
        if heat is not None:
            H_heat, W_heat = heat.shape[:2]
            tform_h = _build_shear_matrix_skimage((H_heat, W_heat), np.deg2rad(shear_x), np.deg2rad(shear_y))
            h = _warp_hwc(heat, tform_h, order=3, cval=cval, mode=mode)

        return img, m, h

    elif image.ndim == 4:       # (z, y, x, c)
        Z, H, W, C = image.shape
        tform = _build_shear_matrix_skimage((H, W), np.deg2rad(shear_x), np.deg2rad(shear_y))

        # Image
        img_out = np.empty_like(image)
        for z in range(Z):
            img_out[z] = _warp_hwc(image[z], tform, order=3, cval=cval, mode=mode)

        # Mask
        m_out = None
        if mask is not None:
            Zm, Hm, Wm, Cm = mask.shape
            assert (Zm, Hm, Wm) == (Z, H, W), "mask shape must match image (z,y,x)"
            tform_m = _build_shear_matrix_skimage((Hm, Wm), np.deg2rad(shear_x), np.deg2rad(shear_y))
            mask_order = 0 if mask_type == "as_mask" else 1
            m_out = np.empty_like(mask)
            for z in range(Z):
                m_out[z] = _warp_hwc(mask[z], tform_m, order=mask_order, cval=cval, mode=mode)

        # Heat
        h_out = None
        if heat is not None:
            Zh, Hh, Wh, Ch = heat.shape
            assert (Zh, Hh, Wh) == (Z, H, W), "heat shape must match image (z,y,x)"
            tform_h = _build_shear_matrix_skimage((Hh, Wh), np.deg2rad(shear_x), np.deg2rad(shear_y))
            h_out = np.empty_like(heat)
            for z in range(Z):
                h_out[z] = _warp_hwc(heat[z], tform_h, order=3, cval=cval, mode=mode)

        return img_out, m_out, h_out

    else:
        raise ValueError(f"Unsupported image ndim: {image.ndim} (expected 3 or 4)")

def shift(
    image: NDArray,
    mask: Optional[NDArray] = None,
    heat: Optional[NDArray] = None,
    shift_range: Optional[tuple] = None,
    cval: float = 0,
    mask_type: str = "as_mask",
    mode: str = "constant",
):
    """
    Shift an image (and optional mask/heatmap) by a random amount within a range.

    Parameters
    ----------
    image : 3D/4D Numpy array
        Image to shift. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    mask : 3D/4D Numpy array, optional
        Mask to shift. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    heat : 3D/4D Numpy array, optional
        Heatmap (float mask) to shift. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    shift_range : Optional[tuple]
        Range (min, max) for random shift in both x and y directions.

    cval : float
        Value used for points outside the boundaries.

    mask_type : str
        How to treat the mask during interpolation. Either as "as_mask" (order 0) or "as_image" (order 1).

    mode : str
        Points outside boundaries are filled according to this mode.

    Returns
    -------
    img : 3D/4D Numpy array
        Shifted image. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    mask : 3D/4D Numpy array, optional
        Shifted mask. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    heat : 3D/4D Numpy array, optional
        Shifted heatmap. E.g. ``(y, x, channels)`` for ``2D`` or ``(y, x, z, channels)`` for ``3D``.

    """
    assert image.ndim in (3, 4), f"Image must be 3D or 4D, got {image.shape}"
    if mask is not None:
        assert mask.ndim in (3, 4), f"Mask must be 3D or 4D, got {mask.shape}"
    if heat is not None:
        assert heat.ndim in (3, 4), f"Heat must be 3D or 4D, got {heat.shape}"
    assert shift_range is not None and len(shift_range) == 2, \
        f"shift_range must be (min, max); got {shift_range}"

    # Get spatial size for (y, x)
    if image.ndim == 3:        # (y, x, c)
        h, w = image.shape[:2]
    else:                      # (z, y, x, c)
        h, w = image.shape[1:3]

    # Sample a percentage and convert to pixel shifts
    shift_perc = random.uniform(shift_range[0], shift_range[1])
    x_pix = int(round(shift_perc * w))
    y_pix = int(round(shift_perc * h))

    # Build per-array shift tuples (keep z and c fixed)
    def get_shift_tuple(arr, x, y):
        if arr.ndim == 3:           # (y, x, c)
            return (y, x, 0)
        elif arr.ndim == 4:         # (z, y, x, c)
            return (0, y, x, 0)
        else:
            raise ValueError(f"Unsupported ndim: {arr.ndim}")

    # Shift image
    img = shift_nd(image, get_shift_tuple(image, x_pix, y_pix),
                   order=3, mode=mode, cval=cval)

    # Shift mask
    if mask is not None:
        order_mask = 0 if mask_type == "as_mask" else 1
        mask = shift_nd(mask, get_shift_tuple(mask, x_pix, y_pix),
                        order=order_mask, mode=mode, cval=cval)

    # Shift heatmap
    if heat is not None:
        heat = shift_nd(heat, get_shift_tuple(heat, x_pix, y_pix),
                        order=3, mode=mode, cval=cval)

    return img, mask, heat


def flip_horizontal(image: NDArray, mask: Optional[NDArray] = None, heat: Optional[NDArray] = None):
    """
    Flip an image (and optional mask/heatmap) horizontally (left-right).

    Parameters
    ----------
    image : 3D/4D Numpy array
        Image to flip. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    mask : 3D/4D Numpy array, optional
        Mask to flip. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    heat : 3D/4D Numpy array, optional
        Heatmap (float mask) to flip. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    Returns
    -------
    img : 3D/4D Numpy array
        Flipped image. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    mask : 3D/4D Numpy array or None
        Flipped mask if provided, else None.

    heat : 3D/4D Numpy array or None
        Flipped heatmap if provided, else None.
    """
    assert image.ndim in (3, 4), f"Image must be 3D or 4D, got {image.shape}"
    if mask is not None:
        assert mask.ndim in (3, 4), f"Mask must be 3D or 4D, got {mask.shape}"
    if heat is not None:
        assert heat.ndim in (3, 4), f"Heatmap must be 3D or 4D, got {heat.shape}"
    img = image[:, ::-1]
    mask = mask[:, ::-1] if mask is not None else None
    heat = heat[:, ::-1] if heat is not None else None
    return img, mask, heat


def flip_vertical(image: NDArray, mask: Optional[NDArray] = None, heat: Optional[NDArray] = None):
    """
    Flip an image (and optional mask/heatmap) vertically (up-down).

    Parameters
    ----------
    image : 3D/4D Numpy array
        Image to flip. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    mask : 3D/4D Numpy array, optional
        Mask to flip. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    heat : 3D/4D Numpy array, optional
        Heatmap (float mask) to flip. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    Returns
    -------
    img : 3D/4D Numpy array
        Flipped image. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    mask : 3D/4D Numpy array, optional
        Flipped mask. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    heat : 3D/4D Numpy array, optional
        Flipped heatmap. E.g. ``(y, x, channels)`` for ``2D`` or ``(y, x, z, channels)`` for ``3D``.
    """
    assert image.ndim in (3, 4), f"Image must be 3D or 4D, got {image.shape}"
    if mask is not None:
        assert mask.ndim in (3, 4), f"Mask must be 3D or 4D, got {mask.shape}"
    if heat is not None:
        assert heat.ndim in (3, 4), f"Heatmap must be 3D or 4D, got {heat.shape}"
    img = image[:, :, ::-1]
    mask = mask[:, :, ::-1] if mask is not None else None
    heat = heat[:, :, ::-1] if heat is not None else None
    return img, mask, heat


def gaussian_blur(image: NDArray, sigma: float | tuple = (0.5, 1.5)):
    """
    Apply Gaussian blur to an image.

    Parameters
    ----------
    image : Numpy array
        Image to Blur. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    sigma : float or tuple
        Standard deviation for Gaussian kernel. If tuple, a random value is chosen from the range.

    Returns
    -------
    img : NDArray
        Blurred image. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.
    """
    # Needed for elastic as integer
    if isinstance(sigma, tuple):
        sigma = random.uniform(sigma[0], sigma[1])

    return gaussian(image, sigma=sigma)


def median_blur(image: NDArray, k_range: Optional[tuple] = None):
    """
    Apply median blur to an image.

    Parameters
    ----------
    image : 3D/4D Numpy array
        Image to Blur. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    k_range : Optional[tuple]
        Range (min, max) for random kernel size (must be odd).

    Returns
    -------
    img : NDArray
        Blurred image. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.
    """
    assert image.ndim in (3, 4), f"Image must be 3D or 4D, got {image.shape}"
    if k_range is None or len(k_range) != 2:
        raise ValueError("k_range must be provided and have length 2")
    
    k = int(random.randint(k_range[0], k_range[1]))
    if k % 2 == 0:
        k += 1
    if k <= 1:
        return image

    # Build filter window that does NOT mix z or channels
    if image.ndim == 3:           # (y, x, c)
        size = (k, k, 1)
    else:                         # (z, y, x, c)
        size = (1, k, k, 1)

    return median_filter(image, size=size)


def motion_blur(image: NDArray, k_range: Optional[tuple] = None):
    """
    Apply motion blur to an image.

    Parameters
    ----------
    image : 3D/4D Numpy array
        Image to flip. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    k_range : Optional[tuple]
        Range (min, max) for random kernel size (must be odd).

    Returns
    -------
    img : NDArray
        Blurred image. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.
    """
    assert image.ndim in (3, 4), f"Image must be 3D or 4D, got {image.shape}"
    if k_range is None or len(k_range) != 2:
        raise ValueError("k_range must be provided and have length 2")

    if image.size == 0:
        return image

    # Sample kernel size (must be odd)
    k = int(random.randint(k_range[0], k_range[1]))
    if k % 2 == 0:
        k += 1

    # Sample motion direction (angle) and intensity direction
    angle = int(random.randint(0, 359))
    direction = float(random.uniform(-1.0, 1.0))

    # Build a vertical line kernel, then rotate it
    base = np.zeros((k, k), dtype=np.float32)
    base[:, k // 2] = np.linspace(direction, 1.0 - direction, num=k).astype(np.float32)

    # Rotate to the sampled angle; normalize
    kernel_rot = rotate(base, angle=angle)
    kernel = kernel_rot.astype(np.float32)
    s = kernel.sum()
    if s != 0.0:
        kernel /= s
    else:
        kernel[k // 2, k // 2] = 1.0  # degenerate fallback

    # Apply blur
    if image.ndim == 3:  # (y, x, c)
        H, W, C = image.shape
        out = np.empty_like(image)
        for c in range(C):
            out[..., c] = cv2.filter2D(image[..., c], ddepth=-1, kernel=kernel)
        return out

    else:               # (z, y, x, c)
        Z, H, W, C = image.shape
        out = np.empty_like(image)
        for z in range(Z):
            for c in range(C):
                out[z, :, :, c] = cv2.filter2D(image[z, :, :, c], ddepth=-1, kernel=kernel)
        return out

def dropout(
    image: NDArray, drop_range: tuple = (0.1, 0.2), random_state: Optional[np.random.RandomState] = None
) -> NDArray:
    """
    Randomly set a fraction of pixels in the image to zero (dropout).

    Parameters
    ----------
    image : 3D/4D Numpy array
        Image to apply dropout. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    drop_range : tuple
        Range for dropout probability. A value is randomly chosen from this range.

    random_state : Optional[np.random.RandomState]
        Random state for reproducibility.

    Returns
    -------
    img : 3D/4D Numpy array
        Image after dropout. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.
    """
    assert image.ndim in (3, 4), f"Image must be 3D or 4D, got {image.shape}"
    assert len(drop_range) == 2, f"Drop range must have 2 elements, got {drop_range}"
    rng = np.random if random_state is None else random_state
    p = rng.uniform(*drop_range)

    if p == 0:
        return image

    if image.ndim == 2:
        mask_shape = image.shape
    else:
        mask_shape = image.shape[:-1]

    keep_mask = rng.binomial(1, 1.0 - p, size=mask_shape).astype(image.dtype)

    if image.ndim > len(mask_shape):
        keep_mask = np.expand_dims(keep_mask, axis=-1)

    image = image * keep_mask
    return image


def elastic(
    image: NDArray,
    mask: Optional[NDArray] = None,
    heat: Optional[NDArray] = None,
    alpha: float | tuple = 14,
    sigma: float = 4,
    mask_type: str = "as_mask",
    cval: float = 0,
    mode: str = "constant",
    random_seed=None,
) -> Tuple[NDArray, Optional[NDArray], Optional[NDArray]]:
    """
    Apply elastic deformation to an image (and optional mask/heatmap).

    Parameters
    ----------
    image : 3D/4D Numpy array
        Image to deform. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    mask : 3D/4D Numpy array, optional
        Mask to deform. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    heat : 3D/4D Numpy array, optional
        Heatmap (float mask) to deform. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    alpha : float, optional
        Scaling factor for deformation intensity.

    sigma : float, optional
        Standard deviation for Gaussian filter.

    cval : float, optional
        Value used for points outside the boundaries.

    mode : str, optional
        Points outside boundaries are filled according to this mode.

    mask_type : str, optional
        How to treat the mask during interpolation. Either as "as_mask" (order 0) or "as_image" (order 1).

    random_seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    img : 3D/4D Numpy array
        Deformed image. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    mask : 3D/4D Numpy array, optional
        Deformed mask. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    heat : 3D/4D Numpy array, optional
        Deformed heatmap. E.g. ``(y, x, channels)`` for ``2D`` or ``(y, x, z, channels)`` for ``3D``.
    """
    assert image.ndim in (3, 4), f"Image must be 3D or 4D, got {image.shape}"
    if mask is not None:
        assert mask.ndim in (3, 4), f"Mask must be 3D or 4D, got {mask.shape}"
    if heat is not None:
        assert heat.ndim in (3, 4), f"Heatmap must be 3D or 4D, got {heat.shape}"
    if random_seed is not None:
        np.random.seed(random_seed)

    def warp_with_new_displacement(
        tensor: NDArray,
        alpha: float,
        sigma: float,
        order: int,
        cval: float,
        mode: str
    ) -> NDArray:
        """
        Apply elastic deformation to a tensor.
        Works for (H, W, C) and (Z, H, W, C). The displacement is generated in (H, W)
        and broadcast to all channels (and all z-slices if 4D).
        """
        assert tensor.ndim in (3, 4), f"Expected 3D or 4D tensor, got {tensor.shape}"

        # Pick spatial (H, W) depending on layout
        if tensor.ndim == 3:             # (H, W, C)
            H, W = tensor.shape[:2]
        else:                             # (Z, H, W, C)
            H, W = tensor.shape[1:3]

        # Choose padding kernel size based on sigma (same logic as before)
        if sigma < 3.0:
            ksize = 3.3 * sigma  # ~99% weight
        elif sigma < 5.0:
            ksize = 2.9 * sigma  # ~97% weight
        else:
            ksize = 2.6 * sigma  # ~95% weight
        ksize = int(max(ksize, 5))
        ksize = ksize + 1 if (ksize % 2 == 0) else ksize
        padding = ksize

        # Build padded random fields, smooth them, then crop back to (H, W)
        H_pad = H + 2 * padding
        W_pad = W + 2 * padding

        rng = np.random.rand(2 * H_pad, W_pad).astype(np.float32) * 2 - 1
        dx_unsmoothed = rng[:H_pad, :]
        dy_unsmoothed = rng[H_pad:, :]

        # Use skimage.filters.gaussian (already imported as `gaussian`)
        dx = gaussian(dx_unsmoothed, sigma=sigma).astype(np.float32) * alpha
        dy = gaussian(dy_unsmoothed, sigma=sigma).astype(np.float32) * alpha

        if padding > 0:
            dx = dx[padding:-padding, padding:-padding]
            dy = dy[padding:-padding, padding:-padding]

        # Let _map_coordinates handle both 3D and 4D layouts
        return _map_coordinates(tensor, dx, dy, order=order, cval=cval, mode=mode)

    alphas, sigmas = _draw_samples(alpha, sigma, nb_images=1)
    alpha_val = alphas[0]
    sigma_val = sigmas[0]

    img = warp_with_new_displacement(image, alpha_val, sigma_val, 3, cval, mode)
    if mask is not None:
        mask_order = 0 if mask_type == "as_mask" else 1
        mask = warp_with_new_displacement(mask, alpha_val, sigma_val, mask_order, cval, mode)
    heat = warp_with_new_displacement(heat, alpha_val, sigma_val, 3, cval, mode) if heat is not None else heat

    return img, mask, heat


## Helpers
def _build_shear_matrix_skimage(image_shape: tuple, shear_x_rad: float, shear_y_rad: float, shift_add: tuple = (0.5, 0.5)) -> ProjectiveTransform:
    """
    Build an affine transformation matrix for shear augmentation using skimage.

    Parameters
    ----------
    image_shape : tuple
        Shape of the image (height, width, ...).

    shear_x_rad : float
        Shear angle in radians for the x direction.

    shear_y_rad : float
        Shear angle in radians for the y direction.

    shift_add : tuple, optional
        Additional shift to apply when centering the transformation.

    Returns
    -------
    matrix : AffineTransform
        Affine transformation matrix for shear.
    """
    h, w = image_shape[:2]
    if h == 0 or w == 0:
        return AffineTransform()

    shift_y = h / 2.0 - shift_add[0]
    shift_x = w / 2.0 - shift_add[1]

    matrix_to_topleft = AffineTransform(translation=[-shift_x, -shift_y])
    matrix_to_center = AffineTransform(translation=[shift_x, shift_y])

    matrix_shear_x = AffineTransform(shear=shear_x_rad)

    matrix_shear_y_rot = AffineTransform(rotation=-np.pi / 2)
    matrix_shear_y = AffineTransform(shear=shear_y_rad)
    matrix_shear_y_rot_inv = AffineTransform(rotation=np.pi / 2)

    # Correct order: shear_x then shear_y (via rotated frame)
    matrix = (
        matrix_to_topleft
        + matrix_shear_x
        + matrix_shear_y_rot
        + matrix_shear_y
        + matrix_shear_y_rot_inv
        + matrix_to_center
    )

    return matrix


def _normalize_cv2_input_arr_(arr: NDArray) -> NDArray:
    """
    Ensure array is contiguous and owns its data for cv2 functions.

    Parameters
    ----------
    arr : NDArray
        Input array.
    """
    flags = arr.flags
    if not flags["OWNDATA"]:
        arr = np.copy(arr)
        flags = arr.flags
    if not flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr


def _draw_samples(alpha: float | tuple, sigma: float | tuple, nb_images: int) -> tuple:
    """
    Draw samples for alpha and sigma parameters.

    Parameters
    ----------
    alpha : float or tuple
        Alpha parameter or range (min, max).

    sigma : float or tuple
        Sigma parameter or range (min, max).

    nb_images : int
        Number of samples to draw.

    Returns
    -------
    alphas : NDArray
        Array of drawn alpha values.

    sigmas : NDArray
        Array of drawn sigma values.
    """

    # Use np.random for all randomness
    def draw_param(param: float | tuple, size: tuple) -> NDArray:
        if isinstance(param, (int, float)):
            out = np.full(size, param)
        elif isinstance(param, str):
            out = np.array([param] * size[0], dtype=object)
        elif isinstance(param, tuple):
            out = np.random.uniform(param[0], param[1], size=size) if len(param) == 2 else np.full(size, param[0])
        else:
            out = np.full(size, param)
        return out

    alphas = draw_param(alpha, (nb_images,))
    sigmas = draw_param(sigma, (nb_images,))

    return alphas, sigmas


_MAPPING_MODE_SCIPY_CV2 = {
    "constant": cv2.BORDER_CONSTANT,
    "edge": cv2.BORDER_REPLICATE,
    "symmetric": cv2.BORDER_REFLECT,
    "reflect": cv2.BORDER_REFLECT_101,
    "wrap": cv2.BORDER_WRAP,
    "nearest": cv2.BORDER_REPLICATE,
}

_MAPPING_ORDER_SCIPY_CV2 = {
    0: cv2.INTER_NEAREST,
    1: cv2.INTER_LINEAR,
    2: cv2.INTER_CUBIC,
    3: cv2.INTER_CUBIC,
    4: cv2.INTER_CUBIC,
    5: cv2.INTER_CUBIC,
}


def _map_coordinates(image: NDArray, dx: NDArray, dy: NDArray, order: int = 1, cval: float = 0, mode: str = "constant") -> NDArray:
    """
    Map input image to new coordinates defined by displacement fields dx and dy.

    Parameters
    ----------
    image : NDArray
        Input image array.

    dx : NDArray
        Displacement field in x direction.

    dy : NDArray
        Displacement field in y direction.

    order : int
        Interpolation order.

    cval : float
        Value used for points outside the boundaries.

    mode : str
        Points outside boundaries are filled according to this mode.

    Returns
    -------
    result : NDArray
        Transformed image array.
    """

    if image.size == 0:
        return np.copy(image)

    dx = dx.astype(np.float32)
    dy = dy.astype(np.float32)

    if order == 0 and image.dtype.name in ["uint64", "int64"]:
        raise Exception(
            "dtypes uint64 and int64 are only supported in "
            "ElasticTransformation for order=0, got order=%d with "
            "dtype=%s." % (order, image.dtype.name)
        )
    assert image.ndim in (3, 4), f"Expected 3D or 4D image, got {image.ndim}D with shape {image.shape}"

    # cv2 params
    border_mode = _MAPPING_MODE_SCIPY_CV2[mode]
    interpolation = _MAPPING_ORDER_SCIPY_CV2[order]
    if image.dtype.kind == "f":
        cval_cast = float(cval)
    else:
        cval_cast = int(cval)

    def _make_maps(h: int, w: int, dx2: NDArray, dy2: NDArray):
        """Build OpenCV remap maps for a single 2D field."""
        y, x = np.meshgrid(
            np.arange(h, dtype=np.float32),
            np.arange(w, dtype=np.float32),
            indexing="ij",
        )
        x_shifted = x - dx2
        y_shifted = y - dy2

        if interpolation == cv2.INTER_NEAREST:
            return x_shifted, y_shifted
        else:
            # returns (map1, map2) as optimized fixed-point/float maps
            return cv2.convertMaps(x_shifted, y_shifted, cv2.CV_32FC1, nninterpolation=False)

    def _remap_hwcn(arr_hwc: NDArray, map1: NDArray, map2: NDArray) -> NDArray:
        """
        Apply cv2.remap to (H, W, C) with any C (remap supports up to 4 channels at once).
        """
        H, W, C = arr_hwc.shape

        if C <= 4:
            border_val = (cval_cast,) * min(max(C, 1), 4)
            res = cv2.remap(
                _normalize_cv2_input_arr_(arr_hwc),
                map1,
                map2,
                interpolation=interpolation,
                borderMode=border_mode,
                borderValue=border_val,
            )
            if res.ndim == 2:
                res = res[..., np.newaxis]
            return res

        # chunk channels in groups of up to 4
        chunks = []
        for i in range(0, C, 4):
            sub = arr_hwc[:, :, i : i + 4]
            border_val = (cval_cast,) * (sub.shape[-1])
            res = cv2.remap(
                _normalize_cv2_input_arr_(sub),
                map1,
                map2,
                interpolation=interpolation,
                borderMode=border_mode,
                borderValue=border_val,
            )
            if res.ndim == 2:
                res = res[..., np.newaxis]
            chunks.append(res)
        return np.concatenate(chunks, axis=2)

    if image.ndim == 3:
        # (H, W, C)
        H, W, C = image.shape
        # accept dx/dy as (H,W)
        assert dx.shape == (H, W) and dy.shape == (H, W), \
            f"For 3D image (H,W,C), dx/dy must be (H,W); got dx {dx.shape}, dy {dy.shape}"

        map1, map2 = _make_maps(H, W, dx, dy)
        return _remap_hwcn(np.copy(image), map1, map2)

    else:
        # (Z, H, W, C)
        Z, H, W, C = image.shape
        result = np.empty_like(image)

        # dx/dy: either (H,W) or (Z,H,W)
        per_slice = dx.ndim == 3 and dy.ndim == 3
        if per_slice:
            assert dx.shape == (Z, H, W) and dy.shape == (Z, H, W), \
                f"For per-slice fields, dx/dy must be (Z,H,W); got dx {dx.shape}, dy {dy.shape}"
        else:
            assert dx.shape == (H, W) and dy.shape == (H, W), \
                f"For broadcast fields, dx/dy must be (H,W); got dx {dx.shape}, dy {dy.shape}"
            # precompute shared maps once
            shared_map1, shared_map2 = _make_maps(H, W, dx, dy)

        for z in range(Z):
            if per_slice:
                map1, map2 = _make_maps(H, W, dx[z], dy[z])
            else:
                map1, map2 = shared_map1, shared_map2

            slice_res = _remap_hwcn(image[z], map1, map2)
            result[z] = slice_res

        return result
