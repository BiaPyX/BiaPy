import numpy as np
import random
import cv2
from scipy.ndimage import median_filter, interpolation, shift
from skimage.transform import AffineTransform, warp
from typing import Optional, Union, Tuple

_MAPPING_MODE_SCIPY_CV2 = {
    "constant": cv2.BORDER_CONSTANT,
    "edge":     cv2.BORDER_REPLICATE,
    "symmetric":cv2.BORDER_REFLECT,
    "reflect":  cv2.BORDER_REFLECT_101,
    "wrap":     cv2.BORDER_WRAP,
    "nearest": cv2.BORDER_REPLICATE,
}

_MAPPING_ORDER_SCIPY_CV2 = {
    0: cv2.INTER_NEAREST,
    1: cv2.INTER_LINEAR,
    2: cv2.INTER_CUBIC,
    3: cv2.INTER_CUBIC,
    4: cv2.INTER_CUBIC,
    5: cv2.INTER_CUBIC
}

try:
    import six.moves as sm
except ImportError:
    class sm:
        @staticmethod
        def xrange(*args, **kwargs):
            return range(*args, **kwargs)

# Done
def rotation_raw(image, angle, mask=None, heat=None, backend="cv2", cval=0, order_image=1, order_mask=0, order_heat=1, mode="constant"):
    """
    Rotate image (and optional mask, heat) by `angle` degrees about center.
    backend: "skimage" (uses _warp_affine_arr_skimage) or "cv2".
    angle: float or (min, max) tuple/list for random sampling.
    """
    # If angle is a range, sample a float value
    if isinstance(angle, (tuple, list)) and len(angle) == 2:
        angle = random.uniform(angle[0], angle[1])
    h, w = image.shape[:2]
    
    # create affine
    center = (np.array((w, h)) - 1) / 2.0
    angle_rad = np.deg2rad(angle)
    tform = (
        AffineTransform(translation=-center) +
        AffineTransform(rotation=angle_rad) +
        AffineTransform(translation=center)
    )

    if backend == "skimage":
        img_aug = _warp_affine_arr_skimage(image, tform, cval=cval, mode=mode, order=order_image, output_shape=(h, w))
        mask_aug = _warp_affine_arr_skimage(mask, tform, cval=cval, mode=mode, order=order_mask, output_shape=(h, w)) if mask is not None else None
        heat_aug = _warp_affine_arr_skimage(heat, tform, cval=cval, mode=mode, order=order_heat, output_shape=(h, w)) if heat is not None else None
        return img_aug, mask_aug, heat_aug

    elif backend == "cv2":
        img_aug = np.squeeze(_warp_affine_arr_cv2(image, tform, cval=cval, mode=mode, order=order_image, output_shape=(h, w)))
        mask_aug = np.squeeze(_warp_affine_arr_cv2(mask, tform, cval=cval, mode=mode, order=order_mask, output_shape=(h, w))) if mask is not None else None
        heat_aug = np.squeeze(_warp_affine_arr_cv2(heat, tform, cval=cval, mode=mode, order=order_heat, output_shape=(h, w))) if heat is not None else None
        return img_aug, mask_aug, heat_aug
    else:
        raise ValueError(f"Unknown backend {backend}")

# Done
def shift_raw(image, mask=None, heat=None, x=None, y=None, shift_range=None, cval=0, mode="constant", order_image=1, order_mask=0, order_heat=1):

    # If shift_range is provided, sample x and y from it
    if shift_range is not None:
        if isinstance(shift_range, (tuple, list)) and len(shift_range) == 2:
            x = random.uniform(shift_range[0], shift_range[1])
            y = random.uniform(shift_range[0], shift_range[1])
        else:
            raise ValueError("shift_range must be a tuple or list of length 2")

    if x is None or y is None:
        raise ValueError("x and y must be specified if shift_range is not provided")

    # Define shift tuples based on actual ndim of each input
    def get_shift_tuple(array, x, y):
        if array.ndim == 2:
            return (y, x)
        elif array.ndim == 3:
            return (y, x, 0)  # do not shift channel dimension
        else:
            raise ValueError(f"Unsupported ndim: {array.ndim}")

    shift_tuple_image = get_shift_tuple(image, x, y)
    img_aug = shift(image, shift_tuple_image, order=order_image, mode=mode, cval=cval)

    mask_aug = None
    if mask is not None:
        shift_tuple_mask = get_shift_tuple(mask, x, y)
        mask_aug = shift(mask, shift_tuple_mask, order=order_mask, mode=mode, cval=cval)

    heat_aug = None
    if heat is not None:
        shift_tuple_heat = get_shift_tuple(heat, x, y)
        heat_aug = shift(heat, shift_tuple_heat, order=order_heat, mode=mode, cval=cval)

    return img_aug, mask_aug, heat_aug

# Done
def flip_horizontal_raw(image, mask=None, heat=None):
    img_aug = np.fliplr(image)
    mask_aug = np.fliplr(mask) if mask is not None else None
    heat_aug = np.fliplr(heat) if heat is not None else None
    return img_aug, mask_aug, heat_aug

# Done
def flip_vertical_raw(image, mask=None, heat=None):
    img_aug = np.flipud(image)
    mask_aug = np.flipud(mask) if mask is not None else None
    heat_aug = np.flipud(heat) if heat is not None else None
    return img_aug, mask_aug, heat_aug

# Done
def shear_raw(image, shear, mask=None, heat=None, cval=0, order_image=1, order_mask=0, order_heat=1, mode="constant", backend="cv2"):
    """Internal helper to parse shear parameters and call the function."""
    shear_x = 0
    shear_y = 0
    if isinstance(shear, (int, float)):
        shear_x = shear
    elif isinstance(shear, tuple) and len(shear) == 2:
        shear_x, shear_y = shear
    elif isinstance(shear, dict):
        shear_x = shear.get('x', 0)
        shear_y = shear.get('y', 0)
    else:
        raise ValueError("Invalid shear parameter format. Use a single number, a tuple (x, y), or a dictionary {'x': val_x, 'y': val_y}.")
    
    def _restore_channels(original, warped):
        if original is not None and original.ndim == 3 and warped.ndim == 2:
            return warped[..., np.newaxis]
        return warped

    h_img, w_img = image.shape[:2]
    h_mask, w_mask = mask.shape[:2] if mask is not None else (None, None)
    h_heat, w_heat = heat.shape[:2] if heat is not None else (None, None)

    if backend == "skimage":
        tform = build_shear_matrix_skimage(image.shape, np.deg2rad(shear_x), np.deg2rad(shear_y))
        img_aug = _restore_channels(image, _warp_affine_arr_skimage(image, tform, cval=cval, mode=mode, order=order_image, output_shape=(h_img, w_img)))
        mask_aug = _restore_channels(mask, _warp_affine_arr_skimage(mask, tform, cval=cval, mode=mode, order=order_mask, output_shape=(h_mask, w_mask))) if mask is not None else None
        heat_aug = _restore_channels(heat, _warp_affine_arr_skimage(heat, tform, cval=cval, mode=mode, order=order_heat, output_shape=(h_heat, w_heat))) if heat is not None else None
        return img_aug, mask_aug, heat_aug
    elif backend == "cv2":
        tform = build_shear_matrix_cv2(image.shape, np.deg2rad(shear_x), np.deg2rad(shear_y))
        img_aug = _restore_channels(image, _warp_affine_arr_cv2(image, tform, cval=cval, mode=mode, order=order_image, output_shape=(h_img, w_img)))
        mask_aug = _restore_channels(mask, _warp_affine_arr_cv2(mask, tform, cval=cval, mode=mode, order=order_mask, output_shape=(h_mask, w_mask))) if mask is not None else None
        heat_aug = _restore_channels(heat, _warp_affine_arr_cv2(heat, tform, cval=cval, mode=mode, order=order_heat, output_shape=(h_heat, w_heat))) if heat is not None else None

        return img_aug, mask_aug, heat_aug
    else:
        raise ValueError(f"Unknown backend {backend}")

# Done
def gaussian_blur_raw(image, sigma, mask=None, heat=None):
    # If sigma is a range, sample a float value
    if isinstance(sigma, (tuple, list)) and len(sigma) == 2:
        sigma = random.uniform(sigma[0], sigma[1])
    if isinstance(sigma, bool):
        sigma = float(sigma)
    ksize = int(max(3.3 * sigma if sigma < 3.0 else 2.9 * sigma if sigma < 5.0 else 2.6 * sigma, 5))
    ksize = ksize + 1 if ksize % 2 == 0 else ksize
    img_aug = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=float(sigma), sigmaY=float(sigma), borderType=cv2.BORDER_REFLECT_101)
    return img_aug, mask, heat

# Done
def median_blur_raw(image, mask=None, heat=None, k_range=None):
    # If k_range is provided, sample k from the range
    if k_range is not None:
        if isinstance(k_range, (tuple, list)) and len(k_range) == 2:
            k = random.randint(k_range[0], k_range[1])
        elif isinstance(k_range, int):
            k = k_range
        else:
            raise ValueError("k_range must be a tuple or list of length 2 or an int")
    if k is None:
        raise ValueError("k must be specified if k_range is not provided")

    # Ensure k is odd
    if k % 2 == 0:
        k += 1
    if k <= 1:
        return image, mask, heat  # no change

    has_zero_sized_axes = (image.size == 0)
    if k > 1 and not has_zero_sized_axes:
        if image.ndim == 2:
            image_aug = median_filter(image, size=k)
        elif image.ndim == 3 and image.shape[-1] == 1:
            image_aug = median_filter(image[..., 0], size=k)[..., np.newaxis]
        else:
            channels = [median_filter(image[..., c], size=k) for c in range(image.shape[-1])]
            image_aug = np.stack(channels, axis=-1)

        return image_aug, mask, heat
    else:
        return image, mask, heat

# Done
def motion_blur_raw(image, mask=None, heat=None, k_range=None, angle=50, direction=-1):
    input_shape = image.shape
    nb_channels = 1 if len(input_shape) == 2 else input_shape[2]
    # If k_range is provided, sample k from the range
    if k_range is not None:
        if isinstance(k_range, (tuple, list)) and len(k_range) == 2:
            k = random.randint(k_range[0], k_range[1])
        elif isinstance(k_range, int):
            k = k_range
        else:
            raise ValueError("k_range must be a tuple or list of length 2")
    # Ensure k is odd
    k_sample = k if k % 2 != 0 else k + 1
    direction_sample = np.clip(direction, -1.0, 1.0)
    direction_sample = (direction_sample + 1.0) / 2.0
    matrix = np.zeros((k_sample, k_sample), dtype=np.float32)
    matrix[:, k_sample // 2] = np.linspace(float(direction_sample), 1.0 - float(direction_sample), num=k_sample)
    matrix_ours, _, _ = rotation_raw(matrix, angle=angle)
    matrix = matrix_ours.astype(np.float32) / np.sum(matrix_ours)
    matrix = [matrix] * nb_channels
    if image.size == 0:
        return image, mask, heat
    if nb_channels == 1 and len(matrix) == 1:
        image = cv2.filter2D(image, -1, matrix[0])
    else:
        for c in range(nb_channels):
            arr_channel = np.copy(image[..., c])
            image[..., c] = cv2.filter2D(arr_channel, -1, matrix[c])
    if len(input_shape) == 3 and image.ndim == 2:
        image = image[:, :, np.newaxis]
    mask_aug = mask
    if mask is not None:
        mask_aug = mask
    heat_aug = heat
    if heat is not None:
        heat_aug = heat
    return image, mask_aug, heat_aug

# Done
def dropout_raw(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    heat: Optional[np.ndarray] = None,
    drop_range: Union[float, Tuple[float, float]] = 0.1,
    per_channel: bool = False,
    random_state: Optional[np.random.RandomState] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    
    rng = np.random if random_state is None else random_state
    p = rng.uniform(*drop_range) if isinstance(drop_range, tuple) else float(drop_range)

    if p == 0:
        return image, mask, heat

    if per_channel:
        # Create a separate mask for each channel.
        mask_shape = image.shape
    else:
        # Create a single mask for all channels. This must handle
        # images with and without a channel axis correctly.
        if image.ndim == 2:  # Grayscale case: (h, w)
            mask_shape = image.shape
        else:  # Multi-channel case: (h, w, c) or (d, h, w, c)
            mask_shape = image.shape[:-1]

    # Create the dropout mask.
    keep_mask = rng.binomial(1, 1.0 - p, size=mask_shape).astype(image.dtype)

    # we need to add it back to the mask for broadcasting.
    if not per_channel and image.ndim > len(mask_shape):
        keep_mask = np.expand_dims(keep_mask, axis=-1)
        
    image_out = image * keep_mask

    return image_out, mask, heat

## not used
def zoom_raw(image, factor, mask=None, heat=None, backend="cv2", cval=0, order_image=1,
                        order_mask=0, order_heat=1, mode="constant"):
    """
    Zoom about center by `factor`. 1.0=no change, >1 zoom in, <1 zoom out.
    """

    h, w = image.shape[:2]
    center_y, center_x = (h - 1) / 2.0, (w - 1) / 2.0

    tform = AffineTransform(translation=(-center_x, -center_y))
    tform += AffineTransform(scale=(factor, factor))
    tform += AffineTransform(translation=(center_x, center_y))

    output_shape=(h, w)
    if backend == "skimage":
        img_aug = _warp_affine_arr_skimage(image, tform, cval=cval, mode=mode, order=order_image, output_shape=output_shape)
        msk_aug = _warp_affine_arr_skimage(mask, tform, cval=cval, mode=mode, order=order_mask, output_shape=output_shape) if mask is not None else None
        heat_aug = _warp_affine_arr_skimage(heat, tform, cval=cval, mode=mode, order=order_heat, output_shape=output_shape) if heat is not None else None
        return img_aug, msk_aug, heat_aug
    elif backend == "cv2":
        img_aug = np.squeeze(_warp_affine_arr_cv2(image, tform, cval=cval, mode=mode, order=order_image, output_shape=output_shape))
        msk_aug = np.squeeze(_warp_affine_arr_cv2(mask, tform, cval=cval, mode=mode, order=order_mask, output_shape=output_shape)) if mask is not None else None
        heat_aug = np.squeeze(_warp_affine_arr_cv2(mask, tform, cval=cval, mode=mode, order=order_heat, output_shape=output_shape)) if heat is not None else None
        return img_aug, msk_aug, heat_aug
    else:
        raise ValueError(f"Unknown backend {backend}")


## Helpers  
def build_shear_matrix_skimage(image_shape, shear_x_rad, shear_y_rad, shift_add=(0.5, 0.5)):
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
        matrix_to_topleft +
        matrix_shear_x +
        matrix_shear_y_rot +
        matrix_shear_y +
        matrix_shear_y_rot_inv +
        matrix_to_center
    )

    return matrix

def build_shear_matrix_cv2(image_shape, shear_x_rad, shear_y_rad, shift_add=(0.5, 0.5)):
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

    matrix = (
        matrix_to_topleft +
        matrix_shear_y_rot_inv +
        matrix_shear_y +
        matrix_shear_y_rot +
        matrix_shear_x +
        matrix_to_center
    )

    return matrix

def _compute_gaussian_blur_ksize(sigma):
    if sigma < 3.0:
        ksize = 3.3 * sigma  # 99% of weight
    elif sigma < 5.0:
        ksize = 2.9 * sigma  # 97% of weight
    else:
        ksize = 2.6 * sigma  # 95% of weight

    ksize = int(max(ksize, 5))
    return ksize

def _normalize_cv2_input_arr_(arr):
    flags = arr.flags
    if not flags["OWNDATA"]:
        arr = np.copy(arr)
        flags = arr.flags
    if not flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr

def _warp_affine_arr_skimage(arr, matrix, cval, mode, order, output_shape):
    image_warped = warp(
        arr,
        matrix.inverse,
        order=order,
        mode=mode,
        cval=cval,
        preserve_range=True,
        output_shape=output_shape,
    )
    # tf.warp will output float64, so just cast to float32
    return image_warped.astype(np.float32)

def _warp_affine_arr_cv2(arr, matrix, cval, mode, order, output_shape):

    if order != 0:
        assert arr.dtype.name != "int32", (
            "Affine only supports cv2-based transformations of int32 "
            "arrays when using order=0, but order was set to %d." % order)

    dsize = (int(np.round(output_shape[1])), int(np.round(output_shape[0])))

    mode_cv2 = _MAPPING_MODE_SCIPY_CV2.get(mode, mode)
    order_cv2 = _MAPPING_ORDER_SCIPY_CV2.get(order, order)

    nb_channels = arr.shape[-1] if arr.ndim == 3 else 1

    M = matrix.params[:2]

    if nb_channels <= 3:
        arr_norm = _normalize_cv2_input_arr_(arr)
        bval = cval if isinstance(cval, (int, float)) else tuple(cval[:nb_channels])
        image_warped = cv2.warpAffine(
            arr_norm,
            M,
            dsize=dsize,
            flags=order_cv2,
            borderMode=mode_cv2,
            borderValue=bval
        )
        if image_warped.ndim == 2:
            image_warped = image_warped[..., np.newaxis]
    else:
        arr_norm = _normalize_cv2_input_arr_(arr)
        chans = []
        for c in sm.xrange(nb_channels):
            bval = cval if isinstance(cval, (int, float)) else int(cval[c])
            single_ch = arr_norm[..., c]
            wch = cv2.warpAffine(
                single_ch,
                M,
                dsize=dsize,
                flags=order_cv2,
                borderMode=mode_cv2,
                borderValue=bval
            )
            if wch.ndim == 2:
                wch = wch[..., np.newaxis]
            chans.append(wch)
        image_warped = np.concatenate(chans, axis=-1)

    return image_warped.astype(np.float32)


## Elastic
def _draw_samples(alpha, sigma, nb_images):
    # Use np.random for all randomness
    def draw_param(param, size):
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

def _generate_shift_maps(shape, alpha, sigma):
    assert len(shape) == 2, ("Expected 2d shape, got %s." % (shape,))
    ksize = _compute_gaussian_blur_ksize(sigma)
    ksize = ksize + 1 if ksize % 2 == 0 else ksize
    padding = ksize
    h, w = shape[0:2]
    h_pad = h + 2*padding
    w_pad = w + 2*padding
    dxdy_unsmoothed = np.random.rand(2 * h_pad, w_pad) * 2 - 1
    dx_unsmoothed = dxdy_unsmoothed[0:h_pad, :]
    dy_unsmoothed = dxdy_unsmoothed[h_pad:, :]
    dx, _, _ = gaussian_blur_raw(dx_unsmoothed, sigma=sigma)
    dx *= alpha
    dy, _, _ = gaussian_blur_raw(dy_unsmoothed, sigma=sigma)
    dy *= alpha
    if padding > 0:
        dx = dx[padding:-padding, padding:-padding]
        dy = dy[padding:-padding, padding:-padding]
    return dx, dy

def _map_coordinates(image, dx, dy, order=1, cval=0, mode="constant"):
        # pylint: disable=invalid-name
        if image.size == 0:
            return np.copy(image)

        dx = dx.astype(np.float32) # from float 64
        dy = dy.astype(np.float32) # from float 64

        if order == 0 and image.dtype.name in ["uint64", "int64"]:
            raise Exception(
                "dtypes uint64 and int64 are only supported in "
                "ElasticTransformation for order=0, got order=%d with "
                "dtype=%s." % (order, image.dtype.name))

        shrt_max = 32767  # maximum of datatype `short`
        backend = "cv2"

        bad_dx_shape_cv2 = (dx.shape[0] >= shrt_max or dx.shape[1] >= shrt_max)
        bad_dy_shape_cv2 = (dy.shape[0] >= shrt_max or dy.shape[1] >= shrt_max)
        if  bad_dx_shape_cv2 or bad_dy_shape_cv2:
            backend = "scipy"

        assert image.ndim == 3, (
            "Expected 3-dimensional image, got %d dimensions." % (image.ndim,))
        result = np.copy(image)
        height, width = image.shape[0:2]
        if backend == "scipy":
            h, w = image.shape[0:2]
            y, x = np.meshgrid(
                np.arange(h).astype(np.float32),
                np.arange(w).astype(np.float32),
                indexing="ij")
            x_shifted = x + (-1) * dx
            y_shifted = y + (-1) * dy

            for c in sm.xrange(image.shape[2]):
                remapped_flat = interpolation.map_coordinates(
                    image[..., c],
                    (y_shifted.flatten(), x_shifted.flatten()),
                    order=order,
                    cval=cval,
                    mode=mode
                )
                remapped = remapped_flat.reshape((height, width))
                result[..., c] = remapped
        else:
            h, w, nb_channels = image.shape

            y, x = np.meshgrid(
                np.arange(h).astype(np.float32),
                np.arange(w).astype(np.float32),
                indexing="ij")
            x_shifted = x + (-1) * dx
            y_shifted = y + (-1) * dy

            if image.dtype.kind == "f":
                cval = float(cval)
            else:
                cval = int(cval)
            border_mode = _MAPPING_MODE_SCIPY_CV2[mode]
            interpolation = _MAPPING_ORDER_SCIPY_CV2[order]

            is_nearest_neighbour = (interpolation == cv2.INTER_NEAREST)
            if is_nearest_neighbour:
                map1, map2 = x_shifted, y_shifted
            else:
                map1, map2 = cv2.convertMaps(
                    x_shifted, y_shifted, cv2.CV_32FC1,
                    nninterpolation=is_nearest_neighbour)

            # remap only supports up to 4 channels
            if nb_channels <= 4:
                result = cv2.remap(
                    _normalize_cv2_input_arr_(image),
                    map1, map2, interpolation=interpolation,
                    borderMode=border_mode, borderValue=(cval, cval, cval))
                if image.ndim == 3 and result.ndim == 2:
                    result = result[..., np.newaxis]
            else:
                current_chan_idx = 0
                result = []
                while current_chan_idx < nb_channels:
                    channels = image[..., current_chan_idx:current_chan_idx+4]
                    result_c = cv2.remap(
                        _normalize_cv2_input_arr_(channels),
                        map1, map2, interpolation=interpolation,
                        borderMode=border_mode, borderValue=(cval, cval, cval))
                    if result_c.ndim == 2:
                        result_c = result_c[..., np.newaxis]
                    result.append(result_c)
                    current_chan_idx += 4
                result = np.concatenate(result, axis=2)

        return result

def elastic_raw(image: np.ndarray,
                mask: np.ndarray = None,
                heat: np.ndarray = None,
                alpha: float = 1000,
                sigma: float = 20,
                order_image: int = 3,
                order_mask: int = 0,
                order_heat: int = 1,
                cval: float = 0,
                mode: str = "constant",
                random_seed=None):

    if random_seed is not None:
        np.random.seed(random_seed)

    def warp_with_new_displacement(tensor, alpha, sigma, order, cval, mode):
        if tensor is None:
            return None
        dx, dy = _generate_shift_maps(
            shape=tensor.shape[:2],
            alpha=alpha,
            sigma=sigma
        )
        return _map_coordinates(tensor, dx, dy, order=order, cval=cval, mode=mode)

    alphas, sigmas = _draw_samples(alpha, sigma, nb_images=1)
    alpha_val = alphas[0]
    sigma_val = sigmas[0]

    img_warped3 = warp_with_new_displacement(image, alpha_val, sigma_val, order_image, cval, mode)
    mask_warped3 = warp_with_new_displacement(mask, alpha_val, sigma_val, order_mask, cval, mode)
    heat_warped3 = warp_with_new_displacement(heat, alpha_val, sigma_val, order_heat, cval, mode)

    return img_warped3, mask_warped3, heat_warped3