"""
Normalization utilities for image and mask data in deep learning workflows.

This module provides the Normalization class, which supports various normalization
strategies for images and masks, including percentile clipping, scaling, and
zero-mean unit-variance normalization. It is designed to work with both NumPy arrays
and PyTorch tensors, and integrates with BiaPy's DatasetFile for reproducible
normalization statistics.
"""
import torch
import copy
import numpy as np
from typing import (
    List,
    Tuple,
    Dict,
    Optional,
)
from numpy.typing import NDArray

torch_numpy_dtype_dict = {
    "bool": [torch.bool, bool],
    "uint8": [torch.uint8, np.uint8],
    "int8": [torch.int8, np.int8],
    "int16": [torch.int16, np.int16],
    "uint16": [torch.uint16, np.uint16],
    "int32": [torch.int32, np.int32],
    "int64": [torch.int64, np.int64],
    "float16": [torch.float16, np.float16],
    "float32": [torch.float32, np.float32],
    "float64": [torch.float64, np.float64],
    "complex64": [torch.complex64, np.complex64],
    "complex128": [torch.complex128, np.complex128],
}

from biapy.data.dataset import DatasetFile

def normalize_image(
    img: NDArray | torch.Tensor,
    norm_module: Dict,
    apply_norm: bool = True,
) -> Tuple[NDArray | torch.Tensor, Dict]:
    """
    Compute and set normalization statistics from a single image.

    Parameters
    ----------
    img (NDArray | torch.Tensor): Input image.
        E.g. ``(y, x, channels)`` in ``2D`` and ``(z, y, x, channels)`` in ``3D``.

    norm_module : dict
        Normalization module dict with the normalization parameters. Expected keys are:
            * ``type``, str: type of normalization to apply to the image. Expected values are:
                - ``div``: normalize the image by dividing it by a value. The value can be either 255 or 65535 depending on the maximum 
                  value of the image or the maximum and minimum values of the image if `div_using_max_and_scale` is `True`.
                - ``scale_range``: normalize the image by using the following operation: ``results = ((x - x_min)/(x_max - x_min)) * (out_max - out_min)``. 
                  In this case, the values used to do the normalization are computed from the data itself.
                - ``zero_mean_unit_variance``: apply zero-mean, unit-variance normalization to the image.
            * ``percentile_clip``, bool: whether to apply percentile clipping to the image before applying the normalization. If True, the values used for clipping
                will be computed from the data itself by using the percentiles specified in `per_lower_bound` and `per_upper_bound` or the values specified in 
                `lower_bound_val` and `upper_bound_val`.
            * ``out_dtype``, str: output dtype to convert the image to after applying the normalization. Expected values are: 'uint8', 'uint16' or 'float32'.
            * For ``zero_mean_unit_variance`` type, expected keys are:
                - ``mean``, list of float or single float: mean to use in the normalization. If a single float is provided, it will be used for all channels. 
                  If None, the mean of the data will be used.
                - ``std``, list of float or single float: standard deviation to use in the normalization. If a single float is provided, it will be used for 
                  all channels. If None, the std of the data will be used.
    """
    assert img.ndim >= 3, "Data should be at least 3D. E.g. (y, x, channels) in 2D and (z, y, x, channels) in 3D"

    assert "type" in norm_module, "'type' key should be in 'norm_module' dict"
    assert norm_module["type"] in ["div", "scale_range", "zero_mean_unit_variance"], (
        "Invalid normalization type. Expected values are: 'div', 'scale_range' and 'zero_mean_unit_variance'"
    )
    assert "percentile_clip" in norm_module, "'percentile_clip' key should be in 'norm_module' dict"
    assert isinstance(norm_module["percentile_clip"], bool), "'percentile_clip' should be a boolean value"
    assert "out_dtype" in norm_module, "'out_dtype' key should be in 'norm_module' dict"

    orig_dtype = str(img.dtype)
    new_norm_info = {
        "type": norm_module["type"],
        "percentile_clip": norm_module["percentile_clip"],
        "orig_dtype": orig_dtype,
        "out_dtype": norm_module["out_dtype"],
        "per_channel_info": {},
    }

    # If the normalization information for each channel is already provided, we will use it to avoid computing it again from the data
    if "per_channel_info" in norm_module:
        per_channel_info = norm_module["per_channel_info"]
        assert isinstance(per_channel_info, dict), "'per_channel_info' should be a dict with a key per channel containing the normalization information for each channel"
        assert len(per_channel_info) == img.shape[-1], (
            "The number of channels in 'per_channel_info' should be the same as the number of channels in the input data"
        )
            
    if norm_module["percentile_clip"]:
        per_lower_bound, per_upper_bound, lower_bound_val, upper_bound_val = None, None, None, None
        if "per_channel_info" not in norm_module:
            if "lower_bound_val" in norm_module:
                assert isinstance(norm_module["lower_bound_val"], list), "'lower_bound_val' should be a list of float/integer values"
                if norm_module["lower_bound_val"][0] != -1:
                    if len(norm_module["lower_bound_val"]) == 1:
                        lower_bound_val = float(norm_module["lower_bound_val"][0])
                    else:
                        assert len(norm_module["lower_bound_val"]) == img.shape[-1], "If more that one lower_bound_val value is provided, the number of "
                        "lower_bound_val values should be the same as the number of channels in the input image"
                        lower_bound_val = float(norm_module["lower_bound_val"][c])
                else:
                    assert "per_lower_bound" in norm_module, "If 'lower_bound_val' is not provided, 'per_lower_bound' should be provided"
                    per_lower_bound = norm_module["per_lower_bound"] 
            else:
                assert "per_lower_bound" in norm_module, "If 'lower_bound_val' is not provided, 'per_lower_bound' should be provided"
                per_lower_bound = norm_module["per_lower_bound"]

            if "upper_bound_val" in norm_module:
                assert isinstance(norm_module["upper_bound_val"], list), "'upper_bound_val' should be a list of float/integer values"
                if norm_module["upper_bound_val"][0] != -1:
                    if len(norm_module["upper_bound_val"]) == 1:
                        upper_bound_val = float(norm_module["upper_bound_val"][0])
                    else:
                        assert len(norm_module["upper_bound_val"]) == img.shape[-1], "If more that one upper_bound_val value is provided, the number of "
                        "upper_bound_val values should be the same as the number of channels in the input image"
                        upper_bound_val = float(norm_module["upper_bound_val"][c])
                else:
                    assert "per_upper_bound" in norm_module, "If 'upper_bound_val' is not provided, 'per_upper_bound' should be provided"
                    per_upper_bound = norm_module["per_upper_bound"]     
            else:
                per_upper_bound = norm_module["per_upper_bound"]                
        else:
            lower_bound_val = [per_channel_info[str(c)].get("lower_bound_val", None) for c in range(img.shape[-1])]
            upper_bound_val = [per_channel_info[str(c)].get("upper_bound_val", None) for c in range(img.shape[-1])]
       
    if norm_module["type"] in ["div", "scale_range"]:
        max_val_to_div, min_val_to_div = None, None
        if "per_channel_info" in norm_module:
            max_val_to_div = [per_channel_info[str(c)].get("max_val_to_div", None) for c in range(img.shape[-1])]
            min_val_to_div = [per_channel_info[str(c)].get("min_val_to_div", None) for c in range(img.shape[-1])]
    else: # 'zero_mean_unit_variance' type
        mean, std = None, None
        if "per_channel_info" in norm_module:
            mean = [per_channel_info[str(c)].get("mean", None) for c in range(img.shape[-1])]
            std = [per_channel_info[str(c)].get("std", None) for c in range(img.shape[-1])]
        else:
            if "mean" in norm_module:
                assert isinstance(norm_module["mean"], list), "'mean' should be a list of float values, just one to be applied to all the channels or one per channel"
                if norm_module["mean"][0] != -1:
                    if len(norm_module["mean"]) == 1:
                        mean = [float(norm_module["mean"][0])] * img.shape[-1]
                    else:
                        assert len(norm_module["mean"]) == img.shape[-1], "If more that one mean value is provided, the number of mean values should be the same as the "
                        "number of channels in the input image"
                        mean = norm_module["mean"]

            if "std" in norm_module:
                assert isinstance(norm_module["std"], list), "'std' should be a list of float values, just one to be applied to all the channels or one per channel"
                if norm_module["std"][0] != -1:
                    if len(norm_module["std"]) == 1:
                        std = [float(norm_module["std"][0])] * img.shape[-1]
                    else:
                        assert len(norm_module["std"]) == img.shape[-1], "If more that one std value is provided, the number of std values should be the same as the "
                        "number of channels in the input image"
                        std = norm_module["std"]

    # Changing dtype to floating tensor
    if isinstance(img, torch.Tensor):
        if not torch.is_floating_point(img):
            img = img.to(torch.float32)
    else:
        if not isinstance(img, np.floating):
            img = img.astype(np.float32)

    # Do the normalization channel by channel to be able to store the normalization information for each channel separately in the norm_info dict
    for c in range(img.shape[-1]):
        new_norm_info["per_channel_info"][f"{c}"] = {}
        if norm_module["percentile_clip"]:
            img[..., c], x_lwr, x_upr = percentile_clip(
                img[..., c], 
                per_lower_bound=per_lower_bound,
                per_upper_bound=per_upper_bound,
                lower_bound_val=lower_bound_val[c] if lower_bound_val is not None else None, 
                upper_bound_val=upper_bound_val[c] if upper_bound_val is not None else None,
                apply_norm=apply_norm
            )
            new_norm_info["per_channel_info"][f"{c}"]["lower_bound_val"] = x_lwr
            new_norm_info["per_channel_info"][f"{c}"]["upper_bound_val"] = x_upr
        
        if norm_module["type"] in ["div", "scale_range"]:
            img[..., c], max_val, min_val = norm_range01(
                img[..., c],
                div_using_max_and_scale=(norm_module["type"] == "scale_range"),
                max_val_to_div = max_val_to_div[c] if max_val_to_div is not None else None,
                min_val_to_div = min_val_to_div[c] if min_val_to_div is not None else None,
                apply_norm=apply_norm
            )
            new_norm_info["per_channel_info"][f"{c}"]["min_val_to_div"] = min_val
            new_norm_info["per_channel_info"][f"{c}"]["max_val_to_div"] = max_val

        elif norm_module["type"] == "zero_mean_unit_variance":
            img[..., c], used_mean, used_std = zero_mean_unit_variance_normalization(
                img[..., c], 
                mean=mean[c] if mean is not None else None,
                std=std[c] if std is not None else None,
                apply_norm=apply_norm
            )
            new_norm_info["per_channel_info"][f"{c}"]["mean"] = used_mean
            new_norm_info["per_channel_info"][f"{c}"]["std"] = used_std

    if isinstance(img, np.ndarray):
        img = img.astype(torch_numpy_dtype_dict[norm_module["out_dtype"]][1])
    else:
        img = img.to(torch_numpy_dtype_dict[norm_module["out_dtype"]][0])
        
    return img, new_norm_info

def normalize_mask(
    mask: NDArray | torch.Tensor,
    norm_module: dict,
    ignore_index: Optional[int] = None,
    n_classes: int = 1,
    instance_problem: bool = False,
    apply_norm: bool = True,
    is_training: bool = True,
) -> Tuple[NDArray | torch.Tensor, dict]:
    """
    Apply normalization to a mask.

    Parameters
    ----------
    mask : NDArray | torch.Tensor
        Mask to normalize. E.g. ``(y, x, channels)`` in ``2D`` and ``(z, y, x, channels)`` in ``3D``.

    norm_module : dict
        Normalization module dict with the normalization parameters. Expected keys are:
            * ``mask_norm``, str: type of normalization to apply to the mask. Expected values are:
                - ``as_mask``: apply normalization as if the mask were a mask. This means that the function will check if the channels 
                  of the mask are binary or not and if they need to be divided by 255 (e.g. if they are in 255 instead of 1). The function 
                  will also check if there are non-binary channels (e.g. distance transform channel) and set them as non-binary in the 
                  normalization information. This is essential to know how to handle the data in other parts of the pipeline such as 
                  during data augmentation.
                - ``as_image``: apply normalization as if the mask were an image. This means that the same normalization specified in `norm_module` 
                  for images will be applied to the mask.

    ignore_index : Optional[int]
        Value of the pixels to ignore when normalizing. If None, it will not be considered that there are pixels to ignore.

    n_classes : int
        Number of classes in the problem. It is used to check if the mask channels are binary or not. If there are more than 2 classes
        and it is an instance segmentation problem, it is expected that there is a channel per class plus one additional channel for the 
        instance ids, and the function will check that the channel with the instance ids is not binary.

    instance_problem : bool
        Whether it is an instance segmentation problem or not. It is used to check if the mask channels are binary or not. If there are more than 2 classes
        and it is an instance segmentation problem, it is expected that there is a channel per class plus one additional channel for the 
        instance ids, and the function will check that the channel with the instance ids is not binary.

    apply_norm : bool
        Whether to apply the normalization or just compute the normalization information. If False, the function will return the original mask and the computed 
        normalization information without applying the normalization.

    is_training : bool
        Whether the normalization is being applied in training or not. If False, the normalization will be applied as if the mask were an image, as we do not want
        to apply the normalization as if it were a mask in test/validation as it could be that the model is expecting the mask to be normalized as an image in 
        test/validation if the normalization information was computed from an image.

    Returns
    -------
    mask : 3D/4D Numpy array or torch.Tensor
        Y element normalized. E.g. ``(y, x, channels)`` in ``2D`` and ``(z, y, x, channels)`` in ``3D``.

    norm_info : dict
        Normalization information computed from the mask. It contains the original dtype of the mask and information about which channels are binary
        or not and if they need to be divided by 255 or not.
    """
    assert mask.ndim >= 3, "Data should be at least 3D. E.g. (y, x, channels) in 2D and (z, y, x, channels) in 3D"
    assert "mask_norm" in norm_module, "'mask_norm' key should be in 'norm_module' dict"

    _ignore_index = -1 if ignore_index is None else ignore_index
    if norm_module["mask_norm"] == "as_mask":
        orig_dtype = str(mask.dtype)
        norm_info = {
            "mask_norm": norm_module["mask_norm"],
            "orig_dtype": orig_dtype,
        }
        if "per_channel_info" in norm_module:
            norm_info["per_channel_info"] = norm_module["per_channel_info"]
            channels_to_analize = len([key for key, val in norm_module["per_channel_info"].items() if val["type"] != "classes"])
        else:
            norm_info["per_channel_info"] = {}
            if n_classes > 2 and instance_problem:
                channels_to_analize = mask.shape[-1] - 1
                norm_info["per_channel_info"][channels_to_analize] = {"type": "classes"}
                norm_info["per_channel_info"][channels_to_analize]["div"] = False
            else:
                channels_to_analize = mask.shape[-1]

            for j in range(channels_to_analize):
                if j not in norm_info["per_channel_info"]:
                    norm_info["per_channel_info"][j] = {"type": "bin"}
                    norm_info["per_channel_info"][j]["div"] = False

                if instance_problem:
                    if len(np.unique(mask[..., j])) > 2 and np.issubdtype(mask.dtype, np.floating):
                        norm_info["per_channel_info"][j]["type"] = "no_bin"
                else:  # In semantic seg, maybe the mask are in 255
                    if np.max(mask[..., j]) > max(n_classes,_ignore_index):
                        norm_info["per_channel_info"][j]["div"] = True
            
        if apply_norm:
            for j in range(channels_to_analize):
                if norm_info["per_channel_info"][j]["div"]:
                    mask[..., j] = mask[..., j] / 255

    # Continue normalization as if it were an image
    # Normalization in test should not be applied to mask/ground truth data
    elif norm_module["mask_norm"] == "as_image" and is_training:
        mask, norm_info = normalize_image(img=mask, norm_module=norm_module, apply_norm=apply_norm) 
        norm_info["mask_norm"] = norm_module["mask_norm"]
    else:
        norm_info = norm_module.copy()

    return mask, norm_info

def update_mask_norm_info(old_mask_norm_info: Dict, new_mask_norm_info: Dict) -> Dict:
    """
    Update the mask normalization information by replacing the values of the old mask normalization information with the 
    ones of the new mask normalization information when they are more restrictive. For example:

        - orig_dtype: float will be set instead of int, as it is more restrictive.
        - For each channel:
            - type: "no_bin" will be set instead of "bin" or "classes", and "classes" will be set instead of "bin", as they are more restrictive.
            - div: True will be set instead of False, as it is more restrictive.
    
    Parameters
    ----------
    old_mask_norm_info: dict
        Old mask normalization information to update.

    new_mask_norm_info: dict
        New mask normalization information to update the old one with.
    
    Returns
    -------
    new_norm: dict
        Updated mask normalization information.
    """
    assert "orig_dtype" in old_mask_norm_info and "per_channel_info" in old_mask_norm_info, (
        "Old mask normalization information should contain 'orig_dtype' and 'per_channel_info' keys"
    )
    new_norm = copy.deepcopy(old_mask_norm_info)

    # Float more restrictive than int
    if (
        "orig_dtype" in new_mask_norm_info 
        and new_mask_norm_info["orig_dtype"] != old_mask_norm_info.get("orig_dtype", None) 
        and "float" in new_mask_norm_info["orig_dtype"]
    ):
        new_norm["orig_dtype"] = new_mask_norm_info["orig_dtype"]

    for channel, channel_info in new_mask_norm_info["per_channel_info"].items():
        old_mask_type = old_mask_norm_info["per_channel_info"].get(channel, {}).get("type", None)

        # Set if no type was set before
        if old_mask_type is None:
            new_norm["per_channel_info"][channel] = copy.deepcopy(channel_info)

        # Set the most restrictive type: "no_bin" > "classes" > "bin"
        if channel_info["type"] == "no_bin":
            new_norm["per_channel_info"][channel]["type"] = "no_bin"
        elif channel_info["type"] == "classes":
            new_norm["per_channel_info"][channel]["type"] = "classes"
        elif channel_info["type"] == "bin":
            # Set "div" to True if the new channel is binary but the old one is not, as it is more restrictive
            if channel_info["div"]:
                new_norm["per_channel_info"][channel]["div"] = True

    return new_norm

def percentile_clip(
    data: NDArray | torch.Tensor,
    per_lower_bound: Optional[float] = None,
    per_upper_bound: Optional[float] = None,
    lower_bound_val: Optional[float] = None,
    upper_bound_val: Optional[float] = None,
    apply_norm: bool = True,
) -> Tuple[NDArray | torch.Tensor, float, float]:
    """
    Percentile clipping.

    Parameters
    ----------
    data (NDArray | torch.Tensor): Input data.
        Data to normalize. E.g. ``(y, x)`` in ``2D`` and ``(z, y, x)`` in ``3D``.

    per_lower_bound : Optional[float]
        Lower bound percentile to use for clipping. Should be between 0 and 100. If
        None, `lower_bound_val` should be provided.
    
    per_upper_bound : Optional[float]
        Upper bound percentile to use for clipping. Should be between 0 and 100. If
        None, `upper_bound_val` should be provided.

    lower_bound_val : Optional[float]
        Lower bound value to use for clipping. If None, `per_lower_bound` should be provided.

    upper_bound_val : Optional[float]
        Upper bound value to use for clipping. If None, `per_upper_bound` should be provided.

    apply_norm : bool
        Whether to apply the percentile clipping or just compute the lower and upper bound values. If False, 
        the function will return the original data and the computed lower and upper bound values without 
        applying the clipping.

    Returns
    -------
    data : 3D/4D Numpy array or torch.Tensor
        Clipped data if `apply_norm` is True. E.g. ``(y, x)`` in ``2D`` and ``(z, y, x)`` in ``3D``.

    x_lwrs : float
        Lower bound used for clipping.

    x_uprs : float
        Upper bound used for clipping.
    """
    if per_lower_bound is None or per_lower_bound == -1:
        assert lower_bound_val is not None, "If 'per_lower_bound' is not provided, 'lower_bound_val' should be provided"
        assert isinstance(lower_bound_val, (int, float)), "'lower_bound_val' should be a single float value"
        x_lwr = lower_bound_val
    else:
        assert per_lower_bound > 0, "Value in 'per_lower_bound' should be less than 100"
        if isinstance(data, np.ndarray):
            x_lwr = float(np.percentile(data, per_lower_bound))
        else:
            x_lwr = float(torch_percentile(data, per_lower_bound))

    if per_upper_bound is None or per_upper_bound == -1:
        assert upper_bound_val is not None, "If 'per_upper_bound' is not provided, 'upper_bound_val' should be provided"
        assert isinstance(upper_bound_val, (int, float)), "'upper_bound_val' should be a single float value"
        x_upr = upper_bound_val
    else:
        assert per_upper_bound < 100, "Value in 'per_upper_bound' should be less than 100"
        if isinstance(data, np.ndarray):
            x_upr = float(np.percentile(data, per_upper_bound))
        else:
            x_upr = float(torch_percentile(data, per_upper_bound))

    if apply_norm:
        if isinstance(data, torch.Tensor):
            data = torch.clamp(data, x_lwr, x_upr)
        else:
            data = np.clip(data, x_lwr, x_upr)
        
    return data, x_lwr, x_upr
    
def torch_percentile(data: torch.Tensor, q: float) -> int | float:
    """
    Return the ``q``-th percentile of the flattened input tensor's data.

    Copied from: https://gist.github.com/sailfish009/28b54c8aa6398148a6358b8f03c0b611

    Parameters
    ----------
    data (torch.Tensor): Input tensor.
        Data to normalize. E.g. ``(y, x, channels)`` in ``2D`` and ``(z, y, x, channels)`` in ``3D``.

    q : float, optional
        Percentile to compute, which must be between 0 and 100 inclusive.

    Returns
    -------
    int | float: Percentile value.
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    k = 1 + round(0.01 * float(q) * (data.numel() - 1))
    result = data.view(-1).kthvalue(k).values.item()
    return result

def norm_range01(
    data: NDArray | torch.Tensor,
    div_using_max_and_scale: bool,
    max_val_to_div: int | float | None,
    min_val_to_div: int | float | None,
    apply_norm: bool = True,
    eps: float = 1e-6,
) -> Tuple[NDArray | torch.Tensor, float, float]:
    """
    Normalize given data by dividing it by a value.

    Parameters
    ----------
    data : 3D/4D Numpy array or torch.Tensor
        Data to normalize. E.g. ``(y, x)`` in ``2D`` and ``(z, y, x)`` in ``3D``.

    div_using_max_and_scale : bool
        Whether to normalize the data by doing a division (when it is ``False``) or by using the following operation
        (when it is ``True``): ``results = ((x - x_min)/(x_max - x_min)) * (out_max - out_min)``.

    max_val_to_div : int or float or None
        Maximum value to use to divide the data. If not provided it will be computed from the data itself. 
        It is 255 or 65535 if `div_using_max_and_scale` is `False` and the maximum value of the data if `div_using_max_and_scale` is `True`.

    min_val_to_div : int or float or None
        Minimum value to use to divide the data. If not provided it will be computed from the data itself. 
        It is 0 if `div_using_max_and_scale` is `False` and the minimum value of the data if `div_using_max_and_scale` is `True`.

    apply_norm : bool
        Whether to apply the normalization or just compute the values to do it. If False, the function will return 
        the original data and the computed values without applying the normalization.
        
    eps : float
        Small value to add to the denominator to prevent division by zero when normalizing by using the
        maximum and minimum values of the data.

    Returns
    -------
    data : 3D/4D Numpy array or torch.Tensor
        Normalized data if `apply_norm` is True. E.g. ``(y, x)`` in ``2D`` and ``(z, y, x)`` in ``3D``.

    max_val_to_div : float
        Maximum value used to divide the data. It is 255 or 65535 if ``div_using_max_and_scale`` is ``False`` and the maximum 
        value of the data if ``div_using_max_and_scale`` is ``True``.

    min_val_to_div : float
        Minimum value used to divide the data. It is 0 if ``div_using_max_and_scale`` is ``False`` and the minimum value of 
        the data if ``div_using_max_and_scale`` is ``True``.
    """
    if max_val_to_div is not None and min_val_to_div is None:
        raise ValueError("If 'max_val_to_div' is provided, 'min_val_to_div' should also be provided")
    if max_val_to_div is None and min_val_to_div is not None:
        raise ValueError("If 'min_val_to_div' is provided, 'max_val_to_div' should also be provided")

    # If the data is already in the range [0, 1], we will not apply the normalization and we will return the original data 
    # and the values used to do the normalization as 1 and 0 respectively to be able to undo the normalization correctly if needed
    if data.min() == 0 and data.max() == 1:
        return data, 1.0, 0.0

    # Changing dtype to floating tensor
    if isinstance(data, torch.Tensor):
        if not torch.is_floating_point(data):
            data = data.to(torch.float32)
    else:
        if not isinstance(data, np.floating):
            data = data.astype(np.float32)

    if max_val_to_div is not None and min_val_to_div is not None:
        assert isinstance(max_val_to_div, (int, float)), "'max_val_to_div' should be a single float value"
        assert isinstance(min_val_to_div, (int, float)), "'min_val_to_div' should be a single float value"
        _max_val_to_div = float(max_val_to_div)
        _min_val_to_div = float(min_val_to_div)
    else:
        if div_using_max_and_scale: 
            _max_val_to_div = float(data.max())
            _min_val_to_div = float(data.min())
        else:
            _max_val_to_div = 65535 if data.max() > 255 else 255
            _min_val_to_div = 0

    if apply_norm:
        data = (data - _min_val_to_div) / (  # type: ignore
            max(_max_val_to_div - _min_val_to_div, eps)
        )

    return data, _max_val_to_div, _min_val_to_div

def zero_mean_unit_variance_normalization(
    data: NDArray | torch.Tensor,
    mean: Optional[float] = None,
    std: Optional[float] = None,
    apply_norm: bool = True,
    eps: float = 1e-6,
) -> Tuple[NDArray | torch.Tensor, float, float]:
    """
    Apply zero-mean, unit-variance normalization.

    Parameters
    ----------
    data : (NDArray | torch.Tensor)
        Data to normalize. E.g. ``(y, x)`` in ``2D`` and ``(z, y, x)`` in ``3D``.

    mean : Optional[float]
        Mean to use in the normalization. If None, the mean of the data will be used.

    std : Optional[float]
        Standard deviation to use in the normalization. If None, the std of the data will be used.

    apply_norm : bool
        Whether to apply the normalization or just compute the mean and std values. If False, the 
        function will return the original data and the computed mean and std values without applying 
        the normalization.

    eps : float
        Small value to add to the denominator to prevent division by zero when normalizing.

    Returns
    -------
    data : 3D/4D Numpy array or torch.Tensor
        Normalized data if `apply_norm` is True. E.g. ``(y, x)`` in ``2D`` and ``(z, y, x)`` in ``3D``.

    mean : float
        Mean used in the normalization.

    std : float
        Standard deviation used in the normalization.
    """
    assert data.ndim >= 2, "Data should be at least 2D. E.g. (y, x) in 2D and (z, y, x) in 3D"

    if isinstance(data, torch.Tensor):
        if not torch.is_floating_point(data):  # type: ignore
            data = data.to(torch.float32)
    else:
        if not isinstance(data, np.floating):
            data = data.astype(np.float32)

    mean = data.mean() if mean is None else mean
    std = data.std() if std is None else std

    if apply_norm:
        data = (data - mean) / (max(std, eps))

    return data, mean, std

def undo_image_norm(
    data: NDArray | torch.Tensor,
    norm_info: Dict,
) -> NDArray | torch.Tensor:
    """
    Unnormalize given input data following the normalization steps done before for normalizing it.

    Parameters
    ----------
    data : 3D/4D Numpy array
        Data to unnormalize. E.g. ``(y, x, channels)`` in ``2D`` and ``(z, y, x, channels)`` in ``3D``.

    norm_info : dict
        Normalization information to undo the normalization. 

    Returns
    -------
    data : 3D/4D Numpy array
        Unnormalized data. E.g. ``(y, x, channels)`` in ``2D`` and ``(z, y, x, channels)`` in ``3D``.
    """
    assert "type" in norm_info, "'type' key should be in 'norm_info' dict. Ensure you input the same normalization dict "
    "used to normalize the data previously"
    assert "per_channel_info" in norm_info, "'per_channel_info' key should be in 'norm_info' dict. Ensure you input the "
    "same normalization dict used to normalize the data previously"

    if norm_info["type"] in ["div", "scale_range"]:
        data = undo_norm_range01(data, norm_info)
    else:  # zero_mean_unit_variance
        data = undo_zero_mean_unit_variance_normalization(data, norm_info)

        if "float" not in str(norm_info["orig_dtype"]):
            if isinstance(data, np.ndarray):
                data = np.round(data)
            else:  # torch.Tensor
                data = torch.round(data)
            mindata = data.min()
            data = data + abs(mindata)  # type: ignore

    if isinstance(data, np.ndarray):
        data = data.astype(torch_numpy_dtype_dict[norm_info["orig_dtype"]][1])
    else:
        data = data.to(torch_numpy_dtype_dict[norm_info["orig_dtype"]][0])

    return data

def undo_norm_range01(
    data: NDArray | torch.Tensor,
    norm_info: Dict,
) -> NDArray | torch.Tensor:
    """
    Undo normalization by multiplaying a factor and optionally summing a minimum value. Opposite function of ``__norm_range01``.

    Parameters
    ----------
    data : 3D/4D Numpy array
        Data to unnormalize. E.g. ``(y, x, channels)`` in ``2D`` and ``(z, y, x, channels)`` in ``3D``.

    norm_info : dict
        Information about the normalization. Expected keys are:
            * ``"max_val_to_div"``, int/float: maximum value used to divide the data in the normalization.
            * ``"min_val_to_div"``, int/float: minimum value used to divide the data in the normalization.

    Returns
    -------
    data : 3D/4D Numpy array
        Unnormalized data. E.g. ``(y, x, channels)`` in ``2D`` and ``(z, y, x, channels)`` in ``3D``.
    """
    assert "per_channel_info" in norm_info, "'per_channel_info' key should be in 'norm_info' dict. Ensure you input the same normalization dict used to normalize the data previously"
    assert len(norm_info["per_channel_info"]) == data.shape[-1], "The number of channels in the input data should be the same as the number of channels in 'per_channel_info' in 'norm_info'. Ensure you input the same normalization dict used to normalize the data previously"
    for c in range(data.shape[-1]):
        assert "max_val_to_div" in norm_info["per_channel_info"][str(c)], f"'max_val_to_div' key should be in 'per_channel_info' for channel {c} in 'norm_info' dict. Ensure you input the same normalization dict used to normalize the data previously"
        assert "min_val_to_div" in norm_info["per_channel_info"][str(c)], f"'min_val_to_div' key should be in 'per_channel_info' for channel {c} in 'norm_info' dict. Ensure you input the same normalization dict used to normalize the data previously"

    # Prevent values go outside expected range
    if isinstance(data, np.ndarray):
        data = np.clip(data, 0, 1)
    else:
        data = torch.clamp(data, 0, 1)

    max_val_to_div = [norm_info["per_channel_info"][str(c)].get("max_val_to_div", None) for c in range(data.shape[-1])]
    min_val_to_div = [norm_info["per_channel_info"][str(c)].get("min_val_to_div", None) for c in range(data.shape[-1])]

    data = (data * max_val_to_div) + min_val_to_div

    return data

def undo_zero_mean_unit_variance_normalization(
    data: NDArray | torch.Tensor,
    norm_info: Dict,
) -> NDArray | torch.Tensor:
    """
    Unnormalization of input data by multiplying by the std and adding the mean.
    
    Opposite function of ``zero_mean_unit_variance_normalization``.

    Parameters
    ----------
    data : 3D/4D Numpy array
        Image to unnormalize. E.g. ``(y, x, channels)`` in ``2D`` and ``(z, y, x, channels)`` in ``3D``.

    norm_info : dict
        Information about the normalization. Expected keys are:
            * ``"mean"``, int/float: mean used in normalization.
            * ``"std"``, int/float: std used in normalization.

    Returns
    -------
    data : 3D/4D Numpy array
        Unnormalized data. E.g. ``(y, x, channels)`` in ``2D`` and ``(z, y, x, channels)`` in ``3D``.
    """
    assert "per_channel_info" in norm_info, "'per_channel_info' key should be in 'norm_info' dict. Ensure you input the same normalization dict used to normalize the data previously"
    assert len(norm_info["per_channel_info"]) == data.shape[-1], "The number of channels in the input data should be the same as the number of channels in 'per_channel_info' in 'norm_info'. Ensure you input the same normalization dict used to normalize the data previously"
    for c in range(data.shape[-1]):
        assert "mean" in norm_info["per_channel_info"][str(c)], f"'mean' key should be in 'per_channel_info' for channel {c} in 'norm_info' dict. Ensure you input the same normalization dict used to normalize the data previously"
        assert "std" in norm_info["per_channel_info"][str(c)], f"'std' key should be in 'per_channel_info' for channel {c} in 'norm_info' dict. Ensure you input the same normalization dict used to normalize the data previously"

    mean = [norm_info["per_channel_info"][str(c)].get("mean", None) for c in range(data.shape[-1])]
    std = [norm_info["per_channel_info"][str(c)].get("std", None) for c in range(data.shape[-1])]

    return (data * std) + mean
