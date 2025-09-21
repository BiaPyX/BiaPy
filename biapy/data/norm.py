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


class Normalization:
    """
    Handles normalization of image and mask data for deep learning workflows.

    This class supports several normalization strategies:
    - "div": Divides values to scale them to [0, 1] range, using 255 or 65535 as divisor.
    - "scale_range": Scales values to [0, 1] using the min and max of the data.
    - "zero_mean_unit_variance": Subtracts mean and divides by standard deviation.

    It also manages mask normalization, percentile clipping, and stores normalization statistics
    for reproducibility.

    Attributes:
        type (str): Normalization type ("div", "scale_range", "zero_mean_unit_variance").
        measure_by (str): Whether to measure stats by "image" or "patch".
        mask_norm (str): Mask normalization mode ("as_mask", "as_image", "none").
        out_dtype (str): Output data type after normalization.
        do_percentile_clipping (bool): Whether to apply percentile clipping.
        channels_to_analize (Optional[int]): Number of mask channels to analyze.
        channel_info (Optional[Dict]): Info about each mask channel.
        train_normalization (bool): If True, normalization is applied during training.
        eps (float): Small value to avoid division by zero.
        ... (other attributes for normalization statistics)
    """

    def __init__(
        self,
        type: str,
        measure_by: str,
        mask_norm: str,
        out_dtype: str,
        percentile_clip: bool,
        per_lower_bound: float = 2,
        per_upper_bound: float = 99.8,
        lower_bound_val: float = -1,
        upper_bound_val: float = -1,
        mean: float = -1,
        std: float = -1,
        channels_to_analize: Optional[int] = None,
        channel_info: Optional[Dict] = None,
        train_normalization: bool = True,
        eps: float = 1e-6,
    ):
        """
        Initialize the Normalization object with the specified configuration.

        Args:
            type (str): Normalization type.
            measure_by (str): "image" or "patch".
            mask_norm (str): Mask normalization mode.
            out_dtype (str): Output data type.
            percentile_clip (bool): Enable percentile clipping.
            per_lower_bound (float): Lower percentile for clipping.
            per_upper_bound (float): Upper percentile for clipping.
            lower_bound_val (float): Explicit lower bound value.
            upper_bound_val (float): Explicit upper bound value.
            mean (float): Fixed mean for normalization.
            std (float): Fixed std for normalization.
            channels_to_analize (Optional[int]): Number of mask channels to analyze.
            channel_info (Optional[Dict]): Info about each mask channel.
            train_normalization (bool): Apply normalization during training.
            eps (float): Epsilon for numerical stability.
        """
        assert type in ["div", "scale_range", "zero_mean_unit_variance"]
        assert measure_by in ["image", "patch"]
        assert mask_norm in ["as_mask", "as_image", "none"]
        assert out_dtype in list(torch_numpy_dtype_dict.keys())

        self.type = type
        self.measure_by = measure_by
        self.mask_norm = mask_norm
        self.out_dtype = out_dtype
        self.do_percentile_clipping = percentile_clip
        self.channels_to_analize = channels_to_analize
        self.channel_info = channel_info
        self.train_normalization = train_normalization
        self.eps = eps

        if percentile_clip:
            self.per_lower_bound = per_lower_bound
            self.per_upper_bound = per_upper_bound
            self.lower_bound_val = lower_bound_val if lower_bound_val != -1 else None
            self.upper_bound_val = upper_bound_val if upper_bound_val != -1 else None
            if self.lower_bound_val:
                print("Percentile clipping [{},{}]".format(self.lower_bound_val, self.lower_bound_val))
            else:
                print("Percentile clipping [{},{}]".format(self.per_lower_bound, self.per_upper_bound))

        if type == "scale_range":
            self.scale_range_min_val = None
            self.scale_range_max_val = None
        elif type == "zero_mean_unit_variance":
            self.fixed_mean = mean if mean != -1 else None
            self.fixed_std = std if std != -1 else None
            self.mean, self.std = None, None
            print("Normalization: using mean {} and std: {}".format(self.fixed_mean, self.fixed_std))

        self.last_X_norm = None
        self.last_Y_norm = None

    def set_stats_from_image(self, image: NDArray | torch.Tensor):
        """
        Compute and set normalization statistics from a single image.

        Args:
            image (NDArray | torch.Tensor): Input image.
        """
        if self.measure_by == "image":
            if self.do_percentile_clipping:
                if isinstance(image, np.ndarray):
                    self.lower_bound_val = float(np.percentile(image, self.per_lower_bound))
                    self.upper_bound_val = float(np.percentile(image, self.per_upper_bound))
                else:
                    self.lower_bound_val = float(self.torch_percentile(image, self.per_lower_bound))
                    self.upper_bound_val = float(self.torch_percentile(image, self.per_upper_bound))

                image, _, _ = self.__percentile_clip(image)

            if self.type == "scale_range":
                self.scale_range_min_val = [image.min()]
                self.scale_range_max_val = [image.max()]
            elif self.type == "zero_mean_unit_variance":
                self.mean = float(image.mean()) if self.fixed_mean is None else self.fixed_mean
                self.std = float(image.std()) if self.fixed_std is None else self.fixed_std

    def set_stats_from_mask(self, mask: NDArray | torch.Tensor, n_classes: int = 1, ignore_index: Optional[int] = None, instance_problem: bool = False):
        """
        Compute and set normalization statistics from a mask.
        
        The mask analysis is done by channel, as some of them may be normalized,
        such as distance channels, while others no e.g. foreground binary channel.

        Args:
            mask (NDArray | torch.Tensor): Input mask.
            n_classes (int): Number of classes.
            ignore_index (Optional[int]): Index to ignore.
            instance_problem (bool): If True, treat as instance segmentation.
        """
        _ignore_index = -1 if ignore_index is None else ignore_index
        if self.mask_norm == "as_mask":
            if not self.channel_info:
                self.channels_to_analize = -1
                self.channel_info = {}
                if n_classes > 2 and instance_problem:
                    self.channels_to_analize = mask.shape[-1] - 1
                    self.channel_info[self.channels_to_analize] = {"type": "classes"}
                    self.channel_info[self.channels_to_analize]["div"] = False
                else:
                    self.channels_to_analize = mask.shape[-1]

            assert self.channels_to_analize
            for j in range(self.channels_to_analize):
                if j not in self.channel_info:
                    self.channel_info[j] = {"type": "bin"}
                    self.channel_info[j]["div"] = False

                if instance_problem:
                    if len(np.unique(mask[..., j])) > 2:
                        self.channel_info[j]["type"] = "no_bin"
                        self.no_bin_channel_found = True
                        self.channel_info[j]["div"] = False
                else:  # In semantic seg, maybe the mask are in 255
                    if np.max(mask[..., j]) > max(n_classes,_ignore_index):
                        self.channel_info[j]["div"] = True
        elif self.mask_norm == "as_image":
            self.set_stats_from_image(mask)

    def set_stats_from_DatasetFile(self, dataset_file: DatasetFile):
        """
        Load normalization statistics from a DatasetFile object.

        Args:
            dataset_file (DatasetFile): DatasetFile containing normalization stats.
        """
        try:
            if self.measure_by == "image":
                if self.do_percentile_clipping:
                    self.lower_bound_val = dataset_file.lower_bound_val
                    self.upper_bound_val = dataset_file.upper_bound_val
                if self.type == "scale_range":
                    self.scale_range_min_val = dataset_file.scale_range_min_val
                    self.scale_range_max_val = dataset_file.scale_range_max_val
                elif self.type == "zero_mean_unit_variance":
                    self.mean = dataset_file.mean if self.fixed_mean is None else self.fixed_mean
                    self.std = dataset_file.std if self.fixed_std is None else self.fixed_std
        except Exception as e:
            print(e)
            raise ValueError("Seems that the DatasetFile input was not created using the same normalization steps.")

    def set_DatasetFile_from_stats(self, dataset_file: DatasetFile) -> DatasetFile:
        """
        Save current normalization statistics into a DatasetFile object.

        Args:
            dataset_file (DatasetFile): DatasetFile to update.

        Returns:
            DatasetFile: Updated DatasetFile with normalization stats.
        """
        if self.measure_by == "image":
            if self.do_percentile_clipping:
                dataset_file.lower_bound_val = self.lower_bound_val
                dataset_file.upper_bound_val = self.upper_bound_val
            if self.type == "scale_range":
                dataset_file.scale_range_min_val = self.scale_range_min_val
                dataset_file.scale_range_max_val = self.scale_range_max_val
            elif self.type == "zero_mean_unit_variance":
                dataset_file.mean = self.mean if self.fixed_mean is None else self.fixed_mean
                dataset_file.std = self.std if self.fixed_std is None else self.fixed_std

        return dataset_file

    def apply_image_norm(
        self,
        img: NDArray | torch.Tensor,
    ) -> Tuple[NDArray | torch.Tensor, Dict]:
        """
        Apply normalization to an image.

        Parameters
        ----------
        img (NDArray | torch.Tensor): Input image.
            X element, for instance, an image. E.g. ``(y, x, channels)`` in ``2D`` and ``(z, y, x, channels)`` in ``3D``.

        Returns
        -------
        Tuple[NDArray | torch.Tensor, Dict]: Normalized image and normalization info.
            img : 3D/4D Numpy array or torch.Tensor
                X element normalized. E.g. ``(y, x, channels)`` in ``2D`` and ``(z, y, x, channels)`` in ``3D``.

            last_X_norm_extra_info : dict
                Values used to normalize the data.
        """
        self.last_X_norm_extra_info = {}
        # Percentile clipping
        if self.do_percentile_clipping:
            img, _, _ = self.__percentile_clip(img)

        if self.type == "div":
            img, xnorm = self.__norm_range01(img, div_using_max_and_scale=False)
        elif self.type == "scale_range":
            if self.measure_by == "image" and self.train_normalization:
                assert self.scale_range_min_val is not None or self.scale_range_max_val is not None, (
                    "'scale_range_max_val' and 'scale_range_min_val' should not be None. Please call "
                    "'Normalization.stats_from_image' or 'Normalization.set_stats_from_DatasetFile' before"
                )
            img, xnorm = self.__norm_range01(img, div_using_max_and_scale=True)
        elif self.type == "zero_mean_unit_variance":
            if self.measure_by == "image" and self.train_normalization:
                assert (
                    self.mean is not None or self.std is not None
                ), "'mean' and 'std' should not be None. Please call 'Normalization.stats_from_image' or 'Normalization.set_stats_from_DatasetFile' before"
            img, xnorm = self.__zero_mean_unit_variance_normalization(img)

        if isinstance(img, np.ndarray):
            img = img.astype(torch_numpy_dtype_dict[self.out_dtype][1])
        else:
            img = img.to(torch_numpy_dtype_dict[self.out_dtype][0])

        self.last_X_norm_extra_info.update(xnorm)
        return img, self.last_X_norm_extra_info

    def apply_mask_norm(
        self,
        mask: NDArray | torch.Tensor,
    ) -> Tuple[NDArray | torch.Tensor, dict]:
        """
        Apply normalization to a mask.

        Parameters
        ----------
        mask (NDArray | torch.Tensor): Input mask.
            Y element, for instance, an image's mask. E.g. ``(y, x, channels)`` in ``2D`` and ``(z, y, x, channels)`` in
            ``3D``.

        Returns
        -------
        Tuple[NDArray | torch.Tensor, dict]: Normalized mask and normalization info.
            mask : 3D/4D Numpy array or torch.Tensor
                Y element normalized. E.g. ``(y, x, channels)`` in ``2D`` and ``(z, y, x, channels)`` in ``3D``.

            last_Y_norm_extra_info : dict
                Values used to normalize the data.
        """
        self.last_Y_norm_extra_info = {}
        if self.mask_norm == "as_mask":
            assert self.channels_to_analize is not None, "Please set first 'channels_to_analize' attribute"
            assert self.channel_info is not None, "Please set first 'channels_to_analize' attribute"
            for j in range(self.channels_to_analize):
                if self.channel_info[j]["div"]:
                    mask[..., j] = mask[..., j] / 255  # type: ignore

        # Normalization in test should not be applied to mask/ground truth data
        elif self.mask_norm == "as_image" and self.train_normalization:
            mask, self.last_Y_norm_extra_info = self.apply_image_norm(mask)

        return mask, self.last_Y_norm_extra_info

    def get_channel_info(self, channel_pos: int) -> Dict:
        """
        Retrieve information about a specific mask channel.

        Args:
            channel_pos (int): Channel index.

        Returns:
            Dict: Channel information.
        """
        assert self.channel_info
        return self.channel_info[channel_pos]

    def __percentile_clip(
        self,
        data: NDArray | torch.Tensor,
    ) -> Tuple[NDArray | torch.Tensor, float, float]:
        """
        Percentile clipping.

        Parameters
        ----------
        data (NDArray | torch.Tensor): Input data.
            Data to normalize. E.g. ``(y, x, channels)`` in ``2D`` and ``(z, y, x, channels)`` in ``3D``.

        Returns
        -------
        Tuple[NDArray | torch.Tensor, float, float]: Clipped data, lower bound, upper bound E.g. ``(y, x, channels)`` in ``2D`` and ``(z, y, x, channels)`` in ``3D``.
        """
        if isinstance(data, torch.Tensor):
            x_lwr = (
                float(self.torch_percentile(data, self.per_lower_bound))
                if self.lower_bound_val is None
                else self.lower_bound_val
            )
            x_upr = (
                float(self.torch_percentile(data, self.per_upper_bound))
                if self.upper_bound_val is None
                else self.upper_bound_val
            )
        else:
            x_lwr = (
                float(np.percentile(data, self.per_lower_bound))
                if self.lower_bound_val is None
                else self.lower_bound_val
            )
            x_upr = (
                float(np.percentile(data, self.per_upper_bound))
                if self.upper_bound_val is None
                else self.upper_bound_val
            )
        if "float" not in str(data.dtype):
            x_lwr = int(x_lwr)
            x_upr = int(x_upr)
        if isinstance(data, torch.Tensor):
            data = torch.clamp(data, x_lwr, x_upr)
        else:
            data = np.clip(data, x_lwr, x_upr)

        return data, x_lwr, x_upr

    def torch_percentile(self, data: torch.Tensor, q: float) -> int | float:
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

    def __norm_range01(
        self,
        data: NDArray | torch.Tensor,
        div_using_max_and_scale: bool,
        per_channel: bool = False,
    ) -> Tuple[NDArray | torch.Tensor, Dict]:
        """
        Normalize given data by dividing it by a value.

        Parameters
        ----------
        data : 3D/4D Numpy array or torch.Tensor
            Data to normalize. E.g. ``(y, x, channels)`` in ``2D`` and ``(z, y, x, channels)`` in ``3D``.

        div_using_max_and_scale : bool
            Whether to normalize the data by doing a division (when it is ``False``) or by using the following operation
            (when it is ``True``): ``results = ((x - x_min)/(x_max - x_min)) * (out_max - out_min)``.

        per_channel : bool
            Whether to normalize the data per channel. It is used only when ``div_using_max_and_scale`` is ``True``.

        Returns
        -------
        Tuple[NDArray | torch.Tensor, Dict]: Normalized data and normalization info.
            data : 3D/4D Numpy array or torch.Tensor
                Normalized data. E.g. ``(y, x, channels)`` in ``2D`` and ``(z, y, x, channels)`` in ``3D``.

            norm_steps : dict
                Contains information about the normalization steps applied. It is a dict containing the following keys:
                * ``"orig_dtype"``, str: original dtype of the sample.
                * ``"div_value"``, int/float (optional): number used to divide during normalization. It is created when ``"div"``
                normalization type is selected.
                * ``"scale_range_min_val"``, int/float (optional): ``x_min`` in the formula above. It is created when ``"scale_range"``
                normalization type is selected.
                * ``"scale_range_max_val"``, int/float (optional): ``x_max`` in the formula above. It is created when ``"scale_range"``
                normalization type is selected.
        """
        norm_steps = {}
        norm_steps["orig_dtype"] = str(data.dtype)

        if div_using_max_and_scale:
            if self.scale_range_min_val is None:
                scale_range_min_val = [] if per_channel else [float(data.min())]
            else:
                scale_range_min_val = self.scale_range_min_val
            if self.scale_range_max_val is None:
                scale_range_max_val = [] if per_channel else [float(data.max())]
            else:
                scale_range_max_val = self.scale_range_max_val

        # Changing dtype to floating tensor
        if isinstance(data, torch.Tensor):
            if not torch.is_floating_point(data):
                data = data.to(torch.float32)
        else:
            if not isinstance(data, np.floating):
                data = data.astype(np.float32)

        if per_channel:
            if div_using_max_and_scale:  # "scale_range" normalization type
                for c in range(data.shape[-1]):
                    scale_range_max_val.append(float(data[..., c].max()))
                    scale_range_min_val.append(float(data[..., c].min()))
                    data[..., c] = (data[..., c] - scale_range_min_val[-1]) / (  # type: ignore
                        scale_range_max_val[-1] - scale_range_min_val[-1] + self.eps
                    )
            else:  # "div" normalization type
                if data.max() > 255:
                    div_value = 65535
                else:
                    div_value = 255
                data = data * (1 / div_value)
                norm_steps["div_value"] = div_value
        else:
            if div_using_max_and_scale:  # "scale_range" normalization type
                data = (data - scale_range_min_val[0]) / (scale_range_max_val[0] - scale_range_min_val[0] + self.eps)
            else:  # "div" normalization type
                if data.max() > 255:
                    div_value = 65535
                else:
                    div_value = 255
                data = data * (1 / div_value)

        if div_using_max_and_scale:
            norm_steps["scale_range_min_val"] = scale_range_min_val
            norm_steps["scale_range_max_val"] = scale_range_max_val
        else:
            norm_steps["div_value"] = div_value

        return data, norm_steps

    def __zero_mean_unit_variance_normalization(
        self,
        data: NDArray | torch.Tensor,
    ) -> Tuple[NDArray | torch.Tensor, Dict]:
        """
        Apply zero-mean, unit-variance normalization.

        Parameters
        ----------
        data : (NDArray | torch.Tensor)
            Data to normalize. E.g. ``(y, x, channels)`` in ``2D`` and ``(z, y, x, channels)`` in ``3D``.

        Returns
        -------
        Tuple[NDArray | torch.Tensor, Dict]: Normalized data and normalization info.
            data : 3D/4D Numpy array or torch.Tensor
                Normalized data. E.g. ``(y, x, channels)`` in ``2D`` and ``(z, y, x, channels)`` in ``3D``.

            norm_steps : dict
                Contains information about the normalization steps applied. It is a dict containing the following keys:
                    * ``"orig_dtype"``, str: original dtype of the sample.
                    * ``"mean"``, int/float (optional): mean used in the normalization.
                    * ``"std"``, int/float (optional): std used in the normalization.
        """
        norm_steps = {}
        norm_steps["orig_dtype"] = str(data.dtype)

        if isinstance(data, torch.Tensor):
            if not torch.is_floating_point(data):  # type: ignore
                data = data.to(torch.float32)
        else:
            if not isinstance(data, np.floating):
                data = data.astype(np.float32)

        mean = data.mean() if self.mean is None else self.mean
        std = data.std() if self.std is None else self.std

        norm_steps["mean"] = mean
        norm_steps["std"] = std

        data = (data - mean) / (std + self.eps)
        return data, norm_steps

    def undo_image_norm(
        self,
        data: NDArray | torch.Tensor,
        norm_extra_info: Optional[Dict],
    ) -> NDArray | torch.Tensor:
        """
        Unnormalize given input data following the normalization steps done before for normalizing it.

        Parameters
        ----------
        data : 3D/4D Numpy array
            Data to unnormalize. E.g. ``(y, x, channels)`` in ``2D`` and ``(z, y, x, channels)`` in ``3D``.

        norm_extra_info : dict
            Normalization extra information to undo the normalization. If nothing provided ``self.last_X_norm_extra_info``
            will be used.

        Returns
        -------
        data : 3D/4D Numpy array
            Unnormalized data. E.g. ``(y, x, channels)`` in ``2D`` and ``(z, y, x, channels)`` in ``3D``.
        """
        extra_info = norm_extra_info if norm_extra_info is not None else self.last_X_norm_extra_info

        if self.type in ["div", "scale_range"]:
            data = self.__undo_norm_range01(data, extra_info)
        else:  # zero_mean_unit_variance
            data = self.__undo_zero_mean_unit_variance_normalization(data, extra_info)

            if "float" not in str(extra_info["orig_dtype"]):
                if isinstance(data, np.ndarray):
                    data = np.round(data)
                else:  # torch.Tensor
                    data = torch.round(data)
                mindata = data.min()
                data = data + abs(mindata)  # type: ignore

        if isinstance(data, np.ndarray):
            data = data.astype(torch_numpy_dtype_dict[extra_info["orig_dtype"]][1])
        else:
            data = data.to(torch_numpy_dtype_dict[extra_info["orig_dtype"]][0])

        return data

    def __undo_norm_range01(
        self,
        data: NDArray | torch.Tensor,
        norm_extra_info: Dict,
    ) -> NDArray | torch.Tensor:
        """
        Undo normalization by multiplaying a factor and optionally summing a minimum value. Opposite function of ``__norm_range01``.

        Parameters
        ----------
        data : 3D/4D Numpy array
            Data to unnormalize. E.g. ``(y, x, channels)`` in ``2D`` and ``(z, y, x, channels)`` in ``3D``.

        norm_extra_info : dict
            Information about the normalization. Expected keys are:
                * ``"div_value"``, int/float (optional): number used to divide during normalization. Needed when 'div'
                  normalization was selected.
                * ``"scale_range_max_val"``, int/float (optional): number used to divide during normalization. Needed
                  when 'scale_range' normalization was selected.
                * ``"scale_range_min_val"``, int/float (optional): number used to divide during normalization. Needed
                  when 'scale_range' normalization was selected.

        Returns
        -------
        data : 3D/4D Numpy array
            Unnormalized data. E.g. ``(y, x, channels)`` in ``2D`` and ``(z, y, x, channels)`` in ``3D``.
        """
        extra_info = norm_extra_info if norm_extra_info is not None else self.last_X_norm_extra_info

        # Prevent values go outside expected range
        if isinstance(data, np.ndarray):
            data = np.clip(data, 0, 1)
        else:
            data = torch.clamp(data, 0, 1)

        if self.type == "div":
            assert (
                "div_value" in extra_info
            ), "'div_value' not in 'norm_dict'. Ensure you input the same normalization dict used to normalize the data previously"
            data = data * extra_info["div_value"]
        elif self.type == "scale_range":
            assert (
                "scale_range_max_val" in extra_info
            ), "'scale_range_max_val' not in 'extra_info'. Ensure you input the same normalization dict used to normalize the data previously"
            assert (
                "scale_range_min_val" in extra_info
            ), "'scale_range_min_val' not in 'extra_info'. Ensure you input the same normalization dict used to normalize the data previously"

            data = (data * extra_info["scale_range_max_val"]) + extra_info["scale_range_min_val"]
        return data

    def __undo_zero_mean_unit_variance_normalization(
        self,
        data: NDArray | torch.Tensor,
        norm_extra_info: Dict,
    ) -> NDArray | torch.Tensor:
        """
        Unnormalization of input data by multiplying by the std and adding the mean.
        
        Opposite function of ``zero_mean_unit_variance_normalization``.

        Parameters
        ----------
        data : 3D/4D Numpy array
            Image to unnormalize. E.g. ``(y, x, channels)`` in ``2D`` and ``(z, y, x, channels)`` in ``3D``.

        norm_extra_info : dict
            Information about the normalization. Expected keys are:
                * ``"mean"``, int/float: mean used in normalization.
                * ``"std"``, int/float: std used in normalization.

        Returns
        -------
        data : 3D/4D Numpy array
            Unnormalized data. E.g. ``(y, x, channels)`` in ``2D`` and ``(z, y, x, channels)`` in ``3D``.
        """
        extra_info = norm_extra_info if norm_extra_info is not None else self.last_X_norm_extra_info
        assert (
            "std" in extra_info
        ), "'std' not in 'norm_dict'. Ensure you input the same normalization dict used to normalize the data previously"

        assert (
            "mean" in extra_info
        ), "'mean' not in 'norm_dict'. Ensure you input the same normalization dict used to normalize the data previously"

        return (data * extra_info["std"]) + extra_info["mean"]

    def __str__(self):
        """Return string representation of the object."""
        return str(self.__dict__)

    def __repr__(self):
        """Return string representation of the object."""
        return str(self.__dict__)

    def copy(self):
        """
        Return a deep copy of the Normalization object.

        Returns:
            Normalization: Deep copy of the object.
        """
        return copy.deepcopy(self)
