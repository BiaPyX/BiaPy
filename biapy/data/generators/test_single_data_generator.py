import os
import h5py
import numpy as np
from torch.utils.data import Dataset
from typing import (
    List,
    Tuple,
    Dict,
    Any,
    Callable,
)

from biapy.data.pre_processing import normalize, norm_range01, percentile_clip
from biapy.data.generators.augmentors import center_crop_single, resize_img
from biapy.data.data_manipulation import load_img_data, sample_satisfy_conds, pad_and_reflect


class test_single_data_generator(Dataset):
    """
    Image data generator without data augmentation. Used only for test data.

    Parameters
    ----------
    ndim : int
        Dimensions of the data (``2`` for 2D and ``3`` for 3D).

    X : list of dict
        X data. Each item in the list represents a sample of the dataset. Each sample is represented as follows:
            * ``"filename"``: name of the image to extract the data sample from.
            * ``"dir"``: directory where the image resides.
            * ``"shape"``: shape of the sample.
            * ``"class_name"``: name of the class.
            * ``"class"``: integer that represents the class.
            * ``"img"`` (optional): image sample itself. It is a ndarrray of  ``(y, x, channels)`` in ``2D`` and
              ``(z, y, x, channels)`` in ``3D``. Provided if the user selected to load images into memory.

    provide_Y: bool, optional
        Whether the ground truth has been provided or not.

    dims: str, optional
        Dimension of the data. Possible options: ``2D`` or ``3D``.

    seed : int, optional
        Seed for random functions.

    instance_problem : bool, optional
        Not used here.

    norm_dict : dict, optional
        Normalization instructions.

    reduce_mem : bool, optional
        To reduce the dtype from float32 to float16.

    crop_center : bool, optional
        Whether to extract a central crop from each image.

    convert_to_rgb : bool, optional
        Whether to convert images into 3-channel, i.e. RGB, by using the information of the first channel.

    filter_conds : list of lists of str, optional
        Filter conditions to be applied to the data. The three variables, ``filter_conds``, ``filter_vals`` and ``filter_vals``
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
        Represent the values of the properties listed in ``filter_conds`` that the images need to satisfy to not be dropped.

    filter_signs : list of list of str, optional
        Signs to do the comparison for data filtering. Options: [``'gt'``, ``'ge'``, ``'lt'``, ``'le'``] that corresponds to
        "greather than", e.g. ">", "greather equal", e.g. ">=", "less than", e.g. "<", and "less equal" e.g. "<=" comparisons.

    preprocess_data : bool, optional
        Whether to apply preprocessing to test data or not.

    preprocess_cfg : dict, optional
        Configuration of the preprocessing.

    data_shape : tuple of int, optional
        Shape of the images to output.

    reflect_to_complete_shape : bool, optional
        Whether to reshape the dimensions that does not satisfy the patch shape selected by padding it with reflect.
    """

    def __init__(
        self,
        ndim: int,
        X: List[Dict],
        provide_Y: bool = False,
        seed: int = 42,
        norm_dict: Dict | None = None,
        crop_center: bool = False,
        reduce_mem: bool = False,
        convert_to_rgb: bool = False,
        filter_props: List[List[str]] = [],
        filter_vals: List[List[str]] | None = None,
        filter_signs: List[List[str]] | None = None,
        preprocess_data: Callable = False,
        preprocess_cfg: dict | None = None,
        data_shape: Tuple[int, ...] = (256, 256, 1),
        reflect_to_complete_shape: bool = True,
    ):
        assert norm_dict != None, "Normalization instructions must be provided with 'norm_dict'"

        self.X = X
        self.provide_Y = provide_Y
        self.convert_to_rgb = convert_to_rgb
        self.filter_samples = True if len(filter_props) > 0 else False
        self.filter_props = filter_props
        self.filter_vals = filter_vals
        self.filter_signs = filter_signs
        self.preprocess_data = preprocess_data
        self.preprocess_cfg = preprocess_cfg
        self.reflect_to_complete_shape = reflect_to_complete_shape
        self.data_shape = data_shape

        if not reduce_mem:
            self.dtype = np.float32
            self.dtype_str = "float32"
        else:
            self.dtype = np.float16
            self.dtype_str = "float16"
        self.crop_center = crop_center
        self.len = len(self.X)
        self.seed = seed
        self.ndim = ndim
        self.o_indexes = np.arange(self.len)

        self.norm_dict = norm_dict
        # Check if a division is required
        if norm_dict["enable"]:
            img, _, xnorm, _ = self.load_sample(0)
            self.norm_dict["orig_dtype"] = img.dtype

    # img, img_class, xnorm, filename
    def load_sample(self, idx: int) -> Tuple[Any, Any, Any, Any, Any]:
        """
        Load one data sample given its corresponding index.

        Parameters
        ----------
        idx : int
            Sample index counter.

        Returns
        -------
        img : 3D/4D ndarray array
            Image read. E.g. ``(z, y, x, num_classes)`` for 3D or ``(y, x, num_classes)`` for 2D.

        img_class: int
            Class of the image. It will be -1 when no class is available.

        xnorm : dict
            X element normalization steps.

        sample : dict
            * ``"filename"``: name of the image to extract the data sample from.
            * ``"dir"``: directory where the image resides.
            * ``"shape"``: shape of the sample.
            * ``"class_name"``: name of the class.
            * ``"class"``: integer that represents the class.
            * ``"img"`` (optional): image sample itself. It is a ndarrray of  ``(y, x, channels)`` in ``2D`` and
              ``(z, y, x, channels)`` in ``3D``. Provided if the user selected to load images into memory.
            * ``"discard"`` (optional): whether the sample should be discarded or not. Present if ``filter_conds``,``filter_vals``
              and ``filter_signs`` were provided.
            * ``"reflected_orig_shape"`` (optional): original shape of the image before reflecting. Present if ``reflect_to_complete_shape``
              is ``True`` and ``crop_center`` is ``False``.
        """
        sample = self.X[idx].copy()

        img, img_file = load_img_data(
            os.path.join(sample["dir"], sample["filename"]),
            is_3d=(self.ndim == 3),
            data_within_zarr_path=sample["path_in_zarr"] if "path_in_zarr" in sample else None,
        )

        if (
            sample["filename"].endswith(".zarr")
            or sample["filename"].endswith(".hdf5")
            or sample["filename"].endswith(".h5")
        ):
            if not self.test_by_chunks:
                raise ValueError(
                    "If you are using Zarr images please set 'TEST.BY_CHUNKS.ENABLE' and configure " "its options."
                )
            
            if img_file is not None and isinstance(img_file, h5py.File):
                sample["img_file_to_close"] = img_file
        else:
            # Skip processing image
            discard = False
            if self.filter_samples:
                discard = sample_satisfy_conds(
                    img,
                    self.filter_props,
                    self.filter_vals,
                    self.filter_signs,
                )
            sample["discard"] = discard

            if not discard:
                # Preprocess test data
                if self.preprocess_data is not None:
                    img = self.preprocess_data(
                        self.preprocess_cfg,
                        x_data=[img],
                        is_2d=(self.ndim == 2),
                    )[0]

                # Reflect data to complete the needed shape
                if not self.crop_center and self.reflect_to_complete_shape:
                    reflected_orig_shape = img.shape
                    img = pad_and_reflect(
                        img,
                        self.data_shape,
                        verbose=True,
                    )
                    sample["reflected_orig_shape"] = reflected_orig_shape

                if self.convert_to_rgb and img.shape[-1] == 1:
                    img = np.repeat(img, 3, axis=-1)

                # Data channel check
                if self.data_shape[-1] != img.shape[-1]:
                    raise ValueError(
                        "Channel of the DATA.PATCH_SIZE given {} does not correspond with the loaded image {}. "
                        "Please, check the channels of the images!".format(self.data_shape[-1], img.shape[-1])
                    )

        img_class = -1
        if "class" in sample:
            img_class = sample["class"]
            img_class = np.expand_dims(np.array(img_class), 0)

        # Normalization
        xnorm = None
        if self.norm_dict["enable"]:
            # Percentile clipping
            if "lower_bound" in self.norm_dict:
                img, _, _ = percentile_clip(
                    img,
                    lower=self.norm_dict["lower_bound"],
                    upper=self.norm_dict["upper_bound"],
                )

            if self.norm_dict["type"] == "div":
                img, xnorm = norm_range01(img, dtype=self.dtype)  # type: ignore
            elif self.norm_dict["type"] == "scale_range":
                img, xnorm = norm_range01(img, dtype=self.dtype, div_using_max_and_scale=True)  # type: ignore
            elif self.norm_dict["type"] == "custom":
                xnorm = {}
                xnorm["mean"] = img.mean() if "mean" not in self.norm_dict else self.norm_dict["mean"]
                xnorm["std"] = img.std() if "std" not in self.norm_dict else self.norm_dict["std"]
                img = normalize(img, xnorm["mean"], xnorm["std"], out_type=self.dtype_str)

        img = np.expand_dims(img, 0).astype(self.dtype)
        return img, img_class, xnorm, sample

    def __len__(self) -> int:
        """Defines the length of the generator"""
        return self.len

    def __getitem__(self, index: int) -> Dict:
        """
        Generation of one sample of data.

        Parameters
        ----------
        index : int
            Sample index counter.

        Returns
        -------
        dict : dict
            Test sample containing:
            * ``"X"``: X data. It is a ndarrray of  ``(y, x, channels)`` in ``2D`` and ``(z, y, x, channels)`` in ``3D``.
            * ``"X_norm"``: X element normalization steps.
            * ``"Y"``: Y data. It is a ndarrray of  ``(y, x, channels)`` in ``2D`` and ``(z, y, x, channels)`` in ``3D``.
            * ``"Y_norm"``: Y element normalization steps.
            * ``"filename"``: name of the image to extract the data sample from.
            * ``"dir"``: directory where the image resides.
            * ``"discard"`` (optional): whether the sample should be discarded or not. Present if ``filter_conds``,``filter_vals``
              and ``filter_signs`` were provided.
            * ``"reflected_orig_shape"`` (optional): original shape of the image before reflecting. Present if ``reflect_to_complete_shape``
              is ``True``.
        """
        img, img_class, norm, sample = self.load_sample(index)

        if self.crop_center and img.shape[:-1] != self.data_shape[:-1]:
            img = center_crop_single(img[0], self.data_shape)
            img = resize_img(img, self.data_shape[:-1])
            img = np.expand_dims(img, 0)

        if norm is not None:
            self.norm_dict.update(norm)

        test_sample = {
            "X": img,
            "X_norm": self.norm_dict,
            "filename": sample["filename"],
            "dir": sample["dir"],
        }
        if self.provide_Y:
            test_sample["Y"] = img_class
        if "discard" in sample:
            test_sample["discard"] = sample["discard"]
        if "reflected_orig_shape" in sample:
            test_sample["reflected_orig_shape"] = sample["reflected_orig_shape"]

        return test_sample

    def get_data_normalization(self) -> Dict:
        return self.norm_dict
