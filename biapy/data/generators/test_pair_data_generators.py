from torch.utils.data import Dataset
import os
import h5py
import numpy as np
from typing import (
    List,
    Tuple,
    Dict,
    Any,
    Callable,
)

from biapy.data.pre_processing import zero_mean_unit_variance_normalization, norm_range01, percentile_clip
from biapy.data.data_manipulation import load_img_data, sample_satisfy_conds, pad_and_reflect


class test_pair_data_generator(Dataset):
    """
    Image data generator without data augmentation. Used only for test data.

    Parameters
    ----------
    X : list of dict
        X data. Each item in the list represents a sample of the dataset. Each sample is represented as follows:
            * ``"filename"``: name of the image to extract the data sample from.
            * ``"dir"``: directory where the image resides.
            * ``"coords"``: dictionary with the coordinates to extract the sample from the image. If ``None`` it implies that a random 
              patch needs to be extracted. Following keys are avaialable:
                * ``"z_start"``: starting point of the patch in Z axis.
                * ``"z_end"``: end point of the patch in Z axis.
                * ``"y_start"``: starting point of the patch in Y axis.
                * ``"y_end"``: end point of the patch in Y axis.
                * ``"x_start"``: starting point of the patch in X axis.
                * ``"x_end"``: end point of the patch in X axis.
            * ``"original_data_shape"``: shape of the image where the samples is extracted (useful for reconstructing it later),
            * ``"shape"``: shape of the sample.
            * ``"img"`` (optional): image sample itself. It is a ndarrray of  ``(y, x, channels)`` in ``2D`` and
              ``(z, y, x, channels)`` in ``3D``. Provided when ``train_in_memory`` is ``True``.
            * ``"gt_associated_id"`` (optional): position of associated ground truth of the sample within its list. Present if the 
              user selected ``PROBLEM.IMAGE_TO_IMAGE.MULTIPLE_RAW_ONE_TARGET_LOADER`` to be ``True``.
            * ``"parallel_data"``(optional): to ``True`` is the sample is a Zarr/H5 file. Not present otherwise.
            * ``"input_axes"`` (optional): order of the axes in Zarr. Not present in non-Zarr/H5 files.
            * ``"path_in_zarr"``(optional): path where the data resides within the Zarr. Provided when ``multiple_data_within_zarr`` was 
              set in ``train_zarr_data_information``.  

    Y : list of dict
        Y data. Each item in the list represents a sample of the dataset. Each sample is represented as follows:
            * ``"filename"``: name of the image to extract the data sample from.
            * ``"dir"``: directory where the image resides.
            * ``"coords"``: dictionary with the coordinates to extract the sample from the image. If ``None`` it implies that a random 
              patch needs to be extracted. Following keys are avaialable:
                * ``"z_start"``: starting point of the patch in Z axis.
                * ``"z_end"``: end point of the patch in Z axis.
                * ``"y_start"``: starting point of the patch in Y axis.
                * ``"y_end"``: end point of the patch in Y axis.
                * ``"x_start"``: starting point of the patch in X axis.
                * ``"x_end"``: end point of the patch in X axis.
            * ``"original_data_shape"``: shape of the image where the samples is extracted (useful for reconstructing it later),
            * ``"shape"``: shape of the sample.
            * ``"img"`` (optional): image sample itself. It is a ndarrray of  ``(y, x, channels)`` in ``2D`` and
              ``(z, y, x, channels)`` in ``3D``. Provided if the user selected to load images into memory.
            * ``"gt_associated_id"`` (optional): position of associated ground truth of the sample within its list. Present if the 
              user selected ``PROBLEM.IMAGE_TO_IMAGE.MULTIPLE_RAW_ONE_TARGET_LOADER`` to be ``True``.
            * ``"parallel_data"``(optional): to ``True`` is the sample is a Zarr/H5 file. Not present otherwise.
            * ``"input_axes"`` (optional): order of the axes in Zarr. Not present in non-Zarr/H5 files.
            * ``"path_in_zarr"``(optional): path where the data resides within the Zarr. Provided when ``multiple_data_within_zarr`` was 
              set in ``train_zarr_data_information``.  

    ndim : int
        Dimensions of the data (``2`` for 2D and ``3`` for 3D).

    norm_dict : str
        Normalization instructions.

    test_by_chunks : bool, optional
        Tell the generator that the data is going to be read by chunks and by H5/Zarr files.

    provide_Y: bool, optional
        Whether to return ground truth or not.

    seed : int, optional
        Seed for random functions.

    instance_problem : bool, optional
        To not divide the labels if being in an instance segmenation problem.

    reduce_mem : bool, optional
        To reduce the dtype from float32 to float16.

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
        X: List[Dict],
        Y: List[Dict],
        ndim: int,
        norm_dict: dict = {},
        test_by_chunks: bool = False,
        provide_Y: bool = False,
        seed: int = 42,
        instance_problem: bool = False,
        reduce_mem: bool = False,
        convert_to_rgb: bool = False,
        filter_props: List[List[str]] = [],
        filter_vals: List[List[str]] | None = None,
        filter_signs: List[List[str]] | None = None,
        preprocess_data: Callable | None = None,
        preprocess_cfg: dict | None = None,
        data_shape: Tuple[int, ...] = (256, 256, 1),
        reflect_to_complete_shape: bool = True,
    ):
        assert norm_dict["mask_norm"] in ["as_mask", "as_image", "none"]
        if preprocess_data is not None and preprocess_cfg is None:
            raise ValueError("'preprocess_cfg' must be set when 'preprocess_data' is provided")

        self.X = X
        self.Y = Y
        self.test_by_chunks = test_by_chunks
        self.provide_Y = provide_Y
        self.convert_to_rgb = convert_to_rgb
        self.norm_dict = norm_dict
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

        self.seed = seed
        self.ndim = ndim
        self.len = len(X)

        # Check if a division is required
        if provide_Y:
            self.Y_norm = {}
            self.Y_norm["type"] = "div"
        img, mask, xnorm, _, _ = self.load_sample(0)

        if norm_dict["enable"]:
            self.norm_dict["orig_dtype"] = img.dtype if isinstance(img, np.ndarray) else "Zarr"

        if xnorm:
            self.norm_dict.update(xnorm)

        if mask is not None and not test_by_chunks:
            self.Y_norm = {}
            if norm_dict["mask_norm"] == "as_mask":
                self.Y_norm["type"] = "div"
                if np.max(mask) > 30 and not instance_problem:
                    self.Y_norm["div"] = 1
            elif norm_dict["mask_norm"] == "as_image":
                self.Y_norm.update(self.norm_dict)

    def norm_X(self, img: np.ndarray) -> Tuple[np.ndarray, Dict | None]:
        """
        X data normalization.

        Parameters
        ----------
        img : 3D/4D Numpy array
            X element, for instance, an image. E.g. ``(y, x, channels)`` in ``2D`` and ``(z, y, x, channels)`` in ``3D``.

        Returns
        -------
        img : 3D/4D Numpy array
            X element normalized. E.g. ``(y, x, channels)`` in ``2D`` and ``(z, y, x, channels)`` in ``3D``.

        xnorm : dict, optional
            Normalization info.
        """
        xnorm = None
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
            img = zero_mean_unit_variance_normalization(img, xnorm["mean"], xnorm["std"], out_type=self.dtype_str)

        return img, xnorm

    def norm_Y(self, mask: np.ndarray) -> Tuple[np.ndarray, Dict | None]:
        """
        Y data normalization.

        Parameters
        ----------
        mask : 3D/4D Numpy array
            Y element, for instance, an image's mask. E.g. ``(y, x, channels)`` in ``2D`` and ``(z, y, x, channels)`` in
            ``3D``.

        Returns
        -------
        mask : 3D/4D Numpy array
            Y element normalized. E.g. ``(y, x, channels)`` in ``2D`` and ``(z, y, x, channels)`` in ``3D``.

        ynorm : dict, optional
            Normalization info.
        """
        ynorm = None
        if self.norm_dict["mask_norm"] == "as_mask":
            if "div" in self.Y_norm:
                mask = mask / 255
        elif self.norm_dict["mask_norm"] == "as_image":
            # Percentile clipping
            if "lower_bound" in self.norm_dict:
                mask, _, _ = percentile_clip(
                    mask,
                    lower=self.norm_dict["lower_bound"],
                    upper=self.norm_dict["upper_bound"],
                )

            if self.norm_dict["type"] == "div":
                mask, ynorm = norm_range01(mask, dtype=self.dtype)  # type: ignore
            elif self.norm_dict["type"] == "scale_range":
                mask, xnorm = norm_range01(mask, dtype=self.dtype, div_using_max_and_scale=True)  # type: ignore
            elif self.norm_dict["type"] == "custom":
                ynorm = {}
                ynorm["mean"] = mask.mean() if "mean" not in self.norm_dict else self.norm_dict["mean"]
                ynorm["std"] = mask.std() if "std" not in self.norm_dict else self.norm_dict["std"]
                mask = zero_mean_unit_variance_normalization(mask, ynorm["mean"], ynorm["std"], out_type=self.dtype_str)

        return mask, ynorm

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

        mask 3D/4D ndarray array
            Mask read. E.g. ``(z, y, x, num_classes)`` for 3D or ``(y, x, num_classes)`` for 2D.

        xnorm : dict
            X element normalization steps.

        ynorm : dict
            Y element normalization steps.

        sample : dict
            * ``"filename"``: name of the image to extract the data sample from.
            * ``"dir"``: directory where the image resides.
            * ``"coords"``: dictionary with the coordinates to extract the sample from the image. If ``None`` it implies that a random 
              patch needs to be extracted. Following keys are avaialable:
                * ``"z_start"``: starting point of the patch in Z axis.
                * ``"z_end"``: end point of the patch in Z axis.
                * ``"y_start"``: starting point of the patch in Y axis.
                * ``"y_end"``: end point of the patch in Y axis.
                * ``"x_start"``: starting point of the patch in X axis.
                * ``"x_end"``: end point of the patch in X axis.
            * ``"original_data_shape"``: shape of the image where the samples is extracted (useful for reconstructing it later),
            * ``"shape"``: shape of the sample.
            * ``"img"`` (optional): image sample itself. It is a ndarrray of  ``(y, x, channels)`` in ``2D`` and
              ``(z, y, x, channels)`` in ``3D``. Provided when ``train_in_memory`` is ``True``.
            * ``"gt_associated_id"`` (optional): position of associated ground truth of the sample within its list. Present if the 
              user selected ``PROBLEM.IMAGE_TO_IMAGE.MULTIPLE_RAW_ONE_TARGET_LOADER`` to be ``True``.
            * ``"discard"`` (optional): whether the sample should be discarded or not. Present if ``filter_conds``,``filter_vals``
              and ``filter_signs`` were provided.
            * ``"reflected_orig_shape"`` (optional): original shape of the image before reflecting. Present if ``reflect_to_complete_shape``
              is ``True``.
            * ``"img_file_to_close"`` (optional): file of the image to close. Present if the loaded file is H5.
            * ``"mask_file_to_close"`` (optional): file of the image to close. Present if the loaded file is H5.
            * ``"parallel_data"``(optional): to ``True`` is the sample is a Zarr/H5 file. Not present otherwise.
            * ``"input_axes"`` (optional): order of the axes in Zarr. Not present in non-Zarr/H5 files.
            * ``"path_in_zarr"``(optional): path where the data resides within the Zarr. Provided when ``multiple_data_within_zarr`` was 
              set in ``train_zarr_data_information``.  
        """
        mask, ynorm = None, None

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

        if self.provide_Y:
            # "gt_associated_id" available only in PROBLEM.IMAGE_TO_IMAGE.MULTIPLE_RAW_ONE_TARGET_LOADER
            if "gt_associated_id" in sample:
                msample = self.Y[sample["gt_associated_id"]]
                mask, mask_file = load_img_data(
                    os.path.join(msample["dir"], msample["filename"]), is_3d=(self.ndim == 3),
                )
            else:
                mask, mask_file = load_img_data(
                    os.path.join(self.Y[idx]["dir"], self.Y[idx]["filename"]),
                    is_3d=(self.ndim == 3),
                    data_within_zarr_path=self.Y[idx]["path_in_zarr"] if "path_in_zarr" in self.Y[idx] else None,
                )
                if mask_file is not None and isinstance(mask_file, h5py.File):
                    sample["mask_file_to_close"] = mask_file
            sample["gt_associated_id"] = mask

        if not self.test_by_chunks:
            # Skip processing image
            discard = False
            if self.filter_samples:
                foreground_filter_requested = any([True for cond in self.filter_props if "foreground" in cond])
                discard = sample_satisfy_conds(
                    img,
                    self.filter_props,
                    self.filter_vals,
                    self.filter_signs,
                    mask=mask if foreground_filter_requested else None,
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
                    if self.provide_Y:
                        mask = self.preprocess_data(
                            self.preprocess_cfg,
                            y_data=[mask],
                            is_2d=(self.ndim == 2),
                            is_y_mask=True,
                        )[0]

                # Reflect data to complete the needed shape
                if self.reflect_to_complete_shape:
                    reflected_orig_shape = img.shape
                    img = pad_and_reflect(
                        img,
                        self.data_shape,
                        verbose=True,
                    )
                    if self.provide_Y:
                        mask = pad_and_reflect(
                            mask,
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

        xnorm = self.norm_dict
        if not self.test_by_chunks:
            img = np.array(img)
            # Normalization
            if self.norm_dict["enable"]:
                img, xnorm = self.norm_X(img)
            if self.provide_Y and self.norm_dict["enable"]:
                mask, ynorm = self.norm_Y(np.array(mask))

            img = np.expand_dims(img, 0).astype(self.dtype)
            if self.provide_Y:
                mask = np.expand_dims(np.array(mask), 0)
                if self.norm_dict["mask_norm"] == "as_mask":
                    mask = mask.astype(np.uint8)

        return img, mask, xnorm, ynorm, sample

    def __len__(self) -> int:
        """Defines the length of the generator"""
        return self.len

    def __getitem__(self, index: int) -> Dict:
        """
        Generation of one pair of data.

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
            * ``"gt_associated_id"`` (optional): position of associated ground truth of the sample within its list. Present if the 
              user selected ``PROBLEM.IMAGE_TO_IMAGE.MULTIPLE_RAW_ONE_TARGET_LOADER`` to be ``True``.
            * ``"discard"``: whether the sample should be discarded or not. Present if ``filter_conds``,``filter_vals`` and
            ``filter_signs`` were provided.
            * ``"reflected_orig_shape"`` (optional): original shape of the image before reflecting. Present if ``reflect_to_complete_shape``
              is ``True``.
            * ``"img_file_to_close"`` (optional): file of the image to close. Present if the loaded file is H5.
            * ``"mask_file_to_close"`` (optional): file of the image to close. Present if the loaded file is H5.
        """
        img, mask, xnorm, ynorm, sample = self.load_sample(index)

        if xnorm is not None:
            self.norm_dict.update(xnorm)
        if ynorm is not None:
            self.Y_norm.update(ynorm)

        if self.provide_Y:
            test_sample = {
                "X": img,
                "X_norm": self.norm_dict,
                "Y": mask,
                "Y_norm": self.Y_norm,
            }
        else:
            test_sample = {
                "X": img,
                "X_norm": self.norm_dict,
            }

        test_sample["filename"] = sample["filename"]
        test_sample["dir"] = sample["dir"]
        if "gt_associated_id" in sample:
            test_sample["gt_associated_id"] = sample["gt_associated_id"]
        if "img_file_to_close" in sample:
            test_sample["img_file_to_close"] = sample["img_file_to_close"]
        if "mask_file_to_close" in sample:
            test_sample["mask_file_to_close"] = sample["mask_file_to_close"]
        if "discard" in sample:
            test_sample["discard"] = sample["discard"]
        if "reflected_orig_shape" in sample:
            test_sample["reflected_orig_shape"] = sample["reflected_orig_shape"]

        return test_sample

    def get_data_normalization(self):
        return self.norm_dict
