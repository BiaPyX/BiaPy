from torch.utils.data import Dataset
import h5py
import os
import numpy as np
from typing import List, Tuple, Dict, Any, Callable, Optional
from numpy.typing import NDArray

from biapy.data.data_manipulation import load_img_data, sample_satisfy_conds, pad_and_reflect
from biapy.data.dataset import BiaPyDataset, DataSample
from biapy.data.norm import Normalization


class test_pair_data_generator(Dataset):
    """
    Image data generator without data augmentation. Used only for test data.

    Parameters
    ----------
    X : BiaPyDataset
        X data.

    Y : BiaPyDataset
        Y data.

    ndim : int
        Dimensions of the data (``2`` for 2D and ``3`` for 3D).

    norm_module : Normalization
        Normalization module that defines the normalization steps to apply.

    test_by_chunks : bool, optional
        Tell the generator that the data is going to be read by chunks and by H5/Zarr files.

    provide_Y: bool, optional
        Whether to return ground truth or not.

    seed : int, optional
        Seed for random functions.

    instance_problem : bool, optional
        To not divide the labels if being in an instance segmenation problem.

    convert_to_rgb : bool, optional
        Whether to convert images into 3-channel, i.e. RGB, by using the information of the first channel.

    filter_props : list of lists of str, optional
        Filter conditions to be applied to the data. The three variables, ``filter_props``, ``filter_vals`` and ``filter_vals``
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
        Represent the values of the properties listed in ``filter_props`` that the images need to satisfy to not be dropped.

    filter_signs : list of list of str, optional
        Signs to do the comparison for data filtering. Options: [``'gt'``, ``'ge'``, ``'lt'``, ``'le'``] that corresponds to
        "greather than", e.g. ">", "greather equal", e.g. ">=", "less than", e.g. "<", and "less equal" e.g. "<=" comparisons.

    preprocess_data : Callable, optional
        Preprocessing function to apply.

    preprocess_cfg : dict, optional
        Configuration of the preprocessing.

    data_shape : tuple of int, optional
        Shape of the images to output.

    reflect_to_complete_shape : bool, optional
        Whether to reshape the dimensions that does not satisfy the patch shape selected by padding it with reflect.

    n_classes : int, optional
        Number of classes to predict.

    ignore_index : int, optional
        Value to ignore in the loss/metrics. In this generator is not used but added for compatibility
        with ``PairBaseDataGenerator``.
    """

    def __init__(
        self,
        X: BiaPyDataset,
        Y: BiaPyDataset,
        ndim: int,
        norm_module: Normalization,
        test_by_chunks: bool = False,
        provide_Y: bool = False,
        seed: int = 42,
        instance_problem: bool = False,
        convert_to_rgb: bool = False,
        filter_props: List[List[str]] = [],
        filter_vals: Optional[List[List[float | int]]] = None,
        filter_signs: Optional[List[List[str]]] = None,
        preprocess_data: Optional[Callable] = None,
        preprocess_cfg: Optional[Dict] = None,
        data_shape: Tuple[int, ...] = (256, 256, 1),
        reflect_to_complete_shape: bool = True,
        n_classes: int = 1,
        ignore_index: Optional[int]=None,
    ):
        if preprocess_data and preprocess_cfg is None:
            raise ValueError("'preprocess_cfg' must be set when 'preprocess_data' is provided")

        self.X = X
        self.Y = Y
        self.test_by_chunks = test_by_chunks
        self.provide_Y = provide_Y
        self.convert_to_rgb = convert_to_rgb
        self.norm_module = norm_module
        self.filter_samples = True if len(filter_props) > 0 else False
        self.filter_props = filter_props
        self.filter_vals = filter_vals
        self.filter_signs = filter_signs
        self.preprocess_data = preprocess_data
        self.preprocess_cfg = preprocess_cfg
        self.reflect_to_complete_shape = reflect_to_complete_shape
        self.data_shape = data_shape
        self.seed = seed
        self.ndim = ndim
        self.instance_problem = instance_problem
        self.n_classes = n_classes
        self.ignore_index = ignore_index

        # As in test entire images are processed one by one X.sample_list and X.dataset_info must match in length. If not
        # means that validation data is being used as test, so we need to clean the sample_list.
        if len(X.dataset_info) != len(X.sample_list):
            new_sample_list = []
            for i in range(len(X.dataset_info)):
                new_sample_list.append(DataSample(fid=i, coords=None))
            X.sample_list = new_sample_list
            if self.provide_Y:
                Y.sample_list = new_sample_list.copy()
        self.len = len(X.sample_list)

        img, mask, _, sample_extra_info, _ = self.load_sample(0, first_load=True)
        if "img_file_to_close" in sample_extra_info and isinstance(sample_extra_info["img_file_to_close"], h5py.File):
            sample_extra_info["img_file_to_close"].close()
        if "mask_file_to_close" in sample_extra_info and isinstance(sample_extra_info["mask_file_to_close"], h5py.File):
            sample_extra_info["mask_file_to_close"].close()
        self.norm_module.orig_dtype = img.dtype if isinstance(img, np.ndarray) else "Zarr"  # type: ignore

        if mask is not None and not test_by_chunks:
            # Store which channels are binary or not (e.g. distance transform channel is not binary)
            self.norm_module.set_stats_from_mask(mask, n_classes=n_classes, ignore_index=ignore_index, instance_problem=instance_problem)

    def load_sample(
        self,
        idx: int,
        first_load: bool = False,
    ) -> Tuple[NDArray, NDArray | None, DataSample, Dict, Dict | None]:
        """
        Load one data sample given its corresponding index.

        Parameters
        ----------
        idx : int
            Sample index counter.

        first_load : bool, optional
            Whether its the first time a sample is loaded to prevent normalizing it.

        Returns
        -------
        img : 3D/4D ndarray array
            Image read. E.g. ``(z, y, x, channels)`` for 3D or ``(y, x, channels)`` for 2D.

        mask 3D/4D ndarray array
            Mask read. E.g. ``(z, y, x, channels)`` for 3D or ``(y, x, channels)`` for 2D.

        sample : DataSample
            Loaded sample.

        sample_extra_info : dict
            Extra information of the loaded sample. Contains the following keys:
            * ``"gt_associated_id"``, int (optional): position of associated ground truth of the sample within its list. Present if the
              user selected ``PROBLEM.IMAGE_TO_IMAGE.MULTIPLE_RAW_ONE_TARGET_LOADER`` to be ``True``.
            * ``"discard"``, bool (optional): whether the sample should be discarded or not. Present if ``filter_props``,``filter_vals``
              and ``filter_signs`` were provided.
            * ``"reflected_orig_shape"``, tuple of int (optional): original shape of the image before reflecting. Present if ``reflect_to_complete_shape``
              is ``True``.
            * ``"img_file_to_close"``, h5py.File (optional): file of the image to close. Present if the loaded file is H5.
            * ``"mask_file_to_close"``, h5py.File (optional): file of the image to close. Present if the loaded file is H5.

        norm_extra_info : dict
            Normalization extra information useful to undo the normalization after.
        """
        mask = None
        norm_extra_info = None
        sample = self.X.sample_list[idx]
        sample_extra_info = {}

        img, img_file = load_img_data(
            self.X.dataset_info[sample.fid].path,
            is_3d=(self.ndim == 3),
            data_within_zarr_path=sample.get_path_in_zarr(),
        )

        if any(self.X.dataset_info[sample.fid].path.endswith(x) for x in [".zarr", ".n5", ".h5", ".hdf5", ".hdf"]):
            if not self.test_by_chunks:
                raise ValueError(
                    "If you are using Zarr images please set 'TEST.BY_CHUNKS.ENABLE' and configure " "its options."
                )
            if img_file and isinstance(img_file, h5py.File):
                sample_extra_info["img_file_to_close"] = img_file

        if self.provide_Y:
            # "gt_associated_id" available only in PROBLEM.IMAGE_TO_IMAGE.MULTIPLE_RAW_ONE_TARGET_LOADER
            associated_id = sample.get_gt_associated_id()
            if associated_id is not None:
                msample = self.Y.sample_list[associated_id]
                mask, mask_file = load_img_data(
                    self.Y.dataset_info[msample.fid].path,
                    is_3d=(self.ndim == 3),
                )
                sample_extra_info["gt_associated_id"] = associated_id
            else:
                msample = self.Y.sample_list[idx]
                mask, mask_file = load_img_data(
                    self.Y.dataset_info[msample.fid].path,
                    is_3d=(self.ndim == 3),
                    data_within_zarr_path=msample.get_path_in_zarr() if not self.instance_problem else None,
                )
                if mask_file and isinstance(mask_file, h5py.File):
                    sample_extra_info["mask_file_to_close"] = mask_file

        if not self.test_by_chunks:
            # Skip processing image
            discard = False
            if self.filter_samples:
                foreground_filter_requested = any([True for cond in self.filter_props if "foreground" in cond])
                assert self.filter_vals and self.filter_signs
                discard = sample_satisfy_conds(
                    img,
                    self.filter_props,
                    self.filter_vals,
                    self.filter_signs,
                    mask=mask if foreground_filter_requested else None,
                )
            sample_extra_info["discard"] = discard

            if not discard:
                # Preprocess test data
                if self.preprocess_data:
                    sample_extra_info["rescaled_shape"] = img.shape
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
                        assert mask is not None
                        mask = pad_and_reflect(
                            mask,
                            self.data_shape,
                            verbose=True,
                        )
                    sample_extra_info["reflected_orig_shape"] = reflected_orig_shape

            if not first_load:
                # Normalization
                img = np.array(img)
                self.norm_module.set_stats_from_image(img)
                img, norm_extra_info = self.norm_module.apply_image_norm(img)
                assert isinstance(img, np.ndarray)
                if self.provide_Y:
                    mask = np.array(mask)
                    self.norm_module.set_stats_from_mask( mask, n_classes=self.n_classes,
                        ignore_index=self.ignore_index, instance_problem=self.instance_problem)
                    mask, _ = self.norm_module.apply_mask_norm(mask)
                    assert isinstance(mask, np.ndarray)

                img = np.expand_dims(img, 0)
                if self.provide_Y:
                    mask = np.expand_dims(np.array(mask), 0)
                    if self.norm_module.mask_norm == "as_mask":
                        mask = mask.astype(np.uint8)

            if self.convert_to_rgb:
                if img.shape[-1] == 1:
                    img = np.repeat(img, 3, axis=-1)
                if self.norm_module.mask_norm == "as_image" and mask and mask.shape[-1] == 1:
                    mask = np.repeat(mask, 3, axis=-1)

            # Data channel check
            if self.data_shape[-1] != img.shape[-1]:
                raise ValueError(
                    "Channel of the DATA.PATCH_SIZE given {} does not correspond with the loaded image {}. "
                    "Please, check the channels of the images!".format(self.data_shape[-1], img.shape[-1])
                )
                
        return img, mask, sample, sample_extra_info, norm_extra_info

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
            * ``"X"``, ndarray: X data. It is a ndarrray of  ``(y, x, channels)`` in ``2D`` and ``(z, y, x, channels)`` in ``3D``.
            * ``"X_norm"``, dict: X element normalization steps.
            * ``"Y"``, ndarray: Y data. It is a ndarrray of  ``(y, x, channels)`` in ``2D`` and ``(z, y, x, channels)`` in ``3D``.
            * ``"filename"``, str: name of the image to extract the data sample from.
            * ``"dir"``, str: directory where the image resides.
            * ``"gt_associated_id"``, int (optional): position of associated ground truth of the sample within its list. Present if the
              user selected ``PROBLEM.IMAGE_TO_IMAGE.MULTIPLE_RAW_ONE_TARGET_LOADER`` to be ``True``.
            * ``"discard"``, bool (optional): whether the sample should be discarded or not. Present if ``filter_props``,``filter_vals`` and
            ``filter_signs`` were provided.
            * ``"reflected_orig_shape"``, tuple of int (optional): original shape of the image before reflecting. Present if ``reflect_to_complete_shape``
              is ``True``.
            * ``"img_file_to_close"``, h5py.File (optional): file of the image to close. Present if the loaded file is H5.
            * ``"mask_file_to_close"``, h5py.File (optional): file of the image to close. Present if the loaded file is H5.
        """
        img, mask, sample, sample_extra_info, norm_extra_info = self.load_sample(index)

        if isinstance(img, np.ndarray):
            img.flags.writeable = False

        test_sample = {
            "X": img,
            "X_norm": norm_extra_info,
        }
        if self.provide_Y and mask is not None:
            if isinstance(mask, np.ndarray):
                mask.flags.writeable = False
            test_sample["Y"] = mask

        path = self.X.dataset_info[sample.fid].path
        test_sample["filename"] = os.path.basename(path)
        test_sample["dir"] = os.path.dirname(path)
        test_sample.update(sample_extra_info)

        return test_sample

    def get_data_normalization(self):
        return self.norm_module
