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
    Optional,
)
from numpy.typing import NDArray

from biapy.data.generators.augmentors import center_crop_single, resize_img
from biapy.data.data_manipulation import (
    load_img_data,
    sample_satisfy_conds,
    pad_and_reflect,
)
from biapy.data.data_3D_manipulation import looks_like_hdf5
from biapy.data.dataset import BiaPyDataset, DataSample
from biapy.data.norm import Normalization


class test_single_data_generator(Dataset):
    """
    Image data generator without data augmentation. Used only for test data.

    Parameters
    ----------
    ndim : int
        Dimensions of the data (``2`` for 2D and ``3`` for 3D).

    X : BiaPyDataset
        X data.

    norm_module : Normalization
        Normalization module that defines the normalization steps to apply.

    provide_Y: bool, optional
        Whether the ground truth has been provided or not.

    dims: str, optional
        Dimension of the data. Possible options: ``2D`` or ``3D``.

    seed : int, optional
        Seed for random functions.

    instance_problem : bool, optional
        Not used here.

    crop_center : bool, optional
        Whether to extract a central crop from each image.

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
    """

    def __init__(
        self,
        ndim: int,
        X: BiaPyDataset,
        norm_module: Normalization,
        provide_Y: bool = False,
        seed: int = 42,
        crop_center: bool = False,
        convert_to_rgb: bool = False,
        filter_props: List[List[str]] = [],
        filter_vals: Optional[List[List[int | float]]] = None,
        filter_signs: Optional[List[List[str]]] = None,
        preprocess_data: Optional[Callable] = None,
        preprocess_cfg: Optional[Dict] = None,
        data_shape: Tuple[int, ...] = (256, 256, 1),
        reflect_to_complete_shape: bool = True,
    ):

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
        self.crop_center = crop_center
        # As in test entire images are processed one by one X.sample_list and X.dataset_info must match in length. If not
        # means that validation data is being used as test, so we need to clean the sample_list.
        if len(X.dataset_info) != len(X.sample_list):
            new_sample_list = []
            for i in range(len(X.dataset_info)):
                new_sample_list.append(DataSample(fid=i, coords=None))
            X.sample_list = new_sample_list
        self.len = len(X.sample_list)
        self.seed = seed
        self.ndim = ndim
        self.o_indexes = np.arange(self.len)

        self.norm_module = norm_module
        # Check if a division is required
        img, _, _, sample_extra_info, _ = self.load_sample(0)
        if "img_file_to_close" in sample_extra_info and isinstance(sample_extra_info["img_file_to_close"], h5py.File):
            sample_extra_info["img_file_to_close"].close()
        if "mask_file_to_close" in sample_extra_info and isinstance(sample_extra_info["mask_file_to_close"], h5py.File):
            sample_extra_info["mask_file_to_close"].close()

    # img, img_class, xnorm, filename
    def load_sample(self, idx: int, first_load: bool = False) -> Tuple[NDArray, int, DataSample, Dict, Optional[Dict]]:
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
        img : 4D/5D ndarray array
            Image read. E.g. ``(1, z, y, x, channels)`` for 3D or ``(1, y, x, channels)`` for 2D.
            If by chunks is being used, the shape will be the loaded data as it is.

        img_class: int
            Class of the image. It will be -1 when no class is available.

        sample : DataSample
            Loaded sample.

        sample_extra_info : dict
            Extra information of the loaded sample. Contains the following keys:
            * ``"discard"``, bool (optional): whether the sample should be discarded or not. Present if ``filter_props``,``filter_vals``
              and ``filter_signs`` were provided.
            * ``"reflected_orig_shape"``, tuple of int (optional): original shape of the image before reflecting. Present if ``reflect_to_complete_shape``
              is ``True``.
            * ``"img_file_to_close"``, h5py.File (optional): file of the image to close. Present if the loaded file is H5.

        norm_extra_info : dict
            Normalization extra information useful to undo the normalization after.
        """
        sample = self.X.sample_list[idx].copy()
        sample_extra_info = {}

        img, img_file = load_img_data(
            self.X.dataset_info[sample.fid].path,
            is_3d=(self.ndim == 3),
            data_within_zarr_path=sample.get_path_in_zarr(),
        )

        if looks_like_hdf5(self.X.dataset_info[sample.fid].path) or any(self.X.dataset_info[sample.fid].path.endswith(x) for x in [".zarr", ".n5"]):
            if img_file and isinstance(img_file, h5py.File):
                sample_extra_info["img_file_to_close"] = img_file
        else:
            # Skip processing image
            discard = False
            if self.filter_samples:
                assert self.filter_vals is not None and self.filter_signs is not None
                discard = sample_satisfy_conds(
                    img,
                    self.filter_props,
                    self.filter_vals,
                    self.filter_signs,
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
                    
                # Reflect data to complete the needed shape
                if not self.crop_center and self.reflect_to_complete_shape:
                    reflected_orig_shape = img.shape
                    img = pad_and_reflect(
                        img,
                        self.data_shape,
                        verbose=True,
                    )
                    sample_extra_info["reflected_orig_shape"] = reflected_orig_shape

        img_class = self.X.dataset_info[sample.fid].get_class_num()

        # Normalization
        norm_extra_info = None
        if not first_load:
            self.norm_module.set_stats_from_image(np.array(img))
            img, norm_extra_info = self.norm_module.apply_image_norm(np.array(img))
            assert isinstance(img, np.ndarray)

        if self.convert_to_rgb and img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)

        # Data channel check
        if self.data_shape[-1] != img.shape[-1]:
            raise ValueError(
                "Channel of the DATA.PATCH_SIZE given {} does not correspond with the loaded image {}. "
                "Please, check the channels of the images!".format(self.data_shape[-1], img.shape[-1])
            )
            
        img = np.expand_dims(img, 0)
        return img, img_class, sample, sample_extra_info, norm_extra_info

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
            * ``"X"``, ndarray: X data. It is a ndarray of  ``(y, x, channels)`` in ``2D`` and ``(z, y, x, channels)`` in ``3D``.
            * ``"X_norm"``, dict: X element normalization steps.
            * ``"Y"``, ndarray: Y data. It is a ndarray of  ``(y, x, channels)`` in ``2D`` and ``(z, y, x, channels)`` in ``3D``.
            * ``"X_filename"``: name of the image to extract the data sample from.
            * ``"X_dir"``, str: directory where the image resides.
            * ``"discard"``, bool (optional): whether the sample should be discarded or not. Present if ``filter_props``,``filter_vals``
              and ``filter_signs`` were provided.
            * ``"reflected_orig_shape"``, tuple of int (optional): original shape of the image before reflecting. Present if ``reflect_to_complete_shape``
              is ``True``.
        """
        img, img_class, sample, sample_extra_info, norm_extra_info = self.load_sample(index)

        if self.crop_center and img.shape[:-1] != self.data_shape[:-1]:
            img = center_crop_single(img[0], self.data_shape)
            img = resize_img(img, self.data_shape[:-1])
            img = np.expand_dims(img, 0)

        path = self.X.dataset_info[sample.fid].path
        test_sample = {
            "X": img,
            "X_norm": norm_extra_info,
            "X_filename": os.path.basename(path),
            "X_dir": os.path.dirname(path),
        }
        test_sample.update(sample_extra_info)
        if self.provide_Y:
            test_sample["Y"] = img_class

        return test_sample

    def get_data_normalization(self) -> Normalization:
        return self.norm_module
