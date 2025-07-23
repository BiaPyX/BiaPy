"""
biapy.data.dataset
==================

This module provides the foundational data structures for managing and organizing datasets 
within BiaPy. It includes representations for both individual data files and data samples, 
as well as the overall dataset structure used during training and inference.

Classes
-------

- DatasetFile:
    Represents metadata and statistics associated with an individual input file.
    This includes the file path, size, shape, and any derived properties needed 
    for data handling.

- DataSample:
    Encapsulates a single sample of the dataset, typically representing one training
    or inference instance. It stores indexing information (e.g., crop position, file ID)
    and can also include per-sample weights, masks, or labels.

- BiaPyDataset:
    Main class that manages a full dataset, composed of a list of `DatasetFile` and 
    a list of `DataSample`. Provides methods to clean or filter the dataset, and 
    supports deep copying for safe reuse.

Typical usage
-------------

```python
from biapy.data.dataset import DatasetFile, DataSample, BiaPyDataset

# Assume dataset_info and sample_list are preconstructed lists of DatasetFile and DataSample
dataset = BiaPyDataset(dataset_info=dataset_info, sample_list=sample_list)

# Clean dataset by keeping only a subset of samples or images
dataset.clean_dataset(samples_to_maintain=[0, 2, 5], clean_by="sample")
"""
from __future__ import annotations
from typing import List, Tuple, Optional
from numpy.typing import NDArray
import copy


class BiaPyDataset:
    """
    X data. Contains mainly two dicts:
    * ``dataset_info"``, list of ``DatasetFile``: files that compose the dataset with their respective stats. Each item corresponds to
        a file in the dataset.
    * ``sample_list"``, list of DataSample: each item in the list represents a sample of the dataset.
    """

    def __init__(
        self,
        dataset_info: List[DatasetFile],
        sample_list: List[DataSample],
    ):
        self.dataset_info = dataset_info
        self.sample_list = sample_list

    def clean_dataset(
        self,
        samples_to_maintain: List[int] | NDArray,
        clean_by: str = "image",
    ):
        """
        Clean dataset by only maintaining the samples that are in ``samples_to_maintain``.

        Parameters
        ----------
        samples_to_maintain : list of int
            Id of the samples to maintain.

        clean_by : str
            Whether to clean the dataset by looking the samples, one by one, or looking at images, which will discard
            all the samples that belong to them. In another words, represent what is inside of ``samples_to_maintain``,
            whether the ids of each sample to maintain or the ids of each image to maintain.
        """
        assert clean_by in ["sample", "image"]
        samples_to_maintain.sort()

        if clean_by == "image":
            # Clean "sample_list" first
            new_x_sample_list = []
            for i, x in enumerate(self.sample_list):
                if x.fid in samples_to_maintain:
                    new_x_sample_list.append(x)

            # Then clean "dataset_info"
            new_x_data_info = []
            for i, data_sample in enumerate(self.dataset_info):
                if i in samples_to_maintain:
                    new_x_data_info.append(data_sample)
        else:  # sample
            # Clean "sample_list" first
            new_x_sample_list = []
            for i, x in enumerate(self.sample_list):
                if i in samples_to_maintain:
                    new_x_sample_list.append(x)

            # Then clean "dataset_info"
            new_x_data_info = []
            for i, data_sample in enumerate(self.dataset_info):
                for x in new_x_sample_list:
                    if i == x.fid:
                        new_x_data_info.append(data_sample)
                        break

        # Reorder the file_id
        for i, data_sample in enumerate(new_x_data_info):
            for j in range(len(new_x_sample_list)):
                if self.dataset_info[new_x_sample_list[j].fid].path == data_sample.path:
                    new_x_sample_list[j].fid = i

        self.dataset_info = new_x_data_info
        self.sample_list = new_x_sample_list

    def copy(self):
        return copy.deepcopy(self)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()


class DatasetFile:
    """
    Object that stores a file information and its respective stats.

    Parameters
    ----------
    path : str
        Path of the file.

    shape : tuple of int
        Shape of the image.

    norm_module : Normalization
        Normalization module that defines the normalization steps to apply.

    parallel_data : bool, optional
        To ``True`` is the sample is a Zarr/H5 file. Not present otherwise.

    scale_range_min_val : list of floats, optional
        Minimum values used for normalize the data per each channel. It is created when ``"scale_range"`` normalization type is selected.

    scale_range_max_val : list of floats, optional
        Maximum values used for normalize the data per each channel. It is created when ``"scale_range"`` normalization type is selected.

    mean : float, optional
        Number used to divide during normalization. It is created when ``"div"`` normalization type is selected.

    std : float optional
        Number used to divide during normalization. It is created when ``"div"`` normalization type is selected.
    """

    def __init__(
        self,
        path: str,
        shape: Optional[Tuple] = None,
        parallel_data: Optional[bool] = None,
        input_axes: Optional[str] = None,
        lower_bound_val: Optional[float] = None,
        upper_bound_val: Optional[float] = None,
        scale_range_min_val: Optional[List[float]] = None,
        scale_range_max_val: Optional[List[float]] = None,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        class_num: Optional[int] = None,
        class_name: Optional[str] = None,
    ):
        self.path = path
        self.shape = shape
        self.lower_bound_val = lower_bound_val
        self.upper_bound_val = upper_bound_val
        self.scale_range_min_val = scale_range_min_val
        self.scale_range_max_val = scale_range_max_val
        self.mean = mean
        self.std = std
        if input_axes is not None:
            self.input_axes = input_axes
        if parallel_data is not None:
            self.parallel_data = parallel_data
        if class_num is not None:
            self.class_num = class_num
        if class_name is not None:
            self.class_name = class_name

    def is_parallel(self) -> bool:
        if hasattr(self, "parallel_data"):
            return self.parallel_data
        else:
            return False

    def get_input_axes(self) -> str | None:
        if hasattr(self, "input_axes"):
            return self.input_axes
        else:
            return None

    def get_class_num(self) -> int:
        if hasattr(self, "class_num"):
            return self.class_num
        else:
            return -1

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()

    def copy(self):
        return copy.deepcopy(self)


class DataSample:
    """
    Sample within a file.

    Parameters
    ----------
    fid : int
        id of the file from ``"dataset_info`` that the sample belongs to.

    coords : PatchCoords
        Coordinates to extract the sample from the image.

    img : ndarray (optional)
        Image sample itself. It is of ``(y, x, channels)`` in ``2D`` and ``(z, y, x, channels)`` in ``3D``.
        Provided when ``train_in_memory`` is ``True``.

    gt_associated_id : int, optional
        Position of associated ground truth of the sample within its list. Present if ``multiple_raw_images`` is ``True``.

    input_axes : tuple of int, optional
        Order of the axes in Zarr. Not present in non-Zarr/H5 files.

    path_in_zarr : str, optional
        Path where the data resides within the Zarr. Provided when ``multiple_data_within_zarr`` was set in ``train_zarr_data_information``.

    class_name : str, optional
        Name of the class.

    class : int, optional
        Represents the class (``-1`` if no ground truth provided).
    """

    def __init__(
        self,
        fid: int,
        coords: Optional[PatchCoords],
        img: Optional[NDArray] = None,
        gt_associated_id: Optional[int] = None,
        input_axes: Optional[str] = None,
        path_in_zarr: Optional[str] = None,
    ):
        self.fid = fid
        self.coords = coords
        if img is not None:
            self.img = img
        if gt_associated_id is not None:
            self.gt_associated_id = gt_associated_id
        if input_axes is not None:
            self.input_axes = input_axes
        if path_in_zarr is not None:
            self.path_in_zarr = path_in_zarr

    def img_is_loaded(self):
        return hasattr(self, "img")

    def get_shape(self) -> Tuple[int, int] | Tuple[int, int, int] | None:
        if self.coords is None:
            return None
        else:
            return self.coords.extract_shape_from_coords()

    def get_path_in_zarr(self) -> str | None:
        if hasattr(self, "path_in_zarr"):
            return self.path_in_zarr
        else:
            return None

    def get_gt_associated_id(self) -> int | None:
        if hasattr(self, "gt_associated_id"):
            return self.gt_associated_id
        else:
            return None

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()

    def copy(self):
        return copy.deepcopy(self)


class PatchCoords:
    """
    Coordinates of a patch within an image.

    Parameters
    ----------
    y_start : int
        Starting point of the patch in Y axis.
    y_end : int
        End point of the patch in Y axis.
    x_start : int
        Starting point of the patch in X axis.
    x_end : int
        End point of the patch in X axis.
    z_start : int, optional
        Starting point of the patch in Z axis.
    z_end : int, optional
        End point of the patch in Z axis.
    """

    def __init__(
        self,
        y_start: int,
        y_end: int,
        x_start: int,
        x_end: int,
        z_start: Optional[int] = None,
        z_end: Optional[int] = None,
    ):
        self.y_start = y_start
        self.y_end = y_end
        self.x_start = x_start
        self.x_end = x_end
        if z_start is not None:
            self.z_start = z_start
        if z_end is not None:
            self.z_end = z_end

    def extract_shape_from_coords(self) -> Tuple[int, int] | Tuple[int, int, int]:
        """
        Extract the shape of the patch that represent this coordinates.
        """
        shape = []
        if hasattr(self, "z_start") and hasattr(self, "z_end"):
            shape += [self.z_end - self.z_start]
        shape += [self.y_end - self.y_start]
        shape += [self.x_end - self.x_start]
        return tuple(shape)

    def __str__(self):
        if hasattr(self, "z_start"):
            return "[{}:{},{}:{},{}:{}]".format(
                self.z_start, self.z_end, self.y_start, self.y_end, self.x_start, self.x_end
            )
        else:
            return "[{}:{},{}:{}]".format(self.y_start, self.y_end, self.x_start, self.x_end)

    def __repr__(self):
        return self.__str__()

    def copy(self):
        return copy.deepcopy(self)
