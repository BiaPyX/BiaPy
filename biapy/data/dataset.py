"""
Dataset utilities for organizing input data in BiaPy.

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

- PatchCoords:
    Encapsulates the coordinates of a patch within an image.

Typical usage
-------------

.. code-block:: python

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
    A lightweight container for dataset information used in BiaPy workflows.

    This class stores and manages image-level and sample-level metadata for 
    training, validation, or testing datasets. It encapsulates:
    
    - ``dataset_info``: A list of ``DatasetFile`` instances, each representing a full image file
      along with relevant metadata.
    - ``sample_list``: A list of ``DataSample`` instances, each representing a patch or sample 
      extracted from one of the images in ``dataset_info``.
    """

    def __init__(
        self,
        dataset_info: List[DatasetFile],
        sample_list: List[DataSample],
    ):
        """
        Initialize the BiaPyDataset object.

        Parameters
        ----------
        dataset_info : list of DatasetFile
            Metadata for each file in the dataset.

        sample_list : list of DataSample
            List of samples or patches extracted from the files in `dataset_info`.
        """
        self.dataset_info = dataset_info
        self.sample_list = sample_list

    def clean_dataset(
        self,
        samples_to_maintain: List[int] | NDArray,
        clean_by: str = "image",
    ):
        """
        Remove unwanted samples or images from the dataset.

        This method filters the dataset to retain only a subset of samples or images.
        It also updates internal IDs to remain consistent after filtering.

        Parameters
        ----------
        samples_to_maintain : list of int or ndarray
            Indices of samples or images to retain, depending on `clean_by`.

        clean_by : str, default="image"
            Strategy for filtering the dataset. Must be one of:
            - "sample": `samples_to_maintain` refers to sample indices.
            - "image": `samples_to_maintain` refers to image indices.
        
        Raises
        ------
        AssertionError
            If `clean_by` is not one of ["sample", "image"].
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
        """
        Create a deep copy of the dataset.

        Returns
        -------
        BiaPyDataset
            A deep copy of the current instance.
        """
        return copy.deepcopy(self)

    def __str__(self):
        """
        Return a string representation of the dataset.

        Returns
        -------
        str
            Human-readable summary of the dataset's internal attributes.
        """
        return str(self.__dict__)

    def __repr__(self):
        """
        Return a developer-friendly representation of the dataset.

        Returns
        -------
        str
            Technical string representation (same as __str__).
        """
        return self.__str__()


class DatasetFile:
    """
    A data structure to store metadata and normalization statistics for a single input file.

    This class encapsulates the file path, shape, and optional information required
    for preprocessing and normalization of bioimage data. It is used internally by
    BiaPy to organize and access input data consistently across different workflows.
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
        """
        Initialize the DatasetFile object.

        Parameters
        ----------
        path : str
            Path to the image file.

        shape : tuple of int, optional
            Shape of the image data.

        parallel_data : bool, optional
            Whether the file is in parallelized format (e.g., Zarr or HDF5). Used to
            determine how the data should be read.

        input_axes : str, optional
            String describing the order of axes in the input data (e.g., 'ZYXC').

        lower_bound_val : float, optional
            Lower bound used for intensity clipping during normalization.

        upper_bound_val : float, optional
            Upper bound used for intensity clipping during normalization.

        scale_range_min_val : list of float, optional
            Minimum values for range scaling per channel, used when normalization type
            is "scale_range".

        scale_range_max_val : list of float, optional
            Maximum values for range scaling per channel, used when normalization type
            is "scale_range".

        mean : float, optional
            Mean value used for mean/std normalization (normalization type "div").

        std : float, optional
            Standard deviation used for mean/std normalization (normalization type "div").

        class_num : int, optional
            Class number associated with the sample (used in classification tasks).

        class_name : str, optional
            Human-readable class label for the sample.
        """
        self.path = path
        self.shape = shape
        self.lower_bound_val = lower_bound_val
        self.upper_bound_val = upper_bound_val
        self.scale_range_min_val = scale_range_min_val
        self.scale_range_max_val = scale_range_max_val
        self.mean = mean
        self.std = std
        self.orig_dtype = ""
        if input_axes is not None:
            self.input_axes = input_axes
        if parallel_data is not None:
            self.parallel_data = parallel_data
        if class_num is not None:
            self.class_num = class_num
        if class_name is not None:
            self.class_name = class_name

    def is_parallel(self) -> bool:
        """
        Return whether the dataset file uses a parallel format (e.g., Zarr or H5).

        Returns
        -------
        bool
            True if the file is marked as parallel, False otherwise.
        """
        if hasattr(self, "parallel_data"):
            return self.parallel_data
        else:
            return False

    def get_input_axes(self) -> str | None:
        """
        Return the axes format string of the dataset, if defined.

        Returns
        -------
        str or None
            The input axes string (e.g., 'ZYXC'), or None if not set.
        """
        if hasattr(self, "input_axes"):
            return self.input_axes
        else:
            return None

    def get_class_num(self) -> int:
        """
        Return the class index associated with the dataset file.

        Returns
        -------
        int
            Class number if defined, otherwise -1.
        """
        if hasattr(self, "class_num"):
            return self.class_num
        else:
            return -1

    def __iter__(self):
        """
        Make the class iterable by yielding key-value pairs of instance variables.
        """
        # Iterate over the instance's dictionary of attributes
        for attr_name, attr_value in self.__dict__.items():
            yield attr_name, attr_value

    def __str__(self):
        """
        Return a string representation of the DatasetFile instance.

        Returns
        -------
        str
            Dictionary-style representation of all instance attributes.
        """
        return str(self.__dict__)

    def __repr__(self):
        """
        Return a developer-friendly representation of the DatasetFile.

        Returns
        -------
        str
            String representation for debugging (same as __str__).
        """
        return self.__str__()

    def copy(self):
        """
        Return a deep copy of the DatasetFile object.

        Returns
        -------
        DatasetFile
            Deep-copied instance of the object.
        """
        return copy.deepcopy(self)


class DataSample:
    """
    Represents a single data sample extracted from a larger dataset file.

    A DataSample contains metadata and optionally the image data of a subvolume
    or patch extracted from a parent image. It is primarily used to organize
    and manipulate training, validation, or test samples during deep learning
    workflows in BiaPy.
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
        """
        Initialize a DataSample object with metadata and optional image data.

        Parameters
        ----------
        fid : int
            Index of the file in ``dataset_info`` to which this sample belongs.

        coords : PatchCoords, optional
            Coordinates describing the location of the sample within the full image volume.

        img : NDArray, optional
            The image patch itself, in shape ``(Y, X, C)`` for 2D or ``(Z, Y, X, C)`` for 3D data.
            Only present when data is loaded into memory.

        gt_associated_id : int, optional
            Index of the associated ground truth sample, used when multiple input images share a label.

        input_axes : str, optional
            Axes ordering used in the image data (e.g., 'ZYXC'), particularly relevant for Zarr/H5 files.

        path_in_zarr : str, optional
            Internal path in the Zarr/H5 file where the data is located. Used when multiple datasets
            are stored in a single file.
        """
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
        """
        Check whether the image data has been loaded into memory.

        Returns
        -------
        bool
            True if image data is present in the sample, False otherwise.
        """
        return hasattr(self, "img")

    def get_shape(self) -> Tuple[int, int] | Tuple[int, int, int] | None:
        """
        Get the spatial shape of the sample based on its coordinates.

        Returns
        -------
        tuple of int or None
            Returns a tuple representing the shape of the patch (2D or 3D),
            or None if coordinates are not defined.
        """
        if self.coords is None:
            return None
        else:
            return self.coords.extract_shape_from_coords()

    def get_path_in_zarr(self) -> str | None:
        """
        Get the internal path in the Zarr/H5 file, if available.

        Returns
        -------
        str or None
            Path to the dataset within the file, or None if not set.
        """
        if hasattr(self, "path_in_zarr"):
            return self.path_in_zarr
        else:
            return None

    def get_gt_associated_id(self) -> int | None:
        """
        Get the index of the ground truth sample associated with this input.

        Returns
        -------
        int or None
            Index of the ground truth, or None if not set.
        """
        if hasattr(self, "gt_associated_id"):
            return self.gt_associated_id
        else:
            return None

    def __str__(self):
        """
        Return a string representation of the DataSample instance.

        Returns
        -------
        str
            Dictionary-style representation of all instance attributes.
        """
        return str(self.__dict__)

    def __repr__(self):
        """
        Return a developer-friendly string representation of the DataSample.

        Returns
        -------
        str
            Same as the output of __str__().
        """
        return self.__str__()

    def copy(self):
        """
        Create a deep copy of the DataSample.

        Returns
        -------
        DataSample
            A new deep-copied instance of the current object.
        """
        return copy.deepcopy(self)


class PatchCoords:
    """
    Coordinates of a 2D or 3D patch within an image volume.

    This class stores the spatial boundaries of a patch, allowing BiaPy to
    extract or reference subvolumes from larger datasets. It supports both
    2D (Y, X) and 3D (Z, Y, X) data.
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
        """
        Initialize a PatchCoords object with spatial boundaries.

        Parameters
        ----------
        y_start : int
            Starting index of the patch along the Y axis.
        y_end : int
            Ending index (exclusive) of the patch along the Y axis.
        x_start : int
            Starting index of the patch along the X axis.
        x_end : int
            Ending index (exclusive) of the patch along the X axis.
        z_start : int, optional
            Starting index of the patch along the Z axis. If None, the patch is considered 2D.
        z_end : int, optional
            Ending index (exclusive) of the patch along the Z axis. Required if `z_start` is provided.
        """
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
        Compute the spatial shape of the patch based on its coordinates.

        Returns
        -------
        tuple of int
            A tuple representing the shape of the patch in the order (Z, Y, X) for 3D
            or (Y, X) for 2D, based on the presence of Z-axis coordinates.
        """
        shape = []
        if hasattr(self, "z_start") and hasattr(self, "z_end"):
            shape += [self.z_end - self.z_start]
        shape += [self.y_end - self.y_start]
        shape += [self.x_end - self.x_start]
        return tuple(shape)

    def __str__(self):
        """
        Return a readable string representation of the patch coordinates.

        Returns
        -------
        str
            Formatted string with coordinate ranges in the format:
            "[Z_start:Z_end,Y_start:Y_end,X_start:X_end]" for 3D,
            or "[Y_start:Y_end,X_start:X_end]" for 2D.
        """
        if hasattr(self, "z_start"):
            return "[{}:{},{}:{},{}:{}]".format(
                self.z_start, self.z_end, self.y_start, self.y_end, self.x_start, self.x_end
            )
        else:
            return "[{}:{},{}:{}]".format(self.y_start, self.y_end, self.x_start, self.x_end)

    def __repr__(self):
        """
        Return the official string representation of the PatchCoords object.

        Returns
        -------
        str
            Same as __str__.
        """
        return self.__str__()
        return self.__str__()

    def copy(self):
        """
        Create a deep copy of the PatchCoords object.

        Returns
        -------
        PatchCoords
            A new PatchCoords object with the same coordinate values.
        """
        return copy.deepcopy(self)
