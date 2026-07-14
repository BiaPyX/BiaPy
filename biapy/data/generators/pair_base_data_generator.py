"""
Base data generator for paired image and mask data in BiaPy.

This module provides the PairBaseDataGenerator class, which supports flexible
data loading, augmentation, and normalization for deep learning workflows.
It includes a wide range of augmentation options for both 2D and 3D data,
and is designed to work with BiaPyDataset objects and normalization modules.
"""
from typing import (
    Tuple,
    Literal,
    Dict,
    List,
)
import warnings
import numpy as np
import random
import torch
import os
import h5py
from tqdm import tqdm
from skimage.util import random_noise
from abc import ABCMeta, abstractmethod
from torch.utils.data import Dataset
from numpy.typing import NDArray

from biapy.data.generators.augmentors import *
from biapy.utils.misc import is_main_process, os_walk_clean
from biapy.data.data_manipulation import pad_to_shape, load_img_data, extract_patch_within_image, enlarge_coords
from biapy.data.data_3D_manipulation import extract_patch_from_efficient_file
from biapy.data.dataset import BiaPyDataset
from biapy.data.norm import normalize_image, normalize_mask, update_mask_norm_info


class PairBaseDataGenerator(Dataset, metaclass=ABCMeta):
    """
    Custom BaseDataGenerator to transform paired image and mask data.

    Parameters
    ----------
    ndim : int
        Dimensions of the data (``2`` for ``2D`` and ``3`` for 3D).

    X : BiaPyDataset
        X dataset.

    Y : BiaPyDataset
        Y dataset.

    norm_module : Dict
        Normalization module that defines the normalization steps to apply.

    seed : int, optional
        Seed for random functions.

    da : bool, optional
        To activate the data augmentation.

    aug_prob : dict of str to float, optional
            Per-augmentation probability of being applied, keyed by the augmentation's internal name
            (e.g. ``{"zoom": 0.5, "rand_rot": 0.3, ...}``). Each enabled augmentation is rolled
            independently against its own probability; missing keys default to ``0.5``.

    rotation90 : bool, optional
        To make square (90, 180,270) degree rotations.

    rand_rot : bool, optional
        To make random degree range rotations.

    rnd_rot_range : tuple of float, optional
        Range of random rotations. E. g. ``(-180, 180)``.

    shear : bool, optional
        To make shear transformations.

    shear_range : tuple of int, optional
        Degree range to make shear. E. g. ``(-20, 20)``.

    zoom : bool, optional
        To make zoom on images.

    zoom_range : tuple of floats, optional
        Zoom range to apply. E. g. ``(0.8, 1.2)``.

    zoom_in_z: bool, optional
        Whether to apply or not zoom in Z axis.

    shift : float, optional
        To make shifts.

    shift_range : tuple of float, optional
        Range to make a shift. E. g. ``(0.1, 0.2)``.

    affine_mode: str, optional
        Method to use when filling in newly created pixels. Same meaning as in `skimage` (and `numpy.pad()`).
        E.g. ``constant``, ``reflect`` etc.

    vflip : bool, optional
        To activate vertical flips.

    hflip : bool, optional
        To activate horizontal flips.

    elastic : bool, optional
        To make elastic deformations.

    e_alpha : tuple of ints, optional
        Strength of the distortion field. E. g. ``(240, 250)``.

    e_sigma : int, optional
        Standard deviation of the gaussian kernel used to smooth the distortion fields.

    e_mode : str, optional
        Parameter that defines the handling of newly created pixels with the elastic transformation.

    g_blur : bool, optional
        To insert gaussian blur on the images.

    g_sigma : tuple of floats, optional
        Standard deviation of the gaussian kernel. E. g. ``(1.0, 2.0)``.

    median_blur : bool, optional
        To blur an image by computing median values over neighbourhoods.

    mb_kernel : tuple of ints, optional
        Median blur kernel size. E. g. ``(3, 7)``.

    motion_blur : bool, optional
        Blur images in a way that fakes camera or object movements.

    motb_k_range : int, optional
        Kernel size to use in motion blur.

    gamma_contrast : bool, optional
        To insert gamma constrast changes on images.

    gc_gamma : tuple of floats, optional
        Exponent for the contrast adjustment. Higher values darken the image. E. g. ``(1.25, 1.75)``.

    brightness : bool, optional
        To aply brightness to the images as `PyTorch Connectomics
        <https://github.com/zudi-lin/pytorch_connectomics/blob/master/connectomics/data/augmentation/grayscale.py>`_.

    brightness_factor : tuple of 2 floats, optional
        Strength of the brightness range, with valid values being ``0 <= brightness_factor <= 1``. E.g. ``(0.1, 0.3)``.

    contrast : boolen, optional
        To apply contrast changes to the images as `PyTorch Connectomics
        <https://github.com/zudi-lin/pytorch_connectomics/blob/master/connectomics/data/augmentation/grayscale.py>`_.

    contrast_factor : tuple of 2 floats, optional
        Strength of the contrast change range, with valid values being ``0 <= contrast_factor <= 1``.
        E.g. ``(0.1, 0.3)``.

    dropout : bool, optional
        To set a certain fraction of pixels in images to zero.

    drop_range : tuple of floats, optional
        Range to take a probability ``p`` to drop pixels. E.g. ``(0, 0.2)`` will take a ``p`` folowing ``0<=p<=0.2``
        and then drop ``p`` percent of all pixels in the image (i.e. convert them to black pixels).

    cutout : bool, optional
        To fill one or more rectangular areas in an image using a fill mode.

    cout_nb_iterations : tuple of ints, optional
        Range of number of areas to fill the image with. E. g. ``(1, 3)``.

    cout_size : tuple of floats, optional
        Range to select the size of the areas in % of the corresponding image size. Values between ``0`` and ``1``.
        E. g. ``(0.2, 0.4)``.

    cout_cval : int, optional
        Value to fill the area of cutout with.

    cout_apply_to_mask : boolen, optional
        Whether to apply cutout to the mask.

    cutblur : boolean, optional
        Blur a rectangular area of the image by downsampling and upsampling it again.

    cblur_size : tuple of floats, optional
        Range to select the size of the area to apply cutblur on. E. g. ``(0.2, 0.4)``.

    cblur_inside : boolean, optional
        If ``True`` only the region inside will be modified (cut LR into HR image). If ``False`` the ``50%`` of the
        times the region inside will be modified (cut LR into HR image) and the other ``50%`` the inverse will be
        done (cut HR into LR image). See Figure 1 of the official `paper <https://arxiv.org/pdf/2004.00448.pdf>`__.

    cutmix : boolean, optional
        Combine two images pasting a region of one image to another.

    cmix_size : tuple of floats, optional
        Range to select the size of the area to paste one image into another. E. g. ``(0.2, 0.4)``.

    cnoise : boolean, optional
        Randomly add noise to a cuboid region in the image.

    cnoise_scale : tuple of floats, optional
        Range to choose a value that will represent the % of the maximum value of the image that will be used as the
        std of the Gaussian Noise distribution. E.g. ``(0.1, 0.2)``.

    cnoise_nb_iterations : tuple of ints, optional
        Number of areas with noise to create. E.g. ``(1, 3)``.

    cnoise_size : tuple of floats, optional
        Range to choose the size of the areas to transform. E.g. ``(0.2, 0.4)``.

    misalignment : boolean, optional
        To add miss-aligment augmentation.

    ms_displacement : int, optional
        Maximum pixel displacement in `xy`-plane for misalignment.

    ms_rotate_ratio : float, optional
        Ratio of rotation-based mis-alignment

    missing_sections : boolean, optional
        Augment the image by creating a black line in a random position.

    missp_iterations : tuple of 2 ints, optional
        Iterations to dilate the missing line with. E.g. ``(30, 40)``.

    missp_channel_pb : float, optional
        Probability of applying missing section to each channel. E.g. ``0.5``.

    grayscale : bool, optional
        Whether to augment images converting partially in grayscale.

    gridmask : bool, optional
        Whether to apply gridmask to the image. See the official `paper <https://arxiv.org/abs/2001.04086v1>`__ for
        more information about it and its parameters.

    grid_ratio : float, optional
        Determines the keep ratio of an input image (``r`` in the original paper).

    grid_d_range : tuple of floats, optional
        Range to choose a ``d`` value. It represents the % of the image size. E.g. ``(0.4,1)``.

    grid_rotate : float, optional
        Rotation of the mask in GridMask. Needs to be between ``[0,1]`` where 1 is 360 degrees.

    grid_invert : bool, optional
        Whether to invert the mask of GridMask.

    channel_shuffle : bool, optional
        Whether to shuflle the channels of the images.

    gaussian_noise : bool, optional
        To apply Gaussian noise to the images.

    gaussian_noise_mean : tuple of ints, optional
        Mean of the Gaussian noise.

    gaussian_noise_var : tuple of ints, optional
        Variance of the Gaussian noise.

    gaussian_noise_use_input_img_mean_and_var : bool, optional
        Whether to use the mean and variance of the input image instead of ``gaussian_noise_mean``
        and ``gaussian_noise_var``.

    poisson_noise : bool, optional
        To apply Poisson noise to the images.

    salt : tuple of ints, optional
        Mean of the gaussian noise.

    salt_amount : tuple of ints, optional
        Variance of the gaussian noise.

    pepper : bool, optional
        To apply poisson noise to the images.

    pepper_amount : tuple of ints, optional
        Mean of the gaussian noise.

    salt_and_pepper : bool, optional
        To apply poisson noise to the images.

    salt_pep_amount : tuple of ints, optional
        Mean of the gaussian noise.

    salt_pep_proportion : bool, optional
        To apply poisson noise to the images.

    random_crops_in_DA : bool, optional
        Decide to make random crops in DA (before transformations).

    shape : 3D int tuple, optional
        Shape of the desired images when using 'random_crops_in_DA'.

    resolution : 2D tuple of floats, optional
        Resolution of the given data ``(y,x)``. E.g. ``(8,8)``.

    prob_map : 4D Numpy array or str, optional
        If it is an array, it should represent the probability map used to make random crops when
        ``random_crops_in_DA`` is set. If str given should be the path to read these maps from.

    val : bool, optional
        Advise the generator that the images will be to validate the model to not make random crops (as the val.
        data must be the same on each epoch). Valid when ``random_crops_in_DA`` is set.

    n_classes : int, optional
        Number of classes.

    ignore_index : int, optional
        Value to ignore in the loss/metrics.

    extra_data_factor : int, optional
        Factor to multiply the batches yielded in a epoch. It acts as if ``X`` and ``Y`` where concatenated
        ``extra_data_factor`` times.

    n2v : bool, optional
        Whether to create `Noise2Void <https://openaccess.thecvf.com/content_CVPR_2019/papers/Krull_Noise2Void_-_Learning_Denoising_From_Single_Noisy_Images_CVPR_2019_paper.pdf>`__
        mask. Used in DENOISING problem type.

    n2v_perc_pix : float, optional
        Input image pixels to be manipulated.

    n2v_manipulator : str, optional
        How to manipulate the input pixels. Most pixel manipulators will compute the replacement value based on a neighborhood.
        Possible options: `normal_withoutCP`: samples the neighborhood according to a normal gaussian distribution, but without
        the center pixel; `normal_additive`: adds a random number to the original pixel value. The random number is sampled from
        a gaussian distribution with zero-mean and sigma = `n2v_neighborhood_radius` ; `normal_fitted`: uses a random value from
        a gaussian normal distribution with mean equal to the mean of the neighborhood and standard deviation equal to the
        standard deviation of the neighborhood ; `identity`: performs no pixel manipulation.

    n2v_neighborhood_radius : int, optional
        Neighborhood size to use when manipulating the values.

    n2v_structMask : Array of ints, optional
        Masking kernel for StructN2V to hide pixels adjacent to main blind spot. Value 1 = 'hidden', Value 0 = 'non hidden'.
        Nested lists equivalent to ndarray. Must have odd length in each dimension (center pixel is blind spot). ``None``
        implies normal N2V masking.

    instance_problem : bool, optional
        Advice the class that the workflow is of instance segmentation to divide the labels by channels.

    flow_channels : dict, optional
        Cellpose flow channels present in the mask, mapping each role to its channel index:
        ``{"Gv": i, "Gh": j, "Gz": k}`` (any role may be absent). These channels are direction fields
        (not plain heatmaps): they are tagged ``"flow"`` in the per-channel normalization info and are
        re-oriented (rotated/flipped/rescaled), not just moved, by the augmentation pipeline. Empty
        (default) for non-flow workflows.

    cellpose_diam_mean : float, optional
        Cellpose reference diameter (pixels) the model is trained at (``DIAM_MEAN``). When > 0 and
        ``flow_channels`` is non-empty, each training patch is rescaled in-plane by
        ``DIAM_MEAN / diameter`` (the per-file diameter, read from ``DatasetFile.diameter``) so cells
        become ~``DIAM_MEAN`` pixels. ``0.0`` (default) disables the rescale. The random scale
        augmentation on top of this normalization is provided by the ``zoom`` augmentation
        (``AUGMENTOR.ZOOM`` / ``zoom_range``).

    random_crop_scale : tuple of ints, optional
        Scale factor the mask used in super-resolution workflow. E.g. ``(2,2)``.

    convert_to_rgb : bool, optional
        In case RGB images are expected, e.g. if ``crop_shape`` channel is 3, those images that are grayscale are
        converted into RGB.

    preprocess_f : function, optional
        The preprocessing function, is necessary in case you want to apply any preprocessing.

    preprocess_cfg : dict, optional
        Configuration parameters for preprocessing, is necessary in case you want to apply any preprocessing.

    """

    def __init__(
        self,
        ndim: int,
        X: BiaPyDataset,
        Y: BiaPyDataset,
        norm_module: Dict,
        seed: int = 0,
        da: bool = True,
        aug_prob: Dict[str, float] = {},
        rotation90: bool = False,
        rand_rot: bool = False,
        rnd_rot_range: Tuple[int, int] = (-180, 180),
        shear: bool = False,
        shear_range: Tuple[int, int] = (-20, 20),
        zoom: bool = False,
        zoom_range: Tuple[float, float] = (0.8, 1.2),
        zoom_in_z: bool = False,
        shift: bool = False,
        shift_range: Tuple[float, float] = (0.1, 0.2),
        affine_mode: Literal["constant", "edge", "symmetric", "reflect", "wrap"] = "constant",
        vflip: bool = False,
        hflip: bool = False,
        elastic: bool = False,
        e_alpha: Tuple[int, int] = (240, 250),
        e_sigma: int = 25,
        e_mode: Literal["constant", "edge", "symmetric", "reflect", "wrap"] = "constant",
        g_blur: bool = False,
        g_sigma: Tuple[float, float] = (1.0, 2.0),
        median_blur: bool = False,
        mb_kernel: Tuple[int, int] = (3, 7),
        motion_blur: bool = False,
        motb_k_range: Tuple[int, int] = (3, 8),
        gamma_contrast: bool = False,
        gc_gamma: Tuple[float, float] = (1.25, 1.75),
        brightness: bool = False,
        brightness_factor: Tuple[int, int] = (1, 3),
        brightness_mode: str = "2D",
        contrast: bool = False,
        contrast_factor: Tuple[int, int] = (1, 3),
        contrast_mode: str = "2D",
        dropout: bool = False,
        drop_range: Tuple[float, float] = (0.0, 0.2),
        cutout: bool = False,
        cout_nb_iterations: Tuple[int, int] = (1, 3),
        cout_size: Tuple[float, float] = (0.2, 0.4),
        cout_cval: int = 0,
        cout_apply_to_mask: bool = False,
        cutblur: bool = False,
        cblur_size: Tuple[float, float] = (0.1, 0.5),
        cblur_down_range: Tuple[int, int] = (2, 8),
        cblur_inside: bool = True,
        cutmix: bool = False,
        cmix_size: Tuple[float, float] = (0.2, 0.4),
        cutnoise: bool = False,
        cnoise_scale: Tuple[float, float] = (0.1, 0.2),
        cnoise_nb_iterations: Tuple[int, int] = (1, 3),
        cnoise_size: Tuple[float, float] = (0.2, 0.4),
        misalignment: bool = False,
        ms_displacement: int = 16,
        ms_rotate_ratio: float = 0.0,
        missing_sections: bool = False,
        missp_iterations: Tuple[int, int] = (30, 40),
        missp_channel_pb: float = 0.5,
        grayscale: bool = False,
        channel_shuffle: bool = False,
        gridmask: bool = False,
        grid_ratio: float = 0.6,
        grid_d_range: Tuple[float, float] = (0.4, 1.0),
        grid_rotate: int = 1,
        grid_invert: bool = False,
        gaussian_noise: bool = False,
        gaussian_noise_mean: int = 0,
        gaussian_noise_var: float = 0.01,
        gaussian_noise_use_input_img_mean_and_var: bool = False,
        poisson_noise: bool = False,
        salt: bool = False,
        salt_amount: float = 0.05,
        pepper: bool = False,
        pepper_amount: float = 0.05,
        salt_and_pepper: bool = False,
        salt_pep_amount: float = 0.05,
        salt_pep_proportion: float = 0.5,
        random_crops_in_DA: bool = False,
        shape: Tuple[int, int, int] = (256, 256, 1),
        resolution: Tuple[int, ...] = (-1,),
        prob_map: Optional[NDArray | str] = None,
        val: bool = False,
        n_classes: int = 1,
        ignore_index: Optional[int] = None,
        extra_data_factor: int = 1,
        n2v: bool = False,
        n2v_perc_pix: float = 0.198,
        n2v_manipulator="uniform_withCP",
        n2v_neighborhood_radius: int = 5,
        n2v_structMask=np.array([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]]),
        n2v_load_gt: bool = False,
        instance_problem: bool = False,
        flow_channels: Dict = {},
        cellpose_diam_mean: float = 0.0,
        random_crop_scale: Tuple[int, ...] = (1, 1),
        convert_to_rgb: bool = False,
        preprocess_f=None,
        preprocess_cfg=None,
    ):
        """
        Initialize the PairBaseDataGenerator.

        Sets up data sources, normalization, augmentation options, and preprocessing
        for paired image and mask data.

        Parameters
        ----------
        See class docstring for full parameter list.
        """
        if preprocess_f and preprocess_cfg is None:
            raise ValueError("'preprocess_cfg' needs to be provided with 'preprocess_f'")

        self.ndim = ndim
        self.z_size = -1
        self.val = val
        self.convert_to_rgb = convert_to_rgb
        self.norm_module = norm_module.copy()
        self.random_crops_in_DA = random_crops_in_DA
        self.prob_map = None
        self.preprocess_f = preprocess_f
        self.preprocess_cfg = preprocess_cfg

        self.random_crop_func = random_3D_crop_pair if ndim == 3 else random_crop_pair
        if random_crops_in_DA and prob_map is not None:
            if isinstance(prob_map, str):
                f = next(os_walk_clean(prob_map))[2]
                self.prob_map = []
                for i in range(len(f)):
                    self.prob_map.append(os.path.join(prob_map, f[i]))
            else:
                self.prob_map = prob_map

        # Super-resolution options
        self.random_crop_scale = random_crop_scale

        sshape = X.sample_list[0].get_shape()
        if sshape and len(sshape) != ndim:
            raise ValueError(
                "Samples in X must be have {} dimensions. Provided: {}".format(ndim, X.sample_list[0].get_shape())
            )

        sshape = Y.sample_list[0].get_shape()
        if sshape and len(sshape) != ndim:
            raise ValueError(
                "Samples in Y must be have {} dimensions. Provided: {}".format(ndim, Y.sample_list[0].get_shape())
            )

        if rotation90 and rand_rot:
            warnings.warn("You selected double rotation type. Maybe you should set only 'rand_rot'?")

        self.X = X
        self.Y = Y
        self.length = len(self.X.sample_list)

        self.real_length = self.length
        self.no_bin_channel_found = False

        # Cellpose flow channels are direction fields, not plain heatmaps. ``flow_channels`` maps each
        # role ("Gv"/"Gh"/"Gz") to its channel index within the mask, so they can be tagged in the
        # per-channel normalization info and re-oriented (not just moved) by the augmentation pipeline.
        self.flow_channels = dict(flow_channels)
        # Positions of the flow components inside the split-off ``heat`` array, filled in once the
        # mask normalization info is known ({"vy": i, "vx": j, "vz": k}). Used for the vector-aware
        # rotation/flip/rescale transforms.
        self.flow_heat = {}
        # Per-sample Cellpose diameter rescale: each patch is rescaled in-plane by DIAM_MEAN / diameter
        # (the diameter is read per source file from DatasetFile.diameter in __getitem__) as part of the
        # zoom augmentation. Any random scale jitter on top comes from ``zoom_range``. DIAM_MEAN <= 0
        # disables it.
        self.cellpose_diam_mean = float(cellpose_diam_mean)
        self.do_cellpose_rescale = da and len(self.flow_channels) > 0 and self.cellpose_diam_mean > 0
        self.shape = shape

        # X data analysis
        img, _ = self.load_sample(0, first_load=True)
        if norm_module["type"] in ["div", "scale_range"]:
            if shape[-1] != img.shape[-1]:
                raise ValueError(
                    "Channel of the patch size given {} does not correspond with the loaded image {}. "
                    "Please, check the channels of the images!".format(shape[-1], img.shape[-1])
                )
        self.Y_channels = img.shape[-1]
        xnorm_info = self.X.dataset_info[self.X.sample_list[0].fid].norm_info
        if xnorm_info is None:
            xnorm_info = self.norm_module
        img, xnorm_example = normalize_image(img, norm_module=xnorm_info)
        del img

        # Y data analysis
        # Loop over a few masks to ensure foreground class is present to decide normalization
        self.mask_norm = None
        if self.norm_module["target_type"] == "mask":
            print("Checking which channel of the mask needs normalization . . .")
            n_samples = min(50, len(self.X.sample_list))
            if instance_problem:
                n_samples = 1
            for i in tqdm(range(n_samples), total=n_samples):
                _, mask = self.load_sample(i, first_load=True)
                # Store which channels are binary or not (e.g. distance transform channel is not binary)
                mask, new_mask_norm = normalize_mask(
                    mask, 
                    norm_module=self.norm_module, 
                    n_classes=n_classes, 
                    ignore_index=ignore_index, 
                    instance_problem=instance_problem,
                    apply_norm=False
                )
                if self.mask_norm is None:
                    self.mask_norm = new_mask_norm
                else:
                    self.mask_norm = update_mask_norm_info(self.mask_norm, new_mask_norm)

            # Tag the flow channels (Gv/Gh/Gz) with a dedicated "flow" type so the augmentation
            # pipeline can distinguish direction fields from ordinary non-binary heatmaps: they are
            # interpolated as floats (like "no_bin") but cannot simply be rotated/flipped as scalar
            # maps. They are auto-detected as "no_bin" above; here we override that tag.
            for j in self.flow_channels.values():
                if j in self.mask_norm["per_channel_info"]:
                    self.mask_norm["per_channel_info"][j]["type"] = "flow"
                    self.mask_norm["per_channel_info"][j]["div"] = False

            # Check if any channel is not binary to set no_bin_channel_found to True
            self.no_bin_channel_found = any(
                self.mask_norm["per_channel_info"][j]["type"] in ("no_bin", "flow")
                for j in range(len(self.mask_norm["per_channel_info"]))
            )

            # Map each flow role to its position within the split-off ``heat`` array (the number of
            # non-binary channels appearing before it, since ``heat`` keeps only those). Used by the
            # vector-aware rotation/flip/rescale augmentors.
            if self.no_bin_channel_found and self.flow_channels:
                per = self.mask_norm["per_channel_info"]
                role_to_comp = {"Gv": "vy", "Gh": "vx", "Gz": "vz"}
                for role, midx in self.flow_channels.items():
                    if role not in role_to_comp or midx not in per:
                        continue
                    hpos = sum(1 for k in range(midx) if per[k]["type"] in ("no_bin", "flow"))
                    self.flow_heat[role_to_comp[role]] = hpos
        else:
            self.mask_norm = self.norm_module

        _, mask = self.load_sample(0)
        self.Y_channels = mask.shape[-1]
        self.Y_dtype = mask.dtype
        del mask

        print("Normalization config used for X (first sample): {}".format(xnorm_info))
        print("Normalization config used for Y: {}".format(self.mask_norm))

        if self.ndim == 2:
            resolution = tuple(resolution[i] for i in [1, 0])  # y, x -> x, y
            self.res_relation = (1.0, resolution[0] / resolution[1])
        else:
            resolution = tuple(resolution[i] for i in [2, 1, 0])  # z, y, x -> x, y, z
            self.res_relation = (
                1.0,
                resolution[0] / resolution[1],
                resolution[0] / resolution[2],
            )
        self.resolution = resolution
        self.o_indexes = np.arange(self.length)
        self.n_classes = n_classes
        self.da = da
        self.aug_prob = aug_prob
        self.cutout = cutout
        self.cout_nb_iterations = cout_nb_iterations
        self.cout_size = cout_size
        self.cout_cval = cout_cval
        self.cout_apply_to_mask = cout_apply_to_mask
        self.cutblur = cutblur
        self.cblur_size = cblur_size
        self.cblur_down_range = cblur_down_range
        self.cblur_inside = cblur_inside
        self.cutmix = cutmix
        self.cmix_size = cmix_size
        self.cutnoise = cutnoise
        self.cnoise_scale = cnoise_scale
        self.cnoise_nb_iterations = cnoise_nb_iterations
        self.cnoise_size = cnoise_size
        self.misalignment = misalignment
        self.ms_displacement = ms_displacement
        self.ms_rotate_ratio = ms_rotate_ratio
        self.brightness = brightness
        self.contrast = contrast
        self.missing_sections = missing_sections
        self.missp_iterations = missp_iterations
        self.missp_channel_pb = missp_channel_pb
        self.grayscale = grayscale
        self.gridmask = gridmask
        self.grid_ratio = grid_ratio
        self.grid_d_range = grid_d_range
        self.grid_rotate = grid_rotate
        self.grid_invert = grid_invert
        self.grid_d_size = (
            self.shape[0] * grid_d_range[0],
            self.shape[1] * grid_d_range[1],
        )
        self.channel_shuffle = channel_shuffle
        self.gaussian_noise = gaussian_noise
        self.gaussian_noise_mean = gaussian_noise_mean
        self.gaussian_noise_var = gaussian_noise_var
        self.gaussian_noise_use_input_img_mean_and_var = gaussian_noise_use_input_img_mean_and_var
        self.poisson_noise = poisson_noise
        self.salt = salt
        self.salt_amount = salt_amount
        self.pepper = pepper
        self.pepper_amount = pepper_amount
        self.salt_and_pepper = salt_and_pepper
        self.salt_pep_amount = salt_pep_amount
        self.salt_pep_proportion = salt_pep_proportion
        self.rand_rot = rand_rot
        self.rnd_rot_range = rnd_rot_range
        self.rotation90 = rotation90
        self.affine_mode = affine_mode
        self.zoom = zoom
        self.zoom_range = zoom_range
        self.zoom_in_z = zoom_in_z

        # Extra size to extract so a later zoom-out / rotation samples real image content instead of
        # padding (see geom_aug_load_shape). Computed once from the fixed options; folds in the
        # largest-cell diameter downscale (DIAM_MEAN / max_diameter).
        self.aug_load_inc = tuple([0] * self.ndim)
        if da:
            extra_downscale = 1.0
            if self.do_cellpose_rescale:
                diams = [
                    getattr(f, "diameter", None) for f in getattr(self.Y, "dataset_info", [])
                ]
                diams = [float(d) for d in diams if d and float(d) > 0]
                if diams:
                    extra_downscale = self.cellpose_diam_mean / max(diams)
            self.aug_load_inc = geom_aug_load_shape(
                tuple(self.shape[: self.ndim]),
                self.ndim,
                self.zoom,
                self.zoom_range,
                self.rand_rot,
                zoom_in_z=self.zoom_in_z,
                extra_downscale=extra_downscale,
            )
        # Enlarged extraction size; bounded by the real image, so small images keep the network size.
        self.aug_load_spatial = tuple(
            int(self.shape[i]) + int(self.aug_load_inc[i]) for i in range(self.ndim)
        )

        self.gamma_contrast = gamma_contrast
        self.gc_gamma = gc_gamma

        self.elastic = elastic
        self.shear = shear
        self.shift = shift
        self.vflip = vflip
        self.hflip = hflip
        self.g_blur = g_blur
        self.median_blur = median_blur
        self.motion_blur = motion_blur
        self.dropout = dropout

        self.drop_range = drop_range
        self.e_alpha = e_alpha
        self.e_sigma = e_sigma
        self.e_mode = e_mode
        self.shear_range = shear_range
        self.shift_range = shift_range
        self.affine_mode = affine_mode
        # Flow channels must pad with zeros (background = no flow), never mirrored: reflecting a flow
        # field fabricates border cells with vectors pointing the wrong way.
        if self.flow_channels:
            self.affine_mode = "constant"
        self.g_sigma = g_sigma
        self.mb_kernel = mb_kernel
        self.motb_k_range = motb_k_range 
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor

        # Instance segmentation options
        self.instance_problem = instance_problem

        # Denoising options
        self.n2v = n2v
        self.val = val
        if self.n2v:
            from biapy.engine.denoising import (
                get_stratified_coords2D,
                get_stratified_coords3D,
                get_value_manipulation,
                apply_structN2Vmask,
                apply_structN2Vmask3D,
            )

            self.box_size = int(np.round(np.sqrt(100 / n2v_perc_pix)))
            self.get_stratified_coords = get_stratified_coords2D if self.ndim == 2 else get_stratified_coords3D
            self.value_manipulation = get_value_manipulation(n2v_manipulator, n2v_neighborhood_radius)
            self.n2v_structMask = n2v_structMask
            self.n2v_load_gt = n2v_load_gt
            self.apply_structN2Vmask_func = apply_structN2Vmask if self.ndim == 2 else apply_structN2Vmask3D

        if extra_data_factor > 1:
            self.extra_data_factor = extra_data_factor
            self.o_indexes = np.concatenate([self.o_indexes] * extra_data_factor)
            self.length = self.length * extra_data_factor
        else:
            self.extra_data_factor = 1

        self.da_options = []
        self.trans_made = ""
        if rotation90:
            self.trans_made += "_rot[90,180,270]"
        if rand_rot:
            self.trans_made += "_rrot" + str(rnd_rot_range)
        if shear:
            self.trans_made += "_shear" + str(shear_range)
        if zoom:
            self.trans_made += "_zoom" + str(zoom_range) + "+" + str(zoom_in_z)
        if shift:
            self.trans_made += "_shift" + str(shift_range)
        if vflip:
            self.trans_made += "_vflip"
        if hflip:
            self.trans_made += "_hflip"
        if elastic:
            self.trans_made += "_elastic" + str(e_alpha) + "+" + str(e_sigma) + "+" + str(e_mode)
        if g_blur:
            self.trans_made += "_gblur" + str(g_sigma)
        if median_blur:
            self.trans_made += "_mblur" + str(mb_kernel)
        if motion_blur:
            self.trans_made += "_motb" + str(motb_k_range)
        if gamma_contrast:
            self.trans_made += "_gcontrast" + str(gc_gamma)
        if brightness:
            self.trans_made += "_brightness" + str(brightness_factor)
        if contrast:
            self.trans_made += "_contrast" + str(contrast_factor)
        if dropout:
            self.trans_made += "_drop" + str(drop_range)

        if grayscale:
            self.trans_made += "_gray"
        if gridmask:
            self.trans_made += (
                "_gridmask"
                + str(self.grid_ratio)
                + "+"
                + str(self.grid_d_range)
                + "+"
                + str(self.grid_rotate)
                + "+"
                + str(self.grid_invert)
            )
        if channel_shuffle:
            self.trans_made += "_chshuffle"
        if cutout:
            self.trans_made += (
                "_cout"
                + str(cout_nb_iterations)
                + "+"
                + str(cout_size)
                + "+"
                + str(cout_cval)
                + "+"
                + str(cout_apply_to_mask)
            )
        if cutblur:
            self.trans_made += "_cblur" + str(cblur_size) + "+" + str(cblur_down_range) + "+" + str(cblur_inside)
        if cutmix:
            self.trans_made += "_cmix" + str(cmix_size)
        if cutnoise:
            self.trans_made += "_cnoi" + str(cnoise_scale) + "+" + str(cnoise_nb_iterations) + "+" + str(cnoise_size)
        if misalignment:
            self.trans_made += "_msalg" + str(ms_displacement) + "+" + str(ms_rotate_ratio)
        if missing_sections:
            self.trans_made += "_missp" + "+" + str(missp_iterations)
        if gaussian_noise:
            self.trans_made += "_gausnoise" + "+" + str(gaussian_noise_mean) + "+" + str(gaussian_noise_var)
        if poisson_noise:
            self.trans_made += "_poisnoise"
        if salt:
            self.trans_made += "_salt" + "+" + str(salt_amount)
        if pepper:
            self.trans_made += "_pepper" + "+" + str(pepper_amount)
        if salt_and_pepper:
            self.trans_made += "_salt_and_pepper" + "+" + str(salt_pep_amount) + "+" + str(salt_pep_proportion)

        self.trans_made = self.trans_made.replace(" ", "")
        self.seed = seed
        random.seed(seed)

        self.indexes = self.o_indexes.copy()

    @abstractmethod
    def save_aug_samples(
        self,
        img: NDArray,
        mask: NDArray,
        orig_images: Dict,
        i: int,
        pos: int,
        out_dir: str,
        point_dict: Dict,
    ):
        """
        Save transformed samples in order to check the generator.

        Parameters
        ----------
        img : 3D/4D Numpy array
            Image to use as sample. E.g. ``(y, x, channels)`` for ``2D`` and ``(z, y, x, channels)`` for ``3D``.

        mask : 3D/4D Numpy array
            Mask to use as sample. E.g. ``(y, x, channels)`` for ``2D`` and ``(z, y, x, channels)`` for ``3D``.

        orig_images : dict
            Dict where the original image and mask are saved in "o_x" and "o_y", respectively.

        i : int
            Number of the sample within the transformed ones.

        pos : int
            Number of the sample within the dataset.

        out_dir : str
            Directory to save the images.

        point_dict : Dict
            Necessary info to draw the patch extracted within the original image. It has ``ox`` and
            ``oy`` representing the ``x`` and ``y`` coordinates of the central point selected during
            the crop extraction, and ``s_x`` and ``s_y`` as the ``(0,0)`` coordinates of the extracted
            patch. For ``3D`` samples it must contain also ``oz`` and ``s_z``.
        """
        raise NotImplementedError

    def __len__(self):
        """Define the number of samples per epoch."""
        return self.length

    def load_sample(
        self, _idx: int, first_load: bool = False, geom_enlarge: bool = False
    ) -> Tuple[NDArray, NDArray]:
        """
        Load one data sample given its corresponding index.

        Parameters
        ----------
        _idx : int
            Sample index counter.

        first_load : bool, optional
            Whether its the first time a sample is loaded to prevent normalizing it.

        geom_enlarge : bool, optional
            Whether to extract a patch larger than the network input so a later zoom-out / rotation
            keeps real image content (see ``aug_load_inc``). Off by default; enable it only where
            ``apply_transform`` warps the patch and crops it back to the network size. Extraction stays
            bounded by the real image.

        Returns
        -------
        img : 3D/4D Numpy array
            X element. E.g. ``(y, x, channels)`` in ``2D`` and ``(z, y, x, channels)`` in ``3D``.

        mask : 3D/4D Numpy array
            Y element. E.g. ``(y, x, channels)`` in ``2D`` and ``(z, y, x, channels)`` in ``3D``.
        """
        idx = _idx % self.real_length
        sample = self.X.sample_list[idx]

        # Extract a bigger patch when a later geometric augmentation could otherwise pull in padding.
        enlarge = geom_enlarge and self.da and (not first_load) and any(self.aug_load_inc)

        # X data
        if sample.img_is_loaded():
            img = sample.img.copy()
        else:
            img, img_file = load_img_data(
                self.X.dataset_info[sample.fid].path,
                is_3d=(self.ndim == 3),
                data_within_zarr_path=sample.get_path_in_zarr(),
            )

            if not self.X.dataset_info[sample.fid].is_parallel():
                # Apply preprocessing
                if self.preprocess_f:
                    img = self.preprocess_f(self.preprocess_cfg, x_data=[img], is_2d=(self.ndim == 2))[0]

                img = pad_to_shape(img, self.shape, verbose=False, mode=self.affine_mode)

                # Extract the sample within the image
                if sample.coords:
                    coords = (
                        enlarge_coords(sample.coords, self.aug_load_inc, img.shape, is_3d=(self.ndim == 3))
                        if enlarge
                        else sample.coords
                    )
                    img = extract_patch_within_image(img, coords, is_3d=(self.ndim == 3))
            else:
                coords = sample.coords
                data_axes_order = self.X.dataset_info[sample.fid].get_input_axes()
                assert coords is not None and data_axes_order is not None
                if enlarge:
                    # Lazy Zarr/H5 reads clamp over-range stops, so a large sentinel extent is safe.
                    coords = enlarge_coords(coords, self.aug_load_inc, (1 << 62,) * 3, is_3d=(self.ndim == 3))
                img = extract_patch_from_efficient_file(img, coords, data_axes_order=data_axes_order)

                # Apply preprocessing after extract sample
                if self.preprocess_f:
                    img = self.preprocess_f(self.preprocess_cfg, x_data=[img], is_2d=(self.ndim == 2))[0]

                if isinstance(img_file, h5py.File):
                    img_file.close()

        # Y data
        # "gt_associated_id" available only in PROBLEM.IMAGE_TO_IMAGE.MULTIPLE_RAW_ONE_TARGET_LOADER
        gt_id = sample.get_gt_associated_id()
        if gt_id is not None:
            msample = self.Y.sample_list[gt_id]
            mask, _ = load_img_data(
                self.Y.dataset_info[msample.fid].path,
                is_3d=(self.ndim == 3),
            )
            # Extract the sample within the image
            if msample.coords:
                coords = (
                    enlarge_coords(msample.coords, self.aug_load_inc, mask.shape, is_3d=(self.ndim == 3))
                    if enlarge
                    else msample.coords
                )
                mask = extract_patch_within_image(mask, coords, is_3d=(self.ndim == 3))
        else:
            msample = self.Y.sample_list[idx]
            if msample.img_is_loaded():
                mask = msample.img.copy()
            else:
                fid = msample.fid
                mask, mask_file = load_img_data(
                    self.Y.dataset_info[fid].path,
                    is_3d=(self.ndim == 3),
                    data_within_zarr_path=msample.get_path_in_zarr(),
                )

                if not self.Y.dataset_info[msample.fid].is_parallel():
                    # Apply preprocessing
                    if self.preprocess_f:
                        mask = self.preprocess_f(
                            self.preprocess_cfg, y_data=[mask], is_2d=(self.ndim == 2), is_y_mask=True
                        )[0]

                    mask = pad_to_shape(mask, self.shape, verbose=False, mode=self.affine_mode)

                    # Extract the sample within the image
                    if msample.coords:
                        coords = msample.coords
                        assert coords is not None
                        if enlarge:
                            coords = enlarge_coords(coords, self.aug_load_inc, mask.shape, is_3d=(self.ndim == 3))
                        mask = extract_patch_within_image(mask, coords, is_3d=(self.ndim == 3))
                else:
                    coords = msample.coords
                    data_axes_order = self.Y.dataset_info[msample.fid].get_input_axes()
                    assert coords is not None and data_axes_order is not None
                    if enlarge:
                        # Lazy Zarr/H5 reads clamp over-range stops, so a large sentinel extent is safe.
                        coords = enlarge_coords(coords, self.aug_load_inc, (1 << 62,) * 3, is_3d=(self.ndim == 3))
                    mask = extract_patch_from_efficient_file(
                        mask,
                        coords,
                        data_axes_order=data_axes_order,
                    )

                    # Apply preprocessing after extract sample
                    if self.preprocess_f:
                        mask = self.preprocess_f(
                            self.preprocess_cfg, y_data=[mask], is_2d=(self.ndim == 2), is_y_mask=True
                        )[0]

                    if self.Y.dataset_info[msample.fid].is_parallel():
                        if mask_file and isinstance(mask_file, h5py.File):
                            mask_file.close()

        # Apply random crops if it is selected
        if sample.coords is None:
            # Capture probability map
            if self.prob_map is not None:
                if isinstance(self.prob_map, list):
                    img_prob = np.load(self.prob_map[idx])
                else:
                    img_prob = self.prob_map[idx]
            else:
                img_prob = None

            # Crop a bigger window when enlarging; bounded by the image, apply_transform crops it back.
            crop_spatial = self.aug_load_spatial if enlarge else self.shape[: self.ndim]
            img, mask = self.random_crop_func(  # type: ignore
                img,
                mask,
                crop_spatial,
                self.val,
                img_prob=img_prob,
                scale=self.random_crop_scale,
            )

        if not first_load:
            xnorm_info = self.X.dataset_info[sample.fid].norm_info
            if xnorm_info is None:
                xnorm_info = self.norm_module
            img, _ = normalize_image(img, norm_module=xnorm_info)
            ynorm = self.Y.dataset_info[sample.fid].norm_info if self.norm_module["norm_target"] else self.mask_norm
            mask, _ = normalize_mask(mask, norm_module=ynorm)
            assert isinstance(img, np.ndarray) and isinstance(mask, np.ndarray)

        if self.convert_to_rgb:
            if img.shape[-1] == 1:
                img = np.repeat(img, 3, axis=-1)
            if self.norm_module["target_type"] == "image" and mask.shape[-1] == 1:
                mask = np.repeat(mask, 3, axis=-1)

        return img, mask

    def getitem(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate one pair of data.

        Parameters
        ----------
        index : int
            Index counter.

        Returns
        -------
        item : 3D/4D Torch tensors
            X and Y (if avail) elements. Each one shape is ``(z, y, x, channels)`` if ``2D`` or ``(y, x, channels)``
            if ``3D``.
        """
        return self.__getitem__(index)

    def cellpose_diam_factor(self, index: int) -> float:
        """
        Compute the per-sample diameter normalization factor for the sample at ``index``.

        The factor is ``DIAM_MEAN / diameter``, with ``diameter`` read from ``DatasetFile.diameter``.
        Returns ``1.0`` when rescaling is disabled or the diameter is unknown.

        Parameters
        ----------
        index : int
            Sample index.

        Returns
        -------
        float
            In-plane diameter normalization factor to feed to :func:`apply_transform`.
        """
        if not self.do_cellpose_rescale:
            return 1.0
        idx = index % self.real_length
        msample = self.Y.sample_list[idx]
        diameter = getattr(self.Y.dataset_info[msample.fid], "diameter", None)
        if diameter is None or diameter <= 0:
            return 1.0
        return self.cellpose_diam_mean / float(diameter)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate one pair of data.

        Parameters
        ----------
        index : int
            Index counter.

        Returns
        -------
        img : 3D/4D Torch tensor
            X element, for instance, an image. E.g. ``(y, x, channels)`` in ``2D`` and ``(z, y, x, channels)`` in ``3D``.

        mask : 3D/4D Torch tensor
            Y element, for instance, a mask. E.g. ``(y, x, channels)`` in ``2D`` and ``(z, y, x, channels)`` in ``3D``.
        """
        # Enlarge the extraction here; apply_transform warps and crops it back to the network size.
        img, mask = self.load_sample(index, geom_enlarge=True)

        # Apply transformations
        if self.da:
            e_img, e_mask = None, None
            if self.cutmix:
                extra_img = np.random.randint(0, self.length - 1) if self.length > 2 else 0
                # The cutmix donor is not warped, so it stays at the network size (geom_enlarge off).
                e_img, e_mask = self.load_sample(extra_img)

            img, mask = self.apply_transform(
                img, mask, e_im=e_img, e_mask=e_mask,
                diam_factor=self.cellpose_diam_factor(index),
            )

        # Prepare mask when denoising with Noise2Void
        if self.n2v:
            img, mask = self.prepare_n2v(img, mask)

        # If no normalization was applied, as is done with torchvision models, it can be an image of uint16
        # so we need to convert it to
        if img.dtype == np.uint16:
            img = torch.from_numpy(img.astype(np.float32))
        else:
            img = torch.from_numpy(img.copy())
        mask = torch.from_numpy(mask.copy())

        return img, mask

    def apply_transform(
        self,
        image: NDArray,
        mask: NDArray,
        e_im: Optional[NDArray],
        e_mask: Optional[NDArray],
        diam_factor: float = 1.0,
    ) -> Tuple[NDArray, NDArray]:
        """
        Transform the input image and its mask at the same time with one of the selected choices based on a probability.

        Parameters
        ----------
        image : 3D/4D Numpy array
            Image to transform. E.g. ``(y, x, channels)`` in ``2D`` and ``(y, x, z, channels)`` in ``3D``.

        mask : 3D/4D Numpy array
            Mask to transform. E.g. ``(y, x, channels)`` in ``2D`` and ``(y, x, z, channels)`` in ``3D``.

        e_im : 3D/4D Numpy array
            Extra image to help transforming ``image``. E.g. ``(y, x, channels)`` in ``2D`` or
            ``(y, x, z, channels)`` in ``3D``.

        e_mask : 3D/4D Numpy array
            Extra mask to help transforming ``mask``. E.g. ``(y, x, channels)`` in ``2D`` or
            ``(y, x, z, channels)`` in ``3D``.

        diam_factor : float, optional
            Per-sample in-plane diameter normalization factor (``DIAM_MEAN / diameter``) folded into
            the zoom so cells reach ~``DIAM_MEAN`` pixels. ``1.0`` (default) disables it.

        Returns
        -------
        image : 3D/4D Numpy array
            Transformed image. E.g. ``(y, x, channels)`` in ``2D`` or ``(y, x, z, channels)`` in ``3D``.

        mask : 3D/4D Numpy array
            Transformed image mask. E.g. ``(y, x, channels)`` in ``2D`` or ``(y, x, z, channels)`` in ``3D``.
        """
        # Split heatmaps from masks
        heat = None
        if self.no_bin_channel_found:
            heat = []
            new_mask = []
            if self.cutmix:
                e_new_mask = []
                e_heat = []
            for j in range(mask.shape[-1]):
                if self.mask_norm["per_channel_info"][j]["type"] in ("no_bin", "flow"):
                    heat.append(np.expand_dims(mask[..., j], -1))
                    if self.cutmix:
                        e_heat.append(np.expand_dims(e_mask[..., j], -1)) # type: ignore
                else:
                    new_mask.append(np.expand_dims(mask[..., j], -1))
                    if self.cutmix:
                        e_new_mask.append(np.expand_dims(e_mask[..., j], -1)) # type: ignore

            heat = np.concatenate(heat, axis=-1)
            if self.cutmix:
                e_heat = np.concatenate(e_heat, axis=-1)
            if len(new_mask) == 0:
                mask = np.zeros(mask.shape[:-1] + (1,))  # Fake mask
                if self.cutmix:
                    e_mask = np.zeros(e_mask.shape[:-1] + (1,))  # type: ignore
            else:
                mask = np.concatenate(new_mask, axis=-1)
                if self.cutmix:
                    e_mask = np.concatenate(e_new_mask, axis=-1)  # type: ignore
            del new_mask
            if self.cutmix:
                del e_new_mask

        # Scale and rotation are rolled independently, then composed into a single warp so the image,
        # mask and flow field are interpolated only once (see affine_transform). The diameter
        # normalization (diam_factor = DIAM_MEAN / diameter, in-plane) is applied whenever needed, even
        # when no augmentation roll fires.
        do_diam_rescale = self.do_cellpose_rescale and diam_factor > 0 and abs(diam_factor - 1.0) > 1e-3
        apply_zoom = self.zoom and random.uniform(0, 1) < self.aug_prob.get("zoom", 0.5)
        apply_rand_rot = self.rand_rot and random.uniform(0, 1) < self.aug_prob.get("rand_rot", 0.5)
        apply_rot90 = self.rotation90 and random.uniform(0, 1) < self.aug_prob.get("rotation90", 0.5)
        if apply_zoom or apply_rand_rot or apply_rot90 or do_diam_rescale:
            scale = random.uniform(self.zoom_range[0], self.zoom_range[1]) if apply_zoom else 1.0
            angle = 0.0
            if apply_rand_rot:
                angle += random.uniform(float(self.rnd_rot_range[0]), float(self.rnd_rot_range[1]))
            if apply_rot90:
                angle += random.choice([90.0, 180.0, 270.0])
            image, mask, heat = affine_transform(
                image,
                mask=mask,
                heat=heat,
                scale_xy=scale * diam_factor,
                scale_z=scale if self.zoom_in_z else 1.0,
                angle=angle,
                mode=self.affine_mode,
                mask_type=self.norm_module["target_type"],
                flow_heat=self.flow_heat,
            )  # type: ignore

        # Crop the (possibly enlarged) patch back to the network size right after the warp, so later
        # augmentations run at self.shape. Random crop, padding with self.affine_mode if a border fell
        # short.
        target = tuple(self.shape[: self.ndim])
        if tuple(image.shape[: self.ndim]) != target:
            image = pad_to_shape(image, target + (image.shape[-1],), mode=self.affine_mode)
            if heat is not None:
                nmask = mask.shape[-1]
                merged = np.concatenate([mask, heat], axis=-1)
                merged = pad_to_shape(merged, target + (merged.shape[-1],), mode=self.affine_mode)
                image, merged = self.random_crop_func(image, merged, target, self.val)  # type: ignore
                mask, heat = merged[..., :nmask], merged[..., nmask:]
            else:
                mask = pad_to_shape(mask, target + (mask.shape[-1],), mode=self.affine_mode)
                image, mask = self.random_crop_func(image, mask, target, self.val)  # type: ignore

        # Convert to grayscale
        if self.grayscale and random.uniform(0, 1) < self.aug_prob.get("grayscale", 0.5):
            image = grayscale(image)

        # Apply channel shuffle
        if self.channel_shuffle and random.uniform(0, 1) < self.aug_prob.get("channel_shuffle", 0.5):
            image = shuffle_channels(image)

        # Apply cblur
        if self.cutblur and random.uniform(0, 1) < self.aug_prob.get("cutblur", 0.5):
            image = cutblur(image, self.cblur_size, self.cblur_down_range, self.cblur_inside)

        # Apply cutmix
        if self.cutmix and random.uniform(0, 1) < self.aug_prob.get("cutmix", 0.5):
            image, mask, heat = cutmix(image, e_im, mask, e_mask, heat, e_heat, self.cmix_size) # type: ignore

        # Apply cutnoise
        if self.cutnoise and random.uniform(0, 1) < self.aug_prob.get("cutnoise", 0.5):
            image = cutnoise(image, self.cnoise_scale, self.cnoise_nb_iterations, self.cnoise_size)

        # Misalignment: threads the flow channels through the same ops and re-orients their vectors.
        if self.misalignment and random.uniform(0, 1) < self.aug_prob.get("misalignment", 0.5):
            image, mask, heat = misalignment(
                image, mask, self.ms_displacement, self.ms_rotate_ratio,
                heat=heat, flow_heat=self.flow_heat,
            )

        # Apply brightness
        if self.brightness and random.uniform(0, 1) < self.aug_prob.get("brightness", 0.5):
            image = brightness(
                image,
                brightness_factor=self.brightness_factor,
            )

        # Apply contrast
        if self.contrast and random.uniform(0, 1) < self.aug_prob.get("contrast", 0.5):
            image = contrast(image, contrast_factor=self.contrast_factor)

        # Apply gamma contrast
        if self.gamma_contrast and random.uniform(0, 1) < self.aug_prob.get("gamma_contrast", 0.5):
            image = gamma_contrast(image, gamma=self.gc_gamma)

        # Apply gaussian noise
        if self.gaussian_noise and random.uniform(0, 1) < self.aug_prob.get("gaussian_noise", 0.5):
            mean = np.mean(image) if self.gaussian_noise_use_input_img_mean_and_var else self.gaussian_noise_mean
            var = (
                np.var(image) * random.uniform(0.9, 1.1)
                if self.gaussian_noise_use_input_img_mean_and_var
                else self.gaussian_noise_var
            )
            image = random_noise(image, mode="gaussian", mean=mean, var=var)

        # Apply poisson noise
        if self.poisson_noise and random.uniform(0, 1) < self.aug_prob.get("poisson_noise", 0.5):
            image = random_noise(image, mode="poisson")

        # Apply salt noise
        if self.salt and random.uniform(0, 1) < self.aug_prob.get("salt", 0.5):
            image = random_noise(image, mode="salt", amount=self.salt_amount)

        # Apply pepper noise
        if self.pepper and random.uniform(0, 1) < self.aug_prob.get("pepper", 0.5):
            image = random_noise(image, mode="pepper", amount=self.pepper_amount)

        # Apply salt & pepper noise
        if self.salt_and_pepper and random.uniform(0, 1) < self.aug_prob.get("salt_and_pepper", 0.5):
            image = random_noise(
                image,
                mode="s&p",
                amount=self.salt_pep_amount,
                salt_vs_pepper=self.salt_pep_proportion,
            )

        # Apply missing parts
        if self.missing_sections and random.uniform(0, 1) < self.aug_prob.get("missing_sections", 0.5):
            image = missing_sections(image, self.missp_iterations, self.missp_channel_pb)

        # Apply GridMask
        if self.gridmask and random.uniform(0, 1) < self.aug_prob.get("gridmask", 0.5):
            image = GridMask(
                image,
                self.z_size,
                self.grid_ratio,
                self.grid_d_size,
                self.grid_rotate,
                self.grid_invert,
            )

        # Cutout only blanks regions (no pixel movement), so it is flow-safe: passing ``heat`` blanks
        # the flow targets wherever the mask is blanked.
        if self.cutout and random.uniform(0, 1) < self.aug_prob.get("cutout", 0.5):
            image, mask, heat = cutout(
                image,
                mask,
                self.z_size,
                self.cout_nb_iterations,
                self.cout_size,
                self.cout_cval,
                self.res_relation,
                self.cout_apply_to_mask,
                heat=heat,
            )

        # Elastic is skipped for flow channels: this non-rigid warp would need the flow vectors
        # re-oriented by its local Jacobian, which is not done here, corrupting the targets.
        if self.elastic and not self.flow_channels and random.uniform(0, 1) < self.aug_prob.get("elastic", 0.5):
            image, mask, heat = elastic(
                image,
                mask=mask,
                heat=heat,
                alpha=self.e_alpha,
                sigma=self.e_sigma,
                mask_type=self.norm_module["target_type"],
                mode=self.e_mode
            ) # type: ignore

        # Shear is skipped for flow channels: it skews directions and the vectors are not re-oriented.
        if self.shear and not self.flow_channels and random.uniform(0, 1) < self.aug_prob.get("shear", 0.5):
            image, mask, heat = shear(
                image, mask=mask, heat=heat,
                shear=self.shear_range,
                mode=self.affine_mode,
                mask_type=self.norm_module["target_type"],
            ) # type: ignore
        
        if self.shift and random.uniform(0, 1) < self.aug_prob.get("shift", 0.5):
            image, mask, heat = shift(
                image, mask=mask, heat=heat,
                shift_range=self.shift_range,
                mode=self.affine_mode,
                mask_type=self.norm_module["target_type"],
            ) # type: ignore

        if self.vflip and random.uniform(0, 1) < self.aug_prob.get("vflip", 0.5):
            image, mask, heat = flip_vertical(
                image, mask=mask, heat=heat, flow_heat=self.flow_heat
            ) # type: ignore

        if self.hflip and random.uniform(0, 1) < self.aug_prob.get("hflip", 0.5):
            image, mask, heat = flip_horizontal(
                image, mask=mask, heat=heat, flow_heat=self.flow_heat
            ) # type: ignore
            
        if self.g_blur and random.uniform(0, 1) < self.aug_prob.get("g_blur", 0.5):
            image = gaussian_blur(
                image,
                sigma=self.g_sigma
            )

        if self.median_blur and random.uniform(0, 1) < self.aug_prob.get("median_blur", 0.5):
            image = median_blur(
                image,
                k_range=self.mb_kernel
            )

        if self.motion_blur and random.uniform(0, 1) < self.aug_prob.get("motion_blur", 0.5):
            image = motion_blur(
                image,
                k_range=self.motb_k_range
            )

        if self.dropout and random.uniform(0, 1) < self.aug_prob.get("dropout", 0.5):
            image = dropout(
                image,
                drop_range=self.drop_range
            )

        # Merge heatmaps and masks again
        if self.no_bin_channel_found:
            new_mask = []
            hi, mi = 0, 0
            for j in range(len(self.mask_norm["per_channel_info"])):
                if self.mask_norm["per_channel_info"][j]["type"] in ("no_bin", "flow"):
                    new_mask.append(np.expand_dims(heat[..., hi], -1))
                    hi += 1
                else:
                    new_mask.append(np.expand_dims(mask[..., mi], -1))
                    mi += 1
            mask = np.concatenate(new_mask, axis=-1)

        return image, mask

    def get_transformed_samples(
        self,
        num_examples: int,
        random_images: bool = True,
        save_to_dir: bool = True,
        out_dir: str = "aug",
        train: bool = False,
        draw_grid: bool = True,
    ):
        """
        Apply selected transformations to a defined number of images from the dataset.

        Parameters
        ----------
        num_examples : int
            Number of examples to generate.

        random_images : bool, optional
            Randomly select images from the dataset. If ``False`` the examples will be generated from the start of
            the dataset.

        save_to_dir : bool, optional
            Save the images generated. The purpose of this variable is to check the images generated by data
            augmentation.

        out_dir : str, optional
            Name of the folder where the examples will be stored.

        train : bool, optional
            To avoid drawing a grid on the generated images. This should be set when the samples will be used for
            training.

        draw_grid : bool, optional
            Draw a grid in the generated samples. Useful to see some types of deformations.

        Returns
        -------
        sample_x : List of 3D/4D Numpy array
            Transformed images. E.g. list of ``(y, x, channels)`` in ``2D`` and ``(z, y, x, channels)`` in ``3D``.

        sample_y : List of 3D/4D Numpy array
            Transformed image mask. E.g. list of ``(y, x, channels)`` in ``2D`` and ``(z, y, x, channels)`` in ``3D``.

        Examples
        --------

        Setting ``elastic=True`` an example output should be similar to the following:

        +----------------------------------------------------+----------------------------------------------------+
        | .. figure:: ../../../img/original_crop_2d.png      | .. figure:: ../../../img/original_crop_mask_2d.png |
        |   :width: 80%                                      |   :width: 80%                                      |
        |   :align: center                                   |   :align: center                                   |
        |                                                    |                                                    |
        |   Original crop                                    |   Original crop mask                               |
        +----------------------------------------------------+----------------------------------------------------+
        | .. figure:: ../../../img/elastic_crop_2d.png       | .. figure:: ../../../img/elastic_crop_mask_2d.png  |
        |   :width: 80%                                      |   :width: 80%                                      |
        |   :align: center                                   |   :align: center                                   |
        |                                                    |                                                    |
        |   Elastic transformation applied                   |   Elastic transformation applied                   |
        +----------------------------------------------------+----------------------------------------------------+

        The grid is only painted if ``train=False`` which should be used just to display transformations made.
        Selecting random rotations between 0 and 180 degrees should generate the following:

        +-----------------------------------------------------------+-----------------------------------------------------------+
        | .. figure:: ../../../img/original_rd_rot_crop_2d.png      | .. figure:: ../../../img/original_rd_rot_crop_mask_2d.png |
        |   :width: 80%                                             |   :width: 80%                                             |
        |   :align: center                                          |   :align: center                                          |
        |                                                           |                                                           |
        |   Original crop                                           |   Original crop mask                                      |
        +-----------------------------------------------------------+-----------------------------------------------------------+
        | .. figure:: ../../../img/rd_rot_crop_2d.png               | .. figure:: ../../../img/rd_rot_crop_mask_2d.png          |
        |   :width: 80%                                             |   :width: 80%                                             |
        |   :align: center                                          |   :align: center                                          |
        |                                                           |                                                           |
        |   Random rotation [0, 180] applied                        |   Random rotation [0, 180] applied                        |
        +-----------------------------------------------------------+-----------------------------------------------------------+
        """
        if random_images == False and num_examples > self.length:
            num_examples = self.length
            print(
                "WARNING: More samples requested than the ones available. 'num_examples' fixed to {}".format(
                    num_examples
                )
            )

        sample_x = []
        sample_y = []

        point_dict = {}
        # Generate the examples
        print("0) Creating samples of data augmentation . . .")
        for i in tqdm(range(num_examples), disable=not is_main_process()):
            if random_images:
                pos = random.randint(0, self.length - 1) if self.length > 2 else 0
                pos = pos % self.real_length
            else:
                pos = i

            # Mirror the training __getitem__ (enlarge, then apply_transform crops back). The prob-map
            # path instead keeps its own crop so it can draw the extracted-patch location.
            use_probmap_points = self.random_crops_in_DA and self.prob_map is not None

            img, mask = self.load_sample(pos, geom_enlarge=not use_probmap_points)

            if save_to_dir:
                orig_images = {}
                # Network-size "before augmentation" reference (centre of the possibly enlarged patch).
                orig_images["o_x"] = center_crop_single(np.copy(img), tuple(self.shape[: self.ndim]))
                orig_images["o_y"] = center_crop_single(np.copy(mask), tuple(self.shape[: self.ndim]))
                if draw_grid:
                    self.draw_grid(orig_images["o_x"])
                    self.draw_grid(orig_images["o_y"])

            # Prob-map path: crop to the network size here and record the crop location for
            # save_aug_samples.
            if use_probmap_points:
                if isinstance(self.prob_map, list):
                    img_prob = np.load(self.prob_map[pos])
                else:
                    img_prob = self.prob_map[pos]

                if self.ndim == 2:
                    img, mask, oy, ox, s_y, s_x = random_crop_pair(  # type: ignore
                        img,
                        mask,
                        self.shape[:2],
                        self.val,
                        img_prob=img_prob,
                        draw_prob_map_points=True,
                        scale=self.random_crop_scale,
                    )
                else:
                    img, mask, oz, oy, ox, s_z, s_y, s_x = random_3D_crop_pair(  # type: ignore
                        img,
                        mask,
                        self.shape[:3],
                        self.val,
                        img_prob=img_prob,
                        draw_prob_map_points=True,
                    )
                if save_to_dir:
                    (
                        point_dict["oy"],
                        point_dict["ox"],
                        point_dict["s_y"],
                        point_dict["s_x"],
                    ) = (oy, ox, s_y, s_x)
                if self.ndim == 3:
                    point_dict["oz"], point_dict["s_z"] = oz, s_z

            sample_x.append(img)
            sample_y.append(mask)

            # Apply transformations
            if self.da:
                if not train and draw_grid:
                    sample_x[i] = self.draw_grid(np.copy(sample_x[i]))
                    sample_y[i] = self.draw_grid(np.copy(sample_y[i]))

                e_img, e_mask = None, None
                if self.cutmix:
                    extra_img = np.random.randint(0, self.length - 1) if self.length > 2 else 0
                    e_img, e_mask = self.load_sample(extra_img)

                sample_x[i], sample_y[i] = self.apply_transform(
                    sample_x[i],
                    sample_y[i],
                    e_im=e_img,
                    e_mask=e_mask,
                    diam_factor=self.cellpose_diam_factor(pos),
                )

            if self.n2v and not self.val:
                img, mask = self.prepare_n2v(img, mask)
                sample_y[i] = mask

            if save_to_dir:
                self.save_aug_samples(sample_x[i], sample_y[i], orig_images, i, pos, out_dir, point_dict)

    def draw_grid(self, im: NDArray, grid_width: Optional[int] = None) -> NDArray:
        """
        Draw grid of the specified size on an image.

        Parameters
        ----------
        im : 3D/4D Numpy array
            Image to draw the grid into. E.g. ``(y, x, channels)`` in ``2D`` or ``(z, y, x, channels)`` in ``3D``.

        grid_width : int, optional
            Grid's width.
        """
        vmax = []
        for c in range(im.shape[-1]):
            vmax.append(np.max(im[...,c]))

        if grid_width is not None and grid_width > 0:
            grid_y = grid_width
            grid_x = grid_width
        else:
            grid_y = im.shape[self.ndim - 2] // 5
            grid_x = im.shape[self.ndim - 2] // 5

        if self.ndim == 2:
            for i in range(0, im.shape[0], grid_y):
                im[i] = vmax
            for j in range(0, im.shape[1], grid_x):
                im[:, j] = vmax
        else:
            for k in range(0, im.shape[0]):
                for i in range(0, im.shape[2], grid_x):
                    im[k, :, i] = vmax
                for j in range(0, im.shape[1], grid_y):
                    im[k, j] = vmax
        return im

    def prepare_n2v(self, _img: NDArray, _mask: NDArray) -> Tuple[NDArray, NDArray]:
        """
        Create Noise2Void mask.

        Parameters
        ----------
        _img : 3D/4D Numpy array
            Image to wipe some pixels from. E.g. ``(y, x, channels)`` in ``2D`` or ``(z, y, x, channels)`` in ``3D``.

        _mask : 3D/4D Numpy array
            Mask to use values from. Only used when the ground truth is loaded with ``self.n2v_load_gt``.
            E.g. ``(y, x, channels)`` in ``2D`` or ``(z, y, x, channels)`` in ``3D``.

        Returns
        -------
        img : 3D/4D Numpy array
            Input image modified removing some pixels. E.g. ``(y, x, channels)`` in ``2D`` or ``(y, x, z, channels)`` in ``3D``.

        mask : 3D/4D Numpy array
            Noise2Void mask created. E.g. ``(y, x, channels)`` in ``2D`` or ``(y, x, z, channels)`` in ``3D``.
        """
        img = _img.copy()
        mask = np.zeros(img.shape[:-1] + (img.shape[-1] * 2,), dtype=np.float32)

        if self.val:
            np.random.seed(0)

        for c in range(self.Y_channels):
            coords = self.get_stratified_coords(box_size=self.box_size, shape=self.shape)
            indexing = coords + (c,)
            indexing_mask = coords + (c + self.Y_channels,)
            if self.n2v_load_gt:
                y_val = _mask[indexing]
            else:
                y_val = img[indexing]
            x_val = self.value_manipulation(img[..., c], coords, self.ndim, self.n2v_structMask)

            mask[indexing] = y_val
            mask[indexing_mask] = 1
            img[indexing] = x_val

            if self.n2v_structMask is not None:
                self.apply_structN2Vmask_func(img[..., c], coords, self.n2v_structMask)
        return img, mask
