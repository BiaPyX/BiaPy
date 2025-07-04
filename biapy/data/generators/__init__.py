import os
from typing import List, Dict, Any, Tuple, Optional
from torch.utils.data import (
    DistributedSampler,
    DataLoader,
    SequentialSampler,
)
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from yacs.config import CfgNode as CN

from biapy.data.pre_processing import calculate_volume_prob_map
from biapy.data.generators.pair_data_2D_generator import Pair2DImageDataGenerator
from biapy.data.generators.pair_data_3D_generator import Pair3DImageDataGenerator
from biapy.data.generators.single_data_2D_generator import Single2DImageDataGenerator
from biapy.data.generators.single_data_3D_generator import Single3DImageDataGenerator
from biapy.data.generators.test_pair_data_generators import test_pair_data_generator
from biapy.data.generators.test_single_data_generator import test_single_data_generator
from biapy.data.generators.chunked_test_pair_data_generator import chunked_test_pair_data_generator
from biapy.data.pre_processing import preprocess_data
from biapy.data.data_manipulation import save_tif, extract_patch_within_image
from biapy.data.dataset import BiaPyDataset, PatchCoords
from biapy.data.norm import Normalization
from biapy.data.data_3D_manipulation import extract_patch_from_efficient_file
from biapy.utils.misc import get_rank, get_world_size, is_dist_avail_and_initialized, os_walk_clean


def create_train_val_augmentors(
    cfg: CN,
    X_train: BiaPyDataset,
    X_val: BiaPyDataset,
    norm_module: Normalization,
    Y_train: Optional[BiaPyDataset] = None,
    Y_val: Optional[BiaPyDataset] = None,
) -> Tuple[DataLoader, DataLoader, Normalization, int, NDArray]:
    """
    Create training and validation generators.

    Parameters
    ----------
    cfg : Config
        BiaPy configuration.

    X_train : BiaPyDataset
        Loaded train X data.

    X_val : BiaPyDataset
        Loaded train Y data.

    norm_module : Normalization
        Normalization module that defines the normalization steps to apply.

    Y_train : BiaPyDataset, optional
        Loaded train Y data.

    Y_val : BiaPyDataset, optional
        Loaded validation Y data.

    Returns
    -------
    train_generator : DataLoader
        Training data generator.

    val_generator : DataLoader
        Validation data generator.

    data_norm : Normalization
        Normalization of the data.

    num_training_steps_per_epoch: int
        Number of training steps per epoch.
    """

    # Calculate the probability map per image
    prob_map = None
    if cfg.DATA.PROBABILITY_MAP and cfg.DATA.EXTRACT_RANDOM_PATCH:
        if os.path.exists(cfg.PATHS.PROB_MAP_DIR):
            print("Loading probability map")
            prob_map_file = os.path.join(cfg.PATHS.PROB_MAP_DIR, cfg.PATHS.PROB_MAP_FILENAME)
            num_files = len(next(os_walk_clean(cfg.PATHS.PROB_MAP_DIR))[2])
            prob_map = cfg.PATHS.PROB_MAP_DIR if num_files > 1 else np.load(prob_map_file)
        else:
            assert Y_train
            prob_map = calculate_volume_prob_map(
                Y_train,
                (cfg.PROBLEM.NDIM == "3D"),
                cfg.DATA.W_FOREGROUND,
                cfg.DATA.W_BACKGROUND,
                save_dir=cfg.PATHS.PROB_MAP_DIR,
            )

    if cfg.PROBLEM.NDIM == "2D":
        if cfg.PROBLEM.TYPE == "CLASSIFICATION" or (
            cfg.PROBLEM.TYPE == "SELF_SUPERVISED" and cfg.PROBLEM.SELF_SUPERVISED.PRETEXT_TASK == "masking"
        ):
            f_name = Single2DImageDataGenerator
        else:
            f_name = Pair2DImageDataGenerator
    else:
        if cfg.PROBLEM.TYPE == "CLASSIFICATION" or (
            cfg.PROBLEM.TYPE == "SELF_SUPERVISED" and cfg.PROBLEM.SELF_SUPERVISED.PRETEXT_TASK == "masking"
        ):
            f_name = Single3DImageDataGenerator
        else:
            f_name = Pair3DImageDataGenerator

    ndim = 3 if cfg.PROBLEM.NDIM == "3D" else 2
    if cfg.PROBLEM.TYPE == "CLASSIFICATION" or (
        cfg.PROBLEM.TYPE == "SELF_SUPERVISED" and cfg.PROBLEM.SELF_SUPERVISED.PRETEXT_TASK == "masking"
    ):
        r_shape = cfg.DATA.PATCH_SIZE
        if cfg.MODEL.ARCHITECTURE == "efficientnet_b0" and cfg.DATA.PATCH_SIZE[:-1] != (
            224,
            224,
        ):
            r_shape = (224, 224) + (cfg.DATA.PATCH_SIZE[-1],)
            print("Changing patch size from {} to {} to use efficientnet_b0".format(cfg.DATA.PATCH_SIZE[:-1], r_shape))
        dic = dict(
            ndim=ndim,
            X=X_train,
            seed=cfg.SYSTEM.SEED,
            da=cfg.AUGMENTOR.ENABLE,
            da_prob=cfg.AUGMENTOR.DA_PROB,
            rotation90=cfg.AUGMENTOR.ROT90,
            rand_rot=cfg.AUGMENTOR.RANDOM_ROT,
            rnd_rot_range=cfg.AUGMENTOR.RANDOM_ROT_RANGE,
            shear=cfg.AUGMENTOR.SHEAR,
            shear_range=cfg.AUGMENTOR.SHEAR_RANGE,
            zoom=cfg.AUGMENTOR.ZOOM,
            zoom_range=cfg.AUGMENTOR.ZOOM_RANGE,
            zoom_in_z=cfg.AUGMENTOR.ZOOM_IN_Z,
            shift=cfg.AUGMENTOR.SHIFT,
            shift_range=cfg.AUGMENTOR.SHIFT_RANGE,
            affine_mode=cfg.AUGMENTOR.AFFINE_MODE,
            vflip=cfg.AUGMENTOR.VFLIP,
            hflip=cfg.AUGMENTOR.HFLIP,
            elastic=cfg.AUGMENTOR.ELASTIC,
            e_alpha=cfg.AUGMENTOR.E_ALPHA,
            e_sigma=cfg.AUGMENTOR.E_SIGMA,
            e_mode=cfg.AUGMENTOR.E_MODE,
            g_blur=cfg.AUGMENTOR.G_BLUR,
            g_sigma=cfg.AUGMENTOR.G_SIGMA,
            median_blur=cfg.AUGMENTOR.MEDIAN_BLUR,
            mb_kernel=cfg.AUGMENTOR.MB_KERNEL,
            motion_blur=cfg.AUGMENTOR.MOTION_BLUR,
            motb_k_range=cfg.AUGMENTOR.MOTB_K_RANGE,
            gamma_contrast=cfg.AUGMENTOR.GAMMA_CONTRAST,
            gc_gamma=cfg.AUGMENTOR.GC_GAMMA,
            dropout=cfg.AUGMENTOR.DROPOUT,
            drop_range=cfg.AUGMENTOR.DROP_RANGE,
            resize_shape=r_shape,
            norm_module=norm_module,
            convert_to_rgb=cfg.DATA.FORCE_RGB,
            preprocess_f=preprocess_data if cfg.DATA.PREPROCESS.TRAIN else None,
            preprocess_cfg=cfg.DATA.PREPROCESS if cfg.DATA.PREPROCESS.TRAIN else None,
        )
    else:
        dic = dict(
            ndim=ndim,
            X=X_train,
            Y=Y_train,
            seed=cfg.SYSTEM.SEED,
            da=cfg.AUGMENTOR.ENABLE,
            da_prob=cfg.AUGMENTOR.DA_PROB,
            rotation90=cfg.AUGMENTOR.ROT90,
            rand_rot=cfg.AUGMENTOR.RANDOM_ROT,
            rnd_rot_range=cfg.AUGMENTOR.RANDOM_ROT_RANGE,
            shear=cfg.AUGMENTOR.SHEAR,
            shear_range=cfg.AUGMENTOR.SHEAR_RANGE,
            zoom=cfg.AUGMENTOR.ZOOM,
            zoom_range=cfg.AUGMENTOR.ZOOM_RANGE,
            zoom_in_z=cfg.AUGMENTOR.ZOOM_IN_Z,
            shift=cfg.AUGMENTOR.SHIFT,
            affine_mode=cfg.AUGMENTOR.AFFINE_MODE,
            shift_range=cfg.AUGMENTOR.SHIFT_RANGE,
            vflip=cfg.AUGMENTOR.VFLIP,
            hflip=cfg.AUGMENTOR.HFLIP,
            elastic=cfg.AUGMENTOR.ELASTIC,
            e_alpha=cfg.AUGMENTOR.E_ALPHA,
            e_sigma=cfg.AUGMENTOR.E_SIGMA,
            e_mode=cfg.AUGMENTOR.E_MODE,
            g_blur=cfg.AUGMENTOR.G_BLUR,
            g_sigma=cfg.AUGMENTOR.G_SIGMA,
            median_blur=cfg.AUGMENTOR.MEDIAN_BLUR,
            mb_kernel=cfg.AUGMENTOR.MB_KERNEL,
            motion_blur=cfg.AUGMENTOR.MOTION_BLUR,
            motb_k_range=cfg.AUGMENTOR.MOTB_K_RANGE,
            gamma_contrast=cfg.AUGMENTOR.GAMMA_CONTRAST,
            gc_gamma=cfg.AUGMENTOR.GC_GAMMA,
            brightness=cfg.AUGMENTOR.BRIGHTNESS,
            brightness_factor=cfg.AUGMENTOR.BRIGHTNESS_FACTOR,
            brightness_mode=cfg.AUGMENTOR.BRIGHTNESS_MODE,
            contrast=cfg.AUGMENTOR.CONTRAST,
            contrast_factor=cfg.AUGMENTOR.CONTRAST_FACTOR,
            contrast_mode=cfg.AUGMENTOR.CONTRAST_MODE,
            dropout=cfg.AUGMENTOR.DROPOUT,
            drop_range=cfg.AUGMENTOR.DROP_RANGE,
            cutout=cfg.AUGMENTOR.CUTOUT,
            cout_nb_iterations=cfg.AUGMENTOR.COUT_NB_ITERATIONS,
            cout_size=cfg.AUGMENTOR.COUT_SIZE,
            cout_cval=cfg.AUGMENTOR.COUT_CVAL,
            cout_apply_to_mask=cfg.AUGMENTOR.COUT_APPLY_TO_MASK,
            cutblur=cfg.AUGMENTOR.CUTBLUR,
            cblur_size=cfg.AUGMENTOR.CBLUR_SIZE,
            cblur_down_range=cfg.AUGMENTOR.CBLUR_DOWN_RANGE,
            cblur_inside=cfg.AUGMENTOR.CBLUR_INSIDE,
            cutmix=cfg.AUGMENTOR.CUTMIX,
            cmix_size=cfg.AUGMENTOR.CMIX_SIZE,
            cutnoise=cfg.AUGMENTOR.CUTNOISE,
            cnoise_size=cfg.AUGMENTOR.CNOISE_SIZE,
            cnoise_nb_iterations=cfg.AUGMENTOR.CNOISE_NB_ITERATIONS,
            cnoise_scale=cfg.AUGMENTOR.CNOISE_SCALE,
            misalignment=cfg.AUGMENTOR.MISALIGNMENT,
            ms_displacement=cfg.AUGMENTOR.MS_DISPLACEMENT,
            ms_rotate_ratio=cfg.AUGMENTOR.MS_ROTATE_RATIO,
            missing_sections=cfg.AUGMENTOR.MISSING_SECTIONS,
            missp_iterations=cfg.AUGMENTOR.MISSP_ITERATIONS,
            missp_channel_pb=cfg.AUGMENTOR.MISSP_CHANNEL_PB,
            grayscale=cfg.AUGMENTOR.GRAYSCALE,
            channel_shuffle=cfg.AUGMENTOR.CHANNEL_SHUFFLE,
            gridmask=cfg.AUGMENTOR.GRIDMASK,
            grid_ratio=cfg.AUGMENTOR.GRID_RATIO,
            grid_d_range=cfg.AUGMENTOR.GRID_D_RANGE,
            grid_rotate=cfg.AUGMENTOR.GRID_ROTATE,
            grid_invert=cfg.AUGMENTOR.GRID_INVERT,
            gaussian_noise=cfg.AUGMENTOR.GAUSSIAN_NOISE,
            gaussian_noise_mean=cfg.AUGMENTOR.GAUSSIAN_NOISE_MEAN,
            gaussian_noise_var=cfg.AUGMENTOR.GAUSSIAN_NOISE_VAR,
            gaussian_noise_use_input_img_mean_and_var=cfg.AUGMENTOR.GAUSSIAN_NOISE_USE_INPUT_IMG_MEAN_AND_VAR,
            poisson_noise=cfg.AUGMENTOR.POISSON_NOISE,
            salt=cfg.AUGMENTOR.SALT,
            salt_amount=cfg.AUGMENTOR.SALT_AMOUNT,
            pepper=cfg.AUGMENTOR.PEPPER,
            pepper_amount=cfg.AUGMENTOR.PEPPER_AMOUNT,
            salt_and_pepper=cfg.AUGMENTOR.SALT_AND_PEPPER,
            salt_pep_amount=cfg.AUGMENTOR.SALT_AND_PEPPER_AMOUNT,
            salt_pep_proportion=cfg.AUGMENTOR.SALT_AND_PEPPER_PROP,
            shape=cfg.DATA.PATCH_SIZE,
            resolution=cfg.DATA.TRAIN.RESOLUTION,
            random_crops_in_DA=cfg.DATA.EXTRACT_RANDOM_PATCH,
            prob_map=prob_map,
            n_classes=cfg.MODEL.N_CLASSES,
            ignore_index=None if not cfg.LOSS.IGNORE_VALUES else cfg.LOSS.VALUE_TO_IGNORE,
            extra_data_factor=cfg.DATA.TRAIN.REPLICATE,
            norm_module=norm_module,
            random_crop_scale=cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING,
            convert_to_rgb=cfg.DATA.FORCE_RGB,
            preprocess_f=preprocess_data if cfg.DATA.PREPROCESS.TRAIN else None,
            preprocess_cfg=cfg.DATA.PREPROCESS if cfg.DATA.PREPROCESS.TRAIN else None,
        )

        if cfg.PROBLEM.NDIM == "3D":
            dic["zflip"] = cfg.AUGMENTOR.ZFLIP
        if cfg.PROBLEM.TYPE == "INSTANCE_SEG":
            dic["instance_problem"] = True
        elif cfg.PROBLEM.TYPE == "DENOISING":
            dic["n2v"] = True
            dic["n2v_perc_pix"] = cfg.PROBLEM.DENOISING.N2V_PERC_PIX
            dic["n2v_manipulator"] = cfg.PROBLEM.DENOISING.N2V_MANIPULATOR
            dic["n2v_neighborhood_radius"] = cfg.PROBLEM.DENOISING.N2V_NEIGHBORHOOD_RADIUS
            dic["n2v_structMask"] = (
                np.array([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]]) if cfg.PROBLEM.DENOISING.N2V_STRUCTMASK else None
            )
            dic["n2v_load_gt"] = cfg.PROBLEM.DENOISING.LOAD_GT_DATA

    print("Initializing train data generator . . .")
    train_generator = f_name(**dic)  # type: ignore
    data_norm = train_generator.get_data_normalization()

    print("Initializing val data generator . . .")
    if cfg.PROBLEM.TYPE == "CLASSIFICATION" or (
        cfg.PROBLEM.TYPE == "SELF_SUPERVISED" and cfg.PROBLEM.SELF_SUPERVISED.PRETEXT_TASK == "masking"
    ):
        val_generator = f_name(
            ndim=ndim,
            X=X_val,
            seed=cfg.SYSTEM.SEED,
            da=False,
            resize_shape=r_shape,
            norm_module=norm_module,
            preprocess_f=preprocess_data if cfg.DATA.PREPROCESS.VAL else None,
            preprocess_cfg=cfg.DATA.PREPROCESS if cfg.DATA.PREPROCESS.VAL else None,
        )
    else:
        dic = dict(
            ndim=ndim,
            X=X_val,
            Y=Y_val,
            da=False,
            shape=cfg.DATA.PATCH_SIZE,
            random_crops_in_DA=cfg.DATA.EXTRACT_RANDOM_PATCH,
            val=True,
            n_classes=cfg.MODEL.N_CLASSES,
            ignore_index=None if not cfg.LOSS.IGNORE_VALUES else cfg.LOSS.VALUE_TO_IGNORE,
            seed=cfg.SYSTEM.SEED,
            norm_module=norm_module,
            resolution=cfg.DATA.VAL.RESOLUTION,
            random_crop_scale=cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING,
            preprocess_f=preprocess_data if cfg.DATA.PREPROCESS.VAL else None,
            preprocess_cfg=cfg.DATA.PREPROCESS if cfg.DATA.PREPROCESS.VAL else None,
        )
        if cfg.PROBLEM.TYPE == "INSTANCE_SEG":
            dic["instance_problem"] = True
        elif cfg.PROBLEM.TYPE == "DENOISING":
            dic["n2v"] = True
            dic["n2v_perc_pix"] = cfg.PROBLEM.DENOISING.N2V_PERC_PIX
            dic["n2v_manipulator"] = cfg.PROBLEM.DENOISING.N2V_MANIPULATOR
            dic["n2v_neighborhood_radius"] = cfg.PROBLEM.DENOISING.N2V_NEIGHBORHOOD_RADIUS

        val_generator = f_name(**dic)  # type: ignore

    # Generate examples of data augmentation
    if cfg.AUGMENTOR.AUG_SAMPLES and cfg.AUGMENTOR.ENABLE:
        print("Creating generator samples . . .")
        train_generator.get_transformed_samples(
            cfg.AUGMENTOR.AUG_NUM_SAMPLES,
            save_to_dir=True,
            train=False,
            out_dir=cfg.PATHS.DA_SAMPLES,
            draw_grid=cfg.AUGMENTOR.DRAW_GRID,
        )

    # Training dataset
    total_batch_size = cfg.TRAIN.BATCH_SIZE * get_world_size() * cfg.TRAIN.ACCUM_ITER
    training_samples = len(train_generator)

    # Set num_workers
    if is_dist_avail_and_initialized() and cfg.SYSTEM.NUM_GPUS >= 1:
        DataLoader_shuffle = None
        if cfg.SYSTEM.NUM_WORKERS == -1:
            num_workers = 2 * cfg.SYSTEM.NUM_GPUS
        else:
            # To not create more than 2 processes per GPU
            num_workers = min(cfg.SYSTEM.NUM_WORKERS, 2 * cfg.SYSTEM.NUM_GPUS)
        sampler_train = DistributedSampler(
            train_generator, num_replicas=get_world_size(), rank=get_rank(), shuffle=True
        )
    else:
        # Reduce number of workers in case there is no training data
        num_workers = min(5, training_samples) if cfg.SYSTEM.NUM_WORKERS == -1 else cfg.SYSTEM.NUM_WORKERS
        sampler_train = None
        DataLoader_shuffle = True

    num_training_steps_per_epoch = training_samples // total_batch_size
    print(f"Train/val generators with {num_workers} workers")
    print("Accumulate grad iterations: %d" % cfg.TRAIN.ACCUM_ITER)
    print("Effective batch size: %d" % total_batch_size)
    print("Sampler_train = %s" % str(sampler_train))
    train_dataset = DataLoader(
        train_generator,
        shuffle=DataLoader_shuffle,
        sampler=sampler_train,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=num_workers,
        pin_memory=cfg.SYSTEM.PIN_MEM,
        drop_last=False,
    )

    # Save a sample to export the model to BMZ
    bmz_input_sample = None
    bmz_input_sample, _ = train_generator.load_sample(0, first_load=True)
    bmz_input_sample = bmz_input_sample.astype(np.float32)

    # Ensure dimensions
    if cfg.PROBLEM.NDIM == "2D":
        if bmz_input_sample.ndim == 3:
            bmz_input_sample = np.expand_dims(bmz_input_sample, 0)
        bmz_input_sample = bmz_input_sample.transpose(0, 3, 1, 2)  # Numpy -> Torch
    else:  # 3D
        if bmz_input_sample.ndim == 4:
            bmz_input_sample = np.expand_dims(bmz_input_sample, 0)
        bmz_input_sample = bmz_input_sample.transpose(0, 4, 1, 2, 3)  # Numpy -> Torch

    # Validation dataset
    sampler_val = None
    if cfg.DATA.VAL.DIST_EVAL:
        if len(val_generator) % get_world_size() != 0:
            print(
                "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                "This will slightly alter validation results as extra duplicate entries are added to achieve "
                "equal num of samples per-process."
            )
        sampler_val = DistributedSampler(val_generator, num_replicas=get_world_size(), rank=get_rank(), shuffle=False)
    else:
        sampler_val = SequentialSampler(val_generator)

    val_dataset = DataLoader(
        val_generator,
        sampler=sampler_val,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=num_workers,
        pin_memory=cfg.SYSTEM.PIN_MEM,
        drop_last=False,
        shuffle=False,
    )

    return train_dataset, val_dataset, data_norm, num_training_steps_per_epoch, bmz_input_sample


def create_test_generator(
    cfg: CN,
    X_test: Any,
    Y_test: Any,
    norm_module: Normalization,
) -> Tuple[test_pair_data_generator | test_single_data_generator, Normalization, NDArray]:
    """
    Create test data generator.

    Parameters
    ----------
    cfg : Config
        BiaPy configuration.

    X_test : 4D Numpy array
        Test data. E.g. ``(num_of_images, y, x, channels)`` for ``2D`` or ``(num_of_images, z, y, x, channels)`` for ``3D``.

    Y_test : 4D Numpy array
        Test data mask/class. E.g. ``(num_of_images, y, x, channels)`` for ``2D`` or ``(num_of_images, z, y, x, channels)`` for ``3D``
        in all the workflows except classification. For this last the shape is ``(num_of_images, class)`` for both ``2D`` and ``3D``.

    norm_module : Normalization
        Normalization module that defines the normalization steps to apply.

    Returns
    -------
    test_generator : test_pair_data_generator/test_single_data_generator
        Test data generator.

    data_norm : Normalization
        Normalization of the data.
    """
    if cfg.PROBLEM.TYPE == "SELF_SUPERVISED" and cfg.PROBLEM.SELF_SUPERVISED.PRETEXT_TASK == "masking":
        provide_Y = False
    else:
        provide_Y = cfg.DATA.TEST.LOAD_GT or cfg.DATA.TEST.USE_VAL_AS_TEST

    ndim: int = 3 if cfg.PROBLEM.NDIM == "3D" else 2
    dic = dict(
        X=X_test,
        provide_Y=provide_Y,
        ndim=ndim,
        seed=cfg.SYSTEM.SEED,
        norm_module=norm_module,
        convert_to_rgb=cfg.DATA.FORCE_RGB,
        filter_props=cfg.DATA.TEST.FILTER_SAMPLES.PROPS,
        filter_vals=cfg.DATA.TEST.FILTER_SAMPLES.VALUES,
        filter_signs=cfg.DATA.TEST.FILTER_SAMPLES.SIGNS,
        preprocess_data=preprocess_data if cfg.DATA.PREPROCESS.TEST else None,
        preprocess_cfg=cfg.DATA.PREPROCESS if cfg.DATA.PREPROCESS.TEST else None,
        reflect_to_complete_shape=cfg.DATA.REFLECT_TO_COMPLETE_SHAPE,
        data_shape=cfg.DATA.PATCH_SIZE,
    )

    if cfg.PROBLEM.TYPE == "CLASSIFICATION" or (
        cfg.PROBLEM.TYPE == "SELF_SUPERVISED" and cfg.PROBLEM.SELF_SUPERVISED.PRETEXT_TASK == "masking"
    ):
        gen_name = test_single_data_generator
        r_shape = cfg.DATA.PATCH_SIZE
        if cfg.MODEL.ARCHITECTURE == "efficientnet_b0" and cfg.DATA.PATCH_SIZE[:-1] != (
            224,
            224,
        ):
            r_shape = (224, 224) + (cfg.DATA.PATCH_SIZE[-1],)
            print("Changing patch size from {} to {} to use efficientnet_b0".format(cfg.DATA.PATCH_SIZE[:-1], r_shape))
        if cfg.PROBLEM.TYPE == "CLASSIFICATION":
            dic["crop_center"] = True
            dic["data_shape"] = r_shape
    else:
        gen_name = test_pair_data_generator
        dic["Y"] = Y_test
        dic["test_by_chunks"] = cfg.TEST.BY_CHUNKS.ENABLE
        dic["instance_problem"] = cfg.PROBLEM.TYPE == "INSTANCE_SEG"
        dic["ignore_index"] = None if not cfg.LOSS.IGNORE_VALUES else cfg.LOSS.VALUE_TO_IGNORE
        dic["n_classes"] = cfg.MODEL.N_CLASSES
    
    test_generator = gen_name(**dic)
    data_norm = test_generator.get_data_normalization()

    # Save a sample to export the model to BMZ
    bmz_input_sample = None
    bmz_input_sample, _, _, _, _ = test_generator.load_sample(0, first_load=True)

    # Ensure a patch size
    patch = PatchCoords(
        z_start=0 if cfg.PROBLEM.NDIM == "3D" else None,
        z_end=cfg.DATA.PATCH_SIZE[0] if cfg.PROBLEM.NDIM == "3D" else None,
        y_start=0,
        y_end=cfg.DATA.PATCH_SIZE[-3],
        x_start=0,
        x_end=cfg.DATA.PATCH_SIZE[-2],
    )
    if isinstance(bmz_input_sample, np.ndarray):
        bmz_input_sample = extract_patch_within_image(
            bmz_input_sample, patch, is_3d=True if cfg.PROBLEM.NDIM == "3D" else False
        )
    else:
        bmz_input_sample = extract_patch_from_efficient_file(
            bmz_input_sample, patch, data_axes_order=cfg.DATA.TEST.INPUT_IMG_AXES_ORDER,
        )
    bmz_input_sample = bmz_input_sample.astype(np.float32)

    # Ensure dimensions
    if cfg.PROBLEM.NDIM == "2D":
        if bmz_input_sample.ndim == 3:
            bmz_input_sample = np.expand_dims(bmz_input_sample, 0)
        bmz_input_sample = bmz_input_sample.transpose(0, 3, 1, 2)  # Numpy -> Torch
    else:  # 3D
        if bmz_input_sample.ndim == 4:
            bmz_input_sample = np.expand_dims(bmz_input_sample, 0)
        bmz_input_sample = bmz_input_sample.transpose(0, 4, 1, 2, 3)  # Numpy -> Torch

    return test_generator, data_norm, bmz_input_sample

def by_chunks_collate_fn(data):
    """
    Collate function to avoid the default one with type checking. It does nothing speciall but stack the images.

    Parameters
    ----------
    data : tuple
        Data tuple.

    Returns
    -------
    data : tuple
        Stacked data in batches.
    """
    return (
        # torch.cat([torch.from_numpy(x[0]) for x in data]),
        [x[0] for x in data],
        np.stack([x[1] for x in data]),
        np.stack([x[2] for x in data if x is not None]) if len(data) > 0 and data[0][2] is not None else None,
        [x[3] for x in data],
        [x[4] for x in data],
        [x[5] for x in data],
    )

def create_chunked_test_generator(
    cfg: CN,
    current_sample: Dict,
    norm_module: Normalization,
    out_dir: str,
    dtype_str: str,
) -> DataLoader:
    """
    Creates a chunked test generator.
    """
    chunked_generator = chunked_test_pair_data_generator(
        sample_to_process=current_sample,
        norm_module=norm_module,
        input_axes=cfg.DATA.TEST.INPUT_IMG_AXES_ORDER,
        mask_input_axes=cfg.DATA.TEST.INPUT_MASK_AXES_ORDER,
        crop_shape=cfg.DATA.PATCH_SIZE,
        padding=cfg.DATA.TEST.PADDING,
        out_dir=out_dir,
        dtype_str=dtype_str,
        n_classes=cfg.MODEL.N_CLASSES,
        ignore_index=None if not cfg.LOSS.IGNORE_VALUES else cfg.LOSS.VALUE_TO_IGNORE,
        instance_problem = cfg.PROBLEM.TYPE == "INSTANCE_SEG",
    )

    # Set num_workers
    if is_dist_avail_and_initialized() and cfg.SYSTEM.NUM_GPUS >= 1:
        if cfg.SYSTEM.NUM_WORKERS == -1:
            num_workers = 2 * cfg.SYSTEM.NUM_GPUS
        else:
            # To not create more than 2 processes per GPU
            num_workers = min(cfg.SYSTEM.NUM_WORKERS, 2 * cfg.SYSTEM.NUM_GPUS)
    else:
        num_workers = 5 if cfg.SYSTEM.NUM_WORKERS == -1 else cfg.SYSTEM.NUM_WORKERS
    print(f"Test generator with {num_workers} workers")

    test_dataset = DataLoader(
        chunked_generator,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=num_workers,
        collate_fn=by_chunks_collate_fn,
        pin_memory=cfg.SYSTEM.PIN_MEM,
        drop_last=False,
    )

    return test_dataset


def check_generator_consistence(
    gen: DataLoader, data_out_dir: str, mask_out_dir: str, filenames: List[str] | None = None
):
    """
    Save all data of a generator in the given path.

    Parameters
    ----------
    gen : Pair2DImageDataGenerator/Single2DImageDataGenerator (2D) or Pair3DImageDataGenerator/Single3DImageDataGenerator (3D)
        Generator to extract the data from.

    data_out_dir : str
        Path to store the generator data samples.

    mask_out_dir : str
        Path to store the generator data mask samples.

    Filenames : List, optional
        Filenames that should be used when saving each image.
    """

    print("Check generator . . .")
    it = iter(gen)

    c = 0
    for i in tqdm(range(len(gen))):
        sample = next(it)
        X_test, Y_test = sample

        for k in range(len(X_test)):
            fil = [filenames[c]] if filenames else ["sample_" + str(c) + ".tif"]
            save_tif(np.expand_dims(X_test[k], 0), data_out_dir, fil, verbose=False)
            save_tif(np.expand_dims(Y_test[k], 0), mask_out_dir, fil, verbose=False)
            c += 1


# To accelerate each first batch in epoch without need to.
# Sources: https://discuss.pytorch.org/t/enumerate-dataloader-slow/87778/4
#          https://github.com/huggingface/pytorch-image-models/pull/140/files
# Explanation:
# When using the data loader of pytorch, at the beginning of every epoch, we have to wait a
# lot and the training speed is very low from the first iteration. It is because the pytorch
# data loader is reinitialized from scratch. With this, we do not waste time, and just the
# first initialization of the the dataloader at the first epoch takes time, but for the next
# epochs, the first iteration of every new epoch is as fast as the iterations in the middle
# of an epoch.
class MultiEpochsDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)  # type: ignore

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """
    Sampler that repeats forever.

    Parameters
    ----------
    sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
