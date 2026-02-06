"""
BiaPy data generators package.

This package provides data generator classes and utility functions for loading,
augmenting, and batching image and mask data for deep learning workflows in BiaPy.
It supports 2D and 3D data, chunked loading, distributed training, and advanced
augmentation pipelines.
"""
import torch
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
from biapy.data.generators.chunked_workflow_process_generator import chunked_workflow_process_generator
from biapy.data.pre_processing import preprocess_data
from biapy.data.data_manipulation import save_tif
from biapy.data.dataset import BiaPyDataset
from biapy.data.norm import Normalization
from biapy.utils.misc import get_rank, get_world_size, is_dist_avail_and_initialized, os_walk_clean
from biapy.models.bmz_utils import extract_BMZ_sample_and_cover

def create_train_val_augmentors(
    cfg: CN,
    system_dict: Dict[str, Any],
    X_train: BiaPyDataset,
    X_val: BiaPyDataset,
    norm_module: Normalization,
    Y_train: Optional[BiaPyDataset] = None,
    Y_val: Optional[BiaPyDataset] = None,
) -> Tuple[DataLoader, DataLoader, Normalization, int, NDArray, NDArray, NDArray]:
    """
    Create training and validation generators.

    Parameters
    ----------
    cfg : Config
        BiaPy configuration.

    system_dict : dict
        System dictionary containing:
            * 'cpu_budget': int, Total CPU budget.
            * 'cpu_per_rank': int, CPU budget per rank.
            * 'main_threads': int, Number of main threads.
            * 'num_workers_hint': int, Hint for the number of workers.

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
            contrast=cfg.AUGMENTOR.CONTRAST,
            contrast_factor=cfg.AUGMENTOR.CONTRAST_FACTOR,
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
            n_classes=cfg.DATA.N_CLASSES,
            ignore_index=cfg.LOSS.IGNORE_INDEX,
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
            val=True,
            n_classes=cfg.DATA.N_CLASSES,
            ignore_index=cfg.LOSS.IGNORE_INDEX,
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

    # ---- Choose num_workers for this DataLoader ----
    # Priority:
    # 1) If user explicitly set SYSTEM.NUM_WORKERS != -1 => respect it
    # 2) Else use the precomputed hint from startup
    if cfg.SYSTEM.NUM_WORKERS != -1:
        num_workers = max(0, int(cfg.SYSTEM.NUM_WORKERS))
    else:
        # Use the value computed earlier at startup
        num_workers = int(system_dict.get("num_workers_hint", 0))

    # Don't spawn more workers than samples (helps tiny datasets / edge cases)
    num_workers = min(num_workers, training_samples) if training_samples > 0 else 0

    # Ensure DataLoader workers don't each spawn many threads
    def worker_init_fn(worker_id):
        torch.set_num_threads(1)

    # Set num_workers
    if is_dist_avail_and_initialized() and cfg.SYSTEM.NUM_GPUS >= 1:
        sampler_train = DistributedSampler(
            train_generator,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=True
        )
        DataLoader_shuffle = False  # IMPORTANT: shuffle must be False when sampler is used
    else:
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
        worker_init_fn=worker_init_fn,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    # Save a sample to export the model to BMZ
    bmz_input_sample = None
    bmz_input_sample, mask_sample = train_generator.load_sample(0, first_load=True)
    bmz_input_sample, cover_raw, cover_gt = extract_BMZ_sample_and_cover(
        img=bmz_input_sample,
        img_gt=mask_sample,
        patch_size=cfg.DATA.PATCH_SIZE,
        is_3d=cfg.PROBLEM.NDIM == "3D",
        input_axis_order=cfg.DATA.TRAIN.INPUT_IMG_AXES_ORDER,
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

    # Validation dataset
    sampler_val = None
    if cfg.DATA.VAL.DIST_EVAL:
        if len(val_generator) % get_world_size() != 0:
            print(
                "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                "This will slightly alter validation results as extra duplicate entries are added to achieve "
                "equal num of samples per-process."
            )
        sampler_val = DistributedSampler(
            val_generator,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=False
        )
    else:
        sampler_val = SequentialSampler(val_generator)

    # Don't spawn more workers than validation samples
    val_samples = len(val_generator)
    num_workers_val = min(num_workers, val_samples) if val_samples > 0 else 0

    val_dataset = DataLoader(
        val_generator,
        sampler=sampler_val,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=num_workers_val,
        pin_memory=cfg.SYSTEM.PIN_MEM,
        drop_last=False,
        shuffle=False,
        worker_init_fn=worker_init_fn,
        persistent_workers=(num_workers_val > 0),
        prefetch_factor=2 if num_workers_val > 0 else None,
    )

    return train_dataset, val_dataset, data_norm, num_training_steps_per_epoch, bmz_input_sample, cover_raw, cover_gt


def create_test_generator(
    cfg: CN,
    X_test: Any,
    Y_test: Any,
    norm_module: Normalization,
) -> Tuple[test_pair_data_generator | test_single_data_generator, Normalization, NDArray, NDArray, NDArray]:
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
        dic["ignore_index"] = cfg.LOSS.IGNORE_INDEX
        dic["n_classes"] = cfg.DATA.N_CLASSES
    
    test_generator = gen_name(**dic)
    data_norm = test_generator.get_data_normalization()

    # Save a sample to export the model to BMZ
    bmz_input_sample = None
    if gen_name == test_single_data_generator:
        bmz_input_sample, _ , _, _, _ = test_generator.load_sample(0, first_load=True) # type: ignore
        mask_sample = None
    else:
        bmz_input_sample, mask_sample, _, _, _, _ = test_generator.load_sample(0, first_load=True) # type: ignore
    bmz_input_sample, cover_raw, cover_gt = extract_BMZ_sample_and_cover(
        img=bmz_input_sample[0] if (isinstance(bmz_input_sample, np.ndarray) and not cfg.TEST.BY_CHUNKS.ENABLE) else bmz_input_sample,
        img_gt=mask_sample[0] if (isinstance(mask_sample, np.ndarray) and not cfg.TEST.BY_CHUNKS.ENABLE) else mask_sample,
        patch_size=cfg.DATA.PATCH_SIZE,
        is_3d=cfg.PROBLEM.NDIM == "3D",
        input_axis_order=cfg.DATA.TEST.INPUT_IMG_AXES_ORDER,
    ) 

    # Ensure dimensions
    if cfg.PROBLEM.NDIM == "2D":
        if bmz_input_sample.ndim == 3:
            bmz_input_sample = np.expand_dims(bmz_input_sample, 0)
        bmz_input_sample = bmz_input_sample.transpose(0, 3, 1, 2)  # Numpy -> Torch
    else:  # 3D
        if bmz_input_sample.ndim == 4:
            bmz_input_sample = np.expand_dims(bmz_input_sample, 0)
        bmz_input_sample = bmz_input_sample.transpose(0, 4, 1, 2, 3)  # Numpy -> Torch

    return test_generator, data_norm, bmz_input_sample, cover_raw, cover_gt

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
    system_dict: Dict[str, Any],
    current_sample: Dict,
    norm_module: Normalization,
    out_dir: str,
    dtype_str: str,
) -> DataLoader:
    """
    Create a DataLoader for chunked test data using chunked_test_pair_data_generator.

    This function sets up a generator for efficient inference on large volumetric datasets
    by processing data in chunks. It configures the generator with the appropriate axes,
    patch size, padding, and normalization, and wraps it in a PyTorch DataLoader with
    optimal worker settings for distributed or single-GPU environments.

    Parameters
    ----------
    cfg : CN
        BiaPy configuration node.
    
    system_dict : dict
        System dictionary containing:
            * 'cpu_budget': int, Total CPU budget.
            * 'cpu_per_rank': int, CPU budget per rank.
            * 'main_threads': int, Number of main threads.  
            * 'num_workers_hint': int, Hint for the number of workers.

    current_sample : dict
        Dictionary containing the sample to process (e.g., file pointers, data arrays).

    norm_module : Normalization
        Normalization module to apply to the data.

    out_dir : str
        Output directory to save results.

    dtype_str : str
        Data type string for output files.

    Returns
    -------
    test_dataset : DataLoader
        PyTorch DataLoader wrapping the chunked test data generator.
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
        n_classes=cfg.DATA.N_CLASSES,
        ignore_index=cfg.LOSS.IGNORE_INDEX,
        instance_problem = cfg.PROBLEM.TYPE == "INSTANCE_SEG",
    )

    # ---- Choose num_workers for this DataLoader ----
    # Priority:
    # 1) Respect explicit SYSTEM.NUM_WORKERS if set
    # 2) Else reuse the precomputed hint from startup (system_dict["num_workers_hint"])
    if cfg.SYSTEM.NUM_WORKERS != -1:
        num_workers = max(0, int(cfg.SYSTEM.NUM_WORKERS))
    else:
        num_workers = int(system_dict.get("num_workers_hint", 0))

    # Cap by dataset length if the generator supports __len__
    try:
        n_chunks = len(chunked_generator)  # may raise TypeError if __len__ not implemented
        if n_chunks > 0:
            num_workers = min(num_workers, n_chunks)
        else:
            num_workers = 0
    except TypeError:
        # length unknown -> keep computed num_workers
        pass

    # Ensure DataLoader workers don't each spawn many threads
    def worker_init_fn(worker_id):
        torch.set_num_threads(1)

    print(f"Chunked test generator with {num_workers} workers")

    test_dataset = DataLoader(
        chunked_generator,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=num_workers,
        collate_fn=by_chunks_collate_fn,
        pin_memory=cfg.SYSTEM.PIN_MEM,
        drop_last=False,
        worker_init_fn=worker_init_fn,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    return test_dataset

def by_chunks_workflow_collate_fn(data):
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
        [x[0] for x in data],
        [x[1] for x in data],
        [x[2] for x in data],
    )

def create_chunked_workflow_process_generator(
    cfg: CN,
    system_dict: Dict[str, Any],
    model_predictions: str,
    out_dir: str,
    dtype_str: str,
) -> DataLoader:
    """
    Create a DataLoader for chunked test data using chunked_workflow_process_generator.

    This function sets up a generator for efficient inference on large volumetric datasets
    by processing data in chunks. It configures the generator with the appropriate axes,
    patch size, padding, and normalization, and wraps it in a PyTorch DataLoader with
    optimal worker settings for distributed or single-GPU environments.

    Parameters
    ----------
    cfg : CN
        BiaPy configuration node.
    
    system_dict : dict
        System dictionary containing:
            * 'cpu_budget': int, Total CPU budget.
            * 'cpu_per_rank': int, CPU budget per rank.
            * 'main_threads': int, Number of main threads.  
            * 'num_workers_hint': int, Hint for the number of workers.

    model_predictions : str
        Path to the model predictions to process.

    out_dir : str
        Output directory to save results.

    dtype_str : str
        Data type string for output files.

    Returns
    -------
    test_dataset : DataLoader
        PyTorch DataLoader wrapping the chunked test data generator.
    """
    if "C" not in cfg.DATA.TEST.INPUT_IMG_AXES_ORDER:
        out_data_order = cfg.DATA.TEST.INPUT_IMG_AXES_ORDER + "C"
    else:
        out_data_order = cfg.DATA.TEST.INPUT_IMG_AXES_ORDER

    chunked_generator = chunked_workflow_process_generator(
        model_predictions=model_predictions,
        input_axes=out_data_order,
        crop_shape=cfg.DATA.PATCH_SIZE,
        out_dir=out_dir,
        dtype_str=dtype_str,
    )

    # ---- Choose num_workers for this DataLoader ----
    # Priority:
    # 1) Respect explicit SYSTEM.NUM_WORKERS if set
    # 2) Else reuse the precomputed hint from startup (system_dict["num_workers_hint"])
    if cfg.SYSTEM.NUM_WORKERS != -1:
        num_workers = max(0, int(cfg.SYSTEM.NUM_WORKERS))
    else:
        num_workers = int(system_dict.get("num_workers_hint", 0))

    # Cap by dataset length if the generator supports __len__
    try:
        n_chunks = len(chunked_generator)  # may raise TypeError if __len__ not implemented
        if n_chunks > 0:
            num_workers = min(num_workers, n_chunks)
        else:
            num_workers = 0
    except TypeError:
        # length unknown -> keep computed num_workers
        pass

    # Ensure DataLoader workers don't each spawn many threads
    def worker_init_fn(worker_id):
        torch.set_num_threads(1)

    print(f"Chunked test generator with {num_workers} workers")

    test_dataset = DataLoader(
        chunked_generator,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=num_workers,
        collate_fn=by_chunks_workflow_collate_fn,
        pin_memory=cfg.SYSTEM.PIN_MEM,
        drop_last=False,
        worker_init_fn=worker_init_fn,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
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