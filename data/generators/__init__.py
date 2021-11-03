import os
import numpy as np
from tqdm import tqdm

from utils.util import calculate_2D_volume_prob_map, calculate_3D_volume_prob_map, save_tif
from data.generators.data_2D_generator import ImageDataGenerator
from data.generators.data_3D_generator import VoxelDataGenerator
from data.generators.simple_data_generators import simple_data_generator


def create_train_val_augmentors(cfg, X_train, Y_train, X_val, Y_val):
    """Create training and validation generators.

       Parameters
       ----------
       cfg : YACS CN object
           Configuration.

       X_train : 4D Numpy array
           Training data. E.g. ``(num_of_images, x, y, channels)``.

       Y_train : 4D Numpy array
           Training data mask. E.g. ``(num_of_images, x, y, 1)``.

       X_val : 4D Numpy array
           Validation data mask. E.g. ``(num_of_images, x, y, channels)``.

       Y_val : 4D Numpy array
           Validation data mask. E.g. ``(num_of_images, x, y, 1)``.

       Returns
       -------
       train_generator : ImageDataGenerator (2D) or VoxelDataGenerator (3D)
           Training data generator.

       val_generator : ImageDataGenerator (2D) or VoxelDataGenerator (3D)
           Validation data generator.
    """

    # Calculate the probability map per image
    prob_map = None
    if cfg.DATA.PROBABILITY_MAP and cfg.DATA.EXTRACT_RANDOM_PATCH:
        if os.path.exists(cfg.PATHS.PROB_MAP_DIR):
            print("Loading probability map")
            prob_map_file = os.path.join(cfg.PATHS.PROB_MAP_DIR, cfg.PATHS.PROB_MAP_FILENAME)
            num_files = len(next(os.walk(cfg.PATHS.PROB_MAP_DIR))[2])
            prob_map = cfg.PATHS.PROB_MAP_DIR if num_files > 1 else np.load(prob_map_file)
        else:
            f_name = calculate_2D_volume_prob_map if cfg.PROBLEM.NDIM == '2D' else calculate_3D_volume_prob_map
            prob_map = f_name(Y_train, cfg.DATA.TRAIN.MASK_PATH, cfg.DATA.W_FOREGROUND, cfg.DATA.W_BACKGROUND,
                              save_dir=cfg.PATHS.PROB_MAP_DIR)

    f_name = ImageDataGenerator if cfg.PROBLEM.NDIM == '2D' else VoxelDataGenerator

    dic = dict(X=X_train, Y=Y_train, batch_size=cfg.TRAIN.BATCH_SIZE, seed=cfg.SYSTEM.SEED,
        shuffle_each_epoch=cfg.AUGMENTOR.SHUFFLE_TRAIN_DATA_EACH_EPOCH, in_memory=cfg.DATA.TRAIN.IN_MEMORY,
        data_paths=[cfg.DATA.TRAIN.PATH, cfg.DATA.TRAIN.MASK_PATH], da=cfg.AUGMENTOR.ENABLE,
        da_prob=cfg.AUGMENTOR.DA_PROB, rotation90=cfg.AUGMENTOR.ROT90, rand_rot=cfg.AUGMENTOR.RANDOM_ROT,
        rnd_rot_range=cfg.AUGMENTOR.RANDOM_ROT_RANGE, shear=cfg.AUGMENTOR.SHEAR, shear_range=cfg.AUGMENTOR.SHEAR_RANGE,
        zoom=cfg.AUGMENTOR.ZOOM, zoom_range=cfg.AUGMENTOR.ZOOM_RANGE, shift=cfg.AUGMENTOR.SHIFT,
        shift_range=cfg.AUGMENTOR.SHIFT_RANGE, vflip=cfg.AUGMENTOR.VFLIP, hflip=cfg.AUGMENTOR.HFLIP,
        elastic=cfg.AUGMENTOR.ELASTIC, e_alpha=cfg.AUGMENTOR.E_ALPHA, e_sigma=cfg.AUGMENTOR.E_SIGMA,
        e_mode=cfg.AUGMENTOR.E_MODE, g_blur=cfg.AUGMENTOR.G_BLUR, g_sigma=cfg.AUGMENTOR.G_SIGMA,
        median_blur=cfg.AUGMENTOR.MEDIAN_BLUR, mb_kernel=cfg.AUGMENTOR.MB_KERNEL, motion_blur=cfg.AUGMENTOR.MOTION_BLUR,
        motb_k_range=cfg.AUGMENTOR.MOTB_K_RANGE, gamma_contrast=cfg.AUGMENTOR.GAMMA_CONTRAST,
        gc_gamma=cfg.AUGMENTOR.GC_GAMMA, brightness=cfg.AUGMENTOR.BRIGHTNESS,
        brightness_factor=cfg.AUGMENTOR.BRIGHTNESS_FACTOR, contrast=cfg.AUGMENTOR.CONTRAST,
        contrast_factor=cfg.AUGMENTOR.CONTRAST_FACTOR, dropout=cfg.AUGMENTOR.DROPOUT, drop_range=cfg.AUGMENTOR.DROP_RANGE,
        cutout=cfg.AUGMENTOR.CUTOUT, cout_nb_iterations=cfg.AUGMENTOR.COUT_NB_ITERATIONS,
        cout_size=cfg.AUGMENTOR.COUT_SIZE, cout_cval=cfg.AUGMENTOR.COUT_CVAL,
        cout_apply_to_mask=cfg.AUGMENTOR.COUT_APPLY_TO_MASK, cutblur=cfg.AUGMENTOR.CUTBLUR,
        cblur_size=cfg.AUGMENTOR.CBLUR_SIZE, cblur_down_range=cfg.AUGMENTOR.CBLUR_DOWN_RANGE,
        cblur_inside=cfg.AUGMENTOR.CBLUR_INSIDE, cutmix=cfg.AUGMENTOR.CUTMIX, cmix_size=cfg.AUGMENTOR.CMIX_SIZE,
        cutnoise=cfg.AUGMENTOR.CUTNOISE, cnoise_size=cfg.AUGMENTOR.CNOISE_SIZE,
        cnoise_nb_iterations=cfg.AUGMENTOR.CNOISE_NB_ITERATIONS, cnoise_scale=cfg.AUGMENTOR.CNOISE_SCALE,
        misalignment=cfg.AUGMENTOR.MISALIGNMENT, ms_displacement=cfg.AUGMENTOR.MS_DISPLACEMENT,
        ms_rotate_ratio=cfg.AUGMENTOR.MS_ROTATE_RATIO, missing_parts=cfg.AUGMENTOR.MISSING_PARTS,
        missp_iterations=cfg.AUGMENTOR.MISSP_ITERATIONS, shape=cfg.DATA.PATCH_SIZE,
        random_crops_in_DA=cfg.DATA.EXTRACT_RANDOM_PATCH, prob_map=prob_map, n_classes=cfg.MODEL.N_CLASSES,
        extra_data_factor=cfg.DATA.TRAIN.REPLICATE)
    if cfg.PROBLEM.NDIM == '3D':
        dic['zflip'] = cfg.AUGMENTOR.ZFLIP

    train_generator = f_name(**dic)

    val_generator = f_name(X=X_val, Y=Y_val, batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle_each_epoch=cfg.AUGMENTOR.SHUFFLE_VAL_DATA_EACH_EPOCH, in_memory=cfg.DATA.VAL.IN_MEMORY,
        data_paths=[cfg.DATA.VAL.PATH, cfg.DATA.VAL.MASK_PATH], da=False, shape=cfg.DATA.PATCH_SIZE,
        random_crops_in_DA=cfg.DATA.EXTRACT_RANDOM_PATCH, val=True, n_classes=cfg.MODEL.N_CLASSES, seed=cfg.SYSTEM.SEED)

    # Generate examples of data augmentation
    if cfg.AUGMENTOR.AUG_SAMPLES:
        train_generator.get_transformed_samples(
            cfg.AUGMENTOR.AUG_NUM_SAMPLES, save_to_dir=True, train=False, out_dir=cfg.PATHS.DA_SAMPLES)

    return train_generator, val_generator


def create_test_augmentor(cfg, X_test, Y_test):
    """Create test data generator.

       Parameters
       ----------
       cfg : YACS CN object
           Configuration.

       X_test : 4D Numpy array
           Test data. E.g. ``(num_of_images, x, y, channels)``.

       Y_test : 4D Numpy array
           Test data mask. E.g. ``(num_of_images, x, y, 1)``.

       Returns
       -------
       test_generator : simple_data_generator
           Test data generator.
    """
    test_generator = simple_data_generator(X=X_test, d_path=cfg.DATA.TEST.PATH, provide_Y=cfg.DATA.TEST.LOAD_GT, Y=Y_test,
        dm_path=cfg.DATA.TEST.MASK_PATH, batch_size=1, dims=cfg.PROBLEM.NDIM, seed=cfg.SYSTEM.SEED)
    return test_generator


def check_generator_consistence(gen, data_out_dir, mask_out_dir, filenames=None):
    """Save all data of a generator in the given path.

       Parameters
       ----------
       gen : ImageDataGenerator (2D) or VoxelDataGenerator (3D)
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
    if filenames is None:
        count = 0
    for i in tqdm(range(len(gen))):
        batch = next(it)

        X_test, Y_test = batch
        for j in tqdm(range(X_test.shape[0]), leave=False):
            fil = filenames[(i*X.shape[0])+j:(i*X.shape[0])+j+1] if filenames is not None else [str(count)+".tif"]
            save_tif(np.expand_dims(X_test[j],0), data_out_dir, fil, verbose=False)
            save_tif(np.expand_dims(Y_test[j],0), mask_out_dir, fil, verbose=False)
            count += 1

