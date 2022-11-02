import os
import numpy as np
from tqdm import tqdm

from utils.util import  save_tif
from data.pre_processing import calculate_2D_volume_prob_map, calculate_3D_volume_prob_map, save_tif
from data.generators.data_2D_generator import ImageDataGenerator
from data.generators.data_2D_generator_img_pair import PairImageDataGenerator
from data.generators.data_2D_generator_classification import ClassImageDataGenerator
from data.generators.data_3D_generator import VoxelDataGenerator
from data.generators.simple_data_generators import simple_data_generator


def create_train_val_augmentors(cfg, X_train, Y_train, X_val, Y_val):
    """Create training and validation generators.

       Parameters
       ----------
       cfg : YACS CN object
           Configuration.

       X_train : 4D Numpy array
           Training data. E.g. ``(num_of_images, y, x, channels)``.

       Y_train : 4D Numpy array
           Training data mask. E.g. ``(num_of_images, y, x, 1)``.

       X_val : 4D Numpy array
           Validation data mask. E.g. ``(num_of_images, y, x, channels)``.

       Y_val : 4D Numpy array
           Validation data mask. E.g. ``(num_of_images, y, x, 1)``.

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

    # Normalization checks
    custom_mean, custom_std = None, None
    if cfg.DATA.NORMALIZATION.TYPE == 'custom':
        if cfg.DATA.NORMALIZATION.CUSTOM_MEAN == -1 and cfg.DATA.NORMALIZATION.CUSTOM_STD == -1:
            print("Train/Val normalization: trying to load mean and std from {}".format(cfg.PATHS.MEAN_INFO_FILE))
            print("Train/Val normalization: trying to load std from {}".format(cfg.PATHS.STD_INFO_FILE))
            if not os.path.exists(cfg.PATHS.MEAN_INFO_FILE) or not os.path.exists(cfg.PATHS.STD_INFO_FILE):
                print("Train/Val normalization: mean and/or std files not found. Calculating it for the first time")
                custom_mean = np.mean(X_train)
                custom_std = np.std(X_train)
                os.makedirs(os.path.dirname(cfg.PATHS.MEAN_INFO_FILE), exist_ok=True)
                np.save(cfg.PATHS.MEAN_INFO_FILE, custom_mean)
                np.save(cfg.PATHS.STD_INFO_FILE, custom_std)
            else:
                custom_mean = np.load(cfg.PATHS.MEAN_INFO_FILE)
                custom_std = np.load(cfg.PATHS.STD_INFO_FILE)
                print("Train/Val normalization values loaded!")
        else:
            custom_mean = cfg.DATA.NORMALIZATION.CUSTOM_MEAN
            custom_std = cfg.DATA.NORMALIZATION.CUSTOM_STD
        print("Train/Val normalization: using mean {} and std: {}".format(custom_mean, custom_std))

    if cfg.PROBLEM.NDIM == '2D':
        if cfg.PROBLEM.TYPE == 'CLASSIFICATION':
            f_name = ClassImageDataGenerator
        elif cfg.PROBLEM.TYPE in ['SUPER_RESOLUTION', 'SELF_SUPERVISED']:
            f_name = PairImageDataGenerator
        else: # Semantic/Instance segmentation and Denoising
            f_name = ImageDataGenerator 
    else:
        f_name = VoxelDataGenerator
    
    ndim = 3 if cfg.PROBLEM.NDIM == "3D" else 2
    if cfg.PROBLEM.TYPE != 'CLASSIFICATION':
        dic = dict(ndim=ndim, X=X_train, Y=Y_train, batch_size=cfg.TRAIN.BATCH_SIZE, seed=cfg.SYSTEM.SEED,
            shuffle_each_epoch=cfg.AUGMENTOR.SHUFFLE_TRAIN_DATA_EACH_EPOCH, in_memory=cfg.DATA.TRAIN.IN_MEMORY,
            data_paths=[cfg.DATA.TRAIN.PATH, cfg.DATA.TRAIN.MASK_PATH], da=cfg.AUGMENTOR.ENABLE,
            da_prob=cfg.AUGMENTOR.DA_PROB, rotation90=cfg.AUGMENTOR.ROT90, rand_rot=cfg.AUGMENTOR.RANDOM_ROT,
            rnd_rot_range=cfg.AUGMENTOR.RANDOM_ROT_RANGE, shear=cfg.AUGMENTOR.SHEAR, shear_range=cfg.AUGMENTOR.SHEAR_RANGE,
            zoom=cfg.AUGMENTOR.ZOOM, zoom_range=cfg.AUGMENTOR.ZOOM_RANGE, shift=cfg.AUGMENTOR.SHIFT,
            affine_mode=cfg.AUGMENTOR.AFFINE_MODE, shift_range=cfg.AUGMENTOR.SHIFT_RANGE, vflip=cfg.AUGMENTOR.VFLIP,
            hflip=cfg.AUGMENTOR.HFLIP, elastic=cfg.AUGMENTOR.ELASTIC, e_alpha=cfg.AUGMENTOR.E_ALPHA,
            e_sigma=cfg.AUGMENTOR.E_SIGMA, e_mode=cfg.AUGMENTOR.E_MODE, g_blur=cfg.AUGMENTOR.G_BLUR,
            g_sigma=cfg.AUGMENTOR.G_SIGMA, median_blur=cfg.AUGMENTOR.MEDIAN_BLUR, mb_kernel=cfg.AUGMENTOR.MB_KERNEL,
            motion_blur=cfg.AUGMENTOR.MOTION_BLUR, motb_k_range=cfg.AUGMENTOR.MOTB_K_RANGE,
            gamma_contrast=cfg.AUGMENTOR.GAMMA_CONTRAST, gc_gamma=cfg.AUGMENTOR.GC_GAMMA, brightness=cfg.AUGMENTOR.BRIGHTNESS,
            brightness_factor=cfg.AUGMENTOR.BRIGHTNESS_FACTOR, brightness_mode=cfg.AUGMENTOR.BRIGHTNESS_MODE,
            contrast=cfg.AUGMENTOR.CONTRAST, contrast_factor=cfg.AUGMENTOR.CONTRAST_FACTOR,
            contrast_mode=cfg.AUGMENTOR.CONTRAST_MODE, brightness_em=cfg.AUGMENTOR.BRIGHTNESS_EM,
            brightness_em_factor=cfg.AUGMENTOR.BRIGHTNESS_EM_FACTOR, brightness_em_mode=cfg.AUGMENTOR.BRIGHTNESS_EM_MODE,
            contrast_em=cfg.AUGMENTOR.CONTRAST_EM, contrast_em_factor=cfg.AUGMENTOR.CONTRAST_EM_FACTOR,
            contrast_em_mode=cfg.AUGMENTOR.CONTRAST_EM_MODE, dropout=cfg.AUGMENTOR.DROPOUT,
            drop_range=cfg.AUGMENTOR.DROP_RANGE, cutout=cfg.AUGMENTOR.CUTOUT,
            cout_nb_iterations=cfg.AUGMENTOR.COUT_NB_ITERATIONS, cout_size=cfg.AUGMENTOR.COUT_SIZE,
            cout_cval=cfg.AUGMENTOR.COUT_CVAL, cout_apply_to_mask=cfg.AUGMENTOR.COUT_APPLY_TO_MASK,
            cutblur=cfg.AUGMENTOR.CUTBLUR, cblur_size=cfg.AUGMENTOR.CBLUR_SIZE, cblur_down_range=cfg.AUGMENTOR.CBLUR_DOWN_RANGE,
            cblur_inside=cfg.AUGMENTOR.CBLUR_INSIDE, cutmix=cfg.AUGMENTOR.CUTMIX, cmix_size=cfg.AUGMENTOR.CMIX_SIZE,
            cutnoise=cfg.AUGMENTOR.CUTNOISE, cnoise_size=cfg.AUGMENTOR.CNOISE_SIZE,
            cnoise_nb_iterations=cfg.AUGMENTOR.CNOISE_NB_ITERATIONS, cnoise_scale=cfg.AUGMENTOR.CNOISE_SCALE,
            misalignment=cfg.AUGMENTOR.MISALIGNMENT, ms_displacement=cfg.AUGMENTOR.MS_DISPLACEMENT,
            ms_rotate_ratio=cfg.AUGMENTOR.MS_ROTATE_RATIO, missing_sections=cfg.AUGMENTOR.MISSING_SECTIONS,
            missp_iterations=cfg.AUGMENTOR.MISSP_ITERATIONS, grayscale=cfg.AUGMENTOR.GRAYSCALE,
            channel_shuffle=cfg.AUGMENTOR.CHANNEL_SHUFFLE, gridmask=cfg.AUGMENTOR.GRIDMASK,
            grid_ratio=cfg.AUGMENTOR.GRID_RATIO, grid_d_range=cfg.AUGMENTOR.GRID_D_RANGE, grid_rotate=cfg.AUGMENTOR.GRID_ROTATE,
            grid_invert=cfg.AUGMENTOR.GRID_INVERT, shape=cfg.DATA.PATCH_SIZE, resolution=cfg.DATA.TRAIN.RESOLUTION,
            random_crops_in_DA=cfg.DATA.EXTRACT_RANDOM_PATCH, prob_map=prob_map, n_classes=cfg.MODEL.N_CLASSES,
            extra_data_factor=cfg.DATA.TRAIN.REPLICATE, norm_custom_mean=custom_mean, norm_custom_std=custom_std)
        if cfg.PROBLEM.NDIM == '3D':
            dic['zflip'] = cfg.AUGMENTOR.ZFLIP

        if cfg.PROBLEM.TYPE == 'INSTANCE_SEG':
            dic['instance_problem'] = True
        elif cfg.PROBLEM.TYPE == 'SUPER_RESOLUTION':
            dic['random_crop_scale'] = cfg.AUGMENTOR.RANDOM_CROP_SCALE
        elif cfg.PROBLEM.TYPE == 'DENOISING':
            dic['n2v']=True
            dic['n2v_perc_pix'] = cfg.PROBLEM.DENOISING.N2V_PERC_PIX
            dic['n2v_manipulator'] = cfg.PROBLEM.DENOISING.N2V_MANIPULATOR
            dic['n2v_neighborhood_radius'] = cfg.PROBLEM.DENOISING.N2V_NEIGHBORHOOD_RADIUS
            dic['n2v_structMask'] = np.array([[0,1,1,1,1,1,1,1,1,1,0]]) if cfg.PROBLEM.DENOISING.N2V_STRUCTMASK else None
    else:
        r_shape = (224,224)+(cfg.DATA.PATCH_SIZE[-1],) if cfg.MODEL.ARCHITECTURE == 'EfficientNetB0' else None
        dic = dict(X=X_train, Y=Y_train, data_path=cfg.DATA.TRAIN.PATH, n_classes=cfg.MODEL.N_CLASSES,
            batch_size=cfg.TRAIN.BATCH_SIZE, seed=cfg.SYSTEM.SEED, shuffle_each_epoch=cfg.AUGMENTOR.SHUFFLE_TRAIN_DATA_EACH_EPOCH,
            da=cfg.AUGMENTOR.ENABLE, in_memory=cfg.DATA.TRAIN.IN_MEMORY, da_prob=cfg.AUGMENTOR.DA_PROB,
            rotation90=cfg.AUGMENTOR.ROT90, rand_rot=cfg.AUGMENTOR.RANDOM_ROT, rnd_rot_range=cfg.AUGMENTOR.RANDOM_ROT_RANGE,
            shear=cfg.AUGMENTOR.SHEAR, shear_range=cfg.AUGMENTOR.SHEAR_RANGE, zoom=cfg.AUGMENTOR.ZOOM,
            zoom_range=cfg.AUGMENTOR.ZOOM_RANGE, shift=cfg.AUGMENTOR.SHIFT, shift_range=cfg.AUGMENTOR.SHIFT_RANGE,
            affine_mode=cfg.AUGMENTOR.AFFINE_MODE, vflip=cfg.AUGMENTOR.VFLIP, hflip=cfg.AUGMENTOR.HFLIP,
            elastic=cfg.AUGMENTOR.ELASTIC, e_alpha=cfg.AUGMENTOR.E_ALPHA, e_sigma=cfg.AUGMENTOR.E_SIGMA,
            e_mode=cfg.AUGMENTOR.E_MODE, g_blur=cfg.AUGMENTOR.G_BLUR, g_sigma=cfg.AUGMENTOR.G_SIGMA,
            median_blur=cfg.AUGMENTOR.MEDIAN_BLUR, mb_kernel=cfg.AUGMENTOR.MB_KERNEL, motion_blur=cfg.AUGMENTOR.MOTION_BLUR,
            motb_k_range=cfg.AUGMENTOR.MOTB_K_RANGE, gamma_contrast=cfg.AUGMENTOR.GAMMA_CONTRAST,
            gc_gamma=cfg.AUGMENTOR.GC_GAMMA, dropout=cfg.AUGMENTOR.DROPOUT, drop_range=cfg.AUGMENTOR.DROP_RANGE,
            resize_shape=r_shape)

    print("Initializing train data generator . . .")
    train_generator = f_name(**dic)

    print("Initializing val data generator . . .")
    if cfg.PROBLEM.TYPE != 'CLASSIFICATION':
        dic = dict(ndim=ndim, X=X_val, Y=Y_val, batch_size=cfg.TRAIN.BATCH_SIZE,
            shuffle_each_epoch=cfg.AUGMENTOR.SHUFFLE_VAL_DATA_EACH_EPOCH, in_memory=cfg.DATA.VAL.IN_MEMORY,
            data_paths=[cfg.DATA.VAL.PATH, cfg.DATA.VAL.MASK_PATH], da=False, shape=cfg.DATA.PATCH_SIZE,
            random_crops_in_DA=cfg.DATA.EXTRACT_RANDOM_PATCH, val=True, n_classes=cfg.MODEL.N_CLASSES, 
            seed=cfg.SYSTEM.SEED, norm_custom_mean=custom_mean, norm_custom_std=custom_std)
        if cfg.PROBLEM.TYPE == 'SUPER_RESOLUTION':
            dic['random_crop_scale'] = cfg.AUGMENTOR.RANDOM_CROP_SCALE
        if cfg.PROBLEM.TYPE == 'DENOISING':
            dic['n2v'] = True
            dic['n2v_perc_pix'] = cfg.PROBLEM.DENOISING.N2V_PERC_PIX
            dic['n2v_manipulator'] = cfg.PROBLEM.DENOISING.N2V_MANIPULATOR
            dic['n2v_neighborhood_radius'] = cfg.PROBLEM.DENOISING.N2V_NEIGHBORHOOD_RADIUS
        val_generator = f_name(**dic)
    else:
        val_generator = f_name(X=X_val, Y=Y_val, data_path=cfg.DATA.VAL.PATH, n_classes=cfg.MODEL.N_CLASSES, in_memory=cfg.DATA.VAL.IN_MEMORY,
            batch_size=cfg.TRAIN.BATCH_SIZE, seed=cfg.SYSTEM.SEED, shuffle_each_epoch=cfg.AUGMENTOR.SHUFFLE_VAL_DATA_EACH_EPOCH, da=False)


    # Generate examples of data augmentation
    if cfg.AUGMENTOR.AUG_SAMPLES:
        print("Creating generator samples . . .")
        train_generator.get_transformed_samples(
            cfg.AUGMENTOR.AUG_NUM_SAMPLES, save_to_dir=True, train=False, out_dir=cfg.PATHS.DA_SAMPLES,
            draw_grid=cfg.AUGMENTOR.DRAW_GRID)

    return train_generator, val_generator


def create_test_augmentor(cfg, X_test, Y_test):
    """Create test data generator.

       Parameters
       ----------
       cfg : YACS CN object
           Configuration.

       X_test : 4D Numpy array
           Test data. E.g. ``(num_of_images, y, x, channels)``.

       Y_test : 4D Numpy array
           Test data mask. E.g. ``(num_of_images, y, x, 1)``.

       Returns
       -------
       test_generator : simple_data_generator
           Test data generator.
    """
    custom_mean, custom_std = None, None
    if cfg.DATA.NORMALIZATION.TYPE == 'custom':
        if cfg.DATA.NORMALIZATION.CUSTOM_MEAN == -1 and cfg.DATA.NORMALIZATION.CUSTOM_STD == -1:
            print("Test normalization: trying to load mean and std from {}".format(cfg.PATHS.MEAN_INFO_FILE))
            print("Test normalization: trying to load std from {}".format(cfg.PATHS.STD_INFO_FILE))
            if not os.path.exists(cfg.PATHS.MEAN_INFO_FILE) or not os.path.exists(cfg.PATHS.STD_INFO_FILE):
                raise FileNotFoundError("Not mean/std files found in {} and {}"
                    .format(cfg.PATHS.MEAN_INFO_FILE, cfg.PATHS.STD_INFO_FILE))
            custom_mean = np.load(cfg.PATHS.MEAN_INFO_FILE)
            custom_std = np.load(cfg.PATHS.STD_INFO_FILE)
        else:
            custom_mean = cfg.DATA.NORMALIZATION.CUSTOM_MEAN
            custom_std = cfg.DATA.NORMALIZATION.CUSTOM_STD
        print("Test normalization: using mean {} and std: {}".format(custom_mean, custom_std))

    if cfg.PROBLEM.TYPE == 'CLASSIFICATION':
        test_generator = ClassImageDataGenerator(X=X_test, Y=Y_test, data_path=cfg.DATA.TEST.PATH,
            n_classes=cfg.MODEL.N_CLASSES, in_memory=cfg.DATA.VAL.IN_MEMORY, batch_size=X_test.shape[0],
            seed=cfg.SYSTEM.SEED, shuffle_each_epoch=False, da=False)
    else:
        instance_problem = True if cfg.PROBLEM.TYPE == 'INSTANCE_SEG' else False
        dic = dict(X=X_test, d_path=cfg.DATA.TEST.PATH, provide_Y=cfg.DATA.TEST.LOAD_GT, Y=Y_test,
            dm_path=cfg.DATA.TEST.MASK_PATH, batch_size=1, dims=cfg.PROBLEM.NDIM, seed=cfg.SYSTEM.SEED,
            instance_problem=instance_problem, norm_custom_mean=custom_mean, norm_custom_std=custom_std)
        test_generator = simple_data_generator(**dic)
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
            fil = filenames[(i*X_test.shape[0])+j:(i*X_test.shape[0])+j+1] if filenames is not None else [str(count)+".tif"]
            save_tif(np.expand_dims(X_test[j],0), data_out_dir, fil, verbose=False)
            save_tif(np.expand_dims(Y_test[j],0), mask_out_dir, fil, verbose=False)
            count += 1

