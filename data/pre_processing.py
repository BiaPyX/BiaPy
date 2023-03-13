import os
import scipy
import numpy as np
from tqdm import tqdm
import pandas as pd
from skimage.segmentation import clear_border, find_boundaries
from skimage.io import imread
from scipy.ndimage.morphology import binary_dilation  as binary_dilation_scipy
from scipy.ndimage.measurements import center_of_mass                                                  
from skimage.morphology import disk, dilation, erosion, binary_erosion, binary_dilation
from skimage.measure import label, regionprops
from skimage.transform import resize
from skimage.feature import peak_local_max

from utils.util import load_data_from_dir, load_3d_images_from_dir, save_tif


#########################
# INSTANCE SEGMENTATION #
#########################
def create_instance_channels(cfg, data_type='train'):
    """Create training and validation new data with appropiate channels based on ``PROBLEM.INSTANCE_SEG.DATA_CHANNELS`` for instance
       segmentation.

       Parameters
       ----------
       cfg : YACS CN object
           Configuration.

	   data_type: str, optional
		   Wheter to create training or validation instance channels.

       Returns
       -------
       filenames: List of str
           Image paths.
    """

    assert data_type in ['train', 'val']

    f_name = load_data_from_dir if cfg.PROBLEM.NDIM == '2D' else load_3d_images_from_dir
    tag = "TRAIN" if data_type == "train" else "VAL"
    Y, _, _, filenames = f_name(getattr(cfg.DATA, tag).MASK_PATH, return_filenames=True)
    print("Creating Y_{} channels . . .".format(data_type))
    if isinstance(Y, list):
        for i in tqdm(range(len(Y))):
            Y[i] = labels_into_channels(Y[i], mode=cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS, save_dir=getattr(cfg.PATHS, tag+'_INSTANCE_CHANNELS_CHECK'),
                          fb_mode=cfg.PROBLEM.INSTANCE_SEG.DATA_CONTOUR_MODE)
    else:
        Y = labels_into_channels(Y, mode=cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS, save_dir=getattr(cfg.PATHS, tag+'_INSTANCE_CHANNELS_CHECK'),
                   fb_mode=cfg.PROBLEM.INSTANCE_SEG.DATA_CONTOUR_MODE)
    save_tif(Y, data_dir=getattr(cfg.DATA, tag).INSTANCE_CHANNELS_MASK_DIR, filenames=filenames, verbose=cfg.TEST.VERBOSE)
    X, _, _, filenames = f_name(getattr(cfg.DATA, tag).PATH, return_filenames=True)
    print("Creating X_{} channels . . .".format(data_type))
    save_tif(X, data_dir=getattr(cfg.DATA, tag).INSTANCE_CHANNELS_DIR, filenames=filenames, verbose=cfg.TEST.VERBOSE)
    
    # Save original X data with the labels 
    for i in range(min(3,len(X))):
        if isinstance(X, list):
            save_tif(X[i], getattr(cfg.PATHS, tag+'_INSTANCE_CHANNELS_CHECK'), filenames=['vol'+str(i)+".tif"], verbose=False)
        else:
            save_tif(np.expand_dims(X[i],0), getattr(cfg.PATHS, tag+'_INSTANCE_CHANNELS_CHECK'), filenames=['vol'+str(i)+".tif"], verbose=False)

def create_test_instance_channels(cfg):
    """Create test new data with appropiate channels based on ``PROBLEM.INSTANCE_SEG.DATA_CHANNELS`` for instance segmentation.

       Parameters
       ----------
       cfg : YACS CN object
           Configuration.
    """

    f_name = load_data_from_dir if cfg.PROBLEM.NDIM == '2D' else load_3d_images_from_dir

    if cfg.DATA.TEST.LOAD_GT:
        Y_test, _, _, test_filenames = f_name(cfg.DATA.TEST.MASK_PATH, return_filenames=True)
        print("Creating Y_test channels . . .")
        if isinstance(Y_test, list):
            for i in tqdm(range(len(Y_test))):
                Y_test[i] = labels_into_channels(Y_test[i], mode=cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS, save_dir=cfg.PATHS.TEST_INSTANCE_CHANNELS_CHECK,
                                            fb_mode=cfg.PROBLEM.INSTANCE_SEG.DATA_CONTOUR_MODE)
        else:
            Y_test = labels_into_channels(Y_test, mode=cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS, save_dir=cfg.PATHS.TEST_INSTANCE_CHANNELS_CHECK,
                                     fb_mode=cfg.PROBLEM.INSTANCE_SEG.DATA_CONTOUR_MODE)
        save_tif(Y_test, data_dir=cfg.DATA.TEST.INSTANCE_CHANNELS_MASK_DIR, filenames=test_filenames, verbose=cfg.TEST.VERBOSE)

    print("Creating X_test channels . . .")
    X_test, _, _, test_filenames = f_name(cfg.DATA.TEST.PATH, return_filenames=True)
    save_tif(X_test, data_dir=cfg.DATA.TEST.INSTANCE_CHANNELS_DIR, filenames=test_filenames, verbose=cfg.TEST.VERBOSE)
    
    # Save original X data with the labels 
    for i in range(min(3,len(X_test))):
        if isinstance(X_test, list):
            save_tif(X_test[i], cfg.PATHS.TEST_INSTANCE_CHANNELS_CHECK, filenames=['vol'+str(i)+".tif"], verbose=True)
        else:
            save_tif(np.expand_dims(X_test[i],0), cfg.PATHS.TEST_INSTANCE_CHANNELS_CHECK, filenames=['vol'+str(i)+".tif"], verbose=True)

def labels_into_channels(data_mask, mode="BC", fb_mode="outer", save_dir=None):
    """Coverts input semantic or instance segmentation data masks into different binary channels to train an instance segmentation
       problem. 

       Parameters
       ----------
       data_mask : 4D/5D Numpy array
           Data mask to create the new array from. It is expected to have just one channel. E.g. ``(10, 200, 1000, 1000, 1)``

       mode : str, optional
           Operation mode. Possible values: ``BC`` and ``BCD``.  ``BC`` corresponds to use binary segmentation+contour.
           ``BCD`` stands for binary segmentation+contour+distances.

       fb_mode : str, optional
          Mode of the find_boundaries function from ``scikit-image`` or "dense". More info in:
          `find_boundaries() <https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.find_boundaries>`_. 
          Choose "dense" to label as contour every pixel that is not in ``B`` channel. 

       save_dir : str, optional
           Path to store samples of the created array just to debug it is correct.

       Returns
       -------
       new_mask : 5D Numpy array
           5D array with 3 channels instead of one. E.g. ``(10, 200, 1000, 1000, 3)``
    """

    assert data_mask.ndim in [5, 4]
    assert mode in ['BC', 'BCM', 'BCD', 'BD', 'BCDv2', 'Dv2', 'BDv2', 'BP']

    d_shape = 4 if data_mask.ndim == 5 else 3
    if mode in ['BCDv2', 'Dv2', 'BDv2']:
        c_number = 4
    elif mode in ['BCD', 'BCM']:
        c_number = 3
    elif mode in ['BC', 'BP', 'BD']:
        c_number = 2

    if 'D' in mode:
        dtype = np.float32  
    else:
        dtype = np.uint8

    new_mask = np.zeros(data_mask.shape[:d_shape] + (c_number,), dtype=dtype)
    for img in tqdm(range(data_mask.shape[0])):
        vol = data_mask[img,...,0].astype(np.int64)
        instances = np.unique(vol)
        instance_count = len(instances)

        # Background distance
        if 'Dv2' in mode:
            # Background distance
            vol_b_dist = np.invert(vol>0)
            vol_b_dist= scipy.ndimage.distance_transform_edt(vol_b_dist)
            vol_b_dist = np.max(vol_b_dist)-vol_b_dist
            new_mask[img,...,3] = vol_b_dist.copy()

        # Semantic mask
        if 'B' in mode and instance_count != 1:
            new_mask[img,...,0] = (vol>0).copy().astype(np.uint8)

        # Central points
        if 'P' in mode and instance_count != 1:
            coords = center_of_mass(vol>0, vol, instances[1:])
            coords = np.round(coords).astype(int)
            for coord in coords:
                if data_mask.ndim == 5:
                    z,y,x = coord
                    new_mask[img,z,y,x,1] = 1
                else:
                    y,x = coord
                    new_mask[img,y,x,1] = 1

            if data_mask.ndim == 5:
                for i in range(new_mask.shape[1]):                                                                                  
                    new_mask[img,i,...,1] = dilation(new_mask[img,i,...,1], disk(3)) 
            else:
                new_mask[img,...,1] = dilation(new_mask[img,...,1], disk(3))  

        # Contour
        if ('C' in mode or 'Dv2' in mode) and instance_count != 1: 
            f = "thick" if fb_mode == "dense" else fb_mode
            new_mask[img,...,1] = find_boundaries(vol, mode=f).astype(np.uint8)
            if fb_mode == "dense" and mode != "BCM":
                new_mask[img,...,1] = 1 - binary_dilation(new_mask[img,...,1], disk(1))
                new_mask[img,...,1] = 1 - ( (vol>0) * new_mask[img,...,1])
            if 'B' in mode:
                # Remove contours from segmentation maps
                new_mask[img,...,0][np.where(new_mask[img,...,1] == 1)] = 0
            if mode == "BCM":
                new_mask[img,...,2] = (vol>0).astype(np.uint8)

        if ('D' in mode or 'Dv2' in mode) and instance_count != 1:
            # Foreground distance
            new_mask[img,...,-1] = scipy.ndimage.distance_transform_edt(new_mask[img,...,0])
            props = regionprops(vol, new_mask[img,...,-1])
            max_values = np.zeros(vol.shape)
            for i in range(len(props)):
                max_values = np.where(vol==props[i].label, props[i].intensity_max, max_values)     
            new_mask[img,...,-1] = max_values - new_mask[img,...,-1]

    # Normalize and merge distance channels
    if 'Dv2' in mode:
        # Normalize background   
        b_min = np.min(new_mask[...,3])
        b_max = np.max(new_mask[...,3])    
        new_mask[...,3] = (new_mask[...,3]-b_min)/(b_max-b_min)

        if instance_count != 1: 
            # Normalize foreground  
            f_min = np.min(new_mask[...,2])
            f_max = np.max(new_mask[...,2])
            new_mask[...,2] = (new_mask[...,2]-f_min)/(f_max-f_min)

            new_mask[...,2] = new_mask[...,3] - new_mask[...,2]

            # The intersection of the channels is the contour channel, so set it to the maximum value 1
            new_mask[...,2][new_mask[...,1]>0] = 1

        if mode == "BCDv2":
            new_mask = new_mask[...,:3]
        elif mode == "BDv2":
            new_mask = new_mask[...,[0,-1]]   
        elif mode == "Dv2":
            new_mask = np.expand_dims(new_mask[...,2], -1)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        suffix = []
        if mode == "Dv2":
            suffix.append('_distance.tif')
        else:
            suffix.append('_semantic.tif')
        if mode in ["BC", "BCM", "BCD", "BCDv2"]:
            suffix.append('_contour.tif')
            if mode in ["BCD", "BCDv2"]:
                suffix.append('_distance.tif')
            elif mode == "BCM":
                suffix.append('_binary_mask.tif')
        elif mode == "BP":
            suffix.append('_points.tif')
        elif mode in ["BDv2", "BD"]:
            suffix.append('_distance.tif')

        for i in range(min(3,len(new_mask))):
            for j in range(len(suffix)):
                aux = new_mask[i,...,j]
                aux = np.expand_dims(np.expand_dims(aux,-1),0)
                save_tif(aux, save_dir, filenames=['vol'+str(i)+suffix[j]], verbose=False)
                save_tif(np.expand_dims(data_mask[i],0), save_dir, filenames=['vol'+str(i)+"_y.tif"], verbose=False)
    return new_mask

#############
# DETECTION #
#############
def create_detection_masks(cfg, data_type='train'):
    """Create detection masks based on CSV files.

       Parameters
       ----------
       cfg : YACS CN object
           Configuration.

	   data_type: str, optional
		   Wheter to create train, validation or test masks.
    """

    assert data_type in ['train', 'val', 'test']

    if data_type == "train":
        tag = "TRAIN"
    elif data_type == "val":
        tag = "VAL"
    else:
        tag = "TEST"
    img_dir = getattr(cfg.DATA, tag).PATH
    label_dir = getattr(cfg.DATA, tag).MASK_PATH
    out_dir = getattr(cfg.DATA, tag).DETECTION_MASK_DIR
    img_ids = sorted(next(os.walk(img_dir))[2])
    img_ext = '.'+img_ids[0].split('.')[-1]
    ids = sorted(next(os.walk(label_dir))[2])
    if len(img_ids) != len(ids):
        raise ValueError("Different number of CSV files and images found ({} vs {}). "
            "Please check that every image has one and only one CSV file".format(len(ids), len(img_ids)))
    if cfg.PROBLEM.NDIM == '2D':
        req_dim = 2 
        req_columns = ['axis-0', 'axis-1']
        req_columns_class = ['axis-0', 'axis-1', 'class']
    else:
        req_dim = 3
        req_columns = ['axis-0', 'axis-1', 'axis-2']
        req_columns_class = ['axis-0', 'axis-1', 'axis-2', 'class']

    print("Creating {} detection masks . . .".format(data_type))
    for i in range(len(ids)):
        img_filename = os.path.splitext(ids[i])[0]+img_ext
        if not os.path.exists(os.path.join(out_dir, img_filename)):
            print("Attempting to create mask from CSV file: {}".format(os.path.join(label_dir, ids[i])))
            if not os.path.exists(os.path.join(img_dir, img_filename)):
                print("WARNING: The image seems to have different name than the CSV file. Using the image "
                    "with the same position as the CSV in the directory. Check if it is correct!")
                img_filename = img_ids[i]
            print("Its respective image seems to be: {}".format(os.path.join(img_dir, img_filename)))
            
            df = pd.read_csv(os.path.join(label_dir, ids[i]))  
            img = imread(os.path.join(img_dir, img_filename))
            
            # Adjust shape
            img = np.squeeze(img)
            if cfg.PROBLEM.NDIM == '2D':
                if img.ndim == 2:
                    img = np.expand_dims(img, -1)
                else:
                    if img.shape[0] <= 3: img = img.transpose((1,2,0))   
            else: 
                if img.ndim == 3: 
                    img = np.expand_dims(img, -1)
                else:
                    if img.shape[0] <= 3: img = img.transpose((1,2,3,0))

            # Discard first index column to not have error if it is not sorted 
            p_number=df.iloc[: , 0].to_list()
            df = df.iloc[: , 1:]
            df = df.rename(columns=lambda x: x.strip()) # trim spaces in column names
            if len(df.columns) == req_dim+1:
                if not all(df.columns == req_columns_class):
                    raise ValueError("CSV columns need to be {}".format(req_columns_class))
            elif len(df.columns) == req_dim:
                if not all(df.columns == req_columns):
                    raise ValueError("CSV columns need to be {}".format(req_columns))
            else:
                raise ValueError("CSV file {} need to have {} or {} columns. Found {}"
                                .format(os.path.join(label_dir, ids[i]), req_dim, req_dim+1, len(df.columns)))

            # Convert them to int in case they are floats
            df['axis-0'] = df['axis-0'].astype('int')
            df['axis-1'] = df['axis-1'].astype('int')
            if cfg.PROBLEM.NDIM == '3D':
                df['axis-2'] = df['axis-2'].astype('int')
            
            df = df.sort_values(by=['axis-0']) 

            # Obtain the points 
            z_axis_point = df['axis-0']                                                                       
            y_axis_point = df['axis-1']    
            if cfg.PROBLEM.NDIM == '3D':                                                                   
                x_axis_point = df['axis-2']
            
            # Class column present
            if len(df.columns) == req_dim+1:
                df['class'] = df['class'].astype('int')
                class_point = np.array(df['class']) 

                uniq = np.sort(np.unique(class_point))        
                if uniq[0] != 1:
                    raise ValueError("Class number must start with 1")    
                if not all(uniq == np.array(range(1,cfg.MODEL.N_CLASSES+1))):
                    raise ValueError("Classes must be consecutive, e.g [1,2,3,4..]. Given {}".format(uniq))   
            else:
                if cfg.MODEL.N_CLASSES > 1:
                    raise ValueError("MODEL.N_CLASSES > 1 but no class specified in CSV files (4th column must have class info)")
                class_point = [1] * len(z_axis_point)

            # Create masks
            print("Creating all points . . .")
            mask = np.zeros((img.shape[:-1]+(cfg.MODEL.N_CLASSES,)), dtype=np.uint8)
            for j in tqdm(range(len(z_axis_point)), total=len(z_axis_point), leave=False):
                a0_coord = z_axis_point[j]
                a1_coord = y_axis_point[j]
                if cfg.PROBLEM.NDIM == '3D':
                    a2_coord = x_axis_point[j]
                c_point = class_point[j]-1

                if c_point+1 > mask.shape[-1]:
                    raise ValueError("Class {} detected while MODEL.N_CLASSES was set to {}. Please check it!"
                        .format(c_point+1, cfg.MODEL.N_CLASSES))

                # Paint the point
                if cfg.PROBLEM.NDIM == '3D':
                    if a0_coord < mask.shape[0] and a1_coord < mask.shape[1] and a2_coord < mask.shape[2]:
                        cpd = cfg.PROBLEM.DETECTION.CENTRAL_POINT_DILATION                                                                                                              
                        if 1 in mask[max(0,a0_coord-1):min(mask.shape[0],a0_coord+2),                                   
                                     max(0,a1_coord-1-cpd):min(mask.shape[1],a1_coord+2+cpd),                                   
                                     max(0,a2_coord-1-cpd):min(mask.shape[2],a2_coord+2+cpd), c_point]: 
                            print("WARNING: possible duplicated point in (3,9,9) neighborhood: coords {} , class {} "
                                  "(point number {} in CSV)".format((a0_coord,a1_coord,a2_coord), c_point, p_number[j]))                                                                                                                                            
                                                                                                                                
                        mask[a0_coord,a1_coord,a2_coord,c_point] = 1                                            
                        if a1_coord+1 < mask.shape[1]: mask[a0_coord,a1_coord+1,a2_coord,c_point] = 1       
                        if a1_coord-1 > 0: mask[a0_coord,a1_coord-1,a2_coord,c_point] = 1                   
                        if a2_coord+1 < mask.shape[2]: mask[a0_coord,a1_coord,a2_coord+1,c_point] = 1       
                        if a2_coord-1 > 0: mask[a0_coord,a1_coord,a2_coord-1,c_point] = 1     
                        if cfg.PROBLEM.DETECTION.CENTRAL_POINT_DILATION == 0:
                            if a1_coord+1 < mask.shape[1] and a2_coord+1 < mask.shape[2]: 
                                mask[a0_coord,a1_coord+1,a2_coord+1,c_point] = 1       
                            if a1_coord-1 > 0 and a2_coord-1 > 0: 
                                mask[a0_coord,a1_coord-1,a2_coord-1,c_point] = 1  
                            if a1_coord-1 > 0 and a2_coord+1 < mask.shape[2]: 
                                mask[a0_coord,a1_coord-1,a2_coord+1,c_point] = 1   
                            if a1_coord+1 < mask.shape[1] and a2_coord-1 > 0: 
                                mask[a0_coord,a1_coord+1,a2_coord-1,c_point] = 1             
                    else:  
                        print("WARNING: discarding point {} which seems to be out of shape: {}"
                              .format([a0_coord,a1_coord,a2_coord], img.shape))                                                                                                     
                else:
                    if a0_coord < mask.shape[0] and a1_coord < mask.shape[1]:                                                                                                              
                        if 1 in mask[max(0,a0_coord-4):min(mask.shape[0],a0_coord+5),                                   
                                    max(0,a1_coord-4):min(mask.shape[1],a1_coord+5), c_point]: 
                            print("WARNING: possible duplicated point in (9,9) neighborhood: coords {} , class {} "
                                  "(point number {} in CSV)".format((a0_coord,a1_coord), c_point, p_number[j]))                                                                                                                                            
                                                                                                                                
                        mask[a0_coord,a1_coord,c_point] = 1                                            
                        if a1_coord+1 < mask.shape[1]: mask[a0_coord,a1_coord+1,c_point] = 1       
                        if a1_coord-1 > 0: mask[a0_coord,a1_coord-1,c_point] = 1                                     
                    else:  
                        print("WARNING: discarding point {} which seems to be out of shape: {}"
                              .format([a0_coord,a1_coord], img.shape))     

            # Dilate the mask
            if cfg.PROBLEM.DETECTION.CENTRAL_POINT_DILATION > 0:
                if cfg.PROBLEM.NDIM == '2D': mask = np.expand_dims(mask,0)
                for k in range(mask.shape[0]): 
                    for ch in range(mask.shape[-1]):                                                                                  
                        mask[k,...,ch] = binary_dilation_scipy(mask[k,...,ch], iterations=1,  structure=disk(cfg.PROBLEM.DETECTION.CENTRAL_POINT_DILATION))                                                                                                                                                    
                if cfg.PROBLEM.NDIM == '2D': mask = mask[0]

            if cfg.PROBLEM.DETECTION.CHECK_POINTS_CREATED:
                print("Check points created to see if some of them are very close that create a large label") 
                error_found = False
                for ch in range(mask.shape[-1]):
                    _, index, counts = np.unique(label(clear_border(mask[...,ch])), return_counts=True, return_index=True)                     
                    # 0 is background so valid element is 1. We will compare that value with the rest                                                                         
                    ref_value = counts[1]                                                                                           
                    for k in range(2,len(counts)):                                                                                  
                        if abs(ref_value - counts[k]) > 5:                                                                          
                            point = np.unravel_index(index[k], mask[...,ch].shape)  
                            print("WARNING: There is a point (coords {}) with size very different from "
                                  "the rest. Maybe that cell has several labels: please check it! Normally all point "
                                  "have {} pixels but this one has {}.".format(point, ref_value, counts[k])) 
                            error_found = True

                if error_found:
                    raise ValueError("Duplicate points have been found so please check them before continuing. "
                                     "If you consider that the points are valid simply disable "
                                     "'PROBLEM.DETECTION.CHECK_POINTS_CREATED' so this check is not done again!")

            save_tif(np.expand_dims(mask,0), out_dir, [img_filename])
        else:
            print("Mask file {} found for CSV file: {}".format(os.path.join(out_dir, img_filename), 
                os.path.join(label_dir, ids[i])))

#######
# SSL #
#######
def create_ssl_source_data_masks(cfg, data_type='train'):
    """Create SSL source data.

       Parameters
       ----------
       cfg : YACS CN object
           Configuration.

	   data_type: str, optional
		   Wheter to create train, validation or test source data.
    """

    assert data_type in ['train', 'val', 'test']
    tag = data_type.upper()
    
    img_dir = getattr(cfg.DATA, tag).PATH
    out_dir = getattr(cfg.DATA, tag).SSL_SOURCE_DIR
    ids = sorted(next(os.walk(img_dir))[2])
    add_noise = True if cfg.PROBLEM.SELF_SUPERVISED.NOISE > 0 else False

    print("Creating {} SSL source. . .".format(data_type))
    for i in range(len(ids)):
        if not os.path.exists(os.path.join(out_dir, ids[i])):
            print("Crappifying file {} to create SSL source".format(os.path.join(img_dir, ids[i])))

            img = imread(os.path.join(img_dir, ids[i]))
            
            # Adjust shape
            img = np.squeeze(img)
            if cfg.PROBLEM.NDIM == '2D':
                if img.ndim == 2:
                    img = np.expand_dims(img, -1)
                else:
                    if img.shape[0] <= 3: img = img.transpose((1,2,0))   
            else: 
                if img.ndim == 3: 
                    img = np.expand_dims(img, -1)
                else:
                    min_val = min(img.shape)
                    channel_pos = img.shape.index(min_val)
                    if channel_pos != 3 and img.shape[channel_pos] <= 4:
                        new_pos = [x for x in range(4) if x != channel_pos]+[channel_pos,]
                        img = img.transpose(new_pos)

            img = crappify(img, resizing_factor=cfg.PROBLEM.SELF_SUPERVISED.RESIZING_FACTOR, 
                add_noise=add_noise, noise_level=cfg.PROBLEM.SELF_SUPERVISED.NOISE)

            save_tif(np.expand_dims(img,0), out_dir, [ids[i]])
        else:
            print("Source file {} found".format(os.path.join(img_dir, ids[i])))

def crappify(input_img, resizing_factor, add_noise=True, noise_level=None, Down_up=True):
    """
    Crappifies input image by adding Gaussian noise and downsampling and upsampling it so the resolution
    gets worsen. 

    input_img : 4D/5D Numpy array
        Data to be modified. E.g. ``(y, x, channels)`` if working with 2D images or
        ``(z, y, x, channels)`` if working with 3D.

    resizing_factor : floats
        Downsizing factor to reshape the image.

    add_noise : boolean, optional
        Indicating whether to add gaussian noise before applying the resizing.

    noise_level: float, optional
        Number between ``[0,1]`` indicating the std of the Gaussian noise N(0,std).

    Down_up : bool, optional
        Indicating whether to perform a final upsampling operation to obtain an image of the 
        same size as the original but with the corresponding loss of quality of downsizing and 
        upsizing.

    Returns
    -------
    img : 4D/5D Numpy array
        Train images. E.g. ``(y, x, channels)`` if working with 2D images or
        ``(z, y, x, channels)`` if working with 3D.
    """
    if input_img.ndim == 3:
        w, h, c = input_img.shape
        org_sz = (w, h)
    else:
        d, w, h, c = input_img.shape
        org_sz = (d, w, h)
        new_d = int(d / np.sqrt(resizing_factor))

    new_w = int(w / np.sqrt(resizing_factor))
    new_h = int(h / np.sqrt(resizing_factor))

    if input_img.ndim == 3:
        targ_sz = (new_w, new_h)
    else:
        targ_sz = (new_d, new_w, new_h)
    
    img = input_img.copy()
    if add_noise:
        img = add_gaussian_noise(img, noise_level)

    img = resize(img, targ_sz, order=1, mode='reflect',
                 clip=True, preserve_range=True, anti_aliasing=False)

    if Down_up:
        img = resize(img, org_sz, order=1, mode='reflect',
                        clip=True, preserve_range=True, anti_aliasing=False)

    return img.astype(input_img.dtype)

def add_gaussian_noise(image, percentage_of_noise):
    """
    Adds Gaussian noise to an input image. 

    Parameters
    ----------
    image : 3D Numpy array
        Image to be added Gaussian Noise with 0 mean and a certain std. E.g. ``(y, x, channels)``.

    percentage_of_noise : float
        percentage of the maximum value of the image that will be used as the std of the Gaussian Noise 
        distribution.

    Returns
    -------
    out : 3D Numpy array
        Transformed image. E.g. ``(y, x, channels)``.
    """
    max_value=np.max(image)
    noise_level=percentage_of_noise*max_value
    noise = np.random.normal(loc=0, scale=noise_level, size=image.shape)
    noisy_img=np.clip(image+noise, 0, max_value).astype(image.dtype)
    return noisy_img

################
# SEMANTIC SEG #
################
def calculate_2D_volume_prob_map(Y, Y_path=None, w_foreground=0.94, w_background=0.06, save_dir=None):
    """Calculate the probability map of the given 2D data.

       Parameters
       ----------
       Y : 4D Numpy array
           Data to calculate the probability map from. E. g. ``(num_of_images, y, x, channel)``

       Y_path : str, optional
           Path to load the data from in case ``Y=None``.

       w_foreground : float, optional
           Weight of the foreground. This value plus ``w_background`` must be equal ``1``.

       w_background : float, optional
           Weight of the background. This value plus ``w_foreground`` must be equal ``1``.

       save_dir : str, optional
           Path to the file where the probability map will be stored.

       Raises
       ------
       ValueError
           if ``Y`` does not have 4 dimensions.

       ValueError
           if ``w_foreground + w_background > 1``.

       Returns
       -------
       Array : Str or 4D Numpy array
           Path where the probability map/s is/are stored if ``Y_path`` was given and there are images of different
           shapes. Otherwise, an array that represents the probability map of ``Y`` or all loaded data files from
           ``Y_path`` will be returned.
    """

    if Y is not None:
        if Y.ndim != 4:
            raise ValueError("'Y' must be a 4D Numpy array")

    if Y is None and Y_path is None:
        raise ValueError("'Y' or 'Y_path' need to be provided")

    if Y is not None:
        prob_map = np.copy(Y).astype(np.float32)
        l = prob_map.shape[0]
        channels = prob_map.shape[-1]
        v = np.max(prob_map)
    else:
        prob_map, _, _ = load_data_from_dir(Y_path)
        l = len(prob_map)
        channels = prob_map[0].shape[-1]
        v = np.max(prob_map[0])

    if isinstance(prob_map, list):
        first_shape = prob_map[0][0].shape
    else:
        first_shape = prob_map[0].shape

    print("Connstructing the probability map . . .")
    maps = []
    diff_shape = False
    for i in tqdm(range(l)):
        if isinstance(prob_map, list):
            _map = prob_map[i][0].copy().astype(np.float32)
        else:
            _map = prob_map[i].copy().astype(np.float32)

        for k in range(channels):
            # Remove artifacts connected to image border
            _map[:,:,k] = clear_border(_map[:,:,k])

            foreground_pixels = (_map[:,:,k] == v).sum()
            background_pixels = (_map[:,:,k] == 0).sum()

            if foreground_pixels == 0:
                _map[:,:,k][np.where(_map[:,:,k] == v)] = 0
            else:
                _map[:,:,k][np.where(_map[:,:,k] == v)] = w_foreground/foreground_pixels
            if background_pixels == 0:
                _map[:,:,k][np.where(_map[:,:,k] == 0)] = 0
            else:
                _map[:,:,k][np.where(_map[:,:,k] == 0)] = w_background/background_pixels

            # Necessary to get all probs sum 1
            s = _map[:,:,k].sum()
            if s == 0:
                t = 1
                for x in _map[:,:,k].shape: t *=x
                _map[:,:,k].fill(1/t)
            else:
                _map[:,:,k] = _map[:,:,k]/_map[:,:,k].sum()

        if first_shape != _map.shape: diff_shape = True
        maps.append(_map)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        if not diff_shape:
            for i in range(len(maps)):
                maps[i] = np.expand_dims(maps[i], 0)
            maps = np.concatenate(maps)
            print("Saving the probability map in {}".format(save_dir))
            np.save(os.path.join(save_dir, 'prob_map.npy'), maps)
            return maps
        else:
            print("As the files loaded have different shapes, the probability map for each one will be stored"
                  " separately in {}".format(save_dir))
            d = len(str(l))
            for i in range(l):
                f = os.path.join(save_dir, 'prob_map'+str(i).zfill(d)+'.npy')
                np.save(f, maps[i])
            return save_dir

def calculate_3D_volume_prob_map(Y, Y_path=None, w_foreground=0.94, w_background=0.06, save_dir=None):
    """Calculate the probability map of the given 3D data.

       Parameters
       ----------
       Y : 5D Numpy array
           Data to calculate the probability map from. E. g. ``(num_subvolumes, z, y, x, channel)``

       Y_path : str, optional
           Path to load the data from in case ``Y=None``.

       w_foreground : float, optional
           Weight of the foreground. This value plus ``w_background`` must be equal ``1``.

       w_background : float, optional
           Weight of the background. This value plus ``w_foreground`` must be equal ``1``.

       save_dir : str, optional
           Path to the directory where the probability map will be stored.

       Returns
       -------
       Array : Str or 5D Numpy array
           Path where the probability map/s is/are stored if ``Y_path`` was given and there are images of different
           shapes. Otherwise, an array that represents the probability map of ``Y`` or all loaded data files from
           ``Y_path`` will be returned.

       Raises
       ------
       ValueError
           if ``Y`` does not have 5 dimensions.
       ValueError
           if ``w_foreground + w_background > 1``.
    """

    if Y is not None:
        if Y.ndim != 5:
            raise ValueError("'Y' must be a 5D Numpy array")

    if Y is None and Y_path is None:
        raise ValueError("'Y' or 'Y_path' need to be provided")

    if Y is not None:
        prob_map = np.copy(Y).astype(np.float32)
        l = prob_map.shape[0]
        channels = prob_map.shape[-1]
        v = np.max(prob_map)
    else:
        prob_map, _, _ = load_3d_images_from_dir(Y_path)
        l = len(prob_map)
        channels = prob_map[0].shape[-1]
        v = np.max(prob_map[0])

    if isinstance(prob_map, list):
        first_shape = prob_map[0][0].shape
    else:
        first_shape = prob_map[0].shape

    print("Constructing the probability map . . .")
    maps = []
    diff_shape = False
    for i in range(l):
        if isinstance(prob_map, list):
            _map = prob_map[i][0].copy().astype(np.float64)
        else:
            _map = prob_map[i].copy().astype(np.float64)

        for k in range(channels):
            for j in range(_map.shape[0]):
                # Remove artifacts connected to image border
                _map[j,:,:,k] = clear_border(_map[j,:,:,k])
            foreground_pixels = (_map[:,:,:,k] == v).sum()
            background_pixels = (_map[:,:,:,k] == 0).sum()

            if foreground_pixels == 0:
                _map[:,:,:,k][np.where(_map[:,:,:,k] == v)] = 0
            else:
                _map[:,:,:,k][np.where(_map[:,:,:,k] == v)] = w_foreground/foreground_pixels
            if background_pixels == 0:
                _map[:,:,:,k][np.where(_map[:,:,:,k] == 0)] = 0
            else:
                _map[:,:,:,k][np.where(_map[:,:,:,k] == 0)] = w_background/background_pixels

            # Necessary to get all probs sum 1
            s = _map[:,:,:,k].sum()
            if s == 0:
                t = 1
                for x in _map[:,:,:,k].shape: t *=x
                _map[:,:,:,k].fill(1/t)
            else:
                _map[:,:,:,k] = _map[:,:,:,k]/_map[:,:,:,k].sum()

        if first_shape != _map.shape: diff_shape = True
        maps.append(_map)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        if not diff_shape:
            for i in range(len(maps)):
                maps[i] = np.expand_dims(maps[i], 0)
            maps = np.concatenate(maps)
            print("Saving the probability map in {}".format(save_dir))
            np.save(os.path.join(save_dir, 'prob_map.npy'), maps)
            return maps
        else:
            print("As the files loaded have different shapes, the probability map for each one will be stored "
                  "separately in {}".format(save_dir))
            d = len(str(l))
            for i in range(l):
                f = os.path.join(save_dir, 'prob_map'+str(i).zfill(d)+'.npy')
                np.save(f, maps[i])
            return save_dir

###########
# GENERAL #
###########
def norm_range01(x):
    norm_steps = {}
    if x.dtype == np.uint8:
        x = x/255
        norm_steps['div_255'] = 1
    else:
        if np.max(x) > 255:
            norm_steps['reduced_{}'.format(x.dtype)] = 1
            x = reduce_dtype(x, 0, 65535, out_min=0, out_max=1, out_type=np.float32)
        elif np.max(x) > 2:
            x = x/255
            norm_steps['div_255'] = 1

    x = x.astype(np.float32)
    return x, norm_steps

def undo_norm_range01(x, xnorm):
    if 'div_255' in xnorm:
        x = (x*255).astype(np.uint8)
    reductions = [key for key, value in xnorm.items() if 'reduced' in key.lower()]
    if len(reductions)>0:
        reductions = reductions[0]
        reductions = reductions.replace('reduced_','')
        x = (x*65535).astype(eval("np.{}".format(reductions) ))
    return x

def reduce_dtype(x, x_min, x_max, out_min=0, out_max=1, out_type=np.float32):
    return ((np.array((x-x_min)/(x_max-x_min))*(out_max-out_min))+out_min).astype(out_type)

def normalize(data, means, stds):
    return (data - means) / stds

def denormalize(data, means, stds):
    return (data * stds) + means