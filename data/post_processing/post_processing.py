import cv2
import os
import math
import statistics
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import fill_voids
import edt
from tqdm import tqdm
from scipy import ndimage as ndi
from scipy.signal import find_peaks
from scipy.spatial import KDTree, cKDTree
from scipy.spatial.distance import cdist
from scipy.ndimage.morphology import binary_erosion, binary_dilation
from scipy.ndimage import rotate, grey_dilation, distance_transform_edt
from scipy.signal import savgol_filter
from scipy.ndimage.filters import median_filter
from scipy.ndimage.measurements import center_of_mass
from skimage import morphology
from skimage.morphology import disk, ball, remove_small_objects, dilation, erosion
from skimage.segmentation import watershed
from skimage.filters import rank, threshold_otsu
from skimage.measure import label, regionprops_table
from skimage.io import imread
from skimage.exposure import equalize_adapthist

from engine.metrics import jaccard_index_numpy
from utils.util import pad_and_reflect, save_tif
from data.pre_processing import normalize
from data.data_3D_manipulation import crop_3D_data_with_overlap
from data.pre_processing import reduce_dtype

def boundary_refinement_watershed(X, Y_pred, erode=True, save_marks_dir=None):
    """Apply watershed to the given predictions with the goal of refine the boundaries of the artifacts.

       Based on https://docs.opencv.org/master/d3/db4/tutorial_py_watershed.html.

       Parameters
       ----------
       X : 4D Numpy array
           Original data to guide the watershed. E.g. ``(img_number, y, x, channels)``.

       Y_pred : 4D Numpy array
           Predicted data to refine the boundaries. E.g. ``(img_number, y, x, channels)``.

       erode : bool, optional
           To extract the sure foreground eroding the artifacts instead of doing with distanceTransform.

       save_marks_dir : str, optional
           Directory to save the markers used to make the watershed. Useful for debugging.

       Returns
       -------
       Array : 4D Numpy array
           Refined boundaries of the predictions. E.g. ``(img_number, y, x, channels)``.

       Examples
       --------

       +------------------------------------------------+------------------------------------------------+
       | .. figure:: ../../../img/lucchi_test_0.png     | .. figure:: ../../../img/lucchi_test_0_gt.png  |
       |   :width: 80%                                  |   :width: 80%                                  |
       |   :align: center                               |   :align: center                               |
       |                                                |                                                |
       |   Original image                               |   Ground truth                                 |
       +------------------------------------------------+------------------------------------------------+
       | .. figure:: ../../../img/lucchi_test_0_pred.png| .. figure:: ../../../img/lucchi_test_0_wa.png  |
       |   :width: 80%                                  |   :width: 80%                                  |
       |   :align: center                               |   :align: center                               |
       |                                                |                                                |
       |   Predicted image                              |   Watershed ouput                              |
       +------------------------------------------------+------------------------------------------------+

       The marks used to guide the watershed is this example are these:

        .. image:: ../../../img/watershed2_marks_test0.png
          :width: 70%
          :align: center
    """

    if save_marks_dir is not None:
        os.makedirs(save_marks_dir, exist_ok=True)

    watershed_predictions = np.zeros(Y_pred.shape[:3])
    kernel = np.ones((3,3),np.uint8)
    d = len(str(X.shape[0]))

    for i in tqdm(range(X.shape[0])):
        im = cv2.cvtColor(X[i,...]*255, cv2.COLOR_GRAY2RGB)
        pred = Y_pred[i,...,0]

        # sure background area
        sure_bg = cv2.dilate(pred, kernel, iterations=3)
        sure_bg = np.uint8(sure_bg)

        # Finding sure foreground area
        if erode:
            sure_fg = cv2.erode(pred, kernel, iterations=3)
        else:
            dist_transform = cv2.distanceTransform(a, cv2.DIST_L2, 5)
            ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255,0)
        sure_fg = np.uint8(sure_fg)

        # Finding unknown region
        unknown_reg = cv2.subtract(sure_bg, sure_fg)

        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers+1

        # Now, mark the region of unknown with zero
        markers[unknown_reg==1] = 0

        if save_marks_dir is not None:
            f = os.path.join(save_marks_dir, "mark_" + str(i).zfill(d) + ".png")
            cv2.imwrite(f, markers)

        markers = cv2.watershed((im).astype(np.uint8), markers)

        watershed_predictions[i] = markers

    # Label all artifacts into 1 and the background with 0
    watershed_predictions[watershed_predictions==1] = 0
    watershed_predictions[watershed_predictions>1] = 1
    watershed_predictions[watershed_predictions==-1] = 0

    return np.expand_dims(watershed_predictions, -1)


def boundary_refinement_watershed2(X, Y_pred, save_marks_dir=None):
    """Apply watershed to the given predictions with the goal of refine the boundaries of the artifacts. This function
       was implemented using scikit instead of opencv as :meth:`post_processing.boundary_refinement_watershed`.

       Based on https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html.

       Parameters
       ----------
       X : 4D Numpy array
           Original data to guide the watershed. E.g. ``(img_number, y, x, channels)``.

       Y_pred : 4D Numpy array
           Predicted data to refine the boundaries. E.g. ``(img_number, y, x, channels)``.

       save_marks_dir : str, optional
           Directory to save the markers used to make the watershed. Useful for debugging.

       Returns
       -------
       Array : 4D Numpy array
           Refined boundaries of the predictions. E.g. ``(img_number, y, x, channels)``.
    """

    if save_marks_dir is not None:
        os.makedirs(save_marks_dir, exist_ok=True)

    watershed_predictions = np.zeros(Y_pred.shape[:3], dtype=np.uint8)
    d = len(str(X.shape[0]))

    for i in tqdm(range(X.shape[0])):

        im = (X[i,...,0]*255).astype(np.uint8)
        pred = (Y_pred[i,...,0]*255).astype(np.uint8)

        # find continuous region
        markers = rank.gradient(pred, disk(12)) < 10
        markers = ndi.label(markers)[0]

        # local gradient (disk(2) is used to keep edges thin)
        gradient = rank.gradient(im, disk(2))

        # process the watershed
        labels = watershed(gradient, markers)

        if save_marks_dir is not None:
            f = os.path.join(save_marks_dir, "mark_" + str(i).zfill(d) + ".png")
            cv2.imwrite(f, markers)

        watershed_predictions[i] = labels

    # Label all artifacts into 1 and the background with 0
    watershed_predictions[watershed_predictions==1] = 0
    watershed_predictions[watershed_predictions>1] = 1

    return np.expand_dims(watershed_predictions, -1)


def watershed_by_channels(data, channels, ths={}, remove_before=False, thres_small_before=10, remove_after=False, thres_small_after=128,
    seed_morph_sequence=[], seed_morph_radius=[], erode_and_dilate_foreground=False, fore_erosion_radius=5, fore_dilation_radius=5, 
    rmv_close_points=False, remove_close_points_radius=-1, resolution=[1,1,1], save_dir=None):
    """
    Convert binary foreground probability maps and instance contours to instance masks via watershed segmentation
    algorithm.

    Implementation based on `PyTorch Connectomics' process.py
    <https://github.com/zudi-lin/pytorch_connectomics/blob/master/connectomics/utils/process.py>`_.

    Parameters
    ----------
    data : 4D Numpy array
        Binary foreground labels and contours data to apply watershed into. E.g. ``(397, 1450, 2000, 2)``.

    channels : str
        Channel type used. Possible options: ``BC``, ``BCM``, ``BCD``, ``BCDv2``, ``Dv2`` and ``BDv2``.

    ths : float, optional
        Thresholds to be used on each channel. ``TH_BINARY_MASK`` used in the semantic mask to create watershed seeds;
        ``TH_CONTOUR`` used in the contours to create watershed seeds; ``TH_FOREGROUND`` used in the semantic mask to create the 
        foreground mask; ``TH_DISTANCE`` used in the distances to create watershed seeds; and ``TH_DIST_FOREGROUND`` used in the distances 
        to create the foreground mask.

    remove_before : bool, optional
        To remove objects before watershed. 

    thres_small_before : int, optional
        Theshold to remove small objects created by the watershed.

    remove_after : bool, optional
        To remove objects after watershed. 

    thres_small_after : int, optional
        Theshold to remove small objects created by the watershed.
        
    seed_morph_sequence : List of str, optional
        List of strings to determine the morphological filters to apply to instance seeds. They will be done in that order.
        E.g. ``['dilate','erode']``.
    
    seed_morph_radius: List of ints, optional
        List of ints to determine the radius of the erosion or dilation for instance seeds.
        
    erode_and_dilate_foreground : bool, optional
        To erode and dilate the foreground mask before using marker controlled watershed. The idea is to 
        remove the small holes that may be produced so the instances grow without them.
    
    fore_erosion_radius: int, optional
        Radius to erode the foreground mask. 

    fore_dilation_radius: int, optional
        Radius to dilate the foreground mask. 
    
    rmv_close_points : bool, optional
        To remove close points to each other. Used in 'BP' channel configuration. 

    remove_close_points_radius : float, optional
        Radius from each point to decide what points to keep. Used in 'BP' channel configuration. 
        E.g. ``10.0`.

    resolution : ndarray of floats
        Resolution of the data, in `(z,y,x)` to calibrate coordinates. E.g. ``[30,8,8]``.    

    save_dir :  str, optional
        Directory to save watershed output into.
    """

    assert channels in ['BC', 'BCM', 'BCD', 'BCDv2', 'Dv2', 'BDv2', 'BP', 'BD']

    def erode_seed_and_foreground():
        nonlocal seed_map
        nonlocal foreground
        if len(seed_morph_sequence) != 0:
            print("Applying {} to seeds . . .".format(seed_morph_sequence))
        if erode_and_dilate_foreground:
            print("Foreground erosion . . .")

        if len(seed_morph_sequence) != 0:
            morph_funcs = []
            for operation in seed_morph_sequence:
                if operation == "dilate":
                    morph_funcs.append(binary_dilation)
                elif operation == "erode":
                    morph_funcs.append(binary_erosion)  

        image3d = True if seed_map.ndim == 3 else False
        if not image3d:
            seed_map = np.expand_dims(seed_map,0)
            foreground = np.expand_dims(foreground,0)

        for i in tqdm(range(seed_map.shape[0])):
            if len(seed_morph_sequence) != 0:
                for k, morph_function in enumerate(morph_funcs):
                    seed_map[i] = morph_function(seed_map[i], disk(radius=seed_morph_radius[k]))

            if erode_and_dilate_foreground:
                foreground[i] = binary_dilation(foreground[i], disk(radius=fore_erosion_radius))
                foreground[i] = binary_erosion(foreground[i], disk(radius=fore_dilation_radius))
       
        if not image3d:
            seed_map = seed_map.squeeze()
            foreground = foreground.squeeze()

    if channels in ["BC", "BCM"]:
        seed_map = (data[...,0] > ths['TH_BINARY_MASK']) * (data[...,1] < ths['TH_CONTOUR'])
        foreground = (data[...,0] > ths['TH_FOREGROUND'])

        if len(seed_morph_sequence) != 0 or erode_and_dilate_foreground:
            erode_seed_and_foreground()
        
        res = (1,)+resolution if len(resolution) == 2 else resolution
        semantic = edt.edt(foreground, anisotropy=res, black_border=False, order='C')
        seed_map = label(seed_map, connectivity=1)
    elif channels in ["BP"]:
        seed_map = (data[...,1] > ths['TH_POINTS'])
        foreground = (data[...,0] > ths['TH_FOREGROUND'])

        print("Creating the central points . . .")
        seed_map = label(seed_map, connectivity=1)
        instances = np.unique(seed_map)[1:]
        seed_coordinates = center_of_mass(seed_map, label(seed_map), instances)
        seed_coordinates = np.round(seed_coordinates).astype(int)

        if rmv_close_points:
            seed_coordinates = remove_close_points(seed_coordinates, remove_close_points_radius, resolution,
                ndim=seed_map.ndim)

        seed_map = np.zeros(data.shape[:-1], dtype=np.uint8) 
        for sd in tqdm(seed_coordinates, total=len(seed_coordinates)):
            z,y,x = sd
            seed_map[z,y,x] = 1

        res = (1,)+resolution if len(resolution) == 2 else resolution
        semantic = -edt.edt(1 - seed_map, anisotropy=res, black_border=False, order='C')

        if len(seed_morph_sequence) != 0 or erode_and_dilate_foreground:
            erode_seed_and_foreground()

        seed_map = label(seed_map, connectivity=1)
    elif channels in ["BD"]:
        semantic = data[...,0]
        seed_map = (data[...,0] > ths['TH_BINARY_MASK']) * (data[...,1] < ths['TH_DISTANCE'])
        foreground = (semantic > ths['TH_FOREGROUND']) * (data[...,1] < ths['TH_DIST_FOREGROUND'])
        seed_map = label(seed_map, connectivity=1)
    elif channels in ["BCD"]:
        semantic = data[...,0]
        seed_map = (data[...,0] > ths['TH_BINARY_MASK']) * (data[...,1] < ths['TH_CONTOUR']) * (data[...,2] < ths['TH_DISTANCE'])
        foreground = (semantic > ths['TH_FOREGROUND']) * (data[...,2] < ths['TH_DIST_FOREGROUND'])
        if len(seed_morph_sequence) != 0 or erode_and_dilate_foreground:
            erode_seed_and_foreground()
        seed_map = label(seed_map, connectivity=1)
    else: # 'BCDv2', 'Dv2', 'BDv2'
        semantic = data[...,-1]
        foreground = None
        if channels == "BCDv2": # 'BCDv2'
            seed_map = (data[...,0] > ths['TH_BINARY_MASK']) * (data[...,1] < ths['TH_CONTOUR']) * (data[...,1] < ths['TH_DISTANCE'])
            background_seed = binary_dilation( ((data[...,0]>ths['TH_BINARY_MASK']) + (data[...,1]>ths['TH_CONTOUR'])).astype(np.uint8), iterations=2)
            seed_map, num = label(seed_map, connectivity=1, return_num=True)

            # Create background seed and label correctly
            background_seed = 1 - background_seed
            background_seed[background_seed==1] = num+1
            seed_map = seed_map + background_seed
            del background_seed
        elif channels == "BDv2": # 'BDv2'
            seed_map = (data[...,0] > ths['TH_BINARY_MASK']) * (data[...,1] < ths['TH_DISTANCE'])
            background_seed = binary_dilation((data[...,1]<ths['TH_DISTANCE']).astype(np.uint8), iterations=2)
            seed_map = label(seed_map, connectivity=1)
            background_seed = label(background_seed, connectivity=1)

            props = regionprops_table(seed_map, properties=('area','centroid'))
            for n in range(len(props['centroid-0'])):
                label_center = [props['centroid-0'][n], props['centroid-1'][n], props['centroid-2'][n]]
                instance_to_remove = background_seed[label_center]
                background_seed[background_seed == instance_to_remove] = 0
            seed_map = seed_map + background_seed
            del background_seed
            seed_map = label(seed_map, connectivity=1) # re-label again
        elif channels == "Dv2": # 'Dv2'
            seed_map = data[...,0] < ths['TH_DISTANCE']
            seed_map = label(seed_map, connectivity=1)

        if len(seed_morph_sequence) != 0:
            erode_seed_and_foreground()

    if remove_before:
        seed_map = remove_small_objects(seed_map, thres_small_before)

    segm = watershed(-semantic, seed_map, mask=foreground)

    if remove_after:
        segm = remove_small_objects(segm, thres_small_after)

    # Choose appropiate dtype
    max_value = np.max(segm)
    if max_value < 255:
        segm = segm.astype(np.uint8)
    elif max_value < 65535:
        segm = segm.astype(np.uint16)
    else:
        segm = segm.astype(np.uint32)

    if save_dir is not None:
        save_tif(np.expand_dims(np.expand_dims(seed_map,-1),0).astype(segm.dtype), save_dir, ["seed_map.tif"], verbose=False)
        save_tif(np.expand_dims(np.expand_dims(semantic,-1),0).astype(np.float32), save_dir, ["semantic.tif"], verbose=False)
        if channels in ["BC", "BCM", "BCD", "BP"]:
            save_tif(np.expand_dims(np.expand_dims(foreground,-1),0).astype(np.uint8), save_dir, ["foreground.tif"], verbose=False)
    return segm


def calculate_zy_filtering(data, mf_size=5):
    """Applies a median filtering in the z and y axes of the provided data.

       Parameters
       ----------
       data : 4D Numpy array
           Data to apply the filter to. E.g. ``(num_of_images, y, x, channels)``.

       mf_size : int, optional
           Size of the median filter. Must be an odd number.

       Returns
       -------
       Array : 4D Numpy array
           Filtered data. E.g. ``(num_of_images, y, x, channels)``.
    """

    out_data = np.copy(data)

    # Must be odd
    if mf_size % 2 == 0:
       mf_size += 1

    for i in range(data.shape[0]):
        for c in range(data.shape[-1]):
            sl = (data[i,...,c]).astype(np.float32)
            sl = cv2.medianBlur(sl, mf_size)
            out_data[i,...,c] = sl

    return out_data

def calculate_z_filtering(data, mf_size=5):
    """Applies a median filtering in the z dimension of the provided data.

       Parameters
       ----------
       data : 4D Numpy array
           Data to apply the filter to. E.g. ``(num_of_images, y, x, channels)``.

       mf_size : int, optional
           Size of the median filter. Must be an odd number.

       Returns
       -------
       Array : 4D Numpy array
           Filtered data. E.g. ``(num_of_images, y, x, channels)``.
    """

    out_data = np.copy(data)

    # Must be odd
    if mf_size % 2 == 0:
       mf_size += 1

    for c in range(out_data.shape[-1]):
        out_data[...,c] = median_filter(data[...,c], size=(mf_size,1,1,1))

    return out_data 

def ensemble8_2d_predictions(o_img, pred_func, batch_size_value=1, n_classes=1):
    """Outputs the mean prediction of a given image generating its 8 possible rotations and flips.

       Parameters
       ----------
       o_img : 3D Numpy array
           Input image. E.g. ``(y, x, channels)``.

       pred_func : function
           Function to make predictions.

       batch_size_value : int, optional
           Batch size value.

       n_classes : int, optional
           Number of classes.

       Returns
       -------
       out : 3D Numpy array
           Output image ensembled. E.g. ``(y, x, channels)``.

       Examples
       --------
       ::

           # EXAMPLE 1
           # Apply ensemble to each image of X_test
           X_test = np.ones((165, 768, 1024, 1))
           out_X_test = np.zeros(X_test.shape, dtype=(np.float32))

           for i in tqdm(range(X_test.shape[0])):
               pred_ensembled = ensemble8_2d_predictions(X_test[i],
                   pred_func=(lambda img_batch_subdiv: model.predict(img_batch_subdiv)), n_classes=n_classes)
               out_X_test[i] = pred_ensembled

           # Notice that here pred_func is created based on model.predict function of Keras
    """

    # Prepare all the image transformations per channel
    total_img = []
    for channel in range(o_img.shape[-1]):
        aug_img = []

        # Transformations per channel
        _img = np.expand_dims(o_img[...,channel], -1)

        # Convert into square image to make the rotations properly
        pad_to_square = _img.shape[0] - _img.shape[1]
        if pad_to_square < 0:
            img = np.pad(_img, [(abs(pad_to_square), 0), (0, 0), (0, 0)], 'reflect')
        else:
            img = np.pad(_img, [(0, 0), (pad_to_square, 0), (0, 0)], 'reflect')

        # Make 8 different combinations of the img
        aug_img.append(img)
        aug_img.append(np.rot90(img, axes=(0, 1), k=1))
        aug_img.append(np.rot90(img, axes=(0, 1), k=2))
        aug_img.append(np.rot90(img, axes=(0, 1), k=3))
        aug_img.append(img[:, ::-1])
        img_aux = img[:, ::-1]
        aug_img.append(np.rot90(img_aux, axes=(0, 1), k=1))
        aug_img.append(np.rot90(img_aux, axes=(0, 1), k=2))
        aug_img.append(np.rot90(img_aux, axes=(0, 1), k=3))
        aug_img = np.array(aug_img)

        total_img.append(aug_img)

    del aug_img, img_aux

    # Merge channels
    total_img = np.concatenate(total_img, -1)

    # Make the prediction
    _decoded_aug_img = []
    l = int(math.ceil(total_img.shape[0]/batch_size_value))
    for i in range(l):
        top = (i+1)*batch_size_value if (i+1)*batch_size_value < total_img.shape[0] else total_img.shape[0]
        r_aux = pred_func(total_img[i*batch_size_value:top])

        # Take just the last output of the network in case it returns more than one output
        if isinstance(r_aux, list):
            r_aux = np.array(r_aux[-1])

        if n_classes > 1:
            r_aux = np.expand_dims(np.argmax(r_aux, -1), -1)

        _decoded_aug_img.append(r_aux)
    _decoded_aug_img = np.concatenate(_decoded_aug_img)

    # Undo the combinations of the img
    arr = []
    for c in range(_decoded_aug_img.shape[-1]):
        # Remove the last channel to make the transformations correctly
        decoded_aug_img = _decoded_aug_img[...,c]

        # Undo the combinations of the image
        out_img = []
        out_img.append(decoded_aug_img[0])
        out_img.append(np.rot90(decoded_aug_img[1], axes=(0, 1), k=3))
        out_img.append(np.rot90(decoded_aug_img[2], axes=(0, 1), k=2))
        out_img.append(np.rot90(decoded_aug_img[3], axes=(0, 1), k=1))
        out_img.append(decoded_aug_img[4][:, ::-1])
        out_img.append(np.rot90(decoded_aug_img[5], axes=(0, 1), k=3)[:, ::-1])
        out_img.append(np.rot90(decoded_aug_img[6], axes=(0, 1), k=2)[:, ::-1])
        out_img.append(np.rot90(decoded_aug_img[7], axes=(0, 1), k=1)[:, ::-1])
        out_img = np.array(out_img)
        out_img = np.expand_dims(out_img, -1)
        arr.append(out_img)

    out_img = np.concatenate(arr, -1)
    del decoded_aug_img, _decoded_aug_img, arr

    if pad_to_square != 0:
        if pad_to_square < 0:
            out = np.zeros((out_img.shape[0], img.shape[0]+pad_to_square, img.shape[1], out_img.shape[-1]))
        else:
            out = np.zeros((out_img.shape[0], img.shape[0], img.shape[1]-pad_to_square, out_img.shape[-1]))
    else:
        out = np.zeros(out_img.shape)

    # Undo the padding
    for i in range(out_img.shape[0]):
        if pad_to_square < 0:
            out[i] = out_img[i,abs(pad_to_square):,:]
        else:
            out[i] = out_img[i,:,abs(pad_to_square):]

    return np.mean(out, axis=0)


def ensemble16_3d_predictions(vol, pred_func, batch_size_value=1, n_classes=1):
    """Outputs the mean prediction of a given image generating its 16 possible rotations and flips.

       Parameters
       ----------
       o_img : 4D Numpy array
           Input image. E.g. ``(z, y, x, channels)``.

       pred_func : function
           Function to make predictions.

       batch_size_value : int, optional
           Batch size value.

       n_classes : int, optional
           Number of classes.

       Returns
       -------
       out : 4D Numpy array
           Output image ensembled. E.g. ``(z, y, x, channels)``.

       Examples
       --------
       ::

           # EXAMPLE 1
           # Apply ensemble to each image of X_test
           X_test = np.ones((10, 165, 768, 1024, 1))
           out_X_test = np.zeros(X_test.shape, dtype=(np.float32))

           for i in tqdm(range(X_test.shape[0])):
               pred_ensembled = ensemble8_2d_predictions(X_test[i],
                   pred_func=(lambda img_batch_subdiv: model.predict(img_batch_subdiv)), n_classes=n_classes)
               out_X_test[i] = pred_ensembled

           # Notice that here pred_func is created based on model.predict function of Keras
    """

    total_vol = []
    for channel in range(vol.shape[-1]):

        aug_vols = []

        # Transformations per channel
        _vol = vol[...,channel]

        # Convert into square image to make the rotations properly
        pad_to_square = _vol.shape[2] - _vol.shape[1]
        if pad_to_square < 0:
            volume = np.pad(_vol, [(0,0), (0,0), (abs(pad_to_square),0)], 'reflect')
        else:
            volume = np.pad(_vol, [(0,0), (pad_to_square,0), (0,0)], 'reflect')

        # Make 16 different combinations of the volume
        aug_vols.append(volume)
        aug_vols.append(rotate(volume, mode='reflect', axes=(2, 1), angle=90, reshape=False))
        aug_vols.append(rotate(volume, mode='reflect', axes=(2, 1), angle=180, reshape=False))
        aug_vols.append(rotate(volume, mode='reflect', axes=(2, 1), angle=270, reshape=False))
        volume_aux = np.flip(volume, 0)
        aug_vols.append(volume_aux)
        aug_vols.append(rotate(volume_aux, mode='reflect', axes=(2, 1), angle=90, reshape=False))
        aug_vols.append(rotate(volume_aux, mode='reflect', axes=(2, 1), angle=180, reshape=False))
        aug_vols.append(rotate(volume_aux, mode='reflect', axes=(2, 1), angle=270, reshape=False))
        volume_aux = np.flip(volume, 1)
        aug_vols.append(volume_aux)
        aug_vols.append(rotate(volume_aux, mode='reflect', axes=(2, 1), angle=90, reshape=False))
        aug_vols.append(rotate(volume_aux, mode='reflect', axes=(2, 1), angle=180, reshape=False))
        aug_vols.append(rotate(volume_aux, mode='reflect', axes=(2, 1), angle=270, reshape=False))
        volume_aux = np.flip(volume, 2)
        aug_vols.append(volume_aux)
        aug_vols.append(rotate(volume_aux, mode='reflect', axes=(2, 1), angle=90, reshape=False))
        aug_vols.append(rotate(volume_aux, mode='reflect', axes=(2, 1), angle=180, reshape=False))
        aug_vols.append(rotate(volume_aux, mode='reflect', axes=(2, 1), angle=270, reshape=False))
        aug_vols = np.array(aug_vols)

        # Add the last channel again
        aug_vols = np.expand_dims(aug_vols, -1)
        total_vol.append(aug_vols)

    del aug_vols, volume_aux
    # Merge channels
    total_vol = np.concatenate(total_vol, -1)

    _decoded_aug_vols = []

    l = int(math.ceil(total_vol.shape[0]/batch_size_value))
    for i in range(l):
        top = (i+1)*batch_size_value if (i+1)*batch_size_value < total_vol.shape[0] else total_vol.shape[0]
        r_aux = pred_func(total_vol[i*batch_size_value:top])

        # Take just the last output of the network in case it returns more than one output
        if isinstance(r_aux, list):
            r_aux = np.array(r_aux[-1])

        if n_classes > 1:
            r_aux = np.expand_dims(np.argmax(r_aux, -1), -1)

        if r_aux.ndim == 4:
            r_aux = np.expand_dims(r_aux, 0)
        _decoded_aug_vols.append(r_aux)

    _decoded_aug_vols = np.concatenate(_decoded_aug_vols)
    volume = np.expand_dims(volume, -1)

    arr = []
    for c in range(_decoded_aug_vols.shape[-1]):
        # Remove the last channel to make the transformations correctly
        decoded_aug_vols = _decoded_aug_vols[...,c]

        # Undo the combinations of the volume
        out_vols = []
        out_vols.append(np.array(decoded_aug_vols[0]))
        out_vols.append(rotate(np.array(decoded_aug_vols[1]), mode='reflect', axes=(2, 1), angle=-90, reshape=False))
        out_vols.append(rotate(np.array(decoded_aug_vols[2]), mode='reflect', axes=(2, 1), angle=-180, reshape=False))
        out_vols.append(rotate(np.array(decoded_aug_vols[3]), mode='reflect', axes=(2, 1), angle=-270, reshape=False))
        out_vols.append(np.flip(np.array(decoded_aug_vols[4]), 0))
        out_vols.append(np.flip(rotate(np.array(decoded_aug_vols[5]), mode='reflect', axes=(2, 1), angle=-90, reshape=False), 0))
        out_vols.append(np.flip(rotate(np.array(decoded_aug_vols[6]), mode='reflect', axes=(2, 1), angle=-180, reshape=False), 0))
        out_vols.append(np.flip(rotate(np.array(decoded_aug_vols[7]), mode='reflect', axes=(2, 1), angle=-270, reshape=False), 0))
        out_vols.append(np.flip(np.array(decoded_aug_vols[8]), 1))
        out_vols.append(np.flip(rotate(np.array(decoded_aug_vols[9]), mode='reflect', axes=(2, 1), angle=-90, reshape=False), 1))
        out_vols.append(np.flip(rotate(np.array(decoded_aug_vols[10]), mode='reflect', axes=(2, 1), angle=-180, reshape=False), 1))
        out_vols.append(np.flip(rotate(np.array(decoded_aug_vols[11]), mode='reflect', axes=(2, 1), angle=-270, reshape=False), 1))
        out_vols.append(np.flip(np.array(decoded_aug_vols[12]), 2))
        out_vols.append(np.flip(rotate(np.array(decoded_aug_vols[13]), mode='reflect', axes=(2, 1), angle=-90, reshape=False), 2))
        out_vols.append(np.flip(rotate(np.array(decoded_aug_vols[14]), mode='reflect', axes=(2, 1), angle=-180, reshape=False), 2))
        out_vols.append(np.flip(rotate(np.array(decoded_aug_vols[15]), mode='reflect', axes=(2, 1), angle=-270, reshape=False), 2))

        out_vols = np.array(out_vols)
        out_vols = np.expand_dims(out_vols, -1)
        arr.append(out_vols)

    out_vols = np.concatenate(arr, -1)
    del decoded_aug_vols, _decoded_aug_vols, arr

    # Create the output data
    if pad_to_square != 0:
        if pad_to_square < 0:
            out = np.zeros((out_vols.shape[0], volume.shape[0], volume.shape[1], volume.shape[2]+pad_to_square, out_vols.shape[-1]))
        else:
            out = np.zeros((out_vols.shape[0], volume.shape[0], volume.shape[1]-pad_to_square, volume.shape[2], out_vols.shape[-1]))
    else:
        out = np.zeros(out_vols.shape)

    # Undo the padding
    for i in range(out_vols.shape[0]):
        if pad_to_square < 0:
            out[i] = out_vols[i,:,:,abs(pad_to_square):,:]
        else:
            out[i] = out_vols[i,:,abs(pad_to_square):,:,:]

    return np.mean(out, axis=0)


def calculate_optimal_mw_thresholds(model, data_path, data_mask_path, patch_size, mode="BC", distance_mask_path=None,
                                    thres_small=5, bin_mask_path=None, use_minimum=False, chart_dir=None, verbose=True):
    """Calculate the optimum values for the marked controlled watershed thresholds.

       Parameters
       ----------
       model: Keras model
           Model to make the predictions.

       data_path : str
           Path to load the samples to infer.

       data_mask_path : str
           Path to load the mask samples.

       patch_size : : 4D tuple
           Shape of the train subvolumes to create. E.g. ``(x, y, z, channels)``.

       mode : str, optional
           Operation mode. Possible values: ``BC`` and ``BCD``. ``BC`` corresponds to use binary segmentation+contour.
           ``BCD`` stands for binary segmentation+contour+distances.

       thres_small : int, optional
           Theshold to remove small objects in the mask and the prediction.

       bin_mask_path : str, optional
           Path of the binary masks to apply to the prediction. Useful to remove segmentation outside the masks.
           If ``None``, no mask is applied.

       use_minimum : bool, optional
           Return the minimum value of TH_BINARY_MASK (and TH_DISTANCE) instead of the mean.

       chart_dir : str, optional
           Path where the charts are stored.

       verbose : bool, optional
           To print saving information.

       Returns
       -------
       global_thbinmask_min_opt: float
           MW_TH_BINARY_MASK optimum value.

       global_thcontour_opt : float
           MW_TH_CONTOUR optimum value.

       global_thfore : float
           MW_TH_FOREGROUND optimum value.

       global_thdist_min_opt : float, optional
           MW_TH_DISTANCE optimum value.

       global_thdistfore : float, optional
           MW_TH_DIST_FOREGROUND optimum value.
    """

    assert mode in ['BC', 'BCD']
    if mode == 'BCD' and distance_mask_path is None:
        raise ValueError("distance_mask_path needs to be not None when mode is 'BCD'")
    if verbose:
        print("Calculating the best thresholds for the mark controlled watershed . . .")
    ids = sorted(next(os.walk(data_path))[2])
    mask_ids = sorted(next(os.walk(data_mask_path))[2])
    ths = [1e-06, 5e-06, 1e-05, 5e-05, 1e-04, 5e-04, 1e-03, 5e-03, 1e-02, 5e-02]
    ths.extend(np.round(np.arange(0.1, 1.0, 0.05),3))

    min_thbinmask = 1
    g_l_thbinmask = []
    l_thbinmask_min = []
    l_thbinmask_opt = []
    g_l_thcontour = []
    l_thcontour_min = []
    l_thcontour_opt = []
    ideal_number_obj = []
    g_l_thfore = []
    l_thfore_max = []
    if mode == 'BCD':
        g_l_thdist = []
        l_thdist_min = []
        l_thdist_opt = []
        g_l_thdistfore = []
        l_thdistfore_max = []

    if mode == 'BCD':
        print("Calculating the max distance value first. . ")
        dis_mask_ids = sorted(next(os.walk(distance_mask_path))[2])
        max_distance = 0
        for i in tqdm(range(len(ids))):
            if dis_mask_ids[i].endswith('.npy'):
                mask = np.load(os.path.join(distance_mask_path, dis_mask_ids[i]))
            else:
                mask = imread(os.path.join(distance_mask_path, dis_mask_ids[i]))
            if mask.shape[-1] != 3:
                raise ValueError("Expected a 3-channel data to be loaded and not {}".format(mask.shape))
            m = np.max(mask[...,2])
            if max_distance < m: max_distance = m
        max_distance += max_distance*0.1
        ths_dis = np.arange(0.1, max_distance, 0.05)

    for i in tqdm(range(len(ids))):
        if verbose: print("Analizing file {}".format(os.path.join(data_path, ids[i])))

        # Read and prepare images
        if ids[i].endswith('.npy'):
            _img = np.load(os.path.join(data_path, ids[i]))
        else:
            _img = imread(os.path.join(data_path, ids[i]))
        _img = np.squeeze(_img)

        # Ensure uint8
        if _img.dtype == np.uint16:
            if np.max(_img) > 255:
                _img = normalize(_img, 0, 65535)
            else:
                _img = _img.astype(np.uint8)

        if mask_ids[i].endswith('.npy'):
            _mask = np.load(os.path.join(data_mask_path, mask_ids[i]))
        else:
            _mask = imread(os.path.join(data_mask_path, mask_ids[i]))
        _mask = np.squeeze(_mask)
        if len(_img.shape) == 3:
            _img = np.expand_dims(_img, axis=-1)
        if len(_mask.shape) == 3:
            _mask = np.expand_dims(_mask, axis=-1)
        _img = np.expand_dims(_img, axis=0)
        _mask = np.expand_dims(_mask, axis=0)
        _mask = _mask.astype(np.uint8)

        _img = np.expand_dims(pad_and_reflect(_img[0], patch_size, verbose=verbose), 0)
        _mask = np.expand_dims(pad_and_reflect(_mask[0], patch_size, verbose=verbose), 0)

        if _img.shape != patch_size:
            _img, _mask = crop_3D_data_with_overlap(_img[0], patch_size, data_mask=_mask[0], overlap=(0,0,0),
                padding=(0,0,0), verbose=verbose, median_padding=False)
        else:
            _img = np.transpose(_img, (1,2,0,3))
            _mask = np.transpose(_mask, (1,2,0,3))

        for k in range(_img.shape[0]):
            img = _img[k]
            mask = _mask[k]

            # Mask adjust
            labels = np.unique(mask)
            if len(labels) != 1:
                mask = remove_small_objects(mask, thres_small)

            if len(labels) == 1:
                if verbose:
                    print("Skip this sample as it is just background")
            else:
                # Prediction
                if np.max(img) > 30: img = img/255
                pred = model.predict(img, verbose=0)

                if bin_mask_path is not None:
                    pred = apply_binary_mask(pred, bin_mask_path)

                # TH_FOREGROUND and TH_DIST_FOREGROUND:
                # Look at the best IoU compared with the original label. Only the region that involve the object is taken
                # into consideration. This is achieved dilating 2 iterations the original object mask. If we do not dilate
                # that label, decreasing the TH will always ensure a IoU >= than the previous TH. This way, the IoU will
                # reach a maximum and then it will start to decrease, as more points that are not from the object are added
                # into it
                # TH_FOREGROUND
                thfore_best = -1
                thfore_max_jac = -1
                l_thfore = []
                for j in range(len(ths)):
                    p = np.expand_dims(pred[...,0] > ths[j],-1).astype(np.uint8)
                    jac = jaccard_index_numpy((mask>0).astype(np.uint8), p)
                    if jac > thfore_max_jac:
                        thfore_max_jac = jac
                        thfore_best = ths[j]
                    l_thfore.append(jac)
                l_thfore_max.append(thfore_best)
                g_l_thfore.append(l_thfore)
                # TH_DIST_FOREGROUND
                if mode == 'BCD':
                    thdistfore_best = -1
                    thdistfore_max_jac = -1
                    l_thdistfore = []
                    for j in range(len(ths_dis)):
                        p = np.expand_dims(pred[...,2] > ths_dis[j],-1).astype(np.uint8)
                        jac = jaccard_index_numpy((mask>0).astype(np.uint8), p)
                        if jac > thdistfore_max_jac:
                            thdistfore_max_jac = jac
                            thdistfore_best = ths_dis[j]
                        l_thdistfore.append(jac)
                    l_thdistfore_max.append(thdistfore_best)
                    g_l_thdistfore.append(l_thdistfore)

                # TH_CONTOUR: obtained the optimum value for the TH_FOREGROUND, the TH_CONTOUR threshold is calculated counting the objects. As this
                # threshold looks at the contour channels, its purpose is to separed the entangled objects. This way, the TH_CONTOUR
                # optimum should be reached when the number of objects of the prediction match the number of real objects
                objs_to_divide = (pred[...,0] > thfore_best).astype(np.uint8)
                thcontour_min = 0
                thcontour_last = 0
                thcontour_repeat_count = 0
                thcontour_op_pos = -1
                thcontour_obj_min_diff = sys.maxsize
                l_thcontour = []
                for k in range(len(ths)):
                    p = (objs_to_divide * (pred[...,1] < ths[k])).astype(np.uint8)

                    p = label(np.squeeze(p), connectivity=1)
                    if len(np.unique(p)) != 1:
                        p = remove_small_objects(p, thres_small)
                    obj_count = len(np.unique(p))
                    l_thcontour.append(obj_count)

                    if abs(obj_count-len(labels)) < thcontour_obj_min_diff:
                        thcontour_obj_min_diff = abs(obj_count-len(labels))
                        thcontour_min = ths[k]
                        thcontour_op_pos = k
                        thcontour_repeat_count = 0

                    if thcontour_obj_min_diff == thcontour_last: thcontour_repeat_count += 1
                    thcontour_last = abs(obj_count-len(labels))

                g_l_thcontour.append(l_thcontour)
                l_thcontour_min.append(thcontour_min)
                thcontour_opt_pos = thcontour_op_pos + int(thcontour_repeat_count/2) if thcontour_repeat_count < 10 else thcontour_op_pos + 2
                if thcontour_opt_pos >= len(ths): thcontour_opt_pos = len(ths)-1
                l_thcontour_opt.append(ths[thcontour_opt_pos])

                # TH_BINARY_MASK and TH_DISTANCE:
                thbinmask_min = 0
                thbinmask_last = 0
                thbinmask_repeat_count = 0
                thbinmask_op_pos = -1
                thbinmask_obj_min_diff = sys.maxsize
                l_thbinmask = []
                in_row = False
                # TH_BINARY_MASK
                for k in range(len(ths)):
                    p = ((pred[...,0] > ths[k])*(pred[...,1] < thcontour_min)).astype(np.uint8)

                    p = label(np.squeeze(p), connectivity=1)
                    obj_count = len(np.unique(p))
                    l_thbinmask.append(obj_count)

                    diff = abs(obj_count-len(labels))
                    if diff <= thbinmask_obj_min_diff and thbinmask_repeat_count < 4 and diff != thbinmask_last:
                        thbinmask_obj_min_diff = diff
                        thbinmask_min = ths[k]
                        thbinmask_op_pos = k
                        thbinmask_repeat_count = 0
                        in_row = True

                    if diff == thbinmask_last and diff == thbinmask_obj_min_diff and in_row:
                        thbinmask_repeat_count += 1
                    elif k != thbinmask_op_pos:
                        in_row = False
                    thbinmask_last = diff

                g_l_thbinmask.append(l_thbinmask)
                l_thbinmask_min.append(thbinmask_min)
                thbinmask_opt_pos = thbinmask_op_pos + thbinmask_repeat_count
                if thbinmask_opt_pos >= len(ths): thbinmask_opt_pos = len(ths)-1
                l_thbinmask_opt.append(ths[thbinmask_opt_pos])

                # TH_DISTANCE
                if mode == 'BCD':
                    thdist_min = 0
                    thdist_last = 0
                    thdist_repeat_count = 0
                    thdist_op_pos = -1
                    thdist_obj_min_diff = sys.maxsize
                    l_thdist = []
                    for k in range(len(ths_dis)):
                        p = ((pred[...,2] > ths_dis[k])*(pred[...,1] < thcontour_min)).astype(np.uint8)

                        p = label(np.squeeze(p), connectivity=1)
                        obj_count = len(np.unique(p))
                        l_thdist.append(obj_count)

                        diff = abs(obj_count-len(labels))
                        if diff <= thdist_obj_min_diff and thdist_repeat_count < 4 and diff != thdist_last:
                            thdist_obj_min_diff = diff
                            thdist_min = ths_dis[k]
                            thdist_op_pos = k
                            thdist_repeat_count = 0
                            in_row = True

                        if diff == thdist_last and diff == thdist_obj_min_diff and in_row:
                            thdist_repeat_count += 1
                        elif k != thdist_op_pos:
                            in_row = False
                        thdist_last = diff

                    g_l_thdist.append(l_thdist)
                    l_thdist_min.append(thdist_min)
                    thdist_opt_pos = thdist_op_pos + thdist_repeat_count
                    if thdist_opt_pos >= len(ths_dis): thdist_opt_pos = len(ths_dis)-1
                    l_thdist_opt.append(ths_dis[thdist_opt_pos])

            # Store the number of nucleus
            ideal_number_obj.append(len(labels))

    ideal_objects = statistics.mean(ideal_number_obj)
    create_th_plot(ths, g_l_thbinmask, "TH_BINARY_MASK", chart_dir)
    create_th_plot(ths, g_l_thbinmask, "TH_BINARY_MASK", chart_dir, per_sample=False, ideal_value=ideal_objects)
    create_th_plot(ths, g_l_thcontour, "TH_CONTOUR", chart_dir)
    create_th_plot(ths, g_l_thcontour, "TH_CONTOUR", chart_dir, per_sample=False, ideal_value=ideal_objects)
    create_th_plot(ths, g_l_thfore, "TH_FOREGROUND", chart_dir)
    create_th_plot(ths, g_l_thfore, "TH_FOREGROUND", chart_dir, per_sample=False)
    if mode == 'BCD':
        create_th_plot(ths_dis, g_l_thdist, "TH_DISTANCE", chart_dir)
        create_th_plot(ths_dis, g_l_thdist, "TH_DISTANCE", chart_dir, per_sample=False, ideal_value=ideal_objects)
        create_th_plot(ths_dis, g_l_thdistfore, "TH_DIST_FOREGROUND", chart_dir)
        create_th_plot(ths_dis, g_l_thdistfore, "TH_DIST_FOREGROUND", chart_dir, per_sample=False)

    if len(ideal_number_obj) > 1:
        global_thbinmask = statistics.mean(l_thbinmask_min)
        global_thbinmask_std = statistics.stdev(l_thbinmask_min)
        global_thbinmask_opt = statistics.mean(l_thbinmask_opt)
        global_thbinmask_opt_std = statistics.stdev(l_thbinmask_opt)
        global_thcontour = statistics.mean(l_thcontour_min)
        global_thcontour_std = statistics.stdev(l_thcontour_min)
        global_thcontour_opt = statistics.mean(l_thcontour_opt)
        global_thcontour_opt_std = statistics.stdev(l_thcontour_opt)
        global_thfore = statistics.mean(l_thfore_max)
        global_thfore_std = statistics.stdev(l_thfore_max)
        if mode == 'BCD':
            global_thdist = statistics.mean(l_thdist_min)
            global_thdist_std = statistics.stdev(l_thdist_min)
            global_thdist_opt = statistics.mean(l_thdist_opt)
            global_thdist_opt_std = statistics.stdev(l_thdist_opt)
            global_thdistfore = statistics.mean(l_thdistfore_max)
            global_thdistfore_std = statistics.stdev(l_thdistfore_max)
    else:
        global_thbinmask = l_thbinmask_min[0]
        global_thbinmask_std = 0
        global_thbinmask_opt = l_thbinmask_opt[0]
        global_thbinmask_opt_std = 0
        global_thcontour = l_thcontour_min[0]
        global_thcontour_std = 0
        global_thcontour_opt = l_thcontour_opt[0]
        global_thcontour_opt_std = 0
        global_thfore = l_thfore_max[0]
        global_thfore_std = 0
        if mode == 'BCD':
            global_thdist = l_thdist_min[0]
            global_thdist_std = 0
            global_thdist_opt = l_thdist_opt[0]
            global_thdist_opt_std = 0
            global_thdistfore = l_thdistfore_max[0]
            global_thdistfore_std = 0

    if verbose:
        if not use_minimum:
            print("MW_TH_BINARY_MASK maximum value is {} (std:{}) so the optimum should be {} (std:{})".format(global_thbinmask, global_thbinmask_std, global_thbinmask_opt, global_thbinmask_opt_std))
        else:
            print("MW_TH_BINARY_MASK minimum value is {}".format(min(l_thbinmask_min)))
        print("MW_TH_CONTOUR minimum value is {} (std:{}) and the optimum is {} (std:{})".format(global_thcontour, global_thcontour_std, global_thcontour_opt, global_thcontour_opt_std))
        print("MW_TH_FOREGROUND optimum should be {} (std:{})".format(global_thfore, global_thfore_std))
    if mode == 'BCD':
        if verbose:
            if not use_minimum:
                print("MW_TH_DISTANCE maximum value is {} (std:{}) so the optimum should be {} (std:{})".format(global_thdist, global_thdist_std, global_thdist_opt, global_thdist_opt_std))
            else:
                print("MW_TH_DISTANCE minimum value is {}".format(min(l_thdist_min)))
            print("MW_TH_DIST_FOREGROUND optimum should be {} (std:{})".format(global_thdistfore, global_thdistfore_std))
        if not use_minimum:
            return global_thbinmask_opt, global_thcontour_opt, global_thfore, global_thdist_opt, global_thdistfore
        else:
            return min(l_thbinmask_min), global_thcontour_opt, global_thfore, min(l_thdist_min), global_thdistfore
    else:
        if not use_minimum:
            return global_thbinmask_opt, global_thcontour_opt, global_thfore
        else:
            return min(l_thbinmask_min), global_thcontour_opt, global_thfore


def create_th_plot(ths, y_list, th_name="TH_BINARY_MASK", chart_dir=None, per_sample=True, ideal_value=None):
    """Create plots for threshold value calculation.

       Parameters
       ----------
       ths : List of floats
           List of thresholds. It will be the ``x`` axis.

       y_list : List of ints/floats
           Values of ``y`` axis.

       th_name : str, optional
           Name of the threshold.

       chart_dir : str, optional
           Path where the charts are stored.

       per_sample : bool, optional
           Create the plot per list in ``y_list``.

       ideal_value : int/float, optional
           Value that should be the ideal optimum. It is going to be marked with a red line in the chart.
    """

    assert th_name in ['TH_BINARY_MASK', 'TH_CONTOUR', 'TH_FOREGROUND', 'TH_DISTANCE', 'TH_DIST_FOREGROUND']
    fig, ax = plt.subplots(figsize=(25,10))
    ths = [str(i) for i in ths]
    num_points=len(ths)

    N = len(y_list)
    colors = list(range(0,N))
    c_labels = list("vol_"+str(i) for i in range(0,N))
    if per_sample:
        for i in range(N):
            l = '_nolegend_'  if i > 30 else c_labels[i]
            ax.plot(ths, y_list[i], label=l, alpha=0.4)

        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    else:
        y_list = np.array(y_list)
        y_mean = np.mean(y_list, axis=0)
        y_std = np.std(y_list, axis=0)
        ax.plot(ths, y_mean, label="sample (mean)")
        plt.fill_between(ths, y_mean-y_std, y_mean+y_std, alpha=0.25)
        if ideal_value is not None:
            plt.axhline(ideal_value, color='r')
            trans = transforms.blended_transform_factory(ax.get_yticklabels()[0].get_transform(), ax.transData)
            ax.text(1.12,ideal_value, "Ideal (mean)", color="red", transform=trans, ha="right", va="center")
        ax.legend(loc='center right')

    # Set labels of x axis
    plt.xticks(ths)
    a = np.arange(num_points)
    ax.xaxis.set_ticks(a)
    ax.xaxis.set_ticklabels(ths)

    plt.title('Threshold '+str(th_name))
    plt.xlabel("Threshold")
    if th_name == 'TH_FOREGROUND' or th_name == 'TH_DIST_FOREGROUND':
        plt.ylabel("IoU")
    else:
        plt.ylabel("Number of objects")
    p = "_per_validation_sample" if per_sample else ""
    plt.savefig(os.path.join(chart_dir, str(th_name)+p+".svg"), format = 'svg', dpi=100)
    plt.show()


def voronoi_on_mask(data, mask, th=0, verbose=False):
    """Apply Voronoi to the voxels not labeled yet marked by the mask. It is done using distances from the un-labeled
       voxels to the cell perimeters.

       Parameters
       ----------
       data : 2D/3D Numpy array
           Data to apply Voronoi. ``(y, x)`` for 2D or ``(z, y, x)`` for 3D. 
           E.g. ``(397, 1450, 2000)`` for 3D.

       mask : 3D/4D Numpy array
           Data mask to determine which points need to be proccessed. ``(z, y, x, channels)`` e.g.
           ``(397, 1450, 2000, 3)``.

       th : float, optional
           Threshold used to binarize the input. If th=0, otsu threshold is used.

       thres_small : int, optional
           Theshold to remove small objects created by the watershed.

       verbose : bool, optional
            To print saving information.

       Returns
       -------
       data : 4D Numpy array
           Image with Voronoi applied. ``(num_of_images, z, y, x)`` e.g. ``(1, 397, 1450, 2000)``
    """

    if data.ndim != 2 and data.ndim != 3:
        raise ValueError("Data must be 2/3 dimensional, provided {}".format(data.shape))
    if mask.ndim != 3 and mask.ndim != 4:
        raise ValueError("Data mask must be 3/4 dimensional, provided {}".format(mask.shape))
    if mask.shape[-1] < 2:
        raise ValueError("Mask needs to have two channels at least, received {}".format(mask.shape[-1]))

    if verbose:
        print("Applying Voronoi {}D . . .".format(data.ndim))
        
    image3d = False if data.ndim != 2 else True

    if image3d:
        data = np.expand_dims(data,0)
        mask = np.expand_dims(mask,0)

	# Extract mask from prediction
    if mask.shape[-1] == 3:
        mask = mask[...,2]
    else:
        mask = mask[...,0]+mask[...,1]

    mask_shape = np.shape(mask)

    # Binarize
    if th == 0:
        thresh = threshold_otsu(mask)
    else:
        thresh = th
    binaryMask = mask > thresh

    # Close to fill holes
    closedBinaryMask = morphology.closing(binaryMask, morphology.ball(radius=5)).astype(np.uint8)

    voronoiCyst = data*closedBinaryMask
    binaryVoronoiCyst = (voronoiCyst > 0)*1
    binaryVoronoiCyst = binaryVoronoiCyst.astype('uint8')

    # Cell Perimeter
    erodedVoronoiCyst = morphology.binary_erosion(binaryVoronoiCyst, morphology.ball(radius=2))
    cellPerimeter = binaryVoronoiCyst - erodedVoronoiCyst

    # Define ids to fill where there is mask but no labels
    idsToFill = np.argwhere((closedBinaryMask==1) & (data==0))
    labelPerId = np.zeros(np.size(idsToFill))

    idsPerim = np.argwhere(cellPerimeter==1)
    labelsPerimIds = voronoiCyst[cellPerimeter==1]

    # Generating voronoi
    for nId in tqdm(range(1,len(idsToFill))):
        distCoord = cdist([idsToFill[nId]], idsPerim)
        idSeedMin = np.argwhere(distCoord==np.min(distCoord))
        idSeedMin = idSeedMin[0][1]
        labelPerId[nId] = labelsPerimIds[idSeedMin]
        voronoiCyst[idsToFill[nId][0], idsToFill[nId][1], idsToFill[nId][2]] = labelsPerimIds[idSeedMin]

    if image3d:
        data = data[0]
        mask = mask[0]
        voronoiCyst = voronoiCyst[0]

    return voronoiCyst


def remove_close_points(points, radius, resolution, classes=None, ndim=3):
    """Remove all points from ``point_list`` that are at a ``radius``
       or less distance from each other.

       Parameters
       ----------
       points : ndarray of floats
           List of 3D points. E.g. ``((0,0,0), (1,1,1)``.

       radius : float
           Radius from each point to decide what points to keep. E.g. ``10.0`.

       resolution : ndarray of floats
           Resolution of the data, in `(z,y,x)` to calibrate coordinates.
           E.g. ``[30,8,8]``.    

       ndim : int, optional
           Number of dimension of the data.

       Returns
       -------
       new_point_list : List of floats
           New list of points after removing those at a distance of ``radius``
           or less from each other.
    """
    print("Removing close points . . .")
    print('Initial number of points: ' + str( len( points ) ) )

    point_list = points.copy()
    if classes is not None:
        class_list = classes.copy()

    # Resolution adjust
    for i in range(len(point_list)):
        point_list[i][0] = point_list[i][0]* resolution[0]
        point_list[i][1] = point_list[i][1]* resolution[1]
        if ndim == 3:
            point_list[i][2] = point_list[i][2]* resolution[2]

    mynumbers = [tuple(point) for point in point_list] 

    if len(mynumbers) == 0:
        return []
        
    tree = cKDTree(mynumbers) # build k-dimensional tree
    
    pairs = tree.query_pairs( radius ) # find all pairs closer than radius
    
    neighbors = {} # create dictionary of neighbors

    for i,j in pairs: # iterate over all pairs
        if i not in neighbors:
            neighbors[i] = {j}
        else:
            neighbors[i].add(j)
        if j not in neighbors:
            neighbors[j] = {i}
        else:
            neighbors[j].add(i)
            
    positions = [i for i in range(0, len( point_list ))]
    
    keep = []
    discard = set()
    for node in positions:
        if node not in discard: # if node already in discard set: skip
            keep.append(node) # add node to keep list
            discard.update(neighbors.get(node,set())) # add node's neighbors to discard set

    # points to keep
    new_point_list = [points[i] for i in keep]
    print( 'Final number of points: ' + str( len( new_point_list ) ) )
    
    if classes is not None:
        new_class_list = [classes[i] for i in keep]
        return new_point_list, new_class_list
    else:
        return new_point_list
    

def detection_watershed(seeds, coords, data_filename, first_dilation, nclasses=1, ndim=3, donuts_classes=[-1], donuts_patch=[13,120,120],
    donuts_nucleus_diameter=30, save_dir=None):
    """
    Grow given detection seeds.
     
    Parameters
    ----------
    seeds : 4D Numpy array
        Binary foreground labels and contours data to apply watershed into. E.g. ``(397, 1450, 2000, 2)``.

    coords : List of 3 ints
        Coordinates of all detected points. 

    data_filename : str
        Path to load the image paired with seeds. 

    first_dilation : str
        Each class seed's dilation before watershed.

    nclasses : int, optional
        Number of classes.

    ndim : int, optional
        Number of dimensions. E.g. for 2D set it to ``2`` and for 3D to ``3``. 

    donuts_classes : List of ints, optional
        Classes to check a donuts type cell. Set to ``-1`` to disable it. 

    donuts_patch : List of ints, optional
        Patch to analize donuts cells. Give a shape that covers all sizes of this type of 
        cells. 

    donuts_nucleus_diameter : int, optional
        Aproximate nucleus diameter for donuts type cells. 

    save_dir :  str, optional
        Directory to save watershed output into.

    Returns
    -------
    segm : 4D Numpy array
        Image with Voronoi applied. ``(num_of_images, z, y, x)`` e.g. ``(1, 397, 1450, 2000)``
    """
    print("Applying detection watershed . . .")
    
    # Read the test image
    img = imread(data_filename)
    img = reduce_dtype(img, np.min(img), np.max(img), out_min=0, out_max=255, out_type=np.uint8)

    # Adjust shape
    img = np.squeeze(img)
    if ndim == 2:
        if img.ndim == 3:
            if img.shape[0] <= 3: 
                img = img.transpose((1,2,0))  
                img = np.mean(img, -1)
    else: 
        if img.ndim == 4: 
            if img.shape[0] <= 3: 
                img = img.transpose((1,2,3,0))
                img = np.mean(img, -1)

    img = equalize_adapthist(img)
    
    # Dilate first the seeds if needed 
    print("Dilating a bit the seeds . . .")
    seeds = seeds.squeeze()
    dilated = False
    new_seeds = np.zeros(seeds.shape, dtype=seeds.dtype)
    for i in range(nclasses):
        if all(x != 0 for x in first_dilation[i]):
            new_seeds += (binary_dilation(seeds == i+1, structure=np.ones(first_dilation[i]))*(i+1)).astype(np.uint8)
            dilated = True
        else:
            new_seeds += ((seeds == i+1)*(i+1)).astype(np.uint8)
    if dilated:
        seeds = np.clip(new_seeds, 0, nclasses)
    seeds = new_seeds 
    del new_seeds

    # Background seed 
    seeds = label(seeds)
    max_seed = np.max(seeds)
    if ndim == 2:
        seeds[:4,:4] = max_seed+1
        background_label = seeds[1,1]
    else:
        seeds[0,:4,:4] = max_seed+1 
        background_label = seeds[0,1,1]
    
    # Try to dilate those instances that have 'donuts' like shape and that might have problems with the watershed
    if donuts_classes[0] != -1:
        for dclass in donuts_classes:
            class_coords = coords[dclass-1]
            nticks = [x//8 for x in donuts_patch]
            nticks = [x+(1-x%2) for x in nticks]
            half_spatch = [x//2 for x in donuts_patch]

            class_check_dir = os.path.join(save_dir, "class_{}_check".format(dclass))

            for i in tqdm(range(len(class_coords)), leave=False):
                c = class_coords[i] 

                # Patch coordinates
                l = seeds[c[0],c[1],c[2]]
                if ndim == 2:
                    y1, y2 = max(c[0]-half_spatch[0],0),min(c[0]+half_spatch[0],img.shape[0])
                    x1, x2 = max(c[1]-half_spatch[1],0),min(c[1]+half_spatch[1],img.shape[1])
                    img_patch = img[y1:y2,x1:x2]
                    seed_patch = seeds[y1:y2,x1:x2]

                    # Extract horizontal and vertical line
                    line_y = img_patch[:, half_spatch[1]]
                    line_x = img_patch[half_spatch[0], :]
                else:
                    z1, z2 = max(c[0]-half_spatch[0],0),min(c[0]+half_spatch[0],img.shape[0])
                    y1, y2 = max(c[1]-half_spatch[1],0),min(c[1]+half_spatch[1],img.shape[1])
                    x1, x2 = max(c[2]-half_spatch[2],0),min(c[2]+half_spatch[2],img.shape[2])
                    img_patch = img[z1:z2,y1:y2,x1:x2]
                    seed_patch = seeds[z1:z2,y1:y2,x1:x2]
                    
                    # Extract horizontal and vertical line
                    line_y = img_patch[half_spatch[0],:, half_spatch[2]]
                    line_x = img_patch[half_spatch[0], half_spatch[1], :]

                fillable_patch = seed_patch.copy()
                seed_patch = (seed_patch == l)*l
                fillable_patch = (fillable_patch == 0)

                aux = np.expand_dims(np.expand_dims((img_patch).astype(np.float32), -1),0)
                save_tif(aux, class_check_dir, ["{}_patch.tif".format(l)], verbose=False)

                # Save the verticial and horizontal lines in the patch to debug 
                patch_y = np.zeros(img_patch.shape, dtype=np.float32)
                if ndim == 2:
                    patch_y[:, half_spatch[1]] = img_patch[:, half_spatch[1]]
                else:
                    patch_y[half_spatch[0],:, half_spatch[2]] = img_patch[half_spatch[0],:, half_spatch[2]]

                aux = np.expand_dims(np.expand_dims((patch_y).astype(np.float32), -1),0)
                save_tif(aux, class_check_dir, ["{}_y_line.tif".format(l)], verbose=False)

                patch_x = np.zeros(img_patch.shape, dtype=np.float32)
                if ndim == 2:
                    patch_x[half_spatch[0], :] = img_patch[half_spatch[0], :]
                else:
                    patch_x[half_spatch[0], half_spatch[1], :] = img_patch[half_spatch[0], half_spatch[1], :]
                aux = np.expand_dims(np.expand_dims((patch_x).astype(np.float32), -1),0)
                save_tif(aux, class_check_dir, ["{}_x_line.tif".format(l)], verbose=False)
                # Save vertical and horizontal line plots to debug 
                plt.title("Line graph")
                plt.plot(list(range(len(line_y))), line_y, color="red")
                plt.savefig(os.path.join(class_check_dir, "{}_line_y.png".format(l)))
                plt.clf()
                plt.title("Line graph")
                plt.plot(list(range(len(line_x))), line_x, color="red")
                plt.savefig(os.path.join(class_check_dir, "{}_line_x.png".format(l)))
                plt.clf()

                # Smooth them to analize easily
                line_y = savgol_filter(line_y, nticks[1], 2)
                line_x = savgol_filter(line_x, nticks[2], 2)

                # Save vertical and horizontal lines again but now filtered
                plt.title("Line graph")
                plt.plot(list(range(len(line_y))), line_y, color="red")
                plt.savefig(os.path.join(class_check_dir, "{}_line_y_filtered.png".format(l)))
                plt.clf()
                plt.title("Line graph")
                plt.plot(list(range(len(line_x))), line_x, color="red")
                plt.savefig(os.path.join(class_check_dir, "{}_line_x_filtered.png".format(l)))
                plt.clf()

                # Find maximums 
                peak_y, _ = find_peaks(line_y)
                peak_x, _ = find_peaks(line_x)

                # Find minimums 
                mins_y, _ = find_peaks(-line_y)
                mins_x, _ = find_peaks(-line_x)

                # Find the donuts shape cells
                # Vertical line
                mid = len(line_y)//2
                mid_value = line_y[min(mins_y, key=lambda x:abs(x-mid))]
                found_left_peak, found_right_peak = False, False
                max_right, max_left = 0., 0.
                max_right_pos, max_left_pos= -1, -1
                for peak_pos in peak_y:
                    if line_y[peak_pos] >= mid_value*1.5:
                        # Left side
                        if peak_pos <= mid:
                            found_left_peak = True
                            if line_y[peak_pos] > max_left: 
                                max_left = line_y[peak_pos]
                                max_left_pos = peak_pos
                        # Right side
                        else:
                            found_right_peak = True  
                            if line_y[peak_pos] > max_right: 
                                max_right = line_y[peak_pos]
                                max_right_pos = peak_pos
                ushape_in_liney = (found_left_peak and found_right_peak)
                y_diff_dilation = max_right_pos-max_left_pos
                if ushape_in_liney:
                    y_left_gradient = min(line_y[:max_left_pos]) < max_left*0.7
                    y_right_gradient = min(line_y[max_right_pos:]) < max_right*0.7

                # Horizontal line
                mid = len(line_x)//2
                mid_value = line_x[min(mins_x, key=lambda x:abs(x-mid))]
                found_left_peak, found_right_peak = False, False
                max_right, max_left = 0., 0.
                max_right_pos, max_left_pos= -1, -1
                for peak_pos in peak_x:
                    if line_x[peak_pos] >= mid_value*1.5:
                        # Left side
                        if peak_pos <= mid:
                            found_left_peak = True
                            if line_x[peak_pos] > max_left: 
                                max_left = line_x[peak_pos]
                                max_left_pos = peak_pos
                        # Right side
                        else:
                            found_right_peak = True  
                            if line_x[peak_pos] > max_right: 
                                max_right = line_x[peak_pos]
                                max_right_pos = peak_pos
                ushape_in_linex = (found_left_peak and found_right_peak)
                x_diff_dilation = max_right_pos-max_left_pos
                if ushape_in_linex:
                    x_left_gradient = min(line_x[:max_left_pos]) < max_left*0.7
                    x_right_gradient = min(line_x[max_right_pos:]) < max_right*0.7

                # Donuts shape cell found
                if ushape_in_liney and ushape_in_linex:
                    # Calculate the dilation to be made based on the nucleus size 
                    if ndim == 2:
                        donuts_cell_dilation = [y_diff_dilation-first_dilation[dclass-1][0], x_diff_dilation-first_dilation[dclass-1][1]]
                        donuts_cell_dilation = [donuts_cell_dilation[0]-int(donuts_cell_dilation[0]*0.4), donuts_cell_dilation[1]-int(donuts_cell_dilation[1]*0.4)]
                    else:
                        donuts_cell_dilation = [first_dilation[dclass-1][0], y_diff_dilation-first_dilation[dclass-1][1], x_diff_dilation-first_dilation[dclass-1][2]]
                        donuts_cell_dilation = [donuts_cell_dilation[0], donuts_cell_dilation[1]-int(donuts_cell_dilation[1]*0.4),
                            donuts_cell_dilation[2]-int(donuts_cell_dilation[2]*0.4)]

                    # If the center is not wide the cell is not very large
                    dilate = True
                    if x_diff_dilation+y_diff_dilation < donuts_nucleus_diameter*2:
                        print("Instance {} has 'donuts' shape but it seems to be not very large!".format(l))
                    else:
                        print("Instance {} has 'donuts' shape!".format(l))
                        if not y_left_gradient:
                            print("    - Its vertical left part seems to have low gradient")
                            dilate = False
                        if not y_right_gradient:
                            print("    - Its vertical right part seems to have low gradient")
                            dilate = False
                        if not x_left_gradient:
                            print("    - Its horizontal left part seems to have low gradient")
                            dilate = False
                        if not x_right_gradient:
                            print("    - Its horizontal right part seems to have low gradient")
                            dilate = False   
                    if dilate:
                        if all(x > 0 for x in donuts_cell_dilation):
                            seed_patch = grey_dilation(seed_patch, footprint=np.ones((donuts_cell_dilation)))
                            if ndim == 2:
                                seeds[y1:y2,x1:x2] += seed_patch * fillable_patch
                            else:
                                seeds[z1:z2,y1:y2,x1:x2] += seed_patch * fillable_patch
                    else:
                        print("    - Not dilating it!")
                else:
                    print("Instance {} checked".format(l))
    
    print("Calculating gradient . . .")
    start = time.time()
    if ndim == 2:
        gradient = rank.gradient(img, disk(3)).astype(np.uint8)
    else:
        gradient = rank.gradient(img, ball(3)).astype(np.uint8)
    end = time.time()
    grad_elapsed = end - start
    print("Gradient took {} seconds".format(int(grad_elapsed)))

    print("Run watershed . . .")
    segm = watershed(gradient, seeds)
    
    # Remove background label
    segm[segm == background_label] = 0

    # Dilate a bit the instances 
    if ndim == 2:
        segm += dilation(segm, disk(5))*(segm==0)
        segm = erosion(segm, disk(3))
    else:
        for i in range(segm.shape[0]):
            dil_slice = (segm[i]==0)
            dil_slice = dilation(segm[i], disk(5))*dil_slice
            segm[i] += dil_slice
            segm[i] = erosion(segm[i], disk(2))
        
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        aux = np.expand_dims(np.expand_dims((img).astype(np.float32), -1),0)
        save_tif(aux, save_dir, ["img.tif"], verbose=False)

        aux = np.expand_dims(np.expand_dims((gradient).astype(np.float32), -1),0)
        save_tif(aux, save_dir, ["gradient.tif"], verbose=False)

        aux = np.expand_dims(np.expand_dims((seeds).astype(np.float32), -1),0)
        save_tif(aux, save_dir, ["seed_map.tif"], verbose=False)

        aux = np.expand_dims(np.expand_dims((segm).astype(np.float32), -1),0)
        save_tif(aux, save_dir, ["watershed.tif"], verbose=False)

    return segm

def remove_instance_by_circularity_central_slice(img, resolution, coords_list=None, circularity_th=0.7):
    """
    Check the properties of input image instances. Apart from label id, number of pixels, area/volume 
    (2D/3D respec. and taking into account the resolution) and circularity properties lists that are 
    returned another one identifying those instances that do not satisfy the circularity threshold
    are marked as 'Strange' whereas the rest are 'Correct'. For 3D the circularity is only measured in the center 
    slice of the instance, which is decided by the given coordinates in ``coords_list`` or calculated taking
    the central slice in z of each instance, which is computationally more expensive as the bboxes are calculated.   
    
    Parameters
    ----------
    img : 2D/3D Numpy array
        Image with instances. E.g. ``(1450, 2000)`` for 2D and ``(397, 1450, 2000)`` for 3D.

    resolution : str
        Path to load the image paired with seeds. 

    coords_list : List of 3 ints, optional
        Coordinates of all detected points. If 

    circularity_th : float, optional
        Circularity threshold value. Those instances that are below that value will be marked as 'Strange'
        in the returned comment list. 

    Returns
    ------- 
    img : 2D/3D Numpy array
        Input image without the instances that do not satisfy the circularity constraint. 
        Image with instances. E.g. ``(1450, 2000)`` for 2D and ``(397, 1450, 2000)`` for 3D.

    labels : Array of ints
        Instance label list. 
    
    npixels : Array of ints
        Number of pixels of each instance. 
        
    areas : Array of ints
        Areas/volumes (2D/3D) of each instance based on the given resolution.
    
    circularities : Array of ints
        Circularity of each instance.

    diameters : Array of ints
        Diameter of each instance obtained from the bounding box.

    comment : List of str
        List containing 'Correct' string when the instance surpass the circularity 
        threshold and 'Strange' otherwise.
    """
    print("Checking the circularity of instances . . .")

    image3d = True if img.ndim == 3 else False
    correct_str = "Correct"
    unsure_str = "Strange"

    comment = []
    label_list_coords = []
    label_list_unique, npixels = np.unique(img, return_counts=True)
    label_list_unique = label_list_unique[1:] # Delete background instance '0'
    npixels = npixels[1:]

    if coords_list is not None:
        total_labels = len(coords_list)
        
        # Obtain each instance labels based on the input image first
        for c in coords_list:
            if image3d:
                label_list_coords.append(img[c[0],c[1],c[2]])
            else:
                label_list_coords.append(img[c[1],c[2]])
            comment.append('none')

        areas = np.zeros(total_labels, dtype=np.uint32)
        diameters = np.zeros(total_labels, dtype=np.uint32)
        diam_calc = False
        # Insert in the label position the area/volume
        for i, pixels in enumerate(npixels):
            label = label_list_unique[i]
            label_index = label_list_coords.index(label)
            if image3d:
                vol = pixels*(resolution[0]+resolution[1]+resolution[2])
            else:
                vol = pixels*(resolution[0]+resolution[1])
            areas[label_index] = vol

    # If no coords_list is given it is calculated by each instance central slice in z
    else:
        label_list_coords = list(label_list_unique).copy()
        total_labels = len(label_list_unique)
        coords_list = [[] for i in range(total_labels)]
        comment = ['none' for i in range(total_labels)]
        areas = np.zeros(total_labels, dtype=np.uint32)
        diameters = np.zeros(total_labels, dtype=np.uint32)
        diam_calc = True

        props = regionprops_table(img, properties=('label', 'bbox')) 
        for k, label in enumerate(props['label']):
            label_index, = np.where(label_list_unique == label)[0]
            pixels = npixels[label_index]

            if image3d:
                z_coord_start = props['bbox-0'][k]
                z_coord_finish = props['bbox-3'][k]
                central_slice = (z_coord_start+z_coord_finish)//2

                vol = pixels*(resolution[0]+resolution[1]+resolution[2])

                diam = max(props['bbox-3'][k]-props['bbox-0'][k],props['bbox-4'][k]-props['bbox-1'][k],props['bbox-5'][k]-props['bbox-2'][k])
            else:
                central_slice = 0
                vol = pixels*(resolution[0]+resolution[1])
                diam = max(props['bbox-2'][k]-props['bbox-0'][k],props['bbox-3'][k]-props['bbox-1'][k])

            slices = []
            if central_slice-1 >= z_coord_start: slices.append(central_slice-1)
            slices.append(central_slice)
            if central_slice+1 <= z_coord_finish: slices.append(central_slice+1)
            coords_list[label_index] = slices.copy()
            areas[label_index] = vol
            diameters[label_index] = diam

    circularities = np.zeros(total_labels, dtype=np.float32)
    circularities_count = np.zeros(total_labels, dtype=np.uint8)
    print("{} instances found before circularity filtering".format(total_labels))

    if not image3d:
        img = np.expand_dims(img,0)  

    # Circularity calculation in the slice where the central point was considerer by the model
    # which is marked by coords_list
    labels_removed = 0
    for i in tqdm(range(img.shape[0]), leave=False):
        props = regionprops_table(img[i], properties=('label','area', 'perimeter', 'bbox'))             
        if len(props['label'])>0:                           
            for k, l in enumerate(props['label']):
                multiple_slices = False
                j = label_list_coords.index(l)
                if image3d:
                    coord = coords_list[j]
                    
                    if len(coord) > 1:
                        coord = [coord[0]]
                        multiple_slices = True

                    if not diam_calc:
                        diam = max(props['bbox-3'][k]-props['bbox-0'][k],props['bbox-4'][k]-props['bbox-1'][k],props['bbox-5'][k]-props['bbox-2'][k])
                else:
                    coord = [0]
                    if not diam_calc:
                        diam = max(props['bbox-2'][k]-props['bbox-0'][k],props['bbox-3'][k]-props['bbox-1'][k])

                # If instances' center point matches the slice i save the circularity
                if coord[0] == i: 
                    if props['perimeter'][k] != 0:
                        circularity = (4 * math.pi * props['area'][k]) / (props['perimeter'][k]*props['perimeter'][k])     
                    else:
                        circularity = 0         
                    circularities[j] += circularity
                    circularities_count[j] += 1

                    if multiple_slices:
                        v = coords_list[j].pop(0)
                        coords_list[j].append(v)

                if not diam_calc:
                    diameters[j] = diam

    # Remove those instances that do not pass the threshold        
    for i in tqdm(range(len(circularities)), leave=False):
        circularities[i] = circularities[i]/circularities_count[i] if circularities_count[i] != 0 else 0
        if circularities[i] > circularity_th:
            comment[i] = correct_str
        else:
            comment[i] = unsure_str
            # Remove that label from the image
            img[img==label_list_coords[i]] = 0
            labels_removed += 1

    if not image3d:
        img = img[0]

    print("Removed {} instances by circularity, {} instances left".format(labels_removed, total_labels-labels_removed))

    return img, label_list_coords, npixels, areas, circularities, diameters, comment
    
def find_neighbors(img, label, neighbors=1):
    """
    Find neighbors of a label in a given image. 
    
    Parameters
    ----------
    img : 2D/3D Numpy array
        Image with instances. E.g. ``(1450, 2000)`` for 2D and ``(397, 1450, 2000)`` for 3D.

    label : int
        Label to find the neighbors of. 

    neighbors : int, optional
        Number of neighbors in each axis to explore.

    Returns
    ------- 
    neighbors  : list of ints
        Neighbors instance ids of the given label. 
    """ 

    list_of_neighbors = []
    label_points = np.where((img==label)>0) 
    if img.ndim == 3:
        for p in range(len(label_points[0])):
            coord = [label_points[0][p],label_points[1][p],label_points[2][p]]
            for i in range(-neighbors,neighbors+1):
                for j in range(-neighbors,neighbors+1):
                    for k in range(-neighbors,neighbors+1):
                        z = min(max(coord[0]+i,0),img.shape[0]-1)
                        y = min(max(coord[1]+j,0),img.shape[1]-1)
                        x = min(max(coord[2]+k,0),img.shape[2]-1)
                        if img[z,y,x] not in list_of_neighbors and img[z,y,x] != label and img[z,y,x] != 0:
                            list_of_neighbors.append(img[z,y,x]) 
    else:
        for p in range(len(label_points[0])):
            coord = [label_points[0][p],label_points[1][p]]
            for i in range(-neighbors,neighbors+1):
                for j in range(-neighbors,neighbors+1):
                        y = min(max(coord[0]+i,0),img.shape[0]-1)
                        x = min(max(coord[1]+j,0),img.shape[1]-1)
                        if img[y,x] not in list_of_neighbors and img[y,x] != label and img[y,x] != 0:
                            list_of_neighbors.append(img[y,x])                            
    return list_of_neighbors
    
def repare_large_blobs(img, size_th=10000):
    """
    Try to repare large instances by merging neighbors ones with it and by removing possible central holes.  

    Parameters
    ----------
    img : 2D/3D Numpy array
        Image with instances. E.g. ``(1450, 2000)`` for 2D and ``(397, 1450, 2000)`` for 3D.

    size_th : int, optional
        Size that the instances need to be larger than to be analised.

    Returns
    ------- 
    img : 2D/3D Numpy array
        Input image without the large instances repaired. E.g. ``(1450, 2000)`` for 2D and 
        ``(397, 1450, 2000)`` for 3D.
    """
    print("Reparing large instances (more than {} pixels) . . .".format(size_th))
    image3d = True if img.ndim == 3 else False

    props = regionprops_table(img, properties=('label', 'area', 'bbox'))
    for k, l in tqdm(enumerate(props['label']), total=len(props['label']), leave=False):
        if props['area'][k] >= size_th:
            if image3d:
                sz,fz,sy,fy,sx,fx = props['bbox-0'][k],props['bbox-3'][k],props['bbox-1'][k],props['bbox-4'][k],props['bbox-2'][k],props['bbox-5'][k]
                patch = img[sz:fz,sy:fy,sx:fx].copy()
            else:
                sy,fy,sx,fx = props['bbox-0'][k],props['bbox-2'][k],props['bbox-1'][k],props['bbox-3'][k]
                patch = img[sy:fy,sx:fx].copy()

            inst_patches, inst_pixels = np.unique(patch, return_counts=True)
            if len(inst_patches) > 2:
                neighbors = find_neighbors(patch, l)

                # Merge neighbors with the big label
                for i in range(len(neighbors)):
                    ind = np.where(props['label'] == neighbors[i])[0] 

                    # Only merge labels if the small neighbor instance is fully contained in the large one
                    contained_in_large_blob = True
                    if image3d:
                        neig_sz, neig_fz= props['bbox-0'][ind],props['bbox-3'][ind]
                        neig_sy, neig_fy= props['bbox-1'][ind],props['bbox-4'][ind]
                        neig_sx, neig_fx= props['bbox-2'][ind],props['bbox-5'][ind]

                        if neig_sz < sz or neig_fz > fz or neig_sy < sy or neig_fy > fy or neig_sx < sx or neig_fx > fx: 
                            neigbor_ind_in_patch = list(inst_patches).index(neighbors[i])
                            pixels_in_patch = inst_pixels[neigbor_ind_in_patch]
                            # pixels outside the patch of that neighbor are greater than 30% means that probably it will 
                            # represent another blob so do not merge 
                            if (props['area'][ind][0]-pixels_in_patch)/props['area'][ind][0]>0.30:
                                contained_in_large_blob = False
                    else:
                        neig_sy, neig_fy= props['bbox-0'][ind],props['bbox-2'][ind]
                        neig_sx, neig_fx= props['bbox-1'][ind],props['bbox-3'][ind]
                        if neig_sy < sy or neig_fy > fy or neig_sx < sx or neig_fx > fx: 
                            contained_in_large_blob = False

                    if contained_in_large_blob:
                        img[img==neighbors[i]] = l
                    
            # Fills holes 
            if image3d:
                patch = img[sz:fz,sy:fy,sx:fx].copy()   
            else:
                patch = img[sy:fy,sx:fx].copy() 
            only_label_patch = patch.copy()
            only_label_patch[only_label_patch!=l] = 0    
            if image3d:
                for i in range(only_label_patch.shape[0]):
                    only_label_patch[i] = fill_voids.fill(only_label_patch[i])*l
            else:
                only_label_patch = fill_voids.fill(only_label_patch)*l
            patch[patch==l] = 0
            patch += only_label_patch
            if image3d:
                img[sz:fz,sy:fy,sx:fx] = patch
            else:
                img[sy:fy,sx:fx] = patch
    return img
        

def apply_binary_mask(X, bin_mask_dir):
    """Apply a binary mask to remove values outside it.

       Parameters
       ----------
       X : 3D/4D Numpy array
           Data to apply the mask. E.g. ``(y, x, channels)`` for 2D or ``(z, y, x, channels)`` for 3D.

       bin_mask_dir : str, optional
           Directory where the binary mask are located.

       Returns
       -------
       X : 3D/4D Numpy array
           Data with the mask applied. E.g. ``(y, x, channels)`` for 2D or ``(z, y, x, channels)`` for 3D.
    """

    if X.ndim != 4 and X.ndim != 3:
        raise ValueError("'X' needs to have 3 or 4 dimensions and not {}".format(X.ndim))
    
    print("Applying binary mask(s) from {}".format(bin_mask_dir))

    ids = sorted(next(os.walk(bin_mask_dir))[2])

    if len(ids) == 1:
        one_file = True
        print("It is assumed that the mask found {} is valid for all 'X' data".format(os.path.join(bin_mask_dir, ids[0])))
    else:
        one_file = False

    if one_file:
        mask = imread(os.path.join(bin_mask_dir, ids[0]))
        mask = np.squeeze(mask)

        if X.ndim != mask.ndim +1 and X.ndim != mask.ndim +2:
            raise ValueError("Mask found has {} dims, shape: {}. Need to be of {} or {} dims instead"
                .format(mask.ndim, mask.shape, mask.ndim+1,mask.ndim+2))

        if mask.ndim == X.ndim-1:
            for c in range(X.shape[-1]):
                X[...,c] = X[...,c]*(mask>0)
        else: # mask.ndim == 2 and X.ndim == 4
            for k in range(X.shape[0]):
                for c in range(X.shape[-1]):
                    X[k,...,c] = X[k,...,c]*(mask>0)
    else:
        for i in tqdm(range(len(ids))):
            mask = imread(os.path.join(bin_mask_dir, ids[i]))
            mask = np.squeeze(mask)

            if X.ndim != mask.ndim +1 and X.ndim != mask.ndim +2:
                raise ValueError("Mask found has {} dims, shape: {}. Need to be of {} or {} dims instead"
                    .format(mask.ndim, mask.shape, mask.ndim+1,mask.ndim+2))
                    
            if mask.ndim == X.ndim-1:
                for c in range(X.shape[-1]):
                    X[...,c] = X[...,c]*(mask>0)
            else: # mask.ndim == 2 and X.ndim == 4
                for k in range(X.shape[0]):
                    for c in range(X.shape[-1]):
                        X[k,...,c] = X[k,...,c]*(mask>0)
    return X
