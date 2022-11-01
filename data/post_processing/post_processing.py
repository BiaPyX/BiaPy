import cv2
import os
import math
import statistics
import sys
import numpy as np
from tqdm import tqdm
from scipy import ndimage as ndi
from skimage import morphology
from skimage.morphology import disk, remove_small_objects
from skimage.segmentation import watershed
from skimage.filters import rank, threshold_otsu
from scipy.ndimage import rotate
from skimage.measure import label
from skimage.io import imsave, imread
from scipy.ndimage.morphology import binary_erosion, binary_dilation
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy_indexed as npi
from scipy.spatial import KDTree, cKDTree
from scipy.spatial.distance import cdist

from engine.metrics import jaccard_index_numpy
from utils.util import pad_and_reflect
from data.pre_processing import normalize
from data.data_3D_manipulation import crop_3D_data_with_overlap


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

       +---------------------------------------------+---------------------------------------------+
       | .. figure:: ../../img/FIBSEM_test_0.png     | .. figure:: ../../img/FIBSEM_test_0_gt.png  |
       |   :width: 80%                               |   :width: 80%                               |
       |   :align: center                            |   :align: center                            |
       |                                             |                                             |
       |   Original image                            |   Ground truth                              |
       +---------------------------------------------+---------------------------------------------+
       | .. figure:: ../../img/FIBSEM_test_0_pred.png| .. figure:: ../../img/FIBSEM_test_0_wa.png  |
       |   :width: 80%                               |   :width: 80%                               |
       |   :align: center                            |   :align: center                            |
       |                                             |                                             |
       |   Predicted image                           |   Watershed ouput                           |
       +---------------------------------------------+---------------------------------------------+

       The marks used to guide the watershed is this example are these:

        .. image:: ../../img/watershed2_marks_test0.png
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


def bc_watershed(data, thres1=0.9, thres2=0.8, thres3=0.85, thres_small=128, remove_before=False, save_dir=None):
    """Convert binary foreground probability maps and instance contours to instance masks via watershed segmentation
       algorithm.

       Implementation based on `PyTorch Connectomics' process.py
       <https://github.com/zudi-lin/pytorch_connectomics/blob/master/connectomics/utils/process.py>`_.

       Parameters
       ----------
       data : 4D Numpy array
           Binary foreground labels and contours data to apply watershed into. E.g. ``(397, 1450, 2000, 2)``.

       thres1 : float, optional
           Threshold used in the semantic mask to create watershed seeds.

       thres2 : float, optional
           Threshold used in the contours to create watershed seeds.

       thres3 : float, optional
           Threshold used in the semantic mask to create the foreground mask.

       thres_small : int, optional
           Theshold to remove small objects created by the watershed.

       remove_before : bool, optional
           To remove objects before watershed. If ``False`` it is done after watershed.

       save_dir :  str, optional
           Directory to save watershed output into.
    """

    v = 255 if np.max(data) <= 1 else 1
    semantic = data[...,0]*v
    seed_map = (data[...,0]*v > int(255*thres1)) * (data[...,1]*v < int(255*thres2))
    foreground = (semantic > int(255*thres3))
    seed_map = label(seed_map, connectivity=1)

    if remove_before:
        seed_map = remove_small_objects(seed_map, thres_small)
        segm = watershed(-semantic, seed_map, mask=foreground)
    else:
        segm = watershed(-semantic, seed_map, mask=foreground)
        segm = remove_small_objects(segm, thres_small)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        f = os.path.join(save_dir, "seed_map.tif")
        aux = np.expand_dims(np.expand_dims((seed_map).astype(np.float32), -1),1)
        imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False)

        f = os.path.join(save_dir, "foreground.tif")
        aux = np.expand_dims(np.expand_dims((foreground).astype(np.float32), -1),1)
        imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False)

        f = os.path.join(save_dir, "watershed.tif")
        aux = np.expand_dims(np.expand_dims((segm).astype(np.float32), -1),1)
        imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False)

    return segm


def bcd_watershed(data, thres1=0.9, thres2=0.8, thres3=0.85, thres4=0.5, thres5=0.0, thres_small=128,
                  remove_before=False, save_dir=None):
    """Convert binary foreground probability maps, instance contours to instance masks via watershed segmentation
       algorithm.

       Implementation based on `PyTorch Connectomics' process.py
       <https://github.com/zudi-lin/pytorch_connectomics/blob/master/connectomics/utils/process.py>`_.

       Parameters
       ----------
       data : 4D Numpy array
           Binary foreground labels and contours data to apply watershed into. E.g. ``(397, 1450, 2000, 2)``.

       thres1 : float, optional
           Threshold used in the semantic mask to create watershed seeds.

       thres2 : float, optional
           Threshold used in the contours to create watershed seeds.

       thres3 : float, optional
           Threshold used in the semantic mask to create the foreground mask.

       thres4 : float, optional
           Threshold used in the distances to create watershed seeds.

       thres5 : float, optional
           Threshold used in the distances to create the foreground mask.

       thres_small : int, optional
           Theshold to remove small objects created by the watershed.

       remove_before : bool, optional
           To remove objects before watershed. If ``False`` it is done after watershed.

       save_dir :  str, optional
           Directory to save watershed output into.
    """

    v = 255 if np.max(data[...,:2]) <= 1 else 1
    semantic = data[...,0]*v
    seed_map = (data[...,0]*v > int(255*thres1)) * (data[...,1]*v < int(255*thres2)) * (data[...,2] > thres4)
    foreground = (semantic > int(255*thres3)) * (data[...,2] > thres5)
    seed_map = label(seed_map, connectivity=1)

    if remove_before:
        seed_map = remove_small_objects(seed_map, thres_small)
        segm = watershed(-semantic, seed_map, mask=foreground)
    else:
        segm = watershed(-semantic, seed_map, mask=foreground)
        segm = remove_small_objects(segm, thres_small)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        f = os.path.join(save_dir, "semantic.tif")
        aux = np.expand_dims(np.expand_dims((semantic).astype(np.float32), -1),1)
        imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False)

        f = os.path.join(save_dir, "seed_map.tif")
        aux = np.expand_dims(np.expand_dims((seed_map).astype(np.float32), -1),1)
        imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False)

        f = os.path.join(save_dir, "foreground.tif")
        aux = np.expand_dims(np.expand_dims((foreground).astype(np.float32), -1),1)
        imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False)

        f = os.path.join(save_dir, "watershed.tif")
        aux = np.expand_dims(np.expand_dims((segm).astype(np.float32), -1),1)
        imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False)

    return segm


def bdv2_watershed(data, bin_th=0.2, thres_small=128, remove_before=False, save_dir=None):
    """Convert binary foreground probability maps, instance contours to instance masks via watershed segmentation
       algorithm.

       Implementation based on `PyTorch Connectomics' process.py
       <https://github.com/zudi-lin/pytorch_connectomics/blob/master/connectomics/utils/process.py>`_.

       Parameters
       ----------
       data : 4D Numpy array
           Binary foreground labels and contours data to apply watershed into. E.g. ``(397, 1450, 2000, 2)``.

       bin_th : float, optional
           Threshold used to binarize the input.

       thres_small : int, optional
           Theshold to remove small objects created by the watershed.

       remove_before : bool, optional
           To remove objects before watershed. If ``False`` it is done after watershed.

       save_dir :  str, optional
           Directory to save watershed output into.
    """

    if data.shape[-1] == 3:
        # Label all the instance seeds
        seed_map = (data[...,0] > bin_th) * (data[...,1] < bin_th)
        seed_map, num = label(seed_map, return_num=True)

        # Create background seed and label correctly
        background_seed = binary_dilation(((data[...,0]+data[...,1]) > bin_th).astype(int), iterations=2)
        background_seed = 1 - background_seed
        background_seed[background_seed==1] = num+1

        seed_map = seed_map + background_seed
        del background_seed
    # Assume is 'Dv2'
    else:
        seed_map = data[...,0] < bin_th
        seed_map = label(seed_map)

    if remove_before:
        seed_map = remove_small_objects(seed_map, thres_small)
        segm = watershed(data[...,-1], seed_map)
    else:
        segm = watershed(data[...,-1], seed_map)
        seed_map = remove_small_objects(seed_map, thres_small)

    if data.shape[-1] == 3:
        # Change background instance value to 0 again
        segm[segm == num+1] = 0

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        f = os.path.join(save_dir, "seed_map.tif")
        aux = np.expand_dims(np.expand_dims((seed_map).astype(np.float32), -1),1)
        imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False)

    return segm


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
           Z filtered data. E.g. ``(num_of_images, y, x, channels)``.
    """

    out_data = np.copy(data)

    # Must be odd
    if mf_size % 2 == 0:
       mf_size += 1

    for i in range(data.shape[0]):
        sl = (data[i]).astype(np.float32)
        sl = cv2.medianBlur(sl, mf_size)
        sl = np.expand_dims(sl,-1) if sl.ndim == 2 else sl
        out_data[i] = sl

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
           Return the minimum value of TH1 (and TH4) instead of the mean.

       chart_dir : str, optional
           Path where the charts are stored.

       verbose : bool, optional
           To print saving information.

       Return
       ------
       global_th1_min_opt: float
           MW_TH1 optimum value.

       global_th2_opt : float
           MW_TH2 optimum value.

       global_th3 : float
           MW_TH3 optimum value.

       global_th4_min_opt : float, optional
           MW_TH4 optimum value.

       global_th5 : float, optional
           MW_TH5 optimum value.
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

    min_th1 = 1
    g_l_th1 = []
    l_th1_min = []
    l_th1_opt = []
    g_l_th2 = []
    l_th2_min = []
    l_th2_opt = []
    ideal_number_obj = []
    g_l_th3 = []
    l_th3_max = []
    if mode == 'BCD':
        g_l_th4 = []
        l_th4_min = []
        l_th4_opt = []
        g_l_th5 = []
        l_th5_max = []

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

                # TH3 and TH5:
                # Look at the best IoU compared with the original label. Only the region that involve the object is taken
                # into consideration. This is achieved dilating 2 iterations the original object mask. If we do not dilate
                # that label, decreasing the TH will always ensure a IoU >= than the previous TH. This way, the IoU will
                # reach a maximum and then it will start to decrease, as more points that are not from the object are added
                # into it
                # TH3
                th3_best = -1
                th3_max_jac = -1
                l_th3 = []
                for j in range(len(ths)):
                    p = np.expand_dims(pred[...,0] > ths[j],-1).astype(np.uint8)
                    jac = jaccard_index_numpy((mask>0).astype(np.uint8), p)
                    if jac > th3_max_jac:
                        th3_max_jac = jac
                        th3_best = ths[j]
                    l_th3.append(jac)
                l_th3_max.append(th3_best)
                g_l_th3.append(l_th3)
                # TH5
                if mode == 'BCD':
                    th5_best = -1
                    th5_max_jac = -1
                    l_th5 = []
                    for j in range(len(ths_dis)):
                        p = np.expand_dims(pred[...,2] > ths_dis[j],-1).astype(np.uint8)
                        jac = jaccard_index_numpy((mask>0).astype(np.uint8), p)
                        if jac > th5_max_jac:
                            th5_max_jac = jac
                            th5_best = ths_dis[j]
                        l_th5.append(jac)
                    l_th5_max.append(th5_best)
                    g_l_th5.append(l_th5)

                # TH2: obtained the optimum value for the TH3, the TH2 threshold is calculated counting the objects. As this
                # threshold looks at the contour channels, its purpose is to separed the entangled objects. This way, the TH2
                # optimum should be reached when the number of objects of the prediction match the number of real objects
                objs_to_divide = (pred[...,0] > th3_best).astype(np.uint8)
                th2_min = 0
                th2_last = 0
                th2_repeat_count = 0
                th2_op_pos = -1
                th2_obj_min_diff = sys.maxsize
                l_th2 = []
                for k in range(len(ths)):
                    p = (objs_to_divide * (pred[...,1] < ths[k])).astype(np.uint8)

                    p = label(np.squeeze(p), connectivity=1)
                    if len(np.unique(p)) != 1:
                        p = remove_small_objects(p, thres_small)
                    obj_count = len(np.unique(p))
                    l_th2.append(obj_count)

                    if abs(obj_count-len(labels)) < th2_obj_min_diff:
                        th2_obj_min_diff = abs(obj_count-len(labels))
                        th2_min = ths[k]
                        th2_op_pos = k
                        th2_repeat_count = 0

                    if th2_obj_min_diff == th2_last: th2_repeat_count += 1
                    th2_last = abs(obj_count-len(labels))

                g_l_th2.append(l_th2)
                l_th2_min.append(th2_min)
                th2_opt_pos = th2_op_pos + int(th2_repeat_count/2) if th2_repeat_count < 10 else th2_op_pos + 2
                if th2_opt_pos >= len(ths): th2_opt_pos = len(ths)-1
                l_th2_opt.append(ths[th2_opt_pos])

                # TH1 and TH4:
                th1_min = 0
                th1_last = 0
                th1_repeat_count = 0
                th1_op_pos = -1
                th1_obj_min_diff = sys.maxsize
                l_th1 = []
                in_row = False
                # TH1
                for k in range(len(ths)):
                    p = ((pred[...,0] > ths[k])*(pred[...,1] < th2_min)).astype(np.uint8)

                    p = label(np.squeeze(p), connectivity=1)
                    obj_count = len(np.unique(p))
                    l_th1.append(obj_count)

                    diff = abs(obj_count-len(labels))
                    if diff <= th1_obj_min_diff and th1_repeat_count < 4 and diff != th1_last:
                        th1_obj_min_diff = diff
                        th1_min = ths[k]
                        th1_op_pos = k
                        th1_repeat_count = 0
                        in_row = True

                    if diff == th1_last and diff == th1_obj_min_diff and in_row:
                        th1_repeat_count += 1
                    elif k != th1_op_pos:
                        in_row = False
                    th1_last = diff

                g_l_th1.append(l_th1)
                l_th1_min.append(th1_min)
                th1_opt_pos = th1_op_pos + th1_repeat_count
                if th1_opt_pos >= len(ths): th1_opt_pos = len(ths)-1
                l_th1_opt.append(ths[th1_opt_pos])

                # TH4
                if mode == 'BCD':
                    th4_min = 0
                    th4_last = 0
                    th4_repeat_count = 0
                    th4_op_pos = -1
                    th4_obj_min_diff = sys.maxsize
                    l_th4 = []
                    for k in range(len(ths_dis)):
                        p = ((pred[...,2] > ths_dis[k])*(pred[...,1] < th2_min)).astype(np.uint8)

                        p = label(np.squeeze(p), connectivity=1)
                        obj_count = len(np.unique(p))
                        l_th4.append(obj_count)

                        diff = abs(obj_count-len(labels))
                        if diff <= th4_obj_min_diff and th4_repeat_count < 4 and diff != th4_last:
                            th4_obj_min_diff = diff
                            th4_min = ths_dis[k]
                            th4_op_pos = k
                            th4_repeat_count = 0
                            in_row = True

                        if diff == th4_last and diff == th4_obj_min_diff and in_row:
                            th4_repeat_count += 1
                        elif k != th4_op_pos:
                            in_row = False
                        th4_last = diff

                    g_l_th4.append(l_th4)
                    l_th4_min.append(th4_min)
                    th4_opt_pos = th4_op_pos + th4_repeat_count
                    if th4_opt_pos >= len(ths_dis): th4_opt_pos = len(ths_dis)-1
                    l_th4_opt.append(ths_dis[th4_opt_pos])

            # Store the number of nucleus
            ideal_number_obj.append(len(labels))

    ideal_objects = statistics.mean(ideal_number_obj)
    create_th_plot(ths, g_l_th1, "TH1", chart_dir)
    create_th_plot(ths, g_l_th1, "TH1", chart_dir, per_sample=False, ideal_value=ideal_objects)
    create_th_plot(ths, g_l_th2, "TH2", chart_dir)
    create_th_plot(ths, g_l_th2, "TH2", chart_dir, per_sample=False, ideal_value=ideal_objects)
    create_th_plot(ths, g_l_th3, "TH3", chart_dir)
    create_th_plot(ths, g_l_th3, "TH3", chart_dir, per_sample=False)
    if mode == 'BCD':
        create_th_plot(ths_dis, g_l_th4, "TH4", chart_dir)
        create_th_plot(ths_dis, g_l_th4, "TH4", chart_dir, per_sample=False, ideal_value=ideal_objects)
        create_th_plot(ths_dis, g_l_th5, "TH5", chart_dir)
        create_th_plot(ths_dis, g_l_th5, "TH5", chart_dir, per_sample=False)

    if len(ideal_number_obj) > 1:
        global_th1 = statistics.mean(l_th1_min)
        global_th1_std = statistics.stdev(l_th1_min)
        global_th1_opt = statistics.mean(l_th1_opt)
        global_th1_opt_std = statistics.stdev(l_th1_opt)
        global_th2 = statistics.mean(l_th2_min)
        global_th2_std = statistics.stdev(l_th2_min)
        global_th2_opt = statistics.mean(l_th2_opt)
        global_th2_opt_std = statistics.stdev(l_th2_opt)
        global_th3 = statistics.mean(l_th3_max)
        global_th3_std = statistics.stdev(l_th3_max)
        if mode == 'BCD':
            global_th4 = statistics.mean(l_th4_min)
            global_th4_std = statistics.stdev(l_th4_min)
            global_th4_opt = statistics.mean(l_th4_opt)
            global_th4_opt_std = statistics.stdev(l_th4_opt)
            global_th5 = statistics.mean(l_th5_max)
            global_th5_std = statistics.stdev(l_th5_max)
    else:
        global_th1 = l_th1_min[0]
        global_th1_std = 0
        global_th1_opt = l_th1_opt[0]
        global_th1_opt_std = 0
        global_th2 = l_th2_min[0]
        global_th2_std = 0
        global_th2_opt = l_th2_opt[0]
        global_th2_opt_std = 0
        global_th3 = l_th3_max[0]
        global_th3_std = 0
        if mode == 'BCD':
            global_th4 = l_th4_min[0]
            global_th4_std = 0
            global_th4_opt = l_th4_opt[0]
            global_th4_opt_std = 0
            global_th5 = l_th5_max[0]
            global_th5_std = 0

    if verbose:
        if not use_minimum:
            print("MW_TH1 maximum value is {} (std:{}) so the optimum should be {} (std:{})".format(global_th1, global_th1_std, global_th1_opt, global_th1_opt_std))
        else:
            print("MW_TH1 minimum value is {}".format(min(l_th1_min)))
        print("MW_TH2 minimum value is {} (std:{}) and the optimum is {} (std:{})".format(global_th2, global_th2_std, global_th2_opt, global_th2_opt_std))
        print("MW_TH3 optimum should be {} (std:{})".format(global_th3, global_th3_std))
    if mode == 'BCD':
        if verbose:
            if not use_minimum:
                print("MW_TH4 maximum value is {} (std:{}) so the optimum should be {} (std:{})".format(global_th4, global_th4_std, global_th4_opt, global_th4_opt_std))
            else:
                print("MW_TH4 minimum value is {}".format(min(l_th4_min)))
            print("MW_TH5 optimum should be {} (std:{})".format(global_th5, global_th5_std))
        if not use_minimum:
            return global_th1_opt, global_th2_opt, global_th3, global_th4_opt, global_th5
        else:
            return min(l_th1_min), global_th2_opt, global_th3, min(l_th4_min), global_th5
    else:
        if not use_minimum:
            return global_th1_opt, global_th2_opt, global_th3
        else:
            return min(l_th1_min), global_th2_opt, global_th3


def create_th_plot(ths, y_list, th_name="TH1", chart_dir=None, per_sample=True, ideal_value=None):
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

    assert th_name in ['TH1', 'TH2', 'TH3', 'TH4', 'TH5']
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
    if th_name == 'TH3' or th_name == 'TH5':
        plt.ylabel("IoU")
    else:
        plt.ylabel("Number of objects")
    p = "_per_validation_sample" if per_sample else ""
    plt.savefig(os.path.join(chart_dir, str(th_name)+p+".svg"), format = 'svg', dpi=100)
    plt.show()


def voronoi_on_mask(data, mask, save_dir, filenames, th=0.3, thres_small=128, verbose=False):
    """Apply Voronoi to the voxels not labeled yet marked by the mask. Its done witk K-nearest neighbors.

       Parameters
       ----------
       data : 4D Numpy array
           Data to apply Voronoi. ``(num_of_images, z, y, x)`` e.g. ``(1, 397, 1450, 2000)``

       mask : 5D Numpy array
           Data mask to determine which points need to be proccessed. ``(num_of_images, z, y, x, channels)`` e.g.
           ``(1, 397, 1450, 2000, 3)``.

       save_dir :  str, optional
           Directory to save the resulting image.

       filenames : List, optional
           Filenames that should be used when saving each image.

       th : float, optional
           Threshold used to binarize the input.

       thres_small : int, optional
           Theshold to remove small objects created by the watershed.

       verbose : bool, optional
            To print saving information.

       Returns
       -------
       data : 4D Numpy array
           Image with Voronoi applied. ``(num_of_images, z, y, x)`` e.g. ``(1, 397, 1450, 2000)``

    """
    if data.ndim != 4:
        raise ValueError("Data must be 4 dimensional, provided {}".format(data.shape))
    if mask.ndim != 5:
        raise ValueError("Data mask must be 5 dimensional, provided {}".format(mask.shape))
    if mask.shape[-1] < 2:
        raise ValueError("Mask needs to have two channels at least, received {}".format(mask.shape[-1]))

    if verbose:
        print("Applying Voronoi . . .")

    os.makedirs(save_dir, exist_ok=True)
    if filenames is not None:
        if len(filenames) != len(data):
            raise ValueError("Filenames array and length of X have different shapes: {} vs {}".format(len(filenames),len(data)))

    _data = data.copy()
    d = len(str(len(_data)))
    for i in range(len(_data)):
        # Obtain centroids of labels
        idx = np.indices(_data[i].shape).reshape(_data[i].ndim, _data[i].size)
        labels, mean = npi.group_by(_data[i], axis=None).mean(idx, axis=1)
        points = mean.transpose((1,0))
        label_points = list(range(len(points)))

        # K-nearest neighbors
        tree = KDTree(points)

        # Create voxel mask
        voronoi_mask = (mask[i,...,0] > th).astype(np.uint8)
        voronoi_mask = label(voronoi_mask)
        voronoi_mask = (remove_small_objects(voronoi_mask, thres_small))>0
        # Remove small objects
        voronoi_mask = binary_dilation(voronoi_mask, iterations=2)
        voronoi_mask = binary_erosion(voronoi_mask, iterations=2)

        # XOR to determine the particular voxels to apply Voronoi
        not_labelled_points = ((_data[i] > 0) != voronoi_mask).astype(np.uint8)
        pos_not_labelled_points = np.argwhere(not_labelled_points>0)
        for j in range(len(pos_not_labelled_points)):
            z = pos_not_labelled_points[j][0]
            x = pos_not_labelled_points[j][1]
            y = pos_not_labelled_points[j][2]
            _data[i,z,x,y] = tree.query(pos_not_labelled_points[j])[1]

        # Save image
        if filenames is None:
            f = os.path.join(save_dir, str(i).zfill(d)+'.tif')
        else:
            f = os.path.join(save_dir, os.path.splitext(filenames[i])[0]+'.tif')
        aux = np.expand_dims(np.expand_dims(_data[i],-1),1).astype(np.float32)
        imsave(f, aux, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False)

    return _data


def voronoi_on_mask_2(data, mask, save_dir, filenames, th=0, verbose=False):
    """Apply Voronoi to the voxels not labeled yet marked by the mask. It is done using distances from the un-labeled
       voxels to the cell perimeters.

       Parameters
       ----------
       data : 4D Numpy array
           Data to apply Voronoi. ``(num_of_images, z, y, x)`` e.g. ``(1, 397, 1450, 2000)``

       mask : 5D Numpy array
           Data mask to determine which points need to be proccessed. ``(num_of_images, z, y, x, channels)`` e.g.
           ``(1, 397, 1450, 2000, 3)``.

       save_dir :  str, optional
           Directory to save the resulting image.

       filenames : List, optional
           Filenames that should be used when saving each image.

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

    if data.ndim != 4:
        raise ValueError("Data must be 4 dimensional, provided {}".format(data.shape))
    if mask.ndim != 5:
        raise ValueError("Data mask must be 5 dimensional, provided {}".format(mask.shape))
    if mask.shape[-1] < 2:
        raise ValueError("Mask needs to have two channels at least, received {}".format(mask.shape[-1]))

    if verbose:
        print("Applying Voronoi 3D . . .")

    os.makedirs(save_dir, exist_ok=True)
    if filenames is not None:
        if len(filenames) != len(data):
            raise ValueError("Filenames array and length of X have different shapes: {} vs {}".format(len(filenames),len(data)))

	# Extract mask from prediction
    if mask.shape[-1] == 3:
        mask = mask[...,2]
    else:
        mask = mask[...,0]+mask[...,1]

    mask_shape = np.shape(mask)
    mask = mask[0]
    data = data[0]

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
    labelPerId = np.zeros(np.size(idsToFill));

    idsPerim = np.argwhere(cellPerimeter==1)
    labelsPerimIds = voronoiCyst[cellPerimeter==1]

    # Generating voronoi
    for nId in range(1,len(idsToFill)):
        distCoord = cdist([idsToFill[nId]], idsPerim)
        idSeedMin = np.argwhere(distCoord==np.min(distCoord))
        idSeedMin = idSeedMin[0][1]
        labelPerId[nId] = labelsPerimIds[idSeedMin]
        voronoiCyst[idsToFill[nId][0], idsToFill[nId][1], idsToFill[nId][2]] = labelsPerimIds[idSeedMin]

    # Save image
    f = os.path.join(save_dir, filenames[0])
    voronoiCyst = np.reshape(voronoiCyst, (1, mask_shape[1], mask_shape[2], mask_shape[3]))
    aux = np.reshape(voronoiCyst, (mask_shape[1], 1, mask_shape[2], mask_shape[3], 1))
    imsave(f, aux.astype(np.uint16), imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False)

    return voronoiCyst


def remove_close_points(points, radius, resolution, ndim=3):
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
    print( 'Initial number of points: ' + str( len( points ) ) )

    point_list = points.copy()

    # Resolution adjust
    for i in range(len(point_list)):
        point_list[i][0] = point_list[i][0]* resolution[0]
        point_list[i][1] = point_list[i][1]* resolution[1]
        if ndim == 3:
            point_list[i][2] = point_list[i][2]* resolution[2]

    mynumbers = [tuple(point) for point in point_list] 

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
    new_point_list = [ points[i] for i in keep]

    print( 'Final number of points: ' + str( len( new_point_list ) ) )
    return new_point_list
    
    
def apply_binary_mask(X, bin_mask_dir):
    """Apply a binary mask to remove values outside it.

       Parameters
       ----------
       X : 4D Numpy array
           Data to apply the mask. E.g. ``(vol_number, y, x, channels)``

       bin_mask_dir : str, optional
           Directory where the binary mask are located.

       Returns
       -------
       X : 4D Numpy array
           Data with the mask applied. E.g. ``(vol_number, y, x, channels)``.
    """

    if X.ndim != 4:
        raise ValueError("'X' needs to have 4 dimensions and not {}".format(X.ndim))

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
        if mask.ndim != 2 and mask.ndim != 3:
            raise ValueError("Mask needs to have 2 or 3 dimensions and not {}".format(mask.ndim))

        for k in tqdm(range(X.shape[0])):
            if mask.ndim == 2:
                for c in range(X.shape[-1]):
                    X[k,:,:,c] = X[k,:,:,c]*(mask>0)
            else:
                X[k] = X[k]*(mask>0)
    else:
        for i in tqdm(range(len(ids))):
            mask = imread(os.path.join(bin_mask_dir, ids[i]))
            mask = np.squeeze(mask)
            if mask.ndim == 2:
                for c in range(X.shape[-1]):
                    X[i,:,:,c] = X[i,:,:,c]*(mask>0)
            else:
                X[i] = X[i]*(mask>0)

    return X
