import cv2
import os
import math
import statistics
import sys
import numpy as np
from skimage import measure
from tqdm import tqdm
from scipy import ndimage as ndi
from skimage.morphology import disk, remove_small_objects
from skimage.segmentation import watershed
from skimage.filters import rank
from scipy.ndimage import rotate
from skimage.measure import label
from skimage.io import imsave, imread
from scipy.ndimage.morphology import binary_dilation

from engine.metrics import jaccard_index_numpy
from utils.util import save_tif, apply_binary_mask


def boundary_refinement_watershed(X, Y_pred, erode=True, save_marks_dir=None):
    """Apply watershed to the given predictions with the goal of refine the boundaries of the artifacts.

       Based on https://docs.opencv.org/master/d3/db4/tutorial_py_watershed.html.

       Parameters
       ----------
       X : 4D Numpy array
           Original data to guide the watershed. E.g. ``(img_number, x, y, channels)``.

       Y_pred : 4D Numpy array
           Predicted data to refine the boundaries. E.g. ``(img_number, x, y, channels)``.

       erode : bool, optional
           To extract the sure foreground eroding the artifacts instead of doing with distanceTransform.

       save_marks_dir : str, optional
           Directory to save the markers used to make the watershed. Useful for debugging.

       Returns
       -------
       Array : 4D Numpy array
           Refined boundaries of the predictions. E.g. ``(img_number, x, y, channels)``.

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
           Original data to guide the watershed. E.g. ``(img_number, x, y, channels)``.

       Y_pred : 4D Numpy array
           Predicted data to refine the boundaries. E.g. ``(img_number, x, y, channels)``.

       save_marks_dir : str, optional
           Directory to save the markers used to make the watershed. Useful for debugging.

       Returns
       -------
       Array : 4D Numpy array
           Refined boundaries of the predictions. E.g. ``(img_number, x, y, channels)``.
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
        seed_map = remove_small_objects(seed_map, thres_small)

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
        seed_map = remove_small_objects(seed_map, thres_small)

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


def calculate_z_filtering(data, mf_size=5):
    """Applies a median filtering in the z dimension of the provided data.

       Parameters
       ----------
       data : 4D Numpy array
           Data to apply the filter to. E.g. ``(num_of_images, x, y, channels)``.

       mf_size : int, optional
           Size of the median filter. Must be an odd number.

       Returns
       -------
       Array : 4D Numpy array
           Z filtered data. E.g. ``(num_of_images, x, y, channels)``.
    """

    out_data = np.copy(data)

    # Must be odd
    if mf_size % 2 == 0:
       mf_size += 1

    for i in range(data.shape[2]):
        sl = (data[:, :, i]).astype(np.float32)
        sl = cv2.medianBlur(sl, mf_size)
        sl = np.expand_dims(sl,-1) if sl.ndim == 2 else sl
        out_data[:, :, i] = sl

    return out_data


def ensemble8_2d_predictions(o_img, pred_func, batch_size_value=1, n_classes=1):
    """Outputs the mean prediction of a given image generating its 8 possible rotations and flips.

       Parameters
       ----------
       o_img : 3D Numpy array
           Input image. E.g. ``(x, y, channels)``.

       pred_func : function
           Function to make predictions.

       batch_size_value : int, optional
           Batch size value.

       n_classes : int, optional
           Number of classes.

       Returns
       -------
       out : 3D Numpy array
           Output image ensembled. E.g. ``(x, y, channels)``.

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
           Input image. E.g. ``(x, y, z, channels)``.

       pred_func : function
           Function to make predictions.

       batch_size_value : int, optional
           Batch size value.

       n_classes : int, optional
           Number of classes.

       Returns
       -------
       out : 4D Numpy array
           Output image ensembled. E.g. ``(x, y, z, channels)``.

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
        pad_to_square = _vol.shape[0] - _vol.shape[1]

        if pad_to_square < 0:
            volume = np.pad(_vol, [(abs(pad_to_square),0), (0,0), (0,0)], 'reflect')
        else:
            volume = np.pad(_vol, [(0,0), (pad_to_square,0), (0,0)], 'reflect')

        # Make 16 different combinations of the volume
        aug_vols.append(volume)
        aug_vols.append(rotate(volume, mode='reflect', axes=(0, 1), angle=90, reshape=False))
        aug_vols.append(rotate(volume, mode='reflect', axes=(0, 1), angle=180, reshape=False))
        aug_vols.append(rotate(volume, mode='reflect', axes=(0, 1), angle=270, reshape=False))
        volume_aux = np.flip(volume, 0)
        aug_vols.append(volume_aux)
        aug_vols.append(rotate(volume_aux, mode='reflect', axes=(0, 1), angle=90, reshape=False))
        aug_vols.append(rotate(volume_aux, mode='reflect', axes=(0, 1), angle=180, reshape=False))
        aug_vols.append(rotate(volume_aux, mode='reflect', axes=(0, 1), angle=270, reshape=False))
        volume_aux = np.flip(volume, 1)
        aug_vols.append(volume_aux)
        aug_vols.append(rotate(volume_aux, mode='reflect', axes=(0, 1), angle=90, reshape=False))
        aug_vols.append(rotate(volume_aux, mode='reflect', axes=(0, 1), angle=180, reshape=False))
        aug_vols.append(rotate(volume_aux, mode='reflect', axes=(0, 1), angle=270, reshape=False))
        volume_aux = np.flip(volume, 2)
        aug_vols.append(volume_aux)
        aug_vols.append(rotate(volume_aux, mode='reflect', axes=(0, 1), angle=90, reshape=False))
        aug_vols.append(rotate(volume_aux, mode='reflect', axes=(0, 1), angle=180, reshape=False))
        aug_vols.append(rotate(volume_aux, mode='reflect', axes=(0, 1), angle=270, reshape=False))
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
        out_vols.append(rotate(np.array(decoded_aug_vols[1]), mode='reflect', axes=(0, 1), angle=-90, reshape=False))
        out_vols.append(rotate(np.array(decoded_aug_vols[2]), mode='reflect', axes=(0, 1), angle=-180, reshape=False))
        out_vols.append(rotate(np.array(decoded_aug_vols[3]), mode='reflect', axes=(0, 1), angle=-270, reshape=False))
        out_vols.append(np.flip(np.array(decoded_aug_vols[4]), 0))
        out_vols.append(np.flip(rotate(np.array(decoded_aug_vols[5]), mode='reflect', axes=(0, 1), angle=-90, reshape=False), 0))
        out_vols.append(np.flip(rotate(np.array(decoded_aug_vols[6]), mode='reflect', axes=(0, 1), angle=-180, reshape=False), 0))
        out_vols.append(np.flip(rotate(np.array(decoded_aug_vols[7]), mode='reflect', axes=(0, 1), angle=-270, reshape=False), 0))
        out_vols.append(np.flip(np.array(decoded_aug_vols[8]), 1))
        out_vols.append(np.flip(rotate(np.array(decoded_aug_vols[9]), mode='reflect', axes=(0, 1), angle=-90, reshape=False), 1))
        out_vols.append(np.flip(rotate(np.array(decoded_aug_vols[10]), mode='reflect', axes=(0, 1), angle=-180, reshape=False), 1))
        out_vols.append(np.flip(rotate(np.array(decoded_aug_vols[11]), mode='reflect', axes=(0, 1), angle=-270, reshape=False), 1))
        out_vols.append(np.flip(np.array(decoded_aug_vols[12]), 2))
        out_vols.append(np.flip(rotate(np.array(decoded_aug_vols[13]), mode='reflect', axes=(0, 1), angle=-90, reshape=False), 2))
        out_vols.append(np.flip(rotate(np.array(decoded_aug_vols[14]), mode='reflect', axes=(0, 1), angle=-180, reshape=False), 2))
        out_vols.append(np.flip(rotate(np.array(decoded_aug_vols[15]), mode='reflect', axes=(0, 1), angle=-270, reshape=False), 2))
        out_vols = np.array(out_vols)
        out_vols = np.expand_dims(out_vols, -1)
        arr.append(out_vols)

    out_vols = np.concatenate(arr, -1)
    del decoded_aug_vols, _decoded_aug_vols, arr

    # Create the output data
    if pad_to_square != 0:
        if pad_to_square < 0:
            out = np.zeros((out_vols.shape[0], volume.shape[0]+pad_to_square, volume.shape[1], volume.shape[2], out_vols.shape[-1]))
        else:
            out = np.zeros((out_vols.shape[0], volume.shape[0], volume.shape[1]-pad_to_square, volume.shape[2], out_vols.shape[-1]))
    else:
        out = np.zeros(out_vols.shape)

    # Undo the padding
    for i in range(out_vols.shape[0]):
        if pad_to_square < 0:
            out[i] = out_vols[i,abs(pad_to_square):,:,:,:]
        else:
            out[i] = out_vols[i,:,abs(pad_to_square):,:,:]

    return np.mean(out, axis=0)


def calculate_optimal_mw_thresholds(model, data_path, data_mask_path, mode="BC", distance_mask_path=None, thres_small=5,
                                    bin_mask_path=None, verbose=True):
    """Calculate the optimum values for the marked controlled watershed thresholds.

       Parameters
       ----------
       model: Keras model
           Model to make the predictions.

       data_path : str
           Path to load the samples to infer.

       data_mask_path : str
           Path to load the mask samples.

       mode : str, optional
           Operation mode. Possible values: ``BC`` and ``BCD``.  ``BC`` corresponds to use binary segmentation+contour.
           ``BCD`` stands for binary segmentation+contour+distances.

       thres_small : int, optional
           Theshold to remove small objects in the mask and the prediction.

       bin_mask_path : str, optional
           Path of the binary masks to apply to the prediction. Useful to remove segmentation outside the masks.
           If ``None``, no mask is applied.

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
    ths.extend(np.arange(0.1, 1.0, 0.05))

    l_th1_min_opt = []
    l_th1_min = []
    l_th2 = []
    l_th2_opt = []
    l_th3 = []
    if mode == 'BCD':
        l_th4_min_opt = []
        l_th4_min = []
        l_th5 = []

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
        if verbose: print("Analizing file {}".format(ids[i]))

        # Read and prepare images
        if ids[i].endswith('.npy'):
            img = np.load(os.path.join(data_path, ids[i]))
        else:
            img = imread(os.path.join(data_path, ids[i]))
        img = np.squeeze(img)
        if mask_ids[i].endswith('.npy'):
            mask = np.load(os.path.join(data_mask_path, mask_ids[i]))
        else:
            mask = imread(os.path.join(data_mask_path, mask_ids[i]))
        mask = np.squeeze(mask)
        if len(img.shape) == 3:
            img = np.expand_dims(img, axis=-1)
        if len(mask.shape) == 3:
            mask = np.expand_dims(mask, axis=-1)
        img = np.transpose(img, (1,2,0,3))
        img = np.expand_dims(img, axis=0)
        mask = np.transpose(mask, (1,2,0,3))
        mask = np.expand_dims(mask, axis=0)
        mask = mask.astype(np.uint8)

        # Mask adjust
        labels = np.unique(mask)
        if len(labels) != 1:
            mask = remove_small_objects(mask, thres_small)

        if len(labels) == 1:
            if verbose:
                print("Skip this sample as it is just background")
        else:
            # Prediction
            if np.max(img) > 100: img = img/255
            pred = model.predict(img, verbose=0)

            if bin_mask_path is not None:
                pred = apply_binary_mask(pred, bin_mask_path)

            for k in tqdm(range(1,len(labels))):
                obj = (mask==labels[k]).astype(np.uint8)
                obj_dil = binary_dilation(obj, iterations=3)

                # TH1 and TH4:
                # Check when all pixels belong to the object dissapear moving the threshold (TH). When find that TH the best
                # value of TH1 is considered the previous (j+1 because is a reversed for)
                th1_min = -1
                # TH1
                for j in range(len(ths)):
                    p = (np.expand_dims(pred[...,0] > ths[j],-1)*(obj_dil > 0)).astype(np.uint8)
                    jac = jaccard_index_numpy(obj, p)
                    if jac < 0.1:
                        th1_min = ths[j-1] if j > 0 else ths[0]
                        break
                if th1_min == -1: th1_min = 1
                th1_min_opt = th1_min-(th1_min*0.2)
                l_th1_min_opt.append(th1_min_opt)
                l_th1_min.append(th1_min)

                # TH4
                if mode == 'BCD':
                    th4_min = -1
                    for j in range(len(ths_dis)):
                        p = (np.expand_dims(pred[...,2] > ths_dis[j],-1)*(obj_dil > 0)).astype(np.uint8)
                        jac = jaccard_index_numpy(obj, p)
                        if jac < 0.1:
                            th4_min = ths_dis[j-1] if j > 0 else ths_dis[0]
                            break
                    if th4_min == -1: th4_min = ths_dis[0]
                    th4_min_opt = th4_min-(th4_min*0.2)
                    l_th4_min_opt.append(th4_min_opt)
                    l_th4_min.append(th4_min)

                # TH3 and TH5:
                # Look at the best IoU compared with the original label. Only the region that involve the object is taken
                # into consideration. This is achieved dilating 2 iterations the original object mask. If we do not dilate
                # that label, decreasing the TH will always ensure a IoU >= than the previous TH. This way, the IoU will
                # reach a maximum and then it will start to decrease, as more points that are not from the object are added
                # into it
                # TH3
                th3_best = -1
                th3_max_jac = -1
                for j in reversed(range(len(ths))):
                    # TH3
                    p = (np.expand_dims(pred[...,0] > ths[j],-1)*(obj_dil > 0)).astype(np.uint8)
                    jac = jaccard_index_numpy(obj, p)
                    if jac > th3_max_jac:
                        th3_max_jac = jac
                        th3_best = ths[j]
                l_th3.append(th3_best)
                # TH5
                if mode == 'BCD':
                    th5_best = -1
                    th5_max_jac = -1
                    for j in reversed(range(len(ths_dis))):
                        # TH5
                        p = (np.expand_dims(pred[...,2] > ths_dis[j],-1)*(obj_dil > 0)).astype(np.uint8)
                        jac = jaccard_index_numpy(obj, p)
                        if jac > th5_max_jac:
                            th5_max_jac = jac
                            th5_best = ths_dis[j]
                    l_th5.append(th5_best)
            del obj, obj_dil

            # TH2: obtained the optimum value for the TH3, the TH2 threshold is calculated counting the objects. As this
            # threshold looks at the contour channels, its purpose is to separed the entangled objects. This way, the TH2
            # optimum should be reached when the number of objects of the prediction match the number of real objects
            best_th3 = statistics.mean(l_th3)
            objs_to_divide = (pred[...,0] > best_th3).astype(np.uint8)
            objs_to_divide = binary_dilation(objs_to_divide, iterations=3)

            th2_min = 0
            th2_last = 0
            th2_repeat_count = 0
            th2_op_pos = -1
            th2_obj_min_diff = sys.maxsize
            for k in range(len(ths)):
                p = (objs_to_divide * (pred[...,1] < ths[k])).astype(np.uint8)
                p = label(np.squeeze(p), connectivity=1)
                if len(np.unique(p)) != 1:
                    p = remove_small_objects(p, thres_small)
                obj_count = len(np.unique(p))
                if abs(obj_count-len(labels)) < th2_obj_min_diff:
                    th2_obj_min_diff = abs(obj_count-len(labels))
                    th2_min = ths[k]
                    th2_op_pos = k
                    th2_repeat_count = 0

                if th2_obj_min_diff == th2_last: th2_repeat_count += 1
                th2_last = abs(obj_count-len(labels))

            l_th2.append(th2_min)
            th2_opt_pos = th2_op_pos + int(th2_repeat_count/2)
            if th2_opt_pos >= len(ths): th2_opt_pos = len(ths)-1
            l_th2_opt.append(ths[th2_opt_pos])

    global_th1_min_opt = statistics.mean(l_th1_min_opt)
    global_th1_min_opt_std = statistics.stdev(l_th1_min_opt)
    global_th1_min = statistics.mean(l_th1_min)
    global_th1_min_std = statistics.stdev(l_th1_min)
    if len(l_th2) != 1:
        global_th2 = statistics.mean(l_th2)
        global_th2_std = statistics.stdev(l_th2)
        global_th2_opt = statistics.mean(l_th2_opt)
        global_th2_opt_std = statistics.stdev(l_th2_opt)
    else:
        global_th2 = l_th2[0]
        global_th2_std = 0
        global_th2_opt = l_th2_opt[0]
        global_th2_opt_std = 0
    global_th3 = statistics.mean(l_th3)
    global_th3_std = statistics.stdev(l_th3)
    if mode == 'BCD':
        global_th4_min_opt = statistics.mean(l_th4_min_opt)
        global_th4_min_opt_std = statistics.stdev(l_th4_min_opt)
        global_th4_min = statistics.mean(l_th4_min)
        global_th4_min_std = statistics.stdev(l_th4_min)
        global_th5 = statistics.mean(l_th5)
        global_th5_std = statistics.stdev(l_th5)

    if verbose:
        print("MW_TH1 maximum value is {} (std:{}) so the optimum should be {} (std:{})".format(global_th1_min, global_th1_min_std, global_th1_min_opt, global_th1_min_opt_std))
        print("MW_TH2 minimum value is {} (std:{}) and the optimum is {} (std:{})".format(global_th2, global_th2_std, global_th2_opt, global_th2_opt_std))
        print("MW_TH3 optimum should be {} (std:{})".format(global_th3, global_th3_std))
    if mode == 'BCD':
        if verbose:
            print("MW_TH4 maximum value is {} (std:{}) so the optimum should be {} (std:{})".format(global_th4_min, global_th4_min_std, global_th4_min_opt, global_th4_min_opt_std))
            print("MW_TH5 optimum should be {} (std:{})".format(global_th5, global_th5_std))
        return global_th1_min_opt, global_th2_opt, global_th3, global_th4_min_opt, global_th5
    else:
        return global_th1_min_opt, global_th2_opt, global_th3
