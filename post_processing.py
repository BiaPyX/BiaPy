import numpy as np
from skimage import measure
import cv2
import os
from tqdm import tqdm
from scipy import ndimage as ndi
from skimage.morphology import disk
from skimage.segmentation import watershed
from skimage.filters import rank
from scipy.ndimage import rotate


def spuriuous_detection_filter(Y, low_score_th=0.6, th=0.45):
    """Based on the first post-processing method proposed in `Oztel et al.` where 
       removes the artifacts with low class score.

       Based on post-processing made in `Oztel et al <https://ieeexplore.ieee.org/abstract/document/8217827?casa_token=ga4A1VzvnykAAAAA:U7iBJ2H0qD4MU4Z1JzdzobNx_vCxwM89fzy39IwAPT8TsRuESFu_rFzKKHspp6-EKTGRoOHh2g>`_.
       
       Parameters
       ----------
       Y : 4D Numpy array
           Data to apply the filter. E.g. ``(img_number, x, y, channels)``. 
    
       low_score_th : float, optional
           Minimun class score that the artifact must have to not be discarded. 
           Must be a value between ``0`` and ``1``. 
            
       th : float, optional
           Threshold applied to binarize the given images. Must be a vaue 
           between ``0`` and ``1``.

       Returns
       -------
       Array : 4D Numpy array
           Filtered data. E.g. ``(img_number, x, y, channels)``. 

       Raises
       ------
       ValueError
           If ``low_score_th`` not in ``[0, 1]``

       ValueError
           If ``th`` not in ``[0, 1]``
    """
        
    if low_score_th < 0 or low_score_th > 1:
        raise ValueError("'low_score_th' must be a float between 0 and 1")
    if th < 0 or th > 1:
        raise ValueError("'th' must be a float between 0 and 1")

    class_Y = np.zeros(Y.shape[:3], dtype=np.uint8)
    class_Y[Y[...,0]>th] = 1 
    
    for i in range(class_Y.shape[0]):
        im = class_Y[i]
        im, num = measure.label(im, connectivity=2, background=0, return_num=True)
    
        for j in range(1,num):
            c_conf = np.mean(Y[i,...,0][im==j])
            if c_conf < low_score_th:
                print("Slice {}: removing artifact {} - pixels: {}"
                      .format(i, j, np.count_nonzero(Y[i,...,0][im==j])))
                class_Y[i][im==j] = 0

    return np.expand_dims(class_Y, -1)


def boundary_refinement_watershed(X, Y_pred, erode=True, save_marks_dir=None):
    """Apply watershed to the given predictions with the goal of refine the 
       boundaries of the artifacts.

       Based on https://docs.opencv.org/master/d3/db4/tutorial_py_watershed.html.

       Parameters
       ----------
       X : 4D Numpy array
           Original data to guide the watershed. E.g. ``(img_number, x, y, channels)``.

       Y_pred : 4D Numpy array
           Predicted data to refine the boundaries. E.g. ``(img_number, x, y, channels)``.

       erode : bool, optional
           To extract the sure foreground eroding the artifacts instead of doing 
           with distanceTransform.  

       save_marks_dir : str, optional
           Directory to save the markers used to make the watershed. Useful for 
           debugging. 

       Returns
       -------
       Array : 4D Numpy array
           Refined boundaries of the predictions. E.g. ``(img_number, x, y, channels)``.
        
       Examples
       --------
        
       +-----------------------------------------+-----------------------------------------+
       | .. figure:: img/FIBSEM_test_0.png       | .. figure:: img/FIBSEM_test_0_gt.png    |
       |   :width: 80%                           |   :width: 80%                           |
       |   :align: center                        |   :align: center                        |
       |                                         |                                         |
       |   Original image                        |   Ground truth                          |
       +-----------------------------------------+-----------------------------------------+
       | .. figure:: img/FIBSEM_test_0_pred.png  | .. figure:: img/FIBSEM_test_0_wa.png    |
       |   :width: 80%                           |   :width: 80%                           |
       |   :align: center                        |   :align: center                        |
       |                                         |                                         |
       |   Predicted image                       |   Watershed ouput                       |
       +-----------------------------------------+-----------------------------------------+

       The marks used to guide the watershed is this example are these:

        .. image:: img/watershed2_marks_test0.png
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
            ret, sure_fg = cv2.threshold(
                dist_transform, 0.7*dist_transform.max(), 255,0)
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
    """Apply watershed to the given predictions with the goal of refine the 
       boundaries of the artifacts. This function was implemented using scikit
       instead of opencv as :meth:`post_processing.boundary_refinement_watershed`.

       Based on https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html. 

       Parameters
       ----------
       X : 4D Numpy array
           Original data to guide the watershed. E.g. ``(img_number, x, y, channels)``.

       Y_pred : 4D Numpy array
           Predicted data to refine the boundaries. E.g. ``(img_number, x, y, channels)``.

       save_marks_dir : str, optional
           Directory to save the markers used to make the watershed. Useful for 
           debugging. 

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
    out_data = np.squeeze(out_data)

    # Must be odd
    if mf_size % 2 == 0:
       mf_size += 1

    for i in range(data.shape[2]):
        sl = (data[:, :, i, 0]).astype(np.float32)
        sl = cv2.medianBlur(sl, mf_size)
        out_data[:, :, i] = sl

    return np.expand_dims(out_data, axis=-1)

def ensemble8_2d_predictions(o_img, pred_func, batch_size_value=1, n_classes=2):
    """Outputs the mean prediction of a given image generating its 8 possible 
       rotations and flips.

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
                   pred_func=(lambda img_batch_subdiv: model.predict(img_batch_subdiv)),   
                   n_classes=n_classes)                                                    
               out_X_test[i] = pred_ensembled   
                                                                                
           # Notice that here pred_func is created based on model.predict function
           # of Keras 
    """

    aug_img = []
        
    # Convert into square image to make the rotations properly
    pad_to_square = o_img.shape[0] - o_img.shape[1]
   
    if pad_to_square < 0:
        img = np.pad(o_img, [(abs(pad_to_square), 0), (0, 0), (0, 0)], 'reflect') 
    else:
        img = np.pad(o_img, [(0, 0), (pad_to_square, 0), (0, 0)], 'reflect')
    
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
    decoded_aug_img = np.zeros(aug_img.shape)
    
    for i in range(aug_img.shape[0]):
        if n_classes > 1:
            decoded_aug_img[i] = np.expand_dims(pred_func(np.expand_dims(aug_img[i], 0))[...,1], -1)
        else:
            decoded_aug_img[i] = pred_func(np.expand_dims(aug_img[i], 0))

    # Undo the combinations of the img
    out_img = []
    out_img.append(decoded_aug_img[0])
    out_img.append(np.rot90(decoded_aug_img[1], axes=(0, 1), k=3))
    out_img.append(np.rot90(decoded_aug_img[2], axes=(0, 1), k=2))
    out_img.append(np.rot90(decoded_aug_img[3], axes=(0, 1), k=1))
    out_img.append(decoded_aug_img[4][:, ::-1])
    out_img.append(np.rot90(decoded_aug_img[5], axes=(0, 1), k=3)[:, ::-1])
    out_img.append(np.rot90(decoded_aug_img[6], axes=(0, 1), k=2)[:, ::-1])
    out_img.append(np.rot90(decoded_aug_img[7], axes=(0, 1), k=1)[:, ::-1])

    # Create the output data
    out_img = np.array(out_img) 
    if pad_to_square != 0:
        if pad_to_square < 0:
            out = np.zeros((out_img.shape[0], img.shape[0]+pad_to_square, 
                            img.shape[1], img.shape[2]))
        else:
            out = np.zeros((out_img.shape[0], img.shape[0], 
                            img.shape[1]-pad_to_square, img.shape[2]))
    else:
        out = np.zeros(out_img.shape)

    # Undo the padding
    for i in range(out_img.shape[0]):
        if pad_to_square < 0:
            out[i] = out_img[i,abs(pad_to_square):,:]
        else:
            out[i] = out_img[i,:,abs(pad_to_square):]

    return np.mean(out, axis=0)


def ensemble16_3d_predictions(vol, pred_func, batch_size_value=1, n_classes=2):
    """Outputs the mean prediction of a given image generating its 16 possible   
       rotations and flips.                                                     
                                                                                
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
                   pred_func=(lambda img_batch_subdiv: model.predict(img_batch_subdiv)),   
                   n_classes=n_classes)                                                    
               out_X_test[i] = pred_ensembled          
                        
           # Notice that here pred_func is created based on model.predict function
           # of Keras
    """  

    aug_vols = []
        
    # Convert into square image to make the rotations properly
    pad_to_square = vol.shape[0] - vol.shape[1]
   
    if pad_to_square < 0:
        volume = np.pad(vol, [(abs(pad_to_square),0), (0,0), (0,0), (0,0)], 'reflect') 
    else:
        volume = np.pad(vol, [(0,0), (pad_to_square,0), (0,0), (0,0)], 'reflect')
    
    # Remove the last channel to make the transformations correctly             
    volume = volume[...,0]

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
    del volume_aux
    aug_vols = np.array(aug_vols)
    
    # Add the last channel again
    aug_vols = np.expand_dims(aug_vols, -1)
    volume = np.expand_dims(volume, -1)

    decoded_aug_vols = np.zeros(aug_vols.shape)
    for i in range(aug_vols.shape[0]):
        if n_classes > 1:
            decoded_aug_vols[i] = np.expand_dims(pred_func(np.expand_dims(aug_vols[i], 0))[...,1], -1)
        else:
            decoded_aug_vols[i] = pred_func(np.expand_dims(aug_vols[i], 0))

    # Remove the last channel to make the transformations correctly
    decoded_aug_vols = decoded_aug_vols[...,0]

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

    # Add the last channel again
    decoded_aug_vols = np.expand_dims(decoded_aug_vols, -1)
    out_vols = np.expand_dims(out_vols, -1)
    
    # Create the output data
    if pad_to_square != 0:
        if pad_to_square < 0:
            out = np.zeros((out_vols.shape[0], volume.shape[0]+pad_to_square, 
                            volume.shape[1], volume.shape[2], volume.shape[3]))
        else:
            out = np.zeros((out_vols.shape[0], volume.shape[0], 
                            volume.shape[1]-pad_to_square, volume.shape[2], 
                            volume.shape[3]))
    else:
        out = np.zeros(out_vols.shape)

    # Undo the padding
    for i in range(out_vols.shape[0]):
        if pad_to_square < 0:
            out[i] = out_vols[i,abs(pad_to_square):,:,:,:]
        else:
            out[i] = out_vols[i,:,abs(pad_to_square):,:,:]

    return np.mean(out, axis=0)
