import numpy as np
from skimage import measure
import cv2
import os
from tqdm import tqdm

def spuriuous_detection_filter(Y, low_score_th=0.65, th=0.5):
    """Based on the first post-processing method proposed in Oztel et al. where 
       removes the artifacts with low class score.
       
       Args:
            Y (4D Numpy array): data to apply the filter. 
            E.g. (img_number, x, y, channels). 
    
            low_score_th (float, optional): the minimun class score that the 
            artifact must have to not be discarded. Must be a vaue between 0 
            and 1. 
            
            th (float, optional): threshold applied to binarize the given images.
            Must be a vaue between 0 and 1.

       Return:
            class_Y (4D Numpy array): filtered data.
            E.g. (img_number, x, y, channels). 
    """
        
    if low_score_th < 0 or low_score_th > 1:
        raise ValueError("'low_score_th' must be a float between 0 and 1")
    if th < 0 or th > 1:
        raise ValueError("'th' must be a float between 0 and 1")

    class_Y = np.zeros(Y.shape)
    class_Y[Y[...,1]>th] = 1 
    
    for i in range(class_Y.shape[0]):
        im = class_Y[i,...,0]
        im, num = measure.label(im, connectivity=2, background=0, return_num=True)
    
        for j in range(num):
            c_conf = np.mean(Y[i,...,1][im==j])
            if c_conf < low_score_th:
                print("Slice {}: removing artifact {}".format(i, j))
                class_Y[i,...,0][im==j] = 0

    return class_Y


def boundary_refinement_watershed(X, Y_pred, erode=True, save_marks_dir=None):
    """Apply watershed to the given predictions with the goal of refine the 
       boundaries of the artifacts.

       Args:
            X (4D Numpy array): original data to guide the watershed.
            E.g. (img_number, x, y, channels).

            Y_pred (4D Numpy array): predicted data to refine the boundaries.
            E.g. (img_number, x, y, channels).

            erode (bool, optional): flag to extract the sure foreground eroding 
            the artifacts instead of doing with distanceTransform.  

            save_marks_dir (str, optional): directory to save the markers used 
            to make the watershed. Useful for debugging. 

        Return:
            watershed_predictions (4D Numpy array): refined boundaries of the 
            predictions.  E.g. (img_number, x, y, channels).
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
            sure_fg = cv2.erode(pred, kernel, iterations=5)
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
