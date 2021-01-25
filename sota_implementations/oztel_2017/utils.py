import os
import numpy as np
from tensorflow.keras import backend as K                                       
from skimage import transform
from skimage import measure
import cv2
from skimage import segmentation, filters
from skimage.morphology import disk
from skimage import feature
from scipy import ndimage
from skimage import util
from scipy.ndimage.filters import median_filter


def create_oztel_patches(imgs, lbls):
    """Create a list of ``32x32`` images patches out of a list of images for each
    class (at least ``80%`` of the pixels have to be of each class).

    Parameters
    ----------
    imgs: 4D Numpy array
        List of input images.

    lbls: 4D Numpy array
        List of label images.
        
    Returns 
    -------
    class0_patches: List
        List of class ``0`` image patches.

    class1_patches: List
        List of class ``1`` image patches.
    """

    original_size = imgs[0].shape
    num_y_patches = original_size[ 0 ] // 32
    num_x_patches = original_size[ 1 ] // 32
    patch_width = 32
    patch_height = 32
    thres0 = patch_width * patch_height * 0.2
    thres1 = patch_width * patch_height * 0.8
    
    class0_patches = []
    class1_patches = []
    for n in range( 0, len( imgs ) ):
        image = imgs[ n ]
        label = lbls[ n ]
        for i in range( 0, num_y_patches ):
            for j in range( 0, num_x_patches ):
                patch = image[ i * patch_width : (i+1) * patch_width,
                                      j * patch_height : (j+1) * patch_height ]
                lbl_patch = label[ i * patch_width : (i+1) * patch_width,
                                      j * patch_height : (j+1) * patch_height ]
                count = np.count_nonzero( lbl_patch )
                if count >= thres1:
                    class1_patches.append( patch )
                elif count <= thres0:
                    class0_patches.append( patch )
    return class0_patches, class1_patches


def ensemble8_2d_predictions(o_img, pred_func, batch_size_value=1, n_classes=2,
                             last_class=True):
    """Outputs the mean prediction of a given image generating its 8 possible 
       rotations and flips. This function is adapted to Oztel implementation as
       their predictions need to be resized. 

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
    
       last_class : bool, optional
           To preserve only the last class. Useful when a binary classification 
           and want to take only the foreground. 

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
        im = np.expand_dims(pred_func(np.expand_dims(aug_img[i], 0))[0,...,1], -1)
        im = transform.resize(im, [1024,1024,1], order=3)
        decoded_aug_img[i] = im

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


def spuriuous_detection_filter(Y, low_score_th=0.6, verbose=False):
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

       verbose : boolean, optional
           Flag to print deleted objects
            
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

    class_Y = np.argmax( Y, axis=-1)

    mito_prob = Y[:,:,:,1]

    for i in range(class_Y.shape[0]):
        im = class_Y[i]
        im, num = measure.label(im, connectivity=2, background=0, return_num=True)
   
        for j in range(1,num):
            c_conf = np.mean( mito_prob[i][ im[:]==j ] )
            if c_conf < low_score_th:
                if verbose:
                    print("Slice {}: removing artifact {} - pixels: {}"
                        .format(i, j, np.count_nonzero(mito_prob[i][im[:]==j])))
                class_Y[i][im[:]==j] = 0
    class_Y = np.expand_dims( class_Y, axis=-1 )
    return class_Y


def watershed_refinement(binary_input, gray_input, open_iter=3,
    dilate_iter=15, erode_iter=7, gauss_sigma=3, canny_sigma=3, 
    save_marks_dir=None):

    """ Apply watershed to given input. """

    if save_marks_dir is not None:
        os.makedirs(save_marks_dir, exist_ok=True)

    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    seg = np.copy( binary_input )

    for i in range( 0, len(binary_input) ):

        thresh = binary_input[i].astype(np.uint8)*255

        # noise removal
        if open_iter == 0:
            opening = thresh
        else:
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = open_iter)

        # sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=dilate_iter)

        # Finding sure foreground area
        sure_fg = cv2.erode(opening, kernel2, iterations=erode_iter)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)

        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers+1

        # Now, mark the region of unknown with zero
        markers[unknown==255] = 0
        img = (gray_input[i, :, :, 0]*255).astype(np.uint8)
        img = filters.gaussian( img, gauss_sigma ) if gauss_sigma >0 else img
        #img = filters.rank.median( img, disk( 3 ) )
        #img = filters.sobel( img )
        if canny_sigma > 0:
            img = feature.canny( img, sigma = canny_sigma,
                                low_threshold=0.01*255, high_threshold=0.2*255 )
            img = np.invert( img )
            img = ndimage.distance_transform_edt( img )
            img = util.invert( img )
        else:
            img = filters.sobel( img )
        seg[i][:,:,0] = segmentation.watershed(img, markers, compactness=0)
        seg[i] = (seg[i]>1).astype(np.float32)        

        if i==0 and save_marks_dir is not None:
            plt.figure(figsize=(20,15))
            plt.subplot(2, 2, 1)
            plt.title( "Original image")
            plt.imshow( gray_input[i, :,:,0], cmap='gray')
            plt.subplot(2, 2, 2)
            plt.title( "Watershed input" )
            plt.imshow( img, cmap='gray' );
            plt.subplot(2, 2, 3)
            plt.title( "Markers" )
            plt.imshow( markers, cmap='jet' );
            plt.subplot(2, 2, 4)
            plt.title( "Binary output" )          
            plt.imshow( seg[i][:,:,0] )
            f = os.path.join(save_marks_dir, "watershed.png")
            plt.savefig(os.path.join(f))

    return seg

def improve_components(test_pred, depth=9):
    """ Apply Z-Filtering to the given data. """
    return median_filter(test_pred, size=(depth,1,1,1)) #run z-smoothing (median filter)
