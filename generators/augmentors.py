import numpy as np
from skimage.transform import resize
import cv2
import random                                                                   
import math


def cutout(img, mask, nb_iterations=(1,3), size=(0.2,0.4), cval=0,
           apply_to_mask=False): 
    """Cutout data augmentation.
       Original `paper <https://arxiv.org/pdf/1708.04552.pdf>`_.
    
       Parameters
       ----------
       img : 3D Numpy array
           Image to transform. E.g. ``(x, y, channels)``.

       mask1 : 3D Numpy array                                                   
           Mask to transform. E.g. ``(x, y, channels)``.   
        
       nb_iterations : tuple of ints, optional
           Number of areas to fill the image with. E.g. ``(1, 3)``.
    
       size : tuple of floats, optional
           Range to choose the size of the areas to create.

       cval : int, optional
           Value to fill the area with.
    
       apply_to_mask : boolean, optional
           To apply cutout to the mask.
        
       Returns
       -------
       out : 3D Numpy array
           Transformed image. E.g. ``(x, y, channels)``.

       mask : 3D Numpy array
           Transformed mask. E.g. ``(x, y, channels)``.
    """

    it = np.random.randint(nb_iterations[0], nb_iterations[1])

    out = img.copy()
    m_out = mask.copy()
    for i in range(it):
        _size = random.uniform(size[0], size[1])
        y_size = int(img.shape[0]*_size)
        x_size = int(img.shape[1]*_size)

        # Choose a random point
        cy = np.random.randint(0, img.shape[0]-(y_size))
        cx = np.random.randint(0, img.shape[1]-(x_size))

        # Apply cutout to all channels
        for i in range(img.shape[-1]):
            out[cy:cy+y_size, cx:cx+x_size, i] = cval
        if apply_to_mask:
            for i in range(mask.shape[-1]):
                m_out[cy:cy+y_size, cx:cx+x_size, i] = 0
        
    return out, m_out


def cutblur(img, size=(0.2,0.4), down_ratio_range=(2,8), only_inside=True):
    """CutBlur data augmentation adapted from https://github.com/clovaai/cutblur 
       Original `paper <https://arxiv.org/pdf/2004.00448.pdf>`_.
    
       Parameters
       ----------
       img : 3D Numpy array
           Image to transform. E.g. ``(x, y, channels)``.
        
       size : float, optional
           Size of the region to transform.
    
       down_ratio_range : tuple of ints, optional
           Downsampling ratio range to be applied. E. g. ``(2, 8)``. 
            
       only_inside : bool, optional
           If ``True`` only the region inside will be modified (cut LR into HR
           image). If ``False`` the ``50%`` of the times the region inside will
           be modified (cut LR into HR image) and the other ``50%`` the inverse
           will be done (cut HR into LR image). See Figure 1 of the official
           `paper <https://arxiv.org/pdf/2004.00448.pdf>`_. 
    
       Returns
       -------
       out : 3D Numpy array
           Transformed image. E.g. ``(x, y, channels)``.
    """
    _size = random.uniform(size[0], size[1])

    y_size = int(img.shape[0]*_size)
    x_size = int(img.shape[1]*_size)

    # Choose a random point
    cy = np.random.randint(0, img.shape[0]-(y_size))
    cx = np.random.randint(0, img.shape[1]-(x_size))

    # Choose a random downsampling ratio
    down_ratio = np.random.randint(down_ratio_range[0], down_ratio_range[1]+1)
    out_shape = np.array(img.shape[:2])//down_ratio

    if not only_inside:
        prob = random.uniform(0, 1)
        inside = True if prob < 0.5 else False
    else:   
        inside = True

    out = img.copy()

    # Apply cutblur to all channels
    for i in range(img.shape[-1]):
        if inside:
            temp = img[cy:cy+y_size, cx:cx+x_size, i].copy()
        else:
            temp = img[..., i].copy()

        downsampled = resize(temp, out_shape, order=1, mode='reflect',
                             clip=True, preserve_range=True, anti_aliasing=True)
        upsampled = resize(downsampled, temp.shape, order=0, mode='reflect',
                           clip=True, preserve_range=True, anti_aliasing=False)
        if inside:
            out[cy:cy+y_size, cx:cx+x_size, i] = upsampled
        else:
            temp[cy:cy+y_size, cx:cx+x_size] = img[cy:cy+y_size, cx:cx+x_size, i]
            out[...,i] = temp
        
    return out


def cutmix(im1, im2, mask1, mask2, size=(0.2,0.4)):
    """Cutmix operation where a region of the image sample is filled with a 
       given second image. This implementation is used for semantic segmentation
       so the mask of the images are also needed. It assumes that the images are
       of the same shape. 

       Parameters
       ----------
       im1 : 3D Numpy array
           Image to transform. E.g. ``(x, y, channels)``.

       im2 : 3D Numpy array
           Image to paste into the region of ``im1``. E.g. ``(x, y, channels)``.

       mask1 : 3D Numpy array
           Mask to transform (belongs to ``im1``). E.g. ``(x, y, channels)``.

       mask2 : 3D Numpy array
           Mask to paste into the region of ``mask1``. E.g. ``(x, y, channels)``.

       size : tuple of floats, optional
           Range to choose the size of the areas to transform. E. g. ``(0.2, 0.4)``.

       Returns
       -------
       out : 3D Numpy array
           Transformed image. E.g. ``(x, y, channels)``.

       m_out : 3D Numpy array
           Transformed mask. E.g. ``(x, y, channels)``.
    """
    _size = random.uniform(size[0], size[1])

    y_size = int(im1.shape[0]*_size)
    x_size = int(im1.shape[1]*_size)

    # Choose a random point from the image to be transformed
    im1cy = np.random.randint(0, im1.shape[0]-(y_size))
    im1cx = np.random.randint(0, im1.shape[1]-(x_size))
    # Choose a random point from the other image
    im2cy = np.random.randint(0, im2.shape[0]-(y_size))
    im2cx = np.random.randint(0, im2.shape[1]-(x_size))

    out = im1.copy()
    m_out = mask1.copy()

    # Apply cutblur to all channels
    for i in range(im1.shape[-1]):
        out[im1cy:im1cy+y_size, 
            im1cx:im1cx+x_size, i] = im2[im2cy:im2cy+y_size, im2cx:im2cx+x_size, i]
        m_out[im1cy:im1cy+y_size,
              im1cx:im1cx+x_size, i] = mask2[im2cy:im2cy+y_size, im2cx:im2cx+x_size, i]

    return out, m_out


def misalignment(img, mask, displacement=16, rotate_ratio=0.0):
    """Mis-alignment data augmentation of image stacks. This augmentation is 
       applied to both images and masks.
    
       Implementation based on https://github.com/zudi-lin/pytorch_connectomics/blob/master/connectomics/data/augmentation/misalign.py .

       Parameters
       ----------
       img : 3D Numpy array                                                     
           Image to transform. E.g. ``(x, y, channels)``.                       
                                                                                
       mask1 : 3D Numpy array                                                   
           Mask to transform. E.g. ``(x, y, channels)``

       displacement : int, optional
           Maximum pixel displacement in ``xy``-plane.

       rotate_ratio : float, optional
           Ratio of rotation-based mis-alignment. 

       Returns                                                                  
       -------                                                                  
       out : 3D Numpy array                                                     
           Transformed image. E.g. ``(x, y, channels)``.                        
                                                                                
       m_out : 3D Numpy array                                                   
           Transformed mask. E.g. ``(x, y, channels)``.  
    
    """
    out = np.zeros(img.shape, img.dtype)  
    m_out = np.zeros(mask.shape, mask.dtype)  

    # 2D
    if img.shape[-1] < 5:
        oy = np.random.randint(1, img.shape[0]-1)                           
        d = np.random.randint(0, displacement)                              
        prob = random.uniform(0, 1)                                             
        if prob < rotate_ratio:
            # Apply misalignment to all channels                                             
            for i in range(img.shape[-1]):                                      
                out[:oy,:,i] = img[:oy,:,i]                                     
                out[oy:,:img.shape[1]-d,i] = img[oy:,d:,i]    
                m_out[:oy,:,i] = mask[:oy,:,i]                                     
                m_out[oy:,:mask.shape[1]-d,i] = mask[oy:,d:,i]    
        else:   
            H, W = img.shape[:2]
            M = random_rotate_matrix(H, displacement)
            H = H - oy
            # Apply misalignment to all channels                                             
            for i in range(img.shape[-1]):                                      
                out[:oy,:,i] = img[:oy,:,i]                                     
                out[oy:,:,i] = cv2.warpAffine(img[oy:,:,i], M, (W,H), 1.0,      
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)                      
                m_out[:oy,:,i] = mask[:oy,:,i]                                  
                m_out[oy:,:,i] = cv2.warpAffine(mask[oy:,:,i], M, (W,H), 1.0,      
                    flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)                          
    # 3D
    else:
        out_shape = (img.shape[0]-displacement, img.shape[1]-displacement,  
                     img.shape[2])
        prob = random.uniform(0, 1)                                         
        mode = 'slip' if prob < 0.5 else 'translation'
        idx = np.random.randint(1, img.shape[-1]-1)

        prob = random.uniform(0, 1)                                                 
        if prob < rotate_ratio:
            H, W = img.shape[:2]
            M = random_rotate_matrix(H, displacement)
            if mode == 'slip':
                out[idx] = cv2.warpAffine(img[idx], M, (H,W), 1.0, 
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) 
                m_out[idx] = cv2.warpAffine(mask[idx], M, (H,W), 1.0, 
                    flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) 
            else:
                for i in range(idx, img.shape[-1]):   
                    out[...,i] = cv2.warpAffine(img[...,i], M, (H,W), 1.0, 
                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                for i in range(idx, mask.shape[-1]):   
                    m_out[...,i] = cv2.warpAffine(mask[...,i], M, (H,W), 1.0, 
                        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)   
        else:
            x0 = random_state.randint(displacement)
            y0 = random_state.randint(displacement)
            x1 = random_state.randint(displacement)
            y1 = random_state.randint(displacement)
            if mode == 'slip':
                out = img[y0:y0+out_shape[0], x0:x0+out_shape[1],:]
                out[idx] = img[idx, y1:y1+out_shape[0], x1:x1+out_shape[1], idx]
                m_out = mask[y0:y0+out_shape[0], x0:x0+out_shape[1], :]
                m_out[idx] = mask[y1:y1+out_shape[0], x1:x1+out_shape[1], idx]
            else:
                out[:idx] = img[y0:y0+out_shape[0], x0:x0+out_shape[1], :idx]
                out[idx:] = img[y1:y1+out_shape[0], x1:x1+out_shape[1], idx:]
                m_out[:idx] = mask[y0:y0+out_shape[0], x0:x0+out_shape[1], :idx]
                m_out[idx:] = mask[y1:y1+out_shape[0], x1:x1+out_shape[1], idx:]

    return out, m_out
                                                                                
def random_rotate_matrix(height, displacement):
    x = (displacement / 2.0)
    y = ((height - displacement) / 2.0) * 1.42
    angle = math.asin(x/y) * 2.0 * 57.2958 # convert radians to degrees
    rand_angle = (random.uniform(0, 1) - 0.5) * 2.0 * angle
    M = cv2.getRotationMatrix2D((height/2, height/2), rand_angle, 1)
    return M


def cutnoise(img, scale=(0.1,0.2), nb_iterations=(1,3), size=(0.2,0.4)):
    """Cutnoise data augmentation. Randomly add noise to a cuboid region in the 
       image to force the model to learn denoising when making predictions. 
       
       Parameters                                                               
       ----------                                                               
       img : 3D Numpy array                                                     
           Image to transform. E.g. ``(x, y, channels)``.                       
                                                                                
       scale : tuple of floats, optional                                                     
           Scale of the random noise. E.g. ``(0.1, 0.2)``.                      
            
       nb_iterations : tuple of ints, optional                                  
           Number of areas with noise to create. E.g. ``(1, 3)``.             
       
       size : boolean, optional                                        
           Range to choose the size of the areas to transform. E.g. ``(0.2, 0.4)``.
                                                                                
       Returns                                                                  
       -------                                                                  
       out : 3D Numpy array                                                     
           Transformed image. E.g. ``(x, y, channels)``.                        
    """

    it = np.random.randint(nb_iterations[0], nb_iterations[1])        
                                                                                
    out = img.copy()                                                            
    for i in range(it):                                                         
        _size = random.uniform(size[0], size[1])                       
        y_size = int(img.shape[0]*_size)                                         
        x_size = int(img.shape[1]*_size)                                         
                                                                                
        # Choose a random point                                                 
        cy = np.random.randint(0, img.shape[0]-(y_size))                        
        cx = np.random.randint(0, img.shape[1]-(x_size))                        
                                                                                
        _scale = random.uniform(scale[0], scale[1])
        noise_shape = (y_size, x_size)
        noise = np.random.uniform(-_scale, _scale, noise_shape) 

        # Apply cutnoise to all channels                                          
        for i in range(img.shape[-1]):                                          
            out[cy:cy+y_size, cx:cx+x_size, i] += noise

        out = np.clip(out, 0, 1)

    return out
