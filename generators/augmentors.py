import numpy as np
from skimage.transform import resize


def cutblur(img, size=0.4, down_ratio_range=(2,8), only_inside=True):
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

    y_size = int(img.shape[0]*size)
    x_size = int(img.shape[1]*size)

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


def cutmix(im1, im2, mask1, mask2, size=0.4):
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

       size : float, optional
           Size of the region to transform.

       Returns
       -------
       out : 3D Numpy array
           Transformed image. E.g. ``(x, y, channels)``.

       m_out : 3D Numpy array
           Transformed mask. E.g. ``(x, y, channels)``.
    """

    y_size = int(im1.shape[0]*size)
    x_size = int(im1.shape[1]*size)

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

