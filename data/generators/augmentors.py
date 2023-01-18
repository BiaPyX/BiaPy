import cv2
import random
import math
import numpy as np
from PIL import Image
from skimage.transform import resize
from skimage.draw import line
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import binary_dilation


def cutout(img, mask, channels, z_size, nb_iterations=(1,3), size=(0.2,0.4), cval=0, res_relation=(1,1), apply_to_mask=False):
    """Cutout data augmentation presented in `Improved Regularization of Convolutional Neural Networks with Cutout
       <https://arxiv.org/pdf/1708.04552.pdf>`_.

       Parameters
       ----------
       img : 3D Numpy array
           Image to transform. E.g. ``(y, x, channels)``.

       mask : 3D Numpy array
           Mask to transform. E.g. ``(y, x, channels)``.

       channels : int
           Size of channel dimension. Used for 3D images as the channels have been merged with the z axis.

       z_size : int
           Size of z dimension. Used for 3D images as the z axis has been merged with the channels. Set to -1 to when
           do not want to be applied.

       nb_iterations : tuple of ints, optional
           Number of areas to fill the image with. E.g. ``(1, 3)``.

       size : tuple of floats, optional
           Range to choose the size of the areas to create.

       cval : int, optional
           Value to fill the area with.

       res_relation: tuple of ints/floats, optional
           Relation between axis resolution in ``(x,y,z)``. E.g. ``(1,1,0.27)`` for anisotropic data of 
           8umx8umx30um resolution.

       apply_to_mask : boolean, optional
           To apply cutout to the mask.

       Returns
       -------
       out : 3D Numpy array
           Transformed image. E.g. ``(y, x, channels)``.

       mask : 3D Numpy array
           Transformed mask. E.g. ``(y, x, channels)``.

       Examples
       --------

       Calling this function with ``nb_iterations=(1,3)``, ``size=(0.05,0.3)``,
       ``apply_to_mask=False`` may result in:

       +-------------------------------------------+-------------------------------------------+
       | .. figure:: ../../img/orig_cutout.png     | .. figure:: ../../img/orig_cutout_mask.png|
       |   :width: 80%                             |   :width: 80%                             |
       |   :align: center                          |   :align: center                          |
       |                                           |                                           |
       |   Input image                             |   Corresponding mask                      |
       +-------------------------------------------+-------------------------------------------+
       | .. figure:: ../../img/cutout.png          | .. figure:: ../../img/cutout_mask.png     |
       |   :width: 80%                             |   :width: 80%                             |
       |   :align: center                          |   :align: center                          |
       |                                           |                                           |
       |   Augmented image                         |   Augmented mask                          |
       +-------------------------------------------+-------------------------------------------+

       The grid is painted for visualization purposes.
    """

    it = np.random.randint(nb_iterations[0], nb_iterations[1])

    out = img.copy()
    m_out = mask.copy()
    for i in range(it):
        _size = random.uniform(size[0], size[1])
        y_size = int(max(min(img.shape[0]*_size*res_relation[1],img.shape[0]),1))
        x_size = int(max(min(img.shape[1]*_size*res_relation[0],img.shape[1]),1))

        # Choose a random point
        cy = np.random.randint(0, img.shape[0]-y_size)
        cx = np.random.randint(0, img.shape[1]-x_size)
        if z_size != -1:
            _z_size = int(max(min(z_size*_size*res_relation[2],z_size),1))
            cz = np.random.randint(0, z_size-_z_size)

            out[cy:cy+y_size, cx:cx+x_size, cz*channels:(cz*channels)+(_z_size*channels)] = cval
            if apply_to_mask:
                m_out[cy:cy+y_size, cx:cx+x_size, cz*channels:(cz*channels)+(_z_size*channels)] = 0
        else:
            out[cy:cy+y_size, cx:cx+x_size] = cval
            if apply_to_mask:
                m_out[cy:cy+y_size, cx:cx+x_size] = 0

    return out, m_out


def cutblur(img, size=(0.2,0.4), down_ratio_range=(2,8), only_inside=True):
    """CutBlur data augmentation introduced in `Rethinking Data Augmentation for Image Super-resolution: A Comprehensive
       Analysis and a New Strategy <https://arxiv.org/pdf/2004.00448.pdf>`_ and adapted from
       https://github.com/clovaai/cutblur .


       Parameters
       ----------
       img : 3D Numpy array
           Image to transform. E.g. ``(y, x, channels)``.

       size : float, optional
           Size of the region to transform.

       down_ratio_range : tuple of ints, optional
           Downsampling ratio range to be applied. E.g. ``(2, 8)``.

       only_inside : bool, optional
           If ``True`` only the region inside will be modified (cut LR into HR image). If ``False`` the ``50%`` of the
           times the region inside will be modified (cut LR into HR image) and the other ``50%`` the inverse will be
           done (cut HR into LR image). See Figure 1 of the official `paper <https://arxiv.org/pdf/2004.00448.pdf>`_.

       Returns
       -------
       out : 3D Numpy array
           Transformed image. E.g. ``(y, x, channels)``.

       Examples
       --------

       Calling this function with ``size=(0.2,0.4)``, ``down_ratio_range=(2,8)``,
       ``only_inside=True`` may result in:

       +-----------------------------------------+-----------------------------------------+
       | .. figure:: ../../img/orig_cutblur.png  | .. figure:: ../../img/cutblur.png       |
       |   :width: 80%                           |   :width: 80%                           |
       |   :align: center                        |   :align: center                        |
       |                                         |                                         |
       |   Input image                           |   Augmented image                       |
       +-----------------------------------------+-----------------------------------------+
       | .. figure:: ../../img/orig_cutblur2.png | .. figure:: ../../img/cutblur2.png      |
       |   :width: 80%                           |   :width: 80%                           |
       |   :align: center                        |   :align: center                        |
       |                                         |                                         |
       |   Input image                           |   Augmented image                       |
       +-----------------------------------------+-----------------------------------------+

       The grid and the red square are painted for visualization purposes.
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
        inside = True if random.uniform(0, 1) < 0.5 else False
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
    """Cutmix augmentation introduced in `CutMix: Regularization Strategy to Train Strong Classifiers with Localizable
       Features <https://arxiv.org/abs/1905.04899>`_. With this augmentation a region of the image sample is filled
       with a given second image. This implementation is used for semantic segmentation so the masks of the images are
       also needed. It assumes that the images are of the same shape.

       Parameters
       ----------
       im1 : 3D Numpy array
           Image to transform. E.g. ``(y, x, channels)``.

       im2 : 3D Numpy array
           Image to paste into the region of ``im1``. E.g. ``(y, x, channels)``.

       mask1 : 3D Numpy array
           Mask to transform (belongs to ``im1``). E.g. ``(y, x, channels)``.

       mask2 : 3D Numpy array
           Mask to paste into the region of ``mask1``. E.g. ``(y, x, channels)``.

       size : tuple of floats, optional
           Range to choose the size of the areas to transform. E.g. ``(0.2, 0.4)``.

       Returns
       -------
       out : 3D Numpy array
           Transformed image. E.g. ``(y, x, channels)``.

       m_out : 3D Numpy array
           Transformed mask. E.g. ``(y, x, channels)``.

       Examples
       --------

       Calling this function with ``size=(0.2,0.4)`` may result in:

       +-------------------------------------------+-------------------------------------------+
       | .. figure:: ../../img/orig_cutmix.png     | .. figure:: ../../img/orig_cutmix_mask.png|
       |   :width: 80%                             |   :width: 80%                             |
       |   :align: center                          |   :align: center                          |
       |                                           |                                           |
       |   Input image                             |   Corresponding mask                      |
       +-------------------------------------------+-------------------------------------------+
       | .. figure:: ../../img/cutmix.png          | .. figure:: ../../img/cutmix_mask.png     |
       |   :width: 80%                             |   :width: 80%                             |
       |   :align: center                          |   :align: center                          |
       |                                           |                                           |
       |   Augmented image                         |   Augmented mask                          |
       +-------------------------------------------+-------------------------------------------+

       The grid is painted for visualization purposes.
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
        out[im1cy:im1cy+y_size, im1cx:im1cx+x_size, i] = im2[im2cy:im2cy+y_size, im2cx:im2cx+x_size, i]
        m_out[im1cy:im1cy+y_size, im1cx:im1cx+x_size, i] = mask2[im2cy:im2cy+y_size, im2cx:im2cx+x_size, i]

    return out, m_out


def cutnoise(img, scale=(0.1,0.2), nb_iterations=(1,3), size=(0.2,0.4)):
    """Cutnoise data augmentation. Randomly add noise to a cuboid region in the image to force the model to learn
       denoising when making predictions.

       Parameters
       ----------
       img : 3D Numpy array
           Image to transform. E.g. ``(y, x, channels)``.

       scale : tuple of floats, optional
           Scale of the random noise. E.g. ``(0.1, 0.2)``.

       nb_iterations : tuple of ints, optional
           Number of areas with noise to create. E.g. ``(1, 3)``.

       size : boolean, optional
           Range to choose the size of the areas to transform. E.g. ``(0.2, 0.4)``.

       Returns
       -------
       out : 3D Numpy array
           Transformed image. E.g. ``(y, x, channels)``.

       Examples
       --------

       Calling this function with ``scale=(0.1,0.2)``, ``nb_iterations=(1,3)`` and
       ``size=(0.2,0.4)`` may result in:

       +------------------------------------------+------------------------------------------+
       | .. figure:: ../../img/orig_cutnoise.png  | .. figure:: ../../img/cutnoise.png       |
       |   :width: 80%                            |   :width: 80%                            |
       |   :align: center                         |   :align: center                         |
       |                                          |                                          |
       |   Input image                            |   Augmented image                        |
       +------------------------------------------+------------------------------------------+
       | .. figure:: ../../img/orig_cutnoise2.png | .. figure:: ../../img/cutnoise2.png      |
       |   :width: 80%                            |   :width: 80%                            |
       |   :align: center                         |   :align: center                         |
       |                                          |                                          |
       |   Input image                            |   Augmented image                        |
       +------------------------------------------+------------------------------------------+

       The grid and the red squares are painted for visualization purposes.
    """

    it = np.random.randint(nb_iterations[0], nb_iterations[1])

    out = img.copy().astype(np.int16)
    for i in range(it):
        _size = random.uniform(size[0], size[1])
        y_size = int(img.shape[0]*_size)
        x_size = int(img.shape[1]*_size)

        # Choose a random point
        cy = np.random.randint(0, img.shape[0]-(y_size))
        cx = np.random.randint(0, img.shape[1]-(x_size))

        max_value = np.max(out)
        _scale = random.uniform(scale[0], scale[1])*max_value
        noise = np.random.normal(loc=0, scale=_scale, size=(y_size, x_size))
        out[cy:cy+y_size, cx:cx+x_size, :] += np.stack((noise,)*out.shape[-1], axis=-1).astype(np.int16)
    return np.clip(out, 0, max_value)


def misalignment(img, mask, displacement=16, rotate_ratio=0.0, c_relation="1_1"):
    """Mis-alignment data augmentation of image stacks. This augmentation is applied to both images and masks.

       Implementation based on `PyTorch Connectomics' misalign.py
       <https://github.com/zudi-lin/pytorch_connectomics/blob/master/connectomics/data/augmentation/misalign.py>`_.

       Parameters
       ----------
       img : 3D Numpy array
           Image to transform. E.g. ``(y, x, channels)``.

       mask : 3D Numpy array
           Mask to transform. E.g. ``(y, x, channels)``

       displacement : int, optional
           Maximum pixel displacement in ``xy``-plane.

       rotate_ratio : float, optional
           Ratio of rotation-based mis-alignment.

       Returns
       -------
       out : 3D Numpy array
           Transformed image. E.g. ``(y, x, channels)``.

       m_out : 3D Numpy array
           Transformed mask. E.g. ``(y, x, channels)``.

       Examples
       --------

       Calling this function with ``displacement=16`` and ``rotate_ratio=0.5`` may result in:

       +------------------------------------------+------------------------------------------+
       | .. figure:: ../../img/orig_miss.png      | .. figure:: ../../img/orig_miss_mask.png |
       |   :width: 80%                            |   :width: 80%                            |
       |   :align: center                         |   :align: center                         |
       |                                          |                                          |
       |   Input image                            |   Corresponding mask                     |
       +------------------------------------------+------------------------------------------+
       | .. figure:: ../../img/miss.png           | .. figure:: ../../img/miss_mask.png      |
       |   :width: 80%                            |   :width: 80%                            |
       |   :align: center                         |   :align: center                         |
       |                                          |                                          |
       |   Augmented image                        |   Augmented mask                         |
       +------------------------------------------+------------------------------------------+

       The grid is painted for visualization purposes.
    """

    out = np.zeros(img.shape, img.dtype)
    m_out = np.zeros(mask.shape, mask.dtype)

    # 2D
    if img.shape[-1] < 5:
        oy = np.random.randint(1, img.shape[0]-1)
        d = np.random.randint(0, displacement)
        if random.uniform(0, 1) < rotate_ratio:
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
                out[oy:,:,i] = cv2.warpAffine(img[oy:,:,i], M, (W,H), 1.0, flags=cv2.INTER_LINEAR,
                                              borderMode=cv2.BORDER_CONSTANT)
                m_out[:oy,:,i] = mask[:oy,:,i]
                m_out[oy:,:,i] = cv2.warpAffine(mask[oy:,:,i], M, (W,H), 1.0, flags=cv2.INTER_NEAREST,
                                                borderMode=cv2.BORDER_CONSTANT)
    # 3D
    else:
        out_shape = (img.shape[0]-displacement, img.shape[1]-displacement,
                     img.shape[2])

        mode = 'slip' if random.uniform(0, 1) < 0.5 else 'translation'

        # Calculate the amount of image and mask channels to determine which slice apply the tranformation to
        idx = np.random.randint(1, img.shape[-1]-1)
        relation = c_relation.split('_')
        img_rel = int(relation[0])
        mask_rel = int(relation[1])
        idx = int(idx - (idx%img_rel))
        idx_mask = int((idx/img_rel)*mask_rel)

        if random.uniform(0, 1) < rotate_ratio:
            out = img.copy()
            m_out = mask.copy()

            H, W = img.shape[:2]
            M = random_rotate_matrix(H, displacement)
            if mode == 'slip':
                # Apply the change to all images/masks in the last dimension that represent the slice selected. This
                # needs to be done because the last dimension is z axis mutiplied by the channels
                for k in range(img_rel):
                    out[...,idx+k] = 0
                    out[...,idx+k] = cv2.warpAffine(img[...,idx+k], M, (H,W), 1.0, flags=cv2.INTER_LINEAR,
                                                    borderMode=cv2.BORDER_CONSTANT)
                for k in range(mask_rel):
                    m_out[...,idx_mask+k] = 0
                    m_out[...,idx_mask+k] = cv2.warpAffine(mask[...,idx_mask+k], M, (H,W), 1.0, flags=cv2.INTER_NEAREST,
                                                           borderMode=cv2.BORDER_CONSTANT)
            else:
                for i in range(idx, img.shape[-1]):
                    out[...,i] = 0
                    out[...,i] = cv2.warpAffine(img[...,i], M, (H,W), 1.0, flags=cv2.INTER_LINEAR,
                                                borderMode=cv2.BORDER_CONSTANT)
                for i in range(idx_mask, mask.shape[-1]):
                    m_out[...,i] = 0
                    m_out[...,i] = cv2.warpAffine(mask[...,i], M, (H,W), 1.0, flags=cv2.INTER_NEAREST,
                                                  borderMode=cv2.BORDER_CONSTANT)
        else:
            random_state = np.random.RandomState()
            x0 = random_state.randint(displacement)
            y0 = random_state.randint(displacement)
            x1 = random_state.randint(displacement)
            y1 = random_state.randint(displacement)
            if mode == 'slip':
                out[y0:y0+out_shape[0],x0:x0+out_shape[1],:] = img[y0:y0+out_shape[0],x0:x0+out_shape[1],:]
                for k in range(img_rel):
                    out[...,idx+k] = 0
                    out[y1:y1+out_shape[0],x1:x1+out_shape[1],idx+k] = img[y1:y1+out_shape[0], x1:x1+out_shape[1],idx+k]

                m_out[y0:y0+out_shape[0], x0:x0+out_shape[1],:] = mask[y0:y0+out_shape[0], x0:x0+out_shape[1],:]
                for k in range(mask_rel):
                    m_out[...,idx_mask+k] = 0
                    m_out[y1:y1+out_shape[0],x1:x1+out_shape[1],idx_mask+k] = mask[y1:y1+out_shape[0],x1:x1+out_shape[1],idx_mask+k]
            else:
                out[y0:y0+out_shape[0],x0:x0+out_shape[1],:idx] = img[y0:y0+out_shape[0],x0:x0+out_shape[1],:idx]
                out[y1:y1+out_shape[0],x1:x1+out_shape[1],idx:] = img[y1:y1+out_shape[0],x1:x1+out_shape[1],idx:]
                m_out[y0:y0+out_shape[0],x0:x0+out_shape[1],:idx_mask] = mask[y0:y0+out_shape[0],x0:x0+out_shape[1],:idx_mask]
                m_out[y1:y1+out_shape[0],x1:x1+out_shape[1],idx_mask:] = mask[y1:y1+out_shape[0],x1:x1+out_shape[1],idx_mask:]

    return out, m_out


def random_rotate_matrix(height, displacement):
    """Auxiliary function for missaligmnet. """
    x = (displacement / 2.0)
    y = ((height - displacement) / 2.0) * 1.42
    angle = math.asin(x/y) * 2.0 * 57.2958 # convert radians to degrees
    rand_angle = (random.uniform(0, 1) - 0.5) * 2.0 * angle
    M = cv2.getRotationMatrix2D((height/2, height/2), rand_angle, 1)
    return M

def brightness(image, brightness_factor=(0,0),  mode='2D'):
    """Randomly adjust brightness between a range.

       Parameters
       ----------
       image : 3D Numpy array
           Image to transform. E.g. ``(y, x, channels)``.

       brightness_factor : tuple of 2 floats, optional
           Range of brightness' intensity. E.g. ``(0.1, 0.3)``.

       mode : str, optional
           One of ``2D`` or ``3D``.

       Returns
       -------
       image : 3D Numpy array
           Transformed image. E.g. ``(y, x, channels)``.

       Examples
       --------

       Calling this function with ``brightness_factor=(0.1,0.3)``, ``mode='mix'``, ``invert=False`` and ``invert_p=0``
       may result in:

       +------------------------------------------+------------------------------------------+
       | .. figure:: ../../img/orig_bright.png    | .. figure:: ../../img/bright.png         |
       |   :width: 80%                            |   :width: 80%                            |
       |   :align: center                         |   :align: center                         |
       |                                          |                                          |
       |   Input image                            |   Augmented image                        |
       +------------------------------------------+------------------------------------------+
       | .. figure:: ../../img/orig_bright2.png   | .. figure:: ../../img/bright2.png        |
       |   :width: 80%                            |   :width: 80%                            |
       |   :align: center                         |   :align: center                         |
       |                                          |                                          |
       |   Input image                            |   Augmented image                        |
       +------------------------------------------+------------------------------------------+

       The grid is painted for visualization purposes.
    """

    if brightness_factor[0] == 0 and brightness_factor[1] == 0: return image

    # Force mode if 2D
    if image.ndim == 3: mode == '3D'

    if mode == '2D':
        b_factor = np.random.uniform(brightness_factor[0], brightness_factor[1], image.shape[-1]*3)
        for z in range(image.shape[2]):
            image[:, :, z] += b_factor[z*3]
            image[:, :, z] = np.clip(image[:, :, z], 0, 1)
    else:
        b_factor = np.random.uniform(brightness_factor[0], brightness_factor[1])
        image += b_factor
        image = np.clip(image, 0, 1)

    return image

def contrast(image, contrast_factor=(0,0), mode='2D'):
    """Contrast augmentation.

       Parameters
       ----------
       image : 3D Numpy array
           Image to transform. E.g. ``(y, x, channels)``.

       contrast_factor : tuple of 2 floats, optional
           Range of contrast's intensity. E.g. ``(0.1, 0.3)``.

       mode : str, optional
           One of ``2D`` or ``3D``.

       Returns
       -------
       image : 3D Numpy array
           Transformed image. E.g. ``(y, x, channels)``.

       Examples
       --------

       Calling this function with ``contrast_factor=(0.1,0.3)``, ``mode='mix'``, ``invert=False`` and ``invert_p=0``
       may result in:

       +------------------------------------------+------------------------------------------+
       | .. figure:: ../../img/orig_contrast.png  | .. figure:: ../../img/contrast.png       |
       |   :width: 80%                            |   :width: 80%                            |
       |   :align: center                         |   :align: center                         |
       |                                          |                                          |
       |   Input image                            |   Augmented image                        |
       +------------------------------------------+------------------------------------------+
       | .. figure:: ../../img/orig_contrast2.png | .. figure:: ../../img/contrast2.png      |
       |   :width: 80%                            |   :width: 80%                            |
       |   :align: center                         |   :align: center                         |
       |                                          |                                          |
       |   Input image                            |   Augmented image                        |
       +------------------------------------------+------------------------------------------+

       The grid is painted for visualization purposes.
    """

    if contrast_factor[0] == 0 and contrast_factor[1] == 0: return image

    # Force mode if 2D
    if image.ndim == 3: mode == '3D'

    if mode == '2D':
        c_factor = np.random.uniform(contrast_factor[0], contrast_factor[1], image.shape[-1]*3)
        for z in range(image.shape[2]):
            image[:, :, z] *= 1 + c_factor[z*3]
            image[:, :, z] = np.clip(image[:, :, z], 0, 1)
    else:
        c_factor = np.random.uniform(contrast_factor[0], contrast_factor[1])
        image *= 1 + c_factor
        image = np.clip(image, 0, 1)

    return image


def brightness_em(image, brightness_factor=(0,0),  mode='2D', invert=False, invert_p=0):
    """Randomly adjust brightness, randomly invert the color space and apply gamma correction. 

       Implementation based on `PyTorch Connectomics' grayscale.py
       <https://github.com/zudi-lin/pytorch_connectomics/blob/master/connectomics/data/augmentation/grayscale.py>`_.

       Parameters
       ----------
       image : 3D Numpy array
           Image to transform. E.g. ``(y, x, channels)``.

       brightness_factor : tuple of 2 floats, optional
           Range of brightness' intensity. E.g. ``(0.1, 0.3)``.

       mode : str, optional
           One of ``2D`` or ``3D``.

       invert : bool, optional
           Whether to invert the images.

       invert_p : float, optional
           Probability of inverting the images.

       Returns
       -------
       image : 3D Numpy array
           Transformed image. E.g. ``(y, x, channels)``.

       Examples
       --------

       Calling this function with ``brightness_factor=(0.1,0.3)``, ``mode='mix'``, ``invert=False`` and ``invert_p=0``
       may result in:

       +------------------------------------------+------------------------------------------+
       | .. figure:: ../../img/orig_bright.png    | .. figure:: ../../img/bright.png         |
       |   :width: 80%                            |   :width: 80%                            |
       |   :align: center                         |   :align: center                         |
       |                                          |                                          |
       |   Input image                            |   Augmented image                        |
       +------------------------------------------+------------------------------------------+
       | .. figure:: ../../img/orig_bright2.png   | .. figure:: ../../img/bright2.png        |
       |   :width: 80%                            |   :width: 80%                            |
       |   :align: center                         |   :align: center                         |
       |                                          |                                          |
       |   Input image                            |   Augmented image                        |
       +------------------------------------------+------------------------------------------+

       The grid is painted for visualization purposes.
    """

    if brightness_factor[0] == 0 and brightness_factor[1] == 0: return image

    # Force mode if 2D
    if img.ndim == 3: mode == '3D'

    b_factor = random.uniform(brightness_factor[0], brightness_factor[1])
    if mode == '2D':
        ran = np.random.rand(image.shape[-1]*3)
        for z in range(image.shape[2]):
            img = image[:, :, z]
            image[:, :, z] += (ran[z*3+1] - 0.5)*b_factor
            image[:, :, z] = np.clip(image[:, :, z], 0, 1)
            image[:, :, z] **= 2.0**(ran[z*3+2]*2 - 1)
    else:
        ran = np.random.rand(3)
        image += (ran[1] - 0.5)*b_factor
        image = np.clip(image, 0, 1)
        image **= 2.0**(ran[2]*2 - 1)

    if invert and random.uniform(0, 1) < invert_p:
        image = 1.0-image
        image = np.clip(image, 0, 1)

    return image


def contrast_em(image, contrast_factor=(0,0), mode='2D', invert=False, invert_p=0):
    """Contrast augmentation. Randomly invert the color space and apply gamma correction. 

       Implementation based on `PyTorch Connectomics' grayscale.py
       <https://github.com/zudi-lin/pytorch_connectomics/blob/master/connectomics/data/augmentation/grayscale.py>`_.

       Parameters
       ----------
       image : 3D Numpy array
           Image to transform. E.g. ``(y, x, channels)``.

       contrast_factor : tuple of 2 floats, optional
           Range of contrast's intensity. E.g. ``(0.1, 0.3)``.

       mode : str, optional
           One of ``2D`` or ``3D``.

       invert : bool, optional
           Whether to invert the image.

       invert_p : float, optional
           Probability of inverting the image.

       Returns
       -------
       image : 3D Numpy array
           Transformed image. E.g. ``(y, x, channels)``.

       Examples
       --------

       Calling this function with ``contrast_factor=(0.1,0.3)``, ``mode='mix'``, ``invert=False`` and ``invert_p=0``
       may result in:

       +------------------------------------------+------------------------------------------+
       | .. figure:: ../../img/orig_contrast.png  | .. figure:: ../../img/contrast.png       |
       |   :width: 80%                            |   :width: 80%                            |
       |   :align: center                         |   :align: center                         |
       |                                          |                                          |
       |   Input image                            |   Augmented image                        |
       +------------------------------------------+------------------------------------------+
       | .. figure:: ../../img/orig_contrast2.png | .. figure:: ../../img/contrast2.png      |
       |   :width: 80%                            |   :width: 80%                            |
       |   :align: center                         |   :align: center                         |
       |                                          |                                          |
       |   Input image                            |   Augmented image                        |
       +------------------------------------------+------------------------------------------+

       The grid is painted for visualization purposes.
    """
    if contrast_factor[0] == 0 and contrast_factor[1] == 0: return image

    # Force mode if 2D
    if image.ndim == 3: mode == '3D'

    c_factor = random.uniform(contrast_factor[0], contrast_factor[1])
    if mode == '2D':
        ran = np.random.rand(image.shape[-1]*3)
        for z in range(image.shape[2]):
            img = image[:, :, z]
            image[:, :, z] *= 1 + (ran[z*3] - 0.5)*c_factor
            image[:, :, z] = np.clip(image[:, :, z], 0, 1)
            image[:, :, z] **= 2.0**(ran[z*3+2]*2 - 1)
    else:
        ran = np.random.rand(3)
        image *= 1 + (ran[0] - 0.5)*c_factor
        image = np.clip(image, 0, 1)
        image **= 2.0**(ran[2]*2 - 1)

    if invert and random.uniform(0, 1) < invert_p:
        image = 1.0-image
        image = np.clip(image, 0, 1)

    return image


def missing_sections(img, iterations=(30,40)):
    """Augment the image by creating a black line in a random position.

       Implementation based on `PyTorch Connectomics' missing_parts.py
       <https://github.com/zudi-lin/pytorch_connectomics/blob/master/connectomics/data/augmentation/missing_parts.py>`_.

       Parameters
       ----------
       img : 3D Numpy array
           Image to transform. E.g. ``(y, x, channels)``.

       iterations : tuple of 2 ints, optional
           Iterations to dilate the missing line with. E.g. ``(30, 40)``.

       Returns
       -------
       out : 3D Numpy array
           Transformed image. E.g. ``(y, x, channels)``.

       Examples
       --------

       Calling this function with ``iterations=(30,40)`` may result in:

       +------------------------------------------+------------------------------------------+
       | .. figure:: ../../img/orig_missing.png   | .. figure:: ../../img/missing.png        |
       |   :width: 80%                            |   :width: 80%                            |
       |   :align: center                         |   :align: center                         |
       |                                          |                                          |
       |   Input image                            |   Augmented image                        |
       +------------------------------------------+------------------------------------------+
       | .. figure:: ../../img/orig_missing2.png  | .. figure:: ../../img/missing2.png       |
       |   :width: 80%                            |   :width: 80%                            |
       |   :align: center                         |   :align: center                         |
       |                                          |                                          |
       |   Input image                            |   Augmented image                        |
       +------------------------------------------+------------------------------------------+

       The grid is painted for visualization purposes.
    """

    it = np.random.randint(iterations[0], iterations[1])

    num_section = img.shape[-1]
    slice_shape = img.shape[:2]
    transforms = {}

    out = img.copy()

    i=0
    while i < num_section:
        if np.random.rand() < 0.5:
            transforms[i] = _prepare_deform_slice(slice_shape, it)
            i += 2 # at most one deformed image in any consecutive 3 images
        i += 1

    num_section = img.shape[-1]
    for i in range(num_section):
        if i in transforms.keys():
            line_mask = transforms[i]
            mean = img[...,i].mean()
            out[:,:,i][line_mask] = mean

    return out


def _prepare_deform_slice(slice_shape, iterations):
    """Auxiliary function for missing_sections. """
    shape = (slice_shape[0], slice_shape[1])
    # randomly choose fixed x or fixed y with p = 1/2
    fixed_x = np.random.rand() < 0.5
    if fixed_x:
        x0, y0 = 0, np.random.randint(1, shape[1] - 2)
        x1, y1 = shape[0] - 1, np.random.randint(1, shape[1] - 2)
    else:
        x0, y0 = np.random.randint(1, shape[0] - 2), 0
        x1, y1 = np.random.randint(1, shape[0] - 2), shape[1] - 1

    # generate the mask of the line that should be blacked out
    line_mask = np.zeros(shape, dtype='bool')
    rr, cc = line(x0, y0, x1, y1)
    line_mask[rr, cc] = 1

    # generate vectorfield pointing towards the line to compress the image
    # first we get the unit vector representing the line
    line_vector = np.array([x1 - x0, y1 - y0], dtype='float32')
    line_vector /= np.linalg.norm(line_vector)
    # next, we generate the normal to the line
    normal_vector = np.zeros_like(line_vector)
    normal_vector[0] = - line_vector[1]
    normal_vector[1] = line_vector[0]

    # find the 2 components where coordinates are bigger / smaller than the line
    # to apply normal vector in the correct direction
    components, n_components = label(np.logical_not(line_mask).view('uint8'))
    assert n_components == 2, "%i" % n_components

    # dilate the line mask
    line_mask = binary_dilation(line_mask, iterations=iterations)

    return line_mask


def shuffle_channels(img):
    """Augment the image by shuffling its channels.

       Parameters
       ----------
       img : 3D/4D Numpy array
           Image to transform. E.g. ``(y, x, channels)`` or ``(z, y, x, channels)``.

       Returns
       -------
       out : 3D/4D Numpy array
           Transformed image. E.g. ``(y, x, channels)`` or ``(z, y, x, channels)``.

       Examples
       --------

       +--------------------------------------------+--------------------------------------------+
       | .. figure:: ../../img/orig_chshuffle.png   | .. figure:: ../../img/chshuffle.png        |
       |   :width: 80%                              |   :width: 80%                              |
       |   :align: center                           |   :align: center                           |
       |                                            |                                            |
       |   Input image                              |   Augmented image                          |
       +--------------------------------------------+--------------------------------------------+

       The grid is painted for visualization purposes.
    """

    if img.ndim != 3 and img.ndim != 4:
        raise ValueError("Image is supposed to be 3 or 4 dimensions but provided {} image shape instead".format(img.shape))

    new_channel_order = np.random.permutation(img.shape[-1])
    return img[...,new_channel_order]


def grayscale(img):
    """Augment the image by converting it into grayscale.

       Parameters
       ----------
       img : 3D/4D Numpy array
           Image to transform. E.g. ``(y, x, channels)`` or ``(z, y, x, channels)``.

       Returns
       -------
       out : 3D/4D Numpy array
           Transformed image. E.g. ``(y, x, channels)`` or ``(z, y, x, channels)``.

       Examples
       --------

       +--------------------------------------------+--------------------------------------------+
       | .. figure:: ../../img/orig_grayscale.png   | .. figure:: ../../img/grayscale.png        |
       |   :width: 80%                              |   :width: 80%                              |
       |   :align: center                           |   :align: center                           |
       |                                            |                                            |
       |   Input image                              |   Augmented image                          |
       +--------------------------------------------+--------------------------------------------+

       The grid is painted for visualization purposes.
    """

    if img.shape[-1] != 3:
        raise ValueError("Image is supposed to have 3 channels (RGB). Provided {} image shape instead".format(img.shape))

    return np.tile(np.expand_dims(np.mean(img, -1), -1), 3)


def GridMask(img, channels, z_size, ratio=0.6, d_range=(30,60), rotate=1, invert=False):
    """GridMask data augmentation presented in `GridMask Data Augmentation <https://arxiv.org/abs/2001.04086v1>`_.
       Code adapted from `<https://github.com/dvlab-research/GridMask/blob/master/imagenet_grid/utils/grid.py>`_.

       Parameters
       ----------
       img : 3D Numpy array
           Image to transform. E.g. ``(y, x, channels)``.

       channels : int
           Size of channel dimension. Used for 3D images as the channels have been merged with the z axis.

       z_size : int
           Size of z dimension. Used for 3D images as the z axis has been merged with the channels. Set to -1 to when
           do not want to be applied.

       ratio : tuple of floats, optional
           Range to choose the size of the areas to create.

       d_range : tuple of ints, optional
           Range to choose the ``d`` value in the original paper.

       rotate : float, optional
           Rotation of the mask in GridMask. Needs to be between ``[0,1]`` where 1 is 360 degrees.

       invert : bool, optional
           Whether to invert the mask.

       Returns
       -------
       out : 3D Numpy array
           Transformed image. E.g. ``(y, x, channels)``.

       Examples
       --------

       Calling this function with the default settings may result in:

       +-------------------------------------------+-------------------------------------------+
       | .. figure:: ../../img/orig_GridMask.png   | .. figure:: ../../img/GridMask.png        |
       |   :width: 80%                             |   :width: 80%                             |
       |   :align: center                          |   :align: center                          |
       |                                           |                                           |
       |   Input image                             |   Augmented image                         |
       +-------------------------------------------+-------------------------------------------+

       The grid is painted for visualization purposes.
    """

    h,w,c = img.shape

    # 1.5 * h, 1.5 * w works fine with the squared images
    # But with rectangular input, the mask might not be able to recover back to the input image shape
    # A square mask with edge length equal to the diagnoal of the input image
    # will be able to cover all the image spot after the rotation. This is also the minimum square.
    hh = math.ceil((math.sqrt(h*h + w*w)))

    d = np.random.randint(d_range[0], d_range[1])

    l = math.ceil(d*ratio)

    mask = np.ones((hh, hh), np.float32)
    st_h = np.random.randint(d)
    st_w = np.random.randint(d)
    for i in range(-1, hh//d+1):
            s = d*i + st_h
            t = s+l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[s:t,:] *= 0
    for i in range(-1, hh//d+1):
            s = d*i + st_w
            t = s+l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[:,s:t] *= 0
    r = np.random.randint(rotate)
    mask = Image.fromarray(np.uint8(mask))
    mask = mask.rotate(r)
    mask = np.asarray(mask)
    mask = mask[(hh-h)//2:(hh-h)//2+h, (hh-w)//2:(hh-w)//2+w]

    if not invert: mask = 1-mask
    if z_size != -1:
        _z_size = np.random.randint(d_range[2], d_range[3])
        cz = np.random.randint(0, z_size-_z_size)
        img[...,cz*channels:(cz*channels)+(_z_size*channels)] *= np.stack((mask,)*(_z_size*channels), axis=-1)
        return img
    else:
        return img*np.stack((mask,)*img.shape[-1], axis=-1)


def random_crop_pair(image, mask, random_crop_size, val=False, draw_prob_map_points=False, img_prob=None, weight_map=None,
        scale=1):
    """Random crop for an image and its mask.

       Parameters
       ----------
       image : Numpy 3D array
           Image. E.g. ``(y, x, channels)``.

       mask : Numpy 3D array
           Image mask. E.g. ``(y, x, channels)``.

       random_crop_size : 2 int tuple
           Size of the crop. E.g. ``(height, width)``.

       val : bool, optional
           If the image provided is going to be used in the validation data. This forces to crop from the origin,
           e. g. ``(0, 0)`` point.

       draw_prob_map_points : bool, optional
           To return the pixel chosen to be the center of the crop.

       img_prob : Numpy 3D array, optional
           Probability of each pixel to be chosen as the center of the crop. E. .g. ``(y, x, channels)``.

       weight_map : bool, optional
           Weight map of the given image. E.g. ``(y, x, channels)``.

       scale : int, optional
           Scale factor the second image given.

       Returns
       -------
       img : 2D Numpy array
           Crop of the given image. E.g. ``(y, x)``.

       weight_map : 2D Numpy array, optional
           Crop of the given image's weigth map. E.g. ``(y, x)``.

       ox : int, optional
           X coordinate in the complete image of the chose central pixel to make the crop.

       oy : int, optional
           Y coordinate in the complete image of the chose central pixel to make the crop.

       x : int, optional
           X coordinate in the complete image where the crop starts.

       y : int, optional
           Y coordinate in the complete image where the crop starts.
    """

    if weight_map is not None:
        img, we = image
    else:
        img = image

    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size[0], random_crop_size[1]
    if val:
        y, x, oy, ox = 0, 0, 0, 0
    else:
        if img_prob is not None:
            prob = img_prob.ravel()

            # Generate the random coordinates based on the distribution
            choices = np.prod(img_prob.shape)
            index = np.random.choice(choices, size=1, p=prob)
            coordinates = np.unravel_index(index, dims=img_prob.shape)
            x = int(coordinates[1][0])
            y = int(coordinates[0][0])
            ox = int(coordinates[1][0])
            oy = int(coordinates[0][0])

            # Adjust the coordinates to be the origin of the crop and control to
            # not be out of the image
            if y < int(random_crop_size[0]/2):
                y = 0
            elif y > img.shape[0] - int(random_crop_size[0]/2):
                y = img.shape[0] - random_crop_size[0]
            else:
                y -= int(random_crop_size[0]/2)

            if x < int(random_crop_size[1]/2):
                x = 0
            elif x > img.shape[1] - int(random_crop_size[1]/2):
                x = img.shape[1] - random_crop_size[1]
            else:
                x -= int(random_crop_size[1]/2)
        else:
            oy, ox = 0, 0
            x = np.random.randint(0, width - dx + 1)
            y = np.random.randint(0, height - dy + 1)

    if draw_prob_map_points == True:
        return img[y:(y+dy), x:(x+dx)], mask[y*scale:(y+dy)*scale, x*scale:(x+dx)*scale], oy, ox, y, x
    else:
        if weight_map is not None:
            return img[y:(y+dy), x:(x+dx)], mask[y*scale:(y+dy)*scale, x*scale:(x+dx)*scale], weight_map[y:(y+dy), x:(x+dx)]
        else:
            return img[y:(y+dy), x:(x+dx)], mask[y*scale:(y+dy)*scale, x*scale:(x+dx)*scale]


def random_3D_crop_pair(image, mask, random_crop_size, val=False, img_prob=None, weight_map=None, draw_prob_map_points=False,
        scale=1):
    """Extracts a random 3D patch from the given image and mask.

       Parameters
       ----------
       image : 4D Numpy array
           Data to extract the patch from. E.g. ``(z, y, x, channels)``.

       mask : 4D Numpy array
           Data mask to extract the patch from. E.g. ``(z, y, x, channels)``.

       random_crop_size : 3D int tuple
           Shape of the patches to create. E.g. ``(z, y, x)``.

       val : bool, optional
           If the image provided is going to be used in the validation data. This forces to crop from the origin, e.g.
           ``(0, 0)`` point.

       img_prob : Numpy 4D array, optional
           Probability of each pixel to be chosen as the center of the crop. E. g. ``(z, y, x, channels)``.

       weight_map : bool, optional
           Weight map of the given image. E.g. ``(y, x, channels)``.

       draw_prob_map_points : bool, optional
           To return the voxel chosen to be the center of the crop.

       scale : int, optional
           Scale factor the second image given.

       Returns
       -------
       img : 4D Numpy array
           Crop of the given image. E.g. ``(z, y, x, channels)``.

       weight_map : 4D Numpy array, optional
           Crop of the given image's weigth map. E.g. ``(z, y, x, channels)``.
    
       oz : int, optional
           Z coordinate in the complete image of the chose central pixel to
           make the crop.

       oy : int, optional
           Y coordinate in the complete image of the chose central pixel to
           make the crop.

       ox : int, optional
           X coordinate in the complete image of the chose central pixel to
           make the crop.

       z : int, optional
           Z coordinate in the complete image where the crop starts.

       y : int, optional
           Y coordinate in the complete image where the crop starts.

       x : int, optional
           X coordinate in the complete image where the crop starts.
    """

    if weight_map is not None:
        vol, we = image
    else:
        vol = image

    deep, cols, rows = vol.shape[0], vol.shape[1], vol.shape[2]
    dz, dy, dx = random_crop_size
    assert rows >= dx
    assert cols >= dy
    assert deep >= dz
    if val:
        x, y, z, ox, oy, oz = 0, 0, 0, 0, 0, 0
    else:
        if img_prob is not None:
            prob = img_prob.ravel()

            # Generate the random coordinates based on the distribution
            choices = np.prod(img_prob.shape)
            index = np.random.choice(choices, size=1, p=prob)
            coordinates = np.unravel_index(index, shape=img_prob.shape)
            x = int(coordinates[2])
            y = int(coordinates[1])
            z = int(coordinates[0])
            ox = int(coordinates[2])
            oy = int(coordinates[1])
            oz = int(coordinates[0])

            # Adjust the coordinates to be the origin of the crop and control to
            # not be out of the volume
            if z < int(random_crop_size[0]/2):
                z = 0
            elif z > vol.shape[0] - int(random_crop_size[0]/2):
                z = vol.shape[0] - random_crop_size[0]
            else:
                z -= int(random_crop_size[0]/2)

            if y < int(random_crop_size[1]/2):
                y = 0
            elif y > vol.shape[1] - int(random_crop_size[1]/2):
                y = vol.shape[1] - random_crop_size[1]
            else:
                y -= int(random_crop_size[1]/2)

            if x < int(random_crop_size[2]/2):
                x = 0
            elif x > vol.shape[2] - int(random_crop_size[2]/2):
                x = vol.shape[2] - random_crop_size[2]
            else:
                x -= int(random_crop_size[2]/2)
        else:
            ox = 0
            oy = 0
            oz = 0
            z = np.random.randint(0, deep - dz + 1)
            y = np.random.randint(0, cols - dy + 1)
            x = np.random.randint(0, rows - dx + 1)

    if draw_prob_map_points:
        return vol[z:(z+dz), y:(y+dy), x:(x+dx)], mask[z*scale:(z+dz)*scale, y*scale:(y+dy)*scale, x*scale:(x+dx)*scale],\
               oz, oy, ox, z, y, x
    else:
        if weight_map is not None:
            return vol[z:(z+dz), y:(y+dy), x:(x+dx)], mask[z*scale:(z+dz)*scale, y*scale:(y+dy)*scale, x*scale:(x+dx)*scale],\
                   weight_map[z:(z+dz), y:(y+dy), x:(x+dx)]
        else:
            return vol[z:(z+dz), y:(y+dy), x:(x+dx)], mask[z:(z+dz), y:(y+dy), x:(x+dx)]


def random_crop_single(image, random_crop_size, val=False, draw_prob_map_points=False, weight_map=None):
    """Random crop for a single image. No crop is done in those dimensions that ``random_crop_size`` is greater that
       the input image shape in those dimensions. For instance, if an input image is ``400x150`` and ``random_crop_size``
       is ``224x224`` the resulting image will be ``224x150``.

       Parameters
       ----------
       image : Numpy 3D array
           Image. E.g. ``(y, x, channels)``.

       random_crop_size : 2 int tuple
           Size of the crop. E.g. ``(y, x)``.

       val : bool, optional
           If the image provided is going to be used in the validation data. This forces to crop from the origin,
           e. g. ``(0, 0)`` point.

       draw_prob_map_points : bool, optional
           To return the pixel chosen to be the center of the crop.

       weight_map : bool, optional
           Weight map of the given image. E.g. ``(y, x, channels)``.

       Returns
       -------
       img : 2D Numpy array
           Crop of the given image. E.g. ``(y, x)``.

       weight_map : 2D Numpy array, optional
           Crop of the given image's weigth map. E.g. ``(y, x)``.

       oy : int, optional
           Y coordinate in the complete image of the chose central pixel to make the crop.

       ox : int, optional
           X coordinate in the complete image of the chose central pixel to make the crop.

       y : int, optional
           Y coordinate in the complete image where the crop starts.

       y : int, optional
           X coordinate in the complete image where the crop starts.
    """

    if weight_map is not None:
        img, we = image
    else:
        img = image

    height, width  = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    assert height >= dy
    assert width >= dx
    if val:
        x, y, z, ox, oy, oz = 0, 0, 0, 0, 0, 0
    else:
        oy, ox = 0, 0
        x = np.random.randint(0, width - dx + 1) if width - dx +1 > 0 else 0
        y = np.random.randint(0, height - dy + 1) if height - dy + 1 > 0 else 0

    if draw_prob_map_points == True:
        return img[y:(y+dy), x:(x+dx)], ox, oy, x, y
    else:
        if weight_map is not None:
            return img[y:(y+dy), x:(x+dx)], weight_map[y:(y+dy), x:(x+dx)]
        else:
            return img[y:(y+dy), x:(x+dx)]


def random_3D_crop_single(image, random_crop_size, val=False, draw_prob_map_points=False, weight_map=None):
    """Random crop for a single image. No crop is done in those dimensions that ``random_crop_size`` is greater that
       the input image shape in those dimensions. For instance, if an input image is ``50x400x150`` and ``random_crop_size``
       is ``30x224x224`` the resulting image will be ``30x224x150``.

       Parameters
       ----------
       image : Numpy 3D array
           Image. E.g. ``(z, y, x, channels)``.

       random_crop_size : 2 int tuple
           Size of the crop. E.g. ``(z, y, x)``.

       val : bool, optional
           If the image provided is going to be used in the validation data. This forces to crop from the origin,
           e. g. ``(0, 0)`` point.

       draw_prob_map_points : bool, optional
           To return the pixel chosen to be the center of the crop.

       weight_map : bool, optional
           Weight map of the given image. E.g. ``(z, y, x, channels)``.

       Returns
       -------
       img : 2D Numpy array
           Crop of the given image. E.g. ``(z, y, x)``.

       weight_map : 2D Numpy array, optional
           Crop of the given image's weigth map. E.g. ``(z, y, x)``.

       ox : int, optional
           Z coordinate in the complete image of the chose central pixel to make the crop.

       oy : int, optional
           Y coordinate in the complete image of the chose central pixel to make the crop.

       ox : int, optional
           X coordinate in the complete image of the chose central pixel to make the crop.

       z : int, optional
           Z coordinate in the complete image where the crop starts.

       y : int, optional
           Y coordinate in the complete image where the crop starts.

       x : int, optional
           X coordinate in the complete image where the crop starts.
    """

    if weight_map is not None:
        img, we = image
    else:
        img = image

    deep, cols, rows = img.shape[0], img.shape[1], img.shape[2]
    dz, dy, dx = random_crop_size
    assert rows >= dx
    assert cols >= dy
    assert deep >= dz
    if val:
        x, y, z, ox, oy, oz = 0, 0, 0, 0, 0, 0
    else:
        if img_prob is not None:
            prob = img_prob.ravel()

            # Generate the random coordinates based on the distribution
            choices = np.prod(img_prob.shape)
            index = np.random.choice(choices, size=1, p=prob)
            coordinates = np.unravel_index(index, shape=img_prob.shape)
            x = int(coordinates[2])
            y = int(coordinates[1])
            z = int(coordinates[0])
            ox = int(coordinates[2])
            oy = int(coordinates[1])
            oz = int(coordinates[0])

            # Adjust the coordinates to be the origin of the crop and control to
            # not be out of the volume
            if z < int(random_crop_size[0]/2):
                z = 0
            elif z > img.shape[0] - int(random_crop_size[0]/2):
                z = img.shape[0] - random_crop_size[0]
            else:
                z -= int(random_crop_size[0]/2)

            if y < int(random_crop_size[1]/2):
                y = 0
            elif y > img.shape[1] - int(random_crop_size[1]/2):
                y = img.shape[1] - random_crop_size[1]
            else:
                y -= int(random_crop_size[1]/2)

            if x < int(random_crop_size[2]/2):
                x = 0
            elif x > img.shape[2] - int(random_crop_size[2]/2):
                x = img.shape[2] - random_crop_size[2]
            else:
                x -= int(random_crop_size[2]/2)
        else:
            ox = 0
            oy = 0
            oz = 0
            z = np.random.randint(0, deep - dz + 1)
            y = np.random.randint(0, cols - dy + 1)
            x = np.random.randint(0, rows - dx + 1)

    if draw_prob_map_points:
        return img[z:(z+dz), y:(y+dy), x:(x+dx)], oz, oy, ox, z, y, x
    else:
        if weight_map is not None:
            return img[z:(z+dz), y:(y+dy), x:(x+dx)], weight_map[z:(z+dz), y:(y+dy), x:(x+dx)]
        else:
            return img[z:(z+dz), y:(y+dy), x:(x+dx)]



def center_crop_single(img, crop_shape):
    """Extract the central patch from a single image.

       Parameters
       ----------
       img : 3D/4D array
           Image. E.g. ``(y, x, channels)`` or ``(z, y, x, channels)``.

       crop_shape : 2/3 int tuple
           Size of the crop. E.g. ``(y, x)`` or ``(z, y, x)``.

       Returns
       -------
       img : 3D/4D Numpy array
           Center crop of the given image. E.g. ``(y, x, channels)`` or ``(z, y, x, channels)``.
    """
    if img.ndim == 4:
        z,y,x,c = img.shape
        startz = max(z//2 - crop_shape[0]//2, 0)
        starty = max(y//2 - crop_shape[1]//2, 0)
        startx = max(x//2 - crop_shape[2]//2, 0)  
        return img[startz:startz+crop_shape[0], starty:starty+crop_shape[1], startx:startx+crop_shape[2]]
    else:
        y,x,c = img.shape
        starty = max(y//2 - crop_shape[0]//2, 0)
        startx = max(x//2 - crop_shape[1]//2, 0)   
        return img[starty:starty+crop_shape[0], startx:startx+crop_shape[1]]

def resize_img(img, shape):
    """Resizes input image to given shape.

       Parameters
       ----------
       img : 3D/4D Numpy array
           Data to extract the patch from. E.g. ``(y, x, channels)`` for ``2D`` or  ``(z, y, x, channels)`` for ``3D``.

       crop_shape : 2D/3D int tuple
           Shape of the patches to create. E.g.  ``(y, x)`` for ``2D`` ``(z, y, x)`` for ``3D``.

       Returns
       -------
       img : 3D/4D Numpy array
           Resized image. E.g. ``(y, x, channels)`` for ``2D`` or  ``(z, y, x, channels)`` for ``3D``.
    """

    return resize(img, shape, order=1, mode='reflect', clip=True, preserve_range=True, anti_aliasing=True) 
