"""
3D paired image and mask data generator for BiaPy.

This module provides the Pair3DImageDataGenerator class, which generates batches of
3D images and their corresponding masks with on-the-fly augmentation.
"""
import numpy as np
import random
from PIL import Image
from typing import Tuple, Optional
from numpy.typing import NDArray

from biapy.data.data_manipulation import save_tif
from biapy.data.generators.pair_base_data_generator import PairBaseDataGenerator


class Pair3DImageDataGenerator(PairBaseDataGenerator):
    """
    Custom 3D data generator. This generator will yield an image and its corresponding mask.

    Parameters
    ----------
    zflip : bool, optional
        To activate flips in z dimension.
    """

    def __init__(self, zflip: bool = False, **kwars):
        """
        Initialize the Pair3DImageDataGenerator.

        Parameters
        ----------
        zflip : bool, optional
            Whether to apply flips in the z dimension.
        **kwars : dict
            Keyword arguments passed to the base PairBaseDataGenerator.
        """
        super().__init__(**kwars)
        sshape = self.X.sample_list[0].get_shape()
        if sshape is None:
            sshape = self.shape
        self.z_size = sshape[0]
        self.zflip = zflip
        self.grid_d_size = (
            self.shape[1] * self.grid_d_range[0],
            self.shape[2] * self.grid_d_range[1],
            self.shape[0] * self.grid_d_range[0],
            self.shape[0] * self.grid_d_range[1],
        )

    def apply_transform(
        self,
        image: NDArray,
        mask: NDArray,
        e_im: Optional[NDArray] = None,
        e_mask: Optional[NDArray] = None,
    ) -> Tuple[NDArray, NDArray]:
        """
        Transform the input image and its mask at the same time with one of the selected choices based on a probability.

        Parameters
        ----------
        image : 4D Numpy array
            Image to transform. E.g. ``(z, y, x, channels)``.

        mask : 4D Numpy array
            Mask to transform. E.g.  ``(z, y, x, channels)``.

        e_im : 4D Numpy array
            Extra image to help transforming ``image``. E.g. ``(z, y, x, channels)``.

        e_mask : 4D Numpy array
            Extra mask to help transforming ``mask``. E.g. ``(z, y, x, channels)``.

        Returns
        -------
        image : 4D Numpy array
            Transformed image. E.g. ``(z, y, x, channels)```.

        mask : 4D Numpy array
            Transformed image mask. E.g.``(z, y, x, channels)``.
        """
        # Apply flips in z
        if self.zflip and random.uniform(0, 1) < self.da_prob:
            image = image[::-1, ...]
            mask  = mask[::-1, ...]
            if e_im is not None:
                e_im  = e_im[::-1, ...]
            if e_mask is not None:
                e_mask = e_mask[::-1, ...]

        return super().apply_transform(image, mask, e_im, e_mask)

    def save_aug_samples(self, img, mask, orig_images, i, pos, out_dir):
        """
        Save augmented and original samples for inspection.

        Parameters
        ----------
        img : 4D Numpy array
            Augmented image sample. E.g. ``(z, y, x, channels)``.
        mask : 4D Numpy array
            Augmented mask sample. E.g. ``(z, y, x, channels)``.
        orig_images : dict
            Dictionary containing original image and mask under keys "o_x" and "o_y".
        i : int
            Index of the augmented sample.
        pos : int
            Index of the sample in the dataset.
        out_dir : str
            Directory to save the images.
        """
        aux = np.expand_dims(orig_images["o_x"], 0).astype(np.float32)
        save_tif(
            aux,
            out_dir,
            [str(i) + "_orig_x_" + str(pos) + "_" + self.trans_made + ".tif"],
            verbose=False,
        )

        aux = np.expand_dims(orig_images["o_y"], 0).astype(np.float32)
        save_tif(
            aux,
            out_dir,
            [str(i) + "_orig_y_" + str(pos) + "_" + self.trans_made + ".tif"],
            verbose=False,
        )

        # Save transformed images/masks
        aux = np.expand_dims(img, 0).astype(np.float32)
        save_tif(
            aux,
            out_dir,
            [str(i) + "_x_aug_" + str(pos) + "_" + self.trans_made + ".tif"],
            verbose=False,
        )
        aux = np.expand_dims(mask, 0).astype(np.float32)
        save_tif(
            aux,
            out_dir,
            [str(i) + "_y_aug_" + str(pos) + "_" + self.trans_made + ".tif"],
            verbose=False,
        )