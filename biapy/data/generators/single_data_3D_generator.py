"""
3D single image data generator for BiaPy.

This module provides the Single3DImageDataGenerator class, which generates batches of
3D images with on-the-fly augmentation for deep learning workflows.
"""
import numpy as np
import random

from biapy.data.data_manipulation import save_tif
from biapy.data.generators.single_base_data_generator import SingleBaseDataGenerator


class Single3DImageDataGenerator(SingleBaseDataGenerator):
    """
    Custom 3D data generator to transform single image data.

    Parameters
    ----------
    zflip : bool, optional
        To activate flips in z dimension.
    """

    def __init__(self, zflip=False, **kwars):
        """
        Initialize the Single3DImageDataGenerator.

        Parameters
        ----------
        zflip : bool, optional
            Whether to apply flips in the z dimension.
        **kwars : dict
            Keyword arguments passed to the base SingleBaseDataGenerator.
        """
        super().__init__(**kwars)
        self.zflip = zflip

    def apply_transform(self, image):
        """
        Apply transformations to the input image, including optional z-flip.

        Parameters
        ----------
        image : 4D Numpy array
            Input image to transform. E.g. ``(z, y, x, channels)``.

        Returns
        -------
        image : 4D Numpy array
            Transformed image. E.g. ``(z, y, x, channels)``.
        """
        if self.zflip and random.uniform(0, 1) < self.da_prob:
            image = image[::-1, ...]

        return super().apply_transform(image)

    def save_aug_samples(self, img, orig_images, i, pos, out_dir, draw_grid):
        """
        Save transformed samples in order to check the generator.

        Parameters
        ----------
        img : 4D Numpy array
            Image to use as sample. E.g. ``(z, y, x, channels)``.

        orig_images : dict
            Dict where the original image is saved in "o_x".

        i : int
            Number of the sample within the transformed ones.

        pos : int
            Number of the sample within the dataset.

        out_dir : str
            Directory to save the images.

        draw_grid : bool
            Whether to draw a grid or not.
        """
        # Original image
        if draw_grid:
            self.draw_grid(orig_images["o_x"])
        aux = np.expand_dims(orig_images["o_x"], 0).astype(np.float32)
        save_tif(
            aux,
            out_dir,
            [str(i) + "_orig_x_" + str(pos) + "_" + self.trans_made + ".tif"],
            verbose=False,
        )
        # Transformed
        aux = np.expand_dims(img, 0).astype(np.float32)
        save_tif(
            aux,
            out_dir,
            [str(i) + "_x_aug_" + str(pos) + "_" + self.trans_made + ".tif"],
            verbose=False,
        )
