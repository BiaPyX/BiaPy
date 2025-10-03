"""
2D single image data generator for BiaPy.

This module provides the Single2DImageDataGenerator class, which generates batches of
2D images with on-the-fly augmentation for deep learning workflows.
"""
import numpy as np

from biapy.data.data_manipulation import save_tif
from biapy.data.generators.single_base_data_generator import SingleBaseDataGenerator


class Single2DImageDataGenerator(SingleBaseDataGenerator):
    """
    Custom 2D data generator to transform single image data.
    """

    def __init__(self, **kwars):
        """
        Initialize the Single2DImageDataGenerator.

        Parameters
        ----------
        **kwars : dict
            Keyword arguments passed to the base SingleBaseDataGenerator.
        """
        super().__init__(**kwars)

    def save_aug_samples(self, img, orig_image, i, pos, out_dir, draw_grid):
        """
        Save transformed samples in order to check the generator.

        Parameters
        ----------
        img : 3D Numpy array
            Image to use as sample. E.g. ``(y, x, channels)```.

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
        if draw_grid:
            self.draw_grid(orig_image["o_x"])
        aux = np.expand_dims(orig_image["o_x"], 0).astype(np.float32)
        save_tif(
            aux,
            out_dir,
            [str(i) + "_" + str(pos) + "_orig_x" + self.trans_made + ".tif"],
            verbose=False,
        )
        # Save transformed images
        aux = np.expand_dims(img, 0).astype(np.float32)
        save_tif(
            aux,
            out_dir,
            [str(i) + "_" + str(pos) + "_x" + self.trans_made + ".tif"],
            verbose=False,
        )
