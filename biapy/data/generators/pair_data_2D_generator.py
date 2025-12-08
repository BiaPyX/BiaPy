"""
2D paired image and mask data generator for BiaPy.

This module provides the Pair2DImageDataGenerator class, which generates batches of
2D images and their corresponding masks with on-the-fly augmentation.
"""
import numpy as np
import os
from PIL import Image
from typing import Dict
from numpy.typing import NDArray

from biapy.data.data_manipulation import save_tif
from biapy.data.generators.pair_base_data_generator import PairBaseDataGenerator


class Pair2DImageDataGenerator(PairBaseDataGenerator):
    """
    Custom 2D data generator to transform paired image and mask data.
    """

    def __init__(self, **kwars):
        """
        Initialize the Pair2DImageDataGenerator.

        Parameters
        ----------
        **kwars : dict
            Keyword arguments passed to the base PairBaseDataGenerator.
        """
        super().__init__(**kwars)

    def save_aug_samples(
        self,
        img: NDArray,
        mask: NDArray,
        orig_images: Dict,
        i: int,
        pos: int,
        out_dir: str,
    ):
        """
        Save transformed samples in order to check the generator.

        Parameters
        ----------
        img : 3D Numpy array
            Image to use as sample. E.g. ``(y, x, channels)``.

        mask : 3D Numpy array
            Mask to use as sample. E.g. ``(y, x, channels)``.

        orig_images: dict
            Dict where the original image and mask are saved in "o_x" and "o_y", respectively.

        i: int
            Number of the sample within the transformed ones.

        pos: int
            Number of the sample within the dataset.

        out_dir: str
            Directory to save the images.
        """
        aux = np.expand_dims(orig_images["o_x"], 0).astype(np.float32)
        save_tif(
            aux,
            out_dir,
            [str(i) + "_" + str(pos) + "_orig_x" + self.trans_made + ".tif"],
            verbose=False,
        )

        aux = np.expand_dims(orig_images["o_y"], 0).astype(np.float32)
        save_tif(
            aux,
            out_dir,
            [str(i) + "_" + str(pos) + "_orig_y" + self.trans_made + ".tif"],
            verbose=False,
        )

        # Save transformed images/masks
        aux = np.expand_dims(img, 0).astype(np.float32)
        save_tif(
            aux,
            out_dir,
            [str(i) + "_" + str(pos) + "_x" + self.trans_made + ".tif"],
            verbose=False,
        )
        aux = np.expand_dims(mask, 0).astype(np.float32)
        save_tif(
            aux,
            out_dir,
            [str(i) + "_" + str(pos) + "_y" + self.trans_made + ".tif"],
            verbose=False,
        )