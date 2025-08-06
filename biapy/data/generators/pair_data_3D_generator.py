"""
3D paired image and mask data generator for BiaPy.

This module provides the Pair3DImageDataGenerator class, which generates batches of
3D images and their corresponding masks with on-the-fly augmentation. It is based on
imgaug, microDL, and custom augmentors for flexible deep learning workflows.
"""
import numpy as np
import random
from PIL import Image
from typing import Tuple, Optional
from numpy.typing import NDArray

from biapy.data.data_manipulation import save_tif
from biapy.data.generators.pair_base_data_generator import PairBaseDataGenerator


class Pair3DImageDataGenerator(PairBaseDataGenerator):
    """Custom 3D data generator based on `imgaug <https://github.com/aleju/imgaug-doc>`_ and our own `augmentors.py <https://github.com/BiaPyX/BiaPy/blob/master/biapy/data/generators/augmentors.py>`_ transformations. This generator will yield an image and its corresponding mask.

    Based on `microDL <https://github.com/czbiohub/microDL>`_ and
    `Shervine's blog <https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly>`_.

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
            Image to transform. E.g. ``(y, x, z, channels)``.

        mask : 4D Numpy array
            Mask to transform. E.g.  ``(y, x, z, channels)``.

        e_im : 4D Numpy array
            Extra image to help transforming ``image``. E.g. ``(y, x, z, channels)``.

        e_mask : 4D Numpy array
            Extra mask to help transforming ``mask``. E.g. ``(y, x, z, channels)``.

        Returns
        -------
        image : 4D Numpy array
            Transformed image. E.g. ``(y, x, z, channels)```.

        mask : 4D Numpy array
            Transformed image mask. E.g.``(y, x, z, channels)``.
        """
        # Transpose them so we can merge the z and c channels easily.
        # z, y, x, c --> x, y, z, c
        image = image.transpose((2, 1, 0, 3))
        mask = mask.transpose((2, 1, 0, 3))

        if e_im is not None and e_mask is not None:
            # Transpose the extra image and mask so we can merge the z and c channels easily.
            # z, y, x, c --> x, y, z, c
            e_im = e_im.transpose((2, 1, 0, 3))
            e_mask = e_mask.transpose((2, 1, 0, 3))
        
        # Apply flips in z as imgaug can not do it
        if self.zflip and random.uniform(0, 1) < self.da_prob:
            l_image = []
            l_mask = []
            for i in range(image.shape[-1]):
                l_image.append(np.expand_dims(np.flip(image[..., i], 2), -1))
            for i in range(mask.shape[-1]):
                l_mask.append(np.expand_dims(np.flip(mask[..., i], 2), -1))
            image = np.concatenate(l_image, axis=-1)
            mask = np.concatenate(l_mask, axis=-1)

        image, mask = super().apply_transform(image, mask, e_im, e_mask)

        # x, y, z, c --> z, y, x, c
        return image.transpose((2, 1, 0, 3)), mask.transpose((2, 1, 0, 3))

    def save_aug_samples(self, img, mask, orig_images, i, pos, out_dir, point_dict):
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
        point_dict : dict
            Information about the crop and selected point for visualization.
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
        del img, mask

        # Save the original images with a red point and a blue square that represents the point selected with
        # the probability map and the random volume extracted from the original data
        if self.random_crops_in_DA and self.prob_map is not None and i == 0:
            aux, auxm = self.load_sample(pos)
            if aux.max() < 1:
                aux *= 255
            if auxm.max() == 1:
                auxm *= 255
            aux, auxm = (aux).astype(np.uint8), auxm.astype(np.uint8)

            print(
                "The selected point of the random crop was [{},{},{}]".format(
                    point_dict["oz"], point_dict["oy"], point_dict["ox"]
                )
            )

            if aux.shape[-1] == 1:
                aux = np.repeat(aux, 3, axis=3)
            if auxm.shape[-1] == 1:
                auxm = np.repeat(auxm, 3, axis=3)

            for s in range(aux.shape[0]):
                if s >= point_dict["s_z"] and s < point_dict["s_z"] + self.shape[0]:
                    im = Image.fromarray(aux[s, ..., 0])
                    im = im.convert("RGB")
                    px = im.load()
                    assert px is not None

                    m = Image.fromarray(auxm[s, ..., 0])
                    m = m.convert("RGB")
                    py = m.load()
                    assert py is not None

                    # Paint a blue square that represents the crop made.
                    # Here the axis are x, y and not y, x (numpy)
                    for row in range(point_dict["s_x"], point_dict["s_x"] + self.shape[2]):
                        px[row, point_dict["s_y"]] = (0, 0, 255)
                        px[row, point_dict["s_y"] + self.shape[1] - 1] = (0, 0, 255)
                        py[row, point_dict["s_y"]] = (0, 0, 255)
                        py[row, point_dict["s_y"] + self.shape[1] - 1] = (0, 0, 255)
                    for col in range(point_dict["s_y"], point_dict["s_y"] + self.shape[1]):
                        px[point_dict["s_x"], col] = (0, 0, 255)
                        px[point_dict["s_x"] + self.shape[2] - 1, col] = (0, 0, 255)
                        py[point_dict["s_x"], col] = (0, 0, 255)
                        py[point_dict["s_x"] + self.shape[2] - 1, col] = (0, 0, 255)

                    # Paint the selected point in red
                    if s == point_dict["oz"]:
                        p_size = 6
                        for row in range(point_dict["ox"] - p_size, point_dict["ox"] + p_size):
                            for col in range(point_dict["oy"] - p_size, point_dict["oy"] + p_size):
                                if col >= 0 and col < aux.shape[1] and row >= 0 and row < aux.shape[2]:
                                    px[row, col] = (255, 0, 0)
                                    py[row, col] = (255, 0, 0)

                    aux[s] = im
                    auxm[s] = m

            aux = np.expand_dims(aux, 0).astype(np.float32)
            save_tif(
                aux,
                out_dir,
                [str(i) + "_" + str(pos) + "_mark_x" + self.trans_made + ".tif"],
                verbose=False,
            )

            auxm = np.expand_dims(auxm, 0).astype(np.float32)
            save_tif(
                auxm,
                out_dir,
                [str(i) + "_" + str(pos) + "_mark_y" + self.trans_made + ".tif"],
                verbose=False,
            )
