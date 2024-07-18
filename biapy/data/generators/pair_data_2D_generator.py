import numpy as np
import os
from PIL import Image
from typing import Union, Tuple, Dict

from biapy.utils.util import save_tif
from biapy.data.generators.pair_base_data_generator import PairBaseDataGenerator


class Pair2DImageDataGenerator(PairBaseDataGenerator):
    """Custom 2D data generator based on `imgaug <https://github.com/aleju/imgaug-doc>`_
    and our own `augmentors.py <https://github.com/BiaPyX/BiaPy/blob/master/biapy/data/generators/augmentors.py>`_
    transformations. This generator will yield an image and its corresponding mask.

    Based on `microDL <https://github.com/czbiohub/microDL>`_ and
    `Shervine's blog <https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly>`_.
    """

    def __init__(self, **kwars):
        super().__init__(**kwars)

    def ensure_shape(
        self, img: np.ndarray, mask: np.ndarray | None
    ) -> Union[Tuple[np.ndarray, Union[np.ndarray, None]], np.ndarray]:
        """
        Ensures ``img`` and ``mask`` correct axis number and their order.

        Parameters
        ----------
        img : Numpy array representing a ``2D`` image
            Image to use as sample.

        mask : Numpy array representing a ``2D`` image
            Mask to use as sample.

        Returns
        -------
        img : 3D Numpy array
            Image to use as sample. E.g. ``(y, x, channels)``.

        mask : 3D Numpy array
            Mask to use as sample. E.g. ``(y, x, channels)``.

        """
        # Shape adjustment
        if img.ndim == 2:
            img = np.expand_dims(img, -1)
        else:
            if img.shape[0] <= 3:
                img = img.transpose((1, 2, 0))
        if self.Y_provided and mask is not None:
            if mask.ndim == 2:
                mask = np.expand_dims(mask, -1)
            else:
                if mask.shape[0] <= 3:
                    mask = mask.transpose((1, 2, 0))

        if img.ndim != 3:
            raise ValueError(f"Image loaded seems to not be 2D: {img.shape}")

        # Super-resolution check. if random_crops_in_DA is activated the images have not been cropped yet,
        # so this check can not be done and it will be done in the random crop
        if not self.random_crops_in_DA and self.Y_provided and any([x != 1 for x in self.random_crop_scale]):
            s = [
                img.shape[0] * self.random_crop_scale[0],
                img.shape[1] * self.random_crop_scale[1],
            ]
            if mask is not None and all(x != y for x, y in zip(s, mask.shape[:-1])):
                raise ValueError(
                    "Images loaded need to be LR and its HR version. LR shape:"
                    " {} vs HR shape {} is not x{} larger".format(
                        img.shape[:-1], mask.shape[:-1], self.random_crop_scale
                    )
                )

        if self.convert_to_rgb and self.shape[-1] == 3 and img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)

        if self.Y_provided:
            return img, mask
        else:
            return img

    def save_aug_samples(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        orig_images: Dict,
        i: int,
        pos: int,
        out_dir: str,
        point_dict: Dict,
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

        point_dict: Dict
            Necessary info to draw the patch extracted within the original image. It has ``ox`` and
            ``oy`` representing the ``x`` and ``y`` coordinates of the central point selected during
            the crop extraction, and ``s_x`` and ``s_y`` as the ``(0,0)`` coordinates of the extracted
            patch.
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

        # Save the original images with a point that represents the selected coordinates to be the center of
        # the crop
        if self.random_crops_in_DA and self.prob_map is not None:
            img, mask = self.load_sample(pos)
            if img.max() < 1:
                img *= 255
            if mask.max() == 1:
                mask *= 255
            img, mask = (img).astype(np.uint8), mask.astype(np.uint8)

            if self.shape[-1] == 1:
                im = Image.fromarray(np.repeat(img, 3, axis=2), "RGB")
            else:
                im = Image.fromarray(img, "RGB")
            px = im.load()
            assert px is not None

            # Paint the selected point in red
            p_size = 6
            for col in range(point_dict["oy"] - p_size, point_dict["oy"] + p_size):
                for row in range(point_dict["ox"] - p_size, point_dict["ox"] + p_size):
                    if col >= 0 and col < img.shape[0] and row >= 0 and row < img.shape[1]:
                        px[row, col] = (255, 0, 0)

            # Paint a blue square that represents the crop made
            for row in range(point_dict["s_x"], point_dict["s_x"] + self.shape[0]):
                px[row, point_dict["s_y"]] = (0, 0, 255)
                px[row, point_dict["s_y"] + self.shape[0] - 1] = (0, 0, 255)
            for col in range(point_dict["s_y"], point_dict["s_y"] + self.shape[0]):
                px[point_dict["s_x"], col] = (0, 0, 255)
                px[point_dict["s_x"] + self.shape[0] - 1, col] = (0, 0, 255)

            im.save(
                os.path.join(
                    out_dir,
                    str(i) + "_" + str(pos) + "_mark_x" + self.trans_made + ".tif",
                )
            )

            if mask.shape[-1] == 1:
                m = Image.fromarray(np.repeat(mask, 3, axis=2), "RGB")
            else:
                m = Image.fromarray(mask, "RGB")
            px = m.load()
            assert px is not None

            # Paint the selected point in red
            for col in range(point_dict["oy"] - p_size, point_dict["oy"] + p_size):
                for row in range(point_dict["ox"] - p_size, point_dict["ox"] + p_size):
                    if col >= 0 and col < mask.shape[0] and row >= 0 and row < mask.shape[1]:
                        px[row, col] = (255, 0, 0)

            # Paint a blue square that represents the crop made
            for row in range(point_dict["s_x"], point_dict["s_x"] + self.shape[0]):
                px[row, point_dict["s_y"]] = (0, 0, 255)
                px[row, point_dict["s_y"] + self.shape[0] - 1] = (0, 0, 255)
            for col in range(point_dict["s_y"], point_dict["s_y"] + self.shape[0]):
                px[point_dict["s_x"], col] = (0, 0, 255)
                px[point_dict["s_x"] + self.shape[0] - 1, col] = (0, 0, 255)

            m.save(
                os.path.join(
                    out_dir,
                    str(i) + "_" + str(pos) + "_mark_y" + self.trans_made + ".tif",
                )
            )
