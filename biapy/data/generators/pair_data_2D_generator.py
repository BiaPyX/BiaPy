import numpy as np
import os
from PIL import Image
from typing import Dict

from biapy.data.data_manipulation import save_tif
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
