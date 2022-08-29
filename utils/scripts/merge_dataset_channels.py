import os
import sys
import numpy as np
from skimage.io import imread
from tqdm import tqdm

code_dir = "/data/BiaPy"
input_dir = "/data/datasets/channel1"
input2_dir = "/data/datasets/channel2"
output_dir = "/data/datasets/channel_merge"
crop_shape = (64, 64, 64, 1)
overlap = (0, 0, 0)
padding = (0, 0, 0)
median_padding = False

sys.path.insert(0, code_dir)
from utils.util import save_tif
ids = sorted(next(os.walk(input_dir))[2])
ids2 = sorted(next(os.walk(input2_dir))[2])
for n, id_ in tqdm(enumerate(ids), total=len(ids)):
    img1 = imread(os.path.join(input_dir, id_))
    img2 = imread(os.path.join(input2_dir, ids2[n]))

    if len(img1.shape) == 3:
        img1 = np.expand_dims(img1, axis=-1)
    if len(img2.shape) == 3:
        img2 = np.expand_dims(img2, axis=-1)

    img = np.concatenate((img1, img2), axis=-1)

    filenames = [id_]
    save_tif(img, output_dir, filenames)

print("Finished!")
