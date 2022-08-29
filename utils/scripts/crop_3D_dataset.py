import os
import sys
import numpy as np
from skimage.io import imread
from tqdm import tqdm

code_dir = "/home/user/BiaPy"
input_dir = "/home/user/datasets/x"
output_dir = "/home/user/datasets/x_out"
crop_shape = (64, 64, 64, 1)
overlap = (0, 0, 0)
padding = (0, 0, 0)
median_padding = False

sys.path.insert(0, code_dir)
from data.data_3D_manipulation import crop_3D_data_with_overlap
from utils.util import save_tif

ids = sorted(next(os.walk(input_dir))[2])
for n, id_ in tqdm(enumerate(ids), total=len(ids)):
    img = imread(os.path.join(input_dir, id_))

    if len(img.shape) == 3:
        img = np.expand_dims(img, axis=-1)

    if img.shape != crop_shape[:3]+(img.shape[-1],):
        img = crop_3D_data_with_overlap(img, crop_shape[:3]+(img.shape[-1],), overlap=overlap, padding=padding,
                                        median_padding=median_padding, verbose=True)
    else:
        img = np.expand_dims(img, axis=0)

    filenames = []
    d = len(str(len(img)))
    for k in range(img.shape[0]):
        filenames.append(os.path.splitext(id_)[0]+"_crop"+str(k).zfill(d)+'.tif')
    save_tif(img, output_dir, filenames)

print("Finished!")
