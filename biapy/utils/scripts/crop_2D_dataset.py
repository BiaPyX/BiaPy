import os
import sys
import numpy as np
from skimage.io import imread
from tqdm import tqdm

code_dir = "/home/user/BiaPy"
input_dir = "/home/user/input"
out_dir = "/home/user/out"
output_prefix = ""
crop_shape = (256, 256, 1)
overlap = (0, 0)
padding = (0, 0)
verbose = True

input_dir_x = os.path.join(input_dir, 'x')
input_dir_y = os.path.join(input_dir, 'y')

sys.path.insert(0, code_dir)
from biapy.data.data_2D_manipulation import crop_data_with_overlap
from biapy.utils.util import save_tif_pair_discard

ids = sorted(next(os.walk(input_dir_x))[2])
for n, id_ in tqdm(enumerate(ids), total=len(ids)):
    if id_.endswith('.npy'):
        img = np.load(os.path.join(input_dir_x, id_))
        mask = np.load(os.path.join(input_dir_y, id_))
    else:
        img = imread(os.path.join(input_dir_x, id_))
        mask = imread(os.path.join(input_dir_y, id_))
    img = np.squeeze(img)
    mask = np.squeeze(mask)

    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, axis=-1)

    if len(img.shape) == 3:
        img = np.expand_dims(img, axis=0)
    if len(mask.shape) == 3:
        mask = np.expand_dims(mask, axis=0)

    if img.shape != crop_shape[:2]+(img.shape[-1],):
        img, mask = crop_data_with_overlap(img, crop_shape[:2]+(img.shape[-1],), data_mask=mask, overlap=overlap,
                                              padding=padding, verbose=verbose)
    else:
        img = np.expand_dims(img, axis=0)
        mask = np.expand_dims(mask, axis=0)

    filenames = []
    d = len(str(len(img)))
    for k in range(img.shape[0]):
        filenames.append(os.path.splitext(id_)[0]+"_crop"+str(k).zfill(d)+'.tif')
    save_tif_pair_discard(img, mask, out_dir, output_prefix, filenames)

print("Finished!")


