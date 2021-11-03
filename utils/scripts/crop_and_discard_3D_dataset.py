import os
import sys
import numpy as np
from skimage.io import imread
from tqdm import tqdm

code_dir = "/home/user/EM_Image_Segmentation"
input_dir = "/home/user/dataset/train"
input_dir_x = os.path.join(input_dir, 'x')
input_dir_y = os.path.join(input_dir, 'y')
output_prefix = "_cropped_discard"
crop_shape = (80, 80, 80, 1)
overlap = (0, 0, 0)
padding = (0, 0, 0)
median_padding = False

sys.path.insert(0, code_dir)
from data.data_3D_manipulation import crop_3D_data_with_overlap
from utils.util import save_tif_pair_discard, pad_and_reflect

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

    if len(img.shape) == 3:
        img = np.expand_dims(img, axis=-1)
    if len(mask.shape) == 3:
        mask = np.expand_dims(mask, axis=-1)

    img = pad_and_reflect(img, crop_shape, True)
    mask = pad_and_reflect(mask, crop_shape, True)

    if img.shape != crop_shape[:3]+(img.shape[-1],):
        img, mask = crop_3D_data_with_overlap(img, crop_shape[:3]+(img.shape[-1],), data_mask=mask, overlap=overlap,
                                              padding=padding, median_padding=median_padding, verbose=True)
    else:
        img = np.expand_dims(img, axis=0)
        mask = np.expand_dims(mask, axis=0)

    img = np.transpose(img, (0,3,1,2,4))
    mask = np.transpose(mask, (0,3,1,2,4))

    filenames = []
    d = len(str(len(img)))
    for k in range(img.shape[0]):
        filenames.append(os.path.splitext(id_)[0]+"_crop"+str(k).zfill(d)+'.tif')
    save_tif_pair_discard(img, mask, input_dir, output_prefix, filenames)

print("Finished!")
