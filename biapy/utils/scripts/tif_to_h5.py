import os
import h5py
import numpy as np
from skimage.io import imread
from tqdm import tqdm
from skimage import measure, feature
from scipy import ndimage
from PIL import ImageEnhance, Image
from os import path

img_shape = (4096, 4096)
input_dir = "/home/user/input"
output_dir = "/home/user/output"
h5_filename = '0_human_instance_seg_pred.h5'
#h5_filename = '1_rat_instance_seg_pred.h5'

input_ids = sorted(next(os.walk(input_dir))[2])
h5file_name = os.path.join(output_dir, h5_filename)
os.makedirs(output_dir, exist_ok=True)

# Allocate memory for the predictions
pred_stack = np.zeros((len(input_ids),) + img_shape, dtype=np.int64)

# Read all the images
for n, id_ in tqdm(enumerate(input_ids), total=len(input_ids)):
    img = imread(os.path.join(input_dir, id_))
    pred_stack[n] = img

# Create the h5 file (using lzf compression to save space)
h5f = h5py.File(h5file_name, 'w')
h5f.create_dataset('main', data=pred_stack, compression="lzf")
h5f.close()

