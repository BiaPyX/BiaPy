import zarr
import os
from skimage.io import imread
import numpy as np
from tqdm import tqdm 

img_shape = (1024, 1024)
input_x_dir = "/home/user/data/x"
input_y_dir = "/home/user/data/y"
output_dir = '/home/user/data/zarr'
outfilename = "training.zarr"
dsx = 'volumes/raw' 
dsy = 'volumes/labels/neuron_ids' 
resolution = (8,8,8)
offset = (0,0,0)

input_x_ids = sorted(next(os.walk(input_x_dir))[2])
input_y_ids = sorted(next(os.walk(input_y_dir))[2])

# Allocate memory for the predictions
pred_stack, pred_mask_stack = [], []

# Read all the images/labels and create their stacks
for n, id_ in tqdm(enumerate(input_x_ids), total=len(input_x_ids)):
    img = imread(os.path.join(input_x_dir, id_))
    pred_stack.append(np.expand_dims(img,0))

    mask = imread(os.path.join(input_y_dir, input_y_ids[n]))
    pred_mask_stack.append(np.expand_dims(mask,0))

pred_stack = np.concatenate(pred_stack).astype(img.dtype)
pred_mask_stack = np.concatenate(pred_mask_stack).astype(mask.dtype)

# Write Zarr 
os.makedirs(output_dir, exist_ok=True)
o = zarr.open(os.path.join(output_dir, outfilename), mode='w')
o[dsx] = pred_stack
o[dsx].attrs['resolution'] = resolution
o[dsx].attrs['offset'] = offset

o[dsy] = pred_mask_stack
o[dsy].attrs['resolution'] = resolution
o[dsy].attrs['offset'] = offset
