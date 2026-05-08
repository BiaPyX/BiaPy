import argparse
import os
import sys
import numpy as np
from tqdm import tqdm
import h5py
import ast
from skimage.transform import resize
from skimage.measure import label
from scipy import ndimage
import torch 
from pathlib import Path
import pandas as pd

sys.path.insert(0, '/net/fibserver1/data/raw/scratch/dfranco/BiaPy')  # Adjust this path as needed
from biapy.data.data_manipulation import save_tif, pad_and_reflect
from biapy.data.data_3D_manipulation import read_chunked_nested_data

parser = argparse.ArgumentParser(
    description="Extract synapse connection patches centered around the given post synaptic points",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument("-raw_input_data", "--raw_input_data", required=True, help="Directory to the folder where the raw data reside")
parser.add_argument("-F_post_pred_post_location", "--F_post_pred_post_location", required=True, help="Directory to the folder where the predicted post synaptic points are stored. The script will find '_pred_post_locations.csv' files in this folder, which should have columns 'x', 'y', 'z' for the coordinates of the predicted post synaptic points.")
parser.add_argument("-output_data", "--output_data", required=True, help="Directory to the folder where the new data will be saved")
parser.add_argument("-patch_size", "--patch_size", type=int, nargs=3, default=(8,96,96), help="Size of the extracted patches (z, y, x)")

parser.add_argument("-raw_data_in_data", "--raw_data_in_data", default="volumes.raw", type=str,
                    help="Raw data inside each h5 datafile, e.g. 'volumes.raw' in CREMI format")
args = vars(parser.parse_args())

input_data_folder = args['raw_input_data']
F_post_pred_post_folder = args['F_post_pred_post_location']
output_data_folder = args['output_data']

print(f"Processing {input_data_folder} folder . . .")
raw_file_ids = sorted(next(os.walk(input_data_folder))[2])
raw_file_ids = [f for f in raw_file_ids if f.endswith('.h5') or f.endswith('.hdf5') or f.endswith('.hdf')]

# Read images
for n, id_ in tqdm(enumerate(raw_file_ids), total=len(raw_file_ids)):
    name = os.path.splitext(id_)[0]
    filename = os.path.join(input_data_folder, id_)

    # Load raw volume (chunked read via BiaPy helper)
    file, raw_data = read_chunked_nested_data(filename, args['raw_data_in_data'])
    raw_data = np.array(raw_data)
    data_shape = raw_data.shape
    if isinstance(file, h5py.File):
        file.close()

    # Put all images within same range
    if raw_data.dtype != np.uint8:
        raw_data = (((raw_data - raw_data.min()) / (raw_data.max() - raw_data.min() + 1e-6)) * 255).astype(np.uint8)

    # Find the corresponding label file
    post_locations_filename = os.path.join(F_post_pred_post_folder, name+"_pred_post_locations.csv")
    if not os.path.exists(post_locations_filename):
        raise ValueError(f"F_post predicted post point file {post_locations_filename} does not exist. Please check your input folders and naming conventions.")

    print(f"Raw file: {filename} ; post locations file: {post_locations_filename}")

    post_point_df = pd.read_csv(post_locations_filename, index_col=False)
    shape_zyx = tuple(int(x) for x in data_shape)
    for idx, row in post_point_df.iterrows():
        x, y, z = int(row['axis-2']), int(row['axis-1']), int(row['axis-0'])
        tag = f"post_{idx}"
        mid_point = (z,y,x)
         
        pad_type = ["even", "even", "even"]
        z_min = mid_point[0] - args['patch_size'][0]//2
        if z_min <= 0:
            z_min = 0
            pad_type[0] = "left"

        z_max = mid_point[0] + args['patch_size'][0]//2
        if z_max >= shape_zyx[0]:
            z_max = shape_zyx[0]-1
            if pad_type[0] != "even":
                pad_type[0] = "even"
            else:
                pad_type[0] = "right"

        y_min = mid_point[1] - args['patch_size'][1]//2
        if y_min <= 0:
            y_min = 0
            pad_type[1] = "left"

        y_max = mid_point[1] + args['patch_size'][1]//2
        if y_max >= shape_zyx[1]:
            y_max = shape_zyx[1]-1
            if pad_type[1] != "even":
                pad_type[1] = "even"
            else:
                pad_type[1] = "right"
        
        x_min = mid_point[2] - args['patch_size'][2]//2
        if x_min <= 0:
            x_min = 0
            pad_type[2] = "left"

        x_max = mid_point[2] + args['patch_size'][2]//2
        if x_max >= shape_zyx[2]:
            x_max = shape_zyx[2]-1
            if pad_type[2] != "even":
                pad_type[2] = "even"
            else:
                pad_type[2] = "right"
                
        raw_patch = raw_data[z_min:z_max, y_min:y_max, x_min:x_max]
        raw_patch = np.expand_dims(raw_patch, axis=-1)
        # raw_patch = pad_and_reflect(raw_patch, args['patch_size'] + (raw_patch.shape[-1],), pad_type=pad_type, verbose=False)

        save_tif(np.expand_dims(raw_patch,0), output_data_folder, [f"{tag}_cube.tif"], verbose=False)

print("Done!")