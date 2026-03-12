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

from biapy.data.data_manipulation import save_tif, read_img_as_ndarray, pad_and_reflect
from biapy.data.data_3D_manipulation import read_chunked_nested_data

def push_pre_point_away(pre_point, post_points, distance):
    """
    Pushes the pre_point further away from the consensus center 
    of the post_points cluster.
    """
    pre_point = np.array(pre_point)
    post_points = np.array(post_points)
    
    # 1. Find the "center" of the post cluster (the trident head)
    post_centroid = np.mean(post_points, axis=0)
    
    # 2. Calculate the vector pointing FROM the cluster TO the pre_point
    # This represents the "outward" direction
    outward_direction = pre_point - post_centroid
    
    # 3. Normalize the vector to unit length
    norm = np.linalg.norm(outward_direction)
    if norm == 0:
        return pre_point  # Safety check if points are identical
        
    unit_vector = outward_direction / norm
    
    # 4. Push the pre_point further along that outward path
    # New Point = original pre_point + (direction * distance)
    return pre_point + (unit_vector * distance)

def _in_bounds(p: np.ndarray, shape_zyx: tuple) -> bool:
    """
    Check if a point p (z,y,x) is within the bounds of a shape (Z,Y,X). 

    Parameters
    ----------
    p : np.ndarray 
        A point in (z,y,x) coordinates. 
    shape_zyx : Tuple[int, int, int] 
        The shape of the volume in (Z,Y,X). 
    
    Returns
    -------
    bool
        True if p is within bounds, False otherwise.
    """
    # p is (3,) z,y,x
    return bool(np.all((p >= 0) & (p < np.asarray(shape_zyx))))

parser = argparse.ArgumentParser(
    description="Creates a new dataset adjusting its resolution",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument("-raw_input_data", "--raw_input_data", required=True, help="Directory to the folder where the raw data reside")
parser.add_argument("-label_input_data", "--label_input_data", required=True, help="Directory to the folder where the labels generated with micro_sam reside")
parser.add_argument("-output_data", "--output_data", required=True, help="Directory to the folder where the new data will be saved")

parser.add_argument("-patch_size", "--patch_size", type=int, nargs=3, default=(8,96,96), help="Directory to the folder where the new data will be saved")
parser.add_argument("-resolution_in_data", "--resolution_in_data", default="volumes.raw", type=str,
                    help="Path to the dataset that contains the 'resolution' attribute, e.g. 'volumes.raw' in CREMI format")
parser.add_argument("-locations_in_file", "--locations_in_file", default="annotations.locations", type=str,
                    help="Path to the labels inside each h5 datafile, e.g. 'annotations.locations' in CREMI format")
parser.add_argument("-raw_data_in_data", "--raw_data_in_data", default="volumes.raw", type=str,
                    help="Raw data inside each h5 datafile, e.g. 'volumes.raw' in CREMI format")
parser.add_argument("-ids_in_file", "--ids_in_file", default="annotations.ids", type=str,
                    help="Path to the IDs inside each h5 datafile, e.g. 'annotations.ids' in CREMI format")
parser.add_argument("-partners_in_file", "--partners_in_file", default="annotations.presynaptic_site.partners", type=str,
                    help="Path to the partners inside each h5 datafile, e.g. 'annotations.presynaptic_site.partners' in CREMI format")
args = vars(parser.parse_args())

input_data_folder = args['raw_input_data']
inpdata = args['raw_input_data'].rstrip('/')
label_data_folder = args['label_input_data']
output_data_folder = args['output_data']
last_folder = os.path.basename(inpdata)
patch_size = tuple(args['patch_size'])

print(f"Processing {input_data_folder} folder . . .")
raw_file_ids = sorted(next(os.walk(input_data_folder))[2])
raw_file_ids = [f for f in raw_file_ids if f.endswith('.h5') or f.endswith('.hdf5') or f.endswith('.hdf')]

# Read images
for n, id_ in tqdm(enumerate(raw_file_ids), total=len(raw_file_ids)):
    name = os.path.splitext(id_)[0]
    filename = os.path.join(input_data_folder, id_)

    # Find the corresponding label file
    label_filename = os.path.join(label_data_folder, "filtered_"+name+"_binarized.tif")
    if not os.path.exists(label_filename):
        raise ValueError(f"Label file {label_filename} does not exist for raw file {filename}. Please check your input folders and naming conventions.")

    print(f"Raw file: {filename} ; label file: {label_filename}")

    seg_data = read_img_as_ndarray(label_filename, is_3d=True)

    # Load raw volume (chunked read via BiaPy helper)
    file, raw_data = read_chunked_nested_data(filename, args['raw_data_in_data'])
    raw_data = np.array(raw_data)
    data_shape = raw_data.shape
    if isinstance(file, h5py.File):
        file.close()

    # Load locations
    file, locations = read_chunked_nested_data(filename, args['locations_in_file'])
    locations = np.array(locations)
    if isinstance(file, h5py.File):
        file.close()
        
    file, partners = read_chunked_nested_data(filename, args['partners_in_file'])
    partners = np.array(partners)
    if isinstance(file, h5py.File):
        file.close()

    file, ids = read_chunked_nested_data(filename, args['ids_in_file'])
    ids = list(np.array(ids))
    if isinstance(file, h5py.File):
        file.close()

    # Determine input resolution
    _, res_ds = read_chunked_nested_data(filename, args['resolution_in_data'])
    try:
        resolution = res_ds.attrs["resolution"]
    except Exception:
        raise ValueError(
            "There is no 'resolution' attribute in '{}'. Add it like: data['{}'].attrs['resolution'] = (8,8,8)".format(
                args['resolution_in_data'], args['resolution_in_data']
            )
        )

    if resolution[0] > 30:
        push_pixels = 15
    else:
        push_pixels = 25

    # Put all images within same range
    if raw_data.dtype != np.uint8:
        raw_data = (((raw_data - raw_data.min()) / (raw_data.max() - raw_data.min() + 1e-6)) * 255).astype(np.uint8)

    id_to_pos = {sid: i for i, sid in enumerate(ids)}
    shape_zyx = tuple(int(x) for x in data_shape)
    pre_post_points = {}  # pre_loc tuple -> list[post_loc tuple]
    pre_points, post_points = set(), set()
    for i in tqdm(range(len(partners))):
        pre_id, post_id = partners[i]
        pre_idx = id_to_pos.get(pre_id)
        post_idx = id_to_pos.get(post_id)
        if pre_idx is None or post_idx is None:
            # inconsistent annotation; skip quietly
            continue

        pre_loc = (locations[pre_idx] // resolution).astype(int)
        post_loc = (locations[post_idx] // resolution).astype(int)

        pre_ok = _in_bounds(pre_loc, shape_zyx)
        post_ok = _in_bounds(post_loc, shape_zyx)

        if pre_ok and post_ok:
            pre_key = tuple(pre_loc.tolist())
            pre_post_points.setdefault(pre_key, []).append(tuple(post_loc.tolist()))

    # Push pre points a bit further in the consensus direction of their post points to ensure they are more centered
    # in the pre synaptic cell and less likely to be on the edge where the segmentation is more difficult
    new_pre_post_points = {}
    for pre_key, post_locs in pre_post_points.items():
        pre_point = np.array(pre_key)

        # If we can we try to push the pre point a bit far so we ensure it 
        # is more centered in the pre synaptic cell
        pre_loc_pushed = push_pre_point_away(pre_point, post_locs, push_pixels)
        pre_loc_pushed = np.round(pre_loc_pushed).astype(int)
        new_pre_ok = _in_bounds(pre_loc_pushed, shape_zyx)
        ref_point = pre_point if not new_pre_ok else pre_loc_pushed

        new_pre_post_points[tuple(ref_point.tolist())] = post_locs
        
    # Extract different cubes with the synapse connection in the middle for synaptic partner assignment
    for pre_key, post_locs in tqdm(new_pre_post_points.items(), total=len(new_pre_post_points)):
        pre_point = np.array(pre_key)
        for post_loc in post_locs:
            post_point = np.array(post_loc)
            if not _in_bounds(pre_point, shape_zyx) or not _in_bounds(post_point, shape_zyx):
                continue

            # Take the middle of the pre and post points as the point to segment around
            mid_point = np.round((pre_point + post_point) / 2).astype(int)
            if not _in_bounds(mid_point, shape_zyx):
                continue
            
            pad_type = ["even", "even", "even"]
            z_min = mid_point[0] - patch_size[0]//2
            if z_min <= 0:
                z_min = 0
                pad_type[0] = "left"

            z_max = mid_point[0] + patch_size[0]//2
            if z_max >= shape_zyx[0]:
                z_max = shape_zyx[0]-1
                if pad_type[0] != "even":
                    pad_type[0] = "even"
                else:
                    pad_type[0] = "right"

            y_min = mid_point[1] - patch_size[1]//2
            if y_min <= 0:
                y_min = 0
                pad_type[1] = "left"

            y_max = mid_point[1] + patch_size[1]//2
            if y_max >= shape_zyx[1]:
                y_max = shape_zyx[1]-1
                if pad_type[1] != "even":
                    pad_type[1] = "even"
                else:
                    pad_type[1] = "right"
            
            x_min = mid_point[2] - patch_size[2]//2
            if x_min <= 0:
                x_min = 0
                pad_type[2] = "left"

            x_max = mid_point[2] + patch_size[2]//2
            if x_max >= shape_zyx[2]:
                x_max = shape_zyx[2]-1
                if pad_type[2] != "even":
                    pad_type[2] = "even"
                else:
                    pad_type[2] = "right"
            
            tag = f"z{z_min}-{z_max}_y{y_min}-{y_max}_x{x_min}-{x_max}"

            label_cube = seg_data[
                z_min:z_max,
                y_min:y_max,
                x_min:x_max
            ]
            label_cube = pad_and_reflect(label_cube, patch_size + (label_cube.shape[-1],), pad_type=pad_type, verbose=False)

            # Identify where each class exists
            post_mask = label_cube[..., 0] > 0
            pre_mask = label_cube[..., 1] > 0
            new_label = np.zeros(label_cube.shape[:-1], dtype=label_cube.dtype)
            # Assign values: Class 1 takes priority, then Class 2
            # (Or vice versa, depending on which you want to 'win' if they overlap)
            new_label[pre_mask] = 2 
            new_label[post_mask] = 1
            label_cube = np.expand_dims(new_label, -1)

            raw_cube = raw_data[
                z_min:z_max,
                y_min:y_max,
                x_min:x_max
            ]
            raw_cube = pad_and_reflect(np.expand_dims(raw_cube,-1), patch_size + (raw_cube.shape[-1],), pad_type=pad_type, verbose=False)
            save_tif(np.expand_dims(raw_cube,0), os.path.join(output_data_folder, "raw"), [f"raw_{name}_{tag}.tif"], verbose=False)
            save_tif(np.expand_dims(label_cube,0), os.path.join(output_data_folder, "label"), [f"label_{name}_{tag}.tif"], verbose=False)

print("Done!")