import argparse
import os
import sys
import numpy as np
from tqdm import tqdm
import h5py
import ast
from pathlib import Path
import pandas as pd
from scipy.ndimage import uniform_filter, label

sys.path.insert(0, '/net/fibserver1/data/raw/scratch/dfranco/BiaPy')  # Adjust this path as needed
from biapy.data.data_manipulation import save_tif, read_img_as_ndarray
from biapy.data.data_3D_manipulation import read_chunked_nested_data

def get_pre_synaptic_coord(raw_patch, seg_patch, post_site, resolution=(40, 4, 4), search_radii_nm=(40, 160, 160), ring_width_factor=0.15,
    push_point=True):
    """
    Identifies the pre-synaptic partner by scanning a stack of 2D rings (annuli) 
    across multiple Z-slices centered on the post-synaptic site.

    The search logic uses a 'Ring-Stack' approach:
    1. It defines a Z-range based on the Z-radius (e.g., +/- 1 slice at 40nm).
    2. In each slice, it defines a 2D ring in the YX plane.
    3. The ring's outer boundary is the search_radii_nm[1].
    4. The ring's inner boundary is defined by (1 - ring_width_factor) * radius.
       This creates a 'hollow' search area that focuses on the expected 
       synaptic interface distance while ignoring the immediate post-site center.
    5. It picks the voxel within this 'hollow' ring that has the darkest 
       local intensity (lowest beam score), indicating the cleft or T-bar.

    Parameters
    ----------
    raw_patch : np.ndarray
        The raw image patch centered around the post synaptic point.
    seg_patch : np.ndarray
        The segmentation patch (same shape as raw_patch) where 1=pre-synaptic cell, 2=post-synaptic cell.
    post_site : tuple
        The (z, y, x) coordinates of the post synaptic point within the patch.
    resolution : tuple
        The physical resolution of the data in (z, y, x) order, e.g. (40nm, 4nm, 4nm).
    search_radii_nm : tuple
        The search radii in nanometers for (z, y, x) directions, e.g. (40nm, 160nm, 160nm). This defines how far from the post site we look for pre-synaptic candidates.
    ring_width_factor : float
        The thickness of the search ring relative to its radius (e.g., 0.2 = outer 20% of the circle).
    push_point : bool
        Whether to apply a final push to the pre-synaptic point away from the post-synaptic island to ensure better separation.
    
    Returns
    -------
    best_pre_raw : np.ndarray or None
        The (z, y, x) coordinates of the identified pre-synaptic point within the patch, after optional pushing.
    """
    res = np.array(resolution, dtype=np.float32)
    Z, Y, X = raw_patch.shape
    post_f = np.array(post_site, dtype=np.float32)
    
    # 1. Smoothing (XY focus)
    smoothed = uniform_filter(raw_patch, size=(1, 5, 5), mode="nearest")

    # 2. Convert physical radii to voxel counts
    # (40nm / 40nm = 1 slice) , (160nm / 4nm = 40 pixels)
    r_vox = np.round(np.array(search_radii_nm) / res).astype(int)
    rz, ry, rx = r_vox

    best_pre_raw = None
    min_intensity = np.inf
    beam_hw = (1, 4, 4) # Focused on high-res XY

    # 3. Iterate through the slices (The Stack)
    z_start = max(0, int(post_f[0] - rz))
    z_end = min(Z, int(post_f[0] + rz + 1))

    for z in range(z_start, z_end):
        # 4. Analyze the 2D Ring in this slice
        # Create a local YX window around the post site
        y_min, y_max = max(0, int(post_f[1] - ry)), min(Y, int(post_f[1] + ry + 1))
        x_min, x_max = max(0, int(post_f[2] - rx)), min(X, int(post_f[2] + rx + 1))
        
        y_grid = np.arange(y_min, y_max)
        x_grid = np.arange(x_min, x_max)
        yy, xx = np.meshgrid(y_grid, x_grid, indexing='ij')
        
        # Calculate physical YX distance from center in this slice
        dy_nm = (yy - post_f[1]) * res[1]
        dx_nm = (xx - post_f[2]) * res[2]
        dists_yx_nm = np.sqrt(dy_nm**2 + dx_nm**2)
        
        # Define the ring (e.g., the outer 20% of the radius)
        # Using the Y radius (160nm) as the threshold
        ring_mask = (dists_yx_nm >= search_radii_nm[1] * (1 - ring_width_factor)) & (dists_yx_nm <= search_radii_nm[1] * (1 + ring_width_factor))
        
        ring_yy = yy[ring_mask]
        ring_xx = xx[ring_mask]

        # 5. Evaluate points in the ring for this slice
        # Sample every 5th point to be "marginal" as requested
        for i in range(0, len(ring_yy), 5):
            cur_y, cur_x = ring_yy[i], ring_xx[i]
            
            if seg_patch[z, cur_y, cur_x] == 1: # Check Pre-mask
                score = _beam_score(smoothed, z, cur_y, cur_x, beam_hw)
                if score < min_intensity:
                    min_intensity = score
                    best_pre_raw = np.array([z, cur_y, cur_x])

    # 6. Fallbacks
    if best_pre_raw is None:
        # Check whole volume inside the radii if ring was empty
        pre_coords = np.argwhere(seg_patch == 1)
        if len(pre_coords) == 0: return None
        
        # Physical distance check
        dists = np.sum(((pre_coords - post_f) * res)**2, axis=1)
        best_pre_raw = pre_coords[np.argmin(dists)]

    if best_pre_raw is None:
        return None

    # 7. Final XY-Only Push (10 pixels)
    if push_point:
        labeled_post, _ = label(seg_patch == 2)
        post_label = labeled_post[tuple(post_f.astype(int))]
        post_island = np.argwhere(labeled_post == post_label) if post_label > 0 else np.argwhere(seg_patch == 2)
        
        final_point = push_pre_point_away(best_pre_raw, post_island, 10, res)
    else:
        final_point = best_pre_raw

    return np.clip(np.round(final_point), [0,0,0], [Z-1, Y-1, X-1]).astype(int)


def _clip_slices(z, y, x, hz, hy, hx, Z, Y, X):
    z0, z1 = max(0, z - hz), min(Z, z + hz + 1)
    y0, y1 = max(0, y - hy), min(Y, y + hy + 1)
    x0, x1 = max(0, x - hx), min(X, x + hx + 1)
    return slice(int(z0), int(z1)), slice(int(y0), int(y1)), slice(int(x0), int(x1))

def _beam_score(smoothed, z, y, x, beam_hw, score_mode="p10"):
    hz, hy, hx = beam_hw
    Z, Y, X = smoothed.shape
    slz, sly, slx = _clip_slices(z, y, x, hz, hy, hx, Z, Y, X)
    slab = smoothed[slz, sly, slx]
    if slab.size == 0: return 255.0
    return float(np.percentile(slab, 10))

def push_pre_point_away(pre_point, post_points, pixel_dist, resolution):
    """
    Pushes the pre_point away from post_centroid in physical space, 
    but constrains the movement to the XY plane.
    """
    res = np.array(resolution, dtype=np.float32)
    pre_f = np.array(pre_point, dtype=np.float32)
    post_centroid = np.mean(post_points, axis=0)
    
    # Calculate vector in nanometers
    outward_nm = (pre_f - post_centroid) * res
    
    # CRITICAL: Zero out the Z component to restrict to XY plane
    outward_nm[0] = 0 
    
    nm_norm = np.linalg.norm(outward_nm)
    
    if nm_norm == 0:
        return pre_f
    
    # Unit vector in physical XY space
    unit_vector_nm = outward_nm / nm_norm
    
    # Push by pixel_dist (direction is physically correct, scale is in voxels)
    push_vector_vox = unit_vector_nm * pixel_dist
    
    return pre_f + push_vector_vox

parser = argparse.ArgumentParser(
    description="Extract synapse connection patches centered around the given post synaptic points",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument("-raw_input_data", "--raw_input_data", required=True, help="Directory to the folder where the raw data reside")
parser.add_argument("-pre_post_cell_seg", "--pre_post_cell_seg", required=True, help="Directory to the folder where the pre/post cells are segmented")
parser.add_argument("-F_post_pred_post_location", "--F_post_pred_post_location", required=True, help="Directory to the folder where the predicted post synaptic points are stored. The script will find '_pred_post_locations.csv' files in this folder, which should have columns 'x', 'y', 'z' for the coordinates of the predicted post synaptic points.")
parser.add_argument("-output_data", "--output_data", required=True, help="Directory to the folder where the final synaptic connections will be placed")
parser.add_argument("-patch_size", "--patch_size", type=int, nargs=3, default=(8,96,96), help="Size of the extracted patches (z, y, x)")

parser.add_argument("-raw_data_in_data", "--raw_data_in_data", default="volumes.raw", type=str,
                    help="Raw data inside each h5 datafile, e.g. 'volumes.raw' in CREMI format")
parser.add_argument("-resolution_in_data", "--resolution_in_data", default="volumes.raw", type=str,
                    help="Path to the dataset that contains the 'resolution' attribute, e.g. 'volumes.raw' in CREMI format")
args = vars(parser.parse_args())


input_data_folder = args['raw_input_data']
pre_post_cell_seg_folder = args['pre_post_cell_seg']
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

    # Determine input resolution
    file, res_ds = read_chunked_nested_data(filename, args['resolution_in_data'])
    try:
        resolution = res_ds.attrs["resolution"]
    except Exception:
        raise ValueError(
            "There is no 'resolution' attribute in '{}'. Add it like: data['{}'].attrs['resolution'] = (8,8,8)".format(
                args['resolution_in_data'], args['resolution_in_data']
            )
        )
    if isinstance(file, h5py.File):
        file.close()

    # Find the corresponding label file
    cell_seg_filename = os.path.join(pre_post_cell_seg_folder, name+".tif")
    if not os.path.exists(cell_seg_filename):
        raise ValueError(f"Predicted cell (pre/post) segmentation file {cell_seg_filename} does not exist. Please check your input folders and naming conventions. ")
    cell_seg = read_img_as_ndarray(cell_seg_filename, is_3d=True)

    # Find the corresponding label file
    post_locations_filename = os.path.join(F_post_pred_post_folder, name+"_pred_post_locations.csv")
    if not os.path.exists(post_locations_filename):
        raise ValueError(f"F_post predicted post point file {post_locations_filename} does not exist. Please check your input folders and naming conventions.")

    print(f"Raw file: {filename} ; post locations file: {post_locations_filename}")

    post_point_df = pd.read_csv(post_locations_filename, index_col=False)
    shape_zyx = tuple(int(x) for x in data_shape)
    pre_points = []
    pre_ids = []
    pre_id_count = 1
    pre_post_pair = []
    for idx, row in post_point_df.iterrows():
        x, y, z = int(row['axis-2']), int(row['axis-1']), int(row['axis-0'])
        post_site = (z,y,x)

        z_min = max(0, post_site[0] - args['patch_size'][0]//2)
        z_max = min(shape_zyx[0], post_site[0] + args['patch_size'][0]//2)
        y_min = max(0, post_site[1] - args['patch_size'][1]//2)
        y_max = min(shape_zyx[1], post_site[1] + args['patch_size'][1]//2)
        x_min = max(0, post_site[2] - args['patch_size'][2]//2)
        x_max = min(shape_zyx[2], post_site[2] + args['patch_size'][2]//2)

        raw_patch = raw_data[z_min:z_max, y_min:y_max, x_min:x_max]
        seg_patch = cell_seg[z_min:z_max, y_min:y_max, x_min:x_max]
        
        # save_tif(np.expand_dims(np.expand_dims(raw_patch,0),-1), output_data_folder, [f"raw_patch.tif"], verbose=False)
        # save_tif(np.expand_dims(seg_patch,0), output_data_folder, [f"seg_patch.tif"], verbose=False)
        
        local_post_site = (post_site[0]-z_min, post_site[1]-y_min, post_site[2]-x_min)
        local_pre_cord = get_pre_synaptic_coord(raw_patch, seg_patch.squeeze(), local_post_site, resolution=resolution, push_point=False)
        if local_pre_cord is not None:
            global_pre_cord = local_pre_cord + np.array([z_min, y_min, x_min])
            pre_points.append(global_pre_cord)
            pre_ids.append(pre_id_count)
            
            pre_post_pair.append((pre_id_count, int(row['post_id'])))
            pre_id_count += 1

    pre_points = np.array(pre_points)
    pre_post_pair = np.array(pre_post_pair)

    os.makedirs(output_data_folder, exist_ok=True)
    pd.DataFrame(zip(pre_ids, pre_points[:,0], pre_points[:,1], pre_points[:,2]), columns=['pre_id', 'axis-0', 'axis-1', 'axis-2']).to_csv(os.path.join(output_data_folder, name+"_pred_pre_locations.csv"), index=False)
    pd.DataFrame(zip(pre_post_pair[:,0], pre_post_pair[:,1]), columns=['pre_id', 'post_id']).to_csv(os.path.join(output_data_folder, name+"_pre_post_mapping.csv"), index=False)

print("Done!")