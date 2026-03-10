import argparse
import os
import sys
import numpy as np
from tqdm import tqdm
import h5py
import ast
from skimage.transform import resize
from micro_sam.util import get_sam_model, precompute_image_embeddings
from micro_sam.prompt_based_segmentation import segment_from_points
from skimage.measure import label, regionprops_table
from scipy import ndimage
import torch 

sys.path.insert(0, '/net/fibserver1/data/raw/scratch/dfranco/BiaPy')  # Adjust this path as needed
from biapy.data.data_manipulation import save_tif
from biapy.data.data_3D_manipulation import read_chunked_nested_data


def get_safe_inward_point(mask, current_yx, search_radius=10):
    # 1. Distance transform: how far is each pixel from the edge?
    dist_map = ndimage.distance_transform_edt(mask)
    
    y, x = map(int, current_yx)
    
    # 2. Define a local search window around your point
    y_min, y_max = max(0, y - search_radius), min(mask.shape[0], y + search_radius)
    x_min, x_max = max(0, x - search_radius), min(mask.shape[1], x + search_radius)
    
    local_dist_window = dist_map[y_min:y_max, x_min:x_max]
    
    # 3. Find the "deepest" point (local center of mass/thickness) in that window
    coords = np.unravel_index(np.argmax(local_dist_window), local_dist_window.shape)
    
    # Convert back to global coordinates
    safe_target = np.array([coords[0] + y_min, coords[1] + x_min])
    
    # 4. Use the linear interpolation we discussed to move slightly toward it
    # This ensures we move "inward" but stay near the original path
    fraction = 0.3
    new_point = current_yx + fraction * (safe_target - current_yx)
    
    return new_point

def keep_largest_component(mask):
    # Ensure mask is binary (0 and 1)
    mask = mask.astype(int)
    
    # Label disconnected regions
    # Each separate island gets a unique integer ID
    labels = label(mask)
    
    # If the mask is empty, just return it
    if labels.max() == 0:
        return mask, 0

    # Find the largest non-background component
    # np.bincount counts occurrences of each label ID
    # We ignore index 0 because that's the background
    counts = np.bincount(labels.flat)
    counts[0] = 0 
    
    largest_label = counts.argmax()
    area = counts[largest_label]

    # Create a new mask containing only the largest component
    new_mask = (labels == largest_label).astype(np.uint8)
    return new_mask, area

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

def return_no_empty_mask(predictor, z, y, x, image_embeddings, shape, min_size=20):
    
    # Try a few points in case the first point is returning an empty mask
    first = True
    for offset in [[0,0],[-1,-1],[1,1],[1,0],[0,1],[-1,0],[0,-1]]:
        _y = min(max(0,y+offset[0]), shape[1])
        _x = min(max(0,x+offset[1]), shape[2])
        if not first:
            print(f"     Trying point: ({z},{_y},{_x})")
        out = segment_from_points(
            predictor=predictor,
            points=np.array([[_y,_x]]), 
            labels=np.array([1]),  
            image_embeddings=image_embeddings, 
            i=z,
        )
        out = out.squeeze().astype(np.uint8)
        out, ref_area = keep_largest_component(out)
        if "acc_out_mask" not in locals():
            acc_out_mask = out
        else:
            # accumulate the masks from different points to increase the chance of getting a non-empty mask
            acc_out_mask = np.logical_or(acc_out_mask, out)
        if out.max() == 1:
            if ref_area < min_size:  # if the segmented area is too small, it's likely a false positive, so we try another point
                print(f"     Mask too small (area={ref_area}). Trying another point")
                continue
            else:
                break
        else:
            first = False
            print("     Mask empty. Trying another point")

    return acc_out_mask.squeeze(), ref_area

parser = argparse.ArgumentParser(
    description="Creates a new dataset adjusting its resolution",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument("-input_data", "--input_data", required=True, help="Directory to the folder where the data reside")
parser.add_argument("-output_data", "--output_data", required=True, help="Directory to the folder where the new data will be saved")

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

input_data_folder = args['input_data']
inpdata = args['input_data'].rstrip('/')
output_data_folder = args['output_data']
last_folder = os.path.basename(inpdata)

print(f"Processing {input_data_folder} folder . . .")
file_ids = sorted(next(os.walk(input_data_folder))[2])
file_ids = [f for f in file_ids if f.endswith('.h5') or f.endswith('.hdf5') or f.endswith('.hdf')]

predictor = get_sam_model(model_type="vit_b_lm")

# Read images
for n, id_ in tqdm(enumerate(file_ids), total=len(file_ids)):
    out_filename = os.path.join(output_data_folder, "filtered_"+os.path.splitext(id_)[0]+".tif")
    if os.path.exists(out_filename):
        print(f"File {out_filename} already exists, skipping.")
        continue

    filename = os.path.join(input_data_folder, id_)
    print(f"FILE: {filename}")

    # Load raw volume (chunked read via BiaPy helper)
    file, data = read_chunked_nested_data(filename, args['raw_data_in_data'])
    data = np.array(data)
    data_shape = data.shape
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
        slc_to_process = [-1,1]
        push_pixels = 15
    else:
        slc_to_process = [-2,-1,1,2]
        push_pixels = 25

    # Put all images within same range
    if data.dtype != np.uint8:
        data = (((data - data.min()) / (data.max() - data.min() + 1e-6)) * 255).astype(np.uint8)

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
    for pre_key, post_locs in pre_post_points.items():
        pre_point = np.array(pre_key)

        # If we can we try to push the pre point a bit far so we ensure it 
        # is more centered in the pre synaptic cell
        pre_loc_pushed = push_pre_point_away(pre_point, post_locs, push_pixels)
        pre_loc_pushed = np.round(pre_loc_pushed).astype(int)
        new_pre_ok = _in_bounds(pre_loc_pushed, shape_zyx)
        ref_point = pre_point if not new_pre_ok else pre_loc_pushed

        pre_points.add(tuple(ref_point.tolist()))
        for post_loc in post_locs:
            post_points.add(post_loc)
    
    # Sort by z to make it easier to visualize in the debugger
    pre_points = sorted(pre_points, key=lambda p: p[0])
    post_points = sorted(post_points, key=lambda p: p[0])

    expected_keys = ['features', 'input_size', 'original_size']
    # Save the embeddings for later use (optional, can be commented out if not needed)
    first_key = os.path.join(output_data_folder, f"{id_}_image_embeddings_{expected_keys[0]}.npy")
    if not os.path.exists(first_key):
        # Compute the image embeddings.
        image_embeddings = precompute_image_embeddings(
            predictor=predictor,
            input_=data,
            ndim=3, 
            verbose=True,
        )  

        os.makedirs(output_data_folder, exist_ok=True)
        for key in expected_keys:
            np.save(os.path.join(output_data_folder, f"{id_}_image_embeddings_{key}.npy"), image_embeddings[key])
    else:
        image_embeddings = {
            expected_keys[0]: np.load(os.path.join(output_data_folder, f"{id_}_image_embeddings_{expected_keys[0]}.npy")),
            expected_keys[1]: np.load(os.path.join(output_data_folder, f"{id_}_image_embeddings_{expected_keys[1]}.npy")), 
            expected_keys[2]: np.load(os.path.join(output_data_folder, f"{id_}_image_embeddings_{expected_keys[2]}.npy")),
        }
        image_embeddings["original_size"] = image_embeddings["original_size"].tolist()
        image_embeddings["input_size"] = torch.Size(image_embeddings["input_size"])

    micro_sam_seg = np.zeros(data.shape + (2,), dtype=np.uint16)
    c_pre, c_post = 0, 0 
    point_info = {"pre": {}, "post": {}}

    print("Segmenting neurons with micro_sam slice by slice. . .")
    for z in tqdm(range(data.shape[0]), total=data.shape[0]):
        # Sky slice if it's all black
        if data[z].max() == 0:
            print("Slice {} is empty, skipping.".format(z))
            continue

        # Fill the presynaptic sites with the segmentation from micro_sam
        while c_pre < len(pre_points) and int(pre_points[c_pre][0]) <= z:
            point = pre_points[c_pre]
            x, y = int(point[2]), int(point[1])
            print("Segmenting presynaptic point at (z,y,x) = ({},{},{})".format(z,y,x))

            ###########################
            # Fill the current slice with the segmentation from micro_sam
            ###########################
            out, ref_area = return_no_empty_mask(predictor, z, y, x, image_embeddings, data.shape)

            is_empty = (micro_sam_seg[z,...,0] == 0)
            update_indices = out.astype(bool) & is_empty
            micro_sam_seg[z, update_indices, 0] = c_pre + 1
            point_info["pre"][c_pre + 1] = [max(z-1,0), min(data.shape[0], z+1)]

            ###########################
            # Fill one slice above and one slice below with the segmentation from micro_sam, using the centroid of the segmented region as prompt
            ###########################

            # Create a new point within the object towards the maximum of the distance transform
            p_new = get_safe_inward_point(out, (y, x), search_radius=int(resolution[0]*0.3))
            p_new = np.round(p_new).astype(int)

            # Something weird happened as the new point should be inside the object
            if out[p_new[0],p_new[1]] == 0:
                import pdb; pdb.set_trace()
                continue 

            for dz in slc_to_process:
                z_new = z + dz
                if z_new < 0 or z_new >= data.shape[0]:
                    continue
                
                out2, ref_area2 = return_no_empty_mask(predictor, z_new, p_new[0], p_new[1], image_embeddings, data.shape)
                if ref_area2 > ref_area*2:  # if the segmented area in the new slice is more than 50% of the original area, keep it. Otherwise, discard it to avoid false positives
                    continue

                is_empty = (micro_sam_seg[z_new,...,0] == 0)
                update_indices = out2.astype(bool) & is_empty
                micro_sam_seg[z_new, update_indices, 0] = c_pre + 1
            
            c_pre += 1
            if c_pre % 10 == 0:
                print(f"Processed {c_pre} presynaptic points out of {len(pre_points)}")

        # Fill the postsynaptic sites with the segmentation from micro_sam
        while c_post < len(post_points) and int(post_points[c_post][0]) <= z:
            point = post_points[c_post]
            x, y = int(point[2]), int(point[1])
            print("Segmenting postsynaptic point at (z,y,x) = ({},{},{})".format(z,y,x))

            ###########################
            # Fill the current slice with the segmentation from micro_sam
            ###########################
            out, ref_area = return_no_empty_mask(predictor, z, y, x, image_embeddings, data.shape)

            is_empty = (micro_sam_seg[z,...,1] == 0)
            update_indices = out.astype(bool) & is_empty
            micro_sam_seg[z, update_indices, 1] = c_post + 1
            point_info["post"][c_post + 1] = [max(z-1,0), min(data.shape[0], z+1)]

            # # Debug
            # if (z,y,x) == (17,132,1105):
            #     import pdb; pdb.set_trace()

            ###########################
            # Fill one slice above and one slice below with the segmentation from micro_sam, using the centroid of the segmented region as prompt
            ###########################

            # Create a new point within the object towards the maximum of the distance transform
            p_new = get_safe_inward_point(out, (y, x), search_radius=int(resolution[0]*0.3))
            p_new = np.round(p_new).astype(int)

            # Something weird happened as the new point should be inside the object
            if out[p_new[0],p_new[1]] == 0:
                import pdb; pdb.set_trace()
                continue 
            
            for dz in slc_to_process:
                z_new = z + dz
                if z_new < 0 or z_new >= data.shape[0]:
                    continue
                
                out2, ref_area2 = return_no_empty_mask(predictor, z_new, p_new[0], p_new[1], image_embeddings, data.shape)
                if ref_area2 > ref_area*2:  # if the segmented area in the new slice is more than 50% of the original area, keep it. Otherwise, discard it to avoid false positives
                    continue

                is_empty = (micro_sam_seg[z_new,...,1] == 0)
                update_indices = out2.astype(bool) & is_empty
                micro_sam_seg[z_new, update_indices, 1] = c_post + 1

            c_post += 1
            if c_post % 10 == 0:
                print(f"Processed {c_post} postsynaptic points out of {len(post_points)}")

    save_tif(np.expand_dims(micro_sam_seg,0), output_data_folder, ["all_"+id_], verbose=True)
    name = os.path.splitext(id_)[0]+"_binarized.tif"
    save_tif(np.expand_dims((micro_sam_seg > 0).astype(np.uint8),0), output_data_folder, ["all_"+name], verbose=True)

    ref_channel = 0
    ref_tag = "pre"
    target_channel = 1
    target_tag = "post"
    len_points = c_pre 

    # Look instance by instance if there are overlaps between other channel's mask
    for i in range(len_points):
        instance = i+1
        if instance in point_info[ref_tag]:
            z_start = point_info[ref_tag][instance][0]
            z_end = point_info[ref_tag][instance][1]

            pre_patch = micro_sam_seg[z_start:z_end, ..., ref_channel]
            post_patch = micro_sam_seg[z_start:z_end, ..., target_channel]
            instance_mask = (pre_patch == instance)

            # if there is any instance in the target channel that overlaps the reference channel
            # then we need to find out the amount of overlap and preserve the target channel is case
            # the overlap is more than 60%
            # 1. Find all target IDs that overlap with the current instance_mask
            overlapping_ids = np.unique(post_patch[instance_mask])
            
            # Remove 0 (background) from the list of IDs
            overlapping_ids = overlapping_ids[overlapping_ids != 0]

            for t_id in overlapping_ids:
                # 2. Create mask for the specific target instance
                target_mask = (post_patch == t_id)
                
                # 3. Calculate Intersection and Union (or just Intersection vs Target Size)
                # Overlap is usually defined as (A ∩ B) / Area_of_B 
                # to see how much of the target is "covered" by the reference
                intersection = np.logical_and(instance_mask, target_mask).sum()
                target_area = target_mask.sum()
                
                overlap_ratio = intersection / target_area

                print(f"Overlap between pre point {instance} and post point {t_id}: {overlap_ratio*100:.2f}%")
                # # 4. If overlap > 60%, preserve the target by removing it from the ref channel
                # if overlap_ratio > 0.60:
                #     # In this case, the post instance "wins" as the pre usually has a larger area and we want to preserve the smaller one that is more likely to be correct.
                pre_patch[target_mask & instance_mask] = 0

    save_tif(np.expand_dims(micro_sam_seg,0), output_data_folder, ["filtered_"+id_], verbose=True)
    name = os.path.splitext(id_)[0]+"_binarized.tif"
    micro_sam_seg = (micro_sam_seg > 0).astype(np.uint8)
    print("")
    for z in range(micro_sam_seg.shape[0]):
         for c in range(micro_sam_seg.shape[-1]):
            micro_sam_seg[z,...,c] = ndimage.binary_closing(micro_sam_seg[z,...,c], structure=np.ones((3,3)))
            micro_sam_seg[z,...,c] = ndimage.binary_opening(micro_sam_seg[z,...,c], structure=np.ones((3,3)))
    save_tif(np.expand_dims(micro_sam_seg,0), output_data_folder, ["filtered_"+name], verbose=True)

print("Done!")