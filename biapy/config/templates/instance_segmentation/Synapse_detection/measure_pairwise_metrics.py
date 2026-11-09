import argparse
import os
import sys
from tqdm import tqdm
import pandas as pd
import ast
from skimage.morphology import disk, dilation
import numpy as np
import h5py
from scipy.optimize import linear_sum_assignment

parser = argparse.ArgumentParser(description="Convert semantic probabilities into points",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-input_pred_dir", "--input_pred_dir", required=True, help="Directory to the folder containing one subfolder per run (e.g. run1, run2, run3), each with the predicted points")
parser.add_argument("-input_gt_dir", "--input_gt_dir", required=True, help="Directory to the folder where the GT points are stored")
parser.add_argument("-BiaPy_dir", "--BiaPy_dir", required=True, help="BiaPy directory")
parser.add_argument("-tolerance", "--tolerance", default=160, type=int, help="Maximum distance far away from a GT point to consider a point as a true positive")
parser.add_argument("-save_matching", "--save_matching", action="store_true", help="Save matching between GT and predicted points")
parser.add_argument("-verbose", "--verbose", action="store_true", help="Enable verbose output")
parser.add_argument("-output_dir", "--output_dir", required=False, help="Output folder to store the final points")
parser.add_argument("-matching_threshold", "--matching_threshold", default=550, type=int, help="Threshold for matching GT and predicted points")

args = vars(parser.parse_args())

locations_path = "annotations.locations"
resolution_path = 'volumes.raw'
partners_path = "annotations.presynaptic_site.partners"
id_path = "annotations.ids"

# Call example:
# python -u /net/fibserver1/data/raw/scratch/dfranco/datasets/synapses/code/dani/metrics/measure_pairwise_metrics.py --input_pred_dir /net/fibserver1/data/raw/samia_dani/SynapseDetectionPaper/results/in_domain/fibsem/predictions/HEMIBRAIN/simpsyn/ -input_gt_dir /net/fibserver1/data/raw/samia_dani/SynapseDetectionPaper/data/COMBINED_DATASETS/FIBSEM/ORIGINAL/HEMIBRAIN/test -matching_threshold 550 --BiaPy_dir /net/fibserver1/data/raw/scratch/dfranco/BiaPy
# The script expects args['input_pred_dir'] to contain one subfolder per run (run1, run2, run3, ...),
# each holding the same CSV outputs that used to live directly in input_pred_dir.

def distance(a, b):
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(np.array(a) - np.array(b))

def cost(pair1, pair2, threshold):
    """Calculate cost between two synapse pairs."""
    pre_dist = distance(pair1[0], pair2[0])
    post_dist = distance(pair1[1], pair2[1])
    if pre_dist > threshold or post_dist > threshold:
        return 2 * threshold
    return 0.5 * (pre_dist + post_dist)

def compute_metrics(pred_pairs, gt_pairs, threshold):
    """
    Compute precision, recall, F1 using Hungarian algorithm.
    """
    n_p, n_g = len(pred_pairs), len(gt_pairs)

    if n_p == 0 or n_g == 0:
        return 0.0, 0.0, 0.0, 0, n_p, n_g

    # Build cost matrix
    size = max(n_p, n_g)
    costs = np.full((size, size), 2 * threshold, dtype=float)

    for i in range(n_p):
        for j in range(n_g):
            costs[i, j] = cost(pred_pairs[i], gt_pairs[j], threshold)

    # Hungarian matching
    ri, ci = linear_sum_assignment(costs)

    # Count true positives (matches within threshold)
    tp = sum(1 for i, j in zip(ri, ci)
             if i < n_p and j < n_g and costs[i, j] <= threshold)

    fp = n_p - tp
    fn = n_g - tp

    precision = tp / n_p if n_p > 0 else 0.0
    recall = tp / n_g if n_g > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return f1, precision, recall, tp, fp, fn

sys.path.insert(0, args['BiaPy_dir'])
from biapy.engine.metrics import detection_metrics
from biapy.data.data_3D_manipulation import read_chunked_nested_data

if args['save_matching'] and args['output_dir'] is None:
    raise ValueError("If --save_matching is set, --output_dir must be provided to specify where to save the GT-predicted point associations and false positives.")

print("Processing GT {} folder . . .".format(args['input_gt_dir']))
gt_ids = sorted(next(os.walk(args['input_gt_dir']))[2])
gt_ids = [fname for fname in gt_ids if ".~lock." not in fname]
if len(gt_ids) == 0:
    raise ValueError(f"No files found in the input GT directory '{args['input_gt_dir']}'. Please check the path and ensure it contains the GT H5 files.")

suffixes = (
    "_pre_post_mapping.csv",
    "_pred_post_locations.csv",
    "_pred_pre_locations.csv",
)

# Each subfolder of input_pred_dir is expected to be one run (e.g. run1, run2, run3)
run_names = sorted(next(os.walk(args['input_pred_dir']))[1])
if len(run_names) == 0:
    raise ValueError(f"No run subfolders found in '{args['input_pred_dir']}'. Expected one subfolder per run (e.g. run1, run2, run3).")


def process_run(pred_dir):
    """Run the per-file pairwise matching for a single run folder and return the per-file metrics."""
    print("Processing {} folder . . .".format(pred_dir))
    pred_ids = sorted(next(os.walk(pred_dir))[2])
    pred_ids = [fname for fname in pred_ids if ".~lock." not in fname]

    if len(pred_ids) == 0:
        raise ValueError(f"No files found in the input prediction directory '{pred_dir}'. Please check the path and ensure it contains the predicted CSV files.")

    # Find the common base IDs between GT and predicted files by removing the suffixes from the predicted file names
    base_ids = []
    for pred_id in pred_ids:
        for suffix in suffixes:
            if pred_id.endswith(suffix):
                base_id = pred_id[:-len(suffix)]
                base_ids.append(base_id)
    base_ids = list(set(base_ids))

    run_metrics = {
        "Precision": [], "Recall": [], "F1": [], "TP": [], "FP": [], "FN": []
    }
    for id in tqdm(base_ids, desc="Processing files"):
        # Read the GT coordinates from the H5 file
        gt_file = [fname for fname in gt_ids if fname.startswith(id)]
        if len(gt_file) == 0:
            print(f"Warning: No GT file found for ID '{id}' in '{args['input_gt_dir']}'. Skipping.")
            continue
        else:
            print(f"Processing GT file '{gt_file[0]}' for ID '{id}'.")
        gt_filename = os.path.join(args['input_gt_dir'], gt_file[0])
        dfile, ids = read_chunked_nested_data(gt_filename, id_path)
        ids = list(np.array(ids))
        if isinstance(dfile, h5py.File):
            dfile.close()

        dfile, partners = read_chunked_nested_data(gt_filename, partners_path)
        partners = np.array(partners)
        if isinstance(dfile, h5py.File):
            dfile.close()

        dfile, locations = read_chunked_nested_data(gt_filename, locations_path)
        locations = np.array(locations)
        if isinstance(dfile, h5py.File):
            dfile.close()

        # Determine input resolution
        try:
            _, res_ds = read_chunked_nested_data(gt_filename, resolution_path)
            resolution = res_ds.attrs["resolution"]
        except Exception:
            raise ValueError(
                "There is no 'resolution' attribute in '{}'. Add it like: data['{}'].attrs['resolution'] = (8,8,8)".format(
                    resolution_path, resolution_path
                )
            )
        print(f"Data resolution: {resolution}")

        gt_pairs = []
        for i in range(len(partners)):
            pre_id, post_id = partners[i]
            pre_position = ids.index(pre_id)
            post_position = ids.index(post_id)
            pre_coord = locations[pre_position]
            post_coord = locations[post_position]
            gt_pairs.append((pre_coord, post_coord))

        # Read the predicted pre coordinates from the CSV file
        pred_pre_csv_path = os.path.join(pred_dir, f"{id}_pred_pre_locations.csv")
        pred_pre_map = {}
        if os.path.exists(pred_pre_csv_path):
            df_pred_pre = pd.read_csv(pred_pre_csv_path)
            pre_id_tag = "pre_id" if "pre_id" in df_pred_pre.columns else "Pre_ID"
            if "Pre_Z" in df_pred_pre.columns:
                zcoords = df_pred_pre["Pre_Z"].tolist()
                ycoords = df_pred_pre["Pre_Y"].tolist()
                xcoords = df_pred_pre["Pre_X"].tolist()
                pred_pre_coordinates = [[int(z), int(y), int(x)] for z, y, x in zip(zcoords, ycoords, xcoords)]
            else:
                zcoords = df_pred_pre["axis-0"].tolist()
                ycoords = df_pred_pre["axis-1"].tolist()
                xcoords = df_pred_pre["axis-2"].tolist()
                pred_pre_coordinates = [[int(z*resolution[0]), int(y*resolution[1]), int(x*resolution[2])] for z, y, x in zip(zcoords, ycoords, xcoords)]

            for pre_id, coord in zip(df_pred_pre[pre_id_tag].tolist(), pred_pre_coordinates):
                pred_pre_map[pre_id] = coord

        # Read the predicted post coordinates from the CSV file
        pred_post_csv_path = os.path.join(pred_dir, f"{id}_pred_post_locations.csv")
        pred_post_map = {}
        if os.path.exists(pred_post_csv_path):
            df_pred_post = pd.read_csv(pred_post_csv_path)
            post_id_tag = "post_id" if "post_id" in df_pred_post.columns else "Post_ID"
            if "Post_Z" in df_pred_post.columns:
                zcoords = df_pred_post["Post_Z"].tolist()
                ycoords = df_pred_post["Post_Y"].tolist()
                xcoords = df_pred_post["Post_X"].tolist()
                pred_post_coordinates = [[int(z), int(y), int(x)] for z, y, x in zip(zcoords, ycoords, xcoords)]
            else:
                zcoords = df_pred_post["axis-0"].tolist()
                ycoords = df_pred_post["axis-1"].tolist()
                xcoords = df_pred_post["axis-2"].tolist()
                pred_post_coordinates = [[int(z*resolution[0]), int(y*resolution[1]), int(x*resolution[2])] for z, y, x in zip(zcoords, ycoords, xcoords)]

            for post_id, coord in zip(df_pred_post[post_id_tag].tolist(), pred_post_coordinates):
                pred_post_map[post_id] = coord

        # Read the predicted post coordinates from the CSV file
        pred_post_csv_path = os.path.join(pred_dir, f"{id}_pre_post_mapping.csv")
        pred_pairs = []
        # check the id to search
        pre_id_tag = "pre_id" if "pre_id" in df_pred_pre.columns else "Pre_ID"
        post_id_tag = "post_id" if "post_id" in df_pred_post.columns else "Post_ID"
        if os.path.exists(pred_post_csv_path):
            df_mapping = pd.read_csv(pred_post_csv_path)
            for _, row in df_mapping.iterrows():
                pre_id = int(row[pre_id_tag])
                post_id = int(row[post_id_tag])
                if pre_id != -1 and post_id != -1:
                    pre_point = pred_pre_map[pre_id]
                    post_point = pred_post_map[post_id]
                    pred_pairs.append((pre_point, post_point))
        else:
            print(f"WARNING: expected file {pred_post_csv_path} not found")

        # Compute metrics
        f1, prec, rec, tp, fp, fn = compute_metrics(pred_pairs, gt_pairs, args["matching_threshold"])

        print("File metrics: Precision: {:.4f}, Recall: {:.4f}, F1 Score: {:.4f} (TP: {}, FP: {}, FN: {})".format(prec, rec, f1, tp, fp, fn))

        run_metrics["Precision"].append(prec)
        run_metrics["Recall"].append(rec)
        run_metrics["F1"].append(f1)
        run_metrics["TP"].append(tp)
        run_metrics["FP"].append(fp)
        run_metrics["FN"].append(fn)

    return run_metrics


# Process each run and gather its overall (per-experiment) metrics, i.e. the totals averaged across
# the files of that run, exactly as previously printed as "Overall pairwise metrics".
per_run_overall = {
    "Precision": [], "Recall": [], "F1": [], "TP": [], "FP": [], "FN": []
}
for run_name in run_names:
    pred_dir = os.path.join(args['input_pred_dir'], run_name)
    run_metrics = process_run(pred_dir)

    run_precision = np.mean(run_metrics["Precision"])
    run_recall = np.mean(run_metrics["Recall"])
    run_f1 = np.mean(run_metrics["F1"])
    run_tp = np.mean(run_metrics["TP"])
    run_fp = np.mean(run_metrics["FP"])
    run_fn = np.mean(run_metrics["FN"])

    print(f"[{run_name}] Overall pairwise metrics:")
    print(f"    Precision: {run_precision:.4f}, Recall: {run_recall:.4f}, F1 Score: {run_f1:.4f} (TP: {run_tp}, FP: {run_fp}, FN: {run_fn})")

    per_run_overall["Precision"].append(run_precision)
    per_run_overall["Recall"].append(run_recall)
    per_run_overall["F1"].append(run_f1)
    per_run_overall["TP"].append(run_tp)
    per_run_overall["FP"].append(run_fp)
    per_run_overall["FN"].append(run_fn)

# Average (and std) of the per-run overall metrics across all runs
print(f"\nOverall pairwise metrics across {len(run_names)} runs ({', '.join(run_names)}):")
for metric in ["Precision", "Recall", "F1", "TP", "FP", "FN"]:
    mean_val = np.mean(per_run_overall[metric])
    std_val = np.std(per_run_overall[metric])
    print(f"    {metric}: {mean_val:.4f} +/- {std_val:.4f}")

print("FINISH!!")
