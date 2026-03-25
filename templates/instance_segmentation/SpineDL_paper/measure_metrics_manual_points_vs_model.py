import argparse
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.measure import label as sklabel, regionprops
from skimage.draw import disk
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
import tifffile
import sys

sys.path.append("/data/dfranco/BiaPy") 

# Try importing biapy, handle gracefully if not present
try:
    from biapy.data.data_manipulation import read_img_as_ndarray
except ImportError:
    print("Warning: BiaPy not found. Make sure it is installed or adjust the sys.path.")

# -----------------------------
# Arguments
# -----------------------------
parser = argparse.ArgumentParser(
    description=(
        "Calculates Precision, Recall, and F1 across expert agreement classes 1-5. "
        "Uses Linear Sum Assignment (Hungarian algorithm) for BOTH expert clustering and DNN matching."
    ),
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--manual_annotation_dir",
    "-manual_annotation_dir",
    required=True,
    help="Directory containing the Manual_annotation*/ XML folders",
)
parser.add_argument(
    "--model_pred_folder",
    "-model_pred_folder",
    required=True,
    help="Directory containing per-image instance masks (SpineDL-Neuron output)",
)
parser.add_argument(
    "--out_folder",
    "-out_folder",
    required=True,
    help="Directory to save the resulting CSV metrics and debug images.",
)
parser.add_argument(
    "--tolerance_px",
    type=int,
    default=8,
    help="Pixel tolerance for BOTH LSA click clustering and LSA matching",
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="If set, generates TIFF files tracing GT, TP, FP, and FN per image and class.",
)

args = vars(parser.parse_args())

# -----------------------------
# Utilities
# -----------------------------

def read_expert_points_from_xml(root_dir: str) -> pd.DataFrame:
    fids = sorted(next(os.walk(root_dir))[1])
    fids = [x for x in fids if "Manual_annotation" in x]

    points, files, folders = [], [], []
    for id_ in tqdm(fids, desc="Scanning expert folders"):
        folder_path = os.path.join(root_dir, id_)
        ids = sorted(next(os.walk(folder_path))[2])
        ids = [x for x in ids if x.endswith(".xml")]
        for xml_name in ids:
            xml_path = os.path.join(folder_path, xml_name)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for marker in root.findall(".//Marker_Type/Marker"):
                x = int(marker.findtext("MarkerX"))
                y = int(marker.findtext("MarkerY"))
                points.append([x, y])
                files.append(xml_name.split("_")[-1].replace(".xml", ""))
                folders.append(os.path.basename(folder_path))

    if len(points) == 0:
        raise RuntimeError("No XML markers found under Manual_annotation* folders.")

    pts = np.array(points, dtype=int)
    return pd.DataFrame({"folder": folders, "file": files, "x": pts[:, 0], "y": pts[:, 1]})


def load_instance_masks(inst_dir: str) -> dict[str, np.ndarray]:
    ids = sorted(next(os.walk(inst_dir))[2])
    out = {}
    for id_ in tqdm(ids, desc=f"Reading instances from {os.path.basename(inst_dir)}"):
        arr = read_img_as_ndarray(os.path.join(inst_dir, id_)).squeeze()
        if arr.ndim != 2:
            raise ValueError(f"Instance mask {id_} is not 2D.")
        if np.max(arr) <= 1:
            arr = sklabel(arr > 0, connectivity=1)
        out[id_] = arr.astype(np.int32, copy=False)
    if not out:
        raise RuntimeError(f"No instance images found in {inst_dir}")
    return out


def draw_points(img_shape, points, color_val, img=None, radius=3):
    """Draws points as circles on a 2D or 3D (RGB) image canvas."""
    if img is None:
        if isinstance(color_val, tuple):
            img = np.zeros((*img_shape, 3), dtype=np.uint8)
        else:
            img = np.zeros(img_shape, dtype=np.uint8)
            
    for pt in points:
        y, x = int(round(pt[0])), int(round(pt[1]))
        # Draw a small disk so the point is visible
        rr, cc = disk((y, x), radius, shape=img_shape)
        if isinstance(color_val, tuple):
            for i, c in enumerate(color_val):
                img[rr, cc, i] = c
        else:
            img[rr, cc] = color_val
    return img

# -----------------------------
# LSA Clustering logic
# -----------------------------

def cluster_manual_points_lsa(df_file: pd.DataFrame, eps: int = 8) -> pd.DataFrame:
    if len(df_file) == 0:
        return pd.DataFrame(columns=["cluster_id", "n_experts", "points"])

    experts = sorted(df_file["folder"].unique())
    clusters = []
    
    for expert in experts:
        exp_df = df_file[df_file["folder"] == expert]
        exp_pts = exp_df[["x", "y"]].to_numpy(dtype=float)
        
        if len(clusters) == 0:
            for pt in exp_pts:
                clusters.append({"points": [pt], "experts": {expert}, "center": pt})
            continue
            
        cluster_centers = np.array([c["center"] for c in clusters])
        distances = distance_matrix(cluster_centers, exp_pts)
        cost_matrix = distances.copy()
        cost_matrix[distances > eps] = 1e6
        
        c_ind, p_ind = linear_sum_assignment(cost_matrix)
        matched_p_indices = set()
        
        for c_idx, p_idx in zip(c_ind, p_ind):
            if distances[c_idx, p_idx] <= eps:
                clusters[c_idx]["points"].append(exp_pts[p_idx])
                clusters[c_idx]["experts"].add(expert)
                clusters[c_idx]["center"] = np.mean(clusters[c_idx]["points"], axis=0)
                matched_p_indices.add(p_idx)
                
        for p_idx, pt in enumerate(exp_pts):
            if p_idx not in matched_p_indices:
                clusters.append({"points": [pt], "experts": {expert}, "center": pt})
                
    records = []
    for i, c in enumerate(clusters):
        records.append({
            "cluster_id": i,
            "n_experts": len(c["experts"]),
            "points": np.array(c["points"])
        })
        
    return pd.DataFrame(records)


# -----------------------------
# Core matching computation
# -----------------------------

def compute_metrics_for_image_lsa(nn_mask: np.ndarray, tol_px: int, man_clusters: pd.DataFrame):
    props = regionprops(nn_mask)
    pred_points = np.array([[p.centroid[1], p.centroid[0]] for p in props], dtype=np.float32)
    
    # Store counts AND points for debugging
    metrics = {k: {"TP": 0, "FP": 0, "FN": 0, "tp_pts": [], "fp_pts": [], "fn_pts": [], "gt_pts": []} for k in range(1, 6)}
    
    for k in range(1, 6):
        if man_clusters.empty:
            metrics[k]["FP"] = len(pred_points)
            metrics[k]["fp_pts"] = pred_points.tolist()
            continue
            
        valid_mask = man_clusters["n_experts"] >= k
        valid_clusters = man_clusters[valid_mask]
        
        if valid_clusters.empty:
            metrics[k]["FP"] = len(pred_points)
            metrics[k]["fp_pts"] = pred_points.tolist()
            continue
            
        true_points = np.array([pts.mean(axis=0) for pts in valid_clusters["points"]], dtype=np.float32)
        metrics[k]["gt_pts"] = true_points.tolist()
            
        _true = true_points
        _pred = pred_points
        
        if len(_true) == 0:
            metrics[k]["FP"] = len(_pred)
            metrics[k]["fp_pts"] = _pred.tolist()
            continue
        if len(_pred) == 0:
            metrics[k]["FN"] = len(_true)
            metrics[k]["fn_pts"] = _true.tolist()
            continue
            
        distances = distance_matrix(_pred, _true)
        cost_matrix = distances.copy()
        cost_matrix[distances > tol_px] = 1e6 
        
        pred_ind, true_ind = linear_sum_assignment(cost_matrix)
        
        matched_pred = set()
        matched_true = set()
        tp_pts, fp_pts, fn_pts = [], [], []
        
        for p_idx, t_idx in zip(pred_ind, true_ind):
            if distances[p_idx, t_idx] <= tol_px:
                # Assign the prediction point to True Positives (could also use true point)
                tp_pts.append(_pred[p_idx])
                matched_pred.add(p_idx)
                matched_true.add(t_idx)
                
        for p_idx, pt in enumerate(_pred):
            if p_idx not in matched_pred:
                fp_pts.append(pt)
                
        for t_idx, pt in enumerate(_true):
            if t_idx not in matched_true:
                fn_pts.append(pt)
        
        metrics[k]["TP"] = len(tp_pts)
        metrics[k]["FP"] = len(fp_pts)
        metrics[k]["FN"] = len(fn_pts)
        metrics[k]["tp_pts"] = tp_pts
        metrics[k]["fp_pts"] = fp_pts
        metrics[k]["fn_pts"] = fn_pts
        
    return metrics


def measure_metrics(experts_df: pd.DataFrame, nn_imgs_1: dict[str, np.ndarray], tol_px: int, out_folder: str, debug: bool):
    os.makedirs(out_folder, exist_ok=True)
    
    if debug:
        debug_dir = os.path.join(out_folder, "debug_images")
        os.makedirs(debug_dir, exist_ok=True)

    nn1_base_to_file = {os.path.splitext(fname)[0]: fname for fname in nn_imgs_1.keys()}
    gt_basenames = set(experts_df["file"].unique())

    common = sorted(gt_basenames.intersection(set(nn1_base_to_file.keys())))
    if not common:
        raise RuntimeError("No overlapping images between GT and predictions.")

    per_file_records = []
    for base in tqdm(common, desc="Processing overlapping sections"):
        df_file = experts_df[experts_df["file"] == base]
        nn_mask = nn_imgs_1[nn1_base_to_file[base]]
        H, W = nn_mask.shape
        
        man_clusters = cluster_manual_points_lsa(df_file, eps=tol_px)
        img_metrics = compute_metrics_for_image_lsa(nn_mask, tol_px, man_clusters)
        
        for k in range(1, 6):
            tp = img_metrics[k]["TP"]
            fp = img_metrics[k]["FP"]
            fn = img_metrics[k]["FN"]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
            recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            
            if pd.isna(precision) or pd.isna(recall) or (precision + recall) == 0:
                f1 = np.nan
            else:
                f1 = 2 * (precision * recall) / (precision + recall)

            per_file_records.append({
                "File": base,
                "Threshold": f"≥ {k} Experts",
                "K": k,
                "Precision": precision,
                "Recall": recall,
                "F1_Score": f1
            })
            
            # --- DEBUG TIFF GENERATION ---
            if debug:
                # 1) GT TIFF (Grayscale)
                gt_img = draw_points((H, W), img_metrics[k]["gt_pts"], color_val=255, radius=3)
                tifffile.imwrite(os.path.join(debug_dir, f"{base}_GT_class{k}.tif"), gt_img)
                
                # 2) Evaluation TIFF (RGB)
                # TP -> Green (0, 255, 0)
                # FP -> Blue  (0, 0, 255)
                # FN -> Red   (255, 0, 0)
                eval_img = draw_points((H, W), img_metrics[k]["tp_pts"], color_val=(0, 255, 0), radius=3)
                eval_img = draw_points((H, W), img_metrics[k]["fp_pts"], color_val=(0, 0, 255), img=eval_img, radius=3)
                eval_img = draw_points((H, W), img_metrics[k]["fn_pts"], color_val=(255, 0, 0), img=eval_img, radius=3)
                tifffile.imwrite(os.path.join(debug_dir, f"{base}_Eval_class{k}.tif"), eval_img)

    per_file_df = pd.DataFrame(per_file_records)
    
    summary_records = []
    for k in range(1, 6):
        k_df = per_file_df[per_file_df["K"] == k]
        
        p_mean, p_std = k_df["Precision"].mean(), k_df["Precision"].std()
        r_mean, r_std = k_df["Recall"].mean(), k_df["Recall"].std()
        f1_mean, f1_std = k_df["F1_Score"].mean(), k_df["F1_Score"].std()
        
        summary_records.append({
            "Consensus": f"≥ {k} Experts",
            "Precision (Macro)": f"{p_mean:.4f} ± {p_std:.4f}" if not pd.isna(p_std) else f"{p_mean:.4f} ± 0.0000",
            "Recall (Macro)": f"{r_mean:.4f} ± {r_std:.4f}" if not pd.isna(r_std) else f"{r_mean:.4f} ± 0.0000",
            "F1_Score (Macro)": f"{f1_mean:.4f} ± {f1_std:.4f}" if not pd.isna(f1_std) else f"{f1_mean:.4f} ± 0.0000",
        })

    summary_df = pd.DataFrame(summary_records)
    
    print("\n" + "="*80)
    print(f"OBJECT DETECTION MACRO METRICS (Dual LSA, Tolerance = {tol_px} px)")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80)
    
    summary_out = os.path.join(out_folder, "metrics_summary_macro.csv")
    per_file_out = os.path.join(out_folder, "metrics_per_file.csv")
    
    summary_df.to_csv(summary_out, index=False)
    per_file_df.to_csv(per_file_out, index=False)
    print(f"\n[+] Results successfully saved to: {out_folder}")
    if debug:
        print(f"[+] Debug images generated in: {debug_dir}")

# -----------------------------
# Main
# -----------------------------

def main():
    tol = int(args["tolerance_px"]) 
    file_dir = args["manual_annotation_dir"]
    nn_dir_1 = args["model_pred_folder"]
    out_dir = args["out_folder"]
    debug = args["debug"]

    print("Reading expert XMLs ...")
    experts_df = read_expert_points_from_xml(file_dir)

    print("Loading model predictions ...")
    nn_imgs_1 = load_instance_masks(nn_dir_1)

    measure_metrics(
        experts_df=experts_df,
        nn_imgs_1=nn_imgs_1,
        tol_px=tol,
        out_folder=out_dir,
        debug=debug
    )

if __name__ == "__main__":
    main()