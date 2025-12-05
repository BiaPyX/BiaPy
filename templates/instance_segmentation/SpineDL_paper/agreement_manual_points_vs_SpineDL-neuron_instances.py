import argparse
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.measure import label as sklabel
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN
import edt

# python -u agreement_manual_points_vs_SpineDL-neuron_instances.py --manual_annotation_dir /home/user/datasets/neuron_test --pred_folder /home/user/medular_lesion/results/medular_lesion_1/per_image_instances

# -----------------------------
# Arguments
# -----------------------------
parser = argparse.ArgumentParser(
    description=(
        "Single Plotly figure (SVG): Manual vs two DNN predictions agreement across classes 0–5 "
        "with tolerance-dilated matching."
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
    "--pred_folder",
    "-pred_folder",
    required=True,
    help="Directory containing predicted instance masks (SpineDL-Neuron output)",
)
parser.add_argument(
    "--tolerance_px",
    type=int,
    default=8,
    help="Pixel tolerance for BOTH click clustering and DNN-vs-GT matching (via dilation)",
)
parser.add_argument(
    "--output_svg",
    default="agreement_manual_vs_SpineDL.svg",
    help="Path/name for the SVG plot output",
)
# NEW: optional auxiliary folder for intermediates
parser.add_argument(
    "--aux_dir",
    default=None,
    help=(
        "Directory to store intermediate calculations (CSV cache of manual clusters). "
        "If not given, a folder named '_aux_cache' is created inside --manual_annotation_dir."
    ),
)

args = vars(parser.parse_args())

from biapy.data.data_manipulation import read_img_as_ndarray

# -----------------------------
# Utilities
# -----------------------------

def read_expert_points_from_xml(root_dir: str) -> pd.DataFrame:
    """Read all expert XML files under 'Manual_annotation*' folders and return (folder, file, x, y)."""
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
    """Load each image in 'inst_dir' as an instance-labeled 2D array (0=background, 1..N=instances)."""
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


def cluster_manual_points(df_file: pd.DataFrame, eps: int = 8) -> pd.DataFrame:
    """
    Cluster manual points (all experts) into 'manual objects' and count unique experts per object.
    Returns columns: cluster_id, n_experts, points (Nx2 array of the raw clicks in the cluster).
    """
    if len(df_file) == 0:
        return pd.DataFrame(columns=["cluster_id", "n_experts", "points"])
    coords = df_file[["x", "y"]].to_numpy(dtype=float)
    labels = DBSCAN(eps=eps, min_samples=1, metric="euclidean").fit_predict(coords)
    dfc = df_file.copy()
    dfc["cluster_id"] = labels
    n_by_cluster = (
        dfc.groupby("cluster_id")["folder"].nunique().reset_index(name="n_experts")
    )
    pts_by_cluster = dfc.groupby("cluster_id")[
        ["x", "y"]
    ].apply(lambda g: g.to_numpy())
    n_by_cluster["points"] = n_by_cluster["cluster_id"].map(pts_by_cluster)
    return n_by_cluster


# -----------------------------
# CSV cache helpers for clusters
# -----------------------------

def save_clusters_csv(cluster_df: pd.DataFrame, csv_path: str) -> None:
    """Save clusters as long-form CSV: one row per point with cluster_id, n_experts, x, y."""
    rows = []
    for _, row in cluster_df.iterrows():
        pts = row["points"]
        cid = int(row["cluster_id"])
        n_exp = int(row["n_experts"])
        if pts is None or len(pts) == 0:
            rows.append({"cluster_id": cid, "n_experts": n_exp, "x": np.nan, "y": np.nan})
        else:
            for (x, y) in pts:
                rows.append({"cluster_id": cid, "n_experts": n_exp, "x": int(x), "y": int(y)})
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def load_clusters_csv(csv_path: str) -> pd.DataFrame:
    """Load long-form CSV and reconstruct the compact cluster DataFrame with a 'points' ndarray."""
    df = pd.read_csv(csv_path)
    groups = []
    for cid, g in df.groupby("cluster_id"):
        # Drop NaNs safely
        gxy = g.dropna(subset=["x", "y"])
        pts = gxy[["x", "y"]].to_numpy(dtype=int)
        n_exp = int(g["n_experts"].iloc[0]) if len(g) else 0
        groups.append({"cluster_id": int(cid), "n_experts": n_exp, "points": pts})
    if not groups:
        return pd.DataFrame(columns=["cluster_id", "n_experts", "points"])
    return pd.DataFrame(groups)

def get_dist_to_clicks(df_file: pd.DataFrame, H: int, W: int) -> np.ndarray:
    """
    Distance (per pixel) to the nearest manual click (all experts merged).
    This is the same for dnn1 and dnn2 for a given image.
    """
    click_img = np.zeros((H, W), dtype=np.uint8)
    if len(df_file) > 0:
        xs = np.rint(df_file["x"].to_numpy()).astype(int)
        ys = np.rint(df_file["y"].to_numpy()).astype(int)
        xs = np.clip(xs, 0, W - 1)
        ys = np.clip(ys, 0, H - 1)
        click_img[ys, xs] = 1

    # Distance to nearest click: edt expects True=background, so pass ~clicks
    dist = edt.edt((click_img == 0).astype(np.uint8))
    return dist


# -----------------------------
# Core computation
# -----------------------------

def compute_dnn_freqs_for_file(
    df_file: pd.DataFrame,
    nn_mask: np.ndarray,
    tol_px: int,
    man_clusters: pd.DataFrame | None = None,
    dist_to_instances: np.ndarray | None = None,
    dist_to_clicks: np.ndarray | None = None,
):
    """Return per-class counts (0..5) for one DNN mask against manual clusters in df_file.

    If `man_clusters` is provided, it is used directly (and may come from CSV cache); otherwise
    clusters are computed on the fly.
    """
    H, W = nn_mask.shape
    nn_ids = [i for i in np.unique(nn_mask) if i != 0]
    assert dist_to_instances is not None, "dist_to_instances must be provided"
    assert dist_to_clicks is not None, "dist_to_clicks must be provided"

    # All manual clicks in this image (for class-0 check)
    xs_all = np.rint(df_file["x"].to_numpy()).astype(int)
    ys_all = np.rint(df_file["y"].to_numpy()).astype(int)
    xs_all = np.clip(xs_all, 0, W - 1) if xs_all.size else xs_all
    ys_all = np.clip(ys_all, 0, H - 1) if ys_all.size else ys_all

    # Manual clusters (cached or given)
    if man_clusters is None:
        man_clusters = cluster_manual_points(df_file, eps=tol_px)

    # Manual counts per class (for normalization/denominator)
    manual_counts = {k: 0 for k in range(6)}
    for _, row in man_clusters.iterrows():
        k = int(row["n_experts"])
        k = max(1, min(5, k))
        manual_counts[k] += 1
    denom_manual = sum(manual_counts[k] for k in range(1, 6))

    dnn_counts = {k: 0 for k in range(6)}

    # Class 0 (DNN-only): for each instance, check if ANY pixel of that instance
    # is within tol of a click. If not, it's DNN-only.
    for obj_id in nn_ids:
        mask_i = (nn_mask == obj_id)
        if mask_i.any():
            min_d = float(np.min(dist_to_clicks[mask_i]))
        else:
            min_d = np.inf
        if not np.isfinite(min_d) or min_d > tol_px:
            dnn_counts[0] += 1

    # Classes 1..5: a manual cluster is a "hit" if ANY of its points lies within tol
    # of ANY instance (using dist_to_instances at those point coordinates).
    for _, row in man_clusters.iterrows():
        pts = row["points"]
        k = int(row["n_experts"])
        k = max(1, min(5, k))
        hit = False
        if pts is not None and len(pts):
            xs = np.clip(pts[:, 0].astype(int), 0, W - 1)
            ys = np.clip(pts[:, 1].astype(int), 0, H - 1)
            if np.any(dist_to_instances[ys, xs] <= tol_px):
                hit = True
        if hit:
            dnn_counts[k] += 1

    return dnn_counts, denom_manual


def make_single_figure(
    experts_df: pd.DataFrame,
    nn_imgs_1: dict[str, np.ndarray],
    tol_px: int,
    out_svg_path: str,
):
    """
    Compute class distributions for Manual, SpineDL-Neuron:
      - Manual: distribution by #experts (1..5). Class 0 = 0.
      - DNN#1/2: hits per manual class (1..5) + class 0 = DNN-only objects.
    Matching uses tolerance by dilating each DNN instance with a 'disk(tol_px)' before point-in-mask tests.
    Only process images whose basenames appear in GT (XMLs) and in BOTH DNN prediction directories.
    Normalize per image to 100 manual neurons, then average across images.
    Uses CSV cache in `aux_dir` to avoid recomputing manual clustering.
    """
    # --- basenames present in DNN folders ---
    nn1_base_to_file = {os.path.splitext(fname)[0]: fname for fname in nn_imgs_1.keys()}
    nn1_basenames = set(nn1_base_to_file.keys())

    # --- basenames present in GT (from expert XMLs) ---
    gt_basenames = set(experts_df["file"].unique())

    # --- intersection only (GT ∩ DNN1) ---
    common = sorted(gt_basenames.intersection(nn1_basenames))
    if not common:
        raise RuntimeError("No overlapping images between GT (XMLs) and both DNN prediction folders.")

    per_file_manual = []
    per_file_dnn1 = []

    for base in tqdm(common, desc="Processing overlapping sections"):
        df_file = experts_df[experts_df["file"] == base]

        # ----- Manual clusters (with CSV cache) -----
        man_clusters = cluster_manual_points(df_file, eps=tol_px)

        # Manual counts (1..5) for normalization
        manual_counts = {k: 0 for k in range(6)}
        for _, row in man_clusters.iterrows():
            k = int(row["n_experts"])
            k = max(1, min(5, k))
            manual_counts[k] += 1
        denom_manual = sum(manual_counts[k] for k in range(1, 6))
        if denom_manual == 0:
            continue  # nothing manual to normalize to

        manual_freqs = np.array([manual_counts[i] for i in range(6)], dtype=float)
        manual_freqs[0] = 0.0  # no manual class 0
        per_file_manual.append(manual_freqs)

        H, W = nn_imgs_1[nn1_base_to_file[base]].shape

        # Distance to clicks (same for both models for this image)
        dist2clicks = get_dist_to_clicks(df_file, H, W)

        # ----- SpineDL-Neuron #1 vs Manual -----
        nn_mask_1 = nn_imgs_1[nn1_base_to_file[base]]
        dist2inst_1 = edt.edt((nn_mask_1 == 0).astype(np.uint8))
        dnn1_counts, _ = compute_dnn_freqs_for_file(
            df_file, nn_mask_1, tol_px,
            man_clusters=man_clusters,
            dist_to_instances=dist2inst_1,
            dist_to_clicks=dist2clicks,
        )
        dnn1_freqs = np.array([dnn1_counts[i] for i in range(6)], dtype=float)
        per_file_dnn1.append(dnn1_freqs)

    if not per_file_manual:
        raise RuntimeError("No valid overlapping sections with manual objects found.")

    manual_mean = np.mean(np.stack(per_file_manual, axis=0), axis=0)
    dnn1_mean = np.mean(np.stack(per_file_dnn1, axis=0), axis=0)

    # -----------------------------
    # Plotly grouped bar chart
    # -----------------------------
    x = list(range(6))
    fig = go.Figure()
    fig.add_bar(
        name="Manual",
        x=x,
        y=manual_mean.tolist(),
        text=[f"{v:.1f}" for v in manual_mean], 
        textfont=dict(size=10),
        textposition="outside",
        hovertemplate="Class %{x}<br>Manual: %{y:.2f}<extra></extra>",
    )
    fig.add_bar(
        name="SpineDL-Neuron",
        x=x,
        y=dnn1_mean.tolist(),
        text=[f"{v:.1f}" for v in dnn1_mean], 
        textfont=dict(size=10),
        textposition="outside",
        hovertemplate="Class %{x}<br>SpineDL-Neuron: %{y:.2f}<extra></extra>",
    )
    fig.update_layout(
        barmode="group",
        title=(
            "Agreement among identification methods<br>(Manual vs SpineDL-Neuron)"
        ),
        xaxis_title="Times identified as neurons in manual analyses",
        yaxis_title="Frequency",
        bargap=0.2,
    )

    # Ensure output directory exists
    fig.write_image(out_svg_path)  # requires kaleido
    print(f"✅ Saved SVG to: {out_svg_path}")

# -----------------------------
# Main
# -----------------------------

def main():
    tol = int(args["tolerance_px"]) 
    file_dir = args["manual_annotation_dir"]
    nn_dir_1 = args["pred_folder"]
    out_svg_path = args["output_svg"]

    print("Reading expert XMLs …")
    experts_df = read_expert_points_from_xml(file_dir)

    print("Loading SpineDL-Neuron #1 instances …")
    nn_imgs_1 = load_instance_masks(nn_dir_1)

    make_single_figure(
        experts_df=experts_df,
        nn_imgs_1=nn_imgs_1,
        tol_px=tol,
        out_svg_path=out_svg_path,
    )


if __name__ == "__main__":
    main()
