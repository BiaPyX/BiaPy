import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.measure import label as sklabel
from skimage.morphology import binary_dilation, disk
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN

# python agreement_manual_vs_dnn.py --input_file_dir /path/to/gt --input_instance_dir /path/to/preds_root

# -----------------------------
# Arguments
# -----------------------------
parser = argparse.ArgumentParser(
    description="Single Plotly figure (SVG): Manual vs DNN agreement across classes 0–5 with tolerance-dilated matching.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("--input_file_dir", "-input_file_dir", required=True,
                    help="Directory containing the Experto*/ XML folders")
parser.add_argument("--input_instance_dir", "-input_instance_dir", required=True,
                    help="Directory containing per-image instance masks (DNN output)")
parser.add_argument("--tolerance_px", type=int, default=8,
                    help="Pixel tolerance for BOTH click clustering and DNN-vs-GT matching (via dilation)")
parser.add_argument("--output_svg", default="agreement_manual_vs_dnn.svg",
                    help="Path/name for the SVG plot output")
parser.add_argument("--also_html", action="store_true",
                    help="Additionally save an interactive HTML")

args = vars(parser.parse_args())

from biapy.data.data_manipulation import read_img_as_ndarray

# -----------------------------
# Utilities
# -----------------------------
def read_expert_points_from_xml(root_dir: str) -> pd.DataFrame:
    """Read all expert XML files under 'Experto*' folders and return (folder, file, x, y)."""
    fids = sorted(next(os.walk(root_dir))[1])
    fids = [x for x in fids if "Experto" in x]

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
        raise RuntimeError("No XML markers found under Experto* folders.")

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
        dfc.groupby("cluster_id")["folder"]
        .nunique()
        .reset_index(name="n_experts")
    )
    pts_by_cluster = dfc.groupby("cluster_id")[["x", "y"]].apply(lambda g: g.to_numpy())
    n_by_cluster["points"] = n_by_cluster["cluster_id"].map(pts_by_cluster)
    return n_by_cluster


def make_single_figure(experts_df: pd.DataFrame,
                       nn_imgs: dict[str, np.ndarray],
                       out_dir: str,
                       tol_px: int,
                       out_svg_path: str,
                       also_html: bool):
    """
    Compute class distributions for Manual and DNN:
      - Manual: distribution by #experts (1..5). Class 0 = 0.
      - DNN:    hits per manual class (1..5) + class 0 = DNN-only objects.
    Matching uses tolerance by dilating each DNN instance with a 'disk(tol_px)' before point-in-mask tests.
    Only process images whose basenames appear in BOTH GT (XMLs) and predictions.
    Normalize per image to 100 manual neurons, then average across images.
    """
    # --- basenames present in DNN folder ---
    nn_base_to_file = {os.path.splitext(fname)[0]: fname for fname in nn_imgs.keys()}
    nn_basenames = set(nn_base_to_file.keys())

    # --- basenames present in GT (from expert XMLs) ---
    gt_basenames = set(experts_df["file"].unique())

    # --- intersection only ---
    common = sorted(nn_basenames.intersection(gt_basenames))
    if not common:
        raise RuntimeError("No overlapping images between GT (XMLs) and DNN predictions.")

    per_file_manual = []
    per_file_dnn = []

    # Structuring element for dilation (tolerance)
    selem = disk(max(1, int(tol_px))) if tol_px > 0 else None

    for base in tqdm(common, desc="Processing overlapping sections"):
        df_file = experts_df[experts_df["file"] == base]

        # ----- Manual clusters -----
        man_clusters = cluster_manual_points(df_file, eps=tol_px)
        manual_counts = {k: 0 for k in range(6)}
        for _, row in man_clusters.iterrows():
            k = int(row["n_experts"])
            k = max(1, min(5, k))
            manual_counts[k] += 1
        denom_manual = sum(manual_counts[k] for k in range(1, 6))
        if denom_manual == 0:
            continue  # nothing manual to normalize to

        # ----- DNN vs Manual mapping with dilation tolerance -----
        nn_mask = nn_imgs[nn_base_to_file[base]]
        H, W = nn_mask.shape
        nn_ids = [i for i in np.unique(nn_mask) if i != 0]

        # Precompute dilated masks for each DNN instance
        dilated_masks = {}
        for obj_id in nn_ids:
            m = (nn_mask == obj_id)
            mdil = binary_dilation(m, selem) if selem is not None else m
            dilated_masks[obj_id] = mdil

        # All manual clicks in this image (for class-0 check)
        xs_all = np.rint(df_file["x"].to_numpy()).astype(int)
        ys_all = np.rint(df_file["y"].to_numpy()).astype(int)
        xs_all = np.clip(xs_all, 0, W - 1) if xs_all.size else xs_all
        ys_all = np.clip(ys_all, 0, H - 1) if ys_all.size else ys_all

        dnn_counts = {k: 0 for k in range(6)}

        # Class 0 (DNN-only): DNN instances with no manual click within tol (i.e., inside dilated mask)
        dnn_only = 0
        for obj_id in nn_ids:
            mdil = dilated_masks[obj_id]
            if xs_all.size == 0 or not mdil[ys_all, xs_all].any():
                dnn_only += 1
        dnn_counts[0] = dnn_only

        # For classes 1..5: a manual cluster is "hit" if any of its points falls in any dilated DNN mask
        for _, row in man_clusters.iterrows():
            pts = row["points"]  # Nx2
            k = int(row["n_experts"])
            k = max(1, min(5, k))
            hit = False
            if pts.size and len(nn_ids) > 0:
                xs = np.clip(pts[:, 0].astype(int), 0, W - 1)
                ys = np.clip(pts[:, 1].astype(int), 0, H - 1)
                for obj_id in nn_ids:
                    if dilated_masks[obj_id][ys, xs].any():
                        hit = True
                        break
            if hit:
                dnn_counts[k] += 1

        # Normalize to 100 manual neurons (per image)
        manual_freqs = np.array([manual_counts[i] for i in range(6)], dtype=float) * (100.0 / denom_manual)
        manual_freqs[0] = 0.0  # no manual class 0
        dnn_freqs = np.array([dnn_counts[i] for i in range(6)], dtype=float) * (100.0 / denom_manual)

        per_file_manual.append(manual_freqs)
        per_file_dnn.append(dnn_freqs)

    if not per_file_manual:
        raise RuntimeError("No valid overlapping sections with manual objects found.")

    manual_mean = np.mean(np.stack(per_file_manual, axis=0), axis=0)
    dnn_mean = np.mean(np.stack(per_file_dnn, axis=0), axis=0)

    # -----------------------------
    # Plotly grouped bar chart
    # -----------------------------
    x = list(range(6))
    fig = go.Figure()
    fig.add_bar(name="Manual", x=x, y=manual_mean.tolist(),
                hovertemplate="Class %{x}<br>Manual: %{y:.2f}<extra></extra>")
    fig.add_bar(name="DNN", x=x, y=dnn_mean.tolist(),
                hovertemplate="Class %{x}<br>DNN: %{y:.2f}<extra></extra>")
    fig.update_layout(
        barmode="group",
        title=("Agreement among identification methods (Manual vs DNN)<br>"
               f"<sup>Matching uses dilation tolerance of {tol_px}px. "
               "Classes 0–5 = times identified in manual analyses (0 = DNN-only). "
               "Bars are mean frequency per 100 manual neurons.</sup>"),
        xaxis_title="Times identified as neurons in manual analyses (Class)",
        yaxis_title="Frequency (normalized)",
        bargap=0.2
    )

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)
    svg_path = os.path.join(out_dir, out_svg_path)
    try:
        fig.write_image(svg_path)  # requires kaleido
        print(f"✅ Saved SVG to: {svg_path}")
    except Exception as e:
        print("⚠️ Could not write SVG (is 'kaleido' installed?). Error:", repr(e))
        # Always save an HTML fallback
        html_path = os.path.splitext(svg_path)[0] + ".html"
        fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
        print(f"💾 Saved HTML fallback to: {html_path}")

    if also_html:
        html_path = os.path.splitext(svg_path)[0] + ".html"
        fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
        print(f"💾 Also saved HTML to: {html_path}")


# -----------------------------
# Main
# -----------------------------
def main():
    tol = int(args["tolerance_px"])
    file_dir = args["input_file_dir"]
    nn_dir = args["input_instance_dir"]
    out_svg_path = args["output_svg"]

    print("Reading expert XMLs …")
    experts_df = read_expert_points_from_xml(file_dir)

    print("Loading DNN instances …")
    nn_imgs = load_instance_masks(nn_dir)

    make_single_figure(
        experts_df=experts_df,
        nn_imgs=nn_imgs,
        out_dir=file_dir,
        tol_px=tol,
        out_svg_path=out_svg_path,
        also_html=bool(args["also_html"]),
    )


if __name__ == "__main__":
    main()
