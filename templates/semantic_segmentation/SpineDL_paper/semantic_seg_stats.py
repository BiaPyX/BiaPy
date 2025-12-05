import os
import argparse
from glob import glob
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from biapy.data.data_manipulation import read_img_as_ndarray, save_tif

# ---------------------- IO HELPERS -----------------------
def load_masks(folder):
    """Load segmentation masks from a folder into a dict {filename: np.array}."""
    masks = {}
    # Accept multiple common extensions; change as needed
    exts = ("*.tif", "*.tiff", "*.png")
    files = []
    for e in exts:
        files.extend(sorted(glob(os.path.join(folder, e))))
    for f in sorted(files):
        name = os.path.basename(f)
        masks[name] = read_img_as_ndarray(f, is_3d=False).squeeze()
    return masks

def intersect_keys(dicts):
    keys = set(dicts[0].keys())
    for d in dicts[1:]:
        keys &= set(d.keys())
    return sorted(keys)

def valid_pixels(mask, void_value):
    return np.ones_like(mask, dtype=bool) if void_value is None else (mask != void_value)

# ---------------------- METRICS (DATASET-LEVEL) --------------------------
def dataset_level_iou(masks_a, masks_b, keys, num_classes, void_label=None):
    """
    Aggregate intersections and unions over **all images** before computing IoU.
    Returns (per_class_iou, mean_iou).

    IoU handling:
      - If union_c == 0 (class absent in both A and B across the dataset): IoU_c = NaN (ignored in mean).
      - If GT has zero but Pred has >0 (hallucinated class): union_c > 0 so IoU_c = 0 → penalized.
    """
    inter = np.zeros(num_classes, dtype=np.float64)
    union = np.zeros(num_classes, dtype=np.float64)

    for k in keys:
        A, B = masks_a[k], masks_b[k]
        if A.shape != B.shape:
            raise ValueError(f"Size mismatch for {k}: {A.shape} vs {B.shape}")
        valid = valid_pixels(A, void_label) & valid_pixels(B, void_label)

        for c in range(num_classes):
            a = (A == c) & valid
            b = (B == c) & valid
            inter[c] += np.logical_and(a, b).sum()
            union[c] += np.logical_or(a, b).sum()

    per_class = np.array([np.nan if union[c] == 0 else inter[c] / union[c] for c in range(num_classes)], dtype=float)
    mean_iou = float(np.nanmean(per_class)) if np.any(np.isfinite(per_class)) else float("nan")
    return per_class, mean_iou

# ----- utils for lower-triangular heatmaps with per-cell text -----
def lower_triangular_matrix(Z):
    """Return (Zmasked, text) where upper triangle is NaN/empty and diagonal kept."""
    Z = np.array(Z, dtype=float)
    n = Z.shape[0]
    mask = np.tril(np.ones_like(Z, dtype=bool))  # lower tri incl diag
    Zm = np.where(mask, Z, np.nan)

    text = np.empty_like(Z, dtype=object)
    for i in range(n):
        for j in range(n):
            if mask[i, j] and np.isfinite(Z[i, j]):
                text[i, j] = f"{Z[i, j]:.2f}"
            else:
                text[i, j] = ""
    return Zm, text

# ---------------------- CONSENSUS (UNANIMOUS) ----------------------
def build_unanimous_consensus(experts, keys, void_value=0):
    """
    Create a per-image consensus mask where a pixel keeps a class ID
    only if ALL experts agree on that class. Otherwise, the pixel is set
    to 'void_value' and will be ignored during IoU computation.
    Returns dict {filename: np.array}.
    """
    consensus = {}
    for k in keys:
        # shape: (E, H, W)
        stack = np.stack([E[k] for E in experts], axis=0)
        # all experts agree per pixel?
        agree = np.all(stack == stack[0], axis=0)
        # where they agree, take that class; else void
        consensus[k] = np.where(agree, stack[0], void_value).astype(stack.dtype)
    return consensus

def save_iou_agreement_maps(preds, consensus_gt, keys, out_dir, void_value=255):
    """
    For each image, save an RGB map with:
      - TP (green): pred == gt != void
      - Errors (red): pred != gt & gt != void
    Void pixels and all others are black.

    preds         : dict {filename: np.uint8 HxW}  (model predictions)
    consensus_gt  : dict {filename: np.uint8 HxW}  (unanimous consensus GT)
    keys          : iterable of filenames
    out_dir       : output directory for PNGs
    void_value    : value used for 'void' in consensus_gt
    """
    os.makedirs(out_dir, exist_ok=True)

    GREEN = np.array([0, 255, 0], dtype=np.uint8)  # TP
    RED   = np.array([255, 0, 0], dtype=np.uint8)  # mismatch/error
    BLACK = np.array([0, 0, 0], dtype=np.uint8)    # void / background

    for k in keys:
        pred = preds[k]
        gt   = consensus_gt[k]

        # Mask valid (non-void) pixels
        valid = gt != void_value
        tp = (pred == gt) & valid
        err = (pred != gt) & valid

        h, w = gt.shape
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[:] = BLACK

        canvas[tp] = GREEN
        canvas[err] = RED

        out_path = os.path.join(out_dir, k)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        Image.fromarray(canvas).save(out_path)


# ---------------------- SAVE CONSENSUS GTs ----------------------
def save_consensus_gt(consensus_dict, out_dir, color_map=None):
    """
    Save each unanimous-consensus GT mask to disk as a PNG.

    consensus_dict : dict {filename: np.array}
    out_dir        : path to the directory where images will be saved
    color_map      : optional (H, W, 3) LUT or callable to colorize masks
    """
    os.makedirs(out_dir, exist_ok=True)

    for name, mask in consensus_dict.items():
        # save_tif expects 5D data (1, D, H, W, 1) or 4D (1, H, W, 1)
        save_tif(np.expand_dims(np.expand_dims(mask, 0), -1), out_dir, [name], verbose=True)


# ---------------------- ARGPARSE CONFIG -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Agreement analysis for semantic segmentation using dataset-level IoU.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # --- INPUT/OUTPUT Paths ---
    parser.add_argument(
        "--pred_folder", type=str, required=True,
        help="Path to the folder containing model predictions (segmentation masks).",
        default="/data/dfranco/exp_results/lesion_medular37/per_image_binarized",
        metavar="PRED_FOLDER"
    )
    parser.add_argument(
        "--experts_root", type=str, required=True,
        help="Root folder containing N subfolders, one per expert's annotations.",
        default="/data/dfranco/datasets/lesion_medular/semantic_seg/V11/test/label",
        metavar="EXPERTS_ROOT"
    )

    # --- Output Filenames (SVGs) ---
    parser.add_argument(
        "--output_bars_svg", type=str, default="agreement_bars.svg",
        help="Output filename for the Model vs Experts bar plot (SVG).",
        metavar="FILE"
    )
    parser.add_argument(
        "--output_overall_svg", type=str, default="agreement_overall_matrix.svg",
        help="Output filename for the overall agreement heatmap matrix (SVG).",
        metavar="FILE"
    )
    parser.add_argument(
        "--output_classes_svg", type=str, default="agreement_class_matrices.svg",
        help="Output filename for the per-class agreement heatmap matrices (SVG).",
        metavar="FILE"
    )
    parser.add_argument(
        "--output_consensus_svg", type=str, default="agreement_consensus_unanimous_perclass.svg",
        help="Output filename for the Model vs Unanimous-Consensus bar plot (SVG).",
        metavar="FILE"
    )
    parser.add_argument(
        "--output_consensus_dir", type=str, default="consensus_unanimous_gt",
        help="Output directory for saving unanimous-consensus GT masks.",
        metavar="DIR"
    )
    parser.add_argument(
        "--output_iou_agreement_dir", type=str, default="consensus_unanimous_iou_maps",
        help="Output directory for saving IoU agreement maps (TP/Errors in RGB).",
        metavar="DIR"
    )

    # --- Class Configuration ---
    parser.add_argument(
        "--class_names", nargs='+', required=False,
        default=["background", "gray matter", "white matter", "ependyma", "damaged region"],
        help="List of class names, in order of their class IDs (e.g., [ID 0, ID 1, ...]).",
        metavar="NAME"
    )
    parser.add_argument(
        "--void_label", type=int, default=None,
        help="Optional: Pixel value used for void/ignore label in masks (e.g., 255). Default is None.",
        metavar="ID"
    )
    parser.add_argument(
        "--background_idx", type=int, default=0,
        help="Index of the background class in CLASS_NAMES, used for excluding it from some plots (e.g., 0).",
        metavar="ID"
    )
    parser.add_argument(
        "--consensus_void_label", type=int, default=0,
        help="Pixel value to use for void when building the unanimous consensus GT mask.",
        metavar="ID"
    )

    args = parser.parse_args()

    # Derived variable
    args.num_classes = len(args.class_names)

    return args

# ---------------------- MAIN -----------------------------
def main():
    args = parse_args()

    # Load predictions & experts
    preds = load_masks(args.pred_folder)
    expert_dirs = sorted([p for p in glob(os.path.join(args.experts_root, "*")) if os.path.isdir(p)])
    if len(expert_dirs) < 2:
        raise RuntimeError(f"Expected >=2 expert subfolders in '{args.experts_root}', found {len(expert_dirs)}.")
    experts = [load_masks(d) for d in expert_dirs]

    # Harmonize filenames
    all_sets = [preds] + experts
    keys = intersect_keys(all_sets)
    if not keys:
        raise RuntimeError("No common filenames across model and experts.")
    preds   = {k: preds[k] for k in keys}
    experts = [{k: E[k] for k in keys} for E in experts]

    names = ["Model"] + [f"Expert {i+1}" for i in range(len(experts))]
    N = len(names)

    # (1) Model vs each expert (overall) + (2) per-class averaged across experts) — DATASET-LEVEL
    model_vs_expert_overall = []
    model_vs_expert_perclass = []
    for E in experts:
        per_c, overall = dataset_level_iou(preds, E, keys, args.num_classes, void_label=args.void_label)
        model_vs_expert_overall.append(overall)
        model_vs_expert_perclass.append(per_c)
    model_vs_expert_perclass = np.vstack(model_vs_expert_perclass)  # (E, C)
    mean_perclass_model = np.nanmean(model_vs_expert_perclass, axis=0)

    # (3) Agreement matrices among all annotators (Model + Experts) — DATASET-LEVEL
    annotators = [preds] + experts
    overall_mat = np.full((N, N), np.nan)
    perclass_mats = [np.full((N, N), np.nan) for _ in range(args.num_classes)]
    for i in range(N):
        for j in range(N):
            if i == j:
                overall_mat[i, j] = 1.0
                for c in range(args.num_classes):
                    perclass_mats[c][i, j] = 1.0
            elif j > i:
                per_c, overall = dataset_level_iou(annotators[i], annotators[j], keys, args.num_classes, void_label=args.void_label)
                overall_mat[i, j] = overall
                overall_mat[j, i] = overall
                for c in range(args.num_classes):
                    perclass_mats[c][i, j] = per_c[c]
                    perclass_mats[c][j, i] = per_c[c]

    overall_Z, overall_text = lower_triangular_matrix(overall_mat)
    perclass_Z_text = [lower_triangular_matrix(pm) for pm in perclass_mats]

    # ---------------------- CONSENSUS (UNANIMOUS) EVALUATION ----------------------
    # Build new GT by keeping only pixels where all experts agree, void elsewhere
    consensus_gt = build_unanimous_consensus(experts, keys, void_value=args.consensus_void_label)

    # Compute dataset-level IoU (per class and overall) between Model and consensus GT
    cons_per_class, cons_overall = dataset_level_iou(
        preds, consensus_gt, keys, args.num_classes, void_label=args.consensus_void_label
    )

    # Save the consensus GT masks to disk
    save_consensus_gt(consensus_gt, args.output_consensus_dir)
    print(f"Saved unanimous-consensus GT masks to: {args.output_consensus_dir}")

    # ---------------------- FIGURE 1: BARS ----------------------
    fig1 = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Model vs Experts<br>(Overall mIoU — dataset-level)",
                        "Model vs Experts<br>(Per-Class mIoU — dataset-level)"],
        horizontal_spacing=0.18
    )

    # Overall bar
    fig1.add_trace(
        go.Bar(
            x=[f"Expert {i+1}" for i in range(len(experts))],
            y=model_vs_expert_overall,
            text=[f"{v:.2f}" if np.isfinite(v) else "NA" for v in model_vs_expert_overall],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="Overall mIoU: %{y:.3f}<extra>%{x}</extra>",
        ),
        row=1, col=1
    )
    fig1.update_xaxes(tickangle=45, automargin=True, row=1, col=1)
    fig1.update_yaxes(title_text="mIoU", title_standoff=8, range=[0, 1], automargin=True, row=1, col=1)

    # Per-class bar
    fig1.add_trace(
        go.Bar(
            x=args.class_names,
            y=mean_perclass_model,
            text=[f"{v:.2f}" if np.isfinite(v) else "NA" for v in mean_perclass_model],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="mIoU: %{y:.3f}<extra>%{x}</extra>",
        ),
        row=1, col=2
    )
    fig1.update_xaxes(tickangle=45, automargin=True, row=1, col=2)
    fig1.update_yaxes(title_text="mIoU", title_standoff=8, range=[0, 1], automargin=True, row=1, col=2)

    fig1.update_layout(
        width=1800, height=650,
        margin=dict(l=60, r=50, t=110, b=70),
        template="plotly_white",
        bargap=0.25,
        showlegend=False,
    )
    for a in fig1.layout.annotations:
        a.font.size = 18
        a.yshift = 12
        a.align = "center"

    fig1.write_image(args.output_bars_svg, format="svg")
    print(f"Saved: {args.output_bars_svg}")

    # ---------------------- FIGURE 2: OVERALL MATRIX ----------------------
    fig2 = go.Figure()
    fig2.add_trace(
        go.Heatmap(
            z=overall_Z, x=names, y=names,
            zmin=0, zmax=1, coloraxis="coloraxis",
            hovertemplate="A: %{y}<br>B: %{x}<br>mIoU: %{z:.3f}<extra></extra>",
            text=overall_text,
            texttemplate="%{text}",
            textfont=dict(size=18),
            zsmooth=False
        )
    )
    fig2.update_xaxes(title_text="", tickangle=35, automargin=True)
    fig2.update_yaxes(title_text="", automargin=True)

    fig2.update_layout(
        title={"text": "Agreement Matrix (dataset-level mIoU)", "x": 0.5, "y": 0.95, "xanchor": "center"},
        coloraxis=dict(
            colorscale="Viridis",
            cmin=0, cmax=1,
            colorbar=dict(title="mIoU", thickness=16, len=0.85)
        ),
        width=950, height=900,
        margin=dict(l=90, r=80, t=120, b=80),
        template="plotly_white",
        showlegend=False
    )
    fig2.write_image(args.output_overall_svg, format="svg")
    print(f"Saved: {args.output_overall_svg}")

    # ---------------------- FIGURE 3: PER-CLASS MATRICES -------------------
    # Use the first 5 class names for subplot titles
    CLASS_TITLES = [name.replace('<br>', ' ') for name in args.class_names]
    # Adjust specs based on the number of classes (up to 6 classes, 2 rows, 3 cols)
    n_plots = args.num_classes
    n_rows = (n_plots + 2) // 3
    specs = [[{"type": "heatmap"}] * 3 for _ in range(n_rows)]
    while len(specs[0]) > 3: # Should not happen based on n_plots, but defensive programming
        specs[0].pop()

    # If the last row is partial, fill with "domain" (empty plot) slots
    last_row_len = n_plots % 3
    if last_row_len == 1:
        specs[-1].pop()
        specs[-1].pop()
        specs[-1].extend([{"type": "domain"}, {"type": "domain"}])
    elif last_row_len == 2:
        specs[-1].pop()
        specs[-1].extend([{"type": "domain"}])

    subplot_titles = CLASS_TITLES[:n_plots] + [""] * (n_rows * 3 - n_plots)

    fig3 = make_subplots(
        rows=n_rows, cols=3,
        specs=specs,
        subplot_titles=subplot_titles,
        shared_yaxes=True,
        horizontal_spacing=0.12,
        vertical_spacing=0.20,
        column_widths=[0.33, 0.33, 0.34],
    )

    # Prepare data for class heatmaps
    positions = [(r+1, c+1) for r in range(n_rows) for c in range(3)]
    for idx, (Z, T) in enumerate(perclass_Z_text):
        if idx >= n_plots: break # should not happen if perclass_Z_text is correct length
        r, c = positions[idx]
        fig3.add_trace(
            go.Heatmap(
                z=Z, x=names, y=names,
                zmin=0, zmax=1, coloraxis="coloraxis",
                hovertemplate="A: %{y}<br>B: %{x}<br>mIoU: %{z:.3f}<extra></extra>",
                text=T,
                texttemplate="%{text}",
                textfont=dict(size=13),
                zsmooth=False
            ),
            row=r, col=c
        )
        fig3.update_xaxes(tickangle=35, tickfont_size=13, automargin=True, row=r, col=c)
        fig3.update_yaxes(showticklabels=(c == 1), tickfont_size=13, automargin=True, row=r, col=c)

    for r in range(1, n_rows + 1):
        for c in range(1, 4):
            # Update shared y-axes, skip empty domain plots
            if r == n_rows and c > last_row_len and last_row_len != 0:
                continue
            if not (r == 1 and c == 1):
                fig3.update_yaxes(matches="y", row=r, col=c)

    fig3.update_layout(
        title={"text": "Agreement Matrix (per-class, dataset-level)", "x": 0.5, "y": 0.98, "xanchor": "center"},
        coloraxis=dict(
            colorscale="Viridis",
            cmin=0, cmax=1,
            colorbar=dict(title="IoU", thickness=18, len=0.86, x=1.02, y=0.5, yanchor="middle")
        ),
        width=2200,
        height=1400,
        margin=dict(l=90, r=120, t=120, b=90),
        template="plotly_white",
        showlegend=False
    )

    for a in fig3.layout.annotations:
        a.font.size = 16
        a.yshift = 10
        a.align = "center"

    fig3.write_image(args.output_classes_svg, format="svg")
    print(f"Saved: {args.output_classes_svg}")

    # ---------------------- FIGURE 4: MODEL vs CONSENSUS (UNANIMOUS) PER-CLASS ----------------------

    # Filter out the background class
    class_names_no_bg = [c for i, c in enumerate(args.class_names) if i != args.background_idx]
    ious_no_bg = [v for i, v in enumerate(cons_per_class) if i != args.background_idx]

    # Hardcoded colors (can be moved to args or an external config if needed)
    color_map = {
        "background": "#000000",
        "gray matter": "#009898",
        "white matter": "#f2e800",
        "ependyma": "#cb579a",
        "damaged region": "#b74c34",
    }
    default_color = "#7f7f7f"

    fig4 = go.Figure()
    for cls, iou in zip(class_names_no_bg, ious_no_bg):
        text_val = f"{iou:.2f}" if np.isfinite(iou) else "NA"
        color = color_map.get(cls, default_color)

        fig4.add_trace(
            go.Bar(
                x=[cls],
                y=[iou],
                name=cls,
                marker=dict(color=color),
                width=0.2,
                text=[text_val],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="IoU: %{y:.3f}<extra>%{x}</extra>",
            )
        )

    fig4.update_xaxes(title_text="", tickangle=45, automargin=True)
    fig4.update_yaxes(
        title_text="IoU (dataset-level, consensus GT)",
        range=[0, 1],
        automargin=True
    )

    fig4.update_layout(
        title={
            "text": "Model vs Unanimous-Consensus GT — Per-Class IoU",
            "x": 0.5, "y": 0.95, "xanchor": "center",
        },
        width=1200,
        height=600,
        margin=dict(l=70, r=60, t=110, b=90),
        template="plotly_white",
        showlegend=True,
        legend=dict(
            orientation="v",
            x=1.02,
            xanchor="left",
            y=1,
            yanchor="top",
        ),
    )

    fig4.write_image(args.output_consensus_svg, format="svg")
    print(f"Saved: {args.output_consensus_svg}")

    save_iou_agreement_maps(
        preds, consensus_gt, keys, args.output_iou_agreement_dir, void_value=args.consensus_void_label
    )
    print(f"Saved IoU agreement maps to: {args.output_iou_agreement_dir}")


if __name__ == "__main__":
    main()