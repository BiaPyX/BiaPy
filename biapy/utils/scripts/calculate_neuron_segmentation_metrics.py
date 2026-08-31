import os
import sys
import itertools
import argparse
import numpy as np
import networkx as nx
from skimage.io import imread
from skimage.metrics import variation_of_information, adapted_rand_error
from skimage.morphology import skeletonize

# Make the BiaPy package importable (this script lives in biapy/utils/scripts/)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

# Regular image extensions supported for both predictions and ground truth
IMAGE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".gif")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate neuron segmentation metrics (VOI, Adapted Rand error and ERL) "
                    "between predictions and ground truth, as commonly used in connectomics "
                    "benchmarks such as SNEMI3D/CREMI.")

    parser.add_argument("input_dir", help="Directory containing the prediction label files.")
    parser.add_argument("gt_dir", help="Directory containing the ground truth label files.")
    parser.add_argument("--spacing", type=float, nargs="+", default=[1.0, 1.0, 1.0],
                        help="Voxel spacing used to weight skeleton edge lengths when computing "
                             "ERL. Give one value per data dimension (e.g. 'z y x' for 3D data, "
                             "'y x' for 2D data). Default: 1 1 1")
    parser.add_argument("--skip-erl", action="store_true",
                        help="Skip ERL computation. Skeletonizing every ground truth object can "
                             "be slow on large volumes with many objects.")
    parser.add_argument("--verbose", action="store_true", help="Print per-file statistics.")

    return parser.parse_args()


def list_images(directory):
    return sorted(f for f in next(os.walk(directory))[2] if f.lower().endswith(IMAGE_EXTENSIONS))


def build_gt_skeleton_graphs(gt, spacing):
    """
    Skeletonize every labeled object in ``gt`` and turn each skeleton into a
    networkx graph whose edges connect neighboring skeleton voxels, weighted
    by their physical (spacing-aware) Euclidean distance.
    """
    spacing = np.asarray(spacing, dtype=np.float64)
    offsets = [o for o in itertools.product([-1, 0, 1], repeat=gt.ndim) if any(o)]

    graphs = {}
    for lbl in np.unique(gt):
        if lbl == 0:
            continue
        obj_mask = gt == lbl
        if obj_mask.sum() < 2:
            continue

        skel = skeletonize(obj_mask)
        coords = {tuple(c) for c in np.argwhere(skel)}
        if len(coords) < 2:
            continue

        G = nx.Graph()
        G.add_nodes_from(coords)
        for c in coords:
            for off in offsets:
                n = tuple(a + b for a, b in zip(c, off))
                # only add each edge once (n > c relies on tuple ordering being total)
                if n > c and n in coords:
                    length = np.linalg.norm(np.asarray(off, dtype=np.float64) * spacing)
                    G.add_edge(c, n, length=length)
        graphs[lbl] = G

    return graphs


def compute_erl(gt_skeleton_graphs, pred):
    """
    Expected Run Length (ERL), as used in the SNEMI3D/CREMI connectomics
    benchmarks. Ground truth neuron skeletons are walked edge by edge; an
    edge is considered "broken" if its endpoints fall into different
    predicted segments, into background, or into a predicted segment that
    also covers a *different* ground truth neuron (a merge error). ERL is
    then the expected length of the correctly-reconstructed skeleton
    fragment containing a uniformly (length-weighted) random point.
    """
    # Predicted labels that overlap more than one ground truth skeleton are merge errors:
    # any edge mapped to one of them must be counted as broken, wherever it lies.
    pred_to_gt_objects = {}
    for gt_lbl, G in gt_skeleton_graphs.items():
        for node in G.nodes:
            p = int(pred[node])
            if p == 0:
                continue
            pred_to_gt_objects.setdefault(p, set()).add(gt_lbl)
    merged_pred_labels = {p for p, gts in pred_to_gt_objects.items() if len(gts) > 1}

    total_length = 0.0
    total_score = 0.0
    per_object = {}
    for gt_lbl, G in gt_skeleton_graphs.items():
        obj_length = sum(d["length"] for _, _, d in G.edges(data=True))
        if obj_length == 0:
            continue

        H = G.copy()
        for u, v in G.edges():
            pu, pv = int(pred[u]), int(pred[v])
            if pu == 0 or pv == 0 or pu != pv or pu in merged_pred_labels:
                H.remove_edge(u, v)

        obj_score = 0.0
        for comp in nx.connected_components(H):
            comp_length = sum(d["length"] for _, _, d in H.subgraph(comp).edges(data=True))
            obj_score += comp_length ** 2

        total_length += obj_length
        total_score += obj_score
        per_object[int(gt_lbl)] = {
            "length": obj_length,
            "erl": obj_score / obj_length if obj_length > 0 else 0.0,
        }

    erl = total_score / total_length if total_length > 0 else 0.0
    return erl, total_length, total_score, per_object


def evaluate_dataset(input_dir, gt_dir, spacing=(1.0, 1.0, 1.0), skip_erl=False, verbose=False):
    """
    Compute VOI, Adapted Rand error and (optionally) ERL for every prediction
    in `input_dir` against its matching (same base name) ground truth in
    `gt_dir`. Returns (per_file_results, dataset_erl) where per_file_results
    is a list of per-file metric dicts and dataset_erl is the skeleton-length
    -weighted ERL over the whole dataset (None if skip_erl).
    """
    gt_by_stem = {os.path.splitext(f)[0]: f for f in list_images(gt_dir)}
    ids = list_images(input_dir)

    per_file_results = []
    erl_total_length = 0.0
    erl_total_score = 0.0

    for id_ in ids:
        stem = os.path.splitext(id_)[0]
        if stem not in gt_by_stem:
            raise FileNotFoundError(
                "No ground truth image matching prediction '{}' was found in {} "
                "(looked for base name '{}').".format(id_, gt_dir, stem))

        pred = imread(os.path.join(input_dir, id_)).astype(np.int64)
        gt = imread(os.path.join(gt_dir, gt_by_stem[stem])).astype(np.int64)

        print(" ")
        print("#######################################")
        print("Analizing file {} (GT: {})".format(
            os.path.join(input_dir, id_), os.path.join(gt_dir, gt_by_stem[stem])))

        if len(spacing) != gt.ndim:
            raise ValueError(
                "spacing has {} values but data is {}D ({}). Provide one value per dimension.".format(
                    len(spacing), gt.ndim, id_))

        voi_split, voi_merge = variation_of_information(gt, pred)
        are, prec, rec = adapted_rand_error(gt, pred)

        result = {
            "file": id_,
            "voi_split": voi_split,
            "voi_merge": voi_merge,
            "voi": voi_split + voi_merge,
            "are": are,
            "are_precision": prec,
            "are_recall": rec,
            "n_voxels": gt.size,
        }

        if not skip_erl:
            gt_graphs = build_gt_skeleton_graphs(gt, spacing)
            erl, length, score, per_object = compute_erl(gt_graphs, pred)
            result["erl"] = erl
            result["erl_skeleton_length"] = length
            erl_total_length += length
            erl_total_score += score
            if verbose:
                for lbl, obj in sorted(per_object.items()):
                    print("  GT object {}: length={:.3f}, ERL={:.3f}".format(lbl, obj["length"], obj["erl"]))

        if verbose:
            print(result)

        per_file_results.append(result)

    dataset_erl = None
    if not skip_erl:
        dataset_erl = erl_total_score / erl_total_length if erl_total_length > 0 else 0.0

    return per_file_results, dataset_erl


def print_summary(per_file_results, dataset_erl, skip_erl=False):
    print("")
    print("#################")
    print("# FINAL RESULTS #")
    print("#################")
    print("")

    weights = np.array([r["n_voxels"] for r in per_file_results], dtype=np.float64)
    weights = weights / weights.sum()

    def wmean(key):
        return float(np.sum([r[key] * w for r, w in zip(per_file_results, weights)]))

    def mean_std(key):
        vals = np.array([r[key] for r in per_file_results], dtype=np.float64)
        return float(np.mean(vals)), float(np.std(vals))

    print("~~~~~~ Per-file stats ~~~~~~")
    for r in per_file_results:
        msg = "{}: VOI-split={:.4f}, VOI-merge={:.4f}, VOI={:.4f}, ARE={:.4f} (prec={:.4f}, rec={:.4f})".format(
            r["file"], r["voi_split"], r["voi_merge"], r["voi"], r["are"], r["are_precision"], r["are_recall"])
        if "erl" in r:
            msg += ", ERL={:.4f} (skeleton length={:.1f})".format(r["erl"], r["erl_skeleton_length"])
        print(msg)
    print("")

    print("~~~~~~ Aggregated stats (mean +/- std across {} files) ~~~~~~".format(len(per_file_results)))
    for key, label in [("voi_split", "VOI-split"), ("voi_merge", "VOI-merge"), ("voi", "VOI (total)"),
                        ("are", "Adapted Rand error"), ("are_precision", "Adapted Rand precision"),
                        ("are_recall", "Adapted Rand recall")]:
        m, s = mean_std(key)
        print("{}: {:.4f} +/- {:.4f}".format(label, m, s))
    if not skip_erl:
        m, s = mean_std("erl")
        print("ERL (per-file mean +/- std): {:.4f} +/- {:.4f}".format(m, s))
    print("")

    print("~~~~~~ Aggregated stats (voxel-count-weighted mean across files) ~~~~~~")
    print("VOI-split: {:.4f}".format(wmean("voi_split")))
    print("VOI-merge: {:.4f}".format(wmean("voi_merge")))
    print("VOI (total): {:.4f}".format(wmean("voi")))
    print("Adapted Rand error: {:.4f}".format(wmean("are")))
    print("Adapted Rand precision: {:.4f}".format(wmean("are_precision")))
    print("Adapted Rand recall: {:.4f}".format(wmean("are_recall")))
    if not skip_erl:
        print("ERL (skeleton-length-weighted over dataset): {:.4f}".format(dataset_erl))
    print("")


def main():
    args = parse_args()
    per_file_results, dataset_erl = evaluate_dataset(
        args.input_dir, args.gt_dir, spacing=args.spacing, skip_erl=args.skip_erl, verbose=args.verbose)
    print_summary(per_file_results, dataset_erl, skip_erl=args.skip_erl)


if __name__ == "__main__":
    main()
