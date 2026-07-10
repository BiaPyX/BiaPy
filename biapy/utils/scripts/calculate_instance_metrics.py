import os
import sys
import argparse
import numpy as np
from skimage.io import imread

# Make the BiaPy package importable (this script lives in biapy/utils/scripts/)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from biapy.utils.matching import matching, wrapper_matching_dataset_lazy

# Regular image extensions supported for both predictions and ground truth
IMAGE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".gif")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate instance segmentation matching metrics (precision, recall, F1, "
                    "panoptic quality, ...) between predictions and ground truth for 3D data.")

    parser.add_argument("input_dir", help="Directory containing the prediction label files.")
    parser.add_argument("gt_dir", help="Directory containing the ground truth label files.")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.75],
                        help="IoU thresholds used for the matching stats. Default: 0.3 0.5 0.75")
    parser.add_argument("--criterion", default="iou", choices=["iou", "iot", "iop"],
                        help="Matching criterion to use. Default: iou")
    parser.add_argument("--verbose", action="store_true", help="Print per-file statistics.")

    return parser.parse_args()


def main():
    args = parse_args()

    all_matching_stats = []

    # Prediction and ground truth files are matched by base name (stem), so the
    # extensions may differ (e.g. a .png prediction against a .tif ground truth).
    def list_images(directory):
        return sorted(f for f in next(os.walk(directory))[2] if f.lower().endswith(IMAGE_EXTENSIONS))

    gt_by_stem = {os.path.splitext(f)[0]: f for f in list_images(args.gt_dir)}

    ids = list_images(args.input_dir)
    for id_ in ids:
        stem = os.path.splitext(id_)[0]
        if stem not in gt_by_stem:
            raise FileNotFoundError(
                "No ground truth image matching prediction '{}' was found in {} "
                "(looked for base name '{}').".format(id_, args.gt_dir, stem))

        img = imread(os.path.join(args.input_dir, id_)).astype(np.int64)
        mask = imread(os.path.join(args.gt_dir, gt_by_stem[stem])).astype(np.int64)

        print(" ")
        print("#######################################")
        print("Analizing file {} (GT: {})".format(
            os.path.join(args.input_dir, id_), os.path.join(args.gt_dir, gt_by_stem[stem])))

        r_stats = matching(mask, img, thresh=args.thresholds, criterion=args.criterion, report_matches=False)
        if args.verbose:
            print(r_stats)
        all_matching_stats.append(r_stats)

    print("#################")
    print("# FINAL RESULTS #")
    print("#################")
    print("")

    stats = wrapper_matching_dataset_lazy(all_matching_stats, args.thresholds, criterion=args.criterion, by_image=True)
    # wrapper returns a single result for one threshold, or a tuple for several
    if len(args.thresholds) == 1:
        stats = (stats,)
    print("~~~~~~ Matching stats ~~~~~~")
    for th, st in zip(args.thresholds, stats):
        print("IoU TH={}".format(th))
        print(st)
    print("")


if __name__ == "__main__":
    main()
