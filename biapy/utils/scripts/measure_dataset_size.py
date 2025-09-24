import os
import os
import argparse
from typing import Iterable, List
from tqdm import tqdm
from biapy.data.data_manipulation import read_img_as_ndarray

# Placeholder: you will provide this function
def load_image(path, is_3d: bool = False):
    """
    Load an image from a given path.
    Returns an array-like object with shape (H, W, C) or (H, W).
    """
    return read_img_as_ndarray(path, is_3d=is_3d).squeeze()

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}

def iter_files(root: str) -> Iterable[str]:
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            yield os.path.join(dirpath, fname)

def filter_images(paths: Iterable[str]) -> List[str]:
    out = []
    for p in paths:
        ext = os.path.splitext(p)[1].lower()
        if ext in IMAGE_EXTS:
            out.append(p)
    return out

def format_in_millions(n: int) -> str:
    """Return value in millions (M) with two decimal places."""
    return f"{n / 1_000_000:.2f}M"

def dataset_size(dataset_path: str, include_channels: bool = False, quiet: bool = False, is_3d: bool = False) -> int:
    """
    Computes the total pixels (H*W) across images in a folder.
    If include_channels=True, counts elements (H*W*C).
    """
    all_paths = filter_images(iter_files(dataset_path))
    total = 0

    iterator = all_paths
    if not quiet:
        iterator = tqdm(all_paths, desc="Scanning images", unit="img")

    for path in iterator:
        try:
            img = load_image(path, is_3d=is_3d)
            # print(f"Loaded {path} with shape {getattr(img, 'shape', None)}")
            if include_channels:
                total += int(getattr(img, "size"))
            else:
                shape = getattr(img, "shape", None)
                if shape is None or len(shape) < 2:
                    raise ValueError("Loaded object has no valid shape (expected at least 2 dims)")
                if not is_3d:
                    h, w = int(shape[0]), int(shape[1])
                    total += h * w
                else:
                    z, h, w = int(shape[0]), int(shape[1]), int(shape[2])
                    total += h * w * z
        except Exception as e:
            if not quiet:
                tqdm.write(f"Skipping {path}: {e}")

    return total

def main():
    parser = argparse.ArgumentParser(description="Measure dataset size by summing image pixels.")
    parser.add_argument("-path", "--path", type=str, help="Path to the dataset directory.")
    parser.add_argument(
        "--include-channels",
        action="store_true",
        help="Count elements including channels (H*W*C) instead of pixels (H*W)."
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress bar and per-file warnings."
    )
    parser.add_argument("-is_3d", "--is_3d", action='store_true', help="Flag to indicate if the input is 3D")
    args = parser.parse_args()


    total = dataset_size(args.path, include_channels=args.include_channels, quiet=args.quiet, is_3d=args.is_3d)
    kind = "elements (H×W×C)" if args.include_channels else "pixels (H×W)"
    print(f"Total dataset size: {total} {kind}  |  {format_in_millions(total)}")

if __name__ == "__main__":

    # Example call: python -u measure_dataset_size.py --is_3d --path "/data/CartoCell/test/raw/"

    main()

