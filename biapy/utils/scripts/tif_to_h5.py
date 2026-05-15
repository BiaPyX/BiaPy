import os
import argparse
import h5py
import numpy as np
from tqdm import tqdm
from biapy.data.data_manipulation import read_img_as_ndarray


def parse_args():
    parser = argparse.ArgumentParser(description="Stack TIFF files from a directory into a single HDF5 file.")
    parser.add_argument("input_dir", type=str, help="Directory containing input TIFF files.")
    parser.add_argument("output_dir", type=str, help="Directory where the HDF5 file will be saved.")
    parser.add_argument("h5_filename", type=str, help="Name of the output HDF5 file (e.g. output.h5).")
    parser.add_argument("--is_3d", action="store_true", help="Set if the images are 3D.")
    parser.add_argument(
        "--compression",
        type=str,
        default="lzf",
        choices=["gzip", "lzf", "none"],
        help="Compression algorithm for the HDF5 dataset (default: lzf).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    compression = None if args.compression == "none" else args.compression

    input_ids = sorted(next(os.walk(args.input_dir))[2])

    imgs = []
    for id_ in tqdm(input_ids, desc="Reading"):
        img = read_img_as_ndarray(os.path.join(args.input_dir, id_), is_3d=args.is_3d)
        imgs.append(img)

    pred_stack = np.stack(imgs, axis=0)
    print(f"Stack shape: {pred_stack.shape}")

    os.makedirs(args.output_dir, exist_ok=True)
    h5_path = os.path.join(args.output_dir, args.h5_filename)
    with h5py.File(h5_path, "w") as h5f:
        h5f.create_dataset("main", data=pred_stack, compression=compression)
    print(f"Saved {h5_path}")


if __name__ == "__main__":
    main()
