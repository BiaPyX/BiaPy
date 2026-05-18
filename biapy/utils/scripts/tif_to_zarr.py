import os
import argparse
import zarr
import numpy as np
from tqdm import tqdm
from biapy.data.data_manipulation import read_img_as_ndarray


def parse_args():
    parser = argparse.ArgumentParser(description="Stack TIFF files from a directory into a single Zarr file.")
    parser.add_argument("input_dir", type=str, help="Directory containing input TIFF files.")
    parser.add_argument("output_dir", type=str, help="Directory where the Zarr file will be saved.")
    parser.add_argument("zarr_filename", type=str, help="Name of the output Zarr file (e.g. data.zarr).")
    parser.add_argument("--is_3d", action="store_true", help="Set if the images are 3D.")
    parser.add_argument("--dsx", type=str, default="volumes/raw", help="Dataset path in the Zarr file (default: volumes/raw).")
    parser.add_argument("--resolution", type=int, nargs=3, default=[8, 8, 8], metavar=("Z", "Y", "X"), help="Resolution in (Z, Y, X) order (default: 8 8 8).")
    parser.add_argument("--offset", type=int, nargs=3, default=[0, 0, 0], metavar=("Z", "Y", "X"), help="Offset in (Z, Y, X) order (default: 0 0 0).")
    return parser.parse_args()


def main():
    args = parse_args()
    resolution = tuple(args.resolution)
    offset = tuple(args.offset)

    input_ids = sorted(next(os.walk(args.input_dir))[2])

    imgs = []
    for id_ in tqdm(input_ids, desc="Reading"):
        img = read_img_as_ndarray(os.path.join(args.input_dir, id_), is_3d=args.is_3d)
        imgs.append(img)

    pred_stack = np.stack(imgs, axis=0)
    print(f"Stack shape: {pred_stack.shape}")

    os.makedirs(args.output_dir, exist_ok=True)
    zarr_path = os.path.join(args.output_dir, args.zarr_filename)
    store = zarr.open_group(zarr_path, mode="w")
    ds = store.create_array(args.dsx, shape=pred_stack.shape, dtype=pred_stack.dtype)
    ds[:] = pred_stack
    ds.attrs["resolution"] = resolution
    ds.attrs["offset"] = offset
    print(f"Saved {zarr_path}")


if __name__ == "__main__":
    main()
