import os
import shutil
import argparse
import zarr
import numpy as np
from tqdm import tqdm
from biapy.data.data_manipulation import read_img_as_ndarray
from biapy.data.ome_zarr import write_image


def parse_args():
    parser = argparse.ArgumentParser(description="Stack TIFF files from a directory into a single Zarr file.")
    parser.add_argument("input_dir", type=str, help="Directory containing input TIFF files.")
    parser.add_argument("output_dir", type=str, help="Directory where the Zarr file will be saved.")
    parser.add_argument("zarr_filename", type=str, help="Name of the output Zarr file (e.g. data.zarr).")
    parser.add_argument("--is_3d", action="store_true", help="Set if the images are 3D.")
    parser.add_argument(
        "--format",
        type=str,
        default="ome-zarr",
        choices=["ome-zarr", "zarr"],
        help="Type of Zarr to write: an OME-Zarr image (default; 2D images are stacked along z and 3D ones along t) "
        "or a plain Zarr with the stack at '--dsx' and its resolution and offset as attributes (e.g. the CREMI "
        "layout used for synapse detection).",
    )
    parser.add_argument("--dsx", type=str, default="volumes/raw", help="Dataset path in the Zarr file (default: volumes/raw). Plain Zarr only.")
    parser.add_argument("--resolution", type=float, nargs=3, default=[8, 8, 8], metavar=("Z", "Y", "X"), help="Resolution in (Z, Y, X) order (default: 8 8 8).")
    parser.add_argument("--offset", type=int, nargs=3, default=[0, 0, 0], metavar=("Z", "Y", "X"), help="Offset in (Z, Y, X) order (default: 0 0 0). Plain Zarr only.")
    parser.add_argument("--unit", type=str, default="nanometer", help="Unit of the resolution (default: nanometer). OME-Zarr only.")
    parser.add_argument("--ome_version", type=str, default="0.5", choices=["0.4", "0.5"], help="OME-Zarr version (default: 0.5).")
    parser.add_argument("--pyramid_levels", type=int, default=1, help="Resolution levels of the OME-Zarr image (default: 1).")
    return parser.parse_args()


def main():
    args = parse_args()
    resolution = tuple(int(v) if float(v).is_integer() else v for v in args.resolution)
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
    if args.format == "ome-zarr":
        # The images are (z, y, x, c) in 3D and (y, x, c) in 2D
        axes = "TZYXC" if args.is_3d else "ZYXC"
        if axes[0] == "T" and len(pred_stack) == 1:
            pred_stack, axes = pred_stack[0], axes[1:]
        sizes = dict(zip("ZYX", resolution))
        if os.path.exists(zarr_path):
            shutil.rmtree(zarr_path)
        write_image(
            zarr_path,
            pred_stack,
            axes,
            scale=[sizes.get(a, 1.0) for a in axes],
            units=[args.unit if a in sizes and args.unit else None for a in axes],
            version=args.ome_version,
            pyramid_levels=args.pyramid_levels,
        )
        print(f"Saved {zarr_path} (OME-Zarr {args.ome_version}, axes {axes})")
        return
    store = zarr.open_group(zarr_path, mode="w")
    ds = store.create_array(args.dsx, shape=pred_stack.shape, dtype=pred_stack.dtype)
    ds[:] = pred_stack
    ds.attrs["resolution"] = resolution
    ds.attrs["offset"] = offset
    print(f"Saved {zarr_path}")


if __name__ == "__main__":
    main()
