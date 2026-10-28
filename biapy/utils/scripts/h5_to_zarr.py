import os
import shutil
import argparse
import h5py
import numpy as np
import zarr
from tqdm import tqdm

from biapy.data.ome_zarr import write_image, write_label

# Usage:
# python h5_to_zarr.py /path/to/h5_dir /path/to/out_dir --axes ZYX --labels neuron_ids=volumes/labels/neuron_ids


def parse_args():
    parser = argparse.ArgumentParser(description="Convert H5 files into Zarr files.")
    parser.add_argument("input", type=str, help="H5 file, or directory with H5 files, to convert.")
    parser.add_argument("output_dir", type=str, help="Directory where the Zarr files will be saved.")
    parser.add_argument(
        "--format",
        type=str,
        default="ome-zarr",
        choices=["ome-zarr", "zarr"],
        help="Type of Zarr to write: an OME-Zarr image (default) with the dataset '--dataset' and, as its label "
        "images, those of '--labels'; or a plain Zarr copying the whole H5 hierarchy and its attributes.",
    )
    parser.add_argument(
        "--dataset", type=str, default="", help="Dataset with the image within the H5 file (default: the first one found). OME-Zarr only."
    )
    parser.add_argument(
        "--axes",
        type=str,
        default="",
        help="Axes of the image dataset, e.g. ZYXC (default: YX, ZYX or ZYXC depending on its dimensions). OME-Zarr only.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="*",
        default=[],
        metavar="NAME=DATASET",
        help="Datasets to add as label images (e.g. the ground truth), with the axes of the image but the channel one. "
        "BiaPy reads them by setting 'DATA.*.INPUT_ZARR_MULTIPLE_DATA_GT_PATH' to 'labels.NAME'. OME-Zarr only.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        nargs=3,
        default=None,
        metavar=("Z", "Y", "X"),
        help="Voxel size in (Z, Y, X) order (default: the 'resolution' attribute of the dataset if it has it, 1 1 1 "
        "otherwise). OME-Zarr only.",
    )
    parser.add_argument("--unit", type=str, default="", help="Unit of the resolution, e.g. nanometer. OME-Zarr only.")
    parser.add_argument("--ome_version", type=str, default="0.5", choices=["0.4", "0.5"], help="OME-Zarr version (default: 0.5).")
    parser.add_argument("--pyramid_levels", type=int, default=1, help="Resolution levels of the OME-Zarr image (default: 1).")
    return parser.parse_args()


def hdf5_to_zarr(hdf5_filename, outdir):
    """Copy the whole hierarchy of an H5 file, with its attributes, into a plain Zarr file."""
    hdf5_file = h5py.File(hdf5_filename, "r")

    # Create zarr dataset
    zarr_name = os.path.join(outdir, os.path.splitext(os.path.basename(hdf5_filename))[0] + os.extsep + "zarr")
    zarr_group = zarr.open_group(zarr_name, mode="w")

    def copy(name, obj):
        if isinstance(obj, h5py.Group):
            zarr_obj = zarr_group.create_group(name)
        elif isinstance(obj, h5py.Dataset):
            zarr_obj = zarr_group.create_array(name, data=obj[()], chunks=obj.chunks or "auto")
        else:
            assert False, "Unsupport HDF5 type."

        # NumPy values (e.g. a 'resolution' array) can not be stored as JSON
        zarr_obj.attrs.update({k: v.tolist() if hasattr(v, "tolist") else v for k, v in obj.attrs.items()})

    hdf5_file.visititems(copy)

    hdf5_file.close()

    return zarr_group


def hdf5_to_ome_zarr(hdf5_filename, outdir, args):
    """Write a dataset of an H5 file, and optionally others as its label images, as an OME-Zarr image."""
    zarr_name = os.path.join(outdir, os.path.splitext(os.path.basename(hdf5_filename))[0] + os.extsep + "zarr")
    with h5py.File(hdf5_filename, "r") as hdf5_file:
        if args.dataset:
            dataset = hdf5_file[args.dataset]
        else:
            datasets = []
            hdf5_file.visititems(lambda name, obj: datasets.append(obj) if isinstance(obj, h5py.Dataset) else None)
            if not datasets:
                raise ValueError(f"No dataset found in {hdf5_filename}")
            dataset = datasets[0]
        axes = args.axes.upper() or {2: "YX", 3: "ZYX", 4: "ZYXC"}.get(dataset.ndim, "")
        if len(axes) != dataset.ndim:
            raise ValueError(f"Set '--axes' for the dataset {dataset.name} of {hdf5_filename} (shape {dataset.shape})")
        resolution = args.resolution or dataset.attrs.get("resolution", [1, 1, 1])
        sizes = dict(zip("ZYX", (float(v) for v in resolution)))

        def scale_units(ax):
            return [sizes.get(a, 1.0) for a in ax], [args.unit if a in sizes and args.unit else None for a in ax]

        if os.path.exists(zarr_name):
            shutil.rmtree(zarr_name)
        scale, units = scale_units(axes)
        write_image(zarr_name, np.asarray(dataset), axes, scale, units, args.ome_version, pyramid_levels=args.pyramid_levels)
        for item in args.labels:
            name, path = item.split("=", 1)
            label_axes = axes.replace("C", "")
            scale, units = scale_units(label_axes)
            write_label(zarr_name, name, np.asarray(hdf5_file[path]), label_axes, scale, units, args.pyramid_levels)
        print(
            f"Saved {zarr_name} (OME-Zarr {args.ome_version}) with the dataset {dataset.name} as image (axes {axes})"
            + "".join(f", {item.split('=', 1)[1]} as label image '{item.split('=', 1)[0]}'" for item in args.labels)
        )


if __name__ == "__main__":
    args = parse_args()
    if os.path.isdir(args.input):
        files = [os.path.join(args.input, f) for f in sorted(next(os.walk(args.input))[2])]
    else:
        files = [args.input]
    os.makedirs(args.output_dir, exist_ok=True)
    for hdf5_file in tqdm(files):
        if args.format == "ome-zarr":
            hdf5_to_ome_zarr(hdf5_file, args.output_dir, args)
        else:
            hdf5_to_zarr(hdf5_file, args.output_dir)
