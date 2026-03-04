#!/usr/bin/env python3
import os
import sys
import argparse

import h5py
import numpy as np

# Call example:
# python h5_to_tif.py --code-dir /home/user/BiaPy --h5file /home/user/file.h5 --out-dir /home/user/out

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert an HDF5 dataset to a TIFF using BiaPy's save_tif."
    )
    p.add_argument(
        "--code-dir",
        required=True,
        help="Path to BiaPy code directory (the folder to add to PYTHONPATH).",
    )
    p.add_argument(
        "--h5file",
        required=True,
        help="Path to the input .h5/.hdf5 file.",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        help="Output directory where the TIFF will be written.",
    )
    p.add_argument(
        "--dataset-key",
        type=str,
        default="main",
        help="HDF5 dataset key to read (default: main).",
    )
    p.add_argument(
        "--out-name",
        default=None,
        help="Output TIFF filename (default: input basename + .tif).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    code_dir = os.path.abspath(args.code_dir)
    h5file = os.path.abspath(args.h5file)
    out_dir = os.path.abspath(args.out_dir)

    if not os.path.isdir(code_dir):
        raise FileNotFoundError(f"--code-dir does not exist or is not a directory: {code_dir}")
    if not os.path.isfile(h5file):
        raise FileNotFoundError(f"--h5file does not exist or is not a file: {h5file}")

    # Import after inserting code_dir
    sys.path.insert(0, code_dir)
    try:
        from biapy.data.data_manipulation import save_tif
        from biapy.data.data_3D_manipulation import read_chunked_nested_data, ensure_3d_shape
        
    except Exception as e:
        raise ImportError(
            "Could not import BiaPy functions. "
            "Check that --code-dir points to the correct BiaPy root."
        ) from e

    file, data = read_chunked_nested_data(h5file, data_path=args.dataset_key)
    data = np.array(data)
    if data.max() == 0:
        print("Warning: data is all zeros, check that the dataset key is correct.")

    data = ensure_3d_shape(data)
    if isinstance(file, h5py.File):
        file.close()

    basename, _ = os.path.splitext(os.path.basename(h5file))
    new_name = basename+"_"+args.dataset_key + ".tif"
    print(f"Saving TIFF to {out_dir} with name {new_name}...")
    save_tif(np.expand_dims(data, 0), out_dir, [new_name], verbose=True)

if __name__ == "__main__":
    main()
