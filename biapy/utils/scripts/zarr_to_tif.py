import sys
import os
import argparse
import numpy as np

# Usage:
# python zarr_to_tif.py --code_dir /data/dfranco/BiaPy --zarrfile /path/to/input.zarr --out_dir /path/to/output_tif_directory/
def parse_args():
    p = argparse.ArgumentParser(description="Convert a Zarr volume to TIFF using BiaPy save_tif.")
    p.add_argument("--code_dir", required=True, help="Path to BiaPy code directory (e.g. /data/dfranco/BiaPy)")
    p.add_argument("--zarrfile", required=True, help="Path to input .zarr (directory or file used by read_chunked_data)")
    p.add_argument("--out_dir", required=True, help="Output directory where TIFF will be written")
    return p.parse_args()

def main():
    args = parse_args()

    code_dir = args.code_dir
    zarrfile = args.zarrfile
    out_dir = args.out_dir

    if not os.path.exists(code_dir):
        raise FileNotFoundError(f"--code_dir does not exist: {code_dir}")
    if not os.path.exists(zarrfile):
        raise FileNotFoundError(f"--zarrfile does not exist: {zarrfile}")
    os.makedirs(out_dir, exist_ok=True)

    # Import BiaPy utilities
    sys.path.insert(0, code_dir)
    from biapy.data.data_manipulation import save_tif
    from biapy.data.data_3D_manipulation import read_chunked_data, ensure_3d_shape

    # Read the data
    data = read_chunked_data(zarrfile)[1]
    data = np.array(data)
    data = ensure_3d_shape(data)

    # Data needs to be like this: (Z, X, Y, C)
    if data.ndim != 4:
        raise ValueError(f"Data should be 4 dimensional (Z,X,Y,C), got shape {data.shape}")

    # Keep same call signature as your original script
    save_tif(np.expand_dims(data, 0), out_dir, [os.path.basename(zarrfile)], verbose=True)

if __name__ == "__main__":
    main()
