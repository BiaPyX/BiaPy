import os
import h5py
import zarr
from tqdm import tqdm 

def hdf5_to_zarr(hdf5_filename, outdir, zarr_group=None):
    try:
        unicode
    except NameError:
        unicode = str

    # Open H5 dataset
    hdf5_file = h5py.File(hdf5_filename, "r")
    data = hdf5_file[list(hdf5_file)[0]]

    # Create zarr dataset
    zarr_name = os.path.join(outdir, os.path.splitext(os.path.basename(hdf5_filename))[0] + os.extsep + "zarr")
    zarr_group = zarr.open_group(zarr_name, mode="w")

    def copy(name, obj):
        if isinstance(obj, h5py.Group):
            zarr_obj = zarr_group.create_group(name)
        elif isinstance(obj, h5py.Dataset):
            zarr_obj = zarr_group.create_dataset(name, data=obj, chunks=obj.chunks)
        else:
            assert False, "Unsupport HDF5 type."

        zarr_obj.attrs.update(obj.attrs) 

    hdf5_file.visititems(copy)

    hdf5_file.close()

    return zarr_group

if __name__ == '__main__':

    hdf5_files_dir = "/PATH_TO_H5_DIR"
    outdir = "/OUT_DIR"

    ids = sorted(next(os.walk(hdf5_files_dir))[2])
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        hdf5_file = os.path.join(hdf5_files_dir, id_)
        hdf5_to_zarr(hdf5_file, outdir)