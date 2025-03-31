import sys
import os
import zarr
import numpy as np

code_dir = "/data/dfranco/BiaPy"
zarrfile = "/data/dfranco/exp_results/hemibrain3.v2/results/hemibrain3.v2_1/per_image/synapses_x12437-13037_y27229-27829_z17176-17776.zarr"
out_dir = "/data/dfranco/exp_results/hemibrain3.v2/results/hemibrain3.v2_1/per_image/"

# Use a function to save the files
sys.path.insert(0, code_dir)
from biapy.data.data_manipulation import save_tif
from biapy.data.data_3D_manipulation import read_chunked_data, ensure_3d_shape

# Read the data
data = read_chunked_data(zarrfile)[1]
data = np.array(data)
data = ensure_3d_shape(data)

# Data needs to be like this: (500, 4096, 4096, 1),  (z,x,y,c)
if data.ndim != 4:
    raise ValueError("Data should be 4 dimensional, given {}".format(data.shape))

save_tif(np.expand_dims(data,0), out_dir , [zarrfile], verbose=True)
