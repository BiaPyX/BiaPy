import sys
import os
import h5py
import numpy as np

code_dir = "/home/user/BiaPy"
h5file = "/home/user/file.h5"
out_dir = "/home/user/out"

# Use a function to save the files
sys.path.insert(0, code_dir)
from utils.util import save_tif

# Load H5 file
h5f = h5py.File(h5file, 'r')
k = h5f.keys()

# Change the key accordingly
print("Keys: {}".format(k))
data = (h5f['main'])

if data.ndim == 3:
    data = np.expand_dims(data, -1)

# Data needs to be like this: (500, 4096, 4096, 1),  (z,x,y,c)
if data.ndim != 4:
    raise ValueError("Data should be 4 dimensional, given {}".format(data.shape))

save_tif(data, out_dir , [h5file.split('.')[0]+".tif"], verbose=True)
