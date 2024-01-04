import os
import sys
from skimage.io import imread
import numpy as np
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Order 3D image axes into (z,y,x,c)",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-input_dir", "--input_dir", required=True, help="Path to the folder with the input images")
parser.add_argument("-input_axis_order", "--input_axis_order", required=True, help="Axis order. E.g. [y,z,x,c]")
parser.add_argument("-output_dir", "--output_dir", required=True, help="Path to the folder with the output images")
parser.add_argument("-BiaPy_dir", "--BiaPy_dir", required=True, help="BiaPy directory")
args = vars(parser.parse_args())

# Call example:
# python order_axes.py --input_dir /scratch/dfranco/thesis/data2/dfranco/ficheros_BiaPy_GUI/demo3D/data/train/x \
# --output_dir /scratch/dfranco/thesis/data2/dfranco/ficheros_BiaPy_GUI/demo3D/data/train/x2 \
# --input_axis_order [y,z,x,c] \
# --BiaPy_dir /scratch/dfranco/thesis/data2/dfranco/BiaPy 

sys.path.insert(0, args['BiaPy_dir'])
from biapy.utils.util import save_tif

data_dir = args['input_dir']
out_dir = args['output_dir']

try:
   order = args['input_axis_order'].strip('][').strip().split(',')
except:
    raise ValueError("'input_axis_order' needs to be a list. E.g. [y,z,x,c]")

chars = [x for x in order if str(x) not in ['x', 'y','z','c']]
if len(chars)>0:
    raise ValueError("'{}' found in 'input_axis_order'. It needs to have values among these: ['x','y','z','c']".format(chars))

z = order.index("z")
y = order.index("y")
x = order.index("x")
c = order.index("c")
new_pos = (z,y,x,c)

print("Processing {} folder . . .".format(data_dir))
ids = sorted(next(os.walk(data_dir))[2])
# Read images
for n, id_ in tqdm(enumerate(ids), total=len(ids)):
    print("Image: {}".format(os.path.join(data_dir, id_)))

    # Load image
    if id_.endswith('.npy'):
        img = np.load(os.path.join(data_dir, id_))
    else:
        img = imread(os.path.join(data_dir, id_))
    img = np.squeeze(img)

    if img.ndim < 3:
        raise ValueError("Read image seems to be 2D: {}. Path: {}".format(img.shape, os.path.join(data_dir, id_)))
        
    if img.ndim == 3: 
        img = np.expand_dims(img, -1)    

    print("Loaded image shape: {}".format(img.shape, order))
    # Make sure the image is (z,y,x,c)
    img = img.transpose(new_pos)
    print("After: {}".format(img.shape))

    save_tif(np.expand_dims(img,0), out_dir, filenames=[id_], verbose=True)
