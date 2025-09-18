import argparse
import os
import sys
import numpy as np
from tqdm import tqdm
import fill_voids

parser = argparse.ArgumentParser(description="Fill tiny holes in semantic/instance masks",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-input_label_dir", "--input_label_dir", required=True, help="Directory to the folder where the labels to fbe fixed reside")
parser.add_argument("-output_label_dir", "--output_label_dir", required=True, help="output folder to store the fixed labels")
parser.add_argument("-is_3d", "--is_3d", action='store_true', help="Flag to indicate if the input is 3D")
parser.add_argument("-BiaPy_dir", "--BiaPy_dir", required=True, help="BiaPy directory")
args = vars(parser.parse_args())

# python fill_holes_in_seg_masks.py --input_label_dir /data/dfranco/thesis/data2/dfranco/datasets/oocyte/arturo_instances/v2/y --output_label_dir /scratch/dfranco/thesis/data2/dfranco/datasets/oocyte/arturo_instances/v3/y --BiaPy_dir ../../ --is_3d

sys.path.insert(0, args['BiaPy_dir'])
from biapy.data.data_manipulation import save_tif, read_img_as_ndarray

data_dir = args['input_label_dir']
out_dir = args['output_label_dir']

print("Processing {} folder . . .".format(data_dir))
ids = sorted(next(os.walk(data_dir))[2])
# Read images
for n, id_ in tqdm(enumerate(ids), total=len(ids)):

    # Load image
    img = read_img_as_ndarray(os.path.join(data_dir, id_), is_3d=args['is_3d'])

    # Fill tiny holes 
    new_img = np.zeros(img.shape, dtype=img.dtype)
    if args['is_3d']:
        for c in range(img.shape[-1]):
            for z in range(len(img)):
                instances_in_slice = np.unique(img[z,...,c])[1:]
                for l in instances_in_slice:
                    new_img[z,...,c] += (fill_voids.fill(img[z,...,c]==l)*l).astype(img.dtype)
    else:
        for c in range(img.shape[-1]):
            instances_in_slice = np.unique(img[...,c])[1:]
            for l in instances_in_slice:
                new_img[...,c] += (fill_voids.fill(img[...,c]==l)*l).astype(img.dtype)

    save_tif(np.expand_dims(new_img,0), out_dir, filenames=[id_], verbose=True)
