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
parser.add_argument("-BiaPy_dir", "--BiaPy_dir", required=True, help="BiaPy directory")
args = vars(parser.parse_args())

# python fill_holes_in_seg_masks.py --input_label_dir /scratch/dfranco/thesis/data2/dfranco/datasets/oocyte/arturo_instances/v2/y 
#   --output_label_dir /scratch/dfranco/thesis/data2/dfranco/datasets/oocyte/arturo_instances/v3/y --BiaPy_dir ../../

sys.path.insert(0, args['BiaPy_dir'])
from biapy.data.data_manipulation import save_tif, imread

data_dir = args['input_label_dir']
out_dir = args['output_label_dir']

print("Processing {} folder . . .".format(data_dir))
ids = sorted(next(os.walk(data_dir))[2])
# Read images
for n, id_ in tqdm(enumerate(ids), total=len(ids)):

    # Load image
    if id_.endswith('.npy'):
        img = np.load(os.path.join(data_dir, id_))
    else:
        img = imread(os.path.join(data_dir, id_))
    img = np.squeeze(img)

    if img.ndim < 3:
        raise ValueError("Read image seems to be 2D: {}. Path: {}".format(img.shape, os.path.join(data_dir, id_)))

    # Make sure the image is (z,y,x,c)
    if img.ndim == 3: 
        img = np.expand_dims(img, -1)
    else:
        min_val = min(img.shape)
        channel_pos = img.shape.index(min_val)
        if channel_pos != 3:
            new_pos = [x for x in range(4) if x != channel_pos]+[channel_pos,]
            img = img.transpose(new_pos)
    
    # Fill tiny holes 
    new_img = np.zeros(img.shape, dtype=np.uint16)
    for c in range(img.shape[-1]):
        for z in range(len(img)):
            instances_in_slice = np.unique(img[z,...,c])[1:]
            for l in instances_in_slice:
                new_img[z,...,c] += (fill_voids.fill(img[z,...,c]==l)*l).astype(np.uint16)

    save_tif(np.expand_dims(new_img,0), out_dir, filenames=[id_], verbose=True)
