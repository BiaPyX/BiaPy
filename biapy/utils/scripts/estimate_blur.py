import os
import sys
import torch
import numpy as np
import argparse
import os
import sys
from tqdm import tqdm
from skimage.measure import blur_effect
from biapy.data.pre_processing import norm_range01, percentile_clip
from biapy.data.data_manipulation import read_img_as_ndarray
import cv2

parser = argparse.ArgumentParser(description="Measure the blur for each image in the given folder",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-input_dir", "--input_dir", required=True, help="Input image directory")
args = vars(parser.parse_args())

# python estimate_blur.py --input_dir /scratch/dfranco/thesis/data2/dfranco/datasets/FuseMyCells/prepared_data/val/raw 

ids = sorted(next(os.walk(args['input_dir']))[2])
for n, id_ in tqdm(enumerate(ids)):
    # img read
    img_path = os.path.join(args['input_dir'], id_)
    img = read_img_as_ndarray(img_path, is_3d=False).squeeze()
    print(img.dtype)

    # # img norm
    # img, _, _ = percentile_clip(img, lower=2., upper=99.8)
    # img, _ = norm_range01(img, div_using_max_and_scale=True)
    # img = (img*255).astype(np.uint8)

    blur = blur_effect(img, h_size=11)
    print("{} - blur_effect: {}".format(id_, blur))

    # Calcula el laplaciano de la imagen y luego la varianza 
    print(img.dtype)
    laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var() 
    print("{} - laplacian: {}".format(id_, laplacian_var))

print("FINISHED!")