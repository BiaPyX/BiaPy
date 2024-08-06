import os
import sys
import numpy as np
from skimage.io import imread
from tqdm import tqdm
import pandas as pd 

code_dir = "/data/dfranco/BiaPy"
input_dir = "/data/dfranco/datasets/polytechnique/V4/train_remove_slices_clean/x"
input_dir2 = "/data/dfranco/datasets/polytechnique/V4/train_remove_slices_clean/y"
out_dir = "/data/dfranco/datasets/polytechnique/V4/train_remove_slices_clean2/x"
out_dir2 = "/data/dfranco/datasets/polytechnique/V4/train_remove_slices_clean2/y"

sys.path.insert(0, code_dir)
from biapy.data.pre_processing import norm_range01
from biapy.utils.util import save_tif, read_img

ids = sorted(next(os.walk(input_dir))[2])
gt_ids = sorted(next(os.walk(input_dir2))[2])
th = 0.55
for n, id_ in tqdm(enumerate(ids), total=len(ids)):
    img = read_img(os.path.join(input_dir, id_), is_3d=True)
    df = pd.read_csv(os.path.join(input_dir2, gt_ids[n]))
    img, norm =  norm_range01(img, dtype=np.float32, div_using_max_and_scale=True, div_using_max_and_scale_per_channel=True)


    # df = df[(df["axis-0"] > 0) & (df["axis-0"] < img.shape[0]-2)]
    # df['axis-0'] -= 1
    # img = img[1:-1]
    df = df.reset_index()  # make sure indexes pair with number of rows

    # Check point color to remove black cells 
    points_to_drop = []
    red_ref = np.quantile(img[...,0], th)*255
    green_ref = np.quantile(img[...,1], th)*255
    blue_ref = np.quantile(img[...,2], th)*255
    for index, row in df.iterrows():
        z,y,x = int(row['axis-0']), int(row['axis-1']), int(row['axis-2'])
        value_r = np.mean(
            img[
                z,
                max(y-2,0):min(img.shape[1],y+2),
                max(x-2,0):min(img.shape[2],x+2),
                0,
            ]*255
        )
        value_g = np.mean(
            img[
                z,
                max(y-2,0):min(img.shape[1],y+2),
                max(x-2,0):min(img.shape[2],x+2),
                1,
            ]*255
        )
        value_b = np.mean(
            img[
                z,
                max(y-2,0):min(img.shape[1],y+2),
                max(x-2,0):min(img.shape[2],x+2),
                2,
            ]*255
        )
        if value_r < red_ref and value_g < green_ref and value_b < blue_ref:
            points_to_drop.append(row['index'])
    if len(points_to_drop)>0:
        print(f"Sample: {id_} {len(points_to_drop)} points discarded from GT (out of {len(df)})")
    for point in points_to_drop:
        df = df[df['index'] != point]

    save_tif(np.expand_dims(img,0), out_dir, [id_])
    os.makedirs(out_dir2, exist_ok=True)
    df =df.sort_values(by=['axis-0','axis-1','axis-2'])
    df.to_csv( os.path.join(out_dir2, gt_ids[n]), index=False)
