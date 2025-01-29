import argparse
import os
import sys
from tqdm import tqdm
from skimage.measure import regionprops_table
import pandas as pd

parser = argparse.ArgumentParser(description="Convert instance data to detection dataset",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-input_dir", "--input_dir", required=True, help="Directory to the folder where the instance masks reside")
parser.add_argument("-output_dir", "--output_dir", required=True, help="Output folder to store the detection data")
parser.add_argument("-BiaPy_dir", "--BiaPy_dir", required=True, help="BiaPy directory")
args = vars(parser.parse_args())

# python convert_instance_data_to_detection.py --input_dir path --output_dir path2 --BiaPy_dir /data/BiaPy

sys.path.insert(0, args['BiaPy_dir'])
from biapy.data.data_manipulation import read_img_as_ndarray

data_dir = args['input_dir']
out_dir = args['output_dir']

print("Processing {} folder . . .".format(data_dir))
ids = sorted(next(os.walk(data_dir))[2])
# Read images
for n, id_ in tqdm(enumerate(ids), total=len(ids)):
    img_path = os.path.join(data_dir, id_)
    print(f"Analising image: {img_path}")
    
    # Load image
    img = read_img_as_ndarray(img_path, is_3d=False)
    
    axis0, axis1, classes = [], [], []
    props = regionprops_table(img[...,0], properties=(["label", "bbox"]))
    for k, l in tqdm(enumerate(props["label"]), total=len(props["label"]), leave=False):

        ycenter = props["bbox-0"][k]+((props["bbox-2"][k]-props["bbox-0"][k])//2)
        xcenter = props["bbox-1"][k]+((props["bbox-3"][k]-props["bbox-1"][k])//2)
        instance_class = img[ycenter,xcenter,1] 
        
        axis0.append(ycenter)
        axis1.append(xcenter)
        classes.append(instance_class)

    df = pd.DataFrame(
        zip(
            axis0,
            axis1,
            classes,
        ),
        columns=['axis-0','axis-1','class'],
    )

    os.makedirs(out_dir, exist_ok=True)
    df.to_csv( os.path.join( out_dir, os.path.splitext(id_)[0] + "_points.csv"))
    