import sys
import os
import argparse
import shutil
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Organize classification data",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument("-image_dir", "--image_dir", help="Image directory")
parser.add_argument("-csv", "--csv", help="CSV file with the class of each image. Must contain three columns with no header.")
parser.add_argument("-out_dir", "--out_dir",help="Output directory to organize the images into")
parser.add_argument("-BiaPy_dir", "--BiaPy_dir", help="BiaPy path")
parser.add_argument("-phase", "--phase", help="Phase of the images", choices=['train', 'test'])
args = vars(parser.parse_args())

# Example call:
# python BiaPy/utils/scripts/from_class_csv_to_folders.py --image_dir train_raw -phase train -csv Training_set.csv -out_dir organized -BiaPy_dir BiaPy


sys.path.insert(0, args['BiaPy_dir'])

df = pd.read_csv(args['csv'], header=None)

if hasattr(args, 'phase'):
    if len(df.columns) != 3:
        raise ValueError("CSV file need to contain three columns and no header")    

if len(df.columns) == 3:
    df.columns = ["phase", "filename", "class"]
elif len(df.columns) == 2:
    df.columns = ["filename", "class"]
else:
    raise ValueError("CSV file need to contain at least two columns and no header")  

df = df.reset_index()
for index, row in tqdm(df.iterrows()):
    if hasattr(args, 'phase'):
        dest_folder = os.path.join(args['out_dir'], str(row['phase']).lower(), str(row['class']))
    else:
        dest_folder = os.path.join(args['out_dir'], str(args['phase']).lower(), str(row['class']))
    os.makedirs(dest_folder, exist_ok=True)

    filename = os.path.join(args['image_dir'], row['filename'])
    print("Copying file {} to {}".format(filename, dest_folder))
    shutil.copy(filename, dest_folder)
