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
args = vars(parser.parse_args())

sys.path.insert(0, args['BiaPy_dir'])

df = pd.read_csv(args['csv'], header=None)

if len(df.columns) != 3:
    raise ValueError("CSV file need to contain three columns with no header")

df.columns = ["phase", "filename", "class"]

df = df.reset_index()
for index, row in tqdm(df.iterrows()):
    dest_folder = os.path.join(args['out_dir'], str(row['phase']).lower(), str(row['class']))
    os.makedirs(dest_folder, exist_ok=True)

    filename = os.path.join(args['image_dir'], row['filename'])
    print("Copying file {} to {}".format(filename, dest_folder))
    shutil.copy(filename, dest_folder)
