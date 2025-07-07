import argparse
from tqdm import tqdm
import json
from pathlib import Path
import os

parser = argparse.ArgumentParser(description="Fill tiny holes in semantic/instance masks",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-input_notebook_dir", "--input_notebook_dir", required=True, help="Directory to the folder where the labels to fbe fixed reside")
args = vars(parser.parse_args())


# Quick func to retrieve all files in a directory and its subdirectories
def get_all_files(directory):
    return [str(p) for p in Path(directory).rglob('*') if p.is_file()]

# python /data/dfranco/BiaPy/biapy/utils/scripts/repair_colab_notebooks.py --input_notebook_dir /data/dfranco/BiaPy/notebooks 

files = get_all_files(args['input_notebook_dir'])
files = [x for x in files if x.endswith(".ipynb")]

print("Processing {} folder . . .".format(files))

# Read images
for n, id_ in tqdm(enumerate(files), total=len(files)):
    print("Processing notebook: {}".format(id_))
    notebook_file = os.path.join(args['input_notebook_dir'], id_)
    
    with open(notebook_file, 'r') as f:
        notebook = json.load(f)

    if 'widgets' in notebook.get('metadata', {}):
        del notebook['metadata']['widgets']

    for cell in notebook.get('cells', []):
        if 'widgets' in cell.get('metadata', {}):
            del cell['metadata']['widgets']

    with open(os.path.join(args['input_notebook_dir'], id_), 'w') as f:
        json.dump(notebook, f, indent=2)

print("Done!")