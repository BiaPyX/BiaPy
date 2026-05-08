import argparse
from tqdm import tqdm
import json
from pathlib import Path
import os

parser = argparse.ArgumentParser(
    description="Repair Jupyter notebooks for GitHub compatibility by removing incomplete 'widgets' metadata. "
                "GitHub displays 'Invalid Notebook' errors when notebooks contain 'metadata.widgets' without required 'state' keys. "
                "This script removes such metadata to allow notebooks to render properly on GitHub.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-input_notebook_dir", "--input_notebook_dir", required=True, 
                    help="Path to the directory containing Jupyter notebooks (.ipynb files) to repair. "
                         "The script will recursively process all notebooks in this directory and its subdirectories.")
args = vars(parser.parse_args())

print("=" * 80)
print("Jupyter Notebook GitHub Compatibility Repair Tool")
print("=" * 80)
print()

def get_all_files(directory):
    """Recursively retrieve all files in a directory and its subdirectories."""
    return [str(p) for p in Path(directory).rglob('*') if p.is_file()]

files = get_all_files(args['input_notebook_dir'])
files = [x for x in files if x.endswith(".ipynb")]

if not files:
    print(f"No Jupyter notebooks (.ipynb files) found in: {args['input_notebook_dir']}")
    print("Please check the directory path and try again.")
    exit(1)

print(f"Found {len(files)} notebook(s) to process in: {args['input_notebook_dir']}")
print()

# Process each notebook
for n, id_ in tqdm(enumerate(files), total=len(files)):
    print("Processing notebook: {}".format(id_))    
    with open(id_, 'r') as f:
        notebook = json.load(f)

    removed_notebook_widgets = False
    removed_cell_widgets = 0

    if 'widgets' in notebook.get('metadata', {}):
        del notebook['metadata']['widgets']
        removed_notebook_widgets = True

    for cell in notebook.get('cells', []):
        if 'widgets' in cell.get('metadata', {}):
            del cell['metadata']['widgets']
            removed_cell_widgets += 1

    with open(id_, 'w') as f:
        json.dump(notebook, f, indent=2)
    
    # Print summary of what was removed
    if removed_notebook_widgets or removed_cell_widgets > 0:
        changes = []
        if removed_notebook_widgets:
            changes.append("notebook-level widgets metadata")
        if removed_cell_widgets > 0:
            changes.append(f"{removed_cell_widgets} cell(s) with widgets metadata")
        print(f"  ✓ Removed: {', '.join(changes)}")
    else:
        print(f"  ✓ No repairs needed")

print()
print("=" * 80)
print("✓ All notebooks have been successfully repaired!")
print("They should now render correctly on GitHub.")
print("=" * 80)