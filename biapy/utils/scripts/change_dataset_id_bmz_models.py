import argparse
import os
from tqdm import tqdm
import zipfile
import yaml
import shutil

parser = argparse.ArgumentParser(description="Change BMZ model dataset ID",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-input_dir", "--input_dir", required=True, help="Directory to the folder where the models are")
parser.add_argument("-tmp_dir", "--tmp_dir", required=True, help="Temporal directory to save the models")
parser.add_argument("-output_dir", "--output_dir", required=True, help="Output folder to store the changed models")
parser.add_argument("-matching_str", "--matching_str", required=True, help="Matching string to select the models in the folder")
parser.add_argument("-id_to_set", "--id_to_set", required=True, help="Training data ID to set")
args = vars(parser.parse_args())

# python /data/dfranco/BiaPy/biapy/utils/scripts/change_dataset_id_bmz_models.py --input_dir /data/dfranco/BMZ/bmz_final_models --tmp_dir /data/dfranco/BMZ/bmz_final_models_TMP --output_dir /data/dfranco/BMZ/bmz_final_models_OUT --matching_str "mitochondria segmentation" --id_to_set sublime-pizza

data_dir = args['input_dir']
out_dir = args['output_dir']

ids = sorted(next(os.walk(data_dir))[2])
ids =  [x for x in ids if args['matching_str'] in x]
# Read images
for n, id_ in tqdm(enumerate(ids), total=len(ids)):
    model_path = os.path.join(data_dir, id_)
    tmp_model_path = os.path.join(args['tmp_dir'], id_)
    print(f"Modifiying model: {model_path}")
    
    # Extract the model
    os.makedirs(tmp_model_path, exist_ok=True)
    with zipfile.ZipFile(model_path, 'r') as zip_ref:
        zip_ref.extractall(tmp_model_path)
    
    # Change the training_data parameter
    rdf_file_path = os.path.join(tmp_model_path, "rdf.yaml")
    with open(rdf_file_path, 'rt', encoding='utf8') as stream:
        temp_cfg = yaml.safe_load(stream)
        temp_cfg["training_data"]["id"] = args['id_to_set']
    
    # Save modified yaml
    with open(rdf_file_path, 'w', encoding='utf8') as outfile:
        yaml.dump(temp_cfg, outfile, default_flow_style=False, encoding="utf-8")
   
    shutil.make_archive(os.path.join(out_dir, id_), 'zip', tmp_model_path)
    

