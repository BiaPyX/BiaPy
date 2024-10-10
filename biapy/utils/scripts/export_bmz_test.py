import sys
import argparse

parser = argparse.ArgumentParser(
    description="Export a model into BioImage Model Zoo (BMZ) format",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument("--code_dir", required=True, help="BiaPy code dir")
parser.add_argument("--jobname", required=True, help="output CSV file name")
parser.add_argument("--config", required=True, help="Path to the configuration file")
parser.add_argument(
    "--result_dir",
    required=True, 
    help="Path to where the resulting output of the job will be stored",
)
parser.add_argument("--model_name", required=True, help="Name of the model")
parser.add_argument("--doc_file", required=True, help="Documentation file")
parser.add_argument("--bmz_folder", required=True, help="BMZ model out folder")
parser.add_argument("--gpu", required=True, help="GPU to use")
parser.add_argument(
    "--reuse_original_bmz_config",
    help="Whether to reuse previous BMZ model information",
    action="store_true",
)
args = vars(parser.parse_args())

sys.path.insert(0, args["code_dir"])
from biapy import BiaPy

biapy = BiaPy(args["config"], result_dir=args["result_dir"], name=args["jobname"], run_id=1, gpu=args["gpu"])
biapy.run_job()

# import pdb; pdb.set_trace()
if args["reuse_original_bmz_config"]:
    biapy.export_model_to_bmz(args["bmz_folder"], reuse_original_bmz_config=True)
else:
    # Create a dict with all BMZ requirements
    bmz_cfg = {}
    bmz_cfg["description"] = "Mitochondria segmentation for electron microscopy"
    bmz_cfg["authors"] = [{"name": "Daniel", "github_user": "danifranco"}]
    bmz_cfg["license"] = "CC-BY-4.0"
    bmz_cfg["tags"] = ["electron-microscopy", "mitochondria"]
    bmz_cfg["cite"] = [
        {"text": "training library", "doi": "10.1101/2024.02.03.576026"},
        {"text": "architecture", "doi": "10.1109/LGRS.2018.2802944"},
        {"text": "data", "doi": "10.48550/arXiv.1812.06024"},
        ]
    bmz_cfg["doc"] = args["doc_file"]
    bmz_cfg["model_name"] = args["model_name"]

    biapy.export_model_to_bmz(args["bmz_folder"], bmz_cfg=bmz_cfg)
