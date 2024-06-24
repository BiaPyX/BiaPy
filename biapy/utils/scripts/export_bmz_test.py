import os
import sys

code_dir = "/data/dfranco/BiaPy"
sys.path.insert(0, code_dir)
from biapy import BiaPy

biapy = BiaPy(
    "/data/dfranco/jobs/2d_instance_segmentation.yaml", 
    result_dir="/data/dfranco/exp_results", 
    name="2d_instance_segmentation", 
    run_id=1, 
    gpu=0
)
biapy.run_job()

# OPTION 1 : train + export (best)
# Create a dict with all BMZ requirements
bmz_cfg = {}
bmz_cfg['description'] = "Test model"
bmz_cfg['authors'] = [{'name': 'Daniel'}]
bmz_cfg['license'] = "CC-BY-4.0"
bmz_cfg['tags'] = [{'modality': 'electron-microscopy', 'content': 'mitochondria'}]
bmz_cfg['cite'] = [{'text': 'Gizmo et al.', 'doi': 'doi:10.1002/xyzacab123'}]
bmz_cfg['doc'] = "/data/dfranco/a.md"

# import pdb; pdb.set_trace()
# biapy.export_model_to_bmz("/data/dfranco/bmz_check", bmz_cfg=bmz_cfg)
biapy.export_model_to_bmz("/data/dfranco/bmz_check", reuse_original_bmz_config=True)

# # OPTION 2: build model + generate input/output by your own
# biapy.prepare_model()

# import torch
# bmz_cfg['test_input'] = torch.zeros((3,100,100))
# bmz_cfg['test_output'] = torch.zeros((3,100,100))
# import pdb; pdb.set_trace()
# biapy.export_model_to_bmz(bmz_cfg)