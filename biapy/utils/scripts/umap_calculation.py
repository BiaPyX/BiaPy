import os
from pathlib import Path
import re
import plotly.express as px
import umap.umap_ as umap
import sys
import argparse
import datetime
import ntpath
import yaml
import torch
import math
from shutil import copyfile
import numpy as np
from yacs.config import CfgNode as CN
import multiprocessing
import argparse

sys.path.insert(0, "/net/ark/scratch/dfranco/BiaPy")

from biapy.utils.misc import (
    load_model_checkpoint,
    update_dict_with_existing_keys,
)
from biapy.data.norm import Normalization

import biapy
from biapy.models import (
    build_model,
)
from biapy.utils.misc import (
    init_devices,
    set_seed,
    is_main_process,
    get_rank,
    to_numpy_format,
    to_pytorch_format
)
from biapy.data.data_manipulation import (
    load_and_prepare_test_data,
)
from biapy.data.data_3D_manipulation import crop_3D_data_with_overlap, merge_3D_data_with_overlap, order_dimensions

from biapy.config.config import Config, update_dependencies
from biapy.engine.check_configuration import (
    check_configuration,
    convert_old_model_cfg_to_current_version,
    diff_between_configs,
)
from biapy.data.generators import (
    create_test_generator,
    create_chunked_test_generator,
)

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="Path to the configuration file")
parser.add_argument(
    "--result_dir",
    help="Path to where the resulting output of the job will be stored",
    default=os.getenv("HOME"),
)
parser.add_argument("--name", help="Job name", default="unknown_job")
parser.add_argument("--run_id", help="Run number of the same job", type=int, default=1)
parser.add_argument(
    "--gpu",
    help="GPU number according to 'nvidia-smi' command / MPS device (Apple Silicon)",
    type=str,
)
parser.add_argument("-v", "--version", action="version", version="BiaPy version " + str(biapy.__version__))

# Distributed training parameters
parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
parser.add_argument(
    "--local_rank",
    default=-1,
    type=int,
    help="Node rank for distributed training. Necessary for using the torch.distributed.launch utility.",
)
parser.add_argument("--dist_on_itp", action="store_true")
parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")
parser.add_argument(
    "--dist_backend",
    type=str,
    default="nccl",
    choices=["nccl", "gloo"],
    help="Backend to use in distributed mode",
)
args = parser.parse_args()

# Job complete name
job_identifier = args.name + "_" + str(args.run_id)

# Prepare working dir
job_dir = os.path.join(args.result_dir, args.name)
cfg_bck_dir = os.path.join(job_dir, "config_files")
os.makedirs(cfg_bck_dir, exist_ok=True)
head, tail = ntpath.split(args.config)
cfg_filename = tail if tail else ntpath.basename(head)
cfg_file = os.path.join(cfg_bck_dir, cfg_filename)

now = datetime.datetime.now()
print("Date     : {}".format(now.strftime("%Y-%m-%d %H:%M:%S")))
print("Arguments: {}".format(args))
print("Job      : {}".format(job_identifier))
print("BiaPy    : {}".format(biapy.__version__))
print("Python   : {}".format(sys.version.split("\n")[0]))
print("PyTorch  : {}".format(torch.__version__))

if not os.path.exists(args.config):
    raise FileNotFoundError("Provided {} config file does not exist".format(args.config))
if is_main_process():
    copyfile(args.config, cfg_file)

# Merge configuration file with the default settings
cfg = Config(job_dir, job_identifier)  # type: ignore

# Translates the input config it to current version
with open(args.config, "r", encoding="utf8") as stream:
    original_cfg = yaml.safe_load(stream)
temp_cfg = CN(convert_old_model_cfg_to_current_version(original_cfg))
if cfg._C.PROBLEM.PRINT_OLD_KEY_CHANGES:
    print("The following changes were made in order to adapt the input configuration:")
    diff_between_configs(original_cfg, temp_cfg)
del original_cfg
cfg._C.merge_from_other_cfg(temp_cfg)  # type: ignore

update_dependencies(cfg)
# cfg.freeze()

# GPU selection
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
opts = []
if args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    num_gpus = len(np.unique(np.array(args.gpu.strip().split(","))))
    opts.extend(["SYSTEM.NUM_GPUS", num_gpus])

# GPU management
device = init_devices(args, cfg.get_cfg_defaults())  # type: ignore
cfg._C.merge_from_list(opts)  # type: ignore
cfg : CN = cfg.get_cfg_defaults()  # type: ignore

# Reproducibility
set_seed(cfg.SYSTEM.SEED)

# Number of CPU calculation
if cfg.SYSTEM.NUM_CPUS == -1:
    cpu_count = multiprocessing.cpu_count()
else:
    cpu_count = cfg.SYSTEM.NUM_CPUS
if cpu_count < 1:
    cpu_count = 1  # At least 1 CPU
torch.set_num_threads(cpu_count)
cfg.merge_from_list(["SYSTEM.NUM_CPUS", cpu_count])

check_configuration(cfg, job_identifier)

print(
    "#################\n"
    "MODEL PREPARATION\n"
    "#################\n"
)
saved_cfg, biapy_ckpt_version = load_model_checkpoint(
    cfg=cfg,
    jobname=job_identifier,
    model_without_ddp=None,
    device=device,
    just_extract_checkpoint_info=True,
)
# Override model specs
tmp_cfg = convert_old_model_cfg_to_current_version(saved_cfg.clone())  # type: ignore
if cfg.PROBLEM.PRINT_OLD_KEY_CHANGES:
    print("The following changes were made in order to adapt the loaded input configuration from checkpoint into the current configuration version:")
    diff_between_configs(saved_cfg, tmp_cfg)  # type: ignore
update_dict_with_existing_keys(cfg["MODEL"], tmp_cfg["MODEL"])

# Check if the merge is coherent
updated_config = cfg.clone()
updated_config["MODEL"]["LOAD_MODEL_FROM_CHECKPOINT"] = False
cfg["MODEL"]["LOAD_CHECKPOINT"] = True
check_configuration(updated_config, job_identifier)

bmz_config = {}
model_output_channels = {"type": "mask", "channels": [2]}
(
    model,
    bmz_config["model_file"],
    bmz_config["model_name"],
    model_build_kwargs,
) = build_model(cfg, model_output_channels["channels"], device)

# Prune the model form the bottleneck
del model.up_path

def new_forward(self, x):
    # Super-resolution
    if self.pre_upsampling:
        x = self.pre_upsampling(x)

    # extra large-kernel input layer
    if self.conv_in:
        x = self.conv_in(x)

    # Down
    blocks = []
    for i, layers in enumerate(zip(self.down_path, self.mpooling_layers)):
        down, pool = layers
        x = down(x)
        if i != len(self.down_path):
            blocks.append(x)
            x = pool(x)

    x = self.bottleneck(x)

    return x
from biapy.models.resunet import ResUNet
model.forward = new_forward.__get__(model, ResUNet)

print(
    "#####################\n"
    "TEST DATA PREPARATION\n"
    "#####################\n"
)
# load test data
test_zarr_data_information = None
if cfg.DATA.TEST.INPUT_ZARR_MULTIPLE_DATA:
    use_gt_path = True
    if cfg.PROBLEM.TYPE != "INSTANCE_SEG" and cfg.PROBLEM.INSTANCE_SEG.TYPE != "synapses":
        use_gt_path = False
    test_zarr_data_information = {
        "raw_path": cfg.DATA.TEST.INPUT_ZARR_MULTIPLE_DATA_RAW_PATH,
        "gt_path": cfg.DATA.TEST.INPUT_ZARR_MULTIPLE_DATA_GT_PATH,
        "use_gt_path": use_gt_path,
    }
use_gt = False
if cfg.DATA.TEST.LOAD_GT or cfg.DATA.TEST.USE_VAL_AS_TEST:
    use_gt = True
cfg.DATA.TEST.GT_PATH = cfg.DATA.TEST.INSTANCE_CHANNELS_MASK_DIR
out_data_order = cfg.DATA.TEST.INPUT_IMG_AXES_ORDER
if "C" not in cfg.DATA.TEST.INPUT_IMG_AXES_ORDER:
    out_data_order += "C"
cfg.DATA.TEST.INPUT_MASK_AXES_ORDER = out_data_order
        
(
    X_test,
    Y_test,
    test_filenames,
) = load_and_prepare_test_data(
    test_path=cfg.DATA.TEST.PATH,
    test_mask_path=cfg.DATA.TEST.GT_PATH if use_gt else None,
    multiple_raw_images=False,
    test_zarr_data_information=test_zarr_data_information,
)
norm_module = Normalization(
    type=cfg.DATA.NORMALIZATION.TYPE,
    measure_by=cfg.DATA.NORMALIZATION.MEASURE_BY,
    mask_norm="as_mask",
    out_dtype="float32" if not cfg.TEST.REDUCE_MEMORY else "float16",
    percentile_clip=cfg.DATA.NORMALIZATION.PERC_CLIP.ENABLE,
    per_lower_bound=cfg.DATA.NORMALIZATION.PERC_CLIP.LOWER_PERC,
    per_upper_bound=cfg.DATA.NORMALIZATION.PERC_CLIP.UPPER_PERC,
    lower_bound_val=cfg.DATA.NORMALIZATION.PERC_CLIP.LOWER_VALUE,
    upper_bound_val=cfg.DATA.NORMALIZATION.PERC_CLIP.UPPER_VALUE,
    mean=cfg.DATA.NORMALIZATION.ZERO_MEAN_UNIT_VAR.MEAN_VAL,
    std=cfg.DATA.NORMALIZATION.ZERO_MEAN_UNIT_VAR.STD_VAL,
)
test_norm_module = norm_module.copy()
(test_generator, data_norm, test_input) = create_test_generator(
    cfg,
    X_test,
    Y_test,
    norm_module=test_norm_module,
)
axes_order = (0, 3, 1, 2) if cfg.PROBLEM.NDIM == "2D" else (0, 4, 1, 2, 3)
axes_order_back = (0, 2, 3, 1) if cfg.PROBLEM.NDIM == "2D" else (0, 2, 3, 4, 1)

def predict_batches_in_test(x_batch):    
    in_img = to_pytorch_format(x_batch, axes_order, device)
    assert isinstance(in_img, torch.Tensor)
    return model(in_img)

print(
    "####################\n"
    "PROCESSING THE IMAGE\n"
    "####################\n"
)
# Process all the images
dtype_str = "float32" if not cfg.TEST.REDUCE_MEMORY else "float16"
features = []
labels = []
for i, current_sample in enumerate(test_generator):  # type: ignore
    current_sample_metrics = {"file": current_sample["filename"]}
    f_numbers = [i]
    if "Y" not in current_sample:
        current_sample["Y"] = None

    # Decide whether to infer by chunks or not
    discarded = False
    _, file_extension = os.path.splitext(current_sample["filename"])

    print(
        "[Rank {} ({})] Processing image (by chunks): {}".format(
            get_rank(), os.getpid(), current_sample["filename"]
        )
    )
    # Create the generator
    test_generator_chunked = create_chunked_test_generator(
        cfg,
        current_sample=current_sample,
        norm_module=norm_module,
        out_dir=cfg.PATHS.RESULT_DIR.PER_IMAGE,
        dtype_str=dtype_str,
    )
    tgen = test_generator_chunked.dataset  # type: ignore

    # Get parallel data shape is ZYX
    _, z_dim, _, y_dim, x_dim = order_dimensions(
        tgen.X_parallel_data.shape, cfg.DATA.TEST.INPUT_IMG_AXES_ORDER  # type: ignore
    )
    parallel_data_shape = [z_dim, y_dim, x_dim]
    samples_visited = {}
    for k, obj_list in enumerate(test_generator_chunked):
        sampler_ids, img, mask, patch_in_data, added_pad, norm_extra_info = obj_list

        if cfg.TEST.VERBOSE:
            print(
                "[Rank {} ({})] Patch number {} processing patches {} from {}".format(
                    get_rank(), os.getpid(), sampler_ids, patch_in_data, parallel_data_shape
                )
            )

        # Pass the batch through the model
        pred = predict_batches_in_test(img)
        # import pdb; pdb.set_trace()
        # shape_view = np.array(pred.shape).prod()
        features.extend(pred.reshape(1,-1).detach().cpu().numpy())
        # import pdb; pdb.set_trace()
        labels.extend([1])
        # labels.extend(target.detach().cpu().numpy().tolist())
        # if k == 30:
        #     break    
    tgen.close_open_files()  # type: ignore
    break

reducer = umap.UMAP(n_components=3, n_neighbors=10, metric="cosine")
projections = reducer.fit_transform(np.array(features))

fig = px.scatter(projections, x=0, y=1,
    color=labels, labels={'color': 'Labels'}
)
out_plot ="/net/ark/scratch/dfranco/umap_wasp.png"
print(f"Saving plot in {out_plot}")
fig.write_image(out_plot)

import pdb; pdb.set_trace()