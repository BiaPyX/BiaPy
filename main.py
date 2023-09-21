import os
import sys
import argparse
import datetime
import ntpath
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from shutil import copyfile
import numpy as np
import importlib
import multiprocessing

from utils.misc import init_devices, is_dist_avail_and_initialized, set_seed
from config.config import Config
from engine.check_configuration import check_configuration

if __name__ == '__main__':

    ##########################
    #   ARGS COMPROBATION    #
    ##########################

    # Normal exec:
    # python -u main.py \
    #     --config $input_job_cfg_file \        
    #     --result_dir $result_dir \        
    #     --name $job_name \          
    #     --run_id $job_counter \        
    #     --gpu 0
    # Distributed:
    # python -u -m torch.distributed.run \
    #     --nproc_per_node=2 \  
    #     main.py \
    #     --config $input_job_cfg_file \       
    #     --result_dir $result_dir \        
    #     --name $job_name \          
    #     --run_id $job_counter \        
    #     --gpu 0,1

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the configuration file")
    parser.add_argument("--result_dir", help="Path to where the resulting output of the job will be stored",
                        default=os.getenv('HOME'))
    parser.add_argument("--name", help="Job name", default="unknown_job")
    parser.add_argument("--run_id", help="Run number of the same job", type=int, default=1)
    parser.add_argument("--gpu", help="GPU number according to 'nvidia-smi' command", default="0", type=str)

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='Node rank for distributed training. Necessary for using the torch.distributed.launch utility.')
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend', type=str, default='nccl', choices=['nccl', 'gloo'],
                        help='Backend to use in distributed mode')
    args = parser.parse_args()

    ############
    #  CHECKS  #
    ############

    # Job complete name
    job_identifier = args.name + '_' + str(args.run_id)

    # Prepare working dir
    job_dir = os.path.join(args.result_dir, args.name)
    cfg_bck_dir = os.path.join(job_dir, 'config_files')
    os.makedirs(cfg_bck_dir, exist_ok=True)
    head, tail = ntpath.split(args.config)
    cfg_filename = tail if tail else ntpath.basename(head)
    cfg_file = os.path.join(cfg_bck_dir,cfg_filename)

    if not os.path.exists(args.config):
        raise FileNotFoundError("Provided {} config file does not exist".format(args.config))
    copyfile(args.config, cfg_file)

    # Merge conf file with the default settings
    cfg = Config(job_dir, job_identifier)
    cfg._C.merge_from_file(cfg_file)
    cfg.update_dependencies()
    #cfg.freeze()

    now = datetime.datetime.now()
    print("Date: {}".format(now.strftime("%Y-%m-%d %H:%M:%S")))
    print("Arguments: {}".format(args))
    print("Job: {}".format(job_identifier))
    print("Python       : {}".format(sys.version.split('\n')[0]))
    print("PyTorch: ", torch.__version__)

    # GPU selection
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    opts = []
    num_gpus = len(np.unique(np.array(args.gpu.strip().split(','))))
    opts.extend(["SYSTEM.NUM_GPUS", num_gpus])

    # GPU management
    device = init_devices(args, cfg.get_cfg_defaults())
    cfg._C.merge_from_list(opts)
    cfg = cfg.get_cfg_defaults()

    # Reproducibility
    set_seed(cfg.SYSTEM.SEED)
    
    # Number of CPU calculation
    if cfg.SYSTEM.NUM_CPUS == -1:
        cpu_count = multiprocessing.cpu_count()
    else:
        cpu_count = cfg.SYSTEM.NUM_CPUS
    torch.set_num_threads(cpu_count)
    cfg.merge_from_list(['SYSTEM.NUM_CPUS', cpu_count])

    check_configuration(cfg)
    print("Configuration details:")
    print(cfg)

    ##########################
    #       TRAIN/TEST       #
    ##########################
    workflowname = str(cfg.PROBLEM.TYPE).lower()
    mdl = importlib.import_module('engine.'+workflowname)
    names = [x for x in mdl.__dict__ if not x.startswith("_")]
    globals().update({k: getattr(mdl, k) for k in names})
    name = [x for x in names if "Base" not in x and "Workflow" in x][0]

    # Initialize workflow
    print("*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*")
    print(f"Initializing {name}")
    print("*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*\n")
    workflow = getattr(mdl, name)(cfg, job_identifier, device, args)

    if cfg.TRAIN.ENABLE:
        workflow.train()

    if cfg.TEST.ENABLE:
        workflow.test()

    if is_dist_avail_and_initialized():
        dist.destroy_process_group()

    print("FINISHED JOB {} !!".format(job_identifier))

