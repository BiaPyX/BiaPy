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

from biapy.utils.misc import init_devices, is_dist_avail_and_initialized, set_seed
from biapy.config.config import Config
from biapy.engine.check_configuration import check_configuration


def run_job(config, result_dir=os.getenv('HOME'), name="unknown_job", run_id=1, gpu=None,
             world_size=1, local_rank=-1, dist_on_itp=False, dist_url='env://', dist_backend='nccl'):
    """
    Run the main functionality of the job.

    Parameters:
        - config (str): Path to the configuration file.
        - result_dir (str, optional): Path to where the resulting output of the job will be stored. Defaults to the home directory.
        - name (str, optional): Job name. Defaults to "unknown_job".
        - run_id (int, optional): Run number of the same job. Defaults to 1.
        - gpu (str, optional): GPU number according to 'nvidia-smi' command. Defaults to None.
        - world_size (int, optional): Number of distributed processes. Defaults to 1.
        - local_rank (int, optional): Node rank for distributed training. Necessary for using the torch.distributed.launch utility. Defaults to -1.
        - dist_on_itp (bool, optional): If True, distributed training is performed. Defaults to False.
        - dist_url (str, optional): URL used to set up distributed training. Defaults to 'env://'.
        - dist_backend (str, optional): Backend to use in distributed mode. Should be either 'nccl' or 'gloo'. Defaults to 'nccl'.
    """

    if dist_backend not in ['nccl', 'gloo']:
        raise ValueError("Invalid value for 'dist_backend'. Should be either 'nccl' or 'gloo'.")
  
    args = argparse.Namespace(
        config=config,
        result_dir=result_dir,
        name=name,
        run_id=run_id,
        gpu=gpu,
        world_size=world_size,
        local_rank=local_rank,
        dist_on_itp=dist_on_itp,
        dist_url=dist_url,
        dist_backend=dist_backend
    )
    
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
    opts = []
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
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

    check_configuration(cfg, job_identifier)
    print("Configuration details:")
    print(cfg)

    ##########################
    #       TRAIN/TEST       #
    ##########################
    workflowname = str(cfg.PROBLEM.TYPE).lower()
    mdl = importlib.import_module('biapy.engine.'+workflowname)
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
        dist.barrier()
        dist.destroy_process_group()

    print("FINISHED JOB {} !!".format(job_identifier))

