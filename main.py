import os
import sys
import argparse
import datetime
import ntpath
import numpy as np
import tensorflow as tf

from shutil import copyfile

from utils.util import set_seed, limit_threads
from engine.trainer import Trainer
from config.config import Config


if __name__ == '__main__':

    ##########################
    #   ARGS COMPROBATION    #
    ##########################

    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--config", required=True, help="Path to the configuration file")
    parser.add_argument("-rdir", "--result_dir", help="Path to where the resulting output of the job will be stored",
                        default=os.getenv('HOME'))
    parser.add_argument("-droot", "--dataroot", help="Dataset root directory", required=True)
    parser.add_argument("-name", "--name", "--name", help="Job name", default="unknown_job")
    parser.add_argument("-rid", "--run_id", "--rid", help="Run number of the same job", type=int, default=1)
    parser.add_argument("-gpu", "--gpu", help="GPU number according to 'nvidia-smi' command", default="0", type=str)
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
    copyfile(args.config, cfg_file)

    # Merge conf file with the default settings
    cfg = Config(job_dir, job_identifier, args.dataroot)
    cfg._C.merge_from_file(cfg_file)
    cfg.update_dependencies()
    cfg = cfg.get_cfg_defaults()
    #cfg.freeze()

    now = datetime.datetime.now()
    print("Date: {}".format(now.strftime("%Y-%m-%d %H:%M:%S")))
    print("Arguments: {}".format(args))
    print("Job: {}".format(job_identifier))
    print("Python       : {}".format(sys.version.split('\n')[0]))
    print("Keras        : {}".format(tf.keras.__version__))
    print("Tensorflow   : {}".format(tf.__version__))
    print("Configuration details:")
    print(cfg)

    # GPU selection
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # CPU limit
    limit_threads(cfg.SYSTEM.NUM_CPUS)

    # Reproducibility
    set_seed(cfg.SYSTEM.SEED)

    assert cfg.PROBLEM.NDIM in ['2D', '3D']
    assert cfg.PROBLEM.TYPE in ['SEMANTIC_SEG', 'INSTANCE_SEG']

    count = 2 if cfg.PROBLEM.NDIM == '2D' else 3
    if len(cfg.DATA.TRAIN.OVERLAP) != count:
        raise ValueError("When PROBLEM.NDIM == {} DATA.TRAIN.OVERLAP tuple must be lenght {}, given {}."
                         .format(cfg.PROBLEM.NDIM, count, cfg.DATA.TRAIN.OVERLAP))
    if len(cfg.DATA.TRAIN.PADDING) != count:
        raise ValueError("When PROBLEM.NDIM == {} DATA.TRAIN.PADDING tuple must be lenght {}, given {}."
                         .format(cfg.PROBLEM.NDIM, count, cfg.DATA.TRAIN.PADDING))
    if len(cfg.DATA.TEST.OVERLAP) != count:
        raise ValueError("When PROBLEM.NDIM == {} DATA.TEST.OVERLAP tuple must be lenght {}, given {}."
                         .format(cfg.PROBLEM.NDIM, count, cfg.DATA.TEST.OVERLAP))
    if len(cfg.DATA.TEST.PADDING) != count:
        raise ValueError("When PROBLEM.NDIM == {} DATA.TEST.PADDING tuple must be lenght {}, given {}."
                         .format(cfg.PROBLEM.NDIM, count, cfg.DATA.TEST.PADDING))
    if len(cfg.DATA.PATCH_SIZE) != count+1:
        raise ValueError("When PROBLEM.NDIM == {} DATA.PATCH_SIZE tuple must be lenght {}, given {}."
                         .format(cfg.PROBLEM.NDIM, count+1, cfg.DATA.PATCH_SIZE))

    if not cfg.TEST.STATS.PER_PATCH and not cfg.TEST.STATS.FULL_IMG:
        raise ValueError("One between 'TEST.STATS.PER_PATCH' or 'TEST.STATS.FULL_IMG' need to be True")

    if cfg.PROBLEM.NDIM == '3D' and not cfg.TEST.STATS.PER_PATCH and not cfg.TEST.STATS.MERGE_PATCHES:
        raise ValueError("One between 'TEST.STATS.PER_PATCH' or 'TEST.STATS.MERGE_PATCHES' need to be True when 'PROBLEM.NDIM'=='3D'")

    if cfg.TEST.MAP and not os.path.isdir(cfg.PATHS.MAP_CODE_DIR):
        raise ValueError("mAP calculation code not found. Please set 'PATHS.MAP_CODE_DIR' variable with the path of the "
                         "Github repo 'mAP_3Dvolume': 0) git clone https://github.com/danifranco/mAP_3Dvolume.git ; "
                         "1) git checkout grand-challenge ")

    if cfg.PROBLEM.NDIM == '3D' and cfg.TEST.STATS.FULL_IMG:
        print("WARNING: cfg.TEST.STATS.FULL_IMG == True while using PROBLEM.NDIM == '3D'. As 3D images are usually 'huge'"
              ", full image statistics will be disabled to avoid GPU memory overflow")

    if cfg.PROBLEM.TYPE == 'INSTANCE_SEG':
        if cfg.DATA.CHANNELS == "B":
            raise ValueError("DATA.CHANNELS must be 'BC' or 'BCD' when PROBLEM.TYPE == 'INSTANCE_SEG'")
        else:
            if cfg.MODEL.N_CLASSES > 1:
                raise ValueError("Not implemented pipeline option")

    if cfg.DATA.VAL.FROM_TRAIN and cfg.DATA.VAL.SPLIT_TRAIN <= 0:
        raise ValueError("'DATA.VAL.SPLIT_TRAIN' needs to be > 0 when 'DATA.VAL.FROM_TRAIN' == True")


    ##########################
    #       TRAIN/TEST       #
    ##########################
    trainer = Trainer(cfg, job_identifier)

    if cfg.TRAIN.ENABLE:
        trainer.train()

    if cfg.TEST.ENABLE:
        trainer.test()

    print("FINISHED JOB {} !!".format(job_identifier))

