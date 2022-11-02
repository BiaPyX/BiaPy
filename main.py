import os
import sys
import argparse
import datetime
import ntpath
import tensorflow as tf

from shutil import copyfile

from utils.util import set_seed, limit_threads
from engine.engine import Engine
from config.config import Config
from engine.check_configuration import check_configuration

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

    if not os.path.exists(args.config):
        raise FileNotFoundError("Provided {} config file does not exist".format(args.config))
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

    # GPU selection
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # CPU limit
    limit_threads(cfg.SYSTEM.NUM_CPUS)

    # Reproducibility
    set_seed(cfg.SYSTEM.SEED)

    check_configuration(cfg)
    print("Configuration details:")
    print(cfg)

    ##########################
    #       TRAIN/TEST       #
    ##########################
    engine = Engine(cfg, job_identifier)

    if cfg.TRAIN.ENABLE:
        engine.train()

    if cfg.TEST.ENABLE:
        engine.test()

    print("FINISHED JOB {} !!".format(job_identifier))

