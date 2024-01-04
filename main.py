import argparse
import os

from biapy import run_job

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
    parser.add_argument("--gpu", help="GPU number according to 'nvidia-smi' command", type=str)

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


    run_job(**vars(args))
