Docker
------

Ensure to complete the previous steps described in `First steps <first_steps.html>`_. After that, you can run it with 
docker as follows:

#. You can firstly check that the code will be able to use a GPU by running: ::

    docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

#. Build the container or pull ours: ::

    # Option A)
    docker pull danifranco/em_image_segmentation

    # Option B)
    cd BiaPy
    docker build -f utils/env/Dockerfile -t em_image_segmentation .

#. Once you have the container, and selecting for example the template ``unet_2d.yaml``,  you can run it as follows:

.. code-block:: bash                                                                                                    

    # Configuration file
    job_cfg_file=/home/user/unet_2d.yaml
    # Where the experiment output directory should be created
    result_dir=/home/user/exp_results
    # Path to the dataset
    data_dir=/home/user/dataset
    # Just a name for the job
    job_name=unet_basic_experiment
    # Number that should be increased when one need to run the same job multiple times (reproducibility)
    job_counter=1
    # Number of the GPU to run the job in (according to 'nvidia-smi' command)
    gpu_number=0

    docker run --rm \
        --gpus all \
        --mount type=bind,source=$job_cfg_file,target=$job_cfg_file \
        --mount type=bind,source=$result_dir,target=$result_dir \
        --mount type=bind,source=$data_dir,target=$data_dir \
        danifranco/em_image_segmentation \
            -cfg $job_cfg_file \
            -rdir $result_dir \
            -droot $data_dir \
            -name $job_name \
            -rid $job_counter \
            -gpu $gpu_num

