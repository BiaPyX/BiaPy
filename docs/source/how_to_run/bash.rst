Bash shell
----------

Ensure to complete the previous steps described in `First steps <first_steps.html>`_:

#. Prepare environment 

Set-up a development environment with all necessary dependencies creating directly a ``conda`` environment using the
file located in `utils/env/environment.yml <https://github.com/danifranco/EM_Image_Segmentation/blob/master/utils/env/environment.yml>`_ ::
    
    conda env create -f utils/env/environment.yml


#. Run the code

Using, for instance, `unet_2d.yaml <https://github.com/danifranco/EM_Image_Segmentation/tree/master/templates/unet_2d.yaml>`_ 
template file, the code can be run as follows:

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

    cd EM_Image_Segmentation

    # Load the environment
    conda activate EM_tools
    
    python -u main.py \
           --config $job_cfg_file \
           --result_dir $result_dir  \ 
           --dataroot $data_dir   \
           --name $job_name    \
           --run_id $job_counter  \
           --gpu $gpu_number  

