Bash shell
----------

Step 0: Prepare environment 
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Clone the `repository <https://github.com/danifranco/EM_Image_Segmentation>`_: ::

    git clone https://github.com/danifranco/EM_Image_Segmentation.git 

You can set-up a development environment with all necessary dependencies creating 
directly a ``conda`` environment using the file located in `utils/env/environment.yml <https://github.com/danifranco/EM_Image_Segmentation/blob/master/utils/env/environment.yml>`_ ::
    
    conda env create -f utils/env/environment.yml


Step 1: Data preparation
~~~~~~~~~~~~~~~~~~~~~~~~
.. _data_preparation:

The data directory tree should be this: ::

    dataset/
    ├── test
    │   ├── x
    │   │   ├── testing-0001.tif
    │   │   ├── testing-0002.tif
    │   │   ├── . . .
    │   │   ├── testing-9999.tif
    │   └── y
    │       ├── testing_groundtruth-0001.tif
    │       ├── testing_groundtruth-0002.tif
    │       ├── . . .
    │       ├── testing_groundtruth-9999.tif
    └── train
        ├── x
        │   ├── training-0001.tif
        │   ├── training-0002.tif
        │   ├── . . .
        │   ├── training-9999.tif
        └── y
            ├── training_groundtruth-0001.tif
            ├── training_groundtruth-0002.tif
            ├── . . .
            ├── training_groundtruth-9999.tif

.. warning:: Ensure that images and their corresponding masks are sorted in the same way. A common approach is to fill with zeros the image number added to the filenames (as in the example). 

In case you use other directory distribution ensure to configure ``DATA.TRAIN.PATH``, ``DATA.TRAIN.MASK_PATH``, ``DATA.VAL.PATH``, ``DATA.VAL.MASK_PATH``, ``DATA.TEST.PATH`` and ``DATA.TEST.MASK_PATH`` variables accordingly. 


Step 2: Choose a template
~~~~~~~~~~~~~~~~~~~~~~~~~

Choose a template from `templates <https://github.com/danifranco/EM_Image_Segmentation/blob/master/templates>`_ directory and modify it for you specific case. Find `config.py <https://github.com/danifranco/EM_Image_Segmentation/blob/master/config/config.py>`_ all configurable variables and their description.


Step 3: Run the code
~~~~~~~~~~~~~~~~~~~~

Using, for instance, `unet_2d.yaml <https://github.com/danifranco/EM_Image_Segmentation/tree/master/templates/unet_2d.yaml>`_ template file, the code can be run as follows:

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
    # Number of the GPU to run the job in (according to 'nvidia-smi' command
    gpu_number=0                   

    # Load the environment
    conda activate EM_tools
    
    python -u main.py \
           --config $job_cfg_file \
           --result_dir $result_dir  \ 
           --dataroot $data_dir   \
           --name $job_name    \
           --run_id $job_counter  \
           --gpu $gpu_number  


Step 4: Analizing the results 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Following the example, you should see that the directory ``/home/user/exp_results/unet_basic_experiment`` has been 
created. If the same experiment is run 5 times, varying ``--run_id`` argument only, you should find the following 
directory tree: ::

    unet_basic_experiment/
    ├── config_files/
    │   └── unet_2d.yaml                                                                                                           
    ├── h5_files
    │   └── model_weights_unet_2d_1.h5
    └── results
        ├── unet_2d_1
        ├── . . .
        └── unet_2d_5
            ├── aug
            │   └── .tif files
            ├── charts
            │   ├── unet_2d_1_jaccard_index.png
            │   ├── unet_2d_1_loss.png
            │   └── model_plot_unet_2d_1.png
            ├── check_crop
            │   └── .tif files
            ├── full_image
            │   └── .tif files
            ├── full_post_processing
            │   └── .tif files
            └── per_image
                └── .tif files

- ``config_files``: directory where the .yaml filed used in the experiment is stored 
    - unet_2d.yaml: YAML configuration file used (it will be overwrited every time the code is run)
- ``h5_files``: directory where model's weights are stored 
    - model_weights_unet_2d_1.h5: model's weights file.
- ``results``: directory where all the generated checks and results will be stored. There, one folder per each run are going to be placed.
    - ``unet_2d_1``: run 1 experiment folder. 
        - ``aug``: image augmentation samples.
        - ``charts``:  
             - ``unet_2d_1_jaccard_index.png``: IoU (jaccard_index) over epochs plot (when training is done).
             - ``unet_2d_1_loss.png``: Loss over epochs plot (when training is done). 
             - ``model_plot_unet_2d_1.png``: plot of the model.
        - ``check_crop``: 
            - ``.tif files``: as there is a process of crop and merge, this files are: crops/patches from a sample file and a representation of the merge process.
        - ``full_image``: 
            - ``.tif files``: output of the model when feeding entire images (without patching). 
        - ``full_post_processing``:
            - ``.tif files``: output of the model when feeding entire images (without patching) and applying post-processing. 
        - ``per_image``:
            - ``.tif files``: predicted patches are combined again to recover the original test image. This folder contains these images. 

.. note:: 
   Here, for visualization purposes, only ``unet_2d_1`` has been described but ``unet_2d_2``, ``unet_2d_3``, ``unet_2d_4``
   and ``unet_2d_5`` will follow the same structure.
