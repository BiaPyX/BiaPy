Step 0: Prepare environment 
---------------------------

Firstly, you need to clone the `repository <https://github.com/danifranco/DeepLearning_EM>`_: ::

    git clone https://github.com/danifranco/DeepLearning_EM.git

After that, you need to set up an environment to run the code. There are two ways
to do it, which both relies on Anaconda environments. 

Option A: In your shell
~~~~~~~~~~~~~~~~~~~~~~~

You can set-up a development environment with all necessary dependencies creating 
directly a ``conda`` environment using the file located in 
[env/DL_EM_base_env.yml](env/DL_EM_base_env.yml): ::
    
    conda env create -f env/DL_EM_base_env.yml


Option B: Singularity container 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Another option is the usage of containers, which increases the consistency of 
reproducibility between different machines. ``conda`` is used again inside the 
container, so with this option only the host GPU, and its driver, that the 
container relies on changes.

`Singularity <https://sylabs.io/guides/3.6/user-guide/index.html>`_ (version 3.6) 
is used to create the container. The definition file is `env/EM_image_segmentation.def <https://github.com/danifranco/DeepLearning_EM/blob/master/env/EM_image_segmentation.def>`_ which should be used to create the container as 
follows: :: 

    sudo singularity build EM_image_segmentation.sif EM_image_segmentation.def

Step 1: Choose a template
-------------------------

Lucchi/Lucchi++
~~~~~~~~~~~~~~~

In `templates <https://github.com/danifranco/DeepLearning_EM/blob/master/templates>`_ directory are located a few different templates that reproduce the experiments reported for Lucchi/Lucchi++ datasets in our paper:

- `FCN32_template.py <https://github.com/danifranco/DeepLearning_EM/blob/master/templates/FCN32_template.py>`_: use this template as a baseline to make segmantic segmentation with a FCN32.
- `MultiResUNet_template.py <https://github.com/danifranco/DeepLearning_EM/blob/master/templates/MultiResUNet_template.py>`_: use this template as a baseline to make segmantic segmentation with a MultiResUNet.
- `Tiramisu_template.py <https://github.com/danifranco/DeepLearning_EM/blob/master/templates/Tiramisu_template.py>`_: use this template as a baseline to make segmantic segmentation with a Tiramisu.
- `nnU-Net_template.py <https://github.com/danifranco/DeepLearning_EM/blob/master/templates/nnU-Net_template.py>`_: use this template as a baseline to make segmantic segmentation with a nnU-Net.
- `MNet_template.py <https://github.com/danifranco/DeepLearning_EM/blob/master/templates/MNet_template.py>`_: use this template as a baseline to make segmantic segmentation with a MNet.
- `SE_U-Net_2D_template.py <https://github.com/danifranco/DeepLearning_EM/blob/master/templates/SE_U-Net_2D_template.py>`_: use this template as a baseline to make segmantic segmentation with our proposed 2D U-Net with SE blocks.
- `Residual_U-Net_2D_template.py <https://github.com/danifranco/DeepLearning_EM/blob/master/templates/Residual_U-Net_2D_template.py>`_: use this template as a baseline to make segmantic segmentation with our proposed 2D Residual U-Net.
- `FCN8_template.py <https://github.com/danifranco/DeepLearning_EM/blob/master/templates/FCN8_template.py>`_: use this template as a baseline to make segmantic segmentation with a FCN8.
- `U-Net_2D_template.py <https://github.com/danifranco/DeepLearning_EM/blob/master/templates/U-Net_2D_template.py>`_: use this template as a baseline to make segmantic segmentation with our proposed 2D U-Net.
- `Attention_U-Net_2D_template.py <https://github.com/danifranco/DeepLearning_EM/blob/master/templates/Attention_U-Net_2D_template.py>`_: use this template as a baseline to make segmantic segmentation with our proposed 2D Attention U-Net.
- `Vanilla_U-Net_3D_template.py <https://github.com/danifranco/DeepLearning_EM/blob/master/templates/Vanilla_U-Net_3D_template.py>`_: use this template as a baseline to make segmantic segmentation with the 3D Vanilla U-Net.
- `SE_U-Net_3D_template.py <https://github.com/danifranco/DeepLearning_EM/blob/master/templates/SE_U-Net_3D_template.py>`_: use this template as a baseline to make segmantic segmentation with our proposed 3D U-Net with SE blocks.
- `Attention_U-Net_3D_template.py <https://github.com/danifranco/DeepLearning_EM/blob/master/templates/Attention_U-Net_3D_template.py>`_: use this template as a baseline to make segmantic segmentation with our proposed 3D Attention U-Net.
- `U-Net_3D_template.py <https://github.com/danifranco/DeepLearning_EM/blob/master/templates/U-Net_3D_template.py>`_: use this template as a baseline to make segmantic segmentation with our proposed 3D U-Net.
- `Residual_U-Net_3D_template.py <https://github.com/danifranco/DeepLearning_EM/blob/master/templates/Residual_U-Net_3D_template.py>`_: use this template as a baseline to make segmantic segmentation with our proposed 3D Residual U-Net.
- `big_data_template.py <https://github.com/danifranco/DeepLearning_EM/blob/master/templates/big_data_template.py>`_: same as the first, but should be used with large datasets, as it makes use of ``flow_from_directory()`` instead of ``flow()`` method. Notice that the dataset directory structure changes.

In case you are interested in reproducing one of the state-of-the-art works implemented in this project, you can use the template prepared on each case: `xiao_template_V1.py <https://github.com/danifranco/EM_Image_Segmentation/blob/master/sota_implementations/xiao_2018/xiao_template_V1.py>`_, `cheng_2D_template.py <https://github.com/danifranco/EM_Image_Segmentation/blob/master/sota_implementations/cheng_2017/cheng_2D_template_V1.py>`_, `cheng_3D_template.py <https://github.com/danifranco/EM_Image_Segmentation/blob/master/sota_implementations/cheng_2017/cheng_3D_template_V1.py>`_, `oztel_template_V1.py <https://github.com/danifranco/EM_Image_Segmentation/blob/master/sota_implementations/oztel_2017/oztel_template_V1.py>`_ or `casser_template_V1.py <https://github.com/danifranco/EM_Image_Segmentation/blob/master/sota_implementations/casser_2018/casser_template_V1.py>`_. 

Kasthuri++
~~~~~~~~~~

In `Kasthuri++ <https://github.com/danifranco/EM_Image_Segmentation/tree/master/templates/Kasthuri%2B%2B>`_ you will find the templates to reproduce Kasthuri++ results reported in our paper. 

Step 2: Data structure 
----------------------

The datasets used in this work can be downloaded from the following links:

- `EPFL Hippocampus/Lucchi <https://www.epfl.ch/labs/cvlab/data/data-em/>`_.
- `Lucchi++ <https://sites.google.com/view/connectomics/>`_.
- `Kasthuri++ <https://sites.google.com/view/connectomics/>`_.

This project follows the same directory structure as `ImageDataGenerator <https://keras.io/preprocessing/image/>`_ class of Keras. The data directory tree should be this: ::
    
    dataset/
    ├── test
    │   ├── x
    │   │   ├── testing-0001.tif
    │   │   ├── testing-0002.tif
    │   │   ├── . . .
    │   └── y
    │       ├── testing_groundtruth-0001.tif
    │       ├── testing_groundtruth-0002.tif
    │       ├── . . .
    └── train
        ├── x
        │   ├── training-0001.tif
        │   ├── training-0002.tif
        │   ├── . . .
        └── y
            ├── training_groundtruth-0001.tif
            ├── training_groundtruth-0002.tif
            ├── . . .

However, when using `big_data_template.py <https://github.com/danifranco/DeepLearning_EM/blob/master/templates/big_data_template.py>`_, as it is based on ``flow_from_directory()`` from Keras, the data to be structured as follows: ::

    dataset/
    ├── test
    │   ├── x
    │   │   └── x
    │   │       ├── im0500.png
    │   │       ├── im0501.png
    │   │       ├── . . .
    │   └── y
    │       └── y
    │   │       ├── im0500.png
    │   │       ├── im0501.png
    │   │       ├── . . .
    └── train
        ├── x
        │   └── x
    	│   	├── im0500.png
    	│       ├── im0501.png
    	│       ├── . . .
        └── y
            └── y
                ├── mask_0097.tif
                ├── mask_0098.tif
    		    ├── mask_0097.tif
                ├── . . .

Step 3: Run the code
--------------------

If you are using e.g. ``bash`` shell you can declare first the needed
paths for the template: ::

    code_dir="/home/user/EM_Image_Segmentation"  # Path to this repo code
    data_dir="/home/user/dataset"                # Path to the dataset
    job_dir="/home/user/out_dir"                 # Path where the output data will be generated
    job_id="400"                                 # Just a string to identify the job
    job_counter=0                                # Number that should be increased when one need to run the same job multiple times
    gpu_number="0"                               # Number of the GPU to run the job in (according to 'nvidia-smi' command)


Option A: In Bash
~~~~~~~~~~~~~~~~~
::

    # Load the environment created first
    conda activate DL_EM_base_env     
    
    python -u template.py $code_dir $data_dir $job_dir --id $job_id --rid $job_counter --gpu $gpu_number


Option B: Singularity container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To have a clear separation of the folder and files multiple paths are mounted
into the container. The folders ``/wd``, ``/code``, ``/data``, ``/out_dir`` are
created in the container on its creation which you should use as mount point.  :: 

    all_jobs_dir="/home/user/"  # the parent directory of job_dir

    singularity exec --nv --no-home --bind ${job_dir}/script_used/:/wd,${code_dir}:/code,${data_dir}:/data,${all_jobs_dir}:/out_dir EM_image_segmentation_imgaug.sif python -u /wd/template.py /code /data /out_dir/${job_id} --id "${job_id}" --rid "${job_counter}" --gpu ${gpu_number}

For more information regarding the container please refer to `Singularity <https://sylabs.io/guides/3.6/user-guide/index.html>`_.
