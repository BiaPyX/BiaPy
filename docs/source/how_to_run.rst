How to run
==========

Step 0: Prepare environment 
---------------------------

Firstly, you need to clone the `repository <https://github.com/danifranco/DeepLearning_EM>`_: ::

    git clone --recursive https://github.com/danifranco/DeepLearning_EM.git

After that, you need to set up an environment to run the code. There are two ways
to do it, which both relies on Anaconda environments. 

Option A: In your shell
~~~~~~~~~~~~~~~~~~~~~~~

You can set-up a development environment with all necessary dependencies creating 
directly a ``conda`` environment using the file located in 
[env/DL_EM_base_env.yml](env/DL_EM_base_env.yml): ::
    
    conda env create -f DL_EM_base_env.yml


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

In `templates <https://github.com/danifranco/DeepLearning_EM/blob/master/templates>`_ directory are located a few different templates that could be used to start your project. Each one is suited to different settings:

- `U-Net_2D_template.py <https://github.com/danifranco/DeepLearning_EM/blob/master/templates/U-Net_2D_template.py>`_: use this template as a baseline to make segmantic segmentation with an 2D U-Net on small datasets.
- `Residual_U-Net_2D_template.py <https://github.com/danifranco/DeepLearning_EM/blob/master/templates/Residual_U-Net_2D_template.py>`_: use this template as a baseline to make segmantic segmentation with a Residual 2D U-Net on small datasets.
- `U-Net_2D_template.py <https://github.com/danifranco/DeepLearning_EM/blob/master/templates/U-Net_2D_template.py>`_: use this template as a baseline to make segmantic segmentation with an 3D U-Net on small datasets.
- `Residual_U-Net_3D_template.py <https://github.com/danifranco/DeepLearning_EM/blob/master/templates/Residual_U-Net_3D_template.py>`_: use this template as a baseline to make segmantic segmentation with a Residual 3D U-Net on small datasets.
- `FCN8_template.py <https://github.com/danifranco/DeepLearning_EM/blob/master/templates/FCN8_template.py>`_: use this template as a baseline to make segmantic segmentation with a FCN8 on small datasets.
- `big_data_template.py <https://github.com/danifranco/DeepLearning_EM/blob/master/templates/big_data_template.py>`_: same as the first, but should be used with large datasets, as it makes use of ``flow_from_directory()`` instead of ``flow()`` method. Notice that the dataset directory structure changes.

In case you are interested in reproducing one of the state-of-the-art works implemented in this project, you can use the template prepared on each case: `xiao_template.py <https://github.com/danifranco/DeepLearning_EM/blob/master/xiao_2018/xiao_template.py>`_, `cheng_template.py <https://github.com/danifranco/DeepLearning_EM/blob/master/cheng_2017/cheng_template.py>`_, `oztel_template.py <https://github.com/danifranco/DeepLearning_EM/blob/master/oztel_2017/oztel_template.py>`_ or `casser_template.py <https://github.com/danifranco/DeepLearning_EM/blob/master/casser_2018/casser_template.py>`_. 

Step 2: Data structure 
----------------------

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

However, as you should be familiarized with, when big datasets are used, which should be using a code based on 3D_template.py, the directory tree changes a little bit. This is because the usage of ``flow_from_directory()``, which needs the data to be structured as follows: ::

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

For instance, one of EM datasets used on this work can be downloaded `here <https://www.epfl.ch/labs/cvlab/data/data-em/>`_.

Step 3: Run the code
--------------------

If you are using e.g. ``bash`` shell you can declare first the needed
paths for the template: ::

    code_dir="/home/user/DeepLearning_EM"  # Path to this repo code
    data_dir="/home/user/dataset"          # Path to the dataset
    job_dir="/home/user/out_dir"           # Path where the output data will be generated
    job_id="400"                           # Just a string to identify the job
    job_counter=0                          # Number that should be increased when one need to run the same job multiple times
    gpu_number="0"                         # Number of the GPU to run the job in (according to 'nvidia-smi' command)


Option A: In Bash
~~~~~~~~~~~~~~~~~
::

    # Load the environment created first
    conda activate DL_EM_base_env     
    
    python -u template.py ${code_dir} "${data_dir}" "${job_dir}" --id "${job_id}" --rid "${job_counter}" --gpu ${gpu_number} 


Option B: Singularity container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To have a clear separation of the folder and files multiple paths are mounted
into the container. The folders ``/wd``, ``/code``, ``/data``, ``/out_dir`` are
created in the container on its creation which you should use as mount point.  :: 

    all_jobs_dir="/home/user/"  # the parent directory of job_dir

    singularity exec --nv --no-home --bind ${job_dir}/script_used/:/wd,${code_dir}:/code,${data_dir}:/data,${all_jobs_dir}:/out_dir EM_image_segmentation_imgaug.sif python -u /wd/template.py /code /data /out_dir/${job_id} --id "${job_id}" --rid "${job_counter}" --gpu ${gpu_number}

.. note::

   As the DET metric (of `Cell Tracking Challenge <http://celltrackingchallenge.net/evaluation-methodology/>`_) needs to be computed with the images placed in a few folders, the templates are prepared to write into the parent directory of ``job_dir`` (i. e. ``job_dir/..``), saving space if the same experiments are run mutiple times. However, this needs the parent directory of ``job_dir`` to be mounted as ``/out_dir``, so ``all_jobs_dir`` is declared here for that purpose. 

You can also set mounts points in read-only but the ``/out_dir`` to ensure you do 
not modify anything else: ::

    singularity exec --nv --no-home --bind ${job_dir}/script_used/:/wd:ro,${code_dir}:/code:ro,${data_dir}:/data:ro,${all_jobs_dir}:/out_dir:rw EM_image_segmentation_imgaug.sif python -u /wd/template.py /code /data /out_dir/${job_id} --id "${job_id}" --rid "${job_counter}" --gpu ${gpu_number}

For more information regarding the container please refer to `Singularity <https://sylabs.io/guides/3.6/user-guide/index.html>`_.
