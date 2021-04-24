Step 0: Prepare environment 
---------------------------

Firstly, you need to clone the `repository <https://github.com/danifranco/DeepLearning_EM>`_: ::

    git clone https://github.com/danifranco/DeepLearning_EM.git

After that, you need to set up an environment to run the code. There are two ways
to do it, which both relies on Anaconda environments. 

Option A: In your shell
~~~~~~~~~~~~~~~~~~~~~~~

You can set-up a development environment with all necessary dependencies creating 
directly a ``conda`` environment using the file located in `env/DL_EM_base_env.yml <https://github.com/danifranco/DeepLearning_EM/blob/master/env/DL_EM_base_env.yml>`_ ::
    
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


Step 1: Data preparation
------------------------

This project follows the same directory structure as `ImageDataGenerator <https://keras.io/preprocessing/image/>`_ class of Keras. The data directory tree should be this: ::

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

Step 1: Choose a template
-------------------------

In `templates <https://github.com/danifranco/DeepLearning_EM/blob/master/templates/v2>`_ directory are located a few different templates:

- `U-Net_2D_template.py <https://github.com/danifranco/DeepLearning_EM/blob/master/templates/v2/U-Net_2D_template.py>`_: use this template as a baseline to make segmantic segmentation with our proposed 2D U-Net.
- `U-Net_3D_template_loading_3D_images.py <https://github.com/danifranco/DeepLearning_EM/blob/master/templates/v2/U-Net_3D_template_loading_3D_images.py>`_: use this template as a baseline to make segmantic segmentation with our proposed 3D U-Net.


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
.. code-block:: bash

    # Load the environment created first
    conda activate DL_EM_base_env     
    
    python -u template.py $code_dir $data_dir $job_dir --id $job_id --rid $job_counter --gpu $gpu_number


Option B: Singularity container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To have a clear separation of the folder and files multiple paths are mounted
into the container. The folders ``/wd``, ``/code``, ``/data``, ``/out_dir`` are
created in the container on its creation which you should use as mount point.

.. code-block:: bash

    # Parent directory of job_dir
    all_jobs_dir="/home/user/" 

    singularity exec \
        --nv \
        --no-home \
        --bind $job_dir/script_used/:/wd,$code_dir:/code,$data_dir:/data,$all_jobs_dir:/out_dir \
        EM_image_segmentation_imgaug.sif \
        python -u /wd/template.py /code /data /out_dir/$job_id --id $job_id --rid $job_counter --gpu $gpu_number

.. seealso:: For more information regarding the container please refer to `Singularity <https://sylabs.io/guides/3.6/user-guide/index.html>`_.
