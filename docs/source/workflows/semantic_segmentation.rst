.. _semantic_segmentation:

Semantic segmentation
---------------------

The goal of this workflow is assign a class to each pixel of the input image. 

* **Input:** 
    * Image. 
    * Class mask where each pixel is labeled with an integer representing a class.
* **Output:**
    * Image with the probability of being part of each class.  

In the figure below an example of this workflow's **input** is depicted. There, only two labels are present in the mask: black pixels, with value 0, represent the background and white ones the mitochondria, labeled with 1. The number of classes is defined by ``MODEL.N_CLASSES`` variable.

.. list-table:: 

  * - .. figure:: ../img/FIBSEM_test_0.png
         :align: center

         Input image

    - .. figure:: ../img/FIBSEM_test_0_gt.png
         :align: center

         Input class mask

The **output** in case that only two classes are present, as in this example, will be an image where each pixel will have the probability of being of class 1. 

If there are **3 or more classes**, the output will be a multi-channel image, with the same number of channels as classes, and the same pixel in each channel will be the probability (in ``[0-1]`` range) of being of the class that represents that channel number. For instance, with 3 classes, e.g. background, mitochondria and contours, the fist channel will represent background, the second mitochondria and the last contour class. 

.. _semantic_segmentation_data_prep:

Data preparation
~~~~~~~~~~~~~~~~

To ensure the proper operation of the library the data directory tree should be something like this: ::

    dataset/
    ├── train
    │   ├── x
    │   │   ├── training-0001.tif
    │   │   ├── training-0002.tif
    │   │   ├── . . .
    │   │   ├── training-9999.tif
    │   └── y
    │       ├── training_groundtruth-0001.tif
    │       ├── training_groundtruth-0002.tif
    │       ├── . . .
    │       ├── training_groundtruth-9999.tif
    └── test
        ├── x
        │   ├── testing-0001.tif
        │   ├── testing-0002.tif
        │   ├── . . .
        │   ├── testing-9999.tif
        └── y
            ├── testing_groundtruth-0001.tif
            ├── testing_groundtruth-0002.tif
            ├── . . .
            ├── testing_groundtruth-9999.tif

.. warning:: Ensure that images and their corresponding masks are sorted in the same way. A common approach is to fill with zeros the image number added to the filenames (as in the example). 

Configuration                                                                                                                 
~~~~~~~~~~~~~

Find in `templates/semantic_segmentation <https://github.com/danifranco/BiaPy/tree/master/templates/semantic_segmentation>`__ folder of BiaPy a few YAML configuration templates for this workflow. 


Special workflow configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here some special configuration options that can be selected in this workflow are described:

* **Metrics**: during the inference phase the performance of the test data is measured using different metrics if test masks were provided (i.e. ground truth) and, consequently, ``DATA.TEST.LOAD_GT`` is enabled. In the case of semantic segmentation the **Intersection over Union** (IoU) metrics is calculated. This metric, also referred as the Jaccard index, is essentially a method to quantify the percent of overlap between the target mask and the prediction output. Depending on the configuration different values are calculated (as explained in :ref:`_config_test`). This values can vary a lot as stated in :cite:p:`Franco-Barranco2021`.

    * ``per patch`` values are calculated if ``TEST.STATS.PER_PATCH`` is enabled. IoU is calculated for each patch separately and then averaged. 
    * ``merge patches`` values are calculated if ``TEST.STATS.MERGE_PATCHES`` is enabled. Notice that depending on the amount of overlap/padding selected the merged image can be different than just concatenating each patch. 
    * ``full image`` values are calculated if ``TEST.STATS.FULL_IMG`` is enabled. This can be done if the model selected if fully convolutional. The results may be slightly different from ``merge patches`` as you may notice and probably no border effect will be seen. 

* **Post-processing**: When ``PROBLEM.NDIM`` is ``2D`` the post-processing will be enabled only if ``TEST.STATS.FULL_IMG`` is enabled. In that case the post-processing will process all 2D predicted images as a unique 3D stack. On the other hand, when ``PROBLEM.NDIM`` is ``3D`` the post-processing will be applied when ``TEST.STATS.PER_PATCH`` and ``TEST.STATS.MERGE_PATCHES`` is selected. In this case, each 3D predicted image will be processed individually.

    * **Z-filtering**: to apply a median filtering in ``z`` axis. Useful to maintain class coherence across 3D volumes. Enable it with ``TEST.POST_PROCESSING.Z_FILTERING`` and use ``TEST.POST_PROCESSING.Z_FILTERING_SIZE`` for the size of the median filter. 

    * **YZ-filtering**: to apply a median filtering in ``y`` and ``z`` axes. Useful to maintain class coherence across 3D volumes that can work slightly better than ``Z-filtering``. Enable it with ``TEST.POST_PROCESSING.YZ_FILTERING`` and use ``TEST.POST_PROCESSING.YZ_FILTERING_SIZE`` for the size of the median filter.  
    
.. _semantic_segmentation_data_run:

Run
~~~

**Command line**: Open a terminal as described in :ref:`installation`. For instance, using `resunet_2d_semantic_segmentation.yaml <https://github.com/danifranco/BiaPy/blob/master/templates/semantic_segmentation/resunet_2d_semantic_segmentation.yaml>`__ template file, the code can be run as follows:

.. code-block:: bash
    
    # Configuration file
    job_cfg_file=/home/user/resunet_2d_semantic_segmentation.yaml       
    # Where the experiment output directory should be created
    result_dir=/home/user/exp_results  
    # Just a name for the job
    job_name=resunet_2d      
    # Number that should be increased when one need to run the same job multiple times (reproducibility)
    job_counter=1
    # Number of the GPU to run the job in (according to 'nvidia-smi' command)
    gpu_number=0                   

    # Move where BiaPy installation resides
    cd BiaPy

    # Load the environment
    conda activate BiaPy_env
    
    python -u main.py \
           --config $job_cfg_file \
           --result_dir $result_dir  \ 
           --name $job_name    \
           --run_id $job_counter  \
           --gpu $gpu_number  


**Docker**: Open a terminal as described in :ref:`installation`. For instance, using `resunet_2d_semantic_segmentation.yaml <https://github.com/danifranco/BiaPy/blob/master/templates/semantic_segmentation/resunet_2d_semantic_segmentation.yaml>`__ template file, the code can be run as follows:

.. code-block:: bash                                                                                                    

    # Configuration file
    job_cfg_file=/home/user/resunet_2d_semantic_segmentation.yaml
    # Where the experiment output directory should be created
    result_dir=/home/user/exp_results
    # Just a name for the job
    job_name=resunet_2d
    # Number that should be increased when one need to run the same job multiple times (reproducibility)
    job_counter=1
    # Number of the GPU to run the job in (according to 'nvidia-smi' command)
    gpu_number=0

    docker run --rm \
        --gpus $gpu_number \
        --mount type=bind,source=$job_cfg_file,target=$job_cfg_file \
        --mount type=bind,source=$result_dir,target=$result_dir \
        --mount type=bind,source=$data_dir,target=$data_dir \
        danifranco/em_image_segmentation \
            -cfg $job_cfg_file \
            -rdir $result_dir \
            -name $job_name \
            -rid $job_counter \
            -gpu $gpu_number

**Colab**: The fastest and easiest way to run it is via Google Colab |colablink|

.. |colablink| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/danifranco/BiaPy/blob/master/templates/notebooks/semantic_segmentation_workflow.ipynb

.. _semantic_segmentation_results:

Results                                                                                                                 
~~~~~~~  

The results are placed in ``results`` folder under ``--result_dir`` directory with the ``--name`` given. An example of this workflow is depicted below:

.. figure:: ../img/unet2d_prediction.gif
   :align: center                  

   Example of semantic segmentation model predictions. From left to right: input image, its mask and the overlap between the mask and the model's output binarized. 


For the examples above, you should see that the directory ``/home/user/exp_results/resunet_2d`` has been created. If the same experiment is run 5 times, varying ``--run_id`` argument only, you should find the following directory tree: ::

    resunet_2d/
    ├── config_files/
    │   └── resunet_2d_semantic_segmentation.yaml                                                                                                           
    ├── checkpoints
    │   └── model_weights_resunet_2d_1.h5
    └── results
        ├── resunet_2d_1
        ├── . . .
        └── resunet_2d_5
            ├── aug
            │   └── .tif files
            ├── charts
            │   ├── resunet_2d_1_jaccard_index.png
            │   ├── resunet_2d_1_loss.png
            │   └── model_plot_resunet_2d_1.png
            ├── check_crop
            │   └── .tif files
            ├── full_image
            │   └── .tif files
            ├── full_image_binarized
            │   └── .tif files
            ├── full_post_processing
            │   └── .tif files
            ├── per_image
            │   └── .tif files
            └── per_image_binarized
                └── .tif files


* ``config_files``: directory where the .yaml filed used in the experiment is stored. 

    * ``resunet_2d_semantic_segmentation.yaml``: YAML configuration file used (it will be overwrited every time the code is run)

* ``checkpoints``: directory where model's weights are stored.

    * ``model_weights_resunet_2d_1.h5``: model's weights file.

* ``results``: directory where all the generated checks and results will be stored. There, one folder per each run are going to be placed.

    * ``resunet_2d_1``: run 1 experiment folder. 

        * ``aug``: image augmentation samples.

        * ``charts``:  

             * ``resunet_2d_1_jaccard_index.png``: IoU (jaccard_index) over epochs plot (when training is done).

             * ``resunet_2d_1_loss.png``: Loss over epochs plot (when training is done). 

             * ``model_plot_resunet_2d_1.png``: plot of the model.

        * ``full_image``: 

            * ``.tif files``: output of the model when feeding entire images (without patching). 

        * ``full_image_binarized``: 

            * ``.tif files``: Same as ``full_image`` but with the image binarized.

        * ``full_post_processing``:

            * ``.tif files``: output of the model when feeding entire images (without patching) and applying post-processing, which in this case only `y` and `z` axes filtering was selected.

        * ``per_image``:

            * ``.tif files``: predicted patches are combined again to recover the original test image. This folder contains these images. 

        * ``per_image_binarized``: 

            * ``.tif files``: Same as ``per_image`` but with the images binarized.

.. note:: 
   Here, for visualization purposes, only ``resunet_2d_1`` has been described but ``resunet_2d_2``, ``resunet_2d_3``, ``resunet_2d_4``
   and ``resunet_2d_5`` will follow the same structure.

