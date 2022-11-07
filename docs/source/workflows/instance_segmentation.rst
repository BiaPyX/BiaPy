.. _instance_segmentation:

Instance segmentation
---------------------


The goal of this workflow is assign an unique id, i.e. integer, to each object of the input image. 

* **Input:** 
    * Image. 
    * Instance mask where each object is identify with a unique label. 
* **Output:**
    * Image with objects identified with a unique label. 


In the figure below an example of this workflow's **input** is depicted. Each color in the mask corresponds to a unique object.

.. list-table::

  * - .. figure:: ../img/mitoem_crop.png
         :align: center

         Input image.  

    - .. figure:: ../img/mitoem_crop_mask.png
         :align: center

         Corresponding input instance mask.


.. _instance_segmentation_data_prep:

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

Problem resolution
~~~~~~~~~~~~~~~~~~

Firstly, a **pre-processing** step is done where the new data representation is created from the input instance masks. The new data is a multi-channel mask with up to three channels (controlled by ``PROBLEM.INSTANCE_SEG.DATA_CHANNELS``). This way, the model is trained with the input images and these new multi-channel masks. Available channels to choose are the following: 

  * Binary mask (referred as ``B`` in the code), contains each instance region without the contour. This mask is binary, i.e. pixels in the instance region are ``1`` and the rest are ``0``.

  * Contour (``C``), contains each instance contour. This mask is binary, i.e. pixels in the contour are ``1`` and the rest are ``0``.

  * Distances (``D``), each pixel containing the euclidean distance of it to the instance contour. This mask is a float, not binary. 

  * Mask (``M``), contains the ``B`` and the ``C`` channels, i.e. the foreground mask. Is simply achieved by binarizing input instance masks. This mask is also binary. 

  * Updated version of distances (``Dv2``), that extends ``D`` channel by calculating the background distances as well. This mask is a float, not binary. The piecewise function is as follows:

.. figure:: ../img/Dv2_equation.svg
  :width: 300px
  :alt: Dv2 channel equation
  :align: center

  where A, B and C denote the binary mask, background and contour, respectively. ``dist`` refers to euclidean distance formula.

``PROBLEM.INSTANCE_SEG.DATA_CHANNELS`` is in charge of selecting the channels to be created. It can be set to one of the following configurations ``BC``, ``BCM``, ``BCD``, ``BCDv2``, ``Dv2`` and ``BDv2``. For instance, ``BC`` will create a 2-channel mask: the first channel will be ``B`` and the second  ``C``. In the image below the creation of 3-channel mask based on ``BCD`` is depicted:

.. figure:: ../img/cysto_instance_bcd_scheme.svg
  :width: 300px
  :alt: multi-channel mask creation
  :align: center

  Process of the new multi-channel mask creation based on ``BCD`` configuration. From instance segmentation labels (left) to contour, binary mask and distances (right). Here a small patch is presented just for the sake of visualization but the process is done for each full resolution image.

After the train phase, the model output will have the same channels as the ones used to train. In the case of binary channels, i.e. ``B``, ``C`` and ``M``, each pixel of each channel will have the probability (in ``[0-1]`` range) of being of the class that represents that channel. Whereas for the ``D`` and ``Dv2`` channel each pixel will have a float that represents the distance.

In a **post-processing** step the multi-channel data information will be used to create the final instance segmentation labels using a marker-controlled watershed. The process is as follows:

* First, instance seeds are created based on ``B``, ``C``, ``D`` and ``Dv2`` (notice that depending on the configuration selected not all of them will be present). For that, each channel is binarized using different thresholds: ``PROBLEM.INSTANCE_SEG.DATA_MW_TH1`` for ``B`` channel, ``PROBLEM.INSTANCE_SEG.DATA_MW_TH2`` for ``C`` and ``PROBLEM.INSTANCE_SEG.DATA_MW_TH4`` for ``D`` or ``Dv2``. These thresholds will decide wheter a point is labeled as a class or not. This way, the seeds are created following this formula: :: 

    seeds = (B > DATA_MW_TH1) * (D > DATA_MW_TH2) * (C < DATA_MW_TH2)  

  Translated to words seeds will be: all pixels part of the binary mask (``B`` channel), which will be those higher than ``PROBLEM.INSTANCE_SEG.DATA_MW_TH1``; and also in the center of each instances, i.e. higher than ``PROBLEM.INSTANCE_SEG.DATA_MW_TH2`` ; but not labeled as contour, i.e. less than ``PROBLEM.INSTANCE_SEG.DATA_MW_TH3``. 

* After that, each instance is labeled with a unique integer, e.g. using `connected component <https://en.wikipedia.org/wiki/Connected-component_labeling>`_. Then a foreground mask is created to delimit the area in which the seeds may grow. This foreground mask is defined based on ``B`` channel using ``PROBLEM.INSTANCE_SEG.DATA_MW_TH3``and ``D`` or ``Dv2`` using ``PROBLEM.INSTANCE_SEG.DATA_MW_TH5``. The formula is as follows: :: 

    foreground mask = (B > DATA_MW_TH3) * (D > DATA_MW_TH5) 

* Finally the seeds are grown using marker-controlled watershed.

Configuration file
~~~~~~~~~~~~~~~~~~

Find in `templates/instance_segmentation <https://github.com/danifranco/BiaPy/tree/master/templates/instance_segmentation>`__ folder of BiaPy a few YAML configuration templates for this workflow. 


Special workflow configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here some special configuration options that can be selected in this workflow are described:

* **Metrics**: during the inference phase the performance of the test data is measured using different metrics if test masks were provided (i.e. ground truth) and, consequently, ``DATA.TEST.LOAD_GT`` is enabled. In the case of instance segmentation the **Intersection over Union** (IoU), **mAP** and **matching metrics** are calculated:
  * **IoU** metric, also referred as the Jaccard index, is essentially a method to quantify the percent of overlap between the target mask and the prediction output. Depending on the configuration different values are calculated (as explained in :ref:`_config_test`). 

  * **mAP**, which is the mean average precision score adapted for 3D images (but can be used in BiaPy for 2D also). It was introduced in in :cite:p:`wei2020mitoem`. It can be enabled with ``TEST.MAP``.

  * **Matching metrics**, that was adapted from Stardist (:cite:p:`weigert2020star`) evaluation `code <https://github.com/stardist/stardist>`_. It calculates precision, recall, accuracy, F1 and panoptic quality based on a defined threshold to decide wheter an instance is a true positive, false positive,  

Run
~~~

**Command line**: Open a terminal as described in :ref:`installation`. For instance, using `resunet_3d_instances_bcd_instances.yaml <https://github.com/danifranco/BiaPy/blob/master/templates/instance_segmentation/resunet_3d_instances_bcd_instances.yaml>`__ template file, the code can be run as follows:

.. code-block:: bash
    
    # Configuration file
    job_cfg_file=/home/user/resunet_3d_instances_bcd_instances.yaml       
    # Where the experiment output directory should be created
    result_dir=/home/user/exp_results  
    # Just a name for the job
    job_name=resunet_instances_3d      
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


**Docker**: Open a terminal as described in :ref:`installation`. For instance, using `resunet_3d_instances_bcd_instances.yaml <https://github.com/danifranco/BiaPy/blob/master/templates/semantic_segmentation/resunet_3d_instances_bcd_instances.yaml>`__ template file, the code can be run as follows:

.. code-block:: bash                                                                                                    

    # Configuration file
    job_cfg_file=/home/user/resunet_3d_instances_bcd_instances.yaml
    # Where the experiment output directory should be created
    result_dir=/home/user/exp_results
    # Just a name for the job
    job_name=resunet_instances_3d
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

.. _instance_segmentation_results:

Results                                                                                                                 
~~~~~~~  

The results are placed in ``results`` folder under ``--result_dir`` directory with the ``--name`` given. An example of this workflow is depicted below:

.. figure:: ../img/unet2d_prediction.gif
   :align: center                  

   Example of semantic segmentation model predictions. From left to right: input image, its mask and the overlap between the mask and the model's output binarized. 


For the examples above, you should see that the directory ``/home/user/exp_results/resunet_instances_3d`` has been created. If the same experiment is run 5 times, varying ``--run_id`` argument only, you should find the following directory tree: ::

    resunet_instances_3d/
    ├── config_files/
    │   └── resunet_3d_instances_bcd_instances.yaml                                                                                                           
    ├── checkpoints
    │   └── model_weights_resunet_instances_3d_1.h5
    └── results
        ├── rresunet_instances_3d_1
        ├── . . .
        └── resunet_instances_3d_5
            ├── aug
            │   └── .tif files
            ├── charts
            │   ├── resunet_instances_3d_1_jaccard_index.png
            │   ├── resunet_instances_3d_1_loss.png
            │   └── model_plot_resunet_instances_3d_1.png
            ├── check_crop
            │   └── .tif files
            ├── per_image
            │   └── .tif files
            └── per_image_binarized
                └── .tif files


* ``config_files``: directory where the .yaml filed used in the experiment is stored. 

    * ``resunet_3d_instances_bcd_instances.yaml``: YAML configuration file used (it will be overwrited every time the code is run)

* ``checkpoints``: directory where model's weights are stored.

    * ``model_weights_resunet_instances_3d_1.h5``: model's weights file.

* ``results``: directory where all the generated checks and results will be stored. There, one folder per each run are going to be placed.

    * ``resunet_instances_3d_1``: run 1 experiment folder. 

        * ``aug``: image augmentation samples.

        * ``charts``:  

             * ``resunet_instances_3d_1_jaccard_index.png``: IoU (jaccard_index) over epochs plot (when training is done).

             * ``resunet_instances_3d_1_loss.png``: Loss over epochs plot (when training is done). 

             * ``model_plot_resunet_instances_3d_1.png``: plot of the model.

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
   Here, for visualization purposes, only ``resunet_instances_3d_1`` has been described but ``resunet_instances_3d_2``, ``resunet_instances_3d_3``, ``resunet_instances_3d_4``
   and ``resunet_instances_3d_5`` will follow the same structure.


    

Evaluation
~~~~~~~~~~

To evaluate the quality of the results there are different options implemented for instance segmentation:

- IoU values will be printed when ``DATA.TEST.LOAD_GT`` is True, as we have GT to compare the predictions with. The results
  will be divided in: per patch, merging patches and full image depending on the options selected to True in
  ``TEST.STATS.*`` variable. Notice that the IoU are only calculated over binary channels (``BC``) and not in distances
  ones (``D`` or ``Dv2``).

- mAP for instance segmentation (introduced in :cite:p:`wei2020mitoem`) with ``TEST.MAP`` to True. It requires the path
  to the code to be set in ``PATHS.MAP_CODE_DIR``. Find `mAP_3Dvolume <https://github.com/danifranco/mAP_3Dvolume>`__ and
  more information of the implementation in :cite:p:`wei2020mitoem`. If ``TEST.VORONOI_ON_MASK`` is True separate values
  are printed, before and after applying it. Follow this steps to download have mAP ready for use:

.. code-block:: bash

     git clone https://github.com/danifranco/mAP_3Dvolume.git
     git checkout grand-challenge

- Other common matching statistics as precision, accuracy, recall, F1 and panoptic quality measured in the way Stardist
  (:cite:p:`schmidt2018cell,weigert2020star`) does. Set ``TEST.MATCHING_STATS`` to True and control the IoU thresholds
  with ``TEST.MATCHING_STATS_THS`` variable.


