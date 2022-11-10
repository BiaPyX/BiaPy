.. _cartocell:

CartoCell, a high-throughput pipeline for accurate 3D image analysis
--------------------------------------------------------------------

This tutorial describes how to reproduce the results of the ``M2 model`` reported in our paper to make instance segmentation of cysts in confocal microscope 3D images. The tutorial is prepared to make inference and load our pretrained model. The citation of our work is as follows:

[Paper under review]

Problem description
~~~~~~~~~~~~~~~~~~~

The goal is to segment and identify automatically each cell of each cyst in confocal microscope 3D images. To solve such task pairs of cyst images and their corresponding instance segmentation labels are provided. Below a pair example is depicted:


.. list-table:: 

  * - .. figure:: ../video/cyst_sample.gif
        :align: center
        :scale: 50%

        Cyst image sample

    - .. figure:: ../video/cyst_instance_prediction.gif 
        :align: center
        :scale: 50%

        Corresponding instance mask 

Data preparation
~~~~~~~~~~~~~~~~

You need to download `test_dataset_raw_images <https://data.mendeley.com/v1/datasets/7gbkxgngpm/draft#folder-ba6774bd-7858-4bfb-aca9-9ac307e72120>`__ and `test_dataset_ground_truth <https://data.mendeley.com/v1/datasets/7gbkxgngpm/draft#folder-efddb305-dec1-46e3-b235-00d7cd670e66>`__ data.


Choose a template
~~~~~~~~~~~~~~~~~

To reproduce the exact results of our manuscript you need to use `cartocell.yaml <https://github.com/danifranco/BiaPy/blob/master/templates/instance_segmentation/CartoCell_paper/cartocell.yaml>`__ configuration file and `model_weights_cartocell.h5 <https://github.com/danifranco/BiaPy/blob/master/templates/instance_segmentation/CartoCell_paper/model_weights_cartocell.h5>`_ to load the pretrained model.  

Then you need to modify ``TEST.PATH`` and ``TEST.MASK_PATH`` with the paths downloaded previouslty, i.e. ``test_dataset_raw_images`` and ``test_dataset_ground_truth`` respectively. You also need to modify ``PATHS.CHECKPOINT_FILE`` with the path of ``model_weights_cartocell.h5`` file.

Run
~~~

To run it via **command line** or **Docker** you can follow the same steps as decribed in :ref:`instance_segmentation_run`. 

**Colab**: |colablink|

.. |colablink| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/danifranco/BiaPy/blob/master/templates/instance_segmentation/CartoCell_paper/CartoCell_workflow.ipynb


Results
~~~~~~~

The results follow same structure as explained in :ref:`instance_segmentation_results`.

                