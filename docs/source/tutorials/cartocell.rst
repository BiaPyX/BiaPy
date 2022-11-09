.. _cartocell:

CartoCell, a high-throughput pipeline for accurate 3D image analysis
--------------------------------------------------------------------

This tutorial describes how to reproduce the results reported in our paper to 
make instance segmentation of cysts in confocal microscope images:

[Paper under review]

Problem description
~~~~~~~~~~~~~~~~~~~

The goal is to segment and identify automatically each cysto in EM images. This is an instance segmentation problem, which is the next step of semantic segmentation, as its requires identifying each blob unequivocally with a given id. To solve such task pairs of 3D confocal cyst images and their corresponding instance sementation annotations are provided. Below a pair example is depicted:


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

The data directory tree need to follow the structure described in :ref:`instance_segmentation_data_prep`.


Choose a template
~~~~~~~~~~~~~~~~~

To reproduce the exact results of our manuscript you can use this configuration template: `cartocell.yaml <https://github.com/danifranco/BiaPy/templates/instance_segmentation/cartocell>`__. 


