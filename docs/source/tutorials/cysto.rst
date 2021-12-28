.. _cysto_tutorial:

3D Cysto Instance Segmentation
-------------------------------

Problem description
~~~~~~~~~~~~~~~~~~~

The goal is to segment and identify automatically each cysto in EM images. This is an instance segmentation problem,
which is the next step of semantic segmentation, as its requires identifying each blob unequivocally with a given
id. In this tutorial, pairs of EM 3D images (``X``) with their corresponding instance sementation annotations
(``Y``) are provided. Here a cysto dataset (not released yet) is used. In this dataset some 3D images of multiple
shapes are used for train and other cystos are used for test. Here is a sample:


.. figure:: ../video/cyst_sample.gif
  :scale: 60%
  :alt: Cysto image sample
  :align: center

  Overview of one cysto image.


Data preparation
~~~~~~~~~~~~~~~~

The data directory tree should follow the structure described `First steps -> Step 1: Data preparation <../how_to_run/first_steps.html#step-1-data-preparation>`_.


Problem resolution
~~~~~~~~~~~~~~~~~~

To produce the cysto instances two main steps are done:

* Firstly, new ``Y'`` data representations are created from the original ``Y``. This new ``Y'`` data is created with up to three channels (controlled by ``DATA.CHANNELS``). In this problem only two channels are presented: binary segmentation (referred as ``B`` in the code), contour (``C``) and distances (``D``). This way, the network will be trained with a bunch of image pairs, each containing an EM image and its ``Y'`` new data representation.

.. figure:: ../img/cysto_instance_bcd_scheme.svg
  :width: 300px
  :alt: Cysto Y data representation
  :align: center

  Process of the new ``Y'`` data representation: from instance segmentation labels (left) to contour, binary segmentation
  and distances (right). Here a patch ``(64, 64, 64)`` is presented just for visualization but the process is done for
  each full resolution image.

* These extra channels predicted by the network are used to create the final instance segmentation labels using a marked controlled watershed (MW). This process involve a few thresholds that may be adjusted depending each case: ``DATA.MW_TH1``, ``DATA.MW_TH2``, ``DATA.MW_TH3``, ``DATA.MW_TH4`` and ``DATA.MW_TH5``. Find their description in `config.py <https://github.com/danifranco/EM_Image_Segmentation/blob/master/config/config.py>`_.


Configuration file
~~~~~~~~~~~~~~~~~~

To create the YAML file you can use the template `resunet_3d_instances_bcd.yaml <https://github.com/danifranco/EM_Image_Segmentation/blob/master/templates/resunet_3d_instances_bcd.yaml>`_ which is prepared for this tutorial.

.. seealso::

   Adapt the configuration file to your specific case and see more configurable options available at `config.py <https://github.com/danifranco/EM_Image_Segmentation/blob/master/config/config.py>`_.

Run
~~~

Run the code with any of the options described in **HOW TO RUN** section that best suits you. For instance, you can run 
it through bash shell as described in: `Bash Shell -> Step 2: Run the code <../how_to_run/bash.html#step-2-run-the-code>`_.

Results
~~~~~~~

The results are placed in ``results`` folder under ``--result_dir`` directory with the ``--name`` given. See `Step-4-analizing-the-results <../          how_to_run/first_steps.html#step-4-analizing-the-results>`_ to find more details about the files and directories created. There
you should find something similiar to these results:

.. figure:: ../video/cyst_instance_prediction.gif 
  :scale: 60% 
  :alt: Cysto instance segmentation results
  :align: center                                                                
                                                                                
  Instance segmentation results for the cysto.
    

Evaluation
~~~~~~~~~~

To evaluate the quality of the results there are different options implemented for instance segmentation:

- IoU values will be printed when ``DATA.TEST.LOAD_GT`` is True, as we have GT to compare the predictions with. The results
  will be divided in: per patch, merging patches and full image depending on the options selected to True in
  ``TEST.STATS.*`` variable. Notice that the IoU are only calculated over binary channels (``BC``) and not in distances
  ones (``D`` or ``Dv2``).

- mAP for instance segmentation (introduced in :cite:p:`wei2020mitoem`) with ``TEST.MAP`` to True. It requires the path
  to the code to be set in ``PATHS.MAP_CODE_DIR``. Find `mAP_3Dvolume <https://github.com/danifranco/mAP_3Dvolume>`_ and
  more information of the implementation in :cite:p:`wei2020mitoem`. If ``TEST.VORONOI_ON_MASK`` is True separate values
  are printed, before and after applying it. Follow this steps to download have mAP ready for use:

.. code-block:: bash

     git clone https://github.com/danifranco/mAP_3Dvolume.git
     git checkout grand-challenge

- Other common matching statistics as precision, accuracy, recall, F1 and panoptic quality measured in the way Stardist
  (:cite:p:`schmidt2018cell,weigert2020star`) does. Set ``TEST.MATCHING_STATS`` to True and control the IoU thresholds
  with ``TEST.MATCHING_STATS_THS`` variable.


