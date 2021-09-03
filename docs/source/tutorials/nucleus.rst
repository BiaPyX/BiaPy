.. _nucleus_tutorial:

3D Nuclei Instance Segmentation
-------------------------------

Problem description
~~~~~~~~~~~~~~~~~~~

The goal is to segment and identify automatically each cell nuclei in EM images.
This is an instance segmentation problem, which is the next step of semantic
segmentation, as its requires identifying each blob unequivocally with a given
id. In this tutorial, pairs of EM 3D images (``X``) with their corresponding instance
sementation annotations (``Y``) are provided. Here NucMM-Z dataset :cite:p:`lin2021nucmm`
is used: 

.. figure:: ../img/nucmm_z_paper_view.png
  :scale: 30%
  :alt: Nucmm-z dataset overview
  :align: center

  Overview of the NucMM-Z dataset volume. Electron microscopy (EM) volume
  covering nearly a whole zebrafish brain. Modified image from :cite:p:`lin2021nucmm`.


In this dataset 27 3D images of ``(64, 64, 64)`` are used for train while the test is
done over the whole Zebrafish volume. Here is a training sample and its ground
truth:

.. list-table:: 

  * - .. figure:: ../video/nucmm_z_volume.gif
         :align: center

         EM tissue volume (``X``).

    - .. figure:: ../video/nucmm_z_volume_mask.gif
         :align: center

         Corresponding instance labels (``Y``).


Data preparation
~~~~~~~~~~~~~~~~

The data directory tree should follow the structure described `First steps -> Step 1: Data preparation <../how_to_run/first_steps.html#step-1-data-preparation>`_.


Problem resolution
~~~~~~~~~~~~~~~~~~

To produce the nuclei instances two main steps are done:

* Firstly, new ``Y'`` data representations are created from the original ``Y``. This new ``Y'`` data is created with up to three channels (controlled by ``DATA.CHANNELS``). Binary segmentation (referred as ``B`` in the code) and contour (``C``). This way, the network will be trained with 27 image pairs provided in NucMM-Z, each containing an EM image and its ``Y'`` new data representation.

.. figure:: ../img/nucmmz_instance_bc_scheme.svg
  :width: 300px
  :alt: Nucmm-z Y data representation
  :align: center

  Process of the new ``Y'`` data representation: from instance segmentation labels (left) to contour and binary
  segmentation (right).

* These extra channels predicted by the network are used to create the final instance segmentation labels using a marked controlled watershed (MW). This process involve a few thresholds that may be adjusted depending each case: ``DATA.MW_TH1``, ``DATA.MW_TH2``, ``DATA.MW_TH3``, ``DATA.MW_TH4`` and ``DATA.MW_TH5``. Find their description in `config.py <https://github.com/danifranco/EM_Image_Segmentation/blob/master/config/config.py>`_.


Configuration file
~~~~~~~~~~~~~~~~~~

To create the YAML file you can use the template `resunet_3d_instances.yaml <https://github.com/danifranco/EM_Image_Segmentation/blob/master/templates/resunet_3d_instances.yaml>`_ which is prepared for this tutorial.

.. seealso::

   Adapt the configuration file to your specific case and see more configurable options available at `config.py <https://github.com/danifranco/EM_Image_Segmentation/blob/master/config/config.py>`_.

Run
~~~

Run the code with any of the options described in **HOW TO RUN** section that best suits you. For instance, you can run 
it through bash shell as described in: `Bash Shell -> Step 2: Run the code <../how_to_run/bash.html#step-2-run-the-code>`_.

Results
~~~~~~~

The resulting instance segmentation should be something like this:

.. figure:: ../video/nucmm_z_instances_medium.gif
  :scale: 80% 
  :alt: Nucmm-z dataset overview                                                
  :align: center                                                                
                                                                                
  Instance segmentation results on the whole dataset.
    
.. figure:: ../video/smallpart_nucmm_z_instances.gif
  :scale: 80%
  :alt: Nucmm-z dataset overview (zoomed version)
  :align: center
    
  Zoom of a small region of the instance prediction.
  

