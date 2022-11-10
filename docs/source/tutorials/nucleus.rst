.. _nucleus_tutorial:

NucMM dataset: 3d neuronal nuclei instance segmentation
-------------------------------------------------------

Problem description
~~~~~~~~~~~~~~~~~~~

The goal is to segment and identify automatically each cell nuclei in EM images. To solve such task pairs of EM images and their corresponding instance segmentation labels are provided. Below a pair example is depicted:


.. list-table:: 

  * - .. figure:: ../video/nucmm_z_volume.gif
         :align: center

         MitoEM-H tissue image sample. 

    - .. figure:: ../video/nucmm_z_volume_mask.gif
         :align: center

         Its corresponding instance mask.


In this dataset ``27`` 3D images of size ``(64, 64, 64)`` voxels, for ``(z,x,y)`` axes, are used for train while the test is
done over the whole Zebrafish volume. Here is a training sample and its ground truth:


.. figure:: ../img/nucmm_z_paper_view.png
  :scale: 30%
  :alt: Nucmm-z dataset overview
  :align: center

  Overview of the NucMM-Z dataset volume. Electron microscopy (EM) volume
  covering nearly a whole zebrafish brain. Modified image from :cite:p:`lin2021nucmm`.


Data preparation
~~~~~~~~~~~~~~~~
      
You need to download NucMM dataset first from these `link <https://drive.google.com/drive/folders/1_4CrlYvzx0ITnGlJOHdgcTRgeSkm9wT8>`__. Once you have donwloaded this data you need to create a directory tree as described in :ref:`instance_segmentation_data_prep`. To adapt the ``.h5`` file format provided by MitoEM authors into ``.tif`` files you can use the script `h5_to_tif.py <https://github.com/danifranco/BiaPy/blob/master/utils/scripts/h5_to_tif.py>`__.

Configuration file
~~~~~~~~~~~~~~~~~~

To create the YAML file you can use the template `resunet_3d_instances.yaml <https://github.com/danifranco/BiaPy/blob/master/templates/instance_segmentation/resunet_3d_instances.yaml>`_ which is prepared for this tutorial.

Run
~~~

To run it via **command line** or **Docker** you can follow the same steps as decribed in :ref:`instance_segmentation_run`. 

Results
~~~~~~~

The results follow same structure as explained in :ref:`instance_segmentation_results`. The results should be something like the following:


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
  
