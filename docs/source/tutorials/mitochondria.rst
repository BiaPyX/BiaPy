.. _mito_tutorial:

2D Mitochondria segmentation
----------------------------

Problem description
~~~~~~~~~~~~~~~~~~~

The goal is to segment automatically mitochondria in EM images. This is a 
semantic segmentation problem where pairs of EM image and its corresponding 
mitochodria mask are provided. Our purpose is to segment automatically other 
mitochondria in images not used during train labeling each pixel with the 
corresponding class: background or foreground. In this example, 
`EPFL Hippocampus <https://www.epfl.ch/labs/cvlab/data/data-em/>`_ dataset is
used, so the foreground class correspond to mitochondria. 

.. list-table:: 

  * - .. figure:: ../img/FIBSEM_test_0.png
         :align: center

         EM tissue image sample.

    - .. figure:: ../img/FIBSEM_test_0_gt.png
         :align: center

         Its corresponding mask.


Data preparation                                                                                                        
~~~~~~~~~~~~~~~~                                                                                                        
                                                                                                                        
The data directory tree should follow the structure described `First steps -> Step 1: Data preparation <../how_to_run/first_steps.html#step-1-data-preparation>`_.
                                                                                                                        
                                                                                                                        
Problem resolution                                                                                                      
~~~~~~~~~~~~~~~~~~     

All the models are prepared to make semantic segmentation. The model will output the probability of each pixel of beeing
the foreground class (mitochondria in this case).


Choose a template
~~~~~~~~~~~~~~~~~

Refer to the code version `V1.0 <https://github.com/danifranco/EM_Image_Segmentation/releases/tag/v1.0>`_ in case you want to reproduce exact results of our paper. Once the code is cloned you can use any of the templates from `here <https://github.com/danifranco/EM_Image_Segmentation/tree/v1.0/templates>`_. 

Otherwise, to create the YAML file you can use the template `unet_2d.yaml <https://github.com/danifranco/EM_Image_Segmentation/blob/master/templates/unet_2d.yaml>`_ which is prepared for this tutorial.
                                                                                                                        
.. seealso::
                                                                                                                        
   Adapt the configuration file to your specific case and see more configurable options available at `config.py <https://github.com/danifranco/EM_Image_Segmentation/blob/master/config/config.py>`_.
      

Run                                                                                                                     
~~~                                                                                                                     
                                                                                                                        
Run the code with any of the options described in **HOW TO RUN** section that best suits you. For instance, you can run 
it through bash shell as described in: `Bash Shell -> Step 2: Run the code <../how_to_run/bash.html#step-2-run-the-code>`_.

  
Results                                                                                                                 
~~~~~~~  

.. figure:: ../img/unet2d_prediction.gif
   :align: center                                                                                                 
                                                                                                                        
   2D U-Net model predictions. From left to right: original test images, its ground truth (GT) and the overlap between
   GT and the model's output. 


