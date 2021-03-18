.. EM Image Segmentation documentation master file, created by
   sphinx-quickstart on Thu Aug  6 09:28:07 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to EM Image Segmentation's documentation!
=================================================

This documentation tries to explain better the code of the project `EM Image Segmentation <https://github.com/danifranco/DeepLearning_EM>`_, which is used to make semantic segmentation for EM images. The code is based on Keras and TensorFlow as backend.

                                                                                
.. image:: img/seg.gif
           :width: 100%                                                         
           :align: center 

.. toctree::
   :maxdepth: 2
   :caption: How to run

   how_to_run  

.. toctree::                                                                    
   :maxdepth: 2                                                                 
   :caption: 2D Data

   data_2d_manipulation
   2d_generator_keras
   2d_generator


.. toctree::                                                                    
   :maxdepth: 2                                                                 
   :caption: 3D Data

   data_3d_manipulation
   3d_generator


.. toctree::
   :maxdepth: 3
   :caption: Networks

   networks_unet2d
   networks_unet3d
   networks_nnunet
   networks_fcn
   networks_fc_densetnet103
   networks_multiresunet
   networks_mnet

.. toctree::                                                                    
   :maxdepth: 3                                                                 
   :caption: Utilities 

   metrics                                                                      
   util                                                                         
   post_processing     


.. toctree::
   :maxdepth: 2
   :caption: Reproduced methods

   casser
   xiao
   cheng
   oztel   


.. toctree::
   :maxdepth: 2
   :caption: Auxiliary code

   grad_cam
   adabound
   callbacks   
   


