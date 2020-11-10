.. EM Image Segmentation documentation master file, created by
   sphinx-quickstart on Thu Aug  6 09:28:07 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to EM Image Segmentation's documentation!
=================================================

This documentation tries to explain better the code of the project `EM Image Segmentation <https://github.com/danifranco/DeepLearning_EM>`_, which is used to make semantic segmentation using U-Net based architecture for EM images. The code is based on Keras and TensorFlow as backend.

Some of state-of-the-art approaches have been reproduced and implemented to compare our method in a more robust way, with the goal of supplement the lack of information in some of those works.

.. toctree::
   :maxdepth: 2

   how_to_run  
   post_processing
   data_manipulation 
   metrics
   util
   grad_cam


.. toctree::
   :maxdepth: 2
   :caption: Networks

   networks_unet
   networks_resunet
   networks_se_unet_2d
   networks_unet_3d
   networks_resunet_3d
   networks_se_unet_3d
   networks_vanilla_unet_3d
   networks_fcn
   networks_fc_densetnet103
   networks_multiresunet
   networks_mnet


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

   adabound
   callbacks   


