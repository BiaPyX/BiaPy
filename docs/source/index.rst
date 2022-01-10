Welcome to EM Image Segmentation's documentation!
=================================================

This documentation tries to explain better the code of the project `EM Image Segmentation <https://github.com/danifranco/DeepLearning_EM>`_, which is used to make semantic segmentation for EM images. The code is based on Keras and TensorFlow as backend.

                                                                                
.. image:: img/seg.gif
           :width: 100%                                                         
           :align: center 

Citation
========

This repository is the base of the following work: ::

    @Article{Franco-Barranco2021,                                                                                           
             author={Franco-Barranco, Daniel and Mu{\~{n}}oz-Barrutia, Arrate and Arganda-Carreras, Ignacio},                        
             title={Stable Deep Neural Network Architectures for Mitochondria Segmentation on Electron Microscopy Volumes},          
             journal={Neuroinformatics},                                                                                             
             year={2021},                                                                                                            
             month={Dec},                                                                                                            
             day={02},                                                                                                               
             issn={1559-0089},                                                                                                       
             doi={10.1007/s12021-021-09556-1},                                                                                       
             url={https://doi.org/10.1007/s12021-021-09556-1}                                                                        
    }        


.. toctree::
   :maxdepth: 1
   :caption: How to run
   :glob:

   how_to_run/colab.rst
   how_to_run/first_steps.rst
   how_to_run/bash.rst
   how_to_run/docker.rst

.. toctree::
   :maxdepth: 1
   :caption: Tutorials
   :hidden:
   :glob:

   tutorials/mitochondria.rst
   tutorials/mitochondria_instance.rst
   tutorials/nucleus.rst
   tutorials/cysto.rst

.. toctree::
   :maxdepth: 1
   :caption: Manuscripts 
   :glob:

   manuscripts/*

.. toctree::                                                                    
   :maxdepth: 1
   :caption: API

   API

.. toctree::
   :maxdepth: 1
   :caption: Bibliography

   bibliography
