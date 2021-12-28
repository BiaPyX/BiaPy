Welcome to EM Image Segmentation's documentation!
=================================================

This documentation tries to explain better the code of the project `EM Image Segmentation <https://github.com/danifranco/DeepLearning_EM>`_, which is used to make semantic segmentation for EM images. The code is based on Keras and TensorFlow as backend.

                                                                                
.. image:: img/seg.gif
           :width: 100%                                                         
           :align: center 

Citation
========

This repository is the base of the following work: ::

    @misc{francobarranco2021stable,
          title={Stable deep neural network architectures for mitochondria segmentation on electron microscopy volumes},
          author={Daniel Franco-Barranco and Arrate Muñoz-Barrutia and Ignacio Arganda-Carreras},
          year={2021},
          eprint={2104.03577},
          archivePrefix={arXiv},
          primaryClass={eess.IV}
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
