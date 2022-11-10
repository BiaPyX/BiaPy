BiaPy: Bioimage analysis pipelines in Python
============================================

`BiaPy <https://github.com/danifranco/BiaPy>`_ is an open source Python library for building bioimage analysis pipelines, also called workflows. This repository is actively under development by the Biomedical Computer Vision group at the `University of the Basque Country <https://www.ehu.eus/en/en-home>`_ and the `Donostia International Physics Center <http://dipc.ehu.es/>`_. 

The library provides an easy way to create image processing pipelines that are typically used in the analysis of biology microscopy images in 2D and 3D. Namely, BiaPy contains ready-to-use solutions for the tasks of `semantic segmentation <workflows/semantic_segmentation.html>`_, `instance segmentation <workflows/instance_segmentation.html>`_, `object detection <workflows/detection.html>`_, `image denoising <workflows/denoising.html>`_, `single image super-resolution <workflows/super_resolution.html>`_, `self-supervised learning <workflows/self_supervision.html>`_ and `image classification <workflows/classification.html>`_. The source code is based on Keras/TensorFlow as backend. Given BiaPy’s deep learning based core, a machine with a graphics processing unit (GPU) is recommended for fast training and execution.

                                                                                
.. image:: img/BiaPy-workflow-examples.svg
   :width: 70%
   :align: center 

   
.. toctree::
   :maxdepth: 1
   :caption: Get started
   :glob:
   
   get_started/installation.rst
   get_started/how_it_works.rst
   get_started/configuration.rst
   get_started/select_workflow.rst
   get_started/faq.rst

.. toctree::
   :maxdepth: 1
   :caption: Workflows
   :glob:

   workflows/semantic_segmentation.rst
   workflows/instance_segmentation.rst
   workflows/detection.rst
   workflows/denoising.rst
   workflows/super_resolution.rst
   workflows/self_supervision.rst
   workflows/classification.rst
   
.. toctree::
   :maxdepth: 1
   :caption: Tutorials
   :glob:

   tutorials/stable.rst
   tutorials/cartocell.rst
   tutorials/mitoem.rst
   tutorials/nucleus.rst

   
.. toctree::                                                                    
   :maxdepth: 1
   :caption: API

   API

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Bibliography

   bibliography
   

Citation
========

This repository is the base of the following works: ::

   @Article{Franco-Barranco2021,                                                                                           
            author={Franco-Barranco, Daniel and Muñoz-Barrutia, Arrate and Arganda-Carreras, Ignacio},                        
            title={Stable Deep Neural Network Architectures for Mitochondria Segmentation on Electron Microscopy Volumes},          
            journal={Neuroinformatics},                                                                                             
            year={2021},                                                                                                            
            month={Dec},                                                                                                            
            day={02},                                                                                                               
            issn={1559-0089},                                                                                                       
            doi={10.1007/s12021-021-09556-1},                                                                                       
            url={https://doi.org/10.1007/s12021-021-09556-1}                                                                        
   }        

  @inproceedings{wei2020mitoem,
                 title={MitoEM dataset: large-scale 3D mitochondria instance segmentation from EM images},
                 author={Wei, Donglai and Lin, Zudi and Franco-Barranco, Daniel and Wendt, Nils and Liu, Xingyu and Yin, Wenjie and Huang, Xin and Gupta, Aarush and Jang, Won-Dong and Wang, Xueying and others},
                 booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
                 pages={66--76},
                 year={2020},
                 organization={Springer}
  }
  