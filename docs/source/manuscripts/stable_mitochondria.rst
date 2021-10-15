Stable DNN architectures for mitochondria segmentation
------------------------------------------------------

This tutorial describes a how to reproduce the results reported in our paper to 
make semantic segmentation of mitochondria in electron microscopy (EM) images: ::

    @misc{francobarranco2021stable,
          title={Stable deep neural network architectures for mitochondria segmentation on electron microscopy volumes},
          author={Daniel Franco-Barranco and Arrate Muñoz-Barrutia and Ignacio Arganda-Carreras},
          year={2021},
          eprint={2104.03577},
          archivePrefix={arXiv},
          primaryClass={eess.IV}
    }


Problem description
~~~~~~~~~~~~~~~~~~~

The goal is to segment automatically mitochondria in EM images as described in :ref:`mito_tutorial`. This is a 
semantic segmentation problem where pairs of EM image and its corresponding 
mitochodria mask are provided. Our purpose is to segment automatically other 
mitochondria in images not used during train labeling each pixel with the 
corresponding class: background or foreground (mitochondria in this case). As an
example, belown are shown two images from EPFL Hippocampus dataset used in this
work: 

.. list-table:: 

  * - .. figure:: ../img/FIBSEM_test_0.png
         :align: center

         EM tissue image

    - .. figure:: ../img/FIBSEM_test_0_gt.png
         :align: center

         Corresponding mask 

Data preparation
~~~~~~~~~~~~~~~~

There are differents datasets used on the above work: 

- `EPFL Hippocampus/Lucchi <https://www.epfl.ch/labs/cvlab/data/data-em/>`_.
- `Lucchi++ <https://sites.google.com/view/connectomics/>`_.
- `Kasthuri++ <https://sites.google.com/view/connectomics/>`_.

Prepare the data as described `here <../how_to_run/first_steps.html#step-1-data-preparation>`_.


Choose a template
~~~~~~~~~~~~~~~~~

Refer to the code version `V1.0 <https://github.com/danifranco/EM_Image_Segmentation/releases/tag/v1.0>`_ in case you want to reproduce exact results of our paper. Once the code is cloned you can use any of the templates from `templates folder <https://github.com/danifranco/EM_Image_Segmentation/tree/v1.0/templates>`_. 

In case you are interested in reproducing one of the state-of-the-art works implemented in that work, you can use the template prepared on each case:

    - Casser et al: [`Paper <https://www.researchgate.net/profile/Daniel-Haehn-2/publication/329705779_Fast_Mitochondria_Segmentation_for_Connectomics/links/5c1ab85b458515a4c7eb0569/Fast-Mitochondria-Segmentation-for-Connectomics.pdf>`_] ; [`Our code <../sota_implementations/casser_2018/casser.html>`_] ; [`Template <https://github.com/danifranco/EM_Image_Segmentation/tree/v1.0/sota_implementations/casser_2018/casser_template_V1.py>`_] ; [|cassercolablink|] 
    - Cheng et al (2D): [`Paper <https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8296349>`_] ; [`Our code <../sota_implementations/cheng_2017/cheng.html>`_] ; [`Template <https://github.com/danifranco/EM_Image_Segmentation/tree/v1.0/sota_implementations/cheng_2017/cheng_2D_template_V1.py>`_] ; [|chengcolablink|] 
    - Cheng et al (3D): [`Paper <https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8296349>`_] ; [`Our code <../sota_implementations/cheng_2017/cheng.html>`_] ; [`Template <https://github.com/danifranco/EM_Image_Segmentation/tree/v1.0/sota_implementations/cheng_2017/cheng_3D_template_V1.py>`_] ; [|cheng3dcolablink|] 
    - Xiao et al: [`Paper <https://www.frontiersin.org/articles/10.3389/fnana.2018.00092/full>`_] ; [`Our code <../sota_implementations/xiao_2018/xiao.html>`_] ; [`Template <https://github.com/danifranco/EM_Image_Segmentation/tree/v1.0/sota_implementations/xiao_2018/xiao_template_V1.py>`_] ; [|xiaocolablink|] 
    - Oztel et al: [`Paper <https://ieeexplore.ieee.org/document/8217827>`_] ; [`Our code <../sota_implementations/oztel_2017/oztel.html>`_] ; [`Template <https://github.com/danifranco/EM_Image_Segmentation/tree/v1.0/sota_implementations/oztel_2017/oztel_template_V1.py>`_] ; [|oztelcolablink|] 


.. |cassercolablink| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/danifranco/EM_Image_Segmentation/blob/master/templates/sota_implementations/Casser_workflow.ipynb

.. |chengcolablink| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/danifranco/EM_Image_Segmentation/blob/master/templates/sota_implementations/Cheng_2D_workflow.ipynb

.. |cheng3dcolablink| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/danifranco/EM_Image_Segmentation/blob/master/templates/sota_implementations/Cheng_3D_workflow.ipynb

.. |xiaocolablink| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/danifranco/EM_Image_Segmentation/blob/master/templates/sota_implementations/Xiao_workflow.ipynb

.. |oztelcolablink| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/danifranco/EM_Image_Segmentation/blob/master/templates/sota_implementations/Oztel_workflow.ipynb

Run
~~~

Run the code as described `here <quick_start.html#step-2-run-the-code>`_.
