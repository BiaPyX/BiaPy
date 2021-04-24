Xiao et al.
...........

As part of our paper we try to reproduce other state-of-the-art approaches for EM semantic segmentation 
that do not provide code. In this case the following paper has been reproduced:

.. code-block:: bash

    Chi Xiao, Xi Chen, Weifu Li, Linlin Li, Lu Wang, Qiwei Xie, and Hua Han, "Automatic 
    mitochondria segmentation for em data using a 3d supervised convolutional network", 
    Frontiers in Neuroanatomy 12 (2018), 92.

`[Paper] <https://www.frontiersin.org/articles/10.3389/fnana.2018.00092/full>`_ `[Our code] <https://github.com/danifranco/EM_Image_Segmentation/tree/master/sota_implementations/xiao_2018>`_ 

We have prepared two templates:

    - `xiao_template_V0.py <https://github.com/danifranco/EM_Image_Segmentation/tree/master/sota_implementations/xiao_2018/xiao_template_V0.py>`_ : exact parameters and training workflow as described in the paper.
    - `xiao_template_V1.py <https://github.com/danifranco/EM_Image_Segmentation/tree/master/sota_implementations/xiao_2018/xiao_template_V1.py>`_ : changes made respect to V0 with which we have achieved better results.

The implementation is based in one file:
    - `3D network <xiao_network.html>`_ : proposed 3D network based on 3D U-Net. 

