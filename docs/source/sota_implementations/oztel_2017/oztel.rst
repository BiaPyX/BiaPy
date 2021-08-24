Oztel et al.
............

As part of our paper we try to reproduce other state-of-the-art approaches for EM semantic segmentation 
that do not provide code. In this case the following paper has been reproduced:

.. code-block:: bash

    Ismail Oztel, Gozde Yolcu, Ilker Ersoy, Tommi White, and Filiz Bunyak, "Mitochondria 
    segmentation in electron microscopy volumes using deep convolutional neural network", 
    2017 IEEE International Conference on Bioinformatics and Biomedicine (BIBM), IEEE, 
    2017, pp. 1195-1200.

`[Paper] <https://ieeexplore.ieee.org/document/8217827>`_ `[Our code] <https://github.com/danifranco/EM_Image_Segmentation/tree/v1.0/sota_implementations/oztel_2017>`_

We have prepared one template:

    -  `oztel_template_V0.py <https://github.com/danifranco/EM_Image_Segmentation/tree/v1.0/sota_implementations/oztel_2017/oztel_template_V0.py>`_ : exact parameters and training workflow as described in the paper.
    -  `oztel_template_V1.py <https://github.com/danifranco/EM_Image_Segmentation/tree/v1.0/sota_implementations/oztel_2017/oztel_template_V1.py>`_ : own changes made V0 to achieve better results.

The implementation is based in one file:                                        
    - `Oztel CNN <oztel_network.html>`_ : proposed network by the authors together with our proposed V1.
    - `Utils <oztel_utils.html>`_ : post-processing methods proposed by the authors and other needed functions. 
