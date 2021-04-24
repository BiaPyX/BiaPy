Cheng et al.
............

As part of our paper we try to reproduce other state-of-the-art approaches for EM semantic segmentation 
that do not provide code. In this case the following paper has been reproduced:

.. code-block:: bash

    H. Cheng and A. Varshney, "Volume segmentation using convolutional neural networks with 
    limited training data", 2017 IEEE International Conference on Image Processing (ICIP), 
    Beijing, 2017, pp. 590-594.

`[Paper] <https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8296349&casa_token=5b69S99XYcYAAAAA:1-kW8nB6nLKm8Fc0adC-i2OFA9CIrW-DD2dcjcIJGcDfzKYfxMv4j2-5COjyyQJ6vIjE818clA&tag=1>`_ `[Our code] <https://github.com/danifranco/EM_Image_Segmentation/tree/master/sota_implementations/cheng_2017>`_ 

We have prepared three templates:

    - `cheng_2D_template_V0.py <https://github.com/danifranco/EM_Image_Segmentation/tree/master/sota_implementations/cheng_2017/cheng_2D_template_V0.py>`_ : exact parameters and training workflow as described in the paper.
    - `cheng_2D_template_V1.py <https://github.com/danifranco/EM_Image_Segmentation/tree/master/sota_implementations/cheng_2017/cheng_2D_template_V1.py>`_ : changes made respect to V0 with which we have achieved better results.
    - `cheng_3D_template_V0.py <https://github.com/danifranco/EM_Image_Segmentation/tree/master/sota_implementations/cheng_2017/cheng_3D_template_V0.py>`_ : exact parameters and training workflow as described in the paper.
    - `cheng_3D_template_V1.py <https://github.com/danifranco/EM_Image_Segmentation/tree/master/sota_implementations/cheng_2017/cheng_3D_template_V1.py>`_ : changes made respect to V0 with which we have achieved better results.

The implementation is based in one file:                                        
    - `2D network <cheng_network.html>`_ : proposed 2D network. 
    - `3D network <cheng_3d_network.html>`_ : proposed 3D network.  
    - `loss <cheng_loss.html>`_ : jaccard based loss proposed in the paper.
    - `2D stochastic downsampling <cheng_sto2D.html>`_ : new layer proposed by the authors to make feature level augmentation.
    - `3D stochastic downsampling <cheng_sto2D.html>`_ : 3D version of the previous layer.
