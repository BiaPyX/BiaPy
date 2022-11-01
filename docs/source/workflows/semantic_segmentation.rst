Semantic segmentation
---------------------

The goal of this workflow is assign a label to each pixel of the input image. To run this workflow two files are required: 1) an input image and 2) its corresponding ground truth. 

In the figure below an example is depicted. There, only two labels are present: black pixels, with value 0, represent the background and white ones the mitochondria, labeled with 1. The number of classes is defined by ``MODEL.N_CLASSES`` variable.

.. list-table:: 

  * - .. figure:: ../img/FIBSEM_test_0.png
         :align: center

         Input image

    - .. figure:: ../img/FIBSEM_test_0_gt.png
         :align: center

         Corresponding ground truth 

The output in case that only two classes are present, as in this example, will be an image with the probability of being of class 1 on each pixel. If there are 3 or more classes, the output will be a multi-channel image, with the same number of channels as classes, and the same pixel in each channel will be the probability of being of the class that represents that channel number. For instance, with 3 classes, e.g. background, mitochondria and contours, the fist channel will represent background, the second mitochondria and the last contour class. 

Data preparation
~~~~~~~~~~~~~~~~

To ensure the proper operation of the library the data directory tree should be something like this: ::

    dataset/
    ├── test
    │   ├── x
    │   │   ├── testing-0001.tif
    │   │   ├── testing-0002.tif
    │   │   ├── . . .
    │   │   ├── testing-9999.tif
    │   └── y
    │       ├── testing_groundtruth-0001.tif
    │       ├── testing_groundtruth-0002.tif
    │       ├── . . .
    │       ├── testing_groundtruth-9999.tif
    └── train
        ├── x
        │   ├── training-0001.tif
        │   ├── training-0002.tif
        │   ├── . . .
        │   ├── training-9999.tif
        └── y
            ├── training_groundtruth-0001.tif
            ├── training_groundtruth-0002.tif
            ├── . . .
            ├── training_groundtruth-9999.tif

.. warning:: Ensure that images and their corresponding masks are sorted in the same way. A common approach is to fill with zeros the image number added to the filenames (as in the example). 

Configuration                                                                                                                 
~~~~~~~~~~~~~

Find in `templates/semantic_segmentation <https://github.com/danifranco/BiaPy/tree/master/templates/semantic_segmentation>`_ folder templates for this workflow. 


Results                                                                                                                 
~~~~~~~  

The results are placed in ``results`` folder under ``--result_dir`` directory with the ``--name`` given. See `Step-4-analizing-the-results <../          how_to_run/first_steps.html#step-4-analizing-the-results>`_ to find more details about the files and directories created. There
you should find something similiar to these results:


.. figure:: ../img/unet2d_prediction.gif
   :align: center                                                                                                 
                                                                                                                        
   2D U-Net model predictions. From left to right: original test images, its ground truth (GT) and the overlap between
   GT and the model's output. 
