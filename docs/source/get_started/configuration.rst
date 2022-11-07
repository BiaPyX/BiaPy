Configuration
-------------

To run BiaPy you need to create a plain text YAML configuration text file built with `YACS <https://github.com/rbgirshick/yacs>`_. This configuration file includes information about the hardware to use (i.e., the number of CPUs/GPUs), the work task, the model name, optional hyperparameters, the optimizer, and the paths lo load/store data from/into. As an example, a full semantic segmentation pipeline can be created as follows:


.. code-block:: yaml

     PROBLEM:
         TYPE: SEMANTIC SEG
         NDIM: 2D
     DATA:
         PATCH_SIZE: (256, 256, 1)
         TRAIN:
             PATH: /TRAIN_PATH
             MASK_PATH: /TRAIN_MASK_PATH
         VAL:
            SPLIT_TRAIN: 0.1
        TEST:
            PATH: /TEST_PATH
    AUGMENTOR:
        ENABLE: True
        RANDOM_ROT: True
    MODEL:
        ARCHITECTURE: unet
    TRAIN:
        OPTIMIZER: SGD 
        LR: 1.E−3
        BATCH_SIZE: 6
        EPOCHS: 360
    TEST:
        POST_PROCESSING:
            YZ_FILTERING: True
            
            

Find in `templates <https://github.com/danifranco/BiaPy/tree/master/templates>`_ folder examples for each workflow. If you do not know which workflow suits your case best go to `Select Workflow <select_workflow.html>`_ page.

All the options can be find in `config.py <https://github.com/danifranco/BiaPy/blob/master/config/config.py>`_ file. Nevertheless, here we will try to explain the most common ones. 

System
~~~~~~

Use ``SYSTEM.NUM_GPUS`` and ``SYSTEM.NUM_CPUS`` to set how many **GPUs** and **CPUs** you want to use. 

Problem specification
~~~~~~~~~~~~~~~~~~~~~

Use ``PROBLEM.TYPE`` to select the **type of workflow** between ``SEMANTIC_SEG``, ``INSTANCE_SEG``, ``DETECTION``, ``DENOISING``, ``SUPER_RESOLUTION``, ``SELF_SUPERVISED`` and ``CLASSIFICATION``. Use ``PROBLEM.NDIM`` to set if you are going to work with ``2D`` or ``3D`` data. 

Data management
~~~~~~~~~~~~~~~

Use ``DATA.PATCH_SIZE`` to determine the **image shape** that the workflow will use. **Dimension** order is ``(y,x,c)`` for 2D and ``(z,y,x,c)`` for 3D. 

Training **data paths** are set with ``DATA.TRAIN.PATH`` and ``DATA.TRAIN.MASK_PATH`` (if needed as it depends on the workflow). Same applies to validation if not extracted from training with ``DATA.VAL.FROM_TRAIN``, as it that case there is no need to set those. For test data, ``DATA.TEST.PATH`` need to be set if ``TEST.ENABLE`` is set. However, ``DATA.TEST.MASK_PATH`` is avoided when ``DATA.TEST.LOAD_GT`` is disabled as normally there is no test ground truth.   

There are two ways to work with the data: 1) **load all images in memory** or 2) **load only each image individually when its required**. This behaviour can be set for training, validation and test data with ``DATA.TRAIN.IN_MEMORY``, ``DATA.VAL.IN_MEMORY`` and ``DATA.TEST.IN_MEMORY`` respectively. 

When loading **training data in memory**, i.e. setting ``DATA.TRAIN.IN_MEMORY``, all the images will be loaded just once. In this process, each image will be cropped into ``DATA.PATCH_SIZE`` patches using ``DATA.TRAIN.OVERLAP`` and ``DATA.TRAIN.PADDING``. Minimum overlap is made by default and the patches always cover the entire image. In this configuration the validation data can be created from the training using ``DATA.VAL.SPLIT_TRAIN`` to set the percentage of the training data used as validation. For this ``DATA.VAL.FROM_TRAIN`` and ``DATA.VAL.IN_MEMORY`` need to be ``True``. In general, loading training data in memory is the fastest approach, but it relays on having enough memory in your computer. 

On the other hand, when **data is not in memory**, i.e. ``DATA.TRAIN.IN_MEMORY`` is disabled, on each training epoch a number of images equal to ``TRAIN.BATCH_SIZE`` are loaded from the disk to train the model. If that image does not match in shape the selected shape, i.e. ``DATA.PATCH_SIZE``, you need to select ``DATA.EXTRACT_RANDOM_PATCH`` to extract a random patch from it. As it requires loading again and again each image, this approach is slower than the first one but it saves memory.  

.. seealso::

    In general, if for some reason the images loaded are smaller than the given patch size, i.e. ``DATA.PATCH_SIZE``, there will be no option to extract a patch from it. For that purpose the variable ``DATA.REFLECT_TO_COMPLETE_SHAPE`` was created so the image can be reshaped in those dimensions to complete ``DATA.PATCH_SIZE`` shape when needed.  

In the case of **test data**, even if ``DATA.TEST.IN_MEMORY`` is selected or not, each image is cropped to ``DATA.PATCH_SIZE`` using ``DATA.TEST.OVERLAP`` and ``DATA.TEST.PADDING``. Minimum overlap is made by default and the patches always cover the entire image. If ground truth is available you can set ``DATA.TEST.LOAD_GT`` to load it and measure the performance of the model. The metrics used depends on the workflow selected.

.. seealso::

    Set ``DATA.TRAIN.RESOLUTION`` and ``DATA.TEST.RESOLUTION`` to let the model know the resolution of training and test data respectively. In training, that information will be taken into account for some data augmentations. In test, that information will be used when the user selects to remove points from predictions in detection workflow. 

Data normalization
~~~~~~~~~~~~~~~~~~

Now two options are available to **normalize the data**:

* Adjust it to **[0-1] range** which is the default option. This is done by setting ``DATA.NORMALIZATION.TYPE`` to ``'div'``. 

* **Custom normalization** providing a mean (``DATA.NORMALIZATION.CUSTOM_MEAN``) and std (``DATA.NORMALIZATION.CUSTOM_STD``). This is done by setting ``DATA.NORMALIZATION.TYPE`` to ``'custom'``. If the mean and std are both ``-1``, which is the default, those values will be calculated based on the training data. Those values will be stored in the job's folder to be read at inference phase so the test images are normalized also using same values. If mean and std are provided those values will be used. 

Data augmentation (DA)
~~~~~~~~~~~~~~~~~~~~~~
``AUGMENTOR.ENABLE`` need to be set to enable DA. Probability of each **transformation** is set by ``AUGMENTOR.DA_PROB`` variable. BiaPy offers a wide range of transformations so please refers to `config.py <https://github.com/danifranco/BiaPy/blob/master/config/config.py>`_ to see the complete list.

Model definition
~~~~~~~~~~~~~~~~
Use ``MODEL.ARCHITECTURE`` to select the model. Different **models for each workflow** are implemented in BiaPy:

* Semantic segmentation: ``unet``, ``resunet``, ``attention_unet``, ``seunet``, ``fcn32``, ``fcn8``, ``nnunet``, ``tiramisu``, ``mnet``, ``multiresunet``, ``seunet`` and ``unetr``.  

* Instance segmentation: ``unet``, ``resunet``, ``attention_unet`` and ``seunet``.

* Detection: ``unet``, ``resunet``, ``attention_unet`` and ``seunet``.

* Denoising: ``unet``, ``resunet``, ``attention_unet`` and ``seunet``.

* Super-resolution: ``edsr``. 

* Self-supervision: ``unet``, ``resunet``, ``attention_unet`` and ``seunet``.

* Classification: ``simple_cnn`` and ``EfficientNetB0``. 

For ``unet``, ``resunet``, ``attention_unet``, ``seunet`` and ``tiramisu`` architectures you can set ``MODEL.FEATURE_MAPS`` to determine the feature maps to use on each network level. In the same way, ``MODEL.DROPOUT_VALUES`` can be set for each level in those networks. For ``tiramisu`` network only the first value of those variables will be taken into account. ``MODEL.DROPOUT_VALUES`` also can be set for ``unetr`` transformer.

Use ``MODEL.BATCH_NORMALIZATION`` to use batch normalization on ``unet``, ``resunet``, ``attention_unet``, ``seunet`` and ``unetr`` models. Except this last transformer, the 3D version of those networks also supports ``Z_DOWN`` option to not make downsampling in z axis, which usually works better in anisotropic data.   

Use ``MODEL.N_CLASSES`` to set the **number of classes** without counting the background class (that should be using 0 label). With ``1`` or ``2`` classes, the problem is cosidered binary and the behaviour is the same. With more than 2 classes a multi-class problem is considered so the output of the models will have also that amount of channels. 

Finally, use ``MODEL.LOAD_CHECKPOINT`` when you want to **load a checkpoint** of the network. For instance, when you want to predict new data you can enable it while deactivating training phase disabling ``TRAIN.ENABLE``.  

Training phase
~~~~~~~~~~~~~~

Set ``TRAIN.ENABLE`` to **activate training phase**. Here you can set ``TRAIN.OPTIMIZER`` between ``SGD`` and ``ADAM`` and its learning rate with ``TRAIN.LR``. If you do not have much expertise you can use ``ADAM`` and ``1.E-4`` as starting point. 

Apart from that you need to specify **how many images will be feed into the network** at the same time with ``TRAIN.BATCH_SIZE``. E.g. if you have 100 training samples and you select a batch size of 6: ``100/6=16.6`` means that 17 batches are needed to input all training data to the network. When done an epoch is completed. 

For training you need to choose how many **epochs** to train the network with ``TRAIN.EPOCHS``. You can also set patience with ``TRAIN.PATIENCE``, which will stop the training process if no improvement in the validation data was made in those epochs. 

.. _config_test:

Test phase
~~~~~~~~~~

Set ``TEST.ENABLE`` to **activate test phase**, sometimes called also as inference or prediction. Here, if the **test images are too big** to input them directly in the GPU, e.g. 3D images, you need to set ``TEST.STATS.PER_PATCH``. With this option each test image will be cropped into ``DATA.PATCH_SIZE`` patches, pass them through the network, and then reconstruct the original image. This option will automatically calculate performance metrics per patch if the ground truth is available (enabled by ``DATA.TEST.LOAD_GT``). Here you can also set ``TEST.STATS.MERGE_PATCHES`` to calculate same metrics but once the patches have been merged into the original image.

In case that the **entire images can be placed in the GPU** you can set only ``TEST.STATS.FULL_IMG`` without ``TEST.STATS.PER_PATCH`` and ``TEST.STATS.MERGE_PATCHES`` as explained above. For simplicity this setting is only available for ``2D``. Here the performance metrics will be calculated if a ground truth the available (enabled by ``DATA.TEST.LOAD_GT``). 

You can use **test-time augmentation** setting ``TEST.AUGMENTATION``, which will create multiple augmented copies of each test image, or patch if ``TEST.STATS.PER_PATCH`` has been selected, by all possible rotations (8 copies in 2D and 16 in 3D). This will slow down the inference process but will return more robust predictions. 

You can use also use ``DATA.REFLECT_TO_COMPLETE_SHAPE`` to ensure that the patches can be made. 

.. seealso::

    If the test images are big and you have memory problems you can set ``TEST.REDUCE_MEMORY`` which will save as much memory as the library can at the price of slow down the inference process. 

Post-processing
~~~~~~~~~~~~~~~

BiaPy offers the following post-processing methods:

* Apply **binary mask** to remove everything not contained in that mask. For this ``DATA.TEST.BINARY_MASKS`` path need to be set. Only implemented in ``TEST.STATS.PER_PATCH`` option. 
* **Z axis filtering** with ``TEST.POST_PROCESSING.Z_FILTERING`` for 3D data when ``TEST.STATS.PER_PATCH`` option is set. Also, **YZ axes filtering** is implemented via ``TEST.POST_PROCESSING.YZ_FILTERING`` variable. 
* In instance segmentation workflow **Voronoi** can be used after creating the instances to ensure all cells are touching each other setting ``TEST.POST_PROCESSING.VORONOI_ON_MASK``.
* In detection worflow ``TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS`` can be used to **remove points** close to each other.