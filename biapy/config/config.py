"""
Configuration management for BiaPy.

This module defines the Config class, which encapsulates all configuration options
for BiaPy workflows using a YACS CfgNode. It provides default values and hierarchical
organization for system, data, augmentation, model, loss, training, inference,
post-processing, and logging parameters. Utility functions are included for updating
dependent configuration variables after merging user-provided configs.
"""
import os
from yacs.config import CfgNode as CN
import copy
from typing import (
    Dict,
)

class Config:
    """
    BiaPy configuration handler.

    This class manages the hierarchical configuration for BiaPy experiments,
    including system resources, problem specification, data loading, augmentation,
    model architecture, loss functions, training, inference, post-processing, and
    logging. It uses YACS CfgNode for flexible and robust configuration management.

    Attributes
    ----------
    _C : CN
        The root YACS configuration node containing all experiment parameters.
    """

    def __init__(self, job_dir: str, job_identifier: str):
        """
        Initialize the Config object with default values.

        Parameters
        ----------
        job_dir : str
            Directory where job outputs (results, checkpoints, logs) will be stored.
        job_identifier : str
            Unique identifier for the job (used in output paths).
        """
        if "/" in job_identifier:
            raise ValueError("Job name can not contain / character. Provided: {}".format(job_identifier))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Config definition
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        _C = CN()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # System
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        _C.SYSTEM = CN()
        # Maximum number of CPUs to use. Set it to "-1" to not set a limit.
        _C.SYSTEM.NUM_CPUS = -1
        # Maximum number of workers to use. You can disable this option by setting 0. With a -1 the workers are calculated automatically.
        _C.SYSTEM.NUM_WORKERS = -1
        # Do not set it as its value will be calculated based in --gpu input arg
        _C.SYSTEM.NUM_GPUS = 0
        # Device to be used when GPU is NOT selected. Most commonly "cpu", but also potentially "mps",
        # "xpu", "xla" or "meta".
        _C.SYSTEM.DEVICE = "cpu"

        # Math seed to generate random numbers. Used to ensure reproducibility in the results.
        _C.SYSTEM.SEED = 0
        # Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
        _C.SYSTEM.PIN_MEM = True

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Problem specification
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        _C.PROBLEM = CN()
        # Whether to check what changed to adapt the given input configuration to the newest. Basically to now what 
        # biapy.engine.check_configuration file's convert_old_model_cfg_to_current_version() function changes
        _C.PROBLEM.PRINT_OLD_KEY_CHANGES = True
        # Possible options: 'SEMANTIC_SEG', 'INSTANCE_SEG', 'DETECTION', 'DENOISING', 'SUPER_RESOLUTION',
        # 'SELF_SUPERVISED', 'CLASSIFICATION' and 'IMAGE_TO_IMAGE'
        _C.PROBLEM.TYPE = "SEMANTIC_SEG"
        # Possible options: '2D' and '3D'
        _C.PROBLEM.NDIM = "2D"

        ### SEMANTIC_SEG
        _C.PROBLEM.SEMANTIC_SEG = CN()
        # Class id to ignore when DATA.N_CLASSES > 2
        _C.PROBLEM.SEMANTIC_SEG.IGNORE_CLASS_ID = 0

        ### INSTANCE_SEG
        _C.PROBLEM.INSTANCE_SEG = CN()
        # Type of instances expected. Options are: ["regular", "synapses"]
        _C.PROBLEM.INSTANCE_SEG.TYPE = "regular" 

        #### For "regular" type of instances ####
        # This variable defines the channels to be used to represent the instances. The approach follows a bottom-up setting,
        # where the intermediate representations of the instances are learned by the model and then fused to create the final
        # instances. The variable needs to be a list with the representations to be extracted from the instances. 
        # These are the options available:
        #   - 'F' stands for 'Foreground', it is a binary representation of each instance
        #   - 'B' stands for 'Background', it is a binary representation of the background, i.e. everything that is not an instance.
        #   - 'P' stands for 'Central part', the extreme case of the F channel, where only the center of mass, or the skeleton is 
        #     represented. It may also be expressed as a heatmap.
        #   - 'C' stands for 'Contour', it is a binary representation of the countours of each instance.
        #   - 'H' stands for 'Horizontal distance', where, for each instance, the pixel/voxel value is the distance to its contour/skeleton 
        #     in the horizontal axis.
        #   - 'V' stands for 'Vertical distance', where, for each instance, the pixel/voxel value is the distance to its contour/skeleton 
        #     in the vertical axis.
        #   - 'Z' stands for 'Depth distance', where, for each instance, the pixel/voxel value is the distance to its contour/skeleton 
        #     in the depth axis.
        #   - 'Db' stands for 'Distance to the boundary', where, for each instance, the pixel/voxel value is the distance to its contour.
        #   - 'Dc' stands for 'Distance to the center', where, for each instance, the pixel/voxel value is the distance to its 
        #     center/skeleton.
        #   - 'Dn' stands for 'Distance to the closest neighbor', where, for each instance, the pixel/voxel value is the distance to its 
        #     closest neighbor instance.
        #   - 'D' stands for 'Db for F and B channels' distance map'. In some cases, background pixel distances are negated and combined 
        #     with a Tanh activation, 
        #     while in others they are not
        #   - 'R' stands for 'Radial distance', distances to boundaries measure how far each pixel is from its object’s edge along fixed 
        #     radial directions, with StarDist being the most prominent approach.
        #   - 'T' stands for 'Touching area', where touching regions between instances are highlighted.
        #   - 'A' stands for 'Affinities' and contains the affinity values for each dimension. Affinities represent images as graphs for 
        #     segmentation, where each voxel is a node connected to neighbors by edges indicating shared object membership. In 3D, nodes 
        #     link along x-, y-, and z-axes, yielding multiple affinity channels, commonly used in connectomics.
        #   - 'E' stands for 'Embeddings' and contains the embedding vectors for each pixel/voxel. Embedding-based approaches map each 
        #     pixel in an input image to a point in an n-dimensional feature space (typically n=24), referred to as the pixel embedding. 
        #     The goal is to ensure that embeddings from the same instance are close together, while embeddings from different instances
        #     are well separated. We grouped here all the channels involved in this process
        #
        #### For "synapse" type of instances ####
        # Possible options: 'B' and 'BF' (still experimental). This variable defines the channels to be used to represent synapse instances based on the input 
        # synapse sites. The meaning of each letter is a follows:
        #   * In 'B', the first and second channels represent the pre and post points as 3D points, respectively.
        #
        #   * In 'BF':
        #       - 'B' stands for 'Binary mask', it is a binary representation of each postsynaptic site
        #       - 'F' stands for 'Flow' and contains the distance values to the corresponding presynaptic site (of each postsynaptic 
        #         site) for each dimension.
        _C.PROBLEM.INSTANCE_SEG.DATA_CHANNELS = ["B", "C"]
        # Details for each channel. It must be a list with a unique element: a dict of dicts. The details can be only set for the following channels:
        #   - 'F' channel. Possible options:
        #       - 'erosion': int, the erosion size to be applied to the channel. Default: 0
        #       - 'dilation': int, the dilation size to be applied to the channel. Default: 0
        #   - 'B' channel. Possible options:
        #       - 'erosion': int, the erosion size to be applied to the channel. Default: 0
        #       - 'dilation': int, the dilation size to be applied to the channel. Default: 0
        #   - 'P' channel. Possible options:
        #       - 'type': str, the type of the channel. Options are: 'centroid', 'skeleton'. Default: 'centroid'
        #       - 'dilation': int, the dilation size to be applied to the channel. Default: 0
        #   - 'C' channel. Possible options:
        #       - 'mode': str, how to create the contours. Corresponds to 'mode' arg of find_boundaries function from scikit-image. 
        #          More info in: https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.find_boundaries.
        #          Options are: 'thick', 'inner', 'outer', 'subpixel' and 'dense'. The last one is used to label as contour every pixel
        #          that is not in the foreground. Default: 'thick'.
        #   - 'H', 'V', 'Z' and 'Db' channels. Possible options:
        #       - 'norm': bool, whether to normalize the distances between 0 and 1. Default: False
        #       - 'mask_values': bool, whether to mask the distance channel to only calculate the loss in non-zero values. Default: True
        #   - 'Dc' channel. Possible options:
        #       - 'type': str, the type of the channel. Options are: 'center', 'skeleton'. Default: 'center'
        #       - 'norm': bool, whether to normalize the distances between 0 and 1. Default: False
        #       - 'mask_values': bool, whether to mask the distance channel to only calculate the loss in non-zero values. Default: True
        #   - 'Dn' channel. Possible options:
        #       - 'closing_size': int, the size of the closing to be applied to the combined distance map. Default: 0
        #       - 'norm': bool, whether to normalize the distances between 0 and 1. Default: False
        #       - 'mask_values': bool, whether to mask the distance channel to only calculate the loss in non-zero values. Default: True
        #       - 'decline_power': int, the power to which the distances are raised to control the decline rate. Default: 3
        #   - 'D' channel. Possible options:
        #       - 'act': str, activation function to be used in the last layer of the model when this channel is selected. Options are: 
        #         'tanh' and 'linear'. Default: 'tanh'
        #       - 'alpha': int, value to scale the distances of the background when 'act' is 'tanh'. Default: 1
        #       - 'beta': int, value to scale the distances of the foreground when 'act' is 'tanh'. Default: 1
        #       - 'norm': bool, whether to normalize the distances between -1 and 1. Default: False
        #   - 'R' channel. Possible options:
        #       - 'nrays': int, the number of rays to be used to represent the radial distances. Default: 32 (in 2D) and 96 (in 3D)
        #       - 'norm': bool, whether to normalize the distances between 0 and 1. Default: False
        #       - 'mask_values': bool, whether to mask the distance channel to only calculate the loss in non-zero values. Default: True
        #   - 'T' channel. Possible options:
        #       - 'thickness': int, the thickness in pixels of the touching area. Default: 2
        #   - 'A' channel. Possible options:
        #       - 'z_affinities': list of int, the offsets in x axis to be calculated. Default: [1]
        #       - 'y_affinities': list of int, the offsets in y axis to be calculated. Default: [1]
        #       - 'x_affinities': list of int, the offsets in z axis to be calculated. Default: [1]
        #       - 'widen_borders': int, the size in pixels to widen the borders of the affinities. Default: 1
        #       Notice that 'z_affinities', 'y_affinities' and 'x_affinities'  need to be lists of the same length. To reproduce the paper
        #       "Superhuman Accuracy on the SNEMI3D Connectomics Challenge" the values are: 
        #       'z_affinities': [1,2,3,4], 'y_affinities': [1,3,9,27] and 'x_affinities': [1,3,9,27]
        #   - 'E' channel. Possible options:
        #       - 'features': int, the number of features to be learned per pixel/voxel. Default: 24
        _C.PROBLEM.INSTANCE_SEG.DATA_CHANNELS_EXTRA_OPTS = [{}]
        # Losses to be applied to each channel. If not provided, the losses will be set automatically depending on the channel.
        # The options are:
        #  - "bce"/"ce": binary cross entropy. Ref: https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
        #  - "l1"/"mae": mean absolute error. Ref: https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html
        #  - "mse": mean squared error. Ref: https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
        #  - "triplet": triplet loss. Ref: https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html
        _C.PROBLEM.INSTANCE_SEG.DATA_CHANNELS_LOSSES = []

        # Weights to be applied to the channels.
        _C.PROBLEM.INSTANCE_SEG.DATA_CHANNEL_WEIGHTS = (1, 1)
        # Whether to add an extra weight map to the loss calculation to focus on the borders between instances. Different weighting maps 
        # can be defined. Options: ["unet-like"]
        _C.PROBLEM.INSTANCE_SEG.BORDER_EXTRA_WEIGHTS = ""
        # Defines how the instances are created. Options:
        #   - "watershed" to use watershed algorithm
        #   - "agglomeration" to use agglomeration algorithm
        #   - "stardist" to use agglomeration algorithm
        _C.PROBLEM.INSTANCE_SEG.INSTANCE_CREATION_PROCESS = ""

        # Options for marker-controlled watershed
        _C.PROBLEM.INSTANCE_SEG.WATERSHED = CN()
        # List of the channels to be used for seed creation. If not provided will be automatically set based on
        # 'PROBLEM.INSTANCE_SEG.DATA_CHANNELS'.
        _C.PROBLEM.INSTANCE_SEG.WATERSHED.SEED_CHANNELS = []
        # Thresholds for the seed channels. If not provided will be automatically set with "auto" for each channel.
        # If set it must have same length than 'PROBLEM.INSTANCE_SEG.WATERSHED.SEED_CHANNELS'. Options:
        #   - A float between 0 and 1 to threshold each channel
        #   - "auto" to automatically define a value by applying an Otsu thresholding
        _C.PROBLEM.INSTANCE_SEG.WATERSHED.SEED_CHANNELS_THRESH = []
        # Defines the topographic surface to grow the seeds. If not provided will be automatically set based on
        # 'PROBLEM.INSTANCE_SEG.DATA_CHANNELS'. If defined it must be a channel name, e.g. "B" or "C". 
        _C.PROBLEM.INSTANCE_SEG.WATERSHED.TOPOGRAPHIC_SURFACE_CHANNEL = ""
        # Channel to be used for growth mask creation. If not provided will be automatically set based on
        # 'PROBLEM.INSTANCE_SEG.DATA_CHANNELS'. Options:
        #   - A list of channel names, e.g. ["B"], ["C"] or ["B", "C"]. The channels must be in 'PROBLEM.INSTANCE_SEG.DATA_CHANNELS'
        #   - "auto" to automatically create it based on the selected channels ('PROBLEM.INSTANCE_SEG.DATA_CHANNELS') 
        _C.PROBLEM.INSTANCE_SEG.WATERSHED.GROWTH_MASK_CHANNELS = []
        # Thresholds for the growth mask channels. If not provided will be automatically set with "auto" for each channel.
        # If set it must have same length than 'PROBLEM.INSTANCE_SEG.WATERSHED.GROWTH_MASK_CHANNELS'. Options:
        #   - A float between 0 and 1 to threshold each channel
        #   - "auto" to automatically define a value depending on the channels provided
        _C.PROBLEM.INSTANCE_SEG.WATERSHED.GROWTH_MASK_CHANNELS_THRESH  = []

        # Options for stardist-kind instance creation
        _C.PROBLEM.INSTANCE_SEG.STARDIST = CN()
        # Probability threshold to consider a pixel/voxel as a potential instance center
        _C.PROBLEM.INSTANCE_SEG.STARDIST.PROB_THRESH = 0.4
        # Non-maximum suppression IoU threshold to filter overlapping instance candidates
        _C.PROBLEM.INSTANCE_SEG.STARDIST.NMS_IOU_THRESH = 0.3

        # Size of small objects to be removed after doing watershed
        _C.PROBLEM.INSTANCE_SEG.DATA_REMOVE_SMALL_OBJ_BEFORE = 10
        # Whether to remove objects before watershed
        _C.PROBLEM.INSTANCE_SEG.DATA_REMOVE_BEFORE_MW = False
        # Sequence of string to determine the morphological filters to apply to instance seeds. They will be done in that order.
        # Possible options 'dilate' and 'erode'. E.g. ['erode','dilate'] to erode first and dilate later.
        _C.PROBLEM.INSTANCE_SEG.SEED_MORPH_SEQUENCE = []
        # Sequence of ints to determine the radius of the erosion or dilation for instance seeds
        _C.PROBLEM.INSTANCE_SEG.SEED_MORPH_RADIUS = []
        # To erode and dilate the foreground mask before using marker controlled watershed. The idea is to remove the small holes
        # that may be produced so the instances grow without them
        _C.PROBLEM.INSTANCE_SEG.ERODE_AND_DILATE_GROWTH_MASK = False
        # Radius to erode the foreground mask
        _C.PROBLEM.INSTANCE_SEG.FORE_EROSION_RADIUS = 5
        # Radius to dilate the foreground mask
        _C.PROBLEM.INSTANCE_SEG.FORE_DILATION_RADIUS = 5
        # Whether to save watershed check files
        _C.PROBLEM.INSTANCE_SEG.DATA_CHECK_MW = False
        # Whether to apply or not the watershed to create instances slice by slice in a 3D problem. This can solve instances invading
        # others if the objects in Z axis overlap too much.
        _C.PROBLEM.INSTANCE_SEG.WATERSHED_BY_2D_SLICES = False

        #### For "synapses" type of instances (only available for 3D H5/Zarr data) ####
        _C.PROBLEM.INSTANCE_SEG.SYNAPSES = CN()
        # Dilation in (z,y,x) to be made for the 'B' channel
        _C.PROBLEM.INSTANCE_SEG.SYNAPSES.POSTSITE_DILATION = [2,4,4]
        # Dilation in (z,y,x) to be made to construct the 'F' channel
        _C.PROBLEM.INSTANCE_SEG.SYNAPSES.POSTSITE_DILATION_DISTANCE_CHANNELS = [3,10,10] 
        # Whether to normalize or not the distances of channel 'F'
        _C.PROBLEM.INSTANCE_SEG.SYNAPSES.NORMALIZE_DISTANCES = False
        _C.PROBLEM.INSTANCE_SEG.SYNAPSES.POINT_CREATION_FUNCTION = "peak_local_max"
        # The minimal allowed distance separating peaks. To find the maximum number of peaks, use min_distance=1.
        _C.PROBLEM.INSTANCE_SEG.SYNAPSES.PEAK_LOCAL_MAX_MIN_DISTANCE = 1
        # Whether the threshold are going to be set as automaticaly (with Otsu thresholding) or manually.
        # Options available: 
        #   * 'auto' to decide the threshold to be applied by measuring it with Otsu
        #   * 'manual' to set a fixed threshold defined by 'PROBLEM.INSTANCE_SEG.SYNAPSES.MIN_TH_TO_BE_PEAK'
        #   * 'relative_by_patch' to use 'PROBLEM.INSTANCE_SEG.SYNAPSES.MIN_TH_TO_BE_PEAK' but relative to the maximum value in the predicted patch 
        #     data. More info in https://scikit-image.org/docs/0.25.x/api/skimage.feature.html#skimage.feature.peak_local_max and 
        #     https://scikit-image.org/docs/0.23.x/api/skimage.feature.html#skimage.feature.blob_log (see 'threshold_rel' argument description)
        #   * 'relative' to use 'PROBLEM.INSTANCE_SEG.SYNAPSES.MIN_TH_TO_BE_PEAK' but relative to the maximum value in the whole predicted data. 
        #     More info in https://scikit-image.org/docs/0.25.x/api/skimage.feature.html#skimage.feature.peak_local_max and 
        #     https://scikit-image.org/docs/0.23.x/api/skimage.feature.html#skimage.feature.blob_log (see 'threshold_rel' argument description)
        _C.PROBLEM.INSTANCE_SEG.SYNAPSES.TH_TYPE = "auto"
        # Minimun value to consider a point as a peak. Corresponds to 'threshold_abs' argument of the function
        # 'peak_local_max' of skimage.feature
        _C.PROBLEM.INSTANCE_SEG.SYNAPSES.MIN_TH_TO_BE_PEAK = 0.2
        # Corresponds to 'exclude_border' argument of 'peak_local_max' or 'blob_log' function of skimage. If True it will exclude
        # peaks from the border of the image to avoid partial detection.
        _C.PROBLEM.INSTANCE_SEG.SYNAPSES.EXCLUDE_BORDER = False
        # Corresponds to 'min_sigma' argument of 'blob_log' function. It is the minimum standard deviation for Gaussian kernel.
        # Keep this low to detect smaller blobs. The standard deviations of the Gaussian filter are given for each axis as a
        # sequence, or as a single number, in which case it is equal for all axes.
        _C.PROBLEM.INSTANCE_SEG.SYNAPSES.BLOB_LOG_MIN_SIGMA = 5
        # Corresponds to 'max_sigma' argument of 'blob_log' function. It is the maximum standard deviation for Gaussian kernel.
        # Keep this high to detect larger blobs. The standard deviations of the Gaussian filter are given for each axis as a
        # sequence, or as a single number, in which case it is equal for all axes.
        _C.PROBLEM.INSTANCE_SEG.SYNAPSES.BLOB_LOG_MAX_SIGMA = 10
        # Corresponds to 'num_sigma' argument of 'blob_log' function. The number of intermediate values of standard deviations
        # to consider between min_sigma and max_sigma.
        _C.PROBLEM.INSTANCE_SEG.SYNAPSES.BLOB_LOG_NUM_SIGMA = 2
        # For removing close points detected during synapse prediction
        _C.PROBLEM.INSTANCE_SEG.SYNAPSES.REMOVE_CLOSE_PRE_POINTS_RADIUS = 0
        _C.PROBLEM.INSTANCE_SEG.SYNAPSES.REMOVE_CLOSE_POST_POINTS_RADIUS = 0
        # If this is activated the post points will be remove not only looking the distance between them but the must lay in the same mask.
        # For creating this mask there can be a few options:
        #   * Implemented: By creating instances binarizing the predictions (using PROBLEM.INSTANCE_SEG.SYNAPSES.MIN_TH_TO_BE_PEAK) and applying 
        #     a connected-components
        #   * TODO: by looking at external neuron segmentation
        _C.PROBLEM.INSTANCE_SEG.SYNAPSES.REMOVE_CLOSE_POINTS_RADIUS_BY_MASK = False

        ### DETECTION
        _C.PROBLEM.DETECTION = CN()
        # Shape of the ellipse that will be used to dilate the central point created from the CSV file. 0 to not dilate and only create a 3x3 square.
        # The value is the radius of the ellipse in pixels. If an integer is given, the shape will be a ball with the given side length.
        # If a list is given, the shape will be a hyperball with the given side lengths. List order is (y,x) or (z,y,x) for 2D and 3D respectively.
        # For example [1, 2, 3] will result in an ellipse with a radius of 1 in the first dimension, 2 in the second and 3 in the third.
        _C.PROBLEM.DETECTION.CENTRAL_POINT_DILATION = [2]
        _C.PROBLEM.DETECTION.CHECK_POINTS_CREATED = True
        # Whether to save watershed check files
        _C.PROBLEM.DETECTION.DATA_CHECK_MW = False

        ### DENOISING
        # Based Noise2Void paper: https://arxiv.org/abs/1811.10980
        _C.PROBLEM.DENOISING = CN()
        # This variable corresponds to n2v_perc_pix from Noise2Void. It explanation is as follows: for faster training multiple
        # pixels per input patch can be manipulated. In our experiments we manipulated about 0.198% of the input pixels per
        # patch. For a patch size of 64 by 64 pixels this corresponds to about 8 pixels. This fraction can be tuned via this variable
        _C.PROBLEM.DENOISING.N2V_PERC_PIX = 0.198
        # This variable corresponds to n2v_manipulator from Noise2Void. Most pixel manipulators will compute the replacement value based
        # on a neighborhood and this variable controls how to do that
        _C.PROBLEM.DENOISING.N2V_MANIPULATOR = "uniform_withCP"
        # This variable corresponds to n2v_neighborhood_radius from Noise2Void. Size of the neighborhood to compute the replacement
        _C.PROBLEM.DENOISING.N2V_NEIGHBORHOOD_RADIUS = 5
        # To apply a structured mask as is proposed in Noise2Void to alleviate the limitation of the method of not removing effectively
        # the structured noise (section 4.4 of their paper).
        _C.PROBLEM.DENOISING.N2V_STRUCTMASK = False
        # Whether to load ground truth data in denoising
        _C.PROBLEM.DENOISING.LOAD_GT_DATA = False

        ### SUPER_RESOLUTION
        _C.PROBLEM.SUPER_RESOLUTION = CN()
        # Upscaling to be done to the input images on every dimension. Examples: (2,2) in 2D or (2,2,2) in 3D.
        _C.PROBLEM.SUPER_RESOLUTION.UPSCALING = ()

        ### SELF_SUPERVISED
        _C.PROBLEM.SELF_SUPERVISED = CN()
        # Pretext task to do. Options are as follows:
        #   - 'crappify': crappifies input image by adding Gaussian noise and downsampling and upsampling it so the resolution
        #                 gets worsen. Then, the model is trained to recover the original images.
        #   - 'masking': mask input image and the model needs to recover the original image. This option can only
        #                be done with 'mae' transformer. This strategy follows the one proposed in:
        #                Masked Autoencoders Are Scalable Vision Learners (https://arxiv.org/pdf/2111.06377.pdf)
        _C.PROBLEM.SELF_SUPERVISED.PRETEXT_TASK = "crappify"
        # Downsizing factor to reshape the image. It will be downsampled and upsampled again by this factor so the
        # quality of the image is worsens
        _C.PROBLEM.SELF_SUPERVISED.RESIZING_FACTOR = 4
        # Number between [0, 1] indicating the std of the Gaussian noise N(0,std).
        _C.PROBLEM.SELF_SUPERVISED.NOISE = 0.2

        ### IMAGE_TO_IMAGE
        _C.PROBLEM.IMAGE_TO_IMAGE = CN()
        # To use a custom data loader to load a random image from each image sample folder. The data needs to be structured
        # in an special way, that is, instead of having images in the training/val folder a folder for each sample is expected,
        # where in each of those different versions of the same data sample will be placed. Visit the following tutorial
        # for a real use case and a more detailed description:
        #   - https://biapy.readthedocs.io/en/latest/tutorials/image-to-image/lightmycells.html
        _C.PROBLEM.IMAGE_TO_IMAGE.MULTIPLE_RAW_ONE_TARGET_LOADER = False

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Dataset
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        _C.DATA = CN()

        # Save all data of a generator in the given path.
        _C.DATA.CHECK_GENERATORS = False

        # _C.PROBLEM.NDIM='2D' -> _C.DATA.PATCH_SIZE=(y,x,c) ; _C.PROBLEM.NDIM='3D' -> _C.DATA.PATCH_SIZE=(z,y,x,c)
        _C.DATA.PATCH_SIZE = (256, 256, 1)

        # Extract random patches during data augmentation (DA)
        _C.DATA.EXTRACT_RANDOM_PATCH = False
        # Create a probability map so the patches extracted will have a high probability of having an object in the middle
        # of it. Useful to avoid extracting patches which no foreground class information. Use it only when
        # 'PROBLEM.TYPE' is 'SEMANTIC_SEG'
        _C.DATA.PROBABILITY_MAP = False  # Used when _C.DATA.EXTRACT_RANDOM_PATCH=True
        _C.DATA.W_FOREGROUND = 0.94  # Used when _C.DATA.PROBABILITY_MAP=True
        _C.DATA.W_BACKGROUND = 0.06  # Used when _C.DATA.PROBABILITY_MAP=True

        # Whether to reshape the dimensions that does not satisfy the patch shape selected by padding it with reflect.
        _C.DATA.REFLECT_TO_COMPLETE_SHAPE = False
        # If 'DATA.PATCH_SIZE' selected has 3 channels, e.g. RGB images are expected, so will force grayscale images to be
        # converted into RGB (e.g. in ImageNet some of the images are grayscale)
        _C.DATA.FORCE_RGB = False
        # If filtering is done, with any of DATA.*.FILTER_SAMPLES.* variables, this will decide how this filtering will be done:
        #   * True: apply filter image by image. 
        #   * False: apply filtering sample by sample. Each sample represents a patch within an image.
        _C.DATA.FILTER_BY_IMAGE = False
        # Determines whether to save filtered images. If 'DATA.FILTER_BY_IMAGE' is enabled, two subfolders will be created: one for 
        # filtered images and another for non-filtered images. Otherwise, no subfolders will be created, and the images will 
        # display filtered patches as black (all zero values) while retaining original patch values in non-filtered areas. 
        _C.DATA.SAVE_FILTERED_IMAGES = False
        # Number of filtered images to save. Only work when 'DATA.SAVE_FILTERED_IMAGES' is True
        _C.DATA.SAVE_FILTERED_IMAGES_NUM = 3 

        _C.DATA.NORMALIZATION = CN()
        # Whether to apply or not a percentile clipping before normalizing the data
        _C.DATA.NORMALIZATION.PERC_CLIP = CN()
        _C.DATA.NORMALIZATION.PERC_CLIP.ENABLE = False
        # Lower and upper bound for percentile clip. Must be set when DATA.NORMALIZATION.PERC_CLIP.ENABLE = 'True'
        _C.DATA.NORMALIZATION.PERC_CLIP.LOWER_PERC = -1.0
        _C.DATA.NORMALIZATION.PERC_CLIP.UPPER_PERC = -1.0
        # Lower and upper values to clip. If these are provided the percentiles are not calculated based on 
        # 'DATA.NORMALIZATION.PERC_CLIP.LOWER_PERC' and 'DATA.NORMALIZATION.PERC_CLIP.UPPER_PERC' 
        _C.DATA.NORMALIZATION.PERC_CLIP.LOWER_VALUE = -1.0
        _C.DATA.NORMALIZATION.PERC_CLIP.UPPER_VALUE = -1.0
        # Normalization type to use. Possible options:
        #   'div' to divide values from 0/255 (or 0/65535 if uint16) in [0,1] range
        #   'scale_range' same as 'div' but scaling the range to [0-max] and then dividing by the maximum value of the data
        #    and not by 255 or 65535
        #   'zero_mean_unit_variance' to substract the mean and divide by std. 
        _C.DATA.NORMALIZATION.TYPE = "zero_mean_unit_variance"
        # Whether to based the normalization on values extracted from the complete image or from each patch. When working with 
        # large images such as Zarr or H5 it is done by patch automatically. The values are also applied for the percentage
        # clipping. Options: ['image', 'patch']
        _C.DATA.NORMALIZATION.MEASURE_BY = "image"
        # Custom normalization variables: mean and std (they are calculated if not provided)
        _C.DATA.NORMALIZATION.ZERO_MEAN_UNIT_VAR = CN()
        _C.DATA.NORMALIZATION.ZERO_MEAN_UNIT_VAR.MEAN_VAL = -1.0
        _C.DATA.NORMALIZATION.ZERO_MEAN_UNIT_VAR.STD_VAL = -1.0

        # Train
        _C.DATA.TRAIN = CN()
        # Whether to check if the data mask contains correct values, e.g. same classes as defined
        _C.DATA.TRAIN.CHECK_DATA = True
        _C.DATA.TRAIN.IN_MEMORY = True
        _C.DATA.TRAIN.PATH = os.path.join("user_data", "train", "x")
        _C.DATA.TRAIN.GT_PATH = os.path.join("user_data", "train", "y")
        # Whether your input Zarr contains the raw images and labels together or not. Use 'DATA.TRAIN.INPUT_ZARR_MULTIPLE_DATA_RAW_PATH'
        # and 'DATA.TRAIN.INPUT_ZARR_MULTIPLE_DATA_GT_PATH' to determine the tag to find within the Zarr
        _C.DATA.TRAIN.INPUT_ZARR_MULTIPLE_DATA = False
        # Paths to the raw and gt within the Zarr file. Only used when 'DATA.TRAIN.INPUT_ZARR_MULTIPLE_DATA' is True.
        # E.g. 'volumes.raw' for raw and 'volumes.labels.neuron_ids' for GT path.
        _C.DATA.TRAIN.INPUT_ZARR_MULTIPLE_DATA_RAW_PATH = ""
        _C.DATA.TRAIN.INPUT_ZARR_MULTIPLE_DATA_GT_PATH = ""
        # For synapse detection. The information must be stored as CREMI dataset (https://cremi.org/data/)
        # Path within the file where the ``ids`` are stored. Reference in CREMI: ``annotations/ids``
        _C.DATA.TRAIN.INPUT_ZARR_MULTIPLE_DATA_ID_PATH = "annotations.ids"
        # Path within the file where the ``types`` are stored (not used). Reference in CREMI: ``annotations/types``
        # _C.DATA.TRAIN.INPUT_ZARR_MULTIPLE_DATA_TYPES_PATH = "annotations.types"
        # Path within the file where the ``partners`` are stored. Reference in CREMI: ``annotations/partners``
        _C.DATA.TRAIN.INPUT_ZARR_MULTIPLE_DATA_PARTNERS_PATH = "annotations.presynaptic_site.partners"
        # Path within the file where the ``locations`` are stored. Reference in CREMI: ``annotations/locations``
        _C.DATA.TRAIN.INPUT_ZARR_MULTIPLE_DATA_LOCATIONS_PATH = "annotations.locations"
        # Path within the file where the ``resolution`` is stored. Reference in CREMI: ``["volumes/raw"].attrs["offset"]``
        _C.DATA.TRAIN.INPUT_ZARR_MULTIPLE_DATA_RESOLUTION_PATH = 'volumes.raw'
        # File to load/save data prepared with the appropiate channels in a instance segmentation problem.
        # E.g. _C.PROBLEM.TYPE ='INSTANCE_SEG' and _C.PROBLEM.INSTANCE_SEG.DATA_CHANNELS != 'B'
        _C.DATA.TRAIN.INSTANCE_CHANNELS_MASK_DIR = os.path.join(
            "user_data", "train", "y_" + "".join(_C.PROBLEM.INSTANCE_SEG.DATA_CHANNELS)
        )
        # Path to load/save detection masks prepared.
        _C.DATA.TRAIN.DETECTION_MASK_DIR = os.path.join("user_data", "train", "y_detection_masks")
        # Path to load/save SSL target prepared.
        _C.DATA.TRAIN.SSL_SOURCE_DIR = os.path.join("user_data", "train", "x_ssl_source")
        # Extra train data generation: number of times to duplicate the train data. Useful when
        # _C.DATA.EXTRACT_RANDOM_PATCH=True is made, as more original train data can be cover on each epoch
        _C.DATA.TRAIN.REPLICATE = 0
        # Percentage of overlap in (y,x)/(z,y,x) when cropping validation. Set to 0 to calculate  the minimun overlap.
        # The values must be floats between range [0, 1). It needs to be a 2D tuple when using _C.PROBLEM.NDIM='2D' and
        # 3D tuple when using _C.PROBLEM.NDIM='3D'
        _C.DATA.TRAIN.OVERLAP = (0, 0)
        # Padding to be done in (y,x)/(z,y,x) when reconstructing train data. Useful to avoid patch 'border effect'.
        _C.DATA.TRAIN.PADDING = (0, 0)
        # Train data resolution. It is not completely necessary but when configured it is taken into account when
        # performing some augmentations, e.g. cutout. If defined it need to be (y,x)/(z,y,x) and needs to be to be a 2D
        # tuple when using _C.PROBLEM.NDIM='2D' and 3D tuple when using _C.PROBLEM.NDIM='3D'
        _C.DATA.TRAIN.RESOLUTION = (-1,)
        # Order of the axes of the image when using Zarr/H5 images in train data.
        _C.DATA.TRAIN.INPUT_IMG_AXES_ORDER = "TZCYX"
        # Order of the axes of the mask when using Zarr/H5 images in train data.
        _C.DATA.TRAIN.INPUT_MASK_AXES_ORDER = "TZCYX"

        # DATA.TRAIN.FILTER_SAMPLES allows removing training images by the conditions based on their properties. When using Zarr each patch within the Zarr will be
        # processed and will not depend on 'DATA.FILTER_BY_IMAGE' variable.
        # Its three variables (PROPS, VALUES and SIGNS) define a set of conditions to remove the images from the training set. If an image satisfies any of the 
        # conditions, the image won't be used for training.
        #
        # In PROPS, we define the property to look at to establish the condition. The available properties are: ['foreground', 'mean', 'min', 'max', 
        # 'target_mean', 'target_min', 'target_max', 'diff', 'diff_by_min_max_ratio', 'diff_by_target_min_max_ratio'].
        #
        #   * 'foreground' is defined as the percentage of pixels/voxels corresponding to the foreground mask. This option is only valid for
        #     SEMANTIC_SEG, INSTANCE_SEG and DETECTION.
        #   * 'mean' is defined as the mean intensity value of the raw image inputs.
        #   * 'min' is defined as the min intensity value of the raw image inputs.
        #   * 'max' is defined as the max intensity value of the raw image inputs.
        #   * 'diff' is defined as the difference between ground truth and raw images. Available for all workflows but SELF_SUPERVISED and DENOISING. 
        #   * 'diff_by_min_max_ratio' is defined as the difference between ground truth and raw images multiplied by the ratio between raw image max and min. Available for all workflows but SELF_SUPERVISED and DENOISING. 
        #   * 'target_mean' is defined as the mean intensity value of the raw image targets. Available for all workflows but SELF_SUPERVISED and DENOISING.
        #   * 'target_min' is defined as the min intensity value of the raw image targets. Available for all workflows but SELF_SUPERVISED and DENOISING. 
        #   * 'target_max' is defined as the max intensity value of the raw image targets. Available for all workflows but SELF_SUPERVISED and DENOISING.  
        #   * 'diff_by_target_min_max_ratio' is defined as the difference between ground truth and raw images multiplied by the ratio between ground truth image max and min. Available for all workflows but SELF_SUPERVISED and DENOISING. 
        #
        # With VALUES and SIGNS, we define the specific values and the comparison operators of each property, respectively.
        # The available operators are: ['gt', 'ge', 'lt', 'le'], that corresponds to "greather than" (or ">"), "greather equal" (or ">="), "less than" (or "<"),
        # and "less equal"  (or "<=").
        #   
        # Here you have a full example of this filtering:
        # If you want to remove those samples that have intensity values lower than 0.00001 and a mean average greater than 100 you should
        # declare the above three variables as follows (notice you need to know the image data type in advance):
        #   _C.DATA.TRAIN.FILTER_SAMPLES.PROPS = [['foreground','mean']]
        #   _C.DATA.TRAIN.FILTER_SAMPLES.VALUES = [[0.00001, 100]]
        #   _C.DATA.TRAIN.FILTER_SAMPLES.SIGNS = [['lt', 'gt']]
        # You can also concatenate more restrictions and they will be applied in order. For instance, if you want to filter those
        # samples with a maximum intensity value greater than 1000, and do that before the condition described above, you can define the
        # variables this way:
        #   _C.DATA.TRAIN.FILTER_SAMPLES.PROPS = [['max'], ['foreground','mean']]
        #   _C.DATA.TRAIN.FILTER_SAMPLES.VALUES = [[1000], [0.00001, 100]]
        #   _C.DATA.TRAIN.FILTER_SAMPLES.SIGNS = [['gt'], ['lt', 'gt']]
        # This way, the images will be removed by 'max' and then by 'foreground' and 'mean'
        _C.DATA.TRAIN.FILTER_SAMPLES = CN()
        # Whether to enable or not the filtering by properties
        _C.DATA.TRAIN.FILTER_SAMPLES.ENABLE = False
        # List of lists of properties to apply a filter. Available properties are: ['foreground', 'mean', 'min', 'max', 
        # 'target_mean', 'target_min', 'target_max', 'diff', 'diff_by_min_max_ratio', 'diff_by_target_min_max_ratio']
        _C.DATA.TRAIN.FILTER_SAMPLES.PROPS = []
        # List of ints/float that represent the values of the properties listed in 'DATA.TRAIN.FILTER_SAMPLES.PROPS'
        # that the images need to satisfy to not be dropped.
        _C.DATA.TRAIN.FILTER_SAMPLES.VALUES = []
        # List of list of signs to do the comparison. Options: ['gt', 'ge', 'lt', 'le'] that corresponds to "greather than", e.g. ">",
        # "greather equal", e.g. ">=", "less than", e.g. "<", and "less equal" e.g. "<=" comparisons.
        _C.DATA.TRAIN.FILTER_SAMPLES.SIGNS = []
        # Whether to normalize the samples before comparison
        _C.DATA.TRAIN.FILTER_SAMPLES.NORM_BEFORE = False

        # PREPROCESSING
        # Same preprocessing will be applied to all selected datasets
        _C.DATA.PREPROCESS = CN()
        # Apply preprocessing to training dataset
        _C.DATA.PREPROCESS.TRAIN = False
        # Apply preprocessing to validation dataset
        _C.DATA.PREPROCESS.VAL = False
        # Apply preprocessing to testing dataset
        _C.DATA.PREPROCESS.TEST = False

        # Resize datasets
        _C.DATA.PREPROCESS.RESIZE = CN()
        _C.DATA.PREPROCESS.RESIZE.ENABLE = False
        # Desired resize size. when using 3D data, size must be also in 3D (ex. (512,512,512))
        _C.DATA.PREPROCESS.RESIZE.OUTPUT_SHAPE = (512, 512)
        # interpolation order: {0: Nearest-neighbor, 1: Bi-linear (default), 2: Bi-quadratic, 3: Bi-cubic, 4: Bi-quartic, 5: Bi-quintic}
        _C.DATA.PREPROCESS.RESIZE.ORDER = 1
        # Points outside the boundaries of the input are filled according to the given mode: {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}
        _C.DATA.PREPROCESS.RESIZE.MODE = "reflect"
        # Used in conjunction with mode ‘constant’, the value outside the image boundaries.
        _C.DATA.PREPROCESS.RESIZE.CVAL = 0.0
        # Whether to clip the output to the range of values of the input image.
        _C.DATA.PREPROCESS.RESIZE.CLIP = True
        # Whether to keep the original range of values.
        _C.DATA.PREPROCESS.RESIZE.PRESERVE_RANGE = True
        # Whether to apply a Gaussian filter to smooth the image prior to downsampling.
        _C.DATA.PREPROCESS.RESIZE.ANTI_ALIASING = False

        # Zoom datasets.
        _C.DATA.PREPROCESS.ZOOM = CN()
        _C.DATA.PREPROCESS.ZOOM.ENABLE = False
        # WARNING: Only implemented for _C.TEST.BY_CHUNKS = True. It will change the zoom of each patch individually.
        # This is useful when the input image has a different resolution than the one used in the training. The value
        # is the zoom factor to be applied to each patch using scipy.ndimage.zoom.
        # "E.g. [1,2,1,3,3] that needs to match _C.DATA.TEST.INPUT_IMG_AXES_ORDER axes"
        _C.DATA.PREPROCESS.ZOOM.ZOOM_FACTOR = [1, 1, 1, 1, 1]

        # Gaussian blur
        _C.DATA.PREPROCESS.GAUSSIAN_BLUR = CN()
        _C.DATA.PREPROCESS.GAUSSIAN_BLUR.ENABLE = False
        # Standard deviation for Gaussian kernel.
        _C.DATA.PREPROCESS.GAUSSIAN_BLUR.SIGMA = 1
        # The mode parameter determines how the array borders are handled: {‘reflect’, ‘constant’, ‘nearest’, ‘mirror’, ‘wrap’} ‘constant’ value = 0
        _C.DATA.PREPROCESS.GAUSSIAN_BLUR.MODE = "nearest"
        # If None, the image is assumed to be a grayscale (single channel) image.
        # Otherwise, this parameter indicates which axis of the array corresponds to channels.
        _C.DATA.PREPROCESS.GAUSSIAN_BLUR.CHANNEL_AXIS = None

        # Median blur
        _C.DATA.PREPROCESS.MEDIAN_BLUR = CN()
        _C.DATA.PREPROCESS.MEDIAN_BLUR.ENABLE = False
        # Desired kernel size (including channels). When using 3D data, size must be also in 3D (ex. (3,7,7,1) for (z,y,x,c))
        _C.DATA.PREPROCESS.MEDIAN_BLUR.KERNEL_SIZE = (3,3,1)

        # Histogram matching. More info at: https://en.wikipedia.org/wiki/Histogram_matching
        _C.DATA.PREPROCESS.MATCH_HISTOGRAM = CN()
        _C.DATA.PREPROCESS.MATCH_HISTOGRAM.ENABLE = False
        # the path of the reference images, from which the reference histogram will be extracted
        _C.DATA.PREPROCESS.MATCH_HISTOGRAM.REFERENCE_PATH = os.path.join("user_data", "test", "x")

        # Contrast Limited Adaptive Histogram Equalization. More info at: https://en.wikipedia.org/wiki/Adaptive_histogram_equalization#Contrast_Limited_AHE
        _C.DATA.PREPROCESS.CLAHE = CN()
        _C.DATA.PREPROCESS.CLAHE.ENABLE = False
        # Defines the shape of contextual regions used in the algorithm.
        # By default, kernel_size is 1/8 of image height by 1/8 of its width.
        _C.DATA.PREPROCESS.CLAHE.KERNEL_SIZE = None
        # Clipping limit, normalized between 0 and 1 (higher values give more contrast).
        _C.DATA.PREPROCESS.CLAHE.CLIP_LIMIT = 0.01

        # Canny or edge detection (only 2D - grayscale or RGB)
        _C.DATA.PREPROCESS.CANNY = CN()
        _C.DATA.PREPROCESS.CANNY.ENABLE = False
        # Lower bound for hysteresis thresholding (linking edges). If None, low_threshold is set to 10% of dtype’s max.
        _C.DATA.PREPROCESS.CANNY.LOW_THRESHOLD = None
        # Upper bound for hysteresis thresholding (linking edges). If None, high_threshold is set to 20% of dtype’s max.
        _C.DATA.PREPROCESS.CANNY.HIGH_THRESHOLD = None

        # Test
        _C.DATA.TEST = CN()
        # Whether to check if the data mask contains correct values, e.g. same classes as defined
        _C.DATA.TEST.CHECK_DATA = True
        _C.DATA.TEST.IN_MEMORY = False
        # Whether to load ground truth (GT)
        _C.DATA.TEST.LOAD_GT = False
        # Whether to use validation data as test instead of trying to load test from _C.DATA.TEST.PATH and
        # _C.DATA.TEST.GT_PATH. _C.DATA.VAL.CROSS_VAL needs to be True.
        _C.DATA.TEST.USE_VAL_AS_TEST = False
        # Path to load the test data from. Not used when _C.DATA.TEST.USE_VAL_AS_TEST == True
        _C.DATA.TEST.PATH = os.path.join("user_data", "test", "x")
        # Path to load the test data masks from. Not used when _C.DATA.TEST.USE_VAL_AS_TEST == True
        _C.DATA.TEST.GT_PATH = os.path.join("user_data", "test", "y")
        # File to load/save data prepared with the appropiate channels in a instance segmentation problem.
        # E.g. _C.PROBLEM.TYPE ='INSTANCE_SEG' and _C.PROBLEM.INSTANCE_SEG.DATA_CHANNELS != 'B'
        _C.DATA.TEST.INSTANCE_CHANNELS_MASK_DIR = os.path.join(
            "user_data", "test", "y_" + "".join(_C.PROBLEM.INSTANCE_SEG.DATA_CHANNELS)
        )
        # Path to load/save detection masks prepared.
        _C.DATA.TEST.DETECTION_MASK_DIR = os.path.join("user_data", "test", "y_detection_masks")
        # Path to load/save SSL target prepared.
        _C.DATA.TEST.SSL_SOURCE_DIR = os.path.join("user_data", "test", "x_ssl_source")
        # Percentage of overlap in (y,x)/(z,y,x) when cropping validation. Set to 0 to calculate  the minimun overlap.
        # The values must be floats between range [0, 1). It needs to be a 2D tuple when using _C.PROBLEM.NDIM='2D' and
        # 3D tuple when using _C.PROBLEM.NDIM='3D'
        _C.DATA.TEST.OVERLAP = (0, 0)
        # Padding to be done in (y,x)/(z,y,xz) when reconstructing test data. Useful to avoid patch 'border effect'
        _C.DATA.TEST.PADDING = (0, 0)
        # Whether to use median values to fill padded pixels or zeros
        _C.DATA.TEST.MEDIAN_PADDING = False
        # Directory where binary masks to apply to resulting images should be. Used when _C.TEST.POST_PROCESSING.APPLY_MASK  == True
        _C.DATA.TEST.BINARY_MASKS = os.path.join("user_data", "test", "bin_mask")
        # Test data resolution. Need to be provided in (z,y,x) order. Only applies when _C.PROBLEM.TYPE = 'DETECTION' now.
        _C.DATA.TEST.RESOLUTION = (-1,)
        # Whether to apply argmax to the predicted images
        _C.DATA.TEST.ARGMAX_TO_OUTPUT = True
        # Order of the axes of the image when using Zarr/H5 images in test data.
        _C.DATA.TEST.INPUT_IMG_AXES_ORDER = "TZCYX"
        # Order of the axes of the mask when using Zarr/H5 images in test data.
        _C.DATA.TEST.INPUT_MASK_AXES_ORDER = "TZCYX"
        # Whether your input Zarr contains the raw images and labels together or not. Use 'DATA.TEST.INPUT_ZARR_MULTIPLE_DATA_RAW_PATH'
        # and 'DATA.TEST.INPUT_ZARR_MULTIPLE_DATA_GT_PATH' to determine the tag to find within the Zarr
        _C.DATA.TEST.INPUT_ZARR_MULTIPLE_DATA = False
        # Paths to the raw and gt within the Zarr file. Only used when 'DATA.TEST.INPUT_ZARR_MULTIPLE_DATA' is True.
        # E.g. 'volumes.raw' for raw and 'volumes.labels.neuron_ids' for GT path.
        _C.DATA.TEST.INPUT_ZARR_MULTIPLE_DATA_RAW_PATH = ""
        _C.DATA.TEST.INPUT_ZARR_MULTIPLE_DATA_GT_PATH = ""
        # For synapse detection. The information must be stored as CREMI dataset (https://cremi.org/data/)
        # Path within the file where the ``ids`` are stored. Reference in CREMI: ``annotations/ids``
        _C.DATA.TEST.INPUT_ZARR_MULTIPLE_DATA_ID_PATH = "annotations.ids"
        # Path within the file where the ``types`` are stored (not used). Reference in CREMI: ``annotations/types``
        # _C.DATA.TEST.INPUT_ZARR_MULTIPLE_DATA_TYPES_PATH = "annotations.types"
        # Path within the file where the ``partners`` are stored. Reference in CREMI: ``annotations/partners``
        _C.DATA.TEST.INPUT_ZARR_MULTIPLE_DATA_PARTNERS_PATH = "annotations.presynaptic_site.partners"
        # Path within the file where the ``locations`` are stored. Reference in CREMI: ``annotations/locations``
        _C.DATA.TEST.INPUT_ZARR_MULTIPLE_DATA_LOCATIONS_PATH = "annotations.locations"
        # Path within the file where the ``resolution`` is stored. Reference in CREMI: ``["volumes/raw"].attrs["offset"]``
        _C.DATA.TEST.INPUT_ZARR_MULTIPLE_DATA_RESOLUTION_PATH = 'volumes.raw'
        # Remove test images by the conditions based on their properties. When using Zarr each patch within the Zarr will be processed and will
        # not depend on 'DATA.FILTER_BY_IMAGE' variable
        # The three variables, DATA.TEST.FILTER_SAMPLES.PROPS, DATA.TEST.FILTER_SAMPLES.VALUES and DATA.TEST.FILTER_SAMPLES.SIGNS will compose a 
        # list of conditions to remove the images. They are list of list of conditions. For instance, the conditions can be like this: [['A'], ['B','C']]. 
        # Then, if the image satisfies the first list of conditions, only 'A' in this first case (from ['A'] list), or satisfy 'B' and 'C' (from ['B','C'] list) 
        # it will be removed from the image. In each sublist all the conditions must be satisfied. Available properties are: ['foreground', 'mean', 'min', 'max', 
        # 'target_mean', 'target_min', 'target_max', 'diff', 'diff_by_min_max_ratio', 'diff_by_target_min_max_ratio'].
        #
        # Each property descrition:
        #   * 'foreground' is defined as the percentage of pixels/voxels corresponding to the foreground mask. This option is only valid for
        #     SEMANTIC_SEG, INSTANCE_SEG and DETECTION.
        #   * 'mean' is defined as the mean intensity value of the raw image inputs.
        #   * 'min' is defined as the min intensity value of the raw image inputs.
        #   * 'max' is defined as the max intensity value of the raw image inputs.
        #   * 'diff' is defined as the difference between ground truth and raw images. Available for all workflows but SELF_SUPERVISED and DENOISING. 
        #   * 'diff_by_min_max_ratio' is defined as the difference between ground truth and raw images multiplied by the ratio between raw image max and min. Available for all workflows but SELF_SUPERVISED and DENOISING. 
        #   * 'target_mean' is defined as the mean intensity value of the raw image targets. Available for all workflows but SELF_SUPERVISED and DENOISING.
        #   * 'target_min' is defined as the min intensity value of the raw image targets. Available for all workflows but SELF_SUPERVISED and DENOISING. 
        #   * 'target_max' is defined as the max intensity value of the raw image targets. Available for all workflows but SELF_SUPERVISED and DENOISING.  
        #   * 'diff_by_target_min_max_ratio' is defined as the difference between ground truth and raw images multiplied by the ratio between ground truth image max and min. Available for all workflows but SELF_SUPERVISED and DENOISING. 
        #
        #
        # A full example of this filtering:
        # If you want to remove those samples that have less than 0.00001 and a mean average more than 100 (you need to know image data type) you should
        # declare the above three variables as follows:
        #   _C.DATA.TEST.FILTER_SAMPLES.PROPS = [['foreground','mean']]
        #   _C.DATA.TEST.FILTER_SAMPLES.VALUES = [[0.00001, 100]]
        #   _C.DATA.TEST.FILTER_SAMPLES.SIGNS = [['lt', 'gt']]
        # You can also concatenate more restrictions and they will be applied in order. For instance, if you want to filter those
        # samples with a max value more than 1000, and do that before the condition described above, you can define the
        # variables this way:
        #   _C.DATA.TEST.FILTER_SAMPLES.PROPS = [['max'], ['foreground','mean']]
        #   _C.DATA.TEST.FILTER_SAMPLES.VALUES = [[1000], [0.00001, 100]]
        #   _C.DATA.TEST.FILTER_SAMPLES.SIGNS = [['gt'], ['lt', 'gt']]
        # This way, the images will be removed by 'max' and then by 'foreground' and 'mean'
        _C.DATA.TEST.FILTER_SAMPLES = CN()
        # Whether to enable or not the filtering by properties
        _C.DATA.TEST.FILTER_SAMPLES.ENABLE = False
        # List of lists of properties to apply a filter. Available properties are: ['foreground', 'mean', 'min', 'max', 
        # 'target_mean', 'target_min', 'target_max', 'diff', 'diff_by_min_max_ratio', 'diff_by_target_min_max_ratio']
        _C.DATA.TEST.FILTER_SAMPLES.PROPS = []
        # List of ints/float that represent the values of the properties listed in 'DATA.TEST.FILTER_SAMPLES.PROPS'
        # that the images need to satisfy to not be dropped.
        _C.DATA.TEST.FILTER_SAMPLES.VALUES = []
        # List of list of signs to do the comparison. Options: ['gt', 'ge', 'lt', 'le'] that corresponds to "greather than", e.g. ">",
        # "greather equal", e.g. ">=", "less than", e.g. "<", and "less equal" e.g. "<=" comparisons.
        _C.DATA.TEST.FILTER_SAMPLES.SIGNS = []
        # Whether to normalize the samples before comparison
        _C.DATA.TEST.FILTER_SAMPLES.NORM_BEFORE = False

        # Validation
        _C.DATA.VAL = CN()
        # Enabling distributed evaluation (recommended during training)
        _C.DATA.VAL.DIST_EVAL = True
        # Whether to create validation data from training set or read it from a directory
        _C.DATA.VAL.FROM_TRAIN = True
        # Use a cross validation strategy instead of just split the train data in two
        _C.DATA.VAL.CROSS_VAL = False
        # Number of folds. Used when _C.DATA.VAL.CROSS_VAL == True
        _C.DATA.VAL.CROSS_VAL_NFOLD = 5
        # Number of the fold to choose as validation. Used when _C.DATA.VAL.CROSS_VAL == True
        _C.DATA.VAL.CROSS_VAL_FOLD = 1
        # Percentage of the training data used as validation. Used when _C.DATA.VAL.FROM_TRAIN = True and _C.DATA.VAL.CROSS_VAL == False
        _C.DATA.VAL.SPLIT_TRAIN = 0.1
        # Create the validation data with random images of the training data. Used when _C.DATA.VAL.FROM_TRAIN = True
        _C.DATA.VAL.RANDOM = True
        # Used when _C.DATA.VAL.FROM_TRAIN = False, as DATA.VAL.FROM_TRAIN = True always implies DATA.VAL.IN_MEMORY = True
        _C.DATA.VAL.IN_MEMORY = True
        # Path to the validation data. Used when _C.DATA.VAL.FROM_TRAIN = False
        _C.DATA.VAL.PATH = os.path.join("user_data", "val", "x")
        # Path to the validation data mask. Used when _C.DATA.VAL.FROM_TRAIN = False
        _C.DATA.VAL.GT_PATH = os.path.join("user_data", "val", "y")
        # Whether your input Zarr contains the raw images and labels together or not. Use 'DATA.VAL.INPUT_ZARR_MULTIPLE_DATA_RAW_PATH'
        # and 'DATA.VAL.INPUT_ZARR_MULTIPLE_DATA_GT_PATH' to determine the tag to find within the Zarr
        _C.DATA.VAL.INPUT_ZARR_MULTIPLE_DATA = False
        # Paths to the raw and gt within the Zarr file. Only used when 'DATA.VAL.INPUT_ZARR_MULTIPLE_DATA' is True.
        # E.g. 'volumes.raw' for raw and 'volumes.labels.neuron_ids' for GT path.
        _C.DATA.VAL.INPUT_ZARR_MULTIPLE_DATA_RAW_PATH = ""
        _C.DATA.VAL.INPUT_ZARR_MULTIPLE_DATA_GT_PATH = ""
        # For synapse detection. The information must be stored as CREMI dataset (https://cremi.org/data/)
        # Path within the file where the ``ids`` are stored. Reference in CREMI: ``annotations/ids``
        _C.DATA.VAL.INPUT_ZARR_MULTIPLE_DATA_ID_PATH = "annotations.ids"
        # Path within the file where the ``types`` are stored (not used). Reference in CREMI: ``annotations/types``
        # _C.DATA.VAL.INPUT_ZARR_MULTIPLE_DATA_TYPES_PATH = "annotations.types"
        # Path within the file where the ``partners`` are stored. Reference in CREMI: ``annotations/partners``
        _C.DATA.VAL.INPUT_ZARR_MULTIPLE_DATA_PARTNERS_PATH = "annotations.presynaptic_site.partners"
        # Path within the file where the ``locations`` are stored. Reference in CREMI: ``annotations/locations``
        _C.DATA.VAL.INPUT_ZARR_MULTIPLE_DATA_LOCATIONS_PATH = "annotations.locations"
        # Path within the file where the ``resolution`` is stored. Reference in CREMI: ``["volumes/raw"].attrs["offset"]``
        _C.DATA.VAL.INPUT_ZARR_MULTIPLE_DATA_RESOLUTION_PATH = 'volumes.raw'
        # File to load/save data prepared with the appropiate channels in a instance segmentation problem.
        # E.g. _C.PROBLEM.TYPE ='INSTANCE_SEG' and _C.PROBLEM.INSTANCE_SEG.DATA_CHANNELS != 'B'
        _C.DATA.VAL.INSTANCE_CHANNELS_MASK_DIR = os.path.join(
            "user_data", "val", "y_" + "".join(_C.PROBLEM.INSTANCE_SEG.DATA_CHANNELS)
        )
        # Path to load/save detection masks prepared.
        _C.DATA.VAL.DETECTION_MASK_DIR = os.path.join("user_data", "val", "y_detection_masks")
        # Path to load/save SSL target prepared.
        _C.DATA.VAL.SSL_SOURCE_DIR = os.path.join("user_data", "val", "x_ssl_source")
        # Percentage of overlap in (y,x)/(z,y,x) when cropping validation. Set to 0 to calculate  the minimun overlap.
        # The values must be floats between range [0, 1). It needs to be a 2D tuple when using _C.PROBLEM.NDIM='2D' and
        # 3D tuple when using _C.PROBLEM.NDIM='3D'. This is only used when the validation is loaded from disk, and thus,
        # not extracted from training.
        _C.DATA.VAL.OVERLAP = (0, 0)
        # Padding to be done in (y,x)/(z,y,x) when cropping validation data. Useful to avoid patch 'border effect'. This
        # is only used when the validation is loaded from disk, and thus, not extracted from training.
        _C.DATA.VAL.PADDING = (0, 0)
        # Not used yet.
        _C.DATA.VAL.RESOLUTION = (-1,)
        # Order of the axes of the image when using Zarr/H5 images in validation data.
        _C.DATA.VAL.INPUT_IMG_AXES_ORDER = "TZCYX"
        # Order of the axes of the mask when using Zarr/H5 images in validation data.
        _C.DATA.VAL.INPUT_MASK_AXES_ORDER = "TZCYX"
        # Remove validation images by the conditions based on their properties. When using Zarr each patch within the Zarr will be processed and will
        # not depend on 'DATA.FILTER_BY_IMAGE' variable 
        # The three variables, DATA.VAL.FILTER_SAMPLES.PROPS, DATA.VAL.FILTER_SAMPLES.VALUES and DATA.VAL.FILTER_SAMPLES.SIGNS will compose a list of 
        # conditions to remove the images. They are list of list of conditions. For instance, the conditions can be like this: [['A'], ['B','C']]. Then, 
        # if the image satisfies the first list of conditions, only 'A' in this first case (from ['A'] list), or satisfy 'B' and 'C' (from ['B','C'] list) 
        # it will be removed from the image. In each sublist all the conditions must be satisfied. Available properties are: ['foreground', 'mean', 'min', 'max', 
        # 'target_mean', 'target_min', 'target_max', 'diff', 'diff_by_min_max_ratio', 'diff_by_target_min_max_ratio'].
        #
        # Each property descrition:
        #   * 'foreground' is defined as the percentage of pixels/voxels corresponding to the foreground mask. This option is only valid for
        #     SEMANTIC_SEG, INSTANCE_SEG and DETECTION.
        #   * 'mean' is defined as the mean intensity value of the raw image inputs.
        #   * 'min' is defined as the min intensity value of the raw image inputs.
        #   * 'max' is defined as the max intensity value of the raw image inputs.
        #   * 'diff' is defined as the difference between ground truth and raw images. Available for all workflows but SELF_SUPERVISED and DENOISING. 
        #   * 'diff_by_min_max_ratio' is defined as the difference between ground truth and raw images multiplied by the ratio between raw image max and min. Available for all workflows but SELF_SUPERVISED and DENOISING. 
        #   * 'target_mean' is defined as the mean intensity value of the raw image targets. Available for all workflows but SELF_SUPERVISED and DENOISING.
        #   * 'target_min' is defined as the min intensity value of the raw image targets. Available for all workflows but SELF_SUPERVISED and DENOISING. 
        #   * 'target_max' is defined as the max intensity value of the raw image targets. Available for all workflows but SELF_SUPERVISED and DENOISING.  
        #   * 'diff_by_target_min_max_ratio' is defined as the difference between ground truth and raw images multiplied by the ratio between ground truth image max and min. Available for all workflows but SELF_SUPERVISED and DENOISING. 
        #
        # A full example of this filtering:
        # If you want to remove those samples that have less than 0.00001 and a mean average more than 100 (you need to know image data type) you should
        # declare the above three variables as follows:
        #   _C.DATA.VAL.FILTER_SAMPLES.PROPS = [['foreground','mean']]
        #   _C.DATA.VAL.FILTER_SAMPLES.VALUES = [[0.00001, 100]]
        #   _C.DATA.VAL.FILTER_SAMPLES.SIGNS = [['lt', 'gt']]
        # You can also concatenate more restrictions and they will be applied in order. For instance, if you want to filter those
        # samples with a max value more than 1000, and do that before the condition described above, you can define the
        # variables this way:
        #   _C.DATA.VAL.FILTER_SAMPLES.PROPS = [['max'], ['foreground','mean']]
        #   _C.DATA.VAL.FILTER_SAMPLES.VALUES = [[1000], [0.00001, 100]]
        #   _C.DATA.VAL.FILTER_SAMPLES.SIGNS = [['gt'], ['lt', 'gt']]
        # This way, the images will be removed by 'max' and then by 'foreground' and 'mean'
        _C.DATA.VAL.FILTER_SAMPLES = CN()
        # Whether to enable or not the filtering by properties
        _C.DATA.VAL.FILTER_SAMPLES.ENABLE = False
        # List of lists of properties to apply a filter. Available properties are: ['foreground', 'mean', 'min', 'max', 
        # 'target_mean', 'target_min', 'target_max', 'diff', 'diff_by_min_max_ratio', 'diff_by_target_min_max_ratio']
        _C.DATA.VAL.FILTER_SAMPLES.PROPS = []
        # List of ints/float that represent the values of the properties listed in 'DATA.VAL.FILTER_SAMPLES.PROPS'
        # that the images need to satisfy to not be dropped.
        _C.DATA.VAL.FILTER_SAMPLES.VALUES = []
        # List of list of signs to do the comparison. Options: ['gt', 'ge', 'lt', 'le'] that corresponds to "greather than", e.g. ">",
        # "greather equal", e.g. ">=", "less than", e.g. "<", and "less equal" e.g. "<=" comparisons.
        _C.DATA.VAL.FILTER_SAMPLES.SIGNS = []
        # Whether to normalize the samples before comparison
        _C.DATA.VAL.FILTER_SAMPLES.NORM_BEFORE = False

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Data augmentation (DA)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        _C.AUGMENTOR = CN()
        # Flag to activate DA
        _C.AUGMENTOR.ENABLE = False
        # Probability of each transformation
        _C.AUGMENTOR.DA_PROB = 0.5
        # Create samples of the DA made. Useful to check the output images made.
        _C.AUGMENTOR.AUG_SAMPLES = True
        # Draw a grid in the augenation samples generated. Used when _C.AUGMENTOR.AUG_SAMPLES=True
        _C.AUGMENTOR.DRAW_GRID = True
        # Number of samples to create
        _C.AUGMENTOR.AUG_NUM_SAMPLES = 10
        # Flag to shuffle the training data on every epoch
        _C.AUGMENTOR.SHUFFLE_TRAIN_DATA_EACH_EPOCH = True
        # Flag to shuffle the validation data on every epoch
        _C.AUGMENTOR.SHUFFLE_VAL_DATA_EACH_EPOCH = False
        # Rotation of 90º to the subvolumes
        _C.AUGMENTOR.ROT90 = False
        # Random rotation between a defined range
        _C.AUGMENTOR.RANDOM_ROT = False
        # Range of random rotations
        _C.AUGMENTOR.RANDOM_ROT_RANGE = (-180, 180)
        # Apply shear to images
        _C.AUGMENTOR.SHEAR = False
        # Shear range. Expected value range is around [-360, 360], with reasonable values being in the range of [-45, 45].
        _C.AUGMENTOR.SHEAR_RANGE = (-20, 20)
        # Apply zoom to images
        _C.AUGMENTOR.ZOOM = False
        # Zoom range. Scaling factor to use, where 1.0 denotes “no change” and 0.5 is zoomed out to 50 percent of the original size.
        _C.AUGMENTOR.ZOOM_RANGE = (0.8, 1.2)
        # Whether to apply or not zoom in Z axis (for 3D volumes).
        _C.AUGMENTOR.ZOOM_IN_Z = False
        # Apply shift
        _C.AUGMENTOR.SHIFT = False
        # Shift range. Translation as a fraction of the image height/width (x-translation, y-translation), where 0 denotes
        # “no change” and 0.5 denotes “half of the axis size”.
        _C.AUGMENTOR.SHIFT_RANGE = (0.1, 0.2)
        # How to fill up the new values created with affine transformations (rotations, shear, shift and zoom).
        # Only keep modes common to skimage & scipy: 'constant', 'reflect', 'wrap' and 'symmetric
        # Dropped 'edge'/'nearest' for simplicity
        _C.AUGMENTOR.AFFINE_MODE = "reflect"
        # Make vertical flips
        _C.AUGMENTOR.VFLIP = False
        # Make horizontal flips
        _C.AUGMENTOR.HFLIP = False
        # Make z-axis flips
        _C.AUGMENTOR.ZFLIP = False
        # Elastic transformations
        _C.AUGMENTOR.ELASTIC = False
        # Strength of the distortion field. Higher values mean that pixels are moved further with respect to the distortion
        # field’s direction. Set this to around 10 times the value of sigma for visible effects.
        _C.AUGMENTOR.E_ALPHA = (12, 16)
        # Standard deviation of the gaussian kernel used to smooth the distortion fields.  Higher values (for 128x128 images
        # around 5.0) lead to more water-like effects, while lower values (for 128x128 images around 1.0 and lower) lead to
        # more noisy, pixelated images. Set this to around 1/10th of alpha for visible effects.
        _C.AUGMENTOR.E_SIGMA = 4
        # Parameter that defines the handling of newly created pixels with the elastic transformation
        _C.AUGMENTOR.E_MODE = "constant"
        # Gaussian blur
        _C.AUGMENTOR.G_BLUR = False
        # Standard deviation of the gaussian kernel. Values in the range 0.0 (no blur) to 3.0 (strong blur) are common.
        _C.AUGMENTOR.G_SIGMA = (1.0, 2.0)
        # To blur an image by computing median values over neighbourhoods
        _C.AUGMENTOR.MEDIAN_BLUR = False
        # Median blur kernel size
        _C.AUGMENTOR.MB_KERNEL = (3, 7)
        # Blur images in a way that fakes camera or object movements
        _C.AUGMENTOR.MOTION_BLUR = False
        # Kernel size to use in motion blur
        _C.AUGMENTOR.MOTB_K_RANGE = (8, 12)
        # Gamma contrast
        _C.AUGMENTOR.GAMMA_CONTRAST = False
        # Exponent for the contrast adjustment. Higher values darken the image
        _C.AUGMENTOR.GC_GAMMA = (1.25, 1.75)
        # To apply brightness changes to images
        _C.AUGMENTOR.BRIGHTNESS = False
        # Strength of the brightness range.
        _C.AUGMENTOR.BRIGHTNESS_FACTOR = (-0.1, 0.1)
        # If apply same contrast to the entire image or select one for each slice. For 2D does not matter but yes for
        # 3D images. Possible values: '2D' or '3D'. Used when '_C.PROBLEM.NDIM' = '3D'.
        _C.AUGMENTOR.BRIGHTNESS_MODE = "3D"
        # To apply contrast changes to images
        _C.AUGMENTOR.CONTRAST = False
        # Strength of the contrast change range.
        _C.AUGMENTOR.CONTRAST_FACTOR = (-0.1, 0.1)
        # If apply same contrast to the entire image or select one for each slice. For 2D does not matter but yes for
        # 3D images. Possible values: '2D' or '3D'. Used when '_C.PROBLEM.NDIM' = '3D'.
        _C.AUGMENTOR.CONTRAST_MODE = "3D"
        # Set a certain fraction of pixels in images to zero (not get confused with the dropout concept of neural networks)
        _C.AUGMENTOR.DROPOUT = False
        # Range to take the probability to drop a pixel
        _C.AUGMENTOR.DROP_RANGE = (0, 0.2)
        # To fill one or more rectangular areas in an image using a fill mode
        _C.AUGMENTOR.CUTOUT = False
        # Range of number of areas to fill the image with. Reasonable values between range [0,4]
        _C.AUGMENTOR.COUT_NB_ITERATIONS = (1, 3)
        # Size of the areas in % of the corresponding image size
        _C.AUGMENTOR.COUT_SIZE = (0.05, 0.3)
        # Value to fill the area of cutout
        _C.AUGMENTOR.COUT_CVAL = 0.0
        # Apply cutout to the segmentation mask
        _C.AUGMENTOR.COUT_APPLY_TO_MASK = False
        # To apply cutblur operation
        _C.AUGMENTOR.CUTBLUR = False
        # Size of the region to apply cutblur
        _C.AUGMENTOR.CBLUR_SIZE = (0.2, 0.4)
        # Range of the downsampling to be made in cutblur
        _C.AUGMENTOR.CBLUR_DOWN_RANGE = (2, 8)
        # Whether to apply cut-and-paste just LR into HR image. If False, HR to LR will be applied also (see Figure 1
        # of the paper https://arxiv.org/pdf/2004.00448.pdf)
        _C.AUGMENTOR.CBLUR_INSIDE = True
        # Apply cutmix operation
        _C.AUGMENTOR.CUTMIX = False
        # Size of the region to apply cutmix
        _C.AUGMENTOR.CMIX_SIZE = (0.2, 0.4)
        # Apply noise to a region of the image
        _C.AUGMENTOR.CUTNOISE = False
        # Range to choose a value that will represent the % of the maximum value of the image that will be used as the std
        # of the Gaussian Noise distribution
        _C.AUGMENTOR.CNOISE_SCALE = (0.05, 0.1)
        # Number of areas to fill with noise
        _C.AUGMENTOR.CNOISE_NB_ITERATIONS = (1, 3)
        # Size of the regions
        _C.AUGMENTOR.CNOISE_SIZE = (0.2, 0.4)
        # Add miss-aligment augmentation
        _C.AUGMENTOR.MISALIGNMENT = False
        # Maximum pixel displacement in 'xy'-plane for misalignment
        _C.AUGMENTOR.MS_DISPLACEMENT = 16
        # Ratio of rotation-based mis-alignment
        _C.AUGMENTOR.MS_ROTATE_RATIO = 0.5
        # Augment the image by creating a black line in a random position
        _C.AUGMENTOR.MISSING_SECTIONS = False
        # Probability of adding a missing section in each channel
        _C.AUGMENTOR.MISSP_CHANNEL_PB = 0.5
        # Iterations to dilate the missing line with
        _C.AUGMENTOR.MISSP_ITERATIONS = (10, 30)
        # Convert images in grasycale gradually based on '_C.AUGMENTOR.GRAY_RANGE'
        _C.AUGMENTOR.GRAYSCALE = False
        # Shuffle channels of the images
        _C.AUGMENTOR.CHANNEL_SHUFFLE = False
        # Apply gridmask to the image. Original paper: https://arxiv.org/pdf/2001.04086v1.pdf
        _C.AUGMENTOR.GRIDMASK = False
        # Determines the keep ratio of an input image
        _C.AUGMENTOR.GRID_RATIO = 0.6
        # Range to choose a d value
        _C.AUGMENTOR.GRID_D_RANGE = (0.4, 1)
        # Rotation of the mask in GridMask. Needs to be between [0,1] where 1 is 360 degrees.
        _C.AUGMENTOR.GRID_ROTATE = 1.0
        # Whether to invert the mask
        _C.AUGMENTOR.GRID_INVERT = False
        # Add Gaussian noise
        _C.AUGMENTOR.GAUSSIAN_NOISE = False
        _C.AUGMENTOR.GAUSSIAN_NOISE_MEAN = 0.0
        _C.AUGMENTOR.GAUSSIAN_NOISE_VAR = 0.05
        _C.AUGMENTOR.GAUSSIAN_NOISE_USE_INPUT_IMG_MEAN_AND_VAR = False
        # Add Poisson noise
        _C.AUGMENTOR.POISSON_NOISE = False
        # Add salt (replaces random pixels with 1)
        _C.AUGMENTOR.SALT = False
        _C.AUGMENTOR.SALT_AMOUNT = 0.05
        # Add pepper (replaces random pixels with 0 (for unsigned images) or -1 (for signed images))
        _C.AUGMENTOR.PEPPER = False
        _C.AUGMENTOR.PEPPER_AMOUNT = 0.05
        # Whether to add Poisson noise
        _C.AUGMENTOR.SALT_AND_PEPPER = False
        _C.AUGMENTOR.SALT_AND_PEPPER_AMOUNT = 0.05
        _C.AUGMENTOR.SALT_AND_PEPPER_PROP = 0.5

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Model definition
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        _C.MODEL = CN()
        # Whether to define manually the model ('biapy'), load a pretrained one from BioImage Model Zoo ('bmz') or use one
        # available in TorchVision ('torchvision').
        # Options: ["biapy", "bmz", "torchvision"]
        _C.MODEL.SOURCE = "biapy"

        #
        # BMZ BACKEND MODELS AND OPTIONS
        #
        _C.MODEL.BMZ = CN()
        # DOI or nickname of the model from BMZ to load. It can not be empty if MODEL.SOURCE = "bmz".
        _C.MODEL.BMZ.SOURCE_MODEL_ID = ""
        # BMZ model export options
        _C.MODEL.BMZ.EXPORT = CN()
        # Whether to activate or not the exporation of the used model to the BMZ format after train and/or test 
        _C.MODEL.BMZ.EXPORT.ENABLE = False
        # Name of the model to create. It should be something meaningful. Take other models in https://bioimage.io/#/ as reference.
        _C.MODEL.BMZ.EXPORT.MODEL_NAME = ""
        # Description of the model. It should be something meaningful. Take other models in https://bioimage.io/#/ as reference.
        # E.g. "Mitochondria segmentation for electron microscopy"
        _C.MODEL.BMZ.EXPORT.DESCRIPTION = ""
        # List of authors of the model. Each item must be a dict containing "name" and "githubuser".
        # E.g. [{"name": "Daniel", "github_user": "danifranco"}]
        _C.MODEL.BMZ.EXPORT.AUTHORS = []
        # License of the model.
        _C.MODEL.BMZ.EXPORT.LICENSE = "CC-BY-4.0"
        # Path to a .md extension file with the documentation of the model. If it is not set so the model documentation will point to 
        # BiaPy doc: https://github.com/BiaPyX/BiaPy/blob/master/README.md". Take other models in https://bioimage.io/#/ as reference.
        _C.MODEL.BMZ.EXPORT.DOCUMENTATION = ""
        # List of tags. Here the type of dataset and the target object should be provided. BiaPy automatically sets the following tags: 
        #   * "biapy": to represent that the model was created with BiaPy.
        #   * "pytorch": to represent that you are using Pytorch
        #   * "2d" or "3d": depending on the image dimensions one or the other is selected.
        #   * workflow tag: depending on the workflow the tag is set. E.g. "semantic-segmentation"
        #
        # So, what you can set for instance is: ["electron-microscopy", "mitochondria"]
        _C.MODEL.BMZ.EXPORT.TAGS = []
        # Citations. It must be a list of dictionaries with keys "text" and "doi". E.g.:
        # [{"text": "training library", "doi": "10.1038/s41592-025-02699-y"}, {"text": "architecture", "doi": "10.1109/LGRS.2018.2802944"},
        #  {"text": "data", "doi": "10.48550/arXiv.1812.06024"}]
        _C.MODEL.BMZ.EXPORT.CITE = []
        # Information of the dataset. It must be a list of just one dict item with keys "name", "doi" and "image_modality". It can have 
        # also "dataset_id" key but it is optional (must match a dataset in the BioImage Model Zoo). E.g: 
        #   [{
        #       "name": "CartoCell", 
        #       "doi": "10.1016/j.crmeth.2023.100597", 
        #       "image_modality": "fluorescence microscopy", 
        #       "dataset_id": "biapy/cartocell_cyst_segmentation",
        #   }]
        _C.MODEL.BMZ.EXPORT.DATASET_INFO = [{}]
        # Version of the model
        _C.MODEL.BMZ.EXPORT.MODEL_VERSION = "0.1.0"
        # If you are loading a BMZ model you can enable this option to avoid setting all above variables and instead reuse the same
        # information that was present in that model. You need still to set 'MODEL.BMZ.EXPORT.ENABLE' to 'True' and nothing else.
        _C.MODEL.BMZ.EXPORT.REUSE_BMZ_CONFIG = False

        #
        # TOCHIVISION BACKEND MODELS AND OPTIONS
        #
        # BiaPy support using models of Torchvision . It can not be empty if MODEL.SOURCE = "torchvision".
        # Models available here: https://pytorch.org/vision/stable/models.html
        # They can be listed with: "from torchvision.models import list_models; list_models()"
        #
        # Semantic segmentation (https://pytorch.org/vision/stable/models.html#semantic-segmentation):
        # 'deeplabv3_mobilenet_v3_large', 'deeplabv3_resnet101', 'deeplabv3_resnet50', 'fcn_resnet101', 'fcn_resnet50',
        # 'lraspp_mobilenet_v3_large'
        #
        # Object Detection (https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection)
        # 'fasterrcnn_mobilenet_v3_large_320_fpn', 'fasterrcnn_mobilenet_v3_large_fpn', 'fasterrcnn_resnet50_fpn',
        # 'fasterrcnn_resnet50_fpn_v2', 'fcos_resnet50_fpn', 'ssd300_vgg16', 'ssdlite320_mobilenet_v3_large',
        # 'retinanet_resnet50_fpn', 'retinanet_resnet50_fpn_v2',
        #
        # Instance Segmentation (https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection)
        # 'maskrcnn_resnet50_fpn', 'maskrcnn_resnet50_fpn_v2'
        #
        # Image classification (https://pytorch.org/vision/stable/models.html#classification):
        # 'alexnet', 'convnext_base', 'convnext_large', 'convnext_small', 'convnext_tiny', 'densenet121', 'densenet161',
        # 'densenet169', 'densenet201', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
        # 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 'efficientnet_v2_l', 'efficientnet_v2_m',
        # 'efficientnet_v2_s', 'googlenet', 'inception_v3', 'maxvit_t', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3',
        # 'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small',  'quantized_googlenet', 'quantized_inception_v3',
        # 'quantized_mobilenet_v2', 'quantized_mobilenet_v3_large', 'quantized_resnet18', 'quantized_resnet50',
        # 'quantized_resnext101_32x8d', 'quantized_resnext101_64x4d', 'quantized_shufflenet_v2_x0_5', 'quantized_shufflenet_v2_x1_0',
        # 'quantized_shufflenet_v2_x1_5', 'quantized_shufflenet_v2_x2_0', 'regnet_x_16gf', 'regnet_x_1_6gf', 'regnet_x_32gf',
        # 'regnet_x_3_2gf', 'regnet_x_400mf', 'regnet_x_800mf', 'regnet_x_8gf', 'regnet_y_128gf', 'regnet_y_16gf', 'regnet_y_1_6gf',
        # 'regnet_y_32gf', 'regnet_y_3_2gf', 'regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_8gf', 'resnet101', 'resnet152',
        # 'resnet18', 'resnet34', 'resnet50', 'resnext101_32x8d', 'resnext101_64x4d', 'resnext50_32x4d', 'retinanet_resnet50_fpn',
        # 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0',
        # 'squeezenet1_0', 'squeezenet1_1', 'swin_b', 'swin_s', 'swin_t', 'swin_v2_b', 'swin_v2_s', 'swin_v2_t',
        # 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'vit_b_16', 'vit_b_32',
        # 'vit_h_14', 'vit_l_16', 'vit_l_32', 'wide_resnet101_2', 'wide_resnet50_2'
        #
        # Listed but not supported:
        #
        # (NOT SUPPORTED) Video classification (https://pytorch.org/vision/stable/models.html#video-classification):
        # 'mc3_18', 'mvit_v1_b', 'mvit_v2_s', 'r2plus1d_18', 'r3d_18','swin3d_s', 'swin3d_t', 's3d', 'swin3d_b'
        #
        # (NOT SUPPORTED) Optical flow (https://pytorch.org/vision/stable/models.html#optical-flow):
        # 'raft_large', 'raft_small'
        #
        # (NOT SUPPORTED) Person Keypoint Detection (https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection)
        # 'keypointrcnn_resnet50_fpn'
        #
        _C.MODEL.TORCHVISION_MODEL_NAME = ""

        #
        # BIAPY BACKEND MODELS
        #
        # Architecture of the network. Possible values are:
        #   * Semantic segmentation: 'unet', 'resunet', 'resunet++', 'attention_unet', 'multiresunet', 'seunet', 'resunet_se', 'unetr', 'unext_v1', 'unext_v2'
        #   * Instance segmentation: 'unet', 'resunet', 'resunet++', 'attention_unet', 'multiresunet', 'seunet', 'resunet_se', 'unetr', 'unext_v1', 'unext_v2'
        #   * Detection: 'unet', 'resunet', 'resunet++', 'attention_unet', 'multiresunet', 'seunet', 'resunet_se', 'unetr', 'unext_v1', 'unext_v2'
        #   * Denoising: 'unet', 'resunet', 'resunet++', 'attention_unet', 'seunet', 'resunet_se', 'unext_v1', 'unext_v2'
        #   * Super-resolution: 'edsr', 'rcan', 'dfcan', 'wdsr', 'unet', 'resunet', 'resunet++', 'seunet', 'resunet_se', 'attention_unet', 'multiresunet', 'unext_v1', 'unext_v2'
        #   * Self-supervision: 'unet', 'resunet', 'resunet++', 'attention_unet', 'multiresunet', 'seunet', 'resunet_se', 'unetr', 'edsr', 'rcan', 'dfcan', 'wdsr', 'vit', 'mae', 'unext_v1', 'unext_v2'
        #   * Classification: 'simple_cnn', 'vit', 'efficientnet_b[0-7]' (only 2D)
        #   * Image to image: 'edsr', 'rcan', 'dfcan', 'wdsr', 'unet', 'resunet', 'resunet++', 'seunet', 'resunet_se', 'attention_unet', 'unetr', 'multiresunet', 'unext_v1', 'unext_v2'
        _C.MODEL.ARCHITECTURE = "unet"
        # Number of feature maps on each level of the network.
        _C.MODEL.FEATURE_MAPS = [16, 32, 64, 128, 256]
        # Values to make the dropout with. Set to 0 to prevent dropout. When using it with 'ViT' or 'unetr'
        # a list with just one number must be provided
        _C.MODEL.DROPOUT_VALUES = [0.0, 0.0, 0.0, 0.0, 0.0]
        # Normalization layer (one of 'bn', 'sync_bn' 'in', 'gn' or 'none').
        _C.MODEL.NORMALIZATION = "bn"
        # Kernel size
        _C.MODEL.KERNEL_SIZE = 3
        # Upsampling layer to use in the model. Options: ["upsampling", "convtranspose"]
        _C.MODEL.UPSAMPLE_LAYER = "convtranspose"
        # Activation function to use along the model
        _C.MODEL.ACTIVATION = "ELU"
        # Number of classes including the background class (that should be using 0 label)
        _C.DATA.N_CLASSES = 2
        # Downsampling to be made in Z. This value will be the third integer of the MaxPooling operation. When facing
        # anysotropic datasets set it to get better performance
        _C.MODEL.Z_DOWN = [0, 0, 0, 0]
        # For each level of the model (U-Net levels), set to true or false if the dimensions of the feature maps are isotropic.
        _C.MODEL.ISOTROPY = [True, True, True, True, True]
        # Include extra convolutional layers with larger kernel at the beginning and end of the U-Net-like model.
        _C.MODEL.LARGER_IO = False
        # Checkpoint: set to True to load previous training weigths (needed for inference or to make fine-tunning)
        _C.MODEL.LOAD_CHECKPOINT = False
        # When loading checkpoints whether only model's weights are going to be loaded or optimizer, epochs and loss_scaler.
        _C.MODEL.LOAD_CHECKPOINT_ONLY_WEIGHTS = True
        # Decide which checkpoint to load from job's dir if PATHS.CHECKPOINT_FILE is ''.
        # Options: 'best_on_val' or 'last_on_train'
        _C.MODEL.LOAD_CHECKPOINT_EPOCH = "best_on_val"
        # Whether to load the model from the checkpoint instead of builiding it following 'MODEL.ARCHITECTURE' when 'MODEL.SOURCE' is "biapy"
        _C.MODEL.LOAD_MODEL_FROM_CHECKPOINT = True
        # Format of the output checkpoint. Options are 'pth' (native PyTorch format) or 'safetensors' (https://github.com/huggingface/safetensors)
        _C.MODEL.OUT_CHECKPOINT_FORMAT = "pth"
        # To skip loading those layers that do not match in shape with the given checkpoint. If this is set to False a regular load function will be 
        # done, which will fail if a layer mismatch is found. Only workes when 'MODEL.LOAD_MODEL_FROM_CHECKPOINT' is True
        _C.MODEL.SKIP_UNMATCHED_LAYERS = False
        # Epochs to save a checkpoint of the model apart from the ones saved with LOAD_CHECKPOINT_ONLY_WEIGHTS. Set it to -1 to
        # not do it.
        _C.MODEL.SAVE_CKPT_FREQ = -1
        # Number of ConvNeXtBlocks in each level.
        _C.MODEL.CONVNEXT_LAYERS = [2, 2, 2, 2, 2]  # CONVNEXT_LAYERS
        # Maximum Stochastic Depth probability for the U-NeXt model.
        _C.MODEL.CONVNEXT_SD_PROB = 0.1
        # Layer Scale parameter for the U-NeXt model.
        _C.MODEL.CONVNEXT_LAYER_SCALE = 1e-6
        # Size of the stem kernel in the U-NeXt model.
        _C.MODEL.CONVNEXT_STEM_K_SIZE = 2

        # TRANSFORMERS MODELS
        # Type of model. Options are "custom", "vit_base_patch16", "vit_large_patch16" and "vit_huge_patch16". On custom setting
        # the rest of the ViT parameters can be modified as other options will set them automatically.
        _C.MODEL.VIT_MODEL = "custom"
        # Size of the patches that are extracted from the input image.
        _C.MODEL.VIT_TOKEN_SIZE = 16
        # Dimension of the embedding space
        _C.MODEL.VIT_EMBED_DIM = 768
        # Number of transformer encoder layers
        _C.MODEL.VIT_NUM_LAYERS = 12
        # Number of heads in the multi-head attention layer.
        _C.MODEL.VIT_NUM_HEADS = 12
        # Size of the dense layers of the final classifier. This value will mutiply 'VIT_EMBED_DIM'
        _C.MODEL.VIT_MLP_RATIO = 4.0
        # Normalization layer epsion
        _C.MODEL.VIT_NORM_EPS = 1e-6

        # Dimension of the embedding space for the MAE decoder
        _C.MODEL.MAE_DEC_HIDDEN_SIZE = 512
        # Number of transformer decoder layers
        _C.MODEL.MAE_DEC_NUM_LAYERS = 8
        # Number of heads in the multi-head attention layer.
        _C.MODEL.MAE_DEC_NUM_HEADS = 16
        # Size of the dense layers of the final classifier
        _C.MODEL.MAE_DEC_MLP_DIMS = 2048
        # Type of the masking strategy. Options: ["grid", "random"]
        _C.MODEL.MAE_MASK_TYPE = "grid"
        # Percentage of the input image to mask (applied only when MODEL.MAE_MASK_TYPE == "random"). Value between 0 and 1.
        _C.MODEL.MAE_MASK_RATIO = 0.5

        # UNETR
        # Multiple of the transformer encoder layers from of which the skip connection signal is going to be extracted
        _C.MODEL.UNETR_VIT_HIDD_MULT = 3
        # Number of filters in the first UNETR's layer of the decoder. In each layer the previous number of filters is doubled.
        _C.MODEL.UNETR_VIT_NUM_FILTERS = 16
        # Decoder activation
        _C.MODEL.UNETR_DEC_ACTIVATION = "relu"
        # Decoder convolutions' kernel size
        _C.MODEL.UNETR_DEC_KERNEL_SIZE = 3

        # Specific for SR models based on U-Net architectures. Options are ["pre", "post"]
        _C.MODEL.UNET_SR_UPSAMPLE_POSITION = "pre"

        # RCAN
        # Number of RG modules
        _C.MODEL.RCAN_RG_BLOCK_NUM = 10
        # Number of RCAB modules in each RG block
        _C.MODEL.RCAN_RCAB_BLOCK_NUM = 20
        # Filters in the convolutions
        _C.MODEL.RCAN_CONV_FILTERS = 16
        # Channel reduction ratio for channel attention
        _C.MODEL.RCAN_REDUCTION_RATIO = 16
        # Whether to maintain or not the upscaling layer. 
        _C.MODEL.RCAN_UPSCALING_LAYER = True

        _C.MODEL.HRNET_64 = CN()
        _C.MODEL.HRNET_64.Z_DOWN = True
        _C.MODEL.HRNET_64.STAGE2 = CN()
        _C.MODEL.HRNET_64.STAGE2.NUM_MODULES = 1
        _C.MODEL.HRNET_64.STAGE2.NUM_BRANCHES = 2
        _C.MODEL.HRNET_64.STAGE2.NUM_BLOCKS = [4, 4]
        _C.MODEL.HRNET_64.STAGE2.NUM_CHANNELS = [64, 128]
        _C.MODEL.HRNET_64.STAGE2.BLOCK = 'BASIC'

        _C.MODEL.HRNET_64.STAGE3 = CN()
        _C.MODEL.HRNET_64.STAGE3.NUM_MODULES = 4
        _C.MODEL.HRNET_64.STAGE3.NUM_BRANCHES = 3
        _C.MODEL.HRNET_64.STAGE3.NUM_BLOCKS = [4, 4, 4]
        _C.MODEL.HRNET_64.STAGE3.NUM_CHANNELS = [64, 128, 256]
        _C.MODEL.HRNET_64.STAGE3.BLOCK = 'BASIC'

        _C.MODEL.HRNET_64.STAGE4 = CN()
        _C.MODEL.HRNET_64.STAGE4.NUM_MODULES = 3
        _C.MODEL.HRNET_64.STAGE4.NUM_BRANCHES = 4
        _C.MODEL.HRNET_64.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
        _C.MODEL.HRNET_64.STAGE4.NUM_CHANNELS = [64, 128, 256, 512]
        _C.MODEL.HRNET_64.STAGE4.BLOCK = 'BASIC'


        # configs for HRNet48
        _C.MODEL.HRNET_48 = CN()
        _C.MODEL.HRNET_48.Z_DOWN = True
        _C.MODEL.HRNET_48.STAGE2 = CN()
        _C.MODEL.HRNET_48.STAGE2.NUM_MODULES = 1
        _C.MODEL.HRNET_48.STAGE2.NUM_BRANCHES = 2
        _C.MODEL.HRNET_48.STAGE2.NUM_BLOCKS = [4, 4]
        _C.MODEL.HRNET_48.STAGE2.NUM_CHANNELS = [48, 96]
        _C.MODEL.HRNET_48.STAGE2.BLOCK = 'BASIC'

        _C.MODEL.HRNET_48.STAGE3 = CN()
        _C.MODEL.HRNET_48.STAGE3.NUM_MODULES = 4
        _C.MODEL.HRNET_48.STAGE3.NUM_BRANCHES = 3
        _C.MODEL.HRNET_48.STAGE3.NUM_BLOCKS = [4, 4, 4]
        _C.MODEL.HRNET_48.STAGE3.NUM_CHANNELS = [48, 96, 192]
        _C.MODEL.HRNET_48.STAGE3.BLOCK = 'BASIC'

        _C.MODEL.HRNET_48.STAGE4 = CN()
        _C.MODEL.HRNET_48.STAGE4.NUM_MODULES = 3
        _C.MODEL.HRNET_48.STAGE4.NUM_BRANCHES = 4
        _C.MODEL.HRNET_48.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
        _C.MODEL.HRNET_48.STAGE4.NUM_CHANNELS = [48, 96, 192, 384]
        _C.MODEL.HRNET_48.STAGE4.BLOCK = 'BASIC'


        # configs for HRNet32
        _C.MODEL.HRNET_32 = CN()
        _C.MODEL.HRNET_32.Z_DOWN = True
        _C.MODEL.HRNET_32.STAGE2 = CN()
        _C.MODEL.HRNET_32.STAGE2.NUM_MODULES = 1
        _C.MODEL.HRNET_32.STAGE2.NUM_BRANCHES = 2
        _C.MODEL.HRNET_32.STAGE2.NUM_BLOCKS = [4, 4]
        _C.MODEL.HRNET_32.STAGE2.NUM_CHANNELS = [32, 64]
        _C.MODEL.HRNET_32.STAGE2.BLOCK = 'BASIC'

        _C.MODEL.HRNET_32.STAGE3 = CN()
        _C.MODEL.HRNET_32.STAGE3.NUM_MODULES = 4
        _C.MODEL.HRNET_32.STAGE3.NUM_BRANCHES = 3
        _C.MODEL.HRNET_32.STAGE3.NUM_BLOCKS = [4, 4, 4]
        _C.MODEL.HRNET_32.STAGE3.NUM_CHANNELS = [32, 64, 128]
        _C.MODEL.HRNET_32.STAGE3.BLOCK = 'BASIC'

        _C.MODEL.HRNET_32.STAGE4 = CN()
        _C.MODEL.HRNET_32.STAGE4.NUM_MODULES = 3
        _C.MODEL.HRNET_32.STAGE4.NUM_BRANCHES = 4
        _C.MODEL.HRNET_32.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
        _C.MODEL.HRNET_32.STAGE4.NUM_CHANNELS = [32, 64, 128, 256]
        _C.MODEL.HRNET_32.STAGE4.BLOCK = 'BASIC'


        # configs for HRNet18
        _C.MODEL.HRNET_18 = CN()
        _C.MODEL.HRNET_18.Z_DOWN = True
        _C.MODEL.HRNET_18.STAGE2 = CN()
        _C.MODEL.HRNET_18.STAGE2.NUM_MODULES = 1
        _C.MODEL.HRNET_18.STAGE2.NUM_BRANCHES = 2
        _C.MODEL.HRNET_18.STAGE2.NUM_BLOCKS = [4, 4]
        _C.MODEL.HRNET_18.STAGE2.NUM_CHANNELS = [18, 36]
        _C.MODEL.HRNET_18.STAGE2.BLOCK = 'BASIC'

        _C.MODEL.HRNET_18.STAGE3 = CN()
        _C.MODEL.HRNET_18.STAGE3.NUM_MODULES = 4
        _C.MODEL.HRNET_18.STAGE3.NUM_BRANCHES = 3
        _C.MODEL.HRNET_18.STAGE3.NUM_BLOCKS = [4, 4, 4]
        _C.MODEL.HRNET_18.STAGE3.NUM_CHANNELS = [18, 36, 72]
        _C.MODEL.HRNET_18.STAGE3.BLOCK = 'BASIC'

        _C.MODEL.HRNET_18.STAGE4 = CN()
        _C.MODEL.HRNET_18.STAGE4.NUM_MODULES = 3
        _C.MODEL.HRNET_18.STAGE4.NUM_BRANCHES = 4
        _C.MODEL.HRNET_18.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
        _C.MODEL.HRNET_18.STAGE4.NUM_CHANNELS = [18, 36, 72, 144]
        _C.MODEL.HRNET_18.STAGE4.BLOCK = 'BASIC'

        # configs for HRNet2x20
        _C.MODEL.HRNET2X_20 = CN()
        _C.MODEL.HRNET2X_20.Z_DOWN = True
        _C.MODEL.HRNET2X_20.STAGE1 = CN()
        _C.MODEL.HRNET2X_20.STAGE1.NUM_MODULES = 1
        _C.MODEL.HRNET2X_20.STAGE1.NUM_BRANCHES = 2
        _C.MODEL.HRNET2X_20.STAGE1.NUM_BLOCKS = [4, 4]
        _C.MODEL.HRNET2X_20.STAGE1.NUM_CHANNELS = [32, 64]
        _C.MODEL.HRNET2X_20.STAGE1.BLOCK = 'BOTTLENECK'

        _C.MODEL.HRNET2X_20.STAGE2 = CN()
        _C.MODEL.HRNET2X_20.STAGE2.NUM_MODULES = 1
        _C.MODEL.HRNET2X_20.STAGE2.NUM_BRANCHES = 3
        _C.MODEL.HRNET2X_20.STAGE2.NUM_BLOCKS = [4, 4, 4]
        _C.MODEL.HRNET2X_20.STAGE2.NUM_CHANNELS = [20, 40, 80]
        _C.MODEL.HRNET2X_20.STAGE2.BLOCK = 'BASIC'

        _C.MODEL.HRNET2X_20.STAGE3 = CN()
        _C.MODEL.HRNET2X_20.STAGE3.NUM_MODULES = 4
        _C.MODEL.HRNET2X_20.STAGE3.NUM_BRANCHES = 4
        _C.MODEL.HRNET2X_20.STAGE3.NUM_BLOCKS = [4, 4, 4, 4]
        _C.MODEL.HRNET2X_20.STAGE3.NUM_CHANNELS = [20, 40, 80, 160]
        _C.MODEL.HRNET2X_20.STAGE3.BLOCK = 'BASIC'

        _C.MODEL.HRNET2X_20.STAGE4 = CN()
        _C.MODEL.HRNET2X_20.STAGE4.NUM_MODULES = 3
        _C.MODEL.HRNET2X_20.STAGE4.NUM_BRANCHES = 5
        _C.MODEL.HRNET2X_20.STAGE4.NUM_BLOCKS = [4, 4, 4, 4, 4]
        _C.MODEL.HRNET2X_20.STAGE4.NUM_CHANNELS = [20, 40, 80, 160, 320]
        _C.MODEL.HRNET2X_20.STAGE4.BLOCK = 'BASIC'

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loss
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        _C.LOSS = CN()
        # Loss type, different options depending on the workflow. If empty the default loss on each case will be set:
        #   * Semantic segmentation:
        #       * "CE" (default): cross entropy. Ref: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        #       * "DICE": Dice loss. Ref: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch
        #       * "W_CE_DICE": CE and Dice (with a weight term on each one that must sum 1). Ref: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch
        #   * Instance segmentation: automatically set depending on the channels selected (PROBLEM.INSTANCE_SEG.DATA_CHANNELS). 
        #     It can be also set manually with PROBLEM.INSTANCE_SEG.DATA_CHANNELS_LOSSES. 
        #   * Detection:
        #       * "CE" (default): cross entropy. Ref: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        #       * "DICE": Dice loss. Ref: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch
        #       * "W_CE_DICE": CE and Dice (with a weight term on each one that must sum 1). Ref: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch
        #   * Denoising:
        #       * "MAE": mean absolute error (MAE or L1 loss). Ref: https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html#torch.nn.L1Loss
        #       * "MSE" (default): mean square error (MSE). Ref: https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss
        #   * Super-resolution:
        #       * "MAE" (default): mean absolute error (MAE or L1 loss). Ref: https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html#torch.nn.L1Loss
        #       * "MSE": mean square error (MSE). Ref: https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss
        #       * "SSIM": structural similarity index measure (SSIM). Ref: https://lightning.ai/docs/torchmetrics/stable/image/structural_similarity.html#torchmetrics.image.StructuralSimilarityIndexMeasure
        #       * "W_MAE_SSIM": MAE and SSIM (with a weight term on each one that must sum 1).
        #       * "W_MSE_SSIM": MSE and SSIM (with a weight term on each one that must sum 1).
        #   * Self-supervision:
        #       These losses can only be set when PROBLEM.SELF_SUPERVISED.PRETEXT_TASK = "crappify". Otherwise it will be automatically set to MSE when
        #       PROBLEM.SELF_SUPERVISED.PRETEXT_TASK = "masking".
        #       * "MAE" (default): mean absolute error (MAE or L1 loss). Ref: https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html#torch.nn.L1Loss
        #       * "MSE": mean square error (MSE). Ref: https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss
        #       * "SSIM": structural similarity index measure (SSIM). Ref: https://lightning.ai/docs/torchmetrics/stable/image/structural_similarity.html#torchmetrics.image.StructuralSimilarityIndexMeasure
        #       * "W_MAE_SSIM": MAE and SSIM (with a weight term on each one that must sum 1).
        #       * "W_MSE_SSIM": MSE and SSIM (with a weight term on each one that must sum 1).
        #   * Classification:
        #       * "CE" (default): cross entropy. Ref: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        #   * Image to image:
        #       * "MAE" (default): mean absolute error (MAE or L1 loss). Ref: https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html#torch.nn.L1Loss
        #       * "MSE": mean square error (MSE). Ref: https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss
        #       * "SSIM": structural similarity index measure (SSIM). Ref: https://lightning.ai/docs/torchmetrics/stable/image/structural_similarity.html#torchmetrics.image.StructuralSimilarityIndexMeasure
        #       * "W_MAE_SSIM": MAE and SSIM (with a weight term on each one that must sum 1).
        #       * "W_MSE_SSIM": MSE and SSIM (with a weight term on each one that must sum 1).
        _C.LOSS.TYPE = ""
        # Weights to be applied in multiple loss combination cases, by multiplying the corresponding weight to each loss.
        # It works for all the workflows but the instance segmentation one, as in that case the weights must be set
        # in PROBLEM.INSTANCE_SEG.DATA_CHANNEL_LOSSES. The weights must sum 1. E.g. [0.3, 0.7].
        _C.LOSS.WEIGHTS = [0.66, 0.34]
        # To adjust the loss function based on the imbalance between classes. This measures how many pixels represent each class and assigns 
        # a weight map to the predictions based on that. It can be used along with LOSS.WEIGHTS. It is used when loss is LOSS.TYPE == "CE" 
        # in detection and semantic segmentation. In instance segmentation if "bce"/"ce" loss is set in 
        # PROBLEM.INSTANCE_SEG.DATA_CHANNEL_LOSSES this will also be applied.
        _C.LOSS.CLASS_REBALANCE = False
        # Whether to ignore a value in the loss and metric calculation. This is only available when LOSS.TYPE == "CE". This value will not only
        # be ignored in the loss computation but in the metrics, e.g. IoU.
        _C.LOSS.IGNORE_INDEX = -1
        _C.LOSS.CONTRAST = CN()
        _C.LOSS.CONTRAST.ENABLE = False
        _C.LOSS.CONTRAST.MEMORY_SIZE = 5000
        _C.LOSS.CONTRAST.PROJ_DIM = 256
        _C.LOSS.CONTRAST.PIXEL_UPD_FREQ = 10
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Training phase
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        _C.TRAIN = CN()
        _C.TRAIN.ENABLE = False
        # Enable verbosity
        _C.TRAIN.VERBOSE = False
        # Optimizer to use. Possible values: "SGD", "ADAM" or "ADAMW"
        _C.TRAIN.OPTIMIZER = "SGD"
        # Learning rate
        _C.TRAIN.LR = 1.0e-4
        # Weight decay
        _C.TRAIN.W_DECAY = 0.02
        # Coefficients used for computing running averages of gradient and its square. Used in ADAM and ADAMW optmizers
        _C.TRAIN.OPT_BETAS = (0.9, 0.999)
        # Batch size
        _C.TRAIN.BATCH_SIZE = 2
        # If memory or # gpus is limited, use this variable to maintain the effective batch size, which is
        # batch_size (per gpu) * nodes * (gpus per node) * accum_iter.
        _C.TRAIN.ACCUM_ITER = 1
        # Number of epochs to train the model
        _C.TRAIN.EPOCHS = 360
        # Epochs to wait with no validation data improvement until the training is stopped
        _C.TRAIN.PATIENCE = -1
        # Metrics to apply during training. Depending on the workflow different ones can be applied. If empty, some
        # default metrics will be configured automatically:
        #   * Semantic segmentation: 'iou' (called also Jaccard index)
        #   * Instance segmentation: automatically set depending on the channels selected (PROBLEM.INSTANCE_SEG.DATA_CHANNELS).
        #   * Detection: 'iou' (called also Jaccard index)
        #   * Denoising: 'mae', 'mse'
        #   * Super-resolution: "psnr", "mae", "mse", "ssim"
        #   * Self-supervision: "psnr", "mae", "mse", "ssim"
        #   * Classification: 'accuracy', 'top-5-accuracy'
        #   * Image to image: "psnr", "mae", "mse", "ssim"
        _C.TRAIN.METRICS = []

        # LR Scheduler
        _C.TRAIN.LR_SCHEDULER = CN()
        _C.TRAIN.LR_SCHEDULER.NAME = ""  # Possible options: 'warmupcosine', 'reduceonplateau', 'onecycle'
        # Lower bound on the learning rate used in 'warmupcosine' and 'reduceonplateau'
        _C.TRAIN.LR_SCHEDULER.MIN_LR = -1.0
        # Factor by which the learning rate will be reduced
        _C.TRAIN.LR_SCHEDULER.REDUCEONPLATEAU_FACTOR = 0.5
        # Number of epochs with no improvement after which learning rate will be reduced. Need to be less than 'TRAIN.PATIENCE'
        # otherwise it makes no sense
        _C.TRAIN.LR_SCHEDULER.REDUCEONPLATEAU_PATIENCE = -1
        # Cosine decay with a warm up consist in 2 phases: 1) a warm up phase which consists of increasing
        # the learning rate from TRAIN.LR_SCHEDULER.MIN_LR to TRAIN.LR value by a factor
        # during a certain number of epochs defined by 'TRAIN.LR_SCHEDULER.WARMUP_COSINE_DECAY_EPOCHS'
        # 2) after this will began the decay of the learning rate value using the cosine function.
        # Find a detailed explanation in: https://scorrea92.medium.com/cosine-learning-rate-decay-e8b50aa455b
        #
        # Epochs to do the warming up.
        _C.TRAIN.LR_SCHEDULER.WARMUP_COSINE_DECAY_EPOCHS = -1

        # Callbacks
        # To determine which value monitor to consider which epoch consider the best to save. Currently not used.
        _C.TRAIN.CHECKPOINT_MONITOR = "val_loss"
        # Add profiler callback to the training
        # _C.TRAIN.PROFILER = False
        # # Batch range to be analyzed
        # _C.TRAIN.PROFILER_BATCH_RANGE='10, 100'

        # _C.TRAIN.MAE_CALLBACK_EPOCHS = 5

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Inference phase
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        _C.TEST = CN()
        _C.TEST.ENABLE = False
        # Tries to reduce the memory footprint by separating crop/merge operations and by changing dtype of the predictions.
        # It is slower and not as precise as the "normal" inference process but saves memory. In 'TEST.BY_CHUNKS' it will
        # only save memory with the datatype change.
        _C.TEST.REDUCE_MEMORY = False
        # In the processing of 3D images, the primary image is segmented into smaller patches. These patches are subsequently
        # passed through a computational network. The outcome is a new image, typically saved as a TIF file, that retains the
        # dimensions of the original input. Notably, if the input image is sizable, this process can be memory-intensive. This
        # is because the quantity of patches is contingent on both the dimensions of the input and the selected padding/overlap
        # parameters (defined as 'DATA.TEST.PADDING' and 'DATA.TEST.OVERLAP').
        # To alleviate potential memory constraints, we offer an alternative: producing an H5/Zarr file with the predicted patches.
        # This method ensures efficient memory usage, as patches are individually incorporated into the H5/Zarr file in their respective
        # positions. This negates the need to store all patches simultaneously for image reconstruction. Importantly, in this
        # approach, only the 'DATA.TEST.PADDING' parameter is considered, excluding 'DATA.TEST.OVERLAP', which sufficiently
        # addresses border effect issues. If the source image is also an H5/Zarr file, it will be processed incrementally, further
        # optimizing memory usage.
        _C.TEST.BY_CHUNKS = CN()
        _C.TEST.BY_CHUNKS.ENABLE = False
        # In the process of 'TEST.BY_CHUNKS' you can enable this variable to save the reconstructed prediction as a TIF too.
        # Be aware of this option and be sure that the prediction can fit in you memory entirely, as it is needed for saving as TIF.
        _C.TEST.BY_CHUNKS.SAVE_OUT_TIF = False
        # In how many iterations the H5 writer needs to flush the data. No need to do so with Zarr files.
        _C.TEST.BY_CHUNKS.FLUSH_EACH = 100

        # Whether after reconstructing the prediction the pipeline will continue each workflow specific steps. For this process
        # the prediction image needs to be loaded into memory so be sure that it can fit in you memory. E.g. in instance
        # segmentation the instances will be created from the prediction.
        _C.TEST.BY_CHUNKS.WORKFLOW_PROCESS = CN()
        _C.TEST.BY_CHUNKS.WORKFLOW_PROCESS.ENABLE = True
        # How the workflow process is going to be done. There are two options:
        #    * 'chunk_by_chunk' : each chunk will be considered as an individual file. Select this operation if you have not enough
        #      memory to process the entire prediction image with 'entire_pred'.
        #    * 'entire_pred': the predicted image will be loaded in memory and processed entirely (be aware of your  memory budget)
        _C.TEST.BY_CHUNKS.WORKFLOW_PROCESS.TYPE = "chunk_by_chunk"
        # Enable verbosity
        _C.TEST.VERBOSE = True
        # Make test-time augmentation. Infer over 8 possible rotations for 2D img and 16 when 3D
        _C.TEST.AUGMENTATION = False
        # Select test-time augmentation mode. Options: "mean" (default), "min", "max".
        _C.TEST.AUGMENTATION_MODE = "mean"
        # Stack 2D images into a 3D image and then process it entirely instead of going image per image
        _C.TEST.ANALIZE_2D_IMGS_AS_3D_STACK = False
        # Whether to reuse the existing ones (from file) or calculate predictions using the model
        _C.TEST.REUSE_PREDICTIONS = False

        # If PROBLEM.NDIM = '2D' this can be activated to process each image entirely instead of patch by patch. Only can be done
        # if the neural network is fully convolutional. Implemented in semantic-segmentation, instance-segmentation and detection workflows.
        _C.TEST.FULL_IMG = False

        # Metrics to apply during training. Depending on the workflow different ones can be applied. If empty, some
        # default metrics will be configured automatically:
        #   * Semantic segmentation: 'iou' (called also Jaccard index)
        #   * Instance segmentation: automatically set depending on the channels selected (PROBLEM.INSTANCE_SEG.DATA_CHANNELS).
        #                            Instance metrics will be always calculated.
        #   * Detection: 'iou' (called also Jaccard index)
        #   * Denoising: 'mae', 'mse'
        #   * Super-resolution: "psnr", "mae", "mse", "ssim". Additionally, if only if PROBLEM.NDIM == '2D', these
        #                       can also be selected:  "fid", "is", "lpips"
        #   * Self-supervision: "psnr", "mae", "mse", "ssim". Additionally, if only if PROBLEM.NDIM == '2D', these
        #                       can also be selected:  "fid", "is", "lpips"
        #   * Classification: 'accuracy'. Always calculated: Confusion matrix
        #   * Image to image: "psnr", "mae", "mse", "ssim". Additionally, if only if PROBLEM.NDIM == '2D', these
        #                     can also be selected:  "fid", "is", "lpips"
        _C.TEST.METRICS = []

        ### Instance segmentation
        # Whether to calculate matching statistics (average overlap, accuracy, recall, precision, etc.)
        _C.TEST.MATCHING_STATS = True
        # Theshold of overlap to consider a TP when calculating the metrics. If more than one value is provided
        # the process is repeated with each of the threshold values
        _C.TEST.MATCHING_STATS_THS = [0.3, 0.5, 0.75]
        # Decide in which thresholds to create a colored image of the TPs, FNs and FPs
        _C.TEST.MATCHING_STATS_THS_COLORED_IMG = []

        ### Detection
        # To decide which function is going to be used to create point from probabilities. Options: ['peak_local_max', 'blob_log']
        # 'peak_local_max': https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.peak_local_max
        # 'blob_log': https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.blob_log
        _C.TEST.DET_POINT_CREATION_FUNCTION = "peak_local_max"
        # The minimal allowed distance separating peaks. To find the maximum number of peaks, use min_distance=1.
        _C.TEST.DET_PEAK_LOCAL_MAX_MIN_DISTANCE = 1
        # Whether the threshold are going to be set as automaticaly (with Otsu thresholding) or manually.
        # Options available: 'auto' or 'manual'. If this last is used TEST.DET_MIN_TH_TO_BE_PEAK needs to be set.
        _C.TEST.DET_TH_TYPE = "manual"
        # Minimun value to consider a point as a peak. Corresponds to 'threshold_abs' argument of the function
        # 'peak_local_max' of skimage.feature
        _C.TEST.DET_MIN_TH_TO_BE_PEAK = 0.2
        # Corresponds to 'exclude_border' argument of 'peak_local_max' or 'blob_log' function of skimage. If True it will exclude
        # peaks from the border of the image to avoid partial detection.
        _C.TEST.DET_EXCLUDE_BORDER = False
        # Corresponds to 'min_sigma' argument of 'blob_log' function. It is the minimum standard deviation for Gaussian kernel.
        # Keep this low to detect smaller blobs. The standard deviations of the Gaussian filter are given for each axis as a
        # sequence, or as a single number, in which case it is equal for all axes.
        _C.TEST.DET_BLOB_LOG_MIN_SIGMA = 5
        # Corresponds to 'max_sigma' argument of 'blob_log' function. It is the maximum standard deviation for Gaussian kernel.
        # Keep this high to detect larger blobs. The standard deviations of the Gaussian filter are given for each axis as a
        # sequence, or as a single number, in which case it is equal for all axes.
        _C.TEST.DET_BLOB_LOG_MAX_SIGMA = 10
        # Corresponds to 'num_sigma' argument of 'blob_log' function. The number of intermediate values of standard deviations
        # to consider between min_sigma and max_sigma.
        _C.TEST.DET_BLOB_LOG_NUM_SIGMA = 2
        # Maximum distance far away from a GT point to consider a point as a true positive
        _C.TEST.DET_TOLERANCE = 10
        # To not take into account during detection metrics calculation to those points outside the bounding box defined with
        # this variable. Order is: [z, y, x] (3D) and [y, x] (2D). For example, using an image of 10x100x200 to not take into
        # account points on the first/last slices and with a border of 15 pixel for x and y axes, this variable could be defined
        # as [1, 15, 15].
        _C.TEST.DET_IGNORE_POINTS_OUTSIDE_BOX = []

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Post-processing
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        _C.TEST.POST_PROCESSING = CN()

        # To apply median filtering to the data
        _C.TEST.POST_PROCESSING.MEDIAN_FILTER = False
        # List of median filters to apply. They are going to be applied in the list order. This can only be used in
        # 'SEMANTIC_SEG', 'INSTANCE_SEG' and 'DETECTION' workflows. There are multiple options to compose the list:
        #   * 'xy' or 'yx': to apply the filter in x and y axes together
        #   * 'zy' or 'yz': to apply the filter in y and z axes together
        #   * 'zx' or 'xz': to apply the filter in x and z axes together
        #   * 'z': to apply the filter only in z axis
        # Those filter that imply 'z' axis are going to be applied only in 3D or in 2D if TEST.ANALIZE_2D_IMGS_AS_3D_STACK is selected
        _C.TEST.POST_PROCESSING.MEDIAN_FILTER_AXIS = []
        _C.TEST.POST_PROCESSING.MEDIAN_FILTER_SIZE = []

        # Apply a binary mask to remove possible segmentation outside it (you need to provide the mask and it must
        # contain two values: '1' -> preserve the pixel ; '0' discard pixel ). A mask for each test sample must be
        # provided and it will be loaded using 'DATA.TEST.BINARY_MASKS' variable.
        _C.TEST.POST_PROCESSING.APPLY_MASK = False

        ### Instance segmentation
        # Whether to measure morphological features on each instances, i.e. 'circularity' (2D), 'elongation' (2D), 'npixels', 'area', 'diameter',
        # 'perimeter', 'sphericity' (3D)
        _C.TEST.POST_PROCESSING.MEASURE_PROPERTIES = CN()
        _C.TEST.POST_PROCESSING.MEASURE_PROPERTIES.ENABLE = False
        # Remove instances by the conditions based in each instance properties. The three variables, TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS,
        # TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.VALUES and TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.SIGNS will compose a list
        # of conditions to remove the instances. They are list of list of conditions. For instance, the conditions can be like this: [['A'], ['B','C']]. Then, if the instance satisfies
        # the first list of conditions, only 'A' in this first case (from ['A'] list), or satisfy 'B' and 'C' (from ['B','C'] list) it will be
        # removed from the image. In each sublist all the conditions must be satisfied. Available properties are: ['circularity', 'elongation',
        # 'npixels', 'area', 'diameter', 'perimeter', 'sphericity']. When this post-processing step is selected two .csv files
        # will be created, one with the properties of each instance from the original image (will be placed in PATHS.RESULT_DIR.PER_IMAGE_INSTANCES
        # path), and another with only instances that remain once this post-processing has been applied (will be placed in
        # PATHS.RESULT_DIR.PER_IMAGE_POST_PROCESSING path). In those csv files two more information columns will appear: a list of conditions
        # that each instance has satisfy or not ('Satisfied', 'No satisfied' respectively), and a comment with two possible values, 'Removed'
        # and 'Correct', telling you if the instance has been removed or not, respectively. Some of the properties follow the formulas used in
        # MorphoLibJ library for Fiji https://doi.org/10.1093/bioinformatics/btw413
        #
        # Each property descrition:
        #   * 'circularity' is defined as the ratio of area over the square of the perimeter, normalized such that the value for a disk equals
        #     one: (4 * PI * area) / (perimeter^2). Only measurable for 2D images (use sphericity for 3D images). While values of circularity
        #     range theoretically within the interval [0;1], the measurements errors of the perimeter may produce circularity values above 1
        #     (Lehmann et al., 201211 ; https://doi.org/10.1093/bioinformatics/btw413).
        #
        #   * 'elongation' is the inverse of the circularity. The values of elongation range from 1 for round particles and increase for
        #     elongated particles. Calculated as: (perimeter^2)/(4 * PI * area) . It is only measurable for 2D images.
        #
        #   * 'npixels' corresponds to the sum of pixels that compose an instance.
        #
        #   * 'area' corresponds to the number of pixels taking into account the image resolution (we call it 'area' also even in a 3D
        #     image for simplicity, but that will be the volume in that case). In the resulting statistics 'volume' will appear in that
        #     case too.
        #
        #   * 'diameter' calculated with the bounding box and by taking the maximum value of the box in x and y axes. In 3D, z axis
        #     is also taken into account. Does not take into account the image resolution.
        #
        #   * 'perimeter', in 2D, approximates the contour as a line through the centers of border pixels using a 4-connectivity. In 3D,
        #     it corresponds to the surface area calculated with diplib library 
        #     (more info here https://diplib.org/diplib-docs/features.html#shape_features_P2A).
        #
        #   * 'sphericity', in 3D, it is the ratio of the squared volume over the cube of the surface area, normalized such that the value
        #     for a ball equals one: (36 * PI)*((volume^2)/(perimeter^3)). Only measurable for 3D images (use circularity for 2D images).
        #
        # A full example of this post-processing:
        # If you want to remove those instances that have less than 100 pixels and circularity less equal to 0.7 you should
        # declare the above three variables as follows:
        #   _C.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS = [['npixels', 'circularity']]
        #   _C.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.VALUES = [[100, 0.7]]
        #   _C.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.SIGNS = [['lt', 'le']]
        # You can also concatenate more restrictions and they will be applied in order. For instance, if you want to remove those
        # instances that are bigger than an specific area, and do that before the condition described above, you can define the
        # variables this way:
        #   _C.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS = [['area'], ['npixels', 'circularity']]
        #   _C.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.VALUES = [[500], [100, 0.7]]
        #   _C.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.SIGNS = [['gt'], ['lt', 'le']]
        # This way, the instances will be removed by 'area' and then by 'npixels' and 'circularity'
        _C.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES = CN()
        # Whether to enable or not the filtering by properties
        _C.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.ENABLE = False
        # List of lists of properties to apply a filter. Available properties are: ['circularity', 'elongation', 'npixels', 'area', 'diameter',
        # 'perimeter', 'sphericity']
        _C.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS = []
        # List of ints/float that represent the values of the properties listed in TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES
        # that the instances need to satisfy to not be dropped.
        _C.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.VALUES = []
        # List of list of signs to do the comparison. Options: ['gt', 'ge', 'lt', 'le'] that corresponds to "greather than", e.g. ">",
        # "greather equal", e.g. ">=", "less than", e.g. "<", and "less equal" e.g. "<=" comparisons.
        _C.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.SIGNS = []

        # Whether to apply Voronoi using 'BC' or 'M' channels need to be present
        _C.TEST.POST_PROCESSING.VORONOI_ON_MASK = False
        # Threshold to be applied to the 'M' channel when expanding the instances with Voronoi. Need to be in [0,1] range.
        # Leave it to 0 to adjust the threhold with Otsu
        _C.TEST.POST_PROCESSING.VORONOI_TH = 0.0
        # Set it to try to repare large instances by merging their neighbors with them and removing possible central holes.
        # Its value determines which instances are going to be repared by size (number of pixels that compose the instance)
        # This option is useful when PROBLEM.INSTANCE_SEG.DATA_CHANNELS is 'BP', as multiple central seeds may appear in big
        # instances. Only works in Instance segmentation workflow.
        _C.TEST.POST_PROCESSING.REPARE_LARGE_BLOBS_SIZE = -1
        # Clear objects connected to the label image border. Only works in Instance segmentation workflow.
        _C.TEST.POST_PROCESSING.CLEAR_BORDER = False
        # Fill holes in instances. Only works in Instance segmentation workflow.
        _C.TEST.POST_PROCESSING.FILL_HOLES = False

        ### Detection
        # To remove close points to each other. This can also be set when using 'BP' channels for instance segmentation.
        _C.TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS = False
        # Distance between points to be considered the same. Only applies when TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS = True
        # This can also be set when using 'BP' channels for instance segmentation.
        _C.TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS_RADIUS = 0
        # Whether to apply a watershed to grow the points detected
        _C.TEST.POST_PROCESSING.DET_WATERSHED = False
        # Structure to dilate the initial seeds before watershed. For instance in a 3D problem: [10,10,4]
        _C.TEST.POST_PROCESSING.DET_WATERSHED_FIRST_DILATION = [-1, -1]
        # List of classes to be consider as 'donuts'. For those class points, the 'donuts' type cell means that their nucleus is
        # to big and that the seeds need to be dilated more so the watershed can grow the instances properly.
        _C.TEST.POST_PROCESSING.DET_WATERSHED_DONUTS_CLASSES = [-1]
        # Patch shape to extract all donuts type cells. It needs to be a bit greater than bigest donuts type cell so all of them can
        # be contained in this patch. This is used to analize that area for each point of class `DET_WATERSHED_DONUTS_CLASSES`.
        _C.TEST.POST_PROCESSING.DET_WATERSHED_DONUTS_PATCH = [13, 120, 120]
        # Diameter (in pixels) that a cell need to have to be considered as donuts type
        _C.TEST.POST_PROCESSING.DET_WATERSHED_DONUTS_NUCLEUS_DIAMETER = 30

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Auxiliary paths
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        _C.PATHS = CN()

        # Directories to store the results
        _C.PATHS.RESULT_DIR = CN()
        _C.PATHS.RESULT_DIR.PATH = os.path.join(job_dir, "results", job_identifier)
        _C.PATHS.RESULT_DIR.PER_IMAGE = os.path.join(_C.PATHS.RESULT_DIR.PATH, "per_image")
        _C.PATHS.RESULT_DIR.PER_IMAGE_BIN = os.path.join(_C.PATHS.RESULT_DIR.PATH, "per_image_binarized")
        _C.PATHS.RESULT_DIR.PER_IMAGE_INSTANCES = os.path.join(_C.PATHS.RESULT_DIR.PATH, "per_image_instances")
        _C.PATHS.RESULT_DIR.PER_IMAGE_POST_PROCESSING = os.path.join(
            _C.PATHS.RESULT_DIR.PATH, "per_image_post_processing"
        )
        _C.PATHS.RESULT_DIR.FULL_IMAGE = os.path.join(_C.PATHS.RESULT_DIR.PATH, "full_image")
        _C.PATHS.RESULT_DIR.FULL_IMAGE_BIN = os.path.join(_C.PATHS.RESULT_DIR.PATH, "full_image_binarized")
        _C.PATHS.RESULT_DIR.FULL_IMAGE_INSTANCES = os.path.join(_C.PATHS.RESULT_DIR.PATH, "full_image_instances")
        _C.PATHS.RESULT_DIR.FULL_IMAGE_POST_PROCESSING = os.path.join(
            _C.PATHS.RESULT_DIR.PATH, "full_image_post_processing"
        )
        _C.PATHS.RESULT_DIR.AS_3D_STACK = os.path.join(_C.PATHS.RESULT_DIR.PATH, "as_3d_stack")
        _C.PATHS.RESULT_DIR.AS_3D_STACK_BIN = os.path.join(_C.PATHS.RESULT_DIR.PATH, "as_3d_stack_binarized")
        _C.PATHS.RESULT_DIR.AS_3D_STACK_POST_PROCESSING = os.path.join(
            _C.PATHS.RESULT_DIR.PATH, "as_3d_stack_post_processing"
        )
        _C.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK = os.path.join(
            _C.PATHS.RESULT_DIR.PATH, "per_image_local_max_check"
        )
        _C.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK_POST_PROCESSING = os.path.join(
            _C.PATHS.RESULT_DIR.PATH, "per_image_local_max_check_post_processing"
        )
        _C.PATHS.RESULT_DIR.DET_ASSOC_POINTS = os.path.join(_C.PATHS.RESULT_DIR.PATH, "point_associations")
        _C.PATHS.RESULT_DIR.INST_ASSOC_POINTS = os.path.join(_C.PATHS.RESULT_DIR.PATH, "instance_associations")
        # Path to store the BMZ model created 
        _C.PATHS.BMZ_EXPORT_PATH = os.path.join(_C.PATHS.RESULT_DIR.PATH, "BMZ_files")

        # Path to store profiler files
        _C.PATHS.PROFILER = os.path.join(_C.PATHS.RESULT_DIR.PATH, "profiler")

        # Name of the folder where the charts of the loss and metrics values while training the network are stored.
        _C.PATHS.CHARTS = os.path.join(_C.PATHS.RESULT_DIR.PATH, "charts")
        # Folder where samples of DA will be stored
        _C.PATHS.DA_SAMPLES = os.path.join(_C.PATHS.RESULT_DIR.PATH, "aug")
        # Folder where generator samples (X) will be stored
        _C.PATHS.GEN_CHECKS = os.path.join(_C.PATHS.RESULT_DIR.PATH, "gen_check")
        # Folder where generator samples (Y) will be stored
        _C.PATHS.GEN_MASK_CHECKS = os.path.join(_C.PATHS.RESULT_DIR.PATH, "gen_mask_check")
        # Paths where a few samples of instance channels created will be stored just to check id there is any problem
        _C.PATHS.TRAIN_INSTANCE_CHANNELS_CHECK = os.path.join(
            _C.PATHS.RESULT_DIR.PATH,
            "train_" + "".join(_C.PROBLEM.INSTANCE_SEG.DATA_CHANNELS) + "_instance_channels",
        )
        _C.PATHS.VAL_INSTANCE_CHANNELS_CHECK = os.path.join(
            _C.PATHS.RESULT_DIR.PATH,
            "val_" + "".join(_C.PROBLEM.INSTANCE_SEG.DATA_CHANNELS) + "_instance_channels",
        )
        _C.PATHS.TEST_INSTANCE_CHANNELS_CHECK = os.path.join(
            _C.PATHS.RESULT_DIR.PATH,
            "test_" + "".join(_C.PROBLEM.INSTANCE_SEG.DATA_CHANNELS) + "_instance_channels",
        )
        # Name of the folder where weights files will be stored/loaded from.
        _C.PATHS.CHECKPOINT = os.path.join(job_dir, "checkpoints")
        # Checkpoint file to load/store the model weights
        _C.PATHS.CHECKPOINT_FILE = ""
        # Name of the folder to store the probability map to avoid recalculating it on every run
        _C.PATHS.PROB_MAP_DIR = os.path.join(job_dir, "prob_map")
        _C.PATHS.PROB_MAP_FILENAME = "prob_map.npy"
        # Watershed debugging folder
        _C.PATHS.WATERSHED_DIR = os.path.join(_C.PATHS.RESULT_DIR.PATH, "watershed")
        # Path where the images used in MAE will be saved suring inference
        _C.PATHS.MAE_OUT_DIR = os.path.join(_C.PATHS.RESULT_DIR.PATH, "MAE_checks")
        # Directory to save filtered images.
        _C.PATHS.FIL_SAMPLES_DIR = os.path.join(_C.PATHS.RESULT_DIR.PATH, "filtering_information")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Logging
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        _C.LOG = CN()
        _C.LOG.LOG_DIR = os.path.join(job_dir, "train_logs")
        _C.LOG.TENSORBOARD_LOG_DIR = os.path.join(_C.PATHS.RESULT_DIR.PATH, "tensorboard")
        _C.LOG.LOG_FILE_PREFIX = job_identifier
        _C.LOG.CHART_CREATION_FREQ = 5

        self._C = _C

    def get_cfg_defaults(self) -> CN:
        """
        Get a clone of the default configuration.

        Returns
        -------
        CN
            A cloned YACS CfgNode object with default values.
        """
        # Return a clone so that the defaults will not be altered
        # This is for the "local variable" use pattern
        return self._C.clone()

    def to_dict(self):
        """
        Convert the configuration to a Python dictionary.

        Returns
        -------
        dict
            Dictionary representation of the configuration.
        """
        return dict(self._C)

    def copy(self):
        """
        Create a deep copy of the Config object.

        Returns
        -------
        Config
            A deep copy of the current Config instance.
        """
        return copy.deepcopy(self)

    def __str__(self):
        """
        Return a string representation of the Config object.

        Returns
        -------
        str
            String representation of the configuration.
        """
        return str(self.__dict__)

    def __repr__(self):
        """
        Return a string representation of the Config object.

        Returns
        -------
        str
            String representation of the configuration.
        """
        return str(self.__dict__)

def update_dependencies(cfg) -> None:
    """
    Update dependent configuration variables after merging user config.

    This function should be called after merging a user-provided .cfg file
    to ensure that all dependent paths and variables are updated accordingly.
    That is, this function should be called after YACS's merge_from_file().

    Parameters
    ----------
    cfg : Config or CN
        The configuration object to update.
    """
    call = getattr(cfg, "_C") if bool(getattr(cfg, "_C", False)) else cfg
    # Remove possible / characters at the end of the paths
    call.DATA.TRAIN.PATH = (
        call.DATA.TRAIN.PATH if call.DATA.TRAIN.PATH[-1] != "/" else call.DATA.TRAIN.PATH[:-1]
    )
    call.DATA.TRAIN.GT_PATH = (
        call.DATA.TRAIN.GT_PATH if call.DATA.TRAIN.GT_PATH[-1] != "/" else call.DATA.TRAIN.GT_PATH[:-1]
    )
    call.DATA.VAL.PATH = (
        call.DATA.VAL.PATH if call.DATA.VAL.PATH[-1] != "/" else call.DATA.VAL.PATH[:-1]
    )
    call.DATA.VAL.GT_PATH = (
        call.DATA.VAL.GT_PATH if call.DATA.VAL.GT_PATH[-1] != "/" else call.DATA.VAL.GT_PATH[:-1]
    )
    call.DATA.TEST.PATH = (
        call.DATA.TEST.PATH if call.DATA.TEST.PATH[-1] != "/" else call.DATA.TEST.PATH[:-1]
    )
    call.DATA.TEST.GT_PATH = (
        call.DATA.TEST.GT_PATH if call.DATA.TEST.GT_PATH[-1] != "/" else call.DATA.TEST.GT_PATH[:-1]
    )
    post_dil = "".join(str(call.PROBLEM.INSTANCE_SEG.SYNAPSES.POSTSITE_DILATION)[1:-1].replace(",","")).replace(" ","_")
    post_d_dil = "".join(str(call.PROBLEM.INSTANCE_SEG.SYNAPSES.POSTSITE_DILATION_DISTANCE_CHANNELS)[1:-1].replace(",","")).replace(" ","_")
    tdata = call.DATA.TRAIN.GT_PATH if not call.DATA.TRAIN.INPUT_ZARR_MULTIPLE_DATA else call.DATA.TRAIN.PATH
    channel_suffix = "".join(call.PROBLEM.INSTANCE_SEG.DATA_CHANNELS) 
    if call.PROBLEM.INSTANCE_SEG.BORDER_EXTRA_WEIGHTS != "":
        channel_suffix += "We"
    if call.PROBLEM.INSTANCE_SEG.TYPE == "regular":
        suffix = ""
        for key, val in call.PROBLEM.INSTANCE_SEG.DATA_CHANNELS_EXTRA_OPTS[0].items():
            suffix += f"_{key}"
            if key in call.PROBLEM.INSTANCE_SEG.DATA_CHANNELS:
                for kdetail, vdetail in val.items():
                    suffix += f"-{kdetail}-{vdetail}"
        call.DATA.TRAIN.INSTANCE_CHANNELS_MASK_DIR = (
            tdata
            + "_"
            + channel_suffix
            + suffix
        )
    else: 
        call.DATA.TRAIN.INSTANCE_CHANNELS_MASK_DIR = (
            tdata
            + "_"
            + channel_suffix
            + "_"
            + post_dil
            + "_" 
            + post_d_dil
        )
    cpd = '_'.join([str(x) for x in call.PROBLEM.DETECTION.CENTRAL_POINT_DILATION])
    call.DATA.TRAIN.DETECTION_MASK_DIR = call.DATA.TRAIN.GT_PATH + "_detection_masks_" + str(cpd)
    call.DATA.TRAIN.SSL_SOURCE_DIR = call.DATA.TRAIN.PATH + "_ssl_source"
    vdata = call.DATA.VAL.GT_PATH if not call.DATA.VAL.INPUT_ZARR_MULTIPLE_DATA else call.DATA.VAL.PATH
    if call.PROBLEM.INSTANCE_SEG.TYPE == "regular":
        call.DATA.VAL.INSTANCE_CHANNELS_MASK_DIR = (
            vdata
            + "_"
            + channel_suffix
            + suffix
        )
    else: 
        call.DATA.VAL.INSTANCE_CHANNELS_MASK_DIR = (
            vdata
            + "_"
            + channel_suffix
            + "_"
            + post_dil 
            + "_" 
            + post_d_dil
        )
    
    # If value is not the default
    call.DATA.VAL.DETECTION_MASK_DIR = call.DATA.VAL.GT_PATH + "_detection_masks_" + str(cpd)
    call.DATA.VAL.SSL_SOURCE_DIR = call.DATA.VAL.PATH + "_ssl_source"
    tdata = call.DATA.TEST.GT_PATH if not call.DATA.TEST.INPUT_ZARR_MULTIPLE_DATA else call.DATA.TEST.PATH
    if call.PROBLEM.INSTANCE_SEG.TYPE == "regular":
        call.DATA.TEST.INSTANCE_CHANNELS_MASK_DIR = (
            tdata
            + "_"
            + channel_suffix
            + suffix
        )
    else: 
        call.DATA.TEST.INSTANCE_CHANNELS_MASK_DIR = (
            tdata
            + "_"
            + channel_suffix
            + "_"
            + post_dil
            + "_" 
            + post_d_dil
        )
    
    # If value is not the default
    if call.DATA.TEST.BINARY_MASKS == os.path.join("user_data", "test", "bin_mask"):
        call.DATA.TEST.BINARY_MASKS = os.path.join(call.DATA.TEST.PATH, "..", "bin_mask")
    call.DATA.TEST.DETECTION_MASK_DIR = call.DATA.TEST.GT_PATH + "_detection_masks_" + str(cpd)
    call.DATA.TEST.SSL_SOURCE_DIR = call.DATA.TEST.PATH + "_ssl_source"
    call.PATHS.TEST_FULL_GT_H5 = os.path.join(call.DATA.TEST.GT_PATH, "h5")

    call.PATHS.TRAIN_INSTANCE_CHANNELS_CHECK = os.path.join(
        call.PATHS.RESULT_DIR.PATH,
        "train_" + channel_suffix + "_instance_channels",
    )
    call.PATHS.VAL_INSTANCE_CHANNELS_CHECK = os.path.join(
        call.PATHS.RESULT_DIR.PATH,
        "val_" + channel_suffix + "_instance_channels",
    )
    call.PATHS.TEST_INSTANCE_CHANNELS_CHECK = os.path.join(
        call.PATHS.RESULT_DIR.PATH,
        "test_" + channel_suffix + "_instance_channels",
    )
