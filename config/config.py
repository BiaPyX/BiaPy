import os
from yacs.config import CfgNode as CN


class Config:
    def __init__(self, job_dir, job_identifier, dataroot):
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Config definition
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        _C = CN()


        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # System
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        _C.SYSTEM = CN()
        # Number of GPUs to use
        _C.SYSTEM.NUM_GPUS = 1
        # Number of CPUs to use
        _C.SYSTEM.NUM_CPUS = 1
        # Math seed
        _C.SYSTEM.SEED = 0


        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Problem specification
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        _C.PROBLEM = CN()
        # Possible options: 'CLASSIFICATION', 'SEMANTIC_SEG' and 'INSTANCE_SEG'
        _C.PROBLEM.TYPE = 'SEMANTIC_SEG'
        # Possible options: '2D' and '3D'
        _C.PROBLEM.NDIM = '2D'


        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Dataset
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        _C.DATA = CN()
        _C.DATA.ROOT_DIR = dataroot
        _C.DATA.CHECK_GENERATORS = False

        # Possible options: 'B', 'BC' and 'BCD'. This variable determines how many channels are going to be used to train
        # the model.
        # 'B' stands for 'Binary segmentation', which is the default setting and means that only binary
        # segmentation mask are going to be used. This setting id the default and should be used when
        # _C.PROBLEM.TYPE = 'SEMANTIC_SEG'.
        # 'BC' stands for 'Binary segmentation' + 'Contour'. Here the library expects mask to be instance segmenation mask
        # so the binary and contour channel are created from that data automatically. This setting should be used when
        # _C.PROBLEM.TYPE = 'INSTANCE_SEG'
        # 'BCD' stands for 'Binary segmentation' + 'Contour' + 'Distance'. As 'BC' but creating an additional channel
        # that represents the distance of the pixels to the border. This setting should be used when
        # _C.PROBLEM.TYPE = 'INSTANCE_SEG'
        # 'Dv2' channels is a updated version of 'D' channel calculating background distance as well.
        _C.DATA.CHANNELS = 'B'
        # Weights to be applied to segmentation (binary and contours) and to distances respectively. E.g. (1, 0.2), 1
        # should be multipled by BCE for the first two channels and 0.2 to MSE for the last channel.
        _C.DATA.CHANNEL_WEIGHTS = (1, 0.2)
        # Contour creation mode. Corresponds to 'fb_mode' arg of find_boundaries function from ``scikit-image``. More
        # info in: https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.find_boundaries
        _C.DATA.CONTOUR_MODE = "thick"

        # To convert the model predictions, which are between 0 and 1 range, into instances with marked controlled
        # watershed (MW) a few thresholds need to be set. There can be up to three channels, as explained above and
        # based on 'DATA.CHANNELS' value. Each threshold is related to one of these channels. See the details in
        # bcd_watershed() and bc_watershed() functions:
        # https://github.com/danifranco/EM_Image_Segmentation/blob/a1c46e6b8afaf577794aff9c30b52748490f147d/data/post_processing/post_processing.py#L172
        #
        # This variables are only used when _C.PROBLEM.TYPE = 'INSTANCE_SEG
        # TH1 controls channel 'B' in the creation of the MW seeds
        _C.DATA.MW_TH1 = 0.2
        # TH2 controls channel 'C' in the creation of the MW seeds
        _C.DATA.MW_TH2 = 0.1
        # TH3 acts over the channel 'B' and is used to limit how much the seeds can be grow
        _C.DATA.MW_TH3 = 0.3
        # TH4 controls channel 'D' in the creation of the MW seeds
        _C.DATA.MW_TH4 = 1.2
        # TH5 acts over the channel 'D' and is used to limit how much the seeds can be grow
        _C.DATA.MW_TH5 = 1.5
        # Size of small objects to be removed after doing watershed
        _C.DATA.REMOVE_SMALL_OBJ = 30
        # Wheter to remove objects before watershed or after it
        _C.DATA.REMOVE_BEFORE_MW = True
        # Wheter to find an optimum value for each threshold with the validation data. If True the previous MW_TH*
        # variables will be replaced by the optimum values found
        _C.DATA.MW_OPTIMIZE_THS = False
        # Wheter to save watershed check files
        _C.DATA.CHECK_MW = True

        # Wheter to reshape de dimensions that does not satisfy the pathc shape selected by padding it with reflect.
        _C.DATA.REFLECT_TO_COMPLETE_SHAPE = False

        # Train
        _C.DATA.TRAIN = CN()
        _C.DATA.TRAIN.IN_MEMORY = True
        _C.DATA.TRAIN.PATH = os.path.join(_C.DATA.ROOT_DIR, 'train', 'x')
        _C.DATA.TRAIN.MASK_PATH = os.path.join(_C.DATA.ROOT_DIR, 'train', 'y')
        # File to load/save data prepared with the appropiate channels in a instance segmentation problem.
        # E.g. _C.PROBLEM.TYPE ='INSTANCE_SEG' and _C.DATA.CHANNELS != 'B'
        _C.DATA.TRAIN.INSTANCE_CHANNELS_DIR = os.path.join(_C.DATA.ROOT_DIR, 'train', 'x_'+_C.DATA.CHANNELS)
        _C.DATA.TRAIN.INSTANCE_CHANNELS_MASK_DIR = os.path.join(_C.DATA.ROOT_DIR, 'train', 'y_'+_C.DATA.CHANNELS)
        # Extra train data generation: number of times to duplicate the train data. Useful when
        # _C.DATA.EXTRACT_RANDOM_PATCH=True is made, as more original train data can be cover on each epoch
        _C.DATA.TRAIN.REPLICATE = 0
        # Percentage of overlap in (x,y)/(x,y,z) when cropping train. Set to 0 to calculate  the minimun overlap.
        # _C.PROBLEM.NDIM='2D' -> _C.DATA.TRAIN.OVERLAP=(0,0). _C.PROBLEM.NDIM ='3D' -> _C.DATA.TRAIN.OVERLAP=(0,0,0)
        _C.DATA.TRAIN.OVERLAP = (0,0)
        # Padding to be done in (x,y)/(x,y,z) when reconstructing train data. Useful to avoid patch 'border effect'.
        _C.DATA.TRAIN.PADDING = (0,0)
        _C.DATA.TRAIN.CHECK_CROP = True # Used when _C.DATA.IN_MEMORY=True

        # Test
        _C.DATA.TEST = CN()
        _C.DATA.TEST.IN_MEMORY = False
        # Wheter to load ground truth (GT)
        _C.DATA.TEST.LOAD_GT = False
        # Wheter to use validation data as test instead of trying to load test from _C.DATA.TEST.PATH and
        # _C.DATA.TEST.MASK_PATH. Currently only used if _C.PROBLEM.TYPE == 'CLASSIFICATION'
        _C.DATA.TEST.USE_VAL_AS_TEST = False
        # Path to load the test data from. Not used when _C.DATA.TEST.USE_VAL == True
        _C.DATA.TEST.PATH = os.path.join(_C.DATA.ROOT_DIR, 'test', 'x')
        # Path to load the test data masks from. Not used when _C.DATA.TEST.USE_VAL == True
        _C.DATA.TEST.MASK_PATH = os.path.join(_C.DATA.ROOT_DIR, 'test', 'y')
        # File to load/save data prepared with the appropiate channels in a instance segmentation problem.
        # E.g. _C.PROBLEM.TYPE ='INSTANCE_SEG' and _C.DATA.CHANNELS != 'B'
        _C.DATA.TEST.INSTANCE_CHANNELS_DIR = os.path.join(_C.DATA.ROOT_DIR, 'test', 'x_'+_C.DATA.CHANNELS)
        _C.DATA.TEST.INSTANCE_CHANNELS_MASK_DIR = os.path.join(_C.DATA.ROOT_DIR, 'test', 'y_'+_C.DATA.CHANNELS)
        # Percentage of overlap in (x,y)/(x,y,z) when cropping test. Set to 0 to calculate  the minimun overlap.
        # _C.PROBLEM.NDIM='2D' -> _C.DATA.TEST.OVERLAP=(0,0). _C.PROBLEM.NDIM ='3D' -> _C.DATA.TEST.OVERLAP=(0,0,0)
        _C.DATA.TEST.OVERLAP = (0,0)
        # Padding to be done in (x,y)/(x,y,z) when reconstructing test data. Useful to avoid patch 'border effect'
        _C.DATA.TEST.PADDING = (0,0)
        # Wheter to use median values to fill padded pixels or zeros
        _C.DATA.TEST.MEDIAN_PADDING = False
        # Directory where binary masks to apply to resulting images should be. Used when _C.TEST.APPLY_MASK  == True
        _C.DATA.TEST.BINARY_MASKS = os.path.join(_C.DATA.ROOT_DIR, 'test', 'bin_mask')

        # Validation
        _C.DATA.VAL = CN()
        # Wheter to create validation data from training set or read it from a directory
        _C.DATA.VAL.FROM_TRAIN = True
        # Use a cross validation strategy instead of just split the train data in two. Currently only used if
        # _C.PROBLEM.TYPE == 'CLASSIFICATION'
        _C.DATA.VAL.CROSS_VAL = False
        # Number of folds. Used when _C.DATA.VAL.CROSS_VAL == True
        _C.DATA.VAL.CROSS_VAL_NFOLD = 5
        # Number of the fold to choose as validation. Used when _C.DATA.VAL.CROSS_VAL == True
        _C.DATA.VAL.CROSS_VAL_FOLD = 1
        # Percentage of the training data used as validation. Used when _C.DATA.VAL.FROM_TRAIN = True and _C.DATA.VAL.CROSS_VAL == False
        _C.DATA.VAL.SPLIT_TRAIN = 0.0
        # Create the validation data with random images of the training data. Used when _C.DATA.VAL.FROM_TRAIN = True
        _C.DATA.VAL.RANDOM = True
        # Used when _C.DATA.VAL.FROM_TRAIN = False, as DATA.VAL.FROM_TRAIN = True always implies DATA.VAL.IN_MEMORY = True
        _C.DATA.VAL.IN_MEMORY = True
        # Path to the validation data. Used when _C.DATA.VAL.FROM_TRAIN = False
        _C.DATA.VAL.PATH = os.path.join(_C.DATA.ROOT_DIR, 'val', 'x')
        # Path to the validation data mask. Used when _C.DATA.VAL.FROM_TRAIN = False
        _C.DATA.VAL.MASK_PATH = os.path.join(_C.DATA.ROOT_DIR, 'val', 'y')
        # File to load/save data prepared with the appropiate channels in a instance segmentation problem.
        # E.g. _C.PROBLEM.TYPE ='INSTANCE_SEG' and _C.DATA.CHANNELS != 'B'
        _C.DATA.VAL.INSTANCE_CHANNELS_DIR = os.path.join(_C.DATA.ROOT_DIR, 'val', 'x_'+_C.DATA.CHANNELS)
        _C.DATA.VAL.INSTANCE_CHANNELS_MASK_DIR = os.path.join(_C.DATA.ROOT_DIR, 'val', 'y_'+_C.DATA.CHANNELS)
        # Percentage of overlap in (x,y)/(x,y,z) when cropping validation. Set to 0 to calculate  the minimun overlap.
        # _C.PROBLEM.NDIM='2D' -> _C.DATA.VAL.OVERLAP=(0,0). _C.PROBLEM.NDIM ='3D' -> _C.DATA.VAL.OVERLAP=(0,0,0)
        _C.DATA.VAL.OVERLAP = (0,0)
        # Padding to be done in (x,y)/(x,y,z) when reconstructing validation data. Useful to avoid patch 'border effect'
        _C.DATA.VAL.PADDING = (0,0)
        # Wheter to use median values to fill padded pixels or zeros
        _C.DATA.VAL.MEDIAN_PADDING = False
        # Directory where validation binary masks should be located. This binary mask will be applied only when MW_TH*
        # optimized values are find, that is, when _C.DATA.MW_OPTIMIZE_THS = True and _C.TEST.APPLY_MASK = True
        _C.DATA.VAL.BINARY_MASKS = os.path.join(_C.DATA.ROOT_DIR, 'val', 'bin_mask')

        # _C.PROBLEM.NDIM='2D' -> _C.DATA.PATCH_SIZE=(x,y,c) ; _C.PROBLEM.NDIM='3D' -> _C.DATA.PATCH_SIZE=(x,y,z,c)
        _C.DATA.PATCH_SIZE = (256, 256, 1)

        # Extract random patches during data augmentation (DA)
        _C.DATA.EXTRACT_RANDOM_PATCH = False
        # Calculate probability map to make random subvolumes to be extracted with high probability of having an object
        # on the middle of it. Useful to avoid extracting a subvolume which less foreground class information.
        _C.DATA.PROBABILITY_MAP = False # Used when _C.DATA.EXTRACT_RANDOM_PATCH=True
        _C.DATA.W_FOREGROUND = 0.94 # Used when _C.DATA.PROBABILITY_MAP=True
        _C.DATA.W_BACKGROUND = 0.06 # Used when _C.DATA.PROBABILITY_MAP=True


        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Data augmentation (DA)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        _C.AUGMENTOR = CN()
        # Flag to activate DA
        _C.AUGMENTOR.ENABLE = False
        # Probability of each transformation
        _C.AUGMENTOR.DA_PROB = 0.5
        # Create samples of the DA made. Useful to check the output images made.
        _C.AUGMENTOR.AUG_SAMPLES = True
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
        # Shear range
        _C.AUGMENTOR.SHEAR_RANGE = (-20, 20)
        # Apply zoom to images
        _C.AUGMENTOR.ZOOM = False
        # Zoom range
        _C.AUGMENTOR.ZOOM_RANGE = (0.8, 1.2)
        # Apply shift
        _C.AUGMENTOR.SHIFT = False
        # Shift range
        _C.AUGMENTOR.SHIFT_RANGE = (0.1, 0.2)
        # How to fill up the new values created with affine transformations (rotations, shear, shift and zoom).
        # Same meaning as in skimage (and numpy.pad()): 'constant', 'edge', 'symmetric', 'reflect' and 'wrap'
        _C.AUGMENTOR.AFFINE_MODE = 'constant'
        # Make vertical flips
        _C.AUGMENTOR.VFLIP = False
        # Make horizontal flips
        _C.AUGMENTOR.HFLIP = False
        # Make z-axis flips
        _C.AUGMENTOR.ZFLIP = False
        # Elastic transformations
        _C.AUGMENTOR.ELASTIC = False
        # Strength of the distortion field
        _C.AUGMENTOR.E_ALPHA = (12, 16)
        # Standard deviation of the gaussian kernel used to smooth the distortion fields
        _C.AUGMENTOR.E_SIGMA = 4
        # Parameter that defines the handling of newly created pixels with the elastic transformation
        _C.AUGMENTOR.E_MODE = 'constant'
        # Gaussian blur
        _C.AUGMENTOR.G_BLUR = False
        # Standard deviation of the gaussian kernel
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
        # Strength of the brightness range, with valid values being 0 <= brightness_factor <= 1
        _C.AUGMENTOR.BRIGHTNESS_FACTOR = (0.1, 0.3)
        # If apply same contrast to the entire image or select one for each slice. For 2D does not matter but yes for
        # 3D images. Possible values: '2D' or '3D'. Used when '_C.PROBLEM.NDIM' = '3D'.
        _C.AUGMENTOR.BRIGHTNESS_MODE = '3D'
        # To apply contrast changes to images
        _C.AUGMENTOR.CONTRAST = False
        # Strength of the contrast change range, with valid values being 0 <= contrast_factor <= 1
        _C.AUGMENTOR.CONTRAST_FACTOR = (0.1, 0.3)
        # If apply same contrast to the entire image or select one for each slice. For 2D does not matter but yes for
        # 3D images. Possible values: '2D' or '3D'. Used when '_C.PROBLEM.NDIM' = '3D'.
        _C.AUGMENTOR.CONTRAST_MODE = '3D'
        # To apply brightness changes to images based on Pytorch Connectomics library.
        _C.AUGMENTOR.BRIGHTNESS_EM = False
        # Strength of the brightness range, with valid values being 0 <= brightness_factor <= 1
        _C.AUGMENTOR.BRIGHTNESS_EM_FACTOR = (0.1, 0.3)
        # If apply same contrast to the entire image or select one for each slice. For 2D does not matter but yes for
        # 3D images. Possible values: '2D' or '3D'. Used when '_C.PROBLEM.NDIM' = '3D'.
        _C.AUGMENTOR.BRIGHTNESS_EM_MODE = '3D'
        # To apply contrast changes to images based on Pytorch Connectomics library.
        _C.AUGMENTOR.CONTRAST_EM = False
        # Strength of the contrast change range, with valid values being 0 <= contrast_em_factor <= 1
        _C.AUGMENTOR.CONTRAST_EM_FACTOR = (0.1, 0.3)
        # If apply same contrast to the entire image or select one for each slice. For 2D does not matter but yes for
        # 3D images. Possible values: '2D' or '3D'. Used when '_C.PROBLEM.NDIM' = '3D'.
        _C.AUGMENTOR.CONTRAST_EM_MODE = '3D'
        # Set a certain fraction of pixels in images to zero (not get confused with the dropout concept of neural networks)
        _C.AUGMENTOR.DROPOUT = False
        # Range to take the probability to drop a pixel
        _C.AUGMENTOR.DROP_RANGE = (0, 0.2)
        # To fill one or more rectangular areas in an image using a fill mode
        _C.AUGMENTOR.CUTOUT = False
        # Range of number of areas to fill the image with
        _C.AUGMENTOR.COUT_NB_ITERATIONS = (1, 3)
        # Size of the areas in % of the corresponding image size
        _C.AUGMENTOR.COUT_SIZE = (0.05, 0.3)
        # Value to fill the area of cutout
        _C.AUGMENTOR.COUT_CVAL = 0
        # Apply cutout to the segmentation mask
        _C.AUGMENTOR.COUT_APPLY_TO_MASK = False
        # To apply cutblur operation
        _C.AUGMENTOR.CUTBLUR = False
        # Size of the region to apply cutblur
        _C.AUGMENTOR.CBLUR_SIZE = (0.2, 0.4)
        # Range of the downsampling to be made in cutblur
        _C.AUGMENTOR.CBLUR_DOWN_RANGE = (2, 8)
        # Wheter to apply cut-and-paste just LR into HR image. If False, HR to LR will  be applied also (see Figure 1
        # of the paper https://arxiv.org/pdf/2004.00448.pdf)
        _C.AUGMENTOR.CBLUR_INSIDE = True
        # Apply cutmix operation
        _C.AUGMENTOR.CUTMIX = False
        # Size of the region to apply cutmix
        _C.AUGMENTOR.CMIX_SIZE = (0.2, 0.4)
        # Apply noise to a region of the image
        _C.AUGMENTOR.CUTNOISE = False
        # Scale of the random noise
        _C.AUGMENTOR.CNOISE_SCALE = (0.1, 0.2)
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
        _C.AUGMENTOR.MISSING_PARTS = False
        # Iterations to dilate the missing line with
        _C.AUGMENTOR.MISSP_ITERATIONS = (10, 30)
        # Convert images in grasycale gradually based on '_C.AUGMENTOR.GRAY_RANGE'
        _C.AUGMENTOR.GRAYSCALE = False
        # Shuffle channels of the images
        _C.AUGMENTOR.CHANNEL_SHUFFLE = False


        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Model definition
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        _C.MODEL = CN()
        # Architecture of the network. Possible values are: 'unet', 'resunet', 'attetion_unet'
        _C.MODEL.ARCHITECTURE = 'unet'
        # Number of feature maps on each level of the network.
        _C.MODEL.FEATURE_MAPS = [16, 32, 64, 128, 256]
        # Depth of the network. Only used when MODEL.ARCHITECTURE = 'tiramisu'. For the rest options it is inferred.
        _C.MODEL.DEPTH = 3
        # To activate the Spatial Dropout instead of use the "normal" dropout layer
        _C.MODEL.SPATIAL_DROPOUT = False
        # Values to make the dropout with. Set to 0 to prevent dropout
        _C.MODEL.DROPOUT_VALUES = [0.1, 0.1, 0.2, 0.2, 0.3]
        # To active batch normalization
        _C.MODEL.BATCH_NORMALIZATION = False
        # Kernel type to use on convolution layers
        _C.MODEL.KERNEL_INIT = 'he_normal'
        # Activation function to use
        _C.MODEL.ACTIVATION = 'elu'
        # Number of classes. If the value > 1 a categorical problem is assumed so softmax/argmax functions may be used.
        _C.MODEL.N_CLASSES = 1
        # Number of output channels. This is not the same as '_C.MODEL.N_CLASSES' because here a categorical problem is
        # not considerer. It defines just a model with more than one output. Can only be used with
        # '_C.MODEL.ARCHITECTURE' in ['unet', 'resunet', 'seunet', 'attention_unet'] and with '_C.MODEL.N_CLASSES' = 1
        _C.MODEL.OUT_CHANNELS = 1
        # Downsampling to be made in Z. This value will be the third integer of the MaxPooling operation. When facing
        # anysotropic datasets set it to get better performance
        _C.MODEL.Z_DOWN = 1
        # Checkpoint: set to True to load previous training weigths (needed for inference or to make fine-tunning)
        _C.MODEL.LOAD_CHECKPOINT = False


        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loss
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        _C.LOSS = CN()
        # Loss type, two options: "CE" -> cross entropy ; "W_CE_DICE", CE and Dice (with a weight term on each one
        # (that must sum 1) to calculate the total loss value.
        _C.LOSS.TYPE = 'CE'


        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Training phase
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        _C.TRAIN = CN()
        _C.TRAIN.ENABLE = False
        # Optimizer to use. Possible values: "SGD" or "ADAM"
        _C.TRAIN.OPTIMIZER = 'SGD'
        _C.TRAIN.LR = 1.E-4
        _C.TRAIN.BATCH_SIZE = 2
        # Number of epochs to train the model
        _C.TRAIN.EPOCHS = 360
        _C.TRAIN.PATIENCE = 50

        # LR Scheduler
        _C.TRAIN.LR_SCHEDULER = CN()
        _C.TRAIN.LR_SCHEDULER.ENABLE = False
        _C.TRAIN.LR_SCHEDULER.NAME = '' # Possible options: 'cosine', 'reduceonplateau'

        # Callbacks
        # To determine which value monitor to stop the training
        _C.TRAIN.EARLYSTOPPING_MONITOR = 'val_loss'
        # To determine which value monitor to consider which epoch consider the best to save
        _C.TRAIN.CHECKPOINT_MONITOR = 'val_loss'

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Inference phase
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        _C.TEST = CN()
        _C.TEST.ENABLE = False
        # Enable verbosity
        _C.TEST.VERBOSE = True
        # Make test-time augmentation. Infer over 8 possible rotations for 2D img and 16 when 3D
        _C.TEST.AUGMENTATION = False
        # Wheter to evaluate or not
        _C.TEST.EVALUATE = True
        # Apply a binary mask to remove possible segmentation outside it
        _C.TEST.APPLY_MASK = False
        # Whether to apply Voronoi using 'B' or 'BC' channels
        _C.TEST.VORONOI_ON_MASK = False
        # Wheter to calculate mAP
        _C.TEST.MAP = False # Only applies when _C.TEST.STATS.MERGE_PATCHES = True
        # Wheter to calculate matching statistics (average overlap, accuracy, recall, precision, etc.)
        _C.TEST.MATCHING_STATS = False
        _C.TEST.MATCHING_STATS_THS = [0.3, 0.5, 0.75]
        _C.TEST.MATCHING_VJI_PAI = False

        # Wheter to return local maximum coords. Only applies when _C.PROBLEM.TYPE = 'DETECTION'
        _C.TEST.DET_LOCAL_MAX_COORDS = False
        # Maximum distance to not consider a point as a true positive. Only applies when _C.PROBLEM.TYPE = 'DETECTION'.
        _C.TEST.DET_TOLERANCE = 10
        # Weights to be multiply by each axis. Useful when dealing with anysotropic data to reduce the distance value
        # on the axis with less resolution. Need to be provided in (x,y,z) order. Only applies when _C.PROBLEM.TYPE = 'DETECTION'.
        _C.TEST.DET_RES_ADJUSTMENT = (1,1,1)

        _C.TEST.STATS = CN()
        _C.TEST.STATS.PER_PATCH = False
        _C.TEST.STATS.MERGE_PATCHES = False # Only used when _C.TEST.STATS.PER_PATCH = True
        _C.TEST.STATS.FULL_IMG = True # Only when if PROBLEM.NDIM = '2D' as 3D images are huge for the GPU

        # When PROBLEM.NDIM = '2D' only applies when _C.TEST.STATS.FULL_IMG = True, if PROBLEM.NDIM = '3D' is applied
        # when _C.TEST.STATS.MERGE_PATCHES = True
        _C.TEST.POST_PROCESSING = CN()
        _C.TEST.POST_PROCESSING.BLENDING = False
        _C.TEST.POST_PROCESSING.YZ_FILTERING = False
        _C.TEST.POST_PROCESSING.YZ_FILTERING_SIZE = 5
        _C.TEST.POST_PROCESSING.Z_FILTERING = False
        _C.TEST.POST_PROCESSING.Z_FILTERING_SIZE = 5


        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Auxiliary paths
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        _C.PATHS = CN()

        ### mAP calculation options
        # Do not forgive to clone the repo:
        #       git clone https://github.com/danifranco/mAP_3Dvolume.git
        #
        # Change the branch:
        #       git checkout grand-challenge
        # Folder where the mAP code should be placed
        _C.PATHS.MAP_CODE_DIR = ''
        # Path to the GT h5 files to calculate the mAP
        _C.PATHS.TEST_FULL_GT_H5 = os.path.join(_C.DATA.TEST.MASK_PATH, 'h5')

        # Directories to store the results
        _C.PATHS.RESULT_DIR = CN()
        _C.PATHS.RESULT_DIR.PATH = os.path.join(job_dir, 'results', job_identifier)
        _C.PATHS.RESULT_DIR.PER_IMAGE = os.path.join(_C.PATHS.RESULT_DIR.PATH, 'per_image')
        _C.PATHS.RESULT_DIR.PER_IMAGE_INSTANCES = os.path.join(_C.PATHS.RESULT_DIR.PATH, 'per_image_instances')
        _C.PATHS.RESULT_DIR.PER_IMAGE_INST_VORONOI = os.path.join(_C.PATHS.RESULT_DIR.PATH, 'per_image_instances_voronoi')
        _C.PATHS.RESULT_DIR.PER_IMAGE_POST_PROCESSING = os.path.join(_C.PATHS.RESULT_DIR.PATH, 'per_image_post_processing')
        _C.PATHS.RESULT_DIR.FULL_IMAGE = os.path.join(_C.PATHS.RESULT_DIR.PATH, 'full_image')
        _C.PATHS.RESULT_DIR.FULL_POST_PROCESSING = os.path.join(_C.PATHS.RESULT_DIR.PATH, 'full_post_processing')
        _C.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK = os.path.join(_C.PATHS.RESULT_DIR.PATH, 'per_image_local_max_check')

        # Name of the folder where the charts of the loss and metrics values while training the network are stored.
        # Additionally, MW_TH* variable charts are stored if _C.DATA.MW_OPTIMIZE_THS = True
        _C.PATHS.CHARTS = os.path.join(_C.PATHS.RESULT_DIR.PATH, 'charts')
        # Directory where weight maps will be stored
        _C.PATHS.LOSS_WEIGHTS = os.path.join(_C.PATHS.RESULT_DIR.PATH, 'loss_weights')
        # Folder where samples of DA will be stored
        _C.PATHS.DA_SAMPLES = os.path.join(_C.PATHS.RESULT_DIR.PATH, 'aug')
        # Folder where crop samples will be stored
        _C.PATHS.CROP_CHECKS = os.path.join(_C.PATHS.RESULT_DIR.PATH, 'check_crop')
        # Folder where generator samples (X) will be stored
        _C.PATHS.GEN_CHECKS = os.path.join(_C.PATHS.RESULT_DIR.PATH, 'gen_check')
        # Folder where generator samples (Y) will be stored
        _C.PATHS.GEN_MASK_CHECKS = os.path.join(_C.PATHS.RESULT_DIR.PATH, 'gen_mask_check')
        # Paths where a few samples of instance channels created will be stored just to check id there is any problem
        _C.PATHS.TRAIN_INSTANCE_CHANNELS_CHECK = os.path.join(_C.PATHS.RESULT_DIR.PATH, 'train_instance_channels')
        _C.PATHS.VAL_INSTANCE_CHANNELS_CHECK = os.path.join(_C.PATHS.RESULT_DIR.PATH, 'val_instance_channels')
        _C.PATHS.TEST_INSTANCE_CHANNELS_CHECK = os.path.join(_C.PATHS.RESULT_DIR.PATH, 'test_instance_channels')
        # Name of the folder where weights files will be stored/loaded from.
        _C.PATHS.CHECKPOINT = os.path.join(job_dir, 'h5_files')
        # Checkpoint file to load/store the model weights
        _C.PATHS.CHECKPOINT_FILE = os.path.join(_C.PATHS.CHECKPOINT, 'model_weights_' + job_identifier + '.h5')
        # Name of the folder to store the probability map to avoid recalculating it on every run
        _C.PATHS.PROB_MAP_DIR = os.path.join(job_dir, 'prob_map')
        _C.PATHS.PROB_MAP_FILENAME = 'prob_map.npy'
        # Watershed dubgging folder
        _C.PATHS.WATERSHED_DIR = os.path.join(_C.PATHS.RESULT_DIR.PATH, 'watershed')
        # To store h5 files needed for the mAP calculation
        _C.PATHS.MAP_H5_DIR = os.path.join(_C.PATHS.RESULT_DIR.PATH, 'mAP_h5_files')

        self._C = _C

    def get_cfg_defaults(self):
        """Get a yacs CfgNode object with default values for my_project."""
        # Return a clone so that the defaults will not be altered
        # This is for the "local variable" use pattern
        return self._C.clone()

    def update_dependencies(self):
        """Update some variables that depend of changes made after merge the .cfg file provide by the user. That is,
           this function should be called after YACS's merge_from_file().
        """
        self._C.DATA.TRAIN.INSTANCE_CHANNELS_DIR = self._C.DATA.TRAIN.PATH+'_'+self._C.DATA.CHANNELS+'_'+self._C.DATA.CONTOUR_MODE
        self._C.DATA.TRAIN.INSTANCE_CHANNELS_MASK_DIR = self._C.DATA.TRAIN.MASK_PATH+'_'+self._C.DATA.CHANNELS+'_'+self._C.DATA.CONTOUR_MODE
        self._C.DATA.VAL.INSTANCE_CHANNELS_DIR = self._C.DATA.VAL.PATH+'_'+self._C.DATA.CHANNELS+'_'+self._C.DATA.CONTOUR_MODE
        self._C.DATA.VAL.INSTANCE_CHANNELS_MASK_DIR = self._C.DATA.VAL.MASK_PATH+'_'+self._C.DATA.CHANNELS+'_'+self._C.DATA.CONTOUR_MODE
        self._C.DATA.VAL.BINARY_MASKS = os.path.join(self._C.DATA.VAL.PATH, '..', 'bin_mask')
        self._C.DATA.TEST.INSTANCE_CHANNELS_DIR = self._C.DATA.TEST.PATH+'_'+self._C.DATA.CHANNELS+'_'+self._C.DATA.CONTOUR_MODE
        self._C.DATA.TEST.INSTANCE_CHANNELS_MASK_DIR = self._C.DATA.TEST.MASK_PATH+'_'+self._C.DATA.CHANNELS+'_'+self._C.DATA.CONTOUR_MODE
        self._C.DATA.TEST.BINARY_MASKS = os.path.join(self._C.DATA.TEST.PATH, '..', 'bin_mask')
        self._C.PATHS.TEST_FULL_GT_H5 = os.path.join(self._C.DATA.TEST.MASK_PATH, 'h5')
