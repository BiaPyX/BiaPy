import os
from yacs.config import CfgNode as CN


class Config:
    def __init__(self, job_dir, job_identifier):
        
        if "/" in job_identifier:
            raise ValueError("Job name can not contain / character. Provided: {}".format(job_identifier))

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Config definition
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        _C = CN()


        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # System
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        _C.SYSTEM = CN()
        # Number of CPUs to use
        _C.SYSTEM.NUM_CPUS = 10
        # Do not set it as its value will be calculated based in --gpu input arg
        _C.SYSTEM.NUM_GPUS = 0

        # Math seed
        _C.SYSTEM.SEED = 0
        # Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
        _C.SYSTEM.PIN_MEM = True

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Problem specification
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        _C.PROBLEM = CN()
        # Possible options: 'SEMANTIC_SEG', 'INSTANCE_SEG', 'DETECTION', 'DENOISING', 'SUPER_RESOLUTION', 
        # 'SELF_SUPERVISED' and 'CLASSIFICATION'
        _C.PROBLEM.TYPE = 'SEMANTIC_SEG'
        # Possible options: '2D' and '3D'
        _C.PROBLEM.NDIM = '2D'

        ### INSTANCE_SEG
        _C.PROBLEM.SEMANTIC_SEG = CN()
        # Class id to ignore when MODEL.N_CLASSES > 2 
        _C.PROBLEM.SEMANTIC_SEG.IGNORE_CLASS_ID = 0

        ### INSTANCE_SEG
        _C.PROBLEM.INSTANCE_SEG = CN()
        # Possible options: 'BC', 'BP', 'BD', 'BCM', 'BCD', 'BCDv2', 'Dv2' and 'BDv2'. This variable determines the channels to be created 
        # based on input instance masks. These option are composed from these individual options:
        #   - 'B' stands for 'Binary segmentation', containing each instance region without the contour. 
        #   - 'C' stands for 'Contour', containing each instance contour. 
        #   - 'D' stands for 'Distance', each pixel containing the distance of it to the instance contour. 
        #   - 'M' stands for 'Mask', contains the B and the C channels, i.e. the foreground mask. 
        #     Is simply achieved by binarizing input instance masks. 
        #   - 'Dv2' stands for 'Distance V2', which is an updated version of 'D' channel calculating background distance as well.
        #   - 'P' stands for 'Points' and contains the central points of an instance (as in Detection workflow) 
        _C.PROBLEM.INSTANCE_SEG.DATA_CHANNELS = 'BC'
        # Whether to mask the distance channel to only calculate the loss in those regions where the binary mask
        # defined by B channel is present
        _C.PROBLEM.INSTANCE_SEG.DISTANCE_CHANNEL_MASK = True

        # Weights to be applied to the channels.
        _C.PROBLEM.INSTANCE_SEG.DATA_CHANNEL_WEIGHTS = (1, 1)
        # Contour creation mode. Corresponds to 'mode' arg of find_boundaries function from ``scikit-image``. More
        # info in: https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.find_boundaries.
        # It can be also set as "dense", to label as contour every pixel that is not in ``B`` channel. 
        _C.PROBLEM.INSTANCE_SEG.DATA_CONTOUR_MODE = "thick" 
        # Whether if the threshold are going to be set as automaticaly (with Otsu thresholding) or manually. 
        # Options available: 'auto' or 'manual'. If this last is used PROBLEM.INSTANCE_SEG.DATA_MW_TH_* need to be set.
        # In case 'auto' was selected you will still need to set 
        _C.PROBLEM.INSTANCE_SEG.DATA_MW_TH_TYPE = "auto" 

        # To convert the model predictions, which are between 0 and 1 range, into instances with marked controlled
        # watershed (MW) a few thresholds need to be set. There can be up to three channels, as explained above and
        # based on 'PROBLEM.INSTANCE_SEG.DATA_CHANNELS' value. Each threshold is related to one of these channels. See the details in
        # https://biapy.readthedocs.io/en/latest/workflows/instance_segmentation.html#problem-resolution
        #
        # This variables are only used when _C.PROBLEM.TYPE = 'INSTANCE_SEG
        # TH_BINARY_MASK controls channel 'B' in the creation of the MW seeds
        _C.PROBLEM.INSTANCE_SEG.DATA_MW_TH_BINARY_MASK = 0.5
        # TH_CONTOUR controls channel 'C' in the creation of the MW seeds
        _C.PROBLEM.INSTANCE_SEG.DATA_MW_TH_CONTOUR = 0.1
        # TH_FOREGROUND acts over the channel 'B' and is used to limit how much the seeds can be grow
        _C.PROBLEM.INSTANCE_SEG.DATA_MW_TH_FOREGROUND = 0.3
        # TH_DISTANCE controls channel 'D' in the creation of the MW seeds
        _C.PROBLEM.INSTANCE_SEG.DATA_MW_TH_DISTANCE = 1.
        # TH_POINTS controls channel 'P' in the creation of the MW seeds
        _C.PROBLEM.INSTANCE_SEG.DATA_MW_TH_POINTS = 0.5
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
        _C.PROBLEM.INSTANCE_SEG.ERODE_AND_DILATE_FOREGROUND = False
        # Radius to erode the foreground mask
        _C.PROBLEM.INSTANCE_SEG.FORE_EROSION_RADIUS = 5
        # Radius to dilate the foreground mask
        _C.PROBLEM.INSTANCE_SEG.FORE_DILATION_RADIUS = 5
        # Whether to save watershed check files
        _C.PROBLEM.INSTANCE_SEG.DATA_CHECK_MW = True
        
        ### DETECTION
        _C.PROBLEM.DETECTION = CN()
        # Size of the disk that will be used to dilate the central point created from the CSV file. 0 to not dilate and only create a 3x3 square.
        _C.PROBLEM.DETECTION.CENTRAL_POINT_DILATION = 3
        _C.PROBLEM.DETECTION.CHECK_POINTS_CREATED = True
        # Whether to save watershed check files
        _C.PROBLEM.DETECTION.DATA_CHECK_MW = True

        ### DENOISING
        # Based Noise2Void paper: https://arxiv.org/abs/1811.10980 
        _C.PROBLEM.DENOISING = CN()
        # This variable corresponds to n2v_perc_pix from Noise2Void. It explanation is as follows: for faster training multiple 
        # pixels per input patch can be manipulated. In our experiments we manipulated about 0.198% of the input pixels per 
        # patch. For a patch size of 64 by 64 pixels this corresponds to about 8 pixels. This fraction can be tuned via this variable
        _C.PROBLEM.DENOISING.N2V_PERC_PIX = 0.198
        # This variable corresponds to n2v_manipulator from Noise2Void. Most pixel manipulators will compute the replacement value based 
        # on a neighborhood and this variable controls how to do that
        _C.PROBLEM.DENOISING.N2V_MANIPULATOR = 'uniform_withCP'
        # This variable corresponds to n2v_neighborhood_radius from Noise2Void. Size of the neighborhood to compute the replacement
        _C.PROBLEM.DENOISING.N2V_NEIGHBORHOOD_RADIUS = 5
        # To apply a structured mask as is proposed in Noise2Void to alleviate the limitation of the method of not removing effectively 
        # the structured noise (section 4.4 of their paper). 
        _C.PROBLEM.DENOISING.N2V_STRUCTMASK = False

        ### SUPER_RESOLUTION
        _C.PROBLEM.SUPER_RESOLUTION = CN()
        _C.PROBLEM.SUPER_RESOLUTION.UPSCALING = 1

        ### SELF_SUPERVISED
        _C.PROBLEM.SELF_SUPERVISED = CN()
        # Pretext task to do. Options are as follows:
        #   - 'crappify': crappifies input image by adding Gaussian noise and downsampling and upsampling it so the resolution
        #                 gets worsen. Then, the model is trained to recover the original images.  
        #   - 'masking': mask input image and the model needs to recover the original image. This option can only
        #                be done with 'mae' transformer. This strategy follows the one proposed in: 
        #                Masked Autoencoders Are Scalable Vision Learners (https://arxiv.org/pdf/2111.06377.pdf)
        _C.PROBLEM.SELF_SUPERVISED.PRETEXT_TASK = 'crappify'
        # Downsizing factor to reshape the image. It will be downsampled and upsampled again by this factor so the 
        # quality of the image is worsens
        _C.PROBLEM.SELF_SUPERVISED.RESIZING_FACTOR = 4
        # Number between [0, 1] indicating the std of the Gaussian noise N(0,std).
        _C.PROBLEM.SELF_SUPERVISED.NOISE = 0.2

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Dataset
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
        _C.DATA.PROBABILITY_MAP = False # Used when _C.DATA.EXTRACT_RANDOM_PATCH=True
        _C.DATA.W_FOREGROUND = 0.94 # Used when _C.DATA.PROBABILITY_MAP=True
        _C.DATA.W_BACKGROUND = 0.06 # Used when _C.DATA.PROBABILITY_MAP=True

        # Whether to reshape de dimensions that does not satisfy the pathc shape selected by padding it with reflect.
        _C.DATA.REFLECT_TO_COMPLETE_SHAPE = False

        _C.DATA.NORMALIZATION = CN()
        # Normalization type to use. Possible options:
        #   'div' to divide values from 0/255 (or 0/65535 if uint16) in [0,1] range
        #   'custom' to use DATA.NORMALIZATION.CUSTOM_MEAN and DATA.NORMALIZATION.CUSTOM_STD to normalize
        #   '' if no normalization to be applied 
        _C.DATA.NORMALIZATION.TYPE = 'div'
        _C.DATA.NORMALIZATION.CUSTOM_MEAN = -1.0
        _C.DATA.NORMALIZATION.CUSTOM_STD = -1.0
        
        # Train
        _C.DATA.TRAIN = CN()
        # Whether to check if the data mask contains correct values, e.g. same classes as defined
        _C.DATA.TRAIN.CHECK_DATA = True
        _C.DATA.TRAIN.IN_MEMORY = True
        _C.DATA.TRAIN.PATH = os.path.join("user_data", 'train', 'x')
        _C.DATA.TRAIN.GT_PATH = os.path.join("user_data", 'train', 'y')
        # File to load/save data prepared with the appropiate channels in a instance segmentation problem.
        # E.g. _C.PROBLEM.TYPE ='INSTANCE_SEG' and _C.PROBLEM.INSTANCE_SEG.DATA_CHANNELS != 'B'
        _C.DATA.TRAIN.INSTANCE_CHANNELS_DIR = os.path.join("user_data", 'train', 'x_'+_C.PROBLEM.INSTANCE_SEG.DATA_CHANNELS)
        _C.DATA.TRAIN.INSTANCE_CHANNELS_MASK_DIR = os.path.join("user_data", 'train', 'y_'+_C.PROBLEM.INSTANCE_SEG.DATA_CHANNELS)
        # Path to load/save detection masks prepared. 
        _C.DATA.TRAIN.DETECTION_MASK_DIR = os.path.join("user_data", 'train', 'y_detection_masks')
        # Path to load/save SSL target prepared. 
        _C.DATA.TRAIN.SSL_SOURCE_DIR = os.path.join("user_data", 'train', 'x_ssl_source')
        # Extra train data generation: number of times to duplicate the train data. Useful when
        # _C.DATA.EXTRACT_RANDOM_PATCH=True is made, as more original train data can be cover on each epoch
        _C.DATA.TRAIN.REPLICATE = 0
        # Percentage of overlap in (y,x)/(z,y,x) when cropping validation. Set to 0 to calculate  the minimun overlap.
        # The values must be floats between range [0, 1). It needs to be a 2D tuple when using _C.PROBLEM.NDIM='2D' and
        # 3D tuple when using _C.PROBLEM.NDIM='3D'
        _C.DATA.TRAIN.OVERLAP = (0,0)
        # Padding to be done in (y,x)/(z,y,x) when reconstructing train data. Useful to avoid patch 'border effect'.
        _C.DATA.TRAIN.PADDING = (0,0)
        # Train data resolution. It is not completely necessary but when configured it is taken into account when
        # performing some augmentations, e.g. cutout. If defined it need to be (y,x)/(z,y,x) and needs to be to be a 2D
        # tuple when using _C.PROBLEM.NDIM='2D' and 3D tuple when using _C.PROBLEM.NDIM='3D'
        _C.DATA.TRAIN.RESOLUTION = (-1,)
        # Minimum foreground percentage that each image loaded need to have to not discard it. This option is only valid for SEMANTIC_SEG, 
        # INSTANCE_SEG and DETECTION. 
        _C.DATA.TRAIN.MINIMUM_FOREGROUND_PER = -1.

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
        _C.DATA.TEST.PATH = os.path.join("user_data", 'test', 'x')
        # Path to load the test data masks from. Not used when _C.DATA.TEST.USE_VAL_AS_TEST == True
        _C.DATA.TEST.GT_PATH = os.path.join("user_data", 'test', 'y')
        # File to load/save data prepared with the appropiate channels in a instance segmentation problem.
        # E.g. _C.PROBLEM.TYPE ='INSTANCE_SEG' and _C.PROBLEM.INSTANCE_SEG.DATA_CHANNELS != 'B'
        _C.DATA.TEST.INSTANCE_CHANNELS_DIR = os.path.join("user_data", 'test', 'x_'+_C.PROBLEM.INSTANCE_SEG.DATA_CHANNELS)
        _C.DATA.TEST.INSTANCE_CHANNELS_MASK_DIR = os.path.join("user_data", 'test', 'y_'+_C.PROBLEM.INSTANCE_SEG.DATA_CHANNELS)
        # Path to load/save detection masks prepared. 
        _C.DATA.TEST.DETECTION_MASK_DIR = os.path.join("user_data", 'test', 'y_detection_masks')
        # Path to load/save SSL target prepared. 
        _C.DATA.TEST.SSL_SOURCE_DIR = os.path.join("user_data", 'test', 'x_ssl_source')
        # Percentage of overlap in (y,x)/(z,y,x) when cropping validation. Set to 0 to calculate  the minimun overlap.
        # The values must be floats between range [0, 1). It needs to be a 2D tuple when using _C.PROBLEM.NDIM='2D' and
        # 3D tuple when using _C.PROBLEM.NDIM='3D'
        _C.DATA.TEST.OVERLAP = (0,0)
        # Padding to be done in (y,x)/(z,y,xz) when reconstructing test data. Useful to avoid patch 'border effect'
        _C.DATA.TEST.PADDING = (0,0)
        # Whether to use median values to fill padded pixels or zeros
        _C.DATA.TEST.MEDIAN_PADDING = False
        # Directory where binary masks to apply to resulting images should be. Used when _C.TEST.POST_PROCESSING.APPLY_MASK  == True
        _C.DATA.TEST.BINARY_MASKS = os.path.join("user_data", 'test', 'bin_mask')
        # Test data resolution. Need to be provided in (z,y,x) order. Only applies when _C.PROBLEM.TYPE = 'DETECTION' now.
        _C.DATA.TEST.RESOLUTION = (-1,)
        # Whether to apply argmax to the predicted images 
        _C.DATA.TEST.ARGMAX_TO_OUTPUT = False

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
        _C.DATA.VAL.SPLIT_TRAIN = 0.0
        # Create the validation data with random images of the training data. Used when _C.DATA.VAL.FROM_TRAIN = True
        _C.DATA.VAL.RANDOM = True
        # Used when _C.DATA.VAL.FROM_TRAIN = False, as DATA.VAL.FROM_TRAIN = True always implies DATA.VAL.IN_MEMORY = True
        _C.DATA.VAL.IN_MEMORY = True
        # Path to the validation data. Used when _C.DATA.VAL.FROM_TRAIN = False
        _C.DATA.VAL.PATH = os.path.join("user_data", 'val', 'x')
        # Path to the validation data mask. Used when _C.DATA.VAL.FROM_TRAIN = False
        _C.DATA.VAL.GT_PATH = os.path.join("user_data", 'val', 'y')
        # File to load/save data prepared with the appropiate channels in a instance segmentation problem.
        # E.g. _C.PROBLEM.TYPE ='INSTANCE_SEG' and _C.PROBLEM.INSTANCE_SEG.DATA_CHANNELS != 'B'
        _C.DATA.VAL.INSTANCE_CHANNELS_DIR = os.path.join("user_data", 'val', 'x_'+_C.PROBLEM.INSTANCE_SEG.DATA_CHANNELS)
        _C.DATA.VAL.INSTANCE_CHANNELS_MASK_DIR = os.path.join("user_data", 'val', 'y_'+_C.PROBLEM.INSTANCE_SEG.DATA_CHANNELS)
        # Path to load/save detection masks prepared. 
        _C.DATA.VAL.DETECTION_MASK_DIR = os.path.join("user_data", 'val', 'y_detection_masks')
        # Path to load/save SSL target prepared. 
        _C.DATA.VAL.SSL_SOURCE_DIR = os.path.join("user_data", 'val', 'x_ssl_source')
        # Percentage of overlap in (y,x)/(z,y,x) when cropping validation. Set to 0 to calculate  the minimun overlap.
        # The values must be floats between range [0, 1). It needs to be a 2D tuple when using _C.PROBLEM.NDIM='2D' and
        # 3D tuple when using _C.PROBLEM.NDIM='3D'. This is only used when the validation is loaded from disk, and thus, 
        # not extracted from training. 
        _C.DATA.VAL.OVERLAP = (0,0)
        # Padding to be done in (y,x)/(z,y,x) when cropping validation data. Useful to avoid patch 'border effect'. This 
        # is only used when the validation is loaded from disk, and thus, not extracted from training. 
        _C.DATA.VAL.PADDING = (0,0)
        # Directory where validation binary masks should be located. This binary mask will be applied only when MW_TH*
        # optimized values are find, that is, when _C.PROBLEM.INSTANCE_SEG.DATA_MW_OPTIMIZE_THS = True and _C.TEST.POST_PROCESSING.APPLY_MASK = True
        _C.DATA.VAL.BINARY_MASKS = os.path.join("user_data", 'val', 'bin_mask')
        # Not used yet.
        _C.DATA.VAL.RESOLUTION = (-1,)


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
        # Apply shift
        _C.AUGMENTOR.SHIFT = False
        # Shift range. Translation as a fraction of the image height/width (x-translation, y-translation), where 0 denotes 
        # “no change” and 0.5 denotes “half of the axis size”.
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
        # Strength of the distortion field. Higher values mean that pixels are moved further with respect to the distortion 
        # field’s direction. Set this to around 10 times the value of sigma for visible effects.
        _C.AUGMENTOR.E_ALPHA = (12, 16)
        # Standard deviation of the gaussian kernel used to smooth the distortion fields.  Higher values (for 128x128 images 
        # around 5.0) lead to more water-like effects, while lower values (for 128x128 images around 1.0 and lower) lead to
        # more noisy, pixelated images. Set this to around 1/10th of alpha for visible effects.
        _C.AUGMENTOR.E_SIGMA = 4
        # Parameter that defines the handling of newly created pixels with the elastic transformation
        _C.AUGMENTOR.E_MODE = 'constant'
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
        _C.AUGMENTOR.BRIGHTNESS_MODE = '3D'
        # To apply contrast changes to images
        _C.AUGMENTOR.CONTRAST = False
        # Strength of the contrast change range. 
        _C.AUGMENTOR.CONTRAST_FACTOR = (-0.1, 0.1)
        # If apply same contrast to the entire image or select one for each slice. For 2D does not matter but yes for
        # 3D images. Possible values: '2D' or '3D'. Used when '_C.PROBLEM.NDIM' = '3D'.
        _C.AUGMENTOR.CONTRAST_MODE = '3D'
        # To apply brightness changes to images based on Pytorch Connectomics library.
        _C.AUGMENTOR.BRIGHTNESS_EM = False
        # Strength of the brightness range
        _C.AUGMENTOR.BRIGHTNESS_EM_FACTOR = (-0.1, 0.1)
        # If apply same contrast to the entire image or select one for each slice. For 2D does not matter but yes for
        # 3D images. Possible values: '2D' or '3D'. Used when '_C.PROBLEM.NDIM' = '3D'.
        _C.AUGMENTOR.BRIGHTNESS_EM_MODE = '3D'
        # To apply contrast changes to images based on Pytorch Connectomics library.
        _C.AUGMENTOR.CONTRAST_EM = False
        # Strength of the contrast change range, with valid values being 0 <= contrast_em_factor <= 1
        _C.AUGMENTOR.CONTRAST_EM_FACTOR = (-0.1, 0.1)
        # If apply same contrast to the entire image or select one for each slice. For 2D does not matter but yes for
        # 3D images. Possible values: '2D' or '3D'. Used when '_C.PROBLEM.NDIM' = '3D'.
        _C.AUGMENTOR.CONTRAST_EM_MODE = '3D'
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
        _C.AUGMENTOR.COUT_CVAL = 0.
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
        # Range to choose a ``d`` value
        _C.AUGMENTOR.GRID_D_RANGE = (0.4, 1)
        # Rotation of the mask in GridMask. Needs to be between [0,1] where 1 is 360 degrees.
        _C.AUGMENTOR.GRID_ROTATE = 1.
        # Whether to invert the mask
        _C.AUGMENTOR.GRID_INVERT = False
        # Add Gaussian noise
        _C.AUGMENTOR.GAUSSIAN_NOISE = False
        _C.AUGMENTOR.GAUSSIAN_NOISE_MEAN = 0.
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

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Model definition
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        _C.MODEL = CN()
        # Architecture of the network. Possible values are: 'unet', 'resunet', 'resunet++', 'attention_unet', 'nnunet',  
        # 'multiresunet', 'seunet', 'simple_cnn', 'efficientnet_b[0-7]', 'unetr', 'edsr', 'rcan', 'dfcan', 'wdsr', 'ViT'
        # 'mae'
        _C.MODEL.ARCHITECTURE = 'unet'
        # Number of feature maps on each level of the network.
        _C.MODEL.FEATURE_MAPS = [16, 32, 64, 128, 256]
        # Values to make the dropout with. Set to 0 to prevent dropout. When using it with 'ViT' or 'unetr' 
        # a list with just one number must be provided 
        _C.MODEL.DROPOUT_VALUES = [0., 0., 0., 0., 0.]
        # To active batch normalization
        _C.MODEL.BATCH_NORMALIZATION = True
        # Kernel size
        _C.MODEL.KERNEL_SIZE = 3
        # Upsampling layer to use in the model. Options: ["upsampling", "convtranspose"]
        _C.MODEL.UPSAMPLE_LAYER = "convtranspose"
        # Activation function to use along the model
        _C.MODEL.ACTIVATION = 'ELU'
        # Las activation to use. Options 'sigmoid', 'softmax' or 'linear'
        _C.MODEL.LAST_ACTIVATION = 'sigmoid' 
        # Number of classes without counting the background class (that should be using 0 label)
        _C.MODEL.N_CLASSES = 2
        # Downsampling to be made in Z. This value will be the third integer of the MaxPooling operation. When facing
        # anysotropic datasets set it to get better performance
        _C.MODEL.Z_DOWN = [0, 0, 0, 0]
        # Checkpoint: set to True to load previous training weigths (needed for inference or to make fine-tunning)
        _C.MODEL.LOAD_CHECKPOINT = False
        # When loading checkpoints whether if only model's weights are going to be loaded or optimizer, epochs and loss_scaler. 
        _C.MODEL.LOAD_CHECKPOINT_ONLY_WEIGHTS = True
        # Decide which checkpoint to load from job's dir if PATHS.CHECKPOINT_FILE is ''. 
        # Options: 'best_on_val' or 'last_on_train'
        _C.MODEL.LOAD_CHECKPOINT_EPOCH = 'best_on_val' 
        # Epochs to save a checkpoint of the model apart from the ones saved with LOAD_CHECKPOINT_ONLY_WEIGHTS. Set it to -1 to 
        # not do it.
        _C.MODEL.SAVE_CKPT_FREQ = -1

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
        _C.MODEL.VIT_MLP_RATIO = 4.
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
        # Percentage of the input image to mask. Value between 0 and 1. 
        _C.MODEL.MAE_MASK_RATIO = 0.5

        # UNETR
        # Multiple of the transformer encoder layers from of which the skip connection signal is going to be extracted
        _C.MODEL.UNETR_VIT_HIDD_MULT = 3
        # Number of filters in the first UNETR's layer of the decoder. In each layer the previous number of filters is doubled.
        _C.MODEL.UNETR_VIT_NUM_FILTERS = 16
        # Decoder activation
        _C.MODEL.UNETR_DEC_ACTIVATION = 'relu'

        # Specific for SR models based on U-Net architectures. Options are ["pre", "post"]
        _C.MODEL.UNET_SR_UPSAMPLE_POSITION = "pre"

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
        # Optimizer to use. Possible values: "SGD", "ADAM" or "ADAMW"
        _C.TRAIN.OPTIMIZER = 'SGD'
        # Learning rate 
        _C.TRAIN.LR = 1.E-4
        # Weight decay
        _C.TRAIN.W_DECAY = 0.05
        # Batch size
        _C.TRAIN.BATCH_SIZE = 2
        # If memory or # gpus is limited, use this variable to maintain the effective batch size, which is 
        # batch_size (per gpu) * nodes * (gpus per node) * accum_iter.
        _C.TRAIN.ACCUM_ITER = 1
        # Number of epochs to train the model
        _C.TRAIN.EPOCHS = 360
        # Epochs to wait with no validation data improvement until the training is stopped
        _C.TRAIN.PATIENCE = -1
        
        # LR Scheduler
        _C.TRAIN.LR_SCHEDULER = CN()
        _C.TRAIN.LR_SCHEDULER.NAME = '' # Possible options: 'warmupcosine', 'reduceonplateau', 'onecycle'
        # Lower bound on the learning rate used in 'warmupcosine' and 'reduceonplateau'
        _C.TRAIN.LR_SCHEDULER.MIN_LR = -1.
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
        # To determine which value monitor to consider which epoch consider the best to save
        _C.TRAIN.CHECKPOINT_MONITOR = 'val_loss'
        # Add profiler callback to the training
        # _C.TRAIN.PROFILER = False
        # # Batch range to be analyzed
        # _C.TRAIN.PROFILER_BATCH_RANGE='10, 100'

        # _C.TRAIN.MAE_CALLBACK_EPOCHS = 5

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Inference phase
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
        # Type of format used to write data. Options available: ["H5", "Zarr"]
        _C.TEST.BY_CHUNKS.FORMAT = 'H5' 
        # In the process of 'TEST.BY_CHUNKS' you can enable this variable to save the reconstructed prediction as a TIF too. 
        # Be aware of this option and be sure that the prediction can fit in you memory entirely, as it is needed for saving as TIF.
        _C.TEST.BY_CHUNKS.SAVE_OUT_TIF = False
        # In how many iterations the H5 writer needs to flush the data. No need to do so with Zarr files.
        _C.TEST.BY_CHUNKS.FLUSH_EACH = 100
        # Whether if after reconstructing the prediction the pipeline will continue each workflow specific steps. For this process
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
        # Whether to evaluate or not
        _C.TEST.EVALUATE = True
        # Stack 2D images into a 3D image and then process it entirely instead of going image per image. This stacks
        # images that have been merged with 'TEST.STATS.MERGE_PATCHES' = True or 'TEST.STATS.FULL_IMG' (priorizes this later)  
        _C.TEST.ANALIZE_2D_IMGS_AS_3D_STACK = False

        _C.TEST.STATS = CN()
        _C.TEST.STATS.PER_PATCH = False
        _C.TEST.STATS.MERGE_PATCHES = False # Only used when _C.TEST.STATS.PER_PATCH = True
        _C.TEST.STATS.FULL_IMG = True # Only when if PROBLEM.NDIM = '2D' as 3D images are huge for the GPU

        ### Instance segmentation
        # Whether to calculate matching statistics (average overlap, accuracy, recall, precision, etc.)
        _C.TEST.MATCHING_STATS = True
        # Theshold of overlap to consider a TP when calculating the metrics. If more than one value is provided 
        # the process is repeated with each of the threshold values
        _C.TEST.MATCHING_STATS_THS = [0.3, 0.5, 0.75]
        # Decide in which thresholds to create a colored image of the TPs, FNs and FPs
        _C.TEST.MATCHING_STATS_THS_COLORED_IMG = [0.3]
        _C.TEST.MATCHING_SEGCOMPARE = False

        ### Detection
        # To decide which function is going to be used to create point from probabilities. Options: ['peak_local_max', 'blob_log']
        # 'peak_local_max': https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.peak_local_max 
        # 'blob_log': https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.blob_log
        _C.TEST.DET_POINT_CREATION_FUNCTION = 'peak_local_max'
        # Minimun value to consider a point as a peak. Corresponds to 'threshold_abs' argument of the function
        # 'peak_local_max' of skimage.feature
        _C.TEST.DET_MIN_TH_TO_BE_PEAK = [0.2]  
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
        _C.TEST.DET_TOLERANCE = [10]

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Post-processing
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # When PROBLEM.NDIM = '2D' only applies when _C.TEST.STATS.FULL_IMG = True, if PROBLEM.NDIM = '3D' is applied
        # when _C.TEST.STATS.MERGE_PATCHES = True
        _C.TEST.POST_PROCESSING = CN()
        _C.TEST.POST_PROCESSING.YZ_FILTERING = False
        _C.TEST.POST_PROCESSING.YZ_FILTERING_SIZE = 5
        _C.TEST.POST_PROCESSING.Z_FILTERING = False
        _C.TEST.POST_PROCESSING.Z_FILTERING_SIZE = 5
        # Apply a binary mask to remove possible segmentation outside it
        _C.TEST.POST_PROCESSING.APPLY_MASK = False

        # Remove instances by the conditions based in each instance properties. The three variables, i.e. TEST.POST_PROCESSING.REMOVE_BY_PROPERTIES,
        # TEST.POST_PROCESSING.REMOVE_BY_PROPERTIES_VALUES and TEST.POST_PROCESSING.REMOVE_BY_PROPERTIES_SIGN will compose a list 
        # of conditions to remove the instances. They are list of list of conditions. For instance, the conditions can be 
        # like this: [['A'], ['B','C']]. Then, if the instance satisfy the first list of conditions, only 'A' in this first case 
        # (from ['A'] list), or satisfy 'B' and 'C' (from ['B','C'] list) it will be removed from the image. In each sublist all the conditions
        # must be satisfied. Available properties are: ['circularity', 'npixels', 'area', 'diameter']. When this post-processing step is 
        # selected two .csv files will be created, one with the properties of each instance from the original image (will be placed in 
        # PATHS.RESULT_DIR.PER_IMAGE_INSTANCES path), and another with only instances that remain once this post-processing has been 
        # applied (will be placed in PATHS.RESULT_DIR.PER_IMAGE_POST_PROCESSING path). In those csv files two more information columns will 
        # appear: a list of conditions that each instance has satisfy or not ('Satisfied', 'No satisfied' respectively), and a comment with two 
        # possible values, 'Strange' and 'Correct', telling you if the instance has been removed or not, respectively. 
        # Each property descrition:
        #   * 'circularity' of an area instance indicates how compact or elongated the instance is. A value of 1 indicates that the instance 
        #     is a perfect circle (compact), and 0 indicates that it is a line (elongated). Calculated as: (4 * PI * area) / (perimeter^2).
        #     In 3D, circularity is only measured in the center slices of the instance (one slice before the central slice, the central 
        #     slice, and one after it). 
        #   * 'area' correspond to the number of pixels taking into account the image resolution (we call it 'area' also even in a 3D 
        #     image for simplicity, but that will be the volume in that case). In the resulting statistics 'volume' will appear in that 
        #     case too.
        #   * 'npixels' corresponds to the sum of pixels that compose an instance.
        #   * 'diameter' calculated with the bounding box and by taking the maximum value of the box in x and y axes. In 3D, z axis 
        #      is also taken into account. Does not take into account the image resolution. 
        _C.TEST.POST_PROCESSING.REMOVE_BY_PROPERTIES = []
        # Values of the properties listed in TEST.POST_PROCESSING.REMOVE_BY_PROPERTIES that the instances need to satisfy to
        # not be droped. Values for each property:
        # * 'circularity' must be a float in [0,1] range. 
        # * 'area', 'npixels' and 'diameter' can be integer/float. 
        _C.TEST.POST_PROCESSING.REMOVE_BY_PROPERTIES_VALUES = []
        # Sign to do the comparison. Options: ['gt', 'ge', 'lt', 'le'] that corresponds to "greather than", e.g. ">", 
        # "greather equal", e.g. ">=", "less than", e.g. "<", and "less equal" e.g. "<=" comparisons.
        _C.TEST.POST_PROCESSING.REMOVE_BY_PROPERTIES_SIGN = []
        # A full example of this post-processing: 
        # If you want to remove those instances that have less than 100 pixels and circularity less equal to 0.7 you should 
        # declare the above three variables as follows:
        # _C.TEST.POST_PROCESSING.REMOVE_BY_PROPERTIES = [['npixels', 'circularity']]
        # _C.TEST.POST_PROCESSING.REMOVE_BY_PROPERTIES_VALUES = [[100, 0.7]]
        # _C.TEST.POST_PROCESSING.REMOVE_BY_PROPERTIES_SIGN = [['lt', 'le']]
        # You can also concatenate more restrictions and they will be applied in order. For instance, if you want to remove those 
        # instances that are bigger than an specific area, and do that before the condition described above, you can define the 
        # variables this way:
        # _C.TEST.POST_PROCESSING.REMOVE_BY_PROPERTIES = [['area'], ['npixels', 'circularity']]
        # _C.TEST.POST_PROCESSING.REMOVE_BY_PROPERTIES_VALUES = [[500], [100, 0.7]]
        # _C.TEST.POST_PROCESSING.REMOVE_BY_PROPERTIES_SIGN = [['gt'], ['lt', 'le']]        
        # This way, the instances will be removed by 'area' and then by 'npixels' and 'circularity'

        ### Instance segmentation
        # Whether to apply Voronoi using 'BC' or 'M' channels need to be present
        _C.TEST.POST_PROCESSING.VORONOI_ON_MASK = False
        # Threshold to be applied to the 'M' channel when expanding the instances with Voronoi. Need to be in [0,1] range.
        # Leave it to 0 to adjust the threhold with Otsu
        _C.TEST.POST_PROCESSING.VORONOI_TH = 0.
        # Set it to try to repare large instances by merging their neighbors with them and removing possible central holes.  
        # Its value determines which instances are going to be repared by size (number of pixels that compose the instance)
        # This option is useful when PROBLEM.INSTANCE_SEG.DATA_CHANNELS is 'BP', as multiple central seeds may appear in big 
        # instances.
        _C.TEST.POST_PROCESSING.REPARE_LARGE_BLOBS_SIZE = -1
        # Clear objects connected to the label image border
        _C.TEST.POST_PROCESSING.CLEAR_BORDER = False

        ### Detection 
        # To remove close points to each other. This can also be set when using 'BP' channels for instance segmentation.
        _C.TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS = False
        # Distance between points to be considered the same. Only applies when TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS = True
        # This can also be set when using 'BP' channels for instance segmentation.
        _C.TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS_RADIUS = [-1.0]
        # Whether to apply a watershed to grow the points detected 
        _C.TEST.POST_PROCESSING.DET_WATERSHED = False
        # Structure per each class to dilate the initial seeds before watershed
        _C.TEST.POST_PROCESSING.DET_WATERSHED_FIRST_DILATION = [[-1,-1],]
        # List of classes to be consider as 'donuts'. For those class points, the 'donuts' type cell means that their nucleus is 
        # to big and that the seeds need to be dilated more so the watershed can grow the instances properly.
        _C.TEST.POST_PROCESSING.DET_WATERSHED_DONUTS_CLASSES = [-1]
        # Patch shape to extract all donuts type cells. It needs to be a bit greater than bigest donuts type cell so all of them can
        # be contained in this patch. This is used to analize that area for each point of class `DET_WATERSHED_DONUTS_CLASSES`.
        _C.TEST.POST_PROCESSING.DET_WATERSHED_DONUTS_PATCH = [13,120,120]
        # Diameter (in pixels) that a cell need to have to be considered as donuts type
        _C.TEST.POST_PROCESSING.DET_WATERSHED_DONUTS_NUCLEUS_DIAMETER = 30
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Auxiliary paths
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        _C.PATHS = CN()

        # Directories to store the results
        _C.PATHS.RESULT_DIR = CN()
        _C.PATHS.RESULT_DIR.PATH = os.path.join(job_dir, 'results', job_identifier)
        _C.PATHS.RESULT_DIR.PER_IMAGE = os.path.join(_C.PATHS.RESULT_DIR.PATH, 'per_image')
        _C.PATHS.RESULT_DIR.PER_IMAGE_BIN = os.path.join(_C.PATHS.RESULT_DIR.PATH, 'per_image_binarized')
        _C.PATHS.RESULT_DIR.PER_IMAGE_INSTANCES = os.path.join(_C.PATHS.RESULT_DIR.PATH, 'per_image_instances')
        _C.PATHS.RESULT_DIR.PER_IMAGE_POST_PROCESSING = os.path.join(_C.PATHS.RESULT_DIR.PATH, 'per_image_post_processing')
        _C.PATHS.RESULT_DIR.FULL_IMAGE = os.path.join(_C.PATHS.RESULT_DIR.PATH, 'full_image')
        _C.PATHS.RESULT_DIR.FULL_IMAGE_BIN = os.path.join(_C.PATHS.RESULT_DIR.PATH, 'full_image_binarized')
        _C.PATHS.RESULT_DIR.AS_3D_STACK_POST_PROCESSING = os.path.join(_C.PATHS.RESULT_DIR.PATH, 'as_3d_stack_post_processing')
        _C.PATHS.RESULT_DIR.DET_LOCAL_MAX_COORDS_CHECK = os.path.join(_C.PATHS.RESULT_DIR.PATH, 'per_image_local_max_check')
        _C.PATHS.RESULT_DIR.DET_ASSOC_POINTS = os.path.join(_C.PATHS.RESULT_DIR.PATH, 'point_associations')
        _C.PATHS.RESULT_DIR.INST_ASSOC_POINTS = os.path.join(_C.PATHS.RESULT_DIR.PATH, 'instance_associations')
        
        # Path to store profiler files
        _C.PATHS.PROFILER = os.path.join(_C.PATHS.RESULT_DIR.PATH, 'profiler')

        # Name of the folder where the charts of the loss and metrics values while training the network are stored.
        # Additionally, MW_TH* variable charts are stored if _C.PROBLEM.INSTANCE_SEG.DATA_MW_OPTIMIZE_THS = True
        _C.PATHS.CHARTS = os.path.join(_C.PATHS.RESULT_DIR.PATH, 'charts')
        # Folder where samples of DA will be stored
        _C.PATHS.DA_SAMPLES = os.path.join(_C.PATHS.RESULT_DIR.PATH, 'aug')
        # Folder where generator samples (X) will be stored
        _C.PATHS.GEN_CHECKS = os.path.join(_C.PATHS.RESULT_DIR.PATH, 'gen_check')
        # Folder where generator samples (Y) will be stored
        _C.PATHS.GEN_MASK_CHECKS = os.path.join(_C.PATHS.RESULT_DIR.PATH, 'gen_mask_check')
        # Paths where a few samples of instance channels created will be stored just to check id there is any problem
        _C.PATHS.TRAIN_INSTANCE_CHANNELS_CHECK = os.path.join(_C.PATHS.RESULT_DIR.PATH, 'train_'+_C.PROBLEM.INSTANCE_SEG.DATA_CHANNELS+'_instance_channels')
        _C.PATHS.VAL_INSTANCE_CHANNELS_CHECK = os.path.join(_C.PATHS.RESULT_DIR.PATH, 'val_'+_C.PROBLEM.INSTANCE_SEG.DATA_CHANNELS+'_instance_channels')
        _C.PATHS.TEST_INSTANCE_CHANNELS_CHECK = os.path.join(_C.PATHS.RESULT_DIR.PATH, 'test_'+_C.PROBLEM.INSTANCE_SEG.DATA_CHANNELS+'_instance_channels')
        # Name of the folder where weights files will be stored/loaded from.
        _C.PATHS.CHECKPOINT = os.path.join(job_dir, 'checkpoints')
        # Checkpoint file to load/store the model weights
        _C.PATHS.CHECKPOINT_FILE = ''
        # Name of the folder to store the probability map to avoid recalculating it on every run
        _C.PATHS.PROB_MAP_DIR = os.path.join(job_dir, 'prob_map')
        _C.PATHS.PROB_MAP_FILENAME = 'prob_map.npy'
        # Watershed debugging folder
        _C.PATHS.WATERSHED_DIR = os.path.join(_C.PATHS.RESULT_DIR.PATH, 'watershed')
        _C.PATHS.MEAN_INFO_FILE = os.path.join(_C.PATHS.CHECKPOINT, 'normalization_mean_value.npy')
        _C.PATHS.STD_INFO_FILE = os.path.join(_C.PATHS.CHECKPOINT, 'normalization_std_value.npy')
        # Path where the images used in MAE will be saved suring inference
        _C.PATHS.MAE_OUT_DIR = os.path.join(_C.PATHS.RESULT_DIR.PATH, 'MAE_checks')
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Logging
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        _C.LOG = CN()
        _C.LOG.LOG_DIR = os.path.join(job_dir, 'train_logs')
        _C.LOG.TENSORBOARD_LOG_DIR = os.path.join(_C.PATHS.RESULT_DIR.PATH, 'tensorboard')
        _C.LOG.LOG_FILE_PREFIX = job_identifier
        _C.LOG.CHART_CREATION_FREQ = 5

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
        # Remove possible / characters at the end of the paths
        self._C.DATA.TRAIN.PATH = self._C.DATA.TRAIN.PATH if self._C.DATA.TRAIN.PATH[-1] != '/' else self._C.DATA.TRAIN.PATH[:-1]
        self._C.DATA.TRAIN.GT_PATH = self._C.DATA.TRAIN.GT_PATH if self._C.DATA.TRAIN.GT_PATH[-1] != '/' else self._C.DATA.TRAIN.GT_PATH[:-1]
        self._C.DATA.VAL.PATH = self._C.DATA.VAL.PATH if self._C.DATA.VAL.PATH[-1] != '/' else self._C.DATA.VAL.PATH[:-1]
        self._C.DATA.VAL.GT_PATH = self._C.DATA.VAL.GT_PATH if self._C.DATA.VAL.GT_PATH[-1] != '/' else self._C.DATA.VAL.GT_PATH[:-1]
        self._C.DATA.TEST.PATH = self._C.DATA.TEST.PATH if self._C.DATA.TEST.PATH[-1] != '/' else self._C.DATA.TEST.PATH[:-1]
        self._C.DATA.TEST.GT_PATH = self._C.DATA.TEST.GT_PATH if self._C.DATA.TEST.GT_PATH[-1] != '/' else self._C.DATA.TEST.GT_PATH[:-1]

        self._C.DATA.TRAIN.INSTANCE_CHANNELS_DIR = self._C.DATA.TRAIN.PATH+'_'+self._C.PROBLEM.INSTANCE_SEG.DATA_CHANNELS+'_'+self._C.PROBLEM.INSTANCE_SEG.DATA_CONTOUR_MODE
        self._C.DATA.TRAIN.INSTANCE_CHANNELS_MASK_DIR = self._C.DATA.TRAIN.GT_PATH+'_'+self._C.PROBLEM.INSTANCE_SEG.DATA_CHANNELS+'_'+self._C.PROBLEM.INSTANCE_SEG.DATA_CONTOUR_MODE
        self._C.DATA.TRAIN.DETECTION_MASK_DIR = self._C.DATA.TRAIN.GT_PATH+'_detection_masks'
        self._C.DATA.TRAIN.SSL_SOURCE_DIR = self._C.DATA.TRAIN.PATH+'_ssl_source'
        self._C.DATA.VAL.INSTANCE_CHANNELS_DIR = self._C.DATA.VAL.PATH+'_'+self._C.PROBLEM.INSTANCE_SEG.DATA_CHANNELS+'_'+self._C.PROBLEM.INSTANCE_SEG.DATA_CONTOUR_MODE
        self._C.DATA.VAL.INSTANCE_CHANNELS_MASK_DIR = self._C.DATA.VAL.GT_PATH+'_'+self._C.PROBLEM.INSTANCE_SEG.DATA_CHANNELS+'_'+self._C.PROBLEM.INSTANCE_SEG.DATA_CONTOUR_MODE
        # If value is not the default
        if self._C.DATA.VAL.BINARY_MASKS == os.path.join("user_data", 'val', 'bin_mask'):
            self._C.DATA.VAL.BINARY_MASKS = os.path.join(self._C.DATA.VAL.PATH, '..', 'bin_mask')
        self._C.DATA.VAL.DETECTION_MASK_DIR = self._C.DATA.VAL.GT_PATH+'_detection_masks'
        self._C.DATA.VAL.SSL_SOURCE_DIR = self._C.DATA.VAL.PATH+'_ssl_source'
        self._C.DATA.TEST.INSTANCE_CHANNELS_DIR = self._C.DATA.TEST.PATH+'_'+self._C.PROBLEM.INSTANCE_SEG.DATA_CHANNELS+'_'+self._C.PROBLEM.INSTANCE_SEG.DATA_CONTOUR_MODE
        self._C.DATA.TEST.INSTANCE_CHANNELS_MASK_DIR = self._C.DATA.TEST.GT_PATH+'_'+self._C.PROBLEM.INSTANCE_SEG.DATA_CHANNELS+'_'+self._C.PROBLEM.INSTANCE_SEG.DATA_CONTOUR_MODE
        # If value is not the default
        if self._C.DATA.TEST.BINARY_MASKS == os.path.join("user_data", 'test', 'bin_mask'):
            self._C.DATA.TEST.BINARY_MASKS = os.path.join(self._C.DATA.TEST.PATH, '..', 'bin_mask')
        self._C.DATA.TEST.DETECTION_MASK_DIR = self._C.DATA.TEST.GT_PATH+'_detection_masks'
        self._C.DATA.TEST.SSL_SOURCE_DIR = self._C.DATA.TEST.PATH+'_ssl_source'
        self._C.PATHS.TEST_FULL_GT_H5 = os.path.join(self._C.DATA.TEST.GT_PATH, 'h5')

        self._C.PATHS.TRAIN_INSTANCE_CHANNELS_CHECK = os.path.join(self._C.PATHS.RESULT_DIR.PATH, 'train_'+self._C.PROBLEM.INSTANCE_SEG.DATA_CHANNELS+'_instance_channels')
        self._C.PATHS.VAL_INSTANCE_CHANNELS_CHECK = os.path.join(self._C.PATHS.RESULT_DIR.PATH, 'val_'+self._C.PROBLEM.INSTANCE_SEG.DATA_CHANNELS+'_instance_channels')
        self._C.PATHS.TEST_INSTANCE_CHANNELS_CHECK = os.path.join(self._C.PATHS.RESULT_DIR.PATH, 'test_'+self._C.PROBLEM.INSTANCE_SEG.DATA_CHANNELS+'_instance_channels')
