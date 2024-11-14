import os
import numpy as np
import collections
from biapy.utils.misc import get_checkpoint_path
from biapy.data.data_manipulation import check_value


def check_configuration(cfg, jobname, check_data_paths=True):
    """
    Check if the configuration is good.
    """

    if cfg.SYSTEM.NUM_WORKERS < 0:
        raise ValueError("'SYSTEM.NUM_WORKERS' can not be less than 0")

    dim_count = 2 if cfg.PROBLEM.NDIM == "2D" else 3

    # Adjust overlap and padding in the default setting if it was not set
    opts = []
    if cfg.PROBLEM.NDIM == "3D":
        if cfg.DATA.TRAIN.OVERLAP == (0, 0):
            opts.extend(["DATA.TRAIN.OVERLAP", (0, 0, 0)])
        if cfg.DATA.TRAIN.PADDING == (0, 0):
            opts.extend(["DATA.TRAIN.PADDING", (0, 0, 0)])
        if cfg.DATA.VAL.OVERLAP == (0, 0):
            opts.extend(["DATA.VAL.OVERLAP", (0, 0, 0)])
        if cfg.DATA.VAL.PADDING == (0, 0):
            opts.extend(["DATA.VAL.PADDING", (0, 0, 0)])
        if cfg.DATA.TEST.OVERLAP == (0, 0):
            opts.extend(["DATA.TEST.OVERLAP", (0, 0, 0)])
        if cfg.DATA.TEST.PADDING == (0, 0):
            opts.extend(["DATA.TEST.PADDING", (0, 0, 0)])

    # Adjust channel weights
    if cfg.PROBLEM.TYPE == "INSTANCE_SEG":
        channels_provided = len(cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS.replace("Dv2", "D"))
        if cfg.MODEL.N_CLASSES > 2:
            channels_provided += 1
        if len(cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNEL_WEIGHTS) != channels_provided:
            if cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNEL_WEIGHTS == (1, 1):
                opts.extend(
                    [
                        "PROBLEM.INSTANCE_SEG.DATA_CHANNEL_WEIGHTS",
                        (1,) * channels_provided,
                    ]
                )

    for phase in ["TRAIN", "VAL", "TEST"]:
        if getattr(cfg.DATA, phase).FILTER_SAMPLES.ENABLE:
            if not (
                len(getattr(cfg.DATA, phase).FILTER_SAMPLES.PROPS)
                == len(getattr(cfg.DATA, phase).FILTER_SAMPLES.VALUES)
                == len(getattr(cfg.DATA, phase).FILTER_SAMPLES.SIGNS)
            ):
                raise ValueError(
                    "'DATA.TRAIN.FILTER_SAMPLES.PROPS', 'DATA.TRAIN.FILTER_SAMPLES.VALUES' and "
                    "'DATA.TRAIN.FILTER_SAMPLES.SIGNS' need to have same length"
                )
            foreground_filter_requested = any(
                [True for cond in getattr(cfg.DATA, phase).FILTER_SAMPLES.PROPS if "foreground" in cond]
            )
            if foreground_filter_requested:
                if cfg.PROBLEM.TYPE not in ["SEMANTIC_SEG", "INSTANCE_SEG", "DETECTION"]:
                    raise ValueError(
                        "'foreground' property can only be used in SEMANTIC_SEG, INSTANCE_SEG and DETECTION workflows"
                    )
                if phase == "TEST" and not cfg.DATA.TEST.LOAD_GT and cfg.DATA.TEST.USE_VAL_AS_TEST:
                    raise ValueError(
                        "'foreground' condition can not be used for filtering when test ground truth is not provided"
                    )

            if len(getattr(cfg.DATA, phase).FILTER_SAMPLES.PROPS) == 0:
                raise ValueError(
                    "'DATA.TRAIN.FILTER_SAMPLES.PROPS' can not be an empty list when "
                    "'DATA.TRAIN.FILTER_SAMPLES.ENABLE' is enabled"
                )

            for i in range(len(getattr(cfg.DATA, phase).FILTER_SAMPLES.PROPS)):
                if not isinstance(
                    getattr(cfg.DATA, phase).FILTER_SAMPLES.PROPS[i],
                    list,
                ):
                    raise ValueError(
                        "'DATA.TRAIN.FILTER_SAMPLES.PROPS' need to be a list of list. E.g. [ ['mean'], ['min', 'max'] ]"
                    )
                if not isinstance(
                    getattr(cfg.DATA, phase).FILTER_SAMPLES.VALUES[i],
                    list,
                ):
                    raise ValueError(
                        "'DATA.TRAIN.FILTER_SAMPLES.VALUES' need to be a list of list. E.g. [ [10], [15, 3] ]"
                    )
                if not isinstance(
                    getattr(cfg.DATA, phase).FILTER_SAMPLES.SIGNS[i],
                    list,
                ):
                    raise ValueError(
                        "'DATA.TRAIN.FILTER_SAMPLES.SIGNS' need to be a list of list. E.g. [ ['gt'], ['le', 'gt'] ]"
                    )

                if not (
                    len(getattr(cfg.DATA, phase).FILTER_SAMPLES.PROPS[i])
                    == len(getattr(cfg.DATA, phase).FILTER_SAMPLES.VALUES[i])
                    == len(getattr(cfg.DATA, phase).FILTER_SAMPLES.SIGNS[i])
                ):
                    raise ValueError(
                        "'DATA.TRAIN.FILTER_SAMPLES.PROPS', 'DATA.TRAIN.FILTER_SAMPLES.VALUES' and "
                        "'DATA.TRAIN.FILTER_SAMPLES.SIGNS' need to have same length"
                    )

                # Check for unique values
                if (
                    len(
                        [
                            item
                            for item, count in collections.Counter(
                                getattr(cfg.DATA, phase).FILTER_SAMPLES.PROPS[i]
                            ).items()
                            if count > 1
                        ]
                    )
                    > 0
                ):
                    raise ValueError("Non repeated values are allowed in 'DATA.TRAIN.FILTER_SAMPLES'")
                for j in range(len(getattr(cfg.DATA, phase).FILTER_SAMPLES.PROPS[i])):
                    if getattr(cfg.DATA, phase).FILTER_SAMPLES.PROPS[i][j] not in ["foreground", "mean", "min", "max"]:
                        raise ValueError(
                            "'DATA.TRAIN.FILTER_SAMPLES.PROPS' can only be one among these: ['foreground', 'mean', 'min', 'max']"
                        )
                    if getattr(cfg.DATA, phase).FILTER_SAMPLES.SIGNS[i][j] not in [
                        "gt",
                        "ge",
                        "lt",
                        "le",
                    ]:
                        raise ValueError(
                            "'DATA.TRAIN.FILTER_SAMPLES.SIGNS' can only be one among these: ['gt', 'ge', 'lt', 'le']"
                        )
                    if getattr(cfg.DATA, phase).FILTER_SAMPLES.PROPS[i][j] == "foreground" and not check_value(
                        getattr(cfg.DATA, phase).FILTER_SAMPLES.VALUES[i][j]
                    ):
                        raise ValueError(
                            "'foreground' property value can only be in [0, 1] range (check 'DATA.TRAIN.FILTER_SAMPLES.VALUES' values)"
                        )

    if len(cfg.DATA.TRAIN.RESOLUTION) == 1 and cfg.DATA.TRAIN.RESOLUTION[0] == -1:
        opts.extend(["DATA.TRAIN.RESOLUTION", (1,) * dim_count])
    if len(cfg.DATA.VAL.RESOLUTION) == 1 and cfg.DATA.VAL.RESOLUTION[0] == -1:
        opts.extend(["DATA.VAL.RESOLUTION", (1,) * dim_count])
    if len(cfg.DATA.TEST.RESOLUTION) == 1 and cfg.DATA.TEST.RESOLUTION[0] == -1:
        opts.extend(["DATA.TEST.RESOLUTION", (1,) * dim_count])

    if cfg.TEST.POST_PROCESSING.REPARE_LARGE_BLOBS_SIZE != -1:
        if cfg.PROBLEM.TYPE != "INSTANCE_SEG":
            raise ValueError(
                "'TEST.POST_PROCESSING.REPARE_LARGE_BLOBS_SIZE' can only be set when 'PROBLEM.TYPE' is 'INSTANCE_SEG'"
            )
        if cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS != "BP":
            raise ValueError(
                "'TEST.POST_PROCESSING.REPARE_LARGE_BLOBS_SIZE' only makes sense when 'PROBLEM.INSTANCE_SEG.DATA_CHANNELS == 'BP'"
            )
    if cfg.TEST.POST_PROCESSING.DET_WATERSHED and cfg.PROBLEM.TYPE != "DETECTION":
        raise ValueError("'TEST.POST_PROCESSING.DET_WATERSHED' can only be set when 'PROBLEM.TYPE' is 'DETECTION'")
    if cfg.TEST.POST_PROCESSING.DET_WATERSHED:
        for x in cfg.TEST.POST_PROCESSING.DET_WATERSHED_FIRST_DILATION:
            if not isinstance(x, list):
                raise ValueError("'TEST.POST_PROCESSING.DET_WATERSHED_FIRST_DILATION' needs to be a list of list")
            if any(y == -1 for y in x):
                raise ValueError(
                    "Please set 'TEST.POST_PROCESSING.DET_WATERSHED_FIRST_DILATION' when using 'TEST.POST_PROCESSING.DET_WATERSHED_FIRST_DILATION'"
                )
            if len(x) != dim_count:
                raise ValueError(
                    "'TEST.POST_PROCESSING.DET_WATERSHED_FIRST_DILATION' needs to be of dimension {} for {} problem".format(
                        dim_count, cfg.PROBLEM.NDIM
                    )
                )
        if cfg.TEST.POST_PROCESSING.DET_WATERSHED_DONUTS_CLASSES != [-1]:
            if len(cfg.TEST.POST_PROCESSING.DET_WATERSHED_DONUTS_CLASSES) > cfg.MODEL.N_CLASSES:
                raise ValueError(
                    "'TEST.POST_PROCESSING.DET_WATERSHED_DONUTS_CLASSES' length can't be greater than 'MODEL.N_CLASSES'"
                )
            if np.max(cfg.TEST.POST_PROCESSING.DET_WATERSHED_DONUTS_CLASSES) > cfg.MODEL.N_CLASSES:
                raise ValueError(
                    "'TEST.POST_PROCESSING.DET_WATERSHED_DONUTS_CLASSES' can not have a class number greater than 'MODEL.N_CLASSES'"
                )
            min_class = np.min(cfg.TEST.POST_PROCESSING.DET_WATERSHED_DONUTS_CLASSES)
            if not all(
                cfg.TEST.POST_PROCESSING.DET_WATERSHED_DONUTS_CLASSES
                == np.array(
                    range(
                        min_class,
                        len(cfg.TEST.POST_PROCESSING.DET_WATERSHED_DONUTS_CLASSES) + 1,
                    )
                )
            ):
                raise ValueError(
                    "'TEST.POST_PROCESSING.DET_WATERSHED_DONUTS_CLASSES' must be consecutive, e.g [1,2,3,4..]"
                )
            if len(cfg.TEST.POST_PROCESSING.DET_WATERSHED_DONUTS_PATCH) != dim_count:
                raise ValueError(
                    "'TEST.POST_PROCESSING.DET_WATERSHED_DONUTS_PATCH' needs to be of dimension {} for {} problem".format(
                        dim_count, cfg.PROBLEM.NDIM
                    )
                )

    if not (
        len(cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS)
        == len(cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.VALUES)
        == len(cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.SIGNS)
    ):
        raise ValueError(
            "'TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS', 'TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.VALUES' and "
            "'TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.SIGNS' need to have same length"
        )

    if (
        cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.ENABLE
        and cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.ENABLE
    ):
        if cfg.PROBLEM.TYPE not in ["INSTANCE_SEG", "DETECTION"]:
            raise ValueError(
                "'TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS' can only be used in INSTANCE_SEG and DETECTION workflows"
            )

        if len(cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS) == 0:
            raise ValueError(
                "'TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS' can not be an empty list when "
                "'TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.ENABLE' is enabled"
            )

        for i in range(len(cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS)):
            if not isinstance(
                cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS[i],
                list,
            ):
                raise ValueError(
                    "'TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS' need to be a list of list. E.g. [ ['circularity'], ['area', 'diameter'] ]"
                )
            if not isinstance(
                cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.VALUES[i],
                list,
            ):
                raise ValueError(
                    "'TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.VALUES' need to be a list of list. E.g. [ [10], [15, 3] ]"
                )
            if not isinstance(
                cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.SIGNS[i],
                list,
            ):
                raise ValueError(
                    "'TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.SIGNS' need to be a list of list. E.g. [ ['gt'], ['le', 'gt'] ]"
                )

            if not (
                len(cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS[i])
                == len(cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.VALUES[i])
                == len(cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.SIGNS[i])
            ):
                raise ValueError(
                    "'TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS', 'TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.VALUES' and "
                    "'TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.SIGNS' need to have same length"
                )

            # Check for unique values
            if (
                len(
                    [
                        item
                        for item, count in collections.Counter(
                            cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS[i]
                        ).items()
                        if count > 1
                    ]
                )
                > 0
            ):
                raise ValueError(
                    "Non repeated values are allowed in 'TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES'"
                )
            for j in range(len(cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS[i])):
                if cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS[i][j] not in [
                    "circularity",
                    "npixels",
                    "area",
                    "diameter",
                    "elongation",
                    "sphericity",
                    "perimeter",
                ]:
                    raise ValueError(
                        "'TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS' can only be one among these: ['circularity', 'npixels', 'area', 'diameter', 'elongation', 'sphericity', 'perimeter']"
                    )
                if (
                    cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS[i][j]
                    in ["circularity", "elongation"]
                    and cfg.PROBLEM.NDIM != "2D"
                ):
                    raise ValueError(
                        "'circularity' or 'elongation' properties can only be measured in 2D images. Delete them from 'TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS'. "
                        "'circularity'-kind property in 3D is 'sphericity'"
                    )
                if (
                    cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS[i][j] == "sphericity"
                    and cfg.PROBLEM.NDIM != "3D"
                ):
                    raise ValueError(
                        "'sphericity' property can only be measured in 3D images. Delete it from 'TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS'. "
                        "'sphericity'-kind property in 2D is 'circularity'"
                    )
                if cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.SIGNS[i][j] not in [
                    "gt",
                    "ge",
                    "lt",
                    "le",
                ]:
                    raise ValueError(
                        "'TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.SIGNS' can only be one among these: ['gt', 'ge', 'lt', 'le']"
                    )
                if cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS[i][
                    j
                ] == "circularity" and not check_value(
                    cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.VALUES[i][j]
                ):
                    raise ValueError(
                        "Circularity can only have values in [0, 1] range (check  'TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.VALUES' values)"
                    )

    if cfg.PROBLEM.TYPE != "INSTANCE_SEG":
        if cfg.TEST.POST_PROCESSING.VORONOI_ON_MASK:
            raise ValueError("'TEST.POST_PROCESSING.VORONOI_ON_MASK' can only be enabled in a 'INSTANCE_SEG' problem")
        if cfg.TEST.POST_PROCESSING.CLEAR_BORDER:
            raise ValueError("'TEST.POST_PROCESSING.CLEAR_BORDER' can only be enabled in a 'INSTANCE_SEG' problem")

    if cfg.TEST.POST_PROCESSING.DET_WATERSHED and cfg.PROBLEM.TYPE != "DETECTION":
        raise ValueError("'TEST.POST_PROCESSING.DET_WATERSHED' can only be set when 'PROBLEM.TYPE' is 'DETECTION'")

    if cfg.TEST.POST_PROCESSING.MEDIAN_FILTER:
        if len(cfg.TEST.POST_PROCESSING.MEDIAN_FILTER_AXIS) == 0:
            raise ValueError(
                "Configure 'TEST.POST_PROCESSING.MEDIAN_FILTER_AXIS' as 'TEST.POST_PROCESSING.MEDIAN_FILTER' is enabled"
            )

        if len(cfg.TEST.POST_PROCESSING.MEDIAN_FILTER_SIZE) == 0:
            raise ValueError(
                "Configure 'TEST.POST_PROCESSING.MEDIAN_FILTER_SIZE' as 'TEST.POST_PROCESSING.MEDIAN_FILTER' is enabled"
            )

        assert len(cfg.TEST.POST_PROCESSING.MEDIAN_FILTER_AXIS) == len(
            cfg.TEST.POST_PROCESSING.MEDIAN_FILTER_SIZE
        ), "'TEST.POST_PROCESSING.MEDIAN_FILTER_AXIS' and 'TEST.POST_PROCESSING.MEDIAN_FILTER_SIZE' lenght must be the same"

        if len(cfg.TEST.POST_PROCESSING.MEDIAN_FILTER_AXIS) > 0 and cfg.PROBLEM.TYPE not in [
            "SEMANTIC_SEG",
            "INSTANCE_SEG",
            "DETECTION",
        ]:
            raise ValueError(
                "'TEST.POST_PROCESSING.MEDIAN_FILTER_AXIS' can only be used when 'PROBLEM.TYPE' is among "
                "['SEMANTIC_SEG', 'INSTANCE_SEG', 'DETECTION']"
            )

        for f in cfg.TEST.POST_PROCESSING.MEDIAN_FILTER_AXIS:
            if cfg.PROBLEM.NDIM == "2D" and "z" in f and not cfg.TEST.ANALIZE_2D_IMGS_AS_3D_STACK:
                raise ValueError(
                    "In 2D z axis filtering can not be done unless 'TEST.ANALIZE_2D_IMGS_AS_3D_STACK' is selected. "
                    "So, please, remove it from 'TEST.POST_PROCESSING.MEDIAN_FILTER_AXIS'"
                )
            if f not in ["xy", "yx", "zy", "yz", "zx", "xz", "z"]:
                raise ValueError(
                    "'TEST.POST_PROCESSING.MEDIAN_FILTER_AXIS' options are ['xy', 'yx', 'zy', 'yz', 'zx', 'xz', 'z']"
                )

    # First update is done here as some checks from this point need to have those updates
    if len(opts) > 0:
        cfg.merge_from_list(opts)
        opts = []

    #### General checks ####
    assert cfg.PROBLEM.NDIM in ["2D", "3D"], "Problem needs to be '2D' or '3D'"
    assert cfg.PROBLEM.TYPE in [
        "SEMANTIC_SEG",
        "INSTANCE_SEG",
        "CLASSIFICATION",
        "DETECTION",
        "DENOISING",
        "SUPER_RESOLUTION",
        "SELF_SUPERVISED",
        "IMAGE_TO_IMAGE",
    ], "PROBLEM.TYPE not in ['SEMANTIC_SEG', 'INSTANCE_SEG', 'CLASSIFICATION', 'DETECTION', 'DENOISING', 'SUPER_RESOLUTION', 'SELF_SUPERVISED', 'IMAGE_TO_IMAGE']"

    if cfg.PROBLEM.NDIM == "3D" and cfg.TEST.FULL_IMG:
        print(
            "WARNING: TEST.FULL_IMG == True while using PROBLEM.NDIM == '3D'. As 3D images are usually 'huge'"
            ", full image statistics will be disabled to avoid GPU memory overflow"
        )

    set_train_metrics = True if len(cfg.TRAIN.METRICS) == 0 else False
    set_test_metrics = True if len(cfg.TEST.METRICS) == 0 else False

    if cfg.PROBLEM.TYPE in [
        "SEMANTIC_SEG",
        "INSTANCE_SEG",
        "DETECTION",
    ]:
        if set_train_metrics:
            opts.extend(["TRAIN.METRICS", ["iou"]])
        if set_test_metrics:
            opts.extend(["TEST.METRICS", ["iou"]])

        assert len(cfg.TRAIN.METRICS) == 0 or all(
            [True if x.lower() in ["iou"] else False for x in cfg.TRAIN.METRICS]
        ), f"'TRAIN.METRICS' needs to be 'iou' in {cfg.PROBLEM.TYPE} workflow"

        assert len(cfg.TEST.METRICS) == 0 or all(
            [True if x.lower() in ["iou"] else False for x in cfg.TEST.METRICS]
        ), f"'TEST.METRICS' needs to be 'iou' in {cfg.PROBLEM.TYPE} workflow"

    elif cfg.PROBLEM.TYPE in [
        "SUPER_RESOLUTION",
        "IMAGE_TO_IMAGE",
        "SELF_SUPERVISED",
    ]:
        if set_train_metrics:
            opts.extend(["TRAIN.METRICS", ["psnr", "mae", "mse", "ssim"]])
        if set_test_metrics:
            metric_default_list = ["psnr", "mae", "mse", "ssim"]
            if cfg.PROBLEM.NDIM == "2D":  # IS, FID and LPIPS implementations only works for 2D images
                metric_default_list += ["is", "fid", "lpips"]
            opts.extend(["TEST.METRICS", metric_default_list])

        assert len(cfg.TRAIN.METRICS) == 0 or all(
            [True if x.lower() in ["psnr", "mae", "mse", "ssim"] else False for x in cfg.TRAIN.METRICS]
        ), f"'TRAIN.METRICS' options are ['psnr', 'mae', 'mse', 'ssim'] in {cfg.PROBLEM.TYPE} workflow"
        assert len(cfg.TEST.METRICS) == 0 or all(
            [
                True if x.lower() in ["psnr", "mae", "mse", "ssim", "fid", "is", "lpips"] else False
                for x in cfg.TEST.METRICS
            ]
        ), f"'TEST.METRICS' options are ['psnr', 'mae', 'mse', 'ssim', 'fid', 'is', 'lpips'] in {cfg.PROBLEM.TYPE} workflow"

        if any([True for x in cfg.TEST.METRICS if x.lower() in ["is", "fid", "lpips"]]) and cfg.PROBLEM.NDIM == "3D":
            raise ValueError("IS, FID and LPIPS metrics can only be measured when PROBLEM.NDIM == '3D'")

    elif cfg.PROBLEM.TYPE == "DENOISING":
        if set_train_metrics:
            opts.extend(["TRAIN.METRICS", ["mae", "mse"]])
        if set_test_metrics:
            opts.extend(["TEST.METRICS", ["mae", "mse"]])

        assert len(cfg.TRAIN.METRICS) == 0 or all(
            [True if x.lower() in ["mae", "mse"] else False for x in cfg.TRAIN.METRICS]
        ), f"'TRAIN.METRICS' options are ['mae', 'mse'] in {cfg.PROBLEM.TYPE} workflow"
        assert len(cfg.TEST.METRICS) == 0 or all(
            [True if x.lower() in ["mae", "mse"] else False for x in cfg.TEST.METRICS]
        ), f"'TEST.METRICS' options are ['mae', 'mse'] in {cfg.PROBLEM.TYPE} workflow"

    elif cfg.PROBLEM.TYPE == "CLASSIFICATION":
        if set_train_metrics:
            opts.extend(["TRAIN.METRICS", ["accuracy", "top-5-accuracy"]])
        if set_test_metrics:
            opts.extend(["TEST.METRICS", ["accuracy"]])

        assert len(cfg.TRAIN.METRICS) == 0 or all(
            [True if x.lower() in ["accuracy", "top-5-accuracy"] else False for x in cfg.TRAIN.METRICS]
        ), f"'TRAIN.METRICS' options are ['accuracy', 'top-5-accuracy'] in {cfg.PROBLEM.TYPE} workflow"
        assert len(cfg.TEST.METRICS) == 0 or all(
            [True if x.lower() in ["accuracy"] else False for x in cfg.TEST.METRICS]
        ), f"'TEST.METRICS' options is 'accuracy' in {cfg.PROBLEM.TYPE} workflow"

        if "top-5-accuracy" in [x.lower() for x in cfg.TRAIN.METRICS] and cfg.MODEL.N_CLASSES < 5:
            raise ValueError("'top-5-accuracy' can only be used when MODEL.N_CLASSES >= 5")

    loss = ""
    if cfg.PROBLEM.TYPE in [
        "SEMANTIC_SEG",
        "DETECTION",
    ]:
        loss = "CE" if cfg.LOSS.TYPE == "" else cfg.LOSS.TYPE
        assert loss in [
            "CE",
            "DICE",
            "W_CE_DICE",
        ], "LOSS.TYPE not in ['CE', 'DICE', 'W_CE_DICE']"

        if loss == "W_CE_DICE":
            assert (
                len(cfg.LOSS.WEIGHTS) == 2
            ), "'LOSS.WEIGHTS' needs to be a list of two floats when using LOSS.TYPE == 'W_CE_DICE'"
            assert sum(cfg.LOSS.WEIGHTS) != 1, "'LOSS.WEIGHTS' values need to sum 1"
    elif cfg.PROBLEM.TYPE in [
        "SUPER_RESOLUTION",
        "SELF_SUPERVISED",
        "IMAGE_TO_IMAGE",
    ]:
        loss = "MAE" if cfg.LOSS.TYPE == "" else cfg.LOSS.TYPE
        assert loss in [
            "MAE",
            "MSE",
        ], "LOSS.TYPE not in ['MAE', 'MSE']"
    elif cfg.PROBLEM.TYPE == "DENOISING":
        loss = "MSE" if cfg.LOSS.TYPE == "" else cfg.LOSS.TYPE
        assert loss == "MSE", "LOSS.TYPE must be 'MSE'"
    elif cfg.PROBLEM.TYPE == "CLASSIFICATION":
        loss = "CE" if cfg.LOSS.TYPE == "" else cfg.LOSS.TYPE
        assert loss == "CE", "LOSS.TYPE must be 'CE'"
    opts.extend(["LOSS.TYPE", loss])

    if cfg.TEST.ENABLE and cfg.TEST.ANALIZE_2D_IMGS_AS_3D_STACK and cfg.PROBLEM.NDIM == "3D":
        raise ValueError("'TEST.ANALIZE_2D_IMGS_AS_3D_STACK' makes no sense when the problem is 3D. Disable it.")

    if cfg.MODEL.SOURCE not in ["biapy", "bmz", "torchvision"]:
        raise ValueError("'MODEL.SOURCE' needs to be one between ['biapy', 'bmz', 'torchvision']")

    if cfg.MODEL.SOURCE == "bmz":
        if cfg.MODEL.BMZ.SOURCE_MODEL_ID == "":
            raise ValueError("'MODEL.BMZ.SOURCE_MODEL_ID' needs to be configured when 'MODEL.SOURCE' is 'bmz'")

    elif cfg.MODEL.SOURCE == "torchvision":
        if cfg.MODEL.TORCHVISION_MODEL_NAME == "":
            raise ValueError(
                "'MODEL.TORCHVISION_MODEL_NAME' needs to be configured when 'MODEL.SOURCE' is 'torchvision'"
            )
        if cfg.TEST.AUGMENTATION:
            print("WARNING: 'TEST.AUGMENTATION' is not available using TorchVision models")
        if cfg.TEST.ANALIZE_2D_IMGS_AS_3D_STACK:
            raise ValueError("'TEST.ANALIZE_2D_IMGS_AS_3D_STACK' can not be activated with TorchVision models")
        if cfg.PROBLEM.NDIM == "3D":
            raise ValueError("TorchVision model's are only available for 2D images")

        if not cfg.TEST.FULL_IMG and cfg.PROBLEM.TYPE != "CLASSIFICATION":
            raise ValueError("With TorchVision models only 'TEST.FULL_IMG' setting is available, so please set it")

    if cfg.TEST.AUGMENTATION and cfg.TEST.REDUCE_MEMORY:
        raise ValueError(
            "'TEST.AUGMENTATION' and 'TEST.REDUCE_MEMORY' are incompatible as the function used to make the rotation "
            "does not support float16 data type."
        )

    if cfg.MODEL.N_CLASSES > 2 and cfg.PROBLEM.TYPE not in [
        "SEMANTIC_SEG",
        "INSTANCE_SEG",
        "DETECTION",
        "CLASSIFICATION",
        "IMAGE_TO_IMAGE",
    ]:
        raise ValueError(
            "'MODEL.N_CLASSES' can only be greater than 2 in the following workflows: 'SEMANTIC_SEG', "
            "'INSTANCE_SEG', 'DETECTION', 'CLASSIFICATION' and 'IMAGE_TO_IMAGE'"
        )

    model_arch = cfg.MODEL.ARCHITECTURE.lower()
    model_will_be_read = cfg.MODEL.LOAD_CHECKPOINT and cfg.MODEL.LOAD_MODEL_FROM_CHECKPOINT
    #### Semantic segmentation ####
    if cfg.PROBLEM.TYPE == "SEMANTIC_SEG":
        if not model_will_be_read and cfg.MODEL.SOURCE == "biapy":
            if cfg.MODEL.N_CLASSES < 2:
                raise ValueError("'MODEL.N_CLASSES' needs to be greater or equal 2 (binary case)")
        elif cfg.MODEL.SOURCE == "torchvision":
            if cfg.MODEL.TORCHVISION_MODEL_NAME not in [
                "deeplabv3_mobilenet_v3_large",
                "deeplabv3_resnet101",
                "deeplabv3_resnet50",
                "fcn_resnet101",
                "fcn_resnet50",
                "lraspp_mobilenet_v3_large",
            ]:
                raise ValueError(
                    "'MODEL.SOURCE' must be one between ['deeplabv3_mobilenet_v3_large', 'deeplabv3_resnet101', "
                    "'deeplabv3_resnet50', 'fcn_resnet101', 'fcn_resnet50', 'lraspp_mobilenet_v3_large' ]"
                )
            if cfg.MODEL.TORCHVISION_MODEL_NAME in ["deeplabv3_mobilenet_v3_large"] and cfg.DATA.PATCH_SIZE[-1] != 3:
                raise ValueError(
                    "'deeplabv3_mobilenet_v3_large' model expects 3 channel data (RGB). "
                    f"'DATA.PATCH_SIZE' set is {cfg.DATA.PATCH_SIZE}"
                )

    #### Instance segmentation ####
    if cfg.PROBLEM.TYPE == "INSTANCE_SEG":
        assert cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS in [
            "A",
            "C",
            "BC",
            "BCM",
            "BCD",
            "BCDv2",
            "Dv2",
            "BDv2",
            "BP",
            "BD",
        ], "PROBLEM.INSTANCE_SEG.DATA_CHANNELS not in ['A','C', 'BC', 'BCM', 'BCD', 'BCDv2', 'Dv2', 'BDv2', 'BP', 'BD']"
        if len(cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNEL_WEIGHTS) != channels_provided:
            raise ValueError(
                "'PROBLEM.INSTANCE_SEG.DATA_CHANNEL_WEIGHTS' needs to be of the same length as the channels selected in 'PROBLEM.INSTANCE_SEG.DATA_CHANNELS'. "
                "E.g. 'PROBLEM.INSTANCE_SEG.DATA_CHANNELS'='BC' 'PROBLEM.INSTANCE_SEG.DATA_CHANNEL_WEIGHTS'=[1,0.5]. "
                "'PROBLEM.INSTANCE_SEG.DATA_CHANNELS'='BCD' 'PROBLEM.INSTANCE_SEG.DATA_CHANNEL_WEIGHTS'=[0.5,0.5,1]. "
                "If 'MODEL.N_CLASSES' > 2 one more weigth need to be provided."
            )
        if cfg.TEST.POST_PROCESSING.VORONOI_ON_MASK:
            if cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS not in [
                "C",
                "BC",
                "BCM",
                "BCD",
                "BCDv2",
            ]:
                raise ValueError(
                    "'PROBLEM.INSTANCE_SEG.DATA_CHANNELS' needs to be one between ['C', 'BC', 'BCM', 'BCD', 'BCDv2'] "
                    "when 'TEST.POST_PROCESSING.VORONOI_ON_MASK' is enabled"
                )
            if not check_value(cfg.TEST.POST_PROCESSING.VORONOI_TH):
                raise ValueError("'TEST.POST_PROCESSING.VORONOI_TH' not in [0, 1] range")
        if (
            cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS not in ["C", "BC", "BCM", "BCD", "BP"]
            and cfg.PROBLEM.INSTANCE_SEG.ERODE_AND_DILATE_FOREGROUND
        ):
            raise ValueError(
                "'PROBLEM.INSTANCE_SEG.ERODE_AND_DILATE_FOREGROUND' can only be used with 'C', 'BC', 'BCM', 'BP' or 'BCD' channels"
            )
        for morph_operation in cfg.PROBLEM.INSTANCE_SEG.SEED_MORPH_SEQUENCE:
            if morph_operation != "dilate" and morph_operation != "erode":
                raise ValueError(
                    "'PROBLEM.INSTANCE_SEG.SEED_MORPH_SEQUENCE' can only be a sequence with 'dilate' or 'erode' operations. "
                    "{} given".format(cfg.PROBLEM.INSTANCE_SEG.SEED_MORPH_SEQUENCE)
                )
        if len(cfg.PROBLEM.INSTANCE_SEG.SEED_MORPH_SEQUENCE) != len(cfg.PROBLEM.INSTANCE_SEG.SEED_MORPH_RADIUS):
            raise ValueError(
                "'PROBLEM.INSTANCE_SEG.SEED_MORPH_SEQUENCE' length and 'PROBLEM.INSTANCE_SEG.SEED_MORPH_RADIUS' length needs to be the same"
            )
        if cfg.PROBLEM.INSTANCE_SEG.DATA_CONTOUR_MODE not in [
            "thick",
            "inner",
            "outer",
            "subpixel",
            "dense",
        ]:
            raise ValueError(
                "'PROBLEM.INSTANCE_SEG.DATA_CONTOUR_MODE' must be one between ['thick', 'inner', 'outer', 'subpixel', 'dense']"
            )
        if cfg.PROBLEM.INSTANCE_SEG.DATA_CONTOUR_MODE == "dense" and cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS == "BCM":
            raise ValueError(
                "'PROBLEM.INSTANCE_SEG.DATA_CONTOUR_MODE' can not be 'dense' when 'PROBLEM.INSTANCE_SEG.DATA_CHANNELS' is 'BCM'"
                " as it does not have sense"
            )
        if cfg.PROBLEM.INSTANCE_SEG.WATERSHED_BY_2D_SLICES:
            if cfg.PROBLEM.NDIM == "2D" and not cfg.TEST.ANALIZE_2D_IMGS_AS_3D_STACK:
                raise ValueError(
                    "'PROBLEM.INSTANCE_SEG.WATERSHED_BY_2D_SLICE' can only be activated when 'PROBLEM.NDIM' == 3D or "
                    "in 2D when 'TEST.ANALIZE_2D_IMGS_AS_3D_STACK' is enabled"
                )
        if cfg.MODEL.SOURCE == "torchvision":
            if cfg.MODEL.TORCHVISION_MODEL_NAME not in [
                "maskrcnn_resnet50_fpn",
                "maskrcnn_resnet50_fpn_v2",
            ]:
                raise ValueError(
                    "'MODEL.SOURCE' must be one between ['maskrcnn_resnet50_fpn', 'maskrcnn_resnet50_fpn_v2']"
                )
            if cfg.PROBLEM.NDIM == "3D":
                raise ValueError("TorchVision model's for instance segmentation are only available for 2D images")
            if cfg.TRAIN.ENABLE:
                raise NotImplementedError  # require bbox generator etc.

    #### Detection ####
    if cfg.PROBLEM.TYPE == "DETECTION":
        if not model_will_be_read and cfg.MODEL.SOURCE == "biapy" and cfg.MODEL.N_CLASSES < 2:
            raise ValueError("'MODEL.N_CLASSES' needs to be greater or equal 2 (binary case)")

        cpd = cfg.PROBLEM.DETECTION.CENTRAL_POINT_DILATION
        if len(cpd) == 1:
            cpd = cpd * 2 if cfg.PROBLEM.NDIM == "2D" else cpd * 3

        if len(cpd) != 3 and cfg.PROBLEM.NDIM == "3D":
            raise ValueError(
                "'PROBLEM.DETECTION.CENTRAL_POINT_DILATION' needs to be a list of three ints in a 3D problem"
            )
        elif len(cpd) != 2 and cfg.PROBLEM.NDIM == "2D":
            raise ValueError(
                "'PROBLEM.DETECTION.CENTRAL_POINT_DILATION' needs to be a list of two ints in a 2D problem"
            )

        opts.extend(["PROBLEM.DETECTION.CENTRAL_POINT_DILATION", cpd])

        if cfg.TEST.POST_PROCESSING.DET_WATERSHED:
            if any(len(x) != dim_count for x in cfg.TEST.POST_PROCESSING.DET_WATERSHED_FIRST_DILATION):
                raise ValueError(
                    "Each structure object defined in 'TEST.POST_PROCESSING.DET_WATERSHED_FIRST_DILATION' "
                    "needs to be of {} dimension".format(dim_count)
                )
            if (
                not cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.ENABLE
                or not cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.ENABLE
            ):
                raise ValueError(
                    "'TEST.POST_PROCESSING.MEASURE_PROPERTIES.ENABLE' and "
                    "'TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.ENABLE' needs to be set when 'TEST.POST_PROCESSING.DET_WATERSHED' is enabled"
                )
            for lprop in cfg.TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS:
                if len(lprop) != 1:
                    raise ValueError(
                        "'TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS' can not be set with more than one property and that property"
                        " needs to be set to 'circularity' or 'sphericity'. This restriction is because 'TEST.POST_PROCESSING.DET_WATERSHED' is enabled"
                    )
                if lprop[0] not in ["circularity", "sphericity"]:
                    raise ValueError(
                        "Only 'circularity' or 'sphericity' can be used in 'TEST.POST_PROCESSING.MEASURE_PROPERTIES.REMOVE_BY_PROPERTIES.PROPS' "
                        "when 'TEST.POST_PROCESSING.DET_WATERSHED' is enabled"
                    )
        if cfg.TEST.DET_POINT_CREATION_FUNCTION not in ["peak_local_max", "blob_log"]:
            raise ValueError("'TEST.DET_POINT_CREATION_FUNCTION' must be one between: ['peak_local_max', 'blob_log']")
        if cfg.MODEL.SOURCE == "torchvision":
            if cfg.MODEL.TORCHVISION_MODEL_NAME not in [
                "fasterrcnn_mobilenet_v3_large_320_fpn",
                "fasterrcnn_mobilenet_v3_large_fpn",
                "fasterrcnn_resnet50_fpn",
                "fasterrcnn_resnet50_fpn_v2",
                "fcos_resnet50_fpn",
                "ssd300_vgg16",
                "ssdlite320_mobilenet_v3_large",
                "retinanet_resnet50_fpn",
                "retinanet_resnet50_fpn_v2",
            ]:
                raise ValueError(
                    "'MODEL.SOURCE' must be one between ['fasterrcnn_mobilenet_v3_large_320_fpn', 'fasterrcnn_mobilenet_v3_large_fpn', "
                    "'fasterrcnn_resnet50_fpn', 'fasterrcnn_resnet50_fpn_v2', 'fcos_resnet50_fpn', 'ssd300_vgg16', 'ssdlite320_mobilenet_v3_large', "
                    "'retinanet_resnet50_fpn', 'retinanet_resnet50_fpn_v2']"
                )
            if cfg.PROBLEM.NDIM == "3D":
                raise ValueError("TorchVision model's for detection are only available for 2D images")
            if cfg.TRAIN.ENABLE:
                raise NotImplementedError  # require bbox generator etc.

        if cfg.TEST.ENABLE and len(cfg.TEST.DET_IGNORE_POINTS_OUTSIDE_BOX) > 0:
            assert [x > 0 for x in cfg.TEST.DET_IGNORE_POINTS_OUTSIDE_BOX], (
                "'TEST.DET_IGNORE_POINTS_OUTSIDE_BOX' needs to be a list " "of positive integers"
            )
            assert len(cfg.TEST.DET_IGNORE_POINTS_OUTSIDE_BOX) == dim_count, (
                "'TEST.DET_IGNORE_POINTS_OUTSIDE_BOX' needs to be of " f"{dim_count} dimension"
            )

    #### Super-resolution ####
    elif cfg.PROBLEM.TYPE == "SUPER_RESOLUTION":
        if not (cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING):
            raise ValueError("Resolution scale must be provided with 'PROBLEM.SUPER_RESOLUTION.UPSCALING' variable")
        assert all(
            i > 0 for i in cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING
        ), "'PROBLEM.SUPER_RESOLUTION.UPSCALING' are not positive integers"
        if len(cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING) != dim_count:
            raise ValueError(f"'PROBLEM.SUPER_RESOLUTION.UPSCALING' needs to be a tuple of {dim_count} integers")
        if cfg.MODEL.SOURCE == "torchvision":
            raise ValueError("'MODEL.SOURCE' as 'torchvision' is not available in super-resolution workflow")
        if cfg.DATA.NORMALIZATION.TYPE not in ["div", "scale_range"]:
            raise ValueError("'DATA.NORMALIZATION.TYPE' in SR workflow needs to be one between ['div','scale_range']")

    #### Self-supervision ####
    elif cfg.PROBLEM.TYPE == "SELF_SUPERVISED":
        if cfg.PROBLEM.SELF_SUPERVISED.PRETEXT_TASK == "crappify":
            if cfg.PROBLEM.SELF_SUPERVISED.RESIZING_FACTOR not in [2, 4, 6]:
                raise ValueError("'PROBLEM.SELF_SUPERVISED.RESIZING_FACTOR' not in [2,4,6]")
            if not check_value(cfg.PROBLEM.SELF_SUPERVISED.NOISE):
                raise ValueError("'PROBLEM.SELF_SUPERVISED.NOISE' not in [0, 1] range")
            if not model_will_be_read and model_arch == "mae":
                raise ValueError(
                    "'MODEL.ARCHITECTURE' can not be 'mae' when 'PROBLEM.SELF_SUPERVISED.PRETEXT_TASK' is 'crappify'"
                )
        elif cfg.PROBLEM.SELF_SUPERVISED.PRETEXT_TASK == "masking":
            if not model_will_be_read and model_arch != "mae":
                raise ValueError(
                    "'MODEL.ARCHITECTURE' needs to be 'mae' when 'PROBLEM.SELF_SUPERVISED.PRETEXT_TASK' is 'masking'"
                )
            assert cfg.MODEL.MAE_MASK_TYPE in [
                "random",
                "grid",
            ], "'MODEL.MAE_MASK_TYPE' needs to be one between ['random', 'grid']"
            if cfg.MODEL.MAE_MASK_TYPE == "random" and not check_value(cfg.MODEL.MAE_MASK_RATIO):
                raise ValueError("'MODEL.MAE_MASK_RATIO' not in [0, 1] range")
        else:
            raise ValueError(
                "'PROBLEM.SELF_SUPERVISED.PRETEXT_TASK' needs to be among these options: ['crappify', 'masking']"
            )
        if cfg.MODEL.SOURCE == "torchvision":
            raise ValueError("'MODEL.SOURCE' as 'torchvision' is not available in self-supervised workflow")

    #### Denoising ####
    elif cfg.PROBLEM.TYPE == "DENOISING":
        if cfg.DATA.TEST.LOAD_GT:
            raise ValueError(
                "Denoising is made in an unsupervised way so there is no ground truth required. Disable 'DATA.TEST.LOAD_GT'"
            )
        if not check_value(cfg.PROBLEM.DENOISING.N2V_PERC_PIX):
            raise ValueError("PROBLEM.DENOISING.N2V_PERC_PIX not in [0, 1] range")
        if cfg.MODEL.SOURCE == "torchvision":
            raise ValueError("'MODEL.SOURCE' as 'torchvision' is not available in denoising workflow")

    #### Classification ####
    elif cfg.PROBLEM.TYPE == "CLASSIFICATION":
        if cfg.TEST.BY_CHUNKS.ENABLE:
            raise ValueError("'TEST.BY_CHUNKS.ENABLE' can not be activated for CLASSIFICATION workflow")
        if cfg.MODEL.SOURCE == "torchvision":
            if cfg.MODEL.TORCHVISION_MODEL_NAME not in [
                "alexnet",
                "convnext_base",
                "convnext_large",
                "convnext_small",
                "convnext_tiny",
                "densenet121",
                "densenet161",
                "densenet169",
                "densenet201",
                "efficientnet_b0",
                "efficientnet_b1",
                "efficientnet_b2",
                "efficientnet_b3",
                "efficientnet_b4",
                "efficientnet_b5",
                "efficientnet_b6",
                "efficientnet_b7",
                "efficientnet_v2_l",
                "efficientnet_v2_m",
                "efficientnet_v2_s",
                "googlenet",
                "inception_v3",
                "maxvit_t",
                "mnasnet0_5",
                "mnasnet0_75",
                "mnasnet1_0",
                "mnasnet1_3",
                "mobilenet_v2",
                "mobilenet_v3_large",
                "mobilenet_v3_small",
                "quantized_googlenet",
                "quantized_inception_v3",
                "quantized_mobilenet_v2",
                "quantized_mobilenet_v3_large",
                "quantized_resnet18",
                "quantized_resnet50",
                "quantized_resnext101_32x8d",
                "quantized_resnext101_64x4d",
                "quantized_shufflenet_v2_x0_5",
                "quantized_shufflenet_v2_x1_0",
                "quantized_shufflenet_v2_x1_5",
                "quantized_shufflenet_v2_x2_0",
                "regnet_x_16gf",
                "regnet_x_1_6gf",
                "regnet_x_32gf",
                "regnet_x_3_2gf",
                "regnet_x_400mf",
                "regnet_x_800mf",
                "regnet_x_8gf",
                "regnet_y_128gf",
                "regnet_y_16gf",
                "regnet_y_1_6gf",
                "regnet_y_32gf",
                "regnet_y_3_2gf",
                "regnet_y_400mf",
                "regnet_y_800mf",
                "regnet_y_8gf",
                "resnet101",
                "resnet152",
                "resnet18",
                "resnet34",
                "resnet50",
                "resnext101_32x8d",
                "resnext101_64x4d",
                "resnext50_32x4d",
                "retinanet_resnet50_fpn",
                "shufflenet_v2_x0_5",
                "shufflenet_v2_x1_0",
                "shufflenet_v2_x1_5",
                "shufflenet_v2_x2_0",
                "squeezenet1_0",
                "squeezenet1_1",
                "swin_b",
                "swin_s",
                "swin_t",
                "swin_v2_b",
                "swin_v2_s",
                "swin_v2_t",
                "vgg11",
                "vgg11_bn",
                "vgg13",
                "vgg13_bn",
                "vgg16",
                "vgg16_bn",
                "vgg19",
                "vgg19_bn",
                "vit_b_16",
                "vit_b_32",
                "vit_h_14",
                "vit_l_16",
                "vit_l_32",
                "wide_resnet101_2",
                "wide_resnet50_2",
            ]:
                raise ValueError(
                    "'MODEL.SOURCE' must be one between [ "
                    "'alexnet', 'convnext_base', 'convnext_large', 'convnext_small', 'convnext_tiny', 'densenet121', 'densenet161', "
                    "'densenet169', 'densenet201', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', "
                    "'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 'efficientnet_v2_l', 'efficientnet_v2_m', "
                    "'efficientnet_v2_s', 'googlenet', 'inception_v3', 'maxvit_t', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3', "
                    "'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small',  'quantized_googlenet', 'quantized_inception_v3', "
                    "'quantized_mobilenet_v2', 'quantized_mobilenet_v3_large', 'quantized_resnet18', 'quantized_resnet50', "
                    "'quantized_resnext101_32x8d', 'quantized_resnext101_64x4d', 'quantized_shufflenet_v2_x0_5', 'quantized_shufflenet_v2_x1_0', "
                    "'quantized_shufflenet_v2_x1_5', 'quantized_shufflenet_v2_x2_0', 'regnet_x_16gf', 'regnet_x_1_6gf', 'regnet_x_32gf', "
                    "'regnet_x_3_2gf', 'regnet_x_400mf', 'regnet_x_800mf', 'regnet_x_8gf', 'regnet_y_128gf', 'regnet_y_16gf', 'regnet_y_1_6gf', "
                    "'regnet_y_32gf', 'regnet_y_3_2gf', 'regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_8gf', 'resnet101', 'resnet152', "
                    "'resnet18', 'resnet34', 'resnet50', 'resnext101_32x8d', 'resnext101_64x4d', 'resnext50_32x4d', 'retinanet_resnet50_fpn', "
                    "'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', "
                    "'squeezenet1_0', 'squeezenet1_1', 'swin_b', 'swin_s', 'swin_t', 'swin_v2_b', 'swin_v2_s', 'swin_v2_t', "
                    "'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'vit_b_16', 'vit_b_32', "
                    "'vit_h_14', 'vit_l_16', 'vit_l_32', 'wide_resnet101_2', 'wide_resnet50_2' "
                    "]"
                )

    #### Image to image ####
    elif cfg.PROBLEM.TYPE == "IMAGE_TO_IMAGE":
        if cfg.MODEL.SOURCE == "torchvision":
            raise ValueError("'MODEL.SOURCE' as 'torchvision' is not available in image to image workflow")
        if cfg.PROBLEM.IMAGE_TO_IMAGE.MULTIPLE_RAW_ONE_TARGET_LOADER:
            if cfg.TRAIN.ENABLE and cfg.DATA.TRAIN.FILTER_SAMPLES.ENABLE:
                raise ValueError(
                    "'DATA.TRAIN.FILTER_SAMPLES.ENABLE' can not be enabled when 'PROBLEM.IMAGE_TO_IMAGE.MULTIPLE_RAW_ONE_TARGET_LOADER' is enabled too"
                )

            if cfg.TRAIN.ENABLE and cfg.DATA.VAL.FILTER_SAMPLES.ENABLE:
                raise ValueError(
                    "'DATA.VAL.FILTER_SAMPLES.ENABLE' can not be enabled when 'PROBLEM.IMAGE_TO_IMAGE.MULTIPLE_RAW_ONE_TARGET_LOADER' is enabled too"
                )

    if cfg.DATA.EXTRACT_RANDOM_PATCH and cfg.DATA.PROBABILITY_MAP:
        if cfg.DATA.W_FOREGROUND + cfg.DATA.W_BACKGROUND != 1:
            raise ValueError(
                "cfg.DATA.W_FOREGROUND+cfg.DATA.W_BACKGROUND need to sum 1. E.g. 0.94 and 0.06 respectively."
            )
    if cfg.DATA.VAL.FROM_TRAIN and cfg.DATA.PREPROCESS.VAL:
        print(
            "WARNING: validation preprocessing will be done based on 'DATA.PREPROCESS.TRAIN', as 'DATA.VAL.FROM_TRAIN' is selected"
        )

    ### Pre-processing ###
    if cfg.DATA.PREPROCESS.TRAIN or cfg.DATA.PREPROCESS.TEST or cfg.DATA.PREPROCESS.VAL:
        if cfg.DATA.PREPROCESS.RESIZE.ENABLE:
            if cfg.PROBLEM.TYPE == "DETECTION":
                raise ValueError("Resizing preprocessing is not available for the DETECTION workflow.")
            if cfg.PROBLEM.NDIM == "3D":
                if cfg.DATA.PREPROCESS.RESIZE.OUTPUT_SHAPE == (512, 512):
                    opts.extend(["DATA.PREPROCESS.RESIZE.OUTPUT_SHAPE", (512, 512, 512)])
                elif len(cfg.DATA.PREPROCESS.RESIZE.OUTPUT_SHAPE) != 3:
                    raise ValueError(
                        "When 'PROBLEM.NDIM' is 3D, 'DATA.PREPROCESS.RESIZE.OUTPUT_SHAPE' must indicate desired size for each dimension."
                        f"Given shape ({cfg.DATA.PREPROCESS.RESIZE.OUTPUT_SHAPE}) is not compatible."
                    )
            if cfg.PROBLEM.NDIM == "2D" and len(cfg.DATA.PREPROCESS.RESIZE.OUTPUT_SHAPE) != 2:
                raise ValueError(
                    "When 'PROBLEM.NDIM' is 2D, 'DATA.PREPROCESS.RESIZE.OUTPUT_SHAPE' must indicate desired size for each dimension."
                    f"Given shape ({cfg.DATA.PREPROCESS.RESIZE.OUTPUT_SHAPE}) is not compatible."
                )
            for i, s in enumerate(cfg.DATA.PREPROCESS.RESIZE.OUTPUT_SHAPE):
                if cfg.DATA.PATCH_SIZE[i] > s:
                    raise ValueError(
                        f"'DATA.PREPROCESS.RESIZE.OUTPUT_SHAPE' {cfg.DATA.PREPROCESS.RESIZE.OUTPUT_SHAPE} can not be smaller than 'DATA.PATCH_SIZE' {cfg.DATA.PATCH_SIZE}."
                    )
        if cfg.DATA.PREPROCESS.CANNY.ENABLE and cfg.PROBLEM.NDIM != "2D":
            raise ValueError("Canny or edge detection can not be activated when 'PROBLEM.NDIM' is 2D.")
        if cfg.DATA.PREPROCESS.MEDIAN_BLUR.ENABLE:
            if cfg.PROBLEM.NDIM == "2D" and len(cfg.DATA.PREPROCESS.MEDIAN_BLUR.KERNEL_SIZE) != 3:
                raise ValueError(
                    "When 'PROBLEM.NDIM' is 2D, 'DATA.PREPROCESS.MEDIAN_BLUR.KERNEL_SIZE' must indicate desired kernel size for each dimension, including channels (y,x,c)."
                    f"Given kernel size ({cfg.DATA.PREPROCESS.MEDIAN_BLUR.KERNEL_SIZE}) is not compatible."
                )
            elif cfg.PROBLEM.NDIM == "3D" and len(cfg.DATA.PREPROCESS.MEDIAN_BLUR.KERNEL_SIZE) != 4:
                raise ValueError(
                    "When 'PROBLEM.NDIM' is 3D, 'DATA.PREPROCESS.MEDIAN_BLUR.KERNEL_SIZE' must indicate desired kernel size for each dimension, including channels (z,y,x,c)."
                    f"Given kernel size ({cfg.DATA.PREPROCESS.MEDIAN_BLUR.KERNEL_SIZE}) is not compatible."
                )
        if cfg.DATA.PREPROCESS.MATCH_HISTOGRAM.ENABLE:
            if not os.path.exists(cfg.DATA.PREPROCESS.MATCH_HISTOGRAM.REFERENCE_PATH):
                raise ValueError(
                    f"Path pointed by 'DATA.PREPROCESS.MATCH_HISTOGRAM.REFERENCE_PATH' does not exist: {cfg.DATA.PREPROCESS.MATCH_HISTOGRAM.REFERENCE_PATH}"
                )
        if cfg.DATA.PREPROCESS.ZOOM.ENABLE and not cfg.TEST.BY_CHUNKS.ENABLE:
            raise ValueError("'DATA.PREPROCESS.ZOOM.ENABLE' can only be activated when 'TEST.BY_CHUNKS.ENABLE' is True")
        if cfg.DATA.PREPROCESS.ZOOM.ENABLE and len(cfg.DATA.PREPROCESS.ZOOM.ZOOM_FACTOR) != len(
            cfg.TEST.BY_CHUNKS.INPUT_IMG_AXES_ORDER
        ):
            raise ValueError(
                "'DATA.PREPROCESS.ZOOM.ZOOM_FACTOR' needs to have the same length as 'TEST.BY_CHUNKS.INPUT_IMG_AXES_ORDER'"
            )

    #### Data ####
    if cfg.TRAIN.ENABLE:
        if check_data_paths:
            if not os.path.exists(cfg.DATA.TRAIN.PATH):
                raise ValueError("Train data dir not found: {}".format(cfg.DATA.TRAIN.PATH))
            if (
                not os.path.exists(cfg.DATA.TRAIN.GT_PATH)
                and cfg.PROBLEM.TYPE not in ["DENOISING", "CLASSIFICATION", "SELF_SUPERVISED"]
                and not cfg.DATA.TRAIN.INPUT_ZARR_MULTIPLE_DATA
            ):
                raise ValueError("Train mask data dir not found: {}".format(cfg.DATA.TRAIN.GT_PATH))
            if not cfg.DATA.VAL.FROM_TRAIN:
                if not os.path.exists(cfg.DATA.VAL.PATH):
                    raise ValueError("Validation data dir not found: {}".format(cfg.DATA.VAL.PATH))
                if (
                    not os.path.exists(cfg.DATA.VAL.GT_PATH)
                    and cfg.PROBLEM.TYPE not in ["DENOISING", "CLASSIFICATION", "SELF_SUPERVISED"]
                    and not cfg.DATA.VAL.INPUT_ZARR_MULTIPLE_DATA
                ):
                    raise ValueError("Validation mask data dir not found: {}".format(cfg.DATA.VAL.GT_PATH))
        if cfg.DATA.TRAIN.INPUT_ZARR_MULTIPLE_DATA:
            if cfg.PROBLEM.NDIM != "3D":
                raise ValueError("'DATA.TRAIN.INPUT_ZARR_MULTIPLE_DATA' to True is only implemented in 3D workflows")
            if (
                cfg.DATA.TRAIN.INPUT_ZARR_MULTIPLE_DATA_RAW_PATH == ""
                or cfg.DATA.TRAIN.INPUT_ZARR_MULTIPLE_DATA_GT_PATH == ""
            ):
                raise ValueError(
                    "'DATA.TRAIN.INPUT_ZARR_MULTIPLE_DATA_RAW_PATH' and 'DATA.TRAIN.INPUT_ZARR_MULTIPLE_DATA_GT_PATH' "
                    "need to be set when 'DATA.TRAIN.INPUT_ZARR_MULTIPLE_DATA' is used."
                )
        if cfg.DATA.VAL.INPUT_ZARR_MULTIPLE_DATA:
            if cfg.PROBLEM.NDIM != "3D":
                raise ValueError("'DATA.VAL.INPUT_ZARR_MULTIPLE_DATA' to True is only implemented in 3D workflows")
            if (
                cfg.DATA.VAL.INPUT_ZARR_MULTIPLE_DATA_RAW_PATH == ""
                or cfg.DATA.VAL.INPUT_ZARR_MULTIPLE_DATA_GT_PATH == ""
            ):
                raise ValueError(
                    "'DATA.VAL.INPUT_ZARR_MULTIPLE_DATA_RAW_PATH' and 'DATA.VAL.INPUT_ZARR_MULTIPLE_DATA_GT_PATH' "
                    "need to be set when 'DATA.VAL.INPUT_ZARR_MULTIPLE_DATA' is used."
                )

    if cfg.TEST.ENABLE and not cfg.DATA.TEST.USE_VAL_AS_TEST and check_data_paths:
        if not os.path.exists(cfg.DATA.TEST.PATH):
            raise ValueError("Test data not found: {}".format(cfg.DATA.TEST.PATH))
        if (
            cfg.DATA.TEST.LOAD_GT
            and not os.path.exists(cfg.DATA.TEST.GT_PATH)
            and cfg.PROBLEM.TYPE not in ["CLASSIFICATION", "SELF_SUPERVISED"]
            and not cfg.TEST.BY_CHUNKS.INPUT_ZARR_MULTIPLE_DATA
        ):
            raise ValueError("Test data mask not found: {}".format(cfg.DATA.TEST.GT_PATH))
    if cfg.TEST.ENABLE and cfg.TEST.BY_CHUNKS.ENABLE:
        if cfg.PROBLEM.NDIM == "2D":
            raise ValueError("'TEST.BY_CHUNKS' can not be activated when 'PROBLEM.NDIM' is 2D")
        assert cfg.TEST.BY_CHUNKS.FORMAT.lower() in [
            "h5",
            "zarr",
        ], "'TEST.BY_CHUNKS.FORMAT' needs to be one between ['H5', 'Zarr']"
        opts.extend(["TEST.BY_CHUNKS.FORMAT", cfg.TEST.BY_CHUNKS.FORMAT.lower()])
        if cfg.TEST.BY_CHUNKS.WORKFLOW_PROCESS.ENABLE:
            assert cfg.TEST.BY_CHUNKS.WORKFLOW_PROCESS.TYPE in [
                "chunk_by_chunk",
                "entire_pred",
            ], "'TEST.BY_CHUNKS.WORKFLOW_PROCESS.TYPE' needs to be one between ['chunk_by_chunk', 'entire_pred']"
        if len(cfg.TEST.BY_CHUNKS.INPUT_IMG_AXES_ORDER) < 3:
            raise ValueError("'TEST.BY_CHUNKS.INPUT_IMG_AXES_ORDER' needs to be at least of length 3, e.g., 'ZYX'")
        if cfg.MODEL.N_CLASSES > 2:
            raise ValueError("Not implemented pipeline option: 'MODEL.N_CLASSES' > 2 and 'TEST.BY_CHUNKS'")
        if cfg.TEST.BY_CHUNKS.INPUT_ZARR_MULTIPLE_DATA:
            if cfg.TEST.BY_CHUNKS.INPUT_ZARR_MULTIPLE_DATA_RAW_PATH == "":
                raise ValueError(
                    "'TEST.BY_CHUNKS.INPUT_ZARR_MULTIPLE_DATA_RAW_PATH' needs to be set when "
                    "'TEST.BY_CHUNKS.INPUT_ZARR_MULTIPLE_DATA' is used."
                )
            if cfg.DATA.TEST.LOAD_GT and cfg.TEST.BY_CHUNKS.INPUT_ZARR_MULTIPLE_DATA_GT_PATH == "":
                raise ValueError(
                    "'TEST.BY_CHUNKS.INPUT_ZARR_MULTIPLE_DATA_GT_PATH' needs to be set when "
                    "'TEST.BY_CHUNKS.INPUT_ZARR_MULTIPLE_DATA' is used."
                )

    if cfg.TRAIN.ENABLE:
        if cfg.DATA.EXTRACT_RANDOM_PATCH and cfg.DATA.PROBABILITY_MAP:
            if not cfg.PROBLEM.TYPE == "SEMANTIC_SEG":
                raise ValueError("'DATA.PROBABILITY_MAP' can only be selected when 'PROBLEM.TYPE' is 'SEMANTIC_SEG'")

        if cfg.DATA.VAL.FROM_TRAIN and not cfg.DATA.VAL.CROSS_VAL and cfg.DATA.VAL.SPLIT_TRAIN <= 0:
            raise ValueError("'DATA.VAL.SPLIT_TRAIN' needs to be > 0 when 'DATA.VAL.FROM_TRAIN' == True")

        if cfg.PROBLEM.NDIM == "2D" and cfg.DATA.TRAIN.INPUT_IMG_AXES_ORDER != "TZCYX":
            raise ValueError("'DATA.TRAIN.INPUT_IMG_AXES_ORDER' can not be set in 2D problems")
        if cfg.PROBLEM.NDIM == "2D" and cfg.DATA.TRAIN.INPUT_MASK_AXES_ORDER != "TZCYX":
            raise ValueError("'DATA.TRAIN.INPUT_MASK_AXES_ORDER' can not be set in 2D problems")
        if len(cfg.DATA.TRAIN.INPUT_IMG_AXES_ORDER) < 3:
            raise ValueError("'DATA.TRAIN.INPUT_IMG_AXES_ORDER' needs to be at least of length 3, e.g., 'ZYX'")
        if len(cfg.DATA.TRAIN.INPUT_MASK_AXES_ORDER) < 3:
            raise ValueError("'DATA.TRAIN.INPUT_MASK_AXES_ORDER' needs to be at least of length 3, e.g., 'ZYX'")

        if cfg.PROBLEM.NDIM == "2D" and cfg.DATA.VAL.INPUT_IMG_AXES_ORDER != "TZCYX":
            raise ValueError("'DATA.VAL.INPUT_IMG_AXES_ORDER' can not be set in 2D problems")
        if cfg.PROBLEM.NDIM == "2D" and cfg.DATA.VAL.INPUT_MASK_AXES_ORDER != "TZCYX":
            raise ValueError("'DATA.VAL.INPUT_MASK_AXES_ORDER' can not be set in 2D problems")
        if len(cfg.DATA.VAL.INPUT_IMG_AXES_ORDER) < 3:
            raise ValueError("'DATA.VAL.INPUT_IMG_AXES_ORDER' needs to be at least of length 3, e.g., 'ZYX'")
        if len(cfg.DATA.VAL.INPUT_MASK_AXES_ORDER) < 3:
            raise ValueError("'DATA.VAL.INPUT_MASK_AXES_ORDER' needs to be at least of length 3, e.g., 'ZYX'")

    if cfg.DATA.VAL.CROSS_VAL:
        if not cfg.DATA.VAL.FROM_TRAIN:
            raise ValueError("'DATA.VAL.CROSS_VAL' can only be used when 'DATA.VAL.FROM_TRAIN' is True")
        if cfg.DATA.VAL.CROSS_VAL_NFOLD < cfg.DATA.VAL.CROSS_VAL_FOLD:
            raise ValueError("'DATA.VAL.CROSS_VAL_NFOLD' can not be less than 'DATA.VAL.CROSS_VAL_FOLD'")
    if cfg.DATA.TEST.USE_VAL_AS_TEST and not cfg.DATA.VAL.CROSS_VAL:
        raise ValueError("'DATA.TEST.USE_VAL_AS_TEST' can only be used when 'DATA.VAL.CROSS_VAL' is selected")
    if len(cfg.DATA.TRAIN.RESOLUTION) != 1 and len(cfg.DATA.TRAIN.RESOLUTION) != dim_count:
        raise ValueError(
            "When PROBLEM.NDIM == {} DATA.TRAIN.RESOLUTION tuple must be length {}, given {}.".format(
                cfg.PROBLEM.NDIM, dim_count, cfg.DATA.TRAIN.RESOLUTION
            )
        )
    if len(cfg.DATA.VAL.RESOLUTION) != 1 and len(cfg.DATA.VAL.RESOLUTION) != dim_count:
        raise ValueError(
            "When PROBLEM.NDIM == {} DATA.VAL.RESOLUTION tuple must be length {}, given {}.".format(
                cfg.PROBLEM.NDIM, dim_count, cfg.DATA.VAL.RESOLUTION
            )
        )
    if cfg.TEST.ANALIZE_2D_IMGS_AS_3D_STACK and cfg.PROBLEM.TYPE == "INSTANCE_SEG":
        if len(cfg.DATA.TEST.RESOLUTION) != 2 and len(cfg.DATA.TEST.RESOLUTION) != 3:
            raise ValueError(
                "'DATA.TEST.RESOLUTION' needs to be a tuple with 2 or 3 values (both valid because "
                "'TEST.ANALIZE_2D_IMGS_AS_3D_STACK' is activated in this case)".format(dim_count)
            )
    else:
        if len(cfg.DATA.TEST.RESOLUTION) != 1 and len(cfg.DATA.TEST.RESOLUTION) != dim_count:
            raise ValueError(
                "When PROBLEM.NDIM == {} DATA.TEST.RESOLUTION tuple must be length {}, given {}.".format(
                    cfg.PROBLEM.NDIM, dim_count, cfg.DATA.TEST.RESOLUTION
                )
            )

    if len(cfg.DATA.TRAIN.OVERLAP) != dim_count:
        raise ValueError(
            "When PROBLEM.NDIM == {} DATA.TRAIN.OVERLAP tuple must be length {}, given {}.".format(
                cfg.PROBLEM.NDIM, dim_count, cfg.DATA.TRAIN.OVERLAP
            )
        )
    if any(not check_value(x) for x in cfg.DATA.TRAIN.OVERLAP):
        raise ValueError("DATA.TRAIN.OVERLAP not in [0, 1] range")
    if len(cfg.DATA.TRAIN.PADDING) != dim_count:
        raise ValueError(
            "When PROBLEM.NDIM == {} DATA.TRAIN.PADDING tuple must be length {}, given {}.".format(
                cfg.PROBLEM.NDIM, dim_count, cfg.DATA.TRAIN.PADDING
            )
        )
    if len(cfg.DATA.VAL.OVERLAP) != dim_count:
        raise ValueError(
            "When PROBLEM.NDIM == {} DATA.VAL.OVERLAP tuple must be length {}, given {}.".format(
                cfg.PROBLEM.NDIM, dim_count, cfg.DATA.VAL.OVERLAP
            )
        )
    if any(not check_value(x) for x in cfg.DATA.VAL.OVERLAP):
        raise ValueError("DATA.VAL.OVERLAP not in [0, 1] range")
    if len(cfg.DATA.VAL.PADDING) != dim_count:
        raise ValueError(
            "When PROBLEM.NDIM == {} DATA.VAL.PADDING tuple must be length {}, given {}.".format(
                cfg.PROBLEM.NDIM, dim_count, cfg.DATA.VAL.PADDING
            )
        )
    if len(cfg.DATA.TEST.OVERLAP) != dim_count:
        raise ValueError(
            "When PROBLEM.NDIM == {} DATA.TEST.OVERLAP tuple must be length {}, given {}.".format(
                cfg.PROBLEM.NDIM, dim_count, cfg.DATA.TEST.OVERLAP
            )
        )
    if any(not check_value(x) for x in cfg.DATA.TEST.OVERLAP):
        raise ValueError("DATA.TEST.OVERLAP not in [0, 1] range")
    if len(cfg.DATA.TEST.PADDING) != dim_count:
        raise ValueError(
            "When PROBLEM.NDIM == {} DATA.TEST.PADDING tuple must be length {}, given {}.".format(
                cfg.PROBLEM.NDIM, dim_count, cfg.DATA.TEST.PADDING
            )
        )
    if len(cfg.DATA.PATCH_SIZE) != dim_count + 1:
        if cfg.MODEL.SOURCE != "bmz":
            raise ValueError(
                "When PROBLEM.NDIM == {} DATA.PATCH_SIZE tuple must be length {}, given {}.".format(
                    cfg.PROBLEM.NDIM, dim_count + 1, cfg.DATA.PATCH_SIZE
                )
            )
        else:
            print(
                "WARNING: when PROBLEM.NDIM == {} DATA.PATCH_SIZE tuple must be length {}, given {}. Not an error "
                "because you are using a model from BioImage Model Zoo (BMZ) and the patch size will be determined by the model."
                " However, this message is printed so you are aware of this. "
            )
    assert cfg.DATA.NORMALIZATION.TYPE in [
        "div",
        "scale_range",
        "custom",
    ], "DATA.NORMALIZATION.TYPE not in ['div', 'scale_range', 'custom']"
    if cfg.DATA.NORMALIZATION.CUSTOM_MEAN != -1 and cfg.DATA.NORMALIZATION.CUSTOM_STD == -1:
        raise ValueError(
            "'DATA.NORMALIZATION.CUSTOM_STD' needs to be provided when 'DATA.NORMALIZATION.CUSTOM_MEAN' is provided too"
        )
    if cfg.DATA.NORMALIZATION.PERC_CLIP:
        if cfg.DATA.NORMALIZATION.PERC_LOWER == -1:
            raise ValueError(
                "'DATA.NORMALIZATION.PERC_LOWER' needs to be set when DATA.NORMALIZATION.PERC_CLIP == 'True'"
            )
        if cfg.DATA.NORMALIZATION.PERC_UPPER == -1:
            raise ValueError(
                "'DATA.NORMALIZATION.PERC_UPPER' needs to be set when DATA.NORMALIZATION.PERC_CLIP == 'True'"
            )
        if not check_value(cfg.DATA.NORMALIZATION.PERC_LOWER, value_range=(0, 100)):
            raise ValueError("'DATA.NORMALIZATION.PERC_LOWER' not in [0, 100] range")
        if not check_value(cfg.DATA.NORMALIZATION.PERC_UPPER, value_range=(0, 100)):
            raise ValueError("'DATA.NORMALIZATION.PERC_UPPER' not in [0, 100] range")
    if cfg.DATA.TRAIN.REPLICATE:
        if cfg.PROBLEM.TYPE == "CLASSIFICATION" or (
            cfg.PROBLEM.TYPE == "SELF_SUPERVISED" and cfg.PROBLEM.SELF_SUPERVISED.PRETEXT_TASK == "masking"
        ):
            print("WARNING: 'DATA.TRAIN.REPLICATE' has no effect in the selected workflow")

    ### Model ###
    if not model_will_be_read and cfg.MODEL.SOURCE == "biapy":
        assert model_arch in [
            "unet",
            "resunet",
            "resunet++",
            "attention_unet",
            "multiresunet",
            "seunet",
            "resunet_se",
            "simple_cnn",
            "efficientnet_b0",
            "efficientnet_b1",
            "efficientnet_b2",
            "efficientnet_b3",
            "efficientnet_b4",
            "efficientnet_b5",
            "efficientnet_b6",
            "efficientnet_b7",
            "unetr",
            "edsr",
            "rcan",
            "dfcan",
            "wdsr",
            "vit",
            "mae",
            "unext_v1",
            "unext_v2",
        ], "MODEL.ARCHITECTURE not in ['unet', 'resunet', 'resunet++', 'attention_unet', 'multiresunet', 'seunet', 'simple_cnn', 'efficientnet_b[0-7]', 'unetr', 'edsr', 'rcan', 'dfcan', 'wdsr', 'vit', 'mae', 'unext_v1', 'unext_v2']"
        if (
            model_arch
            not in [
                "unet",
                "resunet",
                "resunet++",
                "seunet",
                "resunet_se",
                "attention_unet",
                "multiresunet",
                "unetr",
                "vit",
                "mae",
                "unext_v1",
                "unext_v2",
            ]
            and cfg.PROBLEM.NDIM == "3D"
            and cfg.PROBLEM.TYPE != "CLASSIFICATION"
        ):
            raise ValueError(
                "For 3D these models are available: {}".format(
                    [
                        "unet",
                        "resunet",
                        "resunet++",
                        "seunet",
                        "resunet_se",
                        "multiresunet",
                        "attention_unet",
                        "unetr",
                        "vit",
                        "mae",
                        "unext_v1",
                        "unext_v2",
                    ]
                )
            )
        if (
            cfg.MODEL.N_CLASSES > 2
            and cfg.PROBLEM.TYPE != "CLASSIFICATION"
            and model_arch
            not in [
                "unet",
                "resunet",
                "resunet++",
                "seunet",
                "resunet_se",
                "attention_unet",
                "multiresunet",
                "unetr",
                "unext_v1",
                "unext_v2",
            ]
        ):
            raise ValueError(
                "'MODEL.N_CLASSES' > 2 can only be used with 'MODEL.ARCHITECTURE' in ['unet', 'resunet', 'resunet++', 'seunet', 'resunet_se', 'attention_unet', 'multiresunet', 'unetr', 'unext_v1', 'unext_v2']"
            )

        assert len(cfg.MODEL.FEATURE_MAPS) > 2, "'MODEL.FEATURE_MAPS' needs to have at least 3 values"

        # Adjust dropout to feature maps
        if model_arch in ["vit", "unetr", "mae"]:
            if all(x == 0 for x in cfg.MODEL.DROPOUT_VALUES):
                opts.extend(["MODEL.DROPOUT_VALUES", (0.0,)])
            elif len(cfg.MODEL.DROPOUT_VALUES) != 1:
                raise ValueError(
                    "'MODEL.DROPOUT_VALUES' must be list of an unique number when 'MODEL.ARCHITECTURE' is one among ['vit', 'mae', 'unetr']"
                )
            elif not check_value(cfg.MODEL.DROPOUT_VALUES[0]):
                raise ValueError("'MODEL.DROPOUT_VALUES' not in [0, 1] range")
        else:
            if len(cfg.MODEL.FEATURE_MAPS) != len(cfg.MODEL.DROPOUT_VALUES):
                if all(x == 0 for x in cfg.MODEL.DROPOUT_VALUES):
                    opts.extend(["MODEL.DROPOUT_VALUES", (0.0,) * len(cfg.MODEL.FEATURE_MAPS)])
                elif any(not check_value(x) for x in cfg.MODEL.DROPOUT_VALUES):
                    raise ValueError("'MODEL.DROPOUT_VALUES' not in [0, 1] range")
                else:
                    raise ValueError("'MODEL.FEATURE_MAPS' and 'MODEL.DROPOUT_VALUES' lengths must be equal")

        # Adjust Z_DOWN values to feature maps
        if all(x == 0 for x in cfg.MODEL.Z_DOWN):
            opts.extend(["MODEL.Z_DOWN", (2,) * (len(cfg.MODEL.FEATURE_MAPS) - 1)])
        elif any([False for x in cfg.MODEL.Z_DOWN if x != 1 and x != 2]):
            raise ValueError("'MODEL.Z_DOWN' needs to be 1 or 2")
        else:
            if model_arch == "multiresunet" and len(cfg.MODEL.Z_DOWN) != 4:
                raise ValueError("'MODEL.Z_DOWN' length must be 4 when using 'multiresunet'")
            elif len(cfg.MODEL.FEATURE_MAPS) - 1 != len(cfg.MODEL.Z_DOWN):
                raise ValueError("'MODEL.FEATURE_MAPS' length minus one and 'MODEL.Z_DOWN' length must be equal")

        # Adjust ISOTROPY values to feature maps
        if all(x == True for x in cfg.MODEL.ISOTROPY):
            opts.extend(["MODEL.ISOTROPY", (True,) * (len(cfg.MODEL.FEATURE_MAPS))])

    # Correct UPSCALING for other workflows than SR
    if len(cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING) == 0:
        opts.extend(["PROBLEM.SUPER_RESOLUTION.UPSCALING", (1,) * dim_count])

    if len(opts) > 0:
        cfg.merge_from_list(opts)

    if not model_will_be_read and cfg.MODEL.SOURCE == "biapy":
        assert cfg.MODEL.LAST_ACTIVATION.lower() in [
            "relu",
            "tanh",
            "leaky_relu",
            "elu",
            "gelu",
            "silu",
            "sigmoid",
            "softmax",
            "linear",
            "none",
        ], "Get unknown activation key {}".format(cfg.MODEL.LAST_ACTIVATION.lower())

        if cfg.MODEL.UPSAMPLE_LAYER.lower() not in ["upsampling", "convtranspose"]:
            raise ValueError(
                "cfg.MODEL.UPSAMPLE_LAYER' needs to be one between ['upsampling', 'convtranspose']. Provided {}".format(
                    cfg.MODEL.UPSAMPLE_LAYER
                )
            )
        if cfg.PROBLEM.TYPE in [
            "SEMANTIC_SEG",
            "INSTANCE_SEG",
            "DETECTION",
            "DENOISING",
        ]:
            if model_arch not in [
                "unet",
                "resunet",
                "resunet++",
                "seunet",
                "attention_unet",
                "resunet_se",
                "unetr",
                "multiresunet",
                "unext_v1","unext_v2",
            ]:
                raise ValueError(
                    "Architectures available for {} are: ['unet', 'resunet', 'resunet++', 'seunet', 'attention_unet', 'resunet_se', 'unetr', 'multiresunet', 'unext_v1', 'unext_v2']".format(
                        cfg.PROBLEM.TYPE
                    )
                )
        elif cfg.PROBLEM.TYPE == "SUPER_RESOLUTION":
            if cfg.PROBLEM.NDIM == "2D" and model_arch not in [
                "edsr",
                "rcan",
                "dfcan",
                "wdsr",
                "unet",
                "resunet",
                "resunet++",
                "seunet",
                "resunet_se",
                "attention_unet",
                "multiresunet",
                "unext_v1",
                "unext_v2",
            ]:
                raise ValueError(
                    "Architectures available for 2D 'SUPER_RESOLUTION' are: ['edsr', 'rcan', 'dfcan', 'wdsr', 'unet', 'resunet', 'resunet++', 'seunet', 'resunet_se', 'attention_unet', 'multiresunet', 'unext_v1', 'unext_v2']"
                )
            elif cfg.PROBLEM.NDIM == "3D":
                if model_arch not in [
                    "unet",
                    "resunet",
                    "resunet++",
                    "seunet",
                    "attention_unet",
                    "multiresunet",
                    "unext_v1",
                    "unext_v2",
                ]:
                    raise ValueError(
                        "Architectures available for 3D 'SUPER_RESOLUTION' are: ['unet', 'resunet', 'resunet++', 'seunet', 'resunet_se', 'attention_unet', 'multiresunet', 'unext_v1', 'unext_v2']"
                    )
                assert cfg.MODEL.UNET_SR_UPSAMPLE_POSITION in [
                    "pre",
                    "post",
                ], "'MODEL.UNET_SR_UPSAMPLE_POSITION' not in ['pre', 'post']"
        elif cfg.PROBLEM.TYPE == "IMAGE_TO_IMAGE":
            if model_arch not in [
                "edsr",
                "rcan",
                "dfcan",
                "wdsr",
                "unet",
                "resunet",
                "resunet++",
                "seunet",
                "resunet_se",
                "attention_unet",
                "unetr",
                "multiresunet",
                "unext_v1",
                "unext_v2",
            ]:
                raise ValueError(
                    "Architectures available for 'IMAGE_TO_IMAGE' are: ['edsr', 'rcan', 'dfcan', 'wdsr', 'unet', 'resunet', 'resunet++', 'resunet_se', 'seunet', 'attention_unet', 'unetr', 'multiresunet', 'unext_v1', 'unext_v2']"
                )
        elif cfg.PROBLEM.TYPE == "SELF_SUPERVISED":
            if model_arch not in [
                "unet",
                "resunet",
                "resunet++",
                "attention_unet",
                "multiresunet",
                "seunet",
                "resunet_se",
                "unetr",
                "unext_v1",
                "unext_v2",
                "edsr",
                "rcan",
                "dfcan",
                "wdsr",
                "vit",
                "mae",
            ]:
                raise ValueError(
                    "'SELF_SUPERVISED' models available are these: ['unet', 'resunet', 'resunet++', 'attention_unet', 'multiresunet', 'seunet', 'resunet_se', "
                    "'unetr', 'unext_v1', 'edsr', 'rcan', 'dfcan', 'wdsr', 'vit', 'mae']"
                )
        elif cfg.PROBLEM.TYPE == "CLASSIFICATION":
            if model_arch not in ["simple_cnn", "vit"] and "efficientnet" not in model_arch:
                raise ValueError(
                    "Architectures available for 'CLASSIFICATION' are: ['simple_cnn', 'efficientnet_b[0-7]', 'vit']"
                )
            if cfg.PROBLEM.NDIM == "3D" and "efficientnet" in model_arch:
                raise ValueError("EfficientNet architectures are only available for 2D images")
        if model_arch in ["unetr", "vit", "mae"]:
            if model_arch == "mae" and cfg.PROBLEM.TYPE != "SELF_SUPERVISED":
                raise ValueError("'mae' model can only be used in 'SELF_SUPERVISED' workflow")
            if cfg.MODEL.VIT_EMBED_DIM % cfg.MODEL.VIT_NUM_HEADS != 0:
                raise ValueError("'MODEL.VIT_EMBED_DIM' should be divisible by 'MODEL.VIT_NUM_HEADS'")
            if not all([i == cfg.DATA.PATCH_SIZE[0] for i in cfg.DATA.PATCH_SIZE[:-1]]):
                raise ValueError(
                    "'unetr', 'vit' 'mae' models need to have same shape in all dimensions (e.g. DATA.PATCH_SIZE = (80,80,80,1) )"
                )
        # Check that the input patch size is divisible in every level of the U-Net's like architectures, as the model
        # will throw an error not very clear for users
        if model_arch in [
            "unet",
            "resunet",
            "resunet++",
            "seunet",
            "resunet_se",
            "attention_unet",
            "multiresunet",
            "unext_v1",
            "unext_v2",
        ]:
            z_size = cfg.DATA.PATCH_SIZE[0]
            sizes = cfg.DATA.PATCH_SIZE[1:-1]
            for i in range(len(cfg.MODEL.FEATURE_MAPS) - 1):
                if not all(
                    [False for x in sizes if x % (np.power(2, (i + 1))) != 0 or z_size % cfg.MODEL.Z_DOWN[i] != 0]
                ):
                    m = (
                        "The 'DATA.PATCH_SIZE' provided is not divisible by 2 in each of the U-Net's levels. You can:\n 1) Reduce the number "
                        + "of levels (by reducing 'cfg.MODEL.FEATURE_MAPS' array's length)\n 2) Increase 'DATA.PATCH_SIZE'"
                    )
                    if cfg.PROBLEM.NDIM == "3D":
                        m += (
                            "\n 3) If the Z axis is the problem, as the patch size is normally less than in other axis due to resolution, you "
                            + "can tune 'MODEL.Z_DOWN' variable to not downsample the image in all U-Net levels"
                        )
                    raise ValueError(m)
                z_size = z_size // cfg.MODEL.Z_DOWN[i]

    if cfg.MODEL.LOAD_CHECKPOINT and check_data_paths:
        if not os.path.exists(get_checkpoint_path(cfg, jobname)):
            raise FileNotFoundError(f"Model checkpoint not found at {get_checkpoint_path(cfg, jobname)}")

    ### Train ###
    assert cfg.TRAIN.OPTIMIZER in [
        "SGD",
        "ADAM",
        "ADAMW",
    ], "TRAIN.OPTIMIZER not in ['SGD', 'ADAM', 'ADAMW']"

    if cfg.TRAIN.ENABLE and cfg.TRAIN.LR_SCHEDULER.NAME != "":
        if cfg.TRAIN.LR_SCHEDULER.NAME not in [
            "reduceonplateau",
            "warmupcosine",
            "onecycle",
        ]:
            raise ValueError(
                "'TRAIN.LR_SCHEDULER.NAME' must be one between ['reduceonplateau', 'warmupcosine', 'onecycle']"
            )
        if cfg.TRAIN.LR_SCHEDULER.MIN_LR == -1.0 and cfg.TRAIN.LR_SCHEDULER.NAME != "onecycle":
            raise ValueError(
                "'TRAIN.LR_SCHEDULER.MIN_LR' needs to be set when 'TRAIN.LR_SCHEDULER.NAME' is between ['reduceonplateau', 'warmupcosine']"
            )

        if cfg.TRAIN.LR_SCHEDULER.NAME == "reduceonplateau":
            if cfg.TRAIN.LR_SCHEDULER.REDUCEONPLATEAU_PATIENCE == -1:
                raise ValueError(
                    "'TRAIN.LR_SCHEDULER.REDUCEONPLATEAU_PATIENCE' needs to be set when 'TRAIN.LR_SCHEDULER.NAME' is 'reduceonplateau'"
                )
            if cfg.TRAIN.LR_SCHEDULER.REDUCEONPLATEAU_PATIENCE >= cfg.TRAIN.PATIENCE:
                raise ValueError(
                    "'TRAIN.LR_SCHEDULER.REDUCEONPLATEAU_PATIENCE' needs to be less than 'TRAIN.PATIENCE' "
                )

        if cfg.TRAIN.LR_SCHEDULER.NAME == "warmupcosine":
            if cfg.TRAIN.LR_SCHEDULER.WARMUP_COSINE_DECAY_EPOCHS == -1:
                raise ValueError(
                    "'TRAIN.LR_SCHEDULER.WARMUP_COSINE_DECAY_EPOCHS' needs to be set when 'TRAIN.LR_SCHEDULER.NAME' is 'warmupcosine'"
                )
            if cfg.TRAIN.LR_SCHEDULER.WARMUP_COSINE_DECAY_EPOCHS > cfg.TRAIN.EPOCHS:
                raise ValueError("'TRAIN.LR_SCHEDULER.WARMUP_COSINE_DECAY_EPOCHS' needs to be less than 'TRAIN.EPOCHS'")

    #### Augmentation ####
    if cfg.AUGMENTOR.ENABLE:
        if not check_value(cfg.AUGMENTOR.DA_PROB):
            raise ValueError("AUGMENTOR.DA_PROB not in [0, 1] range")
        if cfg.AUGMENTOR.RANDOM_ROT:
            if not check_value(cfg.AUGMENTOR.RANDOM_ROT_RANGE, (-360, 360)):
                raise ValueError("AUGMENTOR.RANDOM_ROT_RANGE values needs to be between [-360,360]")
        if cfg.AUGMENTOR.SHEAR:
            if not check_value(cfg.AUGMENTOR.SHEAR_RANGE, (-360, 360)):
                raise ValueError("AUGMENTOR.SHEAR_RANGE values needs to be between [-360,360]")
        if cfg.AUGMENTOR.ELASTIC:
            if cfg.AUGMENTOR.E_MODE not in ["constant", "nearest", "reflect", "wrap"]:
                raise ValueError("AUGMENTOR.E_MODE not in ['constant', 'nearest', 'reflect', 'wrap']")
        if cfg.AUGMENTOR.BRIGHTNESS:
            if cfg.AUGMENTOR.BRIGHTNESS_MODE not in ["2D", "3D"] and cfg.PROBLEM.NDIM == "3D":
                raise ValueError("AUGMENTOR.BRIGHTNESS_MODE not in ['2D', '3D']")
        if cfg.AUGMENTOR.CONTRAST:
            if cfg.AUGMENTOR.CONTRAST_MODE not in ["2D", "3D"] and cfg.PROBLEM.NDIM == "3D":
                raise ValueError("AUGMENTOR.CONTRAST_MODE not in ['2D', '3D']")
        if cfg.AUGMENTOR.DROPOUT:
            if not check_value(cfg.AUGMENTOR.DROP_RANGE):
                raise ValueError("AUGMENTOR.DROP_RANGE values not in [0, 1] range")
        if cfg.AUGMENTOR.CUTOUT:
            if not check_value(cfg.AUGMENTOR.COUT_SIZE):
                raise ValueError("AUGMENTOR.COUT_SIZE values not in [0, 1] range")
        if cfg.AUGMENTOR.CUTBLUR:
            if not check_value(cfg.AUGMENTOR.CBLUR_SIZE):
                raise ValueError("AUGMENTOR.CBLUR_SIZE values not in [0, 1] range")
            if not check_value(cfg.AUGMENTOR.CBLUR_DOWN_RANGE, (1, 8)):
                raise ValueError("AUGMENTOR.CBLUR_DOWN_RANGE values not in [1, 8] range")
        if cfg.AUGMENTOR.CUTMIX:
            if not check_value(cfg.AUGMENTOR.CMIX_SIZE):
                raise ValueError("AUGMENTOR.CMIX_SIZE values not in [0, 1] range")
        if cfg.AUGMENTOR.CUTNOISE:
            if not check_value(cfg.AUGMENTOR.CNOISE_SCALE):
                raise ValueError("AUGMENTOR.CNOISE_SCALE values not in [0, 1] range")
            if not check_value(cfg.AUGMENTOR.CNOISE_SIZE):
                raise ValueError("AUGMENTOR.CNOISE_SIZE values not in [0, 1] range")
        if cfg.AUGMENTOR.GRIDMASK:
            if not check_value(cfg.AUGMENTOR.GRID_RATIO):
                raise ValueError("AUGMENTOR.GRID_RATIO not in [0, 1] range")
            if cfg.AUGMENTOR.GRID_D_RANGE[0] >= cfg.AUGMENTOR.GRID_D_RANGE[1]:
                raise ValueError(
                    "cfg.AUGMENTOR.GRID_D_RANGE[0] needs to be larger than cfg.AUGMENTOR.GRID_D_RANGE[1]"
                    "Provided {}".format(cfg.AUGMENTOR.GRID_D_RANGE)
                )
            if not check_value(cfg.AUGMENTOR.GRID_D_RANGE):
                raise ValueError("cfg.AUGMENTOR.GRID_D_RANGE values not in [0, 1] range")
            if not check_value(cfg.AUGMENTOR.GRID_ROTATE):
                raise ValueError("AUGMENTOR.GRID_ROTATE not in [0, 1] range")
        if cfg.AUGMENTOR.ZOOM:
            if not check_value(cfg.AUGMENTOR.ZOOM_RANGE, (0.1, 10)):
                raise ValueError("AUGMENTOR.ZOOM_RANGE values needs to be between [0.1,10]")
            if cfg.AUGMENTOR.ZOOM_IN_Z and dim_count == 2:
                print("WARNING: Ignoring AUGMENTOR.ZOOM_IN_Z in 2D problem")
        assert cfg.AUGMENTOR.AFFINE_MODE in [
            "constant",
            "reflect",
            "wrap",
            "symmetric",
        ], "'AUGMENTOR.AFFINE_MODE' needs to be one between ['constant', 'reflect', 'wrap', 'symmetric']"
        if cfg.AUGMENTOR.GAMMA_CONTRAST and cfg.DATA.NORMALIZATION.TYPE == "custom":
            raise ValueError(
                "'AUGMENTOR.GAMMA_CONTRAST' doesn't work correctly on images with negative values, which 'custom' "
                "normalization will lead to"
            )

    # BioImage Model Zoo exportation process
    if cfg.MODEL.BMZ.EXPORT.ENABLE:
        if not cfg.MODEL.BMZ.EXPORT.REUSE_BMZ_CONFIG:
            if cfg.MODEL.BMZ.EXPORT.MODEL_NAME == "":
                raise ValueError(
                    "'MODEL.BMZ.EXPORT.MODEL_NAME' must be set. Remember that it should be something meaningful (take other models names in https://bioimage.io/#/ as reference)."
                )

            if cfg.MODEL.BMZ.EXPORT.DESCRIPTION == "":
                raise ValueError(
                    "'MODEL.BMZ.EXPORT.DESCRIPTION' must be set. Remember that it should be meaninful (take other models descriptions in https://bioimage.io/#/ as reference)."
                )
            if len(cfg.MODEL.BMZ.EXPORT.AUTHORS) == 0:
                raise ValueError(
                    "At least one author must be provided in 'MODEL.BMZ.EXPORT.AUTHORS'. Each author must be a dictionary containing 'name' and 'github_user' keys. E.g. [{'name': 'Daniel', 'github_user': 'danifranco'}]"
                )
            if cfg.MODEL.BMZ.EXPORT.LICENSE == "":
                raise ValueError(
                    "'MODEL.BMZ.EXPORT.LICENSE' must be set. Remember that it should be something meaningful (take other models licenses in https://bioimage.io/#/ as reference)."
                )
            if len(cfg.MODEL.BMZ.EXPORT.TAGS) == 0:
                raise ValueError(
                    "'MODEL.BMZ.EXPORT.TAGS' must be set. Remember that it should be something meaningful (take other models tags in https://bioimage.io/#/ as reference)."
                )
            if len(cfg.MODEL.BMZ.EXPORT.CITE) > 0:
                for d in cfg.MODEL.BMZ.EXPORT.CITE:
                    if not isinstance(d, dict):
                        raise ValueError(
                            "'MODEL.BMZ.EXPORT.CITE' needs to be a list of dicts. E.g. [{'text': 'Gizmo et al.', 'doi': '10.1002/xyzacab123'}, {'text': 'training library', 'doi': '10.1101/2024.02.03.576026'}]"
                        )
                    else:
                        if len(d.keys()) < 2 or "text" not in d:
                            raise ValueError(
                                "'MODEL.BMZ.EXPORT.CITE' malformed. Cite dictionary must have at least 'text' key. E.g. {'text': 'Gizmo et al.', 'doi': '10.1002/xyzacab123'}"
                            )
                        for k in d.keys():
                            if k not in ["text", "doi", "url"]:
                                raise ValueError(
                                    f"'MODEL.BMZ.EXPORT.CITE' malformed. Cite dictionary available keys are: ['text', 'doi', 'url']. Provided {k}. E.g. {'text': 'Gizmo et al.', 'doi': '10.1002/xyzacab123'}"
                                )
                            
            if cfg.MODEL.BMZ.EXPORT.DOCUMENTATION == "":
                print(
                    "WARNING: 'MODEL.BMZ.EXPORT.DOCUMENTATION' not set so the model documentation will point to BiaPy doc: https://github.com/BiaPyX/BiaPy/blob/master/README.md"
                )
            elif not os.path.exists(cfg.MODEL.BMZ.EXPORT.DOCUMENTATION):
                raise ValueError(
                    "'MODEL.BMZ.EXPORT.DOCUMENTATION' path provided doesn't point to a file or can't be reached: {}".format(
                        cfg.MODEL.BMZ.EXPORT.DOCUMENTATION
                    )
                )
            elif not str(cfg.MODEL.BMZ.EXPORT.DOCUMENTATION).endswith(".md"):
                raise ValueError(
                    "'MODEL.BMZ.EXPORT.DOCUMENTATION' file suffix must be .md"
                )
        else:
            if cfg.MODEL.SOURCE != "bmz":
                raise ValueError("Seems that you are not loading a BioImage Model Zoo model. Thus, you can not activate 'MODEL.BMZ.EXPORT.REUSE_BMZ_CONFIG' as there will be nothing to reuse.")

    #### Post-processing ####
    if cfg.TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS:
        if len(cfg.DATA.TEST.RESOLUTION) == 1:
            raise ValueError("'DATA.TEST.RESOLUTION' must be set when using 'TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS'")
        if len(cfg.DATA.TEST.RESOLUTION) != dim_count:
            raise ValueError(
                "'DATA.TEST.RESOLUTION' must match in length to {}, which is the number of "
                "dimensions".format(dim_count)
            )
        if cfg.TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS_RADIUS[0] == -1:
            raise ValueError(
                "'TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS' needs to be set when 'TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS' is True"
            )


def compare_configurations_without_model(actual_cfg, old_cfg, header_message="", old_cfg_version=None):
    """
    Compares two configurations and throws an error if they differ in some critical variables that change workflow behaviour. This
    comparisdon does not take into account model specs.
    """
    print("Comparing configurations . . .")

    vars_to_compare = [
        "PROBLEM.TYPE",
        "PROBLEM.NDIM",
        "DATA.PATCH_SIZE",
        "PROBLEM.INSTANCE_SEG.DATA_CHANNELS",
        "PROBLEM.SELF_SUPERVISED.PRETEXT_TASK",
        "PROBLEM.SUPER_RESOLUTION.UPSCALING",
        "MODEL.N_CLASSES",
    ]

    def get_attribute_recursive(var, attr):
        att = attr.split(".")
        if len(att) == 1:
            return getattr(var, att[0])
        else:
            return get_attribute_recursive(getattr(var, att[0]), ".".join(att[1:]))

    # Old configuration translation
    dim_count = 2 if old_cfg.PROBLEM.NDIM == "2D" else 3
    # BiaPy version less than 3.5.5
    if old_cfg_version is None:
        if isinstance(old_cfg["PROBLEM"]["SUPER_RESOLUTION"]["UPSCALING"], int):
            old_cfg["PROBLEM"]["SUPER_RESOLUTION"]["UPSCALING"] = (old_cfg["PROBLEM"]["SUPER_RESOLUTION"]["UPSCALING"],) * dim_count

    for var_to_compare in vars_to_compare:
        if get_attribute_recursive(actual_cfg, var_to_compare) != get_attribute_recursive(old_cfg, var_to_compare):
            raise ValueError(
                header_message + f"The '{var_to_compare}' value of the compared configurations does not match: " +\
                f"{get_attribute_recursive(actual_cfg, var_to_compare)} (current configuration) vs {get_attribute_recursive(old_cfg, var_to_compare)} (from loaded configuration)"
            )

    print("Configurations seem to be compatible. Continuing . . .")


def convert_old_model_cfg_to_current_version(old_cfg):
    """
    Backward compatibility until commit 6aa291baa9bc5d7fb410454bfcea3a3da0c23604 (version 3.2.0)
    Commit url: https://github.com/BiaPyX/BiaPy/commit/6aa291baa9bc5d7fb410454bfcea3a3da0c23604
    """
    if "TEST" in old_cfg:
        if "STATS" in old_cfg["TEST"]:
            full_image = old_cfg["TEST"]["STATS"]["FULL_IMG"]
            del old_cfg["TEST"]["STATS"]
            old_cfg["TEST"]["FULL_IMG"] = full_image
        if "EVALUATE" in old_cfg["TEST"]:
            del old_cfg["TEST"]["EVALUATE"]
        if "POST_PROCESSING" in old_cfg["TEST"]:
            if "YZ_FILTERING" in old_cfg["TEST"]["POST_PROCESSING"]:
                del old_cfg["TEST"]["POST_PROCESSING"]["YZ_FILTERING"] 
                try:
                    fsize = old_cfg["TEST"]["POST_PROCESSING"]["YZ_FILTERING_SIZE"] 
                except: 
                    fsize = 5
                del old_cfg["TEST"]["POST_PROCESSING"]["YZ_FILTERING_SIZE"]
                
                old_cfg["TEST"]["POST_PROCESSING"]["MEDIAN_FILTER"] = True
                old_cfg["TEST"]["POST_PROCESSING"]["MEDIAN_FILTER_AXIS"] = ["yz"]
                old_cfg["TEST"]["POST_PROCESSING"]["MEDIAN_FILTER_SIZE"] = [fsize]
            if "Z_FILTERING" in old_cfg["TEST"]["POST_PROCESSING"]:
                del old_cfg["TEST"]["POST_PROCESSING"]["Z_FILTERING"] 
                try:
                    fsize = old_cfg["TEST"]["POST_PROCESSING"]["Z_FILTERING_SIZE"] 
                except: 
                    fsize = 5
                del old_cfg["TEST"]["POST_PROCESSING"]["Z_FILTERING_SIZE"]
                
                old_cfg["TEST"]["POST_PROCESSING"]["MEDIAN_FILTER"] = True
                old_cfg["TEST"]["POST_PROCESSING"]["MEDIAN_FILTER_AXIS"] = ["z"]
                old_cfg["TEST"]["POST_PROCESSING"]["MEDIAN_FILTER_SIZE"] = [fsize]

            if "MEASURE_PROPERTIES" in old_cfg["TEST"]["POST_PROCESSING"]:
                if "REMOVE_BY_PROPERTIES" in old_cfg["TEST"]["POST_PROCESSING"]["MEASURE_PROPERTIES"]:
                    if "SIGN" in old_cfg["TEST"]["POST_PROCESSING"]["MEASURE_PROPERTIES"]["REMOVE_BY_PROPERTIES"]:
                        old_cfg["TEST"]["POST_PROCESSING"]["MEASURE_PROPERTIES"]["REMOVE_BY_PROPERTIES"]["SIGNS"] = old_cfg["TEST"]["POST_PROCESSING"]["MEASURE_PROPERTIES"]["REMOVE_BY_PROPERTIES"]["SIGN"]
                        del old_cfg["TEST"]["POST_PROCESSING"]["MEASURE_PROPERTIES"]["REMOVE_BY_PROPERTIES"]["SIGN"]

            if "REMOVE_BY_PROPERTIES" in old_cfg["TEST"]["POST_PROCESSING"]:
                old_cfg["TEST"]["POST_PROCESSING"]["MEASURE_PROPERTIES"] = {}
                old_cfg["TEST"]["POST_PROCESSING"]["MEASURE_PROPERTIES"]["REMOVE_BY_PROPERTIES"] = {}
                old_cfg["TEST"]["POST_PROCESSING"]["MEASURE_PROPERTIES"]["ENABLE"] = True
                old_cfg["TEST"]["POST_PROCESSING"]["MEASURE_PROPERTIES"]["REMOVE_BY_PROPERTIES"]["ENABLE"] = True
                old_cfg["TEST"]["POST_PROCESSING"]["MEASURE_PROPERTIES"]["REMOVE_BY_PROPERTIES"]["PROPS"] = old_cfg["TEST"]["POST_PROCESSING"]["REMOVE_BY_PROPERTIES"]
                del old_cfg["TEST"]["POST_PROCESSING"]["REMOVE_BY_PROPERTIES"]
                if "REMOVE_BY_PROPERTIES_VALUES" in old_cfg["TEST"]["POST_PROCESSING"]:
                    old_cfg["TEST"]["POST_PROCESSING"]["MEASURE_PROPERTIES"]["REMOVE_BY_PROPERTIES"]["VALUES"] = old_cfg["TEST"]["POST_PROCESSING"]["REMOVE_BY_PROPERTIES_VALUES"]
                    del old_cfg["TEST"]["POST_PROCESSING"]["REMOVE_BY_PROPERTIES_VALUES"]
                if "REMOVE_BY_PROPERTIES_SIGN" in old_cfg["TEST"]["POST_PROCESSING"]:
                    old_cfg["TEST"]["POST_PROCESSING"]["MEASURE_PROPERTIES"]["REMOVE_BY_PROPERTIES"]["SIGNS"] = old_cfg["TEST"]["POST_PROCESSING"]["REMOVE_BY_PROPERTIES_SIGN"]
                    del old_cfg["TEST"]["POST_PROCESSING"]["REMOVE_BY_PROPERTIES_SIGN"]

    if "PROBLEM" in old_cfg:
        ndim = 3 if "NDIM" in old_cfg["PROBLEM"] and old_cfg["PROBLEM"]["NDIM"] == "3D" else 2
        if "DETECTION" in old_cfg["PROBLEM"]:
            if "CENTRAL_POINT_DILATION" in old_cfg["PROBLEM"]["DETECTION"]:
                if isinstance(old_cfg["PROBLEM"]["DETECTION"]["CENTRAL_POINT_DILATION"], int):
                    old_cfg["PROBLEM"]["DETECTION"]["CENTRAL_POINT_DILATION"] = [old_cfg["PROBLEM"]["DETECTION"]["CENTRAL_POINT_DILATION"]]

        if "SUPER_RESOLUTION" in old_cfg["PROBLEM"]:
            if "UPSCALING" in old_cfg["PROBLEM"]["SUPER_RESOLUTION"]:
                if isinstance(old_cfg["PROBLEM"]["SUPER_RESOLUTION"]["UPSCALING"], int):
                    old_cfg["PROBLEM"]["SUPER_RESOLUTION"]["UPSCALING"] = (old_cfg["PROBLEM"]["SUPER_RESOLUTION"]["UPSCALING"],)*ndim

    if "DATA" in old_cfg:
        if "TRAIN" in old_cfg["DATA"]:
            if "MINIMUM_FOREGROUND_PER" in old_cfg["DATA"]["TRAIN"]:
                min_fore = old_cfg["DATA"]["TRAIN"]["MINIMUM_FOREGROUND_PER"]
                del old_cfg["DATA"]["TRAIN"]["MINIMUM_FOREGROUND_PER"]
                if min_fore != -1:  
                    old_cfg["DATA"]["TRAIN"]["FILTER_SAMPLES"] = {} 
                    old_cfg["DATA"]["TRAIN"]["FILTER_SAMPLES"]["PROPS"] = [['foreground']]
                    old_cfg["DATA"]["TRAIN"]["FILTER_SAMPLES"]["VALUES"] = [[min_fore]]
                    old_cfg["DATA"]["TRAIN"]["FILTER_SAMPLES"]["SIGNS"] = [['lt']]
        if "VAL" in old_cfg["DATA"]:
            if "BINARY_MASKS" in old_cfg["DATA"]["VAL"]:
                del old_cfg["DATA"]["VAL"]["BINARY_MASKS"]

    if "AUGMENTOR" in old_cfg:
        if "BRIGHTNESS_EM" in old_cfg["AUGMENTOR"]:
            del old_cfg["AUGMENTOR"]["BRIGHTNESS_EM"]
        if "BRIGHTNESS_EM_FACTOR" in old_cfg["AUGMENTOR"]:
            del old_cfg["AUGMENTOR"]["BRIGHTNESS_EM_FACTOR"]
        if "BRIGHTNESS_EM_MODE" in old_cfg["AUGMENTOR"]:
            del old_cfg["AUGMENTOR"]["BRIGHTNESS_EM_MODE"]
        if "CONTRAST_EM" in old_cfg["AUGMENTOR"]:
            del old_cfg["AUGMENTOR"]["CONTRAST_EM"]
        if "CONTRAST_EM_FACTOR" in old_cfg["AUGMENTOR"]:
            del old_cfg["AUGMENTOR"]["CONTRAST_EM_FACTOR"]
        if "CONTRAST_EM_MODE" in old_cfg["AUGMENTOR"]:
            del old_cfg["AUGMENTOR"]["CONTRAST_EM_MODE"]

    if "MODEL" in old_cfg:
        if "BATCH_NORMALIZATION" in old_cfg["MODEL"]:
            if old_cfg["MODEL"]["BATCH_NORMALIZATION"]:
                old_cfg["MODEL"]["NORMALIZATION"] = "bn"
            del old_cfg["MODEL"]["BATCH_NORMALIZATION"]

        if "BMZ" in old_cfg["MODEL"]:
            if "SOURCE_MODEL_DOI" in old_cfg["MODEL"]["BMZ"]:
                model = old_cfg["MODEL"]["BMZ"]["SOURCE_MODEL_DOI"]
                del old_cfg["MODEL"]["BMZ"]["SOURCE_MODEL_DOI"]
                old_cfg["MODEL"]["BMZ"]["SOURCE_MODEL_ID"] = model
            if "EXPORT_MODEL" in old_cfg["MODEL"]["BMZ"]:
                old_cfg["MODEL"]["BMZ"]["EXPORT"] = {}
                try:                
                    enabled = old_cfg["MODEL"]["BMZ"]["EXPORT_MODEL"]["ENABLE"]
                except:
                    enabled = False
                old_cfg["MODEL"]["BMZ"]["EXPORT"]["ENABLED"] = enabled 
                try:                
                    model_name = old_cfg["MODEL"]["BMZ"]["EXPORT_MODEL"]["NAME"]
                except:
                    model_name = ''
                old_cfg["MODEL"]["BMZ"]["EXPORT"]["MODEL_NAME"] = model_name 
                try:                
                    description = old_cfg["MODEL"]["BMZ"]["EXPORT_MODEL"]["DESCRIPTION"]
                except:
                    description = ""
                old_cfg["MODEL"]["BMZ"]["EXPORT"]["DESCRIPTION"] = description 
                try:                
                    authors = old_cfg["MODEL"]["BMZ"]["EXPORT_MODEL"]["AUTHORS"]
                except:
                    authors = []
                old_cfg["MODEL"]["BMZ"]["EXPORT"]["AUTHORS"] = authors 
                try:                
                    license = old_cfg["MODEL"]["BMZ"]["EXPORT_MODEL"]["LICENSE"]
                except:
                    license = "CC-BY-4.0"
                old_cfg["MODEL"]["BMZ"]["EXPORT"]["LICENSE"] = license 
                try:                
                    doc = old_cfg["MODEL"]["BMZ"]["EXPORT_MODEL"]["DOCUMENTATION"]
                except:
                    doc = ""
                old_cfg["MODEL"]["BMZ"]["EXPORT"]["DOCUMENTATION"] = doc 
                try:                
                    tags = old_cfg["MODEL"]["BMZ"]["EXPORT_MODEL"]["TAGS"]
                except:
                    tags = []
                old_cfg["MODEL"]["BMZ"]["EXPORT"]["TAGS"] = tags 
                try:                
                    cite = old_cfg["MODEL"]["BMZ"]["EXPORT_MODEL"]["CITE"]
                except:
                    cite = []
                old_cfg["MODEL"]["BMZ"]["EXPORT"]["CITE"] = cite 
                del old_cfg["MODEL"]["BMZ"]["EXPORT_MODEL"]
    
    if "LOSS" in old_cfg:
        if "TYPE" in old_cfg["LOSS"]:
            del old_cfg["LOSS"]["TYPE"]

    try:
        del old_cfg["PATHS"]["RESULT_DIR"]["BMZ_BUILD"]
    except:
        pass

    return old_cfg
