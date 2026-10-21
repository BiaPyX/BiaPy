"""
Configuration from the wizard's answers: the variables they set, the default values for a first run (as
BiaPy-GUI sets them), the instance representation chosen and the strategy adapted to the data
(:mod:`biapy.wizard.planning`).

Typical use, when the wizard finishes::

    from biapy.wizard.config_builder import build_config, write_config

    cfg, report = build_config(answers)
    write_config(cfg, report, "my_job.yaml")
"""
import os


def to_jsonable(obj):
    """Convert numpy scalars/arrays and tuples to plain JSON types."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "tolist"):  # numpy array or scalar
        return to_jsonable(obj.tolist())
    return obj


def lists_to_tuples(obj):
    """Inverse of what JSON does to tuples. The GUI logic compares shapes as tuples."""
    if isinstance(obj, dict):
        return {k: lists_to_tuples(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return tuple(lists_to_tuples(v) for v in obj)
    return obj


def tuples_to_lists(obj):
    """yaml.safe_dump cannot write tuples; yacs coerces lists back to tuples when needed."""
    if isinstance(obj, dict):
        return {k: tuples_to_lists(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [tuples_to_lists(v) for v in obj]
    return obj


def create_dict_from_key(cfg_key, value, out_cfg):
    keys = cfg_key.split(".")
    if keys[0] not in out_cfg:
        out_cfg[keys[0]] = {}
    if len(keys) == 2:
        out_cfg[keys[0]][keys[1]] = value
    else:
        create_dict_from_key(".".join(keys[1:]), value, out_cfg[keys[0]])


def set_default_config(cfg, sample_info):
    """Default values for a first run, as BiaPy-GUI's ui_utils.set_default_config (batch size fixed to 1 there too)."""
    cfg["SYSTEM"] = {"NUM_CPUS": -1, "NUM_WORKERS": 3}

    if "PROBLEM" not in cfg:
        cfg["PROBLEM"] = {"TYPE": "SEMANTIC_SEG", "NDIM": "2D"}

    if "DATA" not in cfg:
        cfg["DATA"] = {}
    cfg["DATA"]["REFLECT_TO_COMPLETE_SHAPE"] = True

    if "TRAIN" not in cfg:
        cfg["TRAIN"] = {}
    if cfg["TRAIN"]["ENABLE"]:
        cfg["DATA"].setdefault("TRAIN", {})["IN_MEMORY"] = False
        cfg["DATA"].setdefault("VAL", {})
        cfg["DATA"]["VAL"]["IN_MEMORY"] = False
        cfg["DATA"]["VAL"]["FROM_TRAIN"] = True
        cfg["DATA"]["VAL"]["SPLIT_TRAIN"] = 0.1

    if "TEST" not in cfg:
        cfg["TEST"] = {}
    if cfg["TEST"]["ENABLE"]:
        cfg["DATA"].setdefault("TEST", {})["IN_MEMORY"] = False
        if cfg["DATA"]["PATCH_SIZE"][0] != -1:
            cfg["DATA"]["TEST"]["PADDING"] = str(tuple([x // 6 for x in cfg["DATA"]["PATCH_SIZE"][:-1]]))
        if "FULL_IMG" not in cfg["TEST"]:
            cfg["TEST"]["FULL_IMG"] = False

    cfg["DATA"].setdefault("NORMALIZATION", {})
    cfg["DATA"]["NORMALIZATION"]["TYPE"] = "zero_mean_unit_variance"
    cfg["DATA"]["NORMALIZATION"]["PERC_CLIP"] = {"ENABLE": True, "LOWER_PERC": 0.2, "UPPER_PERC": 99.8}

    if cfg["TRAIN"]["ENABLE"]:
        cfg["TRAIN"]["EPOCHS"] = 150
        cfg["TRAIN"]["PATIENCE"] = 50
        # Save the prediction of a few fixed samples after every epoch, so the front-ends can show how the model
        # learns (TRAIN.SAVE_TRAIN_PREDS_FREQ: every N epochs)
        cfg["TRAIN"]["SAVE_TRAIN_PREDS_FREQ"] = 1
        cfg["TRAIN"]["SAVE_TRAIN_PREDS_NUM_SAMPLES"] = 2
        cfg["TRAIN"]["OPTIMIZER"] = ["ADAMW"]  # one per group of parameters (a single group)
        cfg["TRAIN"]["LR"] = [1.0e-4]
        cfg["TRAIN"].setdefault("LR_SCHEDULER", {})
        cfg["TRAIN"]["LR_SCHEDULER"]["NAME"] = "warmupcosine"
        cfg["TRAIN"]["LR_SCHEDULER"]["MIN_LR"] = [1.0e-5]
        cfg["TRAIN"]["LR_SCHEDULER"]["WARMUP_COSINE_DECAY_EPOCHS"] = 10
        # BiaPy-GUI writes LOSS.CLASS_REBALANCE: True, which BiaPy >= 3.7 converts to this (the boolean form
        # is deprecated): rebalance within the channels for instance segmentation and detection
        if cfg["PROBLEM"]["TYPE"] in ["INSTANCE_SEG", "DETECTION"]:
            cfg["PROBLEM"].setdefault(cfg["PROBLEM"]["TYPE"], {})["CLASS_REBALANCE_WITHIN_CHANNELS"] = True

    cfg.setdefault("MODEL", {})
    cfg["MODEL"].setdefault("SOURCE", "biapy")
    if cfg["MODEL"]["SOURCE"] == "biapy":
        if cfg["PROBLEM"]["TYPE"] in ["SEMANTIC_SEG", "INSTANCE_SEG", "DETECTION", "DENOISING", "IMAGE_TO_IMAGE"]:
            cfg["MODEL"]["ARCHITECTURE"] = "resunet"
            if cfg["PROBLEM"]["NDIM"] == "3D":
                cfg["MODEL"]["Z_DOWN"] = [1, 1, 1, 1]
        elif cfg["PROBLEM"]["TYPE"] == "SUPER_RESOLUTION":
            if cfg["PROBLEM"]["NDIM"] == "3D":
                cfg["MODEL"]["ARCHITECTURE"] = "resunet"
                cfg["MODEL"]["Z_DOWN"] = [1, 1, 1, 1]
            else:
                cfg["MODEL"]["ARCHITECTURE"] = "rcan"
        elif cfg["PROBLEM"]["TYPE"] == "SELF_SUPERVISED":
            cfg["MODEL"]["ARCHITECTURE"] = "resunet"
            if cfg["PROBLEM"]["NDIM"] == "3D":
                cfg["MODEL"]["Z_DOWN"] = [1, 1, 1, 1]
        elif cfg["PROBLEM"]["TYPE"] == "CLASSIFICATION":
            cfg["MODEL"]["ARCHITECTURE"] = "vit"

    if cfg["TRAIN"]["ENABLE"]:
        cfg.setdefault("AUGMENTOR", {})
        cfg["AUGMENTOR"]["ENABLE"] = True
        cfg["AUGMENTOR"]["AFFINE_MODE"] = "reflect"
        cfg["AUGMENTOR"]["VFLIP"] = True
        cfg["AUGMENTOR"]["HFLIP"] = True
        if cfg["PROBLEM"]["NDIM"] == "3D":
            cfg["AUGMENTOR"]["ZFLIP"] = True

    data_channels = cfg["DATA"]["PATCH_SIZE"][-1]
    if cfg["PROBLEM"]["TYPE"] in ["SEMANTIC_SEG", "INSTANCE_SEG", "DETECTION", "IMAGE_TO_IMAGE"]:
        if cfg["TRAIN"]["ENABLE"]:
            cfg["AUGMENTOR"]["BRIGHTNESS"] = True
            cfg["AUGMENTOR"]["BRIGHTNESS_FACTOR"] = str((-0.1, 0.1))
            cfg["AUGMENTOR"]["CONTRAST"] = True
            cfg["AUGMENTOR"]["CONTRAST_FACTOR"] = str((-0.1, 0.1))
            cfg["AUGMENTOR"]["ELASTIC"] = True
            cfg["AUGMENTOR"]["ZOOM"] = True
            cfg["AUGMENTOR"]["ZOOM_RANGE"] = str((0.9, 1.1))
            cfg["AUGMENTOR"]["RANDOM_ROT"] = True

        if cfg["PROBLEM"]["TYPE"] == "INSTANCE_SEG":
            cfg["PROBLEM"].setdefault("INSTANCE_SEG", {})
            # Unless the representation was chosen in the wizard (biapy.wizard.instance_seg)
            if "DATA_CHANNELS" not in cfg["PROBLEM"]["INSTANCE_SEG"]:
                cfg["PROBLEM"]["INSTANCE_SEG"]["DATA_CHANNELS"] = "BC"
                cfg["PROBLEM"]["INSTANCE_SEG"]["DATA_MW_TH_TYPE"] = "auto"
        elif cfg["PROBLEM"]["TYPE"] == "DETECTION":
            cfg["PROBLEM"].setdefault("DETECTION", {})
            cfg["PROBLEM"]["DETECTION"]["CHECK_POINTS_CREATED"] = False
            if cfg["TEST"]["ENABLE"]:
                cfg["TEST"].setdefault("POST_PROCESSING", {})
                cfg["TEST"]["POST_PROCESSING"].setdefault("REMOVE_CLOSE_POINTS_RADIUS", 5)
                cfg["TEST"]["DET_TOLERANCE"] = int(0.8 * cfg["TEST"]["POST_PROCESSING"]["REMOVE_CLOSE_POINTS_RADIUS"])
                cfg["TEST"]["DET_MIN_TH_TO_BE_PEAK"] = 0.5
                cfg["TEST"]["POST_PROCESSING"]["REMOVE_CLOSE_POINTS"] = True
    elif cfg["PROBLEM"]["TYPE"] == "DENOISING":
        if cfg["DATA"]["PATCH_SIZE"][0] == -1:
            if cfg["PROBLEM"]["NDIM"] == "2D":
                cfg["DATA"]["PATCH_SIZE"] = (64, 64) + (data_channels,)
            else:
                cfg["DATA"]["PATCH_SIZE"] = (12, 64, 64) + (data_channels,)
            if cfg["TEST"]["ENABLE"]:
                cfg["DATA"]["TEST"]["PADDING"] = str(tuple([x // 6 for x in cfg["DATA"]["PATCH_SIZE"][:-1]]))
        cfg["PROBLEM"].setdefault("DENOISING", {})["N2V_STRUCTMASK"] = True
        cfg["MODEL"]["ARCHITECTURE"] = "unet"
        cfg["MODEL"]["FEATURE_MAPS"] = [32, 64, 96]
        if cfg["PROBLEM"]["NDIM"] == "3D":
            cfg["MODEL"]["Z_DOWN"] = [1, 1]
        cfg["MODEL"]["KERNEL_SIZE"] = 3
        cfg["MODEL"]["UPSAMPLE_LAYER"] = "upsampling"
        cfg["MODEL"]["DROPOUT_VALUES"] = [0, 0, 0]
        cfg["MODEL"]["ACTIVATION"] = "relu"
        # (BiaPy-GUI also sets MODEL.LAST_ACTIVATION, removed from BiaPy's configuration)
        cfg["MODEL"]["NORMALIZATION"] = "bn"
    elif cfg["PROBLEM"]["TYPE"] == "SUPER_RESOLUTION":
        if cfg["DATA"]["PATCH_SIZE"][0] == -1:
            if cfg["PROBLEM"]["NDIM"] == "2D":
                cfg["DATA"]["PATCH_SIZE"] = (48, 48) + (data_channels,)
            else:
                cfg["DATA"]["PATCH_SIZE"] = (6, 128, 128) + (data_channels,)
            if cfg["TEST"]["ENABLE"]:
                cfg["DATA"]["TEST"]["PADDING"] = str(tuple([x // 6 for x in cfg["DATA"]["PATCH_SIZE"][:-1]]))
        cfg["DATA"]["NORMALIZATION"]["TYPE"] = "scale_range"

    if not cfg["TEST"]["ENABLE"] or cfg["PROBLEM"]["TYPE"] != "DETECTION":
        if "POST_PROCESSING" in cfg["TEST"] and "REMOVE_CLOSE_POINTS_RADIUS" in cfg["TEST"]["POST_PROCESSING"]:
            del cfg["TEST"]["POST_PROCESSING"]["REMOVE_CLOSE_POINTS_RADIUS"]
            if len(cfg["TEST"]["POST_PROCESSING"]) == 0:
                del cfg["TEST"]["POST_PROCESSING"]

    if cfg["TRAIN"]["ENABLE"]:
        cfg["TRAIN"]["BATCH_SIZE"] = 1

    cfg["DATA"]["PATCH_SIZE"] = str(tuple(cfg["DATA"]["PATCH_SIZE"]))
    if cfg["PROBLEM"]["TYPE"] == "SUPER_RESOLUTION":
        cfg["PROBLEM"]["SUPER_RESOLUTION"]["UPSCALING"] = str(tuple(cfg["PROBLEM"]["SUPER_RESOLUTION"]["UPSCALING"]))
    return cfg


class WizardError(Exception):
    pass


def _size(shape):
    """Image size as "512x512" (ZxYxX in 3D), from its shape without channels."""
    return "x".join(str(int(v)) for v in shape)


def _factor(ratios):
    """Upsampling factor as "x2", or "x(1, 2, 2)" when it differs per axis."""
    ratios = ["{:g}".format(r) for r in ratios]
    return "x" + ratios[0] if len(set(ratios)) == 1 else "x(" + ", ".join(ratios) + ")"


def _check_target_shapes(workflow, phase, raw, targets, upscaling=None):
    """
    Checks that each target has the shape of its raw image or, in super-resolution, the shape of the raw image
    upsampled by an integer factor: the same factor for all the images, training and test.

    Parameters
    ----------
    workflow : str
        ``PROBLEM.TYPE``.
    phase : str
        "TRAIN" or "TEST".
    raw, targets : tuple
        Shapes of the raw images and of the targets (channels last) and the folder they are in.
    upscaling : dict, optional
        Super-resolution: the factor found so far (``factor``) and in which images (``found``).

    Returns
    -------
    upscaling : dict or None
        Super-resolution: the factor (``factor``, one per axis) and in which images it was found (``found``).
    """
    from biapy.wizard.data_check import list_files

    which = "training" if phase == "TRAIN" else "test"
    (raw_shapes, raw_folder), (target_shapes, target_folder) = raw, targets
    raw_names, target_names = list_files(raw_folder or ""), list_files(target_folder or "")

    for i, (x, y) in enumerate(zip(raw_shapes, target_shapes)):
        x, y = tuple(x[:-1]), tuple(y[:-1])
        # File names, in the order the data check read them (if the folders have not changed since)
        raw_name = raw_names[i] if len(raw_names) == len(raw_shapes) else "number {}".format(i + 1)
        target_name = target_names[i] if len(target_names) == len(target_shapes) else "number {}".format(i + 1)

        def pair():
            return "\n    - raw image {}: {}  (in {})\n    - its target {}: {}  (in {})".format(
                raw_name, _size(x), raw_folder, target_name, _size(y), target_folder
            )

        if workflow != "SUPER_RESOLUTION":
            if x != y:
                raise WizardError(
                    "Each {} target must have the same size as its raw image, but this pair does not:{}\n"
                    "Check that both folders hold the right images (they are paired by file order).".format(which, pair())
                )
            continue

        ratios = [b / a for a, b in zip(x, y)]
        if any(r < 1 for r in ratios):
            raise WizardError(
                "In super-resolution the raw images are the low-resolution ones and their targets the high-resolution "
                "ones (e.g. 512x512 raw images and 1024x1024 targets for a x2 upsampling), but the {} targets are smaller "
                "than their raw images:{}\nCheck that the {} raw images folder holds the low-resolution images and the "
                "target folder the high-resolution ones: they may be swapped.".format(which, pair(), which)
            )
        if any(r % 1 != 0 for r in ratios):
            raise WizardError(
                "In super-resolution each target must be its raw image upsampled by a whole factor (e.g. 512x512 raw "
                "images and 1024x1024 targets for a x2 upsampling), but in this {} pair the target is {} larger:{}\n"
                "Check that both folders hold the right images (they are paired by file order).".format(
                    which, _factor(ratios), pair()
                )
            )
        factor = tuple(int(r) for r in ratios)
        if upscaling is None:
            upscaling = {"factor": factor, "found": "the {} data (e.g. raw image {}: {}, its target: {})".format(
                which, raw_name, _size(x), _size(y))}
        elif factor != upscaling["factor"]:
            raise WizardError(
                "In super-resolution all the targets must be their raw images upsampled by the same factor, in the "
                "training and the test data. It is {} in {}, but {} in this {} pair:{}\nCheck that "
                "the {} raw images are the low-resolution ones and their targets the high-resolution ones.".format(
                    _factor(upscaling["factor"]), upscaling["found"], _factor(factor), which, pair(),
                    which,
                )
            )
    return upscaling


def build_config_from_answers(answers):
    """
    Configuration dictionary from the wizard's answers, checking that they are consistent (BiaPy-GUI's
    ui_utils.export_wizard_summary(), minus the dialogs). See :func:`build_config`.
    """
    biapy_cfg = lists_to_tuples(dict(answers))
    biapy_cfg["data_constraints"] = dict(biapy_cfg.get("data_constraints", {}))
    sample_info = biapy_cfg.pop("sample_info", {})

    # Instance representation chosen in the wizard (not a BiaPy variable): translated into BiaPy's channels.
    # Only asked for models trained from scratch (an answer given before changing that may remain).
    representation = biapy_cfg.pop("instance_representation", -1)
    if (
        representation != -1
        and biapy_cfg.get("PROBLEM.TYPE") == "INSTANCE_SEG"
        and biapy_cfg.get("MODEL.SOURCE") == "biapy"
        and not biapy_cfg.get("MODEL.LOAD_CHECKPOINT")
    ):
        from biapy.wizard.instance_seg import instance_creation, representation_to_biapy

        channels, extra = representation_to_biapy(representation, 3 if biapy_cfg["PROBLEM.NDIM"] == "3D" else 2)
        biapy_cfg["PROBLEM.INSTANCE_SEG.DATA_CHANNELS"] = channels
        biapy_cfg["PROBLEM.INSTANCE_SEG.DATA_CHANNELS_EXTRA_OPTS"] = extra
        biapy_cfg["PROBLEM.INSTANCE_SEG.INSTANCE_CREATION_PROCESS"] = instance_creation(representation)

    data_imposed_classes = biapy_cfg["data_constraints"].get("DATA.N_CLASSES", 2)
    if "model_restrictions" in biapy_cfg and "DATA.N_CLASSES" in biapy_cfg["model_restrictions"]:
        model_imposed_classes = max(2, biapy_cfg["model_restrictions"]["DATA.N_CLASSES"])
    else:
        model_imposed_classes = data_imposed_classes
    if data_imposed_classes != model_imposed_classes:
        raise WizardError(
            "Incompatibility found: the data provided seems to have {} classes whereas the pretrained model is prepared "
            "to work with {} classes. Please select another pretrained model.".format(
                data_imposed_classes, model_imposed_classes
            )
        )

    if "model_restrictions" in biapy_cfg:
        for key, value in biapy_cfg["model_restrictions"].items():
            if key == "TRAIN.ENABLE" and not value and biapy_cfg.get("TRAIN.ENABLE", False):
                raise WizardError(
                    "The selected pretrained model is not prepared to be trained. Please select another pretrained model."
                )
            biapy_cfg[key] = value
        del biapy_cfg["model_restrictions"]
    else:
        # The wizard does not ask for the object size to choose the patch size: it is set after the configuration
        # is built, from the data or the checkpoint to load (biapy.wizard.planning)
        biapy_cfg["DATA.PATCH_SIZE"] = (-1, biapy_cfg["data_constraints"]["DATA.PATCH_SIZE_C"])

    if isinstance(biapy_cfg["DATA.PATCH_SIZE"], str):  # imposed by a model as a string
        import ast

        biapy_cfg["DATA.PATCH_SIZE"] = tuple(ast.literal_eval(biapy_cfg["DATA.PATCH_SIZE"]))
    model_channels = biapy_cfg["DATA.PATCH_SIZE"][-1]
    data_channels = biapy_cfg["data_constraints"]["DATA.PATCH_SIZE_C"]
    if data_channels != model_channels:
        raise WizardError(
            "Incompatibility found: the data provided seems to have {} channels whereas the pretrained model expects {} "
            "channels. Please select another pretrained model.".format(data_channels, model_channels)
        )
    del biapy_cfg["data_constraints"]["DATA.PATCH_SIZE_C"]

    dc = biapy_cfg["data_constraints"]
    upscaling = None  # super-resolution: the same in the training and test data
    for phase in ["TRAIN", "TEST"]:
        if biapy_cfg["{}.ENABLE".format(phase)] and "DATA.{}.GT_PATH".format(phase) in dc:
            if dc["DATA.{}.GT_PATH".format(phase)] != dc["DATA.{}.PATH".format(phase)]:
                raise WizardError(
                    "Incompatibility found: the number of raw images and ground truth images do not match. Each raw image must have a "
                    "ground truth image. Please check both directories:\n    - {} items found in {}\n    - {} items "
                    "found in {}\n".format(
                        dc["DATA.{}.PATH".format(phase)],
                        dc["DATA.{}.PATH_path".format(phase)],
                        dc["DATA.{}.GT_PATH".format(phase)],
                        dc["DATA.{}.GT_PATH_path".format(phase)],
                    )
                )
            x_key, y_key = "DATA.{}.PATH_path_shapes".format(phase), "DATA.{}.GT_PATH_path_shapes".format(phase)
            if x_key in dc and y_key in dc:
                upscaling = _check_target_shapes(
                    biapy_cfg["PROBLEM.TYPE"],
                    phase,
                    (dc[x_key], dc.get("DATA.{}.PATH_path".format(phase))),
                    (dc[y_key], dc.get("DATA.{}.GT_PATH_path".format(phase))),
                    upscaling,
                )
                if biapy_cfg["PROBLEM.TYPE"] == "SUPER_RESOLUTION":
                    biapy_cfg["PROBLEM.SUPER_RESOLUTION.UPSCALING"] = upscaling["factor"]

        for suffix in ["PATH_path_shapes", "GT_PATH_path_shapes", "PATH", "PATH_path", "GT_PATH", "GT_PATH_path"]:
            dc.pop("DATA.{}.{}".format(phase, suffix), None)

    for key, value in dc.items():
        biapy_cfg[key] = value
    del biapy_cfg["data_constraints"]

    for k in [x for x in biapy_cfg.keys() if "CHECKED " in x]:
        del biapy_cfg[k]

    # Asked only for denoising, but it may remain answered after switching to another workflow
    if biapy_cfg.get("PROBLEM.TYPE") != "DENOISING":
        biapy_cfg.pop("PROBLEM.DENOISING.LOAD_GT_DATA", None)

    out_config = {}
    for key, value in biapy_cfg.items():
        if value != -1:
            create_dict_from_key(key, value, out_config)
    out_config = set_default_config(out_config, sample_info)

    # The questions may set variables BiaPy has renamed since (e.g. BiaPy-GUI's): translated as BiaPy does
    # when it loads a configuration, so the file has the current ones
    from biapy.engine.check_configuration import convert_old_model_cfg_to_current_version

    return convert_old_model_cfg_to_current_version(out_config)


def build_config(answers, gpus=None):
    """
    Configuration from the wizard's answers, adapted to the data.

    Parameters
    ----------
    answers : dict
        The wizard's answers, ``{variable: value}`` (e.g. ``"PROBLEM.TYPE": "SEMANTIC_SEG"``) with -1 for the
        questions not answered, plus ``data_constraints`` and ``sample_info`` gathered from the data checks
        (:func:`biapy.wizard.data_check.check_folder`) and ``instance_representation`` if it was asked.
    gpus : list of dict, optional
        GPUs to plan for (``index``, ``name``, ``memory_mb``). Queried with nvidia-smi if not given.

    Returns
    -------
    cfg : dict
        Configuration, to be written with :func:`write_config`.
    report : dict
        ``applied`` (whether it was adapted to the data), ``reason`` when not, and ``lines``, a human readable
        summary of what was decided.

    Raises
    ------
    WizardError
        If the answers are not consistent (e.g. the data does not fit the pretrained model selected).
    """
    import traceback

    from biapy.wizard.planning import plan_config, query_gpus

    cfg = build_config_from_answers(answers)
    try:
        report = plan_config(cfg, answers.get("sample_info", {}), query_gpus() if gpus is None else gpus)
    except Exception:  # the configuration is valid without the plan
        print(traceback.format_exc(), flush=True)
        report = {"applied": False, "reason": "planning failed, see the log"}
    return cfg, report


def write_config(cfg, report, yaml_file, tool="the BiaPy wizard"):
    """Writes the configuration as YAML, with the summary of ``report`` on top as comments."""
    import yaml

    os.makedirs(os.path.dirname(os.path.abspath(yaml_file)), exist_ok=True)
    with open(yaml_file, "w", encoding="utf8") as f:
        if report.get("lines"):
            f.write("# Configuration {} by {}:\n".format("adapted to the data" if report["applied"] else "set", tool))
            f.write("".join("#   {}\n".format(line) for line in report["lines"]))
        yaml.safe_dump(tuples_to_lists(cfg), f, default_flow_style=False)
