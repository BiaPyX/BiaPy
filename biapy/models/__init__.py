import importlib
import os
import json
from pathlib import Path
import pooch
import yaml
import torch
import functools
import torch.nn as nn
from torchinfo import summary
from typing import Optional, Dict, Tuple, List, Callable
from packaging.version import Version
from functools import partial
from yacs.config import CfgNode as CN
import numpy as np

from bioimageio.spec.utils import download
from bioimageio.core.backends.pytorch_backend import load_torch_model
from bioimageio.spec.model.v0_4 import ModelDescr as ModelDescr_v0_4
from bioimageio.spec.model.v0_5 import ModelDescr as ModelDescr_v0_5
from bioimageio.spec import InvalidDescr
from bioimageio.core.digest_spec import get_test_inputs


def build_model(cfg: CN, output_channels: int, device: torch.device) -> Tuple[nn.Module, str, Callable, Dict, Tuple[int, ...]]:
    # model, model_file, model_name, args
    """
    Build selected model

    Parameters
    ----------
    cfg : YACS CN object
        Configuration.

    output_channels : int
        Number of output channels.

    device : Torch device
        Using device. Most commonly "cpu" or "cuda" for GPU, but also potentially "mps",
        "xpu", "xla" or "meta".

    Returns
    -------
    model : Pytorch model
        Selected model.
    """
    # Import the model
    if "efficientnet" in cfg.MODEL.ARCHITECTURE.lower():
        modelname = "efficientnet"
    elif "hrnet" in cfg.MODEL.ARCHITECTURE.lower():
        modelname = "hrnet"
    else:
        modelname = str(cfg.MODEL.ARCHITECTURE).lower()
    mdl = importlib.import_module("biapy.models." + modelname)
    model_file = os.path.abspath(mdl.__file__)  # type: ignore
    names = [x for x in mdl.__dict__ if not x.startswith("_")]
    globals().update({k: getattr(mdl, k) for k in names})

    ndim = 3 if cfg.PROBLEM.NDIM == "3D" else 2
    network_stride = None

    # Put again the specific model name
    if "hrnet" in cfg.MODEL.ARCHITECTURE.lower():
        modelname = cfg.MODEL.ARCHITECTURE.lower()

    # Model building
    if modelname in [
        "unet",
        "resunet",
        "resunet++",
        "seunet",
        "resunet_se",
        "attention_unet",
        "unext_v1",
        "unext_v2",
    ]:
        args = dict(
            image_shape=cfg.DATA.PATCH_SIZE,
            activation=cfg.MODEL.ACTIVATION.lower(),
            feature_maps=cfg.MODEL.FEATURE_MAPS,
            drop_values=cfg.MODEL.DROPOUT_VALUES,
            normalization=cfg.MODEL.NORMALIZATION,
            k_size=cfg.MODEL.KERNEL_SIZE,
            upsample_layer=cfg.MODEL.UPSAMPLE_LAYER,
            z_down=cfg.MODEL.Z_DOWN,
            output_channels=output_channels,
            contrast=cfg.LOSS.CONTRAST.ENABLE, 
            contrast_proj_dim=cfg.LOSS.CONTRAST.PROJ_DIM, 
        )
        if modelname == "unet":
            callable_model = U_Net  # type: ignore
        elif modelname == "resunet":
            callable_model = ResUNet  # type: ignore
            args["isotropy"] = cfg.MODEL.ISOTROPY
            args["larger_io"] = cfg.MODEL.LARGER_IO
        elif modelname == "resunet++":
            callable_model = ResUNetPlusPlus  # type: ignore
        elif modelname == "attention_unet":
            callable_model = Attention_U_Net  # type: ignore
        elif modelname == "seunet":
            callable_model = SE_U_Net  # type: ignore
            args["isotropy"] = cfg.MODEL.ISOTROPY
            args["larger_io"] = cfg.MODEL.LARGER_IO
        elif modelname == "resunet_se":
            callable_model = ResUNet_SE  # type: ignore
            args["isotropy"] = cfg.MODEL.ISOTROPY
            args["larger_io"] = cfg.MODEL.LARGER_IO
        elif modelname == "unext_v1":
            args = dict(
                image_shape=cfg.DATA.PATCH_SIZE,
                feature_maps=cfg.MODEL.FEATURE_MAPS,
                upsample_layer=cfg.MODEL.UPSAMPLE_LAYER,
                z_down=cfg.MODEL.Z_DOWN,
                cn_layers=cfg.MODEL.CONVNEXT_LAYERS,
                layer_scale=cfg.MODEL.CONVNEXT_LAYER_SCALE,
                stochastic_depth_prob=cfg.MODEL.CONVNEXT_SD_PROB,
                isotropy=cfg.MODEL.ISOTROPY,
                stem_k_size=cfg.MODEL.CONVNEXT_STEM_K_SIZE,
                output_channels=output_channels,
            )
            callable_model = U_NeXt_V1  # type: ignore
        elif modelname == "unext_v2":
            args = dict(
                image_shape=cfg.DATA.PATCH_SIZE,
                feature_maps=cfg.MODEL.FEATURE_MAPS,
                upsample_layer=cfg.MODEL.UPSAMPLE_LAYER,
                z_down=cfg.MODEL.Z_DOWN,
                cn_layers=cfg.MODEL.CONVNEXT_LAYERS,
                stochastic_depth_prob=cfg.MODEL.CONVNEXT_SD_PROB,
                isotropy=cfg.MODEL.ISOTROPY,
                stem_k_size=cfg.MODEL.CONVNEXT_STEM_K_SIZE,
                output_channels=output_channels,
            )
            callable_model = U_NeXt_V2  # type: ignore

        if cfg.PROBLEM.TYPE == "SUPER_RESOLUTION":
            args["upsampling_factor"] = cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING
            args["upsampling_position"] = cfg.MODEL.UNET_SR_UPSAMPLE_POSITION

        network_stride = [ 
            len(cfg.MODEL.Z_DOWN), 
            len(cfg.MODEL.Z_DOWN)
        ]
        if ndim == 3:
            network_stride = [np.sum([1 for x in cfg.MODEL.Z_DOWN if x == 2])] + network_stride
        model = callable_model(**args)

    elif "hrnet" in modelname:
        z_down = any([x for x in cfg.MODEL.Z_DOWN if x == 2])
        args = dict(
            image_shape=cfg.DATA.PATCH_SIZE,
            normalization='sync_bn',
            output_channels=output_channels,
            contrast=cfg.LOSS.CONTRAST.ENABLE, 
            contrast_proj_dim=cfg.LOSS.CONTRAST.PROJ_DIM, 
            z_down=z_down,
        )
        if modelname == "hrnet18":
            args["cfg"] = cfg.MODEL.HRNET_18
        elif modelname == 'hrnet32':
            args["cfg"] = cfg.MODEL.HRNET_32
        elif modelname == 'hrnet48':
            args["cfg"] = cfg.MODEL.HRNET_48
        elif modelname == 'hrnet64':
            args = dict(
                cfg.MODEL.HRNET_64,
                norm='sync_bn',
                )
        elif modelname == 'hrnet2x20':
            args["cfg"] = cfg.MODEL.HRNET2X_20
        callable_model = HighResolutionNet  # type: ignore
        model = callable_model(**args)

        network_stride = [4, 4]
        if ndim == 3:
            network_stride = [4 if z_down else 1] + network_stride
    else:
        if modelname == "simple_cnn":
            args = dict(
                image_shape=cfg.DATA.PATCH_SIZE,
                activation=cfg.MODEL.ACTIVATION.lower(),
                n_classes=cfg.DATA.N_CLASSES,
            )
            model = simple_CNN(**args)  # type: ignore
            callable_model = simple_CNN  # type: ignore
        elif "efficientnet" in modelname:
            args = dict(efficientnet_name=cfg.MODEL.ARCHITECTURE.lower(), n_classes=cfg.DATA.N_CLASSES)
            model = efficientnet(**args)  # type: ignore
            callable_model = efficientnet  # type: ignore
        elif modelname == "vit":
            args = dict(
                img_size=cfg.DATA.PATCH_SIZE[0],
                patch_size=cfg.MODEL.VIT_TOKEN_SIZE,
                in_chans=cfg.DATA.PATCH_SIZE[-1],
                ndim=ndim,
                num_classes=cfg.DATA.N_CLASSES,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
            )
            if cfg.MODEL.VIT_MODEL == "custom":
                args2 = dict(
                    embed_dim=cfg.MODEL.VIT_EMBED_DIM,
                    depth=cfg.MODEL.VIT_NUM_LAYERS,
                    num_heads=cfg.MODEL.VIT_NUM_HEADS,
                    mlp_ratio=cfg.MODEL.VIT_MLP_RATIO,
                    drop_rate=cfg.MODEL.DROPOUT_VALUES[0],
                )
                args.update(args2)
                model = VisionTransformer(**args)  # type: ignore
                callable_model = VisionTransformer  # type: ignore
            else:
                model = eval(cfg.MODEL.VIT_MODEL)(**args)
                callable_model = eval(cfg.MODEL.VIT_MODEL)
        elif modelname == "multiresunet":
            args = dict(
                input_channels=cfg.DATA.PATCH_SIZE[-1],
                ndim=ndim,
                alpha=1.67,
                z_down=cfg.MODEL.Z_DOWN,
                output_channels=output_channels,
            )
            if cfg.PROBLEM.TYPE == "SUPER_RESOLUTION":
                args["upsampling_factor"] = cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING
                args["upsampling_position"] = cfg.MODEL.UNET_SR_UPSAMPLE_POSITION

            model = MultiResUnet(**args)  # type: ignore
            callable_model = MultiResUnet  # type: ignore
        elif modelname == "unetr":
            args = dict(
                input_shape=cfg.DATA.PATCH_SIZE,
                patch_size=cfg.MODEL.VIT_TOKEN_SIZE,
                embed_dim=cfg.MODEL.VIT_EMBED_DIM,
                depth=cfg.MODEL.VIT_NUM_LAYERS,
                num_heads=cfg.MODEL.VIT_NUM_HEADS,
                mlp_ratio=cfg.MODEL.VIT_MLP_RATIO,
                num_filters=cfg.MODEL.UNETR_VIT_NUM_FILTERS,
                output_channels=output_channels,
                decoder_activation=cfg.MODEL.UNETR_DEC_ACTIVATION,
                ViT_hidd_mult=cfg.MODEL.UNETR_VIT_HIDD_MULT,
                normalization=cfg.MODEL.NORMALIZATION,
                dropout=cfg.MODEL.DROPOUT_VALUES[0],
                k_size=cfg.MODEL.UNETR_DEC_KERNEL_SIZE,
            )
            model = UNETR(**args)  # type: ignore
            callable_model = UNETR  # type: ignore
        elif modelname == "edsr":
            args = dict(
                ndim=ndim,
                num_filters=64,
                num_of_residual_blocks=16,
                upsampling_factor=cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING,
                num_channels=cfg.DATA.PATCH_SIZE[-1],
            )
            model = EDSR(args)  # type: ignore
            callable_model = EDSR  # type: ignore
        elif modelname == "rcan":
            scale = cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING
            if type(scale) is tuple:
                scale = scale[0]
            args = dict(
                ndim=ndim,
                filters=cfg.MODEL.RCAN_CONV_FILTERS,
                scale=scale,
                num_rg=cfg.MODEL.RCAN_RG_BLOCK_NUM,
                num_rcab=cfg.MODEL.RCAN_RCAB_BLOCK_NUM,
                reduction=cfg.MODEL.RCAN_REDUCTION_RATIO,
                num_channels=cfg.DATA.PATCH_SIZE[-1],
                upscaling_layer=cfg.MODEL.RCAN_UPSCALING_LAYER,
            )
            model = rcan(**args)  # type: ignore
            callable_model = rcan  # type: ignore
        elif modelname == "dfcan":
            args = dict(
                ndim=ndim,
                input_shape=cfg.DATA.PATCH_SIZE,
                scale=cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING,
                n_ResGroup=4,
                n_RCAB=4,
            )
            model = DFCAN(**args)  # type: ignore
            callable_model = DFCAN  # type: ignore
        elif modelname == "wdsr":
            args = dict(
                scale=cfg.PROBLEM.SUPER_RESOLUTION.UPSCALING,
                num_filters=32,
                num_res_blocks=8,
                res_block_expansion=6,
                num_channels=cfg.DATA.PATCH_SIZE[-1],
            )
            model = wdsr(**args)  # type: ignore
            callable_model = wdsr  # type: ignore
        elif modelname == "mae":
            args = dict(
                img_size=cfg.DATA.PATCH_SIZE[0],
                patch_size=cfg.MODEL.VIT_TOKEN_SIZE,
                in_chans=cfg.DATA.PATCH_SIZE[-1],
                ndim=ndim,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                embed_dim=cfg.MODEL.VIT_EMBED_DIM,
                depth=cfg.MODEL.VIT_NUM_LAYERS,
                num_heads=cfg.MODEL.VIT_NUM_HEADS,
                decoder_embed_dim=512,
                decoder_depth=8,
                decoder_num_heads=16,
                mlp_ratio=cfg.MODEL.VIT_MLP_RATIO,
                masking_type=cfg.MODEL.MAE_MASK_TYPE,
                mask_ratio=cfg.MODEL.MAE_MASK_RATIO,
                device=device,
            )
            model = MaskedAutoencoderViT(**args)  # type: ignore
            callable_model = MaskedAutoencoderViT  # type: ignore
    # Check the network created
    model.to(device)
    if cfg.PROBLEM.NDIM == "2D":
        sample_size = (
            1,
            cfg.DATA.PATCH_SIZE[2],
            cfg.DATA.PATCH_SIZE[0],
            cfg.DATA.PATCH_SIZE[1],
        )
    else:
        sample_size = (
            1,
            cfg.DATA.PATCH_SIZE[3],
            cfg.DATA.PATCH_SIZE[0],
            cfg.DATA.PATCH_SIZE[1],
            cfg.DATA.PATCH_SIZE[2],
        )
    # if cfg.PROBLEM.NDIM == "2D":
    # sample_size = (
    #     1,
    #     3,
    #     512,
    #     1024,
    # )
    # else:
    #     sample_size = (
    #         1,
    #         cfg.DATA.PATCH_SIZE[3],
    #         cfg.DATA.PATCH_SIZE[0],
    #         cfg.DATA.PATCH_SIZE[1],
    #         cfg.DATA.PATCH_SIZE[2],
    #     )
    summary(
        model,
        input_size=sample_size,
        col_names=("input_size", "output_size", "num_params"),
        depth=10,
        device=device.type,
    )
    # import pdb; pdb.set_trace()
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    model_file += ":" + str(callable_model.__name__)
    model_name = model_file.rsplit(":", 1)[-1]
    return model, model_file, model_name, args, network_stride


def build_bmz_model(cfg: CN, model: ModelDescr_v0_4 | ModelDescr_v0_5, device: torch.device) -> nn.Module:
    """
    Build a model from Bioimage Model Zoo (BMZ).

    Parameters
    ----------
    cfg : YACS configuration
        Running configuration.

    model : ModelDescr
        BMZ model RDF that contains all the information of the model.

    device : Torch device
        Device used.

    Returns
    -------
    model_instance : Torch model
        Torch model.
    """
    assert model.weights.pytorch_state_dict
    model_instance = load_torch_model(model.weights.pytorch_state_dict, load_state=True, devices=[device])

    # Check the network created
    if cfg.PROBLEM.NDIM == "2D":
        sample_size = (
            1,
            cfg.DATA.PATCH_SIZE[2],
            cfg.DATA.PATCH_SIZE[0],
            cfg.DATA.PATCH_SIZE[1],
        )
    else:
        sample_size = (
            1,
            cfg.DATA.PATCH_SIZE[3],
            cfg.DATA.PATCH_SIZE[0],
            cfg.DATA.PATCH_SIZE[1],
            cfg.DATA.PATCH_SIZE[2],
        )
    summary(
        model_instance,
        input_size=sample_size,
        col_names=("input_size", "output_size", "num_params"),
        depth=10,
        device=device.type,
    )

    return model_instance


def check_bmz_args(
    model_ID: str,
    cfg: CN,
) -> List:
    """
    Check user's provided BMZ arguments.

    Parameters
    ----------
    model_ID : str
        Model identifier. It can be either its ``DOI`` or ``nickname``.

    cfg : YACS configuration
        Running configuration.

    Returns
    -------
    preproc_info: dict
        Preprocessing names that the model is using.
    """
    # Checking BMZ model compatibility using the available model list provided by BMZ
    COLLECTION_URL = "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/collection.json"
    # COLLECTION_URL = "https://raw.githubusercontent.com/bioimage-io/collection-bioimage-io/gh-pages/collection.json"
    collection_path = Path(pooch.retrieve(COLLECTION_URL, known_hash=None))
    with collection_path.open() as f:
        collection = json.load(f)

    # Find the model among all
    model_urls = [
        entry
        for entry in collection["collection"]
        if entry["type"] == "model"
        and (
            ("nickname" in entry and model_ID in entry["nickname"])
            or ("id" in entry and model_ID in entry["id"])
            or ("rdf_source" in entry and model_ID in entry["rdf_source"])
        )
    ]

    if len(model_urls) == 0:
        raise ValueError(f"No model found with the provided DOI/name: {model_ID}")
    if len(model_urls) > 1:
        raise ValueError(f"More than one model found with the provided DOI/name ({model_ID}). Contact BiaPy team.")
    with open(Path(pooch.retrieve(model_urls[0]["rdf_source"], known_hash=None))) as stream:
        try:
            model_rdf = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    workflow_specs = {}
    workflow_specs["workflow_type"] = cfg.PROBLEM.TYPE
    workflow_specs["ndim"] = cfg.PROBLEM.NDIM
    workflow_specs["nclasses"] = cfg.DATA.N_CLASSES

    preproc_info, error, error_message = check_bmz_model_compatibility(model_rdf, workflow_specs=workflow_specs)

    if error:
        raise ValueError(f"Model {model_ID} can not be used in BiaPy. Message:\n{error_message}\n")

    return preproc_info


def check_bmz_model_compatibility(
    model_rdf: Dict,
    workflow_specs: Optional[Dict] = None,
) -> Tuple[List, bool, str]:
    """
    Checks one model compatibility with BiaPy by looking at its RDF file provided by BMZ. This function is the one
    used in BMZ's continuous integration with BiaPy.

    Parameters
    ----------
    model_rdf : dict
        BMZ model RDF that contains all the information of the model.

    workflow_specs : dict
        Specifications of the workflow. If not provided all possible models will be considered.

    Returns
    -------
    preproc_info: dict
        Preprocessing names that the model is using.

    error : bool
        Whether it there is a problem to consume the model in BiaPy or not.

    reason_message: str
        Reason why the model can not be consumed if there is any.
    """
    specific_workflow = "all" if workflow_specs is None else workflow_specs["workflow_type"]
    specific_dims = "all" if workflow_specs is None else workflow_specs["ndim"]
    ref_classes = "all" if workflow_specs is None else workflow_specs["nclasses"]

    preproc_info = []

    # Accepting models that are exported in pytorch_state_dict and with just one input
    if (
        "pytorch_state_dict" in model_rdf["weights"]
        and model_rdf["weights"]["pytorch_state_dict"] is not None
        and len(model_rdf["inputs"]) == 1
    ):

        # TODO: control model.weights.pytorch_state_dict.dependencies conda env to check if all
        # dependencies are installed
        # https://github.com/bioimage-io/collection-bioimage-io/issues/609

        model_version = Version("0.5")
        if "format_version" in model_rdf:
            model_version = Version(model_rdf["format_version"])

        # Capture model kwargs
        model_kwargs = None
        if "kwargs" in model_rdf["weights"]["pytorch_state_dict"]:
            model_kwargs = model_rdf["weights"]["pytorch_state_dict"]["kwargs"]
        elif (
            "architecture" in model_rdf["weights"]["pytorch_state_dict"]
            and "kwargs" in model_rdf["weights"]["pytorch_state_dict"]["architecture"]
        ):
            model_kwargs = model_rdf["weights"]["pytorch_state_dict"]["architecture"]["kwargs"]
        else:
            return preproc_info, True, f"[{specific_workflow}] Couldn't extract kwargs from model description.\n"

        # Check problem type
        if (specific_workflow in ["all", "SEMANTIC_SEG"]) and (
            "semantic-segmentation" in model_rdf["tags"]
            or ("segmentation" in model_rdf["tags"] and "instance-segmentation" not in model_rdf["tags"])
        ):
            # Check number of classes
            classes = -1
            if "n_classes" in model_kwargs:  # BiaPy
                classes = model_kwargs["n_classes"]
            elif "out_channels" in model_kwargs:
                classes = model_kwargs["out_channels"]
            elif "output_channels" in model_kwargs:
                classes = model_kwargs["output_channels"]
            elif "classes" in model_kwargs:
                classes = model_kwargs["classes"]
            if isinstance(classes, list):
                classes = classes[-1]

            if not isinstance(classes, int):
                reason_message = (
                    f"[{specific_workflow}] 'DATA.N_CLASSES' not extracted. Obtained {classes}. Please check it!\n"
                )
                return preproc_info, True, reason_message

            if isinstance(classes, int) and classes != -1:
                if ref_classes != "all":
                    if classes > 2 and ref_classes != classes:
                        reason_message = f"[{specific_workflow}] 'DATA.N_CLASSES' does not match network's output classes. Please check it!\n"
                        return preproc_info, True, reason_message
            else:
                reason_message = f"[{specific_workflow}] Couldn't find the classes this model is returning so please be aware to match it\n"
                return preproc_info, True, reason_message

        elif specific_workflow in ["all", "INSTANCE_SEG"] and "instance-segmentation" in model_rdf["tags"]:
            # TODO: add cellpose tag and create flow post-processing to create images
            pass
        elif specific_workflow in ["all", "DETECTION"] and "detection" in model_rdf["tags"]:
            pass
        elif specific_workflow in ["all", "DENOISING"] and "denoising" in model_rdf["tags"]:
            pass
        elif specific_workflow in ["all", "SUPER_RESOLUTION"] and (
            "super-resolution" in model_rdf["tags"]
            or "superresolution" in model_rdf["tags"]
        ):
            pass
        elif specific_workflow in ["all", "SELF_SUPERVISED"] and "self-supervision" in model_rdf["tags"]:
            pass
        elif specific_workflow in ["all", "CLASSIFICATION"] and "classification" in model_rdf["tags"]:
            pass
        elif specific_workflow in ["all", "IMAGE_TO_IMAGE"] and (
            "pix2pix" in model_rdf["tags"]
            or "image-reconstruction" in model_rdf["tags"]
            or "image-to-image" in model_rdf["tags"]
            or "image-restoration" in model_rdf["tags"]
        ):
            pass
        else:
            reason_message = "[{}] no workflow tag recognized in {}.\n".format(specific_workflow, model_rdf["tags"])
            return preproc_info, True, reason_message

        # Check axes
        axes_order = model_rdf["inputs"][0]["axes"]
        if isinstance(axes_order, list):
            _axes_order = ""
            for axis in axes_order:
                if "type" in axis:
                    if axis["type"] == "batch":
                        _axes_order += "b"
                    elif axis["type"] == "channel":
                        _axes_order += "c"
                    elif "id" in axis:
                        _axes_order += axis["id"]
                elif "id" in axis:
                    if axis["id"] == "channel":
                        _axes_order += "c"
                    else:
                        _axes_order += axis["id"]
            axes_order = _axes_order

        if specific_dims == "2D":
            if axes_order != "bcyx":
                reason_message = (
                    f"[{specific_workflow}] In a 2D problem the axes need to be 'bcyx', found {axes_order}\n"
                )
                return preproc_info, True, reason_message
            elif "2d" not in model_rdf["tags"] and "3d" in model_rdf["tags"]:
                reason_message = f"[{specific_workflow}] Selected model seems to not be 2D\n"
                return preproc_info, True, reason_message
        elif specific_dims == "3D":
            if axes_order != "bczyx":
                reason_message = (
                    f"[{specific_workflow}] In a 3D problem the axes need to be 'bczyx', found {axes_order}\n"
                )
                return preproc_info, True, reason_message
            elif "3d" not in model_rdf["tags"] and "2d" in model_rdf["tags"]:
                reason_message = f"[{specific_workflow}] Selected model seems to not be 3D\n"
                return preproc_info, True, reason_message
        else:  # All
            if axes_order not in ["bcyx", "bczyx"]:
                reason_message = f"[{specific_workflow}] Accepting models only with ['bcyx', 'bczyx'] axis order, found {axes_order}\n"
                return preproc_info, True, reason_message

        # Check preprocessing
        if "preprocessing" in model_rdf["inputs"][0]:
            preproc_info = model_rdf["inputs"][0]["preprocessing"]
            key_to_find = "id" if model_version > Version("0.5.0") else "name"
            if isinstance(preproc_info, list):
                # Remove "ensure_dtype" preprocessing when casting to float, as BiaPy will always do it like that
                new_preproc_info = []
                for preproc in preproc_info:
                    if key_to_find in preproc and not (
                        preproc[key_to_find] == "ensure_dtype"
                        and "kwargs" in preproc
                        and "dtype" in preproc["kwargs"]
                        and "float" in preproc["kwargs"]["dtype"]
                    ):
                        new_preproc_info.append(preproc)
                preproc_info = new_preproc_info.copy()

                # Then if there is still more than one preprocessing not continue as it is not implemented yet
                if len(preproc_info) > 1:
                    reason_message = (
                        f"[{specific_workflow}] More than one preprocessing from BMZ not implemented yet {axes_order}\n"
                    )
                    return preproc_info, True, reason_message
                elif len(preproc_info) == 1:
                    preproc_info = preproc_info[0]
                    if key_to_find in preproc_info:
                        if preproc_info[key_to_find] not in [
                            "zero_mean_unit_variance",
                            "fixed_zero_mean_unit_variance",
                            "scale_range",
                            "scale_linear",
                        ]:
                            reason_message = f"[{specific_workflow}] Not recognized preprocessing found: {preproc_info[key_to_find]}\n"
                            return preproc_info, True, reason_message
                    else:
                        reason_message = (
                            f"[{specific_workflow}] Not recognized preprocessing structure found: {preproc_info}\n"
                        )
                        return preproc_info, True, reason_message

        # Check post-processing
        if model_kwargs is not None and "postprocessing" in model_kwargs and model_kwargs["postprocessing"] is not None:
            reason_message = f"[{specific_workflow}] Currently no postprocessing is supported. Found: {model_kwargs['postprocessing']}\n"
            return preproc_info, True, reason_message
    else:
        reason_message = f"[{specific_workflow}] pytorch_state_dict not found in model RDF\n"
        return preproc_info, True, reason_message

    return preproc_info, False, ""


def check_model_restrictions(cfg: CN, bmz_config: Dict, workflow_specs: Dict) -> List[str]:
    """
    Checks model restrictions to be applied into the current configuration.

    Parameters
    ----------
    cfg : YACS configuration
        Running configuration.

    bmz_config : dict
        BMZ configuration where among other thins the RDF of the model is stored.

    workflow_specs : dict
        Specifications of the workflow. Only expected "workflow_type" key.

    Returns
    -------
    option_list: list of str
        List of variables and values to change in current configuration. These changes
        are imposed by the selected model.
    """
    specific_workflow = workflow_specs["workflow_type"]

    # First let's make sure we have a valid model
    if isinstance(bmz_config["original_bmz_config"], InvalidDescr):
        raise ValueError(f"Failed to load '{cfg.MODEL.BMZ.SOURCE_MODEL_ID}' model")

    # Version of the model
    model_version = Version(bmz_config["original_bmz_config"].format_version)
    opts = {}

    # 1) Change PATCH_SIZE with the one stored in the RDF
    inputs = get_test_inputs(bmz_config["original_bmz_config"])
    input_image_shape = None
    if "input0" in inputs.members:
        input_image_shape = inputs.members["input0"]._data.shape  # type: ignore
    elif "raw" in inputs.members:
        input_image_shape = inputs.members["raw"]._data.shape  # type: ignore
    else:  # ambitious-sloth case
        input_image_shape = inputs.members[list(inputs.members.keys())[0]]._data.shape
    if input_image_shape is None:
        raise ValueError(f"Couldn't load input info from BMZ model's RDF: {inputs}")
    if cfg.DATA.PATCH_SIZE != input_image_shape[2:] + (input_image_shape[1],):
        opts["DATA.PATCH_SIZE"] = input_image_shape[2:] + (input_image_shape[1],)

    # Capture model kwargs
    if hasattr(bmz_config["original_bmz_config"].weights.pytorch_state_dict, "kwargs"):
        model_kwargs = bmz_config["original_bmz_config"].weights.pytorch_state_dict.kwargs
    elif hasattr(bmz_config["original_bmz_config"].weights.pytorch_state_dict, "architecture") and hasattr(
        bmz_config["original_bmz_config"].weights.pytorch_state_dict.architecture, "kwargs"
    ):
        model_kwargs = bmz_config["original_bmz_config"].weights.pytorch_state_dict.architecture.kwargs
    else:
        raise ValueError(f"Couldn't extract kwargs from model description.")

    # 2) Workflow specific restrictions
    # Classes in semantic segmentation
    if specific_workflow in ["SEMANTIC_SEG"]:
        # Check number of classes
        classes = -1
        if "n_classes" in model_kwargs:  # BiaPy
            classes = model_kwargs["n_classes"]
        elif "out_channels" in model_kwargs:
            classes = model_kwargs["out_channels"]
        elif "output_channels" in model_kwargs:
            classes = model_kwargs["output_channels"]
        elif "classes" in model_kwargs:
            classes = model_kwargs["classes"]
        if isinstance(classes, list):
            classes = classes[-1]

        if not isinstance(classes, int):
            raise ValueError(f"Classes not extracted correctly. Obtained {classes}")
        if classes == -1:
            raise ValueError("Classes not found for semantic segmentation dir.")

        opts["DATA.N_CLASSES"] = max(2, classes)

    elif specific_workflow in ["INSTANCE_SEG"]:
        # Assumed it's BC. This needs a more elaborated process. Still deciding this:
        # https://github.com/bioimage-io/spec-bioimage-io/issues/621

        # Defaults
        channels = 2
        channel_code = "BC"
        classes = 2

        if "out_channels" in model_kwargs:
            channels = model_kwargs["out_channels"]
        elif "output_channels" in model_kwargs:
            channels = model_kwargs["output_channels"]

        if "biapy" in bmz_config["original_bmz_config"].tags:
            # CartoCell models
            if (
                "cyst" in bmz_config["original_bmz_config"].tags
                and "3d" in bmz_config["original_bmz_config"].tags
                and "fluorescence" in bmz_config["original_bmz_config"].tags
            ):
                channel_code = "BCM"

            # Handle multihead
            assert isinstance(channels, list)
            if len(channels) == 2:
                classes = channels[-1]
            channels = channels[0]

        else:  # for other models set some defaults
            if isinstance(channels, list):
                channels = channels[-1]
            if channels == 1:
                channel_code = "C"
            elif channels == 2:
                channel_code = "BC"
            elif channels == 8:
                channel_code = "A"

        opts["PROBLEM.INSTANCE_SEG.DATA_CHANNELS"] = channel_code
        opts["PROBLEM.INSTANCE_SEG.DATA_CHANNEL_WEIGHTS"] = [
            1,
        ] * channels
        if classes != 2:
            opts["DATA.N_CLASSES"] = max(2, classes)
        if channel_code == "A":
            opts["LOSS.CLASS_REBALANCE"] = True

    # 3) Change preprocessing to the one stablished by BMZ by translate BMZ keywords into BiaPy's
    # 'zero_mean_unit_variance' and 'fixed_zero_mean_unit_variance' norms of BMZ can be translated to our 'custom' norm
    # providing mean and std
    print(f"[BMZ] Overriding preprocessing steps to the ones fixed in BMZ model: {bmz_config['preprocessing']}")
    key_to_find = "id" if model_version > Version("0.5.0") else "name"
    if key_to_find in bmz_config["preprocessing"]:
        if bmz_config["preprocessing"][key_to_find] in ["fixed_zero_mean_unit_variance", "zero_mean_unit_variance"]:
            if "kwargs" in bmz_config["preprocessing"] and "mean" in bmz_config["preprocessing"]["kwargs"]:
                mean = bmz_config["preprocessing"]["kwargs"]["mean"]
                std = bmz_config["preprocessing"]["kwargs"]["std"]
            elif "mean" in bmz_config["preprocessing"]:
                mean = bmz_config["preprocessing"]["mean"]
                std = bmz_config["preprocessing"]["std"]
            else:
                mean, std = -1.0, -1.0

            opts["DATA.NORMALIZATION.TYPE"] = "zero_mean_unit_variance"
            opts["DATA.NORMALIZATION.ZERO_MEAN_UNIT_VAR.MEAN_VAL"] = mean
            opts["DATA.NORMALIZATION.ZERO_MEAN_UNIT_VAR.STD_VAL"] = std

        # 'scale_linear' norm of BMZ is close to our 'div' norm (TODO: we need to control the "gain" arg)
        elif bmz_config["preprocessing"][key_to_find] == "scale_linear":
            opts["DATA.NORMALIZATION.TYPE"] = "div"

        # 'scale_range' norm of BMZ is as our 'scale_range' norm too
        elif bmz_config["preprocessing"][key_to_find] == "scale_range":
            opts["DATA.NORMALIZATION.TYPE"] = "scale_range"

            # Check if there is percentile clippign
            if (
                float(bmz_config["preprocessing"]["kwargs"]["min_percentile"]) != 0
                or float(bmz_config["preprocessing"]["kwargs"]["max_percentile"]) != 100
            ):
                opts["DATA.NORMALIZATION.PERC_CLIP.ENABLE"] = True
                opts["DATA.NORMALIZATION.PERC_CLIP.LOWER_PERC"] = float(
                    bmz_config["preprocessing"]["kwargs"]["min_percentile"]
                )
                opts["DATA.NORMALIZATION.PERC_CLIP.UPPER_PERC"] = float(
                    bmz_config["preprocessing"]["kwargs"]["max_percentile"]
                )

    option_list = []
    for key, val in opts.items():
        old_val = get_cfg_key_value(cfg, key)
        if old_val != val:
            print(f"[BMZ] Changed '{key}' from {old_val} to {val} as defined in the RDF")
        option_list.append(key)
        option_list.append(val)

    return option_list


def get_cfg_key_value(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def build_torchvision_model(cfg: CN, device: torch.device) -> Tuple[nn.Module, Callable]:
    # Find model in TorchVision
    if "quantized_" in cfg.MODEL.TORCHVISION_MODEL_NAME:
        mdl = importlib.import_module("torchvision.models.quantization", cfg.MODEL.TORCHVISION_MODEL_NAME)
        w_prefix = "_quantizedweights"
        tc_model_name = cfg.MODEL.TORCHVISION_MODEL_NAME.replace("quantized_", "")
        mdl_weigths = importlib.import_module("torchvision.models", cfg.MODEL.TORCHVISION_MODEL_NAME)
    else:
        w_prefix = "_weights"
        tc_model_name = cfg.MODEL.TORCHVISION_MODEL_NAME
        if cfg.PROBLEM.TYPE == "CLASSIFICATION":
            mdl = importlib.import_module("torchvision.models", cfg.MODEL.TORCHVISION_MODEL_NAME)
        elif cfg.PROBLEM.TYPE == "SEMANTIC_SEG":
            mdl = importlib.import_module("torchvision.models.segmentation", cfg.MODEL.TORCHVISION_MODEL_NAME)
        elif cfg.PROBLEM.TYPE in ["INSTANCE_SEG", "DETECTION"]:
            mdl = importlib.import_module("torchvision.models.detection", cfg.MODEL.TORCHVISION_MODEL_NAME)
        mdl_weigths = mdl

    # Import model and weights
    names = [x for x in mdl.__dict__ if not x.startswith("_")]
    for weight_name in names:
        if tc_model_name + w_prefix in weight_name.lower():
            break
    weight_name = weight_name.replace("Quantized", "")
    print(f"Pytorch model selected: {tc_model_name} (weights: {weight_name})")
    globals().update(
        {
            tc_model_name: getattr(mdl, tc_model_name),
            weight_name: getattr(mdl_weigths, weight_name),
        }
    )

    # Load model and weights
    model_torchvision_weights = eval(weight_name).DEFAULT
    model = eval(tc_model_name)(weights=model_torchvision_weights)

    # Create new head
    sample_size = None
    out_classes = cfg.DATA.N_CLASSES if cfg.DATA.N_CLASSES > 2 else 1
    if cfg.PROBLEM.TYPE == "CLASSIFICATION":
        if (
            cfg.DATA.N_CLASSES != 1000
        ):  # 1000 classes are the ones by default in ImageNet, which are the weights loaded by default
            print(
                f"WARNING: Model's head changed from 1000 to {out_classes} so a finetunning is required to have good results"
            )
            if cfg.MODEL.TORCHVISION_MODEL_NAME in ["squeezenet1_0", "squeezenet1_1"]:
                head = torch.nn.Conv2d(
                    model.classifier[1].in_channels,
                    out_classes,
                    kernel_size=1,
                    stride=1,
                )
                model.classifier[1] = head
            else:
                if hasattr(model, "fc"):
                    layer = "fc"
                elif hasattr(model, "classifier"):
                    layer = "classifier"
                else:
                    layer = "head"
                if isinstance(getattr(model, layer), list) or isinstance(
                    getattr(model, layer), torch.nn.modules.container.Sequential
                ):
                    head = torch.nn.Linear(getattr(model, layer)[-1].in_features, out_classes, bias=True)
                    getattr(model, layer)[-1] = head
                else:
                    head = torch.nn.Linear(getattr(model, layer).in_features, out_classes, bias=True)
                    setattr(model, layer, head)

            # Fix sample input shape as required by some models
            if cfg.MODEL.TORCHVISION_MODEL_NAME in ["maxvit_t"]:
                sample_size = (1, 3, 224, 224)
    elif cfg.PROBLEM.TYPE == "SEMANTIC_SEG":
        if cfg.DATA.N_CLASSES != 21:
            print(
                f"WARNING: Model's head changed from 21 to {out_classes} so a finetunning is required to have good results"
            )
        if tc_model_name == "lraspp_mobilenet_v3_large":
            head = torch.nn.Conv2d(model.classifier.low_classifier.in_channels, out_classes, kernel_size=1, stride=1)
            model.classifier.low_classifier = head
            head = torch.nn.Conv2d(model.classifier.high_classifier.in_channels, out_classes, kernel_size=1, stride=1)
            model.classifier.high_classifier = head
        else:
            head = torch.nn.Conv2d(model.classifier[-1].in_channels, out_classes, kernel_size=1, stride=1)
            model.classifier[-1] = head
            head = torch.nn.Conv2d(model.aux_classifier[-1].in_channels, out_classes, kernel_size=1, stride=1)
            model.aux_classifier[-1] = head

    elif cfg.PROBLEM.TYPE == "INSTANCE_SEG":
        # MaskRCNN
        if cfg.DATA.N_CLASSES != 91:  # 91 classes are the ones by default in MaskRCNN
            cls_score = torch.nn.Linear(in_features=1024, out_features=out_classes, bias=True)
            model.roi_heads.box_predictor.cls_score = cls_score
            mask_fcn_logits = torch.nn.Conv2d(
                model.roi_heads.mask_predictor.mask_fcn_logits.in_channels,
                out_classes,
                kernel_size=1,
                stride=1,
            )
            model.roi_heads.mask_predictor.mask_fcn_logits = mask_fcn_logits
            print(f"Model's head changed from 91 to {out_classes} so a finetunning is required")

    # Check the network created
    model.to(device)
    if sample_size is None:
        if cfg.PROBLEM.NDIM == "2D":
            sample_size = (
                1,
                cfg.DATA.PATCH_SIZE[2],
                cfg.DATA.PATCH_SIZE[0],
                cfg.DATA.PATCH_SIZE[1],
            )
        else:
            sample_size = (
                1,
                cfg.DATA.PATCH_SIZE[3],
                cfg.DATA.PATCH_SIZE[0],
                cfg.DATA.PATCH_SIZE[1],
                cfg.DATA.PATCH_SIZE[2],
            )

    summary(
        model,
        input_size=sample_size,
        col_names=("input_size", "output_size", "num_params"),
        depth=10,
        device=device.type,
    )

    return model, model_torchvision_weights.transforms()
