"""
This package (`biapy.models`) is responsible for building and managing deep learning models within the BiaPy framework.

It provides functionalities to:

1.  **Dynamically build models**: Select and instantiate various neural network architectures
    (e.g., U-Net, ResUNet, ViT, ConvNeXt variants, etc.) based on configuration settings.
2.  **Integrate with BioImage Model Zoo (BMZ)**: Facilitate the loading and compatibility
    checking of pre-trained models from the BioImage Model Zoo, enabling easy reuse
    of community-contributed models.
3.  **Extract model source code**: Collect the necessary source code for a given model
    and its dependencies, which is crucial for reproducibility and export functionalities.

The module handles different problem types (e.g., semantic segmentation, super-resolution,
classification) and adapts model configurations (e.g., 2D/3D, input/output channels,
normalization, dropout) accordingly.
"""

from importlib import import_module
import os
import re
import torch
import torch.nn as nn
from torchinfo import summary
from typing import Optional, Dict, Tuple, List, Callable
from packaging.version import Version
from functools import partial
from yacs.config import CfgNode as CN
import ast
from collections import deque, defaultdict
from importlib import import_module
import requests

from bioimageio.core.backends.pytorch_backend import load_torch_model
from bioimageio.spec.model.v0_4 import ModelDescr as ModelDescr_v0_4
from bioimageio.spec.model.v0_5 import ModelDescr as ModelDescr_v0_5


def build_model(
    cfg: CN, output_channels: int, device: torch.device
) -> Tuple[nn.Module, str, Dict, set, List[str], Dict, Tuple[int, ...]]:
    # model, model_file, model_name, args
    """
    Build selected model.

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
    mdl = import_module("biapy.models." + modelname)
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

        network_stride = [1, 1]
        if ndim == 3:
            network_stride = [1] + network_stride
        model = callable_model(**args)

    elif "hrnet" in modelname:
        args = dict(
            image_shape=cfg.DATA.PATCH_SIZE,
            normalization="sync_bn",
            output_channels=output_channels,
            contrast=cfg.LOSS.CONTRAST.ENABLE,
            contrast_proj_dim=cfg.LOSS.CONTRAST.PROJ_DIM,
            head_type=cfg.MODEL.HRNET.HEAD_TYPE,
        )

        if cfg.MODEL.HRNET.CUSTOM:
            args["cfg"] = cfg.MODEL.HRNET
        else:
            if modelname == "hrnet64":
                num_channels = 64
            elif modelname == "hrnet48":
                num_channels = 48
            elif modelname == "hrnet32":
                num_channels = 32
            elif modelname == "hrnet18":
                num_channels = 32
            args["cfg"] = {
                'Z_DOWN': cfg.MODEL.HRNET.Z_DOWN, 
                'STAGE2': {
                    'NUM_MODULES': 1, 
                    'NUM_BRANCHES': 2, 
                    'NUM_BLOCKS': [4, 4], 
                    'NUM_CHANNELS': [num_channels, num_channels*2], 
                    'BLOCK': 'BASIC'
                },
                'STAGE3': {
                    'NUM_MODULES': 4, 
                    'NUM_BRANCHES': 3, 
                    'NUM_BLOCKS': [4, 4, 4], 
                    'NUM_CHANNELS': [num_channels, num_channels*2, num_channels*3], 
                    'BLOCK': 'BASIC',
                },
                'STAGE4': {
                    'NUM_MODULES': 3, 
                    'NUM_BRANCHES': 4, 
                    'NUM_BLOCKS': [4, 4, 4, 4], 
                    'NUM_CHANNELS': [num_channels, num_channels*2, num_channels*3, num_channels*4], 
                    'BLOCK': 'BASIC'
                }
            } 

        callable_model = HighResolutionNet  # type: ignore
        model = callable_model(**args)

        network_stride = [4, 4]
        if ndim == 3:
            network_stride = [4 if args["cfg"].Z_DOWN else 1] + network_stride
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
                decoder_activation=cfg.MODEL.UNETR_DEC_ACTIVATION.lower(),
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

    summary(
        model,
        input_size=sample_size,
        col_names=("input_size", "output_size", "num_params"),
        depth=10,
        device=device.type,
    )

    # Queue for recursive dependency tracing
    dependency_queue = deque()
    dependency_queue.append(callable_model)

    collected_sources, all_import_lines, scanned_files = extract_model(dependency_queue, model_file)
    all_import_lines = merge_import_lines(all_import_lines)

    # Special handling for instance segmentation models with sigma outputs
    if cfg.PROBLEM.TYPE == "INSTANCE_SEG" and "E_sigma" in cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS:
        init_embedding_output(model, n_sigma=2 if cfg.PROBLEM.NDIM == "2D" else 3)

    return model, str(callable_model.__name__), collected_sources, all_import_lines, scanned_files, args, network_stride  # type: ignore


def init_embedding_output(model: nn.Module, n_sigma: int = 2):
    """
    Initialize the output layer of the model for embedding.

    Parameters
    ----------
    model : nn.Module
        The model whose output layer needs to be initialized.
    n_sigma : int
        Number of sigma channels to initialize.
    """
    try:
        with torch.no_grad():
            print("Initialize last layer with size: ", model.last_block.weight.size())
            print("*************************")
            model.last_block.weight[0:n_sigma, :,  :, :].fill_(0)
            model.last_block.bias[0:n_sigma].fill_(0)

            model.last_block.weight[n_sigma : n_sigma + n_sigma, :,  :, :].fill_(0)
            model.last_block.bias[n_sigma : n_sigma + n_sigma].fill_(1)
    except:
        raise ValueError("Could not initialize embedding output layer. Check the model structure.")

def extract_model(dependency_queue: deque, model_file: str) -> Tuple[Dict[str, str], set, List[str]]:
    """
    Extract the source code of the model and its dependencies, ensuring
    dependencies are ordered before the definition that uses them.

    Parameters
    ----------
    dependency_queue : deque
        Queue of model dependencies to be processed.

    model_file : str
        Path to the main model file.

    Returns
    -------
    collected_sources : dict
        Dictionary containing the source code of the collected model dependencies,
        ordered so that dependencies appear before the main model.

    all_import_lines : set
        Set of all external import lines found in the model and its dependencies.

    scanned_files : list
        List of all files that were scanned for dependencies.
    """
    visited_files = set()
    visited_names = set()
    collected_sources: Dict[str, str] = {}
    all_import_lines = set()
    scanned_files = []
    queue = [model_file]

    # {name: source_code} for all class/function definitions
    name_to_source: Dict[str, str] = {}

    # === Step 1: Scan all relevant files and build name → source map ===
    while queue:
        filepath = os.path.abspath(queue.pop())
        if filepath in visited_files:
            continue
        visited_files.add(filepath)
        scanned_files.append(filepath)

        with open(filepath, "r") as f:
            source_lines = f.readlines()
        source_text = "".join(source_lines)
        tree = ast.parse(source_text, filename=filepath)

        biapy_module_names = []

        for node in ast.walk(tree):
            # Import parsing
            if isinstance(node, ast.Import):
                for alias in node.names:
                    mod = alias.name
                    full = f"import {mod}" + (f" as {alias.asname}" if alias.asname else "")
                    if mod.startswith("biapy"):
                        biapy_module_names.append(mod)
                    else:
                        all_import_lines.add(full)
            elif isinstance(node, ast.ImportFrom):
                mod = node.module
                if not mod:
                    continue
                names = ", ".join(
                    f"{alias.name}" + (f" as {alias.asname}" if alias.asname else "") for alias in node.names
                )
                full = f"from {mod} import {names}"
                if mod.startswith("biapy"):
                    biapy_module_names.append(mod)
                else:
                    all_import_lines.add(full)

        # Extract all top-level classes and functions and map name → source
        for _node in tree.body:
            if isinstance(_node, (ast.FunctionDef, ast.ClassDef)):
                name = _node.name
                start_line = _node.lineno - 1
                # Try to find the end of the block
                end_line = start_line + 1
                indent = len(source_lines[start_line]) - len(source_lines[start_line].lstrip())

                while end_line < len(source_lines):
                    line_indent = len(source_lines[end_line]) - len(source_lines[end_line].lstrip())
                    if source_lines[end_line].strip() and line_indent <= indent:
                        break
                    end_line += 1

                name_to_source[name] = "".join(source_lines[start_line:end_line])

        # Follow BiaPy module imports (if file-based)
        visited = set()
        for name in biapy_module_names:
            if name in visited:
                continue
            visited.add(name)
            try:
                from importlib.util import find_spec

                spec = find_spec(name)
                if spec and spec.origin and os.path.isfile(spec.origin):
                    queue.append(spec.origin)
            except Exception as e:
                print(f"Warning: Failed to resolve {name}: {e}")

    # === Step 2: Traverse dependency tree and store definitions in discovery order ===
    # We use a list to store the (name, source) tuples in the order they are found (BFS).
    # This order is: [Dependent_A, Dependency_B, Dependency_C, ...]
    definition_list: List[Tuple[str, str]] = []

    class NameVisitor(ast.NodeVisitor):
        def __init__(self):
            self.names = []

        def visit_Name(self, node):
            self.names.append(node.id)
            self.generic_visit(node)

        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name):
                self.names.append(node.value.id)
            self.generic_visit(node)

    while dependency_queue:
        obj = dependency_queue.pop()
        name = obj.__name__
        if name in visited_names:
            continue
        visited_names.add(name)

        source = name_to_source.get(name)
        if not source:
            print(f"Warning: Source not found for {name}")
            continue

        # Add the definition to the temporary list
        definition_list.append((name, source))

        # Find dependencies
        visitor = NameVisitor()
        visitor.visit(ast.parse(source))

        for dep_name in visitor.names:
            if dep_name not in visited_names and dep_name in name_to_source:

                class FakeObject:
                    def __init__(self, __name__):
                        self.__name__ = __name__

                dependency_queue.append(FakeObject(dep_name))

    # === Step 3: Populate collected_sources in reverse order (Dependencies First) ===
    # By reversing the list, the deepest dependencies (discovered last) are placed 
    # at the start of the dictionary, ensuring they are defined before being used.
    for name, source in reversed(definition_list):
        collected_sources[name] = source

    return collected_sources, sorted(all_import_lines), scanned_files


def merge_import_lines(import_lines: List[str]) -> List[str]:
    """
    Merge import lines by grouping them by module and sorting names within each module.

    Parameters
    ----------
    import_lines : list of str
        List of import lines to be merged.

    Returns
    -------
    merged : list of str
        Merged import lines, sorted and grouped by module.
    """
    grouped = defaultdict(set)
    standalone_imports = set()

    for line in import_lines:
        line = line.strip()
        if line.startswith("import "):
            # Regular import, keep it as-is
            standalone_imports.add(line)
        elif line.startswith("from "):
            try:
                parts = line.split(" import ")
                mod = parts[0][5:].strip()  # remove "from "
                names = parts[1].split(",")
                for name in names:
                    grouped[mod].add(name.strip())
            except Exception as e:
                print(f"Warning: could not parse import line '{line}': {e}")
        else:
            standalone_imports.add(line)

    merged = []

    for mod, names in grouped.items():
        sorted_names = sorted(names)
        merged.append(f"from {mod} import {', '.join(sorted_names)}")

    merged.extend(sorted(standalone_imports))
    return sorted(merged)


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

def find_bmz_models(
    model_ID: Optional[str] = None,
    url: str = "https://hypha.aicell.io/bioimage-io/artifacts/bioimage.io/children?limit=1000000",
    timeout: int = 30,
):
    """
    Query the BioImage.IO Hypha API for *models* and return those whose
    nickname/id/rdf_source contains `model_ID` (case-insensitive).

    Returns list of dicts with: id, alias, nickname, rdf_source, version,
    format_version, artifact_path (id used as path), and a few handy urls.

    Parameters
    ----------
    model_ID : str
        Model identifier. It can be either its ``DOI`` or ``nickname``. Leave it as None
        to get all available models.

    url : str
        URL to the BioImage.IO Hypha API endpoint to query for models.

    timeout : int
        Timeout for the HTTP request in seconds.

    Returns
    -------
    out : list of dict
        List of dictionaries containing model information. Each dictionary has the following
        keys: `id`, `alias`, `nickname`, `rdf_source`, `version`, `format_version`,
        `artifact_path`, `urls` (which contains `covers` and `documentation` URLs), and `raw`
        (the original item from the API response).
    """
    q = str(model_ID).lower() if model_ID else None

    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    items = r.json()

    if isinstance(items, dict) and "children" in items:
        items = items["children"]

    out = []
    for it in items or []:
        if (it or {}).get("type") != "model":
            continue

        # Pull common fields defensively from the manifest config
        cfg = ((it.get("manifest") or {}).get("config")) or {}
        b = cfg.get("bioimageio") or {}
        nickname = b.get("nickname") or it.get("alias")
        rdf_source = b.get("rdf_source") or b.get("source")  # some deployments use 'source'
        version = b.get("version") or cfg.get("version") or it.get("version")
        format_version = b.get("format_version") or cfg.get("format_version")

        # Build haystack for matching (old behavior)
        hay = [
            nickname or "",
            it.get("id") or "",
            rdf_source or "",
        ]
        if q and not any(q in h.lower() for h in hay):
            continue

        out.append(
            {
                "artifact_path": it.get("id"),  # usable as path for other calls
                "id": it.get("id"),
                "alias": it.get("alias"),
                "nickname": nickname,
                "rdf_source": rdf_source,
                "version": version,
                "format_version": format_version,
                "urls": {
                    "covers": (b.get("thumbnails") or {}),
                    "documentation": b.get("documentation"),
                },
                "raw": it,  # keep the original item in case you need more fields
            }
        )

    return out


def check_bmz_args(
    model_ID: str,
    cfg: CN,
) -> Tuple[List, Dict]:
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
    matches = find_bmz_models(model_ID)

    if len(matches) == 0:
        raise ValueError(f"No model found with the provided DOI/name: {model_ID}")
    if len(matches) > 1:
        raise ValueError(f"More than one model found with the provided DOI/name ({model_ID}). Contact BiaPy team.")
    model_dict = matches[0]

    workflow_specs = {}
    workflow_specs["workflow_type"] = cfg.PROBLEM.TYPE
    workflow_specs["ndim"] = cfg.PROBLEM.NDIM
    workflow_specs["nclasses"] = cfg.DATA.N_CLASSES

    preproc_info, error, error_message, opts = check_bmz_model_compatibility(model_dict, workflow_specs=workflow_specs)

    if error:
        raise ValueError(f"Model {model_ID} can not be used in BiaPy. Message:\n{error_message}\n")

    return preproc_info, opts


def check_bmz_model_compatibility(
    model_rdf: Dict,
    workflow_specs: Optional[Dict] = None,
) -> Tuple[List, bool, str, Dict]:
    """
    Check one model compatibility with BiaPy by looking at its RDF file provided by BMZ. This function is the one used in BMZ's continuous integration with BiaPy.

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

    # --------- helpers ---------
    def g(d, *ks, default=None):
        cur = d
        for k in ks:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                return default
        return cur

    m = g(model_rdf, "raw", "manifest", default=model_rdf) or model_rdf

    specific_workflow = "all" if workflow_specs is None else workflow_specs["workflow_type"]
    specific_dims = "all" if workflow_specs is None else workflow_specs["ndim"]
    ref_classes = "all" if workflow_specs is None else workflow_specs["nclasses"]

    preproc_info: List = []
    opts = {}

    # --------- Accept only PyTorch state dict models with a single input ---------
    weights = g(m, "weights", "pytorch_state_dict")
    inputs = g(m, "inputs") or []

    if not (isinstance(weights, dict) and weights):
        reason_message = f"[{specific_workflow}] pytorch_state_dict not found in model RDF\n"
        return preproc_info, True, reason_message, opts
    if not (isinstance(inputs, list) and len(inputs) == 1):
        reason_message = f"[{specific_workflow}] Model needs to have a single input.\n"
        return preproc_info, True, reason_message, opts

    # Model format version (defaults to 0.5 for your legacy logic)
    model_version = Version("0.5")
    fmt = g(m, "format_version")
    if isinstance(fmt, str):
        try:
            model_version = Version(fmt)
        except Exception:
            pass

    # --------- Extract model kwargs ---------
    model_kwargs = None
    if "kwargs" in weights:
        model_kwargs = weights["kwargs"]
    elif "architecture" in weights and isinstance(weights["architecture"], dict):
        model_kwargs = weights["architecture"].get("kwargs", None)
    if model_kwargs is None:
        return preproc_info, True, f"[{specific_workflow}] Couldn't extract kwargs from model description.\n", opts

    # --------- Problem type via tags ---------
    tags = g(m, "tags", default=[]) or []
    
    if (specific_workflow in ["all", "SEMANTIC_SEG"]) and (
        "semantic-segmentation" in tags or ("segmentation" in tags and "instance-segmentation" not in tags)
    ):
        # classes
        classes = -1
        for k in ("n_classes", "out_channels", "output_channels", "classes"):
            if k in model_kwargs:
                classes = model_kwargs[k]
                break
        if isinstance(classes, list):
            classes = classes[-1]

        if not isinstance(classes, int):
            reason_message = (
                f"[{specific_workflow}] 'DATA.N_CLASSES' not extracted. Obtained {classes}. Please check it!\n"
            )
            return preproc_info, True, reason_message, opts
        
        if (
            classes == -1
            and "architecture" in weights
            and isinstance(weights["architecture"], dict)
            and ("callable" in weights["architecture"] or "source" in weights["architecture"])
        ):
            # Check if the model is one of the known architectures and assume it returns 1 class (as is the default in BiaPy)
            for arch in [weights["architecture"].get("callable", None), weights["architecture"].get("source", None)]:
                if arch is not None:
                    arch = str(arch).lower().replace(".py", "")
                    if arch in [
                        "unet",
                        "resunet",
                        "resunet++",
                        "seunet",
                        "attention_unet",
                        "resunet_se",
                        "unetr",
                        "multiresunet",
                        "unext_v1",
                        "unext_v2",
                        "hrnet",
                    ]:
                        classes = 1
                if classes != -1:
                    print(f"[BMZ] Detected BiaPy model ({arch}) so assuming 1 as the class output, which is the default in BiaPy")
                    break 

        if isinstance(classes, int) and classes != -1:
            if ref_classes != "all":
                if classes > 2 and ref_classes != classes:
                    reason_message = f"[{specific_workflow}] 'DATA.N_CLASSES' does not match network's output classes. Please check it!\n"
                    return preproc_info, True, reason_message, opts
        else:
            reason_message = f"[{specific_workflow}] Couldn't find the classes this model is returning so please be aware to match it\n"
            return preproc_info, True, reason_message, opts

        opts["DATA.N_CLASSES"] = max(2, classes)

    elif specific_workflow in ["all", "INSTANCE_SEG"] and "instance-segmentation" in tags:
        # Assumed it's F + C. This needs a more elaborated process. Still deciding this:
        # https://github.com/bioimage-io/spec-bioimage-io/issues/621

        # Defaults
        channels = 2
        channel_code = ["F", "C"]
        classes = 2

        if "out_channels" in model_kwargs:
            channels = model_kwargs["out_channels"]
        elif "output_channels" in model_kwargs:
            channels = model_kwargs["output_channels"]

        if "biapy" in tags:
            # CartoCell models
            if (
                "cyst" in tags
                and "3d" in tags
                and "fluorescence" in tags
            ):
                channel_code = ["F", "C", "M"]

            # Handle multihead
            assert isinstance(channels, list)
            if len(channels) == 2:
                classes = channels[-1]
            channels = channels[0]

        else:  # for other models set some defaults
            if isinstance(channels, list):
                channels = channels[-1]
            if channels == 1:
                channel_code = ["C"]
            elif channels == 2:
                channel_code = ["F", "C"]
            elif channels == 8:
                channel_code = ["A"] # wild-whale

        opts["PROBLEM.INSTANCE_SEG.DATA_CHANNELS"] = channel_code
        opts["PROBLEM.INSTANCE_SEG.DATA_CHANNEL_WEIGHTS"] = [
            1,
        ] * channels
        if classes != 2:
            opts["DATA.N_CLASSES"] = max(2, classes)
        if channel_code == "A":
            opts["LOSS.CLASS_REBALANCE"] = "auto"

    elif specific_workflow in ["all", "DETECTION"] and "detection" in tags:
        pass
    elif specific_workflow in ["all", "DENOISING"] and "denoising" in tags:
        pass
    elif specific_workflow in ["all", "SUPER_RESOLUTION"] and ("super-resolution" in tags or "superresolution" in tags):
        pass
    elif specific_workflow in ["all", "SELF_SUPERVISED"] and "self-supervision" in tags:
        pass
    elif specific_workflow in ["all", "CLASSIFICATION"] and "classification" in tags:
        pass
    elif specific_workflow in ["all", "IMAGE_TO_IMAGE"] and any(
        t in tags for t in ("pix2pix", "image-reconstruction", "image-to-image", "image-restoration")
    ):
        pass
    else:
        reason_message = f"[{specific_workflow}] no workflow tag recognized in {tags}.\n"
        return preproc_info, True, reason_message, opts

    # --------- Axes checks ---------
    axes_order = g(inputs[0], "axes")
    input_image_shape = []

    # Model version > 5   
    if isinstance(axes_order, str):
        input_image_shape = inputs[0].get("shape", {}).get("min", [])
    elif isinstance(axes_order, list):
        _axes_order = ""
        for axis in axes_order:
            if "type" in axis:
                if axis["type"] == "batch":
                    _axes_order += "b"
                    input_image_shape += [1]
                elif axis["type"] == "channel":
                    _axes_order += "c"
                    input_image_shape += [1]
                elif "id" in axis:
                    _axes_order += axis["id"]
                    input_image_shape += [axis["size"]]
            elif "id" in axis:
                if axis["id"] == "channel":
                    _axes_order += "c" 
                    input_image_shape += [1]
                else:
                    if isinstance(axis.get("size"), int):
                        input_image_shape += [axis["size"]]
                    elif isinstance(axis.get("size"), dict) and "min" in axis["size"]:
                        input_image_shape += [axis["size"]["min"]]
                    _axes_order += axis["id"]
        axes_order = _axes_order
    
    try:
        opts["DATA.PATCH_SIZE"] = tuple(input_image_shape[2:] + [input_image_shape[1]]) # (z) y x c
    except Exception:
        reason_message = f"[{specific_workflow}] couldn't extract input image shape from model RDF: {input_image_shape}\n"
        return preproc_info, True, reason_message, opts

    if specific_dims == "2D":
        if axes_order != "bcyx":
            reason_message = f"[{specific_workflow}] In a 2D problem the axes need to be 'bcyx', found {axes_order}\n"
            return preproc_info, True, reason_message, opts
        elif "2d" not in tags and "3d" in tags:
            reason_message = f"[{specific_workflow}] Selected model seems to not be 2D\n"
            return preproc_info, True, reason_message, opts
    elif specific_dims == "3D":
        if axes_order != "bczyx":
            reason_message = f"[{specific_workflow}] In a 3D problem the axes need to be 'bczyx', found {axes_order}\n"
            return preproc_info, True, reason_message, opts
        elif "3d" not in tags and "2d" in tags:
            reason_message = f"[{specific_workflow}] Selected model seems to not be 3D\n"
            return preproc_info, True, reason_message, opts
    else:  # "all"
        if axes_order not in ["bcyx", "bczyx"]:
            reason_message = (
                f"[{specific_workflow}] Accepting models only with ['bcyx', 'bczyx'] axis order, found {axes_order}\n"
            )
            return preproc_info, True, reason_message, opts

    # --------- Preprocessing ---------
    if "preprocessing" in (inputs[0] or {}):
        preproc_info = inputs[0]["preprocessing"]
        key_to_find = "id" if model_version > Version("0.5.0") else "name"
        if isinstance(preproc_info, list):
            # remove ensure_dtype->float casts (BiaPy does it anyway)
            new_preproc_info = []
            for preproc in preproc_info:
                if key_to_find in preproc and not (
                    preproc[key_to_find] == "ensure_dtype"
                    and "kwargs" in preproc
                    and "dtype" in preproc["kwargs"]
                    and "float" in str(preproc["kwargs"]["dtype"])
                ):
                    new_preproc_info.append(preproc)
            preproc_info = new_preproc_info.copy()

            if len(preproc_info) > 1:
                reason_message = (
                    f"[{specific_workflow}] More than one preprocessing from BMZ not implemented yet {axes_order}\n"
                )
                return preproc_info, True, reason_message, opts
            elif len(preproc_info) == 1:
                preproc_info = preproc_info[0]
                if key_to_find in preproc_info:
                    proc_id = preproc_info[key_to_find]
                    if proc_id not in [
                        "zero_mean_unit_variance",
                        "fixed_zero_mean_unit_variance",
                        "scale_range",
                        "scale_linear",
                    ]:
                        reason_message = (
                            f"[{specific_workflow}] Not recognized preprocessing found: {proc_id}\n"
                        )
                        return preproc_info, True, reason_message, opts
                    else:
                        # zero_mean_unit_variance / fixed_zero_mean_unit_variance -> zero_mean_unit_variance(mean,std)
                        if proc_id in ["fixed_zero_mean_unit_variance", "zero_mean_unit_variance"]:
                            if "kwargs" in preproc_info and "mean" in preproc_info["kwargs"]:
                                mean = preproc_info["kwargs"]["mean"]
                                std = preproc_info["kwargs"]["std"]
                            elif "mean" in preproc_info:
                                mean = preproc_info["mean"]
                                std = preproc_info["std"]
                            else:
                                mean, std = -1.0, -1.0

                            opts["DATA.NORMALIZATION.TYPE"] = "zero_mean_unit_variance"
                            opts["DATA.NORMALIZATION.ZERO_MEAN_UNIT_VAR.MEAN_VAL"] = mean
                            opts["DATA.NORMALIZATION.ZERO_MEAN_UNIT_VAR.STD_VAL"] = std

                        # scale_linear ~ div (gain not handled, same as original)
                        elif proc_id == "scale_linear":
                            opts["DATA.NORMALIZATION.TYPE"] = "div"

                        # scale_range -> scale_range (+ optional PERC_CLIP)
                        elif proc_id == "scale_range":
                            opts["DATA.NORMALIZATION.TYPE"] = "scale_range"

                            # Check if there is percentile clippign
                            if (
                                float(preproc_info["kwargs"]["min_percentile"]) != 0
                                or float(preproc_info["kwargs"]["max_percentile"]) != 100
                            ):
                                opts["DATA.NORMALIZATION.PERC_CLIP.ENABLE"] = True
                                opts["DATA.NORMALIZATION.PERC_CLIP.LOWER_PERC"] = float(
                                    preproc_info["kwargs"]["min_percentile"]
                                )
                                opts["DATA.NORMALIZATION.PERC_CLIP.UPPER_PERC"] = float(
                                    preproc_info["kwargs"]["max_percentile"]
                                )
                else:
                    reason_message = (
                        f"[{specific_workflow}] Not recognized preprocessing structure found: {preproc_info}\n"
                    )
                    return preproc_info, True, reason_message, opts

    # --------- Post-processing in kwargs (unsupported) ---------
    if "postprocessing" in model_kwargs and model_kwargs["postprocessing"] is not None:
        reason_message = (
            f"[{specific_workflow}] Currently no postprocessing is supported. Found: {model_kwargs['postprocessing']}\n"
        )
        return preproc_info, True, reason_message, opts

    # All checks passed
    return preproc_info, False, "", opts


def build_torchvision_model(cfg: CN, device: torch.device) -> Tuple[nn.Module, Callable]:
    """
    Build and adapt a model from the `torchvision.models` library based on the configuration.

    This function dynamically loads a pre-trained model from `torchvision.models`
    (e.g., ResNet, DeepLabV3, MaskRCNN, etc.) as specified in the configuration.
    It then adapts the final output layer(s) of the model to match the number of
    classes or output channels required by the specific problem type (e.g.,
    classification, semantic segmentation, instance segmentation).

    Parameters
    ----------
    cfg : YACS CN object
        The configuration object. Key parameters used are:

        - `cfg.MODEL.TORCHVISION_MODEL_NAME`: Name of the torchvision model to load
          (e.g., "resnet50", "deeplabv3_resnet101", "maskrcnn_resnet50_fpn", "quantized_resnet50").
        - `cfg.PROBLEM.TYPE`: Type of problem (e.g., "CLASSIFICATION", "SEMANTIC_SEG",
          "INSTANCE_SEG", "DETECTION") to determine model adaptation logic.
        - `cfg.DATA.N_CLASSES`: Number of output classes required for the problem.
        - `cfg.DATA.PATCH_SIZE`: Input patch size, used for generating the model summary.
        - `cfg.PROBLEM.NDIM`: Number of input dimensions ("2D" or "3D").

    device : torch.device
        The PyTorch device (e.g., "cpu", "cuda", "mps") on which the model
        will be loaded and run.

    Returns
    -------
    model : nn.Module
        The instantiated and adapted PyTorch model from torchvision.
    transforms : Callable
        A callable representing the default preprocessing transforms associated
        with the loaded torchvision model's weights. This should be applied to
        input images before feeding them to the model.

    Notes
    -----
    - Models are loaded with their `DEFAULT` pre-trained weights from torchvision.
    - The final layer adaptation logic is specific to common torchvision model
      structures for classification, semantic segmentation, and instance segmentation.
    - For classification, the final linear layer is replaced. A warning is printed
      if the number of classes differs from ImageNet's default (1000).
    - For semantic segmentation, the final convolutional layer(s) of the classifier
      and auxiliary classifier (if present) are replaced. A warning is printed
      if the number of classes differs from Pascal VOC's default (21).
    - For instance segmentation (MaskRCNN), the box predictor's classification
      score head and the mask predictor's final convolutional layer are replaced.
      A warning is printed if the number of classes differs from COCO's default (91).
    - Special handling is included for `squeezenet` and `lraspp_mobilenet_v3_large`
      due to their unique head structures.
    - For `maxvit_t` in classification, a fixed sample input size of (1, 3, 224, 224)
      is used for the model summary.
    - This function assumes the necessary `torchvision` models and their default
      weights are installed and accessible.

    """
    # Find model in TorchVision
    if "quantized_" in cfg.MODEL.TORCHVISION_MODEL_NAME:
        mdl = import_module("torchvision.models.quantization", cfg.MODEL.TORCHVISION_MODEL_NAME)
        w_prefix = "_quantizedweights"
        tc_model_name = cfg.MODEL.TORCHVISION_MODEL_NAME.replace("quantized_", "")
        mdl_weigths = import_module("torchvision.models", cfg.MODEL.TORCHVISION_MODEL_NAME)
    else:
        w_prefix = "_weights"
        tc_model_name = cfg.MODEL.TORCHVISION_MODEL_NAME
        if cfg.PROBLEM.TYPE == "CLASSIFICATION":
            mdl = import_module("torchvision.models", cfg.MODEL.TORCHVISION_MODEL_NAME)
        elif cfg.PROBLEM.TYPE == "SEMANTIC_SEG":
            mdl = import_module("torchvision.models.segmentation", cfg.MODEL.TORCHVISION_MODEL_NAME)
        elif cfg.PROBLEM.TYPE in ["INSTANCE_SEG", "DETECTION"]:
            mdl = import_module("torchvision.models.detection", cfg.MODEL.TORCHVISION_MODEL_NAME)
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
