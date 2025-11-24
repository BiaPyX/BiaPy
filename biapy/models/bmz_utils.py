"""
This module provides utility functions for managing and preparing BioImage Model Zoo (BMZ) models within the BiaPy framework.

It includes functionalities for:

- Retrieving and parsing BMZ model information (`get_bmz_model_info`).
- Creating necessary environment files for model deployment (`create_environment_file_for_model`).
- Generating visual covers for models based on input/output examples (`create_model_cover`).
- Generating comprehensive documentation files for exported models (`create_model_doc`).
"""
import os
import re
import biapy
import yaml
import numpy as np
from typing import Any, Tuple
from packaging.version import Version
from skimage.transform import resize
from numpy.typing import NDArray

from bioimageio.spec.model.v0_4 import ModelDescr as ModelDescr_v0_4
from bioimageio.spec.model.v0_5 import ModelDescr as ModelDescr_v0_5
from bioimageio.spec._internal.types import ImportantFileSource
from bioimageio.spec._internal.io_basics import Sha256
from bioimageio.spec.model.v0_5 import (
    ArchitectureFromFileDescr,
    ArchitectureFromLibraryDescr,
)
from bioimageio.spec.utils import download

from biapy.data.data_manipulation import imwrite, reduce_dtype, extract_patch_within_image
from biapy.data.dataset import PatchCoords
from biapy.data.data_3D_manipulation import extract_patch_from_efficient_file

def get_bmz_model_info(
    model: ModelDescr_v0_4 | ModelDescr_v0_5, spec_version: Version = Version("0.4.0")
) -> Tuple[ImportantFileSource, Sha256 | None, ArchitectureFromFileDescr | ArchitectureFromLibraryDescr]:
    """
    Gather model info depending on its spec version. Currently supports ``v0_4`` and ``v0_5`` spec model.

    Parameters
    ----------
    model : ModelDescr
        BMZ model RDF that contains all the information of the model.

    spec_version : str
        Version of model's specs.

    Returns
    -------
    model_instance : Torch model
        Torch model.
    """
    assert (
        model.weights.pytorch_state_dict
    ), "Seems that the original BMZ model has no pytorch_state_dict object. Aborting"

    if spec_version > Version("0.5.0"):
        arch = model.weights.pytorch_state_dict.architecture
        if isinstance(arch, ArchitectureFromFileDescr):
            arch_file_path = download(arch.source, sha256=arch.sha256).path
            arch_file_sha256 = arch.sha256
            arch_name = arch.callable
            arch_kwargs = arch.kwargs

            pytorch_architecture = ArchitectureFromFileDescr(
                source=arch_file_path,  # type: ignore
                sha256=arch_file_sha256,
                callable=arch_name,
                kwargs=arch_kwargs,
            )
        else:
            # For a model architecture that is published in a Python package
            # Make sure to include the Python library referenced in `import_from` in the weights entry's `dependencies`
            pytorch_architecture = ArchitectureFromLibraryDescr(
                callable=arch.callable,
                kwargs=arch.kwargs,
                import_from=arch.import_from,
            )
        state_dict_source = model.weights.pytorch_state_dict.source
        state_dict_sha256 = model.weights.pytorch_state_dict.sha256
    else:  # v0_4
        arch_file_sha256 = model.weights.pytorch_state_dict.architecture_sha256

        arch_file_path = download(
            model.weights.pytorch_state_dict.architecture.source_file, sha256=arch_file_sha256
        ).path
        arch_name = model.weights.pytorch_state_dict.architecture.callable_name
        pytorch_architecture = ArchitectureFromFileDescr(
            source=arch_file_path,  # type: ignore
            sha256=arch_file_sha256,
            callable=arch_name,
            kwargs=model.weights.pytorch_state_dict.kwargs,
        )
        state_dict_source = model.weights.pytorch_state_dict.source
        state_dict_sha256 = model.weights.pytorch_state_dict.sha256

    return state_dict_source, state_dict_sha256, pytorch_architecture


def create_environment_file_for_model(building_dir):
    """
    Create a conda environment file (environment.yaml) with the necessary dependencies to build a model with BiaPy.

    Parameters
    ----------
    building_dir : str
        Directory to save the environment.yaml file.

    Returns
    -------
    env_file : str
        Path to the environment.yaml file created.
    """
    env = dict(
        name="biapy",
        dependencies=[
            "python>=3.10",
            "pip",
            {
                "pip": [
                    "timm==1.0.14",
                ]
            },
        ],
    )

    os.makedirs(building_dir, exist_ok=True)
    env_file = os.path.join(building_dir, "environment.yaml")
    with open(env_file, "w", encoding="utf8") as outfile:
        yaml.dump(env, outfile, default_flow_style=False)

    return env_file

def extract_BMZ_sample_and_cover(
    img: Any, 
    img_gt: Any, 
    patch_size=[256,256,1], 
    is_3d=False, 
    input_axis_order: str = "ZYXC"
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Extract a sample patch from the input image and its corresponding img_gt.

    Parameters
    ----------
    img : Any
        The input image from which to extract the patch.

    img_gt : Any
        The img_gt corresponding to the input image.

    patch_size : list, optional
        The size of the patch to extract (default is [256,256,1]).

    is_3d : bool, optional
        Whether the input is 3D (default is False).

    input_axis_order : str, optional
        The axis order for the input data (default is "ZYXC").

    Returns
    -------
    rimg : NDArray
        The extracted image patch. Shape will be (H, W, C) for 2D or (D, H, W, C) for 3D.

    cover_raw : NDArray
        The raw image cover (2D slice). Shape will be (H, W, C).

    cover_gt : NDArray
        The ground truth img_gt cover (2D slice). Shape will be (H, W, C).
    """
    cover_raw, cover_gt = None, img_gt
    mask_available = img_gt is not None and isinstance(img_gt, np.ndarray)
    dims = 2 if not is_3d else 3
    ref_img = img_gt if mask_available else img
    if isinstance(ref_img, np.ndarray):    
        if dims == 2:
            H, W, C = ref_img.shape
            ph, pw = patch_size[0], patch_size[1]
            coords = np.argwhere(ref_img > 0)
            if len(coords) == 0:
                # Entire ref_img empty -> fall back to image center
                y_center, x_center = H // 2, W // 2
            else:
                ymin, xmin, _ = coords.min(axis=0)
                ymax, xmax, _ = coords.max(axis=0)
                y_center = (ymin + ymax) // 2
                x_center = (xmin + xmax) // 2

            y_start = max(0, min(H - ph, y_center - ph // 2))
            x_start = max(0, min(W - pw, x_center - pw // 2))
        elif dims == 3:
            D, H, W, C = ref_img.shape
            pd, ph, pw = patch_size[0], patch_size[1], patch_size[2]

            slice_counts = [np.count_nonzero(ref_img[z]) for z in range(D)]
            best_slice = np.argmax(slice_counts)
            m2d = ref_img[best_slice]
            if slice_counts[best_slice] == 0:
                # Entire ref_img empty -> fall back to volume center
                y_center, x_center = H // 2, W // 2
            else:
                coords = np.argwhere(m2d > 0)
                ymin, xmin, _ = coords.min(axis=0)
                ymax, xmax, _ = coords.max(axis=0)
                y_center = (ymin + ymax) // 2
                x_center = (xmin + xmax) // 2
                
            # Depth placement centered around best_slice, bounded
            half_pd = pd // 2
            z_end = min(D, best_slice + half_pd)
            z_center = z_end - half_pd
            z_start = z_center - half_pd  # works for even/odd since half_pd is floor
            z_start = max(0, min(D - pd, z_start))

            # In-plane placement, bounded
            y_start = max(0, min(H - ph, y_center - ph // 2))
            x_start = max(0, min(W - pw, x_center - pw // 2))

        # Ensure a patch size
        patch = PatchCoords(
            z_start=z_start if is_3d else None,
            z_end=z_start+patch_size[0] if is_3d else None,
            y_start=y_start,
            y_end=y_start+patch_size[-3],
            x_start=x_start,
            x_end=x_start+patch_size[-2],
        )
        rimg = extract_patch_within_image(
            img, patch, is_3d=True if is_3d else False
        )
        if mask_available:
            rimg_gt = extract_patch_within_image(
                img_gt, patch, is_3d=True if is_3d else False
            )
        if is_3d:
            cover_raw = rimg[best_slice-z_start].copy()
            if mask_available:
                cover_gt = rimg_gt[best_slice-z_start].copy()
        else:
            cover_raw = rimg.copy()
            if mask_available:
                cover_gt = rimg_gt.copy()
    else:
        # TODO: take a patch ensuring that it contains img_gt
        patch = PatchCoords(
            z_start=0 if is_3d else None,
            z_end=0+patch_size[0] if is_3d else None,
            y_start=0,
            y_end=0+patch_size[-3],
            x_start=0,
            x_end=0+patch_size[-2],
        )
        rimg = extract_patch_from_efficient_file(
            img, patch, data_axes_order=input_axis_order,
        )
        if mask_available:
            rimg_gt = extract_patch_from_efficient_file(
                img_gt, patch, data_axes_order=input_axis_order,
            )   
        if is_3d:
            cover_raw = rimg[rimg.shape[0] // 2].copy()
            if mask_available:
                cover_gt = rimg_gt[rimg_gt.shape[0] // 2].copy()
        else:
            cover_raw = rimg.copy()
            if mask_available:
                cover_gt = rimg_gt.copy()

    rimg = rimg.astype(np.float32)
    if (dims == 2 and rimg.ndim == 2) or (dims == 3 and rimg.ndim == 3):
        rimg = np.expand_dims(rimg, -1)

    return rimg, cover_raw, cover_gt

def create_model_cover(img, img_gt, out_path, patch_size=256, is_3d=False, workflow="semantic-segmentation"):
    """
    Create a cover based on the files that will be read from ``file_pointers``.

    Parameters
    ----------
    img : NDArray
        Input image. E.g. ``(z, y, x, channels)`` for 3D or ``(y, x, channels)`` for 2D.

    img_gt : NDArray
        Ground truth image. E.g. ``(z, y, x, channels)`` for 3D or ``(y, x, channels)`` for 2D.

    out_path : str
        Directory to save the cover.

    patch_size : int, optional
        Size of the image to create.

    is_3d : bool, optional
        Whether if the images to load are 3D or not.

    workflow : str, optional
        Workflow to create the cover to. Options are: [``"semantic-segmentation"``, ``"instance-segmentation"``,
        ``"detection"``, ``"denoising"``, ``"super-resolution"``, ``"self-supervised"``, ``"classification"``,
        ``"image-to-image"``]

    Returns
    -------
    cover_path : str
        Path to the cover.
    """
    assert workflow in [
        "semantic-segmentation",
        "instance-segmentation",
        "detection",
        "denoising",
        "super-resolution",
        "self-supervised",
        "classification",
        "image-to-image",
    ]
    # If 3D just take middle slice.
    if is_3d and img.ndim == 4:
        img = img[img.shape[0] // 2]
    if is_3d and isinstance(img_gt, np.ndarray) and img_gt.ndim == 4:
        img_gt = img_gt[img_gt.shape[0] // 2]

    # Convert to RGB
    if img.shape[-1] == 1:
        img = np.stack((img[..., 0],) * 3, axis=-1)
    elif img.shape[-1] == 2:
        img = np.concatenate((img, np.zeros(img.shape[:-1] + (1,), dtype=img.dtype)), axis=-1)
    elif img.shape[-1] > 3:
        img = img[..., :3]

    # Resize the images if neccesary
    if img.shape[:-1] != (patch_size, patch_size):
        img = resize(img, (patch_size, patch_size), order=1, clip=True, preserve_range=True, anti_aliasing=True)
    if isinstance(img_gt, np.ndarray) and img_gt.shape[:-1] != (patch_size, patch_size):
        order = 1 if workflow in ["super-resolution", "image-to-image", "denoising", "self-supervised"] else 0
        img_gt = resize(img_gt, (patch_size, patch_size), order=order, clip=True, preserve_range=True, anti_aliasing=True)

    # Normalization
    img = reduce_dtype(img, img.min(), img.max(), out_min=0, out_max=255, out_type="uint8")
    if workflow in ["super-resolution", "image-to-image", "denoising", "self-supervised"]:
        # Normalize img_gt, as in this workflow case it is also a raw image
        img_gt = reduce_dtype(img_gt, img_gt.min(), img_gt.max(), out_min=0, out_max=255, out_type="uint8")

    # Same procedure with the img_gt in those workflows where the target is also an image
    if workflow in ["super-resolution", "image-to-image", "denoising", "self-supervised"]:
        # Convert to RGB
        if img_gt.shape[-1] == 1:
            img_gt = np.stack((img_gt[..., 0],) * 3, axis=-1)
        elif img_gt.shape[-1] == 2:
            img_gt = np.stack((np.zeros(img_gt.shape, dtype=img_gt.dtype), img_gt), axis=-1)
        elif img_gt.shape[-1] > 3:
            img_gt = img_gt[..., :3]

        # Create cover with image and img_gt side-by-side
        out = np.ones((patch_size, patch_size * 2, 3), dtype=img.dtype)
        out[:, :patch_size] = img.copy()
        out[:, patch_size:] = img_gt.copy()
    elif workflow in ["classification"]:
        # In classification the img_gt is a class index, so just create a cover with the image
        out = img.copy()
    else:
        if img_gt.max() <= 1:
            img_gt = (img_gt * 255).astype(np.uint8)

        # Create cover with image and img_gt split by the diagonal
        if img_gt.shape[-1] == 1:
            out = np.ones(img.shape, dtype="uint8")
            for c in range(img.shape[-1]):
                outc = np.tril(img[..., c])
                img_gt_tril = outc == 0
                outc[img_gt_tril] = np.triu(img_gt[..., 0])[img_gt_tril]
                out[..., c] = outc
        else:
            # Create cover with image and img_gt side-by-side considering all channels of the img_gt
            out = np.ones((patch_size, patch_size * (img_gt.shape[-1] + 1), 3), dtype=img.dtype)
            out[:, :patch_size] = img.copy()
            for c in range(img_gt.shape[-1]):
                out[:, patch_size * (c + 1) : patch_size * (c + 2)] = np.stack((img_gt[..., c],) * 3, axis=-1)

    # Save the cover
    os.makedirs(out_path, exist_ok=True)
    cover_path = os.path.join(out_path, "cover.png")
    print(f"Creating cover: {cover_path}")
    imwrite(os.path.join(out_path, "cover.png"), out)

    return cover_path


def create_model_doc(
    biapy_obj: Any,
    bmz_cfg: dict,
    cfg_file: str,
    task_description: str,
    doc_output_path: str,
):
    """
    Create a documentation file with information of the workflow and model used. It will be saved into ``doc_output_path``.

    Parameters
    ----------
    biapy_obj : biapy
        BiaPy class instance.

    bmz_cfg : dict
        BMZ configuration to export the model. Expected keys are:

        description : str
            Description of the model.

        authors : list of dicts
            Authors of the model. Need to be a list of dicts. E.g. ``[{"name": "Gizmo"}]``.

        model_name : str
            Name of the model. If not set a name based on the selected configuration
            will be created.

        license : str
            A `SPDX license identifier <https://spdx.org/licenses/>`__. E.g. "CC-BY-4.0", "MIT",
            "BSD-2-Clause".

        tags : List of str
            Tags to make models more findable on the website. Only set useful information related to
            the data the model was trained with, as the BiaPy will introduce the rest of the tags for you,
            such as dimensions, software ("biapy" in this case), workflow used etc.
            E.g. ``['electron-microscopy','mitochondria']``.

        data : dict
            Information of the data used to train the model. Expected keys:
                * ``name``: Name of the dataset.
                * ``doi``: DOI of the dataset or a reference to find it.
                * ``image_modality``: image modality of the dataset.

    cfg_file : str
        Path to the YAML configuration file used.

    task_description : str
        Description of the task.

    doc_output_path : str
        Output path for the documentation.
    """
    # Check keys
    needed_info = [
        "description",
        "authors",
        "license",
        "tags",
        "model_name",
        "data",
    ]
    for x in needed_info:
        if x not in bmz_cfg:
            raise ValueError(f"'{x}' property must be declared in 'bmz_cfg'")

    if not isinstance(bmz_cfg["data"], dict):
        raise ValueError("'bmz_cfg['data']' needs to be a dict.")
    else:
        needed_info = [
            "name",
            "doi",
            "image_modality",
        ]
        for x in needed_info:
            if x not in bmz_cfg["data"]:
                raise ValueError(f"'{x}' property must be declared in 'bmz_cfg['data']'")

    if not isinstance(bmz_cfg["authors"], list):
        raise ValueError("'bmz_cfg['authors']' needs to be a list of dicts. E.g. [{'name': 'Daniel'}]")
    else:
        if len(bmz_cfg["authors"]) == 0:
            raise ValueError("'bmz_cfg['authors']' can not be empty.")
        for d in bmz_cfg["authors"]:
            if not isinstance(d, dict):
                raise ValueError("'bmz_cfg['authors']' must be a list of dicts. E.g. [{'name': 'Daniel'}]")
            else:
                if len(d.keys()) < 2 or "name" not in d or "github_user" not in d:
                    raise ValueError("Author dictionary must have at least 'name' and 'github_user' keys")
                for k in d.keys():
                    if k not in [
                        "name",
                        "affiliation",
                        "email",
                        "github_user",
                        "orcid",
                    ]:
                        raise ValueError(
                            "Author dictionary available keys are: ['name', 'affiliation', 'email', "
                            f"'github_user', 'orcid']. Provided {k}"
                        )

    biapy_cfg = biapy_obj.cfg # type: ignore
    workflow = biapy_obj.workflow # type: ignore

    # Workflow name
    if biapy_cfg.PROBLEM.TYPE == "SEMANTIC_SEG":
        workflow_name = "Semantic segmentation"
        workflow_tag = "semantic_segmentation"
        metrics_used = "- Intersection over Union (IoU), also referred as the [Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index), is essentially a method to quantify the percent of overlap between the target mask and the prediction output."
    elif biapy_cfg.PROBLEM.TYPE == "INSTANCE_SEG":
        workflow_name = "Instance segmentation"
        workflow_tag = "instance_segmentation"
        metrics_used = "- Intersection over Union (IoU), also referred as the [Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index), is essentially a method to quantify the percent of overlap between the target mask and the prediction output. In this workflow, as in BiaPy a botton-down approach is used to generate the final instances, IoU is used to measure the model output with the mask created out of the instances."
        metrics_used += "\n- Matching metrics, which focus on quantifying correctly predicted instances, transforming instance segmentation results into an object detection framework. In this paradigm, the emphasis shifts from uniquely labeled instances to detecting the presence or absence of instances. This transformation is achieved by establishing a criterion for instance overlap, commonly measured through IoU. Unlike traditional segmentation evaluations that rely on nuanced pixel-level overlaps, this approach simplifies assessment by classifying instances as successful true positives (TP) based on a predefined IoU threshold. In BiaPy, the following metrics are implemented: false positives (FP), true positives (TP), false negatives (FN), [precision](https://en.wikipedia.org/wiki/Precision_and_recall), [recall](https://en.wikipedia.org/wiki/Precision_and_recall), [accuracy](https://en.wikipedia.org/wiki/Accuracy_and_precision#In_binary_classification), [F1-score](https://en.wikipedia.org/wiki/Precision_and_recall#F-measure), mean_true_score (mean IoUs of matched true positives but normalized by the total number of GT objects) and panoptic_quality, which is defined as in Eq. 1 of [Kirillov et al. 'Panoptic Segmentation', CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kirillov_Panoptic_Segmentation_CVPR_2019_paper.pdf). You can find more information of these matching metrics in [3]."
    elif biapy_cfg.PROBLEM.TYPE == "DETECTION":
        workflow_name = "Object detection"
        workflow_tag = "detection"
        metrics_used = "- Intersection over Union (IoU), also referred as the [Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index), is essentially a method to quantify the percent of overlap between the target mask and the prediction output. In this workflow IoU is used to measure the model output with the ground truth point mask created out of the .csv files provided."
        metrics_used += "\n- Detection metrics, used to evaluate the final points captured against the ground truth. In BiaPy, true positives (TP), false negatives (FN), and false positives (FP) are computed and subsequently used to calculate [precision](https://en.wikipedia.org/wiki/Precision_and_recall), [recall](https://en.wikipedia.org/wiki/Precision_and_recall), and [F1-score](https://en.wikipedia.org/wiki/Precision_and_recall#F-measure)."
    elif biapy_cfg.PROBLEM.TYPE == "DENOISING":
        workflow_name = "Image denoising"
        workflow_tag = "denoising"
    elif biapy_cfg.PROBLEM.TYPE == "SUPER_RESOLUTION":
        workflow_name = "Super resolution"
        workflow_tag = "super_resolution"
    elif biapy_cfg.PROBLEM.TYPE == "SELF_SUPERVISED":
        workflow_name = "Self-supervised learning"
        workflow_tag = "self_supervision"
    elif biapy_cfg.PROBLEM.TYPE == "CLASSIFICATION":
        workflow_name = "Image classification"
        workflow_tag = "classification"
        metrics_used = "- Classification metrics, used to evaluate the predicted classes against the ground truth. In BiaPy, true positives (TP), false negatives (FN), and false positives (FP) are computed and subsequently used to calculate the [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix) and the [accuracy](https://en.wikipedia.org/wiki/Accuracy_and_precision#In_binary_classification)."
    elif biapy_cfg.PROBLEM.TYPE == "IMAGE_TO_IMAGE":
        workflow_name = "Image to image"
        workflow_tag = "image_to_image"

    if biapy_cfg.PROBLEM.TYPE in ["DENOISING", "SUPER_RESOLUTION", "IMAGE_TO_IMAGE"]:
        metrics_used = (
            "Metrics to measure the similarity between the prediction and the ground truth in different ways:"
        )
        metrics_used += "\n- [Mean Squared Error (MSE)](https://lightning.ai/docs/torchmetrics/stable/regression/mean_squared_error.html)"
        metrics_used += "\n- [Mean Absolute Error (MAE)](https://lightning.ai/docs/torchmetrics/stable/regression/mean_absolute_error.html)"
        if biapy_cfg.PROBLEM.TYPE in ["SUPER_RESOLUTION", "IMAGE_TO_IMAGE"]:
            metrics_used += "\n- [Structural Similarity Index Measure (SSIM)](https://lightning.ai/docs/torchmetrics/stable/image/structural_similarity.html)"
            metrics_used += "\n- [Frechet Inception Distance (FID)](https://lightning.ai/docs/torchmetrics/stable/image/frechet_inception_distance.html)"
            metrics_used += "\n- [Inception Score (IS)](https://lightning.ai/docs/torchmetrics/stable/image/inception_score.html)"
            metrics_used += "\n- [Learned Perceptual Image Patch Similarity (LPIPS)](https://lightning.ai/docs/torchmetrics/stable/image/learned_perceptual_image_patch_similarity.html)"

    author_mes = ""
    for aut in bmz_cfg["authors"]:
        auth = aut["name"]
        if "Daniel Franco-Barranco" in auth:
            auth = re.sub(
                r"daniel franco-barranco", 
                r"[Daniel Franco-Barranco](https://orcid.org/0000-0002-1109-110X)", 
                auth, 
                flags=re.IGNORECASE
            )
        if "Ignacio Arganda-Carreras" in auth:
            auth = re.sub(
                r"ignacio arganda-carreras",
                r"[Ignacio Arganda-Carreras](https://orcid.org/0000-0003-0229-5722)",
                auth,
                flags=re.IGNORECASE
            )
        if "Arrate Muñoz-Barrutia" in auth:
            auth = re.sub(
                r"arrate muñoz-barrutia",
                r"[Arrate Muñoz-Barrutia](https://orcid.org/0000-0002-1573-1661)",
                auth,
                flags=re.IGNORECASE
            )
        author_mes += f"- {auth} (github: {aut['github_user']})"
        if "email" in aut:
            author_mes += f" , {aut['email']}"
        if "affiliation" in aut:
            author_mes += f" ({aut['affiliation']})"
        if "orcid" in aut:
            author_mes += f" - ORCID: {aut['orcid']}"
        author_mes += "\n"

    # preprocessing info
    if biapy_cfg.DATA.NORMALIZATION.TYPE == "div":
        preproc_info = "- Division to 255 (or 65535 if uint16) to scale the data to [0,1] range.\n"
    elif biapy_cfg.DATA.NORMALIZATION.TYPE == "scale_range":
        preproc_info = "- Scaling the range to [0-max] and then dividing by the maximum value of the data.\n"
    elif biapy_cfg.DATA.NORMALIZATION.TYPE == "zero_mean_unit_variance":    
        preproc_info = "- Zero mean and unit variance normalization. "
        if biapy_cfg.DATA.NORMALIZATION.ZERO_MEAN_UNIT_VAR.MEAN_VAL > 0:
            preproc_info += f"Using provided mean value of {biapy_cfg.DATA.NORMALIZATION.ZERO_MEAN_UNIT_VAR.MEAN_VAL}. "
        else:
            preproc_info += "Mean value calculated from the training data. "
        if biapy_cfg.DATA.NORMALIZATION.ZERO_MEAN_UNIT_VAR.STD_VAL > 0:
            preproc_info += f"Using provided std value of {biapy_cfg.DATA.NORMALIZATION.ZERO_MEAN_UNIT_VAR.STD_VAL}.\n"
        else:
            preproc_info += "Std value calculated from the training data.\n"

    if biapy_cfg.DATA.NORMALIZATION.PERC_CLIP.ENABLE:
        preproc_info += "- Percentile clipping. "
        if biapy_cfg.DATA.NORMALIZATION.PERC_CLIP.LOWER_VALUE >= 0 and biapy_cfg.DATA.NORMALIZATION.PERC_CLIP.UPPER_VALUE >= 0:
            preproc_info += f"Using provided lower and upper values of {biapy_cfg.DATA.NORMALIZATION.PERC_CLIP.LOWER_VALUE} and {biapy_cfg.DATA.NORMALIZATION.PERC_CLIP.UPPER_VALUE}, respectively, to clip the data before normalization.\n"
        else:
            preproc_info += f"Using provided lower and upper percentiles of {biapy_cfg.DATA.NORMALIZATION.PERC_CLIP.LOWER_PERC} and {biapy_cfg.DATA.NORMALIZATION.PERC_CLIP.UPPER_PERC}, respectively, to calculate the values to clip the data before normalization.\n"

    if biapy_cfg.DATA.NORMALIZATION.MEASURE_BY == "image":
        preproc_info += "- Normalization and percentile clipping values calculated from the complete image.\n"
    else:
        preproc_info += "- Normalization and percentile clipping values calculated from each patch.\n"
 
    try:
        with open(cfg_file, "r") as file:
            cfg_data = yaml.safe_load(file)
    except:
        cfg_data = {}

    cfg_data_mes = ""
    if cfg_data != {}:
        cfg_data_mes = yaml.safe_dump(
            cfg_data,
            sort_keys=False,           # keep insertion order
            default_flow_style=False,  # block style (multi-line)
            width=1000                 # avoid line wrapping
        )

    message = ""
    message += f'# {bmz_cfg["model_name"]}\n'
    message += "\n"
    message += f"{task_description}.\n"
    message += "\n"
    message += "## Training details\n"
    message += "\n"
    message += f"This model was trained using [BiaPy](https://biapyx.github.io/) [1]. Complete information on how to train this model can be found in BiaPy's documentation, specifically under [{workflow_name.lower()} workflow](https://biapy.readthedocs.io/en/latest/workflows/{workflow_tag}.html) description. If you want to reproduce this model training please use the configuration described below (Technical specification section).\n"
    message += "\n"
    message += "### Training Data\n"
    message += "\n"
    message += "- Imaging modality: {}\n".format(bmz_cfg["data"]["image_modality"])
    message += f"- Dimensionality: {biapy_cfg.PROBLEM.NDIM}\n"
    message += "- Source: {} ; DOI: {}\n".format(bmz_cfg["data"]["name"], bmz_cfg["data"]["doi"])
    message += "\n"
    message += "### Training procedure\n"
    message += "#### Preprocessing\n"
    message += f"{preproc_info}"
    message += "\n"
    message += "## Evaluation\n"
    message += "\n"
    message += "### Metrics\n"
    message += "\n"
    message += f"In the case of {workflow_name.lower()} the following metrics are calculated:\n"
    message += f"{metrics_used}\n"
    message += f"\nMore info in [BiaPy documentation](https://biapy.readthedocs.io/en/latest/workflows/{workflow_tag}.html#metrics)."
    message += "\n"
    if workflow.test_metrics_message != "" or workflow.train_metrics_message != "":
        message += "### Results\n"
        if workflow.train_metrics_message != "":
            message += "#### Training results\n"
            if biapy_cfg.DATA.VAL.FROM_TRAIN:
                if not biapy_cfg.DATA.VAL.CROSS_VAL:
                    message += "The validation data was extracted from the training data using a split of {}%.\n".format(float(biapy_cfg.DATA.VAL.SPLIT_TRAIN*100))
                else:
                    message += "The validation data was extracted from the training data using cross-validation with {} folds. The fold chosen for this validation is {}.\n".format(int(biapy_cfg.DATA.VAL.CROSS_VAL_NFOLD), int(biapy_cfg.DATA.VAL.CROSS_VAL_FOLD))
            else:
                message += "The validation data was independent from the training data. Please take a look to the validation data path's in the configuration file used to train the model. Specifically to 'DATA.VAL.PATH' and 'DATA.VAL.GT_PATH' variables.\n"
            message += "The final metrics obtained from the training phase are:\n"
            for line in workflow.train_metrics_message.split("\n"):
                if line.strip() != "":
                    message += f"- {line}\n"
        if workflow.test_metrics_message != "":
            message += "#### Test results\n"
            for line in workflow.test_metrics_message.split("\n"):
                if line.strip() != "":
                    message += f"- {line}\n"
        message += "\n**Clarifications on the terminology:**\n"
        if "merge patches" in workflow.test_metrics_message:
            message += "As discussed in [2], it is common to divide the images into smaller patches for processing (and even more when working with 3D images). This approach can lead to discrepancies in metric calculations depending on whether they are computed on individual patches or on the reconstructed full images. To address this, we provide metrics calculated in different manners. Normally the metrics reported are 'merge patches'. Below are explanations for the terms used:\n"
            message += "- Metrics labeled as 'per patch' are calculated on the patches extracted from the images\n"
            message += "- Metrics labeled as 'merge patches' are calculated on the complete images after reconstructing them from the patches\n"
        else: # full image case
            message += "We provide metrics calculated in different manners. Below are explanations for the terms used:\n"
            message += "- Metrics labeled as 'per image' are computed by feeding the complete images into the model and evaluating the predictions on the whole image\n"
        if "as 3D stack" in workflow.test_metrics_message:
            message += "- Metrics labeled as 'as 3D stack' are calculated on the complete 3D images reconstructed from 2D images\n"
        if "post-processing" in workflow.test_metrics_message:
            message += "- Metrics labeled as 'post-processing' are calculated after applying all the post-processings selected\n"
        message += "\n"
    if cfg_data_mes != "":
        message += "## Technical specifications\n"
        message += "This model was trained using BiaPy (v{}). To reproduce the results, make sure to install the same BiaPy version and run it ".format(biapy.__version__)
        message += "with the configuration provided below. You will need to change the paths to the data accordingly.\n"
        message += "```yaml\n"
        message += "{}\n".format(cfg_data_mes)
        message += "```\n"
        message += "\n"
    message += "## Contact\n"
    message += "For problems with BiaPy library itself checkout our [FAQ & Troubleshooting section](https://biapy.readthedocs.io/en/latest/get_started/faq.html).\n"
    message += "\n"
    message += "For questions or issues with this models, please reach out by:\n"
    message += "- Opening a topic with tags bioimageio and biapy on [image.sc](https://forum.image.sc/)\n"
    message += "- Creating an issue in https://github.com/BiaPyX/BiaPy\n"
    message += "\n"
    message += "Model created by:\n"
    message += f"{author_mes}\n"
    message += "\n"
    message += "## References\n"
    message += "> [1] Franco-Barranco, Daniel, et al. \"BiaPy: Accessible deep learning on bioimages.\" Nature Methods (2025): 1-3.\n"
    message += "> \n"
    message += "> [2] Franco-Barranco, Daniel, Arrate Muñoz-Barrutia, and Ignacio Arganda-Carreras. \"Stable deep neural network architectures for mitochondria segmentation on electron microscopy volumes.\" Neuroinformatics 20.2 (2022): 437-450.\n"
    if biapy_cfg.PROBLEM.TYPE == "INSTANCE_SEG":
        message += "> [3] Franco-Barranco, Daniel, et al. \"Current progress and challenges in large-scale 3d mitochondria instance segmentation.\" IEEE transactions on medical imaging 42.12 (2023): 3956-3971.\n"
    f = open(doc_output_path, "w")
    f.write(message)
    f.close()
