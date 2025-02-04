import os
import yaml
import numpy as np
from typing import Tuple
from packaging.version import Version
from yacs.config import CfgNode

from bioimageio.spec.model.v0_4 import ModelDescr as ModelDescr_v0_4
from bioimageio.spec.model.v0_5 import ModelDescr as ModelDescr_v0_5
from bioimageio.spec._internal.types import ImportantFileSource
from bioimageio.spec._internal.io_basics import Sha256
from bioimageio.spec.model.v0_5 import (
    ArchitectureFromFileDescr,
    ArchitectureFromLibraryDescr,
)
from bioimageio.spec.utils import download

from biapy.data.pre_processing import reduce_dtype, calculate_volume_prob_map
from biapy.data.data_manipulation import read_img_as_ndarray, imwrite
from biapy.data.generators.augmentors import random_crop_pair


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
        model.weights.pytorch_state_dict is not None
    ), "Seems that the original BMZ model has no pytorch_state_dict object. Aborting"

    if spec_version > Version("0.5.0"):
        arch = model.weights.pytorch_state_dict.architecture
        if isinstance(arch, ArchitectureFromFileDescr):
            arch_file_path = download(arch.source, sha256=arch.sha256).path
            arch_file_sha256 = arch.sha256
            arch_name = arch.callable
            arch_kwargs = arch.kwargs

            pytorch_architecture = ArchitectureFromFileDescr(
                source=arch_file_path,
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
            source=arch_file_path,
            sha256=arch_file_sha256,
            callable=arch_name,
            kwargs=model.weights.pytorch_state_dict.kwargs,
        )
        state_dict_source = model.weights.pytorch_state_dict.source
        state_dict_sha256 = model.weights.pytorch_state_dict.sha256

    return state_dict_source, state_dict_sha256, pytorch_architecture


def create_environment_file_for_model(building_dir):
    """
    Create a conda environment file (environment.yaml) with the necessary dependencies to build a model
    with BiaPy.

    Parameters
    ----------
    building_dir : str
        Directory to save the environment.yaml file.

    Returns
    -------
    env_file : str
        Path to the environment.yaml file created.
    """
    import biapy

    env = dict(
        name="biapy",
        channels=["conda-forge", "pytorch", "nodefaults"],
        dependencies=[
            "python>=3.10",
            "pip",
            "pytorch >=2.4.0,<3",
            {
                "pip": [
                    "biapy=={}".format(biapy.__version__),
                    "torchvision==0.19.0",
                    "torchaudio==2.4.0",
                    "timm==1.0.8",
                    "pytorch-msssim==1.0.0",
                    "torchmetrics[image]==1.4.*",
                ]
            },
        ],
    )

    os.makedirs(building_dir, exist_ok=True)
    env_file = os.path.join(building_dir, "environment.yaml")
    with open(env_file, "w", encoding="utf8") as outfile:
        yaml.dump(env, outfile, default_flow_style=False)

    return env_file


def create_model_cover(file_paths, out_path, patch_size=256, is_3d=False, workflow="semantic-segmentation"):
    """
    Create a cover based on the files that will be read from ``file_paths``.

    Parameters
    ----------
    file_paths : dict
        Contains information about the input/output images to read. It must have the following keys:
            * ``"input"`` (str): path to the input image to be loaded
            * ``"output"`` (str): path to the output image to be loaded

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
    assert "input" in file_paths
    assert "output" in file_paths
    img = read_img_as_ndarray(str(file_paths["input"]), is_3d=is_3d)
    mask = read_img_as_ndarray(str(file_paths["output"]), is_3d=is_3d)

    # Take a random patch from the image
    prob_map = None
    if workflow in ["semantic-segmentation", "instance-segmentation", "detection"]:
        prob_map = calculate_volume_prob_map([{"img": mask > 0.5}], is_3d, 1, 0)[0]
    img, mask = random_crop_pair(img, mask, (patch_size, patch_size), img_prob=prob_map)

    # If 3D just take middle slice.
    if is_3d and img.ndim == 4:
        img = img[img.shape[0] // 2]
    if is_3d and mask.ndim == 4:
        mask = mask[mask.shape[0] // 2]

    # Convert to RGB
    if img.shape[-1] == 1:
        img = np.stack((img[..., 0],) * 3, axis=-1)
    elif img.shape[-1] == 2:
        img = np.stack((np.zeros(img.shape, dtype=img.dtype), img), axis=-1)
    elif img.shape[-1] > 3:
        img = img[..., :3]

    # Normalize image
    img = reduce_dtype(img, img.min(), img.max(), out_min=0, out_max=255, out_type=np.uint8)

    # Same procedure with the mask in those workflows where the target is also an image
    if workflow in ["super-resolution", "image-to-image", "denoising", "self-supervised"]:
        # Convert to RGB
        if mask.shape[-1] == 1:
            mask = np.stack((mask[..., 0],) * 3, axis=-1)
        elif mask.shape[-1] == 2:
            mask = np.stack((np.zeros(mask.shape, dtype=mask.dtype), mask), axis=-1)
        elif mask.shape[-1] > 3:
            mask = mask[..., :3]

        # Normalize mask, as in this workflow case it is also a raw image
        mask = reduce_dtype(mask, mask.min(), mask.max(), out_min=0, out_max=255, out_type=np.uint8)

        # Create cover with image and mask side-by-side
        out = np.ones((patch_size, patch_size * 2, 3), dtype=img.dtype)
        out[:, :patch_size] = img.copy()
        out[:, patch_size:] = mask.copy()
    else:
        if mask.max() <= 1:
            mask = (mask * 255).astype(np.uint8)

        # Create cover with image and mask split by the diagonal
        if mask.shape[-1] == 1:
            out = np.ones(img.shape, dtype="uint8")
            for c in range(img.shape[-1]):
                outc = np.tril(img[..., c])
                mask_tril = outc == 0
                outc[mask_tril] = np.triu(mask[..., 0])[mask_tril]
                out[..., c] = outc
        else:
            # Create cover with image and mask side-by-side considering all channels of the mask
            out = np.ones((patch_size, patch_size * (mask.shape[-1] + 1), 3), dtype=img.dtype)
            out[:, :patch_size] = img.copy()
            for c in range(mask.shape[-1]):
                out[:, patch_size * (c + 1) : patch_size * (c + 2)] = np.stack((mask[..., c],) * 3, axis=-1)

    # Save the cover
    os.makedirs(out_path, exist_ok=True)
    cover_path = os.path.join(out_path, "cover.png")
    print(f"Creating cover: {cover_path}")
    imwrite(os.path.join(out_path, "cover.png"), out)

    return cover_path


def create_model_doc(
    biapy_cfg: CfgNode,
    bmz_cfg: dict,
    cfg_file: str,
    doc_output_path: str,
):
    """
    Create a documentation file with information of the workflow and model used. It will be saved into
    ``doc_output_path``.

    Parameters
    ----------
    biapy_cfg : CfgNode
        BiaPy configuration.

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

    cfg_file : int, optional
        Size of the image to create.

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

    # Workflow name
    if biapy_cfg.PROBLEM.TYPE == "SEMANTIC_SEG":
        workflow_name = "Semantic segmentation"
        workflow_tag = "semantic_segmentation"
        metrics_used = "- Intersection over Union (IoU), also referred as the [Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index), is essentially a method to quantify the percent of overlap between the target mask and the prediction output."
    elif biapy_cfg.PROBLEM.TYPE == "INSTANCE_SEG":
        workflow_name = "Instance segmentation"
        workflow_tag = "instance_segmentation"
        metrics_used = "- Intersection over Union (IoU), also referred as the [Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index), is essentially a method to quantify the percent of overlap between the target mask and the prediction output. In this workflow, as in BiaPy a botton-down approach is used to generate the final instances, IoU is used to measure the model output with the mask created out of the instances."
        metrics_used += "\n- Matching metrics, which focus on quantifying correctly predicted instances, transforming instance segmentation results into an object detection framework. In this paradigm, the emphasis shifts from uniquely labeled instances to detecting the presence or absence of instances. This transformation is achieved by establishing a criterion for instance overlap, commonly measured through IoU. Unlike traditional segmentation evaluations that rely on nuanced pixel-level overlaps, this approach simplifies assessment by classifying instances as successful true positives (TP) based on a predefined IoU threshold. In BiaPy, the following metrics are implemented: false positives (FP), true positives (TP), false negatives (FN), [precision](https://en.wikipedia.org/wiki/Precision_and_recall), [recall](https://en.wikipedia.org/wiki/Precision_and_recall), [accuracy](https://en.wikipedia.org/wiki/Accuracy_and_precision#In_binary_classification), [F1-score](https://en.wikipedia.org/wiki/Precision_and_recall#F-measure), mean_true_score (mean IoUs of matched true positives but normalized by the total number of GT objects) and panoptic_quality (defined as in Eq. 1 of (Kirillov et al. 'Panoptic Segmentation', CVPR 2019)[https://openaccess.thecvf.com/content_CVPR_2019/papers/Kirillov_Panoptic_Segmentation_CVPR_2019_paper.pdf])."
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
        metrics_used = "- Classification metrics, used to evaluate the predicted classes against the ground truth. In BiaPy, true positives (TP), false negatives (FN), and false positives (FP) are computed and subsequently used to calculate the (confusion matrix)[https://en.wikipedia.org/wiki/Confusion_matrix] and the (accuracy)[https://en.wikipedia.org/wiki/Accuracy_and_precision#In_binary_classification]."
    elif biapy_cfg.PROBLEM.TYPE == "IMAGE_TO_IMAGE":
        workflow_name = "Image to image"
        workflow_tag = "image_to_image"

    if biapy_cfg.PROBLEM.TYPE in ["DENOISING", "SUPER_RESOLUTION", "IMAGE_TO_IMAGE"]:
        metrics_used = (
            "Metrics to measure the similarity between the prediction and the ground truth in different ways:"
        )
        metrics_used += "\n- Mean Squared Error (MSE): [https://lightning.ai/docs/torchmetrics/stable/regression/mean_squared_error.html](https://lightning.ai/docs/torchmetrics/stable/regression/mean_squared_error.html)"
        metrics_used += "\n- Mean Absolute Error (MAE): [https://lightning.ai/docs/torchmetrics/stable/regression/mean_absolute_error.html](https://lightning.ai/docs/torchmetrics/stable/regression/mean_absolute_error.html)"
        if biapy_cfg.PROBLEM.TYPE in ["SUPER_RESOLUTION", "IMAGE_TO_IMAGE"]:
            metrics_used += "\n- Structural Similarity Index Measure (SSIM): [https://lightning.ai/docs/torchmetrics/stable/image/structural_similarity.html](https://lightning.ai/docs/torchmetrics/stable/image/structural_similarity.html)"
            metrics_used += "\n- Frechet Inception Distance (FID): [https://lightning.ai/docs/torchmetrics/stable/image/frechet_inception_distance.html](https://lightning.ai/docs/torchmetrics/stable/image/frechet_inception_distance.html)"
            metrics_used += "\n- Inception Score (IS): [https://lightning.ai/docs/torchmetrics/stable/image/inception_score.html](https://lightning.ai/docs/torchmetrics/stable/image/inception_score.html)"
            metrics_used += "\n- Learned Perceptual Image Patch Similarity (LPIPS): [https://lightning.ai/docs/torchmetrics/stable/image/learned_perceptual_image_patch_similarity.html](https://lightning.ai/docs/torchmetrics/stable/image/learned_perceptual_image_patch_similarity.html)"

    author_mes = ""
    for aut in bmz_cfg["authors"]:
        author_mes += f"- {aut['name']} (github: {aut['github_user']})"
        if "email" in aut:
            author_mes += f" , {aut['email']}"
        if "affiliation" in aut:
            author_mes += f" ({aut['affiliation']})"
        if "orcid" in aut:
            author_mes += f" - ORCID: {aut['orcid']}"
        author_mes += "\n"

    try:
        with open(cfg_file, "r") as file:
            cfg_data = yaml.safe_load(file)
        train_info = cfg_data["TRAIN"]
        aug_info = cfg_data["AUGMENTOR"]
    except:
        train_info = dict(biapy_cfg.TRAIN)
        aug_info = dict(biapy_cfg.AUGMENTOR)

    def dict_to_str(cfg, message, spaces="  "):
        for tparam, val in cfg.items():
            if isinstance(val, CfgNode):
                message += f"{spaces}{tparam}:\n"
                message += dict_to_str(val, message, spaces + "  ")
            else:
                message += f"{spaces}{tparam}: {val}\n"
        return message

    train_params = dict_to_str(train_info, "")
    aug_params = dict_to_str(aug_info, "")

    message = ""
    message += f'# {bmz_cfg["model_name"]}\n'
    message += "\n"
    message += "This model segments mitochondria in electron microscopy images.\n"
    message += "\n"
    message += "## Training\n"
    message += "\n"
    message += f"Complete information on how to train this model can be found in BiaPy documentation, specifically under [{workflow_name.lower()} workflow](https://biapy.readthedocs.io/en/latest/workflows/{workflow_tag}.html) description.\n"
    message += "\n"
    message += "### Training Data\n"
    message += "\n"
    message += "- Imaging modality: {}\n".format(bmz_cfg["data"]["image_modality"])
    message += f"- Dimensionality: {biapy_cfg.PROBLEM.NDIM}\n"
    message += "- Source: {} ; DOI:{}\n".format(bmz_cfg["data"]["name"], bmz_cfg["data"]["doi"])
    message += "\n"
    message += "### Validation (recommended)\n"
    message += "\n"
    message += f"In the case of {workflow_name.lower()} the following metrics are calculated:\n"
    message += f"{metrics_used}\n"
    message += f"\nMore info in [BiaPy documentation](https://biapy.readthedocs.io/en/latest/workflows/{workflow_tag}.html#metrics).\n"
    message += "\n"
    message += "### Training Schedule (BiaPy variables)\n"
    message += "\n"
    message += "#### Training setup\n"
    message += "\n"
    message += "TRAIN:\n"
    message += f"{train_params}\n"
    message += "#### Data augmentation\n"
    message += "\n"
    message += "AUGMENTOR:\n"
    message += f"{aug_params}\n"
    message += "## Contact\n"
    message += "For problems with BiaPy library itself checkout our [FAQ & Troubleshooting section](https://biapy.readthedocs.io/en/latest/get_started/faq.html).\n"
    message += "\n"
    message += "For questions or issues with this models, please reach out by:\n"
    message += "- Opening a topic with tags bioimageio and biapy on [image.sc](https://forum.image.sc/)\n"
    message += "- Creating an issue in https://github.com/BiaPyX/BiaPy\n"
    message += "\n"
    message += "Model created by:\n"
    message += f"{author_mes}\n"

    f = open(doc_output_path, "w")
    f.write(message)
    f.close()
