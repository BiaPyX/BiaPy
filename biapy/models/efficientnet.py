"""
This module provides functionality to create and configure EfficientNet models from `torchvision.models`.

It allows users to instantiate various EfficientNet architectures (e.g., b0-b7),
optionally load pre-trained ImageNet weights, and customize the final
classification layer for specific tasks.
"""
import torchvision.models as models
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url
from typing import List

import torch.nn as nn

from biapy.models.blocks import prepare_activation_layers


def efficientnet(
    efficientnet_name,
    n_classes=2,
    load_imagenet_weights=True,
    head_activations: List[str] = ["ce_softmax"],
    output_channel_info: List[str] = ["pred0"],
    explicit_activations: bool = False,
):
    """
    Create EfficientNet.

    Reference: `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/abs/1905.11946>`_.

    Parameters
    ----------
    efficientnet_name : str, optional
        Efficientnet model name to be loaded. Available options: "efficientnet_b[0-7]"

    n_classes: int, optional
        Number of classes.

    load_imagenet_weights : bool, optional
        Whether to load IMAGENET weights or not. 

    head_activations : List[str], optional
        Activations of the output channels, as defined by the workflow. Only used if ``explicit_activations``.

    output_channel_info : List[str], optional
        Information of each output head, as defined by the workflow. Only used if ``explicit_activations``.

    explicit_activations : bool, optional
        If True, the model applies ``head_activations`` to its output. Defaults to False, as BiaPy workflows
        apply them outside the model.

    Returns
    -------
    model : Torch model
        EfficientNet model.
    """

    def get_state_dict(self, *args, **kwargs):
        kwargs.pop("check_hash")
        return load_state_dict_from_url(self.url, *args, **kwargs)

    WeightsEnum.get_state_dict = get_state_dict

    model = getattr(models, efficientnet_name)(weights="IMAGENET1K_V1" if load_imagenet_weights else None)

    # Change the final classification head.
    model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=n_classes)
    if explicit_activations:
        # Appended to the classifier so the state dict keys do not change
        class_head_activations, _ = prepare_activation_layers(head_activations, output_channel_info, [n_classes])
        model.classifier.append(class_head_activations[0])

    return model
