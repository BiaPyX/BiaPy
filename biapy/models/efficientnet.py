import torchvision.models as models
import torch.nn as nn

def efficientnet(efficientnet_name, image_shape, n_classes=2, load_imagenet_weights=True):
    """
    Create EfficientNet.

    Reference: `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/abs/1905.11946>`_. 

    Parameters
    ----------
    efficientnet_name : str, optional
        Efficientnet model name to be loaded. Available options: "efficientnet_b[0-7]"

    image_shape : 2D tuple
        Dimensions of the input image.

    n_classes: int, optional
        Number of classes.

    Returns
    -------
    model : Torch model
        EfficientNet model.
    """
    model = getattr(models, efficientnet_name)(weights="IMAGENET1K_V1" if load_imagenet_weights else None) 

    # Change the final classification head.
    model.classifier[1] = nn.Linear(in_features=1280, out_features=n_classes)

    return model


