from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense


def efficientnetb0(image_shape, n_classes=2, load_imagenet_weights=True):
    """Create EfficientNetB0.

       Parameters
       ----------
       image_shape : 2D tuple
           Dimensions of the input image.

       n_classes: int, optional
           Number of classes.

       Returns
       -------
       model : Keras model
           Model containing the EfficientNetB0.
    """

    w = 'imagenet' if load_imagenet_weights else ''
    efnb0 = EfficientNetB0(weights=w, include_top=False, input_shape=image_shape, classes=n_classes)

    model = Sequential()
    model.add(efnb0)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))

    return model


