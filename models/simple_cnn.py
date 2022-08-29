from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, Dropout, BatchNormalization, Flatten, Dense


def simple_CNN(image_shape, n_classes=2):
    """Create simple CNN.

       Parameters
       ----------
       image_shape : 2D tuple
           Dimensions of the input image.

       n_classes: int, optional
           Number of classes.

       Returns
       -------
       model : Keras model
           Model containing the simple CNN.
    """


    #dinamic_dim = (None,)*(len(image_shape)-1) + (image_shape[-1],)
    #inputs = Input(dinamic_dim, name="input")
    inputs = Input(image_shape, name="input")

    # Block 1
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    # Block 2
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    # Last convolutional block
    x = Flatten() (x)
    x = Dropout(0.5)(x)
    outputs = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model


