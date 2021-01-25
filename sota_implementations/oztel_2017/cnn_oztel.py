from tensorflow import keras
from tensorflow.keras import layers

def create_oztel_model(num_classes=2, input_shape=(None, None, 1),
                       optimizer='adam', loss='categorical_crossentropy'):
    """Create the CNN proposed by Oztel et. al.

       Parameters
       ----------   
       num_classes : int, optional
           Number of classes to predict.
        
       input_shape : 3D tuple, optional
           Dimensions of the input image.

       optimizer : str, optional
           Optimizer used to minimize the loss function. 
        
       loss : str, optional
           Loss function to use according to Keras options.
        
       Returns
       -------
       model : Keras model
           Model containing the CNN created.


       Here is a picture of the network extracted from the original paper:
                                                                                
       .. image:: img/oztel_network.png
           :width: 100%                                                         
           :align: center
    """

    model = keras.Sequential(
        [
            # Block 1
            keras.Input(shape=input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, kernel_size=(5, 5), padding='same', strides=1),
            layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='same' ),
            layers.Activation('relu'),
            # Block 2
            layers.Conv2D(32, kernel_size=(5, 5), padding='same', activation="relu"),
            layers.AveragePooling2D(pool_size=(3, 3), strides=2, padding='same'),
            layers.Dropout(0.5),
            # Block 3
            layers.Conv2D(64, kernel_size=(5, 5), padding='same', activation="relu"),
            layers.AveragePooling2D(pool_size=(3, 3), strides=2, padding='same'),
            # Block 4
            layers.Conv2D(64, kernel_size=(4, 4), strides=1, padding='valid', activation="relu"),
            layers.Conv2D(num_classes, (1, 1), activation='softmax'),
        ]
    )

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model
