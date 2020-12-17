"""Code extracted from https://github.com/jacobgil/keras-grad-cam """

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Sequential
from tensorflow.python.framework import ops
import tensorflow.keras.backend as K
import numpy as np
import os
import cv2
tf.compat.v1.disable_eager_execution() # to use tf.gradients()


def grad_cam_sample(input_model, image, predicted_class, layer_name, out_dir,
                    n_classes=2):
    """Generates an image with the activation maps in charge of the class 
       decision on a specific layer.
    
       For a more detailed information refer to the paper:
       `Grad-CAM: Visual Explanations from Deep Networks via Gradient-based 
       Localization <https://arxiv.org/abs/1610.02391>`_.

       Parameters
       ----------
       input_model : Keras model
           Model.
      
       image : 2D Numpy array
           Image to visualize the heatmap from. E. g. ``(x, y)``.
    
       predicted_class : int
           Number of the class predicted.

       layer_name : str 
           Keras layer name to extract the features from.
        
       out_dir : str or Path
           Path to save the image on.
        
       n_classes : int
           Total number of classes.

       Examples
       -------- 
       ::

           # Extract the activation maps responsive of selecting the foreground
           # class (1) in a binary segmentation taks on the layer 'conv2d_16'.
           # The image should be any image one could predict() on. Notice that 
           # the number of classes is 2, which should correspond to setting 
           # n_classes=2 on the provided templates 
           grad_cam_sample(unet_model, img, 1, 'conv2d_16', 'out_dir', 2)

       +-----------------------------------+-------------------------------------------+
       | .. figure:: img/FIBSEM_test_0.png | .. figure:: img/out_gradcam_conv2d_16.png |
       |   :width: 80%                     |   :width: 70%                             |
       |   :align: center                  |   :align: center                          |
       |                                   |                                           |
       |   Input image                     |   Output of Grad-CAM                      |
       +-----------------------------------+-------------------------------------------+

       ::
  
           # Notice that, if you select the activation maps of the last layer of
           # the network, for example, 'conv2d_18' in 2D U-Net implementation of
           # this project, the output should be the same as the prediction on
           # the complete image
           grad_cam_sample(unet_model, img, 1, 'conv2d_18', 'out_dir', 2)

       +----------------------------------+-------------------------------------------+
       | .. figure:: img/gradcam_pred.png | .. figure:: img/out_gradcam_conv2d_18.png |
       |   :width: 90%                    |   :width: 70%                             |
       |   :align: center                 |   :align: center                          |
       |                                  |                                           |
       |   Network prediction             |   Output of Grad-CAM                      |
       +----------------------------------+-------------------------------------------+
    """

    os.makedirs(out_dir, exist_ok=True)
    image = np.expand_dims(image, axis=0)

    # Create a model, that is the same as the one provided but adding a last 
    # lambda layer 
    target_layer = lambda x: target_category_loss(x, predicted_class, n_classes)
    x = Lambda(target_layer, output_shape = target_category_loss_output_shape)(input_model.output)
    model = Model(inputs=input_model.input, outputs=x)

    # Compute the gradient of the score for the selected class w.r.t. feature 
    # maps activations of the convolutional layer
    loss = K.sum(model.output)
    conv_output = [l for l in model.layers if l.name == layer_name][0].output
    grads = normalize(_compute_gradients(loss, [conv_output])[0])
    gradient_function = K.function([model.input], [conv_output, grads])
    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam, (image.shape[2], image.shape[1]))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)
    image = image[0, :]*255
    image -= np.min(image)
    image = np.minimum(image, 255)

    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.stack((image[...,0],)*3, axis=-1)
    cam = 255 * cam / np.max(cam)

    cv2.imwrite(os.path.join(out_dir, "out_gradcam_"+layer_name+".jpg"),
                np.uint8(cam))

def target_category_loss(x, predicted_class, n_classes):                        
    return tf.multiply(x, K.one_hot([predicted_class], n_classes))              
                                                                                
def target_category_loss_output_shape(input_shape):                             
    return input_shape                                                          
                                                                                
def normalize(x):                                                               
    # utility function to normalize a tensor by its L2 norm                     
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)                             
                                                                                
def _compute_gradients(tensor, var_list):                                       
    """Necessary to allow 0 gradient value.                                     
                                                                                
       Extracted from https://github.com/jacobgil/keras-grad-cam/issues/17#issuecomment-398053700
    """                                                                         
    grads = tf.gradients(tensor, var_list)                                      
    return [grad if grad is not None else tf.zeros_like(var)                    
           for var, grad in zip(var_list, grads)]   
