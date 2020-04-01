import tensorflow as tf
import numpy as np

def custom_loss(aux1, aux2):

    def loss(y_true,y_pred):
        L_x_w = tf.keras.backend.categorical_crossentropy(y_true,y_pred)
  
        w1 = 0.15
        L_x_w_aux1 = tf.keras.backend.categorical_crossentropy(aux1,y_pred)

        w2 = 0.3
        L_x_w_aux2 = tf.keras.backend.categorical_crossentropy(aux2,y_pred)
              
        total_loss =  L_x_w + w1*L_x_w_aux1 + w2*L_x_w_aux2

        return total_loss
    
    return loss
