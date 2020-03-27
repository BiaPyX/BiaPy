import tensorflow as tf
import numpy as np
import sys
from keras import backend as K


def custom_loss(aux1, aux2):

    def loss(y_true,y_pred):
        
        print("true:" + str(K.int_shape(y_true)))
        print("pred:" + str(K.int_shape(y_pred)))
        print("aux1:" + str(K.int_shape(aux1)))
        print("aux2:" + str(K.int_shape(aux2)))
        L_x_w = tf.keras.backend.categorical_crossentropy(y_true,y_pred)
  
        w1 = 0.15
        L_x_w_aux1 = tf.keras.backend.categorical_crossentropy(aux1,y_pred)

        w2 = 0.3
        L_x_w_aux2 = tf.keras.backend.categorical_crossentropy(aux2,y_pred)
              
        regu = 0

        total_loss =  L_x_w + w1*L_x_w_aux1 + w2*L_x_w_aux2 + regu
        #total_loss =  L_x_w 

        return total_loss
    
    return loss

#def custom_loss(aux1, aux2):
#
#    def loss(y_true,y_pred):
#       
#        comparison0 = tf.equal(y_true, tf.constant(0, dtype=tf.float32))
#        comparison1 = tf.equal(y_true, tf.constant(1, dtype=tf.float32))
#        t_size = tf.cast(tf.size(y_pred), dtype=tf.float32)
#
#        #L_x_w = -1*tf.math.reduce_sum(tf.where(comparison0, tf.math.log(tf.expand_dims(y_pred[...,0], -1)), tf.zeros_like(tf.expand_dims(y_pred[...,0], -1))))\
#        #        + -1*tf.math.reduce_sum(tf.where(comparison1, tf.math.log(tf.expand_dims(y_pred[...,1],-1)), tf.zeros_like(tf.expand_dims(y_pred[...,1], -1))))
#        #L_x_w = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(y_true[...,0], dtype=tf.int32), logits=q) 
#        #L_x_w = -tf.reduce_sum(y_true * tf.log(tf.expand_dims(y_pred[...,1],-1))) - tf.reduce_sum(tf.cast(tf.bitwise.invert(tf.cast(y_true, dtype=tf.int32)),dtype=tf.float32) * tf.log(tf.expand_dims(y_pred[...,0],-1)))
#        reshaped_logits = tf.reshape(y_pred, [-1, 3])
#        reshaped_labels = tf.cast(tf.reshape(y_true, [-1]), dtype=tf.int32)
#        L_x_w =  tf.nn.sparse_softmax_cross_entropy_with_logits(labels=reshaped_labels, logits=reshaped_logits)
#        #flat_true = tf.reshape(y_true, shape=(-1))
#        #flat_logits = tf.reshape(y_pred, shape=(-1,2))
#        # Auxiliary classifiers
##        w1 = 0.15
##        L_x_w_aux1 = -1*tf.math.reduce_sum(tf.where(comparison0, tf.math.log(tf.expand_dims(aux1[...,0],-1)), tf.zeros_like(tf.expand_dims(aux1[...,0],-1))))\
##                + -1*tf.math.reduce_sum(tf.where(comparison1, tf.math.log(tf.expand_dims(aux1[...,1],-1)), tf.zeros_like(tf.expand_dims(aux1[...,1],-1))))
##        w2 = 0.3
##        L_x_w_aux2 = -1*tf.math.reduce_sum(tf.where(comparison0, tf.math.log(tf.expand_dims(aux2[...,0],-1)), tf.zeros_like(tf.expand_dims(aux2[...,0],-1))))\
##                + -1*tf.math.reduce_sum(tf.where(comparison1, tf.math.log(tf.expand_dims(aux2[...,1],-1)), tf.zeros_like(tf.expand_dims(aux2[...,1],-1))))
#
#        regu = 0 
#
#        #total_loss = L_x_w + w1*L_x_w_aux1 
#        #total_loss = L_x_w + w1*L_x_w_aux1 + w2*L_x_w_aux2 + regu 
#        total_loss = L_x_w
#
#        return total_loss
#
#    return loss
#def custom_loss(aux1, aux2):
#
#    def loss(y_true,y_pred):
#       
#        L_x_w = tf.math.reduce_sum(-1*tf.math.log([y_true==1] * y_pred[...,1])) \
#                + tf.math.reduce_sum(-1*tf.math.log([y_true==0] * y_pred[...,0]))
#
#        # Auxiliary classifiers
#        w1 = 0.15
#        L_x_w_aux1 = tf.math.reduce_sum(-1*tf.math.log([y_true==1] * aux1[...,1])) \
#                + tf.math.reduce_sum(-1*tf.math.log([y_true==0] * aux1[...,0]))
#        w2 = 0.3
#        L_x_w_aux2 = tf.math.reduce_sum(-1*tf.math.log([y_true==1] * aux2[...,1])) \
#                + tf.math.reduce_sum(-1*tf.math.log([y_true==0] * aux2[...,0]))
#
#        regu = 0 
#
#        total_loss = L_x_w + w1*L_x_w_aux1 + w2*L_x_w_aux2 + regu 
#
#        return total_loss
#
#    return loss
