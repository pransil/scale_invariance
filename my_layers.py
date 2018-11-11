# my_layers.py
''' layers.py - layer definitions'''

import tensorflow as tf
import numpy as np

import tensorboard_utils


'''Fully connected layer, softmax or relu'''
def fc_layer(actFn, input, out_dim, name="fc"):
    in_dim = input.shape[1]
    with tf.name_scope(name):
        lname = name + actFn
        w_name = lname + "_W"
        b_name = lname + "_B"
        with tf.name_scope(w_name):
            w = tf.get_variable(w_name, shape=[in_dim, out_dim], initializer=tf.contrib.layers.xavier_initializer())
            tf.summary.histogram(w_name, w)
        with tf.name_scope(w_name):
            b = tf.get_variable(b_name, dtype=tf.float32, initializer=np.zeros(out_dim, dtype=np.float32))
            #tf.summary.histogram(b_name, b)
        if actFn == 'relu':
            out = tf.nn.relu(tf.matmul(input, w) + b)
            act = out
        elif actFn == 'softmax':
            act = tf.matmul(input, w) + b
            out = tf.nn.softmax(act)
        elif actFn == 'linear':
            act = tf.matmul(input, w) + b
            out = act
        else:
            print ("Unknown activation function in fc_layer:", actFn)

        tf.summary.histogram('activations', act)
        tf.summary.histogram('y_pred', out)
        return act, out     # Need to rtn act because softmax_cross_entropy_with_logits() does softmax


def define_2relu_layers(x_placeholder, in_dim, out_dim):

    _, fc1_out = fc_layer(actFn='relu', input=x_placeholder, out_dim = out_dim, name="fc1_")
    _, fc2_out = fc_layer(actFn='relu', input=fc1_out, out_dim = out_dim, name="fc2_")
    act, out = fc_layer(actFn='softmax', input=fc2_out, out_dim=out_dim, name="fc3_")
    return act, out