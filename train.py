import tensorflow as tf
import numpy as np

def one_hot(index, num_classes):
    """

    :param index:
    :param num_classes:
    :return: one hot label of one sample
    """
    assert index < num_classes and index >= 0
    tmp = np.zeros(num_classes, dtype=np.float32)
    tmp[index] = 1.0
    return tmp

def input():
    """

    :return: data and labels placeholder for building model
    """
    x = tf.placeholder(
        dtype=tf.float32,
        shape=[None, 153, 204],
        name='input')
    y_ = tf.placeholder(
        dtype=tf.float32,
        shape=[None, 4],
        name='label')
    return x, y_

def conv2d_relu(x, filter_height, filter_width, num_filters_out,
                 stride_y, stride_x, name, padding='SAME'):
    """

    :param x: image
    :param filter_height:
    :param filter_width:
    :param num_filters_out:
    :param stride_y:
    :param stride_x:
    :param name:
    :param padding: default = SAME
    :return: output of conv layer
    """
    input_channels = int(x.get_shape()[-1])
    weights = tf.get_variable(
        name='weights',
        shape=[filter_height, filter_width, input_channels, num_filters_out],
        initializer=tf.contrib.layers.xavier_initializer())
    bias = tf.get_variable(
        name='biases',
        shape=num_filters_out,
        initializer=tf.constant_initializer(0.0))
    conv2d = tf.nn.conv2d(
        weights=weights,
        strides=[1, stride_y, stride_x, 1],
        padding=padding)
    return tf.nn.relu(tf.nn.bias_add(value=conv2d, bias=bias, name='add_bias'))

def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
    """
    Max pooling layer
    :param x:
    :param filter_height:
    :param filter_width:
    :param stride_y:
    :param stride_x:
    :param name:
    :param padding:
    :return: max_pool
    """
    max_pool = tf.nn.max_pool(
        value=x,
        ksize=[1, filter_height, filter_width, 1],
        strides=[1, stride_y, stride_x, 1],
        padding=padding,
        name=name)
    return max_pool

def fully_connected(x, num_out, name, relu=True):
    """
    Fully connected layer, if rele equal True out will be applied ReLu non linearity,
    else return x*weights + bias
    :param x:
    :param num_out:
    :param name:
    :param relu:
    :return:
    """
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable(
            name='weights',
            shape=[int(x.get_shape()[1]), num_out],
            trainable=True)
        biases = tf.get_variable(
            name='biases',
            shape=[num_out],
            trainable=True)
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)
    if relu:
        # Apply ReLu non linearity
        relu = tf.nn.relu(act)
        return relu
    else:
        return act

def dropout(x, keep_prob):
    """
    drop layer, drop units have probility
    :param x:
    :param keep_prob:
    :return:
    """
    return tf.nn.dropout(x, keep_prob)

def flatten(x):


def reference(x):



