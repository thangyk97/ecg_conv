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
    """
    flatten ndarray to vector
    :param x:
    :return:
    """
    shape = x.get_shape().as_list()
    new_shape = np.prod(shape[1:])
    x = tf.reshape(x, [-1, new_shape], name='flatten')
    return x

def reference(x):
    """

    :param x:
    :return: output predict
    """
    with tf.variable_scope('scope1'):
        conv1 = conv2d_relu(x, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        pool1 = max_pool(conv1, 3, 3, 2, 2, padding='VALID', name='pool1')

    with tf.variable_scope('scope2'):
        conv2 = conv2d_relu(pool1, 5, 5, 256, 1, 1, name='conv2')
        pool2 = max_pool(conv2, 3, 3, 2, 2, padding='VALID', name='pool2')

    with tf.variable_scope('scope3'):
        conv3 = conv2d_relu(pool2, 3, 3, 384, 1, 1, name='conv3')

    with tf.variable_scope('scope4'):
        conv4 = conv2d_relu(conv3, 3, 3, 384, 1, 1, name='conv4')

    with tf.variable_scope('scope5'):
        conv5 = conv2d_relu(conv4, 3, 3, 384, 1, 1, name='conv5')
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

    with tf.variable_scope('scope6'):
        flattened = tf.reshape(pool5)
        fc6 = fully_connected(flattened, 4096, name='fc6', relu=True)
        dropout6 = dropout(fc6, 0.5)

    with tf.variable_scope('scope7'):
        fc7 = fully_connected(dropout6, 4096, name='fc7')
        dropout7 = dropout(fc7, 0.7)

    with tf.variable_scope('scope8'):
        fc8 = fully_connected(dropout7, 4, relu=False, name='fc8')

    return fc8



