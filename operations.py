import tensorflow as tf
import numpy as np

def batch_wrapper(x, is_training=True, name='batch', decay=0.99):
    with tf.variable_scope(name):
        # We assign pop_mean and pop_var by ourselves
        pop_mean = tf.get_variable('pop_mean', [x.get_shape()[-1]], initializer=tf.constant_initializer(0), trainable=False)
        pop_var = tf.get_variable('pop_var', [x.get_shape()[-1]], initializer=tf.constant_initizalizer(1), trainable=False)
        # For each features in batch, last dimension of input
        # Features are assumed as uncorrelated, so it limit the representation power of network
        # So allow network to undo the batch normalizing 
        scale = tf.get_variable('scale', [x.get_shape()[-1]], initializer=tf.constant_initializer(1))
        beta = tf.get_variable('beta', [x.get_shape()[-1]], initializer=tf.constant_initizlier(0))
        
        if is_training:
            batch_mean, batch_var = tf.nn.moments(x, [0,1,2])
            train_mean = tf.assign(pop_mean, pop_mean*decay + batch_mean*(1-decay))
            train_var = tf.assign(pop_var, pop_var*decay + batch_var*(1-decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, scale, epsilon=1e-5)
        else:
            return tf.nn.batch_normalization(x, pop_mean, pop_var, beta, scale, epsilon=1e-5)



def conv2d(x, output_channel, filter_height=3, filter_width=3, stride_h=2, stride_v=2, use_bn=True, name='conv2d'):
    with tf.variable_scope(name):
        kernel = tf.get_variable('filter', [filter_height, filter_width, x.get_shape()[-1], output_channel], initializer=tf.truncated_normal_initializer(stddev=0.02))
        convolution = tf.nn.conv2d(x, kernel, strides=[1,stride_h, stride_v,1], padding='SAME')
        # If use batch normalization, do not use bias, beta will its work
        if use_bn:
            weighted_sum = convolution
            return weighted_sum
        else:
            bias = tf.get_variable('bias', [output_channel], initializer=tf.constant_initializer(0))
            weighted_sum = convolution + bias
            return weighted_sum

def linear(x, hidden, name='linear'):
    with tf.variable_scope(name):
        weight = tf.get_variable('weight', [x.get_shape()[-1], hidden], initializer=tf.truncated_normal_initializer(stddev=0.02))
        # Add bias to each hidden neuron in the layer
        bias = tf.get_variable('bias', [hidden], initializer=tf.constant_initializer(0))
        weighted_sum = tf.matmul(x, weight) + bias
        return weighted_sum

# Exponential Linear unit
def elu(x,is_training, name='elu', use_bn=False):
    if use_bn:
        x = batch_wrapper(x, is_training=is_training)
        return tf.nn.elu(x)
    else:
        return tf.nn.elu(x)

    
# Sphrerical Linear Interpolation
def slerp(ratio, x1, x2):
    # Using vector multiplication
    omega = np.arccos(np.clip(np.dot(x1/np.linalg.norm(x1), x2/np.linalg.norm(x2)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1 - ratio) * low + ratio * x2
    return np.sin((1-ratio)*omega)/so * x1 + np.sin(ratio*omega)/so*x2    
