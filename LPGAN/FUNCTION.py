import tensorflow as tf
import numpy as np
import os

from time import localtime, strftime
from datetime import datetime
from scipy import io

def current_time():
    return strftime("%Y-%m-%d %H:%M:%S", localtime())

def safe_casting(data, dtype):
    output = np.clip(data + 0.5, np.iinfo(dtype).min, np.iinfo(dtype).max)
    output = output.astype(dtype)
    return output


def random_pad_to_size(img, size, mask, pad_symmetric, use_random):
    if mask is None:
        mask = np.ones(shape=img.shape)
    s0 = size - img.shape[0]
    s1 = size - img.shape[1]

    if use_random:
        b0 = np.random.randint(0, s0 + 1)
        b1 = np.random.randint(0, s1 + 1)
    else:
        b0 = 0
        b1 = 0
    a0 = s0 - b0
    a1 = s1 - b1
    if pad_symmetric:
        img  = np.pad(img,  ((b0, a0), (b1, a1), (0, 0)), 'symmetric')
    else:
        img  = np.pad(img,  ((b0, a0), (b1, a1), (0, 0)), 'constant')
    mask = np.pad(mask, ((b0, a0), (b1, a1), (0, 0)), 'constant')
    return img, mask, [b0, img.shape[0] - a0, b1, img.shape[1] - a1]

def tf_imgradient(tensor):
    B, G, R = tf.unpack(tensor, axis=-1)
    tensor = tf.pack([R, G, B], axis=-1)
    tensor = tf.image.rgb_to_grayscale(tensor)
    #tensor = tensor * 255;
    sobel_x = tf.constant([[1, 0, -1], [2, 0, -2], [1, 0, -1]], tf.float32)
    sobel_x_filter = tf.reshape(sobel_x, [3, 3, 1, 1])
    sobel_y_filter = tf.transpose(sobel_x_filter, [1, 0, 2, 3])
    #tensor = tf.pad(tensor, [[0, 0], [1, 1], [1, 1], [0, 0]], 'SYMMETRIC')
    fx = tf.nn.conv2d(tensor, sobel_x_filter, strides=[1,1,1,1], padding='VALID')
    fy = tf.nn.conv2d(tensor, sobel_y_filter, strides=[1,1,1,1], padding='VALID')
    g = tf.sqrt(tf.square(fx) + tf.square(fy))
    return g

def initialize_uninitialized(sess):
    global_vars          = tf.global_variables()
    is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

#    print [str(i.name) for i in not_initialized_vars] # only for testing
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))
