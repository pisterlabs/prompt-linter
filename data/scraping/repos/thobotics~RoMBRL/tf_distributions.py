# -*- coding: utf-8 -*-

"""
    tf_distributions.py
    
    Created on  : March 14, 2019
        Author  : anonymous
        Name    : Anonymous
"""

# -*- coding: utf-8 -*-
"""
Distribution functions 
Copyright (c) 2016 openai
https://github.com/openai/iaf/blob/master/tf_utils

Gumbell-softmax functions
Copyright (c) 2016 Eric Jang
https://github.com/ericjang/gumbel-softmax
"""
import numpy as np
import tensorflow as tf


##### Probabilistic functions (partially from OpenAI github)
class DiagonalGaussian(object):
    def __init__(self, mean, logvar, sample=None):
        self.mean = mean
        self.logvar = logvar

        if sample is None:
            noise = tf.random_normal(tf.shape(mean))
            sample = mean + tf.exp(0.5 * logvar) * noise
        self.sample = sample

    def logps(self, sample):
        return gaussian_diag_logps(self.mean, self.logvar, sample)


def gaussian_diag_logps(mean, logvar, sample=None):
    if sample is None:
        noise = tf.random_normal(tf.shape(mean))
        sample = mean + tf.exp(0.5 * logvar) * noise
    return tf.clip_by_value(-0.5 * (np.log(2 * np.pi) + logvar + tf.square(sample - mean) / tf.exp(logvar)),-(10e10),10e10)


def logsumexp(x):
    x_max = tf.reduce_max(x, [1], keep_dims=True)
    return tf.reshape(x_max, [-1]) + tf.log(tf.reduce_sum(tf.exp(x - x_max), [1]))

def exp_normalize(x, axis):
    x_max = tf.reduce_max(x, axis=axis, keep_dims=True)
    y = tf.exp(x - x_max)
    return y / tf.reduce_sum(y, axis=axis, keep_dims=True)
