import os

import tensorflow as tf
import numpy as np


def save_model(actor, basedir=None):
    if not os.path.exists('models/'):
        os.makedirs('models/')

    actor_path = "{}/actor".format(basedir)

    # print('Saving models to {} {}'.format(actor_path, adversary_path))
    tf.saved_model.save(actor, actor_path)


def load_model(basedir=None):
    actor_path = "{}/actor".format(basedir)

    print('Loading model from {}'.format(actor_path))
    actor = tf.saved_model.load(actor_path)
    return actor


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + tf.math.sqrt(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


# Borrowed from openai baselines running_mean_std.py
class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = tf.zeros(shape, 'float32')
        self.var = tf.ones(shape, 'float32')
        self.count = epsilon

    def update(self, x):
        batch_mean = tf.dtypes.cast(tf.math.reduce_mean(x, axis=0), tf.dtypes.float32)
        batch_var = tf.dtypes.cast(tf.math.square(tf.math.reduce_std(x, axis=0)), tf.dtypes.float32)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)
