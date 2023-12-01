import numpy as np
import tensorflow as tf

import tensorflow_probability as tfp


class SVGD(object):
    def __init__(self, grads_list, vars_list, optimizer):
        self.grads_list = grads_list
        self.vars_list = vars_list
        self.optimizer = optimizer
        self.num_particles = len(vars_list)

    def get_pairwise_dist(self, x):
        norm = tf.reshape(tf.reduce_sum(x * x, 1), [-1, 1])
        return norm - 2 * tf.matmul(x, tf.transpose(x)) + tf.transpose(norm)

    def _get_svgd_kernel(self, X):
        stacked_vars = tf.stack(X)
        pairwise_dists = self.get_pairwise_dist(stacked_vars)
        lower = tfp.stats.percentile(
            pairwise_dists, 50.0, interpolation='lower')
        higher = tfp.stats.percentile(
            pairwise_dists, 50.0, interpolation='higher')

        median = (lower + higher) / 2
        median = tf.cast(median, tf.float32)
        h = tf.sqrt(0.5 * median / tf.math.log(len(X) + 1.))
        h = tf.stop_gradient(h)

        # kernel computation
        Kxy = tf.exp(-pairwise_dists / h ** 2 / 2)
        dxkxy = -tf.matmul(Kxy, stacked_vars)
        sumkxy = tf.reduce_sum(Kxy, axis=1, keepdims=True)

        # analytical kernel gradient
        dxkxy = (dxkxy + stacked_vars * sumkxy) / tf.pow(h, 2)

        return Kxy, dxkxy

    def get_num_elements(self, var):
        return int(np.prod(self.var_shape(var)))

    def _flatten(self, grads, variables):
        # from openai/baselines/common/tf_util.py
        flatgrads = tf.concat(axis=0, values=[
            tf.reshape(grad if grad is not None else tf.zeros_like(
                v), [self.get_num_elements(v)])
            for (v, grad) in zip(variables, grads)])
        flatvars = tf.concat(axis=0, values=[
            tf.reshape(var, [self.get_num_elements(var)])for var in variables])
        return flatgrads, flatvars

    def var_shape(self, var):
        out = var.get_shape().as_list()
        return out

    def run(self):
        all_flatgrads = []
        all_flatvars = []

        for grads, variables in zip(self.grads_list, self.vars_list):
            flatgrads, flatvars = self._flatten(grads, variables)
            all_flatgrads.append(flatgrads)
            all_flatvars.append(flatvars)

        Kxy, dxkxy = self._get_svgd_kernel(all_flatvars)
        stacked_grads = tf.stack(all_flatgrads)
        stacked_grads = tf.matmul(Kxy, stacked_grads) - dxkxy
        stacked_grads /= self.num_particles
        flatgrads_list = tf.unstack(stacked_grads, self.num_particles)

        # align index
        all_grads = []
        for flatgrads, variables in zip(flatgrads_list, self.vars_list):
            start = 0
            grads = []

            for var in variables:
                shape = self.var_shape(var)
                end = start + int(np.prod(self.var_shape(var)))
                grads.append(tf.reshape(flatgrads[start:end], shape))
                # next
                start = end

            all_grads.append(grads)

        for grads, variables in zip(all_grads, self.vars_list):
            self.optimizer.apply_gradients(
                [(-grad, var) for grad, var in zip(grads, variables)])

        return
