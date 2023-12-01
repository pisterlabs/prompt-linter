import numpy as np
import tensorflow as tf

from maml_rl.policies import CategoricalMLPPolicy, NormalMLPPolicy
from maml_rl.policies.distributions import CategoricalPd, DiagGaussianPd, CategoricalPdType, DiagGaussianPdType

"""
Code partially adapted from
https://github.com/openai/baselines/blob/tf2/baselines/common/distributions.py
"""


def weighted_mean(tensor, axis=None, weights=None):
    if weights is None:
        out = tf.reduce_mean(tensor)
    if axis is None:
        out = tf.reduce_sum(tensor * weights)
        out = out / tf.reduce_sum(weights)
    else:
        mean_dim = tf.reduce_sum(tensor * weights, axis=axis)
        mean_dim = mean_dim/(tf.reduce_sum(weights, axis=axis))
        out = tf.reduce_mean(mean_dim)
    return out


def weighted_normalize(tensor, axis=None, weights=None, epsilon=1e-8):
    mean = weighted_mean(tensor, axis=axis, weights=weights)
    out = tensor * (1 if weights is None else weights) - mean
    std = tf.math.sqrt(weighted_mean(out ** 2, axis=axis, weights=weights))
    out = out/(std + epsilon)
    return out


# ================================================================
# Flat vectors (from OpenAI Baselines)
# ================================================================

def var_shape(x):
    out = x.get_shape().as_list()
    assert all(isinstance(a, int) for a in out), \
        "shape function assumes that shape is fully known"
    return out


def numel(x):
    return intprod(var_shape(x))


def intprod(x):
    return int(np.prod(x))


def flatgrad(grads, var_list, clip_norm=None):
    if clip_norm is not None:
        grads = [tf.clip_by_norm(grad, clip_norm=clip_norm) for grad in grads]
    return tf.concat(axis=0, values=[
        tf.reshape(grad if grad is not None else tf.zeros_like(v), [numel(v)])
        for (v, grad) in zip(var_list, grads)
    ])


class SetFromFlat(object):
    def __init__(self, var_list, dtype=tf.float32):
        assigns = []
        self.shapes = list(map(var_shape, var_list))
        self.total_size = np.sum([intprod(shape) for shape in self.shapes])
        self.var_list = var_list

    def __call__(self, theta):
        start = 0
        for (shape, v) in zip(self.shapes, self.var_list):
            size = intprod(shape)
            v.assign(tf.reshape(theta[start:start + size], shape))
            start += size


class GetFlat(object):
    def __init__(self, var_list):
        self.var_list = var_list

    def __call__(self):
        return tf.concat(axis=0, values=[tf.reshape(v, [numel(v)]) for v in self.var_list]).numpy()


def flattenallbut0(x):
    return tf.reshape(x, [-1, intprod(x.get_shape().as_list()[1:])])


# ================================================================
# Distributions
# ================================================================

def make_pdtype(latent_shape, ac_space, init_scale=1.0):
    from gym import spaces
    if isinstance(ac_space, spaces.Box):
        assert len(ac_space.shape) == 1
        return DiagGaussianPdType(latent_shape, ac_space.shape[0], init_scale)
    elif isinstance(ac_space, spaces.Discrete):
        return CategoricalPdType(latent_shape, ac_space.n, init_scale)
    else:
        raise ValueError('No implementation for {}'.format(ac_space))


def detach_distribution(pi):
    if isinstance(pi, CategoricalPd):
        distribution = CategoricalPd(logits=tf.identity(pi.logits.numpy()))
    elif isinstance(pi, DiagGaussianPd):
        mean = tf.identity(pi.mean.numpy())
        logstd = tf.Variable(tf.identity(pi.logstd.numpy()), name='old_pi/logstd', trainable=False, dtype=tf.float32) # TODO: trainable=True?
        pdparam = tf.concat([mean, tf.zeros_like(mean) + logstd], axis=-1)
        distribution = DiagGaussianPd(pdparam)
    else:
        raise NotImplementedError('Only `Categorical` and `Normal` '
                                  'policies are valid policies.')
    return distribution


def clone_policy(policy, params=None, with_names=False):

    if params is None:
        params = policy.get_trainable_variables()

    if isinstance(policy, CategoricalMLPPolicy):
        cloned_policy = CategoricalMLPPolicy(input_size=policy.input_size,
                                             output_size=policy.output_size,
                                             hidden_sizes=policy.hidden_sizes,
                                             nonlinearity=policy.nonlinearity)
    elif isinstance(policy, NormalMLPPolicy):
        cloned_policy = NormalMLPPolicy(input_size=policy.input_size,
                                        output_size=policy.output_size,
                                        hidden_sizes=policy.hidden_sizes,
                                        nonlinearity=policy.nonlinearity)
    else:
        raise NotImplementedError('Only `Categorical` and `Normal` '
                                  'policies are valid policies at the moment.')

    #x = tf.zeros(shape=(1, cloned_policy.input_size))
    #cloned_policy(x)

    if with_names:
        cloned_policy.set_params_with_name(params)
    else:
        cloned_policy.set_params(params)

    return cloned_policy

