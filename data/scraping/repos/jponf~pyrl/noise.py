# -*- coding: utf-8 -*-

import abc
import six

import numpy as np


###############################################################################

@six.add_metaclass(abc.ABCMeta)
class ActionNoise(object):

    @abc.abstractmethod
    def reset(self):
        """Prepares the action noise strategy to run a new episode."""

    @abc.abstractmethod
    def __call__(self):
        """Computes the noise for the next action."""
        raise NotImplementedError()


class NullActionNoise(ActionNoise):

    def __init__(self, action_shape):
        self._noise = np.zeros(action_shape)

    def reset(self):
        pass

    def __call__(self):
        return self._noise

    def __repr__(self):
        return "NullActionNoise(action_shape={!r}".format(self._noise.shape)


class NormalActionNoise(ActionNoise):

    def __init__(self, mu, sigma,
                 clip_min=None,
                 clip_max=None,
                 random_state=None):
        assert ((clip_min is None and clip_max is None) or
                (clip_min.shape == clip_max.shape))
        self.mu = mu
        self.sigma = sigma
        self.clip_min = clip_min
        self.clip_max = clip_max

        if random_state is None:
            self.random_state = np.random.RandomState()
            self.random_state.set_state(np.random.get_state())
        else:
            self.random_state = np.random

    def __call__(self):
        noise = self.random_state.normal(self.mu, self.sigma)
        if self.clip_min is not None:
            np.clip(noise, self.clip_min, self.clip_max, out=noise)
        return noise

    def reset(self):
        pass

    def __repr__(self):
        fmt = "NormalActionNoise(mu={!r}, sigma={!r}, random_state={!r})"
        return fmt.format(self.mu, self.sigma, self.random_state)


class OUActionNoise(object):
    """Noise generated with an Ornstein-Uhlenbeck."""

    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        rand_values = np.random.normal(size=self.mu.shape)
        x = (self.x_prev +
             self.theta * (self.mu - self.x_prev) * self.dt +
             self.sigma * np.sqrt(self.dt) * rand_values)
        self.x_prev = x
        return x

    def reset(self):
        if self.x0 is not None:
            self.x_prev = self.x0
        else:
            self.x_prev = np.zeros_like(self.mu)

    def __repr__(self):
        fmt = 'OUNoise(mu={}, sigma={}, theta={}, dt={}, x0={})'
        return fmt.format(self.mu, self.sigma, self.theta, self.dt, self.x0)


class AdaptiveParamNoiseSpec(object):
    """Adaptative Parameter Noise

    Technique is introduced in the paper Parameter Space Noise for Exploration
    https://arxiv.org/abs/1706.01905

    This implementation is taken from OpenAI's baselines.
    """

    def __init__(self,
                 initial_stddev=0.1,
                 desired_stddev=0.1,
                 adoption_coefficient=1.01,
                 current_stddev=None):
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_stddev
        self.adoption_coefficient = adoption_coefficient

        self.current_stddev = initial_stddev

    def adapt(self, distance):
        if distance > self.desired_action_stddev:  # Decrease stddev.
            self.current_stddev /= self.adoption_coefficient
        else:                                      # Increase stddev.
            self.current_stddev *= self.adoption_coefficient

    def __repr__(self):
        fmt = ('AdaptiveParamNoiseSpec(initial_stddev={}, '
               'desired_action_stddev={}, adoption_coefficient={})')
        return fmt.format(self.initial_stddev,
                          self.desired_action_stddev,
                          self.adoption_coefficient)
