"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R

Modified from OpenAI to match the cartpole system implemented by
Russ Tedrake in his awesome book Underactuated Robotics.  Unlike
Russ' book, here we are using moment of inertia
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import autograd.numpy as anp
from autograd import jacobian
from os import path

class ClassicBaseEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self, task):
        self.task = task

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        raise NotImplementedError()

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        if self.task == "balance":
            self.state[1] += np.pi
        return np.array(self.state)

    def render(self, mode='human'):
        raise NotImplementedError()


    def close(self):
        if self.viewer: self.viewer.close()


    def _dyn(self, state, torque):
        """
        Calculate the accelerations
        """
        u = torque
        pos = state[:self.n_coords]
        vel = state[self.n_coords:]

        Minv = self._Minv(pos)
        B = self._B()
        C = self._C(pos, vel)
        G = self._G(pos)
        acc = np.dot(Minv, B.dot(u) - C.dot(vel.reshape((2,1))) - G)
        return acc.flatten()

    def _M(self, pos):
        """
        Inertial Mass matrix
        """
        raise NotImplementedError()

    def _Minv(self, pos):
        """
        Invert the mass matrix
        """
        return np.linalg.inv(self._M(pos))

    def _C(self, pos, vel):
        """
        Coriolis matrix
        """
        raise NotImplementedError()

    def _G(self, pos):
        """
        Gravitional matrix
        """
        raise NotImplementedError()

    def _B(self, pos):
        """
        Force matrix
        """
        raise NotImplementedError()


    def _linearize(self, pos, vel):
        """
        Linearize the system dynamics around a given point
        """
        pos = state[0:2]
        return self._Alin(state), self._Blin(pos)

    def _Alin(self, state):
        pos = state[:self.n_coords]
        vel = state[self.n_coords:]
        ul = np.zeros((self.n_coords, self.n_coords))
        ur = np.eye(self.n_coords)
        Minv = self._Minv(pos)
        deltaG = self._deltaG()
        C = self._C(pos, vel)
        ll = -np.dot(Minv, deltaG)
        lr = -np.dot(Minv, C)
        return np.block([[ul, ur],
                        [ll, lr]])

    def _Blin(self, pos):
        Z = np.dot(self._Minv(pos), self._B(pos))
        return np.block([
                        [np.zeros_like(Z)],
                        [Z]
                        ])

    def _deltaG(self):
        raise NotImplementedError()

    def total_energy(self):
        raise NotImplementedError()


    def kinetic_energ(self, pos, vel):
        return

    def potential_energy(self, pos):
        return

    def desired_energy(self):
        return self.potential_energy()

    def desired_energy(self):
        raise NotImplementedError()

    def _unwrap_angle(self, theta):
        sign = (theta >=0)*1 - (theta < 0)*1
        theta = np.abs(theta) % (2 * np.pi)
        return sign*theta

    def integrate(self):
        """
        Integrate the equations of motion
        """
        raise NotImplementedError()


