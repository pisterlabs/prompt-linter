"""
Code adopted from OpenAI Gym implementation
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import autograd.numpy as anp
from autograd import jacobian
from os import path


class PendubotEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self, task="balance", initial_state=None):
        # set task
        self.task = task

        self.initial_state = initial_state
        
        # gravity
        self.gravity = self.g = 9.8

        # pole 1
        self.m1 = 0.1
        self.l1 = self.len1 = 1.0 # actually half the pole's length
        self.L1 = 2 * self.l1
        self.I1 = (1/12) * self.m1 * (2 * self.len1)**2
        self.inertia_1= (self.I1 + self.m1 * self.len1**2)
        self.pm_len1 = self.m1 * self.len1

        # pole 2
        self.m2 = 0.1
        self.l2 = self.len2 = 1.0 # actually half the pole's length
        self.L2 = 2 * self.l2
        self.I2 = (1/12) * self.m2 * (2 * self.len2)**2
        self.inertia_2 = (self.I2 + self.m2 * self.len2**2)
        self.pm_len2 = self.m2 * self.len2

        # Other params
        self.force_mag = 0.5
        self.dt = 0.02  # seconds between state updates
        self.n_coords = 2

        self.d1 = self.inertia_1 + self.m2 * self.L1**2
        self.d2 = self.inertia_2
        self.d3 = self.m2 * self.L1 * self.l2
        self.d4 = self.m1 * self.l1 + self.m2 * self.L1
        self.d5 = self.m2 * self.l2

        # precompute the jacobian of the dynamics
        self.jacobian = self._jacobian()

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        # high = np.array([
        #     self.x_threshold * 2,
        #     np.finfo(np.float32).max,
        #     self.theta_threshold_radians * 2,
        #     np.finfo(np.float32).max])

        # self.action_space = spaces.Discrete(2)
        # self.observation_space = spaces.Box(-high, high)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        if self.initial_state:
            self.state = self.initial_state
        else:
            self.state = self.np_random.uniform(low=-0.1, high=0.1, size=(2 * self.n_coords,))
            if self.task == "balance":
                self.state[0] += np.pi
        self.steps_beyond_done = None
        return np.array(self.state)

    def is_done(self):
        th1, th2 = self.state[:self.n_coords]
        if self.task == "balance":
            done =  th1 < np.pi - self.theta_threshold_radians \
                    or th1 > np.pi + self.theta_threshold_radians \
                    or th2 < np.pi - self.theta_threshold_radians \
                    or th2 > np.pi + self.theta_threshold_radians
        else:
            bool = False
        return bool(done)

    def step(self, action):
        # assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        # get state
        th1, th2, th1_dot, th2_dot = self.state
        th1 = self._unwrap_angle(th1)
        th2 = self._unwrap_angle(th2)

        # clip torque, update dynamics
        u = np.clip(action, -self.force_mag, self.force_mag)
        acc = self._accels(anp.array([th1, th2, th1_dot, th2_dot, u]))

        # integrate
        th1_acc, th2_acc = acc

        # update pole 1 position and angular velocity
        th1_dot = th1_dot + self.dt * th1_acc
        th1 = th1 + self.dt * th1_dot + 0.5 * th1_acc * self.dt**2

        # update pole 2 position and angular velocity
        th2_dot = th2_dot + self.dt * th2_acc
        th2 = th2 + self.dt * th2_dot + 0.5 * th2_acc * self.dt**2

        # update state
        th1 = self._unwrap_angle(th1)
        th2 = self._unwrap_angle(th2)
        self.state = np.array([th1, th2, th1_dot, th2_dot])
        
        # done = self.is_done()
        done = False

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def _accels(self, vec):
        """
        Calculate the accelerations
        """
        force = vec[-1]
        pos = vec[:self.n_coords]
        vel = vec[self.n_coords:-1]
        state = vec[:-1]
        Minv = self._Minv(pos)
        B = self._B()
        C = self._C(state)
        G = self._G(pos)
        acc = anp.dot(Minv, anp.dot(B, force) - anp.dot(C, vel.reshape((self.n_coords, 1))) - G)
        return acc.flatten()

    def _F(self, vec):
        """
        Return derivative of state-space vector
        """
        qd = vec[self.n_coords:-1]
        qdd = self._accels(vec)
        return anp.array(list(qd) + list(qdd))

    def _M(self, pos):
        """
        Inertial Mass matrix
        """
        th1, th2 = pos
        m11 = self.d1 + self.d2 + 2 * self.d3 * anp.cos(th2)
        m21 = m12 = self.d2 + self.d3 * anp.cos(th2)
        m22 = self.d2

        mass_matrix = anp.array([[m11, m12],
                               [m21, m22]])
        return mass_matrix

    def _C(self, state):
        """
        Coriolis matrix
        """
        th1, th2, th1_dot, th2_dot = state
        c11 = -2 * self.d3 * anp.sin(th2) * th2_dot
        c12 = -self.d3 * anp.sin(th2) * th2_dot
        c21 = self.d3 * anp.sin(th2) * th1_dot
        c22 = 0.0
        return anp.array([[c11, c12],
                        [c21, c22]])

    def _G(self, pos):
        """
        Gravitional matrix
        """
        th1, th2 = pos
        g1 = self.d4 * anp.sin(th1) * self.g
        g2 = self.d5 * anp.sin(th1 + th2) * self.g
        return anp.array([[g1],
                        [g2]])

    def _B(self):
        """
        Force matrix
        """
        return anp.array([[1], [0]])

    def _jacobian(self):
        """
        Return the Jacobian of the full state equation
        """
        return jacobian(self._F)

    def _linearize(self, vec):
        """
        Linearize the dynamics by first order Taylor expansion
        """
        f0 = self._F(vec)
        arr = self.jacobian(vec)
        A = arr[:, :-1]
        B = arr[:, -1].reshape((2 * self.n_coords, 1))
        return f0, A, B

    def _Minv(self, pos):
        """
        Invert the mass matrix
        """
        return anp.linalg.inv(self._M(pos))

    def total_energy(self, state):
        pos = state[:self.n_coords]
        vel = state[self.n_coords:]
        return self.kinetic_energy(pos, vel) + self.potential_energy(pos)

    def kinetic_energy(self, pos, vel):
        return

    def potential_energy(self, pos):
        return

    def _unwrap_angle(self, theta):
        sign = (theta >=0)*1 - (theta < 0)*1
        theta = anp.abs(theta) % (2 * anp.pi)
        return sign*theta

    def integrate(self):
        """
        Integrate the equations of motion
        """
        raise NotImplementedError()
    
    def render(self, mode='human'):
        from gym.envs.classic_control import rendering

        s = self.state

        if self.viewer is None:
            self.viewer = rendering.Viewer(500,500)
            bound = self.L1 + self.L2 + 0.2  # 2.2 for default
            self.viewer.set_bounds(-bound,bound,-bound,bound)

        if s is None: return None

        p1 = [-self.L1 *
              np.cos(s[0]), self.L1 * np.sin(s[0])]

        p2 = [p1[0] - self.L2 * np.cos(s[0] + s[1]),
              p1[1] + self.L2 * np.sin(s[0] + s[1])]

        xys = np.array([[0,0], p1, p2])[:,::-1]
        thetas = [s[0]-np.pi/2, s[0]+s[1]-np.pi/2]
        link_lengths = [self.L1, self.L2]

        self.viewer.draw_line((-2.2, 1), (2.2, 1))
        for ((x,y),th,llen) in zip(xys, thetas, link_lengths):
            l,r,t,b = 0, llen, .1, -.1
            jtransform = rendering.Transform(rotation=th, translation=(x,y))
            link = self.viewer.draw_polygon([(l,b), (l,t), (r,t), (r,b)])
            link.add_attr(jtransform)
            link.set_color(.88, .4, .4) 
            circ = self.viewer.draw_circle(.12)
            circ.set_color(.26, .26, .26)
            circ.add_attr(jtransform)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
        self.viewer = None
