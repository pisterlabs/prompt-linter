"""
Copyright (c) 2023 Robert Bosch GmbH

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

"""
Environment modified from OpenAI's gym inverted pendulum implementation, under MIT license
https://github.com/openai/gym/blob/v0.21.0/gym/envs/classic_control/pendulum.py
Copyright (c) 2016 OpenAI, licensed under the MIT license,
cf. thirdparty_licenses.md file in the root directory of this source tree.
"""

from os import path

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from .utils import tolerance


class SparsePendulumEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(
        self,
        mass: float = 0.3,
        length: float = 0.5,
        friction: float = 0.005,
        action_cost: float = 0.0,
        step_size: float = 1 / 80,
        noise_std: float = 0.0,
    ):
        self.mass = mass
        self.length = length
        self.friction = friction
        self.dt = step_size
        self.action_cost = action_cost
        self.max_torque = 1
        self.g = 9.81
        self.viewer = None
        self.noise_std = noise_std

        high = np.array([300, 300, 300], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_action_cost(self, cost):
        self.action_cost = cost

    def set_noise_std(self, std):
        self.noise_std = std

    def step(self, u):
        u = np.clip(u, -self.max_torque, self.max_torque)
        self.last_u = u  # for rendering
        inertia = self.mass * self.length**2

        th, th_dot = self.state
        # Reward
        cos_th = np.cos(th)
        angle_tolerance = tolerance(cos_th, lower=0.95, upper=1.0, margin=0.1)
        velocity_tolerance = tolerance(th_dot, lower=-0.5, upper=0.5, margin=0.5)
        state_reward = angle_tolerance * velocity_tolerance

        action_tolerance = tolerance(u, lower=-0.1, upper=0.1, margin=0.1)
        action_cost = self.action_cost * (action_tolerance - 1)

        cost = state_reward + action_cost

        # Dynamics
        th_ddot = (
            (self.g / self.length) * np.sin(th)
            + u * (1 / inertia)
            - (self.friction / inertia) * th_dot
        )
        th = th + self.dt * th_dot + self.np_random.normal(loc=0, scale=self.noise_std)
        th_dot = (
            th_dot + self.dt * th_ddot  # + 0.1 * self.np_random.normal(loc=0, scale=self.noise_std)
        )
        self.state = np.array([th, th_dot[0]])

        # Termination for large states or actions
        done = np.any(th > 200 or th_dot > 200 or u > 200)

        return self._get_obs(), cost.item(), done, {}

    def render(self, mode="human"):
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(512, 512)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, 0.2)
            rod.set_color(0.8, 0.3, 0.3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(0.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1.0, 1.0)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u is not None:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def reset(self):
        # high = np.array([np.pi, 0])
        # self.state = self.np_random.uniform(low=-high, high=high)
        self.state = np.array([np.pi, 0])
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
