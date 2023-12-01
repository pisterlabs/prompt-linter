# A Performance-Based Start State Curriculum Framework for Reinforcement Learning
# Copyright (c) 2021 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# This source code is derived from OpenAI Gym 0.10.9
#   (https://github.com/openai/gym/tree/0.10.9)
# Copyright (c) 2016 OpenAI (https://openai.com), licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree,
# and from rllab-curriculum
#   (https://github.com/florensacc/rllab-curriculum)
# Copyright (c) 2016 rllab contributors, licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.


import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class Arm3dKeyEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.ee_indices = [14, 23]  # the hill z-axis is 16
        self.key_hole_center = np.array([0.0, 0.3, -0.55])
        self.goal_dist = 3e-2
        self.kill_outside = True
        self.kill_radius = 0.4
        theta = -np.pi / 2
        d = 0.15
        self.goal_position = np.array(
            [0.0, 0.3, -0.55 - d,  # heel
             0.0, 0.3, -0.25 - d,  # top
             0.0 + 0.15 * np.sin(theta), 0.3 + 0.15 * np.cos(theta), -0.4 - d])  # side
        self.cost_params = {
            'wp': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
            'l1': 0.1,
            'l2': 10.0,
            'alpha': 1e-5}
        mujoco_env.MujocoEnv.__init__(self, 'arm3d_key_tight.xml', 1)
        utils.EzPickle.__init__(self)
        # self.init_qpos = np.array([0.1, 0.1, -1.54, -1.7, 1.54, -0.2, 0])
        self.init_qpos = np.array([1.55, 0.4, -3.75, -1.15, 1.81, -2.09, 0.05])
        self.init_qvel = np.zeros_like(self.init_qpos)


    def step(self, a, goal, env_col):
        # self.forward_dynamics(action)
        self.do_simulation(a, self.frame_skip)
        next_obs = self._get_obs()
        # key_position = self.get_body_com('key_head1')
        # ee_position = self.model.data.site_xpos[0]
        ee_position = next_obs[self.ee_indices[0]:self.ee_indices[1]]
        hill_pos = np.array(ee_position[:3])
        dist = np.sum(np.square(self.goal_position - ee_position) * self.cost_params['wp'])
        dist_cost = np.sqrt(dist) * self.cost_params['l1'] + dist * self.cost_params['l2']
        # reward = - dist_cost
        done = True if np.sqrt(dist) < self.goal_dist else False
        """
        if done:
            print(done)
        """
        # print("Here!!!")
        # print(np.linalg.norm(hill_pos - self.key_hole_center))
        # print(np.linalg.norm(hill_pos - self.goal_position[:3]))
        # if self.kill_outside and (np.linalg.norm(hill_pos - self.goal_position[:3]) > self.kill_radius):
        if done:
            reward = 1.0
        else:
            reward = 0.0
        no_goal_reached = not done
        if self.kill_outside and np.linalg.norm(hill_pos - self.key_hole_center) > self.kill_radius:
            # print("\n****** OUT of region ******")
            # print(self.kill_radius)
            # print(np.linalg.norm(hill_pos - self.key_hole_center))
            done = True
        return next_obs, reward, done, dict(no_goal_reached=no_goal_reached)

    def roll_out(self, a):
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")
        reward = 0
        done = False
        ob = self._get_obs_reduced()
        return ob, reward, done, dict(no_goal_reached = not done)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            self.sim.data.site_xpos.flat,
        ]).reshape(-1)

    def reset_model(self, start=None):
        # print("here")
        # print(self.init_qpos)
        if not start:
            qpos = self.init_qpos # + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        else:
            qpos = np.array(start)
        qvel = self.init_qvel # + self.np_random.randn(self.model.nv) * .1
        # print(qpos.shape)
        # print(qvel.shape)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        # self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = self.model.stat.extent * 0.3
        self.viewer.cam.lookat[0] += 0.0
        self.viewer.cam.lookat[1] += 0.0
        self.viewer.cam.lookat[2] += 0.0
        self.viewer.cam.elevation = 360
        self.viewer.cam.azimuth = -90
"""

from contextlib import contextmanager

    @contextmanager
    def set_kill_outside(self, kill_outside=True, radius=None):
        self.kill_outside = kill_outside
        old_kill_radius = self.kill_radius
        if radius is not None:
            self.kill_radius = radius
        try:
            yield
        finally:
            self.kill_outside = False
            self.kill_radius = old_kill_radius
"""
