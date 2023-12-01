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
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.


import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import copy

class PointEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'point_spiral_2D.xml', 5)
        utils.EzPickle.__init__(self)

    def step(self, a, goal, env_col):
        # print(goal)
        # print(env_col)
        if False and env_col:
            ob_old = copy.deepcopy(self._get_obs())
            self.do_simulation(a, self.frame_skip)
            ob_temp = self._get_obs()
            collision = env_col.check_collision(ob_temp)
            if collision:
                self.set_state(qpos=ob_old[:2], qvel=ob_old[2:4])
        else:
            self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")
        reward = 0
        done = False
        goal_dist = np.linalg.norm(xposafter[:2] - goal)
        if goal_dist < 0.5:
            reward = 1.0
            done = True
        """
        if done:
            reward = 300
        else:
            reward = max(-1 * (goal_dist - 0.3) * 1000, 0)
        """
        ob = self._get_obs()
        return ob, reward, done, dict(no_goal_reached = not done, dist = goal_dist)

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
            self.sim.data.qvel.flat
        ])

    def _get_obs_reduced(self):
        return self.sim.data.qpos[:2].flat

    def reset_model(self, start):
        qpos = np.zeros(self.model.nq)
        qvel = np.zeros(self.model.nv)
        qpos[0] = start[0]
        qpos[1] = start[1]
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = self.model.stat.extent * 1.5
        self.viewer.cam.lookat[0] += 2.5
        self.viewer.cam.lookat[1] += 0.5
        self.viewer.cam.lookat[2] += 0.5
        self.viewer.cam.elevation = -90
        self.viewer.cam.azimuth = 0    
