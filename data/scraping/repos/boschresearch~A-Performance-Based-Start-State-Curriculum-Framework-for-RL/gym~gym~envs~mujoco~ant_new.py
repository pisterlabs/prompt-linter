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

class AntEnvU(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'ant_new.xml', 5)
        utils.EzPickle.__init__(self)

    def step(self, a, goal, env_col):
        """
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore)/self.dt
        lb = self.model.actuator_ctrlrange[:, 0]
        ub = self.model.actuator_ctrlrange[:, 1]
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.5 * 1e-2 * np.square(a / scaling).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.05
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.3 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)
        """

        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")
        reward = 0
        done = False
        goal_dist = np.linalg.norm(xposafter[:2] - goal)
        if goal_dist < 0.5:
            reward = 1.0
            done = True
        ob = self._get_obs()
        return ob, reward, done, dict(no_goal_reached=not done)

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
            # np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            self.sim.data.get_body_xmat("torso").flat,
            self.get_body_com("torso"),
        ]).reshape(-1)

    def _get_obs_reduced(self):
        return self.sim.data.qpos[:2].flat

    def reset_model(self, start):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qpos[0] = start[0]
        qpos[1] = start[1]
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        # self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = self.model.stat.extent * 1.5
        self.viewer.cam.lookat[0] += 0.0
        self.viewer.cam.lookat[1] += 0.0
        self.viewer.cam.lookat[2] += 0.0
        self.viewer.cam.elevation = -90
        self.viewer.cam.azimuth = 0    
