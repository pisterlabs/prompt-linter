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


import time
from gym import Wrapper, logger

class TimeLimit(Wrapper):
    def __init__(self, env, max_episode_seconds=None, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        self._max_episode_seconds = max_episode_seconds
        self._max_episode_steps = max_episode_steps

        self._elapsed_steps = 0
        self._episode_started_at = None

    @property
    def _elapsed_seconds(self):
        return time.time() - self._episode_started_at

    def _past_limit(self):
        """Return true if we are past our limit"""
        if self._max_episode_steps is not None and self._max_episode_steps <= self._elapsed_steps:
            logger.debug("Env has passed the step limit defined by TimeLimit.")
            return True

        if self._max_episode_seconds is not None and self._max_episode_seconds <= self._elapsed_seconds:
            logger.debug("Env has passed the seconds limit defined by TimeLimit.")
            return True

        return False

    def step(self, action, goal, env_col):
        assert self._episode_started_at is not None, "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action, goal, env_col)
        self._elapsed_steps += 1

        if self._past_limit():
            if self.metadata.get('semantics.autoreset'):
                _ = self.reset() # automatically reset the env
            done = True 

        return observation, reward, done, info

    def roll_out(self, action):
        assert self._episode_started_at is not None, "Cannot call env.step() before calling reset()"
        observation = self.env.roll_out(action)
        self._elapsed_steps += 1

        if self._past_limit():
            if self.metadata.get('semantics.autoreset'):
                _ = self.reset() # automatically reset the env
            done = True

        return observation

    def reset(self, start):
        self._episode_started_at = time.time()
        self._elapsed_steps = 0
        return self.env.reset(start)
