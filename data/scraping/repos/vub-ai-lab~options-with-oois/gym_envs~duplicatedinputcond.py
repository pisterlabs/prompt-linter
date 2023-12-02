# This file is part of OOIs.
#
# Copyright 2017, Vrije Universiteit Brussel (http://vub.ac.be)
#
# OOIs is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OOIs is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from gym.envs.algorithmic import algorithmic_env

# Copied and modified from OpenAI Gym
class DuplicatedInputCondEnv(algorithmic_env.TapeAlgorithmicEnv):

    def __init__(self, duplication=2, base=5):
        self.duplication = duplication
        super(DuplicatedInputCondEnv, self).__init__(base=base, chars=True)

    def generate_input_data(self, size):
        res = []
        self._dedup = []

        while len(res) < size:
            char = self.np_random.randint(self.base)
            
            # Odd characters are duplicated
            res.append(char)
            self._dedup.append(char)
            
            if (char % 2) == 1:
                res.append(char)

        return res

    def target_from_input_data(self, input_data):
        return self._dedup
