
import time
from copy import deepcopy

import numpy as np
import retro

from utils import mirror_observation
from openai_baseline import load_baseline

class PongTragetoryGenerator:
    """ Generates tragetories based on two agents playing pong"""

    def __init__(self, player1, player2, tragetory_length = 120):
        self._env = retro.make(game='Pong-Atari2600', players = 2)
        self._player1 = player1
        self._player2 = player2
        self._tragetory_length = tragetory_length
    #

    def build(self):
        observation = self._env.reset()
        done = False
        reward_player1, reward_player2 = (0, 0)

        # game loop
        out = list()
        for _ in range(self._tragetory_length):
            if not done:
                action_player1 = self._player1.showdown_step(observation,
                                                    reward_player1)
                action_player2 = self._player2.showdown_step(mirror_observation(observation),
                                                    reward_player2)

                observation, reward, done, info = self._env.step(
                    np.concatenate((action_player1, action_player2))
                )

                reward_player1, reward_player2 = tuple(reward)
            ##

            out.append(observation)
        ##
        return out
    #
#

def generate_pong_tragetory(player1, player2, tragetory_length = 120):
    """ Generate a tragetory from the given pair of pong player agents """
    generator = PongTragetoryGenerator(player1, player2, tragetory_length)
    return generator.build()
#


##

