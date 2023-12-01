"""
Environments and wrappers for Sonic training.
Part of this file are copied from OpenAI's retro_baselines
https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py
"""
import time
import os
import gym
import numpy as np
import sys

from math import floor
from random import randrange

from retro_contest.local import make
from baselines.common.atari_wrappers import WarpFrame
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines.bench import Monitor


def make_env(game, level, scale_rew=True, video_dir=""):
    """
    Create an environment with some standard wrappers.
    """
    env = make(game=game, state=level, bk2dir=video_dir)
    env = SonicDiscretizer(env)
    if scale_rew:
        env = RewardScaler(env)
    env = WarpFrame(env)
    # env = ShortLife(env)
    env = AllowBacktracking(env)
    return env


def make_sonic_env(
    game, level, num_env, seed=0, log_dir=None, start_index=0, start_method=None
):
    if log_dir is None and log_dir != "":
        log_dir = "/tmp/gym/{}/".format(int(time.time()))
    os.makedirs(log_dir, exist_ok=True)

    def init(rank):
        def _thunk():
            env = make_env(game, level)
            env.seed(seed + rank)
            env = Monitor(
                env, os.path.join(log_dir, str(rank)), allow_early_resets=True
            )
            return env

        return _thunk

    set_global_seeds(seed)

    # When using one environment, no need to start subprocesses
    # if num_env == 1:
    #     return DummyVecEnv([init(0)])

    return SubprocVecEnv(
        [init(i + start_index) for i in range(num_env)], start_method=start_method
    )


class SonicDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """

    def __init__(self, env):
        super(SonicDiscretizer, self).__init__(env)
        buttons = [
            "B",
            "A",
            "MODE",
            "START",
            "UP",
            "DOWN",
            "LEFT",
            "RIGHT",
            "C",
            "Y",
            "X",
            "Z",
        ]
        actions = [
            ["LEFT"],
            ["RIGHT"],
            ["LEFT", "DOWN"],
            ["RIGHT", "DOWN"],
            ["DOWN"],
            ["DOWN", "B"],
            ["B"],
        ]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):  # pylint: disable=W0221
        return self._actions[a].copy()


class RewardScaler(gym.RewardWrapper):
    """
    Bring rewards to a reasonable scale for PPO.

    This is incredibly important and effects performance
    drastically.
    """

    def reward(self, reward):
        return reward * 0.01


class AllowBacktracking(gym.Wrapper):
    """
    Use deltas in max(X) as the reward, rather than deltas
    in X. This way, agents are not discouraged too heavily
    from exploring backwards if there is no way to advance
    head-on in the level.
    """

    def __init__(self, env):
        super(AllowBacktracking, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0

    def reset(self, **kwargs):  # pylint: disable=E0202
        self._cur_x = 0
        self._max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action):  # pylint: disable=E0202
        obs, rew, done, info = self.env.step(action)
        self._cur_x += rew
        rew = max(0, self._cur_x - self._max_x)
        self._max_x = max(self._max_x, self._cur_x)
        return obs, rew, done, info


class ShortLife(gym.Wrapper):
    def __init__(self, env):
        super(ShortLife, self).__init__(env)
        self.steps_done = 0
        self.max_steps = 150
        self.death_count = 0
        self.curr_run_reward = 0

    def reset(self, **kwargs):  # pylint: disable=E0202
        self.steps_done = 0
        self.curr_run_reward = 0
        return self.env.reset(**kwargs)

    def step(self, action):  # pylint: disable=E0202
        obs, rew, done, info = self.env.step(action)
        self.curr_run_reward += rew
        self.steps_done += 1
        if self.steps_done == self.max_steps:
            self.death_count += 1
            self.max_steps += 15 + int(self.curr_run_reward / 2)
            self.reset()
            done = True
        return obs, rew, done, info
