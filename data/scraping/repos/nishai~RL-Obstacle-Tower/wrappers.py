"""
Useful wrappers taken from OpenAI (https://github.com/openai/baselines)
"""

import numpy as np
from collections import deque
import gym
from gym import spaces
import cv2

cv2.ocl.setUseOpenCL(False)


## Actions to be used
# 18 - Forward
# 6  -
# 12 -
# 34 -
# 36 -
# 39 -

HUMAN_ACTIONS = (18, 6, 12, 36, 24, 30)
NUM_ACTIONS = len(HUMAN_ACTIONS)


class HumanActionEnv(gym.ActionWrapper):
    """
    An environment wrapper that limits the action space to the speicifc actions outlined in HUMAN_ACTIONS
    Created by UnixPickle on github
    @see https://github.com/unixpickle/obs-tower2/blob/5c1753524eb84c816875896ad0dd041375737275/obs_tower2/util.py
    """
    def __init__(self, env):
        super().__init__(env)
        self.actions = HUMAN_ACTIONS
        self.action_space = gym.spaces.Discrete(len(self.actions))

    def action(self, act):
        return self.actions[act]



class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers."""
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=0)
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, i):
        return self._frames[i]


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        Expects inputs to be of shape num_channels x height x width.
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0] * k, shp[1], shp[2]), dtype=np.uint8)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))



class PyTorchFrame(gym.ObservationWrapper):
    """
    Image shape to num_channels x height x width
    @see https://github.com/raillab/dqn/blob/master/dqn/wrappers.py
    """
    def __init__(self, env):
        super(PyTorchFrame, self).__init__(env)
        shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(shape[-1], shape[0], shape[1]), dtype=np.uint8)

    def observation(self, observation):
        return np.rollaxis(observation, 2)
