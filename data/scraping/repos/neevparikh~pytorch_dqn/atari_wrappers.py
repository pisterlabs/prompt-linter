import gym
from collections import deque
import numpy as np
import torchvision.transforms as T

# Adapted from OpenAI Baselines:
# https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
class AtariPreprocess(gym.Wrapper):

    def __init__(self, env, shape=(84, 84)):
        """ Preprocessing as described in the Nature DQN paper (Mnih 2015) """
        gym.Wrapper.__init__(self, env)
        self.shape = shape
        self.transforms = T.Compose([
            T.ToPILImage(mode='YCbCr'),
            T.Lambda(lambda img: img.split()[0]),
            T.Resize(self.shape),
            T.Lambda(lambda img: np.array(img, copy=False)),
        ])
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=self.shape,
            dtype=np.uint8,
        )

    def reset(self, **kwargs):
        return self.transforms(self.env.reset(**kwargs))

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        return self.transforms(next_state), reward, done, info

class MaxAndSkipEnv(gym.Wrapper):

    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape,
                                    dtype=np.uint8)
        self._skip = skip

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, done, info

class FrameStack(gym.Wrapper):

    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=((shp[-1] * k,) + shp[:-1]),
            dtype=env.observation_space.dtype)

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


class LazyFrames(object):

    def __init__(self, frames):
        """This object ensures that common frames between the observations are
        only stored once.  It exists purely to optimize memory usage which can
        be huge for DQN's 1M frames replay buffers.  This object should only be
        converted to numpy array before being passed to the model."""
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
