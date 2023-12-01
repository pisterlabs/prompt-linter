from collections import deque

import torch
import numpy as np
import torchvision.transforms as T
import gym
import cv2
import random


class IndexedObservation(gym.ObservationWrapper):
    """ 
    Description:
        Return elements of observation at given indices

    Usage:
        For example, say the base env has observations Box(4) and you want the indices 1 and 3. You
        would pass in indices=[1,3] and the observation_space of the wrapped env would be Box(2).

    Notes:
        - This currently only supports 1D observations but can easily be extended to support
          multidimensional observations
    """
    def __init__(self, env, indices):
        super(IndexedObservation, self).__init__(env)
        self.indices = indices

        assert len(env.observation_space.shape) == 1, env.observation_space
        wrapped_obs_len = env.observation_space.shape[0]
        assert len(indices) <= wrapped_obs_len, indices
        assert all(i < wrapped_obs_len for i in indices), indices
        self.observation_space = gym.spaces.Box(low=env.observation_space.low[indices],
                                                high=env.observation_space.high[indices],
                                                dtype=env.observation_space.dtype)

    def observation(self, observation):
        return observation[self.indices]


class TorchTensorObservation(gym.ObservationWrapper):
    """
    Description:
        Downsample the image observation to a given shape.
    
    Usage:
        Pass in requisite shape (e.g. 84,84) and it will use opencv to resize the observation to
        that shape

    Notes:
        - N/A
    """
    def __init__(self, env, device):
        super(TorchTensorObservation, self).__init__(env)
        self.device = device

    def observation(self, observation):
        return torch.from_numpy(observation).to(dtype=torch.float, device=self.device)


# Adapted from https://github.com/openai/gym/blob/master/gym/wrappers/resize_observation.py
class ResizeObservation(gym.ObservationWrapper):
    """
    Description:
        Downsample the image observation to a given shape.
    
    Usage:
        Pass in requisite shape (e.g. 84,84) and it will use opencv to resize the observation to
        that shape

    Notes:
        - N/A
    """
    def __init__(self, env, shape):
        super(ResizeObservation, self).__init__(env)
        if isinstance(shape, int):
            shape = (shape, shape)
        assert all(x > 0 for x in shape), shape
        self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        observation = cv2.resize(observation, self.shape[::-1], interpolation=cv2.INTER_AREA)
        return observation


class ObservationDictToInfo(gym.Wrapper):
    """
    Description:
        Given an env with an observation dict, extract the given state key as the state and pass the
        existing dict into the info. 
    
    Usage:
        Wrap any Dict observation.

    Notes:
        - By convention, no info is return on reset, so that dict is lost. 
    """
    def __init__(self, env, state_key):
        gym.Wrapper.__init__(self, env)
        assert type(env.observation_space) == gym.spaces.Dict
        self.observation_space = env.observation_space.spaces[state_key]
        self.state_key = state_key

    def reset(self, **kwargs):
        next_state_as_dict = self.env.reset(**kwargs)
        return next_state_as_dict[self.state_key]

    def step(self, action):
        next_state_as_dict, reward, done, info = self.env.step(action)
        info.update(next_state_as_dict)
        return next_state_as_dict[self.state_key], reward, done, info


class ResetARI(gym.Wrapper):
    """
    Description:
        On reset and step, grab the values of the labeled dict from info and return as state.

    Usage:
        Wrap over ARI env. 

    Notes:
        - N/A
    """
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

        # change the observation space to accurately represent
        # the shape of the labeled RAM observations
        self.observation_space = gym.spaces.Box(
            0,
            255,  # max value
            shape=(len(self.env.labels()),),
            dtype=np.uint8)

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        # reset the env and get the current labeled RAM
        return np.array(list(self.env.labels().values()))

    def step(self, action):
        # we don't need the obs here, just the labels in info
        _, reward, done, info = self.env.step(action)
        # grab the labeled RAM out of info and put as next_state
        next_state = np.array(list(info['labels'].values()))
        return next_state, reward, done, info


# Adapted from OpenAI Baselines:
# https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
class AtariPreprocess(gym.Wrapper):
    """
    Description:
        Preprocessing as described in the Nature DQN paper (Mnih 2015) 
    
    Usage:
        Wrap env around this. It will use torchvision to transform the image according to Mnih 2015

    Notes:
        - Should be decomposed into using separate envs for each. 
    """
    def __init__(self, env, shape=(84, 84)):
        gym.Wrapper.__init__(self, env)
        self.shape = shape
        self.transforms = T.Compose([
            T.ToPILImage(mode='YCbCr'),
            T.Lambda(lambda img: img.split()[0]),
            T.Resize(self.shape),
            T.Lambda(lambda img: np.array(img)),
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
    """
    Description:
        Return only every `skip`-th frame. Repeat action, sum reward, and max over last 
        observations.
    
    Usage:
        Wrap env and provide skip param.

    Notes:
        - N/A
    """
    def __init__(self, env, skip=4):
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):

        # np.save("FIRST_FRAME.npy",self.env.render('rgb_array'))
        # if self.episode_steps > self.max_frames - 1000:
        #     print(self.episode_steps )

        total_reward = 0.0
        done = None
        for i in range(self._skip):

            obs, reward, done, info = self.env.step(action)
            # np.save("SECOND_FRAME.npy",self.env.render('rgb_array'))
            # exit()
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


class AtariSkips(gym.Wrapper):
    def __init__(self, env, max_frames=int(108e3)):
        gym.Wrapper.__init__(self, env)
        self.env = env
        self.episode_steps = 0
        self.max_frames = max_frames

    def reset(self):
        ob = self.env.reset()
        self.episode_steps = 0

        for _ in range(random.randrange(30)):
            ob, reward, done, info = self.env.step(0)
            self.episode_steps+=1

            if done:
                ob = self.env.reset()

        return ob

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.episode_steps+=1

        #Should we add noop after death?
        return ob, reward, done or self.episode_steps > self.max_frames, info







   


class FrameStack(gym.Wrapper):
    def __init__(self, env, k, device, cast=torch.float32, scale=True):
        """Stack k last frames.
        cast : torch dtype to cast to. If None, no cast
        scale : bool. If True, divides by 255 (scaling to float). cast must be torch.float
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.cast = cast
        self.device = device
        self.scale = scale
        if self.scale:
            assert cast == torch.float32 or cast == torch.float64, f"Cast must be torch.float, found {self.cast}"
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0,
                                                high=255,
                                                shape=((k,) + shp),
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
        # ob = torch.as_tensor(np.stack(list(self.frames), axis=0), device=self.device)
        # if self.cast is not None:
        #     ob = ob.to(dtype=self.cast)
        # if self.scale:
        #     ob = ob.div_(255)
        ob = np.stack(list(self.frames), axis=0)
        return ob

class LazyFrames(object):
    """
    Description:
        This object ensures that common frames between the observations are only stored once.  It
        exists purely to optimize memory usage which can be huge for DQN's 1M frames replay buffers.
        This object should only be converted to numpy array before being passed to the model.
    
    Usage:
        Wrap frames with this object. 

    Notes:
        - Can be finicky if used without the OpenAI ReplayBuffer
    """
    def __init__(self, frames):
        self._frames = frames

    def _force(self):
        return np.stack(self._frames, axis=0)

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, i):
        return self._frames[i]


class AtariPreprocessPixelInput():
    def __init__(self, shape=(84, 84)):  #Do we still want to do this?
        self.shape = shape
        self.transforms = T.Compose([
            T.ToPILImage(mode='YCbCr'),
            T.Lambda(lambda img: img.split()[0]),
            T.Resize(self.shape),
            T.Lambda(lambda img: np.array(img)),
        ])
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=self.shape,
            dtype=np.uint8,
        )
    # def transforms(self,state):
    #     rgb_weights = [0.2989, 0.5870, 0.1140]
    #     grayscale_image = np.dot(state[...,:3], rgb_weights)
    #     state = cv2.resize(state, (84, 84), interpolation=cv2.INTER_LINEAR)
    #     return state #torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

    def get_state(self, rendered_pixel):
        return self.transforms(rendered_pixel)


class CombineRamPixel(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        # self.env = env
        # print(self.env.env.__dict__)
        # exit()
        # self.env.reset()
        # self.env.render("rgb_array")

        # get_pixel_name = env.unwrapped.spec.id
        # self.pixel_env = gym.make(get_pixel_name.replace('-ram',''))
        # print("Found atari game:",self.pixel_env.unwrapped.spec.id)

        self.pixel_wrap = AtariPreprocessPixelInput()

        self.pixel_shape = self.pixel_wrap.observation_space.shape
        self.ram_shape = self.observation_space.shape
        new_total_shape = (self.ram_shape[0] + self.pixel_shape[0] * self.pixel_shape[1],)

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=new_total_shape,
            dtype=np.uint8,
        )



    def combine_states(self, ram_state, pixel_state):
        # for x in range(len(pixel_state)):
        #     print(pixel_state[x])

        return np.concatenate((ram_state, np.reshape(pixel_state, -1)))

    def observation(self, obs):
        pixel_state = self.pixel_wrap.get_state(self.render(mode='rgb_array'))
        return self.combine_states(obs, pixel_state)
