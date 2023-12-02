from collections import deque
from typing import Optional, Union, List, Callable, Sized, Iterable
from gym.core import RenderFrame
from gym.vector.utils import spaces
from gym.wrappers import TimeLimit
import numpy as np
import cv2
import gym
from baselines.common.atari_wrappers import EpisodicLifeEnv, FireResetEnv, ClipRewardEnv
from gym.vector.utils import batch_space


########################################################################################################################
#                                     MODIFIED ATARI BASELINE WRAPPERS                                                 #
########################################################################################################################
# All these wrappers are taken from OpenAI's Gym Baseline library. We copy-pasted them instead of simply importing
# because we needed to do some small modifications, for example to allow to use a seed when resetting the environment,
# feature that we use for reproducibility

class MaxAndSkipEnvCustom(gym.Wrapper):
    """
    Class that defines a wrapper around gym environments allowing to skip frames and get an observation as the
    pixel-wise maximum between the two most recent skipped frames.

    This differs from the Baselines version because we removed a double definition of the reset method
    """

    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class FrameStackCustom(gym.Wrapper):
    """
    Class that defines a wrapper around gym environments that allows to stack multiple frames into a single observation

    This differs from baselines because we changed the definition of the observation space by stacking on the first
    dimension instead on the last; we also added kwargs to the reset method to allow seed usage when resetting
    """

    def __init__(self, env, k):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=((shp[0] * k,) + shp[1:]),
                                                dtype=env.observation_space.dtype)

    def reset(self, **kwargs):
        ob = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFramesCustom(list(self.frames), axis=0)


class LazyFramesCustom(object):
    """
    Class that defines a frame stacking. This differs from the baseline one because we allowed to define on which
    dimension we want to stack the frames, since we're using the first dimension as the dimension of stacking instead
    of the last
    """

    def __init__(self, frames, axis=-1):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None
        self._axis = axis

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=self._axis)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]


class NoopResetEnvCustom(gym.Wrapper):
    """
    Class that defines a wrapper around gym environments that allows to perform a noop action n times at reset

    This differs from the baseline one because we replaced randint that was not available in our python version with
    integers method, they both do the same thing
    """

    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class WarpFrameCustom(gym.ObservationWrapper):
    """
    Class that defines the wrapper around gym environment that allows to change the observation space dimensionality

    This differs from the baselines version because we allowed to define the actual resize dimension of the observation
    space and we also allowed to select if we want to use grayscale observations or not.
    """

    def __init__(self, env, width: int = 84, height: int = 84, grayscale: bool = True):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = width
        self.height = height
        self.grayscale = grayscale
        self.channels = 1 if self.grayscale else 3
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.height, self.width, self.channels), dtype=np.uint8)

    def observation(self, frame):
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class ScaledFloatFrameCustom(gym.ObservationWrapper):
    """
    Class that defines a wrapper around gym environments that allows to scale observation pixel values from 0-255 range
    to 0-1 range

    This differs from the baseline version because we also added the definition of the observation space after the
    scaling
    """

    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0


########################################################################################################################
#                                                CUSTOM WRAPPERS                                                       #
########################################################################################################################


class ImageTransposeWrapper(gym.ObservationWrapper):
    """
    Class that defines an observation wrapper around gym environment that allows to transpose the obsservations. In fact
    we use observations in which the stacked frames are moved from the last dimension to the first dimension, in order
    to match PyTorch's style when feeding batches to the network.
    """

    def __init__(self, env: gym.Env) -> None:
        """
        Constructor method of the wrapper

        :param env: the environment to wrap around (gym.Env)

        :return: None
        """
        super().__init__(env)
        obs_shape = (self.env.observation_space.shape[2],
                     self.env.observation_space.shape[0],
                     self.env.observation_space.shape[1])
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs_shape,
                                                dtype=self.env.observation_space.dtype)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Method that returns the transposed observation

        :param observation: observation to transpose (np.ndarray)

        :return: the transposed observation (np.ndarray)
        """

        return self.transpose(observation)

    @staticmethod
    def transpose(observation: np.ndarray) -> np.ndarray:
        """
        Static method that transposes the given input observation numpy array

        :param observation: observation to transpose (np.ndarray)

        :return: the transposed observation (np.ndarray)
        """
        return np.transpose(observation, (2, 0, 1))


class ReproducibleEnv(gym.Wrapper):
    """
    Class that defines a wrapper around gym environments that resets the environment with the same seed everytime.
    """

    def __init__(self, env: gym.Env, seed: int) -> None:
        """
        Constructor method of the wrapper

        :param env: the environment to wrap around (gym.Env)
        :param seed: the seed to use for the reset (int)

        :return: None
        """
        gym.Wrapper.__init__(self, env)
        self.seed = seed

    def reset(self, **kwargs) -> np.ndarray:
        """
        Method that resets the environment accordingly to the given seed

        :param kwargs: additional arguments to pass to the environment reset method

        :return: the observation given by the reset method (np.ndarray)
        """
        self.env.action_space.seed(seed=self.seed)
        ob = self.env.reset(seed=self.seed, **kwargs)

        return ob


class VectorEnv(gym.Env):
    """
    Class that defines a gym environment that embeds vectorized environment and steps through each environment in the
    stack
    """

    def __init__(self, environment_maker: Callable, num_envs: int) -> None:
        """
        Constructor method of the environment

        :param environment_maker: callable function that we use to make a single environment of the stack (Callable)
        :param num_envs: the number of environments in the vector (int)

        :return: None
        """
        self.envs = [environment_maker() for _ in range(num_envs)]
        self.num_envs = num_envs

        self.observation_space = batch_space(self.envs[0].observation_space, n=num_envs)
        self.action_space = batch_space(self.envs[0].action_space, n=num_envs)

    def seed(self, seed: int = None) -> list:
        """
        Method that set a seed for each of the stacked environments given an input seed

        :param seed: the starting seed to use, default value is None, indicating to use no seed (int)

        :return: the stacked environments initialized with the given seeds (list)
        """
        seeds = [seed + i for i in range(self.num_envs)]
        return [env.seed(s) for env, s in zip(self.envs, seeds)]

    def reset(self, **kwargs) -> list:
        """
        Method that resets each environment in the stack

        :param kwargs: parameters to pass to the reset method (dict)

        :return: the observations obtained from the resetting of the stacked environments (list)
        """
        if "seed" in kwargs.keys():
            seed = kwargs["seed"]
            kwargs = [{**kwargs, 'seed': seed + i} for i in range(self.num_envs)]
        else:
            kwargs = [kwargs for _ in range(self.num_envs)]

        assert len(kwargs) == len(self.envs)

        return [env.reset(**args) for env, args in zip(self.envs, kwargs)]

    def step(self, actions: list) -> tuple:
        """
        Method that performs each given action to the corresponding stacked environment

        :param actions: list of actions for each environment (list)

        :return: tuple contained the obtained observation, reward, done and info for each environment (tuple)
        """

        assert len(self.envs) == len(actions)
        observations = []
        rewards = []
        dones = []
        infos = []
        for env, a in zip(self.envs, actions):
            observation, reward, done, info = env.step(a)
            if done:
                observation = env.reset()

            observations.append(observation)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        return observations, rewards, dones, infos

    def close(self) -> None:
        """
        Method that closes each environment in the stacK

        :return: None
        """

        for env in self.envs:
            env.close()

    def render(self, mode="human") -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        """
        Method that renders the current step
        """
        pass


def deepmind_atari_wrappers(env: gym.Env,
                            max_episode_steps: int,
                            noop_max: int,
                            frame_skip: int,
                            episode_life: bool,
                            clip_rewards: bool,
                            frame_stack: int,
                            scale: bool,
                            patch_size: int,
                            grayscale: bool,
                            fire_reset: bool) -> gym.Env:
    """
    Function that we use to create an atari environment with all the atari wrappers

    :param env: the environment to wrap (gym.Env)
    :param max_episode_steps: the maximum number of steps per environment, default is None, namely no limit to the
        number of environment steps (int)
    :param noop_max: the maximum number of noop actions to perform when resetting the environment, default is 30 (int)
    :param frame_skip: the number of frames to skip at every step, default is 4 (int)
    :param episode_life: boolean defining if to use life as an episode or not, default is True (bool)
    :param clip_rewards: boolean defining if to clip rewards or not, default is True (bool)
    :param frame_stack: the number of frames to stack at each step, default is 4 (int)
    :param scale: boolean defining if to scale the pixels of the observation or not, default is True (bool)
    :param patch_size: the size of the observation after resizing in pixels, default is 84 pixels (int)
    :param grayscale: boolean defining if to use grayscale observations or not, default is True (bool)
    :param fire_reset: boolean defining if to perform FIRE action at reset or not, default is True (bool)

    :return: the wrapped environment (gym.Env)
    """
    if noop_max > 0:
        env = NoopResetEnvCustom(env, noop_max=noop_max)
    if frame_skip > 0:
        env = MaxAndSkipEnvCustom(env, skip=frame_skip)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings() and fire_reset:
        env = FireResetEnv(env)
    if patch_size is not None:
        env = WarpFrameCustom(env, width=patch_size, height=patch_size, grayscale=grayscale)
    if scale:
        env = ScaledFloatFrameCustom(env)
    if clip_rewards:
        env = ClipRewardEnv(env)

    env = ImageTransposeWrapper(env)

    if frame_stack > 0:
        env = FrameStackCustom(env, frame_stack)

    return env


def atari_deepmind_env(env_name: str, render_mode: str = None, **kwargs) -> gym.Env:
    """
    Function to create a wrapped Atari environment

    :param env_name: the name of the environment to create (str)
    :param render_mode: the render mode to use for the environment, default is None (str)
    :param kwargs: additional arguments for the atari wrappers (dict)

    :return: wrapped environment (gym.Env)
    """

    # create environment with the atari wrappers
    env = gym.make(env_name, obs_type="rgb", render_mode=render_mode, new_step_api=False)
    env = deepmind_atari_wrappers(env=env, **kwargs)
    return env


def vector_atari_deepmind_env(num_envs: int = 4, **kwargs) -> VectorEnv:
    """
    Function to create vectorized and wrapped atari environments

    :param num_envs: the number of stacked environment, default value is 4 (int)
    :param kwargs: additional parameters for the environments and the wrappers (dict)

    :return: the vectorized environments (VectorEnv)
    """

    def __make_atari_deepmind_env() -> gym.Env:
        """
        Utility function that serves as an environment maker function for the vectorized environment

        :return: the atari environment wrapped with the given parameters (gym.Env)
        """
        return atari_deepmind_env(**kwargs)

    env = VectorEnv(environment_maker=__make_atari_deepmind_env, num_envs=num_envs)

    return env
