"""
Useful wrappers taken from OpenAI (https://github.com/openai/baselines)
"""

import numpy as np
import collections
from collections import deque, defaultdict, MutableMapping
import copy
import gym
from gym import spaces
from gym import ObservationWrapper
import cv2
import minihack


cv2.ocl.setUseOpenCL(False)

STATE_KEY = "state"

class PixelObservationWrapper(ObservationWrapper):
    """Augment observations by pixel values."""

    def __init__(
        self, env, pixels_only=True, render_kwargs=None, pixel_keys=("pixels",)
    ):
        """Initializes a new pixel Wrapper.
        Args:
            env: The environment to wrap.
            pixels_only: If `True` (default), the original observation returned
                by the wrapped environment will be discarded, and a dictionary
                observation will only include pixels. If `False`, the
                observation dictionary will contain both the original
                observations and the pixel observations.
            render_kwargs: Optional `dict` containing keyword arguments passed
                to the `self.render` method.
            pixel_keys: Optional custom string specifying the pixel
                observation's key in the `OrderedDict` of observations.
                Defaults to 'pixels'.
        Raises:
            ValueError: If `env`'s observation spec is not compatible with the
                wrapper. Supported formats are a single array, or a dict of
                arrays.
            ValueError: If `env`'s observation already contains any of the
                specified `pixel_keys`.
        """

        super(PixelObservationWrapper, self).__init__(env)

        if render_kwargs is None:
            render_kwargs = {}

        for key in pixel_keys:
            render_kwargs.setdefault(key, {})

            render_mode = render_kwargs[key].pop("mode", "rgb_array")
            assert render_mode == "rgb_array", render_mode
            render_kwargs[key]["mode"] = "rgb_array"

        wrapped_observation_space = env.observation_space

        if isinstance(wrapped_observation_space, spaces.Box):
            self._observation_is_dict = False
            invalid_keys = set([STATE_KEY])
        elif isinstance(wrapped_observation_space, (spaces.Dict, MutableMapping)):
            self._observation_is_dict = True
            invalid_keys = set(wrapped_observation_space.spaces.keys())
        else:
            raise ValueError("Unsupported observation space structure.")

        if not pixels_only:
            # Make sure that now keys in the `pixel_keys` overlap with
            # `observation_keys`
            overlapping_keys = set(pixel_keys) & set(invalid_keys)
            if overlapping_keys:
                raise ValueError(
                    "Duplicate or reserved pixel keys {!r}.".format(overlapping_keys)
                )

        if pixels_only:
            self.observation_space = spaces.Dict()
        elif self._observation_is_dict:
            self.observation_space = copy.deepcopy(wrapped_observation_space)
        else:
            self.observation_space = spaces.Dict()
            self.observation_space.spaces[STATE_KEY] = wrapped_observation_space

        # Extend observation space with pixels.

        pixels_spaces = {}
        for pixel_key in pixel_keys:
            pixels = self.env.render(**render_kwargs[pixel_key])

            if np.issubdtype(pixels.dtype, np.integer):
                low, high = (0, 255)
            elif np.issubdtype(pixels.dtype, np.float):
                low, high = (-float("inf"), float("inf"))
            else:
                raise TypeError(pixels.dtype)

            pixels_space = spaces.Box(
                shape=pixels.shape, low=low, high=high, dtype=pixels.dtype
            )
            pixels_spaces[pixel_key] = pixels_space

        self.observation_space.spaces.update(pixels_spaces)

        self._env = env
        self._pixels_only = pixels_only
        self._render_kwargs = render_kwargs
        self._pixel_keys = pixel_keys

    def observation(self, observation):
        pixel_observation = self._add_pixel_observation(observation)
        return pixel_observation

    def _add_pixel_observation(self, wrapped_observation):
        if self._pixels_only:
            observation = collections.OrderedDict()
        elif self._observation_is_dict:
            observation = type(wrapped_observation)(wrapped_observation)
        else:
            observation = collections.OrderedDict()
            observation[STATE_KEY] = wrapped_observation

        pixel_observations = {
            pixel_key: self.env.render(**self._render_kwargs[pixel_key])
            for pixel_key in self._pixel_keys
        }

        observation.update(pixel_observations)

        return observation

class FlattenObservation(ObservationWrapper):
    r"""Observation wrapper that flattens the observation."""

    def __init__(self, env):
        super(FlattenObservation, self).__init__(env)
        self.observation_space = spaces.flatten_space(env.observation_space)

    def observation(self, observation):
        return spaces.flatten(self.env.observation_space, observation)

        
class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 17.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 17
        # No action in minihack-quest-hard is 17. 
        # Use: env.unwrapped.print_action_meanings() to see all action usages
        #assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = np.random.randint(
                1, self.noop_max + 1
            )  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        # lives = self.env.unwrapped.ale.lives()
        lives = 0
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        # self.lives = self.env.unwrapped.ale.lives()
        self.lives = 0
        return obs

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space["pixel"].shape, dtype=np.uint8)
        self._skip = skip

    def reset(self):
        return self.env.reset()

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

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work.
        Expects inputs to be of shape height x width x num_channels
        """
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8
        )

    def observation(self, frame):
        frame = frame["pixel"]

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self.width, self.height), interpolation=cv2.INTER_AREA
        )
        return frame[:, :, None]


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
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(shp[0] * k, shp[1], shp[2]), dtype=np.uint8
        )

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


class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0


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


class PyTorchFrame(gym.ObservationWrapper):
    """Image shape to num_channels x height x width"""

    def __init__(self, env):
        super(PyTorchFrame, self).__init__(env)
        shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(shape[-1], shape[0], shape[1]), dtype=np.uint8
        )

    def observation(self, observation):
        return np.rollaxis(observation, 2)

###########################################################################
#######################MINI-HACK WRAPPERS##################################
# Additional wrappers obtained from:
# https://github.com/facebookresearch/minihack/blob/main/minihack/agent/common/envs/wrapper.py

class CounterWrapper(gym.Wrapper):
    def __init__(self, env, state_counter="none"):
        # intialize state counter
        self.state_counter = state_counter
        if self.state_counter != "none":
            self.state_count_dict = defaultdict(int)
        # this super() goes to the parent of the particular task, not to object
        super().__init__(env)

    def step(self, action):
        # add state counting to step function if desired
        step_return = self.env.step(action)
        if self.state_counter == "none":
            # do nothing
            return step_return

        obs, reward, done, info = step_return

        if self.state_counter == "ones":
            # treat every state as unique
            state_visits = 1
        elif self.state_counter == "coordinates":
            # use the location of the agent in the dungeon to accumulate visits
            features = obs["blstats"]
            x = features[0]
            y = features[1]
            d = features[12]
            coord = (d, x, y)
            self.state_count_dict[coord] += 1
            state_visits = self.state_count_dict[coord]
        else:
            raise NotImplementedError("state_counter=%s" % self.state_counter)

        obs.update(state_visits=np.array([state_visits]))

        if done:
            self.state_count_dict.clear()

        return step_return

    def reset(self, wizkit_items=None):
        # reset state counter when env resets
        obs = self.env.reset(wizkit_items=wizkit_items)
        if self.state_counter != "none":
            self.state_count_dict.clear()
            # current state counts as one visit
            obs.update(state_visits=np.array([1]))
        return obs

class CropWrapper(gym.Wrapper):
    def __init__(self, env, h=9, w=9, pad=0, keys=("tty_chars", "tty_colors")):
        super().__init__(env)
        self.env = env
        self.h = h
        self.w = w
        self.pad = pad
        self.keys = keys
        assert self.h % 2 == 1
        assert self.w % 2 == 1
        self.last_observation = None
        self._actions = self.env._actions

    def render(self, mode="human", crop=True):
        self.env.render()
        obs = self.last_observation
        tty_chars_crop = obs["tty_chars_crop"]
        tty_colors_crop = obs["tty_colors_crop"]
        rendering = self.env.get_tty_rendering(
            tty_chars_crop, tty_colors_crop, print_guides=True
        )
        print(rendering)

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)

        dh = self.h // 2
        dw = self.w // 2

        (y, x) = next_state["tty_cursor"]
        x += dw
        y += dh

        for key in self.keys:
            obs = next_state[key]
            obs = np.pad(
                obs,
                pad_width=(dw, dh),
                mode="constant",
                constant_values=self.pad,
            )
            next_state[key + "_crop"] = obs[
                y - dh : y + dh + 1, x - dw : x + dw + 1
            ]

        self.last_observation = next_state

        return next_state, reward, done, info

    def reset(self, wizkit_items=None):
        obs = self.env.reset(wizkit_items=wizkit_items)
        obs["tty_chars_crop"] = np.zeros((self.h, self.w), dtype=np.uint8)
        obs["tty_colors_crop"] = np.zeros((self.h, self.w), dtype=np.int8)
        self.last_observation = obs
        return obs


class PrevWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.last_observation = None
        self._actions = self.env._actions

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        next_state["prev_reward"] = np.array([reward], dtype=np.float32)
        next_state["prev_action"] = np.array([action], dtype=np.uint8)

        self.last_observation = next_state

        return next_state, reward, done, info

    def reset(self, wizkit_items=None):
        obs = self.env.reset(wizkit_items=wizkit_items)
        obs["prev_reward"] = np.zeros(1, dtype=np.float32)
        obs["prev_action"] = np.zeros(1, dtype=np.uint8)
        self.last_observation = obs
        return obs

###########END OF MINIHACK WRAPPERS#############################


####RENDERING wrapper provided by simon42 on discord############
class RenderRGB(gym.Wrapper):
    def __init__(self, env, key_name="pixel"):
        super().__init__(env)
        self.last_pixels = None
        self.viewer = None
        self.key_name = key_name

        render_modes = env.metadata['render.modes']
        render_modes.append("rgb_array")
        env.metadata['render.modes'] = render_modes

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.last_pixels = obs[self.key_name]
        return obs, reward, done, info

    def render(self, mode="human", **kwargs):
        img = self.last_pixels

        # Hacky but works
        if mode != "human":
            return img
        else:
            from gym.envs.classic_control import rendering

            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def reset(self):
        obs = self.env.reset()
        self.last_pixels = obs[self.key_name]
        return obs

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
            