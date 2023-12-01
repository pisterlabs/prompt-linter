import numpy as np
import random
import cv2
import gym
from gym.spaces.box import Box


def create_atari_environment(game_name=None):
    env = gym.make(game_name)
    # Strip out the TimeLimit wrapper from Gym, which caps us at 100k frames. We
    # handle this time limit internally instead, which lets us cap at 108k frames
    # (30 minutes). The TimeLimit wrapper also plays poorly with saving and
    # restoring states.
    env = env.env
    assert 'NoFrameskip' in env.spec.id
    env = ALEGrayWrapper(env)
    return env


"""
With respect to FireReset:
    * https://github.com/openai/baselines/issues/240
    * https://github.com/openai/gym/pull/1652
    * https://github.com/openai/gym/issues/1659
    * https://github.com/openai/gym/pull/1661
    * https://github.com/astooke/rlpyt/pull/158
    * https://github.com/mgbellemare/Arcade-Learning-Environment/issues/435
These issues suggested FireReset is unnecessary except for Breakout in which
the ball will not appear until pressing fire. So remove FireReset.
"""


class AtariWrapper(gym.Wrapper):
    """
    Copied from openai/baselines.
        * Max 30 Noop actions.
        * Frame skipping (defaults to 4).
        * Process the last two frames.
        * Downsample the screen (defaults to 84x84).
    """

    def __init__(self, env, frame_skip=4, noop_max=30):
        """Return only every `frame_skip`-th frame"""
        super(AtariWrapper, self).__init__(env)
        # Make sure NOOP = 0, FIRE = 1.
        assert env.get_action_meanings()[0] == 'NOOP'
        assert env.get_action_meanings()[1] == 'FIRE'
        assert len(env.get_action_meanings()) >= 3
        self._noop_action = 0
        self._frame_skip = frame_skip
        self._noop_max = noop_max
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros(
            (2,) + env.observation_space.shape, dtype=np.uint8)
        self._width = 84
        self._height = 84

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        for i in range(self._frame_skip):
            obs, reward, done, info = self.env.step(action)
            # Record the last two frames.
            if i == self._frame_skip - 2:
                self._obs_buffer[0] = obs
            if i == self._frame_skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter

        frame = self._get_processed_obs()

        return frame, total_reward, done, info

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        # Reset recent raw observations.
        self._obs_buffer.fill(0)
        # Noop actions
        noops = random.randint(1, self._noop_max)
        obs = None
        for i in range(noops):
            obs, _, done, _ = self.env.step(self._noop_action)
            # Record the last two frames.
            if i == noops - 2:
                self._obs_buffer[0] = obs
            if i == noops - 1:
                self._obs_buffer[1] = obs
            if done:
                obs = self.env.reset(**kwargs)
        obs = self._get_processed_obs()
        return obs

    def _get_processed_obs(self):
        max_frame = self._obs_buffer.max(axis=0)
        frame = cv2.cvtColor(max_frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA)
        return frame


class ALEGrayWrapper(gym.Wrapper):
    """
    Combine openai/baselines and google/dopamine.
        * Max 30 Noop actions.
        * Frame skipping (defaults to 4).
        * Process the last two frames.
        * Downsample the screen (defaults to 84x84).
        * Directly fetch the grayscale image from the ALE.
        * Negetive reward on life loss.
    """

    def __init__(self, env, frame_skip=4, noop_max=30, punish_on_loss=True):
        """Return only every `frame_skip`-th frame"""
        super(ALEGrayWrapper, self).__init__(env)
        # Make sure NOOP = 0, FIRE = 1.
        assert env.get_action_meanings()[0] == 'NOOP'
        assert env.get_action_meanings()[1] == 'FIRE'
        assert len(env.get_action_meanings()) >= 3
        self._noop_action = 0
        self._frame_skip = frame_skip
        self._noop_max = noop_max
        self._punish_on_loss = punish_on_loss
        # most recent raw observations (for max pooling across time steps)
        obs_dims = env.observation_space.shape
        self._obs_buffer = np.zeros(
            (2,) + (obs_dims[0], obs_dims[1]), dtype=np.uint8)
        self._width = 84
        self._height = 84
        self._lives = 0
        # Unused but necessary.
        self.was_life_loss = False

    def step(self, action):
        """Repeat action, sum reward, and max over last observations.
        We bypass the Gym observation altogether and directly fetch the
        grayscale image from the ALE. This is a little faster.
        """
        total_reward = 0.0
        for i in range(self._frame_skip):
            _, reward, done, info = self.env.step(action)
            # Punish on life loss does help.
            lives = self.env.ale.lives()
            die = (lives < self._lives)
            self._lives = lives
            if self._punish_on_loss and die:
                total_reward -= 1.0
            # Record the last two frames.
            if i == self._frame_skip - 2:
                self.env.ale.getScreenGrayscale(self._obs_buffer[0])
            if i == self._frame_skip - 1:
                self.env.ale.getScreenGrayscale(self._obs_buffer[1])
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter

        frame = self._get_processed_obs()

        return frame, total_reward, done, info

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        # Reset recent raw observations.
        self._obs_buffer.fill(0)
        # Noop actions
        noops = random.randint(1, self._noop_max)
        obs = None
        for i in range(noops):
            _, _, done, _ = self.env.step(self._noop_action)
            # Record the last two frames.
            if i == noops - 2:
                self.env.ale.getScreenGrayscale(self._obs_buffer[0])
            if i == noops - 1:
                self.env.ale.getScreenGrayscale(self._obs_buffer[1])
            if done:
                self.env.reset(**kwargs)
        self._lives = self.env.ale.lives()
        obs = self._get_processed_obs()
        return obs

    def _get_processed_obs(self):
        max_frame = self._obs_buffer.max(axis=0)
        frame = cv2.resize(
            max_frame, (self._width, self._height), interpolation=cv2.INTER_AREA)
        return frame


class EpisodicWrapper(gym.Wrapper):
    """
    Combine openai/baselines and google/dopamine.
        * Max 30 Noop actions.
        * Frame skipping (defaults to 4).
        * Process the last two frames.
        * Downsample the screen (defaults to 84x84).
        * Terminal on life loss.
    """

    def __init__(self, env, frame_skip=4, noop_max=30):
        """Return only every `frame_skip`-th frame"""
        super(EpisodicWrapper, self).__init__(env)
        # Make sure NOOP = 0, FIRE = 1.
        assert env.get_action_meanings()[0] == 'NOOP'
        assert env.get_action_meanings()[1] == 'FIRE'
        assert len(env.get_action_meanings()) >= 3
        self._noop_action = 0
        self._frame_skip = frame_skip
        self._noop_max = noop_max
        # most recent raw observations (for max pooling across time steps)
        obs_dims = env.observation_space.shape
        self._obs_buffer = np.zeros(
            (2,) + (obs_dims[0], obs_dims[1]), dtype=np.uint8)
        self._width = 84
        self._height = 84
        self._lives = 0
        self.was_life_loss = False

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        for i in range(self._frame_skip):
            _, reward, done, info = self.env.step(action)

            lives = self.env.ale.lives()
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            self.was_life_loss = (lives < self._lives and lives > 0)
            self._lives = lives
            # Record the last two frames.
            if i == self._frame_skip - 2:
                self.env.ale.getScreenGrayscale(self._obs_buffer[0])
            if i == self._frame_skip - 1:
                self.env.ale.getScreenGrayscale(self._obs_buffer[1])
            total_reward += reward
            if done or self.was_life_loss:
                break
        # Note that the observation on the done=True frame
        # doesn't matter

        frame = self._get_processed_obs()

        return frame, total_reward, done, info

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        # Reset recent raw observations.
        self._obs_buffer.fill(0)
        if self.was_life_loss:
            # Step out of old state.
            _, _, done, _ = self.env.step(self._noop_action)
            self.env.ale.getScreenGrayscale(self._obs_buffer[1])
        else:
            self.env.reset(**kwargs)
            # Noop actions
            noops = random.randint(1, self._noop_max)
            obs = None
            for i in range(noops):
                _, _, done, _ = self.env.step(self._noop_action)
                # Record the last two frames.
                if i == noops - 2:
                    self.env.ale.getScreenGrayscale(self._obs_buffer[0])
                if i == noops - 1:
                    self.env.ale.getScreenGrayscale(self._obs_buffer[1])
                if done:
                    self.env.reset(**kwargs)
            # Record the initial lives.
            self._lives = self.env.ale.lives()
        obs = self._get_processed_obs()
        return obs

    def _get_processed_obs(self):
        max_frame = self._obs_buffer.max(axis=0)
        frame = cv2.resize(
            max_frame, (self._width, self._height), interpolation=cv2.INTER_AREA)
        return frame
