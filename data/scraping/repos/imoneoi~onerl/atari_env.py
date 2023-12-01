# Borrow a lot from openai baselines:
# https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
# NGRL-Specific, modify resolution

import cv2
import gym
import numpy as np


class NoopResetEnv(gym.Wrapper):
    """Sample initial states by taking random number of no-ops on reset.
    No-op is assumed to be action 0.

    :param gym.Env env: the environment to wrap.
    :param int noop_max: the maximum value of no-ops to run.
    """

    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self):
        self.env.reset()
        noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset()
        return obs


class NoRewardDoneEnv(gym.Wrapper):
    """Resets when no reward for a long time, avoiding emulator glitches"""

    def __init__(self, env, tolerance=None):
        super().__init__(env)
        self._tolerance = tolerance
        self._no_reward_time = 0

    def reset(self):
        self._no_reward_time = 0
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if np.isclose(reward, 0):
            self._no_reward_time += 1
            if (self._tolerance is not None) and (self._no_reward_time >= self._tolerance):
                done = True
                print("no rew done!")
        else:
            self._no_reward_time = 0

        return obs, reward, done, info


class EpisodicLifeEnv(gym.Wrapper):
    """Make end-of-life == end-of-episode, but only reset on true game over. It
    helps the value estimation.

    :param gym.Env env: the environment to wrap.
    """

    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal, then update lives to
        # handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            # for Qbert sometimes we stay in lives == 0 condition for a few
            # frames, so its important to keep lives > 0, so that we only reset
            # once the environment is actually done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self):
        """Calls the Gym environment reset, only when lives are exhausted. This
        way all states are still reachable even though lives are episodic, and
        the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset()
        else:
            # no-op step to advance from terminal/lost life state
            obs = self.env.step(0)[0]
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class FireResetEnv(gym.Wrapper):
    """Take action on reset for environments that are fixed until firing.
    Related discussion: https://github.com/openai/baselines/issues/240

    :param gym.Env env: the environment to wrap.
    """

    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self):
        self.env.reset()
        return self.env.step(1)[0]


class WarpFrame(gym.ObservationWrapper):
    """Warp frames to 224x160

    :param gym.Env env: the environment to wrap.
    """

    def __init__(self, env):
        super().__init__(env)
        self.size = 88, 88
        self.observation_space = gym.spaces.Box(
            low=np.min(env.observation_space.low),
            high=np.max(env.observation_space.high),
            shape=(1, *self.size), dtype=env.observation_space.dtype)

    def observation(self, frame):
        """returns the current observation from a frame"""
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return np.expand_dims(cv2.resize(frame, self.size[::-1]), 0)  # OpenCV requires WxH shape


class ClipRewardEnv(gym.RewardWrapper):
    """clips the reward to {+1, 0, -1} by its sign.

    :param gym.Env env: the environment to wrap.
    """

    def __init__(self, env):
        super().__init__(env)
        self.reward_range = (-1, 1)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign. Note: np.sign(0) == 0."""
        return np.sign(reward)


def create_atari_env(
        name: str,
        episode_life: bool = True,
        clip_rewards: bool = True,
        warp_frame: bool = True,
        fire_start: bool = True,
        no_reward_max_frames: int = 5 * 60 * (60 // 4)  # 5 minutes
):
    env = gym.make("{}Deterministic-v4".format(name))
    env = NoopResetEnv(env, noop_max=1 * (60 // 4))

    if no_reward_max_frames is not None:
        env = NoRewardDoneEnv(env, no_reward_max_frames)
    if episode_life:
        env = EpisodicLifeEnv(env)
    if fire_start and ('FIRE' in env.unwrapped.get_action_meanings()):
        env = FireResetEnv(env)
    if warp_frame:
        env = WarpFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)

    return env
