# Normalize the environment (scale actions/rewards/observations)
# Most of the code taken from openai's rllab repository

import numpy as np
from gym.spaces import Discrete, Box, Tuple
from gym import Env


class NormalizedEnv(Env):

    def __init__(self, env, scale_reward=1.0, normalize_obs=False, normalize_reward=False, obs_alpha=0.001, reward_alpha=0.001):
        self._env = env
        self.spec = self._env.spec
        self.spec.reward_threshold = self.spec.reward_threshold or float('inf')
        self._scale_reward = scale_reward
        self._normalize_obs = normalize_obs
        self._normalize_reward = normalize_reward
        self._obs_alpha = obs_alpha
        self._obs_mean = np.zeros(self._env.observation_space.shape)
        self._obs_var = np.ones(self._env.observation_space.shape)
        self._reward_alpha = reward_alpha
        self._reward_mean = 0.0
        self._reward_var = 1.0

    def _update_obs_estimate(self, obs):
        self._obs_mean = (1 - self._obs_alpha) * \
            self._obs_mean + self._obs_alpha * obs
        self._obs_var = (1 - self._obs_alpha) * self._obs_var + \
            self._obs_alpha * np.square(obs - self._obs_mean)

    def _update_reward_estimate(self, reward):
        self._reward_mean = (1 - self._reward_alpha) * \
            self._reward_mean + self._reward_alpha * reward
        self._reward_var = (1 - self._reward_alpha) * self._reward_var + \
            self._reward_alpha * np.square(reward - self._reward_mean)

    def _apply_normalize_obs(self, obs):
        self._update_obs_estimate(obs)
        return (obs - self._obs_mean) / (np.sqrt(self._obs_var) + 1e-8)

    def _apply_normalize_reward(self, reward):
        self._update_reward_estimate(reward)
        return reward / (np.sqrt(self._reward_var) + 1e-8)

    @property
    def action_space(self):
        if isinstance(self._env.action_space, Box):
            ub = np.ones(self._env.action_space.shape)
            return Box(-1 * ub, ub)
        return self._env.action_space

    @property
    def observation_space(self):
        return self._env.observation_space

    def reset(self, **kwargs):
        observation = self._env.reset(**kwargs)
        if self._normalize_obs:
            return self._apply_normalize_obs(observation)
        else:
            return observation

    def step(self, action):
        if isinstance(self._env.action_space, Box):
            # rescale the action
            lb = self._env.action_space.low
            ub = self._env.action_space.high
            scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
            scaled_action = np.clip(scaled_action, lb, ub)
        else:
            scaled_action = action

        wrapped_step = self._env.step(scaled_action)
        next_obs, reward, done, info = wrapped_step

        if self._normalize_obs:
            next_obs = self._apply_normalize_obs(next_obs)
        if self._normalize_reward:
            reward = self._apply_normalize_reward(reward)

        return next_obs, reward, done, info

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def __getattr__(self, field):
        """
        proxy everything to underlying env
        """
        if hasattr(self._env, field):
            return getattr(self._env, field)
        raise AttributeError(field)

    def __repr__(self):
        if "object at" not in str(self._env):
            env_name = str(env._env)
        else:
            env_name = self._env.__class__.__name__
        return env_name
