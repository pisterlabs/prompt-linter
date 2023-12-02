"""
Adopted from OpenAI gym
https://github.com/openai/gym/blob/master/gym/envs/robotics/robot_env.py
"""
import numpy as np

import gym
from gym import error, spaces
from gym.utils import seeding

class RobotBaseEnv(gym.GoalEnv):
    def __init__(self, n_actions, discrete_actions, n_substeps):
        self.seed()
        self.goal = self._sample_goal()
        self.discrete_actions = discrete_actions
        self.action_space = self._set_action_space(n_actions)
        self.observation_space = self._set_observation_space()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        if self.discrete_actions:
            pass
        else:
            action = np.clip(action, self.action_space.low, self.action_space.high)
        self._apply_action(action)
        obs = self._get_obs()
        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal)
            }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        return obs, reward, done, info

    def reset(self):
        """Resets the environment
        """
        raise NotImplementedError()

    def _get_obs(self):
        """Returns the observation.
        """
        raise NotImplementedError()

    def _set_observation_space(self):
        """Returns the observation space
        """
        raise NotImplementedError()

    def _set_action_space(self):
        """Returns the action space
        """
        raise NotImplementedError()

    def _apply_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        raise NotImplementedError()

    def _sample_goal(self):
        """Samples a new goal and returns it.
        """
        raise NotImplementedError()

    def _env_setup(self, sim, arm, initial_pose):
        """Initial configuration of the environment. Can be used to choose configure initial state,
        choose robot arm, choose simulation, load objects, and extract information from the simulation.
        """
        return

    def _viewer_setup(self):
        """Initial configuration of the viewer. Can be used to set the camera position,
        for example.
        """
        pass

    def _render_callback(self):
        """A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        pass

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass
