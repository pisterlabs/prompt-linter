from BaseEnvironment import BaseEnvironment

import cv2
import gym
import numpy as np

class AtariEnvironment(BaseEnvironment):
    """Wrapper class for the OpenAI Gym Atari Environment."""

    def __init__(self, render=False, **kwargs):
        self.environment = gym.make('Pong-v0')
        args = kwargs.get('args')
        self.width = 84
        self.height = 84
        self.possible_actions = self.environment.action_space
        if (type(self.possible_actions) != type(gym.spaces.discrete.Discrete(0))):
            raise AssertionError("The sample action space does not consist of Discrete actions.")
        self.previous_observation = None
        self.recent_observation = None
        self.total_score = 0
        self.num_games = 0
        self.should_render = args.test
        self.done = False
        self.reset()

    def getReward(self):
        return self.recent_reward

    def getObservation(self):
        return self.recent_observation

    def getActionPerformed(self):
        return self.recent_action

    def resetStatistics(self):
        self.total_score = 0
        self.num_games = 0
        self.reset()

    def getStatistics(self):
        return self.total_score, self.num_games

    def performAction(self, action):
        self.recent_action = action
        observation, reward, done, _ = self.environment.step(self.recent_action)
        self.done = done
        self.recent_observation = self._preprocess_observation_(observation)
        self.recent_reward = reward
        self.total_score += self.recent_reward
        if self.should_render:
            self.environment.render()
        return True

    def isTerminalState(self):
        return self.done

    def getPossibleActions(self):
        return self.possible_actions.n

    def sampleRandomAction(self):
        return self.possible_actions.sample()

    def _preprocess_observation_(self, observation):
        """This method is to preprocess observation images.

        The RGB images from OpenAI are converted CMYK images and the luminance (Y)
        channel is extracted, downsampled to a width and height of 84x84 using
        Anti-aliasing, and returned.
        """
        return cv2.resize(cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY), (self.width, self.height))

    def reset(self):
        self.recent_observation = self._preprocess_observation_(self.environment.reset())
        self.num_games += 1
        self.done = False
        if self.should_render:
            self.environment.render()
