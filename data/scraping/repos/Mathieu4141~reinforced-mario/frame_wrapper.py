"""
This implementation is inspired from OpenAI
https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
"""
import cv2
import gym
import numpy as np


class FrameWrapper(gym.ObservationWrapper):
    def __init__(self, env, width: int, height: int, grayscale: bool, interpolation: int):
        """
        Warp frames as done in the Nature paper and later work.
        observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._interpolation = interpolation

        new_space = gym.spaces.Box(low=0, high=255, shape=(self._height, self._width, 1), dtype=np.uint8,)
        original_space = self.observation_space
        self.observation_space = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, frame):
        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self._width, self._height), interpolation=self._interpolation)
        if len(frame.shape) < 3:
            frame = np.expand_dims(frame, -1)

        return frame
