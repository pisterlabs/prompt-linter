"""
Useful wrappers taken from OpenAI (https://github.com/openai/baselines)
"""

# import numpy as np
# from collections import deque
import gym
# from gym import spaces
# import cv2

# cv2.ocl.setUseOpenCL(False)
        
# Gym wrapper for rendering
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
