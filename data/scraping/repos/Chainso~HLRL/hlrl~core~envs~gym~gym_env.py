import gymnasium as gym

from hlrl.core.envs.env import Env
from hlrl.core.envs.gym.wrappers import ShapedActionWrapper

class GymEnv(Env):
    """
    A environment from OpenAI Gym
    """
    def __init__(self, env: gym.Env):
        """
        Creates the given environment from OpenAI Gym

        Args:
            env: The gym environment to wrap.
        """
        Env.__init__(self)

        self.env = ShapedActionWrapper(env)

        self.state_space = self.env.observation_space.shape
        self.action_space = self.env.action_space.shape

    def step(self, action):
        """
        Takes 1 step into the environment using the given action.

        Args:
            action (object): The action to take in the environment.
        """
        (self.state, self.reward, self.terminal, self.truncated,
            self.info) = self.env.step(action)

        return self.state, self.reward, self.terminal, self.truncated, self.info

    def sample_action(self):
        return self.env.action_space.sample()

    def render(self):
        """
        Renders the gym environment.
        """
        self.env.render()

    def reset(self):
        """
        Resets the environment.
        """
        self.state, self.info = self.env.reset()
        self.reward = 0

        self.terminal = False
        self.truncated = False

        return self.state

