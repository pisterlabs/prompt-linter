import gym
import universe
import numpy as np

from ml.reinforcement.runner.environ.gym_ml import OpenAIGymMLEnvRunner


class OpenAIAtariGymMLEnvRunner(OpenAIGymMLEnvRunner):
    def initialize_environment(self):
        env = gym.make(self.env_name)
        #env.configure(remotes=1)  # creates a local docker container
        return env

    def reset_environment(self):
        observation = self.env.reset()
        return observation

    def step(self, action):
        self.env.render()
        observation, reward, done, info = self.env.step([action])

        while not self.env_has_started(info, observation):
            observation, reward, done, info = self.env.step([action])

        observation = self.format_observation(observation)
        reward = self.format_reward(reward)
        done = self.format_done(done)
        info = self.format_info(info)

        return observation, reward, done, info

    def format_observation(self, observation):
        observation = observation[0]
        observation = observation['vision']
        return np.asarray(observation, dtype='float32')

    def format_reward(self, reward):
        return reward[0]

    def format_done(self, done):
        return done[0]

    def format_info(self, info):
        return info

    def env_has_started(self, info, observation):
        if observation is None:
            return False

        if observation[0] is None:
            return False

        try:
            info_n = info['n']
            info = info_n[0]
            env_state = info.get('env_status.env_state', False)
            return env_state == 'running'
        except KeyError:
            return False



