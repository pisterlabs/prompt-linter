from .common import header, desc

import numpy as np
import random
class ALE(desc):

    def __init__(self, params):
        # Gives common variables to all environments
        super().__init__()

        try:
            import gym
        except:
            print("Failed to import ALE, make sure you have OpenAI gym + ALE installed!")
            
        # Handle Parameters
        env_name = params['env_name']

        # Create GYM instance
        self.env = gym.make(env_name)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        # Store for later
        self.disc = gym.spaces.discrete.Discrete

        # Define header
        #TODO: Check all open ai gym envs to see if action space works the same
        #       Workout num_classes based on action_space type
        if type(self.observation_space) == self.disc:
            self.out = [self.observation_space.n]
        else:
            self.out = list(self.observation_space.shape)

        self.header = header(env_name=env_name, 
                             input_dim=[self.action_space.n],
                             output_dim=self.out, 
                             num_classes=2,
                             info="Simulators gotten from OpenAI Gym",
                             env_min_score = 0.0,
                             env_max_score = 200.0,
                             rl=True)

    def get_header(self):
        return self.header
        
    def render(self, mode=None):
        self.env.render()
    
    def act(self, action):
        # OpenAIGym envs only accept numbers for inputs
        if type(action) == list or type(action) == np.ndarray:
            action = action[0]

        self.observation, self.reward_step, self.done, self.info = self.env.step(action)
        self.reward_total += self.reward_step
        self.steps += 1
        if self.steps >= 2000:
            self.done = True

        # If only a number is given as output    
        # turn to one-hot    
        if type(self.observation_space) == self.disc:
            l = [0] * (self.observation_space.n)
            l[self.observation] = 1
            self.observation = l
        else:
            self.observation = self.observation / 255

        if self.done:
            return self.observation, self.reward_total, self.done, self.info
        else:
            return self.observation, self.reward_step, self.done, self.info

    # Used for cem agent
    def step(self, action):
        return self.act(action)
        
    def reset(self):
        self.steps = 0
        self.observation = None
        self.reward_step = 0
        self.reward_total = 0
        self.done = False
        self.observation = self.env.reset()


        if type(self.observation_space) == self.disc:
            l = [0] * (self.observation_space.n)
            l[self.observation] = 1
            self.observation = l
        else:
            self.observation = self.observation / 255

        return self.observation
        
        
