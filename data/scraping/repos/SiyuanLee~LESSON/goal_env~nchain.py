# copied from openai gym

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class NChainEnv(gym.Env):
    """n-Chain environment
    This game presents moves along a linear chain of states, with two actions:
     0) forward, which moves along the chain but returns no reward
     1) backward, which returns to the beginning and has a small reward
    The end of the chain, however, presents a large reward, and by moving
    'forward' at the end of the chain this large reward can be repeated.
    At each action, there is a small probability that the agent 'slips' and the
    opposite transition is instead taken.
    The observed state is the current state in the chain (0 to n-1).
    This environment is described in section 6.1 of:
    A Bayesian Framework for Reinforcement Learning by Malcolm Strens (2000)
    http://ceit.aut.ac.ir/~shiry/lecture/machine-learning/papers/BRL-2000.pdf
    """
    def __init__(self, n=5, slip=0.2, small=0.001, large=1.0):
        self.n = n
        self.n2 = bin(n-1)
        print("n2", self.n2, len(self.n2)-2)
        self.slip = slip  # probability of 'slipping' an action
        self.small = small  # payout for 'backwards' action
        self.large = large  # payout at end of chain for 'forwards' action
        self.state = 0  # Start at beginning of the chain
        self.action_space = spaces.Box(low=-1., high=1., shape=(1,))
        # self.observation_space = spaces.Discrete(self.n)
        self.observation_space = spaces.Discrete(len(self.n2) - 2)
        self.shuffle_order = np.arange(len(self.n2) - 2)
        np.random.shuffle(self.shuffle_order)
        self.seed()
        target = np.zeros(n)
        target[n-1] = 1
        self.target = target
        self.reward_type = "sparse"
        self.visited_count = np.zeros(n)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # print("action", action)
        success = False
        info = {}
        assert self.action_space.contains(action)
        if self.np_random.rand() < self.slip:
            action = 0 - action  # agent slipped, reverse action taken
        if action < 0 and self.state > 0:  # 'backwards': go back to the beginning, get small reward
            reward = self.small
            self.state -= 1
        elif action > 0 and self.state < self.n - 1:  # 'forwards': go up along the chain
            reward = 0
            self.state += 1
        elif self.state == self.n - 1:  # 'forwards': stay at the end of the chain, collect large reward
            reward = self.large
            success = True
        else:
            reward = 0
        done = False
        info["is_success"] = success
        # print("state", self.state)
        if self.visited_count[self.state] == 0:
            self.visited_count[self.state] = 1
        return self.get_obs(), reward, done, info

    def reset(self):
        self.state = 0
        if self.visited_count[self.state] == 0:
            self.visited_count[self.state] = 1.
        return self.get_obs()

    def get_obs(self):
        new = np.zeros(len(self.n2) - 2)
        # new[self.state] = 1
        new2 = bin(self.state)
        new2 = list(new2[2:])
        new2.reverse()

        for i, ele in enumerate(new2):
            new[-(i+1)] = int(ele)

        new = new[::-1]
        # new = new[self.shuffle_order]

        return {
            "observation": np.copy(new),
            "achieved_goal": np.copy(new),
            "desired_goal": np.copy(self.target),
        }

    @property
    def coverage(self):
        return np.sum(self.visited_count) / self.n