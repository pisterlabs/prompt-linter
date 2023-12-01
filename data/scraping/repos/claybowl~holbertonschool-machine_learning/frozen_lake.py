#!/usr/bin/env python3
"""Module frozen_lake.py
Collection of functions used to build
frozen_lake environments.
"""
import numpy as np
import gymnasium as gym
import random
import time
from IPython.display import clear_output
from gym.envs.toy_text.frozen_lake import generate_random_map


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """Load a frozen lake environment from OpenAI Gym."""
    if desc is not None:
        env = gym.make('FrozenLake-v0', desc=desc, is_slippery=is_slippery)
    elif map_name is not None:
        env = gym.make('FrozenLake-v0', map_name=map_name, is_slippery=is_slippery)
    else:
        env = gym.make('FrozenLake-v0', is_slippery=is_slippery)
    return env

def q_init(env):
    """Initialize Q-table to zeros."""
    return np.zeros([env.observation_space.n, env.action_space.n])

def epsilon_greedy(Q, state, epsilon):
    """Epsilon-greedy policy."""
    if random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()  # Explore action space
    else:
        action = np.argmax(Q[state])  # Exploit learned values
    return action

def train(env, Q, episodes=5000, max_steps=100,
          alpha=0.1, gamma=0.99, epsilon=1,
          min_epsilon=0.1, epsilon_decay=0.05):
    """Train agent to learn Q-values."""
    total_rewards = []
    for episode in range(episodes):
        state = env.reset()
        rewards = 0
        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, done, info = env.step(action)
            if done and reward == 0:
                reward = -1
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma * np.max(Q[new_state]))
            state = new_state
            rewards += reward
            if done:
                break
        epsilon = min_epsilon + (epsilon - min_epsilon) * np.exp(-epsilon_decay * episode)
        total_rewards.append(rewards)
    return Q, total_rewards

def play(env, Q, max_steps=100):
    """Play agent with learned Q-values."""
    state = env.reset()
    total_rewards = 0
    for step in range(max_steps):
        action = np.argmax(Q[state])
        new_state, reward, done, info = env.step(action)
        env.render()
        state = new_state
        total_rewards += reward
        if done:
            break
    return total_rewards
