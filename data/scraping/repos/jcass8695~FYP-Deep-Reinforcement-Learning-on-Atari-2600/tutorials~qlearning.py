'''
Implements Q-Learning for the MountainCar-v0 environment from OpenAI gym
'''
import random
import gym
import numpy as np
from pprint import pprint

n_states = 40
max_training_episodes = 10000
max_training_frames = 10000
testing_episodes = 100
gamma = 0.95
epsilon = 0.02
initial_lr = 1.0
min_lr = 0.003


def obs_to_state(env, obs):
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_dx = (env_high - env_low) / n_states
    a = int((obs[0] - env_low[0]) / env_dx[0])
    b = int((obs[1] - env_low[1]) / env_dx[1])
    return a, b


def training(env):
    q_table = np.zeros((n_states, n_states, 3))
    for i in range(max_training_episodes):
        obs = env.reset()
        total_reward = 0
        eta = max(min_lr, initial_lr * (0.85 ** (i // 100)))
        for _ in range(max_training_frames):
            pos, vel = obs_to_state(env, obs)
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.choice(env.action_space.n)
            else:
                # Gives rewards for the 3 possible actions
                logits = q_table[pos, vel]
                logits_exp = np.exp(logits)

                # Some probability function
                probs = logits_exp / np.sum(logits_exp)
                action = np.random.choice(env.action_space.n, p=probs)

            obs, reward, done, _ = env.step(action)
            total_reward += reward

            new_pos, new_vel = obs_to_state(env, obs)
            q_table[pos][vel][action] += eta * (reward + gamma *
                                                np.max(q_table[new_pos][new_vel]) - q_table[pos][vel][action])

            if done:
                break

    # See https://stackoverflow.com/questions/28697993/numpy-what-is-the-logic-of-the-argmin-and-argmax-functions
    # for explanation of the behaviour of argmax with the axis parameter
    solution_policy = np.argmax(q_table, axis=2)
    return solution_policy


def testing(env, policy, render=False):
    obs = env.reset()
    total_reward = 0

    for i in range(testing_episodes):
        if render:
            env.render()

        pos, vel = obs_to_state(env, obs)
        action = policy[pos][vel]

        obs, reward, done, _ = env.step(action)
        total_reward += gamma ** i * reward
        if done:
            break

    return total_reward


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    q_table = training(env)
    game_scores = [testing(env, q_table) for _ in range(testing_episodes)]
    print('Soln Avg Score:', np.mean(game_scores))
    testing(env, q_table, True)
