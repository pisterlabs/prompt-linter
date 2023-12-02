""" Solving FrozenLake8x8 from OpenAI using Value Iteration
https://github.com/realdiganta/solving_openai/blob/master/FrozenLake8x8/frozenLake8x8.py
    Author: Diganta Kalita  (digankate26@gmail.com) """

import gym
import numpy as np


def value_iteration(env, max_iterations=100000, lmbda=0.9):
    stateValue = [0 for i in range(env.env.nS)]
    newStateValue = stateValue.copy()
    for iter in range(max_iterations):
        for state in range(env.env.nS):
            action_values = []
            for action in range(env.env.nA):
                state_value = 0
                for i in range(len(env.env.P[state][action])):
                    prob, next_state, reward, done = env.env.P[state][action][i]
                    state_action_value = prob * (reward + lmbda * stateValue[next_state])
                    state_value += state_action_value
                action_values.append(state_value)  # the value of each action
                best_action = np.argmax(np.asarray(action_values))  # choose the action which gives the maximum value
                newStateValue[state] = action_values[best_action]  # update the value of the state
        if iter > 1000:
            if sum(stateValue) - sum(newStateValue) < 1e-04:  # if there is negligible difference break the loop
                break
                print(iter)
        else:
            stateValue = newStateValue.copy()
    return stateValue


def get_policy(env, stateValue, gamma=0.9):
    '''
    Get optimal policy for s, based on the action that maximises long term reward (R+gamma*V(s'))
    :param env: 
    :param stateValue: 
    :param gamma: 
    :return: 
    '''
    policy = [0 for i in range(env.env.nS)]
    for state in range(env.env.nS):
        action_values = []
        for action in range(env.env.nA):
            action_value = 0
            for i in range(len(env.env.P[state][action])):
                prob, next_state, r, _ = env.env.P[state][action][i]
                action_value += prob * (r + gamma * stateValue[next_state])
            action_values.append(action_value)
        best_action = np.argmax(np.asarray(action_values))
        policy[state] = best_action
    return policy


def get_score(env, policy, episodes=1000):
    misses = 0
    steps_list = []
    for episode in range(episodes):
        observation = env.reset()
        steps = 0
        while True:

            action = policy[observation]
            observation, reward, done, _ = env.step(action)
            steps += 1
            if done and reward == 1:
                # print('You have got the fucking Frisbee after {} steps'.format(steps))
                steps_list.append(steps)
                break
            elif done and reward == 0:
                # print("You fell in a hole!")
                misses += 1
                break
    print('----------------------------------------------')
    print('You took an average of {:.0f} steps to get the frisbee'.format(np.mean(steps_list)))
    print('And you fell in the hole {:.2f} % of the times'.format((misses / episodes) * 100))
    print('----------------------------------------------')

if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')

    stateValues = value_iteration(env, max_iterations=100000)
    policy = get_policy(env, stateValues)
    get_score(env, policy, episodes=1000)
