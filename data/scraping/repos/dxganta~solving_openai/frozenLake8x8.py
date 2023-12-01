""" Solving FrozenLake8x8 from OpenAI using Value Iteration

    Author: Diganta Kalita  (digankate26@gmail.com) """

import gym
import numpy as np 


def value_iteration(env, max_iterations=100000, lmbda=0.9):
  stateValue = [0 for i in range(env.nS)]
  newStateValue = stateValue.copy()
  for i in range(max_iterations):
    for state in range(env.nS):
      action_values = []      
      for action in range(env.nA):
        state_value = 0
        for i in range(len(env.P[state][action])):
          prob, next_state, reward, done = env.P[state][action][i]
          state_action_value = prob * (reward + lmbda*stateValue[next_state])
          state_value += state_action_value
        action_values.append(state_value)      #the value of each action
        best_action = np.argmax(np.asarray(action_values))   # choose the action which gives the maximum value
        newStateValue[state] = action_values[best_action]  #update the value of the state
    if i > 1000: 
      if sum(stateValue) - sum(newStateValue) < 1e-04:   # if there is negligible difference break the loop
        break
        print(i)
    else:
      stateValue = newStateValue.copy()
  return stateValue 

def get_policy(env,stateValue, lmbda=0.9):
  policy = [0 for i in range(env.nS)]
  for state in range(env.nS):
    action_values = []
    for action in range(env.nA):
      action_value = 0
      for i in range(len(env.P[state][action])):
        prob, next_state, r, _ = env.P[state][action][i]
        action_value += prob * (r + lmbda * stateValue[next_state])
      action_values.append(action_value)
    best_action = np.argmax(np.asarray(action_values))
    policy[state] = best_action
  return policy 


def get_score(env, policy, episodes=1000):
  misses = 0
  steps_list = []
  for episode in range(episodes):
    observation = env.reset()
    steps=0
    while True:
      
      action = policy[observation]
      observation, reward, done, _ = env.step(action)
      steps+=1
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
  print('And you fell in the hole {:.2f} % of the times'.format((misses/episodes) * 100))
  print('----------------------------------------------')



env = gym.make('FrozenLake8x8-v0')

stateValues = value_iteration(env, max_iterations=100000)
policy = get_policy(env, stateValues)
get_score(env, policy,episodes=1000)
