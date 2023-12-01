import numpy as np
import math
from numpy.lib.utils import info
import pandas as pd
import matplotlib.pyplot as plt
import time
## Import for Running the Learners
from hiive.mdptoolbox.mdp import PolicyIteration, ValueIteration, QLearning
from openai import OpenAI_MDPToolbox
## Import for the Environments
from gym.envs.toy_text.frozen_lake import generate_random_map, FrozenLakeEnv
import gym
import gym.spaces as spaces

## Initializing the frozen lake environment with gym
env = gym.make('FrozenLake-v1')

## Initializing Gamma Values to test
# gamma = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
# gamma = np.arange(0.05, 1., 0.05).tolist()
gamma = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]

## Testing and Experiments Below
def plot_results(result_dict, test_name, xlabel, learner):
    '''
    A helper function to help plot
    '''
    for k in result_dict.keys():
        curr = result_dict[k]
        plt.plot(curr[0], curr[1])
        plt.xlabel(xlabel)
        # if xlabel == 'epsilon':
        #     plt.xscale('log')
        plt.ylabel(k)
        plt.title(f'{test_name} vs {k}')
        plt.savefig(f'frozenlake/{learner}/{test_name} vs {k}')
        plt.close()


def test_policy(policy, learner, gamma = 0.99, n_episodes = 10000):
    '''
    Getting the average reward from each learner
    '''
    rewards_list = list()
    for i in range(n_episodes):
        obs = env.reset() ## index for policy
        # obs = 0
        total_reward = 0
        steps = 0
        while True:
            obs, reward, done, info = env.step(policy[obs])
            total_reward += (gamma ** steps) * reward
            steps += 1
            # print(f'Step = {steps} => Observation = {obs}, Reward = {reward}')
            if done:
                break
        rewards_list.append(total_reward)
    # if learner == 'q-learning':
    print(policy)
    print(f'For Learner: {learner} -> Optimal Policy Total Reward = {np.mean(rewards_list)}')


def run_epsiodes(policy, learner, n_episodes = 10000):
    '''
    Seeing how manby times the learner actually solves the problem
    '''
    ## Running episodes per policy
    wins = 0
    steps = list()
    for i in range(n_episodes):
        # print(f'Gamma = {g} and Episode = {i}')
        obs = env.reset()
        num_steps = 0
        run = True
        while run:
            obs, reward, done, info = env.step(policy[obs])
            num_steps += 1
            if done:
                run = False
                if reward == 1:
                    steps.append(num_steps)
                    wins += 1
        

    print(f'For Learner: {learner} -> Average Number of Steps = {np.mean(steps)} and Win % = {(wins / n_episodes) * 100} %')


def value_iteration(P, R):
    '''
    Running Value Iteration to find the optimal policy for the Frozen Lake Problem
    '''
    

    gamma_results = {
        'iterations': [gamma, []],
        'rewards': [gamma, []],
        'time': [gamma, []],
    }

    policies = dict()
    for g in gamma:
        # learner = ValueIteration(P, R, gamma = g)
        learner = ValueIteration(P, R, gamma = g, epsilon = 0.0001)
        learner.run()

        ## Getting Stats
        gamma_results['iterations'][1].append(learner.iter)
        gamma_results['rewards'][1].append(np.mean(learner.V))
        gamma_results['time'][1].append(learner.time)

        ## Getting optimal Policy for gamma
        key = str(g)
        value = list(learner.policy)
        policies[key] = value
    ## Plotting Charts for reward time and iteration
    # print(gamma_results)
    plot_results(gamma_results, f'VI Gamma', 'gamma', 'vi')
    ## Testing Policies for different Gamma
    

def policy_iteration(P, R):
    '''
    Running Policy Iteration to find the optimal policy for the Frozen Lake Problem
    '''
    # gamma = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]

    gamma_results = {
        'iterations': [gamma, []],
        'rewards': [gamma, []],
        'time': [gamma, []],
    }

    policies = dict()
    for g in gamma:
        learner = PolicyIteration(P, R, gamma = g)
        learner.run()

        ## Getting Stats
        gamma_results['iterations'][1].append(learner.iter)
        gamma_results['rewards'][1].append(np.mean(learner.V))
        gamma_results['time'][1].append(learner.time)

        ## Getting optimal Policy for gamma
        key = str(g)
        value = list(learner.policy)
        policies[key] = value
    ## Plotting Charts for reward time and iteration
    # print(gamma_results)
    plot_results(gamma_results, f'PI Gamma', 'gamma', 'pi')

    
def q_learning(P, R):
    '''
    Running Policy Iteration to find the optimal policy for the Frozen Lake Problem
    '''
    # gamma = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]

    gamma_results = {
        # 'iterations': [gamma, []],
        'rewards': [gamma, []],
        'time': [gamma, []],
    }

    policies = dict()
    for g in gamma:
        learner = QLearning(P, R, gamma = g, alpha = 0.99, n_iter = 70000)
        learner.run()

        ## Getting Stats
        # gamma_results['iterations'][1].append(learner.run_stats[-1]['Iteration'])
        gamma_results['rewards'][1].append(np.mean(learner.V))
        gamma_results['time'][1].append(learner.time)

        ## Getting optimal Policy for gamma
        key = str(g)
        value = list(learner.policy)
        policies[key] = value
    ## Plotting Charts for reward time and iteration
    # print(gamma_results)
    plot_results(gamma_results, f'Q Gamma', 'gamma', 'q')
    

    ## Testing Alpha Values
    # alpha = np.arange(0.01, 1.0, 0.01).tolist()
    alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    alpha_results = {
        'rewards': [alpha, []],
        'time': [alpha, []],
    }
    for a in alpha:
        learner = QLearning(P, R, gamma = 0.99, alpha = a, n_iter = 70000)
        learner.run()

        ## Getting Stats
        # gamma_results['iterations'][1].append(learner.iter)
        alpha_results['rewards'][1].append(np.mean(learner.V))
        alpha_results['time'][1].append(learner.time)
        
    # print(gamma_results)
    ## Using Helper FUnctions to Plot for each Gamma on One Plot
    plot_results(alpha_results, f'Q - Alpha', 'alpha', 'q')

    ## Testing Different Values of Epsilon
    # eps = np.arange(0.01, 1.0, 0.01).tolist()
    eps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    eps_results = {
        'rewards': [eps, []],
        'time': [eps, []],
    }
    for e in eps:
        learner = QLearning(P, R, gamma = 0.99, alpha = 0.99, epsilon = e, n_iter = 70000)
        t0 = time.time()
        learner.run()
        total_time = time.time() - t0

        ## Getting Stats
        # eps_results['iterations'][1].append(learner.iter)
        eps_results['rewards'][1].append(np.mean(learner.V))
        eps_results['time'][1].append(total_time)
        
    ## Using Helper FUnctions to Plot for each Gamma on One Plot
    plot_results(eps_results, f'Q - Epsilon', 'epsilon', 'q')

    ## Testing the number of iterations
    n_iters = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000]
    rewards = list()
    times = list()
    for n in n_iters:
        learner = QLearning(P, R, gamma = 0.99, alpha = 0.99, epsilon = 0.0001, n_iter = n)
        t0 = time.time()
        learner.run()
        total_time = time.time() - t0

        rewards.append(np.mean(learner.V))
        times.append(total_time)
    iter_results = {
        'rewards': [n_iters, rewards],
        'time': [n_iters, times]
    }
    plot_results(iter_results, f'Q - N-Iter', 'n_iter', 'q')

def test_optimal_policies():
    '''
    Testing all Learners Episode Experiment
    '''
    frozen_lake = OpenAI_MDPToolbox('FrozenLake-v1')
    P = frozen_lake.P
    R = frozen_lake.R
    ## Times Dictionary
    times = dict()

    ## Q-Learning
    learner = QLearning(P, R, alpha = 0.99, epsilon = 0.001, gamma = 0.99, n_iter = 70000)
    t0 = time.time()
    learner.run()
    total_time = time.time() - t0
    times['q-learner'] = total_time
    policy = learner.policy
    test_policy(policy, 'q-learning', gamma = 0.99)
    run_epsiodes(policy, 'q-learning')
    print(f'Total Time = {total_time}')

    ## Value Iteration
    learner = ValueIteration(P, R, gamma = 0.99)
    t0 = time.time()
    learner.run()
    total_time = time.time() - t0
    times['value_iteration'] = total_time
    policy = learner.policy
    test_policy(policy, 'value-iteration')
    run_epsiodes(policy, 'value-iteration')
    print(f'Total Time = {total_time}')
    
    ## Policy Iteration
    learner = PolicyIteration(P, R, gamma = 0.99)
    t0 = time.time()
    learner.run()
    total_time = time.time() - t0
    times['policy-iteration'] = total_time
    policy = learner.policy
    test_policy(policy, 'policy-iteration')
    run_epsiodes(policy, 'policy-iteration')
    print(f'Total Time = {total_time}')


if __name__ == '__main__':
    # frozen_lake = FrozenLakeEnv(generate_random_map(10))
    frozen_lake = OpenAI_MDPToolbox('FrozenLake-v1')
    P = frozen_lake.P
    R = frozen_lake.R
    # print(P)
    # env.render()

    print(f'------------------------RUNNING VALUE ITERATION------------------------')
    value_iteration(P, R)
    print(f'------------------------RUNNING POLICY ITERATION------------------------')
    policy_iteration(P, R)
    print(f'------------------------RUNNING QLEARNER------------------------')
    q_learning(P, R)


    print('==============================TESTING EPISODES==============================')
    test_optimal_policies()