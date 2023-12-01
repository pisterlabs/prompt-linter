import gym
import itertools
import matplotlib
import numpy as np
import sys
import sklearn.pipeline
import sklearn.preprocessing

import pandas as pd
import sys
import random

from lib import plotting
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler
from collections import namedtuple

import matplotlib.pyplot as plt


"""
WIndy Grid World Environment
"""
# from collections import defaultdict
# from lib.envs.windy_gridworld import WindyGridworldEnv
# from lib import plotting
# env = WindyGridworldEnv()



"""
Cliff Walking Environment
"""
from collections import defaultdict
from lib.envs.cliff_walking import CliffWalkingEnv
from lib import plotting
env = CliffWalkingEnv()


"""
Gym MountainCar Environment
"""
# #with the mountaincar from openAi gym
# env = gym.envs.make("MountainCar-v0")


#samples from the state space to compute the features
observation_examples = np.array([env.observation_space.sample() for x in range(1)])



scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)


#convert states to a feature representation:
#used an RBF sampler here for the feature map
featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
featurizer.fit(scaler.transform(observation_examples))



def featurize_state(state):
	state = np.array([state])
	scaled = scaler.transform([state])
	featurized = featurizer.transform(scaled)
	return featurized[0]



"""
Agent policies
"""

"""
Epsilon Greedy Policy
"""
def make_epsilon_greedy_policy(theta, epsilon, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        phi = featurize_state(observation)
        q_values = np.dot(theta.T, phi)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn



"""
Off Policy - Epsilon Greedy Policy
"""
def behaviour_policy_epsilon_greedy(theta, epsilon, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        phi = featurize_state(observation)
        q_values = np.dot(theta.T, phi)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


"""
Off Policy - Boltzmann Policy
"""
def behaviour_policy_Boltzmann(theta, tau, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * tau / nA
        phi = featurize_state(observation)
        q_values = np.dot(theta.T, phi)
        exp_tau = q_values / tau
        policy = np.exp(exp_tau) / np.sum(np.exp(exp_tau), axis=0)
        A = policy

        return A
    return policy_fn


"""
Greedy Policy
"""
def create_greedy_policy(theta, epsilon, nA):
    def policy_fn(observation):
        A = np.zeros(nA, dtype=float) * epsilon / nA
        phi = featurize_state(observation)
        q_values = np.dot(theta.T, phi)
        best_action = np.argmax(q_values)
        A[best_action] = 1
        return A
    return policy_fn



"""
Baselines Algorithms
"""
def q_learning(env, theta, num_episodes, discount_factor=1.0, alpha = 0.1, epsilon=0.1, epsilon_decay=0.999):

	#q-learning algorithm with linear function approximation here

	#estimator : Estimator of Q^w(s,a)	- function approximator
	stats = plotting.EpisodeStats(
		episode_lengths=np.zeros(num_episodes),
		episode_rewards=np.zeros(num_episodes))  

	theta = np.random.normal(size=(400,env.action_space.n))

	for i_episode in range(num_episodes):
		print "Episode Number, Q Learning:", i_episode

		#this policy here is the off policy epsilon greedy policy?
		#np.argmax(Q[next_state]) - is for the target policy pi which is 
		#greedy (since maximisation over Q) wrt Q(s,a)
		policy = make_epsilon_greedy_policy(theta, epsilon * epsilon_decay**i_episode, env.action_space.n)

		#should be tau here for the Temperature - if using Boltzmann exploration policy
		# off_policy = behaviour_policy_Boltzmann(theta,epsilon * epsilon_decay**i_episode, env.action_space.n )	
		state = env.reset()

		next_action = None

		#for each one step in the environment
		for t in itertools.count():
			if next_action is None:
				action_probs = policy(state)
				action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
			else:
				action = next_action

			next_state, reward, done, _ = env.step(action)
			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t

			#Q values for current state
			features_state = featurize_state(state)
			q_values = np.dot(theta.T, features_state)
			q_values_state_action = q_values[action]


			#next action
			#these actions should be based on off policy for Q-learning
			#taking actions according to the off policy epsilon greedy policy
			next_action_probs = policy(next_state)
			next_action = np.random.choice(np.arange(len(next_action_probs)), p = next_action_probs)

			#next state features and Q(s', a')
			next_features_state = featurize_state(next_state)
			q_values_next = np.dot(theta.T, next_features_state)
			q_values_next_state_next_action = q_values_next[next_action]


			# OR : np.max(q_values_next)
			#this is for the target policy pio
			#which is greedy wrt Q(s,a)
			best_next_action = np.argmax(q_values_next)
			td_target = reward + discount_factor * q_values_next[best_next_action]


			td_error = td_target - q_values_state_action

			theta[:, action] += alpha * td_error * features_state

			if done:
				break
			state = next_state
	return stats



def sarsa(env, estimator, num_episodes, discount_factor=1.0, alpha = 0.1, epsilon=0.1, epsilon_decay=1.0):
	#estimator : Estimator of Q^w(s,a)	- function approximator
	stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes))

	theta = np.random.normal(size=(400,env.action_space.n))

	for i_episode in range(num_episodes):
		print "Episode Number, SARSA:", i_episode
		#agent policy based on the greedy maximisation of Q

		policy = make_epsilon_greedy_policy(theta, epsilon * epsilon_decay**i_episode, env.action_space.n)
		
		state = env.reset()

		action_probs = policy(state)
		action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
		next_action = None

		#for each one step in the environment
		for t in itertools.count():


			next_state, reward, done, _ = env.step(action)
			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t


			#Q values for current state
			features_state = featurize_state(state)
			q_values = np.dot(theta.T, features_state)
			q_values_state_action = q_values[action]


			#next action
			#these actions should be based on off policy for Q-learning
			next_action_probs = policy(next_state)
			next_action = np.random.choice(np.arange(len(next_action_probs)), p = next_action_probs)

			#next state features and Q(s', a')
			next_features_state = featurize_state(next_state)
			q_values_next = np.dot(theta.T, next_features_state)
			q_values_next_state_next_action = q_values_next[next_action]

			td_target = reward + discount_factor * q_values_next_state_next_action

			td_error = td_target - q_values_state_action


			theta[:, action] += alpha * td_error * features_state


			if done:
				break

			state = next_state
			action = next_action

	return stats




def main():

	print "Q Learning"
	theta = np.random.normal(size=(400,env.action_space.n))
	num_episodes = 2000
	smoothing_window = 200
	stats_q_learning = q_learning(env, theta, num_episodes, epsilon=0.1)
	rewards_smoothed_stats_q_learning = pd.Series(stats_q_learning.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()	
	cum_rwd = rewards_smoothed_stats_q_learning
	np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Persistence_Length_Exploration/Results/'  + 'Trial_Q_Learning' + '.npy', cum_rwd)
	plotting.plot_episode_stats(stats_q_learning)
	env.close()


	# print "SARSA"
	# theta = np.random.normal(size=(400,env.action_space.n))
	# num_episodes = 2000
	# smoothing_window = 200
	# stats_sarsa = sarsa(env, theta, num_episodes, epsilon=0.1)
	# rewards_smoothed_stats_sarsa = pd.Series(stats_sarsa.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()	
	# cum_rwd = rewards_smoothed_stats_sarsa
	# np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Persistence_Length_Exploration/Results/'  + 'Trial_SARSA' + '.npy', cum_rwd)
	# plotting.plot_episode_stats(stats_sarsa)
	# env.close()





	
if __name__ == '__main__':
	main()




