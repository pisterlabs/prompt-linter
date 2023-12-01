"""
TO DO:

- Try with different off-policy behaviour policies

- Try with Sigma as a function of state
"""

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



#with the mountaincar from openAi gym
env = gym.envs.make("MountainCar-v0")


observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
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
	scaled = scaler.transform([state])
	featurized = featurizer.transform(scaled)
	return featurized[0]




def make_epsilon_greedy_policy(theta, epsilon, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        phi = featurize_state(observation)
        q_values = np.dot(theta.T, phi)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn



def behaviour_policy_epsilon_greedy(theta, epsilon, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        phi = featurize_state(observation)
        q_values = np.dot(theta.T, phi)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


def create_greedy_policy(theta, epsilon, nA):
    def policy_fn(observation):
        A = np.zeros(nA, dtype=float) * epsilon / nA
        phi = featurize_state(observation)
        q_values = np.dot(theta.T, phi)
        best_action = np.argmax(q_values)
        A[best_action] = 1
        return A
    return policy_fn



from numpy.random import binomial
def binomial_sigma(p):
	sample = binomial(n=1, p=p)
	return sample




def Q_Sigma_Off_Policy_2_Step(env, theta, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0):

	#q-learning algorithm with linear function approximation here

	#estimator : Estimator of Q^w(s,a)	- function approximator
	stats = plotting.EpisodeStats(
		episode_lengths=np.zeros(num_episodes),
		episode_rewards=np.zeros(num_episodes))  

	alpha = 0.01


	for i_episode in range(num_episodes):

		print "Epsisode Number Off Policy Q(sigma) 2 Step", i_episode

		off_policy = behaviour_policy_epsilon_greedy(theta, epsilon * epsilon_decay**i_episode, env.action_space.n)
		policy = make_epsilon_greedy_policy(theta, epsilon * epsilon_decay**i_episode, env.action_space.n)

		state = env.reset()

		next_action = None


		for t in itertools.count():

			if next_action is None:
				action_probs = off_policy(state)
				action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
			else:
				action = next_action

			state_t_1, reward, done, _ = env.step(action)

			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t

			if done:
				break			


			# q_values = estimator.predict(state)
			# q_values_state_action = q_values[action]
			#evaluate Q(current state, current action)
			features_state = featurize_state(state)
			q_values = np.dot(theta.T, features_state)
			q_values_state_action = q_values[action]


			#select sigma value
			probability = 0.5
			sigma_t_1 = binomial_sigma(probability)

			#select next action based on the behaviour policy at next state
			next_action_probs = off_policy(state_t_1)
			action_t_1 = np.random.choice(np.arange(len(next_action_probs)), p = next_action_probs)


			# q_values_t_1 = estimator.predict(state_t_1)
			# q_values_next_state_next_action = q_values_t_1[action_t_1]
			features_state_1 = featurize_state(state_t_1)
			q_values_t_1 = np.dot(theta.T, features_state_1)
			q_values_next_state_next_action = q_values_t_1[action_t_1]


			on_policy_next_action_probs = policy(state_t_1)
			on_policy_a_t_1 = np.random.choice(np.arange(len(on_policy_next_action_probs)), p = on_policy_next_action_probs)
			V_t_1 = np.sum( on_policy_next_action_probs * q_values_t_1 )

			Delta_t = reward + discount_factor * ( sigma_t_1 * q_values_next_state_next_action + (1 - sigma_t_1) * V_t_1  ) - q_values_state_action



			state_t_2, next_reward, done, _ = env.step(action_t_1)
			if done:
				break

			next_next_action_probs = off_policy(state_t_2)
			action_t_2 = np.random.choice(np.arange(len(next_next_action_probs)), p = next_next_action_probs)


			# q_values_t_2 = estimator.predict(state_t_2)
			# q_values_next_next_state_next_next_action = q_values_t_2[action_t_2]
			features_state_2 = featurize_state(state_t_2)
			q_values_t_2 = np.dot(theta.T, features_state_2)
			q_values_next_next_state_next_next_action = q_values_t_2[action_t_2]




			on_policy_next_next_action_probs = policy(state_t_2)
			on_policy_a_t_2 = np.random.choice(np.arange(len(on_policy_next_next_action_probs)), p = on_policy_next_next_action_probs)
			V_t_2 = np.sum( on_policy_next_next_action_probs * q_values_t_2  )
			
			sigma_t_2 = binomial_sigma(probability)



			Delta_t_1 = next_reward + discount_factor * (  sigma_t_2 * q_values_next_next_state_next_next_action + (1 - sigma_t_2) * V_t_2   ) - q_values_next_state_next_action


			"""
			2 step TD Target --- G_t(2)
			"""

			on_policy_action_probability = on_policy_next_action_probs[on_policy_a_t_1]
			off_policy_action_probability = next_action_probs[action_t_1]

			td_target = q_values_state_action + Delta_t + discount_factor * ( (1 - sigma_t_1) *  on_policy_action_probability + sigma_t_1 ) * Delta_t_1



			"""
			Computing Importance Sampling Ratio
			"""
			rho = np.divide( on_policy_action_probability, off_policy_action_probability )
			rho_sigma = sigma_t_1 * rho + 1 - sigma_t_1


			td_error = td_target -  q_values_state_action 

			# estimator.update(state, action, new_td_target)
			theta[:, action] += alpha * rho_sigma * td_error * features_state

			if done:
				break

			state = state_t_1
			
	return stats






def main():
	theta = np.random.normal(size=(400,env.action_space.n))
	num_episodes = 1000

	print "Running for Total Episodes", num_episodes

	smoothing_window = 1

	stats_q_sigma_off_policy = Q_Sigma_Off_Policy_2_Step(env, theta, num_episodes, epsilon=0.1)
	rewards_smoothed_stats_q_sigma_off_policy = pd.Series(stats_q_sigma_off_policy.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()	
	cum_rwd = rewards_smoothed_stats_q_sigma_off_policy
	np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Linear_Approximator/raw_results/'  + 'Off_Policy_Q_Sigma_2_step' + '.npy', cum_rwd)
	plotting.plot_episode_stats(stats_q_sigma_off_policy)
	env.close()


if __name__ == '__main__':
	main()







