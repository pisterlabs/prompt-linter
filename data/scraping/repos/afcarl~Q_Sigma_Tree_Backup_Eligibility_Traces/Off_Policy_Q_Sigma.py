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


from collections import defaultdict
from lib.envs.cliff_walking import CliffWalkingEnv
from lib import plotting
env = CliffWalkingEnv()

from collections import defaultdict
from lib.envs.cliff_walking import CliffWalkingEnv
from lib.envs.windy_gridworld import WindyGridworldEnv
from lib import plotting


#env = CliffWalkingEnv()
env=WindyGridworldEnv()



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




def Q_Sigma_Off_Policy(env, theta, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0):

	#q-learning algorithm with linear function approximation here

	#estimator : Estimator of Q^w(s,a)	- function approximator
	stats = plotting.EpisodeStats(
		episode_lengths=np.zeros(num_episodes),
		episode_rewards=np.zeros(num_episodes))  

	alpha = 0.01


	for i_episode in range(num_episodes):

		print "Epsisode Number Off Policy Q(sigma)", i_episode

		off_policy = behaviour_policy_Boltzmann(theta, tau, env.action_space.n)
		policy = make_epsilon_greedy_policy(theta, epsilon * epsilon_decay**i_episode, env.action_space.n)

		state = env.reset()
		next_action = None


		for t in itertools.count():

			if next_action is None:
				action_probs = policy(state)
				action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
			else:
				action = next_action

			state_t_1, reward, done, _ = env.step(action)

			if done:
				break			

			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t



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


			V_t_1 = np.sum( next_action_probs * q_values_t_1 )

			Delta_t = reward + discount_factor * ( sigma_t_1 * q_values_next_state_next_action + (1 - sigma_t_1) * V_t_1  ) - q_values_state_action


			"""
			target for one step
			1 step TD Target --- G_t(1)
			"""
			td_target = q_values_state_action + Delta_t 

			td_error = td_target -  q_values_state_action 

			# estimator.update(state, action, new_td_target)
			theta[:, action] += alpha * td_error * features_state


			state = state_t_1

	return stats






def main():
	theta = np.random.normal(size=(400,env.action_space.n))
	num_episodes = 1000

	print "Running for Total Episodes", num_episodes
	smoothing_window = 1

	stats_q_sigma_off_policy = Q_Sigma_Off_Policy(env, theta, num_episodes, epsilon=0.1)
	rewards_smoothed_stats_q_sigma_off_policy = pd.Series(stats_q_sigma_off_policy.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()	
	cum_rwd = rewards_smoothed_stats_q_sigma_off_policy
	np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Linear_Approximator/raw_results/'  + 'Off_Policy_Q_Sigma' + '.npy', cum_rwd)
	plotting.plot_episode_stats(stats_q_sigma_off_policy)
	env.close()


if __name__ == '__main__':
	main()







