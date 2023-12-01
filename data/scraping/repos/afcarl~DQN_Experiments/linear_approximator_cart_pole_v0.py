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




# env = gym.envs.make("MountainCar-v0")

#with the cartpole from openAi gym
env = gym.envs.make("CartPole-v0")

# env = gym.envs.make("Acrobot-v1")


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



class Estimator():

	"""
	Class to define the value function - Linear Function Approximator in this case
	"""

	def __init__(self):
		self.models = []

		for _ in range(env.action_space.n):
			model = SGDRegressor(learning_rate = "constant")
			model.partial_fit([self.featurize_state(env.reset())], [0])
			self.models.append(model)


	def featurize_state(self, state):

		scaled = scaler.transform([state])
		featurized = featurizer.transform(scaled)

		return featurized[0]



	def predict(self, s, a=None):

		features = self.featurize_state(s)

		if not a:
			return np.array([m.predict([features])[0] for m in self.models])
		else:
			return self.models[a].predict([features])[0]



	def update(self, s, a, y):

		#updates the estimator parameters for given s,a towards target y

		features = self.featurize_state(s)
		self.models[a].partial_fit([features], [y])




def make_epsilon_greedy_policy(estimator, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.
    
    Args:
        estimator: An estimator that returns q values for a given state
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(observation)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn






def q_learning(env, estimator, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0):

	#q-learning algorithm with linear function approximation here

	#estimator : Estimator of Q^w(s,a)	- function approximator
	stats = plotting.EpisodeStats(
		episode_lengths=np.zeros(num_episodes),
		episode_rewards=np.zeros(num_episodes))  

	for i_episode in range(num_episodes):
		print "Episode Number, Linear Cart Pole:", i_episode

		#agent policy based on the greedy maximisation of Q
		policy = make_epsilon_greedy_policy(estimator, epsilon * epsilon_decay**i_episode, env.action_space.n)
		last_reward = stats.episode_rewards[i_episode - 1]
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

			#update Q-values for the next state
			q_values_next = estimator.predict(next_state)

			#Q-value TD Target
			td_target = reward + discount_factor * np.max(q_values_next)

			#update the Q values
			#not this anymore
			#Q[state][action] += alpha * td_delta

			estimator.update(state, action, td_target)

			if done:
				break

			state = next_state

	return stats


def save_cum_rwd(stats, smoothing_window=1, noshow=False):


    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    cum_rwd = rewards_smoothed

    np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/gym_examples/DQN_Experiments/'  + 'linear_approx_cart_pole_v0_cumulative_reward' + '.npy', cum_rwd)

    return cum_rwd




def main():
	estimator = Estimator()
	num_episodes = 5000
	stats = q_learning(env, estimator, num_episodes, epsilon=0.1)

	cum_rwd = save_cum_rwd(stats, smoothing_window = 1)

	plotting.plot_episode_stats(stats)
	env.close()

if __name__ == '__main__':
	main()


