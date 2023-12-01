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
from numpy import linalg as LA

"""
Grid World Environment

"""
# from collections import defaultdict
# from lib.envs.gridworld import GridworldEnv
# from lib import plotting
# env = GridworldEnv()

"""
WIndy Grid World Environment
"""
from collections import defaultdict
from lib.envs.windy_gridworld import WindyGridworldEnv
from lib import plotting
env = WindyGridworldEnv()

"""
Cliff Walking Environment
"""
# from collections import defaultdict
# from lib.envs.cliff_walking import CliffWalkingEnv
# from lib import plotting
# env = CliffWalkingEnv()


"""
Gym MountainCar Environment
"""
# #with the mountaincar from openAi gym
# env = gym.envs.make("MountainCar-v0")



#samples from the state space to compute the features
observation_examples = np.array([env.observation_space.sample() for x in range(1)])
action_examples = np.array([env.action_space.sample() for a in range(1)])


scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)

scaler_action = sklearn.preprocessing.StandardScaler()
scaler_action.fit(action_examples)


#convert states to a feature representation:
#used an RBF sampler here for the feature map
featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
featurizer.fit(scaler.transform(observation_examples))


featurizer_action = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
featurizer_action.fit(scaler.transform(action_examples))



def featurize_state(state):
	state = np.array([state])
	scaled = scaler.transform([state])
	featurized = featurizer.transform(scaled)
	return featurized[0]


def featurize_action(action):
	action = np.array([action])
	scaled = scaler_action.transform([action])
	featurized_action = featurizer_action.transform(scaled)
	return featurized_action[0]


"""
Agent policies
"""

"""
Epsilon Greedy Policy
"""
def make_epsilon_greedy_policy(w_param, epsilon, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        phi = featurize_state(observation)
        q_values = np.dot(w_param.T, phi)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn



"""
Off Policy - Epsilon Greedy Policy
"""
def behaviour_policy_epsilon_greedy(w_param, epsilon, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        phi = featurize_state(observation)
        q_values = np.dot(w_param.T, phi)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


"""
Off Policy - Boltzmann Policy
"""
def behaviour_policy_Boltzmann(w_param, tau, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * tau / nA
        phi = featurize_state(observation)
        q_values = np.dot(w_param.T, phi)
        exp_tau = q_values / tau
        policy = np.exp(exp_tau) / np.sum(np.exp(exp_tau), axis=0)
        A = policy

        return A
    return policy_fn


"""
Greedy Policy
"""
def create_greedy_policy(w_param, epsilon, nA):
    def policy_fn(observation):
        A = np.zeros(nA, dtype=float) * epsilon / nA
        phi = featurize_state(observation)
        q_values = np.dot(w_param.T, phi)
        best_action = np.argmax(q_values)
        A[best_action] = 1
        return A
    return policy_fn


"""
LP_EXPLORATION : Building the polymer chain of trajectory
"""

def LP_Exploration(w_param, length_polymer_chain, L_p, b_step_size, sigma, action, state, alpha, discount_factor, nA):

	action_samples = np.array([env.action_space.sample() for a_s in range(length_polymer_chain)])

	phi_action_t = featurize_action(action)

	current_state_feature = featurize_state(state)
	current_q_value = np.dot(w_param.T, current_state_feature)[action]

	chain_actions = action
	chain_states = state
	similarity_threshold = 0.5


	for a in range(action_samples.shape[0]):
		#draw theta from a Gaussian distribution
		theta_mean = np.arccos( np.exp(   np.true_divide(-b_step_size, L_p) )  )
		theta = np.random.normal(theta_mean, sigma, 1)

		phi_action_t_1 = featurize_action(action_samples[a])
		action_similarity = np.true_divide(  (np.dot(phi_action_t, phi_action_t_1)),   np.multiply(  LA.norm(phi_action_t), LA.norm(phi_action_t_1) ) )

		similariy_metric = action_similarity - np.absolute(theta)

		if similariy_metric <= similarity_threshold:

			chosen_action = action_samples[a]
			chain_actions = np.append(chain_actions, chosen_action)

			chosen_state, reward, _, _ = env.step(chosen_action)
			chain_states = np.append(chain_states, chosen_state)

			chosen_state_feature = featurize_state(chosen_state)
			chosen_next_q_value = np.dot(w_param.T, chosen_state_feature)[chosen_action]

			w_param[:, chosen_action] += alpha * ( reward +  discount_factor * chosen_next_q_value - current_q_value  ) * current_state_feature


	action_trajectory_chain = chain_actions
	state_trajectory_chain = chain_states

	updated_w_param = w_param
	updated_Q_Value = np.dot(w_param.T, current_state_feature)

	return action_trajectory_chain, state_trajectory_chain, updated_Q_Value, updated_w_param



"""
Baselines Algorithms
"""
def poly_rl_q_learning(env, w_param, num_episodes, discount_factor=1.0, alpha = 0.1, epsilon=0.1, epsilon_decay=0.999, sigma = 0.5, L_p = 200, b_step_size = 1, length_polymer_chain = 500):

	stats = plotting.EpisodeStats(
		episode_lengths=np.zeros(num_episodes),
		episode_rewards=np.zeros(num_episodes))  

	w_param = np.random.normal(size=(400,env.action_space.n))


	for i_episode in range(num_episodes):
		
		print "Episode Number, PolyRL Q Learning:", i_episode

		state = env.reset()

		policy = make_epsilon_greedy_policy(w_param, epsilon * epsilon_decay**i_episode, env.action_space.n)

		#take a sample of action from the action space
		action = env.action_space.sample()

		"""
		Exploration phase - compute the polymer chain

		*** Returns ****
		Trajectory of actions, states
		Updated Q Values
		
		"""
		
		action_trajectory_chain, state_trajectory_chain, updated_Q_Value, updated_w_param = LP_Exploration(w_param, length_polymer_chain, L_p, b_step_size, sigma, action, state, alpha, discount_factor, env.action_space.n)

		w_param = updated_w_param

		#for each step in the environment
		for t in itertools.count():


			# action_trajectory_chain, state_trajectory_chain, updated_Q_Value, updated_w_param = LP_Exploration(w_param, length_polymer_chain, L_p, b_step_size, sigma, action, state, alpha, discount_factor, env.action_space.n)
			# w_param = updated_w_param


			# #choose action based on epsilon greedy policy
			# action_probs = policy(state)
			# action = np.random.choice(np.arange(len(action_probs)), p = action_probs)			

			next_state, reward, done, _ = env.step(action)
			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t

			#Q values for current state, action
			features_state = featurize_state(state)
			q_values = np.dot(w_param.T, features_state)
			q_values_state_action = q_values[action]


			next_action_probs = policy(next_state)
			next_action = np.random.choice(np.arange(len(next_action_probs)), p = next_action_probs)

			#next state features and Q(s', a')
			next_features_state = featurize_state(next_state)
			q_values_next = np.dot(w_param.T, next_features_state)
			q_values_next_state_next_action = q_values_next[next_action]


			# OR : np.max(q_values_next)
			#this is for the target policy pio
			#which is greedy wrt Q(s,a)
			best_next_action = np.argmax(q_values_next)
			td_target = reward + discount_factor * q_values_next[best_next_action]

			td_error = td_target - q_values_state_action

			w_param[:, action] += alpha * td_error * features_state

			if done:
				break

			state = next_state
			action = next_action


	return stats






def main():

	print "PolyRL Q Learning"
	w_param = np.random.normal(size=(400,env.action_space.n))
	num_episodes = 200
	smoothing_window = 100
	stats_q_learning = poly_rl_q_learning(env, w_param, num_episodes, epsilon=0.1)
	rewards_smoothed_stats_q_learning = pd.Series(stats_q_learning.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()	
	cum_rwd = rewards_smoothed_stats_q_learning
	np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Persistence_Length_Exploration/Results/'  + 'Trial_PolyRL' + '.npy', cum_rwd)
	plotting.plot_episode_stats(stats_q_learning)
	env.close()




	
if __name__ == '__main__':
	main()




