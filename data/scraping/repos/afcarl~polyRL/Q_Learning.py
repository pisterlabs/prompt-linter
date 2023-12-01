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
# from collections import defaultdict
# from lib.envs.windy_gridworld import WindyGridworldEnv
# from lib import plotting
# env = WindyGridworldEnv()

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
#with the mountaincar from openAi gym
#env = gym.envs.make("MountainCar-v0")


"""
Gym Continuous Mountain Car Environment
"""

env = gym.envs.make("MountainCarContinuous-v0")



#samples from the state space to compute the features
# observation_examples = np.array([env.observation_space.sample() for x in range(1)])
# action_examples = np.array([env.action_space.sample() for a in range(1)])

observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
action_examples = np.array([env.action_space.sample() for x in range(10000)])


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
featurizer_action.fit(scaler_action.transform(action_examples))


def featurize_state(state):
	# state = np.array([state])
	scaled = scaler.transform([state])
	featurized = featurizer.transform(scaled)
	return featurized[0]


def featurize_action(action):
	# action = np.array([action])
	scaled = scaler_action.transform([action])
	featurized_action = featurizer_action.transform(scaled)
	return featurized_action[0]


def make_epsilon_greedy_policy(w_param, epsilon, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        phi = featurize_state(observation)
        q_values = np.dot(w_param.T, phi)

        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


def LP_Exploration(w_param, length_polymer_chain, L_p, b_step_size, sigma, action, state, alpha, discount_factor, epsilon_one_step):

	phi_action_t = featurize_action(action)

	current_state_feature = featurize_state(state)

	action = discretized_action(action)
	current_q_value = np.dot(w_param.T, current_state_feature)[action]

	chain_actions = action
	chain_states = state
	similarity_threshold = 0.5

	#draw theta from a Gaussian distribution
	theta_mean = np.arccos( np.exp(   np.true_divide(-b_step_size, L_p) )  )
	theta = np.random.normal(theta_mean, sigma, 1)

	one_step_exploratory_action = 0
	action_trajectory_chain= 0
	state_trajectory_chain = 0
	updated_Q_Value = 0
	updated_w_param = 0

	end_traj_action = 0
	end_traj_state = 0


	if epsilon_one_step==0:

		print "Building the exploratory polymer chain"

		while True:

			action_sample = np.random.uniform(low = env.min_action, high = env.max_action, size=(1,))

			phi_action_t_1 = featurize_action(action_sample)
			action_similarity = np.arccos( np.true_divide(  (np.dot(phi_action_t, phi_action_t_1)),   np.multiply(  LA.norm(phi_action_t), LA.norm(phi_action_t_1) ) ) )

			similariy_metric = np.absolute(action_similarity - theta)

			if similariy_metric <= similarity_threshold:
				chosen_action = action_sample
				chain_actions = np.append(chain_actions, chosen_action)
				chosen_state, reward, _, _ = env.step(chosen_action)
				chain_states = np.append(chain_states, chosen_state)	

				#Return discretized action
				chosen_action = discretized_action(chosen_action)
				chosen_state_feature = featurize_state(chosen_state)

				chosen_next_q_value = np.dot(w_param.T, chosen_state_feature)[chosen_action]

				w_param[:, chosen_action] += alpha * ( reward +  discount_factor * chosen_next_q_value - current_q_value  ) * current_state_feature

				if len(chain_actions) == length_polymer_chain:
					end_traj_action = chosen_action
					end_traj_state = chosen_state
					print "LP Exploration done"
					break

		action_trajectory_chain = chain_actions
		state_trajectory_chain = chain_states

		updated_w_param = w_param
		updated_Q_Value = np.dot(w_param.T, current_state_feature)

		"""
		Return the last state and action of the chain
		"""

	elif epsilon_one_step==1:

		while True:

			action_sample = np.random.uniform(low = env.min_action, high = env.max_action, size=(1,))
			phi_action_t_1 = featurize_action(action_sample)
			action_similarity = np.arccos( np.true_divide(  (np.dot(phi_action_t, phi_action_t_1)),   np.multiply(  LA.norm(phi_action_t), LA.norm(phi_action_t_1) ) ) )
			similariy_metric = np.absolute(action_similarity - theta)	

			if similariy_metric <= similarity_threshold:
				chosen_action = action_sample
				chosen_action = discretized_action(action)
				break


		one_step_exploratory_action = chosen_action


	return action_trajectory_chain, state_trajectory_chain, updated_Q_Value, updated_w_param, one_step_exploratory_action, end_traj_action, end_traj_state







def discretized_action(action):

	if action > 0 and action <= 0.1111:
		discretized_selected_action = 0

	elif action > 0.1111 and action <= 0.3333:
		discretized_selected_action = 1

	elif action > 0.3333 and action <= 0.5555:
		discretized_selected_action = 2

	elif action >0.5555 and action <=-0.7777:
		discretized_selected_action = 3

	elif action >0.7777 and action <= 1.0:
		discretized_selected_action = 4

	elif action > - 0.1111 and action <= 0:
		discretized_selected_action = 5

	elif action > - 0.3333 and action <= - 0.1111:
		discretized_selected_action = 6

	elif action > - 0.5555 and action <= - 0.3333:
		discretized_selected_action = 7

	elif action > - 0.7777 and action <= - 0.5555:
		discretized_selected_action = 8

	else:
		discretized_selected_action = 9

	return discretized_selected_action


def recover_cont_action(action):

	if action ==0:
		cont_action = np.random.uniform(0, 0.1111, 1)
	elif action ==1:
		cont_action = np.random.uniform(0.1111, 0.3333, 1)
	elif action ==2:
		cont_action = np.random.uniform(0.3333, 0.5555, 1)
	elif action ==3:
		cont_action = np.random.uniform(0.5555, 0.7777, 1)
	elif action ==4:
		cont_action = np.random.uniform(0.7777, 1.0, 1)

	elif action ==5:
		cont_action = np.random.uniform(-0.1111, 0, 1)
	elif action ==6:
		cont_action = np.random.uniform(-0.3333, -0.1111, 1)
	elif action ==7:
		cont_action = np.random.uniform(-0.5555, -0.3333, 1)
	elif action ==8:
		cont_action = np.random.uniform(-0.7777, -0.5555, 1)
	else:
		cont_action = np.random.uniform(-1.0, -0.7777, 1)

	return cont_action




"""
Algorithm
"""
def Q_Learning(env, w_param, num_episodes, discount_factor=1.0, alpha = 0.001, epsilon=0.1, epsilon_decay=1.0, sigma = 0.05, L_p = 200, b_step_size = 1, length_polymer_chain = 500):

	stats = plotting.EpisodeStats(
		episode_lengths=np.zeros(num_episodes),
		episode_rewards=np.zeros(num_episodes))  


	for i_episode in range(num_episodes):
		
		print "Episode Number, Q Learning:", i_episode

		state = env.reset()
		initial_action = env.action_space.sample()

		action_space = np.linspace(env.min_action, env.max_action, num=10)
		ep_decay = epsilon * epsilon_decay ** i_episode
		policy = make_epsilon_greedy_policy(w_param, ep_decay, w_param.shape[1])

		action_probs = policy(state)
		action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

		action = recover_cont_action(action)


		print "Steps in an episode"
		#for each step in the environment
		for t in itertools.count():

			next_state, reward, done, _ = env.step(action)
			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t

			action = discretized_action(action)

			features_state = featurize_state(state)
			q_values = np.dot(w_param.T, features_state)
			q_values_state_action = q_values[action]


			next_action_probs = policy(next_state)
			next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)


			#next state features and Q(s', a')
			next_features_state = featurize_state(next_state)
			q_values_next = np.dot(w_param.T, next_features_state)
			q_values_next_state_next_action = q_values_next[next_action]


			best_next_action = np.argmax(q_values_next)
			td_target = reward + discount_factor * q_values_next[best_next_action]
			td_error = td_target - q_values_state_action
			w_param[:, action] += alpha * td_error * features_state


			state = next_state
			next_action = recover_cont_action(next_action)
			action = next_action


			if done:
				print "Total Steps for Episode", t
				break


	return stats






def main():

	print "Q Learning"

	#discretizing the action space
	action_space = np.linspace(env.min_action, env.max_action, num=10)
	w_param = np.random.normal(size=(400, action_space.shape[0]))

	num_episodes = 200
	smoothing_window = 100
	stats_poly_q_learning = Q_Learning(env, w_param, num_episodes, epsilon=0.1)
	rewards_smoothed_stats_poly_q_learning = pd.Series(stats_poly_q_learning.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()	
	cum_rwd = rewards_smoothed_stats_poly_q_learning
	np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Persistence_Length_Exploration/Results/'  + 'Trial_PolyRL' + '.npy', cum_rwd)
	plotting.plot_episode_stats(stats_poly_q_learning)
	env.close()




	
if __name__ == '__main__':
	main()




