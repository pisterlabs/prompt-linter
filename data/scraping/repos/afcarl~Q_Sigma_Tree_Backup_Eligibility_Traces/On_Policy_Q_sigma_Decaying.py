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



def Q_Sigma_On_Policy(env, theta, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=0.99):

	#q-learning algorithm with linear function approximation here

	#estimator : Estimator of Q^w(s,a)	- function approximator
	stats = plotting.EpisodeStats(
		episode_lengths=np.zeros(num_episodes),
		episode_rewards=np.zeros(num_episodes)) 
	cumulative_errors = np.zeros(shape=(num_episodes, 1)) 

	alpha = 0.01
	tau=1

	sigma_t_1 = 1
	sigma_decay = 0.995

  
	for i_episode in range(num_episodes):
		state_count=np.zeros(shape=(env.observation_space.n,1))

		print ("Epsisode Number On Policy Q(sigma)", i_episode)

		#off_policy = behaviour_policy_Boltzmann(theta, tau, env.action_space.n)
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
				sigma_t_1 = sigma_t_1 * sigma_decay

				if sigma_t_1 < 0.0001:
					sigma_t_1 = 0.0001
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
			


			#select next action based on the behaviour policy at next state
			next_action_probs = policy(state_t_1)
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


			"""
			target for one step
			1 step TD Target --- G_t(1)
			"""
			td_target = q_values_state_action + Delta_t 

			td_error = td_target -  q_values_state_action 

			# estimator.update(state, action, new_td_target)
			theta[:, action] += alpha * td_error * features_state
			rms_error = np.sqrt(np.sum((td_error)**2))
			cumulative_errors[i_episode, :] += rms_error

			state = state_t_1

	return stats,cumulative_errors



def take_average_results(experiment,num_experiments,num_episodes,env,theta):
	reward_mat=np.zeros([num_episodes,num_experiments])
	error_mat=np.zeros([num_episodes,num_experiments])
	for i in range(num_experiments):
		stats,cum_error=experiment(env,theta,num_episodes)
		reward_mat[:,i]=stats.episode_rewards
		error_mat[:,i]=cum_error.T
		average_reward=np.mean(reward_mat,axis=1)
		average_error=np.mean(error_mat,axis=1)
		np.save('/home/raihan/Desktop/Final_Project_Codes/Windy_GridWorld/Experimental_Results /decaying_based/'  + 'Qsigma_onpolicy_reward' + '.npy',average_reward)
		np.save('/home/raihan/Desktop/Final_Project_Codes/Windy_GridWorld/Experimental_Results /decaying_based/'  + 'Qsigma_onpolicy_error' + '.npy',average_error)
		
	return(average_reward,average_error)



def main():
	theta = np.random.normal(size=(400,env.action_space.n))
	num_episodes = 1000
	num_experiments=20
	print ("Running for Total Episodes", num_episodes)
	smoothing_window = 1

	avg_cum_reward,avg_cum_error=take_average_results(Q_Sigma_On_Policy,num_experiments,num_episodes,env,theta)
	
	env.close()


if __name__ == '__main__':
	main()







