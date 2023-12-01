"""
Using REINFORCE with batch updates (i.e. what OpenAI calls VPG).

I will use the CartPole-v0 environment from OpenAI Gym.

As an experiment in structure, I will make an Agent class that contains its weights, policy, and 

The external code used is numpy, pytorch, and tile coding code from Sutton and Barto (incompleteideas.net I think)
"""
import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Categorical

import matplotlib.pyplot as plt
import random
SEED = 420
random.seed(SEED) # setting a random seed for consistency of the tiling
np.random.seed(SEED)
import gym
import os


class REINFORCE_Minibatches():
    """
    REINFORCE with minibatch updates to reduce variance.

    Functions to be called are .action(obs) which will return an action and
    .train_batch() to run a BATCH's worth of training
    """
    # I'll manually feed in the parameters needed 
    def __init__(self, env, obs_low, obs_high, act_dim, alpha, gamma, hidden_sizes, batch_size):
        self.env = env
        self.gamma = gamma
        self.act_dim = act_dim 
        self.lr = alpha # for neural net, take the learning rate raw
        self.batch_size = batch_size # number of obs/act/ret tuples to collect per batch
        # make the neural net policy
        # after making the tiling, we can set the learning rate based on how many tilings
        # create the list of scale factors from the obs dimension limits
        self.obs_dim = len(obs_low)
        self.obs_low = obs_low
        self.obs_high = obs_high
        self.scale_factors = obs_high - obs_low # making it a tensor here so I don't have to keep doing it in _rescale_obs
        self.policy_net = self._make_softmax_network(self.obs_dim, hidden_sizes, act_dim)
        # for now just use an adam optimiser on the parameters with the learning rate
        self.optim = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)

    # initialise a neural network with input size obs_dim, a list of hidden_sizes, and a softmax output with act_dim logits
    # NOTE: assumes at least one hidden layer
    def _make_softmax_network(self, in_dim, hidden_sizes, out_dim):
        # making a list of layers to pass to nn.sequential later
        layers = []
        input_layer = nn.Linear(in_dim, hidden_sizes[0])
        layers.append(input_layer)
        layers.append(nn.ReLU())
        for i in range(len(hidden_sizes)-1):
            layer = nn.Linear(hidden_sizes[i], hidden_sizes[i+1])
            layers.append(layer)
            layers.append(nn.ReLU())
        # finally add the output layer
        layers.append(nn.Linear(hidden_sizes[-1], out_dim))
        layers.append(nn.Softmax(dim=1))
        net = nn.Sequential(*layers)
        # return the net to be added to the object
        return net

    # rescale an observation using the lows and highs given
    # and return it as a tensor
    # TODO: I should probably rework this to work with batches better
    def _rescale_obs(self, batch_obs):
        # simply divide by the scale factor tensor and it will broadcast automatically
        return (batch_obs / self.scale_factors).unsqueeze(0)

    # get an action from an observation
    # TODO: TEST: returning the entropy of the conditional action distribution given obs
    def action(self, obs):
        # rescale the observations for input to the nn
        rescaled_obs = self._rescale_obs(obs)
        # get the probablities out
        probs = self.policy_net(rescaled_obs)
        dist = Categorical(probs)
        act = dist.sample()
        return act, dist.entropy().item()

    # generate an episode using the current policy and return the observations, actions, and rewards
    def generate_episode(self, render=False):
        # set up logging lists
        ep_obs = []
        ep_acts = []
        ep_rews = []
        ep_entropies = [] # TODO TEST tracking entropy

        # reset the environment for the episode
        env = self.env 
        obs = env.reset()
        done = False

        # loop over steps of the episode
        while not done:
            if render:
                env.render()
            act, entropy = self.action(torch.tensor(obs, dtype=torch.float))
            obsp, reward, done, info = env.step(act.item())
            # track the step
            ep_obs.append(obs)
            ep_acts.append(act)
            ep_rews.append(reward)
            ep_entropies.append(entropy)

            # move along
            obs = obsp
           
        # return the episode (NOTE: we don't keep the final observation as that isn't used anyway)
        return ep_obs, ep_acts, ep_rews, np.mean(ep_entropies)

    # return return for each step based on the rewards of an episode
    # NOTE: I'm now saying the reward from taking action a_t in state s_t is r_t, NOT r_t+1
    def _rets_from_rews(self, ep_rews):
        T = len(ep_rews) 
        rets = torch.tensor(ep_rews, dtype=torch.float)
        for i in reversed(range(T)):
            if i < T-1:
                rets[i] += self.gamma*rets[i+1]
        # return for final timestep is just 0
        return rets

    # separate log probability function. this is probably wasteful as we already construct a distribution when we pick the action
    # NOTE: Now I want it to work with batches
    def _log_prob(self, batch_obs, batch_acts):
        # rescale the observations for input to the nn
        rescaled_obs = self._rescale_obs(batch_obs)
        # get the probablities batch out
        probs = self.policy_net(rescaled_obs)
        # make a batch of distributions
        dist = Categorical(probs)
        # get the log probablities of act batch from the distribution
        lp = dist.log_prob(batch_acts)
        return lp

    # NOTE: this is where this one differs, I do a whole episode at a time
    # for now, does a whole episode
    def _update_minibatch(self, batch_obs, batch_acts, batch_weights):
        # zero out the gradients before computation (doesn't seem we HAVE to do this BEFORE computing loss)
        self.optim.zero_grad()
        # because we are doing a batch of variable size, we take the mean to try to reduce the effect of batch size
        T = len(batch_acts) # batch length
        # batch_weights INCLUDES discount factors because its not an episode anymore
        loss = - torch.mean( batch_weights * self._log_prob(batch_obs, batch_acts) )
        # now we backpropagate and take an optimiser step
        loss.backward()
        self.optim.step()

    # functions to save and load a model
    def save_params(self, filename):
        s_dict = self.policy_net.state_dict()
        torch.save(s_dict, filename)
        print(f"Parameters saved to {filename}")

    def load_params(self, filename):
        s_dict = torch.load(filename)
        self.policy_net.load_state_dict(s_dict)
        print(f"Parameters loaded from {filename}")
        
    # now we train a BATCH at a time instead of an episode at a time
    def train_batch(self):
        N = self.batch_size # number of time steps to collect
        batch_obs = []
        batch_acts = []
        batch_weights = [] # we have to save WEIGHTs, not RETURNS, because we need to include the discount factor
        # TODO TEST logging mean entropy of the batch
        batch_entropies = []
        # log return of completed episodes
        episode_returns = []

        # we have to subsume the episode generation into this function so we can stop it at N steps

        # set up environment at the start of the batch
        env = self.env 
        obs = env.reset()
        done = False
        ep_obs = []
        ep_acts = []
        ep_rews = []

        # loop until we have enough experience
        while len(batch_obs) < N: 
            # if the episode is done, save it and reset the environment
            if done:   
                # log completed run return
                ep_rets = self._rets_from_rews(ep_rews)
                episode_returns.append(ep_rets[0])

                # update experiences
                batch_obs += ep_obs
                batch_acts += ep_acts
                discounts = self.gamma ** torch.arange(len(ep_obs)) # cool way to make pytorch tensor without using list comprehension first
                batch_weights += list(ep_rets * discounts)

                # reset environment
                obs = env.reset()
                done = False
                ep_obs = []
                ep_acts = []
                ep_rews = []
                # move to next loop
                continue

            act, entropy = self.action(torch.tensor(obs, dtype=torch.float))
            obsp, reward, done, info = env.step(act.item())
            # track the step
            ep_obs.append(obs)
            ep_acts.append(act)
            ep_rews.append(reward)
            batch_entropies.append(entropy)

            # move along
            obs = obsp
           
        # once we have enough experience, update on all of it
        self._update_minibatch(torch.tensor(batch_obs, dtype=torch.float), torch.tensor(batch_acts, dtype=torch.float), torch.tensor(batch_weights, dtype=torch.float))

        # return mean return of episodes in the batch and the entropy over the whole batch
        return np.mean(episode_returns), np.mean(batch_entropies)


# make the environment
env = gym.make('CartPole-v0')
# manually specify the highs and lows of each variable
obs_high = torch.tensor([2.4, 5, 0.418, 5], dtype=torch.float)
obs_low = -obs_high

# base learning rate to be divided by the number of tilings
alpha = 2**-6
n_batch = 100
# Moment of truth: train an agent for 50 batches of 5000 steps
fig, ax = plt.subplots(figsize=(20,10))
# for j in range(10):
import time
tic = time.perf_counter()
REINFORCE_agent = REINFORCE_Minibatches(env, obs_low, obs_high, act_dim=2, alpha=alpha, gamma=1., hidden_sizes=[32], batch_size=5000)
# returns = []
# entropies = []
# for i in range(n_batch):
#     ep_ret, entropy = REINFORCE_agent.train_batch()
#     returns.append(ep_ret)
#     entropies.append(entropy)
#     print(f"Batch {i} had average return {ep_ret:0.4f} and mean entropy {entropy:0.4f}")

# REINFORCE_agent.save_params('REINFORCE_policy_params.dat')
REINFORCE_agent.load_params('REINFORCE_policy_params.dat')
returns = []
entropies = []
for i in range(10):
    ep_obs, ep_acts, ep_rews, entropy = REINFORCE_agent.generate_episode(render=True)
    ep_ret = REINFORCE_agent._rets_from_rews(ep_rews)
    returns.append(ep_ret[0])
    entropies.append(entropy)
    print(f"Episode {i} had return {ep_ret[0]} and mean entropy {entropy}")

plt.subplot(121)
plt.plot(returns)
plt.title("Return per episode")
plt.subplot(122)
plt.plot(entropies)
plt.title("Mean entropy per episode")

toc = time.perf_counter()
print(f"{n_batch} batches took {toc - tic:0.4f} seconds")
plt.show()

