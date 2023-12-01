# -*- coding: utf-8 -*-
# !/usr/bin/env python
# Adam Hornsby

"""
Perform a simulation in which the Coherency Maximizing agent chooses between
two choice types (1 and 2). These two choose types should be distinct on one attribute
but the same on another.
"""

SEED = 30
import random
import numpy as np
from sklearn.datasets.samples_generator import make_blobs

# own libraries
from model import CoherencyMaximisingAgent
from plot import plot_simulation_history

np.random.seed(SEED)
random.seed(SEED)

# main configuration for the simulation
CONFIG = {

    # initialise model
    'preference': [0.5, 0.5],  # initialised values of the preferences
    'weights': [0.5, 0.5],  # initialised values of the attention weights
    'lr': 0.01,  # the learning rate
    'c': 1,  # lambda value (i.e., "fussiness" parameter)
    'epsilon_greedy': True,  # use epsilon greedy selection or softmax selection?

    # simulation
    'cluster_centers': [[0.2, 0.2], [0.2, 0.8]],
    'n_choices': 500,
    'cluster_std': 0.05,  # std of the clusters
    'n_timesteps': 10000,

    # outputs
    'save_path': './figures/',
}


def simulate_blobs(centers, n_samples=500, cluster_std=1.5, random_state=32):
    """
    Simulate 2 choice types as clusters within a 2-dimensional space

    # Parameters
    n_samples (int): Number of options to sample
    cluster_std (float): Standard deviation of clusters
    random_state (int): Random state by which to sample options

    """

    X, y = make_blobs(n_samples=n_samples,
                      centers=centers,
                      cluster_std=cluster_std,
                      random_state=random_state)

    # clip values to be within the space (between 0 and 1)
    X = np.clip(X, 0, 1)

    return X, y


def update_agent(mod, action, observation):
    """
    Determine gradient of choice

    # Parameters
    mod (CoherencyMaximisingAgent): Model agent object
    action (int): Either 1 or 0 depending on choice type made
    observation (numpy.ndarray): Two dimensional matrix describing the attributes of the two choices
    """

    mod.update_agent(observation, action)

    # extract the current preference and attention weights, for plotting
    pref = mod.h0.preference_
    attention = mod.h0.attention_weights_

    return pref, attention


def create_random_observation(choice_ones, choice_twos, n):
    """
    Randomly select n of choice_ones and n of choice_twos and then row concatenate
    """

    choice_one = choice_ones[np.random.choice(choice_ones.shape[0], n), :]
    choice_two = choice_twos[np.random.choice(choice_twos.shape[0], n), :]

    return np.vstack([choice_one, choice_two])


def simulate_choices(X, y, model, n_choices, epsilon=0.05, epsilon_greedy=True):
    """
    Simulate the model for n_choices, taking an softmax exploration strategy

    # Parameters
    X (numpy.ndarray): Numpy multidimensional containing all possible observations
    y (numpy.ndarray): Vector describing the choice type (1 or 2) of observations
    model (CoherencyMaximisingAgent): Model agent
    n_choices (int): Number of choices by which to simulate
    epsilon (float): Probability of taking an exploratory action, if epsilon_greedy=True
    epsilon_greedy (bool): Whether to use epsilon greedy exploration or softmax exploration
    """

    pref_hist = list()
    attention_hist = list()

    # now make n_choices according to a e-greedy strategy
    for _ in range(n_choices):

        # select a random two products from choice type 1 and 2
        observation = create_random_observation(X[y == 0], X[y == 1], 1)
        observation = observation.T  # transpose for model compatability

        # determine the probability of making choice 1 or 2
        probs = model.feed_forward(observation)

        if epsilon_greedy:
            # use epsilon greedy exploration
            rnd = np.random.rand()

            if rnd < epsilon:
                action = np.random.choice([0, 1])
            else:
                action = np.argmax(probs)

        else:
            # use softmax exploration
            action = np.random.choice([0, 1], p=probs)

        # update the agent given the choice
        pref, attention = update_agent(model, action, observation)

        # update preference history
        pref_hist.append(pref)
        attention_hist.append(attention)

    return np.vstack(pref_hist), np.vstack(attention_hist)


def main(config):
    """Main entrypoint for the simulation code"""

    # simulate three clusters
    X, y = simulate_blobs(config['cluster_centers'],
                          n_samples=config['n_choices'],
                          cluster_std=config['cluster_std'])

    # initialise the agent
    mod = CoherencyMaximisingAgent(2,
                                   2,
                                   learn_prefs=True,
                                   learn_weights=True,
                                   c=config['c'],
                                   p_eta=config['lr'],
                                   w_eta=config['lr'],
                                   p_init=config['preference'],
                                   w_init=config['weights'])

    # simulate choices for n_timesteps
    pref_hist, att_hist = simulate_choices(X,
                                           y,
                                           mod,
                                           n_choices=config['n_timesteps'],
                                           epsilon_greedy=config['epsilon_greedy'])

    # plot the simulation history in a 2d plot. save to file.
    plot_simulation_history(X,
                            y,
                            pref_hist,
                            att_hist,
                            save_path=config['save_path'] + '2d_axis_plot.eps')
